"""PointPillar + Where2comm with ARCE/C2MAB communication hook.

This wrapper follows the final experimental design:
  PointPillar -> Where2comm confidence mask -> ARCE/C2MAB message transmission
  -> original Where2comm attention fusion -> detection head.

The perception backbone and Where2comm fusion are not replaced. ARCE is a
plug-in communication layer applied between mask generation and fusion.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from opencood.methods.arce.c2mab_local_confidence import local_cav_confidences_from_psm
from opencood.models.common.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.baselines.where2comm.where2comm_arce_fuse import Where2commArce
from opencood.models.common.sub_modules.downsample_conv import DownsampleConv
from opencood.models.common.sub_modules.naive_compress import NaiveCompressor
from opencood.models.common.sub_modules.pillar_vfe import PillarVFE
from opencood.models.common.sub_modules.point_pillar_scatter import PointPillarScatter

try:
    from opencood.methods.arce.arce_fixed_comm import ARCEFixedComm
    _ARCE_FIXED_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    ARCEFixedComm = None
    _ARCE_FIXED_IMPORT_ERROR = e

try:
    from opencood.methods.arce.arce_c2mab_comm import ARCEC2MABComm
    _ARCE_C2MAB_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    ARCEC2MABComm = None
    _ARCE_C2MAB_IMPORT_ERROR = e



def _sanitize_fixed_random_arce_backend_args(args):
    """
    Sanitize Fixed/Random baselines before constructing ARCEFixedComm.

    High-level final YAML may use:
      - arce.mode=random
      - arce.channel.mode=markov
      - dict-style recovery configs in action profiles

    ARCEFixedComm / FixedARCEPolicy expect:
      - arce.mode=fixed
      - channel.mode=fixed
      - action.recovery as a string

    This function only changes the low-level executor config and keeps
    baseline identity via policy=fixed/random.
    """
    out = copy.deepcopy(args or {})

    arce = copy.deepcopy(out.get("arce", {}) or {})
    if not isinstance(arce, dict):
        return out

    mode = str(arce.get("mode", arce.get("policy", "fixed"))).strip().lower()
    policy = str(arce.get("policy", mode)).strip().lower()

    # Only sanitize Fixed / Random baselines. C2MAB is handled by ARCEC2MABComm.
    if mode not in ("fixed", "random") and policy not in ("fixed", "random"):
        return out

    arce.setdefault("original_mode", mode)
    arce.setdefault("original_policy", policy)

    # ARCEFixedComm itself only accepts mode=fixed.
    # Random baseline is preserved by policy=random.
    arce["mode"] = "fixed"
    arce["policy"] = policy if policy in ("fixed", "random") else mode

    # ChannelManager backend only supports fixed mode.
    channel = copy.deepcopy(arce.get("channel", {}) or {})
    channel.setdefault("original_mode", channel.get("mode", None))
    channel["mode"] = "fixed"
    channel.setdefault("fixed_state", "medium")
    arce["channel"] = channel

    # FeatureSizeEstimator may read quantization.mode from the config.
    # Make it explicit so top-level arce.mode=fixed is never interpreted
    # as a quantization mode.
    quantization = copy.deepcopy(arce.get("quantization", {}) or {})
    quantization.setdefault("mode", "fp16")
    arce["quantization"] = quantization

    def _choose_recovery_method(recovery_cfg):
        """
        Convert dict-style recovery config to a FixedARCEPolicy-compatible string.
        """
        if not isinstance(recovery_cfg, dict):
            return recovery_cfg

        if bool(recovery_cfg.get("temporal_cache", False)):
            return "temporal_cache"
        if bool(recovery_cfg.get("spatial_interpolation", False)):
            return "spatial_interpolation"
        if bool(recovery_cfg.get("zero_fill", False)):
            return "zero_fill"
        return "none"

    def _sanitize_recovery_fields(obj):
        """
        Recursively sanitize all nested action/profile dicts.
        """
        if isinstance(obj, dict):
            rec = obj.get("recovery", None)
            if isinstance(rec, dict):
                obj.setdefault("recovery_config", copy.deepcopy(rec))
                obj["recovery"] = _choose_recovery_method(rec)

            for value in list(obj.values()):
                _sanitize_recovery_fields(value)

        elif isinstance(obj, list):
            for value in obj:
                _sanitize_recovery_fields(value)

        elif isinstance(obj, tuple):
            for value in obj:
                _sanitize_recovery_fields(value)

    # Recursively sanitize the whole ARCE subtree, not just fixed_policy/random_policy.
    _sanitize_recovery_fields(arce)

    out["arce"] = arce
    return out


class PointPillarWhere2commArceV2xreal(nn.Module):
    def __init__(self, args):
        super(PointPillarWhere2commArceV2xreal, self).__init__()
        self.max_cav = args["max_cav"]
        self.args = args

        self.pillar_vfe = PillarVFE(args["pillar_vfe"], num_point_features=4, voxel_size=args["voxel_size"], point_cloud_range=args["lidar_range"])
        self.scatter = PointPillarScatter(args["point_pillar_scatter"])
        self.backbone = BaseBEVBackbone(args["base_bev_backbone"], 64)

        if "shrink_header" in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args["shrink_header"])
        else:
            self.shrink_flag = False

        if args["compression"]:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args["compression"])
        else:
            self.compression = False

        self.arce_enabled = bool(args.get("arce", {}).get("enabled", False)) if isinstance(args.get("arce", {}), dict) else False
        self.arce_comm = self._build_arce_comm(args) if self.arce_enabled else None

        self.fusion_net = Where2commArce(args["where2comm_fusion"], arce_comm=self.arce_comm)
        self.multi_scale = args["where2comm_fusion"]["multi_scale"]

        self.num_class = int(args.get("num_class", 3))
        self.cls_head = nn.Conv2d(
            args["head_dim"],
            args["anchor_number"] * self.num_class * self.num_class,
            kernel_size=1
        )
        self.reg_head = nn.Conv2d(
            args["head_dim"],
            7 * args["anchor_number"] * self.num_class,
            kernel_size=1
        )

        if args["backbone_fix"]:
            self.backbone_fix()

    def _build_arce_comm(self, args):
        arce_cfg = args.get("arce", {}) if isinstance(args, dict) else {}
        arce_mode = str(arce_cfg.get("mode", arce_cfg.get("policy", "fixed"))).lower()
        arce_policy = str(arce_cfg.get("policy", arce_mode)).lower()
        if arce_mode in ("dc2mab", "c2mab") or arce_policy in ("dc2mab_sender_ego", "c2mab_sender_ego"):
            if ARCEC2MABComm is None:
                raise ImportError(f"Failed to import ARCEC2MABComm: {_ARCE_C2MAB_IMPORT_ERROR}")
            return ARCEC2MABComm(args)
        if ARCEFixedComm is None:
            raise ImportError(f"Failed to import ARCEFixedComm: {_ARCE_FIXED_IMPORT_ERROR}")
        return ARCEFixedComm(_sanitize_fixed_random_arce_backend_args(args))

    def backbone_fix(self):
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False
        for p in self.scatter.parameters():
            p.requires_grad = False
        for p in self.backbone.parameters():
            p.requires_grad = False
        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False
        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def _local_cav_confidences_from_psm(self, psm_single):
        """Compute local CAV confidence C_i for C2MAB context."""
        return local_cav_confidences_from_psm(psm_single, topk=50)

    def _infer_frame_id(self, data_dict):
        for key in ("frame_id", "timestamp", "sample_idx", "sample_id"):
            if isinstance(data_dict, dict) and key in data_dict:
                value = data_dict[key]
                if torch.is_tensor(value) and value.numel() == 1:
                    return int(value.detach().cpu().item())
                return value
        return None

    def forward(self, data_dict):
        voxel_features = data_dict["processed_lidar"]["voxel_features"]
        voxel_coords = data_dict["processed_lidar"]["voxel_coords"]
        voxel_num_points = data_dict["processed_lidar"]["voxel_num_points"]
        record_len = data_dict["record_len"]
        pairwise_t_matrix = data_dict["pairwise_t_matrix"]

        batch_dict = {
            "voxel_features": voxel_features,
            "voxel_coords": voxel_coords,
            "voxel_num_points": voxel_num_points,
            "record_len": record_len,
        }
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict["spatial_features_2d"]
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        psm_single = self.cls_head(spatial_features_2d)
        local_cav_confidences = self._local_cav_confidences_from_psm(psm_single)

        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        frame_id = self._infer_frame_id(data_dict)
        if self.multi_scale:
            fused_feature, communication_rates, arce_info = self.fusion_net(
                batch_dict["spatial_features"],
                psm_single,
                record_len,
                pairwise_t_matrix,
                self.backbone,
                data_dict=data_dict,
                frame_id=frame_id,
                local_cav_confidences=local_cav_confidences,
            )
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        else:
            fused_feature, communication_rates, arce_info = self.fusion_net(
                spatial_features_2d,
                psm_single,
                record_len,
                pairwise_t_matrix,
                data_dict=data_dict,
                frame_id=frame_id,
                local_cav_confidences=local_cav_confidences,
            )

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        arce_reward_update = None
        if (
            (not self.training)
            and self.arce_enabled
            and self.arce_comm is not None
            and hasattr(self.arce_comm, "update_with_proxy_reward")
        ):
            try:
                # Use the post-fusion classification confidence as the collaborative
                # perception quality signal for proxy reward update.
                # psm shape is usually [B, A, H, W]. We first map logits to
                # probabilities, then take the strongest anchor confidence per cell
                # and average over the BEV map.
                with torch.no_grad():
                    collab_confidence = float(
                        torch.sigmoid(psm)
                        .detach()
                        .max(dim=1)[0]
                        .mean()
                        .cpu()
                        .item()
                    )
                arce_reward_update = self.arce_comm.update_with_proxy_reward(
                    collab_confidence=collab_confidence
                )
            except Exception as exc:
                arce_reward_update = {
                    "num_updated": 0,
                    "error": "{}: {}".format(type(exc).__name__, exc),
                }

        # Keep reward update both as a top-level comm_info field and inside
        # arce_info so existing recursive debug scripts can find it.
        if arce_reward_update is not None:
            if isinstance(arce_info, dict):
                arce_info = dict(arce_info)
                arce_info["reward_update"] = arce_reward_update
            elif isinstance(arce_info, list):
                arce_info = list(arce_info)
                arce_info.append({"reward_update": arce_reward_update})

        output_dict = {"psm": psm, "rm": rm, "com": communication_rates}
        output_dict["comm_info"] = {
            "where2comm_rate": communication_rates,
            "arce": arce_info,
            "arce_enabled": bool(self.arce_enabled),
            "arce_reward_update": arce_reward_update,
        }
        return output_dict
