"""Where2comm fusion with ARCE/C2MAB message transmission hook.

Final insertion point:
    sender feature -> ARCE/C2MAB transport -> AttentionFusion

This module intentionally keeps Where2comm's original confidence-aware
attention fusion unchanged. Fixed transport keeps the native Where2Comm mask.
ARCE-C2MAB can instead rank sender-local spatial units with its own importance
module before non-ego messages enter the fusion module.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional, Tuple

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.common.fuse_modules.self_attn import ScaledDotProductAttention


class Communication(nn.Module):
    """Where2comm communication mask generator with optional raw-mask return."""

    def __init__(self, args):
        super(Communication, self).__init__()
        self.threshold = args["threshold"]
        if "gaussian_smooth" in args:
            self.smooth = True
            kernel_size = args["gaussian_smooth"]["k_size"]
            c_sigma = args["gaussian_smooth"]["c_sigma"]
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False
        else:
            self.smooth = False

    def init_gaussian_filter(self, k_size=5, sigma=1.0):
        center = k_size // 2
        x, y = np.mgrid[0 - center: k_size - center, 0 - center: k_size - center]
        gaussian_kernel = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(
            self.gaussian_filter.weight.device
        ).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()

    def forward(self, batch_confidence_maps, B, return_raw: bool = False):
        _, _, H, W = batch_confidence_maps[0].shape
        communication_masks = []
        raw_communication_masks = []
        confidence_maps_out = []
        communication_rates = []

        for b in range(B):
            ori_communication_maps, _ = batch_confidence_maps[b].sigmoid().max(dim=1, keepdim=True)
            if self.smooth:
                communication_maps = self.gaussian_filter(ori_communication_maps)
            else:
                communication_maps = ori_communication_maps

            L = communication_maps.shape[0]
            if self.training:
                K = int(H * W * random.uniform(0, 1))
                flat_maps = communication_maps.reshape(L, H * W)
                _, indices = torch.topk(flat_maps, k=K, sorted=False)
                communication_mask = torch.zeros_like(flat_maps).to(communication_maps.device)
                ones_fill = torch.ones(L, K, dtype=flat_maps.dtype, device=flat_maps.device)
                communication_mask = torch.scatter(communication_mask, -1, indices, ones_fill).reshape(L, 1, H, W)
            elif self.threshold:
                ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
                zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
                communication_mask = torch.where(communication_maps > self.threshold, ones_mask, zeros_mask)
            else:
                communication_mask = torch.ones_like(communication_maps).to(communication_maps.device)

            # Keep raw mask before forcing ego to all-one. This is needed for
            # complementarity / overlap calculations.
            raw_mask = communication_mask.clone()
            confidence_maps_out.append(communication_maps.detach())
            raw_communication_masks.append(raw_mask)

            communication_rate = communication_mask.sum() / (L * H * W)
            communication_mask[0] = 1

            communication_masks.append(communication_mask)
            communication_rates.append(communication_rate)

        communication_rates = sum(communication_rates) / B
        communication_masks = torch.cat(communication_masks, dim=0)
        raw_communication_masks = torch.cat(raw_communication_masks, dim=0)
        confidence_maps = torch.cat(confidence_maps_out, dim=0)

        if return_raw:
            return communication_masks, communication_rates, raw_communication_masks, confidence_maps
        return communication_masks, communication_rates


class AttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x):
        cav_num, C, H, W = x.shape
        x = x.view(cav_num, C, -1).permute(2, 0, 1)
        x = self.att(x, x, x)
        x = x.permute(1, 2, 0).view(cav_num, C, H, W)[0]
        return x


class Where2commArce(nn.Module):
    def __init__(self, args, arce_comm=None):
        super(Where2commArce, self).__init__()
        self.discrete_ratio = args["voxel_size"][0]
        self.downsample_rate = args["downsample_rate"]
        self.fully = args["fully"]
        if self.fully:
            print("constructing a fully connected communication graph")
        else:
            print("constructing a partially connected communication graph")

        self.multi_scale = args["multi_scale"]
        if self.multi_scale:
            layer_nums = args["layer_nums"]
            num_filters = args["num_filters"]
            self.num_levels = len(layer_nums)
            self.fuse_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                self.fuse_modules.append(AttentionFusion(num_filters[idx]))
        else:
            self.fuse_modules = AttentionFusion(args["in_channels"])

        self.naive_communication = Communication(args["communication"])
        self.arce_comm = arce_comm
        self.last_comm_info: Dict[str, Any] = {}

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        return torch.tensor_split(x, cum_sum_len[:-1].cpu())

    def _maybe_arce_comm(
        self,
        x: torch.Tensor,
        candidate_masks: Optional[torch.Tensor],
        priority_maps: Optional[torch.Tensor],
        record_len: torch.Tensor,
        data_dict: Optional[Dict[str, Any]],
        frame_id: Optional[int],
        local_cav_confidences: Optional[torch.Tensor] = None,
        local_cav_confidence_maps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, Any]]]]:
        if self.arce_comm is None:
            return x, None

        # payload_native means ARCE transports the payload tensor as a generic
        # message and must not consume Where2Comm-specific masks. Other modes may
        # still use Where2Comm mask-native payloads for fixed baselines.
        transport_mode = str(getattr(self.arce_comm, "transport_mode", "")).strip().lower()
        priority_layout_enabled = bool(
            getattr(self.arce_comm, "priority_layout_enabled", False)
        )
        uses_arce_spatial_importance = bool(
            getattr(
                self.arce_comm,
                "uses_arce_spatial_importance",
                False,
            )
        )
        if uses_arce_spatial_importance:
            candidate_masks = None
            priority_maps = None
        elif transport_mode == "payload_native":
            candidate_masks = None
            priority_maps = None
        elif not priority_layout_enabled:
            candidate_masks = (
                priority_maps if priority_maps is not None else candidate_masks
            )
            priority_maps = None

        # Dispatch only kwargs supported by the active ARCE implementation.
        # ARCEC2MABComm supports local_cav_confidences; ARCEFixedComm does not.
        # Both current implementations support message_masks, but this remains
        # compatible with older checkpoints/code snapshots.
        fn = self.arce_comm.communicate_flattened_features
        params = set(inspect.signature(fn).parameters.keys())

        kwargs = {
            "data_dict": data_dict,
            "frame_id": frame_id,
            "ego_index": 0,
            "update_cache": True,
            "return_records": True,
        }
        if "message_masks" in params:
            # Backward-compatible parameter name: this tensor is strictly the
            # binary candidate mask, never the continuous priority map.
            kwargs["message_masks"] = candidate_masks
        if priority_layout_enabled and "priority_maps" in params:
            kwargs["priority_maps"] = priority_maps
        if "local_cav_confidences" in params:
            kwargs["local_cav_confidences"] = local_cav_confidences
        if "local_cav_confidence_maps" in params:
            kwargs["local_cav_confidence_maps"] = local_cav_confidence_maps

        out = fn(x, record_len, **kwargs)

        if isinstance(out, tuple):
            return out[0], out[1]
        return out, None


    @staticmethod
    def _attach_arce_feature_delta(arce_info: Optional[Dict[str, Any]], x_after: torch.Tensor, x_before: torch.Tensor) -> Dict[str, Any]:
        if arce_info is None:
            arce_info = {}
        elif not isinstance(arce_info, dict):
            arce_info = {"raw_arce_info": arce_info}

        with torch.no_grad():
            before = x_before.detach().float()
            after = x_after.detach().float()
            signed_delta = after - before
            delta = signed_delta.abs()
            arce_info["arce_feature_delta"] = {
                "max_abs": float(delta.max().detach().cpu()),
                "mean_abs": float(delta.mean().detach().cpu()),
                "rms": float(
                    torch.sqrt(torch.mean(signed_delta * signed_delta)).detach().cpu()
                ),
                "nz_ratio": float((delta > 1e-6).float().mean().detach().cpu()),
                "before_rms": float(
                    torch.sqrt(torch.mean(before * before)).detach().cpu()
                ),
                "after_rms": float(
                    torch.sqrt(torch.mean(after * after)).detach().cpu()
                ),
                "per_agent": [
                    {
                        "agent_index": int(agent_index),
                        "before_rms": float(
                            torch.sqrt(torch.mean(before[agent_index] ** 2)).detach().cpu()
                        ),
                        "after_rms": float(
                            torch.sqrt(torch.mean(after[agent_index] ** 2)).detach().cpu()
                        ),
                        "delta_rms": float(
                            torch.sqrt(
                                torch.mean(signed_delta[agent_index] ** 2)
                            ).detach().cpu()
                        ),
                        "before_nz_ratio": float(
                            (before[agent_index].abs() > 1e-12)
                            .float().mean().detach().cpu()
                        ),
                        "after_nz_ratio": float(
                            (after[agent_index].abs() > 1e-12)
                            .float().mean().detach().cpu()
                        ),
                    }
                    for agent_index in range(int(before.shape[0]))
                ],
            }
        return arce_info

    def forward(self, x, psm_single, record_len, pairwise_t_matrix, backbone=None, data_dict=None, frame_id=None, local_cav_confidences=None, local_cav_confidence_maps=None):
        _, C, H, W = x.shape
        B = pairwise_t_matrix.shape[0]
        arce_info: Dict[str, Any] = {"enabled": self.arce_comm is not None, "records": []}

        if self.multi_scale:
            ups = []
            for i in range(self.num_levels):
                x = backbone.blocks[i](x)
                if i == 0:
                    if self.fully:
                        communication_rates = torch.tensor(1).to(x.device)
                        communication_masks = torch.ones((x.shape[0], 1, x.shape[-2], x.shape[-1]), dtype=x.dtype, device=x.device)
                        raw_masks = communication_masks.clone()
                        message_scores = raw_masks
                    else:
                        batch_confidence_maps = self.regroup(psm_single, record_len)
                        communication_masks, communication_rates, raw_masks, _confidence_maps = self.naive_communication(
                            batch_confidence_maps, B, return_raw=True
                        )
                        # candidate region = Where2Comm binary raw mask; score = confidence inside valid region
                        message_scores = _confidence_maps * raw_masks
                        if x.shape[-1] != communication_masks.shape[-1] or x.shape[-2] != communication_masks.shape[-2]:
                            communication_masks = F.interpolate(communication_masks, size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False)
                            raw_masks = F.interpolate(
                                raw_masks.float(),
                                size=(x.shape[-2], x.shape[-1]),
                                mode="nearest",
                            ).to(raw_masks.dtype)
                            message_scores = F.interpolate(message_scores, size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False)
                    if not bool(
                        getattr(
                            self.arce_comm,
                            "uses_arce_spatial_importance",
                            False,
                        )
                    ):
                        x = x * communication_masks
                    x_before_arce = x.detach().clone()
                    x, arce_info = self._maybe_arce_comm(
                        x,
                        raw_masks,
                        message_scores,
                        record_len,
                        data_dict,
                        frame_id,
                        local_cav_confidences=local_cav_confidences,
                local_cav_confidence_maps=local_cav_confidence_maps,
                    )
                    arce_info = self._attach_arce_feature_delta(arce_info, x, x_before_arce)

                batch_node_features = self.regroup(x, record_len)
                x_fuse = []
                for b in range(B):
                    x_fuse.append(self.fuse_modules[i](batch_node_features[b]))
                x_fuse = torch.stack(x_fuse)
                if len(backbone.deblocks) > 0:
                    ups.append(backbone.deblocks[i](x_fuse))
                else:
                    ups.append(x_fuse)

            if len(ups) > 1:
                x_fuse = torch.cat(ups, dim=1)
            elif len(ups) == 1:
                x_fuse = ups[0]
            if len(backbone.deblocks) > self.num_levels:
                x_fuse = backbone.deblocks[-1](x_fuse)
        else:
            if self.fully:
                communication_rates = torch.tensor(1).to(x.device)
                communication_masks = torch.ones((x.shape[0], 1, x.shape[-2], x.shape[-1]), dtype=x.dtype, device=x.device)
                raw_masks = communication_masks.clone()
                message_scores = raw_masks
            else:
                batch_confidence_maps = self.regroup(psm_single, record_len)
                communication_masks, communication_rates, raw_masks, _confidence_maps = self.naive_communication(
                    batch_confidence_maps, B, return_raw=True
                )
                # candidate region = Where2Comm binary raw mask; score = confidence inside valid region
                message_scores = _confidence_maps * raw_masks
                if not bool(
                    getattr(
                        self.arce_comm,
                        "uses_arce_spatial_importance",
                        False,
                    )
                ):
                    x = x * communication_masks

            x_before_arce = x.detach().clone()
            x, arce_info = self._maybe_arce_comm(
                x,
                raw_masks,
                message_scores,
                record_len,
                data_dict,
                frame_id,
                local_cav_confidences=local_cav_confidences,
                local_cav_confidence_maps=local_cav_confidence_maps,
            )
            arce_info = self._attach_arce_feature_delta(arce_info, x, x_before_arce)
            batch_node_features = self.regroup(x, record_len)
            x_fuse = []
            for b in range(B):
                x_fuse.append(self.fuse_modules(batch_node_features[b]))
            x_fuse = torch.stack(x_fuse)

        self.last_comm_info = {
            "where2comm_rate": communication_rates,
            "where2comm_mask_applied_to_payload": not bool(
                getattr(
                    self.arce_comm,
                    "uses_arce_spatial_importance",
                    False,
                )
            ),
            "payload_priority_source": (
                "arce_sender_feature_rms"
                if bool(
                    getattr(
                        self.arce_comm,
                        "uses_arce_spatial_importance",
                        False,
                    )
                )
                else "where2comm_or_native"
            ),
            "arce": arce_info,
        }
        return x_fuse, communication_rates, arce_info
