# -*- coding: utf-8 -*-
"""
PointPillar-RoCooper model entry for OpenCOOD.

Save as:
    opencood/models/baselines/rocooper/models/point_pillar_rocooper.py

This file defines the top-level detector:
    PointPillar backbone
    -> optional shrink/compression
    -> RoCooper communication impairment
    -> RoCooper fusion
    -> classification/regression heads

Required companion files:
    opencood/models/baselines/rocooper/components/rocooper_comm.py
    opencood/models/baselines/rocooper/rocooper_fuse.py

Recommended companion files used by rocooper_fuse.py:
    opencood/models/baselines/rocooper/components/rocooper_augmentor.py
    opencood/models/baselines/rocooper/components/rocooper_aggregator.py
    opencood/models/baselines/rocooper/components/rocooper_block_prioritizer.py
    opencood/models/baselines/rocooper/components/rocooper_utils.py
"""

from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from opencood.models.common.sub_modules.pillar_vfe import PillarVFE
from opencood.models.common.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.common.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.common.sub_modules.downsample_conv import DownsampleConv
from opencood.models.common.sub_modules.naive_compress import NaiveCompressor

from opencood.models.baselines.rocooper.components.rocooper_comm import RoCooperComm
from opencood.models.baselines.rocooper.rocooper_fuse import RoCooperFusion


class PointPillarRocooper(nn.Module):
    """
    OpenCOOD top-level model for RoCooper.

    Expected yaml entry:

        model:
          core_method: point_pillar_rocooper
          args:
            voxel_size: [0.4, 0.4, 4]
            lidar_range: [-140.8, -40, -3, 140.8, 40, 1]
            anchor_number: 2
            max_cav: 5
            compression: 0
            backbone_fix: false

            pillar_vfe: ...
            point_pillar_scatter: ...
            base_bev_backbone: ...
            shrink_header: ...

            rocooper: ...
            rocooper_comm: ...
            augmentor: ...
            aggregator: ...
            block_prioritizer: ...

    Forward input data_dict is the normal OpenCOOD intermediate-fusion batch:
        data_dict["processed_lidar"]["voxel_features"]
        data_dict["processed_lidar"]["voxel_coords"]
        data_dict["processed_lidar"]["voxel_num_points"]
        data_dict["record_len"]
        data_dict["pairwise_t_matrix"]

    Forward output:
        {
            "psm": classification prediction map,
            "rm": regression prediction map,
            "comm_info": optional communication statistics,
            "fusion_info": optional RoCooper internal statistics,
            "psm_single": single-agent score map before fusion
        }
    """

    def __init__(self, args: Dict[str, Any]):
        super(PointPillarRocooper, self).__init__()

        self.args = args
        self.max_cav = args.get("max_cav", 5)
        self.backbone_fix_flag = bool(args.get("backbone_fix", False))

        self.anchor_number = self._get_anchor_number(args)

        # ------------------------------------------------------------------
        # 1. PointPillar feature encoder
        # ------------------------------------------------------------------
        self.pillar_vfe = PillarVFE(
            args["pillar_vfe"],
            num_point_features=4,
            voxel_size=args["voxel_size"],
            point_cloud_range=args["lidar_range"],
        )

        self.scatter = PointPillarScatter(args["point_pillar_scatter"])

        # OpenCOOD PointPillar commonly uses 64 channels after PillarVFE.
        self.backbone = BaseBEVBackbone(args["base_bev_backbone"], 64)

        # ------------------------------------------------------------------
        # 2. Optional shrink header
        # ------------------------------------------------------------------
        # OpenCOOD backbone often outputs 384 channels after concatenating
        # multi-scale BEV features. shrink_header usually maps 384 -> 256.
        if "shrink_header" in args and args["shrink_header"] is not None:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args["shrink_header"])
            self.feature_dim = self._infer_shrink_output_dim(args["shrink_header"])
        else:
            self.shrink_flag = False
            self.shrink_conv = None
            self.feature_dim = self._infer_backbone_output_dim(args)

        # ------------------------------------------------------------------
        # 3. Optional naive compressor
        # ------------------------------------------------------------------
        # For RoCooper, feature compression is conceptually before transmission.
        # The actual lossy communication simulation is handled by RoCooperComm.
        compression_ratio = args.get("compression", 0)
        self.compression = compression_ratio is not None and compression_ratio > 0

        if self.compression:
            self.naive_compressor = NaiveCompressor(
                self.feature_dim,
                compression_ratio,
            )
        else:
            self.naive_compressor = None

        # ------------------------------------------------------------------
        # 4. RoCooper communication impairment module
        # ------------------------------------------------------------------
        # This module should keep ego features lossless when impair_ego=False.
        rocooper_comm_args = args.get("rocooper_comm", {})
        self.comm_module = RoCooperComm(rocooper_comm_args)

        # ------------------------------------------------------------------
        # 5. RoCooper fusion module
        # ------------------------------------------------------------------
        # Merge all RoCooper-related configs into one dict, so rocooper_fuse.py
        # can initialize Augmentor / Aggregator / BlockPrioritizer cleanly.
        rocooper_fusion_args = self._build_rocooper_fusion_args(args)
        self.fusion_net = RoCooperFusion(rocooper_fusion_args)

        # ------------------------------------------------------------------
        # 6. Detection heads
        # ------------------------------------------------------------------
        # psm: classification score map
        # rm : box regression map
        self.cls_head = nn.Conv2d(
            self.feature_dim,
            self.anchor_number,
            kernel_size=1,
        )

        self.reg_head = nn.Conv2d(
            self.feature_dim,
            7 * self.anchor_number,
            kernel_size=1,
        )

        if self.backbone_fix_flag:
            self.backbone_fix()

    @staticmethod
    def _get_anchor_number(args: Dict[str, Any]) -> int:
        """
        OpenCOOD configs usually use 'anchor_number', but some old examples
        accidentally used 'anchor_num'. This helper supports both.
        """
        if "anchor_number" in args:
            return int(args["anchor_number"])
        if "anchor_num" in args:
            return int(args["anchor_num"])
        raise KeyError("Model args must contain 'anchor_number' or 'anchor_num'.")

    @staticmethod
    def _infer_shrink_output_dim(shrink_header_args: Dict[str, Any]) -> int:
        """
        Infer output dim of DownsampleConv from yaml.

        Example:
            shrink_header:
              dim: [256]
              input_dim: 384
        """
        if "dim" in shrink_header_args:
            dim = shrink_header_args["dim"]
            if isinstance(dim, (list, tuple)):
                return int(dim[-1])
            return int(dim)

        if "output_dim" in shrink_header_args:
            return int(shrink_header_args["output_dim"])

        # Safe default used by most OpenCOOD PointPillar intermediate configs.
        return 256

    @staticmethod
    def _infer_backbone_output_dim(args: Dict[str, Any]) -> int:
        """
        Infer BEV backbone output dimension when shrink_header is not used.

        OpenCOOD PointPillar BaseBEVBackbone commonly concatenates
        num_upsample_filter, e.g. [128, 128, 128] -> 384.
        """
        backbone_args = args.get("base_bev_backbone", {})

        if "num_upsample_filter" in backbone_args:
            return int(sum(backbone_args["num_upsample_filter"]))

        if "num_filters" in backbone_args:
            # Fallback; not always equal to final output dim.
            filters = backbone_args["num_filters"]
            if isinstance(filters, (list, tuple)):
                return int(filters[-1])
            return int(filters)

        # Safe fallback for common OpenCOOD PointPillar configs.
        return 384

    @staticmethod
    def _build_rocooper_fusion_args(args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect RoCooper-related yaml blocks into a single config dict.

        rocooper_fuse.py can then access:
            cfg["rocooper"]
            cfg["augmentor"]
            cfg["aggregator"]
            cfg["block_prioritizer"]
        """
        cfg = {
            "rocooper": args.get("rocooper", {}),
            "augmentor": args.get("augmentor", {}),
            "aggregator": args.get("aggregator", {}),
            "block_prioritizer": args.get("block_prioritizer", {}),
            "max_cav": args.get("max_cav", 5),
            "feature_dim": None,
        }

        # Keep additional useful raw fields.
        for key in [
            "voxel_size",
            "lidar_range",
            "anchor_number",
            "anchor_num",
            "compression",
            "head_dim",
        ]:
            if key in args:
                cfg[key] = args[key]

        return cfg

    def backbone_fix(self) -> None:
        """
        Freeze backbone and detection heads.

        This follows the finetuning style used by several OpenCOOD models.
        Useful when you first train a clean baseline and later finetune only
        RoCooper communication/fusion modules under lossy communication.
        """
        modules_to_freeze = [
            self.pillar_vfe,
            self.scatter,
            self.backbone,
            self.cls_head,
            self.reg_head,
        ]

        if self.shrink_flag and self.shrink_conv is not None:
            modules_to_freeze.append(self.shrink_conv)

        if self.compression and self.naive_compressor is not None:
            modules_to_freeze.append(self.naive_compressor)

        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

    @staticmethod
    def _unpack_comm_output(
        comm_output: Union[
            torch.Tensor,
            Tuple[torch.Tensor, Dict[str, Any]],
            Dict[str, Any],
        ]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Support multiple possible RoCooperComm return styles.

        Recommended return style from rocooper_comm.py:
            impaired_feature, comm_info

        Also supports:
            impaired_feature
            {"feature": impaired_feature, "comm_info": {...}}
        """
        if isinstance(comm_output, torch.Tensor):
            return comm_output, {}

        if isinstance(comm_output, tuple):
            if len(comm_output) == 2:
                feature, info = comm_output
                if info is None:
                    info = {}
                return feature, info
            if len(comm_output) == 1:
                return comm_output[0], {}

        if isinstance(comm_output, dict):
            feature = (
                comm_output.get("feature")
                or comm_output.get("features")
                or comm_output.get("spatial_features_2d")
                or comm_output.get("x")
            )
            if feature is None:
                raise KeyError(
                    "RoCooperComm returned dict but no feature key was found. "
                    "Expected one of: feature/features/spatial_features_2d/x."
                )
            info = comm_output.get("comm_info", {})
            return feature, info

        raise TypeError(
            "Unsupported RoCooperComm output type: "
            f"{type(comm_output).__name__}"
        )

    @staticmethod
    def _unpack_fusion_output(
        fusion_output: Union[
            torch.Tensor,
            Tuple[torch.Tensor, Dict[str, Any]],
            Dict[str, Any],
        ]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Support multiple possible RoCooperFusion return styles.

        Recommended return style from rocooper_fuse.py:
            fused_feature, fusion_info

        Also supports:
            fused_feature
            {"fused_feature": fused_feature, "fusion_info": {...}}
        """
        if isinstance(fusion_output, torch.Tensor):
            return fusion_output, {}

        if isinstance(fusion_output, tuple):
            if len(fusion_output) == 2:
                feature, info = fusion_output
                if info is None:
                    info = {}
                return feature, info
            if len(fusion_output) == 1:
                return fusion_output[0], {}

        if isinstance(fusion_output, dict):
            feature = (
                fusion_output.get("fused_feature")
                or fusion_output.get("feature")
                or fusion_output.get("features")
                or fusion_output.get("spatial_features_2d")
                or fusion_output.get("x")
            )
            if feature is None:
                raise KeyError(
                    "RoCooperFusion returned dict but no fused feature key was found. "
                    "Expected one of: fused_feature/feature/features/"
                    "spatial_features_2d/x."
                )
            info = fusion_output.get("fusion_info", {})
            return feature, info

        raise TypeError(
            "Unsupported RoCooperFusion output type: "
            f"{type(fusion_output).__name__}"
        )

    def _extract_lidar_inputs(
        self,
        data_dict: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract OpenCOOD processed lidar inputs.
        """
        if "processed_lidar" not in data_dict:
            raise KeyError(
                "data_dict must contain 'processed_lidar'. "
                "Please use OpenCOOD IntermediateFusionDataset."
            )

        processed_lidar = data_dict["processed_lidar"]

        voxel_features = processed_lidar["voxel_features"]
        voxel_coords = processed_lidar["voxel_coords"]
        voxel_num_points = processed_lidar["voxel_num_points"]

        if "record_len" not in data_dict:
            raise KeyError(
                "data_dict must contain 'record_len'. "
                "RoCooper needs record_len to split ego and neighbor CAV features."
            )

        record_len = data_dict["record_len"]
        pairwise_t_matrix = data_dict.get("pairwise_t_matrix", None)

        return (
            voxel_features,
            voxel_coords,
            voxel_num_points,
            record_len,
            pairwise_t_matrix,
        )

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass.

        Main tensor shapes in the common OpenCOOD PointPillar setting:
            total_cav = sum(record_len)

            voxel_features:
                [num_voxels, max_points_per_voxel, point_dim]

            spatial_features_2d before shrink:
                [total_cav, 384, H, W]

            spatial_features_2d after shrink:
                [total_cav, 256, H', W']

            fused_feature:
                [batch_size, 256, H', W']

            psm:
                [batch_size, anchor_number, H', W']

            rm:
                [batch_size, 7 * anchor_number, H', W']
        """
        (
            voxel_features,
            voxel_coords,
            voxel_num_points,
            record_len,
            pairwise_t_matrix,
        ) = self._extract_lidar_inputs(data_dict)

        batch_dict = {
            "voxel_features": voxel_features,
            "voxel_coords": voxel_coords,
            "voxel_num_points": voxel_num_points,
            "record_len": record_len,
        }

        # --------------------------------------------------------------
        # 1. PointPillar feature extraction
        # --------------------------------------------------------------
        # [num_voxels, points, dim] -> encoded pillar features
        batch_dict = self.pillar_vfe(batch_dict)

        # encoded pillar features -> BEV pseudo image
        batch_dict = self.scatter(batch_dict)

        # BEV backbone
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict["spatial_features_2d"]

        # --------------------------------------------------------------
        # 2. Optional shrink
        # --------------------------------------------------------------
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        # RoCooper's BlockPrioritizer is ego-feature-centric in the paper.
        # Do not feed a clean pre-communication classification map as routing
        # input; it can leak undamaged confidence and makes the result hard to
        # compare with the communication-impaired setting.  The current
        # Aggregator ignores psm_single, so keeping it None is the strict path.
        psm_single = None

        # --------------------------------------------------------------
        # 3. Optional feature compression before transmission
        # --------------------------------------------------------------
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        # --------------------------------------------------------------
        # 4. RoCooper communication impairment
        # --------------------------------------------------------------
        # Recommended RoCooperComm signature:
        #     comm_module(
        #         features,
        #         record_len,
        #         pairwise_t_matrix=pairwise_t_matrix,
        #         data_dict=data_dict,
        #     )
        #
        # It should impair only non-ego CAV features when impair_ego=False.
        comm_output = self.comm_module(
            spatial_features_2d,
            record_len,
            pairwise_t_matrix=pairwise_t_matrix,
            data_dict=data_dict,
        )
        impaired_features, comm_info = self._unpack_comm_output(comm_output)

        # --------------------------------------------------------------
        # 5. RoCooper fusion
        # --------------------------------------------------------------
        # Recommended RoCooperFusion signature:
        #     fusion_net(
        #         features,
        #         record_len,
        #         pairwise_t_matrix,
        #         psm_single=psm_single,
        #         data_dict=data_dict,
        #     )
        #
        # Output should be one fused feature per ego sample in the batch.
        fusion_output = self.fusion_net(
            impaired_features,
            record_len,
            pairwise_t_matrix,
            psm_single=psm_single,
            data_dict=data_dict,
        )
        fused_feature, fusion_info = self._unpack_fusion_output(fusion_output)

        # --------------------------------------------------------------
        # 6. Detection heads
        # --------------------------------------------------------------
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {
            "psm": psm,
            "rm": rm,
        }

        if psm_single is not None:
            output_dict["psm_single"] = psm_single

        if comm_info:
            output_dict["comm_info"] = comm_info

        if fusion_info:
            output_dict["fusion_info"] = fusion_info

        return output_dict


# ----------------------------------------------------------------------
# Optional aliases
# ----------------------------------------------------------------------
# Some OpenCOOD dynamic import helpers infer class name from file name.
# Keeping aliases makes the file more tolerant to different loader styles.
PointPillarRoCooper = PointPillarRocooper
PointPillarROCOOPER = PointPillarRocooper