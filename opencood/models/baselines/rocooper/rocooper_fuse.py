# -*- coding: utf-8 -*-
"""
RoCooper fusion wrapper for OpenCOOD.

Save as:
    opencood/models/baselines/rocooper/rocooper_fuse.py

This file is intentionally lightweight.

It only orchestrates RoCooper submodules:
    1. Split each scenario into ego feature and neighbor CAV features.
    2. Optionally align CAV features to ego frame.
    3. Call RoCooperAugmentor for preliminary feature recovery.
    4. Call RoCooperAggregator for multi-scale regional cross-learning.
    5. Return one fused BEV feature map per ego vehicle.

Required companion files:
    opencood/models/baselines/rocooper/components/rocooper_augmentor.py
    opencood/models/baselines/rocooper/components/rocooper_aggregator.py
    opencood/models/baselines/rocooper/components/rocooper_block_prioritizer.py
    opencood/models/baselines/rocooper/components/rocooper_utils.py
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from opencood.models.baselines.rocooper.components.rocooper_augmentor import RoCooperAugmentor
from opencood.models.baselines.rocooper.components.rocooper_aggregator import RoCooperAggregator
from opencood.models.baselines.rocooper.components.rocooper_utils import (
    record_len_to_list,
    validate_feature_record_len,
    align_group_to_ego,
    fallback_fusion_single,
)


class RoCooperFusion(nn.Module):
    """
    RoCooper fusion module.

    This module is called by:
        opencood/models/baselines/rocooper/models/point_pillar_rocooper.py

    Expected input:
        features:
            Tensor with shape [sum(record_len), C, H, W].
            These are BEV features after PointPillar backbone and optional
            communication impairment.

        record_len:
            Tensor/List with shape [B].
            record_len[b] is the number of CAVs in the b-th scenario.
            The first CAV in each scenario is treated as the ego vehicle.

        pairwise_t_matrix:
            Optional OpenCOOD pairwise transformation matrix.
            Usually shape [B, max_cav, max_cav, 4, 4].

    Expected output:
        fused_feature:
            Tensor with shape [B, C, H, W].
            One fused BEV feature map for each ego vehicle.

        fusion_info:
            Dictionary containing debugging / routing / module information.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        super(RoCooperFusion, self).__init__()

        self.cfg = cfg or {}

        self.rocooper_cfg = self.cfg.get("rocooper", {}) or {}
        self.augmentor_cfg = self.cfg.get("augmentor", {}) or {}
        self.aggregator_cfg = self.cfg.get("aggregator", {}) or {}
        self.block_prioritizer_cfg = self.cfg.get("block_prioritizer", {}) or {}

        self.enabled = bool(self.rocooper_cfg.get("enabled", True))

        # RoCooper core switches.
        self.use_ego_anchor = bool(
            self.rocooper_cfg.get("use_ego_anchor", True)
        )

        self.use_augmentor = bool(
            self.rocooper_cfg.get("use_augmentor", True)
        ) and bool(
            self.augmentor_cfg.get("enabled", True)
        )

        self.use_aggregator = bool(
            self.rocooper_cfg.get("use_aggregator", True)
        ) and bool(
            self.aggregator_cfg.get("enabled", True)
        )

        # Feature dimension after PointPillar backbone / shrink header.
        self.channels = self._infer_channels(self.cfg)

        # Fallback fusion when RoCooper is disabled or a scenario has no
        # neighbor CAVs.
        self.fallback_fusion = str(
            self.rocooper_cfg.get("fallback_fusion", "mean")
        ).lower()

        # Feature alignment settings.
        alignment_cfg = self.rocooper_cfg.get("alignment", {}) or {}

        self.use_feature_alignment = bool(
            alignment_cfg.get(
                "enabled",
                self.rocooper_cfg.get("use_feature_alignment", True),
            )
        )

        self.discrete_ratio = float(
            alignment_cfg.get(
                "discrete_ratio",
                self._infer_discrete_ratio(self.cfg),
            )
        )

        # The final PointPillar BEV feature used by the detection head is
        # typically at feature_stride=4 for this OPV2V config.  Using 1 makes
        # the translation in affine normalization four times too large.
        self.downsample_rate = int(
            alignment_cfg.get(
                "downsample_rate",
                self.cfg.get("downsample_rate", 4),
            )
        )

        self.transform_direction = str(
            alignment_cfg.get(
                "transform_direction",
                self.rocooper_cfg.get("transform_direction", "source_to_ego"),
            )
        )

        # Submodules.
        self.augmentor = RoCooperAugmentor(
            cfg=self.augmentor_cfg,
            channels=self.channels,
        )

        self.aggregator = RoCooperAggregator(
            aggregator_cfg=self.aggregator_cfg,
            block_prioritizer_cfg=self.block_prioritizer_cfg,
            channels=self.channels,
        )

    @staticmethod
    def _infer_channels(cfg: Dict[str, Any]) -> int:
        """
        Infer BEV feature channel dimension from config.

        Priority:
            rocooper.in_channels
            aggregator.in_channels
            augmentor.in_channels
            feature_dim
            rocooper.hidden_channels
            default 256
        """
        rocooper_cfg = cfg.get("rocooper", {}) or {}
        aggregator_cfg = cfg.get("aggregator", {}) or {}
        augmentor_cfg = cfg.get("augmentor", {}) or {}

        candidates = [
            rocooper_cfg.get("in_channels", None),
            aggregator_cfg.get("in_channels", None),
            augmentor_cfg.get("in_channels", None),
            cfg.get("feature_dim", None),
            rocooper_cfg.get("hidden_channels", None),
        ]

        for value in candidates:
            if value is not None:
                return int(value)

        return 256

    @staticmethod
    def _infer_discrete_ratio(cfg: Dict[str, Any]) -> float:
        """
        Infer feature alignment discrete ratio.

        In many OpenCOOD configs, voxel_size[0] is 0.4 for OPV2V.
        """
        voxel_size = cfg.get("voxel_size", [0.4, 0.4, 4.0])

        if isinstance(voxel_size, (list, tuple)) and len(voxel_size) > 0:
            return float(voxel_size[0])

        return 0.4

    def reset_history(self) -> None:
        """
        Reset temporal history in Augmentor.

        Useful before starting a new validation sequence or when dataloader
        shuffling makes temporal continuity invalid.
        """
        if hasattr(self.augmentor, "reset_history"):
            self.augmentor.reset_history()

    @staticmethod
    def _unpack_augmentor_output(
        output: Union[
            torch.Tensor,
            Tuple[torch.Tensor, Dict[str, Any]],
            Dict[str, Any],
        ]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Accept several Augmentor return formats.

        Recommended:
            enhanced_others, augmentor_info

        Also accepted:
            enhanced_others
            {"features": enhanced_others, "info": {...}}
        """
        if isinstance(output, torch.Tensor):
            return output, {}

        if isinstance(output, tuple):
            if len(output) == 2:
                features, info = output
                return features, info or {}
            if len(output) == 1:
                return output[0], {}

        if isinstance(output, dict):
            features = (
                output.get("features")
                or output.get("feature")
                or output.get("others")
                or output.get("enhanced_features")
            )

            if features is None:
                raise KeyError(
                    "RoCooperAugmentor returned dict but no feature key "
                    "was found. Expected one of: features, feature, others, "
                    "enhanced_features."
                )

            info = (
                output.get("info")
                or output.get("augmentor_info")
                or {}
            )

            return features, info

        raise TypeError(
            "Unsupported RoCooperAugmentor output type: "
            f"{type(output).__name__}"
        )

    @staticmethod
    def _unpack_aggregator_output(
        output: Union[
            torch.Tensor,
            Tuple[torch.Tensor, Dict[str, Any]],
            Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]],
            Dict[str, Any],
        ],
        original_others: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Accept several Aggregator return formats.

        Recommended:
            fused_ego, updated_others, aggregator_info

        Also accepted:
            fused_ego, aggregator_info
            fused_ego
            {"fused_ego": ..., "updated_others": ..., "info": {...}}
        """
        if isinstance(output, torch.Tensor):
            return output, original_others, {}

        if isinstance(output, tuple):
            if len(output) == 3:
                fused_ego, updated_others, info = output
                return fused_ego, updated_others, info or {}

            if len(output) == 2:
                fused_ego, info = output
                return fused_ego, original_others, info or {}

            if len(output) == 1:
                return output[0], original_others, {}

        if isinstance(output, dict):
            fused_ego = (
                output.get("fused_ego")
                or output.get("ego")
                or output.get("feature")
                or output.get("fused_feature")
            )

            if fused_ego is None:
                raise KeyError(
                    "RoCooperAggregator returned dict but no fused feature "
                    "key was found. Expected one of: fused_ego, ego, feature, "
                    "fused_feature."
                )

            updated_others = (
                output.get("updated_others")
                or output.get("others")
                or original_others
            )

            info = (
                output.get("info")
                or output.get("aggregator_info")
                or {}
            )

            return fused_ego, updated_others, info

        raise TypeError(
            "Unsupported RoCooperAggregator output type: "
            f"{type(output).__name__}"
        )

    def _call_augmentor(
        self,
        others: torch.Tensor,
        batch_idx: int,
        data_dict: Optional[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Call RoCooperAugmentor with tolerant signatures.
        """
        try:
            output = self.augmentor(
                others,
                batch_idx=batch_idx,
                data_dict=data_dict,
            )
        except TypeError:
            output = self.augmentor(others)

        return self._unpack_augmentor_output(output)

    def _call_aggregator(
        self,
        ego: torch.Tensor,
        others: torch.Tensor,
        batch_idx: int,
        psm_single_group: Optional[torch.Tensor],
        data_dict: Optional[Dict[str, Any]],
        others_valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Call RoCooperAggregator with tolerant signatures.
        """
        try:
            output = self.aggregator(
                ego,
                others,
                batch_idx=batch_idx,
                psm_single=psm_single_group,
                data_dict=data_dict,
                others_valid_mask=others_valid_mask,
            )
        except TypeError:
            try:
                output = self.aggregator(
                    ego,
                    others,
                    batch_idx=batch_idx,
                    data_dict=data_dict,
                    others_valid_mask=others_valid_mask,
                )
            except TypeError:
                output = self.aggregator(ego, others)

        return self._unpack_aggregator_output(
            output,
            original_others=others,
        )

    def forward(
        self,
        features: torch.Tensor,
        record_len: Union[torch.Tensor, List[int], Tuple[int, ...]],
        pairwise_t_matrix: Optional[torch.Tensor] = None,
        psm_single: Optional[torch.Tensor] = None,
        data_dict: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            features:
                BEV features with shape [sum(record_len), C, H, W].

            record_len:
                Number of CAVs in each scenario.

            pairwise_t_matrix:
                Pairwise transformation matrix from OpenCOOD.
                Used by rocooper_utils.align_group_to_ego.

            psm_single:
                Optional single-agent classification map before fusion.
                Shape usually [sum(record_len), anchor_num, H, W].
                This module only passes it to Aggregator when needed.

            data_dict:
                Optional original OpenCOOD batch dictionary.

        Returns:
            fused_feature:
                Tensor with shape [B, C, H, W].

            fusion_info:
                Dictionary with module and scenario-level information.
        """
        if features.dim() != 4:
            raise ValueError(
                "RoCooperFusion expects features with shape [N, C, H, W], "
                f"but got {tuple(features.shape)}."
            )

        record_len_list = record_len_to_list(record_len)
        validate_feature_record_len(features, record_len_list)

        if psm_single is not None and psm_single.shape[0] != features.shape[0]:
            raise ValueError(
                "psm_single and features must have the same first dimension. "
                f"Got psm_single.shape[0]={psm_single.shape[0]}, "
                f"features.shape[0]={features.shape[0]}."
            )

        if not self.enabled:
            fused_feature = self._batch_fallback_fusion(
                features=features,
                record_len_list=record_len_list,
            )
            return fused_feature, {
                "rocooper_enabled": False,
                "reason": "RoCooperFusion disabled",
            }

        fused_features: List[torch.Tensor] = []

        fusion_info: Dict[str, Any] = {
            "rocooper_enabled": True,
            "use_ego_anchor": self.use_ego_anchor,
            "use_augmentor": self.use_augmentor,
            "use_aggregator": self.use_aggregator,
            "use_feature_alignment": self.use_feature_alignment,
            "alignment_downsample_rate": self.downsample_rate,
            "alignment_transform_direction": self.transform_direction,
            "fallback_fusion": self.fallback_fusion,
            "batch_size": len(record_len_list),
            "feature_shape": list(features.shape),
            "scenario_info": [],
        }

        start = 0

        for batch_idx, cav_num in enumerate(record_len_list):
            end = start + cav_num

            group = features[start:end]
            psm_single_group = None

            if psm_single is not None:
                psm_single_group = psm_single[start:end]

            start = end

            scenario_info: Dict[str, Any] = {
                "batch_idx": int(batch_idx),
                "num_cav": int(cav_num),
                "num_other_cav": int(max(0, cav_num - 1)),
            }

            # Align all CAV features to ego coordinate frame.
            group = align_group_to_ego(
                group=group,
                batch_idx=batch_idx,
                cav_num=cav_num,
                pairwise_t_matrix=pairwise_t_matrix,
                discrete_ratio=self.discrete_ratio,
                downsample_rate=self.downsample_rate,
                enabled=self.use_feature_alignment,
                transform_direction=self.transform_direction,
            )

            ego = group[0]
            others = group[1:]

            # A CAV whose feature map is completely zero after communication
            # should be treated as missing.  This mask is computed before the
            # Augmentor so Conv/BN/bias layers cannot hallucinate missing
            # neighbors into non-zero tensors.
            if others.numel() > 0 and others.shape[0] > 0:
                others_valid_mask = (
                    others.detach().abs().flatten(1).sum(dim=1) > 1e-8
                )
            else:
                others_valid_mask = torch.zeros(
                    0,
                    dtype=torch.bool,
                    device=others.device,
                )

            scenario_info["num_valid_other_cav_before_augmentor"] = int(
                others_valid_mask.sum().item()
            ) if others_valid_mask.numel() > 0 else 0

            if others.numel() == 0 or others.shape[0] == 0:
                fused_features.append(ego)
                scenario_info["fusion_case"] = "ego_only"
                scenario_info["augmentor"] = {"skipped": True}
                scenario_info["aggregator"] = {"skipped": True}
                fusion_info["scenario_info"].append(scenario_info)
                continue

            # ----------------------------------------------------------
            # 1. Augmentor: preliminary recovery of received CAV features
            # ----------------------------------------------------------
            if self.use_augmentor:
                others, augmentor_info = self._call_augmentor(
                    others=others,
                    batch_idx=batch_idx,
                    data_dict=data_dict,
                )
            else:
                augmentor_info = {
                    "augmentor_enabled": False,
                    "reason": "disabled_by_config",
                }

            scenario_info["augmentor"] = augmentor_info

            # Keep fully missing CAVs exactly zero after Augmentor.  This is
            # essential for hard-drop sanity checks and prevents bias-only
            # pseudo features from being fused as if they were received data.
            if others_valid_mask.numel() == others.shape[0]:
                others = others * others_valid_mask.to(
                    dtype=others.dtype,
                    device=others.device,
                ).view(-1, 1, 1, 1)

            scenario_info["num_valid_other_cav"] = int(
                others_valid_mask.sum().item()
            ) if others_valid_mask.numel() > 0 else 0

            if others_valid_mask.numel() > 0 and int(others_valid_mask.sum().item()) == 0:
                fused_features.append(ego)
                scenario_info["fusion_case"] = "ego_only_all_neighbors_missing"
                scenario_info["aggregator"] = {
                    "skipped": True,
                    "reason": "all_neighbor_features_missing",
                }
                fusion_info["scenario_info"].append(scenario_info)
                continue

            # ----------------------------------------------------------
            # 2. Aggregator: multi-scale regional cross-learning + fusion
            # ----------------------------------------------------------
            if self.use_aggregator:
                fused_ego, updated_others, aggregator_info = self._call_aggregator(
                    ego=ego,
                    others=others,
                    batch_idx=batch_idx,
                    psm_single_group=psm_single_group,
                    data_dict=data_dict,
                    others_valid_mask=others_valid_mask,
                )

                scenario_info["aggregator"] = aggregator_info
                scenario_info["updated_others_shape"] = list(
                    updated_others.shape
                )
                scenario_info["fusion_case"] = "rocooper"

            else:
                fused_ego = fallback_fusion_single(
                    ego=ego,
                    others=others,
                    mode=self.fallback_fusion,
                )

                scenario_info["aggregator"] = {
                    "aggregator_enabled": False,
                    "reason": "disabled_by_config",
                    "fallback_fusion": self.fallback_fusion,
                }
                scenario_info["fusion_case"] = "fallback"

            fused_features.append(fused_ego)
            fusion_info["scenario_info"].append(scenario_info)

        fused_feature = torch.stack(fused_features, dim=0)

        return fused_feature, fusion_info

    def _batch_fallback_fusion(
        self,
        features: torch.Tensor,
        record_len_list: List[int],
    ) -> torch.Tensor:
        """
        Fallback batch-wise fusion.

        Used when RoCooperFusion is disabled.
        """
        fused_features: List[torch.Tensor] = []

        start = 0

        for cav_num in record_len_list:
            end = start + cav_num
            group = features[start:end]
            start = end

            ego = group[0]
            others = group[1:]

            fused = fallback_fusion_single(
                ego=ego,
                others=others,
                mode=self.fallback_fusion,
            )
            fused_features.append(fused)

        return torch.stack(fused_features, dim=0)


# ----------------------------------------------------------------------
# Import aliases
# ----------------------------------------------------------------------

RocooperFusion = RoCooperFusion
ROCOOPERFusion = RoCooperFusion

__all__ = [
    "RoCooperFusion",
    "RocooperFusion",
    "ROCOOPERFusion",
]