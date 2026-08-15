# -*- coding: utf-8 -*-
"""
RoCooper BlockPrioritizer for OpenCOOD.

Save as:
    opencood/models/baselines/rocooper/components/rocooper_block_prioritizer.py

This module implements the BlockPrioritizer component in RoCooper.

Paper-level role:
    In RoCooper Aggregator, BEV features are partitioned into regional blocks.
    BlockPrioritizer dynamically assigns an importance weight to each ego block,
    selects the top-k high-weight blocks as ROI, and lets only these selected
    blocks participate in regional cross-learning.

Key design:
    Selection is self-centric:
        F_ego,k    = Topk(phi(F_ego), F_ego)
        F_others,k = Topk(phi(F_ego), F_others)

Expected common input:
    ego_blocks:
        Tensor [num_blocks, tokens_per_block, C]
        or     [B, num_blocks, tokens_per_block, C]

Optional input:
    others_blocks:
        Tensor [L, num_blocks, tokens_per_block, C]
        or     [B, L, num_blocks, tokens_per_block, C]

Main classes:
    BlockExpert
    ExpertGating
    BlockPrioritizer
"""

from typing import Any, Dict, Optional, Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.baselines.rocooper.components.rocooper_utils import (
    topk_indices_from_scores,
    make_topk_mask,
    gather_topk_blocks,
    safe_float,
    safe_int,
    tensor_summary,
)


class BlockExpert(nn.Module):
    """
    One expert network for assigning block-level importance scores.

    Input:
        block_summary:
            [num_blocks, C] or [B, num_blocks, C]

    Output:
        scores:
            [num_blocks] or [B, num_blocks]
    """

    def __init__(
        self,
        channels: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        activation: str = "gelu",
    ):
        super(BlockExpert, self).__init__()

        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.use_layer_norm = bool(use_layer_norm)

        if activation.lower() == "relu":
            act_layer = nn.ReLU(inplace=True)
        elif activation.lower() == "silu":
            act_layer = nn.SiLU(inplace=True)
        else:
            act_layer = nn.GELU()

        layers = []

        if self.use_layer_norm:
            layers.append(nn.LayerNorm(self.channels))

        layers.extend(
            [
                nn.Linear(self.channels, self.hidden_dim),
                act_layer,
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                act_layer,
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, 1),
            ]
        )

        self.net = nn.Sequential(*layers)

    def forward(self, block_summary: torch.Tensor) -> torch.Tensor:
        if block_summary.dim() not in (2, 3):
            raise ValueError(
                "BlockExpert expects block_summary with shape [N, C] "
                "or [B, N, C], "
                f"but got {tuple(block_summary.shape)}."
            )

        scores = self.net(block_summary).squeeze(-1)

        return scores


class ExpertGating(nn.Module):
    """
    Gating unit for combining multiple experts.

    Input:
        block_summary:
            [num_blocks, C] or [B, num_blocks, C]

    Output:
        gate_weights:
            [num_blocks, num_experts] or [B, num_blocks, num_experts]
    """

    def __init__(
        self,
        channels: int,
        hidden_dim: int = 128,
        num_experts: int = 4,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        activation: str = "gelu",
    ):
        super(ExpertGating, self).__init__()

        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.num_experts = int(num_experts)
        self.dropout = float(dropout)
        self.use_layer_norm = bool(use_layer_norm)

        if activation.lower() == "relu":
            act_layer = nn.ReLU(inplace=True)
        elif activation.lower() == "silu":
            act_layer = nn.SiLU(inplace=True)
        else:
            act_layer = nn.GELU()

        layers = []

        if self.use_layer_norm:
            layers.append(nn.LayerNorm(self.channels))

        layers.extend(
            [
                nn.Linear(self.channels, self.hidden_dim),
                act_layer,
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.num_experts),
            ]
        )

        self.net = nn.Sequential(*layers)

    def forward(self, block_summary: torch.Tensor) -> torch.Tensor:
        if block_summary.dim() not in (2, 3):
            raise ValueError(
                "ExpertGating expects block_summary with shape [N, C] "
                "or [B, N, C], "
                f"but got {tuple(block_summary.shape)}."
            )

        logits = self.net(block_summary)
        gate_weights = torch.softmax(logits, dim=-1)

        return gate_weights


class BlockPrioritizer(nn.Module):
    """
    RoCooper BlockPrioritizer.

    This module computes block weights using multiple experts and a gating unit,
    then selects top-k blocks according to selected_block_ratio.

    Constructor is compatible with:
        BlockPrioritizer(cfg, channels)

    Common usage in Aggregator:
        scores = block_prioritizer(ego_blocks)
        topk_indices = block_prioritizer.select_topk(scores)
        ego_selected = gather_topk_blocks(ego_blocks, topk_indices)
        others_selected = gather_topk_blocks(others_blocks, topk_indices)
    """

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        channels: int = 256,
    ):
        super(BlockPrioritizer, self).__init__()

        self.cfg = cfg or {}
        self.channels = int(channels)

        self.enabled = bool(self.cfg.get("enabled", True))

        self.selected_block_ratio = safe_float(
            self.cfg.get("selected_block_ratio", 0.6),
            0.6,
        )
        self.selected_block_ratio = max(0.0, min(1.0, self.selected_block_ratio))

        self.self_centric_selection = bool(
            self.cfg.get("self_centric_selection", True)
        )

        self.num_experts = max(1, safe_int(self.cfg.get("num_experts", 4), 4))

        self.expert_hidden_dim = safe_int(
            self.cfg.get("expert_hidden_dim", 128),
            128,
        )
        self.gating_hidden_dim = safe_int(
            self.cfg.get("gating_hidden_dim", 128),
            128,
        )

        self.dropout = safe_float(self.cfg.get("dropout", 0.0), 0.0)

        self.use_layer_norm = bool(self.cfg.get("use_layer_norm", True))
        self.activation = str(self.cfg.get("activation", "gelu")).lower()

        self.score_activation = str(
            self.cfg.get("score_activation", "sigmoid")
        ).lower()

        self.score_temperature = safe_float(
            self.cfg.get("score_temperature", 1.0),
            1.0,
        )
        self.score_temperature = max(1e-6, self.score_temperature)

        self.threshold_mode = str(
            self.cfg.get("threshold_mode", "topk")
        ).lower()

        self.bypass_unselected_blocks = bool(
            self.cfg.get("bypass_unselected_blocks", True)
        )

        self.return_debug_stats = bool(
            self.cfg.get("return_debug_stats", False)
        )

        self.min_selected_blocks = safe_int(
            self.cfg.get("min_selected_blocks", 1),
            1,
        )
        self.min_selected_blocks = max(1, self.min_selected_blocks)

        self.max_selected_blocks = self.cfg.get("max_selected_blocks", None)
        if self.max_selected_blocks is not None:
            self.max_selected_blocks = max(1, int(self.max_selected_blocks))

        self.summary_mode = str(
            self.cfg.get("summary_mode", "mean")
        ).lower()

        self.use_score_norm = bool(self.cfg.get("use_score_norm", False))

        self.experts = nn.ModuleList(
            [
                BlockExpert(
                    channels=self.channels,
                    hidden_dim=self.expert_hidden_dim,
                    dropout=self.dropout,
                    use_layer_norm=self.use_layer_norm,
                    activation=self.activation,
                )
                for _ in range(self.num_experts)
            ]
        )

        self.gating = ExpertGating(
            channels=self.channels,
            hidden_dim=self.gating_hidden_dim,
            num_experts=self.num_experts,
            dropout=self.dropout,
            use_layer_norm=self.use_layer_norm,
            activation=self.activation,
        )

        # Optional score refinement. It keeps the module close to the paper's
        # "deep routing learning strategy" while staying lightweight.
        self.score_refine = nn.Sequential(
            nn.LayerNorm(self.channels),
            nn.Linear(self.channels, self.channels),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.channels, 1),
        )

        self.score_refine_weight = safe_float(
            self.cfg.get("score_refine_weight", 0.0),
            0.0,
        )

    # ------------------------------------------------------------------
    # Block summary
    # ------------------------------------------------------------------

    def summarize_blocks(self, blocks: torch.Tensor) -> torch.Tensor:
        """
        Convert block tokens to one vector per block.

        Args:
            blocks:
                [num_blocks, tokens, C]
                or [B, num_blocks, tokens, C]

        Returns:
            block_summary:
                [num_blocks, C]
                or [B, num_blocks, C]
        """
        if blocks.dim() not in (3, 4):
            raise ValueError(
                "summarize_blocks expects blocks with shape "
                "[num_blocks, tokens, C] or [B, num_blocks, tokens, C], "
                f"but got {tuple(blocks.shape)}."
            )

        if blocks.shape[-1] != self.channels:
            raise ValueError(
                "BlockPrioritizer channel mismatch. "
                f"Configured channels={self.channels}, "
                f"but blocks last dim is {blocks.shape[-1]}."
            )

        if self.summary_mode == "max":
            return torch.max(blocks, dim=-2).values

        if self.summary_mode == "meanmax":
            mean_summary = blocks.mean(dim=-2)
            max_summary = torch.max(blocks, dim=-2).values
            return 0.5 * (mean_summary + max_summary)

        if self.summary_mode == "l2":
            return torch.sqrt(torch.clamp(blocks.pow(2).mean(dim=-2), min=1e-12))

        # Default: mean pooling over tokens in one block.
        return blocks.mean(dim=-2)

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------

    def compute_scores(
        self,
        ego_blocks: torch.Tensor,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute importance score for each ego block.

        Args:
            ego_blocks:
                [num_blocks, tokens, C]
                or [B, num_blocks, tokens, C]

            return_details:
                Whether to return expert scores and gating weights.

        Returns:
            scores:
                [num_blocks] or [B, num_blocks]

            details:
                optional dict.
        """
        block_summary = self.summarize_blocks(ego_blocks)

        if not self.enabled:
            scores = torch.ones(
                block_summary.shape[:-1],
                device=block_summary.device,
                dtype=block_summary.dtype,
            )

            if return_details:
                return scores, {
                    "block_summary": block_summary,
                }

            return scores

        expert_scores = torch.stack(
            [expert(block_summary) for expert in self.experts],
            dim=-1,
        )  # [..., num_experts]

        gate_weights = self.gating(block_summary)  # [..., num_experts]

        raw_scores = (expert_scores * gate_weights).sum(dim=-1)

        if self.score_refine_weight != 0:
            refine_scores = self.score_refine(block_summary).squeeze(-1)
            raw_scores = raw_scores + self.score_refine_weight * refine_scores

        raw_scores = raw_scores / self.score_temperature

        scores = self._activate_scores(raw_scores)

        if self.use_score_norm:
            scores = self._normalize_scores(scores)

        if return_details:
            return scores, {
                "block_summary": block_summary,
                "expert_scores": expert_scores,
                "gate_weights": gate_weights,
                "raw_scores": raw_scores,
            }

        return scores

    def _activate_scores(self, raw_scores: torch.Tensor) -> torch.Tensor:
        """
        Apply score activation.
        """
        if self.score_activation == "none":
            return raw_scores

        if self.score_activation == "softplus":
            return F.softplus(raw_scores)

        if self.score_activation == "tanh":
            return 0.5 * (torch.tanh(raw_scores) + 1.0)

        if self.score_activation == "softmax":
            return torch.softmax(raw_scores, dim=-1)

        # Default: sigmoid score in [0, 1].
        return torch.sigmoid(raw_scores)

    @staticmethod
    def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
        """
        Min-max normalize scores along block dimension.
        """
        min_v = scores.min(dim=-1, keepdim=True).values
        max_v = scores.max(dim=-1, keepdim=True).values

        return (scores - min_v) / (max_v - min_v + 1e-6)

    def forward(
        self,
        ego_blocks: torch.Tensor,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward alias of compute_scores.

        Args:
            ego_blocks:
                [num_blocks, tokens, C] or [B, num_blocks, tokens, C]

            return_details:
                Whether to return internal tensors.

        Returns:
            scores or (scores, details)
        """
        return self.compute_scores(
            ego_blocks=ego_blocks,
            return_details=return_details,
        )

    # ------------------------------------------------------------------
    # Top-k / threshold
    # ------------------------------------------------------------------

    def get_k(
        self,
        num_blocks: int,
        ratio: Optional[float] = None,
    ) -> int:
        """
        Compute selected block number k.

        Args:
            num_blocks:
                Total number of blocks.

            ratio:
                Optional selected block ratio. If None, use config value.

        Returns:
            k
        """
        num_blocks = int(num_blocks)

        if num_blocks <= 0:
            return 0

        if ratio is None:
            ratio = self.selected_block_ratio

        ratio = max(0.0, min(1.0, float(ratio)))

        k = int(math.ceil(ratio * num_blocks))
        k = max(self.min_selected_blocks, k)
        k = min(num_blocks, k)

        if self.max_selected_blocks is not None:
            k = min(k, int(self.max_selected_blocks))

        return max(1, k)

    def select_topk(
        self,
        scores: torch.Tensor,
        ratio: Optional[float] = None,
        return_threshold: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Select top-k block indices.

        Args:
            scores:
                [num_blocks] or [B, num_blocks]

            ratio:
                Optional selected ratio.

            return_threshold:
                Whether to also return kth score threshold.

        Returns:
            indices:
                If scores is [num_blocks], return [k].
                If scores is [B, num_blocks], return [B, k].

            threshold:
                Optional scalar or [B].
        """
        if scores.dim() == 1:
            num_blocks = scores.shape[0]
            k = self.get_k(num_blocks, ratio=ratio)

            indices = torch.topk(scores, k=k, dim=0).indices

            if return_threshold:
                threshold = torch.min(scores.index_select(0, indices))
                return indices, threshold

            return indices

        if scores.dim() == 2:
            num_blocks = scores.shape[1]
            k = self.get_k(num_blocks, ratio=ratio)

            topk = torch.topk(scores, k=k, dim=1)
            indices = topk.indices

            if return_threshold:
                threshold = topk.values[:, -1]
                return indices, threshold

            return indices

        raise ValueError(
            "select_topk expects scores with shape [num_blocks] or "
            f"[B, num_blocks], got {tuple(scores.shape)}."
        )

    def compute_threshold(
        self,
        scores: torch.Tensor,
        ratio: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute top-k threshold value.

        This corresponds to Th_pk(W_l) in the paper.

        Args:
            scores:
                [num_blocks] or [B, num_blocks]

        Returns:
            threshold:
                scalar tensor or [B]
        """
        _, threshold = self.select_topk(
            scores,
            ratio=ratio,
            return_threshold=True,
        )

        return threshold

    def make_selection_mask(
        self,
        scores: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
        ratio: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Make boolean mask for selected top-k blocks.

        Args:
            scores:
                [num_blocks] or [B, num_blocks]

            indices:
                Optional top-k indices.

            ratio:
                Optional selected ratio.

        Returns:
            mask:
                [num_blocks] or [B, num_blocks]
        """
        if scores.dim() == 1:
            if indices is None:
                indices = self.select_topk(scores, ratio=ratio)

            return make_topk_mask(
                num_blocks=scores.shape[0],
                indices=indices,
                device=scores.device,
            )

        if scores.dim() == 2:
            batch_size, num_blocks = scores.shape

            if indices is None:
                indices = self.select_topk(scores, ratio=ratio)

            mask = torch.zeros(
                batch_size,
                num_blocks,
                dtype=torch.bool,
                device=scores.device,
            )

            mask.scatter_(dim=1, index=indices.long(), value=True)

            return mask

        raise ValueError(
            "make_selection_mask expects scores with shape [num_blocks] "
            f"or [B, num_blocks], got {tuple(scores.shape)}."
        )

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def route(
        self,
        ego_blocks: torch.Tensor,
        others_blocks: Optional[torch.Tensor] = None,
        ratio: Optional[float] = None,
        return_info: bool = True,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]],
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, Any]],
    ]:
        """
        Compute scores, select top-k, and optionally gather selected blocks.

        Args:
            ego_blocks:
                [num_blocks, tokens, C]
                or [B, num_blocks, tokens, C]

            others_blocks:
                Optional.
                If ego_blocks is 3D:
                    [L, num_blocks, tokens, C]
                If ego_blocks is 4D:
                    [B, L, num_blocks, tokens, C]

            ratio:
                Optional selected block ratio.

            return_info:
                Whether to return info dictionary.

        Returns:
            If others_blocks is None:
                selected_ego, indices, info

            If others_blocks is not None:
                selected_ego, selected_others, indices, info
        """
        scores, details = self.compute_scores(
            ego_blocks,
            return_details=True,
        )

        indices = self.select_topk(scores, ratio=ratio)
        mask = self.make_selection_mask(scores, indices=indices, ratio=ratio)
        threshold = self.compute_threshold(scores, ratio=ratio)

        selected_ego = self._gather_ego_blocks(ego_blocks, indices)

        info = {
            "enabled": self.enabled,
            "selected_block_ratio": (
                self.selected_block_ratio if ratio is None else float(ratio)
            ),
            "num_total_blocks": int(ego_blocks.shape[-3]),
            "num_selected_blocks": int(indices.shape[-1]),
            "threshold_mode": self.threshold_mode,
            "bypass_unselected_blocks": self.bypass_unselected_blocks,
            "indices": indices.detach(),
            "selection_mask": mask.detach(),
            "threshold": threshold.detach(),
        }

        if self.return_debug_stats:
            info["scores"] = scores.detach()
            info["score_summary"] = tensor_summary(scores)
            info["gate_summary"] = tensor_summary(details["gate_weights"])
            info["expert_score_summary"] = tensor_summary(details["expert_scores"])

        if others_blocks is None:
            if return_info:
                return selected_ego, indices, info
            return selected_ego, indices

        selected_others = self._gather_others_blocks(others_blocks, indices)

        if return_info:
            return selected_ego, selected_others, indices, info

        return selected_ego, selected_others, indices

    @staticmethod
    def _gather_ego_blocks(
        ego_blocks: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gather ego top-k blocks.

        Args:
            ego_blocks:
                [num_blocks, tokens, C] or [B, num_blocks, tokens, C]

            indices:
                [k] or [B, k]

        Returns:
            selected ego blocks:
                [k, tokens, C] or [B, k, tokens, C]
        """
        if ego_blocks.dim() == 3:
            if indices.dim() != 1:
                raise ValueError(
                    "For 3D ego_blocks, indices must be [k]. "
                    f"Got indices shape {tuple(indices.shape)}."
                )
            return gather_topk_blocks(ego_blocks, indices)

        if ego_blocks.dim() == 4:
            if indices.dim() != 2:
                raise ValueError(
                    "For 4D ego_blocks, indices must be [B, k]. "
                    f"Got indices shape {tuple(indices.shape)}."
                )

            batch_size = ego_blocks.shape[0]
            selected = []

            for b in range(batch_size):
                selected.append(gather_topk_blocks(ego_blocks[b], indices[b]))

            return torch.stack(selected, dim=0)

        raise ValueError(
            "ego_blocks must be [num_blocks, tokens, C] or "
            f"[B, num_blocks, tokens, C], got {tuple(ego_blocks.shape)}."
        )

    @staticmethod
    def _gather_others_blocks(
        others_blocks: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gather other-CAV blocks using ego-selected indices.

        Args:
            others_blocks:
                [L, num_blocks, tokens, C]
                or [B, L, num_blocks, tokens, C]

            indices:
                [k] or [B, k]

        Returns:
            selected others:
                [L, k, tokens, C]
                or [B, L, k, tokens, C]
        """
        if others_blocks.dim() == 4:
            if indices.dim() != 1:
                raise ValueError(
                    "For 4D others_blocks, indices must be [k]. "
                    f"Got indices shape {tuple(indices.shape)}."
                )
            return gather_topk_blocks(others_blocks, indices)

        if others_blocks.dim() == 5:
            if indices.dim() != 2:
                raise ValueError(
                    "For 5D others_blocks, indices must be [B, k]. "
                    f"Got indices shape {tuple(indices.shape)}."
                )

            batch_size = others_blocks.shape[0]
            selected = []

            for b in range(batch_size):
                selected.append(gather_topk_blocks(others_blocks[b], indices[b]))

            return torch.stack(selected, dim=0)

        raise ValueError(
            "others_blocks must be [L, num_blocks, tokens, C] or "
            f"[B, L, num_blocks, tokens, C], got {tuple(others_blocks.shape)}."
        )

    # ------------------------------------------------------------------
    # Residual routing application
    # ------------------------------------------------------------------

    def apply_routing_residual(
        self,
        original_blocks: torch.Tensor,
        updated_selected_blocks: torch.Tensor,
        indices: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        weighted_residual: bool = True,
    ) -> torch.Tensor:
        """
        Scatter updated selected blocks back with optional block-score weighting.

        This implements the spirit of the paper's routing update:

            x_i^{l+1} =
                w_i * CA(x_i^l, x_others^l) + x_i^l, if w_i >= threshold
                x_i^l,                                otherwise

        Args:
            original_blocks:
                [num_blocks, tokens, C] or [B, num_blocks, tokens, C]

            updated_selected_blocks:
                [k, tokens, C] or [B, k, tokens, C]

            indices:
                [k] or [B, k]

            scores:
                Optional [num_blocks] or [B, num_blocks].

            weighted_residual:
                Whether to weight selected updates by their scores.

        Returns:
            new_blocks:
                Same shape as original_blocks.
        """
        if original_blocks.dim() == 3:
            return self._apply_routing_residual_single(
                original_blocks=original_blocks,
                updated_selected_blocks=updated_selected_blocks,
                indices=indices,
                scores=scores,
                weighted_residual=weighted_residual,
            )

        if original_blocks.dim() == 4:
            if indices.dim() != 2:
                raise ValueError(
                    "For batched original_blocks, indices must be [B, k]."
                )

            outputs = []
            batch_size = original_blocks.shape[0]

            for b in range(batch_size):
                score_b = None
                if scores is not None:
                    score_b = scores[b]

                outputs.append(
                    self._apply_routing_residual_single(
                        original_blocks=original_blocks[b],
                        updated_selected_blocks=updated_selected_blocks[b],
                        indices=indices[b],
                        scores=score_b,
                        weighted_residual=weighted_residual,
                    )
                )

            return torch.stack(outputs, dim=0)

        raise ValueError(
            "original_blocks must be [num_blocks, tokens, C] or "
            f"[B, num_blocks, tokens, C], got {tuple(original_blocks.shape)}."
        )

    @staticmethod
    def _apply_routing_residual_single(
        original_blocks: torch.Tensor,
        updated_selected_blocks: torch.Tensor,
        indices: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        weighted_residual: bool = True,
    ) -> torch.Tensor:
        """
        Single-sample routing residual.
        """
        if original_blocks.dim() != 3:
            raise ValueError(
                "original_blocks must be [num_blocks, tokens, C]."
            )

        if updated_selected_blocks.dim() != 3:
            raise ValueError(
                "updated_selected_blocks must be [k, tokens, C]."
            )

        if indices.dim() != 1:
            raise ValueError("indices must be [k].")

        selected_original = original_blocks.index_select(0, indices.long())
        delta = updated_selected_blocks - selected_original

        if weighted_residual and scores is not None:
            selected_scores = scores.index_select(0, indices.long())
            selected_scores = selected_scores.view(-1, 1, 1)
            delta = selected_scores * delta

        new_selected = selected_original + delta

        new_blocks = original_blocks.clone()
        new_blocks.index_copy_(0, indices.long(), new_selected)

        return new_blocks


# ----------------------------------------------------------------------
# Import aliases
# ----------------------------------------------------------------------

RoCooperBlockPrioritizer = BlockPrioritizer
RocooperBlockPrioritizer = BlockPrioritizer
ROCOOPERBlockPrioritizer = BlockPrioritizer

__all__ = [
    "BlockExpert",
    "ExpertGating",
    "BlockPrioritizer",
    "RoCooperBlockPrioritizer",
    "RocooperBlockPrioritizer",
    "ROCOOPERBlockPrioritizer",
]