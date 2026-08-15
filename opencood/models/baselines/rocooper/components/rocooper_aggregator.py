# -*- coding: utf-8 -*-
"""
RoCooper Aggregator for OpenCOOD.

This replacement is intentionally more conservative than the previous
implementation: it prevents fully missing CAV features from being converted into
bias-only hallucinations, skips cross-attention when no valid neighbor exists,
and adds an explicit gated residual neighbor-fusion path so the detection head
can learn to use aligned CAV information instead of receiving only a heavily
transformed ego feature.

Expected input:
    ego:    Tensor [C, H, W]
    others: Tensor [L, C, H, W]

Expected output:
    fused_ego:       Tensor [C, H, W]
    updated_others:  Tensor [L, C, H, W]
    aggregator_info: Dict
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.baselines.rocooper.components.rocooper_block_prioritizer import BlockPrioritizer
from opencood.models.baselines.rocooper.components.rocooper_utils import (
    valid_num_heads,
    partition_feature_map,
    reverse_partition_feature_map,
    gather_topk_blocks,
    scatter_topk_blocks,
    fallback_fusion_single,
    safe_float,
    safe_int,
    tensor_summary,
)


class ScaleSelector(nn.Module):
    """Lightweight scale selector with fixed/all-scale behavior by default."""

    def __init__(self, cfg: Optional[Dict[str, Any]], scales: List[int]):
        super(ScaleSelector, self).__init__()
        self.cfg = cfg or {}
        self.scales = [int(s) for s in scales if int(s) > 0] or [4]
        self.enabled = bool(self.cfg.get("enabled", True))
        self.select_all_at_start = bool(self.cfg.get("select_all_at_start", True))

        selected_scales = self.cfg.get("selected_scales", None)
        if selected_scales is not None:
            selected_scales = [int(s) for s in selected_scales]
            self.fixed_selected_scales = [s for s in selected_scales if s in self.scales]
        else:
            self.fixed_selected_scales = []

        self.num_selected_scales = safe_int(
            self.cfg.get("num_selected_scales", len(self.scales)),
            len(self.scales),
        )
        self.num_selected_scales = max(1, min(len(self.scales), self.num_selected_scales))
        self.scale_logits = nn.Parameter(torch.zeros(len(self.scales)))

    def forward(
        self,
        ego: torch.Tensor,
        others: Optional[torch.Tensor] = None,
    ) -> Tuple[List[int], Dict[str, Any]]:
        del ego, others
        info: Dict[str, Any] = {
            "scale_selector_enabled": self.enabled,
            "all_scales": self.scales,
        }
        if not self.enabled:
            info.update({"mode": "disabled", "selected_scales": self.scales})
            return self.scales, info
        if len(self.fixed_selected_scales) > 0:
            info.update({"mode": "fixed_selected_scales", "selected_scales": self.fixed_selected_scales})
            return self.fixed_selected_scales, info
        if self.select_all_at_start:
            info.update({"mode": "select_all", "selected_scales": self.scales})
            return self.scales, info

        topk = torch.topk(self.scale_logits, k=self.num_selected_scales, dim=0).indices
        selected = sorted([self.scales[int(i)] for i in topk.detach().cpu().tolist()])
        info.update({
            "mode": "learnable_hard_topk",
            "selected_scales": selected,
            "scale_logits": self.scale_logits.detach(),
        })
        return selected, info


class CrossAttentionBlock(nn.Module):
    """
    Ego-to-neighbor regional cross-attention.

    The previous implementation also updated other CAV features using ego.  That
    makes it easy for zero/missing CAV tensors to become non-zero again and can
    train the module into an ego-only transform.  Here only the ego ROI blocks are
    updated; valid neighbor blocks serve as key/value memory.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        mlp_ratio: float = 2.0,
        residual_scale: float = 0.5,
    ):
        super(CrossAttentionBlock, self).__init__()
        self.channels = int(channels)
        self.num_heads = valid_num_heads(self.channels, int(num_heads))
        self.dropout_value = float(dropout)
        self.mlp_ratio = float(mlp_ratio)
        self.residual_scale = float(residual_scale)

        hidden_dim = max(self.channels, int(self.channels * self.mlp_ratio))

        self.q_norm = nn.LayerNorm(self.channels)
        self.kv_norm = nn.LayerNorm(self.channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.channels,
            num_heads=self.num_heads,
            dropout=self.dropout_value,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.channels),
            nn.Linear(self.channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_value),
            nn.Linear(hidden_dim, self.channels),
            nn.Dropout(self.dropout_value),
        )
        self.dropout = nn.Dropout(self.dropout_value)

    def forward(
        self,
        ego_selected: torch.Tensor,
        others_selected: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if ego_selected.dim() != 3:
            raise ValueError(
                "CrossAttentionBlock expects ego_selected [K, T, C], "
                f"got {tuple(ego_selected.shape)}."
            )
        if others_selected.dim() != 4:
            raise ValueError(
                "CrossAttentionBlock expects others_selected [L, K, T, C], "
                f"got {tuple(others_selected.shape)}."
            )
        if ego_selected.numel() == 0:
            return ego_selected, others_selected
        if others_selected.numel() == 0 or others_selected.shape[0] == 0:
            return ego_selected, others_selected

        num_other, k, tokens, channels = others_selected.shape
        if ego_selected.shape != (k, tokens, channels):
            raise ValueError(
                "ego_selected and others_selected shapes are inconsistent. "
                f"ego={tuple(ego_selected.shape)}, others={tuple(others_selected.shape)}."
            )

        ego_query = self.q_norm(ego_selected)
        others_kv = (
            others_selected.permute(1, 0, 2, 3)
            .contiguous()
            .view(k, num_other * tokens, channels)
        )
        others_kv = self.kv_norm(others_kv)

        attn_out, _ = self.attn(
            query=ego_query,
            key=others_kv,
            value=others_kv,
            need_weights=False,
        )

        # Residual scale keeps early training close to ego-only but still gives
        # a direct gradient path from valid neighbor features to ego blocks.
        updated = ego_selected + self.residual_scale * self.dropout(attn_out)
        updated = updated + self.residual_scale * self.mlp(updated)
        return updated, others_selected


class FeatureSelfRefine(nn.Module):
    """Optional local refinement after selected ROI blocks are scattered back."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        mlp_ratio: float = 2.0,
        max_tokens_for_attention: int = 4096,
        fallback_mode: str = "conv",
        residual_scale: float = 0.25,
    ):
        super(FeatureSelfRefine, self).__init__()
        self.channels = int(channels)
        self.num_heads = valid_num_heads(self.channels, int(num_heads))
        self.dropout_value = float(dropout)
        self.mlp_ratio = float(mlp_ratio)
        self.max_tokens_for_attention = int(max_tokens_for_attention)
        self.fallback_mode = str(fallback_mode).lower()
        self.residual_scale = float(residual_scale)

        hidden_dim = max(self.channels, int(self.channels * self.mlp_ratio))
        self.norm = nn.LayerNorm(self.channels)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.channels,
            num_heads=self.num_heads,
            dropout=self.dropout_value,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.channels),
            nn.Linear(self.channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_value),
            nn.Linear(hidden_dim, self.channels),
            nn.Dropout(self.dropout_value),
        )
        self.conv_fallback = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, groups=self.channels, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                "FeatureSelfRefine expects x [N, C, H, W], "
                f"got {tuple(x.shape)}."
            )
        if x.numel() == 0 or x.shape[0] == 0:
            return x
        n, c, h, w = x.shape
        if c != self.channels:
            raise ValueError(
                f"FeatureSelfRefine channel mismatch: cfg={self.channels}, input={c}."
            )
        tokens = h * w
        if tokens > self.max_tokens_for_attention:
            if self.fallback_mode == "none":
                return x
            return x + self.residual_scale * self.conv_fallback(x)

        seq = x.flatten(2).transpose(1, 2).contiguous()
        seq_norm = self.norm(seq)
        attn_out, _ = self.self_attn(seq_norm, seq_norm, seq_norm, need_weights=False)
        seq = seq + self.residual_scale * attn_out
        seq = seq + self.residual_scale * self.mlp(seq)
        return seq.transpose(1, 2).contiguous().view(n, c, h, w)


class SplitAttentionFusion(nn.Module):
    """Channel-wise split attention for merging multi-scale outputs."""

    def __init__(self, channels: int, max_num_splits: int = 3, reduction_factor: int = 4):
        super(SplitAttentionFusion, self).__init__()
        self.channels = int(channels)
        self.max_num_splits = int(max_num_splits)
        hidden = max(8, self.channels // max(1, int(reduction_factor)))
        self.fc1 = nn.Conv2d(self.channels, hidden, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, self.channels * self.max_num_splits, kernel_size=1, bias=True)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        if len(features) == 0:
            raise ValueError("SplitAttentionFusion received an empty feature list.")
        if len(features) == 1:
            return features[0]
        if len(features) > self.max_num_splits:
            raise ValueError(
                f"Got {len(features)} splits but max_num_splits={self.max_num_splits}."
            )
        base_shape = features[0].shape
        for f in features:
            if f.shape != base_shape:
                raise ValueError(
                    "All split features must have same shape, "
                    f"got {tuple(f.shape)} vs {tuple(base_shape)}."
                )
        stacked = torch.stack(features, dim=1)  # [N, S, C, H, W]
        pooled = stacked.sum(dim=1).mean(dim=(2, 3), keepdim=True)
        logits = self.fc2(self.relu(self.fc1(pooled)))
        n, _, _, _ = logits.shape
        logits = logits.view(n, self.max_num_splits, self.channels, 1, 1)
        logits = logits[:, : len(features)]
        weights = torch.softmax(logits, dim=1)
        return (stacked * weights).sum(dim=1)


class GatedNeighborFusion(nn.Module):
    """
    Explicit final neighbor-to-ego fusion path.

    The previous Aggregator could strongly transform ego while barely changing
    AP when all neighbors were dropped.  This module supplies a direct residual
    path from valid, aligned neighbor features to ego.  When no valid neighbor is
    available, the caller skips it and returns ego unchanged.
    """

    def __init__(
        self,
        channels: int,
        gate_hidden_ratio: float = 0.25,
        gate_bias_init: float = -1.0,
        fusion_mode: str = "mean",
        use_channel_gate: bool = False,
    ):
        super(GatedNeighborFusion, self).__init__()
        self.channels = int(channels)
        self.fusion_mode = str(fusion_mode).lower()
        self.use_channel_gate = bool(use_channel_gate)
        hidden = max(16, int(self.channels * float(gate_hidden_ratio)))
        gate_out_channels = self.channels if self.use_channel_gate else 1
        self.gate_net = nn.Sequential(
            nn.Conv2d(self.channels * 3, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, gate_out_channels, kernel_size=1, bias=True),
        )
        nn.init.constant_(self.gate_net[-1].bias, float(gate_bias_init))

    def _neighbor_context(self, others: torch.Tensor) -> torch.Tensor:
        if self.fusion_mode == "max":
            return torch.max(others, dim=0).values
        if self.fusion_mode == "sum":
            return torch.sum(others, dim=0)
        return torch.mean(others, dim=0)

    def forward(
        self,
        ego: torch.Tensor,
        others: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if others.numel() == 0 or others.shape[0] == 0:
            return ego, {"gated_neighbor_fusion": False, "reason": "no_valid_neighbor"}
        context = self._neighbor_context(others)
        gate_input = torch.cat([ego, context, torch.abs(ego - context)], dim=0).unsqueeze(0)
        gate = torch.sigmoid(self.gate_net(gate_input))[0]
        if gate.shape[0] == 1:
            gate_for_fusion = gate
        else:
            gate_for_fusion = gate
        fused = ego + gate_for_fusion * (context - ego)
        info = {
            "gated_neighbor_fusion": True,
            "fusion_mode": self.fusion_mode,
            "num_valid_neighbors": int(others.shape[0]),
            "gate_mean": float(gate.detach().mean().cpu()) if not self.training else None,
            "gate_max": float(gate.detach().max().cpu()) if not self.training else None,
        }
        return fused, info


class RoCooperAggregator(nn.Module):
    """Multi-scale ROI cross-learning plus explicit gated neighbor fusion."""

    def __init__(
        self,
        aggregator_cfg: Optional[Dict[str, Any]] = None,
        block_prioritizer_cfg: Optional[Dict[str, Any]] = None,
        channels: int = 256,
    ):
        super(RoCooperAggregator, self).__init__()
        self.cfg = aggregator_cfg or {}
        self.block_prioritizer_cfg = block_prioritizer_cfg or {}
        self.channels = int(self.cfg.get("in_channels", channels))

        self.enabled = bool(self.cfg.get("enabled", True))
        self.scales = [int(s) for s in self.cfg.get("scales", [4, 8, 16])]
        self.scales = [s for s in self.scales if s > 0] or [4]
        self.num_rounds = max(1, safe_int(self.cfg.get("num_rounds", 1), 1))
        self.output_fusion_mode = str(self.cfg.get("output_fusion_mode", "gated_residual")).lower()
        self.use_self_attention_after_scatter = bool(
            self.cfg.get("use_self_attention_after_scatter", True)
        )
        self.skip_invalid_others = bool(self.cfg.get("skip_invalid_others", True))
        self.return_debug_stats = bool(self.cfg.get("return_debug_stats", False))

        ca_cfg = self.cfg.get("cross_attention", {}) or {}
        sa_cfg = self.cfg.get("self_attention", {}) or {}
        split_cfg = self.cfg.get("split_attention", {}) or {}
        gate_cfg = self.cfg.get("gated_neighbor_fusion", {}) or {}

        self.scale_selector = ScaleSelector(
            cfg=self.cfg.get("scale_selector", {}) or {},
            scales=self.scales,
        )
        self.block_prioritizer = BlockPrioritizer(
            cfg=self.block_prioritizer_cfg,
            channels=self.channels,
        )
        self.cross_attention = CrossAttentionBlock(
            channels=self.channels,
            num_heads=safe_int(ca_cfg.get("num_heads", 8), 8),
            dropout=safe_float(ca_cfg.get("dropout", 0.0), 0.0),
            mlp_ratio=safe_float(ca_cfg.get("mlp_ratio", 2.0), 2.0),
            residual_scale=safe_float(ca_cfg.get("residual_scale", 0.5), 0.5),
        )
        self.self_refine = FeatureSelfRefine(
            channels=self.channels,
            num_heads=safe_int(sa_cfg.get("num_heads", 8), 8),
            dropout=safe_float(sa_cfg.get("dropout", 0.0), 0.0),
            mlp_ratio=safe_float(sa_cfg.get("mlp_ratio", 2.0), 2.0),
            max_tokens_for_attention=safe_int(sa_cfg.get("max_tokens_for_attention", 4096), 4096),
            fallback_mode=str(sa_cfg.get("fallback_mode", "conv")),
            residual_scale=safe_float(sa_cfg.get("residual_scale", 0.25), 0.25),
        )
        self.split_attention = SplitAttentionFusion(
            channels=self.channels,
            max_num_splits=len(self.scales),
            reduction_factor=safe_int(split_cfg.get("reduction_factor", 4), 4),
        )
        self.gated_neighbor_fusion = GatedNeighborFusion(
            channels=self.channels,
            gate_hidden_ratio=safe_float(gate_cfg.get("gate_hidden_ratio", 0.25), 0.25),
            gate_bias_init=safe_float(gate_cfg.get("gate_bias_init", -1.0), -1.0),
            fusion_mode=str(gate_cfg.get("fusion_mode", "mean")),
            use_channel_gate=bool(gate_cfg.get("use_channel_gate", False)),
        )

    def _validate_inputs(self, ego: torch.Tensor, others: torch.Tensor) -> None:
        if ego.dim() != 3:
            raise ValueError(f"ego must be [C,H,W], got {tuple(ego.shape)}.")
        if others.dim() != 4:
            raise ValueError(f"others must be [L,C,H,W], got {tuple(others.shape)}.")
        if ego.shape[0] != self.channels:
            raise ValueError(f"ego C={ego.shape[0]} but cfg C={self.channels}.")
        if others.shape[0] > 0 and others.shape[1] != self.channels:
            raise ValueError(f"others C={others.shape[1]} but cfg C={self.channels}.")
        if others.shape[0] > 0 and others.shape[-2:] != ego.shape[-2:]:
            raise ValueError(
                f"ego/others spatial mismatch: ego={tuple(ego.shape[-2:])}, others={tuple(others.shape[-2:])}."
            )

    @staticmethod
    def _valid_mask_from_features(others: torch.Tensor) -> torch.Tensor:
        if others.numel() == 0 or others.shape[0] == 0:
            return torch.zeros(0, dtype=torch.bool, device=others.device)
        return others.detach().abs().flatten(1).sum(dim=1) > 1e-8

    def _select_valid_others(
        self,
        others: torch.Tensor,
        others_valid_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if others_valid_mask is None:
            mask = self._valid_mask_from_features(others)
        else:
            mask = others_valid_mask.to(device=others.device).bool().view(-1)
            if mask.numel() != others.shape[0]:
                mask = self._valid_mask_from_features(others)
        valid_others = others[mask] if mask.numel() > 0 else others[:0]
        return valid_others, mask

    def _process_one_scale(
        self,
        ego: torch.Tensor,
        others: torch.Tensor,
        scale: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        scale = int(scale)
        scale_info: Dict[str, Any] = {
            "scale": scale,
            "num_valid_other_cav": int(others.shape[0]),
        }
        if others.numel() == 0 or others.shape[0] == 0:
            scale_info["reason"] = "no_valid_neighbor"
            return ego, others, scale_info

        ego_blocks_batch, ego_meta = partition_feature_map(ego.unsqueeze(0), window_size=scale)
        ego_blocks = ego_blocks_batch[0]
        others_blocks, others_meta = partition_feature_map(others, window_size=scale)

        num_blocks = int(ego_blocks.shape[0])
        scale_info["num_total_blocks"] = num_blocks
        scale_info["tokens_per_block"] = int(ego_blocks.shape[1])

        selected_ego, selected_others, indices, bp_info = self.block_prioritizer.route(
            ego_blocks=ego_blocks,
            others_blocks=others_blocks,
            return_info=True,
        )
        scale_info["block_prioritizer"] = bp_info
        scale_info["num_selected_blocks"] = int(indices.numel())

        updated_ego_selected, updated_others_selected = self.cross_attention(
            ego_selected=selected_ego,
            others_selected=selected_others,
        )

        new_ego_blocks = scatter_topk_blocks(
            original_blocks=ego_blocks,
            updated_blocks=updated_ego_selected,
            indices=indices,
        )
        ego_out = reverse_partition_feature_map(new_ego_blocks.unsqueeze(0), meta=ego_meta)[0]

        new_others_blocks = scatter_topk_blocks(
            original_blocks=others_blocks,
            updated_blocks=updated_others_selected,
            indices=indices,
        )
        others_out = reverse_partition_feature_map(new_others_blocks, meta=others_meta)

        if self.use_self_attention_after_scatter:
            ego_out = self.self_refine(ego_out.unsqueeze(0))[0]
            # Do not refine missing CAVs. This function receives only valid others.
            if others_out.numel() > 0 and others_out.shape[0] > 0:
                others_out = self.self_refine(others_out)
            scale_info["self_refine"] = "applied_to_valid_features"
        else:
            scale_info["self_refine"] = "disabled"

        if self.return_debug_stats:
            scale_info["ego_out_summary"] = tensor_summary(ego_out)
            scale_info["others_out_summary"] = tensor_summary(others_out)
        return ego_out, others_out, scale_info

    def _process_one_round(
        self,
        ego: torch.Tensor,
        others: torch.Tensor,
        round_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        selected_scales, selector_info = self.scale_selector(ego, others)
        round_info: Dict[str, Any] = {
            "round_idx": int(round_idx),
            "scale_selector": selector_info,
            "scale_info": [],
        }
        if others.numel() == 0 or others.shape[0] == 0:
            round_info["reason"] = "no_valid_neighbor"
            return ego, others, round_info

        ego_outputs: List[torch.Tensor] = []
        others_outputs: List[torch.Tensor] = []
        for scale in selected_scales:
            ego_s, others_s, scale_info = self._process_one_scale(ego=ego, others=others, scale=scale)
            ego_outputs.append(ego_s.unsqueeze(0))
            others_outputs.append(others_s)
            round_info["scale_info"].append(scale_info)

        merged_ego = self.split_attention(ego_outputs)[0]
        merged_others = self.split_attention(others_outputs) if len(others_outputs) > 0 else others
        round_info["num_selected_scales"] = len(selected_scales)
        round_info["selected_scales"] = selected_scales
        return merged_ego, merged_others, round_info

    def forward(
        self,
        ego: torch.Tensor,
        others: torch.Tensor,
        batch_idx: Optional[int] = None,
        psm_single: Optional[torch.Tensor] = None,
        data_dict: Optional[Dict[str, Any]] = None,
        others_valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        del psm_single, data_dict
        self._validate_inputs(ego, others)

        valid_others, valid_mask = self._select_valid_others(others, others_valid_mask)
        info: Dict[str, Any] = {
            "aggregator_enabled": self.enabled,
            "batch_idx": None if batch_idx is None else int(batch_idx),
            "num_rounds": self.num_rounds,
            "configured_scales": self.scales,
            "num_other_cav": int(others.shape[0]),
            "num_valid_other_cav": int(valid_mask.sum().item()) if valid_mask.numel() > 0 else 0,
            "output_fusion_mode": self.output_fusion_mode,
            "rounds": [],
        }

        if not self.enabled:
            fused = fallback_fusion_single(ego=ego, others=valid_others, mode="mean")
            info["reason"] = "disabled_fallback_mean_valid_neighbors"
            return fused, others, info

        if valid_others.numel() == 0 or valid_others.shape[0] == 0:
            # Critical: when all neighbors are dropped, do not let attention,
            # Conv/BN, or MLP biases hallucinate neighbor information.
            info["reason"] = "no_valid_neighbor_return_ego"
            return ego, others * 0.0, info

        cur_ego = ego
        cur_valid_others = valid_others
        for round_idx in range(self.num_rounds):
            cur_ego, cur_valid_others, round_info = self._process_one_round(
                ego=cur_ego,
                others=cur_valid_others,
                round_idx=round_idx,
            )
            info["rounds"].append(round_info)

        # Explicit final neighbor fusion makes the model sensitive to valid
        # collaborators; hard-drop should now reduce to ego-only.
        if self.output_fusion_mode in ["gated", "gated_residual", "rocooper_gated"]:
            fused_ego, gate_info = self.gated_neighbor_fusion(cur_ego, cur_valid_others)
            info["final_output"] = "gated_neighbor_residual"
            info["gated_neighbor_fusion"] = gate_info
        elif self.output_fusion_mode in ["mean", "max", "sum", "ego_mean"]:
            mode = "mean" if self.output_fusion_mode == "ego_mean" else self.output_fusion_mode
            fused_ego = fallback_fusion_single(ego=cur_ego, others=cur_valid_others, mode=mode)
            info["final_output"] = "fallback_fusion_after_aggregator"
        else:
            fused_ego = cur_ego
            info["final_output"] = "ego_cross_attention_feature"

        # Scatter valid updated CAVs back to the original other-CAV tensor slots.
        updated_others = others.clone()
        if valid_mask.numel() == others.shape[0] and cur_valid_others.shape[0] == int(valid_mask.sum().item()):
            updated_others[valid_mask] = cur_valid_others
            updated_others[~valid_mask] = 0.0
        else:
            updated_others = cur_valid_others

        if self.return_debug_stats:
            info["fused_ego_summary"] = tensor_summary(fused_ego)
            info["updated_others_summary"] = tensor_summary(updated_others)
        return fused_ego, updated_others, info


RocooperAggregator = RoCooperAggregator
ROCOOPERAggregator = RoCooperAggregator

__all__ = [
    "ScaleSelector",
    "CrossAttentionBlock",
    "FeatureSelfRefine",
    "SplitAttentionFusion",
    "GatedNeighborFusion",
    "RoCooperAggregator",
    "RocooperAggregator",
    "ROCOOPERAggregator",
]
