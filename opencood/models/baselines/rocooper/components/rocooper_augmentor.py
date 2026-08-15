# -*- coding: utf-8 -*-
"""
RoCooper Augmentor for OpenCOOD.

Save as:
    opencood/models/baselines/rocooper/components/rocooper_augmentor.py

This module implements the Augmentor component in RoCooper.

Paper-level role:
    Augmentor preliminarily restores corrupted received CAV features by:
        1. Spatial-channel semantic enhancement.
        2. Temporal history-guided feature enhancement.

Expected input:
    others:
        Tensor with shape [L, C, H, W], where L is the number of neighbor CAVs.

Expected output:
    enhanced_others, augmentor_info

Used by:
    opencood/models/baselines/rocooper/rocooper_fuse.py
"""

from typing import Any, Dict, Optional, Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.baselines.rocooper.components.rocooper_utils import (
    valid_num_heads,
    flatten_hw,
    unflatten_hw,
    update_history_feature,
    tensor_summary,
)


class SpatialChannelEnhancer(nn.Module):
    """
    Spatial-channel semantic enhancement.

    RoCooper paper idea:
        The received features F_others are projected into query, key, and value.
        A channel attention matrix is computed to recover blurred spatial-channel
        dependencies caused by communication noise.

    Input:
        x: [L, C, H, W]

    Output:
        x_enhanced: [L, C, H, W]
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: Optional[int] = None,
        dropout: float = 0.0,
        init_scale: float = 0.0,
        use_learnable_scale: bool = True,
        use_norm: bool = True,
    ):
        super(SpatialChannelEnhancer, self).__init__()

        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels or channels)
        self.use_norm = bool(use_norm)

        if self.use_norm:
            self.input_norm = nn.BatchNorm2d(self.channels)
        else:
            self.input_norm = nn.Identity()

        self.q_proj = nn.Conv2d(
            self.channels,
            self.hidden_channels,
            kernel_size=1,
            bias=False,
        )
        self.k_proj = nn.Conv2d(
            self.channels,
            self.hidden_channels,
            kernel_size=1,
            bias=False,
        )
        self.v_proj = nn.Conv2d(
            self.channels,
            self.hidden_channels,
            kernel_size=1,
            bias=False,
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(
                self.hidden_channels,
                self.channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.channels),
        )

        self.dropout = nn.Dropout(float(dropout))

        if use_learnable_scale:
            self.gamma = nn.Parameter(torch.tensor(float(init_scale)))
        else:
            self.register_buffer("gamma", torch.tensor(float(init_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                Neighbor CAV features [L, C, H, W].

        Returns:
            Enhanced features with the same shape.
        """
        if x.dim() != 4:
            raise ValueError(
                "SpatialChannelEnhancer expects x with shape [L, C, H, W], "
                f"but got {tuple(x.shape)}."
            )

        if x.numel() == 0 or x.shape[0] == 0:
            return x

        l, _, h, w = x.shape
        spatial_size = max(1, h * w)

        x_norm = self.input_norm(x)

        q = self.q_proj(x_norm).flatten(2)  # [L, C_hidden, HW]
        k = self.k_proj(x_norm).flatten(2)  # [L, C_hidden, HW]
        v = self.v_proj(x_norm).flatten(2)  # [L, C_hidden, HW]

        # Channel attention:
        #   [L, C_hidden, HW] x [L, HW, C_hidden] -> [L, C_hidden, C_hidden]
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / math.sqrt(float(spatial_size))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.bmm(attn, v)  # [L, C_hidden, HW]
        out = out.view(l, self.hidden_channels, h, w)
        out = self.out_proj(out)

        return x + self.gamma * out


class TemporalHistoryEnhancer(nn.Module):
    """
    Temporal history-guided feature enhancement.

    RoCooper paper idea:
        Historical features are used to repair current received features because
        adjacent frames have semantic similarity.

    This module uses current features as query and history features as key/value.

    Input:
        current: [L, C, H, W]
        history: [L, C, H, W]

    Output:
        enhanced current feature [L, C, H, W]
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        init_scale: float = 0.0,
        use_learnable_scale: bool = True,
        max_tokens_for_attention: int = 4096,
        fallback_mode: str = "conv",
    ):
        super(TemporalHistoryEnhancer, self).__init__()

        self.channels = int(channels)
        self.num_heads = valid_num_heads(self.channels, int(num_heads))
        self.max_tokens_for_attention = int(max_tokens_for_attention)
        self.fallback_mode = str(fallback_mode).lower()

        self.q_norm = nn.LayerNorm(self.channels)
        self.kv_norm = nn.LayerNorm(self.channels)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.channels,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )

        self.out_norm = nn.LayerNorm(self.channels)

        self.mlp = nn.Sequential(
            nn.Linear(self.channels, self.channels * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.channels * 4, self.channels),
            nn.Dropout(float(dropout)),
        )

        self.fallback_conv = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                padding=1,
                groups=self.channels,
                bias=False,
            ),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.channels),
        )

        if use_learnable_scale:
            self.gamma = nn.Parameter(torch.tensor(float(init_scale)))
        else:
            self.register_buffer("gamma", torch.tensor(float(init_scale)))

    def forward(
        self,
        current: torch.Tensor,
        history: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            current:
                Current received CAV features [L, C, H, W].

            history:
                Historical CAV features [L, C, H, W].

        Returns:
            Enhanced current features.
        """
        if current.dim() != 4:
            raise ValueError(
                "TemporalHistoryEnhancer expects current with shape [L, C, H, W], "
                f"but got {tuple(current.shape)}."
            )

        if current.numel() == 0 or current.shape[0] == 0:
            return current

        if history is None or history.shape != current.shape:
            return current

        l, c, h, w = current.shape
        tokens = h * w

        if tokens > self.max_tokens_for_attention:
            if self.fallback_mode == "none":
                return current

            repaired = self.fallback_conv(history)
            return current + self.gamma * repaired

        q = flatten_hw(current)  # [L, HW, C]
        kv = flatten_hw(history) # [L, HW, C]

        q_norm = self.q_norm(q)
        kv_norm = self.kv_norm(kv)

        attn_out, _ = self.cross_attn(
            q_norm,
            kv_norm,
            kv_norm,
            need_weights=False,
        )

        seq = q + self.gamma * attn_out
        seq = seq + self.gamma * self.mlp(self.out_norm(seq))

        return unflatten_hw(seq, h, w)


class RoCooperAugmentor(nn.Module):
    """
    RoCooper Augmentor.

    The Augmentor performs preliminary recovery for received neighbor CAV
    features before Aggregator fusion.

    It includes:
        1. SpatialChannelEnhancer.
        2. TemporalHistoryEnhancer.
        3. History update:
               F_h = tau * F_current + (1 - tau) * F_h

    Constructor signature is compatible with rocooper_fuse.py:
        RoCooperAugmentor(cfg, channels)

    Forward signature is also tolerant:
        augmentor(others)
        augmentor(others, batch_idx=..., data_dict=...)
    """

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        channels: int = 256,
    ):
        super(RoCooperAugmentor, self).__init__()

        self.cfg = cfg or {}
        self.channels = int(channels)

        self.enabled = bool(self.cfg.get("enabled", True))

        sc_cfg = self.cfg.get("spatial_channel_attention", {}) or {}
        th_cfg = self.cfg.get("temporal_history", {}) or {}

        self.use_spatial_channel = bool(sc_cfg.get("enabled", True))
        self.use_temporal_history = bool(th_cfg.get("enabled", True))

        self.spatial_channel = SpatialChannelEnhancer(
            channels=self.channels,
            hidden_channels=int(sc_cfg.get("hidden_channels", self.channels)),
            dropout=float(sc_cfg.get("dropout", 0.0)),
            init_scale=float(sc_cfg.get("init_scale", 0.0)),
            use_learnable_scale=bool(sc_cfg.get("use_learnable_scale", True)),
            use_norm=bool(sc_cfg.get("use_norm", True)),
        )

        self.temporal = TemporalHistoryEnhancer(
            channels=self.channels,
            num_heads=int(th_cfg.get("num_heads", 8)),
            dropout=float(th_cfg.get("dropout", 0.0)),
            init_scale=float(th_cfg.get("init_scale", 0.0)),
            use_learnable_scale=bool(th_cfg.get("use_learnable_scale", True)),
            max_tokens_for_attention=int(
                th_cfg.get("max_tokens_for_attention", 4096)
            ),
            fallback_mode=str(th_cfg.get("fallback_mode", "conv")),
        )

        self.tau = float(th_cfg.get("tau", 0.8))
        self.tau = max(0.0, min(1.0, self.tau))

        self.init_with_current = bool(th_cfg.get("init_with_current", True))

        # `history_scope` controls how history is stored.
        #
        # "batch":
        #   one history tensor per batch_idx.
        #
        # "global":
        #   one shared history tensor.
        #
        # For normal shuffled training, history is only an approximation.
        # For sequence inference, call reset_history() before a new sequence.
        self.history_scope = str(th_cfg.get("history_scope", "batch")).lower()

        # Whether to use history in train mode.
        # If dataloader is shuffled and you do not want temporal leakage/noise,
        # set this to false in yaml.
        self.use_history_in_train = bool(th_cfg.get("use_history_in_train", True))

        # Shape mismatch policy:
        #   "reset": reset history when shape changes
        #   "skip": skip temporal enhancement when shape changes
        self.shape_mismatch_policy = str(
            th_cfg.get("shape_mismatch_policy", "reset")
        ).lower()

        self.return_debug_stats = bool(
            self.cfg.get("return_debug_stats", False)
        )

        # History containers.
        self._global_history: Optional[torch.Tensor] = None
        self._history_by_batch: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def reset_history(self, batch_idx: Optional[int] = None) -> None:
        """
        Reset stored temporal history.

        Args:
            batch_idx:
                If None, reset all history.
                If int, reset only the history for that batch index.
        """
        if batch_idx is None:
            self._global_history = None
            self._history_by_batch = {}
            return

        batch_idx = int(batch_idx)
        if batch_idx in self._history_by_batch:
            del self._history_by_batch[batch_idx]

    # ------------------------------------------------------------------
    # Internal history helpers
    # ------------------------------------------------------------------

    def _history_key(self, batch_idx: Optional[int]) -> int:
        if batch_idx is None:
            return 0
        return int(batch_idx)

    def _get_history(self, batch_idx: Optional[int]) -> Optional[torch.Tensor]:
        if self.history_scope == "global":
            return self._global_history

        key = self._history_key(batch_idx)
        return self._history_by_batch.get(key, None)

    def _set_history(
        self,
        history: torch.Tensor,
        batch_idx: Optional[int],
    ) -> None:
        history = history.detach()

        if self.history_scope == "global":
            self._global_history = history
            return

        key = self._history_key(batch_idx)
        self._history_by_batch[key] = history

    def _prepare_history(
        self,
        current: torch.Tensor,
        batch_idx: Optional[int],
    ) -> Tuple[Optional[torch.Tensor], bool, str]:
        """
        Prepare history for temporal enhancement.

        Returns:
            history:
                Prepared history tensor or None.

            initialized:
                Whether history is initialized/reset in this call.

            reason:
                Debug message.
        """
        history = self._get_history(batch_idx)

        if history is None:
            if self.init_with_current:
                self._set_history(current, batch_idx)
                return current.detach(), True, "history_initialized_with_current"
            return None, False, "history_missing"

        if history.shape != current.shape:
            if self.shape_mismatch_policy == "reset":
                if self.init_with_current:
                    self._set_history(current, batch_idx)
                    return current.detach(), True, "history_shape_mismatch_reset"
                self._set_history(torch.zeros_like(current), batch_idx)
                return torch.zeros_like(current), True, "history_shape_mismatch_zero"
            return None, False, "history_shape_mismatch_skip"

        return history, False, "history_ready"

    def _update_history(
        self,
        current: torch.Tensor,
        batch_idx: Optional[int],
    ) -> None:
        old_history = self._get_history(batch_idx)

        new_history = update_history_feature(
            history=old_history,
            current=current,
            tau=self.tau,
            init_with_current=self.init_with_current,
        )

        self._set_history(new_history, batch_idx)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        others: torch.Tensor,
        batch_idx: Optional[int] = None,
        data_dict: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            others:
                Neighbor CAV features [L, C, H, W].

            batch_idx:
                Current scenario index in the batch.
                Used as key for temporal history.

            data_dict:
                Optional OpenCOOD batch dict. Reserved for compatibility.

        Returns:
            enhanced_others:
                [L, C, H, W]

            augmentor_info:
                Debugging/statistics dictionary.
        """
        del data_dict

        if others.dim() != 4:
            raise ValueError(
                "RoCooperAugmentor expects others with shape [L, C, H, W], "
                f"but got {tuple(others.shape)}."
            )

        info: Dict[str, Any] = {
            "augmentor_enabled": self.enabled,
            "batch_idx": None if batch_idx is None else int(batch_idx),
            "num_other_cav": int(others.shape[0]),
            "feature_shape": list(others.shape),
            "use_spatial_channel": self.use_spatial_channel,
            "use_temporal_history": self.use_temporal_history,
        }

        if not self.enabled:
            info["reason"] = "disabled"
            return others, info

        if others.numel() == 0 or others.shape[0] == 0:
            info["reason"] = "no_neighbor_cav"
            return others, info

        if others.shape[1] != self.channels:
            raise ValueError(
                "RoCooperAugmentor channel mismatch. "
                f"Configured channels={self.channels}, "
                f"but input has C={others.shape[1]}."
            )

        x = others

        # --------------------------------------------------------------
        # 1. Spatial-channel semantic enhancement
        # --------------------------------------------------------------
        if self.use_spatial_channel:
            x = self.spatial_channel(x)
            info["spatial_channel_status"] = "applied"
        else:
            info["spatial_channel_status"] = "disabled"

        # --------------------------------------------------------------
        # 2. Temporal history enhancement
        # --------------------------------------------------------------
        temporal_allowed = self.use_temporal_history

        if self.training and not self.use_history_in_train:
            temporal_allowed = False
            info["temporal_history_status"] = "disabled_in_train"

        if temporal_allowed:
            history, initialized, history_reason = self._prepare_history(
                current=x,
                batch_idx=batch_idx,
            )

            info["history_initialized"] = bool(initialized)
            info["history_reason"] = history_reason

            x = self.temporal(x, history)

            # The paper updates historical feature after obtaining the recovered
            # current feature.
            self._update_history(
                current=x,
                batch_idx=batch_idx,
            )

            info["temporal_history_status"] = "applied"
            info["history_tau"] = self.tau
            info["history_scope"] = self.history_scope
        else:
            info.setdefault("temporal_history_status", "disabled")

        if self.return_debug_stats:
            info["input_summary"] = tensor_summary(others)
            info["output_summary"] = tensor_summary(x)

        return x, info


# ----------------------------------------------------------------------
# Import aliases
# ----------------------------------------------------------------------

RocooperAugmentor = RoCooperAugmentor
ROCOOPERAugmentor = RoCooperAugmentor

__all__ = [
    "SpatialChannelEnhancer",
    "TemporalHistoryEnhancer",
    "RoCooperAugmentor",
    "RocooperAugmentor",
    "ROCOOPERAugmentor",
]