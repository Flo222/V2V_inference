from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch


@dataclass
class SpatialImportanceResult:
    priority_map: torch.Tensor
    candidate_mask: torch.Tensor
    priority_source: str
    candidate_source: str
    stats: Dict[str, Any]


class ARCESpatialImportance:
    """Sender-local, method-independent spatial importance scoring."""

    SUPPORTED_METHODS = ("feature_rms",)
    SUPPORTED_NORMALIZATIONS = ("none", "max")

    def __init__(self, cfg: Dict[str, Any] = None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.method = str(
            cfg.get("method", cfg.get("type", "feature_rms"))
        ).strip().lower()
        self.normalization = str(
            cfg.get("normalization", "max")
        ).strip().lower()
        self.zero_epsilon = float(cfg.get("zero_epsilon", 1e-12))
        self.normalization_epsilon = float(
            cfg.get("normalization_epsilon", 1e-12)
        )

        if self.method not in self.SUPPORTED_METHODS:
            raise ValueError(
                "Unsupported ARCE spatial importance method {!r}; expected "
                "one of {}.".format(self.method, self.SUPPORTED_METHODS)
            )
        if self.normalization not in self.SUPPORTED_NORMALIZATIONS:
            raise ValueError(
                "Unsupported ARCE spatial importance normalization {!r}; "
                "expected one of {}.".format(
                    self.normalization,
                    self.SUPPORTED_NORMALIZATIONS,
                )
            )
        if self.zero_epsilon < 0.0:
            raise ValueError(
                "zero_epsilon must be non-negative, got {}.".format(
                    self.zero_epsilon
                )
            )
        if self.normalization_epsilon <= 0.0:
            raise ValueError(
                "normalization_epsilon must be positive, got {}.".format(
                    self.normalization_epsilon
                )
            )

    def compute(self, feature: torch.Tensor) -> SpatialImportanceResult:
        if not self.enabled:
            raise RuntimeError(
                "ARCESpatialImportance.compute called while disabled."
            )
        if not torch.is_tensor(feature) or feature.dim() != 3:
            raise ValueError(
                "ARCE spatial importance expects [C,H,W], got {}.".format(
                    tuple(feature.shape)
                    if torch.is_tensor(feature) else type(feature)
                )
            )

        feature_float = feature.detach().float()
        if self.method == "feature_rms":
            raw_priority = torch.sqrt(
                torch.mean(feature_float * feature_float, dim=0)
            )
        else:
            raise RuntimeError(
                "Unhandled spatial importance method {!r}.".format(
                    self.method
                )
            )

        spatial_max_abs = feature_float.abs().amax(dim=0)
        candidate_mask = spatial_max_abs > float(self.zero_epsilon)

        max_priority = (
            float(raw_priority.max().item())
            if raw_priority.numel() > 0 else 0.0
        )
        if self.normalization == "max" and max_priority > 0.0:
            priority = raw_priority / max(
                max_priority,
                float(self.normalization_epsilon),
            )
        else:
            priority = raw_priority

        active_priority = priority[candidate_mask]
        num_total = int(candidate_mask.numel())
        num_candidates = int(candidate_mask.sum().item())
        stats = {
            "enabled": True,
            "method": str(self.method),
            "normalization": str(self.normalization),
            "zero_epsilon": float(self.zero_epsilon),
            "num_total_units": int(num_total),
            "num_candidate_units": int(num_candidates),
            "candidate_ratio": float(
                num_candidates / max(num_total, 1)
            ),
            "raw_priority_max": float(max_priority),
            "priority_min": (
                float(active_priority.min().item())
                if active_priority.numel() > 0 else None
            ),
            "priority_mean": (
                float(active_priority.mean().item())
                if active_priority.numel() > 0 else None
            ),
            "priority_max": (
                float(active_priority.max().item())
                if active_priority.numel() > 0 else None
            ),
        }
        return SpatialImportanceResult(
            priority_map=priority.unsqueeze(0).to(
                device=feature.device,
                dtype=torch.float32,
            ),
            candidate_mask=candidate_mask.unsqueeze(0).to(
                device=feature.device
            ),
            priority_source="arce_sender_feature_rms",
            candidate_source="arce_nonzero_spatial_support",
            stats=stats,
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "method": str(self.method),
            "normalization": str(self.normalization),
            "zero_epsilon": float(self.zero_epsilon),
            "normalization_epsilon": float(
                self.normalization_epsilon
            ),
        }
