"""Final 6D/7D context builder for C2MAB-ARCE.

x_i^t = [B_norm, p_t, d_norm, C_ego, q_cache_i, comp_i_ego, (C_i)].

C_i is the CAV's own local detection confidence, computed before
communication/fusion. AP or AP proxy must not be used as context because
they are action-after feedback signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class PDFContext:
    vector: np.ndarray
    info: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "vector": [float(x) for x in self.vector.tolist()],
            **self.info,
        }


class PDFContextBuilder:
    def __init__(
        self,
        b_max_mbps: float = 27.0,
        stale_max_ms: float = 400.0,
        confidence_threshold: float = 0.3,
        include_cav_confidence: bool = False,
    ):
        self.b_max_mbps = float(b_max_mbps)
        self.stale_max_ms = float(stale_max_ms)
        self.confidence_threshold = float(confidence_threshold)
        self.include_cav_confidence = bool(include_cav_confidence)

    @staticmethod
    def expected_ge_loss(ge: Optional[Dict[str, Any]]) -> float:
        ge = ge or {}
        p_gb = float(ge.get("p_GB", 0.0))
        p_bg = float(ge.get("p_BG", 1.0))
        h = float(ge.get("h", 0.9))
        k = float(ge.get("k", 0.99))
        denom = p_gb + p_bg
        if denom <= 0.0:
            return 0.0
        pi_g = p_bg / denom
        pi_b = p_gb / denom
        return float(pi_g * (1.0 - k) + pi_b * (1.0 - h))

    @staticmethod
    def mean_detection_confidence(pred_scores: Any, threshold: float = 0.3) -> float:
        if pred_scores is None:
            return 0.0
        try:
            import torch
            if torch.is_tensor(pred_scores):
                scores = pred_scores.detach().float().flatten().cpu()
                scores = scores[scores >= float(threshold)]
                return float(scores.mean().item()) if scores.numel() > 0 else 0.0
        except Exception:
            pass
        try:
            values = [float(x) for x in pred_scores]
        except TypeError:
            values = [float(pred_scores)]
        values = [x for x in values if x >= float(threshold)]
        return float(sum(values) / len(values)) if values else 0.0

    def build(
        self,
        channel_profile: Dict[str, Any],
        latency_ms: float,
        ego_confidence: float,
        cache_quality: float,
        complementarity: float = 0.0,
        cav_confidence: Optional[float] = None,
    ) -> PDFContext:
        bandwidth = float(channel_profile.get("bandwidth_mbps", self.b_max_mbps))
        if "loss_rate" in channel_profile:
            p_loss = float(channel_profile.get("loss_rate", 0.0))
        else:
            p_loss = self.expected_ge_loss(channel_profile.get("ge", {}))

        values = [
            bandwidth / max(self.b_max_mbps, 1e-12),
            p_loss,
            float(latency_ms) / max(self.stale_max_ms, 1e-12),
            float(ego_confidence),
            float(cache_quality),
            float(complementarity),
        ]

        if self.include_cav_confidence:
            values.append(float(cav_confidence) if cav_confidence is not None else 0.0)

        vec = np.array(values, dtype=np.float64)
        vec = np.clip(vec, 0.0, 1.0)

        info: Dict[str, Any] = {
            "B_norm": float(vec[0]),
            "p_loss": float(vec[1]),
            "d_norm": float(vec[2]),
            "ego_confidence": float(vec[3]),
            "cache_quality": float(vec[4]),
            "complementarity": float(vec[5]),
            "bandwidth_mbps": bandwidth,
            "latency_ms": float(latency_ms),
        }

        if self.include_cav_confidence:
            info["cav_confidence"] = float(vec[6])

        return PDFContext(vector=vec, info=info)


__all__ = ["PDFContext", "PDFContextBuilder"]
