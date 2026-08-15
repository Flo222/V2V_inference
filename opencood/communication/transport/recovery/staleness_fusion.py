"""Delay/staleness-aware temporal fusion utilities for ARCE.

This module is side-effect free and can be called from PartialReconstructor after
packet-level recovery produces current-frame packets and before cache update.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple
import math
import torch


@dataclass
class StalenessFusionInfo:
    q_recv: float
    q_cache: float
    delay_ms: float
    q_delay: float
    q_eff: float
    alpha: float
    tau_stale_ms: float
    beta: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def compute_staleness_quality(q_recv: float, delay_ms: float, tau_stale_ms: float = 300.0) -> float:
    tau = max(float(tau_stale_ms), 1e-6)
    return float(q_recv) * math.exp(-max(float(delay_ms), 0.0) / tau)


def compute_alpha(q_eff: float, q_cache: float, beta: float = 5.0) -> float:
    z = float(beta) * (float(q_eff) - float(q_cache))
    # stable sigmoid
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def blend_current_cache(current: torch.Tensor, cache: Optional[torch.Tensor], q_recv: float, q_cache: float, delay_ms: float, beta: float = 5.0, tau_stale_ms: float = 300.0) -> Tuple[torch.Tensor, StalenessFusionInfo]:
    q_delay = math.exp(-max(float(delay_ms), 0.0) / max(float(tau_stale_ms), 1e-6))
    q_eff = float(q_recv) * q_delay
    alpha = compute_alpha(q_eff, q_cache, beta=beta)
    if cache is None:
        out = current
        alpha_used = 1.0
    else:
        if cache.shape != current.shape:
            raise ValueError(f'cache shape {tuple(cache.shape)} != current shape {tuple(current.shape)}')
        out = alpha * current + (1.0 - alpha) * cache.to(current.device, dtype=current.dtype)
        alpha_used = alpha
    info = StalenessFusionInfo(
        q_recv=float(q_recv), q_cache=float(q_cache), delay_ms=float(delay_ms),
        q_delay=float(q_delay), q_eff=float(q_eff), alpha=float(alpha_used),
        tau_stale_ms=float(tau_stale_ms), beta=float(beta),
    )
    return out, info


def quality_from_masks(num_available: int, num_total: int) -> float:
    if int(num_total) <= 0:
        return 0.0
    return float(num_available) / float(num_total)
