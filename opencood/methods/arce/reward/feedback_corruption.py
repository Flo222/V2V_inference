"""Explicit corrupted-feedback measure for C2MAB-ARCE.

For the communication setting, we use packet loss rate as the explicit
per-sample corruption measure:

    C_{t,i} = clip(loss_rate, 0, 1)
    w_{t,i} = exp(-alpha * C_{t,i})

This follows the engineering interpretation of CW-C2UCB: unreliable channel
feedback should contribute less to the bandit update.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple


def channel_corruption_measure(
    loss_rate: Optional[float] = None,
    rsrp_dbm: Optional[float] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Return explicit corruption measure C_{t,i} in [0, 1]."""
    if loss_rate is not None:
        c = max(0.0, min(1.0, float(loss_rate)))
        source = "loss_rate"
    elif rsrp_dbm is not None:
        # Lower RSRP means higher corruption. Around -100 dBm is treated as mid.
        quality = 1.0 / (1.0 + math.exp(-(float(rsrp_dbm) + 100.0) / 10.0))
        c = max(0.0, min(1.0, 1.0 - quality))
        source = "rsrp_dbm"
    else:
        c = 0.0
        source = "none"

    info = {
        "feedback_corruption_C": float(c),
        "corruption_source": source,
        "loss_rate": None if loss_rate is None else float(loss_rate),
        "rsrp_dbm": None if rsrp_dbm is None else float(rsrp_dbm),
    }
    return float(c), info


def corruption_feedback_weight(
    corruption_C: float,
    alpha: float = 1.0,
    floor: float = 0.05,
) -> float:
    """Map explicit corruption C to update weight w."""
    c = max(0.0, min(1.0, float(corruption_C)))
    w = math.exp(-float(alpha) * c)
    return float(max(float(floor), min(1.0, w)))


def channel_corruption_weight(
    loss_rate: Optional[float] = None,
    rsrp_dbm: Optional[float] = None,
    alpha: float = 1.0,
    floor: float = 0.05,
) -> Tuple[float, Dict[str, Any]]:
    """Return feedback weight and explicit corruption diagnostics."""
    c, info = channel_corruption_measure(loss_rate=loss_rate, rsrp_dbm=rsrp_dbm)
    w = corruption_feedback_weight(c, alpha=alpha, floor=floor)
    info["feedback_weight"] = float(w)
    info["feedback_weight_formula"] = "exp(-alpha * C)"
    info["feedback_weight_alpha"] = float(alpha)
    info["feedback_weight_floor"] = float(floor)
    return float(w), info


__all__ = [
    "channel_corruption_measure",
    "corruption_feedback_weight",
    "channel_corruption_weight",
]
