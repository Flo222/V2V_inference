"""Simple AP-proxy reward for C2MAB-ARCE.

Reward:
    r_i = w_i * (lambda_delta * DeltaQ + lambda_abs * Q_collab)
          - lambda_cost * cost_norm
          - optional disabled penalties

No-send should have w_i = 0 and cost = 0, so it does not inherit
positive perception gain from other senders.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
from opencood.methods.arce.policies.quant_quality import get_quant_loss


def quantization_loss(quant_mode: str, cfg=None) -> float:
    q_loss = get_quant_loss(quant_mode, cfg=cfg)
    return float(max(0.0, min(1.0, float(q_loss))))


def normalized_cost(cost_bytes: float, budget_bytes: float) -> float:
    # Cost is a physical ratio. Do not clamp the upper bound; values > 1
    # are useful audit signals for over-reference or over-budget transmission.
    return float(max(0.0, float(cost_bytes) / max(float(budget_bytes), 1.0)))


def _positive_budget_or_none(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    return v if v > 0.0 else None


def normalized_delay(delay_ms: float, stale_max_ms: float = 100.0) -> float:
    return float(max(0.0, min(1.0, float(delay_ms) / max(float(stale_max_ms), 1e-6))))


def c2mab_ap_gain_reward(
    ap_proxy_gain: float,
    contribution_weight: float,
    cost_bytes: float,
    budget_bytes: float,
    delay_ms: float,
    budget_violation: bool,
    quant_mode: str = "fp32",
    lambda_ap: float = 1.0,
    lambda_delta: Optional[float] = None,
    lambda_abs: float = 0.0,
    collab_quality: float = 0.0,
    ego_quality: float = 0.0,
    reward_mode: str = "simple_delta",
    cost_reference_bytes: Optional[float] = None,
    frame_budget_bytes: Optional[float] = None,
    link_budget_bytes: Optional[float] = None,
    cost_norm_mode: str = "reference",
    lambda_cost: float = 0.10,
    lambda_delay: float = 0.0,
    lambda_quant: float = 0.0,
    lambda_violate: float = 0.0,
    stale_max_ms: float = 100.0,
    quant_quality_cfg=None,
) -> Tuple[float, Dict[str, float]]:
    if lambda_delta is None:
        lambda_delta = lambda_ap

    w = float(contribution_weight)
    delta_q = float(ap_proxy_gain)
    collab_q = float(collab_quality)
    ego_q = float(ego_quality)

    link_budget = _positive_budget_or_none(link_budget_bytes)
    if link_budget is None:
        link_budget = _positive_budget_or_none(budget_bytes) or 1.0

    frame_budget = _positive_budget_or_none(frame_budget_bytes)
    if frame_budget is None:
        frame_budget = _positive_budget_or_none(budget_bytes) or link_budget

    reference_budget = _positive_budget_or_none(cost_reference_bytes)
    if reference_budget is None:
        reference_budget = frame_budget

    link_cost_norm = normalized_cost(cost_bytes, link_budget)
    cost_norm_frame = normalized_cost(cost_bytes, frame_budget)
    cost_norm_ref = normalized_cost(cost_bytes, reference_budget)

    mode = str(cost_norm_mode or "reference").strip().lower()
    if mode in ("reference", "ref", "reference_bytes", "cost_norm_ref"):
        cost_norm = cost_norm_ref
    elif mode in ("frame", "frame_budget", "cost_norm_frame"):
        cost_norm = cost_norm_frame
    elif mode in ("link", "link_budget", "link_cost_norm", "legacy"):
        cost_norm = link_cost_norm
    else:
        raise ValueError(f"Unsupported reward cost_norm_mode={cost_norm_mode!r}")

    delay_norm = normalized_delay(delay_ms, stale_max_ms=stale_max_ms)
    q_loss = quantization_loss(quant_mode, cfg=quant_quality_cfg)
    violation = 1.0 if bool(budget_violation) else 0.0

    delta_term = float(lambda_delta) * w * delta_q
    abs_ap_term = float(lambda_abs) * w * collab_q
    perception_term = delta_term + abs_ap_term

    cost_penalty = float(lambda_cost) * cost_norm
    delay_penalty = float(lambda_delay) * delay_norm
    quant_penalty = float(lambda_quant) * q_loss
    violation_penalty = float(lambda_violate) * violation

    reward = (
        perception_term
        - cost_penalty
        - delay_penalty
        - quant_penalty
        - violation_penalty
    )

    info = {
        "reward": float(reward),
        "total_reward": float(reward),
        "reward_type": "simple_ap_proxy",
        "reward_mode": str(reward_mode),

        "ego_quality": float(ego_q),
        "collab_quality": float(collab_q),
        "ap_proxy_gain": float(delta_q),
        "raw_delta_quality": float(delta_q),

        "contribution_weight": float(w),
        "credit_weight": float(w),

        "delta_term": float(delta_term),
        "abs_ap_term": float(abs_ap_term),
        "perception_term": float(perception_term),
        "ap_term": float(delta_term),

        "cost_bytes": float(cost_bytes),
        "actual_tx_bytes": float(cost_bytes),
        "budget_bytes": float(budget_bytes),
        "link_budget_bytes": float(link_budget),
        "frame_budget_bytes": float(frame_budget),
        "cost_reference_bytes": float(reference_budget),
        "cost_norm_mode": str(mode),
        "normalized_cost": float(cost_norm),
        "cost_norm": float(cost_norm),
        "cost_norm_ref": float(cost_norm_ref),
        "cost_norm_frame": float(cost_norm_frame),
        "link_cost_norm": float(link_cost_norm),
        "cost_penalty": float(cost_penalty),

        "delay_ms": float(delay_ms),
        "delay_norm": float(delay_norm),
        "delay_penalty": float(delay_penalty),

        "quant_mode": str(quant_mode).lower(),
        "quant_loss": float(q_loss),
        "quant_penalty": float(quant_penalty),

        "budget_violation": float(violation),
        "violation_penalty": float(violation_penalty),

        "lambda_ap": float(lambda_ap),
        "lambda_delta": float(lambda_delta),
        "lambda_abs": float(lambda_abs),
        "lambda_cost": float(lambda_cost),
        "lambda_delay": float(lambda_delay),
        "lambda_quant": float(lambda_quant),
        "lambda_violate": float(lambda_violate),
    }
    return float(reward), info


__all__ = [
    "quantization_loss",
    "normalized_cost",
    "normalized_delay",
    "c2mab_ap_gain_reward",
]
