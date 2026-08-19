"""Build pending reward items for selected GRACE / C2MAB links."""

from __future__ import annotations

from typing import Any, Callable, Dict


def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _clip01(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        v = float(default)
    return max(0.0, min(1.0, v))


def build_selected_pending_reward_item(
    selected: Any,
    record: Dict[str, Any],
    ego_id: Any,
    sender_idx: int,
    tx_bytes: float,
    total_budget_bytes: float,
    link_delay_ms: float,
    fallback_cache_quality: float,
    reward_tau_stale_ms: float,
    effective_receive_quality_fn: Callable[..., float],
) -> Dict[str, Any]:
    latency_info = {}
    if isinstance(record.get("latency", None), dict):
        latency_info = record.get("latency", {})
    elif isinstance(record.get("channel", None), dict):
        latency_info = _as_dict(record.get("channel", {})).get("latency", {}) or {}

    recovery_info = _as_dict(record.get("recovery", {}))
    quality_info = _as_dict(record.get("quality", {}))
    patch_summary = _as_dict(record.get("patch_summary", {}))

    selected_src = float(
        patch_summary.get(
            "num_selected_source_patches",
            patch_summary.get("num_source_packets", 0.0),
        )
        or 0.0
    )
    missing_by_loss = float(
        patch_summary.get(
            "num_missing_by_loss",
            patch_summary.get("num_lost_by_bernoulli", 0.0),
        )
        or 0.0
    )
    fec_recovered = float(
        patch_summary.get(
            "num_fec_recovered_patches",
            patch_summary.get("num_fec_recovered_packets", 0.0),
        )
        or 0.0
    )

    if selected_src > 0.0:
        q_recv = _clip01(
            1.0 - max(0.0, missing_by_loss - fec_recovered) / selected_src
        )
    else:
        q_recv = float(
            quality_info.get(
                "q_recv",
                recovery_info.get("q_recv", record.get("q_recv", 0.0)),
            )
        )

    delay_ms = float(
        latency_info.get(
            "total_delay_ms",
            latency_info.get("delay_ms", link_delay_ms),
        )
        or 0.0
    )

    q_eff = effective_receive_quality_fn(
        q_recv,
        delay_ms,
        tau_stale_ms=reward_tau_stale_ms,
    )

    reward_budget = float(
        selected.record.get(
            "proposal_budget_bytes",
            selected.record.get("link_budget_bytes", total_budget_bytes),
        )
    )
    link_violation = bool(float(tx_bytes) > reward_budget + 1e-6)

    debug_fec_recovery_ratio = 0.0
    if selected_src > 0.0:
        debug_fec_recovery_ratio = _clip01(fec_recovered / max(selected_src, 1.0))

    cache_quality = _clip01(
        selected.record.get("cache_quality", fallback_cache_quality)
    )

    try:
        cache_enabled = int(getattr(selected.action, "cache_enabled", 0))
    except Exception:
        cache_enabled = 0

    try:
        redundancy_ratio = float(getattr(selected.action, "redundancy_ratio", 0.0))
    except Exception:
        redundancy_ratio = 0.0

    channel_profile = selected.record.get("channel_profile", None)
    if not isinstance(channel_profile, dict):
        channel_profile = _as_dict(record.get("channel", {})).get("profile", None)
    if isinstance(channel_profile, dict):
        channel_profile = dict(channel_profile)
    else:
        channel_profile = None

    item = {
        "ego_id": ego_id,
        "sender_id": int(sender_idx),
        "action_id": selected.action_id,
        "context_vector": selected.context.vector,
        "cost_bytes": float(tx_bytes),
        "link_budget_bytes": float(reward_budget),
        "delay_ms": float(delay_ms),
        "q_recv": float(q_recv),
        "q_eff": float(q_eff),
        "budget_violation": bool(link_violation),
        "quant_mode": str(getattr(selected.action, "quant_mode", "")).lower(),
        "channel_state": str(selected.record.get("channel_state", "medium")).lower(),
        "redundancy_ratio": float(redundancy_ratio),
        "cache_enabled": int(cache_enabled),
        "cache_quality": float(cache_quality),
        "cav_confidence": float(
            selected.record.get(
                "cav_confidence",
                selected.record.get("cav_confidence_value", 0.0),
            )
            or 0.0
        ),
        "cav_confidence_source": str(
            selected.record.get("cav_confidence_source", "unknown")
        ),
        "complementarity": float(
            selected.record.get(
                "complementarity",
                selected.record.get("complementarity_normalized", 0.0),
            )
            or 0.0
        ),
        "debug_fec_recovery_ratio": float(debug_fec_recovery_ratio),
        "complementarity_raw": float(
            selected.record.get(
                "complementarity_raw",
                selected.record.get("complementarity", 0.0),
            )
        ),
        "complementarity_normalized": float(
            selected.record.get(
                "complementarity_normalized",
                selected.record.get("complementarity", 0.0),
            )
        ),
        "contribution_weight": float(
            selected.record.get("estimated_packet_ratio", 1.0)
        ),
    }
    if channel_profile is not None:
        item["channel_profile"] = channel_profile
    return item


def build_no_send_pending_reward_item(
    *,
    ego_id: Any,
    sender_idx: int,
    action: Any,
    context_vector: Any,
    no_send_link_budget: float,
    channel_state: str,
    cache_quality: float,
    channel_profile: Any = None,
) -> Dict[str, Any]:
    if isinstance(channel_profile, dict):
        feedback_profile = dict(channel_profile)
    else:
        feedback_profile = {}
    original_loss_rate = feedback_profile.get(
        "loss_rate",
        feedback_profile.get("plr", None),
    )
    if original_loss_rate is not None:
        feedback_profile["physical_loss_rate"] = float(original_loss_rate)
    feedback_profile["loss_rate"] = 0.0
    feedback_profile["plr"] = 0.0
    feedback_profile["feedback_source"] = "no_send_no_physical_transmission"
    feedback_profile["channel_state"] = str(channel_state).lower()

    return {
        "ego_id": ego_id,
        "sender_id": sender_idx,
        "action_id": action.action_id,
        "context_vector": context_vector,
        "cost_bytes": 0.0,
        "link_budget_bytes": float(no_send_link_budget),
        "delay_ms": 0.0,
        "q_recv": 0.0,
        "q_eff": 0.0,
        "budget_violation": False,
        "quant_mode": str(getattr(action, "quant_mode", "")).lower(),
        "channel_state": str(channel_state).lower(),
        "redundancy_ratio": float(getattr(action, "redundancy_ratio", 0.0)),
        "cache_enabled": int(getattr(action, "cache_enabled", 0)),
        "cache_quality": float(cache_quality),
        "debug_fec_recovery_ratio": 0.0,
        "complementarity_raw": 0.0,
        "complementarity_normalized": 0.0,
        # no-send is an explicit action, but it must not inherit positive
        # AP-proxy gain produced by other senders in the same frame.
        # It still updates the policy with zero communication cost.
        "contribution_weight": 0.0,
        "no_send_update": True,
        "channel_profile": feedback_profile,
    }
