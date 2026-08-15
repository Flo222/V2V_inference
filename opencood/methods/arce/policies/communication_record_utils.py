"""Cache-quality and communication-record utilities for GRACE / C2MAB."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple


def cache_quality(
    last_cache_quality: Dict[Tuple[str, str], float],
    policy_key: Tuple[str, str],
) -> float:
    """Read cached reconstruction / reception quality for one ego-sender link."""
    return float(last_cache_quality.get(policy_key, 0.0))


def update_cache_quality_from_record(
    last_cache_quality: Dict[Tuple[str, str], float],
    policy_key: Tuple[str, str],
    record: Dict[str, Any],
    safe_get_nested_fn: Callable[[Dict[str, Any], List[str]], Any],
) -> None:
    """Update cached link quality from a communication record."""
    q = None

    for path in (
        ["quality", "q_recv"],
        ["recovery", "recovery_ratio"],
        ["partial_reconstruction", "recovery_ratio"],
        ["mean_recovery_ratio"],
    ):
        val = safe_get_nested_fn(record, path)
        if val is not None:
            try:
                q = float(val)
                break
            except Exception:
                pass

    if q is None:
        q = 0.0

    last_cache_quality[policy_key] = max(0.0, min(1.0, q))


def make_no_send_record(
    feature: Any,
    frame_id: Any,
    ego_id: Any,
    sender_id: Any,
    action: Optional[Any] = None,
    reason: str = "not_selected_by_oracle",
) -> Dict[str, Any]:
    """Build a standardized no-send communication record."""
    return {
        "frame_id": frame_id,
        "link_id": repr((ego_id, sender_id)),
        "agent_index": int(sender_id) if isinstance(sender_id, int) else str(sender_id),
        "ego_index": int(ego_id) if isinstance(ego_id, int) else str(ego_id),
        "arce_mode": "dc2mab",
        "applied": False,
        "bypassed": False,
        "no_send": True,
        "reason": reason,
        "action": action.as_dict() if action is not None else {"send": 0},
        "transmitted_bytes": 0.0,
        "received_bytes": 0.0,
        "actual_transmitted_bytes": 0.0,
        "actual_received_bytes": 0.0,
        "input_shape": tuple(int(x) for x in feature.shape),
        "output_shape": tuple(int(x) for x in feature.shape),
        "quality": {
            "q_recv": 0.0,
        },
    }
