from __future__ import annotations

from typing import Any, Dict, Iterable


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


def _is_no_send_record(r: Dict[str, Any]) -> bool:
    action = r.get("action", {}) or {}
    send_value = action.get("send", 1)

    if isinstance(send_value, str):
        send_is_zero = send_value.strip().lower() in (
            "0",
            "false",
            "no",
            "none",
            "null",
        )
    else:
        send_is_zero = bool(send_value == 0 or send_value is False)

    return bool(
        r.get("no_send", False)
        or r.get("is_no_send", False)
        or action.get("is_no_send", False)
        or send_is_zero
    )


def _tx_bytes(r: Dict[str, Any]) -> float:
    return _as_float(
        _get(
            r,
            "budget_consistency",
            "actual_tx_bytes",
            default=_get(
                r,
                "budget_consistency",
                "executor_actual_tx_bytes",
                default=_get(
                    r,
                    "size",
                    "actual_transmitted_bytes",
                    default=r.get(
                        "actual_transmitted_bytes",
                        r.get("transmitted_bytes", r.get("tx_bytes", 0.0)),
                    ),
                ),
            ),
        )
    )


def _rx_bytes(r: Dict[str, Any]) -> float:
    return _as_float(
        _get(
            r,
            "size",
            "actual_received_bytes",
            default=r.get(
                "actual_received_bytes",
                r.get("received_bytes", r.get("rx_bytes", 0.0)),
            ),
        )
    )


def summarize_dc2mab_runtime_records(
    records: Iterable[Dict[str, Any]],
    *,
    budget_source: str,
    budget_scope: str,
    system_budget_mbps: float,
    tx_window_ms: float,
    system_budget_bytes: float,
) -> Dict[str, Any]:
    """Build a lightweight compatibility summary for ARCE-C2MAB records.

    This is not the formal paper/table BW evaluator. Formal AP/BW results
    should use opencood/tools/arce_bw_summary.py and the official run scripts.
    """
    total_tx = 0.0
    total_rx = 0.0
    transmitted = 0
    no_send = 0
    frame_ids = set()
    num_records = 0

    for r in records:
        if not isinstance(r, dict):
            continue

        num_records += 1

        frame_id = r.get("frame_id", None)
        if frame_id is not None:
            frame_ids.add(str(frame_id))

        tx = _tx_bytes(r)
        rx = _rx_bytes(r)
        is_no_send = _is_no_send_record(r)

        total_tx += tx
        total_rx += rx

        if is_no_send:
            no_send += 1
        elif tx > 0.0:
            transmitted += 1

    return {
        "mode": "dc2mab",
        "summary_type": "runtime_compatibility",
        "note": (
            "Use opencood/tools/arce_bw_summary.py and official run "
            "scripts for paper AP/BW tables."
        ),
        "num_records": int(num_records),
        "num_frames_observed": int(len(frame_ids)),
        "num_transmitted_links": int(transmitted),
        "num_no_send_links": int(no_send),
        "total_transmitted_bytes": float(total_tx),
        "total_received_bytes": float(total_rx),
        "total_transmitted_MB": float(total_tx / 1_000_000.0),
        "total_received_MB": float(total_rx / 1_000_000.0),
        "system_budget": {
            "budget_scope": str(budget_scope),
            "budget_source": str(budget_source),
            "system_budget_mbps": float(system_budget_mbps),
            "tx_window_ms": float(tx_window_ms),
            "system_budget_bytes": float(system_budget_bytes),
        },
    }


__all__ = ["summarize_dc2mab_runtime_records"]
