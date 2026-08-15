"""Reward-runtime audit export shared by GRACE evaluation entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable


def _stats(values: Iterable[Any]) -> Dict[str, Any]:
    vals = []
    for value in values:
        try:
            vals.append(float(value))
        except Exception:
            pass

    if not vals:
        return {"n": 0}

    vals.sort()

    def percentile(q: float) -> float:
        if len(vals) == 1:
            return float(vals[0])
        index = int(round(float(q) * float(len(vals) - 1)))
        return float(vals[max(0, min(len(vals) - 1, index))])

    return {
        "n": int(len(vals)),
        "min": float(vals[0]),
        "p10": percentile(0.10),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "max": float(vals[-1]),
        "mean": float(sum(vals) / len(vals)),
        "pos": int(sum(1 for value in vals if value > 0.0)),
        "neg": int(sum(1 for value in vals if value < 0.0)),
        "zero": int(sum(1 for value in vals if value == 0.0)),
    }


def _compact_reward_update(update_index: int, update: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "update_index": int(update_index),
        "collab_confidence": update.get("collab_confidence"),
        "ego_confidence": update.get("ego_confidence"),
        "delta_confidence": update.get("delta_confidence"),
        "num_updated": update.get("num_updated"),
        "num_send_updated": update.get("num_send_updated"),
        "num_no_send_updated": update.get("num_no_send_updated"),
        "mean_reward": update.get("mean_reward"),
        "reward_delta_source": update.get("reward_delta_source"),
        "delta_confidence_override": update.get("delta_confidence_override"),
        "ap_proxy_delta": update.get("ap_proxy_delta"),
        "reward_term_summary": update.get("reward_term_summary"),
    }

    delta_debug = update.get("delta_ap_proxy_reward")
    if isinstance(delta_debug, dict):
        row["delta_ap_proxy_used"] = delta_debug.get("delta_ap_proxy_used")
        row["delta_ap_proxy_source"] = delta_debug.get("source")
        row["delta_ap_hat"] = delta_debug.get("delta_ap_hat")

    ap_debug = update.get("ap_proxy_reward")
    if isinstance(ap_debug, dict):
        row["ap_proxy_used"] = ap_debug.get("ap_proxy_used")
        row["collab_confidence_source"] = ap_debug.get(
            "collab_confidence_source"
        )

    ego_debug = update.get("ego_ap_proxy_reward")
    if isinstance(ego_debug, dict):
        row["ego_ap_proxy_used"] = ego_debug.get("ap_proxy_used")
        row["ego_confidence_source"] = ego_debug.get(
            "collab_confidence_source"
        )

    return row


def save_reward_runtime_audit(
    records: Iterable[Dict[str, Any]],
    out_dir: Path,
    frame_count: int,
) -> Dict[str, Any]:
    """Write compact per-frame reward updates and aggregate statistics."""
    rows = []
    for record in records:
        if not isinstance(record, dict):
            continue
        update = record.get("reward_update")
        if isinstance(update, dict):
            rows.append(_compact_reward_update(len(rows), update))

    summary = {
        "frame_count": int(frame_count),
        "reward_update_count": int(len(rows)),
        "delta_confidence": _stats(
            row.get("delta_confidence") for row in rows
        ),
        "mean_reward": _stats(row.get("mean_reward") for row in rows),
        "num_updated": _stats(row.get("num_updated") for row in rows),
        "num_send_updated": _stats(
            row.get("num_send_updated") for row in rows
        ),
        "num_no_send_updated": _stats(
            row.get("num_no_send_updated") for row in rows
        ),
    }
    audit = {
        "frame_count": int(frame_count),
        "reward_update_count": int(len(rows)),
        "summary": summary,
        "rows": rows,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "reward_runtime_audit.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(audit, file, indent=2, ensure_ascii=False)

    return {
        "reward_runtime_audit_json": str(path),
        "reward_update_count": int(len(rows)),
    }


__all__ = ["save_reward_runtime_audit"]
