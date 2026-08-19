from __future__ import annotations

from typing import Any, Dict



def _oracle_superarm_light_summary(oracle_result: Dict[str, Any]) -> Dict[str, Any]:
    selected = oracle_result.get("selected", []) or []
    return {
        "used_budget_bytes": float(oracle_result.get("used_budget_bytes", 0.0)),
        "remaining_budget_bytes": float(oracle_result.get("remaining_budget_bytes", 0.0)),
        "num_candidates": int(oracle_result.get("num_candidates", 0) or 0),
        "num_selected": int(oracle_result.get("num_selected", len(selected)) or 0),
        "selected_sender_ids": [str(getattr(x, "sender_id", "")) for x in selected],
        "selected_action_ids": [str(getattr(x, "action_id", "")) for x in selected],
    }


def build_dc2mab_superarm_record(
    *,
    frame_id: Any,
    batch_idx: int,
    ego_id: Any,
    total_budget_bytes: float,
    budget_scope: str,
    budget_source: str,
    system_budget_mbps: float,
    tx_window_ms: float,
    num_collaborators: int,
    per_link_budget_bytes: float,
    link_budgets: Dict[int, float],
    link_states: Dict[int, str],
    used_cost: float,
    selected_by_sender: Dict[int, Any],
    oracle_result: Dict[str, Any],
    packet_size_bytes: int,
    debug_records: bool = False,
) -> Dict[str, Any]:
    """Build the frame-level C2MAB superarm record.

    Default output is lightweight and suitable for full experiments. Set
    debug_records=True to retain detailed oracle internals.
    """
    record = {
        "frame_id": frame_id,
        "batch_idx": int(batch_idx),
        "ego_id": str(ego_id),
        "dc2mab_superarm": {
            "budget_bytes": float(total_budget_bytes),
            "budget_scope": str(budget_scope),
            "budget_source": str(budget_source),
            "system_budget_mbps": float(system_budget_mbps),
            "tx_window_ms": float(tx_window_ms),
            "num_collaborators": int(num_collaborators),
            "per_link_budget_bytes": float(per_link_budget_bytes),
            "link_budgets": {str(k): float(v) for k, v in link_budgets.items()},
            "link_states": {str(k): str(v) for k, v in link_states.items()},
            "used_budget_bytes": float(used_cost),
            "selected_sender_ids": [str(x) for x in selected_by_sender.keys()],
            "selected_action_ids": [
                proposal.action_id for proposal in selected_by_sender.values()
            ],
            "num_selected": len(selected_by_sender),
            "oracle": (
                {
                    key: value
                    for key, value in oracle_result.items()
                    if key not in ("selected",)
                }
                if bool(debug_records)
                else _oracle_superarm_light_summary(oracle_result)
            ),
            "packetization": {
                "mode": "byte_stream",
                "packet_size_bytes": int(packet_size_bytes),
                "quantize_first": True,
            },
        },
    }
    return record
