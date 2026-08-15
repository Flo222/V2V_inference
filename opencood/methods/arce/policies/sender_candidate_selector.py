"""Sender-side candidate action selector for GRACE / C2MAB."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def build_sender_candidates(
    scored: List[Tuple[Any, Any, float, Dict[str, Any]]],
    sender_topk_actions: int,
    sender_force_quant_coverage: bool = True,
    sender_include_low_cost: bool = True,
):
    """Build a compact candidate set for one sender.

    Input item format:
        (score, action, cost_bytes, cost_info)

    Output item format:
        [score, action, cost_bytes, cost_info, reasons]
    """
    if not scored:
        return []

    scored_by_ucb = sorted(
        scored,
        key=lambda x: float(x[0].ucb),
        reverse=True,
    )

    candidate_map = {}

    def _add_candidate(item, reason: str):
        score, action, cost, cost_info = item
        old_item = candidate_map.get(action.action_id)
        if old_item is None:
            candidate_map[action.action_id] = [
                score,
                action,
                float(cost),
                cost_info,
                {str(reason)},
            ]
        else:
            old_item[4].add(str(reason))

    for item in scored_by_ucb[: max(1, int(sender_topk_actions))]:
        _add_candidate(item, "topk_ucb")

    if bool(sender_force_quant_coverage):
        quant_groups = {}
        for item in scored:
            _, action, _, _ = item
            q = str(getattr(action, "quant_mode", "unknown")).lower()
            quant_groups.setdefault(q, []).append(item)

        for q, items in quant_groups.items():
            best_q = max(items, key=lambda x: float(x[0].ucb))
            _add_candidate(best_q, f"best_ucb_quant_{q}")

    if bool(sender_include_low_cost):
        cheapest_all = min(scored, key=lambda x: float(x[2]))
        _add_candidate(cheapest_all, "cheapest_all")

        quant_groups = {}
        for item in scored:
            _, action, _, _ = item
            q = str(getattr(action, "quant_mode", "unknown")).lower()
            quant_groups.setdefault(q, []).append(item)

        for q, items in quant_groups.items():
            cheapest_q = min(items, key=lambda x: float(x[2]))
            _add_candidate(cheapest_q, f"cheapest_quant_{q}")

    sender_candidates = list(candidate_map.values())
    sender_candidates = sorted(
        sender_candidates,
        key=lambda x: float(x[0].ucb) / max(float(x[2]), 1.0),
        reverse=True,
    )

    return sender_candidates
