from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch

from opencood.methods.arce.context.local_confidence import get_cav_confidence
from opencood.methods.arce.runtime.execution_record_builder import (
    build_no_send_system_budget_record,
)
from opencood.methods.arce.reward.reward_pending_builder import (
    build_no_send_pending_reward_item,
)


def execute_no_send_sender(
    *,
    out: torch.Tensor,
    sender_idx: int,
    frame_id: Any,
    ego_id: Any,
    action: Any,
    link_states: Dict[int, str],
    link_profiles: Dict[int, Dict[str, Any]],
    link_budgets: Dict[int, float],
    per_link_budget_bytes: float,
    budget_scope_cfg: str,
    budget_source_cfg: str,
    system_budget_mbps: float,
    tx_window_ms: float,
    total_budget_bytes: float,
    num_collaborators: int,
    ego_conf: float,
    local_cav_confidences: Optional[torch.Tensor],
    decision_context: Any,
    context_builder: Any,
    pending_reward: Any,
    make_no_send_record_fn: Callable[..., Dict[str, Any]],
    profile_for_state_fn: Callable[[str], Dict[str, Any]],
    profile_scalar_fn: Callable[..., float],
    cache_quality_fn: Callable[[Any, Any], float],
) -> Dict[str, Any]:
    """Execute one no-send branch for ARCE-C2MAB.

    This handles the active decision of not transmitting a sender's current
    feature: zeroing the sender feature, recording the no-send link, and
    adding a no-send pending reward item for policy update.
    """
    no_send_profile = link_profiles.get(
        sender_idx,
        profile_for_state_fn(link_states.get(sender_idx, "medium")),
    )
    no_send_latency_ms = profile_scalar_fn(
        no_send_profile.get(
            "delay_ms",
            no_send_profile.get("fixed_delay_ms", 0.0),
        ),
        0.0,
    )
    no_send_cache_q = cache_quality_fn(ego_id, sender_idx)
    no_send_link_budget = float(
        link_budgets.get(sender_idx, per_link_budget_bytes)
    )

    if decision_context is None:
        raise RuntimeError(
            "Missing proposal-time decision context for "
            "no-send sender {}.".format(sender_idx)
        )

    out[sender_idx] = torch.zeros_like(out[sender_idx])

    rec = make_no_send_record_fn(
        out[sender_idx],
        frame_id,
        ego_id,
        sender_idx,
        action,
    )
    rec["channel_state"] = str(link_states.get(sender_idx, "medium"))
    rec["channel_profile"] = dict(no_send_profile)
    rec["link_budget_bytes"] = float(no_send_link_budget)
    rec["system_budget"] = build_no_send_system_budget_record(
        budget_scope=str(budget_scope_cfg),
        budget_source=str(budget_source_cfg),
        system_budget_mbps=float(system_budget_mbps),
        tx_window_ms=float(tx_window_ms),
        total_budget_bytes=float(total_budget_bytes),
        num_collaborators=int(num_collaborators),
        per_link_budget_bytes=float(per_link_budget_bytes),
        link_budgets=link_budgets,
    )

    try:
        if action is not None:
            context = decision_context
            rec["pdf_action"] = action.as_dict()
            rec["context_vector"] = context.vector.tolist()
            rec["context_source"] = (
                "proposal_time_decision_context"
            )
            rec["selected_for_update"] = True
            rec["no_send_update"] = True
            pending_reward.add(
                build_no_send_pending_reward_item(
                    ego_id=ego_id,
                    sender_idx=sender_idx,
                    action=action,
                    context_vector=context.vector,
                    no_send_link_budget=float(no_send_link_budget),
                    channel_state=str(link_states.get(sender_idx, "medium")),
                    cache_quality=float(no_send_cache_q),
                    channel_profile=no_send_profile,
                )
            )
    except Exception as exc:
        rec["no_send_update_error"] = f"{type(exc).__name__}: {exc}"

    return rec


__all__ = ["execute_no_send_sender"]
