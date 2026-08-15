from __future__ import annotations

import copy
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from opencood.methods.arce.c2mab_local_confidence import get_cav_confidence
from opencood.methods.arce.policies.payload_context import build_payload_pair_context, build_detection_confidence_pair_context
from opencood.methods.arce.policies.payload_transport import is_payload_native_transport
from opencood.methods.arce.policies.action_space import PDFARCEAction
from opencood.methods.arce.policies.ego_greedy_oracle import CAVProposal
from opencood.methods.arce.policies.sender_candidate_selector import build_sender_candidates


def build_c2mab_proposals(
    *,
    features_shape: Sequence[int],
    features: Any,
    collaborator_indices: Sequence[int],
    ego_id: Any,
    ego_index: int,
    ego_confidence: float,
    ego_confidence_source: str,
    total_budget_bytes: float,
    per_link_budget_bytes: float,
    num_collaborators: int,
    budget_scope: str,
    budget_source: str,
    use_channel_profile_budget: bool,
    link_states: Dict[int, str],
    link_profiles: Dict[int, Dict[str, Any]],
    link_budgets: Dict[int, float],
    actions: Sequence[PDFARCEAction],
    no_send_action: Optional[PDFARCEAction],
    arce_cfg: Dict[str, Any],
    context_builder: Any,
    local_cav_confidences: Any,
    local_cav_confidence_maps: Any,
    packet_size_bytes: int,
    sender_topk_actions: int,
    sender_force_quant_coverage: bool,
    sender_include_low_cost: bool,
    profile_for_state_fn: Callable[[str], Dict[str, Any]],
    profile_scalar_fn: Callable[[Any, float], float],
    cache_quality_fn: Callable[[Any, Any], Any],
    estimate_cost_fn: Callable[..., Dict[str, Any]],
    get_policy_fn: Callable[[Any, Any], Any],
) -> Tuple[
    List[CAVProposal],
    Dict[int, PDFARCEAction],
    Dict[int, Any],
]:
    proposals: List[CAVProposal] = []
    no_send_candidates: Dict[int, PDFARCEAction] = {}
    decision_contexts: Dict[int, Any] = {}

    for sender_idx in collaborator_indices:
        state_name = link_states.get(sender_idx, "medium")
        if state_name == "ego_or_padding":
            continue

        profile = link_profiles.get(sender_idx, profile_for_state_fn(state_name))
        link_budget_bytes = float(link_budgets.get(sender_idx, per_link_budget_bytes))

        if use_channel_profile_budget and budget_scope == "global_sum_link":
            proposal_budget_bytes = float(total_budget_bytes)
        else:
            proposal_budget_bytes = float(link_budget_bytes)

        latency_ms = profile_scalar_fn(
            profile.get("delay_ms", profile.get("fixed_delay_ms", 50.0)),
            50.0,
        )
        cache_state_raw = cache_quality_fn(
            ego_id,
            sender_idx,
        )
        if isinstance(cache_state_raw, dict):
            cache_state = copy.deepcopy(cache_state_raw)
            cache_q = float(
                cache_state.get(
                    "cache_valid_unit_ratio",
                    0.0,
                )
            )
        else:
            cache_q = float(cache_state_raw)
            cache_state = {
                "cache_available": bool(cache_q > 0.0),
                "cache_status": "legacy_history_quality",
                "cache_valid_unit_ratio": float(cache_q),
                "cache_num_valid_units": 0,
                "cache_num_total_units": 0,
                "cache_age_frames": None,
                "cache_age_norm": 0.0,
                "cache_context_source":
                    "legacy_last_receive_quality",
            }

        if not math.isfinite(cache_q):
            raise ValueError(
                "Decision Cache quality must be finite, got {}".format(
                    cache_q
                )
            )

        cache_q = max(0.0, min(1.0, cache_q))

        payload_context_cfg = (
            (arce_cfg.get("context", {}) or {}).get("payload_context", {}) or {}
        )
        use_payload_context = bool(
            payload_context_cfg.get(
                "enabled",
                is_payload_native_transport(arce_cfg),
            )
        )

        payload_ctx = build_payload_pair_context(
            features,
            ego_index=int(ego_index),
            sender_idx=int(sender_idx),
        )
        detection_ctx = build_detection_confidence_pair_context(
            local_cav_confidence_maps,
            ego_index=int(ego_index),
            sender_idx=int(sender_idx),
        )
        if bool(detection_ctx.get("valid", False)):
            comp_i_ego = float(detection_ctx.get("complementarity", 0.0))
            comp_source = str(
                detection_ctx.get(
                    "complementarity_source",
                    "local_detection_soft_complementarity",
                )
            )
            comp_stats = dict(detection_ctx)
        else:
            comp_i_ego = float(payload_ctx.get("complementarity", 0.0))
            comp_source = str(
                payload_ctx.get(
                    "complementarity_source",
                    "payload_energy_cosine_distance",
                )
            )
            comp_stats = dict(payload_ctx)
        sender_mask = None
        ego_mask = None

        comp_raw = float(comp_i_ego)
        comp_norm_mode = str(arce_cfg.get("complementarity_norm", "raw_clip")).lower()
        if comp_norm_mode in ("exp", "exp_tau", "legacy_exp"):
            comp_tau = float(arce_cfg.get("complementarity_tau", 5e-5))
            comp_tau = max(comp_tau, 1e-12)
            comp_norm = 1.0 - math.exp(-max(0.0, comp_raw) / comp_tau)
        else:
            comp_norm = max(0.0, min(1.0, float(comp_raw)))
        comp_norm = max(0.0, min(1.0, float(comp_norm)))

        local_sender_conf = get_cav_confidence(
            local_cav_confidences,
            int(sender_idx),
            default=None,
        )
        if local_sender_conf is not None:
            cav_confidence_value = float(local_sender_conf)
            cav_confidence_source = "local_detection_confidence_summary"
        else:
            payload_sender_conf = float(payload_ctx.get("sender_confidence", 0.0))
            cav_confidence_value = float(payload_sender_conf)
            cav_confidence_source = str(
                payload_ctx.get(
                    "sender_confidence_source",
                    "payload_sender_energy_normalized",
                )
            )

        context = context_builder.build(
            channel_profile=profile,
            latency_ms=latency_ms,
            ego_confidence=ego_confidence,
            cache_quality=cache_q,
            complementarity=comp_norm,
            cav_confidence=float(cav_confidence_value),
        )
        decision_contexts[int(sender_idx)] = context

        feasible = []
        for action in actions:
            if getattr(action, "is_no_send", False):
                continue

            cost_info = estimate_cost_fn(
                feature_shape=features_shape,
                action=action,
                budget_bytes=proposal_budget_bytes,
                message_mask=None,
            )
            if not bool(cost_info["feasible"]):
                continue

            feasible.append(
                (
                    action,
                    float(cost_info["estimated_transmitted_bytes"]),
                    cost_info,
                )
            )

        if not feasible:
            no_send_candidates[sender_idx] = no_send_action
            continue

        policy = get_policy_fn(ego_id, sender_idx)
        scored = []
        for action, cost_bytes, cost_info in feasible:
            score = policy.score(action.action_id, context.vector)
            scored.append((score, action, float(cost_bytes), cost_info))

        sender_candidates = build_sender_candidates(
            scored=scored,
            sender_topk_actions=sender_topk_actions,
            sender_force_quant_coverage=sender_force_quant_coverage,
            sender_include_low_cost=sender_include_low_cost,
        )

        for local_rank, (
            score,
            cand_action,
            cand_cost,
            cand_cost_info,
            reasons,
        ) in enumerate(sender_candidates):
            proposals.append(
                CAVProposal(
                    ego_id=ego_id,
                    sender_id=sender_idx,
                    action=cand_action,
                    action_id=cand_action.action_id,
                    context=context,
                    ucb=score.ucb,
                    mean=score.mean,
                    bonus=score.bonus,
                    cost_bytes=float(cand_cost),
                    record={
                        "channel_state": state_name,
                        "ego_confidence": float(ego_confidence),
                        "ego_confidence_source": str(ego_confidence_source),
                        "cav_confidence": float(cav_confidence_value),
                        "cav_confidence_source": str(cav_confidence_source),
                        "decision_cache_available": bool(
                            cache_state.get(
                                "cache_available",
                                False,
                            )
                        ),
                        "decision_cache_status": str(
                            cache_state.get(
                                "cache_status",
                                "unknown",
                            )
                        ),
                        "decision_cache_valid_unit_ratio": float(
                            cache_q
                        ),
                        "decision_cache_num_valid_units": int(
                            cache_state.get(
                                "cache_num_valid_units",
                                0,
                            )
                        ),
                        "decision_cache_num_total_units": int(
                            cache_state.get(
                                "cache_num_total_units",
                                0,
                            )
                        ),
                        "decision_cache_age_frames": cache_state.get(
                            "cache_age_frames",
                            None,
                        ),
                        "decision_cache_age_norm": float(
                            cache_state.get(
                                "cache_age_norm",
                                0.0,
                            )
                        ),
                        "decision_cache_context_source": str(
                            cache_state.get(
                                "cache_context_source",
                                "unknown",
                            )
                        ),
                        "complementarity": float(comp_i_ego),
                        "complementarity_source": str(comp_source),
                        "complementarity_stats": copy.deepcopy(comp_stats),
                        "payload_context": copy.deepcopy(payload_ctx),
                        "payload_context_enabled": bool(use_payload_context),
                        "payload_context_source": str(
                            payload_ctx.get("payload_context_source", "none")
                        ),
                        "channel_profile": profile,
                        "link_budget_bytes": float(link_budget_bytes),
                        "proposal_budget_bytes": float(proposal_budget_bytes),
                        "per_link_budget_bytes": float(per_link_budget_bytes),
                        "system_budget_bytes": float(total_budget_bytes),
                        "num_collaborators": int(num_collaborators),
                        "budget_scope": str(budget_scope),
                        "budget_source": str(budget_source),
                        "proposal_cost_model": str(
                            cand_cost_info.get("cost_model", "byte_stream_quantize_first_with_fec")
                        ),
                        "compact_estimator_enabled": bool(
                            cand_cost_info.get("compact_estimator_enabled", False)
                        ),
                        "compact_estimated_num_tokens": cand_cost_info.get(
                            "compact_estimated_num_tokens", None
                        ),
                        "compact_estimated_mask_ratio": cand_cost_info.get(
                            "compact_estimated_mask_ratio", None
                        ),
                        "compact_estimator_budget_policy": cand_cost_info.get(
                            "compact_estimator_budget_policy", None
                        ),
                        "compact_estimator_predicted_allocated_budget_bytes": cand_cost_info.get(
                            "compact_estimator_predicted_allocated_budget_bytes", None
                        ),
                        "compact_estimator": dict(
                            cand_cost_info.get("compact_estimator", {}) or {}
                        ),
                        "compact_estimator_first_pass": dict(
                            cand_cost_info.get("compact_estimator_first_pass", {}) or {}
                        ),
                        "estimated_tx_bytes": float(cand_cost),
                        "estimated_source_bytes": float(cand_cost_info["source_bytes"]),
                        "estimated_parity_bytes": float(
                            cand_cost_info["parity_packets"] * packet_size_bytes
                        ),
                        "estimated_metadata_bytes": float(cand_cost_info["metadata_bytes"]),
                        "estimated_encoded_bytes": float(cand_cost_info["encoded_bytes"]),
                        "estimated_packet_ratio": float(
                            cand_cost_info["effective_packet_ratio"]
                        ),
                        "num_source_packets": int(cand_cost_info["source_packets"]),
                        "num_parity_packets": int(cand_cost_info["parity_packets"]),
                        "num_encoded_packets": int(cand_cost_info["encoded_packets"]),
                        "max_tx_packets_under_budget": int(
                            cand_cost_info["max_tx_packets_under_budget"]
                        ),
                        "fec_type": str(cand_cost_info["fec_type"]),
                        "rho": float(cand_cost_info["rho"]),
                        "packet_size_bytes": int(packet_size_bytes),
                        "bandwidth_selection": copy.deepcopy(cand_cost_info),
                        "num_feasible_actions": int(len(feasible)),
                        "num_sender_candidate_actions": int(len(sender_candidates)),
                        "complementarity_raw": float(comp_i_ego),
                        "complementarity_normalized": float(comp_norm),
                        "sender_candidate_rank": int(local_rank),
                        "sender_candidate_reasons": sorted(str(x) for x in reasons),
                        "sender_topk_actions": int(sender_topk_actions),
                        "sender_force_quant_coverage": bool(sender_force_quant_coverage),
                        "sender_include_low_cost": bool(sender_include_low_cost),
                    },
                    mask=None,
                    ego_mask=None,
                    complementarity=float(comp_i_ego),
                )
            )

    return proposals, no_send_candidates, decision_contexts
