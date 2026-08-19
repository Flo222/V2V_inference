"""Final diversity-aware greedy knapsack oracle for C2MAB-ARCE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass
class CAVProposal:
    ego_id: Any
    sender_id: Any
    action: Any
    action_id: str
    context: Any
    ucb: float
    mean: float
    bonus: float
    cost_bytes: float
    record: Dict[str, Any]
    mask: Optional[Any] = None
    ego_mask: Optional[Any] = None
    complementarity: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        action_dict = self.action.as_dict() if hasattr(self.action, "as_dict") else str(self.action)
        ctx_dict = self.context.as_dict() if hasattr(self.context, "as_dict") else self.context
        return {
            "ego_id": str(self.ego_id),
            "sender_id": str(self.sender_id),
            "action_id": self.action_id,
            "action": action_dict,
            "context": ctx_dict,
            "ucb": float(self.ucb),
            "mean": float(self.mean),
            "bonus": float(self.bonus),
            "cost_bytes": float(self.cost_bytes),
            "complementarity": float(self.complementarity),
            "record": self.record,
        }


def _to_bool_mask(mask: Any):
    if mask is None or torch is None:
        return None
    if not torch.is_tensor(mask):
        try:
            mask = torch.as_tensor(mask)
        except Exception:
            return None
    return mask.detach().bool().flatten().cpu()


def _overlap_ratio(mask_i: Any, mask_selected: Any) -> float:
    mi = _to_bool_mask(mask_i)
    ms = _to_bool_mask(mask_selected)
    if mi is None or ms is None or mi.numel() == 0 or ms.numel() == 0:
        return 0.0
    n = min(mi.numel(), ms.numel())
    mi = mi[:n]
    ms = ms[:n]
    denom = float(mi.sum().item())
    if denom <= 0.0:
        return 0.0
    return float((mi & ms).sum().item() / max(denom, 1.0))


def _marginal_coverage_ratio(mask_i: Any, mask_selected: Any) -> float:
    """Fraction of CAV i's valid mask not yet covered by ego/selected CAVs.

    This is the dynamic super-arm marginal utility used by the oracle.
    Static CAV-ego complementarity stays in the context and is learned by
    D-LinUCB, rather than being manually multiplied into oracle gain.
    """
    overlap = _overlap_ratio(mask_i, mask_selected)
    return float(max(0.0, min(1.0, 1.0 - overlap)))


def _union_mask(mask_a: Any, mask_b: Any):
    ma = _to_bool_mask(mask_a)
    mb = _to_bool_mask(mask_b)
    if ma is None:
        return mb
    if mb is None:
        return ma
    n = min(ma.numel(), mb.numel())
    out = ma.clone()
    out[:n] = ma[:n] | mb[:n]
    return out


class EgoGreedyKnapsackOracle:
    def __init__(
        self,
        eps_cost: float = 1.0,
        diversity_aware: bool = True,
        min_marginal_coverage: float = 0.001,
        oracle_cost_lambda: float = 0.12,
        explore_warmup_pulls_per_quant: int = 6,
        explore_warmup_pulls_per_rho: int = 4,
        explore_warmup_pulls_per_cache: int = 4,
        explore_bonus: float = 1.0,
    ):
        self.eps_cost = float(eps_cost)
        self.diversity_aware = bool(diversity_aware)
        self.min_marginal_coverage = max(0.0, min(1.0, float(min_marginal_coverage)))

        # Static complementarity is learned through the D-LinUCB context,
        # while dynamic redundancy is handled by marginal coverage.
        self.oracle_cost_lambda = max(0.0, float(oracle_cost_lambda))

        # Bandit warm-up exploration.
        # This is NOT a hand-crafted bad/medium/good -> quant rule.
        # It only guarantees that each quantization arm gets several online
        # reward samples; after warm-up, selection is driven by learned UCB.
        self.explore_warmup_pulls_per_quant = max(0, int(explore_warmup_pulls_per_quant))
        self.explore_bonus = max(0.0, float(explore_bonus))
        self.explore_warmup_pulls_per_rho = max(0, int(explore_warmup_pulls_per_rho))
        self.explore_warmup_pulls_per_cache = max(0, int(explore_warmup_pulls_per_cache))
        self._quant_select_counts = {
            "bad": {"fp16": 0, "int8": 0, "int4": 0},
            "medium": {"fp16": 0, "int8": 0, "int4": 0},
            "good": {"fp16": 0, "int8": 0, "int4": 0},
            "unknown": {"fp16": 0, "int8": 0, "int4": 0},
        }
        self._rho_select_counts = {
            "bad": {"rho0": 0, "rho0p10": 0, "rho0p25": 0, "rho0p60": 0},
            "medium": {"rho0": 0, "rho0p10": 0, "rho0p25": 0, "rho0p60": 0},
            "good": {"rho0": 0, "rho0p10": 0, "rho0p25": 0, "rho0p60": 0},
            "unknown": {"rho0": 0, "rho0p10": 0, "rho0p25": 0, "rho0p60": 0},
        }
        self._cache_select_counts = {
            "bad": {"cache0": 0, "cache1": 0},
            "medium": {"cache0": 0, "cache1": 0},
            "good": {"cache0": 0, "cache1": 0},
            "unknown": {"cache0": 0, "cache1": 0},
        }

    def select(self, proposals: Sequence[CAVProposal], budget_bytes: float) -> Dict[str, Any]:
        """Greedy knapsack over sender-action proposals; one action per sender at most."""
        candidates: List[CAVProposal] = []
        for p in proposals:
            try:
                cost = float(p.cost_bytes)
            except Exception:
                continue
            if cost <= 0.0:
                continue
            candidates.append(p)

        remaining = float(budget_bytes)
        selected: List[CAVProposal] = []
        selected_sender_ids = set()

        # Step 12.4:
        # Dynamic marginal coverage must be measured against ego coverage plus
        # already selected CAVs. Otherwise the first selected CAV always gets
        # marginal_coverage ~= 1 and CAV-ego redundancy is not penalized.
        selected_union_mask = None
        for _p in candidates:
            selected_union_mask = _union_mask(selected_union_mask, getattr(_p, "ego_mask", None))

        ego_mask_initialized = selected_union_mask is not None
        ranked_history: List[Dict[str, Any]] = []
        first_round_candidates: List[Dict[str, Any]] = []

        while True:
            best = None
            best_info = None
            round_candidates: List[Dict[str, Any]] = []
            for p in candidates:
                sid = str(p.sender_id)
                if sid in selected_sender_ids:
                    continue
                if float(p.cost_bytes) > remaining:
                    continue

                comp = float(getattr(p, "complementarity", 0.0))
                if self.diversity_aware:
                    red = _overlap_ratio(p.mask, selected_union_mask)
                    marginal_coverage = _marginal_coverage_ratio(p.mask, selected_union_mask)
                    # Step 12:
                    # Oracle uses dynamic marginal coverage only. Static
                    # CAV-ego complementarity is already part of the context
                    # and should be learned by D-LinUCB, not manually multiplied
                    # here again.
                    gain = max(float(p.ucb), 0.0) * float(marginal_coverage)
                else:
                    red = 0.0
                    marginal_coverage = 1.0
                    gain = max(float(p.ucb), 0.0)

                if self.diversity_aware and float(marginal_coverage) < float(self.min_marginal_coverage):
                    continue

                action_obj = getattr(p, "action", None)
                q = str(getattr(action_obj, "quant_mode", "")).lower()
                if not q:
                    aid = str(getattr(p, "action_id", "")).lower()
                    for _q in ("fp32", "fp16", "int8", "int4"):
                        if _q in aid:
                            q = _q
                            break

                # Learned-UCB oracle.
                # Do NOT manually force bad/medium/good to specific quant modes here.
                # The proposal score is dominated by the learned bandit UCB.
                # A light cost penalty is kept only to respect communication efficiency.
                record = getattr(p, "record", {}) or {}
                state_name = str(record.get("channel_state", "medium")).lower()

                total_budget = max(float(budget_bytes), self.eps_cost)
                cost_norm = min(max(float(p.cost_bytes) / total_budget, 0.0), 1.0)

                # Small cost penalty: avoids wasting budget but does not hard-code quant choices.
                oracle_cost_lambda = float(self.oracle_cost_lambda)
                base_ratio = float(gain) / (1.0 + oracle_cost_lambda * cost_norm)

                # Warm-up exploration bonus for under-sampled quant arms.
                # Once each quant mode has enough selected samples, this term becomes 0.
                if state_name not in self._quant_select_counts:
                    state_name_for_count = "unknown"
                else:
                    state_name_for_count = state_name

                aid_for_parse = str(getattr(p, "action_id", "")).lower()
                if (
                    "rho0p60" in aid_for_parse
                    or "rho0p6" in aid_for_parse
                    or "rho0.6" in aid_for_parse
                ):
                    rho_key = "rho0p60"
                elif "rho0p25" in aid_for_parse or "rho0.25" in aid_for_parse:
                    rho_key = "rho0p25"
                elif (
                    "rho0p10" in aid_for_parse
                    or "rho0p1" in aid_for_parse
                    or "rho0.1" in aid_for_parse
                ):
                    rho_key = "rho0p10"
                elif "rho0p5" in aid_for_parse or "rho0.5" in aid_for_parse:
                    rho_key = "rho0p5_legacy"
                else:
                    rho_key = "rho0"

                cache_key = "cache1" if "cache1" in aid_for_parse else "cache0"

                quant_counts = self._quant_select_counts[state_name_for_count]
                rho_counts = self._rho_select_counts[state_name_for_count]
                cache_counts = self._cache_select_counts[state_name_for_count]

                q_count = int(quant_counts.get(q, 0))
                rho_count = int(rho_counts.get(rho_key, 0))
                cache_count = int(cache_counts.get(cache_key, 0))

                q_gap = (
                    max(0, int(self.explore_warmup_pulls_per_quant) - q_count)
                    if q in quant_counts else 0
                )
                rho_gap = (
                    max(0, int(self.explore_warmup_pulls_per_rho) - rho_count)
                    if rho_key in rho_counts else 0
                )
                cache_gap = max(0, int(self.explore_warmup_pulls_per_cache) - cache_count)

                # Component warm-up:
                # explore quant, redundancy, and cache components under each state.
                # This is exploration only; reward still decides which components remain useful.
                exploration_bonus = (
                    0.60 * float(q_gap > 0)
                    + 0.25 * float(rho_gap > 0)
                    + 0.15 * float(cache_gap > 0)
                ) * float(self.explore_bonus)

                # Exploration should not select spatially redundant CAVs.
                exploration_bonus = float(exploration_bonus) * float(marginal_coverage)
                ratio = float(base_ratio + exploration_bonus)

                info = {
                    "ratio": float(ratio),
                    "sender_id": sid,
                    "action_id": p.action_id,
                    "ucb": float(p.ucb),
                    "gain": float(gain),
                    "base_ucb": float(p.ucb),
                    "marginal_coverage": float(marginal_coverage),
                    "gain_formula": "max(ucb,0)*marginal_coverage",
                    "static_complementarity_context_only": True,
                    "learned_ucb_cost_penalty": True,
                    "channel_state": str(state_name),
                    "quant_mode": q,
                    "rho_key": str(rho_key),
                    "cache_key": str(cache_key),
                    "exploration_bonus": float(exploration_bonus),
                    "cost_norm": float(cost_norm),
                    "oracle_cost_lambda": float(oracle_cost_lambda),
                    "cost_bytes": float(p.cost_bytes),
                    "complementarity": float(comp),
                    "overlap_with_selected": float(red),
                    "remaining_budget_before_select": float(remaining),
                    "ego_mask_initialized": bool(ego_mask_initialized),
                }
                round_candidates.append(info)
                if best is None or ratio > best_info["ratio"]:
                    best = p
                    best_info = info

            if not first_round_candidates and round_candidates:
                first_round_candidates = sorted(
                    round_candidates,
                    key=lambda x: x["ratio"],
                    reverse=True,
                )[:50]

            if best is None:
                break

            selected.append(best)

            # Update warm-up exploration counts for selected quant/rho/cache arms.
            # Step 6: all three action components must be counted; otherwise
            # rho/cache warm-up bonus may remain active for too long.
            try:
                _ssel = str((best_info or {}).get("channel_state", "unknown")).lower()
                if _ssel not in self._quant_select_counts:
                    _ssel = "unknown"

                _qsel = str((best_info or {}).get("quant_mode", "unknown")).lower()
                if _qsel in self._quant_select_counts[_ssel]:
                    self._quant_select_counts[_ssel][_qsel] += 1

                _rho = str((best_info or {}).get("rho_key", "rho0"))
                if _rho in self._rho_select_counts[_ssel]:
                    self._rho_select_counts[_ssel][_rho] += 1

                _cache = str((best_info or {}).get("cache_key", "cache0"))
                if _cache in self._cache_select_counts[_ssel]:
                    self._cache_select_counts[_ssel][_cache] += 1
            except Exception:
                pass
            selected_sender_ids.add(str(best.sender_id))
            remaining -= float(best.cost_bytes)
            selected_union_mask = _union_mask(selected_union_mask, best.mask)
            ranked_history.append(best_info)

        unique_sender_ids = sorted({str(p.sender_id) for p in candidates})
        return {
            "selected": selected,
            "selected_sender_ids": [str(p.sender_id) for p in selected],
            "selected_action_ids": [p.action_id for p in selected],
            "budget_bytes": float(budget_bytes),
            "used_budget_bytes": float(float(budget_bytes) - remaining),
            "remaining_budget_bytes": float(remaining),
            "num_candidates": len(candidates),
            "num_unique_senders": len(unique_sender_ids),
            "unique_sender_ids": unique_sender_ids,
            "num_selected": len(selected),
            "diversity_aware": bool(self.diversity_aware),
            "min_marginal_coverage": float(self.min_marginal_coverage),
            "oracle_cost_lambda": float(self.oracle_cost_lambda),
            "explore_warmup_pulls_per_quant": int(self.explore_warmup_pulls_per_quant),
            "explore_warmup_pulls_per_rho": int(self.explore_warmup_pulls_per_rho),
            "explore_warmup_pulls_per_cache": int(self.explore_warmup_pulls_per_cache),
            "explore_bonus": float(self.explore_bonus),
            "ranked": ranked_history,
            "first_round_candidates": first_round_candidates,
        }


__all__ = ["CAVProposal", "EgoGreedyKnapsackOracle"]


# Proposal construction

import copy
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from opencood.methods.arce.context.local_confidence import get_cav_confidence
from opencood.methods.arce.context.payload_context import build_payload_pair_context, build_detection_confidence_pair_context
from opencood.methods.arce.transport_policy.payload_transport import is_payload_native_transport
from opencood.methods.arce.policy.c2mab.action_space import PDFARCEAction
from opencood.methods.arce.context.sender_candidate_selector import build_sender_candidates


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
