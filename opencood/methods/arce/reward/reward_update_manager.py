"""Reward update manager for GRACE / C2MAB.

This module owns the reward-update stage after communication execution:

1. consume pending link-level reward items;
2. compute AP-proxy-gain dominated reward;
3. pass channel profile into D-LinUCB / corrupted-feedback update;
4. build clean reward_update records for debugging and audit.

The communication executor should only orchestrate this module, instead of
holding the full reward-update logic inline.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from opencood.methods.arce.reward.ap_gain_reward import c2mab_ap_gain_reward


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _build_channel_profile(item: Dict[str, Any]) -> Dict[str, Any]:
    """Build a channel profile for feedback weighting.

    Prefer explicit item["channel_profile"]. If absent, fall back to q_recv
    and use loss_rate = 1 - q_recv.
    """
    channel_profile = item.get("channel_profile", None)
    if isinstance(channel_profile, dict):
        return dict(channel_profile)

    q_recv = _safe_float(item.get("q_recv", item.get("q_eff", 1.0)), 1.0)
    q_recv = max(0.0, min(1.0, q_recv))
    return {"loss_rate": float(1.0 - q_recv)}


def _get_last_corruption_info(policy: Any, action_id: str) -> Dict[str, Any]:
    if hasattr(policy, "last_feedback_corruption_info"):
        try:
            return dict(policy.last_feedback_corruption_info.get(action_id, {}))
        except Exception:
            return {}
    return {}


def _stats(values: List[Any]) -> Dict[str, Any]:
    vals = []
    for v in values:
        try:
            vals.append(float(v))
        except Exception:
            pass

    if not vals:
        return {"n": 0}

    vals = sorted(vals)

    def pct(q: float) -> float:
        if len(vals) == 1:
            return float(vals[0])
        idx = int(round(float(q) * float(len(vals) - 1)))
        idx = max(0, min(len(vals) - 1, idx))
        return float(vals[idx])

    mean = float(sum(vals) / max(len(vals), 1))
    return {
        "n": int(len(vals)),
        "min": float(vals[0]),
        "p10": pct(0.10),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "max": float(vals[-1]),
        "mean": float(mean),
        "pos": int(sum(1 for x in vals if x > 0.0)),
        "neg": int(sum(1 for x in vals if x < 0.0)),
        "zero": int(sum(1 for x in vals if x == 0.0)),
    }


def _summarize_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    fields = [
        "reward",
        "total_reward",
        "ego_quality",
        "collab_quality",
        "raw_delta_quality",
        "delta_term",
        "abs_ap_term",
        "perception_term",
        "ap_term",
        "cost_bytes",
        "actual_tx_bytes",
        "budget_bytes",
        "link_budget_bytes",
        "frame_budget_bytes",
        "cost_reference_bytes",
        "cost_norm",
        "cost_norm_ref",
        "cost_norm_frame",
        "link_cost_norm",
        "cost_penalty",
        "delay_penalty",
        "quant_penalty",
        "violation_penalty",
        "normalized_cost",
        "delay_norm",
        "credit_weight",
        "raw_credit_weight",
    ]
    return {
        "count": int(len(rows)),
        **{field: _stats([r.get(field) for r in rows]) for field in fields},
    }


def _summarize_by_key(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        name = str(row.get(key, "unknown"))
        groups.setdefault(name, []).append(row)

    out = {}
    for name, group_rows in groups.items():
        out[name] = {
            "count": int(len(group_rows)),
            "reward": _stats([r.get("reward") for r in group_rows]),
            "ap_term": _stats([r.get("ap_term") for r in group_rows]),
            "cost_penalty": _stats([r.get("cost_penalty") for r in group_rows]),
            "delay_penalty": _stats([r.get("delay_penalty") for r in group_rows]),
            "quant_penalty": _stats([r.get("quant_penalty") for r in group_rows]),
        }
    return out


def _summarize_reward_terms(reward_infos: List[Dict[str, Any]]) -> Dict[str, Any]:
    send_rows = [
        r for r in reward_infos
        if not bool(r.get("no_send_update", False))
    ]
    no_send_rows = [
        r for r in reward_infos
        if bool(r.get("no_send_update", False))
    ]

    return {
        "all": _summarize_group(reward_infos),
        "send": _summarize_group(send_rows),
        "no_send": _summarize_group(no_send_rows),
        "by_action_id": _summarize_by_key(reward_infos, "action_id"),
        "by_quant_mode": _summarize_by_key(reward_infos, "quant_mode"),
        "by_channel_state": _summarize_by_key(reward_infos, "channel_state"),
    }


def update_pending_rewards(
    pending: List[Dict[str, Any]],
    ego_confidence: float,
    collab_confidence: float,
    budget_bytes: Optional[float],
    get_policy_fn: Callable[[Any, Any], Any],
    reward_lambda_ap: float = 1.0,
    reward_mode: str = "simple_delta",
    reward_lambda_delta: Optional[float] = None,
    reward_lambda_abs: float = 0.0,
    reward_lambda_cost: float = 0.10,
    reward_cost_norm_mode: str = "reference",
    reward_cost_reference_bytes: Optional[float] = None,
    reward_lambda_delay: float = 0.05,
    reward_lambda_quant: float = 0.0,
    reward_lambda_violate: float = 0.0,
    reward_stale_max_ms: float = 100.0,
    quant_quality_cfg: Optional[Dict[str, Any]] = None,
    apply_policy_update: bool = True,
) -> Dict[str, Any]:
    """Update policies from pending communication records.

    Parameters
    ----------
    pending:
        Pending link-level communication records.
    ego_confidence:
        Ego-only AP proxy confidence.
    collab_confidence:
        Collaborative AP proxy confidence after communication/fusion.
    budget_bytes:
        Frame-level or link-level budget fallback.
    get_policy_fn:
        Function returning the LinUCB policy for (ego_id, sender_id).

    Returns
    -------
    summary:
        A clean reward_update summary. Old mixed-reward fields such as
        fec_gain are intentionally not written here.
    """
    ego_confidence = float(ego_confidence)
    collab_confidence = float(collab_confidence)
    delta_conf = float(collab_confidence) - float(ego_confidence)
    lambda_delta_value = (
        float(reward_lambda_delta)
        if reward_lambda_delta is not None
        else float(reward_lambda_ap)
    )

    frame_budget_for_reward = _safe_float(budget_bytes, 1.0)
    if frame_budget_for_reward <= 0.0:
        frame_budget_for_reward = 1.0

    # Credit assignment: only actual send actions share frame-level perception gain.
    # No-send remains an explicit arm update, but it must not inherit another
    # sender's positive AP-proxy gain.
    send_indices = [
        i for i, x in enumerate(pending)
        if not bool(x.get("no_send_update", False))
    ]

    raw_ws = [0.0 for _ in pending]
    for i in send_indices:
        item = pending[i]
        cav_conf = max(_safe_float(item.get("cav_confidence", 0.0)), 0.0)
        comp = max(
            _safe_float(
                item.get(
                    "complementarity",
                    item.get("complementarity_normalized", 0.0),
                )
            ),
            0.0,
        )

        score = cav_conf * comp
        if score <= 1e-12:
            score = max(_safe_float(item.get("contribution_weight", 0.0)), 0.0)
        raw_ws[i] = float(score)

    sw = sum(raw_ws)
    if sw <= 1e-12 and send_indices:
        uniform = 1.0 / float(len(send_indices))
        for i in send_indices:
            raw_ws[i] = uniform
        sw = 1.0

    reward_infos: List[Dict[str, Any]] = []

    for item, raw_w in zip(pending, raw_ws):
        contribution_weight = float(raw_w) / max(sw, 1e-12)

        reward, info = c2mab_ap_gain_reward(
            ap_proxy_gain=delta_conf,
            collab_quality=collab_confidence,
            ego_quality=ego_confidence,
            contribution_weight=contribution_weight,
            cost_bytes=_safe_float(item.get("cost_bytes", 0.0)),
            budget_bytes=_safe_float(
                item.get("link_budget_bytes", frame_budget_for_reward),
                1.0,
            ),
            link_budget_bytes=_safe_float(
                item.get("link_budget_bytes", frame_budget_for_reward),
                1.0,
            ),
            frame_budget_bytes=float(frame_budget_for_reward),
            cost_reference_bytes=reward_cost_reference_bytes,
            cost_norm_mode=str(reward_cost_norm_mode),
            delay_ms=_safe_float(item.get("delay_ms", 0.0)),
            budget_violation=bool(item.get("budget_violation", False)),
            quant_mode=str(item.get("quant_mode", "fp32")),
            lambda_ap=float(reward_lambda_ap),
            lambda_delta=float(lambda_delta_value),
            lambda_abs=float(reward_lambda_abs),
            reward_mode=str(reward_mode),
            lambda_cost=float(reward_lambda_cost),
            lambda_delay=float(reward_lambda_delay),
            lambda_quant=float(reward_lambda_quant),
            lambda_violate=float(reward_lambda_violate),
            stale_max_ms=float(reward_stale_max_ms),
            quant_quality_cfg=quant_quality_cfg,
        )

        policy = get_policy_fn(item["ego_id"], item["sender_id"])
        policy_t_before = int(getattr(policy, "t", -1))
        context_vector = item["context_vector"]
        action_id = str(item["action_id"])
        channel_profile = _build_channel_profile(item)

        feedback_weight = 0.0
        if apply_policy_update:
            feedback_weight = policy.update(
                action_id,
                context_vector,
                reward,
                channel_profile=channel_profile,
            )
        policy_t_after = int(getattr(policy, "t", -1))

        corruption_info = _get_last_corruption_info(policy, action_id)

        info["policy_update_debug"] = {
            "action_id": action_id,
            "reward": float(reward),
            "context_dim": int(len(context_vector)),
            "policy_t_before": int(policy_t_before),
            "policy_t_after": int(policy_t_after),
            "policy_t_delta": int(policy_t_after - policy_t_before),
            "policy_update_applied": bool(apply_policy_update),
            "feedback_weight": float(feedback_weight),
            "feedback_corruption_C": float(
                corruption_info.get("feedback_corruption_C", 0.0)
            ),
            "feedback_corruption_info": dict(corruption_info),
            "channel_profile": dict(channel_profile),
        }

        info.update(
            {
                "ego_id": str(item["ego_id"]),
                "sender_id": str(item["sender_id"]),
                "action_id": action_id,
            }
        )

        info["q_recv"] = _safe_float(item.get("q_recv", 0.0))
        info["quant_mode"] = str(item.get("quant_mode", ""))
        info["channel_state"] = str(item.get("channel_state", ""))
        info["redundancy_ratio"] = _safe_float(item.get("redundancy_ratio", 0.0))
        info["cache_enabled"] = _safe_float(item.get("cache_enabled", 0))
        info["cache_quality"] = _safe_float(item.get("cache_quality", 0.0))
        info["complementarity_raw"] = _safe_float(
            item.get("complementarity_raw", 0.0)
        )
        info["complementarity_normalized"] = _safe_float(
            item.get("complementarity_normalized", 0.0)
        )
        info["complementarity"] = _safe_float(item.get("complementarity", 0.0))
        info["cav_confidence"] = _safe_float(item.get("cav_confidence", 0.0))
        info["cav_confidence_source"] = str(
            item.get("cav_confidence_source", "unknown")
        )
        info["no_send_update"] = bool(item.get("no_send_update", False))
        info["raw_credit_weight"] = float(raw_w)

        reward_infos.append(info)

    send_count = int(
        sum(1 for x in reward_infos if not bool(x.get("no_send_update", False)))
    )
    no_send_count = int(
        sum(1 for x in reward_infos if bool(x.get("no_send_update", False)))
    )

    summary = {
        "collab_confidence": float(collab_confidence),
        "ego_confidence": float(ego_confidence),
        "delta_confidence": float(delta_conf),
        "reward_mode": str(reward_mode),
        "reward_lambda_delta": float(lambda_delta_value),
        "reward_lambda_abs": float(reward_lambda_abs),
        "reward_lambda_cost": float(reward_lambda_cost),
        "reward_cost_norm_mode": str(reward_cost_norm_mode),
        "reward_cost_reference_bytes": (
            None if reward_cost_reference_bytes is None else float(reward_cost_reference_bytes)
        ),
        "frame_budget_bytes": float(frame_budget_for_reward),
        "policy_update_applied": bool(apply_policy_update),
        "num_updated": len(pending),
        "num_send_updated": int(send_count),
        "num_no_send_updated": int(no_send_count),
        "mean_reward": float(
            sum(x["reward"] for x in reward_infos) / max(len(reward_infos), 1)
        ),
        "reward_term_summary": _summarize_reward_terms(reward_infos),
        "link_rewards": reward_infos,
    }
    return summary
