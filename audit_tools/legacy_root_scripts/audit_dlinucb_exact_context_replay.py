#!/usr/bin/env python3
"""Exact-context offline replay audit for ARCE Discounted LinUCB.

The input must contain complete counterfactual action rows: one row per
frame/sender/action.  At replay time only the selected action is used to
update the policy.  Counterfactual labels for unselected actions are used
only to compute offline regret and never enter the policy update.

This is deliberately a link-level diagnostic of the six learned send arms.
Production excludes no-send from scored proposals, so no-send remains in the
input only as a counterfactual label and is not modeled as a seventh bandit
arm.  This tool does not reproduce the multi-sender greedy super-arm oracle or
frame-level credit assignment.
"""

from __future__ import print_function

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


NO_SEND_PREFIX = "send0_"
EXPECTED_ACTIONS = 7
CONTEXT_FIELDS = (
    "B_norm",
    "p_loss",
    "d_norm",
    "ego_confidence",
    "cache_quality",
    "complementarity",
    "cav_confidence",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay production D-LinUCB send-arm learning with exact runtime "
            "context from seven-action counterfactual data."
        )
    )
    parser.add_argument("--csv", required=True, help="Merged counterfactual CSV")
    parser.add_argument("--out_dir", required=True, help="Audit output directory")
    parser.add_argument("--sender_index", default="1")
    parser.add_argument("--reference_bytes", type=float, default=337500.0)
    parser.add_argument("--lambda_abs", type=float, default=0.1)
    parser.add_argument("--lambda_delta", type=float, default=3.0)
    parser.add_argument("--lambda_cost", type=float, default=0.3)
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--markov_discount", type=float, default=0.995)
    parser.add_argument("--tie_tolerance", type=float, default=0.01)
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument(
        "--seeds",
        default="2026,2027,2028,2029,2030",
        help="Seeds used to permute the one-pull-per-action initialization",
    )
    parser.add_argument(
        "--production_feedback_weight",
        choices=("none", "channel_quality", "statistical"),
        default="channel_quality",
    )
    return parser.parse_args()


def as_float(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError("{} must be numeric, got {!r}".format(name, value))
    if not math.isfinite(result):
        raise ValueError("{} must be finite, got {!r}".format(name, value))
    return result


def as_int(value, name):
    return int(as_float(value, name))


def is_no_send(row):
    value = str(row.get("no_send", "")).strip().lower()
    return value in ("1", "true", "yes") or str(row["action_id"]).startswith(
        NO_SEND_PREFIX
    )


def normalized_entropy(counter, action_ids):
    total = float(sum(counter.values()))
    if total <= 0.0 or len(action_ids) <= 1:
        return 0.0
    entropy = 0.0
    for action_id in action_ids:
        p = float(counter.get(action_id, 0)) / total
        if p > 0.0:
            entropy -= p * math.log(p)
    return float(entropy / math.log(float(len(action_ids))))


def mean(values):
    return float(sum(values) / max(len(values), 1))


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(float(x) for x in values)
    index = int(round(float(q) * float(len(ordered) - 1)))
    return float(ordered[max(0, min(index, len(ordered) - 1))])


def summary(values):
    values = [float(x) for x in values]
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": min(values),
        "p10": percentile(values, 0.10),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "max": max(values),
        "mean": mean(values),
    }


def state_features(state):
    state = str(state).strip().lower()
    known = ("good", "medium", "bad")
    if state not in known:
        raise ValueError("Unsupported channel_state={!r}".format(state))
    return [1.0 if state == item else 0.0 for item in known]


def channel_profile(group):
    # Production channel-quality weighting currently consumes loss_rate.
    return {"loss_rate": float(group["context"][1])}


def build_context(group):
    """Read and validate the exact action-independent runtime context."""
    vectors = []
    for row in group:
        available = str(row["decision_context_available"]).strip().lower()
        if available not in ("1", "true", "yes"):
            raise ValueError(
                "decision context unavailable for action {}".format(row["action_id"])
            )
        vectors.append(
            np.asarray(
                [
                    as_float(
                        row["decision_context_" + name],
                        "decision_context_" + name,
                    )
                    for name in CONTEXT_FIELDS
                ],
                dtype=np.float64,
            )
        )
    reference = vectors[0]
    spread = max(float(np.max(np.abs(vector - reference))) for vector in vectors)
    if spread > 1e-9:
        raise ValueError(
            "counterfactual actions do not share one decision context; "
            "max_abs_diff={}".format(spread)
        )
    return reference


def reward_values(row, args):
    if is_no_send(row):
        return {"true": 0.0, "proxy": 0.0}

    tx_bytes = as_float(row["tx_bytes"], "tx_bytes")
    cost = float(args.lambda_cost) * tx_bytes / float(args.reference_bytes)

    true_quality = as_float(
        row["true_quality_mean_0357"], "true_quality_mean_0357"
    )
    true_delta = as_float(
        row["label_true_global_delta_quality_mean_0357"],
        "label_true_global_delta_quality_mean_0357",
    )
    proxy_quality = as_float(row["proxy_collab_quality"], "proxy_collab_quality")
    proxy_delta = as_float(row["proxy_delta_quality"], "proxy_delta_quality")

    return {
        "true": (
            float(args.lambda_abs) * true_quality
            + float(args.lambda_delta) * true_delta
            - cost
        ),
        "proxy": (
            float(args.lambda_abs) * proxy_quality
            + float(args.lambda_delta) * proxy_delta
            - cost
        ),
    }


def load_groups(path, args):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Counterfactual CSV is empty: {}".format(path))

    required = {
        "sequence_id",
        "sequence_frame_idx",
        "sender_index",
        "action_id",
        "channel_state",
        "no_send",
        "tx_bytes",
        "true_quality_mean_0357",
        "label_true_global_delta_quality_mean_0357",
        "proxy_collab_quality",
        "proxy_ego_quality",
        "proxy_delta_quality",
        "decision_context_available",
    }
    required.update("decision_context_" + name for name in CONTEXT_FIELDS)
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise ValueError("CSV is missing required fields: {}".format(missing))

    grouped = defaultdict(list)
    for row in rows:
        key = (
            as_int(row["sequence_id"], "sequence_id"),
            as_int(row["sequence_frame_idx"], "sequence_frame_idx"),
            str(row["sender_index"]),
        )
        grouped[key].append(row)

    action_sets = Counter()
    groups = []
    for key in sorted(grouped):
        group = grouped[key]
        action_ids = tuple(sorted(str(row["action_id"]) for row in group))
        action_sets[action_ids] += 1
        if len(group) != EXPECTED_ACTIONS or len(set(action_ids)) != EXPECTED_ACTIONS:
            raise ValueError(
                "Group {} does not contain seven unique actions: {}".format(
                    key, action_ids
                )
            )
        states = {str(row["channel_state"]).strip().lower() for row in group}
        if len(states) != 1:
            raise ValueError("Group {} has multiple channel states: {}".format(key, states))
        group_by_action = {str(row["action_id"]): row for row in group}
        rewards = {
            action_id: reward_values(row, args)
            for action_id, row in group_by_action.items()
        }
        groups.append(
            {
                "key": key,
                "sender_index": key[2],
                "state": next(iter(states)),
                "rows": group_by_action,
                "context": build_context(group),
                "rewards": rewards,
            }
        )

    if len(action_sets) != 1:
        raise ValueError("Inconsistent action spaces: {}".format(action_sets))

    action_ids = list(next(iter(action_sets.keys())))
    send_action_ids = [
        action_id
        for action_id in action_ids
        if not action_id.startswith(NO_SEND_PREFIX)
    ]
    if len(send_action_ids) != EXPECTED_ACTIONS - 1:
        raise ValueError(
            "Expected six learned send actions, got {}".format(send_action_ids)
        )
    ego_spreads = []
    for group in groups:
        values = [
            as_float(row["proxy_ego_quality"], "proxy_ego_quality")
            for row in group["rows"].values()
        ]
        ego_spreads.append(max(values) - min(values))

    audit = {
        "csv_rows": len(rows),
        "groups": len(groups),
        "actions": action_ids,
        "senders": dict(Counter(group["sender_index"] for group in groups)),
        "states": dict(Counter(group["state"] for group in groups)),
        "proxy_ego_action_spread": summary(ego_spreads),
        "learned_send_actions": send_action_ids,
        "no_send_actions": [
            action_id for action_id in action_ids if action_id not in send_action_ids
        ],
        "exact_runtime_context_available": True,
        "context_note": (
            "Replay uses the exact action-independent 7D decision context "
            "recorded by the runtime collector."
        ),
    }
    return groups, action_ids, send_action_ids, audit


def random_and_fixed_baselines(groups, action_ids):
    random_regrets = []
    fixed_total = {action_id: 0.0 for action_id in action_ids}
    fixed_regret = {action_id: 0.0 for action_id in action_ids}
    for group in groups:
        rewards = {
            action_id: group["rewards"][action_id]["true"]
            for action_id in action_ids
        }
        best = max(rewards.values())
        random_regrets.append(best - mean(list(rewards.values())))
        for action_id in action_ids:
            fixed_total[action_id] += rewards[action_id]
            fixed_regret[action_id] += best - rewards[action_id]
    best_fixed = max(action_ids, key=lambda action_id: fixed_total[action_id])
    return {
        "random_expected_mean_true_regret": mean(random_regrets),
        "best_fixed_action": best_fixed,
        "best_fixed_mean_true_regret": (
            fixed_regret[best_fixed] / max(len(groups), 1)
        ),
    }


def replay_once(
    policy_cls,
    groups,
    action_ids,
    feedback_source,
    discount,
    feedback_weight_mode,
    seed,
    args,
):
    policy = policy_cls(
        action_ids=action_ids,
        context_dim=len(groups[0]["context"]),
        lambda_reg=float(args.lambda_reg),
        discount=float(discount),
        beta=float(args.beta),
        feedback_weight_mode=str(feedback_weight_mode),
    )

    initialization_order = list(action_ids)
    random.Random(seed).shuffle(initialization_order)
    records = []
    selected_counter = Counter()

    for step, group in enumerate(groups):
        context = group["context"]
        scores = {
            action_id: policy.score(action_id, context) for action_id in action_ids
        }
        if step < len(initialization_order):
            selected = initialization_order[step]
            selection_reason = "one_pull_initialization"
        else:
            selected = max(action_ids, key=lambda action_id: scores[action_id].ucb)
            selection_reason = "ucb"

        true_rewards = {
            action_id: group["rewards"][action_id]["true"]
            for action_id in action_ids
        }
        feedback_rewards = {
            action_id: group["rewards"][action_id][feedback_source]
            for action_id in action_ids
        }
        true_best = max(true_rewards.values())
        feedback_best = max(feedback_rewards.values())
        true_top_set = {
            action_id
            for action_id, value in true_rewards.items()
            if true_best - value <= float(args.tie_tolerance)
        }
        feedback_top_set = {
            action_id
            for action_id, value in feedback_rewards.items()
            if feedback_best - value <= float(args.tie_tolerance)
        }

        observed_reward = feedback_rewards[selected]
        weight = policy.update(
            selected,
            context,
            observed_reward,
            channel_profile=channel_profile(group),
        )
        selected_counter[selected] += 1

        score = scores[selected]
        records.append(
            {
                "step": step,
                "sequence_id": group["key"][0],
                "sequence_frame_idx": group["key"][1],
                "sender_index": group["sender_index"],
                "channel_state": group["state"],
                "selected_action": selected,
                "selection_reason": selection_reason,
                "observed_reward": observed_reward,
                "selected_true_reward": true_rewards[selected],
                "true_best_reward": true_best,
                "true_regret": max(0.0, true_best - true_rewards[selected]),
                "feedback_regret": max(
                    0.0, feedback_best - feedback_rewards[selected]
                ),
                "true_top_set_hit": selected in true_top_set,
                "feedback_top_set_hit": selected in feedback_top_set,
                "ucb": score.ucb,
                "ucb_mean": score.mean,
                "ucb_bonus": score.bonus,
                "feedback_weight": weight,
            }
        )

    n = len(records)
    quarter = max(1, n // 4)
    first = records[:quarter]
    last = records[-quarter:]

    def block_metrics(block):
        actions = Counter(row["selected_action"] for row in block)
        return {
            "n": len(block),
            "mean_true_regret": mean([row["true_regret"] for row in block]),
            "mean_feedback_regret": mean(
                [row["feedback_regret"] for row in block]
            ),
            "true_top_set_hit_rate": mean(
                [float(row["true_top_set_hit"]) for row in block]
            ),
            "feedback_top_set_hit_rate": mean(
                [float(row["feedback_top_set_hit"]) for row in block]
            ),
            "mean_ucb_bonus": mean([row["ucb_bonus"] for row in block]),
            "mean_abs_ucb_mean": mean(
                [abs(row["ucb_mean"]) for row in block]
            ),
            "normalized_action_entropy": normalized_entropy(actions, action_ids),
            "action_counter": dict(actions),
        }

    windows = []
    size = max(1, int(args.window_size))
    for start in range(0, n, size):
        block = records[start : start + size]
        metrics = block_metrics(block)
        metrics["start"] = start
        metrics["end"] = start + len(block) - 1
        windows.append(metrics)

    return {
        "seed": seed,
        "feedback_source": feedback_source,
        "discount": discount,
        "feedback_weight_mode": feedback_weight_mode,
        "num_steps": n,
        "initialization_order": initialization_order,
        "selected_actions": dict(selected_counter),
        "cumulative_true_regret": sum(row["true_regret"] for row in records),
        "cumulative_feedback_regret": sum(
            row["feedback_regret"] for row in records
        ),
        "overall": block_metrics(records),
        "first_quarter": block_metrics(first),
        "last_quarter": block_metrics(last),
        "windows": windows,
        "records": records,
    }


def aggregate_replays(name, replays, baselines):
    first_true = [x["first_quarter"]["mean_true_regret"] for x in replays]
    last_true = [x["last_quarter"]["mean_true_regret"] for x in replays]
    first_feedback = [
        x["first_quarter"]["mean_feedback_regret"] for x in replays
    ]
    last_feedback = [
        x["last_quarter"]["mean_feedback_regret"] for x in replays
    ]
    first_hit = [x["first_quarter"]["true_top_set_hit_rate"] for x in replays]
    last_hit = [x["last_quarter"]["true_top_set_hit_rate"] for x in replays]

    last_true_mean = mean(last_true)
    result = {
        "name": name,
        "num_replays": len(replays),
        "num_steps": replays[0]["num_steps"],
        "feedback_source": replays[0]["feedback_source"],
        "discount": replays[0]["discount"],
        "feedback_weight_mode": replays[0]["feedback_weight_mode"],
        "baselines": baselines,
        "first_quarter_mean_true_regret": summary(first_true),
        "last_quarter_mean_true_regret": summary(last_true),
        "first_quarter_mean_feedback_regret": summary(first_feedback),
        "last_quarter_mean_feedback_regret": summary(last_feedback),
        "first_quarter_true_top_set_hit_rate": summary(first_hit),
        "last_quarter_true_top_set_hit_rate": summary(last_hit),
        "late_true_regret_vs_random": (
            last_true_mean
            / max(baselines["random_expected_mean_true_regret"], 1e-12)
        ),
        "late_true_regret_vs_best_fixed": (
            last_true_mean
            / max(baselines["best_fixed_mean_true_regret"], 1e-12)
        ),
        "per_seed": [
            {
                key: value
                for key, value in replay.items()
                if key not in ("records", "windows")
            }
            for replay in replays
        ],
    }
    return result


def build_diagnosis(results):
    stationary_true = results["stationary_true"]
    stationary_proxy = results["stationary_proxy"]
    markov_true = results["markov_true"]
    markov_proxy = results["markov_proxy"]

    def late(item, field):
        return float(item[field]["mean"])

    st_true = late(stationary_true, "last_quarter_mean_true_regret")
    st_proxy_true = late(stationary_proxy, "last_quarter_mean_true_regret")
    st_proxy_obj = late(stationary_proxy, "last_quarter_mean_feedback_regret")
    mk_true = late(markov_true, "last_quarter_mean_true_regret")
    mk_proxy_true = late(markov_proxy, "last_quarter_mean_true_regret")
    mk_proxy_obj = late(markov_proxy, "last_quarter_mean_feedback_regret")

    core_beats_random = stationary_true["late_true_regret_vs_random"] < 1.0
    proxy_optimizes_itself = st_proxy_obj < late(
        stationary_proxy, "first_quarter_mean_feedback_regret"
    )
    proxy_gap = st_proxy_true - st_true
    markov_proxy_gap = mk_proxy_true - mk_true

    if not core_beats_random:
        primary = (
            "The stationary true-reward replay does not beat random selection. "
            "Inspect context linear realizability, beta/lambda scaling, action "
            "initialization and the D-LinUCB implementation before tuning Proxy."
        )
    elif proxy_gap > 0.02 or markov_proxy_gap > 0.02:
        primary = (
            "D-LinUCB learns materially better with true reward than with the "
            "deployed Proxy reward. The principal bottleneck is reward/Proxy "
            "misspecification, not lack of additional discount tuning."
        )
    elif mk_true > st_true + 0.02:
        primary = (
            "The core works in the stationary slice but degrades under Markov "
            "transitions even with exact runtime context. Focus next on discount "
            "selection, exploration scaling and channel-conditioned feedback weighting."
        )
    else:
        primary = (
            "This exact-context link-level replay does not isolate a dominant "
            "failure. Inspect parameter sensitivity next, then run joint "
            "super-arm credit-assignment replay if link-level learning is sound."
        )

    return {
        "primary_conclusion": primary,
        "stationary_true_beats_random": core_beats_random,
        "stationary_proxy_optimizes_proxy_objective": proxy_optimizes_itself,
        "stationary_proxy_minus_true_late_true_regret": proxy_gap,
        "markov_proxy_minus_true_late_true_regret": markov_proxy_gap,
        "stationary_true_late_regret": st_true,
        "stationary_proxy_late_true_regret": st_proxy_true,
        "stationary_proxy_late_proxy_regret": st_proxy_obj,
        "markov_true_late_regret": mk_true,
        "markov_proxy_late_true_regret": mk_proxy_true,
        "markov_proxy_late_proxy_regret": mk_proxy_obj,
        "limitations": [
            "Replay evaluates the six learned send arms; no-send is a fixed reject path in production and is not treated as a seventh learned arm.",
            "Replay is link-level and does not reproduce the greedy super-arm oracle.",
            "Counterfactual groups were sampled, so replay time is not every online frame.",
            "True labels are used only for offline evaluation, never for policy updates.",
        ],
    }


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from opencood.methods.arce.policies.discounted_linucb import DiscountedLinUCB

    groups, all_action_ids, action_ids, input_audit = load_groups(args.csv, args)
    sender_groups = [
        group for group in groups if group["sender_index"] == str(args.sender_index)
    ]
    if not sender_groups:
        raise ValueError("No groups found for sender_index={}".format(args.sender_index))
    stationary_groups = [
        group for group in sender_groups if group["state"] == "medium"
    ]
    if len(stationary_groups) < len(action_ids) * 2:
        raise ValueError(
            "Need at least {} medium-state groups, found {}".format(
                len(action_ids) * 2, len(stationary_groups)
            )
        )

    seeds = [int(item.strip()) for item in str(args.seeds).split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one replay seed is required")

    scenario_specs = {
        "stationary_true": (stationary_groups, "true", 1.0, "none"),
        "stationary_proxy": (stationary_groups, "proxy", 1.0, "none"),
        "markov_true": (sender_groups, "true", args.markov_discount, "none"),
        "markov_proxy": (sender_groups, "proxy", args.markov_discount, "none"),
        "markov_proxy_production_weight": (
            sender_groups,
            "proxy",
            args.markov_discount,
            args.production_feedback_weight,
        ),
    }

    results = {}
    detailed = {}
    for name, (scenario_groups, source, discount, weight_mode) in scenario_specs.items():
        baselines = random_and_fixed_baselines(scenario_groups, action_ids)
        replays = [
            replay_once(
                DiscountedLinUCB,
                scenario_groups,
                action_ids,
                source,
                discount,
                weight_mode,
                seed,
                args,
            )
            for seed in seeds
        ]
        results[name] = aggregate_replays(name, replays, baselines)
        detailed[name] = replays

    report = {
        "input_audit": input_audit,
        "replay_scope": {
            "sender_index": str(args.sender_index),
            "stationary_state": "medium",
            "stationary_groups": len(stationary_groups),
            "markov_groups": len(sender_groups),
            "context_dim": len(sender_groups[0]["context"]),
            "context_fields": list(CONTEXT_FIELDS),
            "counterfactual_actions": all_action_ids,
            "learned_bandit_arms": action_ids,
            "excluded_from_bandit_scoring": [
                action_id
                for action_id in all_action_ids
                if action_id not in action_ids
            ],
            "action_semantics": (
                "Production-faithful send-arm diagnostic: six send actions are "
                "scored and updated; no-send is not a learned arm. Regret and "
                "fixed/random baselines are computed within the six send arms."
            ),
            "one_pull_initialization": True,
        },
        "reward_definition": {
            "true": (
                "send ? lambda_abs*true_quality + lambda_delta*true_global_delta "
                "- lambda_cost*tx_bytes/reference_bytes : 0"
            ),
            "proxy": (
                "send ? lambda_abs*proxy_collab_quality + "
                "lambda_delta*proxy_delta_quality - "
                "lambda_cost*tx_bytes/reference_bytes : 0"
            ),
            "lambda_abs": args.lambda_abs,
            "lambda_delta": args.lambda_delta,
            "lambda_cost": args.lambda_cost,
            "reference_bytes": args.reference_bytes,
        },
        "policy": {
            "implementation": "production DiscountedLinUCB import",
            "lambda_reg": args.lambda_reg,
            "beta": args.beta,
            "markov_discount": args.markov_discount,
            "seeds": seeds,
        },
        "scenarios": results,
    }
    report["diagnosis"] = build_diagnosis(results)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "dlinucb_counterfactual_replay_report.json"
    detail_path = out_dir / "dlinucb_counterfactual_replay_details.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    detail_path.write_text(
        json.dumps(detailed, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("===== D-LinUCB exact-context send-arm replay =====")
    print("groups: total={} sender={} stationary_medium={}".format(
        len(groups), len(sender_groups), len(stationary_groups)
    ))
    print(
        "scenario | feedback | gamma | late true regret | late feedback regret | "
        "late true top-set | random regret"
    )
    for name in scenario_specs:
        item = results[name]
        print(
            "{} | {} | {:.4f} | {:.6f} | {:.6f} | {:.4f} | {:.6f}".format(
                name,
                item["feedback_source"],
                item["discount"],
                item["last_quarter_mean_true_regret"]["mean"],
                item["last_quarter_mean_feedback_regret"]["mean"],
                item["last_quarter_true_top_set_hit_rate"]["mean"],
                item["baselines"]["random_expected_mean_true_regret"],
            )
        )
    print("\nDiagnosis:")
    print(report["diagnosis"]["primary_conclusion"])
    print("\nSaved:", report_path)
    print("Saved:", detail_path)


if __name__ == "__main__":
    main()
