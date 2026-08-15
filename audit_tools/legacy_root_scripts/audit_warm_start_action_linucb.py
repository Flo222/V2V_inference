#!/usr/bin/env python3
"""Grouped-CV warm-start audit for shared action-conditioned LinUCB."""

from __future__ import print_function

import argparse
import json
from collections import Counter
from pathlib import Path

import audit_dlinucb_exact_context_replay as replay
from audit_shared_action_linucb_replay import SharedActionLinUCB, csv_values


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--sender_index", default="1")
    parser.add_argument("--reference_bytes", type=float, default=337500.0)
    parser.add_argument("--lambda_abs", type=float, default=0.1)
    parser.add_argument("--lambda_delta", type=float, default=3.0)
    parser.add_argument("--lambda_cost", type=float, default=0.3)
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--tie_tolerance", type=float, default=0.01)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--betas", default="0,0.02,0.05,0.1,0.2,0.5")
    parser.add_argument(
        "--online_feedback_sources", default="true,proxy"
    )
    return parser.parse_args()


def mean(values):
    return float(sum(values) / max(len(values), 1))


def fold_sequences(sequence_ids, folds):
    ordered = sorted(set(int(x) for x in sequence_ids))
    return [ordered[index::folds] for index in range(folds)]


def initialize_policy(
    train_groups,
    action_ids,
    context_dim,
    lambda_reg,
    beta,
):
    policy = SharedActionLinUCB(
        action_ids=action_ids,
        context_dim=context_dim,
        lambda_reg=lambda_reg,
        discount=1.0,
        beta=beta,
        feedback_weight_mode="none",
    )
    # Full-information labels are allowed only from training sequences.
    for group in train_groups:
        for action_id in action_ids:
            policy.update(
                action_id,
                group["context"],
                group["rewards"][action_id]["true"],
            )
    return policy


def evaluate_fold(
    train_groups,
    test_groups,
    action_ids,
    args,
    beta,
    feedback_source,
):
    policy = initialize_policy(
        train_groups=train_groups,
        action_ids=action_ids,
        context_dim=len(test_groups[0]["context"]),
        lambda_reg=float(args.lambda_reg),
        beta=float(beta),
    )

    regrets = []
    top_hits = []
    action_counter = Counter()
    for group in test_groups:
        scores = {
            action_id: policy.score(action_id, group["context"])
            for action_id in action_ids
        }
        selected = max(
            action_ids, key=lambda action_id: scores[action_id].ucb
        )
        true_rewards = {
            action_id: group["rewards"][action_id]["true"]
            for action_id in action_ids
        }
        best = max(true_rewards.values())
        top_set = {
            action_id
            for action_id, reward in true_rewards.items()
            if best - reward <= float(args.tie_tolerance)
        }
        regrets.append(max(0.0, best - true_rewards[selected]))
        top_hits.append(float(selected in top_set))
        action_counter[selected] += 1

        observed = group["rewards"][selected][feedback_source]
        policy.update(selected, group["context"], observed)

    return {
        "num_train_groups": len(train_groups),
        "num_test_groups": len(test_groups),
        "mean_true_regret": mean(regrets),
        "true_top_set_hit_rate": mean(top_hits),
        "action_counter": dict(action_counter),
    }


def main():
    args = parse_args()
    args.beta = 1.0
    args.markov_discount = 1.0
    args.production_feedback_weight = "none"

    groups, all_actions, send_actions, input_audit = replay.load_groups(
        args.csv, args
    )
    groups = [
        group
        for group in groups
        if group["sender_index"] == str(args.sender_index)
    ]
    if not groups:
        raise ValueError("no matching sender groups")

    folds = max(2, int(args.folds))
    sequence_folds = fold_sequences(
        [group["key"][0] for group in groups], folds
    )
    betas = csv_values(args.betas, float)
    feedback_sources = [
        value.strip()
        for value in str(args.online_feedback_sources).split(",")
        if value.strip()
    ]
    if any(x not in ("true", "proxy") for x in feedback_sources):
        raise ValueError("feedback sources must be true and/or proxy")

    baselines = replay.random_and_fixed_baselines(groups, send_actions)
    rows = []
    for feedback_source in feedback_sources:
        for beta in betas:
            fold_results = []
            for fold_index, test_sequences in enumerate(sequence_folds):
                test_set = set(test_sequences)
                train_groups = [
                    group
                    for group in groups
                    if group["key"][0] not in test_set
                ]
                test_groups = [
                    group
                    for group in groups
                    if group["key"][0] in test_set
                ]
                result = evaluate_fold(
                    train_groups,
                    test_groups,
                    send_actions,
                    args,
                    beta,
                    feedback_source,
                )
                result["fold"] = int(fold_index)
                result["test_sequences"] = list(test_sequences)
                fold_results.append(result)

            regret = mean([x["mean_true_regret"] for x in fold_results])
            top_hit = mean(
                [x["true_top_set_hit_rate"] for x in fold_results]
            )
            row = {
                "beta": float(beta),
                "online_feedback_source": feedback_source,
                "mean_true_regret": regret,
                "true_top_set_hit_rate": top_hit,
                "regret_vs_random": regret
                / max(
                    baselines["random_expected_mean_true_regret"], 1e-12
                ),
                "regret_vs_best_fixed": regret
                / max(
                    baselines["best_fixed_mean_true_regret"], 1e-12
                ),
                "folds": fold_results,
            }
            rows.append(row)
            print(
                "feedback={} beta={:.3f} regret={:.6f} "
                "vs_random={:.4f} vs_fixed={:.4f} top_set={:.4f}".format(
                    feedback_source,
                    beta,
                    regret,
                    row["regret_vs_random"],
                    row["regret_vs_best_fixed"],
                    top_hit,
                ),
                flush=True,
            )

    key = lambda row: (
        row["mean_true_regret"], -row["true_top_set_hit_rate"]
    )
    report = {
        "input_audit": input_audit,
        "scope": {
            "sender_index": str(args.sender_index),
            "groups": len(groups),
            "sequences": sorted(
                set(int(group["key"][0]) for group in groups)
            ),
            "folds": folds,
            "learned_send_actions": send_actions,
            "counterfactual_actions": all_actions,
            "training_labels": (
                "Full-information true rewards from training sequences only."
            ),
            "test_feedback": (
                "Only the selected action updates the policy on held-out "
                "sequences."
            ),
        },
        "baselines": baselines,
        "best_true_online_feedback": min(
            [x for x in rows if x["online_feedback_source"] == "true"],
            key=key,
        ) if "true" in feedback_sources else None,
        "best_proxy_online_feedback": min(
            [x for x in rows if x["online_feedback_source"] == "proxy"],
            key=key,
        ) if "proxy" in feedback_sources else None,
        "rows": rows,
    }

    output = Path(args.out_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Saved:", output)


if __name__ == "__main__":
    main()
