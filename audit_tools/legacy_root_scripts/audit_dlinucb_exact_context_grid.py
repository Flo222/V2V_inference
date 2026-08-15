#!/usr/bin/env python3
"""Parameter grid for the ARCE exact-context D-LinUCB replay audit."""

from __future__ import print_function

import argparse
import json
import sys
from pathlib import Path

import audit_dlinucb_exact_context_replay as replay


def csv_values(text, cast):
    values = [cast(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("parameter grid cannot be empty")
    return values


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
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--seeds", default="2026,2027,2028,2029,2030")
    parser.add_argument("--betas", default="0,0.05,0.1,0.2,0.5,1.0")
    parser.add_argument("--discounts", default="1,0.999,0.995,0.99,0.975,0.95")
    return parser.parse_args()


def compact(item):
    return {
        "late_true_regret": item["last_quarter_mean_true_regret"]["mean"],
        "late_feedback_regret": item[
            "last_quarter_mean_feedback_regret"
        ]["mean"],
        "late_true_top_set_hit_rate": item[
            "last_quarter_true_top_set_hit_rate"
        ]["mean"],
        "late_true_regret_vs_random": item["late_true_regret_vs_random"],
        "late_true_regret_vs_best_fixed": item[
            "late_true_regret_vs_best_fixed"
        ],
        "random_expected_mean_true_regret": item["baselines"][
            "random_expected_mean_true_regret"
        ],
        "best_fixed_action": item["baselines"]["best_fixed_action"],
        "best_fixed_mean_true_regret": item["baselines"][
            "best_fixed_mean_true_regret"
        ],
    }


def frame_gap_summary(groups):
    gaps = []
    previous = None
    for group in groups:
        current = (group["key"][0], group["key"][1])
        if previous is not None and current[0] == previous[0]:
            gap = current[1] - previous[1]
            if gap > 0:
                gaps.append(gap)
        previous = current
    return replay.summary(gaps)


def main():
    args = parse_args()
    repo_root = Path.cwd()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from opencood.methods.arce.policies.discounted_linucb import DiscountedLinUCB

    # load_groups reads the reward parameters from args.
    args.beta = 1.0
    args.markov_discount = 0.995
    args.production_feedback_weight = "channel_quality"
    groups, all_actions, send_actions, input_audit = replay.load_groups(
        args.csv, args
    )
    groups = [
        group
        for group in groups
        if group["sender_index"] == str(args.sender_index)
    ]
    if not groups:
        raise ValueError("No groups for sender_index={}".format(args.sender_index))

    betas = csv_values(args.betas, float)
    discounts = csv_values(args.discounts, float)
    seeds = csv_values(args.seeds, int)
    baselines = replay.random_and_fixed_baselines(groups, send_actions)
    rows = []

    specs = (
        ("true", "none"),
        ("proxy", "none"),
        ("proxy", "channel_quality"),
    )
    for beta in betas:
        args.beta = beta
        for discount in discounts:
            for feedback_source, weight_mode in specs:
                runs = [
                    replay.replay_once(
                        DiscountedLinUCB,
                        groups,
                        send_actions,
                        feedback_source,
                        discount,
                        weight_mode,
                        seed,
                        args,
                    )
                    for seed in seeds
                ]
                result = replay.aggregate_replays(
                    "grid", runs, baselines
                )
                row = {
                    "beta": beta,
                    "discount": discount,
                    "feedback_source": feedback_source,
                    "feedback_weight_mode": weight_mode,
                }
                row.update(compact(result))
                rows.append(row)
                print(
                    "beta={:.3f} gamma={:.4f} feedback={}/{} "
                    "late_true_regret={:.6f} vs_random={:.4f}".format(
                        beta,
                        discount,
                        feedback_source,
                        weight_mode,
                        row["late_true_regret"],
                        row["late_true_regret_vs_random"],
                    ),
                    flush=True,
                )

    true_rows = [row for row in rows if row["feedback_source"] == "true"]
    proxy_rows = [
        row
        for row in rows
        if row["feedback_source"] == "proxy"
        and row["feedback_weight_mode"] == "none"
    ]
    weighted_rows = [
        row
        for row in rows
        if row["feedback_source"] == "proxy"
        and row["feedback_weight_mode"] == "channel_quality"
    ]

    key = lambda row: (row["late_true_regret"], -row["late_true_top_set_hit_rate"])
    report = {
        "input_audit": input_audit,
        "scope": {
            "sender_index": str(args.sender_index),
            "groups": len(groups),
            "counterfactual_actions": all_actions,
            "learned_send_actions": send_actions,
            "context_fields": list(replay.CONTEXT_FIELDS),
            "frame_gap": frame_gap_summary(groups),
            "discount_semantics": (
                "Discount is applied once per audited group. Intermediate "
                "online updates between sampled frames are unavailable, so "
                "this is a sparse-stream sensitivity audit, not an exact "
                "reconstruction of the original online trajectory."
            ),
        },
        "grid": {
            "betas": betas,
            "discounts": discounts,
            "seeds": seeds,
        },
        "best_true_feedback": min(true_rows, key=key),
        "best_proxy_feedback": min(proxy_rows, key=key),
        "best_proxy_channel_weighted": min(weighted_rows, key=key),
        "true_feedback_ranking": sorted(true_rows, key=key),
        "proxy_feedback_ranking": sorted(proxy_rows, key=key),
        "proxy_channel_weighted_ranking": sorted(weighted_rows, key=key),
        "rows": rows,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n===== best settings by late true regret =====")
    for name in (
        "best_true_feedback",
        "best_proxy_feedback",
        "best_proxy_channel_weighted",
    ):
        print(name, json.dumps(report[name], ensure_ascii=False))
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
