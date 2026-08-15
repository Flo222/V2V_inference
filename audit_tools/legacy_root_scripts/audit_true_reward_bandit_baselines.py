#!/usr/bin/env python3
"""Online true-reward baselines for ARCE six-action counterfactual data."""

from __future__ import print_function

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import audit_context_reward_learnability as learn


def values(text, cast):
    result = [cast(item.strip()) for item in str(text).split(",") if item.strip()]
    if not result:
        raise ValueError("empty parameter list")
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--sender_index", default="1")
    parser.add_argument("--reference_bytes", type=float, default=337500.0)
    parser.add_argument("--lambda_abs", type=float, default=0.1)
    parser.add_argument("--lambda_delta", type=float, default=3.0)
    parser.add_argument("--lambda_cost", type=float, default=0.3)
    parser.add_argument("--tie_tolerance", type=float, default=0.01)
    parser.add_argument("--seeds", default="2026,2027,2028,2029,2030")
    parser.add_argument("--ucb_cs", default="0,0.02,0.05,0.1,0.2,0.5,1")
    parser.add_argument("--epsilons", default="0,0.02,0.05,0.1,0.2")
    return parser.parse_args()


def choose_unseen(counts, rng):
    unseen = [index for index, count in enumerate(counts) if count == 0]
    return rng.choice(unseen) if unseen else None


def replay(groups, actions, mode, parameter, by_state, seed, tolerance):
    rng = random.Random(seed)
    count = defaultdict(lambda: np.zeros(len(actions), dtype=np.int64))
    total = defaultdict(lambda: np.zeros(len(actions), dtype=np.float64))
    records = []

    for group in groups:
        bucket = group["state"] if by_state else "global"
        counts = count[bucket]
        totals = total[bucket]
        unseen = choose_unseen(counts, rng)
        if unseen is not None:
            selected = unseen
            reason = "initialization"
        else:
            means = totals / np.maximum(counts, 1)
            if mode == "ucb":
                steps = max(int(np.sum(counts)), 1)
                bonus = float(parameter) * np.sqrt(
                    np.log(float(steps) + 1.0) / counts
                )
                selected = int(np.argmax(means + bonus))
                reason = "ucb"
            elif mode == "epsilon_greedy":
                if rng.random() < float(parameter):
                    selected = rng.randrange(len(actions))
                    reason = "epsilon"
                else:
                    selected = int(np.argmax(means))
                    reason = "greedy"
            else:
                raise ValueError("unknown mode {}".format(mode))

        rewards = np.asarray(
            [group["rewards"][action] for action in actions], dtype=np.float64
        )
        observed = float(rewards[selected])
        counts[selected] += 1
        totals[selected] += observed
        best = float(np.max(rewards))
        regret = max(0.0, best - observed)
        records.append(
            {
                "regret": regret,
                "top_hit": regret <= float(tolerance),
                "selected": selected,
                "reason": reason,
            }
        )

    quarter = max(1, len(records) // 4)
    late = records[-quarter:]
    return {
        "mean_regret": float(np.mean([row["regret"] for row in records])),
        "late_mean_regret": float(np.mean([row["regret"] for row in late])),
        "late_top_set_hit_rate": float(
            np.mean([row["top_hit"] for row in late])
        ),
        "late_action_counter": dict(
            Counter(actions[row["selected"]] for row in late)
        ),
    }


def aggregate(runs, random_regret, fixed_regret):
    late = [run["late_mean_regret"] for run in runs]
    hits = [run["late_top_set_hit_rate"] for run in runs]
    return {
        "late_mean_regret": float(np.mean(late)),
        "late_regret_min": float(np.min(late)),
        "late_regret_max": float(np.max(late)),
        "late_top_set_hit_rate": float(np.mean(hits)),
        "late_regret_vs_random": float(np.mean(late) / max(random_regret, 1e-12)),
        "late_regret_vs_best_fixed": float(
            np.mean(late) / max(fixed_regret, 1e-12)
        ),
        "per_seed": runs,
    }


def main():
    args = parse_args()
    groups, actions = learn.load_dataset(args.csv, args)
    reward_matrix = np.asarray(
        [[group["rewards"][action] for action in actions] for group in groups],
        dtype=np.float64,
    )
    random_regret = learn.expected_random_regret(reward_matrix)
    fixed_index = int(np.argmax(np.mean(reward_matrix, axis=0)))
    fixed_regret = float(
        np.mean(np.max(reward_matrix, axis=1) - reward_matrix[:, fixed_index])
    )
    seeds = values(args.seeds, int)

    rows = []
    specs = []
    for by_state in (False, True):
        for value in values(args.ucb_cs, float):
            specs.append(("ucb", value, by_state))
        for value in values(args.epsilons, float):
            specs.append(("epsilon_greedy", value, by_state))

    for mode, parameter, by_state in specs:
        runs = [
            replay(
                groups,
                actions,
                mode,
                parameter,
                by_state,
                seed,
                args.tie_tolerance,
            )
            for seed in seeds
        ]
        row = {
            "mode": mode,
            "parameter": parameter,
            "conditioning": "channel_state" if by_state else "global",
        }
        row.update(aggregate(runs, random_regret, fixed_regret))
        rows.append(row)
        print(
            "{} {} parameter={:.3f} late_regret={:.6f} vs_random={:.4f}".format(
                row["conditioning"],
                mode,
                parameter,
                row["late_mean_regret"],
                row["late_regret_vs_random"],
            ),
            flush=True,
        )

    ranking = sorted(rows, key=lambda row: row["late_mean_regret"])
    report = {
        "scope": {
            "groups": len(groups),
            "states": dict(Counter(group["state"] for group in groups)),
            "actions": actions,
            "feedback": "selected-action true reward only",
        },
        "baselines": {
            "random_expected_mean_regret": random_regret,
            "global_best_fixed_action": actions[fixed_index],
            "global_best_fixed_mean_regret": fixed_regret,
        },
        "best": ranking[0],
        "ranking": ranking,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n===== best online baseline =====")
    print(json.dumps(ranking[0], indent=2, ensure_ascii=False))
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
