#!/usr/bin/env python3
"""Full-information learnability audit for ARCE action rewards.

This is an upper-bound diagnostic, not an online policy evaluation.  Every
training frame contributes labels for all six send actions.  If a model cannot
beat context-free baselines under this favorable setting, D-LinUCB cannot be
expected to learn a stable contextual policy from the same context and reward.
"""

from __future__ import print_function

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


NO_SEND_PREFIX = "send0_"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--sender_index", default="1")
    parser.add_argument("--reference_bytes", type=float, default=337500.0)
    parser.add_argument("--lambda_abs", type=float, default=0.1)
    parser.add_argument("--lambda_delta", type=float, default=3.0)
    parser.add_argument("--lambda_cost", type=float, default=0.3)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--tie_tolerance", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def number(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError("{} must be numeric, got {!r}".format(name, value))
    if not math.isfinite(result):
        raise ValueError("{} must be finite, got {!r}".format(name, value))
    return result


def true_reward(row, args):
    quality = number(row["true_quality_mean_0357"], "true_quality")
    delta = number(
        row["label_true_global_delta_quality_mean_0357"], "true_delta"
    )
    tx_bytes = number(row["tx_bytes"], "tx_bytes")
    return (
        float(args.lambda_abs) * quality
        + float(args.lambda_delta) * delta
        - float(args.lambda_cost) * tx_bytes / float(args.reference_bytes)
    )


def load_dataset(path, args):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("CSV is empty")

    required = {
        "sequence_id",
        "sequence_frame_idx",
        "sender_index",
        "action_id",
        "channel_state",
        "true_quality_mean_0357",
        "label_true_global_delta_quality_mean_0357",
        "tx_bytes",
        "decision_context_available",
    }
    required.update("decision_context_" + name for name in CONTEXT_FIELDS)
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise ValueError("CSV missing fields: {}".format(missing))

    grouped = defaultdict(list)
    for row in rows:
        if str(row["sender_index"]) != str(args.sender_index):
            continue
        key = (
            int(float(row["sequence_id"])),
            int(float(row["sequence_frame_idx"])),
            str(row["sender_index"]),
        )
        grouped[key].append(row)
    if not grouped:
        raise ValueError("No rows for sender_index={}".format(args.sender_index))

    action_sets = Counter()
    parsed = []
    for key in sorted(grouped):
        group = grouped[key]
        send_rows = [
            row
            for row in group
            if not str(row["action_id"]).startswith(NO_SEND_PREFIX)
        ]
        actions = tuple(sorted(str(row["action_id"]) for row in send_rows))
        action_sets[actions] += 1
        if len(actions) != 6 or len(set(actions)) != 6:
            raise ValueError("Group {} lacks six send actions".format(key))

        contexts = []
        for row in group:
            if str(row["decision_context_available"]).lower() not in (
                "1",
                "true",
                "yes",
            ):
                raise ValueError("Missing context in group {}".format(key))
            contexts.append(
                np.asarray(
                    [
                        number(
                            row["decision_context_" + name],
                            "decision_context_" + name,
                        )
                        for name in CONTEXT_FIELDS
                    ],
                    dtype=np.float64,
                )
            )
        context = contexts[0]
        spread = max(
            float(np.max(np.abs(value - context))) for value in contexts
        )
        if spread > 1e-9:
            raise ValueError("Context varies within group {}".format(key))

        by_action = {str(row["action_id"]): row for row in send_rows}
        parsed.append(
            {
                "key": key,
                "sequence_id": key[0],
                "state": str(group[0]["channel_state"]).strip().lower(),
                "context": context,
                "rewards": {
                    action: true_reward(by_action[action], args)
                    for action in actions
                },
            }
        )
    if len(action_sets) != 1:
        raise ValueError("Inconsistent action sets: {}".format(action_sets))
    return parsed, list(next(iter(action_sets.keys())))


def expected_random_regret(reward_matrix):
    best = np.max(reward_matrix, axis=1)
    return float(np.mean(best - np.mean(reward_matrix, axis=1)))


def selection_metrics(reward_matrix, predicted, tolerance):
    selected = np.argmax(predicted, axis=1)
    best = np.max(reward_matrix, axis=1)
    chosen = reward_matrix[np.arange(len(selected)), selected]
    regrets = np.maximum(best - chosen, 0.0)
    top_hits = regrets <= float(tolerance)
    pairwise_correct = 0
    pairwise_total = 0
    frame_spearman = []
    for truth, estimate in zip(reward_matrix, predicted):
        corr = spearmanr(truth, estimate).correlation
        if corr is not None and math.isfinite(float(corr)):
            frame_spearman.append(float(corr))
        for left in range(len(truth)):
            for right in range(left + 1, len(truth)):
                true_diff = truth[left] - truth[right]
                pred_diff = estimate[left] - estimate[right]
                if abs(true_diff) <= float(tolerance):
                    continue
                pairwise_total += 1
                pairwise_correct += int(true_diff * pred_diff > 0.0)
    return {
        "mean_regret": float(np.mean(regrets)),
        "p90_regret": float(np.quantile(regrets, 0.9)),
        "top_set_hit_rate": float(np.mean(top_hits)),
        "pairwise_accuracy": (
            float(pairwise_correct / pairwise_total)
            if pairwise_total
            else None
        ),
        "mean_frame_spearman": (
            float(np.mean(frame_spearman)) if frame_spearman else None
        ),
        "selected_action_indices": selected.tolist(),
    }


def fit_predict_per_action(model_factory, x_train, y_train, x_test):
    predictions = []
    for action_index in range(y_train.shape[1]):
        model = model_factory()
        model.fit(x_train, y_train[:, action_index])
        predictions.append(model.predict(x_test))
    return np.column_stack(predictions)


def constant_predictions(y_train, n_test):
    return np.tile(np.mean(y_train, axis=0), (n_test, 1))


def state_predictions(train_states, y_train, test_states):
    global_means = np.mean(y_train, axis=0)
    by_state = {}
    for state in sorted(set(train_states)):
        indices = [i for i, value in enumerate(train_states) if value == state]
        by_state[state] = np.mean(y_train[indices], axis=0)
    return np.vstack([by_state.get(state, global_means) for state in test_states])


def main():
    args = parse_args()
    groups, actions = load_dataset(args.csv, args)
    x = np.vstack([group["context"] for group in groups])
    y = np.asarray(
        [[group["rewards"][action] for action in actions] for group in groups],
        dtype=np.float64,
    )
    sequence_groups = np.asarray(
        [group["sequence_id"] for group in groups], dtype=np.int64
    )
    states = [group["state"] for group in groups]
    unique_sequences = sorted(set(sequence_groups.tolist()))
    folds = min(int(args.folds), len(unique_sequences))
    if folds < 2:
        raise ValueError("Need at least two sequence groups")

    model_factories = {
        "ridge_alpha_0.1": lambda: make_pipeline(
            StandardScaler(), Ridge(alpha=0.1)
        ),
        "ridge_alpha_1": lambda: make_pipeline(
            StandardScaler(), Ridge(alpha=1.0)
        ),
        "ridge_alpha_10": lambda: make_pipeline(
            StandardScaler(), Ridge(alpha=10.0)
        ),
        "rf_depth_4_leaf_8": lambda: RandomForestRegressor(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=8,
            random_state=int(args.seed),
            n_jobs=-1,
        ),
        "rf_depth_8_leaf_8": lambda: RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=8,
            random_state=int(args.seed),
            n_jobs=-1,
        ),
        "rf_depth_8_leaf_16": lambda: RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=16,
            random_state=int(args.seed),
            n_jobs=-1,
        ),
    }
    predictions = {
        "train_mean_fixed": np.zeros_like(y),
        "train_state_fixed": np.zeros_like(y),
    }
    predictions.update({name: np.zeros_like(y) for name in model_factories})

    splitter = GroupKFold(n_splits=folds)
    fold_rows = []
    for fold_index, (train, test) in enumerate(
        splitter.split(x, y, groups=sequence_groups)
    ):
        predictions["train_mean_fixed"][test] = constant_predictions(
            y[train], len(test)
        )
        predictions["train_state_fixed"][test] = state_predictions(
            [states[i] for i in train], y[train], [states[i] for i in test]
        )
        for name, factory in model_factories.items():
            predictions[name][test] = fit_predict_per_action(
                factory, x[train], y[train], x[test]
            )
        fold_rows.append(
            {
                "fold": fold_index,
                "train_sequences": sorted(set(sequence_groups[train].tolist())),
                "test_sequences": sorted(set(sequence_groups[test].tolist())),
                "train_groups": len(train),
                "test_groups": len(test),
            }
        )

    random_regret = expected_random_regret(y)
    results = {}
    for name, prediction in predictions.items():
        metrics = selection_metrics(y, prediction, args.tie_tolerance)
        metrics.pop("selected_action_indices", None)
        metrics["regret_vs_random"] = metrics["mean_regret"] / max(
            random_regret, 1e-12
        )
        results[name] = metrics

    ranking = sorted(results, key=lambda name: results[name]["mean_regret"])
    best_fixed_index = int(np.argmax(np.mean(y, axis=0)))
    report = {
        "scope": {
            "groups": len(groups),
            "sequences": unique_sequences,
            "actions": actions,
            "context_fields": list(CONTEXT_FIELDS),
            "folds": folds,
            "cross_validation": "GroupKFold by sequence_id",
            "note": (
                "All six action labels are visible during training. Results "
                "are a favorable learnability upper bound, not online regret."
            ),
        },
        "reward": {
            "lambda_abs": args.lambda_abs,
            "lambda_delta": args.lambda_delta,
            "lambda_cost": args.lambda_cost,
            "reference_bytes": args.reference_bytes,
        },
        "baselines": {
            "random_expected_mean_regret": random_regret,
            "global_best_fixed_action": actions[best_fixed_index],
            "global_best_fixed_mean_regret": float(
                np.mean(np.max(y, axis=1) - y[:, best_fixed_index])
            ),
        },
        "folds": fold_rows,
        "model_ranking": ranking,
        "results": results,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("===== exact-context reward learnability =====")
    print("groups:", len(groups), "sequences:", len(unique_sequences))
    print("random regret: {:.6f}".format(random_regret))
    print(
        "global best fixed: {} regret={:.6f}".format(
            report["baselines"]["global_best_fixed_action"],
            report["baselines"]["global_best_fixed_mean_regret"],
        )
    )
    for name in ranking:
        item = results[name]
        print(
            "{} regret={:.6f} vs_random={:.4f} top_set={:.4f} "
            "pairwise={:.4f} frame_spearman={:.4f}".format(
                name,
                item["mean_regret"],
                item["regret_vs_random"],
                item["top_set_hit_rate"],
                item["pairwise_accuracy"],
                item["mean_frame_spearman"],
            )
        )
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
