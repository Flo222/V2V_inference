#!/usr/bin/env python
"""Grouped-CV audit for a formula-preserving ARCE reward proxy."""

from __future__ import print_function

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from opencood.tools import audit_ap_proxy_v33_rich_box_groupcv as base


LABEL_PERCEPTION = "label_true_combined_perception"
LABEL_REWARD = "label_true_formula_reward"

RECEIVER_FEATURES = [
    "rx_q_recv_unit",
    "rx_q_cache_unit",
    "rx_q_eff_unit",
    "rx_q_recv_packet",
    "rx_q_cache_packet",
    "rx_q_eff_packet",
    "rx_num_source_packets",
    "rx_num_transmitted_source_packets",
    "rx_num_source_dropped_by_budget",
    "rx_num_received_packets",
    "rx_num_direct_received_source_packets",
    "rx_num_fec_recovered_source_packets",
    "rx_num_missing_source_packets",
    "rx_num_temporal_filled_packets",
    "rx_num_zero_filled_packets",
    "rx_num_total_units",
    "rx_num_current_recovered_units",
    "rx_num_temporal_filled_units",
    "rx_num_effective_recovered_units",
    "rx_cache_hit",
    "rx_tx_source_ratio",
    "rx_direct_receive_ratio",
    "rx_effective_unit_ratio",
]


def _is_no_send(row):
    action_id = str(row.get("action_id", ""))
    text = str(row.get("no_send", "")).strip().lower()
    return action_id.startswith("send0_") or text in {"1", "true", "yes"}


def _unique(names):
    result = []
    seen = set()
    for name in names:
        if name not in seen:
            result.append(name)
            seen.add(name)
    return result


def _augment_rows(rows, lambda_abs, lambda_delta, lambda_cost, reference_bytes):
    states = ("good", "medium", "bad")
    quants = ("fp16", "int8", "int4")
    senders = sorted({int(row["sender_index"]) for row in rows})

    for row in rows:
        no_send = _is_no_send(row)
        tx_bytes = float(row.get("tx_bytes", 0.0) or 0.0)
        cost_norm = tx_bytes / float(reference_bytes)

        if no_send:
            perception = 0.0
            reward = 0.0
        else:
            absolute = float(row[base.LABEL_ABSOLUTE])
            global_delta = float(row[base.LABEL_GLOBAL])
            perception = (
                float(lambda_abs) * absolute
                + float(lambda_delta) * global_delta
            )
            reward = perception - float(lambda_cost) * cost_norm

        row[LABEL_PERCEPTION] = str(perception)
        row[LABEL_REWARD] = str(reward)
        row["ctx_cache"] = str(float(row.get("cache", 0.0) or 0.0))
        row["ctx_no_send"] = "1.0" if no_send else "0.0"
        row["ctx_tx_norm_ref"] = str(cost_norm)

        state = str(row.get("channel_state", "")).strip().lower()
        for value in states:
            row["ctx_state_" + value] = "1.0" if state == value else "0.0"

        quant = str(row.get("quant_mode", "")).strip().lower()
        for value in quants:
            row["ctx_quant_" + value] = "1.0" if quant == value else "0.0"

        sender = int(row["sender_index"])
        for value in senders:
            row["ctx_sender_" + str(value)] = (
                "1.0" if sender == value else "0.0"
            )

    context_features = (
        ["ctx_state_" + value for value in states]
        + ["ctx_quant_" + value for value in quants]
        + ["ctx_cache", "ctx_no_send", "ctx_tx_norm_ref"]
        + ["ctx_sender_" + str(value) for value in senders]
    )
    return context_features


def _matrix(rows, feature_cols, label_col):
    return base._matrix(rows, feature_cols, label_col)


def _true_reward(rows):
    return np.asarray(
        [float(row[LABEL_REWARD]) for row in rows],
        dtype=np.float64,
    )


def _predict_formula_reward(
    model,
    rows,
    feature_cols,
    lambda_cost,
    reference_bytes,
):
    pred = np.zeros(len(rows), dtype=np.float64)
    send_indices = [
        index for index, row in enumerate(rows)
        if not _is_no_send(row)
    ]
    if send_indices:
        send_rows = [rows[index] for index in send_indices]
        x, _ = _matrix(send_rows, feature_cols, LABEL_PERCEPTION)
        perception_pred = model.predict(x)
        for local_index, row_index in enumerate(send_indices):
            tx_bytes = float(rows[row_index].get("tx_bytes", 0.0) or 0.0)
            pred[row_index] = (
                float(perception_pred[local_index])
                - float(lambda_cost)
                * tx_bytes
                / float(reference_bytes)
            )
    return pred


def _fit(
    rows,
    feature_cols,
    n_estimators,
    max_depth,
    min_samples_leaf,
    seed,
):
    send_rows = [row for row in rows if not _is_no_send(row)]
    x, y = _matrix(send_rows, feature_cols, LABEL_PERCEPTION)
    return base._fit_model(
        x,
        y,
        n_estimators,
        max_depth,
        min_samples_leaf,
        seed,
    )


def _fold_summary(fold_metrics):
    return base._aggregate_folds(fold_metrics, "combined_perception")


def _selection_score(summary):
    return base._selection_score(summary, "combined_perception")


def _cross_validate(
    rows,
    feature_cols,
    args,
    max_depth,
    min_samples_leaf,
):
    groups = np.asarray(
        [int(row["sequence_id"]) for row in rows],
        dtype=np.int64,
    )
    indices = np.arange(len(rows))
    splitter = GroupKFold(n_splits=int(args.folds))
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(indices, groups=groups)
    ):
        train_rows = [rows[int(index)] for index in train_idx]
        val_rows = [rows[int(index)] for index in val_idx]
        model = _fit(
            train_rows,
            feature_cols,
            int(args.selection_estimators),
            int(max_depth),
            int(min_samples_leaf),
            int(args.seed) + int(fold),
        )
        y_true = _true_reward(val_rows)
        y_pred = _predict_formula_reward(
            model,
            val_rows,
            feature_cols,
            float(args.lambda_cost),
            float(args.reference_bytes),
        )
        metrics = base._action_metrics(
            val_rows,
            y_true,
            y_pred,
            float(args.tie_tolerance),
        )
        metrics["fold"] = int(fold)
        metrics["train_sequences"] = sorted(
            {int(row["sequence_id"]) for row in train_rows}
        )
        metrics["validation_sequences"] = sorted(
            {int(row["sequence_id"]) for row in val_rows}
        )
        fold_metrics.append(metrics)

    summary = _fold_summary(fold_metrics)
    return {
        "max_depth": int(max_depth),
        "min_samples_leaf": int(min_samples_leaf),
        "summary": summary,
        "selection_score": float(_selection_score(summary)),
        "folds": fold_metrics,
    }


def _gate(metrics, args):
    checks = {
        "pairwise": (
            float(metrics["tie_aware_pairwise_ranking_accuracy"])
            >= float(args.min_pairwise)
        ),
        "frame_spearman": (
            float(metrics["frame_ranking_spearman_mean"])
            >= float(args.min_frame_spearman)
        ),
        "top_set": (
            float(metrics["top_set_match_rate"])
            >= float(args.min_top_set)
        ),
        "regret": (
            float(metrics["selected_action_regret_mean"])
            <= float(args.max_regret)
        ),
    }
    return {"checks": checks, "passed": bool(all(checks.values()))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--holdout_fraction", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--selection_estimators", type=int, default=200)
    parser.add_argument("--final_estimators", type=int, default=500)
    parser.add_argument("--max_depths", default="4,6,8")
    parser.add_argument("--min_samples_leaves", default="8,16,32")
    parser.add_argument("--tie_tolerance", type=float, default=0.01)
    parser.add_argument("--lambda_abs", type=float, default=0.1)
    parser.add_argument("--lambda_delta", type=float, default=3.0)
    parser.add_argument("--lambda_cost", type=float, default=0.3)
    parser.add_argument("--reference_bytes", type=float, default=337500.0)
    parser.add_argument("--min_pairwise", type=float, default=0.65)
    parser.add_argument("--min_frame_spearman", type=float, default=0.4)
    parser.add_argument("--min_top_set", type=float, default=0.5)
    parser.add_argument("--max_regret", type=float, default=0.03)
    args = parser.parse_args()

    rows = base._load_rows(Path(args.csv))
    context_features = _augment_rows(
        rows,
        args.lambda_abs,
        args.lambda_delta,
        args.lambda_cost,
        args.reference_bytes,
    )

    existing = base._feature_sets()
    head = existing["head_only"]
    detection_features = _unique(
        list(head["absolute"]) + list(head["delta"])
    )
    feature_sets = {
        "P0_detection": detection_features,
        "P1_context": context_features,
        "P2_detection_context": _unique(
            detection_features + context_features
        ),
    }
    missing_receiver = [
        name for name in RECEIVER_FEATURES if name not in rows[0]
    ]
    if missing_receiver:
        print(
            "receiver feature families skipped; missing columns:",
            ",".join(missing_receiver),
        )
    else:
        feature_sets.update(
            {
                "P3_receiver": list(RECEIVER_FEATURES),
                "P4_context_receiver": _unique(
                    context_features + RECEIVER_FEATURES
                ),
                "P5_detection_context_receiver": _unique(
                    detection_features
                    + context_features
                    + RECEIVER_FEATURES
                ),
            }
        )

    (
        development_rows,
        holdout_rows,
        development_sequences,
        holdout_sequences,
    ) = base._sequence_split(rows, float(args.holdout_fraction))

    depths = [int(value) for value in args.max_depths.split(",")]
    leaves = [int(value) for value in args.min_samples_leaves.split(",")]
    results = {}

    for family, feature_cols in feature_sets.items():
        candidates = []
        for depth in depths:
            for leaf in leaves:
                candidate = _cross_validate(
                    development_rows,
                    feature_cols,
                    args,
                    depth,
                    leaf,
                )
                candidates.append(candidate)
                print(
                    "{} depth={} leaf={} score={:.6f}".format(
                        family,
                        depth,
                        leaf,
                        candidate["selection_score"],
                    )
                )

        best = sorted(
            candidates,
            key=lambda item: item["selection_score"],
            reverse=True,
        )[0]
        model = _fit(
            development_rows,
            feature_cols,
            int(args.final_estimators),
            int(best["max_depth"]),
            int(best["min_samples_leaf"]),
            int(args.seed),
        )
        y_true = _true_reward(holdout_rows)
        y_pred = _predict_formula_reward(
            model,
            holdout_rows,
            feature_cols,
            args.lambda_cost,
            args.reference_bytes,
        )
        metrics = base._action_metrics(
            holdout_rows,
            y_true,
            y_pred,
            float(args.tie_tolerance),
        )
        results[family] = {
            "num_features": len(feature_cols),
            "feature_cols": feature_cols,
            "selected_hyperparameters": {
                "max_depth": best["max_depth"],
                "min_samples_leaf": best["min_samples_leaf"],
            },
            "cv_summary": best["summary"],
            "holdout_metrics": metrics,
            "holdout_gate": _gate(metrics, args),
        }

    report = {
        "status": "offline_diagnostic_only",
        "source_csv": str(args.csv),
        "receiver_features_available": not bool(missing_receiver),
        "missing_receiver_features": missing_receiver,
        "receiver_feature_policy": (
            "post_action_reward_proxy_only_not_bandit_context"
        ),
        "reward_formula": {
            "lambda_abs": args.lambda_abs,
            "lambda_delta": args.lambda_delta,
            "lambda_cost": args.lambda_cost,
            "reference_bytes": args.reference_bytes,
            "no_send_reward": 0.0,
        },
        "development_sequences": development_sequences,
        "holdout_sequences": holdout_sequences,
        "num_rows": len(rows),
        "results": results,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "combined_proxy_context_groupcv_report.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== holdout summary =====")
    for family, result in results.items():
        metrics = result["holdout_metrics"]
        print(
            "{} pairwise={:.4f} frame_spearman={:.4f} "
            "top_set={:.4f} regret={:.4f} passed={}".format(
                family,
                metrics["tie_aware_pairwise_ranking_accuracy"],
                metrics["frame_ranking_spearman_mean"],
                metrics["top_set_match_rate"],
                metrics["selected_action_regret_mean"],
                result["holdout_gate"]["passed"],
            )
        )
    print("saved:", output)


if __name__ == "__main__":
    main()
