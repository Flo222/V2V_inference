#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from opencood.methods.arce.reward.proxy.ap_proxy_features import (
    DENSE_AP_PROXY_FEATURES,
    PAIRED_SPATIAL_AP_PROXY_FEATURES,
    PSM_V3_EXTRA_FEATURES,
    REGRESSION_AP_PROXY_FEATURES,
)


LABEL_ABSOLUTE = "true_quality_mean_0357"
LABEL_GLOBAL = "label_true_global_delta_quality_mean_0357"
LABEL_MARGINAL = "label_true_delta_quality_mean_0357"


def _prefixed(prefix: str, names: Sequence[str]) -> List[str]:
    return [prefix + name for name in names]


def _paired_head_features(head_names: Sequence[str]) -> List[str]:
    result: List[str] = []
    for prefix in ("collab_", "ego_", "diff_"):
        result.extend(_prefixed(prefix, head_names))
    return result


def _feature_sets() -> Dict[str, Dict[str, List[str]]]:
    robust_psm = [
        name
        for name in list(DENSE_AP_PROXY_FEATURES) + list(PSM_V3_EXTRA_FEATURES)
        if name not in {
            "dense_max_conf",
            "dense_p99",
        }
    ]
    robust_reg = [
        name
        for name in REGRESSION_AP_PROXY_FEATURES
        if name not in {
            "reg_abs_max",
            "reg_abs_p99",
        }
    ]
    robust_spatial = [
        name
        for name in PAIRED_SPATIAL_AP_PROXY_FEATURES
        if not name.startswith("reg_diff_")
        and name != "spatial_prob_max_abs"
    ]
    robust_paired = [
        name
        for name in PAIRED_SPATIAL_AP_PROXY_FEATURES
        if name not in {"spatial_prob_max_abs", "reg_diff_abs_max"}
    ]
    full_head = (
        list(DENSE_AP_PROXY_FEATURES)
        + list(PSM_V3_EXTRA_FEATURES)
        + list(REGRESSION_AP_PROXY_FEATURES)
    )
    return {
        "v2_psm": {
            "absolute": _prefixed("collab_", DENSE_AP_PROXY_FEATURES),
            "delta": _paired_head_features(DENSE_AP_PROXY_FEATURES),
        },
        "robust_psm": {
            "absolute": _prefixed("collab_", robust_psm),
            "delta": _paired_head_features(robust_psm) + robust_spatial,
        },
        "robust_psm_rm": {
            "absolute": _prefixed("collab_", robust_psm + robust_reg),
            "delta": (
                _paired_head_features(robust_psm + robust_reg)
                + robust_paired
            ),
        },
        "full_v3": {
            "absolute": _prefixed("collab_", full_head),
            "delta": (
                _paired_head_features(full_head)
                + list(PAIRED_SPATIAL_AP_PROXY_FEATURES)
            ),
        },
    }


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _matrix(
    rows: Sequence[Dict[str, str]],
    feature_cols: Sequence[str],
    label_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(
        [[float(row[name]) for name in feature_cols] for row in rows],
        dtype=np.float32,
    )
    y = np.asarray([float(row[label_col]) for row in rows], dtype=np.float32)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Proxy dataset contains non-finite values.")
    return x, y


def _rankdata(values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(values), dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _correlation(x: Iterable[float], y: Iterable[float]) -> float:
    x = np.asarray(list(x), dtype=np.float64)
    y = np.asarray(list(y), dtype=np.float64)
    if len(x) < 2 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": _correlation(y_true, y_pred),
        "spearman": _correlation(_rankdata(y_true), _rankdata(y_pred)),
        "true_mean": float(np.mean(y_true)),
        "pred_mean": float(np.mean(y_pred)),
        "mean_bias": float(np.mean(y_pred) - np.mean(y_true)),
    }


def _majority_sign_accuracy(values: np.ndarray, eps: float = 1e-9) -> float:
    comparable = np.abs(values) > float(eps)
    signs = np.sign(values[comparable])
    if signs.size == 0:
        return float("nan")
    positive = float(np.mean(signs > 0.0))
    return max(positive, 1.0 - positive)


def _frame_key(row: Dict[str, str]) -> Tuple[int, int, int]:
    return (
        int(row["sequence_id"]),
        int(row["frame_idx"]),
        int(row["sender_index"]),
    )


def _action_metrics(
    rows: Sequence[Dict[str, str]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tie_tolerance: float,
) -> Dict[str, Any]:
    by_frame: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_frame[_frame_key(row)].append(index)

    pair_correct = 0
    pair_total = 0
    exact_top1_hits = 0
    top_set_hits = 0
    frame_spearman: List[float] = []
    regrets: List[float] = []
    for indices in by_frame.values():
        frame_true = y_true[indices]
        frame_pred = y_pred[indices]
        predicted_local = int(np.argmax(frame_pred))
        true_local = int(np.argmax(frame_true))
        true_best = float(np.max(frame_true))
        selected_true = float(frame_true[predicted_local])
        exact_top1_hits += int(predicted_local == true_local)
        top_set_hits += int(
            selected_true >= true_best - float(tie_tolerance)
        )
        regrets.append(max(0.0, true_best - selected_true))

        corr = _correlation(_rankdata(frame_true), _rankdata(frame_pred))
        if math.isfinite(corr):
            frame_spearman.append(corr)
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                true_diff = float(frame_true[left] - frame_true[right])
                if abs(true_diff) <= float(tie_tolerance):
                    continue
                pred_diff = float(frame_pred[left] - frame_pred[right])
                pair_total += 1
                pair_correct += int(true_diff * pred_diff > 0.0)

    comparable = np.abs(y_true) > 1e-9
    sign_accuracy = (
        float(np.mean(np.sign(y_true[comparable]) == np.sign(y_pred[comparable])))
        if int(np.sum(comparable)) > 0
        else float("nan")
    )
    majority = _majority_sign_accuracy(y_true)
    return {
        "regression": _regression_metrics(y_true, y_pred),
        "sign_accuracy": sign_accuracy,
        "majority_sign_baseline": majority,
        "sign_lift": sign_accuracy - majority,
        "sign_comparable": int(np.sum(comparable)),
        "tie_tolerance": float(tie_tolerance),
        "tie_aware_pairwise_ranking_accuracy": (
            float(pair_correct) / float(pair_total)
            if pair_total else float("nan")
        ),
        "tie_aware_pairwise_comparisons": int(pair_total),
        "frame_ranking_spearman_mean": (
            float(np.mean(frame_spearman))
            if frame_spearman else float("nan")
        ),
        "exact_top1_match_rate": (
            float(exact_top1_hits) / float(len(by_frame))
            if by_frame else float("nan")
        ),
        "top_set_match_rate": (
            float(top_set_hits) / float(len(by_frame))
            if by_frame else float("nan")
        ),
        "selected_action_regret_mean": (
            float(np.mean(regrets)) if regrets else float("nan")
        ),
        "selected_action_regret_p90": (
            float(np.quantile(regrets, 0.90))
            if regrets else float("nan")
        ),
        "num_frames": int(len(by_frame)),
    }


def _fit_model(
    x: np.ndarray,
    y: np.ndarray,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    seed: int,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_leaf=int(min_samples_leaf),
        random_state=int(seed),
        n_jobs=-1,
    )
    model.fit(x, y)
    return model


def _finite_mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _aggregate_folds(
    fold_metrics: Sequence[Dict[str, Any]],
    target: str,
) -> Dict[str, float]:
    if target == "absolute":
        return {
            key: _finite_mean(
                [fold["regression"][key] for fold in fold_metrics]
            )
            for key in (
                "mae",
                "rmse",
                "r2",
                "pearson",
                "spearman",
                "mean_bias",
            )
        }
    return {
        "mae": _finite_mean(
            [fold["regression"]["mae"] for fold in fold_metrics]
        ),
        "r2": _finite_mean(
            [fold["regression"]["r2"] for fold in fold_metrics]
        ),
        "pearson": _finite_mean(
            [fold["regression"]["pearson"] for fold in fold_metrics]
        ),
        "spearman": _finite_mean(
            [fold["regression"]["spearman"] for fold in fold_metrics]
        ),
        "sign_lift": _finite_mean(
            [fold["sign_lift"] for fold in fold_metrics]
        ),
        "pairwise": _finite_mean(
            [
                fold["tie_aware_pairwise_ranking_accuracy"]
                for fold in fold_metrics
            ]
        ),
        "frame_spearman": _finite_mean(
            [fold["frame_ranking_spearman_mean"] for fold in fold_metrics]
        ),
        "top_set": _finite_mean(
            [fold["top_set_match_rate"] for fold in fold_metrics]
        ),
        "regret": _finite_mean(
            [fold["selected_action_regret_mean"] for fold in fold_metrics]
        ),
    }


def _selection_score(summary: Dict[str, float], target: str) -> float:
    if target == "absolute":
        pearson = summary.get("pearson", float("nan"))
        spearman = summary.get("spearman", float("nan"))
        mae = summary.get("mae", float("inf"))
        if not math.isfinite(pearson) or not math.isfinite(spearman):
            return float("-inf")
        return float(pearson + 0.25 * spearman - 0.5 * mae)

    pairwise = summary.get("pairwise", float("nan"))
    frame_spearman = summary.get("frame_spearman", float("nan"))
    top_set = summary.get("top_set", float("nan"))
    regret = summary.get("regret", float("inf"))
    if (
        not math.isfinite(pairwise)
        or not math.isfinite(frame_spearman)
        or not math.isfinite(top_set)
    ):
        return float("-inf")
    return float(
        pairwise
        + 0.25 * frame_spearman
        + 0.25 * top_set
        - regret
    )


def _candidate_grid(args: argparse.Namespace) -> List[Tuple[int, int]]:
    depths = [int(value) for value in str(args.max_depths).split(",")]
    leaves = [
        int(value) for value in str(args.min_samples_leaves).split(",")
    ]
    return [(depth, leaf) for depth in depths for leaf in leaves]


def _sequence_split(
    rows: Sequence[Dict[str, str]],
    holdout_fraction: float,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[int], List[int]]:
    sequence_ids = sorted({int(row["sequence_id"]) for row in rows})
    if len(sequence_ids) < 6:
        raise ValueError("v3.1 audit requires at least six sequences.")
    holdout_count = max(
        1,
        int(math.ceil(len(sequence_ids) * float(holdout_fraction))),
    )
    development_sequences = sequence_ids[:-holdout_count]
    holdout_sequences = sequence_ids[-holdout_count:]
    development_set = set(development_sequences)
    holdout_set = set(holdout_sequences)
    return (
        [
            row for row in rows
            if int(row["sequence_id"]) in development_set
        ],
        [
            row for row in rows
            if int(row["sequence_id"]) in holdout_set
        ],
        development_sequences,
        holdout_sequences,
    )


def _cross_validate_candidate(
    rows: Sequence[Dict[str, str]],
    feature_cols: Sequence[str],
    label_col: str,
    target: str,
    args: argparse.Namespace,
    max_depth: int,
    min_samples_leaf: int,
) -> Dict[str, Any]:
    groups = np.asarray(
        [int(row["sequence_id"]) for row in rows],
        dtype=np.int64,
    )
    splitter = GroupKFold(n_splits=int(args.folds))
    fold_metrics: List[Dict[str, Any]] = []
    row_indices = np.arange(len(rows))
    for fold_index, (train_idx, val_idx) in enumerate(
        splitter.split(row_indices, groups=groups)
    ):
        train_rows = [rows[int(index)] for index in train_idx]
        val_rows = [rows[int(index)] for index in val_idx]
        train_x, train_y = _matrix(train_rows, feature_cols, label_col)
        val_x, val_y = _matrix(val_rows, feature_cols, label_col)
        model = _fit_model(
            train_x,
            train_y,
            int(args.selection_estimators),
            max_depth,
            min_samples_leaf,
            int(args.seed) + fold_index,
        )
        pred = model.predict(val_x)
        metrics = (
            {"regression": _regression_metrics(val_y, pred)}
            if target == "absolute"
            else _action_metrics(
                val_rows,
                val_y,
                pred,
                float(args.tie_tolerance),
            )
        )
        metrics["fold"] = int(fold_index)
        metrics["train_sequences"] = sorted(
            {int(row["sequence_id"]) for row in train_rows}
        )
        metrics["validation_sequences"] = sorted(
            {int(row["sequence_id"]) for row in val_rows}
        )
        fold_metrics.append(metrics)

    summary = _aggregate_folds(fold_metrics, target)
    return {
        "max_depth": int(max_depth),
        "min_samples_leaf": int(min_samples_leaf),
        "folds": fold_metrics,
        "summary": summary,
        "selection_score": _selection_score(summary, target),
    }


def _save_model(
    path: Path,
    model: Any,
    feature_cols: Sequence[str],
    label_col: str,
    proxy_type: str,
    feature_set: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(
            {
                "model": model,
                "feature_cols": list(feature_cols),
                "label_col": str(label_col),
                "proxy_type": str(proxy_type),
                "feature_definition": "canonical_psm_rm_v31_groupcv",
                "feature_set": str(feature_set),
                "input_schema_version": 3,
                "experimental_do_not_enable_without_holdout_gate": True,
            },
            handle,
        )


def _holdout_gate(
    absolute: Dict[str, Any],
    global_delta: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    abs_reg = absolute["regression"]
    delta_reg = global_delta["regression"]
    checks = {
        "absolute_r2_positive": float(abs_reg["r2"]) > 0.0,
        "absolute_pearson": (
            float(abs_reg["pearson"]) >= float(args.min_abs_pearson)
        ),
        "delta_r2_positive": float(delta_reg["r2"]) > 0.0,
        "delta_pearson": (
            float(delta_reg["pearson"]) >= float(args.min_delta_pearson)
        ),
        "global_sign_lift_positive": (
            float(global_delta["sign_lift"]) > 0.0
        ),
        "pairwise": (
            float(global_delta["tie_aware_pairwise_ranking_accuracy"])
            >= float(args.min_pairwise)
        ),
        "frame_spearman": (
            float(global_delta["frame_ranking_spearman_mean"])
            >= float(args.min_frame_spearman)
        ),
        "top_set": (
            float(global_delta["top_set_match_rate"])
            >= float(args.min_top_set)
        ),
        "regret": (
            float(global_delta["selected_action_regret_mean"])
            <= float(args.max_regret)
        ),
    }
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "note": (
            "This is a reference holdout gate. The same sequences were already "
            "inspected during v3 development, so a separate untouched test is "
            "still required for final paper reporting."
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--holdout_fraction", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--tie_tolerance", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--selection_estimators", type=int, default=200)
    parser.add_argument("--final_estimators", type=int, default=500)
    parser.add_argument("--max_depths", default="4,6,8")
    parser.add_argument("--min_samples_leaves", default="8,16,32")
    parser.add_argument("--min_abs_pearson", type=float, default=0.4)
    parser.add_argument("--min_delta_pearson", type=float, default=0.4)
    parser.add_argument("--min_pairwise", type=float, default=0.65)
    parser.add_argument("--min_frame_spearman", type=float, default=0.4)
    parser.add_argument("--min_top_set", type=float, default=0.5)
    parser.add_argument("--max_regret", type=float, default=0.03)
    args = parser.parse_args()

    rows = _load_rows(Path(args.csv))
    feature_sets = _feature_sets()
    required = {
        "sequence_id",
        "frame_idx",
        "sender_index",
        LABEL_ABSOLUTE,
        LABEL_GLOBAL,
        LABEL_MARGINAL,
    }
    for definition in feature_sets.values():
        required.update(definition["absolute"])
        required.update(definition["delta"])
    missing = sorted(required.difference(rows[0].keys() if rows else []))
    if missing:
        raise RuntimeError(
            "Missing v3.1 dataset columns: {}".format(missing)
        )

    (
        development_rows,
        holdout_rows,
        development_sequences,
        holdout_sequences,
    ) = _sequence_split(rows, float(args.holdout_fraction))
    if int(args.folds) > len(development_sequences):
        raise ValueError("folds cannot exceed development sequence count.")

    target_specs = {
        "absolute": LABEL_ABSOLUTE,
        "global_delta": LABEL_GLOBAL,
        "marginal_delta": LABEL_MARGINAL,
    }
    candidates: Dict[str, List[Dict[str, Any]]] = {
        target: [] for target in target_specs
    }
    grid = _candidate_grid(args)
    for feature_set_name, definition in feature_sets.items():
        for target, label_col in target_specs.items():
            feature_cols = (
                definition["absolute"]
                if target == "absolute"
                else definition["delta"]
            )
            for max_depth, min_samples_leaf in grid:
                result = _cross_validate_candidate(
                    development_rows,
                    feature_cols,
                    label_col,
                    target,
                    args,
                    max_depth,
                    min_samples_leaf,
                )
                result["feature_set"] = feature_set_name
                result["num_features"] = int(len(feature_cols))
                candidates[target].append(result)
                print(
                    "target={} features={} depth={} leaf={} score={:.6f}".format(
                        target,
                        feature_set_name,
                        max_depth,
                        min_samples_leaf,
                        float(result["selection_score"]),
                    )
                )

    selected: Dict[str, Dict[str, Any]] = {}
    holdout_metrics: Dict[str, Dict[str, Any]] = {}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for target, label_col in target_specs.items():
        ranked = sorted(
            candidates[target],
            key=lambda item: float(item["selection_score"]),
            reverse=True,
        )
        best = ranked[0]
        feature_set_name = str(best["feature_set"])
        definition = feature_sets[feature_set_name]
        feature_cols = (
            definition["absolute"]
            if target == "absolute"
            else definition["delta"]
        )
        train_x, train_y = _matrix(
            development_rows,
            feature_cols,
            label_col,
        )
        holdout_x, holdout_y = _matrix(
            holdout_rows,
            feature_cols,
            label_col,
        )
        model = _fit_model(
            train_x,
            train_y,
            int(args.final_estimators),
            int(best["max_depth"]),
            int(best["min_samples_leaf"]),
            int(args.seed),
        )
        holdout_pred = model.predict(holdout_x)
        metrics = (
            {"regression": _regression_metrics(holdout_y, holdout_pred)}
            if target == "absolute"
            else _action_metrics(
                holdout_rows,
                holdout_y,
                holdout_pred,
                float(args.tie_tolerance),
            )
        )
        selected[target] = {
            "feature_set": feature_set_name,
            "csv_feature_cols": feature_cols,
            "model_feature_cols": (
                [
                    name[len("collab_"):]
                    if name.startswith("collab_") else name
                    for name in feature_cols
                ]
                if target == "absolute" else feature_cols
            ),
            "max_depth": int(best["max_depth"]),
            "min_samples_leaf": int(best["min_samples_leaf"]),
            "cv_summary": best["summary"],
            "cv_selection_score": float(best["selection_score"]),
        }
        holdout_metrics[target] = metrics
        _save_model(
            out_dir / "{}_experimental.pkl".format(target),
            model,
            selected[target]["model_feature_cols"],
            label_col,
            "{}_experimental".format(target),
            feature_set_name,
        )

    gate = _holdout_gate(
        holdout_metrics["absolute"],
        holdout_metrics["global_delta"],
        args,
    )
    report = {
        "source_csv": str(args.csv),
        "status": (
            "reference_holdout_passed"
            if gate["passed"]
            else "do_not_enable"
        ),
        "num_rows": int(len(rows)),
        "num_sequences": int(
            len(development_sequences) + len(holdout_sequences)
        ),
        "development_sequences": development_sequences,
        "reference_holdout_sequences": holdout_sequences,
        "development_rows": int(len(development_rows)),
        "reference_holdout_rows": int(len(holdout_rows)),
        "selection_protocol": {
            "split": "GroupKFold on development sequences only",
            "folds": int(args.folds),
            "selection_estimators": int(args.selection_estimators),
            "final_estimators": int(args.final_estimators),
            "candidate_grid": [
                {
                    "max_depth": int(depth),
                    "min_samples_leaf": int(leaf),
                }
                for depth, leaf in grid
            ],
            "feature_sets": {
                name: {
                    "absolute_features": definition["absolute"],
                    "delta_features": definition["delta"],
                }
                for name, definition in feature_sets.items()
            },
        },
        "selected": selected,
        "reference_holdout_metrics": holdout_metrics,
        "reference_holdout_gate": gate,
        "all_cv_candidates": candidates,
    }
    safe_report = _json_safe(report)
    report_path = out_dir / "ap_proxy_v31_groupcv_report.json"
    report_path.write_text(
        json.dumps(safe_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = {
        "status": safe_report["status"],
        "selected": safe_report["selected"],
        "reference_holdout_metrics": safe_report[
            "reference_holdout_metrics"
        ],
        "reference_holdout_gate": safe_report[
            "reference_holdout_gate"
        ],
        "report": str(report_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("saved report:", report_path)
    print(
        "IMPORTANT: experimental models must not be enabled unless the gate "
        "passes and a separate untouched test is completed."
    )


if __name__ == "__main__":
    main()
