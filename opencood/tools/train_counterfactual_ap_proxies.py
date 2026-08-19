#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from opencood.methods.arce.reward.proxy.ap_proxy_features import (
    HEAD_AP_PROXY_FEATURES,
    PAIRED_SPATIAL_AP_PROXY_FEATURES,
)


ABS_MODEL_FEATURES = list(HEAD_AP_PROXY_FEATURES)
ABS_CSV_FEATURES = ["collab_" + name for name in HEAD_AP_PROXY_FEATURES]
DELTA_FEATURES = (
    ABS_CSV_FEATURES
    + ["ego_" + name for name in HEAD_AP_PROXY_FEATURES]
    + ["diff_" + name for name in HEAD_AP_PROXY_FEATURES]
    + list(PAIRED_SPATIAL_AP_PROXY_FEATURES)
)


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
        rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = rank
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
    }


def _majority_sign_accuracy(values: np.ndarray, eps: float) -> float:
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


def _delta_action_metrics(
    rows: Sequence[Dict[str, str]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tie_tolerance: float,
) -> Dict[str, Any]:
    send_indices = [
        index
        for index, row in enumerate(rows)
        if str(row.get("no_send", "")).strip().lower() not in {"true", "1"}
        and not str(row.get("action_id", "")).startswith("send0_")
    ]
    true_send = y_true[send_indices]
    pred_send = y_pred[send_indices]
    comparable = np.abs(true_send) > 1e-9
    global_sign_accuracy = (
        float(np.mean(np.sign(true_send[comparable]) == np.sign(pred_send[comparable])))
        if int(np.sum(comparable)) > 0
        else float("nan")
    )
    global_majority = _majority_sign_accuracy(true_send, 1e-9)

    by_frame: Dict[Tuple[int, int, int], List[int]] = {}
    for index, row in enumerate(rows):
        by_frame.setdefault(_frame_key(row), []).append(index)

    pair_correct = 0
    pair_total = 0
    exact_top1_hits = 0
    top_set_hits = 0
    frame_spearman = []
    regrets = []
    marginal_true = []
    marginal_pred = []
    for indices in by_frame.values():
        frame_true = y_true[indices]
        frame_pred = y_pred[indices]
        no_send_local = next(
            (
                local
                for local, index in enumerate(indices)
                if str(rows[index].get("action_id", "")).startswith("send0_")
                or str(rows[index].get("no_send", "")).strip().lower()
                in {"true", "1"}
            ),
            None,
        )
        if no_send_local is not None:
            baseline_pred = float(frame_pred[no_send_local])
            for local, index in enumerate(indices):
                if local == no_send_local:
                    continue
                marginal_true.append(
                    float(rows[index]["label_true_delta_quality_mean_0357"])
                )
                marginal_pred.append(float(frame_pred[local]) - baseline_pred)

        predicted_local = int(np.argmax(frame_pred))
        true_local = int(np.argmax(frame_true))
        true_best = float(np.max(frame_true))
        predicted_true = float(frame_true[predicted_local])
        exact_top1_hits += int(predicted_local == true_local)
        top_set_hits += int(
            predicted_true >= true_best - float(tie_tolerance)
        )
        regrets.append(max(0.0, true_best - predicted_true))

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

    marginal_true_array = np.asarray(marginal_true, dtype=np.float64)
    marginal_pred_array = np.asarray(marginal_pred, dtype=np.float64)
    marginal_comparable = np.abs(marginal_true_array) > 1e-9
    marginal_sign_accuracy = (
        float(
            np.mean(
                np.sign(marginal_true_array[marginal_comparable])
                == np.sign(marginal_pred_array[marginal_comparable])
            )
        )
        if int(np.sum(marginal_comparable)) > 0
        else float("nan")
    )
    marginal_majority = _majority_sign_accuracy(
        marginal_true_array,
        1e-9,
    )
    return {
        "send_only": _regression_metrics(true_send, pred_send),
        "global_send_sign_accuracy": global_sign_accuracy,
        "global_send_majority_sign_baseline": global_majority,
        "global_send_sign_lift": global_sign_accuracy - global_majority,
        "global_send_sign_comparable": int(np.sum(comparable)),
        "centered_marginal_pearson": _correlation(
            marginal_true_array,
            marginal_pred_array,
        ),
        "centered_marginal_spearman": _correlation(
            _rankdata(marginal_true_array),
            _rankdata(marginal_pred_array),
        ),
        "centered_marginal_sign_accuracy": marginal_sign_accuracy,
        "centered_marginal_majority_sign_baseline": marginal_majority,
        "centered_marginal_sign_lift": (
            marginal_sign_accuracy - marginal_majority
        ),
        "centered_marginal_sign_comparable": int(
            np.sum(marginal_comparable)
        ),
        "tie_tolerance": float(tie_tolerance),
        "tie_aware_pairwise_ranking_accuracy": (
            float(pair_correct) / float(pair_total) if pair_total else float("nan")
        ),
        "tie_aware_pairwise_comparisons": int(pair_total),
        "frame_ranking_spearman_mean": (
            float(np.mean(frame_spearman)) if frame_spearman else float("nan")
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
            float(np.quantile(regrets, 0.90)) if regrets else float("nan")
        ),
        "num_validation_frames": int(len(by_frame)),
    }


def _sequence_holdout_split(
    rows: Sequence[Dict[str, str]],
    validation_fraction: float,
) -> Tuple[
    List[Dict[str, str]],
    List[Dict[str, str]],
    List[int],
    List[int],
]:
    sequence_ids = sorted({int(row["sequence_id"]) for row in rows})
    if len(sequence_ids) < 3:
        raise ValueError(
            "Sequence holdout requires at least three sequence_id values."
        )
    validation_count = max(
        1,
        int(math.ceil(len(sequence_ids) * validation_fraction)),
    )
    if validation_count >= len(sequence_ids):
        raise ValueError("Validation split leaves no training sequences.")
    train_sequences = sequence_ids[:-validation_count]
    validation_sequences = sequence_ids[-validation_count:]
    train_set = set(train_sequences)
    validation_set = set(validation_sequences)
    train_rows = [
        row for row in rows if int(row["sequence_id"]) in train_set
    ]
    validation_rows = [
        row for row in rows if int(row["sequence_id"]) in validation_set
    ]
    return (
        train_rows,
        validation_rows,
        train_sequences,
        validation_sequences,
    )


def _fit_model(args: argparse.Namespace, x: np.ndarray, y: np.ndarray):
    model = RandomForestRegressor(
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth),
        min_samples_leaf=int(args.min_samples_leaf),
        random_state=int(args.seed),
        n_jobs=-1,
    )
    model.fit(x, y)
    return model


def _save_model(
    path: Path,
    model: Any,
    feature_cols: Sequence[str],
    label_col: str,
    proxy_type: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(
            {
                "model": model,
                "feature_cols": list(feature_cols),
                "label_col": str(label_col),
                "proxy_type": str(proxy_type),
                "feature_definition": "canonical_psm_rm_head_v3",
                "input_schema_version": 3,
            },
            handle,
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_abs_model", required=True)
    parser.add_argument("--out_delta_model", required=True)
    parser.add_argument("--out_meta", required=True)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--tie_tolerance", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_leaf", type=int, default=2)
    args = parser.parse_args()

    rows = _load_rows(Path(args.csv))
    required = set(
        [
            "frame_idx",
            "sequence_id",
            "sequence_frame_idx",
            "sender_index",
            "true_quality_mean_0357",
            "label_true_global_delta_quality_mean_0357",
            "label_true_delta_quality_mean_0357",
        ]
        + ABS_CSV_FEATURES
        + DELTA_FEATURES
    )
    missing = sorted(required.difference(rows[0].keys() if rows else []))
    if missing:
        raise RuntimeError(
            "Missing v3 counterfactual dataset columns: {}".format(missing)
        )

    (
        train_rows,
        validation_rows,
        train_sequences,
        validation_sequences,
    ) = _sequence_holdout_split(
        rows,
        float(args.validation_fraction),
    )
    abs_train_x, abs_train_y = _matrix(
        train_rows,
        ABS_CSV_FEATURES,
        "true_quality_mean_0357",
    )
    abs_val_x, abs_val_y = _matrix(
        validation_rows,
        ABS_CSV_FEATURES,
        "true_quality_mean_0357",
    )
    delta_train_x, delta_train_y = _matrix(
        train_rows,
        DELTA_FEATURES,
        "label_true_global_delta_quality_mean_0357",
    )
    delta_val_x, delta_val_y = _matrix(
        validation_rows,
        DELTA_FEATURES,
        "label_true_global_delta_quality_mean_0357",
    )

    abs_model = _fit_model(args, abs_train_x, abs_train_y)
    delta_model = _fit_model(args, delta_train_x, delta_train_y)
    abs_train_pred = abs_model.predict(abs_train_x)
    abs_val_pred = abs_model.predict(abs_val_x)
    delta_train_pred = delta_model.predict(delta_train_x)
    delta_val_pred = delta_model.predict(delta_val_x)

    _save_model(
        Path(args.out_abs_model),
        abs_model,
        ABS_MODEL_FEATURES,
        "true_quality_mean_0357",
        "absolute_frame_quality_proxy",
    )
    _save_model(
        Path(args.out_delta_model),
        delta_model,
        DELTA_FEATURES,
        "label_true_global_delta_quality_mean_0357",
        "paired_delta_ap_proxy",
    )

    train_frame_keys = sorted({_frame_key(row) for row in train_rows})
    validation_frame_keys = sorted(
        {_frame_key(row) for row in validation_rows}
    )
    meta = {
        "source_csv": str(args.csv),
        "feature_definition": "canonical_psm_rm_head_v3",
        "split": "sequence_holdout",
        "num_rows": int(len(rows)),
        "num_sequences": int(
            len(train_sequences) + len(validation_sequences)
        ),
        "num_frame_sender_groups": int(
            len(train_frame_keys) + len(validation_frame_keys)
        ),
        "train_rows": int(len(train_rows)),
        "validation_rows": int(len(validation_rows)),
        "train_frame_sender_groups": int(len(train_frame_keys)),
        "validation_frame_sender_groups": int(len(validation_frame_keys)),
        "train_sequences": train_sequences,
        "validation_sequences": validation_sequences,
        "absolute_proxy": {
            "csv_feature_cols": ABS_CSV_FEATURES,
            "model_feature_cols": ABS_MODEL_FEATURES,
            "train": _regression_metrics(abs_train_y, abs_train_pred),
            "validation": _regression_metrics(abs_val_y, abs_val_pred),
        },
        "delta_proxy": {
            "feature_cols": DELTA_FEATURES,
            "label": (
                "global collaborative frame quality minus ego-only frame quality"
            ),
            "marginal_audit_label": (
                "action frame quality minus same-frame target-sender no-send quality"
            ),
            "train": _regression_metrics(delta_train_y, delta_train_pred),
            "validation": _regression_metrics(delta_val_y, delta_val_pred),
            "validation_action_metrics": _delta_action_metrics(
                validation_rows,
                delta_val_y,
                delta_val_pred,
                float(args.tie_tolerance),
            ),
        },
        "params": {
            "seed": int(args.seed),
            "validation_fraction": float(args.validation_fraction),
            "tie_tolerance": float(args.tie_tolerance),
            "n_estimators": int(args.n_estimators),
            "max_depth": int(args.max_depth),
            "min_samples_leaf": int(args.min_samples_leaf),
        },
    }
    out_meta = Path(args.out_meta)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    safe_meta = _json_safe(meta)
    out_meta.write_text(
        json.dumps(safe_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(safe_meta, indent=2, ensure_ascii=False))
    print("saved absolute proxy:", args.out_abs_model)
    print("saved delta proxy:", args.out_delta_model)
    print("saved metadata:", out_meta)


if __name__ == "__main__":
    main()
