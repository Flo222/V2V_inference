from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import List

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


BASE_FEATURES = [
    "dense_mean_conf",
    "dense_max_conf",
    "dense_sum_conf",
    "dense_std_conf",
    "dense_count_gt_03",
    "dense_count_gt_05",
    "dense_count_gt_07",
    "dense_top10_mean",
    "dense_top50_mean",
]

DEFAULT_FEATURE_COLS = (
    ["collab_" + k for k in BASE_FEATURES]
    + ["ego_" + k for k in BASE_FEATURES]
    + ["diff_" + k for k in BASE_FEATURES]
)


def load_rows(csv_path: Path):
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "r2": float(r2_score(y_true, y_pred)),
        "y_true_mean": float(np.mean(y_true)),
        "y_pred_mean": float(np.mean(y_pred)),
        "y_true_min": float(np.min(y_true)),
        "y_true_max": float(np.max(y_true)),
        "y_pred_min": float(np.min(y_pred)),
        "y_pred_max": float(np.max(y_pred)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--label_col", default="label_delta_quality_mean_0357")
    parser.add_argument("--out_model", required=True)
    parser.add_argument("--out_meta", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_leaf", type=int, default=2)
    args = parser.parse_args()

    rows = load_rows(Path(args.csv))
    feature_cols = list(DEFAULT_FEATURE_COLS)

    missing = [
        c for c in feature_cols + [args.label_col]
        if not rows or c not in rows[0]
    ]
    if missing:
        raise RuntimeError("missing columns: {}".format(missing))

    X = np.asarray([[float(r[c]) for c in feature_cols] for r in rows], dtype=np.float32)
    y = np.asarray([float(r[args.label_col]) for r in rows], dtype=np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=float(args.test_size),
        random_state=int(args.seed),
    )

    model = RandomForestRegressor(
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth),
        min_samples_leaf=int(args.min_samples_leaf),
        random_state=int(args.seed),
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    meta = {
        "model_type": "RandomForestRegressor",
        "proxy_type": "paired_delta_ap_proxy",
        "source_csv": str(args.csv),
        "label_col": args.label_col,
        "feature_cols": feature_cols,
        "num_rows": int(len(rows)),
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val)),
        "seed": int(args.seed),
        "test_size": float(args.test_size),
        "params": {
            "n_estimators": int(args.n_estimators),
            "max_depth": int(args.max_depth),
            "min_samples_leaf": int(args.min_samples_leaf),
        },
        "train_metrics": metrics(y_train, train_pred),
        "val_metrics": metrics(y_val, val_pred),
        "feature_importances": {
            c: float(v) for c, v in zip(feature_cols, model.feature_importances_)
        },
        "label_definition": "delta frame quality = quality(collab psm/rm) - quality(ego_psm/ego_rm), quality = TP/(TP+FP+FN)",
    }

    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    with out_model.open("wb") as f:
        pickle.dump({
            "model": model,
            "feature_cols": feature_cols,
            "label_col": args.label_col,
            "proxy_type": "paired_delta_ap_proxy",
        }, f)

    out_meta = Path(args.out_meta)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print("saved model:", out_model)
    print("saved meta:", out_meta)


if __name__ == "__main__":
    main()
