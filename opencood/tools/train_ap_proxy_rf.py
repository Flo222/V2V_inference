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


DEFAULT_FEATURE_COLS = [
    # Must match PointPillarWhere2commArce.ap_proxy_feature_cols.
    # These features are available online at reward-update time from dense psm.
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--label_col", default="label_quality_mean_0357")
    parser.add_argument("--out_model", default="audit_runs/step8_ap_proxy_reward/ap_proxy_dense_rf.pkl")
    parser.add_argument("--out_meta", default="audit_runs/step8_ap_proxy_reward/ap_proxy_dense_rf_meta.json")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=8)
    return parser.parse_args()


def load_rows(path: Path, feature_cols: List[str], label_col: str):
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise RuntimeError(f"No rows in {path}")

    x = []
    y = []
    kept = 0

    for row in rows:
        try:
            feats = [float(row[c]) for c in feature_cols]
            label = float(row[label_col])
        except Exception:
            continue
        x.append(feats)
        y.append(label)
        kept += 1

    if kept <= 0:
        raise RuntimeError("No valid training rows.")

    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), kept


def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
        "y_true_mean": float(np.mean(y_true)),
        "y_pred_mean": float(np.mean(y_pred)),
        "y_true_min": float(np.min(y_true)),
        "y_true_max": float(np.max(y_true)),
        "y_pred_min": float(np.min(y_pred)),
        "y_pred_max": float(np.max(y_pred)),
    }


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    out_model = Path(args.out_model)
    out_meta = Path(args.out_meta)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    feature_cols = list(DEFAULT_FEATURE_COLS)
    x, y, n = load_rows(csv_path, feature_cols, args.label_col)

    if n < 10:
        raise RuntimeError(f"Too few rows for AP proxy training: {n}")

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=float(args.test_size),
        random_state=int(args.seed),
    )

    model = RandomForestRegressor(
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth),
        random_state=int(args.seed),
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)

    with out_model.open("wb") as f:
        pickle.dump(model, f)

    importances = {
        c: float(v)
        for c, v in zip(feature_cols, model.feature_importances_)
    }

    meta = {
        "model_type": "RandomForestRegressor",
        "source_csv": str(csv_path),
        "label_col": str(args.label_col),
        "feature_cols": feature_cols,
        "num_rows": int(n),
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val)),
        "seed": int(args.seed),
        "test_size": float(args.test_size),
        "params": {
            "n_estimators": int(args.n_estimators),
            "max_depth": int(args.max_depth),
            "min_samples_leaf": 2,
        },
        "train_metrics": metrics(y_train, train_pred),
        "val_metrics": metrics(y_val, val_pred),
        "feature_importances": importances,
        "label_definition": "frame quality = TP / (TP + FP + FN); default label is mean of IoU 0.3/0.5/0.7 qualities",
    }

    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print("saved model:", out_model)
    print("saved meta:", out_meta)


if __name__ == "__main__":
    main()
