from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.methods.arce.policies.ap_proxy_features import (
    DENSE_AP_PROXY_FEATURES,
    dense_ap_proxy_features,
)


DENSE_FEATURE_COLS = list(DENSE_AP_PROXY_FEATURES)


def move_to_cuda(x: Any):
    if torch.is_tensor(x):
        return x.cuda()
    if isinstance(x, dict):
        return {k: move_to_cuda(v) for k, v in x.items()}
    if isinstance(x, list):
        return [move_to_cuda(v) for v in x]
    return x


def dense_features_from_psm(psm: torch.Tensor, prefix: str) -> Dict[str, float]:
    base = dense_ap_proxy_features(psm)
    return {prefix + k: v for k, v in base.items()}


def quality_from_postprocess(
    dataset: Any,
    batch_data: Dict[str, Any],
    psm: torch.Tensor,
    rm: torch.Tensor,
) -> Dict[str, float]:
    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(
        batch_data,
        {"ego": {"psm": psm, "rm": rm}},
    )

    out: Dict[str, float] = {}
    for iou in [0.3, 0.5, 0.7]:
        result_stat = {
            iou: {
                "tp": [],
                "fp": [],
                "gt": 0,
                "score": [],
            }
        }
        eval_utils.caluclate_tp_fp(
            pred_box_tensor,
            pred_score,
            gt_box_tensor,
            result_stat,
            iou,
        )
        tp = float(sum(result_stat[iou]["tp"]))
        fp = float(sum(result_stat[iou]["fp"]))
        gt = float(result_stat[iou]["gt"])
        fn = max(0.0, gt - tp)
        denom = tp + fp + fn
        q = 0.0 if denom <= 0 else tp / denom
        out["quality_0p{}".format(str(iou).split(".")[1])] = float(q)

    out["quality_mean_0357"] = (
        out["quality_0p3"] + out["quality_0p5"] + out["quality_0p7"]
    ) / 3.0
    return out


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--fusion_method", default="intermediate")
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_meta", required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    hypes = yaml_utils.load_yaml(None, argparse.Namespace(model_dir=args.model_dir))
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=int(args.num_workers),
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
    )

    model = train_utils.create_model(hypes).cuda()
    _, model = train_utils.load_saved_model(args.model_dir, model)
    model.eval()

    rows = []
    max_frames = None if args.max_frames is None or args.max_frames < 0 else int(args.max_frames)

    with torch.no_grad():
        for frame_idx, batch_data in enumerate(tqdm(loader)):
            if max_frames is not None and frame_idx >= max_frames:
                break

            batch_data = move_to_cuda(batch_data)
            output_dict = model(batch_data["ego"])

            required = ["psm", "rm", "ego_psm", "ego_rm"]
            missing = [k for k in required if k not in output_dict]
            if missing:
                raise RuntimeError("model output missing keys for delta AP proxy: {}".format(missing))

            collab_quality = quality_from_postprocess(
                dataset,
                batch_data,
                output_dict["psm"],
                output_dict["rm"],
            )
            ego_quality = quality_from_postprocess(
                dataset,
                batch_data,
                output_dict["ego_psm"],
                output_dict["ego_rm"],
            )

            row: Dict[str, Any] = {
                "frame_idx": int(frame_idx),
            }
            row.update(dense_features_from_psm(output_dict["psm"], "collab_"))
            row.update(dense_features_from_psm(output_dict["ego_psm"], "ego_"))

            for k in DENSE_FEATURE_COLS:
                row["diff_" + k] = float(row["collab_" + k] - row["ego_" + k])

            for k, v in collab_quality.items():
                row["collab_" + k] = float(v)
            for k, v in ego_quality.items():
                row["ego_" + k] = float(v)

            row["label_delta_quality_0p3"] = row["collab_quality_0p3"] - row["ego_quality_0p3"]
            row["label_delta_quality_0p5"] = row["collab_quality_0p5"] - row["ego_quality_0p5"]
            row["label_delta_quality_0p7"] = row["collab_quality_0p7"] - row["ego_quality_0p7"]
            row["label_delta_quality_mean_0357"] = (
                row["label_delta_quality_0p3"]
                + row["label_delta_quality_0p5"]
                + row["label_delta_quality_0p7"]
            ) / 3.0

            rows.append(row)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    feature_cols = (
        ["collab_" + k for k in DENSE_FEATURE_COLS]
        + ["ego_" + k for k in DENSE_FEATURE_COLS]
        + ["diff_" + k for k in DENSE_FEATURE_COLS]
    )
    label_cols = [
        "label_delta_quality_0p3",
        "label_delta_quality_0p5",
        "label_delta_quality_0p7",
        "label_delta_quality_mean_0357",
    ]

    fieldnames = ["frame_idx"] + feature_cols + [
        "collab_quality_0p3",
        "collab_quality_0p5",
        "collab_quality_0p7",
        "collab_quality_mean_0357",
        "ego_quality_0p3",
        "ego_quality_0p5",
        "ego_quality_0p7",
        "ego_quality_mean_0357",
    ] + label_cols

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    meta = {
        "model_dir": args.model_dir,
        "fusion_method": args.fusion_method,
        "num_rows": len(rows),
        "feature_cols": feature_cols,
        "label_cols": label_cols,
        "label_definition": "paired delta frame quality = quality(collab psm/rm) - quality(ego_psm/ego_rm), quality = TP/(TP+FP+FN)",
    }

    out_meta = Path(args.out_meta)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print("saved csv:", out_csv)
    print("saved meta:", out_meta)


if __name__ == "__main__":
    main()
