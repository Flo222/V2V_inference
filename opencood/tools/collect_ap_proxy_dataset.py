from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils


FEATURE_COLS = [
    "dense_mean_conf",
    "dense_max_conf",
    "dense_sum_conf",
    "dense_std_conf",
    "dense_count_gt_03",
    "dense_count_gt_05",
    "dense_count_gt_07",
    "dense_top10_mean",
    "dense_top50_mean",
    "num_pred_boxes",
    "mean_pred_score",
    "max_pred_score",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--fusion_method", default="intermediate",
                        choices=["late", "early", "intermediate"])
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--out_csv", default="audit_runs/step8_ap_proxy_reward/ap_proxy_dataset.csv")
    parser.add_argument("--out_meta", default="audit_runs/step8_ap_proxy_reward/ap_proxy_dataset_meta.json")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--global_sort_detections", action="store_true")
    return parser.parse_args()


def to_device(x: Any, device: torch.device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [to_device(v, device) for v in x]
    return x


def find_tensor_by_key(obj: Any, keys: Iterable[str]):
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and torch.is_tensor(obj[k]):
                return obj[k]
        for v in obj.values():
            found = find_tensor_by_key(v, keys)
            if found is not None:
                return found
    if isinstance(obj, list):
        for v in obj:
            found = find_tensor_by_key(v, keys)
            if found is not None:
                return found
    return None


def dense_features_from_psm(psm: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        prob = torch.sigmoid(psm).detach()

        if prob.dim() == 4:
            dense = prob.max(dim=1)[0]
        else:
            dense = prob.reshape(prob.shape[0], -1)

        flat = dense.reshape(-1).float()

        if flat.numel() == 0:
            return {
                "dense_mean_conf": 0.0,
                "dense_max_conf": 0.0,
                "dense_sum_conf": 0.0,
                "dense_std_conf": 0.0,
                "dense_count_gt_03": 0.0,
                "dense_count_gt_05": 0.0,
                "dense_count_gt_07": 0.0,
                "dense_top10_mean": 0.0,
                "dense_top50_mean": 0.0,
            }

        top10 = torch.topk(flat, k=min(10, flat.numel())).values
        top50 = torch.topk(flat, k=min(50, flat.numel())).values

        return {
            "dense_mean_conf": float(flat.mean().cpu().item()),
            "dense_max_conf": float(flat.max().cpu().item()),
            "dense_sum_conf": float(flat.sum().cpu().item()),
            "dense_std_conf": float(flat.std(unbiased=False).cpu().item()),
            "dense_count_gt_03": float((flat > 0.3).sum().cpu().item()),
            "dense_count_gt_05": float((flat > 0.5).sum().cpu().item()),
            "dense_count_gt_07": float((flat > 0.7).sum().cpu().item()),
            "dense_top10_mean": float(top10.mean().cpu().item()),
            "dense_top50_mean": float(top50.mean().cpu().item()),
        }


def pred_score_features(pred_box_tensor, pred_score) -> Dict[str, float]:
    if pred_box_tensor is None:
        return {
            "num_pred_boxes": 0.0,
            "mean_pred_score": 0.0,
            "max_pred_score": 0.0,
        }

    n = int(pred_box_tensor.shape[0])
    if pred_score is None or n <= 0:
        return {
            "num_pred_boxes": float(n),
            "mean_pred_score": 0.0,
            "max_pred_score": 0.0,
        }

    score = pred_score.detach().reshape(-1).float()
    if score.numel() == 0:
        return {
            "num_pred_boxes": float(n),
            "mean_pred_score": 0.0,
            "max_pred_score": 0.0,
        }

    score = score[:n]
    return {
        "num_pred_boxes": float(n),
        "mean_pred_score": float(score.mean().cpu().item()),
        "max_pred_score": float(score.max().cpu().item()),
    }


def frame_quality(pred_box_tensor, pred_score, gt_box_tensor, iou: float) -> Dict[str, float]:
    stat = {float(iou): {"tp": [], "fp": [], "gt": 0, "score": []}}
    eval_utils.caluclate_tp_fp(
        pred_box_tensor,
        pred_score,
        gt_box_tensor,
        stat,
        float(iou),
    )

    s = stat[float(iou)]
    tp = float(sum(s["tp"]))
    fp = float(sum(s["fp"]))
    gt = float(s["gt"])
    fn = max(gt - tp, 0.0)

    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(gt, 1.0)
    quality = tp / max(tp + fp + fn, 1.0)

    return {
        "tp_{}".format(str(iou).replace(".", "p")): tp,
        "fp_{}".format(str(iou).replace(".", "p")): fp,
        "gt_{}".format(str(iou).replace(".", "p")): gt,
        "precision_{}".format(str(iou).replace(".", "p")): precision,
        "recall_{}".format(str(iou).replace(".", "p")): recall,
        "quality_{}".format(str(iou).replace(".", "p")): quality,
    }


def main():
    args = parse_args()

    out_csv = Path(args.out_csv)
    out_meta = Path(args.out_meta)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    opt = argparse.Namespace(model_dir=args.model_dir, fusion_method=args.fusion_method)
    hypes = yaml_utils.load_yaml(None, opt)

    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=int(args.num_workers),
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    model = train_utils.create_model(hypes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    _, model = train_utils.load_saved_model(args.model_dir, model)
    if hasattr(model, "update_epoch"):
        model.update_epoch(999)
    model.eval()

    rows = []
    max_frames = None if args.max_frames is None or int(args.max_frames) < 0 else int(args.max_frames)

    with torch.no_grad():
        for i, batch_data in tqdm(enumerate(loader), total=len(loader)):
            if max_frames is not None and i >= max_frames:
                break

            batch_data = to_device(batch_data, device)

            output_dict = model(batch_data["ego"])
            psm = find_tensor_by_key(output_dict, ["psm", "cls_preds"])
            if psm is None:
                raise RuntimeError("Cannot find psm/cls_preds in model output.")

            # Use the same forward output for both AP-proxy features and labels.
            # Calling inference_utils here would run model(...) a second time, which
            # mutates ARCE online policy / pending reward state and misaligns features
            # with labels.
            try:
                pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(
                    batch_data,
                    {"ego": output_dict},
                )
            except Exception as exc:
                raise RuntimeError(
                    "dataset.post_process(batch_data, output_dict) failed; "
                    "collector requires single-forward label extraction."
                ) from exc

            row = {
                "frame_index": int(i),
                "model_dir": str(args.model_dir),
                "fusion_method": str(args.fusion_method),
            }
            row.update(dense_features_from_psm(psm))
            row.update(pred_score_features(pred_box_tensor, pred_score))

            for iou in [0.3, 0.5, 0.7]:
                row.update(frame_quality(pred_box_tensor, pred_score, gt_box_tensor, iou))

            row["label_quality_05"] = row["quality_0p5"]
            row["label_quality_mean_0357"] = (
                row["quality_0p3"] + row["quality_0p5"] + row["quality_0p7"]
            ) / 3.0

            rows.append(row)

    fieldnames = [
        "frame_index",
        "model_dir",
        "fusion_method",
    ] + FEATURE_COLS + [
        "tp_0p3", "fp_0p3", "gt_0p3", "precision_0p3", "recall_0p3", "quality_0p3",
        "tp_0p5", "fp_0p5", "gt_0p5", "precision_0p5", "recall_0p5", "quality_0p5",
        "tp_0p7", "fp_0p7", "gt_0p7", "precision_0p7", "recall_0p7", "quality_0p7",
        "label_quality_05",
        "label_quality_mean_0357",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    meta = {
        "model_dir": str(args.model_dir),
        "fusion_method": str(args.fusion_method),
        "num_rows": len(rows),
        "feature_cols": FEATURE_COLS,
        "label_cols": ["label_quality_05", "label_quality_mean_0357"],
        "label_definition": "frame quality = TP / (TP + FP + FN), using OpenCOOD eval_utils.caluclate_tp_fp per frame",
    }
    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print("saved csv:", out_csv)
    print("saved meta:", out_meta)


if __name__ == "__main__":
    main()
