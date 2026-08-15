#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets.cosdh_dataset_builder import (
    build_dataset_cosdh as build_dataset,
)
from opencood.tools import inference_utils_cosdh as inference_utils
from opencood.tools import train_utils_cosdh as train_utils


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_indices(text):
    if not text.strip():
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def tensor_report(left, right, atol=0.0, rtol=0.0):
    if left is None or right is None:
        equal = left is None and right is None
        return {
            "both_none": bool(equal),
            "shape_equal": bool(equal),
            "equal": bool(equal),
            "allclose": bool(equal),
            "max_abs_diff": 0.0 if equal else None,
        }
    if not torch.is_tensor(left) or not torch.is_tensor(right):
        return {
            "both_none": False,
            "shape_equal": False,
            "equal": False,
            "allclose": False,
            "max_abs_diff": None,
        }
    shape_equal = tuple(left.shape) == tuple(right.shape)
    if not shape_equal:
        return {
            "both_none": False,
            "shape_equal": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "equal": False,
            "allclose": False,
            "max_abs_diff": None,
        }
    left_cpu = left.detach().cpu()
    right_cpu = right.detach().cpu()
    equal = bool(torch.equal(left_cpu, right_cpu))
    allclose = bool(torch.allclose(left_cpu, right_cpu, atol=atol, rtol=rtol))
    max_diff = 0.0
    if left_cpu.numel() > 0:
        max_diff = float((left_cpu - right_cpu).abs().max().item())
    return {
        "both_none": False,
        "shape_equal": True,
        "shape": list(left_cpu.shape),
        "equal": equal,
        "allclose": allclose,
        "max_abs_diff": max_diff,
    }


def get_batch(dataset, index, device, seed):
    # Retrieve each mode from a fresh deterministic sample to avoid relying on
    # in-place flags added by inference_late_fusion.
    set_seed(seed + int(index))
    sample = dataset[index]
    if sample is None:
        return None
    batch = dataset.collate_batch_test([sample])
    if batch is None:
        return None
    return train_utils.to_device(batch, device)


def run_one(dataset, model, device, index, enabled, seed):
    batch = get_batch(dataset, index, device, seed)
    if batch is None:
        return None, {}
    transport = model.cosdh_legacy_native_transport
    transport.configure({
        "enabled": bool(enabled),
        "mode": "ideal" if enabled else "disabled",
        "intermediate_enabled": bool(enabled),
        "late_enabled": False,
        "require_exact_roundtrip": True,
    })
    with torch.no_grad():
        result = inference_utils.inference_late_fusion(
            batch, model, dataset
        )
    info = copy.deepcopy(transport.latest_info)
    return result, info


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare original CoSDH Clean inference against the same forward "
            "with an exact FP16 encoded Intermediate byte round-trip."
        )
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--epoch", type=int, default=-1)
    parser.add_argument(
        "--indices",
        default="0,1,10,59,100,500,1000,2169",
    )
    parser.add_argument("--random-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    opt = SimpleNamespace(model_dir=args.model_dir)
    hypes = yaml_utils.load_yaml(None, opt)
    if hypes.get("test_dir"):
        hypes["validate_dir"] = hypes["test_dir"]
    hypes["fusion"]["core_method"] = "intermediatelate"
    model_args = hypes["model"]["args"]
    model_args["cosdh_legacy_native"] = {
        "enabled": True,
        "mode": "ideal",
        "intermediate_enabled": True,
        "late_enabled": False,
        "require_exact_roundtrip": True,
    }
    for key in (
        "cosdh_paper_native",
        "arce",
        "cosdh_markov",
        "cosdh_late_markov",
    ):
        if isinstance(model_args.get(key), dict):
            model_args[key]["enabled"] = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Creating model")
    model = train_utils.create_model(hypes)
    epoch, model = train_utils.load_saved_model(
        args.model_dir, model, epoch=args.epoch
    )
    model = model.to(device).eval()
    print("Loaded epoch:", epoch)
    dataset = build_dataset(hypes, visualize=False, train=False)
    dataset_len = len(dataset)
    print("Dataset length:", dataset_len)

    requested = parse_indices(args.indices)
    invalid = [idx for idx in requested if idx < 0 or idx >= dataset_len]
    if invalid:
        raise ValueError("indices outside dataset: {}".format(invalid))

    random_pool = [idx for idx in range(dataset_len) if idx not in requested]
    rng = random.Random(args.seed)
    random_count = min(int(args.random_count), len(random_pool))
    random_indices = rng.sample(random_pool, random_count)
    indices = requested + random_indices

    reports = []
    overall_pass = True
    skipped = []

    for position, index in enumerate(indices, start=1):
        print("[{}/{}] sample {}".format(position, len(indices), index))
        legacy, legacy_info = run_one(
            dataset, model, device, index, False, args.seed
        )
        ideal, ideal_info = run_one(
            dataset, model, device, index, True, args.seed
        )
        if legacy is None or ideal is None:
            skipped.append(index)
            continue

        boxes = tensor_report(
            legacy.get("pred_box_tensor"), ideal.get("pred_box_tensor")
        )
        scores = tensor_report(
            legacy.get("pred_score"), ideal.get("pred_score")
        )
        gt_boxes = tensor_report(
            legacy.get("gt_box_tensor"), ideal.get("gt_box_tensor")
        )
        scale_records = ideal_info.get("scale_records", [])
        bytes_equal = bool(ideal_info.get("all_bytes_equal", False))
        scale_count_ok = len(scale_records) == 3
        source_received_equal = (
            int(ideal_info.get("source_bytes", -1))
            == int(ideal_info.get("received_bytes", -2))
        )
        sample_pass = bool(
            boxes["equal"]
            and scores["equal"]
            and gt_boxes["equal"]
            and bytes_equal
            and scale_count_ok
            and source_received_equal
            and not ideal_info.get("ucb_arce_used", True)
        )
        overall_pass = overall_pass and sample_pass
        report = {
            "sample_index": int(index),
            "pass": sample_pass,
            "boxes": boxes,
            "scores": scores,
            "gt_boxes": gt_boxes,
            "transport": ideal_info,
            "legacy_transport_enabled": bool(
                legacy_info.get("enabled", False)
            ),
        }
        reports.append(report)
        if not sample_pass:
            print(json.dumps(report, indent=2, ensure_ascii=False))
            break
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "model_dir": str(Path(args.model_dir).resolve()),
        "epoch": int(epoch),
        "fixed_indices": requested,
        "random_seed": int(args.seed),
        "random_count": int(random_count),
        "tested_count": int(len(reports)),
        "skipped_indices": skipped,
        "all_prediction_tensors_exact": all(
            item["boxes"]["equal"] and item["scores"]["equal"]
            for item in reports
        ),
        "all_transport_bytes_exact": all(
            item["transport"].get("all_bytes_equal", False)
            for item in reports
        ),
        "all_three_scales_observed": all(
            len(item["transport"].get("scale_records", [])) == 3
            for item in reports
        ),
        "late_transport_enabled": False,
        "ucb_arce_used": False,
        "overall_pass": bool(
            overall_pass
            and not skipped
            and len(reports) == len(indices)
        ),
        "samples": reports,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        print("saved:", output)
    if not summary["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
