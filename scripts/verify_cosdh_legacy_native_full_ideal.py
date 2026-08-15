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
        return None, {}, {}
    expected_late_cav_ids = [str(key) for key in batch.keys() if key != "ego"]
    transport = model.cosdh_legacy_native_transport
    transport.configure({
        "enabled": bool(enabled),
        "mode": "ideal" if enabled else "disabled",
        "intermediate_enabled": bool(enabled),
        "late_enabled": bool(enabled),
        "require_exact_roundtrip": True,
    })
    with torch.no_grad():
        result = inference_utils.inference_late_fusion(
            batch, model, dataset
        )
    info = copy.deepcopy(transport.latest_info)
    meta = {
        "expected_late_cav_ids": expected_late_cav_ids,
        "expected_late_count": int(len(expected_late_cav_ids)),
    }
    return result, info, meta


def late_record_report(info, expected_count, use_dir):
    records = info.get("late_records", [])
    required = set(["cls_preds", "reg_preds"])
    if use_dir:
        required.add("dir_preds")
    records_ok = len(records) == int(expected_count)
    field_sets_ok = True
    byte_counts_ok = True
    for record in records:
        fields = record.get("fields", [])
        names = set(item.get("field") for item in fields)
        field_sets_ok = field_sets_ok and required.issubset(names)
        byte_counts_ok = byte_counts_ok and (
            int(record.get("source_bytes", -1))
            == int(record.get("received_bytes", -2))
            and bool(record.get("bytes_equal", False))
        )
    return {
        "expected_count": int(expected_count),
        "observed_count": int(len(records)),
        "record_count_equal": bool(records_ok),
        "required_fields_present": bool(field_sets_ok),
        "record_bytes_exact": bool(byte_counts_ok),
        "late_all_bytes_equal": bool(
            info.get("late_all_bytes_equal", False)
        ),
        "source_received_equal": bool(
            int(info.get("late_source_bytes", -1))
            == int(info.get("late_received_bytes", -2))
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare original CoSDH Clean inference with exact Intermediate "
            "FP16 and Late dense-head byte round-trips."
        )
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--epoch", type=int, default=-1)
    parser.add_argument(
        "--indices", default="0,1,10,59,100,500,1000,2169"
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
        "late_enabled": True,
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
    indices = requested + rng.sample(random_pool, random_count)

    reports = []
    overall_pass = True
    skipped = []
    use_dir = bool(getattr(model, "use_dir", False))

    for position, index in enumerate(indices, start=1):
        print("[{}/{}] sample {}".format(position, len(indices), index))
        legacy, legacy_info, legacy_meta = run_one(
            dataset, model, device, index, False, args.seed
        )
        ideal, ideal_info, ideal_meta = run_one(
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
        gt_labels = tensor_report(
            legacy.get("gt_label_tensor"), ideal.get("gt_label_tensor")
        )
        scale_records = ideal_info.get("scale_records", [])
        intermediate_ok = bool(
            len(scale_records) == 3
            and ideal_info.get("intermediate_all_bytes_equal", False)
            and int(ideal_info.get("intermediate_source_bytes", -1))
            == int(ideal_info.get("intermediate_received_bytes", -2))
        )
        late_check = late_record_report(
            ideal_info,
            ideal_meta["expected_late_count"],
            use_dir,
        )
        sample_pass = bool(
            boxes["equal"]
            and scores["equal"]
            and gt_boxes["equal"]
            and gt_labels["equal"]
            and intermediate_ok
            and late_check["record_count_equal"]
            and late_check["required_fields_present"]
            and late_check["record_bytes_exact"]
            and late_check["late_all_bytes_equal"]
            and late_check["source_received_equal"]
            and ideal_info.get("late_enabled", False)
            and not ideal_info.get("ucb_arce_used", True)
        )
        overall_pass = overall_pass and sample_pass
        report = {
            "sample_index": int(index),
            "pass": sample_pass,
            "boxes": boxes,
            "scores": scores,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "transport": ideal_info,
            "late_check": late_check,
            "ideal_meta": ideal_meta,
            "legacy_transport_enabled": bool(
                legacy_info.get("enabled", False)
            ),
            "legacy_meta": legacy_meta,
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
        "all_intermediate_bytes_exact": all(
            item["transport"].get(
                "intermediate_all_bytes_equal", False
            ) for item in reports
        ),
        "all_late_bytes_exact": all(
            item["late_check"]["late_all_bytes_equal"]
            and item["late_check"]["source_received_equal"]
            for item in reports
        ),
        "all_late_output_counts_exact": all(
            item["late_check"]["record_count_equal"]
            for item in reports
        ),
        "all_three_scales_observed": all(
            len(item["transport"].get("scale_records", [])) == 3
            for item in reports
        ),
        "late_transport_enabled": True,
        "late_payload_type": "dense_detection_heads_before_dataset_postprocess",
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
