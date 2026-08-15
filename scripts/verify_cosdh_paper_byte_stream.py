#!/usr/bin/env python3
from __future__ import print_function

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from opencood.data_utils.datasets.cosdh_dataset_builder import \
    build_dataset_cosdh as build_dataset
from opencood.hypes_yaml import yaml_utils
from opencood.tools import inference_utils_cosdh
from opencood.tools import train_utils


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_dir, hypes, device):
    model = train_utils.create_model(hypes)
    model_dir_path = Path(model_dir)
    numeric = []
    non_numeric = []
    for checkpoint in model_dir_path.glob("net_epoch*.pth"):
        match = re.fullmatch(r"net_epoch(\d+)\.pth", checkpoint.name)
        if match:
            numeric.append((int(match.group(1)), checkpoint))
        else:
            non_numeric.append(checkpoint)
    if not numeric:
        raise FileNotFoundError("No numeric net_epoch*.pth in {}".format(model_dir))
    expected_epoch = max(numeric, key=lambda item: item[0])[0]

    hidden = []
    try:
        for checkpoint in non_numeric:
            temporary = checkpoint.with_name("." + checkpoint.name + ".verify_hidden")
            checkpoint.rename(temporary)
            hidden.append((temporary, checkpoint))
        epoch, model = train_utils.load_saved_model(model_dir, model)
    finally:
        for temporary, original in reversed(hidden):
            if temporary.exists():
                temporary.rename(original)
    if int(epoch) != int(expected_epoch):
        raise RuntimeError(
            "Expected epoch {}, loader returned {}".format(expected_epoch, epoch)
        )
    return int(epoch), model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = args.model_dir.rstrip("/") + "/config.yaml"
    hypes = yaml_utils.load_yaml(config, None)
    hypes["fusion"]["core_method"] = "intermediatelate"

    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    batch = None
    for index, item in enumerate(loader):
        if index == args.sample_index:
            batch = train_utils.to_device(item, device)
            break
    if batch is None:
        raise IndexError("sample_index outside dataset")

    epoch, model = load_model(args.model_dir, hypes, device)
    with torch.no_grad():
        result = inference_utils_cosdh.inference_late_fusion(
            batch, model, dataset
        )

    info = getattr(model, "latest_paper_native_info", {}) or {}
    native = info.get("native_payload", {}) or {}
    metadata = native.get("metadata", {}) or {}
    mode = str(info.get("mode", "unknown"))
    links = info.get("channel_links", []) or []

    boundary_pass = bool(
        info.get("joint_transport_calls_this_frame") == 1
        and info.get("share_intermediate_late_budget") is True
        and info.get("no_policy") is True
        and info.get("extra_quantization") == "none"
        and info.get("extra_redundancy") == "none"
        and info.get("extra_selection") == "none"
        and metadata.get("intermediate_scale_count") == 3
        and metadata.get("intermediate_dtype") == "float16"
        and metadata.get("late_dtype") == "float32"
        and metadata.get("coordinates_in_byte_stream") is True
        and metadata.get("ucb_arce_used") is False
    )
    ideal_pass = True
    markov_pass = True
    if mode == "ideal":
        ideal_pass = bool(
            info.get("ideal_roundtrip_exact") is True
            and info.get("source_bytes") == info.get("sent_bytes_before_loss")
            and info.get("source_bytes") == info.get("received_valid_bytes")
        )
    elif mode == "markov":
        markov_pass = bool(
            links
            and all(link.get("state") in ("good", "medium", "bad") for link in links)
            and all(link.get("sent_bytes_before_loss", 0) <= link.get("budget_bytes", 0) for link in links)
        )
    else:
        ideal_pass = False
        markov_pass = False

    report = {
        "model_dir": args.model_dir,
        "epoch": epoch,
        "mode": mode,
        "executor_type": info.get("executor_type"),
        "predictions_present": result.get("pred_box_tensor") is not None,
        "boundary": {
            "joint_transport_calls_this_frame": info.get("joint_transport_calls_this_frame"),
            "share_intermediate_late_budget": info.get("share_intermediate_late_budget"),
            "intermediate_scale_count": metadata.get("intermediate_scale_count"),
            "late_segment_count": metadata.get("late_segment_count"),
            "intermediate_dtype": metadata.get("intermediate_dtype"),
            "late_dtype": metadata.get("late_dtype"),
            "coordinates_in_byte_stream": metadata.get("coordinates_in_byte_stream"),
            "ucb_arce_used": metadata.get("ucb_arce_used"),
            "pass": boundary_pass,
        },
        "bytes": {
            "source": info.get("source_bytes"),
            "sent_before_loss": info.get("sent_bytes_before_loss"),
            "received_valid": info.get("received_valid_bytes"),
            "intermediate_total": info.get("intermediate_total_bytes"),
            "late_total": info.get("late_total_bytes"),
            "coordinates": info.get("coordinate_bytes"),
            "headers": info.get("header_bytes"),
        },
        "ideal_roundtrip_exact": info.get("ideal_roundtrip_exact"),
        "channel_links": links,
    }
    report["overall_pass"] = bool(boundary_pass and ideal_pass and markov_pass)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["overall_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
