#!/usr/bin/env python3
from __future__ import print_function

import argparse
import copy
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from opencood.data_utils.datasets.cosdh_dataset_builder import build_dataset_cosdh as build_dataset
from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils
from opencood.tools import inference_utils_cosdh


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_dir, hypes, device):
    import re
    from pathlib import Path

    model = train_utils.create_model(hypes)
    model_dir_path = Path(model_dir)

    numeric_checkpoints = []
    non_numeric_checkpoints = []

    for checkpoint in model_dir_path.glob("net_epoch*.pth"):
        match = re.fullmatch(
            r"net_epoch(\d+)\.pth",
            checkpoint.name,
        )
        if match:
            numeric_checkpoints.append(
                (int(match.group(1)), checkpoint)
            )
        else:
            non_numeric_checkpoints.append(checkpoint)

    if not numeric_checkpoints:
        raise FileNotFoundError(
            "No numeric checkpoint such as "
            "net_epoch20.pth was found in {}".format(
                model_dir
            )
        )

    latest_epoch = max(
        numeric_checkpoints,
        key=lambda item: item[0],
    )[0]

    # This repository's load_saved_model() does not accept an epoch
    # argument and its checkpoint scanner crashes on names such as
    # net_epoch_bestval_at11.pth. Temporarily move only those
    # non-numeric files aside, call the original loader unchanged,
    # then restore them immediately.
    hidden = []

    try:
        for checkpoint in non_numeric_checkpoints:
            temporary = checkpoint.with_name(
                "." + checkpoint.name + ".verify_hidden"
            )
            checkpoint.rename(temporary)
            hidden.append((temporary, checkpoint))

        epoch, model = train_utils.load_saved_model(
            model_dir,
            model,
        )
    finally:
        for temporary, original in reversed(hidden):
            if temporary.exists():
                temporary.rename(original)

    if int(epoch) != int(latest_epoch):
        raise RuntimeError(
            "Expected latest numeric epoch {}, but loader returned {}"
            .format(latest_epoch, epoch)
        )

    return int(epoch), model.to(device).eval()


def tensor_close(a, b):
    if a is None or b is None:
        return a is None and b is None, None
    if tuple(a.shape) != tuple(b.shape):
        return False, None
    diff = float((a.detach().float() - b.detach().float()).abs().max().item())
    return bool(torch.allclose(
        a.detach().float(),
        b.detach().float(),
        atol=1e-6,
        rtol=1e-5,
    )), diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--run_channel", action="store_true")
    args = parser.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = args.model_dir.rstrip("/") + "/config.yaml"
    base = yaml_utils.load_yaml(config, None)
    base["fusion"]["core_method"] = "intermediatelate"

    dataset = build_dataset(base, visualize=False, train=False)
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
    for idx, item in enumerate(loader):
        if idx == args.sample_index:
            batch = train_utils.to_device(item, device)
            break
    if batch is None:
        raise IndexError("sample_index outside dataset")

    identity_hypes = copy.deepcopy(base)
    identity_args = identity_hypes["model"]["args"]
    identity_args["cosdh_paper_native"]["identity_transport"] = True
    identity_args["arce"]["enabled"] = True

    disabled_hypes = copy.deepcopy(base)
    disabled_args = disabled_hypes["model"]["args"]
    disabled_args["cosdh_paper_native"]["identity_transport"] = False
    disabled_args["arce"]["enabled"] = False

    identity_epoch, identity_model = load_model(
        args.model_dir, identity_hypes, device
    )
    disabled_epoch, disabled_model = load_model(
        args.model_dir, disabled_hypes, device
    )

    with torch.no_grad():
        identity_result = inference_utils_cosdh.inference_late_fusion(
            batch, identity_model, dataset
        )
        disabled_result = inference_utils_cosdh.inference_late_fusion(
            batch, disabled_model, dataset
        )

    boxes_ok, boxes_diff = tensor_close(
        identity_result.get("pred_box_tensor"),
        disabled_result.get("pred_box_tensor"),
    )
    scores_ok, scores_diff = tensor_close(
        identity_result.get("pred_score"),
        disabled_result.get("pred_score"),
    )

    info = getattr(identity_model, "latest_paper_native_info", {}) or {}
    payload = info.get("native_payload", {}) or {}
    metadata = payload.get("metadata", {}) or {}
    segments = metadata.get("segments", []) or []
    intermediate_segments = [
        item for item in segments
        if item.get("kind") == "intermediate"
    ]
    late_segments = [
        item for item in segments
        if item.get("kind") == "late"
    ]

    boundary_pass = bool(
        info.get("joint_transport_calls_this_frame") == 1
        and info.get("share_intermediate_late_budget") is True
        and len(intermediate_segments) == 3
        and len(late_segments) >= 2
        and metadata.get("selection_before_encoder") is True
        and metadata.get("fp16_before_wire") is True
        and metadata.get("decoder_after_wire") is True
        and metadata.get("transform_after_decoder") is True
        and metadata.get("late_dense_pre_nms") is True
        and metadata.get("confidence_filter_after_wire") is True
        and metadata.get("nms_after_wire") is True
    )

    report = {
        "model_dir": args.model_dir,
        "identity_epoch": identity_epoch,
        "disabled_epoch": disabled_epoch,
        "identity_equivalence": {
            "boxes_allclose": boxes_ok,
            "boxes_max_abs_diff": boxes_diff,
            "scores_allclose": scores_ok,
            "scores_max_abs_diff": scores_diff,
            "pass": bool(boxes_ok and scores_ok),
        },
        "boundary": {
            "joint_transport_calls_this_frame":
                info.get("joint_transport_calls_this_frame"),
            "share_intermediate_late_budget":
                info.get("share_intermediate_late_budget"),
            "intermediate_segments": len(intermediate_segments),
            "late_segments": len(late_segments),
            "coordinate_budget_accounting":
                metadata.get("coordinate_budget_accounting"),
            "pass": boundary_pass,
        },
    }

    if args.run_channel:
        channel_hypes = copy.deepcopy(base)
        channel_args = channel_hypes["model"]["args"]
        channel_args["cosdh_paper_native"]["identity_transport"] = False
        channel_args["arce"]["enabled"] = True
        channel_epoch, channel_model = load_model(
            args.model_dir, channel_hypes, device
        )
        with torch.no_grad():
            inference_utils_cosdh.inference_late_fusion(
                batch, channel_model, dataset
            )
        channel_info = getattr(
            channel_model, "latest_paper_native_info", {}
        ) or {}
        report["channel"] = {
            "epoch": channel_epoch,
            "executor_type": channel_info.get("executor_type"),
            "enabled": channel_info.get("enabled"),
            "joint_transport_calls_this_frame":
                channel_info.get("joint_transport_calls_this_frame"),
            "record_count": len(channel_info.get("records", []) or []),
            "pass": bool(
                channel_info.get(
                    "joint_transport_calls_this_frame"
                ) == 1
            ),
        }

    report["overall_pass"] = bool(
        report["identity_equivalence"]["pass"]
        and report["boundary"]["pass"]
        and (
            not args.run_channel
            or report.get("channel", {}).get("pass", False)
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["overall_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
