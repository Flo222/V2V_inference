#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify V2X-ViT native-payload integration on OPV2V or V2X-Real.

Checks:
  1. The original clean model and the new ARCE wrapper with ARCE disabled load
     the same checkpoint and produce equivalent psm/rm on the same batch.
  2. With ARCE enabled, the payload is emitted after shrink/native compressor,
     before regroup padding and before HxW prior repetition.
  3. Reported bytes separate the complete model tensor from actual non-ego
     sender-to-ego feature bytes.
"""

from __future__ import print_function

import argparse
import copy
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils


CLEAN_CORE = {
    "opv2v": "point_pillar_transformer_opv2v",
    "v2xreal": "point_pillar_transformer_v2xreal",
}
WRAPPER_CORE = {
    "opv2v": "point_pillar_transformer_opv2v_arce",
    "v2xreal": "point_pillar_transformer_v2xreal_arce",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_hypes(base, core_method, arce_enabled):
    hypes = copy.deepcopy(base)
    hypes.setdefault("model", {})["core_method"] = core_method
    args = hypes["model"].setdefault("args", {})
    arce = copy.deepcopy(args.get("arce", {}) or {})
    arce["enabled"] = bool(arce_enabled)
    arce["transport_mode"] = "payload_native"
    compact = arce.setdefault("compact_sparse", {})
    compact.update({
        "enabled": False,
        "source": "none",
        "budget_aware_topk": False,
    })
    payload = arce.setdefault("payload", {})
    payload.update({
        "interface": "native_payload_v1",
        "stage": "post_native_compressor",
        "prior_bytes_per_link": 12,
        "pose_bytes_per_link": 64,
    })
    args["arce"] = arce
    return hypes


def load_model(model_dir, hypes, device):
    model = train_utils.create_model(hypes)
    epoch, model = train_utils.load_saved_model(model_dir, model)
    model = model.to(device)
    model.eval()
    return int(epoch), model


def max_abs(a, b):
    return float((a.detach().float() - b.detach().float()).abs().max().item())


def allclose(a, b, atol, rtol):
    return bool(torch.allclose(
        a.detach().float(),
        b.detach().float(),
        atol=float(atol),
        rtol=float(rtol),
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--dataset", required=True, choices=["opv2v", "v2xreal"])
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument(
        "--skip_enabled_channel",
        action="store_true",
        help="Only run the clean-equivalence check.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = args.model_dir.rstrip("/") + "/config.yaml"
    base = yaml_utils.load_yaml(config_path, None)

    dataset = build_dataset(base, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=int(args.num_workers),
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    batch = None
    for index, item in enumerate(loader):
        if index == int(args.sample_index):
            batch = item
            break
    if batch is None:
        raise IndexError("sample_index is outside the dataset")

    batch = train_utils.to_device(batch, device)
    ego = batch["ego"]
    record_len = ego["record_len"]
    record_len_list = [
        int(v) for v in record_len.detach().cpu().reshape(-1).tolist()
    ]

    clean_hypes = clone_hypes(
        base,
        CLEAN_CORE[args.dataset],
        arce_enabled=False,
    )
    wrapper_off_hypes = clone_hypes(
        base,
        WRAPPER_CORE[args.dataset],
        arce_enabled=False,
    )

    clean_epoch, clean_model = load_model(
        args.model_dir, clean_hypes, device
    )
    wrapper_epoch, wrapper_off = load_model(
        args.model_dir, wrapper_off_hypes, device
    )

    with torch.no_grad():
        clean_out = clean_model(ego)
        wrapper_off_out = wrapper_off(ego)

    psm_diff = max_abs(clean_out["psm"], wrapper_off_out["psm"])
    rm_diff = max_abs(clean_out["rm"], wrapper_off_out["rm"])
    psm_equal = allclose(
        clean_out["psm"], wrapper_off_out["psm"],
        args.atol, args.rtol
    )
    rm_equal = allclose(
        clean_out["rm"], wrapper_off_out["rm"],
        args.atol, args.rtol
    )

    report = {
        "dataset": args.dataset,
        "model_dir": args.model_dir,
        "sample_index": int(args.sample_index),
        "device": str(device),
        "record_len": record_len_list,
        "clean_model_class": clean_model.__class__.__name__,
        "wrapper_model_class": wrapper_off.__class__.__name__,
        "clean_epoch": clean_epoch,
        "wrapper_epoch": wrapper_epoch,
        "clean_equivalence": {
            "psm_max_abs_diff": psm_diff,
            "rm_max_abs_diff": rm_diff,
            "psm_allclose": psm_equal,
            "rm_allclose": rm_equal,
            "atol": float(args.atol),
            "rtol": float(args.rtol),
            "pass": bool(psm_equal and rm_equal),
        },
    }

    if not args.skip_enabled_channel:
        wrapper_on_hypes = clone_hypes(
            base,
            WRAPPER_CORE[args.dataset],
            arce_enabled=True,
        )
        enabled_epoch, wrapper_on = load_model(
            args.model_dir, wrapper_on_hypes, device
        )
        with torch.no_grad():
            enabled_out = wrapper_on(ego)

        comm_info = enabled_out.get("comm_info")
        if not isinstance(comm_info, dict):
            raise RuntimeError("Enabled wrapper did not return dict comm_info")
        native = comm_info.get("native_payload")
        if not isinstance(native, dict):
            raise RuntimeError(
                "comm_info.native_payload is missing; communication boundary "
                "was not observable"
            )
        metadata = native.get("metadata", {}) or {}

        expected_collaborators = sum(max(0, n - 1) for n in record_len_list)
        expected_non_ego = (
            int(native["bytes_per_agent"]) * expected_collaborators
        )
        expected_aux = (
            int(native["per_link_aux_bytes_estimate"])
            * expected_collaborators
        )

        boundary_pass = (
            native.get("interface") == "native_payload_v1"
            and native.get("stage")
                == "post_shrink_and_native_compressor"
            and metadata.get("ego_transmitted") is False
            and metadata.get("max_cav_padding_transmitted") is False
            and metadata.get("prior_repeated_over_hw") is False
            and int(native.get("collaborator_count", -1))
                == expected_collaborators
            and int(native.get(
                "non_ego_feature_bytes_before_channel", -1
            )) == expected_non_ego
            and int(native.get("total_aux_bytes_estimate", -1))
                == expected_aux
        )

        report["enabled_channel"] = {
            "loaded_epoch": enabled_epoch,
            "payload": native,
            "expected_collaborators": expected_collaborators,
            "expected_non_ego_feature_bytes": expected_non_ego,
            "expected_aux_bytes": expected_aux,
            "boundary_pass": bool(boundary_pass),
        }

    report["overall_pass"] = bool(
        report["clean_equivalence"]["pass"]
        and (
            args.skip_enabled_channel
            or report["enabled_channel"]["boundary_pass"]
        )
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["overall_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
