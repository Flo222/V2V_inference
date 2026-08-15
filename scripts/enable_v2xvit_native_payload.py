#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import copy
from pathlib import Path

import yaml


def default_arce(markov):
    return {
        "enabled": True,
        "mode": "fixed",
        "seed": 2026,
        "link_scope": "non_ego",
        "transport_mode": "payload_native",
        "payload": {
            "interface": "native_payload_v1",
            "stage": "post_native_compressor",
            "prior_bytes_per_link": 12,
            "pose_bytes_per_link": 64,
        },
        "channel_state_markov": copy.deepcopy(markov or {}),
        "channel": {
            "mode": "fixed",
            "fixed_state": "medium",
            "state_source": "dataset_link_markov_override",
        },
        "packetizer": {
            "mode": "byte_stream",
            "packet_size_bytes": 1024,
            "pad_value": 0.0,
        },
        "quantization": {
            "mode": "int8",
            "granularity": "per_tensor",
            "compute_error": True,
        },
        "fec": {
            "enabled": True,
            "use_padded_packet_size": True,
        },
        "recovery": "temporal_cache",
        "recovery_config": {
            "temporal_cache": True,
            "spatial_interpolation": True,
            "zero_fill": True,
        },
        "fixed_policy": {
            "profiles": {
                "good": {
                    "quant_mode": "fp16",
                    "fec_type": "none",
                    "redundancy_ratio": 0.0,
                    "recovery": "temporal_cache",
                },
                "medium": {
                    "quant_mode": "int8",
                    "fec_type": "xor",
                    "xor_group_size": 4,
                    "redundancy_ratio": 0.25,
                    "recovery": "temporal_cache",
                },
                "bad": {
                    "quant_mode": "int4",
                    "fec_type": "raptor_sim",
                    "redundancy_ratio": 0.6,
                    "recovery": "temporal_cache",
                },
            }
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True, choices=["opv2v", "v2xreal"])
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    path = Path(args.config).expanduser().resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    model = data.setdefault("model", {})
    model_args = model.setdefault("args", {})

    if args.dataset == "opv2v":
        model["core_method"] = "point_pillar_transformer_opv2v_arce"
    else:
        model["core_method"] = "point_pillar_transformer_v2xreal_arce"

    markov = (
        data.get("wild_setting", {})
        .get("channel_state_markov", {})
    )
    existing = model_args.get("arce", {})
    if not isinstance(existing, dict) or not existing:
        existing = default_arce(markov)

    existing["enabled"] = True
    existing["transport_mode"] = "payload_native"
    compact = existing.setdefault("compact_sparse", {})
    compact.update({
        "enabled": False,
        "source": "none",
        "budget_aware_topk": False,
    })
    payload = existing.setdefault("payload", {})
    payload.update({
        "interface": "native_payload_v1",
        "stage": "post_native_compressor",
        "prior_bytes_per_link": 12,
        "pose_bytes_per_link": 64,
    })
    if markov and not existing.get("channel_state_markov"):
        existing["channel_state_markov"] = copy.deepcopy(markov)
    model_args["arce"] = existing

    output = Path(args.output).expanduser().resolve() if args.output else path
    if output == path:
        backup = path.with_suffix(path.suffix + ".before_native_payload")
        if not backup.exists():
            backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    output.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(output)
    print("core_method={}".format(model["core_method"]))
    print("transport_mode={}".format(model_args["arce"]["transport_mode"]))


if __name__ == "__main__":
    main()
