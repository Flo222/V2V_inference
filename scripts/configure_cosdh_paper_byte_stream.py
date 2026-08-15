#!/usr/bin/env python3
from __future__ import print_function

import argparse
import copy
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", choices=["opv2v", "v2xreal"], required=True)
    parser.add_argument("--mode", choices=["ideal", "markov"], required=True)
    args = parser.parse_args()

    path = Path(args.config).expanduser().resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    model_args = data.setdefault("model", {}).setdefault("args", {})

    source_markov = copy.deepcopy(model_args.get("cosdh_markov", {}) or {})
    byte_channel = copy.deepcopy(source_markov)
    byte_channel["mode"] = args.mode
    byte_channel["enabled"] = args.mode == "markov"
    byte_channel["protect_headers"] = True
    packet = byte_channel.setdefault("packetization", {})
    packet.setdefault("packet_size_bytes", 1024)
    packet["zero_fill_missing"] = True

    paper = model_args.setdefault("cosdh_paper_native", {})
    paper.update(
        {
            "enabled": True,
            "paper_native_in_train": False,
            "apply_transport_in_train": False,
            "fp16_wire": True,
            "fp16_in_train": False,
            "joint_frame_payload": True,
            "share_intermediate_late_budget": True,
            "identity_transport": args.mode == "ideal",
            "channel_mode": args.mode,
            "nonzero_epsilon": 0.0,
            "byte_channel": byte_channel,
        }
    )

    # Keep both datasets on the same one-frame shared-budget interpretation.
    late = model_args.setdefault("cosdh_late_markov", {})
    late["enabled"] = args.mode == "markov"
    late["share_profile_with_intermediate"] = True

    # This experiment intentionally bypasses all ARCE/UCB logic.
    arce = model_args.setdefault("arce", {})
    arce["enabled"] = False

    backup = path.with_suffix(path.suffix + ".before_cosdh_byte_stream")
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print("updated:", path)
    print("dataset={}".format(args.dataset))
    print("channel_mode={}".format(args.mode))
    print("ucb_arce_enabled=false")
    print("joint_frame_payload=true")
    print("share_intermediate_late_budget=true")
    print("intermediate_dtype=float16")
    print("late_dtype=float32")


if __name__ == "__main__":
    main()
