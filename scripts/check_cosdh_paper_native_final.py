#!/usr/bin/env python3
from __future__ import print_function

import argparse
import json
import subprocess

from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    hypes = yaml_utils.load_yaml(args.config, None)
    model = train_utils.create_model(hypes)
    transport = getattr(model, "cosdh_paper_transport", None)

    changed_arce = subprocess.check_output(
        ["git", "diff", "--name-only", "--", "opencood/methods/arce"],
        text=True,
    ).strip().splitlines()

    result = {
        "model_class": model.__class__.__name__,
        "paper_native_enabled": bool(
            getattr(model, "cosdh_paper_native_enabled", False)
        ),
        "paper_native_in_train": bool(
            getattr(model, "cosdh_paper_native_in_train", False)
        ),
        "compression_ratio": int(
            getattr(model, "compression_ratio", 0)
        ),
        "num_scales": int(len(getattr(model, "fusion_net", []))),
        "transport_executor": (
            transport.executor_type if transport is not None else None
        ),
        "joint_frame_payload": bool(
            getattr(model, "cosdh_paper_native_cfg", {}).get(
                "joint_frame_payload", False
            )
        ),
        "share_intermediate_late_budget": bool(
            getattr(model, "cosdh_paper_native_cfg", {}).get(
                "share_intermediate_late_budget", False
            )
        ),
        "legacy_share_profile": bool(
            getattr(
                model,
                "cosdh_late_markov_share_profile",
                False,
            )
        ),
        "arce_ucb_core_modified": changed_arce,
    }
    result["pass"] = bool(
        result["paper_native_enabled"]
        and result["compression_ratio"] == 16
        and result["num_scales"] == 3
        and result["joint_frame_payload"]
        and result["share_intermediate_late_budget"]
        and result["legacy_share_profile"]
        and not result["arce_ucb_core_modified"]
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
