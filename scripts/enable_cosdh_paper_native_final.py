#!/usr/bin/env python3
from __future__ import print_function

import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["opv2v", "v2xreal"],
    )
    args = parser.parse_args()

    path = Path(args.config).expanduser().resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    model_args = data.setdefault("model", {}).setdefault("args", {})

    paper = model_args.setdefault("cosdh_paper_native", {})
    # Remove keys left by older rejected patches.
    paper.pop("coordinate_bytes_per_token", None)
    paper.update(
        {
            "enabled": True,
            "paper_native_in_train": False,
            "apply_transport_in_train": False,
            "fp16_wire": True,
            "fp16_in_train": False,
            "joint_frame_payload": True,
            "share_intermediate_late_budget": True,
            "identity_transport": False,
            "nonzero_epsilon": 0.0,
            "coordinate_encoding": "compact_indices_sidecar",
            "coordinate_bytes_per_cell": 4,
            "coordinate_budget_accounting":
                "reported_sidecar_not_in_ucb_action_cost",
        }
    )

    # Legacy/fallback semantics are made identical on both datasets too.
    late = model_args.setdefault("cosdh_late_markov", {})
    late["enabled"] = True
    late["share_profile_with_intermediate"] = True

    arce = model_args.setdefault("arce", {})
    arce["enabled"] = True
    arce["transport_mode"] = "compact_sparse"
    compact = arce.setdefault("compact_sparse", {})
    # Coordinates are represented by the adapter's compact indices, not by
    # an ARCE/UCB-specific coordinate option.
    compact.pop("coordinate_encoding", None)
    compact.pop("coordinate_bytes_per_token", None)
    compact.update(
        {
            "enabled": True,
            "source": "cosdh_paper_joint_scalar_mask",
            "threshold": 0.0,
            "budget_aware_topk": False,
            "empty_mask_policy": "zero_tokens",
        }
    )

    backup = path.with_suffix(
        path.suffix + ".before_cosdh_paper_native_final"
    )
    if not backup.exists():
        backup.write_text(
            path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    path.write_text(
        yaml.safe_dump(
            data,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    print("updated:", path)
    print("dataset={}".format(args.dataset))
    print("joint_frame_payload=true")
    print("share_intermediate_late_budget=true")
    print("cosdh_late_markov.share_profile_with_intermediate=true")
    print("ARCE/UCB core files are not modified.")


if __name__ == "__main__":
    main()
