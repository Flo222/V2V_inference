#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode", choices=("ideal", "disabled"), default="ideal"
    )
    parser.add_argument(
        "--late-payload",
        choices=("candidate_records", "dense_heads"),
        default="candidate_records",
    )
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    path = Path(args.config).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if not args.no_backup:
        backup = path.with_name(
            path.name + ".official_faithful_" +
            datetime.now().strftime("%Y%m%d_%H%M%S") + ".bak"
        )
        shutil.copy2(str(path), str(backup))
        print("backup:", backup)

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    model_args = data["model"]["args"]
    enabled = args.mode == "ideal"
    model_args["cosdh_legacy_native"] = {
        "enabled": enabled,
        "mode": "ideal" if enabled else "disabled",
        "intermediate_enabled": enabled,
        "late_enabled": enabled,
        "late_payload_type": args.late_payload,
        "require_exact_roundtrip": True,
    }

    # The official-code-faithful audit must be isolated from all damage and
    # policy modules.  Their source files are not modified.
    for key in (
        "cosdh_paper_native", "arce", "cosdh_markov", "cosdh_late_markov"
    ):
        if isinstance(model_args.get(key), dict):
            model_args[key]["enabled"] = False

    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print("updated:", path)
    print("mode={}".format(args.mode))
    print("late_payload_type={}".format(args.late_payload))
    print("paper_native=false")
    print("markov=false")
    print("arce_ucb=false")


if __name__ == "__main__":
    main()
