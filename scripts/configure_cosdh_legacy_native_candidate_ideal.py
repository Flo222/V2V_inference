#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("ideal", "disabled"), default="ideal")
    args = parser.parse_args()
    path = Path(args.config).expanduser().resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    model_args = data["model"]["args"]
    enabled = args.mode == "ideal"
    model_args["cosdh_legacy_native"] = {
        "enabled": enabled,
        "mode": "ideal" if enabled else "disabled",
        "intermediate_enabled": enabled,
        "late_enabled": enabled,
        "late_payload_type": "candidate_records",
        "require_exact_roundtrip": True,
    }
    for key in ("cosdh_paper_native", "arce", "cosdh_markov", "cosdh_late_markov"):
        if isinstance(model_args.get(key), dict):
            model_args[key]["enabled"] = False
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print("updated:", path)
    print("mode={}".format(args.mode))
    print("legacy_native_enabled={}".format(str(enabled).lower()))
    print("intermediate_enabled={}".format(str(enabled).lower()))
    print("late_enabled={}".format(str(enabled).lower()))
    print("late_payload_type=candidate_records")
    print("paper_native=false")
    print("markov=false")
    print("arce_ucb=false")


if __name__ == "__main__":
    main()
