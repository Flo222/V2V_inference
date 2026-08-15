#!/usr/bin/env python
from __future__ import print_function

import argparse
from pathlib import Path

import yaml


def set_disabled(mapping, key):
    value = mapping.get(key)
    if isinstance(value, dict):
        value["enabled"] = False


def main():
    parser = argparse.ArgumentParser(
        description="Configure CoSDH legacy-native Intermediate+Late Ideal."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode", choices=["disabled", "ideal"], required=True
    )
    args_cli = parser.parse_args()

    path = Path(args_cli.config).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    model_args = data["model"]["args"]
    enabled = args_cli.mode == "ideal"
    model_args["cosdh_legacy_native"] = {
        "enabled": enabled,
        "mode": "ideal" if enabled else "disabled",
        "intermediate_enabled": enabled,
        "late_enabled": enabled,
        "require_exact_roundtrip": True,
    }
    for key in (
        "cosdh_paper_native",
        "arce",
        "cosdh_markov",
        "cosdh_late_markov",
    ):
        set_disabled(model_args, key)

    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print("updated:", path)
    print("mode={}".format(args_cli.mode))
    print("legacy_native_enabled={}".format(str(enabled).lower()))
    print("intermediate_enabled={}".format(str(enabled).lower()))
    print("late_enabled={}".format(str(enabled).lower()))
    print("paper_native=false")
    print("markov=false")
    print("arce_ucb=false")


if __name__ == "__main__":
    main()
