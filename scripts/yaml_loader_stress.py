#!/usr/bin/env python3
"""Stress-test OpenCOOD YAML loading against a V2X-Real annotation tree."""
from __future__ import print_function

import argparse
from pathlib import Path
import yaml

from opencood.hypes_yaml.yaml_utils import load_yaml, _loader_with_float_resolver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--parse-only", action="store_true",
                        help="validate YAML syntax without applying yaml_parser")
    args = parser.parse_args()
    paths = sorted(args.root.rglob("*.yaml"))
    if not paths:
        raise SystemExit("No YAML files found")
    for index, path in enumerate(paths, 1):
        try:
            if args.parse_only:
                content = path.read_text(encoding="utf-8")
                base = yaml.Loader if "!!python/" in content else yaml.SafeLoader
                yaml.load(content, Loader=_loader_with_float_resolver(base))
            else:
                load_yaml(str(path))
        except Exception as exc:
            raise RuntimeError("YAML load failed at {}: {}".format(path, exc))
        if index % 1000 == 0:
            print("loaded {} / {}".format(index, len(paths)), flush=True)
    print("YAML_STRESS_OK count={}".format(len(paths)))


if __name__ == "__main__":
    main()
