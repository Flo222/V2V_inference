#!/usr/bin/env python
from __future__ import print_function

import argparse
import json
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    checks = {}

    compressor = root / "opencood/models/baselines/cosdh/components/naive_compress_cosdh.py"
    model_file = root / "opencood/models/baselines/cosdh/models/point_pillar_cosdh_markov.py"
    transport = root / "opencood/models/baselines/cosdh/transport/cosdh_legacy_native_transport.py"

    compressor_text = compressor.read_text(encoding="utf-8")
    model_text = model_file.read_text(encoding="utf-8")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_args = cfg["model"]["args"]
    legacy_cfg = model_args.get("cosdh_legacy_native", {})

    checks["codec_boundary_methods"] = (
        "def encode_for_wire" in compressor_text
        and "def decode_from_wire" in compressor_text
    )
    checks["transport_file"] = transport.is_file()
    checks["model_import"] = "CosDHLegacyNativeTransport" in model_text
    checks["model_roundtrip_call"] = "roundtrip_intermediate" in model_text
    checks["legacy_native_enabled"] = bool(legacy_cfg.get("enabled", False))
    checks["legacy_native_mode_ideal"] = legacy_cfg.get("mode") == "ideal"
    checks["late_disabled"] = not bool(legacy_cfg.get("late_enabled", False))
    checks["paper_native_disabled"] = not bool(
        model_args.get("cosdh_paper_native", {}).get("enabled", False)
    )
    checks["markov_disabled"] = not bool(
        model_args.get("cosdh_markov", {}).get("enabled", False)
    )
    checks["arce_disabled"] = not bool(
        model_args.get("arce", {}).get("enabled", False)
    )
    checks["overall_pass"] = all(checks.values())
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    if not checks["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
