#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import json
from pathlib import Path

import yaml


def contains(path, needle):
    return path.is_file() and needle in path.read_text(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    cfg_path = Path(args.config).resolve()
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    model_args = data["model"]["args"]
    cfg = model_args.get("cosdh_legacy_native", {})

    compressor = root / "opencood/models/baselines/cosdh/components/naive_compress_cosdh.py"
    model = root / "opencood/models/baselines/cosdh/models/point_pillar_cosdh_markov.py"
    transport = root / "opencood/models/baselines/cosdh/transport/cosdh_legacy_native_transport.py"
    helper = root / "opencood/models/baselines/cosdh/transport/cosdh_legacy_candidate_postprocess.py"
    inference = root / "opencood/tools/inference_utils_cosdh.py"

    report = {
        "codec_encode_boundary": contains(compressor, "def encode_for_wire"),
        "codec_decode_boundary": contains(compressor, "def decode_from_wire"),
        "transport_initialized": contains(model, "CosDHLegacyNativeTransport"),
        "intermediate_wire_boundary": contains(model, "roundtrip_intermediate"),
        "transport_file": contains(transport, "roundtrip_late_candidates"),
        "candidate_helper": contains(helper, "candidate_post_process_ideal"),
        "candidate_inference_boundary": (
            contains(inference, "COSDH_OFFICIAL_FAITHFUL_CANDIDATE_LATE")
            or contains(inference, "COSDH_LEGACY_CANDIDATE_IDEAL")
        ),
        "dense_late_boundary": contains(inference, "roundtrip_late_output"),
        "legacy_native_enabled": bool(cfg.get("enabled", False)),
        "legacy_native_mode_ideal": str(cfg.get("mode", "")).lower() == "ideal",
        "intermediate_enabled": bool(cfg.get("intermediate_enabled", False)),
        "late_enabled": bool(cfg.get("late_enabled", False)),
        "valid_late_payload": str(cfg.get("late_payload_type", "")).lower()
            in ("candidate_records", "dense_heads"),
        "paper_native_disabled": not bool(
            model_args.get("cosdh_paper_native", {}).get("enabled", False)
        ),
        "markov_disabled": not bool(
            model_args.get("cosdh_markov", {}).get("enabled", False)
        ),
        "late_markov_disabled": not bool(
            model_args.get("cosdh_late_markov", {}).get("enabled", False)
        ),
        "arce_disabled": not bool(
            model_args.get("arce", {}).get("enabled", False)
        ),
    }
    report["overall_pass"] = all(report.values())
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
