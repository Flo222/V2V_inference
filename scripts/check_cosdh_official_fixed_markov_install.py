#!/usr/bin/env python3
from __future__ import print_function
import argparse
import json
from pathlib import Path
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    cfg = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    ma = cfg["model"]["args"]
    model = (root / "opencood/models/baselines/cosdh/models/point_pillar_cosdh_markov.py").read_text(encoding="utf-8")
    inf = (root / "opencood/tools/inference_utils_cosdh.py").read_text(encoding="utf-8")
    t = ma.get("cosdh_official_fixed_markov", {}) or {}
    a = ma.get("arce", {}) or {}
    fp = a.get("fixed_policy", {}) or {}
    fa = a.get("fixed_action", {}) or {}
    checks = {
        "transport_file": (root / "opencood/models/baselines/cosdh/transport/cosdh_official_fixed_markov_transport.py").is_file(),
        "postprocess_file": (root / "opencood/models/baselines/cosdh/transport/cosdh_official_fixed_markov_postprocess.py").is_file(),
        "model_init": "COSDH_OFFICIAL_FIXED_MARKOV_INIT" in model,
        "joint_stream_model_path": "COSDH_OFFICIAL_FIXED_MARKOV_JOINT_STREAM" in model,
        "inference_reorder_path": "COSDH_OFFICIAL_FIXED_MARKOV_INFERENCE" in inf,
        "enabled": bool(t.get("enabled", False)),
        "valid_mode": str(t.get("mode", "")) in ("ideal_check", "fixed_markov"),
        "packet_1024": int(t.get("packet_size_bytes", 0)) == 1024 and int((a.get("packetizer", {}) or {}).get("packet_size_bytes", 0)) == 1024,
        "segment_order": list(t.get("segment_order", [])) == ["scale0", "scale1", "scale2", "late_candidates"],
        "budget_scope": str((a.get("scheduler", {}) or {}).get("budget_scope", "")) in ("global_sum_link", "system_equal_split"),
        "budget_source": str((a.get("scheduler", {}) or {}).get("budget_source", "")) == "channel_profiles",
        "fixed_policy_present": isinstance(fp, dict) and bool(fp),
        "fixed_policy_fp32": str(fp.get("quant_mode", "")) == "fp32",
        "fixed_policy_no_fec": str(fp.get("fec_type", "")) == "none",
        "fixed_policy_rho0": float(fp.get("redundancy_ratio", -1)) == 0.0,
        "fixed_policy_recovery_string": isinstance(fp.get("recovery"), str) and fp.get("recovery") == "zero_fill",
        "no_top_level_recovery_dict": not isinstance(a.get("recovery"), dict),
        "send1": int(fa.get("send", 0)) == 1,
        "rho0": float(fa.get("redundancy_ratio", -1)) == 0.0,
        "cache0": int(fa.get("cache_enabled", -1)) == 0,
        "old_markov_disabled": not bool((ma.get("cosdh_markov", {}) or {}).get("enabled", False)),
        "old_late_markov_disabled": not bool((ma.get("cosdh_late_markov", {}) or {}).get("enabled", False)),
        "legacy_ideal_disabled": not bool((ma.get("cosdh_legacy_native", {}) or {}).get("enabled", False)),
    }
    checks["overall_pass"] = all(checks.values())
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    raise SystemExit(0 if checks["overall_pass"] else 2)


if __name__ == "__main__":
    main()
