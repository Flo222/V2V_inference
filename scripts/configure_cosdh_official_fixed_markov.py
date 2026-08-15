#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import copy
import shutil
from datetime import datetime
from pathlib import Path

import yaml

DEFAULT_TRANSITION = {
    "good": {"good": 0.85, "medium": 0.13, "bad": 0.02},
    "medium": {"good": 0.10, "medium": 0.80, "bad": 0.10},
    "bad": {"good": 0.03, "medium": 0.17, "bad": 0.80},
}
DEFAULT_PROFILES = {
    "good": {"bandwidth_mbps": 27.0, "plr": 0.05, "loss_rate": 0.05, "delay_ms": 10.0, "fixed_delay_ms": 10.0, "temporal_source": "current"},
    "medium": {"bandwidth_mbps": 5.0, "plr": 0.20, "loss_rate": 0.20, "delay_ms": 50.0, "fixed_delay_ms": 50.0, "temporal_source": "current"},
    "bad": {"bandwidth_mbps": 1.0, "plr": 0.35, "loss_rate": 0.35, "delay_ms": 100.0, "fixed_delay_ms": 100.0, "temporal_source": "previous_frame"},
}


def _from_old_cosdh(old):
    old = old if isinstance(old, dict) else {}
    profiles = copy.deepcopy(DEFAULT_PROFILES)
    for state, item in (old.get("state_profiles", {}) or {}).items():
        if state not in profiles or not isinstance(item, dict):
            continue
        profiles[state].update(item)
        if "packet_loss_rate" in item:
            profiles[state]["plr"] = float(item["packet_loss_rate"])
            profiles[state]["loss_rate"] = float(item["packet_loss_rate"])
        if "delay_ms" in item:
            profiles[state]["fixed_delay_ms"] = float(item["delay_ms"])
    transition = copy.deepcopy(old.get("transition_matrix", DEFAULT_TRANSITION))
    initial = str(old.get("initial_state", "medium"))
    states = list(old.get("states", ["good", "medium", "bad"]))
    return profiles, transition, initial, states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=("ideal-check", "markov", "disabled"), default="markov")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    path = Path(args.config).expanduser().resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    model_args = data["model"]["args"]
    if not args.no_backup:
        backup = path.with_name(path.name + ".cosdh_fixed_markov_hotfix_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".bak")
        shutil.copy2(str(path), str(backup))
        print("backup:", backup)

    enabled = args.mode != "disabled"
    runtime_mode = "fixed_markov" if args.mode == "markov" else "ideal_check"
    model_args["cosdh_official_fixed_markov"] = {
        "enabled": enabled,
        "mode": runtime_mode if enabled else "disabled",
        "packet_size_bytes": 1024,
        "segment_order": ["scale0", "scale1", "scale2", "late_candidates"],
        "zero_fill_missing": True,
        "atomic_late_records": True,
    }

    old_markov = model_args.get("cosdh_markov", {}) or {}
    profiles, transition, initial, states = _from_old_cosdh(old_markov)

    # FixedARCEPolicy reads either arce.fixed_policy or top-level action keys.
    # Keep recovery as a string action name, never as a recovery-options dict.
    fixed_policy = {
        "enabled": True,
        "default_state": initial,
        "quant_mode": "fp32",
        "fec_type": "none",
        "redundancy_ratio": 0.0,
        "recovery": "zero_fill",
        "recovery_priority": ["zero_fill"],
        "use_feasibility_fallback": False,
    }

    model_args["arce"] = {
        "enabled": enabled,
        "mode": "fixed",
        "policy": "fixed",
        "seed": int(args.seed),
        "link_scope": "non_ego",
        "record_per_frame": True,
        "record_per_link": True,
        "late_policy": "allow",
        "enable_deadline_drop": False,
        "packetizer": {"mode": "byte_stream", "packet_size_bytes": 1024, "Lp": 1024},
        "scheduler": {
            "fps": 10.0,
            "tx_window_ms": 100.0,
            "frame_interval_ms": 100.0,
            "budget_source": "channel_profiles",
            "budget_scope": "global_sum_link",
            "system_budget_mbps": 5.0,
        },
        "channel": {
            "mode": "markov",
            "loss_model": "bernoulli",
            "latency_model": "fixed_state_delay",
            "profiles": profiles,
            "bernoulli_loss_rates": {s: float(profiles[s]["plr"]) for s in profiles},
            "fixed_delay_ms": {s: float(profiles[s]["delay_ms"]) for s in profiles},
            "states": states,
            "init_state": initial,
            "transition_matrix": transition,
        },
        "channel_state_markov": {
            "enabled": True,
            "states": states,
            "initial_state": initial,
            "transition_matrix": transition,
        },
        "fixed_policy": fixed_policy,
        # Audit-only mirror. The executor's policy parser uses fixed_policy above.
        "fixed_action": {
            "send": 1,
            "quant_mode": "fp32",
            "fec_type": "none",
            "redundancy_ratio": 0.0,
            "cache_enabled": 0,
            "recovery": "zero_fill",
            "recovery_priority": ["zero_fill"],
        },
        "fec": {"enabled": False},
        "redundancy": {"enabled": False, "ratio": 0.0},
        # Options are deliberately stored under recovery_config so they are not
        # mistaken for ARCEAction.recovery, which must be a string.
        "recovery_config": {
            "priority": ["zero_fill"],
            "zero_fill": True,
            "temporal_cache": False,
            "spatial_interpolation": False,
        },
    }

    for key in ("cosdh_legacy_native", "cosdh_paper_native", "cosdh_markov", "cosdh_late_markov"):
        if isinstance(model_args.get(key), dict):
            model_args[key]["enabled"] = False

    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print("updated:", path)
    print("mode={}".format(args.mode))
    print("packet_size_bytes=1024")
    print("segment_order=scale0,scale1,scale2,late_candidates")
    print("budget_scope=global_sum_link/fixed_equal_split")
    print("fixed_policy=fp32,no_fec,rho0,zero_fill")
    print("fixed_action=send1,rho0,cache0")
    print("old_cosdh_markov=false")
    print("legacy_ideal=false")
    print("arce_ucb=false")


if __name__ == "__main__":
    main()
