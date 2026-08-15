#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml


SUPPORTED_RECOVERY = {
    "none", "zero", "zero_fill", "spatial", "spatial_interpolation",
    "temporal", "temporal_cache", "arce",
}
OBSOLETE_REWARD_KEYS = {
    "alpha_q", "alpha_cost", "alpha_delay", "alpha_violation",
    "stale_norm_ms",
}
EXPECTED_PROFILES = {
    "good": {"bandwidth_mbps": 27.0, "plr": 0.05, "delay_ms": 10.0},
    "medium": {"bandwidth_mbps": 5.0, "plr": 0.20, "delay_ms": 50.0},
    "bad": {"bandwidth_mbps": 1.0, "plr": 0.35, "delay_ms": 100.0},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preflight a prepared final Markov+C2MAB runtime model."
    )
    p.add_argument("--model-dir", required=True)
    p.add_argument("--build-model", action="store_true")
    return p.parse_args()


def require_mapping(parent: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise RuntimeError("Expected mapping at {!r}".format(key))
    return value


def close(a: Any, b: float) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-9
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    config_path = model_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    checkpoints = sorted(model_dir.glob("net_epoch*.pth"))
    if not checkpoints:
        raise FileNotFoundError("No net_epoch*.pth in {}".format(model_dir))

    cfg = yaml.safe_load(config_path.read_text())
    model = require_mapping(cfg, "model")
    model_args = require_mapping(model, "args")
    arce = require_mapping(model_args, "arce")

    if str(arce.get("mode", "")).strip().lower() not in {"dc2mab", "c2mab"}:
        raise RuntimeError("Runtime config is not C2MAB")

    recovery = str(arce.get("recovery", "")).strip().lower()
    if recovery not in SUPPORTED_RECOVERY:
        raise RuntimeError("Unsupported scalar recovery method: {!r}".format(recovery))

    reward = require_mapping(arce, "reward")
    obsolete = sorted(OBSOLETE_REWARD_KEYS.intersection(reward))
    if obsolete:
        raise RuntimeError("Obsolete reward keys remain: {}".format(obsolete))
    for key in (
        "mode", "lambda_delta", "lambda_abs", "lambda_cost",
        "lambda_delay", "lambda_quant", "lambda_violate", "stale_max_ms",
    ):
        if key not in reward:
            raise RuntimeError("Missing runtime reward field: {}".format(key))

    action = require_mapping(arce, "action_space")
    required_action_keys = (
        "send_values",
        "online_quant_modes",
        "online_redundancy_ratios",
        "cache_values",
    )
    for key in required_action_keys:
        if key not in action:
            raise RuntimeError(
                "Missing current action-space key: {}".format(key)
            )

    send_values = {int(x) for x in action["send_values"]}
    declared_send = {
        int(x) for x in action.get("send", action["send_values"])
    }
    if not send_values.issubset(declared_send):
        raise RuntimeError(
            "Online send values are not covered by declared send values: "
            "{} vs {}".format(sorted(send_values), sorted(declared_send))
        )

    online_quant = {
        str(x).strip().lower()
        for x in action["online_quant_modes"]
    }
    declared_quant = {
        str(x).strip().lower()
        for x in action.get("quant", action["online_quant_modes"])
    }
    if not online_quant.issubset(declared_quant):
        raise RuntimeError(
            "Online quant modes are not covered by declared quant modes: "
            "{} vs {}".format(
                sorted(online_quant),
                sorted(declared_quant),
            )
        )
    if "fp32" in online_quant:
        raise RuntimeError(
            "FP32 must not appear in online C2MAB send actions."
        )

    online_rho = {
        float(x) for x in action["online_redundancy_ratios"]
    }
    declared_rho = {
        float(x)
        for x in action.get(
            "rho",
            action["online_redundancy_ratios"],
        )
    }
    if not online_rho.issubset(declared_rho):
        raise RuntimeError(
            "Online redundancy ratios are not covered by declared rho: "
            "{} vs {}".format(
                sorted(online_rho),
                sorted(declared_rho),
            )
        )

    cache_values = {int(x) for x in action["cache_values"]}
    declared_cache = {
        int(x)
        for x in action.get("cache", action["cache_values"])
    }
    if not cache_values.issubset(declared_cache):
        raise RuntimeError(
            "Online cache values are not covered by declared cache values: "
            "{} vs {}".format(
                sorted(cache_values),
                sorted(declared_cache),
            )
        )

    context = require_mapping(arce, "context")
    c2mab = require_mapping(arce, "c2mab")
    context_dim = int(c2mab.get("context_dim", 6))
    include_cav = bool(context.get("include_cav_confidence", True))
    expected_dim = 7 if include_cav else 6
    if context_dim != expected_dim:
        raise RuntimeError(
            "context_dim={} conflicts with include_cav_confidence={}".format(
                context_dim, include_cav
            )
        )
    for key in ("b_max_mbps", "stale_max_ms"):
        if key not in context:
            raise RuntimeError("Missing current context key: {}".format(key))

    channel = require_mapping(arce, "channel")
    profiles = require_mapping(channel, "profiles")
    for state, expected in EXPECTED_PROFILES.items():
        profile = require_mapping(profiles, state)
        plr = profile.get("plr", profile.get("loss_rate"))
        actual = {
            "bandwidth_mbps": profile.get("bandwidth_mbps"),
            "plr": plr,
            "delay_ms": profile.get("delay_ms"),
        }
        for key, target in expected.items():
            if not close(actual.get(key), target):
                raise RuntimeError(
                    "Profile mismatch {}.{}: {} != {}".format(
                        state, key, actual.get(key), target
                    )
                )

    result = {
        "static_config_pass": True,
        "model_dir": str(model_dir),
        "checkpoint": str(checkpoints[-1]),
        "recovery": recovery,
        "reward_mode": reward.get("mode"),
        "context_dim": context_dim,
        "include_cav_confidence": include_cav,
        "requested_quant_modes": action.get("online_quant_modes"),
        "requested_redundancy_ratios": action.get("online_redundancy_ratios"),
        "model_build_pass": None,
    }

    if args.build_model:
        import opencood.hypes_yaml.yaml_utils as yaml_utils
        from opencood.tools import train_utils

        hypes = yaml_utils.load_yaml(str(config_path))
        instance = train_utils.create_model(hypes)
        epoch, instance = train_utils.load_saved_model(str(model_dir), instance)
        comm = getattr(instance, "arce_comm", None)
        if comm is None:
            raise RuntimeError("Constructed model does not expose arce_comm")
        action_ids = list(getattr(comm, "action_ids", []) or [])
        if not action_ids:
            raise RuntimeError("Constructed C2MAB action space is empty")

        actual_quant = sorted({
            str(getattr(a, "quant_mode", "")).lower()
            for a in getattr(comm, "actions", [])
            if not bool(getattr(a, "is_no_send", False))
        })
        actual_rho = sorted({
            float(getattr(a, "redundancy_ratio", 0.0))
            for a in getattr(comm, "actions", [])
            if not bool(getattr(a, "is_no_send", False))
        })
        expected_quant = sorted(str(x).lower() for x in action["online_quant_modes"])
        expected_rho = sorted(float(x) for x in action["online_redundancy_ratios"])
        if actual_quant != expected_quant:
            raise RuntimeError(
                "Built quant actions differ: {} != {}".format(
                    actual_quant, expected_quant
                )
            )
        if actual_rho != expected_rho:
            raise RuntimeError(
                "Built rho actions differ: {} != {}".format(actual_rho, expected_rho)
            )
        if int(getattr(comm, "context_dim", -1)) != context_dim:
            raise RuntimeError(
                "Built context_dim differs: {} != {}".format(
                    getattr(comm, "context_dim", None), context_dim
                )
            )

        result.update({
            "model_build_pass": True,
            "loaded_epoch": int(epoch),
            "action_count": len(action_ids),
            "actual_quant_modes": actual_quant,
            "actual_redundancy_ratios": actual_rho,
            "actual_context_dim": int(getattr(comm, "context_dim", -1)),
            "actual_reward_mode": str(getattr(comm, "reward_mode", "")),
        })

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
