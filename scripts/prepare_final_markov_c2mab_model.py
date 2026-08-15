#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
import shutil
from pathlib import Path
from typing import Any, Dict

import yaml


STATES = ["good", "medium", "bad"]
TRANSITION = {
    "good": {"good": 0.85, "medium": 0.13, "bad": 0.02},
    "medium": {"good": 0.10, "medium": 0.80, "bad": 0.10},
    "bad": {"good": 0.03, "medium": 0.17, "bad": 0.80},
}
PROFILES = {
    "good": {"bandwidth_mbps": 27.0, "loss_rate": 0.05, "plr": 0.05, "delay_ms": 10.0},
    "medium": {"bandwidth_mbps": 5.0, "loss_rate": 0.20, "plr": 0.20, "delay_ms": 50.0},
    "bad": {"bandwidth_mbps": 1.0, "loss_rate": 0.35, "plr": 0.35, "delay_ms": 100.0},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare an immutable runtime model directory for final Markov+C2MAB evaluation.")
    p.add_argument("--source-model-dir", required=True)
    p.add_argument("--runtime-model-dir", required=True)
    p.add_argument("--test-dir", required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--reward-profile",
        default="r2b",
        choices=("r2b", "preserve"),
        help=(
            "Effective reward schema for runtime config. r2b applies the current "
            "AP-delta-cost profile used by the reward refactor; preserve requires "
            "the source config to already use the current lambda_* schema."
        ),
    )
    return p.parse_args()


def require_mapping(parent: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = parent.setdefault(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping at {key}, got {type(value).__name__}")
    return value


def normalize_recovery_fields(arce: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize legacy/saved recovery config without changing semantics.

    Some saved C2MAB configs serialize the complete recovery parameter mapping
    into ``arce.recovery``.  ARCEFixedComm/FixedARCEPolicy expects that field to
    be a scalar method name, while detailed switches belong in
    ``arce.recovery_config``.  Keep the full mapping and derive the intended
    method deterministically.
    """
    raw = arce.get("recovery")
    details = arce.get("recovery_config")
    normalized = {
        "changed": False,
        "source_type": type(raw).__name__,
        "method": raw if isinstance(raw, str) else None,
    }

    if not isinstance(raw, dict):
        return normalized

    if not isinstance(details, dict):
        arce["recovery_config"] = copy.deepcopy(raw)

    priority = raw.get("priority", raw.get("recovery_priority", ()))
    if isinstance(priority, str):
        priority = [priority]
    if not isinstance(priority, (list, tuple)):
        priority = []

    aliases = {
        "temporal": "temporal_cache",
        "cache": "temporal_cache",
        "spatial": "spatial_interpolation",
        "zero": "zero_fill",
        "arce": "arce",
        "none": "none",
    }
    supported = {
        "none", "zero", "zero_fill", "spatial",
        "spatial_interpolation", "temporal", "temporal_cache", "arce",
    }

    method = None
    for item in priority:
        candidate = str(item).strip().lower()
        candidate = aliases.get(candidate, candidate)
        if candidate in supported:
            method = candidate
            break

    if method is None and bool(raw.get("temporal_cache", False)):
        method = "temporal_cache"
    if method is None:
        temporal_fusion = raw.get("temporal_fusion")
        if isinstance(temporal_fusion, dict) and bool(temporal_fusion.get("enabled", False)):
            method = "temporal_cache"
    if method is None and bool(raw.get("spatial_interpolation", False)):
        method = "spatial_interpolation"
    if method is None and bool(raw.get("zero_fill", False)):
        method = "zero_fill"
    if method is None:
        method = "zero_fill"

    arce["recovery"] = method
    normalized.update({
        "changed": True,
        "method": method,
        "recovery_config_preserved": True,
    })
    return normalized



OBSOLETE_REWARD_KEYS = {
    "alpha_q",
    "alpha_cost",
    "alpha_delay",
    "alpha_violation",
    "stale_norm_ms",
}

R2B_REWARD = {
    "mode": "ap_delta_cost",
    "lambda_abs": 0.1,
    "lambda_delta": 3.0,
    "lambda_cost": 0.1,
    "lambda_delay": 0.0,
    "lambda_quant": 0.0,
    "lambda_violate": 0.0,
    "stale_max_ms": 100.0,
}


def normalize_reward_fields(arce: Dict[str, Any], profile: str) -> Dict[str, Any]:
    """Create a runtime reward config compatible with the current C2MAB code.

    The saved ``final_proxy`` configuration belongs to the retired reward
    implementation.  Its alpha_* terms cannot be renamed one-for-one because
    the current reward has different semantics.  Therefore we either preserve
    an already-modern lambda_* configuration, or explicitly apply the tested
    R2b AP-delta-cost profile while retaining the full legacy mapping in the
    manifest for traceability.
    """
    raw = arce.get("reward", {}) or {}
    if not isinstance(raw, dict):
        raise TypeError(
            f"Expected arce.reward to be a mapping, got {type(raw).__name__}"
        )

    original = copy.deepcopy(raw)
    obsolete_present = sorted(OBSOLETE_REWARD_KEYS.intersection(raw))
    is_modern = not obsolete_present and any(
        key in raw
        for key in (
            "lambda_delta", "lambda_abs", "lambda_ap", "lambda_cost",
            "lambda_delay", "lambda_quant", "lambda_violate",
        )
    )

    if profile == "preserve":
        if not is_modern:
            raise RuntimeError(
                "REWARD_PROFILE=preserve was requested, but the source reward "
                "config is legacy/obsolete. Use REWARD_PROFILE=r2b or provide "
                "a source config that already uses lambda_* fields."
            )
        effective = copy.deepcopy(raw)
        changed = False
        reason = "source_already_uses_current_schema"
    elif profile == "r2b":
        effective = copy.deepcopy(R2B_REWARD)
        # tau_stale_ms is still consumed by receive-quality bookkeeping and is
        # orthogonal to the new reward equation, so retain it when available.
        effective["tau_stale_ms"] = float(raw.get("tau_stale_ms", 300.0))
        changed = effective != raw
        reason = (
            "legacy_final_proxy_replaced_with_current_r2b"
            if obsolete_present or not is_modern
            else "explicit_r2b_runtime_profile"
        )
    else:
        raise ValueError(f"Unsupported reward profile: {profile}")

    arce["reward"] = effective
    return {
        "changed": bool(changed),
        "profile": str(profile),
        "reason": reason,
        "obsolete_keys_found": obsolete_present,
        "original": original,
        "effective": copy.deepcopy(effective),
        "exact_legacy_equivalence": False if changed else True,
    }


def normalize_action_space_fields(arce: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate saved action-space fields to the current online schema.

    The saved ``final_36`` YAML belongs to the legacy action design and uses
    rho=[0, 0.25, 0.5].  The current reward_test branch intentionally defines
    the online C2MAB space as one canonical no-send arm plus
    3 quant modes x 4 rho values x 2 cache flags, where
    rho=[0, 0.10, 0.25, 0.60].  Several current modules explicitly reject
    rho=0.5 as a legacy arm.  Therefore the immutable runtime copy must migrate
    the legacy list instead of merely aliasing it, otherwise model construction
    silently drops rho=0.5 and produces an incomplete action space.
    """
    action = arce.get("action_space", {}) or {}
    if not isinstance(action, dict):
        raise TypeError(
            "Expected arce.action_space to be a mapping, got {}".format(
                type(action).__name__
            )
        )

    original = copy.deepcopy(action)
    populated = {}

    if "send_values" not in action:
        action["send_values"] = copy.deepcopy(action.get("send", [0, 1]))
        populated["send_values"] = "send_or_current_default"
    if "online_quant_modes" not in action:
        action["online_quant_modes"] = copy.deepcopy(
            action.get("quant", ["fp16", "int8", "int4"])
        )
        populated["online_quant_modes"] = "quant_or_current_default"
    if "cache_values" not in action:
        action["cache_values"] = copy.deepcopy(action.get("cache", [0, 1]))
        populated["cache_values"] = "cache_or_current_default"

    source_rho = action.get(
        "online_redundancy_ratios",
        action.get("rho", [0.0, 0.10, 0.25, 0.60]),
    )
    source_rho = [float(x) for x in source_rho]
    legacy_rho_050 = (
        any(abs(x - 0.50) < 1e-12 for x in source_rho)
        and not any(abs(x - 0.60) < 1e-12 for x in source_rho)
    )

    migration = None
    if legacy_rho_050:
        effective_rho = [0.0, 0.10, 0.25, 0.60]
        action["rho"] = copy.deepcopy(effective_rho)
        action["online_redundancy_ratios"] = copy.deepcopy(effective_rho)
        migration = {
            "changed": True,
            "reason": "legacy_rho_0p50_migrated_to_current_online_0p10_0p60_space",
            "source": source_rho,
            "effective": effective_rho,
            "expected_action_count": 25,
        }
    else:
        action["online_redundancy_ratios"] = copy.deepcopy(source_rho)
        if "rho" not in action:
            action["rho"] = copy.deepcopy(source_rho)
        migration = {
            "changed": False,
            "reason": "source_already_uses_current_online_rho_schema",
            "source": source_rho,
            "effective": copy.deepcopy(source_rho),
            "expected_action_count": 1 + (
                len(action["online_quant_modes"])
                * len(source_rho)
                * len(action["cache_values"])
            ),
        }

    arce["action_space"] = action
    return {
        "changed": bool(populated) or bool(migration["changed"]),
        "aliases_populated": populated,
        "rho_schema_migration": migration,
        "original": original,
        "effective": copy.deepcopy(action),
    }

def normalize_context_fields(arce: Dict[str, Any]) -> Dict[str, Any]:
    """Preserve the saved six-dimensional context under current defaults."""
    context = arce.get("context", {}) or {}
    if not isinstance(context, dict):
        raise TypeError(
            "Expected arce.context to be a mapping, got {}".format(
                type(context).__name__
            )
        )
    c2mab = arce.get("c2mab", {}) or {}
    if not isinstance(c2mab, dict):
        raise TypeError(
            "Expected arce.c2mab to be a mapping, got {}".format(
                type(c2mab).__name__
            )
        )

    original = copy.deepcopy(context)
    populated = {}

    # Current PDFContextBuilder consumes b_max_mbps/stale_max_ms.  Saved
    # configs may still use the older normalization names.  Materialize the
    # current keys unconditionally so preflight and runtime inspect the same
    # explicit values.  The fallbacks exactly match the defaults in the
    # current ARCEC2MABComm constructor.
    if "b_max_mbps" not in context:
        if "normalize_bandwidth_by_mbps" in context:
            context["b_max_mbps"] = float(context["normalize_bandwidth_by_mbps"])
            populated["b_max_mbps"] = "normalize_bandwidth_by_mbps"
        else:
            context["b_max_mbps"] = 27.0
            populated["b_max_mbps"] = "current_runtime_default"

    if "stale_max_ms" not in context:
        if "normalize_delay_by_ms" in context:
            context["stale_max_ms"] = float(context["normalize_delay_by_ms"])
            populated["stale_max_ms"] = "normalize_delay_by_ms"
        elif "deadline_ms" in context:
            context["stale_max_ms"] = float(context["deadline_ms"])
            populated["stale_max_ms"] = "deadline_ms"
        else:
            context["stale_max_ms"] = 400.0
            populated["stale_max_ms"] = "current_runtime_default"

    requested_dim = int(c2mab.get("context_dim", context.get("dim", 6)))
    if "include_cav_confidence" not in context:
        # In the current code, absence means True and forces context_dim=7.
        # The saved model explicitly records context_dim=6, so preserve that.
        context["include_cav_confidence"] = bool(requested_dim == 7)
        populated["include_cav_confidence"] = "derived_from_context_dim"

    expected_dim = 7 if bool(context.get("include_cav_confidence")) else 6
    if requested_dim != expected_dim:
        raise RuntimeError(
            "Inconsistent C2MAB context: context_dim={} but "
            "include_cav_confidence={} requires {}".format(
                requested_dim,
                bool(context.get("include_cav_confidence")),
                expected_dim,
            )
        )

    arce["context"] = context
    return {
        "changed": bool(populated),
        "fields_populated": populated,
        "requested_context_dim": requested_dim,
        "effective_context_dim": expected_dim,
        "original": original,
        "effective": copy.deepcopy(context),
    }

def link_checkpoints(source: Path, target: Path) -> int:
    count = 0
    for item in sorted(source.iterdir()):
        if not item.name.startswith("net_epoch") or item.suffix != ".pth":
            continue
        dst = target / item.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(item.resolve())
        count += 1
    if count == 0:
        raise FileNotFoundError(f"No net_epoch*.pth checkpoint found in {source}")
    return count


def main() -> None:
    args = parse_args()
    source = Path(args.source_model_dir).resolve()
    target = Path(args.runtime_model_dir).resolve()
    config_path = source / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    cfg = yaml.safe_load(config_path.read_text())
    if not isinstance(cfg, dict):
        raise TypeError("config.yaml must contain a mapping")

    model = require_mapping(cfg, "model")
    model_args = require_mapping(model, "args")

    # Current OPV2V/Where2Comm-ARCE configs store ARCE beside
    # where2comm_fusion at model.args.arce.  Older experiment snapshots may
    # store it under model.args.where2comm_fusion.arce, so retain a read-only
    # fallback for compatibility without creating a new empty mapping first.
    arce_path = "model.args.arce"
    arce = model_args.get("arce")
    if not isinstance(arce, dict):
        w2c = model_args.get("where2comm_fusion")
        legacy_arce = w2c.get("arce") if isinstance(w2c, dict) else None
        if isinstance(legacy_arce, dict):
            arce = legacy_arce
            arce_path = "model.args.where2comm_fusion.arce"
        else:
            raise RuntimeError(
                "Source model has no ARCE configuration at model.args.arce "
                "or model.args.where2comm_fusion.arce"
            )

    mode = str(arce.get("mode", "")).strip().lower()
    if mode not in {"dc2mab", "c2mab"}:
        raise RuntimeError(
            f"Source model is not a C2MAB model: {arce_path}.mode={mode!r}"
        )

    recovery_normalization = normalize_recovery_fields(arce)
    reward_normalization = normalize_reward_fields(arce, args.reward_profile)
    action_space_normalization = normalize_action_space_fields(arce)
    context_normalization = normalize_context_fields(arce)

    cfg["validate_dir"] = str(Path(args.test_dir).resolve())

    wild = require_mapping(cfg, "wild_setting")
    wild["seed"] = int(args.seed)
    wild_markov = require_mapping(wild, "channel_state_markov")
    wild_markov.update({
        "enabled": True,
        "scope": "link",
        "initial_state": "medium",
        "states": list(STATES),
        "transition_matrix": TRANSITION,
        "delay_ms": {k: v["delay_ms"] for k, v in PROFILES.items()},
        "profiles": PROFILES,
        "seed": int(args.seed),
    })

    arce["enabled"] = True
    arce["mode"] = "dc2mab"
    arce["seed"] = int(args.seed)
    arce["debug_records"] = False

    channel = require_mapping(arce, "channel")
    channel.update({
        "mode": "markov",
        "initial_state": "medium",
        "seed": int(args.seed),
        "states": list(STATES),
        "transition_matrix": [
            [0.85, 0.13, 0.02],
            [0.10, 0.80, 0.10],
            [0.03, 0.17, 0.80],
        ],
        "profiles": PROFILES,
    })

    arce["profiles"] = PROFILES
    scheduler = require_mapping(arce, "scheduler")
    scheduler.update({
        "fps": 10,
        "tx_window_ms": 100.0,
        "frame_interval_ms": 100.0,
        "budget_source": "channel_profiles",
        "budget_scope": "global_sum_link",
    })
    metrics = require_mapping(arce, "metrics")
    metrics.update({
        "save_frame_records": True,
        "save_link_records": True,
        "save_detection_state_index": True,
    })

    target.mkdir(parents=True, exist_ok=True)
    (target / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    )
    checkpoint_count = link_checkpoints(source, target)

    manifest = {
        "source_model_dir": str(source),
        "runtime_model_dir": str(target),
        "test_dir": str(Path(args.test_dir).resolve()),
        "seed": int(args.seed),
        "checkpoint_count": checkpoint_count,
        "channel_profiles": PROFILES,
        "transition_matrix": TRANSITION,
        "action_space_effective": arce.get("action_space"),
        "action_space_normalization": action_space_normalization,
        "context_effective": arce.get("context"),
        "context_normalization": context_normalization,
        "c2mab_config_preserved": arce.get("c2mab"),
        "reward_profile": str(args.reward_profile),
        "reward_config_effective": arce.get("reward"),
        "reward_normalization": reward_normalization,
        "recovery_method": arce.get("recovery"),
        "recovery_normalization": recovery_normalization,
        "recovery_config_preserved": arce.get("recovery_config"),
    }
    import json
    (target / "final_markov_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    print(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
