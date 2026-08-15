from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml


FIXED_ACTIONS: List[Dict[str, Any]] = [
    {
        "id": "fp32_none",
        "name": "fp32_none",
        "quant_mode": "fp32",
        "fec_type": "none",
        "redundancy_ratio": 0.0,
        "recovery": "arce",
    },
    {
        "id": "fp16_none",
        "name": "fp16_none",
        "quant_mode": "fp16",
        "fec_type": "none",
        "redundancy_ratio": 0.0,
        "recovery": "arce",
    },
    {
        "id": "int8_none",
        "name": "int8_none",
        "quant_mode": "int8",
        "fec_type": "none",
        "redundancy_ratio": 0.0,
        "recovery": "arce",
    },
    {
        "id": "int4_none",
        "name": "int4_none",
        "quant_mode": "int4",
        "fec_type": "none",
        "redundancy_ratio": 0.0,
        "recovery": "arce",
    },
    {
        "id": "int8_xor_r025",
        "name": "int8_xor_r025",
        "quant_mode": "int8",
        "fec_type": "xor",
        "xor_group_size": 4,
        "redundancy_ratio": 0.25,
        "recovery": "arce",
    },
    {
        "id": "int8_xor_r050",
        "name": "int8_xor_r050",
        "quant_mode": "int8",
        "fec_type": "xor",
        "xor_group_size": 4,
        "redundancy_ratio": 0.50,
        "recovery": "arce",
    },
    {
        "id": "int4_xor_r025",
        "name": "int4_xor_r025",
        "quant_mode": "int4",
        "fec_type": "xor",
        "xor_group_size": 4,
        "redundancy_ratio": 0.25,
        "recovery": "arce",
    },
    {
        "id": "int4_xor_r050",
        "name": "int4_xor_r050",
        "quant_mode": "int4",
        "fec_type": "xor",
        "xor_group_size": 4,
        "redundancy_ratio": 0.50,
        "recovery": "arce",
    },
    {
        "id": "int8_raptor_r025",
        "name": "int8_raptor_r025",
        "quant_mode": "int8",
        "fec_type": "raptor_sim",
        "redundancy_ratio": 0.25,
        "recovery": "arce",
    },
    {
        "id": "int8_raptor_r050",
        "name": "int8_raptor_r050",
        "quant_mode": "int8",
        "fec_type": "raptor_sim",
        "redundancy_ratio": 0.50,
        "recovery": "arce",
    },
    {
        "id": "int4_raptor_r025",
        "name": "int4_raptor_r025",
        "quant_mode": "int4",
        "fec_type": "raptor_sim",
        "redundancy_ratio": 0.25,
        "recovery": "arce",
    },
    {
        "id": "int4_raptor_r050",
        "name": "int4_raptor_r050",
        "quant_mode": "int4",
        "fec_type": "raptor_sim",
        "redundancy_ratio": 0.50,
        "recovery": "arce",
    },
]


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: str, cfg: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def get_arce_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    model = cfg.setdefault("model", {})
    args = model.setdefault("args", {})
    arce = args.setdefault("arce", {})
    return arce


def sync_top_arce(cfg: Dict[str, Any]) -> None:
    arce = get_arce_cfg(cfg)
    cfg["arce"] = copy.deepcopy(arce)


def clean_action(action: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(action)
    out.pop("id", None)
    return out


def make_strict_fixed_cfg(base_cfg: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    action_id = action["id"]

    cfg["name"] = f"{base_cfg.get('name', 'point_pillar_v2xvit_opv2v_arce')}_fixed_{action_id}"

    arce = get_arce_cfg(cfg)
    arce["enabled"] = True
    arce["mode"] = "fixed"
    arce["policy"] = "fixed"

    action_body = clean_action(action)

    # Strict fixed: Markov channel still changes, but good/medium/bad all use the same action.
    arce["fixed_policy"] = {
        "enabled": True,
        "default_state": "medium",
        "profiles": {
            "good": copy.deepcopy(action_body),
            "medium": copy.deepcopy(action_body),
            "bad": copy.deepcopy(action_body),
        },
    }

    arce.pop("random_policy", None)
    sync_top_arce(cfg)
    return cfg


def make_random_cfg(base_cfg: Dict[str, Any], seed: int, include_fp32: bool = False) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["name"] = f"{base_cfg.get('name', 'point_pillar_v2xvit_opv2v_arce')}_random_seed{seed}"

    arce = get_arce_cfg(cfg)
    arce["enabled"] = True
    arce["mode"] = "fixed"
    arce["policy"] = "random"

    action_space = []
    for action in FIXED_ACTIONS:
        if action["id"] == "fp32_none" and not include_fp32:
            continue
        action_space.append(clean_action(action))

    arce["random_policy"] = {
        "enabled": True,
        "seed": int(seed),
        "sample_mode": "uniform",
        "action_space": action_space,
    }

    # Keep fixed_policy for compatibility, but it will not be used when policy=random.
    sync_top_arce(cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        required=True,
        help="Base GitHub ARCE YAML, e.g. opencood/hypes_yaml/point_pillar_v2xvit_opv2v_arce.yaml",
    )
    parser.add_argument(
        "--out_dir",
        default="opencood/hypes_yaml/arce_baselines",
        help="Output directory.",
    )
    parser.add_argument(
        "--random_seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument(
        "--include_fp32_in_random",
        action="store_true",
        help="Whether random baseline can sample fp32_none.",
    )
    args = parser.parse_args()

    base_cfg = load_yaml(args.base)
    out_dir = Path(args.out_dir)
    fixed_dir = out_dir / "fixed_sweep"
    random_dir = out_dir / "random"

    for action in FIXED_ACTIONS:
        cfg = make_strict_fixed_cfg(base_cfg, action)
        path = fixed_dir / f"point_pillar_v2xvit_opv2v_arce_fixed_{action['id']}.yaml"
        save_yaml(str(path), cfg)

    for seed in args.random_seeds:
        cfg = make_random_cfg(
            base_cfg,
            seed=seed,
            include_fp32=args.include_fp32_in_random,
        )
        path = random_dir / f"point_pillar_v2xvit_opv2v_arce_random_seed{seed}.yaml"
        save_yaml(str(path), cfg)

    print(f"[OK] Fixed YAMLs saved to: {fixed_dir}")
    print(f"[OK] Random YAMLs saved to: {random_dir}")


if __name__ == "__main__":
    main()
