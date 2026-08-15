# -*- coding: utf-8 -*-
"""Prepare an evaluation directory for CoopDiff + Markov.

It copies/hardlinks the original CoopDiff checkpoint directory and rewrites only
config.yaml so the trained checkpoint is evaluated with point_pillar_diff_stu_markov.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml


def default_markov_cfg():
    return {
        "enabled": True,
        "impair_ego": False,
        "fps": 10.0,
        "states": ["good", "medium", "bad"],
        "initial_state": "medium",
        "transition_matrix": {
            "good": {"good": 0.85, "medium": 0.13, "bad": 0.02},
            "medium": {"good": 0.10, "medium": 0.80, "bad": 0.10},
            "bad": {"good": 0.03, "medium": 0.17, "bad": 0.80},
        },
        "state_profiles": {
            "good": {
                "bandwidth_mbps": 27.0,
                "packet_loss_rate": 0.05,
                "delay_ms": 10.0,
                "temporal_source": "current",
            },
            "medium": {
                "bandwidth_mbps": 5.0,
                "packet_loss_rate": 0.20,
                "delay_ms": 50.0,
                "temporal_source": "current",
            },
            "bad": {
                "bandwidth_mbps": 1.0,
                "packet_loss_rate": 0.35,
                "delay_ms": 100.0,
                "temporal_source": "previous_frame",
            },
        },
        "packetization": {
            "packet_size_bytes": 1024,
            "bytes_per_value": 4,
            "selection_policy": "raster",
            "zero_fill_missing": True,
        },
        # null/None means all multiscale features are transmitted through the channel.
        # Use [2] if you only want to impair CoopDiff's diffusion scale.
        "active_scales": None,
        "verbose": False,
    }


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(str(src), str(dst))
    except Exception:
        shutil.copy2(str(src), str(dst))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", required=True, help="Original CoopDiff checkpoint directory.")
    parser.add_argument("--dst_dir", required=True, help="New eval directory for CoopDiff-Markov.")
    parser.add_argument("--validate_dir", default=None, help="Override validate_dir in copied config.yaml.")
    parser.add_argument("--root_dir", default=None, help="Override root_dir in copied config.yaml.")
    parser.add_argument("--active_scales", default=None,
                        help="Comma-separated active scales. Default keeps config default: all scales. Example: 2")
    parser.add_argument("--disable_markov", action="store_true")
    args = parser.parse_args()

    src = Path(args.src_dir)
    dst = Path(args.dst_dir)
    if not src.exists():
        raise FileNotFoundError(src)
    if not (src / "config.yaml").exists():
        raise FileNotFoundError(src / "config.yaml")

    dst.mkdir(parents=True, exist_ok=True)

    # Copy/hardlink checkpoint files and lightweight metadata. Avoid copying npy/vis/log dirs.
    for item in src.iterdir():
        if item.is_dir():
            continue
        if item.name == "config.yaml":
            continue
        if item.suffix == ".pth" or item.name.startswith("net_epoch") or item.name == "latest.pth":
            hardlink_or_copy(item, dst / item.name)
        elif item.name.startswith("valid_ave_loss"):
            shutil.copy2(str(item), str(dst / item.name))

    cfg = yaml.load((src / "config.yaml").read_text(), Loader=yaml.Loader)
    cfg.setdefault("model", {}).setdefault("args", {})
    cfg.setdefault("fusion", {})

    cfg["model"]["core_method"] = "point_pillar_diff_stu_markov"
    cfg["fusion"]["core_method"] = "IntermediateFusionDatasetCoopdiff"
    cfg["name"] = cfg.get("name", src.name) + "_markov_eval"

    if args.validate_dir is not None:
        cfg["validate_dir"] = args.validate_dir
    if args.root_dir is not None:
        cfg["root_dir"] = args.root_dir

    markov_cfg = default_markov_cfg()
    if args.disable_markov:
        markov_cfg["enabled"] = False
    if args.active_scales is not None:
        raw = args.active_scales.strip()
        markov_cfg["active_scales"] = None if raw == "" else [int(x) for x in raw.split(",")]

    cfg["coopdiff_markov"] = markov_cfg
    cfg["model"]["args"]["coopdiff_markov"] = markov_cfg

    (dst / "config.yaml").write_text(yaml.dump(cfg, sort_keys=False, default_flow_style=False))

    print("[OK] prepared CoopDiff-Markov eval dir")
    print("src_dir:", src)
    print("dst_dir:", dst)
    print("model.core_method:", cfg["model"]["core_method"])
    print("fusion.core_method:", cfg["fusion"]["core_method"])
    print("validate_dir:", cfg.get("validate_dir"))
    print("markov.enabled:", cfg["coopdiff_markov"]["enabled"])
    print("markov.active_scales:", cfg["coopdiff_markov"].get("active_scales"))


if __name__ == "__main__":
    main()
