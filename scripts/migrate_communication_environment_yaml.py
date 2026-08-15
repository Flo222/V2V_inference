"""One-off, explicit migration for the shared Markov communication setup.

The target list is deliberately narrow: it contains the active Markov baseline
experiments plus the V2X-Real Where2Comm C2MAB experiment.  It moves the
physical link definition into ``communication_environment`` and removes the
same settings from model-local transport/ARCE configuration.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml


TARGETS = (
    "opencood/hypes_yaml/point_pillar_v2xvit_opv2v_arce_markov.yaml",
    "opencood/hypes_yaml/point_pillar_rocooper_opv2v_markov_sync_add_coopdiff.yaml",
    "opencood/hypes_yaml/v2xreal/point_pillar_v2xvit_native_payload_arce_markov_v2xreal_vc.yaml",
    "opencood/hypes_yaml/v2xreal/point_pillar_cosdh_markov_v2xreal_vc.yaml",
    "opencood/hypes_yaml/v2xreal/point_pillar_v2xvit_markov_v2xreal_vc.yaml",
    "opencood/hypes_yaml/v2xreal/point_pillar_diff_student_markov_v2xreal_vc.yaml",
    "opencood/hypes_yaml/v2xreal/point_pillar_rocooper_markov_v2xreal_vc.yaml",
    "opencood/hypes_yaml/opv2v/lidar_only/pointpillar_cosdh_markov.yaml",
    "opencood/hypes_yaml/v2xreal/point_pillar_where2comm_arce_c2mab_v2xreal_vc.yaml",
)

_PHYSICAL_KEYS = frozenset({
    "transition_matrix", "state_profiles", "state_params", "states",
    "initial_state", "init_state", "bandwidth_mbps", "packet_loss_rate",
    "packet_loss_mean", "packet_loss_std", "zero_fraction", "loss_rate",
    "plr", "delay_ms", "delay_mean_ms", "delay_std_ms",
    "max_delay_frames", "fixed_delay_ms", "bernoulli_loss_rates",
})


def shared_environment() -> Dict[str, Any]:
    """Canonical experiment-level Good/Medium/Bad Markov environment."""
    return {
        "enabled": True,
        "strict": True,
        "seed": 2026,
        "channel": {
            "mode": "markov",
            "initial_state": "medium",
            "frame_interval_ms": 100.0,
            "loss_model": "bernoulli",
            "transition_matrix": {
                "good": {"good": 0.85, "medium": 0.13, "bad": 0.02},
                "medium": {"good": 0.10, "medium": 0.80, "bad": 0.10},
                "bad": {"good": 0.03, "medium": 0.17, "bad": 0.80},
            },
            "bernoulli_loss_rates": {"good": 0.05, "medium": 0.20, "bad": 0.35},
            "profiles": {
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
        },
        "latency": {"enabled": True, "frame_interval_ms": 100.0},
    }


def strip_private_physics(value: Any) -> None:
    if isinstance(value, dict):
        for key in list(value):
            if key in _PHYSICAL_KEYS or key == "channel_state_markov":
                value.pop(key)
                continue
            strip_private_physics(value[key])
    elif isinstance(value, list):
        for child in value:
            strip_private_physics(child)


def migrate(hypes: Dict[str, Any]) -> Dict[str, Any]:
    hypes["communication_environment"] = shared_environment()
    wild = hypes.get("wild_setting")
    if isinstance(wild, dict):
        wild.pop("channel_state_markov", None)

    args = ((hypes.get("model") or {}).get("args") or {})
    if not isinstance(args, dict):
        return hypes

    # These names select baseline-native payload adapters.  They retain their
    # payload options, but no longer own the Markov state or physical profile.
    for key in ("cosdh_markov", "coopdiff_markov"):
        cfg = args.get(key)
        if isinstance(cfg, dict):
            strip_private_physics(cfg)

    coopdiff = args.get("coopdiff_markov")
    if isinstance(coopdiff, dict):
        # CoopDiff now delegates framing to ARCE's ByteStreamPacketizer.
        # These keys only described the removed cell-major serializer.
        packetization = coopdiff.get("packetization")
        if isinstance(packetization, dict):
            for key in (
                "bytes_per_value", "serialization_order", "selection_policy",
                "send_nonzero_only", "nonzero_epsilon",
            ):
                packetization.pop(key, None)

    rocooper = args.get("rocooper_comm")
    if isinstance(rocooper, dict):
        rocooper.pop("markov_channel", None)
        # RoCooper recreates its native payload operators from the injected
        # profile.  Keeping this block would leave hidden mean/std loss and
        # delay knobs that are outside the public environment.
        rocooper.pop("network_loss", None)
        rocooper.pop("channel_fading", None)

    arce = args.get("arce")
    if isinstance(arce, dict):
        # Executor construction can use defaults because the actual manager
        # is injected by train_utils immediately after model construction.
        arce.pop("channel", None)
        arce.pop("channel_state_markov", None)
        strip_private_physics(arce)

    # A small number of configs expose the same adapter config through a
    # top-level alias.  Clean it too, while leaving unrelated YAML untouched.
    for key in ("cosdh_markov", "coopdiff_markov"):
        cfg = hypes.get(key)
        if isinstance(cfg, dict):
            strip_private_physics(cfg)
    return hypes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    for relative in TARGETS:
        path = args.root / relative
        hypes = yaml.safe_load(path.read_text(encoding="utf-8"))
        if args.verify:
            from opencood.communication.experiment_channel import (
                build_experiment_channel_manager,
                validate_experiment_channel_configuration,
            )
            validate_experiment_channel_configuration(hypes)
            manager = build_experiment_channel_manager(hypes)
            assert manager is not None and manager.mode == "markov"
            assert manager.get_profile("bad")["bandwidth_mbps"] == 1.0
            print("verified", relative)
            continue
        migrated = migrate(hypes)
        rendered = yaml.safe_dump(migrated, allow_unicode=True, sort_keys=False)
        if args.apply:
            path.write_text(rendered, encoding="utf-8")
        print(("updated" if args.apply else "would update"), relative)


if __name__ == "__main__":
    main()
