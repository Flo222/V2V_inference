#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from opencood.methods.arce.executors.fixed_executor import ARCEFixedComm


def make_cfg(out_dir: str, quant: str, rho: float):
    fec_type = "none" if rho <= 0 else "raptor_sim"
    return {
        "arce": {
            "enabled": True,
            "mode": "fixed",
            "policy": "fixed",
            "link_scope": "non_ego",
            "seed": 2026,
            "fixed_action": {
                "send": 1,
                "quant": quant,
                "quant_mode": quant,
                "rho": rho,
                "redundancy_ratio": rho,
                "cache": 0,
                "cache_enabled": 0,
                "fec_type": fec_type,
                "decode_overhead": 0.0,
            },
            "quantization": {
                "enabled": quant != "fp32",
                "mode": quant,
                "granularity": "per_tensor",
                "compute_error": True,
                "pack_int4": quant == "int4",
            },
            "packetizer": {"packet_size_bytes": 64},
            "fec": {
                "enabled": rho > 0,
                "type": fec_type,
                "default_type": "raptor_sim",
                "redundancy_ratio": rho,
                "decode_overhead": 0.0,
                "degree_distribution": "robust_soliton",
                "seed": 2026,
            },
            "scheduler": {
                "budget_source": "system_budget",
                "budget_scope": "system_equal_split",
                # 0.5 Mbps x 100 ms = 6250 bytes for the single link.
                "system_budget_mbps": 0.5,
                "total_budget_mbps": 0.5,
                "tx_window_ms": 100.0,
            },
            "channel": {
                "mode": "fixed",
                "bernoulli_loss_rates": {"good": 0.20, "medium": 0.20, "bad": 0.20},
                "fixed_delay_ms": {"good": 0.0, "medium": 0.0, "bad": 0.0},
                "profiles": {
                    "good": {"bandwidth_mbps": 0.5, "plr": 0.20, "delay_ms": 0.0},
                    "medium": {"bandwidth_mbps": 0.5, "plr": 0.20, "delay_ms": 0.0},
                    "bad": {"bandwidth_mbps": 0.5, "plr": 0.20, "delay_ms": 0.0},
                },
            },
            "compression_audit": {"enabled": False},
            "fec_recovery_audit": {
                "enabled": True,
                "strict": True,
                "experiment_name": "experiment4_joint_compression_redundancy",
                "output_dir": out_dir,
                "file_name": "joint_audit.jsonl",
                "save_tensors": False,
                "require_no_budget_drop": False,
                "require_all_encoded_transmitted": False,
                "require_budget_not_exceeded": True,
            },
        }
    }


def run_case(quant: str, rho: float):
    out = tempfile.TemporaryDirectory(prefix="arce_exp4_")
    comm = ARCEFixedComm(make_cfg(out.name, quant, rho))
    x = torch.linspace(-4.0, 4.0, steps=16 * 16 * 16).reshape(16, 16, 16)
    _, record = comm.communicate_feature(
        x,
        link_id=(0, 0, 1),
        frame_id="scene/frame0001",
        agent_index=1,
        ego_index=0,
        channel_state="good",
        update_cache=False,
    )
    path = os.path.join(out.name, "joint_audit.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        audit = json.loads(next(line for line in f if line.strip()))
    return out, record, audit


def main() -> int:
    cases = {}
    handles = []
    try:
        for quant in ("fp16", "int8", "int4"):
            for rho in (0.0, 0.25, 0.60):
                out, record, audit = run_case(quant, rho)
                handles.append(out)
                cases[(quant, rho)] = (record, audit)
                assert audit["sanity"]["passed"] is True, audit
                assert audit["sanity"]["budget_not_exceeded"] is True, audit
                assert record["fec_recovery_audit"]["sanity_passed"] is True
                assert audit["budget"]["bandwidth_budget_bytes"] is not None

        for quant in ("fp16", "int8", "int4"):
            baseline = cases[(quant, 0.0)][1]
            prev_parity = -1
            prev_tx_parity = -1
            prev_final = -1.0
            prev_nmse = float("inf")
            for rho in (0.0, 0.25, 0.60):
                audit = cases[(quant, rho)][1]
                packet = audit["packet"]
                assert packet["num_source_packets"] == baseline["packet"]["num_source_packets"]
                assert packet["source_tx_fingerprint"] == baseline["packet"]["source_tx_fingerprint"]
                assert packet["source_loss_fingerprint"] == baseline["packet"]["source_loss_fingerprint"]
                assert packet["num_parity_packets"] >= prev_parity
                assert packet["num_transmitted_parity_packets"] >= prev_tx_parity
                assert packet["source_final_recovery_ratio"] + 1e-12 >= prev_final
                assert audit["fec_feature_error"]["nmse"] <= prev_nmse + 1e-10
                prev_parity = packet["num_parity_packets"]
                prev_tx_parity = packet["num_transmitted_parity_packets"]
                prev_final = packet["source_final_recovery_ratio"]
                prev_nmse = audit["fec_feature_error"]["nmse"]

        source_counts = [cases[(q, 0.25)][1]["packet"]["num_source_packets"] for q in ("fp16", "int8", "int4")]
        source_tx_ratios = [cases[(q, 0.25)][1]["packet"]["source_tx_ratio"] for q in ("fp16", "int8", "int4")]
        assert source_counts[0] > source_counts[1] > source_counts[2], source_counts
        assert source_tx_ratios[0] <= source_tx_ratios[1] <= source_tx_ratios[2], source_tx_ratios

        # The finite budget must produce at least one meaningful joint case:
        # INT4 sends parity, while FP16 remains more source-constrained.
        assert cases[("int4", 0.60)][1]["packet"]["num_transmitted_parity_packets"] > 0
        assert cases[("fp16", 0.60)][1]["packet"]["source_tx_ratio"] < cases[("int4", 0.60)][1]["packet"]["source_tx_ratio"]
    finally:
        for out in handles:
            out.cleanup()

    print("Experiment 4 joint compression/redundancy smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
