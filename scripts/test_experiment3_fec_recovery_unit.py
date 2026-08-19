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


def make_cfg(out_dir: str, rho: float):
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
                "quant": "int8",
                "quant_mode": "int8",
                "rho": rho,
                "redundancy_ratio": rho,
                "cache": 0,
                "cache_enabled": 0,
                "fec_type": fec_type,
                "decode_overhead": 0.0,
            },
            "quantization": {
                "enabled": True,
                "mode": "int8",
                "granularity": "per_tensor",
                "compute_error": True,
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
                "system_budget_mbps": 100000.0,
                "tx_window_ms": 100.0,
            },
            "channel": {
                "mode": "fixed",
                "bernoulli_loss_rates": {"good": 0.30, "medium": 0.30, "bad": 0.30},
                "fixed_delay_ms": {"good": 0.0, "medium": 0.0, "bad": 0.0},
            },
            "compression_audit": {"enabled": False},
            "fec_recovery_audit": {
                "enabled": True,
                "strict": True,
                "output_dir": out_dir,
                "file_name": "fec_recovery_audit.jsonl",
                "save_tensors": False,
                "require_no_budget_drop": True,
                "require_all_encoded_transmitted": True,
            },
        }
    }


def run_case(rho: float):
    out = tempfile.TemporaryDirectory(prefix="arce_exp3_")
    comm = ARCEFixedComm(make_cfg(out.name, rho))
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
    path = os.path.join(out.name, "fec_recovery_audit.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        audit = json.loads(next(line for line in f if line.strip()))
    return out, record, audit


def main() -> int:
    out0, record0, audit0 = run_case(0.0)
    out6, record6, audit6 = run_case(0.60)
    try:
        assert audit0["sanity"]["passed"] is True, audit0
        assert audit6["sanity"]["passed"] is True, audit6
        assert audit0["packet"]["num_parity_packets"] == 0, audit0
        assert audit6["packet"]["num_parity_packets"] > 0, audit6
        assert audit0["packet"]["source_loss_fingerprint"] == audit6["packet"]["source_loss_fingerprint"], (audit0, audit6)
        assert audit0["packet"]["num_source_packets"] == audit6["packet"]["num_source_packets"]
        assert audit6["packet"]["num_missing_source_packets"] <= audit6["packet"]["num_direct_missing_source_packets"]
        assert audit6["fec_feature_error"]["nmse"] <= audit6["direct_feature_error"]["nmse"] + 1e-10
        assert record6["fec_recovery_audit"]["sanity_passed"] is True
    finally:
        out0.cleanup(); out6.cleanup()
    print("Experiment 3 FEC recovery smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
