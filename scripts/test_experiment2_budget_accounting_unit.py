#!/usr/bin/env python3
"""Smoke-test frame ids and source/parity budget accounting."""

from __future__ import annotations

import os
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from opencood.methods.arce.executors.fixed_executor import ARCEFixedComm


def make_cfg(out_dir: str, mode: str, rho: float, fec_type: str):
    return {
        "arce": {
            "enabled": True,
            "mode": "fixed",
            "policy": "fixed",
            "link_scope": "non_ego",
            "fixed_action": {
                "send": 1,
                "quant": mode,
                "quant_mode": mode,
                "rho": rho,
                "redundancy_ratio": rho,
                "cache": 0,
                "cache_enabled": 0,
                "fec_type": fec_type,
            },
            "quantization": {
                "enabled": mode != "fp32",
                "mode": mode,
                "granularity": "per_tensor",
                "compute_error": True,
                "pack_int4": mode == "int4",
            },
            "packetizer": {"packet_size_bytes": 128},
            "scheduler": {
                "budget_source": "system_budget",
                "budget_scope": "system_equal_split",
                "system_budget_mbps": 100000.0,
                "tx_window_ms": 100.0,
            },
            "channel": {
                "mode": "fixed",
                "bernoulli_loss_rates": {"good": 0.0, "medium": 0.0, "bad": 0.0},
            },
            "compression_audit": {
                "enabled": True,
                "strict": True,
                "experiment_name": "experiment2_unit",
                "output_dir": out_dir,
                "save_tensors": False,
                "require_no_budget_drop": False,
                "require_no_bernoulli_loss": True,
                "require_no_fec_parity": fec_type == "none",
                "require_all_source_transmitted": False,
                "require_quant_equals_recovered": False,
            },
        }
    }


def run_source_drop_case() -> None:
    with tempfile.TemporaryDirectory(prefix="arce_exp2_src_") as out_dir:
        comm = ARCEFixedComm(make_cfg(out_dir, "fp32", 0.0, "none"))
        x = torch.linspace(-3.0, 3.0, steps=8 * 9 * 10).reshape(8, 9, 10)
        _, record = comm.communicate_feature(
            x,
            link_id=(0, 0, 1),
            frame_id="sceneA/frame0001",
            agent_index=1,
            ego_index=0,
            channel_state="good",
            budget_bytes=4 * 128,
            update_cache=False,
        )
        size = record["size"]
        assert record["frame_id"] == "sceneA/frame0001"
        assert size["actual_num_source_packets"] == 23
        assert size["actual_num_transmitted_source_packets"] == 4
        assert size["num_source_dropped_by_budget"] == 19
        assert size["actual_num_transmitted_parity_packets"] == 0
        assert size["num_parity_dropped_by_budget"] == 0
        assert size["num_missing_by_budget"] == 19
        assert record["compression_audit"]["sanity_passed"]
        print("source-drop accounting OK")


def run_parity_drop_case() -> None:
    with tempfile.TemporaryDirectory(prefix="arce_exp2_parity_") as out_dir:
        comm = ARCEFixedComm(make_cfg(out_dir, "int8", 0.5, "raptor_sim"))
        x = torch.linspace(-3.0, 3.0, steps=8 * 9 * 10).reshape(8, 9, 10)
        _, record = comm.communicate_feature(
            x,
            link_id=(0, 0, 1),
            frame_id=17,
            agent_index=1,
            ego_index=0,
            channel_state="good",
            # INT8 gives 6 source packets. This budget sends all 6 source
            # packets and one repair packet, leaving only parity truncated.
            budget_bytes=7 * 128,
            update_cache=False,
        )
        size = record["size"]
        assert size["actual_num_source_packets"] == 6
        assert size["actual_num_parity_packets"] > 1
        assert size["actual_num_transmitted_source_packets"] == 6
        assert size["num_source_dropped_by_budget"] == 0
        assert size["actual_num_transmitted_parity_packets"] == 1
        assert size["num_parity_dropped_by_budget"] == (
            size["actual_num_parity_packets"] - 1
        )
        assert size["num_missing_by_budget"] == size["num_parity_dropped_by_budget"]
        assert record["compression_audit"]["sanity_passed"]
        print("parity-drop accounting OK")


def main() -> None:
    run_source_drop_case()
    run_parity_drop_case()
    print("Experiment 2 budget-accounting smoke test passed.")


if __name__ == "__main__":
    main()
