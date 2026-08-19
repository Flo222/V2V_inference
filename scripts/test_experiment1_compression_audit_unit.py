#!/usr/bin/env python3
"""Small installation smoke test; no dataset or checkpoint is required."""

from __future__ import annotations

import os
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from opencood.methods.arce.executors.fixed_executor import ARCEFixedComm


def run_mode(mode: str) -> None:
    with tempfile.TemporaryDirectory(prefix="arce_exp1_") as out_dir:
        cfg = {
            "arce": {
                "enabled": True,
                "mode": "fixed",
                "policy": "fixed",
                "link_scope": "non_ego",
                "fixed_action": {
                    "send": 1,
                    "quant": mode,
                    "quant_mode": mode,
                    "rho": 0.0,
                    "redundancy_ratio": 0.0,
                    "cache": 0,
                    "cache_enabled": 0,
                    "fec_type": "none",
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
                    "output_dir": out_dir,
                    "save_tensors": False,
                },
            }
        }
        comm = ARCEFixedComm(cfg)
        x = torch.linspace(-3.0, 3.0, steps=8 * 9 * 10).reshape(8, 9, 10)
        y, record = comm.communicate_feature(
            x,
            link_id=(0, 0, 1),
            frame_id=0,
            agent_index=1,
            ego_index=0,
            channel_state="good",
            update_cache=False,
        )
        audit = record["compression_audit"]
        assert audit["requested_quant_mode"] == mode
        assert audit["actual_quant_mode"] == mode
        assert audit["clean_transport_allclose"]
        assert audit["sanity_passed"]
        if mode == "int4":
            assert record["packetization"]["source_tensor_kind"] == "packed_int4"
        print(
            "%s OK: valid_bytes=%s packets=%s quant_nmse=%.8g"
            % (
                mode,
                audit["quantized_valid_stream_bytes"],
                audit["source_packet_count"],
                audit["quant_nmse"],
            )
        )


def main() -> None:
    for mode in ("fp32", "fp16", "int8", "int4"):
        run_mode(mode)
    print("Experiment 1 audit installation smoke test passed.")


if __name__ == "__main__":
    main()
