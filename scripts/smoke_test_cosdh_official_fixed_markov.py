#!/usr/bin/env python3
from __future__ import print_function
import torch
from opencood.models.baselines.cosdh.transport.cosdh_official_fixed_markov_transport import CosDHOfficialFixedMarkovTransport


def cfg(mode):
    return {
        "enabled": True, "mode": mode, "packet_size_bytes": 1024,
        "segment_order": ["scale0", "scale1", "scale2", "late_candidates"],
        "zero_fill_missing": True, "atomic_late_records": True,
    }


def arce():
    return {
        "enabled": True, "mode": "fixed", "policy": "fixed", "seed": 2026,
        "link_scope": "non_ego",
        "packetizer": {"mode": "byte_stream", "packet_size_bytes": 1024, "Lp": 1024},
        "scheduler": {"tx_window_ms": 100.0, "budget_source": "system_budget", "budget_scope": "system_equal_split", "system_budget_mbps": 1000.0},
        "channel": {"bernoulli_loss_rates": {"good": 0.0, "medium": 0.0, "bad": 0.0}, "fixed_delay_ms": {"good": 10.0, "medium": 50.0, "bad": 100.0}},
        "fixed_policy": {
            "enabled": True,
            "quant_mode": "fp32",
            "fec_type": "none",
            "redundancy_ratio": 0.0,
            "recovery": "zero_fill",
            "recovery_priority": ["zero_fill"],
        },
        "fixed_action": {"send": 1, "quant_mode": "fp32", "fec_type": "none", "redundancy_ratio": 0.0, "cache_enabled": 0},
        "fec": {"enabled": False},
        "recovery_config": {"priority": ["zero_fill"], "zero_fill": True, "temporal_cache": False, "spatial_interpolation": False},
    }


def main():
    tr = CosDHOfficialFixedMarkovTransport(cfg("ideal_check"), arce())
    record_len = torch.tensor([2])
    scales = [
        torch.randn(2, 4, 8, 8).half(),
        torch.randn(2, 8, 4, 4).half(),
        torch.randn(2, 16, 2, 2).half(),
    ]
    boxes = torch.randn(37, 7).float()
    scores = torch.rand(37).float()
    tr.start_frame(record_len=record_len, link_key_aliases=["ego", "cav1"], data_dict={})
    tr.set_late_candidates("cav1", boxes, scores)
    out = tr.communicate_joint_frame(scales, record_len, {}, ["ego", "cav1"])
    assert all(torch.equal(a, b) for a, b in zip(scales, out))
    rb, rs = tr.get_received_late_candidates("cav1")
    assert torch.equal(boxes, rb)
    assert torch.equal(scores, rs)
    assert tr.latest_info["packet_size_bytes"] == 1024
    assert tr.latest_info["num_links"] == 1
    print("SMOKE PASS")
    print(tr.latest_info)


if __name__ == "__main__":
    main()
