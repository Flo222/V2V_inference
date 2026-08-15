#!/usr/bin/env python3
from __future__ import annotations

import torch

from opencood.methods.arce.audit.compression_auditor import _analyze_source_retention


def run_case(kind: str, stream: torch.Tensor, packet_size: int, tx_mask: torch.Tensor):
    source = torch.arange(32, dtype=torch.float32).reshape(4, 2, 4)
    valid_bytes = int(stream.numel() * stream.element_size())
    result = _analyze_source_retention(
        source_feature=source,
        source_tx_mask=tx_mask,
        packet_size_bytes=packet_size,
        valid_stream_bytes=valid_bytes,
        source_tensor_kind=kind,
        stream_tensor=stream,
    )
    assert result["available"] and result["layout_supported"]
    assert result["packet_selection_is_prefix"]
    ratios = result["channel_retention_ratio"]
    assert ratios == [1.0, 1.0, 0.0, 0.0], ratios
    assert abs(result["retained_value_ratio"] - 0.5) < 1e-12


def main() -> int:
    # FP32: 32 values * 4 bytes, 16-byte packets; first 4/8 packets retained.
    run_case(
        "q_tensor",
        torch.zeros(32, dtype=torch.float32),
        16,
        torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.bool),
    )
    # Packed INT4: 32 values -> 16 bytes, 4-byte packets; first 2/4 retained.
    run_case(
        "packed_int4",
        torch.zeros(16, dtype=torch.uint8),
        4,
        torch.tensor([1, 1, 0, 0], dtype=torch.bool),
    )
    print("Experiment 2.5 budget-retention mapping smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
