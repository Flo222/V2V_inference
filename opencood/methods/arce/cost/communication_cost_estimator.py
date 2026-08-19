"""Communication cost estimator for GRACE / C2MAB.

This module estimates the byte-stream transmission cost of one ARCE action.
It is used by the oracle proposal stage before actual communication execution.

The estimator covers:
1. no-send action cost;
2. raw fp32 feature bytes;
3. quantization compression ratio;
4. packetization;
5. redundancy / parity packets;
6. budget-aware packet allocation;
7. estimated transmitted bytes.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Callable, Dict, Optional, Sequence


def _exact_raptorq_parity_packets(
    source_packets: int,
    rho: float,
    block_source_packets: int,
) -> int:
    """Estimate repair packets for exact-ratio protected blocks.

    Source symbols that cannot form one complete exact-ratio group are the
    best-effort tail and therefore do not receive repair symbols.
    """
    ratio = Fraction(str(float(rho))).limit_denominator(1000)
    source_unit = int(ratio.denominator)
    repair_unit = int(ratio.numerator)
    block_limit = max(1, int(block_source_packets))
    protected_per_block = block_limit - block_limit % source_unit
    if protected_per_block <= 0:
        return 0

    remaining = max(0, int(source_packets))
    parity_packets = 0
    while remaining >= source_unit:
        protected = min(protected_per_block, remaining)
        protected -= protected % source_unit
        if protected <= 0:
            break
        parity_packets += protected // source_unit * repair_unit
        remaining -= protected
    return int(parity_packets)


def estimate_byte_stream_fec_cost(
    feature_shape: Sequence[int],
    action: Any,
    budget_bytes: Optional[float],
    packet_size_bytes: int,
    metadata_bytes_per_packet: int,
    raw_feature_bytes_fp32_fn: Callable[[Sequence[int]], float],
    quant_ratio_to_fp32: Dict[str, float],
    compact_token_info: Optional[Dict[str, Any]] = None,
    raptorq_block_source_packets: int = 20,
    raptorq_metadata_bytes_per_packet: int = 8,
) -> Dict[str, Any]:
    if getattr(action, "is_no_send", False):
        return {
            "feasible": True,
            "send": 0,
            "quant_mode": str(getattr(action, "quant_mode", "fp32")),
            "fec_type": "none",
            "rho": 0.0,
            "raw_fp32_bytes": 0.0,
            "source_bytes": 0.0,
            "source_packets": 0,
            "parity_packets": 0,
            "encoded_packets": 0,
            "metadata_bytes": 0.0,
            "encoded_bytes": 0.0,
            "estimated_transmitted_bytes": 0.0,
            "max_tx_packets_under_budget": 0,
            "effective_packet_ratio": 0.0,
            "packet_size_bytes": int(packet_size_bytes),
            "budget_bytes": float(budget_bytes) if budget_bytes is not None else None,
            "proposal_budget_share": 0.0,
            "cost_model": "no_send",
            "compact_estimator": {},
            "compact_estimator_enabled": False,
            "compact_estimated_num_tokens": None,
            "compact_estimated_mask_ratio": None,
        }

    q = str(getattr(action, "quant_mode", "fp32")).strip().lower()
    quant_ratio = float(quant_ratio_to_fp32.get(q, 0.5))

    compact_token_info = compact_token_info or {}
    compact_enabled = bool(compact_token_info.get("compact_enabled", False))
    compact_tokens = compact_token_info.get("num_tokens", None)

    if compact_enabled and compact_tokens is not None and len(feature_shape) == 3:
        channels = int(feature_shape[0])
        raw_fp32 = float(channels * int(compact_tokens) * 4)
        cost_model = "compact_sparse_mask_aware"
    else:
        raw_fp32 = raw_feature_bytes_fp32_fn(feature_shape)
        cost_model = "allocation_aware_quant_share"

    source_bytes = float(raw_fp32 * quant_ratio)

    packet_size = int(packet_size_bytes)
    fec_type = str(getattr(action, "fec_type", "none")).strip().lower()
    is_raptorq = fec_type == "raptorq" and float(
        getattr(action, "redundancy_ratio", 0.0)
    ) > 0.0
    metadata_per_packet = max(0, int(metadata_bytes_per_packet))
    if is_raptorq:
        metadata_per_packet = max(
            metadata_per_packet,
            int(raptorq_metadata_bytes_per_packet),
        )
    packet_unit = float(packet_size + metadata_per_packet)

    source_packets = (
        int(math.ceil(source_bytes / max(packet_size, 1)))
        if source_bytes > 0
        else 0
    )

    rho = float(getattr(action, "redundancy_ratio", 0.0))
    if is_raptorq:
        parity_packets = _exact_raptorq_parity_packets(
            source_packets=source_packets,
            rho=rho,
            block_source_packets=raptorq_block_source_packets,
        )
    else:
        parity_packets = int(math.ceil(source_packets * max(rho, 0.0)))
    encoded_packets = int(source_packets + parity_packets)

    metadata_bytes = float(encoded_packets * metadata_per_packet)
    encoded_bytes = float(encoded_packets * packet_size + metadata_bytes)

    if budget_bytes is None:
        target_tx_bytes = float(encoded_bytes)
        proposal_share = 1.0
    else:
        budget = float(max(0.0, budget_bytes))
        proposal_share = float(quant_ratio) * float(1.0 + max(rho, 0.0))
        proposal_share = max(0.02, min(1.0, proposal_share))
        target_tx_bytes = float(min(encoded_bytes, budget * proposal_share))

        if target_tx_bytes > 0.0 and target_tx_bytes < packet_unit:
            target_tx_bytes = packet_unit

    if encoded_packets <= 0 or target_tx_bytes <= 0.0:
        max_tx_packets = 0
    else:
        max_tx_packets = int(
            min(
                encoded_packets,
                math.floor(target_tx_bytes / packet_unit),
            )
        )

    feasible = max_tx_packets > 0
    estimated_tx = float(max_tx_packets * packet_unit)
    effective_ratio = float(max_tx_packets / max(1, encoded_packets))

    return {
        "feasible": bool(feasible),
        "send": int(getattr(action, "send", 1)),
        "quant_mode": q,
        "fec_type": fec_type,
        "rho": float(rho),
        "raw_fp32_bytes": float(raw_fp32),
        "source_bytes": float(source_bytes),
        "source_packets": int(source_packets),
        "parity_packets": int(parity_packets),
        "encoded_packets": int(encoded_packets),
        "metadata_bytes": float(metadata_bytes),
        "encoded_bytes": float(encoded_bytes),
        "estimated_transmitted_bytes": float(estimated_tx),
        "max_tx_packets_under_budget": int(max_tx_packets),
        "effective_packet_ratio": float(effective_ratio),
        "packet_size_bytes": int(packet_size),
        "wire_packet_size_bytes": int(packet_size + metadata_per_packet),
        "raptorq_block_source_packets": (
            int(raptorq_block_source_packets) if is_raptorq else None
        ),
        "budget_bytes": float(budget_bytes) if budget_bytes is not None else None,
        "proposal_budget_share": float(proposal_share),
        "cost_model": str(cost_model),
        "compact_estimator": dict(compact_token_info),
        "compact_estimator_enabled": bool(compact_enabled),
        "compact_estimated_num_tokens": (
            int(compact_tokens) if compact_enabled and compact_tokens is not None else None
        ),
        "compact_estimated_mask_ratio": (
            float(compact_token_info.get("mask_ratio"))
            if compact_enabled and compact_token_info.get("mask_ratio") is not None
            else None
        ),
    }
