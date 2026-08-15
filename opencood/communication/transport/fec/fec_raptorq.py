"""RFC 6330 RaptorQ codec adapter used by GRACE block transport.

The external ``raptorq`` package performs the standardized codec work.  This
module only converts between PyTorch packet tensors and serialized RaptorQ
encoding packets.  It deliberately does not make scheduling or budget
decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


RAPTORQ_PAYLOAD_ID_BYTES = 4
GRACE_BLOCK_ID_BYTES = 2
GRACE_SOURCE_COUNT_BYTES = 2
GRACE_BLOCK_HEADER_BYTES = (
    GRACE_BLOCK_ID_BYTES + GRACE_SOURCE_COUNT_BYTES
)


def require_raptorq_backend():
    try:
        from raptorq import Decoder, Encoder
    except ImportError as exc:
        raise RuntimeError(
            "RFC 6330 RaptorQ backend is unavailable. Install the pinned "
            "Linux wheel with: pip install raptorq==1.6.3"
        ) from exc
    return Encoder, Decoder


def _validate_source_packets(source_packets: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(source_packets):
        raise TypeError("source_packets must be a torch.Tensor")
    if source_packets.dim() != 2:
        raise ValueError(
            "RaptorQ source packets must have shape [K, symbol_bytes], got {}"
            .format(tuple(source_packets.shape))
        )
    if source_packets.dtype != torch.uint8:
        raise ValueError(
            "RaptorQ source packets must use torch.uint8, got {}"
            .format(source_packets.dtype)
        )
    if int(source_packets.shape[0]) <= 0:
        raise ValueError("A RaptorQ source block must contain at least one packet")
    if int(source_packets.shape[1]) <= 0:
        raise ValueError("RaptorQ symbol size must be positive")
    return source_packets


def _row_to_bytes(row: torch.Tensor) -> bytes:
    row = row.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    return row.numpy().tobytes()


def _bytes_rows_to_tensor(rows: List[bytes], device: torch.device) -> torch.Tensor:
    if not rows:
        return torch.empty((0, 0), dtype=torch.uint8, device=device)
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise RuntimeError("RaptorQ produced variable-size encoding packets")
    flat = torch.tensor(
        list(b"".join(rows)),
        dtype=torch.uint8,
        device=device,
    )
    return flat.reshape(len(rows), width)


@dataclass(frozen=True)
class RaptorQEncodedBlock:
    block_id: int
    source_packets: torch.Tensor
    wire_packets: torch.Tensor
    num_repair_packets: int
    source_symbol_bytes: int
    grace_header_bytes: int = GRACE_BLOCK_HEADER_BYTES
    raptorq_payload_id_bytes: int = RAPTORQ_PAYLOAD_ID_BYTES

    @property
    def num_source_packets(self) -> int:
        return int(self.source_packets.shape[0])

    @property
    def num_encoded_packets(self) -> int:
        return int(self.wire_packets.shape[0])

    @property
    def wire_packet_bytes(self) -> int:
        return int(self.wire_packets.shape[1])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "block_id": int(self.block_id),
            "codec": "raptorq",
            "standard": "RFC6330",
            "is_standard_raptorq": True,
            "num_source_packets": int(self.num_source_packets),
            "num_repair_packets": int(self.num_repair_packets),
            "num_encoded_packets": int(self.num_encoded_packets),
            "source_symbol_bytes": int(self.source_symbol_bytes),
            "grace_header_bytes": int(self.grace_header_bytes),
            "raptorq_payload_id_bytes": int(self.raptorq_payload_id_bytes),
            "wire_packet_bytes": int(self.wire_packet_bytes),
        }


@dataclass
class RaptorQBlockDecodeResult:
    recovered_packets: torch.Tensor
    direct_received_source_mask: torch.Tensor
    fec_recovered_source_mask: torch.Tensor
    recovered_source_mask: torch.Tensor
    missing_source_mask: torch.Tensor
    receive_mask: torch.Tensor
    full_recovery: bool

    @property
    def num_fec_recovered_source_packets(self) -> int:
        return int(self.fec_recovered_source_mask.sum().item())


class RaptorQBlockCodec:
    """Encode and decode one independent GRACE protection block."""

    def __init__(self, source_symbol_bytes: int = 1024):
        self.source_symbol_bytes = int(source_symbol_bytes)
        if self.source_symbol_bytes <= 0 or self.source_symbol_bytes > 65535:
            raise ValueError("source_symbol_bytes must be in [1, 65535]")

    @property
    def wire_packet_bytes(self) -> int:
        return (
            self.source_symbol_bytes
            + RAPTORQ_PAYLOAD_ID_BYTES
            + GRACE_BLOCK_HEADER_BYTES
        )

    def encode_block(
        self,
        source_packets: torch.Tensor,
        num_repair_packets: int,
        block_id: int,
    ) -> RaptorQEncodedBlock:
        source_packets = _validate_source_packets(source_packets)
        if int(source_packets.shape[1]) != self.source_symbol_bytes:
            raise ValueError(
                "Source symbol width mismatch: expected {}, got {}"
                .format(self.source_symbol_bytes, int(source_packets.shape[1]))
            )
        num_repair_packets = int(num_repair_packets)
        if num_repair_packets < 0:
            raise ValueError("num_repair_packets must be non-negative")
        block_id = int(block_id)
        if block_id < 0 or block_id > 65535:
            raise ValueError("block_id must fit in two bytes")

        Encoder, _ = require_raptorq_backend()
        block_cpu = source_packets.detach().to(
            device="cpu", dtype=torch.uint8
        ).contiguous()
        data = block_cpu.numpy().tobytes()
        encoder = Encoder.with_defaults(data, self.source_symbol_bytes)
        serialized_packets = list(
            encoder.get_encoded_packets(num_repair_packets)
        )

        expected = int(source_packets.shape[0]) + num_repair_packets
        if len(serialized_packets) != expected:
            raise RuntimeError(
                "RaptorQ backend partitioned a GRACE block unexpectedly: "
                "expected {} packets, got {}. Reduce source_packets_per_block."
                .format(expected, len(serialized_packets))
            )

        expected_rq_bytes = self.source_symbol_bytes + RAPTORQ_PAYLOAD_ID_BYTES
        source_count = int(source_packets.shape[0])
        if source_count > 65535:
            raise ValueError("RaptorQ source_count must fit in two bytes")
        block_header = (
            block_id.to_bytes(GRACE_BLOCK_ID_BYTES, "big")
            + source_count.to_bytes(GRACE_SOURCE_COUNT_BYTES, "big")
        )
        wire_rows = []
        for packet in serialized_packets:
            packet = bytes(packet)
            if len(packet) != expected_rq_bytes:
                raise RuntimeError(
                    "Unexpected serialized RaptorQ packet size: expected {}, got {}"
                    .format(expected_rq_bytes, len(packet))
                )
            wire_rows.append(block_header + packet)

        wire_packets = _bytes_rows_to_tensor(
            wire_rows,
            device=source_packets.device,
        )
        return RaptorQEncodedBlock(
            block_id=block_id,
            source_packets=source_packets.clone(),
            wire_packets=wire_packets,
            num_repair_packets=num_repair_packets,
            source_symbol_bytes=self.source_symbol_bytes,
            grace_header_bytes=GRACE_BLOCK_HEADER_BYTES,
        )

    def decode_block(
        self,
        encoded_block: RaptorQEncodedBlock,
        receive_mask: torch.Tensor,
        fill_value: int = 0,
    ) -> RaptorQBlockDecodeResult:
        receive_mask = torch.as_tensor(
            receive_mask,
            dtype=torch.bool,
            device=encoded_block.wire_packets.device,
        ).flatten()
        if int(receive_mask.numel()) != encoded_block.num_encoded_packets:
            raise ValueError("RaptorQ block receive mask length mismatch")

        k = encoded_block.num_source_packets
        recovered = torch.full_like(encoded_block.source_packets, int(fill_value))
        direct = torch.zeros(k, dtype=torch.bool, device=recovered.device)

        for local_source_id in range(k):
            if bool(receive_mask[local_source_id].item()):
                row = encoded_block.wire_packets[local_source_id]
                payload_start = (
                    GRACE_BLOCK_HEADER_BYTES + RAPTORQ_PAYLOAD_ID_BYTES
                )
                recovered[local_source_id] = row[payload_start:]
                direct[local_source_id] = True

        _, Decoder = require_raptorq_backend()
        decoder = Decoder.with_defaults(
            k * self.source_symbol_bytes,
            self.source_symbol_bytes,
        )
        decoded: Optional[bytes] = None
        for encoded_id in range(encoded_block.num_encoded_packets):
            if not bool(receive_mask[encoded_id].item()):
                continue
            wire = _row_to_bytes(encoded_block.wire_packets[encoded_id])
            header_block_id = int.from_bytes(
                wire[:GRACE_BLOCK_ID_BYTES], "big"
            )
            if header_block_id != encoded_block.block_id:
                raise RuntimeError("RaptorQ block header mismatch")
            source_count_start = GRACE_BLOCK_ID_BYTES
            header_source_count = int.from_bytes(
                wire[
                    source_count_start:
                    source_count_start + GRACE_SOURCE_COUNT_BYTES
                ],
                "big",
            )
            if header_source_count != k:
                raise RuntimeError("RaptorQ source-count header mismatch")
            decoded = decoder.decode(wire[GRACE_BLOCK_HEADER_BYTES:])
            if decoded is not None:
                break

        fec_recovered = torch.zeros_like(direct)
        if decoded is not None:
            expected_bytes = k * self.source_symbol_bytes
            decoded = bytes(decoded)
            if len(decoded) != expected_bytes:
                raise RuntimeError(
                    "RaptorQ decoded length mismatch: expected {}, got {}"
                    .format(expected_bytes, len(decoded))
                )
            decoded_tensor = torch.tensor(
                list(decoded),
                dtype=torch.uint8,
                device=recovered.device,
            ).reshape(k, self.source_symbol_bytes)
            recovered.copy_(decoded_tensor)
            fec_recovered = ~direct

        recovered_mask = direct | fec_recovered
        missing = ~recovered_mask
        return RaptorQBlockDecodeResult(
            recovered_packets=recovered,
            direct_received_source_mask=direct,
            fec_recovered_source_mask=fec_recovered,
            recovered_source_mask=recovered_mask,
            missing_source_mask=missing,
            receive_mask=receive_mask,
            full_recovery=bool(recovered_mask.all().item()),
        )
