from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch


def _require_tensor(x: Any, name: str = "tensor") -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} should be a torch.Tensor, got {type(x)}.")
    return x


@dataclass
class BytePacketizationResult:
    packets: torch.Tensor
    valid_bytes: torch.Tensor
    original_num_bytes: int
    original_shape: Tuple[int, ...]
    original_dtype: torch.dtype
    packet_size_bytes: int
    source_tensor_kind: str = "q_tensor"

    @property
    def num_packets(self) -> int:
        return int(self.packets.shape[0])

    def to_meta_dict(self) -> Dict[str, Any]:
        return {
            "mode": "byte_stream",
            "source_tensor_kind": self.source_tensor_kind,
            "packet_size_bytes": int(self.packet_size_bytes),
            "num_packets": int(self.num_packets),
            "num_source_packets": int(self.num_packets),
            "original_num_bytes": int(self.original_num_bytes),
            "original_shape": tuple(int(x) for x in self.original_shape),
            "original_dtype": str(self.original_dtype),
            "valid_bytes_sum": int(self.valid_bytes.sum().item())
            if self.valid_bytes.numel() > 0
            else 0,
        }


class ByteStreamPacketizer:
    """
    Q(F) -> byte stream v -> fixed-size packets.

    Lp = 1024 Bytes by default.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        packet_cfg = cfg.get("packetizer", cfg)
        self.packet_size_bytes = int(
            packet_cfg.get("packet_size_bytes", packet_cfg.get("Lp", 1024))
        )
        if self.packet_size_bytes <= 0:
            raise ValueError(
                f"packet_size_bytes should be positive, got {self.packet_size_bytes}."
            )

    def tensor_to_bytes(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.detach().contiguous()
        return tensor.view(torch.uint8).flatten()

    def bytes_to_tensor(
        self,
        byte_stream: torch.Tensor,
        shape: Sequence[int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return byte_stream.contiguous().view(dtype).view(*shape)

    def packetize(
        self,
        tensor: torch.Tensor,
        source_tensor_kind: str = "q_tensor",
    ) -> BytePacketizationResult:
        tensor = _require_tensor(tensor, "tensor")
        byte_stream = self.tensor_to_bytes(tensor)
        num_bytes = int(byte_stream.numel())
        packet_size = int(self.packet_size_bytes)

        num_packets = int(math.ceil(num_bytes / packet_size)) if num_bytes > 0 else 0

        if num_packets == 0:
            packets = torch.empty(
                (0, packet_size),
                dtype=torch.uint8,
                device=tensor.device,
            )
            valid_bytes = torch.empty(
                (0,),
                dtype=torch.long,
                device=tensor.device,
            )
        else:
            padded_num_bytes = num_packets * packet_size
            padded = torch.zeros(
                (padded_num_bytes,),
                dtype=torch.uint8,
                device=tensor.device,
            )
            padded[:num_bytes] = byte_stream
            packets = padded.view(num_packets, packet_size)

            valid_bytes = torch.full(
                (num_packets,),
                packet_size,
                dtype=torch.long,
                device=tensor.device,
            )
            last_valid = num_bytes - (num_packets - 1) * packet_size
            valid_bytes[-1] = int(last_valid)

        return BytePacketizationResult(
            packets=packets,
            valid_bytes=valid_bytes,
            original_num_bytes=num_bytes,
            original_shape=tuple(int(x) for x in tensor.shape),
            original_dtype=tensor.dtype,
            packet_size_bytes=packet_size,
            source_tensor_kind=source_tensor_kind,
        )

    def unpacketize(
        self,
        packets: torch.Tensor,
        meta: BytePacketizationResult,
    ) -> torch.Tensor:
        packets = _require_tensor(packets, "packets")
        if meta.original_num_bytes == 0:
            return torch.empty(
                meta.original_shape,
                dtype=meta.original_dtype,
                device=packets.device,
            )

        byte_stream = packets.reshape(-1)[: meta.original_num_bytes]
        return self.bytes_to_tensor(
            byte_stream=byte_stream,
            shape=meta.original_shape,
            dtype=meta.original_dtype,
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "mode": "byte_stream",
            "packet_size_bytes": int(self.packet_size_bytes),
        }
