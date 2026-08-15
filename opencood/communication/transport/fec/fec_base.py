"""
Base classes and common dataclasses for ARCE FEC modules.

This file defines the shared interface for packet-level FEC / redundancy
coding used by ARCE communication simulation.

Concrete implementations:
    fec_none.py
        No FEC. Source packets are transmitted directly.

    fec_xor.py
        Real XOR parity recovery over integer quantized packets.

    fec_raptor_sim.py
        Raptor / fountain-code style threshold simulation.

Mask convention:
    loss_mask[i] == True
        encoded packet i is lost.

    receive_mask[i] == True
        encoded packet i is received.

Tensor convention:
    source_packets:
        [K, ...]
        K is the number of source packets.

    encoded_packets:
        [N, ...]
        N = K + parity / repair packets.

    recovered_packets:
        [K, ...]
        Recovered source-packet tensor.

This file does NOT:
    - implement XOR parity;
    - implement Raptor simulation;
    - sample packet loss;
    - quantize features;
    - reconstruct missing spatial regions.

Those are handled by concrete FEC modules, channel modules,
compression modules, and recovery modules.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

from opencood.communication.transport.fec import (
    FEC_TYPE_NONE,
    FEC_TYPE_XOR,
    FEC_TYPE_RAPTOR_SIM,
    DEFAULT_FEC_TYPE,
    DEFAULT_REDUNDANCY_RATIO,
    DEFAULT_XOR_GROUP_SIZE,
    DEFAULT_RAPTOR_DECODE_OVERHEAD,
    normalize_fec_type,
    normalize_fec_config,
    normalize_redundancy_ratio,
    normalize_group_size,
    normalize_decode_overhead,
    estimate_encoded_packets,
    effective_redundancy_ratio,
)


def _require_tensor(x: Any, name: str = "tensor") -> torch.Tensor:
    """
    Validate torch.Tensor input.
    """
    if not torch.is_tensor(x):
        raise TypeError(f"{name} should be a torch.Tensor, got {type(x)}.")
    return x


def _as_non_negative_int(value: Any, name: str) -> int:
    """
    Convert value to non-negative int.
    """
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} should be convertible to int, got {value}.")

    if value < 0:
        raise ValueError(f"{name} should be non-negative, got {value}.")

    return value


def _as_positive_int(value: Any, name: str) -> int:
    """
    Convert value to positive int.
    """
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} should be convertible to int, got {value}.")

    if value <= 0:
        raise ValueError(f"{name} should be positive, got {value}.")

    return value


def _as_bool_tensor(
    mask: Any,
    expected_len: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    name: str = "mask",
) -> torch.Tensor:
    """
    Convert mask to torch.BoolTensor.

    Parameters
    ----------
    mask : tensor-like
        Boolean mask.

    expected_len : int, optional
        Expected flattened length.

    device : torch.device or str, optional
        Target device.

    name : str
        Name used in error messages.

    Returns
    -------
    torch.BoolTensor
        Flattened boolean tensor.
    """
    if torch.is_tensor(mask):
        out = mask.to(dtype=torch.bool)
        if device is not None:
            out = out.to(device=device)
    else:
        out = torch.as_tensor(mask, dtype=torch.bool, device=device)

    out = out.flatten()

    if expected_len is not None and out.numel() != int(expected_len):
        raise ValueError(
            f"{name} length mismatch: expected {expected_len}, "
            f"got {out.numel()}."
        )

    return out


def loss_mask_to_receive_mask(
    loss_mask: Any,
    expected_len: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Convert loss mask to receive mask.

    loss_mask[i] == True means encoded packet i is lost.
    receive_mask[i] == True means encoded packet i is received.
    """
    loss_mask = _as_bool_tensor(
        loss_mask,
        expected_len=expected_len,
        device=device,
        name="loss_mask",
    )
    return ~loss_mask


def receive_mask_to_loss_mask(
    receive_mask: Any,
    expected_len: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Convert receive mask to loss mask.

    receive_mask[i] == True means encoded packet i is received.
    loss_mask[i] == True means encoded packet i is lost.
    """
    receive_mask = _as_bool_tensor(
        receive_mask,
        expected_len=expected_len,
        device=device,
        name="receive_mask",
    )
    return ~receive_mask


def resolve_receive_mask(
    num_encoded_packets: int,
    receive_mask: Optional[Any] = None,
    loss_mask: Optional[Any] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Resolve receive mask from receive_mask or loss_mask.

    Priority:
        1. receive_mask
        2. loss_mask
        3. all packets received

    Parameters
    ----------
    num_encoded_packets : int
        Number of encoded packets N.

    receive_mask : tensor-like, optional
        True means received.

    loss_mask : tensor-like, optional
        True means lost.

    device : torch.device or str, optional
        Target device.

    Returns
    -------
    torch.BoolTensor
        Shape [N]. True means received.
    """
    n = _as_non_negative_int(num_encoded_packets, "num_encoded_packets")

    if receive_mask is not None:
        return _as_bool_tensor(
            receive_mask,
            expected_len=n,
            device=device,
            name="receive_mask",
        )

    if loss_mask is not None:
        return loss_mask_to_receive_mask(
            loss_mask,
            expected_len=n,
            device=device,
        )

    return torch.ones(n, dtype=torch.bool, device=device)


def resolve_loss_mask(
    num_encoded_packets: int,
    receive_mask: Optional[Any] = None,
    loss_mask: Optional[Any] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Resolve loss mask from loss_mask or receive_mask.

    Priority:
        1. loss_mask
        2. receive_mask
        3. no packet lost

    Returns
    -------
    torch.BoolTensor
        Shape [N]. True means lost.
    """
    n = _as_non_negative_int(num_encoded_packets, "num_encoded_packets")

    if loss_mask is not None:
        return _as_bool_tensor(
            loss_mask,
            expected_len=n,
            device=device,
            name="loss_mask",
        )

    if receive_mask is not None:
        return receive_mask_to_loss_mask(
            receive_mask,
            expected_len=n,
            device=device,
        )

    return torch.zeros(n, dtype=torch.bool, device=device)


def validate_packet_tensor(
    packets: torch.Tensor,
    name: str = "packets",
    min_dim: int = 1,
) -> torch.Tensor:
    """
    Validate packet tensor.

    The first dimension is always packet index.

    Examples:
        [K, C, H, W]
        [K, C, packet_h, packet_w]
        [K, D]
    """
    packets = _require_tensor(packets, name)

    if packets.dim() < min_dim:
        raise ValueError(
            f"{name} should have at least {min_dim} dimension(s), "
            f"got shape {tuple(packets.shape)}."
        )

    if packets.shape[0] < 0:
        raise ValueError(f"{name} first dimension should be non-negative.")

    return packets


def count_mask(mask: torch.Tensor) -> int:
    """
    Count True values in a boolean mask.
    """
    mask = _as_bool_tensor(mask, name="mask")
    return int(mask.sum().item())


@dataclass(frozen=True)
class EncodedPacketMeta:
    """
    Metadata for one encoded packet.

    Attributes
    ----------
    encoded_id : int
        Index in encoded_packets.

    kind : str
        "source", "parity", or "repair".

    source_id : int or None
        Source-packet id if this encoded packet is a direct source packet.

    group_id : int or None
        Group id for XOR parity.

    source_ids : tuple
        Source ids covered by this encoded packet.
        For a source packet, this is usually (source_id,).
        For XOR parity, this is the group source ids.
        For Raptor-sim repair packet, this can be empty or all source ids.

    note : str
        Optional human-readable note.
    """

    encoded_id: int
    kind: str
    source_id: Optional[int] = None
    group_id: Optional[int] = None
    source_ids: Tuple[int, ...] = field(default_factory=tuple)
    note: str = ""

    @property
    def is_source(self) -> bool:
        return self.kind == "source"

    @property
    def is_parity(self) -> bool:
        return self.kind == "parity"

    @property
    def is_repair(self) -> bool:
        return self.kind == "repair"

    def as_dict(self) -> Dict[str, Any]:
        """
        Export as JSON-serializable dict.
        """
        return {
            "encoded_id": int(self.encoded_id),
            "kind": self.kind,
            "source_id": None if self.source_id is None else int(self.source_id),
            "group_id": None if self.group_id is None else int(self.group_id),
            "source_ids": tuple(int(x) for x in self.source_ids),
            "note": self.note,
        }


@dataclass
class FECEncodeResult:
    """
    Result of FEC encoding.

    Attributes
    ----------
    source_packets : torch.Tensor
        Original source packets, shape [K, ...].

    encoded_packets : torch.Tensor
        Source + parity / repair packets, shape [N, ...].

    encoded_metas : list of EncodedPacketMeta
        Metadata for each encoded packet.

    fec_type : str
        none / xor / raptor_sim.

    redundancy_ratio_config : float
        Redundancy ratio from config.

    group_size : int or None
        XOR group size if applicable.

    decode_overhead : float
        Raptor-sim decode overhead if applicable.

    info : dict
        Extra JSON-friendly encoding information.
    """

    source_packets: torch.Tensor
    encoded_packets: torch.Tensor
    encoded_metas: List[EncodedPacketMeta]

    fec_type: str
    redundancy_ratio_config: float = 0.0
    group_size: Optional[int] = None
    decode_overhead: float = 0.0

    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_source_packets(self) -> int:
        return int(self.source_packets.shape[0])

    @property
    def num_encoded_packets(self) -> int:
        return int(self.encoded_packets.shape[0])

    @property
    def num_parity_packets(self) -> int:
        return int(self.num_encoded_packets - self.num_source_packets)

    @property
    def effective_redundancy_ratio(self) -> float:
        return effective_redundancy_ratio(
            self.num_source_packets,
            self.num_parity_packets,
        )

    def validate(self) -> None:
        """
        Validate internal consistency.
        """
        validate_packet_tensor(self.source_packets, "source_packets")
        validate_packet_tensor(self.encoded_packets, "encoded_packets")

        if len(self.encoded_metas) != self.num_encoded_packets:
            raise ValueError(
                "encoded_metas length mismatch: "
                f"{len(self.encoded_metas)} vs {self.num_encoded_packets}."
            )

        if self.num_encoded_packets < self.num_source_packets:
            raise ValueError(
                "num_encoded_packets cannot be smaller than "
                "num_source_packets."
            )

        for idx, meta in enumerate(self.encoded_metas):
            if int(meta.encoded_id) != idx:
                raise ValueError(
                    f"encoded_metas[{idx}].encoded_id should be {idx}, "
                    f"got {meta.encoded_id}."
                )

    def get_receive_mask(
        self,
        receive_mask: Optional[Any] = None,
        loss_mask: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Resolve receive mask on the same device as encoded_packets.
        """
        return resolve_receive_mask(
            num_encoded_packets=self.num_encoded_packets,
            receive_mask=receive_mask,
            loss_mask=loss_mask,
            device=self.encoded_packets.device,
        )

    def get_loss_mask(
        self,
        receive_mask: Optional[Any] = None,
        loss_mask: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Resolve loss mask on the same device as encoded_packets.
        """
        return resolve_loss_mask(
            num_encoded_packets=self.num_encoded_packets,
            receive_mask=receive_mask,
            loss_mask=loss_mask,
            device=self.encoded_packets.device,
        )

    def as_dict(self, include_metas: bool = False) -> Dict[str, Any]:
        """
        Export JSON-friendly summary.

        Parameters
        ----------
        include_metas : bool
            If True, include every encoded packet meta.
            For large packet counts, this can be verbose.
        """
        result = {
            "fec_type": self.fec_type,
            "num_source_packets": int(self.num_source_packets),
            "num_parity_packets": int(self.num_parity_packets),
            "num_encoded_packets": int(self.num_encoded_packets),
            "redundancy_ratio_config": float(self.redundancy_ratio_config),
            "effective_redundancy_ratio": float(self.effective_redundancy_ratio),
            "group_size": self.group_size,
            "decode_overhead": float(self.decode_overhead),
            "source_shape": tuple(int(x) for x in self.source_packets.shape),
            "encoded_shape": tuple(int(x) for x in self.encoded_packets.shape),
            "source_dtype": str(self.source_packets.dtype),
            "encoded_dtype": str(self.encoded_packets.dtype),
            "info": copy.deepcopy(self.info),
        }

        if include_metas:
            result["encoded_metas"] = [m.as_dict() for m in self.encoded_metas]

        return result


@dataclass
class FECDecodeResult:
    """
    Result of FEC decoding.

    Attributes
    ----------
    recovered_packets : torch.Tensor
        Recovered source packet tensor, shape [K, ...].

    recovered_source_mask : torch.BoolTensor
        Shape [K].
        True means source packet is available after direct receive or FEC recovery.

    direct_received_source_mask : torch.BoolTensor
        Shape [K].
        True means source packet was directly received from the channel.

    fec_recovered_source_mask : torch.BoolTensor
        Shape [K].
        True means source packet was recovered by FEC, not directly received.

    missing_source_mask : torch.BoolTensor
        Shape [K].
        True means source packet is still missing after FEC.

    receive_mask : torch.BoolTensor
        Shape [N].
        True means encoded packet was received.

    loss_mask : torch.BoolTensor
        Shape [N].
        True means encoded packet was lost.

    full_recovery : bool
        True if every source packet is available.

    recovery_ratio : float
        recovered_source_count / num_source_packets.

    info : dict
        Extra JSON-friendly decoding information.
    """

    recovered_packets: torch.Tensor

    recovered_source_mask: torch.Tensor
    direct_received_source_mask: torch.Tensor
    fec_recovered_source_mask: torch.Tensor
    missing_source_mask: torch.Tensor

    receive_mask: torch.Tensor
    loss_mask: torch.Tensor

    fec_type: str
    full_recovery: bool
    recovery_ratio: float

    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_source_packets(self) -> int:
        return int(self.recovered_packets.shape[0])

    @property
    def num_encoded_packets(self) -> int:
        return int(self.receive_mask.numel())

    @property
    def num_recovered_source_packets(self) -> int:
        return count_mask(self.recovered_source_mask)

    @property
    def num_direct_received_source_packets(self) -> int:
        return count_mask(self.direct_received_source_mask)

    @property
    def num_fec_recovered_source_packets(self) -> int:
        return count_mask(self.fec_recovered_source_mask)

    @property
    def num_missing_source_packets(self) -> int:
        return count_mask(self.missing_source_mask)

    @property
    def num_received_encoded_packets(self) -> int:
        return count_mask(self.receive_mask)

    @property
    def num_lost_encoded_packets(self) -> int:
        return count_mask(self.loss_mask)

    def validate(self) -> None:
        """
        Validate masks and recovered packet shape.
        """
        validate_packet_tensor(self.recovered_packets, "recovered_packets")

        k = self.num_source_packets
        n = self.num_encoded_packets

        for name, mask in (
            ("recovered_source_mask", self.recovered_source_mask),
            ("direct_received_source_mask", self.direct_received_source_mask),
            ("fec_recovered_source_mask", self.fec_recovered_source_mask),
            ("missing_source_mask", self.missing_source_mask),
        ):
            _as_bool_tensor(mask, expected_len=k, device=self.recovered_packets.device, name=name)

        _as_bool_tensor(self.receive_mask, expected_len=n, device=self.recovered_packets.device, name="receive_mask")
        _as_bool_tensor(self.loss_mask, expected_len=n, device=self.recovered_packets.device, name="loss_mask")

    def as_dict(self) -> Dict[str, Any]:
        """
        Export JSON-friendly summary.
        """
        return {
            "fec_type": self.fec_type,
            "num_source_packets": int(self.num_source_packets),
            "num_encoded_packets": int(self.num_encoded_packets),
            "num_received_encoded_packets": int(self.num_received_encoded_packets),
            "num_lost_encoded_packets": int(self.num_lost_encoded_packets),
            "num_recovered_source_packets": int(self.num_recovered_source_packets),
            "num_direct_received_source_packets": int(self.num_direct_received_source_packets),
            "num_fec_recovered_source_packets": int(self.num_fec_recovered_source_packets),
            "num_missing_source_packets": int(self.num_missing_source_packets),
            "full_recovery": bool(self.full_recovery),
            "recovery_ratio": float(self.recovery_ratio),
            "recovered_shape": tuple(int(x) for x in self.recovered_packets.shape),
            "recovered_dtype": str(self.recovered_packets.dtype),
            "info": copy.deepcopy(self.info),
        }


class FECBase(ABC):
    """
    Abstract base class for ARCE packet-level FEC modules.

    Concrete subclasses should implement:

        encode(source_packets) -> FECEncodeResult

        decode(encode_result, receive_mask/loss_mask) -> FECDecodeResult

    Important:
        Concrete implementations should keep source-packet order unchanged.
        recovered_packets[i] must correspond to source_packets[i].
    """

    fec_type = DEFAULT_FEC_TYPE

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = normalize_fec_config(cfg or {})

        self.enabled = bool(cfg["enabled"])
        self.fec_type = normalize_fec_type(cfg["type"])

        if not self.enabled:
            self.fec_type = FEC_TYPE_NONE

        self.redundancy_ratio = normalize_redundancy_ratio(
            cfg.get("redundancy_ratio", DEFAULT_REDUNDANCY_RATIO)
        )

        self.group_size = cfg.get("group_size", None)
        if self.group_size is not None:
            self.group_size = normalize_group_size(self.group_size)

        self.decode_overhead = normalize_decode_overhead(
            cfg.get("decode_overhead", DEFAULT_RAPTOR_DECODE_OVERHEAD)
        )

    @abstractmethod
    def encode(self, source_packets: torch.Tensor, **kwargs) -> FECEncodeResult:
        """
        Encode source packets into encoded packets.

        Parameters
        ----------
        source_packets : torch.Tensor
            Shape [K, ...].

        Returns
        -------
        FECEncodeResult
        """
        raise NotImplementedError

    @abstractmethod
    def decode(
        self,
        encode_result: FECEncodeResult,
        receive_mask: Optional[Any] = None,
        loss_mask: Optional[Any] = None,
        **kwargs,
    ) -> FECDecodeResult:
        """
        Decode received encoded packets.

        Parameters
        ----------
        encode_result : FECEncodeResult
            Result returned by encode().

        receive_mask : tensor-like, optional
            Shape [N]. True means encoded packet is received.

        loss_mask : tensor-like, optional
            Shape [N]. True means encoded packet is lost.

        Returns
        -------
        FECDecodeResult
        """
        raise NotImplementedError

    def estimate_encoded_counts(self, num_source_packets: int) -> Tuple[int, int]:
        """
        Estimate encoded packet and parity packet counts.
        """
        return estimate_encoded_packets(
            num_source_packets=num_source_packets,
            fec_type=self.fec_type,
            redundancy_ratio=self.redundancy_ratio,
            group_size=self.group_size,
        )

    def _build_source_metas(self, num_source_packets: int) -> List[EncodedPacketMeta]:
        """
        Build direct-source encoded packet metas.
        """
        k = _as_non_negative_int(num_source_packets, "num_source_packets")

        metas = []
        for i in range(k):
            metas.append(
                EncodedPacketMeta(
                    encoded_id=i,
                    kind="source",
                    source_id=i,
                    group_id=None,
                    source_ids=(i,),
                    note="direct source packet",
                )
            )

        return metas

    def _initial_decode_tensors(
        self,
        encode_result: FECEncodeResult,
        receive_mask: torch.Tensor,
        fill_value: Union[int, float] = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize recovered_packets and direct_received_source_mask.

        Directly received source packets are copied into recovered_packets.
        Missing source packets are filled with fill_value.

        Returns
        -------
        recovered_packets : torch.Tensor
            Shape [K, ...].

        direct_received_source_mask : torch.BoolTensor
            Shape [K].
        """
        encode_result.validate()

        source_packets = encode_result.source_packets
        encoded_packets = encode_result.encoded_packets

        k = encode_result.num_source_packets
        n = encode_result.num_encoded_packets

        receive_mask = _as_bool_tensor(
            receive_mask,
            expected_len=n,
            device=encoded_packets.device,
            name="receive_mask",
        )

        recovered_packets = torch.empty_like(source_packets)
        recovered_packets.fill_(fill_value)

        direct_received_source_mask = torch.zeros(
            k,
            dtype=torch.bool,
            device=encoded_packets.device,
        )

        for meta in encode_result.encoded_metas:
            if not meta.is_source:
                continue

            if meta.source_id is None:
                continue

            src_id = int(meta.source_id)
            enc_id = int(meta.encoded_id)

            if src_id < 0 or src_id >= k:
                raise ValueError(
                    f"Invalid source_id={src_id} for K={k}."
                )

            if bool(receive_mask[enc_id].item()):
                recovered_packets[src_id] = encoded_packets[enc_id]
                direct_received_source_mask[src_id] = True

        return recovered_packets, direct_received_source_mask

    def _finalize_decode_result(
        self,
        recovered_packets: torch.Tensor,
        direct_received_source_mask: torch.Tensor,
        fec_recovered_source_mask: torch.Tensor,
        receive_mask: torch.Tensor,
        info: Optional[Dict[str, Any]] = None,
    ) -> FECDecodeResult:
        """
        Build FECDecodeResult from masks and recovered packets.
        """
        recovered_packets = validate_packet_tensor(
            recovered_packets,
            "recovered_packets",
        )

        k = int(recovered_packets.shape[0])
        n = int(receive_mask.numel())

        direct_received_source_mask = _as_bool_tensor(
            direct_received_source_mask,
            expected_len=k,
            device=recovered_packets.device,
            name="direct_received_source_mask",
        )

        fec_recovered_source_mask = _as_bool_tensor(
            fec_recovered_source_mask,
            expected_len=k,
            device=recovered_packets.device,
            name="fec_recovered_source_mask",
        )

        receive_mask = _as_bool_tensor(
            receive_mask,
            expected_len=n,
            device=recovered_packets.device,
            name="receive_mask",
        )

        recovered_source_mask = direct_received_source_mask | fec_recovered_source_mask
        missing_source_mask = ~recovered_source_mask
        loss_mask = ~receive_mask

        num_recovered = int(recovered_source_mask.sum().item())
        recovery_ratio = float(num_recovered / k) if k > 0 else 1.0
        full_recovery = bool(num_recovered == k)

        result = FECDecodeResult(
            recovered_packets=recovered_packets,
            recovered_source_mask=recovered_source_mask,
            direct_received_source_mask=direct_received_source_mask,
            fec_recovered_source_mask=fec_recovered_source_mask,
            missing_source_mask=missing_source_mask,
            receive_mask=receive_mask,
            loss_mask=loss_mask,
            fec_type=self.fec_type,
            full_recovery=full_recovery,
            recovery_ratio=recovery_ratio,
            info=info or {},
        )

        result.validate()
        return result

    def get_config(self) -> Dict[str, Any]:
        """
        Return JSON-friendly FEC config.
        """
        return {
            "enabled": bool(self.enabled),
            "fec_type": self.fec_type,
            "redundancy_ratio": float(self.redundancy_ratio),
            "group_size": self.group_size,
            "decode_overhead": float(self.decode_overhead),
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"enabled={self.enabled}, "
            f"fec_type={self.fec_type}, "
            f"redundancy_ratio={self.redundancy_ratio}, "
            f"group_size={self.group_size}, "
            f"decode_overhead={self.decode_overhead})"
        )