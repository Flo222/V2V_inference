"""
Communication size estimator for ARCE feature transmission.

This module estimates communication overhead for intermediate BEV features.

It supports:
    1. raw feature size estimation;
    2. quantized feature size estimation;
    3. packet-level size estimation;
    4. FEC / redundancy overhead estimation;
    5. received-byte estimation from receive/loss masks;
    6. JSON-serializable size summary for logging.

Important:
    This module only estimates sizes.
    It does NOT:
        - split feature tensors into packets;
        - quantize tensors;
        - sample packet loss;
        - perform FEC encoding / decoding;
        - reconstruct missing patches.

Those operations are handled by:
    packetizer.py
    opencood.communication.transport.quantization.*
    opencood.communication.channel.*
    opencood.communication.transport.fec.*
    opencood.communication.transport.recovery.*
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


QUANT_MODE_FP32 = "fp32"
QUANT_MODE_FLOAT32 = "float32"
QUANT_MODE_FP16 = "fp16"
QUANT_MODE_FLOAT16 = "float16"
QUANT_MODE_INT8 = "int8"
QUANT_MODE_UINT8 = "uint8"
QUANT_MODE_INT4 = "int4"

FEC_TYPE_NONE = "none"
FEC_TYPE_XOR = "xor"
FEC_TYPE_RAPTOR_SIM = "raptor_sim"
FEC_TYPE_RAPTOR = "raptor"

DEFAULT_RAW_BITS = 32
DEFAULT_QUANT_MODE = QUANT_MODE_FP32

QUANT_MODE_TO_BITS = {
    QUANT_MODE_FP32: 32,
    QUANT_MODE_FLOAT32: 32,
    QUANT_MODE_FP16: 16,
    QUANT_MODE_FLOAT16: 16,
    QUANT_MODE_INT8: 8,
    QUANT_MODE_UINT8: 8,
    QUANT_MODE_INT4: 4,
}

CANONICAL_QUANT_MODE = {
    QUANT_MODE_FLOAT32: QUANT_MODE_FP32,
    QUANT_MODE_FLOAT16: QUANT_MODE_FP16,
    QUANT_MODE_UINT8: QUANT_MODE_INT8,
}

VALID_FEC_TYPES = (
    FEC_TYPE_NONE,
    FEC_TYPE_XOR,
    FEC_TYPE_RAPTOR_SIM,
    FEC_TYPE_RAPTOR,
)


def _is_torch_tensor(x: Any) -> bool:
    """
    Return True if x is a torch.Tensor.
    """
    return torch is not None and torch.is_tensor(x)


def _as_float(value: Any, name: str) -> float:
    """
    Convert value to float.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} should be convertible to float, got {value}.")


def _as_non_negative_float(value: Any, name: str) -> float:
    """
    Convert value to non-negative float.
    """
    value = _as_float(value, name)

    if value < 0.0:
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


def normalize_quant_mode(mode: Optional[str]) -> str:
    """
    Normalize quantization mode.

    Parameters
    ----------
    mode : str or None
        Quantization mode. Supported:
            fp32 / float32
            fp16 / float16
            int8 / uint8
            int4

    Returns
    -------
    str
        Canonical mode:
            fp32 / fp16 / int8 / int4
    """
    if mode is None:
        return DEFAULT_QUANT_MODE

    mode = str(mode).strip().lower()

    if mode not in QUANT_MODE_TO_BITS:
        raise ValueError(
            f"Unsupported quantization mode: {mode}. "
            f"Supported modes: {tuple(QUANT_MODE_TO_BITS.keys())}."
        )

    return CANONICAL_QUANT_MODE.get(mode, mode)


def quant_mode_to_bits(mode: Optional[str]) -> int:
    """
    Return bits per value for a quantization mode.
    """
    mode = normalize_quant_mode(mode)
    return int(QUANT_MODE_TO_BITS[mode])


def normalize_fec_type(fec_type: Optional[str]) -> str:
    """
    Normalize FEC type.
    """
    if fec_type is None:
        return FEC_TYPE_NONE

    fec_type = str(fec_type).strip().lower()

    if fec_type not in VALID_FEC_TYPES:
        raise ValueError(
            f"Unsupported fec_type: {fec_type}. "
            f"Supported FEC types: {VALID_FEC_TYPES}."
        )

    if fec_type == FEC_TYPE_RAPTOR:
        return FEC_TYPE_RAPTOR_SIM

    return fec_type


def bits_to_bytes(num_bits: Any, ceil_to_byte: bool = True) -> float:
    """
    Convert bits to bytes.

    Parameters
    ----------
    num_bits : int or float
        Number of bits.

    ceil_to_byte : bool
        If True, return ceil(num_bits / 8) as an integer byte count.
        If False, return exact num_bits / 8 as float.

    Returns
    -------
    float
        Number of bytes.
    """
    num_bits = _as_non_negative_float(num_bits, "num_bits")

    if ceil_to_byte:
        return float(int(math.ceil(num_bits / 8.0)))

    return float(num_bits / 8.0)


def bytes_to_megabytes(num_bytes: Any, decimal: bool = True) -> float:
    """
    Convert bytes to MB.

    Parameters
    ----------
    num_bytes : int or float
        Number of bytes.

    decimal : bool
        If True, use 1 MB = 1,000,000 bytes.
        If False, use 1 MiB = 1,048,576 bytes.
    """
    num_bytes = _as_non_negative_float(num_bytes, "num_bytes")
    denom = 1_000_000.0 if decimal else 1024.0 * 1024.0
    return float(num_bytes / denom)


def normalize_feature_shape(
    feature_or_shape: Union[Any, Sequence[int]]
) -> Tuple[int, int, int]:
    """
    Normalize feature tensor or shape to [C, H, W].

    Supports:
        torch.Tensor [C, H, W]
        torch.Tensor [N, C, H, W]
        shape [C, H, W]
        shape [N, C, H, W]

    Returns
    -------
    tuple
        (C, H, W)
    """
    if _is_torch_tensor(feature_or_shape):
        shape = tuple(int(x) for x in feature_or_shape.shape)
    elif isinstance(feature_or_shape, (list, tuple)):
        shape = tuple(int(x) for x in feature_or_shape)
    else:
        raise TypeError(
            "feature_or_shape should be a torch.Tensor, list, or tuple, "
            f"got {type(feature_or_shape)}."
        )

    if len(shape) == 3:
        c, h, w = shape
    elif len(shape) == 4:
        _, c, h, w = shape
    else:
        raise ValueError(
            "Feature shape should be [C, H, W] or [N, C, H, W], "
            f"got {shape}."
        )

    if c <= 0 or h <= 0 or w <= 0:
        raise ValueError(f"Feature shape should be positive, got {(c, h, w)}.")

    return int(c), int(h), int(w)


def estimate_num_elements(feature_or_shape: Union[Any, Sequence[int]]) -> int:
    """
    Estimate number of feature elements for one feature [C, H, W].
    """
    c, h, w = normalize_feature_shape(feature_or_shape)
    return int(c * h * w)


def estimate_feature_bits(
    feature_or_shape: Union[Any, Sequence[int]],
    bits_per_value: Any = DEFAULT_RAW_BITS,
) -> int:
    """
    Estimate feature size in bits.

    Parameters
    ----------
    feature_or_shape : tensor or shape
        Feature tensor or shape.

    bits_per_value : int
        Bits per feature value.

    Returns
    -------
    int
        Number of bits.
    """
    bits_per_value = _as_positive_int(bits_per_value, "bits_per_value")
    return int(estimate_num_elements(feature_or_shape) * bits_per_value)


def estimate_feature_bytes(
    feature_or_shape: Union[Any, Sequence[int]],
    bits_per_value: Any = DEFAULT_RAW_BITS,
    ceil_to_byte: bool = True,
) -> float:
    """
    Estimate feature size in bytes.
    """
    bits = estimate_feature_bits(feature_or_shape, bits_per_value)
    return bits_to_bytes(bits, ceil_to_byte=ceil_to_byte)


def estimate_feature_bytes_by_quant_mode(
    feature_or_shape: Union[Any, Sequence[int]],
    quant_mode: Optional[str] = DEFAULT_QUANT_MODE,
    ceil_to_byte: bool = True,
) -> float:
    """
    Estimate feature bytes under a quantization mode.
    """
    bits = quant_mode_to_bits(quant_mode)
    return estimate_feature_bytes(
        feature_or_shape,
        bits_per_value=bits,
        ceil_to_byte=ceil_to_byte,
    )


def compression_ratio_from_bits(
    quant_bits: Any,
    raw_bits: Any = DEFAULT_RAW_BITS,
) -> float:
    """
    Compute compression ratio relative to raw_bits.

    Example:
        raw_bits=32, quant_bits=8 -> 0.25
    """
    quant_bits = _as_positive_int(quant_bits, "quant_bits")
    raw_bits = _as_positive_int(raw_bits, "raw_bits")
    return float(quant_bits / raw_bits)


def compression_ratio_from_quant_mode(
    quant_mode: Optional[str],
    raw_bits: Any = DEFAULT_RAW_BITS,
) -> float:
    """
    Compute compression ratio for a quantization mode.
    """
    return compression_ratio_from_bits(
        quant_bits=quant_mode_to_bits(quant_mode),
        raw_bits=raw_bits,
    )


def estimate_packet_bits_from_meta(
    packet_meta: Any,
    channels: int,
    bits_per_value: Any,
    use_padded_size: bool = False,
) -> int:
    """
    Estimate one packet size in bits from FeaturePacketMeta-like object.

    The meta object is expected to have:
        valid_h, valid_w, padded_h, padded_w

    If use_padded_size=True:
        count C * padded_h * padded_w values.
    Else:
        count C * valid_h * valid_w values.
    """
    channels = _as_positive_int(channels, "channels")
    bits_per_value = _as_positive_int(bits_per_value, "bits_per_value")

    if use_padded_size:
        h = _as_positive_int(getattr(packet_meta, "padded_h"), "padded_h")
        w = _as_positive_int(getattr(packet_meta, "padded_w"), "padded_w")
    else:
        h = _as_positive_int(getattr(packet_meta, "valid_h"), "valid_h")
        w = _as_positive_int(getattr(packet_meta, "valid_w"), "valid_w")

    return int(channels * h * w * bits_per_value)


def estimate_packet_bytes_from_meta(
    packet_meta: Any,
    channels: int,
    bits_per_value: Any,
    use_padded_size: bool = False,
    ceil_to_byte: bool = True,
) -> float:
    """
    Estimate one packet size in bytes from FeaturePacketMeta-like object.
    """
    bits = estimate_packet_bits_from_meta(
        packet_meta=packet_meta,
        channels=channels,
        bits_per_value=bits_per_value,
        use_padded_size=use_padded_size,
    )
    return bits_to_bytes(bits, ceil_to_byte=ceil_to_byte)


def estimate_packets_bytes_from_metas(
    metas: Iterable[Any],
    channels: int,
    bits_per_value: Any,
    use_padded_size: bool = False,
    ceil_each_packet: bool = True,
) -> Tuple[List[float], float]:
    """
    Estimate packet bytes for a list of packet metas.

    Parameters
    ----------
    metas : iterable
        Packet metadata list.

    channels : int
        Feature channels.

    bits_per_value : int
        Bits per feature value.

    use_padded_size : bool
        If True, count padded packet tensor size.
        If False, count only valid original feature area.

    ceil_each_packet : bool
        If True, each packet is byte-aligned separately.
        If False, total bits are summed first and then converted to bytes.

    Returns
    -------
    packet_bytes : list
        Bytes per packet.

    total_bytes : float
        Total packet bytes.
    """
    metas = list(metas)
    packet_bits = [
        estimate_packet_bits_from_meta(
            packet_meta=m,
            channels=channels,
            bits_per_value=bits_per_value,
            use_padded_size=use_padded_size,
        )
        for m in metas
    ]

    if ceil_each_packet:
        packet_bytes = [bits_to_bytes(bits, ceil_to_byte=True) for bits in packet_bits]
        total_bytes = float(sum(packet_bytes))
        return packet_bytes, total_bytes

    total_bits = int(sum(packet_bits))
    total_bytes = bits_to_bytes(total_bits, ceil_to_byte=True)

    packet_bytes = [
        bits_to_bytes(bits, ceil_to_byte=False)
        for bits in packet_bits
    ]

    return packet_bytes, total_bytes


def estimate_redundancy_packets(
    num_source_packets: Any,
    fec_type: Optional[str] = FEC_TYPE_NONE,
    redundancy_ratio: Any = 0.0,
    group_size: Optional[int] = None,
) -> int:
    """
    Estimate number of parity / redundant packets.

    Rules:
        none:
            0 parity packets.

        xor:
            If group_size is provided:
                parity = ceil(K / group_size)
            Else:
                parity = ceil(K * redundancy_ratio)

        raptor_sim:
            parity = ceil(K * redundancy_ratio)

    Parameters
    ----------
    num_source_packets : int
        Number of source packets K.

    fec_type : str
        none / xor / raptor_sim.

    redundancy_ratio : float
        Redundancy ratio rho.

    group_size : int, optional
        XOR group size. One parity packet per group.

    Returns
    -------
    int
        Number of parity / redundant packets.
    """
    k = _as_non_negative_int(num_source_packets, "num_source_packets")
    fec_type = normalize_fec_type(fec_type)
    redundancy_ratio = _as_non_negative_float(
        redundancy_ratio,
        "redundancy_ratio",
    )

    if k == 0:
        return 0

    if fec_type == FEC_TYPE_NONE:
        return 0

    if fec_type == FEC_TYPE_XOR and group_size is not None:
        group_size = _as_positive_int(group_size, "group_size")
        return int(math.ceil(k / group_size))

    return int(math.ceil(k * redundancy_ratio))


def estimate_encoded_packets(
    num_source_packets: Any,
    fec_type: Optional[str] = FEC_TYPE_NONE,
    redundancy_ratio: Any = 0.0,
    group_size: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Estimate source + parity packet count.

    Returns
    -------
    tuple
        (num_encoded_packets, num_parity_packets)
    """
    k = _as_non_negative_int(num_source_packets, "num_source_packets")

    parity = estimate_redundancy_packets(
        num_source_packets=k,
        fec_type=fec_type,
        redundancy_ratio=redundancy_ratio,
        group_size=group_size,
    )

    return int(k + parity), int(parity)


def estimate_average_packet_bytes(
    total_source_bytes: Any,
    num_source_packets: Any,
) -> float:
    """
    Estimate average source packet size in bytes.
    """
    total_source_bytes = _as_non_negative_float(
        total_source_bytes,
        "total_source_bytes",
    )
    num_source_packets = _as_positive_int(
        num_source_packets,
        "num_source_packets",
    )

    return float(total_source_bytes / num_source_packets)


def estimate_parity_bytes(
    source_packet_bytes: Optional[Sequence[float]] = None,
    num_parity_packets: Any = 0,
    average_packet_bytes: Optional[float] = None,
) -> float:
    """
    Estimate parity bytes.

    For XOR:
        parity packet size is approximately the size of one source packet.
        If source_packet_bytes is given, use their average.

    For Raptor-sim:
        same average packet size assumption is used.

    Parameters
    ----------
    source_packet_bytes : sequence, optional
        Per-source-packet byte sizes.

    num_parity_packets : int
        Number of parity / redundant packets.

    average_packet_bytes : float, optional
        Override average packet size.

    Returns
    -------
    float
        Estimated parity bytes.
    """
    num_parity_packets = _as_non_negative_int(
        num_parity_packets,
        "num_parity_packets",
    )

    if num_parity_packets == 0:
        return 0.0

    if average_packet_bytes is not None:
        avg = _as_non_negative_float(
            average_packet_bytes,
            "average_packet_bytes",
        )
    elif source_packet_bytes is not None:
        source_packet_bytes = list(source_packet_bytes)
        if len(source_packet_bytes) == 0:
            avg = 0.0
        else:
            avg = float(sum(source_packet_bytes) / len(source_packet_bytes))
    else:
        raise ValueError(
            "Either source_packet_bytes or average_packet_bytes must be provided "
            "when estimating parity bytes."
        )

    return float(num_parity_packets * avg)


def _mask_to_num_true(mask: Any) -> int:
    """
    Count True values in a tensor/list/numpy array-like mask.
    """
    if mask is None:
        raise ValueError("mask is None.")

    if _is_torch_tensor(mask):
        return int(mask.to(dtype=torch.bool).sum().item())

    try:
        import numpy as np
        if isinstance(mask, np.ndarray):
            return int(mask.astype(bool).sum())
    except Exception:
        pass

    return int(sum(bool(x) for x in mask))


def estimate_received_packets(
    num_encoded_packets: Any,
    receive_mask: Optional[Any] = None,
    loss_mask: Optional[Any] = None,
    num_received_packets: Optional[int] = None,
) -> int:
    """
    Estimate number of received packets.

    Priority:
        1. num_received_packets
        2. receive_mask
        3. loss_mask
        4. assume all encoded packets are received
    """
    n = _as_non_negative_int(num_encoded_packets, "num_encoded_packets")

    if num_received_packets is not None:
        num_received_packets = _as_non_negative_int(
            num_received_packets,
            "num_received_packets",
        )
        if num_received_packets > n:
            raise ValueError(
                f"num_received_packets={num_received_packets} "
                f"cannot exceed num_encoded_packets={n}."
            )
        return int(num_received_packets)

    if receive_mask is not None:
        received = _mask_to_num_true(receive_mask)
        if received > n:
            raise ValueError(
                f"receive_mask has {received} True values, "
                f"but num_encoded_packets={n}."
            )
        return int(received)

    if loss_mask is not None:
        lost = _mask_to_num_true(loss_mask)
        if lost > n:
            raise ValueError(
                f"loss_mask has {lost} True values, "
                f"but num_encoded_packets={n}."
            )
        return int(n - lost)

    return int(n)


@dataclass
class SizeEstimate:
    """
    JSON-serializable communication size estimate.

    All byte fields are floats for flexibility, although most of them
    are integer-valued when byte alignment is used.
    """

    raw_bits_per_value: int
    quant_bits_per_value: int
    quant_mode: str

    original_shape: Tuple[int, int, int]
    num_elements: int

    raw_bits: int
    raw_bytes: float

    compressed_bits: int
    compressed_bytes: float
    compression_ratio: float

    fec_type: str
    redundancy_ratio_config: float
    group_size: Optional[int]

    num_source_packets: int
    num_parity_packets: int
    num_encoded_packets: int
    effective_redundancy_ratio: float

    source_packet_bytes_avg: float
    parity_bytes: float
    transmitted_bytes: float

    num_received_packets: int
    num_lost_packets: int
    received_bytes: float

    transmitted_mb: float
    received_mb: float

    def as_dict(self) -> Dict[str, Any]:
        """
        Export as JSON-serializable dict.
        """
        result = asdict(self)
        result["original_shape"] = tuple(int(x) for x in self.original_shape)
        return result


class FeatureSizeEstimator:
    """
    Size estimator for ARCE feature transmission.

    Recommended usage:

        estimator = FeatureSizeEstimator(arce_cfg)

        size_info = estimator.estimate(
            feature_or_shape=feature,
            quant_mode="int8",
            num_source_packets=100,
            fec_type="xor",
            group_size=4,
        )

    It can also be initialized with YAML-style config:

        arce:
          quantization:
            mode: int8
          fec:
            type: xor
            group_size: 4
            redundancy_ratio: 0.25
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}

        self.full_cfg = cfg
        self.quant_cfg = self._extract_quant_cfg(cfg)
        self.fec_cfg = self._extract_fec_cfg(cfg)

        self.raw_bits = int(
            self.quant_cfg.get(
                "raw_bits",
                self.quant_cfg.get("raw_bits_per_value", DEFAULT_RAW_BITS),
            )
        )

        self.default_quant_mode = normalize_quant_mode(
            self.quant_cfg.get("mode", DEFAULT_QUANT_MODE)
        )

        self.default_fec_type = normalize_fec_type(
            self.fec_cfg.get("type", FEC_TYPE_NONE)
        )

        self.default_redundancy_ratio = _as_non_negative_float(
            self.fec_cfg.get("redundancy_ratio", 0.0),
            "redundancy_ratio",
        )

        self.default_group_size = self.fec_cfg.get("group_size", None)
        if self.default_group_size is not None:
            self.default_group_size = _as_positive_int(
                self.default_group_size,
                "group_size",
            )

        self.ceil_to_byte = bool(
            cfg.get(
                "ceil_to_byte",
                self.fec_cfg.get("ceil_to_byte", True),
            )
        )

        self.ceil_each_packet = bool(
            cfg.get(
                "ceil_each_packet",
                self.fec_cfg.get("ceil_each_packet", True),
            )
        )

        self.use_padded_packet_size = bool(
            cfg.get(
                "use_padded_packet_size",
                self.fec_cfg.get("use_padded_packet_size", False),
            )
        )

    @staticmethod
    def _extract_quant_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accept full ARCE config or direct quantization config.
        """
        if "quantization" in cfg and isinstance(cfg["quantization"], dict):
            return cfg["quantization"]
        return cfg

    @staticmethod
    def _extract_fec_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accept full ARCE config or direct fec config.
        """
        if "fec" in cfg and isinstance(cfg["fec"], dict):
            return cfg["fec"]
        return cfg

    def estimate(
        self,
        feature_or_shape: Union[Any, Sequence[int]],
        quant_mode: Optional[str] = None,
        num_source_packets: Optional[int] = None,
        packet_metas: Optional[Iterable[Any]] = None,
        fec_type: Optional[str] = None,
        redundancy_ratio: Optional[float] = None,
        group_size: Optional[int] = None,
        receive_mask: Optional[Any] = None,
        loss_mask: Optional[Any] = None,
        num_received_packets: Optional[int] = None,
        use_padded_packet_size: Optional[bool] = None,
    ) -> SizeEstimate:
        """
        Estimate communication size for one transmitted feature message.

        Parameters
        ----------
        feature_or_shape : tensor or shape
            Feature tensor or shape [C, H, W] / [N, C, H, W].

        quant_mode : str, optional
            fp32 / fp16 / int8 / int4.

        num_source_packets : int, optional
            Source packet count. Required if packet_metas is None.

        packet_metas : iterable, optional
            FeaturePacketMeta-like list from packetizer.
            If provided, packet sizes are estimated from valid packet areas.

        fec_type : str, optional
            none / xor / raptor_sim.

        redundancy_ratio : float, optional
            Redundancy ratio.

        group_size : int, optional
            XOR group size.

        receive_mask / loss_mask / num_received_packets : optional
            Used to estimate received bytes.

        use_padded_packet_size : bool, optional
            If True, count padded packet tensor size.
            If False, count only valid original area.

        Returns
        -------
        SizeEstimate
            Structured size estimate.
        """
        c, h, w = normalize_feature_shape(feature_or_shape)
        original_shape = (c, h, w)
        num_elements = int(c * h * w)

        raw_bits_per_value = _as_positive_int(self.raw_bits, "raw_bits")
        quant_mode = normalize_quant_mode(quant_mode or self.default_quant_mode)
        quant_bits_per_value = quant_mode_to_bits(quant_mode)

        raw_bits = int(num_elements * raw_bits_per_value)
        raw_bytes = bits_to_bytes(raw_bits, ceil_to_byte=self.ceil_to_byte)

        compressed_bits = int(num_elements * quant_bits_per_value)
        compressed_bytes_by_feature = bits_to_bytes(
            compressed_bits,
            ceil_to_byte=self.ceil_to_byte,
        )

        compression_ratio = compression_ratio_from_bits(
            quant_bits=quant_bits_per_value,
            raw_bits=raw_bits_per_value,
        )

        fec_type = normalize_fec_type(fec_type or self.default_fec_type)

        if redundancy_ratio is None:
            redundancy_ratio = self.default_redundancy_ratio
        redundancy_ratio = _as_non_negative_float(
            redundancy_ratio,
            "redundancy_ratio",
        )

        if group_size is None:
            group_size = self.default_group_size
        if group_size is not None:
            group_size = _as_positive_int(group_size, "group_size")

        use_padded_packet_size = (
            self.use_padded_packet_size
            if use_padded_packet_size is None
            else bool(use_padded_packet_size)
        )

        if packet_metas is not None:
            packet_metas = list(packet_metas)
            num_source_packets = len(packet_metas)

            source_packet_bytes, compressed_bytes = estimate_packets_bytes_from_metas(
                metas=packet_metas,
                channels=c,
                bits_per_value=quant_bits_per_value,
                use_padded_size=use_padded_packet_size,
                ceil_each_packet=self.ceil_each_packet,
            )
        else:
            if num_source_packets is None:
                num_source_packets = 1

            num_source_packets = _as_positive_int(
                num_source_packets,
                "num_source_packets",
            )

            compressed_bytes = compressed_bytes_by_feature
            avg_source_packet_bytes = compressed_bytes / num_source_packets
            source_packet_bytes = [
                float(avg_source_packet_bytes)
                for _ in range(num_source_packets)
            ]

        num_encoded_packets, num_parity_packets = estimate_encoded_packets(
            num_source_packets=num_source_packets,
            fec_type=fec_type,
            redundancy_ratio=redundancy_ratio,
            group_size=group_size,
        )

        effective_redundancy_ratio = (
            float(num_parity_packets / num_source_packets)
            if num_source_packets > 0
            else 0.0
        )

        source_packet_bytes_avg = estimate_average_packet_bytes(
            total_source_bytes=compressed_bytes,
            num_source_packets=num_source_packets,
        )

        parity_bytes = estimate_parity_bytes(
            source_packet_bytes=source_packet_bytes,
            num_parity_packets=num_parity_packets,
            average_packet_bytes=source_packet_bytes_avg,
        )

        transmitted_bytes = float(compressed_bytes + parity_bytes)

        num_received_packets = estimate_received_packets(
            num_encoded_packets=num_encoded_packets,
            receive_mask=receive_mask,
            loss_mask=loss_mask,
            num_received_packets=num_received_packets,
        )

        num_lost_packets = int(num_encoded_packets - num_received_packets)

        if num_encoded_packets > 0:
            received_bytes = transmitted_bytes * (
                float(num_received_packets) / float(num_encoded_packets)
            )
        else:
            received_bytes = 0.0

        return SizeEstimate(
            raw_bits_per_value=int(raw_bits_per_value),
            quant_bits_per_value=int(quant_bits_per_value),
            quant_mode=quant_mode,
            original_shape=original_shape,
            num_elements=int(num_elements),
            raw_bits=int(raw_bits),
            raw_bytes=float(raw_bytes),
            compressed_bits=int(compressed_bits),
            compressed_bytes=float(compressed_bytes),
            compression_ratio=float(compression_ratio),
            fec_type=fec_type,
            redundancy_ratio_config=float(redundancy_ratio),
            group_size=group_size,
            num_source_packets=int(num_source_packets),
            num_parity_packets=int(num_parity_packets),
            num_encoded_packets=int(num_encoded_packets),
            effective_redundancy_ratio=float(effective_redundancy_ratio),
            source_packet_bytes_avg=float(source_packet_bytes_avg),
            parity_bytes=float(parity_bytes),
            transmitted_bytes=float(transmitted_bytes),
            num_received_packets=int(num_received_packets),
            num_lost_packets=int(num_lost_packets),
            received_bytes=float(received_bytes),
            transmitted_mb=bytes_to_megabytes(transmitted_bytes),
            received_mb=bytes_to_megabytes(received_bytes),
        )

    def estimate_from_packetization_result(
        self,
        packetization_result: Any,
        quant_mode: Optional[str] = None,
        fec_type: Optional[str] = None,
        redundancy_ratio: Optional[float] = None,
        group_size: Optional[int] = None,
        receive_mask: Optional[Any] = None,
        loss_mask: Optional[Any] = None,
        num_received_packets: Optional[int] = None,
    ) -> SizeEstimate:
        """
        Estimate size from PacketizationResult.

        Expected fields:
            packetization_result.original_shape
            packetization_result.metas
        """
        return self.estimate(
            feature_or_shape=packetization_result.original_shape,
            quant_mode=quant_mode,
            packet_metas=packetization_result.metas,
            fec_type=fec_type,
            redundancy_ratio=redundancy_ratio,
            group_size=group_size,
            receive_mask=receive_mask,
            loss_mask=loss_mask,
            num_received_packets=num_received_packets,
        )

    def get_config(self) -> Dict[str, Any]:
        """
        Export estimator config.
        """
        return {
            "raw_bits": int(self.raw_bits),
            "default_quant_mode": self.default_quant_mode,
            "default_fec_type": self.default_fec_type,
            "default_redundancy_ratio": float(self.default_redundancy_ratio),
            "default_group_size": self.default_group_size,
            "ceil_to_byte": bool(self.ceil_to_byte),
            "ceil_each_packet": bool(self.ceil_each_packet),
            "use_padded_packet_size": bool(self.use_padded_packet_size),
        }

    def __repr__(self) -> str:
        return (
            "FeatureSizeEstimator("
            f"raw_bits={self.raw_bits}, "
            f"default_quant_mode={self.default_quant_mode}, "
            f"default_fec_type={self.default_fec_type}, "
            f"default_redundancy_ratio={self.default_redundancy_ratio}, "
            f"default_group_size={self.default_group_size})"
        )