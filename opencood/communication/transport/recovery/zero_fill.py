"""
Zero-fill recovery for ARCE communication simulation.

This module provides the simplest packet recovery strategy:

    missing source packet -> fill with 0 or a configured constant

It can operate on:
    1. packet tensors:
        packets [K, ...]

    2. full feature tensors with packet metadata:
        feature [C, H, W] + packet metas

Typical usage after FEC decoding:

    decode_result = fec.decode(...)

    zero_result = zero_fill_packets(
        packets=decode_result.recovered_packets,
        missing_mask=decode_result.missing_source_mask,
        fill_value=0.0,
    )

    recovered_packets = zero_result.packets

Important:
    zero-fill does not recover semantic information.
    It only makes the tensor complete so that unpacketize() and V2X-ViT
    fusion can continue without shape errors.

Mask convention:
    missing_mask[i] == True
        source packet i is missing and should be zero-filled.

    available_mask[i] == True
        source packet i is available and should be kept.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from opencood.communication.transport.recovery import (
    RECOVERY_METHOD_ZERO_FILL,
    build_recovery_count_dict,
    available_mask_to_missing_mask,
    missing_mask_to_available_mask,
)


def _require_tensor(x: Any, name: str = "tensor") -> torch.Tensor:
    """
    Validate torch.Tensor input.
    """
    if not torch.is_tensor(x):
        raise TypeError(f"{name} should be a torch.Tensor, got {type(x)}.")
    return x


def _as_bool_tensor(
    mask: Any,
    expected_len: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    name: str = "mask",
) -> torch.Tensor:
    """
    Convert mask to flattened torch.BoolTensor.

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
        Shape [expected_len] if expected_len is provided.
    """
    if mask is None:
        raise ValueError(f"{name} is None.")

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


def _resolve_missing_mask(
    num_packets: int,
    missing_mask: Optional[Any] = None,
    available_mask: Optional[Any] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Resolve missing mask from missing_mask or available_mask.

    Priority:
        1. missing_mask
        2. available_mask
        3. all packets are available

    Returns
    -------
    torch.BoolTensor
        Shape [num_packets].
        True means packet is missing.
    """
    num_packets = int(num_packets)

    if num_packets < 0:
        raise ValueError(f"num_packets should be non-negative, got {num_packets}.")

    if missing_mask is not None:
        return _as_bool_tensor(
            missing_mask,
            expected_len=num_packets,
            device=device,
            name="missing_mask",
        )

    if available_mask is not None:
        available_mask = _as_bool_tensor(
            available_mask,
            expected_len=num_packets,
            device=device,
            name="available_mask",
        )
        return available_mask_to_missing_mask(available_mask)

    return torch.zeros(num_packets, dtype=torch.bool, device=device)


def _count_true(mask: torch.Tensor) -> int:
    """
    Count True values in mask.
    """
    mask = _as_bool_tensor(mask, name="mask")
    return int(mask.sum().item())


def _safe_fill_value_for_tensor(fill_value: Union[int, float], tensor: torch.Tensor):
    """
    Return fill value compatible with tensor dtype.

    torch assignment usually handles scalar casting, but this function keeps
    intent explicit and avoids surprises with integer tensors.
    """
    if tensor.dtype in (
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
    ):
        return int(fill_value)

    return float(fill_value)


@dataclass
class ZeroFillResult:
    """
    Result of zero-fill recovery.

    Attributes
    ----------
    recovered : torch.Tensor
        Recovered tensor. It can be:
            packet tensor [K, ...]
            feature tensor [C, H, W]

    filled_mask : torch.BoolTensor
        Packet-level mask. Shape [K].
        True means this packet was filled by zero-fill.

    available_mask : torch.BoolTensor
        Packet-level mask. Shape [K].
        True means this packet is available after zero-fill.

    missing_mask_before : torch.BoolTensor
        Packet-level mask before zero-fill. Shape [K].

    still_missing_mask : torch.BoolTensor
        Packet-level mask after zero-fill. Shape [K].
        For zero-fill, this is normally all False because every missing packet
        is filled with a constant.

    info : dict
        JSON-friendly statistics.
    """

    recovered: torch.Tensor
    filled_mask: torch.Tensor
    available_mask: torch.Tensor
    missing_mask_before: torch.Tensor
    still_missing_mask: torch.Tensor
    info: Dict[str, Any]

    @property
    def num_packets(self) -> int:
        return int(self.filled_mask.numel())

    @property
    def num_zero_filled_packets(self) -> int:
        return _count_true(self.filled_mask)

    @property
    def recovery_ratio(self) -> float:
        if self.num_packets <= 0:
            return 1.0
        return float(_count_true(self.available_mask) / self.num_packets)

    def as_dict(self) -> Dict[str, Any]:
        """
        Return JSON-friendly summary.

        Tensor masks are summarized by counts rather than serialized fully.
        """
        result = copy.deepcopy(self.info)
        result.update(
            {
                "num_packets": int(self.num_packets),
                "num_zero_filled_packets": int(self.num_zero_filled_packets),
                "recovery_ratio": float(self.recovery_ratio),
                "recovered_shape": tuple(int(x) for x in self.recovered.shape),
                "recovered_dtype": str(self.recovered.dtype),
            }
        )
        return result


def zero_fill_packets(
    packets: torch.Tensor,
    missing_mask: Optional[Any] = None,
    available_mask: Optional[Any] = None,
    fill_value: Union[int, float] = 0.0,
    clone: bool = True,
) -> ZeroFillResult:
    """
    Zero-fill missing packet tensors.

    Parameters
    ----------
    packets : torch.Tensor
        Packet tensor with shape [K, ...].

    missing_mask : tensor-like, optional
        Shape [K]. True means packet is missing and should be filled.

    available_mask : tensor-like, optional
        Shape [K]. True means packet is available.
        Used only when missing_mask is not provided.

    fill_value : int or float
        Value used to fill missing packets.

    clone : bool
        If True, operate on a cloned tensor.
        If False, modify packets in-place.

    Returns
    -------
    ZeroFillResult
        Zero-filled packet tensor and statistics.
    """
    packets = _require_tensor(packets, "packets")

    if packets.dim() < 1:
        raise ValueError(
            f"packets should have at least 1 dimension, got {tuple(packets.shape)}."
        )

    num_packets = int(packets.shape[0])

    missing_mask = _resolve_missing_mask(
        num_packets=num_packets,
        missing_mask=missing_mask,
        available_mask=available_mask,
        device=packets.device,
    )

    recovered = packets.clone() if clone else packets
    fill_value_cast = _safe_fill_value_for_tensor(fill_value, recovered)

    if num_packets > 0 and bool(missing_mask.any().item()):
        recovered[missing_mask] = fill_value_cast

    filled_mask = missing_mask.clone()
    still_missing_mask = torch.zeros_like(missing_mask)
    available_after = torch.ones_like(missing_mask)

    num_zero_filled = int(filled_mask.sum().item())

    counts = build_recovery_count_dict(
        num_source_packets=num_packets,
        num_fec_recovered=0,
        num_temporal_filled=0,
        num_spatial_filled=0,
        num_zero_filled=num_zero_filled,
        num_still_missing=0,
    )

    info = {
        "method": RECOVERY_METHOD_ZERO_FILL,
        "fill_value": float(fill_value),
        "input_shape": tuple(int(x) for x in packets.shape),
        "input_dtype": str(packets.dtype),
        "num_packets": int(num_packets),
        "num_missing_before": int(num_zero_filled),
        "num_zero_filled_packets": int(num_zero_filled),
        "num_still_missing_packets": 0,
        "recovery_ratio": float(counts["recovery_ratio"]),
        "counts": counts,
        "note": (
            "Missing packets are filled with a constant value. "
            "This makes the packet tensor complete but does not restore "
            "semantic information."
        ),
    }

    return ZeroFillResult(
        recovered=recovered,
        filled_mask=filled_mask,
        available_mask=available_after,
        missing_mask_before=missing_mask,
        still_missing_mask=still_missing_mask,
        info=info,
    )


def zero_fill_feature_by_metas(
    feature: torch.Tensor,
    packet_metas: Sequence[Any],
    missing_mask: Optional[Any] = None,
    available_mask: Optional[Any] = None,
    fill_value: Union[int, float] = 0.0,
    clone: bool = True,
) -> ZeroFillResult:
    """
    Zero-fill missing spatial regions in a full feature tensor.

    Parameters
    ----------
    feature : torch.Tensor
        Feature tensor with shape [C, H, W].

    packet_metas : sequence
        Packet metadata list. Each meta is expected to have:
            packet_id
            spatial_slice

        This is compatible with FeaturePacketMeta from packetizer.py.

    missing_mask : tensor-like, optional
        Shape [K]. True means packet is missing.

    available_mask : tensor-like, optional
        Shape [K]. True means packet is available.

    fill_value : int or float
        Value used to fill missing spatial regions.

    clone : bool
        If True, operate on a cloned feature.
        If False, modify feature in-place.

    Returns
    -------
    ZeroFillResult
        Zero-filled feature tensor and packet-level statistics.
    """
    feature = _require_tensor(feature, "feature")

    if feature.dim() != 3:
        raise ValueError(
            "zero_fill_feature_by_metas expects feature shape [C, H, W], "
            f"got {tuple(feature.shape)}."
        )

    packet_metas = list(packet_metas)
    num_packets = len(packet_metas)

    missing_mask = _resolve_missing_mask(
        num_packets=num_packets,
        missing_mask=missing_mask,
        available_mask=available_mask,
        device=feature.device,
    )

    recovered = feature.clone() if clone else feature
    fill_value_cast = _safe_fill_value_for_tensor(fill_value, recovered)

    for meta in packet_metas:
        packet_id = int(meta.packet_id)

        if packet_id < 0 or packet_id >= num_packets:
            raise ValueError(
                f"Invalid packet_id={packet_id} for num_packets={num_packets}."
            )

        if bool(missing_mask[packet_id].item()):
            h_slice, w_slice = meta.spatial_slice
            recovered[:, h_slice, w_slice] = fill_value_cast

    filled_mask = missing_mask.clone()
    still_missing_mask = torch.zeros_like(missing_mask)
    available_after = torch.ones_like(missing_mask)

    num_zero_filled = int(filled_mask.sum().item())

    counts = build_recovery_count_dict(
        num_source_packets=num_packets,
        num_fec_recovered=0,
        num_temporal_filled=0,
        num_spatial_filled=0,
        num_zero_filled=num_zero_filled,
        num_still_missing=0,
    )

    info = {
        "method": RECOVERY_METHOD_ZERO_FILL,
        "target": "feature_by_metas",
        "fill_value": float(fill_value),
        "input_shape": tuple(int(x) for x in feature.shape),
        "input_dtype": str(feature.dtype),
        "num_packets": int(num_packets),
        "num_missing_before": int(num_zero_filled),
        "num_zero_filled_packets": int(num_zero_filled),
        "num_still_missing_packets": 0,
        "recovery_ratio": float(counts["recovery_ratio"]),
        "counts": counts,
        "note": (
            "Missing packet spatial regions in the full feature map are "
            "filled with a constant value."
        ),
    }

    return ZeroFillResult(
        recovered=recovered,
        filled_mask=filled_mask,
        available_mask=available_after,
        missing_mask_before=missing_mask,
        still_missing_mask=still_missing_mask,
        info=info,
    )


def zero_fill_from_fec_decode(
    decode_result: Any,
    fill_value: Union[int, float] = 0.0,
    clone: bool = True,
) -> ZeroFillResult:
    """
    Zero-fill packets using a FECDecodeResult-like object.

    Expected fields:
        decode_result.recovered_packets
        decode_result.missing_source_mask

    Parameters
    ----------
    decode_result : object
        FECDecodeResult from fec_base.py or compatible object.

    fill_value : int or float
        Value used for missing packets.

    clone : bool
        If True, clone recovered_packets before filling.

    Returns
    -------
    ZeroFillResult
        Zero-filled packets and statistics.
    """
    if not hasattr(decode_result, "recovered_packets"):
        raise AttributeError("decode_result should have recovered_packets.")

    if not hasattr(decode_result, "missing_source_mask"):
        raise AttributeError("decode_result should have missing_source_mask.")

    return zero_fill_packets(
        packets=decode_result.recovered_packets,
        missing_mask=decode_result.missing_source_mask,
        fill_value=fill_value,
        clone=clone,
    )


def apply_zero_fill_to_packetization_result(
    packetization_result: Any,
    missing_mask: Optional[Any] = None,
    available_mask: Optional[Any] = None,
    fill_value: Union[int, float] = 0.0,
    clone: bool = True,
) -> ZeroFillResult:
    """
    Zero-fill packets inside a PacketizationResult-like object.

    Expected fields:
        packetization_result.packets

    Parameters
    ----------
    packetization_result : object
        Result from FeaturePacketizer.packetize().

    missing_mask : tensor-like, optional
        Shape [K]. True means packet is missing.

    available_mask : tensor-like, optional
        Shape [K]. True means packet is available.

    Returns
    -------
    ZeroFillResult
        Zero-filled packets and statistics.
    """
    if not hasattr(packetization_result, "packets"):
        raise AttributeError("packetization_result should have packets.")

    return zero_fill_packets(
        packets=packetization_result.packets,
        missing_mask=missing_mask,
        available_mask=available_mask,
        fill_value=fill_value,
        clone=clone,
    )


class ZeroFillRecovery:
    """
    Object-oriented wrapper for zero-fill recovery.

    YAML style:

        arce:
          recovery:
            zero_fill: true
            zero_fill_value: 0.0

    Or direct config:

        zero_fill = ZeroFillRecovery({"fill_value": 0.0})
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = self._extract_zero_fill_cfg(cfg or {})

        self.enabled = bool(cfg.get("enabled", cfg.get("zero_fill", True)))
        self.fill_value = float(
            cfg.get(
                "fill_value",
                cfg.get("zero_fill_value", 0.0),
            )
        )

    @staticmethod
    def _extract_zero_fill_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accept full ARCE config or direct recovery config.
        """
        if "recovery" in cfg and isinstance(cfg["recovery"], dict):
            return cfg["recovery"]

        return cfg

    def recover_packets(
        self,
        packets: torch.Tensor,
        missing_mask: Optional[Any] = None,
        available_mask: Optional[Any] = None,
        fill_value: Optional[Union[int, float]] = None,
        clone: bool = True,
    ) -> ZeroFillResult:
        """
        Zero-fill packet tensor.

        If this recovery is disabled, return packets unchanged and keep masks.
        """
        packets = _require_tensor(packets, "packets")

        if fill_value is None:
            fill_value = self.fill_value

        if not self.enabled:
            num_packets = int(packets.shape[0])
            missing = _resolve_missing_mask(
                num_packets=num_packets,
                missing_mask=missing_mask,
                available_mask=available_mask,
                device=packets.device,
            )
            available = missing_mask_to_available_mask(missing)

            info = {
                "method": RECOVERY_METHOD_ZERO_FILL,
                "enabled": False,
                "num_packets": int(num_packets),
                "num_missing_before": int(missing.sum().item()),
                "num_zero_filled_packets": 0,
                "num_still_missing_packets": int(missing.sum().item()),
                "recovery_ratio": (
                    float(available.sum().item() / num_packets)
                    if num_packets > 0
                    else 1.0
                ),
                "note": "Zero-fill recovery is disabled.",
            }

            return ZeroFillResult(
                recovered=packets.clone() if clone else packets,
                filled_mask=torch.zeros_like(missing),
                available_mask=available,
                missing_mask_before=missing,
                still_missing_mask=missing,
                info=info,
            )

        return zero_fill_packets(
            packets=packets,
            missing_mask=missing_mask,
            available_mask=available_mask,
            fill_value=fill_value,
            clone=clone,
        )

    def recover_feature_by_metas(
        self,
        feature: torch.Tensor,
        packet_metas: Sequence[Any],
        missing_mask: Optional[Any] = None,
        available_mask: Optional[Any] = None,
        fill_value: Optional[Union[int, float]] = None,
        clone: bool = True,
    ) -> ZeroFillResult:
        """
        Zero-fill spatial regions in a full feature map using packet metadata.
        """
        feature = _require_tensor(feature, "feature")

        if fill_value is None:
            fill_value = self.fill_value

        if not self.enabled:
            num_packets = len(list(packet_metas))
            missing = _resolve_missing_mask(
                num_packets=num_packets,
                missing_mask=missing_mask,
                available_mask=available_mask,
                device=feature.device,
            )
            available = missing_mask_to_available_mask(missing)

            info = {
                "method": RECOVERY_METHOD_ZERO_FILL,
                "enabled": False,
                "target": "feature_by_metas",
                "num_packets": int(num_packets),
                "num_missing_before": int(missing.sum().item()),
                "num_zero_filled_packets": 0,
                "num_still_missing_packets": int(missing.sum().item()),
                "recovery_ratio": (
                    float(available.sum().item() / num_packets)
                    if num_packets > 0
                    else 1.0
                ),
                "note": "Zero-fill recovery is disabled.",
            }

            return ZeroFillResult(
                recovered=feature.clone() if clone else feature,
                filled_mask=torch.zeros_like(missing),
                available_mask=available,
                missing_mask_before=missing,
                still_missing_mask=missing,
                info=info,
            )

        return zero_fill_feature_by_metas(
            feature=feature,
            packet_metas=packet_metas,
            missing_mask=missing_mask,
            available_mask=available_mask,
            fill_value=fill_value,
            clone=clone,
        )

    def recover_from_fec_decode(
        self,
        decode_result: Any,
        fill_value: Optional[Union[int, float]] = None,
        clone: bool = True,
    ) -> ZeroFillResult:
        """
        Zero-fill missing packets from a FECDecodeResult-like object.
        """
        if fill_value is None:
            fill_value = self.fill_value

        if not self.enabled:
            if not hasattr(decode_result, "recovered_packets"):
                raise AttributeError("decode_result should have recovered_packets.")
            if not hasattr(decode_result, "missing_source_mask"):
                raise AttributeError("decode_result should have missing_source_mask.")

            return self.recover_packets(
                packets=decode_result.recovered_packets,
                missing_mask=decode_result.missing_source_mask,
                fill_value=fill_value,
                clone=clone,
            )

        return zero_fill_from_fec_decode(
            decode_result=decode_result,
            fill_value=fill_value,
            clone=clone,
        )

    def get_config(self) -> Dict[str, Any]:
        """
        Return JSON-friendly zero-fill config.
        """
        return {
            "enabled": bool(self.enabled),
            "method": RECOVERY_METHOD_ZERO_FILL,
            "fill_value": float(self.fill_value),
        }

    def __repr__(self) -> str:
        return (
            "ZeroFillRecovery("
            f"enabled={self.enabled}, "
            f"fill_value={self.fill_value})"
        )


# Backward-compatible aliases.
ZeroFill = ZeroFillRecovery
ZeroFillReconstructor = ZeroFillRecovery


__all__ = [
    "ZeroFillResult",
    "zero_fill_packets",
    "zero_fill_feature_by_metas",
    "zero_fill_from_fec_decode",
    "apply_zero_fill_to_packetization_result",
    "ZeroFillRecovery",
    "ZeroFill",
    "ZeroFillReconstructor",
]