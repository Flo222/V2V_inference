"""
High-level feature quantizer for ARCE communication simulation.

This module wraps low-level quantization functions from quant_utils.py and
provides a clean interface for ARCE:

    feature / packets
    -> quantize
    -> optional int4 packing
    -> dequantize
    -> quantization error statistics
    -> logging-friendly metadata

Supported modes:
    fp32:
        no quantization.

    fp16:
        real float16 cast and cast back.

    int8:
        symmetric signed integer quantization in range [-127, 127].

    int4:
        symmetric signed integer quantization in range [-7, 7].
        Stored as torch.int8 by default. Optional packed uint8 storage is
        supported for byte-level experiments.

Typical ARCE usage:

    quantizer = FeatureQuantizer(arce_cfg)

    result = quantizer.quantize_feature(feature, mode="int8")
    q_tensor = result.q_tensor          # integer or float quantized tensor
    x_hat = result.dequantized          # float tensor for V2X-ViT
    meta = result.meta                  # scale / dtype / shape metadata

For real XOR FEC:
    Use result.q_tensor on packet tensors.
    Perform XOR in the integer domain.
    Then call quantizer.dequantize(recovered_q, result.meta).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from opencood.communication.transport.quantization import (
    QUANT_MODE_FP32,
    QUANT_MODE_FP16,
    QUANT_MODE_INT8,
    QUANT_MODE_INT4,
    DEFAULT_RAW_BITS,
    normalize_quant_mode,
    quant_mode_to_bits,
    compression_ratio_from_quant_mode,
    is_integer_quant_mode,
    is_float_quant_mode,
    get_quant_config_summary,
)

from opencood.communication.transport.quantization.quant_utils import (
    QuantizationMeta,
    QuantizationResult,
    quantize_tensor,
    dequantize_tensor,
    compute_quantization_error,
    estimate_tensor_bits,
    estimate_tensor_bytes,
    pack_int4_signed,
    unpack_int4_signed,
)


def _extract_quant_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Accept either full ARCE config or direct quantization config.

    Supported:
        FeatureQuantizer(arce_cfg)
        FeatureQuantizer(arce_cfg["quantization"])
    """
    cfg = cfg or {}

    if "quantization" in cfg and isinstance(cfg["quantization"], dict):
        return cfg["quantization"]

    return cfg


def _require_tensor(x: Any, name: str = "tensor") -> torch.Tensor:
    """
    Validate torch.Tensor input.
    """
    if not torch.is_tensor(x):
        raise TypeError(f"{name} should be a torch.Tensor, got {type(x)}.")
    return x


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


def _as_bool(value: Any) -> bool:
    """
    Convert common config values to bool.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("true", "1", "yes", "y", "on"):
            return True
        if value in ("false", "0", "no", "n", "off"):
            return False

    return bool(value)


def _normalize_granularity(granularity: Optional[str]) -> str:
    """
    Normalize quantization granularity.
    """
    if granularity is None:
        return "per_tensor"

    granularity = str(granularity).strip().lower()

    if granularity not in ("per_tensor", "per_channel"):
        raise ValueError(
            f"Unsupported quantization granularity: {granularity}. "
            "Expected per_tensor or per_channel."
        )

    return granularity


def _infer_channel_dim(x: torch.Tensor, requested_channel_dim: Optional[int]) -> int:
    """
    Infer channel dimension.

    For feature:
        [C, H, W] -> channel_dim = 0

    For packet tensor:
        [M, C, H, W] -> channel_dim = 1

    For batched feature:
        [N, C, H, W] -> channel_dim = 1
    """
    if requested_channel_dim is not None:
        return int(requested_channel_dim)

    if x.dim() == 3:
        return 0

    if x.dim() == 4:
        return 1

    return 0


@dataclass
class FeatureQuantizationResult:
    """
    High-level quantization result.

    Attributes
    ----------
    q_tensor : torch.Tensor
        Quantized representation.

        fp32:
            float tensor.

        fp16:
            float16 tensor.

        int8:
            torch.int8 tensor in range [-127, 127].

        int4:
            torch.int8 tensor in range [-7, 7], unless packed_int4 is used.

    dequantized : torch.Tensor
        Dequantized float tensor. This is the tensor passed to V2X-ViT.

    meta : QuantizationMeta
        Metadata needed for dequantization.

    error : dict
        Quantization error statistics.

    packed_tensor : torch.Tensor or None
        Packed uint8 tensor for int4 when pack_int4=True.

    info : dict
        Logging-friendly summary.
    """

    q_tensor: torch.Tensor
    dequantized: torch.Tensor
    meta: QuantizationMeta
    error: Dict[str, float]
    packed_tensor: Optional[torch.Tensor]
    info: Dict[str, Any]

    @property
    def mode(self) -> str:
        return self.meta.mode

    @property
    def bits(self) -> int:
        return int(self.meta.bits)

    @property
    def compression_ratio(self) -> float:
        return float(self.meta.compression_ratio)

    def as_dict(self) -> Dict[str, Any]:
        """
        Return JSON-friendly quantization summary.
        """
        result = dict(self.info)
        result["meta"] = self.meta.as_dict()
        result["error"] = dict(self.error)
        return result


class FeatureQuantizer:
    """
    High-level quantizer for ARCE feature communication.

    YAML style:

        arce:
          quantization:
            enabled: true
            mode: int8
            raw_bits: 32
            granularity: per_tensor
            channel_dim: null
            compute_error: true
            pack_int4: false

    Notes
    -----
    If enabled is false, the quantizer falls back to fp32 passthrough.

    If enabled is omitted:
        - mode=fp32 means disabled / passthrough;
        - mode=fp16/int8/int4 means enabled.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.full_cfg = cfg or {}
        self.cfg = _extract_quant_cfg(cfg)

        self.mode = normalize_quant_mode(
            self.cfg.get("mode", QUANT_MODE_FP32)
        )

        if "enabled" in self.cfg:
            self.enabled = _as_bool(self.cfg.get("enabled"))
        else:
            self.enabled = self.mode != QUANT_MODE_FP32

        self.raw_bits = _as_positive_int(
            self.cfg.get(
                "raw_bits",
                self.cfg.get("raw_bits_per_value", DEFAULT_RAW_BITS),
            ),
            "raw_bits",
        )

        self.granularity = _normalize_granularity(
            self.cfg.get("granularity", "per_tensor")
        )

        channel_dim_cfg = self.cfg.get("channel_dim", None)
        self.channel_dim = None if channel_dim_cfg is None else int(channel_dim_cfg)

        self.output_dtype = self._parse_output_dtype(
            self.cfg.get("output_dtype", None)
        )

        self.clone_fp32 = _as_bool(self.cfg.get("clone_fp32", False))
        self.compute_error = _as_bool(self.cfg.get("compute_error", True))
        self.pack_int4 = _as_bool(self.cfg.get("pack_int4", False))

        self.eps = float(self.cfg.get("eps", 1e-8))

    @staticmethod
    def _parse_output_dtype(dtype_value: Optional[Any]) -> Optional[torch.dtype]:
        """
        Parse output dtype from config.

        Supported strings:
            float32 / fp32
            float16 / fp16
            bfloat16 / bf16
        """
        if dtype_value is None:
            return None

        if isinstance(dtype_value, torch.dtype):
            return dtype_value

        dtype_value = str(dtype_value).strip().lower()

        if dtype_value in ("float32", "fp32", "torch.float32"):
            return torch.float32

        if dtype_value in ("float16", "fp16", "half", "torch.float16"):
            return torch.float16

        if dtype_value in ("bfloat16", "bf16", "torch.bfloat16"):
            return torch.bfloat16

        raise ValueError(f"Unsupported output_dtype: {dtype_value}")

    def _resolve_mode(self, mode: Optional[str] = None, enabled: Optional[bool] = None) -> str:
        """
        Resolve effective quantization mode.

        If quantization is disabled, always return fp32.
        """
        if enabled is None:
            enabled = self.enabled
        else:
            enabled = bool(enabled)

        if not enabled:
            return QUANT_MODE_FP32

        return normalize_quant_mode(mode or self.mode)

    def _resolve_granularity(self, granularity: Optional[str] = None) -> str:
        """
        Resolve granularity.
        """
        return _normalize_granularity(granularity or self.granularity)

    def _resolve_channel_dim(
        self,
        x: torch.Tensor,
        channel_dim: Optional[int] = None,
    ) -> int:
        """
        Resolve channel dimension.
        """
        if channel_dim is not None:
            return int(channel_dim)

        return _infer_channel_dim(x, self.channel_dim)

    def quantize(
        self,
        x: torch.Tensor,
        mode: Optional[str] = None,
        enabled: Optional[bool] = None,
        granularity: Optional[str] = None,
        channel_dim: Optional[int] = None,
        output_dtype: Optional[torch.dtype] = None,
        compute_error: Optional[bool] = None,
        pack_int4: Optional[bool] = None,
        clone_fp32: Optional[bool] = None,
    ) -> FeatureQuantizationResult:
        """
        Quantize a tensor and return q_tensor, dequantized tensor, metadata,
        and quantization error.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor. Can be:
                [C, H, W]
                [M, C, H, W]
                [N, C, H, W]

        mode : str, optional
            fp32 / fp16 / int8 / int4.

        enabled : bool, optional
            If False, force fp32 passthrough.

        granularity : str, optional
            per_tensor or per_channel.

        channel_dim : int, optional
            Channel dimension for per_channel quantization.

        output_dtype : torch.dtype, optional
            Dtype of dequantized output.

        compute_error : bool, optional
            Whether to compute quantization error.

        pack_int4 : bool, optional
            Whether to additionally pack int4 q_tensor into uint8 bytes.

        clone_fp32 : bool, optional
            Whether fp32 passthrough clones x.

        Returns
        -------
        FeatureQuantizationResult
        """
        x = _require_tensor(x, "x")

        effective_mode = self._resolve_mode(mode=mode, enabled=enabled)
        effective_granularity = self._resolve_granularity(granularity)
        effective_channel_dim = self._resolve_channel_dim(x, channel_dim)

        if output_dtype is None:
            output_dtype = self.output_dtype

        if compute_error is None:
            compute_error = self.compute_error

        if pack_int4 is None:
            pack_int4 = self.pack_int4

        if clone_fp32 is None:
            clone_fp32 = self.clone_fp32

        low_result: QuantizationResult = quantize_tensor(
            x=x,
            mode=effective_mode,
            granularity=effective_granularity,
            channel_dim=effective_channel_dim,
            output_dtype=output_dtype,
            raw_bits=self.raw_bits,
            clone_fp32=clone_fp32,
            eps=self.eps,
        )

        if compute_error:
            error = compute_quantization_error(
                original=x,
                recovered=low_result.dequantized,
            )
        else:
            error = {
                "mse": 0.0,
                "mae": 0.0,
                "max_abs_error": 0.0,
                "mean_abs_original": 0.0,
                "relative_mae": 0.0,
            }

        packed_tensor = None
        packed_num_bytes = None

        if effective_mode == QUANT_MODE_INT4 and pack_int4:
            packed_tensor = pack_int4_signed(low_result.q_tensor)
            packed_num_bytes = int(packed_tensor.numel())

        info = self._build_info(
            x=x,
            result=low_result,
            error=error,
            packed_tensor=packed_tensor,
            packed_num_bytes=packed_num_bytes,
        )

        return FeatureQuantizationResult(
            q_tensor=low_result.q_tensor,
            dequantized=low_result.dequantized,
            meta=low_result.meta,
            error=error,
            packed_tensor=packed_tensor,
            info=info,
        )

    def quantize_feature(
        self,
        feature: torch.Tensor,
        mode: Optional[str] = None,
        **kwargs,
    ) -> FeatureQuantizationResult:
        """
        Quantize one feature tensor.

        Expected shape:
            [C, H, W]

        For per_channel quantization, channel_dim defaults to 0.
        """
        feature = _require_tensor(feature, "feature")

        if feature.dim() != 3:
            raise ValueError(
                "quantize_feature expects shape [C, H, W], "
                f"got {tuple(feature.shape)}."
            )

        kwargs.setdefault("channel_dim", 0)
        return self.quantize(feature, mode=mode, **kwargs)

    def quantize_batch(
        self,
        features: torch.Tensor,
        mode: Optional[str] = None,
        **kwargs,
    ) -> FeatureQuantizationResult:
        """
        Quantize a batch of features.

        Expected shape:
            [N, C, H, W]

        For per_channel quantization, channel_dim defaults to 1.
        """
        features = _require_tensor(features, "features")

        if features.dim() != 4:
            raise ValueError(
                "quantize_batch expects shape [N, C, H, W], "
                f"got {tuple(features.shape)}."
            )

        kwargs.setdefault("channel_dim", 1)
        return self.quantize(features, mode=mode, **kwargs)

    def quantize_packets(
        self,
        packets: torch.Tensor,
        mode: Optional[str] = None,
        **kwargs,
    ) -> FeatureQuantizationResult:
        """
        Quantize packet tensor.

        Expected shape:
            [M, C, packet_h, packet_w]

        For per_channel quantization, channel_dim defaults to 1.

        This is the preferred API before real XOR FEC, because XOR should be
        performed on q_tensor in integer modes.
        """
        packets = _require_tensor(packets, "packets")

        if packets.dim() != 4:
            raise ValueError(
                "quantize_packets expects shape [M, C, H, W], "
                f"got {tuple(packets.shape)}."
            )

        kwargs.setdefault("channel_dim", 1)
        return self.quantize(packets, mode=mode, **kwargs)

    def quantize_dequantize(
        self,
        x: torch.Tensor,
        mode: Optional[str] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Convenience API: return only dequantized tensor and logging info.

        This is useful for quick integration:

            x_hat, q_info = quantizer.quantize_dequantize(x, mode="int8")
        """
        result = self.quantize(x, mode=mode, **kwargs)
        return result.dequantized, result.as_dict()

    def dequantize(
        self,
        q_tensor: torch.Tensor,
        meta: QuantizationMeta,
        output_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Dequantize tensor using metadata.

        This is used after FEC recovery in the integer domain.
        """
        return dequantize_tensor(
            q_tensor=q_tensor,
            meta=meta,
            output_dtype=output_dtype,
        )

    def unpack_and_dequantize_int4(
        self,
        packed_tensor: torch.Tensor,
        meta: QuantizationMeta,
        original_numel: Optional[int] = None,
        shape: Optional[Sequence[int]] = None,
        output_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Unpack packed int4 tensor and dequantize it.

        Parameters
        ----------
        packed_tensor : torch.Tensor
            Packed uint8 tensor from pack_int4_signed().

        meta : QuantizationMeta
            Quantization metadata from int4 quantization.

        original_numel : int, optional
            Number of int4 values before packing padding.

        shape : sequence, optional
            Shape of unpacked q_tensor.

        output_dtype : torch.dtype, optional
            Dtype of dequantized tensor.
        """
        if normalize_quant_mode(meta.mode) != QUANT_MODE_INT4:
            raise ValueError(
                "unpack_and_dequantize_int4 requires meta.mode == int4, "
                f"got {meta.mode}."
            )

        q_tensor = unpack_int4_signed(
            packed=packed_tensor,
            original_numel=original_numel,
            shape=shape,
            device=packed_tensor.device,
        )

        return self.dequantize(
            q_tensor=q_tensor,
            meta=meta,
            output_dtype=output_dtype,
        )

    def _build_info(
        self,
        x: torch.Tensor,
        result: QuantizationResult,
        error: Dict[str, float],
        packed_tensor: Optional[torch.Tensor],
        packed_num_bytes: Optional[int],
    ) -> Dict[str, Any]:
        """
        Build logging-friendly quantization info.
        """
        meta = result.meta
        mode = normalize_quant_mode(meta.mode)

        estimated_bits = estimate_tensor_bits(x, mode=mode, raw_bits=self.raw_bits)
        estimated_bytes = estimate_tensor_bytes(x, mode=mode, ceil_to_byte=True)

        raw_bits_total = int(x.numel() * self.raw_bits)
        raw_bytes_total = float((raw_bits_total + 7) // 8)

        info = {
            "enabled": bool(mode != QUANT_MODE_FP32 or self.enabled),
            "mode": mode,
            "bits": int(meta.bits),
            "raw_bits": int(meta.raw_bits),
            "compression_ratio": float(meta.compression_ratio),
            "is_integer": bool(is_integer_quant_mode(mode)),
            "is_float": bool(is_float_quant_mode(mode)),
            "input_shape": tuple(int(v) for v in x.shape),
            "input_dtype": str(x.dtype),
            "q_shape": tuple(int(v) for v in result.q_tensor.shape),
            "q_dtype": str(result.q_tensor.dtype),
            "dequantized_shape": tuple(int(v) for v in result.dequantized.shape),
            "dequantized_dtype": str(result.dequantized.dtype),
            "estimated_bits": int(estimated_bits),
            "estimated_bytes": float(estimated_bytes),
            "raw_bits_total": int(raw_bits_total),
            "raw_bytes_total": float(raw_bytes_total),
            "granularity": meta.granularity,
            "channel_dim": int(meta.channel_dim),
            "scale_summary": meta.scale_summary(),
            "error": dict(error),
            "packed_int4": bool(packed_tensor is not None),
            "packed_num_bytes": packed_num_bytes,
        }

        return info

    def estimate_bits(
        self,
        x_or_shape: Any,
        mode: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> int:
        """
        Estimate tensor bits under effective quantization mode.
        """
        effective_mode = self._resolve_mode(mode=mode, enabled=enabled)
        return estimate_tensor_bits(
            x_or_shape,
            mode=effective_mode,
            raw_bits=self.raw_bits,
        )

    def estimate_bytes(
        self,
        x_or_shape: Any,
        mode: Optional[str] = None,
        enabled: Optional[bool] = None,
        ceil_to_byte: bool = True,
    ) -> float:
        """
        Estimate tensor bytes under effective quantization mode.
        """
        effective_mode = self._resolve_mode(mode=mode, enabled=enabled)
        bits = self.estimate_bits(
            x_or_shape,
            mode=effective_mode,
            enabled=True,
        )

        if ceil_to_byte:
            return float((bits + 7) // 8)

        return float(bits / 8.0)

    def get_mode_bits(self, mode: Optional[str] = None, enabled: Optional[bool] = None) -> int:
        """
        Return bits per value for the effective mode.
        """
        effective_mode = self._resolve_mode(mode=mode, enabled=enabled)
        return int(quant_mode_to_bits(effective_mode))

    def get_compression_ratio(
        self,
        mode: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> float:
        """
        Return compression ratio for the effective mode.
        """
        effective_mode = self._resolve_mode(mode=mode, enabled=enabled)
        return float(
            compression_ratio_from_quant_mode(
                effective_mode,
                raw_bits=self.raw_bits,
            )
        )

    def get_config(self) -> Dict[str, Any]:
        """
        Export quantizer config.
        """
        return {
            "enabled": bool(self.enabled),
            "mode": self.mode,
            "raw_bits": int(self.raw_bits),
            "granularity": self.granularity,
            "channel_dim": self.channel_dim,
            "output_dtype": str(self.output_dtype) if self.output_dtype is not None else None,
            "clone_fp32": bool(self.clone_fp32),
            "compute_error": bool(self.compute_error),
            "pack_int4": bool(self.pack_int4),
            "eps": float(self.eps),
            "mode_summary": get_quant_config_summary(
                self.mode,
                raw_bits=self.raw_bits,
            ),
        }

    def __repr__(self) -> str:
        return (
            "FeatureQuantizer("
            f"enabled={self.enabled}, "
            f"mode={self.mode}, "
            f"raw_bits={self.raw_bits}, "
            f"granularity={self.granularity}, "
            f"channel_dim={self.channel_dim}, "
            f"pack_int4={self.pack_int4})"
        )