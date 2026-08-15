"""
Low-level quantization utilities for ARCE feature compression.

This module provides real quantization utilities for intermediate BEV
features in OPV2V / V2X-ViT experiments.

Supported modes:
    fp32:
        no quantization.

    fp16:
        real float16 casting and cast back to float.

    int8:
        symmetric signed integer quantization with range [-127, 127].

    int4:
        symmetric signed integer quantization with range [-7, 7].
        Values are stored in torch.int8 tensors by default, because PyTorch
        does not provide a standard int4 dtype. Optional bit packing helpers
        are provided for communication-size / storage experiments.

Important:
    This module performs quantization math only.
    It does NOT:
        - split features into packets;
        - perform FEC;
        - sample packet loss;
        - reconstruct missing packets;
        - log communication statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch

from opencood.communication.transport.quantization import (
    QUANT_MODE_FP32,
    QUANT_MODE_FP16,
    QUANT_MODE_INT8,
    QUANT_MODE_INT4,
    DEFAULT_RAW_BITS,
    normalize_quant_mode,
    quant_mode_to_bits,
    get_quant_range,
    is_integer_quant_mode,
    is_float_quant_mode,
    compression_ratio_from_quant_mode,
)


TensorOrNone = Optional[torch.Tensor]


def _require_tensor(x: Any, name: str = "tensor") -> torch.Tensor:
    """
    Validate that input is a torch.Tensor.
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


def _canonical_dtype(dtype: Optional[torch.dtype], fallback: torch.dtype) -> torch.dtype:
    """
    Return dtype if not None, otherwise fallback.
    """
    return fallback if dtype is None else dtype


def get_symmetric_quant_params(mode: str) -> Tuple[int, int, int]:
    """
    Get symmetric quantization parameters.

    Parameters
    ----------
    mode : str
        Quantization mode. Supports int8 / int4.

    Returns
    -------
    tuple
        (qmin, qmax, bits)

    Notes
    -----
    We use signed symmetric ranges:
        int8: [-127, 127]
        int4: [-7, 7]

    This keeps zero exactly representable and avoids asymmetric zero-point
    handling.
    """
    mode = normalize_quant_mode(mode)

    quant_range = get_quant_range(mode)
    if quant_range is None:
        raise ValueError(
            f"Mode {mode} is not an integer quantization mode. "
            "Expected int8 or int4."
        )

    qmin, qmax = quant_range
    bits = quant_mode_to_bits(mode)

    return int(qmin), int(qmax), int(bits)


def _safe_scale_from_max_abs(
    max_abs: torch.Tensor,
    qmax: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Convert max_abs to scale safely.

    Formula:
        scale = max_abs / qmax

    If max_abs is too small, use scale=1.0 to avoid division by zero.
    """
    if qmax <= 0:
        raise ValueError(f"qmax should be positive, got {qmax}.")

    scale = max_abs / float(qmax)

    one = torch.ones_like(scale)
    scale = torch.where(scale > eps, scale, one)

    return scale


def compute_symmetric_scale(
    x: torch.Tensor,
    mode: str,
    granularity: str = "per_tensor",
    channel_dim: int = 0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute symmetric quantization scale.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    mode : str
        int8 or int4.

    granularity : str
        "per_tensor" or "per_channel".

    channel_dim : int
        Channel dimension used when granularity="per_channel".
        For a feature [C, H, W], channel_dim=0.
        For packets [M, C, H, W], channel_dim=1 may be used.

    eps : float
        Numerical stability threshold.

    Returns
    -------
    torch.Tensor
        Per-tensor scale: scalar tensor.
        Per-channel scale: tensor with shape [C].
    """
    x = _require_tensor(x, "x")
    _, qmax, _ = get_symmetric_quant_params(mode)

    granularity = str(granularity).strip().lower()

    if granularity == "per_tensor":
        max_abs = x.detach().abs().max()
        return _safe_scale_from_max_abs(max_abs, qmax, eps=eps)

    if granularity == "per_channel":
        ndim = x.dim()
        if ndim == 0:
            raise ValueError("per_channel quantization requires tensor dim >= 1.")

        if channel_dim < 0:
            channel_dim = ndim + channel_dim

        if channel_dim < 0 or channel_dim >= ndim:
            raise ValueError(
                f"channel_dim={channel_dim} out of range for tensor dim={ndim}."
            )

        reduce_dims = [dim for dim in range(ndim) if dim != channel_dim]

        max_abs = x.detach().abs()
        if reduce_dims:
            max_abs = max_abs.amax(dim=reduce_dims)
        scale = _safe_scale_from_max_abs(max_abs, qmax, eps=eps)
        return scale

    raise ValueError(
        f"Unsupported quantization granularity: {granularity}. "
        "Expected per_tensor or per_channel."
    )


def _reshape_scale_for_broadcast(
    scale: torch.Tensor,
    x: torch.Tensor,
    granularity: str = "per_tensor",
    channel_dim: int = 0,
) -> torch.Tensor:
    """
    Reshape scale for broadcasting over x.
    """
    granularity = str(granularity).strip().lower()

    if granularity == "per_tensor":
        return scale

    if granularity != "per_channel":
        raise ValueError(
            f"Unsupported granularity: {granularity}. "
            "Expected per_tensor or per_channel."
        )

    if channel_dim < 0:
        channel_dim = x.dim() + channel_dim

    view_shape = [1] * x.dim()
    view_shape[channel_dim] = -1

    return scale.view(*view_shape)


def symmetric_quantize(
    x: torch.Tensor,
    mode: str,
    scale: Optional[torch.Tensor] = None,
    granularity: str = "per_tensor",
    channel_dim: int = 0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric signed integer quantization.

    Formula:
        q = round(x / scale).clamp(qmin, qmax)

    Parameters
    ----------
    x : torch.Tensor
        Float tensor to quantize.

    mode : str
        int8 or int4.

    scale : torch.Tensor, optional
        Precomputed scale. If None, compute it from x.

    granularity : str
        per_tensor or per_channel.

    channel_dim : int
        Channel dimension for per_channel quantization.

    Returns
    -------
    q : torch.Tensor
        Quantized integer tensor stored as torch.int8.
        For int4, values are still stored in torch.int8 with range [-7, 7].

    scale : torch.Tensor
        Scale used for quantization.
    """
    x = _require_tensor(x, "x")
    mode = normalize_quant_mode(mode)

    if not is_integer_quant_mode(mode):
        raise ValueError(
            f"symmetric_quantize only supports int8 / int4, got {mode}."
        )

    qmin, qmax, _ = get_symmetric_quant_params(mode)

    if scale is None:
        scale = compute_symmetric_scale(
            x,
            mode=mode,
            granularity=granularity,
            channel_dim=channel_dim,
            eps=eps,
        )
    else:
        scale = _require_tensor(scale, "scale").to(device=x.device, dtype=x.dtype)

    scale_b = _reshape_scale_for_broadcast(
        scale=scale,
        x=x,
        granularity=granularity,
        channel_dim=channel_dim,
    )

    q = torch.round(x / scale_b).clamp(qmin, qmax).to(torch.int8)

    return q, scale.detach()


def symmetric_dequantize(
    q: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
    granularity: str = "per_tensor",
    channel_dim: int = 0,
) -> torch.Tensor:
    """
    Dequantize symmetric integer tensor.

    Formula:
        x_hat = q.float() * scale

    Parameters
    ----------
    q : torch.Tensor
        Quantized integer tensor. Usually torch.int8.

    scale : torch.Tensor
        Quantization scale.

    output_dtype : torch.dtype
        Output float dtype.

    granularity : str
        per_tensor or per_channel.

    channel_dim : int
        Channel dimension for per_channel quantization.

    Returns
    -------
    torch.Tensor
        Dequantized tensor.
    """
    q = _require_tensor(q, "q")
    scale = _require_tensor(scale, "scale").to(device=q.device)

    x_float = q.to(torch.float32)

    scale_b = _reshape_scale_for_broadcast(
        scale=scale.to(torch.float32),
        x=x_float,
        granularity=granularity,
        channel_dim=channel_dim,
    )

    x_hat = x_float * scale_b

    return x_hat.to(output_dtype)


def fp16_quant_dequant(
    x: torch.Tensor,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Real FP16 quantize-dequantize operation.

    This casts x to torch.float16 and then casts it back to output_dtype.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    output_dtype : torch.dtype, optional
        Output dtype. If None, use x.dtype.

    Returns
    -------
    torch.Tensor
        FP16-rounded tensor in output_dtype.
    """
    x = _require_tensor(x, "x")
    output_dtype = _canonical_dtype(output_dtype, x.dtype)

    return x.to(torch.float16).to(output_dtype)


def fp32_passthrough(
    x: torch.Tensor,
    clone: bool = False,
) -> torch.Tensor:
    """
    FP32 mode. By default, return x directly.

    Parameters
    ----------
    clone : bool
        If True, return x.clone().
    """
    x = _require_tensor(x, "x")
    return x.clone() if clone else x


@dataclass
class QuantizationMeta:
    """
    Metadata needed for dequantization and logging.

    Attributes
    ----------
    mode : str
        fp32 / fp16 / int8 / int4.

    bits : int
        Bits per value.

    raw_bits : int
        Raw bits per value, normally 32.

    granularity : str
        per_tensor or per_channel.

    channel_dim : int
        Channel dimension used in per_channel quantization.

    scale : torch.Tensor or None
        Quantization scale for integer modes.

    qmin, qmax : int or None
        Integer quantization range.

    original_dtype : torch.dtype
        Original tensor dtype.

    original_shape : tuple
        Original tensor shape.
    """

    mode: str
    bits: int
    raw_bits: int
    granularity: str
    channel_dim: int
    scale: TensorOrNone
    qmin: Optional[int]
    qmax: Optional[int]
    original_dtype: torch.dtype
    original_shape: Tuple[int, ...]

    @property
    def compression_ratio(self) -> float:
        return float(self.bits / self.raw_bits)

    def scale_summary(self) -> Optional[Dict[str, float]]:
        """
        Return a JSON-friendly summary of scale.
        """
        if self.scale is None:
            return None

        s = self.scale.detach().float()
        return {
            "scale_shape": tuple(int(x) for x in s.shape),
            "scale_min": float(s.min().item()),
            "scale_max": float(s.max().item()),
            "scale_mean": float(s.mean().item()),
        }

    def as_dict(self) -> Dict[str, Any]:
        """
        Return JSON-serializable metadata summary.

        The full scale tensor is not serialized here; only its summary is.
        """
        return {
            "mode": self.mode,
            "bits": int(self.bits),
            "raw_bits": int(self.raw_bits),
            "compression_ratio": float(self.compression_ratio),
            "granularity": self.granularity,
            "channel_dim": int(self.channel_dim),
            "qmin": self.qmin,
            "qmax": self.qmax,
            "original_dtype": str(self.original_dtype),
            "original_shape": tuple(int(x) for x in self.original_shape),
            "scale_summary": self.scale_summary(),
        }


@dataclass
class QuantizationResult:
    """
    Result of quantization.

    q_tensor:
        Quantized representation.

        fp32:
            float tensor, same values as input.

        fp16:
            torch.float16 tensor.

        int8:
            torch.int8 tensor in range [-127, 127].

        int4:
            torch.int8 tensor in range [-7, 7].

    dequantized:
        Float tensor after dequantization / casting back.
        This tensor should be passed to the perception model.

    meta:
        Quantization metadata.
    """

    q_tensor: torch.Tensor
    dequantized: torch.Tensor
    meta: QuantizationMeta

    def as_dict(self) -> Dict[str, Any]:
        """
        Return JSON-friendly summary.
        """
        return self.meta.as_dict()


def quantize_tensor(
    x: torch.Tensor,
    mode: str = QUANT_MODE_FP32,
    granularity: str = "per_tensor",
    channel_dim: int = 0,
    output_dtype: Optional[torch.dtype] = None,
    raw_bits: int = DEFAULT_RAW_BITS,
    clone_fp32: bool = False,
    eps: float = 1e-8,
) -> QuantizationResult:
    """
    Quantize a tensor and also return dequantized float tensor.

    This is the main low-level API used by feature_quantizer.py.

    Parameters
    ----------
    x : torch.Tensor
        Input feature / packet tensor.

    mode : str
        fp32 / fp16 / int8 / int4.

    granularity : str
        per_tensor or per_channel for integer quantization.

    channel_dim : int
        Channel dimension for per_channel quantization.

    output_dtype : torch.dtype, optional
        Dtype of dequantized output. If None, use x.dtype.

    raw_bits : int
        Raw feature bits, normally 32.

    clone_fp32 : bool
        If True, fp32 mode clones x.

    eps : float
        Scale stability epsilon.

    Returns
    -------
    QuantizationResult
    """
    x = _require_tensor(x, "x")
    mode = normalize_quant_mode(mode)
    raw_bits = _as_positive_int(raw_bits, "raw_bits")
    output_dtype = _canonical_dtype(output_dtype, x.dtype)

    bits = quant_mode_to_bits(mode)
    original_shape = tuple(int(v) for v in x.shape)

    if mode == QUANT_MODE_FP32:
        q_tensor = fp32_passthrough(x, clone=clone_fp32)
        dequantized = q_tensor.to(output_dtype)

        meta = QuantizationMeta(
            mode=mode,
            bits=bits,
            raw_bits=raw_bits,
            granularity="none",
            channel_dim=int(channel_dim),
            scale=None,
            qmin=None,
            qmax=None,
            original_dtype=x.dtype,
            original_shape=original_shape,
        )
        return QuantizationResult(q_tensor=q_tensor, dequantized=dequantized, meta=meta)

    if mode == QUANT_MODE_FP16:
        q_tensor = x.to(torch.float16)
        dequantized = q_tensor.to(output_dtype)

        meta = QuantizationMeta(
            mode=mode,
            bits=bits,
            raw_bits=raw_bits,
            granularity="none",
            channel_dim=int(channel_dim),
            scale=None,
            qmin=None,
            qmax=None,
            original_dtype=x.dtype,
            original_shape=original_shape,
        )
        return QuantizationResult(q_tensor=q_tensor, dequantized=dequantized, meta=meta)

    if mode in (QUANT_MODE_INT8, QUANT_MODE_INT4):
        q_tensor, scale = symmetric_quantize(
            x=x,
            mode=mode,
            scale=None,
            granularity=granularity,
            channel_dim=channel_dim,
            eps=eps,
        )
        dequantized = symmetric_dequantize(
            q=q_tensor,
            scale=scale,
            output_dtype=output_dtype,
            granularity=granularity,
            channel_dim=channel_dim,
        )

        qmin, qmax, _ = get_symmetric_quant_params(mode)

        meta = QuantizationMeta(
            mode=mode,
            bits=bits,
            raw_bits=raw_bits,
            granularity=granularity,
            channel_dim=int(channel_dim),
            scale=scale.detach(),
            qmin=qmin,
            qmax=qmax,
            original_dtype=x.dtype,
            original_shape=original_shape,
        )
        return QuantizationResult(q_tensor=q_tensor, dequantized=dequantized, meta=meta)

    raise ValueError(f"Unsupported quantization mode: {mode}")


def dequantize_tensor(
    q_tensor: torch.Tensor,
    meta: QuantizationMeta,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Dequantize a tensor according to QuantizationMeta.

    Parameters
    ----------
    q_tensor : torch.Tensor
        Quantized representation.

    meta : QuantizationMeta
        Metadata returned by quantize_tensor().

    output_dtype : torch.dtype, optional
        Output dtype. If None, use meta.original_dtype.

    Returns
    -------
    torch.Tensor
        Dequantized float tensor.
    """
    q_tensor = _require_tensor(q_tensor, "q_tensor")
    output_dtype = _canonical_dtype(output_dtype, meta.original_dtype)

    mode = normalize_quant_mode(meta.mode)

    if mode == QUANT_MODE_FP32:
        return q_tensor.to(output_dtype)

    if mode == QUANT_MODE_FP16:
        return q_tensor.to(output_dtype)

    if mode in (QUANT_MODE_INT8, QUANT_MODE_INT4):
        if meta.scale is None:
            raise ValueError("Integer dequantization requires meta.scale.")

        return symmetric_dequantize(
            q=q_tensor,
            scale=meta.scale,
            output_dtype=output_dtype,
            granularity=meta.granularity,
            channel_dim=meta.channel_dim,
        )

    raise ValueError(f"Unsupported quantization mode: {mode}")


def fake_quantize_tensor(
    x: torch.Tensor,
    mode: str = QUANT_MODE_INT8,
    granularity: str = "per_tensor",
    channel_dim: int = 0,
    output_dtype: Optional[torch.dtype] = None,
    raw_bits: int = DEFAULT_RAW_BITS,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, QuantizationMeta]:
    """
    Quantize then dequantize a tensor.

    This returns only the dequantized tensor and metadata, which is useful
    when the perception model should consume float tensors while still
    experiencing real quantization error.
    """
    result = quantize_tensor(
        x=x,
        mode=mode,
        granularity=granularity,
        channel_dim=channel_dim,
        output_dtype=output_dtype,
        raw_bits=raw_bits,
        eps=eps,
    )
    return result.dequantized, result.meta


def compute_quantization_error(
    original: torch.Tensor,
    recovered: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute quantization error statistics.

    Returns
    -------
    dict
        mse, mae, max_abs_error, mean_abs_original, relative_mae
    """
    original = _require_tensor(original, "original")
    recovered = _require_tensor(recovered, "recovered")

    if tuple(original.shape) != tuple(recovered.shape):
        raise ValueError(
            f"Shape mismatch: original {tuple(original.shape)}, "
            f"recovered {tuple(recovered.shape)}."
        )

    o = original.detach().float()
    r = recovered.detach().float()

    diff = r - o
    abs_diff = diff.abs()

    mse = float((diff * diff).mean().item())
    mae = float(abs_diff.mean().item())
    max_abs_error = float(abs_diff.max().item()) if abs_diff.numel() > 0 else 0.0
    mean_abs_original = float(o.abs().mean().item()) if o.numel() > 0 else 0.0

    relative_mae = mae / (mean_abs_original + 1e-12)

    return {
        "mse": mse,
        "mae": mae,
        "max_abs_error": max_abs_error,
        "mean_abs_original": mean_abs_original,
        "relative_mae": float(relative_mae),
    }


def estimate_tensor_bits(
    x: Union[torch.Tensor, Sequence[int]],
    mode: str,
    raw_bits: int = DEFAULT_RAW_BITS,
) -> int:
    """
    Estimate tensor size in bits under a quantization mode.

    Parameters
    ----------
    x : torch.Tensor or shape
        Tensor or shape sequence.

    mode : str
        Quantization mode.

    raw_bits : int
        Unused for calculation except for validation compatibility.
    """
    _ = _as_positive_int(raw_bits, "raw_bits")
    mode = normalize_quant_mode(mode)
    bits = quant_mode_to_bits(mode)

    if torch.is_tensor(x):
        numel = int(x.numel())
    elif isinstance(x, (list, tuple)):
        numel = 1
        for dim in x:
            numel *= int(dim)
    else:
        raise TypeError(f"x should be tensor or shape, got {type(x)}.")

    return int(numel * bits)


def estimate_tensor_bytes(
    x: Union[torch.Tensor, Sequence[int]],
    mode: str,
    ceil_to_byte: bool = True,
) -> float:
    """
    Estimate tensor size in bytes under a quantization mode.
    """
    bits = estimate_tensor_bits(x, mode=mode)

    if ceil_to_byte:
        return float((bits + 7) // 8)

    return float(bits / 8.0)


def pack_int4_signed(q_int4: torch.Tensor) -> torch.Tensor:
    """
    Pack signed int4 values into bytes.

    Input:
        q_int4:
            torch integer tensor with values in [-7, 7].
            It is typically stored as torch.int8.

    Output:
        packed:
            torch.uint8 tensor. Two int4 values are stored in one byte.

    Encoding:
        signed int4 value v in [-8, 7] is represented as:
            unsigned = v & 0x0F

        first value goes to low nibble.
        second value goes to high nibble.

    Notes
    -----
    This helper is optional. ARCE can perform real INT4 quantization using
    torch.int8 storage and count communication size as 4 bits per value.
    Use packing only when you want an actual byte-level representation.
    """
    q_int4 = _require_tensor(q_int4, "q_int4")

    if q_int4.numel() == 0:
        return torch.empty(0, dtype=torch.uint8, device=q_int4.device)

    q = q_int4.to(torch.int16)

    if int(q.min().item()) < -8 or int(q.max().item()) > 7:
        raise ValueError(
            "q_int4 values should be in signed int4 range [-8, 7]. "
            f"Got min={int(q.min().item())}, max={int(q.max().item())}."
        )

    flat = q.flatten()
    original_numel = flat.numel()

    if original_numel % 2 == 1:
        pad = torch.zeros(1, dtype=flat.dtype, device=flat.device)
        flat = torch.cat([flat, pad], dim=0)

    unsigned = torch.bitwise_and(flat, 0x0F).to(torch.uint8)

    low = unsigned[0::2]
    high = unsigned[1::2] << 4

    packed = torch.bitwise_or(low, high)

    return packed


def unpack_int4_signed(
    packed: torch.Tensor,
    original_numel: Optional[int] = None,
    shape: Optional[Sequence[int]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Unpack signed int4 values from bytes.

    Parameters
    ----------
    packed : torch.Tensor
        torch.uint8 tensor produced by pack_int4_signed().

    original_numel : int, optional
        Number of original int4 values before padding.

    shape : sequence, optional
        If provided, reshape output to this shape.

    device : torch.device, optional
        Output device. If None, use packed.device.

    Returns
    -------
    torch.Tensor
        torch.int8 tensor with signed int4 values.
    """
    packed = _require_tensor(packed, "packed")

    if device is None:
        device = packed.device

    p = packed.to(device=device, dtype=torch.uint8).flatten()

    low = torch.bitwise_and(p, 0x0F)
    high = torch.bitwise_and(p >> 4, 0x0F)

    vals = torch.empty(
        p.numel() * 2,
        dtype=torch.int16,
        device=device,
    )
    vals[0::2] = low.to(torch.int16)
    vals[1::2] = high.to(torch.int16)

    # Convert unsigned 4-bit two's-complement to signed int4.
    vals = torch.where(vals >= 8, vals - 16, vals)

    if original_numel is not None:
        original_numel = int(original_numel)
        vals = vals[:original_numel]

    vals = vals.to(torch.int8)

    if shape is not None:
        shape = tuple(int(v) for v in shape)
        expected = 1
        for dim in shape:
            expected *= dim

        if expected != vals.numel():
            raise ValueError(
                f"shape={shape} requires {expected} elements, "
                f"but unpacked tensor has {vals.numel()}."
            )

        vals = vals.view(*shape)

    return vals


def clone_quantization_meta_with_scale(
    meta: QuantizationMeta,
    scale: Optional[torch.Tensor],
) -> QuantizationMeta:
    """
    Create a copy of QuantizationMeta with a new scale.

    Useful for packet-level FEC / reconstruction when scale must be moved
    to another device or detached explicitly.
    """
    return QuantizationMeta(
        mode=meta.mode,
        bits=meta.bits,
        raw_bits=meta.raw_bits,
        granularity=meta.granularity,
        channel_dim=meta.channel_dim,
        scale=scale,
        qmin=meta.qmin,
        qmax=meta.qmax,
        original_dtype=meta.original_dtype,
        original_shape=meta.original_shape,
    )


__all__ = [
    "get_symmetric_quant_params",
    "compute_symmetric_scale",
    "symmetric_quantize",
    "symmetric_dequantize",
    "fp16_quant_dequant",
    "fp32_passthrough",
    "QuantizationMeta",
    "QuantizationResult",
    "quantize_tensor",
    "dequantize_tensor",
    "fake_quantize_tensor",
    "compute_quantization_error",
    "estimate_tensor_bits",
    "estimate_tensor_bytes",
    "pack_int4_signed",
    "unpack_int4_signed",
    "clone_quantization_meta_with_scale",
]