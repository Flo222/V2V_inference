"""
Feature compression / quantization package for ARCE communication simulation.

This package contains quantization-side utilities used by the ARCE
communication layer in OPV2V / V2X-ViT experiments.

Planned modules:

1. quant_utils.py
   Low-level quantization utilities:
       - FP16 casting
       - symmetric INT8 quantization
       - symmetric INT4 quantization
       - dequantization helpers
       - quantization error metrics

2. feature_quantizer.py
   High-level feature quantizer:
       - accepts an intermediate BEV feature tensor
       - performs real quantize -> dequantize operations
       - returns quantized representation, recovered float feature,
         and quantization metadata

Design note:
    Keep this __init__.py lightweight.

    Do not eagerly import FeatureQuantizer or quant_utils here during early
    development. Those modules may depend on torch and may still be modified.
    Eager imports here can make the entire opencood.communication.transport.quantization package fail
    when only one submodule has an error.

    Concrete modules should be imported directly where they are used:

        from opencood.communication.transport.quantization.feature_quantizer import FeatureQuantizer

    instead of:

        from opencood.communication.transport.quantization import FeatureQuantizer
"""

QUANT_MODE_FP32 = "fp32"
QUANT_MODE_FLOAT32 = "float32"

QUANT_MODE_FP16 = "fp16"
QUANT_MODE_FLOAT16 = "float16"

QUANT_MODE_INT8 = "int8"
QUANT_MODE_UINT8 = "uint8"

QUANT_MODE_INT4 = "int4"

DEFAULT_QUANT_MODE = QUANT_MODE_FP32
DEFAULT_RAW_BITS = 32

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

VALID_QUANT_MODES = (
    QUANT_MODE_FP32,
    QUANT_MODE_FLOAT32,
    QUANT_MODE_FP16,
    QUANT_MODE_FLOAT16,
    QUANT_MODE_INT8,
    QUANT_MODE_UINT8,
    QUANT_MODE_INT4,
)

INTEGER_QUANT_MODES = (
    QUANT_MODE_INT8,
    QUANT_MODE_UINT8,
    QUANT_MODE_INT4,
)

FLOAT_QUANT_MODES = (
    QUANT_MODE_FP32,
    QUANT_MODE_FLOAT32,
    QUANT_MODE_FP16,
    QUANT_MODE_FLOAT16,
)


def normalize_quant_mode(mode=None):
    """
    Normalize quantization mode.

    Parameters
    ----------
    mode : str or None
        Quantization mode. Supported values:
            fp32 / float32
            fp16 / float16
            int8 / uint8
            int4

    Returns
    -------
    str
        Canonical quantization mode:
            fp32 / fp16 / int8 / int4

    Raises
    ------
    ValueError
        If the quantization mode is unsupported.
    """
    if mode is None:
        return DEFAULT_QUANT_MODE

    mode = str(mode).strip().lower()

    if mode not in VALID_QUANT_MODES:
        raise ValueError(
            f"Unsupported quantization mode: {mode}. "
            f"Supported modes: {VALID_QUANT_MODES}."
        )

    return CANONICAL_QUANT_MODE.get(mode, mode)


def is_valid_quant_mode(mode):
    """
    Check whether a quantization mode is valid.

    Parameters
    ----------
    mode : str
        Input quantization mode.

    Returns
    -------
    bool
        True if valid, otherwise False.
    """
    try:
        normalize_quant_mode(mode)
        return True
    except ValueError:
        return False


def quant_mode_to_bits(mode=None):
    """
    Return bits per feature value for a quantization mode.

    Parameters
    ----------
    mode : str or None
        Quantization mode.

    Returns
    -------
    int
        Bits per value.
    """
    mode = normalize_quant_mode(mode)
    return int(QUANT_MODE_TO_BITS[mode])


def compression_ratio_from_quant_mode(mode=None, raw_bits=DEFAULT_RAW_BITS):
    """
    Compute compression ratio relative to raw FP32 feature transmission.

    Formula:
        compression_ratio = quant_bits / raw_bits

    Examples
    --------
    fp32:
        32 / 32 = 1.0

    fp16:
        16 / 32 = 0.5

    int8:
        8 / 32 = 0.25

    int4:
        4 / 32 = 0.125

    Parameters
    ----------
    mode : str or None
        Quantization mode.

    raw_bits : int
        Raw bits per value. Default is 32.

    Returns
    -------
    float
        Compression ratio.
    """
    raw_bits = int(raw_bits)

    if raw_bits <= 0:
        raise ValueError(f"raw_bits should be positive, got {raw_bits}.")

    quant_bits = quant_mode_to_bits(mode)
    return float(quant_bits / raw_bits)


def is_integer_quant_mode(mode):
    """
    Return whether the quantization mode is integer-based.

    Integer modes:
        int8
        int4

    Parameters
    ----------
    mode : str
        Quantization mode.

    Returns
    -------
    bool
        True if mode is int8 / int4, otherwise False.
    """
    mode = normalize_quant_mode(mode)
    return mode in (QUANT_MODE_INT8, QUANT_MODE_INT4)


def is_float_quant_mode(mode):
    """
    Return whether the quantization mode is float-based.

    Float modes:
        fp32
        fp16

    Parameters
    ----------
    mode : str
        Quantization mode.

    Returns
    -------
    bool
        True if mode is fp32 / fp16, otherwise False.
    """
    mode = normalize_quant_mode(mode)
    return mode in (QUANT_MODE_FP32, QUANT_MODE_FP16)


def get_quant_range(mode):
    """
    Return integer quantization range for a quantization mode.

    Parameters
    ----------
    mode : str
        Quantization mode.

    Returns
    -------
    tuple or None
        For int8:
            (-127, 127)

        For int4:
            (-7, 7)

        For fp32 / fp16:
            None

    Notes
    -----
    We use symmetric signed quantization. INT8 uses [-127, 127] instead of
    [-128, 127] to keep the range symmetric around zero.
    """
    mode = normalize_quant_mode(mode)

    if mode == QUANT_MODE_INT8:
        return -127, 127

    if mode == QUANT_MODE_INT4:
        return -7, 7

    return None


def get_quant_config_summary(mode=None, raw_bits=DEFAULT_RAW_BITS):
    """
    Return a JSON-serializable summary for a quantization mode.

    Parameters
    ----------
    mode : str or None
        Quantization mode.

    raw_bits : int
        Raw bits per feature value.

    Returns
    -------
    dict
        Quantization configuration summary.
    """
    mode = normalize_quant_mode(mode)
    quant_bits = quant_mode_to_bits(mode)
    quant_range = get_quant_range(mode)

    return {
        "quant_mode": mode,
        "raw_bits": int(raw_bits),
        "quant_bits": int(quant_bits),
        "compression_ratio": float(
            compression_ratio_from_quant_mode(mode, raw_bits=raw_bits)
        ),
        "is_integer": bool(is_integer_quant_mode(mode)),
        "is_float": bool(is_float_quant_mode(mode)),
        "quant_range": quant_range,
    }


__all__ = [
    "QUANT_MODE_FP32",
    "QUANT_MODE_FLOAT32",
    "QUANT_MODE_FP16",
    "QUANT_MODE_FLOAT16",
    "QUANT_MODE_INT8",
    "QUANT_MODE_UINT8",
    "QUANT_MODE_INT4",
    "DEFAULT_QUANT_MODE",
    "DEFAULT_RAW_BITS",
    "QUANT_MODE_TO_BITS",
    "CANONICAL_QUANT_MODE",
    "VALID_QUANT_MODES",
    "INTEGER_QUANT_MODES",
    "FLOAT_QUANT_MODES",
    "normalize_quant_mode",
    "is_valid_quant_mode",
    "quant_mode_to_bits",
    "compression_ratio_from_quant_mode",
    "is_integer_quant_mode",
    "is_float_quant_mode",
    "get_quant_range",
    "get_quant_config_summary",
]