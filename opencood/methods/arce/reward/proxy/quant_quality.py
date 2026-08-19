from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


DEFAULT_QUANT_QUALITY = {
    "fp32": 1.00,
    "fp16": 0.99,
    "int8": 0.90,
    "int4": 0.65,
    "none": 1.00,
}


def normalize_quant_mode(quant_mode: Any) -> str:
    if quant_mode is None:
        return "none"

    q = str(quant_mode).strip().lower()
    aliases = {
        "": "none",
        "null": "none",
        "no_send": "none",
        "send0": "none",
        "raw": "fp32",
        "float32": "fp32",
        "float": "fp32",
        "float16": "fp16",
        "half": "fp16",
        "fp16_half": "fp16",
        "8bit": "int8",
        "int8_uniform": "int8",
        "4bit": "int4",
        "int4_uniform": "int4",
        "packed_int4": "int4",
        "int4_packed": "int4",
    }
    return aliases.get(q, q)


def _find_prior_cfg(cfg: Any) -> Optional[Mapping[str, Any]]:
    if not isinstance(cfg, dict):
        return None

    if isinstance(cfg.get("quant_quality_prior"), dict):
        return cfg["quant_quality_prior"]

    for key in ("arce", "c2mab", "oracle", "reward"):
        sub = cfg.get(key)
        if isinstance(sub, dict) and isinstance(sub.get("quant_quality_prior"), dict):
            return sub["quant_quality_prior"]

    return None


def merge_quant_quality_prior(cfg: Any = None) -> Dict[str, float]:
    out = dict(DEFAULT_QUANT_QUALITY)
    prior = _find_prior_cfg(cfg)

    if prior:
        for k, v in prior.items():
            out[normalize_quant_mode(k)] = float(v)

    return out


def get_quant_quality(
    quant_mode: Any,
    cfg: Any = None,
    default: Optional[float] = None,
    strict: bool = False,
) -> float:
    table = merge_quant_quality_prior(cfg)
    q = normalize_quant_mode(quant_mode)

    if q == "none":
        return float(table.get("none", 1.0))

    if q in table:
        return float(table[q])

    if strict:
        raise ValueError("Unknown quant_mode: %s" % str(quant_mode))

    if default is not None:
        return float(default)

    return float(table.get("int4", 0.65))


def get_quant_loss(
    quant_mode: Any,
    cfg: Any = None,
    default: Optional[float] = None,
    strict: bool = False,
) -> float:
    return float(
        1.0 - get_quant_quality(
            quant_mode,
            cfg=cfg,
            default=default,
            strict=strict,
        )
    )


__all__ = [
    "DEFAULT_QUANT_QUALITY",
    "normalize_quant_mode",
    "merge_quant_quality_prior",
    "get_quant_quality",
    "get_quant_loss",
]
