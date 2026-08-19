"""Common utilities for C2MAB-ARCE.

This module only contains stateless constants and pure helper functions.
It is split out from arce_c2mab_comm.py to keep the main controller readable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch


CHANNEL_STATE_ID_TO_NAME = {
    -1: "ego_or_padding",
    0: "good",
    1: "medium",
    2: "bad",
}


DEFAULT_CHANNEL_PROFILES = {
    "good": {
        "state_name": "good",
        "bandwidth_mbps": 27.0,
        "loss_rate": 0.05,
        "plr": 0.05,
        "delay_ms": 10.0,
        "fixed_delay_ms": 10.0,
        "temporal_source": "current",
    },
    "medium": {
        "state_name": "medium",
        "bandwidth_mbps": 5.0,
        "loss_rate": 0.20,
        "plr": 0.20,
        "delay_ms": 50.0,
        "fixed_delay_ms": 50.0,
        "temporal_source": "current",
    },
    "bad": {
        "state_name": "bad",
        "bandwidth_mbps": 1.0,
        "loss_rate": 0.35,
        "plr": 0.35,
        "delay_ms": 100.0,
        "fixed_delay_ms": 100.0,
        "temporal_source": "previous_frame",
    },
}


QUANT_RATIO_TO_FP32 = {
    "fp32": 1.0,
    "fp16": 0.5,
    "int8": 0.25,
    "int4": 0.125,
}


def extract_arce_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = cfg or {}
    if "arce" in cfg and isinstance(cfg["arce"], dict):
        return cfg["arce"]
    return cfg


def as_list_record_len(record_len: Any) -> List[int]:
    if torch.is_tensor(record_len):
        return [int(x) for x in record_len.detach().cpu().flatten().tolist()]
    if isinstance(record_len, (list, tuple)):
        return [int(x) for x in record_len]
    return [int(record_len)]


def safe_get_nested(d: Any, keys: Sequence[str], default: Any = None) -> Any:
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def profile_scalar(value: Any, default: float = 0.0) -> float:
    """Convert scalar/range-style channel profile values to float."""
    if value is None:
        return float(default)

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, (list, tuple)):
        nums = []
        for v in value:
            try:
                nums.append(float(v))
            except Exception:
                pass
        return float(sum(nums) / len(nums)) if nums else float(default)

    if isinstance(value, dict):
        for keys in (
            ("mean",),
            ("value",),
            ("default",),
            ("min", "max"),
            ("low", "high"),
        ):
            vals = []
            ok = True
            for k in keys:
                if k not in value:
                    ok = False
                    break
                try:
                    vals.append(float(value[k]))
                except Exception:
                    ok = False
                    break
            if ok and vals:
                return float(sum(vals) / len(vals))
        return float(default)

    try:
        return float(value)
    except Exception:
        return float(default)


def normalize_state_name(state_name: Any) -> str:
    state_name = str(state_name).strip().lower()
    if state_name == "mid":
        return "medium"
    if state_name in ("good", "medium", "bad", "ego_or_padding"):
        return state_name
    return "medium"


__all__ = [
    "CHANNEL_STATE_ID_TO_NAME",
    "DEFAULT_CHANNEL_PROFILES",
    "QUANT_RATIO_TO_FP32",
    "extract_arce_cfg",
    "as_list_record_len",
    "safe_get_nested",
    "profile_scalar",
    "normalize_state_name",
]
