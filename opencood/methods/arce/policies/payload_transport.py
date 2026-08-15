from __future__ import annotations

import copy
from typing import Any, Dict

TRANSPORT_COMPACT_SPARSE = "compact_sparse"
TRANSPORT_PAYLOAD_NATIVE = "payload_native"

_PAYLOAD_NATIVE_ALIASES = {
    "payload_native",
    "native_payload",
    "payload-native",
    "native-payload",
}


def normalize_transport_mode(cfg_or_mode: Any, default: str = TRANSPORT_COMPACT_SPARSE) -> str:
    if isinstance(cfg_or_mode, dict):
        mode = cfg_or_mode.get("transport_mode", cfg_or_mode.get("payload_mode", default))
    else:
        mode = cfg_or_mode

    mode = str(mode if mode is not None else default).strip().lower()
    if mode in _PAYLOAD_NATIVE_ALIASES:
        return TRANSPORT_PAYLOAD_NATIVE
    if mode in ("compact", "compact_sparse", "mask_native", "where2comm_mask"):
        return TRANSPORT_COMPACT_SPARSE
    return mode or default


def is_payload_native_transport(cfg_or_mode: Any) -> bool:
    return normalize_transport_mode(cfg_or_mode) == TRANSPORT_PAYLOAD_NATIVE


def apply_payload_native_transport_to_arce_cfg(arce_cfg: Any) -> Dict[str, Any]:
    cfg = copy.deepcopy(arce_cfg) if isinstance(arce_cfg, dict) else {}
    mode = normalize_transport_mode(cfg)
    cfg["transport_mode"] = mode

    if mode == TRANSPORT_PAYLOAD_NATIVE:
        compact_cfg = copy.deepcopy(cfg.get("compact_sparse", {}) or {})
        compact_cfg.update(
            {
                "enabled": False,
                "source": "none",
                "budget_aware_topk": False,
            }
        )
        cfg["compact_sparse"] = compact_cfg

    return cfg


def compact_sparse_cfg_for_transport(arce_cfg: Any) -> Dict[str, Any]:
    cfg = arce_cfg if isinstance(arce_cfg, dict) else {}
    if is_payload_native_transport(cfg):
        return {
            "enabled": False,
            "source": "none",
            "budget_aware_topk": False,
            "transport_mode": TRANSPORT_PAYLOAD_NATIVE,
        }
    return copy.deepcopy(cfg.get("compact_sparse", {}) or {})


__all__ = [
    "TRANSPORT_COMPACT_SPARSE",
    "TRANSPORT_PAYLOAD_NATIVE",
    "normalize_transport_mode",
    "is_payload_native_transport",
    "apply_payload_native_transport_to_arce_cfg",
    "compact_sparse_cfg_for_transport",
]
