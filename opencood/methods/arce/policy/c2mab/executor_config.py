from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from opencood.methods.arce.transport_policy.payload_transport import (
    apply_payload_native_transport_to_arce_cfg,
)


_DEFAULT_LOSS_RATES = {
    "good": 0.05,
    "medium": 0.20,
    "bad": 0.35,
}

_DEFAULT_DELAY_MS = {
    "good": 10.0,
    "medium": 50.0,
    "bad": 100.0,
}

_DEFAULT_DELAY_POLICY = {
    "good": "current",
    "medium": "current",
    "bad": "previous_frame",
}


def _dict_or_empty(x: Any) -> Dict[str, Any]:
    return copy.deepcopy(x) if isinstance(x, dict) else {}


def build_c2mab_executor_cfg(
    cfg: Optional[Dict[str, Any]],
    arce_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Build ARCEFixedComm config for ARCE-C2MAB runtime execution.

    ARCE-C2MAB makes online communication decisions, while ARCEFixedComm is
    reused as the executor for quantization, packetization, channel loss,
    delay, FEC, redundancy, cache, and feature recovery.

    Defaults here are fallback values only. Existing YAML config values take
    precedence.
    """
    base_cfg = copy.deepcopy(cfg) if isinstance(cfg, dict) else {}

    if isinstance(base_cfg.get("arce", None), dict):
        executor_arce_cfg = copy.deepcopy(base_cfg["arce"])
    else:
        executor_arce_cfg = copy.deepcopy(arce_cfg)

    executor_arce_cfg["mode"] = "fixed"
    executor_arce_cfg["policy"] = "fixed"
    executor_arce_cfg["priority_layout_enabled"] = True
    compact_cfg = _dict_or_empty(executor_arce_cfg.get("compact_sparse", None))
    compact_cfg["priority_layout_enabled"] = True
    executor_arce_cfg["compact_sparse"] = compact_cfg
    executor_arce_cfg = apply_payload_native_transport_to_arce_cfg(executor_arce_cfg)

    quant_cfg = _dict_or_empty(executor_arce_cfg.get("quantization", None))
    quant_cfg.setdefault("mode", "fp32")
    executor_arce_cfg["quantization"] = quant_cfg

    packetizer_cfg = _dict_or_empty(executor_arce_cfg.get("packetizer", None))
    packetizer_cfg["mode"] = "byte_stream"
    packetizer_cfg.setdefault("packet_size_bytes", 1024)
    executor_arce_cfg["packetizer"] = packetizer_cfg

    channel_cfg = _dict_or_empty(executor_arce_cfg.get("channel", None))
    channel_cfg["mode"] = "fixed"
    channel_cfg.setdefault("fixed_state", "medium")
    channel_cfg.setdefault("state_source", "dataset_link_markov_override")
    channel_cfg["loss_model"] = "bernoulli"
    channel_cfg["latency_model"] = "fixed_state_delay"
    channel_cfg["bernoulli_loss_rates"] = {
        **_DEFAULT_LOSS_RATES,
        **_dict_or_empty(channel_cfg.get("bernoulli_loss_rates", None)),
    }
    channel_cfg["fixed_delay_ms"] = {
        **_DEFAULT_DELAY_MS,
        **_dict_or_empty(channel_cfg.get("fixed_delay_ms", None)),
    }
    channel_cfg.setdefault(
        "jitter_ms",
        {
            "good": [0.0, 0.0],
            "medium": [0.0, 0.0],
            "bad": [0.0, 0.0],
        },
    )
    executor_arce_cfg["channel"] = channel_cfg

    delay_cfg = _dict_or_empty(executor_arce_cfg.get("delay", None))
    delay_cfg["policy_by_state"] = {
        **_DEFAULT_DELAY_POLICY,
        **_dict_or_empty(delay_cfg.get("policy_by_state", None)),
    }
    executor_arce_cfg["delay"] = delay_cfg

    fec_cfg = _dict_or_empty(executor_arce_cfg.get("fec", None))
    fec_cfg.setdefault("enabled", True)
    fec_cfg.setdefault("type", "action")
    fec_cfg.setdefault("default_type", "raptor_sim")
    executor_arce_cfg["fec"] = fec_cfg

    redundancy_cfg = _dict_or_empty(executor_arce_cfg.get("redundancy", None))
    redundancy_cfg.setdefault("enabled", True)
    executor_arce_cfg["redundancy"] = redundancy_cfg

    patch_cfg = _dict_or_empty(executor_arce_cfg.get("patch_selection", None))
    patch_cfg["enabled"] = False
    executor_arce_cfg["patch_selection"] = patch_cfg

    base_cfg["arce"] = executor_arce_cfg
    base_cfg["mode"] = "fixed"
    base_cfg["policy"] = "fixed"
    base_cfg["quantization"] = executor_arce_cfg["quantization"]
    base_cfg["packetizer"] = executor_arce_cfg["packetizer"]
    base_cfg["channel"] = executor_arce_cfg["channel"]
    base_cfg["delay"] = executor_arce_cfg["delay"]
    base_cfg["fec"] = executor_arce_cfg["fec"]
    base_cfg["redundancy"] = executor_arce_cfg["redundancy"]

    return base_cfg


__all__ = ["build_c2mab_executor_cfg"]
