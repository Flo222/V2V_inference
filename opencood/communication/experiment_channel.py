"""Experiment-scoped public communication environment.

Baseline adapters receive this object; they never construct a physical channel
from baseline-local YAML.  The object deliberately contains no payload logic.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from opencood.communication.channel.channel_manager import ChannelManager


# These parameters describe a physical link.  When the experiment-level
# environment is enabled they must not be hidden in a baseline model block,
# otherwise two baselines can silently run under different conditions.
_PRIVATE_PHYSICAL_KEYS = frozenset({
    "transition_matrix",
    "state_profiles",
    "state_params",
    "bandwidth_mbps",
    "packet_loss_rate",
    "packet_loss_mean",
    "packet_loss_std",
    "zero_fraction",
    "delay_ms",
    "delay_mean_ms",
    "delay_std_ms",
    "max_delay_frames",
})


def _find_private_physical_paths(value: Any, path: str = "model.args") -> list:
    """Return baseline-local physical-channel settings below ``model.args``."""
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = "{}.{}".format(path, key)
            if str(key) in _PRIVATE_PHYSICAL_KEYS:
                found.append(child_path)
            found.extend(_find_private_physical_paths(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_find_private_physical_paths(child, "{}[{}]".format(path, index)))
    return found


def validate_experiment_channel_configuration(hypes: Dict[str, Any]) -> None:
    """Fail early if an enabled shared environment still has private physics."""
    cfg = (hypes or {}).get("communication_environment", {}) or {}
    if not bool(cfg.get("enabled", False)) or not bool(cfg.get("strict", True)):
        return
    model_args = ((hypes or {}).get("model", {}) or {}).get("args", {}) or {}
    paths = _find_private_physical_paths(model_args)
    if paths:
        preview = ", ".join(paths[:8])
        if len(paths) > 8:
            preview += ", ..."
        raise ValueError(
            "communication_environment.strict forbids baseline-private "
            "Markov/bandwidth/loss/delay settings; move them to "
            "communication_environment.channel. Found: {}".format(preview)
        )


def build_experiment_channel_manager(hypes: Dict[str, Any]) -> Optional[ChannelManager]:
    cfg = (hypes or {}).get("communication_environment", None)
    if not cfg or not bool(cfg.get("enabled", False)):
        return None
    validate_experiment_channel_configuration(hypes)
    channel = copy.deepcopy(cfg.get("channel", {}))
    if not channel:
        raise ValueError("communication_environment.enabled requires communication_environment.channel")
    return ChannelManager({
        "seed": int(cfg.get("seed", channel.get("seed", 0))),
        "channel": channel,
        "latency": copy.deepcopy(cfg.get("latency", {})),
    })


def inject_experiment_channel_manager(model: Any, manager: Optional[ChannelManager]) -> int:
    """Inject one manager into thin baseline adapters and return their count."""
    if manager is None:
        return 0
    count = 0
    seen = set()
    candidates = list(model.modules())
    for module in list(candidates):
        candidates.extend(value for value in vars(module).values()
                          if value is not None and not isinstance(value, (str, bytes, int, float, bool)))
    for module in candidates:
        if id(module) in seen:
            continue
        seen.add(id(module))
        setter = getattr(module, "set_channel_manager", None)
        if callable(setter):
            setter(manager)
            count += 1
    setattr(model, "experiment_channel_manager", manager)
    setattr(model, "experiment_channel_adapter_count", count)
    return count
