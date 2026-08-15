"""Fixed Good / Medium / Bad profiles for the public communication channel.

Profiles describe bandwidth, independent Bernoulli packet-loss probability,
and jitter.  Packet sampling itself remains in :class:`ChannelManager`.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

from opencood.communication.channel import (
    CHANNEL_STATE_BAD,
    CHANNEL_STATE_GOOD,
    CHANNEL_STATE_MEDIUM,
    DEFAULT_BANDWIDTH_MBPS,
    DEFAULT_JITTER_MS,
    VALID_CHANNEL_STATES,
    normalize_channel_state,
)


DEFAULT_PACKET_LOSS_RATES = {
    CHANNEL_STATE_GOOD: 0.05,
    CHANNEL_STATE_MEDIUM: 0.20,
    CHANNEL_STATE_BAD: 0.35,
}


def _as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} should be convertible to float, got {value}.")


def _probability(value: Any, name: str) -> float:
    value = _as_float(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} should be in [0, 1], got {value}.")
    return value


def _jitter(value: Any, state: str) -> Tuple[float, float]:
    value = DEFAULT_JITTER_MS[state] if value is None else value
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"jitter_ms for '{state}' must contain [min_ms, max_ms].")
    low = _as_float(value[0], f"{state}.jitter_ms[0]")
    high = _as_float(value[1], f"{state}.jitter_ms[1]")
    if low < 0.0 or high < low:
        raise ValueError(f"Invalid jitter_ms for '{state}': {value}.")
    return low, high


class FixedChannel:
    """Validated Good/Medium/Bad profiles without a GE model."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = self._extract_channel_cfg(cfg or {})
        self.mode = str(cfg.get("mode", "fixed")).strip().lower()
        self.fixed_state = normalize_channel_state(
            cfg.get("fixed_state", CHANNEL_STATE_MEDIUM)
        )
        self.profiles = self._build_profiles(cfg.get("profiles", {}))

    @staticmethod
    def _extract_channel_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
        return cfg["channel"] if isinstance(cfg.get("channel"), dict) else cfg

    def _build_profiles(self, raw_profiles: Any) -> Dict[str, Dict[str, Any]]:
        raw_profiles = raw_profiles or {}
        if not isinstance(raw_profiles, dict):
            raise ValueError("channel.profiles must be a dictionary.")
        normalized = {
            normalize_channel_state(key): value or {}
            for key, value in raw_profiles.items()
        }
        profiles = {}
        for state in VALID_CHANNEL_STATES:
            user = normalized.get(state, {})
            if not isinstance(user, dict):
                raise ValueError(f"Profile '{state}' must be a dictionary.")
            if "ge" in user:
                raise ValueError(
                    "GE profile fields are no longer supported; use "
                    "packet_loss_rate with Bernoulli loss."
                )
            bandwidth = _as_float(
                user.get("bandwidth_mbps", user.get("bandwidth", DEFAULT_BANDWIDTH_MBPS[state])),
                f"{state}.bandwidth_mbps",
            )
            if bandwidth <= 0.0:
                raise ValueError(f"bandwidth_mbps for '{state}' must be positive.")
            loss_rate = _probability(
                user.get("packet_loss_rate", user.get("loss_rate", DEFAULT_PACKET_LOSS_RATES[state])),
                f"{state}.packet_loss_rate",
            )
            profile = copy.deepcopy(user)
            profile.update({
                "state_name": state,
                "bandwidth_mbps": bandwidth,
                "packet_loss_rate": loss_rate,
                "jitter_ms": _jitter(user.get("jitter_ms", user.get("jitter")), state),
            })
            profiles[state] = profile
        return profiles

    def set_fixed_state(self, state: str) -> None:
        self.fixed_state = normalize_channel_state(state)

    def get_profile(self, state: Optional[str] = None) -> Dict[str, Any]:
        state = normalize_channel_state(self.fixed_state if state is None else state)
        return copy.deepcopy(self.profiles[state])

    def get_current_profile(self) -> Dict[str, Any]:
        return self.get_profile()

    def get_bandwidth_mbps(self, state: Optional[str] = None) -> float:
        return float(self.get_profile(state)["bandwidth_mbps"])

    def get_packet_loss_rate(self, state: Optional[str] = None) -> float:
        return float(self.get_profile(state)["packet_loss_rate"])

    def get_jitter_range_ms(self, state: Optional[str] = None) -> Tuple[float, float]:
        jitter = self.get_profile(state)["jitter_ms"]
        return float(jitter[0]), float(jitter[1])

    def step(
        self,
        frame_id: Optional[int] = None,
        link_id: Optional[Any] = None,
        state: Optional[str] = None,
    ) -> Dict[str, Any]:
        profile = self.get_profile(state)
        profile.update({"frame_id": frame_id, "link_id": link_id, "mode": self.mode})
        return profile

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "fixed_state": self.fixed_state,
            "profiles": copy.deepcopy(self.profiles),
        }
