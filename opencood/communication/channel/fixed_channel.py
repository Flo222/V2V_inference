"""
Fixed Good / Medium / Bad channel profiles for ARCE communication.

This module provides a lightweight fixed channel profile provider.
It does NOT sample packet loss by itself. Packet loss sampling is handled
by gilbert_elliott.py and orchestrated by channel_manager.py.

Expected YAML style:

arce:
  channel:
    mode: fixed
    fixed_state: medium

    profiles:
      good:
        bandwidth_mbps: 27.0
        ge:
          p_GB: 0.378
          p_BG: 0.883
          h: 0.905
          k: 0.969
        jitter_ms: [2.0, 8.0]

      medium:
        bandwidth_mbps: 5.0
        ge:
          p_GB: 0.378
          p_BG: 0.883
          h: 0.810
          k: 0.938
        jitter_ms: [5.0, 20.0]

      bad:
        bandwidth_mbps: 1.0
        ge:
          p_GB: 0.417
          p_BG: 0.973
          h: 0.620
          k: 0.948
        jitter_ms: [10.0, 40.0]
"""

import copy
from typing import Any, Dict, Optional, Tuple

from opencood.communication.channel import (
    CHANNEL_STATE_GOOD,
    CHANNEL_STATE_MEDIUM,
    CHANNEL_STATE_BAD,
    VALID_CHANNEL_STATES,
    DEFAULT_BANDWIDTH_MBPS,
    DEFAULT_JITTER_MS,
    normalize_channel_state,
)


DEFAULT_GE_PROFILES = {
    CHANNEL_STATE_GOOD: {
        # Derived GE-5% profile:
        # keep the GE-10% transition dynamics and scale state loss rates by 0.5
        "p_GB": 0.378,
        "p_BG": 0.883,
        "h": 0.905,
        "k": 0.969,
    },
    CHANNEL_STATE_MEDIUM: {
        # GE-10% profile
        "p_GB": 0.378,
        "p_BG": 0.883,
        "h": 0.810,
        "k": 0.938,
    },
    CHANNEL_STATE_BAD: {
        # GE-15% profile
        "p_GB": 0.417,
        "p_BG": 0.973,
        "h": 0.620,
        "k": 0.948,
    },
}


def _as_float(value: Any, name: str) -> float:
    """
    Convert a value to float and raise a clear error when conversion fails.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} should be convertible to float, got {value}.")


def _validate_probability(value: float, name: str) -> float:
    """
    Validate that a probability-like value is within [0, 1].
    """
    value = _as_float(value, name)

    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} should be in [0, 1], got {value}.")

    return value


def compute_expected_ge_loss(p_gb: float, p_bg: float, h: float, k: float) -> float:
    """
    Compute the stationary expected packet loss rate of a Gilbert-Elliott model.

    Parameters
    ----------
    p_gb : float
        Transition probability from Good state to Bad state.
    p_bg : float
        Transition probability from Bad state to Good state.
    h : float
        Success probability-related parameter in Bad state.
        Bad-state packet loss rate is 1 - h.
    k : float
        Success probability-related parameter in Good state.
        Good-state packet loss rate is 1 - k.

    Returns
    -------
    float
        Stationary expected packet loss rate.
    """
    p_gb = _validate_probability(p_gb, "p_gb")
    p_bg = _validate_probability(p_bg, "p_bg")
    h = _validate_probability(h, "h")
    k = _validate_probability(k, "k")

    denom = p_gb + p_bg

    if denom <= 0.0:
        # Degenerate case: no state transition.
        # If both transition probabilities are 0, the stationary distribution
        # depends on the initial state. For profile validation, returning
        # Good-state loss is the least surprising fallback.
        return 1.0 - k

    pi_good = p_bg / denom
    pi_bad = p_gb / denom

    loss_good = 1.0 - k
    loss_bad = 1.0 - h

    return pi_good * loss_good + pi_bad * loss_bad


def canonicalize_ge_profile(ge_cfg: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """
    Convert a GE profile to canonical keys and validate it.

    Supported input key styles:
        p_GB / p_gb / p
        p_BG / p_bg / r
        h
        k

    Returns canonical keys:
        p_GB, p_BG, h, k, loss_good, loss_bad, expected_loss
    """
    ge_cfg = ge_cfg or {}

    p_gb = ge_cfg.get("p_GB", ge_cfg.get("p_gb", ge_cfg.get("p")))
    p_bg = ge_cfg.get("p_BG", ge_cfg.get("p_bg", ge_cfg.get("r")))
    h = ge_cfg.get("h")
    k = ge_cfg.get("k")

    required = {
        "p_GB": p_gb,
        "p_BG": p_bg,
        "h": h,
        "k": k,
    }

    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Missing GE parameter(s): {missing}.")

    p_gb = _validate_probability(p_gb, "p_GB")
    p_bg = _validate_probability(p_bg, "p_BG")
    h = _validate_probability(h, "h")
    k = _validate_probability(k, "k")

    expected_loss = compute_expected_ge_loss(p_gb, p_bg, h, k)

    profile = {
        "p_GB": p_gb,
        "p_BG": p_bg,
        "h": h,
        "k": k,
        "loss_good": 1.0 - k,
        "loss_bad": 1.0 - h,
        "expected_loss": expected_loss,
    }

    if "expected_loss" in ge_cfg:
        profile["expected_loss_config"] = _validate_probability(
            ge_cfg["expected_loss"],
            "expected_loss",
        )

    return profile


def _normalize_jitter_ms(jitter_ms: Any, state: str) -> Tuple[float, float]:
    """
    Normalize and validate a jitter range.

    Parameters
    ----------
    jitter_ms : list / tuple
        Expected format: [min_ms, max_ms].
    state : str
        Channel state name, used only for error messages.

    Returns
    -------
    tuple
        (min_jitter_ms, max_jitter_ms)
    """
    if jitter_ms is None:
        jitter_ms = DEFAULT_JITTER_MS[state]

    if not isinstance(jitter_ms, (list, tuple)) or len(jitter_ms) != 2:
        raise ValueError(
            f"jitter_ms for state '{state}' should be a list/tuple "
            f"with two values, got {jitter_ms}."
        )

    low = _as_float(jitter_ms[0], f"{state}.jitter_ms[0]")
    high = _as_float(jitter_ms[1], f"{state}.jitter_ms[1]")

    if low < 0.0 or high < 0.0:
        raise ValueError(
            f"jitter_ms for state '{state}' should be non-negative, "
            f"got {jitter_ms}."
        )

    if low > high:
        raise ValueError(
            f"jitter_ms min should not exceed max for state '{state}', "
            f"got {jitter_ms}."
        )

    return low, high


def _normalize_profile_key_map(profiles_cfg: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Normalize profile dictionary keys to lower-case channel states.

    This allows YAML keys such as Good, GOOD, good.
    """
    profiles_cfg = profiles_cfg or {}
    normalized = {}

    for key, value in profiles_cfg.items():
        state = normalize_channel_state(key)
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValueError(
                f"Channel profile for state '{key}' should be a dict, "
                f"got {type(value)}."
            )
        normalized[state] = value

    return normalized


class FixedChannel:
    """
    Fixed Good / Medium / Bad channel profile provider.

    This class returns the same channel profile at every frame.
    It is used for controlled evaluation under fixed Good, Medium, or Bad
    communication conditions.

    It does not:
        - evolve channel state over time;
        - sample packet loss;
        - modify feature tensors.

    Those responsibilities belong to:
        - Markov bandwidth model, if added later;
        - GilbertElliott packet loss model;
        - ARCE communication pipeline.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        """
        Parameters
        ----------
        cfg : dict
            Channel config. It can be either:
                1. arce.channel dictionary;
                2. full arce dictionary containing a "channel" field.
        """
        cfg = self._extract_channel_cfg(cfg or {})

        self.mode = str(cfg.get("mode", "fixed")).strip().lower()
        self.fixed_state = normalize_channel_state(
            cfg.get("fixed_state", CHANNEL_STATE_MEDIUM)
        )

        self.profiles = self._build_profiles(cfg.get("profiles", {}))

    @staticmethod
    def _extract_channel_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accept either full ARCE config or direct channel config.
        """
        if "channel" in cfg and isinstance(cfg["channel"], dict):
            return cfg["channel"]
        return cfg

    def _build_profiles(self, profiles_cfg: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Build validated profiles for good / medium / bad states.
        """
        profiles_cfg = _normalize_profile_key_map(profiles_cfg)
        profiles = {}

        for state in VALID_CHANNEL_STATES:
            user_profile = profiles_cfg.get(state, {})

            bandwidth_mbps = user_profile.get(
                "bandwidth_mbps",
                user_profile.get("bandwidth", DEFAULT_BANDWIDTH_MBPS[state]),
            )
            bandwidth_mbps = _as_float(
                bandwidth_mbps,
                f"{state}.bandwidth_mbps",
            )

            if bandwidth_mbps <= 0.0:
                raise ValueError(
                    f"bandwidth_mbps for state '{state}' should be positive, "
                    f"got {bandwidth_mbps}."
                )

            default_ge = DEFAULT_GE_PROFILES[state]
            user_ge = user_profile.get("ge", {}) or {}

            merged_ge = copy.deepcopy(default_ge)
            merged_ge.update(user_ge)
            ge_profile = canonicalize_ge_profile(merged_ge)

            jitter_ms = user_profile.get(
                "jitter_ms",
                user_profile.get("jitter", DEFAULT_JITTER_MS[state]),
            )
            jitter_ms = _normalize_jitter_ms(jitter_ms, state)

            profiles[state] = {
                "state_name": state,
                "bandwidth_mbps": bandwidth_mbps,
                "ge": ge_profile,
                "jitter_ms": jitter_ms,
            }

        return profiles

    def set_fixed_state(self, state: str) -> None:
        """
        Change the fixed channel state at runtime.

        This is useful for quick channel sweeps without rebuilding the object.
        """
        self.fixed_state = normalize_channel_state(state)

    def get_profile(self, state: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a validated channel profile.

        Parameters
        ----------
        state : str, optional
            If None, use self.fixed_state.

        Returns
        -------
        dict
            Deep-copied channel profile:
                {
                    "state_name": str,
                    "bandwidth_mbps": float,
                    "ge": dict,
                    "jitter_ms": tuple
                }
        """
        if state is None:
            state = self.fixed_state

        state = normalize_channel_state(state)

        if state not in self.profiles:
            raise KeyError(f"Channel profile for state '{state}' does not exist.")

        return copy.deepcopy(self.profiles[state])

    def get_current_profile(self) -> Dict[str, Any]:
        """
        Alias of get_profile(None).
        """
        return self.get_profile(None)

    def get_bandwidth_mbps(self, state: Optional[str] = None) -> float:
        """
        Get bandwidth in Mbps for a channel state.
        """
        return float(self.get_profile(state)["bandwidth_mbps"])

    def get_ge_profile(self, state: Optional[str] = None) -> Dict[str, float]:
        """
        Get GE profile for a channel state.
        """
        return self.get_profile(state)["ge"]

    def get_jitter_range_ms(self, state: Optional[str] = None) -> Tuple[float, float]:
        """
        Get jitter range in milliseconds for a channel state.
        """
        jitter = self.get_profile(state)["jitter_ms"]
        return float(jitter[0]), float(jitter[1])

    def step(
        self,
        frame_id: Optional[int] = None,
        link_id: Optional[Any] = None,
        state: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return the fixed channel profile for current frame / link.

        The method name "step" is kept for compatibility with future
        dynamic channel models. For a fixed channel, it simply returns the
        same profile every time.

        Parameters
        ----------
        frame_id : int, optional
            Current frame index. Only stored in returned metadata.
        link_id : any, optional
            Link identifier, for example (batch_idx, sender_idx).
            Only stored in returned metadata.
        state : str, optional
            Override state for this query. If None, use self.fixed_state.

        Returns
        -------
        dict
            Channel profile with metadata.
        """
        profile = self.get_profile(state)
        profile["mode"] = self.mode
        profile["frame_id"] = frame_id
        profile["link_id"] = link_id

        return profile

    def as_dict(self) -> Dict[str, Any]:
        """
        Export all fixed channel profiles as a dictionary.
        """
        return {
            "mode": self.mode,
            "fixed_state": self.fixed_state,
            "profiles": copy.deepcopy(self.profiles),
        }

    def __repr__(self) -> str:
        return (
            f"FixedChannel(mode={self.mode}, "
            f"fixed_state={self.fixed_state}, "
            f"bandwidth_mbps={self.get_bandwidth_mbps():.3f})"
        )