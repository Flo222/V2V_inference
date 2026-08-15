"""
Channel modeling package for ARCE communication simulation.

This package provides lightweight constants and config utilities for:
    - fixed Good / Medium / Bad channel profiles;
    - Gilbert-Elliott packet loss;
    - latency estimation;
    - unified channel management.

Concrete classes should be imported from their own files, for example:
    from opencood.communication.channel.channel_manager import ChannelManager
    from opencood.communication.channel.gilbert_elliott import GilbertElliott
    from opencood.communication.channel.fixed_channel import FixedChannel
"""

from typing import Any, Dict, Optional


CHANNEL_STATE_GOOD = "good"
CHANNEL_STATE_MEDIUM = "medium"
CHANNEL_STATE_BAD = "bad"

DEFAULT_CHANNEL_STATE = CHANNEL_STATE_MEDIUM

VALID_CHANNEL_STATES = (
    CHANNEL_STATE_GOOD,
    CHANNEL_STATE_MEDIUM,
    CHANNEL_STATE_BAD,
)


CHANNEL_MODE_FIXED = "fixed"
CHANNEL_MODE_MARKOV = "markov"
CHANNEL_MODE_TRACE = "trace"

DEFAULT_CHANNEL_MODE = CHANNEL_MODE_FIXED

VALID_CHANNEL_MODES = (
    CHANNEL_MODE_FIXED,
    CHANNEL_MODE_MARKOV,
    CHANNEL_MODE_TRACE,
)


DEFAULT_GE_PARAMS = {
    CHANNEL_STATE_GOOD: {
        "p_GB": 0.378,
        "p_BG": 0.883,
        "h": 0.905,
        "k": 0.969,
    },
    CHANNEL_STATE_MEDIUM: {
        "p_GB": 0.378,
        "p_BG": 0.883,
        "h": 0.810,
        "k": 0.938,
    },
    CHANNEL_STATE_BAD: {
        "p_GB": 0.417,
        "p_BG": 0.973,
        "h": 0.620,
        "k": 0.948,
    },
}


DEFAULT_BANDWIDTH_MBPS = {
    CHANNEL_STATE_GOOD: 27.0,
    CHANNEL_STATE_MEDIUM: 5.0,
    CHANNEL_STATE_BAD: 1.0,
}


DEFAULT_JITTER_MS = {
    CHANNEL_STATE_GOOD: (2.0, 8.0),
    CHANNEL_STATE_MEDIUM: (5.0, 20.0),
    CHANNEL_STATE_BAD: (10.0, 40.0),
}


DEFAULT_CHANNEL_PROFILES = {
    CHANNEL_STATE_GOOD: {
        "state_name": CHANNEL_STATE_GOOD,
        "bandwidth_mbps": DEFAULT_BANDWIDTH_MBPS[CHANNEL_STATE_GOOD],
        "ge": DEFAULT_GE_PARAMS[CHANNEL_STATE_GOOD],
        "jitter_ms": DEFAULT_JITTER_MS[CHANNEL_STATE_GOOD],
    },
    CHANNEL_STATE_MEDIUM: {
        "state_name": CHANNEL_STATE_MEDIUM,
        "bandwidth_mbps": DEFAULT_BANDWIDTH_MBPS[CHANNEL_STATE_MEDIUM],
        "ge": DEFAULT_GE_PARAMS[CHANNEL_STATE_MEDIUM],
        "jitter_ms": DEFAULT_JITTER_MS[CHANNEL_STATE_MEDIUM],
    },
    CHANNEL_STATE_BAD: {
        "state_name": CHANNEL_STATE_BAD,
        "bandwidth_mbps": DEFAULT_BANDWIDTH_MBPS[CHANNEL_STATE_BAD],
        "ge": DEFAULT_GE_PARAMS[CHANNEL_STATE_BAD],
        "jitter_ms": DEFAULT_JITTER_MS[CHANNEL_STATE_BAD],
    },
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "1", "yes", "y", "on"):
            return True
        if text in ("false", "0", "no", "n", "off"):
            return False

    return bool(value)


def _as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError("%s should be convertible to float, got %s." % (name, value))


def _as_non_negative_float(value: Any, name: str) -> float:
    value = _as_float(value, name)
    if value < 0.0:
        raise ValueError("%s should be non-negative, got %s." % (name, value))
    return value


def _as_probability(value: Any, name: str) -> float:
    value = _as_float(value, name)
    if value < 0.0 or value > 1.0:
        raise ValueError("%s should be in [0, 1], got %s." % (name, value))
    return value


def normalize_channel_state(state: Optional[str] = None) -> str:
    """
    Normalize channel state name.

    Supported:
        good
        medium
        bad
    """
    if state is None:
        return DEFAULT_CHANNEL_STATE

    state = str(state).strip().lower()

    if state not in VALID_CHANNEL_STATES:
        raise ValueError(
            "Unsupported channel state: %s. Supported states: %s."
            % (state, VALID_CHANNEL_STATES)
        )

    return state


def is_valid_channel_state(state: Optional[str]) -> bool:
    try:
        normalize_channel_state(state)
        return True
    except ValueError:
        return False


def normalize_channel_mode(mode: Optional[str] = None) -> str:
    """
    Normalize channel mode name.
    """
    if mode is None:
        return DEFAULT_CHANNEL_MODE

    mode = str(mode).strip().lower()

    if mode not in VALID_CHANNEL_MODES:
        raise ValueError(
            "Unsupported channel mode: %s. Supported modes: %s."
            % (mode, VALID_CHANNEL_MODES)
        )

    return mode


def normalize_ge_params(params: Optional[Dict[str, Any]] = None,
                        state: str = DEFAULT_CHANNEL_STATE) -> Dict[str, float]:
    """
    Normalize Gilbert-Elliott parameters.

    Parameters
    ----------
    params : dict
        Should contain:
            p_GB : probability from Good hidden state to Bad hidden state
            p_BG : probability from Bad hidden state to Good hidden state
            h    : success probability in Good hidden state
            k    : success probability in Bad hidden state

    state : str
        Default profile state used when fields are missing.
    """
    state = normalize_channel_state(state)
    base = dict(DEFAULT_GE_PARAMS[state])

    if params:
        base.update(params)

    return {
        "p_GB": _as_probability(base.get("p_GB", 0.0), "ge.p_GB"),
        "p_BG": _as_probability(base.get("p_BG", 0.0), "ge.p_BG"),
        "h": _as_probability(base.get("h", 1.0), "ge.h"),
        "k": _as_probability(base.get("k", 1.0), "ge.k"),
    }


def ge_expected_loss(params: Optional[Dict[str, Any]] = None,
                     state: str = DEFAULT_CHANNEL_STATE) -> float:
    """
    Compute stationary expected packet loss rate of a GE model.

    h and k are success probabilities, so losses are:
        loss_G = 1 - h
        loss_B = 1 - k

    Stationary hidden-state probabilities:
        pi_G = p_BG / (p_GB + p_BG)
        pi_B = p_GB / (p_GB + p_BG)
    """
    ge = normalize_ge_params(params, state)

    p_gb = ge["p_GB"]
    p_bg = ge["p_BG"]
    h = ge["h"]
    k = ge["k"]

    denom = p_gb + p_bg
    if denom <= 0.0:
        pi_g = 1.0
        pi_b = 0.0
    else:
        pi_g = p_bg / denom
        pi_b = p_gb / denom

    return float(pi_g * (1.0 - h) + pi_b * (1.0 - k))


def normalize_jitter_ms(value: Any = None,
                        state: str = DEFAULT_CHANNEL_STATE):
    """
    Normalize jitter range in milliseconds.

    Returns
    -------
    tuple
        (min_jitter_ms, max_jitter_ms)
    """
    state = normalize_channel_state(state)

    if value is None:
        value = DEFAULT_JITTER_MS[state]

    if isinstance(value, (int, float)):
        v = _as_non_negative_float(value, "jitter_ms")
        return (0.0, v)

    if isinstance(value, str):
        parts = [x.strip() for x in value.split(",") if x.strip()]
        if len(parts) == 1:
            v = _as_non_negative_float(parts[0], "jitter_ms")
            return (0.0, v)
        if len(parts) == 2:
            lo = _as_non_negative_float(parts[0], "jitter_ms[0]")
            hi = _as_non_negative_float(parts[1], "jitter_ms[1]")
            if hi < lo:
                lo, hi = hi, lo
            return (lo, hi)

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return DEFAULT_JITTER_MS[state]
        if len(value) == 1:
            v = _as_non_negative_float(value[0], "jitter_ms[0]")
            return (0.0, v)
        lo = _as_non_negative_float(value[0], "jitter_ms[0]")
        hi = _as_non_negative_float(value[1], "jitter_ms[1]")
        if hi < lo:
            lo, hi = hi, lo
        return (lo, hi)

    raise ValueError("Unsupported jitter_ms value: %s" % repr(value))


def normalize_channel_profile(profile: Optional[Dict[str, Any]] = None,
                              state: str = DEFAULT_CHANNEL_STATE) -> Dict[str, Any]:
    """
    Normalize one channel profile.
    """
    state = normalize_channel_state(state)

    base = {
        "state_name": state,
        "bandwidth_mbps": DEFAULT_BANDWIDTH_MBPS[state],
        "ge": dict(DEFAULT_GE_PARAMS[state]),
        "jitter_ms": DEFAULT_JITTER_MS[state],
    }

    if profile:
        base.update(profile)

    state_name = normalize_channel_state(base.get("state_name", state))

    bandwidth_mbps = _as_non_negative_float(
        base.get("bandwidth_mbps", DEFAULT_BANDWIDTH_MBPS[state_name]),
        "bandwidth_mbps",
    )

    ge = normalize_ge_params(
        base.get("ge", DEFAULT_GE_PARAMS[state_name]),
        state=state_name,
    )

    jitter_ms = normalize_jitter_ms(
        base.get("jitter_ms", DEFAULT_JITTER_MS[state_name]),
        state=state_name,
    )

    return {
        "state_name": state_name,
        "bandwidth_mbps": float(bandwidth_mbps),
        "ge": ge,
        "jitter_ms": jitter_ms,
        "expected_loss": ge_expected_loss(ge, state=state_name),
    }


def extract_channel_cfg(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Accept full ARCE config, full YAML config, or direct channel config.
    """
    cfg = cfg or {}

    if "arce" in cfg and isinstance(cfg["arce"], dict):
        cfg = cfg["arce"]

    if "channel" in cfg and isinstance(cfg["channel"], dict):
        return cfg["channel"]

    return cfg


def normalize_channel_config(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Normalize channel config.

    Supported YAML style:

        arce:
          channel:
            mode: fixed
            fixed_state: medium
            seed: 2026
            profiles:
              good:
                bandwidth_mbps: 27.0
                ge: ...
              medium:
                bandwidth_mbps: 5.0
                ge: ...
              bad:
                bandwidth_mbps: 1.0
                ge: ...
    """
    channel_cfg = extract_channel_cfg(cfg)

    mode = normalize_channel_mode(channel_cfg.get("mode", DEFAULT_CHANNEL_MODE))
    fixed_state = normalize_channel_state(
        channel_cfg.get("fixed_state", channel_cfg.get("state", DEFAULT_CHANNEL_STATE))
    )

    try:
        seed = int(channel_cfg.get("seed", 0))
    except (TypeError, ValueError):
        raise ValueError("channel.seed should be convertible to int.")

    raw_profiles = channel_cfg.get("profiles", {}) or {}

    profiles = {}
    for state in VALID_CHANNEL_STATES:
        raw_profile = raw_profiles.get(state, None)
        profiles[state] = normalize_channel_profile(raw_profile, state=state)

    return {
        "mode": mode,
        "fixed_state": fixed_state,
        "seed": int(seed),
        "profiles": profiles,
    }


def get_channel_config_summary(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return normalize_channel_config(cfg)


__all__ = [
    "CHANNEL_STATE_GOOD",
    "CHANNEL_STATE_MEDIUM",
    "CHANNEL_STATE_BAD",
    "DEFAULT_CHANNEL_STATE",
    "VALID_CHANNEL_STATES",
    "CHANNEL_MODE_FIXED",
    "CHANNEL_MODE_MARKOV",
    "CHANNEL_MODE_TRACE",
    "DEFAULT_CHANNEL_MODE",
    "VALID_CHANNEL_MODES",
    "DEFAULT_GE_PARAMS",
    "DEFAULT_BANDWIDTH_MBPS",
    "DEFAULT_JITTER_MS",
    "DEFAULT_CHANNEL_PROFILES",
    "normalize_channel_state",
    "is_valid_channel_state",
    "normalize_channel_mode",
    "normalize_ge_params",
    "ge_expected_loss",
    "normalize_jitter_ms",
    "normalize_channel_profile",
    "extract_channel_cfg",
    "normalize_channel_config",
    "get_channel_config_summary",
]
