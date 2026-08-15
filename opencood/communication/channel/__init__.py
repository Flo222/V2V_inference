"""Shared channel constants and normalization helpers.

The project uses Good/Medium/Bad bandwidth profiles with independent
Bernoulli packet loss.  Gilbert-Elliott support is intentionally absent.
"""
from typing import Optional


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
DEFAULT_CHANNEL_MODE = CHANNEL_MODE_FIXED
VALID_CHANNEL_MODES = (CHANNEL_MODE_FIXED, CHANNEL_MODE_MARKOV)

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


def normalize_channel_state(state: Optional[str] = None) -> str:
    state = DEFAULT_CHANNEL_STATE if state is None else str(state).strip().lower()
    if state not in VALID_CHANNEL_STATES:
        raise ValueError(
            "Unsupported channel state: {}. Supported: {}.".format(
                state, VALID_CHANNEL_STATES
            )
        )
    return state


def is_valid_channel_state(state: Optional[str]) -> bool:
    try:
        normalize_channel_state(state)
        return True
    except ValueError:
        return False


def normalize_channel_mode(mode: Optional[str] = None) -> str:
    mode = DEFAULT_CHANNEL_MODE if mode is None else str(mode).strip().lower()
    if mode not in VALID_CHANNEL_MODES:
        raise ValueError(
            "Unsupported channel mode: {}. Supported: {}.".format(
                mode, VALID_CHANNEL_MODES
            )
        )
    return mode


__all__ = [
    "CHANNEL_STATE_GOOD", "CHANNEL_STATE_MEDIUM", "CHANNEL_STATE_BAD",
    "DEFAULT_CHANNEL_STATE", "VALID_CHANNEL_STATES",
    "CHANNEL_MODE_FIXED", "CHANNEL_MODE_MARKOV", "DEFAULT_CHANNEL_MODE",
    "VALID_CHANNEL_MODES", "DEFAULT_BANDWIDTH_MBPS", "DEFAULT_JITTER_MS",
    "normalize_channel_state", "is_valid_channel_state", "normalize_channel_mode",
]
