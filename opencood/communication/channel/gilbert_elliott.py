"""
Gilbert-Elliott packet loss model for ARCE communication.

This module implements a two-state Markov packet loss model:

    G: Good state, low packet loss rate
    B: Bad state, high packet loss rate

Parameters follow the convention used in our channel configs:

    p_GB: P(next_state = B | current_state = G)
    p_BG: P(next_state = G | current_state = B)
    k: success-related parameter in Good state
       Good-state loss rate = 1 - k
    h: success-related parameter in Bad state
       Bad-state loss rate = 1 - h

The stationary expected loss rate is:

    pi_G = p_BG / (p_GB + p_BG)
    pi_B = p_GB / (p_GB + p_BG)

    expected_loss = pi_G * (1 - k) + pi_B * (1 - h)

Important convention:
    loss_mask[i] == True means packet i is lost.
    loss_mask[i] == False means packet i is received.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch


GE_STATE_GOOD = "G"
GE_STATE_BAD = "B"
VALID_GE_STATES = (GE_STATE_GOOD, GE_STATE_BAD)

# Medium / GE-10% profile as a safe default.
DEFAULT_GE_PROFILE = {
    "p_GB": 0.378,
    "p_BG": 0.883,
    "h": 0.810,
    "k": 0.938,
}


def _as_float(value: Any, name: str) -> float:
    """
    Convert a value to float with a clear error message.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} should be convertible to float, got {value}.")


def _validate_probability(value: Any, name: str) -> float:
    """
    Validate that a probability-like value lies in [0, 1].
    """
    value = _as_float(value, name)

    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} should be in [0, 1], got {value}.")

    return value


def normalize_ge_state(state: Optional[str]) -> str:
    """
    Normalize GE state name.

    Parameters
    ----------
    state : str
        Input state. Supports G / B / good / bad.

    Returns
    -------
    str
        "G" or "B".
    """
    if state is None:
        return GE_STATE_GOOD

    state = str(state).strip()

    if state.upper() == "G" or state.lower() == "good":
        return GE_STATE_GOOD

    if state.upper() == "B" or state.lower() == "bad":
        return GE_STATE_BAD

    raise ValueError(
        f"Invalid Gilbert-Elliott state: {state}. "
        f"Expected one of {VALID_GE_STATES}, good, or bad."
    )


def canonicalize_ge_profile(ge_cfg: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """
    Convert GE config to canonical keys and validate it.

    Supported input styles:
        p_GB / p_gb / p
        p_BG / p_bg / r
        h
        k

    Returns
    -------
    dict
        {
            "p_GB": float,
            "p_BG": float,
            "h": float,
            "k": float,
            "loss_good": float,
            "loss_bad": float,
            "expected_loss": float,
            "pi_good": float,
            "pi_bad": float,
        }
    """
    ge_cfg = ge_cfg or {}

    # Allow nested config style:
    # {"ge": {"p_GB": ..., ...}}
    if "ge" in ge_cfg and isinstance(ge_cfg["ge"], dict):
        ge_cfg = ge_cfg["ge"]

    merged = dict(DEFAULT_GE_PROFILE)
    merged.update(ge_cfg)

    p_gb = merged.get("p_GB", merged.get("p_gb", merged.get("p")))
    p_bg = merged.get("p_BG", merged.get("p_bg", merged.get("r")))
    h = merged.get("h")
    k = merged.get("k")

    missing = []
    if p_gb is None:
        missing.append("p_GB")
    if p_bg is None:
        missing.append("p_BG")
    if h is None:
        missing.append("h")
    if k is None:
        missing.append("k")

    if missing:
        raise ValueError(f"Missing GE parameter(s): {missing}.")

    p_gb = _validate_probability(p_gb, "p_GB")
    p_bg = _validate_probability(p_bg, "p_BG")
    h = _validate_probability(h, "h")
    k = _validate_probability(k, "k")

    loss_good = 1.0 - k
    loss_bad = 1.0 - h

    pi_good, pi_bad = compute_stationary_distribution(p_gb, p_bg)
    expected_loss = pi_good * loss_good + pi_bad * loss_bad

    profile = {
        "p_GB": p_gb,
        "p_BG": p_bg,
        "h": h,
        "k": k,
        "loss_good": loss_good,
        "loss_bad": loss_bad,
        "pi_good": pi_good,
        "pi_bad": pi_bad,
        "expected_loss": expected_loss,
    }

    if "expected_loss" in ge_cfg:
        profile["expected_loss_config"] = _validate_probability(
            ge_cfg["expected_loss"],
            "expected_loss",
        )

    return profile


def compute_stationary_distribution(p_gb: float, p_bg: float) -> Tuple[float, float]:
    """
    Compute stationary distribution of the two-state GE Markov chain.

    Parameters
    ----------
    p_gb : float
        Transition probability from Good to Bad.
    p_bg : float
        Transition probability from Bad to Good.

    Returns
    -------
    tuple
        (pi_good, pi_bad)
    """
    p_gb = _validate_probability(p_gb, "p_GB")
    p_bg = _validate_probability(p_bg, "p_BG")

    denom = p_gb + p_bg

    if denom <= 0.0:
        # Degenerate case: no transition at all.
        # The stationary distribution depends on the initial state.
        # For validation/logging, we use Good-state fallback.
        return 1.0, 0.0

    pi_good = p_bg / denom
    pi_bad = p_gb / denom

    return pi_good, pi_bad


def compute_expected_ge_loss(
    p_gb: float,
    p_bg: float,
    h: float,
    k: float,
) -> float:
    """
    Compute stationary expected packet loss rate.

    Good-state loss = 1 - k
    Bad-state loss  = 1 - h
    """
    p_gb = _validate_probability(p_gb, "p_GB")
    p_bg = _validate_probability(p_bg, "p_BG")
    h = _validate_probability(h, "h")
    k = _validate_probability(k, "k")

    pi_good, pi_bad = compute_stationary_distribution(p_gb, p_bg)

    return pi_good * (1.0 - k) + pi_bad * (1.0 - h)


def _stable_int_seed(base_seed: int, link_id: Any) -> int:
    """
    Create a stable per-link seed.

    Python's built-in hash may vary across processes because of hash
    randomization. Therefore, use md5(repr(link_id)) for reproducibility.
    """
    text = repr(link_id).encode("utf-8")
    digest = hashlib.md5(text).hexdigest()
    link_seed = int(digest[:8], 16)

    return int((int(base_seed) + link_seed) % (2**32 - 1))


class GilbertElliott:
    """
    Gilbert-Elliott two-state packet loss model.

    This model maintains independent state for each link_id by default.
    A link_id can be any hashable or repr-able object, for example:

        link_id = (batch_idx, sender_idx)
        link_id = (frame_id, ego_id, sender_id)

    Recommended usage:

        ge = GilbertElliott(ge_cfg, seed=2026)
        loss_mask, info = ge.sample_loss_mask(
            num_packets=100,
            link_id=(0, 1),
            device=feature.device,
        )

    loss_mask is a torch.BoolTensor:
        True  -> packet lost
        False -> packet received
    """

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        p_gb: Optional[float] = None,
        p_bg: Optional[float] = None,
        h: Optional[float] = None,
        k: Optional[float] = None,
        init_state: str = GE_STATE_GOOD,
        seed: Optional[int] = None,
        transition_before_loss: bool = True,
        per_link_state: bool = True,
        per_link_rng: bool = True,
    ):
        """
        Parameters
        ----------
        cfg : dict, optional
            GE config. Supports p_GB / p_BG / h / k.
            Also supports nested {"ge": {...}}.

        p_gb, p_bg, h, k : float, optional
            Direct GE parameters. These override cfg if provided.

        init_state : str
            Initial state for new links. Supports "G", "B", "good", "bad".

        seed : int, optional
            Random seed for reproducible packet loss sampling.

        transition_before_loss : bool
            If True, update Markov state before sampling loss for each packet.
            If False, sample loss using current state first, then update state.

        per_link_state : bool
            If True, maintain independent GE state for each link_id.
            If False, all links share one global GE state.

        per_link_rng : bool
            If True, each link gets a deterministic RNG derived from seed and link_id.
            If False, all links share one RNG.
        """
        if cfg is None:
            cfg = {}

        if "ge" in cfg and isinstance(cfg["ge"], dict):
            cfg = cfg["ge"]

        cfg = dict(cfg)

        if p_gb is not None:
            cfg["p_GB"] = p_gb
        if p_bg is not None:
            cfg["p_BG"] = p_bg
        if h is not None:
            cfg["h"] = h
        if k is not None:
            cfg["k"] = k

        profile = canonicalize_ge_profile(cfg)

        self.p_gb = profile["p_GB"]
        self.p_bg = profile["p_BG"]
        self.h = profile["h"]
        self.k = profile["k"]

        self.loss_good = profile["loss_good"]
        self.loss_bad = profile["loss_bad"]
        self.pi_good = profile["pi_good"]
        self.pi_bad = profile["pi_bad"]
        self.expected_loss = profile["expected_loss"]

        self.init_state = normalize_ge_state(
            cfg.get("init_state", init_state)
        )

        self.seed = 0 if seed is None else int(seed)
        self.transition_before_loss = bool(transition_before_loss)
        self.per_link_state = bool(per_link_state)
        self.per_link_rng = bool(per_link_rng)

        self._global_link_key = "__global_ge_link__"
        self._states: Dict[Any, str] = {}
        self._rngs: Dict[Any, np.random.Generator] = {}

        self._global_rng = np.random.default_rng(self.seed)

    def _normalize_link_id(self, link_id: Any = None) -> Any:
        """
        Normalize link id depending on whether per-link state is enabled.
        """
        if not self.per_link_state:
            return self._global_link_key

        if link_id is None:
            return self._global_link_key

        return link_id

    def _get_rng(self, link_id: Any = None) -> np.random.Generator:
        """
        Return RNG for a link.
        """
        if not self.per_link_rng:
            return self._global_rng

        key = self._normalize_link_id(link_id)

        if key not in self._rngs:
            seed = _stable_int_seed(self.seed, key)
            self._rngs[key] = np.random.default_rng(seed)

        return self._rngs[key]

    def reset(self, link_id: Any = None, state: Optional[str] = None) -> None:
        """
        Reset GE state.

        Parameters
        ----------
        link_id : any, optional
            If None, reset all known link states and RNGs.
            If not None, reset only that link.

        state : str, optional
            New initial state. If None, use self.init_state.
        """
        state = normalize_ge_state(state or self.init_state)

        if link_id is None:
            self._states.clear()
            self._rngs.clear()
            return

        key = self._normalize_link_id(link_id)
        self._states[key] = state

        if key in self._rngs:
            seed = _stable_int_seed(self.seed, key)
            self._rngs[key] = np.random.default_rng(seed)

    def get_state(self, link_id: Any = None) -> str:
        """
        Get current GE state for a link.
        """
        key = self._normalize_link_id(link_id)

        if key not in self._states:
            self._states[key] = self.init_state

        return self._states[key]

    def set_state(self, state: str, link_id: Any = None) -> None:
        """
        Manually set GE state for a link.
        """
        key = self._normalize_link_id(link_id)
        self._states[key] = normalize_ge_state(state)

    def _transition_state(self, state: str, rng: np.random.Generator) -> str:
        """
        Perform one GE Markov transition.
        """
        state = normalize_ge_state(state)

        u = rng.random()

        if state == GE_STATE_GOOD:
            if u < self.p_gb:
                return GE_STATE_BAD
            return GE_STATE_GOOD

        if state == GE_STATE_BAD:
            if u < self.p_bg:
                return GE_STATE_GOOD
            return GE_STATE_BAD

        raise ValueError(f"Unexpected GE state: {state}")

    def _loss_prob_by_state(self, state: str) -> float:
        """
        Return packet loss probability of a state.
        """
        state = normalize_ge_state(state)

        if state == GE_STATE_GOOD:
            return self.loss_good

        return self.loss_bad

    def packet_loss(self, link_id: Any = None) -> bool:
        """
        Sim2net-style packet loss API.

        Returns
        -------
        bool
            True means packet is lost.
            False means packet is received.
        """
        lost, _ = self.packet_loss_with_info(link_id=link_id)
        return lost

    def packet_loss_with_info(self, link_id: Any = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Sample loss for a single packet and return debug information.

        Returns
        -------
        tuple
            lost : bool
                True means packet is lost.
            info : dict
                Packet-level GE state and probability information.
        """
        key = self._normalize_link_id(link_id)
        rng = self._get_rng(key)

        state_before = self.get_state(key)

        if self.transition_before_loss:
            state_for_loss = self._transition_state(state_before, rng)
            state_after = state_for_loss
        else:
            state_for_loss = state_before
            state_after = self._transition_state(state_before, rng)

        loss_prob = self._loss_prob_by_state(state_for_loss)
        u_loss = rng.random()
        lost = bool(u_loss < loss_prob)

        self._states[key] = state_after

        info = {
            "model": "gilbert_elliott",
            "link_id": repr(key),
            "state_before": state_before,
            "state_for_loss": state_for_loss,
            "state_after": state_after,
            "loss_prob": float(loss_prob),
            "u_loss": float(u_loss),
            "lost": lost,
        }

        return lost, info

    def sample_loss_mask(
        self,
        num_packets: int,
        link_id: Any = None,
        device: Optional[Union[str, torch.device]] = None,
        return_info: bool = True,
    ):
        """
        Sample packet loss mask for multiple packets.

        Parameters
        ----------
        num_packets : int
            Number of packets / encoded blocks.

        link_id : any, optional
            Link identifier. Independent GE state is maintained per link
            when per_link_state=True.

        device : str or torch.device, optional
            Device of returned torch.BoolTensor.

        return_info : bool
            If True, return (loss_mask, info).
            If False, return loss_mask only.

        Returns
        -------
        loss_mask : torch.BoolTensor
            Shape: [num_packets].
            True means packet lost.
            False means packet received.

        info : dict, optional
            Aggregate GE sampling information.
        """
        num_packets = int(num_packets)

        if num_packets < 0:
            raise ValueError(f"num_packets should be non-negative, got {num_packets}.")

        key = self._normalize_link_id(link_id)
        rng = self._get_rng(key)

        initial_state = self.get_state(key)

        loss_array = np.zeros(num_packets, dtype=np.bool_)
        state_for_loss_list = []

        num_good_state = 0
        num_bad_state = 0

        for packet_idx in range(num_packets):
            state_before = self.get_state(key)

            if self.transition_before_loss:
                state_for_loss = self._transition_state(state_before, rng)
                state_after = state_for_loss
            else:
                state_for_loss = state_before
                state_after = self._transition_state(state_before, rng)

            loss_prob = self._loss_prob_by_state(state_for_loss)
            lost = rng.random() < loss_prob

            loss_array[packet_idx] = bool(lost)
            state_for_loss_list.append(state_for_loss)

            if state_for_loss == GE_STATE_GOOD:
                num_good_state += 1
            else:
                num_bad_state += 1

            self._states[key] = state_after

        final_state = self.get_state(key)

        loss_mask = torch.as_tensor(
            loss_array,
            dtype=torch.bool,
            device=device,
        )

        if not return_info:
            return loss_mask

        num_lost = int(loss_array.sum())
        num_received = int(num_packets - num_lost)
        empirical_loss = float(num_lost / num_packets) if num_packets > 0 else 0.0

        info = {
            "model": "gilbert_elliott",
            "link_id": repr(key),
            "num_packets": int(num_packets),
            "num_lost": num_lost,
            "num_received": num_received,
            "empirical_loss": empirical_loss,
            "expected_loss": float(self.expected_loss),
            "state_initial": initial_state,
            "state_final": final_state,
            "num_good_state": int(num_good_state),
            "num_bad_state": int(num_bad_state),
            "loss_good": float(self.loss_good),
            "loss_bad": float(self.loss_bad),
            "p_GB": float(self.p_gb),
            "p_BG": float(self.p_bg),
            "h": float(self.h),
            "k": float(self.k),
            "transition_before_loss": bool(self.transition_before_loss),
        }

        return loss_mask, info

    def sample_receive_mask(
        self,
        num_packets: int,
        link_id: Any = None,
        device: Optional[Union[str, torch.device]] = None,
        return_info: bool = True,
    ):
        """
        Sample receive mask.

        receive_mask[i] == True means packet i is received.

        This is simply the inverse of loss_mask.
        """
        if return_info:
            loss_mask, info = self.sample_loss_mask(
                num_packets=num_packets,
                link_id=link_id,
                device=device,
                return_info=True,
            )
            receive_mask = ~loss_mask
            return receive_mask, info

        loss_mask = self.sample_loss_mask(
            num_packets=num_packets,
            link_id=link_id,
            device=device,
            return_info=False,
        )
        return ~loss_mask

    def get_params(self) -> Dict[str, float]:
        """
        Return GE parameters and derived statistics.
        """
        return {
            "p_GB": float(self.p_gb),
            "p_BG": float(self.p_bg),
            "h": float(self.h),
            "k": float(self.k),
            "loss_good": float(self.loss_good),
            "loss_bad": float(self.loss_bad),
            "pi_good": float(self.pi_good),
            "pi_bad": float(self.pi_bad),
            "expected_loss": float(self.expected_loss),
        }

    def get_state_dict(self) -> Dict[str, str]:
        """
        Return current states of all known links.

        Keys are repr(link_id) strings so they can be JSON-serializable.
        """
        return {repr(k): v for k, v in self._states.items()}

    def __repr__(self) -> str:
        return (
            "GilbertElliott("
            f"p_GB={self.p_gb:.6f}, "
            f"p_BG={self.p_bg:.6f}, "
            f"h={self.h:.6f}, "
            f"k={self.k:.6f}, "
            f"loss_good={self.loss_good:.6f}, "
            f"loss_bad={self.loss_bad:.6f}, "
            f"expected_loss={self.expected_loss:.6f}, "
            f"init_state={self.init_state}, "
            f"seed={self.seed})"
        )