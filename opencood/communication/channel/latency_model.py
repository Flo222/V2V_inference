"""
Latency model for ARCE communication simulation.

This module estimates communication latency as:

    d_total_ms = d_tx_ms + jitter_ms + proc_delay_ms

where:

    d_tx_ms = 8 * transmitted_bytes / (bandwidth_mbps * 1e6) * 1000

Definitions:
    transmitted_bytes:
        Number of bytes after quantization and redundancy coding.

    bandwidth_mbps:
        Current available bandwidth in Mbps.

    jitter_ms:
        Bounded random access / queueing jitter. It captures small stochastic
        variations from wireless medium access, scheduling, queueing, and
        OS / network stack buffering.

    proc_delay_ms:
        Small processing overhead for packetization, coding, decoding,
        reconstruction, and bookkeeping.

Important:
    This module only estimates latency and returns metadata.
    It does not drop packets, does not drop messages, and does not modify
    features by itself.

    The ARCE communication pipeline decides how to handle late messages.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from opencood.communication.channel import (
    CHANNEL_STATE_GOOD,
    CHANNEL_STATE_MEDIUM,
    CHANNEL_STATE_BAD,
    VALID_CHANNEL_STATES,
    DEFAULT_JITTER_MS,
    normalize_channel_state,
)


DEFAULT_DEADLINE_MS = 100.0
DEFAULT_PROC_DELAY_MS = 2.0
DEFAULT_FRAME_INTERVAL_MS = 100.0


def _as_float(value: Any, name: str) -> float:
    """
    Convert a value to float with a clear error message.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} should be convertible to float, got {value}.")


def _as_non_negative_float(value: Any, name: str) -> float:
    """
    Convert a value to float and validate that it is non-negative.
    """
    value = _as_float(value, name)

    if value < 0.0:
        raise ValueError(f"{name} should be non-negative, got {value}.")

    return value


def _stable_int_seed(base_seed: int, link_id: Any) -> int:
    """
    Create a stable deterministic seed from base_seed and link_id.

    Python's built-in hash() is intentionally randomized across processes,
    so use md5(repr(link_id)) for reproducibility.
    """
    text = repr(link_id).encode("utf-8")
    digest = hashlib.md5(text).hexdigest()
    link_seed = int(digest[:8], 16)

    return int((int(base_seed) + link_seed) % (2**32 - 1))


def normalize_jitter_range(jitter_ms: Any, name: str = "jitter_ms") -> Tuple[float, float]:
    """
    Normalize a jitter range.

    Expected format:
        [min_ms, max_ms] or (min_ms, max_ms)

    Returns:
        (min_ms, max_ms)
    """
    if not isinstance(jitter_ms, (list, tuple)) or len(jitter_ms) != 2:
        raise ValueError(
            f"{name} should be a list/tuple with two values, got {jitter_ms}."
        )

    low = _as_non_negative_float(jitter_ms[0], f"{name}[0]")
    high = _as_non_negative_float(jitter_ms[1], f"{name}[1]")

    if low > high:
        raise ValueError(
            f"{name}[0] should not exceed {name}[1], got {jitter_ms}."
        )

    return low, high


def compute_tx_delay_ms(transmitted_bytes: Any, bandwidth_mbps: Any) -> float:
    """
    Compute serialization / transmission delay in milliseconds.

    Formula:
        d_tx_ms = 8 * transmitted_bytes / (bandwidth_mbps * 1e6) * 1000

    Parameters
    ----------
    transmitted_bytes : int or float
        Number of transmitted bytes.

    bandwidth_mbps : int or float
        Available bandwidth in Mbps.

    Returns
    -------
    float
        Transmission delay in milliseconds.
    """
    transmitted_bytes = _as_non_negative_float(
        transmitted_bytes,
        "transmitted_bytes",
    )
    bandwidth_mbps = _as_float(bandwidth_mbps, "bandwidth_mbps")

    if bandwidth_mbps <= 0.0:
        raise ValueError(
            f"bandwidth_mbps should be positive, got {bandwidth_mbps}."
        )

    return 8.0 * transmitted_bytes / (bandwidth_mbps * 1e6) * 1000.0


def compute_transmitted_bytes(
    raw_bytes: Any,
    compression_ratio: Any = 1.0,
    redundancy_ratio: Any = 0.0,
) -> float:
    """
    Compute transmitted bytes after compression and redundancy.

    Formula:
        transmitted_bytes = raw_bytes * compression_ratio * (1 + redundancy_ratio)

    Parameters
    ----------
    raw_bytes : int or float
        Original feature size in bytes.

    compression_ratio : float
        Ratio after quantization / compression.
        Examples:
            FP32 -> 1.0
            FP16 -> 0.5
            INT8 -> 0.25
            INT4 -> 0.125

    redundancy_ratio : float
        Extra redundancy ratio.
        Example:
            0.25 means 25% extra parity / redundant packets.

    Returns
    -------
    float
        Estimated transmitted bytes.
    """
    raw_bytes = _as_non_negative_float(raw_bytes, "raw_bytes")
    compression_ratio = _as_non_negative_float(
        compression_ratio,
        "compression_ratio",
    )
    redundancy_ratio = _as_non_negative_float(
        redundancy_ratio,
        "redundancy_ratio",
    )

    return raw_bytes * compression_ratio * (1.0 + redundancy_ratio)


class LatencyModel:
    """
    Size-bandwidth-plus-jitter latency model.

    It supports fixed Good / Medium / Bad channel profiles:

        good   -> default jitter [2, 8] ms
        medium -> default jitter [5, 20] ms
        bad    -> default jitter [10, 40] ms

    Typical usage:

        latency_model = LatencyModel(arce_cfg)
        info = latency_model.estimate(
            transmitted_bytes=120000,
            bandwidth_mbps=5.0,
            channel_state="medium",
            link_id=(0, 1),
        )

    Returned info includes:
        tx_delay_ms
        jitter_ms
        proc_delay_ms
        total_delay_ms
        deadline_ms
        late
        delay_slots
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        """
        Parameters
        ----------
        cfg : dict, optional
            Can be either:
                1. full arce config containing "latency" and optional "seed";
                2. latency config directly.
        """
        cfg = cfg or {}

        self.full_cfg = cfg
        self.latency_cfg = self._extract_latency_cfg(cfg)

        self.enabled = bool(self.latency_cfg.get("enabled", True))
        self.model = str(
            self.latency_cfg.get("model", "size_bandwidth_plus_jitter")
        ).strip().lower()

        self.deadline_ms = _as_non_negative_float(
            self.latency_cfg.get("deadline_ms", DEFAULT_DEADLINE_MS),
            "deadline_ms",
        )
        self.proc_delay_ms = _as_non_negative_float(
            self.latency_cfg.get("proc_delay_ms", DEFAULT_PROC_DELAY_MS),
            "proc_delay_ms",
        )
        self.frame_interval_ms = _as_non_negative_float(
            self.latency_cfg.get("frame_interval_ms", DEFAULT_FRAME_INTERVAL_MS),
            "frame_interval_ms",
        )

        self.jitter_distribution = str(
            self.latency_cfg.get("jitter_distribution", "uniform")
        ).strip().lower()

        if self.jitter_distribution not in ("uniform", "none", "constant"):
            raise ValueError(
                "jitter_distribution should be one of "
                "uniform / none / constant, got "
                f"{self.jitter_distribution}."
            )

        self.jitter_ms = self._build_jitter_config(
            self.latency_cfg.get("jitter_ms", None)
        )

        self.late_policy = str(
            self.latency_cfg.get("late_policy", "cache_only")
        ).strip().lower()

        self.feasibility_mode = str(
            self.latency_cfg.get("feasibility_mode", "worst_case_jitter")
        ).strip().lower()

        self.seed = int(self._extract_seed(cfg))
        self.per_link_rng = bool(self.latency_cfg.get("per_link_rng", True))

        self._global_rng = np.random.default_rng(self.seed)
        self._rngs: Dict[Any, np.random.Generator] = {}

    @staticmethod
    def _extract_latency_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accept either full ARCE config or direct latency config.
        """
        if "latency" in cfg and isinstance(cfg["latency"], dict):
            return cfg["latency"]

        return cfg

    @staticmethod
    def _extract_seed(cfg: Dict[str, Any]) -> int:
        """
        Extract seed from full ARCE config or latency config.

        Priority:
            cfg["latency"]["seed"]
            cfg["seed"]
            0
        """
        if "latency" in cfg and isinstance(cfg["latency"], dict):
            if "seed" in cfg["latency"]:
                return int(cfg["latency"]["seed"])

        if "seed" in cfg:
            return int(cfg["seed"])

        return 0

    def _build_jitter_config(
        self,
        jitter_cfg: Optional[Any],
    ) -> Dict[str, Tuple[float, float]]:
        """
        Build jitter ranges for good / medium / bad states.

        Supported YAML styles:

        Style 1:
            jitter_ms:
              good: [2.0, 8.0]
              medium: [5.0, 20.0]
              bad: [10.0, 40.0]

        Style 2:
            jitter_ms: [5.0, 20.0]

        Style 3:
            omitted, use defaults.
        """
        result = {}

        if jitter_cfg is None:
            for state in VALID_CHANNEL_STATES:
                result[state] = normalize_jitter_range(
                    DEFAULT_JITTER_MS[state],
                    name=f"default_jitter_ms.{state}",
                )
            return result

        if isinstance(jitter_cfg, (list, tuple)):
            shared_range = normalize_jitter_range(
                jitter_cfg,
                name="jitter_ms",
            )
            for state in VALID_CHANNEL_STATES:
                result[state] = shared_range
            return result

        if not isinstance(jitter_cfg, dict):
            raise ValueError(
                "jitter_ms should be a dict, list, tuple, or None, "
                f"got {type(jitter_cfg)}."
            )

        normalized_cfg = {
            normalize_channel_state(key): value
            for key, value in jitter_cfg.items()
        }

        for state in VALID_CHANNEL_STATES:
            value = normalized_cfg.get(state, DEFAULT_JITTER_MS[state])
            result[state] = normalize_jitter_range(
                value,
                name=f"jitter_ms.{state}",
            )

        return result

    def _get_rng(self, link_id: Any = None) -> np.random.Generator:
        """
        Get RNG for jitter sampling.
        """
        if not self.per_link_rng:
            return self._global_rng

        key = "__global_latency_link__" if link_id is None else link_id

        if key not in self._rngs:
            seed = _stable_int_seed(self.seed, key)
            self._rngs[key] = np.random.default_rng(seed)

        return self._rngs[key]

    def reset_rng(self, link_id: Any = None) -> None:
        """
        Reset jitter RNG.

        If link_id is None, reset all RNGs.
        If link_id is provided, reset only that link's RNG.
        """
        if link_id is None:
            self._rngs.clear()
            self._global_rng = np.random.default_rng(self.seed)
            return

        if link_id in self._rngs:
            seed = _stable_int_seed(self.seed, link_id)
            self._rngs[link_id] = np.random.default_rng(seed)

    def get_jitter_range_ms(
        self,
        channel_state: str = CHANNEL_STATE_MEDIUM,
        channel_profile: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, float]:
        """
        Get jitter range for a channel state.

        If channel_profile has "jitter_ms", use it first.
        Otherwise use self.jitter_ms[state].
        """
        state = normalize_channel_state(channel_state)

        if channel_profile is not None and "jitter_ms" in channel_profile:
            return normalize_jitter_range(
                channel_profile["jitter_ms"],
                name=f"channel_profile.{state}.jitter_ms",
            )

        return self.jitter_ms[state]

    def sample_jitter_ms(
        self,
        channel_state: str = CHANNEL_STATE_MEDIUM,
        link_id: Any = None,
        channel_profile: Optional[Dict[str, Any]] = None,
        use_max_jitter: bool = False,
        use_min_jitter: bool = False,
    ) -> float:
        """
        Sample or select jitter in milliseconds.

        Parameters
        ----------
        use_max_jitter : bool
            If True, return the upper bound. Useful for feasibility checks.

        use_min_jitter : bool
            If True, return the lower bound.

        Returns
        -------
        float
            Jitter in milliseconds.
        """
        low, high = self.get_jitter_range_ms(
            channel_state=channel_state,
            channel_profile=channel_profile,
        )

        if self.jitter_distribution == "none":
            return 0.0

        if use_max_jitter:
            return float(high)

        if use_min_jitter:
            return float(low)

        if self.jitter_distribution == "constant":
            return float((low + high) / 2.0)

        rng = self._get_rng(link_id)
        return float(rng.uniform(low, high))

    def estimate(
        self,
        transmitted_bytes: Any,
        bandwidth_mbps: Any,
        channel_state: str = CHANNEL_STATE_MEDIUM,
        link_id: Any = None,
        frame_id: Optional[int] = None,
        channel_profile: Optional[Dict[str, Any]] = None,
        jitter_ms: Optional[float] = None,
        use_max_jitter: bool = False,
        use_min_jitter: bool = False,
        deadline_ms: Optional[float] = None,
        proc_delay_ms: Optional[float] = None,
        frame_interval_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Estimate total communication delay.

        Parameters
        ----------
        transmitted_bytes : int or float
            Number of bytes to transmit.

        bandwidth_mbps : int or float
            Current bandwidth in Mbps.

        channel_state : str
            good / medium / bad.

        link_id : any
            Link identifier, used only for per-link jitter RNG.

        frame_id : int, optional
            Current frame id, stored in returned metadata.

        channel_profile : dict, optional
            Channel profile from FixedChannel / ChannelManager.
            If it contains jitter_ms, that range has priority.

        jitter_ms : float, optional
            Manually specified jitter. If None, sample from configured range.

        use_max_jitter : bool
            Use jitter upper bound instead of random sampling.

        use_min_jitter : bool
            Use jitter lower bound instead of random sampling.

        deadline_ms : float, optional
            Override default deadline.

        proc_delay_ms : float, optional
            Override default processing delay.

        frame_interval_ms : float, optional
            Override default frame interval.

        Returns
        -------
        dict
            Latency information.
        """
        state = normalize_channel_state(channel_state)

        transmitted_bytes = _as_non_negative_float(
            transmitted_bytes,
            "transmitted_bytes",
        )
        bandwidth_mbps = _as_float(bandwidth_mbps, "bandwidth_mbps")

        if bandwidth_mbps <= 0.0:
            raise ValueError(
                f"bandwidth_mbps should be positive, got {bandwidth_mbps}."
            )

        deadline_ms = self.deadline_ms if deadline_ms is None else (
            _as_non_negative_float(deadline_ms, "deadline_ms")
        )
        proc_delay_ms = self.proc_delay_ms if proc_delay_ms is None else (
            _as_non_negative_float(proc_delay_ms, "proc_delay_ms")
        )
        frame_interval_ms = self.frame_interval_ms if frame_interval_ms is None else (
            _as_non_negative_float(frame_interval_ms, "frame_interval_ms")
        )

        tx_delay_ms = compute_tx_delay_ms(
            transmitted_bytes=transmitted_bytes,
            bandwidth_mbps=bandwidth_mbps,
        )

        if jitter_ms is None:
            jitter_ms = self.sample_jitter_ms(
                channel_state=state,
                link_id=link_id,
                channel_profile=channel_profile,
                use_max_jitter=use_max_jitter,
                use_min_jitter=use_min_jitter,
            )
        else:
            jitter_ms = _as_non_negative_float(jitter_ms, "jitter_ms")

        total_delay_ms = tx_delay_ms + jitter_ms + proc_delay_ms
        late = total_delay_ms > deadline_ms

        if frame_interval_ms <= 0.0:
            delay_slots = 0
        else:
            delay_slots = int(math.ceil(total_delay_ms / frame_interval_ms))

        return {
            "model": self.model,
            "enabled": bool(self.enabled),
            "frame_id": frame_id,
            "link_id": repr(link_id),
            "channel_state": state,
            "transmitted_bytes": float(transmitted_bytes),
            "bandwidth_mbps": float(bandwidth_mbps),
            "tx_delay_ms": float(tx_delay_ms),
            "jitter_ms": float(jitter_ms),
            "proc_delay_ms": float(proc_delay_ms),
            "total_delay_ms": float(total_delay_ms),
            "deadline_ms": float(deadline_ms),
            "late": bool(late),
            "late_policy": self.late_policy,
            "frame_interval_ms": float(frame_interval_ms),
            "delay_slots": int(delay_slots),
            "use_max_jitter": bool(use_max_jitter),
            "use_min_jitter": bool(use_min_jitter),
        }

    def estimate_from_raw(
        self,
        raw_bytes: Any,
        bandwidth_mbps: Any,
        compression_ratio: Any = 1.0,
        redundancy_ratio: Any = 0.0,
        channel_state: str = CHANNEL_STATE_MEDIUM,
        link_id: Any = None,
        frame_id: Optional[int] = None,
        channel_profile: Optional[Dict[str, Any]] = None,
        use_max_jitter: bool = False,
    ) -> Dict[str, Any]:
        """
        Estimate latency from raw feature size, compression ratio, and redundancy ratio.

        This is convenient for ARCE action feasibility checks.
        """
        transmitted_bytes = compute_transmitted_bytes(
            raw_bytes=raw_bytes,
            compression_ratio=compression_ratio,
            redundancy_ratio=redundancy_ratio,
        )

        info = self.estimate(
            transmitted_bytes=transmitted_bytes,
            bandwidth_mbps=bandwidth_mbps,
            channel_state=channel_state,
            link_id=link_id,
            frame_id=frame_id,
            channel_profile=channel_profile,
            use_max_jitter=use_max_jitter,
        )

        info["raw_bytes"] = float(raw_bytes)
        info["compression_ratio"] = float(compression_ratio)
        info["redundancy_ratio"] = float(redundancy_ratio)

        return info

    def is_late(
        self,
        transmitted_bytes: Any,
        bandwidth_mbps: Any,
        channel_state: str = CHANNEL_STATE_MEDIUM,
        link_id: Any = None,
        channel_profile: Optional[Dict[str, Any]] = None,
        deadline_ms: Optional[float] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Return whether a message is late under random jitter sampling.
        """
        info = self.estimate(
            transmitted_bytes=transmitted_bytes,
            bandwidth_mbps=bandwidth_mbps,
            channel_state=channel_state,
            link_id=link_id,
            channel_profile=channel_profile,
            deadline_ms=deadline_ms,
        )

        return bool(info["late"]), info

    def is_feasible(
        self,
        transmitted_bytes: Any,
        bandwidth_mbps: Any,
        channel_state: str = CHANNEL_STATE_MEDIUM,
        channel_profile: Optional[Dict[str, Any]] = None,
        deadline_ms: Optional[float] = None,
        use_max_jitter: bool = True,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check whether a transmitted message can meet the deadline.

        By default, use worst-case jitter. This is safer for action filtering:
        an action should not be considered feasible only because it may get
        lucky with a small random jitter.
        """
        info = self.estimate(
            transmitted_bytes=transmitted_bytes,
            bandwidth_mbps=bandwidth_mbps,
            channel_state=channel_state,
            link_id=None,
            channel_profile=channel_profile,
            use_max_jitter=use_max_jitter,
            deadline_ms=deadline_ms,
        )

        feasible = not bool(info["late"])
        info["feasible"] = feasible

        return feasible, info

    def is_action_feasible(
        self,
        raw_bytes: Any,
        compression_ratio: Any,
        redundancy_ratio: Any,
        bandwidth_mbps: Any,
        channel_state: str = CHANNEL_STATE_MEDIUM,
        channel_profile: Optional[Dict[str, Any]] = None,
        deadline_ms: Optional[float] = None,
        use_max_jitter: bool = True,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check feasibility for an ARCE action.

        Action variables:
            compression_ratio:
                FP32 -> 1.0
                FP16 -> 0.5
                INT8 -> 0.25
                INT4 -> 0.125

            redundancy_ratio:
                0.25 means 25% parity / redundancy overhead.
        """
        transmitted_bytes = compute_transmitted_bytes(
            raw_bytes=raw_bytes,
            compression_ratio=compression_ratio,
            redundancy_ratio=redundancy_ratio,
        )

        feasible, info = self.is_feasible(
            transmitted_bytes=transmitted_bytes,
            bandwidth_mbps=bandwidth_mbps,
            channel_state=channel_state,
            channel_profile=channel_profile,
            deadline_ms=deadline_ms,
            use_max_jitter=use_max_jitter,
        )

        info["raw_bytes"] = float(raw_bytes)
        info["compression_ratio"] = float(compression_ratio)
        info["redundancy_ratio"] = float(redundancy_ratio)

        return feasible, info

    def get_byte_budget(
        self,
        bandwidth_mbps: Any,
        channel_state: str = CHANNEL_STATE_MEDIUM,
        channel_profile: Optional[Dict[str, Any]] = None,
        deadline_ms: Optional[float] = None,
        proc_delay_ms: Optional[float] = None,
        use_max_jitter: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute the maximum transmitted bytes that can fit within the deadline.

        Deadline condition:
            tx_delay_ms + jitter_ms + proc_delay_ms <= deadline_ms

        Therefore:
            tx_budget_ms = deadline_ms - jitter_ms - proc_delay_ms

            byte_budget = tx_budget_ms / 1000 * bandwidth_mbps * 1e6 / 8
        """
        state = normalize_channel_state(channel_state)

        bandwidth_mbps = _as_float(bandwidth_mbps, "bandwidth_mbps")
        if bandwidth_mbps <= 0.0:
            raise ValueError(
                f"bandwidth_mbps should be positive, got {bandwidth_mbps}."
            )

        deadline_ms = self.deadline_ms if deadline_ms is None else (
            _as_non_negative_float(deadline_ms, "deadline_ms")
        )
        proc_delay_ms = self.proc_delay_ms if proc_delay_ms is None else (
            _as_non_negative_float(proc_delay_ms, "proc_delay_ms")
        )

        jitter_ms = self.sample_jitter_ms(
            channel_state=state,
            link_id=None,
            channel_profile=channel_profile,
            use_max_jitter=use_max_jitter,
        )

        tx_budget_ms = deadline_ms - jitter_ms - proc_delay_ms
        tx_budget_ms = max(0.0, tx_budget_ms)

        byte_budget = tx_budget_ms / 1000.0 * bandwidth_mbps * 1e6 / 8.0

        return {
            "channel_state": state,
            "bandwidth_mbps": float(bandwidth_mbps),
            "deadline_ms": float(deadline_ms),
            "jitter_ms": float(jitter_ms),
            "proc_delay_ms": float(proc_delay_ms),
            "tx_budget_ms": float(tx_budget_ms),
            "byte_budget": float(byte_budget),
            "use_max_jitter": bool(use_max_jitter),
        }

    def get_config(self) -> Dict[str, Any]:
        """
        Export latency model configuration.
        """
        return {
            "enabled": bool(self.enabled),
            "model": self.model,
            "deadline_ms": float(self.deadline_ms),
            "proc_delay_ms": float(self.proc_delay_ms),
            "frame_interval_ms": float(self.frame_interval_ms),
            "jitter_distribution": self.jitter_distribution,
            "jitter_ms": {
                state: tuple(value)
                for state, value in self.jitter_ms.items()
            },
            "late_policy": self.late_policy,
            "feasibility_mode": self.feasibility_mode,
            "seed": int(self.seed),
            "per_link_rng": bool(self.per_link_rng),
        }

    def __repr__(self) -> str:
        return (
            "LatencyModel("
            f"enabled={self.enabled}, "
            f"model={self.model}, "
            f"deadline_ms={self.deadline_ms:.3f}, "
            f"proc_delay_ms={self.proc_delay_ms:.3f}, "
            f"frame_interval_ms={self.frame_interval_ms:.3f}, "
            f"jitter_distribution={self.jitter_distribution})"
        )