"""
Channel manager for ARCE communication simulation.

This module provides one unified interface for:

1. Fixed Good / Medium / Bad channel profile.
2. Bernoulli packet loss sampling.
3. Size-bandwidth-plus-jitter latency estimation.

It does NOT:
    - packetize features;
    - perform quantization;
    - perform FEC;
    - reconstruct missing feature patches;
    - modify feature tensors.

Those operations are handled by:
    opencood.communication.transport.packetization.*
    opencood.communication.transport.quantization.*
    opencood.communication.transport.fec.*
    opencood.communication.transport.recovery.*
    opencood.methods.arce.executors.fixed_executor
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Dict, Optional, Tuple, Union

import torch

def _stable_seed(*items):
    s = "|".join(map(str, items))
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)

from opencood.communication.channel import (
    CHANNEL_STATE_GOOD,
    CHANNEL_STATE_MEDIUM,
    CHANNEL_STATE_BAD,
    VALID_CHANNEL_STATES,
    normalize_channel_state,
)

from opencood.communication.channel.fixed_channel import FixedChannel
from opencood.communication.channel.latency_model import LatencyModel


def _extract_seed(cfg: Dict[str, Any]) -> int:
    """
    Extract seed from ARCE config.

    Priority:
        cfg["channel"]["seed"]
        cfg["seed"]
        0
    """
    if "channel" in cfg and isinstance(cfg["channel"], dict):
        if "seed" in cfg["channel"]:
            return int(cfg["channel"]["seed"])

    if "seed" in cfg:
        return int(cfg["seed"])

    return 0


def _extract_channel_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accept either full ARCE config or direct channel config.
    """
    if "channel" in cfg and isinstance(cfg["channel"], dict):
        return cfg["channel"]

    return cfg


class ChannelManager:
    """
    Unified channel manager for ARCE.

    Typical usage in arce_fixed_comm.py:

        channel_manager = ChannelManager(arce_cfg)

        profile = channel_manager.step(
            frame_id=frame_id,
            link_id=(batch_idx, sender_idx)
        )

        loss_mask, loss_info = channel_manager.sample_packet_loss(
            num_packets=num_encoded_packets,
            link_id=(batch_idx, sender_idx),
            device=feature.device,
            frame_id=frame_id
        )

        latency_info = channel_manager.estimate_latency(
            transmitted_bytes=transmitted_bytes,
            link_id=(batch_idx, sender_idx),
            frame_id=frame_id
        )

    Convention:
        loss_mask[i] == True  means packet i is lost.
        loss_mask[i] == False means packet i is received.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        """
        Parameters
        ----------
        cfg : dict
            Can be either:
                1. full ARCE config;
                2. direct channel config.

            Recommended full ARCE YAML style:

            arce:
              seed: 2026

              channel:
                mode: fixed
                fixed_state: medium
                profiles:
                  good: ...
                  medium: ...
                  bad: ...

              latency:
                enabled: true
                deadline_ms: 100.0
                proc_delay_ms: 2.0
        """
        cfg = cfg or {}

        self.full_cfg = cfg
        self.channel_cfg = _extract_channel_cfg(cfg)

        self.seed = _extract_seed(cfg)

        self.mode = str(
            self.channel_cfg.get("mode", "fixed")
        ).strip().lower()

        if self.mode not in ("fixed", "markov"):
            raise NotImplementedError(
                f"ChannelManager supports mode='fixed' or mode='markov', "
                f"got mode='{self.mode}'."
            )

        self.fixed_channel = FixedChannel(self.channel_cfg)
        self.latency_model = LatencyModel(cfg)

        configured_loss_model = str(
            self.channel_cfg.get("loss_model", "bernoulli")
        ).strip().lower()
        if configured_loss_model != "bernoulli":
            raise ValueError(
                "ChannelManager supports only loss_model='bernoulli'; "
                "Gilbert-Elliott support has been removed."
            )
        self.loss_model = "bernoulli"

        self.bernoulli_loss_rates = {
            "good": 0.05,
            "medium": 0.20,
            "bad": 0.35,
        }
        self.bernoulli_loss_rates.update(
            self.channel_cfg.get("bernoulli_loss_rates", {})
        )
        self.latency_model_type = self.channel_cfg.get("latency_model", "size_bandwidth_jitter")

        self.fixed_delay_ms = {
            "good": 10.0,
            "medium": 50.0,
            "bad": 100.0,
        }
        self.fixed_delay_ms.update(
            self.channel_cfg.get("fixed_delay_ms", {})
        )

        # Link-level Good/Medium/Bad evolution belongs here, rather than in
        # individual baseline wrappers.  A state is advanced at most once for
        # one (link, frame) pair so multi-scale messages share one physical
        # channel state and budget context.
        self.markov_states = tuple(VALID_CHANNEL_STATES)
        self.markov_initial_state = normalize_channel_state(
            self.channel_cfg.get(
                "initial_state",
                self.channel_cfg.get("fixed_state", CHANNEL_STATE_MEDIUM),
            )
        )
        self.markov_transition = self._build_markov_transition()
        self._markov_state_by_link: Dict[str, str] = {}
        self._markov_last_frame_by_link: Dict[str, Any] = {}
        self._markov_rng_by_link: Dict[str, torch.Generator] = {}

    def _build_markov_transition(self) -> Dict[str, Tuple[float, ...]]:
        """Normalize a Good/Medium/Bad transition matrix once."""
        default = {
            "good": {"good": 0.85, "medium": 0.13, "bad": 0.02},
            "medium": {"good": 0.10, "medium": 0.80, "bad": 0.10},
            "bad": {"good": 0.03, "medium": 0.17, "bad": 0.80},
        }
        raw = self.channel_cfg.get("transition_matrix", default) or default
        result: Dict[str, Tuple[float, ...]] = {}
        for index, state in enumerate(self.markov_states):
            row = raw.get(state, raw.get(state.upper(), default[state])) \
                if isinstance(raw, dict) else raw[index]
            if isinstance(row, dict):
                values = [float(row.get(dst, 0.0)) for dst in self.markov_states]
            else:
                values = [float(value) for value in row]
            if len(values) != len(self.markov_states) or sum(values) <= 0.0:
                raise ValueError(
                    "Each ChannelManager Markov transition row must provide "
                    "positive probabilities for good, medium and bad."
                )
            total = float(sum(values))
            result[state] = tuple(value / total for value in values)
        return result

    @staticmethod
    def _markov_link_key(link_id: Any) -> str:
        return repr(link_id)

    def _advance_markov_state(self, link_id: Any, frame_id: Optional[Any]) -> str:
        key = self._markov_link_key(link_id)
        if key not in self._markov_state_by_link:
            self._markov_state_by_link[key] = self.markov_initial_state
            self._markov_last_frame_by_link[key] = frame_id
            return self.markov_initial_state
        if frame_id is not None and self._markov_last_frame_by_link.get(key) == frame_id:
            return self._markov_state_by_link[key]
        generator = self._markov_rng_by_link.get(key)
        if generator is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(_stable_seed(self.seed, "channel_state", key))
            self._markov_rng_by_link[key] = generator
        current = self._markov_state_by_link[key]
        probabilities = torch.tensor(
            self.markov_transition[current], dtype=torch.float32
        )
        next_index = int(torch.multinomial(probabilities, 1, generator=generator).item())
        next_state = self.markov_states[next_index]
        self._markov_state_by_link[key] = next_state
        self._markov_last_frame_by_link[key] = frame_id
        return next_state

    def reset(self, link_id: Any = None) -> None:
        """
        Reset latency and Markov state for all links or one link.

        Parameters
        ----------
        link_id : any, optional
            If None, reset all link states.
            If not None, reset only this link's Markov and latency state.
        """
        self.latency_model.reset_rng(link_id=link_id)
        if link_id is None:
            self._markov_state_by_link.clear()
            self._markov_last_frame_by_link.clear()
            self._markov_rng_by_link.clear()
        else:
            key = self._markov_link_key(link_id)
            self._markov_state_by_link.pop(key, None)
            self._markov_last_frame_by_link.pop(key, None)
            self._markov_rng_by_link.pop(key, None)

    def set_fixed_state(self, state: str) -> None:
        """
        Change fixed Good / Medium / Bad channel state at runtime.
        """
        state = normalize_channel_state(state)
        self.fixed_channel.set_fixed_state(state)
        if self.mode == "markov":
            self.markov_initial_state = state
            self.reset()

    def get_current_state(self) -> str:
        """
        Return current fixed channel state.
        """
        if self.mode == "markov":
            return self.markov_initial_state
        return self.fixed_channel.fixed_state

    def get_profile(self, state: Optional[str] = None) -> Dict[str, Any]:
        """
        Get channel profile.

        Parameters
        ----------
        state : str, optional
            good / medium / bad.
            If None, use current fixed state.

        Returns
        -------
        dict
            Channel profile:
                {
                    "state_name": str,
                    "bandwidth_mbps": float,
                    "packet_loss_rate": float,
                    "jitter_ms": tuple
                }
        """
        return self.fixed_channel.get_profile(state)

    def step(
        self,
        frame_id: Optional[int] = None,
        link_id: Any = None,
        state: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return current channel profile for a frame/link.

        For fixed channel mode, this returns the same profile every time.
        The method is kept as a unified interface for future dynamic channels.

        Parameters
        ----------
        frame_id : int, optional
            Current frame index.

        link_id : any, optional
            Link identifier, for example (batch_idx, sender_idx).

        state : str, optional
            Override channel state. If None, use fixed state.

        Returns
        -------
        dict
            Channel profile with metadata.
        """
        if state is None and self.mode == "markov":
            state = self._advance_markov_state(link_id=link_id, frame_id=frame_id)
        profile = self.fixed_channel.step(
            frame_id=frame_id,
            link_id=link_id,
            state=state,
        )

        return profile

    def sample_packet_loss(
        self,
        num_packets: int,
        link_id: Any = None,
        device: Optional[Union[str, torch.device]] = None,
        frame_id: Optional[int] = None,
        state: Optional[str] = None,
        return_info: bool = True,
    ):
        """
        Sample a deterministic Bernoulli packet-loss mask.

        Parameters
        ----------
        num_packets : int
            Number of encoded packets / blocks.

        link_id : any, optional
            Link identifier. Usually:
                (batch_idx, sender_idx)
            or:
                (frame_id, ego_id, sender_id)

        device : str or torch.device, optional
            Device of returned torch.BoolTensor.

        frame_id : int, optional
            Current frame id, stored in returned info.

        state : str, optional
            Override channel state. If None, use fixed state.

        return_info : bool
            If True, return (loss_mask, info).
            If False, return loss_mask only.

        Returns
        -------
        loss_mask : torch.BoolTensor
            Shape [num_packets].
            True means packet lost.
            False means packet received.

        info : dict, optional
            Bernoulli sampling metadata.
        """

        device = device or torch.device("cpu")
        profile = self.step(frame_id=frame_id, link_id=link_id, state=state)
        state_name = profile["state_name"]

        p = float(self.bernoulli_loss_rates.get(
            state_name, profile.get("packet_loss_rate", 0.0)
        ))
        if p < 0.0 or p > 1.0:
            raise ValueError("Bernoulli packet_loss_rate must be in [0, 1].")
        g = torch.Generator(device="cpu")
        g.manual_seed(_stable_seed(self.seed, "bernoulli", link_id, frame_id, state_name))
        loss_mask = (torch.rand((num_packets,), generator=g) < p).to(device=device)
        info = {
            "frame_id": frame_id,
            "link_id": repr(link_id),
            "channel_mode": self.mode,
            "channel_state": state_name,
            "model": "bernoulli",
            "loss_rate": p,
            "bandwidth_mbps": float(profile["bandwidth_mbps"]),
            "jitter_ms_range": tuple(profile["jitter_ms"]),
            "num_packets": int(num_packets),
            "num_lost": int(loss_mask.sum().item()),
            "num_received": int((~loss_mask).sum().item()),
            "empirical_loss": float(
                loss_mask.float().mean().item() if num_packets > 0 else 0.0
            ),
            "expected_loss": p,
        }
        if not return_info:
            return loss_mask
        return loss_mask, info

    def sample_receive_mask(
        self,
        num_packets: int,
        link_id: Any = None,
        device: Optional[Union[str, torch.device]] = None,
        frame_id: Optional[int] = None,
        state: Optional[str] = None,
        return_info: bool = True,
    ):
        """
        Sample receive mask.

        receive_mask[i] == True means packet i is received.
        """
        if return_info:
            loss_mask, info = self.sample_packet_loss(
                num_packets=num_packets,
                link_id=link_id,
                device=device,
                frame_id=frame_id,
                state=state,
                return_info=True,
            )
            receive_mask = ~loss_mask
            return receive_mask, info

        loss_mask = self.sample_packet_loss(
            num_packets=num_packets,
            link_id=link_id,
            device=device,
            frame_id=frame_id,
            state=state,
            return_info=False,
        )
        return ~loss_mask

    def estimate_latency(
        self,
        transmitted_bytes: Any,
        link_id: Any = None,
        frame_id: Optional[int] = None,
        state: Optional[str] = None,
        bandwidth_mbps: Optional[float] = None,
        channel_profile: Optional[Dict[str, Any]] = None,
        use_max_jitter: bool = False,
        use_min_jitter: bool = False,
        deadline_ms: Optional[float] = None,
        proc_delay_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Estimate communication latency for one message.

        Parameters
        ----------
        transmitted_bytes : int or float
            Number of bytes after quantization and redundancy.

        link_id : any, optional
            Link identifier.

        frame_id : int, optional
            Current frame id.

        state : str, optional
            good / medium / bad. If None, use current fixed state.

        bandwidth_mbps : float, optional
            Override bandwidth. If None, use channel profile bandwidth.

        channel_profile : dict, optional
            Precomputed profile. If None, call self.step().

        use_max_jitter : bool
            If True, use jitter upper bound. Useful for feasibility checks.

        use_min_jitter : bool
            If True, use jitter lower bound.

        deadline_ms : float, optional
            Override latency deadline.

        proc_delay_ms : float, optional
            Override processing delay.

        Returns
        -------
        dict
            Latency metadata.
        """
        if channel_profile is None:
            channel_profile = self.step(
                frame_id=frame_id,
                link_id=link_id,
                state=state,
            )
        else:
            channel_profile = copy.deepcopy(channel_profile)
        state_name = channel_profile["state_name"]

        if self.latency_model_type == "fixed_state_delay":
            delay_ms = float(self.fixed_delay_ms[state_name])
            return {
                "model": "fixed_state_delay",
                "state": state_name,
                "num_bytes": int(transmitted_bytes),
                "bandwidth_mbps": float(
                    channel_profile["bandwidth_mbps"]
                    if bandwidth_mbps is None else bandwidth_mbps
                ),
                "transmission_delay_ms": 0.0,
                "processing_delay_ms": 0.0,
                "jitter_ms": 0.0,
                "total_delay_ms": delay_ms,
                "late": False,
            }
            
        if bandwidth_mbps is None:
            bandwidth_mbps = float(channel_profile["bandwidth_mbps"])

        latency_info = self.latency_model.estimate(
            transmitted_bytes=transmitted_bytes,
            bandwidth_mbps=bandwidth_mbps,
            channel_state=state_name,
            link_id=link_id,
            frame_id=frame_id,
            channel_profile=channel_profile,
            use_max_jitter=use_max_jitter,
            use_min_jitter=use_min_jitter,
            deadline_ms=deadline_ms,
            proc_delay_ms=proc_delay_ms,
        )

        latency_info["channel_mode"] = self.mode
        latency_info["channel_state"] = state_name

        return latency_info

    def estimate_latency_from_raw(
        self,
        raw_bytes: Any,
        compression_ratio: Any = 1.0,
        redundancy_ratio: Any = 0.0,
        link_id: Any = None,
        frame_id: Optional[int] = None,
        state: Optional[str] = None,
        use_max_jitter: bool = False,
        deadline_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Estimate latency from raw feature bytes, compression ratio, and redundancy ratio.
        """
        profile = self.step(frame_id=frame_id, link_id=link_id, state=state)

        latency_info = self.latency_model.estimate_from_raw(
            raw_bytes=raw_bytes,
            bandwidth_mbps=profile["bandwidth_mbps"],
            compression_ratio=compression_ratio,
            redundancy_ratio=redundancy_ratio,
            channel_state=profile["state_name"],
            link_id=link_id,
            frame_id=frame_id,
            channel_profile=profile,
            use_max_jitter=use_max_jitter,
        )

        if deadline_ms is not None:
            latency_info = self.latency_model.estimate_from_raw(
                raw_bytes=raw_bytes,
                bandwidth_mbps=profile["bandwidth_mbps"],
                compression_ratio=compression_ratio,
                redundancy_ratio=redundancy_ratio,
                channel_state=profile["state_name"],
                link_id=link_id,
                frame_id=frame_id,
                channel_profile=profile,
                use_max_jitter=use_max_jitter,
            )
            # Re-estimate with deadline override through lower-level API.
            transmitted_bytes = latency_info["transmitted_bytes"]
            latency_info = self.estimate_latency(
                transmitted_bytes=transmitted_bytes,
                link_id=link_id,
                frame_id=frame_id,
                state=profile["state_name"],
                channel_profile=profile,
                use_max_jitter=use_max_jitter,
                deadline_ms=deadline_ms,
            )
            latency_info["raw_bytes"] = float(raw_bytes)
            latency_info["compression_ratio"] = float(compression_ratio)
            latency_info["redundancy_ratio"] = float(redundancy_ratio)

        latency_info["channel_mode"] = self.mode
        latency_info["channel_state"] = profile["state_name"]

        return latency_info

    def is_action_feasible(
        self,
        raw_bytes: Any,
        compression_ratio: Any,
        redundancy_ratio: Any,
        link_id: Any = None,
        frame_id: Optional[int] = None,
        state: Optional[str] = None,
        deadline_ms: Optional[float] = None,
        use_max_jitter: bool = True,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check whether an ARCE action can meet the frame-level deadline.

        This is useful for later fixed-policy validation or C²MAB action filtering.

        Feasibility condition:
            tx_delay + jitter + proc_delay <= deadline
        """
        profile = self.step(frame_id=frame_id, link_id=link_id, state=state)

        feasible, info = self.latency_model.is_action_feasible(
            raw_bytes=raw_bytes,
            compression_ratio=compression_ratio,
            redundancy_ratio=redundancy_ratio,
            bandwidth_mbps=profile["bandwidth_mbps"],
            channel_state=profile["state_name"],
            channel_profile=profile,
            deadline_ms=deadline_ms,
            use_max_jitter=use_max_jitter,
        )

        info["frame_id"] = frame_id
        info["link_id"] = repr(link_id)
        info["channel_mode"] = self.mode
        info["channel_state"] = profile["state_name"]

        return feasible, info

    def get_byte_budget(
        self,
        link_id: Any = None,
        frame_id: Optional[int] = None,
        state: Optional[str] = None,
        deadline_ms: Optional[float] = None,
        use_max_jitter: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute maximum transmitted bytes under current channel and deadline.
        """
        profile = self.step(frame_id=frame_id, link_id=link_id, state=state)

        budget_info = self.latency_model.get_byte_budget(
            bandwidth_mbps=profile["bandwidth_mbps"],
            channel_state=profile["state_name"],
            channel_profile=profile,
            deadline_ms=deadline_ms,
            use_max_jitter=use_max_jitter,
        )

        budget_info["frame_id"] = frame_id
        budget_info["link_id"] = repr(link_id)
        budget_info["channel_mode"] = self.mode
        budget_info["channel_state"] = profile["state_name"]

        return budget_info

    def get_frame_budget(
        self,
        frame_interval_ms: float,
        link_id: Any = None,
        frame_id: Optional[Any] = None,
        state: Optional[str] = None,
        packet_size_bytes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return one link's physical byte budget for a simulation frame.

        This is the shared budget API for baseline-native payload adapters.
        It intentionally knows nothing about feature layout or packet contents;
        adapters decide how to consume the resulting packet-aligned budget.
        """
        interval_ms = float(frame_interval_ms)
        if interval_ms <= 0.0:
            raise ValueError("frame_interval_ms must be positive")
        profile = self.step(frame_id=frame_id, link_id=link_id, state=state)
        raw_budget_bytes = int(
            float(profile["bandwidth_mbps"]) * 1e6 / 8.0 * interval_ms / 1000.0
        )
        packet_size = None if packet_size_bytes is None else int(packet_size_bytes)
        if packet_size is not None and packet_size <= 0:
            raise ValueError("packet_size_bytes must be positive when provided")
        budget_bytes = raw_budget_bytes
        budget_packets = None
        if packet_size is not None:
            budget_packets = max(0, raw_budget_bytes // packet_size)
            budget_bytes = int(budget_packets * packet_size)
        return {
            "frame_id": frame_id,
            "link_id": repr(link_id),
            "channel_mode": self.mode,
            "channel_state": profile["state_name"],
            "profile": profile,
            "bandwidth_mbps": float(profile["bandwidth_mbps"]),
            "frame_interval_ms": interval_ms,
            "raw_budget_bytes": raw_budget_bytes,
            "budget_bytes": budget_bytes,
            "packet_size_bytes": packet_size,
            "budget_packets": budget_packets,
        }

    def sample_link(
        self,
        num_packets: int,
        transmitted_bytes: Any,
        link_id: Any = None,
        frame_id: Optional[int] = None,
        state: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        use_max_jitter: bool = False,
        return_receive_mask: bool = False,
    ):
        """
        Convenience API: sample packet loss and estimate latency together.

        Parameters
        ----------
        num_packets : int
            Number of encoded packets.

        transmitted_bytes : int or float
            Number of transmitted bytes.

        return_receive_mask : bool
            If True, return receive_mask instead of loss_mask.

        Returns
        -------
        mask : torch.BoolTensor
            By default, loss_mask:
                True means lost.
            If return_receive_mask=True:
                True means received.

        info : dict
            Combined channel, Bernoulli-loss, and latency information.
        """
        profile = self.step(frame_id=frame_id, link_id=link_id, state=state)

        loss_mask, loss_info = self.sample_packet_loss(
            num_packets=num_packets,
            link_id=link_id,
            device=device,
            frame_id=frame_id,
            state=profile["state_name"],
            return_info=True,
        )

        latency_info = self.estimate_latency(
            transmitted_bytes=transmitted_bytes,
            link_id=link_id,
            frame_id=frame_id,
            state=profile["state_name"],
            channel_profile=profile,
            use_max_jitter=use_max_jitter,
        )

        info = {
            "frame_id": frame_id,
            "link_id": repr(link_id),
            "channel_mode": self.mode,
            "channel_state": profile["state_name"],
            "bandwidth_mbps": float(profile["bandwidth_mbps"]),
            "jitter_ms_range": tuple(profile["jitter_ms"]),
            "loss": loss_info,
            "latency": latency_info,
            "num_packets": int(num_packets),
            "transmitted_bytes": float(transmitted_bytes),
            "late": bool(latency_info.get("late", False)),
        }

        if return_receive_mask:
            return ~loss_mask, info

        return loss_mask, info

    def get_config(self) -> Dict[str, Any]:
        """
        Export ChannelManager configuration.
        """
        return {
            "mode": self.mode,
            "seed": int(self.seed),
            "fixed_channel": self.fixed_channel.as_dict(),
            "latency": self.latency_model.get_config(),
            "loss_model": self.loss_model,
            "bernoulli_loss_rates": dict(self.bernoulli_loss_rates),
        }

    def __repr__(self) -> str:
        return (
            "ChannelManager("
            f"mode={self.mode}, "
            f"state={self.get_current_state()}, "
            f"seed={self.seed})"
        )
