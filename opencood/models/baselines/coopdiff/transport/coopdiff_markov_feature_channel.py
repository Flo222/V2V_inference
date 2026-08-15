# -*- coding: utf-8 -*-
"""Shared-byte-stream Markov feature channel for CoopDiff.

CoopDiff retains its multi-scale feature interface and temporal cache, while
packet framing and byte reconstruction are delegated to ARCE's canonical
``ByteStreamPacketizer``.  There is deliberately no cell-level serializer or
cell-to-packet mapping in this adapter.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from opencood.communication.channel.channel_manager import ChannelManager
from opencood.communication.transport.packetization.byte_stream_packetizer import (
    ByteStreamPacketizer,
)


class CoopDiffMarkovFeatureChannel(nn.Module):
    """ARCE-byte-stream transport adapter for CoopDiff multi-scale features."""

    API_VERSION = 3

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        super().__init__()
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.impair_ego = bool(cfg.get("impair_ego", False))
        self.fps = float(cfg.get("fps", 10.0))
        self.initial_state = str(cfg.get("initial_state", "medium"))
        self.states = list(cfg.get("states", ["good", "medium", "bad"]))
        if not self.states:
            raise ValueError("coopdiff_markov.states must not be empty")
        if self.initial_state not in self.states:
            self.initial_state = self.states[0]

        transition_matrix = cfg.get("transition_matrix", {
            "good": {"good": 0.85, "medium": 0.13, "bad": 0.02},
            "medium": {"good": 0.10, "medium": 0.80, "bad": 0.10},
            "bad": {"good": 0.03, "medium": 0.17, "bad": 0.80},
        })
        state_profiles = cfg.get("state_profiles", {
            "good": {"bandwidth_mbps": 27.0, "packet_loss_rate": 0.05, "delay_ms": 10.0, "temporal_source": "current"},
            "medium": {"bandwidth_mbps": 5.0, "packet_loss_rate": 0.20, "delay_ms": 50.0, "temporal_source": "current"},
            "bad": {"bandwidth_mbps": 1.0, "packet_loss_rate": 0.35, "delay_ms": 100.0, "temporal_source": "previous_frame"},
        })
        self.channel_manager = ChannelManager({
            "seed": int(cfg.get("seed", 0)),
            "channel": {
                "mode": "markov",
                "initial_state": self.initial_state,
                "transition_matrix": transition_matrix,
                "profiles": state_profiles,
                "loss_model": "bernoulli",
                "bernoulli_loss_rates": {
                    state: float(profile.get("packet_loss_rate", 0.0))
                    for state, profile in state_profiles.items()
                },
            },
        })

        packet_cfg = cfg.get("packetization", {}) or {}
        self.byte_packetizer = ByteStreamPacketizer(packet_cfg)
        self.packet_size_bytes = int(self.byte_packetizer.packet_size_bytes)
        self.zero_fill_missing = bool(packet_cfg.get("zero_fill_missing", True))
        if not self.zero_fill_missing:
            raise ValueError(
                "CoopDiff ARCE byte-stream transport requires zero_fill_missing=true."
            )

        active_scales = cfg.get("active_scales", None)
        self.active_scales = None if active_scales is None else {int(x) for x in active_scales}
        self.verbose = bool(cfg.get("verbose", False))
        self._frame_sessions: Dict[str, Dict[str, Any]] = {}
        self._delay_cache = defaultdict(lambda: deque(maxlen=16))
        self._frame_index = -1
        self._current_frame_id = None
        self.latest_info: List[Dict[str, Any]] = []
        self.records: List[Dict[str, Any]] = []

    def set_channel_manager(self, channel_manager: ChannelManager) -> None:
        """Inject the experiment-owned public physical channel without reset."""
        if not isinstance(channel_manager, ChannelManager):
            raise TypeError("channel_manager must be a ChannelManager instance")
        self.channel_manager = channel_manager

    def reset(self, clear_cache: bool = True, clear_records: bool = True) -> None:
        self.channel_manager.reset()
        self._frame_sessions = {}
        self._frame_index = -1
        self._current_frame_id = None
        self.latest_info = []
        if clear_cache:
            self._delay_cache = defaultdict(lambda: deque(maxlen=16))
        if clear_records:
            self.records = []

    def set_channel_state(self, state: str) -> None:
        state = str(state)
        if state not in self.states:
            raise ValueError("Unknown Markov state: {}. Valid: {}".format(state, self.states))
        self.initial_state = state
        self.channel_manager.set_fixed_state(state)

    def get_records(self) -> List[Dict[str, Any]]:
        return self.records

    def get_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "enabled": self.enabled,
            "packetizer": self.byte_packetizer.get_config(),
            "num_records": len(self.records),
            "states": {},
        }
        sum_keys = (
            "source_elements", "message_bytes", "payload_sent_bytes",
            "payload_received_bytes", "offered_packets", "sent_packets",
            "received_packets", "consumed_bytes",
        )
        for record in self.records:
            state = str(record.get("state", "unknown"))
            item = summary["states"].setdefault(state, {"count": 0})
            item["count"] += 1
            for key in sum_keys:
                value = int(record.get(key, 0) or 0)
                item[key] = int(item.get(key, 0)) + value
                summary["total_" + key] = int(summary.get("total_" + key, 0)) + value
        return summary

    def start_frame(self, frame_id: Optional[Any] = None) -> None:
        self._frame_index += 1
        self._frame_sessions = {}
        self.latest_info = []
        self._current_frame_id = frame_id

    def _delay_slots_from_profile(self, profile: Dict[str, Any]) -> int:
        temporal_source = str(profile.get("temporal_source", "current"))
        delay_ms = float(profile.get("delay_ms", 0.0))
        frame_ms = 1000.0 / max(self.fps, 1e-6)
        if temporal_source == "current":
            return 0
        if temporal_source == "previous_frame":
            return max(1, int(round(delay_ms / frame_ms)))
        return max(0, int(delay_ms // frame_ms))

    def _get_or_create_session(self, link_key: str) -> Dict[str, Any]:
        if link_key in self._frame_sessions:
            return self._frame_sessions[link_key]
        budget = self.channel_manager.get_frame_budget(
            frame_interval_ms=1000.0 / max(self.fps, 1e-6),
            link_id=link_key,
            frame_id=self._current_frame_id,
            packet_size_bytes=self.packet_size_bytes,
        )
        profile = budget["profile"]
        session = {
            "state": budget["channel_state"],
            "profile": profile,
            "link_key": link_key,
            "bandwidth_mbps": float(budget["bandwidth_mbps"]),
            "packet_loss_rate": float(profile.get("packet_loss_rate", 0.0)),
            "delay_slots": self._delay_slots_from_profile(profile),
            "initial_budget_bytes": int(budget["budget_bytes"]),
            "remaining_budget_bytes": int(budget["budget_bytes"]),
            "initial_budget_packets": int(budget["budget_packets"] or 0),
            "remaining_budget_packets": int(budget["budget_packets"] or 0),
        }
        self._frame_sessions[link_key] = session
        return session

    def _select_delayed_feature(
        self, link_key: str, scale_idx: int, current: torch.Tensor, delay_slots: int
    ) -> torch.Tensor:
        cache_key = "{}_scale{}".format(link_key, int(scale_idx))
        cache = self._delay_cache[cache_key]
        cache.append(current.detach().clone())
        if delay_slots <= 0:
            return current
        index = len(cache) - 1 - int(delay_slots)
        if index >= 0:
            cached = cache[index].to(device=current.device, dtype=current.dtype)
            if cached.shape == current.shape:
                return cached
        return torch.zeros_like(current)

    def _assemble_byte_stream(
        self,
        outputs: List[torch.Tensor],
        global_index: int,
        link_key: str,
        delay_slots: int,
        active_scale_indices: Sequence[int],
    ) -> Tuple[List[Dict[str, Any]], Any]:
        entries: List[Dict[str, Any]] = []
        chunks: List[torch.Tensor] = []
        byte_offset = 0
        for scale_idx in active_scale_indices:
            delayed = self._select_delayed_feature(
                link_key, scale_idx, outputs[scale_idx][global_index], delay_slots
            )
            byte_chunk = self.byte_packetizer.tensor_to_bytes(delayed)
            entry = {
                "scale_idx": int(scale_idx),
                "shape": tuple(int(v) for v in delayed.shape),
                "dtype": delayed.dtype,
                "num_elements": int(delayed.numel()),
                "byte_offset": int(byte_offset),
                "num_bytes": int(byte_chunk.numel()),
            }
            byte_offset += int(byte_chunk.numel())
            chunks.append(byte_chunk)
            entries.append(entry)
        stream = torch.cat(chunks, dim=0) if chunks else torch.empty(
            0, dtype=torch.uint8, device=outputs[0].device
        )
        return entries, self.byte_packetizer.packetize(
            stream, source_tensor_kind="coopdiff_multiscale_byte_stream"
        )

    def _sample_packet_keep(
        self, offered_packets: int, sent_packets: int, link_key: str, device: torch.device
    ) -> torch.Tensor:
        keep = torch.zeros(offered_packets, dtype=torch.bool, device=device)
        if sent_packets > 0:
            keep[:sent_packets] = self.channel_manager.sample_receive_mask(
                num_packets=sent_packets,
                link_id=link_key,
                frame_id=self._current_frame_id,
                device=device,
                return_info=False,
            )
        return keep

    @staticmethod
    def _overlap_bytes(start: int, end: int, packet_index: int, packet_size: int) -> int:
        return max(0, min(end, (packet_index + 1) * packet_size) - max(start, packet_index * packet_size))

    def _recover_packets(self, packet_result: Any, packet_keep: torch.Tensor) -> torch.Tensor:
        received = torch.zeros_like(packet_result.packets)
        indices = torch.nonzero(packet_keep, as_tuple=False).view(-1)
        if indices.numel() > 0:
            received.index_copy_(0, indices, packet_result.packets.index_select(0, indices))
        return received

    def _process_link(
        self,
        outputs: List[torch.Tensor],
        global_index: int,
        link_key: str,
        session: Dict[str, Any],
        active_scale_indices: Sequence[int],
        num_scales: int,
    ) -> Dict[str, Any]:
        entries, packet_result = self._assemble_byte_stream(
            outputs, global_index, link_key, int(session["delay_slots"]), active_scale_indices
        )
        offered_packets = int(packet_result.num_packets)
        sent_packets = min(offered_packets, int(session["remaining_budget_packets"]))
        consumed_bytes = int(sent_packets * self.packet_size_bytes)
        session["remaining_budget_packets"] = max(0, int(session["remaining_budget_packets"]) - sent_packets)
        session["remaining_budget_bytes"] = int(session["remaining_budget_packets"] * self.packet_size_bytes)

        packet_keep = self._sample_packet_keep(
            offered_packets, sent_packets, link_key, outputs[0].device
        )
        recovered_stream = self.byte_packetizer.unpacketize(
            self._recover_packets(packet_result, packet_keep), packet_result
        )
        received_packets = int(packet_keep[:sent_packets].sum().item()) if sent_packets else 0
        per_scale = []
        source_elements = 0
        payload_sent_bytes = 0
        payload_received_bytes = 0
        for entry in entries:
            start = int(entry["byte_offset"])
            end = start + int(entry["num_bytes"])
            recovered = self.byte_packetizer.bytes_to_tensor(
                recovered_stream[start:end], entry["shape"], entry["dtype"]
            )
            outputs[int(entry["scale_idx"])][global_index] = recovered
            first_packet = start // self.packet_size_bytes if end > start else -1
            last_packet = (end - 1) // self.packet_size_bytes if end > start else -1
            sent_end = min(end, sent_packets * self.packet_size_bytes)
            scale_sent_bytes = max(0, sent_end - start)
            scale_received_bytes = sum(
                self._overlap_bytes(start, end, int(index), self.packet_size_bytes)
                for index in torch.nonzero(packet_keep, as_tuple=False).view(-1).tolist()
            )
            first_sent_packet = max(first_packet, 0)
            last_sent_packet = min(last_packet, sent_packets - 1)
            sent_touched = max(0, last_sent_packet - first_sent_packet + 1)
            received_touched = int(packet_keep[first_sent_packet:first_sent_packet + sent_touched].sum().item()) if sent_touched else 0
            per_scale.append({
                "scale_idx": int(entry["scale_idx"]),
                "shape": list(entry["shape"]),
                "source_elements": int(entry["num_elements"]),
                "message_bytes": int(entry["num_bytes"]),
                "payload_sent_bytes": int(scale_sent_bytes),
                "payload_received_bytes": int(scale_received_bytes),
                "first_packet": int(first_packet),
                "last_packet": int(last_packet),
                "sent_packets_touched": int(sent_touched),
                "received_packets_touched": int(received_touched),
            })
            source_elements += int(entry["num_elements"])
            payload_sent_bytes += int(scale_sent_bytes)
            payload_received_bytes += int(scale_received_bytes)

        return {
            "state": session["state"],
            "bandwidth_mbps": float(session["bandwidth_mbps"]),
            "packet_loss_rate": float(session["packet_loss_rate"]),
            "delay_slots": int(session["delay_slots"]),
            "packetizer": self.byte_packetizer.get_config(),
            "packetizer_mode": "byte_stream",
            "packet_size_bytes": int(self.packet_size_bytes),
            "active_scales": [int(x) for x in active_scale_indices],
            "num_scales": int(num_scales),
            "source_elements": int(source_elements),
            "message_bytes": int(packet_result.original_num_bytes),
            "payload_sent_bytes": int(payload_sent_bytes),
            "payload_received_bytes": int(payload_received_bytes),
            "offered_packets": int(offered_packets),
            "sent_packets": int(sent_packets),
            "received_packets": int(received_packets),
            "budget_dropped_packets": int(max(0, offered_packets - sent_packets)),
            "loss_dropped_packets": int(max(0, sent_packets - received_packets)),
            "consumed_bytes": int(consumed_bytes),
            "wire_bytes": int(consumed_bytes),
            "transmitted_wire_bytes": int(consumed_bytes),
            "received_wire_bytes": int(received_packets * self.packet_size_bytes),
            "tx_bytes": int(consumed_bytes),
            "rx_bytes": int(received_packets * self.packet_size_bytes),
            "initial_budget_bytes": int(session["initial_budget_bytes"]),
            "initial_budget_packets": int(session["initial_budget_packets"]),
            "remaining_budget_bytes_after": int(session["remaining_budget_bytes"]),
            "remaining_budget_packets_after": int(session["remaining_budget_packets"]),
            "per_scale": per_scale,
        }

    def forward_multiscale(
        self,
        feature_list: Sequence[torch.Tensor],
        record_len: torch.Tensor,
        frame_id: Optional[Any] = None,
        scale_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
        if not feature_list or (not self.enabled) or record_len is None:
            return list(feature_list), []
        outputs = [feature.clone() for feature in feature_list]
        num_scales = len(outputs)
        candidates = list(range(num_scales)) if scale_indices is None else [int(x) for x in scale_indices]
        active = [
            index for index in candidates
            if 0 <= index < num_scales and (self.active_scales is None or index in self.active_scales)
        ]
        if not active:
            return outputs, []
        self.start_frame(frame_id=frame_id)
        lengths = [int(value) for value in (
            record_len.detach().cpu().tolist() if torch.is_tensor(record_len) else record_len
        )]
        start = 0
        frame_records = []
        for batch_index, cav_count in enumerate(lengths):
            for local_index in range(cav_count):
                global_index = start + local_index
                if local_index == 0 and not self.impair_ego:
                    continue
                link_key = "b{}_cav{}".format(batch_index, local_index)
                info = {
                    "frame_index": int(self._frame_index), "frame_id": frame_id,
                    "batch": int(batch_index), "cav": int(local_index),
                    "global_idx": int(global_index), "link_key": link_key,
                }
                info.update(self._process_link(
                    outputs, global_index, link_key, self._get_or_create_session(link_key), active, num_scales
                ))
                self.latest_info.append(info)
                self.records.append(info)
                frame_records.append(info)
                if self.verbose:
                    print("[CoopDiff-ARCE-ByteStream] frame={} link={} state={} packets={}/{} recv={} wire={}B".format(
                        self._frame_index, link_key, info["state"], info["sent_packets"],
                        info["offered_packets"], info["received_packets"], info["consumed_bytes"]
                    ))
            start += cav_count
        return outputs, frame_records

    def forward(
        self,
        x: torch.Tensor,
        record_len: torch.Tensor,
        frame_id: Optional[Any] = None,
        scale_idx: int = 0,
        num_scales: int = 1,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        outputs, records = self.forward_multiscale(
            [x], record_len=record_len, frame_id=frame_id, scale_indices=[0]
        )
        for record in records:
            record["logical_scale_idx"] = int(scale_idx)
            record["logical_num_scales"] = int(num_scales)
        return outputs[0], records
