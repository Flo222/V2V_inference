# -*- coding: utf-8 -*-
"""
Packet-faithful Markov feature channel for CoopDiff.

The channel is inserted immediately before CoopDiff's AttFusion and only
impairs non-ego messages by default.

Physical model
--------------
* One Markov state is maintained for every ego <- collaborator link.
* All enabled CoopDiff scales of one link are serialized into ONE continuous
  virtual byte stream in the current frame.
* The stream is cut into fixed-size packets. The per-frame bandwidth budget is
  enforced in packets, then Bernoulli loss is sampled independently per sent
  packet.
* Received packet bytes are mapped back to feature values; unavailable values
  are zero-filled.
* Delay is applied before serialization using a per-link, per-scale cache.

No real ``uint8`` buffer is materialized. Packet membership is calculated from
byte offsets, which is equivalent for fixed-width feature values and avoids a
large temporary byte tensor.
"""

from __future__ import annotations

from collections import defaultdict, deque
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class CoopDiffMarkovFeatureChannel(nn.Module):
    """Packet-faithful multi-scale channel. API version 2."""

    API_VERSION = 2
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        super(CoopDiffMarkovFeatureChannel, self).__init__()
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

        self.transition_matrix = cfg.get(
            "transition_matrix",
            {
                "good": {"good": 0.85, "medium": 0.13, "bad": 0.02},
                "medium": {"good": 0.10, "medium": 0.80, "bad": 0.10},
                "bad": {"good": 0.03, "medium": 0.17, "bad": 0.80},
            },
        )
        self.state_profiles = cfg.get(
            "state_profiles",
            {
                "good": {
                    "bandwidth_mbps": 27.0,
                    "packet_loss_rate": 0.05,
                    "delay_ms": 10.0,
                    "temporal_source": "current",
                },
                "medium": {
                    "bandwidth_mbps": 5.0,
                    "packet_loss_rate": 0.20,
                    "delay_ms": 50.0,
                    "temporal_source": "current",
                },
                "bad": {
                    "bandwidth_mbps": 1.0,
                    "packet_loss_rate": 0.35,
                    "delay_ms": 100.0,
                    "temporal_source": "previous_frame",
                },
            },
        )

        packet_cfg = cfg.get("packetization", {})
        self.packet_size_bytes = int(packet_cfg.get("packet_size_bytes", 1024))
        self.bytes_per_value = int(packet_cfg.get("bytes_per_value", 4))
        self.zero_fill_missing = bool(packet_cfg.get("zero_fill_missing", True))
        self.selection_policy = str(packet_cfg.get("selection_policy", "raster")).lower()
        self.serialization_order = str(packet_cfg.get("serialization_order", "cell_major")).lower()
        self.send_nonzero_only = bool(packet_cfg.get("send_nonzero_only", False))
        self.nonzero_epsilon = float(packet_cfg.get("nonzero_epsilon", 0.0))

        if self.packet_size_bytes <= 0:
            raise ValueError("packet_size_bytes must be positive")
        if self.bytes_per_value <= 0:
            raise ValueError("bytes_per_value must be positive")
        if self.selection_policy not in ("raster", "magnitude"):
            raise ValueError("selection_policy must be raster or magnitude")
        if self.serialization_order != "cell_major":
            raise ValueError("Only serialization_order=cell_major is supported")

        active_scales = cfg.get("active_scales", None)
        self.active_scales = None if active_scales is None else {int(x) for x in active_scales}
        self.verbose = bool(cfg.get("verbose", False))

        self._link_state: Dict[str, str] = {}
        self._link_initialized = set()
        self._frame_sessions: Dict[str, Dict[str, Any]] = {}
        self._delay_cache = defaultdict(lambda: deque(maxlen=16))
        self._frame_index = -1
        self._current_frame_id = None
        self.latest_info: List[Dict[str, Any]] = []
        self.records: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Runtime control / logging API
    # ------------------------------------------------------------------
    def reset(self, clear_cache: bool = True, clear_records: bool = True) -> None:
        self._link_state = {}
        self._link_initialized = set()
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
        self._link_state = {}
        self._link_initialized = set()

    def get_records(self) -> List[Dict[str, Any]]:
        return self.records

    def get_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "enabled": self.enabled,
            "packet_size_bytes": self.packet_size_bytes,
            "bytes_per_value": self.bytes_per_value,
            "num_records": len(self.records),
            "states": {},
            "total_selected_units": 0,
            "total_sent_units": 0,
            "total_received_units": 0,
            "total_message_bytes": 0,
            "total_consumed_bytes": 0,
            "total_offered_packets": 0,
            "total_sent_packets": 0,
            "total_received_packets": 0,
        }
        sum_keys = [
            "selected_units",
            "sent_units",
            "received_units",
            "message_bytes",
            "consumed_bytes",
            "offered_packets",
            "sent_packets",
            "received_packets",
        ]
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

    # ------------------------------------------------------------------
    # Markov state / delay helpers
    # ------------------------------------------------------------------
    def _sample_next_state(self, current: str, device: torch.device) -> str:
        probs_dict = self.transition_matrix.get(current, {})
        probs = torch.tensor(
            [float(probs_dict.get(state, 0.0)) for state in self.states],
            dtype=torch.float32,
            device=device,
        )
        if probs.sum() <= 0:
            probs = torch.ones(len(self.states), dtype=torch.float32, device=device)
        probs = probs / probs.sum().clamp_min(1e-12)
        return self.states[int(torch.multinomial(probs, 1).item())]

    def _next_state(self, link_key: str, device: torch.device) -> str:
        # The first frame uses initial_state exactly. Transition starts from the
        # second observed frame of this link.
        if link_key not in self._link_initialized:
            state = self.initial_state
            self._link_initialized.add(link_key)
        else:
            state = self._sample_next_state(
                self._link_state.get(link_key, self.initial_state), device
            )
        self._link_state[link_key] = state
        return state

    def _delay_slots_from_profile(self, profile: Dict[str, Any]) -> int:
        temporal_source = str(profile.get("temporal_source", "current"))
        delay_ms = float(profile.get("delay_ms", 0.0))
        frame_ms = 1000.0 / max(self.fps, 1e-6)
        if temporal_source == "current":
            return 0
        if temporal_source == "previous_frame":
            return max(1, int(round(delay_ms / frame_ms)))
        return max(0, int(delay_ms // frame_ms))

    def _get_or_create_session(self, link_key: str, device: torch.device) -> Dict[str, Any]:
        if link_key in self._frame_sessions:
            return self._frame_sessions[link_key]

        state = self._next_state(link_key, device)
        profile = self.state_profiles.get(state, {})
        bandwidth_mbps = float(profile.get("bandwidth_mbps", 0.0))
        raw_budget_bytes = int(bandwidth_mbps * 1e6 / 8.0 / max(self.fps, 1e-6))
        budget_packets = max(0, raw_budget_bytes // self.packet_size_bytes)
        budget_bytes = int(budget_packets * self.packet_size_bytes)
        session = {
            "state": state,
            "profile": profile,
            "bandwidth_mbps": bandwidth_mbps,
            "packet_loss_rate": float(profile.get("packet_loss_rate", 0.0)),
            "delay_slots": self._delay_slots_from_profile(profile),
            "initial_budget_bytes": budget_bytes,
            "remaining_budget_bytes": budget_bytes,
            "initial_budget_packets": int(budget_packets),
            "remaining_budget_packets": int(budget_packets),
        }
        self._frame_sessions[link_key] = session
        return session

    def _select_delayed_feature(
        self,
        link_key: str,
        scale_idx: int,
        current: torch.Tensor,
        delay_slots: int,
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

    # ------------------------------------------------------------------
    # Serialization / packet helpers
    # ------------------------------------------------------------------
    def _serialize_feature(self, message: torch.Tensor) -> Dict[str, Any]:
        if message.dim() != 3:
            raise ValueError("Expected [C,H,W], got {}".format(tuple(message.shape)))
        channels, height, width = [int(v) for v in message.shape]
        cells = message.permute(1, 2, 0).contiguous().view(height * width, channels)
        scores = cells.detach().abs().sum(dim=1)

        if self.send_nonzero_only:
            selected = torch.nonzero(scores > self.nonzero_epsilon, as_tuple=False).view(-1)
        else:
            selected = torch.arange(height * width, device=message.device, dtype=torch.long)

        if self.selection_policy == "magnitude" and selected.numel() > 0:
            local_order = torch.argsort(scores[selected], descending=True)
            selected = selected[local_order]

        if selected.numel() > 0:
            values = cells[selected].contiguous().view(-1)
        else:
            values = message.new_empty((0,))

        return {
            "values": values,
            "selected_cells": selected,
            "channels": channels,
            "height": height,
            "width": width,
            "selected_units": int(selected.numel()),
            "num_values": int(values.numel()),
            "message_bytes": int(values.numel() * self.bytes_per_value),
            "source": message,
        }

    def _restore_feature(self, entry: Dict[str, Any], received_values: torch.Tensor) -> torch.Tensor:
        source = entry["source"]
        channels = int(entry["channels"])
        height = int(entry["height"])
        width = int(entry["width"])
        selected = entry["selected_cells"]

        if self.zero_fill_missing:
            cells = source.new_zeros((height * width, channels))
        else:
            cells = source.permute(1, 2, 0).contiguous().view(height * width, channels).clone()

        if selected.numel() > 0:
            cells[selected] = received_values.view(-1, channels)
        return cells.view(height, width, channels).permute(2, 0, 1).contiguous()

    def _sample_packet_keep(
        self,
        total_packets: int,
        sent_packets: int,
        loss_rate: float,
        device: torch.device,
    ) -> torch.Tensor:
        keep = torch.zeros(total_packets, dtype=torch.bool, device=device)
        if sent_packets <= 0:
            return keep
        if loss_rate <= 0.0:
            keep[:sent_packets] = True
        elif loss_rate < 1.0:
            keep[:sent_packets] = torch.rand(sent_packets, device=device) >= loss_rate
        return keep

    def _value_masks(
        self,
        byte_offset: int,
        num_values: int,
        sent_packets: int,
        packet_keep: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if num_values <= 0:
            empty = torch.empty(0, dtype=torch.bool, device=device)
            empty_long = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty, empty_long, empty_long

        starts = int(byte_offset) + torch.arange(num_values, device=device, dtype=torch.long) * self.bytes_per_value
        first_packet = starts // self.packet_size_bytes
        last_packet = (starts + self.bytes_per_value - 1) // self.packet_size_bytes
        sent = last_packet < int(sent_packets)
        received = sent.clone()
        if received.any():
            safe_first = first_packet.clamp(min=0, max=max(int(packet_keep.numel()) - 1, 0))
            safe_last = last_packet.clamp(min=0, max=max(int(packet_keep.numel()) - 1, 0))
            received = received & packet_keep[safe_first] & packet_keep[safe_last]
        return sent, received, first_packet, last_packet

    def _process_link(
        self,
        outputs: List[torch.Tensor],
        global_index: int,
        link_key: str,
        session: Dict[str, Any],
        active_scale_indices: Sequence[int],
        num_scales: int,
    ) -> Dict[str, Any]:
        entries = []
        byte_cursor = 0

        for scale_idx in active_scale_indices:
            current = outputs[scale_idx][global_index]
            delayed = self._select_delayed_feature(
                link_key,
                scale_idx,
                current,
                int(session["delay_slots"]),
            )
            entry = self._serialize_feature(delayed)
            entry["scale_idx"] = int(scale_idx)
            entry["byte_offset"] = int(byte_cursor)
            byte_cursor += int(entry["message_bytes"])
            entries.append(entry)

        total_message_bytes = int(byte_cursor)
        offered_packets = (
            int(math.ceil(total_message_bytes / float(self.packet_size_bytes)))
            if total_message_bytes > 0
            else 0
        )
        remaining_before_packets = int(session["remaining_budget_packets"])
        sent_packets = min(offered_packets, remaining_before_packets)
        consumed_bytes = int(sent_packets * self.packet_size_bytes)
        session["remaining_budget_packets"] = max(0, remaining_before_packets - sent_packets)
        session["remaining_budget_bytes"] = int(
            session["remaining_budget_packets"] * self.packet_size_bytes
        )

        packet_keep = self._sample_packet_keep(
            offered_packets,
            sent_packets,
            float(session["packet_loss_rate"]),
            outputs[0].device,
        )
        received_packets = int(packet_keep[:sent_packets].sum().item()) if sent_packets > 0 else 0

        per_scale_stats = []
        total_selected_cells = 0
        total_sent_cells = 0
        total_received_cells = 0
        payload_sent_bytes = 0
        payload_received_bytes = 0

        for entry in entries:
            values = entry["values"]
            sent_mask, recv_mask, first_packet, last_packet = self._value_masks(
                int(entry["byte_offset"]),
                int(entry["num_values"]),
                sent_packets,
                packet_keep,
                values.device,
            )
            received_values = values * recv_mask.to(values.dtype)
            scale_idx = int(entry["scale_idx"])
            outputs[scale_idx][global_index] = self._restore_feature(entry, received_values)

            channels = int(entry["channels"])
            selected_cells = int(entry["selected_units"])
            if selected_cells > 0:
                sent_cells_mask = sent_mask.view(selected_cells, channels).all(dim=1)
                recv_cells_mask = recv_mask.view(selected_cells, channels).all(dim=1)
                sent_cells = int(sent_cells_mask.sum().item())
                received_cells = int(recv_cells_mask.sum().item())
            else:
                sent_cells = 0
                received_cells = 0

            sent_values = int(sent_mask.sum().item())
            received_values_count = int(recv_mask.sum().item())
            payload_sent = int(sent_values * self.bytes_per_value)
            payload_received = int(received_values_count * self.bytes_per_value)

            if entry["num_values"] > 0:
                scale_first_packet = int(first_packet.min().item())
                scale_last_packet = int(last_packet.max().item())
                scale_offered_packets = scale_last_packet - scale_first_packet + 1
                scale_sent_packet_end = min(scale_last_packet, sent_packets - 1)
                scale_sent_packets = max(0, scale_sent_packet_end - scale_first_packet + 1)
                if scale_sent_packets > 0:
                    scale_received_packets = int(
                        packet_keep[
                            scale_first_packet: scale_first_packet + scale_sent_packets
                        ].sum().item()
                    )
                else:
                    scale_received_packets = 0
            else:
                scale_first_packet = -1
                scale_last_packet = -1
                scale_offered_packets = 0
                scale_sent_packets = 0
                scale_received_packets = 0

            per_scale_stats.append(
                {
                    "scale_idx": scale_idx,
                    "shape": [int(v) for v in entry["source"].shape],
                    "selected_units": selected_cells,
                    "sent_units": sent_cells,
                    "received_units": received_cells,
                    "num_values": int(entry["num_values"]),
                    "sent_values": sent_values,
                    "received_values": received_values_count,
                    "message_bytes": int(entry["message_bytes"]),
                    "payload_sent_bytes": payload_sent,
                    "payload_received_bytes": payload_received,
                    "first_packet": scale_first_packet,
                    "last_packet": scale_last_packet,
                    "offered_packets_touched": int(scale_offered_packets),
                    "sent_packets_touched": int(scale_sent_packets),
                    "received_packets_touched": int(scale_received_packets),
                }
            )

            total_selected_cells += selected_cells
            total_sent_cells += sent_cells
            total_received_cells += received_cells
            payload_sent_bytes += payload_sent
            payload_received_bytes += payload_received

        return {
            "state": session["state"],
            "bandwidth_mbps": float(session["bandwidth_mbps"]),
            "packet_loss_rate": float(session["packet_loss_rate"]),
            "delay_slots": int(session["delay_slots"]),
            "packet_size_bytes": int(self.packet_size_bytes),
            "bytes_per_value": int(self.bytes_per_value),
            "serialization_order": self.serialization_order,
            "selection_policy": self.selection_policy,
            "send_nonzero_only": self.send_nonzero_only,
            "active_scales": [int(x) for x in active_scale_indices],
            "num_scales": int(num_scales),
            "selected_units": int(total_selected_cells),
            "sent_units": int(total_sent_cells),
            "received_units": int(total_received_cells),
            "message_bytes": total_message_bytes,
            "payload_sent_bytes": int(payload_sent_bytes),
            "payload_received_bytes": int(payload_received_bytes),
            "offered_packets": int(offered_packets),
            "sent_packets": int(sent_packets),
            "received_packets": int(received_packets),
            "budget_dropped_packets": int(max(0, offered_packets - sent_packets)),
            "loss_dropped_packets": int(max(0, sent_packets - received_packets)),
            "consumed_bytes": consumed_bytes,
            "wire_bytes": consumed_bytes,
            "initial_budget_bytes": int(session["initial_budget_bytes"]),
            "initial_budget_packets": int(session["initial_budget_packets"]),
            "remaining_budget_bytes_after": int(session["remaining_budget_bytes"]),
            "remaining_budget_packets_after": int(session["remaining_budget_packets"]),
            "per_scale": per_scale_stats,
        }

    # ------------------------------------------------------------------
    # Public forward API
    # ------------------------------------------------------------------
    def forward_multiscale(
        self,
        feature_list: Sequence[torch.Tensor],
        record_len: torch.Tensor,
        frame_id: Optional[Any] = None,
        scale_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
        if not feature_list:
            return list(feature_list), []
        if (not self.enabled) or record_len is None:
            return list(feature_list), []

        outputs = [feature.clone() for feature in feature_list]
        num_scales = len(outputs)
        if scale_indices is None:
            candidates = list(range(num_scales))
        else:
            candidates = [int(x) for x in scale_indices]
        active = [
            index for index in candidates
            if 0 <= index < num_scales
            and (self.active_scales is None or index in self.active_scales)
        ]
        if not active:
            return outputs, []

        self.start_frame(frame_id=frame_id)
        if torch.is_tensor(record_len):
            lengths = [int(value) for value in record_len.detach().cpu().tolist()]
        else:
            lengths = [int(value) for value in record_len]

        start = 0
        frame_records = []
        for batch_index, cav_count in enumerate(lengths):
            for local_index in range(cav_count):
                global_index = start + local_index
                if local_index == 0 and not self.impair_ego:
                    continue

                link_key = "b{}_cav{}".format(batch_index, local_index)
                session = self._get_or_create_session(link_key, outputs[0].device)
                stat = self._process_link(
                    outputs,
                    global_index,
                    link_key,
                    session,
                    active,
                    num_scales,
                )
                info = {
                    "frame_index": int(self._frame_index),
                    "frame_id": frame_id,
                    "batch": int(batch_index),
                    "cav": int(local_index),
                    "global_idx": int(global_index),
                    "link_key": link_key,
                }
                info.update(stat)
                self.latest_info.append(info)
                self.records.append(info)
                frame_records.append(info)

                if self.verbose:
                    print(
                        "[CoopDiff-Packet-Markov] frame={} link={} state={} "
                        "payload={}B packets={}/{} recv={} wire={}B".format(
                            self._frame_index,
                            link_key,
                            info["state"],
                            info["message_bytes"],
                            info["sent_packets"],
                            info["offered_packets"],
                            info["received_packets"],
                            info["consumed_bytes"],
                        )
                    )
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
        # Compatibility path for older call sites. The actual CoopDiff model uses
        # forward_multiscale so all scales share one continuous packet stream.
        outputs, records = self.forward_multiscale(
            [x],
            record_len=record_len,
            frame_id=frame_id,
            scale_indices=[0],
        )
        for record in records:
            record["logical_scale_idx"] = int(scale_idx)
            record["logical_num_scales"] = int(num_scales)
        return outputs[0], records
