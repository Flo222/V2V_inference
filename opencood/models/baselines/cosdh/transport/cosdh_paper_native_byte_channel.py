from __future__ import print_function

import copy
import math
import random
from collections import defaultdict, deque

import numpy as np


class CosDHPaperNativeByteChannel(object):
    """Fixed, no-policy byte channel for paper-native CoSDH frames.

    One sender frame is already serialized in the fixed order
    scale0 -> scale1 -> scale2 -> late cls -> late reg -> late dir.
    This channel never quantizes, compresses, adds redundancy, caches a model
    feature, or performs content selection.  It only applies:

      * one Markov state per sender-to-ego link and perception frame;
      * one byte budget shared by all segments in that sender frame;
      * sequential packetization and bandwidth truncation;
      * Bernoulli packet loss;
      * configured frame delay.

    ``mode=ideal`` preserves the same serialization boundary but transmits all
    bytes with zero loss and zero delay.
    """

    def __init__(self, cfg=None):
        cfg = copy.deepcopy(cfg or {})
        self.mode = str(cfg.get("mode", "ideal")).lower()
        if self.mode not in ("ideal", "markov"):
            raise ValueError("byte channel mode must be ideal or markov")

        self.fps = float(cfg.get("fps", 10.0))
        packet_cfg = cfg.get("packetization", {}) or {}
        self.packet_size_bytes = int(
            packet_cfg.get("packet_size_bytes", cfg.get("packet_size_bytes", 1024))
        )
        self.zero_fill_missing = bool(
            packet_cfg.get("zero_fill_missing", True)
        )
        self.protect_headers = bool(cfg.get("protect_headers", True))
        self.verbose = bool(cfg.get("verbose", False))

        self.states = list(cfg.get("states", ["good", "medium", "bad"]))
        self.initial_state = str(cfg.get("initial_state", "medium"))
        if self.initial_state not in self.states:
            self.initial_state = self.states[0]

        self.transition_matrix = copy.deepcopy(
            cfg.get(
                "transition_matrix",
                {
                    "good": {"good": 0.85, "medium": 0.13, "bad": 0.02},
                    "medium": {"good": 0.10, "medium": 0.80, "bad": 0.10},
                    "bad": {"good": 0.03, "medium": 0.17, "bad": 0.80},
                },
            )
        )
        self.state_profiles = copy.deepcopy(
            cfg.get(
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
        )

        self._link_state = {}
        self._delay_cache = defaultdict(lambda: deque(maxlen=16))
        self.frame_index = -1
        self.latest_info = []

    @property
    def enabled(self):
        return self.mode == "markov"

    def start_frame(self):
        self.frame_index += 1
        self.latest_info = []

    def _next_state(self, link_key):
        current = self._link_state.get(link_key, self.initial_state)
        row = self.transition_matrix.get(current, {}) or {}
        probs = [max(0.0, float(row.get(state, 0.0))) for state in self.states]
        total = sum(probs)
        if total <= 0:
            probs = [1.0 / float(len(self.states)) for _ in self.states]
        else:
            probs = [value / total for value in probs]

        sample = random.random()
        cumulative = 0.0
        next_state = self.states[-1]
        for state, prob in zip(self.states, probs):
            cumulative += prob
            if sample <= cumulative:
                next_state = state
                break
        self._link_state[link_key] = next_state
        return next_state

    def _delay_slots(self, profile):
        frame_ms = 1000.0 / max(self.fps, 1e-6)
        temporal_source = str(profile.get("temporal_source", "current")).lower()
        delay_ms = float(profile.get("delay_ms", 0.0))
        slots = int(delay_ms // frame_ms)
        if temporal_source in ("previous", "previous_frame", "history"):
            slots = max(1, slots)
        return max(0, slots)

    @staticmethod
    def _frame_signature(frame):
        return tuple(
            (
                str(segment.get("kind")),
                str(segment.get("name")),
                tuple(segment.get("shape", [])),
                int(segment.get("byte_length", 0)),
            )
            for segment in frame.get("segments", [])
        )

    def _select_delayed_frame(self, link_key, current_frame, delay_slots):
        cache = self._delay_cache[link_key]
        cache.append(copy.deepcopy(current_frame))
        if delay_slots <= 0:
            return current_frame, False, True

        index = len(cache) - 1 - int(delay_slots)
        if index < 0:
            return current_frame, True, False

        delayed = cache[index]
        compatible = self._frame_signature(delayed) == self._frame_signature(current_frame)
        if not compatible:
            return current_frame, True, False
        return delayed, True, True

    @staticmethod
    def _segment_byte_stats(segments, sent_limit, valid_mask):
        stats = []
        for segment in segments:
            start = int(segment["stream_offset"])
            end = start + int(segment["byte_length"])
            sent_bytes = max(0, min(end, sent_limit) - start)
            received_bytes = int(valid_mask[start:end].sum()) if end > start else 0
            stats.append(
                {
                    "kind": str(segment.get("kind")),
                    "name": str(segment.get("name")),
                    "source_bytes": int(segment.get("byte_length", 0)),
                    "sent_bytes_before_loss": int(sent_bytes),
                    "received_valid_bytes": int(received_bytes),
                }
            )
        return stats

    def _ideal(self, frame, link_key):
        stream = frame["stream"].copy()
        valid = np.ones(stream.shape[0], dtype=np.bool_)
        info = {
            "mode": "ideal",
            "link_key": str(link_key),
            "state": "ideal",
            "bandwidth_mbps": None,
            "packet_loss_rate": 0.0,
            "delay_slots": 0,
            "source_bytes": int(stream.shape[0]),
            "budget_bytes": int(stream.shape[0]),
            "sent_bytes_before_loss": int(stream.shape[0]),
            "received_valid_bytes": int(stream.shape[0]),
            "budget_truncated_bytes": 0,
            "packet_loss_bytes": 0,
            "total_packets": int(math.ceil(stream.shape[0] / float(max(self.packet_size_bytes, 1)))) if stream.shape[0] else 0,
            "sent_packets": int(math.ceil(stream.shape[0] / float(max(self.packet_size_bytes, 1)))) if stream.shape[0] else 0,
            "received_packets": int(math.ceil(stream.shape[0] / float(max(self.packet_size_bytes, 1)))) if stream.shape[0] else 0,
            "used_delayed_frame": False,
            "delay_frame_available": True,
        }
        info["segments"] = self._segment_byte_stats(
            frame["segments"], int(stream.shape[0]), valid
        )
        return {"stream": stream, "valid": valid, "frame": frame}, info

    def _markov(self, current_frame, link_key):
        state = self._next_state(link_key)
        profile = self.state_profiles[state]
        delay_slots = self._delay_slots(profile)
        source_frame, used_delayed, delay_available = self._select_delayed_frame(
            link_key, current_frame, delay_slots
        )

        source_stream = source_frame["stream"]
        source_bytes = int(source_stream.shape[0])
        bandwidth_mbps = float(profile.get("bandwidth_mbps", 0.0))
        raw_budget = int(
            bandwidth_mbps * 1e6 / 8.0 / max(self.fps, 1e-6)
        )
        budget_packets = max(0, raw_budget // max(self.packet_size_bytes, 1))
        budget_bytes = int(budget_packets * self.packet_size_bytes)
        sent_limit = min(source_bytes, budget_bytes)

        total_packets = int(
            math.ceil(source_bytes / float(max(self.packet_size_bytes, 1)))
        ) if source_bytes else 0
        sent_packets = int(
            math.ceil(sent_limit / float(max(self.packet_size_bytes, 1)))
        ) if sent_limit else 0

        received = np.zeros(source_bytes, dtype=np.uint8)
        valid = np.zeros(source_bytes, dtype=np.bool_)
        packet_loss_rate = float(profile.get("packet_loss_rate", 0.0))
        received_packets = 0

        if delay_available:
            for packet_index in range(sent_packets):
                start = packet_index * self.packet_size_bytes
                end = min(start + self.packet_size_bytes, sent_limit)
                if start >= end:
                    continue
                dropped = random.random() < packet_loss_rate
                if not dropped:
                    received[start:end] = source_stream[start:end]
                    valid[start:end] = True
                    received_packets += 1

            if self.protect_headers:
                for segment in source_frame.get("segments", []):
                    header_start = int(segment["stream_offset"])
                    header_end = header_start + int(segment.get("header_bytes", 0))
                    header_end = min(header_end, sent_limit)
                    if header_end > header_start:
                        received[header_start:header_end] = source_stream[header_start:header_end]
                        valid[header_start:header_end] = True

        valid_sent_bytes = int(valid[:sent_limit].sum())
        info = {
            "mode": "markov",
            "link_key": str(link_key),
            "state": str(state),
            "bandwidth_mbps": bandwidth_mbps,
            "packet_loss_rate": packet_loss_rate,
            "delay_slots": int(delay_slots),
            "source_bytes": source_bytes,
            "raw_budget_bytes": int(raw_budget),
            "budget_bytes": int(budget_bytes),
            "sent_bytes_before_loss": int(sent_limit),
            "received_valid_bytes": int(valid.sum()),
            "budget_truncated_bytes": int(max(0, source_bytes - sent_limit)),
            "packet_loss_bytes": int(max(0, sent_limit - valid_sent_bytes)),
            "total_packets": int(total_packets),
            "sent_packets": int(sent_packets),
            "received_packets": int(received_packets),
            "used_delayed_frame": bool(used_delayed),
            "delay_frame_available": bool(delay_available),
            "protect_headers": bool(self.protect_headers),
        }
        info["segments"] = self._segment_byte_stats(
            source_frame["segments"], sent_limit, valid
        )

        if self.verbose:
            print(
                "[CoSDH-Paper-Byte] link={} state={} bw={}Mbps plr={} "
                "delay={} bytes recv/sent/source={}/{}/{}".format(
                    link_key,
                    state,
                    bandwidth_mbps,
                    packet_loss_rate,
                    delay_slots,
                    info["received_valid_bytes"],
                    info["sent_bytes_before_loss"],
                    source_bytes,
                )
            )

        return {"stream": received, "valid": valid, "frame": source_frame}, info

    def transmit_frame(self, sender_frames, link_keys):
        """Transmit all sender streams for one perception frame.

        This is one frame-level call.  Every sender gets one physical link
        state/budget; every segment of that sender shares that one budget.
        """
        if len(sender_frames) != len(link_keys):
            raise ValueError("sender_frames and link_keys length mismatch")
        self.start_frame()

        results = []
        infos = []
        for frame, link_key in zip(sender_frames, link_keys):
            if self.mode == "ideal":
                result, info = self._ideal(frame, link_key)
            else:
                result, info = self._markov(frame, link_key)
            results.append(result)
            infos.append(info)

        self.latest_info = copy.deepcopy(infos)
        return results, infos


__all__ = ["CosDHPaperNativeByteChannel"]
