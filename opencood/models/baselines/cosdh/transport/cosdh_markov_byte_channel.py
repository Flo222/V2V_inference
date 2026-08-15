from collections import defaultdict, deque
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosDHMarkovByteChannel(nn.Module):
    """
    CoSDH Markov channel, multi-scale shared-link-budget version.

    Experimental meaning:
      - Keep CoSDH's original multi-scale communication selection/mask.
      - Do NOT introduce quantization.
      - Do NOT use 4x4/8x8 spatial patch packets.
      - Treat each CoSDH-selected BEV cell feature vector as one raw-float
        communication unit.
      - Compute each unit size as C * bytes_per_value Bytes.
      - Map the selected message size to fixed-size packets for bandwidth budget.
      - Apply link-level good/medium/bad Markov state transition.
      - Apply current/previous-frame delay policy.
      - Missing / unsent / lost units are zero-filled.

    Key alignment with V2X-ViT-Markov / Where2comm-Markov:
      - One Markov state is sampled per ego<-CAV link per frame.
      - One bandwidth budget is shared by all CoSDH scales for the same link
        within that frame. The session key excludes scale_idx; only delay cache
        is scale-specific because feature shapes differ across scales.
      - All CoSDH scales may communicate, but they consume the same link budget
        instead of each scale getting an independent full bandwidth budget.

    Interface:
        forward(x, record_len, communication_mask=None, frame_id=None,
                scale_idx=0, num_scales=1)

    where x is the CoSDH-selected message after:
        warp_x = warp_x * communication_masks
    """

    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or {}

        self.enabled = bool(cfg.get("enabled", False))
        self.impair_ego = bool(cfg.get("impair_ego", False))
        self.scope = cfg.get("scope", "link")

        self.fps = float(cfg.get("fps", 10.0))

        packet_cfg = cfg.get("packetization", {})
        self.packet_size_bytes = int(packet_cfg.get("packet_size_bytes", 1024))
        self.bytes_per_value = int(packet_cfg.get("bytes_per_value", 4))
        self.zero_fill_missing = bool(packet_cfg.get("zero_fill_missing", True))
        # When the selected CoSDH message exceeds the byte budget, the original
        # implementation kept the first cells in raster order.  That creates a
        # spatial bias unrelated to communication importance.  Use random
        # budget truncation by default; set to ``raster`` to reproduce the old
        # behavior or ``magnitude`` for a deterministic energy-based proxy.
        self.selection_policy = str(cfg.get("selection_policy", "random")).lower()

        self.states = cfg.get("states", ["good", "medium", "bad"])
        self.initial_state = cfg.get("initial_state", "medium")
        if self.initial_state not in self.states:
            self.initial_state = self.states[0]

        self.transition_matrix = cfg.get("transition_matrix", {
            "good": {"good": 0.85, "medium": 0.13, "bad": 0.02},
            "medium": {"good": 0.10, "medium": 0.80, "bad": 0.10},
            "bad": {"good": 0.03, "medium": 0.17, "bad": 0.80},
        })

        self.state_profiles = cfg.get("state_profiles", {
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
        })

        self.verbose = bool(cfg.get("verbose", False))

        # Link-level Markov state persists across frames.
        self._link_state = {}

        # Optional CAV-id aliases for the current ego frame.  When available,
        # intermediate CoSDH features and late dense detections use the same
        # physical link key, so they share one Markov state and one byte budget.
        self._link_key_aliases = None

        # Per-frame shared budget sessions. Reset by start_frame().
        self._frame_sessions = {}

        # Delay cache per link and scale.
        self._delay_cache = defaultdict(lambda: deque(maxlen=8))

        self.latest_info = []
        self._frame_index = -1

        # Cache only when at least one profile can request previous-frame data.
        self.need_delay_cache = False
        frame_ms = 1000.0 / max(self.fps, 1e-6)
        for profile in self.state_profiles.values():
            if profile.get("temporal_source", "current") == "previous_frame":
                self.need_delay_cache = True
            if float(profile.get("delay_ms", 0.0)) >= frame_ms:
                self.need_delay_cache = True

    def _normalize_link_aliases(self, link_key_aliases):
        if link_key_aliases is None:
            return None

        if torch.is_tensor(link_key_aliases):
            link_key_aliases = link_key_aliases.detach().cpu().tolist()

        # DataLoader with batch size 1 may wrap the list once.
        if isinstance(link_key_aliases, (list, tuple)) and len(link_key_aliases) == 1 \
                and isinstance(link_key_aliases[0], (list, tuple)):
            link_key_aliases = link_key_aliases[0]

        if not isinstance(link_key_aliases, (list, tuple)):
            link_key_aliases = [link_key_aliases]

        aliases = []
        for item in link_key_aliases:
            if torch.is_tensor(item):
                item = item.detach().cpu().item() if item.numel() == 1 else item.detach().cpu().tolist()
            if isinstance(item, (list, tuple)) and len(item) == 1:
                item = item[0]
            aliases.append(str(item))
        return aliases

    def start_frame(self, frame_id=None, link_key_aliases=None):
        """Start a new inference frame/session.

        This should be called once before CoSDH iterates over its multi-scale
        fusion modules. It ensures all scales in the same frame share the same
        link state and bandwidth budget.  If CAV ids are provided, all messages
        from the same non-ego CAV also share the same physical link key across
        intermediate and late branches.
        """
        self._frame_index += 1
        self._frame_sessions = {}
        self.latest_info = []
        aliases = self._normalize_link_aliases(link_key_aliases)
        if aliases is not None:
            self._link_key_aliases = aliases

    def _resolve_link_key(self, b, local_idx):
        aliases = self._link_key_aliases
        if aliases is not None and int(b) == 0 and int(local_idx) < len(aliases):
            return "link_{}".format(aliases[int(local_idx)])
        return "b{}_cav{}".format(int(b), int(local_idx))

    def _next_state(self, link_key, device):
        cur = self._link_state.get(link_key, self.initial_state)
        probs_dict = self.transition_matrix.get(cur, {})

        probs = torch.tensor(
            [float(probs_dict.get(s, 0.0)) for s in self.states],
            dtype=torch.float32,
            device=device,
        )
        if probs.sum() <= 0:
            probs = torch.ones(len(self.states), dtype=torch.float32, device=device)

        probs = probs / probs.sum().clamp_min(1e-6)
        nxt = self.states[torch.multinomial(probs, 1).item()]
        self._link_state[link_key] = nxt
        return nxt

    def _delay_slots_from_profile(self, profile):
        temporal_source = profile.get("temporal_source", "current")

        if temporal_source == "current":
            return 0

        if temporal_source == "previous_frame":
            delay_ms = float(profile.get("delay_ms", 100.0))
            frame_ms = 1000.0 / max(self.fps, 1e-6)
            return max(1, int(round(delay_ms / frame_ms)))

        delay_ms = float(profile.get("delay_ms", 0.0))
        frame_ms = 1000.0 / max(self.fps, 1e-6)
        return int(delay_ms // frame_ms)

    def _get_or_create_session(self, link_key, device):
        """Return the shared per-frame session for this link.

        The session stores one Markov state and one remaining byte budget shared
        by all CoSDH scales in the current frame.
        """
        if link_key in self._frame_sessions:
            return self._frame_sessions[link_key]

        state = self._next_state(link_key, device)
        profile = self.state_profiles[state]

        bandwidth_mbps = float(profile.get("bandwidth_mbps", 0.0))
        budget_bytes = int(bandwidth_mbps * 1e6 / 8.0 / max(self.fps, 1e-6))
        budget_packets = max(0, budget_bytes // max(self.packet_size_bytes, 1))
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
        }
        self._frame_sessions[link_key] = session
        return session

    def _get_spatial_mask(self, msg, communication_mask, global_idx):
        """Build a spatial selected mask [H, W].

        CoSDH communication_mask is usually [sum_cav, 1, H, W]. If channel
        dimension is not 1, reduce it to spatial selection by max over C.
        If communication_mask is scalar/unavailable, fall back to non-zero
        locations of the already masked message.
        """
        C, H, W = msg.shape

        if torch.is_tensor(communication_mask) and communication_mask.numel() > 1:
            try:
                if communication_mask.dim() == 4 and communication_mask.shape[0] > global_idx:
                    one = communication_mask[global_idx]

                    if one.dim() == 3:
                        if one.shape[0] != 1:
                            one = one.max(dim=0, keepdim=True)[0]
                    elif one.dim() == 2:
                        one = one.unsqueeze(0)

                    if one.shape[-2:] != (H, W):
                        one = F.interpolate(
                            one.unsqueeze(0).float(),
                            size=(H, W),
                            mode="nearest",
                        ).squeeze(0)

                    return one[0] > 0

                if communication_mask.dim() == 3 and communication_mask.shape[0] > global_idx:
                    one = communication_mask[global_idx].unsqueeze(0)
                    if one.shape[-2:] != (H, W):
                        one = F.interpolate(
                            one.unsqueeze(0).float(),
                            size=(H, W),
                            mode="nearest",
                        ).squeeze(0)
                    return one[0] > 0

            except Exception:
                pass

        return msg.abs().sum(dim=0) > 0

    def _select_delayed_message(self, link_key, scale_idx, cur_msg, cur_mask, delay_slots):
        cache_key = f"{link_key}_scale{int(scale_idx)}"

        if self.need_delay_cache:
            cache = self._delay_cache[cache_key]
            cache.append((cur_msg.detach().clone(), cur_mask.detach().clone()))
        else:
            cache = self._delay_cache[cache_key]

        if delay_slots <= 0:
            return cur_msg, cur_mask

        idx = len(cache) - 1 - int(delay_slots)
        if idx >= 0:
            msg, mask = cache[idx]
            return msg.to(cur_msg.device), mask.to(cur_msg.device)

        return torch.zeros_like(cur_msg), torch.zeros_like(cur_mask, dtype=torch.bool)

    def _apply_channel_to_cells(self, msg, spatial_mask, session, scale_idx, num_scales):
        """Apply shared-budget cell-vector packet channel.

        msg: [C,H,W]
        spatial_mask: [H,W], True means CoSDH selected this BEV cell.
        session: shared link session for this frame.
        """
        C, H, W = msg.shape
        flat_mask = spatial_mask.reshape(-1)

        selected_idx = torch.nonzero(flat_mask, as_tuple=False).flatten()
        selected_cells = int(selected_idx.numel())

        cell_bytes = int(C * self.bytes_per_value)
        total_message_bytes = int(selected_cells * cell_bytes)
        num_packets = int(math.ceil(total_message_bytes / float(max(self.packet_size_bytes, 1)))) if selected_cells > 0 else 0

        if selected_cells <= 0:
            return torch.zeros_like(msg), {
                "selected_cells": 0,
                "message_bytes": 0,
                "num_packets": 0,
                "remaining_budget_bytes_before": int(session["remaining_budget_bytes"]),
                "remaining_budget_bytes_after": int(session["remaining_budget_bytes"]),
                "sent_units": 0,
                "received_units": 0,
                "source_payload_bytes": 0,
                "packet_size_bytes": int(self.packet_size_bytes),
                "num_source_packets": 0,
                "num_transmitted_packets": 0,
                "num_received_packets": 0,
                "consumed_bytes": 0,
                "transmitted_wire_bytes": 0,
                "received_wire_bytes": 0,
                "received_payload_bytes": 0,
            }

        remaining_before = int(session["remaining_budget_bytes"])
        max_send_cells = remaining_before // max(cell_bytes, 1)
        max_send_cells = max(0, min(selected_cells, int(max_send_cells)))

        if max_send_cells <= 0:
            return torch.zeros_like(msg), {
                "selected_cells": selected_cells,
                "message_bytes": total_message_bytes,
                "num_packets": num_packets,
                "remaining_budget_bytes_before": remaining_before,
                "remaining_budget_bytes_after": remaining_before,
                "sent_units": 0,
                "received_units": 0,
                "source_payload_bytes": int(total_message_bytes),
                "packet_size_bytes": int(self.packet_size_bytes),
                "num_source_packets": int(num_packets),
                "num_transmitted_packets": 0,
                "num_received_packets": 0,
                "consumed_bytes": 0,
                "transmitted_wire_bytes": 0,
                "received_wire_bytes": 0,
                "received_payload_bytes": 0,
            }

        if max_send_cells >= selected_cells:
            sent_idx = selected_idx
        elif self.selection_policy == "raster":
            sent_idx = selected_idx[:max_send_cells]
        elif self.selection_policy == "magnitude":
            flat_energy = msg.abs().sum(dim=0).reshape(-1)
            topk = torch.topk(flat_energy[selected_idx], k=max_send_cells, largest=True).indices
            sent_idx = selected_idx[topk]
        else:
            perm = torch.randperm(selected_cells, device=msg.device)[:max_send_cells]
            sent_idx = selected_idx[perm]

        consumed_bytes = int(max_send_cells * cell_bytes)
        session["remaining_budget_bytes"] = max(0, remaining_before - consumed_bytes)

        packet_loss_rate = float(session["packet_loss_rate"])

        if packet_loss_rate <= 0:
            recv = torch.ones(max_send_cells, dtype=torch.bool, device=msg.device)
        else:
            packets_per_cell = max(1, int(math.ceil(cell_bytes / float(max(self.packet_size_bytes, 1)))))
            keep_prob = (1.0 - packet_loss_rate) ** packets_per_cell
            recv = torch.rand(max_send_cells, device=msg.device) < keep_prob

        recv_idx = sent_idx[recv]

        keep_flat = torch.zeros(H * W, dtype=torch.bool, device=msg.device)
        keep_flat[recv_idx] = True
        keep_mask = keep_flat.view(1, H, W).to(dtype=msg.dtype)

        out = msg * keep_mask

        tx_packets = int(math.ceil(consumed_bytes / float(max(self.packet_size_bytes, 1)))) if consumed_bytes > 0 else 0
        received_payload_bytes = int(recv_idx.numel() * cell_bytes)
        rx_packets = int(math.ceil(received_payload_bytes / float(max(self.packet_size_bytes, 1)))) if received_payload_bytes > 0 else 0

        return out, {
            "selected_cells": selected_cells,
            "message_bytes": total_message_bytes,
            "source_payload_bytes": int(total_message_bytes),
            "num_packets": num_packets,
            "num_source_packets": int(num_packets),
            "packet_size_bytes": int(self.packet_size_bytes),
            "cell_bytes": cell_bytes,
            "remaining_budget_bytes_before": remaining_before,
            "remaining_budget_bytes_after": int(session["remaining_budget_bytes"]),
            "initial_budget_bytes": int(session["initial_budget_bytes"]),
            "initial_budget_packets": int(session["initial_budget_packets"]),
            "sent_units": int(max_send_cells),
            "received_units": int(recv_idx.numel()),
            "consumed_bytes": int(consumed_bytes),
            "tx_payload_bytes": int(consumed_bytes),
            "received_payload_bytes": received_payload_bytes,
            "num_transmitted_packets": tx_packets,
            "num_received_packets": rx_packets,
            "transmitted_wire_bytes": int(tx_packets * self.packet_size_bytes),
            "received_wire_bytes": int(rx_packets * self.packet_size_bytes),
            "rx_wire_estimated_from_received_units": True,
            "scale_idx": int(scale_idx),
            "num_scales": int(num_scales),
        }

    def forward(self, x, record_len, communication_mask=None, frame_id=None,
                scale_idx=0, num_scales=1):
        """
        x: CoSDH-selected message after `warp_x = warp_x * communication_masks`,
           shape [sum_cav, C, H, W].
        """
        if not self.enabled:
            return x, []

        if x is None or record_len is None:
            return x, []

        # Compatibility: if caller forgot to call start_frame(), start a session
        # at the first scale. The correct path calls start_frame() in
        # PointPillarCosdhMarkov before the scale loop.
        if int(scale_idx) == 0 and not self._frame_sessions:
            self.start_frame(frame_id=frame_id)

        out = x.clone()

        if torch.is_tensor(record_len):
            record_len_list = record_len.detach().cpu().tolist()
        else:
            record_len_list = list(record_len)

        start = 0

        for b, cav_num in enumerate(record_len_list):
            cav_num = int(cav_num)

            for local_idx in range(cav_num):
                global_idx = start + local_idx

                if local_idx == 0 and not self.impair_ego:
                    continue

                link_key = self._resolve_link_key(b, local_idx)
                session = self._get_or_create_session(link_key, x.device)

                cur_msg = out[global_idx]
                cur_mask = self._get_spatial_mask(cur_msg, communication_mask, global_idx)

                delayed_msg, delayed_mask = self._select_delayed_message(
                    link_key,
                    scale_idx,
                    cur_msg,
                    cur_mask,
                    session["delay_slots"],
                )

                impaired_msg, stat = self._apply_channel_to_cells(
                    delayed_msg,
                    delayed_mask,
                    session,
                    scale_idx=scale_idx,
                    num_scales=num_scales,
                )

                if impaired_msg.shape != out[global_idx].shape:
                    # Delay cache may contain an old feature from another scale.
                    # In that case the cached message is invalid for this scale.
                    impaired_msg = torch.zeros_like(out[global_idx])
                out[global_idx] = impaired_msg

                info = {
                    "batch": b,
                    "cav": local_idx,
                    "link_key": link_key,
                    "state": session["state"],
                    "bandwidth_mbps": session["bandwidth_mbps"],
                    "packet_loss_rate": session["packet_loss_rate"],
                    "delay_slots": session["delay_slots"],
                }
                info.update(stat)
                self.latest_info.append(info)

                if self.verbose:
                    print(
                        "[CoSDH-Markov-MultiScaleShared] "
                        "scale={}/{} b={} cav={} state={} bw={}Mbps plr={} delay={} "
                        "cells recv/sent/selected={}/{}/{} budget_bytes {}/{}".format(
                            int(scale_idx),
                            int(num_scales),
                            b,
                            local_idx,
                            session["state"],
                            session["bandwidth_mbps"],
                            session["packet_loss_rate"],
                            session["delay_slots"],
                            stat["received_units"],
                            stat["sent_units"],
                            stat["selected_cells"],
                            stat["remaining_budget_bytes_after"],
                            stat.get("initial_budget_bytes", stat.get("budget_bytes", 0)),
                        )
                    )

            start += cav_num

        return out, self.latest_info
