# -*- coding: utf-8 -*-
"""CoSDH official-code-faithful fixed Markov transport.

The executor keeps the released checkpoint graph but replaces in-process
message passing with the same fixed byte-stream communication semantics used
by the other ARCE baselines:

    scale0 FP16 -> scale1 FP16 -> scale2 FP16 -> Late [x,y,z,h,w,l,yaw,score]
    -> one joint byte stream per non-ego link
    -> fixed 1024-byte packets
    -> prefix budget truncation
    -> Bernoulli packet loss on transmitted packets
    -> zero-fill missing source packets
    -> restore the three encoded scales and complete Late records

No extra quantization, FEC, redundancy, cache, spatial interpolation, UCB, or
C2MAB policy is applied here.  Channel state, budget calculation, Bernoulli
sampling, and fixed-state latency are delegated to the existing ARCEFixedComm
backend so the rules remain aligned with the other baselines.
"""
from __future__ import print_function

import copy
import math

import torch

from opencood.methods.arce.arce_fixed_comm import ARCEFixedComm


class CosDHOfficialFixedMarkovTransport(object):
    SEGMENT_ORDER = ("scale0", "scale1", "scale2", "late_candidates")

    def __init__(self, cfg=None, arce_cfg=None):
        self.cfg = copy.deepcopy(cfg) if isinstance(cfg, dict) else {}
        self.enabled = bool(self.cfg.get("enabled", False))
        self.mode = str(self.cfg.get("mode", "disabled")).strip().lower()
        self.packet_size_bytes = int(self.cfg.get("packet_size_bytes", 1024))
        self.atomic_late_records = bool(
            self.cfg.get("atomic_late_records", True)
        )
        self.zero_fill_missing = bool(
            self.cfg.get("zero_fill_missing", True)
        )
        self.segment_order = list(
            self.cfg.get("segment_order", self.SEGMENT_ORDER)
        )
        if self.packet_size_bytes != 1024:
            raise ValueError(
                "CoSDH fixed-Markov must use the common 1024-byte packet size; "
                "got {}".format(self.packet_size_bytes)
            )
        if tuple(self.segment_order) != self.SEGMENT_ORDER:
            raise ValueError(
                "segment_order must be {}".format(self.SEGMENT_ORDER)
            )

        backend_cfg = {"arce": copy.deepcopy(arce_cfg or {})}
        self.backend = ARCEFixedComm(backend_cfg)
        backend_packet = int(
            self.backend.byte_packetizer.packet_size_bytes
        )
        if backend_packet != self.packet_size_bytes:
            raise ValueError(
                "ARCE backend packet size {} != required {}".format(
                    backend_packet, self.packet_size_bytes
                )
            )

        self._frame_started = False
        self.latest_info = {}
        # ``_link_records`` is deliberately reset per model forward.  Keep a
        # separate append-only audit stream for generic evaluators.
        self.records = []
        self.start_frame()

    @staticmethod
    def _normalize_record_len(record_len):
        if record_len is None:
            return []
        if torch.is_tensor(record_len):
            return [
                int(v) for v in record_len.detach().cpu().reshape(-1)
            ]
        return [int(v) for v in record_len]

    @staticmethod
    def _tensor_bytes(tensor):
        if not torch.is_tensor(tensor):
            raise TypeError("payload segment must be a torch.Tensor")
        return tensor.detach().contiguous().view(torch.uint8).flatten()

    @staticmethod
    def _bytes_to_tensor(byte_stream, shape, dtype):
        return byte_stream.contiguous().view(dtype).view(*shape)

    def _link_key(self, batch_idx, local_idx):
        aliases = self._link_key_aliases
        try:
            if isinstance(aliases, (list, tuple)):
                if aliases and isinstance(aliases[0], (list, tuple)):
                    return str(aliases[batch_idx][local_idx])
                return str(aliases[local_idx])
        except (IndexError, TypeError):
            pass
        return "b{}_cav{}".format(int(batch_idx), int(local_idx))

    def start_frame(
        self,
        record_len=None,
        link_key_aliases=None,
        data_dict=None,
    ):
        self._record_len = self._normalize_record_len(record_len)
        self._link_key_aliases = copy.deepcopy(link_key_aliases)
        self._data_dict = data_dict if isinstance(data_dict, dict) else {}
        self._source_late = {}
        self._received_late = {}
        self._link_records = []
        self._frame_started = True
        self.latest_info = self._build_latest_info()

    def get_records(self):
        return list(self.records)

    def reset_records(self):
        self.records = []

    def bind_ego_context(self, record_len, link_key_aliases, data_dict):
        """Bind ego metadata without erasing candidates prepared in inference."""
        lengths = self._normalize_record_len(record_len)
        if not self._frame_started:
            self.start_frame(record_len, link_key_aliases, data_dict)
            return
        self._record_len = lengths
        self._link_key_aliases = copy.deepcopy(link_key_aliases)
        self._data_dict = data_dict if isinstance(data_dict, dict) else {}

    def set_late_candidates(self, cav_id, boxes_local, scores):
        if not torch.is_tensor(boxes_local) or not torch.is_tensor(scores):
            raise TypeError("Late candidates must be torch tensors")
        if boxes_local.dim() != 2 or int(boxes_local.shape[1]) != 7:
            raise ValueError("boxes_local must have shape [N, 7]")
        if scores.dim() != 1 or int(scores.shape[0]) != int(boxes_local.shape[0]):
            raise ValueError("scores must have shape [N] matching boxes")
        if boxes_local.dtype != torch.float32 or scores.dtype != torch.float32:
            raise TypeError(
                "fixed-Markov Late candidates must be FP32, got {} and {}"
                .format(boxes_local.dtype, scores.dtype)
            )
        records = torch.cat(
            (boxes_local.detach().contiguous(), scores[:, None].contiguous()),
            dim=1,
        )
        self._source_late[str(cav_id)] = records

    def get_received_late_candidates(self, cav_id, device=None):
        key = str(cav_id)
        item = self._received_late.get(key, None)
        if item is None:
            empty_boxes = torch.empty((0, 7), dtype=torch.float32)
            empty_scores = torch.empty((0,), dtype=torch.float32)
            if device is not None:
                empty_boxes = empty_boxes.to(device)
                empty_scores = empty_scores.to(device)
            return empty_boxes, empty_scores
        boxes, scores = item
        if device is not None:
            boxes = boxes.to(device)
            scores = scores.to(device)
        return boxes, scores

    def _frame_id(self, data_dict):
        if hasattr(self.backend, "_infer_frame_id_from_data_dict"):
            return self.backend._infer_frame_id_from_data_dict(data_dict)
        return None

    def _channel_state(self, data_dict, batch_idx, local_idx):
        if hasattr(self.backend, "_get_external_channel_state"):
            return self.backend._get_external_channel_state(
                data_dict, batch_idx, local_idx
            )
        return None, "backend_has_no_external_state_reader"

    def _per_link_budget(self, state_name, num_collaborators):
        if hasattr(self.backend, "_link_budget_bytes_for_state"):
            return float(
                self.backend._link_budget_bytes_for_state(
                    state_name, num_collaborators
                )
            )
        if hasattr(self.backend, "_per_link_budget_bytes"):
            return float(
                self.backend._per_link_budget_bytes(num_collaborators)
            )
        return float("inf")

    def _communicate_bytes(
        self,
        source_bytes,
        link_id,
        frame_id,
        batch_idx,
        local_idx,
        num_collaborators,
        data_dict,
    ):
        packet_result = self.backend.byte_packetizer.packetize(
            source_bytes,
            source_tensor_kind="cosdh_native_joint_uint8",
        )
        source_packets = packet_result.packets
        num_packets = int(packet_result.num_packets)

        requested_state, state_source = self._channel_state(
            data_dict, batch_idx, local_idx
        )
        state_name, resolved_source = self.backend._resolve_active_channel_state(
            requested_channel_state=requested_state,
            link_id=link_id,
            frame_id=frame_id,
        )
        profile = self.backend._profile_for_state(state_name)

        if self.mode in ("ideal", "ideal_check", "unlimited_ideal"):
            budget_bytes = float("inf")
        elif self.mode == "fixed_markov":
            budget_bytes = self._per_link_budget(
                state_name, num_collaborators
            )
        else:
            raise RuntimeError(
                "Unsupported fixed-Markov transport mode: {}".format(self.mode)
            )

        tx_mask = self.backend._select_encoded_packets_by_budget(
            encoded_packets=source_packets,
            budget_bytes=budget_bytes,
            packet_size_bytes=self.packet_size_bytes,
        )
        receive_mask = torch.zeros_like(tx_mask)
        tx_count = int(tx_mask.sum().item())

        if tx_count > 0:
            if self.mode in ("ideal", "ideal_check", "unlimited_ideal"):
                loss_mask_tx = torch.zeros(
                    (tx_count,), dtype=torch.bool, device=source_bytes.device
                )
                loss_info = {
                    "model": "disabled_for_ideal_check",
                    "plr": 0.0,
                    "num_packets": tx_count,
                    "num_received": tx_count,
                    "num_lost": 0,
                }
            else:
                loss_mask_tx, loss_info = self.backend._sample_bernoulli_loss(
                    num_packets=tx_count,
                    state_name=state_name,
                    link_id=link_id,
                    frame_id=frame_id,
                    device=source_bytes.device,
                )
            receive_mask[tx_mask] = ~loss_mask_tx
        else:
            loss_info = {
                "model": "bernoulli",
                "plr": float(profile.get("plr", 0.0)),
                "num_packets": 0,
                "num_received": 0,
                "num_lost": 0,
                "reason": "zero_budget",
            }

        recovered_packets = source_packets.clone()
        recovered_packets[~receive_mask] = 0
        recovered_bytes = self.backend.byte_packetizer.unpacketize(
            recovered_packets, packet_result
        )

        valid_byte_mask = receive_mask[:, None].expand(
            -1, self.packet_size_bytes
        ).reshape(-1)[: int(packet_result.original_num_bytes)]

        transmitted_wire_bytes = int(tx_count * self.packet_size_bytes)
        received_wire_bytes = int(
            receive_mask.sum().item() * self.packet_size_bytes
        )
        latency = self.backend._estimate_fixed_latency(
            transmitted_bytes=transmitted_wire_bytes,
            state_name=state_name,
            link_id=link_id,
            frame_id=frame_id,
            bandwidth_mbps=float(profile.get("bandwidth_mbps", 0.0)),
        )

        return recovered_bytes, valid_byte_mask, {
            "channel_state": state_name,
            "requested_channel_state": requested_state,
            "external_channel_state_source": state_source,
            "resolved_channel_state_source": resolved_source,
            "profile": copy.deepcopy(profile),
            "budget_bytes": budget_bytes,
            "packet_size_bytes": self.packet_size_bytes,
            "num_source_packets": num_packets,
            "num_transmitted_packets": tx_count,
            "num_received_packets": int(receive_mask.sum().item()),
            "num_missing_by_budget": int(num_packets - tx_count),
            "num_lost_by_bernoulli": int(
                tx_count - receive_mask.sum().item()
            ),
            "transmitted_wire_bytes": transmitted_wire_bytes,
            "received_wire_bytes": received_wire_bytes,
            "source_payload_bytes": int(packet_result.original_num_bytes),
            "received_valid_payload_bytes": int(valid_byte_mask.sum().item()),
            "loss": loss_info,
            "latency": latency,
            "tx_mask": tx_mask.detach().cpu().tolist(),
            "receive_mask": receive_mask.detach().cpu().tolist(),
        }

    def communicate_joint_frame(
        self,
        encoded_scales,
        record_len,
        data_dict,
        link_key_aliases=None,
    ):
        if not self.enabled:
            return encoded_scales
        if len(encoded_scales) != 3:
            raise ValueError(
                "CoSDH fixed-Markov expects exactly three encoded scales"
            )
        for idx, tensor in enumerate(encoded_scales):
            if not torch.is_tensor(tensor) or tensor.dim() != 4:
                raise TypeError("encoded scale {} must be [N,C,H,W]".format(idx))
            if tensor.dtype != torch.float16:
                raise TypeError(
                    "encoded scale {} must be FP16, got {}".format(
                        idx, tensor.dtype
                    )
                )

        self.bind_ego_context(record_len, link_key_aliases, data_dict)
        lengths = self._record_len
        total = sum(lengths)
        if any(int(t.shape[0]) != total for t in encoded_scales):
            raise ValueError("encoded scale batch does not match record_len")

        recovered_scales = [t.clone() for t in encoded_scales]
        frame_id = self._frame_id(data_dict)
        offset = 0
        link_records = []

        for batch_idx, cav_num in enumerate(lengths):
            num_collaborators = max(0, int(cav_num) - 1)
            for local_idx in range(1, int(cav_num)):
                global_idx = offset + local_idx
                cav_id = self._link_key(batch_idx, local_idx)
                source_late = self._source_late.get(
                    str(cav_id),
                    torch.empty(
                        (0, 8),
                        dtype=torch.float32,
                        device=encoded_scales[0].device,
                    ),
                ).to(encoded_scales[0].device)

                parts = []
                segment_meta = []
                cursor = 0
                for scale_idx, tensor in enumerate(encoded_scales):
                    row = tensor[global_idx:global_idx + 1].contiguous()
                    raw = self._tensor_bytes(row)
                    parts.append(raw)
                    segment_meta.append({
                        "name": "scale{}".format(scale_idx),
                        "offset": cursor,
                        "num_bytes": int(raw.numel()),
                        "shape": tuple(int(v) for v in row.shape),
                        "dtype": row.dtype,
                    })
                    cursor += int(raw.numel())

                late_raw = self._tensor_bytes(source_late)
                parts.append(late_raw)
                segment_meta.append({
                    "name": "late_candidates",
                    "offset": cursor,
                    "num_bytes": int(late_raw.numel()),
                    "shape": tuple(int(v) for v in source_late.shape),
                    "dtype": source_late.dtype,
                })
                cursor += int(late_raw.numel())

                joint = torch.cat(parts, dim=0) if parts else torch.empty(
                    (0,), dtype=torch.uint8, device=encoded_scales[0].device
                )
                link_id = (int(batch_idx), 0, int(local_idx))
                recovered, valid, channel_record = self._communicate_bytes(
                    joint,
                    link_id=link_id,
                    frame_id=frame_id,
                    batch_idx=batch_idx,
                    local_idx=local_idx,
                    num_collaborators=num_collaborators,
                    data_dict=data_dict,
                )

                segment_records = []
                for scale_idx in range(3):
                    meta = segment_meta[scale_idx]
                    begin = meta["offset"]
                    end = begin + meta["num_bytes"]
                    restored = self._bytes_to_tensor(
                        recovered[begin:end], meta["shape"], meta["dtype"]
                    )
                    recovered_scales[scale_idx][
                        global_idx:global_idx + 1
                    ] = restored
                    seg_valid = valid[begin:end]
                    segment_records.append({
                        "name": meta["name"],
                        "source_bytes": meta["num_bytes"],
                        "received_valid_bytes": int(seg_valid.sum().item()),
                        "receive_fraction": float(
                            seg_valid.float().mean().item()
                        ) if meta["num_bytes"] > 0 else 1.0,
                    })

                late_meta = segment_meta[3]
                begin = late_meta["offset"]
                end = begin + late_meta["num_bytes"]
                late_count = int(source_late.shape[0])
                if late_count > 0:
                    late_restored = self._bytes_to_tensor(
                        recovered[begin:end], late_meta["shape"], torch.float32
                    )
                    late_valid_bytes = valid[begin:end].view(late_count, 32)
                    candidate_valid = late_valid_bytes.all(dim=1)
                    if not self.atomic_late_records:
                        candidate_valid = torch.ones_like(candidate_valid)
                    kept = late_restored[candidate_valid]
                else:
                    candidate_valid = torch.empty(
                        (0,), dtype=torch.bool, device=joint.device
                    )
                    kept = torch.empty(
                        (0, 8), dtype=torch.float32, device=joint.device
                    )
                self._received_late[str(cav_id)] = (
                    kept[:, :7].contiguous(), kept[:, 7].contiguous()
                )
                segment_records.append({
                    "name": "late_candidates",
                    "source_bytes": int(late_meta["num_bytes"]),
                    "candidate_count_source": late_count,
                    "candidate_count_received_complete": int(
                        candidate_valid.sum().item()
                    ),
                    "bytes_per_candidate": 32,
                    "atomic_records": bool(self.atomic_late_records),
                })

                link_record = {
                    "frame_id": frame_id,
                    "batch_idx": int(batch_idx),
                    "local_idx": int(local_idx),
                    "cav_id": str(cav_id),
                    "link_id": repr(link_id),
                    "payload_order": list(self.segment_order),
                    "joint_source_bytes": int(joint.numel()),
                    "segments": segment_records,
                }
                link_record.update(channel_record)
                link_records.append(link_record)

            offset += int(cav_num)

        self._link_records = link_records
        self.records.extend(copy.deepcopy(link_records))
        self.latest_info = self._build_latest_info()
        return recovered_scales

    def _build_latest_info(self):
        records = copy.deepcopy(getattr(self, "_link_records", []))
        return {
            "enabled": bool(self.enabled),
            "mode": str(self.mode),
            "executor_type": "cosdh_official_fixed_markov_joint_byte_stream",
            "packetization": "byte_stream",
            "packet_size_bytes": int(self.packet_size_bytes),
            "budget_allocation": "common_arce_fixed_equal_split",
            "budget_truncation": "packet_prefix",
            "loss_model": "bernoulli",
            "latency_model": "fixed_state_delay",
            "recovery": "zero_fill_missing_source_packets",
            "late_record_policy": "complete_32_byte_records_only",
            "segment_order": list(self.segment_order),
            "record_len": list(getattr(self, "_record_len", [])),
            "num_links": len(records),
            "source_bytes": int(sum(
                int(r.get("joint_source_bytes", 0)) for r in records
            )),
            "transmitted_wire_bytes": int(sum(
                int(r.get("transmitted_wire_bytes", 0)) for r in records
            )),
            "received_wire_bytes": int(sum(
                int(r.get("received_wire_bytes", 0)) for r in records
            )),
            "received_valid_payload_bytes": int(sum(
                int(r.get("received_valid_payload_bytes", 0)) for r in records
            )),
            "num_missing_by_budget": int(sum(
                int(r.get("num_missing_by_budget", 0)) for r in records
            )),
            "num_lost_by_bernoulli": int(sum(
                int(r.get("num_lost_by_bernoulli", 0)) for r in records
            )),
            "link_records": records,
            "ucb_arce_used": False,
            "fec_used": False,
            "redundancy_ratio": 0.0,
            "cache_used": False,
        }
