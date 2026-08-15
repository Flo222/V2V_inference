# -*- coding: utf-8 -*-
"""Read-only ideal-channel wire-volume instrumentation for OpenCOOD baselines.

The auditor uses forward hooks at each baseline's native communication boundary.
It never modifies tensors or model outputs.  Per sender->ego link, native
segments are accumulated into one byte stream and padded to packet_size_bytes.

Supported baselines:
    v2xvit, where2comm, cosdh, coopdiff, rocooper
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def _record_len_list(value: Any) -> List[int]:
    if torch.is_tensor(value):
        return [int(v) for v in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return []


def _valid_count_from_mask(mask: Any, batch_index: int, max_agents: int) -> int:
    if torch.is_tensor(mask) and mask.dim() >= 2 and batch_index < mask.shape[0]:
        row = mask[batch_index]
        if row.dim() > 1:
            row = row.reshape(row.shape[0], -1).any(dim=1)
        return min(max_agents, int(row.to(dtype=torch.bool).sum().item()))
    return max_agents


def _tensor_bytes_per_value(tensor: torch.Tensor, override: Optional[int]) -> int:
    if override is not None and int(override) > 0:
        return int(override)
    return int(tensor.element_size())


class IdealWireAuditor(object):
    """Collect ideal-channel offered payload and packetized wire bytes."""

    def __init__(
        self,
        model: torch.nn.Module,
        baseline: str,
        packet_size_bytes: int = 1024,
        bytes_per_value: Optional[int] = None,
        sparse_metadata: str = "indices",
        sparse_index_bytes: int = 4,
        prior_channels: int = 3,
        joint_link_stream: bool = True,
    ) -> None:
        self.model = model
        self.baseline = str(baseline).strip().lower()
        self.packet_size_bytes = max(1, int(packet_size_bytes))
        self.bytes_per_value = (
            None if bytes_per_value is None or int(bytes_per_value) <= 0
            else int(bytes_per_value)
        )
        self.sparse_metadata = str(sparse_metadata).strip().lower()
        if self.sparse_metadata not in ("none", "indices", "bitmask"):
            raise ValueError(
                "sparse_metadata must be one of none/indices/bitmask, got {}"
                .format(sparse_metadata)
            )
        self.sparse_index_bytes = max(0, int(sparse_index_bytes))
        self.prior_channels = max(0, int(prior_channels))
        self.joint_link_stream = bool(joint_link_stream)

        self._handles = []
        self._segments: List[Dict[str, Any]] = []
        self._frame_active = False
        self._w2c_pending: Optional[Dict[str, Any]] = None
        self._cosdh_pending: Dict[int, Dict[str, Any]] = {}

        self._install_hooks()

    def close(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles = []

    def start_frame(self) -> None:
        self._segments = []
        self._frame_active = True
        self._w2c_pending = None
        self._cosdh_pending = {}

    def finish_frame(self, frame_id: Any, sample_index: int) -> List[Dict[str, Any]]:
        self._frame_active = False
        grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
        for seg in self._segments:
            if self.joint_link_stream:
                key = (seg["batch"], seg["sender"])
            else:
                key = (seg["batch"], seg["sender"], seg["segment_name"])
            grouped[key].append(seg)

        rows: List[Dict[str, Any]] = []
        for key, segments in sorted(grouped.items(), key=lambda item: str(item[0])):
            source_bytes = int(sum(int(s["source_bytes"]) for s in segments))
            value_bytes = int(sum(int(s["value_bytes"]) for s in segments))
            metadata_bytes = int(sum(int(s["metadata_bytes"]) for s in segments))
            tx_packets = (
                int(math.ceil(source_bytes / float(self.packet_size_bytes)))
                if source_bytes > 0 else 0
            )
            tx_wire_bytes = int(tx_packets * self.packet_size_bytes)
            batch_idx = int(segments[0]["batch"])
            sender_idx = int(segments[0]["sender"])
            segment_names = [str(s["segment_name"]) for s in segments]
            rows.append({
                "sample_index": int(sample_index),
                "frame_id": frame_id,
                "batch": batch_idx,
                "cav": sender_idx,
                "sender_id": sender_idx,
                "link_key": "b{}_cav{}".format(batch_idx, sender_idx),
                "state": "ideal",
                "kind": "ideal_native_wire_audit",
                "segment_name": "+".join(segment_names),
                "segments": segments,
                "source_bytes": source_bytes,
                "source_payload_bytes": source_bytes,
                "value_bytes": value_bytes,
                "metadata_bytes": metadata_bytes,
                "packet_size_bytes": int(self.packet_size_bytes),
                "num_source_packets": tx_packets,
                "num_transmitted_packets": tx_packets,
                "num_received_packets": tx_packets,
                "transmitted_wire_bytes": tx_wire_bytes,
                "received_wire_bytes": tx_wire_bytes,
                "tx_bytes": tx_wire_bytes,
                "rx_bytes": tx_wire_bytes,
                "budget_truncated_bytes": 0,
                "wire_definition": (
                    "native sender payload, grouped per sender->ego link, then "
                    "padded to fixed-size packets"
                ),
                "sparse_metadata_mode": self.sparse_metadata,
            })
        return rows

    # ------------------------------------------------------------------
    # Generic record helpers
    # ------------------------------------------------------------------
    def _add_dense_tensor(
        self,
        tensor: torch.Tensor,
        record_len: Sequence[int],
        segment_name: str,
        channel_last: bool = False,
        feature_channels: Optional[int] = None,
    ) -> None:
        if not self._frame_active or not torch.is_tensor(tensor):
            return
        lens = [int(v) for v in record_len]
        if not lens:
            return
        bpv = _tensor_bytes_per_value(tensor, self.bytes_per_value)

        if channel_last:
            if tensor.dim() != 5:
                return
            _, max_agents, height, width, channels_total = tensor.shape
            channels = (
                int(feature_channels) if feature_channels is not None
                else int(channels_total)
            )
            per_agent_values = int(channels * height * width)
            for b, cav_num in enumerate(lens):
                for sender in range(1, min(int(cav_num), int(max_agents))):
                    value_bytes = int(per_agent_values * bpv)
                    self._add_segment(
                        b, sender, segment_name, value_bytes, 0,
                        {"channels": channels, "height": int(height), "width": int(width)}
                    )
            return

        if tensor.dim() != 4:
            return
        total, channels, height, width = tensor.shape
        if sum(lens) != int(total):
            return
        per_agent_values = int(channels * height * width)
        offset = 0
        for b, cav_num in enumerate(lens):
            for sender in range(1, int(cav_num)):
                value_bytes = int(per_agent_values * bpv)
                self._add_segment(
                    b, sender, segment_name, value_bytes, 0,
                    {"channels": int(channels), "height": int(height), "width": int(width)}
                )
            offset += int(cav_num)

    def _add_sparse_masks(
        self,
        masks: torch.Tensor,
        record_len: Sequence[int],
        channels: int,
        height: int,
        width: int,
        bytes_per_value: int,
        segment_name: str,
        fully: bool = False,
    ) -> None:
        if not self._frame_active:
            return
        lens = [int(v) for v in record_len]
        if not lens or sum(lens) <= 0:
            return

        if fully:
            resized = torch.ones(
                (sum(lens), 1, int(height), int(width)),
                dtype=torch.bool,
                device=masks.device if torch.is_tensor(masks) else "cpu",
            )
        else:
            if not torch.is_tensor(masks) or masks.dim() != 4:
                return
            resized = masks.detach().float()
            if tuple(resized.shape[-2:]) != (int(height), int(width)):
                resized = F.interpolate(
                    resized,
                    size=(int(height), int(width)),
                    mode="nearest",
                )
            resized = resized > 0.5

        offset = 0
        for b, cav_num in enumerate(lens):
            for sender in range(1, int(cav_num)):
                idx = offset + sender
                if idx >= resized.shape[0]:
                    continue
                selected_cells = int(resized[idx, 0].sum().item())
                value_bytes = int(selected_cells * int(channels) * int(bytes_per_value))
                metadata_bytes = self._sparse_metadata_bytes(
                    selected_cells=selected_cells,
                    height=int(height),
                    width=int(width),
                )
                self._add_segment(
                    b,
                    sender,
                    segment_name,
                    value_bytes,
                    metadata_bytes,
                    {
                        "selected_cells": selected_cells,
                        "channels": int(channels),
                        "height": int(height),
                        "width": int(width),
                    },
                )
            offset += int(cav_num)

    def _sparse_metadata_bytes(self, selected_cells: int, height: int, width: int) -> int:
        if selected_cells <= 0 or self.sparse_metadata == "none":
            return 0
        if self.sparse_metadata == "indices":
            return int(selected_cells * self.sparse_index_bytes)
        return int(math.ceil((int(height) * int(width)) / 8.0))

    def _add_segment(
        self,
        batch_idx: int,
        sender_idx: int,
        segment_name: str,
        value_bytes: int,
        metadata_bytes: int,
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        value_bytes = max(0, int(value_bytes))
        metadata_bytes = max(0, int(metadata_bytes))
        if value_bytes + metadata_bytes <= 0:
            return
        row = {
            "batch": int(batch_idx),
            "sender": int(sender_idx),
            "segment_name": str(segment_name),
            "value_bytes": value_bytes,
            "metadata_bytes": metadata_bytes,
            "source_bytes": int(value_bytes + metadata_bytes),
        }
        if detail:
            row.update(detail)
        self._segments.append(row)

    # ------------------------------------------------------------------
    # Hook installation
    # ------------------------------------------------------------------
    def _install_hooks(self) -> None:
        if self.baseline == "v2xvit":
            self._install_v2xvit()
        elif self.baseline == "where2comm":
            self._install_where2comm()
        elif self.baseline == "cosdh":
            self._install_cosdh()
        elif self.baseline == "coopdiff":
            self._install_coopdiff()
        elif self.baseline == "rocooper":
            self._install_rocooper()
        else:
            raise ValueError("Unsupported baseline: {}".format(self.baseline))

    def _install_v2xvit(self) -> None:
        fusion = getattr(self.model, "fusion_net", None)
        if fusion is None:
            raise AttributeError("V2X-ViT model has no fusion_net")

        def pre_hook(module, inputs):
            del module
            if not self._frame_active or len(inputs) < 2:
                return
            feature, valid_mask = inputs[0], inputs[1]
            if not torch.is_tensor(feature) or feature.dim() != 5:
                return
            batch, max_agents = int(feature.shape[0]), int(feature.shape[1])
            lens = [
                _valid_count_from_mask(valid_mask, b, max_agents)
                for b in range(batch)
            ]
            channels = max(0, int(feature.shape[-1]) - self.prior_channels)
            self._add_dense_tensor(
                feature,
                lens,
                segment_name="v2xvit_native_feature",
                channel_last=True,
                feature_channels=channels,
            )

        self._handles.append(fusion.register_forward_pre_hook(pre_hook))

    def _install_where2comm(self) -> None:
        fusion = getattr(self.model, "fusion_net", None)
        if fusion is None:
            raise AttributeError("Where2Comm model has no fusion_net")
        communication = getattr(fusion, "naive_communication", None)
        if communication is None:
            raise AttributeError("Where2Comm fusion has no naive_communication")

        def fusion_pre(module, inputs):
            del module
            if not self._frame_active or len(inputs) < 3:
                return
            x, _, record_len = inputs[:3]
            if not torch.is_tensor(x):
                return
            self._w2c_pending = {
                "record_len": _record_len_list(record_len),
                "shape": tuple(int(v) for v in x.shape),
                "mask_seen": False,
                "bytes_per_value": _tensor_bytes_per_value(x, self.bytes_per_value),
            }

        def block_hook(module, inputs, output):
            del module, inputs
            if self._w2c_pending is not None and torch.is_tensor(output) and output.dim() == 4:
                self._w2c_pending["shape"] = tuple(int(v) for v in output.shape)
                self._w2c_pending["bytes_per_value"] = _tensor_bytes_per_value(
                    output, self.bytes_per_value
                )

        def comm_hook(module, inputs, output):
            del module, inputs
            pending = self._w2c_pending
            if not self._frame_active or pending is None:
                return
            masks = output[0] if isinstance(output, (list, tuple)) else None
            shape = pending.get("shape", ())
            if len(shape) != 4:
                return
            _, channels, height, width = shape
            self._add_sparse_masks(
                masks=masks,
                record_len=pending["record_len"],
                channels=int(channels),
                height=int(height),
                width=int(width),
                bytes_per_value=int(pending["bytes_per_value"]),
                segment_name="where2comm_selected_feature",
                fully=bool(getattr(fusion, "fully", False)),
            )
            pending["mask_seen"] = True

        def fusion_post(module, inputs, output):
            del module, inputs, output
            pending = self._w2c_pending
            if not self._frame_active or pending is None or pending.get("mask_seen"):
                self._w2c_pending = None
                return
            shape = pending.get("shape", ())
            if len(shape) == 4 and bool(getattr(fusion, "fully", False)):
                total, channels, height, width = shape
                masks = torch.ones((int(total), 1, int(height), int(width)))
                self._add_sparse_masks(
                    masks=masks,
                    record_len=pending["record_len"],
                    channels=int(channels),
                    height=int(height),
                    width=int(width),
                    bytes_per_value=int(pending["bytes_per_value"]),
                    segment_name="where2comm_full_feature",
                    fully=True,
                )
            self._w2c_pending = None

        self._handles.append(fusion.register_forward_pre_hook(fusion_pre))
        self._handles.append(communication.register_forward_hook(comm_hook))
        self._handles.append(fusion.register_forward_hook(fusion_post))

        if bool(getattr(fusion, "multi_scale", False)):
            backbone = getattr(self.model, "backbone", None)
            blocks = getattr(backbone, "blocks", None)
            if blocks is not None and len(blocks) > 0:
                self._handles.append(blocks[0].register_forward_hook(block_hook))

    def _install_cosdh(self) -> None:
        fusion_modules = getattr(self.model, "fusion_net", None)
        if fusion_modules is None:
            raise AttributeError("CoSDH model has no fusion_net")
        modules = list(fusion_modules) if isinstance(fusion_modules, Iterable) else [fusion_modules]

        for scale_idx, fuse_module in enumerate(modules):
            communication = getattr(fuse_module, "naive_communication", None)
            if communication is None:
                continue

            def pre_hook(module, inputs, scale_idx=scale_idx):
                if not self._frame_active or len(inputs) < 3:
                    return
                x, _, record_len = inputs[:3]
                if not torch.is_tensor(x) or x.dim() != 4:
                    return
                self._cosdh_pending[id(module)] = {
                    "scale_idx": int(scale_idx),
                    "record_len": _record_len_list(record_len),
                    "shape": tuple(int(v) for v in x.shape),
                    "bytes_per_value": _tensor_bytes_per_value(x, self.bytes_per_value),
                    "fully": bool(getattr(module, "fully", False)),
                }

            def comm_hook(comm_module, inputs, output, parent=fuse_module):
                del comm_module, inputs
                pending = self._cosdh_pending.get(id(parent))
                if not self._frame_active or pending is None:
                    return
                masks = output[0] if isinstance(output, (list, tuple)) else None
                shape = pending["shape"]
                _, channels, height, width = shape
                self._add_sparse_masks(
                    masks=masks,
                    record_len=pending["record_len"],
                    channels=int(channels),
                    height=int(height),
                    width=int(width),
                    bytes_per_value=int(pending["bytes_per_value"]),
                    segment_name="cosdh_scale{}".format(pending["scale_idx"]),
                    fully=bool(pending["fully"]),
                )

            self._handles.append(fuse_module.register_forward_pre_hook(pre_hook))
            self._handles.append(communication.register_forward_hook(comm_hook))

        if not self._handles:
            raise RuntimeError("No CoSDH communication hooks were installed")

    def _install_coopdiff(self) -> None:
        fuse_modules = getattr(self.model, "fuse_modules", None)
        if fuse_modules is None:
            raise AttributeError("CoopDiff model has no fuse_modules")
        modules = list(fuse_modules) if isinstance(fuse_modules, torch.nn.ModuleList) else [fuse_modules]

        for scale_idx, module in enumerate(modules):
            def pre_hook(fuse_module, inputs, scale_idx=scale_idx):
                del fuse_module
                if not self._frame_active or len(inputs) < 2:
                    return
                feature, record_len = inputs[:2]
                self._add_dense_tensor(
                    feature,
                    _record_len_list(record_len),
                    segment_name="coopdiff_scale{}".format(scale_idx),
                )
            self._handles.append(module.register_forward_pre_hook(pre_hook))

    def _install_rocooper(self) -> None:
        comm_module = getattr(self.model, "comm_module", None)
        if comm_module is None:
            raise AttributeError("RoCooper model has no comm_module")

        def pre_hook(module, inputs):
            del module
            if not self._frame_active or len(inputs) < 2:
                return
            feature, record_len = inputs[:2]
            self._add_dense_tensor(
                feature,
                _record_len_list(record_len),
                segment_name="rocooper_comm_input",
            )

        self._handles.append(comm_module.register_forward_pre_hook(pre_hook))
