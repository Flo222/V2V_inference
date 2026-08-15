"""Read-only compression and budget audit for ARCE experiments.

The auditor records one JSON object for every sender -> ego communication link.
It can be used in two modes without changing inference tensors:

Experiment 1 (pure compression):
    source -> quantize/dequantize -> lossless packet transport

Experiment 2 (limited budget, no packet loss/FEC):
    source -> quantize/dequantize -> source-packet budget truncation -> recovery

All tensors are detached before statistics or snapshots are computed. The
module does not sample randomness, write into pipeline tensors, update caches,
or change the communication result returned to the fusion model.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def _safe_name(value: Any) -> str:
    text = str(value)
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text)
    return text[:160] if text else "unknown"


def _tensor_num_bytes(x: Optional[torch.Tensor]) -> int:
    if x is None or not torch.is_tensor(x):
        return 0
    return int(x.numel() * x.element_size())


def _tensor_summary(x: Optional[torch.Tensor]) -> Dict[str, Any]:
    if x is None or not torch.is_tensor(x):
        return {"available": False}

    y = x.detach()
    result: Dict[str, Any] = {
        "available": True,
        "shape": [int(v) for v in y.shape],
        "dtype": str(y.dtype),
        "numel": int(y.numel()),
        "num_bytes": _tensor_num_bytes(y),
    }
    if y.numel() == 0:
        result.update(
            {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "abs_max": 0.0,
                "zero_ratio": 0.0,
            }
        )
        return result

    z = y.float()
    result.update(
        {
            "min": float(z.min().item()),
            "max": float(z.max().item()),
            "mean": float(z.mean().item()),
            "std": float(z.std(unbiased=False).item()),
            "abs_max": float(z.abs().max().item()),
            "zero_ratio": float((z == 0).float().mean().item()),
        }
    )
    return result


def _pair_metrics(
    reference: Optional[torch.Tensor],
    candidate: Optional[torch.Tensor],
    eps: float = 1e-12,
) -> Dict[str, Any]:
    if reference is None or candidate is None:
        return {"available": False, "reason": "missing_tensor"}
    if not torch.is_tensor(reference) or not torch.is_tensor(candidate):
        return {"available": False, "reason": "not_tensor"}
    if tuple(reference.shape) != tuple(candidate.shape):
        return {
            "available": False,
            "reason": "shape_mismatch",
            "reference_shape": [int(v) for v in reference.shape],
            "candidate_shape": [int(v) for v in candidate.shape],
        }

    a_raw = reference.detach()
    b_raw = candidate.detach()
    a = a_raw.float().reshape(-1)
    b = b_raw.float().reshape(-1)

    if a.numel() == 0:
        return {
            "available": True,
            "mse": 0.0,
            "nmse": 0.0,
            "mae": 0.0,
            "max_abs_error": 0.0,
            "cosine_similarity": 1.0,
            "exact_equal": True,
            "allclose": True,
        }

    diff = a - b
    mse = torch.mean(diff * diff)
    energy = torch.mean(a * a)
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom.item()) <= eps:
        cosine = 1.0 if torch.equal(a_raw, b_raw) else 0.0
    else:
        cosine = float(torch.dot(a, b).div(denom).item())

    return {
        "available": True,
        "mse": float(mse.item()),
        "nmse": float((mse / energy.clamp_min(eps)).item()),
        "mae": float(diff.abs().mean().item()),
        "max_abs_error": float(diff.abs().max().item()),
        "cosine_similarity": float(cosine),
        "exact_equal": bool(torch.equal(a_raw, b_raw)),
        "allclose": bool(torch.allclose(a_raw, b_raw, rtol=1e-5, atol=1e-6)),
    }


def _mask_to_ranges(mask: torch.Tensor) -> List[List[int]]:
    """Compress a 1-D boolean mask into [start, end) ranges."""
    if mask is None or not torch.is_tensor(mask):
        return []
    values = mask.detach().to(dtype=torch.bool, device="cpu").flatten().tolist()
    ranges: List[List[int]] = []
    start: Optional[int] = None
    for index, selected in enumerate(values):
        if selected and start is None:
            start = int(index)
        elif not selected and start is not None:
            ranges.append([int(start), int(index)])
            start = None
    if start is not None:
        ranges.append([int(start), int(len(values))])
    return ranges


def _summary_1d(values: torch.Tensor, eps: float = 1e-12) -> Dict[str, Any]:
    if values is None or not torch.is_tensor(values) or values.numel() == 0:
        return {"available": False}
    x = values.detach().float().cpu().flatten()
    full = x >= (1.0 - eps)
    zero = x <= eps
    partial = ~(full | zero)
    nz = torch.nonzero(x > eps, as_tuple=False).flatten()
    z = torch.nonzero(x <= eps, as_tuple=False).flatten()
    return {
        "available": True,
        "length": int(x.numel()),
        "mean": float(x.mean().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "num_fully_retained": int(full.sum().item()),
        "num_partially_retained": int(partial.sum().item()),
        "num_zero_retained": int(zero.sum().item()),
        "last_nonzero_index": int(nz[-1].item()) if nz.numel() else None,
        "first_zero_index": int(z[0].item()) if z.numel() else None,
    }


def _analyze_source_retention(
    *,
    source_feature: torch.Tensor,
    source_tx_mask: torch.Tensor,
    packet_size_bytes: int,
    valid_stream_bytes: int,
    source_tensor_kind: str,
    stream_tensor: torch.Tensor,
) -> Dict[str, Any]:
    """Map selected source packets back to original feature values.

    The byte-stream packetizer serializes the quantized tensor in contiguous
    row-major order. For FP32/FP16/INT8, a value is counted as retained only
    when all bytes of that value are selected. For packed INT4, each selected
    byte retains its two 4-bit values (the final byte may contain one value).

    This function is diagnostic only; it does not alter packets or features.
    """
    if source_tx_mask is None or not torch.is_tensor(source_tx_mask):
        return {"available": False, "reason": "missing_source_tx_mask"}
    if source_feature is None or not torch.is_tensor(source_feature):
        return {"available": False, "reason": "missing_source_feature"}

    mask_cpu = source_tx_mask.detach().to(dtype=torch.bool, device="cpu").flatten()
    packet_ranges = _mask_to_ranges(mask_cpu)
    num_values = int(source_feature.numel())
    packet_size_bytes = int(packet_size_bytes)
    valid_stream_bytes = int(valid_stream_bytes)
    if packet_size_bytes <= 0 or valid_stream_bytes < 0:
        return {"available": False, "reason": "invalid_packet_metadata"}

    retained = torch.zeros((num_values,), dtype=torch.bool)
    selected_value_ranges: List[List[int]] = []
    selected_byte_ranges: List[List[int]] = []

    packed_int4 = str(source_tensor_kind) == "packed_int4"
    element_size = 1 if packed_int4 else int(stream_tensor.element_size())

    for packet_start, packet_end in packet_ranges:
        byte_start = int(packet_start * packet_size_bytes)
        byte_end = int(min(valid_stream_bytes, packet_end * packet_size_bytes))
        if byte_end <= byte_start:
            continue
        selected_byte_ranges.append([byte_start, byte_end])

        if packed_int4:
            value_start = int(min(num_values, byte_start * 2))
            value_end = int(min(num_values, byte_end * 2))
        else:
            # Count only values whose complete byte representation is selected.
            value_start = int(min(num_values, (byte_start + element_size - 1) // element_size))
            value_end = int(min(num_values, byte_end // element_size))
        if value_end > value_start:
            retained[value_start:value_end] = True
            selected_value_ranges.append([value_start, value_end])

    retained_value_count = int(retained.sum().item())
    result: Dict[str, Any] = {
        "available": True,
        "source_shape": [int(v) for v in source_feature.shape],
        "source_num_values": num_values,
        "source_tensor_kind": str(source_tensor_kind),
        "storage_element_size_bytes": int(element_size),
        "packet_size_bytes": packet_size_bytes,
        "valid_stream_bytes": valid_stream_bytes,
        "selected_source_packet_count": int(mask_cpu.sum().item()),
        "selected_source_packet_ranges": packet_ranges,
        "selected_byte_ranges": selected_byte_ranges,
        "selected_value_ranges": selected_value_ranges,
        "retained_value_count": retained_value_count,
        "retained_value_ratio": float(retained_value_count / max(1, num_values)),
        "packet_selection_is_prefix": bool(
            len(packet_ranges) == 0
            or (
                len(packet_ranges) == 1
                and packet_ranges[0][0] == 0
                and packet_ranges[0][1] == int(mask_cpu.sum().item())
            )
        ),
    }

    # Support [C,H,W] and [1,C,H,W]. Other layouts still get flat retention.
    shape = tuple(int(v) for v in source_feature.shape)
    if len(shape) == 3:
        c, h, w = shape
        retained_chw = retained.reshape(c, h, w)
    elif len(shape) == 4 and shape[0] == 1:
        _, c, h, w = shape
        retained_chw = retained.reshape(shape)[0]
    else:
        result.update({
            "layout_supported": False,
            "layout_reason": "expected_[C,H,W]_or_[1,C,H,W]",
            "retained_value_mask": retained,
        })
        return result

    channel_retention = retained_chw.float().mean(dim=(1, 2))
    spatial_retention = retained_chw.float().mean(dim=0)
    channel_summary = _summary_1d(channel_retention)
    spatial_flat_summary = _summary_1d(spatial_retention.flatten())
    half = max(1, int(c // 2))
    front_mean = float(channel_retention[:half].mean().item())
    back_mean = float(channel_retention[half:].mean().item()) if half < c else front_mean

    result.update({
        "layout_supported": True,
        "channel_count": int(c),
        "height": int(h),
        "width": int(w),
        "channel_retention_ratio": [float(v) for v in channel_retention.tolist()],
        "channel_summary": channel_summary,
        "spatial_summary": spatial_flat_summary,
        "front_half_channel_retention_mean": front_mean,
        "back_half_channel_retention_mean": back_mean,
        "channel_front_back_bias": float(front_mean - back_mean),
        # Tensor fields are removed before JSON output and may be included in snapshots.
        "retained_value_mask": retained,
        "spatial_retention_ratio_tensor": spatial_retention,
        "channel_retention_ratio_tensor": channel_retention,
    })
    return result


class CompressionAuditor:
    """Write compression/budget records without changing inference."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        self.cfg = dict(cfg)
        self.enabled = _as_bool(cfg.get("enabled", False))
        self.strict = _as_bool(cfg.get("strict", False))
        self.experiment_name = str(
            cfg.get("experiment_name", "experiment1_pure_compression_correctness")
        )
        self.require_no_budget_drop = _as_bool(
            cfg.get("require_no_budget_drop", True)
        )
        self.require_no_bernoulli_loss = _as_bool(
            cfg.get("require_no_bernoulli_loss", True)
        )
        self.require_no_fec_parity = _as_bool(
            cfg.get("require_no_fec_parity", True)
        )
        self.require_all_source_transmitted = _as_bool(
            cfg.get("require_all_source_transmitted", True)
        )
        self.require_quant_equals_recovered = _as_bool(
            cfg.get("require_quant_equals_recovered", True)
        )
        self.output_dir = os.path.abspath(
            os.path.expanduser(str(cfg.get("output_dir", "audit_runs/compression")))
        )
        self.file_name = str(cfg.get("file_name", "compression_audit.jsonl"))
        self.save_tensors = _as_bool(cfg.get("save_tensors", False))
        self.save_first_n_links = max(0, int(cfg.get("save_first_n_links", 0)))
        self._snapshot_count = 0
        self._record_count = 0
        self._jsonl_path = os.path.join(self.output_dir, self.file_name)
        self._snapshot_dir = os.path.join(self.output_dir, "tensor_snapshots")

        if self.enabled:
            os.makedirs(self.output_dir, exist_ok=True)
            if self.save_tensors:
                os.makedirs(self._snapshot_dir, exist_ok=True)
            with open(self._jsonl_path, "w", encoding="utf-8") as f:
                f.write("")

    def reset(self) -> None:
        self._snapshot_count = 0
        self._record_count = 0

    def _write_json(self, record: Dict[str, Any]) -> None:
        with open(self._jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write("\n")

    def _save_snapshot(
        self,
        frame_id: Any,
        ego_index: int,
        agent_index: int,
        quant_mode: str,
        feature_input: torch.Tensor,
        source_feature: torch.Tensor,
        quant_dequantized: torch.Tensor,
        recovered_compact: torch.Tensor,
        recovered_dense: torch.Tensor,
        stream_tensor: torch.Tensor,
        retention_analysis: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if not self.save_tensors:
            return None
        if self._snapshot_count >= self.save_first_n_links:
            return None

        name = (
            "frame_%s_ego_%s_sender_%s_%s_%04d.pt"
            % (
                _safe_name(frame_id),
                int(ego_index),
                int(agent_index),
                _safe_name(quant_mode),
                int(self._snapshot_count),
            )
        )
        path = os.path.join(self._snapshot_dir, name)
        payload = {
            "frame_id": frame_id,
            "ego_index": int(ego_index),
            "agent_index": int(agent_index),
            "quant_mode": str(quant_mode),
            "feature_input_dense": feature_input.detach().cpu().clone(),
            "source_payload_before_quant": source_feature.detach().cpu().clone(),
            "quantized_then_dequantized": quant_dequantized.detach().cpu().clone(),
            "recovered_payload_compact": recovered_compact.detach().cpu().clone(),
            "recovered_feature_dense": recovered_dense.detach().cpu().clone(),
            "transmitted_storage_tensor": stream_tensor.detach().cpu().clone(),
        }
        if retention_analysis and retention_analysis.get("available", False):
            for key in (
                "retained_value_mask",
                "spatial_retention_ratio_tensor",
                "channel_retention_ratio_tensor",
            ):
                value = retention_analysis.get(key)
                if torch.is_tensor(value):
                    payload[key] = value.detach().cpu().clone()
            payload["selected_source_packet_ranges"] = retention_analysis.get(
                "selected_source_packet_ranges", []
            )
            payload["selected_value_ranges"] = retention_analysis.get(
                "selected_value_ranges", []
            )
        torch.save(payload, path)
        self._snapshot_count += 1
        return path

    def record(
        self,
        *,
        frame_id: Any,
        link_id: Any,
        agent_index: int,
        ego_index: int,
        requested_quant_mode: str,
        actual_quant_mode: str,
        source_tensor_kind: str,
        feature_input: torch.Tensor,
        source_feature: torch.Tensor,
        quant_dequantized: torch.Tensor,
        recovered_compact: torch.Tensor,
        recovered_dense: torch.Tensor,
        stream_tensor: torch.Tensor,
        packet_result: Any,
        source_tx_mask: torch.Tensor,
        budget_accounting: Optional[Dict[str, Any]] = None,
        comm_record: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        try:
            size = comm_record.get("size", {}) or {}
            packet_size = int(getattr(packet_result, "packet_size_bytes", 0))
            num_source_packets = int(getattr(packet_result, "num_packets", 0))
            valid_stream_bytes = int(getattr(packet_result, "original_num_bytes", 0))
            padded_source_bytes = int(num_source_packets * packet_size)
            source_fp32_reference_bytes = int(source_feature.numel() * 4)

            quant_metrics = _pair_metrics(source_feature, quant_dequantized)
            transport_metrics = _pair_metrics(quant_dequantized, recovered_compact)
            end_to_end_metrics = _pair_metrics(source_feature, recovered_compact)

            requested = str(requested_quant_mode).strip().lower()
            actual = str(actual_quant_mode).strip().lower()
            # Budget accounting is passed directly from communicate_feature's
            # runtime locals.  This is authoritative for both dense and
            # method-native/compact-sparse payloads and avoids silently losing
            # fields when a transport adapter rewrites the communication record.
            runtime_accounting = dict(budget_accounting or {})

            def _pick(name: str, size_name: Optional[str] = None, default: Any = 0) -> Any:
                if name in runtime_accounting:
                    return runtime_accounting[name]
                return size.get(size_name or name, default)

            actual_tx_bytes = int(round(float(_pick(
                "actual_transmitted_bytes", default=0.0
            ))))
            budget_bytes = float(_pick(
                "bandwidth_budget_bytes", default=0.0
            ) or 0.0)

            runtime_num_source = int(_pick(
                "num_source_packets", "actual_num_source_packets", num_source_packets
            ))
            num_parity_packets = int(_pick(
                "num_parity_packets", "actual_num_parity_packets", 0
            ))
            num_encoded_packets = int(_pick(
                "num_encoded_packets", "actual_num_encoded_packets",
                runtime_num_source + num_parity_packets
            ))
            num_admitted_source = int(_pick(
                "num_admitted_source_packets",
                "actual_num_admitted_source_packets",
                int(source_tx_mask.to(dtype=torch.bool).sum().item()),
            ))
            num_tx_source = int(_pick(
                "num_transmitted_source_packets",
                "actual_num_transmitted_source_packets",
                int(source_tx_mask.to(dtype=torch.bool).sum().item()),
            ))
            num_tx_parity = int(_pick(
                "num_transmitted_parity_packets",
                "actual_num_transmitted_parity_packets",
                0,
            ))
            num_source_drop = int(_pick(
                "num_source_dropped_by_budget", default=(num_source_packets - num_tx_source)
            ))
            num_parity_drop = int(_pick(
                "num_parity_dropped_by_budget", default=(num_parity_packets - num_tx_parity)
            ))
            num_total_drop = int(_pick(
                "num_missing_by_budget",
                default=(num_source_drop + num_parity_drop),
            ))

            runtime_accounting_complete = bool(
                budget_accounting is not None
                and all(key in runtime_accounting for key in (
                    "bandwidth_budget_bytes",
                    "num_source_packets",
                    "num_admitted_source_packets",
                    "num_parity_packets",
                    "num_encoded_packets",
                    "num_transmitted_source_packets",
                    "num_transmitted_parity_packets",
                    "num_source_dropped_by_budget",
                    "num_parity_dropped_by_budget",
                    "num_missing_by_budget",
                    "actual_transmitted_bytes",
                ))
            )
            accounting_valid = bool(
                runtime_num_source == num_source_packets
                and num_encoded_packets == runtime_num_source + num_parity_packets
                and 0 <= num_admitted_source <= num_source_packets
                and num_admitted_source == num_tx_source
                and num_tx_source + num_source_drop == num_source_packets
                and num_tx_parity + num_parity_drop == num_parity_packets
                and num_source_drop + num_parity_drop == num_total_drop
                and num_tx_source == int(source_tx_mask.to(dtype=torch.bool).sum().item())
            )

            checks = {
                "requested_matches_actual_quant": bool(requested == actual),
                "frame_id_present": frame_id is not None,
                "budget_packet_accounting_valid": accounting_valid,
                "runtime_budget_accounting_complete": runtime_accounting_complete,
                "int4_is_packed": bool(
                    requested != "int4" or source_tensor_kind == "packed_int4"
                ),
            }
            if self.require_no_fec_parity:
                checks["no_fec_parity"] = num_parity_packets == 0
            if self.require_no_budget_drop:
                checks["no_budget_drop"] = num_total_drop == 0
            if self.require_no_bernoulli_loss:
                checks["no_bernoulli_loss"] = int(
                    _pick("num_lost_by_bernoulli", default=0)
                ) == 0
            if self.require_all_source_transmitted:
                checks["all_source_packets_transmitted"] = bool(
                    num_tx_source == num_source_packets
                    and actual_tx_bytes == padded_source_bytes
                )
            if self.require_quant_equals_recovered:
                checks["quant_equals_recovered"] = bool(
                    transport_metrics.get("available", False)
                    and transport_metrics.get("allclose", False)
                )
            checks["passed"] = bool(all(checks.values()))

            retention_analysis = _analyze_source_retention(
                source_feature=source_feature,
                source_tx_mask=source_tx_mask,
                packet_size_bytes=packet_size,
                valid_stream_bytes=valid_stream_bytes,
                source_tensor_kind=source_tensor_kind,
                stream_tensor=stream_tensor,
            )

            snapshot_path = self._save_snapshot(
                frame_id=frame_id,
                ego_index=ego_index,
                agent_index=agent_index,
                quant_mode=requested,
                feature_input=feature_input,
                source_feature=source_feature,
                quant_dequantized=quant_dequantized,
                recovered_compact=recovered_compact,
                recovered_dense=recovered_dense,
                stream_tensor=stream_tensor,
                retention_analysis=retention_analysis,
            )

            source_tx_ratio = float(num_tx_source / max(1, num_source_packets))
            source_recovery_ratio = float(
                int(_pick("num_recovered_source_packets", default=0))
                / max(1, num_source_packets)
            )
            budget_utilization = (
                float(actual_tx_bytes / budget_bytes)
                if math.isfinite(budget_bytes) and budget_bytes > 0.0
                else None
            )

            retention_json = dict(retention_analysis)
            for tensor_key in (
                "retained_value_mask",
                "spatial_retention_ratio_tensor",
                "channel_retention_ratio_tensor",
            ):
                retention_json.pop(tensor_key, None)

            audit_record: Dict[str, Any] = {
                "experiment": self.experiment_name,
                "frame_id": frame_id,
                "link_id": str(link_id),
                "ego_index": int(ego_index),
                "agent_index": int(agent_index),
                "requested_quant_mode": requested,
                "actual_quant_mode": actual,
                "source_tensor_kind": str(source_tensor_kind),
                "feature_input_dense": _tensor_summary(feature_input),
                "source_payload_before_quant": _tensor_summary(source_feature),
                "quantized_then_dequantized": _tensor_summary(quant_dequantized),
                "transmitted_storage_tensor": _tensor_summary(stream_tensor),
                "recovered_payload_compact": _tensor_summary(recovered_compact),
                "recovered_feature_dense": _tensor_summary(recovered_dense),
                "sizes": {
                    "source_payload_actual_dtype_bytes": _tensor_num_bytes(source_feature),
                    "source_payload_fp32_reference_bytes": source_fp32_reference_bytes,
                    "quantized_valid_stream_bytes": valid_stream_bytes,
                    "source_packet_count": num_source_packets,
                    "packet_size_bytes": packet_size,
                    "padded_source_packet_bytes": padded_source_bytes,
                    "padding_bytes": int(max(0, padded_source_bytes - valid_stream_bytes)),
                    "bandwidth_budget_bytes": budget_bytes,
                    "actual_transmitted_bytes": actual_tx_bytes,
                    "actual_transmitted_source_bytes": int(
                        round(float(_pick("actual_transmitted_source_bytes", default=0.0)))
                    ),
                    "actual_transmitted_parity_bytes": int(
                        round(float(_pick("actual_transmitted_parity_bytes", default=0.0)))
                    ),
                    "compression_ratio_vs_fp32": float(
                        source_fp32_reference_bytes / max(1, valid_stream_bytes)
                    ),
                    "budget_utilization": budget_utilization,
                },
                "quantization_error": quant_metrics,
                # Kept for backward compatibility with Experiment-1 summary.
                "clean_transport_error": transport_metrics,
                "budget_transport_error": transport_metrics,
                "end_to_end_error": end_to_end_metrics,
                "channel": {
                    "state": comm_record.get("channel_state"),
                    "plr": float(
                        ((comm_record.get("channel", {}) or {}).get("loss", {}) or {}).get(
                            "plr", 0.0
                        )
                    ),
                },
                "packet_outcome": {
                    "num_source_packets": num_source_packets,
                    "num_parity_packets": num_parity_packets,
                    "num_encoded_packets": num_encoded_packets,
                    "num_transmitted_source_packets": num_tx_source,
                    "num_transmitted_parity_packets": num_tx_parity,
                    "num_source_dropped_by_budget": num_source_drop,
                    "num_parity_dropped_by_budget": num_parity_drop,
                    "num_missing_by_budget": num_total_drop,
                    "num_lost_by_bernoulli": int(
                        _pick("num_lost_by_bernoulli", default=0)
                    ),
                    "num_direct_received_source_packets": int(
                        _pick("num_direct_received_source_packets", default=0)
                    ),
                    "num_fec_recovered_source_packets": int(
                        _pick("num_fec_recovered_source_packets", default=0)
                    ),
                    "num_recovered_source_packets": int(
                        _pick("num_recovered_source_packets", default=0)
                    ),
                    "num_missing_source_packets": int(
                        _pick("num_missing_source_packets", default=0)
                    ),
                    "source_tx_ratio": source_tx_ratio,
                    "source_recovery_ratio": source_recovery_ratio,
                },
                "budget_accounting": {
                    "source": "runtime_locals" if budget_accounting is not None else "comm_record_size_fallback",
                    "runtime_complete": runtime_accounting_complete,
                    "num_source_packets": num_source_packets,
                    "num_parity_packets": num_parity_packets,
                    "num_encoded_packets": num_encoded_packets,
                    "num_transmitted_source_packets": num_tx_source,
                    "num_transmitted_parity_packets": num_tx_parity,
                    "num_source_dropped_by_budget": num_source_drop,
                    "num_parity_dropped_by_budget": num_parity_drop,
                    "num_missing_by_budget": num_total_drop,
                    "bandwidth_budget_bytes": budget_bytes,
                    "actual_transmitted_bytes": actual_tx_bytes,
                },
                "budget_retention": retention_json,
                "sanity": checks,
                "snapshot_path": snapshot_path,
            }
            self._write_json(audit_record)
            self._record_count += 1

            return {
                "requested_quant_mode": requested,
                "actual_quant_mode": actual,
                "quant_nmse": float(quant_metrics.get("nmse", math.nan)),
                "quant_cosine_similarity": float(
                    quant_metrics.get("cosine_similarity", math.nan)
                ),
                "clean_transport_nmse": float(
                    transport_metrics.get("nmse", math.nan)
                ),
                "clean_transport_allclose": bool(
                    transport_metrics.get("allclose", False)
                ),
                "source_payload_fp32_reference_bytes": source_fp32_reference_bytes,
                "quantized_valid_stream_bytes": valid_stream_bytes,
                "source_packet_count": num_source_packets,
                "num_transmitted_source_packets": num_tx_source,
                "num_source_dropped_by_budget": num_source_drop,
                "num_parity_dropped_by_budget": num_parity_drop,
                "source_tx_ratio": source_tx_ratio,
                "source_recovery_ratio": source_recovery_ratio,
                "retained_value_ratio": float(
                    retention_analysis.get("retained_value_ratio", math.nan)
                ),
                "channel_front_back_bias": float(
                    retention_analysis.get("channel_front_back_bias", math.nan)
                ),
                "padding_bytes": int(max(0, padded_source_bytes - valid_stream_bytes)),
                "sanity_passed": bool(checks["passed"]),
            }
        except Exception as exc:
            error_record = {
                "experiment": self.experiment_name,
                "frame_id": frame_id,
                "link_id": str(link_id),
                "ego_index": int(ego_index),
                "agent_index": int(agent_index),
                "error": "%s: %s" % (type(exc).__name__, str(exc)),
            }
            try:
                self._write_json(error_record)
            except Exception:
                pass
            if self.strict:
                raise
            return {"error": error_record["error"], "sanity_passed": False}
