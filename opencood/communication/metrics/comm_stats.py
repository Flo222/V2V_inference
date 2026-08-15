"""
Communication statistics utilities for ARCE / OpenCOOD experiments.

This module aggregates per-link communication records produced by:

    opencood.methods.arce.arce_fixed_comm.ARCEFixedComm

It provides:
    1. online communication metric accumulation;
    2. per-frame / per-channel / per-quant / per-FEC summaries;
    3. flat record conversion for csv/json logging;
    4. compact experiment-level communication summary.

This module does NOT:
    - write files to disk;
    - run detection evaluation;
    - modify feature tensors;
    - perform communication simulation.

File writing should be handled by comm_logger.py later.

Typical usage:

    stats = CommStats()

    for record in arce_comm.get_records():
        stats.add_record(record)

    summary = stats.summarize()
    by_channel = stats.summarize_by_channel_state()
    by_fec = stats.summarize_by_fec_type()

Or directly:

    stats = CommStats.from_arce_comm(arce_comm)
    summary = stats.summarize()
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from opencood.communication.metrics import (
    METRIC_CHANNEL_STATE,
    METRIC_QUANT_MODE,
    METRIC_FEC_TYPE,
    METRIC_LATE,
    METRIC_TRANSMITTED_BYTES,
    METRIC_RECEIVED_BYTES,
    METRIC_NUM_SOURCE_PACKETS,
    METRIC_NUM_ENCODED_PACKETS,
    METRIC_NUM_LOST_PACKETS,
    METRIC_NUM_FEC_RECOVERED,
    METRIC_NUM_TEMPORAL_FILLED,
    METRIC_NUM_SPATIAL_FILLED,
    METRIC_NUM_ZERO_FILLED,
    METRIC_NUM_STILL_MISSING,
    METRIC_PACKET_LOSS_RATIO,
    METRIC_RECOVERY_RATIO,
    DEFAULT_SUM_METRICS,
    DEFAULT_MEAN_METRICS,
    DEFAULT_COUNT_METRICS,
    extract_record_metrics,
    flatten_dict,
    init_metric_accumulator,
    update_metric_accumulator,
    finalize_metric_accumulator,
    summarize_records,
    summarize_by_key,
    safe_divide,
    bytes_to_mb,
    is_number,
    to_float,
    to_int,
)


def _json_safe(value: Any) -> Any:
    """
    Convert common Python objects into JSON-friendly values.

    This is intentionally conservative and avoids importing numpy / torch.
    """
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    return str(value)


def _record_frame_key(metrics: Dict[str, Any]) -> str:
    """
    Return frame grouping key from flat metrics.
    """
    frame_id = metrics.get("frame_id", None)
    return "None" if frame_id is None else str(frame_id)


def _record_group_key(metrics: Dict[str, Any], key: str) -> str:
    """
    Return string group key from flat metrics.
    """
    value = metrics.get(key, None)
    return "None" if value is None else str(value)


def _is_bypassed(metrics: Dict[str, Any]) -> bool:
    """
    Return whether a flat metric record is bypassed.
    """
    return bool(metrics.get("bypassed", False))


@dataclass
class CommStatsSummary:
    """
    Container for communication statistics summary.

    Attributes
    ----------
    overall : dict
        Overall summary across all records.

    by_frame : dict
        Per-frame summary.

    by_channel_state : dict
        Per-channel-state summary.

    by_quant_mode : dict
        Per-quant-mode summary.

    by_fec_type : dict
        Per-FEC-type summary.

    extra : dict
        Additional metadata.
    """

    overall: Dict[str, Any]
    by_frame: Dict[str, Any] = field(default_factory=dict)
    by_channel_state: Dict[str, Any] = field(default_factory=dict)
    by_quant_mode: Dict[str, Any] = field(default_factory=dict)
    by_fec_type: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """
        Return JSON-friendly dictionary.
        """
        return _json_safe(
            {
                "overall": self.overall,
                "by_frame": self.by_frame,
                "by_channel_state": self.by_channel_state,
                "by_quant_mode": self.by_quant_mode,
                "by_fec_type": self.by_fec_type,
                "extra": self.extra,
            }
        )

    def to_json(self, indent: int = 2, ensure_ascii: bool = False) -> str:
        """
        Return JSON string.
        """
        return json.dumps(
            self.as_dict(),
            indent=indent,
            ensure_ascii=ensure_ascii,
        )


class CommStats:
    """
    Online communication statistics tracker.

    This class stores raw records and flat metric records, and maintains an
    online accumulator.

    Parameters
    ----------
    records : iterable of dict, optional
        Initial ARCE communication records.

    skip_bypassed : bool
        If True, bypassed records are ignored during accumulation.

    keep_records : bool
        If True, store raw records internally.

    keep_flat_metrics : bool
        If True, store flattened metric records internally.
    """

    def __init__(
        self,
        records: Optional[Iterable[Dict[str, Any]]] = None,
        skip_bypassed: bool = False,
        keep_records: bool = True,
        keep_flat_metrics: bool = True,
    ):
        self.skip_bypassed = bool(skip_bypassed)
        self.keep_records = bool(keep_records)
        self.keep_flat_metrics = bool(keep_flat_metrics)

        self.records: List[Dict[str, Any]] = []
        self.flat_metrics: List[Dict[str, Any]] = []

        self.accumulator = init_metric_accumulator()

        self.frame_accumulators: Dict[str, Dict[str, Any]] = {}
        self.channel_accumulators: Dict[str, Dict[str, Any]] = {}
        self.quant_accumulators: Dict[str, Dict[str, Any]] = {}
        self.fec_accumulators: Dict[str, Dict[str, Any]] = {}

        if records is not None:
            self.update(records)

    @classmethod
    def from_arce_comm(
        cls,
        arce_comm: Any,
        skip_bypassed: bool = False,
        keep_records: bool = True,
        keep_flat_metrics: bool = True,
    ) -> "CommStats":
        """
        Build CommStats from an ARCEFixedComm-like object.

        Expected method:
            arce_comm.get_records()
        """
        if not hasattr(arce_comm, "get_records"):
            raise AttributeError("arce_comm should have get_records() method.")

        return cls(
            records=arce_comm.get_records(),
            skip_bypassed=skip_bypassed,
            keep_records=keep_records,
            keep_flat_metrics=keep_flat_metrics,
        )

    def reset(self) -> None:
        """
        Clear all stored records and accumulators.
        """
        self.records.clear()
        self.flat_metrics.clear()

        self.accumulator = init_metric_accumulator()
        self.frame_accumulators.clear()
        self.channel_accumulators.clear()
        self.quant_accumulators.clear()
        self.fec_accumulators.clear()

    def _update_group_accumulator(
        self,
        group_dict: Dict[str, Dict[str, Any]],
        group_key: str,
        metrics: Dict[str, Any],
    ) -> None:
        """
        Update one grouped accumulator dictionary.
        """
        if group_key not in group_dict:
            group_dict[group_key] = init_metric_accumulator()

        update_metric_accumulator(group_dict[group_key], metrics)

    def add_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Add one raw ARCE communication record.

        Parameters
        ----------
        record : dict
            Per-link record generated by ARCEFixedComm.

        Returns
        -------
        dict or None
            Flat metric dictionary, or None if skipped.
        """
        record = copy.deepcopy(record or {})
        metrics = extract_record_metrics(record)

        if self.skip_bypassed and _is_bypassed(metrics):
            return None

        if self.keep_records:
            self.records.append(record)

        if self.keep_flat_metrics:
            self.flat_metrics.append(copy.deepcopy(metrics))

        update_metric_accumulator(self.accumulator, metrics)

        self._update_group_accumulator(
            self.frame_accumulators,
            _record_frame_key(metrics),
            metrics,
        )
        self._update_group_accumulator(
            self.channel_accumulators,
            _record_group_key(metrics, METRIC_CHANNEL_STATE),
            metrics,
        )
        self._update_group_accumulator(
            self.quant_accumulators,
            _record_group_key(metrics, METRIC_QUANT_MODE),
            metrics,
        )
        self._update_group_accumulator(
            self.fec_accumulators,
            _record_group_key(metrics, METRIC_FEC_TYPE),
            metrics,
        )

        return metrics

    def update(self, records: Iterable[Dict[str, Any]]) -> None:
        """
        Add multiple records.
        """
        for record in records:
            self.add_record(record)

    def add_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add one already-flat metric dictionary.

        This is useful when the caller has already used extract_record_metrics().
        """
        metrics = copy.deepcopy(metrics or {})

        if self.skip_bypassed and _is_bypassed(metrics):
            return metrics

        if self.keep_flat_metrics:
            self.flat_metrics.append(copy.deepcopy(metrics))

        update_metric_accumulator(self.accumulator, metrics)

        self._update_group_accumulator(
            self.frame_accumulators,
            _record_frame_key(metrics),
            metrics,
        )
        self._update_group_accumulator(
            self.channel_accumulators,
            _record_group_key(metrics, METRIC_CHANNEL_STATE),
            metrics,
        )
        self._update_group_accumulator(
            self.quant_accumulators,
            _record_group_key(metrics, METRIC_QUANT_MODE),
            metrics,
        )
        self._update_group_accumulator(
            self.fec_accumulators,
            _record_group_key(metrics, METRIC_FEC_TYPE),
            metrics,
        )

        return metrics

    def get_records(self) -> List[Dict[str, Any]]:
        """
        Return stored raw records.
        """
        return copy.deepcopy(self.records)

    def get_flat_metrics(self) -> List[Dict[str, Any]]:
        """
        Return stored flat metric records.
        """
        return copy.deepcopy(self.flat_metrics)

    def get_flat_records(
        self,
        include_raw_flattened_fields: bool = False,
        sep: str = ".",
        max_depth: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return records suitable for CSV logging.

        Parameters
        ----------
        include_raw_flattened_fields : bool
            If True, flatten raw records and merge them with extracted metrics.
            Extracted metrics take priority.

        sep : str
            Separator for flattened raw fields.

        max_depth : int, optional
            Maximum flattening depth.
        """
        if not include_raw_flattened_fields:
            return self.get_flat_metrics()

        output = []

        for idx, record in enumerate(self.records):
            raw_flat = flatten_dict(record, sep=sep, max_depth=max_depth)
            metrics = (
                self.flat_metrics[idx]
                if idx < len(self.flat_metrics)
                else extract_record_metrics(record)
            )

            merged = dict(raw_flat)
            merged.update(copy.deepcopy(metrics))
            output.append(_json_safe(merged))

        return output

    def summarize(self) -> Dict[str, Any]:
        """
        Return overall summary.
        """
        return finalize_metric_accumulator(self.accumulator)

    def summarize_by_frame(self) -> Dict[str, Any]:
        """
        Return per-frame summary.
        """
        return {
            key: finalize_metric_accumulator(acc)
            for key, acc in self.frame_accumulators.items()
        }

    def summarize_by_channel_state(self) -> Dict[str, Any]:
        """
        Return per-channel-state summary.
        """
        return {
            key: finalize_metric_accumulator(acc)
            for key, acc in self.channel_accumulators.items()
        }

    def summarize_by_quant_mode(self) -> Dict[str, Any]:
        """
        Return per-quant-mode summary.
        """
        return {
            key: finalize_metric_accumulator(acc)
            for key, acc in self.quant_accumulators.items()
        }

    def summarize_by_fec_type(self) -> Dict[str, Any]:
        """
        Return per-FEC-type summary.
        """
        return {
            key: finalize_metric_accumulator(acc)
            for key, acc in self.fec_accumulators.items()
        }

    def summarize_by_metric_key(self, key: str) -> Dict[str, Any]:
        """
        Summarize records grouped by a flat metric key.

        If raw records are stored, this uses summarize_by_key().
        Otherwise, it groups stored flat metrics.
        """
        if self.records:
            return summarize_by_key(
                self.records,
                key=key,
                skip_bypassed=self.skip_bypassed,
            )

        grouped: Dict[str, Dict[str, Any]] = {}

        for metrics in self.flat_metrics:
            if self.skip_bypassed and _is_bypassed(metrics):
                continue

            group_key = _record_group_key(metrics, key)

            if group_key not in grouped:
                grouped[group_key] = init_metric_accumulator()

            update_metric_accumulator(grouped[group_key], metrics)

        return {
            group_key: finalize_metric_accumulator(acc)
            for group_key, acc in grouped.items()
        }

    def full_summary(self) -> CommStatsSummary:
        """
        Return overall and grouped summaries.
        """
        return CommStatsSummary(
            overall=self.summarize(),
            by_frame=self.summarize_by_frame(),
            by_channel_state=self.summarize_by_channel_state(),
            by_quant_mode=self.summarize_by_quant_mode(),
            by_fec_type=self.summarize_by_fec_type(),
            extra={
                "skip_bypassed": bool(self.skip_bypassed),
                "keep_records": bool(self.keep_records),
                "keep_flat_metrics": bool(self.keep_flat_metrics),
            },
        )

    def get_compact_summary(self) -> Dict[str, Any]:
        """
        Return compact high-level communication summary.

        This is useful for printing during inference.
        """
        summary = self.summarize()
        sums = summary.get("sum", {}) or {}
        means = summary.get("mean", {}) or {}

        total_tx = to_float(summary.get("total_transmitted_bytes", 0.0))
        total_rx = to_float(summary.get("total_received_bytes", 0.0))
        total_encoded = to_int(summary.get("total_encoded_packets", 0))
        total_lost = to_int(summary.get("total_lost_packets", 0))
        total_source = to_int(summary.get("total_source_packets", 0))

        return {
            "num_records": int(summary.get("num_records", 0)),
            "num_non_bypassed_records": int(
                summary.get("num_non_bypassed_records", 0)
            ),
            "total_transmitted_mb": bytes_to_mb(total_tx),
            "total_received_mb": bytes_to_mb(total_rx),
            "total_source_packets": int(total_source),
            "total_encoded_packets": int(total_encoded),
            "total_lost_packets": int(total_lost),
            "overall_packet_loss_ratio": safe_divide(total_lost, total_encoded),
            "total_fec_recovered_packets": to_int(
                sums.get(METRIC_NUM_FEC_RECOVERED, 0)
            ),
            "total_temporal_filled_packets": to_int(
                sums.get(METRIC_NUM_TEMPORAL_FILLED, 0)
            ),
            "total_spatial_filled_packets": to_int(
                sums.get(METRIC_NUM_SPATIAL_FILLED, 0)
            ),
            "total_zero_filled_packets": to_int(
                sums.get(METRIC_NUM_ZERO_FILLED, 0)
            ),
            "total_still_missing_packets": to_int(
                sums.get(METRIC_NUM_STILL_MISSING, 0)
            ),
            "mean_packet_loss_ratio": to_float(
                means.get(f"mean_{METRIC_PACKET_LOSS_RATIO}", 0.0)
            ),
            "mean_recovery_ratio": to_float(
                means.get(f"mean_{METRIC_RECOVERY_RATIO}", 0.0)
            ),
        }

    def get_last_record(self) -> Optional[Dict[str, Any]]:
        """
        Return last raw record if available.
        """
        if not self.records:
            return None

        return copy.deepcopy(self.records[-1])

    def get_last_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Return last flat metric record if available.
        """
        if not self.flat_metrics:
            return None

        return copy.deepcopy(self.flat_metrics[-1])

    def merge(self, other: "CommStats") -> None:
        """
        Merge another CommStats object into this one.

        Raw records are preferred when available. Otherwise flat metrics are
        merged directly.
        """
        if not isinstance(other, CommStats):
            raise TypeError(f"other should be CommStats, got {type(other)}.")

        if other.records:
            self.update(other.records)
            return

        for metrics in other.flat_metrics:
            self.add_metrics(metrics)

    def to_jsonable(self, full: bool = True) -> Dict[str, Any]:
        """
        Return JSON-friendly dictionary.
        """
        if full:
            return self.full_summary().as_dict()

        return _json_safe(self.get_compact_summary())

    def __len__(self) -> int:
        """
        Number of accumulated records.
        """
        return int(self.accumulator.get("num_records", 0))

    def __repr__(self) -> str:
        compact = self.get_compact_summary()
        return (
            "CommStats("
            f"num_records={compact['num_records']}, "
            f"num_non_bypassed={compact['num_non_bypassed_records']}, "
            f"tx_mb={compact['total_transmitted_mb']:.4f}, "
            f"loss={compact['overall_packet_loss_ratio']:.4f})"
        )


def build_comm_stats(
    records: Iterable[Dict[str, Any]],
    skip_bypassed: bool = False,
) -> CommStats:
    """
    Convenience function to build CommStats from records.
    """
    return CommStats(
        records=records,
        skip_bypassed=skip_bypassed,
        keep_records=True,
        keep_flat_metrics=True,
    )


def summarize_comm_records(
    records: Iterable[Dict[str, Any]],
    skip_bypassed: bool = False,
    full: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to summarize communication records.

    Parameters
    ----------
    records : iterable of dict
        ARCE communication records.

    skip_bypassed : bool
        If True, ignore bypassed records.

    full : bool
        If True, return overall + grouped summary.
        If False, return compact summary.
    """
    stats = build_comm_stats(
        records=records,
        skip_bypassed=skip_bypassed,
    )

    return stats.to_jsonable(full=full)


def extract_flat_comm_records(
    records: Iterable[Dict[str, Any]],
    skip_bypassed: bool = False,
) -> List[Dict[str, Any]]:
    """
    Convert raw ARCE records into flat metric records.

    This is useful before writing CSV.
    """
    flat = []

    for record in records:
        metrics = extract_record_metrics(record)

        if skip_bypassed and _is_bypassed(metrics):
            continue

        flat.append(_json_safe(metrics))

    return flat


def print_compact_summary(
    records_or_stats: Union[Iterable[Dict[str, Any]], CommStats],
    prefix: str = "[ARCE CommStats]",
    skip_bypassed: bool = False,
) -> Dict[str, Any]:
    """
    Print and return compact summary.

    This function is intentionally lightweight and safe to call during
    inference/debugging.
    """
    if isinstance(records_or_stats, CommStats):
        stats = records_or_stats
    else:
        stats = build_comm_stats(
            records=records_or_stats,
            skip_bypassed=skip_bypassed,
        )

    compact = stats.get_compact_summary()

    msg = (
        f"{prefix} "
        f"records={compact['num_records']} "
        f"non_bypassed={compact['num_non_bypassed_records']} "
        f"tx={compact['total_transmitted_mb']:.4f}MB "
        f"rx={compact['total_received_mb']:.4f}MB "
        f"loss={compact['overall_packet_loss_ratio']:.4f} "
        f"fec_rec={compact['total_fec_recovered_packets']} "
        f"tmp={compact['total_temporal_filled_packets']} "
        f"spatial={compact['total_spatial_filled_packets']} "
        f"zero={compact['total_zero_filled_packets']}"
    )

    print(msg)
    return compact


__all__ = [
    "CommStatsSummary",
    "CommStats",
    "build_comm_stats",
    "summarize_comm_records",
    "extract_flat_comm_records",
    "print_compact_summary",
]