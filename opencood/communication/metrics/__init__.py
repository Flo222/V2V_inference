"""
Communication metrics package for ARCE / OpenCOOD experiments.

This package contains metric utilities for communication-aware cooperative
perception experiments.

Planned modules:

1. comm_logger.py
   Write per-frame / per-link communication records to jsonl / csv.

2. metric_tracker.py
   Online aggregation of communication metrics during inference.

3. summary_writer.py
   Save final communication summaries together with detection metrics.

Design note:
    Keep this __init__.py lightweight.

    Do not eagerly import concrete logger/tracker classes here. Those modules
    may depend on json, pandas, torch, OpenCOOD logging paths, or evaluation
    code. Import concrete modules directly where they are used:

        from opencood.communication.metrics.comm_logger import CommLogger
        from opencood.communication.metrics.metric_tracker import CommMetricTracker
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


METRIC_GROUP_SIZE = "size"
METRIC_GROUP_PACKET = "packet"
METRIC_GROUP_CHANNEL = "channel"
METRIC_GROUP_LATENCY = "latency"
METRIC_GROUP_QUANTIZATION = "quantization"
METRIC_GROUP_FEC = "fec"
METRIC_GROUP_RECOVERY = "recovery"
METRIC_GROUP_SUMMARY = "summary"

VALID_METRIC_GROUPS = (
    METRIC_GROUP_SIZE,
    METRIC_GROUP_PACKET,
    METRIC_GROUP_CHANNEL,
    METRIC_GROUP_LATENCY,
    METRIC_GROUP_QUANTIZATION,
    METRIC_GROUP_FEC,
    METRIC_GROUP_RECOVERY,
    METRIC_GROUP_SUMMARY,
)


# Size / bandwidth metrics.
METRIC_RAW_BYTES = "raw_bytes"
METRIC_COMPRESSED_BYTES = "compressed_bytes"
METRIC_PARITY_BYTES = "parity_bytes"
METRIC_TRANSMITTED_BYTES = "transmitted_bytes"
METRIC_RECEIVED_BYTES = "received_bytes"
METRIC_TRANSMITTED_MB = "transmitted_mb"
METRIC_RECEIVED_MB = "received_mb"
METRIC_COMPRESSION_RATIO = "compression_ratio"
METRIC_REDUNDANCY_RATIO = "redundancy_ratio"


# Packet metrics.
METRIC_NUM_SOURCE_PACKETS = "num_source_packets"
METRIC_NUM_PARITY_PACKETS = "num_parity_packets"
METRIC_NUM_ENCODED_PACKETS = "num_encoded_packets"
METRIC_NUM_RECEIVED_PACKETS = "num_received_packets"
METRIC_NUM_LOST_PACKETS = "num_lost_packets"
METRIC_PACKET_LOSS_RATIO = "packet_loss_ratio"


# Channel / latency metrics.
METRIC_CHANNEL_STATE = "channel_state"
METRIC_BANDWIDTH_MBPS = "bandwidth_mbps"
METRIC_EXPECTED_LOSS = "expected_loss"
METRIC_EMPIRICAL_LOSS = "empirical_loss"
METRIC_TX_DELAY_MS = "tx_delay_ms"
METRIC_JITTER_MS = "jitter_ms"
METRIC_PROC_DELAY_MS = "proc_delay_ms"
METRIC_TOTAL_DELAY_MS = "total_delay_ms"
METRIC_DEADLINE_MS = "deadline_ms"
METRIC_LATE = "late"
METRIC_DELAY_SLOTS = "delay_slots"


# Quantization metrics.
METRIC_QUANT_MODE = "quant_mode"
METRIC_QUANT_BITS = "quant_bits"
METRIC_QUANT_MSE = "quant_mse"
METRIC_QUANT_MAE = "quant_mae"
METRIC_QUANT_RELATIVE_MAE = "quant_relative_mae"


# FEC / recovery metrics.
METRIC_FEC_TYPE = "fec_type"
METRIC_RECOVERY_RATIO = "recovery_ratio"
METRIC_FULL_RECOVERY = "full_recovery"
METRIC_NUM_DIRECT_RECEIVED = "num_direct_received_packets"
METRIC_NUM_FEC_RECOVERED = "num_fec_recovered_packets"
METRIC_NUM_TEMPORAL_FILLED = "num_temporal_filled_packets"
METRIC_NUM_SPATIAL_FILLED = "num_spatial_filled_packets"
METRIC_NUM_ZERO_FILLED = "num_zero_filled_packets"
METRIC_NUM_STILL_MISSING = "num_still_missing_packets"


DEFAULT_SUM_METRICS = (
    METRIC_RAW_BYTES,
    METRIC_COMPRESSED_BYTES,
    METRIC_PARITY_BYTES,
    METRIC_TRANSMITTED_BYTES,
    METRIC_RECEIVED_BYTES,
    METRIC_NUM_SOURCE_PACKETS,
    METRIC_NUM_PARITY_PACKETS,
    METRIC_NUM_ENCODED_PACKETS,
    METRIC_NUM_RECEIVED_PACKETS,
    METRIC_NUM_LOST_PACKETS,
    METRIC_NUM_DIRECT_RECEIVED,
    METRIC_NUM_FEC_RECOVERED,
    METRIC_NUM_TEMPORAL_FILLED,
    METRIC_NUM_SPATIAL_FILLED,
    METRIC_NUM_ZERO_FILLED,
    METRIC_NUM_STILL_MISSING,
)

DEFAULT_MEAN_METRICS = (
    METRIC_COMPRESSION_RATIO,
    METRIC_REDUNDANCY_RATIO,
    METRIC_PACKET_LOSS_RATIO,
    METRIC_EXPECTED_LOSS,
    METRIC_EMPIRICAL_LOSS,
    METRIC_TX_DELAY_MS,
    METRIC_JITTER_MS,
    METRIC_PROC_DELAY_MS,
    METRIC_TOTAL_DELAY_MS,
    METRIC_DEADLINE_MS,
    METRIC_RECOVERY_RATIO,
    METRIC_QUANT_MSE,
    METRIC_QUANT_MAE,
    METRIC_QUANT_RELATIVE_MAE,
)

DEFAULT_COUNT_METRICS = (
    METRIC_LATE,
    METRIC_FULL_RECOVERY,
)


def safe_divide(numerator: Any, denominator: Any, default: float = 0.0) -> float:
    """
    Safely compute numerator / denominator.

    Parameters
    ----------
    numerator : number
        Numerator.

    denominator : number
        Denominator.

    default : float
        Returned when denominator is zero or invalid.

    Returns
    -------
    float
        Division result.
    """
    try:
        numerator = float(numerator)
        denominator = float(denominator)
    except (TypeError, ValueError):
        return float(default)

    if abs(denominator) < 1e-12:
        return float(default)

    return float(numerator / denominator)


def bytes_to_mb(num_bytes: Any, decimal: bool = True) -> float:
    """
    Convert bytes to MB.

    Parameters
    ----------
    num_bytes : number
        Number of bytes.

    decimal : bool
        If True, use 1 MB = 1,000,000 bytes.
        If False, use 1 MiB = 1,048,576 bytes.

    Returns
    -------
    float
        MB / MiB value.
    """
    try:
        num_bytes = float(num_bytes)
    except (TypeError, ValueError):
        return 0.0

    denom = 1_000_000.0 if decimal else 1024.0 * 1024.0
    return float(num_bytes / denom)


def normalize_metric_group(group: Optional[str]) -> str:
    """
    Normalize metric group name.

    Parameters
    ----------
    group : str or None
        Metric group name.

    Returns
    -------
    str
        Normalized group name.
    """
    if group is None:
        return METRIC_GROUP_SUMMARY

    group = str(group).strip().lower()

    if group not in VALID_METRIC_GROUPS:
        raise ValueError(
            f"Unsupported metric group: {group}. "
            f"Expected one of {VALID_METRIC_GROUPS}."
        )

    return group


def normalize_metric_names(
    metrics: Optional[Sequence[str]],
    default: Optional[Sequence[str]] = None,
) -> Tuple[str, ...]:
    """
    Normalize a metric-name sequence.

    Parameters
    ----------
    metrics : sequence / str / None
        Metric names. A comma-separated string is also accepted.

    default : sequence, optional
        Default metric list when metrics is None.

    Returns
    -------
    tuple
        Normalized metric names.
    """
    if metrics is None:
        metrics = default or ()

    if isinstance(metrics, str):
        metrics = [
            item.strip()
            for item in metrics.split(",")
            if item.strip()
        ]

    if not isinstance(metrics, (list, tuple)):
        raise ValueError(
            f"metrics should be list, tuple, str, or None, got {type(metrics)}."
        )

    normalized = []
    for metric in metrics:
        name = str(metric).strip()
        if name and name not in normalized:
            normalized.append(name)

    return tuple(normalized)


def get_nested(record: Dict[str, Any], path: Sequence[str], default: Any = None) -> Any:
    """
    Read a nested value from a dictionary.

    Example
    -------
    get_nested(record, ["size", "actual_transmitted_bytes"])
    """
    cur = record

    for key in path:
        if not isinstance(cur, dict):
            return default

        if key not in cur:
            return default

        cur = cur[key]

    return cur


def set_nested(record: Dict[str, Any], path: Sequence[str], value: Any) -> Dict[str, Any]:
    """
    Set a nested value in a dictionary.

    Returns the modified dictionary.
    """
    if len(path) == 0:
        raise ValueError("path should not be empty.")

    cur = record

    for key in path[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]

    cur[path[-1]] = value
    return record


def flatten_dict(
    data: Dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
    max_depth: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Flatten a nested dictionary.

    Example
    -------
    {"size": {"tx": 1}} -> {"size.tx": 1}
    """
    items = {}

    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)

        if (
            isinstance(value, dict)
            and (max_depth is None or new_key.count(sep) < int(max_depth))
        ):
            items.update(
                flatten_dict(
                    value,
                    parent_key=new_key,
                    sep=sep,
                    max_depth=max_depth,
                )
            )
        else:
            items[new_key] = value

    return items


def is_number(value: Any) -> bool:
    """
    Return whether value is a finite int/float-like number.
    """
    if isinstance(value, bool):
        return False

    try:
        value = float(value)
    except (TypeError, ValueError):
        return False

    return math.isfinite(value)


def to_float(value: Any, default: float = 0.0) -> float:
    """
    Convert value to float safely.
    """
    if not is_number(value):
        return float(default)

    return float(value)


def to_int(value: Any, default: int = 0) -> int:
    """
    Convert value to int safely.
    """
    if not is_number(value):
        return int(default)

    return int(value)


def extract_record_metrics(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract standard communication metrics from one ARCE record.

    This function is aligned with records generated by:
        opencood.methods.arce.arce_fixed_comm.ARCEFixedComm

    Parameters
    ----------
    record : dict
        Per-link communication record.

    Returns
    -------
    dict
        Flat standard metric dictionary.
    """
    record = record or {}

    size = record.get("size", {}) or {}
    channel = record.get("channel", {}) or {}
    channel_profile = channel.get("profile", {}) or {}
    channel_loss = channel.get("loss", {}) or {}
    latency = channel.get("latency", {}) or {}
    action = record.get("action", {}) or {}
    quant = record.get("quantization", {}) or {}
    fec_decode = record.get("fec_decode", {}) or {}
    partial = record.get("partial_reconstruction", {}) or {}

    num_encoded = to_int(
        size.get(
            "actual_num_encoded_packets",
            fec_decode.get("num_encoded_packets", 0),
        )
    )
    num_lost = to_int(
        size.get(
            "actual_num_lost_encoded_packets",
            fec_decode.get("num_lost_encoded_packets", 0),
        )
    )
    num_received = to_int(
        size.get(
            "actual_num_received_encoded_packets",
            fec_decode.get("num_received_encoded_packets", 0),
        )
    )

    transmitted_bytes = to_float(
        size.get(
            "actual_transmitted_bytes",
            size.get("transmitted_bytes", 0.0),
        )
    )
    received_bytes = to_float(
        size.get(
            "actual_received_bytes",
            size.get("received_bytes", 0.0),
        )
    )

    quant_error = quant.get("error", {}) or {}

    metrics = {
        "frame_id": record.get("frame_id", None),
        "link_id": record.get("link_id", None),
        "agent_index": record.get("agent_index", None),
        "ego_index": record.get("ego_index", None),
        "bypassed": bool(record.get("bypassed", False)),

        METRIC_RAW_BYTES: to_float(
            size.get(
                "raw_bytes",
                size.get("raw_bytes_fp32_reference", 0.0),
            )
        ),
        METRIC_COMPRESSED_BYTES: to_float(size.get("compressed_bytes", 0.0)),
        METRIC_PARITY_BYTES: to_float(
            size.get(
                "actual_parity_bytes",
                size.get("parity_bytes", 0.0),
            )
        ),
        METRIC_TRANSMITTED_BYTES: transmitted_bytes,
        METRIC_RECEIVED_BYTES: received_bytes,
        METRIC_TRANSMITTED_MB: bytes_to_mb(transmitted_bytes),
        METRIC_RECEIVED_MB: bytes_to_mb(received_bytes),
        METRIC_COMPRESSION_RATIO: to_float(size.get("compression_ratio", 0.0)),
        METRIC_REDUNDANCY_RATIO: to_float(
            size.get(
                "actual_effective_redundancy_ratio",
                size.get("effective_redundancy_ratio", 0.0),
            )
        ),

        METRIC_NUM_SOURCE_PACKETS: to_int(
            size.get(
                "actual_num_source_packets",
                fec_decode.get("num_source_packets", 0),
            )
        ),
        METRIC_NUM_PARITY_PACKETS: to_int(
            size.get(
                "actual_num_parity_packets",
                size.get("num_parity_packets", 0),
            )
        ),
        METRIC_NUM_ENCODED_PACKETS: num_encoded,
        METRIC_NUM_RECEIVED_PACKETS: num_received,
        METRIC_NUM_LOST_PACKETS: num_lost,
        METRIC_PACKET_LOSS_RATIO: safe_divide(num_lost, num_encoded),

        METRIC_CHANNEL_STATE: channel_profile.get(
            "state_name",
            action.get("channel_state", None),
        ),
        METRIC_BANDWIDTH_MBPS: to_float(channel_profile.get("bandwidth_mbps", 0.0)),
        METRIC_EXPECTED_LOSS: to_float(channel_loss.get("expected_loss", 0.0)),
        METRIC_EMPIRICAL_LOSS: to_float(channel_loss.get("empirical_loss", 0.0)),

        METRIC_TX_DELAY_MS: to_float(latency.get("tx_delay_ms", 0.0)),
        METRIC_JITTER_MS: to_float(latency.get("jitter_ms", 0.0)),
        METRIC_PROC_DELAY_MS: to_float(latency.get("proc_delay_ms", 0.0)),
        METRIC_TOTAL_DELAY_MS: to_float(latency.get("total_delay_ms", 0.0)),
        METRIC_DEADLINE_MS: to_float(latency.get("deadline_ms", 0.0)),
        METRIC_LATE: bool(latency.get("late", False)),
        METRIC_DELAY_SLOTS: to_int(latency.get("delay_slots", 0)),

        METRIC_QUANT_MODE: action.get("quant_mode", quant.get("mode", None)),
        METRIC_QUANT_BITS: to_int(action.get("quant_bits", quant.get("bits", 0))),
        METRIC_QUANT_MSE: to_float(quant_error.get("mse", 0.0)),
        METRIC_QUANT_MAE: to_float(quant_error.get("mae", 0.0)),
        METRIC_QUANT_RELATIVE_MAE: to_float(quant_error.get("relative_mae", 0.0)),

        METRIC_FEC_TYPE: action.get("fec_type", fec_decode.get("fec_type", None)),
        METRIC_RECOVERY_RATIO: to_float(partial.get("recovery_ratio", 0.0)),
        METRIC_FULL_RECOVERY: bool(fec_decode.get("full_recovery", False)),
        METRIC_NUM_DIRECT_RECEIVED: to_int(
            partial.get(
                "num_direct_received_packets",
                fec_decode.get("num_direct_received_source_packets", 0),
            )
        ),
        METRIC_NUM_FEC_RECOVERED: to_int(
            partial.get(
                "num_fec_recovered_packets",
                fec_decode.get("num_fec_recovered_source_packets", 0),
            )
        ),
        METRIC_NUM_TEMPORAL_FILLED: to_int(partial.get("num_temporal_filled_packets", 0)),
        METRIC_NUM_SPATIAL_FILLED: to_int(partial.get("num_spatial_filled_packets", 0)),
        METRIC_NUM_ZERO_FILLED: to_int(partial.get("num_zero_filled_packets", 0)),
        METRIC_NUM_STILL_MISSING: to_int(partial.get("num_still_missing_packets", 0)),
    }

    return metrics


def init_metric_accumulator() -> Dict[str, Any]:
    """
    Create an empty metric accumulator.

    Returns
    -------
    dict
        Accumulator state.
    """
    return {
        "num_records": 0,
        "num_non_bypassed_records": 0,
        "sum": {},
        "count": {},
        "category_count": {},
    }


def update_metric_accumulator(
    accumulator: Dict[str, Any],
    metrics: Dict[str, Any],
    sum_metrics: Optional[Sequence[str]] = None,
    mean_metrics: Optional[Sequence[str]] = None,
    count_metrics: Optional[Sequence[str]] = None,
    category_metrics: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Update accumulator with one flat metric dictionary.

    Parameters
    ----------
    accumulator : dict
        Accumulator from init_metric_accumulator().

    metrics : dict
        Flat metric dictionary.

    sum_metrics : sequence, optional
        Metrics to sum.

    mean_metrics : sequence, optional
        Metrics to average.

    count_metrics : sequence, optional
        Boolean metrics to count True.

    category_metrics : sequence, optional
        Categorical metrics to count by value.

    Returns
    -------
    dict
        Updated accumulator.
    """
    sum_metrics = normalize_metric_names(sum_metrics, DEFAULT_SUM_METRICS)
    mean_metrics = normalize_metric_names(mean_metrics, DEFAULT_MEAN_METRICS)
    count_metrics = normalize_metric_names(count_metrics, DEFAULT_COUNT_METRICS)
    category_metrics = normalize_metric_names(
        category_metrics,
        (
            METRIC_CHANNEL_STATE,
            METRIC_QUANT_MODE,
            METRIC_FEC_TYPE,
        ),
    )

    accumulator["num_records"] = int(accumulator.get("num_records", 0)) + 1

    if not bool(metrics.get("bypassed", False)):
        accumulator["num_non_bypassed_records"] = int(
            accumulator.get("num_non_bypassed_records", 0)
        ) + 1

    sum_dict = accumulator.setdefault("sum", {})
    count_dict = accumulator.setdefault("count", {})
    category_count = accumulator.setdefault("category_count", {})

    for name in sum_metrics:
        value = to_float(metrics.get(name, 0.0))
        sum_dict[name] = to_float(sum_dict.get(name, 0.0)) + value

    for name in mean_metrics:
        value = metrics.get(name, None)
        if is_number(value):
            sum_dict[name] = to_float(sum_dict.get(name, 0.0)) + float(value)
            count_dict[name] = int(count_dict.get(name, 0)) + 1

    for name in count_metrics:
        if bool(metrics.get(name, False)):
            sum_dict[name] = to_int(sum_dict.get(name, 0)) + 1
        else:
            sum_dict.setdefault(name, 0)

    for name in category_metrics:
        value = metrics.get(name, None)
        if value is None:
            continue

        value = str(value)
        category_count.setdefault(name, {})
        category_count[name][value] = int(category_count[name].get(value, 0)) + 1

    return accumulator


def finalize_metric_accumulator(accumulator: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert accumulator into final summary.

    Parameters
    ----------
    accumulator : dict
        Accumulator state.

    Returns
    -------
    dict
        JSON-friendly summary.
    """
    accumulator = accumulator or {}

    num_records = int(accumulator.get("num_records", 0))
    num_non_bypassed = int(accumulator.get("num_non_bypassed_records", 0))

    sum_dict = copy.deepcopy(accumulator.get("sum", {}) or {})
    count_dict = copy.deepcopy(accumulator.get("count", {}) or {})
    category_count = copy.deepcopy(accumulator.get("category_count", {}) or {})

    mean_dict = {}

    for name in DEFAULT_MEAN_METRICS:
        if name in sum_dict:
            mean_dict[f"mean_{name}"] = safe_divide(
                sum_dict.get(name, 0.0),
                count_dict.get(name, 0),
            )

    total_tx = to_float(sum_dict.get(METRIC_TRANSMITTED_BYTES, 0.0))
    total_rx = to_float(sum_dict.get(METRIC_RECEIVED_BYTES, 0.0))
    total_encoded = to_float(sum_dict.get(METRIC_NUM_ENCODED_PACKETS, 0.0))
    total_lost = to_float(sum_dict.get(METRIC_NUM_LOST_PACKETS, 0.0))
    total_source = to_float(sum_dict.get(METRIC_NUM_SOURCE_PACKETS, 0.0))

    summary = {
        "num_records": int(num_records),
        "num_non_bypassed_records": int(num_non_bypassed),
        "sum": sum_dict,
        "mean": mean_dict,
        "category_count": category_count,

        "total_transmitted_bytes": float(total_tx),
        "total_received_bytes": float(total_rx),
        "total_transmitted_mb": bytes_to_mb(total_tx),
        "total_received_mb": bytes_to_mb(total_rx),

        "total_encoded_packets": int(total_encoded),
        "total_lost_packets": int(total_lost),
        "total_source_packets": int(total_source),

        "overall_packet_loss_ratio": safe_divide(total_lost, total_encoded),
        "avg_transmitted_bytes_per_record": safe_divide(total_tx, num_records),
        "avg_received_bytes_per_record": safe_divide(total_rx, num_records),
        "avg_transmitted_bytes_per_non_bypassed_record": safe_divide(
            total_tx,
            num_non_bypassed,
        ),
        "avg_received_bytes_per_non_bypassed_record": safe_divide(
            total_rx,
            num_non_bypassed,
        ),
    }

    return summary


def summarize_records(
    records: Iterable[Dict[str, Any]],
    skip_bypassed: bool = False,
) -> Dict[str, Any]:
    """
    Summarize ARCE communication records.

    Parameters
    ----------
    records : iterable of dict
        Records generated by ARCEFixedComm.

    skip_bypassed : bool
        If True, bypassed records are ignored.

    Returns
    -------
    dict
        Summary dictionary.
    """
    accumulator = init_metric_accumulator()

    for record in records:
        metrics = extract_record_metrics(record)

        if skip_bypassed and bool(metrics.get("bypassed", False)):
            continue

        update_metric_accumulator(accumulator, metrics)

    return finalize_metric_accumulator(accumulator)


def summarize_by_key(
    records: Iterable[Dict[str, Any]],
    key: str,
    skip_bypassed: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Summarize records grouped by a flat extracted metric key.

    Examples
    --------
    summarize_by_key(records, "channel_state")
    summarize_by_key(records, "quant_mode")
    summarize_by_key(records, "fec_type")
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for record in records:
        metrics = extract_record_metrics(record)

        if skip_bypassed and bool(metrics.get("bypassed", False)):
            continue

        value = metrics.get(key, None)
        value = "None" if value is None else str(value)
        grouped.setdefault(value, []).append(record)

    return {
        group_key: summarize_records(group_records, skip_bypassed=skip_bypassed)
        for group_key, group_records in grouped.items()
    }


def get_metric_config_summary() -> Dict[str, Any]:
    """
    Return JSON-friendly metrics package summary.
    """
    return {
        "metric_groups": VALID_METRIC_GROUPS,
        "sum_metrics": DEFAULT_SUM_METRICS,
        "mean_metrics": DEFAULT_MEAN_METRICS,
        "count_metrics": DEFAULT_COUNT_METRICS,
    }


__all__ = [
    "METRIC_GROUP_SIZE",
    "METRIC_GROUP_PACKET",
    "METRIC_GROUP_CHANNEL",
    "METRIC_GROUP_LATENCY",
    "METRIC_GROUP_QUANTIZATION",
    "METRIC_GROUP_FEC",
    "METRIC_GROUP_RECOVERY",
    "METRIC_GROUP_SUMMARY",
    "VALID_METRIC_GROUPS",
    "METRIC_RAW_BYTES",
    "METRIC_COMPRESSED_BYTES",
    "METRIC_PARITY_BYTES",
    "METRIC_TRANSMITTED_BYTES",
    "METRIC_RECEIVED_BYTES",
    "METRIC_TRANSMITTED_MB",
    "METRIC_RECEIVED_MB",
    "METRIC_COMPRESSION_RATIO",
    "METRIC_REDUNDANCY_RATIO",
    "METRIC_NUM_SOURCE_PACKETS",
    "METRIC_NUM_PARITY_PACKETS",
    "METRIC_NUM_ENCODED_PACKETS",
    "METRIC_NUM_RECEIVED_PACKETS",
    "METRIC_NUM_LOST_PACKETS",
    "METRIC_PACKET_LOSS_RATIO",
    "METRIC_CHANNEL_STATE",
    "METRIC_BANDWIDTH_MBPS",
    "METRIC_EXPECTED_LOSS",
    "METRIC_EMPIRICAL_LOSS",
    "METRIC_TX_DELAY_MS",
    "METRIC_JITTER_MS",
    "METRIC_PROC_DELAY_MS",
    "METRIC_TOTAL_DELAY_MS",
    "METRIC_DEADLINE_MS",
    "METRIC_LATE",
    "METRIC_DELAY_SLOTS",
    "METRIC_QUANT_MODE",
    "METRIC_QUANT_BITS",
    "METRIC_QUANT_MSE",
    "METRIC_QUANT_MAE",
    "METRIC_QUANT_RELATIVE_MAE",
    "METRIC_FEC_TYPE",
    "METRIC_RECOVERY_RATIO",
    "METRIC_FULL_RECOVERY",
    "METRIC_NUM_DIRECT_RECEIVED",
    "METRIC_NUM_FEC_RECOVERED",
    "METRIC_NUM_TEMPORAL_FILLED",
    "METRIC_NUM_SPATIAL_FILLED",
    "METRIC_NUM_ZERO_FILLED",
    "METRIC_NUM_STILL_MISSING",
    "DEFAULT_SUM_METRICS",
    "DEFAULT_MEAN_METRICS",
    "DEFAULT_COUNT_METRICS",
    "safe_divide",
    "bytes_to_mb",
    "normalize_metric_group",
    "normalize_metric_names",
    "get_nested",
    "set_nested",
    "flatten_dict",
    "is_number",
    "to_float",
    "to_int",
    "extract_record_metrics",
    "init_metric_accumulator",
    "update_metric_accumulator",
    "finalize_metric_accumulator",
    "summarize_records",
    "summarize_by_key",
    "get_metric_config_summary",
]