#!/usr/bin/env python3
"""Summarize Experiment 2: limited budget, no loss, no FEC."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from typing import Any, Dict, Iterable, List, Optional


AP_RE = re.compile(
    r"Average Precision at IOU 0\.3 is ([0-9.]+), "
    r"The Average Precision at IOU 0\.5 is ([0-9.]+), "
    r"The Average Precision at IOU 0\.7 is ([0-9.]+)"
)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError("Invalid JSONL %s:%d: %s" % (path, line_no, exc))
    return records


def _safe_get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _finite(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        if value is None:
            continue
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def _mean(values: Iterable[Any]) -> Optional[float]:
    vals = _finite(values)
    return float(statistics.mean(vals)) if vals else None


def _median(values: Iterable[Any]) -> Optional[float]:
    vals = _finite(values)
    return float(statistics.median(vals)) if vals else None


def _max(values: Iterable[Any]) -> Optional[float]:
    vals = _finite(values)
    return float(max(vals)) if vals else None


def _sum_int(values: Iterable[Any]) -> int:
    total = 0
    for value in values:
        try:
            total += int(value or 0)
        except (TypeError, ValueError):
            pass
    return int(total)


def _ratio_true(values: Iterable[Any]) -> float:
    vals = [bool(v) for v in values]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _extract_ap(log_path: str) -> Dict[str, Optional[float]]:
    result = {"ap_03": None, "ap_05": None, "ap_07": None}
    if not os.path.exists(log_path):
        return result
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = AP_RE.search(line)
            if match:
                result = {
                    "ap_03": float(match.group(1)),
                    "ap_05": float(match.group(2)),
                    "ap_07": float(match.group(3)),
                }
    return result


def summarize_mode(mode_dir: str, mode: str) -> Dict[str, Any]:
    audit_path = os.path.join(mode_dir, "audit", "compression_budget_audit.jsonl")
    records = _read_jsonl(audit_path)
    good = [r for r in records if "error" not in r]
    errors = [r for r in records if "error" in r]

    source_packets = [
        _safe_get(r, "packet_outcome", "num_source_packets", default=0) for r in good
    ]
    tx_source = [
        _safe_get(r, "packet_outcome", "num_transmitted_source_packets", default=0)
        for r in good
    ]
    dropped_source = [
        _safe_get(r, "packet_outcome", "num_source_dropped_by_budget", default=0)
        for r in good
    ]

    row: Dict[str, Any] = {
        "mode": mode,
        "record_count": len(records),
        "valid_record_count": len(good),
        "error_record_count": len(errors),
        "unique_frame_count": len(
            {str(r.get("frame_id")) for r in good if r.get("frame_id") is not None}
        ),
        "frame_id_present_ratio": _ratio_true(
            r.get("frame_id") is not None for r in good
        ),
        "requested_actual_match_ratio": _ratio_true(
            _safe_get(r, "sanity", "requested_matches_actual_quant", default=False)
            for r in good
        ),
        "budget_accounting_valid_ratio": _ratio_true(
            _safe_get(r, "sanity", "budget_packet_accounting_valid", default=False)
            for r in good
        ),
        "sanity_pass_ratio": _ratio_true(
            _safe_get(r, "sanity", "passed", default=False) for r in good
        ),
        "mean_source_fp32_bytes": _mean(
            _safe_get(r, "sizes", "source_payload_fp32_reference_bytes") for r in good
        ),
        "mean_quantized_valid_bytes": _mean(
            _safe_get(r, "sizes", "quantized_valid_stream_bytes") for r in good
        ),
        "mean_bandwidth_budget_bytes": _mean(
            _safe_get(r, "sizes", "bandwidth_budget_bytes") for r in good
        ),
        "mean_actual_tx_bytes": _mean(
            _safe_get(r, "sizes", "actual_transmitted_bytes") for r in good
        ),
        "mean_budget_utilization": _mean(
            _safe_get(r, "sizes", "budget_utilization") for r in good
        ),
        "mean_source_packet_count": _mean(source_packets),
        "mean_tx_source_packets": _mean(tx_source),
        "mean_source_dropped_by_budget": _mean(dropped_source),
        "median_source_dropped_by_budget": _median(dropped_source),
        "max_source_dropped_by_budget": _max(dropped_source),
        "total_source_dropped_by_budget": _sum_int(dropped_source),
        "total_parity_dropped_by_budget": _sum_int(
            _safe_get(r, "packet_outcome", "num_parity_dropped_by_budget", default=0)
            for r in good
        ),
        "total_parity_packets": _sum_int(
            _safe_get(r, "packet_outcome", "num_parity_packets", default=0)
            for r in good
        ),
        "total_bernoulli_loss": _sum_int(
            _safe_get(r, "packet_outcome", "num_lost_by_bernoulli", default=0)
            for r in good
        ),
        "mean_source_tx_ratio": _mean(
            _safe_get(r, "packet_outcome", "source_tx_ratio") for r in good
        ),
        "mean_source_recovery_ratio": _mean(
            _safe_get(r, "packet_outcome", "source_recovery_ratio") for r in good
        ),
        "mean_quant_nmse": _mean(
            _safe_get(r, "quantization_error", "nmse") for r in good
        ),
        "mean_budget_transport_nmse": _mean(
            _safe_get(r, "budget_transport_error", "nmse") for r in good
        ),
        "median_budget_transport_nmse": _median(
            _safe_get(r, "budget_transport_error", "nmse") for r in good
        ),
        "max_budget_transport_nmse": _max(
            _safe_get(r, "budget_transport_error", "nmse") for r in good
        ),
        "mean_budget_transport_cosine": _mean(
            _safe_get(r, "budget_transport_error", "cosine_similarity") for r in good
        ),
        "mean_end_to_end_nmse": _mean(
            _safe_get(r, "end_to_end_error", "nmse") for r in good
        ),
        "snapshot_count": int(sum(1 for r in good if r.get("snapshot_path"))),
    }
    row.update(_extract_ap(os.path.join(mode_dir, "inference.log")))

    row["pass"] = bool(
        row["valid_record_count"] > 0
        and row["error_record_count"] == 0
        and row["frame_id_present_ratio"] == 1.0
        and row["requested_actual_match_ratio"] == 1.0
        and row["budget_accounting_valid_ratio"] == 1.0
        and row["sanity_pass_ratio"] == 1.0
        and row["total_parity_packets"] == 0
        and row["total_parity_dropped_by_budget"] == 0
        and row["total_bernoulli_loss"] == 0
    )
    return row


def _nondecreasing(values: List[Optional[float]]) -> bool:
    return bool(
        all(v is not None for v in values)
        and all(float(values[i]) <= float(values[i + 1]) + 1e-12 for i in range(len(values) - 1))
    )


def _nonincreasing(values: List[Optional[float]]) -> bool:
    return bool(
        all(v is not None for v in values)
        and all(float(values[i]) + 1e-12 >= float(values[i + 1]) for i in range(len(values) - 1))
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--modes", nargs="+", default=["fp32", "fp16", "int8", "int4"])
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    rows = [summarize_mode(os.path.join(root, mode), mode) for mode in args.modes]
    by_mode = {row["mode"]: row for row in rows}
    cross: Dict[str, Any] = {}
    ordered = ["fp32", "fp16", "int8", "int4"]
    if all(mode in by_mode for mode in ordered):
        valid_bytes = [by_mode[m]["mean_quantized_valid_bytes"] for m in ordered]
        cross["valid_bytes_strictly_decrease"] = bool(
            all(v is not None for v in valid_bytes)
            and valid_bytes[0] > valid_bytes[1] > valid_bytes[2] > valid_bytes[3]
        )
        cross["source_tx_ratio_non_decreasing"] = _nondecreasing(
            [by_mode[m]["mean_source_tx_ratio"] for m in ordered]
        )
        cross["source_budget_drop_non_increasing"] = _nonincreasing(
            [by_mode[m]["mean_source_dropped_by_budget"] for m in ordered]
        )
        cross["source_recovery_ratio_non_decreasing"] = _nondecreasing(
            [by_mode[m]["mean_source_recovery_ratio"] for m in ordered]
        )

    os.makedirs(root, exist_ok=True)
    json_path = os.path.join(root, "experiment2_summary.json")
    csv_path = os.path.join(root, "experiment2_summary.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"modes": rows, "cross_mode_checks": cross}, f, indent=2, ensure_ascii=False)

    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== Experiment 2 summary =====")
    for row in rows:
        print(
            "{mode:>5s} pass={passed!s:<5s} frames={frames:<4d} records={records:<5d} "
            "src_pkt={src!s:<9} tx_src={tx!s:<9} drop_src={drop!s:<9} "
            "recovery={rec!s:<10} budget_nmse={nmse!s:<12} AP@0.7={ap!s}".format(
                mode=row["mode"],
                passed=row["pass"],
                frames=row["unique_frame_count"],
                records=row["valid_record_count"],
                src=row["mean_source_packet_count"],
                tx=row["mean_tx_source_packets"],
                drop=row["mean_source_dropped_by_budget"],
                rec=row["mean_source_recovery_ratio"],
                nmse=row["mean_budget_transport_nmse"],
                ap=row["ap_07"],
            )
        )
    print("cross_mode_checks:", json.dumps(cross, ensure_ascii=False))
    print("saved:", json_path)
    print("saved:", csv_path)

    failed = [row["mode"] for row in rows if not row["pass"]]
    if failed:
        print("FAILED modes:", ", ".join(failed))
        return 2 if args.strict else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
