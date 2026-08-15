#!/usr/bin/env python3
"""Summarize Experiment 1 compression-audit JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional


AP_RE = re.compile(
    r"Average Precision at IOU 0\.3 is ([0-9.]+), "
    r"The Average Precision at IOU 0\.5 is ([0-9.]+), "
    r"The Average Precision at IOU 0\.7 is ([0-9.]+)"
)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
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


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(statistics.mean(vals)) if vals else None


def _median(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(statistics.median(vals)) if vals else None


def _max(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(max(vals)) if vals else None


def _extract_ap(log_path: str) -> Dict[str, Optional[float]]:
    result = {"ap_03": None, "ap_05": None, "ap_07": None}
    if not os.path.exists(log_path):
        return result
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = AP_RE.search(line)
            if m:
                result = {
                    "ap_03": float(m.group(1)),
                    "ap_05": float(m.group(2)),
                    "ap_07": float(m.group(3)),
                }
    return result


def _safe_get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def summarize_mode(mode_dir: str, mode: str) -> Dict[str, Any]:
    audit_path = os.path.join(mode_dir, "audit", "compression_audit.jsonl")
    records = _read_jsonl(audit_path)
    good = [r for r in records if "error" not in r]
    errors = [r for r in records if "error" in r]

    requested_match = [
        bool(_safe_get(r, "sanity", "requested_matches_actual_quant", default=False))
        for r in good
    ]
    sanity_pass = [bool(_safe_get(r, "sanity", "passed", default=False)) for r in good]
    transport_close = [
        bool(_safe_get(r, "clean_transport_error", "allclose", default=False))
        for r in good
    ]
    int4_packed = [
        bool(_safe_get(r, "sanity", "int4_is_packed", default=False)) for r in good
    ]

    row: Dict[str, Any] = {
        "mode": mode,
        "record_count": len(records),
        "valid_record_count": len(good),
        "error_record_count": len(errors),
        "requested_actual_match_ratio": (
            float(sum(requested_match) / len(requested_match)) if requested_match else 0.0
        ),
        "sanity_pass_ratio": float(sum(sanity_pass) / len(sanity_pass)) if sanity_pass else 0.0,
        "clean_transport_allclose_ratio": (
            float(sum(transport_close) / len(transport_close)) if transport_close else 0.0
        ),
        "int4_packed_ratio": float(sum(int4_packed) / len(int4_packed)) if int4_packed else 0.0,
        "mean_source_fp32_bytes": _mean(
            _safe_get(r, "sizes", "source_payload_fp32_reference_bytes") for r in good
        ),
        "mean_quantized_valid_bytes": _mean(
            _safe_get(r, "sizes", "quantized_valid_stream_bytes") for r in good
        ),
        "mean_source_packet_count": _mean(
            _safe_get(r, "sizes", "source_packet_count") for r in good
        ),
        "mean_padding_bytes": _mean(_safe_get(r, "sizes", "padding_bytes") for r in good),
        "mean_actual_tx_bytes": _mean(
            _safe_get(r, "sizes", "actual_transmitted_bytes") for r in good
        ),
        "mean_compression_ratio_vs_fp32": _mean(
            _safe_get(r, "sizes", "compression_ratio_vs_fp32") for r in good
        ),
        "mean_quant_nmse": _mean(
            _safe_get(r, "quantization_error", "nmse") for r in good
        ),
        "median_quant_nmse": _median(
            _safe_get(r, "quantization_error", "nmse") for r in good
        ),
        "max_quant_nmse": _max(
            _safe_get(r, "quantization_error", "nmse") for r in good
        ),
        "mean_quant_cosine": _mean(
            _safe_get(r, "quantization_error", "cosine_similarity") for r in good
        ),
        "max_clean_transport_nmse": _max(
            _safe_get(r, "clean_transport_error", "nmse") for r in good
        ),
        "total_budget_drop": int(
            sum(int(_safe_get(r, "packet_outcome", "num_missing_by_budget", default=0)) for r in good)
        ),
        "total_bernoulli_loss": int(
            sum(int(_safe_get(r, "packet_outcome", "num_lost_by_bernoulli", default=0)) for r in good)
        ),
        "total_missing_source": int(
            sum(int(_safe_get(r, "packet_outcome", "num_missing_source_packets", default=0)) for r in good)
        ),
        "snapshot_count": int(sum(1 for r in good if r.get("snapshot_path"))),
    }
    row.update(_extract_ap(os.path.join(mode_dir, "inference.log")))

    row["pass"] = bool(
        row["valid_record_count"] > 0
        and row["error_record_count"] == 0
        and row["requested_actual_match_ratio"] == 1.0
        and row["clean_transport_allclose_ratio"] == 1.0
        and row["sanity_pass_ratio"] == 1.0
        and row["total_budget_drop"] == 0
        and row["total_bernoulli_loss"] == 0
        and row["total_missing_source"] == 0
        and (mode != "int4" or row["int4_packed_ratio"] == 1.0)
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--modes", nargs="+", default=["fp32", "fp16", "int8", "int4"])
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    rows = [summarize_mode(os.path.join(root, mode), mode) for mode in args.modes]

    # Cross-mode checks: these are warnings rather than hard assumptions because
    # packet padding may make actual transmitted bytes non-monotonic for tiny payloads.
    by_mode = {r["mode"]: r for r in rows}
    cross_checks: Dict[str, Any] = {}
    if all(m in by_mode for m in ("fp32", "fp16", "int8", "int4")):
        valid = [by_mode[m]["mean_quantized_valid_bytes"] for m in ("fp32", "fp16", "int8", "int4")]
        cross_checks["valid_bytes_strictly_decrease"] = bool(
            all(v is not None for v in valid)
            and valid[0] > valid[1] > valid[2] > valid[3]
        )
        errors = [by_mode[m]["mean_quant_nmse"] for m in ("fp32", "fp16", "int8", "int4")]
        cross_checks["quant_nmse_non_decreasing"] = bool(
            all(v is not None for v in errors)
            and errors[0] <= errors[1] <= errors[2] <= errors[3]
        )

    os.makedirs(root, exist_ok=True)
    json_path = os.path.join(root, "experiment1_summary.json")
    csv_path = os.path.join(root, "experiment1_summary.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"modes": rows, "cross_mode_checks": cross_checks}, f, indent=2, ensure_ascii=False)

    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== Experiment 1 summary =====")
    for row in rows:
        print(
            "{mode:>5s} pass={pass_!s:<5s} records={records:<5d} "
            "valid_bytes={bytes_!s:<12} packets={packets!s:<8} "
            "quant_nmse={nmse!s:<12} cos={cos!s:<12} AP@0.7={ap!s}".format(
                mode=row["mode"],
                pass_=row["pass"],
                records=row["valid_record_count"],
                bytes_=row["mean_quantized_valid_bytes"],
                packets=row["mean_source_packet_count"],
                nmse=row["mean_quant_nmse"],
                cos=row["mean_quant_cosine"],
                ap=row["ap_07"],
            )
        )
    print("cross_mode_checks:", json.dumps(cross_checks, ensure_ascii=False))
    print("saved:", json_path)
    print("saved:", csv_path)

    failed = [r["mode"] for r in rows if not r["pass"]]
    if failed:
        print("FAILED modes:", ", ".join(failed))
        return 2 if args.strict else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
