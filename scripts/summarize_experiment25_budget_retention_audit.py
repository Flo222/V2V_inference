#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from typing import Any, Dict, Iterable, List, Optional


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _get(row: Dict[str, Any], *keys: str, default=None):
    cur: Any = row
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _finite(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
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


def _ratio_true(values: Iterable[Any]) -> float:
    vals = [bool(v) for v in values]
    return float(sum(vals) / len(vals)) if vals else 0.0


def summarize_mode(mode_dir: str, mode: str) -> Dict[str, Any]:
    path = os.path.join(mode_dir, "audit", "budget_retention_audit.jsonl")
    rows = _read_jsonl(path)
    good = [r for r in rows if "error" not in r]
    retention = [r.get("budget_retention", {}) or {} for r in good]
    supported = [r for r in retention if r.get("available") and r.get("layout_supported")]

    channel_vectors = [r.get("channel_retention_ratio") for r in supported]
    channel_vectors = [v for v in channel_vectors if isinstance(v, list) and v]
    mean_channel: List[float] = []
    if channel_vectors:
        width = len(channel_vectors[0])
        if all(len(v) == width for v in channel_vectors):
            mean_channel = [
                float(statistics.mean(float(v[i]) for v in channel_vectors))
                for i in range(width)
            ]

    row: Dict[str, Any] = {
        "mode": mode,
        "record_count": len(rows),
        "valid_record_count": len(good),
        "error_record_count": len(rows) - len(good),
        "unique_frame_count": len({str(r.get("frame_id")) for r in good}),
        "retention_available_ratio": _ratio_true(r.get("available") for r in retention),
        "layout_supported_ratio": _ratio_true(r.get("layout_supported") for r in retention),
        "prefix_packet_selection_ratio": _ratio_true(
            r.get("packet_selection_is_prefix") for r in supported
        ),
        "mean_source_tx_ratio": _mean(
            _get(r, "packet_outcome", "source_tx_ratio") for r in good
        ),
        "mean_retained_value_ratio": _mean(
            r.get("retained_value_ratio") for r in supported
        ),
        "mean_channel_front_back_bias": _mean(
            r.get("channel_front_back_bias") for r in supported
        ),
        "mean_front_half_channel_retention": _mean(
            r.get("front_half_channel_retention_mean") for r in supported
        ),
        "mean_back_half_channel_retention": _mean(
            r.get("back_half_channel_retention_mean") for r in supported
        ),
        "mean_fully_retained_channels": _mean(
            _get(r, "channel_summary", "num_fully_retained") for r in supported
        ),
        "mean_partially_retained_channels": _mean(
            _get(r, "channel_summary", "num_partially_retained") for r in supported
        ),
        "mean_zero_retained_channels": _mean(
            _get(r, "channel_summary", "num_zero_retained") for r in supported
        ),
        "mean_last_nonzero_channel": _mean(
            _get(r, "channel_summary", "last_nonzero_index") for r in supported
        ),
        "mean_first_zero_channel": _mean(
            _get(r, "channel_summary", "first_zero_index") for r in supported
        ),
        "mean_spatial_retention_min": _mean(
            _get(r, "spatial_summary", "min") for r in supported
        ),
        "mean_spatial_retention_max": _mean(
            _get(r, "spatial_summary", "max") for r in supported
        ),
        "mean_spatial_retention_mean": _mean(
            _get(r, "spatial_summary", "mean") for r in supported
        ),
        "mean_channel_retention_ratio": mean_channel,
    }
    ratio_gap = None
    if row["mean_source_tx_ratio"] is not None and row["mean_retained_value_ratio"] is not None:
        ratio_gap = abs(row["mean_source_tx_ratio"] - row["mean_retained_value_ratio"])
    row["packet_value_ratio_abs_gap"] = ratio_gap
    row["pass"] = bool(
        row["valid_record_count"] > 0
        and row["error_record_count"] == 0
        and row["retention_available_ratio"] == 1.0
        and row["layout_supported_ratio"] == 1.0
        and ratio_gap is not None
        and ratio_gap < 0.01
    )

    csv_path = os.path.join(mode_dir, "mean_channel_retention.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["channel_index", "mean_retention_ratio"])
        for index, value in enumerate(mean_channel):
            writer.writerow([index, value])
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--modes", nargs="+", default=["fp32", "fp16", "int8", "int4"])
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    rows = [summarize_mode(os.path.join(root, mode), mode) for mode in args.modes]
    by_mode = {r["mode"]: r for r in rows}
    ordered = [m for m in ("fp32", "fp16", "int8", "int4") if m in by_mode]
    retained = [by_mode[m]["mean_retained_value_ratio"] for m in ordered]
    cross = {
        "retained_value_ratio_non_decreasing": bool(
            all(v is not None for v in retained)
            and all(float(retained[i]) <= float(retained[i + 1]) + 1e-12 for i in range(len(retained) - 1))
        ),
        "all_modes_prefix_packet_selection": bool(
            all(by_mode[m]["prefix_packet_selection_ratio"] == 1.0 for m in ordered)
        ),
    }

    json_path = os.path.join(root, "experiment25_summary.json")
    csv_path = os.path.join(root, "experiment25_summary.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"modes": rows, "cross_mode_checks": cross}, f, indent=2, ensure_ascii=False)

    scalar_keys: List[str] = []
    for row in rows:
        for key, value in row.items():
            if key != "mean_channel_retention_ratio" and key not in scalar_keys:
                scalar_keys.append(key)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerows([{k: v for k, v in r.items() if k in scalar_keys} for r in rows])

    print("\n===== Experiment 2.5 summary =====")
    for row in rows:
        print(
            "{mode:>5s} pass={passed!s:<5s} retained={ret!s:<10} full_ch={full!s:<8} "
            "partial_ch={partial!s:<8} zero_ch={zero!s:<8} front_back_bias={bias!s}".format(
                mode=row["mode"], passed=row["pass"], ret=row["mean_retained_value_ratio"],
                full=row["mean_fully_retained_channels"], partial=row["mean_partially_retained_channels"],
                zero=row["mean_zero_retained_channels"], bias=row["mean_channel_front_back_bias"],
            )
        )
    print("cross_mode_checks:", json.dumps(cross, ensure_ascii=False))
    print("saved:", json_path)
    print("saved:", csv_path)

    failed = [r["mode"] for r in rows if not r["pass"]]
    return 2 if args.strict and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
