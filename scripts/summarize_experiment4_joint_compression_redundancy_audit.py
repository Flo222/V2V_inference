#!/usr/bin/env python3
"""Summarize Experiment 4: finite-budget compression + redundancy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from typing import Any, Dict, Iterable, List, Optional, Tuple

AP_RE = re.compile(
    r"Average Precision at IOU 0\.3 is ([0-9.]+), "
    r"The Average Precision at IOU 0\.5 is ([0-9.]+), "
    r"The Average Precision at IOU 0\.7 is ([0-9.]+)"
)


def _tag(x: float) -> str:
    return ("%.2f" % float(x)).replace(".", "p")


def _condition(plr: float, quant: str, rho: float) -> str:
    return "plr_%s_quant_%s_rho_%s" % (_tag(plr), quant, _tag(rho))


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError("Invalid JSONL %s:%d: %s" % (path, line_no, exc))
    return out


def _get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
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
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            out.append(value)
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


def _sum(values: Iterable[Any]) -> int:
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


def _extract_ap(path: str) -> Dict[str, Optional[float]]:
    result = {"ap_03": None, "ap_05": None, "ap_07": None}
    if not os.path.exists(path):
        return result
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = AP_RE.search(line)
            if m:
                result = {
                    "ap_03": float(m.group(1)),
                    "ap_05": float(m.group(2)),
                    "ap_07": float(m.group(3)),
                }
    return result


def _record_key(record: Dict[str, Any]) -> Tuple[str, int, int]:
    return (
        str(record.get("frame_id")),
        int(record.get("ego_index", -1)),
        int(record.get("agent_index", -1)),
    )


def _nondecreasing(values: List[Optional[float]], tol: float = 1e-12) -> bool:
    vals = [v for v in values if v is not None]
    return all(vals[i] <= vals[i + 1] + tol for i in range(len(vals) - 1))


def _nonincreasing(values: List[Optional[float]], tol: float = 1e-12) -> bool:
    vals = [v for v in values if v is not None]
    return all(vals[i] + tol >= vals[i + 1] for i in range(len(vals) - 1))


def summarize_condition(root: str, plr: float, quant: str, rho: float) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    name = _condition(plr, quant, rho)
    condition_dir = os.path.join(root, name)
    path = os.path.join(condition_dir, "audit", "joint_audit.jsonl")
    records = _read_jsonl(path)
    good = [r for r in records if "error" not in r]
    errors = [r for r in records if "error" in r]
    packet = lambda r, k, default=None: _get(r, "packet", k, default=default)
    budget = lambda r, k, default=None: _get(r, "budget", k, default=default)

    summary: Dict[str, Any] = {
        "condition": name,
        "plr": float(plr),
        "quant_mode": str(quant),
        "rho": float(rho),
        "fec_type": "none" if rho <= 0 else "raptor_sim",
        "record_count": len(records),
        "valid_record_count": len(good),
        "error_record_count": len(errors),
        "unique_frame_count": len({str(r.get("frame_id")) for r in good}),
        "frame_id_present_ratio": _ratio_true(r.get("frame_id") is not None for r in good),
        "sanity_pass_ratio": _ratio_true(_get(r, "sanity", "passed", default=False) for r in good),
        "budget_not_exceeded_ratio": _ratio_true(_get(r, "sanity", "budget_not_exceeded", default=False) for r in good),
        "mean_bandwidth_budget_bytes": _mean(budget(r, "bandwidth_budget_bytes") for r in good),
        "mean_actual_tx_bytes": _mean(budget(r, "actual_transmitted_bytes") for r in good),
        "mean_actual_rx_bytes": _mean(budget(r, "actual_received_bytes") for r in good),
        "mean_budget_utilization": _mean(budget(r, "budget_utilization") for r in good),
        "mean_source_packets": _mean(packet(r, "num_source_packets") for r in good),
        "mean_parity_packets": _mean(packet(r, "num_parity_packets") for r in good),
        "mean_encoded_packets": _mean(packet(r, "num_encoded_packets") for r in good),
        "mean_tx_source_packets": _mean(packet(r, "num_transmitted_source_packets") for r in good),
        "mean_tx_parity_packets": _mean(packet(r, "num_transmitted_parity_packets") for r in good),
        "mean_source_dropped_by_budget": _mean(packet(r, "num_source_dropped_by_budget") for r in good),
        "mean_parity_dropped_by_budget": _mean(packet(r, "num_parity_dropped_by_budget") for r in good),
        "total_source_dropped_by_budget": _sum(packet(r, "num_source_dropped_by_budget") for r in good),
        "total_parity_dropped_by_budget": _sum(packet(r, "num_parity_dropped_by_budget") for r in good),
        "mean_source_tx_ratio": _mean(packet(r, "source_tx_ratio") for r in good),
        "mean_parity_tx_ratio": _mean(packet(r, "parity_tx_ratio") for r in good),
        "mean_encoded_tx_ratio": _mean(packet(r, "encoded_tx_ratio") for r in good),
        "all_source_transmitted_ratio": _ratio_true(int(packet(r, "num_source_dropped_by_budget", 0)) == 0 for r in good),
        "any_parity_transmitted_ratio": _ratio_true(int(packet(r, "num_transmitted_parity_packets", 0)) > 0 for r in good if int(packet(r, "num_parity_packets", 0)) > 0),
        "all_parity_transmitted_ratio": _ratio_true(int(packet(r, "num_parity_dropped_by_budget", 0)) == 0 for r in good if int(packet(r, "num_parity_packets", 0)) > 0),
        "mean_source_channel_loss_ratio": _mean(packet(r, "source_channel_loss_ratio_of_transmitted") for r in good),
        "mean_parity_channel_loss_ratio": _mean(packet(r, "parity_channel_loss_ratio_of_transmitted") for r in good if int(packet(r, "num_transmitted_parity_packets", 0)) > 0),
        "mean_direct_received_source_packets": _mean(packet(r, "num_direct_received_source_packets") for r in good),
        "mean_fec_recovered_source_packets": _mean(packet(r, "num_fec_recovered_source_packets") for r in good),
        "mean_missing_source_packets": _mean(packet(r, "num_missing_source_packets") for r in good),
        "total_fec_recovered_source_packets": _sum(packet(r, "num_fec_recovered_source_packets") for r in good),
        "total_missing_source_packets": _sum(packet(r, "num_missing_source_packets") for r in good),
        "mean_direct_recovery_ratio": _mean(packet(r, "source_direct_recovery_ratio") for r in good),
        "mean_final_recovery_ratio": _mean(packet(r, "source_final_recovery_ratio") for r in good),
        "mean_fec_recovery_fraction_of_direct_missing": _mean(packet(r, "fec_recovery_fraction_of_direct_missing") for r in good),
        "mean_direct_nmse": _mean(_get(r, "direct_feature_error", "nmse") for r in good),
        "median_direct_nmse": _median(_get(r, "direct_feature_error", "nmse") for r in good),
        "mean_fec_nmse": _mean(_get(r, "fec_feature_error", "nmse") for r in good),
        "median_fec_nmse": _median(_get(r, "fec_feature_error", "nmse") for r in good),
        "max_fec_nmse": _max(_get(r, "fec_feature_error", "nmse") for r in good),
        "mean_nmse_reduction": _mean(_get(r, "fec_gain", "nmse_reduction") for r in good),
        "mean_relative_nmse_reduction": _mean(_get(r, "fec_gain", "relative_nmse_reduction") for r in good),
        "mean_direct_cosine": _mean(_get(r, "direct_feature_error", "cosine_similarity") for r in good),
        "mean_fec_cosine": _mean(_get(r, "fec_feature_error", "cosine_similarity") for r in good),
        "mean_cosine_gain": _mean(_get(r, "fec_gain", "cosine_gain") for r in good),
        "snapshot_count": sum(1 for r in good if r.get("snapshot_path")),
    }
    summary.update(_extract_ap(os.path.join(condition_dir, "inference.log")))
    summary["pass"] = bool(
        len(good) > 0
        and len(errors) == 0
        and summary["frame_id_present_ratio"] == 1.0
        and summary["sanity_pass_ratio"] == 1.0
        and summary["budget_not_exceeded_ratio"] == 1.0
    )
    return summary, good


def build_checks(
    summaries: Dict[Tuple[float, str, float], Dict[str, Any]],
    records: Dict[Tuple[float, str, float], List[Dict[str, Any]]],
    plrs: List[float],
    quants: List[str],
    rhos: List[float],
) -> Dict[str, Any]:
    checks: Dict[str, Any] = {"by_plr_quant": {}, "by_plr_rho": {}}

    # Across rho: source payload, source budget selection and source channel
    # loss must be paired. Only parity availability should change.
    for plr in plrs:
        for quant in quants:
            key_name = "plr_%s_quant_%s" % (_tag(plr), quant)
            base_rho = rhos[0]
            base = {_record_key(r): r for r in records[(plr, quant, base_rho)]}
            match: Dict[str, Any] = {}
            for rho in rhos:
                cur = {_record_key(r): r for r in records[(plr, quant, rho)]}
                common = sorted(set(base) & set(cur))
                def ratio(field: str) -> float:
                    if not common:
                        return 0.0
                    ok = 0
                    for k in common:
                        if _get(base[k], "packet", field) == _get(cur[k], "packet", field):
                            ok += 1
                    return float(ok / len(common))
                match[str(rho)] = {
                    "matched_links": len(common),
                    "source_packet_count_match_ratio": ratio("num_source_packets"),
                    "source_tx_fingerprint_match_ratio": ratio("source_tx_fingerprint"),
                    "source_loss_fingerprint_match_ratio": ratio("source_loss_fingerprint"),
                }

            ordered = [summaries[(plr, quant, rho)] for rho in rhos]
            checks["by_plr_quant"][key_name] = {
                "condition_match": match,
                "all_source_packet_counts_match": all(v["source_packet_count_match_ratio"] == 1.0 for v in match.values()),
                "all_source_tx_fingerprints_match": all(v["source_tx_fingerprint_match_ratio"] == 1.0 for v in match.values()),
                "all_source_loss_fingerprints_match": all(v["source_loss_fingerprint_match_ratio"] == 1.0 for v in match.values()),
                "mean_source_tx_ratio_constant": max(float(s["mean_source_tx_ratio"] or 0.0) for s in ordered) - min(float(s["mean_source_tx_ratio"] or 0.0) for s in ordered) <= 1e-12,
                "mean_parity_packets_non_decreasing": _nondecreasing([s["mean_parity_packets"] for s in ordered]),
                "mean_tx_parity_packets_non_decreasing": _nondecreasing([s["mean_tx_parity_packets"] for s in ordered]),
                "mean_final_recovery_ratio_non_decreasing": _nondecreasing([s["mean_final_recovery_ratio"] for s in ordered]),
                "mean_fec_nmse_non_increasing": _nonincreasing([s["mean_fec_nmse"] for s in ordered]),
            }

    # Across quantization at fixed PLR/rho: lower bit width should reduce source
    # packets and increase the source fraction that fits in the same budget.
    for plr in plrs:
        for rho in rhos:
            key_name = "plr_%s_rho_%s" % (_tag(plr), _tag(rho))
            ordered = [summaries[(plr, q, rho)] for q in quants]
            source_packets = [s["mean_source_packets"] for s in ordered]
            checks["by_plr_rho"][key_name] = {
                "quant_order": list(quants),
                "mean_source_packets_strictly_decrease": all(
                    source_packets[i] is not None and source_packets[i + 1] is not None
                    and float(source_packets[i]) > float(source_packets[i + 1])
                    for i in range(len(source_packets) - 1)
                ),
                "mean_source_tx_ratio_non_decreasing": _nondecreasing([s["mean_source_tx_ratio"] for s in ordered]),
                "mean_parity_tx_ratio_non_decreasing": _nondecreasing([s["mean_parity_tx_ratio"] for s in ordered]),
            }
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--plrs", nargs="+", type=float, required=True)
    parser.add_argument("--quant-modes", nargs="+", required=True)
    parser.add_argument("--rhos", nargs="+", type=float, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    plrs = [float(v) for v in args.plrs]
    quants = [str(v) for v in args.quant_modes]
    rhos = [float(v) for v in args.rhos]

    summary_map: Dict[Tuple[float, str, float], Dict[str, Any]] = {}
    record_map: Dict[Tuple[float, str, float], List[Dict[str, Any]]] = {}
    rows: List[Dict[str, Any]] = []
    for plr in plrs:
        for quant in quants:
            for rho in rhos:
                summary, records = summarize_condition(root, plr, quant, rho)
                summary_map[(plr, quant, rho)] = summary
                record_map[(plr, quant, rho)] = records
                rows.append(summary)

    checks = build_checks(summary_map, record_map, plrs, quants, rhos)
    output = {"conditions": rows, "cross_condition_checks": checks}

    json_path = os.path.join(root, "experiment4_summary.json")
    csv_path = os.path.join(root, "experiment4_summary.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(output, ensure_ascii=False, indent=2))
    print("Wrote:", json_path)
    print("Wrote:", csv_path)

    failures: List[str] = []
    for row in rows:
        if not row["pass"]:
            failures.append("condition failed: %s" % row["condition"])
    for key, item in checks["by_plr_quant"].items():
        for field in (
            "all_source_packet_counts_match",
            "all_source_tx_fingerprints_match",
            "all_source_loss_fingerprints_match",
            "mean_source_tx_ratio_constant",
            "mean_parity_packets_non_decreasing",
            "mean_tx_parity_packets_non_decreasing",
            "mean_final_recovery_ratio_non_decreasing",
            "mean_fec_nmse_non_increasing",
        ):
            if not item[field]:
                failures.append("%s: %s=false" % (key, field))
    for key, item in checks["by_plr_rho"].items():
        for field in (
            "mean_source_packets_strictly_decrease",
            "mean_source_tx_ratio_non_decreasing",
            "mean_parity_tx_ratio_non_decreasing",
        ):
            if not item[field]:
                failures.append("%s: %s=false" % (key, field))

    if failures:
        print("Experiment-4 checks with warnings/failures:")
        for failure in failures:
            print(" -", failure)
        if args.strict:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
