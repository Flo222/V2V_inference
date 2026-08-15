#!/usr/bin/env python3
"""Summarize Experiment 3: sufficient budget, fixed PLR, pure FEC."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

AP_RE = re.compile(
    r"Average Precision at IOU 0\.3 is ([0-9.]+), "
    r"The Average Precision at IOU 0\.5 is ([0-9.]+), "
    r"The Average Precision at IOU 0\.7 is ([0-9.]+)"
)


def _condition(plr: float, rho: float) -> str:
    def t(x: float) -> str:
        return ("%.2f" % x).replace(".", "p")
    return "plr_%s_rho_%s" % (t(plr), t(rho))


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
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
    out = []
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
                result = {"ap_03": float(m.group(1)), "ap_05": float(m.group(2)), "ap_07": float(m.group(3))}
    return result


def _record_key(record: Dict[str, Any]) -> Tuple[str, int, int]:
    return (
        str(record.get("frame_id")),
        int(record.get("ego_index", -1)),
        int(record.get("agent_index", -1)),
    )


def summarize_condition(root: str, plr: float, rho: float) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    name = _condition(plr, rho)
    condition_dir = os.path.join(root, name)
    path = os.path.join(condition_dir, "audit", "fec_recovery_audit.jsonl")
    records = _read_jsonl(path)
    good = [r for r in records if "error" not in r]
    errors = [r for r in records if "error" in r]

    row: Dict[str, Any] = {
        "condition": name,
        "plr": float(plr),
        "rho": float(rho),
        "fec_type": "none" if rho <= 0 else "raptor_sim",
        "record_count": len(records),
        "valid_record_count": len(good),
        "error_record_count": len(errors),
        "unique_frame_count": len({str(r.get("frame_id")) for r in good}),
        "sanity_pass_ratio": _ratio_true(_get(r, "sanity", "passed", default=False) for r in good),
        "no_budget_drop_ratio": _ratio_true(_get(r, "sanity", "no_budget_drop", default=False) for r in good),
        "all_encoded_transmitted_ratio": _ratio_true(_get(r, "sanity", "all_encoded_transmitted", default=False) for r in good),
        "mean_source_packets": _mean(_get(r, "packet", "num_source_packets") for r in good),
        "mean_parity_packets": _mean(_get(r, "packet", "num_parity_packets") for r in good),
        "mean_encoded_packets": _mean(_get(r, "packet", "num_encoded_packets") for r in good),
        "mean_direct_received_source_packets": _mean(_get(r, "packet", "num_direct_received_source_packets") for r in good),
        "mean_fec_recovered_source_packets": _mean(_get(r, "packet", "num_fec_recovered_source_packets") for r in good),
        "mean_missing_source_packets": _mean(_get(r, "packet", "num_missing_source_packets") for r in good),
        "total_fec_recovered_source_packets": _sum(_get(r, "packet", "num_fec_recovered_source_packets") for r in good),
        "total_missing_source_packets": _sum(_get(r, "packet", "num_missing_source_packets") for r in good),
        "mean_source_empirical_loss_ratio": _mean(_get(r, "packet", "source_empirical_loss_ratio") for r in good),
        "mean_parity_empirical_loss_ratio": _mean(_get(r, "packet", "parity_empirical_loss_ratio") for r in good),
        "mean_direct_recovery_ratio": _mean(_get(r, "packet", "source_direct_recovery_ratio") for r in good),
        "mean_final_recovery_ratio": _mean(_get(r, "packet", "source_final_recovery_ratio") for r in good),
        "mean_fec_recovery_fraction_of_direct_missing": _mean(_get(r, "packet", "fec_recovery_fraction_of_direct_missing") for r in good),
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
        "mean_actual_tx_bytes": _mean(_get(r, "bytes", "actual_transmitted_bytes") for r in good),
        "mean_actual_rx_bytes": _mean(_get(r, "bytes", "actual_received_bytes") for r in good),
        "snapshot_count": sum(1 for r in good if r.get("snapshot_path")),
    }
    row.update(_extract_ap(os.path.join(condition_dir, "inference.log")))
    row["pass"] = bool(
        row["valid_record_count"] > 0
        and row["error_record_count"] == 0
        and row["sanity_pass_ratio"] == 1.0
        and row["no_budget_drop_ratio"] == 1.0
        and row["all_encoded_transmitted_ratio"] == 1.0
    )
    return row, good


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--plrs", nargs="+", type=float, required=True)
    parser.add_argument("--rhos", nargs="+", type=float, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    rows: List[Dict[str, Any]] = []
    records_by_condition: Dict[Tuple[float, float], List[Dict[str, Any]]] = {}
    for plr in args.plrs:
        for rho in args.rhos:
            row, records = summarize_condition(root, plr, rho)
            rows.append(row)
            records_by_condition[(plr, rho)] = records

    cross: Dict[str, Any] = {"by_plr": {}}
    for plr in args.plrs:
        group = [r for r in rows if abs(float(r["plr"]) - plr) < 1e-12]
        group.sort(key=lambda r: float(r["rho"]))
        baseline_records = records_by_condition.get((plr, min(args.rhos)), [])
        baseline_map = {_record_key(r): r for r in baseline_records}
        matched = 0
        fingerprint_matches = 0
        source_count_matches = 0
        condition_match_ratios = {}
        for rho in sorted(args.rhos):
            current = records_by_condition.get((plr, rho), [])
            cur_map = {_record_key(r): r for r in current}
            common = sorted(set(baseline_map) & set(cur_map))
            local_f = 0
            local_k = 0
            for key in common:
                b = baseline_map[key]; c = cur_map[key]
                if _get(b, "packet", "source_loss_fingerprint") == _get(c, "packet", "source_loss_fingerprint"):
                    local_f += 1
                if _get(b, "packet", "num_source_packets") == _get(c, "packet", "num_source_packets"):
                    local_k += 1
            matched += len(common)
            fingerprint_matches += local_f
            source_count_matches += local_k
            condition_match_ratios[str(rho)] = {
                "matched_links": len(common),
                "source_loss_fingerprint_match_ratio": float(local_f / len(common)) if common else 0.0,
                "source_packet_count_match_ratio": float(local_k / len(common)) if common else 0.0,
            }

        parity = [r["mean_parity_packets"] for r in group]
        final_rec = [r["mean_final_recovery_ratio"] for r in group]
        fec_nmse = [r["mean_fec_nmse"] for r in group]
        cross["by_plr"][str(plr)] = {
            "condition_match": condition_match_ratios,
            "all_source_loss_fingerprints_match": bool(matched > 0 and fingerprint_matches == matched),
            "all_source_packet_counts_match": bool(matched > 0 and source_count_matches == matched),
            "mean_parity_packets_non_decreasing": bool(all(parity[i] <= parity[i + 1] + 1e-12 for i in range(len(parity) - 1))),
            "mean_final_recovery_ratio_non_decreasing": bool(all(final_rec[i] <= final_rec[i + 1] + 1e-12 for i in range(len(final_rec) - 1))),
            "mean_fec_nmse_non_increasing": bool(all(fec_nmse[i] + 1e-12 >= fec_nmse[i + 1] for i in range(len(fec_nmse) - 1))),
        }

    payload = {"conditions": rows, "cross_condition_checks": cross}
    os.makedirs(root, exist_ok=True)
    json_path = os.path.join(root, "experiment3_summary.json")
    csv_path = os.path.join(root, "experiment3_summary.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    fields = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== Experiment 3 summary =====")
    for r in rows:
        print(
            "PLR=%-4.2f rho=%-4.2f pass=%-5s parity=%-8s direct=%.4f final=%.4f "
            "fec_rec=%-8s direct_nmse=%.5f fec_nmse=%.5f AP@0.7=%s" % (
                r["plr"], r["rho"], str(r["pass"]), str(r["mean_parity_packets"]),
                float(r["mean_direct_recovery_ratio"] or 0.0),
                float(r["mean_final_recovery_ratio"] or 0.0),
                str(r["mean_fec_recovered_source_packets"]),
                float(r["mean_direct_nmse"] or 0.0), float(r["mean_fec_nmse"] or 0.0),
                str(r["ap_07"]),
            )
        )
    print("cross_condition_checks:", json.dumps(cross, ensure_ascii=False, indent=2))
    print("saved:", json_path)
    print("saved:", csv_path)

    if args.strict:
        ok = all(bool(r["pass"]) for r in rows)
        for item in cross["by_plr"].values():
            ok = ok and item["all_source_loss_fingerprints_match"] and item["all_source_packet_counts_match"]
        if not ok:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
