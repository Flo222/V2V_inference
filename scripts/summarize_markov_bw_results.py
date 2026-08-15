#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge all per-baseline summary.json files."""

from __future__ import print_function

import argparse
import csv
import json
from pathlib import Path


COLUMNS = [
    "dataset",
    "baseline",
    "name",
    "evaluated_frame_count",
    "communication_frame_count",
    "total_collaborator_opportunities",
    "unique_transmitted_link_count",
    "total_tx_MB",
    "avg_total_tx_MB_per_frame",
    "avg_tx_MB_per_communication_frame",
    "avg_tx_MB_per_collaborator_opportunity",
    "total_rx_valid_MB",
    "total_source_before_budget_MB",
    "total_budget_truncated_MB",
    "state_link_counts",
    "policy_detected",
    "record_extraction_pass",
    "pass",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    rows = []

    for path in sorted(root.glob("*/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        row = dict(data)
        row["policy_detected"] = (
            data.get("policy_audit", {}).get("policy_detected")
        )
        rows.append(row)

    json_path = root / "all_baselines_markov_bw_summary.json"
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = root / "all_baselines_markov_bw_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for item in rows:
            row = {}
            for key in COLUMNS:
                value = item.get(key)
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)
                row[key] = value
            writer.writerow(row)

    print("\n===== Combined Markov BW Summary =====")
    for item in rows:
        print(
            "{:<28s} frames={:<4} tx={:>10.6f} MB/frame "
            "comm_frame={:>10.6f} MB pass={}".format(
                str(item.get("name", "")),
                int(item.get("evaluated_frame_count", 0)),
                float(item.get("avg_total_tx_MB_per_frame") or 0.0),
                float(item.get("avg_tx_MB_per_communication_frame") or 0.0),
                bool(item.get("pass", False)),
            )
        )
    print("CSV :", csv_path)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()
