#!/usr/bin/env python3
from __future__ import print_function

import argparse
import csv
import json
from pathlib import Path


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    root = Path(args.input_root).expanduser().resolve()
    summaries = []
    for path in sorted(root.rglob("summary.json")):
        try:
            item = load_json(path)
        except Exception:
            continue
        item["summary_path"] = str(path)
        summaries.append(item)

    grouped = {}
    for item in summaries:
        key = (str(item.get("dataset")), str(item.get("baseline")))
        grouped.setdefault(key, {})[str(item.get("channel_mode", "unknown"))] = item

    rows = []
    for (dataset, baseline), pair in sorted(grouped.items()):
        ideal = pair.get("ideal", {})
        markov = pair.get("markov", {})
        ideal_bw = ideal.get("avg_total_tx_MB_per_frame")
        markov_bw = markov.get("avg_total_tx_MB_per_frame")
        delta = None
        ratio = None
        if ideal_bw is not None and markov_bw is not None:
            delta = float(markov_bw) - float(ideal_bw)
            if float(ideal_bw) != 0.0:
                ratio = float(markov_bw) / float(ideal_bw)
        rows.append({
            "dataset": dataset,
            "baseline": baseline,
            "ideal_frames": ideal.get("evaluated_frame_count"),
            "ideal_tx_MB_per_frame": ideal_bw,
            "ideal_source_MB": ideal.get("total_source_before_budget_MB"),
            "markov_frames": markov.get("evaluated_frame_count"),
            "markov_tx_MB_per_frame": markov_bw,
            "markov_rx_MB_total": markov.get("total_rx_valid_MB"),
            "markov_source_MB_total": markov.get("total_source_before_budget_MB"),
            "markov_minus_ideal_MB_per_frame": delta,
            "markov_to_ideal_ratio": ratio,
            "ideal_pass": ideal.get("pass"),
            "markov_pass": markov.get("pass"),
            "ideal_summary": ideal.get("summary_path"),
            "markov_summary": markov.get("summary_path"),
        })

    out_csv = Path(args.output_csv)
    out_json = Path(args.output_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys()) if rows else ["dataset", "baseline"]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)

    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
