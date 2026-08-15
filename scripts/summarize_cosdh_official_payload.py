#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize a candidate-equivalence JSON produced by the verifier."""
from __future__ import print_function

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()

    source = Path(args.input).resolve()
    data = json.loads(source.read_text(encoding="utf-8"))
    samples = data.get("samples", [])
    if not samples:
        raise RuntimeError("no samples in {}".format(source))

    total_intermediate = 0
    total_late = 0
    total_candidates = 0
    total_links = 0
    zero_link_frames = 0
    scale_bytes = {}
    frame_bytes = []

    for sample in samples:
        info = sample.get("transport", {})
        inter = int(info.get("intermediate_source_bytes", 0))
        late = int(info.get("late_source_bytes", 0))
        total_intermediate += inter
        total_late += late
        frame_bytes.append(inter + late)
        records = info.get("late_records", [])
        links = len(records)
        total_links += links
        if links == 0:
            zero_link_frames += 1
        total_candidates += sum(
            int(item.get("candidate_count", 0)) for item in records
        )
        for record in info.get("scale_records", []):
            idx = int(record.get("scale_idx", -1))
            scale_bytes[idx] = scale_bytes.get(idx, 0) + int(
                record.get("source_bytes", 0)
            )

    frames = len(samples)
    total = total_intermediate + total_late
    summary = {
        "source": str(source),
        "frame_count": frames,
        "effective_late_links": total_links,
        "zero_link_frames": zero_link_frames,
        "total_intermediate_bytes": total_intermediate,
        "total_late_bytes": total_late,
        "total_bytes": total,
        "average_bytes_per_frame": total / float(frames),
        "average_MB_per_frame_decimal": total / float(frames) / 1e6,
        "average_bytes_per_effective_link": (
            total / float(total_links) if total_links else 0.0
        ),
        "average_candidates_per_effective_link": (
            total_candidates / float(total_links) if total_links else 0.0
        ),
        "late_bytes_per_candidate": (
            total_late / float(total_candidates) if total_candidates else 0.0
        ),
        "intermediate_fraction": (
            total_intermediate / float(total) if total else 0.0
        ),
        "late_fraction": total_late / float(total) if total else 0.0,
        "scale_source_bytes": {
            str(key): value for key, value in sorted(scale_bytes.items())
        },
        "minimum_frame_bytes": min(frame_bytes),
        "maximum_frame_bytes": max(frame_bytes),
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)

    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    if args.output_csv:
        with Path(args.output_csv).open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for key, value in summary.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        writer.writerow(["{}.{}".format(key, subkey), subvalue])
                else:
                    writer.writerow([key, value])


if __name__ == "__main__":
    main()
