#!/usr/bin/env python
"""Build a final summary for separate AP-only/BW-only evaluation runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


AP_PATTERN = re.compile(
    r"Average Precision at IOU 0\.3 is ([0-9.]+), "
    r"The Average Precision at IOU 0\.5 is ([0-9.]+), "
    r"The Average Precision at IOU 0\.7 is ([0-9.]+)"
)


def extract_ap(ap_log: Path) -> Tuple[float, float, float]:
    if not ap_log.is_file():
        raise FileNotFoundError("AP log does not exist: {}".format(ap_log))

    match = None
    with ap_log.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            current = AP_PATTERN.search(line)
            if current:
                match = current

    if match is None:
        raise RuntimeError(
            "No final AP line was found in: {}".format(ap_log)
        )
    return tuple(float(value) for value in match.groups())


def load_bw(bw_json: Path) -> Dict[str, Any]:
    if not bw_json.is_file():
        raise FileNotFoundError("BW summary does not exist: {}".format(bw_json))
    with bw_json.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError("BW summary must contain a JSON object.")
    return value


def build_summary(
    out_dir: Path,
    method: str,
    scenario: str,
    include_ap: bool,
    include_bw: bool,
) -> Dict[str, Any]:
    if not include_ap and not include_bw:
        raise ValueError("At least one of include_ap/include_bw must be true.")

    ap03: Optional[float] = None
    ap05: Optional[float] = None
    ap07: Optional[float] = None
    if include_ap:
        ap03, ap05, ap07 = extract_ap(out_dir / "ap.log")

    bw = load_bw(out_dir / "bw.json") if include_bw else {}
    protocol = (
        "separate_pass_ap_bw"
        if include_ap and include_bw
        else "ap_only"
        if include_ap
        else "bw_only"
    )

    return {
        "Method": str(method),
        "Scenario": str(scenario),
        "evaluation_protocol": protocol,
        "AP@0.3-{}".format(scenario): ap03,
        "AP@0.5-{}".format(scenario): ap05,
        "AP@0.7-{}".format(scenario): ap07,
        "BW-{}".format(scenario): bw.get("BW"),
        "total_tx_MB": bw.get("total_tx_MB"),
        "frame_count": bw.get("frame_count"),
        "record_count": bw.get("record_count"),
        "transmitted_link_count": bw.get("transmitted_link_count"),
        "no_send_count": bw.get("no_send_count"),
        "int4_count": bw.get("int4_count"),
        "packed_int4_count": bw.get("packed_int4_count"),
        "all_int4_packed": bw.get("all_int4_packed"),
    }


def save_summary(out_dir: Path, row: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "final_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(row, file, indent=2, ensure_ascii=False)

    csv_path = out_dir / "final_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    print(json.dumps(row, indent=2, ensure_ascii=False))
    print("saved:", summary_path)
    print("saved:", csv_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--include_ap", type=int, choices=(0, 1), required=True)
    parser.add_argument("--include_bw", type=int, choices=(0, 1), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    row = build_summary(
        out_dir=out_dir,
        method=args.method,
        scenario=args.scenario,
        include_ap=bool(args.include_ap),
        include_bw=bool(args.include_bw),
    )
    save_summary(out_dir, row)


if __name__ == "__main__":
    main()
