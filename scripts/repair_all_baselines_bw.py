#!/usr/bin/env python3
"""Repair the bandwidth portion of an existing all-baselines evaluation.

The initial runner selected BW channel mode by searching the runtime path for
``markov``.  Since the common output root itself contains that word, all Ideal
audits were unintentionally run as Markov.  This utility never reruns AP: it
reuses the existing runtime directories, reruns only BW summaries that are
missing/invalid/wrong-mode, and then updates the corresponding result rows.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

from run_all_baselines_ideal_markov import (
    OUT_ROOT, ROOT, RUNS, SEED, bandwidth_command, read_bandwidth,
)


def has_ap(row: dict) -> bool:
    return all(row.get(key) is not None for key in ("ap0_3", "ap0_5", "ap0_7"))


def valid_summary(path: Path, mode: str) -> bool:
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(summary.get("pass")) and summary.get("channel_mode") == mode


def run_audit(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        result = subprocess.run(command, cwd=str(ROOT), stdout=log,
                                stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError("bandwidth audit failed ({}) - see {}".format(
            result.returncode, log_path))


def write_global_summary(out_root: Path) -> None:
    rows = []
    for dataset, baseline, _source, _markov in RUNS:
        for mode in ("ideal", "markov"):
            result = out_root / dataset / baseline / mode / "result.json"
            if result.exists():
                rows.append(json.loads(result.read_text(encoding="utf-8")))
    fields = ["dataset", "baseline", "channel_mode", "metric_type", "ap0_3",
              "ap0_5", "ap0_7", "bw_tx_MB_per_frame",
              "bw_tx_bytes_per_frame", "bw_summary_pass", "checkpoint",
              "checkpoint_sha256", "runtime_config", "status",
              "elapsed_minutes", "error"]
    with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (out_root / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--only", default=None,
                        help="optional dataset/baseline filter, e.g. opv2v/v2xvit")
    parser.add_argument("--mode", choices=("ideal", "markov", "both"), default="both",
                        help="channel mode to audit (default: both)")
    parser.add_argument("--force", action="store_true",
                        help="rerun even a valid same-mode audit")
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    repaired = skipped = failed = 0

    for dataset, baseline, _source, _markov in RUNS:
        key = "{}/{}".format(dataset, baseline)
        if args.only and args.only not in key:
            continue
        if baseline == "nofusion":
            # No Fusion deliberately has no sender-to-ego payload; its zero
            # record is already canonical and does not use wire_bw_audit.py.
            skipped += 2
            continue
        for mode in ("ideal", "markov"):
            if args.mode != "both" and mode != args.mode:
                continue
            run_dir = out_root / dataset / baseline / mode
            result_path = run_dir / "result.json"
            runtime = run_dir / "model_runtime"
            summary_path = run_dir / "bandwidth" / "summary.json"
            if not runtime.exists():
                print("MISSING_RUNTIME {} {}".format(key, mode), flush=True)
                failed += 1
                continue
            if not args.force and valid_summary(summary_path, mode):
                action = "reused"
                skipped += 1
            else:
                action = "reran"
                try:
                    run_audit(
                        bandwidth_command(dataset, baseline, mode, runtime,
                                          run_dir / "bandwidth"),
                        run_dir / "bandwidth.log")
                    if not valid_summary(summary_path, mode):
                        raise RuntimeError("audit wrote a non-passing or wrong-mode summary")
                    repaired += 1
                except Exception as exc:
                    failed += 1
                    row = (json.loads(result_path.read_text(encoding="utf-8"))
                           if result_path.exists() else
                           {"dataset": dataset, "baseline": baseline,
                            "channel_mode": mode})
                    row["status"] = "failed"
                    row["error"] = str(exc)
                    row["bw_summary_pass"] = False
                    result_path.write_text(json.dumps(row, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
                    print("FAILED {} {}: {}".format(key, mode, exc), flush=True)
                    continue

            row = (json.loads(result_path.read_text(encoding="utf-8"))
                   if result_path.exists() else
                   {"dataset": dataset, "baseline": baseline,
                    "channel_mode": mode})
            row.update(read_bandwidth(summary_path))
            row["bw_audit_action"] = action
            row["bw_audit_repaired_unix"] = round(time.time(), 3)
            if has_ap(row):
                row["status"] = "complete"
                row.pop("error", None)
            else:
                row["status"] = "bw_complete_ap_missing"
                row["error"] = "Bandwidth audit passed; AP inference still needs repair."
            result_path.write_text(json.dumps(row, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
            print("{} {} {} BW={:.6f} MB/frame".format(
                action.upper(), key, mode, row["bw_tx_MB_per_frame"]), flush=True)

    write_global_summary(out_root)
    print("BW_REPAIR_DONE repaired={} reused={} failed={}".format(
        repaired, skipped, failed), flush=True)
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
