#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the Markov BW audit manifest in isolated subprocesses."""

from __future__ import print_function

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--max_frames", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated run names. Empty runs all.",
    )
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--allow_policy", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = yaml.safe_load(
        manifest_path.read_text(encoding="utf-8")
    )
    runs = manifest.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise ValueError("Manifest must contain a non-empty runs list.")

    selected = set(
        item.strip()
        for item in str(args.only).split(",")
        if item.strip()
    )

    script_dir = Path(__file__).resolve().parent
    audit_script = script_dir / "markov_bw_audit.py"
    statuses = []

    for run in runs:
        name = str(run["name"])
        if selected and name not in selected:
            continue

        run_out = out_root / name
        command = [
            sys.executable,
            str(audit_script),
            "--name", name,
            "--dataset", str(run["dataset"]),
            "--baseline", str(run["baseline"]),
            "--model_dir", str(run["model_dir"]),
            "--epoch", str(run.get("epoch", "auto")),
            "--max_frames", str(args.max_frames),
            "--num_workers", str(args.num_workers),
            "--seed", str(args.seed),
            "--out_dir", str(run_out),
        ]

        if run.get("test_dir"):
            command.extend(["--test_dir", str(run["test_dir"])])
        if args.allow_policy:
            command.append("--allow_policy")

        print("\n===== {} =====".format(name), flush=True)
        print(" ".join(command), flush=True)

        log_path = run_out / "run.log"
        run_out.mkdir(parents=True, exist_ok=True)

        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                cwd=str(Path.cwd()),
            )
            for line in process.stdout:
                sys.stdout.write(line)
                log_file.write(line)
            return_code = process.wait()

        statuses.append({
            "name": name,
            "return_code": int(return_code),
            "out_dir": str(run_out),
            "log": str(log_path),
        })

        if return_code != 0 and not args.continue_on_error:
            break

    status_path = out_root / "run_status.json"
    status_path.write_text(
        json.dumps(statuses, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summarize_script = script_dir / "summarize_markov_bw_results.py"
    subprocess.check_call([
        sys.executable,
        str(summarize_script),
        "--root", str(out_root),
    ])

    failures = [item for item in statuses if item["return_code"] != 0]
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
