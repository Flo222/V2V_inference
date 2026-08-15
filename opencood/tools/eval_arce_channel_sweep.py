"""
Channel sweep evaluation entry for ARCE / OpenCOOD / OPV2V.

This script repeatedly calls:

    opencood/tools/inference_arce.py

with different fixed ARCE channel states:

    good
    medium
    bad

For each channel state, it creates an isolated run directory and saves:

    stdout.log
    command.txt
    command.json
    comm_logs/
      arce_comm_records.jsonl
      arce_comm_flat.jsonl
      arce_comm_flat.csv
      arce_comm_summary.json
      arce_comm_summary_full.json
      arce_comm_summary_from_logs.json
      arce_comm_summary_full_from_logs.json
      arce_comm_flat_from_logs.csv
      arce_comm_group_summary_from_logs.csv
      arce_comm_report_from_logs.md

After all runs, this script writes sweep-level summaries:

    sweep_runs.jsonl
    sweep_summary.json
    sweep_summary.csv
    sweep_state_summary.csv
    sweep_report.md

Typical usage:

    python opencood/tools/eval_arce_channel_sweep.py \
      --model_dir opencood/logs/your_arce_model \
      --fusion_method intermediate \
      --states good medium bad \
      --max_samples 100

Quick debug:

    python opencood/tools/eval_arce_channel_sweep.py \
      --model_dir opencood/logs/your_arce_model \
      --fusion_method intermediate \
      --states medium \
      --max_samples 5 \
      --overwrite

This script does NOT:
    - implement ARCE communication;
    - modify model weights;
    - train a model;
    - compute new communication records by itself.

It only orchestrates ARCE inference runs and summarizes communication logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALID_CHANNEL_STATES = ("good", "medium", "bad")

DEFAULT_PREFIX = "arce_comm"

COMPACT_KEYS = (
    "num_records",
    "num_non_bypassed_records",
    "total_transmitted_mb",
    "total_received_mb",
    "total_source_packets",
    "total_encoded_packets",
    "total_lost_packets",
    "overall_packet_loss_ratio",
    "total_fec_recovered_packets",
    "total_temporal_filled_packets",
    "total_spatial_filled_packets",
    "total_zero_filled_packets",
    "total_still_missing_packets",
    "mean_packet_loss_ratio",
    "mean_recovery_ratio",
)


# ----------------------------------------------------------------------
# Argument parser
# ----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """
    Build command-line parser.
    """
    parser = argparse.ArgumentParser(
        description="Run ARCE fixed-channel sweep: Good / Medium / Bad."
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="OpenCOOD model directory containing config.yaml and checkpoint.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Sweep output directory. "
            "Default: <model_dir>/arce_channel_sweep"
        ),
    )

    parser.add_argument(
        "--states",
        type=str,
        nargs="+",
        default=["good", "medium", "bad"],
        help=(
            "Channel states to evaluate. "
            "Use: good medium bad, or all."
        ),
    )

    parser.add_argument(
        "--fusion_method",
        type=str,
        default="intermediate",
        choices=[
            "intermediate",
            "late",
            "early",
            "no",
            "no_fusion",
            "single",
        ],
        help="Fusion method passed to inference_arce.py.",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Maximum test samples per run. -1 means all.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers passed to inference_arce.py.",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeats per state when --seeds is not provided.",
    )

    parser.add_argument(
        "--seed_start",
        type=int,
        default=0,
        help="First seed when --seeds is not provided.",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Explicit seeds. If provided, each state is evaluated once per seed. "
            "--repeats is ignored."
        ),
    )

    parser.add_argument(
        "--late_policy",
        type=str,
        default=None,
        choices=["allow", "drop", "cache_only"],
        help="Override arce.late_policy for all runs.",
    )

    parser.add_argument(
        "--arce_mode",
        type=str,
        default=None,
        choices=["fixed", "bypass", "disabled"],
        help="Override arce.mode for all runs.",
    )

    parser.add_argument(
        "--arce_enabled",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Override arce.enabled for all runs.",
    )

    parser.add_argument(
        "--comm_prefix",
        type=str,
        default=DEFAULT_PREFIX,
        help="Communication log prefix.",
    )

    parser.add_argument(
        "--no_save_comm",
        action="store_true",
        help="Do not pass --save_comm to inference_arce.py.",
    )

    parser.add_argument(
        "--skip_bypassed_comm",
        action="store_true",
        help="Skip bypassed ego/local records in summaries.",
    )

    parser.add_argument(
        "--flatten_raw_for_csv",
        action="store_true",
        help="Ask inference_arce.py / CommLogger to flatten raw nested fields into CSV.",
    )

    parser.add_argument(
        "--comm_flush_every",
        type=int,
        default=0,
        help="Flush communication logs every N records during inference.",
    )

    parser.add_argument(
        "--save_eval_json",
        action="store_true",
        help="Pass --save_eval_json to inference_arce.py.",
    )

    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="Pass --save_vis to inference_arce.py.",
    )

    parser.add_argument(
        "--save_npy",
        action="store_true",
        help="Pass --save_npy to inference_arce.py.",
    )

    parser.add_argument(
        "--vis_interval",
        type=int,
        default=40,
        help="Visualization interval passed to inference_arce.py.",
    )

    parser.add_argument(
        "--summarize",
        action="store_true",
        default=True,
        help="Run summarize_comm_logs.py after each inference run.",
    )

    parser.add_argument(
        "--no_summarize",
        action="store_true",
        help="Disable summarize_comm_logs.py after each inference run.",
    )

    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to launch child scripts.",
    )

    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES for subprocesses.",
    )

    parser.add_argument(
        "--extra_inference_args",
        type=str,
        default="",
        help=(
            "Extra raw arguments appended to inference_arce.py command. "
            "Example: \"--some_flag value\""
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Subprocess timeout in seconds. 0 means no timeout.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing per-run directory before running.",
    )

    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a run if its from-logs summary already exists.",
    )

    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue sweep even if one run fails.",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print commands without running them.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output.",
    )

    return parser


# ----------------------------------------------------------------------
# Basic helpers
# ----------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    """
    Create directory if missing.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_safe(value: Any) -> Any:
    """
    Convert values to JSON-friendly types.
    """
    if value is None:
        return None

    if isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    return str(value)


def write_json(path: Path, data: Any, indent: int = 2) -> None:
    """
    Write JSON file.
    """
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, indent=indent, ensure_ascii=False)


def append_jsonl(path: Path, data: Any) -> None:
    """
    Append one JSON object to JSONL.
    """
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(data), ensure_ascii=False))
        f.write("\n")


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """
    Load JSON file. Return None if missing or invalid.
    """
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return None
    except Exception:
        return None


def collect_csv_fieldnames(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """
    Collect stable CSV fieldnames.
    """
    fieldnames: List[str] = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    return fieldnames


def csv_value(value: Any) -> Any:
    """
    Convert a value for CSV writing.
    """
    value = json_safe(value)

    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)

    return value


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """
    Write rows to CSV.
    """
    rows = [json_safe(row) for row in rows]

    fieldnames = collect_csv_fieldnames(rows)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fieldnames})


def shell_join(cmd: Sequence[str]) -> str:
    """
    Return shell-escaped command string.
    """
    return " ".join(shlex.quote(str(x)) for x in cmd)


def normalize_states(states: Sequence[str]) -> List[str]:
    """
    Normalize channel state list.

    Supports:
        --states all
        --states good medium bad
    """
    if not states:
        return list(VALID_CHANNEL_STATES)

    normalized: List[str] = []

    for state in states:
        state = str(state).strip().lower()

        if state == "all":
            for item in VALID_CHANNEL_STATES:
                if item not in normalized:
                    normalized.append(item)
            continue

        if state not in VALID_CHANNEL_STATES:
            raise ValueError(
                f"Unsupported channel state: {state}. "
                f"Expected one of {VALID_CHANNEL_STATES} or all."
            )

        if state not in normalized:
            normalized.append(state)

    return normalized


def resolve_seeds(opt: argparse.Namespace) -> List[int]:
    """
    Resolve seed list from --seeds or --repeats / --seed_start.
    """
    if opt.seeds is not None and len(opt.seeds) > 0:
        return [int(seed) for seed in opt.seeds]

    repeats = int(opt.repeats)

    if repeats <= 0:
        raise ValueError(f"--repeats should be positive, got {repeats}.")

    return [int(opt.seed_start) + idx for idx in range(repeats)]


def resolve_output_dir(opt: argparse.Namespace) -> Path:
    """
    Resolve sweep output directory.
    """
    if opt.output_dir is not None:
        return ensure_dir(Path(opt.output_dir).resolve())

    return ensure_dir(Path(opt.model_dir).resolve() / "arce_channel_sweep")


def compact_float(value: Any, digits: int = 6) -> str:
    """
    Format float for markdown.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if abs(value) >= 1000:
        return f"{value:.2f}"

    return f"{value:.{digits}f}"


# ----------------------------------------------------------------------
# Command building
# ----------------------------------------------------------------------

def build_inference_command(
    opt: argparse.Namespace,
    state: str,
    seed: int,
    comm_log_dir: Path,
) -> List[str]:
    """
    Build command for inference_arce.py.
    """
    script = PROJECT_ROOT / "opencood" / "tools" / "inference_arce.py"

    cmd = [
        str(opt.python),
        str(script),
        "--model_dir",
        str(Path(opt.model_dir).resolve()),
        "--fusion_method",
        str(opt.fusion_method),
        "--num_workers",
        str(int(opt.num_workers)),
        "--seed",
        str(int(seed)),
        "--comm_log_dir",
        str(comm_log_dir),
        "--comm_prefix",
        str(opt.comm_prefix),
        "--arce_channel_state",
        str(state),
        "--comm_flush_every",
        str(int(opt.comm_flush_every)),
    ]

    if int(opt.max_samples) > 0:
        cmd.extend(["--max_samples", str(int(opt.max_samples))])

    if not opt.no_save_comm:
        cmd.append("--save_comm")

    if opt.skip_bypassed_comm:
        cmd.append("--skip_bypassed_comm")

    if opt.flatten_raw_for_csv:
        cmd.append("--flatten_raw_for_csv")

    if opt.save_eval_json:
        cmd.append("--save_eval_json")

    if opt.save_vis:
        cmd.append("--save_vis")
        cmd.extend(["--vis_interval", str(int(opt.vis_interval))])

    if opt.save_npy:
        cmd.append("--save_npy")

    if opt.late_policy is not None:
        cmd.extend(["--arce_late_policy", str(opt.late_policy)])

    if opt.arce_mode is not None:
        cmd.extend(["--arce_mode", str(opt.arce_mode)])

    if opt.arce_enabled is not None:
        cmd.extend(["--arce_enabled", str(opt.arce_enabled)])

    if opt.extra_inference_args:
        cmd.extend(shlex.split(opt.extra_inference_args))

    return cmd


def build_summarize_command(
    opt: argparse.Namespace,
    comm_log_dir: Path,
) -> List[str]:
    """
    Build command for summarize_comm_logs.py.
    """
    script = PROJECT_ROOT / "opencood" / "tools" / "summarize_comm_logs.py"

    cmd = [
        str(opt.python),
        str(script),
        "--log_dir",
        str(comm_log_dir),
        "--prefix",
        str(opt.comm_prefix),
        "--all_outputs",
    ]

    if opt.skip_bypassed_comm:
        cmd.append("--skip_bypassed")

    if opt.quiet:
        cmd.append("--quiet")

    return cmd


def build_env(opt: argparse.Namespace) -> Dict[str, str]:
    """
    Build subprocess environment.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if "PYTHONPATH" not in env or not env["PYTHONPATH"]
        else str(PROJECT_ROOT) + os.pathsep + env["PYTHONPATH"]
    )

    if opt.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(opt.cuda_visible_devices)

    return env


# ----------------------------------------------------------------------
# Run helpers
# ----------------------------------------------------------------------

def run_command(
    cmd: Sequence[str],
    stdout_path: Path,
    env: Dict[str, str],
    timeout: int = 0,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Run command and redirect stdout/stderr to file.

    Returns
    -------
    dict
        status, return_code, elapsed_sec.
    """
    start = time.time()

    if dry_run:
        stdout_path.write_text(
            "[DRY RUN]\n" + shell_join(cmd) + "\n",
            encoding="utf-8",
        )
        return {
            "status": "dry_run",
            "return_code": 0,
            "elapsed_sec": 0.0,
            "timeout": False,
        }

    with stdout_path.open("w", encoding="utf-8") as f:
        f.write("[COMMAND]\n")
        f.write(shell_join(cmd))
        f.write("\n\n[OUTPUT]\n")
        f.flush()

        try:
            proc = subprocess.run(
                list(cmd),
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=None if int(timeout) <= 0 else int(timeout),
                check=False,
            )

            elapsed = time.time() - start

            return {
                "status": "success" if proc.returncode == 0 else "failed",
                "return_code": int(proc.returncode),
                "elapsed_sec": float(elapsed),
                "timeout": False,
            }

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            f.write("\n[TIMEOUT]\n")
            f.write(f"Command timed out after {timeout} seconds.\n")
            return {
                "status": "timeout",
                "return_code": -1,
                "elapsed_sec": float(elapsed),
                "timeout": True,
            }


def prepare_run_dir(
    run_dir: Path,
    overwrite: bool = False,
) -> Path:
    """
    Prepare per-run directory.
    """
    if run_dir.exists() and overwrite:
        shutil.rmtree(run_dir)

    ensure_dir(run_dir)
    return run_dir


def read_compact_summary(comm_log_dir: Path, prefix: str) -> Dict[str, Any]:
    """
    Read compact summary from summarizer output or logger output.

    Priority:
        1. <prefix>_summary_from_logs.json
        2. <prefix>_summary.json
    """
    from_logs_path = comm_log_dir / f"{prefix}_summary_from_logs.json"
    logger_path = comm_log_dir / f"{prefix}_summary.json"

    payload = load_json(from_logs_path)

    if payload is None:
        payload = load_json(logger_path)

    if payload is None:
        return {}

    if "compact" in payload and isinstance(payload["compact"], dict):
        return payload["compact"]

    if "overall" in payload and isinstance(payload["overall"], dict):
        # Fallback to some overall fields if compact is missing.
        overall = payload["overall"]
        return {
            "num_records": overall.get("num_records", 0),
            "num_non_bypassed_records": overall.get("num_non_bypassed_records", 0),
            "total_transmitted_mb": overall.get("total_transmitted_mb", 0.0),
            "total_received_mb": overall.get("total_received_mb", 0.0),
            "overall_packet_loss_ratio": overall.get("overall_packet_loss_ratio", 0.0),
        }

    return {}


def build_run_name(state: str, repeat_idx: int, seed: int) -> str:
    """
    Build deterministic per-run name.
    """
    return f"{state}_repeat{repeat_idx:02d}_seed{int(seed)}"


def run_one_sweep_item(
    opt: argparse.Namespace,
    state: str,
    repeat_idx: int,
    seed: int,
    output_dir: Path,
    env: Dict[str, str],
) -> Dict[str, Any]:
    """
    Run one channel-state / seed item.

    Returns
    -------
    dict
        Run row for sweep summary.
    """
    run_name = build_run_name(state, repeat_idx, seed)
    run_dir = output_dir / run_name
    comm_log_dir = run_dir / "comm_logs"

    prepare_run_dir(run_dir, overwrite=opt.overwrite)
    ensure_dir(comm_log_dir)

    summary_from_logs_path = comm_log_dir / f"{opt.comm_prefix}_summary_from_logs.json"

    base_row: Dict[str, Any] = {
        "run_name": run_name,
        "channel_state": state,
        "repeat_idx": int(repeat_idx),
        "seed": int(seed),
        "run_dir": str(run_dir),
        "comm_log_dir": str(comm_log_dir),
    }

    if opt.skip_existing and summary_from_logs_path.exists():
        compact = read_compact_summary(comm_log_dir, opt.comm_prefix)
        row = dict(base_row)
        row.update(
            {
                "status": "skipped_existing",
                "return_code": 0,
                "elapsed_sec": 0.0,
                "summarize_status": "existing",
                "summarize_return_code": 0,
            }
        )
        for key in COMPACT_KEYS:
            row[key] = compact.get(key, None)
        return row

    inference_cmd = build_inference_command(
        opt=opt,
        state=state,
        seed=seed,
        comm_log_dir=comm_log_dir,
    )

    command_txt = run_dir / "command.txt"
    command_json = run_dir / "command.json"
    stdout_log = run_dir / "stdout.log"

    command_txt.write_text(shell_join(inference_cmd) + "\n", encoding="utf-8")
    write_json(
        command_json,
        {
            "inference_command": inference_cmd,
            "inference_command_shell": shell_join(inference_cmd),
            "state": state,
            "seed": seed,
            "repeat_idx": repeat_idx,
            "run_dir": str(run_dir),
            "comm_log_dir": str(comm_log_dir),
        },
    )

    if not opt.quiet:
        print(f"[ARCE sweep] Running {run_name}")
        print(f"  command: {shell_join(inference_cmd)}")

    inference_status = run_command(
        cmd=inference_cmd,
        stdout_path=stdout_log,
        env=env,
        timeout=int(opt.timeout),
        dry_run=bool(opt.dry_run),
    )

    summarize_status = {
        "status": "not_run",
        "return_code": None,
        "elapsed_sec": 0.0,
        "timeout": False,
    }

    if inference_status["return_code"] == 0 and (opt.summarize and not opt.no_summarize):
        summarize_cmd = build_summarize_command(
            opt=opt,
            comm_log_dir=comm_log_dir,
        )
        summarize_log = run_dir / "summarize_stdout.log"

        if not opt.quiet:
            print(f"[ARCE sweep] Summarizing {run_name}")
            print(f"  command: {shell_join(summarize_cmd)}")

        summarize_status = run_command(
            cmd=summarize_cmd,
            stdout_path=summarize_log,
            env=env,
            timeout=int(opt.timeout),
            dry_run=bool(opt.dry_run),
        )

        command_payload = load_json(command_json) or {}
        command_payload["summarize_command"] = summarize_cmd
        command_payload["summarize_command_shell"] = shell_join(summarize_cmd)
        write_json(command_json, command_payload)

    compact = read_compact_summary(comm_log_dir, opt.comm_prefix)

    row = dict(base_row)
    row.update(
        {
            "status": inference_status["status"],
            "return_code": inference_status["return_code"],
            "elapsed_sec": inference_status["elapsed_sec"],
            "timeout": inference_status["timeout"],
            "summarize_status": summarize_status["status"],
            "summarize_return_code": summarize_status["return_code"],
            "summarize_elapsed_sec": summarize_status["elapsed_sec"],
            "stdout_log": str(stdout_log),
        }
    )

    for key in COMPACT_KEYS:
        row[key] = compact.get(key, None)

    write_json(run_dir / "run_result.json", row)

    return row


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------

def is_number(value: Any) -> bool:
    """
    Whether value is finite number-like.
    """
    if isinstance(value, bool):
        return False

    try:
        value = float(value)
    except (TypeError, ValueError):
        return False

    return math.isfinite(value)


def mean(values: Sequence[float]) -> Optional[float]:
    """
    Mean of values. Return None for empty.
    """
    if not values:
        return None

    return float(sum(values) / len(values))


def std(values: Sequence[float]) -> Optional[float]:
    """
    Population std of values. Return None for fewer than 2.
    """
    if len(values) < 2:
        return None

    m = mean(values)
    return float(math.sqrt(sum((x - m) ** 2 for x in values) / len(values)))


def aggregate_rows_by_state(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate sweep rows by channel state.

    Only successful / skipped_existing rows with numeric values are used.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for row in rows:
        state = str(row.get("channel_state", "unknown"))
        grouped.setdefault(state, []).append(row)

    output: List[Dict[str, Any]] = []

    for state, state_rows in grouped.items():
        item: Dict[str, Any] = {
            "channel_state": state,
            "num_runs": len(state_rows),
            "num_success": sum(1 for r in state_rows if r.get("status") in ("success", "skipped_existing", "dry_run")),
            "num_failed": sum(1 for r in state_rows if r.get("status") not in ("success", "skipped_existing", "dry_run")),
        }

        for key in COMPACT_KEYS:
            values = [
                float(row[key])
                for row in state_rows
                if key in row and is_number(row[key])
            ]

            item[f"mean_{key}"] = mean(values)
            item[f"std_{key}"] = std(values)
            item[f"min_{key}"] = min(values) if values else None
            item[f"max_{key}"] = max(values) if values else None

        output.append(item)

    state_order = {state: idx for idx, state in enumerate(VALID_CHANNEL_STATES)}
    output.sort(key=lambda x: state_order.get(x["channel_state"], 999))

    return output


def build_markdown_report(
    rows: Sequence[Dict[str, Any]],
    state_summary: Sequence[Dict[str, Any]],
    opt: argparse.Namespace,
    output_dir: Path,
) -> str:
    """
    Build sweep-level markdown report.
    """
    lines: List[str] = []

    lines.append("# ARCE Channel Sweep Report")
    lines.append("")
    lines.append(f"- Model dir: `{Path(opt.model_dir).resolve()}`")
    lines.append(f"- Output dir: `{output_dir}`")
    lines.append(f"- Fusion method: `{opt.fusion_method}`")
    lines.append(f"- Max samples: `{opt.max_samples}`")
    lines.append(f"- States: `{', '.join(normalize_states(opt.states))}`")
    lines.append(f"- Seeds: `{', '.join(str(s) for s in resolve_seeds(opt))}`")
    lines.append("")

    lines.append("## Run-level summary")
    lines.append("")
    lines.append(
        "| State | Seed | Status | Tx MB | Rx MB | Loss | Recovery | FEC Rec | Temporal | Spatial | Zero | Log |"
    )
    lines.append(
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    )

    for row in rows:
        log_dir = row.get("comm_log_dir", "")
        lines.append(
            "| "
            f"{row.get('channel_state')} | "
            f"{row.get('seed')} | "
            f"{row.get('status')} | "
            f"{compact_float(row.get('total_transmitted_mb', 0.0), 4)} | "
            f"{compact_float(row.get('total_received_mb', 0.0), 4)} | "
            f"{compact_float(row.get('overall_packet_loss_ratio', 0.0), 4)} | "
            f"{compact_float(row.get('mean_recovery_ratio', 0.0), 4)} | "
            f"{row.get('total_fec_recovered_packets', '')} | "
            f"{row.get('total_temporal_filled_packets', '')} | "
            f"{row.get('total_spatial_filled_packets', '')} | "
            f"{row.get('total_zero_filled_packets', '')} | "
            f"`{log_dir}` |"
        )

    lines.append("")
    lines.append("## State-level mean summary")
    lines.append("")
    lines.append(
        "| State | Runs | Success | Mean Tx MB | Mean Rx MB | Mean Loss | Mean Recovery | Mean FEC Rec | Mean Temporal | Mean Spatial | Mean Zero |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )

    for row in state_summary:
        lines.append(
            "| "
            f"{row.get('channel_state')} | "
            f"{row.get('num_runs')} | "
            f"{row.get('num_success')} | "
            f"{compact_float(row.get('mean_total_transmitted_mb', 0.0), 4)} | "
            f"{compact_float(row.get('mean_total_received_mb', 0.0), 4)} | "
            f"{compact_float(row.get('mean_overall_packet_loss_ratio', 0.0), 4)} | "
            f"{compact_float(row.get('mean_mean_recovery_ratio', 0.0), 4)} | "
            f"{compact_float(row.get('mean_total_fec_recovered_packets', 0.0), 2)} | "
            f"{compact_float(row.get('mean_total_temporal_filled_packets', 0.0), 2)} | "
            f"{compact_float(row.get('mean_total_spatial_filled_packets', 0.0), 2)} | "
            f"{compact_float(row.get('mean_total_zero_filled_packets', 0.0), 2)} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Each row corresponds to one full inference run under a fixed channel state.")
    lines.append("- Good / Medium / Bad state behavior is controlled by ARCE YAML and `FixedARCEPolicy`.")
    lines.append("- GE packet loss acts on encoded packets, meaning source packets plus FEC parity / repair packets.")
    lines.append("- Detection AP is printed in each run's `stdout.log`; this sweep script mainly aggregates communication metrics.")
    lines.append("")

    return "\n".join(lines)


def write_sweep_outputs(
    rows: Sequence[Dict[str, Any]],
    state_summary: Sequence[Dict[str, Any]],
    opt: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, str]:
    """
    Write sweep-level output files.
    """
    paths = {
        "runs_jsonl": output_dir / "sweep_runs.jsonl",
        "summary_json": output_dir / "sweep_summary.json",
        "summary_csv": output_dir / "sweep_summary.csv",
        "state_summary_csv": output_dir / "sweep_state_summary.csv",
        "report_md": output_dir / "sweep_report.md",
    }

    if paths["runs_jsonl"].exists():
        paths["runs_jsonl"].unlink()

    for row in rows:
        append_jsonl(paths["runs_jsonl"], row)

    write_json(
        paths["summary_json"],
        {
            "config": {
                "model_dir": str(Path(opt.model_dir).resolve()),
                "output_dir": str(output_dir),
                "states": normalize_states(opt.states),
                "seeds": resolve_seeds(opt),
                "fusion_method": opt.fusion_method,
                "max_samples": int(opt.max_samples),
                "skip_bypassed_comm": bool(opt.skip_bypassed_comm),
                "late_policy": opt.late_policy,
                "arce_mode": opt.arce_mode,
                "arce_enabled": opt.arce_enabled,
            },
            "runs": list(rows),
            "state_summary": list(state_summary),
        },
    )

    write_csv(paths["summary_csv"], rows)
    write_csv(paths["state_summary_csv"], state_summary)

    report = build_markdown_report(
        rows=rows,
        state_summary=state_summary,
        opt=opt,
        output_dir=output_dir,
    )
    paths["report_md"].write_text(report, encoding="utf-8")

    return {key: str(path) for key, path in paths.items()}


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(opt: argparse.Namespace) -> None:
    """
    Main channel sweep entry.
    """
    states = normalize_states(opt.states)
    seeds = resolve_seeds(opt)
    output_dir = resolve_output_dir(opt)
    ensure_dir(output_dir)

    env = build_env(opt)

    if not opt.quiet:
        print("[ARCE sweep] Configuration")
        print(f"  model_dir: {Path(opt.model_dir).resolve()}")
        print(f"  output_dir: {output_dir}")
        print(f"  states: {states}")
        print(f"  seeds: {seeds}")
        print(f"  fusion_method: {opt.fusion_method}")
        print(f"  max_samples: {opt.max_samples}")

    rows: List[Dict[str, Any]] = []

    start_time = time.time()

    for state in states:
        for repeat_idx, seed in enumerate(seeds):
            try:
                row = run_one_sweep_item(
                    opt=opt,
                    state=state,
                    repeat_idx=repeat_idx,
                    seed=seed,
                    output_dir=output_dir,
                    env=env,
                )
                rows.append(row)

                if not opt.quiet:
                    print(
                        "[ARCE sweep] Done "
                        f"{row['run_name']} status={row['status']} "
                        f"tx={row.get('total_transmitted_mb')}MB "
                        f"loss={row.get('overall_packet_loss_ratio')}"
                    )

                if row.get("return_code", 0) != 0 and not opt.continue_on_error:
                    raise RuntimeError(
                        f"Run failed: {row['run_name']}. "
                        f"See {row.get('stdout_log')}"
                    )

            except Exception as exc:
                error_row = {
                    "run_name": build_run_name(state, repeat_idx, seed),
                    "channel_state": state,
                    "repeat_idx": int(repeat_idx),
                    "seed": int(seed),
                    "status": "exception",
                    "return_code": -1,
                    "error": str(exc),
                }
                rows.append(error_row)

                if not opt.continue_on_error:
                    state_summary = aggregate_rows_by_state(rows)
                    write_sweep_outputs(
                        rows=rows,
                        state_summary=state_summary,
                        opt=opt,
                        output_dir=output_dir,
                    )
                    raise

                print(f"[WARN] Run exception but continue_on_error=True: {exc}")

    elapsed = time.time() - start_time

    state_summary = aggregate_rows_by_state(rows)

    output_paths = write_sweep_outputs(
        rows=rows,
        state_summary=state_summary,
        opt=opt,
        output_dir=output_dir,
    )

    if not opt.quiet:
        print(f"[ARCE sweep] Finished in {elapsed:.2f}s")
        print("[ARCE sweep] Output files:")
        for key, path in output_paths.items():
            print(f"  - {key}: {path}")

        print("[ARCE sweep] State summary:")
        for row in state_summary:
            print(
                f"  {row['channel_state']}: "
                f"runs={row['num_runs']} "
                f"success={row['num_success']} "
                f"tx={row.get('mean_total_transmitted_mb')}MB "
                f"loss={row.get('mean_overall_packet_loss_ratio')} "
                f"recovery={row.get('mean_mean_recovery_ratio')}"
            )


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)