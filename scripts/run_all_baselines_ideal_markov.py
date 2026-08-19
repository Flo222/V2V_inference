#!/usr/bin/env python3
"""Full-test AP and wire-bandwidth evaluation under Ideal and Markov channels.

Each baseline/dataset has one *ideal checkpoint package* (config + newest
numeric checkpoint).  The Markov run creates a separate runtime directory
with the Markov communication config, but links/copies that exact same
checkpoint.  Therefore a Markov directory never represents a second model
checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/home/server/v2x_projects/V2V_inference")
OLD_LOGS = Path("/home/server/v2x_projects/OPV2V/opencood/logs")
OUT_ROOT = ROOT / "opencood/logs/all_baselines_ideal_markov_test_20260815"
CKPT_ROOT = ROOT / "opencood/logs/ideal_checkpoints_20260815"
OPV2V_TEST = "/home/server/v2x_projects/OPV2V/opv2v_data_dumping/test"
V2XREAL_TEST = "/data/v2xreal/test"
SEED = 2026

# The primary V2X-ViT experiment uses the compression=32 models fine-tuned
# under perfect communication.  Ideal and Markov therefore materialize the
# exact same checkpoint; only the runtime channel config differs.
CHECKPOINT_SOURCE_OVERRIDES = {
    ("opv2v", "v2xvit"):
        ROOT / "opencood/logs/point_pillar_v2xvit_opv2v_compression32_perfect_ft",
    ("v2xreal", "v2xvit"):
        ROOT / "opencood/logs/point_pillar_v2xvit_v2xreal_compression32_perfect_ft",
}
MARKOV_CONFIG_OVERRIDES = {
    ("opv2v", "v2xvit"):
        ROOT / "opencood/logs/compat_configs_20260818/v2xvit_opv2v_markov_compression32/config.yaml",
    ("v2xreal", "v2xvit"):
        ROOT / "opencood/logs/compat_configs_20260818/v2xvit_v2xreal_markov_compression32/config.yaml",
}

# source: the ideal-trained checkpoint/config source supplied by the original
# baseline logs. markov_config_source supplies only the runtime channel config.
RUNS = [
    ("opv2v", "nofusion", "point_pillar_nofusion_opv2v_range140_e20_2026_06_29_22_58_15", None),
    ("opv2v", "v2xvit", "point_pillar_v2xvit_opv2v_2026_05_13_20_33_54", "point_pillar_v2xvit_opv2v_2026_05_13_20_33_54_arce_eval"),
    ("opv2v", "where2comm", "point_pillar_where2comm_2026_06_09_12_48_25", "where2comm_markov_trueloss_fp32_rho0_cache0"),
    ("opv2v", "cosdh", "opv2v_cosdh_2026_06_16_12_43_59", "opv2v_cosdh_markov_byte_2026_06_16"),
    ("opv2v", "rocooper", "point_pillar_rocooper_opv2v_2026_06_22_12_52_04", "point_pillar_rocooper_opv2v_2026_06_22_12_52_04"),
    ("opv2v", "coopdiff", "point_pillar_diffstudent_opv2v_e30_mapped_from_teacher_e20", "point_pillar_diffstudent_opv2v_e30_mapped_from_teacher_e20_markov_eval"),
    ("v2xreal", "nofusion", "point_pillar_nofusion_v2xreal_vc_2026_06_30_18_51_35", None),
    ("v2xreal", "v2xvit", "point_pillar_v2xvit_v2xreal_vc_2026_05_11_22_45_46", "point_pillar_v2xvit_v2xreal_markov_eval"),
    ("v2xreal", "where2comm", "point_pillar_where2comm_v2xreal_vc_2026_06_21_16_32_43", "point_pillar_where2comm_v2xreal_markov_eval"),
    ("v2xreal", "cosdh", "point_pillar_cosdh_v2xreal_vc_2026_06_21_23_50_49", "point_pillar_cosdh_markov_v2xreal_eval"),
    ("v2xreal", "rocooper", "point_pillar_rocooper_markov_v2xreal_vc_2026_06_29_16_37_39", "point_pillar_rocooper_markov_v2xreal_vc_2026_06_29_16_37_39"),
    ("v2xreal", "coopdiff", "point_pillar_diffstudent_v2xreal_vc_mapped_from_teacher_partialvfe_backbone20", "point_pillar_diffstudent_v2xreal_vc_mapped_from_teacher_partialvfe_backbone20_markov_eval"),
]


def latest_checkpoint(directory: Path) -> Path:
    candidates = []
    for path in directory.glob("net_epoch*.pth"):
        match = re.fullmatch(r"net_epoch(\d+)\.pth", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError("No numeric net_epoch*.pth in {}".format(directory))
    return max(candidates)[1]


def config_file(directory: Path) -> Path:
    direct = directory / "config.yaml"
    if direct.exists():
        return direct
    configs = sorted(directory.glob("config*.yaml"))
    if not configs:
        raise FileNotFoundError("No config*.yaml in {}".format(directory))
    return configs[0]


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def materialize_runtime(target: Path, checkpoint: Path, config: Path, meta: dict) -> None:
    reset_dir(target)
    try:
        (target / checkpoint.name).symlink_to(checkpoint)
        linked = True
    except OSError:
        shutil.copy2(checkpoint, target / checkpoint.name)
        linked = False
    shutil.copy2(config, target / "config.yaml")
    # Historical OPV2V logs use a relative test split path.  A runtime
    # directory lives under V2V_inference, so preserve the config otherwise
    # but make only the formal test path absolute.
    dataset = meta.get("dataset")
    if dataset in ("opv2v", "v2xreal"):
        test_dir = OPV2V_TEST if dataset == "opv2v" else V2XREAL_TEST
        runtime_config = target / "config.yaml"
        config_text = runtime_config.read_text(encoding="utf-8")
        config_text = re.sub(
            r"^(\s*(?:validate_dir|test_dir):\s*)[^#\n]+",
            lambda match: match.group(1) + test_dir,
            config_text,
            flags=re.MULTILINE,
        )
        runtime_config.write_text(config_text, encoding="utf-8")
    meta = dict(meta)
    meta.update({
        "checkpoint_source": str(checkpoint),
        "checkpoint_sha256": checksum(checkpoint),
        "checkpoint_runtime_path": str(target / checkpoint.name),
        "checkpoint_is_symlink": linked,
        "runtime_config_source": str(config),
    })
    (target / "checkpoint_provenance.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        child_env = dict(os.environ)
        child_env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        completed = subprocess.run(command, cwd=str(ROOT), stdout=log,
                                   stderr=subprocess.STDOUT, text=True,
                                   env=child_env)
    if completed.returncode:
        raise RuntimeError("command failed ({}) - see {}".format(completed.returncode, log_path))


def ap_command(dataset: str, baseline: str, runtime: Path) -> list[str]:
    if dataset == "opv2v":
        if baseline == "cosdh":
            return [sys.executable, "opencood/tools/inference_cosdh.py", "--model_dir", str(runtime), "--fusion_method", "intermediatelate"]
        if baseline == "coopdiff":
            return [sys.executable, "opencood/tools/inference_coopdiff_markov.py", "--model_dir", str(runtime), "--fusion_method", "intermediate", "--max_samples", "-1", "--num_workers", "0", "--seed", str(SEED)]
        command = [sys.executable, "opencood/tools/inference.py", "--model_dir", str(runtime), "--fusion_method", "intermediate", "--max_frames", "-1", "--num_workers", "0"]
        if baseline != "nofusion":
            command += ["--seed", str(SEED)]
        return command
    if baseline == "cosdh":
        return [sys.executable, "opencood/tools/inference_cosdh.py", "--model_dir", str(runtime), "--fusion_method", "intermediatelate"]
    if baseline == "coopdiff":
        return [sys.executable, "opencood/tools/inference_coopdiff_v2xreal.py", "--model_dir", str(runtime), "--fusion_method", "intermediate", "--dataset_mode", "vc", "--max_samples", "-1", "--num_workers", "0", "--seed", str(SEED)]
    # The V2X-Real No Fusion checkpoint has max_cav=1 and uses the normal
    # intermediate dataset/model interface.  This codebase has no separate
    # inference_nofusion helper, so intermediate is the correct single-agent
    # execution path; bandwidth remains explicitly zero below.
    return [sys.executable, "opencood/tools/inference_v2xreal.py", "--model_dir", str(runtime), "--fusion_method", "intermediate", "--dataset_mode", "vc"]


def parse_ap(log_path: Path, dataset: str) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    values = {}
    # Covers OPV2V's 'Average Precision at IOU 0.3 is X' and V2X-Real
    # variants such as 'mAP at IoU 0.3: X'.  Use the final match per IoU.
    for threshold in ("0.3", "0.5", "0.7"):
        pattern = re.compile(r"(?:average precision|mAP).*?(?:iou|IOU)\s*" + re.escape(threshold) + r"[^0-9]*([0-9]+(?:\.[0-9]+)?)", re.I)
        found = pattern.findall(text)
        if not found:
            # Some V2X-Real writers use 'mAP@0.3 = X'.
            found = re.findall(r"mAP\s*@\s*" + re.escape(threshold) + r"\s*(?:is|[:=])\s*([0-9]+(?:\.[0-9]+)?)", text, re.I)
        if not found:
            raise ValueError("Could not parse AP@{} from {}".format(threshold, log_path))
        values["ap" + threshold.replace(".", "_")] = float(found[-1])
    values["metric_type"] = "AP" if dataset == "opv2v" else "mAP"
    return values


def zero_bandwidth(out_dir: Path, checkpoint: Path, dataset: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "baseline": "nofusion", "dataset": dataset, "frame_count": None,
        "tx_bytes_total": 0, "tx_MB_total": 0.0,
        "tx_bytes_per_frame": 0.0, "tx_MB_per_frame": 0.0,
        "definition": "No Fusion has max_cav=1 and exports no sender-to-ego message.",
        "checkpoint_source": str(checkpoint), "checkpoint_sha256": checksum(checkpoint),
        "pass": True,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def bandwidth_command(dataset: str, baseline: str, channel_mode: str,
                      runtime: Path, out_dir: Path) -> list[str]:
    test_dir = OPV2V_TEST if dataset == "opv2v" else V2XREAL_TEST
    return [sys.executable, "scripts/wire_bw_audit.py", "--name", "{}_{}".format(dataset, baseline), "--dataset", dataset,
            "--baseline", baseline, "--model_dir", str(runtime), "--channel_mode", channel_mode,
            "--test_dir", test_dir, "--out_dir", str(out_dir), "--max_frames", "-1", "--num_workers", "0", "--seed", str(SEED),
            # Some historical baseline YAMLs intentionally retain their
            # selection policy.  The experiment explicitly audits that setup.
            "--allow_policy"]


def read_bandwidth(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    frame_count = data.get("evaluated_frame_count") or 0
    total_bytes = data.get("total_tx_bytes")
    avg_bytes = (float(total_bytes) / float(frame_count)
                 if total_bytes is not None and frame_count else None)
    return {"bw_tx_MB_per_frame": data.get("avg_total_tx_MB_per_frame"),
            "bw_tx_bytes_per_frame": avg_bytes,
            "bw_summary_pass": data.get("pass")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--only", default=None, help="optional dataset/baseline filter")
    parser.add_argument("--mode", choices=("ideal", "markov", "both"), default="both",
                        help="channel mode to execute (default: both)")
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for dataset, baseline, source_name, markov_name in RUNS:
        key = "{}/{}".format(dataset, baseline)
        if args.only and args.only not in key:
            continue
        source_dir = CHECKPOINT_SOURCE_OVERRIDES.get(
            (dataset, baseline), OLD_LOGS / source_name
        )
        checkpoint = latest_checkpoint(source_dir)
        ideal_cfg = config_file(source_dir)
        checkpoint_package = CKPT_ROOT / dataset / baseline
        # This package is deliberately the only durable copy of the selected model.
        materialize_runtime(checkpoint_package, checkpoint, ideal_cfg, {"dataset": dataset, "baseline": baseline, "role": "ideal_checkpoint_package"})
        for mode in ("ideal", "markov"):
            if args.mode != "both" and mode != args.mode:
                continue
            run_dir = out_root / dataset / baseline / mode
            result_path = run_dir / "result.json"
            if args.skip_completed and result_path.exists():
                prior = json.loads(result_path.read_text(encoding="utf-8"))
                if prior.get("status") == "complete":
                    all_rows.append(prior)
                    continue
            runtime = run_dir / "model_runtime"
            runtime_cfg = ideal_cfg if mode == "ideal" or markov_name is None else MARKOV_CONFIG_OVERRIDES.get(
                (dataset, baseline), config_file(ROOT / "opencood/logs" / markov_name)
            )
            materialize_runtime(runtime, checkpoint, runtime_cfg, {"dataset": dataset, "baseline": baseline, "channel_mode": mode, "ideal_checkpoint_package": str(checkpoint_package)})
            started = time.time()
            row = {"dataset": dataset, "baseline": baseline, "channel_mode": mode,
                   "checkpoint": checkpoint.name, "checkpoint_sha256": checksum(checkpoint),
                   "runtime_config": str(runtime_cfg), "status": "failed"}
            try:
                ap_log = run_dir / "ap.log"
                run_command(ap_command(dataset, baseline, runtime), ap_log)
                row.update(parse_ap(ap_log, dataset))
                bw_dir = run_dir / "bandwidth"
                if baseline == "nofusion":
                    zero_bandwidth(bw_dir, checkpoint, dataset)
                else:
                    run_command(bandwidth_command(dataset, baseline, mode, runtime, bw_dir), run_dir / "bandwidth.log")
                row.update(read_bandwidth(bw_dir / "summary.json"))
                row["status"] = "complete"
            except Exception as exc:
                row["error"] = str(exc)
            row["elapsed_minutes"] = round((time.time() - started) / 60.0, 3)
            result_path.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
            all_rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    # A filtered invocation must not erase results written by earlier runs.
    all_rows = []
    for dataset, baseline, _source_name, _markov_name in RUNS:
        for mode in ("ideal", "markov"):
            result_path = out_root / dataset / baseline / mode / "result.json"
            if result_path.exists():
                all_rows.append(json.loads(result_path.read_text(encoding="utf-8")))
    fields = ["dataset", "baseline", "channel_mode", "metric_type", "ap0_3", "ap0_5", "ap0_7", "bw_tx_MB_per_frame", "bw_tx_bytes_per_frame", "checkpoint", "checkpoint_sha256", "runtime_config", "status", "elapsed_minutes", "error"]
    with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(all_rows)
    (out_root / "summary.json").write_text(json.dumps(all_rows, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
