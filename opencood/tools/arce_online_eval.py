#!/usr/bin/env python
"""Single-pass prequential AP/BW evaluation for ARCE.

The same model instance, Markov trace, actions, and online policy updates are
used for both perception and communication metrics.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.methods.arce.runtime.communication_volume_summary import (
    summarize_bw_records,
)
from opencood.data_utils.datasets import build_dataset
from opencood.tools import inference_utils, train_utils
from opencood.tools.arce_bw_breakdown_utils import (
    extract_action_id,
    extract_cache,
    extract_quant_mode,
    extract_rho,
    extract_tx_bytes,
    is_communication_record,
    is_no_send,
    save_arce_bw_breakdown,
)
from opencood.tools.arce_eval_runtime import set_deterministic_seed
from opencood.tools.arce_reward_audit import save_reward_runtime_audit
from opencood.utils import eval_utils


IOU_THRESHOLDS = (0.3, 0.5, 0.7)


def _json_default(value: Any):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _empty_result_stat() -> Dict[float, Dict[str, Any]]:
    return {
        iou: {"tp": [], "fp": [], "gt": 0, "score": []}
        for iou in IOU_THRESHOLDS
    }


def _serializable_result_stat(frame_stat: Dict[float, Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    for iou in IOU_THRESHOLDS:
        stat = frame_stat[iou]
        out["{:.1f}".format(iou)] = {
            "tp": list(stat.get("tp", [])),
            "fp": list(stat.get("fp", [])),
            "gt": int(stat.get("gt", 0)),
            "score": list(stat.get("score", [])),
        }
    return out


def _merge_result_stats(stats: Iterable[Dict[float, Dict[str, Any]]]):
    out = _empty_result_stat()
    for stat in stats:
        for iou in IOU_THRESHOLDS:
            out[iou]["tp"].extend(stat[iou]["tp"])
            out[iou]["fp"].extend(stat[iou]["fp"])
            out[iou]["score"].extend(stat[iou]["score"])
            out[iou]["gt"] += int(stat[iou]["gt"])
    return out


def _ap_summary(result_stat, global_sort: bool) -> Dict[str, Optional[float]]:
    out = {}
    for iou in IOU_THRESHOLDS:
        key = "ap@{:.1f}".format(iou)
        stat = result_stat[iou]
        if int(stat["gt"]) <= 0:
            out[key] = None
            continue
        ap, _, _ = eval_utils.calculate_ap(
            copy.deepcopy(result_stat), iou, bool(global_sort)
        )
        out[key] = float(ap)
    return out


def _frame_quality(frame_stat) -> Dict[str, float]:
    out = {}
    values = []
    for iou in IOU_THRESHOLDS:
        tp = float(sum(frame_stat[iou]["tp"]))
        fp = float(sum(frame_stat[iou]["fp"]))
        fn = max(float(frame_stat[iou]["gt"]) - tp, 0.0)
        denom = tp + fp + fn
        quality = tp / denom if denom > 0.0 else 0.0
        out["quality@{:.1f}".format(iou)] = float(quality)
        values.append(float(quality))
    out["quality_mean_0357"] = float(sum(values) / len(values))
    return out


def _get_nested(value: Any, *keys: str) -> Any:
    cur = value
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _first(record: Dict[str, Any], paths, default=None):
    for path in paths:
        value = _get_nested(record, *path)
        if value is not None:
            return value
    return default


def _float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        value = float(value)
        return value if math.isfinite(value) else None
    except Exception:
        return None


def _stats(values: Iterable[Any]) -> Dict[str, Any]:
    vals = sorted(v for v in (_float(x) for x in values) if v is not None)
    if not vals:
        return {"n": 0}

    def pct(q):
        index = int(round(q * (len(vals) - 1)))
        return float(vals[max(0, min(len(vals) - 1, index))])

    return {
        "n": len(vals),
        "min": float(vals[0]),
        "p10": pct(0.10),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "max": float(vals[-1]),
        "mean": float(sum(vals) / len(vals)),
    }


def _normalized_entropy(
    actions: Iterable[str], action_space_size: Optional[int] = None
) -> Optional[float]:
    counts = Counter(str(x) for x in actions if x is not None)
    total = sum(counts.values())
    if total <= 0:
        return None
    denominator_size = max(len(counts), int(action_space_size or 0))
    if denominator_size <= 1:
        return 0.0
    entropy = -sum(
        (count / total) * math.log(count / total) for count in counts.values()
    )
    return float(entropy / math.log(denominator_size))


def _compact_comm_record(record: Dict[str, Any]) -> Dict[str, Any]:
    proposal = _first(record, [
        ("dc2mab", "proposal"),
        ("proposal",),
    ], {})
    action_id = extract_action_id(record)
    channel_state = _first(record, [
        ("channel_state",),
        ("dc2mab", "proposal", "channel_state"),
        ("proposal", "channel_state"),
    ], "unknown")
    mean = _first(record, [
        ("mean",), ("dc2mab", "proposal", "mean"), ("proposal", "mean")
    ])
    bonus = _first(record, [
        ("bonus",), ("dc2mab", "proposal", "bonus"), ("proposal", "bonus")
    ])
    ucb = _first(record, [
        ("ucb",), ("dc2mab", "proposal", "ucb"), ("proposal", "ucb")
    ])
    oracle_bonus = _first(record, [
        ("dc2mab", "selection_score", "exploration_bonus"),
    ])
    oracle_ratio = _first(record, [
        ("dc2mab", "selection_score", "ratio"),
    ])
    return {
        "frame_id": record.get("frame_id"),
        "ego_id": str(record.get("ego_id", "")),
        "sender_id": str(record.get("sender_id", record.get("agent_index", ""))),
        "action_id": str(action_id),
        "channel_state": str(channel_state),
        "quant_mode": extract_quant_mode(record),
        "rho": extract_rho(record),
        "cache": extract_cache(record),
        "no_send": bool(is_no_send(record)),
        "tx_bytes": float(extract_tx_bytes(record)),
        "ucb": _float(ucb),
        "ucb_mean": _float(mean),
        "ucb_bonus": _float(bonus),
        "oracle_exploration_bonus": _float(oracle_bonus),
        "oracle_selection_score": _float(oracle_ratio),
        "complementarity": _float(
            record.get("complementarity", proposal.get("complementarity"))
            if isinstance(proposal, dict) else record.get("complementarity")
        ),
    }


def _compact_reward_update(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for record in reversed(records):
        update = record.get("reward_update") if isinstance(record, dict) else None
        if not isinstance(update, dict):
            continue
        return {
            "delta_confidence": update.get("delta_confidence"),
            "mean_reward": update.get("mean_reward"),
            "num_updated": update.get("num_updated"),
            "reward_delta_source": update.get("reward_delta_source"),
            "delta_confidence_override": update.get("delta_confidence_override"),
            "reward_term_summary": update.get("reward_term_summary"),
            "link_rewards": update.get("link_rewards", []),
        }
    return None


def _get_comm(model):
    comm = getattr(model, "arce_comm", None)
    if comm is None and hasattr(model, "module"):
        comm = getattr(model.module, "arce_comm", None)
    if comm is None:
        raise AttributeError("Model does not expose arce_comm.")
    return comm


def _record_count(comm) -> int:
    records = getattr(comm, "records", None)
    return len(records) if isinstance(records, list) else len(comm.get_records())


def _records_since(comm, start: int) -> List[Dict[str, Any]]:
    records = getattr(comm, "records", None)
    if isinstance(records, list):
        return copy.deepcopy(records[start:])
    return comm.get_records()[start:]


def _run_inference(fusion_method, batch, model, dataset):
    if fusion_method == "intermediate":
        return inference_utils.inference_intermediate_fusion(batch, model, dataset)
    if fusion_method == "late":
        return inference_utils.inference_late_fusion(batch, model, dataset)
    if fusion_method == "early":
        return inference_utils.inference_early_fusion(batch, model, dataset)
    raise ValueError("Unsupported fusion_method: {}".format(fusion_method))


def _window_summary(
    frames, global_sort: bool, action_space_size: Optional[int] = None
) -> Dict[str, Any]:
    result_stat = _merge_result_stats(x["_result_stat"] for x in frames)
    actions = [a for x in frames for a in x["actions"]]
    tx_bytes = sum(float(x["tx_bytes"]) for x in frames)
    by_state = defaultdict(list)
    for action in actions:
        by_state[str(action.get("channel_state", "unknown"))].append(
            action.get("action_id")
        )
    rewards = [x.get("mean_reward") for x in frames]
    return {
        "start_frame": int(frames[0]["frame_index"]),
        "end_frame": int(frames[-1]["frame_index"]),
        "frame_count": int(len(frames)),
        **_ap_summary(result_stat, global_sort),
        "total_tx_MB": float(tx_bytes / 1_000_000.0),
        "bw_MB_per_frame": float(tx_bytes / len(frames) / 1_000_000.0),
        "reward": _stats(rewards),
        "action_entropy": _normalized_entropy(
            (a.get("action_id") for a in actions), action_space_size
        ),
        "action_entropy_by_state": {
            state: _normalized_entropy(ids, action_space_size)
            for state, ids in by_state.items()
        },
        "action_counter": dict(Counter(a.get("action_id") for a in actions)),
        "quant_counter": dict(Counter(a.get("quant_mode") for a in actions)),
        "channel_state_counter": dict(
            Counter(a.get("channel_state") for a in actions)
        ),
        "ucb_mean": _stats(a.get("ucb_mean") for a in actions),
        "ucb_bonus": _stats(a.get("ucb_bonus") for a in actions),
        "ucb": _stats(a.get("ucb") for a in actions),
        "oracle_exploration_bonus": _stats(
            a.get("oracle_exploration_bonus") for a in actions
        ),
        "oracle_selection_score": _stats(
            a.get("oracle_selection_score") for a in actions
        ),
        "ucb_bonus_abs_over_mean_abs": (
            float(
                sum(abs(a["ucb_bonus"]) for a in actions if a.get("ucb_bonus") is not None)
                / max(
                    sum(abs(a["ucb_mean"]) for a in actions if a.get("ucb_mean") is not None),
                    1e-12,
                )
            )
            if any(a.get("ucb_bonus") is not None for a in actions)
            else None
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-pass online ARCE AP/BW/reward evaluation."
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--method", default="ARCE-C2MAB")
    parser.add_argument("--scenario", default="Markov")
    parser.add_argument("--fusion_method", default="intermediate")
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--progress_interval", type=int, default=50)
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--window_stride", type=int, default=100)
    parser.add_argument("--warmup_frames", type=int, default=500)
    parser.add_argument("--global_sort_detections", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    set_deterministic_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hypes = yaml_utils.load_yaml(os.path.join(args.model_dir, "config.yaml"))
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=dataset.collate_batch_test,
        pin_memory=False,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_utils.create_model(hypes).to(device)
    _, model = train_utils.load_saved_model(args.model_dir, model)
    if hasattr(model, "update_epoch"):
        model.update_epoch(999)
    model.eval()
    comm = _get_comm(model)
    action_space_size = len(getattr(comm, "action_ids", []) or []) or None

    max_frames = None if int(args.max_frames) < 0 else int(args.max_frames)
    result_stat = _empty_result_stat()
    frame_rows = []
    all_runtime_records = []
    trace_path = out_dir / "online_trace.jsonl"
    runtime_trace_path = out_dir / "runtime_records.jsonl"
    started_at = time.perf_counter()
    last_progress_at = started_at
    last_progress_count = 0

    def report_progress(count, final=False):
        nonlocal last_progress_at, last_progress_count
        now = time.perf_counter()
        elapsed = max(now - started_at, 1e-9)
        interval_elapsed = max(now - last_progress_at, 1e-9)
        interval_count = max(count - last_progress_count, 0)
        print(
            (
                "{} progress frames={} elapsed_s={:.1f} "
                "fps={:.3f} interval_fps={:.3f}{}"
            ).format(
                args.method,
                count,
                elapsed,
                count / elapsed,
                interval_count / interval_elapsed,
                " final" if final else "",
            ),
            flush=True,
        )
        last_progress_at = now
        last_progress_count = count

    with trace_path.open("w", encoding="utf-8") as trace_file, runtime_trace_path.open(
        "w", encoding="utf-8"
    ) as runtime_trace_file:
        with torch.no_grad():
            for frame_index, batch in enumerate(loader):
                if max_frames is not None and frame_index >= max_frames:
                    break

                batch = train_utils.to_device(batch, device)
                record_start = _record_count(comm)
                pred_boxes, pred_scores, gt_boxes = _run_inference(
                    args.fusion_method, batch, model, dataset
                )
                new_records = _records_since(comm, record_start)
                all_runtime_records.extend(new_records)
                for runtime_record in new_records:
                    runtime_trace_file.write(
                        json.dumps(
                            {
                                "frame_index": int(frame_index),
                                "record": runtime_record,
                            },
                            ensure_ascii=False,
                            default=_json_default,
                        ) + "\n"
                    )

                frame_stat = _empty_result_stat()
                for iou in IOU_THRESHOLDS:
                    eval_utils.caluclate_tp_fp(
                        pred_boxes, pred_scores, gt_boxes, frame_stat, iou
                    )
                    eval_utils.caluclate_tp_fp(
                        pred_boxes, pred_scores, gt_boxes, result_stat, iou
                    )

                actions = [
                    _compact_comm_record(record)
                    for record in new_records
                    if is_communication_record(record)
                ]
                reward_update = _compact_reward_update(new_records) or {}
                tx_bytes = sum(float(x["tx_bytes"]) for x in actions)
                frame_row = {
                    "frame_index": int(frame_index),
                    "frame_id": actions[0].get("frame_id") if actions else frame_index,
                    "tx_bytes": float(tx_bytes),
                    "bw_MB": float(tx_bytes / 1_000_000.0),
                    "num_pred_boxes": int(0 if pred_boxes is None else len(pred_boxes)),
                    "num_gt_boxes": int(0 if gt_boxes is None else len(gt_boxes)),
                    **_frame_quality(frame_stat),
                    "mean_reward": reward_update.get("mean_reward"),
                    "delta_confidence": reward_update.get("delta_confidence"),
                    "reward_delta_source": reward_update.get("reward_delta_source"),
                    "actions": actions,
                    "reward_update": reward_update,
                    "_result_stat": frame_stat,
                }
                frame_rows.append(frame_row)

                serializable = {
                    k: v for k, v in frame_row.items() if k != "_result_stat"
                }
                serializable["eval_stat"] = _serializable_result_stat(frame_stat)
                trace_file.write(
                    json.dumps(
                        serializable,
                        ensure_ascii=False,
                        default=_json_default,
                    ) + "\n"
                )

                count = frame_index + 1
                if args.progress_interval > 0 and count % args.progress_interval == 0:
                    report_progress(count)

    frame_count = len(frame_rows)
    if frame_count <= 0:
        raise RuntimeError("No frames were evaluated.")
    if frame_count != last_progress_count:
        report_progress(frame_count, final=True)

    # Exact dataset AP from this same online trajectory.
    eval_utils.eval_final_results(
        copy.deepcopy(result_stat), str(out_dir), bool(args.global_sort_detections)
    )
    ap = _ap_summary(result_stat, bool(args.global_sort_detections))
    bw = summarize_bw_records(
        all_runtime_records,
        method=args.method,
        scenario=args.scenario,
        num_frames=frame_count,
    )
    breakdown = save_arce_bw_breakdown(all_runtime_records, out_dir)
    reward_audit = save_reward_runtime_audit(
        all_runtime_records, out_dir, frame_count
    )
    bw_output = dict(bw)
    bw_output.update({
        "bw_breakdown_json": str(out_dir / "bw_breakdown.json"),
        "reward_runtime_audit_json": reward_audit["reward_runtime_audit_json"],
        "reward_update_count": reward_audit.get("reward_update_count"),
        "avg_tokens_per_token_record": breakdown.get("avg_tokens_per_token_record"),
        "avg_tx_bytes_per_token": breakdown.get("avg_tx_bytes_per_token"),
        "bad_legacy_action_ids": breakdown.get("bad_legacy_action_ids", []),
    })
    with (out_dir / "bw.json").open("w", encoding="utf-8") as f:
        json.dump(bw_output, f, indent=2, ensure_ascii=False, default=_json_default)

    windows = []
    size = max(1, int(args.window_size))
    stride = max(1, int(args.window_stride))
    for start in range(0, frame_count, stride):
        stop = min(start + size, frame_count)
        windows.append(
            _window_summary(
                frame_rows[start:stop],
                bool(args.global_sort_detections),
                action_space_size,
            )
        )

    warmup = max(0, min(int(args.warmup_frames), frame_count))
    phase_summary = {
        "all": _window_summary(
            frame_rows, bool(args.global_sort_detections), action_space_size
        ),
        "warmup": (
            _window_summary(
                frame_rows[:warmup],
                bool(args.global_sort_detections),
                action_space_size,
            )
            if warmup > 0 else None
        ),
        "steady_state_candidate": (
            _window_summary(
                frame_rows[warmup:],
                bool(args.global_sort_detections),
                action_space_size,
            )
            if warmup < frame_count else None
        ),
        "warmup_frames": int(warmup),
        "warmup_is_provisional": True,
    }

    rolling_payload = {
        "window_size": size,
        "window_stride": stride,
        "windows": windows,
        "phase_summary": phase_summary,
    }
    with (out_dir / "rolling_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            rolling_payload, f, indent=2, ensure_ascii=False, default=_json_default
        )

    final = {
        "Method": args.method,
        "Scenario": args.scenario,
        "evaluation_protocol": "online_prequential_single_pass",
        "AP@0.3-Markov": ap["ap@0.3"],
        "AP@0.5-Markov": ap["ap@0.5"],
        "AP@0.7-Markov": ap["ap@0.7"],
        **bw,
        "BW-Markov": bw.get("BW"),
        "online_trace_jsonl": str(trace_path),
        "runtime_records_jsonl": str(runtime_trace_path),
        "seed": int(args.seed),
        "rolling_metrics_json": str(out_dir / "rolling_metrics.json"),
        "bw_breakdown_json": str(out_dir / "bw_breakdown.json"),
        "reward_runtime_audit_json": reward_audit[
            "reward_runtime_audit_json"
        ],
        "reward_update_count": reward_audit.get("reward_update_count"),
        "avg_tokens_per_token_record": breakdown.get(
            "avg_tokens_per_token_record"
        ),
        "avg_tx_bytes_per_token": breakdown.get("avg_tx_bytes_per_token"),
        "bad_legacy_action_ids": breakdown.get("bad_legacy_action_ids", []),
    }
    with (out_dir / "final_summary.json").open("w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False, default=_json_default)

    with (out_dir / "final_table.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "Method", "AP@0.3-Markov", "AP@0.5-Markov", "AP@0.7-Markov",
            "BW-Markov", "total_tx_MB", "frame_count", "record_count",
            "transmitted_link_count", "no_send_count", "int4_count",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: final.get(key) for key in fields})

    with (out_dir / "bw.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "method", "scenario", "frame_count", "record_count",
            "transmitted_link_count", "no_send_count", "BW",
            "bw_MB_per_frame", "total_tx_MB", "int4_count",
            "packed_int4_count", "all_int4_packed",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: bw_output.get(key) for key in fields})

    print(json.dumps(final, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
