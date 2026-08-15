#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opencood.tools.arce_bw_breakdown_utils import (
    extract_action_id,
    extract_cache,
    extract_quant_mode,
    extract_rho,
    extract_rx_bytes,
    extract_tx_bytes,
    is_communication_record,
    is_no_send,
)
from opencood.utils import eval_utils

STATES = ("good", "medium", "bad")
EXPECTED_PROFILES = {
    "good": {"bandwidth_mbps": 27.0, "plr": 0.05, "delay_ms": 10.0},
    "medium": {"bandwidth_mbps": 5.0, "plr": 0.20, "delay_ms": 50.0},
    "bad": {"bandwidth_mbps": 1.0, "plr": 0.35, "delay_ms": 100.0},
}
STATE_SEVERITY = {"good": 0, "medium": 1, "bad": 2}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize final Markov+C2MAB online audit.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--runtime-jsonl", default=None)
    p.add_argument("--trace-jsonl", default=None)
    p.add_argument("--final-summary", default=None)
    p.add_argument("--warmup-frames", type=int, default=500)
    return p.parse_args()


def nested(d: Any, path: Sequence[str], default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def first(d: Dict[str, Any], paths: Sequence[Sequence[str]], default=None):
    for path in paths:
        value = nested(d, path, None)
        if value is not None:
            return value
    return default


def as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        value = float(value)
        return value if math.isfinite(value) else None
    except Exception:
        return None


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def stats(values: Iterable[Any]) -> Dict[str, Any]:
    vals = sorted(v for v in (as_float(x) for x in values) if v is not None)
    if not vals:
        return {"n": 0}

    def pct(q: float) -> float:
        i = int(round(q * (len(vals) - 1)))
        return float(vals[max(0, min(len(vals) - 1, i))])

    return {
        "n": len(vals),
        "min": float(vals[0]),
        "p10": pct(0.10),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "max": float(vals[-1]),
        "mean": float(sum(vals) / len(vals)),
    }


def ratio(num: float, den: float) -> Optional[float]:
    return float(num / den) if den > 0 else None


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def normalize_state(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    return text if text in STATES else "unknown"


def action_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    action = record.get("action") if isinstance(record.get("action"), dict) else {}
    pdf_action = record.get("pdf_action") if isinstance(record.get("pdf_action"), dict) else {}
    proposal = first(record, [("dc2mab", "proposal"), ("proposal",)], {})
    proposal_action = proposal.get("action", {}) if isinstance(proposal, dict) else {}

    action_id = extract_action_id(record)
    no_send = bool(is_no_send(record))
    quant = extract_quant_mode(record)
    rho_text = extract_rho(record)
    cache_text = extract_cache(record)

    requested_quant = str(first({"pdf": pdf_action, "proposal": proposal_action}, [
        ("pdf", "quant_mode"), ("proposal", "quant_mode")
    ], quant)).lower()
    requested_rho = as_float(first({"pdf": pdf_action, "proposal": proposal_action}, [
        ("pdf", "redundancy_ratio"), ("proposal", "redundancy_ratio")
    ], rho_text))
    requested_cache = as_int(first({"pdf": pdf_action, "proposal": proposal_action}, [
        ("pdf", "cache_enabled"), ("proposal", "cache_enabled")
    ], cache_text), 0)
    requested_send = as_int(first({"pdf": pdf_action, "proposal": proposal_action}, [
        ("pdf", "send"), ("proposal", "send")
    ], 0 if no_send else 1), 0)

    actual_rho = as_float(first(record, [
        ("action", "redundancy_ratio"), ("redundancy_ratio",), ("rho",)
    ], rho_text))
    actual_cache = as_int(first(record, [
        ("action", "cache_enabled"), ("cache_enabled",), ("cache",)
    ], cache_text), 0)
    actual_send = 0 if no_send else as_int(action.get("send", 1), 1)

    mismatch = False
    if not no_send:
        mismatch = (
            requested_send != actual_send
            or requested_quant != str(quant).lower()
            or abs(float(requested_rho or 0.0) - float(actual_rho or 0.0)) > 1e-9
            or requested_cache != actual_cache
        )

    return {
        "action_id": str(action_id),
        "no_send": no_send,
        "send": actual_send,
        "quant_mode": str(quant).lower(),
        "rho": float(actual_rho or 0.0),
        "cache": int(actual_cache),
        "fec_type": str(first(record, [
            ("action", "fec_type"), ("fec_encode", "fec_type"), ("pdf_action", "fec_type")
        ], "none")),
        "requested_send": requested_send,
        "requested_quant_mode": requested_quant,
        "requested_rho": float(requested_rho or 0.0),
        "requested_cache": requested_cache,
        "execution_mismatch": bool(mismatch),
    }


def extract_link_row(wrapper: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    record = wrapper.get("record", wrapper)
    if not isinstance(record, dict) or not is_communication_record(record):
        return None
    frame_index = as_int(wrapper.get("frame_index", record.get("frame_id", -1)), -1)
    action = action_from_record(record)
    state = normalize_state(first(record, [
        ("channel_state",),
        ("dc2mab", "proposal", "channel_state"),
        ("proposal", "channel_state"),
    ], "unknown"))
    profile = first(record, [("channel", "profile"), ("channel_profile",)], {})
    if not isinstance(profile, dict):
        profile = {}

    source = as_int(first(record, [
        ("packet", "num_source_packets"),
        ("bandwidth_selection", "num_source_packets"),
        ("size", "actual_num_source_packets"),
    ], 0))
    parity = as_int(first(record, [
        ("packet", "num_parity_packets"),
        ("bandwidth_selection", "num_parity_packets"),
        ("size", "actual_num_parity_packets"),
    ], 0))
    tx_source = as_int(first(record, [
        ("packet", "num_transmitted_source_packets"),
        ("bandwidth_selection", "num_tx_source_packets"),
        ("size", "actual_num_transmitted_source_packets"),
    ], 0))
    tx_parity = as_int(first(record, [
        ("packet", "num_transmitted_parity_packets"),
        ("bandwidth_selection", "num_tx_parity_packets"),
        ("size", "actual_num_transmitted_parity_packets"),
    ], 0))
    source_budget_drop = as_int(first(record, [
        ("packet", "num_source_dropped_by_budget"),
        ("bandwidth_selection", "num_source_dropped_by_budget"),
        ("size", "num_source_dropped_by_budget"),
    ], 0))
    parity_budget_drop = as_int(first(record, [
        ("packet", "num_parity_dropped_by_budget"),
        ("bandwidth_selection", "num_parity_dropped_by_budget"),
        ("size", "num_parity_dropped_by_budget"),
    ], 0))
    direct = as_int(first(record, [
        ("packet", "num_direct_received_source_packets"),
        ("size", "num_direct_received_source_packets"),
        ("partial_reconstruction", "num_direct_received_packets"),
    ], 0))
    fec_recovered = as_int(first(record, [
        ("packet", "num_fec_recovered_source_packets"),
        ("size", "num_fec_recovered_source_packets"),
        ("partial_reconstruction", "num_fec_recovered_packets"),
    ], 0))
    missing = as_int(first(record, [
        ("packet", "num_missing_source_packets"),
        ("size", "num_missing_source_packets"),
        ("partial_reconstruction", "num_still_missing"),
    ], 0))

    allocated_budget = as_float(first(record, [
        ("system_budget", "allocated_budget_bytes"),
        ("budget_consistency", "allocated_budget_bytes"),
        ("bandwidth_selection", "budget_bytes"),
        ("size", "bandwidth_budget_bytes"),
        ("link_budget_bytes",),
        ("system_budget", "per_link_budget_bytes"),
    ], 0.0)) or 0.0
    tx_bytes = float(extract_tx_bytes(record))
    rx_bytes = float(extract_rx_bytes(record))
    proposal = first(record, [("dc2mab", "proposal"), ("proposal",)], {})
    if not isinstance(proposal, dict):
        proposal = {}
    selection = first(record, [("dc2mab", "selection_score"),], {})
    if not isinstance(selection, dict):
        selection = {}

    row = {
        "frame_index": frame_index,
        "frame_id": str(record.get("frame_id", frame_index)),
        "ego_id": str(record.get("ego_id", record.get("ego_index", ""))),
        "sender_id": str(record.get("sender_id", record.get("agent_index", ""))),
        "channel_state": state,
        "bandwidth_mbps": as_float(profile.get("bandwidth_mbps")),
        "plr": as_float(profile.get("plr", profile.get("loss_rate"))),
        "delay_ms": as_float(profile.get("delay_ms")),
        **action,
        "allocated_budget_bytes": allocated_budget,
        "tx_bytes": tx_bytes,
        "rx_bytes": rx_bytes,
        "budget_utilization": ratio(tx_bytes, allocated_budget),
        "budget_exceeded": bool(tx_bytes > allocated_budget + 1e-6),
        "source_packets": source,
        "parity_packets": parity,
        "tx_source_packets": tx_source,
        "tx_parity_packets": tx_parity,
        "source_budget_drop": source_budget_drop,
        "parity_budget_drop": parity_budget_drop,
        "source_tx_ratio": ratio(tx_source, source),
        "parity_tx_ratio": ratio(tx_parity, parity),
        "direct_received_source_packets": direct,
        "fec_recovered_source_packets": fec_recovered,
        "missing_source_packets": missing,
        "final_source_recovery_ratio": ratio(direct + fec_recovered, source),
        "q_recv": as_float(first(record, [("quality", "q_recv"),], None)),
        "ucb": as_float(proposal.get("ucb")),
        "ucb_mean": as_float(proposal.get("mean")),
        "ucb_bonus": as_float(proposal.get("bonus")),
        "oracle_selection_score": as_float(selection.get("ratio")),
        "complementarity": as_float(record.get("complementarity", proposal.get("complementarity"))),
        "ego_confidence": as_float(record.get("ego_confidence", proposal.get("ego_confidence"))),
        "cav_confidence": as_float(record.get("cav_confidence", proposal.get("cav_confidence"))),
    }
    expected = EXPECTED_PROFILES.get(state)
    row["profile_match"] = bool(
        expected is not None
        and row["bandwidth_mbps"] is not None
        and row["plr"] is not None
        and row["delay_ms"] is not None
        and abs(row["bandwidth_mbps"] - expected["bandwidth_mbps"]) < 1e-6
        and abs(row["plr"] - expected["plr"]) < 1e-6
        and abs(row["delay_ms"] - expected["delay_ms"]) < 1e-6
    )
    return row


def merge_eval_stats(frames: Sequence[Dict[str, Any]]) -> Dict[float, Dict[str, Any]]:
    merged = {iou: {"tp": [], "fp": [], "gt": 0, "score": []} for iou in (0.3, 0.5, 0.7)}
    for frame in frames:
        raw = frame.get("eval_stat", {})
        for iou in (0.3, 0.5, 0.7):
            stat = raw.get(f"{iou:.1f}", {}) if isinstance(raw, dict) else {}
            merged[iou]["tp"].extend(stat.get("tp", []) or [])
            merged[iou]["fp"].extend(stat.get("fp", []) or [])
            merged[iou]["score"].extend(stat.get("score", []) or [])
            merged[iou]["gt"] += as_int(stat.get("gt", 0))
    return merged


def ap_from_frames(frames: Sequence[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    merged = merge_eval_stats(frames)
    out = {}
    for iou in (0.3, 0.5, 0.7):
        if merged[iou]["gt"] <= 0:
            out[f"ap_{int(iou*10):02d}"] = None
        else:
            ap, _, _ = eval_utils.calculate_ap(merged, iou, False)
            out[f"ap_{int(iou*10):02d}"] = float(ap)
    return out


def frame_state_labels(frame: Dict[str, Any]) -> Tuple[str, str]:
    actions = frame.get("actions", []) or []
    states = [normalize_state(x.get("channel_state")) for x in actions]
    states = [x for x in states if x in STATES]
    if not states:
        return "no_link", "no_link"
    unique = sorted(set(states), key=lambda x: STATE_SEVERITY[x])
    signature = unique[0] if len(unique) == 1 else "mixed_" + "_".join(unique)
    worst = max(states, key=lambda x: STATE_SEVERITY[x])
    return signature, worst


def entropy(items: Iterable[Any]) -> Optional[float]:
    counter = Counter(str(x) for x in items if x is not None)
    total = sum(counter.values())
    if total <= 0:
        return None
    return float(-sum((n / total) * math.log(n / total) for n in counter.values()))


def summarize_link_group(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "record_count": len(rows),
        "tx_MB": sum(r["tx_bytes"] for r in rows) / 1_000_000.0,
        "rx_MB": sum(r["rx_bytes"] for r in rows) / 1_000_000.0,
        "no_send_ratio": ratio(sum(bool(r["no_send"]) for r in rows), len(rows)),
        "budget_exceeded_ratio": ratio(sum(bool(r["budget_exceeded"]) for r in rows), len(rows)),
        "execution_mismatch_ratio": ratio(sum(bool(r["execution_mismatch"]) for r in rows), len(rows)),
        "profile_match_ratio": ratio(sum(bool(r["profile_match"]) for r in rows), len(rows)),
        "mean_budget_utilization": stats(r["budget_utilization"] for r in rows).get("mean"),
        "mean_source_tx_ratio": stats(r["source_tx_ratio"] for r in rows).get("mean"),
        "mean_parity_tx_ratio": stats(r["parity_tx_ratio"] for r in rows).get("mean"),
        "mean_final_source_recovery_ratio": stats(r["final_source_recovery_ratio"] for r in rows).get("mean"),
        "mean_fec_recovered_source_packets": stats(r["fec_recovered_source_packets"] for r in rows).get("mean"),
        "mean_missing_source_packets": stats(r["missing_source_packets"] for r in rows).get("mean"),
        "mean_reward_proxy_q_recv": stats(r["q_recv"] for r in rows).get("mean"),
        "action_entropy_nats": entropy(r["action_id"] for r in rows),
        "action_counter": dict(Counter(r["action_id"] for r in rows)),
        "quant_counter": dict(Counter(r["quant_mode"] for r in rows)),
        "rho_counter": dict(Counter(str(r["rho"]) for r in rows)),
        "cache_counter": dict(Counter(str(r["cache"]) for r in rows)),
    }


def transition_summary(rows: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, int]], List[Dict[str, Any]]]:
    by_link = defaultdict(list)
    for row in rows:
        if row["channel_state"] in STATES:
            by_link[(row["ego_id"], row["sender_id"])].append(row)
    counts = {s: {t: 0 for t in STATES} for s in STATES}
    run_lengths = defaultdict(list)
    for link_rows in by_link.values():
        link_rows = sorted(link_rows, key=lambda x: (x["frame_index"], x["frame_id"]))
        seq = [r["channel_state"] for r in link_rows]
        for a, b in zip(seq, seq[1:]):
            counts[a][b] += 1
        if seq:
            current = seq[0]
            length = 1
            for state in seq[1:]:
                if state == current:
                    length += 1
                else:
                    run_lengths[current].append(length)
                    current, length = state, 1
            run_lengths[current].append(length)
    csv_rows = []
    for s in STATES:
        den = sum(counts[s].values())
        for t in STATES:
            csv_rows.append({
                "from_state": s,
                "to_state": t,
                "count": counts[s][t],
                "probability": ratio(counts[s][t], den),
            })
    return {
        "counts": counts,
        "probabilities": {
            s: {t: ratio(counts[s][t], sum(counts[s].values())) for t in STATES}
            for s in STATES
        },
        "run_length": {s: stats(run_lengths[s]) for s in STATES},
    }, csv_rows


def phase_summary(frames: Sequence[Dict[str, Any]], rows: Sequence[Dict[str, Any]], start: int, stop: int) -> Dict[str, Any]:
    f = [x for x in frames if start <= as_int(x.get("frame_index"), -1) < stop]
    r = [x for x in rows if start <= x["frame_index"] < stop]
    rewards = [x.get("mean_reward") for x in f]
    actions = [a for x in f for a in (x.get("actions", []) or [])]
    return {
        "start_frame": start,
        "stop_frame_exclusive": stop,
        "frame_count": len(f),
        "link_record_count": len(r),
        "ap": ap_from_frames(f) if f else {},
        "mean_reward": stats(rewards),
        "action_entropy_nats": entropy(a.get("action_id") for a in actions),
        "ucb_mean": stats(a.get("ucb_mean") for a in actions),
        "ucb_bonus": stats(a.get("ucb_bonus") for a in actions),
        "quant_counter": dict(Counter(a.get("quant_mode") for a in actions)),
        "rho_counter": dict(Counter(str(a.get("rho")) for a in actions)),
        "state_counter": dict(Counter(a.get("channel_state") for a in actions)),
        "tx_MB": sum(x.get("tx_bytes", 0.0) for x in f) / 1_000_000.0,
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    runtime_path = Path(args.runtime_jsonl) if args.runtime_jsonl else out_dir / "runtime_records.jsonl"
    trace_path = Path(args.trace_jsonl) if args.trace_jsonl else out_dir / "online_trace.jsonl"
    final_path = Path(args.final_summary) if args.final_summary else out_dir / "final_summary.json"
    if not runtime_path.is_file():
        raise FileNotFoundError(runtime_path)
    if not trace_path.is_file():
        raise FileNotFoundError(trace_path)

    wrappers = read_jsonl(runtime_path)
    frames = read_jsonl(trace_path)
    link_rows = [x for x in (extract_link_row(w) for w in wrappers) if x is not None]
    if not link_rows:
        raise RuntimeError("No communication records found in runtime_records.jsonl")

    final = json.loads(final_path.read_text()) if final_path.is_file() else {}
    state_groups = {s: [r for r in link_rows if r["channel_state"] == s] for s in STATES}
    state_groups["unknown"] = [r for r in link_rows if r["channel_state"] not in STATES]

    by_state = {state: summarize_link_group(rows) for state, rows in state_groups.items()}
    overall = summarize_link_group(link_rows)
    state_counter = Counter(r["channel_state"] for r in link_rows)
    overall["state_counter"] = dict(state_counter)
    overall["state_ratio"] = {k: ratio(v, len(link_rows)) for k, v in state_counter.items()}

    transition, transition_rows = transition_summary(link_rows)

    frame_groups = defaultdict(list)
    worst_groups = defaultdict(list)
    for frame in frames:
        signature, worst = frame_state_labels(frame)
        frame_groups[signature].append(frame)
        worst_groups[worst].append(frame)

    frame_state_perception = []
    for label, group in sorted(frame_groups.items()):
        row = {"grouping": "state_signature", "state_group": label, "frame_count": len(group), **ap_from_frames(group)}
        row["mean_quality_0357"] = stats(x.get("quality_mean_0357") for x in group).get("mean")
        row["mean_reward"] = stats(x.get("mean_reward") for x in group).get("mean")
        frame_state_perception.append(row)
    for label, group in sorted(worst_groups.items()):
        row = {"grouping": "worst_link_state", "state_group": label, "frame_count": len(group), **ap_from_frames(group)}
        row["mean_quality_0357"] = stats(x.get("quality_mean_0357") for x in group).get("mean")
        row["mean_reward"] = stats(x.get("mean_reward") for x in group).get("mean")
        frame_state_perception.append(row)

    state_action_rows = []
    for state in STATES + ("unknown",):
        rows = state_groups[state]
        counter = Counter((r["action_id"], r["quant_mode"], r["rho"], r["cache"], r["no_send"]) for r in rows)
        for (aid, quant, rho, cache, no_send), count in counter.most_common():
            subset = [r for r in rows if (r["action_id"], r["quant_mode"], r["rho"], r["cache"], r["no_send"]) == (aid, quant, rho, cache, no_send)]
            state_action_rows.append({
                "channel_state": state,
                "action_id": aid,
                "quant_mode": quant,
                "rho": rho,
                "cache": cache,
                "no_send": int(no_send),
                "count": count,
                "state_action_ratio": ratio(count, len(rows)),
                "mean_tx_bytes": stats(r["tx_bytes"] for r in subset).get("mean"),
                "mean_budget_utilization": stats(r["budget_utilization"] for r in subset).get("mean"),
                "mean_final_source_recovery_ratio": stats(r["final_source_recovery_ratio"] for r in subset).get("mean"),
                "mean_fec_recovered_source_packets": stats(r["fec_recovered_source_packets"] for r in subset).get("mean"),
                "mean_ucb": stats(r["ucb"] for r in subset).get("mean"),
                "mean_ucb_bonus": stats(r["ucb_bonus"] for r in subset).get("mean"),
            })

    n_frames = len(frames)
    warmup = max(0, min(int(args.warmup_frames), n_frames))
    tail_size = max(1, min(500, n_frames // 4 if n_frames >= 4 else n_frames))
    phases = {
        "all": phase_summary(frames, link_rows, 0, n_frames),
        "warmup": phase_summary(frames, link_rows, 0, warmup) if warmup > 0 else None,
        "steady_state_candidate": phase_summary(frames, link_rows, warmup, n_frames) if warmup < n_frames else None,
        "tail": phase_summary(frames, link_rows, max(0, n_frames - tail_size), n_frames),
    }

    diagnostic_rows = sorted(
        link_rows,
        key=lambda r: (r["fec_recovered_source_packets"], r["source_budget_drop"], r["tx_bytes"]),
        reverse=True,
    )[:100]

    checks = {
        "frame_trace_present": len(frames) > 0,
        "runtime_link_records_present": len(link_rows) > 0,
        "known_state_ratio": ratio(sum(r["channel_state"] in STATES for r in link_rows), len(link_rows)),
        "profile_match_ratio": overall["profile_match_ratio"],
        "budget_exceeded_ratio": overall["budget_exceeded_ratio"],
        "action_execution_mismatch_ratio": overall["execution_mismatch_ratio"],
        "all_expected_states_observed": all(state_counter.get(s, 0) > 0 for s in STATES),
        "multiple_actions_observed": len(set(r["action_id"] for r in link_rows)) > 1,
        "ap_present": all(final.get(k) is not None for k in ("AP@0.3-Markov", "AP@0.5-Markov", "AP@0.7-Markov")),
    }
    checks["pass"] = bool(
        checks["frame_trace_present"]
        and checks["runtime_link_records_present"]
        and (checks["known_state_ratio"] or 0.0) == 1.0
        and (checks["profile_match_ratio"] or 0.0) == 1.0
        and (checks["budget_exceeded_ratio"] or 0.0) == 0.0
        and (checks["action_execution_mismatch_ratio"] or 0.0) == 0.0
        and checks["ap_present"]
    )

    summary = {
        "evaluation": final,
        "paths": {
            "runtime_records_jsonl": str(runtime_path),
            "online_trace_jsonl": str(trace_path),
            "final_summary_json": str(final_path),
        },
        "overall": overall,
        "by_channel_state": by_state,
        "markov_transition": transition,
        "frame_state_perception": frame_state_perception,
        "phases": phases,
        "checks": checks,
        "notes": {
            "statewise_ap_definition": "Frames are grouped either by the exact set of link states in the frame or by the worst link state. Per-link AP is not mathematically identifiable after multi-link fusion.",
            "online_protocol": "One continuous prequential trajectory; C2MAB policy and Markov state are not reset between frames.",
            "source_parity_scheduler": "Original project source-first ordering is preserved.",
        },
    }

    (out_dir / "final_markov_c2mab_audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    write_csv(out_dir / "state_action_summary.csv", state_action_rows, [
        "channel_state", "action_id", "quant_mode", "rho", "cache", "no_send", "count", "state_action_ratio",
        "mean_tx_bytes", "mean_budget_utilization", "mean_final_source_recovery_ratio",
        "mean_fec_recovered_source_packets", "mean_ucb", "mean_ucb_bonus",
    ])
    write_csv(out_dir / "frame_state_perception.csv", frame_state_perception, [
        "grouping", "state_group", "frame_count", "ap_03", "ap_05", "ap_07", "mean_quality_0357", "mean_reward",
    ])
    write_csv(out_dir / "markov_transition.csv", transition_rows, ["from_state", "to_state", "count", "probability"])
    write_csv(out_dir / "diagnostic_links_top100.csv", diagnostic_rows, [
        "frame_index", "frame_id", "ego_id", "sender_id", "channel_state", "action_id", "quant_mode", "rho", "cache", "no_send",
        "bandwidth_mbps", "plr", "delay_ms", "allocated_budget_bytes", "tx_bytes", "rx_bytes", "budget_utilization",
        "source_packets", "parity_packets", "tx_source_packets", "tx_parity_packets", "source_budget_drop", "parity_budget_drop",
        "direct_received_source_packets", "fec_recovered_source_packets", "missing_source_packets", "final_source_recovery_ratio",
        "q_recv", "ucb", "ucb_mean", "ucb_bonus", "oracle_selection_score", "complementarity", "ego_confidence", "cav_confidence",
    ])

    print(json.dumps({
        "pass": checks["pass"],
        "frame_count": len(frames),
        "link_record_count": len(link_rows),
        "state_counter": dict(state_counter),
        "action_count": len(set(r["action_id"] for r in link_rows)),
        "no_send_ratio": overall["no_send_ratio"],
        "profile_match_ratio": checks["profile_match_ratio"],
        "budget_exceeded_ratio": checks["budget_exceeded_ratio"],
        "execution_mismatch_ratio": checks["action_execution_mismatch_ratio"],
        "ap": {k: final.get(k) for k in ("AP@0.3-Markov", "AP@0.5-Markov", "AP@0.7-Markov")},
        "summary": str(out_dir / "final_markov_c2mab_audit_summary.json"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
