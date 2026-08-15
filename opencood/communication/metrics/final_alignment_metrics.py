"""Final metrics for Markov time-varying Where2comm-ARCE experiments."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json, math
from collections import defaultdict, Counter


def _get(d: Dict[str, Any], *keys, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_frame_id(r: Dict[str, Any]) -> str:
    for k in ('frame_id', 'sample_idx', 'sample_id'):
        if k in r and r[k] is not None:
            return str(r[k])
    # fallback: composite
    return str(_get(r, 'meta', 'frame_id', default='unknown'))


def get_tx_bytes(r: Dict[str, Any]) -> float:
    return float(r.get('tx_bytes', r.get('transmitted_bytes', _get(r, 'size', 'tx_bytes', default=0.0))) or 0.0)


def get_rx_bytes(r: Dict[str, Any]) -> float:
    return float(r.get('rx_bytes', r.get('received_bytes', _get(r, 'size', 'rx_bytes', default=0.0))) or 0.0)


def get_budget_bytes(r: Dict[str, Any]) -> float:
    return float(r.get('B_link_bytes', r.get('budget_bytes', _get(r, 'budget', 'link_bytes', default=0.0))) or 0.0)


def get_total_budget_bytes(frame_records: List[Dict[str, Any]]) -> float:
    vals = [float(r.get('B_total_bytes', _get(r, 'budget', 'total_bytes', default=0.0)) or 0.0) for r in frame_records]
    return max(vals) if vals else 0.0


def get_effective_patch_ratio(r: Dict[str, Any]) -> float:
    return float(r.get('effective_patch_ratio', _get(r, 'patch', 'effective_patch_ratio', default=0.0)) or 0.0)


def get_q_eff(r: Dict[str, Any]) -> float:
    return float(r.get('q_eff', _get(r, 'recovery', 'q_eff', default=0.0)) or 0.0)


def is_selected_send(r: Dict[str, Any]) -> bool:
    send = r.get('send', _get(r, 'action', 'send', default=None))
    if send is None:
        no_send = r.get('no_send', _get(r, 'action', 'no_send', default=False))
        return not bool(no_send) and get_tx_bytes(r) > 0
    return bool(int(send)) and get_tx_bytes(r) > 0


def summarize_records(records: List[Dict[str, Any]], fps: float = 10.0, q_min: float = 0.3, eff_patch_min: float = 0.2) -> Dict[str, Any]:
    by_frame: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_frame[infer_frame_id(r)].append(r)
    n_frames = max(len(by_frame), 1)
    frame_summaries = []
    total_tx = 0.0
    total_rx = 0.0
    success = 0
    budget_viol = 0
    channel_counter = Counter()
    action_counter = Counter()
    patch_totals = defaultdict(float)

    for fid, rs in by_frame.items():
        tx = sum(get_tx_bytes(r) for r in rs)
        rx = sum(get_rx_bytes(r) for r in rs)
        total_tx += tx
        total_rx += rx
        total_budget = get_total_budget_bytes(rs)
        link_viol = any((get_budget_bytes(r) > 0 and get_tx_bytes(r) > get_budget_bytes(r) + 1e-6) for r in rs)
        total_viol = bool(total_budget > 0 and tx > total_budget + 1e-6)
        viol = link_viol or total_viol
        budget_viol += int(viol)
        selected = [r for r in rs if is_selected_send(r)]
        valid_link = any((get_q_eff(r) >= q_min or get_effective_patch_ratio(r) >= eff_patch_min) for r in selected)
        succ = (not viol) and len(selected) > 0 and valid_link
        success += int(succ)
        state = (rs[0].get('channel_state') or _get(rs[0], 'channel', 'state', default='unknown'))
        channel_counter[str(state)] += 1
        for r in rs:
            q = r.get('quant_mode', _get(r, 'action', 'quant', default='unknown'))
            rho = r.get('rho', r.get('redundancy_ratio', _get(r, 'action', 'rho', default='unknown')))
            cache = r.get('cache_enabled', _get(r, 'action', 'cache', default='unknown'))
            action_counter[f'q={q}|rho={rho}|cache={cache}|send={int(is_selected_send(r))}'] += 1
            for k in ('num_total_patches','num_valid_patches','num_selected_source_patches','num_received_patches','num_fec_recovered_patches','num_missing_by_budget','num_missing_by_loss','num_temporal_filled','num_spatial_filled','num_zero_filled'):
                patch_totals[k] += float(r.get(k, _get(r, 'patch', k, default=0)) or 0)
        frame_summaries.append({'frame_id': fid, 'channel_state': state, 'tx_bytes': tx, 'rx_bytes': rx, 'success': succ, 'budget_violation': viol})

    avg_tx = total_tx / n_frames
    return {
        'num_frames': n_frames,
        'avg_tx_bytes_per_frame': avg_tx,
        'avg_tx_kb_per_frame': avg_tx / 1024.0,
        'avg_mbps': avg_tx * 8.0 * fps / 1e6,
        'avg_rx_bytes_per_frame': total_rx / n_frames,
        'success_rate': success / n_frames,
        'budget_violation_rate': budget_viol / n_frames,
        'channel_frame_counts': dict(channel_counter),
        'action_distribution': dict(action_counter),
        'patch_totals': dict(patch_totals),
        'frames': frame_summaries,
    }


def summarize_by_channel(records: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
    groups = defaultdict(list)
    for r in records:
        state = r.get('channel_state') or _get(r, 'channel', 'state', default='unknown')
        groups[str(state)].append(r)
    return {s: summarize_records(rs, **kwargs) for s, rs in groups.items()}
