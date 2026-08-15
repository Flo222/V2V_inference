#!/usr/bin/env python
"""Matched-state seven-action counterfactual audit for online ARCE.

For each sampled frame, every action starts from an identical copy of the
pre-frame Markov, cache, and bandit state. Counterfactual trials do not update
the policy. After the trials, the untouched online communicator executes the
frame once normally so the real stream can continue.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import time
import warnings
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import inference_utils, train_utils
from opencood.tools.arce_bw_breakdown_utils import is_communication_record
from opencood.tools.arce_online_eval import (
    IOU_THRESHOLDS,
    _compact_comm_record,
    _empty_result_stat,
    _float,
    _frame_quality,
    _get_comm,
    _records_since,
    _stats,
)
from opencood.utils import eval_utils
from opencood.methods.arce.policies.ap_proxy_features import (
    HEAD_AP_PROXY_FEATURES,
    PAIRED_SPATIAL_AP_PROXY_FEATURES,
    paired_head_ap_proxy_features,
)
from opencood.methods.arce.policies.decoded_box_proxy_features import (
    PAIRED_DECODED_MATCH_FEATURES,
    RICH_DECODED_BOX_FEATURES,
    decoded_box_features,
    delta_decoded_feature_names,
    no_send_decoded_feature_names,
    paired_decoded_box_features,
)


warnings.filterwarnings(
    "once", message=r"nn\.functional\.sigmoid is deprecated.*"
)
warnings.filterwarnings(
    "once", message=r"invalid value encountered in intersection.*"
)


RECEIVER_TRANSPORT_FEATURES = (
    "rx_q_recv_unit",
    "rx_q_cache_unit",
    "rx_q_eff_unit",
    "rx_q_recv_packet",
    "rx_q_cache_packet",
    "rx_q_eff_packet",
    "rx_num_source_packets",
    "rx_num_transmitted_source_packets",
    "rx_num_source_dropped_by_budget",
    "rx_num_received_packets",
    "rx_num_direct_received_source_packets",
    "rx_num_fec_recovered_source_packets",
    "rx_num_missing_source_packets",
    "rx_num_temporal_filled_packets",
    "rx_num_zero_filled_packets",
    "rx_num_total_units",
    "rx_num_current_recovered_units",
    "rx_num_temporal_filled_units",
    "rx_num_effective_recovered_units",
    "rx_cache_hit",
    "rx_tx_source_ratio",
    "rx_direct_receive_ratio",
    "rx_effective_unit_ratio",
)


RUNTIME_CONTEXT_NAMES = (
    "B_norm",
    "p_loss",
    "d_norm",
    "ego_confidence",
    "cache_quality",
    "complementarity",
    "cav_confidence",
)

RUNTIME_CONTEXT_CSV_FIELDS = (
    "decision_context_available",
    "decision_context_source",
    "decision_context_B_norm",
    "decision_context_p_loss",
    "decision_context_d_norm",
    "decision_context_ego_confidence",
    "decision_context_cache_quality",
    "decision_context_complementarity",
    "decision_context_cav_confidence",
    "update_context_available",
    "update_context_B_norm",
    "update_context_p_loss",
    "update_context_d_norm",
    "update_context_ego_confidence",
    "update_context_cache_quality",
    "update_context_complementarity",
    "update_context_cav_confidence",
    "decision_update_context_max_abs_diff",
)


def _context_from_vector(vector, source):
    if not isinstance(vector, (list, tuple)) or len(vector) != 7:
        return {
            "available": False,
            "source": str(source),
            "vector": None,
        }
    values = [_finite_number(value, float("nan")) for value in vector]
    if not all(math.isfinite(value) for value in values):
        return {
            "available": False,
            "source": str(source),
            "vector": None,
        }
    return {
        "available": True,
        "source": str(source),
        "vector": list(values),
        **dict(zip(RUNTIME_CONTEXT_NAMES, values)),
    }


def _decision_context_from_record(record):
    """Extract the exact context attached to a scored send proposal."""
    record = record if isinstance(record, dict) else {}
    dc2mab = record.get("dc2mab")
    dc2mab = dc2mab if isinstance(dc2mab, dict) else {}
    proposal = dc2mab.get("proposal")
    proposal = proposal if isinstance(proposal, dict) else {}
    context = proposal.get("context")
    context = context if isinstance(context, dict) else {}

    vector = context.get("vector")
    if not isinstance(vector, (list, tuple)):
        if all(context.get(name) is not None for name in RUNTIME_CONTEXT_NAMES):
            vector = [context[name] for name in RUNTIME_CONTEXT_NAMES]

    return _context_from_vector(vector, "dc2mab.proposal.context")


def _update_context_from_record(record, decision_context):
    """Extract the context used for the reward update when it is recorded."""
    record = record if isinstance(record, dict) else {}
    vector = record.get("context_vector")
    update = _context_from_vector(vector, "record.context_vector")
    if update.get("available"):
        return update
    if isinstance(decision_context, dict) and decision_context.get("available"):
        copied = copy.deepcopy(decision_context)
        copied["source"] = "dc2mab.proposal.context"
        return copied
    return update


def _flatten_runtime_contexts(row):
    decision = row.get("decision_context")
    decision = decision if isinstance(decision, dict) else {}
    update = row.get("reward_update_context")
    update = update if isinstance(update, dict) else {}

    row["decision_context_available"] = bool(decision.get("available", False))
    row["decision_context_source"] = str(decision.get("source", "missing"))
    row["update_context_available"] = bool(update.get("available", False))

    for name in RUNTIME_CONTEXT_NAMES:
        row["decision_context_" + name] = decision.get(name)
        row["update_context_" + name] = update.get(name)

    if decision.get("available") and update.get("available"):
        row["decision_update_context_max_abs_diff"] = max(
            abs(float(a) - float(b))
            for a, b in zip(decision["vector"], update["vector"])
        )
    else:
        row["decision_update_context_max_abs_diff"] = None


def _attach_group_decision_context(action_rows):
    """Attach one action-independent decision context to all seven trials."""
    candidates = [
        row.get("decision_context")
        for row in action_rows
        if isinstance(row, dict)
        and not bool(row.get("no_send", False))
        and isinstance(row.get("decision_context"), dict)
        and bool(row["decision_context"].get("available", False))
    ]
    if not candidates:
        raise RuntimeError(
            "No scored send proposal context was recorded for this seven-action group."
        )

    canonical = candidates[0]
    for other in candidates[1:]:
        max_diff = max(
            abs(float(a) - float(b))
            for a, b in zip(canonical["vector"], other["vector"])
        )
        if max_diff > 1e-9:
            raise RuntimeError(
                "Counterfactual send trials do not share one decision context: "
                "max_abs_diff={}".format(max_diff)
            )

    for row in action_rows:
        if not isinstance(row, dict) or row.get("error") is not None:
            continue
        row["decision_context"] = copy.deepcopy(canonical)
        _flatten_runtime_contexts(row)


def _finite_number(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _first_number(candidates, default=0.0):
    for mapping, key in candidates:
        if isinstance(mapping, dict) and mapping.get(key) is not None:
            return _finite_number(mapping.get(key), default)
    return float(default)


def _receiver_transport_features(record):
    """Return post-action receiver measurements; never use as bandit context."""
    record = record if isinstance(record, dict) else {}
    quality = record.get("quality")
    quality = quality if isinstance(quality, dict) else {}
    packet = record.get("packet")
    packet = packet if isinstance(packet, dict) else {}
    partial = record.get("partial_reconstruction")
    partial = partial if isinstance(partial, dict) else {}
    temporal = partial.get("temporal_cache")
    temporal = temporal if isinstance(temporal, dict) else {}

    def pick(*candidates):
        return _first_number(candidates, 0.0)

    source_packets = pick(
        (packet, "num_source_packets"),
        (quality, "num_source_packets"),
    )
    transmitted_source = pick(
        (packet, "num_transmitted_source_packets"),
    )
    direct_received = pick(
        (packet, "num_direct_received_source_packets"),
        (partial, "num_direct_received_packets"),
    )
    total_units = pick(
        (temporal, "num_total_units"),
    )
    effective_units = pick(
        (temporal, "num_effective_recovered_units"),
        (partial, "num_effective_recovered_units"),
    )

    result = {
        "rx_q_recv_unit": pick(
            (quality, "q_recv_unit"),
            (temporal, "q_recv_unit"),
        ),
        "rx_q_cache_unit": pick(
            (quality, "q_cache_unit"),
            (temporal, "q_cache_unit"),
        ),
        "rx_q_eff_unit": pick(
            (quality, "q_eff_unit"),
            (temporal, "q_eff_unit"),
        ),
        "rx_q_recv_packet": pick(
            (quality, "q_recv_packet"),
            (temporal, "q_recv_packet"),
        ),
        "rx_q_cache_packet": pick(
            (quality, "q_cache_packet"),
            (temporal, "q_cache_packet"),
        ),
        "rx_q_eff_packet": pick(
            (quality, "q_eff_packet"),
            (temporal, "q_eff_packet"),
        ),
        "rx_num_source_packets": source_packets,
        "rx_num_transmitted_source_packets": transmitted_source,
        "rx_num_source_dropped_by_budget": pick(
            (packet, "num_source_dropped_by_budget"),
        ),
        "rx_num_received_packets": pick(
            (packet, "num_received_packets"),
        ),
        "rx_num_direct_received_source_packets": direct_received,
        "rx_num_fec_recovered_source_packets": pick(
            (packet, "num_fec_recovered_source_packets"),
            (partial, "num_fec_recovered_packets"),
        ),
        "rx_num_missing_source_packets": pick(
            (packet, "num_missing_source_packets"),
            (quality, "num_still_missing"),
            (partial, "num_still_missing"),
        ),
        "rx_num_temporal_filled_packets": pick(
            (quality, "num_temporal_filled_packets"),
            (temporal, "num_temporal_filled_packets"),
            (partial, "num_temporal_filled_packets"),
        ),
        "rx_num_zero_filled_packets": pick(
            (partial, "num_zero_filled_packets"),
        ),
        "rx_num_total_units": total_units,
        "rx_num_current_recovered_units": pick(
            (temporal, "num_current_recovered_units"),
            (partial, "num_current_recovered_units"),
        ),
        "rx_num_temporal_filled_units": pick(
            (temporal, "num_temporal_filled_units"),
            (partial, "num_temporal_filled_units"),
        ),
        "rx_num_effective_recovered_units": effective_units,
        "rx_cache_hit": 1.0 if bool(temporal.get("cache_hit", False)) else 0.0,
        "rx_tx_source_ratio": (
            transmitted_source / source_packets if source_packets > 0.0 else 0.0
        ),
        "rx_direct_receive_ratio": (
            direct_received / source_packets if source_packets > 0.0 else 0.0
        ),
        "rx_effective_unit_ratio": (
            effective_units / total_units if total_units > 0.0 else 0.0
        ),
    }
    return {name: _finite_number(result.get(name), 0.0)
            for name in RECEIVER_TRANSPORT_FEATURES}


def _core_model(model):
    return model.module if hasattr(model, "module") else model


def _bind_comm(model, comm) -> None:
    core = _core_model(model)
    core.arce_comm = comm
    fusion = getattr(core, "fusion_net", None)
    if fusion is None or not hasattr(fusion, "arce_comm"):
        raise AttributeError("Model fusion_net does not expose arce_comm.")
    fusion.arce_comm = comm


def _record_len(batch: Dict[str, Any]) -> int:
    value = batch["ego"]["record_len"]
    if torch.is_tensor(value):
        values = value.detach().cpu().view(-1).tolist()
        return int(values[0]) if values else 0
    if isinstance(value, (list, tuple)):
        return int(value[0]) if value else 0
    return int(value)


def _sequence_position(dataset: Any, frame_index: int) -> Tuple[int, int]:
    """Map a global dataset index to the OPV2V scenario and local frame."""
    boundaries = getattr(dataset, "len_record", None)
    if not isinstance(boundaries, (list, tuple)) or not boundaries:
        return 0, int(frame_index)

    previous = 0
    for sequence_id, boundary in enumerate(boundaries):
        boundary = int(boundary)
        if int(frame_index) < boundary:
            return int(sequence_id), int(frame_index) - int(previous)
        previous = boundary
    raise IndexError(
        "frame_index {} exceeds dataset len_record {}".format(
            frame_index,
            list(boundaries),
        )
    )


def _run_model(batch, model, dataset):
    output = model(batch["ego"])
    output_dict = OrderedDict([("ego", output)])
    pred_boxes, pred_scores, gt_boxes = inference_utils._post_process_compatible(
        batch, output_dict, dataset
    )
    return output, pred_boxes, pred_scores, gt_boxes


def _quality(pred_boxes, pred_scores, gt_boxes) -> Dict[str, float]:
    stat = _empty_result_stat()
    for iou in IOU_THRESHOLDS:
        eval_utils.caluclate_tp_fp(
            pred_boxes, pred_scores, gt_boxes, stat, iou
        )
    return _frame_quality(stat)


def _quality_from_heads(dataset, batch, psm, rm) -> Dict[str, float]:
    output_dict = OrderedDict([
        ("ego", {"psm": psm, "rm": rm}),
    ])
    pred_boxes, pred_scores, gt_boxes = inference_utils._post_process_compatible(
        batch,
        output_dict,
        dataset,
    )
    return _quality(pred_boxes, pred_scores, gt_boxes)


def _tensor_summary(value: Any) -> Dict[str, Any]:
    if not torch.is_tensor(value):
        return {"available": False}
    value = value.detach().float()
    return {
        "available": True,
        "shape": list(value.shape),
        "min": float(value.min().cpu()),
        "max": float(value.max().cpu()),
        "mean": float(value.mean().cpu()),
        "std": float(value.std(unbiased=False).cpu()),
        "rms": float(torch.sqrt(torch.mean(value * value)).cpu()),
    }


def _tensor_diff(value: Any, reference: Any) -> Dict[str, Any]:
    if not torch.is_tensor(value) or not torch.is_tensor(reference):
        return {"available": False}
    if tuple(value.shape) != tuple(reference.shape):
        return {
            "available": False,
            "shape_mismatch": [list(value.shape), list(reference.shape)],
        }
    diff = (value.detach().float() - reference.detach().float()).abs()
    return {
        "available": True,
        "max_abs": float(diff.max().cpu()),
        "mean_abs": float(diff.mean().cpu()),
        "rms": float(torch.sqrt(torch.mean(diff * diff)).cpu()),
        "nz_ratio": float((diff > 1e-6).float().mean().cpu()),
    }


def _feature_delta(output: Dict[str, Any]) -> Dict[str, Any]:
    arce = ((output.get("comm_info") or {}).get("arce") or {})
    if isinstance(arce, dict):
        value = arce.get("arce_feature_delta")
        if isinstance(value, dict):
            return dict(value)
    return {}


def _sender_feature_stats(
    feature_delta: Dict[str, Any], sender_index: int
) -> Dict[str, Any]:
    rows = feature_delta.get("per_agent", [])
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if (
            isinstance(row, dict)
            and int(row.get("agent_index", -1)) == int(sender_index)
        ):
            return dict(row)
    return {}


def _reward_update(output: Dict[str, Any]) -> Dict[str, Any]:
    comm_info = output.get("comm_info") or {}
    value = comm_info.get("arce_reward_update")
    return dict(value) if isinstance(value, dict) else {}


def _sign(value: Any, eps: float = 1e-9) -> int:
    value = _float(value)
    if value is None or abs(value) <= eps:
        return 0
    return 1 if value > 0.0 else -1


def _pearson(xs: Iterable[Any], ys: Iterable[Any]) -> Optional[float]:
    pairs = [
        (float(x), float(y)) for x, y in zip(xs, ys)
        if _float(x) is not None and _float(y) is not None
    ]
    if len(pairs) < 2:
        return None
    x_mean = sum(x for x, _ in pairs) / len(pairs)
    y_mean = sum(y for _, y in pairs) / len(pairs)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    x_var = sum((x - x_mean) ** 2 for x, _ in pairs)
    y_var = sum((y - y_mean) ** 2 for _, y in pairs)
    denominator = math.sqrt(x_var * y_var)
    return float(numerator / denominator) if denominator > 1e-12 else None


def _average_ranks(values: List[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = 0.5 * (start + stop - 1) + 1.0
        for pos in range(start, stop):
            ranks[order[pos]] = rank
        start = stop
    return ranks


def _spearman(xs: Iterable[Any], ys: Iterable[Any]) -> Optional[float]:
    pairs = [
        (float(x), float(y)) for x, y in zip(xs, ys)
        if _float(x) is not None and _float(y) is not None
    ]
    if len(pairs) < 2:
        return None
    return _pearson(
        _average_ranks([x for x, _ in pairs]),
        _average_ranks([y for _, y in pairs]),
    )


def _frame_ranking(
    rows: List[Dict[str, Any]],
    tie_tolerance: float = 0.01,
) -> Dict[str, Any]:
    valid = [
        row for row in rows
        if _float(row.get("proxy_delta_quality")) is not None
        and _float(row.get("true_quality_mean_0357")) is not None
    ]
    comparable = 0
    correct = 0
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            true_diff = (
                float(valid[i]["true_quality_mean_0357"])
                - float(valid[j]["true_quality_mean_0357"])
            )
            proxy_diff = (
                float(valid[i]["proxy_delta_quality"])
                - float(valid[j]["proxy_delta_quality"])
            )
            if abs(true_diff) <= float(tie_tolerance):
                continue
            comparable += 1
            correct += int(true_diff * proxy_diff > 0.0)

    true_values = [float(row["true_quality_mean_0357"]) for row in valid]
    proxy_values = [float(row["proxy_delta_quality"]) for row in valid]
    true_span = max(true_values) - min(true_values) if true_values else None
    proxy_span = max(proxy_values) - min(proxy_values) if proxy_values else None
    true_top = max(valid, key=lambda row: row["true_quality_mean_0357"]) if valid else None
    proxy_top = max(valid, key=lambda row: row["proxy_delta_quality"]) if valid else None
    top1_is_comparable = bool(
        true_span is not None
        and proxy_span is not None
        and true_span > 1e-9
        and proxy_span > 1e-9
    )
    return {
        "n": len(valid),
        "true_quality_span": true_span,
        "proxy_delta_span": proxy_span,
        "spearman": _spearman(
            [row["proxy_delta_quality"] for row in valid],
            [row["true_quality_mean_0357"] for row in valid],
        ),
        "pairwise_comparable": comparable,
        "pairwise_correct": correct,
        "tie_tolerance": float(tie_tolerance),
        "pairwise_accuracy": (
            float(correct / comparable) if comparable > 0 else None
        ),
        "true_top_action": true_top.get("action_id") if true_top else None,
        "proxy_top_action": proxy_top.get("action_id") if proxy_top else None,
        "top1_match": (
            bool(true_top["action_id"] == proxy_top["action_id"])
            if true_top and proxy_top and top1_is_comparable else None
        ),
        "top_set_match": (
            bool(
                float(proxy_top["true_quality_mean_0357"])
                >= float(true_top["true_quality_mean_0357"])
                - float(tie_tolerance)
            )
            if true_top and proxy_top and top1_is_comparable else None
        ),
        "selected_action_regret": (
            max(
                0.0,
                float(true_top["true_quality_mean_0357"])
                - float(proxy_top["true_quality_mean_0357"]),
            )
            if true_top and proxy_top and top1_is_comparable else None
        ),
    }


def _majority_sign_accuracy(
    rows: List[Dict[str, Any]],
    value_name: str,
) -> Optional[float]:
    signs = [
        _sign(row.get(value_name))
        for row in rows
        if _sign(row.get(value_name)) != 0
    ]
    if not signs:
        return None
    positive = sum(sign > 0 for sign in signs) / len(signs)
    return float(max(positive, 1.0 - positive))


def _summary(frame_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    actions = [action for frame in frame_rows for action in frame["actions"]]
    send = [action for action in actions if not action.get("no_send", False)]
    global_sign_valid = [
        action for action in send
        if _sign(action.get("true_global_delta_quality")) != 0
        and _sign(action.get("proxy_delta_quality")) != 0
    ]
    action_sign_valid = [
        action for action in send
        if _sign(action.get("true_delta_quality")) != 0
        and _sign(action.get("proxy_action_delta")) != 0
    ]
    ranking = [frame.get("ranking", {}) for frame in frame_rows]
    global_majority = _majority_sign_accuracy(
        send,
        "true_global_delta_quality",
    )
    marginal_majority = _majority_sign_accuracy(
        send,
        "true_delta_quality",
    )
    by_state = defaultdict(list)
    for action in actions:
        by_state[str(action.get("channel_state", "unknown"))].append(action)

    stage_names = (
        "feature_input_dense",
        "source_payload_before_quant",
        "quantized_then_dequantized",
        "recovered_payload_compact",
        "recovered_feature_dense",
    )
    first_zero_stage = Counter()
    for action in send:
        stages = action.get("transport_feature_stages") or {}
        found = False
        for stage_name in stage_names:
            stage = stages.get(stage_name) or {}
            if not stage.get("available", False):
                continue
            if (_float(stage.get("abs_max")) or 0.0) <= 1e-12:
                first_zero_stage[stage_name] += 1
                found = True
                break
        if not found:
            if stages:
                first_zero_stage["never_zero"] += 1
            else:
                first_zero_stage["audit_unavailable"] += 1

    transport_stage_summary = {}
    for stage_name in stage_names:
        stage_rows = [
            (action.get("transport_feature_stages") or {}).get(stage_name) or {}
            for action in send
        ]
        transport_stage_summary[stage_name] = {
            "available": sum(bool(row.get("available", False)) for row in stage_rows),
            "abs_max": _stats(row.get("abs_max") for row in stage_rows),
            "std": _stats(row.get("std") for row in stage_rows),
            "zero_ratio": _stats(row.get("zero_ratio") for row in stage_rows),
        }

    def group(rows):
        return {
            "count": len(rows),
            "true_quality": _stats(row.get("true_quality_mean_0357") for row in rows),
            "true_action_delta": _stats(
                row.get("true_delta_quality") for row in rows
            ),
            "true_global_delta": _stats(
                row.get("true_global_delta_quality") for row in rows
            ),
            "proxy_global_delta": _stats(
                row.get("proxy_delta_quality") for row in rows
            ),
            "proxy_action_delta": _stats(
                row.get("proxy_action_delta") for row in rows
            ),
            "tx_bytes": _stats(row.get("tx_bytes") for row in rows),
            "proxy_true_global_delta_pearson": _pearson(
                [row.get("proxy_delta_quality") for row in rows],
                [row.get("true_global_delta_quality") for row in rows],
            ),
            "proxy_true_global_delta_spearman": _spearman(
                [row.get("proxy_delta_quality") for row in rows],
                [row.get("true_global_delta_quality") for row in rows],
            ),
            "centered_proxy_true_action_delta_pearson": _pearson(
                [row.get("proxy_action_delta") for row in rows],
                [row.get("true_delta_quality") for row in rows],
            ),
            "centered_proxy_true_action_delta_spearman": _spearman(
                [row.get("proxy_action_delta") for row in rows],
                [row.get("true_delta_quality") for row in rows],
            ),
            "proxy_true_quality_pearson": _pearson(
                [row.get("proxy_collab_quality") for row in rows],
                [row.get("true_quality_mean_0357") for row in rows],
            ),
            "proxy_true_quality_spearman": _spearman(
                [row.get("proxy_collab_quality") for row in rows],
                [row.get("true_quality_mean_0357") for row in rows],
            ),
            "feature_changed_ratio": (
                float(
                    sum(
                        _float((row.get("sender_feature") or {}).get("after_nz_ratio"))
                        not in (None, 0.0)
                        for row in rows
                    ) / len(rows)
                ) if rows else None
            ),
            "psm_changed_vs_no_send_ratio": (
                float(
                    sum(
                        (_float((row.get("psm_vs_no_send") or {}).get("max_abs")) or 0.0)
                        > 1e-12
                        for row in rows
                    ) / len(rows)
                ) if rows else None
            ),
        }

    return {
        "num_audited_frames": len(frame_rows),
        "num_action_trials": len(actions),
        "action_counter": dict(Counter(row.get("action_id") for row in actions)),
        "overall": group(actions),
        "send_only": group(send),
        "by_channel_state": {
            state: group(rows) for state, rows in sorted(by_state.items())
        },
        "by_action_id": {
            action_id: group(
                [row for row in actions if row.get("action_id") == action_id]
            )
            for action_id in sorted(
                set(str(row.get("action_id")) for row in actions)
            )
        },
        "sender_before_rms": _stats(
            (row.get("sender_feature") or {}).get("before_rms") for row in send
        ),
        "sender_after_rms": _stats(
            (row.get("sender_feature") or {}).get("after_rms") for row in send
        ),
        "transport_stage_summary": transport_stage_summary,
        "first_zero_stage_counter": dict(first_zero_stage),
        "sign_accuracy": (
            float(
                sum(
                    _sign(row["true_global_delta_quality"])
                    == _sign(row["proxy_delta_quality"])
                    for row in global_sign_valid
                ) / len(global_sign_valid)
            )
            if global_sign_valid else None
        ),
        "sign_majority_baseline": global_majority,
        "sign_comparable": len(global_sign_valid),
        "marginal_sign_accuracy": (
            float(
                sum(
                    _sign(row["true_delta_quality"])
                    == _sign(row["proxy_action_delta"])
                    for row in action_sign_valid
                ) / len(action_sign_valid)
            )
            if action_sign_valid else None
        ),
        "marginal_sign_majority_baseline": marginal_majority,
        "marginal_sign_comparable": len(action_sign_valid),
        "ranking_pairwise_accuracy": _stats(
            row.get("pairwise_accuracy") for row in ranking
        ),
        "ranking_spearman": _stats(row.get("spearman") for row in ranking),
        "ranking_top1_match_rate": (
            float(
                sum(bool(row.get("top1_match")) for row in ranking)
                / max(sum(row.get("top1_match") is not None for row in ranking), 1)
            )
            if any(row.get("top1_match") is not None for row in ranking)
            else None
        ),
        "ranking_top_set_match_rate": (
            float(
                sum(bool(row.get("top_set_match")) for row in ranking)
                / max(
                    sum(
                        row.get("top_set_match") is not None
                        for row in ranking
                    ),
                    1,
                )
            )
            if any(
                row.get("top_set_match") is not None for row in ranking
            )
            else None
        ),
        "ranking_selected_action_regret": _stats(
            row.get("selected_action_regret") for row in ranking
        ),
        "online_selected_true_regret": _stats(
            frame.get("online_selected_true_regret") for frame in frame_rows
        ),
        "online_selected_true_top1_rate": (
            float(
                sum(bool(frame.get("online_selected_is_true_top")) for frame in frame_rows)
                / max(
                    sum(
                        frame.get("online_selected_is_true_top") is not None
                        for frame in frame_rows
                    ),
                    1,
                )
            )
            if any(
                frame.get("online_selected_is_true_top") is not None
                for frame in frame_rows
            )
            else None
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Matched-state counterfactual audit over the ARCE action space."
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument(
        "--out_csv",
        default="",
        help="Counterfactual proxy dataset CSV; defaults beside out_json.",
    )
    parser.add_argument("--max_frames", type=int, default=500)
    parser.add_argument("--audit_frames", type=int, default=20)
    parser.add_argument("--audit_stride", type=int, default=25)
    parser.add_argument("--audit_start", type=int, default=0)
    parser.add_argument("--sender_index", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--progress_interval", type=int, default=50)
    return parser.parse_args()


def _write_proxy_dataset(
    path: Path,
    frame_rows: List[Dict[str, Any]],
) -> int:
    head_feature_cols = (
        ["collab_" + name for name in HEAD_AP_PROXY_FEATURES]
        + ["ego_" + name for name in HEAD_AP_PROXY_FEATURES]
        + ["diff_" + name for name in HEAD_AP_PROXY_FEATURES]
        + list(PAIRED_SPATIAL_AP_PROXY_FEATURES)
    )
    decoded_feature_cols = (
        list(RICH_DECODED_BOX_FEATURES)
        + list(no_send_decoded_feature_names())
        + list(delta_decoded_feature_names())
        + list(PAIRED_DECODED_MATCH_FEATURES)
    )
    feature_cols = (
        head_feature_cols
        + decoded_feature_cols
        + list(RECEIVER_TRANSPORT_FEATURES)
    )
    fieldnames = [
        "frame_idx",
        "sequence_id",
        "sequence_frame_idx",
        "sender_index",
        "action_id",
        "executed_action_id",
        "channel_state",
        "quant_mode",
        "cache",
        "no_send",
        "tx_bytes",
        "true_quality_mean_0357",
        "true_ego_quality_mean_0357",
        "label_true_global_delta_quality_mean_0357",
        "label_true_delta_quality_mean_0357",
        "proxy_collab_quality",
        "proxy_ego_quality",
        "proxy_delta_quality",
        "proxy_action_delta",
        "proxy_source",
    ] + list(RUNTIME_CONTEXT_CSV_FIELDS) + feature_cols

    rows = []
    for frame in frame_rows:
        for action in frame.get("actions", []):
            if action.get("error") is not None:
                continue
            if action.get("true_delta_quality") is None:
                continue
            row = {
                "frame_idx": int(frame["frame_index"]),
                "sequence_id": int(frame["sequence_id"]),
                "sequence_frame_idx": int(frame["sequence_frame_index"]),
                "sender_index": int(frame["sender_index"]),
                "action_id": action.get("action_id"),
                "executed_action_id": action.get("executed_action_id"),
                "channel_state": action.get("channel_state"),
                "quant_mode": action.get("quant_mode"),
                "cache": action.get("cache"),
                "no_send": bool(action.get("no_send", False)),
                "tx_bytes": action.get("tx_bytes"),
                "true_quality_mean_0357": action.get(
                    "true_quality_mean_0357"
                ),
                "true_ego_quality_mean_0357": action.get(
                    "true_ego_quality_mean_0357"
                ),
                "label_true_global_delta_quality_mean_0357": action.get(
                    "true_global_delta_quality"
                ),
                "label_true_delta_quality_mean_0357": action.get(
                    "true_delta_quality"
                ),
                "proxy_collab_quality": action.get("proxy_collab_quality"),
                "proxy_ego_quality": action.get("proxy_ego_quality"),
                "proxy_delta_quality": action.get("proxy_delta_quality"),
                "proxy_action_delta": action.get("proxy_action_delta"),
                "proxy_source": action.get("proxy_source"),
            }
            for name in RUNTIME_CONTEXT_CSV_FIELDS:
                row[name] = action.get(name)
            for name in feature_cols:
                row[name] = action.get(name)
            rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    args = parse_args()
    output_path = Path(args.out_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hypes = yaml_utils.load_yaml(os.path.join(args.model_dir, "config.yaml"))
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=dataset.collate_batch_test,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_utils.create_model(hypes).to(device)
    _, model = train_utils.load_saved_model(args.model_dir, model)
    if hasattr(model, "update_epoch"):
        model.update_epoch(999)
    model.eval()
    base_comm = _get_comm(model)
    action_ids = list(getattr(base_comm, "action_ids", []) or [])
    if not action_ids:
        raise RuntimeError("Counterfactual audit requires a C2MAB action space.")
    if not hasattr(base_comm, "set_forced_action"):
        raise RuntimeError("ARCEC2MABComm forced-action audit API is missing.")
    compression_jsonl = output_path.parent / "counterfactual_transport_stages.jsonl"
    compression_jsonl.write_text("", encoding="utf-8")

    target_indices = set(
        int(args.audit_start) + i * max(1, int(args.audit_stride))
        for i in range(max(0, int(args.audit_frames)))
    )
    frame_rows = []
    started_at = time.perf_counter()
    last_progress_at = started_at
    last_progress_frame = 0
    model_forward_count = 0

    def report_progress(frames_processed: int, final: bool = False) -> None:
        nonlocal last_progress_at, last_progress_frame
        now = time.perf_counter()
        elapsed = max(now - started_at, 1e-9)
        interval_elapsed = max(now - last_progress_at, 1e-9)
        interval_frames = max(frames_processed - last_progress_frame, 0)
        print(
            (
                "progress frames={} audited={} forwards={} "
                "elapsed_s={:.1f} frame_fps={:.3f} "
                "interval_fps={:.3f} forward_fps={:.3f}{}"
            ).format(
                frames_processed,
                len(frame_rows),
                model_forward_count,
                elapsed,
                frames_processed / elapsed,
                interval_frames / interval_elapsed,
                model_forward_count / elapsed,
                " final" if final else "",
            ),
            flush=True,
        )
        last_progress_at = now
        last_progress_frame = frames_processed

    with torch.no_grad():
        for frame_index, batch in enumerate(loader):
            if int(args.max_frames) >= 0 and frame_index >= int(args.max_frames):
                break
            batch = train_utils.to_device(batch, device)
            should_audit = (
                frame_index in target_indices
                and _record_len(batch) > int(args.sender_index)
            )

            if should_audit:
                # base_comm has no accumulated debug records, but retains the
                # exact online Markov, cache, and policy state before this frame.
                state_snapshot = copy.deepcopy(base_comm)
                action_rows = []
                psm_by_action = {}
                decoded_by_action = {}

                for action_id in action_ids:
                    trial_comm = copy.deepcopy(state_snapshot)
                    trial_comm.clear_records()
                    compression_auditor = getattr(
                        getattr(trial_comm, "executor", None),
                        "compression_auditor",
                        None,
                    )
                    if compression_auditor is not None:
                        compression_auditor.enabled = True
                        compression_auditor.strict = False
                        compression_auditor.save_tensors = False
                        compression_auditor.output_dir = str(output_path.parent)
                        compression_auditor._jsonl_path = str(compression_jsonl)
                    trial_comm.set_forced_action(
                        action_id, sender_index=int(args.sender_index)
                    )
                    trial_comm.set_policy_updates_enabled(False)
                    _bind_comm(model, trial_comm)
                    start = len(trial_comm.get_records())
                    try:
                        output, pred_boxes, pred_scores, gt_boxes = _run_model(
                            batch, model, dataset
                        )
                        model_forward_count += 1
                        records = _records_since(trial_comm, start)
                        raw_comm_records = [
                            record for record in records
                            if is_communication_record(record)
                        ]
                        compact = [
                            _compact_comm_record(record)
                            for record in raw_comm_records
                        ]
                        target_record = next(
                            (
                                record for record in compact
                                if str(record.get("sender_id"))
                                == str(args.sender_index)
                            ),
                            compact[0] if compact else {},
                        )
                        raw_target_record = next(
                            (
                                record for record in raw_comm_records
                                if str(record.get("sender_id"))
                                == str(args.sender_index)
                            ),
                            raw_comm_records[0] if raw_comm_records else {},
                        )
                        quality = _quality(
                            pred_boxes, pred_scores, gt_boxes
                        )
                        ego_quality = _quality_from_heads(
                            dataset,
                            batch,
                            output["ego_psm"],
                            output["ego_rm"],
                        )
                        update = _reward_update(output)
                        proxy_features = paired_head_ap_proxy_features(
                            output["psm"],
                            output["rm"],
                            output["ego_psm"],
                            output["ego_rm"],
                        )
                        scores = (
                            pred_scores.detach().float().view(-1)
                            if torch.is_tensor(pred_scores) else None
                        )
                        feature_delta = _feature_delta(output)
                        decision_context = _decision_context_from_record(
                            raw_target_record
                        )
                        row = {
                            "action_id": str(action_id),
                            "executed_action_id": target_record.get("action_id"),
                            "no_send": bool(target_record.get("no_send", False)),
                            "channel_state": target_record.get("channel_state"),
                            "quant_mode": target_record.get("quant_mode"),
                            "cache": target_record.get("cache"),
                            "tx_bytes": target_record.get("tx_bytes"),
                            "num_pred_boxes": int(
                                0 if pred_boxes is None else len(pred_boxes)
                            ),
                            "pred_score_mean": (
                                float(scores.mean().cpu())
                                if scores is not None and scores.numel() else None
                            ),
                            "pred_score_max": (
                                float(scores.max().cpu())
                                if scores is not None and scores.numel() else None
                            ),
                            "true_quality_mean_0357": quality["quality_mean_0357"],
                            "true_ego_quality_mean_0357": ego_quality[
                                "quality_mean_0357"
                            ],
                            **quality,
                            "proxy_collab_quality": update.get("collab_confidence"),
                            "proxy_ego_quality": update.get("ego_confidence"),
                            "proxy_delta_quality": update.get("delta_confidence"),
                            "proxy_source": update.get("reward_delta_source"),
                            "feature_delta": feature_delta,
                            "transport_feature_stages": raw_target_record.get(
                                "compression_audit"
                            ),
                            "sender_feature": _sender_feature_stats(
                                feature_delta, int(args.sender_index)
                            ),
                            "psm": _tensor_summary(output.get("psm")),
                            "policy_update_applied": update.get("policy_update_applied"),
                            "decision_context": decision_context,
                            "reward_update_context": _update_context_from_record(
                                raw_target_record,
                                decision_context,
                            ),
                        }
                        row.update(proxy_features)
                        row.update(
                            _receiver_transport_features(raw_target_record)
                        )
                        row.update(
                            decoded_box_features(pred_boxes, pred_scores)
                        )
                        psm_by_action[action_id] = output.get("psm").detach().clone()
                        decoded_by_action[action_id] = (
                            (
                                pred_boxes.detach().float().cpu().clone()
                                if torch.is_tensor(pred_boxes)
                                else None
                            ),
                            (
                                pred_scores.detach().float().cpu().reshape(-1).clone()
                                if torch.is_tensor(pred_scores)
                                else None
                            ),
                        )
                    except Exception as exc:
                        row = {
                            "action_id": str(action_id),
                            "error": "{}: {}".format(type(exc).__name__, exc),
                        }
                    action_rows.append(row)

                _attach_group_decision_context(action_rows)

                no_send_id = next(
                    (
                        action_id for action_id in action_ids
                        if str(action_id).startswith("send0_")
                    ),
                    None,
                )
                no_send_row = next(
                    (row for row in action_rows if row["action_id"] == no_send_id),
                    None,
                )
                no_send_psm = psm_by_action.get(no_send_id)
                no_send_decoded = decoded_by_action.get(no_send_id)
                if no_send_row is not None and "error" not in no_send_row:
                    baseline_quality = float(no_send_row["true_quality_mean_0357"])
                    baseline_proxy = _float(
                        no_send_row.get("proxy_delta_quality")
                    )
                    for row in action_rows:
                        if "error" in row:
                            continue
                        row["true_delta_quality"] = float(
                            row["true_quality_mean_0357"] - baseline_quality
                        )
                        row["true_global_delta_quality"] = float(
                            row["true_quality_mean_0357"]
                            - row["true_ego_quality_mean_0357"]
                        )
                        row_proxy = _float(row.get("proxy_delta_quality"))
                        row["proxy_action_delta"] = (
                            float(row_proxy - baseline_proxy)
                            if row_proxy is not None and baseline_proxy is not None
                            else None
                        )
                        row["psm_vs_no_send"] = _tensor_diff(
                            psm_by_action.get(row["action_id"]), no_send_psm
                        )
                        for name in RICH_DECODED_BOX_FEATURES:
                            reference_name = "no_send_" + name
                            delta_name = "paired_delta_" + name
                            reference_value = float(no_send_row[name])
                            row[reference_name] = reference_value
                            row[delta_name] = (
                                float(row[name]) - reference_value
                            )
                        current_decoded = decoded_by_action.get(row["action_id"])
                        if (
                            current_decoded is not None
                            and no_send_decoded is not None
                        ):
                            row.update(
                                paired_decoded_box_features(
                                    current_decoded[0],
                                    current_decoded[1],
                                    no_send_decoded[0],
                                    no_send_decoded[1],
                                    iou_threshold=0.3,
                                )
                            )

                valid_states = sorted(
                    set(
                        str(row.get("channel_state")) for row in action_rows
                        if row.get("channel_state") is not None
                    )
                )
                sequence_id, sequence_frame_index = _sequence_position(
                    dataset,
                    frame_index,
                )
                frame_rows.append({
                    "frame_index": int(frame_index),
                    "sequence_id": int(sequence_id),
                    "sequence_frame_index": int(sequence_frame_index),
                    "sender_index": int(args.sender_index),
                    "channel_states": valid_states,
                    "matched_channel_state": len(valid_states) <= 1,
                    "actions": action_rows,
                    "ranking": _frame_ranking(action_rows),
                })

                # Restore the untouched pre-frame online state and execute the
                # actual policy once. This is the only run that advances it.
                base_comm = state_snapshot
                base_comm.clear_forced_action()
                base_comm.set_policy_updates_enabled(True)
                _bind_comm(model, base_comm)

            online_start = len(base_comm.get_records())
            _run_model(batch, model, dataset)
            model_forward_count += 1
            base_comm = _get_comm(model)
            online_records = _records_since(base_comm, online_start)

            if should_audit:
                online_actions = [
                    _compact_comm_record(record) for record in online_records
                    if is_communication_record(record)
                ]
                frame_row = frame_rows[-1]
                frame_row["online_actions"] = online_actions
                online_target = next(
                    (
                        row for row in online_actions
                        if str(row.get("sender_id")) == str(args.sender_index)
                    ),
                    None,
                )
                selected_action_id = (
                    online_target.get("action_id") if online_target else None
                )
                frame_row["online_selected_action"] = selected_action_id
                selected_trial = next(
                    (
                        row for row in frame_row["actions"]
                        if row.get("action_id") == selected_action_id
                        and "error" not in row
                    ),
                    None,
                )
                valid_trials = [
                    row for row in frame_row["actions"]
                    if "error" not in row
                    and _float(row.get("true_quality_mean_0357")) is not None
                ]
                if selected_trial is not None and valid_trials:
                    best_quality = max(
                        float(row["true_quality_mean_0357"])
                        for row in valid_trials
                    )
                    selected_quality = float(
                        selected_trial["true_quality_mean_0357"]
                    )
                    frame_row["online_selected_true_quality"] = selected_quality
                    frame_row["online_selected_true_regret"] = float(
                        best_quality - selected_quality
                    )
                    frame_row["online_selected_is_true_top"] = bool(
                        abs(best_quality - selected_quality) <= 1e-9
                    )
            base_comm.clear_records()

            frames_processed = frame_index + 1
            if (
                should_audit
                or (
                    int(args.progress_interval) > 0
                    and frames_processed % int(args.progress_interval) == 0
                )
            ):
                report_progress(frames_processed)

    frames_processed = min(
        len(dataset),
        int(args.max_frames) if int(args.max_frames) >= 0 else len(dataset),
    )
    if frames_processed != last_progress_frame:
        report_progress(frames_processed, final=True)

    payload = {
        "model_dir": args.model_dir,
        "protocol": "online_state_matched_single_sender_counterfactual",
        "feature_definition": (
            "canonical_psm_rm_head_plus_rich_paired_decoded_box_v33"
        ),
        "decoded_box_feature_schema": "v3.3_rich_paired_aabb_iou",
        "decoded_box_pair_iou_threshold": 0.3,
        "decoded_box_class_handling": (
            "class_agnostic because inference post-process does not expose "
            "predicted class labels"
        ),
        "decoded_box_reference_semantics": (
            "matched-state no-send action for the audited sender; this is "
            "not necessarily an ego-only frame when other senders exist"
        ),
        "sender_index": int(args.sender_index),
        "action_ids": action_ids,
        "summary": _summary(frame_rows),
        "frames": frame_rows,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    csv_path = (
        Path(args.out_csv)
        if str(args.out_csv).strip()
        else output_path.with_name("counterfactual_proxy_dataset.csv")
    )
    csv_rows = _write_proxy_dataset(csv_path, frame_rows)
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print("saved:", output_path)
    print("saved proxy dataset:", csv_path, "rows:", csv_rows)


if __name__ == "__main__":
    main()
