#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


def _as_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _get_nested(obj: Any, path: Iterable[str], default: Any = None) -> Any:
    cur = obj
    for key in path:
        if isinstance(cur, dict):
            if key not in cur:
                return default
            cur = cur[key]
        else:
            if not hasattr(cur, key):
                return default
            cur = getattr(cur, key)
    return cur


def _first_present(record: Dict[str, Any], paths: List[List[str]], default: Any = None) -> Any:
    for path in paths:
        value = _get_nested(record, path, None)
        if value is not None:
            return value
    return default


def move_to_cuda(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.cuda()
    if isinstance(x, dict):
        return {k: move_to_cuda(v) for k, v in x.items()}
    if isinstance(x, list):
        return [move_to_cuda(v) for v in x]
    if isinstance(x, tuple):
        return tuple(move_to_cuda(v) for v in x)
    return x


def normalize_method(method: str) -> str:
    key = str(method).strip().lower().replace("_", "").replace("-", "")
    if key in ("nofusion", "no"):
        return "nofusion"
    if key in ("v2xvit", "v2xvitmarkov"):
        return "v2xvit"
    if key in ("where2comm", "w2c"):
        return "where2comm"
    if key in ("arcec2mab", "c2mab", "where2commgrace", "arce"):
        return "arce_c2mab"
    if key in ("coopdiff", "coopdiffmarkov"):
        return "coopdiff"
    if key in ("rocooper", "rocoopermarkov"):
        return "rocooper"
    raise ValueError("Unsupported method for native BW summary: %s" % method)


def is_communication_record(record: Dict[str, Any]) -> bool:
    if not isinstance(record, dict):
        return False

    action = record.get("action")
    if isinstance(action, dict) and (
        action.get("action_id") is not None
        or action.get("quant_mode") is not None
        or action.get("send") is not None
        or action.get("send_flag") is not None
    ):
        return True

    if isinstance(record.get("dc2mab"), dict):
        dc = record["dc2mab"]
        if "selected" in dc or "proposal" in dc:
            return True

    if bool(record.get("no_send", False)):
        return True

    comm_paths = [
        ["packetization", "original_num_bytes"],
        ["byte_stream_packetization", "original_num_bytes"],
        ["budget_consistency", "executor_pre_budget_encoded_bytes"],
        ["budget_consistency", "proposal_estimated_encoded_bytes"],
        ["budget_consistency", "executor_actual_tx_bytes"],
        ["budget_consistency", "actual_tx_bytes"],
        ["size", "actual_transmitted_bytes"],
        ["size", "raw_bytes_fp32_reference"],
        ["actual_transmitted_bytes"],
        ["transmitted_bytes"],
        ["tx_bytes"],
    ]
    return _first_present(record, comm_paths, None) is not None


def is_no_send(record: Dict[str, Any]) -> bool:
    if bool(record.get("no_send", False)):
        return True
    action = record.get("action") or {}
    if isinstance(action, dict):
        action_id = str(action.get("action_id", ""))
        if action_id.startswith("send0_"):
            return True
        if bool(action.get("is_no_send", False)):
            return True
        if str(action.get("quant_mode", "")).lower() == "none":
            return True
    return False


def dense_native_bytes(record: Dict[str, Any]) -> float:
    value = _first_present(
        record,
        [
            ["packetization", "original_num_bytes"],
            ["byte_stream_packetization", "original_num_bytes"],
            ["packetization", "source_num_bytes"],
            ["size", "raw_bytes_fp32_reference"],
            ["size", "original_num_bytes"],
        ],
        None,
    )
    return float(_as_float(value, 0.0) or 0.0)


def arce_c2mab_pre_budget_encoded_bytes(record: Dict[str, Any]) -> Optional[float]:
    value = _first_present(
        record,
        [
            ["budget_consistency", "executor_pre_budget_encoded_bytes"],
            ["budget_consistency", "proposal_estimated_encoded_bytes"],
            ["dc2mab", "proposal", "record", "estimated_encoded_bytes"],
            ["dc2mab", "proposal", "estimated_encoded_bytes"],
            ["proposal", "estimated_encoded_bytes"],
        ],
        None,
    )
    if value is None:
        return None
    return float(_as_float(value, 0.0) or 0.0)


def actual_tx_bytes(record: Dict[str, Any]) -> float:
    value = _first_present(
        record,
        [
            ["budget_consistency", "executor_actual_tx_bytes"],
            ["budget_consistency", "actual_tx_bytes"],
            ["size", "actual_transmitted_bytes"],
            ["actual_transmitted_bytes"],
            ["transmitted_bytes"],
            ["tx_bytes"],
        ],
        None,
    )
    return float(_as_float(value, 0.0) or 0.0)


def is_selected(record: Dict[str, Any]) -> bool:
    if not is_communication_record(record) or is_no_send(record):
        return False

    selected = _get_nested(record, ["dc2mab", "selected"], None)
    if selected is not None:
        return bool(selected)

    selected = record.get("selected", None)
    if selected is not None:
        return bool(selected)

    action = record.get("action") or {}
    if isinstance(action, dict):
        action_id = str(action.get("action_id", ""))
        if action_id.startswith("send1_"):
            return True
        send = action.get("send", action.get("send_flag", None))
        if send is not None:
            return bool(send)

    # Fixed ARCE baselines normally only emit executed communication records.
    return actual_tx_bytes(record) > 0.0 or dense_native_bytes(record) > 0.0


def action_id(record: Dict[str, Any]) -> str:
    action = record.get("action") or {}
    return str(action.get("action_id", "")) if isinstance(action, dict) else ""


def action_quant(record: Dict[str, Any]) -> str:
    action = record.get("action") or {}
    return str(action.get("quant_mode", "")) if isinstance(action, dict) else ""


def action_rho(record: Dict[str, Any]) -> str:
    action = record.get("action") or {}
    if not isinstance(action, dict):
        return ""
    return str(action.get("redundancy_ratio", action.get("rho", "")))



def find_channel_records(model: torch.nn.Module, required_key: str = "message_bytes") -> List[Dict[str, Any]]:
    """Find method-native communication records from model submodules."""
    all_records: List[Dict[str, Any]] = []

    for _, module in model.named_modules():
        records = None

        if hasattr(module, "get_records"):
            try:
                records = module.get_records()
            except Exception:
                records = None

        if records is None and hasattr(module, "records"):
            try:
                records = getattr(module, "records")
            except Exception:
                records = None

        if not isinstance(records, list):
            continue

        for r in records:
            if isinstance(r, dict) and required_key in r:
                all_records.append(r)

    return all_records



def _to_int(value: Any, default: int = 0) -> int:
    try:
        if torch.is_tensor(value):
            if value.numel() == 0:
                return default
            return int(value.detach().cpu().flatten()[0].item())
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _as_dict_list(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    if isinstance(value, tuple):
        return [x for x in value if isinstance(x, dict)]
    return []


def rocooper_native_bw_from_output(
    output: Any,
    bytes_per_value: int = 4,
) -> Dict[str, Any]:
    """Estimate RoCooper native offered BW from fusion_info block routing.

    RoCooper exchanges selected feature blocks. Existing runtime output exposes
    block routing under:
        output["fusion_info"]["scenario_info"][*]["aggregator"]["rounds"]
    Each scale_info item contains num_selected_blocks and tokens_per_block.
    """
    info = {
        "native_bytes": 0.0,
        "num_scenarios": 0,
        "num_rounds": 0,
        "num_scale_records": 0,
        "num_selected_blocks": 0,
        "tokens_per_block_sum": 0,
        "num_valid_other_cav_sum": 0,
        "missing_scale_info": 0,
    }

    fusion_info = output.get("fusion_info", {}) if isinstance(output, dict) else {}
    scenario_info = _as_dict_list(fusion_info.get("scenario_info"))

    for scenario in scenario_info:
        info["num_scenarios"] += 1

        feature_shape = scenario.get("updated_others_shape")
        if not isinstance(feature_shape, (list, tuple)):
            feature_shape = fusion_info.get("feature_shape")

        channels = 0
        if isinstance(feature_shape, (list, tuple)) and len(feature_shape) >= 2:
            channels = _to_int(feature_shape[1], 0)

        aggregator = scenario.get("aggregator", {})
        if not isinstance(aggregator, dict):
            continue

        scenario_valid_others = _to_int(
            aggregator.get(
                "num_valid_other_cav",
                scenario.get("num_valid_other_cav", scenario.get("num_other_cav", 0)),
            ),
            0,
        )

        rounds = _as_dict_list(aggregator.get("rounds"))
        for round_info in rounds:
            info["num_rounds"] += 1
            scale_infos = _as_dict_list(round_info.get("scale_info"))
            if not scale_infos:
                info["missing_scale_info"] += 1

            for scale_info in scale_infos:
                selected_blocks = _to_int(scale_info.get("num_selected_blocks"), 0)
                tokens_per_block = _to_int(scale_info.get("tokens_per_block"), 0)
                valid_others = _to_int(
                    scale_info.get("num_valid_other_cav", scenario_valid_others),
                    scenario_valid_others,
                )

                if channels <= 0:
                    # Fallback: infer C from selected block shape if exposed.
                    bp = scale_info.get("block_prioritizer", {})
                    if isinstance(bp, dict):
                        selected = bp.get("selected_others")
                        if torch.is_tensor(selected) and selected.dim() >= 1:
                            channels = int(selected.shape[-1])

                if selected_blocks <= 0 or tokens_per_block <= 0 or valid_others <= 0 or channels <= 0:
                    continue

                record_bytes = (
                    float(valid_others)
                    * float(selected_blocks)
                    * float(tokens_per_block)
                    * float(channels)
                    * float(bytes_per_value)
                )

                info["native_bytes"] += record_bytes
                info["num_scale_records"] += 1
                info["num_selected_blocks"] += int(selected_blocks)
                info["tokens_per_block_sum"] += int(tokens_per_block)
                info["num_valid_other_cav_sum"] += int(valid_others)

    return info



def rocooper_channel_native_bw_from_output(
    output: Any,
    bytes_per_value: int = 4,
) -> Dict[str, Any]:
    """RoCooper main-table BW: non-ego feature tensor bytes at comm input.

    The paper's BlockPrioritizer is an Aggregator-side regional cross-learning
    module after feature reception, so its multi-round selected-block volume is
    kept as diagnostic, not as main communication BW.
    """
    info = {
        "native_bytes": 0.0,
        "num_non_ego": 0,
        "channels": 0,
        "height": 0,
        "width": 0,
        "feature_shape_found": False,
        "comm_info_found": False,
    }

    if not isinstance(output, dict):
        return info

    comm_info = output.get("comm_info", {})
    fusion_info = output.get("fusion_info", {})

    if isinstance(comm_info, dict):
        info["comm_info_found"] = True
        num_non_ego = _to_int(comm_info.get("num_non_ego", 0), 0)
    else:
        num_non_ego = 0

    feature_shape = None
    if isinstance(fusion_info, dict):
        feature_shape = fusion_info.get("feature_shape")

    # Fallback through scenario_info if feature_shape is not directly available.
    if not isinstance(feature_shape, (list, tuple)) and isinstance(fusion_info, dict):
        scenarios = _as_dict_list(fusion_info.get("scenario_info"))
        for scenario in scenarios:
            candidate = scenario.get("updated_others_shape")
            if isinstance(candidate, (list, tuple)):
                feature_shape = candidate
                break

    if isinstance(feature_shape, (list, tuple)) and len(feature_shape) >= 4:
        info["feature_shape_found"] = True
        # Shape is [total_cav, C, H, W] or [num_other, C, H, W].
        channels = _to_int(feature_shape[1], 0)
        height = _to_int(feature_shape[2], 0)
        width = _to_int(feature_shape[3], 0)

        if num_non_ego <= 0:
            total_cav = _to_int(feature_shape[0], 0)
            num_non_ego = max(0, total_cav - 1)

        info["num_non_ego"] = int(num_non_ego)
        info["channels"] = int(channels)
        info["height"] = int(height)
        info["width"] = int(width)

        if num_non_ego > 0 and channels > 0 and height > 0 and width > 0:
            info["native_bytes"] = float(num_non_ego * channels * height * width * bytes_per_value)

    return info


def extract_where2comm_rate(output: Any) -> Optional[float]:
    candidates = [
        ["comm_info", "where2comm_rate"],
        ["comm_info", "comm_rate"],
        ["communication_rates"],
        ["where2comm_rate"],
    ]
    for path in candidates:
        value = _get_nested(output, path, None)
        if value is None:
            continue
        if torch.is_tensor(value):
            if value.numel() == 0:
                continue
            return float(value.detach().float().mean().item())
        if isinstance(value, (list, tuple)):
            vals = [_as_float(v, None) for v in value]
            vals = [v for v in vals if v is not None]
            if vals:
                return float(sum(vals) / len(vals))
        try:
            return float(value)
        except Exception:
            pass
    return None


def load_hypes(model_dir: str) -> Dict[str, Any]:
    return yaml_utils.load_yaml(os.path.join(model_dir, "config.yaml"), None)


def build_loader(hypes: Dict[str, Any], batch_size: int, num_workers: int) -> DataLoader:
    dataset = build_dataset(hypes, visualize=False, train=False)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )


def build_model(model_dir: str, hypes: Dict[str, Any]) -> torch.nn.Module:
    model = train_utils.create_model(hypes)
    model.cuda()
    _, model = train_utils.load_saved_model(model_dir, model)
    model.eval()
    return model


def summarize_native_bw(
    model_dir: str,
    method: str,
    scenario: str,
    max_frames: int,
    batch_size: int,
    num_workers: int,
    progress_interval: int,
) -> Dict[str, Any]:
    method_key = normalize_method(method)

    if method_key == "nofusion":
        hypes = load_hypes(model_dir) if model_dir else None
        frame_count = 0
        if hypes is not None:
            loader = build_loader(hypes, batch_size=batch_size, num_workers=num_workers)
            limit = None if max_frames is None or max_frames < 0 else int(max_frames)
            frame_count = len(loader) if limit is None else min(len(loader), limit)
        return {
            "method": method,
            "scenario": scenario,
            "frame_count": int(frame_count),
            "native_total_MB": 0.0,
            "native_BW_MB_per_frame": 0.0,
            "native_bw_rule": "NoFusion sends no cooperative message.",
            "actual_total_MB": 0.0,
            "actual_BW_MB_per_frame": 0.0,
            "raw_record_count": 0,
            "record_count": 0,
            "skipped_non_comm_record_count": 0,
            "applied_link_count": 0,
            "no_send_count": 0,
            "where2comm_rate_avg": None,
            "where2comm_rate_min": None,
            "where2comm_rate_max": None,
            "where2comm_rate_count": 0,
            "notes": [],
        }

    hypes = load_hypes(model_dir)
    loader = build_loader(hypes, batch_size=batch_size, num_workers=num_workers)
    model = build_model(model_dir, hypes)

    if method_key in ("v2xvit", "where2comm", "arce_c2mab") and not hasattr(model, "arce_comm"):
        raise RuntimeError(
            "Model has no arce_comm records. Native BW for this method needs "
            "a method-specific extractor or an ARCE-enabled config: %s" % model_dir
        )

    max_n = None if max_frames is None or max_frames < 0 else int(max_frames)

    frame_count = 0
    raw_record_count = 0
    record_count = 0
    skipped_non_comm_record_count = 0
    applied_link_count = 0
    no_send_count = 0

    native_total_bytes = 0.0
    actual_total_bytes = 0.0

    where2comm_rates: List[float] = []
    action_id_counter: Dict[str, int] = defaultdict(int)
    quant_counter: Dict[str, int] = defaultdict(int)
    rho_counter: Dict[str, int] = defaultdict(int)

    missing_arce_pre_budget_count = 0
    rocooper_scale_record_count = 0
    rocooper_selected_block_count = 0
    rocooper_round_count = 0
    rocooper_scenario_count = 0
    rocooper_missing_scale_info_count = 0
    rocooper_block_routing_total_bytes = 0.0
    rocooper_channel_feature_shape_missing_count = 0
    rocooper_channel_comm_info_missing_count = 0
    notes: List[str] = []

    prev_record_count = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_n is not None and i >= max_n:
                break

            batch = move_to_cuda(batch)
            output = model(batch["ego"])
            frame_count += 1

            if method_key == "rocooper":
                # Main-table communication BW: non-ego feature tensor bytes at
                # RoCooper communication module input.
                channel_info = rocooper_channel_native_bw_from_output(output, bytes_per_value=4)

                native_total_bytes += float(channel_info.get("native_bytes", 0.0) or 0.0)
                raw_record_count += 1
                record_count += 1
                applied_link_count += int(channel_info.get("num_non_ego", 0) or 0)

                if not bool(channel_info.get("feature_shape_found", False)):
                    rocooper_channel_feature_shape_missing_count += 1
                if not bool(channel_info.get("comm_info_found", False)):
                    rocooper_channel_comm_info_missing_count += 1

                # Diagnostic only: Aggregator internal multi-round/multi-scale
                # block-routing processing volume. This is not the main BW.
                rocooper_info = rocooper_native_bw_from_output(output, bytes_per_value=4)
                rocooper_block_routing_total_bytes += float(rocooper_info.get("native_bytes", 0.0) or 0.0)
                rocooper_scale_record_count += int(rocooper_info.get("num_scale_records", 0) or 0)
                rocooper_selected_block_count += int(rocooper_info.get("num_selected_blocks", 0) or 0)
                rocooper_round_count += int(rocooper_info.get("num_rounds", 0) or 0)
                rocooper_scenario_count += int(rocooper_info.get("num_scenarios", 0) or 0)
                rocooper_missing_scale_info_count += int(rocooper_info.get("missing_scale_info", 0) or 0)

                if "rocooper_actual_bw_not_available_from_fusion_info" not in notes:
                    notes.append("rocooper_actual_bw_not_available_from_fusion_info")
                if "rocooper_block_routing_volume_is_diagnostic_not_main_bw" not in notes:
                    notes.append("rocooper_block_routing_volume_is_diagnostic_not_main_bw")

                if progress_interval > 0 and frame_count % progress_interval == 0:
                    print("%s native BW frames: %d" % (method, frame_count), flush=True)

                continue

            if method_key == "coopdiff":
                channel_records = find_channel_records(model, required_key="message_bytes")
                if not hasattr(model, "_native_bw_prev_channel_record_count"):
                    model._native_bw_prev_channel_record_count = 0

                prev = int(model._native_bw_prev_channel_record_count)
                new_channel_records = channel_records[prev:]
                model._native_bw_prev_channel_record_count = len(channel_records)

                raw_record_count += len(new_channel_records)
                record_count += len(new_channel_records)
                applied_link_count += len(new_channel_records)

                native_frame_bytes = sum(
                    float(r.get("message_bytes", 0.0) or 0.0)
                    for r in new_channel_records
                    if isinstance(r, dict)
                )
                actual_frame_bytes = sum(
                    float(r.get("consumed_bytes", 0.0) or 0.0)
                    for r in new_channel_records
                    if isinstance(r, dict)
                )

                native_total_bytes += float(native_frame_bytes)
                actual_total_bytes += float(actual_frame_bytes)

                if progress_interval > 0 and frame_count % progress_interval == 0:
                    print("%s native BW frames: %d" % (method, frame_count), flush=True)

                continue

            records = model.arce_comm.get_records()
            new_records = records[prev_record_count:]
            prev_record_count = len(records)

            raw_record_count += len(new_records)
            comm_records = [
                r for r in new_records
                if isinstance(r, dict) and is_communication_record(r)
            ]
            skipped_non_comm_record_count += len(new_records) - len(comm_records)
            record_count += len(comm_records)

            selected_records = [r for r in comm_records if is_selected(r)]
            no_send_count += sum(1 for r in comm_records if is_no_send(r))
            applied_link_count += len(selected_records)

            rate = extract_where2comm_rate(output)
            if rate is not None:
                where2comm_rates.append(float(rate))

            if method_key == "v2xvit":
                native_frame_bytes = sum(dense_native_bytes(r) for r in selected_records)

            elif method_key == "where2comm":
                dense_frame_bytes = sum(dense_native_bytes(r) for r in selected_records)
                if rate is None:
                    native_frame_bytes = dense_frame_bytes
                    if "where2comm_rate_missing_fallback_to_dense_bytes" not in notes:
                        notes.append("where2comm_rate_missing_fallback_to_dense_bytes")
                else:
                    native_frame_bytes = dense_frame_bytes * float(rate)
                    if "where2comm_native_bw_is_rate_based_estimate_requires_mask_validation" not in notes:
                        notes.append(
                            "where2comm_native_bw_is_rate_based_estimate_requires_mask_validation"
                        )

            elif method_key == "arce_c2mab":
                native_frame_bytes = 0.0
                for r in selected_records:
                    v = arce_c2mab_pre_budget_encoded_bytes(r)
                    if v is None:
                        missing_arce_pre_budget_count += 1
                        continue
                    native_frame_bytes += float(v)

            elif method_key == "coopdiff":
                # CoopDiff native records are accumulated in CoopDiffMarkovFeatureChannel.
                # We count only records newly generated by this frame.
                channel_records = find_channel_records(model, required_key="message_bytes")
                if not hasattr(model, "_native_bw_prev_channel_record_count"):
                    model._native_bw_prev_channel_record_count = 0
                prev = int(model._native_bw_prev_channel_record_count)
                new_channel_records = channel_records[prev:]
                model._native_bw_prev_channel_record_count = len(channel_records)

                native_frame_bytes = sum(float(r.get("message_bytes", 0.0) or 0.0) for r in new_channel_records)
                actual_total_bytes += sum(float(r.get("consumed_bytes", 0.0) or 0.0) for r in new_channel_records)

                record_count += len(new_channel_records)
                applied_link_count += len(new_channel_records)

            else:
                raise AssertionError(method_key)

            native_total_bytes += float(native_frame_bytes)
            if method_key != "coopdiff":
                actual_total_bytes += sum(actual_tx_bytes(r) for r in selected_records)

            for r in selected_records:
                aid = action_id(r)
                if aid:
                    action_id_counter[aid] += 1
                q = action_quant(r)
                if q:
                    quant_counter[q] += 1
                rho = action_rho(r)
                if rho:
                    rho_counter[rho] += 1

            if progress_interval > 0 and frame_count % progress_interval == 0:
                print("%s native BW frames: %d" % (method, frame_count), flush=True)

    denom = max(float(frame_count), 1.0)

    if method_key == "v2xvit":
        rule = "Dense native feature bytes before ARCE budget/loss."
    elif method_key == "where2comm":
        rule = (
            "Rate-based Where2Comm native estimate: dense original bytes multiplied "
            "by reported where2comm_rate; requires mask-level validation."
        )
    elif method_key == "arce_c2mab":
        rule = "ARCE-C2MAB selected action pre-budget encoded bytes; no dense fallback."
    elif method_key == "coopdiff":
        rule = "CoopDiff native offered BW from CoopDiffMarkovFeatureChannel message_bytes."
    elif method_key == "rocooper":
        rule = "RoCooper native offered BW from non-ego feature tensor bytes at communication module input."
    else:
        rule = "Unknown method rule."

    if method_key == "arce_c2mab" and missing_arce_pre_budget_count > 0:
        notes.append("arce_c2mab_missing_pre_budget_records_not_counted")

    return {
        "method": method,
        "scenario": scenario,
        "frame_count": int(frame_count),
        "native_total_MB": float(native_total_bytes / 1_000_000.0),
        "native_BW_MB_per_frame": float(native_total_bytes / 1_000_000.0 / denom),
        "native_bw_rule": rule,
        "actual_total_MB": float(actual_total_bytes / 1_000_000.0),
        "actual_BW_MB_per_frame": float(actual_total_bytes / 1_000_000.0 / denom),
        "raw_record_count": int(raw_record_count),
        "record_count": int(record_count),
        "skipped_non_comm_record_count": int(skipped_non_comm_record_count),
        "applied_link_count": int(applied_link_count),
        "no_send_count": int(no_send_count),
        "where2comm_rate_avg": (
            float(sum(where2comm_rates) / len(where2comm_rates))
            if where2comm_rates
            else None
        ),
        "where2comm_rate_min": float(min(where2comm_rates)) if where2comm_rates else None,
        "where2comm_rate_max": float(max(where2comm_rates)) if where2comm_rates else None,
        "where2comm_rate_count": int(len(where2comm_rates)),
        "missing_arce_pre_budget_count": int(missing_arce_pre_budget_count),
        "rocooper_scale_record_count": int(rocooper_scale_record_count),
        "rocooper_selected_block_count": int(rocooper_selected_block_count),
        "rocooper_round_count": int(rocooper_round_count),
        "rocooper_scenario_count": int(rocooper_scenario_count),
        "rocooper_missing_scale_info_count": int(rocooper_missing_scale_info_count),
        "rocooper_block_routing_total_MB": float(rocooper_block_routing_total_bytes / 1_000_000.0),
        "rocooper_block_routing_BW_MB_per_frame": float(rocooper_block_routing_total_bytes / 1_000_000.0 / denom),
        "rocooper_channel_feature_shape_missing_count": int(rocooper_channel_feature_shape_missing_count),
        "rocooper_channel_comm_info_missing_count": int(rocooper_channel_comm_info_missing_count),
        "action_id_counter": dict(sorted(action_id_counter.items())),
        "quant_counter": dict(sorted(quant_counter.items())),
        "rho_counter": dict(sorted(rho_counter.items())),
        "notes": notes,
    }


def write_csv(path: str, summary: Dict[str, Any]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    keys = [
        "method",
        "scenario",
        "frame_count",
        "native_BW_MB_per_frame",
        "native_total_MB",
        "actual_BW_MB_per_frame",
        "actual_total_MB",
        "raw_record_count",
        "record_count",
        "skipped_non_comm_record_count",
        "applied_link_count",
        "no_send_count",
        "where2comm_rate_avg",
        "where2comm_rate_min",
        "where2comm_rate_max",
        "where2comm_rate_count",
        "missing_arce_pre_budget_count",
        "rocooper_scale_record_count",
        "rocooper_selected_block_count",
        "rocooper_round_count",
        "rocooper_scenario_count",
        "rocooper_missing_scale_info_count",
        "rocooper_block_routing_BW_MB_per_frame",
        "rocooper_channel_feature_shape_missing_count",
        "rocooper_channel_comm_info_missing_count",
        "native_bw_rule",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerow({k: summary.get(k) for k in keys})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize native offered BW for OPV2V main-table methods."
    )
    parser.add_argument("--model_dir", default="", help="OpenCOOD model/log directory.")
    parser.add_argument("--method", required=True, help="NoFusion, V2X-VIT, Where2Comm, ARCE-C2MAB.")
    parser.add_argument("--scenario", default="Markov")
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--progress_interval", type=int, default=50)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_csv", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = summarize_native_bw(
        model_dir=args.model_dir,
        method=args.method,
        scenario=args.scenario,
        max_frames=args.max_frames,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        progress_interval=args.progress_interval,
    )

    out_dir = os.path.dirname(args.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.out_csv:
        write_csv(args.out_csv, summary)

    print("===== Native BW summary =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("saved json:", args.out_json)
    if args.out_csv:
        print("saved csv:", args.out_csv)


if __name__ == "__main__":
    main()
