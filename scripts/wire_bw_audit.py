#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Measure sender-to-ego wire bytes under ideal or Markov channels for one
collaborative perception baseline.

The primary metric is:

    total actual bytes entering the physical Markov channel
    -------------------------------------------------------
                    evaluated frame count

All sender->ego links and all native payload segments/scales are summed inside
each perception frame before averaging.

Supported baseline families:
    - Where2Comm with ARCE fixed Markov executor
    - V2X-ViT with ARCE fixed Markov executor
    - RoCooper Markov communication module
    - CoopDiff Markov feature channel
    - CoSDH paper-native mixed-dtype byte stream

This script does not change a model config on disk and does not change model
or channel source code.
"""

from __future__ import print_function

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils
from opencood.communication.metrics.ideal_wire_auditor import IdealWireAuditor
from opencood.data_utils.datasets import build_dataset


MB = 1000.0 * 1000.0
MIB = 1024.0 * 1024.0


def json_safe(value):
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        if value.numel() == 1:
            return json_safe(value.detach().cpu().item())
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "as_dict") and callable(value.as_dict):
        try:
            return json_safe(value.as_dict())
        except Exception:
            pass
    return str(value)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(data), handle, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False))
            handle.write("\n")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            encoded = {}
            for key in fieldnames:
                value = json_safe(row.get(key, ""))
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)
                encoded[key] = value
            writer.writerow(encoded)


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_hypes(model_dir):
    class Opt(object):
        pass

    opt = Opt()
    opt.model_dir = str(model_dir)
    try:
        return yaml_utils.load_yaml(None, opt)
    except Exception:
        config_path = Path(model_dir) / "config.yaml"
        if not config_path.exists():
            raise
        return yaml_utils.load_yaml(str(config_path), None)


def patch_test_dir(hypes, test_dir):
    if not test_dir:
        return
    test_dir = str(Path(test_dir).expanduser())
    if "validate_dir" in hypes:
        hypes["validate_dir"] = test_dir
    elif "test_dir" in hypes:
        hypes["test_dir"] = test_dir
    else:
        # Most OpenCOOD test datasets read validate_dir when train=False.
        hypes["validate_dir"] = test_dir


def recursively_override_markov_seed(value, seed, path=()):
    if isinstance(value, dict):
        for key in list(value.keys()):
            child_path = path + (str(key).lower(),)
            if str(key).lower() == "seed" and any(
                ("markov" in part or "channel" in part)
                for part in child_path[:-1]
            ):
                value[key] = int(seed)
            else:
                recursively_override_markov_seed(value[key], seed, child_path)
    elif isinstance(value, list):
        for item in value:
            recursively_override_markov_seed(item, seed, path)


def get_model_args(hypes):
    return (
        hypes.get("model", {})
        .get("args", {})
    )


def audit_policy_config(hypes, baseline):
    model_args = get_model_args(hypes)
    findings = {
        "baseline": baseline,
        "arce_enabled": None,
        "arce_mode": None,
        "arce_policy": None,
        "cosdh_channel_mode": None,
        "policy_detected": False,
        "warnings": [],
    }

    arce = model_args.get("arce", hypes.get("arce", {})) or {}
    if isinstance(arce, dict) and arce:
        findings["arce_enabled"] = bool(arce.get("enabled", False))
        findings["arce_mode"] = arce.get("mode")
        findings["arce_policy"] = arce.get("policy")
        mode_text = "{} {}".format(
            arce.get("mode", ""), arce.get("policy", "")
        ).lower()
        if "c2mab" in mode_text or "dc2mab" in mode_text:
            findings["policy_detected"] = True
            findings["warnings"].append(
                "C2MAB/DC2MAB policy is enabled; this is not a fixed baseline."
            )

    paper_cfg = model_args.get("cosdh_paper_native", {}) or {}
    if isinstance(paper_cfg, dict):
        findings["cosdh_channel_mode"] = paper_cfg.get("channel_mode")
        if baseline == "cosdh":
            if str(paper_cfg.get("channel_mode", "")).lower() != "markov":
                findings["warnings"].append(
                    "CoSDH paper byte stream is not configured as markov."
                )
            if findings["arce_enabled"]:
                findings["warnings"].append(
                    "CoSDH has ARCE enabled; expected no-policy byte stream."
                )
                findings["policy_detected"] = True

    return findings


def resolve_numeric_checkpoint(model_dir, requested_epoch):
    model_dir = Path(model_dir)
    candidates = []
    for path in model_dir.glob("net_epoch*.pth"):
        match = re.fullmatch(r"net_epoch(\d+)\.pth", path.name)
        if match:
            candidates.append((int(match.group(1)), path))

    if not candidates:
        raise FileNotFoundError(
            "No numeric checkpoint such as net_epoch20.pth in {}".format(
                model_dir
            )
        )

    if str(requested_epoch).lower() == "auto":
        return max(candidates, key=lambda item: item[0])

    epoch = int(requested_epoch)
    for item in candidates:
        if item[0] == epoch:
            return item

    raise FileNotFoundError(
        "Requested epoch {} not found in {}. Available: {}".format(
            epoch,
            model_dir,
            sorted(item[0] for item in candidates),
        )
    )


def load_model_with_exact_checkpoint(hypes, model_dir, epoch, out_dir, device):
    selected_epoch, checkpoint = resolve_numeric_checkpoint(
        model_dir, epoch
    )

    runtime_dir = Path(out_dir) / "model_runtime_checkpoint"
    if runtime_dir.exists():
        shutil.rmtree(str(runtime_dir))
    runtime_dir.mkdir(parents=True, exist_ok=True)

    runtime_checkpoint = runtime_dir / checkpoint.name
    try:
        runtime_checkpoint.symlink_to(checkpoint.resolve())
    except Exception:
        shutil.copy2(str(checkpoint), str(runtime_checkpoint))

    config_src = Path(model_dir) / "config.yaml"
    if config_src.exists():
        shutil.copy2(str(config_src), str(runtime_dir / "config.yaml"))

    model = train_utils.create_model(hypes)
    model.to(device)

    loaded = train_utils.load_saved_model(str(runtime_dir), model)
    loaded_epoch = selected_epoch
    if isinstance(loaded, tuple) and len(loaded) >= 2:
        try:
            loaded_epoch = int(loaded[0])
        except Exception:
            loaded_epoch = selected_epoch
        model = loaded[1]
    elif isinstance(loaded, torch.nn.Module):
        model = loaded

    model.to(device)
    model.eval()
    return loaded_epoch, checkpoint, model


def build_loader(hypes, baseline, num_workers):
    if baseline == "cosdh":
        from opencood.data_utils.datasets.cosdh_dataset_builder import (
            build_dataset_cosdh,
        )
        dataset = build_dataset_cosdh(
            hypes, visualize=False, train=False
        )
    else:
        dataset = build_dataset(
            hypes, visualize=False, train=False
        )

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=int(num_workers),
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    return dataset, loader


def scalar_int(value, default=0):
    try:
        if torch.is_tensor(value):
            return int(value.detach().reshape(-1)[0].item())
        return int(np.asarray(value).reshape(-1)[0])
    except Exception:
        return int(default)


def collaborator_count(batch_data):
    try:
        record_len = batch_data["ego"]["record_len"]
        if torch.is_tensor(record_len):
            values = record_len.detach().cpu().reshape(-1).tolist()
        else:
            values = np.asarray(record_len).reshape(-1).tolist()
        return sum(max(0, int(value) - 1) for value in values)
    except Exception:
        return 0


def frame_id_from_batch(batch_data, sample_index):
    try:
        ego = batch_data.get("ego", {})
        for key in ("frame_id", "timestamp", "sample_idx", "sample_id"):
            if key in ego:
                return json_safe(ego[key])
    except Exception:
        pass
    return int(sample_index)


def transformation_distance(cav_content):
    matrix = cav_content.get("transformation_matrix")
    if matrix is None:
        return float("inf")
    if torch.is_tensor(matrix):
        array = matrix.detach().cpu().numpy()
    else:
        array = np.asarray(matrix)
    while array.ndim > 2:
        array = array[0]
    if array.shape != (4, 4):
        return float("inf")
    tx = float(array[0, 3])
    ty = float(array[1, 3])
    return math.sqrt(tx * tx + ty * ty)


def normalize_cosdh_batch(batch_data, dataset):
    """Match CoSDH late senders to ego record_len/comm_range.

    Some CoSDH datasets keep out-of-range agent entries in batch_data even
    though ego.record_len excludes them. The normal inference post-process
    filters them by comm_range. This audit performs the same filtering before
    constructing paper-native late messages.
    """
    if "ego" not in batch_data:
        return batch_data

    expected = max(
        0, scalar_int(batch_data["ego"].get("record_len"), 1) - 1
    )
    non_ego = [
        (key, value)
        for key, value in batch_data.items()
        if key != "ego"
    ]
    if len(non_ego) == expected:
        return batch_data

    params = getattr(dataset, "params", {}) or {}
    comm_range = float(params.get("comm_range", float("inf")))
    ranked = []
    for key, value in non_ego:
        distance = transformation_distance(value)
        ranked.append((distance, key, value))
    ranked.sort(key=lambda item: item[0])

    selected = [
        (key, value)
        for distance, key, value in ranked
        if distance <= comm_range
    ]
    if len(selected) != expected:
        selected = [
            (key, value) for _, key, value in ranked[:expected]
        ]

    if len(selected) != expected:
        raise RuntimeError(
            "CoSDH batch has {} valid late senders, record_len expects {}"
            .format(len(selected), expected)
        )

    result = OrderedDict()
    result["ego"] = batch_data["ego"]
    for key, value in selected:
        result[key] = value
    return result


_SIMPLE_TYPES = (str, int, float, bool, bytes, bytearray, type(None))


def discover_record_sources(model, max_depth=5):
    found = []
    visited = set()

    def add_source(obj):
        if obj is None:
            return
        obj_id = id(obj)
        for existing, _ in found:
            if id(existing) == obj_id:
                return
        getter = getattr(obj, "get_records", None)
        if callable(getter):
            try:
                records = getter()
                if isinstance(records, (list, tuple)):
                    found.append((obj, "get_records"))
                    return
            except Exception:
                pass
        for attr in (
            "records",
            "_records",
            "communication_records",
            "audit_records",
        ):
            records = getattr(obj, attr, None)
            if isinstance(records, list):
                found.append((obj, attr))
                return

    def walk(obj, depth):
        if obj is None or depth < 0:
            return
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        add_source(obj)

        if depth == 0 or isinstance(obj, _SIMPLE_TYPES):
            return
        if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
            return

        if isinstance(obj, dict):
            for child in obj.values():
                walk(child, depth - 1)
            return
        if isinstance(obj, (list, tuple)):
            for child in obj:
                walk(child, depth - 1)
            return

        if hasattr(obj, "named_modules"):
            try:
                for _, module in obj.named_modules():
                    add_source(module)
            except Exception:
                pass

        if hasattr(obj, "__dict__"):
            for name, child in vars(obj).items():
                if name.startswith("_parameters") or name.startswith("_buffers"):
                    continue
                if child is obj:
                    continue
                walk(child, depth - 1)

    walk(model, int(max_depth))
    return found


def source_records(source):
    obj, kind = source
    if kind == "get_records":
        records = obj.get_records()
    else:
        records = getattr(obj, kind, [])
    return list(records) if isinstance(records, (list, tuple)) else []


def reset_record_sources(sources):
    for source in sources:
        obj, kind = source
        for method_name in ("reset_records", "clear_records"):
            method = getattr(obj, method_name, None)
            if callable(method):
                try:
                    method()
                    break
                except Exception:
                    pass
        else:
            if kind != "get_records":
                records = getattr(obj, kind, None)
                if isinstance(records, list):
                    records[:] = []
            else:
                records = getattr(obj, "records", None)
                if isinstance(records, list):
                    records[:] = []


def record_offsets(sources):
    offsets = {}
    for source in sources:
        try:
            offsets[id(source[0])] = len(source_records(source))
        except Exception:
            offsets[id(source[0])] = 0
    return offsets


def collect_new_raw_records(sources, offsets):
    rows = []
    for source in sources:
        obj = source[0]
        try:
            all_rows = source_records(source)
        except Exception:
            continue
        start = int(offsets.get(id(obj), 0))
        offsets[id(obj)] = len(all_rows)
        rows.extend(copy.deepcopy(all_rows[start:]))
    return rows


def nested_get(data, path):
    cur = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def finite_number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


TX_PATHS = (
    ("transmitted_wire_bytes",),
    ("tx_wire_bytes",),
    ("tx_bytes",),
    ("size", "actual_transmitted_bytes"),
    ("actual_tx_bytes",),
    ("actual_transmitted_bytes",),
    ("sent_bytes_before_loss",),
    ("consumed_bytes",),
    ("channel", "latency", "transmitted_bytes"),
    ("transport", "actual_tx_bytes"),
)

RX_PATHS = (
    ("received_wire_bytes",),
    ("rx_wire_bytes",),
    ("rx_bytes",),
    ("received_valid_bytes",),
    ("actual_rx_bytes",),
    ("actual_received_bytes",),
    ("effective_received_bytes",),
    ("received_bytes",),
)

SOURCE_PATHS = (
    ("source_payload_bytes",),
    ("joint_source_bytes",),
    ("source_bytes",),
    ("raw_bytes",),
    ("message_bytes",),
    ("encoded_bytes",),
    ("size", "compressed_bytes"),
    ("compressed_bytes",),
)

BUDGET_PATHS = (
    ("budget_bytes",),
    ("bandwidth_budget_bytes",),
    ("initial_budget_bytes",),
    ("size", "bandwidth_budget_bytes"),
    ("system_budget", "link_budget_bytes"),
)

TRUNCATED_PATHS = (
    ("budget_truncated_bytes",),
    ("dropped_by_budget_bytes",),
    ("source_dropped_by_budget_bytes",),
)

STATE_PATHS = (
    ("state",),
    ("channel_state",),
    ("channel", "profile", "state_name"),
    ("profile", "state_name"),
)

LINK_PATHS = (
    ("link_key",),
    ("link_id",),
    ("agent_index",),
    ("cav",),
    ("cav_id",),
    ("sender_id",),
)

FRAME_PATHS = (
    ("frame_id",),
    ("frame_index",),
    ("sample_index",),
)


def first_number(data, paths):
    for path in paths:
        value = finite_number(nested_get(data, path))
        if value is not None:
            return value
    return None


def first_value(data, paths):
    for path in paths:
        value = nested_get(data, path)
        if value is not None:
            return json_safe(value)
    return None


def recursive_numeric_key(data, accepted, rejected=()):
    if not isinstance(data, dict):
        return None
    stack = [data]
    while stack:
        item = stack.pop()
        for key, value in item.items():
            lower = str(key).lower()
            if isinstance(value, dict):
                stack.append(value)
                continue
            if isinstance(value, list):
                continue
            if lower in accepted and not any(token in lower for token in rejected):
                number = finite_number(value)
                if number is not None:
                    return number
    return None


def normalize_record(raw, fallback_frame_id, sample_index, source_name):
    if not isinstance(raw, dict):
        return None

    tx = first_number(raw, TX_PATHS)
    if tx is None:
        tx = recursive_numeric_key(
            raw,
            {
                "transmitted_wire_bytes",
                "tx_wire_bytes",
                "tx_bytes",
                "actual_tx_bytes",
                "actual_transmitted_bytes",
                "sent_bytes_before_loss",
                "consumed_bytes",
                "transmitted_bytes",
            },
            rejected=("estimate", "predicted", "budget", "source"),
        )

    if tx is None:
        return None

    rx = first_number(raw, RX_PATHS)
    source_bytes = first_number(raw, SOURCE_PATHS)
    budget = first_number(raw, BUDGET_PATHS)
    truncated = first_number(raw, TRUNCATED_PATHS)

    if truncated is None and source_bytes is not None:
        truncated = max(0.0, float(source_bytes) - float(tx))

    state = first_value(raw, STATE_PATHS)
    link_key = first_value(raw, LINK_PATHS)
    frame_id = first_value(raw, FRAME_PATHS)
    if frame_id is None:
        frame_id = fallback_frame_id

    scale_idx = raw.get("scale_idx")
    segment_name = raw.get("name", raw.get("segment_name"))
    kind = raw.get("kind")
    no_send = bool(raw.get("no_send", False))
    bypassed = bool(raw.get("bypassed", False))

    return {
        "sample_index": int(sample_index),
        "frame_id": frame_id,
        "link_key": link_key,
        "state": state,
        "tx_bytes": max(0.0, float(tx)),
        "rx_bytes": None if rx is None else max(0.0, float(rx)),
        "source_bytes": (
            None if source_bytes is None
            else max(0.0, float(source_bytes))
        ),
        "budget_bytes": (
            None if budget is None else max(0.0, float(budget))
        ),
        "budget_truncated_bytes": (
            None if truncated is None
            else max(0.0, float(truncated))
        ),
        "scale_idx": json_safe(scale_idx),
        "segment_name": json_safe(segment_name),
        "kind": json_safe(kind),
        "no_send": no_send,
        "bypassed": bypassed,
        "record_source": source_name,
    }


_CHILD_RECORD_KEYS = (
    "link_records",
    "wire_records",
    "channel_links",
    "records",
    "latest_info",
    "frame_records",
    "links",
)


def extract_records_from_structure(value, fallback_frame_id, sample_index, source_name):
    result = []
    visited = set()

    def walk(item, path):
        if item is None:
            return
        item_id = id(item)
        if item_id in visited:
            return
        if isinstance(item, (dict, list, tuple)):
            visited.add(item_id)

        if isinstance(item, dict):
            child_found = False
            for key in _CHILD_RECORD_KEYS:
                child = item.get(key)
                if isinstance(child, (list, tuple)):
                    child_found = True
                    for index, record in enumerate(child):
                        walk(record, "{}.{}[{}]".format(path, key, index))
            if child_found:
                return

            normalized = normalize_record(
                item,
                fallback_frame_id=fallback_frame_id,
                sample_index=sample_index,
                source_name="{}:{}".format(source_name, path),
            )
            if normalized is not None:
                result.append(normalized)
                return

            for key, child in item.items():
                if isinstance(child, (dict, list, tuple)):
                    walk(child, "{}.{}".format(path, key))
            return

        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                walk(child, "{}[{}]".format(path, index))

    walk(value, "root")
    return result


def record_dedup_key(row):
    return (
        json.dumps(json_safe(row.get("frame_id")), sort_keys=True),
        json.dumps(json_safe(row.get("link_key")), sort_keys=True),
        str(row.get("state")),
        round(float(row.get("tx_bytes", 0.0)), 6),
        None if row.get("rx_bytes") is None else round(float(row["rx_bytes"]), 6),
        None if row.get("source_bytes") is None else round(float(row["source_bytes"]), 6),
        json.dumps(json_safe(row.get("scale_idx")), sort_keys=True),
        str(row.get("segment_name")),
        str(row.get("kind")),
        bool(row.get("no_send")),
    )


def deduplicate_records(rows):
    result = []
    seen = set()
    for row in rows:
        key = record_dedup_key(row)
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def latest_model_structures(model, output):
    structures = [("model_output", output)]
    for attr in (
        "latest_paper_native_info",
        "latest_info",
        "comm_info",
    ):
        if hasattr(model, attr):
            structures.append(("model.{}".format(attr), getattr(model, attr)))
    for name, module in model.named_modules():
        for attr in ("latest_info", "comm_info"):
            if hasattr(module, attr):
                structures.append((
                    "module.{}.{}".format(name, attr),
                    getattr(module, attr),
                ))
    return structures


def run_one_frame(batch_data, model, dataset, baseline):
    if baseline == "cosdh":
        from opencood.tools import inference_utils_cosdh
        normalized = normalize_cosdh_batch(batch_data, dataset)
        return inference_utils_cosdh.inference_late_fusion(
            normalized, model, dataset
        )
    return model(batch_data["ego"])


def summarize(name, dataset_name, baseline, model_dir, epoch, checkpoint,
              max_frames, seed, frame_rows, link_rows, diagnostics,
              policy_audit, elapsed_seconds):
    frame_count = len(frame_rows)
    total_tx = sum(float(row["tx_bytes"]) for row in frame_rows)
    total_rx = sum(float(row["rx_bytes"]) for row in frame_rows)
    total_source = sum(float(row["source_bytes"]) for row in frame_rows)
    total_truncated = sum(
        float(row["budget_truncated_bytes"]) for row in frame_rows
    )
    communication_frames = sum(
        1 for row in frame_rows if int(row["collaborator_count"]) > 0
    )
    tx_positive_frames = sum(
        1 for row in frame_rows if float(row["tx_bytes"]) > 0
    )
    total_collaborators = sum(
        int(row["collaborator_count"]) for row in frame_rows
    )

    unique_links = set()
    unique_state_links = set()
    state_tx = Counter()
    state_links = Counter()

    for row in link_rows:
        frame = json.dumps(json_safe(row.get("frame_id")), sort_keys=True)
        link = json.dumps(json_safe(row.get("link_key")), sort_keys=True)
        if row.get("link_key") is None:
            link = "record:{}".format(row.get("record_index"))
        unique_links.add((frame, link))
        state = str(row.get("state") or "unknown")
        unique_state_links.add((frame, link, state))
        state_tx[state] += float(row.get("tx_bytes", 0.0))

    for _, _, state in unique_state_links:
        state_links[state] += 1

    summary = {
        "name": name,
        "dataset": dataset_name,
        "baseline": baseline,
        "model_dir": str(model_dir),
        "checkpoint_epoch": int(epoch),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "requested_max_frames": int(max_frames),
        "evaluated_frame_count": int(frame_count),
        "communication_frame_count": int(communication_frames),
        "tx_positive_frame_count": int(tx_positive_frames),
        "total_collaborator_opportunities": int(total_collaborators),
        "unique_transmitted_link_count": int(len(unique_links)),
        "total_tx_bytes": int(round(total_tx)),
        "total_tx_MB": total_tx / MB,
        "total_tx_MiB": total_tx / MIB,
        "avg_total_tx_MB_per_frame": (
            total_tx / MB / frame_count if frame_count else None
        ),
        "avg_total_tx_MiB_per_frame": (
            total_tx / MIB / frame_count if frame_count else None
        ),
        "avg_tx_MB_per_communication_frame": (
            total_tx / MB / communication_frames
            if communication_frames else None
        ),
        "avg_tx_MB_per_collaborator_opportunity": (
            total_tx / MB / total_collaborators
            if total_collaborators else None
        ),
        "avg_tx_MB_per_unique_link_record": (
            total_tx / MB / len(unique_links)
            if unique_links else None
        ),
        "total_rx_valid_MB": total_rx / MB,
        "total_source_before_budget_MB": total_source / MB,
        "total_budget_truncated_MB": total_truncated / MB,
        "state_link_counts": dict(state_links),
        "state_tx_MB": {
            key: value / MB for key, value in state_tx.items()
        },
        "missing_record_frame_count": int(
            diagnostics["missing_record_frame_count"]
        ),
        "record_extraction_sources": dict(
            diagnostics["record_extraction_sources"]
        ),
        "policy_audit": policy_audit,
        "seed": int(seed),
        "elapsed_seconds": float(elapsed_seconds),
        "definition": (
            "actual bytes entering Markov channel before random packet loss; "
            "all sender->ego links and native scales/segments are summed per "
            "perception frame; zero-collaborator frames remain in denominator"
        ),
    }

    denominator = max(1, communication_frames)
    missing_ratio = (
        diagnostics["missing_record_frame_count"] / float(denominator)
    )
    summary["record_extraction_pass"] = bool(missing_ratio <= 0.05)
    summary["pass"] = bool(
        frame_count > 0
        and summary["record_extraction_pass"]
        and not policy_audit.get("policy_detected", False)
    )
    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description="Audit per-frame sender-to-ego wire bytes for ideal or Markov channels."
    )
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--dataset", required=True, choices=["opv2v", "v2xreal"]
    )
    parser.add_argument(
        "--baseline",
        required=True,
        choices=[
            "where2comm",
            "v2xvit",
            "rocooper",
            "coopdiff",
            "cosdh",
        ],
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument(
        "--channel_mode", required=True, choices=["ideal", "markov"],
        help="ideal uses read-only native-boundary hooks; markov reads runtime transport records.",
    )
    parser.add_argument("--packet_size_bytes", type=int, default=1024)
    parser.add_argument(
        "--bytes_per_value", type=int, default=0,
        help="Ideal audit only. 0 uses tensor.element_size().",
    )
    parser.add_argument(
        "--sparse_metadata", choices=["none", "indices", "bitmask"],
        default="indices",
        help="Ideal Where2Comm/CoSDH sparse-position metadata model.",
    )
    parser.add_argument("--sparse_index_bytes", type=int, default=4)
    parser.add_argument(
        "--separate_segment_packets", action="store_true",
        help="Packetize every native scale/segment separately instead of one joint stream per link.",
    )
    parser.add_argument("--epoch", default="auto")
    parser.add_argument("--max_frames", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test_dir", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--allow_policy",
        action="store_true",
        help="Allow C2MAB/DC2MAB configs. Default rejects them.",
    )
    parser.add_argument(
        "--continue_on_frame_error",
        action="store_true",
        help="Record frame errors and continue. Failed frames are excluded.",
    )
    parser.add_argument("--progress_interval", type=int, default=20)
    return parser


def main():
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.model_dir).expanduser().resolve()
    config_path = model_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    set_seed(args.seed)

    hypes = load_hypes(model_dir)
    patch_test_dir(hypes, args.test_dir)
    if args.channel_mode == "markov":
        recursively_override_markov_seed(hypes, args.seed)

    policy_audit = audit_policy_config(hypes, args.baseline)
    policy_audit["channel_mode"] = args.channel_mode
    write_json(out_dir / "policy_audit.json", policy_audit)
    if policy_audit["policy_detected"] and not args.allow_policy:
        raise RuntimeError(
            "Policy-driven config detected. Use a fixed/no-policy model "
            "config, or pass --allow_policy intentionally."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset, loader = build_loader(
        hypes, args.baseline, args.num_workers
    )
    loaded_epoch, checkpoint, model = load_model_with_exact_checkpoint(
        hypes=hypes,
        model_dir=model_dir,
        epoch=args.epoch,
        out_dir=out_dir,
        device=device,
    )

    ideal_auditor = None
    if args.channel_mode == "ideal":
        ideal_auditor = IdealWireAuditor(
            model=model, baseline=args.baseline,
            packet_size_bytes=args.packet_size_bytes,
            bytes_per_value=(args.bytes_per_value if args.bytes_per_value > 0 else None),
            sparse_metadata=args.sparse_metadata,
            sparse_index_bytes=args.sparse_index_bytes,
            joint_link_stream=not args.separate_segment_packets,
        )
        sources = []
        offsets = {}
        source_manifest = [{
            "class": ideal_auditor.__class__.__name__,
            "kind": "read_only_forward_hooks",
            "packet_size_bytes": int(args.packet_size_bytes),
            "sparse_metadata": args.sparse_metadata,
            "joint_link_stream": not args.separate_segment_packets,
        }]
    else:
        sources = discover_record_sources(model)
        reset_record_sources(sources)
        offsets = record_offsets(sources)
        source_manifest = []
        for obj, kind in sources:
            source_manifest.append({
                "class": obj.__class__.__name__,
                "kind": kind,
            })
    write_json(out_dir / "record_sources.json", source_manifest)

    frame_rows = []
    normalized_link_rows = []
    frame_errors = []
    diagnostics = {
        "missing_record_frame_count": 0,
        "record_extraction_sources": Counter(),
    }

    total = len(loader)
    if int(args.max_frames) > 0:
        total = min(total, int(args.max_frames))

    start_time = time.time()
    progress = tqdm(enumerate(loader), total=total)

    with torch.no_grad():
        for sample_index, batch_data in progress:
            if int(args.max_frames) > 0 and sample_index >= int(args.max_frames):
                break

            frame_id = frame_id_from_batch(batch_data, sample_index)
            collabs = collaborator_count(batch_data)

            try:
                batch_data = train_utils.to_device(batch_data, device)
                if ideal_auditor is not None:
                    ideal_auditor.start_frame()
                output = run_one_frame(
                    batch_data=batch_data,
                    model=model,
                    dataset=dataset,
                    baseline=args.baseline,
                )

                if ideal_auditor is not None:
                    records = ideal_auditor.finish_frame(
                        frame_id=frame_id, sample_index=sample_index
                    )
                    diagnostics["record_extraction_sources"]["ideal_native_hooks"] += len(records)
                else:
                    new_raw = collect_new_raw_records(sources, offsets)
                    records = []
                    if new_raw:
                        for raw in new_raw:
                            normalized = normalize_record(
                                raw, fallback_frame_id=frame_id,
                                sample_index=sample_index, source_name="record_source",
                            )
                            if normalized is not None:
                                records.append(normalized)
                        diagnostics["record_extraction_sources"]["record_source"] += len(records)

                    if not records:
                        for structure_name, structure in latest_model_structures(model, output):
                            extracted = extract_records_from_structure(
                                structure, fallback_frame_id=frame_id,
                                sample_index=sample_index, source_name=structure_name,
                            )
                            if extracted:
                                diagnostics["record_extraction_sources"][structure_name] += len(extracted)
                                records.extend(extracted)

                records = deduplicate_records(records)

                if collabs > 0 and not records:
                    diagnostics["missing_record_frame_count"] += 1

                frame_tx = sum(float(row["tx_bytes"]) for row in records)
                frame_rx = sum(
                    float(row["rx_bytes"])
                    for row in records
                    if row.get("rx_bytes") is not None
                )
                frame_source = sum(
                    float(row["source_bytes"])
                    for row in records
                    if row.get("source_bytes") is not None
                )
                frame_truncated = sum(
                    float(row["budget_truncated_bytes"])
                    for row in records
                    if row.get("budget_truncated_bytes") is not None
                )

                state_counter = Counter(
                    str(row.get("state") or "unknown")
                    for row in records
                )

                frame_rows.append({
                    "sample_index": int(sample_index),
                    "frame_id": frame_id,
                    "collaborator_count": int(collabs),
                    "record_count": len(records),
                    "tx_bytes": int(round(frame_tx)),
                    "tx_MB": frame_tx / MB,
                    "rx_bytes": int(round(frame_rx)),
                    "rx_MB": frame_rx / MB,
                    "source_bytes": int(round(frame_source)),
                    "source_MB": frame_source / MB,
                    "budget_truncated_bytes": int(round(frame_truncated)),
                    "budget_truncated_MB": frame_truncated / MB,
                    "states": dict(state_counter),
                    "record_extraction_missing": bool(collabs > 0 and not records),
                })

                for record_index, row in enumerate(records):
                    item = dict(row)
                    item["record_index"] = int(record_index)
                    normalized_link_rows.append(item)

                if (
                    int(args.progress_interval) > 0
                    and (sample_index + 1) % int(args.progress_interval) == 0
                ):
                    progress.set_description(
                        "{} frames tx={:.3f}MB".format(
                            sample_index + 1,
                            sum(item["tx_bytes"] for item in frame_rows) / MB,
                        )
                    )

            except Exception as exc:
                frame_errors.append({
                    "sample_index": int(sample_index),
                    "frame_id": frame_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                if not args.continue_on_frame_error:
                    write_json(out_dir / "frame_errors.json", frame_errors)
                    raise

    elapsed = time.time() - start_time

    summary = summarize(
        name=args.name,
        dataset_name=args.dataset,
        baseline=args.baseline,
        model_dir=model_dir,
        epoch=loaded_epoch,
        checkpoint=checkpoint,
        max_frames=args.max_frames,
        seed=args.seed,
        frame_rows=frame_rows,
        link_rows=normalized_link_rows,
        diagnostics=diagnostics,
        policy_audit=policy_audit,
        elapsed_seconds=elapsed,
    )
    summary["frame_error_count"] = len(frame_errors)
    if frame_errors:
        summary["pass"] = False

    write_csv(out_dir / "per_frame_tx.csv", frame_rows)
    write_jsonl(out_dir / "per_link_records.jsonl", normalized_link_rows)
    write_json(out_dir / "frame_errors.json", frame_errors)
    summary["channel_mode"] = args.channel_mode
    summary["packet_size_bytes"] = int(args.packet_size_bytes)
    summary["sparse_metadata"] = args.sparse_metadata if args.channel_mode == "ideal" else None
    summary["definition"] = (
        "ideal: native payload at the baseline communication boundary, grouped over all "
        "sender->ego links and padded to fixed-size packets"
        if args.channel_mode == "ideal" else
        "markov: actual bytes entering the configured channel before random loss; all "
        "sender->ego links and native scales/segments are summed per frame"
    )
    write_json(out_dir / "summary.json", summary)

    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))

    if ideal_auditor is not None:
        ideal_auditor.close()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not summary["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
