# -*- coding: utf-8 -*-
"""
Inference entry for CoopDiff + Markov channel.

This is a normal OpenCOOD inference script plus:
- optional CLI overrides for coopdiff_markov config;
- communication records saved as jsonl/csv/json summary;
- model.update_epoch(init_epoch) so the checkpoint epoch controls CoopDiff phase.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("CoopDiff Markov inference")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--fusion_method", type=str, default="intermediate", choices=["late", "early", "intermediate"])
    parser.add_argument("--save_npy", action="store_true")
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--global_sort_detections", action="store_true")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=-1,
                        help="Exact net_epochN.pth to load; negative means latest checkpoint.")

    parser.add_argument("--save_comm", action="store_true")
    parser.add_argument("--comm_log_dir", type=str, default=None)
    parser.add_argument("--comm_prefix", type=str, default="coopdiff_markov_comm")

    parser.add_argument("--markov_enabled", type=str, default=None, choices=["true", "false"])
    parser.add_argument("--markov_fixed_state", type=str, default=None, choices=["good", "medium", "bad"])
    parser.add_argument("--markov_active_scales", type=str, default=None,
                        help="Comma-separated scale indices, e.g. '2'. Empty means all scales.")
    parser.add_argument("--markov_verbose", action="store_true")
    parser.add_argument("--packet_size_bytes", type=int, default=None)
    parser.add_argument("--bytes_per_value", type=int, default=None)
    parser.add_argument("--selection_policy", type=str, default=None,
                        choices=["raster", "magnitude"])
    parser.add_argument("--send_nonzero_only", type=str, default=None,
                        choices=["true", "false"])
    return parser


def str_to_bool(v: Optional[str]) -> Optional[bool]:
    if v is None:
        return None
    v = str(v).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError("Cannot parse bool: {}".format(v))


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if torch.is_tensor(value):
        if value.numel() == 1:
            return json_safe(value.detach().cpu().item())
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(x) for x in value]
    return str(value)


def save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, indent=2, ensure_ascii=False)


def load_hypes(opt: argparse.Namespace) -> Dict[str, Any]:
    try:
        return yaml_utils.load_yaml(None, opt)
    except Exception:
        import yaml
        cfg_path = os.path.join(opt.model_dir, "config.yaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.load(f, Loader=yaml.Loader)


def ensure_markov_cfg_visibility(hypes: Dict[str, Any]) -> Dict[str, Any]:
    model_args = hypes.setdefault("model", {}).setdefault("args", {})
    top_cfg = hypes.get("coopdiff_markov", None)
    arg_cfg = model_args.get("coopdiff_markov", None)
    if top_cfg is None and arg_cfg is None:
        hypes["coopdiff_markov"] = {"enabled": False}
        model_args["coopdiff_markov"] = copy.deepcopy(hypes["coopdiff_markov"])
    elif top_cfg is None:
        hypes["coopdiff_markov"] = copy.deepcopy(arg_cfg)
    elif arg_cfg is None:
        model_args["coopdiff_markov"] = copy.deepcopy(top_cfg)
    else:
        merged = copy.deepcopy(top_cfg)
        merged.update(copy.deepcopy(arg_cfg))
        hypes["coopdiff_markov"] = copy.deepcopy(merged)
        model_args["coopdiff_markov"] = copy.deepcopy(merged)
    return hypes


def ensure_markov_model_core(hypes: Dict[str, Any]) -> Dict[str, Any]:
    """Allow vanilla CoopDiff checkpoints/configs to use the parameter-free wrapper."""
    model_cfg = hypes.setdefault("model", {})
    core = str(model_cfg.get("core_method", ""))
    mapping = {
        "point_pillar_diff_stu": "point_pillar_diff_stu_markov",
        "point_pillar_diff_stu_v2xreal": "point_pillar_diff_stu_markov_v2xreal",
    }
    if core in mapping:
        model_cfg["core_method"] = mapping[core]
        print("CoopDiff model wrapper: {} -> {}".format(core, mapping[core]))
    return hypes


def load_checkpoint(model_dir: str, model: Any, epoch: int):
    if int(epoch) < 0:
        return train_utils.load_saved_model(model_dir, model)
    checkpoint_path = os.path.join(model_dir, "net_epoch{}.pth".format(int(epoch)))
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))
    print("resuming by loading epoch {}".format(int(epoch)))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    del checkpoint
    return int(epoch), model


def apply_cli_overrides(hypes: Dict[str, Any], opt: argparse.Namespace) -> Dict[str, Any]:
    hypes = ensure_markov_cfg_visibility(hypes)
    cfg = hypes["coopdiff_markov"]
    enabled = str_to_bool(opt.markov_enabled)
    if enabled is not None:
        cfg["enabled"] = enabled
    if opt.markov_fixed_state is not None:
        st = opt.markov_fixed_state
        cfg["initial_state"] = st
        cfg["transition_matrix"] = {
            "good": {"good": 1.0, "medium": 0.0, "bad": 0.0},
            "medium": {"good": 0.0, "medium": 1.0, "bad": 0.0},
            "bad": {"good": 0.0, "medium": 0.0, "bad": 1.0},
        }
        # Force all current states to selected fixed state by setting initial and deterministic transitions.
        cfg["initial_state"] = st
    if opt.markov_active_scales is not None:
        raw = opt.markov_active_scales.strip()
        cfg["active_scales"] = None if raw == "" else [int(x) for x in raw.split(",")]
    if opt.markov_verbose:
        cfg["verbose"] = True

    packet_cfg = cfg.setdefault("packetization", {})
    if opt.packet_size_bytes is not None:
        packet_cfg["packet_size_bytes"] = int(opt.packet_size_bytes)
    if opt.bytes_per_value is not None:
        packet_cfg["bytes_per_value"] = int(opt.bytes_per_value)
    if opt.selection_policy is not None:
        packet_cfg["selection_policy"] = str(opt.selection_policy)
    nonzero_only = str_to_bool(opt.send_nonzero_only)
    if nonzero_only is not None:
        packet_cfg["send_nonzero_only"] = nonzero_only
    packet_cfg.setdefault("serialization_order", "cell_major")
    packet_cfg.setdefault("zero_fill_missing", True)

    hypes["model"]["args"]["coopdiff_markov"] = copy.deepcopy(cfg)
    return hypes


def find_markov_modules(model: Any) -> List[Any]:
    found = []
    seen = set()

    def maybe_add(obj: Any):
        if obj is None:
            return
        if id(obj) in seen:
            return
        if hasattr(obj, "get_records") and hasattr(obj, "get_summary"):
            found.append(obj)
            seen.add(id(obj))

    if hasattr(model, "coopdiff_markov_channel"):
        maybe_add(getattr(model, "coopdiff_markov_channel"))
    if hasattr(model, "named_modules"):
        for _, m in model.named_modules():
            if hasattr(m, "coopdiff_markov_channel"):
                maybe_add(getattr(m, "coopdiff_markov_channel"))
    return found


def get_offsets(modules: Sequence[Any]) -> Dict[int, int]:
    out = {}
    for m in modules:
        try:
            out[id(m)] = len(m.get_records())
        except Exception:
            out[id(m)] = 0
    return out


def collect_new_records(modules: Sequence[Any], offsets: Dict[int, int], sample_index: int) -> List[Dict[str, Any]]:
    rows = []
    for m in modules:
        try:
            recs = m.get_records()
        except Exception:
            continue
        start = offsets.get(id(m), 0)
        offsets[id(m)] = len(recs)
        for r in recs[start:]:
            item = copy.deepcopy(r)
            item.setdefault("sample_index", int(sample_index))
            rows.append(item)
    return rows


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(json_safe(r), ensure_ascii=False) + "\n")


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    old_fields = []
    if exists and os.path.getsize(path) > 0:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            old_fields = next(reader, [])
    if old_fields and set(fieldnames).issubset(set(old_fields)):
        fieldnames = old_fields
    elif exists and old_fields:
        # Schema expanded. Rewrite simple CSV to avoid malformed columns.
        existing_rows = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
        fieldnames = sorted(set(old_fields) | set(fieldnames))
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in existing_rows:
                writer.writerow(r)
        exists = True
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists or os.path.getsize(path) == 0:
            writer.writeheader()
        for r in rows:
            writer.writerow({k: json_safe(r.get(k, "")) for k in fieldnames})


def main() -> None:
    opt = build_parser().parse_args()
    torch.manual_seed(int(opt.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(opt.seed))

    hypes = load_hypes(opt)
    hypes = apply_cli_overrides(hypes, opt)
    hypes = ensure_markov_model_core(hypes)

    print("CoopDiff packet channel config:")
    print(json.dumps(json_safe(hypes.get("coopdiff_markov", {})), indent=2, ensure_ascii=False))

    print("Dataset Building")
    dataset = build_dataset(hypes, visualize=True, train=False)
    print("{} samples found.".format(len(dataset)))
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=int(opt.num_workers),
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    print("Creating Model")
    model = train_utils.create_model(hypes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Loading Model from checkpoint")
    init_epoch, model = load_checkpoint(opt.model_dir, model, opt.epoch)
    if hasattr(model, "update_epoch"):
        model.update_epoch(init_epoch)
    model.to(device)
    model.eval()

    markov_modules = find_markov_modules(model)
    if markov_modules:
        for m in markov_modules:
            if hasattr(m, "reset"):
                m.reset(clear_cache=True, clear_records=True)
        print("Found {} CoopDiff-Markov module(s).".format(len(markov_modules)))
    else:
        print("WARNING: no CoopDiff-Markov module found in model.")
    offsets = get_offsets(markov_modules)

    comm_dir = opt.comm_log_dir or os.path.join(opt.model_dir, "coopdiff_markov_logs")
    if opt.save_comm:
        os.makedirs(comm_dir, exist_ok=True)
        # Start fresh for this run.
        for suffix in ["jsonl", "csv"]:
            path = os.path.join(comm_dir, opt.comm_prefix + "." + suffix)
            if os.path.exists(path):
                os.remove(path)

    result_stat = {
        0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
    }

    for i, batch_data in tqdm(enumerate(loader), total=len(loader)):
        if opt.max_samples is not None and int(opt.max_samples) >= 0 and i >= int(opt.max_samples):
            break
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            if opt.fusion_method == "late":
                pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_late_fusion(
                    batch_data, model, dataset
                )
            elif opt.fusion_method == "early":
                pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_early_fusion(
                    batch_data, model, dataset
                )
            else:
                pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_intermediate_fusion(
                    batch_data, model, dataset
                )

            for thr in [0.3, 0.5, 0.7]:
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, thr)

            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, "npy")
                os.makedirs(npy_save_path, exist_ok=True)
                inference_utils.save_prediction_gt(
                    pred_box_tensor,
                    gt_box_tensor,
                    batch_data["ego"]["origin_lidar"][0],
                    i,
                    npy_save_path,
                )

            if opt.save_vis:
                vis_save_path = os.path.join(opt.model_dir, "vis")
                os.makedirs(vis_save_path, exist_ok=True)
                dataset.visualize_result(
                    pred_box_tensor,
                    gt_box_tensor,
                    batch_data["ego"]["origin_lidar"],
                    False,
                    os.path.join(vis_save_path, "%05d.png" % i),
                    dataset=dataset,
                )

        if opt.save_comm:
            rows = collect_new_records(markov_modules, offsets, sample_index=i)
            write_jsonl(os.path.join(comm_dir, opt.comm_prefix + ".jsonl"), rows)
            write_csv(os.path.join(comm_dir, opt.comm_prefix + ".csv"), rows)

    eval_utils.eval_final_results(result_stat, opt.model_dir, opt.global_sort_detections)

    if opt.save_comm:
        summaries = []
        for idx, m in enumerate(markov_modules):
            try:
                item = m.get_summary()
                item["module_index"] = idx
                summaries.append(item)
            except Exception as exc:
                summaries.append({"module_index": idx, "error": str(exc)})
        save_json(os.path.join(comm_dir, opt.comm_prefix + "_summary.json"), summaries)
        print("Saved CoopDiff-Markov communication logs to", comm_dir)


if __name__ == "__main__":
    main()
