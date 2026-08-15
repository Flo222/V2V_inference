"""
ARCE inference entry for OpenCOOD / OPV2V / V2X-ViT.

This script runs normal OpenCOOD inference while collecting ARCE communication
records from the model.

Main responsibilities:
    1. Load YAML / checkpoint from model_dir.
    2. Apply optional ARCE command-line overrides.
    3. Run OpenCOOD inference.
    4. Collect per-frame / per-link ARCE communication records.
    5. Save communication logs:
        - arce_comm_records.jsonl
        - arce_comm_flat.jsonl
        - arce_comm_flat.csv
        - arce_comm_summary.json
        - arce_comm_summary_full.json
    6. Save normal detection evaluation results.

Important:
    This script does not implement ARCE communication itself.
    ARCE communication should already be integrated into the model, usually in:

        opencood/models/baselines/v2xvit/point_pillar_transformer_opv2v_arce.py

    The model should contain an ARCEFixedComm-like object, commonly:
        self.arce_comm = ARCEFixedComm(...)

Typical command:

    python opencood/tools/inference_arce.py \\
      --model_dir opencood/logs/point_pillar_v2xvit_opv2v_arce_xxx \\
      --fusion_method intermediate \\
      --save_comm

Optional channel override:

    python opencood/tools/inference_arce.py \\
      --model_dir opencood/logs/xxx \\
      --fusion_method intermediate \\
      --save_comm \\
      --arce_channel_state medium
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# ----------------------------------------------------------------------
# OpenCOOD imports
# ----------------------------------------------------------------------

from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils
from opencood.tools import inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils


# Visualization imports are optional in some OpenCOOD versions.
try:
    from opencood.visualization import simple_vis
except Exception:
    simple_vis = None


# ARCE metric logger.
try:
    from opencood.communication.metrics.comm_logger import CommLogger
    from opencood.communication.metrics.comm_stats import CommStats
except Exception:
    CommLogger = None
    CommStats = None


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """
    Build command-line parser.
    """
    parser = argparse.ArgumentParser(
        description="OpenCOOD V2X-ViT ARCE inference with communication logging."
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing config.yaml and trained checkpoint.",
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
        help="OpenCOOD inference fusion method.",
    )
    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="Save visualization images.",
    )
    parser.add_argument(
        "--save_npy",
        action="store_true",
        help="Save prediction / GT numpy files if supported by OpenCOOD.",
    )
    parser.add_argument(
        "--vis_interval",
        type=int,
        default=40,
        help="Visualization interval.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Maximum number of test samples. -1 means all.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for inference script.",
    )

    # ARCE-specific logging options.
    parser.add_argument(
        "--save_comm",
        action="store_true",
        help="Save ARCE communication logs.",
    )
    parser.add_argument(
        "--comm_log_dir",
        type=str,
        default=None,
        help=(
            "Directory for ARCE communication logs. "
            "Default: <model_dir>/arce_comm_logs"
        ),
    )
    parser.add_argument(
        "--comm_prefix",
        type=str,
        default="arce_comm",
        help="Prefix for communication log files.",
    )
    parser.add_argument(
        "--skip_bypassed_comm",
        action="store_true",
        help="Skip bypassed ego / disabled records in communication summaries.",
    )
    parser.add_argument(
        "--flatten_raw_for_csv",
        action="store_true",
        help="Also flatten raw nested ARCE records into CSV.",
    )
    parser.add_argument(
        "--comm_flush_every",
        type=int,
        default=0,
        help="Flush communication CSV / summary every N records. 0 means final only.",
    )

    # ARCE runtime overrides.
    parser.add_argument(
        "--arce_channel_state",
        type=str,
        default=None,
        choices=["good", "medium", "bad"],
        help="Override fixed ARCE channel state for all links.",
    )
    parser.add_argument(
        "--arce_late_policy",
        type=str,
        default=None,
        choices=["allow", "drop", "cache_only"],
        help="Override ARCE late-message policy.",
    )
    parser.add_argument(
        "--arce_enabled",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Override arce.enabled.",
    )
    parser.add_argument(
        "--arce_mode",
        type=str,
        default=None,
        choices=["fixed", "bypass", "disabled"],
        help="Override arce.mode.",
    )

    # Output control.
    parser.add_argument(
        "--save_eval_json",
        action="store_true",
        help="Save detection result_stat as JSON-friendly file.",
    )

    return parser


# ----------------------------------------------------------------------
# Basic utilities
# ----------------------------------------------------------------------

def set_random_seed(seed: int) -> None:
    """
    Set reproducible inference seeds.

    Besides seeding random generators, disable CuDNN autotuning so repeated
    Experiment-3 conditions produce the same model feature and compact payload.
    """
    import random
    import numpy as np

    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # Older PyTorch versions do not support warn_only.
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass


def str_to_bool(value: Optional[str]) -> Optional[bool]:
    """
    Convert string to bool.
    """
    if value is None:
        return None

    value = str(value).strip().lower()

    if value in ("true", "1", "yes", "y", "on"):
        return True

    if value in ("false", "0", "no", "n", "off"):
        return False

    raise ValueError(f"Cannot convert to bool: {value}")


def json_safe(value: Any) -> Any:
    """
    Convert values to JSON-friendly format.
    """
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if torch.is_tensor(value):
        if value.numel() == 1:
            return json_safe(value.detach().cpu().item())
        return json_safe(value.detach().cpu().tolist())

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    return str(value)


def ensure_dir(path: str) -> str:
    """
    Create directory if missing.
    """
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path: str, data: Any, indent: int = 2) -> None:
    """
    Save JSON file.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, indent=indent, ensure_ascii=False)


# ----------------------------------------------------------------------
# YAML / config helpers
# ----------------------------------------------------------------------

def load_hypes_from_model_dir(opt: argparse.Namespace) -> Dict[str, Any]:
    """
    Load OpenCOOD YAML config.

    First try OpenCOOD yaml_utils.load_yaml(None, opt), then fall back to
    <model_dir>/config.yaml.
    """
    try:
        return yaml_utils.load_yaml(None, opt)
    except Exception as exc:
        config_path = os.path.join(opt.model_dir, "config.yaml")

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Failed to load YAML with yaml_utils and config.yaml not found: "
                f"{config_path}. Original error: {exc}"
            )

        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)


def ensure_arce_cfg_visibility(hypes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure ARCE config is visible both at top-level hypes["arce"] and inside
    hypes["model"]["args"]["arce"].

    Some OpenCOOD models only receive model.args during construction, so putting
    ARCE config only at the top level may not be enough.
    """
    hypes = hypes or {}

    model_cfg = hypes.setdefault("model", {})
    model_args = model_cfg.setdefault("args", {})

    top_arce = hypes.get("arce", None)
    arg_arce = model_args.get("arce", None)

    if top_arce is None and arg_arce is None:
        hypes["arce"] = {}
        model_args["arce"] = hypes["arce"]
        return hypes

    if top_arce is None:
        hypes["arce"] = copy.deepcopy(arg_arce)
    elif arg_arce is None:
        model_args["arce"] = copy.deepcopy(top_arce)
    else:
        merged = copy.deepcopy(top_arce)
        merged.update(copy.deepcopy(arg_arce))
        hypes["arce"] = copy.deepcopy(merged)
        model_args["arce"] = copy.deepcopy(merged)

    return hypes


def apply_arce_cli_overrides(
    hypes: Dict[str, Any],
    opt: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Apply command-line ARCE overrides into YAML hypes.
    """
    hypes = ensure_arce_cfg_visibility(hypes)

    arce_cfg = hypes.setdefault("arce", {})

    enabled_override = str_to_bool(opt.arce_enabled)
    if enabled_override is not None:
        arce_cfg["enabled"] = enabled_override

    if opt.arce_mode is not None:
        arce_cfg["mode"] = opt.arce_mode

    if opt.arce_late_policy is not None:
        arce_cfg["late_policy"] = opt.arce_late_policy

    if opt.arce_channel_state is not None:
        channel_cfg = arce_cfg.setdefault("channel", {})
        channel_cfg["mode"] = "fixed"
        channel_cfg["fixed_state"] = opt.arce_channel_state

        fixed_policy = arce_cfg.setdefault("fixed_policy", {})
        fixed_policy["default_state"] = opt.arce_channel_state

    hypes["model"]["args"]["arce"] = copy.deepcopy(arce_cfg)

    return hypes


# ----------------------------------------------------------------------
# Model / ARCE object discovery
# ----------------------------------------------------------------------

def load_model(
    hypes: Dict[str, Any],
    model_dir: str,
    device: torch.device,
) -> torch.nn.Module:
    """
    Create model and load checkpoint.
    """
    model = train_utils.create_model(hypes)
    model.to(device)

    loaded = train_utils.load_saved_model(model_dir, model)

    # OpenCOOD variants often return (init_epoch, model).
    if isinstance(loaded, tuple) and len(loaded) >= 2:
        model = loaded[1]
    elif isinstance(loaded, torch.nn.Module):
        model = loaded

    model.to(device)
    model.eval()
    return model


def is_arce_comm_like(obj: Any) -> bool:
    """
    Check whether object looks like ARCEFixedComm.
    """
    return (
        hasattr(obj, "get_records")
        and hasattr(obj, "get_summary")
        and (
            hasattr(obj, "communicate_feature")
            or hasattr(obj, "communicate_agent_features")
        )
    )


def find_arce_modules(model: Any, max_depth: int = 4) -> List[Any]:
    """
    Find ARCEFixedComm-like objects inside a model.

    ARCEFixedComm is not an nn.Module, so named_modules() alone is not enough.
    This function also recursively inspects object attributes.
    """
    found: List[Any] = []
    visited = set()

    def add_if_arce(obj: Any) -> None:
        if obj is None:
            return

        obj_id = id(obj)

        if obj_id in visited:
            return

        visited.add(obj_id)

        if is_arce_comm_like(obj):
            found.append(obj)

    def walk(obj: Any, depth: int) -> None:
        if obj is None or depth < 0:
            return

        obj_id = id(obj)

        if obj_id in visited:
            return

        visited.add(obj_id)

        if is_arce_comm_like(obj):
            found.append(obj)
            return

        # Inspect common direct attributes first.
        for attr_name in (
            "arce_comm",
            "arce_fixed_comm",
            "comm",
            "comm_module",
            "communication",
            "communication_module",
        ):
            if hasattr(obj, attr_name):
                attr = getattr(obj, attr_name)
                if is_arce_comm_like(attr):
                    found.append(attr)

        # Inspect nn.Module submodules.
        if hasattr(obj, "named_modules"):
            try:
                for _, submodule in obj.named_modules():
                    for attr_name in (
                        "arce_comm",
                        "arce_fixed_comm",
                        "comm",
                        "comm_module",
                        "communication",
                        "communication_module",
                    ):
                        if hasattr(submodule, attr_name):
                            attr = getattr(submodule, attr_name)
                            add_if_arce(attr)
            except Exception:
                pass

        # Inspect __dict__ recursively.
        if depth > 0 and hasattr(obj, "__dict__"):
            for _, value in vars(obj).items():
                if value is obj:
                    continue

                if isinstance(value, (str, int, float, bool, bytes)):
                    continue

                if torch.is_tensor(value):
                    continue

                if isinstance(value, dict):
                    for item in value.values():
                        walk(item, depth - 1)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        walk(item, depth - 1)
                else:
                    walk(value, depth - 1)

    walk(model, max_depth)

    # Remove duplicates while preserving order.
    unique = []
    seen = set()
    for item in found:
        if id(item) not in seen:
            seen.add(id(item))
            unique.append(item)

    return unique


def reset_arce_modules(arce_modules: Sequence[Any]) -> None:
    """
    Reset ARCE modules before inference.
    """
    for module in arce_modules:
        if hasattr(module, "reset"):
            try:
                module.reset(clear_cache=True, clear_records=True)
            except TypeError:
                module.reset()


def apply_runtime_arce_overrides_to_modules(
    arce_modules: Sequence[Any],
    opt: argparse.Namespace,
) -> None:
    """
    Apply runtime overrides directly to discovered ARCE objects.
    """
    for module in arce_modules:
        if opt.arce_channel_state is not None and hasattr(module, "set_channel_state"):
            module.set_channel_state(opt.arce_channel_state)

        if opt.arce_late_policy is not None and hasattr(module, "late_policy"):
            module.late_policy = opt.arce_late_policy


def get_arce_record_offsets(arce_modules: Sequence[Any]) -> Dict[int, int]:
    """
    Return current record offsets for each ARCE module.
    """
    offsets = {}

    for module in arce_modules:
        try:
            offsets[id(module)] = len(module.get_records())
        except Exception:
            offsets[id(module)] = 0

    return offsets


def collect_new_arce_records(
    arce_modules: Sequence[Any],
    offsets: Dict[int, int],
    fallback_frame_id: Any,
    sample_index: int,
) -> List[Dict[str, Any]]:
    """
    Collect newly generated ARCE records since previous offsets.
    """
    new_records: List[Dict[str, Any]] = []

    for module in arce_modules:
        module_id = id(module)
        start = int(offsets.get(module_id, 0))

        try:
            records = module.get_records()
        except Exception:
            continue

        current = len(records)
        offsets[module_id] = current

        for record in records[start:current]:
            item = copy.deepcopy(record)

            if item.get("frame_id", None) is None:
                item["frame_id"] = fallback_frame_id

            item.setdefault("sample_index", int(sample_index))
            new_records.append(item)

    return new_records


# ----------------------------------------------------------------------
# Dataset / inference helpers
# ----------------------------------------------------------------------

def build_test_loader(
    hypes: Dict[str, Any],
    opt: argparse.Namespace,
) -> Tuple[Any, DataLoader]:
    """
    Build OpenCOOD test dataset and dataloader.
    """
    dataset = build_dataset(hypes, visualize=True, train=False)

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=int(opt.num_workers),
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    return dataset, data_loader


def to_device(batch_data: Any, device: torch.device) -> Any:
    """
    Move OpenCOOD batch data to device.
    """
    return train_utils.to_device(batch_data, device)


def run_inference_one(
    batch_data: Dict[str, Any],
    model: torch.nn.Module,
    dataset: Any,
    fusion_method: str,
):
    """
    Run one OpenCOOD inference call and return:
        pred_box_tensor, pred_score, gt_box_tensor
    """
    method = str(fusion_method).strip().lower()

    if method == "intermediate":
        func_name = "inference_intermediate_fusion"
    elif method == "late":
        func_name = "inference_late_fusion"
    elif method == "early":
        func_name = "inference_early_fusion"
    elif method in ("no", "no_fusion", "single"):
        func_name = "inference_no_fusion"
    else:
        raise ValueError(f"Unsupported fusion_method: {fusion_method}")

    if not hasattr(inference_utils, func_name):
        raise AttributeError(
            f"opencood.tools.inference_utils has no function {func_name}. "
            f"Please check your OpenCOOD version."
        )

    infer_func = getattr(inference_utils, func_name)
    output = infer_func(batch_data, model, dataset)

    if not isinstance(output, (list, tuple)) or len(output) < 3:
        raise RuntimeError(
            f"{func_name} should return at least 3 values: "
            "pred_box_tensor, pred_score, gt_box_tensor."
        )

    pred_box_tensor, pred_score, gt_box_tensor = output[:3]
    return pred_box_tensor, pred_score, gt_box_tensor


def init_result_stat(iou_thresholds: Sequence[float]) -> Dict[float, Dict[str, Any]]:
    """
    Initialize OpenCOOD result_stat dict.
    """
    result_stat = {}

    for thresh in iou_thresholds:
        result_stat[float(thresh)] = {
            "tp": [],
            "fp": [],
            "gt": 0,
            "score": [],
        }

    return result_stat


def update_detection_result_stat(
    result_stat: Dict[float, Dict[str, Any]],
    pred_box_tensor: Any,
    pred_score: Any,
    gt_box_tensor: Any,
    iou_thresholds: Sequence[float],
) -> None:
    """
    Update OpenCOOD detection evaluation statistics.

    OpenCOOD versions differ slightly:
        - eval_utils.caluclate_tp_fp(infer_result, result_stat, thresh)
        - eval_utils.calculate_tp_fp(...)
    This helper supports common variants.
    """
    infer_result = {
        "pred_box_tensor": pred_box_tensor,
        "pred_score": pred_score,
        "gt_box_tensor": gt_box_tensor,
    }

    calc_func = None

    if hasattr(eval_utils, "caluclate_tp_fp"):
        calc_func = getattr(eval_utils, "caluclate_tp_fp")
    elif hasattr(eval_utils, "calculate_tp_fp"):
        calc_func = getattr(eval_utils, "calculate_tp_fp")

    if calc_func is None:
        raise AttributeError(
            "eval_utils has neither caluclate_tp_fp nor calculate_tp_fp."
        )

    for thresh in iou_thresholds:
        thresh = float(thresh)

        try:
            calc_func(infer_result, result_stat, thresh)
        except TypeError:
            calc_func(
                pred_box_tensor,
                pred_score,
                gt_box_tensor,
                result_stat,
                thresh,
            )


def finalize_detection_eval(
    result_stat: Dict[float, Dict[str, Any]],
    model_dir: str,
) -> Any:
    """
    Finalize OpenCOOD detection evaluation.
    """
    if not hasattr(eval_utils, "eval_final_results"):
        raise AttributeError("eval_utils has no eval_final_results().")

    try:
        return eval_utils.eval_final_results(
            result_stat,
            model_dir,
            "global_sort",
        )
    except TypeError:
        return eval_utils.eval_final_results(
            result_stat,
            model_dir,
        )


def resolve_frame_id(batch_data: Dict[str, Any], sample_index: int) -> Any:
    """
    Resolve the dataset frame identifier used by communication audit.

    Priority:
        frame_id -> timestamp -> sample_idx -> sample_id -> loader index.

    The loader index is only a deterministic fallback when the dataset does not
    expose a native identifier.
    """
    try:
        ego_data = batch_data.get("ego", {})
        for key in ("frame_id", "timestamp", "sample_idx", "sample_id"):
            if key in ego_data:
                value = json_safe(ego_data[key])
                # OpenCOOD uses batch size one for inference. Normalize a
                # one-element list/tuple so filenames and JSON records contain
                # the actual scalar id instead of ``[id]``.
                if isinstance(value, list) and len(value) == 1:
                    return value[0]
                return value
    except Exception:
        pass

    return int(sample_index)


def attach_frame_identity_to_batch(
    batch_data: Dict[str, Any],
    frame_id: Any,
    sample_index: int,
) -> None:
    """Expose the resolved id to the model without changing model outputs.

    ``inference_arce`` historically added a fallback id only after the model
    forward, so ARCE's internal audit record still contained ``frame_id=None``.
    This helper writes metadata keys into the ego data dictionary before the
    forward call. Model layers ignore these keys; communication modules can read
    them for deterministic per-frame logging.
    """
    if not isinstance(batch_data, dict):
        return
    ego_data = batch_data.get("ego", None)
    if not isinstance(ego_data, dict):
        return
    ego_data["frame_id"] = frame_id
    ego_data["audit_sample_index"] = int(sample_index)


def maybe_save_prediction_npy(
    opt: argparse.Namespace,
    batch_data: Dict[str, Any],
    pred_box_tensor: Any,
    pred_score: Any,
    gt_box_tensor: Any,
    sample_index: int,
) -> None:
    """
    Save prediction / GT npy files if supported by OpenCOOD.
    """
    if not opt.save_npy:
        return

    npy_dir = ensure_dir(os.path.join(opt.model_dir, "npy"))

    if hasattr(inference_utils, "save_prediction_gt"):
        try:
            origin_lidar = batch_data["ego"]["origin_lidar"][0]
            inference_utils.save_prediction_gt(
                pred_box_tensor,
                gt_box_tensor,
                origin_lidar,
                sample_index,
                npy_dir,
            )
        except Exception as exc:
            print(f"[WARN] Failed to save npy for sample {sample_index}: {exc}")


def maybe_save_visualization(
    opt: argparse.Namespace,
    hypes: Dict[str, Any],
    batch_data: Dict[str, Any],
    pred_box_tensor: Any,
    gt_box_tensor: Any,
    sample_index: int,
) -> None:
    """
    Save simple visualization if supported.
    """
    if not opt.save_vis:
        return

    if simple_vis is None:
        print("[WARN] simple_vis is unavailable; skip visualization.")
        return

    if opt.vis_interval <= 0:
        return

    if sample_index % int(opt.vis_interval) != 0:
        return

    vis_dir = ensure_dir(os.path.join(opt.model_dir, "vis_arce"))
    vis_path = os.path.join(vis_dir, f"{sample_index:05d}.png")

    try:
        origin_lidar = batch_data["ego"]["origin_lidar"][0]
        gt_range = hypes["postprocess"]["gt_range"]

        simple_vis.visualize(
            pred_box_tensor,
            gt_box_tensor,
            origin_lidar,
            gt_range,
            vis_path,
            method="3d",
            left_hand=False,
        )
    except Exception as exc:
        print(f"[WARN] Failed to save visualization for sample {sample_index}: {exc}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(opt: argparse.Namespace) -> None:
    """
    Main ARCE inference function.
    """
    set_random_seed(opt.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[ARCE Inference] model_dir = {opt.model_dir}")
    print(f"[ARCE Inference] device = {device}")

    hypes = load_hypes_from_model_dir(opt)
    hypes = apply_arce_cli_overrides(hypes, opt)

    dataset, data_loader = build_test_loader(hypes, opt)
    model = load_model(hypes, opt.model_dir, device)

    arce_modules = find_arce_modules(model)
    reset_arce_modules(arce_modules)
    apply_runtime_arce_overrides_to_modules(arce_modules, opt)

    if len(arce_modules) == 0:
        print(
            "[WARN] No ARCEFixedComm-like module found in model. "
            "Inference will run, but communication records will be empty. "
            "Check whether point_pillar_transformer_opv2v_arce.py creates "
            "self.arce_comm = ARCEFixedComm(...)."
        )
    else:
        print(f"[ARCE Inference] Found {len(arce_modules)} ARCE communication module(s).")
        for idx, module in enumerate(arce_modules):
            print(f"  - ARCE module {idx}: {module}")

    comm_log_dir = opt.comm_log_dir
    if comm_log_dir is None:
        comm_log_dir = os.path.join(opt.model_dir, "arce_comm_logs")
    ensure_dir(comm_log_dir)

    logger = None
    if opt.save_comm:
        if CommLogger is None:
            print(
                "[WARN] CommLogger import failed. "
                "Communication records will be kept in memory only."
            )
        else:
            logger = CommLogger(
                log_dir=comm_log_dir,
                prefix=opt.comm_prefix,
                reset_on_init=True,
                keep_in_memory=True,
                write_raw_jsonl=True,
                write_flat_jsonl=True,
                write_csv=True,
                write_summary=True,
                skip_bypassed=opt.skip_bypassed_comm,
                flatten_raw_for_csv=opt.flatten_raw_for_csv,
                flush_every=opt.comm_flush_every,
            )

    iou_thresholds = [0.3, 0.5, 0.7]
    result_stat = init_result_stat(iou_thresholds)

    all_comm_records: List[Dict[str, Any]] = []
    arce_record_offsets = get_arce_record_offsets(arce_modules)

    total_samples = len(data_loader)
    if opt.max_samples is not None and int(opt.max_samples) > 0:
        total_samples = min(total_samples, int(opt.max_samples))

    start_time = time.time()

    with torch.no_grad():
        pbar = tqdm(enumerate(data_loader), total=total_samples)

        for sample_index, batch_data in pbar:
            if opt.max_samples is not None and int(opt.max_samples) > 0:
                if sample_index >= int(opt.max_samples):
                    break

            batch_data = to_device(batch_data, device)
            frame_id = resolve_frame_id(batch_data, sample_index)
            attach_frame_identity_to_batch(
                batch_data=batch_data,
                frame_id=frame_id,
                sample_index=sample_index,
            )

            pred_box_tensor, pred_score, gt_box_tensor = run_inference_one(
                batch_data=batch_data,
                model=model,
                dataset=dataset,
                fusion_method=opt.fusion_method,
            )

            update_detection_result_stat(
                result_stat=result_stat,
                pred_box_tensor=pred_box_tensor,
                pred_score=pred_score,
                gt_box_tensor=gt_box_tensor,
                iou_thresholds=iou_thresholds,
            )

            maybe_save_prediction_npy(
                opt=opt,
                batch_data=batch_data,
                pred_box_tensor=pred_box_tensor,
                pred_score=pred_score,
                gt_box_tensor=gt_box_tensor,
                sample_index=sample_index,
            )

            maybe_save_visualization(
                opt=opt,
                hypes=hypes,
                batch_data=batch_data,
                pred_box_tensor=pred_box_tensor,
                gt_box_tensor=gt_box_tensor,
                sample_index=sample_index,
            )

            new_records = collect_new_arce_records(
                arce_modules=arce_modules,
                offsets=arce_record_offsets,
                fallback_frame_id=frame_id,
                sample_index=sample_index,
            )

            if new_records:
                all_comm_records.extend(new_records)

                if logger is not None:
                    logger.log_records(new_records)

            pbar.set_description(
                f"sample={sample_index} comm_records={len(all_comm_records)}"
            )

    elapsed = time.time() - start_time

    print(f"[ARCE Inference] Finished inference in {elapsed:.2f}s")

    # Detection evaluation.
    eval_output = finalize_detection_eval(result_stat, opt.model_dir)

    if opt.save_eval_json:
        eval_json_path = os.path.join(opt.model_dir, "arce_inference_result_stat.json")
        save_json(eval_json_path, result_stat)
        print(f"[ARCE Inference] Saved detection result_stat to {eval_json_path}")

    # Communication logging / summary.
    if logger is not None:
        outputs = logger.finalize()
        logger.print_summary()
        print("[ARCE Inference] Communication logs saved:")
        for key, value in outputs.get("paths", {}).items():
            print(f"  - {key}: {value}")

    elif opt.save_comm and all_comm_records:
        # Fallback if CommLogger is unavailable.
        raw_path = os.path.join(comm_log_dir, f"{opt.comm_prefix}_records_fallback.json")
        save_json(raw_path, all_comm_records)
        print(f"[ARCE Inference] Saved fallback communication records to {raw_path}")

    if CommStats is not None and all_comm_records:
        stats = CommStats(
            records=all_comm_records,
            skip_bypassed=opt.skip_bypassed_comm,
            keep_records=True,
            keep_flat_metrics=True,
        )
        compact = stats.get_compact_summary()
        print("[ARCE Inference] Compact communication summary:")
        print(json.dumps(json_safe(compact), indent=2, ensure_ascii=False))

    if len(all_comm_records) == 0:
        print(
            "[WARN] No communication records were collected. "
            "This usually means the model did not call ARCEFixedComm during forward."
        )

    print("[ARCE Inference] Detection eval output:")
    print(eval_output)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)