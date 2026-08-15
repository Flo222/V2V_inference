from __future__ import print_function

import copy
import json
import math
import os
import random
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader

from opencood.hypes_yaml.yaml_utils import load_yaml


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _disable_native_channels(hypes):
    """Disable baseline-specific impairments while preserving clean feature shapes.

    This audit measures the feature that a plug-in communication module would receive,
    so native Markov/ARCE damage must not alter the sampled values first.
    """
    out = copy.deepcopy(hypes)
    args = out.get("model", {}).get("args", {})
    if not isinstance(args, dict):
        return out

    arce = args.get("arce")
    if isinstance(arce, dict):
        arce["enabled"] = False

    coopdiff = args.get("coopdiff_markov")
    if isinstance(coopdiff, dict):
        coopdiff["enabled"] = False

    # RoCooper is hooked before comm_module, but disabling its native damage also
    # keeps downstream inference deterministic and avoids irrelevant warnings.
    rocooper = args.get("rocooper_comm")
    if isinstance(rocooper, dict):
        rocooper["enabled"] = False

    for key in ("cosdh_markov", "cosdh_late_markov", "late_markov"):
        cfg = args.get(key)
        if isinstance(cfg, dict):
            cfg["enabled"] = False

    where = args.get("where2comm")
    if isinstance(where, dict):
        cfg = where.get("cosdh_markov")
        if isinstance(cfg, dict):
            cfg["enabled"] = False

    return out


def load_runtime(model_dir, num_workers=0, disable_native_channel=True):
    config_path = os.path.join(model_dir, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError("Missing config.yaml: {}".format(config_path))
    hypes = load_yaml(config_path)
    if disable_native_channel:
        hypes = _disable_native_channels(hypes)

    fusion_name = str(hypes.get("fusion", {}).get("core_method", "")).strip().lower()
    is_cosdh = fusion_name == "intermediatelate"

    if is_cosdh:
        from opencood.data_utils.datasets.cosdh_dataset_builder import build_dataset_cosdh
        from opencood.tools import train_utils_cosdh as runtime_train_utils
        dataset = build_dataset_cosdh(hypes, visualize=False, train=False)
    else:
        from opencood.data_utils.datasets import build_dataset
        from opencood.tools import train_utils as runtime_train_utils
        dataset = build_dataset(hypes, visualize=False, train=False)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=dataset.collate_batch_test,
        pin_memory=False,
        drop_last=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = runtime_train_utils.create_model(hypes).to(device)
    loaded_epoch, model = runtime_train_utils.load_saved_model(model_dir, model)
    if hasattr(model, "update_epoch"):
        model.update_epoch(999)
    model.eval()
    return hypes, dataset, loader, device, model, int(loaded_epoch)


def run_inference(fusion_method, batch, model, dataset):
    fusion_method = str(fusion_method).strip().lower()
    if fusion_method == "intermediatelate":
        from opencood.tools import inference_utils_cosdh
        return inference_utils_cosdh.inference_late_fusion(batch, model, dataset)

    from opencood.tools import inference_utils
    if fusion_method == "intermediate":
        return inference_utils.inference_intermediate_fusion(batch, model, dataset)
    if fusion_method == "late":
        return inference_utils.inference_late_fusion(batch, model, dataset)
    if fusion_method == "early":
        return inference_utils.inference_early_fusion(batch, model, dataset)
    raise ValueError("Unsupported fusion_method: {}".format(fusion_method))


def get_record_len(batch):
    candidates = []
    if isinstance(batch, dict):
        if "ego" in batch and isinstance(batch["ego"], dict):
            candidates.append(batch["ego"].get("record_len"))
        candidates.append(batch.get("record_len"))
    for value in candidates:
        if torch.is_tensor(value):
            return [int(x) for x in value.detach().cpu().reshape(-1).tolist()]
        if isinstance(value, (list, tuple)):
            return [int(x) for x in value]
    return []


def iter_tensors(obj, prefix=""):
    if torch.is_tensor(obj):
        yield prefix or "tensor", obj
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = "{}.{}".format(prefix, key) if prefix else str(key)
            for item in iter_tensors(value, child):
                yield item
        return
    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            child = "{}[{}]".format(prefix, idx) if prefix else "[{}]".format(idx)
            for item in iter_tensors(value, child):
                yield item


def split_sender_features(tensor, record_len):
    """Return [(batch_index, sender_index, feature_tensor), ...], excluding ego."""
    if not torch.is_tensor(tensor) or tensor.numel() == 0:
        return []
    lengths = [int(x) for x in record_len]
    if not lengths:
        return []
    total = int(sum(lengths))
    result = []

    if tensor.dim() >= 3 and int(tensor.shape[0]) == total:
        offset = 0
        for batch_index, count in enumerate(lengths):
            for local_index in range(1, count):
                result.append((batch_index, local_index, tensor[offset + local_index]))
            offset += count
        return result

    if tensor.dim() >= 4 and int(tensor.shape[0]) == len(lengths):
        if tensor.dim() >= 5 and int(tensor.shape[1]) >= max(lengths):
            for batch_index, count in enumerate(lengths):
                for local_index in range(1, count):
                    result.append((batch_index, local_index, tensor[batch_index, local_index]))
            return result

    if len(lengths) == 1 and lengths[0] == 2 and tensor.dim() == 3:
        return [(0, 1, tensor)]
    return []


def tensor_bytes(tensor):
    return int(tensor.numel()) * int(tensor.element_size())


def parse_profiles(text):
    profiles = OrderedDict()
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 4:
            raise ValueError("Bad profile {!r}; expected name:bw:plr:delay".format(item))
        name, bw, plr, delay = parts
        profiles[name] = {
            "bandwidth_mbps": float(bw),
            "plr": float(plr),
            "delay_ms": float(delay),
        }
    if not profiles:
        raise ValueError("No channel profiles parsed")
    return profiles


def budget_bytes(bandwidth_mbps, tx_window_ms):
    return int(math.floor(float(bandwidth_mbps) * 1e6 / 8.0 * float(tx_window_ms) / 1000.0))


def current_source_first_plan(k_source, rho, capacity_packets):
    k_source = int(k_source)
    capacity_packets = max(int(capacity_packets), 0)
    parity = int(math.ceil(k_source * float(rho))) if k_source > 0 else 0
    tx_source = min(k_source, capacity_packets)
    tx_parity = min(parity, max(capacity_packets - k_source, 0))
    return {
        "source_packets": k_source,
        "parity_generated": parity,
        "encoded_packets": k_source + parity,
        "tx_source_packets": tx_source,
        "tx_parity_packets": tx_parity,
        "source_dropped_by_budget": k_source - tx_source,
        "parity_dropped_by_budget": parity - tx_parity,
    }


def budget_feasible_plan(k_available, rho, capacity_packets):
    k_available = int(k_available)
    capacity_packets = max(int(capacity_packets), 0)
    rho = max(float(rho), 0.0)
    if capacity_packets <= 0 or k_available <= 0:
        return {"planned_source_packets": 0, "planned_parity_packets": 0}
    if rho <= 0.0:
        return {"planned_source_packets": min(k_available, capacity_packets), "planned_parity_packets": 0}
    k = min(k_available, int(math.floor(capacity_packets / (1.0 + rho))))
    r = min(int(math.ceil(k * rho)), max(capacity_packets - k, 0))
    return {"planned_source_packets": int(k), "planned_parity_packets": int(r)}


def json_dump(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
