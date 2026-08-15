from __future__ import print_function

import argparse
import csv
import json
import os
import re
from collections import defaultdict

import torch

from baseline_feature_budget_common import (
    get_record_len,
    iter_tensors,
    load_runtime,
    run_inference,
    set_seed,
    split_sender_features,
    tensor_bytes,
)
from opencood.tools import train_utils


def parse_args():
    p = argparse.ArgumentParser(description="Discover candidate fusion-input hook modules.")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--fusion_method", default="intermediate")
    p.add_argument("--max_frames", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--module_regex",
        default=r"(^|\.)(fusion_net|fuse_modules|fusion|fuse)(\.|$)",
        help="Regex applied to model.named_modules() names.",
    )
    p.add_argument("--all_modules", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    _, dataset, loader, device, model, loaded_epoch = load_runtime(
        args.model_dir, args.num_workers
    )
    pattern = re.compile(args.module_regex)
    events = []
    current = {"frame": -1, "record_len": []}
    handles = []

    def make_hook(module_name, module_class):
        def hook(_module, inputs):
            for tensor_path, tensor in iter_tensors(inputs, "input"):
                if tensor.dim() < 3:
                    continue
                splits = split_sender_features(tensor, current["record_len"])
                events.append({
                    "frame_index": int(current["frame"]),
                    "module_name": module_name,
                    "module_class": module_class,
                    "tensor_path": tensor_path,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "tensor_bytes": tensor_bytes(tensor),
                    "split_sender_count": len(splits),
                    "per_sender_shapes": [list(x[2].shape) for x in splits[:8]],
                    "per_sender_bytes": [tensor_bytes(x[2]) for x in splits[:8]],
                })
        return hook

    selected_names = []
    for name, module in model.named_modules():
        if not name:
            continue
        if args.all_modules or pattern.search(name):
            selected_names.append(name)
            handles.append(module.register_forward_pre_hook(make_hook(name, module.__class__.__name__)))

    if not handles:
        raise RuntimeError("No modules matched regex: {}".format(args.module_regex))

    with torch.no_grad():
        for frame_index, batch in enumerate(loader):
            if frame_index >= int(args.max_frames):
                break
            current["frame"] = frame_index
            current["record_len"] = get_record_len(batch)
            batch = train_utils.to_device(batch, device)
            run_inference(args.fusion_method, batch, model, dataset)

    for handle in handles:
        handle.remove()

    grouped = defaultdict(list)
    for event in events:
        grouped[(event["module_name"], event["module_class"], event["tensor_path"])].append(event)

    rows = []
    for (name, cls, tensor_path), items in grouped.items():
        split_counts = [x["split_sender_count"] for x in items]
        shapes = []
        seen = set()
        for x in items:
            value = tuple(x["shape"])
            if value not in seen:
                seen.add(value)
                shapes.append(list(value))
        score = 0
        lower = name.lower()
        if "fusion" in lower or "fuse" in lower:
            score += 3
        if max(split_counts or [0]) > 0:
            score += 6
        if any(len(x["shape"]) in (4, 5) for x in items):
            score += 2
        rows.append({
            "score": score,
            "module_name": name,
            "module_class": cls,
            "tensor_path": tensor_path,
            "call_count": len(items),
            "max_split_sender_count": max(split_counts or [0]),
            "shapes": json.dumps(shapes, ensure_ascii=False),
            "example_per_sender_shapes": json.dumps(items[0]["per_sender_shapes"], ensure_ascii=False),
            "example_per_sender_bytes": json.dumps(items[0]["per_sender_bytes"], ensure_ascii=False),
        })
    rows.sort(key=lambda x: (-x["score"], x["module_name"], x["tensor_path"]))

    csv_path = os.path.join(args.out_dir, "candidate_tx_hooks.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["module_name"])
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(args.out_dir, "candidate_tx_hooks.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model_dir": args.model_dir,
            "loaded_epoch": loaded_epoch,
            "module_regex": args.module_regex,
            "selected_modules": selected_names,
            "candidates": rows,
        }, f, ensure_ascii=False, indent=2)

    print("Loaded epoch {}".format(loaded_epoch))
    print("Top candidate hook modules:")
    for row in rows[:20]:
        print("score={score:2d} module={module_name} path={tensor_path} senders={max_split_sender_count} shapes={shapes}".format(**row))
    print("Saved: {}".format(csv_path))


if __name__ == "__main__":
    main()
