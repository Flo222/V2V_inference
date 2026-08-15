#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count effective collaborators in an OpenCOOD test split.

Primary definition:
    collaborator_count = record_len - 1

This is the number of non-ego CAVs that actually enter the model after the
dataset loader's own filtering, max_cav limit, communication range logic, and
sample validity checks.
"""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_hypes(config_path: str):
    from opencood.hypes_yaml import yaml_utils
    try:
        return yaml_utils.load_yaml(config_path, None)
    except TypeError:
        return yaml_utils.load_yaml(config_path)


def override_test_dir(hypes, test_dir: str) -> None:
    if not test_dir:
        return
    hypes["validate_dir"] = test_dir
    hypes["test_dir"] = test_dir


def build_test_dataset(hypes):
    from opencood.data_utils.datasets import build_dataset
    try:
        return build_dataset(hypes, visualize=False, train=False)
    except TypeError:
        return build_dataset(hypes, train=False)


def scalar_int(value) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            raise ValueError("empty tensor")
        return int(value.reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("empty ndarray")
        return int(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("empty list/tuple")
        return scalar_int(value[0])
    return int(value)


def extract_record_len(batch) -> int:
    candidates = []
    if isinstance(batch, dict):
        candidates.append(batch.get("record_len"))
        ego = batch.get("ego")
        if isinstance(ego, dict):
            candidates.append(ego.get("record_len"))
            candidates.append(ego.get("cav_num"))
        candidates.append(batch.get("cav_num"))

    for value in candidates:
        if value is not None:
            return scalar_int(value)

    keys = list(batch.keys()) if isinstance(batch, dict) else str(type(batch))
    raise KeyError(f"Could not find record_len in batch. Keys/type: {keys}")


def resolve_config(model_dir: str, config: str) -> Path:
    path = Path(config).expanduser().resolve() if config else Path(model_dir).expanduser().resolve() / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    return path


def dataset_split_path(hypes):
    for key in ("test_dir", "validate_dir", "root_dir"):
        value = hypes.get(key)
        if value:
            return key, str(value)
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--model_dir", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--test_dir", default="")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_frames", type=int, default=0, help="0 = all frames")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    seed_everything(args.seed)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = resolve_config(args.model_dir, args.config)
    hypes = load_hypes(str(config_path))
    override_test_dir(hypes, args.test_dir)
    split_key, split_path = dataset_split_path(hypes)

    dataset = build_test_dataset(hypes)
    collate_fn = getattr(dataset, "collate_batch_test", None)
    if collate_fn is None:
        raise AttributeError(f"{type(dataset).__name__} has no collate_batch_test")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    limit = len(dataset) if args.max_frames <= 0 else min(args.max_frames, len(dataset))
    rows = []
    dist = Counter()
    failures = []

    for frame_index, batch in enumerate(loader):
        if frame_index >= limit:
            break
        try:
            record_len = extract_record_len(batch)
            collaborator_count = max(record_len - 1, 0)
            dist[collaborator_count] += 1
            rows.append({
                "frame_index": frame_index,
                "record_len": record_len,
                "collaborator_count": collaborator_count,
                "has_collaborator": int(collaborator_count > 0),
            })
        except Exception as exc:
            failures.append({"frame_index": frame_index, "error": f"{type(exc).__name__}: {exc}"})

        if (frame_index + 1) % 500 == 0:
            print(f"[{args.dataset_name}] processed {frame_index + 1}/{limit}", flush=True)

    total = len(rows)
    active = sum(r["has_collaborator"] for r in rows)
    no_collab = total - active
    total_links = sum(r["collaborator_count"] for r in rows)
    max_collab = max((r["collaborator_count"] for r in rows), default=0)
    mean_all = total_links / total if total else 0.0
    mean_active = total_links / active if active else 0.0

    summary = {
        "dataset_name": args.dataset_name,
        "config_path": str(config_path),
        "split_path_key": split_key,
        "split_path": split_path,
        "dataset_class": type(dataset).__name__,
        "dataset_length_reported": len(dataset),
        "frames_requested": limit,
        "frames_counted": total,
        "failed_frames": len(failures),
        "frames_with_collaborator": active,
        "frames_without_collaborator": no_collab,
        "active_frame_ratio": active / total if total else 0.0,
        "total_collaborator_links": total_links,
        "mean_collaborators_per_all_frame": mean_all,
        "mean_collaborators_per_active_frame": mean_active,
        "max_collaborators_in_one_frame": max_collab,
        "collaborator_count_distribution": {str(k): dist[k] for k in sorted(dist)},
        "definition": "effective collaborator_count = collated record_len - 1",
        "seed": args.seed,
    }

    with open(out_dir / "frames.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_index", "record_len", "collaborator_count", "has_collaborator"])
        writer.writeheader()
        writer.writerows(rows)

    with open(out_dir / "collaborator_distribution.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["collaborator_count", "frame_count", "frame_ratio"])
        writer.writeheader()
        for count in sorted(dist):
            writer.writerow({
                "collaborator_count": count,
                "frame_count": dist[count],
                "frame_ratio": dist[count] / total if total else 0.0,
            })

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if failures:
        with open(out_dir / "failures.json", "w", encoding="utf-8") as f:
            json.dump(failures, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
