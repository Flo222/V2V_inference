#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path

import argparse
import csv
import json
import os

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.tools.arce_eval_runtime import set_deterministic_seed
from opencood.tools.arce_reward_audit import save_reward_runtime_audit
from opencood.data_utils.datasets import build_dataset
from opencood.methods.arce.runtime.communication_volume_summary import summarize_bw_records
from opencood.tools.arce_bw_breakdown_utils import save_arce_bw_breakdown


def write_csv(path: str, summary: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "method",
        "scenario",
        "frame_count",
        "record_count",
        "transmitted_link_count",
        "no_send_count",
        "BW",
        "bw_MB_per_frame",
        "total_tx_MB",
        "int4_count",
        "packed_int4_count",
        "all_int4_packed",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({k: summary.get(k, "") for k in fields})


def main():
    parser = argparse.ArgumentParser(
        description="Summarize ARCE communication bandwidth from a model directory."
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--scenario", default="Markov")
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--progress_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    set_deterministic_seed(args.seed)

    hypes = yaml_utils.load_yaml(os.path.join(args.model_dir, "config.yaml"))
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_batch_test,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_utils.create_model(hypes).to(device)
    _, model = train_utils.load_saved_model(args.model_dir, model)
    model.eval()

    if not hasattr(model, "arce_comm"):
        raise AttributeError(
            "Model has no attribute 'arce_comm'. "
            "Please confirm this model_dir uses ARCE communication."
        )

    frame_count = 0
    max_frames = None if args.max_frames is None or args.max_frames < 0 else int(args.max_frames)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_frames is not None and i >= max_frames:
                break

            batch = train_utils.to_device(batch, device)
            _ = model(batch["ego"])
            frame_count += 1

            if args.progress_interval > 0 and frame_count % args.progress_interval == 0:
                print(f"{args.method} frames: {frame_count}", flush=True)

    records = model.arce_comm.get_records()
    summary = summarize_bw_records(
        records,
        method=args.method,
        scenario=args.scenario,
        num_frames=frame_count,
    )

    out_dir = Path(args.out_json).parent
    breakdown = save_arce_bw_breakdown(records, out_dir)
    reward_audit_info = save_reward_runtime_audit(records, out_dir, frame_count)

    summary["bw_breakdown_json"] = str(out_dir / "bw_breakdown.json")
    summary["reward_runtime_audit_json"] = reward_audit_info.get("reward_runtime_audit_json")
    summary["reward_update_count"] = reward_audit_info.get("reward_update_count")
    summary["avg_tokens_per_token_record"] = breakdown.get("avg_tokens_per_token_record")
    summary["avg_tx_bytes_per_token"] = breakdown.get("avg_tx_bytes_per_token")
    summary["bad_legacy_action_ids"] = breakdown.get("bad_legacy_action_ids", [])

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.out_csv:
        write_csv(args.out_csv, summary)

    print("===== BW summary =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("saved json:", args.out_json)
    if args.out_csv:
        print("saved csv:", args.out_csv)


if __name__ == "__main__":
    main()
