from __future__ import print_function

import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np
import torch

from baseline_feature_budget_common import (
    budget_bytes,
    budget_feasible_plan,
    current_source_first_plan,
    get_record_len,
    iter_tensors,
    load_runtime,
    parse_profiles,
    run_inference,
    set_seed,
    split_sender_features,
    tensor_bytes,
)
from opencood.communication.transport.quantization.feature_quantizer import FeatureQuantizer
from opencood.communication.transport.packetization.byte_stream_packetizer import ByteStreamPacketizer
from opencood.tools import train_utils


def parse_float_list(text):
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_str_list(text):
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="Profile real pre-fusion feature sizes and budget fit.")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--hook_modules", required=True, help="Comma-separated module@tensor_path specs; @tensor_path is optional")
    p.add_argument("--extra_metadata_bytes", type=int, default=0, help="Unquantized per-sender metadata bytes, e.g. 12 bytes for V2X-ViT velocity/delay/infra priors")
    p.add_argument("--packetization_mode", choices=("concat", "per_part"), default="concat")
    p.add_argument("--fusion_method", default="intermediate")
    p.add_argument("--max_frames", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--quant_modes", default="fp32,fp16,int8,int4")
    p.add_argument("--rhos", default="0,0.1,0.25,0.6")
    p.add_argument(
        "--profiles",
        default="good:27:0.05:10,medium:5:0.20:50,bad:1:0.35:100",
    )
    p.add_argument("--tx_window_ms", type=float, default=100.0)
    p.add_argument("--packet_size_bytes", type=int, default=1024)
    p.add_argument(
        "--budget_scope",
        choices=("per_link", "frame_shared", "both"),
        default="both",
        help="per_link gives every link the full profile budget; frame_shared divides by active senders.",
    )
    p.add_argument("--granularity", choices=("per_tensor", "per_channel"), default="per_tensor")
    return p.parse_args()


def quantize_part(tensor, mode, granularity, packetizer):
    cfg = {
        "quantization": {
            "enabled": mode != "fp32",
            "mode": mode,
            "raw_bits": 32,
            "granularity": granularity,
            "channel_dim": 0,
            "compute_error": True,
            "pack_int4": True,
        }
    }
    quantizer = FeatureQuantizer(cfg)
    if hasattr(quantizer, "quantize_feature"):
        result = quantizer.quantize_feature(tensor, mode=mode)
    else:
        result = quantizer.quantize(tensor, mode=mode)
    stream = result.packed_tensor if result.packed_tensor is not None else result.q_tensor
    packet_result = packetizer.packetize(
        stream,
        source_tensor_kind="packed_int4" if result.packed_tensor is not None else "q_tensor",
    )
    error = result.error or {}
    return {
        "payload_bytes": int(packet_result.original_num_bytes),
        "source_packets": int(packet_result.num_packets),
        "nmse": float(error.get("nmse", 0.0)),
        "cosine": float(error.get("cosine_similarity", error.get("cosine", 1.0))),
        "q_dtype": str(result.q_tensor.dtype),
        "packed_int4": bool(result.packed_tensor is not None),
    }


def mean(values):
    values = [float(x) for x in values if x is not None and np.isfinite(float(x))]
    return float(sum(values) / len(values)) if values else None


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    quant_modes = parse_str_list(args.quant_modes)
    rhos = parse_float_list(args.rhos)
    profiles = parse_profiles(args.profiles)
    hook_specs = parse_str_list(args.hook_modules)
    parsed_hooks = []
    for spec in hook_specs:
        if "@" in spec:
            name, tensor_path = spec.split("@", 1)
            parsed_hooks.append((name.strip(), tensor_path.strip()))
        else:
            parsed_hooks.append((spec.strip(), None))
    hook_names = [x[0] for x in parsed_hooks]

    _, dataset, loader, device, model, loaded_epoch = load_runtime(args.model_dir, args.num_workers)
    modules = dict(model.named_modules())
    missing = [name for name in hook_names if name not in modules]
    if missing:
        raise KeyError("Hook module(s) not found: {}".format(missing))

    packetizer = ByteStreamPacketizer({"packetizer": {"packet_size_bytes": int(args.packet_size_bytes)}})
    current = {"frame": -1, "record_len": [], "captures": []}
    handles = []

    def make_hook(module_name, required_path):
        def hook(_module, inputs):
            for tensor_path, tensor in iter_tensors(inputs, "input"):
                if required_path is not None and tensor_path != required_path:
                    continue
                splits = split_sender_features(tensor, current["record_len"])
                if not splits:
                    continue
                for batch_index, sender_index, feature in splits:
                    current["captures"].append({
                        "module_name": module_name,
                        "tensor_path": tensor_path,
                        "batch_index": int(batch_index),
                        "sender_index": int(sender_index),
                        "feature": feature.detach(),
                    })
        return hook

    for name, required_path in parsed_hooks:
        handles.append(modules[name].register_forward_pre_hook(make_hook(name, required_path)))

    feature_rows = []
    budget_rows = []

    with torch.no_grad():
        for frame_index, batch in enumerate(loader):
            if frame_index >= int(args.max_frames):
                break
            current["frame"] = frame_index
            current["record_len"] = get_record_len(batch)
            current["captures"] = []
            batch = train_utils.to_device(batch, device)
            run_inference(args.fusion_method, batch, model, dataset)

            grouped = defaultdict(list)
            for capture in current["captures"]:
                key = (capture["batch_index"], capture["sender_index"])
                grouped[key].append(capture)

            num_senders = sum(max(int(x) - 1, 0) for x in current["record_len"])
            if num_senders <= 0:
                continue

            for (batch_index, sender_index), captures in sorted(grouped.items()):
                # Avoid duplicate capture of exactly the same storage/shape from the same hook call.
                unique = []
                seen = set()
                for item in captures:
                    feature = item["feature"]
                    key = (item["module_name"], item["tensor_path"], tuple(feature.shape), int(feature.data_ptr()))
                    if key in seen:
                        continue
                    seen.add(key)
                    unique.append(item)
                captures = unique
                feature_raw_bytes = sum(tensor_bytes(x["feature"]) for x in captures)
                raw_numel = sum(int(x["feature"].numel()) for x in captures)
                metadata_bytes = max(int(args.extra_metadata_bytes), 0)
                raw_bytes = feature_raw_bytes + metadata_bytes
                shape_desc = [
                    {"module": x["module_name"], "path": x["tensor_path"], "shape": list(x["feature"].shape), "dtype": str(x["feature"].dtype)}
                    for x in captures
                ]

                for mode in quant_modes:
                    part_results = [
                        quantize_part(x["feature"], mode, args.granularity, packetizer)
                        for x in captures
                    ]
                    feature_payload_bytes = sum(x["payload_bytes"] for x in part_results)
                    payload_bytes = feature_payload_bytes + metadata_bytes
                    source_packets_per_part = sum(x["source_packets"] for x in part_results) + (1 if metadata_bytes > 0 else 0)
                    source_packets_concat = int(math.ceil(payload_bytes / float(args.packet_size_bytes))) if payload_bytes > 0 else 0
                    source_packets = source_packets_concat if args.packetization_mode == "concat" else source_packets_per_part
                    weighted_nmse_num = sum(
                        float(x["nmse"]) * int(captures[i]["feature"].numel())
                        for i, x in enumerate(part_results)
                    )
                    nmse = weighted_nmse_num / max(raw_numel, 1)
                    feature_rows.append({
                        "frame_index": frame_index,
                        "batch_index": batch_index,
                        "sender_index": sender_index,
                        "num_active_senders": num_senders,
                        "hook_modules": ",".join(hook_specs),
                        "parts": len(captures),
                        "shapes": json.dumps(shape_desc, ensure_ascii=False),
                        "raw_numel": raw_numel,
                        "feature_raw_bytes": feature_raw_bytes,
                        "metadata_bytes": metadata_bytes,
                        "raw_bytes": raw_bytes,
                        "quant_mode": mode,
                        "feature_quant_payload_bytes": feature_payload_bytes,
                        "quant_payload_bytes": payload_bytes,
                        "source_packets": source_packets,
                        "source_packets_concat": source_packets_concat,
                        "source_packets_per_part": source_packets_per_part,
                        "packetization_mode": args.packetization_mode,
                        "packet_size_bytes": int(args.packet_size_bytes),
                        "payload_compression_ratio_vs_raw": float(payload_bytes / raw_bytes) if raw_bytes else 0.0,
                        "quant_nmse": float(nmse),
                        "packed_int4": any(x["packed_int4"] for x in part_results),
                    })

                    scopes = [args.budget_scope] if args.budget_scope != "both" else ["per_link", "frame_shared"]
                    for state, profile in profiles.items():
                        frame_budget = budget_bytes(profile["bandwidth_mbps"], args.tx_window_ms)
                        for scope in scopes:
                            link_budget = frame_budget if scope == "per_link" else int(math.floor(frame_budget / float(num_senders)))
                            capacity_packets = int(math.floor(link_budget / float(args.packet_size_bytes)))
                            for rho in rhos:
                                current_plan = current_source_first_plan(source_packets, rho, capacity_packets)
                                feasible = budget_feasible_plan(source_packets, rho, capacity_packets)
                                row = {
                                    "frame_index": frame_index,
                                    "batch_index": batch_index,
                                    "sender_index": sender_index,
                                    "num_active_senders": num_senders,
                                    "quant_mode": mode,
                                    "rho": rho,
                                    "channel_state": state,
                                    "bandwidth_mbps": profile["bandwidth_mbps"],
                                    "plr": profile["plr"],
                                    "delay_ms": profile["delay_ms"],
                                    "tx_window_ms": args.tx_window_ms,
                                    "budget_scope": scope,
                                    "frame_budget_bytes": frame_budget,
                                    "link_budget_bytes": link_budget,
                                    "capacity_packets": capacity_packets,
                                    "quant_payload_bytes": payload_bytes,
                                    "packet_size_bytes": int(args.packet_size_bytes),
                                    **current_plan,
                                    "source_tx_ratio_current": float(current_plan["tx_source_packets"] / source_packets) if source_packets else 1.0,
                                    "parity_tx_ratio_current": float(current_plan["tx_parity_packets"] / current_plan["parity_generated"]) if current_plan["parity_generated"] else 0.0,
                                    "expected_received_source_packets_current": float(current_plan["tx_source_packets"] * (1.0 - profile["plr"])),
                                    "expected_received_parity_packets_current": float(current_plan["tx_parity_packets"] * (1.0 - profile["plr"])),
                                    **feasible,
                                    "expected_received_source_packets_feasible": float(feasible["planned_source_packets"] * (1.0 - profile["plr"])),
                                    "expected_received_parity_packets_feasible": float(feasible["planned_parity_packets"] * (1.0 - profile["plr"])),
                                }
                                budget_rows.append(row)

    for handle in handles:
        handle.remove()

    if not feature_rows:
        raise RuntimeError("No sender features captured. Run discover_baseline_tx_hooks.py and choose a module whose max_split_sender_count > 0.")

    def write_csv(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(os.path.join(args.out_dir, "feature_sizes_per_link.csv"), feature_rows)
    write_csv(os.path.join(args.out_dir, "budget_fit_per_link.csv"), budget_rows)

    grouped_summary = defaultdict(list)
    for row in budget_rows:
        key = (row["quant_mode"], row["rho"], row["channel_state"], row["budget_scope"])
        grouped_summary[key].append(row)
    summary_rows = []
    for key, rows in sorted(grouped_summary.items()):
        quant_mode, rho, state, scope = key
        summary_rows.append({
            "quant_mode": quant_mode,
            "rho": rho,
            "channel_state": state,
            "budget_scope": scope,
            "record_count": len(rows),
            "mean_quant_payload_bytes": mean([x["quant_payload_bytes"] for x in rows]),
            "mean_source_packets": mean([x["source_packets"] for x in rows]),
            "mean_link_budget_bytes": mean([x["link_budget_bytes"] for x in rows]),
            "mean_capacity_packets": mean([x["capacity_packets"] for x in rows]),
            "mean_tx_source_packets_current": mean([x["tx_source_packets"] for x in rows]),
            "mean_tx_parity_packets_current": mean([x["tx_parity_packets"] for x in rows]),
            "mean_source_tx_ratio_current": mean([x["source_tx_ratio_current"] for x in rows]),
            "mean_parity_tx_ratio_current": mean([x["parity_tx_ratio_current"] for x in rows]),
            "all_source_fit_ratio_current": mean([1.0 if x["tx_source_packets"] == x["source_packets"] else 0.0 for x in rows]),
            "any_parity_fit_ratio_current": mean([1.0 if x["tx_parity_packets"] > 0 else 0.0 for x in rows]),
            "mean_planned_source_packets_feasible": mean([x["planned_source_packets"] for x in rows]),
            "mean_planned_parity_packets_feasible": mean([x["planned_parity_packets"] for x in rows]),
            "mean_expected_received_source_current": mean([x["expected_received_source_packets_current"] for x in rows]),
            "mean_expected_received_parity_current": mean([x["expected_received_parity_packets_current"] for x in rows]),
        })
    write_csv(os.path.join(args.out_dir, "budget_fit_summary.csv"), summary_rows)

    manifest = {
        "model_dir": args.model_dir,
        "loaded_epoch": loaded_epoch,
        "hook_modules": hook_names,
        "max_frames": args.max_frames,
        "quant_modes": quant_modes,
        "rhos": rhos,
        "profiles": profiles,
        "tx_window_ms": args.tx_window_ms,
        "packet_size_bytes": args.packet_size_bytes,
        "budget_scope": args.budget_scope,
        "feature_record_count": len(feature_rows),
        "budget_record_count": len(budget_rows),
        "notes": [
            "current rows reproduce source-first packet ordering: all source packets precede parity packets",
            "budget_feasible rows are a comparison plan only; they do not modify the project",
            "expected received counts use packet_count*(1-PLR) and are not an FEC decoding simulation",
            "payload counts follow the current quantizer and byte-stream packetizer; quantization metadata/headers are not added unless current packetizer includes them",
        ],
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("Loaded epoch {}".format(loaded_epoch))
    print("Captured {} feature/quant records".format(len(feature_rows)))
    print("Generated {} budget rows".format(len(budget_rows)))
    print("Summary: {}".format(os.path.join(args.out_dir, "budget_fit_summary.csv")))


if __name__ == "__main__":
    main()
