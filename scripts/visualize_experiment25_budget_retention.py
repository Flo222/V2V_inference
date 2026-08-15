#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--modes", nargs="+", default=["fp32", "fp16", "int8", "int4"])
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("matplotlib unavailable; retention tensors and CSV files were still saved:", exc)
        return 0

    for mode in args.modes:
        snapshot_dir = os.path.join(args.root, mode, "audit", "tensor_snapshots")
        output_dir = os.path.join(args.root, mode, "retention_plots")
        os.makedirs(output_dir, exist_ok=True)
        for path in sorted(glob.glob(os.path.join(snapshot_dir, "*.pt"))):
            data = torch.load(path, map_location="cpu")
            base = os.path.splitext(os.path.basename(path))[0]
            channel = data.get("channel_retention_ratio_tensor")
            spatial = data.get("spatial_retention_ratio_tensor")
            if torch.is_tensor(channel):
                plt.figure(figsize=(8, 3))
                plt.plot(range(int(channel.numel())), channel.flatten().numpy(), marker=".")
                plt.ylim(-0.05, 1.05)
                plt.xlabel("Channel index")
                plt.ylabel("Retained value ratio")
                plt.title("%s channel retention" % mode)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, base + "_channel.png"), dpi=160)
                plt.close()
            if torch.is_tensor(spatial):
                plt.figure(figsize=(10, 4))
                plt.imshow(spatial.numpy(), vmin=0.0, vmax=1.0, aspect="auto")
                plt.colorbar(label="Retained channel ratio")
                plt.xlabel("W")
                plt.ylabel("H")
                plt.title("%s spatial retention" % mode)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, base + "_spatial.png"), dpi=160)
                plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
