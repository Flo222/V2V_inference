# -*- coding: utf-8 -*-
"""
Run OpenCOOD inference on either validation or test split without permanently
modifying config.yaml.

This wrapper:
1. Reads model_dir/config.yaml.
2. Temporarily changes validate_dir to /data/v2xreal/validate or /data/v2xreal/test.
3. Runs opencood/tools/inference_v2xreal.py.
4. Restores the original config.yaml automatically.

Examples:

No Fusion test:
python scripts/run_infer_split_v2xreal.py \
  --model_dir opencood/logs/point_pillar_nofusion_v2xreal_vc_2026_05_12_10_24_18 \
  --fusion_method nofusion \
  --dataset_mode vc \
  --epoch 19 \
  --split test

V2X-ViT test:
python scripts/run_infer_split_v2xreal.py \
  --model_dir opencood/logs/point_pillar_v2xvit_v2xreal_vc_2026_05_11_22_45_46 \
  --fusion_method intermediate \
  --dataset_mode vc \
  --epoch 19 \
  --split test
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=str)
    parser.add_argument(
        "--fusion_method",
        required=True,
        type=str,
        choices=["nofusion", "late", "early", "intermediate"],
    )
    parser.add_argument("--dataset_mode", default="vc", type=str)
    parser.add_argument("--epoch", required=True, type=int)
    parser.add_argument("--split", required=True, type=str, choices=["validate", "test"])
    parser.add_argument("--log_name", default=None, type=str)
    return parser.parse_args()


def replace_validate_dir(config_text: str, split: str) -> str:
    target_dir = f"/data/v2xreal/{split}"
    new_lines = []
    replaced = False

    for line in config_text.splitlines():
        if line.strip().startswith("validate_dir:"):
            indent = line[: len(line) - len(line.lstrip())]
            new_lines.append(f"{indent}validate_dir: {target_dir}")
            replaced = True
        else:
            new_lines.append(line)

    if not replaced:
        raise RuntimeError("No validate_dir field found in config.yaml")

    return "\n".join(new_lines) + "\n"


def main():
    args = parse_args()

    model_dir = Path(args.model_dir)
    config_path = model_dir / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Cannot find config.yaml: {config_path}")

    original_config = config_path.read_text()

    if args.log_name is None:
        log_name = f"inference_{args.fusion_method}_{args.split}_epoch{args.epoch}.txt"
    else:
        log_name = args.log_name

    log_path = model_dir / log_name

    try:
        new_config = replace_validate_dir(original_config, args.split)
        config_path.write_text(new_config)

        print("=" * 80)
        print(f"Model dir: {model_dir}")
        print(f"Split: {args.split}")
        print(f"Fusion method: {args.fusion_method}")
        print(f"Dataset mode: {args.dataset_mode}")
        print(f"Epoch: {args.epoch}")
        print(f"Log file: {log_path}")
        print("=" * 80)

        cmd = [
            sys.executable,
            "opencood/tools/inference_v2xreal.py",
            "--model_dir",
            str(model_dir),
            "--fusion_method",
            args.fusion_method,
            "--dataset_mode",
            args.dataset_mode,
            "--epoch",
            str(args.epoch),
        ]

        with log_path.open("w") as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                f.write(line)

            return_code = process.wait()

        if return_code != 0:
            raise RuntimeError(f"inference.py failed with return code {return_code}")

    finally:
        config_path.write_text(original_config)
        print("=" * 80)
        print(f"Restored original config.yaml: {config_path}")
        print("=" * 80)


if __name__ == "__main__":
    main()