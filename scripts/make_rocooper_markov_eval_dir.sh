#!/usr/bin/env bash
set -euo pipefail
SRC_DIR="${1:?Usage: bash scripts/make_rocooper_markov_eval_dir.sh <rocooper_v2xreal_log_dir> [markov_eval_dir]}"
DST_DIR="${2:-${SRC_DIR}_markov_eval}"
cd "$(dirname "$0")/.."
rm -rf "$DST_DIR"
cp -r "$SRC_DIR" "$DST_DIR"
cp opencood/hypes_yaml/v2xreal/point_pillar_rocooper_markov_v2xreal_vc.yaml "$DST_DIR/config.yaml"
echo "Created RoCooper Markov eval dir: $DST_DIR"
echo "Run: python opencood/tools/inference_v2xreal.py --model_dir $DST_DIR --fusion_method intermediate --dataset_mode vc"
