#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR=${1:?Usage: bash run_where2comm_one_sample_check.sh <model_dir>}
python opencood/tools/inference_arce.py \
  --model_dir "$MODEL_DIR" \
  --fusion_method intermediate \
  --save_comm \
  --max_samples 1 \
  --num_workers 0 \
  --comm_log_dir "$MODEL_DIR/final_alignment_one_sample"
python opencood/tools/summarize_final_alignment_metrics.py \
  --comm-jsonl "$MODEL_DIR/final_alignment_one_sample/arce_comm_flat.jsonl" \
  --out "$MODEL_DIR/final_alignment_one_sample/final_metrics_summary.json" || true
