#!/usr/bin/env bash
set -euo pipefail
MODEL_DIRS=(
  "opencood/logs/point_pillar_v2xvit_ours_extension"
  "opencood/logs/rocooper_ours_extension"
  "opencood/logs/coopdiff_ours_extension"
)
for MODEL_DIR in "${MODEL_DIRS[@]}"; do
  if [ ! -d "$MODEL_DIR" ]; then
    echo "[SKIP] missing $MODEL_DIR"
    continue
  fi
  LOG_DIR="$MODEL_DIR/final_alignment_extension_seed2026"
  echo "[EXTENSION] $MODEL_DIR"
  python opencood/tools/inference_arce.py \
    --model_dir "$MODEL_DIR" \
    --fusion_method intermediate \
    --save_comm \
    --comm_log_dir "$LOG_DIR" \
    --num_workers 4
  python opencood/tools/summarize_final_alignment_metrics.py \
    --comm-jsonl "$LOG_DIR/arce_comm_flat.jsonl" \
    --out "$LOG_DIR/final_metrics_summary.json" || true
done
