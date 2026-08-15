#!/usr/bin/env bash
set -euo pipefail
MODEL_DIRS=(
  "opencood/logs/point_pillar_where2comm_arce_ablate_no_comp"
  "opencood/logs/point_pillar_where2comm_arce_ablate_no_div"
  "opencood/logs/point_pillar_where2comm_arce_ablate_no_cache"
  "opencood/logs/point_pillar_where2comm_arce_ablate_no_red"
  "opencood/logs/point_pillar_where2comm_arce_ours_full"
)
for MODEL_DIR in "${MODEL_DIRS[@]}"; do
  LOG_DIR="$MODEL_DIR/final_alignment_ablation_seed2026"
  echo "[ABLATION] $MODEL_DIR"
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
