#!/usr/bin/env bash
set -euo pipefail

# 修改为你的实际模型目录。每个目录应包含 config.yaml 和 checkpoint。
MODEL_DIRS=(
  "opencood/logs/point_pillar_where2comm_arce_fixed"
  "opencood/logs/point_pillar_where2comm_arce_random"
  "opencood/logs/point_pillar_where2comm_arce_c2mab"
  "opencood/logs/point_pillar_where2comm_arce_c2mab_comp"
  "opencood/logs/point_pillar_where2comm_arce_c2mab_comp_div"
)

for MODEL_DIR in "${MODEL_DIRS[@]}"; do
  NAME=$(basename "$MODEL_DIR")
  LOG_DIR="$MODEL_DIR/final_alignment_markov_seed2026"
  echo "============================================================"
  echo "[RUN] $NAME"
  echo "MODEL_DIR=$MODEL_DIR"
  echo "LOG_DIR=$LOG_DIR"
  echo "============================================================"
  python opencood/tools/inference_arce.py \
    --model_dir "$MODEL_DIR" \
    --fusion_method intermediate \
    --save_comm \
    --comm_log_dir "$LOG_DIR" \
    --num_workers 4
  python opencood/tools/summarize_final_alignment_metrics.py \
    --comm-jsonl "$LOG_DIR/arce_comm_flat.jsonl" \
    --out "$LOG_DIR/final_metrics_summary.json" || true
  python opencood/tools/export_statewise_eval_index.py \
    --comm-jsonl "$LOG_DIR/arce_comm_flat.jsonl" \
    --out "$LOG_DIR/statewise_eval_index.csv" || true
done
