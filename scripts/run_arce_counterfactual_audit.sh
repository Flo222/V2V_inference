#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:$PYTHONPATH

MODEL_DIR="${MODEL_DIR:-opencood/logs/main_opv2v_where2comm_grace_full}"
OUT_DIR="${OUT_DIR:-audit_runs/arce_counterfactual_7action}"
MAX_FRAMES="${MAX_FRAMES:-500}"
AUDIT_FRAMES="${AUDIT_FRAMES:-20}"
AUDIT_STRIDE="${AUDIT_STRIDE:-25}"
AUDIT_START="${AUDIT_START:-0}"
SENDER_INDEX="${SENDER_INDEX:-1}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-10}"

mkdir -p "$OUT_DIR"

echo "===== ARCE matched-state counterfactual audit ====="
echo "MODEL_DIR=$MODEL_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "MAX_FRAMES=$MAX_FRAMES"
echo "AUDIT_FRAMES=$AUDIT_FRAMES"
echo "AUDIT_STRIDE=$AUDIT_STRIDE"
echo "AUDIT_START=$AUDIT_START"
echo "SENDER_INDEX=$SENDER_INDEX"
echo "PROGRESS_INTERVAL=$PROGRESS_INTERVAL"

PYTHONUNBUFFERED=1 python opencood/tools/audit_arce_counterfactual.py \
  --model_dir "$MODEL_DIR" \
  --out_json "$OUT_DIR/counterfactual_7action.json" \
  --max_frames "$MAX_FRAMES" \
  --audit_frames "$AUDIT_FRAMES" \
  --audit_stride "$AUDIT_STRIDE" \
  --audit_start "$AUDIT_START" \
  --sender_index "$SENDER_INDEX" \
  --progress_interval "$PROGRESS_INTERVAL" \
  2>&1 | tee "$OUT_DIR/counterfactual.log"

echo "saved: $OUT_DIR/counterfactual_7action.json"
