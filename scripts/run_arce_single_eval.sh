#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:$PYTHONPATH

METHOD_NAME="${METHOD_NAME:?Need METHOD_NAME}"
MODEL_DIR="${MODEL_DIR:?Need MODEL_DIR}"
OUT_DIR="${OUT_DIR:?Need OUT_DIR}"

RUN_AP="${RUN_AP:-1}"
RUN_BW="${RUN_BW:-1}"
MAX_FRAMES="${MAX_FRAMES:--1}"
SCENARIO="${SCENARIO:-Markov}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-50}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-2026}"
SINGLE_PASS_AP_BW="${SINGLE_PASS_AP_BW:-1}"
WINDOW_SIZE="${WINDOW_SIZE:-100}"
WINDOW_STRIDE="${WINDOW_STRIDE:-100}"
WARMUP_FRAMES="${WARMUP_FRAMES:-500}"
GLOBAL_SORT_DETECTIONS="${GLOBAL_SORT_DETECTIONS:-0}"

export OUT_DIR
export METHOD_NAME
export MODEL_DIR
export SCENARIO
export RUN_AP
export RUN_BW
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

for flag_name in RUN_AP RUN_BW SINGLE_PASS_AP_BW GLOBAL_SORT_DETECTIONS; do
  flag_value="${!flag_name}"
  if [ "$flag_value" != "0" ] && [ "$flag_value" != "1" ]; then
    echo "$flag_name must be 0 or 1, got: $flag_value" >&2
    exit 2
  fi
done

if [ "$RUN_AP" = "0" ] && [ "$RUN_BW" = "0" ]; then
  echo "At least one of RUN_AP or RUN_BW must be 1." >&2
  exit 2
fi

if [ ! -f "$MODEL_DIR/config.yaml" ]; then
  echo "Missing model config: $MODEL_DIR/config.yaml" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR/final_summary.json" "$OUT_DIR/final_table.csv"

echo "===== Single ARCE evaluation ====="
echo "Method: $METHOD_NAME"
echo "Model dir: $MODEL_DIR"
echo "Output dir: $OUT_DIR"
echo "RUN_AP: $RUN_AP"
echo "RUN_BW: $RUN_BW"
echo "MAX_FRAMES: $MAX_FRAMES"
echo "NUM_WORKERS: $NUM_WORKERS"
echo "SEED: $SEED"
echo "Scenario: $SCENARIO"
echo "Single-pass AP+BW: $SINGLE_PASS_AP_BW"
echo

echo "===== save config snapshot ====="
cp "$MODEL_DIR/config.yaml" "$OUT_DIR/config.yaml"

if [ "$RUN_AP" = "1" ] && [ "$RUN_BW" = "1" ] && [ "$SINGLE_PASS_AP_BW" = "1" ]; then
  echo
  echo "===== Run AP+BW on one online trajectory: $METHOD_NAME ====="
  EXTRA_ARGS=()
  if [ "$GLOBAL_SORT_DETECTIONS" = "1" ]; then
    EXTRA_ARGS+=(--global_sort_detections)
  fi
  PYTHONUNBUFFERED=1 python opencood/tools/arce_online_eval.py \
    --model_dir "$MODEL_DIR" \
    --out_dir "$OUT_DIR" \
    --method "$METHOD_NAME" \
    --scenario "$SCENARIO" \
    --fusion_method intermediate \
    --max_frames "$MAX_FRAMES" \
    --num_workers "$NUM_WORKERS" \
    --progress_interval "$PROGRESS_INTERVAL" \
    --window_size "$WINDOW_SIZE" \
    --window_stride "$WINDOW_STRIDE" \
    --warmup_frames "$WARMUP_FRAMES" \
    --seed "$SEED" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$OUT_DIR/online_eval.log"

  echo
  echo "===== Finished single-pass ARCE evaluation ====="
  echo "Output dir: $OUT_DIR"
  exit 0
fi

if [ "$RUN_AP" = "1" ]; then
  rm -f "$OUT_DIR/ap.log" "$OUT_DIR/ap_summary.txt"
  echo
  echo "===== Run AP: $METHOD_NAME ====="
  AP_EXTRA_ARGS=()
  if [ "$GLOBAL_SORT_DETECTIONS" = "1" ]; then
    AP_EXTRA_ARGS+=(--global_sort_detections)
  fi
  python opencood/tools/inference.py \
    --model_dir "$MODEL_DIR" \
    --fusion_method intermediate \
    --max_frames "$MAX_FRAMES" \
    --num_workers "$NUM_WORKERS" \
    --seed "$SEED" \
    "${AP_EXTRA_ARGS[@]}" \
    2>&1 | tee "$OUT_DIR/ap.log"

  echo
  echo "===== Extract AP ====="
  grep -E "Average Precision|AP@|ap_30|ap_50|ap_70" -n "$OUT_DIR/ap.log" \
    | tee "$OUT_DIR/ap_summary.txt"
else
  echo
  echo "===== Skip AP because RUN_AP=0 ====="
fi

if [ "$RUN_BW" = "1" ]; then
  rm -f "$OUT_DIR/bw.json" "$OUT_DIR/bw.csv" "$OUT_DIR/bw.log"
  echo
  echo "===== Run BW: $METHOD_NAME ====="
  PYTHONUNBUFFERED=1 python opencood/tools/arce_bw_summary.py \
    --model_dir "$MODEL_DIR" \
    --method "$METHOD_NAME" \
    --scenario "$SCENARIO" \
    --max_frames "$MAX_FRAMES" \
    --out_json "$OUT_DIR/bw.json" \
    --out_csv "$OUT_DIR/bw.csv" \
    --num_workers "$NUM_WORKERS" \
    --progress_interval "$PROGRESS_INTERVAL" \
    --seed "$SEED" \
    2>&1 | tee "$OUT_DIR/bw.log"
else
  echo
  echo "===== Skip BW because RUN_BW=0 ====="
fi

echo
echo "===== Build single final summary ====="
python opencood/tools/build_arce_eval_summary.py \
  --out_dir "$OUT_DIR" \
  --method "$METHOD_NAME" \
  --scenario "$SCENARIO" \
  --include_ap "$RUN_AP" \
  --include_bw "$RUN_BW"

echo
echo "===== Finished single ARCE evaluation ====="
echo "Output dir: $OUT_DIR"
