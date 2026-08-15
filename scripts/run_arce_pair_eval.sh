#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:$PYTHONPATH

METHOD="${METHOD:-both}"          # arce | baseline | fixed | both
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

TAG="${TAG:-payload_aligned_eval}"
OUT_ROOT="${OUT_ROOT:-outputs}"

ARCE_OUT_DIR="${ARCE_OUT_DIR:-${OUT_ROOT}/arce_c2mab/${TAG}}"
BASELINE_OUT_DIR="${BASELINE_OUT_DIR:-${OUT_ROOT}/baselines/where2comm_arce_fixed/${TAG}_compare}"

export RUN_AP
export RUN_BW
export MAX_FRAMES
export SCENARIO
export PROGRESS_INTERVAL
export NUM_WORKERS
export SEED
export SINGLE_PASS_AP_BW
export WINDOW_SIZE
export WINDOW_STRIDE
export WARMUP_FRAMES
export GLOBAL_SORT_DETECTIONS

run_arce() {
  echo
  echo "========== Run ARCE-C2MAB =========="
  echo "OUT_DIR=${ARCE_OUT_DIR}"
  OUT_DIR="${ARCE_OUT_DIR}" bash scripts/run_arce_c2mab.sh
}

run_baseline() {
  echo
  echo "========== Run Where2Comm-ARCE-Fixed baseline =========="
  echo "OUT_DIR=${BASELINE_OUT_DIR}"
  OUT_DIR="${BASELINE_OUT_DIR}" bash scripts/run_where2comm_arce_fixed.sh
}

case "${METHOD}" in
  arce|c2mab)
    run_arce
    ;;
  baseline|fixed|where2comm)
    run_baseline
    ;;
  both|all)
    run_arce
    run_baseline
    ;;
  *)
    echo "Unknown METHOD=${METHOD}" >&2
    echo "Use METHOD=arce | baseline | both" >&2
    exit 2
    ;;
esac

echo
echo "========== Finished =========="
echo "METHOD=${METHOD}"
echo "RUN_AP=${RUN_AP}"
echo "RUN_BW=${RUN_BW}"
echo "MAX_FRAMES=${MAX_FRAMES}"
echo "ARCE_OUT_DIR=${ARCE_OUT_DIR}"
echo "BASELINE_OUT_DIR=${BASELINE_OUT_DIR}"
