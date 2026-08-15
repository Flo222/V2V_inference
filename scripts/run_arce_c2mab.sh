#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

METHOD_NAME="${METHOD_NAME:-ARCE-C2MAB}"
MODEL_DIR="${MODEL_DIR:-opencood/logs/main_opv2v_where2comm_grace_full}"
OUT_DIR="${OUT_DIR:-outputs/arce_c2mab/default}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"

export METHOD_NAME
export MODEL_DIR
export OUT_DIR

if [ "$RUN_PREFLIGHT" = "1" ]; then
  echo "===== GRACE static preflight ====="
  python scripts/preflight_final_markov_c2mab_runtime.py \
    --model-dir "$MODEL_DIR"
elif [ "$RUN_PREFLIGHT" != "0" ]; then
  echo "RUN_PREFLIGHT must be 0 or 1, got: $RUN_PREFLIGHT" >&2
  exit 2
fi

bash scripts/run_arce_single_eval.sh
