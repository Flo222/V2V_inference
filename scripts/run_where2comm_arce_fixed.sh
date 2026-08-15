#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

METHOD_NAME="${METHOD_NAME:-Where2Comm-ARCE-Fixed}"
MODEL_DIR="${MODEL_DIR:-opencood/logs/where2comm_markov_trueloss_fp32_rho0_cache0}"
OUT_DIR="${OUT_DIR:-outputs/baselines/where2comm_arce_fixed/default}"

export METHOD_NAME
export MODEL_DIR
export OUT_DIR

bash scripts/run_arce_single_eval.sh
