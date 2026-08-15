#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MAX_FRAMES="${MAX_FRAMES:-200}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-2026}"
ONLY="${ONLY:-}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/audit_runs/all_baselines_markov_bw_test${MAX_FRAMES}}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

args=(
  python scripts/run_markov_bw_manifest.py
  --manifest scripts/markov_bw_baselines.yaml
  --out_root "${OUT_ROOT}"
  --max_frames "${MAX_FRAMES}"
  --num_workers "${NUM_WORKERS}"
  --seed "${SEED}"
)

if [[ -n "${ONLY}" ]]; then
  args+=(--only "${ONLY}")
fi

if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
  args+=(--continue_on_error)
fi

echo "Project: ${PROJECT_ROOT}"
echo "Frames : ${MAX_FRAMES}"
echo "Output : ${OUT_ROOT}"
echo "Only   : ${ONLY:-all}"
"${args[@]}"
