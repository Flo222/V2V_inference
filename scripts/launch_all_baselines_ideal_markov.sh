#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}"
# inference.py enables deterministic PyTorch algorithms when a seed is given.
# CUDA 10.2+ requires this workspace selection for deterministic cuBLAS GEMM.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate opencood
cd /home/server/v2x_projects/V2V_inference
# Conda's opencood activation hook may retain the legacy OPV2V checkout.
# Evaluation must import the unified V2V_inference implementation instead.
export PYTHONPATH=/home/server/v2x_projects/V2V_inference
export PYTHONDONTWRITEBYTECODE=1

python scripts/run_all_baselines_ideal_markov.py --skip-completed "$@"
