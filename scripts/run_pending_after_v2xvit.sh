#!/usr/bin/env bash
# Serial continuation for the remaining evaluation repairs.  It waits for the
# active compression=32 V2X-ViT job so the single GPU is never oversubscribed.
set -euo pipefail

ROOT=/home/server/v2x_projects/V2V_inference
OUT="$ROOT/opencood/logs/all_baselines_ideal_markov_test_20260815"
PY=/home/server/anaconda3/envs/opencood/bin/python
export PYTHONPATH="$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$ROOT"

while pgrep -f '[r]un_all_baselines_ideal_markov.py --only v2xvit' >/dev/null; do
  sleep 60
done

# Preserve current policy YAMLs. --allow_policy is intentional: the audit
# records their configured communication rather than rejecting their presence.
"$PY" scripts/run_all_baselines_ideal_markov.py --only cosdh --mode both
"$PY" scripts/run_all_baselines_ideal_markov.py --only opv2v/coopdiff --mode ideal
"$PY" scripts/run_all_baselines_ideal_markov.py --only v2xreal/where2comm --mode markov
"$PY" scripts/repair_all_baselines_bw.py --only opv2v/rocooper --mode markov --force
"$PY" scripts/repair_all_baselines_bw.py --only v2xvit --mode markov --force

echo "PENDING_REPAIRS_DONE $(date -Is)"
