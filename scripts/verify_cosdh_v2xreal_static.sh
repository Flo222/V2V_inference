#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python -m py_compile \
  opencood/models/baselines/cosdh/models/point_pillar_cosdh_v2xreal.py \
  opencood/models/baselines/cosdh/models/point_pillar_cosdh_markov_v2xreal.py \
  opencood/tools/check_cosdh_v2xreal.py

python opencood/tools/check_cosdh_v2xreal.py \
  --hypes_yaml opencood/hypes_yaml/v2xreal/point_pillar_cosdh_v2xreal_vc.yaml

python opencood/tools/check_cosdh_v2xreal.py \
  --hypes_yaml opencood/hypes_yaml/v2xreal/point_pillar_cosdh_markov_v2xreal_vc.yaml
