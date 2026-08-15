#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python opencood/tools/check_rocooper_v2xreal.py \
  --hypes_yaml opencood/hypes_yaml/v2xreal/point_pillar_rocooper_v2xreal_vc.yaml
python opencood/tools/check_rocooper_v2xreal.py \
  --hypes_yaml opencood/hypes_yaml/v2xreal/point_pillar_rocooper_markov_v2xreal_vc.yaml
