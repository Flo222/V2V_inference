#!/usr/bin/env bash
set -euo pipefail

# Run from OPV2V repo root after installing the patch.
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
python -m py_compile \
  opencood/models/baselines/where2comm/point_pillar_where2comm_v2xreal.py \
  opencood/models/baselines/where2comm/point_pillar_where2comm_arce_v2xreal.py \
  opencood/data_utils/datasets/basedataset_v2xreal.py \
  opencood/data_utils/datasets/intermediate_fusion_dataset_v2xreal.py \
  opencood/tools/inference_v2xreal.py \
  opencood/tools/check_where2comm_v2xreal.py

python opencood/tools/check_where2comm_v2xreal.py \
  --hypes_yaml opencood/hypes_yaml/v2xreal/point_pillar_where2comm_v2xreal_vc.yaml

python opencood/tools/check_where2comm_v2xreal.py \
  --hypes_yaml opencood/hypes_yaml/v2xreal/point_pillar_where2comm_arce_c2mab_v2xreal_vc.yaml
