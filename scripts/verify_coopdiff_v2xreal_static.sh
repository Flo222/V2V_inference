#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python -m py_compile \
  opencood/models/baselines/coopdiff/models/point_pillar_base_multi_scale_teacher_diff_v2xreal.py \
  opencood/models/baselines/coopdiff/models/point_pillar_diff_stu_v2xreal.py \
  opencood/models/baselines/coopdiff/models/point_pillar_diff_stu_markov_v2xreal.py \
  opencood/data_utils/datasets/intermediate_fusion_dataset_coopdiff_v2xreal.py \
  opencood/tools/check_coopdiff_v2xreal.py \
  opencood/tools/inference_coopdiff_v2xreal.py

python opencood/tools/check_coopdiff_v2xreal.py \
  --hypes_yaml opencood/hypes_yaml/v2xreal/point_pillar_diff_teacher_v2xreal_vc.yaml

python opencood/tools/check_coopdiff_v2xreal.py \
  --hypes_yaml opencood/hypes_yaml/v2xreal/point_pillar_diff_student_v2xreal_vc.yaml

python opencood/tools/check_coopdiff_v2xreal.py \
  --hypes_yaml opencood/hypes_yaml/v2xreal/point_pillar_diff_student_markov_v2xreal_vc.yaml
