#!/usr/bin/env bash
set -euo pipefail

# Run this script from the unpacked patch directory, pass your OPV2V repo root.
# Example:
#   bash scripts/install_where2comm_v2xreal_patch.sh ~/v2x_projects/OPV2V

if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/install_where2comm_v2xreal_patch.sh /path/to/OPV2V" >&2
  exit 1
fi

REPO_ROOT="$1"
PATCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for p in \
  opencood/models/baselines/where2comm/point_pillar_where2comm_v2xreal.py \
  opencood/models/baselines/where2comm/point_pillar_where2comm_arce_v2xreal.py \
  opencood/data_utils/datasets/basedataset_v2xreal.py \
  opencood/data_utils/datasets/intermediate_fusion_dataset_v2xreal.py \
  opencood/hypes_yaml/v2xreal/point_pillar_where2comm_v2xreal_vc.yaml \
  opencood/hypes_yaml/v2xreal/point_pillar_where2comm_arce_c2mab_v2xreal_vc.yaml \
  opencood/tools/inference_v2xreal.py \
  opencood/tools/check_where2comm_v2xreal.py; do
  mkdir -p "${REPO_ROOT}/$(dirname "$p")"
  cp -av "${PATCH_ROOT}/${p}" "${REPO_ROOT}/${p}"
done

echo "Installed Where2Comm V2X-Real patch into ${REPO_ROOT}"
