#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate opencood

export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

RUN_STAMP="${RUN_STAMP:?RUN_STAMP is required}"

CFG="opencood/logs/main_opv2v_where2comm_grace_full/config.yaml"
BACKUP_DIR="refactor_backups/importance_i0_i1_full_${RUN_STAMP}"
mkdir -p "$BACKUP_DIR"
cp "$CFG" "$BACKUP_DIR/config.yaml"

restore_config() {
    cp "$BACKUP_DIR/config.yaml" "$CFG"
    echo "Original config restored: $CFG"
}

trap restore_config EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

set_variant() {
    VARIANT="$1" python - <<'PY'
import os
import yaml
from pathlib import Path

variant = os.environ["VARIANT"]
if variant not in {"I0", "I1"}:
    raise ValueError(variant)

p = Path(
    "opencood/logs/main_opv2v_where2comm_grace_full/config.yaml"
)
cfg = yaml.load(
    p.read_text(encoding="utf-8"),
    Loader=yaml.Loader,
)

arce = cfg["model"]["args"]["arce"]

spatial = arce.setdefault("spatial_importance", {})
spatial["enabled"] = True
spatial["method"] = "feature_rms"
spatial["normalization"] = "max"

compact = arce.setdefault("compact_sparse", {})
compact["enabled"] = True
compact["source"] = "arce_sender_feature"
compact["priority_layout_enabled"] = True
compact["payload_layout"] = "KC"
compact["sort_by_score"] = variant == "I1"

zero_codec = arce.setdefault("zero_codec", {})
zero_codec["enabled"] = False

p.write_text(
    yaml.dump(
        cfg,
        sort_keys=False,
        allow_unicode=True,
    ),
    encoding="utf-8",
)

print("variant:", variant)
print("spatial_importance.enabled:", spatial["enabled"])
print("compact_sparse.sort_by_score:", compact["sort_by_score"])
print("zero_codec.enabled:", zero_codec["enabled"])
PY
}

echo "========================================"
echo "Run I0: ARCE candidates, original order"
echo "========================================"

set_variant I0

METHOD=arce \
RUN_AP=1 \
RUN_BW=1 \
MAX_FRAMES=-1 \
TAG="importance_I0_original_order_full_${RUN_STAMP}" \
bash scripts/run_arce_pair_eval.sh

echo "========================================"
echo "Run I1: ARCE candidates, RMS order"
echo "========================================"

set_variant I1

METHOD=arce \
RUN_AP=1 \
RUN_BW=1 \
MAX_FRAMES=-1 \
TAG="importance_I1_rms_order_full_${RUN_STAMP}" \
bash scripts/run_arce_pair_eval.sh

echo "========================================"
echo "I0 and I1 full evaluations completed"
echo "========================================"
