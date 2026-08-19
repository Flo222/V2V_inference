#!/usr/bin/env bash
# Launch V2X-Real only after the preceding OPV2V compression fine-tune
# completed successfully.  This keeps the two experiments sequential on GPU 0.
set -u

# The conda activation hook appends to PYTHONPATH.  Initialise it explicitly
# because nounset would otherwise reject an unset inherited environment.
export PYTHONPATH="${PYTHONPATH:-}"

PROJECT=/home/server/v2x_projects/V2V_inference
OPV_DIR=$PROJECT/opencood/logs/point_pillar_v2xvit_opv2v_compression32_perfect_ft
V2XR_DIR=$PROJECT/opencood/logs/point_pillar_v2xvit_v2xreal_compression32_perfect_ft
STATUS_LOG=$V2XR_DIR/launch_after_opv2v.log

while pgrep -f 'opencood/tools/train.py.*point_pillar_v2xvit_opv2v_compression32_perfect_ft' >/dev/null; do
    sleep 30
done

if ! test -f "$OPV_DIR/net_epoch22.pth" || ! grep -q 'Training Finished' "$OPV_DIR/train_run.log"; then
    echo "OPV2V did not complete successfully; V2X-Real was not started." >> "$STATUS_LOG"
    exit 1
fi

echo "OPV2V completed; starting V2X-Real at $(date -Is)." >> "$STATUS_LOG"
source /home/server/anaconda3/etc/profile.d/conda.sh
conda activate opencood
cd "$PROJECT"
exec env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 \
    python opencood/tools/train.py \
    --hypes_yaml opencood/logs/point_pillar_v2xvit_v2xreal_compression32_perfect_ft/config.yaml \
    --model_dir opencood/logs/point_pillar_v2xvit_v2xreal_compression32_perfect_ft \
    >> "$V2XR_DIR/train_run.log" 2>&1
