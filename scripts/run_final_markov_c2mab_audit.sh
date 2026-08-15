#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

SOURCE_MODEL_DIR="${SOURCE_MODEL_DIR:-$ROOT_DIR/opencood/logs/point_pillar_where2comm_arce_c2mab_comp_div}"
TEST_DIR="${TEST_DIR:-$ROOT_DIR/opv2v_data_dumping/test}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/audit_runs/final_markov_c2mab_seed2026_test200}"
RUNTIME_MODEL_DIR="$OUT_ROOT/model_runtime"

METHOD_NAME="${METHOD_NAME:-Where2Comm-ARCE-C2MAB-CompDiv}"
SEED="${SEED:-2026}"
MAX_FRAMES="${MAX_FRAMES:-200}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-20}"
WINDOW_SIZE="${WINDOW_SIZE:-50}"
WINDOW_STRIDE="${WINDOW_STRIDE:-50}"
WARMUP_FRAMES="${WARMUP_FRAMES:-50}"
RESET_OUT="${RESET_OUT:-1}"
GLOBAL_SORT_DETECTIONS="${GLOBAL_SORT_DETECTIONS:-0}"
REWARD_PROFILE="${REWARD_PROFILE:-r2b}"
RUN_MODEL_PREFLIGHT="${RUN_MODEL_PREFLIGHT:-1}"

export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

if [ ! -f "$SOURCE_MODEL_DIR/config.yaml" ]; then
  echo "ERROR: missing source model config: $SOURCE_MODEL_DIR/config.yaml" >&2
  exit 2
fi
if ! find -L "$SOURCE_MODEL_DIR" -maxdepth 1 -type f -name 'net_epoch*.pth' -print -quit | grep -q .; then
  echo "ERROR: no checkpoint found in $SOURCE_MODEL_DIR" >&2
  exit 2
fi
if [ ! -d "$TEST_DIR" ]; then
  echo "ERROR: test dataset directory does not exist: $TEST_DIR" >&2
  exit 2
fi

if [ "$RESET_OUT" = "1" ] && [ -d "$OUT_ROOT" ]; then
  rm -rf "$OUT_ROOT"
fi
mkdir -p "$OUT_ROOT"

python scripts/prepare_final_markov_c2mab_model.py \
  --source-model-dir "$SOURCE_MODEL_DIR" \
  --runtime-model-dir "$RUNTIME_MODEL_DIR" \
  --test-dir "$TEST_DIR" \
  --seed "$SEED" \
  --reward-profile "$REWARD_PROFILE" \
  | tee "$OUT_ROOT/prepare_model.log"

CHECKPOINT=$(find -L "$RUNTIME_MODEL_DIR" -maxdepth 1 -type f -name 'net_epoch*.pth' | sort -V | tail -1)
echo "Linked checkpoint: $CHECKPOINT"

PREFLIGHT_ARGS=(--model-dir "$RUNTIME_MODEL_DIR")
if [ "$RUN_MODEL_PREFLIGHT" = "1" ]; then
  PREFLIGHT_ARGS+=(--build-model)
fi
python scripts/preflight_final_markov_c2mab_runtime.py "${PREFLIGHT_ARGS[@]}" \
  | tee "$OUT_ROOT/preflight_runtime.log"

echo "===== Final Markov + C2MAB online audit ====="
echo "Source model: $SOURCE_MODEL_DIR"
echo "Runtime model: $RUNTIME_MODEL_DIR"
echo "Test dir: $TEST_DIR"
echo "Output: $OUT_ROOT"
echo "Seed: $SEED"
echo "Max frames: $MAX_FRAMES"
echo "Warmup frames: $WARMUP_FRAMES"
echo "Reward profile: $REWARD_PROFILE"
echo "Markov profiles: Good=27Mbps/PLR0.05/10ms, Medium=5Mbps/PLR0.20/50ms, Bad=1Mbps/PLR0.35/100ms"

EXTRA_ARGS=()
if [ "$GLOBAL_SORT_DETECTIONS" = "1" ]; then
  EXTRA_ARGS+=(--global_sort_detections)
fi

PYTHONUNBUFFERED=1 python opencood/tools/arce_online_eval.py \
  --model_dir "$RUNTIME_MODEL_DIR" \
  --out_dir "$OUT_ROOT" \
  --method "$METHOD_NAME" \
  --scenario "Markov-27-5-1Mbps" \
  --fusion_method intermediate \
  --max_frames "$MAX_FRAMES" \
  --num_workers "$NUM_WORKERS" \
  --progress_interval "$PROGRESS_INTERVAL" \
  --window_size "$WINDOW_SIZE" \
  --window_stride "$WINDOW_STRIDE" \
  --warmup_frames "$WARMUP_FRAMES" \
  --seed "$SEED" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$OUT_ROOT/online_eval.log"

python scripts/summarize_final_markov_c2mab_audit.py \
  --out-dir "$OUT_ROOT" \
  --warmup-frames "$WARMUP_FRAMES" \
  | tee "$OUT_ROOT/final_audit.log"

echo
echo "Finished. Main outputs:"
echo "  $OUT_ROOT/final_summary.json"
echo "  $OUT_ROOT/final_markov_c2mab_audit_summary.json"
echo "  $OUT_ROOT/state_action_summary.csv"
echo "  $OUT_ROOT/frame_state_perception.csv"
echo "  $OUT_ROOT/markov_transition.csv"
echo "  $OUT_ROOT/diagnostic_links_top100.csv"
