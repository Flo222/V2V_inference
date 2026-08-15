#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  cat >&2 <<'EOF'
Usage:
  bash scripts/run_wire_bw_pair.sh \
    BASELINE DATASET IDEAL_MODEL_DIR MARKOV_MODEL_DIR OUT_ROOT [MAX_FRAMES]

BASELINE: where2comm | v2xvit | rocooper | coopdiff | cosdh
DATASET : opv2v | v2xreal

Optional environment variables:
  IDEAL_EPOCH=auto
  MARKOV_EPOCH=auto
  TEST_DIR=/path/to/test
  NUM_WORKERS=0
  SEED=2026
  PACKET_SIZE_BYTES=1024
  SPARSE_METADATA=indices   # none | indices | bitmask
  SPARSE_INDEX_BYTES=4
  BYTES_PER_VALUE=0         # 0 = tensor.element_size()
  ALLOW_POLICY=0
EOF
  exit 2
fi

BASELINE=$1
DATASET=$2
IDEAL_MODEL_DIR=$3
MARKOV_MODEL_DIR=$4
OUT_ROOT=$5
MAX_FRAMES=${6:-200}

IDEAL_EPOCH=${IDEAL_EPOCH:-auto}
MARKOV_EPOCH=${MARKOV_EPOCH:-auto}
NUM_WORKERS=${NUM_WORKERS:-0}
SEED=${SEED:-2026}
PACKET_SIZE_BYTES=${PACKET_SIZE_BYTES:-1024}
SPARSE_METADATA=${SPARSE_METADATA:-indices}
SPARSE_INDEX_BYTES=${SPARSE_INDEX_BYTES:-4}
BYTES_PER_VALUE=${BYTES_PER_VALUE:-0}
TEST_DIR=${TEST_DIR:-}
ALLOW_POLICY=${ALLOW_POLICY:-0}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

COMMON=(
  --dataset "$DATASET"
  --baseline "$BASELINE"
  --max_frames "$MAX_FRAMES"
  --num_workers "$NUM_WORKERS"
  --seed "$SEED"
  --packet_size_bytes "$PACKET_SIZE_BYTES"
  --sparse_metadata "$SPARSE_METADATA"
  --sparse_index_bytes "$SPARSE_INDEX_BYTES"
  --bytes_per_value "$BYTES_PER_VALUE"
)

if [[ -n "$TEST_DIR" ]]; then
  COMMON+=(--test_dir "$TEST_DIR")
fi
if [[ "$ALLOW_POLICY" == "1" ]]; then
  COMMON+=(--allow_policy)
fi

mkdir -p "$OUT_ROOT"

python scripts/wire_bw_audit.py \
  --name "${DATASET}_${BASELINE}_ideal" \
  --channel_mode ideal \
  --model_dir "$IDEAL_MODEL_DIR" \
  --epoch "$IDEAL_EPOCH" \
  --out_dir "$OUT_ROOT/ideal" \
  "${COMMON[@]}"

python scripts/wire_bw_audit.py \
  --name "${DATASET}_${BASELINE}_markov" \
  --channel_mode markov \
  --model_dir "$MARKOV_MODEL_DIR" \
  --epoch "$MARKOV_EPOCH" \
  --out_dir "$OUT_ROOT/markov" \
  "${COMMON[@]}"

python scripts/summarize_wire_bw_pairs.py \
  --input_root "$OUT_ROOT" \
  --output_csv "$OUT_ROOT/wire_bw_pair_summary.csv" \
  --output_json "$OUT_ROOT/wire_bw_pair_summary.json"

echo "Done: $OUT_ROOT/wire_bw_pair_summary.csv"
