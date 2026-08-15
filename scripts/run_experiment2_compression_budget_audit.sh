#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR to the existing trained Where2Comm model directory}"
OUT_ROOT="${OUT_ROOT:-audit_runs/experiment2_compression_budget}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-2026}"
SAVE_TENSORS="${SAVE_TENSORS:-1}"
SAVE_FIRST_N_LINKS="${SAVE_FIRST_N_LINKS:-12}"
QUANT_MODES="${QUANT_MODES:-fp32 fp16 int8 int4}"
SYSTEM_BUDGET_MBPS="${SYSTEM_BUDGET_MBPS:-300.0}"
TX_WINDOW_MS="${TX_WINDOW_MS:-100.0}"

MODEL_DIR="$(realpath "$MODEL_DIR")"
mkdir -p "$OUT_ROOT"
OUT_ROOT="$(realpath "$OUT_ROOT")"

if [ ! -f "$MODEL_DIR/config.yaml" ]; then
  echo "Missing $MODEL_DIR/config.yaml" >&2
  exit 1
fi

python - "$SYSTEM_BUDGET_MBPS" "$TX_WINDOW_MS" <<'PY'
import sys
mbps = float(sys.argv[1])
window_ms = float(sys.argv[2])
frame_bytes = mbps * 1_000_000.0 / 8.0 * window_ms / 1000.0
print("Experiment-2 fixed total frame budget: %.3f MB/frame (%g Mbps, %g ms)" % (
    frame_bytes / 1_000_000.0, mbps, window_ms
))
print("The frame budget is split equally across all non-ego collaborators.")
PY

make_runtime_dir() {
  local mode="$1"
  local mode_dir="$OUT_ROOT/$mode"
  local runtime_dir="$mode_dir/model_runtime"
  local audit_dir="$mode_dir/audit"

  rm -rf "$mode_dir"
  mkdir -p "$runtime_dir" "$audit_dir" "$mode_dir/comm_logs"

  # Checkpoints are read-only links. The trained model directory is untouched.
  while IFS= read -r -d '' file; do
    local base
    base="$(basename "$file")"
    if [ "$base" != "config.yaml" ]; then
      ln -s "$file" "$runtime_dir/$base"
    fi
  done < <(find "$MODEL_DIR" -maxdepth 1 \( -type f -o -type l \) -print0)

  # Never run an audit with randomly initialized model weights.
  if ! compgen -G "$runtime_dir/net_epoch*.pth" > /dev/null; then
    echo "ERROR: no net_epoch*.pth linked into $runtime_dir" >&2
    echo "Source model directory: $MODEL_DIR" >&2
    exit 1
  fi

  echo "Linked checkpoint:"
  ls -lh "$runtime_dir"/net_epoch*.pth

  python - "$MODEL_DIR/config.yaml" "$runtime_dir/config.yaml" "$mode" "$audit_dir" \
    "$SAVE_TENSORS" "$SAVE_FIRST_N_LINKS" "$SYSTEM_BUDGET_MBPS" "$TX_WINDOW_MS" <<'PY'
import copy
import os
import sys
import yaml

src, dst, mode, audit_dir, save_tensors, save_first_n, budget_mbps, tx_window_ms = sys.argv[1:]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

model = cfg.setdefault("model", {})
args = model.setdefault("args", {})
arce = copy.deepcopy(args.get("arce", cfg.get("arce", {})) or {})

arce["enabled"] = True
arce["mode"] = "fixed"
arce["policy"] = "fixed"
arce["link_scope"] = "non_ego"
arce["keep_tensor_results"] = False

# Fixed action: compression only. No redundancy and no temporal cache.
arce["fixed_action"] = {
    "send": 1,
    "quant": mode,
    "quant_mode": mode,
    "rho": 0.0,
    "redundancy_ratio": 0.0,
    "cache": 0,
    "cache_enabled": 0,
    "fec_type": "none",
    "action_id": "experiment2_%s_rho0_cache0" % mode,
}
quant_cfg = copy.deepcopy(arce.get("quantization", {}) or {})
quant_cfg.update({
    "enabled": mode != "fp32",
    "mode": mode,
    "granularity": "per_tensor",
    "compute_error": True,
    "pack_int4": mode == "int4",
})
arce["quantization"] = quant_cfg

# Experiment 2 isolates budget truncation: all channel loss and delay are off.
channel = copy.deepcopy(arce.get("channel", {}) or {})
channel["mode"] = "fixed"
channel["bernoulli_loss_rates"] = {"good": 0.0, "medium": 0.0, "bad": 0.0}
channel["fixed_delay_ms"] = {"good": 0.0, "medium": 0.0, "bad": 0.0}
profiles = copy.deepcopy(channel.get("profiles", {}) or {})
for state in ("good", "medium", "bad"):
    profile = copy.deepcopy(profiles.get(state, {}) or {})
    profile.update({
        "bandwidth_mbps": float(budget_mbps),
        "loss_rate": 0.0,
        "plr": 0.0,
        "delay_ms": 0.0,
    })
    profiles[state] = profile
channel["profiles"] = profiles
arce["channel"] = channel

# A fixed total frame budget is shared equally by all non-ego collaborators.
scheduler = copy.deepcopy(arce.get("scheduler", {}) or {})
scheduler.update({
    "budget_source": "system_budget",
    "budget_scope": "system_equal_split",
    "system_budget_mbps": float(budget_mbps),
    "total_budget_mbps": float(budget_mbps),
    "tx_window_ms": float(tx_window_ms),
})
arce["scheduler"] = scheduler

# Current-frame feature only; no cache can hide packet truncation.
delay = copy.deepcopy(arce.get("delay", {}) or {})
delay["policy_by_state"] = {"good": "current", "medium": "current", "bad": "current"}
arce["delay"] = delay

arce["compression_audit"] = {
    "enabled": True,
    "strict": True,
    "experiment_name": "experiment2_limited_budget_no_loss_no_fec",
    "output_dir": os.path.abspath(audit_dir),
    "file_name": "compression_budget_audit.jsonl",
    "save_tensors": str(save_tensors).lower() in ("1", "true", "yes", "on"),
    "save_first_n_links": int(save_first_n),
    # Budget drops and F_quant != F_recv are expected in this experiment.
    "require_no_budget_drop": False,
    "require_no_bernoulli_loss": True,
    "require_no_fec_parity": True,
    "require_all_source_transmitted": False,
    "require_quant_equals_recovered": False,
}

args["arce"] = copy.deepcopy(arce)
cfg["arce"] = copy.deepcopy(arce)

wild = cfg.setdefault("wild_setting", {})
markov = copy.deepcopy(wild.get("channel_state_markov", {}) or {})
markov["enabled"] = False
wild["channel_state_markov"] = markov

with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
print("runtime config:", dst)
PY

  echo
  echo "===== Experiment 2: $mode ====="
  echo "runtime model: $runtime_dir"
  echo "audit output: $audit_dir"
  PYTHONUNBUFFERED=1 python opencood/tools/inference_arce.py \
    --model_dir "$runtime_dir" \
    --fusion_method intermediate \
    --max_samples "$MAX_SAMPLES" \
    --num_workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --save_comm \
    --comm_log_dir "$mode_dir/comm_logs" \
    --comm_prefix "experiment2_${mode}" \
    --save_eval_json \
    2>&1 | tee "$mode_dir/inference.log"
}

for mode in $QUANT_MODES; do
  case "$mode" in
    fp32|fp16|int8|int4) make_runtime_dir "$mode" ;;
    *) echo "Unsupported mode in QUANT_MODES: $mode" >&2; exit 1 ;;
  esac
done

python scripts/summarize_experiment2_compression_budget_audit.py \
  --root "$OUT_ROOT" \
  --modes $QUANT_MODES \
  --strict

echo
echo "Experiment 2 finished: $OUT_ROOT"
