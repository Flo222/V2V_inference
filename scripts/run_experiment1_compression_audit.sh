#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR to the existing trained Where2Comm model directory}"
OUT_ROOT="${OUT_ROOT:-audit_runs/experiment1_compression_clean}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-2026}"
SAVE_TENSORS="${SAVE_TENSORS:-1}"
SAVE_FIRST_N_LINKS="${SAVE_FIRST_N_LINKS:-12}"
QUANT_MODES="${QUANT_MODES:-fp32 fp16 int8 int4}"
UNLIMITED_BUDGET_MBPS="${UNLIMITED_BUDGET_MBPS:-100000.0}"

MODEL_DIR="$(realpath "$MODEL_DIR")"
mkdir -p "$OUT_ROOT"
OUT_ROOT="$(realpath "$OUT_ROOT")"

if [ ! -f "$MODEL_DIR/config.yaml" ]; then
  echo "Missing $MODEL_DIR/config.yaml" >&2
  exit 1
fi

make_runtime_dir() {
  local mode="$1"
  local mode_dir="$OUT_ROOT/$mode"
  local runtime_dir="$mode_dir/model_runtime"
  local audit_dir="$mode_dir/audit"

  rm -rf "$mode_dir"
  mkdir -p "$runtime_dir" "$audit_dir" "$mode_dir/comm_logs"

  # Link checkpoints and other model-root files, but never touch the original.
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
    "$SAVE_TENSORS" "$SAVE_FIRST_N_LINKS" "$UNLIMITED_BUDGET_MBPS" <<'PY'
import copy
import os
import sys
import yaml

src, dst, mode, audit_dir, save_tensors, save_first_n, budget_mbps = sys.argv[1:]
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

# Set both the action and the base quantizer mode. This makes the requested
# precision explicit without changing the original model config or core output.
arce["fixed_action"] = {
    "send": 1,
    "quant": mode,
    "quant_mode": mode,
    "rho": 0.0,
    "redundancy_ratio": 0.0,
    "cache": 0,
    "cache_enabled": 0,
    "fec_type": "none",
    "action_id": "experiment1_%s_rho0_cache0" % mode,
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

# Experiment 1: no random loss and no budget truncation.
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

scheduler = copy.deepcopy(arce.get("scheduler", {}) or {})
scheduler.update({
    "budget_source": "system_budget",
    "budget_scope": "system_equal_split",
    "system_budget_mbps": float(budget_mbps),
    "total_budget_mbps": float(budget_mbps),
    "tx_window_ms": float(scheduler.get("tx_window_ms", 100.0)),
})
arce["scheduler"] = scheduler

# Disable all temporal/cache paths for this pure compression test.
delay = copy.deepcopy(arce.get("delay", {}) or {})
delay["policy_by_state"] = {"good": "current", "medium": "current", "bad": "current"}
arce["delay"] = delay

arce["compression_audit"] = {
    "enabled": True,
    "strict": True,
    "output_dir": os.path.abspath(audit_dir),
    "file_name": "compression_audit.jsonl",
    "save_tensors": str(save_tensors).lower() in ("1", "true", "yes", "on"),
    "save_first_n_links": int(save_first_n),
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
  echo "===== Experiment 1: $mode ====="
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
    --comm_prefix "experiment1_${mode}" \
    --save_eval_json \
    2>&1 | tee "$mode_dir/inference.log"
}

for mode in $QUANT_MODES; do
  case "$mode" in
    fp32|fp16|int8|int4) make_runtime_dir "$mode" ;;
    *) echo "Unsupported mode in QUANT_MODES: $mode" >&2; exit 1 ;;
  esac
done

python scripts/summarize_experiment1_compression_audit.py \
  --root "$OUT_ROOT" \
  --modes $QUANT_MODES \
  --strict

echo
echo "Experiment 1 finished: $OUT_ROOT"
