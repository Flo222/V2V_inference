#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR to the trained Where2Comm model directory}"
OUT_ROOT="${OUT_ROOT:-audit_runs/experiment4_joint_compression_redundancy}"
MAX_SAMPLES="${MAX_SAMPLES:-20}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-2026}"
SAVE_TENSORS="${SAVE_TENSORS:-1}"
SAVE_FIRST_N_LINKS="${SAVE_FIRST_N_LINKS:-12}"
QUANT_MODES="${QUANT_MODES:-fp16 int8 int4}"
PLRS="${PLRS:-0.20 0.35}"
RHOS="${RHOS:-0 0.25 0.60}"
SYSTEM_BUDGET_MBPS="${SYSTEM_BUDGET_MBPS:-100}"
TX_WINDOW_MS="${TX_WINDOW_MS:-100}"

# Keep independent conditions deterministic so source payload, source budget
# selection, and source Bernoulli loss are paired across rho.
export PYTHONHASHSEED="${PYTHONHASHSEED:-$SEED}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

MODEL_DIR="$(realpath "$MODEL_DIR")"
mkdir -p "$OUT_ROOT"
OUT_ROOT="$(realpath "$OUT_ROOT")"

if [ ! -f "$MODEL_DIR/config.yaml" ]; then
  echo "Missing $MODEL_DIR/config.yaml" >&2
  exit 1
fi

for quant in $QUANT_MODES; do
  case "$quant" in
    fp32|fp16|int8|int4) ;;
    *) echo "Unsupported quant mode: $quant" >&2; exit 1 ;;
  esac
done

condition_name() {
  python - "$1" "$2" "$3" <<'PY'
import sys
plr = float(sys.argv[1]); quant = sys.argv[2]; rho = float(sys.argv[3])
def t(x): return ("%.2f" % x).replace(".", "p")
print("plr_%s_quant_%s_rho_%s" % (t(plr), quant, t(rho)))
PY
}

make_runtime_dir() {
  local plr="$1"
  local quant="$2"
  local rho="$3"
  local condition
  condition="$(condition_name "$plr" "$quant" "$rho")"
  local condition_dir="$OUT_ROOT/$condition"
  local runtime_dir="$condition_dir/model_runtime"
  local audit_dir="$condition_dir/audit"

  rm -rf "$condition_dir"
  mkdir -p "$runtime_dir" "$audit_dir" "$condition_dir/comm_logs"

  while IFS= read -r -d '' file; do
    local base
    base="$(basename "$file")"
    if [ "$base" != "config.yaml" ]; then
      ln -s "$file" "$runtime_dir/$base"
    fi
  done < <(find "$MODEL_DIR" -maxdepth 1 \( -type f -o -type l \) -print0)

  if ! compgen -G "$runtime_dir/net_epoch*.pth" > /dev/null; then
    echo "ERROR: no net_epoch*.pth linked into $runtime_dir" >&2
    exit 1
  fi

  python - "$MODEL_DIR/config.yaml" "$runtime_dir/config.yaml" "$quant" \
    "$plr" "$rho" "$audit_dir" "$SAVE_TENSORS" "$SAVE_FIRST_N_LINKS" \
    "$SYSTEM_BUDGET_MBPS" "$TX_WINDOW_MS" "$SEED" <<'PY'
import copy
import os
import sys
import yaml

(src, dst, quant_mode, plr, rho, audit_dir, save_tensors, save_first_n,
 budget_mbps, tx_window_ms, seed) = sys.argv[1:]
plr = float(plr); rho = float(rho); budget_mbps = float(budget_mbps)
tx_window_ms = float(tx_window_ms); seed = int(seed)

with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

model = cfg.setdefault("model", {})
args = model.setdefault("args", {})
arce = copy.deepcopy(args.get("arce", cfg.get("arce", {})) or {})
arce.update({
    "enabled": True,
    "mode": "fixed",
    "policy": "fixed",
    "link_scope": "non_ego",
    "keep_tensor_results": False,
    "seed": seed,
})

fec_type = "none" if rho <= 0 else "raptor_sim"
arce["fixed_action"] = {
    "send": 1,
    "quant": quant_mode,
    "quant_mode": quant_mode,
    "rho": rho,
    "redundancy_ratio": rho,
    "cache": 0,
    "cache_enabled": 0,
    "fec_type": fec_type,
    "decode_overhead": 0.0,
    "action_id": "experiment4_%s_plr%.2f_rho%.2f" % (quant_mode, plr, rho),
}

quant = copy.deepcopy(arce.get("quantization", {}) or {})
quant.update({
    "enabled": quant_mode != "fp32",
    "mode": quant_mode,
    "granularity": "per_tensor",
    "compute_error": True,
    "pack_int4": quant_mode == "int4",
})
arce["quantization"] = quant

fec = copy.deepcopy(arce.get("fec", {}) or {})
fec.update({
    "enabled": rho > 0,
    "type": fec_type,
    "default_type": "raptor_sim",
    "redundancy_ratio": rho,
    "decode_overhead": 0.0,
    "degree_distribution": "robust_soliton",
    "robust_soliton_c": 0.1,
    "robust_soliton_delta": 0.5,
    "seed": seed,
})
arce["fec"] = fec

channel = copy.deepcopy(arce.get("channel", {}) or {})
channel["mode"] = "fixed"
channel["bernoulli_loss_rates"] = {"good": plr, "medium": plr, "bad": plr}
channel["fixed_delay_ms"] = {"good": 0.0, "medium": 0.0, "bad": 0.0}
profiles = copy.deepcopy(channel.get("profiles", {}) or {})
for state in ("good", "medium", "bad"):
    p = copy.deepcopy(profiles.get(state, {}) or {})
    p.update({
        "bandwidth_mbps": budget_mbps,
        "loss_rate": plr,
        "plr": plr,
        "delay_ms": 0.0,
    })
    profiles[state] = p
channel["profiles"] = profiles
arce["channel"] = channel

scheduler = copy.deepcopy(arce.get("scheduler", {}) or {})
scheduler.update({
    "budget_source": "system_budget",
    "budget_scope": "system_equal_split",
    "system_budget_mbps": budget_mbps,
    "total_budget_mbps": budget_mbps,
    "tx_window_ms": tx_window_ms,
})
arce["scheduler"] = scheduler

delay = copy.deepcopy(arce.get("delay", {}) or {})
delay["policy_by_state"] = {"good": "current", "medium": "current", "bad": "current"}
arce["delay"] = delay

# Disable prior compression audit. Reuse the read-only FEC auditor with finite
# budget requirements relaxed so it can decompose budget, channel, and FEC.
arce["compression_audit"] = {"enabled": False}
arce["fec_recovery_audit"] = {
    "enabled": True,
    "strict": True,
    "experiment_name": "experiment4_joint_compression_redundancy",
    "output_dir": os.path.abspath(audit_dir),
    "file_name": "joint_audit.jsonl",
    "save_tensors": str(save_tensors).lower() in ("1", "true", "yes", "on"),
    "save_first_n_links": int(save_first_n),
    "require_no_budget_drop": False,
    "require_all_encoded_transmitted": False,
    "require_budget_not_exceeded": True,
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
  echo "===== Experiment 4: PLR=$plr quant=$quant rho=$rho ====="
  echo "Finite system budget: $SYSTEM_BUDGET_MBPS Mbps x $TX_WINDOW_MS ms"
  ls -lh "$runtime_dir"/net_epoch*.pth
  PYTHONUNBUFFERED=1 python opencood/tools/inference_arce.py \
    --model_dir "$runtime_dir" \
    --fusion_method intermediate \
    --max_samples "$MAX_SAMPLES" \
    --num_workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --save_comm \
    --comm_log_dir "$condition_dir/comm_logs" \
    --comm_prefix "experiment4_${condition}" \
    --save_eval_json \
    2>&1 | tee "$condition_dir/inference.log"
}

for plr in $PLRS; do
  for quant in $QUANT_MODES; do
    for rho in $RHOS; do
      make_runtime_dir "$plr" "$quant" "$rho"
    done
  done
done

python scripts/summarize_experiment4_joint_compression_redundancy_audit.py \
  --root "$OUT_ROOT" --plrs $PLRS --quant-modes $QUANT_MODES --rhos $RHOS --strict

echo
echo "Experiment 4 finished: $OUT_ROOT"
