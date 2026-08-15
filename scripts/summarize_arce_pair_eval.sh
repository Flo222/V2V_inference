#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TAG="${TAG:-payload_aligned_eval}"
OUT_ROOT="${OUT_ROOT:-outputs}"

ARCE_OUT_DIR="${ARCE_OUT_DIR:-${OUT_ROOT}/arce_c2mab/${TAG}}"
BASELINE_OUT_DIR="${BASELINE_OUT_DIR:-${OUT_ROOT}/baselines/where2comm_arce_fixed/${TAG}_compare}"

export ARCE_OUT_DIR
export BASELINE_OUT_DIR

python - <<'PY'
import json
import os
from pathlib import Path

items = {
    "ARCE-C2MAB": Path(os.environ["ARCE_OUT_DIR"]),
    "Where2Comm-ARCE-Fixed": Path(os.environ["BASELINE_OUT_DIR"]),
}

print("===== Pair Summary =====")

for name, out_dir in items.items():
    print("\n===", name, "===")
    summary_path = out_dir / "final_summary.json"
    bw_path = out_dir / "bw.json"
    breakdown_path = out_dir / "bw_breakdown.json"
    byte_audit_path = out_dir / "byte_accounting_audit.json"

    if not summary_path.exists():
        print("missing:", summary_path)
        continue

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if bw_path.exists():
        bw = json.loads(bw_path.read_text(encoding="utf-8"))
        print("BW:", bw.get("BW"))
        print("total_tx_MB:", bw.get("total_tx_MB"))
        print("record_count:", bw.get("record_count"))
        print("transmitted_link_count:", bw.get("transmitted_link_count"))
        print("no_send_count:", bw.get("no_send_count"))
        print("reward_runtime_audit_json:", bw.get("reward_runtime_audit_json"))

    if breakdown_path.exists():
        br = json.loads(breakdown_path.read_text(encoding="utf-8"))
        counter = br.get("counter", {}) or {}
        print("avg_tokens_per_token_record:", br.get("avg_tokens_per_token_record"))
        print("avg_tx_bytes_per_token:", br.get("avg_tx_bytes_per_token"))
        print("quant_counter:", counter.get("quant_mode"))
        print("rho_counter:", counter.get("rho"))
        print("bad_legacy_action_ids:", br.get("bad_legacy_action_ids"))

    if byte_audit_path.exists():
        ba = json.loads(byte_audit_path.read_text(encoding="utf-8"))
        print("byte_audit_rows:", len(ba.get("rows", [])))
        print("mismatch_counts:", ba.get("mismatch_counts"))
PY
