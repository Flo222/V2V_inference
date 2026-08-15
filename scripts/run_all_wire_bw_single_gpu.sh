#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATASET="${DATASET:-opv2v}"
GPU_ID="${GPU_ID:-0}"
MAX_FRAMES="${MAX_FRAMES:-200}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/audit_runs/wire_bw_all/${DATASET}_n${MAX_FRAMES}}"

PACKET_SIZE_BYTES="${PACKET_SIZE_BYTES:-1024}"
SPARSE_METADATA="${SPARSE_METADATA:-indices}"
SPARSE_INDEX_BYTES="${SPARSE_INDEX_BYTES:-4}"
BYTES_PER_VALUE="${BYTES_PER_VALUE:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-2026}"
ALLOW_POLICY="${ALLOW_POLICY:-0}"
TEST_DIR="${TEST_DIR:-}"

mkdir -p "${OUT_ROOT}/logs"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

run_one() {
    local baseline="$1"
    local ideal_dir="$2"
    local markov_dir="$3"
    local ideal_epoch="$4"
    local markov_epoch="$5"

    local baseline_out="${OUT_ROOT}/${baseline}"
    local log_file="${OUT_ROOT}/logs/${baseline}.log"

    echo
    echo "============================================================"
    echo "Running baseline: ${baseline}"
    echo "GPU: ${GPU_ID}"
    echo "Ideal model: ${ideal_dir}"
    echo "Markov model: ${markov_dir}"
    echo "Output: ${baseline_out}"
    echo "============================================================"

    if [[ ! -f "${ideal_dir}/config.yaml" ]]; then
        echo "[ERROR] Missing ${ideal_dir}/config.yaml" | tee "${log_file}"
        return 1
    fi

    if [[ ! -f "${markov_dir}/config.yaml" ]]; then
        echo "[ERROR] Missing ${markov_dir}/config.yaml" | tee "${log_file}"
        return 1
    fi

    IDEAL_EPOCH="${ideal_epoch}" \
    MARKOV_EPOCH="${markov_epoch}" \
    TEST_DIR="${TEST_DIR}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    SEED="${SEED}" \
    PACKET_SIZE_BYTES="${PACKET_SIZE_BYTES}" \
    SPARSE_METADATA="${SPARSE_METADATA}" \
    SPARSE_INDEX_BYTES="${SPARSE_INDEX_BYTES}" \
    BYTES_PER_VALUE="${BYTES_PER_VALUE}" \
    ALLOW_POLICY="${ALLOW_POLICY}" \
    bash scripts/run_wire_bw_pair.sh \
        "${baseline}" \
        "${DATASET}" \
        "${ideal_dir}" \
        "${markov_dir}" \
        "${baseline_out}" \
        "${MAX_FRAMES}" \
        2>&1 | tee "${log_file}"

    local status=${PIPESTATUS[0]}

    if [[ ${status} -eq 0 ]]; then
        echo "[PASS] ${baseline}"
    else
        echo "[FAIL] ${baseline}, exit status=${status}"
    fi

    # 清理上一个模型进程遗留的缓存引用
    python - <<'PY' >/dev/null 2>&1 || true
try:
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except Exception:
    pass
PY

    return "${status}"
}

declare -A STATUS

run_one \
    where2comm \
    "${WHERE2COMM_IDEAL_DIR:?Missing WHERE2COMM_IDEAL_DIR}" \
    "${WHERE2COMM_MARKOV_DIR:?Missing WHERE2COMM_MARKOV_DIR}" \
    "${WHERE2COMM_IDEAL_EPOCH:-auto}" \
    "${WHERE2COMM_MARKOV_EPOCH:-auto}"
STATUS[where2comm]=$?

run_one \
    v2xvit \
    "${V2XVIT_IDEAL_DIR:?Missing V2XVIT_IDEAL_DIR}" \
    "${V2XVIT_MARKOV_DIR:?Missing V2XVIT_MARKOV_DIR}" \
    "${V2XVIT_IDEAL_EPOCH:-auto}" \
    "${V2XVIT_MARKOV_EPOCH:-auto}"
STATUS[v2xvit]=$?

run_one \
    cosdh \
    "${COSDH_IDEAL_DIR:?Missing COSDH_IDEAL_DIR}" \
    "${COSDH_MARKOV_DIR:?Missing COSDH_MARKOV_DIR}" \
    "${COSDH_IDEAL_EPOCH:-auto}" \
    "${COSDH_MARKOV_EPOCH:-auto}"
STATUS[cosdh]=$?

run_one \
    rocooper \
    "${ROCOOPER_IDEAL_DIR:?Missing ROCOOPER_IDEAL_DIR}" \
    "${ROCOOPER_MARKOV_DIR:?Missing ROCOOPER_MARKOV_DIR}" \
    "${ROCOOPER_IDEAL_EPOCH:-auto}" \
    "${ROCOOPER_MARKOV_EPOCH:-auto}"
STATUS[rocooper]=$?

run_one \
    coopdiff \
    "${COOPDIFF_IDEAL_DIR:?Missing COOPDIFF_IDEAL_DIR}" \
    "${COOPDIFF_MARKOV_DIR:?Missing COOPDIFF_MARKOV_DIR}" \
    "${COOPDIFF_IDEAL_EPOCH:-auto}" \
    "${COOPDIFF_MARKOV_EPOCH:-auto}"
STATUS[coopdiff]=$?

echo
echo "============================================================"
echo "Generating unified summary"
echo "============================================================"

python scripts/summarize_all_wire_bw.py \
    --input_root "${OUT_ROOT}" \
    --output_csv "${OUT_ROOT}/all_baselines_wire_bw_summary.csv" \
    --output_json "${OUT_ROOT}/all_baselines_wire_bw_summary.json" \
    --output_md "${OUT_ROOT}/all_baselines_wire_bw_summary.md"

summary_status=$?

echo
echo "===================== RUN STATUS ============================"
for baseline in where2comm v2xvit cosdh rocooper coopdiff; do
    if [[ "${STATUS[$baseline]}" -eq 0 ]]; then
        printf "%-12s PASS\n" "${baseline}"
    else
        printf "%-12s FAIL (%s)\n" "${baseline}" "${STATUS[$baseline]}"
    fi
done

if [[ ${summary_status} -eq 0 ]]; then
    echo "summary      PASS"
else
    echo "summary      FAIL (${summary_status})"
fi

echo
echo "Unified CSV:"
echo "${OUT_ROOT}/all_baselines_wire_bw_summary.csv"

echo
echo "Unified Markdown:"
echo "${OUT_ROOT}/all_baselines_wire_bw_summary.md"

# 只要有一个基线失败，脚本整体返回非0
for baseline in where2comm v2xvit cosdh rocooper coopdiff; do
    if [[ "${STATUS[$baseline]}" -ne 0 ]]; then
        exit 1
    fi
done

exit "${summary_status}"
