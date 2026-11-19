#!/usr/bin/env bash
# tools/bench_r9700_verify_wide.sh - HIP (gfx1201 / Radeon R9700) build, parity CTest suite,
# and per-feature / combined A/B timing runner for wide verify windows (>8 rows).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build-hip-w16}"
JOBS="${STRATA_BUILD_JOBS:-$(nproc 2>/dev/null || echo 16)}"
SKIP_BUILD="${SKIP_BUILD:-0}"
CONFIG="${CONFIG:-}"
STRATA_ARGS="${STRATA_ARGS:-}"
BATCH_SLOTS="${BATCH_SLOTS:-4}"
MAX_NEW="${MAX_NEW:-128}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
LOG_DIR="${LOG_DIR:-${BUILD_DIR}/r9700_ab_logs}"

TARGETS=(
  strata
  cpu_act_quant_wide_test
  fused_gr_q8_16_parity
  qsa_scores_16_parity
  gdn_groups_split_parity
  batch_split_policy_test
  cpu_reduce_miss_parity
  pcie_overlap_graph_test
  moe_wide_bucket_parity
  mmvq_il_narrow_acc_parity
  native_wmma_gfx12_test
  batch_df2_adaptive_test
  batch_tail_skip_isolation_test
  batch_route_affinity_test
  gr_parity
  gdn_parity
  verify_batch_parity
  batch_rows_parity
  verify_parity
  native_multi_parity
  mmvq_il_parity
  iq_multi_parity
  pool_tasks_test
  batch_telemetry_test
  elementwise_parity
  fused_gr_bench
  qsa_select_bench
  b6_mmvq_bench
)

CTEST_REGEX='^(cpu_act_quant_wide_test|fused_gr_q8_16_parity_on|fused_gr_q8_16_parity_off|qsa_scores_16_parity|gdn_groups_split_parity|batch_split_policy_test|cpu_reduce_miss_parity|pcie_overlap_graph_test|moe_wide_bucket_parity|mmvq_il_narrow_acc_parity|native_wmma_gfx12_test|batch_df2_adaptive_test|batch_tail_skip_isolation_test|batch_route_affinity_test|gr_parity|gdn_parity|gdn_parity_split|verify_batch_parity|batch_rows_parity|verify_parity|native_multi_parity|mmvq_il_parity|iq_multi_parity|pool_tasks_test|batch_telemetry_test|elementwise_parity)$'

if [[ "${SKIP_BUILD}" != "1" ]]; then
  echo "=== [1/3] Configuring HIP gfx1201 wide build in ${BUILD_DIR} ==="
  cmake -S "${ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_ENABLE_CUDA=OFF \
    -DSTRATA_ENABLE_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
    -DSTRATA_WINDOW_MAX_T=16 \
    -DSTRATA_BUILD_TESTS=ON

  echo "=== [1/3] Building targets (-j ${JOBS}) ==="
  cmake --build "${BUILD_DIR}" -j "${JOBS}" --target "${TARGETS[@]}"
fi

echo "=== [2/3] Running CTest wide-verify parity suite ==="
ctest --test-dir "${BUILD_DIR}" --output-on-failure -R "${CTEST_REGEX}"

echo "=== [2b/3] Checking native_wmma_gfx12 availability on R9700 ==="
"${BUILD_DIR}/native_wmma_gfx12_test"

if [[ -z "${CONFIG}" && -z "${STRATA_ARGS}" ]]; then
  echo "=== [3/3] Neither CONFIG nor STRATA_ARGS is set; skipping model A/B sweep ==="
  echo "Batch + DFlash2 example:"
  echo "  CONFIG=/path/to/strata-model.json EXTRA_ARGS='--draft dflash2 --draft-model /path/to/df2 --draft-cap 16' $0"
  echo "Solo DFlash2 CLI example:"
  echo "  STRATA_ARGS='--native /path/to/model.gguf --draft dflash2 --draft-model /path/to/df2 --draft-cap 16 --tokens 1,2,3,4,5,6,7,8 --max-new 128 --greedy' $0"
  exit 0
fi

mkdir -p "${LOG_DIR}"
if [[ -n "${CONFIG}" ]]; then
  echo "=== [3/3] Running R9700 batch A/B timing sweep (CONFIG=${CONFIG}, slots=${BATCH_SLOTS}, max_new=${MAX_NEW}) ==="
else
  echo "=== [3/3] Running R9700 solo CLI A/B timing sweep (STRATA_ARGS=${STRATA_ARGS}) ==="
fi

run_arm() {
  local name="$1"
  shift
  local log="${LOG_DIR}/${name}.log"
  echo "--- arm: ${name} ($*) ---"
  if [[ -n "${CONFIG}" ]]; then
    env "$@" \
      BATCH_TEST_LOG="${log}.engine" \
      STRATA_BATCH_DF2_MAX_T=16 \
      STRATA_BATCH_DF2_TIMING=1 \
      STRATA_VERIFY_PROFILE=1 \
      python3 "${ROOT}/tools/batch_test.py" \
        --exe "${BUILD_DIR}/strata" \
        --config "${CONFIG}" \
        --batch "${BATCH_SLOTS}" \
        --n "${BATCH_SLOTS}" \
        --max-new "${MAX_NEW}" \
        --skip-solo \
        --extra "${EXTRA_ARGS}" \
        >"${log}" 2>&1 || {
          echo "arm ${name} failed (see ${log} and ${log}.engine)"
          return 1
        }
    grep -E '\[batch-df2-timing\]|strata batch:|tok/s' "${log}" "${log}.engine" 2>/dev/null | tail -n 10 || true
  else
    local -a strata_argv=()
    read -r -a strata_argv <<< "${STRATA_ARGS}"
    env "$@" \
      STRATA_DECODE_TIMING=1 \
      STRATA_VERIFY_PROFILE=1 \
      "${BUILD_DIR}/strata" "${strata_argv[@]}" \
        >"${log}" 2>&1 || {
          echo "arm ${name} failed (see ${log})"
          return 1
        }
    grep -E '\[batch-df2-timing\]|strata batch:|decode|tok/s' "${log}" | tail -n 10 || true
  fi
}

run_arm baseline \
  STRATA_HC_Q8_16=0 STRATA_QSA_SCORES_16=0 STRATA_GDN_GROUPS_SPLIT=0 STRATA_GDN_AB_WIDE=0 \
  STRATA_BATCH_SPLIT=0 STRATA_PCIE_FETCH_OVERLAP=0 STRATA_CPU_REDUCE_MISS=0 \
  STRATA_CPU_EXPERT_NO_SPLIT=0 STRATA_MOE_BUCKET_NOSPLIT=0 STRATA_MOE_BUCKET_CHUNK=0 \
  STRATA_MOE_LDS_NT8=0 STRATA_MOE_GRID_CAP=0 STRATA_MMVQ_IL_NARROW_ACC=0 \
  STRATA_VERIFY_WMMA_DENSE=0 STRATA_VERIFY_WMMA_MOE=0 STRATA_BATCH_DF2_ADAPTIVE_T=0 \
  STRATA_BATCH_DF2_TARGET_ROWS=8 STRATA_BATCH_DF2_MAX_T=0 \
  STRATA_BATCH_TAIL_SKIP=0 STRATA_DRAFT_TAIL_SKIP=0 STRATA_BATCH_ROUTE_AFFINITY=0 \
  STRATA_BATCH_PCIE_AUTO=0 STRATA_BATCH_PCIE_BALANCE=0 \
  STRATA_STAGING_BLOBS=0 STRATA_PCIE_GROUP_ROWS=16

run_arm hc_q8_16            STRATA_HC_Q8_16=1
run_arm qsa_scores_16       STRATA_QSA_SCORES_16=1
run_arm gdn_groups_split    STRATA_GDN_GROUPS_SPLIT=1 STRATA_GDN_AB_WIDE=1
run_arm batch_split         STRATA_BATCH_SPLIT=1
run_arm pcie_overlap        STRATA_PCIE_FETCH_OVERLAP=1 STRATA_STAGING_BLOBS=32 STRATA_PCIE_GROUP_ROWS=32
run_arm cpu_reduce_miss     STRATA_CPU_REDUCE_MISS=1
run_arm moe_lds_nt8         STRATA_MOE_BUCKET_NOSPLIT=1 STRATA_MOE_LDS_NT8=1 STRATA_MOE_GRID_CAP=64
run_arm mmvq_il_narrow_acc  STRATA_MMVQ_IL_NARROW_ACC=1
run_arm wmma_dense          STRATA_VERIFY_WMMA_DENSE=1
run_arm df2_adaptive_t      STRATA_BATCH_DF2_ADAPTIVE_T=1
run_arm draft_tail_skip     STRATA_BATCH_TAIL_SKIP=7 STRATA_DRAFT_TAIL_SKIP=6
run_arm route_affinity      STRATA_BATCH_ROUTE_AFFINITY=1

run_arm combined_safe \
  STRATA_HC_Q8_16=1 STRATA_QSA_SCORES_16=1 STRATA_GDN_GROUPS_SPLIT=1 STRATA_GDN_AB_WIDE=1 \
  STRATA_BATCH_SPLIT=1 STRATA_PCIE_FETCH_OVERLAP=1 STRATA_STAGING_BLOBS=32 STRATA_PCIE_GROUP_ROWS=32 \
  STRATA_CPU_REDUCE_MISS=1 STRATA_MOE_BUCKET_NOSPLIT=1 STRATA_MOE_LDS_NT8=1 STRATA_MOE_GRID_CAP=64 \
  STRATA_MMVQ_IL_NARROW_ACC=1 STRATA_VERIFY_WMMA_DENSE=1 STRATA_VERIFY_WMMA_MOE=1

run_arm combined_full \
  STRATA_HC_Q8_16=1 STRATA_QSA_SCORES_16=1 STRATA_GDN_GROUPS_SPLIT=1 STRATA_GDN_AB_WIDE=1 \
  STRATA_BATCH_SPLIT=1 STRATA_PCIE_FETCH_OVERLAP=1 STRATA_STAGING_BLOBS=32 STRATA_PCIE_GROUP_ROWS=32 \
  STRATA_CPU_REDUCE_MISS=1 STRATA_MOE_BUCKET_NOSPLIT=1 STRATA_MOE_LDS_NT8=1 STRATA_MOE_GRID_CAP=64 \
  STRATA_MMVQ_IL_NARROW_ACC=1 STRATA_VERIFY_WMMA_DENSE=1 STRATA_VERIFY_WMMA_MOE=1 \
  STRATA_BATCH_DF2_ADAPTIVE_T=1 STRATA_BATCH_TAIL_SKIP=7 STRATA_DRAFT_TAIL_SKIP=6 \
  STRATA_BATCH_ROUTE_AFFINITY=1

echo "=== Done. Logs in ${LOG_DIR} ==="
