// src/hip/pm4_tape.cpp - Tape recorder, static per-kernel-family effect table,
// fixed-capacity kernarg arena, and per-window fail-closed latch for GFX12 PM4 replay.
//
// Port note: from warpfront/hipfire rdna-compute + railgun (Apache-2.0)
// Ported from warpfront/hipfire crates/rdna-compute/src/replay.rs (pointer_effects,
// expected_kernarg_bytes, recorded_resource_accesses, ReplayQuiescence) and
// crates/railgun/src/plan.rs (ResourceId, Access, Effect).

#include "strata/hip/pm4_tape.hpp"

#if (defined(__linux__) && defined(STRATA_WITH_HSA)) || defined(STRATA_PM4_HOST_TEST)

#include <algorithm>
#include <cstring>
#include <mutex>

namespace strata::hip::pm4 {

const char* tape_status_to_string(TapeStatus status) noexcept {
    switch (status) {
        case TapeStatus::Ok:
            return "ok";
        case TapeStatus::InvalidatedUnknownKernel:
            return "unknown kernel family (missing static effect-table entry)";
        case TapeStatus::InvalidatedKernargSizeMismatch:
            return "kernarg byte length mismatch against static contract";
        case TapeStatus::InvalidatedPointerOffsetOutOfBounds:
            return "pointer effect offset out of bounds of kernarg block";
        case TapeStatus::InvalidatedUnalignedPointerOffset:
            return "pointer effect offset is not 8-byte aligned";
        case TapeStatus::InvalidatedUnresolvedPointer:
            return "device pointer could not be resolved to a live allocation";
        case TapeStatus::InvalidatedArenaOverflow:
            return "per-window kernarg arena overflow";
        case TapeStatus::InvalidatedInvalidGeometry:
            return "launch geometry invalid or exceeds GFX12 grid.y/z <= 65536 limit";
        case TapeStatus::InvalidatedIncompleteCapture:
            return "recorded launch count differs from captured hipGraph kernel node count";
        case TapeStatus::InvalidatedNonQuiescent:
            return "previous window replay did not prove quiescence";
        case TapeStatus::InvalidatedEmptyTape:
            return "captured tape has zero dispatches";
    }
    return "unknown tape error";
}

namespace {

constexpr KernelPointerEffect vload(uint16_t offset) noexcept {
    return KernelPointerEffect{offset, MemAccessKind::VmemLoad};
}

constexpr KernelPointerEffect vstore(uint16_t offset) noexcept {
    return KernelPointerEffect{offset, MemAccessKind::VmemStore};
}

constexpr KernelPointerEffect sload(uint16_t offset) noexcept {
    return KernelPointerEffect{offset, MemAccessKind::SmemLoad};
}

// ---- Static per-kernel-family pointer effect arrays for Strata verify/decode ----
// Non-const or read-modify-write device pointers are conservatively classified as
// VmemStore (matching hipfire crates/rdna-compute/src/replay.rs:878-880).

// gpu_stamp_kernel(uint64_t* dst, uint64_t a, uint64_t b) -> 24B
constexpr KernelPointerEffect kStampEffects[] = {
    vstore(0),
};

// copy_i32_from_mapped(int32_t* dst, const int32_t* src, int64_t n) -> 24B
constexpr KernelPointerEffect kCopyI32Effects[] = {
    vstore(0),
    vload(8),
};

// copy_step_pos_from_mapped(int32_t* step_d, int32_t* pos_d, const int32_t* step_m, int T, int n_head) -> 32B
constexpr KernelPointerEffect kCopyStepPosEffects[] = {
    vstore(0),
    vstore(8),
    vload(16),
};

// embedding_gather_dev(const void* table, const int32_t* tok, float* out, int64_t d, int T) -> 40B
constexpr KernelPointerEffect kEmbeddingGatherEffects[] = {
    vload(0),
    vload(8),
    vstore(16),
};

// broadcast_streams(const float* x, float* R, int64_t n_embd, int64_t hc, int T) -> 32B
constexpr KernelPointerEffect kBroadcastStreamsEffects[] = {
    vload(0),
    vstore(8),
};

// native_quantize_q8_1_kernel(const float* x, void* vy, int kx, int kx_padded) -> 24B
constexpr KernelPointerEffect kQuantizeQ81Effects[] = {
    vload(0),
    vstore(8),
};

// native_mmvq_multi_kernel(const void* vx, const void* vy, float* dst, int ncols_x, int nrows_y, int s_x, int s_dst) -> 40B
constexpr KernelPointerEffect kNativeMmvqMultiEffects[] = {
    vload(0),
    vload(8),
    vstore(16),
};

// bf16_gemv_fp32_mmvf_multi(const float* x, const uint16_t* w, float* y, int K, int N, int ncols) -> 40B
constexpr KernelPointerEffect kBf16GemvMultiEffects[] = {
    vload(0),
    vload(8),
    vstore(16),
};

// fused_gr_read_multi(const float* R, const float* w_norm, const uint16_t* w_down, const uint16_t* w_up,
//                     float* R_out, float* lo, float* rs, float* mixed, void* q8_mixed, float eps, int T) -> 80B
constexpr KernelPointerEffect kFusedGrReadMultiEffects[] = {
    vload(0),
    sload(8),
    vload(16),
    vload(24),
    vstore(32),
    vstore(40),
    vstore(48),
    vstore(56),
    vstore(64),
};

// gdn_conv_l2_multi(const float* qkv, float* conv_state, const float* conv_w, float* qkv_out, float* h_out, int C, int T) -> 48B
constexpr KernelPointerEffect kGdnConvL2MultiEffects[] = {
    vload(0),
    vstore(8),
    vload(16),
    vstore(24),
    vstore(32),
};

// gdn_ab_multi(const float* a_raw, const float* b_raw, const float* a_log, const float* dt_bias,
//              float* gate_out, float* beta_out, int HV, int T) -> 56B
constexpr KernelPointerEffect kGdnAbMultiEffects[] = {
    vload(0),
    vload(8),
    sload(16),
    sload(24),
    vstore(32),
    vstore(40),
};

// gdn_step_norm_multi(float* state, const float* qkv, const float* gate, const float* beta,
//                     const float* z, const float* norm_w, float* y_out, float eps, int HK, int HV, int T) -> 72B
constexpr KernelPointerEffect kGdnStepNormMultiEffects[] = {
    vstore(0),
    vload(8),
    vload(16),
    vload(24),
    vload(32),
    sload(40),
    vstore(48),
};

// native_qsa_rms_norm_rope(float* x, const float* gamma, const int32_t* pos, int n_heads, int head_dim, int rope_dim, float eps, float theta, int T) -> 48B
constexpr KernelPointerEffect kQsaRmsNormRopeEffects[] = {
    vstore(0),
    sload(8),
    vload(16),
};

// fwht256_pair_inplace_cuda(float* a, float* b, int64_t n_a, int64_t n_b) -> 32B
constexpr KernelPointerEffect kFwhtPairEffects[] = {
    vstore(0),
    vstore(8),
};

// kv_append_int8_dev_multi(int8_t* k_pool, float* ks_pool, int8_t* v_pool, float* vs_pool,
//                          const float* k_in, const float* v_in, const int32_t* step, int max_cells, int T) -> 64B
constexpr KernelPointerEffect kKvAppendInt8Effects[] = {
    vstore(0),
    vstore(8),
    vstore(16),
    vstore(24),
    vload(32),
    vload(40),
    vload(48),
};

// native_qsa_indexer_append_steps(const float* idx_raw, const int32_t* step, const float* gamma,
//                                 float* idx_tail, float* idx_dead, float* idx_pooled, int32_t* idx_block_pos, int T, float eps) -> 64B
constexpr KernelPointerEffect kQsaIndexerAppendEffects[] = {
    vload(0),
    vload(8),
    sload(16),
    vstore(24),
    vstore(32),
    vstore(40),
    vstore(48),
};

// qsa_block_scores(const float* qidx, const float* idx_pooled, const float* idx_tail,
//                  const int32_t* idx_block_pos, const int32_t* step, float* scores, int max_blocks, int T) -> 56B
constexpr KernelPointerEffect kQsaBlockScoresEffects[] = {
    vload(0),
    vload(8),
    vload(16),
    vload(24),
    vload(32),
    vstore(40),
};

// qsa_block_topk(const float* scores, const int32_t* step, int32_t* sel, int max_blocks, int k_blocks, int T) -> 40B
constexpr KernelPointerEffect kQsaBlockTopkEffects[] = {
    vload(0),
    vload(8),
    vstore(16),
};

// qsa_kv_resolve(const int32_t* sel_blocks, const int32_t* step, int32_t* sel_cells, int cap, int T) -> 32B
constexpr KernelPointerEffect kQsaKvResolveEffects[] = {
    vload(0),
    vload(8),
    vstore(16),
};

// qsa_decode_attn_batch_int8(const float* q, const int8_t* k_pool, const float* ks_pool,
//                            const int8_t* v_pool, const float* vs_pool, const int32_t* sel,
//                            const int32_t* step, float* scratch, float* out, int cap, int T) -> 80B
constexpr KernelPointerEffect kQsaDecodeAttnBatchEffects[] = {
    vload(0),
    vload(8),
    vload(16),
    vload(24),
    vload(32),
    vload(40),
    vload(48),
    vstore(56),
    vstore(64),
};

// native_qsa_gate_quantize_q8_1(const float* attn, const float* qfull, void* xq_out, int n_heads, int head_dim) -> 32B
constexpr KernelPointerEffect kQsaGateQuantizeEffects[] = {
    vload(0),
    vload(8),
    vstore(16),
};

// native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int T) -> 32B
constexpr KernelPointerEffect kRouterTop10Effects[] = {
    vload(0),
    vstore(8),
    vstore(16),
};

// resident_plan_all_resident(const int32_t* ids, const int32_t* d_res, const uint8_t* cache_base,
//                            const uint64_t* slot_off, int32_t* plan_out, int n_items, int k) -> 48B
constexpr KernelPointerEffect kResidentPlanArEffects[] = {
    vload(0),
    vload(8),
    vload(16),
    vload(24),
    vstore(32),
};

// shared_expert_multi(const void* xq, const void* w_gate, const void* w_up, const void* w_down,
//                     const uint16_t* w_gate_inp, const float* xm, float* sh_out, int N, int n_ff, int T) -> 72B
constexpr KernelPointerEffect kSharedExpertMultiEffects[] = {
    vload(0),
    vload(8),
    vload(16),
    vload(24),
    vload(32),
    vload(40),
    vstore(48),
};

// quantize_q8_1_rows(const float* x, void* vy, int n_rows, int n_cols) -> 24B
constexpr KernelPointerEffect kQuantizeRowsEffects[] = {
    vload(0),
    vstore(8),
};

// native_expert_grouped(const uint64_t* gp, const int32_t* gs, const int32_t* gn,
//                       const int32_t* p_dst, const int32_t* p_tok, const void* xq,
//                       void* hit_scratch, float* parts_out, int cap, int N, int n_ff) -> 80B
constexpr KernelPointerEffect kNativeExpertGroupedEffects[] = {
    vload(0),
    vload(8),
    vload(16),
    vload(24),
    vload(32),
    vload(40),
    vstore(48),
    vstore(56),
};

// native_moe_combine_multi(const float* parts, const float* weights, const float* shared,
//                          float* bo_out, int N, int K, int T) -> 48B
constexpr KernelPointerEffect kNativeMoeCombineEffects[] = {
    vload(0),
    vload(8),
    vload(16),
    vstore(24),
};

// gr_write_multi(const float* R_in, const float* bo, const float* inj, float* R_out, int N, int HC, int T) -> 48B
constexpr KernelPointerEffect kGrWriteMultiEffects[] = {
    vload(0),
    vload(8),
    vload(16),
    vstore(24),
};

// argmax_rows(const float* logits, uint8_t* scratch, int32_t* out_ids, int T, int V) -> 32B
constexpr KernelPointerEffect kArgmaxRowsEffects[] = {
    vload(0),
    vstore(8),
    vstore(16),
};

#define CONTRACT_ROW(name_lit, bytes, arr, dyn_grid) \
    KernelContract{name_lit, bytes, arr, sizeof(arr) / sizeof((arr)[0]), dyn_grid}

constexpr KernelContract kStrataVerifyContracts[] = {
    CONTRACT_ROW("gpu_stamp_kernel", 24u, kStampEffects, false),
    CONTRACT_ROW("copy_i32_from_mapped", 24u, kCopyI32Effects, false),
    CONTRACT_ROW("copy_step_pos_from_mapped", 32u, kCopyStepPosEffects, false),
    CONTRACT_ROW("embedding_gather_dev", 40u, kEmbeddingGatherEffects, false),
    CONTRACT_ROW("broadcast_streams", 32u, kBroadcastStreamsEffects, false),
    CONTRACT_ROW("native_quantize_q8_1_kernel", 24u, kQuantizeQ81Effects, false),
    CONTRACT_ROW("native_mmvq_multi_kernel", 40u, kNativeMmvqMultiEffects, false),
    CONTRACT_ROW("bf16_gemv_fp32_mmvf_multi", 40u, kBf16GemvMultiEffects, false),
    CONTRACT_ROW("fused_gr_read_multi", 80u, kFusedGrReadMultiEffects, false),
    CONTRACT_ROW("gdn_conv_l2_multi", 48u, kGdnConvL2MultiEffects, false),
    CONTRACT_ROW("gdn_ab_multi", 56u, kGdnAbMultiEffects, false),
    CONTRACT_ROW("gdn_step_norm_multi", 72u, kGdnStepNormMultiEffects, false),
    CONTRACT_ROW("native_qsa_rms_norm_rope", 48u, kQsaRmsNormRopeEffects, false),
    CONTRACT_ROW("fwht256_pair_inplace_cuda", 32u, kFwhtPairEffects, false),
    CONTRACT_ROW("kv_append_int8_dev_multi", 64u, kKvAppendInt8Effects, false),
    CONTRACT_ROW("native_qsa_indexer_append_steps", 64u, kQsaIndexerAppendEffects, false),
    CONTRACT_ROW("qsa_block_scores", 56u, kQsaBlockScoresEffects, true),
    CONTRACT_ROW("qsa_block_topk", 40u, kQsaBlockTopkEffects, false),
    CONTRACT_ROW("qsa_kv_resolve", 32u, kQsaKvResolveEffects, false),
    CONTRACT_ROW("qsa_decode_attn_batch_int8", 80u, kQsaDecodeAttnBatchEffects, true),
    CONTRACT_ROW("native_qsa_gate_quantize_q8_1", 32u, kQsaGateQuantizeEffects, false),
    CONTRACT_ROW("native_router_top10_multi", 32u, kRouterTop10Effects, false),
    CONTRACT_ROW("resident_plan_all_resident", 48u, kResidentPlanArEffects, false),
    CONTRACT_ROW("shared_expert_multi", 72u, kSharedExpertMultiEffects, false),
    CONTRACT_ROW("quantize_q8_1_rows", 24u, kQuantizeRowsEffects, false),
    CONTRACT_ROW("native_expert_grouped", 80u, kNativeExpertGroupedEffects, false),
    CONTRACT_ROW("native_moe_combine_multi", 48u, kNativeMoeCombineEffects, false),
    CONTRACT_ROW("gr_write_multi", 48u, kGrWriteMultiEffects, false),
    CONTRACT_ROW("argmax_rows", 32u, kArgmaxRowsEffects, false),
};

#undef CONTRACT_ROW

#define ALIAS_ROW(sym_lit, fam_lit) KernelSymbolAlias{sym_lit, fam_lit}

// Explicit, source-verified canonicalization table for concrete kernel symbols in Strata sources:
// - verify_kernels.cu:1913,1934: gpu_stamp_kernel<{true,false}>
// - native_mmvq.cu:162,3467: native_quantize_q8_1_kernel
// - native_mmvq.cu:1431,1757,1773,1774,1779,1780: native_mmvq_multi_kernel<F, NCOLS, NW, ROWS, TS, PAIR>
// - native_bf16.cu:119,498-504,545,560-566: bf16_f32_mmvf_multi_kernel<BLOCK_SIZE, NT, EXACT_T, AUX>
// - Real __global__ kernel identifiers across src/kernels/cuda/*.cu
#define MMVQ_TRAIT_ROWS_FOR_NCOLS(F_STR, NCOLS_STR) \
    ALIAS_ROW("native_mmvq_multi_kernel<" F_STR ", " NCOLS_STR ", 4, 1, false, false>", "native_mmvq_multi_kernel"), \
    ALIAS_ROW("native_mmvq_multi_kernel<" F_STR ", " NCOLS_STR ", 4, 1, true, false>", "native_mmvq_multi_kernel"), \
    ALIAS_ROW("native_mmvq_multi_kernel<" F_STR ", " NCOLS_STR ", 4, 2, false, false>", "native_mmvq_multi_kernel"), \
    ALIAS_ROW("native_mmvq_multi_kernel<" F_STR ", " NCOLS_STR ", 4, 2, true, false>", "native_mmvq_multi_kernel"), \
    ALIAS_ROW("native_mmvq_multi_kernel<" F_STR ", " NCOLS_STR ", 2, 2, false, false>", "native_mmvq_multi_kernel"), \
    ALIAS_ROW("native_mmvq_multi_kernel<" F_STR ", " NCOLS_STR ", 4, 1, false, false, false, false>", "native_mmvq_multi_kernel"), \
    ALIAS_ROW("native_mmvq_multi_kernel<" F_STR ", " NCOLS_STR ", 4, 1, true, false, false, false>", "native_mmvq_multi_kernel"), \
    ALIAS_ROW("native_mmvq_multi_kernel<" F_STR ", " NCOLS_STR ", 4, 2, false, false, false, false>", "native_mmvq_multi_kernel"), \
    ALIAS_ROW("native_mmvq_multi_kernel<" F_STR ", " NCOLS_STR ", 4, 2, true, false, false, false>", "native_mmvq_multi_kernel"), \
    ALIAS_ROW("native_mmvq_multi_kernel<" F_STR ", " NCOLS_STR ", 2, 2, false, false, false, false>", "native_mmvq_multi_kernel")

#define MMVQ_TRAIT_ROWS(F_STR) \
    MMVQ_TRAIT_ROWS_FOR_NCOLS(F_STR, "2"), \
    MMVQ_TRAIT_ROWS_FOR_NCOLS(F_STR, "3"), \
    MMVQ_TRAIT_ROWS_FOR_NCOLS(F_STR, "4"), \
    MMVQ_TRAIT_ROWS_FOR_NCOLS(F_STR, "5"), \
    MMVQ_TRAIT_ROWS_FOR_NCOLS(F_STR, "6"), \
    MMVQ_TRAIT_ROWS_FOR_NCOLS(F_STR, "7"), \
    MMVQ_TRAIT_ROWS_FOR_NCOLS(F_STR, "8"), \
    MMVQ_TRAIT_ROWS_FOR_NCOLS(F_STR, "9"), \
    MMVQ_TRAIT_ROWS_FOR_NCOLS(F_STR, "10"), \
    MMVQ_TRAIT_ROWS_FOR_NCOLS(F_STR, "11"), \
    MMVQ_TRAIT_ROWS_FOR_NCOLS(F_STR, "12"), \
    MMVQ_TRAIT_ROWS_FOR_NCOLS(F_STR, "13"), \
    MMVQ_TRAIT_ROWS_FOR_NCOLS(F_STR, "14"), \
    MMVQ_TRAIT_ROWS_FOR_NCOLS(F_STR, "15"), \
    MMVQ_TRAIT_ROWS_FOR_NCOLS(F_STR, "16")

#define BF16_MMVF_ROWS_FOR_BLOCK(B_STR) \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 1, true>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 2, true>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 3, true>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 4, true>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 5, true>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 6, true>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 8, false>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 9, true>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 10, true>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 11, true>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 12, true>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 13, true>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 14, true>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 15, true>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 16, true>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 1, true, false>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 2, true, false>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 3, true, false>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 4, true, false>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 5, true, false>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 6, true, false>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 8, false, false>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 9, true, false>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 10, true, false>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 11, true, false>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 12, true, false>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 13, true, false>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 14, true, false>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 15, true, false>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 16, true, false>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 1, true, true>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 2, true, true>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 3, true, true>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 4, true, true>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 5, true, true>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 6, true, true>", "bf16_gemv_fp32_mmvf_multi"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 8, false, true>", "bf16_gemv_fp32_mmvf_multi")

constexpr KernelSymbolAlias kStrataKernelSymbolAliases[] = {
    // 1. gpu_stamp_kernel (src/kernels/cuda/verify_kernels.cu:1913, launched via launch_pdl at :1934)
    ALIAS_ROW("gpu_stamp_kernel", "gpu_stamp_kernel"),
    ALIAS_ROW("gpu_stamp_kernel<true>", "gpu_stamp_kernel"),
    ALIAS_ROW("gpu_stamp_kernel<false>", "gpu_stamp_kernel"),
    ALIAS_ROW("strata::kernels::(anonymous namespace)::gpu_stamp_kernel<true>", "gpu_stamp_kernel"),
    ALIAS_ROW("strata::kernels::(anonymous namespace)::gpu_stamp_kernel<false>", "gpu_stamp_kernel"),

    // 2. native_quantize_q8_1_kernel (src/kernels/cuda/native_mmvq.cu:162, launched via launch_pdl at :3467)
    ALIAS_ROW("native_quantize_q8_1_kernel", "native_quantize_q8_1_kernel"),
    ALIAS_ROW("strata::kernels::(anonymous namespace)::native_quantize_q8_1_kernel", "native_quantize_q8_1_kernel"),

    // 3. native_mmvq_multi_kernel (src/kernels/cuda/native_mmvq.cu:1431, launched via launch_pdl at :1757,1773,1774,1779,1780)
    ALIAS_ROW("native_mmvq_multi_kernel", "native_mmvq_multi_kernel"),
    MMVQ_TRAIT_ROWS("Q4KTraits"),
    MMVQ_TRAIT_ROWS("Q5KTraits"),
    MMVQ_TRAIT_ROWS("Q6KTraits"),
    MMVQ_TRAIT_ROWS("Q20Traits"),
    MMVQ_TRAIT_ROWS("Q3KTraits"),
    MMVQ_TRAIT_ROWS("IQ4XSTraits"),
    MMVQ_TRAIT_ROWS("SmallTraits<Q40Block, 4>"),
    MMVQ_TRAIT_ROWS("SmallTraits<Q50Block, 4>"),
    MMVQ_TRAIT_ROWS("SmallTraits<Q80Block, 8>"),
    MMVQ_TRAIT_ROWS("SmallTraits<IQ4NLBlock, 4>"),

    // 4. bf16_f32_mmvf_multi_kernel (src/kernels/cuda/native_bf16.cu:119, launched via launch_pdl at :498-504,545,560-566)
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel", "bf16_gemv_fp32_mmvf_multi"),
    ALIAS_ROW("bf16_gemv_fp32_mmvf_multi", "bf16_gemv_fp32_mmvf_multi"),
    BF16_MMVF_ROWS_FOR_BLOCK("32"),
    BF16_MMVF_ROWS_FOR_BLOCK("64"),
    BF16_MMVF_ROWS_FOR_BLOCK("96"),
    BF16_MMVF_ROWS_FOR_BLOCK("128"),
    BF16_MMVF_ROWS_FOR_BLOCK("160"),
    BF16_MMVF_ROWS_FOR_BLOCK("192"),
    BF16_MMVF_ROWS_FOR_BLOCK("224"),
    BF16_MMVF_ROWS_FOR_BLOCK("256"),

    // 5. Real __global__ kernel symbols across src/kernels/cuda/ mapped to their contract families
    ALIAS_ROW("copy_i32_from_mapped_kernel", "copy_i32_from_mapped"),
    ALIAS_ROW("copy_i32_from_mapped", "copy_i32_from_mapped"),
    ALIAS_ROW("copy_step_pos_from_mapped_kernel", "copy_step_pos_from_mapped"),
    ALIAS_ROW("copy_step_pos_from_mapped", "copy_step_pos_from_mapped"),
    ALIAS_ROW("embedding_gather_dev_kernel", "embedding_gather_dev"),
    ALIAS_ROW("embedding_gather_dev", "embedding_gather_dev"),
    ALIAS_ROW("broadcast_streams_kernel", "broadcast_streams"),
    ALIAS_ROW("broadcast_streams", "broadcast_streams"),
    ALIAS_ROW("gdn_conv_l2_multi_kernel", "gdn_conv_l2_multi"),
    ALIAS_ROW("gdn_conv_l2_multi", "gdn_conv_l2_multi"),
    ALIAS_ROW("gdn_ab_multi_kernel", "gdn_ab_multi"),
    ALIAS_ROW("gdn_ab_multi_kernel<1, true>", "gdn_ab_multi"),
    ALIAS_ROW("gdn_ab_multi_kernel<2, true>", "gdn_ab_multi"),
    ALIAS_ROW("gdn_ab_multi_kernel<3, true>", "gdn_ab_multi"),
    ALIAS_ROW("gdn_ab_multi_kernel<4, true>", "gdn_ab_multi"),
    ALIAS_ROW("gdn_ab_multi_kernel<5, true>", "gdn_ab_multi"),
    ALIAS_ROW("gdn_ab_multi_kernel<6, true>", "gdn_ab_multi"),
    ALIAS_ROW("gdn_ab_multi_kernel<8, false>", "gdn_ab_multi"),
    ALIAS_ROW("gdn_ab_multi_kernel<16, false>", "gdn_ab_multi"),
    ALIAS_ROW("gdn_ab_multi", "gdn_ab_multi"),
    ALIAS_ROW("gdn_step_norm_multi_kernel", "gdn_step_norm_multi"),
    ALIAS_ROW("gdn_step_norm_multi_kernel<false, false>", "gdn_step_norm_multi"),
    ALIAS_ROW("gdn_step_norm_multi_kernel<true, false>", "gdn_step_norm_multi"),
    ALIAS_ROW("gdn_step_norm_multi_kernel<true, true>", "gdn_step_norm_multi"),
    ALIAS_ROW("gdn_step_norm_multi", "gdn_step_norm_multi"),
    ALIAS_ROW("norm_rope_kernel", "native_qsa_rms_norm_rope"),
    ALIAS_ROW("norm_rope_kernel<false>", "native_qsa_rms_norm_rope"),
    ALIAS_ROW("norm_rope_kernel<true>", "native_qsa_rms_norm_rope"),
    ALIAS_ROW("native_qsa_rms_norm_rope", "native_qsa_rms_norm_rope"),
    ALIAS_ROW("fwht256_pair_kernel", "fwht256_pair_inplace_cuda"),
    ALIAS_ROW("fwht256_pair_inplace_cuda", "fwht256_pair_inplace_cuda"),
    ALIAS_ROW("append", "native_qsa_indexer_append_steps"),
    ALIAS_ROW("append<false>", "native_qsa_indexer_append_steps"),
    ALIAS_ROW("append<true>", "native_qsa_indexer_append_steps"),
    ALIAS_ROW("native_qsa_indexer_append_steps", "native_qsa_indexer_append_steps"),
    ALIAS_ROW("block_scores_kernel", "qsa_block_scores"),
    ALIAS_ROW("qsa_block_scores", "qsa_block_scores"),
    ALIAS_ROW("block_topk_kernel", "qsa_block_topk"),
    ALIAS_ROW("qsa_block_topk", "qsa_block_topk"),
    ALIAS_ROW("resolve_kernel", "qsa_kv_resolve"),
    ALIAS_ROW("qsa_kv_resolve", "qsa_kv_resolve"),
    ALIAS_ROW("attn_chunk_kernel", "qsa_decode_attn_batch_int8"),
    ALIAS_ROW("attn_chunk_kernel<1, false, false, false>", "qsa_decode_attn_batch_int8"),
    ALIAS_ROW("attn_chunk_kernel<1, true, false, false>", "qsa_decode_attn_batch_int8"),
    ALIAS_ROW("qsa_decode_attn_batch_int8", "qsa_decode_attn_batch_int8"),
    ALIAS_ROW("gate_quantize_q8_1_kernel", "native_qsa_gate_quantize_q8_1"),
    ALIAS_ROW("native_qsa_gate_quantize_q8_1", "native_qsa_gate_quantize_q8_1"),
    ALIAS_ROW("route_multi", "native_router_top10_multi"),
    ALIAS_ROW("native_router_top10_multi", "native_router_top10_multi"),
    ALIAS_ROW("resident_plan_kernel", "resident_plan_all_resident"),
    ALIAS_ROW("resident_plan_kernel<128>", "resident_plan_all_resident"),
    ALIAS_ROW("resident_plan_kernel<256>", "resident_plan_all_resident"),
    ALIAS_ROW("resident_plan_all_resident", "resident_plan_all_resident"),
    ALIAS_ROW("quantize_q8_1_kernel", "quantize_q8_1_rows"),
    ALIAS_ROW("quantize_q8_1_rows", "quantize_q8_1_rows"),
    ALIAS_ROW("combine", "native_moe_combine_multi"),
    ALIAS_ROW("combine<false, false>", "native_moe_combine_multi"),
    ALIAS_ROW("combine<true, false>", "native_moe_combine_multi"),
    ALIAS_ROW("combine<false, true>", "native_moe_combine_multi"),
    ALIAS_ROW("combine<true, true>", "native_moe_combine_multi"),
    ALIAS_ROW("native_moe_combine_multi", "native_moe_combine_multi"),
    ALIAS_ROW("post_multi", "gr_write_multi"),
    ALIAS_ROW("gr_write_multi", "gr_write_multi"),
    ALIAS_ROW("argmax_rows_kernel", "argmax_rows"),
    ALIAS_ROW("argmax_rows", "argmax_rows"),
};

#undef BF16_MMVF_ROWS_FOR_BLOCK
#undef MMVQ_TRAIT_ROWS
#undef MMVQ_TRAIT_ROWS_FOR_NCOLS
#undef ALIAS_ROW

constexpr size_t align_up_size(size_t value, size_t alignment) noexcept {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

uint64_t read_u64_le(const uint8_t* p) noexcept {
    return static_cast<uint64_t>(p[0]) |
           (static_cast<uint64_t>(p[1]) << 8) |
           (static_cast<uint64_t>(p[2]) << 16) |
           (static_cast<uint64_t>(p[3]) << 24) |
           (static_cast<uint64_t>(p[4]) << 32) |
           (static_cast<uint64_t>(p[5]) << 40) |
           (static_cast<uint64_t>(p[6]) << 48) |
           (static_cast<uint64_t>(p[7]) << 56);
}

struct HostKernelEntry {
    const void* func = nullptr;
    std::string family;
};

std::mutex& host_kernel_mutex() {
    static std::mutex m;
    return m;
}

std::vector<HostKernelEntry>& host_kernel_registry() {
    static std::vector<HostKernelEntry> reg;
    return reg;
}

thread_local Pm4Tape* g_tls_active_tape = nullptr;

}  // namespace

const KernelContract* registered_kernel_contracts(size_t& out_count) noexcept {
    out_count = sizeof(kStrataVerifyContracts) / sizeof(kStrataVerifyContracts[0]);
    return kStrataVerifyContracts;
}

const KernelSymbolAlias* registered_kernel_symbol_aliases(size_t& out_count) noexcept {
    out_count = sizeof(kStrataKernelSymbolAliases) / sizeof(kStrataKernelSymbolAliases[0]);
    return kStrataKernelSymbolAliases;
}

const char* canonicalize_kernel_symbol(const char* symbol_name) noexcept {
    if (symbol_name == nullptr || symbol_name[0] == '\0') {
        return nullptr;
    }
    const size_t alias_count =
        sizeof(kStrataKernelSymbolAliases) / sizeof(kStrataKernelSymbolAliases[0]);
    for (size_t i = 0; i < alias_count; ++i) {
        if (kStrataKernelSymbolAliases[i].symbol_name != nullptr &&
            std::strcmp(kStrataKernelSymbolAliases[i].symbol_name, symbol_name) == 0) {
            return kStrataKernelSymbolAliases[i].canonical_family;
        }
    }
    const size_t contract_count =
        sizeof(kStrataVerifyContracts) / sizeof(kStrataVerifyContracts[0]);
    for (size_t i = 0; i < contract_count; ++i) {
        if (kStrataVerifyContracts[i].family_name != nullptr &&
            std::strcmp(kStrataVerifyContracts[i].family_name, symbol_name) == 0) {
            return kStrataVerifyContracts[i].family_name;
        }
    }
    return nullptr;
}

const KernelContract* lookup_kernel_contract_in(
    const char* family_name,
    const KernelContract* custom_table,
    size_t custom_count) noexcept {
    if (family_name == nullptr || family_name[0] == '\0') {
        return nullptr;
    }
    const KernelContract* table = custom_table;
    size_t count = custom_count;
    const char* lookup_key = family_name;
    if (table == nullptr) {
        table = registered_kernel_contracts(count);
        const char* canonical = canonicalize_kernel_symbol(family_name);
        if (canonical != nullptr) {
            lookup_key = canonical;
        }
    }
    for (size_t i = 0; i < count; ++i) {
        if (table[i].family_name != nullptr &&
            std::strcmp(table[i].family_name, lookup_key) == 0) {
            return &table[i];
        }
    }
    return nullptr;
}

const KernelContract* lookup_kernel_contract(const char* family_name) noexcept {
    return lookup_kernel_contract_in(family_name, nullptr, 0);
}

// ---- AllocationTable ----

bool AllocationTable::insert(uint64_t base, uint64_t size) noexcept {
    if (base == 0 || size == 0 || (base + size) < base) {
        return false;
    }
    const uint64_t end = base + size;
    for (const auto& existing : ranges_) {
        if (base < existing.end() && existing.base < end) {
            return false;  // overlapping allocations are rejected
        }
    }
    ranges_.push_back(AllocationRange{base, size});
    std::sort(ranges_.begin(), ranges_.end(),
              [](const AllocationRange& a, const AllocationRange& b) { return a.base < b.base; });
    return true;
}

bool AllocationTable::resolve(uint64_t device_ptr, AllocationRange& out_range) const noexcept {
    if (device_ptr == 0 || ranges_.empty()) {
        return false;
    }
    auto it = std::upper_bound(
        ranges_.begin(), ranges_.end(), device_ptr,
        [](uint64_t ptr, const AllocationRange& r) { return ptr < r.base; });
    if (it == ranges_.begin()) {
        return false;
    }
    --it;
    if (!it->contains(device_ptr)) {
        return false;
    }
    out_range = *it;
    return true;
}

bool AllocationTable::resolver_thunk(
    uint64_t device_ptr, AllocationRange& out_range, void* user_ctx) noexcept {
    if (user_ctx == nullptr) {
        return false;
    }
    const auto* table = static_cast<const AllocationTable*>(user_ctx);
    return table->resolve(device_ptr, out_range);
}

// ---- KernargArena ----

KernargArena::KernargArena(size_t capacity_bytes) : storage_(capacity_bytes, 0u) {}

void KernargArena::reinitialize(size_t capacity_bytes) {
    storage_.assign(capacity_bytes, 0u);
    used_ = 0;
}

bool KernargArena::allocate_and_copy(
    const void* src, size_t size, uint32_t& out_offset) noexcept {
    const size_t aligned_start = align_up_size(used_, kKernargSlotAlignment);
    const size_t padded_size = align_up_size(size == 0 ? 1u : size, kKernargTailAlignment);
    if (aligned_start > storage_.size() || padded_size > (storage_.size() - aligned_start)) {
        return false;  // fail closed on arena overflow without reallocating
    }
    if (aligned_start > UINT32_MAX) {
        return false;
    }
    if (aligned_start > used_) {
        std::memset(storage_.data() + used_, 0, aligned_start - used_);
    }
    std::memset(storage_.data() + aligned_start, 0, padded_size);
    if (src != nullptr && size > 0) {
        std::memcpy(storage_.data() + aligned_start, src, size);
    }
    out_offset = static_cast<uint32_t>(aligned_start);
    used_ = aligned_start + padded_size;
    return true;
}

// ---- Pm4Tape ----

Pm4Tape::Pm4Tape(size_t arena_capacity_bytes) : arena_(arena_capacity_bytes) {}

bool Pm4Tape::begin_recording(ReplayQuiescence prev_quiescence) noexcept {
    finalized_ = false;
    if (prev_quiescence != ReplayQuiescence::Proven) {
        recording_ = false;
        status_ = TapeStatus::InvalidatedNonQuiescent;
        failure_reason_ = tape_status_to_string(status_);
        return false;
    }
    arena_.reset();
    dispatches_.clear();
    resources_.clear();
    status_ = TapeStatus::Ok;
    failure_reason_.clear();
    recording_ = true;
    return true;
}

void Pm4Tape::invalidate(TapeStatus status, const char* reason) noexcept {
    // Fresh/finalized tapes carry the benign placeholders Ok (post-begin_recording)
    // or InvalidatedEmptyTape (constructed/reset); either accepts the first real
    // failure. A tape already holding a concrete failure keeps its first reason,
    // so a secondary latch (e.g. set_quiescence) never masks why we really refused.
    if (status_ == TapeStatus::Ok || status_ == TapeStatus::InvalidatedEmptyTape) {
        status_ = (status == TapeStatus::Ok) ? TapeStatus::InvalidatedUnknownKernel : status;
        failure_reason_ = (reason != nullptr && reason[0] != '\0')
                              ? reason
                              : tape_status_to_string(status_);
    }
    recording_ = false;
}

ResourceId Pm4Tape::intern_resource(const AllocationRange& range) {
    for (size_t i = 0; i < resources_.size(); ++i) {
        if (resources_[i] == range) {
            return static_cast<ResourceId>(i);
        }
    }
    const auto id = static_cast<ResourceId>(resources_.size());
    resources_.push_back(range);
    return id;
}

bool Pm4Tape::record_launch(
    const char* kernel_family,
    const void* host_function,
    std::array<uint32_t, 3> hip_grid_workgroups,
    std::array<uint32_t, 3> hip_workgroup_size,
    uint32_t dynamic_lds_bytes,
    const void* kernarg_data,
    size_t kernarg_bytes,
    const KernelMetadata* loader_meta,
    const KernelPm4Metadata* pm4_meta) noexcept {
    if (!recording_ || status_ != TapeStatus::Ok) {
        return false;
    }

    const KernelContract* contract =
        lookup_kernel_contract_in(kernel_family, custom_contracts_, custom_contract_count_);
    if (contract == nullptr) {
        std::string msg = "unknown kernel family: ";
        msg += (kernel_family != nullptr ? kernel_family : "<null>");
        invalidate(TapeStatus::InvalidatedUnknownKernel, msg.c_str());
        return false;
    }

    if (kernarg_bytes != contract->expected_kernarg_bytes) {
        std::string msg = std::string(contract->family_name) + ": expected " +
                          std::to_string(contract->expected_kernarg_bytes) +
                          " kernarg bytes, got " + std::to_string(kernarg_bytes);
        invalidate(TapeStatus::InvalidatedKernargSizeMismatch, msg.c_str());
        return false;
    }

    LaunchGeometry geom{};
    const PacketErrorCode geom_err =
        LaunchGeometry::from_hip_workgroups(hip_grid_workgroups, hip_workgroup_size, geom);
    if (geom_err != PacketErrorCode::Ok) {
        std::string msg = std::string(contract->family_name) + ": " +
                          packet_error_to_string(geom_err);
        invalidate(TapeStatus::InvalidatedInvalidGeometry, msg.c_str());
        return false;
    }

    uint32_t arena_offset = 0;
    if (!arena_.allocate_and_copy(kernarg_data, kernarg_bytes, arena_offset)) {
        std::string msg = std::string(contract->family_name) + ": kernarg arena overflow";
        invalidate(TapeStatus::InvalidatedArenaOverflow, msg.c_str());
        return false;
    }

    const uint8_t* stored_bytes = arena_.slot_ptr(arena_offset);
    std::vector<ResourceEffect> resolved_effects;
    resolved_effects.reserve(contract->pointer_effect_count);

    for (size_t i = 0; i < contract->pointer_effect_count; ++i) {
        const KernelPointerEffect pe = contract->pointer_effects[i];
        if ((pe.offset % 8u) != 0u) {
            invalidate(TapeStatus::InvalidatedUnalignedPointerOffset,
                       "pointer effect offset is not 8-byte aligned");
            return false;
        }
        if (static_cast<size_t>(pe.offset) + 8u > kernarg_bytes) {
            invalidate(TapeStatus::InvalidatedPointerOffsetOutOfBounds,
                       "pointer effect offset exceeds kernarg size");
            return false;
        }
        const uint64_t ptr = read_u64_le(stored_bytes + pe.offset);
        if (ptr == 0ULL) {
            // Null pointers are skipped (matching hipfire replay.rs:2237).
            continue;
        }
        if (resolver_ == nullptr) {
            invalidate(TapeStatus::InvalidatedUnresolvedPointer,
                       "no address range resolver configured for non-null device pointer");
            return false;
        }
        AllocationRange range{};
        if (!resolver_(ptr, range, resolver_ctx_) || !range.contains(ptr)) {
            std::string msg = std::string(contract->family_name) +
                              ": unresolved pointer argument at offset " +
                              std::to_string(pe.offset);
            invalidate(TapeStatus::InvalidatedUnresolvedPointer, msg.c_str());
            return false;
        }
        const ResourceId rid = intern_resource(range);
        resolved_effects.push_back(ResourceEffect{
            /*resource=*/rid,
            /*allocation_base=*/range.base,
            /*allocation_bytes=*/range.size,
            /*access_ptr=*/ptr,
            /*kernarg_offset=*/pe.offset,
            /*access=*/pe.access,
        });
    }

    RecordedDispatch disp{};
    disp.kernel_family = contract->family_name;
    disp.host_function = host_function;
    disp.hip_grid_workgroups = hip_grid_workgroups;
    disp.hip_workgroup_size = hip_workgroup_size;
    disp.geometry = geom;
    disp.dynamic_lds_bytes = dynamic_lds_bytes;
    disp.kernarg_arena_offset = arena_offset;
    disp.kernarg_bytes = static_cast<uint32_t>(kernarg_bytes);
    disp.kernarg_padded_bytes =
        static_cast<uint32_t>(align_up_size(kernarg_bytes == 0 ? 1u : kernarg_bytes, kKernargTailAlignment));
    if (loader_meta != nullptr && pm4_meta != nullptr) {
        disp.loader_meta = *loader_meta;
        disp.pm4_meta = *pm4_meta;
        disp.has_hsa_metadata = true;
    } else {
        // Default valid metadata for host-side planning when HSA descriptors are bound later.
        disp.loader_meta.kernel_object = 0x7f000000ULL;
        disp.loader_meta.kernarg_segment_size = disp.kernarg_padded_bytes;
        disp.loader_meta.kernarg_segment_alignment = 16u;
        disp.loader_meta.group_segment_size = 0u;
        disp.loader_meta.private_segment_size = 0u;
        disp.loader_meta.dynamic_callstack = false;
        disp.pm4_meta.code_entry = 0x7f000100ULL;
        disp.pm4_meta.compute_pgm_rsrc1 = 0x000f0004u;
        disp.pm4_meta.compute_pgm_rsrc2 = 0x00000084u;
        disp.pm4_meta.compute_pgm_rsrc3 = 0x00000000u;
        disp.pm4_meta.kernel_code_properties = kAmdKernelCodePropertyEnableSgprKernargSegmentPtr;
    }
    disp.effects = std::move(resolved_effects);
    dispatches_.push_back(std::move(disp));
    return true;
}

TapeStatus Pm4Tape::finalize_recording(size_t expected_graph_kernel_nodes) noexcept {
    recording_ = false;
    if (status_ != TapeStatus::Ok) {
        finalized_ = false;
        return status_;
    }
    if (dispatches_.empty()) {
        invalidate(TapeStatus::InvalidatedEmptyTape, "captured window recorded zero launches");
        finalized_ = false;
        return status_;
    }
    if (expected_graph_kernel_nodes != 0 && dispatches_.size() != expected_graph_kernel_nodes) {
        std::string msg = "incomplete tape capture: recorded " +
                          std::to_string(dispatches_.size()) +
                          " launches via launch_pdl, but hipGraph captured " +
                          std::to_string(expected_graph_kernel_nodes) + " kernel nodes";
        invalidate(TapeStatus::InvalidatedIncompleteCapture, msg.c_str());
        finalized_ = false;
        return status_;
    }
    finalized_ = true;
    return TapeStatus::Ok;
}

bool Pm4Tape::bind_hsa_metadata(
    size_t dispatch_index,
    const KernelMetadata& loader_meta,
    const KernelPm4Metadata& pm4_meta) noexcept {
    if (dispatch_index >= dispatches_.size()) {
        return false;
    }
    if (loader_meta.kernel_object == 0ULL ||
        pm4_meta.code_entry == 0ULL ||
        (pm4_meta.code_entry & 0xffULL) != 0ULL) {
        return false;
    }
    RecordedDispatch& disp = dispatches_[dispatch_index];
    if (loader_meta.kernarg_segment_size < disp.kernarg_bytes) {
        return false;
    }
    disp.loader_meta = loader_meta;
    disp.pm4_meta = pm4_meta;
    disp.has_hsa_metadata = true;
    return true;
}

// ---- Pm4WindowCaptureSlot ----

bool Pm4WindowCaptureSlot::begin_capture() noexcept {
    if (state_ == Pm4WindowState::LatchedFallback) {
        return false;  // latched per-window: no retry storm
    }
    if (state_ == Pm4WindowState::Captured) {
        return false;  // already captured
    }
    ++capture_attempts_;
    if (!tape_.begin_recording(quiescence_)) {
        state_ = Pm4WindowState::LatchedFallback;
        return false;
    }
    state_ = Pm4WindowState::Recording;
    return true;
}

bool Pm4WindowCaptureSlot::end_capture(size_t expected_graph_kernel_nodes) noexcept {
    if (state_ != Pm4WindowState::Recording) {
        return false;
    }
    if (tape_.finalize_recording(expected_graph_kernel_nodes) != TapeStatus::Ok) {
        state_ = Pm4WindowState::LatchedFallback;
        return false;
    }
    state_ = Pm4WindowState::Captured;
    return true;
}

void Pm4WindowCaptureSlot::latch_fallback(TapeStatus status, const char* reason) noexcept {
    tape_.invalidate(status, reason);
    state_ = Pm4WindowState::LatchedFallback;
}

void Pm4WindowCaptureSlot::reset() noexcept {
    state_ = Pm4WindowState::Uninitialized;
    quiescence_ = ReplayQuiescence::Proven;
    capture_attempts_ = 0;
    tape_.begin_recording(ReplayQuiescence::Proven);
    tape_.invalidate(TapeStatus::InvalidatedEmptyTape, "reset");
}

// ---- Host kernel family registry & TLS capture scope ----

void register_host_kernel_family(const void* host_function, const char* family_name) {
    if (host_function == nullptr || family_name == nullptr || family_name[0] == '\0') {
        return;
    }
    const char* canonical = canonicalize_kernel_symbol(family_name);
    if (canonical == nullptr) {
        return;  // refuse unverified/unknown family names (fail-closed)
    }
    std::lock_guard<std::mutex> lock(host_kernel_mutex());
    auto& reg = host_kernel_registry();
    for (auto& entry : reg) {
        if (entry.func == host_function) {
            entry.family = canonical;
            return;
        }
    }
    reg.push_back(HostKernelEntry{host_function, canonical});
}

const char* lookup_host_kernel_family(const void* host_function) noexcept {
    if (host_function == nullptr) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(host_kernel_mutex());
    const auto& reg = host_kernel_registry();
    for (const auto& entry : reg) {
        if (entry.func == host_function) {
            return entry.family.c_str();
        }
    }
    return nullptr;
}

void clear_host_kernel_families() noexcept {
    std::lock_guard<std::mutex> lock(host_kernel_mutex());
    host_kernel_registry().clear();
}

Pm4Tape* active_pm4_tape() noexcept {
    return g_tls_active_tape;
}

bool is_pm4_capture_scope_active() noexcept {
    return g_tls_active_tape != nullptr;
}

Pm4TapeScopeGuard::Pm4TapeScopeGuard(Pm4Tape* tape) noexcept : prev_(g_tls_active_tape) {
    g_tls_active_tape = tape;
}

Pm4TapeScopeGuard::~Pm4TapeScopeGuard() noexcept {
    g_tls_active_tape = prev_;
}

}  // namespace strata::hip::pm4

#endif  // (defined(__linux__) && defined(STRATA_WITH_HSA)) || defined(STRATA_PM4_HOST_TEST)
