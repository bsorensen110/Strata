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
// Re-derived from the real __global__ kernel definitions in src/kernels/cuda/ and
// include/strata/kernels/pdl.hpp:53-76 (detail::pack_kernarg_field).
// Non-const or read-modify-write device pointers are conservatively classified as
// VmemStore (matching hipfire crates/rdna-compute/src/replay.rs:878-880).
// Multi-kernel host wrappers or kernels with indirect pointer-to-pointer tables
// (fused_gr_read_multi, shared_expert_multi, native_expert_grouped, kv_append_int8_dev_multi)
// are intentionally omitted so lookup_kernel_contract() refuses them (fail-closed).

// 1. gpu_stamp_kernel (src/kernels/cuda/verify_kernels.cu:1913)
//    (unsigned long long* buf, int i) -> 12B
constexpr KernelPointerEffect kStampEffects[] = {
    vstore(0),
};

// 2. copy_i32_from_mapped_kernel (src/kernels/cuda/elementwise.cu:458)
//    (int32_t* dst, const volatile int32_t* src, int64_t n) -> 24B
constexpr KernelPointerEffect kCopyI32Effects[] = {
    vstore(0),
    vload(8),
};

// 3. copy_step_pos_from_mapped_kernel (src/kernels/cuda/verify_kernels.cu:1837-1839)
//    (int32_t* step, int32_t* pos, const volatile int32_t* m_step, int T, int MT, int NH, int NKV, int IQ) -> 44B
constexpr KernelPointerEffect kCopyStepPosEffects[] = {
    vstore(0),
    vstore(8),
    vload(16),
};

// 4. embedding_gather_dev_kernel (src/kernels/cuda/verify_kernels.cu:671-675)
//    (const uint8_t* codes, const float* scales, const float* offsets, const int32_t* tokens,
//     int64_t n, int code_bits, int code_bias, int group_elems,
//     unsigned long long row_codes, unsigned long long row_groups, float* out) -> 80B
constexpr KernelPointerEffect kEmbeddingGatherEffects[] = {
    vload(0),
    vload(8),
    vload(16),
    vload(24),
    vstore(72),
};

// 5. broadcast_streams_kernel (src/kernels/cuda/verify_kernels.cu:691)
//    (const float* x, float* R, int64_t n, int hc) -> 28B
constexpr KernelPointerEffect kBroadcastStreamsEffects[] = {
    vload(0),
    vstore(8),
};

// 6. native_quantize_q8_1_kernel (src/kernels/cuda/native_mmvq.cu:162)
//    (const float* x_, Q81Block* y_, int n_in) -> 20B
constexpr KernelPointerEffect kQuantizeQ81Effects[] = {
    vload(0),
    vstore(8),
};

// 7. native_mmvq_multi_kernel (src/kernels/cuda/native_mmvq.cu:1431-1434)
//    (const F::Block* w, const Q81Block* x_, float* y_, int n_in, int n_out,
//     const F::Block* w2, float* y2, const int32_t* row_ids) -> 56B
constexpr KernelPointerEffect kNativeMmvqMultiEffects[] = {
    vload(0),
    vload(8),
    vstore(16),
    vload(32),
    vstore(40),
    vload(48),
};

// 8. bf16_f32_mmvf_multi_kernel (src/kernels/cuda/native_bf16.cu:119-122)
//    (const float* x_, int64_t ldx, const uint16_t* w, float* y_, int64_t ldy,
//     int n_in, int n_tok, int fast, const uint16_t* w_aux, float* y_aux, int64_t ldy_aux) -> 80B
constexpr KernelPointerEffect kBf16MmvfMultiEffects[] = {
    vload(0),
    vload(16),
    vstore(24),
    vload(56),
    vstore(64),
};

// 9. gdn_conv_l2_multi_kernel (src/kernels/cuda/verify_kernels.cu:87-89)
//    (float* hist, const float* qkv, const float* w, float* h, int C, int qk_heads, float eps, int t_begin, bool commit) -> 49B
constexpr KernelPointerEffect kGdnConvL2MultiEffects[] = {
    vstore(0),
    vload(8),
    vload(16),
    vstore(24),
};

// 10. gdn_ab_multi_kernel (src/kernels/cuda/verify_kernels.cu:199-203)
//     (const float* x, const uint16_t* wa, const uint16_t* wb, const float* dt,
//      const float* ssm_a, float* gate, float* beta, int n, int h_v, int T, int fast) -> 72B
constexpr KernelPointerEffect kGdnAbMultiEffects[] = {
    vload(0),
    vload(8),
    vload(16),
    sload(24),
    sload(32),
    vstore(40),
    vstore(48),
};

// 11. gdn_step_norm_multi_kernel (src/kernels/cuda/verify_kernels.cu:369-378,
//     include/strata/kernels/verify_kernels.hpp:53-59, src/kernels/cuda/verify_kernels.cu:324-330)
//     (float* state, const float* hbuf, int C, const float* gate, const float* beta,
//      const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int T,
//      const int32_t* n_keep, int t_out_begin, GdnQ81* xq, GdnGroups grp, GdnQ81Il xq_il) -> 344B
constexpr KernelPointerEffect kGdnStepNormMultiEffects[] = {
    vstore(0),
    vload(8),
    vload(24),
    vload(32),
    vload(40),
    sload(48),
    vstore(64),
    vload(88),
    vstore(104),
    vstore(112),
    vstore(120),
    vstore(128),
    vstore(136),
    vstore(144),
    vstore(152),
    vstore(160),
    vstore(168),
    vload(176),
    vload(184),
    vload(192),
    vload(200),
    vload(208),
    vload(216),
    vload(224),
    vload(232),
    vstore(312),
    vstore(320),
    vstore(328),
};

// 12. norm_rope_kernel (src/kernels/cuda/native_rope.cu:71-75, include/strata/kernels/mrope.hpp:35-39)
//     (const float* x, int in_stride, const float* gamma, float* out, int n_cols, int n_rot,
//      float epsilon, float theta_scale, float freq_scale, float corr_low, float corr_high,
//      float ext_factor, float mscale, const int* positions, const int32_t* mtab, RopeTab rt) -> 112B
constexpr KernelPointerEffect kQsaRmsNormRopeEffects[] = {
    vload(0),
    sload(16),
    vstore(24),
    vload(72),
    vload(80),
    vload(88),
    vload(96),
};

// 13. fwht256_pair_kernel (src/kernels/cuda/kv_q4.cu:75)
//     (float* a, float* b, int64_t n_rows, float scale) -> 28B
constexpr KernelPointerEffect kFwhtPairEffects[] = {
    vstore(0),
    vstore(8),
};

// 14. append (src/kernels/cuda/native_qsa_indexer.cu:120-127, include/strata/kernels/mrope.hpp:35-39)
//     (const float* raw, const int32_t* pos_dev, int pos_stride, int n_steps, int pos_base,
//      const float* gamma, float epsilon, float* tail, float* dead, float* pooled, int32_t* block_pos,
//      int max_cells, float theta_scale, float freq_scale, float corr_low, float corr_high,
//      float ext_factor, float mscale, const int32_t* mtab, RopeTab rt) -> 144B
constexpr KernelPointerEffect kQsaIndexerAppendEffects[] = {
    vload(0),
    vload(8),
    sload(32),
    vstore(48),
    vstore(56),
    vstore(64),
    vstore(72),
    vload(112),
    vload(120),
    vload(128),
};

// 15. block_scores_kernel (src/kernels/cuda/qsa_select.cu:114-118)
//     (const float* pooled, const float* dead, const float* q_idx, const int32_t* steps,
//      int64_t max_blocks, float* out) -> 48B
constexpr KernelPointerEffect kQsaBlockScoresEffects[] = {
    vload(0),
    vload(8),
    vload(16),
    vload(24),
    vstore(40),
};

// 16. block_topk_kernel (src/kernels/cuda/qsa_select.cu:158-160)
//     (const float* scores, const int32_t* steps, int64_t max_blocks, int64_t cap, int32_t* ids) -> 40B
constexpr KernelPointerEffect kQsaBlockTopkEffects[] = {
    vload(0),
    vload(8),
    vstore(32),
};

// 17. resolve_kernel (src/kernels/cuda/kv_stream.cu:191-192, include/strata/kernels/kv_stream.hpp:123-133)
//     (KvStreamMap m, const int32_t* ids, const int32_t* steps, int n_q, int cap, int page_size) -> 100B
constexpr KernelPointerEffect kQsaKvResolveEffects[] = {
    vstore(0),
    vstore(8),
    vstore(16),
    vstore(24),
    vstore(32),
    vstore(40),
    vstore(48),
    vload(72),
    vload(80),
};

// 18. attn_chunk_kernel (src/kernels/cuda/qsa_decode_attn.cu:350-355,
//     include/strata/kernels/qsa_decode_attn.hpp:23-33)
//     (const float* q, QsaAttnPools p, const int32_t* ids, const int32_t* step, int n_kv_heads,
//      int page_size, float scale, float* part_acc, float* part_m, float* part_l,
//      int n_chunks, int cap, long long scratch_stride) -> 152B
constexpr KernelPointerEffect kQsaDecodeAttnBatchEffects[] = {
    vload(0),
    vload(8),
    vload(16),
    vload(24),
    vload(32),
    vload(40),
    vload(48),
    vload(56),
    vload(64),
    vload(72),
    vload(80),
    vload(88),
    vstore(112),
    vstore(120),
    vstore(128),
};

// 19. gate_quantize_q8_1_kernel (src/kernels/cuda/native_qsa.cu:107-110)
//     (const float* attn, const float* q_full, QsaQ81Block* xq, int n_head, int head_dim,
//      int* bm, int* pm, float* dl, int n_in, int cp) -> 64B
constexpr KernelPointerEffect kQsaGateQuantizeEffects[] = {
    vload(0),
    vload(8),
    vstore(16),
    vstore(32),
    vstore(40),
    vstore(48),
};

// 20. route_multi (src/kernels/cuda/native_router.cu:100-101)
//     (const float* logits, int32_t* ids, float* weights, int n_tok) -> 28B
constexpr KernelPointerEffect kRouterTop10Effects[] = {
    vload(0),
    vstore(8),
    vstore(16),
};

// 21. resident_plan_kernel (src/kernels/cuda/verify_kernels.cu:1580-1583)
//     (const int32_t* ids, int n, int k, const int32_t* res, int n_expert,
//      const uint8_t* cache_base, const unsigned long long* slot_off, long long blob,
//      int32_t* pl, long long capx, uint32_t* skip, uint32_t ring, volatile uint32_t* plan_err) -> 96B
constexpr KernelPointerEffect kResidentPlanArEffects[] = {
    vload(0),
    vload(16),
    vload(32),
    vload(40),
    vstore(56),
    vstore(72),
    vstore(88),
};

// 22. quantize_q8_1_kernel (src/kernels/cuda/iq_kernels.cu:2885)
//     (const float* x, block_q8_1* y, long long n) -> 24B
constexpr KernelPointerEffect kQuantizeRowsEffects[] = {
    vload(0),
    vstore(8),
};

// 23. combine (src/kernels/cuda/native_moe.cu:42-44)
//     (const float* parts, const float* weights, const float* shared, float* output,
//      int64_t n_embd, int k, const float* sg) -> 56B
constexpr KernelPointerEffect kNativeMoeCombineEffects[] = {
    vload(0),
    vload(8),
    vload(16),
    vstore(24),
    vload(48),
};

// 24. post_multi (src/kernels/cuda/native_gr_postops.cu:85-87)
//     (const float* residual, const float* block_out, const float* inject, float* output,
//      int n_embd, int hc, float scale, long long r_stride, long long b_stride, long long i_stride) -> 72B
constexpr KernelPointerEffect kGrWriteMultiEffects[] = {
    vload(0),
    vload(8),
    vload(16),
    vstore(24),
};

// 25. argmax_rows_kernel (src/kernels/cuda/verify_kernels.cu:1010-1011)
//     (const float* logits, int n, int per_block, ArgRow* rows, int32_t* out, uint32_t* nf_flags) -> 40B
constexpr KernelPointerEffect kArgmaxRowsEffects[] = {
    vload(0),
    vstore(16),
    vstore(24),
    vstore(32),
};

#define CONTRACT_ROW(name_lit, bytes, arr, dyn_grid) \
    KernelContract{name_lit, bytes, arr, sizeof(arr) / sizeof((arr)[0]), dyn_grid}

constexpr KernelContract kStrataVerifyContracts[] = {
    CONTRACT_ROW("gpu_stamp_kernel", 12u, kStampEffects, false),
    CONTRACT_ROW("copy_i32_from_mapped_kernel", 24u, kCopyI32Effects, false),
    CONTRACT_ROW("copy_step_pos_from_mapped_kernel", 44u, kCopyStepPosEffects, false),
    CONTRACT_ROW("embedding_gather_dev_kernel", 80u, kEmbeddingGatherEffects, false),
    CONTRACT_ROW("broadcast_streams_kernel", 28u, kBroadcastStreamsEffects, false),
    CONTRACT_ROW("native_quantize_q8_1_kernel", 20u, kQuantizeQ81Effects, false),
    CONTRACT_ROW("native_mmvq_multi_kernel", 56u, kNativeMmvqMultiEffects, false),
    CONTRACT_ROW("bf16_f32_mmvf_multi_kernel", 80u, kBf16MmvfMultiEffects, false),
    CONTRACT_ROW("gdn_conv_l2_multi_kernel", 49u, kGdnConvL2MultiEffects, false),
    CONTRACT_ROW("gdn_ab_multi_kernel", 72u, kGdnAbMultiEffects, false),
    CONTRACT_ROW("gdn_step_norm_multi_kernel", 344u, kGdnStepNormMultiEffects, false),
    CONTRACT_ROW("norm_rope_kernel", 112u, kQsaRmsNormRopeEffects, false),
    CONTRACT_ROW("fwht256_pair_kernel", 28u, kFwhtPairEffects, false),
    CONTRACT_ROW("append", 144u, kQsaIndexerAppendEffects, false),
    CONTRACT_ROW("block_scores_kernel", 48u, kQsaBlockScoresEffects, true),
    CONTRACT_ROW("block_topk_kernel", 40u, kQsaBlockTopkEffects, false),
    CONTRACT_ROW("resolve_kernel", 100u, kQsaKvResolveEffects, false),
    CONTRACT_ROW("attn_chunk_kernel", 152u, kQsaDecodeAttnBatchEffects, true),
    CONTRACT_ROW("gate_quantize_q8_1_kernel", 64u, kQsaGateQuantizeEffects, false),
    CONTRACT_ROW("route_multi", 28u, kRouterTop10Effects, false),
    CONTRACT_ROW("resident_plan_kernel", 96u, kResidentPlanArEffects, false),
    CONTRACT_ROW("quantize_q8_1_kernel", 24u, kQuantizeRowsEffects, false),
    CONTRACT_ROW("combine", 56u, kNativeMoeCombineEffects, false),
    CONTRACT_ROW("post_multi", 72u, kGrWriteMultiEffects, false),
    CONTRACT_ROW("argmax_rows_kernel", 40u, kArgmaxRowsEffects, false),
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
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 1, true>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 2, true>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 3, true>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 4, true>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 5, true>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 6, true>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 8, false>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 9, true>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 10, true>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 11, true>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 12, true>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 13, true>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 14, true>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 15, true>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 16, true>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 1, true, false>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 2, true, false>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 3, true, false>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 4, true, false>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 5, true, false>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 6, true, false>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 8, false, false>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 9, true, false>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 10, true, false>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 11, true, false>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 12, true, false>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 13, true, false>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 14, true, false>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 15, true, false>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 16, true, false>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 1, true, true>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 2, true, true>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 3, true, true>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 4, true, true>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 5, true, true>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 6, true, true>", "bf16_f32_mmvf_multi_kernel"), \
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel<" B_STR ", 8, false, true>", "bf16_f32_mmvf_multi_kernel")

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
    ALIAS_ROW("bf16_f32_mmvf_multi_kernel", "bf16_f32_mmvf_multi_kernel"),
    ALIAS_ROW("bf16_gemv_fp32_mmvf_multi", "bf16_f32_mmvf_multi_kernel"),
    BF16_MMVF_ROWS_FOR_BLOCK("32"),
    BF16_MMVF_ROWS_FOR_BLOCK("64"),
    BF16_MMVF_ROWS_FOR_BLOCK("96"),
    BF16_MMVF_ROWS_FOR_BLOCK("128"),
    BF16_MMVF_ROWS_FOR_BLOCK("160"),
    BF16_MMVF_ROWS_FOR_BLOCK("192"),
    BF16_MMVF_ROWS_FOR_BLOCK("224"),
    BF16_MMVF_ROWS_FOR_BLOCK("256"),

    // 5. Real __global__ kernel symbols across src/kernels/cuda/ mapped to their contract families
    ALIAS_ROW("copy_i32_from_mapped_kernel", "copy_i32_from_mapped_kernel"),
    ALIAS_ROW("copy_i32_from_mapped", "copy_i32_from_mapped_kernel"),
    ALIAS_ROW("copy_step_pos_from_mapped_kernel", "copy_step_pos_from_mapped_kernel"),
    ALIAS_ROW("copy_step_pos_from_mapped", "copy_step_pos_from_mapped_kernel"),
    ALIAS_ROW("embedding_gather_dev_kernel", "embedding_gather_dev_kernel"),
    ALIAS_ROW("embedding_gather_dev", "embedding_gather_dev_kernel"),
    ALIAS_ROW("broadcast_streams_kernel", "broadcast_streams_kernel"),
    ALIAS_ROW("broadcast_streams", "broadcast_streams_kernel"),
    ALIAS_ROW("gdn_conv_l2_multi_kernel", "gdn_conv_l2_multi_kernel"),
    ALIAS_ROW("gdn_conv_l2_multi", "gdn_conv_l2_multi_kernel"),
    ALIAS_ROW("gdn_ab_multi_kernel", "gdn_ab_multi_kernel"),
    ALIAS_ROW("gdn_ab_multi_kernel<1, true>", "gdn_ab_multi_kernel"),
    ALIAS_ROW("gdn_ab_multi_kernel<2, true>", "gdn_ab_multi_kernel"),
    ALIAS_ROW("gdn_ab_multi_kernel<3, true>", "gdn_ab_multi_kernel"),
    ALIAS_ROW("gdn_ab_multi_kernel<4, true>", "gdn_ab_multi_kernel"),
    ALIAS_ROW("gdn_ab_multi_kernel<5, true>", "gdn_ab_multi_kernel"),
    ALIAS_ROW("gdn_ab_multi_kernel<6, true>", "gdn_ab_multi_kernel"),
    ALIAS_ROW("gdn_ab_multi_kernel<8, false>", "gdn_ab_multi_kernel"),
    ALIAS_ROW("gdn_ab_multi_kernel<16, false>", "gdn_ab_multi_kernel"),
    ALIAS_ROW("gdn_ab_multi", "gdn_ab_multi_kernel"),
    ALIAS_ROW("gdn_step_norm_multi_kernel", "gdn_step_norm_multi_kernel"),
    ALIAS_ROW("gdn_step_norm_multi_kernel<false, false>", "gdn_step_norm_multi_kernel"),
    ALIAS_ROW("gdn_step_norm_multi_kernel<true, false>", "gdn_step_norm_multi_kernel"),
    ALIAS_ROW("gdn_step_norm_multi_kernel<true, true>", "gdn_step_norm_multi_kernel"),
    ALIAS_ROW("gdn_step_norm_multi", "gdn_step_norm_multi_kernel"),
    ALIAS_ROW("norm_rope_kernel", "norm_rope_kernel"),
    ALIAS_ROW("norm_rope_kernel<false>", "norm_rope_kernel"),
    ALIAS_ROW("norm_rope_kernel<true>", "norm_rope_kernel"),
    ALIAS_ROW("native_qsa_rms_norm_rope", "norm_rope_kernel"),
    ALIAS_ROW("fwht256_pair_kernel", "fwht256_pair_kernel"),
    ALIAS_ROW("fwht256_pair_inplace_cuda", "fwht256_pair_kernel"),
    ALIAS_ROW("append", "append"),
    ALIAS_ROW("append<false>", "append"),
    ALIAS_ROW("append<true>", "append"),
    ALIAS_ROW("native_qsa_indexer_append_steps", "append"),
    ALIAS_ROW("block_scores_kernel", "block_scores_kernel"),
    ALIAS_ROW("qsa_block_scores", "block_scores_kernel"),
    ALIAS_ROW("block_topk_kernel", "block_topk_kernel"),
    ALIAS_ROW("qsa_block_topk", "block_topk_kernel"),
    ALIAS_ROW("resolve_kernel", "resolve_kernel"),
    ALIAS_ROW("qsa_kv_resolve", "resolve_kernel"),
    ALIAS_ROW("attn_chunk_kernel", "attn_chunk_kernel"),
    ALIAS_ROW("attn_chunk_kernel<1, false, false, false>", "attn_chunk_kernel"),
    ALIAS_ROW("attn_chunk_kernel<1, true, false, false>", "attn_chunk_kernel"),
    ALIAS_ROW("qsa_decode_attn_batch_int8", "attn_chunk_kernel"),
    ALIAS_ROW("gate_quantize_q8_1_kernel", "gate_quantize_q8_1_kernel"),
    ALIAS_ROW("native_qsa_gate_quantize_q8_1", "gate_quantize_q8_1_kernel"),
    ALIAS_ROW("route_multi", "route_multi"),
    ALIAS_ROW("native_router_top10_multi", "route_multi"),
    ALIAS_ROW("resident_plan_kernel", "resident_plan_kernel"),
    ALIAS_ROW("resident_plan_kernel<128>", "resident_plan_kernel"),
    ALIAS_ROW("resident_plan_kernel<256>", "resident_plan_kernel"),
    ALIAS_ROW("resident_plan_all_resident", "resident_plan_kernel"),
    ALIAS_ROW("quantize_q8_1_kernel", "quantize_q8_1_kernel"),
    ALIAS_ROW("quantize_q8_1_rows", "quantize_q8_1_kernel"),
    ALIAS_ROW("combine", "combine"),
    ALIAS_ROW("combine<false, false>", "combine"),
    ALIAS_ROW("combine<true, false>", "combine"),
    ALIAS_ROW("combine<false, true>", "combine"),
    ALIAS_ROW("combine<true, true>", "combine"),
    ALIAS_ROW("native_moe_combine_multi", "combine"),
    ALIAS_ROW("post_multi", "post_multi"),
    ALIAS_ROW("gr_write_multi", "post_multi"),
    ALIAS_ROW("argmax_rows_kernel", "argmax_rows_kernel"),
    ALIAS_ROW("argmax_rows", "argmax_rows_kernel"),
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
