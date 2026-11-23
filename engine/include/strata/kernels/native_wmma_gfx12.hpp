// include/strata/kernels/native_wmma_gfx12.hpp - RDNA4 (gfx12 wave32) INT8 WMMA kernels and lane-mapping oracle.
#pragma once

#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cstdint>

namespace strata::kernels {

/// True when running on an RDNA4 (gfx1200 / gfx1201) HIP build with wave32 INT8 WMMA compiled in.
bool native_wmma_gfx12_available() noexcept;

/// Dense Q8_0 x Q8_1 via gfx12 16x16x16 INT8 wave32 WMMA (for wide verify windows, typically ncols >= 9).
/// Returns true if launched, false if unavailable/ineligible so caller falls back cleanly.
bool native_wmma_dense_q8_0(const void* weights, const void* x_q8_1, float* y,
                            int n_in, int n_out, int ncols, void* stream);

/// Multi-segment dense Q8_0 (QKV / gate+up fused launch) via gfx12 INT8 wave32 WMMA.
bool native_wmma_dense_q8_0_segs(const NativeMmvqSeg* segs, int n_segs, const void* x_q8_1,
                                 int n_in, int ncols, void* stream);

/// Dense Q6_K x Q8_1 via gfx12 16x16x16 INT8 wave32 WMMA.
bool native_wmma_dense_q6_k(const void* weights, const void* x_q8_1, float* y,
                            int n_in, int n_out, int ncols, void* stream);

/// Grouped MoE Q8_0/Q6_K helper for wide groups via gfx12 INT8 wave32 WMMA (returns false if not handled).
bool native_wmma_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr,
                                const int32_t* grp_start, const int32_t* n_groups,
                                const int32_t* ent_dst, const int32_t* ent_tok,
                                int64_t cap_groups, int64_t cap_entries,
                                const void* x_q8_1, void* scratch, float* out, void* stream);

/// Software oracle modeling exact gfx12 wave32 `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12` lane layout:
/// lane l: l16 = l & 15, hi = l >> 4, A[l16][8*hi+j], B[8*hi+j][l16], D[8*hi+i][l16] (i,j = 0..7).
void native_wmma_gfx12_emulate_16x16x16_i8(const int8_t a_lane[32][8], const int8_t b_lane[32][8],
                                           int32_t d_lane[32][8]);

/// Tiled Q8_0 x Q8_1 CPU reference executing the exact gfx12 16x16x16 WMMA lane packing and scale broadcast.
void native_wmma_gfx12_q8_0_cpu_ref_tiled(const void* weights, const void* x_q8_1, float* y,
                                          int n_in, int n_out, int ncols);

/// Tiled Q6_K x Q8_1 CPU reference executing the exact gfx12 16x16x16 WMMA sub-block unpacking and scale broadcast.
void native_wmma_gfx12_q6_k_cpu_ref_tiled(const void* weights, const void* x_q8_1, float* y,
                                          int n_in, int n_out, int ncols);

}  // namespace strata::kernels
