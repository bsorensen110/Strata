// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// Port note: hipfire block_i8_128_quant.hip (reference divide and byte order);
// block_i4_128_quant.hip:220-343 (producer half-wave DPP/reciprocal/fallback).
// Adapted for A8 with a wider, conservative midpoint exclusion, not A4's bound.
#pragma once
#if defined(__HIPCC__)
#include "strata/prefill/moe_iu8_gfx.hpp"
#include <hip/hip_runtime.h>
#include <cmath>

namespace strata::prefill::iu8 {
struct OwnedBlock {
    float d;
    int s;
    int8_t q[8];
};
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1100__) || defined(__gfx1151__) || \
    defined(__gfx1200__) || defined(__gfx1201__)
template<int Mask>
__device__ __forceinline__ float row_max_peer(float v) {
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x160 | Mask, 15, 15, true));
}
template<int Mask>
__device__ __forceinline__ int row_sum_peer(int v) {
    return __builtin_amdgcn_update_dpp(0, v, 0x160 | Mask, 15, 15, true);
}
// Each 16-lane DPP row owns a K128 block; lane j owns x[8j..8j+7].
__device__ __forceinline__ OwnedBlock quantize_owned(const float x[8]) {
    float amax = 0.0f;
    bool finite = true;
    for (int k = 0; k < 8; ++k) {
        amax = fmaxf(amax, fabsf(x[k]));
        finite = finite && isfinite(x[k]);
    }
    if (__builtin_amdgcn_ballot_w32(!finite)) __builtin_trap();
    amax = fmaxf(amax, row_max_peer<8>(amax));
    amax = fmaxf(amax, row_max_peer<4>(amax));
    amax = fmaxf(amax, row_max_peer<2>(amax));
    amax = fmaxf(amax, row_max_peer<1>(amax));
    const bool zero = amax == 0.0f;
    OwnedBlock out{};
    out.d = zero ? 0.0f : __fdiv_rn(amax, 127.0f);
    const float d = zero ? 1.0f : out.d;
    float reciprocal = __builtin_amdgcn_rcpf(d);
    reciprocal = __fmaf_rn(reciprocal, __fmaf_rn(-d, reciprocal, 1.0f), reciprocal);
    float q[8], frac = 0.0f;
    for (int k = 0; k < 8; ++k) {
        const float y = zero ? 0.0f : __fmul_rn(x[k], reciprocal);
        q[k] = rintf(y);
        frac = fmaxf(frac, fabsf(y - q[k]));
    }
    // At |x/d|<=127, reciprocal refinement + RN multiply err <2^-14.
    // Margin 2^-12 also covers RN divide error (<=2^-18). Subnormals
    // and midpoint-near values redo IEEE divides for the whole wave.
    const bool exact = zero || (amax >= 0x1p-100f && frac <= 0.5f - 0x1p-12f);
    if (__builtin_amdgcn_ballot_w32(!exact)) {
        for (int k = 0; k < 8; ++k) q[k] = zero ? 0.0f : rintf(__fdiv_rn(x[k], out.d));
    }
    out.s = 0;
    for (int k = 0; k < 8; ++k) {
        out.q[k] = (int8_t) (int) fminf(fmaxf(q[k], -127.0f), 127.0f);
        out.s += out.q[k];
    }
    out.s += row_sum_peer<8>(out.s);
    out.s += row_sum_peer<4>(out.s);
    out.s += row_sum_peer<2>(out.s);
    out.s += row_sum_peer<1>(out.s);
    return out;
}
#else
__device__ __forceinline__ OwnedBlock quantize_owned(const float[8]) {
    __builtin_trap();
    return {};
}
#endif

__device__ __forceinline__ void store_owned(const OwnedBlock& v, Block* out, int lane) {
    if (!out) return;
    const int j = lane & 15;
    for (int k = 0; k < 8; ++k) out->qs[8 * j + ((k & 1) ? 4 : 0) + (k >> 1)] = v.q[k];
    if (j == 0) { out->d = v.d; out->s = v.s; }
}
}  // namespace strata::prefill::iu8
#endif
