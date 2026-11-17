// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniil Markevich
// Port note: hipfire block_i4_128_quant.hip:53-216,220-343.
// The producer-owned DPP emit retains the full reference 8-candidate search.
#pragma once
#if defined(__HIPCC__)
#include "strata/prefill/moe_iu4_gfx.hpp"
#include "strata/prefill/moe_iu8_quant.cuh"

namespace strata::prefill::iu4 {
struct OwnedBlock { float d; int s; uint32_t qs; };
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1100__) || defined(__gfx1151__) || \
    defined(__gfx1200__) || defined(__gfx1201__)
__device__ __forceinline__ float row_sum(float v) {
    v = __fadd_rn(v, iu8::row_max_peer<8>(v));
    v = __fadd_rn(v, iu8::row_max_peer<4>(v));
    v = __fadd_rn(v, iu8::row_max_peer<2>(v));
    return __fadd_rn(v, iu8::row_max_peer<1>(v));
}
// Norm/rotate and SwiGLU producers can pass their eight live register values
// directly. Each 16-lane row owns one K128 block; no ownership remap or LDS.
__device__ __forceinline__ OwnedBlock quantize_owned(const float x[8]) {
    float amax = 0.0f;
    bool finite = true;
    for (int k = 0; k < 8; ++k) {
        amax = fmaxf(amax, fabsf(x[k])); finite = finite && isfinite(x[k]);
    }
    if (__builtin_amdgcn_ballot_w32(!finite)) __builtin_trap();
    amax = fmaxf(amax, iu8::row_max_peer<8>(amax));
    amax = fmaxf(amax, iu8::row_max_peer<4>(amax));
    amax = fmaxf(amax, iu8::row_max_peer<2>(amax));
    amax = fmaxf(amax, iu8::row_max_peer<1>(amax));
    const bool zero = amax == 0.0f;
    float best_d = 1.0f, best_mse = 1.0e30f;
    float best_q[8];
    for (int k = 0; k < 8; ++k)
        best_q[k] = zero ? 0.0f : fminf(fmaxf(rintf(x[k]), -8.0f), 7.0f);
    for (int ci = 0; ci < kCandidates; ++ci) {
        const float d = __fmul_rn(__fmul_rn(__fdiv_rn(amax, 7.0f), 0.5f),
                                  __fadd_rn(1.0f, __fdiv_rn((float) ci, 7.0f)));
        const float safe_d = zero ? 1.0f : d;
        float reciprocal = __builtin_amdgcn_rcpf(safe_d);
        reciprocal = __fmaf_rn(reciprocal, __fmaf_rn(-safe_d, reciprocal, 1.0f), reciprocal);
        float q[8], frac = 0.0f;
        for (int k = 0; k < 8; ++k) {
            const float y = zero ? 0.0f : __fmul_rn(x[k], reciprocal);
            q[k] = rintf(y); frac = fmaxf(frac, fabsf(y - q[k]));
        }
        // Full-grid |x/d|<=14, unlike C2's 8.17. The wider 2^-16
        // exclusion also covers divide rounding; tiny blocks use IEEE divides.
        const bool exact = zero || (amax >= 0x1p-100f && frac <= 0.5f - 0x1p-16f);
        if (__builtin_amdgcn_ballot_w32(!exact))
            for (int k = 0; k < 8; ++k) q[k] = zero ? 0.0f : rintf(__fdiv_rn(x[k], d));
        float mse[2] = {};
        for (int k = 0; k < 8; ++k) {
            q[k] = fminf(fmaxf(q[k], -8.0f), 7.0f);
            const float error = __fmaf_rn(-q[k], d, x[k]);
            mse[k >> 2] = __fmaf_rn(error, error, mse[k >> 2]);
        }
        const float error = __fadd_rn(row_sum(mse[0]), row_sum(mse[1]));
        if (!zero && error < best_mse) {
            best_mse = error; best_d = d;
            for (int k = 0; k < 8; ++k) best_q[k] = q[k];
        }
    }
    OwnedBlock out{best_d, 0, 0};
    for (int k = 0; k < 8; ++k) {
        const int q = (int) best_q[k];
        out.qs |= (uint32_t) (q & 15) << (4 * k); out.s += q;
    }
    out.s += iu8::row_sum_peer<8>(out.s);
    out.s += iu8::row_sum_peer<4>(out.s);
    out.s += iu8::row_sum_peer<2>(out.s);
    out.s += iu8::row_sum_peer<1>(out.s);
    return out;
}
#else
__device__ __forceinline__ OwnedBlock quantize_owned(const float[8]) { __builtin_trap(); return {}; }
#endif
__device__ __forceinline__ void store_owned(const OwnedBlock& q, Block* out, int lane) {
    ((uint32_t*) out->qs)[lane & 15] = q.qs;
    if ((lane & 15) == 0) { out->d = q.d; out->s = q.s; }
}
}  // namespace strata::prefill::iu4
#endif
