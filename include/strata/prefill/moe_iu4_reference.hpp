// SPDX-License-Identifier: MIT
// Port note: hipfire block_i4_128_quant.hip portable eight-candidate oracle.
#pragma once
#if defined(STRATA_MOE_IU4_CPU_ORACLE)
#include "strata/prefill/moe_iu4_gfx.hpp"
#include "strata/prefill/moe_iu8_reference.hpp"

namespace strata::prefill::iu4::reference {
namespace r8 = iu8::reference;
inline float add(float a, float b) { volatile float c = a + b; return c; }
inline int nearest(float x, float d) {
    return (int) std::fmin(std::fmax(std::nearbyint(r8::divide(x, d)), -8.0f), 7.0f);
}
inline Block quantize_block(const float* x, bool owned = false) {
    float amax = 0.0f;
    for (int k = 0; k < 128; ++k) {
        r8::require(std::isfinite(x[k]), "A4 refuses nonfinite input");
        amax = std::fmax(amax, std::fabs(x[k]));
    }
    float best = 1.0e30f, d = 1.0f;
    if (amax != 0.0f)
    for (int ci = 0; ci < 8; ++ci) {
        const float candidate = r8::mul(r8::mul(r8::divide(amax, 7.0f), 0.5f),
                                         add(1.0f, r8::divide((float) ci, 7.0f)));
        float mse[32] = {};
        for (int lane = 0; lane < 32; ++lane)
        for (int j = 0; j < 4; ++j) {
            const int k = lane * 4 + j;
            const float error = std::fma(-(float) nearest(x[k], candidate), candidate, x[k]);
            mse[lane] = std::fma(error, error, mse[lane]);
        }
        if (!owned) {
            for (int mask = 16; mask; mask >>= 1) {
                float next[32];
                for (int l = 0; l < 32; ++l) next[l] = add(mse[l], mse[l ^ mask]);
                std::copy_n(next, 32, mse);
            }
        } else {
            float v[2][16];
            for (int l = 0; l < 16; ++l) { v[0][l] = mse[2*l]; v[1][l] = mse[2*l+1]; }
            for (int mask = 8; mask; mask >>= 1)
            for (int b = 0; b < 2; ++b) {
                float next[16];
                for (int l = 0; l < 16; ++l) next[l] = add(v[b][l], v[b][l ^ mask]);
                std::copy_n(next, 16, v[b]);
            }
            mse[0] = add(v[0][0], v[1][0]);
        }
        if (mse[0] < best) { best = mse[0]; d = candidate; }
    }
    Block out{d, 0, {}};
    for (int k = 0; k < 128; ++k) {
        const int q = amax == 0.0f ? 0 : nearest(x[k], d);
        out.qs[k / 2] |= (uint8_t) ((q & 15) << (4 * (k & 1))); out.s += q;
    }
    return out;
}
inline std::vector<Block> quantize(const float* x, int k, int n) {
    r8::require(k > 0 && k % 128 == 0 && n > 0, "A4 oracle needs K128");
    std::vector<Block> out((size_t) (k / 128) * n);
    for (int t = 0; t < n; ++t)
    for (int kb = 0; kb < k / 128; ++kb)
        out[(size_t) kb * n + t] = quantize_block(x + (size_t) t * k + kb * 128);
    return out;
}
}  // namespace strata::prefill::iu4::reference
#endif
