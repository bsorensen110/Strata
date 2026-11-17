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
inline float fold_block(float acc, float sc, float zp, const Block& x, int c, bool symmetric) {
    const float t1 = r8::mul(sc, x.d);
    const float p = r8::mul(t1, (float) c);
    const float term = symmetric ? p : std::fma(r8::mul(zp, x.d), (float) x.s, p);
    return add(acc, term);
}
inline float fold(const iu8::Pack& w, const Block* x, int n, int token, int row, bool symmetric,
                   const uint16_t* zp = nullptr) {
    float acc = 0.0f;
    for (int kb = 0; kb < w.cols / 128; ++kb) {
        const auto& b = x[(size_t) kb * n + token];
        int c = 0;
        for (int k = 0; k < 128; ++k)
            c += (iu8::signed_code(w, (size_t) row * w.cols + kb * 128 + k) + (symmetric ? 0 : 8)) * code(b, k);
        const size_t h = (size_t) row * (w.cols / 128) + kb;
        const float sc = iu8::half_value(w.scales[h]);
        acc = fold_block(acc, sc, zp ? iu8::half_value(zp[h]) : r8::mul(-8.0f, sc), b, c, symmetric);
    }
    return acc;
}
inline int fragment_byte(int row, int k) {
    return ((row / 16) * 2 + k / 32) * 256 +
        ((row % 16) + 16 * ((k % 32) / 16)) * 8 + (k % 16) / 2;
}
inline int nibble(uint8_t b, int k, bool sign) {
    const int u = (b >> (4 * (k & 1))) & 15;
    return sign && u >= 8 ? u - 16 : u;
}
// Contract emulation uses physically staged bytes, not scalar weight dots.
// The K32 fragment and K16 row-stride loads intentionally have separate code.
inline void emulate(const iu8::Pack& w, const Block* x, int n, bool gfx12, bool symmetric,
                     float* y, int ld, const uint16_t* zp = nullptr) {
    for (int row0 = 0; row0 < w.rows; row0 += 128)
    for (int token0 = 0; token0 < n; token0 += 128) {
        std::vector<float> acc(128 * 128);
        for (int kb = 0; kb < w.cols / 128; ++kb) {
            std::vector<int> dots(128 * 128);
            if (gfx12) {
                for (int slab = 0; slab < 2; ++slab) {
                    uint8_t ws[4096] = {}, xs[4096] = {};
                    for (int r = 0; r < 128; ++r)
                    for (int k = 0; k < 64; k += 2) {
                        const int offset = fragment_byte(r, k);
                        if (row0+r < w.rows) {
                            const size_t i = (size_t) (row0+r) * w.cols + kb*128 + slab*64 + k;
                            ws[offset] = w.codes[i/2] ^ (symmetric ? 0x88 : 0);
                        }
                        if (token0+r < n) xs[offset] = x[(size_t) kb*n+token0+r].qs[slab*32+k/2];
                    }
                    for (int r = 0; r < 128; ++r)
                    for (int t = 0; t < 128; ++t)
                    for (int step = 0; step < 2; ++step)
                    for (int half = 0; half < 2; ++half)
                    for (int k = 0; k < 16; ++k) {
                        const int rk = step*32 + half*16 + k;
                        dots[t*128+r] += nibble(ws[fragment_byte(r, rk)], k, symmetric) *
                                         nibble(xs[fragment_byte(t, rk)], k, true);
                    }
                }
            } else {
                std::vector<uint8_t> ws(128*168), xs(128*72);
                for (int r = 0; r < 128; ++r)
                for (int k = 0; k < 128; k += 2) {
                    if (row0+r < w.rows) ws[r*168+(kb&1)*64+k/2] =
                        w.codes[((size_t)(row0+r)*w.cols+kb*128+k)/2] ^ (symmetric ? 0x88 : 0);
                    if (token0+r < n) xs[r*72+8+k/2] = x[(size_t)kb*n+token0+r].qs[k/2];
                }
                for (int r = 0; r < 128; ++r)
                for (int t = 0; t < 128; ++t)
                for (int step = 0; step < 8; ++step)
                for (int k = 0; k < 16; ++k) {
                    const int rk = step*16+k;
                    dots[t*128+r] += nibble(ws[r*168+(kb&1)*64+rk/2], k, symmetric) *
                                     nibble(xs[t*72+8+rk/2], k, true);
                }
            }
            for (int wave = 0; wave < 8; ++wave)
            for (int lane = 0; lane < 32; ++lane)
            for (int nb = 0; nb < 4; ++nb)
            for (int rg = 0; rg < 2; ++rg)
            for (int j = 0; j < 8; ++j) {
                const int t = wave%2*64+nb*16+(lane&15);
                const int r = wave/2*32+rg*16+(gfx12 ? 8*(lane>>4)+j : 2*j+(lane>>4));
                if (token0+t >= n || row0+r >= w.rows) continue;
                const size_t h = (size_t)(row0+r)*(w.cols/128)+kb;
                const float sc = iu8::half_value(w.scales[h]);
                acc[t*128+r] = fold_block(acc[t*128+r], sc,
                    zp ? iu8::half_value(zp[h]) : r8::mul(-8.0f, sc),
                    x[(size_t)kb*n+token0+t], dots[t*128+r], symmetric);
            }
        }
        for (int t = 0; t < 128 && token0+t < n; ++t)
        for (int r = 0; r < 128 && row0+r < w.rows; ++r)
            y[(size_t)(token0+t)*ld+row0+r] = acc[t*128+r];
    }
}
}  // namespace strata::prefill::iu4::reference
#endif
