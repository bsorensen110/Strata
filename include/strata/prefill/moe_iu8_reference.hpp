// SPDX-License-Identifier: MIT
// Port note: hipfire block_i8_128_quant.hip and gemm_mq6sym_i8.gfx1201.hip.
// Test-only, scalar IEEE oracle; never included by engine translation units.
#pragma once
#if defined(STRATA_MOE_IU8_CPU_ORACLE)
#include "strata/prefill/moe_iu8_gfx.hpp"
#include "strata/prefill/moe_iu8_pack.hpp"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace strata::prefill::iu8::reference {
inline float mul(float a, float b) { volatile float v = a * b; return v; }
inline float divide(float a, float b) { volatile float v = a / b; return v; }
inline void require(bool b, const char* message) { if (!b) throw std::runtime_error(message); }

inline std::vector<Block> quantize(const float* x, int k, int n) {
    require(k > 0 && k % 128 == 0 && n > 0, "oracle needs K128");
    std::vector<Block> out((size_t) (k / 128) * n);
    for (int row = 0; row < n; ++row)
    for (int kb = 0; kb < k / 128; ++kb) {
        Block& b = out[(size_t) kb * n + row];
        const float* v = x + (size_t) row * k + kb * 128;
        float amax = 0.0f;
        for (int j = 0; j < 128; ++j) {
            require(std::isfinite(v[j]), "oracle refuses nonfinite activation");
            amax = std::fmax(amax, std::fabs(v[j]));
        }
        b.d = amax == 0.0f ? 0.0f : divide(amax, 127.0f);
        b.s = 0;
        for (int j = 0; j < 128; ++j) {
            const float q = amax == 0.0f ? 0.0f :
                std::fmin(std::fmax(std::nearbyint(divide(v[j], b.d)), -127.0f), 127.0f);
            b.qs[byte_index(j)] = (int8_t) (int) q;
            b.s += (int) q;
        }
    }
    return out;
}

inline float fold(const Pack& w, const Block* x, int n, int token, int row) {
    float acc = 0.0f;
    for (int kb = 0; kb < w.cols / 128; ++kb) {
        const Block& b = x[(size_t) kb * n + token];
        int32_t c = 0;
        for (int j = 0; j < 128; ++j)
            c += signed_code(w, (size_t) row * w.cols + kb * 128 + j) * b.qs[byte_index(j)];
        const float sd = mul(half_value(w.scales[(size_t) row * (w.cols / 128) + kb]), b.d);
        acc = std::fma(sd, (float) c, acc);
    }

    // Emulate the tile's lane halves and K16 instructions independently of fold().
    inline void narrow(const Pack& w, const Block* x, int n, int entry, bool gfx12, float* y, int ld) {
        const auto tile = kNarrowTiles.at((size_t) entry);
        require(n <= tile.tokens || tile.split, "oracle tile capacity");
        for (int row0 = 0; row0 < w.rows; row0 += tile.rows)
        for (int token0 = 0; token0 < n; token0 += tile.tokens)
        for (int wave = 0; wave < tile.rows / 16; ++wave)
        for (int lane = 0; lane < 32; ++lane)
        for (int tg = 0; tg < tile.tokens / 16; ++tg)
        for (int j = 0; j < 8; ++j) {
            const int row = row0 + wave * 16 + (lane & 15);
            const int token = token0 + tg * 16 + (gfx12 ? 8 * (lane >> 4) + j : 2 * j + (lane >> 4));
            if (row >= w.rows || token >= n) continue;
            float acc = 0.0f;
            for (int kb = 0; kb < w.cols / 128; ++kb) {
                const auto& b = x[(size_t) kb * n + token];
                int c = 0;
                for (int step = 0; step < 8; ++step)
                for (int h = 0; h < (gfx12 ? 2 : 1); ++h)
                for (int q = 0; q < (gfx12 ? 8 : 16); ++q) {
                    const int k = gfx12 ? 64 * h + 8 * step + q : 16 * step + q;
                    c += signed_code(w, (size_t) row * w.cols + kb * 128 + k) * b.qs[byte_index(k)];
                }
                acc = std::fma(mul(half_value(w.scales[(size_t) row * (w.cols / 128) + kb]), b.d), (float) c, acc);
            }
            y[(size_t) token * ld + row] = acc;
        }
    }
    return acc;
}

inline std::vector<float> activations(int k, int n, uint32_t seed = 41) {
    std::mt19937 rng(seed);
    std::vector<float> out((size_t) k * n);
    for (float& x : out) x = (float) ((int) (rng() & 65535) - 32768) / 32768.0f;
    std::fill_n(out.data(), k, 0.0f);
    return out;
}

inline std::vector<uint8_t> source(int type, int m, int k, uint32_t seed = 17) {
    require((type == GGML_TYPE_Q4_0 && k % 128 == 0) || (type == GGML_TYPE_Q6_K && k % 256 == 0),
            "synthetic weight geometry");
    std::vector<uint8_t> out((size_t) m * ggml_row_size((ggml_type) type, k));
    std::mt19937 rng(seed);
    if (type == GGML_TYPE_Q4_0) {
        for (int r = 0; r < m; ++r)
        for (int b = 0; b < k / 32; ++b) {
            uint8_t* p = out.data() + ((size_t) r * (k / 32) + b) * 18;
            const uint16_t scale = (uint16_t) (0x2800 + ((r + b / 4) % 17) * 16);
            std::memcpy(p, &scale, 2);
            for (int j = 0; j < 16; ++j) p[2 + j] = (uint8_t) rng();
        }
    } else {
        // GGML Q6_K: ql[128], qh[64], scales[16], f16 d. Unit subscales
        // make each K128 exactly symmetric, without changing source quantization.
        for (size_t off = 0; off < out.size(); off += 210) {
            for (int j = 0; j < 192; ++j) out[off + j] = (uint8_t) rng();
            std::fill_n(out.data() + off + 192, 16, uint8_t{1});
            const uint16_t scale = (uint16_t) (0x2400 + ((off / 210) % 19) * 16);
            std::memcpy(out.data() + off + 208, &scale, 2);
        }
    }
    return out;
}

inline double rms_f16(const Pack& w, const std::vector<float>& x, int n, const std::vector<float>& y) {
    double num = 0.0, den = 0.0;
    for (int token = 0; token < n; ++token)
    for (int row = 0; row < w.rows; ++row) {
        double ref = 0;
        for (int k = 0; k < w.cols; ++k) {
            const size_t i = (size_t) row * w.cols + k;
            const float weight = mul(half_value(w.scales[i / 128]), (float) signed_code(w, i));
            const float activation = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x[(size_t) token * w.cols + k]));
            ref += (double) weight * activation;
        }
        const double err = y[(size_t) token * w.rows + row] - ref;
        num += err * err; den += ref * ref;
    }
    return den == 0 ? (num == 0 ? 0.0 : INFINITY) : std::sqrt(num / den);
}
}  // namespace strata::prefill::iu8::reference
#endif
