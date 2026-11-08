// src/kernels/cpu/iq_avx512.cpp - plan v0.3 P6: the i-quant expert rows in 512-bit lanes, several tokens at once.
//
// ggml-cpu's x86 dot products for these formats are AVX2 and single-token: every token re-decodes the weights
// and a sign is applied with five instructions per 32 values.  Here 64 values are decoded once (8 or 16 grid
// lookups, one 64-bit sign mask, one vector of scales) and every token applies them with a masked subtract, a
// `maddubs` and a `madd`.  The arithmetic is ggml's (ggml-cpu/quants.c, the `_generic` references): integer sums
// per block, times d_x * d_y * the format's constant - only the order of the float additions differs.
//
// Formats: IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ3_S (21), IQ2_S (22).  IQ1_M stays on ggml-cpu (no shipped
// model has IQ1_M expert rows: the 'Coder IQ1_M' pack's gate/up are IQ2_S / IQ3_XXS / IQ3_S).
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::kernels::cpu {
namespace {

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
inline uint32_t u32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint16_t u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline uint64_t u64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// E-2: the IQ3 grids by one AVX-512 gather (the same words, the same results) - measured 3-5% SLOWER than the scalar
// lookups on a Ryzen 5 7600 (Zen 4 gathers are microcoded), so opt-in: STRATA_IQ_GATHER=1
// 0.2.5: DEFAULT on where gathers are fast (Intel server and HEDT cores with AVX-512: cpu_gather_fast), off on Zen 4 (microcoded);
// STRATA_IQ_GATHER=1 / 0 force it either way.  The same words are loaded, so the results are bit for bit the scalar decode's.
const bool gather = [] {
    const char* v = std::getenv("STRATA_IQ_GATHER");
    return v != nullptr && v[0] != '\0' ? std::atoi(v) != 0 : cpu_gather_fast();
}();
// the grid lookups by one 64-bit gather of eight entries (IQ2_XXS / IQ2_XS / IQ2_S): `idx` = eight 32-bit indices
inline __m512i gather64(__m256i idx, const void* grid) { return _mm512_i32gather_epi64(idx, grid, 8); }
// E-2: software prefetch distance in bytes (STRATA_IQ_PREFETCH; 0 = off): -2-3% gate/up time on IQ3_S decode
// (the rows are read at ~30 GB/s by 6 cores, near what DDR5 with 4 KB pages gives: the pool is memory-bound)
const int prefetch_ahead = [] {
    const char* v = std::getenv("STRATA_IQ_PREFETCH");
    return v ? std::atoi(v) : 2048;
}();

// lanes 0-7 -> s0, 8-15 -> s1, 16-23 -> s2, 24-31 -> s3 (int16 lanes of a maddubs result: two values each)
inline __m512i scales4(int s0, int s1, int s2, int s3) {
    const __m512i idx = _mm512_set_epi16(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
                                         0, 0, 0, 0, 0, 0, 0, 0);
    const uint64_t packed = (uint64_t) (uint16_t) s0 | ((uint64_t) (uint16_t) s1 << 16) |
                            ((uint64_t) (uint16_t) s2 << 32) | ((uint64_t) (uint16_t) s3 << 48);
    return _mm512_permutexvar_epi16(idx, _mm512_set1_epi64((long long) packed));
}

// ---- per format: 64 values (chunk j of a 256-value block) -> grid magnitudes, sign mask, scales
template <int TY> struct Fmt;

template <> struct Fmt<16> {   // IQ2_XXS: d, qs[32] u16
    static constexpr int bytes = 66;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        const uint32_t a0 = u32(q), a1 = u32(q + 4), b0 = u32(q + 8), b1 = u32(q + 12);
        g = gather ? gather64(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long) ((uint64_t) a0 | ((uint64_t) b0 << 32)))), iq2xxs_grid)
                   : _mm512_set_epi64((long long) iq2xxs_grid[b0 >> 24], (long long) iq2xxs_grid[(b0 >> 16) & 255],
                             (long long) iq2xxs_grid[(b0 >> 8) & 255], (long long) iq2xxs_grid[b0 & 255],
                             (long long) iq2xxs_grid[a0 >> 24], (long long) iq2xxs_grid[(a0 >> 16) & 255],
                             (long long) iq2xxs_grid[(a0 >> 8) & 255], (long long) iq2xxs_grid[a0 & 255]);
        const uint64_t s = (uint64_t) ksigns_iq2xs[a1 & 127] | ((uint64_t) ksigns_iq2xs[(a1 >> 7) & 127] << 8) |
                           ((uint64_t) ksigns_iq2xs[(a1 >> 14) & 127] << 16) | ((uint64_t) ksigns_iq2xs[(a1 >> 21) & 127] << 24) |
                           ((uint64_t) ksigns_iq2xs[b1 & 127] << 32) | ((uint64_t) ksigns_iq2xs[(b1 >> 7) & 127] << 40) |
                           ((uint64_t) ksigns_iq2xs[(b1 >> 14) & 127] << 48) | ((uint64_t) ksigns_iq2xs[(b1 >> 21) & 127] << 56);
        m = _cvtu64_mask64(s);
        const int sa = 2 * (int) (a1 >> 28) + 1, sb = 2 * (int) (b1 >> 28) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
};

template <> struct Fmt<17> {   // IQ2_XS: d, qs[32] u16 (9-bit grid index + 7-bit sign index), scales[8]
    static constexpr int bytes = 74;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        uint16_t v[8];
        std::memcpy(v, q, 16);
        g = gather ? gather64(_mm256_and_si256(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*) q)), _mm256_set1_epi32(511)), iq2xs_grid)
                   : _mm512_set_epi64((long long) iq2xs_grid[v[7] & 511], (long long) iq2xs_grid[v[6] & 511],
                             (long long) iq2xs_grid[v[5] & 511], (long long) iq2xs_grid[v[4] & 511],
                             (long long) iq2xs_grid[v[3] & 511], (long long) iq2xs_grid[v[2] & 511],
                             (long long) iq2xs_grid[v[1] & 511], (long long) iq2xs_grid[v[0] & 511]);
        uint64_t s = 0;
        for (int l = 0; l < 8; ++l) s |= (uint64_t) ksigns_iq2xs[v[l] >> 9] << (8 * l);
        m = _cvtu64_mask64(s);
        const uint8_t s0 = b[66 + 2 * j], s1 = b[66 + 2 * j + 1];
        sc = scales4(2 * (s0 & 15) + 1, 2 * (s0 >> 4) + 1, 2 * (s1 & 15) + 1, 2 * (s1 >> 4) + 1);
    }
};

template <> struct Fmt<22> {   // IQ2_S: d, qs[64] (32 grid bytes, 32 sign bytes), qh[8], scales[8]
    static constexpr int bytes = 82;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* qs = b + 2 + 8 * j;
        const uint8_t h0 = b[66 + 2 * j], h1 = b[66 + 2 * j + 1];
        if (gather) {   // the 10-bit indices: the byte plus two high bits from qh (h0 for lanes 0-3, h1 for 4-7), one 64-bit gather
            const __m256i hi = _mm256_and_si256(_mm256_sllv_epi32(_mm256_setr_epi32(h0, h0, h0, h0, h1, h1, h1, h1),
                                                                   _mm256_setr_epi32(8, 6, 4, 2, 8, 6, 4, 2)), _mm256_set1_epi32(0x300));
            g = gather64(_mm256_or_si256(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) qs)), hi), iq2s_grid);
        } else
        g = _mm512_set_epi64((long long) iq2s_grid[qs[7] | ((h1 << 2) & 0x300)], (long long) iq2s_grid[qs[6] | ((h1 << 4) & 0x300)],
                             (long long) iq2s_grid[qs[5] | ((h1 << 6) & 0x300)], (long long) iq2s_grid[qs[4] | ((h1 << 8) & 0x300)],
                             (long long) iq2s_grid[qs[3] | ((h0 << 2) & 0x300)], (long long) iq2s_grid[qs[2] | ((h0 << 4) & 0x300)],
                             (long long) iq2s_grid[qs[1] | ((h0 << 6) & 0x300)], (long long) iq2s_grid[qs[0] | ((h0 << 8) & 0x300)]);
        m = _cvtu64_mask64(u64(b + 2 + 32 + 8 * j));
        const uint8_t s0 = b[74 + 2 * j], s1 = b[74 + 2 * j + 1];
        sc = scales4(2 * (s0 & 15) + 1, 2 * (s0 >> 4) + 1, 2 * (s1 & 15) + 1, 2 * (s1 >> 4) + 1);
    }
};

template <> struct Fmt<18> {   // IQ3_XXS: d, qs[64] grid bytes, 8 x u32 (4 x 7-bit sign index + 4-bit scale)
    static constexpr int bytes = 98;
    static constexpr float K = 0.25f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        // E-2: one gather of the 16 grid words (the scalar lookups assembled with set_epi32 cost ~30 uops)
        g = gather ? _mm512_i32gather_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) q)), iq3xxs_grid, 4)
                   : _mm512_set_epi32((int) iq3xxs_grid[q[15]], (int) iq3xxs_grid[q[14]], (int) iq3xxs_grid[q[13]], (int) iq3xxs_grid[q[12]],
                             (int) iq3xxs_grid[q[11]], (int) iq3xxs_grid[q[10]], (int) iq3xxs_grid[q[9]], (int) iq3xxs_grid[q[8]],
                             (int) iq3xxs_grid[q[7]], (int) iq3xxs_grid[q[6]], (int) iq3xxs_grid[q[5]], (int) iq3xxs_grid[q[4]],
                             (int) iq3xxs_grid[q[3]], (int) iq3xxs_grid[q[2]], (int) iq3xxs_grid[q[1]], (int) iq3xxs_grid[q[0]]);
        const uint32_t a = u32(b + 2 + 64 + 8 * j), c = u32(b + 2 + 64 + 8 * j + 4);
        const uint64_t s = (uint64_t) ksigns_iq2xs[a & 127] | ((uint64_t) ksigns_iq2xs[(a >> 7) & 127] << 8) |
                           ((uint64_t) ksigns_iq2xs[(a >> 14) & 127] << 16) | ((uint64_t) ksigns_iq2xs[(a >> 21) & 127] << 24) |
                           ((uint64_t) ksigns_iq2xs[c & 127] << 32) | ((uint64_t) ksigns_iq2xs[(c >> 7) & 127] << 40) |
                           ((uint64_t) ksigns_iq2xs[(c >> 14) & 127] << 48) | ((uint64_t) ksigns_iq2xs[(c >> 21) & 127] << 56);
        m = _cvtu64_mask64(s);
        const int sa = 2 * (int) (a >> 28) + 1, sb = 2 * (int) (c >> 28) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
};

template <> struct Fmt<21> {   // IQ3_S: d, qs[64], qh[8], signs[32], scales[4]
    static constexpr int bytes = 110;
    static constexpr float K = 1.0f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        const uint32_t h0 = b[66 + 2 * j], h1 = b[66 + 2 * j + 1];
        if (gather) {   // E-2: the 9-bit indices (the high bit from qh under a mask), one gather
            __m512i idx = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) q));
            idx = _mm512_mask_add_epi32(idx, (__mmask16) (h0 | (h1 << 8)), idx, _mm512_set1_epi32(256));
            g = _mm512_i32gather_epi32(idx, iq3s_grid, 4);
        } else {
#define G3(k, h, kk) (int) iq3s_grid[q[k] | (((h >> kk) & 1) << 8)]
        g = _mm512_set_epi32(G3(15, h1, 7), G3(14, h1, 6), G3(13, h1, 5), G3(12, h1, 4),
                             G3(11, h1, 3), G3(10, h1, 2), G3(9, h1, 1), G3(8, h1, 0),
                             G3(7, h0, 7), G3(6, h0, 6), G3(5, h0, 5), G3(4, h0, 4),
                             G3(3, h0, 3), G3(2, h0, 2), G3(1, h0, 1), G3(0, h0, 0));
#undef G3
        }
        m = _cvtu64_mask64(u64(b + 74 + 8 * j));
        const uint8_t s = b[106 + j];
        const int sa = 2 * (s & 15) + 1, sb = 2 * (s >> 4) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
};

// IQ4_XS (23): d, scales_h (u16), scales_l[4], qs[128].  Chunk j = the 64 values of two 32-value sub-blocks (2j, 2j+1): the
// table lookup of the nibbles is one vpshufb in 512-bit lanes (kvalues_iq4nl broadcast to every lane), laid out as ggml's AVX2 dot lays
// out its two 256-bit halves (low nibbles of 16 bytes, then the high nibbles), so the integer lanes line up with that dot's.
// The decode of the nibble planes follows the layout of PR #5243 (bsorensen110); only this format's part is taken.
template <> struct Fmt<23> {
    static constexpr int bytes = 136;
    static constexpr float K = 1.0f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const __m128i b0 = _mm_loadu_si128((const __m128i*) (b + 8 + 32 * j));
        const __m128i b1 = _mm_loadu_si128((const __m128i*) (b + 24 + 32 * j));
        const __m128i m4 = _mm_set1_epi8(0x0f);
        const __m256i n0 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_and_si128(b0, m4)), _mm_and_si128(_mm_srli_epi16(b0, 4), m4), 1);
        const __m256i n1 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_and_si128(b1, m4)), _mm_and_si128(_mm_srli_epi16(b1, 4), m4), 1);
        const __m512i nib = _mm512_inserti64x4(_mm512_castsi256_si512(n0), n1, 1);
        const __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*) kvalues_iq4nl));
        const __m512i q4 = _mm512_shuffle_epi8(lut, nib);
        g = _mm512_abs_epi8(q4);
        m = _mm512_movepi8_mask(q4);
        const uint8_t sl = b[4 + j];
        const uint16_t sh = (uint16_t) (u16(b + 2) >> (4 * j));
        const int ls0 = ((sl & 0x0f) | ((sh << 4) & 0x30)) - 32;
        const int ls1 = ((sl >> 4) | ((sh << 2) & 0x30)) - 32;
        sc = scales4(ls0, ls0, ls1, ls1);
    }
};

template <bool Singleton> struct RowAccumulator;
template <> struct RowAccumulator<false> { using Type = __m512; };
template <> struct RowAccumulator<true> { using Type = __m256; };

template <int TY, int NT, bool Singleton = false>
inline void row_dot(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    // Decode weights once for the group; select the accumulation order at compile time.
    using Acc = typename RowAccumulator<Singleton>::Type;
    Acc accf[NT];
    for (int t = 0; t < NT; ++t) {
        if constexpr (Singleton) accf[t] = _mm256_setzero_ps();
        else accf[t] = _mm512_setzero_ps();
    }
    const __m512i zero = _mm512_setzero_si512();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * Fmt<TY>::bytes;
        // E-2: the weight rows stream from DRAM (4 KB pages when large pages are refused): ask for the bytes a few
        // blocks ahead (the next row's start included) before the decode needs them
        if (prefetch_ahead > 0) {
            const uint8_t* ahead = blk + (size_t) prefetch_ahead;
            _mm_prefetch((const char*) ahead, _MM_HINT_T0);
            _mm_prefetch((const char*) ahead + 64, _MM_HINT_T0);
        }
        __m512i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm512_setzero_si512();
        for (int j = 0; j < 4; ++j) {
            __m512i g, sc;
            __mmask64 m;
            Fmt<TY>::decode(blk, j, g, m, sc);
            for (int t = 0; t < NT; ++t) {
                const __m512i yv = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j));
                const __m512i ys = _mm512_mask_sub_epi8(yv, m, zero, yv);
                acci[t] = _mm512_dpwssd_epi32(acci[t], _mm512_maddubs_epi16(g, ys), sc);
            }
        }
        if constexpr (Singleton) {
            const float dx = h2f(u16(blk));
            for (int t = 0; t < NT; ++t) {
                // Preserve ggml AVX2's eight integer lanes BEFORE floating conversion.
                const __m256i folded = _mm256_add_epi32(_mm512_castsi512_si256(acci[t]),
                                                       _mm512_extracti64x4_epi64(acci[t], 1));
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d),
                                         _mm256_cvtepi32_ps(folded), accf[t]);
            }
        } else {
            const float dx = h2f(u16(blk)) * Fmt<TY>::K;
            for (int t = 0; t < NT; ++t)
                accf[t] = _mm512_fmadd_ps(_mm512_set1_ps(dx * y[t][i].d), _mm512_cvtepi32_ps(acci[t]), accf[t]);
        }
    }
    if constexpr (Singleton) {
        for (int t = 0; t < NT; ++t) {
            __m128 sum = _mm256_extractf128_ps(accf[t], 1);
            sum = _mm_add_ps(sum, _mm256_castps256_ps128(accf[t]));
            sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
            sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
            res[t] = Fmt<TY>::K * _mm_cvtss_f32(sum);
        }
    } else {
        for (int t = 0; t < NT; ++t) res[t] = _mm512_reduce_add_ps(accf[t]);
    }
}

template <int TY, int NT, bool Singleton = false>
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, float* const* ff,
             int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    const int nb = n / QK_K;
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<TY, NT, Singleton>(blob + (size_t) r * gu_row, nb, y, g);
        row_dot<TY, NT, Singleton>(blob + up_off + (size_t) r * gu_row, nb, y, u);
        for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}

template <int TY, int NT>
void dot_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<TY, NT>(w + (size_t) r * row_bytes, n / QK_K, y, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

template <int TY, bool Singleton = false>
void gu_rows_nt(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                float* const* ff, int r0, int r1) {
    switch (nt) {
        case 1: gu_rows<TY, 1, Singleton>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 2: gu_rows<TY, 2, Singleton>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 3: gu_rows<TY, 3, Singleton>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 4: gu_rows<TY, 4, Singleton>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 5: gu_rows<TY, 5, Singleton>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 6: gu_rows<TY, 6, Singleton>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 7: gu_rows<TY, 7, Singleton>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: gu_rows<TY, 8, Singleton>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    }
}

template <int TY>
void dot_rows_nt(int nt, const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0,
                 int r1) {
    switch (nt) {
        case 1: dot_rows<TY, 1>(w, row_bytes, n, act, out, r0, r1); break;
        case 2: dot_rows<TY, 2>(w, row_bytes, n, act, out, r0, r1); break;
        case 3: dot_rows<TY, 3>(w, row_bytes, n, act, out, r0, r1); break;
        case 4: dot_rows<TY, 4>(w, row_bytes, n, act, out, r0, r1); break;
        default: for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = nt - t0 < 4 ? nt - t0 : 4;
            dot_rows_nt<TY>(k, w, row_bytes, n, act + t0, out + t0, r0, r1);
        }
    }
}


// The sign masks of IQ2_XXS / IQ2_XS / IQ3_XXS come from a 7-bit index whose 8th bit is the parity of the other seven (ksigns_iq2xs[i] =
// i | parity(i) << 7).  The table needs a load and a few shifts and ors a group of eight values; popcount does it for all eight groups
// of 64 values in one vector: measured on Zen 5, one-token IQ2_XXS rows 33 -> 45-50 GB/s (16 threads, DRAM-sized), cache-resident 16.8 -> 13.4 ns a block.
// (AVX512_VPOPCNTDQ + VL: Zen 4, Zen 5, Ice Lake and later; iq512x_supported() asks for it.)
inline __mmask64 sign_mask_from_idx(__m256i si) {
    const __m256i par = _mm256_slli_epi32(_mm256_and_si256(_mm256_popcnt_epi32(si), _mm256_set1_epi32(1)), 7);
    return _cvtu64_mask64((uint64_t) _mm_cvtsi128_si64(_mm256_cvtepi32_epi8(_mm256_or_si256(si, par))));
}
inline __m256i sign_idx_xxs(uint32_t lo4, uint32_t hi4) {   // four 7-bit fields of each of two words -> eight dwords
    const __m256i w = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_set1_epi32((int) lo4)), _mm_set1_epi32((int) hi4), 1);
    return _mm256_and_si256(_mm256_srlv_epi32(w, _mm256_setr_epi32(0, 7, 14, 21, 0, 7, 14, 21)), _mm256_set1_epi32(127));
}

template <int TY> struct FmtX : Fmt<TY> {};
template <> struct FmtX<16> : Fmt<16> {
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        const uint32_t a0 = u32(q), a1 = u32(q + 4), b0 = u32(q + 8), b1 = u32(q + 12);
        g = gather ? gather64(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long) ((uint64_t) a0 | ((uint64_t) b0 << 32)))), iq2xxs_grid)
                   : _mm512_set_epi64((long long) iq2xxs_grid[b0 >> 24], (long long) iq2xxs_grid[(b0 >> 16) & 255],
                             (long long) iq2xxs_grid[(b0 >> 8) & 255], (long long) iq2xxs_grid[b0 & 255],
                             (long long) iq2xxs_grid[a0 >> 24], (long long) iq2xxs_grid[(a0 >> 16) & 255],
                             (long long) iq2xxs_grid[(a0 >> 8) & 255], (long long) iq2xxs_grid[a0 & 255]);
        m = sign_mask_from_idx(sign_idx_xxs(a1, b1));
        const int sa = 2 * (int) (a1 >> 28) + 1, sb = 2 * (int) (b1 >> 28) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
};
template <> struct FmtX<17> : Fmt<17> {
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        uint16_t v[8];
        std::memcpy(v, q, 16);
        g = gather ? gather64(_mm256_and_si256(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*) q)), _mm256_set1_epi32(511)), iq2xs_grid)
                   : _mm512_set_epi64((long long) iq2xs_grid[v[7] & 511], (long long) iq2xs_grid[v[6] & 511],
                             (long long) iq2xs_grid[v[5] & 511], (long long) iq2xs_grid[v[4] & 511],
                             (long long) iq2xs_grid[v[3] & 511], (long long) iq2xs_grid[v[2] & 511],
                             (long long) iq2xs_grid[v[1] & 511], (long long) iq2xs_grid[v[0] & 511]);
        m = sign_mask_from_idx(_mm256_srli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*) q)), 9));
        const uint8_t s0 = b[66 + 2 * j], s1 = b[66 + 2 * j + 1];
        sc = scales4(2 * (s0 & 15) + 1, 2 * (s0 >> 4) + 1, 2 * (s1 & 15) + 1, 2 * (s1 >> 4) + 1);
    }
};
template <> struct FmtX<18> : Fmt<18> {
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        g = gather ? _mm512_i32gather_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) q)), iq3xxs_grid, 4)
                   : _mm512_set_epi32((int) iq3xxs_grid[q[15]], (int) iq3xxs_grid[q[14]], (int) iq3xxs_grid[q[13]], (int) iq3xxs_grid[q[12]],
                             (int) iq3xxs_grid[q[11]], (int) iq3xxs_grid[q[10]], (int) iq3xxs_grid[q[9]], (int) iq3xxs_grid[q[8]],
                             (int) iq3xxs_grid[q[7]], (int) iq3xxs_grid[q[6]], (int) iq3xxs_grid[q[5]], (int) iq3xxs_grid[q[4]],
                             (int) iq3xxs_grid[q[3]], (int) iq3xxs_grid[q[2]], (int) iq3xxs_grid[q[1]], (int) iq3xxs_grid[q[0]]);
        const uint32_t a = u32(b + 2 + 64 + 8 * j), c = u32(b + 2 + 64 + 8 * j + 4);
        m = sign_mask_from_idx(sign_idx_xxs(a, c));
        const int sa = 2 * (int) (a >> 28) + 1, sb = 2 * (int) (c >> 28) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
};

// ---- the DeepSeek CPU lane's exact rows (iq512x_rows): the same 64-value decode, but summed as ggml's AVX2 dots sum, so a
// token's row is ggml-cpu's vec_dot bit for bit at any number of tokens: the integer sums of a block fold to ggml's eight lanes
// (low half = the even 32-value chunk, high half = the odd one), the block scale is f16(d) * y.d, eight float lanes accumulate
// across blocks, and K * ggml's hsum comes last.  vpdpwssd is the madd and the add in one.
inline __m256i fold256(__m512i a) { return _mm256_add_epi32(_mm512_castsi512_si256(a), _mm512_extracti64x4_epi64(a, 1)); }

inline float hsum8_ggml(__m256 x) {
    __m128 s = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

template <int TY, int NT>
inline void row_dot_x(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    const __m512i zero = _mm512_setzero_si512();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * FmtX<TY>::bytes;
        if (prefetch_ahead > 0) {
            const uint8_t* ahead = blk + (size_t) prefetch_ahead;
            _mm_prefetch((const char*) ahead, _MM_HINT_T0);
            _mm_prefetch((const char*) ahead + 64, _MM_HINT_T0);
        }
        __m512i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm512_setzero_si512();
        for (int j = 0; j < 4; ++j) {
            __m512i g, sc;
            __mmask64 m;
            FmtX<TY>::decode(blk, j, g, m, sc);
            for (int t = 0; t < NT; ++t) {
                const __m512i yv = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j));
                const __m512i ys = _mm512_mask_sub_epi8(yv, m, zero, yv);
                acci[t] = _mm512_dpwssd_epi32(acci[t], _mm512_maddubs_epi16(g, ys), sc);
            }
        }
        const float dx = h2f(u16(blk));
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(fold256(acci[t])), accf[t]);
    }
    for (int t = 0; t < NT; ++t) res[t] = FmtX<TY>::K * hsum8_ggml(accf[t]);
}

template <int TY, int NT>
void dot_rows_x(const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot_x<TY, NT>(w + (size_t) r * row_bytes, n / QK_K, y, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

// many tokens an expert (the prompt): signed int8 weights decoded once per block, activations as u8 (q8 + 128, once a call),
// sum(w * q) = sum(w * (q + 128)) - 128 * sum(w) with the second sum once a weight block; tokens in tiles of 16 and rows in
// groups of 8 so the tile's slice of a block and the group's accumulators stay in L1 (the AVX2 form's layout)
template <int TY>
void dot_rows_wide(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int T, float* const* out, int r0, int r1) {
    constexpr int kTile = 16, kRows = 8;
    const int nb = n / QK_K;
    static thread_local std::vector<uint8_t> ubuf;
    static thread_local std::vector<float> accb;
    ubuf.resize((size_t) (T < kTile ? T : kTile) * n);
    accb.resize((size_t) kRows * kTile * 8);
    const __m512i zero = _mm512_setzero_si512(), one8 = _mm512_set1_epi8(1), flip = _mm512_set1_epi8((char) 0x80);
    for (int t0 = 0; t0 < T; t0 += kTile) {
        const int tt = T - t0 < kTile ? T - t0 : kTile;
        const block_q8_K* y[kTile];
        for (int t = 0; t < tt; ++t) {
            y[t] = (const block_q8_K*) act[t0 + t];
            uint8_t* u = ubuf.data() + (size_t) t * n;
            for (int i = 0; i < nb; ++i)
                for (int j = 0; j < 4; ++j)
                    _mm512_storeu_si512((void*) (u + (size_t) i * 256 + 64 * j),
                                        _mm512_xor_si512(_mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j)), flip));
        }
        for (int rg = r0; rg < r1; rg += kRows) {
            const int rr = r1 - rg < kRows ? r1 - rg : kRows;
            for (int k = 0; k < rr * tt; ++k) _mm256_storeu_ps(&accb[(size_t) k * 8], _mm256_setzero_ps());
            for (int i = 0; i < nb; ++i)
                for (int q = 0; q < rr; ++q) {
                    const uint8_t* blk = w + (size_t) (rg + q) * row_bytes + (size_t) i * FmtX<TY>::bytes;
                    if (prefetch_ahead > 0) {
                        _mm_prefetch((const char*) blk + prefetch_ahead, _MM_HINT_T0);
                        _mm_prefetch((const char*) blk + prefetch_ahead + 64, _MM_HINT_T0);
                    }
                    __m512i w8[4], sc[4];
                    __m512i corr = zero;
                    for (int j = 0; j < 4; ++j) {
                        __m512i g, s;
                        __mmask64 m;
                        FmtX<TY>::decode(blk, j, g, m, s);
                        w8[j] = _mm512_mask_sub_epi8(g, m, zero, g);
                        sc[j] = s;
                        corr = _mm512_dpwssd_epi32(corr, _mm512_maddubs_epi16(one8, w8[j]), s);
                    }
                    const __m256i corr128 = _mm256_slli_epi32(fold256(corr), 7);
                    const float dx = h2f(u16(blk));
                    float* accq = &accb[(size_t) q * tt * 8];
                    for (int t = 0; t < tt; ++t) {
                        const uint8_t* u = ubuf.data() + (size_t) t * n + (size_t) i * 256;
                        __m512i a = zero;
                        for (int j = 0; j < 4; ++j)
                            a = _mm512_dpwssd_epi32(a, _mm512_maddubs_epi16(_mm512_loadu_si512((const void*) (u + 64 * j)), w8[j]), sc[j]);
                        const __m256i s = _mm256_sub_epi32(fold256(a), corr128);
                        float* ap = accq + (size_t) t * 8;
                        _mm256_storeu_ps(ap, _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(s), _mm256_loadu_ps(ap)));
                    }
                }
            for (int q = 0; q < rr; ++q)
                for (int t = 0; t < tt; ++t)
                    out[t0 + t][rg + q] = FmtX<TY>::K * hsum8_ggml(_mm256_loadu_ps(&accb[((size_t) q * tt + t) * 8]));
        }
    }
}

template <int TY>
void dot_rows_x_nt(int nt, const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1) {
    switch (nt) {
        case 1: dot_rows_x<TY, 1>(w, row_bytes, n, act, out, r0, r1); break;
        case 2: dot_rows_x<TY, 2>(w, row_bytes, n, act, out, r0, r1); break;
        case 3: dot_rows_x<TY, 3>(w, row_bytes, n, act, out, r0, r1); break;
        case 4: dot_rows_x<TY, 4>(w, row_bytes, n, act, out, r0, r1); break;
        case 5: dot_rows_x<TY, 5>(w, row_bytes, n, act, out, r0, r1); break;
        case 6: dot_rows_x<TY, 6>(w, row_bytes, n, act, out, r0, r1); break;
        default: dot_rows_wide<TY>(w, row_bytes, n, act, nt, out, r0, r1); break;
    }
}


// ---- IQ1_M (29), AVX-512, 1..6 tokens: ggml's dot lane for lane.  The grid indices come out of two pdep (as ggml's BMI2 path), the
// +-1 delta signs out of one pext into an 8-bit mask a 64-value step; the signed grid bytes are split into |g| and a negative mask;
// the delta dot is the pair sums of the activation (made once a call) times +-scale.  d is f16 in the top nibbles of the scales.
inline void iq1m_pre(const block_q8_K* y, int nb, int16_t* p) {
    const __m256i one8 = _mm256_set1_epi8(1);
    for (int i = 0; i < nb; ++i)
        for (int c = 0; c < 8; ++c)
            _mm256_storeu_si256((__m256i*) (p + ((size_t) i * 8 + c) * 16),
                                _mm256_maddubs_epi16(one8, _mm256_loadu_si256((const __m256i*) (y[i].qs + 32 * c))));
}

template <int NT>
void rows_m(const uint8_t* w, size_t rb, int n, const block_q8_K* const* y, const int16_t* const* P, float* const* out, int r0, int r1) {
    const int nb = n / QK_K;
    const __m256i mask = _mm256_set1_epi16(0x7), mone = _mm256_set1_epi16(1), mtwo8 = _mm256_set1_epi8(2);
    const __m256i scales_shift = _mm256_set_epi64x(9, 3, 6, 0);
    const __m512i zero = _mm512_setzero_si512();
    // the 32 scale words a 64-weight step (8 each of chunks 4 it .. 4 it + 3) out of the 16 decoded scales (their order: 4 e + j for
    // shift e, word j): one vpermw a step instead of two vpshufb + adds + an insert
    __m512i sidx[4];
    for (int it = 0; it < 4; ++it)
        sidx[it] = _mm512_set_epi16(12 + it, 12 + it, 12 + it, 12 + it, 12 + it, 12 + it, 12 + it, 12 + it, 4 + it, 4 + it, 4 + it, 4 + it, 4 + it, 4 + it,
                                    4 + it, 4 + it, 8 + it, 8 + it, 8 + it, 8 + it, 8 + it, 8 + it, 8 + it, 8 + it, it, it, it, it, it, it, it, it);
    for (int r = r0; r < r1; ++r) {
        __m256 a1[NT], a2[NT];
        for (int t = 0; t < NT; ++t) a1[t] = a2[t] = _mm256_setzero_ps();
        const uint8_t* row = w + (size_t) r * rb;
        for (int i = 0; i < nb; ++i) {
            const uint8_t* blk = row + (size_t) i * 56;
            const uint8_t* qs = blk;
            const uint8_t* qh = blk + 32;
            const uint8_t* scb = blk + 48;
            if (prefetch_ahead > 0) {
                _mm_prefetch((const char*) blk + prefetch_ahead, _MM_HINT_T0);
                _mm_prefetch((const char*) blk + prefetch_ahead + 64, _MM_HINT_T0);
            }
            const uint16_t sc0 = u16(scb), sc1 = u16(scb + 2), sc2 = u16(scb + 4), sc3 = u16(scb + 6);
            const uint16_t d16 = (uint16_t) ((sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000));
            __m256i scales = _mm256_set1_epi64x((long long) u64(scb));
            scales = _mm256_srlv_epi64(scales, scales_shift);
            scales = _mm256_add_epi16(_mm256_slli_epi16(_mm256_and_si256(scales, mask), 1), mone);
            const __m512i sctab = _mm512_castsi256_si512(scales);
            __m512i s1[NT], s2[NT];
            for (int t = 0; t < NT; ++t) s1[t] = s2[t] = zero;
            for (int it = 0; it < 4; ++it) {
                const uint32_t qh32 = u32(qh + 4 * it);
                const uint64_t p1 = _pdep_u64(u32(qs + 8 * it), 0x00ff00ff00ff00ffULL) | _pdep_u64(qh32 & 0x7777, 0x0f000f000f000f00ULL);
                const uint64_t p2 = _pdep_u64(u32(qs + 8 * it + 4), 0x00ff00ff00ff00ffULL) | _pdep_u64((qh32 >> 16) & 0x7777, 0x0f000f000f000f00ULL);
                uint16_t ix[8];
                std::memcpy(ix, &p1, 8);
                std::memcpy(ix + 4, &p2, 8);
                const __m512i g = _mm512_set_epi64((long long) iq1s_grid[ix[7]], (long long) iq1s_grid[ix[6]], (long long) iq1s_grid[ix[5]],
                                                   (long long) iq1s_grid[ix[4]], (long long) iq1s_grid[ix[3]], (long long) iq1s_grid[ix[2]],
                                                   (long long) iq1s_grid[ix[1]], (long long) iq1s_grid[ix[0]]);
                const __m512i sc = _mm512_permutexvar_epi16(sidx[it], sctab);
                // the delta sign of each 8-weight group (bit 3 of its qh nibble) negates that group's 4 scale words
                const __mmask32 dneg = (__mmask32) (_pdep_u32(_pext_u32(qh32, 0x88888888u), 0x11111111u) * 15u);
                const __m512i ssc = _mm512_mask_sub_epi16(sc, dneg, zero, sc);
                const __m512i ag = _mm512_abs_epi8(g);
                const __mmask64 gneg = _mm512_movepi8_mask(g);
                for (int t = 0; t < NT; ++t) {
                    const __m512i q8 = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * it));
                    const __m512i pp = _mm512_loadu_si512((const void*) (P[t] + ((size_t) i * 8 + 2 * it) * 16));
                    const __m512i ys = _mm512_mask_sub_epi8(q8, gneg, zero, q8);
                    s1[t] = _mm512_dpwssd_epi32(s1[t], _mm512_maddubs_epi16(ag, ys), sc);
                    s2[t] = _mm512_dpwssd_epi32(s2[t], pp, ssc);
                }
            }
            const float dx = h2f(d16);
            for (int t = 0; t < NT; ++t) {
                const __m256 d = _mm256_set1_ps(y[t][i].d * dx);
                a1[t] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(fold256(s1[t])), a1[t]);
                a2[t] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(fold256(s2[t])), a2[t]);
            }
        }
        for (int t = 0; t < NT; ++t) out[t][r] = hsum8_ggml(a1[t]) + 0.125f * hsum8_ggml(a2[t]);
    }
}

void rows_m_nt(int nt, const uint8_t* w, size_t rb, int n, const void* const* act, float* const* out, int r0, int r1) {
    const int nb = n / QK_K;
    static thread_local std::vector<int16_t> pre;
    for (int t0 = 0; t0 < nt; t0 += 6) {
        const int k = nt - t0 < 6 ? nt - t0 : 6;
        const block_q8_K* y[6];
        const int16_t* P[6];
        float* o[6];
        pre.resize((size_t) k * n);
        for (int t = 0; t < k; ++t) {
            y[t] = (const block_q8_K*) act[t0 + t];
            iq1m_pre(y[t], nb, pre.data() + (size_t) t * n);
            P[t] = pre.data() + (size_t) t * n;
            o[t] = out[t0 + t];
        }
        switch (k) {
            case 1: rows_m<1>(w, rb, n, y, P, o, r0, r1); break;
            case 2: rows_m<2>(w, rb, n, y, P, o, r0, r1); break;
            case 3: rows_m<3>(w, rb, n, y, P, o, r0, r1); break;
            case 4: rows_m<4>(w, rb, n, y, P, o, r0, r1); break;
            case 5: rows_m<5>(w, rb, n, y, P, o, r0, r1); break;
            default: rows_m<6>(w, rb, n, y, P, o, r0, r1); break;
        }
    }
}

}  // namespace

void iq512_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1) {
    switch (type) {
        case 16: gu_rows_nt<16>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 17: gu_rows_nt<17>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 18: gu_rows_nt<18>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 21: gu_rows_nt<21>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 22: gu_rows_nt<22>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: break;
    }
}

// The linked ggml AVX2 singleton order, for the formats validated against its production dot.
void iq512_reference_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                             const void* const* act, int nt, float* const* ff, int r0, int r1) {
    switch (type) {
        case 18: gu_rows_nt<18, true>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 21: gu_rows_nt<21, true>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 22: gu_rows_nt<22, true>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 23: gu_rows_nt<23, true>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
    }
}

void iq512_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1) {
    switch (type) {
        case 16: dot_rows_nt<16>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 17: dot_rows_nt<17>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 18: dot_rows_nt<18>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 21: dot_rows_nt<21>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 22: dot_rows_nt<22>(nt, w, row_bytes, n, act, out, r0, r1); break;
        default: break;
    }
}

static bool has_vpopcnt() {
    static const bool ok = [] {
#if defined(_MSC_VER)
        int r[4];
        __cpuidex(r, 7, 0);
        return ((r[2] >> 14) & 1) != 0;   // AVX512_VPOPCNTDQ
#else
        unsigned a, b, c, d;
        __cpuid_count(7, 0, a, b, c, d);
        return ((c >> 14) & 1u) != 0;
#endif
    }();
    return ok;
}

bool iq512_gather_on() noexcept { return gather; }

bool iq512x_supported(int type) noexcept {
    return has_vpopcnt() && (type == 16 || type == 17 || type == 18 || type == 21 || type == 22 || type == 29);
}

void iq512x_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                 int r0, int r1) {
    switch (type) {
        case 16: dot_rows_x_nt<16>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 17: dot_rows_x_nt<17>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 18: dot_rows_x_nt<18>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 21: dot_rows_x_nt<21>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 22: dot_rows_x_nt<22>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 23: dot_rows_x_nt<23>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 29: rows_m_nt(nt, w, row_bytes, n, act, out, r0, r1); break;
        default: break;
    }
}

// ---- gate + up in ONE pass over the activations (the idea of PR 5243's row_gu_dot, bsorensen110): the weights of the two rows are decoded side by
// side and each token's 64 activation bytes are loaded once and applied to both.  The sums are ggml's AVX2 dot's (the integer lanes folded to eight,
// f16 scale times y.d, eight float lanes, its hsum, K last), so a token's gate and up are ggml-cpu's vec_dot bit for bit at any token count: the
// same bits as the one-token path of native_expert.cpp (which runs ggml's dot), and the 2+ token rows differ from the 16-lane i-quant rows by
// rounding only.  The sign masks of IQ2_XXS / IQ2_XS / IQ3_XXS come from VPOPCNTDQ (FmtX).
namespace {

template <int TY, int NT>
void gu_fused(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, float* const* ff, int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    const int nb = n / QK_K;
    const __m512i zero = _mm512_setzero_si512();
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gr = blob + (size_t) r * gu_row;
        const uint8_t* ur = blob + up_off + (size_t) r * gu_row;
        __m256 ag[NT], au[NT];
        for (int t = 0; t < NT; ++t) ag[t] = au[t] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* gb = gr + (size_t) i * FmtX<TY>::bytes;
            const uint8_t* ub = ur + (size_t) i * FmtX<TY>::bytes;
            if (prefetch_ahead > 0) {
                _mm_prefetch((const char*) gb + prefetch_ahead, _MM_HINT_T0);
                _mm_prefetch((const char*) gb + prefetch_ahead + 64, _MM_HINT_T0);
                _mm_prefetch((const char*) ub + prefetch_ahead, _MM_HINT_T0);
                _mm_prefetch((const char*) ub + prefetch_ahead + 64, _MM_HINT_T0);
            }
            __m512i ig[NT], iu[NT];
            for (int t = 0; t < NT; ++t) ig[t] = iu[t] = zero;
            for (int j = 0; j < 4; ++j) {
                __m512i gg, gs, ug, us;
                __mmask64 gm, um;
                FmtX<TY>::decode(gb, j, gg, gm, gs);
                FmtX<TY>::decode(ub, j, ug, um, us);
                for (int t = 0; t < NT; ++t) {
                    const __m512i yv = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j));
                    ig[t] = _mm512_dpwssd_epi32(ig[t], _mm512_maddubs_epi16(gg, _mm512_mask_sub_epi8(yv, gm, zero, yv)), gs);
                    iu[t] = _mm512_dpwssd_epi32(iu[t], _mm512_maddubs_epi16(ug, _mm512_mask_sub_epi8(yv, um, zero, yv)), us);
                }
            }
            const float dg = h2f(u16(gb)), du = h2f(u16(ub));
            for (int t = 0; t < NT; ++t) {
                const float yd = y[t][i].d;
                ag[t] = _mm256_fmadd_ps(_mm256_set1_ps(dg * yd), _mm256_cvtepi32_ps(fold256(ig[t])), ag[t]);
                au[t] = _mm256_fmadd_ps(_mm256_set1_ps(du * yd), _mm256_cvtepi32_ps(fold256(iu[t])), au[t]);
            }
        }
        for (int t = 0; t < NT; ++t) {
            const float g = FmtX<TY>::K * hsum8_ggml(ag[t]), u = FmtX<TY>::K * hsum8_ggml(au[t]);
            ff[t][r] = (g / (1.f + std::exp(-g))) * u;
        }
    }
}

template <int TY>
void gu_fused_nt(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, float* const* ff, int r0, int r1) {
    switch (nt) {
        case 1: gu_fused<TY, 1>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 2: gu_fused<TY, 2>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 3: gu_fused<TY, 3>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 4: gu_fused<TY, 4>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default:
            for (int t0 = 0; t0 < nt; t0 += 4) {
                const int k = nt - t0 < 4 ? nt - t0 : 4;
                gu_fused_nt<TY>(k, blob, gu_row, up_off, n, act + t0, ff + t0, r0, r1);
            }
    }
}

}  // namespace

bool iq512f_supported(int type) noexcept { return has_vpopcnt() && (type == 16 || type == 17 || type == 18 || type == 21 || type == 22); }

void iq512_fused_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                         float* const* ff, int r0, int r1) {
    switch (type) {
        case 16: gu_fused_nt<16>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 17: gu_fused_nt<17>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 18: gu_fused_nt<18>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 21: gu_fused_nt<21>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 22: gu_fused_nt<22>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: break;
    }
}

// ---- Q5_0 (6) down rows in 512-bit lanes: two 32-value blocks a vector, the weights of a block pair decoded once for
// every token.  The arithmetic is ggml-cpu's AVX2 dot, order for order: per block the 8 integer lanes (values 4l .. 4l+3 of the
// block, one vpdpbusd lane each), converted, scaled by f16(dx) * f16(dy) and fused into ONE eight-lane float accumulator in block order,
// hsum last: a token's row is ggml-cpu's dot bit for bit.  An odd block count ends on a 256-bit block.  (An IQ4_NL version of the same
// shape measured 5-11% SLOWER than iq4nl256_down_rows on Zen 4 at 2-4 tokens and was dropped.)
namespace {

inline __m512i pair_q8_0(const block_q8_0* a, const block_q8_0* b) {
    return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*) a->qs)),
                              _mm256_loadu_si256((const __m256i*) b->qs), 1);
}

// 32 signed weights of a Q5_0 block: the nibble (low plane, then high plane) minus 16 where the fifth bit is clear
inline __m256i q50_weights(const block_q5_0* x) {
    const __m128i bits = _mm_loadu_si128((const __m128i*) x->qs);
    const __m128i m4 = _mm_set1_epi8(0x0f);
    const __m256i nib = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_and_si128(bits, m4)), _mm_and_si128(_mm_srli_epi16(bits, 4), m4), 1);
    uint32_t qh;
    std::memcpy(&qh, x->qh, 4);
    return _mm256_mask_sub_epi8(nib, (__mmask32) ~qh, nib, _mm256_set1_epi8(16));
}

template <int NT>
void qd_rows(const uint8_t* w, size_t row_bytes, int n, const block_q8_0* const* y, float* const* out, int r0, int r1) {
    constexpr size_t kBlk = sizeof(block_q5_0);
    const int nb = n / 32;
    const int pf = prefetch_ahead;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        __m256 acc[NT];
        for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
        int ib = 0;
        for (; ib + 1 < nb; ib += 2) {
            const uint8_t* b0 = row + (size_t) ib * kBlk;
            if (pf > 0) { _mm_prefetch((const char*) b0 + pf, _MM_HINT_T0); _mm_prefetch((const char*) b0 + pf + 64, _MM_HINT_T0); }
            const __m512i wq = _mm512_inserti64x4(_mm512_castsi256_si512(q50_weights((const block_q5_0*) b0)), q50_weights((const block_q5_0*) (b0 + kBlk)), 1);
            const float d0 = h2f(((const block_q5_0*) b0)->d);
            const float d1 = h2f(((const block_q5_0*) (b0 + kBlk))->d);
            const __m512i aw = _mm512_abs_epi8(wq);
            const __mmask64 neg = _mm512_movepi8_mask(wq);
            for (int t = 0; t < NT; ++t) {
                const __m512i yv = pair_q8_0(&y[t][ib], &y[t][ib + 1]);
                const __m512i p = _mm512_dpbusd_epi32(_mm512_setzero_si512(), aw, _mm512_mask_sub_epi8(yv, neg, _mm512_setzero_si512(), yv));
                const __m256 lo = _mm256_cvtepi32_ps(_mm512_castsi512_si256(p));
                const __m256 hi = _mm256_cvtepi32_ps(_mm512_extracti64x4_epi64(p, 1));
                acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d0 * h2f(y[t][ib].d)), lo, acc[t]);
                acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d1 * h2f(y[t][ib + 1].d)), hi, acc[t]);
            }
        }
        if (ib < nb) {
            const uint8_t* b0 = row + (size_t) ib * kBlk;
            const __m256i wq = q50_weights((const block_q5_0*) b0);
            const float d0 = h2f(((const block_q5_0*) b0)->d);
            for (int t = 0; t < NT; ++t) {
                const __m256i yv = _mm256_loadu_si256((const __m256i*) y[t][ib].qs);
                const __m256i p = _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_abs_epi8(wq), _mm256_sign_epi8(yv, wq));
                acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d0 * h2f(y[t][ib].d)), _mm256_cvtepi32_ps(p), acc[t]);
            }
        }
        for (int t = 0; t < NT; ++t) out[t][r] = hsum8_ggml(acc[t]);
    }
}

void qd_rows_nt(int nt, const uint8_t* w, size_t row_bytes, int n, const void* const* hq, float* const* out, int r0, int r1) {
    const block_q8_0* y[8];
    for (int t = 0; t < nt; ++t) y[t] = (const block_q8_0*) hq[t];
    switch (nt) {
        case 1: qd_rows<1>(w, row_bytes, n, y, out, r0, r1); break;
        case 2: qd_rows<2>(w, row_bytes, n, y, out, r0, r1); break;
        case 3: qd_rows<3>(w, row_bytes, n, y, out, r0, r1); break;
        case 4: qd_rows<4>(w, row_bytes, n, y, out, r0, r1); break;
        default:
            for (int t0 = 0; t0 < nt; t0 += 4) {
                const int k = nt - t0 < 4 ? nt - t0 : 4;
                qd_rows_nt(k, w, row_bytes, n, hq + t0, out + t0, r0, r1);
            }
    }
}

}  // namespace

void q50_512_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt, float* const* out, int r0, int r1) {
    qd_rows_nt(nt, w, row_bytes, n, hq, out, r0, r1);
}


#if defined(_MSC_VER)
#define STRATA_FORCEINLINE __forceinline
#else
#define STRATA_FORCEINLINE __attribute__((always_inline)) inline
#endif

namespace {
STRATA_FORCEINLINE int prefetch_distance() noexcept {
    static const int v = [] {
        const char* e = std::getenv("STRATA_IQ_PREFETCH");
        return e ? std::atoi(e) : 2048;
    }();
    return v;
}
template <int NT>
STRATA_FORCEINLINE void iq4nl512_row(const block_iq4_nl* x, int nb, const block_q8_0* const* y, float* res) {
    const __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*) kvalues_iq4nl));
    const __m256i m4 = _mm256_set1_epi8(0x0f);
    const __m512i zero = _mm512_setzero_si512();
    const int pref = prefetch_distance();
    __m512 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm512_setzero_ps();
    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        if (pref > 0) _mm_prefetch((const char*) (x + ib) + pref, _MM_HINT_T0);
        const __m128i b0 = _mm_loadu_si128((const __m128i*) x[ib + 0].qs);
        const __m128i b1 = _mm_loadu_si128((const __m128i*) x[ib + 1].qs);
        const __m256i n0 = _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(b0), _mm_srli_epi16(b0, 4), 1), m4);
        const __m256i n1 = _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(b1), _mm_srli_epi16(b1, 4), 1), m4);
        const __m512i q4 = _mm512_shuffle_epi8(lut, _mm512_inserti64x4(_mm512_castsi256_si512(n0), n1, 1));
        const __m512i aq = _mm512_abs_epi8(q4);
        const __mmask64 neg = _mm512_movepi8_mask(q4);
        const float dx0 = h2f(x[ib + 0].d), dx1 = h2f(x[ib + 1].d);
        for (int t = 0; t < NT; ++t) {
            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*) y[t][ib + 0].qs);
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) y[t][ib + 1].qs);
            const __m512i q8 = _mm512_inserti64x4(_mm512_castsi256_si512(q8_0), q8_1, 1);
            const __m512i ys = _mm512_mask_sub_epi8(q8, neg, zero, q8);
            const __m512i p32 = _mm512_dpbusd_epi32(zero, aq, ys);
            const float d0 = h2f(y[t][ib + 0].d) * dx0;
            const float d1 = h2f(y[t][ib + 1].d) * dx1;
            const __m512 d_pair = _mm512_mask_blend_ps((__mmask16) 0xFF00, _mm512_set1_ps(d0), _mm512_set1_ps(d1));
            accf[t] = _mm512_fmadd_ps(d_pair, _mm512_cvtepi32_ps(p32), accf[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        const __m256 sum256 = _mm256_add_ps(_mm512_castps512_ps256(accf[t]), _mm512_extractf32x8_ps(accf[t], 1));
        float sumf = hsum8_ggml(sum256);
        for (int k = ib; k < nb; ++k) {
            const float d = h2f(y[t][k].d) * h2f(x[k].d);
            int s0 = 0, s1 = 0;
            for (int j = 0; j < QK4_NL / 2; ++j) {
                s0 += y[t][k].qs[j + 0] * kvalues_iq4nl[x[k].qs[j] & 0xf];
                s1 += y[t][k].qs[j + QK4_NL / 2] * kvalues_iq4nl[x[k].qs[j] >> 4];
            }
            sumf += d * (float) (s0 + s1);
        }
        res[t] = sumf;
    }
}

template <int NT>
void iq4nl512_rows_nt(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, float* const* out, int r0, int r1) {
    const block_q8_0* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_0*) hq[t];
    const int nb = n / QK4_NL;
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        iq4nl512_row<NT>((const block_iq4_nl*) (w + (size_t) r * row_bytes), nb, y, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

// ---- Q2_0 (42) down-projection rows in 512-bit AVX-512 VNNI: 1 block_q2_0 (64 weights = 2 block_q8_0) per step,
// bit-exact against ggml_vec_dot_q2_0_q8_0.
STRATA_FORCEINLINE int hsum8_i32(__m256i a) {
    const __m128i s128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(s128, s128);
    const __m128i s64 = _mm_add_epi32(hi64, s128);
    const __m128i hi32 = _mm_shuffle_epi32(s64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(s64, hi32));
}

template <int NT>
STRATA_FORCEINLINE void q2_512_row(const uint8_t* x, int nb, const block_q8_0* const* y, float* res) {
    const __m128i m3 = _mm_set1_epi8(3);
    const __m512i one = _mm512_set1_epi8(1);
    const __m512i zero = _mm512_setzero_si512();
    float sumf[NT];
    for (int t = 0; t < NT; ++t) sumf[t] = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const uint8_t* blk = x + (size_t) i * 18;
        const float d0 = h2f(u16(blk));
        const __m128i b = _mm_loadu_si128((const __m128i*) (blk + 2));
        const __m128i c0 = _mm_and_si128(b, m3);
        const __m128i c1 = _mm_and_si128(_mm_srli_epi16(b, 2), m3);
        const __m128i c2 = _mm_and_si128(_mm_srli_epi16(b, 4), m3);
        const __m128i c3 = _mm_and_si128(_mm_srli_epi16(b, 6), m3);
        const __m128i a0 = _mm_unpacklo_epi8(c0, c1), a1 = _mm_unpacklo_epi8(c2, c3);
        const __m128i b0 = _mm_unpackhi_epi8(c0, c1), b1 = _mm_unpackhi_epi8(c2, c3);
        const __m256i lo = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi16(a0, a1)), _mm_unpackhi_epi16(a0, a1), 1);
        const __m256i hi = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi16(b0, b1)), _mm_unpackhi_epi16(b0, b1), 1);
        const __m512i codes = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
        const __m512i w = _mm512_sub_epi8(codes, one);
        const __m512i ax = _mm512_abs_epi8(w);
        const __mmask64 neg = _mm512_movepi8_mask(w);
        const __mmask64 zmask = _mm512_cmpeq_epi8_mask(w, zero);
        for (int t = 0; t < NT; ++t) {
            const __m256i y0 = _mm256_loadu_si256((const __m256i*) y[t][2 * i + 0].qs);
            const __m256i y1 = _mm256_loadu_si256((const __m256i*) y[t][2 * i + 1].qs);
            const __m512i qy = _mm512_maskz_mov_epi8(~zmask, _mm512_inserti64x4(_mm512_castsi256_si512(y0), y1, 1));
            const __m512i sy = _mm512_mask_sub_epi8(qy, neg, zero, qy);
            const __m512i p32 = _mm512_dpbusd_epi32(zero, ax, sy);
            const int s0 = hsum8_i32(_mm512_castsi512_si256(p32));
            const int s1 = hsum8_i32(_mm512_extracti64x4_epi64(p32, 1));
            float sumi = 0.0f;
            sumi += h2f(y[t][2 * i + 0].d) * (float) s0;
            sumi += h2f(y[t][2 * i + 1].d) * (float) s1;
            sumf[t] += d0 * sumi;
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = sumf[t];
}

template <int NT>
void q2_512_rows_nt(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, float* const* out, int r0, int r1) {
    const block_q8_0* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_0*) hq[t];
    const int nb = n / 64;
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        q2_512_row<NT>(w + (size_t) r * row_bytes, nb, y, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

}  // namespace

void iq4nl512_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt,
                        float* const* out, int r0, int r1) {
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int m = nt - t0 < 4 ? nt - t0 : 4;
        switch (m) {
            case 1: iq4nl512_rows_nt<1>(w, row_bytes, n, hq + t0, out + t0, r0, r1); break;
            case 2: iq4nl512_rows_nt<2>(w, row_bytes, n, hq + t0, out + t0, r0, r1); break;
            case 3: iq4nl512_rows_nt<3>(w, row_bytes, n, hq + t0, out + t0, r0, r1); break;
            default: iq4nl512_rows_nt<4>(w, row_bytes, n, hq + t0, out + t0, r0, r1); break;
        }
    }
}

void q2_512_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt,
                      float* const* out, int r0, int r1) {
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int m = nt - t0 < 4 ? nt - t0 : 4;
        switch (m) {
            case 1: q2_512_rows_nt<1>(w, row_bytes, n, hq + t0, out + t0, r0, r1); break;
            case 2: q2_512_rows_nt<2>(w, row_bytes, n, hq + t0, out + t0, r0, r1); break;
            case 3: q2_512_rows_nt<3>(w, row_bytes, n, hq + t0, out + t0, r0, r1); break;
            default: q2_512_rows_nt<4>(w, row_bytes, n, hq + t0, out + t0, r0, r1); break;
        }
    }
}

namespace {

constexpr int kKQMaxT = 4;

STRATA_FORCEINLINE void unpack_k4_scales(const uint8_t* sc, uint64_t& sc64, uint64_t& min64) {
    constexpr uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    const uint32_t u0 = u32(sc), u1 = u32(sc + 4), u2 = u32(sc + 8);
    const uint32_t sc_lo = u0 & kmask1;
    const uint32_t sc_hi = (u2 & kmask2) | (((u0 >> 6) & kmask3) << 4);
    const uint32_t min_lo = u1 & kmask1;
    const uint32_t min_hi = ((u2 >> 4) & kmask2) | (((u1 >> 6) & kmask3) << 4);
    sc64 = (uint64_t) sc_lo | ((uint64_t) sc_hi << 32);
    min64 = (uint64_t) min_lo | ((uint64_t) min_hi << 32);
}

template <int NT>
STRATA_FORCEINLINE void dot_q4_k_512(const block_q4_K* x, int nb, const block_q8_K* const* y, float* out) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m512i zero = _mm512_setzero_si512();
    const __m512i shuf0 = _mm512_mask_blend_epi64((__mmask8) 0xF0, _mm512_set1_epi16(0x0100), _mm512_set1_epi16(0x0302));
    const __m512i step4 = _mm512_set1_epi8(4);
    const int pref = prefetch_distance();
    __m256 acc[NT];
    __m128 acc_m[NT];
    for (int t = 0; t < NT; ++t) {
        acc[t] = _mm256_setzero_ps();
        acc_m[t] = _mm_setzero_ps();
    }
    for (int i = 0; i < nb; ++i) {
        if (pref > 0) _mm_prefetch((const char*) (x + i) + pref, _MM_HINT_T0);
        const uint8_t* xb = (const uint8_t*) &x[i];
        const float dx = h2f(u16(xb)), dminx = h2f(u16(xb + 2));
        uint64_t sc64, min64;
        unpack_k4_scales(x[i].scales, sc64, min64);
        const __m512i sctab = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_cvtsi64_si128((long long) sc64)));
        const __m128i mins128 = _mm_cvtepu8_epi16(_mm_cvtsi64_si128((long long) min64));
        __m512i sumi512[NT];
        for (int t = 0; t < NT; ++t) sumi512[t] = zero;
        __m512i shuf = shuf0;
        for (int j = 0; j < 4; ++j) {
            const __m256i q4bits = _mm256_loadu_si256((const __m256i*) (x[i].qs + 32 * j));
            const __m256i q4l = _mm256_and_si256(q4bits, m4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
            const __m512i q4 = _mm512_inserti64x4(_mm512_castsi256_si512(q4l), q4h, 1);
            const __m512i sc = _mm512_shuffle_epi8(sctab, shuf);
            shuf = _mm512_add_epi8(shuf, step4);
            for (int t = 0; t < NT; ++t) {
                const __m512i q8v = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j));
                sumi512[t] = _mm512_dpwssd_epi32(sumi512[t], _mm512_maddubs_epi16(q4, q8v), sc);
            }
        }
        for (int t = 0; t < NT; ++t) {
            const float d = y[t][i].d * dx;
            const float dmin = -y[t][i].d * dminx;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y[t][i].bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_castsi256_si128(q8sums), _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(mins128, q8s);
            acc_m[t] = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m[t]);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(fold256(sumi512[t])), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        __m128 m = _mm_add_ps(acc_m[t], _mm_movehl_ps(acc_m[t], acc_m[t]));
        m = _mm_add_ss(m, _mm_movehdup_ps(m));
        out[t] = hsum8_ggml(acc[t]) + _mm_cvtss_f32(m);
    }
}

template <int NT>
STRATA_FORCEINLINE void dot_q5_k_512(const block_q5_K* x, int nb, const block_q8_K* const* y, float* out) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m512i m10 = _mm512_set1_epi8(0x10);
    const __m128i mzero = _mm_setzero_si128();
    const __m512i zero = _mm512_setzero_si512();
    const __m512i shuf0 = _mm512_mask_blend_epi64((__mmask8) 0xF0, _mm512_set1_epi16(0x0100), _mm512_set1_epi16(0x0302));
    const __m512i step4 = _mm512_set1_epi8(4);
    const __m512i bmask0 = _mm512_mask_blend_epi64((__mmask8) 0xF0, _mm512_set1_epi8(1), _mm512_set1_epi8(2));
    const int pref = prefetch_distance();
    __m256 acc[NT];
    float summs[NT];
    for (int t = 0; t < NT; ++t) {
        acc[t] = _mm256_setzero_ps();
        summs[t] = 0.f;
    }
    for (int i = 0; i < nb; ++i) {
        if (pref > 0) _mm_prefetch((const char*) (x + i) + pref, _MM_HINT_T0);
        const uint8_t* xb = (const uint8_t*) &x[i];
        const float dx = h2f(u16(xb)), dminx = h2f(u16(xb + 2));
        uint64_t sc64, min64;
        unpack_k4_scales(x[i].scales, sc64, min64);
        const __m512i sctab = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_cvtsi64_si128((long long) sc64)));
        const __m128i mins128 = _mm_cvtepu8_epi16(_mm_cvtsi64_si128((long long) min64));
        const __m512i hbits = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*) x[i].qh));
        __m512i sumi512[NT];
        for (int t = 0; t < NT; ++t) sumi512[t] = zero;
        __m512i shuf = shuf0, bmask = bmask0;
        for (int j = 0; j < 4; ++j) {
            const __m256i q5bits = _mm256_loadu_si256((const __m256i*) (x[i].qs + 32 * j));
            const __m256i q5l = _mm256_and_si256(q5bits, m4);
            const __m256i q5h = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
            const __m512i q4 = _mm512_inserti64x4(_mm512_castsi256_si512(q5l), q5h, 1);
            const __mmask64 hm = _mm512_test_epi8_mask(hbits, bmask);
            const __m512i q5 = _mm512_mask_add_epi8(q4, hm, q4, m10);
            const __m512i sc = _mm512_shuffle_epi8(sctab, shuf);
            shuf = _mm512_add_epi8(shuf, step4);
            bmask = _mm512_slli_epi16(bmask, 2);
            for (int t = 0; t < NT; ++t) {
                const __m512i q8v = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j));
                sumi512[t] = _mm512_dpwssd_epi32(sumi512[t], _mm512_maddubs_epi16(q5, q8v), sc);
            }
        }
        for (int t = 0; t < NT; ++t) {
            const float d = y[t][i].d * dx;
            const float dmin = -y[t][i].d * dminx;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y[t][i].bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_castsi256_si128(q8sums), _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(mins128, q8s);
            const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
            summs[t] += dmin * _mm_extract_epi32(hsum, 0);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(fold256(sumi512[t])), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) out[t] = hsum8_ggml(acc[t]) + summs[t];
}

template <int NT>
STRATA_FORCEINLINE void dot_q6_k_512(const block_q6_K* x, int nb, const block_q8_K* const* y, float* out) {
    const __m512i m15 = _mm512_set1_epi8(15);
    const __m512i m3 = _mm512_set1_epi8(3);
    const __m512i m12 = _mm512_set1_epi8(12);
    const __m512i m48 = _mm512_set1_epi8(48);
    const __m512i m192 = _mm512_set1_epi8(-64);
    const __m512i zero = _mm512_setzero_si512();
    const __m512i shuf_a = _mm512_set_epi32(
        0x07060706, 0x07060706, 0x07060706, 0x07060706,
        0x05040504, 0x05040504, 0x05040504, 0x05040504,
        0x03020302, 0x03020302, 0x03020302, 0x03020302,
        0x01000100, 0x01000100, 0x01000100, 0x01000100);
    const __m512i shuf_b = _mm512_add_epi8(shuf_a, _mm512_set1_epi8(8));
    const int pref = prefetch_distance();
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        if (pref > 0) _mm_prefetch((const char*) (x + i) + pref, _MM_HINT_T0);
        const float dx = h2f(x[i].d);
        const __m256i scales_16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) x[i].scales));
        __m512i sumi512[NT];
        for (int t = 0; t < NT; ++t) sumi512[t] = zero;
        for (int j = 0; j < 2; ++j) {
            const __m512i sctab = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (x[i].scales + 8 * j))));
            const __m512i ql = _mm512_loadu_si512((const void*) (x[i].ql + 64 * j));
            const __m512i qh = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*) (x[i].qh + 32 * j)));
            const __m512i h0 = _mm512_slli_epi16(_mm512_and_si512(qh, m3), 4);
            const __m512i h1 = _mm512_slli_epi16(_mm512_and_si512(qh, m12), 2);
            const __m512i h2 = _mm512_and_si512(qh, m48);
            const __m512i h3 = _mm512_srli_epi16(_mm512_and_si512(qh, m192), 2);
            const __m512i h01 = _mm512_mask_blend_epi64((__mmask8) 0xF0, h0, h1);
            const __m512i h23 = _mm512_mask_blend_epi64((__mmask8) 0xF0, h2, h3);
            const __m512i q6_a = _mm512_or_si512(_mm512_and_si512(ql, m15), h01);
            const __m512i q6_b = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(ql, 4), m15), h23);
            const __m512i sc_a = _mm512_shuffle_epi8(sctab, shuf_a);
            const __m512i sc_b = _mm512_shuffle_epi8(sctab, shuf_b);
            for (int t = 0; t < NT; ++t) {
                const int8_t* q8 = y[t][i].qs + 128 * j;
                sumi512[t] = _mm512_dpwssd_epi32(sumi512[t], _mm512_maddubs_epi16(q6_a, _mm512_loadu_si512((const void*) q8)), sc_a);
                sumi512[t] = _mm512_dpwssd_epi32(sumi512[t], _mm512_maddubs_epi16(q6_b, _mm512_loadu_si512((const void*) (q8 + 64))), sc_b);
            }
        }
        for (int t = 0; t < NT; ++t) {
            const float d = y[t][i].d * dx;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y[t][i].bsums);
            const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);
            const __m256i sumi = _mm256_sub_epi32(fold256(sumi512[t]), q8sclsub);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) out[t] = hsum8_ggml(acc[t]);
}

template <int NT>
STRATA_FORCEINLINE void dot_q5_0_512(const block_q5_0* x, int nb, const block_q8_0* const* y, float* out) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m512i mf0 = _mm512_set1_epi8((char) 0xF0);
    const __m512i zero = _mm512_setzero_si512();
    const int pref = prefetch_distance();
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        if (pref > 0) _mm_prefetch((const char*) (x + ib) + pref, _MM_HINT_T0);
        const __m128i b0 = _mm_loadu_si128((const __m128i*) x[ib + 0].qs);
        const __m128i b1 = _mm_loadu_si128((const __m128i*) x[ib + 1].qs);
        const __m256i n0 = _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(b0), _mm_srli_epi16(b0, 4), 1), m4);
        const __m256i n1 = _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(b1), _mm_srli_epi16(b1, 4), 1), m4);
        const __m512i n01 = _mm512_inserti64x4(_mm512_castsi256_si512(n0), n1, 1);
        const __mmask64 hm = (__mmask64) u32(x[ib + 0].qh) | ((__mmask64) u32(x[ib + 1].qh) << 32);
        const __m512i qx = _mm512_mask_add_epi8(n01, ~hm, n01, mf0);
        const __m512i ax = _mm512_abs_epi8(qx);
        const __mmask64 neg = _mm512_movepi8_mask(qx);
        const __m128 xd = _mm_cvtph_ps(_mm_cvtsi32_si128((int) x[ib + 0].d | ((int) x[ib + 1].d << 16)));
        for (int t = 0; t < NT; ++t) {
            const __m256i y0 = _mm256_loadu_si256((const __m256i*) y[t][ib + 0].qs);
            const __m256i y1 = _mm256_loadu_si256((const __m256i*) y[t][ib + 1].qs);
            const __m512i qy = _mm512_inserti64x4(_mm512_castsi256_si512(y0), y1, 1);
            const __m512i sy = _mm512_mask_sub_epi8(qy, neg, zero, qy);
            const __m512 pf = _mm512_cvtepi32_ps(_mm512_dpbusd_epi32(zero, ax, sy));
            const __m128 yd = _mm_cvtph_ps(_mm_cvtsi32_si128((int) y[t][ib + 0].d | ((int) y[t][ib + 1].d << 16)));
            const __m128 d01 = _mm_mul_ps(xd, yd);
            acc[t] = _mm256_fmadd_ps(_mm256_broadcastss_ps(d01), _mm512_castps512_ps256(pf), acc[t]);
            acc[t] = _mm256_fmadd_ps(_mm256_broadcastss_ps(_mm_movehdup_ps(d01)), _mm512_extractf32x8_ps(pf, 1), acc[t]);
        }
    }
    for (; ib < nb; ++ib) {
        const __m128i b0 = _mm_loadu_si128((const __m128i*) x[ib].qs);
        const __m256i n0 = _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(b0), _mm_srli_epi16(b0, 4), 1), m4);
        const __mmask32 hm = (__mmask32) u32(x[ib].qh);
        const __m256i qx = _mm256_mask_add_epi8(n0, ~hm, n0, _mm256_set1_epi8((char) 0xF0));
        const __m256i ax = _mm256_abs_epi8(qx);
        const float dx0 = h2f(x[ib].d);
        for (int t = 0; t < NT; ++t) {
            const __m256i qy = _mm256_loadu_si256((const __m256i*) y[t][ib].qs);
            const __m256i sy = _mm256_sign_epi8(qy, qx);
            const __m256i p32 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ax, sy);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx0 * h2f(y[t][ib].d)), _mm256_cvtepi32_ps(p32), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) out[t] = hsum8_ggml(acc[t]);
}

template <int NT>
STRATA_FORCEINLINE void dot_q5_1_512(const block_q5_1* x, int nb, const block_q8_1* const* y, float* out) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m512i m10 = _mm512_set1_epi8(0x10);
    const __m512i zero = _mm512_setzero_si512();
    const int pref = prefetch_distance();
    __m256 acc[NT];
    float summs[NT];
    for (int t = 0; t < NT; ++t) {
        acc[t] = _mm256_setzero_ps();
        summs[t] = 0.f;
    }
    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        if (pref > 0) _mm_prefetch((const char*) (x + ib) + pref, _MM_HINT_T0);
        const __m128i b0 = _mm_loadu_si128((const __m128i*) x[ib + 0].qs);
        const __m128i b1 = _mm_loadu_si128((const __m128i*) x[ib + 1].qs);
        const __m256i n0 = _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(b0), _mm_srli_epi16(b0, 4), 1), m4);
        const __m256i n1 = _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(b1), _mm_srli_epi16(b1, 4), 1), m4);
        const __m512i n01 = _mm512_inserti64x4(_mm512_castsi256_si512(n0), n1, 1);
        const __mmask64 hm = (__mmask64) u32(x[ib + 0].qh) | ((__mmask64) u32(x[ib + 1].qh) << 32);
        const __m512i q5 = _mm512_mask_add_epi8(n01, hm, n01, m10);
        const uint8_t* x0 = (const uint8_t*) &x[ib + 0];
        const uint8_t* x1 = (const uint8_t*) &x[ib + 1];
        const __m128 x_dm = _mm_cvtph_ps(_mm_cvtsi64_si128((long long) ((uint64_t) u32(x0) | ((uint64_t) u32(x1) << 32))));
        for (int t = 0; t < NT; ++t) {
            const uint8_t* yb0 = (const uint8_t*) &y[t][ib + 0];
            const uint8_t* yb1 = (const uint8_t*) &y[t][ib + 1];
            const __m128 y_ds = _mm_cvtph_ps(_mm_cvtsi64_si128((long long) ((uint64_t) u32(yb0) | ((uint64_t) u32(yb1) << 32))));
            const __m128 p_ds = _mm_mul_ps(x_dm, y_ds);
            const __m128 p_hi = _mm_movehl_ps(p_ds, p_ds);
            summs[t] += _mm_cvtss_f32(_mm_movehdup_ps(p_ds));
            summs[t] += _mm_cvtss_f32(_mm_movehdup_ps(p_hi));
            const __m256i y0 = _mm256_loadu_si256((const __m256i*) y[t][ib + 0].qs);
            const __m256i y1 = _mm256_loadu_si256((const __m256i*) y[t][ib + 1].qs);
            const __m512i q8 = _mm512_inserti64x4(_mm512_castsi256_si512(y0), y1, 1);
            const __m512 pf = _mm512_cvtepi32_ps(_mm512_dpbusd_epi32(zero, q5, q8));
            acc[t] = _mm256_fmadd_ps(_mm512_castps512_ps256(pf), _mm256_broadcastss_ps(p_ds), acc[t]);
            acc[t] = _mm256_fmadd_ps(_mm512_extractf32x8_ps(pf, 1), _mm256_broadcastss_ps(p_hi), acc[t]);
        }
    }
    for (; ib < nb; ++ib) {
        const uint8_t* x0 = (const uint8_t*) &x[ib];
        const __m128i b0 = _mm_loadu_si128((const __m128i*) x[ib].qs);
        const __m256i n0 = _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(b0), _mm_srli_epi16(b0, 4), 1), m4);
        const __mmask32 hm = (__mmask32) u32(x[ib].qh);
        const __m256i q5 = _mm256_mask_add_epi8(n0, hm, n0, _mm256_set1_epi8(0x10));
        const float dx0 = h2f(u16(x0)), mx0 = h2f(u16(x0 + 2));
        for (int t = 0; t < NT; ++t) {
            const uint8_t* yb0 = (const uint8_t*) &y[t][ib];
            summs[t] += mx0 * h2f(u16(yb0 + 2));
            const __m256i q8 = _mm256_loadu_si256((const __m256i*) y[t][ib].qs);
            const __m256i p32 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), q5, q8);
            acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_set1_ps(dx0 * h2f(u16(yb0))), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) out[t] = hsum8_ggml(acc[t]) + summs[t];
}

template <int NT>
STRATA_FORCEINLINE void dot_q8_0_512(const block_q8_0* x, int nb, const block_q8_0* const* y, float* out) {
    const __m512i zero = _mm512_setzero_si512();
    const int pref = prefetch_distance();
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        if (pref > 0) _mm_prefetch((const char*) (x + ib) + pref, _MM_HINT_T0);
        const __m256i x0 = _mm256_loadu_si256((const __m256i*) x[ib + 0].qs);
        const __m256i x1 = _mm256_loadu_si256((const __m256i*) x[ib + 1].qs);
        const __m512i qx = _mm512_inserti64x4(_mm512_castsi256_si512(x0), x1, 1);
        const __m512i ax = _mm512_abs_epi8(qx);
        const __mmask64 neg = _mm512_movepi8_mask(qx);
        const __m128 xd = _mm_cvtph_ps(_mm_cvtsi32_si128((int) x[ib + 0].d | ((int) x[ib + 1].d << 16)));
        for (int t = 0; t < NT; ++t) {
            const __m256i y0 = _mm256_loadu_si256((const __m256i*) y[t][ib + 0].qs);
            const __m256i y1 = _mm256_loadu_si256((const __m256i*) y[t][ib + 1].qs);
            const __m512i qy = _mm512_inserti64x4(_mm512_castsi256_si512(y0), y1, 1);
            const __m512i sy = _mm512_mask_sub_epi8(qy, neg, zero, qy);
            const __m512 pf = _mm512_cvtepi32_ps(_mm512_dpbusd_epi32(zero, ax, sy));
            const __m128 yd = _mm_cvtph_ps(_mm_cvtsi32_si128((int) y[t][ib + 0].d | ((int) y[t][ib + 1].d << 16)));
            const __m128 d01 = _mm_mul_ps(xd, yd);
            acc[t] = _mm256_fmadd_ps(_mm256_broadcastss_ps(d01), _mm512_castps512_ps256(pf), acc[t]);
            acc[t] = _mm256_fmadd_ps(_mm256_broadcastss_ps(_mm_movehdup_ps(d01)), _mm512_extractf32x8_ps(pf, 1), acc[t]);
        }
    }
    for (; ib < nb; ++ib) {
        const __m256i qx = _mm256_loadu_si256((const __m256i*) x[ib].qs);
        const __m256i ax = _mm256_abs_epi8(qx);
        const float dx0 = h2f(x[ib].d);
        for (int t = 0; t < NT; ++t) {
            const __m256i qy = _mm256_loadu_si256((const __m256i*) y[t][ib].qs);
            const __m256i sy = _mm256_sign_epi8(qy, qx);
            const __m256i p32 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ax, sy);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx0 * h2f(y[t][ib].d)), _mm256_cvtepi32_ps(p32), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) out[t] = hsum8_ggml(acc[t]);
}

template <int NT>
void kq512_row_nt(int type, const uint8_t* w, int n, const void* const* act, float* out) {
    switch (type) {
        case 12: dot_q4_k_512<NT>((const block_q4_K*) w, n / QK_K, (const block_q8_K* const*) act, out); break;
        case 13: dot_q5_k_512<NT>((const block_q5_K*) w, n / QK_K, (const block_q8_K* const*) act, out); break;
        case 14: dot_q6_k_512<NT>((const block_q6_K*) w, n / QK_K, (const block_q8_K* const*) act, out); break;
        case 6:  dot_q5_0_512<NT>((const block_q5_0*) w, n / QK5_0, (const block_q8_0* const*) act, out); break;
        case 7:  dot_q5_1_512<NT>((const block_q5_1*) w, n / QK5_1, (const block_q8_1* const*) act, out); break;
        case 8:  dot_q8_0_512<NT>((const block_q8_0*) w, n / QK8_0, (const block_q8_0* const*) act, out); break;
        default: break;
    }
}

void kq512_row(int type, const uint8_t* w, int n, const void* const* act, int nt, float* out) {
    switch (nt) {
        case 1: kq512_row_nt<1>(type, w, n, act, out); break;
        case 2: kq512_row_nt<2>(type, w, n, act, out); break;
        case 3: kq512_row_nt<3>(type, w, n, act, out); break;
        default: kq512_row_nt<4>(type, w, n, act, out); break;
    }
}

template <int NT>
STRATA_FORCEINLINE void gu_q4_k_512(const block_q4_K* xg, const block_q4_K* xu, int nb, const block_q8_K* const* y,
                                    float* out_g, float* out_u) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m512i zero = _mm512_setzero_si512();
    const __m512i shuf0 = _mm512_mask_blend_epi64((__mmask8) 0xF0, _mm512_set1_epi16(0x0100), _mm512_set1_epi16(0x0302));
    const __m512i step4 = _mm512_set1_epi8(4);
    const int pref = prefetch_distance();
    __m256 acc_g[NT], acc_u[NT];
    __m128 accm_g[NT], accm_u[NT];
    for (int t = 0; t < NT; ++t) {
        acc_g[t] = acc_u[t] = _mm256_setzero_ps();
        accm_g[t] = accm_u[t] = _mm_setzero_ps();
    }
    for (int i = 0; i < nb; ++i) {
        if (pref > 0) {
            _mm_prefetch((const char*) (xg + i) + pref, _MM_HINT_T0);
            _mm_prefetch((const char*) (xu + i) + pref, _MM_HINT_T0);
        }
        const uint8_t* xbg = (const uint8_t*) &xg[i];
        const uint8_t* xbu = (const uint8_t*) &xu[i];
        const float dxg = h2f(u16(xbg)), dminxg = h2f(u16(xbg + 2));
        const float dxu = h2f(u16(xbu)), dminxu = h2f(u16(xbu + 2));
        uint64_t sc64_g, min64_g, sc64_u, min64_u;
        unpack_k4_scales(xg[i].scales, sc64_g, min64_g);
        unpack_k4_scales(xu[i].scales, sc64_u, min64_u);
        const __m512i sctab_g = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_cvtsi64_si128((long long) sc64_g)));
        const __m512i sctab_u = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_cvtsi64_si128((long long) sc64_u)));
        const __m128i mins_g = _mm_cvtepu8_epi16(_mm_cvtsi64_si128((long long) min64_g));
        const __m128i mins_u = _mm_cvtepu8_epi16(_mm_cvtsi64_si128((long long) min64_u));
        __m512i sumi_g[NT], sumi_u[NT];
        for (int t = 0; t < NT; ++t) sumi_g[t] = sumi_u[t] = zero;
        __m512i shuf = shuf0;
        for (int j = 0; j < 4; ++j) {
            const __m256i gbits = _mm256_loadu_si256((const __m256i*) (xg[i].qs + 32 * j));
            const __m256i ubits = _mm256_loadu_si256((const __m256i*) (xu[i].qs + 32 * j));
            const __m512i q4g = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_and_si256(gbits, m4)),
                                                   _mm256_and_si256(_mm256_srli_epi16(gbits, 4), m4), 1);
            const __m512i q4u = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_and_si256(ubits, m4)),
                                                   _mm256_and_si256(_mm256_srli_epi16(ubits, 4), m4), 1);
            const __m512i sc_g = _mm512_shuffle_epi8(sctab_g, shuf);
            const __m512i sc_u = _mm512_shuffle_epi8(sctab_u, shuf);
            shuf = _mm512_add_epi8(shuf, step4);
            for (int t = 0; t < NT; ++t) {
                const __m512i q8v = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j));
                sumi_g[t] = _mm512_dpwssd_epi32(sumi_g[t], _mm512_maddubs_epi16(q4g, q8v), sc_g);
                sumi_u[t] = _mm512_dpwssd_epi32(sumi_u[t], _mm512_maddubs_epi16(q4u, q8v), sc_u);
            }
        }
        for (int t = 0; t < NT; ++t) {
            const float yd = y[t][i].d;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y[t][i].bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_castsi256_si128(q8sums), _mm256_extracti128_si256(q8sums, 1));
            accm_g[t] = _mm_fmadd_ps(_mm_set1_ps(-yd * dminxg), _mm_cvtepi32_ps(_mm_madd_epi16(mins_g, q8s)), accm_g[t]);
            accm_u[t] = _mm_fmadd_ps(_mm_set1_ps(-yd * dminxu), _mm_cvtepi32_ps(_mm_madd_epi16(mins_u, q8s)), accm_u[t]);
            acc_g[t] = _mm256_fmadd_ps(_mm256_set1_ps(yd * dxg), _mm256_cvtepi32_ps(fold256(sumi_g[t])), acc_g[t]);
            acc_u[t] = _mm256_fmadd_ps(_mm256_set1_ps(yd * dxu), _mm256_cvtepi32_ps(fold256(sumi_u[t])), acc_u[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        __m128 mg = _mm_add_ps(accm_g[t], _mm_movehl_ps(accm_g[t], accm_g[t]));
        mg = _mm_add_ss(mg, _mm_movehdup_ps(mg));
        __m128 mu = _mm_add_ps(accm_u[t], _mm_movehl_ps(accm_u[t], accm_u[t]));
        mu = _mm_add_ss(mu, _mm_movehdup_ps(mu));
        out_g[t] = hsum8_ggml(acc_g[t]) + _mm_cvtss_f32(mg);
        out_u[t] = hsum8_ggml(acc_u[t]) + _mm_cvtss_f32(mu);
    }
}

template <int NT>
STRATA_FORCEINLINE void gu_q5_k_512(const block_q5_K* xg, const block_q5_K* xu, int nb, const block_q8_K* const* y,
                                    float* out_g, float* out_u) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m512i m10 = _mm512_set1_epi8(0x10);
    const __m128i mzero = _mm_setzero_si128();
    const __m512i zero = _mm512_setzero_si512();
    const __m512i shuf0 = _mm512_mask_blend_epi64((__mmask8) 0xF0, _mm512_set1_epi16(0x0100), _mm512_set1_epi16(0x0302));
    const __m512i step4 = _mm512_set1_epi8(4);
    const __m512i bmask0 = _mm512_mask_blend_epi64((__mmask8) 0xF0, _mm512_set1_epi8(1), _mm512_set1_epi8(2));
    const int pref = prefetch_distance();
    __m256 acc_g[NT], acc_u[NT];
    float summs_g[NT], summs_u[NT];
    for (int t = 0; t < NT; ++t) {
        acc_g[t] = acc_u[t] = _mm256_setzero_ps();
        summs_g[t] = summs_u[t] = 0.f;
    }
    for (int i = 0; i < nb; ++i) {
        if (pref > 0) {
            _mm_prefetch((const char*) (xg + i) + pref, _MM_HINT_T0);
            _mm_prefetch((const char*) (xu + i) + pref, _MM_HINT_T0);
        }
        const uint8_t* xbg = (const uint8_t*) &xg[i];
        const uint8_t* xbu = (const uint8_t*) &xu[i];
        const float dxg = h2f(u16(xbg)), dminxg = h2f(u16(xbg + 2));
        const float dxu = h2f(u16(xbu)), dminxu = h2f(u16(xbu + 2));
        uint64_t sc64_g, min64_g, sc64_u, min64_u;
        unpack_k4_scales(xg[i].scales, sc64_g, min64_g);
        unpack_k4_scales(xu[i].scales, sc64_u, min64_u);
        const __m512i sctab_g = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_cvtsi64_si128((long long) sc64_g)));
        const __m512i sctab_u = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_cvtsi64_si128((long long) sc64_u)));
        const __m128i mins_g = _mm_cvtepu8_epi16(_mm_cvtsi64_si128((long long) min64_g));
        const __m128i mins_u = _mm_cvtepu8_epi16(_mm_cvtsi64_si128((long long) min64_u));
        const __m512i hbits_g = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*) xg[i].qh));
        const __m512i hbits_u = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*) xu[i].qh));
        __m512i sumi_g[NT], sumi_u[NT];
        for (int t = 0; t < NT; ++t) sumi_g[t] = sumi_u[t] = zero;
        __m512i shuf = shuf0, bmask = bmask0;
        for (int j = 0; j < 4; ++j) {
            const __m256i gbits = _mm256_loadu_si256((const __m256i*) (xg[i].qs + 32 * j));
            const __m256i ubits = _mm256_loadu_si256((const __m256i*) (xu[i].qs + 32 * j));
            const __m512i q4g = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_and_si256(gbits, m4)),
                                                   _mm256_and_si256(_mm256_srli_epi16(gbits, 4), m4), 1);
            const __m512i q4u = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_and_si256(ubits, m4)),
                                                   _mm256_and_si256(_mm256_srli_epi16(ubits, 4), m4), 1);
            const __m512i q5g = _mm512_mask_add_epi8(q4g, _mm512_test_epi8_mask(hbits_g, bmask), q4g, m10);
            const __m512i q5u = _mm512_mask_add_epi8(q4u, _mm512_test_epi8_mask(hbits_u, bmask), q4u, m10);
            const __m512i sc_g = _mm512_shuffle_epi8(sctab_g, shuf);
            const __m512i sc_u = _mm512_shuffle_epi8(sctab_u, shuf);
            shuf = _mm512_add_epi8(shuf, step4);
            bmask = _mm512_slli_epi16(bmask, 2);
            for (int t = 0; t < NT; ++t) {
                const __m512i q8v = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j));
                sumi_g[t] = _mm512_dpwssd_epi32(sumi_g[t], _mm512_maddubs_epi16(q5g, q8v), sc_g);
                sumi_u[t] = _mm512_dpwssd_epi32(sumi_u[t], _mm512_maddubs_epi16(q5u, q8v), sc_u);
            }
        }
        for (int t = 0; t < NT; ++t) {
            const float yd = y[t][i].d;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y[t][i].bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_castsi256_si128(q8sums), _mm256_extracti128_si256(q8sums, 1));
            const __m128i pg = _mm_madd_epi16(mins_g, q8s);
            const __m128i pu = _mm_madd_epi16(mins_u, q8s);
            summs_g[t] += (-yd * dminxg) * _mm_extract_epi32(_mm_hadd_epi32(_mm_hadd_epi32(pg, mzero), mzero), 0);
            summs_u[t] += (-yd * dminxu) * _mm_extract_epi32(_mm_hadd_epi32(_mm_hadd_epi32(pu, mzero), mzero), 0);
            acc_g[t] = _mm256_fmadd_ps(_mm256_set1_ps(yd * dxg), _mm256_cvtepi32_ps(fold256(sumi_g[t])), acc_g[t]);
            acc_u[t] = _mm256_fmadd_ps(_mm256_set1_ps(yd * dxu), _mm256_cvtepi32_ps(fold256(sumi_u[t])), acc_u[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        out_g[t] = hsum8_ggml(acc_g[t]) + summs_g[t];
        out_u[t] = hsum8_ggml(acc_u[t]) + summs_u[t];
    }
}

template <int NT>
STRATA_FORCEINLINE void gu_q6_k_512(const block_q6_K* xg, const block_q6_K* xu, int nb, const block_q8_K* const* y,
                                    float* out_g, float* out_u) {
    const __m512i m15 = _mm512_set1_epi8(15);
    const __m512i m3 = _mm512_set1_epi8(3);
    const __m512i m12 = _mm512_set1_epi8(12);
    const __m512i m48 = _mm512_set1_epi8(48);
    const __m512i m192 = _mm512_set1_epi8(-64);
    const __m512i zero = _mm512_setzero_si512();
    const __m512i shuf_a = _mm512_set_epi32(
        0x07060706, 0x07060706, 0x07060706, 0x07060706,
        0x05040504, 0x05040504, 0x05040504, 0x05040504,
        0x03020302, 0x03020302, 0x03020302, 0x03020302,
        0x01000100, 0x01000100, 0x01000100, 0x01000100);
    const __m512i shuf_b = _mm512_add_epi8(shuf_a, _mm512_set1_epi8(8));
    const int pref = prefetch_distance();
    __m256 acc_g[NT], acc_u[NT];
    for (int t = 0; t < NT; ++t) acc_g[t] = acc_u[t] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        if (pref > 0) {
            _mm_prefetch((const char*) (xg + i) + pref, _MM_HINT_T0);
            _mm_prefetch((const char*) (xu + i) + pref, _MM_HINT_T0);
        }
        const float dxg = h2f(xg[i].d), dxu = h2f(xu[i].d);
        const __m256i sc16_g = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) xg[i].scales));
        const __m256i sc16_u = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) xu[i].scales));
        __m512i sumi_g[NT], sumi_u[NT];
        for (int t = 0; t < NT; ++t) sumi_g[t] = sumi_u[t] = zero;
        for (int j = 0; j < 2; ++j) {
            const __m512i sctab_g = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (xg[i].scales + 8 * j))));
            const __m512i sctab_u = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (xu[i].scales + 8 * j))));
            auto dec = [&](const block_q6_K& b, __m512i& qa, __m512i& qb) {
                const __m512i ql = _mm512_loadu_si512((const void*) (b.ql + 64 * j));
                const __m512i qh = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*) (b.qh + 32 * j)));
                const __m512i h01 = _mm512_mask_blend_epi64((__mmask8) 0xF0, _mm512_slli_epi16(_mm512_and_si512(qh, m3), 4),
                                                            _mm512_slli_epi16(_mm512_and_si512(qh, m12), 2));
                const __m512i h23 = _mm512_mask_blend_epi64((__mmask8) 0xF0, _mm512_and_si512(qh, m48),
                                                            _mm512_srli_epi16(_mm512_and_si512(qh, m192), 2));
                qa = _mm512_or_si512(_mm512_and_si512(ql, m15), h01);
                qb = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(ql, 4), m15), h23);
            };
            __m512i qga, qgb, qua, qub;
            dec(xg[i], qga, qgb);
            dec(xu[i], qua, qub);
            const __m512i sc_ga = _mm512_shuffle_epi8(sctab_g, shuf_a);
            const __m512i sc_gb = _mm512_shuffle_epi8(sctab_g, shuf_b);
            const __m512i sc_ua = _mm512_shuffle_epi8(sctab_u, shuf_a);
            const __m512i sc_ub = _mm512_shuffle_epi8(sctab_u, shuf_b);
            for (int t = 0; t < NT; ++t) {
                const __m512i q8a = _mm512_loadu_si512((const void*) (y[t][i].qs + 128 * j));
                const __m512i q8b = _mm512_loadu_si512((const void*) (y[t][i].qs + 128 * j + 64));
                sumi_g[t] = _mm512_dpwssd_epi32(sumi_g[t], _mm512_maddubs_epi16(qga, q8a), sc_ga);
                sumi_g[t] = _mm512_dpwssd_epi32(sumi_g[t], _mm512_maddubs_epi16(qgb, q8b), sc_gb);
                sumi_u[t] = _mm512_dpwssd_epi32(sumi_u[t], _mm512_maddubs_epi16(qua, q8a), sc_ua);
                sumi_u[t] = _mm512_dpwssd_epi32(sumi_u[t], _mm512_maddubs_epi16(qub, q8b), sc_ub);
            }
        }
        for (int t = 0; t < NT; ++t) {
            const float yd = y[t][i].d;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y[t][i].bsums);
            const __m256i sub_g = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, sc16_g), 5);
            const __m256i sub_u = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, sc16_u), 5);
            acc_g[t] = _mm256_fmadd_ps(_mm256_set1_ps(yd * dxg), _mm256_cvtepi32_ps(_mm256_sub_epi32(fold256(sumi_g[t]), sub_g)), acc_g[t]);
            acc_u[t] = _mm256_fmadd_ps(_mm256_set1_ps(yd * dxu), _mm256_cvtepi32_ps(_mm256_sub_epi32(fold256(sumi_u[t]), sub_u)), acc_u[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        out_g[t] = hsum8_ggml(acc_g[t]);
        out_u[t] = hsum8_ggml(acc_u[t]);
    }
}

template <int NT>
void kq512_gu_row_nt(int type, const uint8_t* g_ptr, const uint8_t* u_ptr, int n, const void* const* act,
                     float* g_out, float* u_out) {
    switch (type) {
        case 12:
            gu_q4_k_512<NT>((const block_q4_K*) g_ptr, (const block_q4_K*) u_ptr, n / QK_K,
                            (const block_q8_K* const*) act, g_out, u_out);
            break;
        case 13:
            gu_q5_k_512<NT>((const block_q5_K*) g_ptr, (const block_q5_K*) u_ptr, n / QK_K,
                            (const block_q8_K* const*) act, g_out, u_out);
            break;
        case 14:
            gu_q6_k_512<NT>((const block_q6_K*) g_ptr, (const block_q6_K*) u_ptr, n / QK_K,
                            (const block_q8_K* const*) act, g_out, u_out);
            break;
        default:
            kq512_row_nt<NT>(type, g_ptr, n, act, g_out);
            kq512_row_nt<NT>(type, u_ptr, n, act, u_out);
            break;
    }
}

void kq512_gu_row(int type, const uint8_t* g_ptr, const uint8_t* u_ptr, int n, const void* const* act, int nt,
                  float* g_out, float* u_out) {
    switch (nt) {
        case 1: kq512_gu_row_nt<1>(type, g_ptr, u_ptr, n, act, g_out, u_out); break;
        case 2: kq512_gu_row_nt<2>(type, g_ptr, u_ptr, n, act, g_out, u_out); break;
        case 3: kq512_gu_row_nt<3>(type, g_ptr, u_ptr, n, act, g_out, u_out); break;
        default: kq512_gu_row_nt<4>(type, g_ptr, u_ptr, n, act, g_out, u_out); break;
    }
}

}  // namespace

void kq512_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1) {
    for (int t0 = 0; t0 < nt; t0 += kKQMaxT) {
        const int m = nt - t0 < kKQMaxT ? nt - t0 : kKQMaxT;
        float g[kKQMaxT], u[kKQMaxT];
        for (int r = r0; r < r1; ++r) {
            kq512_gu_row(type, blob + (size_t) r * gu_row, blob + up_off + (size_t) r * gu_row, n, act + t0, m, g, u);
            for (int t = 0; t < m; ++t) ff[t0 + t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
        }
    }
}

void kq512_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1) {
    for (int t0 = 0; t0 < nt; t0 += kKQMaxT) {
        const int m = nt - t0 < kKQMaxT ? nt - t0 : kKQMaxT;
        float s[kKQMaxT];
        for (int r = r0; r < r1; ++r) {
            kq512_row(type, w + (size_t) r * row_bytes, n, act + t0, m, s);
            for (int t = 0; t < m; ++t) out[t0 + t][r] = s[t];
        }
    }
}

void q8k_quant_avx512(const float* x, void* vy, int64_t k) {
    block_q8_K* y = (block_q8_K*) vy;
    const int64_t nb = k / QK_K;
    const __m512 absm = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
    const __m512 magic = _mm512_set1_ps(12582912.f);
    const __m512i mant = _mm512_set1_epi32(0x007fffff), off = _mm512_set1_epi32(0x00400000);
    const __m512i c127 = _mm512_set1_epi32(127);
    for (int64_t i = 0; i < nb; ++i, x += QK_K) {
        __m512 m = _mm512_setzero_ps();
        // abs_x is the first operand of _mm512_max_ps(abs_x, m) so any NaN lane returns m (matching the C reference).
        for (int j = 0; j < QK_K; j += 16)
            m = _mm512_max_ps(_mm512_and_ps(_mm512_loadu_ps(x + j), absm), m);
        const __m256 m256 = _mm256_max_ps(_mm512_castps512_ps256(m), _mm512_extractf32x8_ps(m, 1));
        __m128 h = _mm_max_ps(_mm256_castps256_ps128(m256), _mm256_extractf128_ps(m256, 1));
        h = _mm_max_ps(h, _mm_movehl_ps(h, h));
        h = _mm_max_ss(h, _mm_movehdup_ps(h));
        const float amax = _mm_cvtss_f32(h);
        if (!amax) {
            y[i].d = 0;
            std::memset(y[i].qs, 0, QK_K);
            continue;
        }
        float mx = 0.f;
        const __m512 va = _mm512_set1_ps(amax);
        for (int j = 0; j < QK_K; j += 16) {
            const uint32_t msk = (uint32_t) _mm512_cmp_ps_mask(_mm512_and_ps(_mm512_loadu_ps(x + j), absm), va, _CMP_EQ_OQ);
            if (msk) {
                int b = 0;
                while (!((msk >> b) & 1u)) ++b;
                mx = x[j + b];
                break;
            }
        }
        const float iscale = -127.f / mx;
        const __m512 vis = _mm512_set1_ps(iscale);
        for (int j = 0; j < QK_K; j += 16) {
            __m512 p = _mm512_mul_ps(vis, _mm512_loadu_ps(x + j));
#if defined(__GNUC__) && !defined(__clang__)
            __asm__("" : "+v"(p));
#endif
            const __m512 t = _mm512_add_ps(p, magic);
            __m512i v = _mm512_min_epi32(_mm512_sub_epi32(_mm512_and_si512(_mm512_castps_si512(t), mant), off), c127);
            v = _mm512_srai_epi32(_mm512_slli_epi32(v, 24), 24);
            y[i].bsums[j / 16] = (int16_t) _mm512_reduce_add_epi32(v);
            _mm_storeu_si128((__m128i*) (y[i].qs + j), _mm512_cvtepi32_epi8(v));
        }
        y[i].d = 1 / iscale;
    }
}

void bf16_rows_dot_512(const uint16_t* w, int rows, int cols, const float* x, float* out) {
    for (int r = 0; r < rows; ++r) {
        const uint16_t* wr = w + (size_t) r * (size_t) cols;
        __m512 acc = _mm512_setzero_ps();
        int c = 0;
        for (; c + 16 <= cols; c += 16) {
            const __m512i h = _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*) (wr + c)));
            acc = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(h, 16)), _mm512_loadu_ps(x + c), acc);
        }
        const __m256 a256 = _mm256_add_ps(_mm512_castps512_ps256(acc), _mm512_extractf32x8_ps(acc, 1));
        float s = hsum8_ggml(a256);
        for (; c < cols; ++c) {
            const uint32_t b = (uint32_t) wr[c] << 16;
            float f;
            std::memcpy(&f, &b, 4);
            s += f * x[c];
        }
        out[r] = s;
    }
}

void bf16_rows_dot_multi_512(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) {
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int m = nt - t0 < 4 ? nt - t0 : 4;
        const float* x0 = x + (size_t) (t0 + 0) * cols;
        const float* x1 = m > 1 ? x + (size_t) (t0 + 1) * cols : x0;
        const float* x2 = m > 2 ? x + (size_t) (t0 + 2) * cols : x0;
        const float* x3 = m > 3 ? x + (size_t) (t0 + 3) * cols : x0;
        for (int r = 0; r < rows; ++r) {
            const uint16_t* wr = w + (size_t) r * (size_t) cols;
            __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
            __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
            int c = 0;
            for (; c + 16 <= cols; c += 16) {
                const __m512 wf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*) (wr + c))), 16));
                a0 = _mm512_fmadd_ps(wf, _mm512_loadu_ps(x0 + c), a0);
                if (m > 1) a1 = _mm512_fmadd_ps(wf, _mm512_loadu_ps(x1 + c), a1);
                if (m > 2) a2 = _mm512_fmadd_ps(wf, _mm512_loadu_ps(x2 + c), a2);
                if (m > 3) a3 = _mm512_fmadd_ps(wf, _mm512_loadu_ps(x3 + c), a3);
            }
            auto reduce = [](__m512 v) {
                return hsum8_ggml(_mm256_add_ps(_mm512_castps512_ps256(v), _mm512_extractf32x8_ps(v, 1)));
            };
            float s0 = reduce(a0), s1 = m > 1 ? reduce(a1) : 0.f, s2 = m > 2 ? reduce(a2) : 0.f, s3 = m > 3 ? reduce(a3) : 0.f;
            for (; c < cols; ++c) {
                const uint32_t b = (uint32_t) wr[c] << 16;
                float f;
                std::memcpy(&f, &b, 4);
                s0 += f * x0[c];
                if (m > 1) s1 += f * x1[c];
                if (m > 2) s2 += f * x2[c];
                if (m > 3) s3 += f * x3[c];
            }
            out[(size_t) (t0 + 0) * rows + r] = s0;
            if (m > 1) out[(size_t) (t0 + 1) * rows + r] = s1;
            if (m > 2) out[(size_t) (t0 + 2) * rows + r] = s2;
            if (m > 3) out[(size_t) (t0 + 3) * rows + r] = s3;
        }
    }
}


}  // namespace strata::kernels::cpu
