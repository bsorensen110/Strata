// include/strata/kernels/kv_q2.hpp - 2-bit KV storage with Walsh-Hadamard rotation for the QSA layers
// (`--kv q2`). Design and the measurement plan: docs/kv-q2.md.
//
// The same shape as kv_q4.hpp, at half the code width: K and V are rotated by the orthonormal 256-point Hadamard
// matrix (fwht256_cuda, reused unchanged) before quantization, the query is rotated the same way so <Hq,Hk> =
// <q,k>, and the attention output is rotated back. What is lossy is the 2-bit rounding, and at 2 bits that
// rounding is large enough to reach the model's output - which is why this format is meant to be used with the
// FP16 sliding window + attention sinks of kv_window.hpp: the `sinks` leading cells and the `window` most recent
// cells stay exact, and only the bulk is 2-bit.
//
// Codes are 2 bits per value in 32-value blocks: an fp16 scale plus 8 bytes of codes (10 B per block, 8 blocks per
// head per cell -> 64 B of codes + 16 B of scales = 80 B per head per cell, vs 144 B for q4_0 and 512 B for fp16).
// The shipping codebook is the 4-level Lloyd-Max grid (q2_lm4_level) scaled by the block's RMS, for K AND V: the
// parity gate measured it 1.80x better than a uniform grid over the block max on both, and better than the 3-level
// grid on both per-element RMS and attention output. The 3-level grid (q2_lm3_level) is kept because the gate
// compares the two and because the entropy-coded "1.5-bit V" of SKVQ (arXiv 2405.06219) is a writer/reader change
// later, not a layout change now. 2 bits stored is not 1.5 bits stored, and nothing here claims otherwise.
//
// With KV streaming (--kv-resident) the codes live in the host copy and the resident slots like q4_0's (KvHostPools
// k_q2/v_q2, the same per-cell granule), so both options combine.
#pragma once

#include "strata/kernels/f16_bits.hpp"      // for STRATA_HD: __host__ __device__ under nvcc, empty on host-only
#include "strata/kernels/qsa.hpp"

#include <cmath>
#include <cstdint>

namespace strata::kernels {

inline constexpr int Q2_0 = 32;      ///< values per block, as in q4_0

#pragma pack(push, 1)
struct block_q2_0 {
    uint16_t d;             // scale (fp16 bits)
    uint8_t qs[Q2_0 / 4];   // 32 2-bit codes = 8 bytes; element j in bits [2*(j%4), +2) of qs[j/4]
};
#pragma pack(pop)

static_assert(sizeof(block_q2_0) == 10, "block_q2_0 must be 10 bytes: 2 of scale + 8 of codes");

/// Bytes per cell and KV head: 8 blocks of 32 for head_dim 256 -> 80 (8 x 10).
inline uint64_t kv_q2_bytes_per_head(int head_dim) { return (uint64_t) (head_dim / Q2_0) * sizeof(block_q2_0); }

/// Bytes per cell (one token, one layer): K and V of every KV head.
inline uint64_t kv_q2_bytes_per_cell(const QsaShapes& s) {
    return (uint64_t) s.n_head_kv * kv_q2_bytes_per_head((int) s.head_dim) * 2;
}

/// The 4-level codebook K uses: the code in {0,1,2,3} nearest to x / d, mapped so 0 -> -3, 1 -> -1, 2 -> +1,
/// 3 -> +3. Device and host share this mapping so the parity test can reproduce the kernel bit for bit.
/// d is the block's scale; d == 0 (an all-zero block) takes the code for 0.
inline int q2_code_4level(float x, float d) {
    if (!(d > 0.f)) return 1;                              // 1 -> -1 * d == 0 when d is 0
    const float q = x * (1.0f / d);                        // the device multiplies by the reciprocal; x / d is a
    int c = (int) std::nearbyintf(q);                      // different float, so the host has to do it the same way
    if (c < -3) c = -3;                                    // (nearbyintf = nearest, ties to even = __float2int_rn)
    if (c > 3) c = 3;
    return (c + 3) >> 1;                                   // -3 -> 0, -1 -> 1, +1 -> 2, +3 -> 3
}
inline float q2_value_4level(int code, float d) { return d * (float) (2 * code - 3); }   // 0..3 -> -3,-1,+1,+3

/// The 3-level codebook V uses: 0 -> -1, 1 -> 0, 2 -> +1, and 3 is never written.
inline int q2_code_3level(float x, float d) {
    if (!(d > 0.f)) return 1;                              // 1 -> 0 * d == 0
    const float q = x * (1.0f / d);                        // reciprocal multiply, as on the device
    if (q >= 0.5f) return 2;
    if (q <= -0.5f) return 0;
    return 1;
}
inline float q2_value_3level(int code, float d) { return d * (float) (code - 1); }         // 0..2 -> -1, 0, +1

/// Lloyd-Max levels for a Gaussian source: the codebook SKVQ's gain rests on. A uniform grid over the block's
/// max puts its inner level at amax/3 ~ 0.9 sigma, so a typical value (one sigma) is quantized 0.9 sigma away
/// from where it lives; these put the inner level at 0.459 sigma, measured 1.80x better on both K and V at the
/// same 2 bits. s is the block's RMS, not its max. The levels are functions, not an array: nvcc will read a
/// namespace-scope constexpr scalar in device code, but not a constexpr array.
STRATA_HD inline float q2_lm4_level(int code) {
    return code == 0 ? -1.5097f : code == 1 ? -0.4591f : code == 2 ? 0.4591f : 1.5097f;
}
STRATA_HD inline float q2_lm3_level(int code) {
    return code == 0 ? -1.4544f : code == 2 ? 1.4544f : 0.0f;
}
constexpr float kQ2Lm4T = 0.9844f;                            // midpoint of 0.4591 and 1.5097
constexpr float kQ2Lm3T = 0.7272f;                            // midpoint of 0 and 1.4544

/// Written to mirror the device exactly: reciprocal multiply, same thresholds, and s == 0 falls through to the
/// same code the warp takes (2 for the 4-level, 1 for the 3-level), because -0.0 and +0.0 are not the same bits.
STRATA_HD inline int q2_code_lm4(float x, float s) {
    const float q = s > 0.f ? x * (1.0f / s) : 0.f;
    const float a = q < 0.f ? -q : q;
    if (a < kQ2Lm4T) return q < 0.f ? 1 : 2;
    return q < 0.f ? 0 : 3;
}
STRATA_HD inline float q2_value_lm4(int code, float s) { return s * q2_lm4_level(code); }
STRATA_HD inline int q2_code_lm3(float x, float s) {
    const float q = s > 0.f ? x * (1.0f / s) : 0.f;
    if (q >= kQ2Lm3T) return 2;
    if (q <= -kQ2Lm3T) return 0;
    return 1;
}
STRATA_HD inline float q2_value_lm3(int code, float s) { return s * q2_lm3_level(code); }

/// Append the (already rotated) cell at step[kStepPos]. With a host copy (KV streaming) it is written there too,
/// and to VRAM only if its block is resident. `v3level` selects V's codebook; K is always the 4-level one. The
/// engine appends BOTH pools in one call with v3level = false, the shipping setting (4-level Lloyd-Max for K and
/// V). True is used only by the parity gate, which measures the 3-level grid against it.
void kv_append_q2_step(uint8_t* k_q2, uint8_t* v_q2, const int32_t* page_table, const int32_t* step,
                       const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host = nullptr, bool v3level = true);

void kv_append_q2_steps(uint8_t* k_q2, uint8_t* v_q2, const int32_t* page_table, const int32_t* step,
                        int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                        void* stream, const KvHostPools* host = nullptr, bool v3level = true);

/// The prompt path: T consecutive (rotated) cells from pos0, K/V [T, n_head_kv, 256]; also into `stage` (identity
/// layout, the one-layer staging pool of a streamed session) when given.
void kv_append_q2(uint8_t* k_q2, uint8_t* v_q2, const int32_t* page_table, int64_t pos0, int64_t T, const float* K,
                  const float* V, const QsaShapes& s, void* stream, const KvHostPools* host = nullptr,
                  const KvHostPools* stage = nullptr, bool v3level = true);

/// Gather step[kStepWidth] cells named by `ids` into FP16 scratch [id][kv_head][head_dim] (still rotated).
void kv_gather_q2_step(const uint8_t* k_q2, const uint8_t* v_q2, const int32_t* page_table, const int32_t* ids,
                       const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                       uint16_t* v_scratch, void* stream, bool v3level = true);

}  // namespace strata::kernels
