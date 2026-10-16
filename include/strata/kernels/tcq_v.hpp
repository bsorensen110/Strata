// include/strata/kernels/tcq_v.hpp - trellis-coded 2-bit V (per-token 32-value blocks), device-compatible.
//
// Format: block_tcq2_0 = { f16 scale, 8 bytes of codes } = 10 bytes per 32 values, the same byte shape as
// block_q2_0 in kv_q2.hpp. Only the meaning of a code changes: each value's 2-bit code is its position inside
// the union of grid points that the trellis state allows (see below), not a standalone level. The decoder walks
// the state machine, so decoding value t needs the codes of values 0..t-1. Each block starts in state 0.
//
// Grid and trellis (numpy reference: tools/tcq_trellis_gen.py + tools/tcq_v_multistate.py):
//   grid point j = 0..7 has level (j - 3.5) * delta(states); value = scale * level(j).
//   subset k = j % 4. State s has z0 = s & 1 (its union is {j : j % 2 == z0}, i.e. 2*idx + z0 for idx 0..3).
//   The code of a value is idx (2 bits); the grid point is j = 2*idx + z0, and the next state is
//   next(s, j % 4): the trellis is the Ungerboeck parity check h0, h1 (octal) on a nu-bit observer register.
//   4 states: h0 = 05, h1 = 02 (nu 2). 8 states: h0 = 013, h1 = 04 (nu 3). 16 states: h0 = 023, h1 = 04 (nu 4).
//
// Encoder: Viterbi over the trellis (min squared error), 32 steps, cost per state, back pointers in per-thread
// arrays. Decoder: one state walk, 32 table steps. No namespace-scope tables (a device can't read them).
#pragma once

#include <cmath>    // sqrtf: device-callable under nvcc, host otherwise
#include <cstdint>

#include "strata/kernels/f16_bits.hpp"  // STRATA_HD, f16_from_f32, f32_from_f16

namespace strata::kernels {

inline constexpr int kTcqBlock = 32;     ///< values per block (the scale granule)
inline constexpr int kTcqMaxStates = 16;

struct block_tcq2_0 {
    uint16_t d;       ///< f16 scale (RMS of the block)
    uint8_t qs[8];    ///< 32 two-bit codes; value t's code in bits [2*(t%4), +2) of qs[t/4]
};
static_assert(sizeof(block_tcq2_0) == 10, "block_tcq2_0 must be 10 bytes: 2 of scale + 8 of codes");

STRATA_HD inline int tcq_nu(int states) { return states >= 16 ? 4 : states >= 8 ? 3 : 2; }
STRATA_HD inline int tcq_h0(int states) { return states >= 16 ? 023 : states >= 8 ? 013 : 05; }
STRATA_HD inline int tcq_h1(int states) { return states >= 8 ? 04 : 02; }

/// Grid step, fitted on a unit Gaussian for each trellis (tools/tcq_v_multistate.py, fit_delta).
STRATA_HD inline float tcq_delta(int states) { return states >= 16 ? 0.4940f : states >= 8 ? 0.4935f : 0.4945f; }

/// Level of grid point j in units of the block scale.
STRATA_HD inline float tcq_level(int states, int j) { return ((float) j - 3.5f) * tcq_delta(states); }

/// Next state after taking subset k (= j % 4) from state s; -1 when k is not in s's union.
STRATA_HD inline int tcq_next(int states, int s, int k) {
    const int z0 = s & 1;
    if ((k & 1) != z0) return -1;
    const int u = k >> 1;                      // the free bit of the subset (the 2nd code bit's worth of choice)
    const int nu = tcq_nu(states), h0 = tcq_h0(states), h1 = tcq_h1(states);
    int ns = 0;
    for (int i = 0; i < nu; ++i) {
        const int r_next = (i + 1 < nu) ? ((s >> (i + 1)) & 1) : 0;
        const int den = (h0 >> (i + 1)) & 1;
        const int num = (h1 >> (i + 1)) & 1;
        ns |= (r_next ^ (den & z0) ^ (num & u)) << i;
    }
    return ns;
}

/// Grid index of a value's code in state s: the code is the position inside the union {2*idx + z0}.
STRATA_HD inline int tcq_point(int s, int code) { return 2 * code + (s & 1); }

STRATA_HD inline int tcq_code_at(const uint8_t* qs, int t) { return (qs[t >> 2] >> (2 * (t & 3))) & 3; }

/// Decode one block: out[t] = scale * level(point of code t in the walked state).
STRATA_HD inline void tcq_decode_block(int states, float scale, const uint8_t* qs, float* out) {
    int s = 0;
    for (int t = 0; t < kTcqBlock; ++t) {
        const int j = tcq_point(s, tcq_code_at(qs, t));
        out[t] = scale * tcq_level(states, j);
        s = tcq_next(states, s, j & 3);
    }
}

/// Viterbi encoder for one block of normalized values u (= x / scale). Writes the 8 code bytes (zeroed first).
/// Returns the total squared error of the chosen path (in units of the scale).
STRATA_HD inline float tcq_encode_block(int states, const float* u, uint8_t* qs) {
    constexpr float kInf = 1.0e30f;
    float cost[kTcqMaxStates], next_cost[kTcqMaxStates];
    uint8_t back_state[kTcqBlock][kTcqMaxStates];
    uint8_t back_point[kTcqBlock][kTcqMaxStates];
    for (int s = 0; s < states; ++s) cost[s] = (s == 0) ? 0.0f : kInf;
    for (int t = 0; t < kTcqBlock; ++t) {
        float e[4];
        int jb[4];
        for (int k = 0; k < 4; ++k) {  // nearest point inside subset {k, k+4}
            const float d0 = u[t] - tcq_level(states, k);
            const float d1 = u[t] - tcq_level(states, k + 4);
            const bool pick1 = d1 * d1 < d0 * d0;
            e[k] = pick1 ? d1 * d1 : d0 * d0;
            jb[k] = pick1 ? k + 4 : k;
        }
        for (int s = 0; s < states; ++s) next_cost[s] = kInf;
        for (int s = 0; s < states; ++s) {
            for (int k = 0; k < 4; ++k) {
                const int ns = tcq_next(states, s, k);
                if (ns < 0) continue;
                const float c = cost[s] + e[k];
                if (c < next_cost[ns]) {
                    next_cost[ns] = c;
                    back_state[t][ns] = (uint8_t) s;
                    back_point[t][ns] = (uint8_t) jb[k];
                }
            }
        }
        for (int s = 0; s < states; ++s) cost[s] = next_cost[s];
    }
    int cur = 0;
    for (int s = 1; s < states; ++s)
        if (cost[s] < cost[cur]) cur = s;
    const float best = cost[cur];
    for (int i = 0; i < 8; ++i) qs[i] = 0;
    for (int t = kTcqBlock - 1; t >= 0; --t) {
        const int j = back_point[t][cur];
        const int code = j >> 1;              // position inside the union: j = 2*code + z0
        qs[t >> 2] |= (uint8_t) (code << (2 * (t & 3)));
        cur = back_state[t][cur];
    }
    return best;
}

/// Writer for one token's block: scale = f16(RMS of x), codes from the Viterbi path over x / scale.
STRATA_HD inline void tcq_quantize_block(int states, const float* x, block_tcq2_0* out) {
    float ss = 0.0f;
    for (int t = 0; t < kTcqBlock; ++t) ss += x[t] * x[t];
    const uint16_t d = f16_from_f32(sqrtf(ss * (1.0f / kTcqBlock)));
    const float s = f32_from_f16(d);
    float u[kTcqBlock];
    for (int t = 0; t < kTcqBlock; ++t) u[t] = s > 0.0f ? x[t] / s : 0.0f;
    tcq_encode_block(states, u, out->qs);
    out->d = d;
}

}  // namespace strata::kernels
