// src/kernels/cuda/fused_gr.cu - see include/strata/kernels/fused_gr.hpp.
#include "strata/core/emulate.hpp"
#include "strata/kernels/q8_1_finite.hpp"   // #606: q8_1_ds
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "s26_tsum.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace strata::kernels {
namespace {

constexpr int N = 2560;         // n_embd
constexpr int HC = 4;           // streams
constexpr int D = N * HC;       // 10240
constexpr int LR = 320;         // hc_lr
// the latency-hidden norm/up (gr_norm_fast_kernel / gr_up_fast_kernel, STRATA_GR_FAST): gfx906's, built for CUDA too
#if defined(STRATA_HIP_GFX906) || !defined(__HIPCC__)
#define STRATA_GR_FAST_BUILD 1
#else
#define STRATA_GR_FAST_BUILD 0
#endif
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int DOWN_BLOCKS = LR / WARPS;          // 40 blocks of 8 rows; one more for the inject rows
constexpr int UP_COLS = 32;                      // columns d per `up` block (x 4 streams = 128 rows)
constexpr int UP_BLOCKS = N / UP_COLS;           // 80

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

// 8 bf16 packed in a uint4, unpacked to two float4 once when reused across tokens.
struct Bf16x8 { float4 a, b; };
__device__ __forceinline__ Bf16x8 unpack8(const uint4 w) {
    return {
        make_float4(__uint_as_float(w.x << 16), __uint_as_float(w.x & 0xffff0000u),
                    __uint_as_float(w.y << 16), __uint_as_float(w.y & 0xffff0000u)),
        make_float4(__uint_as_float(w.z << 16), __uint_as_float(w.z & 0xffff0000u),
                    __uint_as_float(w.w << 16), __uint_as_float(w.w & 0xffff0000u))
    };
}
__device__ __forceinline__ float dot8u(const Bf16x8& w, const float4 x0, const float4 x1) {
    float acc = 0.0f;
    acc = fmaf(w.a.x, x0.x, acc);
    acc = fmaf(w.a.y, x0.y, acc);
    acc = fmaf(w.a.z, x0.z, acc);
    acc = fmaf(w.a.w, x0.w, acc);
    acc = fmaf(w.b.x, x1.x, acc);
    acc = fmaf(w.b.y, x1.y, acc);
    acc = fmaf(w.b.z, x1.z, acc);
    acc = fmaf(w.b.w, x1.w, acc);
    return acc;
}
__device__ __forceinline__ float dot8u_ptr(const Bf16x8& w, const float* x) {
    const float4* x4 = reinterpret_cast<const float4*>(x);
    return dot8u(w, x4[0], x4[1]);
}

// 8 bf16 packed in a uint4 against 8 16-byte-aligned floats (same 8-FMA order).
__device__ __forceinline__ float dot8(const uint4 w, const float* x) {
    return dot8u_ptr(unpack8(w), x);
}

__global__ void __launch_bounds__(THREADS) gr_down_kernel(FusedGrArgs a) {
    __shared__ __align__(16) float xn[D];
    __shared__ float part[WARPS][HC];
    __shared__ float s_rs[HC];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    // 1. R' * w_norm into shared memory, and the per-stream sums of squares of R'.
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = t * 4; i < D; i += THREADS * 4) {
        const int c = i / N, d = i - c * N;
        float4 r = *reinterpret_cast<const float4*>(a.R + i);
        if (a.apply) {
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r.x = fmaf(b.x, gw[c], r.x); r.y = fmaf(b.y, gw[c], r.y);
            r.z = fmaf(b.z, gw[c], r.z); r.w = fmaf(b.w, gw[c], r.w);
        }
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + i);
        float sq = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        *reinterpret_cast<float4*>(xn + i) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    __syncthreads();
    if (t < HC) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w][t];
        s_rs[t] = rsqrtf(s / (float) N + a.eps);
        if (blockIdx.x == 0) a.rs[t] = s_rs[t];
    }
    __syncthreads();
    for (int i = t; i < D; i += THREADS) xn[i] *= s_rs[i / N];
    __syncthreads();
    // 2. one warp per output row: 10240 bf16 = 1280 chunks of 8, 40 per lane.
    const bool inject_block = blockIdx.x == DOWN_BLOCKS;
    const int row = inject_block ? warp : blockIdx.x * WARPS + warp;
    if (inject_block && (a.w_inject == nullptr || warp >= HC)) return;
    const uint16_t* wrow = (inject_block ? a.w_inject : a.w_down) + (size_t) row * D;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
    float acc = 0.0f;
#pragma unroll 4
    for (int j = lane; j < D / 8; j += 32) acc += dot8(__ldg(w4 + j), xn + j * 8);
    acc = warp_sum(acc);
    if (lane != 0) return;
    if (inject_block) {
        a.inject_out[row] = acc;
    } else {
        const float x = acc / (float) HC;
        a.lo[row] = x / (1.0f + __expf(-x));
    }
}

__global__ void __launch_bounds__(THREADS) gr_up_kernel(FusedGrArgs a) {
    __shared__ __align__(16) float lo[LR];
    __shared__ float g[HC][UP_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int d0 = blockIdx.x * UP_COLS;
    for (int k = t; k < LR; k += THREADS) lo[k] = a.lo[k];
    __syncthreads();
    // 128 rows (4 streams x 32 columns), 16 per warp: 320 bf16 = 40 chunks of 8.
    for (int r = warp; r < HC * UP_COLS; r += WARPS) {
        const int c = r / UP_COLS, dd = r - c * UP_COLS, i = c * N + d0 + dd;
        const uint4* w4 = reinterpret_cast<const uint4*>(a.w_up + (size_t) i * LR);
        float acc = dot8(__ldg(w4 + lane), lo + lane * 8);
        if (lane < LR / 8 - 32) acc += dot8(__ldg(w4 + 32 + lane), lo + (32 + lane) * 8);
        acc = warp_sum(acc);
        if (lane == 0) {
            float rv = a.R[i];
            if (a.apply) {
                rv = fmaf(a.bo_prev[d0 + dd], 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC), rv);
                a.R_out[i] = rv;                       // this block owns column d0+dd of every stream
            }
            const float x = rv * a.w_norm[i] * a.rs[c];
            g[c][dd] = x * sigmoidf_(acc);
        }
    }
    __syncthreads();
    if (t < UP_COLS) {
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[c][t];
        a.mixed[d0 + t] = s / (float) HC;
    }
}

// ================================ plan v0.3 P6: T tokens, one weight read ================================
struct GrMulti {
    FusedGrArgs a[kFusedGrMaxT];
    float* xn;
    int T;
};


// S26 STRATA_QFUSE=1: the q8_1 image of `mixed`, written by the up kernels below. A q8_1 block is 32 columns and an
// up block owns UPM_COLS = 16, so the second of the two blocks that own a 32-column group (a per-group counter,
// incremented after the block's writes are fenced, reset by that block for the next launch) reads the 32 values back
// and quantizes them with native_quantize_q8_1_kernel's quantizer: one warp per token, the same XOR-tree max and sum,
// d = amax / 127, roundf(x / d), ds = (d, sum) - the bytes the separate launch writes.
struct GrQ81 { half2 ds; int8_t qs[32]; };
static_assert(2 * 16 == 32, "two up blocks per q8_1 group");
__device__ __forceinline__ void gr_q8_tail(const GrMulti& m, int d0) {
    __shared__ int last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned* c = m.a[0].q8_cnt + d0 / 32;
        last = atomicAdd(c, 1u) == 1u;
        if (last) atomicExch(c, 0u);
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp >= m.T) return;
    const int c0 = (d0 / 32) * 32;
    const float xi = *reinterpret_cast<const volatile float*>(m.a[warp].mixed + c0 + lane);
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const float d = q8_1_finite(amax / 127.0f);   // #606: as native_quantize_q8_1_kernel - finite blocks bit for bit
    const int8_t q = q8_1_quant(xi, d, amax);
    GrQ81* y = reinterpret_cast<GrQ81*>(m.a[warp].q8_mixed) + c0 / 32;
    y->qs[lane] = q;
    if (lane == 0) y->ds = q8_1_ds(d, sum);   // #606: clamped scale/sum - an unclamped pair NaN-poisons the dot path
}
// Step 1 of `gr_down_kernel`, one block per token, same threads and reduction order: rs[t] and xn[t] to global.
__global__ void __launch_bounds__(THREADS) gr_norm_multi_kernel(GrMulti m) {
    __shared__ float part[WARPS][HC];
    __shared__ float s_rs[HC];
    const FusedGrArgs& a = m.a[blockIdx.x];
    float* xn = m.xn + (size_t) blockIdx.x * D;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = t * 4; i < D; i += THREADS * 4) {
        const int c = i / N, d = i - c * N;
        float4 r = *reinterpret_cast<const float4*>(a.R + i);
        if (a.apply) {
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r.x = fmaf(b.x, gw[c], r.x); r.y = fmaf(b.y, gw[c], r.y);
            r.z = fmaf(b.z, gw[c], r.z); r.w = fmaf(b.w, gw[c], r.w);
        }
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + i);
        float sq = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        *reinterpret_cast<float4*>(xn + i) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    __syncthreads();
    if (t < HC) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w][t];
        s_rs[t] = rsqrtf(s / (float) N + a.eps);
        a.rs[t] = s_rs[t];
    }
    __syncthreads();
    for (int i = t; i < D; i += THREADS) xn[i] *= s_rs[i / N];
}

// Step 2 of `gr_down_kernel` for T tokens.  One warp per row (so each lane accumulates the same chunks in the
// same order as the single-token kernel); per tile the lane's weight chunks are loaded BEFORE the activation
// tile is staged, so the DRAM and L2 traffic are in flight together.
//
// TILEV = xn floats per token staged at a time.  The tile only changes the staging granularity: the lane's chunk
// order (lane + 32*q within the tile, tiles ascending) is strictly increasing for either value, so the results are
// bitwise identical to `gr_down_kernel` for both.  2560 stages 320 chunks of 8 per tile (10 per lane); cards whose
// opt-in below 8 * 2560 * 4 B slices the tokens - sm_75 (64 KiB) carries 6 tokens of it - run TILEV 1280 instead,
// which fits all eight tokens in one 40 KiB launch and stages 160 chunks of 8 (5 per lane, half the registers held
// for the weight prefetch); smaller tiles raise how many blocks share an SM (the 41-block grid), e.g. three blocks
// of four tokens instead of one on sm_75.
template <int TILEV, int MAX_T = kFusedGrMaxT>
__global__ void __launch_bounds__(THREADS) gr_down_multi_kernel(GrMulti m) {
    constexpr int TQ = TILEV / 8 / 32;      // uint4 weight chunks per lane per tile
    extern __shared__ __align__(16) float tile[];   // [T][TILEV]
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = m.T;
    const bool inject_block = blockIdx.x == DOWN_BLOCKS;
    const int row = inject_block ? warp : blockIdx.x * WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) (active ? row : 0) * D;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
    float acc[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) acc[k] = 0.0f;
    for (int base = 0; base < D; base += TILEV) {
        uint4 wv[TQ];
        if (active) {
#pragma unroll
            for (int q = 0; q < TQ; ++q) wv[q] = __ldg(w4 + base / 8 + lane + 32 * q);
        }
        __syncthreads();                                   // the previous tile is consumed
        const float4* src4 = reinterpret_cast<const float4*>(m.xn);
        float4* tile4 = reinterpret_cast<float4*>(tile);
        for (int i = t; i < T * (TILEV / 4); i += THREADS) {
            const int k = i / (TILEV / 4), off = i - k * (TILEV / 4);
            tile4[i] = src4[((size_t) k * D + base) / 4 + off];
        }
        __syncthreads();
        if (!active) continue;
#pragma unroll
        for (int q = 0; q < TQ; ++q) {
            const int j = lane + 32 * q;
#pragma unroll
            for (int k = 0; k < MAX_T; ++k)
                if (k < T) acc[k] += dot8(wv[q], tile + k * TILEV + j * 8);
        }
    }
    if (!active) return;
    float s[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) s[k] = k < T ? warp_sum(acc[k]) : 0.0f;
    // lane k writes token k (every lane holds every sum after the xor reduction)
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        if (k >= T || lane != k) continue;
        if (inject_block) {
            m.a[k].inject_out[row] = s[k];
        } else {
            const float x = s[k] / (float) HC;
            m.a[k].lo[row] = x / (1.0f + __expf(-x));
        }
    }
}

constexpr int UPM_COLS = 16;                      // columns per block (x 4 streams = 64 rows, 8 per warp)
constexpr int UPM_BLOCKS = N / UPM_COLS;          // 160

// `gr_up_kernel` for T tokens: each row of w_up read once; the T dots reduced by xor so every lane holds every
// sum, and lane k runs token k's epilogue - the T epilogues in parallel instead of one after another.
template <int MAX_T = kFusedGrMaxT, bool EXACT_T = false>
__global__ void __launch_bounds__(THREADS) gr_up_multi_kernel(GrMulti m) {
    __shared__ __align__(16) float lo[MAX_T][LR];
    __shared__ float g[MAX_T][HC][UPM_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    const int d0 = blockIdx.x * UPM_COLS;
    for (int i = t; i < T * LR; i += THREADS) lo[i / LR][i % LR] = m.a[i / LR].lo[i % LR];
    __syncthreads();
    for (int r = warp; r < HC * UPM_COLS; r += WARPS) {
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
        const Bf16x8 wa = unpack8(__ldg(w4 + lane));
        const Bf16x8 wb = unpack8(lane < LR / 8 - 32 ? __ldg(w4 + 32 + lane) : make_uint4(0, 0, 0, 0));
        // the epilogue inputs of this lane's token, fetched while the dots run
        float rv = 0.0f, wn = 0.0f, rsc = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
            rv = a.R[i];
            wn = a.w_norm[i];
            rsc = a.rs[c];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) {
            if (!EXACT_T && k >= T) break;
            float acc = dot8u_ptr(wa, lo[k] + lane * 8);
            if (lane < LR / 8 - 32) acc += dot8u_ptr(wb, lo[k] + (32 + lane) * 8);
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        if (lane < T) {
            if (apply) {
                rv = fmaf(bo, 2.0f * sigmoidf_(ip / (float) HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsc;
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[k][c][col];
        m.a[k].mixed[d0 + col] = s / (float) HC;
    }
    if (m.a[0].q8_mixed != nullptr) gr_q8_tail(m, d0);   // S26 STRATA_QFUSE
}

// ==================== S27 STRATA_HC_PACK: the packed weight read (hc-rdna4-proposals-20261009 3.2) ============
// The packed arm changes only how the weight BYTES arrive: the 13 bytes per 8 values (8 lows, 4 nibbles, 1 flag
// byte) are composed back to the identical BF16 bits in registers, and the escapes - values whose exponent left
// [111,126] - are overwritten with their exact bits from the side list.  Everything else (CTA mapping, tiles,
// lane order, the eight fmaf in dot8u's order, the xor reduction, the LDS contract) is the kernel it copies.
//
// gfx1201 ISA: composition uses 32-bit ops only (no v_add_u16 / v_sub_u16 / v_add_u8 / v_min_u8 / v_add3_u16 /
// v_lshl_add_u16).  The three fields are disjoint - sign bit 15, exponent bits 7..14, mantissa bits 0..6 - and
// the largest exponent 126 << 7 = 0x3F00 never carries into bit 15, so the adds below are the ORs.
__device__ __forceinline__ uint32_t hc_pack_word(uint32_t lo2, uint32_t nb) {
    const uint32_t a0 = lo2 & 0xFFu, a1 = (lo2 >> 8) & 0xFFu;
    const uint32_t v0 = ((a0 & 0x80u) << 8) + ((uint32_t(HC_PACK_EXP_BASE) + (nb & 0xFu)) << 7) + (a0 & 0x7Fu);
    const uint32_t v1 = ((a1 & 0x80u) << 8) + ((uint32_t(HC_PACK_EXP_BASE) + ((nb >> 4) & 0xFu)) << 7) + (a1 & 0x7Fu);
    return v0 + (v1 << 16);
}
// One chunk = 8 values: lows lo.x = bytes 0..3, lo.y = bytes 4..7; nibble byte b carries values 2b (low nibble)
// and 2b+1.  `fl` is the chunk's flag byte (bit i: value i is an escape) - the pack has no in-band marker,
// because the window uses all 16 nibble values and the low byte all 8 bits.
__device__ __forceinline__ uint4 hc_compose8(uint2 lo, uint32_t nb, uint32_t fl, uint32_t& emask) {
    uint4 w;
    w.x = hc_pack_word(lo.x & 0xFFFFu, nb & 0xFFu);
    w.y = hc_pack_word((lo.x >> 16) & 0xFFFFu, (nb >> 8) & 0xFFu);
    w.z = hc_pack_word(lo.y & 0xFFFFu, (nb >> 16) & 0xFFu);
    w.w = hc_pack_word((lo.y >> 16) & 0xFFFFu, (nb >> 24) & 0xFFu);
    emask = fl;
    return w;
}
// Overwrite the escape values of a composed chunk, ascending value order, with the exact bits from the side
// list starting at `slot`.  0.041% of values, so this loop almost never runs its body.
__device__ __forceinline__ void hc_apply_escapes(uint4& w, uint32_t emask, const uint16_t* esc_value, uint32_t slot) {
#pragma unroll
    for (int i = 0; i < HC_PACK_VALUES; ++i) {
        if ((emask & (1u << i)) == 0u) continue;
        const uint32_t sh = (i & 1) ? 16u : 0u;
        const uint32_t mask = 0xFFFFu << sh;
        const uint32_t bits = uint32_t(__ldg(esc_value + slot)) << sh;
        // Update the vector component explicitly: indexing from &w.x into sibling struct members is not a
        // standard C++ array operation, even though the CUDA vector layout is contiguous.
        switch (i >> 1) {
            case 0: w.x = (w.x & ~mask) | bits; break;
            case 1: w.y = (w.y & ~mask) | bits; break;
            case 2: w.z = (w.z & ~mask) | bits; break;
            default: w.w = (w.w & ~mask) | bits; break;
        }
        ++slot;
    }
}
// Escapes this warp's lanes before this one found in this group: a 5-step shuffle scan of the per-lane counts
// (a ballot would count lanes, not escapes - a lane's chunk can hold up to 8 of them).  Warp-converged: every
// lane of the warp calls it, with cnt 0 where it read nothing.
__device__ __forceinline__ uint32_t hc_warp_before(uint32_t cnt) {
    uint32_t run = cnt, before = 0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const uint32_t t = __shfl_up_sync(0xffffffffu, run, o);
        if ((threadIdx.x & 31u) >= uint32_t(o)) {
            before += t;
            run += t;
        }
    }
    return before;
}

// gr_up_multi_kernel with the packed weight arrival: same CTA mapping (UPM_BLOCKS, 16 columns, warp w walks
// rows w, w+WARPS, ... of its 64), same lo[] tiles, same warp_sum, same lane == k epilogue.  A warp's two
// chunks per row are the group's lane-th chunk (group 0) and the (32 + lane)-th (group 1, lanes < LR/8-32);
// the lanes that read nothing in the plain read compose to zeros here too, so the dots are the same zeros.
template <int MAX_T = kFusedGrMaxT, bool EXACT_T = false>
__global__ void __launch_bounds__(THREADS) gr_up_multi_packed_kernel(GrMulti m) {
    __shared__ __align__(16) float lo[MAX_T][LR];
    __shared__ float g[MAX_T][HC][UPM_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    const int d0 = blockIdx.x * UPM_COLS;
    for (int i = t; i < T * LR; i += THREADS) lo[i / LR][i % LR] = m.a[i / LR].lo[i % LR];
    __syncthreads();
    const HcPackedWeights& pk = m.a[0].pack_up;
    if (pk.lows == nullptr) return;
    const uint2* lows = reinterpret_cast<const uint2*>(pk.lows);
    const uint32_t* nib = reinterpret_cast<const uint32_t*>(pk.nibbles);
    const uint8_t* flag = pk.flags;
    const uint32_t chunks_per_row = LR / HC_PACK_VALUES;      // 40: whole chunks, LR = 320
    for (int r = warp; r < HC * UPM_COLS; r += WARPS) {
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const size_t crow = (size_t) i * chunks_per_row;
        const uint32_t wb = __ldg(pk.esc_warp_row +
                                  (blockIdx.x * pk.warps_per_block + warp) * pk.rows_per_warp + (r - warp) / WARPS);
        uint4 wa4, wb4;
        uint32_t ea = 0, eb = 0;
        const uint2 la = __ldg(lows + crow + lane);
        const uint32_t na = __ldg(nib + crow + lane);
        ea = __ldg(flag + crow + lane);
        wa4 = hc_compose8(la, na, ea, ea);
        const bool second = lane < LR / 8 - 32;
        uint2 lb = make_uint2(0, 0);
        uint32_t nb2 = 0, fb = 0;
        if (second) {
            lb = __ldg(lows + crow + 32 + lane);
            nb2 = __ldg(nib + crow + 32 + lane);
            fb = __ldg(flag + crow + 32 + lane);
        }
        eb = fb;
        wb4 = second ? hc_compose8(lb, nb2, fb, eb) : make_uint4(0, 0, 0, 0);
        hc_apply_escapes(wa4, ea, pk.esc_value,
                         wb + __ldg(pk.esc_row_group + i * pk.groups_per_row + 0) + hc_warp_before(__popc(ea)));
        hc_apply_escapes(wb4, eb, pk.esc_value,
                         wb + __ldg(pk.esc_row_group + i * pk.groups_per_row + 1) + hc_warp_before(__popc(eb)));
        const Bf16x8 wa = unpack8(wa4);
        const Bf16x8 wb8 = unpack8(wb4);
        // the epilogue inputs of this lane's token, fetched while the dots run
        float rv = 0.0f, wn = 0.0f, rsc = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
            rv = a.R[i];
            wn = a.w_norm[i];
            rsc = a.rs[c];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) {
            if (!EXACT_T && k >= T) break;
            float acc = dot8u_ptr(wa, lo[k] + lane * 8);
            if (lane < LR / 8 - 32) acc += dot8u_ptr(wb8, lo[k] + (32 + lane) * 8);
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        if (lane < T) {
            if (apply) {
                rv = fmaf(bo, 2.0f * sigmoidf_(ip / (float) HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsc;
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[k][c][col];
        m.a[k].mixed[d0 + col] = s / (float) HC;
    }
    if (m.a[0].q8_mixed != nullptr) gr_q8_tail(m, d0);   // S26 STRATA_QFUSE
}


// ================================ hc read v3 (opt-in: STRATA_GR_V3=1) - two kernels, stream-split ================
// The norm kernel runs on only T blocks (~16 us of pure latency per call) and `down` on 41 blocks (~280 GB/s).
// v3: `down` is split over (row group, stream[, column half]) = 164 or 328 blocks; each block stages its slice of
// R' * w_norm for the T tokens, reduces that slice's sum of squares itself, and writes UNSCALED partial dots.  The
// rms scale is per stream, so  w_down . xn = sum_c rs[c] * (w_down[:, c] . (R'[c] * w_norm[c]))  - `up` applies it
// in its prologue (lo, inject, rs).  Same maths, ANOTHER SUMMATION ORDER: not bitwise the default kernels, hence
// opt-in.  Dynamic shared memory is T * (N / S) floats: S (1 or 2 column halves) is the smallest that fits the
// card's opt-in limit at kFusedGrMaxT tokens (Ampere 99 KB: S = 1; Turing / HIP 64 KB: S = 2); a card where
// neither fits keeps the default kernels.
constexpr int PR = LR + HC;                        // partial rows per (token, stream): 320 down + 4 inject
constexpr int TQ3 = N / 8 / 32;                    // uint4 weight chunks per lane in one stream's slice (10)

template <int S, int MAX_T = kFusedGrMaxT, bool EXACT_T = false>
__global__ void __launch_bounds__(THREADS) gr_down_v3_kernel(GrMulti m, float* __restrict__ part, float* __restrict__ ssg) {
    extern __shared__ __align__(16) float xs[];    // [T][N / S]
    constexpr int R2 = 1;                          // down rows per warp
    __shared__ float red[WARPS][MAX_T];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    // S = 2: each stream's 2560 columns in two halves (blockIdx.y = stream * S + half): twice the blocks
    constexpr int SL = N / S, TQS = SL / 8 / 32;
    const int rg = blockIdx.x, c = blockIdx.y / S, h = blockIdx.y - (blockIdx.y / S) * S;
    constexpr int NDB = LR / (WARPS * R2);          // down row blocks per stream; block NDB = the inject rows
    const bool inject_block = rg == NDB;
    // warp w owns rows row0 + w * R2 + r (r < R2); the inject block: warps 0-3, one row each
    const int row0 = inject_block ? warp : (rg * WARPS + warp) * R2;
    const int nrows = inject_block ? ((m.a[0].w_inject != nullptr && warp < HC) ? 1 : 0) : R2;
    const bool active = nrows > 0;
    const uint16_t* wbase = inject_block ? m.a[0].w_inject : m.a[0].w_down;
    float ssp[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        ssp[k] = 0.0f;
        if (!EXACT_T && k >= T) continue;
        const FusedGrArgs& a = m.a[k];
        const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
        const float4* R4 = reinterpret_cast<const float4*>(a.R + (size_t) c * N + (size_t) h * SL);
        const float4* G4 = reinterpret_cast<const float4*>(a.w_norm + (size_t) c * N + (size_t) h * SL);
        const float4* B4 = reinterpret_cast<const float4*>(a.bo_prev + (size_t) h * SL);
        float4* X4 = reinterpret_cast<float4*>(xs + (size_t) k * SL);
        for (int i = t; i < SL / 4; i += THREADS) {
            float4 r = R4[i];
            if (a.apply) {
                const float4 b = B4[i];
                r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y);
                r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
            }
            const float4 g = G4[i];
            ssp[k] += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
            X4[i] = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
        }
    }
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        if (!EXACT_T && k >= T) break;
        const float v = warp_sum(ssp[k]);
        if (lane == 0) red[warp][k] = v;
    }
    __syncthreads();
    if (rg == 0 && t < T) {
        float sum = 0.0f;
        for (int w = 0; w < WARPS; ++w) sum += red[w][t];
        ssg[(t * HC + c) * S + h] = sum;
    }
    if (!active) return;
#pragma unroll
    for (int r = 0; r < R2; ++r) {
        if (r >= nrows) break;
        const uint4* w4 = reinterpret_cast<const uint4*>(wbase + (size_t) (row0 + r) * D + (size_t) c * N + (size_t) h * SL);
        float acc[MAX_T];
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) acc[k] = 0.0f;
#pragma unroll
        for (int q = 0; q < TQS; ++q) {
            const int j = lane + 32 * q;
            const Bf16x8 wvq = unpack8(__ldg(w4 + j));
#pragma unroll
            for (int k = 0; k < MAX_T; ++k)
                if (EXACT_T || k < T) acc[k] += dot8u_ptr(wvq, xs + (size_t) k * SL + j * 8);
        }
        const int prow = inject_block ? LR + warp : row0 + r;
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) {
            if (!EXACT_T && k >= T) break;
            const float v = warp_sum(acc[k]);
            if (lane == 0) part[(((size_t) k * HC + c) * S + h) * PR + prow] = v;
        }
    }
}

template <int S, int MAX_T = kFusedGrMaxT, bool EXACT_T = false>
__global__ void __launch_bounds__(THREADS) gr_up_v3_kernel(GrMulti m, const float* __restrict__ part,
                                                           const float* __restrict__ ssg) {
    __shared__ __align__(16) float lo[MAX_T][LR];
    __shared__ float rsS[MAX_T][HC];
    __shared__ float g[MAX_T][HC][UPM_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    const int d0 = blockIdx.x * UPM_COLS;
    constexpr int RPW = HC * UPM_COLS / WARPS;     // 8 rows per warp
    if (t < T * HC) {
        const int k = t / HC, c = t - k * HC;
        float ss = 0.0f;
#pragma unroll
        for (int h = 0; h < S; ++h) ss += ssg[t * S + h];
        const float r = rsqrtf(ss / (float) N + m.a[k].eps);
        rsS[k][c] = r;
        if (blockIdx.x == 0) m.a[k].rs[c] = r;
    }
    __syncthreads();
    for (int i = t; i < T * LR; i += THREADS) {
        const int k = i / LR, r = i - k * LR;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            float p = 0.0f;
#pragma unroll
            for (int h = 0; h < S; ++h) p += part[(((size_t) k * HC + c) * S + h) * PR + r];
            sum = fmaf(rsS[k][c], p, sum);
        }
        const float x = sum / (float) HC;
        lo[k][r] = x / (1.0f + __expf(-x));
    }
    if (blockIdx.x == 0 && t < T * HC) {
        const int k = t / HC, cc = t - k * HC;
        if (m.a[k].w_inject != nullptr) {
            float sum = 0.0f;
#pragma unroll
            for (int c = 0; c < HC; ++c) {
                float p = 0.0f;
#pragma unroll
                for (int h = 0; h < S; ++h) p += part[(((size_t) k * HC + c) * S + h) * PR + LR + cc];
                sum = fmaf(rsS[k][c], p, sum);
            }
            m.a[k].inject_out[cc] = sum;
        }
    }
    __syncthreads();
#pragma unroll
    for (int q = 0; q < RPW; ++q) {
        const int r = warp + q * WARPS;
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
        const Bf16x8 wa = unpack8(__ldg(w4 + lane));
        const Bf16x8 wb = unpack8(lane < LR / 8 - 32 ? __ldg(w4 + 32 + lane) : make_uint4(0, 0, 0, 0));
        float rv = 0.0f, wn = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
            rv = a.R[i];
            wn = a.w_norm[i];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) {
            if (!EXACT_T && k >= T) break;
            float acc = dot8u_ptr(wa, lo[k] + lane * 8);
            if (lane < LR / 8 - 32) acc += dot8u_ptr(wb, lo[k] + (32 + lane) * 8);
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        if (lane < T) {
            if (apply) {
                rv = fmaf(bo, 2.0f * sigmoidf_(ip / (float) HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsS[lane][c];
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) sum += g[k][c][col];
        m.a[k].mixed[d0 + col] = sum / (float) HC;
    }
}
// ================================ #315: split / staged - the same arithmetic, split differently ==================
// The multi read's variants beside the plain read (0.1.31's default; not main's opt-in STRATA_GR_V3 read, which sums
// in another order).  Every variant computes each output with exactly the plain read's operations in its order (so
// bitwise the plain read and the single-token read); only who does what, and when, changes.  `fused_gr_check`
// compares them on the card before a verify window uses them (STRATA_HC_SPLIT below).
//  - the norm (split and staged): one block per token AND stream instead of one per token.  Each thread visits
//    exactly the elements, in the order, it visited for that stream in the plain read, so every sum of squares and rs
//    is the same; it stores R' * w_norm and scales it in place, with the plain read's two roundings.
//    Before this scratch reuse, the split norm measured 0.23 ms per 4-token round (96 launches) on an RTX 4080
//    SUPER, compared with 0.58 ms for the plain read's norm.
//  - the down projection, split: the plain read's kernel.
//  - the down projection, staged: the plain read's 8 rows per block and lane order, but the activations arrive by
//    cp.async in half-stream tiles, two in flight, so no thread waits on a chain of loads, and they sit in shared
//    memory as two planes (floats 0-3 and 4-7 of every 8), so a warp's reads hit no bank twice.  The next tile's
//    weights are loaded while the current one is used.  A tile is 160 = 5 x 32 chunks of 8, so a lane still
//    accumulates its chunks lane + 32 q in ascending order, as in the plain read with either tile.  Before sm_80
//    (and on HIP) the staging is a plain copy: the same bits, only not asynchronous.
constexpr int kHcPlain = 1, kHcSplit = 2, kHcStaged = 3, kHcSmallCta = 4, kHcReuseTwoRows = 5,
              kHcRegisterPipe = 6, kHcRegisterHalf = 7;
// kHcPacked = 8 is the packed arm of kHcStaged (below); the row split takes the next slot.
// STRATA_HC_SPLIT=9 (env value 9) selects kHcLdsAccum: the staged down kernel with the per-token accumulators in
// dynamic LDS.  The env value is one digit and 8 already names the row split (constant 9), so the new variant is
// the next constant, 10, and env 9 maps to it.
constexpr int kHcRowSplit = 9;
constexpr int kHcLdsAccum = 10;
// STRATA_HC_SPLIT=10 (env value 10, two characters - see env_variant) selects kHcRegisterPipeHalf: the
// register-tuple pipeline (6) with BOTH of its register arrays split in halves, the schedule the half kernel (7)
// uses for the activation tuple applied to the weight prefetch as well.  The env parser is one digit and every
// one-digit value is taken, so this is the first two-character value; it is matched before the one-digit '1'.
constexpr int kHcRegisterPipeHalf = 11;
// STRATA_HC_PACK=1 (S27, hc-rdna4-proposals-20261009 3.2): the STAGED read with the weight bytes arriving packed
// (include/strata/kernels/hc_pack.hpp).  Not a variant of its own - it is kHcStaged with the packed arm latched
// (the latch lives below launch_multi, so the predicate is declared here).
bool pack_on();
// (g_pack), so the staged grid, tiles, lane order, dot order and LDS contract are the staged read's unchanged.
constexpr int kHcPacked = 8;
constexpr int HC_SMALL_CTA_THREADS = 128;
constexpr int HC_SMALL_CTA_WARPS = HC_SMALL_CTA_THREADS / 32;
constexpr int HC_SMALL_CTA_BLOCKS = LR / HC_SMALL_CTA_WARPS;
constexpr int HC_REUSE_THREADS = 128;
constexpr int HC_REUSE_WARPS = HC_REUSE_THREADS / 32;
constexpr int HC_REUSE_ROWS_PER_WARP = 2;
constexpr int HC_REUSE_ROWS_PER_BLOCK = HC_REUSE_WARPS * HC_REUSE_ROWS_PER_WARP;
constexpr int HC_REUSE_BLOCKS = LR / HC_REUSE_ROWS_PER_BLOCK;
static_assert(LR % HC_REUSE_ROWS_PER_BLOCK == 0, "reuse CTA rows tile the down projection");
static_assert(HC_REUSE_THREADS % 32 == 0, "reuse CTA has whole wave32 warps");
static_assert(HC_REUSE_BLOCKS == DOWN_BLOCKS, "reuse CTA keeps the original row-CTA count");
constexpr int H_TILE = 1280;                          // staged tile: half a stream = 160 chunks of 8, 5 per lane
constexpr int HQ = H_TILE / 8 / 32;
constexpr int N_HTILES = D / H_TILE;                  // 8
static_assert(HC_SMALL_CTA_THREADS % 32 == 0 && LR % HC_SMALL_CTA_WARPS == 0, "whole warp rows for CTA variant");
static_assert(N % H_TILE == 0, "a staged tile never straddles two streams");
static_assert(H_TILE % 256 == 0, "a staged tile holds whole rounds of 32 chunks: the plain read's lane order");

__global__ void __launch_bounds__(THREADS) gr_norm_split_kernel(GrMulti m) {
    __shared__ float part[WARPS];
    __shared__ float s_rs;
    const FusedGrArgs& a = m.a[blockIdx.x];
    const int c = blockIdx.y;
    float* xn = m.xn + (size_t) blockIdx.x * D;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    float ss = 0.0f;
    for (int i = t * 4; i < D; i += THREADS * 4) {
        if (i / N != c) continue;                       // another stream's element: another block's
        const int d = i - c * N;
        float4 r = *reinterpret_cast<const float4*>(a.R + i);
        if (a.apply) {
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y);
            r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
        }
        const float sq = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
        ss += sq;
        // Retain the unscaled product in the existing scratch instead of reloading R
        // and recomputing the pending write after the reduction.
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + i);
        *reinterpret_cast<float4*>(xn + i) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
    }
    const float v = warp_sum(ss);
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (t == 0) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w];
        s_rs = rsqrtf(s / (float) N + a.eps);
        a.rs[c] = s_rs;
    }
    __syncthreads();
    const float rs = s_rs;
    for (int d = t; d < N; d += THREADS) xn[c * N + d] *= rs;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIPCC__)
#define STRATA_GR_CP_ASYNC 1
#endif
// CUDA sm_80+ can enqueue a global-to-shared copy; HIP/gfx12 performs the ordinary vector load/store here.
// These helpers deliberately describe the CUDA group operations rather than implying HIP has asynchronous copies.
__device__ __forceinline__ void gr_copy_gmem_to_smem16(void* smem, const void* gmem) {
#if defined(STRATA_GR_CP_ASYNC)
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa), "l"(gmem) : "memory");
#else
    *reinterpret_cast<float4*>(smem) = *reinterpret_cast<const float4*>(gmem);
#endif
}
// CUDA cp.async queue operations are deliberately named as CUDA-specific.
__device__ __forceinline__ void gr_cuda_async_group_commit() {
#if defined(STRATA_GR_CP_ASYNC)
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void gr_cuda_async_group_wait1() {
#if defined(STRATA_GR_CP_ASYNC)
    asm volatile("cp.async.wait_group 1;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void gr_cuda_async_group_wait0() {
#if defined(STRATA_GR_CP_ASYNC)
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#endif
}

// The staged read uses the original CTA mapping. The gfx12 copy is synchronous; do not infer overlap from CUDA helper names.
__device__ __forceinline__ void stage_htile(const GrMulti& m, int T, int h, float4* buf,
                                            int t, int nthreads) {
    for (int i = t; i < T * (H_TILE / 4); i += nthreads) {
        const int k = i / (H_TILE / 4), s4 = i - k * (H_TILE / 4);
        const float* src = m.xn + (size_t) k * D + (size_t) h * H_TILE + (size_t) s4 * 4;
        gr_copy_gmem_to_smem16(buf + (size_t) k * (H_TILE / 4) +
                               (s4 & 1) * (H_TILE / 8) + (s4 >> 1), src);
    }
}

// No direct global-to-LDS DMA path: the on-device gfx1201 probe reports
// __builtin_amdgcn_global_load_async_to_lds_b128 not invocable, so the staged tiles reach LDS by way of the
// per-thread copy in `stage_htile` above and nothing else.
//
// A software pipeline for the staged read - tile h + 2 read into a thread's own slots before tile h's arithmetic,
// with no barrier in between, so the load is in flight during it - was tried on gfx1201 and is NOT used.  Its slot
// array, intended as per-thread VGPR slots, measured private_segment_fixed_size 48 bytes at T=1 through 176 bytes
// at T=8 on gfx1201 (ROCm 7.17), with scratch_store_b128 between the global load and scratch_load_b128 before the
// LDS store: per-thread private scratch, not VGPR slots, so the tile travels through memory - the copy this
// pipeline existed to avoid.  Written as scalars (not the float4 memcpy) it measured the same.  No performance was
// measured, and the variant is not dispatched; see the STRATA_HC_SPLIT list below.

// The register tuple the opt-in pipelined variants (STRATA_HC_SPLIT=6 and 7) carry the next tile in.  A
// `HcChain<N>` is N+1 float4 in the thread's own registers, every slot reached at a compile-time index: the
// first attempt's runtime-indexed slot array lowered to per-thread private scratch on gfx1201 (see the note
// above), while this compile-time chain measured vgpr_spill 0 and private_segment_fixed_size 0 on gfx1201
// (ROCm 7.17) for every T the dispatch uses.  Slot d of a thread's tuple takes float4 index
// t + d*THREADS of the tile; a slot past the runtime total is zeroed and never stored.
template <int I> struct HcChain { float4 head; HcChain<I - 1> tail; };
template <> struct HcChain<0> { float4 head; };

template <int Dd, int NN>
__device__ __forceinline__ void hc_load_all(HcChain<NN>& c, const GrMulti& m, int h, int t, int total) {
    const int idx = t + Dd * THREADS;
    if (idx < total) {
        const int k = idx / (H_TILE / 4), s4 = idx - k * (H_TILE / 4);
        c.head = *reinterpret_cast<const float4*>(m.xn + (size_t) k * D + (size_t) h * H_TILE + (size_t) s4 * 4);
    } else {
        c.head = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    if constexpr (NN > 0) hc_load_all<Dd + 1, NN - 1>(c.tail, m, h, t, total);
}

// Store the tuple into the staged tile layout - the same destination `stage_htile` writes, the same guarded
// slots, and only in-bounds values.
template <int Dd, int NN>
__device__ __forceinline__ void hc_scatter_all(const HcChain<NN>& c, float4* buf, int t, int total) {
    const int idx = t + Dd * THREADS;
    if (idx < total) {
        const int k = idx / (H_TILE / 4), s4 = idx - k * (H_TILE / 4);
        buf[(size_t) k * (H_TILE / 4) + (s4 & 1) * (H_TILE / 8) + (s4 >> 1)] = c.head;
    }
    if constexpr (NN > 0) hc_scatter_all<Dd + 1, NN - 1>(c.tail, buf, t, total);
}

// `dot8` with its 8 activations as two float4: the same eight fmaf in the same order.
__device__ __forceinline__ float dot8v(const uint4 w, const float4 x0, const float4 x1) {
    float acc = 0.0f;
    acc = fmaf(__uint_as_float(w.x << 16), x0.x, acc);
    acc = fmaf(__uint_as_float(w.x & 0xffff0000u), x0.y, acc);
    acc = fmaf(__uint_as_float(w.y << 16), x0.z, acc);
    acc = fmaf(__uint_as_float(w.y & 0xffff0000u), x0.w, acc);
    acc = fmaf(__uint_as_float(w.z << 16), x1.x, acc);
    acc = fmaf(__uint_as_float(w.z & 0xffff0000u), x1.y, acc);
    acc = fmaf(__uint_as_float(w.w << 16), x1.z, acc);
    acc = fmaf(__uint_as_float(w.w & 0xffff0000u), x1.w, acc);
    return acc;
}

template <int MAX_T = kFusedGrMaxT, bool EXACT_T = false, int BLOCK_THREADS = THREADS>
__global__ void __launch_bounds__(BLOCK_THREADS) gr_down_staged_kernel(GrMulti m) {
    extern __shared__ __align__(16) float4 hbuf[];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    constexpr int CTA_WARPS = BLOCK_THREADS / 32;
    constexpr int CTA_BLOCKS = LR / CTA_WARPS;
    const int T = EXACT_T ? MAX_T : m.T;
    const bool inject_block = blockIdx.x == CTA_BLOCKS;
    const int row = inject_block ? warp : blockIdx.x * CTA_WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) (active ? row : 0) * D;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
    const size_t buf_f4 = (size_t) T * (H_TILE / 4);
    float acc[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) acc[k] = 0.0f;
    uint4 wv[HQ], wnext[HQ];
    if (active) {
#pragma unroll
        for (int q = 0; q < HQ; ++q) wv[q] = __ldg(w4 + lane + 32 * q);
    }
    stage_htile(m, T, 0, hbuf, t, BLOCK_THREADS);
    gr_cuda_async_group_commit();
    stage_htile(m, T, 1, hbuf + buf_f4, t, BLOCK_THREADS);
    gr_cuda_async_group_commit();
#pragma unroll 1
    for (int h = 0; h < N_HTILES; ++h) {
        if (active && h + 1 < N_HTILES) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) wnext[q] = __ldg(w4 + (h + 1) * (H_TILE / 8) + lane + 32 * q);
        }
        if (h + 1 < N_HTILES) gr_cuda_async_group_wait1();
        else gr_cuda_async_group_wait0();
        __syncthreads();
        const float4* cur = hbuf + (h & 1) * buf_f4;
        if (active) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) {
                const int j = lane + 32 * q;
                const Bf16x8 wvq = unpack8(wv[q]);
#pragma unroll
                for (int k = 0; k < MAX_T; ++k) {
                    if (EXACT_T || k < T) {
                        const float4* pk = cur + (size_t) k * (H_TILE / 4);
                        acc[k] += dot8u(wvq, pk[j], pk[H_TILE / 8 + j]);
                    }
                }
            }
        }
        __syncthreads();
        if (h + 2 < N_HTILES) stage_htile(m, T, h + 2, hbuf + (h & 1) * buf_f4, t, BLOCK_THREADS);
        gr_cuda_async_group_commit();
        if (h + 1 < N_HTILES) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) wv[q] = wnext[q];
        }
    }
    if (!active) return;
    float s[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) s[k] = (EXACT_T || k < T) ? warp_sum(acc[k]) : 0.0f;
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        if ((!EXACT_T && k >= T) || lane != k) continue;
        if (inject_block) m.a[k].inject_out[row] = s[k];
        else {
            const float x = s[k] / (float) HC;
            m.a[k].lo[row] = x / (1.0f + __expf(-x));
        }
    }
}

// gr_down_staged_kernel with the packed weight arrival (STRATA_HC_PACK=1): the same grid (DOWN_BLOCKS + 1), the
// same 256-thread CTA, the same two-buffer tile schedule and stage_htile calls, the same prefetch of the next
// tile's weights, the same j = lane + 32*q lane order, the same dot8u per token, the same warp_sum and lane == k
// epilogue.  Only the weight bytes differ: a chunk arrives as 8 low bytes + 4 nibble bytes + 1 flag byte and is
// composed in registers at the dot site, exactly where the plain kernel unpacks its uint4.
//
// A warp's chunks in tile h are 160h + lane + 32q (q < HQ = 5), so its escape-prefix group is 5h + q and its lane
// is the chunk's lane - the group is the 32 chunks one warp reads in one tile, in the order it reads them.
// D = 10240 and LR = 320 are both multiples of 8, so every row is a whole number of chunks (1280 and 40) and
// there is no tail to handle.
template <int MAX_T = kFusedGrMaxT, bool EXACT_T = false>
__global__ void __launch_bounds__(THREADS) gr_down_staged_packed_kernel(GrMulti m) {
    extern __shared__ __align__(16) float4 hbuf[];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    constexpr int CTA_WARPS = THREADS / 32;
    constexpr int CTA_BLOCKS = LR / CTA_WARPS;
    const int T = EXACT_T ? MAX_T : m.T;
    const bool inject_block = blockIdx.x == CTA_BLOCKS;
    const int row = inject_block ? warp : blockIdx.x * CTA_WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    // w_inject null: the launch lets pack_inject be null too, and the plain kernel's injection CTA writes nothing
    // there.  Take the same exit before touching the (absent) pack.
    const HcPackedWeights& pk = inject_block ? m.a[0].pack_inject : m.a[0].pack_down;
    if (pk.lows == nullptr) return;
    const uint2* lows = reinterpret_cast<const uint2*>(pk.lows);
    const uint32_t* nib = reinterpret_cast<const uint32_t*>(pk.nibbles);
    const uint8_t* flag = pk.flags;
    const size_t crow = (size_t) (active ? row : 0) * (D / HC_PACK_VALUES);      // 1280 chunks per row
    const uint32_t wb = active ? __ldg(pk.esc_warp_row + (blockIdx.x * pk.warps_per_block + warp) * pk.rows_per_warp)
                               : 0u;
    const size_t buf_f4 = (size_t) T * (H_TILE / 4);
    float acc[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) acc[k] = 0.0f;
    uint2 lo[HQ], lonext[HQ];
    uint32_t nb[HQ], nbnext[HQ];
    uint32_t fl[HQ], flnext[HQ];
    if (active) {
#pragma unroll
        for (int q = 0; q < HQ; ++q) {
            const size_t c = lane + 32 * q;
            lo[q] = __ldg(lows + crow + c);
            nb[q] = __ldg(nib + crow + c);
            fl[q] = __ldg(flag + crow + c);
        }
    }
    stage_htile(m, T, 0, hbuf, t, THREADS);
    gr_cuda_async_group_commit();
    stage_htile(m, T, 1, hbuf + buf_f4, t, THREADS);
    gr_cuda_async_group_commit();
#pragma unroll 1
    for (int h = 0; h < N_HTILES; ++h) {
        if (active && h + 1 < N_HTILES) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) {
                const size_t c = (h + 1) * (H_TILE / 8) + lane + 32 * q;
                lonext[q] = __ldg(lows + crow + c);
                nbnext[q] = __ldg(nib + crow + c);
                flnext[q] = __ldg(flag + crow + c);
            }
        }
        if (h + 1 < N_HTILES) gr_cuda_async_group_wait1();
        else gr_cuda_async_group_wait0();
        __syncthreads();
        const float4* cur = hbuf + (h & 1) * buf_f4;
        if (active) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) {
                const int j = lane + 32 * q;
                uint4 w4;
                uint32_t emask = 0;
                w4 = hc_compose8(lo[q], nb[q], fl[q], emask);
                hc_apply_escapes(w4, emask, pk.esc_value,
                                 wb + __ldg(pk.esc_row_group + row * pk.groups_per_row + h * HQ + q) +
                                     hc_warp_before(__popc(emask)));
                const Bf16x8 wvq = unpack8(w4);
#pragma unroll
                for (int k = 0; k < MAX_T; ++k) {
                    if (EXACT_T || k < T) {
                        const float4* pk4 = cur + (size_t) k * (H_TILE / 4);
                        acc[k] += dot8u(wvq, pk4[j], pk4[H_TILE / 8 + j]);
                    }
                }
            }
        }
        __syncthreads();
        if (h + 2 < N_HTILES) stage_htile(m, T, h + 2, hbuf + (h & 1) * buf_f4, t, THREADS);
        gr_cuda_async_group_commit();
        if (h + 1 < N_HTILES) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) {
                lo[q] = lonext[q];
                nb[q] = nbnext[q];
                fl[q] = flnext[q];
            }
        }
    }
    if (!active) return;
    float s[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) s[k] = (EXACT_T || k < T) ? warp_sum(acc[k]) : 0.0f;
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        if ((!EXACT_T && k >= T) || lane != k) continue;
        if (inject_block) m.a[k].inject_out[row] = s[k];
        else {
            const float x = s[k] / (float) HC;
            m.a[k].lo[row] = x / (1.0f + __expf(-x));
        }
    }
}

// Opt-in 4-warp CTA that gives each warp two independent output rows while staging the activation tile once.
template <int MAX_T>
__global__ void __launch_bounds__(HC_REUSE_THREADS) gr_down_reuse_two_rows_kernel(GrMulti m) {
    extern __shared__ __align__(16) float4 hbuf[];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    constexpr int CTA_ROWS = HC_REUSE_ROWS_PER_BLOCK;
    const int T = MAX_T;
    const bool inject_block = blockIdx.x == HC_REUSE_BLOCKS;
    const size_t buf_f4 = (size_t) T * (H_TILE / 4);
    const uint16_t* wbase = inject_block ? m.a[0].w_inject : m.a[0].w_down;
    float acc[HC_REUSE_ROWS_PER_WARP][MAX_T];
    uint4 wv[HC_REUSE_ROWS_PER_WARP][HQ], wnext[HC_REUSE_ROWS_PER_WARP][HQ];
    bool active[HC_REUSE_ROWS_PER_WARP];
#pragma unroll
    for (int r = 0; r < HC_REUSE_ROWS_PER_WARP; ++r) {
        const int row = inject_block ? warp * HC_REUSE_ROWS_PER_WARP + r
                                     : blockIdx.x * CTA_ROWS + warp * HC_REUSE_ROWS_PER_WARP + r;
        active[r] = inject_block ? (m.a[0].w_inject != nullptr && row < HC) : (row < LR);
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) acc[r][k] = 0.0f;
        const uint4* w4 = reinterpret_cast<const uint4*>(wbase + (size_t) (active[r] ? row : 0) * D);
        if (active[r]) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) wv[r][q] = __ldg(w4 + lane + 32 * q);
        }
    }
    stage_htile(m, T, 0, hbuf, t, HC_REUSE_THREADS);
    gr_cuda_async_group_commit();
    stage_htile(m, T, 1, hbuf + buf_f4, t, HC_REUSE_THREADS);
    gr_cuda_async_group_commit();
#pragma unroll 1
    for (int h = 0; h < N_HTILES; ++h) {
#pragma unroll
        for (int r = 0; r < HC_REUSE_ROWS_PER_WARP; ++r) {
            const int row = inject_block ? warp * HC_REUSE_ROWS_PER_WARP + r
                                         : blockIdx.x * CTA_ROWS + warp * HC_REUSE_ROWS_PER_WARP + r;
            const uint4* w4 = reinterpret_cast<const uint4*>(wbase + (size_t) (active[r] ? row : 0) * D);
            if (active[r] && h + 1 < N_HTILES) {
#pragma unroll
                for (int q = 0; q < HQ; ++q) wnext[r][q] = __ldg(w4 + (h + 1) * (H_TILE / 8) + lane + 32 * q);
            }
        }
        if (h + 1 < N_HTILES) gr_cuda_async_group_wait1();
        else gr_cuda_async_group_wait0();
        __syncthreads();
        const float4* cur = hbuf + (h & 1) * buf_f4;
#pragma unroll
        for (int r = 0; r < HC_REUSE_ROWS_PER_WARP; ++r) {
            if (active[r]) {
#pragma unroll
                for (int q = 0; q < HQ; ++q) {
                    const int j = lane + 32 * q;
                    const Bf16x8 wvq = unpack8(wv[r][q]);
#pragma unroll
                    for (int k = 0; k < MAX_T; ++k) {
                        const float4* pk = cur + (size_t) k * (H_TILE / 4);
                        acc[r][k] += dot8u(wvq, pk[j], pk[H_TILE / 8 + j]);
                    }
                }
            }
        }
        __syncthreads();
        if (h + 2 < N_HTILES) stage_htile(m, T, h + 2, hbuf + (h & 1) * buf_f4, t, HC_REUSE_THREADS);
        gr_cuda_async_group_commit();
        if (h + 1 < N_HTILES) {
#pragma unroll
            for (int r = 0; r < HC_REUSE_ROWS_PER_WARP; ++r) {
#pragma unroll
                for (int q = 0; q < HQ; ++q) wv[r][q] = wnext[r][q];
            }
        }
    }
#pragma unroll
    for (int r = 0; r < HC_REUSE_ROWS_PER_WARP; ++r) {
        if (!active[r]) continue;
        const int row = inject_block ? warp * HC_REUSE_ROWS_PER_WARP + r
                                     : blockIdx.x * CTA_ROWS + warp * HC_REUSE_ROWS_PER_WARP + r;
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) {
            const float s = warp_sum(acc[r][k]);
            if (lane != k) continue;
            if (inject_block) m.a[k].inject_out[row] = s;
            else {
                const float x = s / (float) HC;
                m.a[k].lo[row] = x / (1.0f + __expf(-x));
            }
        }
    }
}

// Opt-in (STRATA_HC_SPLIT=6): the staged schedule with the NEXT tile carried in registers.  Same CTA mapping,
// same tile, same weights, same dot and the same order as `gr_down_staged_kernel`; only the arrival of tile
// h + 2 changes.  The staged kernel reads global -> LDS after the barrier of tile h; here tile h + 2 is read
// into the thread's own tuple (registers) while tile h + 1's dot runs, and scattered to the LDS buffer after
// that dot, before the barrier of tile h + 2.  The load is a plain global -> VGPR load (gfx1201 has no
// invocable global -> LDS async copy; see the note above), so nothing here claims an asynchronous copy.
// The prime (tiles 0, 1 to LDS, tile 2 to the tuple) and the double-buffer/barrier ownership are the staged
// kernel's: the scatter at h writes buffer h & 1, which tile h's dot just released, and tile h + 2 is read
// from it at h + 2.  Bitwise the staged kernel: identical values, identical accumulation order.
template <int MAX_T = kFusedGrMaxT, bool EXACT_T = false>
__global__ void __launch_bounds__(THREADS) gr_down_register_pipe_kernel(GrMulti m) {
    extern __shared__ __align__(16) float4 hbuf[];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    const bool inject_block = blockIdx.x == DOWN_BLOCKS;
    const int row = inject_block ? warp : blockIdx.x * WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) (active ? row : 0) * D;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
    const size_t buf_f4 = (size_t) T * (H_TILE / 4);
    const int total = T * (H_TILE / 4);
    constexpr int NS = (MAX_T * (H_TILE / 4) + THREADS - 1) / THREADS;   // tuple slots for the widest launch
    static_assert(NS * THREADS >= MAX_T * (H_TILE / 4), "the tuple covers the widest tile");
    float acc[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) acc[k] = 0.0f;
    uint4 wv[HQ], wnext[HQ];
    if (active) {
#pragma unroll
        for (int q = 0; q < HQ; ++q) wv[q] = __ldg(w4 + lane + 32 * q);
    }
    stage_htile(m, T, 0, hbuf, t, THREADS);
    gr_cuda_async_group_commit();
    stage_htile(m, T, 1, hbuf + buf_f4, t, THREADS);
    gr_cuda_async_group_commit();
    HcChain<NS - 1> tup;
    hc_load_all<0, NS - 1>(tup, m, 2, t, total);
#pragma unroll 1
    for (int h = 0; h < N_HTILES; ++h) {
        if (active && h + 1 < N_HTILES) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) wnext[q] = __ldg(w4 + (h + 1) * (H_TILE / 8) + lane + 32 * q);
        }
        if (h + 1 < N_HTILES) gr_cuda_async_group_wait1();
        else gr_cuda_async_group_wait0();
        __syncthreads();
        const float4* cur = hbuf + (h & 1) * buf_f4;
        if (active) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) {
                const int j = lane + 32 * q;
                const Bf16x8 wvq = unpack8(wv[q]);
#pragma unroll
                for (int k = 0; k < MAX_T; ++k) {
                    if (EXACT_T || k < T) {
                        const float4* pk = cur + (size_t) k * (H_TILE / 4);
                        acc[k] += dot8u(wvq, pk[j], pk[H_TILE / 8 + j]);
                    }
                }
            }
        }
        __syncthreads();                                // buffer h & 1 is free: the tile it held was just dotted
        if (h + 2 < N_HTILES) hc_scatter_all<0, NS - 1>(tup, hbuf + (h & 1) * buf_f4, t, total);
        if (h + 3 < N_HTILES) hc_load_all<0, NS - 1>(tup, m, h + 3, t, total);
        gr_cuda_async_group_commit();                   // an empty group at the end keeps the wait counts simple
        if (h + 1 < N_HTILES) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) wv[q] = wnext[q];
        }
    }
    if (!active) return;
    float s[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) s[k] = (EXACT_T || k < T) ? warp_sum(acc[k]) : 0.0f;
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        if ((!EXACT_T && k >= T) || lane != k) continue;
        if (inject_block) m.a[k].inject_out[row] = s[k];
        else {
            const float x = s[k] / (float) HC;
            m.a[k].lo[row] = x / (1.0f + __expf(-x));
        }
    }
}

// Opt-in (STRATA_HC_SPLIT=7): the same pipeline with the tuple split in two compile-time halves, so only
// half the slots are live across a full dot.  The second half is loaded mid-dot (at q == 2, a plain global
// load interleaved with the arithmetic) and the first half after the scatter, which is the schedule the
// synthetic A/B measured; the halves are slots [0, H0) and [H0, NS) of the same tile, so the scatter covers
// every slot exactly once and the buffer holds the whole tile before its barrier.  Same bitwise contract as
// the pipe variant above.
template <int MAX_T = kFusedGrMaxT, bool EXACT_T = false>
__global__ void __launch_bounds__(THREADS) gr_down_register_half_kernel(GrMulti m) {
    extern __shared__ __align__(16) float4 hbuf[];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    const bool inject_block = blockIdx.x == DOWN_BLOCKS;
    const int row = inject_block ? warp : blockIdx.x * WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) (active ? row : 0) * D;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
    const size_t buf_f4 = (size_t) T * (H_TILE / 4);
    const int total = T * (H_TILE / 4);
    constexpr int NS = (MAX_T * (H_TILE / 4) + THREADS - 1) / THREADS;
    constexpr int H0 = (NS + 1) / 2;                    // the first half's slots; the rest are t1's
    static_assert(H0 >= 1 && NS - H0 >= 0, "both halves exist for the widest tile");
    float acc[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) acc[k] = 0.0f;
    uint4 wv[HQ], wnext[HQ];
    if (active) {
#pragma unroll
        for (int q = 0; q < HQ; ++q) wv[q] = __ldg(w4 + lane + 32 * q);
    }
    stage_htile(m, T, 0, hbuf, t, THREADS);
    gr_cuda_async_group_commit();
    stage_htile(m, T, 1, hbuf + buf_f4, t, THREADS);
    gr_cuda_async_group_commit();
    HcChain<H0 - 1> t0;
    HcChain<NS - H0 - 1> t1;
    hc_load_all<0, H0 - 1>(t0, m, 2, t, total);
#pragma unroll 1
    for (int h = 0; h < N_HTILES; ++h) {
        if (active && h + 1 < N_HTILES) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) wnext[q] = __ldg(w4 + (h + 1) * (H_TILE / 8) + lane + 32 * q);
        }
        if (h + 1 < N_HTILES) gr_cuda_async_group_wait1();
        else gr_cuda_async_group_wait0();
        __syncthreads();
        const float4* cur = hbuf + (h & 1) * buf_f4;
        if (active) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) {
                if (q == 2 && h + 2 < N_HTILES)
                    hc_load_all<H0, NS - H0 - 1>(t1, m, h + 2, t, total);   // in flight during the dot
                const int j = lane + 32 * q;
                const Bf16x8 wvq = unpack8(wv[q]);
#pragma unroll
                for (int k = 0; k < MAX_T; ++k) {
                    if (EXACT_T || k < T) {
                        const float4* pk = cur + (size_t) k * (H_TILE / 4);
                        acc[k] += dot8u(wvq, pk[j], pk[H_TILE / 8 + j]);
                    }
                }
            }
        }
        if (h + 2 < N_HTILES) hc_load_all<H0, NS - H0 - 1>(t1, m, h + 2, t, total);   // the warps that did not dot
        __syncthreads();
        if (h + 2 < N_HTILES) {
            hc_scatter_all<0, H0 - 1>(t0, hbuf + (h & 1) * buf_f4, t, total);
            hc_scatter_all<H0, NS - H0 - 1>(t1, hbuf + (h & 1) * buf_f4, t, total);
        }
        if (h + 3 < N_HTILES) hc_load_all<0, H0 - 1>(t0, m, h + 3, t, total);
        gr_cuda_async_group_commit();
        if (h + 1 < N_HTILES) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) wv[q] = wnext[q];
        }
    }
    if (!active) return;
    float s[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) s[k] = (EXACT_T || k < T) ? warp_sum(acc[k]) : 0.0f;
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        if ((!EXACT_T && k >= T) || lane != k) continue;
        if (inject_block) m.a[k].inject_out[row] = s[k];
        else {
            const float x = s[k] / (float) HC;
            m.a[k].lo[row] = x / (1.0f + __expf(-x));
        }
    }
}

// Opt-in (STRATA_HC_SPLIT=10): the register-tuple pipeline with BOTH register arrays split in halves.
//
// What 6 and 7 actually are, read from the two kernels above: the weight prefetch (`wv` / `wnext`, the double
// buffer that keeps tile h + 1's weights in flight across tile h's dot) is IDENTICAL in 6 and 7 - 7 is 6 with the
// activation tuple split in halves and the second half loaded mid-dot.  So the union of 6 and 7 is 7, and the
// combination worth measuring is the half treatment applied to the register array 7 left whole: the weights.
// This kernel keeps 7's activation schedule unchanged and splits the weight prefetch the same way - the first
// half of `wnext` (q < W0) is issued before the barrier, as 6 and 7 do, and the second half is issued mid-dot
// (at q == W0), so the live range of `wnext[W0, HQ)` starts inside the dot instead of before the barrier.
//
// Same grid (DOWN_BLOCKS + 1), same 256-thread CTA, same two-buffer tile schedule, same shared-memory contract
// as staged, same weights, same dot and the same q-outer / k-inner order: the loads this moves are pure data
// movement, so the sums and their order are staged's, and the bitwise contract is the pipe and half kernels'
// (checked against staged at every T by fused_gr_selftest).  The halves cover every slot exactly once and the
// scatter writes the whole tile before its barrier, as in 7.
template <int MAX_T = kFusedGrMaxT, bool EXACT_T = false>
__global__ void __launch_bounds__(THREADS) gr_down_register_pipe_half_kernel(GrMulti m) {
    extern __shared__ __align__(16) float4 hbuf[];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    const bool inject_block = blockIdx.x == DOWN_BLOCKS;
    const int row = inject_block ? warp : blockIdx.x * WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) (active ? row : 0) * D;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
    const size_t buf_f4 = (size_t) T * (H_TILE / 4);
    const int total = T * (H_TILE / 4);
    constexpr int NS = (MAX_T * (H_TILE / 4) + THREADS - 1) / THREADS;
    constexpr int H0 = (NS + 1) / 2;                    // activation tuple: slots [0, H0) and [H0, NS)
    constexpr int W0 = (HQ + 1) / 2;                    // weight prefetch: q [0, W0) and [W0, HQ)
    static_assert(H0 >= 1 && NS - H0 >= 0, "both tuple halves exist for the widest tile");
    static_assert(W0 >= 1 && HQ - W0 >= 1, "both weight halves exist");
    float acc[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) acc[k] = 0.0f;
    uint4 wv[HQ], wnext[HQ];
    if (active) {
#pragma unroll
        for (int q = 0; q < HQ; ++q) wv[q] = __ldg(w4 + lane + 32 * q);
    }
    stage_htile(m, T, 0, hbuf, t, THREADS);
    gr_cuda_async_group_commit();
    stage_htile(m, T, 1, hbuf + buf_f4, t, THREADS);
    gr_cuda_async_group_commit();
    HcChain<H0 - 1> t0;
    HcChain<NS - H0 - 1> t1;
    hc_load_all<0, H0 - 1>(t0, m, 2, t, total);
#pragma unroll 1
    for (int h = 0; h < N_HTILES; ++h) {
        // the first weight half in flight before the barrier (6 and 7 issue all HQ here)
        if (active && h + 1 < N_HTILES) {
#pragma unroll
            for (int q = 0; q < W0; ++q) wnext[q] = __ldg(w4 + (h + 1) * (H_TILE / 8) + lane + 32 * q);
        }
        if (h + 1 < N_HTILES) gr_cuda_async_group_wait1();
        else gr_cuda_async_group_wait0();
        __syncthreads();
        const float4* cur = hbuf + (h & 1) * buf_f4;
        if (active) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) {
                if (q == W0 && h + 1 < N_HTILES)
#pragma unroll
                    for (int q2 = W0; q2 < HQ; ++q2)
                        wnext[q2] = __ldg(w4 + (h + 1) * (H_TILE / 8) + lane + 32 * q2);   // in flight during the dot
                if (q == 2 && h + 2 < N_HTILES)
                    hc_load_all<H0, NS - H0 - 1>(t1, m, h + 2, t, total);                   // as 7
                const int j = lane + 32 * q;
                const Bf16x8 wvq = unpack8(wv[q]);
#pragma unroll
                for (int k = 0; k < MAX_T; ++k) {
                    if (EXACT_T || k < T) {
                        const float4* pk = cur + (size_t) k * (H_TILE / 4);
                        acc[k] += dot8u(wvq, pk[j], pk[H_TILE / 8 + j]);
                    }
                }
            }
        }
        if (h + 2 < N_HTILES) hc_load_all<H0, NS - H0 - 1>(t1, m, h + 2, t, total);   // the warps that did not dot
        __syncthreads();
        if (h + 2 < N_HTILES) {
            hc_scatter_all<0, H0 - 1>(t0, hbuf + (h & 1) * buf_f4, t, total);
            hc_scatter_all<H0, NS - H0 - 1>(t1, hbuf + (h & 1) * buf_f4, t, total);
        }
        if (h + 3 < N_HTILES) hc_load_all<0, H0 - 1>(t0, m, h + 3, t, total);
        gr_cuda_async_group_commit();
        if (h + 1 < N_HTILES) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) wv[q] = wnext[q];
        }
    }
    if (!active) return;
    float s[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) s[k] = (EXACT_T || k < T) ? warp_sum(acc[k]) : 0.0f;
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        if ((!EXACT_T && k >= T) || lane != k) continue;
        if (inject_block) m.a[k].inject_out[row] = s[k];
        else {
            const float x = s[k] / (float) HC;
            m.a[k].lo[row] = x / (1.0f + __expf(-x));
        }
    }
}

// Opt-in (STRATA_HC_SPLIT=8): the staged down projection split over twice the blocks.  The staged read runs the
// 320 rows on 40 CTAs of 8 warps, one row per warp, and leaves 23 of the 64 CUs idle at every T.  This variant
// keeps the 256-thread CTA and gives each CTA 4 rows, each row shared by a warp PAIR: the even warp walks the
// staged lanes 0-15's lane jobs, the odd warp the staged lanes 16-31's, both over the whole tile sequence, so
// every lane's accumulation order is the staged one.  The pair meets at the warp_sum offset-16 step: the odd
// warp publishes its raw per-lane sums to shared memory, the even warp adds them to its own and runs the
// offsets 8,4,2,1 over lanes 0-15 - the staged shuffle tree, bit for bit.  The inject CTA is the staged one.
constexpr int RS_ROWS_PER_CTA = WARPS / 2;              // 4 rows per 256-thread CTA
constexpr int RS_BLOCKS = LR / RS_ROWS_PER_CTA;         // 80 down blocks; block 80 = the inject rows
static_assert(RS_BLOCKS * RS_ROWS_PER_CTA == LR, "the row split covers every down row");
template <int MAX_T = kFusedGrMaxT, bool EXACT_T = false>
__global__ void __launch_bounds__(THREADS) gr_down_row_split_kernel(GrMulti m) {
    extern __shared__ __align__(16) float4 hbuf[];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    const bool inject_block = blockIdx.x == RS_BLOCKS;
    const int pair = warp >> 1;                                   // the row pair within the CTA
    const int row = inject_block ? warp : blockIdx.x * RS_ROWS_PER_CTA + pair;
    const bool active = inject_block ? (m.a[0].w_inject != nullptr && warp < HC) : lane < 16;
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) row * D;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
    // the inject CTA walks whole rows on 32 lanes (the staged mapping); the pair split is the down warps'
    const int j = inject_block ? lane : (warp & 1) * 16 + lane;    // the staged lane job this thread walks
    const size_t buf_f4 = (size_t) T * (H_TILE / 4);
    float acc[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) acc[k] = 0.0f;
    uint4 wv[HQ], wnext[HQ];
    if (active) {
#pragma unroll
        for (int q = 0; q < HQ; ++q) wv[q] = __ldg(w4 + j + 32 * q);
    }
    stage_htile(m, T, 0, hbuf, t, THREADS);
    gr_cuda_async_group_commit();
    stage_htile(m, T, 1, hbuf + buf_f4, t, THREADS);
    gr_cuda_async_group_commit();
#pragma unroll 1
    for (int h = 0; h < N_HTILES; ++h) {
        if (active && h + 1 < N_HTILES) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) wnext[q] = __ldg(w4 + (h + 1) * (H_TILE / 8) + j + 32 * q);
        }
        if (h + 1 < N_HTILES) gr_cuda_async_group_wait1();
        else gr_cuda_async_group_wait0();
        __syncthreads();
        const float4* cur = hbuf + (h & 1) * buf_f4;
        if (active) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) {
                const int jj = j + 32 * q;
                const Bf16x8 wvq = unpack8(wv[q]);
#pragma unroll
                for (int k = 0; k < MAX_T; ++k) {
                    if (EXACT_T || k < T) {
                        const float4* pk = cur + (size_t) k * (H_TILE / 4);
                        acc[k] += dot8u(wvq, pk[jj], pk[H_TILE / 8 + jj]);
                    }
                }
            }
        }
        __syncthreads();
        if (h + 2 < N_HTILES) stage_htile(m, T, h + 2, hbuf + (h & 1) * buf_f4, t, THREADS);
        gr_cuda_async_group_commit();
        if (h + 1 < N_HTILES) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) wv[q] = wnext[q];
        }
    }
    if (inject_block) {                                          // the staged inject CTA, unchanged
        if (!active) return;
        float s[MAX_T];
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) s[k] = (EXACT_T || k < T) ? warp_sum(acc[k]) : 0.0f;
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) {
            if ((!EXACT_T && k >= T) || lane != k) continue;
            m.a[k].inject_out[row] = s[k];
        }
        return;
    }
    // The pair's meeting.  The odd warp publishes the raw per-lane sums of the staged lanes 16-31; the even
    // warp adds its own (the staged lanes 0-15's) - the warp_sum offset-16 step - and runs the offsets 8,4,2,1
    // over lanes 0-15.  Per-lane sums staged, tree staged, bits staged.  The tiles are consumed: the exchange
    // reuses their space (4 pairs x MAX_T x 16 floats <= 512 floats, the first tile buffer holds 1280).
    float* exch = reinterpret_cast<float*>(hbuf);
    __syncthreads();
    if ((warp & 1) && lane < 16) {
#pragma unroll
        for (int k = 0; k < MAX_T; ++k)
            if (EXACT_T || k < T) exch[(pair * MAX_T + k) * 16 + lane] = acc[k];
    }
    __syncthreads();
    const bool lead = !(warp & 1) && lane < 16;   // the even warp's lanes 0-15 finish the row
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        if (!EXACT_T && k >= T) continue;
        float v = 0.0f;
        if (lead) v = acc[k] + exch[(pair * MAX_T + k) * 16 + lane];   // the offset-16 step
        // The offsets 8,4,2,1 pair lanes within 0-15, so a full-mask shuffle is the staged tree for the
        // lead lanes (the others carry 0 and their sums are discarded).  A partial mask with the odd
        // warp's lanes already exited faults on gfx1201: the shfl emulation converges the whole warp.
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lead && lane == k) {
            const float x = v / (float) HC;
            m.a[k].lo[row] = x / (1.0f + __expf(-x));
        }
    }
}

// gr_down_staged_kernel with the per-token accumulators in dynamic LDS (opt-in STRATA_HC_SPLIT=9): the same grid,
// the same 256-thread CTA, the same two-buffer tile schedule and stage_htile calls, the same weight prefetch, the
// same q-outer / k-inner loop and the same dot8u per (q, k).  Only where the running sums live: staged keeps
// `acc[MAX_T]` and `s[MAX_T]` in the thread's registers (measured on gfx1201: 112 VGPR at T=1, 152 at T>=3,
// RESULTS.md), and this kernel keeps one per-token sum per thread in a dynamic-LDS row - THREADS*MAX_T floats on
// top of the two staged tiles - so the array is not live across the tile loop.  The row is addressed as
// t*MAX_T + k with k a compile-time constant; a runtime k lowers to per-thread private scratch on gfx1201 (the
// HcChain note above), which is the thing this form avoids.
//
// The per-token addition order is staged's exactly: for each token, the chunks enter in ascending q, and the
// epilogue warp-sums the same per-thread sum.  The sums are bitwise the same as staged's.
//
// LDS cost: the row adds THREADS*MAX_T bytes*4 to the staged 2*T*H_TILE floats, so a launch of ct tokens needs
// ct*(2*H_TILE + THREADS)*4 bytes - 11264 B per token on this geometry.  The 64 KiB cards fit ct <= 5; the
// dispatch below keeps larger launches on staged.
template <int MAX_T = kFusedGrMaxT, bool EXACT_T = false>
__global__ void __launch_bounds__(THREADS) gr_down_lds_accum_kernel(GrMulti m) {
    extern __shared__ __align__(16) float4 hbuf[];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    constexpr int CTA_WARPS = THREADS / 32;
    constexpr int CTA_BLOCKS = LR / CTA_WARPS;
    const int T = EXACT_T ? MAX_T : m.T;
    const bool inject_block = blockIdx.x == CTA_BLOCKS;
    const int row = inject_block ? warp : blockIdx.x * CTA_WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) (active ? row : 0) * D;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
    const size_t buf_f4 = (size_t) T * (H_TILE / 4);
    float* const lacc = reinterpret_cast<float*>(hbuf + 2 * buf_f4) + (size_t) t * MAX_T;
    if (active) {
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) lacc[k] = 0.0f;
    }
    uint4 wv[HQ], wnext[HQ];
    if (active) {
#pragma unroll
        for (int q = 0; q < HQ; ++q) wv[q] = __ldg(w4 + lane + 32 * q);
    }
    stage_htile(m, T, 0, hbuf, t, THREADS);
    gr_cuda_async_group_commit();
    stage_htile(m, T, 1, hbuf + buf_f4, t, THREADS);
    gr_cuda_async_group_commit();
#pragma unroll 1
    for (int h = 0; h < N_HTILES; ++h) {
        if (active && h + 1 < N_HTILES) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) wnext[q] = __ldg(w4 + (h + 1) * (H_TILE / 8) + lane + 32 * q);
        }
        if (h + 1 < N_HTILES) gr_cuda_async_group_wait1();
        else gr_cuda_async_group_wait0();
        __syncthreads();
        const float4* cur = hbuf + (h & 1) * buf_f4;
        if (active) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) {
                const int j = lane + 32 * q;
                const Bf16x8 wvq = unpack8(wv[q]);
#pragma unroll
                for (int k = 0; k < MAX_T; ++k) {
                    if (EXACT_T || k < T) {
                        const float4* pk = cur + (size_t) k * (H_TILE / 4);
                        lacc[k] += dot8u(wvq, pk[j], pk[H_TILE / 8 + j]);
                    }
                }
            }
        }
        __syncthreads();
        if (h + 2 < N_HTILES) stage_htile(m, T, h + 2, hbuf + (h & 1) * buf_f4, t, THREADS);
        gr_cuda_async_group_commit();
        if (h + 1 < N_HTILES) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) wv[q] = wnext[q];
        }
    }
    if (!active) return;
    // Converged, as staged: every lane warp-sums every token's sum, and only lane k keeps it.  Guarding the
    // warp sum itself (only lane k running it) leaves __shfl_xor_sync(0xffffffff) with lanes already exited,
    // which the shfl emulation on gfx1201 answers with the wrong bits - the row-split kernel's note above.
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        const float s = (EXACT_T || k < T) ? warp_sum(lacc[k]) : 0.0f;
        if ((!EXACT_T && k >= T) || lane != k) continue;
        if (inject_block) m.a[k].inject_out[row] = s;
        else {
            const float x = s / (float) HC;
            m.a[k].lo[row] = x / (1.0f + __expf(-x));
        }
    }
}

// the current device is Volta (sm_70): the fast norm/up is its default
bool cur_dev_volta() {
#if defined(__HIPCC__)
    return false;
#else
    static int per_dev[64];   // 0 unknown, 1 no, 2 yes
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    if (!per_dev[dev]) {
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        per_dev[dev] = major == 7 && minor == 0 ? 2 : 1;
    }
    return per_dev[dev] == 2;
#endif
}

// tokens the LDS-accumulator down kernel (STRATA_HC_SPLIT=9) may carry in one launch on each card; filled with
// the shared-memory opt-in in down_chunk below.
static int chunk_lds[64] = {};

// The tokens a down kernel may carry in one launch on the current card, and the plain read's tile: the plain read
// stages n_tok * TILEV floats (1280 on sm_75, 2560 elsewhere), staged two tiles of n_tok * H_TILE.  The shared-memory
// opt-in is set here once per device (a per-DEVICE setting: a layer split runs these kernels on two cards).
int down_chunk(bool staged, int* tile_out) {
    static bool attr[64] = {};
    static int chunk[64] = {};   // tokens the plain down kernel may carry in one launch on this card
    static int chunk_staged[64] = {};  // the same for staged
    static int tile[64] = {};    // the plain down kernel's TILEV on this card (1280 on sm_75, 2560 elsewhere)
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) {
        *tile_out = 2560;
        return kFusedGrMaxT;
    }
    if (!attr[dev]) {
        int optin = 0;
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        optin = strata::smem_optin_of(optin);   // STRATA_EMULATE_CC (tests only)
        // the down kernel stages n_tok*TILEV floats of dynamic shared memory - 80 KB at the full 8 tokens of the
        // CUDA tile.  sm_75 gets the smaller tile: all eight tokens fit one 40 KiB launch there (no more slicing),
        // the TQ-5 prefetch holds half the registers, and the smaller blocks raise how many of the 41-block grid
        // share an SM.  Cards whose opt-in is still below that (or that report no opt-in at all) slice the tokens;
        // the down kernel's outputs (lo, inject_out) are strictly per-token, so the chunk boundaries are safe, and
        // the up kernel below still sees every token of the batch in one launch.
#if defined(__HIPCC__)
#if defined(STRATA_HIP_GFX906)
        const bool small_tile = false;  // gfx906: 64 KiB LDS, the 2560 tile as before (8 tokens slice into launches)
#else
        const bool small_tile = true;   // all eight tokens fit gfx1100's 64 KiB LDS at this tile
#endif
#else
        int cc_maj = 0, cc_min = 0;
        cudaDeviceGetAttribute(&cc_maj, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&cc_min, cudaDevAttrComputeCapabilityMinor, dev);
        const bool small_tile = strata::cc_major_of(cc_maj) * 10 + strata::cc_minor_of(cc_min) == 75;
#endif
        const int tv = small_tile ? 1280 : 2560;
        tile[dev] = tv;
        int want = (int) (kFusedGrMaxT * tv * sizeof(float));
        if (optin > 0 && want > optin) want = optin;
        if (small_tile) {
            cudaFuncSetAttribute(gr_down_multi_kernel<1280>, cudaFuncAttributeMaxDynamicSharedMemorySize, want);
        } else {
            cudaFuncSetAttribute(gr_down_multi_kernel<2560>, cudaFuncAttributeMaxDynamicSharedMemorySize, want);
        }
        auto set_staged_attr = [&](auto kernel, int max_t) {
            int ws = (int) (2 * max_t * H_TILE * sizeof(float));
            if (optin > 0 && ws > optin) ws = optin;
            if (ws > 48 * 1024)
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ws);
            cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        };
        set_staged_attr(gr_down_staged_kernel<1, true>, 1);
        set_staged_attr(gr_down_staged_kernel<2, true>, 2);
        set_staged_attr(gr_down_staged_kernel<3, true>, 3);
        set_staged_attr(gr_down_staged_kernel<4, true>, 4);
        set_staged_attr(gr_down_staged_kernel<5, true>, 5);
        set_staged_attr(gr_down_staged_kernel<6, true>, 6);
        set_staged_attr(gr_down_staged_kernel<7, true>, 7);
        set_staged_attr(gr_down_staged_kernel<8, true>, 8);
        set_staged_attr(gr_down_staged_kernel<4, false>, 4);
        set_staged_attr(gr_down_staged_kernel<kFusedGrMaxT, false>, kFusedGrMaxT);
        // Configure the 4-warp and 2-row-per-warp variants under the same shared-memory contract.
        set_staged_attr(gr_down_staged_kernel<1, true, HC_SMALL_CTA_THREADS>, 1);
        set_staged_attr(gr_down_staged_kernel<2, true, HC_SMALL_CTA_THREADS>, 2);
        set_staged_attr(gr_down_staged_kernel<3, true, HC_SMALL_CTA_THREADS>, 3);
        set_staged_attr(gr_down_staged_kernel<4, true, HC_SMALL_CTA_THREADS>, 4);
        set_staged_attr(gr_down_staged_kernel<5, true, HC_SMALL_CTA_THREADS>, 5);
        set_staged_attr(gr_down_staged_kernel<6, true, HC_SMALL_CTA_THREADS>, 6);
        set_staged_attr(gr_down_staged_kernel<7, true, HC_SMALL_CTA_THREADS>, 7);
        set_staged_attr(gr_down_staged_kernel<8, true, HC_SMALL_CTA_THREADS>, 8);
        set_staged_attr(gr_down_staged_kernel<kFusedGrMaxT, false, HC_SMALL_CTA_THREADS>, kFusedGrMaxT);
        // The register-tuple variants (6/7) stage the same two tiles: the same shared-memory contract.
        // Only the instantiations the dispatch uses are configured: on gfx1201 the exact tuple kernels
        // measure private_segment_fixed_size 0 and vgpr_spill 0 up to MAX_T 4 (pipe) / 7 (half), and the
        // generic (runtime-T) kernel measures 0 for both; the exact kernels past that point spill, so
        // they are not dispatched and not compiled in (see the launch below).
        set_staged_attr(gr_down_register_pipe_kernel<1, true>, 1);
        set_staged_attr(gr_down_register_pipe_kernel<2, true>, 2);
        set_staged_attr(gr_down_register_pipe_kernel<3, true>, 3);
        set_staged_attr(gr_down_register_pipe_kernel<4, true>, 4);
        set_staged_attr(gr_down_register_pipe_kernel<kFusedGrMaxT, false>, kFusedGrMaxT);
        set_staged_attr(gr_down_register_half_kernel<1, true>, 1);
        set_staged_attr(gr_down_register_half_kernel<2, true>, 2);
        set_staged_attr(gr_down_register_half_kernel<3, true>, 3);
        set_staged_attr(gr_down_register_half_kernel<4, true>, 4);
        set_staged_attr(gr_down_register_half_kernel<kFusedGrMaxT, false>, kFusedGrMaxT);
        // The pipe+half combination (env 10) stages the same two tiles: the same shared-memory contract, and the
        // same instantiations the dispatch uses (exact 1..4, generic past that).
        set_staged_attr(gr_down_register_pipe_half_kernel<1, true>, 1);
        set_staged_attr(gr_down_register_pipe_half_kernel<2, true>, 2);
        set_staged_attr(gr_down_register_pipe_half_kernel<3, true>, 3);
        set_staged_attr(gr_down_register_pipe_half_kernel<4, true>, 4);
        set_staged_attr(gr_down_register_pipe_half_kernel<kFusedGrMaxT, false>, kFusedGrMaxT);
        // The row split (8) stages the same two tiles per CTA: the same shared-memory contract.
        set_staged_attr(gr_down_row_split_kernel<1, true>, 1);
        set_staged_attr(gr_down_row_split_kernel<2, true>, 2);
        set_staged_attr(gr_down_row_split_kernel<3, true>, 3);
        set_staged_attr(gr_down_row_split_kernel<4, true>, 4);
        set_staged_attr(gr_down_row_split_kernel<5, true>, 5);
        set_staged_attr(gr_down_row_split_kernel<6, true>, 6);
        set_staged_attr(gr_down_row_split_kernel<kFusedGrMaxT, false>, kFusedGrMaxT);
        // The LDS-accumulator down kernel (STRATA_HC_SPLIT=9): the two staged tiles PLUS the per-token
        // accumulator row (THREADS floats per token of MAX_T), so its opt-in is larger than staged's.
        auto set_lds_accum_attr = [&](auto kernel, int max_t) {
            int ws = (int) (max_t * (2 * H_TILE + THREADS) * sizeof(float));
            if (optin > 0 && ws > optin) ws = optin;
            if (ws > 48 * 1024)
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ws);
            cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        };
        set_lds_accum_attr(gr_down_lds_accum_kernel<1, true>, 1);
        set_lds_accum_attr(gr_down_lds_accum_kernel<2, true>, 2);
        set_lds_accum_attr(gr_down_lds_accum_kernel<3, true>, 3);
        set_lds_accum_attr(gr_down_lds_accum_kernel<4, true>, 4);
        set_lds_accum_attr(gr_down_lds_accum_kernel<5, true>, 5);
        set_lds_accum_attr(gr_down_lds_accum_kernel<6, true>, 6);
        set_lds_accum_attr(gr_down_lds_accum_kernel<kFusedGrMaxT, false>, kFusedGrMaxT);
        cudaGetLastError();      // drop any error the attempt left behind
        // The opt-in is a promise a pre-Volta card does not keep: an sm_60 answers 65536 and accepts the
        // cudaFuncSetAttribute for 61440 B, then fails the LAUNCH with "invalid argument".  What such a card
        // will launch is its per-block limit, so the capacity comes from that below sm_70 - the same tokens
        // the "no opt-in" branch assumes, but taken from the attribute that is actually enforced.
#if defined(__HIPCC__)
        const int usable = optin > 0 ? optin : 48 * 1024;
#else
        int cc = 0, per_block = 0;
        cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&per_block, cudaDevAttrMaxSharedMemoryPerBlock, dev);
        cc = strata::cc_major_of(cc);
        const int usable = (cc >= 7 && optin > 0) ? optin : per_block;
#endif
        const int capacity = usable / (int) (tv * sizeof(float));
        chunk[dev] = capacity < 1 ? 1 : (capacity > kFusedGrMaxT ? kFusedGrMaxT : capacity);
        const int capacity_staged = usable / (int) (2 * H_TILE * sizeof(float));
        chunk_staged[dev] = capacity_staged < 1 ? 1 : (capacity_staged > kFusedGrMaxT ? kFusedGrMaxT : capacity_staged);
        // the LDS-accumulator kernel stages 2*H_TILE + THREADS floats per token (the two tiles plus its own
        // accumulator row), so it slices earlier than staged: 5 tokens on a 64 KiB card, 8 on a 96 KiB one.
        const int capacity_lds = usable / (int) ((2 * H_TILE + THREADS) * sizeof(float));
        chunk_lds[dev] = capacity_lds < 1 ? 1 : (capacity_lds > kFusedGrMaxT ? kFusedGrMaxT : capacity_lds);
        attr[dev] = true;
    }
    *tile_out = tile[dev] ? tile[dev] : 2560;
    return staged ? chunk_staged[dev] : chunk[dev];
}

// The tokens the LDS-accumulator down kernel (STRATA_HC_SPLIT=9) carries in one launch on the current card.  The
// card's opt-in is read by down_chunk, so this is called after it.  A card that cannot fit even one token of the
// kernel's shape answers 0 and the launch takes staged.
int down_chunk_lds() {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) return kFusedGrMaxT;
    return chunk_lds[dev];
}

// The multi read as `variant` (kHcPlain, kHcSplit or kHcStaged): the norm, the down projection in launches of as
// many tokens as fit the card, the up projection; the profile's stamps after the norm and after the down projection.
// kHcPlain is the default read exactly as before (never main's opt-in STRATA_GR_V3 path, which fused_gr_read_multi
// takes first).
#if STRATA_GR_FAST_BUILD
__global__ void __launch_bounds__(THREADS) gr_norm_fast_kernel(GrMulti m);   // below, with the AMD fast path
__global__ void __launch_bounds__(THREADS) gr_up_fast_kernel(GrMulti m);
}  // namespace
static bool gr_fast();
namespace {
#endif
// STRATA_GR_DOWN_MAX4=1 (opt-in): launches of up to 4 tokens hold 4 tokens' sums per thread instead of kFusedGrMaxT (the
// same bits; checked bitwise on the 5070 for 0.1.40). Default off: a 10-pair interleaved decode A/B on the RTX 5070 (Q2_0,
// default config) did not show the +2.8-3.5% of bench #832 (medians 0.97-0.99 of the off arm).
bool gr_down_max4() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_GR_DOWN_MAX4");
        return v != nullptr && std::atoi(v) != 0;
    }();
    return on;
}

void launch_multi(const GrMulti& m, int variant, cudaStream_t st, unsigned long long* stamp_buf, int stamp_i0) {
    const int n_tok = m.T;
#if STRATA_GR_FAST_BUILD
    // gfx906: the latency-hidden norm/up (STRATA_GR_FAST=0: off) - the same sums in the same order as the kernels
    // they replace (gr_parity checks the multi-token read against single-token calls bitwise)
    const bool fast = gr_fast();
    if (variant >= kHcSplit) gr_norm_split_kernel<<<dim3((unsigned) n_tok, HC), THREADS, 0, st>>>(m);
    else if (fast) gr_norm_fast_kernel<<<n_tok, THREADS, 0, st>>>(m);
    else gr_norm_multi_kernel<<<n_tok, THREADS, 0, st>>>(m);
#else
    if (variant >= kHcSplit) gr_norm_split_kernel<<<dim3((unsigned) n_tok, HC), THREADS, 0, st>>>(m);
    else gr_norm_multi_kernel<<<n_tok, THREADS, 0, st>>>(m);
#endif
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, (void*) st);
    const bool staged = variant >= kHcStaged;
    // STRATA_HC_PACK=1: the staged read's packed arm (the staged variant, all three matrices packed).  The
    // small-CTA and reuse CTAs have their own kernels and stay on the plain bytes.
    const bool packed = staged && variant == kHcStaged && pack_on() && m.a[0].pack_down.lows != nullptr &&
                        m.a[0].pack_up.lows != nullptr &&
                        (m.a[0].w_inject == nullptr || m.a[0].pack_inject.lows != nullptr);
    int tv = 2560;
    const int chunk_tok_base = down_chunk(staged, &tv);
    // The LDS-accumulator kernel (STRATA_HC_SPLIT=9) carries more LDS per token than staged, so its launches
    // slice to what the card fits for its own shape (5 tokens on a 64 KiB card).
    const int lds_cap = variant == kHcLdsAccum ? down_chunk_lds() : chunk_tok_base;
    const int chunk_tok = lds_cap < chunk_tok_base ? lds_cap : chunk_tok_base;
    const size_t per_tok = (staged ? (size_t) 2 * H_TILE : (size_t) tv) * sizeof(float);
    static const bool no_multi_gr = [] {
        const char* no = std::getenv("STRATA_NO_MULTI_GR");
        return no != nullptr && no[0] != '\0' && no[0] != '0';
    }();
    const bool exact_t = !no_multi_gr;
    for (int c0 = 0; c0 < n_tok; c0 += chunk_tok) {
        const int ct = n_tok - c0 < chunk_tok ? n_tok - c0 : chunk_tok;
        GrMulti c{};
        if (ct == n_tok) {
            c = m;
        } else {
            c.xn = m.xn + (size_t) c0 * D;
            c.T = ct;
            for (int k = 0; k < ct; ++k) c.a[k] = m.a[c0 + k];
        }
        const size_t smem = (size_t) ct * per_tok;
        // #443, STRATA_GR_DOWN_MAX4: launches of up to 4 tokens hold 4 tokens' sums per thread instead of
        // kFusedGrMaxT - the same tile, block size, accumulation order and plain/split/staged path, so the same bits
        // (decided once per device: this runs per layer when decode is not captured)
        const bool max4 = ct <= 4 && gr_down_max4();
        if (staged && (variant == kHcSmallCta || variant == kHcReuseTwoRows)) {
            const bool reuse_rows = variant == kHcReuseTwoRows;
            const unsigned grid = (reuse_rows ? HC_REUSE_BLOCKS : HC_SMALL_CTA_BLOCKS) + 1;
            const size_t cta_smem = (size_t) ct * 2 * H_TILE * sizeof(float);
            if (reuse_rows) {
                if (exact_t && ct == 1) gr_down_reuse_two_rows_kernel<1><<<grid, HC_REUSE_THREADS, cta_smem, st>>>(c);
                else if (exact_t && ct == 2) gr_down_reuse_two_rows_kernel<2><<<grid, HC_REUSE_THREADS, cta_smem, st>>>(c);
                else if (exact_t && ct == 3) gr_down_reuse_two_rows_kernel<3><<<grid, HC_REUSE_THREADS, cta_smem, st>>>(c);
                else if (exact_t && ct == 4) gr_down_reuse_two_rows_kernel<4><<<grid, HC_REUSE_THREADS, cta_smem, st>>>(c);
                else if (exact_t && ct == 5) gr_down_reuse_two_rows_kernel<5><<<grid, HC_REUSE_THREADS, cta_smem, st>>>(c);
                else if (exact_t && ct == 6) gr_down_reuse_two_rows_kernel<6><<<grid, HC_REUSE_THREADS, cta_smem, st>>>(c);
                else if (exact_t && ct == 7) gr_down_reuse_two_rows_kernel<7><<<grid, HC_REUSE_THREADS, cta_smem, st>>>(c);
                else gr_down_reuse_two_rows_kernel<8><<<grid, HC_REUSE_THREADS, cta_smem, st>>>(c);
            } else {
                if (exact_t && ct == 1) gr_down_staged_kernel<1, true, HC_SMALL_CTA_THREADS><<<grid, HC_SMALL_CTA_THREADS, cta_smem, st>>>(c);
                else if (exact_t && ct == 2) gr_down_staged_kernel<2, true, HC_SMALL_CTA_THREADS><<<grid, HC_SMALL_CTA_THREADS, cta_smem, st>>>(c);
                else if (exact_t && ct == 3) gr_down_staged_kernel<3, true, HC_SMALL_CTA_THREADS><<<grid, HC_SMALL_CTA_THREADS, cta_smem, st>>>(c);
                else if (exact_t && ct == 4) gr_down_staged_kernel<4, true, HC_SMALL_CTA_THREADS><<<grid, HC_SMALL_CTA_THREADS, cta_smem, st>>>(c);
                else if (exact_t && ct == 5) gr_down_staged_kernel<5, true, HC_SMALL_CTA_THREADS><<<grid, HC_SMALL_CTA_THREADS, cta_smem, st>>>(c);
                else if (exact_t && ct == 6) gr_down_staged_kernel<6, true, HC_SMALL_CTA_THREADS><<<grid, HC_SMALL_CTA_THREADS, cta_smem, st>>>(c);
                else if (exact_t && ct == 7) gr_down_staged_kernel<7, true, HC_SMALL_CTA_THREADS><<<grid, HC_SMALL_CTA_THREADS, cta_smem, st>>>(c);
                else gr_down_staged_kernel<8, true, HC_SMALL_CTA_THREADS><<<grid, HC_SMALL_CTA_THREADS, cta_smem, st>>>(c);
            }
        } else if (staged && (variant == kHcRegisterPipe || variant == kHcRegisterHalf ||
                              variant == kHcRegisterPipeHalf)) {
            // opt-in (STRATA_HC_SPLIT=6/7, env 10 for the combination): the staged grid, CTA size and
            // shared-memory contract; the tile schedule is the kernel's own.  A launch of exactly ct tokens is its
            // own instantiation, as staged.
            const bool pipe = variant == kHcRegisterPipe;
            const bool pipe_half = variant == kHcRegisterPipeHalf;
            const unsigned grid = DOWN_BLOCKS + 1;
            const size_t cta_smem = (size_t) ct * 2 * H_TILE * sizeof(float);
            // A launch of ct <= 4 tokens is its own instantiation; past that the tuple's live slots push the
            // exact kernels over the 256-VGPR budget on gfx1201 (measured: pipe spills from MAX_T 5, half at
            // MAX_T 8), so those launches take the generic kernel, which measures vgpr_spill 0 and
            // private_segment_fixed_size 0 for both.  The runtime-T bounds change the codegen, not the sums.
            if (pipe) {
                if (exact_t && ct == 1) gr_down_register_pipe_kernel<1, true><<<grid, THREADS, cta_smem, st>>>(c);
                else if (exact_t && ct == 2) gr_down_register_pipe_kernel<2, true><<<grid, THREADS, cta_smem, st>>>(c);
                else if (exact_t && ct == 3) gr_down_register_pipe_kernel<3, true><<<grid, THREADS, cta_smem, st>>>(c);
                else if (exact_t && ct == 4) gr_down_register_pipe_kernel<4, true><<<grid, THREADS, cta_smem, st>>>(c);
                else gr_down_register_pipe_kernel<kFusedGrMaxT, false><<<grid, THREADS, cta_smem, st>>>(c);
            } else if (pipe_half) {
                if (exact_t && ct == 1) gr_down_register_pipe_half_kernel<1, true><<<grid, THREADS, cta_smem, st>>>(c);
                else if (exact_t && ct == 2) gr_down_register_pipe_half_kernel<2, true><<<grid, THREADS, cta_smem, st>>>(c);
                else if (exact_t && ct == 3) gr_down_register_pipe_half_kernel<3, true><<<grid, THREADS, cta_smem, st>>>(c);
                else if (exact_t && ct == 4) gr_down_register_pipe_half_kernel<4, true><<<grid, THREADS, cta_smem, st>>>(c);
                else gr_down_register_pipe_half_kernel<kFusedGrMaxT, false><<<grid, THREADS, cta_smem, st>>>(c);
            } else {
                if (exact_t && ct == 1) gr_down_register_half_kernel<1, true><<<grid, THREADS, cta_smem, st>>>(c);
                else if (exact_t && ct == 2) gr_down_register_half_kernel<2, true><<<grid, THREADS, cta_smem, st>>>(c);
                else if (exact_t && ct == 3) gr_down_register_half_kernel<3, true><<<grid, THREADS, cta_smem, st>>>(c);
                else if (exact_t && ct == 4) gr_down_register_half_kernel<4, true><<<grid, THREADS, cta_smem, st>>>(c);
                else gr_down_register_half_kernel<kFusedGrMaxT, false><<<grid, THREADS, cta_smem, st>>>(c);
            }
        } else if (staged && variant == kHcRowSplit) {
            // opt-in (STRATA_HC_SPLIT=8): the row split over 81 blocks; the same shared-memory contract as staged
            const unsigned grid = RS_BLOCKS + 1;
            const size_t cta_smem = (size_t) ct * 2 * H_TILE * sizeof(float);
            if (exact_t && ct == 1) gr_down_row_split_kernel<1, true><<<grid, THREADS, cta_smem, st>>>(c);
            else if (exact_t && ct == 2) gr_down_row_split_kernel<2, true><<<grid, THREADS, cta_smem, st>>>(c);
            else if (exact_t && ct == 3) gr_down_row_split_kernel<3, true><<<grid, THREADS, cta_smem, st>>>(c);
            else if (exact_t && ct == 4) gr_down_row_split_kernel<4, true><<<grid, THREADS, cta_smem, st>>>(c);
            else if (exact_t && ct == 5) gr_down_row_split_kernel<5, true><<<grid, THREADS, cta_smem, st>>>(c);
            else if (exact_t && ct == 6) gr_down_row_split_kernel<6, true><<<grid, THREADS, cta_smem, st>>>(c);
            else gr_down_row_split_kernel<kFusedGrMaxT, false><<<grid, THREADS, cta_smem, st>>>(c);
        } else if (staged && variant == kHcLdsAccum) {
            // opt-in (STRATA_HC_SPLIT=9): the staged grid, CTA size and tile schedule; the per-token sums live in
            // dynamic LDS.  The launch carries the accumulator row (THREADS floats per token) on top of the two
            // staged tiles, and the chunk loop above sliced ct to what the card fits for that shape.
            const unsigned grid = DOWN_BLOCKS + 1;
            const size_t cta_smem = (size_t) ct * (2 * H_TILE + THREADS) * sizeof(float);
            if (exact_t && ct == 1) gr_down_lds_accum_kernel<1, true><<<grid, THREADS, cta_smem, st>>>(c);
            else if (exact_t && ct == 2) gr_down_lds_accum_kernel<2, true><<<grid, THREADS, cta_smem, st>>>(c);
            else if (exact_t && ct == 3) gr_down_lds_accum_kernel<3, true><<<grid, THREADS, cta_smem, st>>>(c);
            else if (exact_t && ct == 4) gr_down_lds_accum_kernel<4, true><<<grid, THREADS, cta_smem, st>>>(c);
            else if (exact_t && ct == 5) gr_down_lds_accum_kernel<5, true><<<grid, THREADS, cta_smem, st>>>(c);
            else if (exact_t && ct == 6) gr_down_lds_accum_kernel<6, true><<<grid, THREADS, cta_smem, st>>>(c);
            else gr_down_lds_accum_kernel<kFusedGrMaxT, false><<<grid, THREADS, cta_smem, st>>>(c);
        } else if (packed) {
            // the staged grid, CTA size and shared-memory contract; only the weight bytes arrive packed
            const size_t cta_smem = (size_t) ct * 2 * H_TILE * sizeof(float);
            if (exact_t && ct == 1) gr_down_staged_packed_kernel<1, true><<<DOWN_BLOCKS + 1, THREADS, cta_smem, st>>>(c);
            else if (exact_t && ct == 2) gr_down_staged_packed_kernel<2, true><<<DOWN_BLOCKS + 1, THREADS, cta_smem, st>>>(c);
            else if (exact_t && ct == 3) gr_down_staged_packed_kernel<3, true><<<DOWN_BLOCKS + 1, THREADS, cta_smem, st>>>(c);
            else if (exact_t && ct == 4) gr_down_staged_packed_kernel<4, true><<<DOWN_BLOCKS + 1, THREADS, cta_smem, st>>>(c);
            else if (exact_t && ct == 5) gr_down_staged_packed_kernel<5, true><<<DOWN_BLOCKS + 1, THREADS, cta_smem, st>>>(c);
            else if (exact_t && ct == 6) gr_down_staged_packed_kernel<6, true><<<DOWN_BLOCKS + 1, THREADS, cta_smem, st>>>(c);
            else gr_down_staged_packed_kernel<kFusedGrMaxT, false><<<DOWN_BLOCKS + 1, THREADS, cta_smem, st>>>(c);
        } else if (staged) {
            // #783 PR-g (stuchapin909): a launch of exactly ct <= 6 tokens is its own instantiation, the loop bounds
            // are compile-time (the same sums in the same order); STRATA_NO_MULTI_GR=1 keeps the generic kernels
            if (exact_t && ct == 1) gr_down_staged_kernel<1, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else if (exact_t && ct == 2) gr_down_staged_kernel<2, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else if (exact_t && ct == 3) gr_down_staged_kernel<3, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else if (exact_t && ct == 4) gr_down_staged_kernel<4, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else if (exact_t && ct == 5) gr_down_staged_kernel<5, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else if (exact_t && ct == 6) gr_down_staged_kernel<6, true><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else if (max4) gr_down_staged_kernel<4><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else gr_down_staged_kernel<><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
        } else if (tv == 1280) {
            if (max4) gr_down_multi_kernel<1280, 4><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else gr_down_multi_kernel<1280><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
        } else {
            if (max4) gr_down_multi_kernel<2560, 4><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
            else gr_down_multi_kernel<2560><<<DOWN_BLOCKS + 1, THREADS, smem, st>>>(c);
        }
    }
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, (void*) st);
#if STRATA_GR_FAST_BUILD
    if (fast) gr_up_fast_kernel<<<UPM_BLOCKS, THREADS, 0, st>>>(m);
    else
#endif
    if (packed) {
        switch (no_multi_gr ? kFusedGrMaxT : n_tok) {
            case 1: gr_up_multi_packed_kernel<1, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
            case 2: gr_up_multi_packed_kernel<2, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
            case 3: gr_up_multi_packed_kernel<3, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
            case 4: gr_up_multi_packed_kernel<4, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
            case 5: gr_up_multi_packed_kernel<5, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
            case 6: gr_up_multi_packed_kernel<6, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
            default: gr_up_multi_packed_kernel<kFusedGrMaxT, false><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
        }
        return;
    }
    switch (no_multi_gr ? kFusedGrMaxT : n_tok) {
        case 1: gr_up_multi_kernel<1, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
        case 2: gr_up_multi_kernel<2, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
        case 3: gr_up_multi_kernel<3, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
        case 4: gr_up_multi_kernel<4, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
        case 5: gr_up_multi_kernel<5, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
        case 6: gr_up_multi_kernel<6, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
        default: gr_up_multi_kernel<kFusedGrMaxT, false><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
    }
}

/// STRATA_HC_SPLIT: 2 = staged (the promote escape hatch), unset = the promoted default - the register-half
/// read (7) on a card whose check latched it, staged on a card that did not; see fused_gr_check.
/// 1 = split, 3 = small CTA, 5 = 2 rows/warp (4-warp CTA), 0 = plain,
/// 6 = staged with the next tile in a register tuple, 7 = the same with the tuple in two halves, 8 = the row
/// split, 9 = staged with the per-token accumulators in dynamic LDS, 10 = the register pipeline with both
/// register arrays in halves (the 6/7 combination).  6, 7, 8, 9 and 10 are opt-in and self-checked against
/// staged before use.  Any other value takes the staged default: the first software-pipeline attempt (a
/// runtime-indexed slot array) is not dispatched - it lowers to per-thread private scratch on gfx1201, not the
/// VGPR slots it meant to use.
///
/// Syntax: the parser reads the value's leading characters, so every one-digit value above is one character.
/// 10 is the first two-character value: it is matched as the exact string "10" (the two leading characters, so
/// "10x" reads as 10) BEFORE the one-digit '1', which would otherwise win.  A value that matches nothing takes
/// the staged default, as before.
int env_variant() {
    const char* e = std::getenv("STRATA_HC_SPLIT");
    if (e == nullptr || e[0] == '\0') return kHcStaged;
    if (e[0] == '1' && e[1] == '0') return kHcRegisterPipeHalf;   // two characters, checked before the '1' below
    if (e[0] == '0') return kHcPlain;
    if (e[0] == '1') return kHcSplit;
    if (e[0] == '3') return kHcSmallCta;
    if (e[0] == '5') return kHcReuseTwoRows;
    if (e[0] == '6') return kHcRegisterPipe;
    if (e[0] == '7') return kHcRegisterHalf;
    if (e[0] == '8') return kHcRowSplit;
    if (e[0] == '9') return kHcLdsAccum;
    return kHcStaged;
}

// The half-tuple down kernel (STRATA_HC_SPLIT=7) is clamped per token count: T <= kHcHalfWinMaxT runs the half
// kernel, T above it runs staged.  Measured on the R9700 at T=1..8 (paired fresh processes, 200 iters, RESULTS.md),
// the half kernel wins or ties at every T, so the boundary is kFusedGrMaxT and the clamp is a no-op safety net:
// it only matters if kFusedGrMaxT ever grows past the measured range.  Only variant 7 is clamped, so the staged
// default and every other opt-in path are returned unchanged.  The self-test is NOT clamped - it still checks the
// half kernel against staged at every T.
constexpr int kHcHalfWinMaxT = kFusedGrMaxT;
// The pipe+half combination (STRATA_HC_SPLIT=10) is clamped the same way, with its own boundary.  The clamp's
// reference is the default read it would fall back to, so the boundary answers "above which T does this variant
// stop beating staged": measured at T=1..8 (paired fresh processes, 200 iters, two rounds, RESULTS.md P5) it
// beats staged at T=2..8 (-1.6 to -3.6 us) and ties at T=1, so the boundary is kFusedGrMaxT and the clamp is a
// no-op safety net, as variant 7's.  (How it ranks against variant 7 is a promote question, not a clamp
// question; P5 records it - the combination does not beat 7, so 7 stays the promote candidate.)  The boundary is
// a dispatch choice only - the self-test is not clamped, so latching on 10 still means "bit for bit equal to
// staged at every T".
constexpr int kHcPipeHalfWinMaxT = kFusedGrMaxT;
int variant_for_T(int v, int T) {
    if (v == kHcRegisterHalf && T > kHcHalfWinMaxT) return kHcStaged;
    if (v == kHcRegisterPipeHalf && T > kHcPipeHalfWinMaxT) return kHcStaged;
    return v;
}

// P2 opt-in (STRATA_HC_GRAPH=1): replay the norm->down->up launch sequence as a captured HIP graph.  The graph is
// keyed by everything the launches read (variant, token count, the scratch/stream/stamp identities, and every
// per-token pointer and flag), so a replay runs exactly the same kernels on exactly the same arguments - the same
// bits.  A key is captured once and replayed on repeats; keys that never repeat (the production shape, where the
// token buffers move every call) fill the cache to its cap and the plain launch takes over, so the gate is a
// bench/loop tool, not a production default.  Gate off: the launch path is byte-identical to before.
namespace {
bool hc_graph_env() {
    static const bool on = [] { const char* e = std::getenv("STRATA_HC_GRAPH"); return e != nullptr && std::atoi(e) != 0; }();
    return on;
}
struct GraphEntry { unsigned long long key; cudaGraphExec_t exec; };
std::vector<GraphEntry>& graph_cache() { static std::vector<GraphEntry> v; return v; }
std::mutex& graph_mu() { static std::mutex mu; return mu; }
unsigned long long args_key(const GrMulti& m, int variant, const unsigned long long* stamp_buf, int stamp_i0) {
    unsigned long long h = 1469598125493960393ull;
    auto mix = [&h](unsigned long long v) { h = (h ^ v) * 1099511628211ull; };
    mix((unsigned long long) variant);
    mix((unsigned long long) m.T);
#if STRATA_GR_FAST_BUILD
    mix((unsigned long long) gr_fast());   // the fast arm launches different kernels under the same pointers
#endif   // the fast arm launches different kernels under the same pointers
    mix((unsigned long long) (uintptr_t) m.xn);
    mix((unsigned long long) (uintptr_t) stamp_buf);
    mix((unsigned long long) stamp_i0);
    for (int t = 0; t < m.T; ++t) {
        const FusedGrArgs& a = m.a[t];
        mix((unsigned long long) (uintptr_t) a.R);      mix((unsigned long long) (uintptr_t) a.R_out);
        mix((unsigned long long) a.apply);               mix((unsigned long long) (uintptr_t) a.bo_prev);
        mix((unsigned long long) (uintptr_t) a.inj_prev); mix((unsigned long long) (uintptr_t) a.w_norm);
        mix((unsigned long long) (uintptr_t) a.w_down); mix((unsigned long long) (uintptr_t) a.w_up);
        mix((unsigned long long) (uintptr_t) a.w_inject); mix((unsigned long long) (uintptr_t) a.lo);
        mix((unsigned long long) (uintptr_t) a.rs);     mix((unsigned long long) (uintptr_t) a.inject_out);
        mix((unsigned long long) (uintptr_t) a.mixed);
        // the packed arm reads the same matrices through the pack pointers - they belong in the key too
        mix((unsigned long long) (uintptr_t) a.pack_down.lows);
        mix((unsigned long long) (uintptr_t) a.pack_up.lows);
        mix((unsigned long long) (uintptr_t) a.pack_inject.lows);
    }
    return h;
}
// true: the read is enqueued (replayed, or captured-and-launched on this first call).  false: capture is not
// available for this call - the caller must plain-launch.  A capture failure latches the gate off for good.
bool graph_replay(const GrMulti& m, int variant, cudaStream_t st, unsigned long long* stamp_buf, int stamp_i0) {
    const unsigned long long key = args_key(m, variant, stamp_buf, stamp_i0);
    std::lock_guard<std::mutex> lk(graph_mu());
    auto& cache = graph_cache();
    for (const GraphEntry& e : cache)
        if (e.key == key) return cudaGraphLaunch(e.exec, st) == cudaSuccess;
    static std::atomic<bool> broken{false};
    if (broken.load() || cache.size() >= 32) return false;   // keys never repeat here: plain launches
    cudaGraph_t g = nullptr;
    cudaError_t e = cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
    if (e == cudaSuccess) {
        launch_multi(m, variant, st, stamp_buf, stamp_i0);
        e = cudaStreamEndCapture(st, &g);
    }
    if (e != cudaSuccess || g == nullptr) {
        if (g) cudaGraphDestroy(g);
        cudaGetLastError();   // clear the stale error so the caller's own check sees this call, not the capture
        if (!broken.exchange(true))
            std::fprintf(stderr, "fused_gr: STRATA_HC_GRAPH capture failed (%s); plain launches from here on\n",
                         cudaGetErrorString(e));
        return false;
    }
    cudaGraphExec_t exec = nullptr;
    e = cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0);
    cudaGraphDestroy(g);
    if (e != cudaSuccess || exec == nullptr) {
        cudaGetLastError();
        broken.store(true);
        return false;
    }
    cache.push_back({key, exec});
    return cudaGraphLaunch(exec, st) == cudaSuccess;
}
}  // namespace

// per device: the variant `fused_gr_check` chose (0 = not checked yet)
std::atomic<int> g_variant[64];
// `g_pack_check` stores the final self-test decision; `pack_on` consults it outside the self-test. The device-view
// descriptors themselves travel by value in GrMulti, never as a pointer to host memory.
std::atomic<int> g_pack_check[64];
bool env_pack() {
    const char* e = std::getenv("STRATA_HC_PACK");
    return e != nullptr && std::atoi(e) != 0;
}
bool g_force_pack = false;   // the self-test runs the packed arm whatever the env says

bool pack_on() {
    int dev = 0;
    cudaGetDevice(&dev);
    if (g_force_pack) return true;  // self-test-only: exercise the packed candidate before committing a decision
    const int checked = dev >= 0 && dev < 64 ? g_pack_check[dev].load() : 0;
    if (checked != 0) return checked == 1;
    return env_pack();
}


#if STRATA_GR_FAST_BUILD
// ---- AMD fast path (STRATA_GR_FAST, default 1; 0 = the old kernels): the same arithmetic per value in the same order, latency
// hidden.  gr_norm: one block per token (as before) but a thread's 10 float4 of R / bo / w_norm are loaded
// before any is used, and xn stays in registers until rs is known (was: store, then re-read the 40 KB).
// gr_up: a warp's 8 rows of w_up and their epilogue inputs are loaded before the dots (was: one row's load chain
// after another).
constexpr int NQ = D / (THREADS * 4);   // 10 float4 per thread
static_assert(D % (THREADS * 4) == 0, "whole float4 per thread");
__global__ void __launch_bounds__(THREADS) gr_norm_fast_kernel(GrMulti m) {
    __shared__ float part[WARPS][HC];
    __shared__ float s_rs[HC];
    const FusedGrArgs& a = m.a[blockIdx.x];
    float* xn = m.xn + (size_t) blockIdx.x * D;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    float4 r[NQ], g[NQ];
#pragma unroll
    for (int k = 0; k < NQ; ++k) {
        const int i = t * 4 + k * THREADS * 4;
        r[k] = *reinterpret_cast<const float4*>(a.R + i);
        g[k] = *reinterpret_cast<const float4*>(a.w_norm + i);
    }
    if (a.apply) {
#pragma unroll
        for (int k = 0; k < NQ; ++k) {
            const int i = t * 4 + k * THREADS * 4, c = i / N, d = i - c * N;
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r[k].x = fmaf(b.x, gw[c], r[k].x); r[k].y = fmaf(b.y, gw[c], r[k].y);
            r[k].z = fmaf(b.z, gw[c], r[k].z); r[k].w = fmaf(b.w, gw[c], r[k].w);
        }
    }
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int k = 0; k < NQ; ++k) {
        const int i = t * 4 + k * THREADS * 4, c = i / N;
        const float sq = r[k].x * r[k].x + r[k].y * r[k].y + r[k].z * r[k].z + r[k].w * r[k].w;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        r[k] = make_float4(r[k].x * g[k].x, r[k].y * g[k].y, r[k].z * g[k].z, r[k].w * g[k].w);
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    __syncthreads();
    if (t < HC) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w][t];
        s_rs[t] = rsqrtf(s / (float) N + a.eps);
        a.rs[t] = s_rs[t];
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < NQ; ++k) {
        const int i = t * 4 + k * THREADS * 4;
        const float sr = s_rs[i / N];
        *reinterpret_cast<float4*>(xn + i) = make_float4(r[k].x * sr, r[k].y * sr, r[k].z * sr, r[k].w * sr);
    }
}
// gr_up: 8 lanes per row instead of a 32-lane warp.  Lane j holds the old lanes j, j+8, j+16, j+24 (and chunk
// 32+j, the old lane j's second chunk): the xor tree's stages 16 and 8 are its own adds (a + b == b + a), stages
// 4, 2, 1 are shuffles inside the 8 - the same tree, bitwise the same sum, 3 shuffles a token instead of 5 and no
// idle lanes on the tail chunks.  A block's 64 rows are 2 passes of 32 groups, their weights loaded up front.
__device__ __forceinline__ float xor8(float v) {
#pragma unroll
#if defined(__HIPCC__)
    for (int o = 4; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
#else
    for (int o = 4; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
#endif
    return v;
}
__global__ void __launch_bounds__(THREADS) gr_up_fast_kernel(GrMulti m) {
    __shared__ __align__(16) float lo[kFusedGrMaxT][LR];
    __shared__ float g[kFusedGrMaxT][HC][UPM_COLS];
    const int t = threadIdx.x, j = t & 7, grp = t >> 3;
    const int T = m.T;
    const int d0 = blockIdx.x * UPM_COLS;
    static_assert(LR == 40 * 8 && HC * UPM_COLS == 64 && THREADS == 256, "geometry");
    uint4 w[2][5];
    float rv[2] = {0.0f, 0.0f}, wn[2] = {0.0f, 0.0f}, rsc[2] = {0.0f, 0.0f}, bo[2] = {0.0f, 0.0f}, ip[2] = {0.0f, 0.0f};
    const bool apply = j < T && m.a[j < T ? j : 0].apply;
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int r = grp + 32 * p, c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
#pragma unroll
        for (int q = 0; q < 4; ++q) w[p][q] = __ldg(w4 + j + 8 * q);
        w[p][4] = __ldg(w4 + 32 + j);
        if (j < T) {
            const FusedGrArgs& a = m.a[j];
            rv[p] = a.R[i];
            wn[p] = a.w_norm[i];
            rsc[p] = a.rs[c];
            if (apply) { bo[p] = a.bo_prev[d0 + dd]; ip[p] = a.inj_prev[c]; }
        }
    }
    for (int i = t; i < T * LR; i += THREADS) lo[i / LR][i % LR] = m.a[i / LR].lo[i % LR];
    __syncthreads();
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int r = grp + 32 * p, c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) {
            if (k >= T) break;
            const float* l = lo[k];
            const float p0 = dot8(w[p][0], l + j * 8) + dot8(w[p][4], l + (32 + j) * 8);   // old lane j
            const float p1 = dot8(w[p][1], l + (j + 8) * 8);                               // old lane j + 8
            const float p2 = dot8(w[p][2], l + (j + 16) * 8);                              // old lane j + 16
            const float p3 = dot8(w[p][3], l + (j + 24) * 8);                              // old lane j + 24
            const float s = xor8((p0 + p2) + (p1 + p3));   // stage 16: (j, j+16), (j+8, j+24); stage 8; then 4, 2, 1
            if (j == k) mine = s;
        }
        if (j < T) {
            float x0 = rv[p];
            if (apply) {
                x0 = fmaf(bo[p], 2.0f * sigmoidf_(ip[p] / (float) HC), x0);
                m.a[j].R_out[i] = x0;
            }
            const float x = x0 * wn[p] * rsc[p];
            g[j][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[k][c][col];
        m.a[k].mixed[d0 + col] = s / (float) HC;
    }
    if (m.a[0].q8_mixed != nullptr) gr_q8_tail(m, d0);   // S26 STRATA_QFUSE, as gr_up_multi_kernel
}
#endif

#if defined(STRATA_HIP_GFX906)
// ---- AMD: `gr_down_multi` split along K.  The CUDA kernel is one warp per row of w_down (320 + 4 rows), 41
// blocks: on a 60-CU gfx906 that left most of the card idle and read the 6.5 MB matrix at ~100 GB/s.  Here a block
// is 8 wavefronts = 8 rows of one K-slice of 2048 (the slice of every token's xn staged in LDS once), 41 x 5
// blocks, and a second kernel adds the 5 slice sums in a fixed order and runs the epilogue.  A token's result does
// not depend on how many tokens share the window (each column is reduced on its own).
constexpr int GS_SL = 2048;                   // K per slice
constexpr int GS_S = D / GS_SL;               // 5 slices
constexpr int GS_ROWS = LR + HC;              // 324: the down rows, then the inject rows
constexpr int GS_WAVES = 8;
static_assert(D % GS_SL == 0 && GS_SL == 64 * 8 * 4, "a lane takes 4 chunks of 8 per slice");
// the slice sums: a module-scope device array is per device by construction, and needs no allocation during a
// graph capture
__device__ float g_gr_part[GS_S * kFusedGrMaxT * GS_ROWS];
__global__ void __launch_bounds__(GS_WAVES * 64) gr_down_split_kernel(GrMulti m) {
    float* part = g_gr_part;
    extern __shared__ __align__(16) float tile[];   // [T][GS_SL]
    const int t = threadIdx.x, lane = t & 63, wave = t >> 6;
    const int T = m.T, sl = blockIdx.y;
    const bool inject_block = blockIdx.x == DOWN_BLOCKS;
    const int row = inject_block ? wave : blockIdx.x * GS_WAVES + wave;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || wave >= HC));
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) (active ? row : 0) * D;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow) + sl * (GS_SL / 8);
    uint4 wv[4];
    if (active) {
#pragma unroll
        for (int q = 0; q < 4; ++q) wv[q] = __ldg(w4 + lane + 64 * q);
    }
    const float4* src4 = reinterpret_cast<const float4*>(m.xn);
    float4* tile4 = reinterpret_cast<float4*>(tile);
    for (int i = t; i < T * (GS_SL / 4); i += GS_WAVES * 64) {
        const int k = i / (GS_SL / 4), off = i - k * (GS_SL / 4);
        tile4[i] = src4[((size_t) k * D + (size_t) sl * GS_SL) / 4 + off];
    }
    __syncthreads();
    if (!active) return;
    float acc[kFusedGrMaxT];
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) acc[k] = 0.0f;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int j = lane + 64 * q;
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k)
            if (k < T) acc[k] += dot8(wv[q], tile + k * GS_SL + j * 8);
    }
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) {
        if (k >= T) break;
        float v = acc[k];
#pragma unroll
        for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off, 64);
        if (lane == 0) part[((size_t) sl * kFusedGrMaxT + k) * GS_ROWS + (inject_block ? LR + row : row)] = v;
    }
}
__global__ void gr_down_finish_kernel(GrMulti m) {
    const float* part = g_gr_part;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = i / GS_ROWS, r = i - k * GS_ROWS;
    if (k >= m.T) return;
    const bool inject = r >= LR;
    if (inject && m.a[0].w_inject == nullptr) return;
    float s = 0.0f;
#pragma unroll
    for (int sl = 0; sl < GS_S; ++sl) s += part[((size_t) sl * kFusedGrMaxT + k) * GS_ROWS + r];
    if (inject) {
        m.a[k].inject_out[r - LR] = s;
    } else {
        const float x = s / (float) HC;
        m.a[k].lo[r] = x / (1.0f + __expf(-x));
    }
}

#endif

// ================================ S23 experiment (STRATA_HC_Q8=1): the read with the GGUF's Q8_0 projections ======
// The pack holds the hyper-connection projections as BF16 (iq_pack.py --compat-bf16 rounds the GGUF's Q8_0 values to
// BF16: 1.20 GiB per verify step); this read takes the Q8_0 bytes as stored (0.65 GiB - and the GGUF's own values,
// not a rounding of them).  The layout is the stream split of the v3 read above, at 4 chunks per stream: `down` runs
// on (10 groups of 32 rows + the inject rows) x 16 chunks of 640 columns = 176 blocks, each staging its chunk of
// R' * w_norm (unscaled) for the T tokens and writing the chunk's partial dots and sums of squares; `up` sums them
// per stream in a fixed order, applies rs, SwiGLU-free silu, and runs the default epilogue.  Another summation order
// and other weights than the default read: an output-changing step (measured, quality-checked).
constexpr int Q8B = 34;                             // Q8_0 block: fp16 d + 32 int8
constexpr int Q8_RPW = 4;                           // down rows per warp
constexpr int Q8_RG = LR / (WARPS * Q8_RPW);        // 10 row groups (+1: the inject rows)
constexpr int Q8_KC = 640, Q8_NKC = D / Q8_KC;      // 16 chunks, 4 per stream
constexpr int Q8_CPS = N / Q8_KC;                   // chunks per stream
constexpr int Q8_SPB = Q8_KC / 128;                 // 5 steps of 4 Q8_0 blocks (8 lanes x 4 values each)
__device__ __forceinline__ uint32_t q8_ld16(const uint8_t* p) { return *(const uint16_t*) p; }
__device__ __forceinline__ float q8_half(uint32_t h) { return __half2float(__ushort_as_half((unsigned short) h)); }
__device__ __forceinline__ float4 q8_four(uint32_t q, float d) {
    return make_float4(d * (float) (int8_t) (q & 255u), d * (float) (int8_t) ((q >> 8) & 255u),
                       d * (float) (int8_t) ((q >> 16) & 255u), d * (float) (int8_t) (q >> 24));
}

template <int T>
__global__ void __launch_bounds__(THREADS) gr_down_q8_kernel(GrMulti m, float* __restrict__ part, float* __restrict__ ssg) {
    __shared__ __align__(16) float xs[T][Q8_KC];
    __shared__ float red[WARPS][T];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int rg = blockIdx.x, kc = blockIdx.y, c = kc / Q8_CPS;
    const bool inj = rg == Q8_RG;
    // S25: the inject rows are F32 in the GGUF with BF16-exact values, so by default they are read as the pack's BF16
    // (exact); a Q8_0 copy (STRATA_HC_Q8_INJECT=1) is the other option
    const bool inj_bf16 = inj && m.a[0].q8_inject == nullptr;
    const int nrows = inj ? (((m.a[0].q8_inject != nullptr || m.a[0].w_inject != nullptr) && warp < HC) ? 1 : 0) : Q8_RPW;
    const int row0 = inj ? warp : (rg * WARPS + warp) * Q8_RPW;
    const uint8_t* wb = inj ? m.a[0].q8_inject : m.a[0].q8_down;
    const int sub = lane & 7, bl = lane >> 3;       // lanes 8b..8b+7: one Q8_0 block, 4 values each
    uint32_t q[Q8_RPW][Q8_SPB];
    float dq[Q8_RPW][Q8_SPB];
#pragma unroll
    for (int r = 0; r < Q8_RPW; ++r) {
        if (r >= nrows || inj_bf16) break;
        const uint8_t* row = wb + (size_t) (row0 + r) * (D / 32) * Q8B + (size_t) kc * (Q8_KC / 32) * Q8B;
#pragma unroll
        for (int s = 0; s < Q8_SPB; ++s) {
            const uint8_t* blk = row + (4 * s + bl) * Q8B;
            dq[r][s] = q8_half(q8_ld16(blk));
            q[r][s] = q8_ld16(blk + 2 + 4 * sub) | q8_ld16(blk + 4 + 4 * sub) << 16;
        }
    }
    float ssp[T];
#pragma unroll
    for (int k = 0; k < T; ++k) {
        ssp[k] = 0.0f;
        const FusedGrArgs& a = m.a[k];
        const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
        for (int i = t; i < Q8_KC / 4; i += THREADS) {
            const int col = kc * Q8_KC + 4 * i;
            float4 r = *reinterpret_cast<const float4*>(a.R + col);
            if (a.apply) {
                const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + (col - c * N));
                r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y);
                r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
            }
            const float4 g = *reinterpret_cast<const float4*>(a.w_norm + col);
            ssp[k] += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
            *reinterpret_cast<float4*>(&xs[k][4 * i]) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
        }
    }
#pragma unroll
    for (int k = 0; k < T; ++k) {
        const float v = warp_sum(ssp[k]);
        if (lane == 0) red[warp][k] = v;
    }
    __syncthreads();
    if (rg == 0 && t < T) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += red[w][t];
        ssg[t * Q8_NKC + kc] = s;
    }
    if (nrows == 0) return;
#pragma unroll
    for (int r = 0; r < Q8_RPW; ++r) {
        if (r >= nrows) break;
        float acc[T];
#pragma unroll
        for (int k = 0; k < T; ++k) acc[k] = 0.0f;
#pragma unroll
        for (int s = 0; s < Q8_SPB; ++s) {
            const int v = 32 * (4 * s + bl) + 4 * sub;
            float4 w;
            if (inj_bf16) {
                const uint2 b = *reinterpret_cast<const uint2*>(m.a[0].w_inject + (size_t) row0 * D + (size_t) kc * Q8_KC + v);
                w = make_float4(__uint_as_float(b.x << 16), __uint_as_float(b.x & 0xffff0000u),
                                __uint_as_float(b.y << 16), __uint_as_float(b.y & 0xffff0000u));
            } else {
                w = q8_four(q[r][s], dq[r][s]);
            }
#pragma unroll
            for (int k = 0; k < T; ++k) {
                const float4 x = *reinterpret_cast<const float4*>(&xs[k][v]);
                acc[k] = fmaf(w.x, x.x, acc[k]); acc[k] = fmaf(w.y, x.y, acc[k]);
                acc[k] = fmaf(w.z, x.z, acc[k]); acc[k] = fmaf(w.w, x.w, acc[k]);
            }
        }
        const int prow = inj ? LR + warp : row0 + r;
#pragma unroll
        for (int k = 0; k < T; ++k) {
            const float v = warp_sum(acc[k]);
            if (lane == 0) part[((size_t) k * Q8_NKC + kc) * PR + prow] = v;
        }
    }
}

template <int T>
__global__ void __launch_bounds__(THREADS) gr_up_q8_kernel(GrMulti m, const float* __restrict__ part,
                                                           const float* __restrict__ ssg) {
    __shared__ __align__(16) float lo[T][LR];
    __shared__ float rsS[T][HC];
    __shared__ float g[T][HC][UPM_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int d0 = blockIdx.x * UPM_COLS;
    if (t < T * HC) {
        const int k = t / HC, c = t - k * HC;
        float ss = 0.0f;
#pragma unroll
        for (int j = 0; j < Q8_CPS; ++j) ss += ssg[k * Q8_NKC + c * Q8_CPS + j];
        const float r = rsqrtf(ss / (float) N + m.a[k].eps);
        rsS[k][c] = r;
        if (blockIdx.x == 0) m.a[k].rs[c] = r;
    }
    __syncthreads();
    for (int i = t; i < T * PR; i += THREADS) {
        const int k = i / PR, r = i - k * PR;
        if (r >= LR && (blockIdx.x != 0 || m.a[k].w_inject == nullptr)) continue;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            float p = 0.0f;
#pragma unroll
            for (int j = 0; j < Q8_CPS; ++j) p += part[((size_t) k * Q8_NKC + c * Q8_CPS + j) * PR + r];
            sum = fmaf(rsS[k][c], p, sum);
        }
        if (r < LR) {
            const float x = sum / (float) HC;
            lo[k][r] = x / (1.0f + __expf(-x));
        } else {
            m.a[k].inject_out[r - LR] = sum;
        }
    }
    __syncthreads();
    constexpr int RPW8 = HC * UPM_COLS / WARPS;     // 8 rows per warp
#pragma unroll 1
    for (int qq = 0; qq < RPW8; ++qq) {
        const int r = warp + qq * WARPS;
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint8_t* row = m.a[0].q8_up + (size_t) i * (LR / 32) * Q8B;
        float4 w[3];
#pragma unroll
        for (int s = 0; s < 3; ++s) {   // lane: values 4 lane + 128 s (the third step: lanes 0-15)
            const int v = 4 * lane + 128 * s;
            if (v < LR) {
                const uint8_t* blk = row + (v >> 5) * Q8B;
                w[s] = q8_four(q8_ld16(blk + 2 + (v & 31)) | q8_ld16(blk + 4 + (v & 31)) << 16, q8_half(q8_ld16(blk)));
            } else {
                w[s] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        float rv = 0.0f, wn = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
            rv = a.R[i];
            wn = a.w_norm[i];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < T; ++k) {
            float acc = 0.0f;
#pragma unroll
            for (int s = 0; s < 3; ++s) {
                const int v = 4 * lane + 128 * s;
                if (v < LR) {
                    const float4 x = *reinterpret_cast<const float4*>(&lo[k][v]);
                    acc = fmaf(w[s].x, x.x, acc); acc = fmaf(w[s].y, x.y, acc);
                    acc = fmaf(w[s].z, x.z, acc); acc = fmaf(w[s].w, x.w, acc);
                }
            }
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        if (lane < T) {
            if (apply) {
                rv = fmaf(bo, 2.0f * sigmoidf_(ip / (float) HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsS[lane][c];
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) sum += g[k][c][col];
        m.a[k].mixed[d0 + col] = sum / (float) HC;
    }
    if (m.a[0].q8_mixed != nullptr) gr_q8_tail(m, d0);   // S26 STRATA_QFUSE
}


// S26 STRATA_TSUM=1: gr_down_q8_kernel / gr_up_q8_kernel with each row's T per-token warp sums (and the sums of
// squares) as one transposed butterfly (s26_tsum.cuh): bitwise the same values (S26 ws4 harness, every output equal,
// T 1-4: 1.00 / 1.03 / 1.03 / 1.15x per read). PF (all rows' weights in registers before the part / lo phase) was
// slower (0.92-1.03x) and is off.
using s26ts::tsum;
using s26ts::tsum_token;
using s26ts::tsum_lane;
using s26ts::pow2_ceil;
// gr_down_q8_kernel / gr_up_q8_kernel with every per-token warp sum done as one transposed butterfly per row (tsum)
template <int T, int RPW = Q8_RPW>
__global__ void __launch_bounds__(THREADS) gr_down_q8_fast_kernel(GrMulti m, float* __restrict__ part, float* __restrict__ ssg) {
    constexpr int P = pow2_ceil(T);
    constexpr int RG_ = LR / (WARPS * RPW);
    __shared__ __align__(16) float xs[T][Q8_KC];
    __shared__ float red[WARPS][T];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int rg = blockIdx.x, kc = blockIdx.y, c = kc / Q8_CPS;
    const bool inj = rg == RG_;
    const bool inj_bf16 = inj && m.a[0].q8_inject == nullptr;
    const int nrows = inj ? (((m.a[0].q8_inject != nullptr || m.a[0].w_inject != nullptr) && warp < HC) ? 1 : 0) : RPW;
    const int row0 = inj ? warp : (rg * WARPS + warp) * RPW;
    const uint8_t* wb = inj ? m.a[0].q8_inject : m.a[0].q8_down;
    const int sub = lane & 7, bl = lane >> 3;
    const int tk = tsum_token<P>(lane);
    const bool owner = lane == tsum_lane<P>(tk) && tk < T;
    uint32_t q[RPW][Q8_SPB];
    float dq[RPW][Q8_SPB];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        if (r >= nrows || inj_bf16) break;
        const uint8_t* row = wb + (size_t) (row0 + r) * (D / 32) * Q8B + (size_t) kc * (Q8_KC / 32) * Q8B;
#pragma unroll
        for (int s = 0; s < Q8_SPB; ++s) {
            const uint8_t* blk = row + (4 * s + bl) * Q8B;
            dq[r][s] = q8_half(q8_ld16(blk));
            q[r][s] = q8_ld16(blk + 2 + 4 * sub) | q8_ld16(blk + 4 + 4 * sub) << 16;
        }
    }
    float ssp[P];
#pragma unroll
    for (int k = 0; k < P; ++k) ssp[k] = 0.0f;
#pragma unroll
    for (int k = 0; k < T; ++k) {
        const FusedGrArgs& a = m.a[k];
        const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
        for (int i = t; i < Q8_KC / 4; i += THREADS) {
            const int col = kc * Q8_KC + 4 * i;
            float4 r = *reinterpret_cast<const float4*>(a.R + col);
            if (a.apply) {
                const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + (col - c * N));
                r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y);
                r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
            }
            const float4 g = *reinterpret_cast<const float4*>(a.w_norm + col);
            ssp[k] += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
            *reinterpret_cast<float4*>(&xs[k][4 * i]) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
        }
    }
    {
        const float v = tsum<P>(ssp, lane);
        if (owner) red[warp][tk] = v;
    }
    __syncthreads();
    if (rg == 0 && t < T) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += red[w][t];
        ssg[t * Q8_NKC + kc] = s;
    }
    if (nrows == 0) return;
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        if (r >= nrows) break;
        float acc[P];
#pragma unroll
        for (int k = 0; k < P; ++k) acc[k] = 0.0f;
#pragma unroll
        for (int s = 0; s < Q8_SPB; ++s) {
            const int v = 32 * (4 * s + bl) + 4 * sub;
            float4 w;
            if (inj_bf16) {
                const uint2 b = *reinterpret_cast<const uint2*>(m.a[0].w_inject + (size_t) row0 * D + (size_t) kc * Q8_KC + v);
                w = make_float4(__uint_as_float(b.x << 16), __uint_as_float(b.x & 0xffff0000u),
                                __uint_as_float(b.y << 16), __uint_as_float(b.y & 0xffff0000u));
            } else {
                w = q8_four(q[r][s], dq[r][s]);
            }
#pragma unroll
            for (int k = 0; k < T; ++k) {
                const float4 x = *reinterpret_cast<const float4*>(&xs[k][v]);
                acc[k] = fmaf(w.x, x.x, acc[k]); acc[k] = fmaf(w.y, x.y, acc[k]);
                acc[k] = fmaf(w.z, x.z, acc[k]); acc[k] = fmaf(w.w, x.w, acc[k]);
            }
        }
        const int prow = inj ? LR + warp : row0 + r;
        const float v = tsum<P>(acc, lane);
        if (owner) part[((size_t) tk * Q8_NKC + kc) * PR + prow] = v;
    }
}

template <int T, bool PF = false>
__global__ void __launch_bounds__(THREADS) gr_up_q8_fast_kernel(GrMulti m, const float* __restrict__ part,
                                                                const float* __restrict__ ssg) {
    constexpr int P = pow2_ceil(T);
    constexpr int RPW8 = HC * UPM_COLS / WARPS;
    __shared__ __align__(16) float lo[T][LR];
    __shared__ float rsS[T][HC];
    __shared__ float g[T][HC][UPM_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int d0 = blockIdx.x * UPM_COLS;
    const int tk = tsum_token<P>(lane);
    const bool owner = lane == tsum_lane<P>(tk) && tk < T;
    // PF: every row's weights in registers before the part / lo phase (its latency hides the weight read)
    uint32_t pq[PF ? RPW8 : 1][3];
    float pd[PF ? RPW8 : 1][3];
    if constexpr (PF) {
#pragma unroll
        for (int qq = 0; qq < RPW8; ++qq) {
            const int r = warp + qq * WARPS;
            const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
            const uint8_t* row = m.a[0].q8_up + (size_t) i * (LR / 32) * Q8B;
#pragma unroll
            for (int s = 0; s < 3; ++s) {
                const int v = 4 * lane + 128 * s;
                if (v < LR) {
                    const uint8_t* blk = row + (v >> 5) * Q8B;
                    pq[qq][s] = q8_ld16(blk + 2 + (v & 31)) | q8_ld16(blk + 4 + (v & 31)) << 16;
                    pd[qq][s] = q8_half(q8_ld16(blk));
                } else {
                    pq[qq][s] = 0u; pd[qq][s] = 0.0f;
                }
            }
        }
    }
    if (t < T * HC) {
        const int k = t / HC, c = t - k * HC;
        float ss = 0.0f;
#pragma unroll
        for (int j = 0; j < Q8_CPS; ++j) ss += ssg[k * Q8_NKC + c * Q8_CPS + j];
        const float r = rsqrtf(ss / (float) N + m.a[k].eps);
        rsS[k][c] = r;
        if (blockIdx.x == 0) m.a[k].rs[c] = r;
    }
    __syncthreads();
    for (int i = t; i < T * PR; i += THREADS) {
        const int k = i / PR, r = i - k * PR;
        if (r >= LR && (blockIdx.x != 0 || m.a[k].w_inject == nullptr)) continue;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            float p = 0.0f;
#pragma unroll
            for (int j = 0; j < Q8_CPS; ++j) p += part[((size_t) k * Q8_NKC + c * Q8_CPS + j) * PR + r];
            sum = fmaf(rsS[k][c], p, sum);
        }
        if (r < LR) {
            const float x = sum / (float) HC;
            lo[k][r] = x / (1.0f + __expf(-x));
        } else {
            m.a[k].inject_out[r - LR] = sum;
        }
    }
    __syncthreads();
#pragma unroll(PF ? RPW8 : 2)
    for (int qq = 0; qq < RPW8; ++qq) {
        const int r = warp + qq * WARPS;
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint8_t* row = m.a[0].q8_up + (size_t) i * (LR / 32) * Q8B;
        float4 w[3];
#pragma unroll
        for (int s = 0; s < 3; ++s) {
            const int v = 4 * lane + 128 * s;
            if (PF) {
                w[s] = v < LR ? q8_four(pq[PF ? qq : 0][s], pd[PF ? qq : 0][s]) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            } else if (v < LR) {
                const uint8_t* blk = row + (v >> 5) * Q8B;
                w[s] = q8_four(q8_ld16(blk + 2 + (v & 31)) | q8_ld16(blk + 4 + (v & 31)) << 16, q8_half(q8_ld16(blk)));
            } else {
                w[s] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        float rv = 0.0f, wn = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (owner) {
            const FusedGrArgs& a = m.a[tk];
            rv = a.R[i];
            wn = a.w_norm[i];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float acc[P];
#pragma unroll
        for (int k = 0; k < P; ++k) acc[k] = 0.0f;
#pragma unroll
        for (int k = 0; k < T; ++k) {
#pragma unroll
            for (int s = 0; s < 3; ++s) {
                const int v = 4 * lane + 128 * s;
                if (v < LR) {
                    const float4 x = *reinterpret_cast<const float4*>(&lo[k][v]);
                    acc[k] = fmaf(w[s].x, x.x, acc[k]); acc[k] = fmaf(w[s].y, x.y, acc[k]);
                    acc[k] = fmaf(w[s].z, x.z, acc[k]); acc[k] = fmaf(w[s].w, x.w, acc[k]);
                }
            }
        }
        const float mine = tsum<P>(acc, lane);
        if (owner) {
            if (apply) {
                rv = fmaf(bo, 2.0f * sigmoidf_(ip / (float) HC), rv);
                m.a[tk].R_out[i] = rv;
            }
            const float x = rv * wn * rsS[tk][c];
            g[tk][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) sum += g[k][c][col];
        m.a[k].mixed[d0 + col] = sum / (float) HC;
    }
    if (m.a[0].q8_mixed != nullptr) gr_q8_tail(m, d0);   // S26 STRATA_QFUSE
}

template <int T> void launch_q8_t(const GrMulti& m, float* part, float* ssg, cudaStream_t st, unsigned long long* stamp_buf,
                                  int stamp_i0) {
    static const bool ts = [] { const char* v = std::getenv("STRATA_TSUM"); return v && v[0] == '1'; }();
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, (void*) st);
    if (ts) gr_down_q8_fast_kernel<T, 4><<<dim3(LR / (WARPS * 4) + 1, Q8_NKC), THREADS, 0, st>>>(m, part, ssg);
    else gr_down_q8_kernel<T><<<dim3(Q8_RG + 1, Q8_NKC), THREADS, 0, st>>>(m, part, ssg);
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, (void*) st);
    if (ts) gr_up_q8_fast_kernel<T, false><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
    else gr_up_q8_kernel<T><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
}
void launch_q8(const GrMulti& m, float* scratch, cudaStream_t st, unsigned long long* stamp_buf, int stamp_i0) {
    float* part = scratch;
    float* ssg = scratch + (size_t) m.T * Q8_NKC * PR;
    switch (m.T) {
        case 1: launch_q8_t<1>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        case 2: launch_q8_t<2>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        case 3: launch_q8_t<3>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        case 4: launch_q8_t<4>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        case 5: launch_q8_t<5>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        case 6: launch_q8_t<6>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        case 7: launch_q8_t<7>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        default: launch_q8_t<8>(m, part, ssg, st, stamp_buf, stamp_i0); break;
    }
}

}  // namespace

#if STRATA_GR_FAST_BUILD
int g_gr_fast = -1;   // fused_gr_set_fast (the bench); -1 = STRATA_GR_FAST, else the card's default
static bool gr_fast() {
    static const int env = [] { const char* v = std::getenv("STRATA_GR_FAST"); return v ? (std::atoi(v) != 0 ? 1 : 0) : -1; }();
    if (g_gr_fast >= 0) return g_gr_fast != 0;
    if (env >= 0) return env != 0;
#if defined(__HIPCC__)
    return true;
#else
    { const char* v = std::getenv("STRATA_SM70_TABLE"); return cur_dev_volta() && v != nullptr && std::atoi(v) != 0; }   // PR 1401: opt-in   // CUDA: on Volta (V100-SXM2: 50.9 -> 46.7 us a read at T 1, bitwise); elsewhere opt-in
#endif
}
#else
int g_gr_fast = -1;
#endif
void fused_gr_set_fast(int on) { g_gr_fast = on; }

bool fused_gr_read_multi(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream, unsigned long long* stamp_buf,
                         int stamp_i0) {
    if (n_tok < 1 || n_tok > kFusedGrMaxT || xn_scratch == nullptr) {
        std::fprintf(stderr, "fused_gr_read_multi: invalid arguments\n");
        std::exit(1);
    }
    GrMulti m;
    for (int t = 0; t < n_tok; ++t) {
        m.a[t] = a[t];
        const FusedGrArgs& x = a[t];
        if (!x.R || !x.w_norm || !x.w_down || !x.w_up || !x.lo || !x.rs || !x.mixed || (x.w_inject && !x.inject_out) ||
            (x.apply && (!x.bo_prev || !x.inj_prev || !x.R_out)) || x.w_down != a[0].w_down || x.w_up != a[0].w_up ||
            x.w_inject != a[0].w_inject || x.w_norm != a[0].w_norm || x.q8_down != a[0].q8_down ||
            x.q8_up != a[0].q8_up || x.q8_inject != a[0].q8_inject) {
            std::fprintf(stderr, "fused_gr_read_multi: invalid arguments for token %d\n", t);
            std::exit(1);
        }
    }
    // S26 STRATA_QFUSE: every token needs its q8_1 destination and token 0 the counters; otherwise none is written
    bool q8 = a[0].q8_cnt != nullptr;
    for (int t = 0; t < n_tok; ++t) q8 = q8 && a[t].q8_mixed != nullptr;
    if (!q8)
        for (int t = 0; t < n_tok; ++t) m.a[t].q8_mixed = nullptr;
    m.xn = xn_scratch;
    m.T = n_tok;
    cudaStream_t st = (cudaStream_t) stream;
    if (a[0].q8_down != nullptr && a[0].q8_up != nullptr) {   // inject: Q8_0 copy, or the BF16 rows
        launch_q8(m, xn_scratch, st, stamp_buf, stamp_i0);   // S23 experiment: STRATA_HC_Q8=1
        const cudaError_t eq = cudaGetLastError();
        if (eq != cudaSuccess) {
            std::fprintf(stderr, "fused_gr_read_multi q8: %s\n", cudaGetErrorString(eq));
            std::exit(1);
        }
        return q8;
    }
    // STRATA_GR_V3=1: the two-kernel read above (another summation order - opt-in)
    static const bool v3 = [] { const char* v = std::getenv("STRATA_GR_V3"); return v != nullptr && std::atoi(v) != 0; }();
    static int split3[64] = {};   // per device: 0 = not decided yet, 1 / 2 = column halves S, -1 = does not fit
    int dev3 = 0;
    if (v3) {
        cudaGetDevice(&dev3);
        if (dev3 >= 0 && dev3 < 64 && split3[dev3] == 0) {
            int optin = 0;
            cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev3);
            const int limit = optin > 0 ? optin : 48 * 1024;
            const int need1 = (int) (kFusedGrMaxT * N * sizeof(float)), need6 = (int) (6 * N * sizeof(float)),
                      need5 = (int) (5 * N * sizeof(float)), need2 = need1 / 2;
            int split = -1;
            if (need1 <= limit &&
                cudaFuncSetAttribute(gr_down_v3_kernel<1, kFusedGrMaxT, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, need1) == cudaSuccess &&
                cudaFuncSetAttribute(gr_down_v3_kernel<1, 6, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, need6) == cudaSuccess &&
                cudaFuncSetAttribute(gr_down_v3_kernel<1, 5, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, need5) == cudaSuccess) {
                cudaFuncSetAttribute(gr_down_v3_kernel<1, 1, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                cudaFuncSetAttribute(gr_down_v3_kernel<1, 2, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                cudaFuncSetAttribute(gr_down_v3_kernel<1, 3, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                cudaFuncSetAttribute(gr_down_v3_kernel<1, 4, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                cudaFuncSetAttribute(gr_down_v3_kernel<1, 5, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                cudaFuncSetAttribute(gr_down_v3_kernel<1, 6, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                cudaFuncSetAttribute(gr_down_v3_kernel<1, 4, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                cudaFuncSetAttribute(gr_down_v3_kernel<1, kFusedGrMaxT, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                split = 1;
            } else if (need2 <= limit)
                // #375 (kenh0u): the S = 2 split (a 64 KB opt-in card: Turing) disagrees with itself in gr_parity
                // (graph replay vs direct call) - such a card keeps the default read until that split is fixed
                std::fprintf(stderr, "strata: STRATA_GR_V3=1 needs the two-half split on this card, which fails its "
                                     "checks (#375): the default read is used\n");
            cudaGetLastError();   // drop any error the attempts left behind
            split3[dev3] = split;
        }
    }
    const int split = v3 && dev3 >= 0 && dev3 < 64 ? split3[dev3] : -1;
    if (split > 0) {   // 2 kernels; scratch = partials + sums of squares
        float* part = xn_scratch;
        float* ssg = xn_scratch + (size_t) n_tok * HC * 2 * PR;   // room for S = 2
        const size_t sm = (size_t) n_tok * (N / split) * sizeof(float);
        if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, stream);
        if (split == 2) gr_down_v3_kernel<2><<<dim3(LR / WARPS + 1, HC * 2), THREADS, sm, st>>>(m, part, ssg);
        else if (n_tok == 1) gr_down_v3_kernel<1, 1, true><<<dim3(LR / WARPS + 1, HC), THREADS, sm, st>>>(m, part, ssg);
        else if (n_tok == 2) gr_down_v3_kernel<1, 2, true><<<dim3(LR / WARPS + 1, HC), THREADS, sm, st>>>(m, part, ssg);
        else if (n_tok == 3) gr_down_v3_kernel<1, 3, true><<<dim3(LR / WARPS + 1, HC), THREADS, sm, st>>>(m, part, ssg);
        else if (n_tok == 4) gr_down_v3_kernel<1, 4, true><<<dim3(LR / WARPS + 1, HC), THREADS, sm, st>>>(m, part, ssg);
        else if (n_tok == 5) gr_down_v3_kernel<1, 5, true><<<dim3(LR / WARPS + 1, HC), THREADS, sm, st>>>(m, part, ssg);
        else if (n_tok == 6) gr_down_v3_kernel<1, 6, true><<<dim3(LR / WARPS + 1, HC), THREADS, sm, st>>>(m, part, ssg);
        else gr_down_v3_kernel<1, kFusedGrMaxT, false><<<dim3(LR / WARPS + 1, HC), THREADS, sm, st>>>(m, part, ssg);
        if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, stream);
        if (split == 2) gr_up_v3_kernel<2><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
        else if (n_tok == 1) gr_up_v3_kernel<1, 1, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
        else if (n_tok == 2) gr_up_v3_kernel<1, 2, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
        else if (n_tok == 3) gr_up_v3_kernel<1, 3, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
        else if (n_tok == 4) gr_up_v3_kernel<1, 4, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
        else if (n_tok == 5) gr_up_v3_kernel<1, 5, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
        else if (n_tok == 6) gr_up_v3_kernel<1, 6, true><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
        else gr_up_v3_kernel<1, kFusedGrMaxT, false><<<UPM_BLOCKS, THREADS, 0, st>>>(m, part, ssg);
        const cudaError_t e3 = cudaGetLastError();
        if (e3 != cudaSuccess) {
            std::fprintf(stderr, "fused_gr_read_multi v3: %s\n", cudaGetErrorString(e3));
            std::exit(1);
        }
        return false;   // the v3 read: no q8_1 (the caller quantizes)
    }
#if defined(STRATA_HIP_GFX906)
    // gfx906, opt-in STRATA_GR_SPLIT=1: the latency-hidden norm/up (STRATA_GR_FAST) with the K-split down (~3% faster
    // per verify window).  Its fixed-order finish sums the K slices in another order than the single-token
    // fused_gr_read, so the multi-token read is then no longer bitwise equal to T single-token calls (gr_parity's
    // contract); off by default for that reason.
    static const bool split_on = std::getenv("STRATA_GR_SPLIT") && std::string(std::getenv("STRATA_GR_SPLIT")) == "1";
    if (split_on) {
        const bool fast = gr_fast();
        if (fast) gr_norm_fast_kernel<<<n_tok, THREADS, 0, st>>>(m);
        else gr_norm_multi_kernel<<<n_tok, THREADS, 0, st>>>(m);
        if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, stream);
        gr_down_split_kernel<<<dim3(DOWN_BLOCKS + 1, GS_S), GS_WAVES * 64, (size_t) n_tok * GS_SL * sizeof(float), st>>>(
            m);
        gr_down_finish_kernel<<<(unsigned) ((n_tok * GS_ROWS + 255) / 256), 256, 0, st>>>(m);
        if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, stream);
        if (fast) gr_up_fast_kernel<<<UPM_BLOCKS, THREADS, 0, st>>>(m);
        else gr_up_multi_kernel<<<UPM_BLOCKS, THREADS, 0, st>>>(m);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "fused_gr_read_multi: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
        return false;   // the gfx906 STRATA_GR_SPLIT read: no q8_1 (the caller quantizes), like the v3 path above
    }
#endif
    // the default read (STRATA_GR_V3 unset): v1, or the bitwise-equal v2 / v3 this card's check accepted
    const int gv = variant_for_T(fused_gr_variant(), m.T);
    // STRATA_HC_GRAPH=1: replay the launch sequence as a captured graph when this key has been seen before;
    // capture failure (or a key that never repeats) falls through to the plain launch below.
    if (!(hc_graph_env() && graph_replay(m, gv, st, stamp_buf, stamp_i0)))
        launch_multi(m, gv, st, stamp_buf, stamp_i0);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fused_gr_read_multi: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return q8;
}

bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr) {
    return n_embd == N && hc == HC && hc_lr == LR;
}

void fused_gr_read(const FusedGrArgs& a, void* stream) {
    if (!a.R || !a.w_norm || !a.w_down || !a.w_up || !a.lo || !a.rs || !a.mixed ||
        (a.w_inject && !a.inject_out) || (a.apply && (!a.bo_prev || !a.inj_prev || !a.R_out)) ||
        (a.apply && a.inj_prev == a.inject_out)) {
        std::fprintf(stderr, "fused_gr_read: invalid arguments\n");
        std::exit(1);
    }
    cudaStream_t st = (cudaStream_t) stream;
    gr_down_kernel<<<DOWN_BLOCKS + 1, THREADS, 0, st>>>(a);
    gr_up_kernel<<<UP_BLOCKS, THREADS, 0, st>>>(a);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fused_gr_read: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}


namespace {

/// The plain read, split, staged, small-CTA, two-row-per-warp, the two register-tuple variants, and (for one
/// token) the single-token read on random bf16 weights and random inputs, 1..8 tokens, with and without the
/// pending write and with and without the inject weights; every output of each multi variant is compared with
/// the plain read bit for bit (and the plain read with the single-token read), and the variants that reuse the
/// staged tile schedule (5, 6, 7) with the staged read bit for bit.  `why[v]` records the first difference for
/// a multi variant; false if the check itself could not run.
bool fused_gr_selftest(bool ok_variant[12], std::string why[12]) {
    constexpr int TM = kFusedGrMaxT;
    constexpr int NV = 12;  // 0 plain, 1 split, 2 staged, 3 single, 4 small CTA, 5 reuse CTA, 6 register pipe,
                            // 7 register half, 8 packed staged (STRATA_HC_PACK), 9 row split, 10 LDS accumulators
                            // (STRATA_HC_SPLIT=9), 11 register pipe+half (STRATA_HC_SPLIT=10)
    std::mt19937 rng(20260930u);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<uint16_t> h_down((size_t) LR * D), h_up((size_t) D * LR), h_inj((size_t) HC * D);
    std::vector<float> h_norm(D), h_R((size_t) TM * D), h_bo((size_t) TM * N), h_ip((size_t) TM * HC);
    for (auto& w : h_down) w = bf16_from_f32(0.02f * nd(rng));
    for (auto& w : h_up) w = bf16_from_f32(0.05f * nd(rng));
    for (auto& w : h_inj) w = bf16_from_f32(0.02f * nd(rng));
    for (auto& w : h_norm) w = 1.0f + 0.1f * nd(rng);
    for (auto& x : h_R) x = nd(rng);
    for (auto& x : h_bo) x = 0.5f * nd(rng);
    for (auto& x : h_ip) x = 2.0f * nd(rng);
    for (int v = 0; v < NV; ++v) { ok_variant[v] = v == 1; why[v].clear(); }

    // the packed arm's arenas: built and validated on the CPU from the same BF16 bytes the device reads
    std::vector<uint8_t> ar_down, ar_up, ar_inj;
    HcPackedMatrix pm_down, pm_up, pm_inj;
    bool pack_ok = true;
    std::string pack_err;
    auto build_pack = [&](const std::vector<uint16_t>& src, const HcPackGeometry& g, std::vector<uint8_t>& ar,
                          HcPackedMatrix& p) {
        if (!pack_ok) return;
        ar.assign(hc_pack_bytes(g, hc_pack_count_escapes(src.data(), g)), 0);
        std::string e;
        if (!hc_pack_build(src.data(), g, ar.data(), ar.size(), &p, &e) ||
            !hc_pack_validate(src.data(), g, p, ar.data(), &e)) {
            pack_ok = false;
            pack_err = e;
        }
    };
    build_pack(h_down, hc_pack_geometry_down(), ar_down, pm_down);
    build_pack(h_up, hc_pack_geometry_up(), ar_up, pm_up);
    build_pack(h_inj, hc_pack_geometry_inject(), ar_inj, pm_inj);
    if (!pack_ok) {
        ok_variant[kHcPacked] = false;
        why[kHcPacked] = "packing the check's weights on the CPU: " + pack_err;
    }
    const size_t n_set = (size_t) TM * (D + D + LR + HC + HC + N);   // R_out, xn, lo, rs, inject, mixed (floats)
    const size_t bytes = h_down.size() * 2 + h_up.size() * 2 + h_inj.size() * 2 + ar_down.size() + ar_up.size() +
                         ar_inj.size() +
                         (h_norm.size() + h_R.size() + h_bo.size() + h_ip.size() + NV * n_set) * 4 + 32 * 256;
    uint8_t* base = nullptr;
    if (cudaMalloc((void**) &base, bytes) != cudaSuccess) {
        cudaGetLastError();
        why[0] = "no room for the check (" + std::to_string(bytes >> 20) + " MiB)";
        return false;
    }
    size_t off = 0;
    auto take = [&](size_t b) { void* p = base + off; off += (b + 255) / 256 * 256; return p; };
    uint16_t* d_down = (uint16_t*) take(h_down.size() * 2);
    uint16_t* d_up = (uint16_t*) take(h_up.size() * 2);
    uint16_t* d_inj = (uint16_t*) take(h_inj.size() * 2);
    uint8_t *d_pdown = (uint8_t*) take(ar_down.size()), *d_pup = (uint8_t*) take(ar_up.size()),
            *d_pinj = (uint8_t*) take(ar_inj.size());
    const HcPackedWeights pk_down = hc_pack_device_view(pm_down, d_pdown), pk_up = hc_pack_device_view(pm_up, d_pup),
                          pk_inj = hc_pack_device_view(pm_inj, d_pinj);
    float* d_norm = (float*) take(h_norm.size() * 4);
    float* d_R = (float*) take(h_R.size() * 4);
    float* d_bo = (float*) take(h_bo.size() * 4);
    float* d_ip = (float*) take(h_ip.size() * 4);
    struct Set { float *R_out, *xn, *lo, *rs, *inj, *mixed; } set[NV];
    for (Set& x : set) {
        x.R_out = (float*) take((size_t) TM * D * 4); x.xn = (float*) take((size_t) TM * D * 4);
        x.lo = (float*) take((size_t) TM * LR * 4); x.rs = (float*) take((size_t) TM * HC * 4);
        x.inj = (float*) take((size_t) TM * HC * 4); x.mixed = (float*) take((size_t) TM * N * 4);
    }
    cudaStream_t st = nullptr;
    bool ok = off <= bytes && cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) == cudaSuccess &&
              cudaMemcpyAsync(d_down, h_down.data(), h_down.size() * 2, cudaMemcpyHostToDevice, st) == cudaSuccess &&
              cudaMemcpyAsync(d_up, h_up.data(), h_up.size() * 2, cudaMemcpyHostToDevice, st) == cudaSuccess &&
              cudaMemcpyAsync(d_inj, h_inj.data(), h_inj.size() * 2, cudaMemcpyHostToDevice, st) == cudaSuccess &&
              cudaMemcpyAsync(d_norm, h_norm.data(), h_norm.size() * 4, cudaMemcpyHostToDevice, st) == cudaSuccess &&
              cudaMemcpyAsync(d_R, h_R.data(), h_R.size() * 4, cudaMemcpyHostToDevice, st) == cudaSuccess &&
              cudaMemcpyAsync(d_bo, h_bo.data(), h_bo.size() * 4, cudaMemcpyHostToDevice, st) == cudaSuccess &&
              cudaMemcpyAsync(d_ip, h_ip.data(), h_ip.size() * 4, cudaMemcpyHostToDevice, st) == cudaSuccess &&
              cudaMemcpyAsync(d_pdown, ar_down.data(), ar_down.size(), cudaMemcpyHostToDevice, st) == cudaSuccess &&
              cudaMemcpyAsync(d_pup, ar_up.data(), ar_up.size(), cudaMemcpyHostToDevice, st) == cudaSuccess &&
              cudaMemcpyAsync(d_pinj, ar_inj.data(), ar_inj.size(), cudaMemcpyHostToDevice, st) == cudaSuccess;
    if (!ok) why[0] = "setting up the check failed";
    std::vector<float> h1, h2;
    // true when equal; false with the first difference in `w` (or a read-back failure in `ok`)
    auto same = [&](const float* d1, const float* d2, size_t n, const char* what, int T, int apply, int inj,
                    std::string& w) {
        h1.resize(n);
        h2.resize(n);
        if (cudaMemcpyAsync(h1.data(), d1, n * 4, cudaMemcpyDeviceToHost, st) != cudaSuccess ||
            cudaMemcpyAsync(h2.data(), d2, n * 4, cudaMemcpyDeviceToHost, st) != cudaSuccess ||
            cudaStreamSynchronize(st) != cudaSuccess) {
            why[0] = "reading back the check failed";
            ok = false;
            return false;
        }
        for (size_t i = 0; i < n; ++i) {
            uint32_t u1, u2;
            std::memcpy(&u1, &h1[i], 4);
            std::memcpy(&u2, &h2[i], 4);
            if (u1 != u2) {
                char buf[192];
                std::snprintf(buf, sizeof buf, "%s differs for %d token(s)%s%s at %zu (%08x, not %08x)", what, T,
                              apply ? " with the pending write" : "", inj ? "" : " without the inject weights", i,
                              u2, u1);
                w = buf;
                return false;
            }
        }
        return true;
    };
    auto all_same = [&](const Set& s1, const Set& s2, int T, int apply, int inj, std::string& w) {
        return same(s1.lo, s2.lo, (size_t) T * LR, "lo", T, apply, inj, w) &&
               same(s1.rs, s2.rs, (size_t) T * HC, "rs", T, apply, inj, w) &&
               same(s1.inj, s2.inj, (size_t) T * HC, "inject", T, apply, inj, w) &&
               same(s1.mixed, s2.mixed, (size_t) T * N, "mixed", T, apply, inj, w) &&
               same(s1.R_out, s2.R_out, (size_t) T * D, "R", T, apply, inj, w);
    };
    // Split is the bitwise-single-token reference; staged is checked against it separately. The reuse CTA and the
    // register-tuple variants must match the staged candidate exactly, so those gates isolate the tile schedule
    // and CTA shape, not any pre-existing staged delta.
    ok_variant[2] = ok_variant[3] = ok_variant[4] = ok_variant[5] = ok_variant[6] = ok_variant[7] = ok;
    ok_variant[kHcRowSplit] = ok;
    ok_variant[kHcLdsAccum] = ok;
    ok_variant[kHcRegisterPipeHalf] = ok;
    if (pack_ok) ok_variant[kHcPacked] = ok;
    bool single_ok = ok;
    for (int inj = 0; inj < 2 && ok; ++inj) {
      for (int apply = 0; apply < 2 && ok; ++apply) {
        for (int T = 1; T <= TM && ok; ++T) {
            FusedGrArgs a[NV][TM];
            for (int v = 0; v < NV; ++v) {
                const size_t span = (size_t) ((uint8_t*) (set[v].mixed + (size_t) TM * N) - (uint8_t*) set[v].R_out);
                ok = ok && cudaMemsetAsync(set[v].R_out, 0xFF, span, st) == cudaSuccess;   // the whole set
                for (int k = 0; k < T; ++k) {
                    FusedGrArgs& x = a[v][k];
                    x.R = d_R + (size_t) k * D; x.R_out = set[v].R_out + (size_t) k * D; x.apply = apply != 0;
                    x.bo_prev = d_bo + (size_t) k * N; x.inj_prev = d_ip + (size_t) k * HC;
                    x.w_norm = d_norm; x.w_down = d_down; x.w_up = d_up;
                    x.w_inject = inj ? d_inj : nullptr;   // the inject block stays inert without the weights
                    if (v == kHcPacked && pack_ok) {      // the same weights, arriving packed
                        x.pack_down = pk_down; x.pack_up = pk_up;
                        if (inj) x.pack_inject = pk_inj;
                    }
                    x.eps = 1e-6f;
                    x.lo = set[v].lo + (size_t) k * LR; x.rs = set[v].rs + (size_t) k * HC;
                    x.inject_out = set[v].inj + (size_t) k * HC; x.mixed = set[v].mixed + (size_t) k * N;
                }
            }
            for (int v = 0; v < 3; ++v) {
                GrMulti m;
                for (int k = 0; k < T; ++k) m.a[k] = a[v][k];
                m.xn = set[v].xn;
                m.T = T;
                launch_multi(m, v + 1, st, nullptr, 0);
            }
            GrMulti small_cta_m;
            for (int k = 0; k < T; ++k) small_cta_m.a[k] = a[4][k];
            small_cta_m.xn = set[4].xn;
            small_cta_m.T = T;
            launch_multi(small_cta_m, kHcSmallCta, st, nullptr, 0);
            GrMulti reuse_rows_m;
            for (int k = 0; k < T; ++k) reuse_rows_m.a[k] = a[5][k];
            reuse_rows_m.xn = set[5].xn;
            reuse_rows_m.T = T;
            launch_multi(reuse_rows_m, kHcReuseTwoRows, st, nullptr, 0);
            GrMulti pipe_m;
            for (int k = 0; k < T; ++k) pipe_m.a[k] = a[6][k];
            pipe_m.xn = set[6].xn;
            pipe_m.T = T;
            launch_multi(pipe_m, kHcRegisterPipe, st, nullptr, 0);
            GrMulti half_m;
            for (int k = 0; k < T; ++k) half_m.a[k] = a[7][k];
            half_m.xn = set[7].xn;
            half_m.T = T;
            launch_multi(half_m, kHcRegisterHalf, st, nullptr, 0);
            GrMulti row_split_m;
            for (int k = 0; k < T; ++k) row_split_m.a[k] = a[kHcRowSplit][k];
            row_split_m.xn = set[kHcRowSplit].xn;
            row_split_m.T = T;
            launch_multi(row_split_m, kHcRowSplit, st, nullptr, 0);
            GrMulti lds_accum_m;
            for (int k = 0; k < T; ++k) lds_accum_m.a[k] = a[kHcLdsAccum][k];
            lds_accum_m.xn = set[kHcLdsAccum].xn;
            lds_accum_m.T = T;
            launch_multi(lds_accum_m, kHcLdsAccum, st, nullptr, 0);
            GrMulti pipe_half_m;
            for (int k = 0; k < T; ++k) pipe_half_m.a[k] = a[kHcRegisterPipeHalf][k];
            pipe_half_m.xn = set[kHcRegisterPipeHalf].xn;
            pipe_half_m.T = T;
            launch_multi(pipe_half_m, kHcRegisterPipeHalf, st, nullptr, 0);
            if (pack_ok) {   // the packed arm, forced on for the check whatever the env says
                g_force_pack = true;
                GrMulti packed_m;
                for (int k = 0; k < T; ++k) packed_m.a[k] = a[8][k];
                packed_m.xn = set[8].xn;
                packed_m.T = T;
                launch_multi(packed_m, kHcStaged, st, nullptr, 0);
                g_force_pack = false;
            }
            if (T == 1) fused_gr_read(a[3][0], st);
            if (cudaGetLastError() != cudaSuccess || cudaStreamSynchronize(st) != cudaSuccess) {
                why[0] = "a kernel of the check failed";
                ok = false;
                break;
            }
            if (ok_variant[2] && !all_same(set[0], set[1], T, apply, inj, why[2])) ok_variant[2] = false;
            if (ok && ok_variant[3] && !all_same(set[0], set[2], T, apply, inj, why[3])) ok_variant[3] = false;
            if (ok_variant[4] && !all_same(set[0], set[4], T, apply, inj, why[4])) ok_variant[4] = false;
            if (ok_variant[5] && !all_same(set[2], set[5], T, apply, inj, why[5])) ok_variant[5] = false;
            if (ok_variant[6] && !all_same(set[2], set[6], T, apply, inj, why[6])) ok_variant[6] = false;
            if (ok_variant[7] && !all_same(set[2], set[7], T, apply, inj, why[7])) ok_variant[7] = false;
            if (pack_ok && ok_variant[8] && !all_same(set[2], set[8], T, apply, inj, why[8])) ok_variant[8] = false;
            if (ok_variant[kHcRowSplit] && !all_same(set[2], set[kHcRowSplit], T, apply, inj, why[kHcRowSplit]))
                ok_variant[kHcRowSplit] = false;
            if (ok_variant[kHcLdsAccum] && !all_same(set[2], set[kHcLdsAccum], T, apply, inj, why[kHcLdsAccum]))
                ok_variant[kHcLdsAccum] = false;
            if (ok_variant[kHcRegisterPipeHalf] &&
                !all_same(set[2], set[kHcRegisterPipeHalf], T, apply, inj, why[kHcRegisterPipeHalf]))
                ok_variant[kHcRegisterPipeHalf] = false;
            // Every T above runs one-shot multi-read, but verify windows also exercise the T=1 and T=TM chunks
            // after down_chunk has split the launch.  Keep both exact endpoints in the device self-test.
            if (pack_ok && ok_variant[kHcPacked] && T == 1 &&
                !all_same(set[2], set[kHcPacked], 1, apply, inj, why[kHcPacked])) ok_variant[kHcPacked] = false;
            if (pack_ok && ok_variant[kHcPacked] && T == TM &&
                !all_same(set[2], set[kHcPacked], TM, apply, inj, why[kHcPacked])) ok_variant[kHcPacked] = false;
            if (ok && T == 1 && single_ok && !all_same(set[3], set[0], T, apply, inj, why[1])) single_ok = false;
        }
      }
    }
    if (!single_ok && ok) {                            // the plain read itself disagrees with the single-token read
        why[2] = why[3] = why[4] = why[5] = why[6] = why[7] = why[kHcRowSplit] = why[kHcLdsAccum] =
            "the plain read differs from the single-token read: " + why[1];
        ok_variant[2] = ok_variant[3] = ok_variant[4] = ok_variant[5] = ok_variant[6] = ok_variant[7] =
            ok_variant[kHcRowSplit] = ok_variant[kHcLdsAccum] = ok_variant[kHcRegisterPipeHalf] = false;
    }
    if (st != nullptr) {
        cudaStreamSynchronize(st);
        cudaStreamDestroy(st);
    }
    cudaFree(base);
    cudaGetLastError();
    if (!ok) {
        ok_variant[2] = ok_variant[3] = ok_variant[4] = ok_variant[5] = ok_variant[6] = ok_variant[7] =
            ok_variant[kHcRowSplit] = ok_variant[kHcLdsAccum] = ok_variant[kHcRegisterPipeHalf] = false;
    }
    return ok;
}

}  // namespace

int fused_gr_hc_pack() { return pack_on() ? 1 : 0; }

int fused_gr_variant() {
    int dev = 0;
    cudaGetDevice(&dev);
    const int v = dev >= 0 && dev < 64 ? g_variant[dev].load() : 0;
    if (v > 0) return v;
    // A standalone benchmark may request a path directly without running fused_gr_check.
    const char* e = std::getenv("STRATA_HC_BENCH_DIRECT");
    if (e != nullptr && e[0] != '0') return env_variant();
    // not checked on this card: the plain read, unless STRATA_HC_SPLIT names a variant (a test such as gr_parity)
    e = std::getenv("STRATA_HC_SPLIT");
    return e != nullptr && (e[0] == '1' || e[0] == '2' || e[0] == '3' || e[0] == '5' || e[0] == '6' || e[0] == '7' ||
                            e[0] == '8' || e[0] == '9')
               ? env_variant()
               : kHcPlain;
}

int fused_gr_variant_for_T(int T) { return variant_for_T(fused_gr_variant(), T); }

void fused_gr_check() {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64 || g_variant[dev].load() > 0) return;
    const int want = env_variant();
    // Promote-to-default (RESULTS.md P4): with STRATA_HC_SPLIT unset the default read is the register-half
    // variant (7) on a card whose check latched it - the latch publishes only on a bit-for-bit match with the
    // staged read at every T - and staged on a card that did not.  STRATA_HC_SPLIT=2 stays the escape hatch:
    // an explicit 2 latches staged even on a card that latched 7.
    const char* const split_env = std::getenv("STRATA_HC_SPLIT");
    const bool env_unset = split_env == nullptr || split_env[0] == '\0';
    if (want == kHcPlain) {
        g_variant[dev].store(kHcPlain);
        std::fprintf(stderr, "strata hc: CUDA%d: the hyper-connection read runs as the plain one (STRATA_HC_SPLIT=0)\n",
                     dev);
        return;
    }
    constexpr int NV = 12;                            // the arity fused_gr_selftest writes
    bool okv[NV];
    std::string why[NV];
    const bool ran = fused_gr_selftest(okv, why);
    int use = kHcPlain;
    // STRATA_HC_PACK: latch the packed arm on this device only if the check saw it match staged bit for bit
    const bool pack_use = env_pack() && ran && okv[kHcPacked];
    if (want == kHcRegisterPipeHalf && okv[kHcRegisterPipeHalf]) use = kHcRegisterPipeHalf;
    else if (want == kHcLdsAccum && okv[kHcLdsAccum]) use = kHcLdsAccum;
    else if (want == kHcRowSplit && okv[kHcRowSplit]) use = kHcRowSplit;
    else if (want == kHcRegisterPipe && okv[kHcRegisterPipe]) use = kHcRegisterPipe;
    else if (want == kHcRegisterHalf && okv[kHcRegisterHalf]) use = kHcRegisterHalf;
    else if (want == kHcReuseTwoRows && okv[kHcReuseTwoRows]) use = kHcReuseTwoRows;
    else if (want == kHcSmallCta && okv[kHcSmallCta]) use = kHcSmallCta;
    else if (env_unset && want == kHcStaged && okv[kHcRegisterHalf]) use = kHcRegisterHalf;   // promote-to-default
    else if (want >= kHcStaged && okv[kHcStaged]) use = kHcStaged;
    else if (okv[kHcSplit]) use = kHcSplit;
    g_variant[dev].store(use);
    if (env_pack() && !pack_use && ran) {
        std::fprintf(stderr, "strata hc: CUDA%d: the packed read differs from staged - not used: %s\n", dev,
                     why[kHcPacked].c_str());
    }
    if (dev >= 0 && dev < 64) g_pack_check[dev].store(pack_use ? 1 : 2);
    if (!ran)
        std::fprintf(stderr, "strata hc: CUDA%d: the check of split/staged could not run (%s)\n", dev, why[0].c_str());
    static const char* const name[NV] = {"",        "plain",   "split",   "staged",      "small-CTA staged",
                                          "2-row-per-warp staged", "register-pipe staged", "register-half staged",
                                          "packed staged", "row-split staged", "LDS-accumulator staged",
                                          "register-pipe-half staged"};
    for (int v = kHcRegisterPipeHalf; v >= kHcSplit; --v)
        if (v <= want && !okv[v] && ran) {
            const char* against = v >= kHcReuseTwoRows ? "the staged read" : "the plain read";
            std::fprintf(stderr, "strata hc: CUDA%d: the %s read differs from %s on this card - not used: %s\n",
                         dev, name[v], against, why[v].c_str());
        }
    static const char* const what[NV] = {
        "", "the plain read (the norm per token, the down projection on 41 blocks)",
        "split (the norm per token and stream, then the plain down projection)",
        "staged (the norm per token and stream, the staged down projection)",
        "small-CTA staged (the norm per token and stream, four-warps-per-block staged down projection)",
        "2-row-per-warp staged (four warps per CTA, two rows sharing each activation tile)",
        "register-pipe staged (the staged read with the next tile carried in a register tuple, STRATA_HC_SPLIT=6)",
        "register-half staged (the same with the tuple in two halves, STRATA_HC_SPLIT=7)",
        "packed staged (the staged read with the weights arriving packed, STRATA_HC_PACK=1)",
        "row-split staged (four rows per CTA, a warp pair per row, 81 blocks, STRATA_HC_SPLIT=8)",
        "LDS-accumulator staged (the staged read with the per-token sums in dynamic LDS, STRATA_HC_SPLIT=9)",
        ("register-pipe-half staged (the register pipeline with the activation tuple AND the weight prefetch in "
         "two halves, STRATA_HC_SPLIT=10)")
    };
    // one std::string, printed with c_str(): a std::string through a varargs %s does not compile
    const std::string pack_msg =
        pack_use ? "packed candidate self-test passed (13 bytes per 8 values; a caller must supply packed descriptors; "
                  "the production loader is not wired yet)"
                 : env_pack()
                       ? "as BF16 (STRATA_HC_PACK=1 asked for the packed read, the check rejected "
                         "it on this card: " +
                             why[kHcPacked] + ")"
                       : std::string("as BF16 (STRATA_HC_PACK unset)");
    std::fprintf(stderr, "strata hc: CUDA%d: the hc weights arrive %s\n", dev, pack_msg.c_str());
    // With the promote, the unset-env latch line names the effective variant and says it is the default.
    const std::string promote_msg =
        (env_unset && use == kHcRegisterHalf)
            ? " (the promoted default on this card; STRATA_HC_SPLIT=2 keeps the staged read)"
            : std::string();
    std::fprintf(stderr, "strata hc: CUDA%d: the hyper-connection read runs as %s%s%s\n", dev, what[use],
                 use >= kHcSplit ? "; checked bit for bit against the plain read on this card (STRATA_HC_SPLIT=1 or 0 "
                                   "for the earlier ones)" : "",
                 promote_msg.c_str());
}


}  // namespace strata::kernels
