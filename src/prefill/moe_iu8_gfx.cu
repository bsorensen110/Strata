// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// Port note: hipfire block_i8_128_quant.hip and block_i4_128_quant.hip:220-343.
// Producer-owned K128 blocks, folded x-only grids, one quantization shared by GU.
#include "strata/prefill/moe_iu8_gfx.hpp"
#if defined(__HIPCC__)
#include "strata/prefill/moe_iu8_quant.cuh"
#include "strata/prefill/moe_iu8_pack.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <climits>
#include <cstdio>
#include <cstdlib>

namespace strata::prefill::iu8 {
namespace {
void check(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "HIP MoE IU8: %s: %s\n", what, hipGetErrorString(e)); std::exit(1);
    }
}
[[noreturn]] void invalid(const char* what) {
    std::fprintf(stderr, "HIP MoE IU8: invalid %s\n", what); std::exit(1);
}
unsigned producer_grid(int cols, int rows) {
    if (cols <= 0 || cols % 128 || rows <= 0) invalid("producer K128 shape");
    const uint64_t groups = (uint64_t) (cols / 128) * rows;
    if ((groups + 15) / 16 > INT_MAX) invalid("producer grid extent");
    return (unsigned) ((groups + 15) / 16);
}

// Port note: hipfire gemm_mq6sym_i8.gfx1201.hip, prefill 128x128 / 256 threads /
// 36864 B LDS. Adapt planar W4/W6 and grouped slot prefixes; keep each K128 fold
// separate. gfx11 mirrors K16 operands; gfx12 splits K across lane halves.
using I2 = int __attribute__((ext_vector_type(2)));
using I4 = int __attribute__((ext_vector_type(4)));
using I8 = int __attribute__((ext_vector_type(8)));
using F8 = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ void barrier() {
#if defined(__gfx1100__) || defined(__gfx1151__)
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_waitcnt(0x7 | (0x3f << 10));
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#else
    __syncthreads();
#endif
}

__device__ __forceinline__ int decode(const WeightView& w, size_t i) {
    const size_t bit = i * w.bits, byte = bit / 8;
    unsigned word = w.codes[byte];
    if (bit % 8 + w.bits > 8) word |= (unsigned) w.codes[byte + 1] << 8;
    return (int) ((word >> (bit % 8)) & ((1u << w.bits) - 1)) - (1 << (w.bits - 1));
}

__global__ void __launch_bounds__(256) grouped_prefill(Product p) {
#if defined(__gfx1100__) || defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__)
    const int mtiles = (p.weights[0].rows + 127) / 128;
    const int tile = (int) blockIdx.x / mtiles, row0 = ((int) blockIdx.x % mtiles) * 128;
    int expert = p.n - 1;
    while (expert > 0 && tile < p.ranges[expert].first) --expert;
    const TileRange range = p.ranges[expert];
    const int token0 = range.begin + (tile - range.first) * 128;
    const WeightView w = p.weights[expert];
    const int tid = threadIdx.x, lane = tid & 31, lo = lane & 15, hi = lane >> 4;
    const int wave = tid >> 5, myrow = wave * 16 + lo;
    extern __shared__ __align__(16) unsigned char memory[];
    auto* ws = (int8_t*) memory;
    auto* xs = ws + 128 * 128;
    auto* scales_w = (float*) (xs + 128 * 128);
    auto* scales_x = scales_w + 128;
    static_assert(2 * 128 * 128 + 2 * 128 * sizeof(float) <= kPrefillLds);
    F8 acc[8] = {};
    for (int kb = 0; kb < w.cols / 128; ++kb) {
        for (int i = tid; i < 128 * 128; i += 256) {
            const int r = i / 128, k = i % 128;
            ws[r * 128 + byte_index(k)] = row0 + r < w.rows
                ? (int8_t) decode(w, (size_t) (row0 + r) * w.cols + kb * 128 + k) : 0;
            const int token = token0 + r;
            xs[i] = token < range.end ? p.x[(int64_t) kb * p.total_rows + token].qs[k] : 0;
        }
        if (tid < 128) {
            scales_w[tid] = row0 + tid < w.rows
                ? __half2float(__ushort_as_half(w.scales[(size_t) (row0 + tid) * (w.cols / 128) + kb])) : 0.0f;
            scales_x[tid] = token0 + tid < range.end
                ? p.x[(int64_t) kb * p.total_rows + token0 + tid].d : 0.0f;
        }
        barrier();
        I8 dot[8] = {};
        for (int step = 0; step < 8; ++step) {
#if defined(__gfx1200__) || defined(__gfx1201__)
            const I2 b = *(const I2*) (ws + myrow * 128 + hi * 64 + step * 8);
            for (int tg = 0; tg < 8; ++tg) {
                const I2 a = *(const I2*) (xs + (tg * 16 + lo) * 128 + hi * 64 + step * 8);
                dot[tg] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                    true, a, true, b, dot[tg], false);
            }
#else
            const I4 b = *(const I4*) (ws + myrow * 128 + step * 16);
            for (int tg = 0; tg < 8; ++tg) {
                const I4 a = *(const I4*) (xs + (tg * 16 + lo) * 128 + step * 16);
                dot[tg] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                    true, a, true, b, dot[tg], false);
            }
#endif
        }
        for (int tg = 0; tg < 8; ++tg)
        for (int j = 0; j < 8; ++j) {
#if defined(__gfx1200__) || defined(__gfx1201__)
            const int token = tg * 16 + hi * 8 + j;
#else
            const int token = tg * 16 + 2 * j + hi;
#endif
            // Frozen arithmetic: exact C, sd=RN(f32(s_h)*d), ascending K128 FMA.
            const float sd = __fmul_rn(scales_w[myrow], scales_x[token]);
            acc[tg][j] = __fmaf_rn(sd, (float) dot[tg][j], acc[tg][j]);
        }
        barrier();
    }
    if (row0 + myrow < w.rows) {
        for (int tg = 0; tg < 8; ++tg)
        for (int j = 0; j < 8; ++j) {
#if defined(__gfx1200__) || defined(__gfx1201__)
            const int slot = token0 + tg * 16 + hi * 8 + j;
#else
            const int slot = token0 + tg * 16 + 2 * j + hi;
#endif
            if (slot < range.end) {
                const int dstrow = p.ids ? p.ids[slot] : slot;
                p.dst[(int64_t) dstrow * p.ld + row0 + myrow] = acc[tg][j];
            }
        }
    }
#elif defined(__HIP_DEVICE_COMPILE__)
    __builtin_trap();
#endif
}

template<bool Scatter, bool SwiGLU>
__global__ void produce(const float* x, const int32_t* map, Block* out, float* h,
                        int cols, int ld, int rows, int topk) {
    const int64_t group = (int64_t) blockIdx.x * 16 + threadIdx.x / 16;
    const int kb = (int) (group % (cols / 128));
    const int64_t row = group / (cols / 128);
    const bool live = row < rows;
    const int j = threadIdx.x & 15;
    const int k0 = kb * 128 + j * 8;
    const int64_t source = live ? ((!Scatter && !SwiGLU && map) ? map[row] : row) : 0;
    float xv[8] = {};
    for (int k = 0; live && k < 8; ++k) {
        if constexpr (SwiGLU) {
            const float g = x[row * 2 * cols + k0 + k], u = x[row * 2 * cols + cols + k0 + k];
            xv[k] = g / (1.0f + __expf(-g)) * u;
            if (h) h[row * cols + k0 + k] = xv[k];
        } else {
            xv[k] = x[source * ld + k0 + k];
        }
    }
    const OwnedBlock q = quantize_owned(xv);
    if (!live) return;  // After wave-uniform ballots, including the final inactive half-wave.
    if constexpr (Scatter) {
        const int64_t routed_rows = (int64_t) rows * topk;
        for (int rank = 0; rank < topk; ++rank)
            store_owned(q, out + (int64_t) kb * routed_rows + map[row * topk + rank], threadIdx.x);
    } else {
        store_owned(q, out + (int64_t) kb * rows + row, threadIdx.x);
    }
}
}  // namespace

bool admitted(int tokens, int cols, void* stream) {
    if (tokens < 64 || cols <= 0 || cols % 128 || !usable()) return false;
    hipStreamCaptureStatus status{};
    check(hipStreamIsCapturing((hipStream_t) stream, &status), "capture admission");
    return status == hipStreamCaptureStatusNone;
}

void run(const Product& p, void* stream) {
    if (!p.x || !p.dst || p.n <= 0 || p.n > 16 || p.total_rows <= 0 || p.tiles <= 0)
        invalid("grouped operands");
    if (!admitted(p.tokens, p.weights[0].cols, stream)) invalid("grouped admission");
    int device = -1;
    check(hipGetDevice(&device), "grouped device");
    const int m = p.weights[0].rows, k = p.weights[0].cols;
    if (m <= 0 || m > INT_MAX - 127 || p.ld < m) invalid("grouped output shape");
    int64_t tiles = 0;
    for (int e = 0; e < p.n; ++e) {
        const auto& w = p.weights[e];
        const auto& r = p.ranges[e];
        if (!w.symmetric_verified || !w.codes || !w.scales || w.device != device ||
            w.rows != m || w.cols != k || (w.bits != 4 && w.bits != 6) ||
            r.begin < 0 || r.end < r.begin || r.end > p.total_rows || r.first != tiles ||
            (e > 0 && r.begin != p.ranges[e - 1].end))
            invalid("verified weights or sorted tile ranges");
        tiles += ((int64_t) r.end - r.begin + 127) / 128;
    }
    const int64_t grid = tiles * ((m + 127) / 128);
    if (tiles != p.tiles || grid <= 0 || grid > INT_MAX) invalid("grouped folded grid");
    grouped_prefill<<<(unsigned) grid, 256, kPrefillLds, (hipStream_t) stream>>>(p);
    check(hipGetLastError(), "grouped prefill launch");
}

Workspace::~Workspace() {
    if (!gate && !down) return;
    int prev = -1;
    check(hipGetDevice(&prev), "workspace device");
    check(hipSetDevice(device), "workspace owner");
    if (gate) check(hipFree(gate), "free GU A8");
    if (down) check(hipFree(down), "free down A8");
    check(hipSetDevice(prev), "restore workspace device");
}
void Workspace::reserve(int gc, int dc, int rows) {
    if (gc <= 0 || dc <= 0 || gc % 128 || dc % 128 || rows <= 0) invalid("workspace shape");
    int current = -1;
    check(hipGetDevice(&current), "reserve device");
    if (device != -1 && device != current) invalid("workspace device ownership");
    device = current;
    auto grow = [](Block*& ptr, size_t& capacity, size_t bytes) {
        if (bytes <= capacity) return;
        if (ptr) check(hipFree(ptr), "resize A8");
        check(hipMalloc((void**) &ptr, bytes), "allocate A8");
        capacity = bytes;
    };
    grow(gate, gate_bytes, (size_t) (gc / 128) * rows * sizeof(Block));
    grow(down, down_bytes, (size_t) (dc / 128) * rows * sizeof(Block));
}

void quantize(const float* x, const int32_t* src, Block* out, int cols, int ld, int rows, void* stream) {
    if (!x || !out || ld < cols) invalid("quantize operands");
    produce<false, false><<<producer_grid(cols, rows), 256, 0, (hipStream_t) stream>>>(
        x, src, out, nullptr, cols, ld, rows, 0);
    check(hipGetLastError(), "quantize launch");
}
void quantize_scatter(const float* x, const int32_t* slot, Block* out, int cols, int ld,
                      int tokens, int topk, void* stream) {
    if (!x || !slot || !out || ld < cols || topk <= 0 || (int64_t) tokens * topk > INT_MAX)
        invalid("scatter operands");
    produce<true, false><<<producer_grid(cols, tokens), 256, 0, (hipStream_t) stream>>>(
        x, slot, out, nullptr, cols, ld, tokens, topk);
    check(hipGetLastError(), "scatter launch");
}
void swiglu_quantize(const float* gu, float* h, Block* out, int ff, int rows, void* stream) {
    if (!gu || !out) invalid("SwiGLU operands");
    produce<false, true><<<producer_grid(ff, rows), 256, 0, (hipStream_t) stream>>>(
        gu, nullptr, out, h, ff, 2 * ff, rows, 0);
    check(hipGetLastError(), "SwiGLU quantize launch");
}
}  // namespace strata::prefill::iu8
#endif
