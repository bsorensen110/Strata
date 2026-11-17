// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// Port note: hipfire block_i8_128_quant.hip and block_i4_128_quant.hip:220-343.
// Producer-owned K128 blocks, folded x-only grids, one quantization shared by GU.
#include "strata/prefill/moe_iu8_gfx.hpp"
#if defined(__HIPCC__)
#include "strata/prefill/moe_iu8_quant.cuh"
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
