// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{norm.cu,common.cuh,unary.cu}.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "strata/kernels/native_qsa.hpp"
#define STRATA_Q81_FASTMATH_DIV 1
#include "strata/kernels/q8_1_finite.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        value += __shfl_xor_sync(0xffffffffu, value, offset, 32);
    return value;
}
template<int BlockSize>
__global__ void norm(const float* input, const float* __restrict__ gamma, float* output,
                     int n_cols, float epsilon) {
    const int tid = threadIdx.x;
    const std::size_t row_offset = std::size_t(blockIdx.x) * n_cols;
    input += row_offset; output += row_offset;
    float partial = 0.0f;
    for (std::size_t col = tid; col < std::size_t(n_cols); col += BlockSize) {
        const float x = input[col];
        partial += x * x;
    }
    __shared__ float sums[32];
    partial = warp_sum(partial);
    const int lane = tid % 32;
    if (lane == 0) sums[tid / 32] = partial;
    // All reads of input for the reduction precede this barrier. Afterwards,
    // each thread reads/writes only its own elements, allowing exact in-place use.
    __syncthreads();
    partial = lane < BlockSize / 32 ? sums[lane] : 0.0f;
    partial = warp_sum(partial);
    const float mean = partial / n_cols;
    const float scale = rsqrtf(mean + epsilon);
    for (std::size_t col = tid; col < std::size_t(n_cols); col += BlockSize)
        output[col] = scale * input[col] * gamma[col];
}
// As norm, with gamma[groups * n_cols]: row r is group r % groups and reads its own slice of gamma.
template<int BlockSize>
__global__ void norm_grouped(const float* input, const float* __restrict__ gamma, float* output,
                             int n_cols, int groups, float epsilon) {
    const int tid = threadIdx.x;
    const std::size_t row_offset = std::size_t(blockIdx.x) * n_cols;
    input += row_offset; output += row_offset;
    gamma += std::size_t(blockIdx.x % unsigned(groups)) * n_cols;
    float partial = 0.0f;
    for (std::size_t col = tid; col < std::size_t(n_cols); col += BlockSize) {
        const float x = input[col];
        partial += x * x;
    }
    __shared__ float sums[32];
    partial = warp_sum(partial);
    const int lane = tid % 32;
    if (lane == 0) sums[tid / 32] = partial;
    __syncthreads();
    partial = lane < BlockSize / 32 ? sums[lane] : 0.0f;
    partial = warp_sum(partial);
    const float mean = partial / n_cols;
    const float scale = rsqrtf(mean + epsilon);
    for (std::size_t col = tid; col < std::size_t(n_cols); col += BlockSize)
        output[col] = scale * input[col] * gamma[col];
}
__global__ void gate(const float* attn, const float* __restrict__ q_full, float* output,
                     int n_head, int head_dim) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= std::size_t(n_head) * head_dim) return;
    const std::size_t head = i / head_dim, channel = i % head_dim;
    const float raw = q_full[head * 2 * head_dim + head_dim + channel];
    const float sigmoid = 1.0f / (1.0f + expf(-raw));
    output[i] = attn[i] * sigmoid;
}
struct QsaQ81Block { half2 ds; int8_t qs[32]; };
__global__ void gate_quantize_q8_1_kernel(const float* __restrict__ attn, const float* __restrict__ q_full,
                                          QsaQ81Block* __restrict__ xq, int n_head, int head_dim,
                                          int* __restrict__ bm, int* __restrict__ pm, float* __restrict__ dl,
                                          int n_in, int cp) {
#pragma clang fp contract(off)
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = std::size_t(n_head) * head_dim;
    float xi = 0.0f;
    if (i < count) {
        const std::size_t head = i / head_dim, channel = i % head_dim;
        const float raw = q_full[head * 2 * head_dim + head_dim + channel];
        const float sigmoid = 1.0f / (1.0f + expf(-raw));
        xi = attn[i] * sigmoid;
    }
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (i < count) {
        const float d = q8_1_finite(q81_div(amax, 127.0f));
        const int8_t q = q8_1_quant(xi, d, amax);
        xq[i / 32].qs[i % 32] = q;
        const auto ds = q8_1_ds(d, sum);
        if (i % 32 == 0) xq[i / 32].ds = ds;
        if (bm != nullptr) {
            const int lane = int(threadIdx.x) & 31;
            unsigned v = unsigned(uint8_t(q)) << (8 * (lane & 3));
            v |= __shfl_xor_sync(0xffffffffu, v, 1);
            v |= __shfl_xor_sync(0xffffffffu, v, 2);
            const int nb = n_in / 32, c = int(i / std::size_t(n_in)), b = int((i - std::size_t(c) * n_in) / 32), pp = lane >> 2;
            if ((lane & 3) == 0) {
                bm[(std::size_t(b) * 8 + pp) * cp + c] = int(v);
                pm[(std::size_t(pp) * nb + b) * cp + c] = int(v);
            }
            if (lane == 0) dl[std::size_t(b) * cp + c] = __low2float(ds);
        }
    }
}

__global__ void fwht256_gate_quantize_q8_1_kernel(const float* __restrict__ attn, const float* __restrict__ q_full,
                                                  QsaQ81Block* __restrict__ xq, int n_head,
                                                  int* __restrict__ bm, int* __restrict__ pm, float* __restrict__ dl,
                                                  int n_in, int cp) {
#pragma clang fp contract(off)
    const int r = int(blockIdx.x) * int(blockDim.y) + int(threadIdx.y);
    if (r >= n_head) return;

    constexpr int warp_size = 32;
    constexpr int N = 256;
    constexpr int el_w = N / warp_size;   // 8
    constexpr float scale = 1.0f / 16.0f;

    const float* row_src = attn + std::size_t(r) * N;
    const float* row_gate = q_full + std::size_t(r) * (2 * N) + N;
    const int lane = int(threadIdx.x);

    float reg[el_w];
#pragma unroll
    for (int i = 0; i < el_w; ++i) reg[i] = row_src[i * warp_size + lane] * scale;

#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val = reg[j];
            const float val2 = __shfl_xor_sync(0xffffffffu, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }
#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k] = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

    const int nb = bm != nullptr ? n_in / 32 : 0;
#pragma unroll
    for (int b_h = 0; b_h < el_w; ++b_h) {
        const int col = b_h * warp_size + lane;
        const float raw = row_gate[col];
        const float sigmoid = 1.0f / (1.0f + expf(-raw));
        const float xi = reg[b_h] * sigmoid;

        float amax = fabsf(xi), sum = xi;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);

        const std::size_t blk = std::size_t(r) * el_w + std::size_t(b_h);
        const float d = q8_1_finite(q81_div(amax, 127.0f));
        const int8_t q = q8_1_quant(xi, d, amax);
        xq[blk].qs[lane] = q;
        const auto ds = q8_1_ds(d, sum);
        if (lane == 0) xq[blk].ds = ds;
        if (bm != nullptr) {
            unsigned v = unsigned(uint8_t(q)) << (8 * (lane & 3));
            v |= __shfl_xor_sync(0xffffffffu, v, 1);
            v |= __shfl_xor_sync(0xffffffffu, v, 2);
            const int c = int(blk / std::size_t(nb)), b = int(blk % std::size_t(nb)), pp = lane >> 2;
            if ((lane & 3) == 0) {
                bm[(std::size_t(b) * 8 + pp) * cp + c] = int(v);
                pm[(std::size_t(pp) * nb + b) * cp + c] = int(v);
            }
            if (lane == 0) dl[std::size_t(b) * cp + c] = __low2float(ds);
        }
    }
}
std::size_t elements(int cols, int rows) {
    if (cols <= 0 || rows <= 0 || std::uint64_t(cols) * rows > std::uint64_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("native QSA requires positive bounded dimensions");
    return std::size_t(cols) * rows;
}
bool valid(const void* ptr, std::size_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(ptr);
    return ptr && address % 4 == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, std::size_t an, const void* b, std::size_t bn) {
    const auto ap = reinterpret_cast<std::uintptr_t>(a), bp = reinterpret_cast<std::uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
void buffers(const float* input, std::size_t in_bytes, const float* weight, std::size_t weight_bytes,
             float* output, void* stream) {
    if (!stream || !valid(input, in_bytes) || !valid(weight, weight_bytes) || !valid(output, in_bytes) ||
        overlap(input, in_bytes, weight, weight_bytes) || overlap(output, in_bytes, weight, weight_bytes) ||
        (input != output && overlap(input, in_bytes, output, in_bytes)))
        throw std::invalid_argument("native QSA requires a stream, aligned spans, and disjoint buffers or exact input/output alias");
}
void check_launch() {
    const auto result = cudaGetLastError();
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("native QSA launch: ") + cudaGetErrorString(result));
}
} // namespace

void native_qsa_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_qsa_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_qsa_rms_norm_weighted(const float* input, const float* gamma, float* output,
                                  int n_cols, int n_rows, float epsilon, void* stream) {
    const auto count = elements(n_cols, n_rows);
    if (!std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native QSA requires finite nonnegative epsilon");
    buffers(input, count * 4, gamma, std::size_t(n_cols) * 4, output, stream);
    if (n_cols < 1024)
        norm<256><<<unsigned(n_rows), 256, 0, static_cast<cudaStream_t>(stream)>>>(input, gamma, output, n_cols, epsilon);
    else
        norm<1024><<<unsigned(n_rows), 1024, 0, static_cast<cudaStream_t>(stream)>>>(input, gamma, output, n_cols, epsilon);
    check_launch();
}
void native_qsa_rms_norm_grouped(const float* input, const float* gamma, float* output,
                                 int n_cols, int groups, int n_rows, float epsilon, void* stream) {
    if (groups < 1) throw std::invalid_argument("native QSA requires positive bounded dimensions");
    const auto count = elements(n_cols, n_rows);
    if (!std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native QSA requires finite nonnegative epsilon");
    buffers(input, count * 4, gamma, std::size_t(n_cols) * groups * 4, output, stream);
    if (n_cols < 1024)
        norm_grouped<256><<<unsigned(n_rows), 256, 0, static_cast<cudaStream_t>(stream)>>>(input, gamma, output, n_cols,
                                                                                         groups, epsilon);
    else
        norm_grouped<1024><<<unsigned(n_rows), 1024, 0, static_cast<cudaStream_t>(stream)>>>(input, gamma, output, n_cols,
                                                                                           groups, epsilon);
    check_launch();
}
void native_qsa_gate_apply(const float* attn, const float* q_full, float* output,
                           int n_head, int head_dim, void* stream) {
    const auto count = elements(head_dim, n_head);
    buffers(attn, count * 4, q_full, count * 8, output, stream);
    gate<<<unsigned((count + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(attn, q_full, output, n_head, head_dim);
    check_launch();
}
void native_qsa_gate_quantize_q8_1(const float* attn, const float* q_full, void* x_q8_1,
                                   int n_head, int head_dim, void* stream, void* x_il, int ncols, bool fwht256) {
    const auto count = elements(head_dim, n_head);
    if (head_dim % 32 != 0 || (fwht256 && head_dim != 256) || !stream || !valid(attn, count * 4) ||
        !valid(q_full, count * 8) || !x_q8_1)
        throw std::invalid_argument("native_qsa_gate_quantize_q8_1 requires head_dim % 32 == 0 and valid buffers");
    int *bm = nullptr, *pm = nullptr;
    float* dl = nullptr;
    int n_in = 0, cp = 0;
    if (x_il != nullptr && ncols >= 2 && ncols <= 8 && (n_head % ncols) == 0) {
        n_in = (n_head / ncols) * head_dim;
        cp = ncols <= 2 ? 2 : (ncols <= 4 ? 4 : 8);
        const std::size_t words = std::size_t(n_in / 32) * 8 * std::size_t(cp);
        bm = static_cast<int*>(x_il);
        pm = bm + words;
        dl = reinterpret_cast<float*>(bm + 2 * words);
    }
    if (fwht256) {
        constexpr int rows_per_block = 4;
        const dim3 grid(unsigned((n_head + rows_per_block - 1) / rows_per_block));
        const dim3 block(32, rows_per_block);
        fwht256_gate_quantize_q8_1_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
            attn, q_full, static_cast<QsaQ81Block*>(x_q8_1), n_head, bm, pm, dl, n_in, cp);
    } else {
        gate_quantize_q8_1_kernel<<<unsigned((count + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
            attn, q_full, static_cast<QsaQ81Block*>(x_q8_1), n_head, head_dim, bm, pm, dl, n_in, cp);
    }
    check_launch();
}
} // namespace strata::kernels
