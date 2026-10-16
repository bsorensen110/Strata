// src/kernels/cuda/tcq_v.cu - device writer and reader for the trellis-coded 2-bit V (tcq_v.hpp, tcq_v_cuda.hpp).
// Built with --fmad=false (CMakeLists): the Viterbi and the scale sum must round exactly like the host reference.
#include "strata/kernels/tcq_v_cuda.hpp"

#include <cuda_runtime.h>

namespace strata::kernels {
namespace {

constexpr int kThreads = 128;

__global__ void tcq_quantize_kernel(int states, const float* __restrict__ x, block_tcq2_0* __restrict__ out,
                                    int64_t n_blocks) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_blocks) return;
    tcq_quantize_block(states, x + i * kTcqBlock, out + i);
}

// Output block j = (r * n_head_kv + head) * nb + b, nb = head_dim / 32. The source row is ids[r] (or r).
__global__ void tcq_dequantize_kernel(int states, const block_tcq2_0* __restrict__ in, float* __restrict__ out,
                                      const int32_t* __restrict__ ids, int64_t n_blocks, int64_t blocks_per_row,
                                      int64_t blocks_per_row_src) {
    const int64_t j = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_blocks) return;
    const int64_t r = j / blocks_per_row;
    const int64_t within = j - r * blocks_per_row;
    const int64_t src_row = ids ? (int64_t) ids[r] : r;
    const block_tcq2_0 b = in[src_row * blocks_per_row_src + within];
    tcq_decode_block(states, f32_from_f16(b.d), b.qs, out + j * kTcqBlock);
}

// Same walk, written as the engine's fp16 scratch (RNE), one thread per block.
__global__ void tcq_gather_f16_kernel(int states, const block_tcq2_0* __restrict__ in, uint16_t* __restrict__ out,
                                      const int32_t* __restrict__ ids, int64_t n_blocks, int64_t blocks_per_row) {
    const int64_t j = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_blocks) return;
    const int64_t r = j / blocks_per_row;
    const int64_t within = j - r * blocks_per_row;
    const int64_t src_row = ids ? (int64_t) ids[r] : r;
    const block_tcq2_0 b = in[src_row * blocks_per_row + within];
    float v[kTcqBlock];
    tcq_decode_block(states, f32_from_f16(b.d), b.qs, v);
    uint16_t* o = out + j * kTcqBlock;
    for (int t = 0; t < kTcqBlock; ++t) o[t] = f16_from_f32(v[t]);
}

}  // namespace

void tcq_v_quantize_rows(int states, const float* x, block_tcq2_0* out, int64_t n_rows, int n_head_kv, int head_dim,
                         void* stream) {
    const int64_t n_blocks = n_rows * n_head_kv * (head_dim / kTcqBlock);
    if (n_blocks <= 0) return;
    const int grid = (int) ((n_blocks + kThreads - 1) / kThreads);
    tcq_quantize_kernel<<<grid, kThreads, 0, (cudaStream_t) stream>>>(states, x, out, n_blocks);
}

void tcq_v_gather_f16(int states, const block_tcq2_0* in, uint16_t* out_f16, const int32_t* ids, int64_t n_sel,
                      int n_head_kv, int head_dim, void* stream) {
    const int64_t blocks_per_row = (int64_t) n_head_kv * (head_dim / kTcqBlock);
    const int64_t n_blocks = n_sel * blocks_per_row;
    if (n_blocks <= 0) return;
    const int grid = (int) ((n_blocks + kThreads - 1) / kThreads);
    tcq_gather_f16_kernel<<<grid, kThreads, 0, (cudaStream_t) stream>>>(states, in, out_f16, ids, n_blocks,
                                                                       blocks_per_row);
}

void tcq_v_dequantize_rows(int states, const block_tcq2_0* in, float* out, const int32_t* ids, int64_t n_sel,
                           int n_head_kv, int head_dim, void* stream) {
    const int64_t blocks_per_row = (int64_t) n_head_kv * (head_dim / kTcqBlock);
    const int64_t n_blocks = n_sel * blocks_per_row;
    if (n_blocks <= 0) return;
    const int grid = (int) ((n_blocks + kThreads - 1) / kThreads);
    tcq_dequantize_kernel<<<grid, kThreads, 0, (cudaStream_t) stream>>>(states, in, out, ids, n_blocks,
                                                                       blocks_per_row, blocks_per_row);
}

}  // namespace strata::kernels
