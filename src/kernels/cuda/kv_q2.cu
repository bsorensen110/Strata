// src/kernels/cuda/kv_q2.cu - see include/strata/kernels/kv_q2.hpp. 2-bit KV with Walsh-Hadamard rotation.
//
// Same shape as kv_q4.cu: one warp per 32-value block, lane = element, the scale from a warp-wide maximum that
// every lane agrees on, and the same page-table addressing, host-copy write and staging-pool write, so KV
// streaming combines with it exactly as it does with q4_0.
#include "strata/kernels/kv_q2.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_stream.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_q2: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// One 32-value group in one warp (lane = element). `v3level` picks the codebook: the 4-level {-3,-1,+1,+3} for K
// (d = amax/3, so the largest magnitude lands on the outer level) and the 3-level {-1,0,+1} for V (d = amax).
// Returns the scale bits; `byte` is lane t's packed byte for t % 4 == 0 (elements 4i..4i+3 in qs[i], element j in
// bits [2*(j%4), +2)). Ties in |x| resolve to the larger value in EVERY lane, so all lanes agree on d - the same
// rule as q4's, and the reason a block never gets two scales.
__device__ __forceinline__ uint16_t q2_group(float x, int lane, uint8_t& byte, bool v3level) {
    float ss = x * x;                                                       // Lloyd-Max: the scale is the block's
#pragma unroll                                                            // RMS, not its max
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float s = sqrtf(ss * (1.0f / (float) Q2_0));
    const float is = s > 0.0f ? 1.0f / s : 0.0f;
    const float q = x * is;
    uint8_t code;
    if (v3level) {
        code = (uint8_t) (q >= kQ2Lm3T ? 2 : (q <= -kQ2Lm3T ? 0 : 1));
    } else {
        const float a = q < 0.0f ? -q : q;
        code = (uint8_t) (a < kQ2Lm4T ? (q < 0.0f ? 1 : 2) : (q < 0.0f ? 0 : 3));
    }
    // Four lanes per byte. Only lanes with t % 4 == 0 produce a byte that is stored, so the three shuffles that
    // lanes 29..31 run read out-of-range source lanes and are discarded - they are never written.
    uint8_t packed = code;
#pragma unroll
    for (int sub = 1; sub < 4; ++sub)
        packed = (uint8_t) (packed | (__shfl_sync(0xffffffffu, code, lane + sub) << (2 * sub)));
    byte = packed;
    return f16_from_f32(s);
}

__device__ __forceinline__ void q2_store(uint8_t* pool, long long row, int b, int lane, uint16_t d, uint8_t byte) {
    block_q2_0* blk = reinterpret_cast<block_q2_0*>(pool + row * (long long) sizeof(block_q2_0) * 8) + b;
    if (lane == 0) blk->d = d;
    if ((lane & 3) == 0) blk->qs[lane >> 2] = byte;
}

// One block = one 32-value group of one KV head of K (plane 0) or V (plane 1) for token step_idx; 32 threads.
__global__ void kv_append_q2_kernel(uint8_t* __restrict__ k_q2, uint8_t* __restrict__ v_q2,
                                    const int32_t* __restrict__ table, const int32_t* __restrict__ step,
                                    int step_stride, int planes, const float* __restrict__ kcur,
                                    const float* __restrict__ vcur, int kv_heads, int head_dim, int page_size,
                                    KvHostPools host, bool v3level) {
    const int step_idx = (int) (blockIdx.z / (unsigned) planes);
    const bool is_v = (blockIdx.z % (unsigned) planes) == 1u;
    const long long pos = (long long) __ldg(step + (long long) step_idx * step_stride + kStepPos);
    const int h = blockIdx.x, b = blockIdx.y, t = threadIdx.x;
    const long long step_off = (long long) step_idx * kv_heads * head_dim;
    const float x = (is_v ? vcur : kcur)[step_off + h * head_dim + b * Q2_0 + t];
    uint8_t byte;
    const uint16_t d = q2_group(x, t, byte, is_v ? v3level : false);
    const long long page = (long long) table[pos / page_size];
    if (page >= 0) q2_store(is_v ? v_q2 : k_q2, (page * kv_heads + h) * page_size + (pos % page_size), b, t, d, byte);
    if (host.k_q2 != nullptr)
        q2_store(is_v ? host.v_q2 : host.k_q2, ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size), b,
                 t, d, byte);
}

// The prompt path: T consecutive (rotated) cells from pos0, K/V [T, n_head_kv, 256]; also into the staging pool
// (identity layout) when given.
__global__ void kv_append_q2_batch_kernel(uint8_t* __restrict__ k_q2, uint8_t* __restrict__ v_q2,
                                          const int32_t* __restrict__ table, int64_t pos0,
                                          const float* __restrict__ K, const float* __restrict__ V,
                                          int kv_heads, int head_dim, int page_size, int is_v_grid, KvHostPools host,
                                          KvHostPools stage, bool v3level) {
    const long long t = blockIdx.x;
    const long long pos = pos0 + t;
    const int h = blockIdx.y, b = blockIdx.z, th = threadIdx.x;
    const bool is_v = is_v_grid != 0;
    const float x = (is_v ? V : K)[t * (kv_heads * head_dim) + h * head_dim + b * Q2_0 + th];
    uint8_t byte;
    const uint16_t d = q2_group(x, (int) th, byte, is_v ? v3level : false);
    const long long page = (long long) table[pos / page_size];
    const long long row_id = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
    if (page >= 0) q2_store(is_v ? v_q2 : k_q2, (page * kv_heads + h) * page_size + (pos % page_size), b, th, d, byte);
    if (host.k_q2 != nullptr) q2_store(is_v ? host.v_q2 : host.k_q2, row_id, b, th, d, byte);
    if (stage.k_q2 != nullptr) q2_store(is_v ? stage.v_q2 : stage.k_q2, row_id, b, th, d, byte);
}

// Gather step[kStepWidth] cells into FP16 scratch (the non-fused attention paths)
__global__ void kv_gather_q2_kernel(const uint8_t* __restrict__ k_q2, const uint8_t* __restrict__ v_q2,
                                    const int32_t* __restrict__ table, const int32_t* __restrict__ ids,
                                    const int32_t* __restrict__ step, int kv_heads, int head_dim, int page_size,
                                    uint16_t* __restrict__ k_scratch, uint16_t* __restrict__ v_scratch,
                                    bool v3level) {
    const long long n_ids = (long long) __ldg(step + kStepWidth);
    const int blocks_per_head = head_dim / Q2_0;                          // 8
    const int bytes_per_head = blocks_per_head * sizeof(block_q2_0);      // 80
    const long long total_blocks = n_ids * kv_heads * blocks_per_head;

    const long long blk_idx = (long long) blockIdx.x * blockDim.y + threadIdx.y;
    if (blk_idx >= total_blocks) return;

    const int t = threadIdx.x;
    const long long id = blk_idx / (kv_heads * blocks_per_head);
    const int rem = (int) (blk_idx % (kv_heads * blocks_per_head));
    const int h = rem / blocks_per_head;
    const int b = rem % blocks_per_head;

    const int cell = ids[id];
    const long long page = (long long) table[cell / page_size];
    const long long row = (page * kv_heads + h) * page_size + (cell % page_size);

    const block_q2_0* k_blk = reinterpret_cast<const block_q2_0*>(k_q2 + row * bytes_per_head) + b;
    const block_q2_0* v_blk = reinterpret_cast<const block_q2_0*>(v_q2 + row * bytes_per_head) + b;
    const int shift = 2 * (t & 3);
    const int kc = (k_blk->qs[t >> 2] >> shift) & 3;
    const int vc = (v_blk->qs[t >> 2] >> shift) & 3;
    const long long dst_offset = ((id * kv_heads + h) * head_dim) + (b * Q2_0 + t);
    k_scratch[dst_offset] = f16_from_f32(f32_from_f16(k_blk->d) * q2_lm4_level(kc));
    v_scratch[dst_offset] = f16_from_f32(f32_from_f16(v_blk->d) * (v3level ? q2_lm3_level(vc) : q2_lm4_level(vc)));
}

void need_256(const QsaShapes& s, const char* what) {
    if (s.head_dim != 256) {
        std::fprintf(stderr, "%s: head_dim must be 256 (the Hadamard transform's size)\n", what);
        std::exit(1);
    }
}

}  // namespace

void kv_append_q2_steps(uint8_t* k_q2, uint8_t* v_q2, const int32_t* page_table, const int32_t* step,
                        int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                        void* stream, const KvHostPools* host, bool v3level) {
    if (n_steps <= 0) return;
    need_256(s, "kv_append_q2");
    const int planes = (v_q2 == nullptr || (v_q2 == k_q2 && vcur == kcur)) ? 1 : 2;
    const dim3 grid((unsigned) s.n_head_kv, (unsigned) (s.head_dim / Q2_0), (unsigned) (n_steps * planes));
    kv_append_q2_kernel<<<grid, 32, 0, (cudaStream_t) stream>>>(
        k_q2, v_q2, page_table, step, step_stride, planes, kcur, vcur, (int) s.n_head_kv, (int) s.head_dim,
        (int) s.page_size, host ? *host : KvHostPools{}, v3level);
    check("kv_append_q2 launch");
}

void kv_append_q2_step(uint8_t* k_q2, uint8_t* v_q2, const int32_t* page_table, const int32_t* step,
                       const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host, bool v3level) {
    kv_append_q2_steps(k_q2, v_q2, page_table, step, 0, 1, kcur, vcur, s, stream, host, v3level);
}

void kv_append_q2(uint8_t* k_q2, uint8_t* v_q2, const int32_t* page_table, int64_t pos0, int64_t T, const float* K,
                  const float* V, const QsaShapes& s, void* stream, const KvHostPools* host, const KvHostPools* stage,
                  bool v3level) {
    if (T <= 0) return;
    need_256(s, "kv_append_q2");
    const dim3 grid((unsigned) T, (unsigned) s.n_head_kv, (unsigned) (s.head_dim / Q2_0));
    cudaStream_t cs = (cudaStream_t) stream;
    const KvHostPools h = host ? *host : KvHostPools{}, st = stage ? *stage : KvHostPools{};
    for (int is_v = 0; is_v < 2; ++is_v)
        kv_append_q2_batch_kernel<<<grid, 32, 0, cs>>>(k_q2, v_q2, page_table, pos0, K, V, (int) s.n_head_kv,
                                                       (int) s.head_dim, (int) s.page_size, is_v, h, st, v3level);
    check("kv_append_q2 batch launch");
}

void kv_gather_q2_step(const uint8_t* k_q2, const uint8_t* v_q2, const int32_t* page_table, const int32_t* ids,
                       const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                       uint16_t* v_scratch, void* stream, bool v3level) {
    if (max_ids <= 0) return;
    const int blocks_per_head = (int) (s.head_dim / Q2_0);
    const int64_t total_blocks = max_ids * s.n_head_kv * blocks_per_head;
    const int rows_per_block = 4;
    const unsigned num_blocks = (unsigned) ((total_blocks + rows_per_block - 1) / rows_per_block);
    kv_gather_q2_kernel<<<dim3(num_blocks), dim3(32, rows_per_block), 0, (cudaStream_t) stream>>>(
        k_q2, v_q2, page_table, ids, step, (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size, k_scratch,
        v_scratch, v3level);
    check("kv_gather_q2 launch");
}

}  // namespace strata::kernels
