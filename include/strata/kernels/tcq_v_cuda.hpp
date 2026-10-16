// include/strata/kernels/tcq_v_cuda.hpp - device writer and reader for the trellis-coded 2-bit V (tcq_v.hpp).
//
// Layout (per-token rows, the same order as the q2/q4 pools without a page table):
//   fp32 rows     x[n_rows][n_head_kv][head_dim]
//   block pool    out[n_rows][n_head_kv][head_dim / 32]  (block_tcq2_0, 10 bytes each)
// The writer is one thread per block; the reader is one thread per block (32 sequential trellis steps).
// The reader can select rows: ids (device int32, length n_sel) picks the source row for each output row; ids ==
// nullptr reads rows 0..n_sel-1 in order. All pointers are device pointers. `stream` is a cudaStream_t.
#pragma once

#include <cstdint>

#include "strata/kernels/tcq_v.hpp"

namespace strata::kernels {

void tcq_v_quantize_rows(int states, const float* x, block_tcq2_0* out, int64_t n_rows, int n_head_kv, int head_dim,
                         void* stream);

void tcq_v_dequantize_rows(int states, const block_tcq2_0* in, float* out, const int32_t* ids, int64_t n_sel,
                           int n_head_kv, int head_dim, void* stream);

/// Gather to the engine's fp16 scratch format (round-to-nearest-even bits, f16_from_f32). The same row selection
/// as tcq_v_dequantize_rows; out_f16 is [n_sel][n_head_kv][head_dim] of uint16 bits.
void tcq_v_gather_f16(int states, const block_tcq2_0* in, uint16_t* out_f16, const int32_t* ids, int64_t n_sel,
                      int n_head_kv, int head_dim, void* stream);

}  // namespace strata::kernels
