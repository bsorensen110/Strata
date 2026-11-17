// SPDX-License-Identifier: MIT
// Port note: hipfire block_i8_128_quant.hip and gemm_mq6sym_i8.gfx1201.hip.
#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace strata::prefill::iu8 {
#if defined(__HIPCC__) || defined(STRATA_USE_HIP) || defined(STRATA_MOE_IU8_CPU_ORACLE)
struct Block {
    float d;
    int32_t s;
    int8_t qs[128];
};
static_assert(sizeof(Block) == 136 && offsetof(Block, qs) == 8);
constexpr int byte_index(int k) { return (k & ~7) + ((k & 1) ? 4 : 0) + ((k & 7) >> 1); }
#else
using Block = void;
static_assert(std::is_void_v<Block>);
#endif

#if defined(__HIPCC__) || defined(STRATA_USE_HIP)
// One quantization per original row, replicated into sorted route slots, not per projection.
void quantize(const float* x, const int32_t* src, Block* out, int cols, int ld, int rows, void* stream);
void quantize_scatter(const float* x, const int32_t* slot, Block* out, int cols, int ld,
                      int tokens, int topk, void* stream);
// Producer-owned down sidecar; GU is [rows][2*ff], split gate and up halves.
void swiglu_quantize(const float* gu, float* h, Block* out, int ff, int rows, void* stream);
#else
inline void quantize(...) {}
inline void quantize_scatter(...) {}
inline void swiglu_quantize(...) {}
static_assert(std::is_void_v<decltype(quantize(nullptr))>);
#endif
}  // namespace strata::prefill::iu8
