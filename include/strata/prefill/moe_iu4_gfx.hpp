// SPDX-License-Identifier: MIT
// Port note: hipfire block_i4_128_quant.hip, reference eight-candidate policy.
#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace strata::prefill::iu4 {
#if defined(__HIPCC__) || defined(STRATA_USE_HIP) || defined(STRATA_MOE_IU4_CPU_ORACLE)
struct Block { float d; int32_t s; uint8_t qs[64]; };
static_assert(sizeof(Block) == 72 && offsetof(Block, qs) == 8);
inline constexpr int kCandidates = 8;  // reference search, not hipfire's separately configured C2 emit
inline int code(const Block& b, int k) {
    const int u = (b.qs[k / 2] >> (4 * (k & 1))) & 15;
    return u < 8 ? u : u - 16;
}
#else
using Block = void;
static_assert(std::is_void_v<Block>);
#endif
#if defined(__HIPCC__) || defined(STRATA_USE_HIP)
void quantize(const float* x, Block* out, int cols, int rows, void* stream);
void swiglu_window(const float* gu, float* h, Block* out, int ff, int rows,
                   const int32_t* starts, const int32_t* count, void* stream);
#else
inline void quantize(...) {}
inline void swiglu_window(...) {}
#endif
}  // namespace strata::prefill::iu4
