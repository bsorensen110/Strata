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
struct WeightView {
    const uint8_t* codes = nullptr;
    const uint16_t *scales = nullptr, *zero_points = nullptr;
    int rows = 0, cols = 0, device = -1;
    bool symmetric_verified = false;
};
struct Product {
    WeightView weight;
    const WeightView* table = nullptr;
    const int32_t *count = nullptr, *starts = nullptr, *tok = nullptr, *dst = nullptr, *route = nullptr;
    const Block* x = nullptr;
    float* y = nullptr;
    int rows = 0, cols = 0, tokens = 0, x_rows = 0, groups = 1;
    int64_t ld = 0;
    bool sorted_x = false, sorted_y = false, symmetric = true, banded = false;
};
inline constexpr int kGfx12Lds = 20480, kGfx11Lds = 30720;
bool usable();
void run(const Product& p, void* stream);
void quantize(const float* x, Block* out, int cols, int rows, void* stream);
void swiglu_window(const float* gu, float* h, Block* out, int ff, int rows,
                   const int32_t* starts, const int32_t* count, void* stream);
#else
using Product = void;
using WeightView = void;
inline void run(...) {}
inline void quantize(...) {}
inline void swiglu_window(...) {}
#endif
}  // namespace strata::prefill::iu4
