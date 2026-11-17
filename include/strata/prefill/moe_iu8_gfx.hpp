// SPDX-License-Identifier: MIT
// Port note: hipfire block_i8_128_quant.hip and gemm_mq6sym_i8.gfx1201.hip.
#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include "strata/prefill/moe_iu8_tiles.hpp"

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
struct WeightView {
    const uint8_t* codes = nullptr;
    const uint16_t* scales = nullptr;
    int rows = 0, cols = 0, bits = 0, device = -1;
    bool symmetric_verified = false;
};
struct TileRange { int32_t begin = 0, end = 0, first = 0; };
struct Product {
    WeightView weights[16];
    TileRange ranges[16];
    int n = 0, tiles = 0, total_rows = 0, tokens = 0;
    const Block* x = nullptr;
    const int32_t* ids = nullptr;
    float* dst = nullptr;
    int64_t ld = 0;
};
inline constexpr int kPrefillLds = 36864;
bool admitted(int tokens, int cols, void* stream);
void run(const Product& p, void* stream);

// Null count: one matrix. Otherwise one device-planned expert group per CTA
// row tile; starts/tok/dst are the verifier's existing sorted route tables.
struct Window {
    WeightView weight;
    const WeightView* table = nullptr;
    const int32_t *count = nullptr, *starts = nullptr, *tok = nullptr, *dst = nullptr, *route = nullptr;
    const Block* x = nullptr;
    void* y = nullptr;
    int rows = 0, cols = 0, tokens = 0, x_rows = 0, groups = 1, topk = 0;
    int64_t ld = 0;
    bool sorted_x = false, sorted_y = false;
};
bool window_admitted(int tokens, int cols);
void run_narrow(const Window& p, int entry, void* stream);
void quantize_window(const float* x, Block* out, int cols, int rows, void* stream);
void swiglu_window(const float* gu, float* h, Block* out, int ff, int rows,
                   const int32_t* starts, const int32_t* count, void* stream);

struct Workspace {
    Block *gate = nullptr, *down = nullptr;
    size_t gate_bytes = 0, down_bytes = 0;
    int device = -1;
    Workspace() = default;
    ~Workspace();
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    void reserve(int gate_cols, int down_cols, int rows);
};
// One quantization per original row, replicated into sorted route slots, not per projection.
void quantize(const float* x, const int32_t* src, Block* out, int cols, int ld, int rows, void* stream);
void quantize_scatter(const float* x, const int32_t* slot, Block* out, int cols, int ld,
                      int tokens, int topk, void* stream);
// Producer-owned down sidecar; GU is [rows][2*ff], split gate and up halves.
void swiglu_quantize(const float* gu, float* h, Block* out, int ff, int rows, void* stream);
#else
using WeightView = void;
using Product = void;
inline void run(...) {}
static_assert(std::is_void_v<WeightView> && std::is_void_v<Product>);
inline void quantize(...) {}
inline void quantize_scatter(...) {}
inline void swiglu_quantize(...) {}
static_assert(std::is_void_v<decltype(quantize(nullptr))>);
#endif
}  // namespace strata::prefill::iu8
