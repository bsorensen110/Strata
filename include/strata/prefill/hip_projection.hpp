// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// Port note: hipfire gemm_mq6sym_i8.gfx1201.hip:146-406.
// Strata adaptation: planar exact W4/W6 packs, unchanged K128 arithmetic.
#pragma once
#include "strata/prefill/moe_iu8_tiles.hpp"
#include <array>
#include <climits>
#include <cstdint>
#if defined(__HIPCC__) || defined(STRATA_USE_HIP)
#include "strata/prefill/moe_iu8_gfx.hpp"
#endif

namespace strata::prefill::iu8 {
struct X4Plan {
    int entry = -1;
    uint32_t x = 0, y = 0;
};
inline X4Plan x4_plan(const std::array<int, 4>& rows, int count, int k, int n) {
    if (count < 2 || count > 4 || k <= 0 || k % 128 || n < 9 || n > 64) return {};
    int largest = 0;
    for (int i = 0; i < 4; ++i) {
        if (i < count ? (rows[i] <= 0 || rows[i] > INT_MAX-127) : rows[i] != 0) return {};
        if (rows[i] > largest) largest = rows[i];
    }
    const int entry = n > 48 ? 9 : narrow_entry(largest, n);
    const auto t = kNarrowTiles[entry];
    if (t.bf16 || (!t.split && n > t.tokens)) return {};
    int64_t blocks = 0;
    for (int m : rows) blocks += ((int64_t) m + t.rows - 1) / t.rows;
    const int slices = t.split ? (n + t.tokens - 1) / t.tokens : 1;
    if (blocks <= 0 || blocks > INT_MAX || slices > 65536) return {};
    return {entry, (uint32_t) blocks, (uint32_t) slices};
}

#if defined(__HIPCC__) || defined(STRATA_USE_HIP)
// F32 overwrite only; inactive members are zero-initialized. The opt-in is
// frozen before graph capture, not read lazily from the process environment.
bool strata_gemm_sym_i8_x4(const std::array<Window, 4>& members, int count, bool enabled, void* stream);
void launch_sym_i8_x4(const std::array<Window, 4>& members, const X4Plan& plan, void* stream);
// Internal standalone twin, also used to check the >48-row split entry.
void launch_sym_i8_member(const Window& member, int entry, void* stream);
#endif
}  // namespace strata::prefill::iu8
