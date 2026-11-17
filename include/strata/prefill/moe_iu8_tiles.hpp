// SPDX-License-Identifier: MIT
// Port note: hipfire gemm_mq6sym_i8.gfx1201.hip narrow entry contract.
#pragma once
#include <array>

namespace strata::prefill::iu8 {
struct NarrowTile { int rows, tokens; bool split, bf16; };
inline constexpr std::array<NarrowTile, 24> kNarrowTiles{{
    {32,16,false,false}, {32,32,false,false}, {32,48,false,false},
    {64,16,false,false}, {64,32,false,false}, {64,48,false,false},
    {32,16,true,false},  {32,32,true,false},  {32,48,true,false},
    {64,16,true,false},  {64,32,true,false},  {64,48,true,false},
    {32,16,false,true},  {32,32,false,true},  {32,48,false,true},
    {64,16,false,true},  {64,32,false,true},  {64,48,false,true},
    {32,16,true,true},   {32,32,true,true},   {32,48,true,true},
    {64,16,true,true},   {64,32,true,true},   {64,48,true,true},
}};
inline int narrow_entry(int m, int n) {
    if (m <= 2560 && n >= 24) return 9;
    if (n <= 16) return m <= 2560 ? 0 : 3;
    if (n <= 32) return m >= 6144 && m <= 12288 ? 1 : 4;
    return 5;
}
}  // namespace strata::prefill::iu8
