// SPDX-License-Identifier: MIT
// Port note: hipfire narrow IU8 and IU4 tile/operand contracts.
// Explicit precision-qualified table; never alias Q4 sidecars as Q8 MMQ.
#pragma once
#if defined(__HIPCC__) || defined(STRATA_USE_HIP) || defined(STRATA_MOE_IU8_CPU_ORACLE) || defined(STRATA_MOE_IU4_CPU_ORACLE)
#include "strata/core/arch_defaults.hpp"
#include "strata/prefill/moe_iu8_tiles.hpp"
#include "ggml.h"
#include <type_traits>

namespace strata::prefill::mmq {
enum class HipEntry { None, Iu8Prefill, Iu8Narrow, Iu4K32, Iu4K16 };
struct HipConfig {
    HipEntry entry;
    ggml_prec precision;
    int rows, tokens, threads, lds, k_step, narrow;
    bool split, bf16;
};
inline constexpr HipConfig kHipNone{HipEntry::None, GGML_PREC_DEFAULT, 0,0,0,0,0,-1,false,false};
inline constexpr HipConfig kHipIu4K32{HipEntry::Iu4K32, GGML_PREC_Q4, 128,128,256,20480,32,-1,false,false};
inline constexpr HipConfig kHipIu4K16{HipEntry::Iu4K16, GGML_PREC_Q4, 128,128,256,30720,16,-1,false,false};
inline constexpr HipConfig kHipIu8Prefill{HipEntry::Iu8Prefill, GGML_PREC_Q8, 128,128,256,36864,16,-1,false,false};
inline constexpr auto kHipNarrowConfigs = [] {
    std::array<HipConfig,24> table{};
    for (size_t i = 0; i < table.size(); ++i) {
        const auto t = iu8::kNarrowTiles[i];
        table[i] = {HipEntry::Iu8Narrow, GGML_PREC_Q8, t.rows,t.tokens,t.rows/16*32,
                    t.tokens*136,16,(int)i,t.split,t.bf16};
    }
    return table;
}();
static_assert(std::is_standard_layout_v<HipConfig> && std::is_aggregate_v<HipConfig>);
static_assert(kHipIu4K32.precision == GGML_PREC_Q4 && kHipIu4K32.lds == 20480 && kHipIu4K32.k_step == 32);
static_assert(kHipIu4K16.precision == GGML_PREC_Q4 && kHipIu4K16.lds == 30720 && kHipIu4K16.k_step == 16);
static_assert(kHipIu8Prefill.precision == GGML_PREC_Q8 && kHipIu8Prefill.threads == 256);
static_assert(kHipNarrowConfigs[23].narrow == 23 && kHipNarrowConfigs[23].split && kHipNarrowConfigs[23].bf16);

inline HipConfig hip_config(ggml_prec precision, const char* arch, int m, int k, int n,
                             int weight_bits, bool window, int narrow = -1) {
    if (m <= 0 || k <= 0 || k % 128 || n <= 0) return kHipNone;
    if (precision == GGML_PREC_Q4) {
        if (weight_bits != 4 || !strata::core::hip_moe_iu4_capable(arch)) return kHipNone;
        return strata::kernels::gfx_arch_is(arch,"gfx1201") ? kHipIu4K32 : kHipIu4K16;
    }
    if (precision != GGML_PREC_Q8 || (weight_bits != 4 && weight_bits != 6) ||
        !strata::core::hip_moe_iu8_capable(arch)) return kHipNone;
    if (!window) return n >= 64 ? kHipIu8Prefill : kHipNone;
    if (n > 48 || narrow < -1 || narrow >= (int) kHipNarrowConfigs.size()) return kHipNone;
    const auto c = kHipNarrowConfigs[narrow < 0 ? iu8::narrow_entry(m,n) : narrow];
    return !c.split && n > c.tokens ? kHipNone : c;
}
}  // namespace strata::prefill::mmq
#endif
