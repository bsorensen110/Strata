// SPDX-License-Identifier: MIT
#pragma once

#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_FP8_ATTN_CPU_ORACLE)
#include "strata/kernels/gfx_arch.hpp"
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace strata::kernels::fp8_probe {

inline constexpr int kDim = 256;
inline constexpr int kTile = 128;
inline constexpr int kChunk = 64;  // the incumbent merge's unit, not hipfire's record layout

inline bool exact_arch(const char* arch) {
    // hipfire attention_flash_fp8_e4m3_tile_gqa.gfx1201.hip:36-38,94,136-137.
    // This qualifies packed conversion, NOT native FP8 WMMA.
    return gfx_arch_is(arch, "gfx1201");
}

inline bool resolve(const char* value, bool& on) {
    on = false;
    if (!value || std::strcmp(value, "0") == 0) return true;
    if (std::strcmp(value, "1") != 0) return false;
    on = true;
    return true;
}

enum class Refusal { none, disabled, arch, conversion, wave, shape, format, queries, bounds, buffers, count };

inline const char* reason(Refusal r) {
    switch (r) {
        case Refusal::none: return "packed-conversion FP16 read probe admitted (not FP8 WMMA)";
        case Refusal::disabled: return "probe disabled";
        case Refusal::arch: return "packed-conversion probe requires exact gfx1201; FP8 WMMA is not qualified";
        case Refusal::conversion: return "this device image lacks the guarded FP8 packed-conversion builtins";
        case Refusal::wave: return "probe requires wave32";
        case Refusal::shape: return "probe supports only D256 GQA8/GQA12";
        case Refusal::format: return "probe reads FP16 pools only; quantized KV keeps its incumbent route";
        case Refusal::queries: return "probe is decode-only (one query)";
        case Refusal::bounds: return "probe shape exceeds integer/grid bounds";
        case Refusal::buffers: return "probe requires complete FP16 pools and decode buffers";
        case Refusal::count: break;
    }
    return "invalid probe refusal";
}

inline Refusal admit(bool requested, const char* arch, bool conversions, int wave,
                     int64_t heads, int64_t kv_heads, int64_t dim, int64_t cap,
                     int64_t page_size, int64_t queries, bool fp16, bool buffers) {
    if (!requested) return Refusal::disabled;
    if (!exact_arch(arch)) return Refusal::arch;
    if (!conversions) return Refusal::conversion;
    if (wave != 32) return Refusal::wave;
    if (kv_heads <= 0 || dim != kDim || heads % kv_heads != 0 ||
        (heads / kv_heads != 8 && heads / kv_heads != 12)) return Refusal::shape;
    if (!fp16) return Refusal::format;
    if (queries != 1) return Refusal::queries;
    if (cap <= 0 || cap > INT_MAX - kTile || page_size <= 0 || page_size > INT_MAX ||
        kv_heads > 65535 || heads > INT_MAX) return Refusal::bounds;
    if (!buffers) return Refusal::buffers;
    return Refusal::none;
}

// Existing scratch = [acc][m][l], indexed by [kv_head, chunk64, local_head].
inline std::size_t record(int kv_head, int chunk, int head, int chunks, int group) {
    return (static_cast<std::size_t>(kv_head) * chunks + chunk) * group + head;
}

}  // namespace strata::kernels::fp8_probe
#endif
