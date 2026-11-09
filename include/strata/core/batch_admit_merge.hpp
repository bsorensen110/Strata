#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::core {

/// Computes the prompt token position `merge_at` from which the trailing segment `[a0, to)` is merged
/// with the final prompt token `n - 1` into the first decode/admission verify window (`[merge_at, n)`).
/// Returns -1 when no merge should occur.
inline int64_t admit_merge_from(int64_t a0, int64_t to, int64_t n, int mt,
                                bool all_ar, bool single_stage_hybrid, int64_t win_w) noexcept {
    if (mt < 2 || to <= a0 || to != n - 1) return -1;
    if (all_ar) return std::max(a0, n - (int64_t) mt);
    if (single_stage_hybrid && win_w >= 2) {
        const int64_t rem = (to - a0) % win_w;
        if (rem > 0 && rem + 1 <= (int64_t) mt) return to - rem;
    }
    return -1;
}

/// Computes the remaining inter-chunk decode budget (ms) after a prompt prefill chunk of `wall_ms`
/// that already spent `mid_chunk_decode_ms` inside mid-chunk layer-yield decode steps.
inline double inter_chunk_decode_budget_ms(double eff_share, double wall_ms,
                                           double mid_chunk_decode_ms, bool net_share = true) noexcept {
    if (eff_share <= 0.0 || wall_ms <= 0.0) return 0.0;
    const double mid = net_share ? std::max(0.0, mid_chunk_decode_ms) : 0.0;
    const double pf_ms = std::max(0.0, wall_ms - mid);
    return std::max(0.0, eff_share * pf_ms - mid);
}

/// Returns the number of written DFlash2 KV ring slots `[0, n_slots)` that must be copied by
/// `Dflash2Drafter::copy_state_from` given `src.ring_hi_` and ring capacity `k_ring`.
inline int dflash2_ring_copy_slots(int64_t ring_hi, int k_ring) noexcept {
    if (ring_hi <= 0 || k_ring <= 0) return 0;
    return ring_hi >= (int64_t) k_ring ? k_ring : (int) ring_hi;
}

/// Returns true when `Verifier::instantiate_evicting` should proactively evict the least-recently-used
/// captured batch window graph (`exec_bm_`) before instantiating a new graph so free VRAM stays at or
/// above `floor_vram_bytes` (preventing ROCm `<512 MiB` GTT paging `#2065`).
inline bool should_evict_batch_graph_for_vram(size_t free_vram_bytes, size_t floor_vram_bytes,
                                              size_t cached_batch_graphs) noexcept {
    if (floor_vram_bytes == 0 || cached_batch_graphs == 0) return false;
    return free_vram_bytes < floor_vram_bytes;
}

}  // namespace strata::core
