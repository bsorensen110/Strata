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

/// Returns true when single-stage hybrid MoE batch admission may merge a partial read_windows remainder
/// with the final prompt token into a single admission verify window, regardless of dynamic pcie_frac.
inline bool single_stage_hybrid_admit_merge(bool merge_rem_on, int n_pipe, bool verifier0_live) noexcept {
    return merge_rem_on && n_pipe == 1 && verifier0_live;
}

/// Finds the prompt's own last turn checkpoint boundary in `(0, n - 1)` regardless of `resume`
/// (so a `from_live` BGEN continuation whose `resume > turn_at` still recognizes and moves its own
/// turn checkpoint in O(1)), returning -1 when disabled or when the boundary equals `root_at`.
template <class Tok>
inline int64_t prompt_last_turn_boundary(const Tok* ids, int64_t n, int64_t turn_token,
                                         int64_t tail_role_token, bool req_ckpt, int64_t root_at) noexcept {
    if (!req_ckpt || ids == nullptr || n <= 1 || turn_token < 0) return -1;
    int64_t turn_at = -1;
    for (int64_t i = n - 1; i > 0; --i) {
        if ((int64_t) ids[i] == turn_token) {
            turn_at = i;
            break;
        }
    }
    if (turn_at > 0 && tail_role_token >= 0) {
        for (int64_t i = turn_at - 1; i > 0; --i) {
            if ((int64_t) ids[i] == turn_token) {
                if ((int64_t) ids[i + 1] == tail_role_token) turn_at = i;
                break;
            }
        }
    }
    if (turn_at <= 0 || turn_at == root_at) return -1;
    return turn_at;
}

/// Returns true when checkpoints in `bs[slot_source].checks` should be moved in O(1) rather than copied:
/// either on a same-slot BGEN continuation (`admit_slot == slot_source`), or on a full slot->solo GEN
/// handoff (`admit_slot < 0 && slot_source >= 0 && !has_slot_ck`) when `solo_handoff_fast` is enabled.
inline bool should_move_slot_checkpoints_to_solo(bool slot_ckpt_move, bool solo_handoff_fast,
                                                 int admit_slot, int slot_source, bool has_slot_ck) noexcept {
    if (!slot_ckpt_move || slot_source < 0) return false;
    if (admit_slot == slot_source) return true;
    return solo_handoff_fast && admit_slot < 0 && !has_slot_ck;
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
