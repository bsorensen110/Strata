#pragma once

#include <algorithm>
#include <cstddef>

namespace strata::core {

struct Df2SlotBudgetInput {
    int room = 0;               ///< Max extra draft rows this slot can accept (0..kMaxDrafts)
    bool draft_ready = false;   ///< True if drafts and dprob[] are populated for this slot
    const float* dprob = nullptr;
};

/// Greedily allocates up to `spare` extra verify rows across `n_slots` active slots
/// by picking at each step the slot whose next unallocated draft position has the
/// highest cumulative chain probability `prod_{j=0..k} dprob[j] >= conf_thresh`.
/// Because `0 <= dprob[j] <= 1`, cumulative chain probabilities within each slot are
/// monotonically non-increasing, so greedy selection is globally EV-optimal and
/// always produces a contiguous prefix of size `out_extra[s]` for each slot `s`.
inline void allocate_dprob_spare_rows(const Df2SlotBudgetInput* slots,
                                      int n_slots,
                                      int spare,
                                      double conf_thresh,
                                      int* out_extra) {
    if (n_slots <= 0 || !out_extra) return;
    double chain[64];
    const int n = std::min(n_slots, 64);
    for (int s = 0; s < n_slots; ++s) out_extra[s] = 0;
    for (int s = 0; s < n; ++s) chain[s] = 1.0;
    if (!slots || spare <= 0) return;

    for (int step = 0; step < spare; ++step) {
        int best_s = -1;
        double best_chain = -1.0;
        for (int s = 0; s < n; ++s) {
            if (!slots[s].draft_ready || !slots[s].dprob) continue;
            const int idx = out_extra[s];
            if (idx >= slots[s].room) continue;
            const float p = slots[s].dprob[idx];
            if (!(p > 0.0f)) continue;
            const double cand = chain[s] * (double) p;
            if (conf_thresh > 0.0 && cand < conf_thresh) continue;
            if (cand > best_chain) {
                best_chain = cand;
                best_s = s;
            }
        }
        if (best_s < 0) break;
        chain[best_s] = best_chain;
        ++out_extra[best_s];
    }
}

/// Computes the per-slot `k_want` ceiling for the batched DFlash2 head launch.
/// When `dprob_budget_on` is true, allows each slot to generate up to `8 / nact_next`
/// drafts (e.g., 4 per slot at 2 active slots, keeping total head rows <= 8 so
/// `record_batch_head` stays a single interleaved MMVQ launch) so a higher-confidence
/// slot can claim spare verify rows released by a lower-confidence peer.
inline int batch_df2_head_max_per_slot(int spare_next,
                                       int nact_next,
                                       bool dprob_budget_on,
                                       int env_override = 0) {
    if (env_override > 0) return env_override;
    const int n = std::max(1, nact_next);
    const int even_ceil = std::max(1, (std::max(0, spare_next) + n - 1) / n);
    if (!dprob_budget_on || spare_next <= 0) return even_ceil;
    const int il_cap_per_slot = std::max(even_ceil, 8 / n);
    return std::min(spare_next, std::max(even_ceil, il_cap_per_slot));
}

/// Opt-in (STRATA_BATCH_DF2_ADAPTIVE_T=1): adaptively clamps the DFlash2 verify window capacity between
/// [max(active_slots, target_rows), max_win_cap] using the smoothed draft acceptance rate. Keeping low-acceptance
/// streams near target_rows (default 8) avoids wide >8-row verify penalty while letting high-acceptance streams expand.
inline int compute_adaptive_df2_win_cap(int active_slots,
                                        int max_win_cap,
                                        int target_rows,
                                        float ema_accept_rate,
                                        bool adaptive_on) {
    if (!adaptive_on) return max_win_cap;
    const int floor_rows = std::max(1, active_slots);
    const int ceil_rows = std::max(floor_rows, max_win_cap);
    const int tgt = std::clamp(target_rows, floor_rows, ceil_rows);
    if (ema_accept_rate <= 0.35f) return tgt;
    if (ema_accept_rate >= 0.70f) return ceil_rows;
    const float t = (ema_accept_rate - 0.35f) / 0.35f;
    return std::clamp(tgt + (int) ((float) (ceil_rows - tgt) * t + 0.5f), tgt, ceil_rows);
}

}  // namespace strata::core
