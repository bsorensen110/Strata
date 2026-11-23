// include/strata/core/tail_skip_policy.hpp - per-row anchor/draft tail-skip isolation policy.
#pragma once

namespace strata::core {

/// Decides whether a missed distinct expert (routed across the window at minimum rank `min_rank`, and
/// touching at least one anchor/committed row iff `any_anchor_row` is true) may be tail-skipped.
/// - `base_tail_rank` (STRATA_ROUTE_TAIL_SKIP): applies uniformly to all rows when > 0.
/// - `batch_tail_rank` / `draft_tail_rank` (STRATA_BATCH_TAIL_SKIP / STRATA_DRAFT_TAIL_SKIP): apply ONLY when
///   every token routing the expert is a speculative draft row (`!any_anchor_row`), so committed/anchor rows of
///   any batch slot are never degraded.
inline bool should_tail_skip_expert(int min_rank,
                                    bool any_anchor_row,
                                    int base_tail_rank,
                                    int batch_tail_rank,
                                    int draft_tail_rank) noexcept {
    if (base_tail_rank > 0 && min_rank >= base_tail_rank) return true;
    if (!any_anchor_row) {
        if (draft_tail_rank > 0 && min_rank >= draft_tail_rank) return true;
        if (batch_tail_rank > 0 && min_rank >= batch_tail_rank) return true;
    }
    return false;
}

}  // namespace strata::core
