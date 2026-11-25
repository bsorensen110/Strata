// include/strata/core/tail_skip_policy.hpp - per-row anchor/draft tail-skip isolation policy.
#pragma once

#include <cstdint>

namespace strata::core {

/// Populates per-row draft flags (`row_is_draft`) and optional in-window row ranks (`row_rank_in_slot`)
/// for a solo linear or tree verify window of `n_rows` executed rows:
/// - When `has_drafts` is false (prompt/merged/first-token window): all rows `0..n_rows-1` are marked `0`
///   (committed/anchor) so draft-only tail-skip never drops any expert they route.
/// - When `has_drafts` is true (decode window, including DF2 trees and padded dummy leaves): row `0` is marked
///   `0` (anchor) and rows `1..n_rows-1` are marked `1` (draft-only), so `STRATA_DRAFT_TAIL_SKIP=R` drops
///   missed experts routed only by draft rows at rank `>= R`.
inline void fill_solo_row_is_draft(uint8_t* row_is_draft,
                                   uint8_t* row_rank_in_slot,
                                   int n_rows,
                                   bool has_drafts) noexcept {
    if (n_rows <= 0 || row_is_draft == nullptr) return;
    for (int i = 0; i < n_rows; ++i) {
        const uint8_t is_draft = (has_drafts && i > 0) ? 1u : 0u;
        row_is_draft[i] = is_draft;
        if (row_rank_in_slot != nullptr) {
            row_rank_in_slot[i] = is_draft ? static_cast<uint8_t>(i) : 0u;
        }
    }
}

/// RAII binder that sets `row_is_draft` / `row_rank_in_slot` dispatch pointers for one verify window
/// and clears both to nullptr on scope exit.
class ScopedDraftRowPointers {
 public:
    ScopedDraftRowPointers(const uint8_t*& dst_is_draft,
                           const uint8_t*& dst_rank_in_slot,
                           const uint8_t* is_draft,
                           const uint8_t* rank_in_slot) noexcept
        : dst_is_draft_(dst_is_draft), dst_rank_in_slot_(dst_rank_in_slot) {
        dst_is_draft_ = is_draft;
        dst_rank_in_slot_ = rank_in_slot;
    }
    ~ScopedDraftRowPointers() {
        dst_is_draft_ = nullptr;
        dst_rank_in_slot_ = nullptr;
    }
    ScopedDraftRowPointers(const ScopedDraftRowPointers&) = delete;
    ScopedDraftRowPointers& operator=(const ScopedDraftRowPointers&) = delete;

 private:
    const uint8_t*& dst_is_draft_;
    const uint8_t*& dst_rank_in_slot_;
};

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
