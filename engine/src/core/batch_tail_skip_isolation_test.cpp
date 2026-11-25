// src/core/batch_tail_skip_isolation_test.cpp - unit test for per-row anchor/draft tail-skip isolation (6B).
#include "strata/core/tail_skip_policy.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>

int main() {
    using strata::core::should_tail_skip_expert;

    // All switches off -> never skip.
    assert(!should_tail_skip_expert(9, false, 0, 0, 0));
    assert(!should_tail_skip_expert(9, true, 0, 0, 0));

    // Legacy STRATA_ROUTE_TAIL_SKIP applies to both anchor and draft rows at min_rank >= base_tail_rank.
    assert(should_tail_skip_expert(7, true, 7, 0, 0));
    assert(!should_tail_skip_expert(6, true, 7, 0, 0));

    // Draft-only / batch tail-skip NEVER skips an expert if any anchor row routes it.
    assert(!should_tail_skip_expert(9, true, 0, 7, 0));
    assert(!should_tail_skip_expert(9, true, 0, 0, 6));
    assert(!should_tail_skip_expert(9, true, 0, 7, 6));

    // Draft-only missed experts beyond threshold ARE skipped when no anchor row touches them.
    assert(should_tail_skip_expert(7, false, 0, 7, 0));
    assert(!should_tail_skip_expert(6, false, 0, 7, 0));
    assert(should_tail_skip_expert(6, false, 0, 0, 6));
    assert(!should_tail_skip_expert(5, false, 0, 0, 6));

    // Solo / tree row marking: row 0 is anchor (0), rows 1..n_rows-1 (including tree dummies) are drafts (1).
    auto check_or_fail = [](bool cond, const char* msg) {
        if (!cond) {
            std::fprintf(stderr, "FAIL: %s\n", msg);
            std::abort();
        }
    };
    uint8_t is_draft[8] = {};
    uint8_t rank_in_slot[8] = {};
    strata::core::fill_solo_row_is_draft(is_draft, rank_in_slot, 8, true);
    check_or_fail(is_draft[0] == 0 && rank_in_slot[0] == 0, "solo anchor row 0 marked non-draft");
    for (int i = 1; i < 8; ++i) {
        check_or_fail(is_draft[i] == 1 && rank_in_slot[i] == static_cast<uint8_t>(i),
                      "solo/tree rows >= 1 marked draft with in-window rank");
    }

    // Any expert routed by row 0 (anchor) is immune to STRATA_DRAFT_TAIL_SKIP even at rank 9;
    // experts routed only by draft rows (1..7) at rank >= 6 are dropped.
    const bool anchor_touched_from_row0 = (is_draft[0] == 0);
    check_or_fail(!should_tail_skip_expert(9, anchor_touched_from_row0, 0, 0, 6),
                  "anchor row 0 immune to draft tail skip");
    const bool draft_only_tree_tail = (is_draft[5] != 0) && (is_draft[7] != 0);
    check_or_fail(draft_only_tree_tail, "tree branch & dummy rows marked draft");
    check_or_fail(should_tail_skip_expert(6, false, 0, 0, 6),
                  "draft-only tree tail at rank >= R is tail-skipped");
    check_or_fail(!should_tail_skip_expert(5, false, 0, 0, 6),
                  "draft-only tree tail below rank R is preserved");

    // Prompt / merged / first-token windows mark every row as committed (0) so no row can be skipped.
    strata::core::fill_solo_row_is_draft(is_draft, rank_in_slot, 5, false);
    for (int i = 0; i < 5; ++i) {
        check_or_fail(is_draft[i] == 0 && rank_in_slot[i] == 0,
                      "merged/prompt rows marked committed");
        check_or_fail(!should_tail_skip_expert(9, is_draft[i] == 0, 0, 0, 6),
                      "merged/prompt rows never tail-skipped");
    }

    // ScopedDraftRowPointers binds during the verify scope and always resets to nullptr on exit.
    const uint8_t* ptr_draft = nullptr;
    const uint8_t* ptr_rank = nullptr;
    {
        strata::core::ScopedDraftRowPointers guard(ptr_draft, ptr_rank, is_draft, rank_in_slot);
        check_or_fail(ptr_draft == is_draft && ptr_rank == rank_in_slot,
                      "ScopedDraftRowPointers binds pointers in scope");
    }
    check_or_fail(ptr_draft == nullptr && ptr_rank == nullptr,
                  "ScopedDraftRowPointers resets pointers on scope exit");

    std::printf("batch_tail_skip_isolation_test: ok\n");
    return 0;
}
