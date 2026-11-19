// src/core/batch_df2_adaptive_test.cpp - unit test for adaptive DFlash2 window cap (6A).
#include "strata/core/batch_df2_budget.hpp"

#include <cassert>
#include <cstdio>

int main() {
    using strata::core::compute_adaptive_df2_win_cap;
    using strata::core::allocate_dprob_spare_rows;
    using strata::core::batch_df2_head_max_per_slot;

    // Default OFF returns max_win_cap unchanged regardless of EMA acceptance.
    assert(compute_adaptive_df2_win_cap(2, 16, 8, 0.0f, false) == 16);
    assert(compute_adaptive_df2_win_cap(4, 12, 8, 0.9f, false) == 12);

    // Low acceptance clamps down to target_rows (8).
    assert(compute_adaptive_df2_win_cap(2, 16, 8, 0.20f, true) == 8);
    assert(compute_adaptive_df2_win_cap(2, 16, 8, 0.35f, true) == 8);

    // High acceptance allows full max_win_cap (16).
    assert(compute_adaptive_df2_win_cap(2, 16, 8, 0.70f, true) == 16);
    assert(compute_adaptive_df2_win_cap(2, 16, 8, 0.95f, true) == 16);

    // Mid acceptance interpolates strictly inside [target_rows, max_win_cap] and stays >= active_slots.
    const int mid = compute_adaptive_df2_win_cap(2, 16, 8, 0.525f, true);
    assert(mid == 12);
    assert(compute_adaptive_df2_win_cap(10, 16, 8, 0.10f, true) == 10);

    // Existing dprob budget contracts remain intact.
    const float probs0[] = {0.9f, 0.8f, 0.7f};
    const float probs1[] = {0.5f, 0.4f, 0.4f};
    const strata::core::Df2SlotBudgetInput inputs[2] = {{3, true, probs0}, {3, true, probs1}};
    int extra[2] = {};
    allocate_dprob_spare_rows(inputs, 2, 3, 0.3, extra);
    assert(extra[0] == 3 && extra[1] == 0);
    assert(batch_df2_head_max_per_slot(6, 2, true, 0) >= 4);

    std::printf("batch_df2_adaptive_test: ok\n");
    return 0;
}
