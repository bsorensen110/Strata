// src/core/batch_split_policy_test.cpp - unit tests for slot-boundary G=2 split helper (STRATA_BATCH_SPLIT).
#include "strata/core/batch_split_policy.hpp"

#include <cstdio>
#include <vector>

using strata::core::batch_split_point;

int main() {
    int bad = 0;
    auto check_eq = [&](const char* name, int got, int want) {
        if (got != want) {
            std::fprintf(stderr, "FAIL %s: got %d expected %d\n", name, got, want);
            ++bad;
        }
    };

    // <= max_group never splits
    {
        const int s8[8] = {0, 0, 1, 1, 2, 2, 3, 3};
        check_eq("T=8", batch_split_point(s8, 8, 8), 0);
        check_eq("T=4", batch_split_point(s8, 4, 8), 0);
    }
    // Single slot spanning >8 rows has no slot boundary -> returns 0 (safe no-split fallback)
    {
        const int s10[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        check_eq("single_slot_10", batch_split_point(s10, 10, 8), 0);
    }
    // Slot wrapping across middle: boundaries at 4 and 10 -> boundary 4 leaves 6 rows in [4,10), boundary 10 fails (10 > 8)!
    {
        const int asym[12] = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2};
        check_eq("asym_split_at_4", batch_split_point(asym, 12, 8), 4);
    }
    // Equal 2-slot 8+8 = 16 -> splits cleanly at 8
    {
        const int s16[16] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
        check_eq("two_slots_8_8", batch_split_point(s16, 16, 8), 8);
    }
    // 4 slots x 3 rows = 12 -> boundaries at 3, 6, 9 -> chooses 6 (exact T/2)
    {
        const int s12[12] = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3};
        check_eq("four_slots_3each", batch_split_point(s12, 12, 8), 6);
    }
    // Tie break: 9 rows with 2 slots [5, 4] vs [4, 5] -> boundaries 5 and 4? Only boundary 4 in [4,5] case.
    {
        const int s9a[9] = {0, 0, 0, 0, 1, 1, 1, 1, 1};
        check_eq("nine_4_5", batch_split_point(s9a, 9, 8), 4);
        const int s10_tie[10] = {0, 0, 0, 0, 1, 1, 2, 2, 2, 2};   // boundaries at 4 (|8-10|=2) and 6 (|12-10|=2) -> smaller s = 4
        check_eq("ten_tie_smaller", batch_split_point(s10_tie, 10, 8), 4);
    }
    // Past 2 * max_group cannot keep both groups <= max_group
    {
        const int s17[17] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
        check_eq("T=17_over_2x", batch_split_point(s17, 17, 8), 0);
    }
    // Null pointer safety
    check_eq("null_slots", batch_split_point(nullptr, 10, 8), 0);

    std::printf("batch_split_policy_test: %s\n", bad ? "FAILED" : "ok");
    return bad ? 1 : 0;
}
