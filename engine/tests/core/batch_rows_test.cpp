// The batch window's row layout (include/strata/core/batch_rows.hpp): plain one-row slots, a slot's contiguous group of
// proposed rows (--batch-mtp), and the hand-off capacity of a layer split's stages.  Header-only, CPU.
#include "strata/core/batch_rows.hpp"
#include "strata/core/batch_df2_budget.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}
const char* run(const int* rows, const long long* p, int S, int n_slots, int max_t, int hbase, bool handoff) {
    int64_t pos[8];
    for (int t = 0; t < S; ++t) pos[t] = p[t];
    return strata::core::batch_rows_error(rows, S, pos, n_slots, max_t, hbase, handoff, 8);
}
}  // namespace

int main() {
    using strata::core::batch_rows_error;
    // plain batch: one row per slot, any positions
    {
        const int rows[] = {0, 1, 2};
        const long long pos[] = {10, 500, 3};
        check(run(rows, pos, 3, 3, 8, 0, false) == nullptr, "one row per slot");
        check(run(rows, pos, 3, 3, 8, 0, true) == nullptr, "one row per slot through the hand-off");
    }
    // --batch-mtp: a slot's two rows at consecutive positions; also on a layer split (S > slots)
    {
        const int rows[] = {1, 1, 0, 0};
        const long long pos[] = {40, 41, 7, 8};
        check(run(rows, pos, 4, 2, 8, 0, false) == nullptr, "two rows per slot, one stage");
        check(run(rows, pos, 4, 2, 8, 0, true) == nullptr, "two rows per slot across a layer split (S > slots)");
    }
    // eight rows = four slots of two: the whole hand-off
    {
        const int rows[] = {0, 0, 1, 1, 2, 2, 3, 3};
        const long long pos[] = {1, 2, 5, 6, 9, 10, 20, 21};
        check(run(rows, pos, 8, 4, 8, 0, true) == nullptr, "eight rows fill the hand-off");
        check(run(rows, pos, 8, 4, 8, 1, true) != nullptr, "a group past the hand-off's end");
        check(run(rows, pos, 8, 4, 8, 1, false) == nullptr, "no hand-off: the base does not bound the rows");
    }
    // refused layouts
    {
        const int rows[] = {0, 1, 0};
        const long long pos[] = {5, 9, 6};
        check(run(rows, pos, 3, 2, 8, 0, false) != nullptr, "a slot's rows must be contiguous");
    }
    {
        const int rows[] = {0, 0};
        const long long pos[] = {5, 7};
        check(run(rows, pos, 2, 1, 8, 0, false) != nullptr, "a slot's rows need consecutive positions");
    }
    {
        const int rows[] = {0, 2};
        const long long pos[] = {5, 7};
        check(run(rows, pos, 2, 2, 8, 0, false) != nullptr, "a slot past the slots held");
        const int neg[] = {-1};
        const long long p1[] = {1};
        check(run(neg, p1, 1, 2, 8, 0, false) != nullptr, "a negative slot");
    }
    {
        const int rows[] = {0, 0, 0};
        const long long pos[] = {1, 2, 3};
        check(run(rows, pos, 3, 1, 2, 0, false) != nullptr, "more rows than the window holds");
        check(run(rows, pos, 0, 1, 8, 0, false) != nullptr, "an empty window");
        check(run(rows, pos, 3, 1, 8, -1, false) != nullptr, "a negative hand-off base");
        check(run(rows, pos, 3, 1, 8, 0, false) == nullptr, "three rows of one slot are a group");
    }
    // Cross-slot dprob-ranked spare-row budget (C-F2)
    {
        using strata::core::Df2SlotBudgetInput;
        using strata::core::allocate_dprob_spare_rows;
        using strata::core::batch_df2_head_max_per_slot;

        const float p0[7] = {0.85f, 0.40f, 0.90f, 0.90f, 0.0f, 0.0f, 0.0f};
        const float p1[7] = {0.95f, 0.92f, 0.90f, 0.88f, 0.0f, 0.0f, 0.0f};
        const Df2SlotBudgetInput in_a[2] = {{4, true, p0}, {4, true, p1}};
        int out_a[2] = {};
        allocate_dprob_spare_rows(in_a, 2, 6, 0.50, out_a);
        check(out_a[0] == 1 && out_a[1] == 4, "dprob budget: confident slot claims spare rows released by low-confidence peer");

        const float q0[7] = {0.90f, 0.90f, 0.90f, 0.90f, 0.0f, 0.0f, 0.0f};
        const float q1[7] = {0.98f, 0.98f, 0.98f, 0.98f, 0.0f, 0.0f, 0.0f};
        const Df2SlotBudgetInput in_b[2] = {{4, true, q0}, {4, true, q1}};
        int out_b[2] = {};
        allocate_dprob_spare_rows(in_b, 2, 6, 0.50, out_b);
        check(out_b[0] == 2 && out_b[1] == 4, "dprob budget: greedy chain-prob ranking allocates 2+4 across asymmetric slots");

        const Df2SlotBudgetInput in_c[2] = {{4, true, q1}, {4, true, q1}};
        int out_c[2] = {};
        allocate_dprob_spare_rows(in_c, 2, 6, 0.50, out_c);
        check(out_c[0] == 3 && out_c[1] == 3, "dprob budget: equal chain-prob ties break symmetrically (3+3)");

        check(batch_df2_head_max_per_slot(6, 2, true, 0) == 4 &&
              batch_df2_head_max_per_slot(6, 2, false, 0) == 3 &&
              batch_df2_head_max_per_slot(14, 2, true, 0) == 7,
              "batch_df2_head_max_per_slot: returns 4 for 2-slot 8-row dprob budget and 7 for 16-row");
    }
    if (failures != 0) return EXIT_FAILURE;
    std::printf("batch_rows_test: ok\n");
    return EXIT_SUCCESS;
}
