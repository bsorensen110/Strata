// Readiness-tier and row-count MMQ expert sorting (src/prefill/mmq_resident_sort.hpp): header-only, no GPU.
#include "../../src/prefill/mmq_resident_sort.hpp"

#include <cstdio>
#include <vector>

namespace d = strata::prefill::detail;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main(int argc, char** argv) {
    if (argc == 2) {
        CHECK(d::mmq_staged_sort_mode() == std::atoi(argv[1]));
        return failed ? 1 : 0;
    }
    // 1. Eligibility on streamed vs staged walks
    {
        const int32_t all_res[4] = {0, 1, 2, 3};
        const int32_t part_res[4] = {0, -1, 2, -1};
        CHECK(d::mmq_resident_sort_eligible(true, true, true, true, all_res, 4, true, true));
        CHECK(!d::mmq_resident_sort_eligible(true, true, true, true, part_res, 4, true, true));
        CHECK(!d::mmq_resident_sort_eligible(true, true, true, true, all_res, 4, true, false));
        // Staged walk (!stream_all): resident-only sort still requires all-resident unless allow_staged_tiers=true
        CHECK(!d::mmq_resident_sort_eligible(true, true, true, true, part_res, 4, false, true, false));
        CHECK(d::mmq_resident_sort_eligible(true, true, true, true, part_res, 4, false, true, true));
        CHECK(d::mmq_resident_sort_eligible(true, true, true, false, nullptr, 4, false, true, true));
        CHECK(!d::mmq_resident_sort_eligible(false, true, true, true, all_res, 4, false, true, true));
        CHECK(!d::mmq_resident_sort_eligible(true, false, true, true, all_res, 4, false, true, true));
    }

    // Staged modes keep the partial group and non-local rows intact while rebuilding all routes.
    {
        CHECK(d::mmq_staged_sort_mode_from(nullptr) == 0);
        CHECK(d::mmq_staged_sort_mode_from("") == 0);
        CHECK(d::mmq_staged_sort_mode_from("0") == 0);
        CHECK(d::mmq_staged_sort_mode_from("1") == 1);
        CHECK(d::mmq_staged_sort_mode_from("2") == 2);
        CHECK(d::mmq_staged_sort_mode_from("3") == 3);
        const int32_t res[] = {0, 1, 2, 3, 4, -1, 6, -1, 8, -1, -1, -1};
        CHECK(d::mmq_staged_sort_eligible(true, true, true, res, 12, false));
        CHECK(!d::mmq_staged_sort_eligible(false, true, true, res, 12, false));
        CHECK(!d::mmq_staged_sort_eligible(true, false, true, res, 12, false));
        CHECK(!d::mmq_staged_sort_eligible(true, true, false, res, 12, false));
        CHECK(!d::mmq_staged_sort_eligible(true, true, true, nullptr, 12, false));
        CHECK(!d::mmq_staged_sort_eligible(true, true, true, res, 0, false));
        CHECK(!d::mmq_staged_sort_eligible(true, true, true, res, 12, true));

        const std::vector<int32_t> count = {8, 1, 7, 2, 6, 3, 5, 4, 9, 10, 11, 2};
        const std::vector<std::vector<int32_t>> expected = {
            {1, 3, 6, 4, 2, 0, 5, 7, 8, 9, 10},
            {1, 3, 5, 7, 6, 4, 2, 0, 8, 9, 10},
            {2, 0, 8, 7, 6, 4, 1, 3, 5, 9, 10}
        };
        std::vector<int32_t> ids;
        for (int32_t row = 0; row < 11; ++row)
            for (int32_t e = 0; e < 12; ++e)
                if (row < count[(size_t) e]) ids.push_back(e);
        for (int mode = 1; mode <= 3; ++mode) {
            std::vector<int32_t> order = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            std::vector<int32_t> off(13, 0), slot(ids.size(), -1), src(ids.size(), -1);
            int32_t cursor = 0;
            for (size_t e = 0; e < count.size(); ++e) {
                off[e] = cursor;
                cursor += count[e];
            }
            off.back() = cursor;
            const int32_t cpu_off = off[11];
            d::mmq_staged_sort_rows(ids.data(), (int64_t) ids.size(), 1, count, res, 3, true,
                                    off, order, slot.data(), src.data(), mode);
            CHECK(order == expected[(size_t) mode - 1]);
            cursor = 0;
            for (int32_t e : order) {
                CHECK(off[(size_t) e] == cursor);
                cursor += count[(size_t) e];
            }
            CHECK(off[11] == cpu_off && off.back() == (int32_t) ids.size());
            std::vector<int32_t> next(off.begin(), off.end() - 1);
            for (size_t i = 0; i < ids.size(); ++i) {
                const int32_t e = ids[i];
                CHECK(slot[i] == next[(size_t) e]++);
                CHECK(src[(size_t) slot[i]] == (int32_t) i);
            }
        }
        // Equal counts use expert ID as the tie-breaker; mode 3 only reverses non-resident groups.
        const std::vector<int32_t> ties(5, 1), tie_ids = {4, 3, 2, 1, 0};
        const int32_t all_res[] = {0, 1, 2, 3, 4};
        const int32_t no_res[] = {-1, -1, -1, -1, -1};
        for (const int32_t* residency : {all_res, no_res}) {
            std::vector<int32_t> order = {0, 1, 2, 3, 4}, off(6), slot(5), src(5);
            off.back() = 5;
            d::mmq_staged_sort_rows(tie_ids.data(), 5, 1, ties, residency, 2, true,
                                    off, order, slot.data(), src.data(), 3);
            const std::vector<int32_t> want = residency == all_res
                ? std::vector<int32_t>{0, 1, 2, 3, 4} : std::vector<int32_t>{2, 3, 0, 1, 4};
            CHECK(order == want);
        }
    }

    // 2. Readiness-tier + row-count sort with partial tail group kept and CPU-shared expert offsets preserved
    {
        // 7 routed experts: 6 GPU-local (group size 4 -> span = 4, tail = 2) and 1 CPU-shared (expert 6)
        // GPU-local experts in ID order: 0, 1, 2, 3 (in span 0..4), 4, 5 (in tail 4..6)
        // Counts: e0=5 (Tier 2), e1=3 (Tier 1), e2=4 (Tier 0), e3=1 (Tier 0), e4=2, e5=1, e6(CPU)=2
        const int32_t res[7] = {-1, -1, 10, 11, -1, -1, -1};
        const char pin[7] = {0, 1, 0, 0, 1, 0, 1};
        std::vector<int32_t> count = {5, 3, 4, 1, 2, 1, 2};
        // Build routes (K = 2, T = 9 -> 18 routes)
        std::vector<int32_t> ids = {
            0, 2,  1, 0,  2, 3,  0, 1,  4, 6,  0, 2,  1, 4,  0, 5,  2, 6
        };
        const int64_t routes = (int64_t) ids.size();
        const int32_t K = 2;
        std::vector<int32_t> order = {0, 1, 2, 3, 4, 5};
        std::vector<int32_t> off(8, 0);
        int32_t r = 0;
        for (int32_t e : order) { off[(size_t) e] = r; r += count[(size_t) e]; }
        const int32_t cpu_off = r;   // 16
        off[6] = r; r += count[6];   // 16..18
        off[7] = r;                  // 18

        std::vector<int32_t> slot((size_t) routes, -1), src((size_t) routes, -1);
        const size_t span = d::mmq_resident_sort_span(order.size(), 4, true);
        CHECK(span == 4);
        d::mmq_resident_sort_rows(ids.data(), routes, K, count, off, order, slot.data(), src.data(), span, res, pin);

        // Expected order[0..4): Tier 0 sorted by count (e3:1, e2:4), then Tier 1 (e1:3), then Tier 2 (e0:5)
        // Expected tail order[4..6): unchanged (e4, e5)
        CHECK(order.size() == 6);
        CHECK(order[0] == 3 && order[1] == 2 && order[2] == 1 && order[3] == 0);
        CHECK(order[4] == 4 && order[5] == 5);

        // Offsets for GPU-local experts follow `order`, and CPU-shared expert 6 keeps `cpu_off` (16)
        CHECK(off[3] == 0);
        CHECK(off[2] == 1);
        CHECK(off[1] == 5);
        CHECK(off[0] == 8);
        CHECK(off[4] == 13);
        CHECK(off[5] == 15);
        CHECK(off[6] == cpu_off);
        CHECK(off[7] == 18);

        // Every route i maps to a valid row inside its expert's [off[e], off[e] + count[e]) and src[slot[i]] == i / K
        std::vector<int32_t> seen((size_t) routes, 0);
        for (int64_t i = 0; i < routes; ++i) {
            const int32_t e = ids[(size_t) i];
            const int32_t p = slot[(size_t) i];
            CHECK(p >= off[(size_t) e] && p < off[(size_t) e] + count[(size_t) e]);
            CHECK(src[(size_t) p] == (int32_t) (i / K));
            ++seen[(size_t) p];
        }
        for (int32_t v : seen) CHECK(v == 1);
    }

    // 3. Layer-major `mmq_lm_sort_uniq` with and without Tier 2 ID-order preservation
    {
        const int32_t res[6] = {-1, 0, -1, -1, 1, -1};
        const char pin[6] = {0, 0, 1, 0, 0, 1};
        const std::vector<int32_t> count = {8, 5, 4, 2, 1, 9};
        // Tiers: e0=T2(8), e1=T0(5), e2=T1(4), e3=T2(2), e4=T0(1), e5=T1(9)
        std::vector<int32_t> uniq_pf = {0, 1, 2, 3, 4, 5}, off_pf(7, 0);
        d::mmq_lm_sort_uniq(count, off_pf, uniq_pf, res, pin, true);
        // Tier 0 sorted by count: 4(1), 1(5); Tier 1 sorted by count: 2(4), 5(9); Tier 2 in ID order: 0(8), 3(2)
        CHECK(uniq_pf[0] == 4 && uniq_pf[1] == 1);
        CHECK(uniq_pf[2] == 2 && uniq_pf[3] == 5);
        CHECK(uniq_pf[4] == 0 && uniq_pf[5] == 3);
        CHECK(off_pf[4] == 0 && off_pf[1] == 1 && off_pf[2] == 6 && off_pf[5] == 10 && off_pf[0] == 19 && off_pf[3] == 27 && off_pf[6] == 29);

        std::vector<int32_t> uniq_nopf = {0, 1, 2, 3, 4, 5}, off_nopf(7, 0);
        d::mmq_lm_sort_uniq(count, off_nopf, uniq_nopf, res, pin, false);
        // Tier 2 also sorted by count: 3(2), 0(8)
        CHECK(uniq_nopf[4] == 3 && uniq_nopf[5] == 0);
        CHECK(off_nopf[3] == 19 && off_nopf[0] == 21 && off_nopf[6] == 29);
    }

    if (failed == 0) std::printf("mmq_resident_sort_test: ok\n");
    return failed ? 1 : 0;
}
