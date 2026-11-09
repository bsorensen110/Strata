// The default CPU share is not armed when every expert is resident (#1595): no GPU, no model.
#include <cstdio>
#include <string>

#include "strata/prefill/share_rules.hpp"
#include "../../src/prefill/mmq_resident_sort.hpp"

using strata::prefill::share_rules::batch_prefill_policy;
using strata::prefill::share_rules::default_share_has_work;
using strata::prefill::share_rules::batch_cpu_share_blocked;
using strata::prefill::share_rules::batch_co_prefill_active;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    const auto serial_policy = batch_prefill_policy(true, false, true, true);
    CHECK(!serial_policy.block_cpu_share);           // serial layer-yield scheduling may take CPU work
    CHECK(serial_policy.co_prefill_ring);             // co-prefill ring remains active independently
    const auto concurrent_policy = batch_prefill_policy(true, true, true, true);
    CHECK(concurrent_policy.block_cpu_share);         // concurrent reader owns the pool interval
    CHECK(concurrent_policy.co_prefill_ring);         // still a co-prefill workload
    const auto disabled_policy = batch_prefill_policy(true, true, false, true);
    CHECK(!disabled_policy.block_cpu_share);          // CPU sharing opt-in is off
    CHECK(disabled_policy.co_prefill_ring);            // ring policy is independent of share opt-in
    const auto idle_policy = batch_prefill_policy(true, false, true, false);
    CHECK(!idle_policy.block_cpu_share);
    CHECK(!idle_policy.co_prefill_ring);               // no active decode slots -> full solo ring

    const long long pairs = 48 * 512;
    CHECK(default_share_has_work(pairs - 1, pairs));       // one expert streams: the share has something to take
    CHECK(default_share_has_work(1500, pairs));            // a 12 GB card: most of them stream
    CHECK(!default_share_has_work(pairs, pairs));          // every expert resident: nothing to take
    CHECK(!default_share_has_work(pairs + 100, pairs));    // a cache with spare slots
    CHECK(default_share_has_work(0, pairs));               // empty cache
    CHECK(default_share_has_work(100000, 0));              // total unknown: keep today's behaviour
    static_assert(!default_share_has_work(10, 10), "constexpr");

    // BW-MEASURE E25: the default limit is 3,072 tokens, 1,024 for a class that keeps the old one
    CHECK(strata::prefill::share_rules::default_share_limit() == 3072);
    CHECK(strata::prefill::share_rules::default_share_limit(true) == strata::prefill::share_rules::kShareLimitNew);
    CHECK(strata::prefill::share_rules::default_share_limit(false) == 1024);
    CHECK(strata::prefill::share_rules::gfx12_share_limit(true) == 1536 && strata::prefill::share_rules::gfx12_share_limit(false) == 3072);   // AMD-Q
    CHECK(strata::prefill::share_rules::gfx12_share_limit(true, false) == 1536);
    CHECK(strata::prefill::share_rules::gfx12_share_limit(true, true) == 2560);
    CHECK(!strata::prefill::share_rules::batch_cpu_share_enabled_from(nullptr, true, false));
    CHECK(strata::prefill::share_rules::batch_cpu_share_enabled_from(nullptr, true, true));
    CHECK(!strata::prefill::share_rules::batch_cpu_share_enabled_from(nullptr, false, true));
    CHECK(!strata::prefill::share_rules::batch_cpu_share_enabled_from("0", true, true));
    CHECK(strata::prefill::share_rules::batch_cpu_share_enabled_from("1", false, false));
    CHECK(!strata::prefill::share_rules::batch_lend_pinned_from(nullptr, true, false));
    CHECK(strata::prefill::share_rules::batch_lend_pinned_from(nullptr, true, true));
    CHECK(!strata::prefill::share_rules::batch_lend_pinned_from(nullptr, false, true));
    CHECK(!strata::prefill::share_rules::batch_lend_pinned_from("0", true, true));
    CHECK(strata::prefill::share_rules::batch_lend_pinned_from("1", false, false));
    static_assert(strata::prefill::share_rules::kShareLimitOld == 1024, "the old limit (the way back is STRATA_PREFILL_CPU_SHARE_MAX=1024)");

    // H11: memory verdicts
    using namespace strata::prefill::share_rules;
    const uint64_t G = 1ull << 30;
    ShareMemory m;
    m.buffer_bytes = G / 3;
    CHECK(default_share_memory_verdict(m).arm);                                   // nothing known: today's behaviour
    m.avail_ram = 20 * G; m.avail_commit = 30 * G;
    CHECK(default_share_memory_verdict(m).arm);                                   // plenty of RAM and commit
    m.avail_ram = 3 * G;
    CHECK(!default_share_memory_verdict(m).arm);                                  // RAM below buffers + margin
    CHECK(default_share_memory_verdict(m).why.find("RAM is free") != std::string::npos);
    m.avail_ram = kShareRamMargin + m.buffer_bytes;
    CHECK(default_share_memory_verdict(m).arm);                                   // exactly enough
    m.avail_ram = kShareRamMargin + m.buffer_bytes - 1;
    CHECK(!default_share_memory_verdict(m).arm);                                  // one byte short
    m.avail_ram = 20 * G; m.avail_commit = 3 * G;
    {
        const ShareVerdict v = default_share_memory_verdict(m);
        CHECK(!v.arm);                                                            // Windows commit nearly used up
        CHECK(v.why.find("commit") != std::string::npos);
    }
    m.avail_commit = kUnknown;
    m.inplace_bytes = 18 * G;                                                     // file pages the RAM cannot keep
    {
        const ShareVerdict v = default_share_memory_verdict(m);
        CHECK(!v.arm);
        CHECK(v.why.find("file pages") != std::string::npos);
    }
    m.inplace_bytes = 10 * G;                                                     // they fit with the margin
    CHECK(default_share_memory_verdict(m).arm);
    m.avail_ram = kUnknown; m.inplace_bytes = 500 * G;
    CHECK(default_share_memory_verdict(m).arm);                                   // RAM unknown: never blocks

    // mmq_staged_sort: resident-first full groups + count-sorted staged/remainder groups, preserving tail & CPU offsets
    {
        namespace d = strata::prefill::detail;
        CHECK(d::mmq_staged_sort_from(nullptr, true));
        CHECK(!d::mmq_staged_sort_from(nullptr, false));
        CHECK(!d::mmq_staged_sort_from("0", true));
        CHECK(d::mmq_staged_sort_from("1", false));

        const std::vector<int32_t> res = {10, -1, 20, 30, -1, -1};
        CHECK(d::mmq_staged_sort_eligible(true, true, true, true, res.data(), 6, false));
        CHECK(!d::mmq_staged_sort_eligible(true, true, true, true, res.data(), 6, true));

        const std::vector<int32_t> count = {8, 3, 2, 5, 4, 2};
        std::vector<int32_t> order = {0, 1, 2, 3, 4};   // expert 5 is on_cpu (not in order)
        std::vector<int32_t> off = {0, 8, 11, 13, 18, 22, 24};
        std::vector<int32_t> ids;
        for (int32_t e = 0; e < 6; ++e)
            for (int32_t r = 0; r < count[(size_t) e]; ++r) ids.push_back(e);
        const int64_t routes = (int64_t) ids.size();
        std::vector<int32_t> slot((size_t) routes, -1), src((size_t) routes, -1);

        d::mmq_staged_sort_rows(ids.data(), routes, 1, count, res.data(), 2, true,
                                off, order, slot.data(), src.data());
        const std::vector<int32_t> want_order = {2, 3, 1, 0, 4};
        CHECK(order == want_order);
        CHECK(off[2] == 0 && off[3] == 2 && off[1] == 7 && off[0] == 10 && off[4] == 18);
        CHECK(off[5] == 22 && off[6] == 24);   // CPU expert 5 and total end preserved
        for (int64_t i = 0; i < routes; ++i) {
            const int32_t e = ids[(size_t) i];
            const int32_t p = slot[(size_t) i];
            CHECK(p >= off[(size_t) e] && p < off[(size_t) e] + count[(size_t) e]);
            CHECK(src[(size_t) p] == (int32_t) i);
        }

        // Mode 2 (sort_mode = 2): pure count-sort across full groups [0, span), preserving tail & CPU offsets
        std::vector<int32_t> order2 = {0, 1, 2, 3, 4};
        std::vector<int32_t> off2 = {0, 8, 11, 13, 18, 22, 24};
        d::mmq_staged_sort_rows(ids.data(), routes, 1, count, res.data(), 2, true,
                                off2, order2, slot.data(), src.data(), /*sort_mode=*/2);
        const std::vector<int32_t> want_order2 = {2, 1, 3, 0, 4};
        CHECK(order2 == want_order2);
        CHECK(off2[2] == 0 && off2[1] == 2 && off2[3] == 5 && off2[0] == 10 && off2[4] == 18);
        CHECK(off2[5] == 22 && off2[6] == 24);

        // Mode 3 (sort_mode = 3): form count-matched groups, then place pure-resident groups first and staged groups
        // in descending count order so heavy compute overlaps with PCIe copies and light groups finish the tail.
        const std::vector<int32_t> res3 = {10, -1, 20, 30, -1, -1, -1};
        const std::vector<int32_t> count3 = {8, 3, 2, 5, 4, 6, 2};
        std::vector<int32_t> order3 = {0, 1, 2, 3, 4, 5};
        std::vector<int32_t> off3 = {0, 8, 11, 13, 18, 22, 28, 30};
        std::vector<int32_t> ids3;
        for (int32_t e = 0; e < 7; ++e)
            for (int32_t r = 0; r < count3[(size_t) e]; ++r) ids3.push_back(e);
        const int64_t routes3 = (int64_t) ids3.size();
        std::vector<int32_t> slot3((size_t) routes3, -1), src3((size_t) routes3, -1);
        d::mmq_staged_sort_rows(ids3.data(), routes3, 1, count3, res3.data(), 2, true,
                                off3, order3, slot3.data(), src3.data(), /*sort_mode=*/3);
        // Count-sorted groups of 2: G0={2(cnt=2,res),1(cnt=3,stg)}, G1={4(cnt=4,stg),3(cnt=5,res)}, G2={5(cnt=6,stg),0(cnt=8,res)}
        // With res3[5]=40 (if 5 were resident, G2 would be pure-resident): here G0,G1,G2 all have 1 staged expert, so descending -> G2, G1, G0!
        const std::vector<int32_t> want_order3 = {5, 0, 4, 3, 2, 1};
        CHECK(order3 == want_order3);
        CHECK(off3[6] == 28 && off3[7] == 30);

        // Multi-slice CPU share helpers & T-bucket isolation
        CHECK(prefill_cpu_share_max_t_from(nullptr, 8, false) == 8);
        CHECK(prefill_cpu_share_max_t_from(nullptr, 8, true) == 32);
        CHECK(prefill_cpu_share_max_t_from("16", 8, true) == 16);
        CHECK(prefill_cpu_share_max_t_from("8", 8, true) == 8);

        CHECK(cpu_share_bucket(19) == 0);
        CHECK(cpu_share_bucket(51) == 0);
        CHECK(cpu_share_bucket(309) == 1);
        CHECK(cpu_share_bucket(609) == 2);
        CHECK(cpu_share_bucket(2048) == 3);

        // Unmeasured (cpu_c_ms = 0, cpu_g_ms = 0): caps total job slices at llround(share * nstream)
        CHECK(cpu_share_can_take_next(/*cur_jobs=*/49, /*next_jobs=*/1, /*nstream=*/100, /*cur_cpu_experts=*/49, 0.5, 0.0, 0.0));
        CHECK(!cpu_share_can_take_next(/*cur_jobs=*/50, /*next_jobs=*/1, /*nstream=*/100, /*cur_cpu_experts=*/50, 0.5, 0.0, 0.0));

        // Measured: 1-slice experts match llround((g / (c + g)) * nstream), and multi-slice experts
        // balance CPU slice time against 0.88 * remaining GPU stream time (leaving 12% margin for H2D row copy).
        // With c_ms = 0.04, g_ms = 0.08, nstream = 300:
        CHECK(cpu_share_can_take_next(/*cur_jobs=*/60, /*next_jobs=*/2, /*nstream=*/300, /*cur_cpu_experts=*/60, 0.67, 0.04, 0.08));
        CHECK(cpu_share_can_take_next(/*cur_jobs=*/210, /*next_jobs=*/2, /*nstream=*/300, /*cur_cpu_experts=*/135, 0.67, 0.04, 0.08));
        // At cur_jobs=254, cur_cpu_experts=157: lhs = 255 * 0.04 = 10.20 ms > 0.88 * 142.5 * 0.08 = 10.032 ms -> rejected!
        CHECK(!cpu_share_can_take_next(/*cur_jobs=*/254, /*next_jobs=*/2, /*nstream=*/300, /*cur_cpu_experts=*/157, 0.67, 0.04, 0.08));
    }
    if (failed == 0) std::printf("share_rules_test: ok\n");
    return failed ? 1 : 0;
}
