// The default CPU share is not armed when every expert is resident (#1595): no GPU, no model.
#include <cstdio>
#include <string>

#include "strata/prefill/share_rules.hpp"

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
    if (failed == 0) std::printf("share_rules_test: ok\n");
    return failed ? 1 : 0;
}
