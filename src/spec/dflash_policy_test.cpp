#include "strata/spec/dflash_policy.hpp"
#include <cstdio>

using strata::spec::DFlashPolicy;
static int fails;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); ++fails; } } while (0)

int main() {
    using strata::spec::dflash_attention_bucket;
    CHECK(dflash_attention_bucket(1, 8192) == 64);
    CHECK(dflash_attention_bucket(64, 8192) == 64);
    CHECK(dflash_attention_bucket(65, 8192) == 256);
    CHECK(dflash_attention_bucket(257, 8192) == 1024);
    CHECK(dflash_attention_bucket(1025, 8192) == 4096);
    CHECK(dflash_attention_bucket(4097, 8192) == 8192);
    CHECK(dflash_attention_bucket(90, 98) == 98);
    CHECK(dflash_attention_bucket(2052, 2052) == 2052);
    CHECK(dflash_attention_bucket(INT64_MAX, INT64_MAX) == INT64_MAX);
    CHECK(dflash_attention_bucket(0, 8192) == 0);
    CHECK(dflash_attention_bucket(8193, 8192) == 0);
    using strata::spec::dflash_proposal_fits;
    CHECK(dflash_proposal_fits(90, 7, false, 100, 97));
    CHECK(!dflash_proposal_fits(90, 7, true, 100, 97));
    CHECK(dflash_proposal_fits(90, 7, true, 100, 98));
    CHECK(!dflash_proposal_fits(93, 7, false, 100, 100));
    CHECK(!dflash_proposal_fits(-1, 3, true, 100, 100));
    CHECK(!dflash_proposal_fits(INT64_MAX, 7, true, INT64_MAX, INT64_MAX));
    CHECK(!dflash_proposal_fits(90, 0, true, 100, 100));
    for (int cap = 0; cap <= 7; ++cap) {
        DFlashPolicy p(cap);
        int counts[8] = {};
        for (int i = 0; i < 512; ++i) {
            const int k = p.choose();
            CHECK(k >= 0 && k <= cap);
            if (k < 0 || k > cap) break;
            ++counts[k];
            // Long blocks accept a little more, but K=2 gives the best throughput.
            p.observe(k, k == 0 ? 1 : k == 1 ? 2 : 3, k <= 2 ? 10 : 50);
        }
        const int expected = cap < 2 ? cap : 2;
        CHECK(counts[expected] > 400);
        for (int k = 0; k <= cap; ++k) CHECK(counts[k] >= 2);
    }
    {
        DFlashPolicy p(7);
        int plain = 0, resumed = 0;
        for (int i = 0; i < 512; ++i) {
            const int k = p.choose();
            plain += k == 0;
            p.observe(k, 1, k == 0 ? 10 : 40);   // drafts all rejected
        }
        CHECK(plain > 400);
        for (int i = 0; i < 2048; ++i) {
            const int k = p.choose();
            if (i >= 1536) resumed += k == 2;
            p.observe(k, k == 2 ? 3 : 1, k == 0 || k == 2 ? 10 : 40);
        }
        CHECK(resumed > 400);   // periodic probes discover useful drafting again
    }
    {
        DFlashPolicy p(7);
        int wide = 0;
        for (int i = 0; i < 2048; ++i) {
            int k = p.choose();
            if (i >= 1536) wide += k == 5;
            p.observe(k, k == 5 ? 6 : 1, 10);
        }
        CHECK(wide > 400);   // deferred probes must still discover better wide forwards
        p.reset();
        int plain = 0;
        for (int i = 0; i < 512; ++i) {
            int k = p.choose();
            plain += k == 0;
            p.observe(k, 1, k == 0 ? 10 : 40);
        }
        CHECK(plain > 400);   // a new prompt must not inherit stale acceptance/costs
    }
    {
        using strata::spec::dflash_truncate_prefix;
        const float probs[7] = {0.82f, 0.76f, 0.68f, 0.54f, 0.49f, 0.90f, 0.90f};
        CHECK(dflash_truncate_prefix(probs, 7, 0.0f) == 7);
        CHECK(dflash_truncate_prefix(probs, 7, 0.50f) == 4);
        // Per-depth slope: depth 0 threshold 0.45, depth 1 0.55, depth 2 0.65, depth 3 0.75 (0.54 < 0.75 stops at 3).
        CHECK(dflash_truncate_prefix(probs, 7, 0.45f, 0.10f) == 3);
        // Cumulative probability floor: 0.82 * 0.76 = 0.6232 >= 0.50, * 0.68 = 0.4237 < 0.50 -> stops at 2.
        CHECK(dflash_truncate_prefix(probs, 7, 0.40f, 0.0f, 0.50f) == 2);
    }
    {
        using strata::spec::DFlashTelemetry;
        DFlashTelemetry tel;
        tel.record_proposal(5, 3);   // 2 rows truncated by confidence gate
        tel.record_window(3, 2);     // 3 offered, 2 accepted (depths 0,1 ok; depth 2 tested & rejected)
        tel.record_proposal(3, 0);   // all 3 rows truncated -> target-only round
        tel.record_window(0, 0);
        tel.record_proposal(7, 7);
        tel.record_window(7, 7);     // all 7 depths tested and accepted
        CHECK(tel.proposed_rows == 15);
        CHECK(tel.gate_truncated_rows == 5);
        CHECK(tel.target_only_rounds == 1);
        CHECK(tel.windows_by_k[3] == 1 && tel.accepted_by_k[3] == 2);
        CHECK(tel.windows_by_k[7] == 1 && tel.accepted_by_k[7] == 7);
        CHECK(tel.depth_tested[0] == 2 && tel.depth_ok[0] == 2);
        CHECK(tel.depth_tested[1] == 2 && tel.depth_ok[1] == 2);
        CHECK(tel.depth_tested[2] == 2 && tel.depth_ok[2] == 1);
        CHECK(tel.depth_tested[6] == 1 && tel.depth_ok[6] == 1);
    }
    {
        using strata::spec::DFlashConfidencePolicy;
        DFlashConfidencePolicy cp(7, 3);
        CHECK(cp.choose() == 3);
        const float high7[7] = {0.92f, 0.90f, 0.88f, 0.87f, 0.86f, 0.85f, 0.84f};
        // Two consecutive confident full prefixes grow K step by step toward max_k=7.
        for (int i = 0; i < 10; ++i) {
            const int k = cp.choose();
            cp.observe(k, k, high7);
        }
        CHECK(cp.choose() == 7);
        // A single truncated round does not immediately collapse K (hysteresis).
        const float dip7[7] = {0.90f, 0.88f, 0.40f, 0.30f, 0.20f, 0.10f, 0.10f};
        cp.observe(7, 2, dip7);
        CHECK(cp.choose() == 7);
        // A second consecutive truncation shrinks K to prefix + 1 = 3.
        cp.observe(7, 2, dip7);
        CHECK(cp.choose() == 3);
        cp.reset();
        CHECK(cp.choose() == 3);
    }
    std::printf("dflash_policy_test: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
