#include "strata/spec/df2_sample_walk.hpp"

#include <cmath>
#include <cstdio>
#include <random>

namespace {
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}
}

int main() {
    constexpr int trials = 500000;
    int32_t candidates[strata::spec::kDf2SelectorTopK];
    float scores[strata::spec::kDf2SelectorTopK * strata::spec::kDf2SelectorTopK];
    int32_t drafts[1] = {};
    float selected_q[1] = {};
    int32_t q_tokens[strata::spec::kDf2SelectorTopK] = {};
    float q_probs[strata::spec::kDf2SelectorTopK] = {};
    for (int j = 0; j < strata::spec::kDf2SelectorTopK; ++j) {
        candidates[j] = j < 2 ? j : -1;
        for (int prev = 0; prev < strata::spec::kDf2SelectorTopK; ++prev)
            scores[prev * strata::spec::kDf2SelectorTopK + j] = -INFINITY;
    }
    constexpr float q0 = 0.7f, q1 = 0.3f;
    scores[0] = std::log(q0);
    scores[1] = std::log(q1);
    const int32_t target_ids[] = {0, 1};
    const double target_probs[] = {0.2, 0.8};
    const strata::core::SpecP target{target_ids, target_probs, 2};
    std::mt19937_64 rng(20261119);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    long long q_count[2] = {}, p_count[2] = {};

    for (int t = 0; t < trials; ++t) {
        if (!strata::spec::df2_sample_selector_walk(1, candidates, scores, 0xDFA5u, (uint64_t) t,
                                                    drafts, selected_q, q_tokens, q_probs)) {
            check(false, "selector walk accepts a valid finite two-candidate row");
            break;
        }
        const int draft = drafts[0];
        if (draft < 0 || draft > 1) { ++failures; continue; }
        ++q_count[draft];
        strata::core::SpecQ q{q_tokens, q_probs, 2};
        const auto result = strata::core::spec_verify_row(target, draft, q, uniform(rng), uniform(rng));
        ++p_count[result.token];
    }

    auto chi2 = [](long long a, long long b, double p, int n) {
        const double ea = p * n, eb = (1.0 - p) * n;
        return (a - ea) * (a - ea) / ea + (b - eb) * (b - eb) / eb;
    };
    check(chi2(q_count[0], q_count[1], q0, trials) < 10.83, "sampled selector proposals match q by chi-square");
    check(chi2(p_count[0], p_count[1], 0.2, trials) < 10.83,
          "rejection sampling restores the target marginal at T > 0");

    int32_t packed[strata::spec::kDf2SampleQStride] = {};
    strata::spec::pack_df2_spec_q(1, q_tokens, q_probs, packed);
    float unpacked = 0.0f;
    std::memcpy(&unpacked, packed + strata::spec::kDf2SampleQEntries, sizeof(float));
    check(packed[0] == 0 && packed[1] == 1 && packed[2] == -1 && std::fabs(unpacked - q0) < 1e-6f,
          "selector conditionals pack in the verifier's q-row format");

    const int32_t greedy_sentinel = 77;
    int32_t greedy_output = greedy_sentinel;
    if (strata::spec::df2_sample_walk_for_request(false, false, 1.0f) ||
        strata::spec::df2_sample_walk_for_request(true, true, 0.0f) ||
        strata::spec::df2_sample_walk_for_request(true, false, 0.0f))
        greedy_output = drafts[0];
    check(greedy_output == greedy_sentinel, "disabled and greedy sampling leave the existing draft path untouched");

    std::printf("df2_sample_walk_test: %s (q %.4f, target %.4f)\n", failures == 0 ? "OK" : "FAILED",
                (double) q_count[0] / trials, (double) p_count[0] / trials);
    return failures == 0 ? 0 : 1;
}
