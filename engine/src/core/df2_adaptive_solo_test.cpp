// src/core/df2_adaptive_solo_test.cpp - host test for adaptive solo DFlash2 width selection.
#include "strata/spec/df2_adaptive_solo.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

void check(bool ok, const char* message) {
    if (!ok) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

}  // namespace

int main() {
    using strata::spec::Df2AdaptiveSolo;
    using strata::spec::df2_solo_prefix_width;
    constexpr std::array<int, 7> drafts{11, 12, 13, 14, 15, 16, 17};
    constexpr std::array<int, 4> lookup{21, 22, 23, 24};
    constexpr int base = df2_solo_prefix_width(12, (int) drafts.size());
    check(base == 8, "wide configured DF2 prefix must stop at the seventh draft");
    check(df2_solo_prefix_width(base, 3) == 4, "short handoff must stop at its produced drafts");
    check(df2_solo_prefix_width(5, (int) drafts.size()) == 5, "confidence-limited prefix must stay short");

    std::array<double, Df2AdaptiveSolo::kMaxWidth + 1> costs;
    costs.fill(1.0);
    Df2AdaptiveSolo policy;
    policy.reset(base);

    auto round = [&](int available, int kept, int reject_at, unsigned allowed = (1u << 10) | (1u << 12)) {
        const int chain = policy.choose_chain(available, kept, allowed, costs.data());
        check(chain <= available && policy.tier() <= base + available, "tier exceeds produced rows");
        check(((allowed | (1u << base)) >> policy.tier()) & 1u, "tier is not allowed");
        std::array<int, base + lookup.size()> window{};
        window[0] = 10;
        std::copy(drafts.begin(), drafts.end(), window.begin() + 1);
        std::copy_n(lookup.begin(), chain, window.begin() + base);
        // The verifier emits the next token from each input row, up to the first mismatch.
        auto out = window;
        std::copy_n(window.begin() + 1, base + chain - 1, out.begin());
        if (reject_at > 0 && reject_at < base + chain) out[reject_at - 1] = -1;
        int accepted = 0;
        while (accepted < base + chain - 1 && window[accepted + 1] == out[accepted]) ++accepted;
        policy.observe(accepted, chain);
        return chain;
    };
    auto rate_is = [&](int position, double expected, const char* message) {
        check(std::abs(policy.acceptance(position) - expected) < 1e-12, message);
    };

    check(round(0, 0, 0) == 0, "no chain must use only DF2");
    rate_is(7, 2.0 / 3, "last DF2 proposal was accepted");
    rate_is(8, 0.5, "no lookup position was tested");
    policy.reset(base);
    check(round(1, 1, 0) == 1, "keep a real one-row legacy chain");
    rate_is(7, 2.0 / 3, "last DF2 proposal belongs to DF2");
    rate_is(8, 2.0 / 3, "first lookup proposal belongs to lookup");
    rate_is(9, 0.5, "unoffered lookup position was not tested");

    policy.reset(base);
    check(round(4, 4, 2) == 4, "fixed chain can supply all four rows");
    rate_is(1, 2.0 / 3, "first DF2 proposal was accepted");
    rate_is(2, 1.0 / 3, "first rejection counts as tested");
    rate_is(3, 0.5, "DF2 rows after rejection were not tested");
    rate_is(8, 0.5, "early DF2 rejection never reaches lookup");
    policy.reset(base);
    round(4, 4, 9);
    rate_is(8, 2.0 / 3, "first lookup proposal was accepted");
    rate_is(9, 1.0 / 3, "second lookup proposal was rejected");
    rate_is(10, 0.5, "lookup rows after rejection were not tested");
    policy.reset(base);
    round(4, 4, 0);
    rate_is(11, 2.0 / 3, "last lookup proposal was accepted");
    rate_is(12, 0.5, "all-accepted round does not invent a rejection");

    policy.reset(base);
    for (int i = 0; i < 8; ++i) {
        round(4, 4, 0);
        check(policy.tier() == base, "tier changed before eight observations");
    }
    for (int i = 0; i < 120 && policy.tier() == base; ++i) round(4, 4, 0);
    check(policy.tier() == 10, "full reachable chain should train the ten-row tier");
    for (int i = 0; i < 7; ++i) {
        check(round(4, 0, 1) == 2, "adaptive tier should append two real lookup rows");
        check(policy.tier() == 10, "early rejection broke tier cooldown");
    }
    for (int i = 0; i < 120 && policy.tier() > base; ++i) round(4, 0, 1);
    check(policy.tier() == base, "repeated early rejection should return to base");

    policy.reset(base);
    for (int i = 0; i < 120; ++i) round(4, 4, 0, 1u << 12);
    check(policy.tier() == 12, "tier must skip missing sizes");
    check(round(1, 0, 0) == 0 && policy.tier() == base, "short match cannot reach another allowed tier");
    for (int i = 0; i < 120; ++i) round(4, 4, 0);
    check(policy.tier() > base, "full match should restore a wider tier");
    check(round(0, 0, 0) == 0 && policy.tier() == base, "no match must drop a wide tier immediately");

    std::printf("df2_adaptive_solo_test: ok\n");
    return 0;
}
