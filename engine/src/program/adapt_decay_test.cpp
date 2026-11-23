// --adapt-decay validation (include/strata/program/adapt_decay.hpp), CPU only, no model (#1284).
#include "strata/program/adapt_decay.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <limits>

using strata::program::adapt_decay_checked;
using strata::program::kAdaptDecayDefault;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
}  // namespace

int main() {
    // inside the range: the value itself, no line (bit for bit what atof gave before, cast to float)
    for (double v : {0.0, 0.25, 0.5, 0.7, 0.9, 0.97, 0.999, 1.0}) {
        check(adapt_decay_checked(v, nullptr) == (float) v, "a value in 0..1 is kept");
    }
    // every default and tune_knobs value of the engine is unchanged
    check(adapt_decay_checked(0.7, nullptr) == kAdaptDecayDefault, "the default stays 0.7");
    // outside: the default
    for (double v : {1.0000001, 1.5, 2.0, 1e30, -0.1, -1.0, -1e30}) {
        check(adapt_decay_checked(v, nullptr) == kAdaptDecayDefault, "an out-of-range value falls back to the default");
    }
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    check(adapt_decay_checked(inf, nullptr) == kAdaptDecayDefault, "inf falls back");
    check(adapt_decay_checked(-inf, nullptr) == kAdaptDecayDefault, "-inf falls back");
    check(adapt_decay_checked(nan, nullptr) == kAdaptDecayDefault, "NaN falls back");
    // the result is always usable: finite and in 0..1
    for (double v : {-5.0, 0.3, 7.0, nan, inf}) {
        const float r = adapt_decay_checked(v, nullptr);
        check(std::isfinite(r) && r >= 0.f && r <= 1.f, "the result is finite and in 0..1");
    }
    // a counts vector driven by an accepted value stays finite over many adaptations (the bug: >1 overflowed to inf)
    float usage = 1.f;
    const float d = adapt_decay_checked(1.5, nullptr);
    for (int i = 0; i < 100000; ++i) usage = usage * d + 1.f;
    check(std::isfinite(usage), "counts stay finite with the value the engine ends up using");
    using namespace strata::program;
    check(!adapt_coverage_enabled(nullptr) && !adapt_coverage_enabled("") && !adapt_coverage_enabled("0"),
          "coverage defaults are opt-in");
    check(adapt_coverage_enabled("1"), "coverage defaults accept opt-in");
    check(adapt_exchange_limit(false) == 96 && adapt_exchange_limit(true) == 192, "exchange limit follows opt-in");
    auto coverage = [](bool enabled, int64_t live, int64_t total, bool given, int expected_every, int expected_swaps, float expected_decay) {
        int every = 8, swaps = 96;
        float decay = 0.7f;
        adapt_coverage_defaults(enabled, live, total, every, swaps, decay, given, given, given);
        check(every == expected_every && swaps == expected_swaps && decay == expected_decay, "coverage policy");
    };
    coverage(false, 50, 100, false, 8, 96, 0.7f);
    coverage(true, 14, 100, false, 8, 96, 0.7f);
    coverage(true, 15, 100, false, 2, 192, 0.92f);
    coverage(true, 65, 100, false, 2, 192, 0.92f);
    coverage(true, 66, 100, false, 8, 96, 0.7f);
    coverage(true, 50, 100, true, 8, 96, 0.7f);
    coverage(true, 0, 0, false, 8, 96, 0.7f);
    {
        int every = 8, swaps = 96;
        float decay = 0.7f;
        adapt_coverage_defaults(true, 50, 100, every, swaps, decay, true, false, true);
        check(every == 8 && swaps == 192 && decay == 0.7f, "individual CLI overrides are kept");
        every = 0;
        swaps = 96;
        adapt_coverage_defaults(true, 50, 100, every, swaps, decay, false, false, false);
        check(every == 0 && swaps == 96 && decay == 0.7f, "disabled adaptive tier stays disabled");
    }
    std::printf("adapt_decay_test: %d checks passed\n", checks);
    return 0;
}
