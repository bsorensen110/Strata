// tests/core/batch_mtp_depth_test.cpp - adaptive --batch-mtp draft depth (include/strata/core/batch_mtp_depth.hpp):
// the env parse (unset = off), the acceptance EMA, the depth gate ema^d >= floor, the probability trim, and the
// window fit that never drops a slot's first proposal.
//
// Header-only, CPU: no verifier, no drafter, no GPU.
#include "strata/core/batch_mtp_depth.hpp"

#include <cstdio>

using namespace strata::core;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    if (ok) return;
    ++g_fail;
    std::fprintf(stderr, "FAIL: %s\n", what);
}
}  // namespace

int main() {
    // env parse
    check(!mtp_depth_config(nullptr, nullptr, 7).on(), "unset is off");
    check(!mtp_depth_config("", "", 7).on(), "empty is off");
    check(!mtp_depth_config("1", nullptr, 7).on(), "1 is off");
    check(!mtp_depth_config("junk", nullptr, 7).on(), "unparsable is off");
    check(mtp_depth_config("4", nullptr, 7).max_depth == 4, "4 parses");
    check(mtp_depth_config("99", nullptr, 7).max_depth == 7, "clamped to cap");
    check(mtp_depth_config("-3", nullptr, 7).max_depth == 1, "negative is off");
    check(mtp_depth_config("4", "0.7", 7).floor == 0.7f, "floor parses");
    check(mtp_depth_config("4", "0", 7).floor == 0.5f, "floor 0 rejected");
    check(mtp_depth_config("4", "1.5", 7).floor == 0.5f, "floor >1 rejected");

    // EMA
    check(mtp_depth_ema(0.5f, true, 0.25f) == 0.625f, "ema up");
    check(mtp_depth_ema(0.5f, false, 0.25f) == 0.375f, "ema down");
    float e = 0.5f;
    for (int i = 0; i < 100; ++i) e = mtp_depth_ema(e, true, 0.25f);
    check(e > 0.99f && e <= 1.0f, "ema converges to 1");

    // depth gate
    MtpDepthConfig c = mtp_depth_config("4", "0.5", 7);
    check(mtp_depth_for(0.5f, c) == 1, "new slot stays at 1");
    check(mtp_depth_for(0.0f, c) == 1, "never below 1");
    check(mtp_depth_for(0.75f, c) == 2, "0.75^2=0.56 -> 2");
    check(mtp_depth_for(0.80f, c) == 3, "0.8^3=0.512 -> 3");
    check(mtp_depth_for(0.95f, c) == 4, "capped at max_depth");
    check(mtp_depth_for(1.0f, c) == 4, "ema 1 -> max");
    MtpDepthConfig off;
    check(mtp_depth_for(1.0f, off) == 1, "off -> 1");

    // probability trim
    const float p1[4] = {0.9f, 0.9f, 0.5f, 0.9f};
    check(mtp_depth_trim(p1, 4, 0.5f) == 2, "0.9*0.9=0.81 keep, *0.5=0.405 stop");
    const float p2[3] = {0.1f, 0.99f, 0.99f};
    check(mtp_depth_trim(p2, 3, 0.5f) == 1, "first always kept");
    check(mtp_depth_trim(nullptr, 3, 0.5f) == 3, "no probs keeps all");
    check(mtp_depth_trim(p1, 0, 0.5f) == 0, "n 0");

    // fit
    {
        int k[4] = {3, 3, 3, 3};
        const float ema[4] = {0.9f, 0.6f, 0.8f, 0.7f};
        const int rows = mtp_depth_fit(k, ema, 4, 8);
        check(rows == 8, "fits to 8");
        for (int a = 0; a < 4; ++a) check(k[a] == 1, "8 rows: one proposal each");
    }
    {
        int k[4] = {3, 3, 3, 3};
        const float ema[4] = {0.9f, 0.6f, 0.8f, 0.7f};
        const int rows = mtp_depth_fit(k, ema, 4, 12);
        check(rows == 12, "fits to 12");
        check(k[0] == 3 && k[1] == 1 && k[2] == 3 && k[3] == 1, "the best slot keeps its depth, the worst loses first");
    }
    {
        int k[2] = {2, 1};
        const float ema[2] = {0.9f, 0.9f};
        check(mtp_depth_fit(k, ema, 2, 16) == 5 && k[0] == 2, "under cap untouched");
    }
    {
        int k[2] = {0, 2};
        const float ema[2] = {0.9f, 0.9f};
        check(mtp_depth_fit(k, ema, 2, 3) == 3 && k[0] == 0 && k[1] == 1, "slot without draft stays at 0");
    }

    if (g_fail != 0) {
        std::fprintf(stderr, "batch_mtp_depth_test: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("batch_mtp_depth_test: ok\n");
    return 0;
}
