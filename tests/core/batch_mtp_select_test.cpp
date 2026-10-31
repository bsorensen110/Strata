// STRATA_BATCH_MTP_SELECT's slot choice (include/strata/core/batch_mtp_select.hpp): CPU-only.
#include "strata/core/batch_mtp_select.hpp"

#include <cstdio>

using namespace strata::core;

static int g_fail = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) {                                                           \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                         \
        }                                                                     \
    } while (0)

int main() {
    // config
    CHECK(!mtp_select_config(nullptr, nullptr).on);
    CHECK(!mtp_select_config("", "1").on);
    CHECK(!mtp_select_config("0", "1").on);
    CHECK(mtp_select_config("1", nullptr).on && mtp_select_config("1", nullptr).starve == 2);
    CHECK(mtp_select_config("5", nullptr).starve == 5);
    CHECK(mtp_select_config("1", nullptr).overlap_w == 0.0f);
    CHECK(mtp_select_config("1", "1").overlap_w == 1.0f);
    CHECK(mtp_select_config("1", "-3").overlap_w == 0.0f);

    int out[8] = {};
    const int wait0[6] = {0, 0, 0, 0, 0, 0};
    const float ema[6] = {0.2f, 0.9f, 0.5f, 0.8f, 0.1f, 0.7f};

    // off: the rotation order's first m
    {
        const MtpSelectConfig off;
        CHECK(mtp_select_slots(6, wait0, ema, nullptr, 4, off, out) == 4);
        CHECK(out[0] == 0 && out[1] == 1 && out[2] == 2 && out[3] == 3);
    }
    // on, room for all: unchanged
    {
        const MtpSelectConfig c = mtp_select_config("1", nullptr);
        CHECK(mtp_select_slots(3, wait0, ema, nullptr, 4, c, out) == 3);
        CHECK(out[0] == 0 && out[1] == 1 && out[2] == 2);
    }
    // on: the highest acceptance, returned in the rotation order
    {
        const MtpSelectConfig c = mtp_select_config("1", nullptr);
        CHECK(mtp_select_slots(6, wait0, ema, nullptr, 3, c, out) == 3);
        CHECK(out[0] == 1 && out[1] == 3 && out[2] == 5);
    }
    // starvation: a slot that waited `starve` windows goes first, whatever its acceptance
    {
        const MtpSelectConfig c = mtp_select_config("2", nullptr);
        const int wait[6] = {0, 0, 0, 0, 2, 0};
        CHECK(mtp_select_slots(6, wait, ema, nullptr, 3, c, out) == 3);
        CHECK(out[0] == 1 && out[1] == 3 && out[2] == 4);
    }
    // more starved than room: the first in the rotation order
    {
        const MtpSelectConfig c = mtp_select_config("2", nullptr);
        const int wait[6] = {3, 3, 3, 3, 0, 0};
        CHECK(mtp_select_slots(6, wait, ema, nullptr, 2, c, out) == 2);
        CHECK(out[0] == 0 && out[1] == 1);
    }
    // a simulated run: no slot waits more than `starve` windows
    {
        const MtpSelectConfig c = mtp_select_config("2", nullptr);
        int wait[6] = {};
        int most = 0;
        for (int w = 0; w < 100; ++w) {
            const int k = mtp_select_slots(6, wait, ema, nullptr, 2, c, out);
            CHECK(k == 2);
            bool ran[6] = {};
            for (int i = 0; i < k; ++i) ran[out[i]] = true;
            for (int i = 0; i < 6; ++i) {
                wait[i] = ran[i] ? 0 : wait[i] + 1;
                if (wait[i] > most) most = wait[i];
            }
        }
        CHECK(most <= 2);
    }
    // the bound holds for many shapes and acceptance mixes (or the tightest one the window allows)
    {
        uint32_t rng = 12345u;
        auto next = [&]() { rng = rng * 1664525u + 1013904223u; return rng >> 8; };
        for (int n = 2; n <= 12; ++n)
            for (int m = 1; m < n; ++m)
                for (int starve = 2; starve <= 4; ++starve) {
                    char sv[8];
                    std::snprintf(sv, sizeof sv, "%d", starve);
                    const MtpSelectConfig c = mtp_select_config(sv, nullptr);
                    const int fair = (n + m - 1) / m - 1;
                    const int bound = starve > fair ? starve : fair;
                    int wait[12] = {}, out12[12] = {};
                    float em[12];
                    int most = 0;
                    for (int w = 0; w < 300; ++w) {
                        for (int i = 0; i < n; ++i) em[i] = (float) (next() % 1000) / 1000.0f;
                        const int k = mtp_select_slots(n, wait, em, nullptr, m, c, out12);
                        CHECK(k == m);
                        bool ran[12] = {};
                        for (int i = 0; i < k; ++i) ran[out12[i]] = true;
                        for (int i = 0; i < n; ++i) {
                            wait[i] = ran[i] ? 0 : wait[i] + 1;
                            if (wait[i] > most) most = wait[i];
                        }
                    }
                    CHECK(most <= bound);
                }
    }
    // overlap: with equal acceptance, the slot whose experts overlap the chosen one's wins
    {
        const MtpSelectConfig c = mtp_select_config("1", "1");
        RouteSig s0, s1, s2;
        for (int e = 0; e < 40; ++e) {
            route_sig_add(s0, 3, e);
            route_sig_add(s2, 3, e);       // the same experts as slot 0
            route_sig_add(s1, 3, 100 + e);  // others
        }
        CHECK(route_sig_count(s0) > 30);
        CHECK(route_sig_common(s0, s2) == route_sig_count(s0));
        CHECK(route_sig_common(s0, s1) < route_sig_count(s0) / 4);
        const RouteSig* sig[3] = {&s0, &s1, &s2};
        const float e3[3] = {0.9f, 0.5f, 0.5f};
        const int w3[3] = {0, 0, 0};
        CHECK(mtp_select_slots(3, w3, e3, sig, 2, c, out) == 2);
        CHECK(out[0] == 0 && out[1] == 2);
        // without the overlap weight: the tie keeps the rotation order (slot 1)
        const MtpSelectConfig c0 = mtp_select_config("1", nullptr);
        CHECK(mtp_select_slots(3, w3, e3, sig, 2, c0, out) == 2);
        CHECK(out[0] == 0 && out[1] == 1);
    }
    {
        RouteSig s;
        route_sig_add(s, 0, -1);   // a negative expert is ignored
        CHECK(route_sig_count(s) == 0);
    }
    if (g_fail == 0) std::printf("batch_mtp_select_test: ok\n");
    return g_fail == 0 ? 0 : 1;
}
