// src/kernels/tcq_v_host_parity.cpp - host-only parity gate for include/strata/kernels/tcq_v.hpp.
//
// Oracle: tests/data/tcq_v_golden.txt, written by tools/tcq_golden.py from the numpy reference (no code shared with
// this file). Checks, per trellis (4, 8, 16 states):
//   1. the next-state table equals the numpy table entry for entry, and delta equals the fitted constant;
//   2. the decoder reproduces the golden reconstruction from the golden codes (max abs diff <= 1e-5 at scale 1);
//   3. the C++ Viterbi path costs the same as the numpy path on every block (|cost_cpp - cost_np| <= 1e-3), and
//      the C++ path's own decode reproduces the cost it reports (encoder/decoder agreement);
//   4. code mismatches against numpy are reported (ties in float32 vs float64 can flip a code without changing cost).
// Then the writer: f16 RMS scale and a zero block.
//
// Build (host only, no CUDA):  cl /nologo /std:c++20 /EHsc /W4 /I include src\kernels\tcq_v_host_parity.cpp
// Run:                          tcq_v_host_parity.exe tests\data\tcq_v_golden.txt
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "strata/kernels/tcq_v.hpp"

using namespace strata::kernels;

static int fails = 0;
#define CHECK(cond, ...)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            ++fails;                                                  \
            std::printf("FAIL line %d: ", __LINE__);                  \
            std::printf(__VA_ARGS__);                                 \
            std::printf("\n");                                        \
        }                                                             \
    } while (0)

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "tests/data/tcq_v_golden.txt";
    std::ifstream in(path);
    if (!in) {
        std::printf("cannot open %s\n", path);
        return 2;
    }
    std::string tag;
    int ver = 0;
    in >> tag >> ver;
    CHECK(tag == "TCQ_GOLDEN" && ver == 1, "bad golden header '%s' %d", tag.c_str(), ver);

    for (int sec = 0; sec < 3; ++sec) {
        std::string w_states, w_delta, w_blocks, w_mse, w_next;
        int states = 0, nblocks = 0;
        float delta = 0.0f, mse_np = 0.0f;
        std::string kSTATES, kDELTA, kBLOCKS, kMSE;
        in >> kSTATES >> states >> kDELTA >> delta >> kBLOCKS >> nblocks >> kMSE >> mse_np;
        CHECK(kSTATES == "STATES" && kDELTA == "DELTA" && kBLOCKS == "BLOCKS" && kMSE == "MSE", "section header");
        in >> w_next;
        CHECK(w_next == "NEXT", "NEXT tag");
        std::vector<int> next(states * 4);
        for (auto& v : next) in >> v;

        CHECK(std::fabs(tcq_delta(states) - delta) < 1e-7f, "%d states: delta %.7f vs golden %.7f", states,
              tcq_delta(states), delta);
        int table_bad = 0;
        for (int s = 0; s < states; ++s)
            for (int k = 0; k < 4; ++k)
                if (tcq_next(states, s, k) != next[s * 4 + k]) ++table_bad;
        CHECK(table_bad == 0, "%d states: %d NEXT entries differ from numpy", states, table_bad);

        double worst_dec = 0.0, worst_gap = 0.0, mse_cpp_sum = 0.0;
        int code_mismatch = 0, dec_bad = 0;
        for (int b = 0; b < nblocks; ++b) {
            std::string tu, tq, tr;
            in >> tu;
            float u[kTcqBlock];
            for (auto& v : u) in >> v;
            in >> tq;
            int q[8];
            for (auto& v : q) in >> v;
            in >> tr;
            float r[kTcqBlock];
            for (auto& v : r) in >> v;
            CHECK(tu == "U" && tq == "Q" && tr == "R", "block %d tags", b);

            uint8_t qs[8];
            for (int i = 0; i < 8; ++i) qs[i] = (uint8_t) q[i];

            // 2. decoder vs golden reconstruction
            float dec[kTcqBlock];
            tcq_decode_block(states, 1.0f, qs, dec);
            for (int t = 0; t < kTcqBlock; ++t) {
                const double diff = std::fabs((double) dec[t] - (double) r[t]);
                if (diff > worst_dec) worst_dec = diff;
                if (diff > 1e-5) ++dec_bad;
            }

            // 3. encoder: cost of the C++ path vs the numpy path's cost on the same block
            uint8_t cq[8];
            const float best = tcq_encode_block(states, u, cq);
            double np_cost = 0.0, cpp_cost = 0.0;
            for (int t = 0; t < kTcqBlock; ++t) np_cost += (double) (r[t] - u[t]) * (double) (r[t] - u[t]);
            float dc_all[kTcqBlock];
            tcq_decode_block(states, 1.0f, cq, dc_all);
            for (int t = 0; t < kTcqBlock; ++t) cpp_cost += (double) (dc_all[t] - u[t]) * (double) (dc_all[t] - u[t]);
            mse_cpp_sum += cpp_cost;
            const double gap_np = std::fabs(cpp_cost - np_cost);
            if (gap_np > worst_gap) worst_gap = gap_np;
            CHECK(std::fabs(cpp_cost - (double) best) < 1e-3, "%d states block %d: path cost %.6f vs reported %.6f",
                  states, b, cpp_cost, (double) best);
            for (int t = 0; t < kTcqBlock; ++t)
                if (tcq_code_at(cq, t) != tcq_code_at(qs, t)) ++code_mismatch;
        }
        const int n_values = nblocks * kTcqBlock;
        CHECK(dec_bad == 0, "%d states: %d decoded values differ from numpy by > 1e-5 (worst %.2e)", states, dec_bad,
              worst_dec);
        CHECK(worst_gap <= 1e-3, "%d states: worst per-block cost gap vs numpy %.2e", states, worst_gap);
        std::printf("%2d states: decoder worst |diff| %.1e | code mismatches %d / %d values | path cost gap %.1e | "
                    "C++ mse %.5f vs numpy %.5f\n",
                    states, worst_dec, code_mismatch, n_values, worst_gap, mse_cpp_sum / n_values, mse_np);
    }

    // 4. writer: f16 RMS scale, decode round trip, zero block
    {
        float x[kTcqBlock];
        float ss = 0.0f;
        for (int t = 0; t < kTcqBlock; ++t) {
            x[t] = std::sin(0.7f * (float) t) * 1.5f + 0.1f * (float) t;
            ss += x[t] * x[t];
        }
        const float rms = std::sqrt(ss / kTcqBlock);
        block_tcq2_0 blk{};
        tcq_quantize_block(4, x, &blk);
        const float s = f32_from_f16(blk.d);
        CHECK(std::fabs(s - rms) <= rms * 2e-3f, "writer scale %.6f vs RMS %.6f", s, rms);
        float dec[kTcqBlock];
        tcq_decode_block(4, s, blk.qs, dec);
        double num = 0.0, den = 0.0;
        for (int t = 0; t < kTcqBlock; ++t) {
            num += (dec[t] - x[t]) * (dec[t] - x[t]);
            den += x[t] * x[t];
        }
        const double rel = std::sqrt(num / den);
        std::printf("writer: scale %.6f (RMS %.6f), block relative RMS error %.3f (2-bit)\n", s, rms, rel);
        CHECK(rel < 0.5, "writer block error %.3f too large", rel);

        float z[kTcqBlock] = {};
        block_tcq2_0 zb{};
        tcq_quantize_block(4, z, &zb);
        float zd[kTcqBlock];
        tcq_decode_block(4, f32_from_f16(zb.d), zb.qs, zd);
        bool all_zero = zb.d == 0;
        for (float v : zd) all_zero = all_zero && v == 0.0f;
        CHECK(all_zero, "zero block must decode to exact zeros");
    }

    std::printf(fails ? "FAILED: %d check(s)\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
