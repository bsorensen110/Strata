// src/kernels/df2_gumbel_parity.cpp - P3 parity test for host gumbel_exp_host vs device gumbel_exp
// and host q_T / top-2 Gumbel score / photo-finish branch-gate helpers.
// Host subtest always runs; device subtest exits 0 with a SKIP marker when STRATA_TEST_NO_GPU=1 or no GPU is present.
#include "strata/core/spec_prob.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/spec/df2_tree.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const char* msg) {
    if (!ok) {
        std::fprintf(stderr, "df2_gumbel_parity FAIL: %s\n", msg);
        ++g_failures;
    }
}

void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "df2_gumbel_parity: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void run_host_tests() {
    using namespace strata::core;
    using namespace strata::spec;

    // 1. Determinism, positivity, and edge-input finiteness for gumbel_exp_host.
    const std::array<uint64_t, 5> edge_seeds{0ULL, 1ULL, 0x9E3779B97F4A7C15ULL, UINT64_MAX - 1, UINT64_MAX};
    const std::array<uint64_t, 5> edge_ctrs{0ULL, 1ULL, 1ULL << 56, UINT64_MAX - 1, UINT64_MAX};
    const std::array<uint32_t, 5> edge_toks{0U, 1U, 248319U, UINT32_MAX - 1, UINT32_MAX};
    for (uint64_t s : edge_seeds) {
        for (uint64_t c : edge_ctrs) {
            for (uint32_t t : edge_toks) {
                const double e1 = gumbel_exp_host(s, c, t);
                const double e2 = gumbel_exp_host(s, c, t);
                check(e1 == e2 && std::isfinite(e1) && e1 > 0.0,
                      "gumbel_exp_host is deterministic, finite, and strictly positive on edge triples");
            }
        }
    }

    // 2. Empirical Exp(1) mean (~1.0) and variance (~1.0) over 100k samples.
    {
        constexpr int kN = 100000;
        double sum = 0.0;
        double sum2 = 0.0;
        for (int i = 0; i < kN; ++i) {
            const double e = gumbel_exp_host(0xCAFEBABE12345678ULL, static_cast<uint64_t>(i), static_cast<uint32_t>(i % 257));
            sum += e;
            sum2 += e * e;
        }
        const double mean = sum / kN;
        const double var = sum2 / kN - mean * mean;
        check(std::fabs(mean - 1.0) < 0.02, "gumbel_exp_host empirical mean matches Exp(1)");
        check(std::fabs(var - 1.0) < 0.05, "gumbel_exp_host empirical variance matches Exp(1)");
    }

    // 3. q_T constant-logit-shift invariance across T = 0.3, 1.0, 1.5.
    {
        std::array<int32_t, kDf2TreeSelectorTopK> toks{};
        toks.fill(-1);
        std::array<float, kDf2TreeSelectorTopK> scores{};
        std::array<float, kDf2TreeSelectorTopK> shifted{};
        scores.fill(-INFINITY);
        shifted.fill(-INFINITY);
        for (int j = 0; j < 4; ++j) {
            toks[static_cast<size_t>(j)] = 100 + j;
            scores[static_cast<size_t>(j)] = -0.3f * static_cast<float>(j);
            shifted[static_cast<size_t>(j)] = scores[static_cast<size_t>(j)] + 25.0f;
        }
        for (float temp : {0.3f, 1.0f, 1.5f}) {
            const auto r_a = df2_gm_row_top2(toks.data(), scores.data(), 0x12345ULL, 77ULL, temp);
            const auto r_b = df2_gm_row_top2(toks.data(), shifted.data(), 0x12345ULL, 77ULL, temp);
            check(r_a.valid && r_b.valid && r_a.has_runner && r_b.has_runner, "df2_gm_row_top2 succeeds");
            check(r_a.tok0 == r_b.tok0 && r_a.tok1 == r_b.tok1 &&
                      std::fabs(r_a.q0 - r_b.q0) < 1e-5f &&
                      std::fabs(r_a.q1 - r_b.q1) < 1e-5f &&
                      std::fabs(r_a.delta_g - r_b.delta_g) < 1e-5,
                  "df2_gm_row_top2 is invariant under constant logit shift");
        }
        // Exact dyadic shift invariance down to FP64 machine precision.
        std::array<float, kDf2TreeSelectorTopK> dyadic{};
        std::array<float, kDf2TreeSelectorTopK> dyadic_shift{};
        dyadic.fill(-INFINITY);
        dyadic_shift.fill(-INFINITY);
        for (int j = 0; j < 4; ++j) {
            dyadic[static_cast<size_t>(j)] = -0.25f * static_cast<float>(j);
            dyadic_shift[static_cast<size_t>(j)] = dyadic[static_cast<size_t>(j)] + 16.0f;
        }
        const auto d_a = df2_gm_row_top2(toks.data(), dyadic.data(), 0x12345ULL, 77ULL, 0.5f);
        const auto d_b = df2_gm_row_top2(toks.data(), dyadic_shift.data(), 0x12345ULL, 77ULL, 0.5f);
        check(d_a.valid && d_b.valid && d_a.tok0 == d_b.tok0 && d_a.tok1 == d_b.tok1 &&
                  std::fabs(d_a.delta_g - d_b.delta_g) < 1e-12,
              "df2_gm_row_top2 is exact under dyadic constant logit shift");
    }

    // 4. Duplicate candidate token merging + single-candidate row behavior.
    {
        std::array<int32_t, kDf2TreeSelectorTopK> toks{};
        toks.fill(-1);
        std::array<float, kDf2TreeSelectorTopK> scores{};
        scores.fill(-INFINITY);
        toks[0] = 42;
        toks[1] = 42;
        scores[0] = 0.0f;
        scores[1] = 0.0f;
        const auto single_dup = df2_gm_row_top2(toks.data(), scores.data(), 0x1ULL, 2ULL, 1.0f);
        check(single_dup.valid && single_dup.tok0 == 42 && !single_dup.has_runner &&
                  !std::isfinite(single_dup.delta_g) &&
                  !df2_gm_photo_finish_fires(single_dup, 0.35f) &&
                  !df2_gm_photo_finish_fires(single_dup, 10.0f),
              "duplicate single unique token yields no runner and never fires photo-finish");
    }

    // 5. Boundary threshold and monotonic fire rate over 10k counters.
    {
        std::array<int32_t, kDf2TreeSelectorTopK> toks{};
        toks.fill(-1);
        std::array<float, kDf2TreeSelectorTopK> scores{};
        scores.fill(-INFINITY);
        toks[0] = 10;
        toks[1] = 20;
        toks[2] = 30;
        scores[0] = std::log(0.45f);
        scores[1] = std::log(0.40f);
        scores[2] = std::log(0.15f);

        const auto row = df2_gm_row_top2(toks.data(), scores.data(), 0x555ULL, 99ULL, 0.9f);
        check(row.valid && row.has_runner && row.delta_g >= 0.0, "multi-candidate row has non-negative delta_g");
        const float dg = static_cast<float>(row.delta_g);
        check(df2_gm_photo_finish_fires(row, dg + 1e-3f), "photo-finish fires strictly below delta");
        check(!df2_gm_photo_finish_fires(row, std::max(0.0f, dg - 1e-3f)),
              "photo-finish does not fire above/at delta");
        check(!df2_gm_photo_finish_fires(row, 0.0f) && !df2_gm_photo_finish_fires(row, -0.5f),
              "photo-finish rejects non-positive delta");

        const std::array<float, 4> deltas{0.15f, 0.35f, 0.70f, 1.50f};
        std::array<int, 4> fires{};
        for (int i = 0; i < 10000; ++i) {
            const auto r = df2_gm_row_top2(toks.data(), scores.data(), 0xABCDULL, static_cast<uint64_t>(i), 0.9f);
            for (size_t d = 0; d < deltas.size(); ++d) {
                if (df2_gm_photo_finish_fires(r, deltas[d])) ++fires[d];
            }
        }
        check(fires[0] > 0 && fires[0] < fires[1] && fires[1] < fires[2] && fires[2] < fires[3],
              "photo-finish fire rate increases strictly with delta");
        std::printf("df2_gumbel_parity host: %s (fires/10k @0.15=%d @0.35=%d @0.70=%d @1.50=%d)\n",
                    g_failures == 0 ? "OK" : "FAILED", fires[0], fires[1], fires[2], fires[3]);
    }
}

void run_device_parity_test() {
    const char* no_gpu = std::getenv("STRATA_TEST_NO_GPU");
    if (no_gpu != nullptr && std::strcmp(no_gpu, "1") == 0) {
        std::printf("SKIP: df2_gumbel_parity device subtest: STRATA_TEST_NO_GPU=1\n");
        return;
    }
    int dev_count = 0;
    const cudaError_t dc = cudaGetDeviceCount(&dev_count);
    if (dc != cudaSuccess || dev_count <= 0) {
        std::printf("SKIP: df2_gumbel_parity device subtest: no GPU device available\n");
        return;
    }

    constexpr int kN = 4096;
    std::vector<uint64_t> seeds(kN);
    std::vector<uint64_t> counters(kN);
    std::vector<uint32_t> tokens(kN);
    std::vector<double> host_out(kN);
    std::vector<double> dev_host_out(kN, 0.0);

    std::mt19937_64 rng(0xD3F29876ULL);
    int idx = 0;
    const std::array<uint64_t, 4> edge_s{0ULL, 1ULL, 0x9E3779B97F4A7C15ULL, UINT64_MAX};
    const std::array<uint64_t, 4> edge_c{0ULL, 1ULL, (1ULL << 56) | 7ULL, UINT64_MAX};
    const std::array<uint32_t, 4> edge_t{0U, 1U, 248319U, UINT32_MAX};
    for (uint64_t s : edge_s) {
        for (uint64_t c : edge_c) {
            for (uint32_t t : edge_t) {
                seeds[static_cast<size_t>(idx)] = s;
                counters[static_cast<size_t>(idx)] = c;
                tokens[static_cast<size_t>(idx)] = t;
                ++idx;
            }
        }
    }
    for (; idx < kN; ++idx) {
        seeds[static_cast<size_t>(idx)] = rng();
        counters[static_cast<size_t>(idx)] = (idx % 3 == 0) ? static_cast<uint64_t>(idx) : rng();
        tokens[static_cast<size_t>(idx)] = static_cast<uint32_t>(rng() % 248320ULL);
    }
    for (int i = 0; i < kN; ++i) {
        host_out[static_cast<size_t>(i)] =
            strata::core::gumbel_exp_host(seeds[static_cast<size_t>(i)],
                                          counters[static_cast<size_t>(i)],
                                          tokens[static_cast<size_t>(i)]);
    }

    uint64_t* d_seed = nullptr;
    uint64_t* d_counter = nullptr;
    uint32_t* d_token = nullptr;
    double* d_out = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_seed), kN * sizeof(uint64_t)), "cudaMalloc d_seed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_counter), kN * sizeof(uint64_t)), "cudaMalloc d_counter");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_token), kN * sizeof(uint32_t)), "cudaMalloc d_token");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_out), kN * sizeof(double)), "cudaMalloc d_out");

    check_cuda(cudaMemcpy(d_seed, seeds.data(), kN * sizeof(uint64_t), cudaMemcpyHostToDevice), "H2D seed");
    check_cuda(cudaMemcpy(d_counter, counters.data(), kN * sizeof(uint64_t), cudaMemcpyHostToDevice), "H2D counter");
    check_cuda(cudaMemcpy(d_token, tokens.data(), kN * sizeof(uint32_t), cudaMemcpyHostToDevice), "H2D token");

    if (!strata::kernels::eval_gumbel_exp_device(d_seed, d_counter, d_token, kN, d_out, nullptr)) {
        std::fprintf(stderr, "df2_gumbel_parity FAIL: eval_gumbel_exp_device failed\n");
        ++g_failures;
    } else {
        check_cuda(cudaMemcpy(dev_host_out.data(), d_out, kN * sizeof(double), cudaMemcpyDeviceToHost), "D2H out");
        int bit_exact = 0;
        int ulp1_ocml_log = 0;
        int gt1_ulp = 0;
        uint64_t max_ulp = 0;
        for (int i = 0; i < kN; ++i) {
            const double hv = host_out[static_cast<size_t>(i)];
            const double dv = dev_host_out[static_cast<size_t>(i)];
            if (!std::isfinite(dv) || !(dv > 0.0)) {
                ++gt1_ulp;
                continue;
            }
            uint64_t hbits = 0, dbits = 0;
            std::memcpy(&hbits, &hv, sizeof(double));
            std::memcpy(&dbits, &dv, sizeof(double));
            const uint64_t ulp = (hbits >= dbits) ? (hbits - dbits) : (dbits - hbits);
            if (ulp > max_ulp) max_ulp = ulp;
            if (ulp == 0) {
                ++bit_exact;
            } else if (ulp == 1) {
                ++ulp1_ocml_log;
            } else {
                if (gt1_ulp < 5) {
                    std::fprintf(stderr,
                                 "df2_gumbel_parity MISMATCH i=%d seed=0x%llx ctr=0x%llx tok=%u "
                                 "host=%.17g (0x%llx) dev=%.17g (0x%llx) ulp=%llu\n",
                                 i, static_cast<unsigned long long>(seeds[static_cast<size_t>(i)]),
                                 static_cast<unsigned long long>(counters[static_cast<size_t>(i)]),
                                 tokens[static_cast<size_t>(i)], hv,
                                 static_cast<unsigned long long>(hbits), dv,
                                 static_cast<unsigned long long>(dbits),
                                 static_cast<unsigned long long>(ulp));
                }
                ++gt1_ulp;
            }
        }
        check(gt1_ulp == 0, "device gumbel_exp == host gumbel_exp_host within <= 1 ULP across 4096 triples");
        std::printf("df2_gumbel_parity device: %s (%d triples, bit_exact=%d, ulp1_ocml_log=%d, max_ulp=%llu)\n",
                    gt1_ulp == 0 ? "OK" : "FAILED", kN, bit_exact, ulp1_ocml_log,
                    static_cast<unsigned long long>(max_ulp));
    }

    (void) cudaFree(d_seed);
    (void) cudaFree(d_counter);
    (void) cudaFree(d_token);
    (void) cudaFree(d_out);
}

}  // namespace

int main() {
    run_host_tests();
    run_device_parity_test();
    return g_failures == 0 ? 0 : 1;
}
