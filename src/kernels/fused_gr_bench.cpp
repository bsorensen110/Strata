// src/kernels/fused_gr_bench.cpp - fused_gr_read_multi and the isolated down-kernel arm timing.
//
//     build/fused_gr_bench [iters] [T min] [T max]
#include "strata/kernels/fused_gr.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace K = strata::kernels;

static uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int iters = argc > 1 ? std::atoi(argv[1]) : 500;
    const int t_lo = argc > 2 ? std::atoi(argv[2]) : 1, t_hi = argc > 3 ? std::atoi(argv[3]) : 6;
    const int N = 2560, HC = 4, D = N * HC, LR = 320, TM = K::kFusedGrMaxT;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto fill = [&](std::vector<float>& v, float sc) { for (auto& x : v) x = sc * nd(rng); };
    std::vector<float> R((size_t) TM * D), bo((size_t) TM * N), inj((size_t) TM * HC), wn(D);
    fill(R, 1.0f); fill(bo, 0.5f); fill(inj, 1.0f); fill(wn, 0.3f);
    for (auto& x : wn) x += 1.0f;
    std::vector<uint16_t> wd((size_t) LR * D), wu((size_t) D * LR), wi((size_t) HC * D);
    for (auto& x : wd) x = bf16(0.02f * nd(rng));
    for (auto& x : wu) x = bf16(0.05f * nd(rng));
    for (auto& x : wi) x = bf16(0.02f * nd(rng));
    float *dR, *dbo, *dinj, *dwn, *dxn, *dlo, *drs, *dio, *dmix;
    uint16_t *dwd, *dwu, *dwi;
    cudaMalloc((void**) &dR, R.size() * 4);
    cudaMalloc((void**) &dbo, bo.size() * 4);
    cudaMalloc((void**) &dinj, inj.size() * 4);
    cudaMalloc((void**) &dwn, wn.size() * 4);
    cudaMalloc((void**) &dxn, (size_t) TM * D * 4);
    cudaMalloc((void**) &dlo, (size_t) TM * LR * 4);
    cudaMalloc((void**) &drs, (size_t) TM * HC * 4);
    cudaMalloc((void**) &dio, (size_t) TM * HC * 4);
    cudaMalloc((void**) &dmix, (size_t) TM * N * 4);
    cudaMalloc((void**) &dwd, wd.size() * 2);
    cudaMalloc((void**) &dwu, wu.size() * 2);
    cudaMalloc((void**) &dwi, wi.size() * 2);
    cudaMemcpy(dbo, bo.data(), bo.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dinj, inj.data(), inj.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dwn, wn.data(), wn.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dwd, wd.data(), wd.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dwu, wu.data(), wu.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dwi, wi.data(), wi.size() * 2, cudaMemcpyHostToDevice);
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    int failures = 0;
    const bool hc_variant_bench = std::getenv("STRATA_HC_BENCH_DIRECT") != nullptr;
    const char* expect_reuse_env = std::getenv("STRATA_HC_EXPECT_REUSE");
    const bool expect_reuse_variant = expect_reuse_env != nullptr && expect_reuse_env[0] != '0';
    const char* expect_variant_env = std::getenv("STRATA_HC_EXPECT_VARIANT");
    const int expect_variant = expect_variant_env != nullptr ? std::atoi(expect_variant_env) : -1;
    const int selected_hc_variant = hc_variant_bench ? K::fused_gr_variant() : 0;
    if (hc_variant_bench && selected_hc_variant != 3 && selected_hc_variant != 4 && selected_hc_variant != 5 &&
        selected_hc_variant != 6 && selected_hc_variant != 7)
        return 2;
    if (expect_reuse_variant && selected_hc_variant != 5) {
        std::fprintf(stderr, "fused_gr_bench: expected two-row reuse variant 5, selected %d\n", selected_hc_variant);
        return 2;
    }
    if (expect_variant >= 0 && selected_hc_variant != expect_variant) {
        std::fprintf(stderr, "fused_gr_bench: expected HC variant %d, selected %d\n", expect_variant,
                     selected_hc_variant);
        return 2;
    }
    if (hc_variant_bench) {
        const char* variant_name = selected_hc_variant == 3    ? "staged"
                                   : selected_hc_variant == 4  ? "small-CTA staged"
                                   : selected_hc_variant == 5  ? "2-row-per-warp staged"
                                   : selected_hc_variant == 6  ? "register-pipe staged"
                                                               : "register-half staged";
        std::fprintf(stderr, "fused_gr_bench: selected HC variant %d (%s)\n", selected_hc_variant, variant_name);
        K::fused_gr_set_fast(0);
    }
    for (int T = t_lo; T <= t_hi; ++T)
        for (int apply = 0; apply < 2; ++apply)
            for (int inject = 0; inject < 2; ++inject) {
                K::FusedGrArgs a[TM];
                for (int t = 0; t < T; ++t) {
                    a[t].R = dR + (size_t) t * D; a[t].R_out = dR + (size_t) t * D; a[t].apply = apply;
                    a[t].bo_prev = dbo + (size_t) t * N; a[t].inj_prev = dinj + (size_t) t * HC;
                    a[t].w_norm = dwn; a[t].w_down = dwd; a[t].w_up = dwu; a[t].w_inject = inject ? dwi : nullptr;
                    a[t].lo = dlo + (size_t) t * LR; a[t].rs = drs + (size_t) t * HC;
                    a[t].inject_out = dio + (size_t) t * HC; a[t].mixed = dmix + (size_t) t * N;
                }
                std::vector<float> out[2];
                const int measurement_arms = hc_variant_bench ? 1 : 2;
                for (int f = 0; f < measurement_arms; ++f) {
                    if (hc_variant_bench && K::fused_gr_variant() != selected_hc_variant) return 2;
                    cudaMemcpy(dR, R.data(), R.size() * 4, cudaMemcpyHostToDevice);
                    cudaMemset(dlo, 0xff, (size_t) TM * LR * 4);
                    cudaMemset(drs, 0xff, (size_t) TM * HC * 4);
                    cudaMemset(dio, 0xff, (size_t) TM * HC * 4);
                    cudaMemset(dmix, 0xff, (size_t) TM * N * 4);
                    K::fused_gr_set_fast(hc_variant_bench ? 0 : f);
                    K::fused_gr_read_multi(a, T, dxn, s);
                    cudaStreamSynchronize(s);
                    auto& o = out[f];
                    o.resize((size_t) TM * (D + LR + HC + HC + N));
                    if (!hc_variant_bench) {
                        float* p = o.data();
                        cudaMemcpy(p, dR, (size_t) TM * D * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * D;
                        cudaMemcpy(p, dlo, (size_t) TM * LR * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * LR;
                        cudaMemcpy(p, drs, (size_t) TM * HC * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * HC;
                        cudaMemcpy(p, dio, (size_t) TM * HC * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * HC;
                        cudaMemcpy(p, dmix, (size_t) TM * N * 4, cudaMemcpyDeviceToHost);
                    }
                }
                const bool same = hc_variant_bench ? true : std::memcmp(out[0].data(), out[1].data(), out[0].size() * 4) == 0;
                if (hc_variant_bench) {
                    double kernel_us = 1e30;
                    for (int round = 0; round < 3; ++round) {
                        K::fused_gr_set_fast(0);
                        for (int i = 0; i < 30; ++i) K::fused_gr_read_multi(a, T, dxn, s);
                        cudaEventRecord(e0, s);
                        for (int i = 0; i < iters; ++i) K::fused_gr_read_multi(a, T, dxn, s);
                        cudaEventRecord(e1, s);
                        cudaEventSynchronize(e1);
                        float ms = 0.0f;
                        cudaEventElapsedTime(&ms, e0, e1);
                        kernel_us = std::fmin(kernel_us, 1e3 * ms / iters);
                    }
                    std::printf("T %d apply %d inject %d | HC-read %6.1f us (%s) | %s\n", T, apply, inject, kernel_us,
                                selected_hc_variant == 3    ? "staged"
                                : selected_hc_variant == 4  ? "small-CTA staged"
                                : selected_hc_variant == 5  ? "2-row-per-warp staged"
                                : selected_hc_variant == 6  ? "register-pipe staged"
                                                            : "register-half staged",
                                hc_variant_bench ? "timed-only; parity is a separate gate"
                                                 : (same ? "bitwise equal" : "DIFFERS"));
                    if (!same) ++failures;
                    continue;
                }
                double us[2] = {1e30, 1e30};
                for (int round = 0; round < 3; ++round)
                    for (int f = 0; f < 2; ++f) {
                        K::fused_gr_set_fast(f);
                        for (int i = 0; i < 20; ++i) K::fused_gr_read_multi(a, T, dxn, s);
                        cudaEventRecord(e0, s);
                        for (int i = 0; i < iters; ++i) K::fused_gr_read_multi(a, T, dxn, s);
                        cudaEventRecord(e1, s);
                        cudaEventSynchronize(e1);
                        float ms = 0;
                        cudaEventElapsedTime(&ms, e0, e1);
                        us[f] = std::fmin(us[f], 1e3 * ms / iters);
                    }
                std::printf("T %d apply %d inject %d | old %6.1f us  fast %6.1f us | %s\n", T, apply, inject, us[0], us[1],
                            same ? "bitwise equal" : "DIFFERS");
                if (!same) ++failures;
            }
    std::printf("fused_gr_bench: %d differing\n", failures);
    return failures ? 1 : 0;
}
