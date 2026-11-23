// src/kernels/cpu/cpu_act_quant_wide_test.cpp - tests wide-window (T = 9..16) CPU activation quantization
// (ExpertPool::defer_act_quant) and ExpertJobMulti slicing across s2 and native Q4_0 experts.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "ggml.h"

#include <algorithm>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

void make_s2_blob(uint8_t* b, std::mt19937& rng) {
    for (size_t i = 0; i < cpu::O_GU_SCALES; ++i) b[i] = (uint8_t) rng();
    for (size_t i = cpu::O_GU_SCALES; i < cpu::BLOB; i += 2) {
        const uint16_t h = (uint16_t) (0x1C00 + rng() % 0x0800);
        std::memcpy(b + i, &h, 2);
    }
}

}  // namespace

int main() {
    if (!cpu::cpu_avx1_ok()) {
        std::printf("cpu_act_quant_wide_test: no AVX on this CPU, skipped\n");
        return 0;
    }

    constexpr int T = strata::kernels::kWindowMaxT;
    if (T <= cpu::MAXT) {
        std::printf("cpu_act_quant_wide_test: narrow build (kWindowMaxT=%d), checking T=%d only\n", T, T);
    }

    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, cpu::H, cpu::FF, f, err)) {
        std::fprintf(stderr, "native_fmt failed: %s\n", err.c_str());
        return 1;
    }

    std::mt19937 rng(42);
    std::vector<uint8_t> native_blob(f.bytes);
    {
        std::vector<float> w((size_t) cpu::H * cpu::FF);
        for (size_t i = 0; i < w.size(); ++i) w[i] = 0.02f * std::cos((float) i * 0.13f);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, w.data(), native_blob.data(), 0, cpu::FF, cpu::H, nullptr);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, w.data(), native_blob.data() + f.up_off, 0, cpu::FF, cpu::H, nullptr);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, w.data(), native_blob.data() + f.down_off, 0, cpu::H, cpu::FF, nullptr);
    }
    std::vector<uint8_t> s2_blob(cpu::BLOB);
    make_s2_blob(s2_blob.data(), rng);

    std::vector<float> xf((size_t) T * cpu::H);
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < cpu::H; ++i)
            xf[(size_t) t * cpu::H + i] = std::sin((float) (i + 29 * t) * 0.041f) * (1.0f + 0.05f * t);
    }

    // Single-row references for all t in [0, T).
    std::vector<cpu::ActQ> ref_act(T);
    std::vector<uint8_t> ref_nact((size_t) T * cpu::kNativeActBytes, 0x5a);
    std::vector<int> toks(T);
    for (int t = 0; t < T; ++t) {
        toks[t] = t;
        cpu::act_quant_q8_1(xf.data() + (size_t) t * cpu::H, cpu::H, ref_act[t]);
        cpu::native_quant_act(f, xf.data() + (size_t) t * cpu::H, ref_nact.data() + (size_t) t * cpu::kNativeActBytes);
    }

    cpu::ExpertPool pool(3, false);
    int fail = 0;

    // 1. Test deferred native activation quantization across rows 0..T-1 (especially rows 8..15) + run_split_multi_native.
    {
        std::vector<cpu::ActQ> def_act(T);
        std::vector<uint8_t> def_nact((size_t) T * cpu::kNativeActBytes, 0x5a);
        std::vector<float> out_wide((size_t) T * cpu::H, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> out_split((size_t) T * cpu::H, std::numeric_limits<float>::quiet_NaN());

        pool.clear_act_quant();
        pool.defer_act_quant(xf.data(), def_nact.data(), def_act.data(), toks.data(), T, true);

        cpu::ExpertJobMulti wide_job;
        wide_job.blob = native_blob.data();
        wide_job.nt = T;
        for (int t = 0; t < T; ++t) {
            wide_job.nact[t] = def_nact.data() + (size_t) t * cpu::kNativeActBytes;
            wide_job.out[t] = out_wide.data() + (size_t) t * cpu::H;
        }
        pool.run_split_multi_native(f, &wide_job, 1);

        // Compare deferred nact against direct native_quant_act for every row (catching MAXT=8 truncation at t >= 8).
        for (int t = 0; t < T; ++t) {
            if (std::memcmp(def_nact.data() + (size_t) t * cpu::kNativeActBytes,
                            ref_nact.data() + (size_t) t * cpu::kNativeActBytes, cpu::kNativeActBytes) != 0) {
                std::fprintf(stderr, "native defer_act_quant mismatch at row %d/%d\n", t, T);
                ++fail;
            }
        }

        // Compare single T-row job against <=MAXT-chunked jobs bitwise.
        std::vector<cpu::ExpertJobMulti> chunk_jobs;
        for (int t0 = 0; t0 < T; t0 += cpu::MAXT) {
            const int m = (std::min)(cpu::MAXT, T - t0);
            cpu::ExpertJobMulti cj;
            cj.blob = native_blob.data();
            cj.nt = m;
            for (int t = 0; t < m; ++t) {
                cj.nact[t] = ref_nact.data() + (size_t) (t0 + t) * cpu::kNativeActBytes;
                cj.out[t] = out_split.data() + (size_t) (t0 + t) * cpu::H;
            }
            chunk_jobs.push_back(cj);
        }
        pool.clear_act_quant();
        pool.run_split_multi_native(f, chunk_jobs.data(), (int) chunk_jobs.size());

        for (int t = 0; t < T; ++t) {
            const float* a = out_wide.data() + (size_t) t * cpu::H;
            const float* b = out_split.data() + (size_t) t * cpu::H;
            double l2 = 0.0;
            for (int d = 0; d < cpu::H; ++d) {
                if (!std::isfinite(a[d])) {
                    std::fprintf(stderr, "native wide non-finite output at row %d dim %d\n", t, d);
                    ++fail;
                    break;
                }
                l2 += (double) a[d] * a[d];
            }
            if (l2 <= 0.0) {
                std::fprintf(stderr, "native wide output is all zero at row %d\n", t);
                ++fail;
            }
            if (std::memcmp(a, b, (size_t) cpu::H * sizeof(float)) != 0) {
                std::fprintf(stderr, "native wide vs chunked output differs at row %d/%d\n", t, T);
                ++fail;
            }
        }
    }

    // 2. Test s2 path if CPU supports AVX-512 (same check as expert_multi_test): defer_act_quant + run_split_multi.
    if (cpu::cpu_features().usable()) {
        std::vector<cpu::ActQ> def_act(T);
        std::vector<float> out_wide((size_t) T * cpu::H, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> out_split((size_t) T * cpu::H, std::numeric_limits<float>::quiet_NaN());

        pool.clear_act_quant();
        pool.defer_act_quant(xf.data(), nullptr, def_act.data(), toks.data(), T, false);

        cpu::ExpertJobMulti wide_job;
        wide_job.blob = s2_blob.data();
        wide_job.nt = T;
        for (int t = 0; t < T; ++t) {
            wide_job.act[t] = &def_act[t];
            wide_job.out[t] = out_wide.data() + (size_t) t * cpu::H;
        }
        pool.run_split_multi(&wide_job, 1);

        for (int t = 0; t < T; ++t) {
            if (std::memcmp(&def_act[t], &ref_act[t], sizeof(cpu::ActQ)) != 0) {
                std::fprintf(stderr, "s2 defer_act_quant mismatch at row %d/%d\n", t, T);
                ++fail;
            }
        }

        std::vector<cpu::ExpertJobMulti> chunk_jobs;
        for (int t0 = 0; t0 < T; t0 += cpu::MAXT) {
            const int m = (std::min)(cpu::MAXT, T - t0);
            cpu::ExpertJobMulti cj;
            cj.blob = s2_blob.data();
            cj.nt = m;
            for (int t = 0; t < m; ++t) {
                cj.act[t] = &ref_act[t0 + t];
                cj.out[t] = out_split.data() + (size_t) (t0 + t) * cpu::H;
            }
            chunk_jobs.push_back(cj);
        }
        pool.clear_act_quant();
        pool.run_split_multi(chunk_jobs.data(), (int) chunk_jobs.size());

        for (int t = 0; t < T; ++t) {
            const float* a = out_wide.data() + (size_t) t * cpu::H;
            const float* b = out_split.data() + (size_t) t * cpu::H;
            if (std::memcmp(a, b, (size_t) cpu::H * sizeof(float)) != 0) {
                std::fprintf(stderr, "s2 wide vs chunked output differs at row %d/%d\n", t, T);
                ++fail;
            }
        }
    } else {
        std::printf("cpu_act_quant_wide_test: s2 AVX-512 check skipped (%s)\n", cpu::cpu_features().reason());
    }

    if (fail == 0) std::printf("cpu_act_quant_wide_test: ok (T=%d)\n", T);
    return fail ? 1 : 0;
}
