// src/kernels/fused_gr_q8_16_parity.cpp - bitwise parity for wide (T = 9..16) Q8_0 hyper-connection reads
// (STRATA_HC_Q8_16=1 single-pass vs STRATA_HC_Q8_16=0 sliced fallback and single-token calls, direct and CUDA graph).
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

uint16_t float_to_bf16(float f) {
    uint32_t i;
    std::memcpy(&i, &f, 4);
    i = (i + ((i >> 16) & 1u) + 0x7FFFu) & 0xFFFF0000u;
    return (uint16_t) (i >> 16);
}

template <typename T>
T* alloc_zero(size_t n) {
    T* p = nullptr;
    check(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    check(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}

template <typename T>
T* up(const std::vector<T>& h) {
    T* p = alloc_zero<T>(h.size());
    check(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "h2d");
    return p;
}

template <typename T>
std::vector<T> down(const T* p, size_t n) {
    std::vector<T> h(n);
    check(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "d2h");
    return h;
}

}  // namespace

int main() {
    if (k::kWindowMaxT <= 8) {
        std::printf("fused_gr_q8_16_parity: narrow build (kWindowMaxT=%d), skipped\n", k::kWindowMaxT);
        return 77;
    }

    constexpr int N = 2560, HC = 4, LR = 320, D = HC * N;
    constexpr int W = k::kFusedGrWideT;
    constexpr float EPS = 1e-6f;
    const char* env_on = std::getenv("STRATA_HC_Q8_16");
    const bool expect_single_pass = env_on == nullptr || env_on[0] == '\0' || std::atoi(env_on) != 0;

    std::mt19937 rng(777);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto gen_bf16 = [&](size_t n, float scale) {
        std::vector<uint16_t> b(n);
        for (auto& x : b) x = float_to_bf16(nd(rng) * scale);
        return b;
    };

    std::vector<float> h_norm(D), h_r((size_t) W * D), h_bo((size_t) W * N), h_inj((size_t) W * HC);
    for (auto& x : h_norm) x = 1.0f + 0.1f * nd(rng);
    for (int t = 0; t < W; ++t) {
        for (int c = 0; c < HC; ++c) {
            const float sc = 1.0f + 0.5f * (float) c;
            for (int d = 0; d < N; ++d) h_r[(size_t) t * D + c * N + d] = nd(rng) * sc;
            h_inj[(size_t) t * HC + c] = nd(rng) * 2.0f;
        }
        for (int d = 0; d < N; ++d) h_bo[(size_t) t * N + d] = nd(rng) * 0.5f;
    }
    const auto b_down = gen_bf16((size_t) LR * D, 0.15f);
    const auto b_up = gen_bf16((size_t) D * LR, 0.15f);
    const auto b_inj = gen_bf16((size_t) HC * D, 0.05f);

    float* d_norm = up(h_norm);
    uint16_t* d_down_bf16 = up(b_down);
    uint16_t* d_up_bf16 = up(b_up);
    uint16_t* d_inj_bf16 = up(b_inj);

    uint8_t* d_q8_down = alloc_zero<uint8_t>(((size_t) LR * D / 32) * 34);
    uint8_t* d_q8_up = alloc_zero<uint8_t>(((size_t) D * LR / 32) * 34);
    uint8_t* d_q8_inj = alloc_zero<uint8_t>(((size_t) HC * D / 32) * 34);
    k::fused_gr_quantize_q8_0(d_down_bf16, d_q8_down, (int64_t) LR * D / 32, nullptr);
    k::fused_gr_quantize_q8_0(d_up_bf16, d_q8_up, (int64_t) D * LR / 32, nullptr);
    k::fused_gr_quantize_q8_0(d_inj_bf16, d_q8_inj, (int64_t) HC * D / 32, nullptr);

    float* d_r = up(h_r);
    float* d_r_out = alloc_zero<float>((size_t) W * D);
    float* d_bo = up(h_bo);
    float* d_inj = up(h_inj);
    float* d_lo = alloc_zero<float>((size_t) W * LR);
    float* d_rs = alloc_zero<float>((size_t) W * HC);
    float* d_inj_out = alloc_zero<float>((size_t) W * HC);
    float* d_mixed = alloc_zero<float>((size_t) W * N);
    float* d_xn = alloc_zero<float>((size_t) W * D);

    const size_t q_stride = (size_t) (N / 32) * 36;
    const size_t il_bytes = k::native_q8_1_il_bytes(N, W);
    uint8_t* d_q = alloc_zero<uint8_t>((size_t) W * q_stride);
    uint8_t* d_q_ref = alloc_zero<uint8_t>((size_t) W * q_stride);
    uint8_t* d_il = alloc_zero<uint8_t>(il_bytes);
    uint8_t* d_il_ref = alloc_zero<uint8_t>(il_bytes);
    unsigned* d_cnt = alloc_zero<unsigned>(N / 32);

    std::vector<k::FusedGrArgs> args(W);
    for (int t = 0; t < W; ++t) {
        auto& a = args[t];
        a.R = d_r + (size_t) t * D;
        a.R_out = d_r_out + (size_t) t * D;
        a.apply = true;
        a.bo_prev = d_bo + (size_t) t * N;
        a.inj_prev = d_inj + (size_t) t * HC;
        a.w_norm = d_norm;
        a.w_down = d_down_bf16;
        a.w_up = d_up_bf16;
        a.w_inject = d_inj_bf16;
        a.q8_down = d_q8_down;
        a.q8_up = d_q8_up;
        a.q8_inject = d_q8_inj;
        a.eps = EPS;
        a.lo = d_lo + (size_t) t * LR;
        a.rs = d_rs + (size_t) t * HC;
        a.inject_out = d_inj_out + (size_t) t * HC;
        a.mixed = d_mixed + (size_t) t * N;
        a.q8_mixed = d_q + (size_t) t * q_stride;
        a.q8_cnt = d_cnt;
    }

    struct Snap {
        std::vector<float> r_out, lo, rs, inj_out, mixed;
    };
    auto snap = [&](int T) {
        Snap s;
        s.r_out = down(d_r_out, (size_t) T * D);
        s.lo = down(d_lo, (size_t) T * LR);
        s.rs = down(d_rs, (size_t) T * HC);
        s.inj_out = down(d_inj_out, (size_t) T * HC);
        s.mixed = down(d_mixed, (size_t) T * N);
        return s;
    };
    auto same_snap = [](const Snap& a, const Snap& b) {
        return std::memcmp(a.r_out.data(), b.r_out.data(), a.r_out.size() * 4) == 0 &&
               std::memcmp(a.lo.data(), b.lo.data(), a.lo.size() * 4) == 0 &&
               std::memcmp(a.rs.data(), b.rs.data(), a.rs.size() * 4) == 0 &&
               std::memcmp(a.inj_out.data(), b.inj_out.data(), a.inj_out.size() * 4) == 0 &&
               std::memcmp(a.mixed.data(), b.mixed.data(), a.mixed.size() * 4) == 0;
    };

    cudaStream_t st = nullptr;
    check(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    check(cudaDeviceSynchronize(), "sync init");

    int bad = 0;
    for (int T = k::kFusedGrMaxT + 1; T <= W; ++T) {
        for (int pass = 0; pass < 2; ++pass) {
            const bool use_graph = pass == 1;
            if (use_graph && T != 9 && T != W) continue;
            std::vector<k::FusedGrArgs> qa(args.begin(), args.begin() + T);
            qa[0].q8_il = d_il;
            const size_t tqb = (size_t) T * q_stride;
            const size_t tib = k::native_q8_1_il_bytes(N, T);
            check(cudaMemsetAsync(d_r_out, 0xff, (size_t) T * D * 4, st), "poison R_out");
            check(cudaMemsetAsync(d_q, 0x5a, tqb, st), "poison q8");
            check(cudaMemsetAsync(d_il, 0, tib, st), "zero il");
            check(cudaMemsetAsync(d_il_ref, 0, tib, st), "zero il ref");

            bool wrote_q8 = false, wrote_il = false;
            if (use_graph) {
                cudaGraph_t g = nullptr;
                cudaGraphExec_t gx = nullptr;
                check(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "cap begin");
                wrote_q8 = k::fused_gr_read_multi(qa.data(), T, d_xn, st, nullptr, 0, &wrote_il);
                check(cudaStreamEndCapture(st, &g), "cap end");
                check(cudaGraphInstantiate(&gx, g, nullptr, nullptr, 0), "inst");
                check(cudaGraphLaunch(gx, st), "launch 1");
                check(cudaStreamSynchronize(st), "sync g1");
                for (size_t i = 0; i < (size_t) T * D; ++i) h_r[i] = -0.7f * h_r[i] + 0.001f * (float) (i % 17);
                check(cudaMemcpyAsync(d_r, h_r.data(), h_r.size() * 4, cudaMemcpyHostToDevice, st), "replay R");
                check(cudaMemsetAsync(d_q, 0x5a, tqb, st), "replay poison q8");
                check(cudaMemsetAsync(d_il, 0, tib, st), "replay zero il");
                check(cudaGraphLaunch(gx, st), "launch replay");
                check(cudaStreamSynchronize(st), "sync g2");
                check(cudaGraphExecDestroy(gx), "destroy gx");
                check(cudaGraphDestroy(g), "destroy g");
            } else {
                wrote_q8 = k::fused_gr_read_multi(qa.data(), T, d_xn, st, nullptr, 0, &wrote_il);
                check(cudaStreamSynchronize(st), "sync direct");
            }

            if (!wrote_q8) {
                std::fprintf(stderr, "T=%d (%s): wrote_q8=false\n", T, use_graph ? "graph" : "direct");
                ++bad;
            }
            if (wrote_il != expect_single_pass) {
                std::fprintf(stderr, "T=%d (%s): wrote_il=%d expected %d\n", T, use_graph ? "graph" : "direct",
                             (int) wrote_il, (int) expect_single_pass);
                ++bad;
            }
            const Snap multi = snap(T);
            const auto hq = down(d_q, tqb);
            const auto hil = down(d_il, tib);

            // Reference: T single-token Q8_0 calls + native_quantize_q8_1_il on multi's mixed output.
            for (int t = 0; t < T; ++t) {
                k::FusedGrArgs one = qa[t];
                one.q8_mixed = nullptr;
                one.q8_cnt = nullptr;
                one.q8_il = nullptr;
                k::fused_gr_read_multi(&one, 1, d_xn, st);
            }
            check(cudaStreamSynchronize(st), "sync single");
            const Snap ref = snap(T);
            if (!same_snap(multi, ref)) {
                std::fprintf(stderr, "T=%d (%s, single_pass=%d): values differ from 1-token Q8_0 calls\n",
                             T, use_graph ? "graph" : "direct", (int) expect_single_pass);
                ++bad;
            }

            // Use multi.mixed (same bytes as ref.mixed) to generate reference q8_mixed + q8_il.
            k::native_quantize_q8_1_il(d_mixed, d_q_ref, d_il_ref, N, T, st);
            check(cudaStreamSynchronize(st), "sync quant ref");
            const auto hq_ref = down(d_q_ref, tqb);
            const auto hil_ref = down(d_il_ref, tib);
            if (std::memcmp(hq.data(), hq_ref.data(), tqb) != 0) {
                std::fprintf(stderr, "T=%d (%s, single_pass=%d): q8_mixed differs from native_quantize_q8_1\n",
                             T, use_graph ? "graph" : "direct", (int) expect_single_pass);
                ++bad;
            }
            if (expect_single_pass && std::memcmp(hil.data(), hil_ref.data(), tib) != 0) {
                std::fprintf(stderr, "T=%d (%s): q8_il differs from native_quantize_q8_1_il\n",
                             T, use_graph ? "graph" : "direct");
                ++bad;
            }
        }
    }

    std::printf("fused_gr_q8_16_parity (STRATA_HC_Q8_16=%d): %s\n",
                (int) expect_single_pass, bad == 0 ? "ok" : "FAILED");
    return bad ? 1 : 0;
}
