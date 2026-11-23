// src/kernels/gdn_groups_split_parity.cpp - bitwise parity for STRATA_GDN_GROUPS_SPLIT=1 (window + q8_1/q8_il + commit)
// and STRATA_GDN_AB_WIDE=1 (T = 9..16) against per-group / 1-token kernels.
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/window_limits.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

int bad = 0;

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
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
T* up(const std::vector<T>& h) {
    T* d = nullptr;
    ck(cudaMalloc(&d, h.size() * sizeof(T) + 64), "malloc");
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}

template <typename T>
std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}

void report(const char* what, bool ok) {
    std::printf("  %-64s %s\n", what, ok ? "bitwise" : "*** DIFFERS ***");
    if (!ok) ++bad;
}

void gdn_groups_case(const std::vector<int>& ntok, bool commit, bool with_qfuse) {
    const int S = 128, h_k = 16, h_v = 48;
    const int qk = S * h_k, C = 2 * qk + S * h_v, vd = S * h_v;
    const float eps = 1e-6f;
    int T = 0;
    for (int n : ntok) T += n;
    const int G = (int) ntok.size();
    std::mt19937 rng(311 + T * 13 + G + (commit ? 1 : 0) + (with_qfuse ? 7 : 0));
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> h((size_t) T * C), gate((size_t) T * h_v), beta((size_t) T * h_v), z((size_t) T * vd), gamma(S);
    for (auto& v : h) v = nd(rng) * 0.1f;
    for (auto& v : gate) v = -(0.5f + 3.0f * (float) (rng() % 100) / 100.0f);
    for (auto& v : beta) v = (float) (rng() % 100) / 100.0f;
    for (auto& v : z) v = nd(rng);
    for (auto& v : gamma) v = 1.0f + 0.1f * nd(rng);

    const size_t st_n = (size_t) S * h_v * S;
    std::vector<std::vector<float>> st0(G);
    std::vector<float*> st_g(G), st_r(G);
    std::vector<int32_t*> keep(G);
    k::GdnGroups gg;
    gg.n = G;
    int first = 0;
    for (int i = 0; i < G; ++i) {
        st0[i].resize(st_n);
        for (auto& v : st0[i]) v = nd(rng) * 0.1f;
        st_g[i] = up(st0[i]);
        st_r[i] = up(st0[i]);
        const int32_t kp = commit ? ntok[i] : 0;
        keep[i] = up(std::vector<int32_t>{kp});
        gg.state[i] = st_g[i];
        gg.keep[i] = commit ? keep[i] : nullptr;
        gg.first[i] = first;
        gg.ntok[i] = ntok[i];
        first += ntok[i];
    }

    const float* d_h = up(h);
    const float* d_gate = up(gate);
    const float* d_beta = up(beta);
    const float* d_z = up(z);
    const float* d_gm = up(gamma);
    float *d_y = nullptr, *d_yr = nullptr;
    ck(cudaMalloc(&d_y, (size_t) T * vd * 4), "malloc y");
    ck(cudaMalloc(&d_yr, (size_t) T * vd * 4), "malloc yr");
    ck(cudaMemset(d_y, 0, (size_t) T * vd * 4), "memset y");
    ck(cudaMemset(d_yr, 0, (size_t) T * vd * 4), "memset yr");

    const size_t xq_bytes = (size_t) T * (vd / 32) * 36;
    const size_t xq_il_bytes = k::native_q8_1_il_bytes(vd, T);
    uint8_t *d_xq = nullptr, *d_xq_r = nullptr, *d_il = nullptr, *d_il_r = nullptr;
    if (with_qfuse && !commit) {
        ck(cudaMalloc(&d_xq, xq_bytes), "malloc xq");
        ck(cudaMalloc(&d_xq_r, xq_bytes), "malloc xq r");
        ck(cudaMalloc(&d_il, xq_il_bytes), "malloc il");
        ck(cudaMalloc(&d_il_r, xq_il_bytes), "malloc il r");
        ck(cudaMemset(d_xq, 0, xq_bytes), "memset xq");
        ck(cudaMemset(d_xq_r, 0, xq_bytes), "memset xq r");
        ck(cudaMemset(d_il, 0, xq_il_bytes), "memset il");
        ck(cudaMemset(d_il_r, 0, xq_il_bytes), "memset il r");
    }

    cudaStream_t st = nullptr;
    ck(cudaStreamCreate(&st), "stream");
    k::gdn_step_norm_groups(gg, d_h, C, d_gate, d_beta, d_z, d_gm, eps, d_y, h_k, h_v, commit, st, 0, d_xq, d_il);
    for (int i = 0; i < G; ++i) {
        const int f = gg.first[i];
        k::gdn_step_norm_multi(st_r[i], d_h + (size_t) f * C, C, d_gate + (size_t) f * h_v,
                               d_beta + (size_t) f * h_v, d_z + (size_t) f * vd, d_gm, eps,
                               d_yr + (size_t) f * vd, h_k, h_v, ntok[i], commit ? keep[i] : nullptr, st, 0);
    }
    if (with_qfuse && !commit) k::native_quantize_q8_1_il(d_yr, d_xq_r, d_il_r, vd, T, st);
    ck(cudaStreamSynchronize(st), "sync");
    cudaStreamDestroy(st);

    bool same = true;
    for (int i = 0; i < G; ++i) {
        const auto a = down(st_g[i], st_n), b = down(st_r[i], st_n);
        same = same && std::memcmp(a.data(), b.data(), st_n * 4) == 0;
        if (!commit) same = same && std::memcmp(a.data(), st0[i].data(), st_n * 4) == 0;
    }
    if (!commit && !with_qfuse) {
        const auto a = down(d_y, (size_t) T * vd), b = down(d_yr, (size_t) T * vd);
        same = same && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
    }
    if (!commit && with_qfuse) {
        const auto q_a = down(d_xq, xq_bytes), q_b = down(d_xq_r, xq_bytes);
        const auto il_a = down(d_il, xq_il_bytes), il_b = down(d_il_r, xq_il_bytes);
        same = same && std::memcmp(q_a.data(), q_b.data(), xq_bytes) == 0;
        same = same && std::memcmp(il_a.data(), il_b.data(), xq_il_bytes) == 0;
    }

    char name[128], rows[64] = "";
    for (int i = 0; i < G && i < 8; ++i)
        std::snprintf(rows + std::strlen(rows), sizeof rows - std::strlen(rows), "%s%d", i ? "," : "", ntok[i]);
    std::snprintf(name, sizeof name, "gdn_step_norm_groups(split) %s%s [%s]",
                  commit ? "commit" : "window", with_qfuse ? "+qfuse" : "", rows);
    report(name, same);

    for (int i = 0; i < G; ++i) { cudaFree(st_g[i]); cudaFree(st_r[i]); cudaFree(keep[i]); }
    cudaFree(const_cast<float*>(d_h));
    cudaFree(const_cast<float*>(d_gate));
    cudaFree(const_cast<float*>(d_beta));
    cudaFree(const_cast<float*>(d_z));
    cudaFree(const_cast<float*>(d_gm));
    cudaFree(d_y);
    cudaFree(d_yr);
    if (d_xq) { cudaFree(d_xq); cudaFree(d_xq_r); cudaFree(d_il); cudaFree(d_il_r); }
}

void ab_wide_case(int n_tok) {
    const int n_embd = 2560, h_v = 48;
    std::mt19937 rng(900 + n_tok);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) n_tok * n_embd), dt(h_v), ssm_a(h_v);
    std::vector<uint16_t> wa((size_t) h_v * n_embd), wb((size_t) h_v * n_embd);
    for (auto& v : x) v = nd(rng) * 0.5f;
    for (auto& v : dt) v = nd(rng) * 0.1f;
    for (auto& v : ssm_a) v = -(0.2f + std::abs(nd(rng)));
    for (auto& v : wa) v = float_to_bf16(nd(rng) * 0.05f);
    for (auto& v : wb) v = float_to_bf16(nd(rng) * 0.05f);

    float* d_x = up(x);
    uint16_t* d_wa = up(wa);
    uint16_t* d_wb = up(wb);
    float* d_dt = up(dt);
    float* d_sa = up(ssm_a);
    float *d_g = nullptr, *d_b = nullptr, *d_gr = nullptr, *d_br = nullptr;
    const size_t n = (size_t) n_tok * h_v;
    ck(cudaMalloc(&d_g, n * 4), "malloc g");
    ck(cudaMalloc(&d_b, n * 4), "malloc b");
    ck(cudaMalloc(&d_gr, n * 4), "malloc gr");
    ck(cudaMalloc(&d_br, n * 4), "malloc br");

    k::gdn_ab_multi(d_x, d_wa, d_wb, d_dt, d_sa, d_g, d_b, n_embd, h_v, n_tok, nullptr);
    for (int t = 0; t < n_tok; ++t)
        k::gdn_ab_multi(d_x + (size_t) t * n_embd, d_wa, d_wb, d_dt, d_sa, d_gr + (size_t) t * h_v,
                        d_br + (size_t) t * h_v, n_embd, h_v, 1, nullptr);
    ck(cudaDeviceSynchronize(), "sync ab");

    const auto ga = down(d_g, n), gb = down(d_gr, n);
    const auto ba = down(d_b, n), bb = down(d_br, n);
    const bool same = std::memcmp(ga.data(), gb.data(), n * 4) == 0 && std::memcmp(ba.data(), bb.data(), n * 4) == 0;
    char name[96];
    std::snprintf(name, sizeof name, "gdn_ab_multi(STRATA_GDN_AB_WIDE=1) n_tok=%d", n_tok);
    report(name, same);

    cudaFree(d_x); cudaFree(d_wa); cudaFree(d_wb); cudaFree(d_dt); cudaFree(d_sa);
    cudaFree(d_g); cudaFree(d_b); cudaFree(d_gr); cudaFree(d_br);
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    std::printf("gdn_groups_split_parity\n");
    gdn_groups_case({4, 4}, false, false);
    gdn_groups_case({4, 4}, false, true);
    gdn_groups_case({1, 2, 2, 3}, false, false);
    gdn_groups_case({1, 2, 2, 3}, true, false);
    if constexpr (k::kWindowMaxT > 8) {
        gdn_groups_case({4, 5}, false, false);
        gdn_groups_case({4, 5}, false, true);
        gdn_groups_case({2, 3, 4, 5}, false, false);
        gdn_groups_case({2, 3, 4, 5}, false, true);
        gdn_groups_case({1, 2, 3, 4}, true, false);
        for (int t = 9; t <= k::kWindowMaxT; ++t) ab_wide_case(t);
    }
    if constexpr (k::kWindowMaxT >= 16) {
        gdn_groups_case({2, 2, 2, 2, 2, 2, 2, 2}, false, false);
        gdn_groups_case({2, 2, 2, 2, 2, 2, 2, 2}, false, true);
        gdn_groups_case({2, 2, 2, 2, 2, 2, 2, 2}, true, false);
    }
    std::printf("gdn_groups_split_parity: %s\n", bad ? "FAILED" : "ok");
    return bad ? 1 : 0;
}
