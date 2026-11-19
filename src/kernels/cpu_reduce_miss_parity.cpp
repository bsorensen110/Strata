// src/kernels/cpu_reduce_miss_parity.cpp - GPU parity for STRATA_CPU_REDUCE_MISS:
// doorbell_publish_res(..., w, w_out) + host weighted CPU-miss reduction + native_moe_combine_multi_hits_mapped
// against standard full parts + native_moe_combine_multi across T = 1..16 (all-hit, mixed hit/miss, all-miss, gated/ungated).
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/window_limits.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA %s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}

template <typename T>
T* upload(const std::vector<T>& h) {
    T* d = dalloc<T>(h.size());
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "h2d");
    return d;
}

template <typename T>
std::vector<T> download(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "d2h");
    return h;
}

}  // namespace

int main() {
    constexpr int N = 2560, K = 10, NE = 64;
    const int max_t = k::kWindowMaxT;

    cudaStream_t st = nullptr;
    ck(cudaStreamCreate(&st), "stream");

    // Mapped host buffers for doorbell + flag + reduced misses
    float *h_x = nullptr, *m_x = nullptr, *h_w = nullptr, *m_w = nullptr, *h_miss = nullptr, *m_miss = nullptr;
    int32_t *h_ids = nullptr, *m_ids = nullptr;
    uint32_t *h_seq = nullptr, *m_seq = nullptr, *h_flag = nullptr, *m_flag = nullptr;
    auto alloc_map = [&](size_t bytes, auto** hp, auto** dp) {
        ck(cudaHostAlloc((void**) hp, bytes + 64, cudaHostAllocMapped), "hostAlloc");
        std::memset(*hp, 0, bytes + 64);
        ck(cudaHostGetDevicePointer((void**) dp, *hp, 0), "getDevPtr");
    };
    alloc_map((size_t) max_t * N * 4, &h_x, &m_x);
    alloc_map((size_t) max_t * K * 4, &h_w, &m_w);
    alloc_map((size_t) max_t * N * 4, &h_miss, &m_miss);
    alloc_map((size_t) max_t * K * 4, &h_ids, &m_ids);
    alloc_map(64, &h_seq, &m_seq);
    alloc_map(64, &h_flag, &m_flag);

    std::mt19937 rng(2026);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    int bad = 0;
    for (int T : {1, 4, 8, max_t}) {
        for (int pattern = 0; pattern < 3; ++pattern) {
            // 0 = all hits (0 misses), 1 = mixed per token, 2 = all misses
            for (bool gated : {false, true}) {
                std::vector<int32_t> h_res(NE, -1);
                if (pattern == 0) {
                    for (int e = 0; e < NE; ++e) h_res[e] = e;
                } else if (pattern == 1) {
                    for (int e = 0; e < NE; ++e) h_res[e] = (e % 3 == 0) ? -1 : e;
                }
                std::vector<float> x_in((size_t) T * N), w_in((size_t) T * K), sh((size_t) T * N), shg(T);
                std::vector<int32_t> ids_in((size_t) T * K);
                for (auto& v : x_in) v = nd(rng);
                for (auto& v : w_in) v = 0.05f + 0.1f * std::abs(nd(rng));
                for (auto& v : sh) v = nd(rng) * 0.5f;
                for (auto& v : shg) v = nd(rng) * 1.5f;
                for (int t = 0; t < T; ++t) {
                    for (int j = 0; j < K; ++j) {
                        int eid = (t * 7 + j * 5) % NE;
                        if (pattern == 1 && t % 2 == 0) eid = 3 * ((j + t) % (NE / 3));   // even tokens 100% miss!
                        ids_in[(size_t) t * K + j] = eid;
                    }
                }

                // Synthesize expert outputs for all [T, K] entries
                std::vector<float> all_exp((size_t) T * K * N);
                for (auto& v : all_exp) v = nd(rng) * 0.25f;

                std::vector<float> parts_ref((size_t) T * K * N), hits_dev((size_t) T * K * N, 0.0f);
                std::vector<int32_t> dst;
                std::vector<bool> is_hit((size_t) T * K, false);
                for (int t = 0; t < T; ++t) {
                    bool any_m = false;
                    for (int j = 0; j < K; ++j) {
                        const int idx = t * K + j;
                        const bool hit = h_res[ids_in[idx]] >= 0;
                        is_hit[idx] = hit;
                        const float* src = all_exp.data() + (size_t) idx * N;
                        std::memcpy(parts_ref.data() + (size_t) idx * N, src, (size_t) N * 4);
                        if (hit) {
                            std::memcpy(hits_dev.data() + (size_t) idx * N, src, (size_t) N * 4);
                            dst.push_back(idx);
                        } else {
                            any_m = true;
                        }
                    }
                    // Host miss reduction (what expert_source computes)
                    float* mr = h_miss + (size_t) t * N;
                    if (!any_m) {
                        std::memset(mr, 0, (size_t) N * 4);
                    } else {
                        bool first = true;
                        for (int j = 0; j < K; ++j) {
                            const int idx = t * K + j;
                            if (is_hit[idx]) continue;
                            const float wt = w_in[idx];
                            const float* src = all_exp.data() + (size_t) idx * N;
                            if (first) {
                                for (int d = 0; d < N; ++d) mr[d] = src[d] * wt;
                                first = false;
                            } else {
                                for (int d = 0; d < N; ++d) mr[d] += src[d] * wt;
                            }
                        }
                    }
                }

                float *d_x = upload(x_in), *d_w = upload(w_in), *d_sh = upload(sh), *d_shg = upload(shg);
                float *d_hits = upload(hits_dev), *d_parts = upload(parts_ref);
                float *d_out = dalloc<float>((size_t) T * N), *d_ref = dalloc<float>((size_t) T * N);
                int32_t *d_ids = upload(ids_in), *d_res = upload(h_res);
                int32_t* d_dst = upload(dst.empty() ? std::vector<int32_t>{0} : dst);
                int32_t* d_cnt = upload(std::vector<int32_t>{(int32_t) dst.size()});

                *h_seq = 0;
                *h_flag = 1;
                std::memset(h_w, 0, (size_t) T * K * 4);
                k::doorbell_publish_res(d_x, d_ids, d_res, NE, (int64_t) T * N, (int64_t) T * K,
                                        m_x, m_ids, m_seq, st, d_w, m_w);
                k::native_moe_combine_multi_hits_mapped(d_hits, m_miss, m_flag, 1u, d_dst, d_cnt, d_w,
                                                        d_sh, gated ? d_shg : nullptr, d_out, N, K, T, st);

                // Reference: scale shared if gated, then native_moe_combine_multi
                std::vector<float> sh_ref = sh;
                if (gated) {
                    for (int t = 0; t < T; ++t) {
                        const float g = 1.0f / (1.0f + std::exp(-shg[t]));
                        for (int d = 0; d < N; ++d) sh_ref[(size_t) t * N + d] *= g;
                    }
                }
                float* d_sh_ref = upload(sh_ref);
                ck(cudaDeviceSynchronize(), "sync sh_ref");
                k::native_moe_combine_multi(d_parts, d_w, d_sh_ref, d_ref, N, K, T, st);
                ck(cudaStreamSynchronize(st), "sync");

                // Check mapped weights copied by doorbell_publish_res
                const bool w_ok = std::memcmp(h_w, w_in.data(), (size_t) T * K * 4) == 0;
                const auto got = download(d_out, (size_t) T * N);
                const auto ref = download(d_ref, (size_t) T * N);
                double worst = 0.0, mag = 1e-30;
                for (size_t i = 0; i < got.size(); ++i) {
                    worst = std::max(worst, (double) std::fabs(got[i] - ref[i]));
                    mag = std::max(mag, (double) std::fabs(ref[i]));
                }
                const double rel = worst / mag;
                const bool bit_eq = std::memcmp(got.data(), ref.data(), got.size() * 4) == 0;
                const bool ok = w_ok && (pattern == 0 && !gated ? bit_eq : rel < 2e-5);
                if (!ok) {
                    size_t first_bad = 0;
                    for (size_t i = 0; i < got.size(); ++i) {
                        if (std::fabs(got[i] - ref[i]) > 1e-4f) { first_bad = i; break; }
                    }
                    const size_t t0 = first_bad / N, d0 = first_bad % N;
                    double man = sh[t0 * N + d0];
                    if (gated) man = sh[t0 * N + d0] * (1.0 / (1.0 + std::exp(-(double) shg[t0])));
                    for (int j = 0; j < K; ++j) man += (double) all_exp[(t0 * K + j) * N + d0] * (double) w_in[t0 * K + j];
                    std::fprintf(stderr, "FAIL T=%d pattern=%d gated=%d: w_ok=%d bit=%d rel=%.3e first=(t=%zu d=%zu got=%.6g ref=%.6g man=%.6g nh=%zu)\n",
                                 T, pattern, (int) gated, (int) w_ok, (int) bit_eq, rel,
                                 t0, d0, got[first_bad], ref[first_bad], man, dst.size());
                    ++bad;
                }

                cudaFree(d_x); cudaFree(d_w); cudaFree(d_sh); cudaFree(d_shg);
                cudaFree(d_hits); cudaFree(d_parts); cudaFree(d_out); cudaFree(d_ref);
                cudaFree(d_ids); cudaFree(d_res); cudaFree(d_dst); cudaFree(d_cnt); cudaFree(d_sh_ref);
            }
        }
    }

    cudaFreeHost(h_x); cudaFreeHost(h_w); cudaFreeHost(h_miss); cudaFreeHost(h_ids);
    cudaFreeHost(h_seq); cudaFreeHost(h_flag);
    cudaStreamDestroy(st);

    std::printf("cpu_reduce_miss_parity: %s\n", bad ? "FAILED" : "ok");
    return bad ? 1 : 0;
}
