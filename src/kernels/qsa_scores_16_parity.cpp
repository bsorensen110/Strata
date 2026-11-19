// src/kernels/qsa_scores_16_parity.cpp - bitwise parity for STRATA_QSA_SCORES_16 (T = 9..16 rows in one launch)
// against single-query qsa_block_scores calls.
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"
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

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
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

}  // namespace

int main() {
    if (k::kWindowMaxT <= 8) {
        std::printf("qsa_scores_16_parity: narrow build (kWindowMaxT=%d), skipped\n", k::kWindowMaxT);
        return 77;
    }

    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t max_blocks = 3000;
    int bad = 0;

    for (int rows : {9, 12, k::kWindowMaxT}) {
        std::mt19937 rng(100 + rows);
        std::normal_distribution<float> nd(0.f, 1.f);
        k::QsaScoreRows sr;
        sr.n = rows;
        std::vector<const float*> pooled(rows), dead(rows);
        std::vector<float> q((size_t) rows * s.idx_n_head * s.idx_dim);
        for (auto& x : q) x = nd(rng);
        std::vector<int32_t> steps((size_t) rows * k::kStepCount);

        for (int r = 0; r < rows; ++r) {
            std::vector<float> p((size_t) max_blocks * s.idx_dim), d((size_t) s.idx_dim);
            for (auto& x : p) x = nd(rng);
            for (auto& x : d) x = nd(rng);
            pooled[r] = up(p);
            dead[r] = up(d);
            sr.pooled[r] = pooled[r];
            sr.dead[r] = dead[r];
            const int64_t pos = 500 + 733 * r + (r % 4) * 7;
            steps[(size_t) r * k::kStepCount + k::kStepPos] = (int32_t) pos;
            steps[(size_t) r * k::kStepCount + k::kStepNKv] = (int32_t) (pos + 1);
            steps[(size_t) r * k::kStepCount + k::kStepNBid] = (int32_t) ((pos + 1) / s.idx_block);
            steps[(size_t) r * k::kStepCount + k::kStepWidth] = (int32_t) k::qsa_selection_width(pos + 1, s);
        }

        const float* d_q = up(q);
        const int32_t* d_steps = up(steps);
        const size_t n = (size_t) rows * max_blocks;
        float *d_rows = nullptr, *d_ref = nullptr;
        ck(cudaMalloc(&d_rows, n * 4), "malloc rows");
        ck(cudaMalloc(&d_ref, n * 4), "malloc ref");
        ck(cudaMemset(d_rows, 0, n * 4), "memset rows");
        ck(cudaMemset(d_ref, 0, n * 4), "memset ref");

        k::qsa_block_scores_rows(sr, d_q, d_steps, rows, max_blocks, s, d_rows, nullptr);
        for (int r = 0; r < rows; ++r) {
            k::qsa_block_scores(pooled[r], dead[r], d_q + (size_t) r * s.idx_n_head * s.idx_dim,
                                d_steps + (size_t) r * k::kStepCount, 1, max_blocks, s,
                                d_ref + (size_t) r * max_blocks, nullptr);
        }
        ck(cudaDeviceSynchronize(), "sync");
        const auto a = down(d_rows, n);
        const auto b = down(d_ref, n);
        const bool same = std::memcmp(a.data(), b.data(), n * 4) == 0;
        std::printf("  qsa_block_scores_rows rows=%d: %s\n", rows, same ? "bitwise" : "DIFFERENT");
        if (!same) ++bad;

        for (int r = 0; r < rows; ++r) {
            cudaFree(const_cast<float*>(pooled[r]));
            cudaFree(const_cast<float*>(dead[r]));
        }
        cudaFree(const_cast<float*>(d_q));
        cudaFree(const_cast<int32_t*>(d_steps));
        cudaFree(d_rows);
        cudaFree(d_ref);
    }

    std::printf("qsa_scores_16_parity: %s\n", bad ? "FAILED" : "ok");
    return bad ? 1 : 0;
}
