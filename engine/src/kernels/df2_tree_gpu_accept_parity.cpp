// src/kernels/df2_tree_gpu_accept_parity.cpp - P7 GPU parity test for launch_df2_tree_sample_accept against host
// df2_tree_sample_accept across >= 10,000 random trees, distributions, and stochastic/point-mass proposals.
// Exits 0 with a SKIP marker when STRATA_TEST_NO_GPU=1 or no GPU context is available.
#include "strata/core/spec_prob.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/spec/df2_tree.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "df2_tree_gpu_accept_parity: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct CaseData {
    strata::spec::Df2TreeSpec tree;
    std::array<strata::core::SpecPRow, strata::spec::kDf2TreeMaxNodes> probs{};
    strata::spec::Df2TreeSampleQ q_nodes;
    bool has_q = false;
    uint64_t seed = 0;
    uint64_t pos0 = 0;
};

CaseData make_random_case(std::mt19937_64& rng, int idx) {
    CaseData c;
    c.seed = rng() ^ 0x9E3779B97F4A7C15ULL;
    c.pos0 = (idx % 17 == 0) ? ((rng() & ((1ULL << 40) - 1)) | (1ULL << 39)) : (rng() % 50000ULL);

    // Build valid caterpillar/fanout/linear tree of 2..8 nodes with unique tokens and unique siblings.
    const int shape_kind = idx % 4;
    const int32_t anchor = static_cast<int32_t>(100 + (rng() % 5000));
    std::vector<int32_t> pool;
    pool.reserve(32);
    for (int i = 0; i < 32; ++i) pool.push_back(anchor + 1 + i);
    std::shuffle(pool.begin(), pool.end(), rng);

    if (shape_kind == 0) {
        const int nd = (idx % 16 == 0) ? 7 : (1 + static_cast<int>(rng() % 7));
        c.tree = strata::spec::make_linear_tree(anchor, pool.data(), nd);
    } else if (shape_kind == 1) {
        const int prefix_len = static_cast<int>(rng() % 4);
        const int fanout = 2 + static_cast<int>(rng() % 3);
        const int tail_len = static_cast<int>(rng() % 4);
        c.tree = strata::spec::make_prefix_fanout_tree(anchor, pool.data(), prefix_len, pool.data() + 4, fanout,
                                                       pool.data() + 10, tail_len, 8);
    } else if (shape_kind == 2) {
        const int spine_len = 2 + static_cast<int>(rng() % 4);
        uint8_t mask[7] = {};
        for (int s = 0; s < spine_len; ++s) mask[s] = (rng() % 2 == 0) ? 1 : 0;
        mask[rng() % spine_len] = 1;
        c.tree = strata::spec::make_caterpillar_tree(anchor, pool.data(), spine_len, pool.data() + 8, mask);
    } else {
        const int prefix_len = static_cast<int>(rng() % 3);
        c.tree = strata::spec::make_prefix_fanout_tree(anchor, pool.data(), prefix_len, pool.data() + 4, 2,
                                                       pool.data() + 10, 2, 8);
    }
    if (!c.tree.valid() || c.tree.count < 2) {
        c.tree = strata::spec::make_linear_tree(anchor, pool.data(), 3);
    }

    c.has_q = (idx % 2 == 1);
    const bool zero_resid_case = c.has_q && (idx % 23 == 1);

    std::uniform_real_distribution<double> u_logit(-4.0, 4.0);
    for (int u = 0; u < c.tree.count; ++u) {
        auto& row = c.probs[static_cast<size_t>(u)];
        const int n_p = (idx % 19 == 0) ? 1 : ((idx % 13 == 0) ? 64 : (2 + static_cast<int>(rng() % 63)));
        row.n = n_p;

        // Include children's tokens early in P_u most of the time, sometimes omit to exercise pd == 0.
        std::vector<int32_t> ids;
        ids.reserve(static_cast<size_t>(n_p));
        for (int ch = u + 1; ch < c.tree.count && static_cast<int>(ids.size()) < n_p; ++ch) {
            if (c.tree.parent[static_cast<size_t>(ch)] == u && (rng() % 5 != 0 || zero_resid_case)) {
                ids.push_back(c.tree.token[static_cast<size_t>(ch)]);
            }
        }
        int32_t next_tok = 9000;
        while (static_cast<int>(ids.size()) < n_p) {
            int32_t cand = (rng() % 3 == 0) ? pool[static_cast<size_t>(rng() % pool.size())] : next_tok++;
            if (std::find(ids.begin(), ids.end(), cand) == ids.end()) ids.push_back(cand);
        }

        double sum = 0.0;
        for (int i = 0; i < n_p; ++i) {
            row.id[i] = ids[static_cast<size_t>(i)];
            const double w = (idx % 11 == 0) ? 1.0 : std::exp(u_logit(rng));
            row.p[i] = w;
            sum += w;
        }
        for (int i = 0; i < n_p; ++i) row.p[i] /= sum;
        for (int i = n_p; i < strata::core::kSpecQEntries; ++i) {
            row.id[i] = -1;
            row.p[i] = 0.0;
        }
    }

    if (c.has_q) {
        for (int ch = 1; ch < c.tree.count; ++ch) {
            const int p_node = c.tree.parent[static_cast<size_t>(ch)];
            const auto& parent_p = c.probs[static_cast<size_t>(p_node)];
            auto& nq = c.q_nodes.node[static_cast<size_t>(ch)];
            if (zero_resid_case) {
                // Force q >= p everywhere on parent_p's support (and cover ch token) so next_z <= 0 fires.
                nq.n = std::min(parent_p.n, strata::core::kSpecQEntries);
                for (int i = 0; i < nq.n; ++i) {
                    nq.id[static_cast<size_t>(i)] = parent_p.id[i];
                    nq.q[static_cast<size_t>(i)] = static_cast<float>(std::min(1.0, parent_p.p[i] + 0.05));
                }
                continue;
            }
            const int q_n = 1 + static_cast<int>(rng() % 15);
            nq.n = q_n;
            float qsum = 0.0f;
            for (int j = 0; j < q_n; ++j) {
                if (j == 0 && rng() % 4 != 0) {
                    nq.id[static_cast<size_t>(j)] = c.tree.token[static_cast<size_t>(ch)];
                } else if (j < parent_p.n && rng() % 2 == 0) {
                    nq.id[static_cast<size_t>(j)] = parent_p.id[j];  // deliberately allows duplicate Q.id entries
                } else {
                    nq.id[static_cast<size_t>(j)] = static_cast<int32_t>(7000 + (rng() % 50));
                }
                const float w = static_cast<float>(std::exp(u_logit(rng) * 0.5));
                nq.q[static_cast<size_t>(j)] = w;
                qsum += w;
            }
            for (int j = 0; j < q_n; ++j) nq.q[static_cast<size_t>(j)] /= qsum;
        }
    }
    return c;
}

bool compare_case(int idx, const CaseData& c, const strata::core::TreeSampleAcceptResult& gpu_res) {
    std::array<strata::core::SpecP, strata::spec::kDf2TreeMaxNodes> p_views{};
    for (int u = 0; u < c.tree.count; ++u) {
        p_views[static_cast<size_t>(u)] = strata::core::SpecP{
            c.probs[static_cast<size_t>(u)].id, c.probs[static_cast<size_t>(u)].p, c.probs[static_cast<size_t>(u)].n};
    }
    int32_t host_bonus = 0;
    int64_t host_resamp = 0;
    const strata::spec::Df2TreeCommitWalk host_walk = strata::spec::df2_tree_sample_accept(
        c.tree, p_views.data(), c.has_q ? &c.q_nodes : nullptr, c.seed, c.pos0, &host_bonus, &host_resamp);

    const bool gpu_branch_hit = (gpu_res.flags & 1u) != 0;
    const bool gpu_runner_hit = (gpu_res.flags & 2u) != 0;
    bool ok = static_cast<int>(gpu_res.n_keep) == host_walk.n_keep &&
              gpu_res.bonus_token == host_bonus &&
              static_cast<int64_t>(gpu_res.resamples) == host_resamp &&
              gpu_branch_hit == host_walk.branch_hit &&
              gpu_runner_hit == host_walk.runner_up_hit;
    for (int k = 0; ok && k < host_walk.n_keep; ++k) {
        if (gpu_res.path[k] != host_walk.path[static_cast<size_t>(k)]) ok = false;
    }
    if (!ok) {
        std::fprintf(stderr,
                     "MISMATCH at case %d (has_q=%d count=%d seed=0x%llx pos0=%llu): "
                     "host(n=%d bonus=%d resamp=%lld b=%d r=%d) vs gpu(n=%u bonus=%d resamp=%u b=%d r=%d)\n",
                     idx, c.has_q ? 1 : 0, c.tree.count, static_cast<unsigned long long>(c.seed),
                     static_cast<unsigned long long>(c.pos0), host_walk.n_keep, host_bonus,
                     static_cast<long long>(host_resamp), host_walk.branch_hit ? 1 : 0,
                     host_walk.runner_up_hit ? 1 : 0, static_cast<unsigned>(gpu_res.n_keep),
                     gpu_res.bonus_token, static_cast<unsigned>(gpu_res.resamples),
                     gpu_branch_hit ? 1 : 0, gpu_runner_hit ? 1 : 0);
    }
    return ok;
}

}  // namespace

int main() {
    const char* no_gpu = std::getenv("STRATA_TEST_NO_GPU");
    if (no_gpu != nullptr && std::strcmp(no_gpu, "1") == 0) {
        std::printf("SKIP: df2_tree_gpu_accept_parity: STRATA_TEST_NO_GPU=1\n");
        return 0;
    }
    int dev_count = 0;
    const cudaError_t dc = cudaGetDeviceCount(&dev_count);
    if (dc != cudaSuccess || dev_count <= 0) {
        std::printf("SKIP: df2_tree_gpu_accept_parity: no GPU device available\n");
        return 0;
    }

    constexpr int kCases = 12000;
    constexpr int kBatch = 256;
    std::mt19937_64 rng(0xDF27BEEFULL);

    std::vector<CaseData> cases;
    cases.reserve(kCases);
    for (int i = 0; i < kCases; ++i) cases.push_back(make_random_case(rng, i));

    strata::core::SpecPRow* d_probs = nullptr;
    strata::core::TreeSampleAcceptDesc* d_descs = nullptr;
    strata::core::TreeSampleAcceptQ* d_qs = nullptr;
    strata::core::TreeSampleAcceptResult* d_res = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_probs),
                          static_cast<size_t>(kBatch) * strata::spec::kDf2TreeMaxNodes * sizeof(strata::core::SpecPRow)),
               "cudaMalloc d_probs");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_descs),
                          static_cast<size_t>(kBatch) * sizeof(strata::core::TreeSampleAcceptDesc)),
               "cudaMalloc d_descs");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_qs),
                          static_cast<size_t>(kBatch) * sizeof(strata::core::TreeSampleAcceptQ)),
               "cudaMalloc d_qs");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_res),
                          static_cast<size_t>(kBatch) * sizeof(strata::core::TreeSampleAcceptResult)),
               "cudaMalloc d_res");

    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    std::vector<strata::core::SpecPRow> h_probs(static_cast<size_t>(kBatch) * strata::spec::kDf2TreeMaxNodes);
    std::vector<strata::core::TreeSampleAcceptDesc> h_descs(static_cast<size_t>(kBatch));
    std::vector<strata::core::TreeSampleAcceptQ> h_qs(static_cast<size_t>(kBatch));
    std::vector<strata::core::TreeSampleAcceptResult> h_res(static_cast<size_t>(kBatch));

    for (int base = 0; base < kCases; base += kBatch) {
        const int n = std::min(kBatch, kCases - base);
        for (int b = 0; b < n; ++b) {
            const CaseData& c = cases[static_cast<size_t>(base + b)];
            for (int u = 0; u < strata::spec::kDf2TreeMaxNodes; ++u) {
                h_probs[static_cast<size_t>(b) * strata::spec::kDf2TreeMaxNodes + static_cast<size_t>(u)] =
                    c.probs[static_cast<size_t>(u)];
            }
            strata::spec::df2_tree_pack_gpu_accept_input(
                c.tree, c.has_q ? &c.q_nodes : nullptr, c.seed, c.pos0,
                h_descs[static_cast<size_t>(b)], &h_qs[static_cast<size_t>(b)]);
        }
        check_cuda(cudaMemcpyAsync(d_probs, h_probs.data(),
                                   static_cast<size_t>(n) * strata::spec::kDf2TreeMaxNodes * sizeof(strata::core::SpecPRow),
                                   cudaMemcpyHostToDevice, stream),
                   "H2D probs");
        check_cuda(cudaMemcpyAsync(d_descs, h_descs.data(),
                                   static_cast<size_t>(n) * sizeof(strata::core::TreeSampleAcceptDesc),
                                   cudaMemcpyHostToDevice, stream),
                   "H2D descs");
        check_cuda(cudaMemcpyAsync(d_qs, h_qs.data(),
                                   static_cast<size_t>(n) * sizeof(strata::core::TreeSampleAcceptQ),
                                   cudaMemcpyHostToDevice, stream),
                   "H2D qs");

        for (int b = 0; b < n; ++b) {
            const CaseData& c = cases[static_cast<size_t>(base + b)];
            if (!strata::kernels::launch_df2_tree_sample_accept(
                    d_probs + static_cast<size_t>(b) * strata::spec::kDf2TreeMaxNodes,
                    d_descs + b, c.has_q ? (d_qs + b) : nullptr, d_res + b, stream)) {
                std::fprintf(stderr, "df2_tree_gpu_accept_parity: launch failed at case %d\n", base + b);
                return 1;
            }
        }
        check_cuda(cudaMemcpyAsync(h_res.data(), d_res,
                                   static_cast<size_t>(n) * sizeof(strata::core::TreeSampleAcceptResult),
                                   cudaMemcpyDeviceToHost, stream),
                   "D2H res");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize batch");
        for (int b = 0; b < n; ++b) {
            if (!compare_case(base + b, cases[static_cast<size_t>(base + b)], h_res[static_cast<size_t>(b)])) {
                return 1;
            }
        }
    }

    // Mapped-pinned staging path matching Verifier::run_tree.
    constexpr int kMappedCases = 64;
    strata::core::TreeSampleAcceptDesc* h_mdesc = nullptr;
    strata::core::TreeSampleAcceptDesc* m_mdesc = nullptr;
    strata::core::TreeSampleAcceptQ* h_mq = nullptr;
    strata::core::TreeSampleAcceptQ* m_mq = nullptr;
    strata::core::TreeSampleAcceptResult* h_mout = nullptr;
    strata::core::TreeSampleAcceptResult* m_mout = nullptr;
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&h_mdesc), sizeof(*h_mdesc), cudaHostAllocMapped),
               "alloc mapped desc");
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&m_mdesc), h_mdesc, 0), "devptr desc");
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&h_mq), sizeof(*h_mq), cudaHostAllocMapped),
               "alloc mapped q");
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&m_mq), h_mq, 0), "devptr q");
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&h_mout), sizeof(*h_mout), cudaHostAllocMapped),
               "alloc mapped out");
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&m_mout), h_mout, 0), "devptr out");

    for (int i = 0; i < kMappedCases; ++i) {
        const CaseData& c = cases[static_cast<size_t>(i)];
        check_cuda(cudaMemcpyAsync(d_probs, c.probs.data(),
                                   strata::spec::kDf2TreeMaxNodes * sizeof(strata::core::SpecPRow),
                                   cudaMemcpyHostToDevice, stream),
                   "H2D mapped probs");
        strata::spec::df2_tree_pack_gpu_accept_input(
            c.tree, c.has_q ? &c.q_nodes : nullptr, c.seed, c.pos0, *h_mdesc, c.has_q ? h_mq : nullptr);
        std::memset(h_mout, 0, sizeof(*h_mout));
        if (!strata::kernels::launch_df2_tree_sample_accept(
                d_probs, m_mdesc, c.has_q ? m_mq : nullptr, m_mout, stream)) {
            std::fprintf(stderr, "df2_tree_gpu_accept_parity: mapped launch failed at %d\n", i);
            return 1;
        }
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize mapped");
        if (!compare_case(i, c, *h_mout)) return 1;
    }

    check_cuda(cudaFreeHost(h_mdesc), "free mapped desc");
    check_cuda(cudaFreeHost(h_mq), "free mapped q");
    check_cuda(cudaFreeHost(h_mout), "free mapped out");
    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    check_cuda(cudaFree(d_probs), "cudaFree d_probs");
    check_cuda(cudaFree(d_descs), "cudaFree d_descs");
    check_cuda(cudaFree(d_qs), "cudaFree d_qs");
    check_cuda(cudaFree(d_res), "cudaFree d_res");

    std::printf("df2_tree_gpu_accept_parity: OK (%d batched + %d mapped exact matches)\n",
                kCases, kMappedCases);
    return 0;
}
