#include "strata/spec/df2_sample_walk.hpp"
#include "strata/spec/df2_tree.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}
}

int main() {
    constexpr int trials = 500000;
    int32_t candidates[strata::spec::kDf2SelectorTopK];
    float scores[strata::spec::kDf2SelectorTopK * strata::spec::kDf2SelectorTopK];
    int32_t drafts[1] = {};
    float selected_q[1] = {};
    int32_t q_tokens[strata::spec::kDf2SelectorTopK] = {};
    float q_probs[strata::spec::kDf2SelectorTopK] = {};
    for (int j = 0; j < strata::spec::kDf2SelectorTopK; ++j) {
        candidates[j] = j < 2 ? j : -1;
        for (int prev = 0; prev < strata::spec::kDf2SelectorTopK; ++prev)
            scores[prev * strata::spec::kDf2SelectorTopK + j] = -INFINITY;
    }
    constexpr float q0 = 0.7f, q1 = 0.3f;
    scores[0] = std::log(q0);
    scores[1] = std::log(q1);
    const int32_t target_ids[] = {0, 1};
    const double target_probs[] = {0.2, 0.8};
    const strata::core::SpecP target{target_ids, target_probs, 2};
    std::mt19937_64 rng(20261119);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    long long q_count[2] = {}, p_count[2] = {};

    for (int t = 0; t < trials; ++t) {
        if (!strata::spec::df2_sample_selector_walk(1, candidates, scores, 0xDFA5u, (uint64_t) t,
                                                    drafts, selected_q, q_tokens, q_probs)) {
            check(false, "selector walk accepts a valid finite two-candidate row");
            break;
        }
        const int draft = drafts[0];
        if (draft < 0 || draft > 1) { ++failures; continue; }
        ++q_count[draft];
        strata::core::SpecQ q{q_tokens, q_probs, 2};
        const auto result = strata::core::spec_verify_row(target, draft, q, uniform(rng), uniform(rng));
        ++p_count[result.token];
    }

    auto chi2 = [](long long a, long long b, double p, int n) {
        const double ea = p * n, eb = (1.0 - p) * n;
        return (a - ea) * (a - ea) / ea + (b - eb) * (b - eb) / eb;
    };
    check(chi2(q_count[0], q_count[1], q0, trials) < 10.83, "sampled selector proposals match q by chi-square");
    check(chi2(p_count[0], p_count[1], 0.2, trials) < 10.83,
          "rejection sampling restores the target marginal at T > 0");

    int32_t packed[strata::spec::kDf2SampleQStride] = {};
    strata::spec::pack_df2_spec_q(1, q_tokens, q_probs, packed);
    float unpacked = 0.0f;
    std::memcpy(&unpacked, packed + strata::spec::kDf2SampleQEntries, sizeof(float));
    check(packed[0] == 0 && packed[1] == 1 && packed[2] == -1 && std::fabs(unpacked - q0) < 1e-6f,
          "selector conditionals pack in the verifier's q-row format");

    const int32_t greedy_sentinel = 77;
    int32_t greedy_output = greedy_sentinel;
    if (strata::spec::df2_sample_walk_for_request(false, false, 1.0f) ||
        strata::spec::df2_sample_walk_for_request(true, true, 0.0f) ||
        strata::spec::df2_sample_walk_for_request(true, false, 0.0f))
        greedy_output = drafts[0];
    check(greedy_output == greedy_sentinel, "disabled and greedy sampling leave the existing draft path untouched");

    std::printf("df2_sample_walk_test: %s (q %.4f, target %.4f)\n", failures == 0 ? "OK" : "FAILED",
                (double) q_count[0] / trials, (double) p_count[0] / trials);

    // ---- Lever C1: Linear bit-exact equivalence oracle & lossless tree-aware rejection sampling oracle ----
    {
        // 1. Exact identity vs linear `spec_verify_row` + `spec_sample` when the tree is linear (0->1->2).
        constexpr int kVocab = 4;
        constexpr double kTarget[kVocab][kVocab] = {
            {0.40, 0.30, 0.20, 0.10},
            {0.15, 0.45, 0.25, 0.15},
            {0.25, 0.20, 0.35, 0.20},
            {0.10, 0.20, 0.30, 0.40},
        };
        const int32_t kIds[kVocab] = {0, 1, 2, 3};
        const int32_t kTopTok[7 * strata::spec::kDf2SelectorTopK] = {
            0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            1, 2, 3, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            2, 3, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        };
        std::array<float, 7 * strata::spec::kDf2SelectorTopK * strata::spec::kDf2SelectorTopK> sel{};
        sel.fill(-INFINITY);
        const double q_base[3][4] = {
            {0.50, 0.25, 0.15, 0.10},
            {0.45, 0.30, 0.15, 0.10},
            {0.40, 0.35, 0.15, 0.10},
        };
        for (int r = 0; r < 3; ++r) {
            for (int prev = 0; prev < (r == 0 ? 1 : 4); ++prev) {
                for (int j = 0; j < 4; ++j) {
                    const double v = q_base[r][(j + prev) % 4];
                    sel[(r * strata::spec::kDf2SelectorTopK + prev) * strata::spec::kDf2SelectorTopK + j] =
                        static_cast<float>(std::log(v));
                }
            }
        }

        int lin_exact_mismatch = 0;
        for (int trial = 0; trial < 5000; ++trial) {
            const uint64_t seed = 0xC100ULL + static_cast<uint64_t>(trial);
            const uint64_t pos0 = 17 + static_cast<uint64_t>(trial % 13);
            int32_t anchor = trial % kVocab;

            int32_t drafts[7] = {};
            float sel_q[7] = {};
            int32_t qtok[7 * strata::spec::kDf2SelectorTopK] = {};
            float qprob[7 * strata::spec::kDf2SelectorTopK] = {};
            check(strata::spec::df2_sample_selector_walk(2, kTopTok, sel.data(), seed, pos0, drafts, sel_q, qtok, qprob),
                  "linear equivalence walk succeeds");

            strata::spec::Df2TreeSpec lin_tree = strata::spec::make_linear_tree(anchor, drafts, 2);
            strata::spec::Df2TreeSampleQ lin_q;
            check(strata::spec::df2_tree_populate_sampled_q(lin_tree, anchor, drafts, 2, kTopTok, sel.data(), false,
                                                            seed, pos0, lin_q),
                  "linear populate_sampled_q succeeds");

            strata::core::SpecP p_views[3] = {
                {kIds, kTarget[anchor], kVocab},
                {kIds, kTarget[drafts[0]], kVocab},
                {kIds, kTarget[drafts[1]], kVocab},
            };
            int32_t tree_bonus = -1;
            int64_t tree_resamp = 0;
            const auto twalk = strata::spec::df2_tree_sample_accept(lin_tree, p_views, &lin_q, seed, pos0,
                                                                    &tree_bonus, &tree_resamp);

            // Linear oracle using `spec_verify_row` + `spec_sample`
            int lin_acc = 0;
            int32_t lin_bonus = -1;
            for (int t = 0; t < 2; ++t) {
                const uint64_t pos_t = pos0 + static_cast<uint64_t>(t);
                const float ua = strata::core::philox_uniform_host(seed, pos_t | strata::core::kSpecCtrAccept);
                const float ur = strata::core::philox_uniform_host(seed, pos_t | strata::core::kSpecCtrResid);
                const strata::core::SpecQ qrow{
                    qtok + t * strata::spec::kDf2SelectorTopK, qprob + t * strata::spec::kDf2SelectorTopK, 4};
                const auto res = strata::core::spec_verify_row(p_views[t], drafts[t], qrow, ua, ur);
                if (res.accepted) {
                    ++lin_acc;
                } else {
                    lin_bonus = res.token;
                    break;
                }
            }
            if (lin_acc == 2) {
                const float ub = strata::core::philox_uniform_host(seed, pos0 + 2);
                lin_bonus = strata::core::spec_sample(p_views[2], ub);
            }
            if (twalk.accepted_drafts() != lin_acc || tree_bonus != lin_bonus) ++lin_exact_mismatch;
        }
        check(lin_exact_mismatch == 0, "linear tree sample accept is bit-exact against spec_verify_row + spec_sample");

        // 2. Multi-step 3-token autoregressive Markov rollout oracle (64 joint outcomes):
        //    compare exact target Markov law vs linear oracle, deterministic tree (point-mass),
        //    stochastic tree (dedup=false), and stochastic tree (dedup=true), plus corrupt negative control.
        constexpr int kRollouts = 500000;
        constexpr int kSeqs = 64;
        std::array<double, kSeqs> exact_p{};
        for (int a = 0; a < kVocab; ++a) {
            for (int b = 0; b < kVocab; ++b) {
                for (int c = 0; c < kVocab; ++c) {
                    exact_p[(a * 4 + b) * 4 + c] = kTarget[0][a] * kTarget[a][b] * kTarget[b][c];
                }
            }
        }

        // Fixed drafter state transitions conditioned on previous committed token (with deliberately skewed q != p
        // and runner-up candidates so branch hits and reject resamples trigger frequently).
        auto fill_selector_for_anchor = [&](int32_t anc, int32_t* toptok_out, float* seltab_out,
                                            int32_t* greedy_drafts, float* greedy_probs) {
            for (int r = 0; r < 3; ++r) {
                const int shift = (anc + r) % 4;
                const double q0_vec[4] = {0.42, 0.35, 0.15, 0.08};
                for (int j = 0; j < strata::spec::kDf2SelectorTopK; ++j) {
                    toptok_out[r * strata::spec::kDf2SelectorTopK + j] = (j < 4) ? ((j + shift) % 4) : -1;
                }
                for (int prev = 0; prev < strata::spec::kDf2SelectorTopK; ++prev) {
                    for (int j = 0; j < strata::spec::kDf2SelectorTopK; ++j) {
                        const int rot = (j + prev) % 4;
                        seltab_out[(r * strata::spec::kDf2SelectorTopK + prev) * strata::spec::kDf2SelectorTopK + j] =
                            (j < 4 && prev < 4) ? static_cast<float>(std::log(q0_vec[rot])) : -INFINITY;
                    }
                }
            }
            int prev_slot = 0;
            for (int r = 0; r < 3; ++r) {
                int best_j = 0;
                float best_s = -INFINITY;
                for (int j = 0; j < 4; ++j) {
                    const float sc =
                        seltab_out[(r * strata::spec::kDf2SelectorTopK + prev_slot) * strata::spec::kDf2SelectorTopK + j];
                    if (sc > best_s) { best_s = sc; best_j = j; }
                }
                greedy_drafts[r] = toptok_out[r * strata::spec::kDf2SelectorTopK + best_j];
                double sum_e = 0.0;
                for (int j = 0; j < 4; ++j) {
                    sum_e += std::exp(static_cast<double>(
                        seltab_out[(r * strata::spec::kDf2SelectorTopK + prev_slot) * strata::spec::kDf2SelectorTopK + j] -
                        best_s));
                }
                greedy_probs[r] = static_cast<float>(1.0 / sum_e);
                prev_slot = best_j;
            }
        };

        auto run_rollout = [&](int mode, bool dedup, strata::spec::Df2TreeAcceptHist* hist_out) {
            // mode 0 = linear oracle, 1 = deterministic tree point-mass, 2 = stochastic sampled tree, 3 = corrupt neg ctrl
            std::array<long long, kSeqs> counts{};
            for (int r = 0; r < kRollouts; ++r) {
                const uint64_t seed = 0x5A170000ULL + static_cast<uint64_t>(r);
                int32_t seq[3] = {};
                int produced = 0;
                int32_t x = 0;
                uint64_t pos = 0;
                uint64_t round_idx = 0;
                while (produced < 3) {
                    int32_t toptok_buf[7 * strata::spec::kDf2SelectorTopK] = {};
                    float seltab_buf[7 * strata::spec::kDf2SelectorTopK * strata::spec::kDf2SelectorTopK] = {};
                    int32_t g_drafts[7] = {};
                    float g_probs[7] = {};
                    fill_selector_for_anchor(x, toptok_buf, seltab_buf, g_drafts, g_probs);

                    int32_t drafts[7] = {};
                    float dprobs[7] = {};
                    int32_t qtok[7 * strata::spec::kDf2SelectorTopK] = {};
                    float qprob[7 * strata::spec::kDf2SelectorTopK] = {};
                    const int nd = std::min(2, 3 - produced);
                    const bool sample_walk = (mode == 0 || mode == 2 || mode == 3);
                    const uint64_t draft_key = pos ^ (round_idx << 32);
                    ++round_idx;
                    if (sample_walk) {
                        strata::spec::df2_sample_selector_walk(nd, toptok_buf, seltab_buf, seed, draft_key,
                                                               drafts, dprobs, qtok, qprob);
                    } else {
                        for (int i = 0; i < 7; ++i) { drafts[i] = g_drafts[i]; dprobs[i] = g_probs[i]; }
                    }

                    if (mode == 0 || produced == 2) {
                        int a = 0;
                        int32_t bonus = -1;
                        int32_t cur = x;
                        for (int t = 0; t < nd; ++t) {
                            const uint64_t pt = pos + static_cast<uint64_t>(t);
                            const float ua = strata::core::philox_uniform_host(seed, pt | strata::core::kSpecCtrAccept);
                            const float ur = strata::core::philox_uniform_host(seed, pt | strata::core::kSpecCtrResid);
                            const strata::core::SpecP P_t{kIds, kTarget[cur], kVocab};
                            const strata::core::SpecQ Q_t = sample_walk
                                ? strata::core::SpecQ{qtok + t * strata::spec::kDf2SelectorTopK,
                                                      qprob + t * strata::spec::kDf2SelectorTopK, 4}
                                : strata::core::SpecQ{nullptr, nullptr, -1};
                            const auto vr = strata::core::spec_verify_row(P_t, drafts[t], Q_t, ua, ur);
                            if (vr.accepted) {
                                if (produced < 3) seq[produced++] = drafts[t];
                                cur = drafts[t];
                                ++a;
                            } else {
                                bonus = vr.token;
                                break;
                            }
                        }
                        if (a == nd) {
                            const strata::core::SpecP P_end{kIds, kTarget[cur], kVocab};
                            const float ub = strata::core::philox_uniform_host(seed, pos + static_cast<uint64_t>(nd));
                            bonus = strata::core::spec_sample(P_end, ub);
                        }
                        if (produced < 3) seq[produced++] = bonus;
                        x = bonus;
                        pos += static_cast<uint64_t>(a + 1);
                    } else {
                        strata::spec::Df2TreeBuildOptions topts;
                        topts.draft_conf = 0.8;  // forces cut=0 fanout whenever primary < 0.8 (always here)
                        topts.min_branch_p = 0.05f;
                        topts.max_fanout = 2;
                        topts.max_nodes = std::min(5, 1 + (3 - produced) + 2);
                        topts.dedup_siblings = dedup;
                        strata::spec::Df2TreeSpec tree = strata::spec::df2_tree_build_from_selector_ex(
                            x, drafts, dprobs, nd, toptok_buf, seltab_buf, topts);
                        strata::spec::Df2TreeSampleQ sq;
                        if (mode == 2 || mode == 3) {
                            strata::spec::df2_tree_populate_sampled_q(tree, x, drafts, nd, toptok_buf, seltab_buf,
                                                                      dedup, seed, draft_key, sq);
                        }
                        if (mode == 3) {
                            // Deliberate bug for negative control: flatten q on node 1 to distort rejection ratio.
                            for (int i = 0; i < sq.node[1].n; ++i) sq.node[1].q[i] = 1.0f / static_cast<float>(sq.node[1].n);
                        }
                        std::array<strata::core::SpecP, strata::spec::kDf2TreeMaxNodes> pv{};
                        for (int i = 0; i < tree.count; ++i) {
                            pv[static_cast<size_t>(i)] =
                                strata::core::SpecP{kIds, kTarget[tree.token[static_cast<size_t>(i)]], kVocab};
                        }
                        int32_t bonus = -1;
                        int64_t resamp = 0;
                        const auto walk = strata::spec::df2_tree_sample_accept(
                            tree, pv.data(), (mode == 2 || mode == 3) ? &sq : nullptr, seed, pos, &bonus, &resamp);
                        const int a = walk.accepted_drafts();
                        if (hist_out != nullptr) hist_out->observe(tree, walk, true, resamp);
                        for (int k = 1; k <= a && produced < 3; ++k) {
                            seq[produced++] = tree.token[walk.path[static_cast<size_t>(k)]];
                        }
                        if (produced < 3) seq[produced++] = bonus;
                        x = bonus;
                        pos += static_cast<uint64_t>(a + 1);
                    }
                }
                ++counts[(seq[0] * 4 + seq[1]) * 4 + seq[2]];
            }
            return counts;
        };

        auto pearson_chi2 = [&](const std::array<long long, kSeqs>& c) {
            double x2 = 0.0;
            for (int i = 0; i < kSeqs; ++i) {
                const double e = exact_p[static_cast<size_t>(i)] * static_cast<double>(kRollouts);
                const double d = static_cast<double>(c[static_cast<size_t>(i)]) - e;
                x2 += (d * d) / e;
            }
            return x2;
        };
        auto two_sample_chi2 = [&](const std::array<long long, kSeqs>& a, const std::array<long long, kSeqs>& b) {
            double x2 = 0.0;
            for (int i = 0; i < kSeqs; ++i) {
                const double sum = static_cast<double>(a[static_cast<size_t>(i)] + b[static_cast<size_t>(i)]);
                if (sum > 0.0) {
                    const double d = static_cast<double>(a[static_cast<size_t>(i)] - b[static_cast<size_t>(i)]);
                    x2 += (d * d) / sum;
                }
            }
            return x2;
        };

        strata::spec::Df2TreeAcceptHist hist_det, hist_stoch, hist_dedup;
        const auto c_lin = run_rollout(0, false, nullptr);
        const auto c_det = run_rollout(1, false, &hist_det);
        const auto c_stoch = run_rollout(2, false, &hist_stoch);
        const auto c_dedup = run_rollout(2, true, &hist_dedup);
        const auto c_bad = run_rollout(3, false, nullptr);

        // 63 df chi-square p=0.001 threshold is 103.44.
        const double chi2_lin = pearson_chi2(c_lin);
        const double chi2_det = pearson_chi2(c_det);
        const double chi2_stoch = pearson_chi2(c_stoch);
        const double chi2_dedup = pearson_chi2(c_dedup);
        const double chi2_bad = pearson_chi2(c_bad);
        const double chi2_lin_det = two_sample_chi2(c_lin, c_det);
        const double chi2_lin_stoch = two_sample_chi2(c_lin, c_stoch);
        const double chi2_lin_dedup = two_sample_chi2(c_lin, c_dedup);
        check(chi2_lin < 103.5, "linear sampled oracle matches exact 3-step Markov distribution");
        check(chi2_det < 103.5, "point-mass tree rejection sampling preserves exact 3-step Markov distribution");
        check(chi2_stoch < 103.5, "stochastic tree rejection sampling (dedup=false) preserves exact Markov distribution");
        check(chi2_dedup < 103.5, "stochastic tree rejection sampling (dedup=true) preserves exact Markov distribution");
        check(chi2_lin_det < 103.5, "point-mass tree matches linear sampled oracle two-sample chi2");
        check(chi2_lin_stoch < 103.5, "stochastic tree matches linear sampled oracle two-sample chi2");
        check(chi2_lin_dedup < 103.5, "stochastic dedup tree matches linear sampled oracle two-sample chi2");
        check(chi2_bad > 150.0, "corrupted proposal q negative control fails chi-square distribution check");
        check(hist_det.sampled_tree_rounds > 0 && hist_det.runner_hits > 0 && hist_det.reject_resamples > 0,
              "deterministic sampled tree actively engages runner-up branches and residual resamples");
        check(hist_stoch.sampled_tree_rounds > 0 && hist_stoch.runner_hits > 0 && hist_stoch.reject_resamples > 0,
              "stochastic sampled tree actively engages runner-up branches and residual resamples");
        check(hist_dedup.sampled_tree_rounds > 0 && hist_dedup.runner_hits > 0 && hist_dedup.reject_resamples > 0,
              "dedup stochastic sampled tree actively engages runner-up branches and residual resamples");
        std::printf("df2_sample_walk_test tree_oracle: %s (lin=%.2f det=%.2f stoch=%.2f dedup=%.2f "
                    "lin_vs_det=%.2f lin_vs_stoch=%.2f lin_vs_dedup=%.2f bad=%.2f "
                    "det_rhits=%lld stoch_rhits=%lld det_resamp=%lld stoch_resamp=%lld)\n",
                    failures == 0 ? "OK" : "FAILED", chi2_lin, chi2_det, chi2_stoch, chi2_dedup,
                    chi2_lin_det, chi2_lin_stoch, chi2_lin_dedup, chi2_bad,
                    (long long) hist_det.runner_hits, (long long) hist_stoch.runner_hits,
                    (long long) hist_det.reject_resamples, (long long) hist_stoch.reject_resamples);
    }

    return failures == 0 ? 0 : 1;
}
