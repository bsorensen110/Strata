#pragma once

#include "strata/core/spec_prob.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace strata::spec {

inline constexpr int kDf2SelectorTopK = 16;
inline constexpr int kDf2SampleQEntries = 64;
inline constexpr int kDf2SampleQStride = 2 * kDf2SampleQEntries;
static_assert(kDf2SampleQStride == strata::core::kSpecQStride);

inline bool df2_sample_walk_env() {
    static const bool enabled = [] {
        const char* e = std::getenv("STRATA_DF2_SAMPLE_WALK");
        return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();
    return enabled;
}

inline bool df2_sample_walk_for_request(bool enabled, bool greedy, float temperature) {
    return enabled && !greedy && temperature > 0.0f;
}

inline uint64_t df2_sample_mix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Scores use [row][previous candidate slot][candidate slot], with previous slot zero at row zero.
// Optional `max_q[i]` exports the draw-independent row maximum `max_j q[i][j]` before drawing `drafts[i]`
// so downstream window/tree gates never condition verification participation on the random draw outcome.
inline bool df2_sample_selector_walk(int rows, const int32_t* candidates, const float* scores,
                                    uint64_t seed, uint64_t position, int32_t* drafts, float* selected_q,
                                    int32_t* q_tokens, float* q_probs, float* max_q = nullptr) {
    if (rows < 0 || rows > 7 || (rows > 0 && (!candidates || !scores || !drafts || !q_tokens || !q_probs)))
        return false;
    int previous = 0;
    for (int i = 0; i < rows; ++i) {
        const float* row_scores = scores + (static_cast<size_t>(i) * kDf2SelectorTopK + previous) * kDf2SelectorTopK;
        const int32_t* row_tokens = candidates + static_cast<size_t>(i) * kDf2SelectorTopK;
        float max_score = -INFINITY;
        for (int j = 0; j < kDf2SelectorTopK; ++j)
            if (row_tokens[j] >= 0 && std::isfinite(row_scores[j])) max_score = std::max(max_score, row_scores[j]);
        if (!std::isfinite(max_score)) return false;

        double weights[kDf2SelectorTopK] = {};
        double total = 0.0;
        for (int j = 0; j < kDf2SelectorTopK; ++j) {
            if (row_tokens[j] < 0 || !std::isfinite(row_scores[j])) continue;
            weights[j] = std::exp(static_cast<double>(row_scores[j] - max_score));
            total += weights[j];
        }
        if (!(total > 0.0) || !std::isfinite(total)) return false;

        float sum_q = 0.0f;
        for (int j = 0; j < kDf2SelectorTopK; ++j) {
            const float q = static_cast<float>(weights[j] / total);
            q_tokens[static_cast<size_t>(i) * kDf2SelectorTopK + j] = weights[j] > 0.0 ? row_tokens[j] : -1;
            q_probs[static_cast<size_t>(i) * kDf2SelectorTopK + j] = q;
            sum_q += q;
        }
        if (!(sum_q > 0.0f) || !std::isfinite(sum_q)) return false;
        float row_max_q = 0.0f;
        for (int j = 0; j < kDf2SelectorTopK; ++j) {
            float& q_slot = q_probs[static_cast<size_t>(i) * kDf2SelectorTopK + j];
            q_slot /= sum_q;
            if (q_slot > row_max_q) row_max_q = q_slot;
        }
        if (max_q != nullptr) max_q[i] = row_max_q;

        const uint64_t bits = df2_sample_mix(seed ^ df2_sample_mix(position + static_cast<uint64_t>(i)));
        const double u = (static_cast<double>(bits >> 11) + 0.5) * (1.0 / 9007199254740992.0);
        double cumulative = 0.0;
        int picked = -1;
        int last = -1;
        for (int j = 0; j < kDf2SelectorTopK; ++j) {
            const float q = q_probs[static_cast<size_t>(i) * kDf2SelectorTopK + j];
            if (!(q > 0.0f)) continue;
            last = j;
            cumulative += q;
            if (u < cumulative) { picked = j; break; }
        }
        if (picked < 0) picked = last;
        if (picked < 0) return false;
        drafts[i] = row_tokens[picked];
        if (selected_q != nullptr) selected_q[i] = q_probs[static_cast<size_t>(i) * kDf2SelectorTopK + picked];
        previous = picked;
    }
    return true;
}

/// Computes draw-independent prefix-survival products `surv_out[i] = prod_{r=0..i} clamp(max_q[r], 0, 1)`.
inline void df2_sample_prefix_survival(const float* max_q, int rows, float* surv_out) noexcept {
    if (rows <= 0 || max_q == nullptr || surv_out == nullptr) return;
    double prod = 1.0;
    for (int i = 0; i < rows; ++i) {
        const double q = std::clamp(static_cast<double>(max_q[i]), 0.0, 1.0);
        prod *= q;
        surv_out[i] = static_cast<float>(prod);
    }
}

inline void pack_df2_spec_q(int rows, const int32_t* q_tokens, const float* q_probs, int32_t* packed) {
    for (int i = 0; i < rows; ++i) {
        int32_t* row = packed + static_cast<size_t>(i) * kDf2SampleQStride;
        std::fill(row, row + kDf2SampleQEntries, -1);
        int32_t* q_bits = row + kDf2SampleQEntries;
        std::fill(q_bits, q_bits + kDf2SampleQEntries, 0);
        for (int j = 0; j < kDf2SelectorTopK; ++j) {
            const int32_t id = q_tokens[static_cast<size_t>(i) * kDf2SelectorTopK + j];
            if (id < 0) continue;
            row[j] = id;
            std::memcpy(&q_bits[j], &q_probs[static_cast<size_t>(i) * kDf2SelectorTopK + j], sizeof(float));
        }
    }
}

}  // namespace strata::spec
