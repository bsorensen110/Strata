#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace strata::spec {

// Attention kernels mask cells using the device step record. A bounded launch
// bucket can therefore serve later positions without rebuilding the draft graph.
inline int64_t dflash_attention_bucket(int64_t visible, int64_t capacity) {
    if (visible < 1 || visible > capacity) return 0;
    int64_t cells = std::min<int64_t>(64, capacity);
    while (cells < visible) {
        if (cells > capacity / 4) return capacity;
        cells *= 4;
    }
    return cells;
}

// A target verifies anchor + K proposals; DFlash2 also evaluates an anchor
// in its draft forward. Subtraction keeps untrusted large positions overflow-safe.
inline bool dflash_proposal_fits(int64_t pos, int k, bool dflash2,
                                 int64_t target_capacity, int64_t draft_capacity) {
    return pos >= 0 && k >= 1 && k <= 7 && pos <= target_capacity &&
           pos <= draft_capacity && k + 1 <= target_capacity - pos &&
           k + (dflash2 ? 1 : 0) <= draft_capacity - pos;
}

// Measure real forwards at each block length: non-causal DFlash logits at K=2
// are not a truncated K=7 forward. Score the committed tokens per millisecond,
// including the draft that produced this window. K=0 maintains context but
// skips proposing. Periodic probes allow drafting to resume as the text changes.
class DFlashPolicy {
public:
    explicit DFlashPolicy(int max_k) : max_k_(std::clamp(max_k, 0, 7)) { reset(); }

    // Costs and acceptance belong to this request; graph capture happens once
    // per width in the verifier, so retain only its warm state across requests.
    void reset() {
        n_.fill(0); tokens_.fill(0); ms_.fill(0);
        best_ = rounds_ = 0;
        probe_ = max_k_ >= 4 ? 4 : 0;
    }

    int choose() {
        // Start with small windows; wide non-causal forwards are still measured
        // by periodic probes, without charging every short reply for a full sweep.
        static constexpr int order[] = {2, 3, 1, 0};
        for (int k : order)
            if (k <= max_k_ && n_[k] < 2) return k;
        if (rounds_ % 16 == 0) {
            const int probe = probe_++ % (max_k_ + 1);
            return probe;
        }
        int best = best_;
        for (int k = 0; k <= max_k_; ++k)
            if (n_[k] > 0 && tokens_[k] / ms_[k] > 1.04 * tokens_[best] / ms_[best]) best = k;
        best_ = best;
        return best;
    }

    void observe(int k, int emitted, double ms) {
        if (k < 0 || k > max_k_ || emitted < 1 || emitted > k + 1 || !std::isfinite(ms) || ms <= 0) return;
        // The first window of each width can capture a CUDA graph. Its one-time
        // cost must not make an otherwise useful block look permanently slow.
        if (seen_[k]++ == 0) return;
        if (n_[k] == 0) { tokens_[k] = emitted; ms_[k] = ms; }
        else {
            tokens_[k] = 0.8 * tokens_[k] + 0.2 * emitted;
            ms_[k] = 0.8 * ms_[k] + 0.2 * ms;
        }
        ++n_[k];
        ++rounds_;
    }

private:
    int max_k_, best_ = 0, rounds_ = 0, probe_ = 0;
    std::array<int, 8> n_{};
    std::array<int, 8> seen_{};
    std::array<double, 8> tokens_{}, ms_{};
};

// Truncate a proposed block [0..k) using a base probability floor, an optional
// per-depth linear slope (depth_slope >= 0 raises the bar for deeper positions),
// and an optional cumulative survival floor (product of prefix confidences).
inline int dflash_truncate_prefix(const float* dprob, int k, float min_p,
                                  float depth_slope = 0.0f, float min_cum_p = 0.0f) {
    if (k <= 0) return 0;
    if (!dprob || (min_p <= 0.0f && depth_slope <= 0.0f && min_cum_p <= 0.0f)) return k;
    int prefix = 0;
    float cum = 1.0f;
    for (int i = 0; i < k; ++i) {
        const float p = dprob[i];
        if (!std::isfinite(p)) break;
        const float thr = std::clamp(min_p + depth_slope * (float) i, 0.0f, 0.98f);
        cum *= p;
        if (p < thr || (min_cum_p > 0.0f && cum < min_cum_p)) break;
        ++prefix;
    }
    return prefix;
}

// Per-request DFlash acceptance and truncation telemetry (up to K=7 proposals).
struct DFlashTelemetry {
    int64_t proposed_rows = 0;
    int64_t gate_truncated_rows = 0;
    int64_t target_only_rounds = 0;
    std::array<int64_t, 8> windows_by_k{};
    std::array<int64_t, 8> accepted_by_k{};
    std::array<int64_t, 7> depth_tested{};
    std::array<int64_t, 7> depth_ok{};

    void reset() { *this = DFlashTelemetry{}; }

    void record_proposal(int proposed_k, int offered_k) {
        const int pk = std::clamp(proposed_k, 0, 7);
        const int ok = std::clamp(offered_k, 0, pk);
        proposed_rows += pk;
        gate_truncated_rows += (pk - ok);
    }

    void record_window(int offered_k, int accepted_a) {
        const int k = std::clamp(offered_k, 0, 7);
        const int a = std::clamp(accepted_a, 0, k);
        ++windows_by_k[(size_t) k];
        accepted_by_k[(size_t) k] += a;
        if (k == 0) ++target_only_rounds;
        for (int d = 0; d < k && d < 7 && d <= a; ++d) {
            ++depth_tested[(size_t) d];
            depth_ok[(size_t) d] += (d < a);
        }
    }
};

// Deterministic confidence-adaptive block controller: grows toward max_k when
// two consecutive proposals survive the confidence gate with high minimum
// confidence, and shrinks toward prefix + 1 after two consecutive truncations
// (hysteresis avoids single-round width oscillation and graph churn).
class DFlashConfidencePolicy {
public:
    explicit DFlashConfidencePolicy(int max_k, int init_k = 3, float expand_prob = 0.80f)
        : max_k_(std::clamp(max_k, 1, 7)),
          init_k_(std::clamp(init_k > 0 ? init_k : 3, 1, max_k_)),
          expand_prob_(expand_prob) {
        reset();
    }

    void reset() {
        cur_k_ = init_k_;
        high_streak_ = 0;
        trunc_streak_ = 0;
    }

    int choose() const { return cur_k_; }

    void observe(int proposed_k, int prefix_k, const float* dprob) {
        if (proposed_k <= 0) return;
        float min_conf = 1.0f;
        if (dprob) {
            for (int i = 0; i < prefix_k; ++i)
                min_conf = std::min(min_conf, dprob[i]);
        }
        if (prefix_k >= proposed_k && min_conf >= expand_prob_) {
            trunc_streak_ = 0;
            if (++high_streak_ >= 2 && cur_k_ < max_k_) {
                ++cur_k_;
                high_streak_ = 0;
            }
        } else if (prefix_k < proposed_k) {
            high_streak_ = 0;
            if (++trunc_streak_ >= 2) {
                cur_k_ = std::clamp(prefix_k + 1, 1, max_k_);
                trunc_streak_ = 0;
            }
        } else {
            high_streak_ = 0;
            trunc_streak_ = 0;
        }
    }

private:
    int max_k_, init_k_, cur_k_ = 3;
    int high_streak_ = 0, trunc_streak_ = 0;
    float expand_prob_ = 0.80f;
};

}  // namespace strata::spec
