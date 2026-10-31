#pragma once
// Adaptive draft depth for --batch-mtp windows (STRATA_BATCH_MTP_DEPTH=N, opt-in; docs/BATCHING.md).
//
// Without it every selected slot verifies exactly one MTP proposal (two rows). With it a slot drafts and verifies up
// to N proposals, and how many it gets is gated twice:
//
//   1. by its acceptance rate: an exponential moving average of "the target picked the slot's first proposal".
//      Before the draft, the slot asks for the deepest d (1..N) whose expected chain acceptance ema^d is still at
//      least `floor`. A slot whose proposals keep failing stays at one proposal and pays for no extra draft steps.
//   2. by the draft's own probability: once the drafts are on the host, proposal j (j >= 1) keeps its row only while
//      the product of the draft probabilities up to it is at least `floor`. The first proposal always keeps its row,
//      as today. (Not applied when the drafts stay on the device, STRATA_BATCH_MTP_DEVTOK: the rows are fixed before
//      the drafts land.)
//
// The window's row cap is then shared: every selected slot keeps its anchor and its first proposal (today's two
// rows); a slot's deeper proposals are dropped, lowest expected value (ema^k) first, until the window fits.
// Answers do not change: a proposal the target did not pick is never emitted (batch_spec_accept).
//
// Header-only, CPU: no verifier, no drafter, no GPU (tests/core/batch_mtp_depth_test.cpp).
#include <cmath>
#include <cstdlib>

namespace strata::core {

struct MtpDepthConfig {
    int max_depth = 1;        ///< proposals per slot at most; 1 = off (one proposal, today's windows)
    float floor = 0.5f;       ///< the expected acceptance the deepest proposal row must still reach
    float alpha = 0.25f;      ///< EMA weight of the newest window
    float ema_init = 0.5f;    ///< a new request's acceptance estimate (below the gate for depth 2 at floor 0.5)
    bool on() const { return max_depth > 1; }
};

/// STRATA_BATCH_MTP_DEPTH (N, clamped to [1, max_cap]) and STRATA_BATCH_MTP_DEPTH_FLOOR (a probability in (0, 1]).
/// Unset, empty, unparsable or 1: off.
inline MtpDepthConfig mtp_depth_config(const char* depth_env, const char* floor_env, int max_cap) {
    MtpDepthConfig c;
    if (depth_env != nullptr && depth_env[0] != '\0') {
        const int n = std::atoi(depth_env);
        c.max_depth = n < 1 ? 1 : (n > max_cap ? (max_cap < 1 ? 1 : max_cap) : n);
    }
    if (floor_env != nullptr && floor_env[0] != '\0') {
        const float f = (float) std::atof(floor_env);
        if (f > 0.0f && f <= 1.0f) c.floor = f;
    }
    return c;
}

/// The slot's acceptance estimate after one window that offered its first proposal.
inline float mtp_depth_ema(float ema, bool accepted, float alpha) {
    return (1.0f - alpha) * ema + alpha * (accepted ? 1.0f : 0.0f);
}

/// How many proposals the slot drafts next: the deepest d in [1, max_depth] with ema^d >= floor; at least 1.
inline int mtp_depth_for(float ema, const MtpDepthConfig& c) {
    int d = 1;
    float e = ema;
    while (d < c.max_depth) {
        e *= ema;
        if (!(e >= c.floor)) break;
        ++d;
    }
    return d;
}

/// How many of `n` drafted proposals keep a row by their probabilities: the first always (n > 0), proposal j while
/// probs[0] * ... * probs[j] >= floor.  `probs` null: all n.
inline int mtp_depth_trim(const float* probs, int n, float floor) {
    if (n <= 0) return 0;
    if (probs == nullptr) return n;
    int k = 1;
    float cum = probs[0];
    while (k < n) {
        cum *= probs[k];
        if (!(cum >= floor)) break;
        ++k;
    }
    return k;
}

/// Fit the selected slots' proposal counts `k[0, n_sel)` into `cap` rows (each slot: 1 anchor + k[a] proposals).
/// Proposals past the first are dropped one at a time, the one with the lowest ema[a]^k[a] first (ties: the later
/// slot), never below min(k[a], 1).  Returns the rows used; more than `cap` only when the first proposals alone do
/// not fit (the caller selects at most cap / 2 slots, so that does not happen).
inline int mtp_depth_fit(int* k, const float* ema, int n_sel, int cap) {
    int rows = 0;
    for (int a = 0; a < n_sel; ++a) rows += 1 + k[a];
    while (rows > cap) {
        int worst = -1;
        float worst_v = 0.0f;
        for (int a = 0; a < n_sel; ++a) {
            if (k[a] <= 1) continue;
            const float v = std::pow(ema[a], (float) k[a]);
            if (worst < 0 || v <= worst_v) { worst = a; worst_v = v; }
        }
        if (worst < 0) break;
        --k[worst];
        --rows;
    }
    return rows;
}

}  // namespace strata::core
