// include/strata/core/spec_prob.hpp - PROBABILISTIC DRAFT ACCEPTANCE (STRATA_SPEC_PROB=1, off by default).
//
// Speculative rejection sampling (Leviathan et al. / Chen et al.) for the MTP drafts of a SAMPLED request.  Under
// sampling the verify window used to accept a draft only when it EQUALS the token the target sampled for that row
// (exact match).  For a draft that is the head's argmax - a point mass - that is already the standard rule, but it
// throws away what the draft head knows: its whole distribution q.  Here the draft layer SAMPLES its draft d from q
// (its own Philox stream) and the verifier accepts d with probability min(1, p(d) / q(d)); on a reject it samples the
// replacement from the residual norm(max(0, p - q)).  The output token of every row is then distributed exactly as
// p, the target's chain (penalties -> top_k -> top_p -> min_p -> temperature), whatever q is.
//
//   p   the TARGET'S distribution at the row: the list the sampler really samples from, after penalties, top_k,
//       top_p, min_p and temperature, renormalised.  At most 64 entries (kSelMax).  NOT the raw softmax: using it
//       would accept tokens the chain never emits (the llama.cpp #29924 class of bug).
//   q   the DRAFTER'S distribution: the same chain over the draft head's logits, with the drafter's own top_k /
//       top_p cut (each list is cut by its own rule - q never borrows p's cut).  A sparse list of at most 64
//       (id, probability) pairs, probabilities summing to 1.
//
// "NO q MEANS POINT MASS, NEVER q = 0."  A drafter without a candidate list (prompt lookup, suffix drafts, the
// lookup chain) proposes one token for sure: q is a delta at d, the accept probability is p(d) and the residual is p
// with d removed.  Treating "no list" as q = 0 would reject every such draft (llama.cpp #29924).  The engine only
// builds q lists for the MTP drafts; every other row of a window is a point-mass row.
//
// THE VERIFY ROWS ARE INDEPENDENT.  Row t of a window at pos0 judges draft t (token t + 1 of the window) with its
// own uniforms, three disjoint Philox counter spaces keyed by the row's position pos0 + t:
//     u_pick   counter = pos          the plain sample (the window's last row: the bonus token, and every row of a
//                                     window with no draft) - the same draw the exact-match sampler uses
//     u_accept counter = pos | 1<<62  accept / reject
//     u_resid  counter = pos | 1<<63  the residual sample after a reject
//     (the drafter's own draw: counter = pos | 1<<61)
// Row t writes out[t] = d when it accepts, and the residual sample (never d) when it rejects, so the host's
// unchanged loop `while (a < T-1 && window[a+1] == out[a]) ++a` takes exactly the accepted prefix, and out[a] is
// the bonus token.  Rows after the first reject are computed and ignored.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace strata::core {

/// The default confidence gate tests the drawn token. gated_proposal.hpp carries its admitted
/// mass and a NULL outcome to the verifier; a failed gate is still a proposal attempt.
/// Explicit `top` tests q's largest probability before the draw and uses the normalized-q path.
inline bool spec_gate_pick() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_SPEC_PROB_GATE");
        return e == nullptr || std::strcmp(e, "top") != 0;
    }();
    return on;
}

/// STRATA_SPEC_PROB_DT=x: the drafter's chain runs at x * the request's temperature (default 1).  Any q is valid for
/// rejection sampling; the acceptance sum min(p, q) is what it changes.
inline float spec_draft_temp_scale() {
    static const float s = [] {
        const char* e = std::getenv("STRATA_SPEC_PROB_DT");
        const float v = e != nullptr ? (float) std::atof(e) : 1.0f;
        return v > 0.0f ? v : 1.0f;
    }();
    return s;
}

/// STRATA_SPEC_MIN_TEMP=x: coupled / probabilistic drafting only for requests with temperature >= x (default 0: every
/// sampled request).  Measured on the RTX 3060 and 5070: sampled drafting gains at temperature 1.0 and loses below
/// about 0.7, where the draft head's argmax is the better guess; below x the drafter keeps its argmax drafts.
inline float spec_min_temp() {
    static const float s = [] {
        const char* e = std::getenv("STRATA_SPEC_MIN_TEMP");
        return e != nullptr ? (float) std::atof(e) : 0.0f;
    }();
    return s;
}

/// One q row = 128 int32 slots: 64 token ids (-1 ends the list), then the 64 probabilities as float bits.
inline constexpr int kSpecQEntries = 64;
inline constexpr int kSpecQStride = 2 * kSpecQEntries;

inline constexpr uint64_t kSpecCtrAccept = 1ull << 62;
inline constexpr uint64_t kSpecCtrResid = 1ull << 63;
inline constexpr uint64_t kSpecCtrDraft = 1ull << 61;

/// Post-chain distribution row transferable from GPU to host without copying full logits (`n <= kSpecQEntries`).
struct SpecPRow {
    int32_t id[kSpecQEntries];
    double p[kSpecQEntries];
    int32_t n;
};

/// Packed 16-byte result of GPU-side lossless sampled-tree rejection walk (`df2_tree_sample_accept`).
struct alignas(16) TreeSampleAcceptResult {
    uint8_t path[8];      ///< accepted physical node indices [0..n_keep-1]
    uint8_t n_keep;       ///< number of kept nodes (1..8; 0 on invalid)
    uint8_t flags;        ///< bit 0 = branch_hit, bit 1 = runner_up_hit
    uint16_t resamples;   ///< reject resample count (0 or 1 per walk)
    int32_t bonus_token;  ///< bonus token sampled at deepest kept node
};
static_assert(sizeof(TreeSampleAcceptResult) == 16, "TreeSampleAcceptResult must be packed 16 bytes");

/// Compact tree descriptor staged to mapped host memory for `launch_df2_tree_sample_accept`.
struct TreeSampleAcceptDesc {
    uint64_t seed;
    uint64_t pos0;
    int32_t token[8];
    int32_t count;
    int8_t parent[8];
    uint8_t depth[8];
    uint8_t slot_rank[8];
};

/// Optional per-node stochastic proposal distributions for `launch_df2_tree_sample_accept` (`n[c] < 0` => point mass).
struct TreeSampleAcceptQ {
    int32_t n[8];
    int32_t id[8][kSpecQEntries];
    float q[8][kSpecQEntries];
};

/// Host Philox 4x32-10 draw matching `sampler.cu::philox_uniform` bit-for-bit in `[0, 1)`.
inline float philox_uniform_host(uint64_t seed, uint64_t counter) noexcept {
    uint32_t c0 = static_cast<uint32_t>(counter);
    uint32_t c1 = static_cast<uint32_t>(counter >> 32);
    uint32_t c2 = static_cast<uint32_t>(seed);
    uint32_t c3 = static_cast<uint32_t>(seed >> 32);
    for (int i = 0; i < 10; ++i) {
        const uint32_t hi0 = static_cast<uint32_t>((0x9E3779B9ull * static_cast<uint64_t>(c0)) >> 32);
        const uint32_t hi1 = static_cast<uint32_t>((0xBB67AE85ull * static_cast<uint64_t>(c2)) >> 32);
        const uint32_t lo0 = 0x9E3779B9u * c0;
        const uint32_t lo1 = 0xBB67AE85u * c2;
        const uint32_t n0 = hi1 ^ c1 ^ static_cast<uint32_t>(i);
        const uint32_t n1 = lo1;
        const uint32_t n2 = hi0 ^ c3;
        const uint32_t n3 = lo0;
        c0 = n0;
        c1 = n1;
        c2 = n2;
        c3 = n3;
    }
    return static_cast<float>(c0 >> 8) * (1.0f / 16777216.0f);
}

inline constexpr uint64_t kTreeBranchCounterMask = (1ull << 56) - 1;
inline constexpr uint64_t kTreeBranchMaxRounds = 1ull << 24;
inline constexpr int kTreeBranchMaxNodeIdx = 31;

/// Returns true iff `key` fits inside the 56-bit base field and `idx` fits in the 5-bit slot field [0, 31].
inline constexpr bool tree_branch_counter_in_bounds(uint64_t key, int idx) noexcept {
    return (key >> 56) == 0 && idx >= 0 && idx <= kTreeBranchMaxNodeIdx;
}

/// Returns true iff `(pos, round_idx)` will not wrap in `pos ^ (round_idx << 32)` before 56-bit masking.
inline constexpr bool tree_branch_round_in_bounds(uint64_t pos, uint64_t round_idx) noexcept {
    return (pos >> 56) == 0 && round_idx < kTreeBranchMaxRounds;
}

/// Disjoint 64-bit Philox counters for tree acceptance checks:
/// `sib_idx == 0` reuses `pos_u | kSpecCtrAccept` (`010` tag, bit-identical to linear `spec_verify_row`);
/// `sib_idx > 0` uses top 3-bit tag `011` + 5-bit `sib_idx`, disjoint from plain (`000`), draft (`001`),
/// accept (`010`), and residual (`100`).
inline constexpr uint64_t tree_branch_accept_counter(uint64_t pos_u, int sib_idx) noexcept {
    if (sib_idx <= 0) return pos_u | kSpecCtrAccept;
    return (pos_u & kTreeBranchCounterMask) | (0x3ull << 61) | (static_cast<uint64_t>(sib_idx & 0x1F) << 56);
}

/// Disjoint 64-bit counter key (top 3-bit tag `101`) for host without-replacement branch draws.
inline constexpr uint64_t tree_branch_draft_counter(uint64_t pos0, int node_idx) noexcept {
    return (pos0 & kTreeBranchCounterMask) | (0x5ull << 61) | (static_cast<uint64_t>(node_idx & 0x1F) << 56);
}

/// Checked variant of `tree_branch_draft_counter` that rejects 56-bit key wrap or node-index overflow.
inline constexpr bool tree_branch_draft_counter_checked(uint64_t pos0, int node_idx, uint64_t& out) noexcept {
    if (!tree_branch_counter_in_bounds(pos0, node_idx)) return false;
    out = tree_branch_draft_counter(pos0, node_idx);
    return true;
}

/// The post-chain distribution of a row: `n` ids and their probabilities (double, the sampler's own `ex`).
struct SpecP {
    const int32_t* id;
    const double* p;
    int n;
};

/// The drafter's list for the row (n < 0: no list, a point mass at the draft).
struct SpecQ {
    const int32_t* id;
    const float* q;
    int n;
};

struct SpecResult {
    int32_t token;
    int accepted;   // 1: the draft was kept
};

/// The inverse-CDF pick of the sampler's tail, over weights w[i] / sum.
inline int32_t spec_pick(const SpecP& P, const double* w, double sum, float u) {
    double cum = 0.0;
    int last = -1;
    for (int i = 0; i < P.n; ++i) {
        if (!(w[i] > 0.0)) continue;
        last = i;
        cum += w[i] / sum;
        if ((double) u < cum) return P.id[i];
    }
    return last >= 0 ? P.id[last] : P.id[P.n > 0 ? P.n - 1 : 0];
}

/// A plain sample from p (no draft on this row: the last row of a window).  Matches the sampler's tail.
inline int32_t spec_sample(const SpecP& P, float u_pick) {
    double cum = 0.0;
    int32_t pick = P.id[P.n > 0 ? P.n - 1 : 0];
    for (int i = 0; i < P.n; ++i) {
        cum += P.p[i];
        if ((double) u_pick < cum) { pick = P.id[i]; break; }
    }
    return pick;
}

/// One verify row with a draft `d`: accept with min(1, p(d)/q(d)), else the residual sample.
inline SpecResult spec_verify_row(const SpecP& P, int32_t d, const SpecQ& Q, float u_accept, float u_resid) {
    double pd = 0.0;
    for (int i = 0; i < P.n; ++i)
        if (P.id[i] == d) { pd = P.p[i]; break; }
    double qd = -1.0;   // < 0: point mass
    if (Q.n >= 0)
        for (int i = 0; i < Q.n; ++i)
            if (Q.id[i] == d) { qd = (double) Q.q[i]; break; }
    const bool point = !(qd > 0.0);   // no list, or d is not in it (it always is for a drafted token): a point mass
    const bool accept = pd > 0.0 && (point ? (double) u_accept < pd : (double) u_accept * qd < pd);
    if (accept) return {d, 1};
    // the residual over p's support: max(0, p - q) (q = delta at d for a point mass: p without d)
    double w[64];
    double sum = 0.0;
    const int n = P.n < 64 ? P.n : 64;
    for (int i = 0; i < n; ++i) {
        double wi;
        if (point) {
            wi = P.id[i] == d ? 0.0 : P.p[i];
        } else {
            double qi = 0.0;
            for (int j = 0; j < Q.n; ++j)
                if (Q.id[j] == P.id[i]) { qi = (double) Q.q[j]; break; }
            wi = P.p[i] - qi;
            if (wi < 0.0) wi = 0.0;
        }
        w[i] = wi;
        sum += wi;
    }
    if (!(sum > 0.0)) return {d, 1};   // p and q agree everywhere (rounding): the draft is as likely as it can be
    return {spec_pick(SpecP{P.id, P.p, n}, w, sum, u_resid), 0};
}

}  // namespace strata::core
