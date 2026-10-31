#pragma once
// STRATA_BATCH_MTP_SELECT (opt-in): which slots a --batch-mtp window carries when it cannot carry every active slot
// (more slots than the window has room for at two rows each).  Today the windows take the slots in turn
// (round-robin).  With the switch, a window first takes the longest-waiting slots it must take so that no slot waits
// more than `starve` windows (or ceil(slots / window) - 1, if that is more: the window cannot do better); it fills the rest with the slots whose drafts the target accepts most often (their
// acceptance EMA, STRATA_BATCH_MTP_DEPTH's), and, with STRATA_BATCH_MTP_SELECT_OVERLAP, with the slots whose
// experts overlap most with those already chosen (fewer distinct experts per layer: less CPU and PCIe work).
//
// Which slot runs when never changes what a slot emits: each window's rows are judged by the target alone.
// CPU-only rules, so they have a unit test (tests/core/batch_mtp_select_test.cpp).
#include <cstdint>
#include <algorithm>
#include <vector>
#include <cstdlib>

namespace strata::core {

/// A slot's routed experts in its last window, as a 1024-bit set of (layer, expert) hashes.
constexpr int kRouteSigWords = 16;
struct RouteSig {
    uint64_t w[kRouteSigWords] = {};
};

inline void route_sig_clear(RouteSig& s) {
    for (uint64_t& x : s.w) x = 0;
}

inline void route_sig_add(RouteSig& s, int64_t layer, int32_t expert) {
    if (expert < 0) return;
    uint64_t h = (uint64_t) layer * 0x9E3779B97F4A7C15ull ^ (uint64_t) (uint32_t) expert * 0xC2B2AE3D27D4EB4Full;
    h ^= h >> 29;
    h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 32;
    const unsigned bit = (unsigned) (h % (uint64_t) (64 * kRouteSigWords));
    s.w[bit / 64] |= 1ull << (bit % 64);
}

inline int route_sig_popcount64(uint64_t x) {
    int n = 0;
    while (x != 0) { x &= x - 1; ++n; }
    return n;
}

inline int route_sig_count(const RouteSig& s) {
    int n = 0;
    for (uint64_t x : s.w) n += route_sig_popcount64(x);
    return n;
}

inline int route_sig_common(const RouteSig& a, const RouteSig& b) {
    int n = 0;
    for (int i = 0; i < kRouteSigWords; ++i) n += route_sig_popcount64(a.w[i] & b.w[i]);
    return n;
}

inline void route_sig_or(RouteSig& into, const RouteSig& s) {
    for (int i = 0; i < kRouteSigWords; ++i) into.w[i] |= s.w[i];
}

struct MtpSelectConfig {
    bool on = false;         ///< STRATA_BATCH_MTP_SELECT
    float overlap_w = 0.0f;  ///< STRATA_BATCH_MTP_SELECT_OVERLAP: the weight of the expert overlap (0: off)
    int starve = 2;          ///< a slot that waited this many windows goes first
};

/// sel_env: STRATA_BATCH_MTP_SELECT (unset, "" or "0": off; "1": on; N >= 2: on, a slot waits at most N windows).
/// overlap_env: STRATA_BATCH_MTP_SELECT_OVERLAP (the weight; "1" means 1.0; unset: 0).
inline MtpSelectConfig mtp_select_config(const char* sel_env, const char* overlap_env) {
    MtpSelectConfig c;
    if (sel_env == nullptr || sel_env[0] == '\0') return c;
    const int v = std::atoi(sel_env);
    if (v <= 0) return c;
    c.on = true;
    c.starve = v >= 2 ? (v > 64 ? 64 : v) : 2;
    if (overlap_env != nullptr && overlap_env[0] != '\0') {
        const double w = std::atof(overlap_env);
        c.overlap_w = w > 0.0 ? (float) (w > 8.0 ? 8.0 : w) : 0.0f;
    }
    return c;
}

/// `n` candidates in the rotation order, at most `m` chosen; `wait[i]` windows since candidate i last ran, `ema[i]` its
/// acceptance EMA, `sig[i]` its last routing (null: unknown).  Writes the chosen candidates' indices to `out` in the
/// rotation order and returns how many.  Off, or with room for every candidate: the first `m` in the rotation order.
inline int mtp_select_slots(int n, const int* wait, const float* ema, const RouteSig* const* sig, int m,
                            const MtpSelectConfig& c, int* out) {
    if (n <= 0 || m <= 0) return 0;
    if (!c.on || n <= m) {
        const int k = n < m ? n : m;
        for (int i = 0; i < k; ++i) out[i] = i;
        return k;
    }
    constexpr int kMax = 64;
    if (n > kMax) n = kMax;
    bool chosen[kMax] = {};
    int count = 0;
    RouteSig uni;
    bool uni_any = false;
    auto take = [&](int i) {
        chosen[i] = true;
        ++count;
        if (sig != nullptr && sig[i] != nullptr) {
            route_sig_or(uni, *sig[i]);
            uni_any = true;
        }
    };
    // Deadlines first.  A slot that waited w windows must run within `bound - w` more windows; the bound can be no
    // tighter than the window allows (n slots, m a window: some slot waits ceil(n / m) - 1 windows).  Taking the
    // longest waits first, this window must take r of them, the most that any deadline t needs:
    // r = max over t of (slots due by t) - m * t.  The rest of the window goes to the best scores.
    const int fair = (n + m - 1) / m - 1;
    const int bound = c.starve > fair ? c.starve : fair;
    std::vector<int> by_wait((size_t) n);
    for (int i = 0; i < n; ++i) by_wait[(size_t) i] = i;
    std::stable_sort(by_wait.begin(), by_wait.end(), [&](int x, int y) { return wait[x] > wait[y]; });
    int r = 0;
    for (int k = 0; k < n; ++k) {
        const int d = bound - wait[by_wait[(size_t) k]];
        const int need = k + 1 - m * (d > 0 ? d : 0);
        if (need > r) r = need;
    }
    for (int k = 0; k < n && k < r && count < m; ++k) take(by_wait[(size_t) k]);
    while (count < m) {
        int best = -1;
        float best_s = 0.0f;
        for (int i = 0; i < n; ++i) {
            if (chosen[i]) continue;
            float s = ema[i] + 1e-3f * (float) (wait[i] < 1000 ? wait[i] : 1000);
            if (c.overlap_w > 0.0f && uni_any && sig != nullptr && sig[i] != nullptr) {
                const int tot = route_sig_count(*sig[i]);
                if (tot > 0) s += c.overlap_w * (float) route_sig_common(*sig[i], uni) / (float) tot;
            }
            if (best < 0 || s > best_s) { best = i; best_s = s; }   // (a tie keeps the earlier one in the rotation)
        }
        if (best < 0) break;
        take(best);
    }
    int k = 0;
    for (int i = 0; i < n; ++i)
        if (chosen[i]) out[k++] = i;
    return k;
}

}  // namespace strata::core
