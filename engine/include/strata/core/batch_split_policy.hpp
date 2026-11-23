// include/strata/core/batch_split_policy.hpp - slot-boundary G=2 split helper for wide batch verify windows.
#pragma once

#include <cstdlib>

namespace strata::core {

/// Opt-in gate for splitting wide (T > 8) batch verify windows into two sequential groups at a slot boundary.
inline bool batch_split_enabled() noexcept {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_BATCH_SPLIT");
        return v == nullptr || v[0] == '\0' || std::atoi(v) != 0;
    }();
    return on;
}

/// Finds a slot-boundary split point `s` in `[1, T - 1]` for a batch window of `T` rows so that both `[0, s)` and
/// `[s, T)` keep every slot's rows contiguous in a single group and hold at most `max_group` rows, preferring the
/// split closest to `T / 2` (ties broken toward smaller `s`). Returns 0 when no valid slot-boundary split exists.
inline int batch_split_point(const int* slots, int T, int max_group = 8) noexcept {
    if (slots == nullptr || max_group <= 0 || T <= max_group || T > 2 * max_group) return 0;
    int best_s = 0;
    int best_dist = T + 1;
    for (int s = 1; s < T; ++s) {
        if (slots[s - 1] == slots[s]) continue;
        if (s > max_group || (T - s) > max_group) continue;
        const int rem = T - s;
        const int dist = s >= rem ? s - rem : rem - s;
        if (dist < best_dist || (dist == best_dist && s < best_s)) {
            best_dist = dist;
            best_s = s;
        }
    }
    return best_s;
}

}  // namespace strata::core
