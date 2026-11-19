// include/strata/core/batch_route_affinity.hpp - router-affinity micro-batch slot grouping.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::core {

inline int affinity_popcount64(uint64_t x) noexcept {
    int c = 0;
    while (x != 0) {
        x &= (x - 1);
        ++c;
    }
    return c;
}

/// Jaccard similarity [0, 1] between two 64-bit recent-expert routing signatures.
inline double route_affinity_jaccard(uint64_t a, uint64_t b) noexcept {
    const uint64_t u = a | b;
    if (u == 0) return 0.0;
    return (double) affinity_popcount64(a & b) / (double) affinity_popcount64(u);
}

/// Orders `active_slots` into affinity groups of size <= `max_group_size` by greedily picking a seed slot
/// and adding the remaining unassigned slots with highest overlap to the current group's combined signature.
/// Returns `active_slots` unchanged when `max_group_size <= 0`, `active_slots.size() <= max_group_size`, or
/// when all signatures are zero.
inline std::vector<int> order_slots_by_route_affinity(const std::vector<int>& active_slots,
                                                      const uint64_t* sigs,
                                                      size_t n_sigs,
                                                      int max_group_size) {
    if (max_group_size <= 0 || active_slots.size() <= (size_t) max_group_size || !sigs) return active_slots;
    bool any_nonzero = false;
    for (int s : active_slots) {
        if (s >= 0 && (size_t) s < n_sigs && sigs[s] != 0) { any_nonzero = true; break; }
    }
    if (!any_nonzero) return active_slots;

    const size_t n = active_slots.size();
    std::vector<uint8_t> used(n, 0);
    std::vector<int> ordered;
    ordered.reserve(n);

    for (size_t done = 0; done < n;) {
        size_t seed = 0;
        while (seed < n && used[seed]) ++seed;
        used[seed] = 1;
        ordered.push_back(active_slots[seed]);
        ++done;
        uint64_t grp_sig = (active_slots[seed] >= 0 && (size_t) active_slots[seed] < n_sigs)
                               ? sigs[active_slots[seed]]
                               : 0ull;
        for (int g = 1; g < max_group_size && done < n; ++g) {
            size_t best_idx = n;
            double best_sim = -1.0;
            int best_inter = -1;
            for (size_t j = 0; j < n; ++j) {
                if (used[j]) continue;
                const int sj = active_slots[j];
                const uint64_t jsig = (sj >= 0 && (size_t) sj < n_sigs) ? sigs[sj] : 0ull;
                const double sim = route_affinity_jaccard(grp_sig, jsig);
                const int inter = affinity_popcount64(grp_sig & jsig);
                if (best_idx == n || sim > best_sim || (sim == best_sim && inter > best_inter)) {
                    best_idx = j;
                    best_sim = sim;
                    best_inter = inter;
                }
            }
            if (best_idx >= n) break;
            used[best_idx] = 1;
            const int sb = active_slots[best_idx];
            if (sb >= 0 && (size_t) sb < n_sigs) grp_sig |= sigs[sb];
            ordered.push_back(sb);
            ++done;
        }
    }
    return ordered;
}

}  // namespace strata::core
