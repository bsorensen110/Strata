// Host-only opt-in grouping permutation. No HIP calls, device math, or precision changes.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::prefill::detail {

inline bool mmq_resident_sort_flag(const char* value) {
    return value != nullptr && std::strcmp(value, "1") == 0;
}

inline bool mmq_resident_sort_requested() {
    static const bool on = mmq_resident_sort_flag(std::getenv("STRATA_MMQ_RESIDENT_SORT_NE"));
    return on;
}

// STRATA_MMQ_GROUP_SORT=1: the same row-count order for the full 16-expert groups only; the last, partial group keeps
// its experts in ID order. A full group's launch has 160 * ntx (gate/up) or 320 * ntx (down) tiles, at or above the
// 90 % tile efficiency on the cards' SM counts, so it runs one block per tile with no stream-k K split whatever its tile
// width: each expert's rows get the same bits in any full group. The partial group can take the stream-k split, whose
// K boundaries depend on the group's tile count - leaving it unchanged keeps the prompt's bits.
inline bool mmq_group_sort_requested() {
    static const bool on = mmq_resident_sort_flag(std::getenv("STRATA_MMQ_GROUP_SORT"));
    return on;
}

/// STRATA_MMQ_STAGED_SORT: in the staged walk (`!stream_all`), mode 1 places full resident MMQ groups first (sorted by
/// row count) and sorts the remaining full groups by row count; mode 2 sorts all full groups purely by row count.
inline int mmq_staged_sort_mode_from(const char* v, int default_mode = 0) {
    if (v == nullptr || v[0] == '\0') return default_mode;
    return std::atoi(v);
}

inline bool mmq_staged_sort_from(const char* v, bool default_on = false) {
    return mmq_staged_sort_mode_from(v, default_on ? 1 : 0) != 0;
}

inline int mmq_staged_sort_mode(bool default_on = false) {
    return mmq_staged_sort_mode_from(std::getenv("STRATA_MMQ_STAGED_SORT"), default_on ? 3 : 0);
}

inline bool mmq_staged_sort_requested(bool default_on = false) {
    return mmq_staged_sort_mode(default_on) != 0;
}

// How many of `n` ordered experts the sort may move: all of them (STRATA_MMQ_RESIDENT_SORT_NE), or the full groups
// of `group` experts (STRATA_MMQ_GROUP_SORT).
inline size_t mmq_resident_sort_span(size_t n, size_t group, bool keep_tail) {
    return keep_tail && group > 0 ? n - n % group : n;
}

inline bool mmq_resident_sort_eligible(bool requested, bool use_mmq, bool native_layout, bool has_cache,
                                      const int32_t* layer_resident, int64_t n_expert,
                                      bool stream_all, bool layer_stream_empty) {
    // native-layout packs too: slot/src/off are built the same way for both layouts, and nothing reads off[e + 1] as a
    // count (4x GP100, sm_60: 1.2K prompt 7.27 -> 5.32 s, 35K 63.3 -> 54.1 s, the same bits)
    (void) native_layout;
    if (!requested || !use_mmq || !has_cache || !layer_resident || n_expert <= 0) return false;
    // All experts, including unselected ones: leave every partial/streamed layer in ID order.
    if (stream_all && !layer_stream_empty) return false;
    for (int64_t e = 0; e < n_expert; ++e) if (layer_resident[e] < 0) return false;
    return true;
}

inline bool mmq_staged_sort_eligible(bool requested, bool use_mmq, bool native_layout, bool has_cache,
                                     const int32_t* layer_resident, int64_t n_expert, bool stream_all) {
    (void) native_layout;
    if (!requested || !use_mmq || !has_cache || !layer_resident || n_expert <= 0 || stream_all) return false;
    return true;
}

inline void mmq_rebuild_routes(const int32_t* ids, int64_t routes, int32_t k,
                               const std::vector<int32_t>& count, std::vector<int32_t>& off,
                               const std::vector<int32_t>& order, int32_t* slot, int32_t* src) {
    const int32_t prev_end = off.empty() ? 0 : off.back();
    int32_t cursor = 0;
    for (int32_t e : order) {
        off[(size_t) e] = cursor;
        cursor += count[(size_t) e];
    }
    for (size_t e = 0; e < count.size(); ++e)
        if (count[e] == 0) off[e] = 0;
    if (!off.empty()) off.back() = std::max(prev_end, cursor);
    std::vector<int32_t> fill(off.begin(), off.end() - 1);
    // Preserve token-major/rank-major traversal, hence the original within-expert row order.
    for (int64_t i = 0; i < routes; ++i) {
        const int32_t e = ids[(size_t) i];
        const int32_t p = fill[(size_t) e]++;
        slot[(size_t) i] = p;
        src[(size_t) p] = (int32_t) (i / k);
    }
}

// Preconditions are established by the existing caller: IDs were range-checked and counted;
// order contains each positive-count expert once in ID order; off has count.size()+1 entries;
// routes fits its original int32 offsets, and slot/src each have routes writable entries.
// After sorting, off[e] is an expert-indexed START, not an ID-prefix end at off[e+1].
// Only order[0, span) is sorted (span >= order.size(): all of it); the entries after it keep their positions.
inline void mmq_resident_sort_rows(const int32_t* ids, int64_t routes, int32_t k,
                                   const std::vector<int32_t>& count, std::vector<int32_t>& off,
                                   std::vector<int32_t>& order, int32_t* slot, int32_t* src,
                                   size_t span = SIZE_MAX) {
    std::stable_sort(order.begin(), order.begin() + (ptrdiff_t) std::min(span, order.size()), [&](int32_t a, int32_t b) {
        return count[(size_t) a] != count[(size_t) b]
            ? count[(size_t) a] < count[(size_t) b] : a < b;
    });
    mmq_rebuild_routes(ids, routes, k, count, off, order, slot, src);
}

inline void mmq_staged_sort_rows(const int32_t* ids, int64_t routes, int32_t k,
                                 const std::vector<int32_t>& count, const int32_t* layer_resident,
                                 size_t group, bool keep_tail,
                                 std::vector<int32_t>& off, std::vector<int32_t>& order,
                                 int32_t* slot, int32_t* src, int sort_mode = 1) {
    const size_t span = mmq_resident_sort_span(order.size(), group, keep_tail);
    auto cmp_cnt = [&](int32_t a, int32_t b) {
        return count[(size_t) a] != count[(size_t) b]
            ? count[(size_t) a] < count[(size_t) b] : a < b;
    };
    if (sort_mode == 1 && layer_resident != nullptr && group > 0 && span > 0) {
        auto end_it = order.begin() + (ptrdiff_t) span;
        auto mid_it = std::stable_partition(order.begin(), end_it, [&](int32_t e) {
            return layer_resident[(size_t) e] >= 0;
        });
        std::stable_sort(order.begin(), mid_it, cmp_cnt);
        const size_t n_res = (size_t) (mid_it - order.begin());
        const size_t res_full = (n_res / group) * group;
        std::stable_sort(order.begin() + (ptrdiff_t) res_full, end_it, cmp_cnt);
    } else if (sort_mode == 3 && layer_resident != nullptr && group > 0 && span > 0) {
        std::stable_sort(order.begin(), order.begin() + (ptrdiff_t) span, cmp_cnt);
        const size_t n_grp = span / group;
        std::vector<int32_t> reordered;
        reordered.reserve(span);
        // 1. Pure-resident full groups first (ascending count) so m.copy pre-stages up to STAGE_GRP non-resident blobs
        for (size_t g = 0; g < n_grp; ++g) {
            bool all_res = true;
            for (size_t q = 0; q < group; ++q)
                if (layer_resident[(size_t) order[g * group + q]] < 0) { all_res = false; break; }
            if (all_res)
                reordered.insert(reordered.end(), order.begin() + (ptrdiff_t) (g * group),
                                 order.begin() + (ptrdiff_t) ((g + 1) * group));
        }
        // 2. Staged full groups in descending count order so heavy GPU compute overlaps with PCIe copies and light groups finish last
        for (size_t g = n_grp; g > 0; --g) {
            const size_t gi = g - 1;
            bool all_res = true;
            for (size_t q = 0; q < group; ++q)
                if (layer_resident[(size_t) order[gi * group + q]] < 0) { all_res = false; break; }
            if (!all_res)
                reordered.insert(reordered.end(), order.begin() + (ptrdiff_t) (gi * group),
                                 order.begin() + (ptrdiff_t) ((gi + 1) * group));
        }
        std::copy(reordered.begin(), reordered.end(), order.begin());
    } else {
        std::stable_sort(order.begin(), order.begin() + (ptrdiff_t) std::min(span, order.size()), cmp_cnt);
    }
    mmq_rebuild_routes(ids, routes, k, count, off, order, slot, src);
}

}  // namespace strata::prefill::detail
