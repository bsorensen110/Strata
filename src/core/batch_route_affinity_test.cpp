// src/core/batch_route_affinity_test.cpp - unit test for router-affinity micro-batch grouping (6C).
#include "strata/core/batch_route_affinity.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    using strata::core::order_slots_by_route_affinity;
    using strata::core::route_affinity_jaccard;

    assert(route_affinity_jaccard(0ull, 0ull) == 0.0);
    assert(route_affinity_jaccard(0xFFull, 0xFFull) == 1.0);
    assert(route_affinity_jaccard(0x0Full, 0xF0ull) == 0.0);

    // 4 active slots in mixed order: {0, 1, 2, 3}, where (0, 2) share expert cluster A and (1, 3) share cluster B.
    const std::vector<int> active = {0, 1, 2, 3};
    const uint64_t sigs[4] = {
        0x000000FFull,   // slot 0: cluster A
        0xFF000000ull,   // slot 1: cluster B
        0x000000FEull,   // slot 2: heavy overlap with slot 0
        0xFE000000ull,   // slot 3: heavy overlap with slot 1
    };
    const auto grouped = order_slots_by_route_affinity(active, sigs, 4, 2);
    assert(grouped.size() == 4);
    assert(grouped[0] == 0 && grouped[1] == 2);
    assert(grouped[2] == 1 && grouped[3] == 3);

    // All-zero signatures preserve original order.
    const uint64_t zero_sigs[4] = {};
    assert(order_slots_by_route_affinity(active, zero_sigs, 4, 2) == active);

    // Single group covering all active slots preserves original order.
    assert(order_slots_by_route_affinity(active, sigs, 4, 4) == active);

    std::printf("batch_route_affinity_test: ok\n");
    return 0;
}
