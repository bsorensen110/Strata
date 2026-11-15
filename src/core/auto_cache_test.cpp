#include "strata/core/auto_cache.hpp"
#include "strata/core/native_dense.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace strata::core;
namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
constexpr int64_t MiB = 1ll << 20;
}

int main() {
    const int64_t blob = 2560ll * 1024;   // 2.5 MiB
    const int64_t min_slots = 300;
    // the A4000 of #4474: 9951 MiB free after the weights, 700 reserved + 184 draft head
    const int64_t free_b = 9951 * MiB, reserve_b = (700 + 184) * MiB;
    const AutoCacheSlots a = auto_cache_slots(free_b, reserve_b, blob, min_slots);
    const AutoCacheSlots plain = auto_cache_slots(free_b, reserve_b, blob, min_slots, 0);
    check(a.late_priced && !plain.late_priced, "the late allocation is priced by default and off with late_mib 0");
    check(plain.slots == (free_b - reserve_b) / blob, "late_mib 0 is the old sizing");
    check(a.slots == (free_b - reserve_b - kLateAllocMib * MiB) / blob, "the cache leaves reserve + late allocation");
    check(plain.slots - a.slots == kLateAllocMib * MiB / blob || plain.slots - a.slots == kLateAllocMib * MiB / blob + 1,
          "256 MiB fewer bytes: about 100 slots of 2.5 MiB");
    // the card ends with the reserve: free - slots*blob - priced - 244 MiB unpriced >= reserve
    check(free_b - a.slots * blob - (184 + 244) * MiB >= 700 * MiB - blob, "the end free is the reserve, not 241 MiB under it");
    // a small card: not enough for min_slots after the late allocation -> the old sizing (#496 lowers the reserve elsewhere)
    const AutoCacheSlots small = auto_cache_slots(1200 * MiB, 700 * MiB, blob, min_slots);
    check(!small.late_priced && small.slots == (500 * MiB) / blob, "a card that cannot spare it keeps the old number");
    const AutoCacheSlots tiny = auto_cache_slots(900 * MiB, 700 * MiB, blob, min_slots);
    check(!tiny.late_priced && tiny.slots == 80, "less than min_slots stays as it was");
    check(auto_cache_slots(500 * MiB, 700 * MiB, blob, min_slots).slots == 0, "a negative room is no slots");
    check(auto_cache_slots(1000 * MiB, 700 * MiB, 0, min_slots).slots == 0, "no blob size: no slots, no division");
    // exactly at the limit: min_slots after the late allocation
    const int64_t edge_free = reserve_b + (kLateAllocMib * MiB) + min_slots * blob;
    check(auto_cache_slots(edge_free, reserve_b, blob, min_slots).late_priced, "exactly min_slots after the late allocation: priced");
    check(!auto_cache_slots(edge_free - 1, reserve_b, blob, min_slots).late_priced, "one byte less: the old sizing");
    // Audit finding 10: pricing a NativeDense stage as one contiguous 256B-aligned arena (one 2 MiB granule)
    // instead of one cudaMalloc per matrix returns the saved granule slack directly to auto_cache_slots.
    const std::vector<uint64_t> hc_192(192, 2560ull / 32ull * 34ull * 1280ull);   // 192 x ~3.32 MiB HC Q8_0
    uint64_t per_matrix_bytes = 0;
    for (uint64_t s : hc_192) per_matrix_bytes += NativeDense::alloc_bytes(s);
    const uint64_t arena_bytes = NativeDense::arena_alloc_bytes(hc_192);
    const int64_t saved_granules = (int64_t) (per_matrix_bytes - arena_bytes);
    check(saved_granules >= 128 * MiB, "arena pricing reclaims >= 128 MiB of 2 MiB granule slack on 192 HC tensors");
    const AutoCacheSlots before_arena = auto_cache_slots(free_b - per_matrix_bytes, reserve_b, blob, min_slots);
    const AutoCacheSlots after_arena = auto_cache_slots(free_b - arena_bytes, reserve_b, blob, min_slots);
    check(after_arena.slots - before_arena.slots == saved_granules / blob ||
              after_arena.slots - before_arena.slots == saved_granules / blob + 1,
          "reclaimed NativeDense arena granules convert directly into expert cache slots");

    std::printf("auto_cache_test: %d checks passed\n", checks);
    return 0;
}
