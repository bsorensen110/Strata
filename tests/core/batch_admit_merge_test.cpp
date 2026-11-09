#include "strata/core/batch_admit_merge.hpp"

#include <cmath>
#include <cstdio>

using namespace strata::core;

static int g_fail = 0;
static void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_fail;
    }
}

int main() {
    // 1. All-resident zero-doorbell path merges up to mt tokens from the tail.
    check(admit_merge_from(0, 47, 48, 16, true, false, 16) == 32, "all_ar merges last 16 tokens");
    check(admit_merge_from(40, 47, 48, 16, true, false, 16) == 40, "all_ar clamps at a0");

    // 2. Single-stage hybrid MoE merges the partial read_windows remainder + final token when rem + 1 <= mt.
    check(admit_merge_from(4703, 4711, 4712, 16, false, true, 16) == 4703,
          "4712-token chat prompt tail [4703, 4711) merges 8+1=9 rows into BGEN admission window");
    check(admit_merge_from(0, 47, 48, 16, false, true, 16) == 32,
          "48-token prompt [0, 47) leaves 2x16 to read_windows and merges 15+1=16 into BGEN admission");
    check(admit_merge_from(0, 61, 62, 16, false, true, 16) == 48,
          "62-token prompt [0, 61) leaves 3x16 to read_windows and merges 13+1=14 into BGEN admission");
    check(admit_merge_from(0, 32, 33, 16, false, true, 16) == -1,
          "exact multiple of win_w (rem == 0) does not split a full read_windows batch");
    check(admit_merge_from(0, 15, 16, 8, false, true, 16) == -1,
          "rem + 1 > mt does not merge");
    check(admit_merge_from(0, 47, 48, 16, false, false, 16) == -1,
          "multi-stage non-all_ar does not merge");
    check(admit_merge_from(0, 47, 48, 1, true, true, 16) == -1,
          "mt < 2 never merges");

    // 3. Inter-chunk decode budget subtracts mid-chunk layer-yield decode time when net_share is enabled.
    const double b_net = inter_chunk_decode_budget_ms(0.5, 1238.0, 88.0, true);
    check(std::fabs(b_net - 487.0) < 1e-6,
          "net inter-chunk decode budget subtracts mid-chunk decode from both chunk wall time and target budget");
    check(inter_chunk_decode_budget_ms(0.5, 300.0, 220.0, true) == 0.0,
          "net inter-chunk decode budget clamps to 0 when mid-chunk yields already satisfied decode_share");
    const double b_legacy = inter_chunk_decode_budget_ms(0.5, 1238.0, 88.0, false);
    check(std::fabs(b_legacy - 619.0) < 1e-6,
          "legacy inter-chunk decode budget preserves wall_ms * eff_share when net_share is off");

    // 7. dflash2_ring_copy_slots clamps copy width to [0, k_ring].
    check(dflash2_ring_copy_slots(0, 2064) == 0, "empty ring copies 0 slots");
    check(dflash2_ring_copy_slots(62, 2064) == 62, "62-token conversation copies only 62 of 2064 ring slots");
    check(dflash2_ring_copy_slots(274, 2064) == 274, "274-token solo handoff copies only 274 of 2064 ring slots");
    check(dflash2_ring_copy_slots(4712, 2064) == 2064, "wrapped ring copies full 2064 slots");

    // 8. should_evict_batch_graph_for_vram prevents ROCm <512 MiB GTT paging (#2065).
    const size_t mib = 1024ULL * 1024ULL;
    check(should_evict_batch_graph_for_vram(510 * mib, 544 * mib, 16),
          "evicts LRU batch graph when free VRAM is below 544 MiB floor");
    check(!should_evict_batch_graph_for_vram(560 * mib, 544 * mib, 16),
          "keeps batch graph when free VRAM is above 544 MiB floor");
    check(!should_evict_batch_graph_for_vram(544 * mib, 544 * mib, 16),
          "keeps batch graph when free VRAM equals 544 MiB floor");
    check(!should_evict_batch_graph_for_vram(500 * mib, 0, 16),
          "disabled when floor is 0");
    check(!should_evict_batch_graph_for_vram(500 * mib, 544 * mib, 0),
          "does not evict when no batch graphs are cached");

    if (g_fail == 0) std::printf("batch_admit_merge_test: all checks passed\n");
    return g_fail == 0 ? 0 : 1;
}
