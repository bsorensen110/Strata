#include "strata/core/auto_cache.hpp"
#include "strata/core/dflash2.hpp"
#include "strata/core/geometry.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/session.hpp"
#include "strata/core/verify.hpp"
#include "strata/prefill/prefill.hpp"

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

    // Audit finding 8: exact DFlash2 + Verifier bind pricing from single-source allocator formulas.
    const ModelGeometry g{};
    QsaState qsa{};
    qsa.max_cells = 32768;
    SessionState ss{};
    ss.k = (int) g.n_expert_used;
    ss.max_cells = 32768;
    ss.qsa_states = &qsa;
    SessionState ss_fallback{};
    ss_fallback.k = (int) g.n_expert_used;
    ss_fallback.max_cells = 32768;

    const std::vector<int> taps = {5, 15, 25, 35, 45};
    const uint64_t prompt_taps_b = strata::prefill::Prefill::tap_bytes(taps, 2048, g.n_embd);
    check(prompt_taps_b == 5ull * 2048ull * 2560ull * 4ull, "Prefill::tap_bytes matches 2048x5x2560xf32 (100 MiB)");
    check(prompt_taps_b == 100ull * (uint64_t) MiB, "solo prompt taps exact 100 MiB instead of hardcoded 112 MiB");
    check(strata::prefill::Prefill::tap_bytes(taps, 2048, g.n_embd, 0, 24) +
                  strata::prefill::Prefill::tap_bytes(taps, 2048, g.n_embd, 24, 48) ==
              prompt_taps_b,
          "Prefill::tap_bytes partitions cleanly across stage layer ranges");

    const uint64_t ver_taps_t8_b = Verifier::tap_bytes(taps, 8, g.n_embd, 0, g.n_layers, g.n_layers);
    const uint64_t ver_taps_t16_b = Verifier::tap_bytes(taps, 16, g.n_embd, 0, g.n_layers, g.n_layers);
    check(ver_taps_t8_b == 8ull * 5ull * 2560ull * 4ull, "Verifier::tap_bytes T=8 exact");
    check(ver_taps_t16_b == 2ull * ver_taps_t8_b, "Verifier::tap_bytes scales linearly with max_t");

    const uint64_t df2_logits_solo_b = Dflash2Drafter::logits_bytes(1, /*blob_shared=*/false, g.n_vocab);
    const uint64_t df2_logits_batch_b = Dflash2Drafter::logits_bytes(8, /*blob_shared=*/false, g.n_vocab);
    const Dflash2Drafter df2_default;
    check(df2_default.bind_bytes(0, g.n_vocab, "", /*head_shared=*/false, true) == df2_logits_solo_b,
          "default solo Dflash2Drafter::bind_bytes matches 7-row native logits_bytes");
    check(df2_logits_solo_b == 7ull * (uint64_t) g.n_vocab * 4ull, "solo parent logits_bytes uses 7 draft rows");
    check(Dflash2Drafter::logits_bytes(1, false, 48000) == 7ull * 48000ull * 4ull,
          "pruned-vocab solo logits_bytes uses 7 x 48000 f32");
    check(df2_logits_batch_b == 16ull * (uint64_t) g.n_vocab * 4ull, "batch parent logits_bytes uses 16 draft rows");

    const int64_t iq4_xs_max_blob = 2867200;   // ~2.734 MiB max expert blob in live Flash-Next IQ4_XS pack
    const uint64_t staging_blobs = (uint64_t) Verifier::staging_blobs_cap();
    const uint64_t ver_arena_t8_b = Verifier::arena_bytes(g, ss, g.n_vocab, 8, iq4_xs_max_blob, 0);
    const uint64_t ver_arena_t16_b = Verifier::arena_bytes(g, ss, g.n_vocab, 16, iq4_xs_max_blob, 0);
    const uint64_t arena_major_floor_b =
        staging_blobs * (uint64_t) iq4_xs_max_blob +                                  // staging_
        2ull * (uint64_t) g.n_gdn_layers() * 8ull * (uint64_t) g.ssm_conv_channels * 4ull +   // qkv_L_ + h_L_
        8ull * (uint64_t) g.n_vocab * 4ull +                                        // head_logits_
        2ull * 8ull * (uint64_t) g.hc * (uint64_t) g.n_embd * 4ull;                 // R_ + xn_
    check(ver_arena_t8_b > arena_major_floor_b,
          "Verifier::arena_bytes exceeds sum of major independent constituent buffers");
    check(ver_arena_t8_b == Verifier::arena_bytes(g, ss_fallback, g.n_vocab, 8, iq4_xs_max_blob, 0),
          "Verifier::arena_bytes falls back cleanly when qsa_states is null");
    const uint64_t staging_b = staging_blobs * (uint64_t) iq4_xs_max_blob;
    check(ver_arena_t8_b >= staging_b + (uint64_t) (21 * MiB) &&
              ver_arena_t8_b <= staging_b + (uint64_t) (52 * MiB),
          "Verifier::arena_bytes T=8 falls in measured staging + 21-52 MiB window");
    check(ver_arena_t16_b > ver_arena_t8_b, "Verifier::arena_bytes grows monotonically with max_t");

    const uint64_t exact_solo_df2_b =
        prompt_taps_b + ver_taps_t8_b + df2_default.bind_bytes(0, g.n_vocab, "", false, true) + ver_arena_t8_b;
    const AutoCacheSlots solo_slots =
        auto_cache_slots(free_b, (int64_t) (700 * MiB + exact_solo_df2_b), blob, min_slots);
    check(solo_slots.late_priced && solo_slots.slots > 0, "solo DFlash2 pricing preserves kLateAllocMib invariant");

    std::printf(
        "auto_cache_test: %d checks passed (solo df2 T=8 exact=%.2f MiB: prompt_taps=%.2f, ver_taps=%.2f, "
        "df2_logits=%.2f, ver_arena=%.2f; T=16 ver_arena=%.2f MiB)\n",
        checks, (double) exact_solo_df2_b / (double) MiB, (double) prompt_taps_b / (double) MiB,
        (double) ver_taps_t8_b / (double) MiB, (double) df2_logits_solo_b / (double) MiB,
        (double) ver_arena_t8_b / (double) MiB, (double) ver_arena_t16_b / (double) MiB);
    return 0;
}
