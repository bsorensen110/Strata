// Audit finding 9: verify parent DFlash2 drafter row sizing (solo = 8 rows / 7 logit rows vs multi-slot batch = 64 rows / 16 logit rows)
// and bind-time logits sizing using host-side allocation arithmetic (no GPU required).
#include "strata/core/dflash2.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using strata::core::Dflash2Drafter;

int g_fail = 0;

void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++g_fail;
    }
}

void check_eq_u64(uint64_t got, uint64_t want, const char* what) {
    if (got != want) {
        std::fprintf(stderr, "FAIL: %s: got=%llu want=%llu\n",
                     what, (unsigned long long) got, (unsigned long long) want);
        ++g_fail;
    }
}

void set_env_var(const char* name, const char* val) {
#if defined(_WIN32)
    _putenv_s(name, val);
#else
    setenv(name, val, 1);
#endif
}

} // namespace

int main() {
    set_env_var("CUDA_VISIBLE_DEVICES", "-1");
    set_env_var("HIP_VISIBLE_DEVICES", "-1");
    set_env_var("STRATA_DF2_HEAD", "1");

    // 1. Slot/row/logits-row sizing rules for parent (blob_shared=false) and clones (blob_shared=true).
    check_eq_u64(Dflash2Drafter::effective_batch_slots(0, false), 1, "eff slots batch=0 parent");
    check_eq_u64(Dflash2Drafter::effective_batch_slots(1, false), 1, "eff slots batch=1 parent");
    check_eq_u64(Dflash2Drafter::effective_batch_slots(2, false), (uint64_t) Dflash2Drafter::kBatchMaxSlots,
                 "eff slots batch=2 parent");
    check_eq_u64(Dflash2Drafter::effective_batch_slots(8, false), (uint64_t) Dflash2Drafter::kBatchMaxSlots,
                 "eff slots batch=8 parent");
    check_eq_u64(Dflash2Drafter::effective_batch_slots(8, true), 1, "eff slots batch=8 clone");

    check_eq_u64(Dflash2Drafter::row_count(0, false), (uint64_t) Dflash2Drafter::kRows, "rows batch=0 parent");
    check_eq_u64(Dflash2Drafter::row_count(1, false), (uint64_t) Dflash2Drafter::kRows, "rows batch=1 parent");
    check_eq_u64(Dflash2Drafter::row_count(2, false),
                 (uint64_t) (Dflash2Drafter::kBatchMaxSlots * Dflash2Drafter::kRows), "rows batch=2 parent");
    check_eq_u64(Dflash2Drafter::row_count(8, false),
                 (uint64_t) (Dflash2Drafter::kBatchMaxSlots * Dflash2Drafter::kRows), "rows batch=8 parent");
    check_eq_u64(Dflash2Drafter::row_count(8, true), (uint64_t) Dflash2Drafter::kRows, "rows batch=8 clone");

    check_eq_u64(Dflash2Drafter::logits_row_count(0, false), (uint64_t) Dflash2Drafter::kDrafts, "logits rows batch=0 parent");
    check_eq_u64(Dflash2Drafter::logits_row_count(1, false), (uint64_t) Dflash2Drafter::kDrafts, "logits rows batch=1 parent");
    check_eq_u64(Dflash2Drafter::logits_row_count(2, false), 16, "logits rows batch=2 parent");
    check_eq_u64(Dflash2Drafter::logits_row_count(8, false), 16, "logits rows batch=8 parent");
    check_eq_u64(Dflash2Drafter::logits_row_count(8, true), (uint64_t) Dflash2Drafter::kDrafts, "logits rows batch=8 clone");

    // 2. Exact allocation delta on Qwen3.8-Flash-Next DFlash2 shape (N=2560, L=5, NH=32, NKV=8, HD=128, I=7680, rank=256, 5 taps).
    Dflash2Drafter::BufferShape solo_shape;
    solo_shape.batch_slots = 1;
    solo_shape.blob_shared = false;

    Dflash2Drafter::BufferShape batch_shape = solo_shape;
    batch_shape.batch_slots = 8;

    Dflash2Drafter::BufferShape clone_shape = solo_shape;
    clone_shape.batch_slots = 8;
    clone_shape.blob_shared = true;

    for (std::size_t ring_elem : {std::size_t{2}, std::size_t{4}}) {
        const auto solo = Dflash2Drafter::alloc_breakdown(solo_shape, ring_elem);
        const auto batch = Dflash2Drafter::alloc_breakdown(batch_shape, ring_elem);
        const auto clone = Dflash2Drafter::alloc_breakdown(clone_shape, ring_elem);

        check_eq_u64(solo.batch_only, 0, "solo batch_only must be 0");
        check_eq_u64(clone.batch_only, 0, "clone batch_only must be 0");
        check(batch.batch_only > 0, "multi-slot batch_only must be non-zero");
        check_eq_u64(solo.row_buffers, clone.row_buffers, "solo parent row_buffers equals clone row_buffers");
        check_eq_u64(solo.ring_buffers, batch.ring_buffers, "ring_buffers independent of batch_slots");
        check_eq_u64(solo.ring_buffers, clone.ring_buffers, "ring_buffers equal between parent and clone");
        check(batch.total_alloc > solo.total_alloc, "batch alloc strictly greater than solo alloc");
        check_eq_u64(batch.total_alloc - solo.total_alloc,
                     (batch.row_buffers - solo.row_buffers) + batch.batch_only,
                     "exact delta equals row_buffers delta plus batch_only");

        const uint64_t saved_alloc = batch.total_alloc - solo.total_alloc;
        if (ring_elem == 4) {
            // Also report logits delta for both native vocab (248320) and typical pruned vocab (48000).
            const uint64_t logits_batch_native = Dflash2Drafter::logits_bytes(8, false, 248320);
            const uint64_t logits_solo_native = Dflash2Drafter::logits_bytes(1, false, 248320);
            const uint64_t saved_logits_native = logits_batch_native - logits_solo_native;

            const uint64_t logits_batch_pruned = Dflash2Drafter::logits_bytes(8, false, 48000);
            const uint64_t logits_solo_pruned = Dflash2Drafter::logits_bytes(1, false, 48000);
            const uint64_t saved_logits_pruned = logits_batch_pruned - logits_solo_pruned;

            std::printf(
                "dflash2_size_test (Flash-Next parent): solo_alloc=%llu B (%.2f MiB) vs batch_alloc=%llu B (%.2f MiB), "
                "saved_alloc=%llu B (%.2f MiB) [row_buf=%llu B, batch_only=%llu B]; "
                "saved_logits_native(248320)=%llu B (%.2f MiB), saved_logits_pruned(48000)=%llu B (%.2f MiB)\n",
                (unsigned long long) solo.total_alloc, (double) solo.total_alloc / (1024.0 * 1024.0),
                (unsigned long long) batch.total_alloc, (double) batch.total_alloc / (1024.0 * 1024.0),
                (unsigned long long) saved_alloc, (double) saved_alloc / (1024.0 * 1024.0),
                (unsigned long long) (batch.row_buffers - solo.row_buffers),
                (unsigned long long) batch.batch_only,
                (unsigned long long) saved_logits_native, (double) saved_logits_native / (1024.0 * 1024.0),
                (unsigned long long) saved_logits_pruned, (double) saved_logits_pruned / (1024.0 * 1024.0));
        }
    }

    // 3. Default-constructed drafter reflects solo parent defaults and bind_bytes matches logits_bytes.
    Dflash2Drafter default_df2;
    check_eq_u64(default_df2.alloc_slots(), 1, "default alloc_slots == 1");
    check_eq_u64(default_df2.alloc_rows(), (uint64_t) Dflash2Drafter::kRows, "default alloc_rows == 8");
    check_eq_u64(default_df2.logits_rows(), (uint64_t) Dflash2Drafter::kDrafts, "default logits_rows == 7");
    check(!default_df2.is_multi_batch(), "default is_multi_batch == false");

    const uint64_t expected_native_bind = (uint64_t) Dflash2Drafter::kDrafts * 248320ull * sizeof(float);
    check_eq_u64(default_df2.bind_bytes(5120, 248320, "", /*head_shared=*/false, /*same_head_device=*/true),
                 expected_native_bind, "solo bind_bytes (native same-device head)");
    check_eq_u64(default_df2.clone_bind_bytes(248320, "", /*same_head_device=*/true),
                 expected_native_bind, "clone_bind_bytes (native same-device head)");

    // Pruned vocab file with STRATA_DF2_HEAD=0: includes gathered head rows + device vocab table when not shared.
    const char* tmp_vocab = "dflash2_size_test.vocab.bin";
    {
        std::FILE* vf = std::fopen(tmp_vocab, "wb");
        check(vf != nullptr, "create temp vocab file");
        if (vf != nullptr) {
            std::vector<int32_t> ids(16000, 0);
            std::fwrite(ids.data(), sizeof(int32_t), ids.size(), vf);
            std::fclose(vf);
        }
    }
    set_env_var("STRATA_DF2_HEAD", "0");
    check_eq_u64(Dflash2Drafter::resolve_dvocab(248320, tmp_vocab, true), 16000, "resolve_dvocab reads pruned file");
    const uint64_t expected_pruned_logits = (uint64_t) Dflash2Drafter::kDrafts * 16000ull * sizeof(float);
    const uint64_t expected_pruned_head = 16000ull * 5120ull + 16000ull * 4ull;
    check_eq_u64(default_df2.bind_bytes(5120, 248320, tmp_vocab, /*head_shared=*/false, /*same_head_device=*/true),
                 expected_pruned_logits + expected_pruned_head, "solo bind_bytes (pruned unshared)");
    check_eq_u64(default_df2.bind_bytes(5120, 248320, tmp_vocab, /*head_shared=*/true, /*same_head_device=*/true),
                 expected_pruned_logits, "solo bind_bytes (pruned shared)");
    std::remove(tmp_vocab);

    if (g_fail != 0) {
        std::printf("dflash2_size_test: %d FAILED\n", g_fail);
        return 1;
    }
    std::printf("dflash2_size_test: all passed\n");
    return 0;
}
