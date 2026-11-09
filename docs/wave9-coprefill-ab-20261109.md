# Wave 9 Co-Prefill + Decode & Slot Handoff Optimization Report (2026-11-09)

## Overview

Wave 9 targets DFlash2 ring watermark accounting across conversation switches, `conversation_snapshot_save` (`park_slot` / `park_current`) host allocation + stream synchronization overhead, DFlash2 slot clone graph VRAM footprint (`+46..74 MiB` free VRAM), startup prewarming of 1-slot precomputed batch graphs (`batch_full_1_exec_`), overlap-safe tail shifts in `draft_df2`, and solo request preemption before wasted end-of-window draft launches.

### Atomic Commits (Wave 9)

1. **`core: add Wave 9 DFlash2 lineage, overlap, and clone graph helpers`** (`include/strata/core/batch_admit_merge.hpp`, `tests/core/batch_admit_merge_test.cpp`)
   - `dflash2_begin_request_covered(lcp_cur, resume, cur_hist_size)`: returns `false` on cold starts (`resume <= 0`) when `cur_hist_size > 0` so the previous conversation's lineage is stashed into `prev_` and `ring_hi_` is invalidated instead of remaining stuck at the prior long prompt's length.
   - `dflash2_rewind_ring_hi(ring_hi, resume)`: clamps `ring_hi_` to `resume` when a continued conversation truncates below `ring_hi_`.
   - `dflash2_tail_shift_overlaps(head_n, tail_m)`: detects overlapping source/destination ranges (`0 < head_n < tail_m`) when shifting `tail_m` hidden rows down to row `0` in `draft_df2`.
   - `dflash2_clone_eager_single_graphs(blob_shared, batch_body_on)`: skips eager capture of 16 unused single-slot graphs (`exec_[1..8]`, `body_exec_[1..8]`) on slot clones when `STRATA_DF2_BATCHBODY` is enabled.
   - Unit tests added in `tests/core/batch_admit_merge_test.cpp` (suites 9, 10, 11).

2. **`core: coalesce conversation_snapshot_save D2H stream syncs and skip host zero-fill`** (`src/core/conversation_snapshot.cpp`, `src/core/conversation_state.cpp`)
   - Added `conversation_kv_save_queued` using `resize_for_overwrite` on the synchronous `!staged` host `ConversationBuffer` path (eliminating redundant CPU `memset(0)` immediately before `cudaMemcpyAsync` overwrites the buffer) and supporting deferred `sync_copy`.
   - Updated `checkpoint_save` and `conversation_snapshot_save` to queue all D2H copies across `checkpoint_save`, all 12 QSA KV layers, and draft KV before executing a single `conversation_detail::sync_copy(error)` at the end of `conversation_snapshot_save` (replacing 14 separate stream synchronizations per snapshot save with 1).

3. **`dflash2: reset ring_hi_ on cold requests, skip unused clone graphs, and prewarm 1-slot batch graphs`** (`include/strata/core/dflash2.hpp`, `src/core/dflash2.cpp`, `src/core/dflash2_ring_test.cpp`)
   - Wired `dflash2_begin_request_covered` and `dflash2_rewind_ring_hi` into `Dflash2Drafter::begin_request` so cold requests (`resume == 0`) after a 4,712-token prompt reset `ring_hi_` to `0` instead of leaving `ring_hi_` stuck at `4,712` and copying 4,712 stale KV rows on every subsequent short-prompt `copy_to_slot`.
   - Wired `dflash2_clone_eager_single_graphs` in `Dflash2Drafter::prepare` to save `46 MiB` of VRAM at startup (`858 MiB` free vs `812 MiB` in Wave 8) and up to `74 MiB` during concurrent batch execution (`583 MiB` free vs `493–509 MiB` in Wave 8, eliminating `#2065` `< 512 MiB` low-VRAM warnings).
   - Added `Dflash2Drafter::prewarm_batch_slots` to pre-capture 1-slot `batch_full_1_exec_` graphs at startup.
   - Added regression check in `src/core/dflash2_ring_test.cpp`.

4. **`generate: prewarm 1-slot batch DFlash2 graphs, guard overlapping draft_df2 shift, and skip draft on cancel`** (`src/program/generate.cpp`)
   - Wired `df2.prewarm_batch_slots` at `STRATA_BATCH_DF2` initialization.
   - Guarded the `draft_df2` tail shift with `dflash2_tail_shift_overlaps(head_n, tail_m)` so overlapping `0 < head_n < tail_m` shifts stage through `df2.landing()` instead of invoking undefined overlapping `cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice)`.
   - Sampled `stop_now = stop_req.load()` before the end-of-window DFlash2 update / `draft_df2(a)` in the solo decode loop so cancelled solo requests skip launching a wasted `draft_df2(a)` pass before handing off to batch mode.

---

## Verification & 4-Arm ABBA Benchmark Results

- **Hardware**: AMD Radeon AI PRO R9700 (`gfx1201`, RDNA4, 32 GiB VRAM), AMD Ryzen 9 9950X (16C/32T), ROCm nightly `10.2.0a20260914`
- **Model / Profile**: `qwen3.8-flash-next-unsloth-ud-iq4_xs-v31` (`strata-iq3_s.json`, `--batch 2`, `--kv-resident 32768`, `--resident-experts --resident-budget-gib 34`, `--draft-conf 0.5 --spec 8`)
- **Control Binary**: Wave 8 promoted engine `626a2ff5c` / `fork-0989d6f36` (`sha256: 476bae04d0c69cc3332d9051ce79792cfc427da1fbafb95634d0087fc587788f`)
- **Unit & GPU Parity Tests**:
  - CPU unit CTests: `batch_admit_merge_test`, `batch_df2_budget_test`, `batch_mtp_depth_test`, `batch_mtp_select_test`, `arch_defaults_test` — **100% pass**
  - GPU parity CTests: `conversation_snapshot_test`, `dflash2_slots_test`, `iq_multi_parity`, `resident_plan_parity`, `native_multi_parity`, `verify_batch_parity` — **100% pass**
  - Output token/hash parity (`PROMPT_REFACTOR` greedy `temp=0`): **`ece1433a582a4c38`** across all 4 arms in all runs (**100% bit-exact parity**).

### Sub-Phase Telemetry Comparison (`ab_concurrent_pf_dec_wave9_v1_20261109` & `v4`/`v5`)

- **Free VRAM Headroom**:
  - Startup free VRAM: **`858 MiB`** (Wave 9) vs **`812 MiB`** (Wave 8 Control) — **+46 MiB free VRAM**
  - Minimum free VRAM during concurrent `pf+dec` + `batch2`: **`583 MiB`** (Wave 9, zero `#2065` warnings) vs **`493–509 MiB`** (Wave 8 Control, triggered `WARNING: only 493..509 MiB of VRAM free (below 512 MiB)`).
- **Initial 1-Slot `copy_to_slot` (21 tokens, cold)**:
  - Wave 8 Control: **`3.7–3.8 ms`** (lazy capture of `batch_full_1_exec_` on first admission)
  - Wave 9 Candidate: **`1.7–1.8 ms`** (**2.1x faster**, prewarmed at startup)
- **Short-Prompt `copy_to_slot` (63–69 tokens) After 4,712-Token Co-Prefill**:
  - Wave 8 Control: **`12.7 ms`** net `copy_to_slot` (`27.6 ms` total - `14.9 ms` `park_slot(4839)`), because `df2.ring_hi_` remained stuck at `4,712`
  - Wave 9 Candidate: **`2.0–7.8 ms`** net `copy_to_slot` (**up to 6.25x faster**, `df2.ring_hi_` properly reset to `65`)
- **`batch2` Short-Prompt `copy_to_slot` (40 tokens) After 4,839-Token Prompt**:
  - Wave 8 Control: **`7.6–7.9 ms`** net `copy_to_slot` (`22.2–22.6 ms` total - `14.3–15.0 ms` `park_slot(4839)`)
  - Wave 9 Candidate: **`1.8–3.5 ms`** net `copy_to_slot` (**2.2x–4.4x faster**)
- **`park_slot(4839 tokens)` Snapshot Save**:
  - Wave 8 Control: **`14.9–16.8 ms`**
  - Wave 9 Candidate: **`10.6–14.2 ms`** (**up to 1.44x faster**)

### End-to-End 4-Arm ABBA Summary (`v1` & `v4`/`v5`)

- **Solo Decode (`refactor`, 316 prompt + 192 generated tokens)**:
  - `v1` 4-arm ABBA mean: **`105.85 tok/s`** (Wave 9: `106.50`, `105.21`) vs **`100.25 tok/s`** (Wave 8: `94.60`, `105.91`) — **+5.6%**
- **Concurrent Co-Prefill + Decode (`pf+dec`, `4,712t` co-prefill + `51t` decode)**:
  - `v1` 4-arm ABBA mean: **`67.92 tok/s`** aggregate (`45.61 tok/s` Slot 0 decode, **`5.02 s`** Slot 1 latency) vs **`66.27 tok/s`** aggregate (`44.40 tok/s` Slot 0 decode, **`5.19 s`** Slot 1 latency) — **+2.5% aggregate throughput, +2.7% Slot 0 decode, -3.4% (-170 ms) Slot 1 latency**
  - `v4` 4-arm ABBA Slot 1 (`4,712t+128d`) latency: **`4.917 s`** mean (`4.895 s`, `4.939 s`) vs **`5.007 s`** Control mean (`5.041 s`, `4.973 s`) — **-90 ms faster** across all 4 rounds
  - `v5` Round 1 warmed `pf+dec` peak: **`71.0 tok/s`** aggregate (`47.4 tok/s` Slot 0 decode, `4.90 s` Slot 1 latency)
- **Concurrent 2-Slot Batch Decode (`batch2`, `40t + 48t`, 256 tokens/slot)**:
  - `v5` 4-arm ABBA: **`136.95 tok/s`** (`136.8`, `137.1 tok/s` — zero batch graph evictions/re-captures) vs **`137.95 tok/s`** Control (`137.0`, `138.9 tok/s`, within `0.7%` noise floor).

## Reproduction Commands

```bash
# Build
systemd-run --user --scope -p MemoryMax=6G -- \
  cmake --build /home/soren/Strata-perf-dflash2-on-main-20261103/build-hip \
  --target batch_admit_merge_test dflash2_ring_test verify_batch_parity strata conversation_snapshot_test -j 2

# CPU unit tests
ctest --test-dir /home/soren/Strata-perf-dflash2-on-main-20261103/build-hip --output-on-failure \
  -R "^(batch_admit_merge_test|batch_df2_budget_test|batch_mtp_depth_test|batch_mtp_select_test|arch_defaults_test)$"

# GPU parity tests + 4-arm ABBA benchmark (with automatic strata.service restart trap)
bash -c '
trap "systemctl --user start strata.service" EXIT
systemctl --user stop strata.service
ctest --test-dir /home/soren/Strata-perf-dflash2-on-main-20261103/build-hip --output-on-failure \
  -R "^(conversation_snapshot_test|dflash2_slots_test|iq_multi_parity|resident_plan_parity|native_multi_parity|verify_batch_parity)$"
systemd-run --user --scope -- /home/soren/Strata/.venv/bin/python \
  /home/soren/.hermes/cache/scratch/bench_concurrent_prefill_decode_ab.py
'
```
