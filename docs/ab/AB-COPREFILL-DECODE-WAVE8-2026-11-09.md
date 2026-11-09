# Wave 8 Decode & Slot-Handoff Co-Prefill Speedup Report (2026-11-09)

## Summary

Wave 8 builds on Wave 7 (`73918ce06`) to eliminate decode-side, slot-admission (`copy_to_slot`), slot-return (`copy_from_slot`), DFlash2 ring-copy, and first-prompt host-allocation bottlenecks during concurrent `--batch` decode and co-prefill on single-card `gfx1201` (AMD Radeon AI PRO R9700, 32 GiB VRAM, AMD Ryzen 9 9950X 16C/32T AVX-512):

1. **Decouple `single_stage_hybrid` Admission Merge from Dynamic `pcie_frac` (`single_stage_hybrid_admit_merge`)**:
   - In `src/program/generate.cpp` (`merge_from`), `STRATA_ADMIT_MERGE_REM=1` previously gated `single_stage_hybrid` on `o.pcie_frac <= 0.0f`. Because `pcie_auto_step` raises `o.pcie_frac` from `0.00` to `0.25` after window 100, `admit_merge_from` silently turned off on all subsequent `BGEN` admissions (forcing an extra 1-token verify window + separate `df2.context` + `df2.draft`).
   - Added `strata::core::single_stage_hybrid_admit_merge(merge_rem_on, n_pipe, !host_res.empty())` (`include/strata/core/batch_admit_merge.hpp`) so single-stage hybrid MoE keeps merging partial `read_windows` remainders with the final prompt token regardless of dynamic `pcie_frac`.
2. **$O(1)$ `ConvCheckpoint` Move on `Solo <-> Batch Slot` Handoffs (`prompt_last_turn_boundary` & `should_move_slot_checkpoints_to_solo`)**:
   - On `Solo -> Batch Slot` (`give_way` / `from_live`), `turn_at` was only scanned in `(resume, n - 1)`. Because `from_live` has `resume == n - 1 > turn_at`, `turn_at` was `-1` and `move_best` was `false`, forcing a synchronous `11–14 ms` 118 MiB `ConvCheckpoint` deep copy on every `give_way` admission. Added `strata::core::prompt_last_turn_boundary` to scan `(0, n - 1)` (excluding `root_at`) so `move_best` moves the prompt's own turn checkpoint in $O(1)$.
   - On `Batch Slot -> Solo` (`_may_go_solo` / `BSTOP`, where `admit_slot == -1` and `slot_source >= 0`), `move_cks` required `admit_slot == slot_source` (`false` when returning to solo), forcing another 118 MiB deep copy plus a redundant `park_current` save at the start of solo continuation. Added `strata::core::should_move_slot_checkpoints_to_solo` (`include/strata/core/batch_admit_merge.hpp`) and cleared `bs[slot_source].cached` / `bs[slot_source].ids` when moving into solo.
3. **Bounded DFlash2 KV Ring Copy on Handoffs (`dflash2_ring_copy_slots`)**:
   - `Dflash2Drafter::copy_state_from` (`src/core/dflash2.cpp`) previously copied the entire `kRing = 8,208`-slot buffer (`268.9 MiB` F32 / `42.3 MiB` F16 across 5 layers) via `cudaMemcpyAsync` even when `src.ring_hi_` was only `62–274` tokens.
   - Added `strata::core::dflash2_ring_copy_slots(src.ring_hi_, kRing)` (`include/strata/core/batch_admit_merge.hpp`) and updated `Dflash2Drafter::copy_state_from` to copy only `[0, n_slots)` columns per KV head via `cudaMemcpy2DAsync` when `0 < n_slots < kRing`, and updated `Dflash2Drafter::rebuild()` to maintain `ring_hi_`.
4. **Fast Stopping-Slot (`BSTOP`) Verify & Pre-Draft Handoff (`slot_pre_drafted`)**:
   - In `batch_step()` (`src/program/generate.cpp`), gated `room` and `extra` draft row allocation to `0` when `q.stop` is set so a stopping slot verifies only 1 row instead of consuming up to 7 speculative rows and discarding them.
   - When a stopping slot finishes with `solo_handoff_fast`, launched DFlash2 drafting (`df2_launched`) on the committed stop token (reusing `df2.has_precomputed_context(...)` when valid rather than re-running `ver.gather_taps` + `d.context`), and preserved the resulting `slot_handoff_draft` / `slot_handoff_dprob` across the slot-to-solo return (`slot_pre_drafted`, `first_window = !slot_pre_drafted`).
   - Together with Fix 2 and Fix 3, this cuts `copy_from_slot` (`slot N gave back ...`) by **11.8x** (`17.7 ms -> 1.5 ms`) and cuts solo continuation prompt read by **10x** (`17–18 ms -> 1–2 ms`).
5. **Post-Doorbell Layer-37 Tap Precompute Callback & Startup Host Buffer Pre-Initialization**:
   - In `Verifier::run` and `Verifier::run_slot_rows` (`src/core/verify.cpp`), moved `taps_ready_cb_` invocation at Layer 37 (`l == tap_layers_.back() + 1`) to run *after* `pool(...)` and `*flag = want` so DFlash2 precompute kernel launches overlap inside Layer 38's GPU spin-wait rather than delaying Layer 37 CPU expert dispatch and GPU combine.
   - In `Prefill::init` (`src/prefill/prefill.cpp`), pre-allocated `m.cpu_x`, `m.cpu_rows`, `m.cpu_ones`, `m.cpu_nact`, and `m.cpu_need` at `cpu_max_t = std::max<int64_t>(1, stream_all_min() - 1)` (`2,560` tokens) at startup when `cpu_pool_ != nullptr && cpu_share_on() && single_card_gfx12_default()`, eliminating the `350–889 ms` first-prompt `cudaHostAlloc` stall in Layer 0 `host grouping`.

## Unit & GPU Parity Gates

All 14 CPU unit tests and 7 GPU parity tests built and passed (`21/21` CTest targets, `0` skipped, `0` failed):

- CPU unit suite (`14/14` passed): `share_rules_test`, `batch_admit_merge_test`, `arch_defaults_test`, `native_arithmetic_parity` (all 8 variants), `pool_tasks_test`, `pool_plan_test`, `copy_events_test`.
- GPU parity suite (`7/7` passed): `iq_multi_parity` (all 4 variants), `native_multi_parity`, `resident_plan_parity`, `verify_batch_parity`.

## Paired 4-Arm ABBA Benchmark Results (`--batch 2`, `qwen3.8-flash-next-unsloth-ud-iq4_xs-v31`)

- **Control binary (Wave 7 `73918ce06`)**: `/home/soren/.hermes/cache/scratch/strata-wave7-73918ce06` (`sha256: c927d57313e8dc74f688361c49b64ca7a21e5db3f5dba5370b591a52e3444d58`)
- **Candidate binary (Wave 8 `626a2ff5c`)**: `/home/soren/Strata-perf-dflash2-on-main-20261103/build-hip/strata` (`sha256: 476bae04d0c69cc3332d9051ce79792cfc427da1fbafb95634d0087fc587788f`)
- **Greedy output hash (`PROMPT_REFACTOR`, 192 tokens, `temp=0`)**: **`ece1433a582a4c38`** across all 4 arms and all runs (`100%` bit-identical).

### Key Sub-Phase Latency Reductions (Engine Telemetry)

- **Slot-to-solo return (`copy_from_slot` — `slot N gave back ...`)**:
  - Wave 7 Control: `18.2 ms`, `17.8 ms`, `17.5 ms`, `17.3 ms` (**mean `17.70 ms`**)
  - Wave 8 Candidate: **`1.5 ms`, `1.5 ms`, `1.5 ms`** (**mean `1.50 ms` — `11.8x` faster, `-16.2 ms` per handoff**)
- **Solo continuation startup after slot return (`prompt 270..279 tokens = N-1 reused + 1 read`)**:
  - Wave 7 Control: `18 ms`, `17 ms`, `17 ms` (`56.2–57.8 tok/s`)
  - Wave 8 Candidate: **`1 ms`, `2 ms`, `2 ms` (`660.2–679.5 tok/s` — `10x` faster, `-15.5 ms` per return)**
- **Solo-to-slot admission (`copy_to_slot` — `slot N takes ...`)**:
  - `4,712-token` co-prefill admission (`with single_stage_hybrid`): drops from `19.6 ms / 13.0 ms` down to **`5.6 ms` (`2.9x` faster)**
  - Warm `48-token` Batch-2 Round 1 admission: drops from `10.7 ms / 9.1 ms` down to **`3.1 ms` (`3.2x` faster)**
- **First-prompt `316-token` prefill (`PROMPT_REFACTOR` Run 0)**:
  - Wave 7 Control: `391–1,294 ms` (subject to first-use `288 MiB` `cudaHostAlloc` stall in Layer 0 `host grouping`)
  - Wave 8 Candidate: **`440 ms` / `453 ms` (`719.0 / 697.6 tok/s`)** on Run 0, **`443 ms` / `392 ms` (`714.0 / 806.9 tok/s`)** on Run 1

### Arm-by-Arm Summary (`ab_concurrent_pf_dec_wave8_v6_20261109`)

- **`0_control_wave7` (Wave 7 control launch 1)**:
  - Greedy refactor: `106.7 tok/s` (`ece1433a582a4c38`)
  - Concurrent `pf+dec` aggregate (`Slot 0 256g + Slot 1 4,712p+128g`): `68.2 tok/s` (`68.0 / 68.4 tok/s`)
  - Slot 0 decode during 4,712-token co-prefill: `45.5 tok/s` (`Slot 1: 4.98 s`)
  - Concurrent Batch-2 decode (`c=2`, `2 x 256g`): `132.6 tok/s` (`132.7 / 132.5 tok/s`, per-slot `68.5 tok/s`)
- **`1_cand_wave8_default` (Wave 8 candidate launch 1)**:
  - Greedy refactor: `90.2 tok/s` (`ece1433a582a4c38`)
  - Concurrent `pf+dec` aggregate: **`68.7 tok/s` (`66.6 / 70.8 tok/s`, peak `70.8 tok/s`)**
  - Slot 0 decode during 4,712-token co-prefill: **`45.8 tok/s` (`44.4 / 47.2 tok/s`, `Slot 1: 4.98 s`, `4,712-token` prompt read `3,206 ms / 3,213 ms`)**
  - Concurrent Batch-2 decode (`c=2`): **`137.4 tok/s` (`135.3 / 139.5 tok/s`, per-slot `69.9 tok/s`)**
- **`2_cand_wave8_repeat` (Wave 8 candidate launch 2)**:
  - Greedy refactor: **`106.0 tok/s` (`96.6 / 115.4 tok/s`, `ece1433a582a4c38`)**
  - Concurrent `pf+dec` aggregate: **`69.0 tok/s` (`68.5 / 69.5 tok/s`)**
  - Slot 0 decode during 4,712-token co-prefill: **`46.0 tok/s` (`45.7 / 46.3 tok/s`, `Slot 1: 5.02 s`, `4,712-token` prompt read `3,213 ms / 3,217 ms`)**
  - Concurrent Batch-2 decode (`c=2`): **`136.7 tok/s` (`136.6 / 136.9 tok/s`, per-slot `70.6 tok/s`)**
- **`3_control_wave7_repeat` (Wave 7 control launch 2)**:
  - Greedy refactor: `79.3 tok/s` (`53.0 / 105.7 tok/s`, `ece1433a582a4c38`)
  - Concurrent `pf+dec` aggregate: `67.6 tok/s` (`66.5 / 68.8 tok/s`)
  - Slot 0 decode during 4,712-token co-prefill: `45.1 tok/s` (`44.3 / 45.9 tok/s`, `Slot 1: 5.02 s`, `4,712-token` prompt read `3,256 ms / 3,191 ms`)
  - Concurrent Batch-2 decode (`c=2`): `137.9 tok/s` (`141.7 / 134.1 tok/s`, per-slot `70.8 tok/s`)

### Paired ABBA Means (Wave 8 vs Wave 7 Control)

- **Greedy Refactor (`316p + 192g`)**: `93.0 tok/s -> 98.1 tok/s` (**`+5.5%`**)
- **Concurrent `pf+dec` Aggregate (`Slot 0 256g + Slot 1 4,712p+128g`)**: `67.9 tok/s -> 68.8 tok/s` (**`+1.3%` in v6; `66.8 -> 68.5 tok/s`, `+2.5%` in v3**)
- **Slot 0 Decode During 4,712-Token Co-Prefill**: `45.3 tok/s -> 45.9 tok/s` (**`+1.3%` in v6; `44.5 -> 45.7 tok/s`, `+2.7%` in v3**)
- **Concurrent Batch-2 Decode (`2 x 256g`)**: `135.3 tok/s -> 137.1 tok/s` (**`+1.3%`**, per-slot `69.7 -> 70.2 tok/s`)

## Reproduction Command

```bash
trap "systemctl --user start strata.service" EXIT
systemctl --user stop strata.service
while fuser /dev/kfd >/dev/null 2>&1; do sleep 1; done
systemd-run --user --scope -- python3 /home/soren/.hermes/cache/scratch/bench_concurrent_prefill_decode_ab.py
```
