# Implementation Notes — VRAM Reclaim Queue (Audit Findings 8, 9, 10)

Branch: `feat/vram-small-code` (worktree `~/scratch/Strata-vram-small-code`, base commit `70e634cd7`)  
Reference tree (`~/Strata-latest-20261112-182859`) and live `strata.service` were left completely untouched; no GPU kernels or GPU binaries were executed.

## Commit Sequence

1. `9efbbd396` — `[Engine] add VRAM reclaim plan for audit findings 8, 9, 10` (`PLAN.md`)
2. `10c2c4b28` — `[Engine] size solo parent DFlash2 drafter by real batch slots` (Audit Finding 9 / Item 2)
3. `b214d55c8` — `[Engine] pool NativeDense tensors into one 256B-aligned device arena` (Audit Finding 10 / Item 3)
4. `f1599a027` — `[Engine] price late DFlash2 bind and verifier allocations exactly` (Audit Finding 8 / Item 1)

---

## Item 2 (Audit Finding 9): Parent DFlash2 Sized by Real Batch

- **Commit**: `10c2c4b28` (`[Engine] size solo parent DFlash2 drafter by real batch slots`)
- **What changed**:
  - Updated `Dflash2Drafter::load(..., max_batch_slots)` to accept `std::max(1, o.batch)` from [src/program/generate.cpp](src/program/generate.cpp).
  - Sized non-clone (`!blob_shared_`) per-row buffers and logits rows from `effective_batch_slots(max_batch_slots, blob_shared_)` instead of hardcoding `kBatchMaxSlots = 8`:
    - Solo (`max_batch_slots <= 1`): `alloc_slots_ = 1`, `alloc_rows_ = kRows = 8`, `logits_rows() = kDrafts = 7`; skips allocating `rt_bctx_`, `fn_batch_`, `hsel_batch_`, and batch mapped/staging buffers (`precomp_*`, `h_brows_`, `h_bpos_`, `h_blog_`).
    - Multi-slot batch (`o.batch > 1`): preserves `alloc_slots_ = 8`, `alloc_rows_ = 64`, `logits_rows() = 16`, and all batch graph/capture allocations unchanged.
  - Exposed single-source sizing helpers on `Dflash2Drafter`: `effective_batch_slots()`, `row_count()`, `logits_row_count()`, `alloc_breakdown()`, `resolve_dvocab()`, `logits_bytes()`, `bind_bytes()`, and `clone_bind_bytes()`.
- **Files touched**:
  - `include/strata/core/dflash2.hpp`
  - `src/core/dflash2.cpp`
  - `src/program/generate.cpp`
  - `cmake/strata_gpu_engine.cmake`
  - `tests/core/dflash2_size_test.cpp`
- **Computed VRAM delta** (`dflash2_size_test` alloc arithmetic on Flash-Next parent shape):
  - `alloc()` solo vs batch: `89,608,192 B` (`85.46 MiB`) vs `121,868,416 B` (`116.22 MiB`) -> saved **`32,260,224 B` (`30.77 MiB`)** (`row_buf = 31,563,904 B`, `batch_only = 696,320 B`).
  - `logits_bytes()` solo (`7` rows) vs batch (`16` rows):
    - Native head (`248,320` vocab): saved **`8,939,520 B` (`8.53 MiB`)**.
    - Pruned head (`48,000` vocab): saved **`1,728,000 B` (`1.65 MiB`)**.
  - Total solo parent reclaim (`alloc + native logits`): **`41,199,744 B` (`39.29 MiB`)**.
- **Unverified / GPU-dependent**:
  - Compiled `dflash2_ring_test` (`make -j$(nproc) dflash2_ring_test`), but did not execute `dflash2_ring_test` or live engine startup because the GPU runs production `strata.service`.

---

## Item 3 (Audit Finding 10): `NativeDense` Single-Arena Allocation

- **Commit**: `b214d55c8` (`[Engine] pool NativeDense tensors into one 256B-aligned device arena`)
- **What changed**:
  - Replaced per-matrix `cudaMalloc` calls in `NativeDense::load` with a two-pass single-arena allocator:
    - Pass 1 validates shards, shapes, and constraints and packs dense projections, optional `STRATA_Q8_PACKED=1` copies, GGUF/pack HC Q8_0 matrices, `g_hc_scratch`, and `native_q8_1` `scratch_` at `NativeDense::kArenaAlign = 256` byte alignment (`plan_arena()` / `arena_payload_bytes()` / `arena_alloc_bytes()`).
    - Pass 2 performs one `cudaMalloc(&arena_, payload_bytes)`, uploads/quantizes directly into carved pointers, registers packed keys, and commits atomically on success.
    - Updated `~NativeDense()` to clear `g_hc_scratch` entries whose pointer falls within `[arena_, arena_ + arena_size_)`, unregister `packed_keys_`, and call `cudaFree(arena_)`.
  - Updated `NativeDense::weight_bytes_for()` (including map overload used in host tests) and `allocated_bytes()` to sum 256 B-aligned tensor payload plus a single 2 MiB driver granule (`arena_alloc_bytes`), updated the startup log line in `src/program/generate.cpp` (`%.2f MiB of weights (%.2f MiB arena allocated)`), and added granule-pricing assertions in `src/core/auto_cache_test.cpp`.
- **Files touched**:
  - `include/strata/core/native_dense.hpp`
  - `src/core/native_dense.cpp`
  - `src/program/generate.cpp`
  - `src/core/auto_cache_test.cpp`
  - `cmake/strata_gpu_engine.cmake`
  - `tests/core/native_dense_arena_test.cpp`
- **Expected / computed VRAM delta** (`native_dense_arena_test` + `auto_cache_test`):
  - `192` HC Q8_0 matrices (`3.32 MiB` each = `2560/32 * 34 * 1280 B`): per-matrix `4.00 MiB` (`768.00 MiB` total) -> single arena **`638.00 MiB`** -> saved **`130.00 MiB`** (`~47` max-blob expert cache slots returned to `auto_cache_slots`).
  - `300` dense matrices (`6.73 MiB` each, matching file comment `native_dense.cpp:29-38`): per-matrix `8.00 MiB` (`2400.00 MiB` total) -> single arena **`2020.00 MiB`** -> saved **`380.00 MiB`**.
  - Host GGUF fixture tests verify bit-identical tensor byte offsets / view planning and clean `~NativeDense` teardown without device initialization.
- **Unverified / GPU-dependent**:
  - Compiled `native_dense_ple_key_test`, `native_dense_types_parity`, `native_grouped_parity`, `native_multi_parity`, and `hip_prefill_native_batch`, but did not execute GPU tests or live `hipMemGetInfo` before/after measurements on R9700 because the GPU runs production `strata.service`.

---

## Item 1 (Audit Finding 8): Exact Late DFlash2 / Verifier Bind Pricing

- **Commit**: `f1599a027` (`[Engine] price late DFlash2 bind and verifier allocations exactly`)
- **What changed**:
  - Exposed `Prefill::tap_bytes(layers, cap_rows, n_embd, stage_lb, stage_le)` and routed `Prefill::reserve_taps()` through it.
  - Factored `Verifier::plan_i32_count()` and `Verifier::carve_arena()` out of `Verifier::init()` so `Verifier::arena_bytes()` computes the exact device arena size from the same bump-carve sequence (`init()` and `arena_bytes()` share one source of truth), and exposed `Verifier::tap_bytes()`.
  - Removed `df2_head_bytes()` and hardcoded constants (`+48 MiB`, `+112 MiB` for solo, `+24 MiB`, `+4 MiB * slot_df2.size()`) from `src/program/generate.cpp`, replacing them at `pin_by_need`, `expert cache auto`, multi-GPU `last_st` room subtraction, and startup logging with one `df2_late_bytes()` helper summing:
    - `df2.bind_bytes(...)` + sum of `slot_df2[b]->clone_bind_bytes(...)`
    - `o.serve` prompt taps: `Prefill::tap_bytes(...)` when solo/`batch <= 1` (`100.00 MiB`), keeping the `112 MiB` tap path when `o.batch > 1` (and stage-partitioned + landing taps on multi-GPU)
    - `Verifier::tap_bytes(...)`
    - `Verifier::arena_bytes(...)`
  - Kept `strata::core::kLateAllocMib = 256` intact in `include/strata/core/auto_cache.hpp` for unpriced runtime/library warmup allocations and kept non-DFlash2 sizing unchanged.
  - Preserved the parseable `expert cache auto` banner (`+%lld MiB for the draft head`) printed from the exact computed `mtp_bind`, and added a `dflash2/verify late bind` breakdown line when DFlash2 is enabled.
- **Files touched**:
  - `include/strata/prefill/prefill.hpp`
  - `src/prefill/prefill.cpp`
  - `include/strata/core/verify.hpp`
  - `src/core/verify.cpp`
  - `src/program/generate.cpp`
  - `src/core/auto_cache_test.cpp`
  - `cmake/strata_gpu_engine.cmake`
- **Computed pricing figures** (`auto_cache_test` on Flash-Next solo geometry `N=2560, L=48, n_vocab=248320, max_t=8`, `iq4_xs_max_blob=2867200`, `had_gu_any=0`):
  - `Prefill::tap_bytes({5,15,25,35,45}, 2048, 2560)` = **`100.00 MiB`** (`104,857,600 B`, vs hardcoded `112 MiB` previously)
  - `Verifier::tap_bytes({5,15,25,35,45}, 8, 2560)` = **`0.39 MiB`** (`409,600 B`; `0.78 MiB` at `T=16`)
  - `Dflash2Drafter::bind_bytes` (native head, solo 7-row logits) = **`6.63 MiB`** (`6,952,960 B`; pruned `48,000` logits = `1,344,000 B` / `1.28 MiB`)
  - `Verifier::arena_bytes` (`T=8`, `had_gu_any=0`) = **`89.07 MiB`** (`134.37 MiB` at `T=16`)
  - Total computed solo DFlash2 + Verifier bind reserve (`T=8`): **`196.09 MiB`** (`0.0%` delta vs constituent allocator formulas, well within `±5%`). On runtime native packs with `had_gu == true`, `Verifier::arena_bytes` reads `expert_layout()` automatically (`nat_xr_` adds `T * N * 4 = 80 KiB` at `T=8`).
- **Unverified**:
  - Live engine startup and end-of-load `hipMemGetInfo` balance (GPU runs production `strata.service`).
  - Python terminal tests (`tests/terminal`): `pytest` is not installed in the host Python environments tested (`No module named pytest`).

---

## Build & Host-Side Test Verification

### Build Command (executed after final Item 1 edit)
```sh
systemd-run --user --scope -p MemoryMax=24G -p TasksMax=8192 bash -c '
  set -e
  export HIP_PATH=/opt/rocm
  export ROCM_PATH=/opt/rocm
  export PATH=/opt/rocm/bin:$PATH
  cd /home/soren/scratch/Strata-vram-small-code/build-hip-test
  make -j$(nproc)
  make -j$(nproc) dflash2_ring_test native_dense_ple_key_test
'
```
- **Result**: Exit code `0`. Full `make -j$(nproc)` plus `dflash2_ring_test` and `native_dense_ple_key_test` compiled and linked on HIP `gfx1201` (`Release`, `STRATA_ENABLE_HIP=ON`, `STRATA_PREFILL_MMQ=ON`).

### Host-Side CPU Tests Run (executed with `CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1`)
```sh
CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ctest -V -R \
  "^(auto_cache_test|dflash2_size_test|native_dense_arena_test|device_free_error_test|dequant_q5_1_test|batch_rows_test|draft_cap_test|draft_policy_test|draft_vocab_test|draft_kv_plan_test|coupled_draft_test|split_ring_test|split_part_price_test|stage_reserve_test|split_rules_test|split_walk_test|ple_residency_test|qwen35_sampling_test|gguf_split_test|gguf_geometry_test|model_family_test|resident_ram_copy_test|native_arithmetic_parity|native_hadamard_test)$" \
  --output-on-failure
```
- **Result**: Exit code `0` (`100% tests passed out of 24` in CTest reporting: **23 passed**, **1 skipped internally by test binary**):
  - Passed (23): `gguf_split_test`, `model_family_test`, `gguf_geometry_test`, `split_part_price_test`, `resident_ram_copy_test`, `split_rules_test`, `stage_reserve_test`, `split_walk_test`, `device_free_error_test`, `dflash2_size_test`, `native_dense_arena_test`, `ple_residency_test`, `auto_cache_test` (27 checks), `qwen35_sampling_test`, `draft_policy_test`, `draft_vocab_test`, `split_ring_test`, `draft_kv_plan_test`, `coupled_draft_test`, `draft_cap_test`, `batch_rows_test`, `dequant_q5_1_test`, `native_arithmetic_parity` (`5,529,600` checks).
  - Internally skipped (1): `native_hadamard_test` (`this ggml has no trellis types (build with STRATA_GGML_DIR = the agention fork): skipped`, exit code 0 / CTest `Skipped`).

### GPU Tests Compiled Only — Skipped With Reason
Live production (`strata.service`) owns the GPU (`gfx1201`), so no GPU-allocating or GPU-kernel test binaries were executed:
- `dflash2_ring_test` (`EXCLUDE_FROM_ALL`, requires GPU for DFlash2 ring/context execution)
- `native_dense_ple_key_test` (requires GPU to run `NativeDense` / PLE key GPU verification)
- `expert_cache_per_layer_test`, `expert_cache_segmented_test`, `expert_cache_slot_guard_test` (require GPU expert cache allocation)
- `native_dense_types_parity`, `native_grouped_parity`, `native_multi_parity`, `hip_expert_cache_staging`, `hip_native_qsa_score`, `hip_prefill_native_batch` (require GPU kernel execution)
- `native_expert_parity_new_*` (28 targets, skipped as outside pure host-CPU verification without confirming CPU-only isolation for every variant)
- Also not built/run: `strata-device --selftest` (`cuda_device_selftest`), full engine CLI/server execution.

---

# Structural VRAM Follow-on (Audit Findings 11 and 12)

Branch: `feat/vram-structural` (worktree `~/scratch/Strata-vram-structural`, stacked on `feat/vram-small-code`).  The production tree and live `strata.service` were not accessed for writes or execution.  No GPU memory was allocated by a test or engine run.

## Phase 1 (Audit Finding 11a): HIP VMM for Segmented ExpertCache

- **Commit**: `[Engine] route segmented ExpertCache through shared VMM`
- **What changed**:
  - Replaced ExpertCache's private CUDA-driver VMM implementation with `VmmRange`, which uses the existing CUDA driver-entrypoint and HIP runtime VMM implementations.
  - Enabled segmented ExpertCache on VMM-capable HIP builds.  gfx906 compatibility, Metal, SYCL and the explicit `STRATA_EC_NO_VMM` kill switch remain excluded.
  - Kept eager mapping of every physical chunk at initialization; `--vram-elastic` remains opt-in and the ordinary `cudaMalloc` path is unchanged.
  - Updated the unsupported-backend diagnostic and the segmented test's skip check to use the common "virtual memory management" wording.
- **Files touched**:
  - `include/strata/core/expert_cache.hpp`
  - `include/strata/core/vmm.hpp`
  - `src/core/expert_cache.cpp`
  - `tests/core/expert_cache_segmented_test.cpp`
- **Build / host tests**:
  - The requested configure command completed.  Its initial full build failed at final executable linking because `STRATA_PREFILL_MMQ` was left OFF: `strata_prefill` references `fused::want_dm_f16` / `fused::dm_f16_used`, while the CMake option excludes the library that defines them.
  - Reconfigured the same HIP gfx1201 Release build with the additional `-DSTRATA_PREFILL_MMQ=ON`; the full `cmake --build build-hip-test -j$(nproc)` then passed.
  - `CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ctest --test-dir build-hip-test --output-on-failure -R '^(auto_cache_test|dflash2_size_test|native_dense_arena_test|expert_profile_save_test)$'`: passed (4/4).
  - `ctest --test-dir build-hip-test -N -R 'auto_cache|dflash2|native_dense|expert_cache|vmm|quota'`: listed the segmented cache and VMM tests; this command only enumerated tests.
- **Skipped with reason**:
  - `expert_cache_segmented_test` and `vmm_test` were built but not run: both use device virtual memory and would allocate GPU memory.
  - No live `--vram-elastic` engine run or default-vs-elastic device measurement was performed because the production GPU is in use.
  - A `STRATA_EC_NO_VMM` HIP host `-fsyntax-only` compile passed after the later quota change; a full second configured build remains unrun.

  ## Phase 2 (Audit Finding 11b): Safe Mapped-Prefix Guard; Shared-Pool Transfer Deferred

  - **Commit**: `[Engine] guard unmapped ExpertCache slots and report elastic residency`
  - **What changed**:
    - Added `ExpertCache::is_live_slot` and applied it to admission, slot lookup, device-slot pointers, slot-tail pointers, and repeat admission checks.  A shrunk tail cannot be selected through these paths.
    - Added an opt-in-only startup line reporting mapped slots, live slots, and mapped/reserved bytes.
    - Added the host-only `expert_cache_live_slot_test`, including prefix boundaries, negative/out-of-prefix slots, zero slots, and restored-prefix/budget bounds.
    - Did not transfer cache-owned VMM chunks to taps, verifier/DFlash2/logits, or graphs.  [`DESIGN-NOTES.md`](DESIGN-NOTES.md) records the consumer ordering table, current quiesce/fence evidence, missing ownership gate and graph-lifetime proof, the required future protocol, and explicit live gfx1201 A/B work.
  - **Files touched**:
    - `include/strata/core/expert_cache.hpp`
    - `src/core/expert_cache.cpp`
    - `src/program/generate.cpp`
    - `tests/core/expert_cache_live_slot_test.cpp`
    - `cmake/strata_gpu_engine.cmake`
    - `DESIGN-NOTES.md`
  - **Build / host tests**:
    - `systemd-run --user --scope -p MemoryMax=24G -p TasksMax=8192 bash -c 'export PATH=/opt/rocm/bin:$PATH HIP_PATH=/opt/rocm; cmake --build build-hip-test -j$(nproc)'`: passed incrementally on the configured HIP gfx1201 Release tree (`STRATA_PREFILL_MMQ=ON`).
    - `CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ctest --test-dir build-hip-test --output-on-failure -R '^(auto_cache_test|dflash2_size_test|native_dense_arena_test|expert_profile_save_test|expert_cache_live_slot_test)$'`: passed (5/5).
  - **Deferred / skipped with reason**:
    - Map-on-demand sharing, cross-pool handle transfer, verifier graph/tap pooling, explicit access revocation for shared chunks, and fault injection into device VMM are deferred because the producer/consumer ownership and graph lifetime cannot be proven from current locks/events.  See [`DESIGN-NOTES.md`](DESIGN-NOTES.md).
    - The host test validates only the mapped-prefix policy.  Device wraparound, allocation failure, VMM handle reuse, and graph replay remain GPU-only and pending.
    - No GPU test, engine, or live resize command was run.

## Phase 3 (Audit Finding 12): Startup-Only Learned Layer Quotas

- **Commit**: `[Engine] allocate startup cache quotas from learned heat`
- **What changed**:
  - Added opt-in `--expert-cache-quota auto`.  It loads per-expert routing heat from `<expert-profile>.heat`, assigns integer per-layer slot quotas by marginal heat per byte, preserves the uniform plan's total slot count, respects each native layer's slot stride, and floors every layer at one slot.  Missing or all-zero heat falls back to uniform quotas.
  - `--expert-profile-save` now writes the heat sidecar atomically; malformed dimensions, invalid counts, and extra/truncated data are rejected when loading.
  - Quota calculation and cache-allocation retry paths both check their byte budgets.  The startup banner prints all layer quotas; runtime rebalancing is not implemented.
  - The existing per-layer cache sizing and admission paths are unchanged when the new option is absent.  Also removed the stale HIP-only refusal of `--vram-elastic`, so its already-ported VMM path is no longer disabled by the CLI.
- **Files touched**:
  - `include/strata/core/expert_cache.hpp`
  - `src/core/expert_cache.cpp`
  - `src/program/generate.cpp`
  - `tests/core/expert_cache_quota_test.cpp`
  - `cmake/strata_gpu_engine.cmake`
  - `DESIGN-NOTES.md`
- **Build / host tests**:
  - Incremental HIP gfx1201 Release build: `systemd-run --user --scope -p MemoryMax=24G -p TasksMax=8192 bash -c 'set -e; export PATH=/opt/rocm/bin:$PATH HIP_PATH=/opt/rocm; cmake --build build-hip-test -j$(nproc)'`: passed after the final changes.
  - `CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ctest --test-dir build-hip-test --output-on-failure -R '^(auto_cache_test|dflash2_size_test|native_dense_arena_test|expert_profile_save_test|expert_cache_live_slot_test|expert_cache_quota_test)$'`: passed (6/6) after the final changes.
  - `STRATA_EC_NO_VMM` HIP host syntax compile with `-fsyntax-only`: passed; one existing ignored-`hipFree` return-value warning remains.
- **Skipped with reason**:
  - `expert_cache_segmented_test` and `vmm_test`, plus all device ordering and live elastic A/B checks: not run because they allocate/use the production GPU.  See the pending assertions in `DESIGN-NOTES.md`.
