# VRAM Reclaim Implementation Plan (Small Code Queue)

Branch: `feat/vram-small-code` (base `70e634cd7`)  
Reference audit: `docs/VRAM-EXPERT-RESIDENCY-AUDIT-2026-11-14.md` (Findings 9, 10, 8)  
Implementation order: **Item 2 (Finding 9) → Item 3 (Finding 10) → Item 1 (Finding 8)** (Item 1 consumes Item 2's updated drafter sizing and Item 3's arena pricing model).

---

## Grounding Notes vs Codebase `70e634cd7`

Before implementation, source inspection identified four exact codebase locations/symbols that differ slightly from summarized notes:
1. Auto-cache host test lives at `src/core/auto_cache_test.cpp` (registered in `cmake/strata_gpu_engine.cmake`), not `tests/c++/auto_cache_test.cpp`.
2. `Verifier::arena_bytes` and `Dflash2Drafter::bind_bytes` do not exist yet in `include/strata/core/verify.hpp` / `include/strata/core/dflash2.hpp`; we will introduce both from the allocators' own bump/bind logic so there is a single source of truth.
3. The `"mtp_bind"` banner at `src/program/generate.cpp:8216-8219` computes `(mtp_bind >> 20)` from heuristic constants (`df2_head_bytes() + 48 MiB + (serve ? 112 MiB : 0) + ((batch_max_t() > 8 ? 24 : 0) + slot_df2.size() * 4) MiB`, ~190 MiB in live solo `--lookup-max-t 16`) at three call sites (`7873`, `8184`, `8806`).
4. `NativeDense::weight_bytes_for()` (`src/core/native_dense.cpp:215`) currently mirrors individual `cudaMalloc` 2 MiB granule rounding (`alloc_bytes(tensor_bytes)`).

---

## Item 2 (Audit Finding 9): Parent DFlash2 sized by real batch

### Problem
`Dflash2Drafter::alloc()` (`src/core/dflash2.cpp:384-447`) sizes the parent drafter (`!blob_shared_`) with `B = kBatchMaxSlots * kRows = 64`, `HMax = max(16, 2 * kDrafts) = 16`, `il_max = 16`, plus batch-only staging (`rt_bctx_`, `fn_batch_`, `hsel_batch_`, `topval_`/`toptok_` for 56 rows, mapped `h_bmeta_`/`h_bout_`), and `bind_raw()` allocates `logits_` for `16` rows. Only `blob_shared_` slot clones shrink to `kRows = 8`. In live solo production (`--batch` off; deferred tap chunks capped at 8 in `70e634cd7`), 7/8 of row buffers and 9/16 of `logits_` (plus all batch-only scratch) are unused.

### Plan
1. **`include/strata/core/dflash2.hpp` & `src/core/dflash2.cpp`**:
   - Add `int max_batch_slots = 1` parameter (default `1` for solo/standalone tools) to `Dflash2Drafter::load(...)` (or setter `set_batch_slots(int)` before `load`), clamped to `[1, kBatchMaxSlots]`. Note `clone_alloc` sets `blob_shared_ = true` and `alloc_slots_ = 1`.
   - When `!blob_shared_` and `batch_slots > 1` (multi-slot batch mode), keep `alloc_slots_ = kBatchMaxSlots` (`B = 64`), `HMax = max(16, 2 * kDrafts) = 16`, `il_max = 16`, and allocate `rt_bctx_`, `fn_batch_`, `hsel_batch_`, `h_bmeta_`, `h_bout_` so batch graph captures (`xqc_ + 80 * N_`, `rt_bblk_ = rt_bctx_ + 64 * 128`, `record_batch_body` / `record_batch_head`) keep their exact 64-row layout.
   - When `blob_shared_` OR `batch_slots == 1` (solo parent): size `B = kRows = 8`, `HMax = kDrafts = 7`, `il_max = kRows = 8`, `topval_`/`toptok_` = `kDrafts * 16`, `logits_` `hmax = kDrafts = 7`, and omit `rt_bctx_`, `fn_batch_`, `hsel_batch_`, `h_bmeta_`, `h_bout_`.
   - Fix `record_kv()` (`dflash2.cpp:594`) and batch helpers (`lines 1578, 1790, 1982, 2003, 2125`) to use `alloc_rows_` (`alloc_slots_ * kRows`) rather than hardcoding `blob_shared_ ? kRows : kBatchMaxSlots * kRows` for `l_stride`. (Verify every consumer in `dflash2.cpp`: `rebuild()`, `context()`, `prewarm_batch_slots()`, `prewarm_precompute()`, `precompute_batch_context()`, `draft_batch_slots_launch()` already gate multi-row/batch paths on `!blob_shared_ && rt_bctx_ != nullptr`.)
   - Expose pure CPU-callable sizing helpers on `Dflash2Drafter` (`alloc_rows()`, `logits_rows()`, `row_buffer_bytes(...)`, `bind_bytes(...)`, `clone_bind_bytes(...)`) so Item 1 and unit tests share one formula.
2. **`src/program/generate.cpp`**:
   - Pass `(o.batch > 1) ? strata::core::Dflash2Drafter::kBatchMaxSlots : 1` when calling `df2.load(...)` (`line 7380`). Note `batch_df2_wanted` requires `o.batch >= 2`; `--batch 1` uses solo parent behavior safely.

### Expected Byte Delta (Qwen3.8 draft config: `N=2560, L=5, NKV*HD=1024, n_dvocab=248320`)
- `xqp_`, `xqc_`, `xdp_`, `xdc_` (`64 -> 8` rows): `-1.54 MiB` (`1,612,800 B`)
- `xq_`, `xil_` (`16 -> 8` rows): `-0.11 MiB`
- `rt_bctx_` (`2 * 64 * 128 * 4` omitted): `-0.06 MiB` (`65,536 B`)
- `fcy_`, `hctx_`, `h_`, `xn_`, `xc_`, `o_`, `mlp_`, `fn_` (`8 * 56 * 2560 * 4`): `-4.38 MiB` (`4,587,520 B`)
- `kc_`, `vc_` f16 (`2 * 5 * 56 * 1024 * 2`): `-1.09 MiB` (`1,146,880 B`)
- `dyn_`, `q_`, `qn_`, `k_`, `v_`, `kblk_`, `ao_`, `gate_`, `up_`, `part_`, `hsel_`, `topscratch_`: ~`-6 to -10 MiB` (pack `I_`/`NH_`/`rank_` dependent)
- `fn_batch_`, `hsel_batch_` omitted: `-0.55 MiB` (`573,440 B`) + `56 * rank_ * 4`
- `logits_` (`16 -> 7` rows at `n_dvocab = 248320`): `-8.53 MiB` (`8,939,520 B`)
- **Total solo parent reclamation**: **~22–28 MiB** (computed deterministically by `Dflash2Drafter` sizing helpers; 0 B change for `o.batch > 1`).

### Risks & Mitigations
- Risk: `record_kv()` `l_stride` mismatch corrupting multi-layer K/V projections. Mitigation: replace `blob_shared_ ? kRows : kBatchMaxSlots * kRows` with `alloc_rows_` across all projection sites.
- Risk: solo code hitting `rt_bctx_` or `xqc_ + 80 * N_` offset overflow. Mitigation: `rt_bctx_ == nullptr` when `alloc_slots_ <= 1`, and solo deferred-tap chunking (`70e634cd7`) already caps chunks at `kRows = 8`.

### Test Strategy
- Build `dflash2_ring_test` (`EXCLUDE_FROM_ALL`) and `strata-device` to confirm compilation on HIP (`STRATA_HIP=ON`). Do not execute GPU binaries.
- Add a host-side unit test (pure C++ / CPU, checking row buffer + `logits_` arithmetic for solo `batch=0/1` -> 8 rows / 7 logit rows vs `batch>1` -> 64 rows / 16 logit rows, `record_kv` stride consistency, and `clone_alloc`).

---

## Item 3 (Audit Finding 10): `NativeDense` single-arena allocation

### Problem
`NativeDense::load()` (`src/core/native_dense.cpp:285-591`) performs one `cudaMalloc` per dense projection (~301), packed Q8 copy (if enabled), GGUF/quantized `hc_q8` (~194), `g_hc_scratch[dev][2][2]` (4), and `scratch_`. HIP/ROCm drivers back >=1 MiB allocations with 2 MiB granules, wasting ~0.68 MiB per 3.32 MiB HC tensor (130.5 MiB on HC alone) plus unmeasured slack across ~300 dense tensors. `WeightTable::load` (`src/core/weights.cpp`) already avoids this using one contiguous arena with 256-byte-aligned offsets.

### Plan
1. **Two-pass single-arena allocator in `NativeDense::load`**:
   - Add pure static arena helpers to `include/strata/core/native_dense.hpp`:
     `kArenaAlign = 256`, `kAllocGranule = 2 MiB`, `kSmallAlloc = 1 MiB`, `arena_align(bytes)`, `alloc_bytes(bytes)`, `arena_bytes(sizes, n)`, `arena_alloc_bytes(sizes, n)`.
   - Pass 1 (dry plan, no device allocation or full shard retention beyond headers): walk GGUF shards in the exact same order and filter rules as today (`bf16_as_q8_0`, `native_mmvq_supported`, `blk.1.ple_key.weight`, optional `STRATA_Q8_PACKED`, GGUF `hc_q8`, pack-quantized `hc_q8`, `g_hc_scratch` if needed on `dev`, `scratch_`), recording `arena_align(size)` offsets and exact payload sizes in a descriptor list. Fail early before any `cudaMalloc` if validation fails or `pending` is empty.
   - Single allocation: `cudaMalloc(&arena, arena_size)` wrapped in RAII `DevicePtr` until commit.
   - Pass 2 (populate): upload/quantize/pack each tensor directly into `arena + offset` (`cudaMemcpy` for GGUF payloads; one-tensor scratch conversion for `bf16_as_q8_0`, `STRATA_Q8_PACKED`, skipped BF16 `hc_q8`; `fused_gr_quantize_q8_0` for in-arena BF16 `hc_q8`; register `native_q8_0_packed_register` with `(item_data_ptr, item_packed_ptr)`). Keep tensor views, `WeightRef` fields (`native_data`, `native_type`, `native_q8_1`, `hc_q8`, `hc_q8_from_gguf`), and strides bit-identical.
   - Teardown (`~NativeDense()`): store `arena_ = nullptr` / `arena_bytes_ = 0` (or `weights_` holds only base `cudaMalloc` allocations cleanly): unregister `packed_keys_` before freeing `arena_`, and null any `g_hc_scratch[dev][c][i]` whose address falls within `[arena_, arena_ + arena_size)` so multi-instance / reload lifecycles remain clean.
2. **Accounting & pricing consistency**:
   - Update `NativeDense::allocated_` in `load()` to `alloc_bytes(arena_size)` (one granule for the single arena, inclusive of HC Q8 & scratches that live inside it).
   - Update `NativeDense::weight_bytes_for()` (`native_dense.cpp:215`) to accumulate `arena_align(tensor_bytes)` across eligible matrices and apply `alloc_bytes(total)` once at the end instead of per-matrix `alloc_bytes`. Note `generate.cpp:6225` (`carve_bytes` in `--layer-split auto`) already uses `weight_bytes_for`.
   - Update the startup print line at `src/program/generate.cpp:5421` to report both payload and arena-granule allocation (`%.2f MiB in 1 arena` alongside payload weights).
   - Update `src/core/auto_cache_test.cpp` to assert single-granule arena pricing (`NativeDense::arena_alloc_bytes` vs per-matrix `alloc_bytes`) feeds extra free bytes into `auto_cache_slots`.

### Expected Byte Delta
- Dense (~301 tensors) + HC Q8 (~194 tensors) packed at 256 B alignment incurs `< 0.1 MiB` internal alignment padding and only `<= 2 MiB` single-arena tail rounding.
- Savings: **130.5 MiB minimum** (on the 192–194 HC Q8 3.32 MiB tensors: `192 * 0.68 MiB`) plus per-tensor 2 MiB granule slack across the 301 dense projections (**audit upper bound estimate ~130–400 MiB total**, directly reflected in higher `hipMemGetInfo` free bytes at `auto_cache_slots`).

### Risks & Mitigations
- Risk: `g_hc_scratch[dev][c][i]` dangling pointer after `~NativeDense()`. Mitigation: pointer-range check `[arena_base, arena_base + arena_bytes_)` in `~NativeDense()` before `cudaFree(arena_)` cleanly nulls all matching global scratch slots.
- Risk: `STRATA_Q8_PACKED` registry (`packed_keys_`) or `hc_q8` double-free. Mitigation: `native_q8_0_packed_unregister` runs before `cudaFree(arena_)`, matching existing order; only the arena pointer is freed.
- Risk: Host memory spike during conversion. Mitigation: stream one tensor at a time in Pass 2 instead of holding converted `std::vector<uint8_t>` across all shards.

### Test Strategy
- Host-side C++ unit test covering 256 B offset packing, bit-identical payload offsets / non-overlap inside a host-simulated arena, single-granule `arena_alloc_bytes` accounting, and `NativeDense::weight_bytes_for` parity against synthetic GGUF shards + `WeightTable` index without requiring a GPU (check existing `tests/core/native_dense_ple_key_test.cpp` and `native_dense_*_parity`, skipping GPU-dependent binaries if no device is usable / respecting the no-GPU rule).

---

## Item 1 (Audit Finding 8): Price late drafter/bind allocations exactly

### Problem
`src/program/generate.cpp` (`7873-7875`, `8184-8186`, `8805-8808`) prices DFlash2 post-cache allocations using heuristic constants:
`df2_head_bytes() + (48 << 20) + (o.serve ? (112 << 20) : 0) + (((batch_max_t() > 8 ? 24 : 0) + slot_df2.size() * 4) << 20)`.
On solo production (`--serve`, `--lookup-max-t 16`), this produces `+190 MiB` from magic constants instead of reading actual allocator formulas (`Prefill::reserve_taps` = 100 MiB vs 112, `df2.bind` `logits_` = 6.63 MiB after Item 2, `Verifier::arena_bytes` + `tap_dev_` = ~75–85 MiB). Non-DFlash2 runs continue to rely on `kLateAllocMib = 256` (`#4474`, which already includes non-DFlash2 verifier baseline + cuBLAS/lazy workspaces).

### Plan
1. **Single-source-of-truth allocator calculators**:
   - `Verifier::arena_bytes(...)` (`include/strata/core/verify.hpp`, `src/core/verify.cpp`): extract `Verifier::init`'s `carve(Bump&)` lambda into a shared layout function parameterized cleanly so BOTH `Verifier::arena_bytes(...)` and `Verifier::init(...)` run the exact same bump sequence (`max_blob` & `had_gu` optionally overridable for host unit tests, defaulting to `expert_layout()`).
   - `Verifier::tap_bytes(tap_layers, max_t, n_embd, stage_lb, stage_le, total_layers)`: matches `verify.cpp:1066, 1084` (`tap_dev_`).
   - `Prefill::tap_bytes(tap_layers, cap_rows, n_embd, stage_lb, stage_le)` (`include/strata/prefill/prefill.hpp`, `src/prefill/prefill.cpp`): called directly by `Prefill::reserve_taps(...)`.
   - `Dflash2Drafter::bind_bytes(...)` and `Dflash2Drafter::clone_bind_bytes(...)` (`include/strata/core/dflash2.hpp`, `src/core/dflash2.cpp`): matches `Dflash2Drafter::bind` (`dhead` + `dvocab` + `logits_rows() * n_dvocab * 4`, reflecting Item 2's `logits_rows()` = 7 in solo vs 16 in batch).
2. **`src/program/generate.cpp` (`mtp_bind`, `pin_by_need`, `df2_after`, startup banner)**:
   - Replace `df2_head_bytes()` and all magic MiB constants (`48 << 20`, `112 << 20` on solo, `24 << 20`, `slot_df2.size() * 4 << 20`) with a helper computing exact DFlash2 post-cache bind + verifier bytes:
     - `df2.bind_bytes(head.row_bytes(), n_vocab, df2_vocab_path(), df2_expect_share, same_head_dev)`
     - `+` sum of `slot->clone_bind_bytes(n_vocab, df2_vocab_path(), same_head_dev)` (`o.batch > 1`)
     - `+` prompt taps: if `o.serve`: `(o.batch > 1) ? (112 << 20)` (keeps conservative batch tap path) else `Prefill::tap_bytes(df2.tap_layers(), 2048, g.n_embd)` (plus `dland` when `multi_gpu`)
     - `+` `Verifier::tap_bytes(df2.tap_layers(), batch_max_t(), g.n_embd, lb, le, g.n_layers)`
     - `+` `Verifier::arena_bytes(g, ss, n_vocab, batch_max_t())`.
   - Keep `strata::core::kLateAllocMib` (`256 MiB`) in `auto_cache_slots` untouched for cuBLAS/rocBLAS/hipBLASLt workspaces, lazy kernels, and verify graph captures.
   - Startup banner at `generate.cpp:8216-8219` automatically prints the exact computed `(mtp_bind >> 20)` (and `late_priced` `kLateAllocMib`).
3. **`src/core/auto_cache_test.cpp` (`cmake/strata_gpu_engine.cmake`)**:
   - Link `auto_cache_test` to `strata_engine` (host-only, no device calls needed) and extend it with assertions verifying:
     - Solo `dflash2 + prompt_taps + draft_logits` exact sum matches (`Prefill::tap_bytes` + `Dflash2Drafter::bind_bytes` + `Verifier::tap_bytes` + `Verifier::arena_bytes`) with zero magic slack.
     - Single-source consistency (`Verifier::arena_bytes` monotonicity in `max_t` and `n_vocab`, `Dflash2Drafter::bind_bytes` solo 7-row vs batch 16-row `logits_`, pruned vocab vs native head vs shared head).
     - `NativeDense` single-arena pricing delta increases `auto_cache_slots` coherently.

### Expected Byte Delta (Solo Qwen3.8 + DFlash2 + `--lookup-max-t 16` + `--serve` + `head_native`)
- Old `mtp_bind`: `6.63 (df2_head_bytes) + 48 + 112 + 24` = **190.63 MiB**
- New `mtp_bind`: `6.63 MiB (df2.logits_, 7 rows) + 100.00 MiB (Prefill::tap_bytes) + 0.78 MiB (Verifier::tap_bytes) + Verifier::arena_bytes(max_t=16)` (~`75–85 MiB`) = **`~182–192 MiB`** (within ±5% of actual allocations, and dynamically tracks `max_t`, `n_vocab`, `head_native` vs pruned vocab, `slot_df2`, `o.serve`, and Item 2's 8-row vs 64-row sizing!).

### Risks & Mitigations
- Risk: Double-counting `Verifier::arena_bytes` on non-DFlash2 runs (`kLateAllocMib = 256` already prices non-DFlash2 baseline arena + cuBLAS). Mitigation: only charge `Verifier::arena_bytes` inside the `use_df2` branch where `(48 << 20)` + `(24 << 20)` previously approximated it, leaving non-DFlash2 paths byte-identical.
- Risk: Drift between `Verifier::init` and `Verifier::arena_bytes`. Mitigation: single shared `carve_layout` function used by both.

---

## Global Verification Strategy (Host-Side Only)
- Never run GPU tests or the production engine; never touch `~/Strata-latest-20261112-182859` or `strata.service`.
- Build with HIP host toolchain inside a `systemd-run --user --scope -p MemoryMax=24G -p TasksMax=8192` scope in `~/scratch/Strata-vram-small-code/build-hip-test`.
- Run CPU/host-side `ctest` targets matching `auto_cache|dflash2|native|dense|expert_cache` plus any new host tests; explicitly document every GPU-requiring test as skipped-with-reason.

---

# Structural VRAM Follow-on (Findings 11 and 12)

This addendum applies to the `feat/vram-structural` worktree, based on the completed small-code
queue above. Default behavior remains unchanged; do not touch the production tree or run a GPU
allocator or GPU test.

## Phase 1 — HIP segmented ExpertCache

Route the segmented arena through `strata::core::Vmm`/`VmmRange`, which already supplies CUDA and
HIP implementations. Remove only the implicit HIP compile-out; preserve an explicit
`STRATA_EC_NO_VMM` compile-time override, gfx906/Metal exclusions, VMM runtime probing, eager
mapping at open, and non-segmented allocation arithmetic. Make the unsupported-device error
consistent with the existing test's VMM diagnostic.

**Risk:** VMM support can be established by the host build and the existing runtime probe, but
mapping, access and release correctness on gfx1201 require a GPU test that is prohibited here.

**Acceptance:** Full HIP build; `expert_cache` and `vmm` host CTests pass; the segmented test is
registered without a HIP exclusion (but is not executed if it needs a device); default cache
sizing stays unchanged.

## Phase 2 — map-on-demand / shared chunks

First establish ownership and device-ordering for each allocation. Record producer, consumer,
stream, event/fence and quiescence point for cache slots, prompt taps, verifier arena, DFlash2
logits, and any graph-captured pool. Do not hand a physical chunk to a consumer until graph replay
is impossible and the residency table and all accesses to the tail are drained. Tap and verifier
regions used by captured graphs must remain mapped for the full graph lifetime unless graph
invalidation is proved.

**Safety gate:** If existing code cannot prove those ordering/lifetime conditions without a live
GPU A/B, stop short of reallocating the consumers. Ship safe mapped/live accounting and host-only
refusal/budget checks if useful; document which pools stay excluded and which GPU checks remain
pending.

**Acceptance:** Host-testable wraparound, zero-shrink, over-budget refusal and unmapped-tail
selection checks; accurate mapped/live startup reporting; complete device-ordering table; every
GPU-only assertion named as pending.

## Phase 3 — learned startup per-layer quotas

Add opt-in `--expert-cache-quota auto`. Use persisted per-layer heat only when available; the
legacy `STRP` v1 profile contains a ranking, not miss counts, so rank position alone must not be
treated as heat mass. Keep existing readers and writers compatible and fall back to the existing
uniform quota when no learned heat is available. Allocate integer slots using exact sized-slot
bytes, floor each layer at the minimum, and enforce the total auto-cache byte budget.

**Acceptance:** Host tests for exact byte cap, total-slot preservation, concentrated heat with
minimum floors, and uniform fallback. Print the compact quota vector. Do not rebalance at runtime.

## Commit sequence

1. `[Docs] update VRAM structural implementation plan`
2. `[Engine] enable HIP VMM for segmented expert cache`
3. `[Engine] document safe cache chunk lending boundary`
4. `[Engine] add opt-in learned per-layer cache quotas`
