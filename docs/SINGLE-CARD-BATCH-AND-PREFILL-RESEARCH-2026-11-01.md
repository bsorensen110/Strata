# Single-Card Batched Decode & Concurrent Prefill Research, Implementation & Benchmarks (`gfx1201` / RDNA4)

**Date:** 2026-11-01
**Hardware:** Single AMD Radeon AI PRO R9700 (`gfx1201`, 32 GiB VRAM, 64 WGP / 128 CU wave32, ~640 GB/s peak / ~480 GB/s effective memory bandwidth) + 16-core AVX-VNNI CPU (64 GiB DDR5, ~85 GB/s, 15 pool workers)
**Model & Serving Config:** `swift-1.5-iq3_s` + `Qwen3.5-35B-A3B-DFlash` (`106,299`-token `draft_vocab.bin`, `--serve --batch 4 --draft dflash2 --draft-cap 6 --dflash-no-mtp --batch-draft-window 8 --batch-draft-depth 1 --batch-suffix-draft 3 --expert-cache auto --expert-cache-per-layer --resident-experts --pcie-frac 0.15 --kv int8 --kv-resident 32768 --kv-pool-tokens 262144 --pool-workers 15 --vram-reserve-mib 576`)
**Promoted Production Binary (`strata.service`):** `75cd16bde97ddff79db9934db9cbc1c8` (`Strata-integ` `2170b617`; upstream PR #4937 `42864fcc`)

---

## Executive Summary & Complete Measured Progression (`gfx1201`, 256 Greedy Tokens/Slot)

| Workload | Metric | Original Baseline (`69.1 / 97.9 tok/s`) | Prod Baseline (`ccd1b635`, serial co-draft) | Wave 1 (`afc6ae5f`, `BATCHHEAD` + batch QSA/GDN/KV) | Wave 2 (`75cd16bd`, `BATCHBODY` + Layer-36 overlap + `BGEN` & `lend()` fixes) | Total Gain (Wave 2 vs `ccd1b635`) |
|---|---|---|---|---|---|---|
| **`c=2` Cold (R1, 2×256)** | Window (`ms/win`) | — | 50.90 ms (`emit 5.23`) | 43.80 ms (`emit 4.84`) | **43.28 ms (`emit 3.76`)** | **−15.0% window (−28.1% emit)** |
| **`c=2` Cold (R1, 2×256)** | Engine (`rows/s`) | — | 125.1 rows/s | 144.4 rows/s | **145.2 rows/s** | **+16.1%** |
| **`c=2` Cold (R1, 2×256)** | HTTP Wall (`tok/s`) | 69.08 tok/s (34.5/slot) | 105.4 tok/s (52.7/slot) | 117.1 tok/s (58.5/slot) | **124.08 tok/s (62.04/slot)** | **+17.7% (+79.6% vs orig)** |
| **`c=2` Warm (R2, 2×256)** | Window (`ms/win`) | — | 43.46 ms (`emit 5.31`) | 39.99 ms (`emit 4.89`) | **39.77 ms (`emit 3.88`)** | **−8.5% window (−26.9% emit)** |
| **`c=2` Warm (R2, 2×256)** | Engine (`rows/s`) | — | 144.6 rows/s | 157.7 rows/s | **159.9 rows/s** | **+10.6%** |
| **`c=2` Warm (R2, 2×256)** | HTTP Wall (`tok/s`) | — | 122.0 tok/s (61.0/slot) | 132.7 tok/s (66.4/slot) | **132.96 tok/s (66.48/slot)** | **+9.0%** |
| **`c=4` Cold (R1, 4×256)** | Window (`ms/win`) | — | 49.36 ms (`emit 9.54`) | 44.80 ms (`emit 7.86`) | **42.46 ms (`emit 5.46`)** | **−14.0% window (−42.8% emit)** |
| **`c=4` Cold (R1, 4×256)** | Engine (`rows/s`) | — | 148.5 rows/s | 163.6 rows/s | **167.8 rows/s** | **+13.0%** |
| **`c=4` Cold (R1, 4×256)** | HTTP Wall (`tok/s`) | 97.93 tok/s (24.5/slot) | 124.5 tok/s (31.1/slot) | 137.6 tok/s (34.4/slot) | **140.00 tok/s (35.00/slot)** | **+12.4% (+43.0% vs orig)** |
| **`c=4` Warm (R2, 4×256)** | HTTP Wall (`tok/s`) | — | 125.6 tok/s (31.4/slot) | 136.1 tok/s (34.0/slot) | **143.76 tok/s (35.94/slot)** | **+14.5% (+46.8% vs orig)** |
| **`c=2` + Concurrent 3,200-Tok Prefill** | Decode (`tok/s`) | — | ~16 tok/s (~8/slot) | ~16 tok/s (~8/slot) | **70.39 tok/s (35.20/slot)** | **~4.4× faster decode under prefill** |
| **`c=2` + Concurrent 3,200-Tok Prefill** | Prefill (`tok/s`) | — | 2,140 tok/s (un-chunked) | 2,140 tok/s (un-chunked) | **1,532 tok/s (`2048` + `1179` chunks)** | Smooth interleaved decode + prefill |
| **Slot Admission (`BGEN` move)** | Move time (`ms`) | ~450 ms | ~450 ms | ~450 ms | **23.7–29.6 ms** | **~18× faster slot admission** |
| **4-Slot DFlash2 Draft (`32` rows)** | Draft time (`ms/win`) | ~12.0 ms | 9.54 ms | 7.63 ms | **4.57 ms (standalone) / 5.46 ms (`emit` total)** | **2.6× faster multi-slot draft** |

### Atomic Commits Implemented (Wave 2)
- **Integration branch (`Strata-integ` `integ/community-2026-10`) / Upstream PR #4937 (`pr/dflash2-batch-optimizations`):**
  1. `996650ea` / `a6a4d7c4` — `fix(batch,dflash2): preserve n-1 prompt tap, initial draft & suffix drafter on BGEN admission`
  2. `5928dc61` / `4c4ceef7` — `perf(batch,dflash2): add device-to-device Dflash2Drafter::copy_slot & enable direct GDN slot copy with kv_pool`
  3. `2999c0b1` / `a8fccb1e` — `perf(dflash2): batch context projection & 5-layer body across active slots (STRATA_DF2_BATCHBODY)`
  4. `19e21378` / `900b60ce` — `perf(verify,dflash2): overlap DFlash2 context projections with target verify layers 37-47`
  5. `f211c1a2` / `c7cc3c4a` — `fix(expert-cache): interleave per-layer slots rank-major so prefill lend() borrows coldest experts evenly`
  6. `2170b617` / `42864fcc` — `perf(batch,prefill): cap prefill chunk size and scale decode share when batch slots are active`

---

## Part 1: Newly Discovered Correctness & Performance Bugs Fixed in Wave 2

### 1.1 Missing Last-Prompt-Token (`n - 1`) Tap & Missing First-Window Draft on Batch Admission (`BGEN`) (`996650ea` / `a6a4d7c4`)
- **Where:** `src/program/generate.cpp` (`BGEN` admission rewrite and post-verify DFlash2 hook)
- **Mechanism:**
  1. When a batch request arrives via `BGEN <slot> <max_new> ...`, the engine rewrites the admission pass to `GEN 1 ...` (`max_new = 1`) and saves the real target length in `admit_max_new`.
  2. Prefill reads tokens `0 .. n - 2` (`n - 1` tokens) and feeds their layer taps into `df2` (`slot 0`), leaving `slot 0` with `have_ctx_ = n - 1`.
  3. The admission pass then runs one target verify window at position `p = n - 1` (`T = 1`) for the final prompt token `ids[n - 1]`, producing the first generated token `x = outv[0]` (`produced_n = 1`) and writing the hidden-state tap of `ids[n - 1]` into `ver.final_tap_dev()`.
  4. The post-verify DFlash2 draft gate checked `if (df_active && produced_n < max_new)`. Because `max_new == 1` (rewritten by `BGEN`), `produced_n < max_new` (`1 < 1`) evaluated to **`false`**, even though `admit_slot >= 0 && admit_max_new > 1`.
  5. Consequently:
     - `df2.feed_context` / `df2.draft_block` **never ran for position `n - 1` (`ids[n - 1]`)**.
     - `copy_to_slot` copied `slot 0`'s DFlash2 KV ring (`have_ctx_ = n - 1`) into `slot admit_slot`, leaving position `n - 1` permanently missing from `slot admit_slot`'s DFlash2 KV ring for the entire generation.
     - `slot admit_slot` entered its first batch window with `sl.draft_ready = false` (`0` proposals offered on its first batch window).
- **Fix Implemented:**
  - Extended the post-verify DFlash2 condition to `const bool admit_continues = admit_slot >= 0 && admit_max_new > 1; if (df_active && (produced_n < max_new || admit_continues))` so `df2.feed_context` commits `ids[n - 1]`'s tap and `df2.draft_block` produces the initial speculative tree (`admit_df2_draft`, `admit_df2_ready = true`), which is transferred to `bs[admit_slot]` on admission.

### 1.2 `SuffixDrafter` (`--batch-suffix-draft`) Wiped Out on Admission (`996650ea` / `a6a4d7c4`)
- **Where:** `src/program/generate.cpp` (`copy_to_slot` and `BGEN` admission initialization)
- **Mechanism:**
  - Inside `copy_to_slot`, `bs[b].sfx.reset(ids.data(), ids.size())` populated the n-gram suffix tree from the prompt tokens, and then `sl = BSlot{}` zero-initialized the entire `BSlot`, wiping out `sl.sfx` (`sl.sfx.empty() == true`) before the first batch window.
- **Fix Implemented:**
  - Saved `SuffixDrafter saved_sfx = std::move(sl.sfx)` before `sl = BSlot{}`, restored `sl.sfx = std::move(saved_sfx)`, and fed the first emitted token `ids.back()`.

### 1.3 Synchronous Host-Vector Round-Trip on Every Slot Admission/Switch (`5928dc61` / `4c4ceef7`)
- **Where:** `include/strata/core/dflash2.hpp`, `src/core/dflash2.cpp`, `src/program/generate.cpp`
- **Mechanism:**
  - When `--kv-pool-tokens` was active (`moved == true`), `if (slot_copy_direct && !moved)` skipped `direct_slot_copy` and fell back to `conversation_checkpoint_save` (D2H) + `conversation_checkpoint_restore` (H2D) + `cudaDeviceSynchronize()` just to copy the 36 GDN recurrent state tensors (`~18 MiB`).
  - Similarly, `df2.save_state` + `df2.restore_state` copied up to `40.3 MiB` of DFlash2 KV ring from GPU slot `0` -> pageable host `std::vector<uint8_t>` -> GPU slot `b` with `cudaDeviceSynchronize()`, stalling all active decoding slots for `~450 ms` on admission.
- **Fix Implemented:**
  - Added `Dflash2Drafter::copy_slot(int from_slot, int to_slot, int64_t n_keep, std::string& err)` using `cudaMemcpy2D` device-to-device on `cs_`, and enabled `direct_slot_copy(true, b, ids, moved, e)` with `kv_pool` (skipping only `kv_mode == 1` QSA states already moved by `pool_move`).
  - **Measured Result:** Slot admission move time dropped from `~450 ms` to **`23.7–29.6 ms`** (`~18x` faster).

### 1.4 Multi-Slot `STRATA_DF2_BATCHBODY` Attention Output Stride (`NH_ * HD_ = 4096` vs `N_ = 2560`) (`2999c0b1` / `a8fccb1e`)
- **Where:** `src/core/dflash2.cpp:1040` and `src/kernels/cuda/dflash2.cu:289–299`
- **Mechanism:**
  - In DFlash2 (`Qwen3.5-35B-A3B-DFlash`), hidden width `N_ = 2560`, while attention has `NH_ = 32` heads of `HD_ = 128` (`NH_ * HD_ = 4096`, `o_proj` input width `4096 -> 2560`).
  - `df2_attention_f16` / `df2_attention` quantizes `NH_ * HD_ = 4096` floats per row into `xqp` (`4096` int8s/row) and `xdp` (`128` scales/row) for `o_proj`.
  - In the per-slot attention loop (`b0 = idx * 8`), indexing `xqp_ + b0 * N_` (`b0 * 2560`) instead of `xqp_ + b0 * NH_ * HD_` (`b0 * 4096`) caused slot `idx + 1`'s attention output to overwrite the upper `1,536` columns of slot `idx`'s `o_proj` input.
- **Fix Implemented:**
  - Changed the per-slot `df2_attention_f16` / `df2_attention` output offsets to `xqp_ + (size_t) b0 * NH_ * HD_` and `xdp_ + (size_t) b0 * ((NH_ * HD_) / 32)`, restoring full speculative acceptance across all slots (`73.2–73.8%` at `c=2`, `82.0%` at `c=4`).

---

## Part 2: Batched DFlash2 Body (`STRATA_DF2_BATCHBODY`) & Layer-36 Context Overlap (`2999c0b1`, `19e21378`)

### 2.1 Multi-Slot Batched DFlash2 Context & 5-Layer Body (`STRATA_DF2_BATCHBODY=1`) (`2999c0b1` / `a8fccb1e`)
- **Implementation (`src/kernels/cuda/dflash2.cu`, `src/core/dflash2.cpp`, `src/core/dflash2_small_bench.cpp`):**
  - Generalized `gemm8_kernel` with template parameter `int R = 8` (`R = 8` for `ncols <= 8`, `R = 4` for `ncols <= 16`, `R = 2` for `ncols <= 32`) so `acc[R][NCOLS]` stays at `64` FP32 accumulators per thread with zero VGPR spills at `16`, `24`, and `32` columns.
  - Updated `conv_quant_kernel` and `conv_rms_quant_kernel` to reset causal depthwise conv windows at 8-row slot boundaries (`(b & 7) - o < 0`) and added `df2_embed_tokens_batch`, `df2_rms_quant_batch`, and `df2_conv_rms_quant_batch`.
  - Scaled DFlash2 workspace allocations to `B = kRows * 4 = 32` columns so `Dflash2Drafter::draft_batch_begin` reads the `~435 MB` of DFlash2 body weights once per verify window across all active slots (`2..4`) instead of `n_items` times.
  - Verified bitwise parity in `dflash2_small_bench` across `ncols = 16, 24, 32`.

### 2.2 Overlap DFlash2 Context Projections with Target Verify Layers 37–47 (`19e21378` / `900b60ce`)
- **Implementation (`include/strata/core/verify.hpp`, `src/core/verify.cpp`, `src/core/dflash2.cpp`, `src/program/generate.cpp`):**
  - Target verify only taps Layers `[1, 10, 19, 28, 36]`; Layers `37..47` (`11` layers + final norm + output head, `~8–9 ms` on `gfx1201`) do not write to `tap_dev_`.
  - Added `Verifier::set_taps_ready`, invoked in `Verifier::run_layers` immediately after Layer 36's event synchronization (`m_done_[36]`), which calls `Dflash2Drafter::precompute_batch_context(tap_dev_, S, pos)` on `df2.stream()` concurrently with target verify Layers `37..47`.
  - Removed the spurious `&& stage_taps == nullptr` guard on `can_use_precomp` in `Dflash2Drafter::draft_batch_begin` so single-GPU `--batch 4` uses the precomputed context `rt_ctx_`, `fcy_`, and 5× `{kc_l, vc_l}` directly in `draft_batch_begin` and only launches `df2_kv_ring_f16` per slot after token selection.
  - **Measured Result:** Post-verify `emit` time dropped from `4.89 ms` to **`3.76–3.88 ms/window` at `c=2`** and from `7.88 ms` to **`5.46 ms/window` at `c=4`**!

---

## Part 3: Concurrent Prefill & Rank-Major `--expert-cache-per-layer` Slot Lending (`f211c1a2`, `2170b617`)

### 3.1 Rank-Major `--expert-cache-per-layer` Slot Interleaving (`f211c1a2` / `c7cc3c4a`)
- **Root Cause Fixed:**
  - `ExpertCache::configure_per_layer` previously assigned cache slots in layer-major order (`slot = layer_idx * slots_per_layer + rank`), so `lend()` borrowing the contiguous tail `[slots() - need_p, slots())` evicted 100% of resident experts in Layers 42–47 (at `need_p = 1,367` for `--prefill 8192`) or Layers 32–47 (at `need_p = 3,907` for `--prefill 16384`) for the entire prompt read — including during interleaved `piped_decode` / `step_batch` decode windows!
- **Fix Implemented (`src/core/expert_cache.cpp`, `src/program/generate.cpp`, `tests/core/expert_cache_per_layer_test.cpp`):**
  - Interleaved per-layer slots in **rank-major order** (`slot = rank * n_layers + layer_idx`) and updated the initial residency sort in `src/program/generate.cpp` so rank `r` of layer `l` is placed at `r * n_layers + l`.
  - `lend()` borrowing `342` tail slots (`0.74 GiB` for a `2,048`-token chunk) now evicts only the `~7` **coldest-ranked** experts per layer uniformly across all 48 layers, keeping the top `~226` hottest experts per layer resident in VRAM throughout concurrent prefill and decode.

### 3.2 Dynamic `STRATA_BATCH_PREFILL_CHUNK` (`2048`) & `batch_decode_share` Scaling (`2170b617` / `42864fcc`)
- **Implementation (`include/strata/program/batch_read.hpp`, `src/program/generate.cpp`, `tests/program/batch_read_test.cpp`):**
  - Added `kDefaultBatchPrefillChunk = 2048`, `batch_prefill_chunk(...)` (`STRATA_BATCH_PREFILL_CHUNK`, default `2048` on single GPU when `batch_on()` is true), and `batch_decode_share(...)` (`0.5` at 1 active slot, `0.75` at 2 slots, `1.0` at `3+` slots), plus post-final-chunk `piped_decode` yielding when `batch_on()` is true.
- **Measured Result (`c=2` active decode + concurrent `3,200`-token prompt prefill arrival on `gfx1201`):**
  - **Before:** Prefill `lend(8192)` borrowed `1,367` slots (`2.96 GiB`), wiped out layers 42–47, and ran un-chunked, collapsing concurrent decode to **`~8 tok/s/slot`**.
  - **After (`75cd16bd`):** Prefill ran in two chunks (`2,048` + `1,179` tokens, borrowing only `342` slots = `0.74 GiB`), interleaved `1,069 ms` of `piped_decode` (`173` tokens decoded mid-prefill), and sustained **`70.39 tok/s` aggregate decode (`35.20 tok/s/slot`, `~4.4x` faster) simultaneously with `1,532 tok/s` prompt processing**.

---

## Part 4: Detailed Engine Telemetry Logs (`75cd16bd` on `gfx1201`)

### 4.1 `c=2` (`2 × 256` Greedy Tokens)
- **Round 1 (Cold):** `512 tokens in 4.13 s` -> **`124.08 tok/s` (`62.04 tok/s/slot`)**
  ```text
  strata batch: slot 0 takes 44 tokens (moved in 24.2 ms)
  strata batch: slot 1 takes 38 tokens (moved in 24.3 ms)
  strata batch: 81 windows, avg 6.28 rows, 43.28 ms/window = run 38.91 (CUDA0 GPU-reach wait 19.94 + CPU experts 16.43) + commit 0.02 + emit 3.76 + adapt wait 0.59; per stage (ms/window): CUDA0 wait 19.94 + pool 16.43; per layer-window: CPU experts 9.58, VRAM hits 64.31, PCIe 1.10; routed entries per window: VRAM 3087.0 (82.2%), PCIe 56.9 (1.5%), CPU 613.2 (16.3%); adaptive tier (async): 20 rounds started, 1920 experts swapped in; 145.2 rows/s over 3506 ms of wall time (admissions included)
  strata batch: DFlash2 accepted 353/478 offered proposals (73.8%), 3.26 tokens per slot-round over 156 slot-rounds; windows averaged 1.93 slots x 3.15 proposal rows (asked depth 1)
  ```
- **Round 2 (Warm):** `512 tokens in 3.85 s` -> **`132.96 tok/s` (`66.48 tok/s/slot`)**
  ```text
  strata batch: slot 2 takes 39 tokens (moved in 24.1 ms)
  strata batch: slot 3 takes 43 tokens (moved in 23.8 ms)
  strata batch: 75 windows, avg 6.36 rows, 39.77 ms/window = run 35.31 (CUDA0 GPU-reach wait 23.42 + CPU experts 9.63) + commit 0.02 + emit 3.88 + adapt wait 0.57; per stage (ms/window): CUDA0 wait 23.42 + pool 9.63; per layer-window: CPU experts 5.01, VRAM hits 73.52, PCIe 0.36; routed entries per window: VRAM 3528.7 (92.4%), PCIe 17.5 (0.5%), CPU 274.5 (7.2%); adaptive tier (async): 19 rounds started, 1728 experts swapped in; 159.9 rows/s over 2983 ms of wall time (admissions included)
  strata batch: DFlash2 accepted 328/448 offered proposals (73.2%), 3.20 tokens per slot-round over 149 slot-rounds; windows averaged 1.99 slots x 3.03 proposal rows (asked depth 1)
  ```

### 4.2 `c=4` (`4 × 256` Greedy Tokens)
- **Round 1:** `1024 tokens in 7.31 s` -> **`140.00 tok/s` (`35.00 tok/s/slot`)**
  ```text
  strata batch: slot 0 takes 40 tokens (moved in 29.6 ms)
  strata batch: slot 1 takes 38 tokens (moved in 26.1 ms)
  strata batch: slot 3 takes 43 tokens (moved in 25.8 ms)
  strata batch: slot 2 takes 40 tokens (moved in 23.7 ms)
  strata batch: 143 windows, avg 7.13 rows, 42.46 ms/window = run 36.47 (CUDA0 GPU-reach wait 25.66 + CPU experts 8.34) + commit 0.02 + emit 5.46 + adapt wait 0.51; per stage (ms/window): CUDA0 wait 25.66 + pool 8.34; per layer-window: CPU experts 4.49, VRAM hits 73.32, PCIe 0.30; routed entries per window: VRAM 3519.1 (93.4%), PCIe 14.9 (0.4%), CPU 232.2 (6.2%); adaptive tier (async): 35 rounds started, 3456 experts swapped in; 167.8 rows/s over 6072 ms of wall time (admissions included)
  strata batch: DFlash2 accepted 468/571 offered proposals (82.0%), 1.85 tokens per slot-round over 551 slot-rounds; windows averaged 3.85 slots x 1.13 proposal rows (asked depth 1)
  ```
- **Round 2 (Warm):** `1024 tokens in 7.12 s` -> **`143.76 tok/s` (`35.94 tok/s/slot`)**

---

## Part 5: Further Recommendations (Wave 3) for Batched/Concurrent Request Processing

With `75cd16bd` promoted, the remaining time in a `4 × 256` concurrent burst (`7,120 ms` HTTP wall time) splits into:
1. **`143` Batch Decode Windows (`6,072 ms` = `42.46 ms/window`):**
   - `CUDA0 GPU-reach wait`: `25.66 ms/win` (`535 µs/layer`)
   - `CPU experts`: `8.34 ms/win` (`174 µs/layer` across `4.49` non-resident experts/layer)
   - `emit` (DFlash2 `batch_body` + `batch_head` after Layer-36 context overlap): `5.46 ms/win` (`3.88 ms/win` at `c=2`)
   - Host bookkeeping / `adapt wait`: `3.00 ms/win`
2. **Sequential `BGEN` Request Admissions (`1,048 ms` = `14.7%` of total wall time!):**
   - Admitting 4 short 40-token prompts sequentially through `ss` (session 0) costs `4 × (~215 ms prefill + 19 ms 1-row verify + 25 ms copy_to_slot) = ~1,048 ms` before all 4 slots are decoding together.

Here are the **4 highest-leverage Wave 3 recommendations**, ranked by impact:

### Recommendation 1 (Highest Impact on Concurrent Bursts — Est. `-750 to -850 ms` per 4-request burst, `+12% to +15%` HTTP wall `tok/s`): Direct-to-Slot Admission & Multi-Request Short-Prompt Coalescing
- **Where:** `src/program/generate.cpp` (`BGEN` admission handler & `copy_to_slot`)
- **Bottleneck:**
  - Every `BGEN <slot>` request currently runs its prompt prefill and 1-row initial verify on the main solo `SessionState ss` (`slot 0`), and then copies GDN states, non-pooled QSA states, and DFlash2 KV rings from `ss` (`slot 0`) into `*bslot_ss[0][slot]` (`copy_to_slot`, `~25 ms/request`), plus another `copy_from_slot` (`~19 ms`) if a slot finishes or parks.
  - Furthermore, when 2–4 requests arrive simultaneously in `in_lines`, each short prompt (`T = 38..43` tokens) runs a separate 48-layer `Prefill::run` pass (`~215 ms` each, where `T = 40` severely under-utilizes `Prefill::run`'s GEMM tiles and streams weights 4 separate times!).
- **Recommended Implementation:**
  1. **Direct-in-Slot `BGEN` Admission:** Bind `Prefill::run`, the 1-row admission `ver.run`, and `df2.set_slot(admit_slot)` **directly to `*bslot_ss[0][admit_slot]`** when `admit_slot >= 0`, completely eliminating `copy_to_slot` (`~100 ms` saved per 4-request burst).
  2. **Verify-Window Admission for Short Prompts (`T_new <= 8`) / Multi-Request `BGEN` Coalescing:** When multiple `BGEN` requests are queued in `in_lines` at once, coalesce their prompt chunks or admit short uncached tails (`T_new <= 8`) directly inside `ver.run_slot_rows` alongside active decoding slots. Cutting 4-request admission from `1,048 ms` to `~250 ms` raises `c=4` HTTP wall throughput from **`143.8 tok/s` to `~162 tok/s` (`>40 tok/s/slot`)**.

### Recommendation 2 (Est. `-1.6 to -2.0 ms/window` at `c=4`, `+4% to +5%` decode `tok/s`): CUDA Graph Capture (`batch_body_exec_[n_items]`) & Multi-Slot `df2_attention_f16` (`blockIdx.z = idx`)
- **Where:** `src/core/dflash2.cpp:950–1068` and `src/kernels/cuda/dflash2.cu:215–315`
- **Bottleneck:**
  - With `precompute_batch_context` overlapped behind Target Layers 37–47, `draft_batch_begin` only needs to run `df2_kv_ring_f16` + the 5-layer `batch_body` + `batch_head`.
  - However, `use_batch_body` currently launches **~70 individual HIP kernels eagerly on `cs_`** every window (including `5 × 2 × n_items = 40` per-slot `df2_kv_ring_f16` and `df2_attention_f16` launches at `c=4`), which accounts for the `1.58 ms/window` gap between `c=2` (`emit = 3.88 ms`) and `c=4` (`emit = 5.46 ms`).
- **Recommended Implementation:**
  1. Extend `df2_kv_ring_f16` and `df2_attention_f16` to take `blockIdx.z = idx` (`0 .. n_items - 1`), reading per-slot `slot_id`, `ring_len`, and `write_off` from `d_meta_ + 64` so all `n_items` slots run in **1 launch per layer** instead of `n_items` launches (`30` fewer kernel launches/window at `c=4`).
  2. Capture the entire `use_batch_body` sequence (`df2_embed_tokens_batch` through `df2_pack_head_rows`) into a `cudaGraphExec_t batch_body_exec_[kDf2MaxSlots]` indexed by `n_items` (`2, 3, 4`). Because all device buffer addresses (`x_`, `xq_`, `xd_`, `qkv_`, `xqp_`, `xdp_`, `fgu_`, `d_meta_`) are fixed when `can_use_precomp` is true, graph replay will drop `c=4` `emit` from `5.46 ms` to **`~3.5–3.8 ms/window`**.

### Recommendation 3 (Est. `-2.0 to -3.5 ms/window` on `run`): Early Router Doorbell (`STRATA_BATCH_EXPERT_AHEAD`) & `M_e`-Bucketed Resident MoE Dispatch
- **Where:** `src/core/verify.cpp:3625–3740` and `src/kernels/cuda/s2_expert_grouped.cu`
- **Bottleneck:**
  - In `75cd16bd` at `c=4`, `run = 36.47 ms/window` comprises `CUDA0 GPU-reach wait = 25.66 ms` + `CPU experts = 8.34 ms` (`4.49` CPU misses/layer at `93.4%` VRAM hit rate).
  - Whenever a layer has a burst of `7–10` CPU expert misses, the 15-worker CPU pool takes `~250–320 µs` while the GPU's resident MoE kernel takes `~210 µs`, stalling the GPU on `wait_flag_ge(flagB)`.
- **Recommended Implementation:**
  1. **Fire the CPU pool doorbell (`flagA`) immediately after `moe_router_topk`** (before the shared expert or dense attention/GDN projections finish on the GPU) so the 15 CPU worker threads get an extra `~80–100 µs/layer` (`~4.3 ms/window`) head start on non-resident experts while preserving 100% bit parity.
  2. **Sort `plan.res_pairs` descending by token multiplicity `M_e` (`M_e >= 2` before `M_e == 1`):** In an 8-row window (`80` routed pairs across `~58` distinct resident experts), running multi-token experts first and single-token experts at the tail reduces wave tail-effect across the 64 WGPs before `wait_flag_ge`.

### Recommendation 4 (Ultra-Low Inter-Token Jitter Under Long 60k–200k Prefills): Mid-Chunk Layer-Boundary Decode Yielding (`STRATA_BATCH_PREFILL_LAYER_YIELD=16`)
- **Where:** `src/prefill/prefill.cpp` and `src/program/generate.cpp`
- **Bottleneck:**
  - `STRATA_BATCH_PREFILL_CHUNK=2048` + rank-major expert lending raised concurrent decode during prefill from `~8 tok/s/slot` to **`35.20 tok/s/slot` (`70.39 tok/s` aggregate)**.
  - Each `2,048`-token prefill chunk takes `~1.0 s` on `gfx1201`, meaning active decode slots pause for `~1.0 s` during a chunk and then burst `~35–45` tokens during the `piped_decode` yield between chunks.
- **Recommended Implementation:**
  - Inside `Prefill::run` (`src/prefill/prefill.cpp`), only `X` (`[2048, 2560]` FP16 = `10 MiB`) and `R` (`[2048, 4, 2560]` FP16 = `40 MiB`) are live between layers (`for l = 0..47`), and `Verifier::run_slot_rows` uses independent activation buffers (`b_act_`, `b_res_`).
  - Adding an optional layer-boundary callback every `16` layers (`l = 15, 31`) that yields to `1–2` `step_batch()` decode windows caps worst-case inter-token latency (ITL) during a 100k prefill to **`~300 ms`** while keeping the `2,048`-row prefill GEMM tile intact.

---

## Part 6: Exact Reproduction Commands

```bash
# 1. Build strata + unit & parity tests in Strata-integ
systemd-run --user --scope -p MemoryMax=6G -- cmake --build /home/soren/Strata-integ/build-verify \
  --target strata dflash2_small_bench dflash2_slots_test dflash2_tap_test expert_cache_per_layer_test batch_read_test -j8

# 2. Run CPU & GPU parity CTests
systemd-run --user --scope -- env PYTHONPATH=/home/soren/Strata-integ/third_party/llama.cpp/gguf-py \
  LD_LIBRARY_PATH=/opt/rocm/lib ctest --test-dir /home/soren/Strata-integ/build-verify --output-on-failure \
  -R '^(dflash2_slots_test|dflash2_tap_test|expert_cache_per_layer_test|batch_read_test|arch_defaults_test|mmvq_il_parity|quantize_il_parity|batch_mtp_depth_test|batch_mtp_select_test)$'

# 3. Run multi-slot DFlash2 kernel bitwise parity (ncols = 16, 24, 32)
systemd-run --user --scope -- env LD_LIBRARY_PATH=/opt/rocm/lib /home/soren/Strata-integ/build-verify/dflash2_small_bench
```
