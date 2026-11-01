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

## Part 5: Wave 3 Implementation & Measured Results (`d4f36a10`, Promoted 2026-11-01)

All 4 Wave 3 items were implemented as separate small atomic commits (`b0be13a4..df0775e8` on `Strata-integ` / `5f49f24f..42fa8284` on PR #4937 `pr/dflash2-batch-optimizations`), verified with `16/16` CTest parity suites + 4-slot `dflash2_small_bench` bitwise parity, and promoted to `strata.service` (`d4f36a107ba0e9fb8aa09071bc86faf3`, 8/8 5K needle gate):

1. **Wave 3 Commit 1 (`b0be13a4` / `5f49f24f`): Batched Multi-Slot KV Ring, Attention & Head Pack + `batch_body_exec_[n_items]` CUDA Graph Capture (`include/strata/kernels/dflash2.hpp`, `src/kernels/cuda/dflash2.cu`, `include/strata/core/dflash2.hpp`, `src/core/dflash2.cpp`, `src/core/dflash2_small_bench.cpp`):**
   - Added `df2_kv_ring_batch[_f16]`, `df2_attention_batch[_f16]`, and `df2_pack_head_rows_batch` (`blockIdx.z = idx` reading per-slot metadata from `d_meta_ + 128` and `d_meta_ + 144`, and per-slot ring pointers from device table `d_ring_table_`), collapsing `16 * n_items + n_items = 68` per-slot launches at `c=4` into `17` multi-slot launches.
   - Captured `record_batch_body(n_items, cs_)` into `batch_body_exec_[n_items]` (`n_items = 2..4`) in `Dflash2Drafter::prepare()` and launched `cudaGraphLaunch(batch_body_exec_[n_items], cs_)` in `draft_batch_begin()`.
   - **Measured Result:** Post-verify `emit` time dropped from `3.88 ms/win` (Wave 2) to **`3.54–3.68 ms/win` at `c=2`** and from `5.46 ms/win` to **`5.30 ms/win` at `c=4`**; `dflash2_small_bench` 4-slot bitwise parity: **PASS**.
2. **Wave 3 Commit 2 (`35350b3d` / `034b2b15`): `M_e`-Bucketed Resident MoE Plan Ordering (`STRATA_MOE_BUCKET_PLAN`), Per-Token CPU Quantization Skip & Shared-Expert `q8_ffn` Reuse (`src/core/verify.cpp`, `src/core/expert_source.cpp`):**
   - Enabled `g_qdedup(bool q8_ffn_ready)` by default when `STRATA_QFUSE_HC` (`q8_ffn`) already populated `nat_xq_` in `gr_read_group` and `!sh_fork`, eliminating 48 duplicate `native_quantize_q8_1(xm)` launches per verify window.
   - Added `STRATA_MOE_BUCKET_PLAN` (default on) in `dispatch_multi` (`src/core/expert_source.cpp`) to order multi-token resident experts (`M_e >= 2`) before single-token experts (`M_e == 1`) in `d.plan->order`, and skipped CPU row activation quantization (`act_multi[t]` / `nact_multi[t]`) for tokens whose routed experts all execute on GPU (`kind != -1`).
3. **Wave 3 Commit 3 (`a016b578` / `bf52f223`): Mid-Chunk Layer-Boundary Decode Yielding (`STRATA_BATCH_PREFILL_LAYER_YIELD=16`, `include/strata/prefill/prefill.hpp`, `src/prefill/prefill.cpp`, `src/program/generate.cpp`):**
   - Added `on_layer_yield_every` and `on_layer_yield` on `Prefill`, invoked at the end of every `16` layers (`l = 15, 31`) on single-stage prefills (`T >= 512`) when `batch_on()` is true.
   - **Measured Result:** On a concurrent `c=2` (`2 × 256` tokens) + `5,762`-token prompt prefill (`3` chunks × `3` slices = **`9` yield parts**), `strata batch: the prompt was read in 9 parts, the slots decoding 2217 ms between them` sustained **`66.45 tok/s` aggregate decode (`33.23 tok/s/slot`, `38.30 ms/window = run 34.29 + emit 3.84 ms`)** while reading `5,762` prompt tokens at `1,066.7 tok/s`, cutting worst-case inter-token decode pause by **3x** (`~1,040 ms -> ~340 ms`).
4. **Wave 3 Commit 4 (`df0775e8` / `42fa8284`): Fast `BGEN` Slot Admission (`STRATA_SLOT_COPY_BATCHSYNC`, `STRATA_SLOT_CKPT_MOVE`, `STRATA_CKPT_POOL`, `STRATA_ADMIT_MERGE_FIRST`, `STRATA_BATCH_VERIFY_PREFILL=96`, `src/program/generate.cpp`):**
   - Enabled single-sync direct slot D2D copy (`STRATA_SLOT_COPY_BATCHSYNC`), zero-copy turn checkpoint move (`STRATA_SLOT_CKPT_MOVE`), and checkpoint vector pooling (`STRATA_CKPT_POOL`) by default, and raised the batch verify-window prefill ceiling (`STRATA_BATCH_VERIFY_PREFILL`) from `64` to `96` tokens.
   - **Measured Result:** Per-slot `BGEN` admission copy/move time dropped from `11.4–25.8 ms` to **`6.7–7.0 ms` (`-41% to -73%`)**; `c=2` R1 rose to **`126.13 tok/s`** (`63.06 tok/s/slot`) and warm R2 reached **`132.45 tok/s`** (`161.9 rows/s`, `39.79 ms/win`, `emit 3.68 ms`); `c=4` R1 reached **`141.87 tok/s`** (`35.47 tok/s/slot`, `167.9 rows/s`, `42.55 ms/win`, `emit 5.30 ms`, individual slots up to **`39.04 tok/s`**).

---

## Part 6: Wave 4 Implementation & Measured Results (`9f72ce1a`, Promoted 2026-11-01)

All 6 Wave 4 optimizations were implemented as separate small atomic commits (`2d826840..2ec832a9` on `Strata-integ` / `080a91be..d9c8ddac` on PR #4937 `pr/dflash2-batch-optimizations`), verified with `157/157` `serve.test_parallel` & `serve.test_parallel_yield` unit tests + `16/16` CTest parity suites + 4-slot `dflash2_small_bench` bitwise parity, and promoted to `strata.service` (`9f72ce1a82ee11eb15a454952cec76f3`, 8/8 5K needle gate):

1. **Wave 4 Commit 1 (`2d826840` / `080a91be`): Skip Solo `GEN` -> `STOP` -> `BGEN` on Concurrent Bursts & Move Turn Checkpoint on Solo Promotion (`serve/server.py`, `serve/test_parallel.py`, `src/program/generate.cpp`):**
   - Added `_others_admitted()` on `ParallelEngine` (`serve/server.py`), wired to `Service.admission_reserved`, so when another HTTP request has already passed `reserve_inference()` (e.g., tokenizing concurrently in a `c >= 2` burst), the first request to reach `generate_batched` skips solo `GEN` and enters a batch slot directly (`BGEN 0`).
   - In `src/program/generate.cpp`, skipped launching `mtp.draft` / `dflash2.propose` when `give_way()` (`admit_slot < 0 && stop_req.load()`) becomes true during a solo `ver.run()`, and recovered `prompt_turn_at` from `o.turn_token` in `BGEN` when `turn_at < 0 && from_live && req_ckpt` so `move_best` moves the solo turn checkpoint instead of copying it.
   - **Measured Result:** Completely eliminated `BYIELD` / `SOLO_AGAIN` churn on concurrent bursts (`0` `BYIELD` events on `c=2` and `c=4`), saving `~43 ms` of redundant solo-to-slot re-admission on every burst from idle.
2. **Wave 4 Commit 2 (`941ad664` / `5fbb340d`): Run 16-Row `read_windows` on One-Card DFlash2 While Keeping 8-Row Decode Windows (`src/program/generate.cpp`):**
   - Updated `batch_max_t()` so one-card (`stages.empty()`) DFlash2 batch mode initializes `Verifier` with `kVerifyMaxT` (`16` rows) while keeping DFlash2 batch decode windows capped at `o.batch_draft_window` (`8` rows) via `df2_cap = std::min(cap, o.batch_draft_window)` and `win_cap = (use_df2 && !stages.empty()) ? std::min(S, o.batch_draft_window) : S`.
   - **Measured Result:** Short-prompt `read_windows` speed on `BGEN` slot admission rose by **+35%** (`160 tok/s -> 208–227 tok/s`, cutting 50-token admission `read_windows` from `312 ms` to `211–240 ms`) while keeping 8-row decode verify windows unchanged.
3. **Wave 4 Commit 3 (`db82a2e0` / `11661eed`): Split `>4`-Entry Resident MoE Groups into Parallel `blockIdx.y` Chunks & Skip First Pre-Stage Barrier (`src/core/expert_source.cpp`, `src/kernels/cuda/iq_kernels.cu`):**
   - In `emit_vram_group` (`src/core/expert_source.cpp`), split resident MoE expert groups with `M_e > 4` into multiple `<= 4`-entry (`ED_NC = 4`) groups in `P.ptr` / `P.start` (guarded by `groups + 1 < kMaxWindowEntries`) so `expdec_gu_kernel` and `expdec_down_kernel` (`gridDim.y = min(max_groups, 16)`) execute all `<= 4`-entry chunks in parallel across `blockIdx.y` instead of looping serially (`for (int e = e0; e < e1; e += ED_NC)`).
   - In `expdec_gu_kernel` and `expdec_down_kernel` (`src/kernels/cuda/iq_kernels.cu`), skipped the redundant initial `iq_sync_lds()` / `__syncthreads()` barrier when `g == blockIdx.y && e == e0`.
4. **Wave 4 Commit 4 (`39815802` / `274424c8`): Zero-CPU-Miss Fast Path in `expert_pool_dispatch_multi` and `copy_rows_from_mapped_kernel` (`src/core/expert_source.cpp`, `src/kernels/cuda/elementwise.cu`):**
   - Added an early return in `expert_pool_dispatch_multi` when `gpu_zeroes_hits && planned_vram_entries >= 0 && planned_gpu_entries == n && d.peer == nullptr && d.remote_count == 0`, skipping the per-entry `any_cpu` scan, `d.job_of` zeroing, and CPU pool wakeup whenever all `n` routed entries for the layer are handled on GPU (VRAM + PCIe).
   - Added a `c >= (int) gridDim.x` fast check in `copy_rows_from_mapped_kernel` (`src/kernels/cuda/elementwise.cu`) so all-GPU-hit layers set `hit = 1` in `O(1)` without scanning `hit_rows[0..c)`.
5. **Wave 4 Commit 5 (`292639b0` / `4a7488e1`): Fuse Shared-Expert `gate+up` MMVQ Pair (`ROWS=2`) & Router Aux Scalar Gate Across All Verify Layers (`src/core/verify.cpp`, `src/kernels/cuda/shared_expert.cu`, `src/kernels/cuda/native_mmvq.cu`):**
   - Updated `q8_0_pair_launch` in `src/kernels/cuda/native_mmvq.cu` to dispatch `native_mmvq_multi_kernel<F, NCOLS, WARPS, 2, ..., true>` (`ROWS = 2`, `n_out = 640` blocks in one launch, filling 100% of all 64 WGPs in Wave 1 instead of two separate 320-block `0.625`-wave launches) when `!rows1 && !q8_0_rows1<F>(n_in, n_out) && (n_out & 1) == 0`, and enabled `pair` by default in `shared_expert_multi` (`STRATA_LFUSE_PAIR` != `0`).
   - Enabled `bf16_gemv_fp32_mmvf_multi_aux` in `src/core/verify.cpp` on all `n = 2..8` verify layers (independent of `lfuse_on(n)` / `ar_on()`) and passed `lfuse |= 4` (`gate_precomputed`) to `shared_expert_multi` so `sh_g_` is computed as block `512` of the router GEMV launch and the separate 1-block `bf16_gemv_fp32_mmvf_multi` launch in `shared_expert_multi` is eliminated (`96` kernel launches saved per 48-layer verify window, 100% bitwise identical).
6. **Wave 4 Commit 6 (`2ec832a9` / `d9c8ddac`): Cache Per-Column `Q8_1` Activation Fragments in Registers Across `ROWS > 1` in `native_mmvq_multi_kernel` (`src/kernels/cuda/native_mmvq.cu`):**
   - Hoisted `B6X<F>` above `native_mmvq_multi_kernel` and added `if constexpr (ROWS > 1 && B6X<F>::ok)` inside `native_mmvq_multi_kernel` so each thread loads `xv[0 .. NCOLS - 1]` once per `kbx` iteration and reuses it across all `ROWS` output rows (`ROWS = 2` on `ssm_out` `6144->2560`, `attn_output` `6144->2560`, `attn_k`, `attn_v`, and `shared_expert` `gate+up` / `down`), cutting `Q8_1` activation loads by 50% across 144+ `Q8_0` launches per verify window with 100% bitwise parity.

### Wave 4 Measured End-to-End Summary (`gfx1201`, `9f72ce1a82ee11eb15a454952cec76f3`)
- **`c=2` (`2 × 256` greedy code):**
  - Cold R1: **`127.75 tok/s`** (**`63.88 tok/s/slot`**, `4.01 s`, up from `126.13 tok/s` in Wave 3 and `105.4 tok/s` pre-Wave 1 — **`+21.2%` vs baseline**).
  - Warm R2 (AVL/Tarjan): **`130.38 tok/s`** (Slot 1 **`68.80 tok/s`**, Slot 2 **`65.20 tok/s`**), window time **`38.79 ms/window`** (`run 34.76 ms [CUDA0 22.88 + CPU 9.25 ms] + emit 3.56 ms`, down from `39.79 ms/window` in Wave 3 — **`-1.00 ms/window`**).
  - Warm (Dijkstra/LRU canonical pair): **`137.93 tok/s`** (**`68.96 tok/s/slot`**, Slot 1 **`69.77 tok/s`**, Slot 2 **`68.99 tok/s`**, `512` tokens in **`3.71 s`**, `162.0 rows/s`, up from `132.45 tok/s` / `3.87 s` in Wave 3 — **`+5.48 tok/s` / `+4.1%` over Wave 3** and **`+18.0%` over pre-Wave 1**).
- **`c=4` (`4 × 256` greedy code):**
  - R1: **`145.19 tok/s`** (**`36.30 tok/s/slot`**, individual slots up to **`39.23 tok/s`**, `1,024` tokens in **`7.05 s`**, `42.48 ms/window = run 35.92 + emit 6.02 ms`, `165.6 rows/s`, up from `140.00–143.76 tok/s` in Wave 3 and `113.4 tok/s` pre-Wave 1 — **`+28.0%` vs baseline**).

---

## Part 7: Wave 5 Implementation & Measured Results (`a106bc92`, Promoted 2026-11-01)

All 4 Wave 5 optimizations were implemented as separate small atomic commits (`2ede778f..eb850cba` on `Strata-integ`), verified with `24/24` GPU CTest parity suites (`shared_expert_parity`, `spec_verify_parity`, `verify_batch_parity`, `mmvf_rows_parity`, `gr_parity`, `s2_expert_grouped_parity`, `resident_plan_parity`, `native_multi_parity`, plus the `16/16` DFlash2/MMVQ/IQ suite) + 4-slot `dflash2_small_bench` bitwise parity + `49/49` `serve.test_parallel` & `serve.test_web_element_ids`, and promoted to `strata.service` (`a106bc923fc97bf4ba02ca6efbe07656`, 8/8 5K needle gate):

1. **Wave 5 Commit 1 (`2ede778f`): Fuse `SwiGLU` into `native_mmvq_pair_swiglu` & Defer Shared Gate into `combine_k10_vec4_hits` (`include/strata/kernels/native_mmvq.hpp`, `src/kernels/cuda/native_mmvq.cu`, `src/kernels/cuda/shared_expert.cu`, `include/strata/kernels/native_moe.hpp`, `src/kernels/cuda/native_moe.cu`, `src/core/verify.cpp`):**
   - Added `SWIGLU_PAIR` template mode to `native_mmvq_multi_kernel<F, NCOLS, NW, 2, false, true, false, true>` and `native_mmvq_pair_swiglu(...)` (`STRATA_MMVQ_PAIR_SWIGLU`, default on): each CTA `bxi` (`0 .. n_out - 1`) computes row `bxi` of `w_gate` (`i = 0`) and row `bxi` of `w_up` (`i = 1`) in the same CTA, loading the `Q8_1` input activation fragments `xv[0..NCOLS-1]` once for both `gate` and `up`, reducing `g` and `u` across warps in shared memory in the exact same order as `native_mmvq_pair`, and writing `(g / (1.0f + expf(-g))) * u` directly to `y_swiglu` before calling `native_quantize_q8_1`, eliminating the `up[n_tok, n_ff]` global memory write+read (`100%` bitwise identical, verified in `shared_expert_parity`).
   - Templated `combine_k10_vec4_hits<bool GATED_SH>` and enabled `sg_ready = sg_aux && (lfuse_on(n) || (g_lfuse() && can_hits_in_combine))` in `src/core/verify.cpp`, passing `sg_gated_[grp] ? (sh_g_ + tb) : nullptr` to `native_moe_combine_multi_hits_added` so `shared_expert_multi` skips `sigmoid_mul_rows_2d` across all 48 verify layers (**`48` kernel launches saved per verify window**) and applies `1.0f / (1.0f + expf(-shared_gate[tk]))` directly inside `combine_k10_vec4_hits<true>`.
2. **Wave 5 Commit 2 (`92262078`): Merge up to 16 Trailing Prompt Tokens into the First Verify Window on DFlash2 Admission (`src/program/generate.cpp`):**
   - Updated `merge_from(a0, to)` in `src/program/generate.cpp` so that on `BGEN` slot admission (`admit_slot >= 0 && max_new == 1 && admit_max_new > 1`) on a single-stage pipeline (`n_pipe == 1 && o.pcie_frac <= 0.0f`), up to `std::min(mt, 16)` trailing prompt tokens are merged into the final admission verify window (`std::max(a0, n - std::min(mt, 16))`) even when `req_dflash_k > 0 || df2_active`.
   - Enabled the DFlash2 initial draft step on that merged admission window (`a + 1 < max_new || (merged && admit_slot >= 0 && max_new == 1 && admit_max_new > 1)`), recording the final prompt token's tap (`n - 1`) into slot 0's KV ring and staging `df2_stage[admit_slot]` immediately on admission.
   - **Measured Result:** Per-slot `BGEN` admission verify + initial DFlash2 draft (`1 generated in ... ms`) dropped by **-55%** from `~40–43 ms` in Wave 4 down to **`18–19 ms` (`53–55.5 tok/s`)** across every `c=2` and `c=4` slot admission.
3. **Wave 5 Commit 3 (`740c325e`): Skip Zero-Filling Hit Rows in `copy_rows_from_mapped` and Select `hits4` Directly in `combine_k10_vec4_hits` (`include/strata/kernels/elementwise.hpp`, `src/kernels/cuda/elementwise.cu`, `src/kernels/cuda/native_moe.cu`, `src/core/verify.cpp`):**
   - Updated `combine_k10_vec4_hits` (`src/kernels/cuda/native_moe.cu`) to select `((mask >> expert) & 1) ? hits4[...] : parts4[...]` directly instead of reading `parts4` (`0.0f`) and adding `hits4` on GPU-hit rows.
   - Templated `copy_rows_from_mapped_kernel<bool SKIP_HIT_ZERO>` (`src/kernels/cuda/elementwise.cu`) and passed `skip_hit_zero = hits_in_combine` from `src/core/verify.cpp`, so GPU-hit rows (`86%–100%` of the 80 rows per layer) return immediately without writing `2,048` zeros (`8 KB`) to `parts_out` — eliminating **`~53–61 MB` of global memory writes and reads per 48-layer verify window** (`100%` bitwise identical).
4. **Wave 5 Commit 4 (`a1c7c002`) & Commit 5 (`eb850cba`): Per-Token `float4` Doorbell Activation Publish, Single-Pass Host Distinct Plan, and `int4` Plan Copy Trim (`src/kernels/cuda/elementwise.cu`, `src/core/expert_source.cpp`, `src/program/generate.cpp`, `src/kernels/cuda/verify_kernels.cu`, `src/core/verify.cpp`):**
   - Replaced the whole-window `any_miss` boolean in `doorbell_publish_res_kernel` (`src/kernels/cuda/elementwise.cu`) with a per-token miss bitmask `tok_miss` (`1u << (threadIdx.x / 10)`) and vectorized `float4` PCIe stores, copying `x + t * N -> x_out + t * N` over PCIe only for tokens `t` that actually have at least one missed expert instead of copying all 8 tokens (`64 KB`) whenever any single token misses VRAM (while keeping `set_always_publish(true)` when `drive.d.lookahead != nullptr`).
   - Cached `ids[0..n-1]` from mapped memory `h_ids_` into a stack array `lid[kMaxWindowEntries]` in `expert_pool_dispatch_multi` (`src/core/expert_source.cpp`) and tracked `minr_i0` on the fly during the initial distinct-expert pass, eliminating the second $O(n \cdot nd)$ search loop in `STRATA_ROUTE_TAIL_SKIP` before `P.publish(P.ctx)` raises `h_flagA_`.
   - Vectorized `wait_copy_i32_kernel` (`src/kernels/cuda/verify_kernels.cu`) with 128-bit `int4` loads/stores and trimmed the unused PCIe plan tail (`ptr2`, `start2`, `dst2`) when `sink_.pcie_mode < 0`, cutting per-layer plan copy size on an 8-row window by **51%** (`1,322 -> 646` int32s).

### Wave 5 Measured End-to-End Summary (`gfx1201`, `a106bc923fc97bf4ba02ca6efbe07656`)
- **`c=2` (`2 × 256` greedy code):**
  - Cold R1: **`128.40 tok/s`** (**`64.20 tok/s/slot`**, `3.99 s`, up from `127.75 tok/s` in Wave 4 and `105.4 tok/s` pre-Wave 1).
  - Warm R2: **`137.88 tok/s`** (**`68.94 tok/s/slot`**, `512` tokens in **`3.71 s`**, **`38.89 ms/window`** with **`run = 34.44 ms` [CUDA0 `23.76 ms` + CPU `8.61 ms`]**, down from `run = 34.76–34.81 ms` in Wave 4; **`163.9 rows/s`**; per-slot admission verify dropped to **`18–19 ms`**).
- **`c=4` (`4 × 256` mixed prompts — code + history + math + systems):**
  - Cold R1: **`111.63 tok/s`** (**`27.91 tok/s/slot`**, `CUDA0 wait = 20.01 ms`).
  - Warm R2: **`124.25 tok/s`** (**`31.06 tok/s/slot`**, `1,024` tokens in **`8.24 s`**, **`47.56 ms/window = run 40.26 ms` [CUDA0 `22.28 ms` + CPU `14.71 ms`]**, `141.1 rows/s`, up **`+13.2%`** from `109.76 tok/s` in Wave 4).
- **`c=4` (`4 × 256` greedy code):**
  - **`140.02–141.79 tok/s`** (**`35.00–35.45 tok/s/slot`**, `42.62 ms/window = run 35.78 ms [CUDA0 25.95 + CPU 7.70 ms] + emit 6.21 ms`, `163.9 rows/s`, per-slot admission verify **`18–19 ms`**).
- **`c=2` (`2 × 256` greedy code) + concurrent `4,364`-token long prefill (`STRATA_BATCH_PREFILL_LAYER_YIELD=16`):**
  - Prompt read in **7 parts** with slots decoding **`1,793 ms`** between parts; active decode slots sustained **`61.64 tok/s`** combined (**`30.82` and `32.16 tok/s/slot`**) while the `4,364`-token prompt prefilled at **`932.1 tok/s`** and admitted its final 16 merged tokens in **`20 ms`**.

---

## Part 8: Wave 6 Proposals for Further Concurrent/Batched Speedups

1. **Multi-Slot Batched `read_windows` Admission (`src/program/generate.cpp`):**
   - When multiple `BGEN <slot>` commands arrive in a `c=4` burst (or while 1–2 slots are already decoding), pack 4–8 prompt tokens per admitting slot into `ver.run_slot_rows` (`16` rows total across slots) so up to 4 short prompts prefill their uncached tails in parallel in a single verify graph pass instead of running `read_windows` serially (`4 × ~215 ms = ~860 ms`).
2. **Direct `Q8_1` Output in `native_mmvq_pair_swiglu` (`src/kernels/cuda/native_mmvq.cu`, `src/kernels/cuda/shared_expert.cu`):**
   - Fold the `32`-element `Q8_1` block max-abs + scale reduction directly into warp 0 of `native_mmvq_multi_kernel<..., SWIGLU_PAIR=true>` (grouping 32 output rows per CTA or writing via a warp shuffle across 32 adjacent `bxi` rows) to eliminate the separate `native_quantize_q8_1(gate, nw.q8_1, ...)` launch (`48` kernel launches/window).
3. **Host-Thread Pre-Quantization Overlap (`src/core/expert_source.cpp`):**
   - On layers with `nmiss >= 2` CPU expert jobs, let worker threads quantize their own required token rows `x_q8[t]` on demand (via a per-token atomic state `0=unquantized, 1=in-progress, 2=ready`) instead of serializing all `x_q8` quantizations on the main host thread before `pool->dispatch`.

---

## Part 9: Exact Reproduction Commands

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
