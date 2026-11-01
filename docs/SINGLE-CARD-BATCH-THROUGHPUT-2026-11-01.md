# Single-Card `gfx1201` Batched Throughput Optimization (2026-11-01)

## 1. Summary

Implemented, unit-tested, A/B benchmarked on `gfx1201` (AMD Radeon AI PRO R9700, 32 GiB VRAM, `swift-1.5-iq3_s`), and promoted to production (`strata.service`, binary md5 `afc6ae5f487934c6ca07eafd07c0ec4e`) all 9 single-card batched decode optimizations across `c=2` (2 concurrent requests) and `c=4` (4 concurrent requests) with `--batch-draft-window 8 --batch-draft-depth 1`.

### Headline Measured Results (`gfx1201`, `parallel: 4`, 256 tokens/request, `temperature=0`)

| Workload | Metric | Prod Baseline (`ccd1b635`) | Optimized (`afc6ae5f`, `B0`) | Delta |
|---|---|---|---|---|
| **`c=2` Cold (R1)** | Window time (`ms/window`) | 50.90 ms | **43.80 ms** | **−13.9%** |
| **`c=2` Cold (R1)** | Engine batch rate (`rows/s`) | 125.1 rows/s | **144.4 rows/s** | **+15.4%** |
| **`c=2` Cold (R1)** | Client HTTP wall (`tok/s`) | 105.4 tok/s (52.7/slot) | **117.1 tok/s (58.5/slot)** | **+11.1%** |
| **`c=2` Warm (R2)** | Window time (`ms/window`) | 43.52 ms | **40.34 ms** | **−7.3%** |
| **`c=2` Warm (R2)** | Engine batch rate (`rows/s`) | 142.6 rows/s | **157.7 rows/s** | **+10.6%** |
| **`c=2` Warm (R2)** | Client HTTP wall (`tok/s`) | 120.6 tok/s (60.3/slot) | **128.4 tok/s (64.2/slot)** | **+6.5%** |
| **`c=2` Warm (R3)** | Window time (`ms/window`) | 43.46 ms | **39.99 ms** | **−8.0%** |
| **`c=2` Warm (R3)** | Engine batch rate (`rows/s`) | 144.6 rows/s | **157.1 rows/s** | **+8.6%** |
| **`c=2` Warm (R3)** | Client HTTP wall (`tok/s`) | 122.0 tok/s (61.0/slot) | **132.7 tok/s (66.4/slot)** | **+8.8%** |
| **`c=4` Cold (R1)** | Window time (`ms/window`) | 49.36 ms | **44.80 ms** | **−9.2%** |
| **`c=4` Cold (R1)** | Engine batch rate (`rows/s`) | 148.5 rows/s | **163.6 rows/s** | **+10.2%** |
| **`c=4` Cold (R1)** | Client HTTP wall (`tok/s`) | 124.5 tok/s (31.1/slot) | **137.6 tok/s (34.4/slot)** | **+10.5%** |
| **`c=4` Warm (R2)** | Window time (`ms/window`) | 49.20 ms | **44.98 ms** | **−8.6%** |
| **`c=4` Warm (R2)** | Engine batch rate (`rows/s`) | 149.0 rows/s | **163.0 rows/s** | **+9.4%** |
| **`c=4` Warm (R2)** | Client HTTP wall (`tok/s`) | 125.6 tok/s (31.4/slot) | **135.8 tok/s (34.0/slot)** | **+8.1%** |
| **`c=4` Warm (R3)** | Window time (`ms/window`) | 49.16 ms | **45.21 ms** | **−8.0%** |
| **`c=4` Warm (R3)** | Engine batch rate (`rows/s`) | 149.1 rows/s | **163.3 rows/s** | **+9.5%** |
| **`c=4` Warm (R3)** | Client HTTP wall (`tok/s`) | 125.3 tok/s (31.3/slot) | **136.1 tok/s (34.0/slot)** | **+8.6%** |

---

## 2. Item-by-Item Implementation & Findings (Items 1–9)

### Item 1: Promote `STRATA_DF2_BATCHHEAD=1` + single-card 8-row window (`W=8, depth=1`)
- **Change:** Promoted `Strata-integ` (`afc6ae5f487934c6ca07eafd07c0ec4e`) to `strata.service` with `STRATA_DF2_BATCHHEAD=1`, `--batch-draft-window 8 --batch-draft-depth 1`, and `"parallel": 4`.
- **Effect:** At `c=4`, `emit` + DFlash2 co-draft time drops from `9.58 ms/window` (`A_prod_baseline`) to **`7.86–7.92 ms/window`** (`7.63 ms/window` in `bt_draft`, `2.41 ms/call` across `3.17` slots). At `c=2`, `emit` drops from `5.23–5.31 ms/window` to **`4.84–4.89 ms/window`** (`4.77 ms/window` in `bt_draft`).

### Item 2: Fix CPU expert worker over-subscription (`--pool-workers 22` -> `15`) & VRAM reserve (`832` -> `576 MiB`)
- **Change:** Updated `/home/soren/Strata/strata-swift-iq3_s.json` from `--pool-workers 22` to `--pool-workers 15` (matching the 15 non-host physical cores of the 16-core CPU and eliminating the startup warning) and subtracted the duplicate `256 MiB` late-alloc reserve (`--vram-reserve-mib 832` -> `576`).
- **Effect:** Increases `expert cache auto` budget from `8,514` to **`8,614` slots (`+100` resident experts in VRAM)** with `668 MiB` VRAM still free after all graphs load, and trims 4-slot `CPU experts` (`ms_pool`) from `8.38 / 8.18 / 7.78 ms` to **`7.83 / 7.67 / 7.74 ms`**.

### Item 3: Default-enable `STRATA_BATCH_GDN_GROUPS=1` on HIP (`src/core/verify.cpp:350`)
- **Change:** Enabled `g_batch_gdn_groups` by default on `STRATA_USE_HIP` unless `STRATA_BATCH_GDN_GROUPS=0` is explicitly set.
- **Effect:** Collapses per-slot `gdn_step_norm_multi` launches across all 36 GDN layers in both `record_window` (`verify.cpp:1974`) and `capture_commit_batch` (`verify.cpp:4411`) into a single `gdn_step_norm_groups` launch per GDN layer (`108` fewer kernel launches per 4-slot window + commit).

### Item 4: Enable `STRATA_VERIFY_DEFER_CPU` / `STRATA_VERIFY_DEFER_CPU_BELOW` in batch windows & skip dead PCIe launches (`src/core/verify.cpp:2513, 2733, 3042, 3065, 5085`, `src/program/generate.cpp:10555`)
- **Change:**
  1. Guarded `deferred(l)` and `heavy_layer(T, l)` in `Verifier::record_window` with `!dflash2_tap_capture(tap_on_, batch_rec_, l, tap_slot_)` so DFlash2 tap layers (`1, 12, 23, 34, 45`) never defer their CPU expert contribution past tap capture.
  2. Enabled `defer_ok` for `batch_rec_` when `STRATA_VERIFY_DEFER_CPU=1` is set, wired `sink_.weights` / `sink_.partial` and `h_pflag_` in `Verifier::run_slot_rows` and `stage_batch`.
  3. In `Verifier::record_window` (`verify.cpp:2733`) and `generate.cpp:10555`, guarded the 4 PCIe-share kernel nodes per layer (`wait_flag_ge(m_flagB_)`, `fetch_blobs`, `rebase_ptrs`, `grouped(p_ptr2)`) on `sink_.pcie_mode >= 0` and set `eff_pcie_mode = -1` whenever no layer has a pinned PCIe complement (`!any_pcie_layer`, saving 192 kernel nodes per window when PCIe streaming is inactive).
- **Measured A/B (`B2` vs `B3`):** At warm steady state (`~94%` VRAM expert hit rate, `~4.0` CPU experts/layer-window), the 15-worker CPU pool finishes layer `l`'s CPU experts in `~160 µs/layer` (`7.67 ms / 48 layers`) while the GPU is still executing layer `l`'s `~74` VRAM experts + shared expert (`~220 µs/layer`). Because `h_flag_` is already raised before the GPU reaches `wait_flag_ge(m_flag_)`, `STRATA_VERIFY_DEFER_CPU=1` (`B3`: `45.37 / 45.41 / 45.20 ms/win`) is within noise of the exact bit-preserving path (`B2`: `44.79 / 44.85 ms/win`) at warm steady state and remains opt-in (since it alters output bits).

### Item 5: `STRATA_BATCH_DF2_DEPTH` (`auto|1|2|3|0`) & 2-slot vs 4-slot window sizing (`include/strata/core/batch_mtp_depth.hpp`, `src/program/generate.cpp:13354`)
- **Change:** Added `df2_depth_config` supporting `"auto"`, `"0"`, and explicit integer depths `1..kDrafts`, with unit tests in `tests/core/batch_mtp_depth_test.cpp`.
- **Measured A/B (`B1` with `STRATA_BATCH_DF2_DEPTH=3, floor=0.50` vs `B0` with `STRATA_BATCH_DF2_DEPTH=0`):**
  - On single-card `--batch-draft-window 8 --batch-draft-depth 1`, `df2_window_plan(8, n_ready, 6, 1)` already gives `1` proposal row/slot at `c=4` (`4 × 2 = 8` rows) and `3` proposal rows/slot at `c=2` (`2 × 4 = 8` rows).
  - Enabling `STRATA_BATCH_DF2_DEPTH=3, floor=0.50` at `c=2` (`B1`) throttled slots down to `2.31–2.35` proposals/slot (`5.36–5.80` rows/window instead of `6.28–6.36`), reducing tokens per slot-round from `3.26` to `2.80–2.88` and increasing total windows for `2×256` tokens from `80` to `95` (`122.5–129.2 tok/s` in `B1` vs **`128.4–132.7 tok/s` in `B0`**).
  - Keeping `STRATA_BATCH_DF2_DEPTH` off by default when unset (`default_auto = false`) preserves the full `3.26 tokens/slot-round` at `c=2` and `1.87 tokens/slot-round` at `c=4`.

### Item 6: Multi-row QSA batch attention (`STRATA_BATCH_ATTN_ROWS=1`) with `--kv-resident` & batched per-slot KV/indexer appends (`src/core/verify.cpp:350, 2106, 2248, 2292, 4445`)
- **Change:**
  1. Enabled `g_batch_attn_rows` by default on `STRATA_USE_HIP`.
  2. In `Verifier::record_window` (`verify.cpp:2292`), removed the `kv_mode != 1` exclusion on `rows_ok` and inserted per-slot grouped `qsa_kv_resolve` between `qsa_block_topk` and `qsa_decode_attn_batch_rows`.
  3. Grouped contiguous per-slot rows (`while (t < te && brow_[t] == brow_[first]) ++t;`) in `record_window` and `capture_commit_batch` to call `kv_append_q8_steps` / `kv_append_q4_steps` and `native_qsa_indexer_append_steps` once per active slot instead of once per row.
- **Effect:** Together with Item 3 and Item 7, reduces `CUDA0 GPU-reach wait` from `28.47 / 28.84 / 29.29 ms/window` (`A_prod_baseline`) to **`25.97 / 26.62 / 26.98 ms/window`** at `c=4` (`−2.3 to −2.5 ms/window`), and from `26.30 / 26.23 / 27.91 ms/window` to **`20.26 / 23.81 / 25.26 ms/window`** at `c=2` (`−2.4 to −6.0 ms/window`).

### Item 7: Move batch PLE `gather_batch` out of synchronous `stage_batch` into Layer 0 post-launch (`include/strata/core/verify.hpp:314`, `src/core/verify.cpp:4940, 5048`)
- **Change:** Stored `ple_rows_b_` and `ple_pending_b_ = true` in `Verifier::stage_batch` (when `!staging_deferred_ && lb_ == 0`), and executed `ss_->ple.table->gather_batch(ple_rows_b_, S, h_ple_, err)` inside `Verifier::run_slot_rows` immediately after `launch_batch()` + `cudaStreamQuery(cs_)` kicks off the GPU verify graph.
- **Effect:** Overlaps the 8-row host PLE table gather (`~0.2–0.4 ms/window`) with GPU Layer 0 token embedding, input RMSNorm, and GDN projections instead of delaying the GPU graph launch every window.

### Item 8: Overlap `Dflash2Drafter` batch co-drafting with async commit, PLE anchor prefetch, and token flush (`include/strata/core/dflash2.hpp:130`, `src/core/dflash2.cpp:748`, `src/program/generate.cpp:10581, 14463`)
- **Change:**
  1. Split `Dflash2Drafter::draft_batch` into `draft_batch_begin` (queues per-slot body graphs, `df2_pack_head_rows`, batched head GEMM, `df2_topk`, and `df2_select_multi` + async D2H copy onto `df2.stream()`) and `draft_batch_end` (`cudaStreamSynchronize(cs_)` + copies `h_batch_tok_` / `h_batch_prob_` into caller buffers).
  2. Enabled `Verifier::set_commit_async(stages.empty())` in `--serve` (`generate.cpp:10581`).
  3. Flushed `stdout` immediately after the token emit loop (`generate.cpp:14463`, before waiting on DFlash2 co-drafting) and overlapped `df2.draft_batch_begin` on `df2.stream()` with host PLE anchor row prefetch (`sx.ple.table->prefetch_rows`) and `ver.wait_commit(err)` (`commit_bm_` on `ver.cs_`).

### Item 9: Pruned DFlash2 `draft_vocab` (`40,525` tokens vs `106,299` tokens)
- **Measured A/B (`B0_integ_opt_106k` vs `B2_integ_opt_40k_vocab`):**
  - Switching `--mtp-draft-vocab` from `/home/soren/Strata-data/mtp/rt/draft_vocab.bin` (`106,299` tokens, `212.9 MiB` `Q6_K` head, `+396 MiB` draft head reserve) to `/home/soren/Strata-integ/data/draft_vocab_en.bin` (`40,525` tokens, `81.2 MiB` `Q6_K` head, `+262 MiB` draft head reserve) freed **`134 MiB` of VRAM** (`+53` expert cache slots, `8,614 -> 8,667`) and reduced 4-slot `bt_draft` from `7.63–7.69 ms/window` to **`7.36–7.46 ms/window`** (`−0.25 ms/window`) and 2-slot `bt_draft` from `4.77–4.82 ms/window` to **`4.20–4.43 ms/window`** (`−0.45 ms/window`).
  - However, because `STRATA_DF2_BATCHHEAD=1` already amortizes the `Q6_K` draft head GEMM across all slots in a single launch, the `0.25–0.45 ms/window` head speedup was outweighed on Python code generation by the missing `65,774` tail code/identifier tokens in `draft_vocab_en.bin` (`c=4` warm R2/R3: `135.8 / 136.1 tok/s` with `106k` vs `134.2 / 128.6 tok/s` with `40k`; `c=2` warm R2/R3: `128.4 / 132.7 tok/s` with `106k` vs `127.1 / 126.5 tok/s` with `40k`).
  - **Decision:** Keep the full `106,299`-token `draft_vocab.bin` in production on this 32 GiB `gfx1201` card (`668 MiB` VRAM remains free even with `parallel: 4` and `--vram-reserve-mib 576`).

---

## 3. Wave 2, Wave 3 & Wave 4 Follow-Ups (`75cd16bd` -> `d4f36a10` -> `9f72ce1a`)

See [`docs/SINGLE-CARD-BATCH-AND-PREFILL-RESEARCH-2026-11-01.md`](SINGLE-CARD-BATCH-AND-PREFILL-RESEARCH-2026-11-01.md) for:
- **Wave 2 (`75cd16bde97ddff79db9934db9cbc1c8`, commits `996650ea..2170b617` / PR #4937 `a6a4d7c4..42864fcc`):** Batched DFlash2 body (`STRATA_DF2_BATCHBODY`), Layer-36 context projection overlap, `Dflash2Drafter::copy_slot`, `BGEN` `n-1` tap/draft/suffix fixes, rank-major `--expert-cache-per-layer` slot lending, and `STRATA_BATCH_PREFILL_CHUNK=2048`.
- **Wave 3 (`d4f36a107ba0e9fb8aa09071bc86faf3`, commits `b0be13a4..df0775e8` / PR #4937 `8ce4a2cb..e6569011`):** Multi-slot `df2_kv_ring_batch[_f16]`, `df2_attention_batch[_f16]`, and `df2_pack_head_rows_batch` + `batch_body_exec_[n_items]` CUDA Graph capture (`emit` down to **`3.54–3.68 ms/win` at `c=2`** and **`5.30 ms/win` at `c=4`**), `M_e`-bucketed resident MoE plan ordering (`STRATA_MOE_BUCKET_PLAN`) + shared-expert `q8_ffn` reuse, mid-chunk layer-boundary prefill yielding (`STRATA_BATCH_PREFILL_LAYER_YIELD=16`, 9 yield parts across a `5,762`-token prefill sustaining **`66.45 tok/s` decode / `33.23 tok/s/slot`**), and fast `BGEN` admission (`6.7–7.0 ms` slot moves, `-41%`).
- **Wave 4 (`9f72ce1a82ee11eb15a454952cec76f3`, commits `2d826840..2ec832a9` / PR #4937 `080a91be..d9c8ddac`):** Direct `BGEN` slot admission on concurrent bursts (skipping solo `GEN -> STOP -> BGEN` churn) + turn checkpoint move on solo promotion, 16-row `read_windows` on one-card DFlash2 while keeping 8-row decode windows (`+35%` faster short-prompt admission), `>4`-entry resident MoE group splitting into parallel `blockIdx.y` chunks + first pre-stage barrier skip, zero-CPU-miss fast path in `expert_pool_dispatch_multi` & `copy_rows_from_mapped_kernel`, fused shared-expert `gate+up` MMVQ pair (`ROWS=2`) + router aux scalar gate across all 48 verify layers (`96` launches saved/window), and register-cached `Q8_1` activation fragments across `ROWS>1` in `native_mmvq_multi_kernel` (`c=2` cold **`127.75 tok/s`**, warm **`130.38–137.93 tok/s`** / **`68.96 tok/s/slot`**, window **`38.79 ms`**; `c=4` **`145.19 tok/s`** / **`36.30 tok/s/slot`**, individual slots up to **`39.23 tok/s`**).

---

## 4. Reproduction

```bash
# 1. Build strata + unit tests
systemd-run --user --scope -- cmake --build /home/soren/Strata-integ/build-verify \
  --target strata batch_mtp_depth_test batch_spec_slots_test batch_mtp_select_test dflash2_slots_test dflash2_tap_test -j32

# 2. Run CPU & GPU parity CTests (with strata.service stopped for GPU tests)
systemd-run --user --scope -- env PYTHONPATH=/home/soren/Strata-integ/third_party/llama.cpp/gguf-py \
  LD_LIBRARY_PATH=/opt/rocm/lib ctest --test-dir /home/soren/Strata-integ/build-verify --output-on-failure \
  -R '^(dflash2_slots_test|dflash2_tap_test|arch_defaults_test|qsa_select_bench|qsa_select_bench_fp32_tiled|native_mmvq_wide_parity|mmvq_il_parity|quantize_il_parity|iq_multi_parity|iq_multi_parity_kq_nov2k|iq_multi_parity_kq_swiglu|iq_multi_parity_kq_nosub16|quantize_act_parity|bf16_gemv_parity|batch_mtp_depth_test|batch_mtp_select_test|batch_spec_slots_test)$'

# 3. Run c=2 and c=4 A/B harness (with strata.service stopped)
systemd-run --user --scope -- /home/soren/Strata/.venv/bin/python \
  /home/soren/.hermes/cache/scratch/ab_single_card_batch.py A_prod_baseline B0_integ_opt_106k B2_integ_opt_40k_vocab B3_integ_opt_40k_defercpu
```
