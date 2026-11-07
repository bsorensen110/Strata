# Batched DFlash2 Body, 16-Row Batch Verify, & `--batch 2` R9700 A/B Benchmark (2026-11-07)

## 1. Branch & Binary Provenance

- **Worktree**: `/home/soren/Strata-perf-dflash2-on-main-20261103`
- **Branch**: `feat/disk-conv-cache-df2` & `fork-main` pushed to `fork` (`bsorensen110/Strata`)
- **Commits**:
  - `9dc7e7132`: `perf(dflash2,verify): add batched DFlash2 body kernels, single-sync context, and 16-row batch verify`
  - `6456d44b2`: `perf(generate): wire 2-slot DFlash2 batch body, confidence-trimmed room check, and wide window gate`
- **Previous Production Engine (`fork-4383fcec3`)**:
  - Path: `/home/soren/.local/share/strata-upstream-v0.2.10/components/engine/fork-4383fcec3/strata`
  - SHA-256: `848f7880444ea49ef9681ca46082532145df7bc0ab5370e5e9214017fd3db702`
- **Promoted Production Engine (`fork-6456d44b2`)**:
  - Build path: `/home/soren/Strata-perf-dflash2-on-main-20261103/build-hip/strata`
  - Promoted path: `/home/soren/.local/share/strata-upstream-v0.2.10/components/engine/fork-6456d44b2/strata`
  - SHA-256: `cfb22e5b4f9e41df0d17f76787c6e8bf25dff043b03bfb2ea1dbdc7989498e08`
  - Rollback script: `/home/soren/strata-promote-backup-20261107-124500/rollback.sh`

## 2. Optimizations & Bug Fixes

1. **`STRATA_DF2_BATCHBODY=1` (`src/core/dflash2.cpp:1561`, `src/kernels/cuda/dflash2.cu`)**:
   - Batches the 8-layer DFlash2 transformer body across active slots (`n_items * kRows` columns in a single `df2_gemm` call per layer, plus multi-slot `df2_kv_ring_batch_f16` / `df2_attention_batch_f16`).
   - Verified bitwise identical to single-slot `Dflash2Drafter::draft()` (draft tokens, draft probabilities, and FP16 K/V rings) in `dflash2_ring_test` Test 6.
2. **Single-Sync `Dflash2Drafter::context()` with Per-Sub-Batch Metadata Slots (`src/core/dflash2.cpp:740-776`)**:
   - Indexes mapped host/device metadata by `s = o / kRows` (`h_meta_ + s * 32`, `m_meta_ + s * 32`) so multi-chunk `context()` updates queue back-to-back on `s_` with a single `cudaStreamSynchronize(s_)` at the end.
3. **`STRATA_DF2_TAP_OVERLAP=1` (`src/core/verify.cpp:695-720`)**:
   - Records `tap_ready_ev_` immediately after the final DFlash2 tap layer (`layer 36` of 48) in the verify graph so DFlash2 tap staging can overlap the remaining 11 target layers (`layers 37..47`).
4. **16-Row Multi-Slot Batch Verify Support (`STRATA_BATCH_DF2_MAX_T=8..16`)**:
   - Extended `kAttnRowsMax` from `8` to `16` (`include/strata/kernels/qsa_decode_attn.hpp:78`), added multi-row `lane_cell_attn()` support on `gfx12` (`src/kernels/cuda/qsa_decode_attn.cu`), and updated `GdnGroups` slot-run counting (`src/core/verify.cpp:585`).
5. **Three Bugs Found & Fixed**:
   - **`Verifier::ple_rows_b_` buffer overflow (`include/strata/core/verify.hpp:755`)**: `ple_rows_b_` was declared as `uint32_t ple_rows_b_[kVerifyMaxT * 4]` (`64` `uint32_t`s = 4 rows of `PLE_N_HEADS = 16`), whereas `Verifier::stage_batch` (`src/core/verify.cpp:4735`) writes `S * PLE_N_HEADS` (`up to 256` `uint32_t`s). Any batch window with `S >= 5` overflowed `Verifier` by up to 768 bytes into `ver_same.exec_bm_`. Fixed by sizing `ple_rows_b_[kVerifyLongT * PLE_N_HEADS]`.
   - **`native_quantize_q8_1_il` invalid argument for `nc > 4` (`src/core/dflash2.cpp:634, 1694`)**: Restricted `native_quantize_q8_1_il` to `2 <= nc <= 4` and used `native_quantize_q8_1` + `native_q8_1_interleave` for `5 <= nc <= 8`.
   - **False-positive host-deferred DFlash2 catch-ups in `--batch 2` (`src/program/generate.cpp:14657, 14886`)**: When `--draft-conf 0.5` trimmed a slot's draft count to `0` (`nrow[t] == 1`) due to `dprob[0] < 0.5`, line 14883 mistook `nrow[t] == 1` for "no spare window row", dropping the slot from async co-drafting (`31–47` times per 80-window run) and forcing a synchronous blocking `d.draft()` on the next window. Fixed via `df2_has_room[b]` (`want > 0` before confidence trimming), reducing deferred rows to `0`.

## 3. `STRATA_DF2_LOOKUP_SKIP=8` Mechanics & Tradeoff Analysis

- In `src/program/generate.cpp:19255-19368`, `defer_df2()` only activates when `--lookup` (`o.suffix_draft > 0`) is enabled and `sfx.last_match() >= 8`.
- When triggered, it defers `df2.draft()` at the end of round `r` and, if `DraftPolicy::choose()` selects the suffix lookup at the start of round `r + 1`, calls `df2.context()` (which still runs the 8-layer DFlash2 pass on the `a + 1` committed rows to keep the K/V ring contiguous, omitting only the 8 draft query rows and vocab head) and accepts `sbuf` **without** the normal `sbuf[0] == drafts[0]` agreement guard (`generate.cpp:19348`).
- On R9700 (`gfx1201`), omitting the draft query rows/head saves only ~0.35 ms while losing DFlash2's top-1 agreement filter; a single diverging suffix window costs ~25 ms. In live A/B testing (`16`-row window), `STRATA_DF2_LOOKUP_SKIP=8` reduced solo code decode from **89.7 tok/s** to **87.1 tok/s** (-2.9%) and is kept **off (`0`)**.

## 4. R9700 (`gfx1201`) Live A/B Benchmark Results

Model: `qwen3.8-flash-next-unsloth-ud-iq4_xs-v31` + `qwen3.8-flash-next-dflash2.q4_0-selq4.pack`, `--batch 2`, `--spec 8`, `--draft-conf 0.5`, `--resident-experts --resident-budget-gib 34`, `9936` GPU expert cache slots (`207/layer`).
Reproduction scripts: `/home/soren/.hermes/cache/scratch/bench_df2_batch2.py` and `/home/soren/.hermes/cache/scratch/bench_df2_final_ab.py`.

### 4.1 Ablation & Progression (256-token Greedy Code Solo & 2×256 Concurrent Batch)

- **Baseline (`MAX_T=8`, pre-`df2_has_room` fix, `BATCHBODY=0`, `TAP_OVERLAP=0`)**:
  - Solo greedy code: **85.5 tok/s**
  - `2×256` concurrent batch aggregate: **92.4 tok/s** (`48.3 tok/s` per slot; `32.5` deferred rows/run)
- **`MAX_T=8` + `df2_has_room` fix (`BATCHBODY=0`, `TAP_OVERLAP=0`)**:
  - Solo greedy code: **85.2 tok/s**
  - `2×256` concurrent batch aggregate: **106.8 tok/s** (`55.8 tok/s` per slot; `0` deferred rows, **+15.6%** vs baseline)
- **`MAX_T=8` + `df2_has_room` + `STRATA_DF2_BATCHBODY=1` + `STRATA_DF2_TAP_OVERLAP=1` (Promoted Winner)**:
  - Solo greedy code: **94.0 tok/s** (**+9.9%** vs baseline; `84.3 tok/s` on C++ refactor prompt)
  - `2×256` concurrent batch aggregate: **117.2 tok/s** (**`59.1 tok/s` per slot**, **+26.8%** vs `92.4 tok/s` baseline, **+9.7%** vs `106.8 tok/s` without `BATCHBODY`+`TAP_OVERLAP`)

### 4.2 Batch Verify Window Cap Sweep (`STRATA_BATCH_DF2_MAX_T`, with `BATCHBODY=1` & `TAP_OVERLAP=1`)

- **`STRATA_BATCH_DF2_MAX_T=8` (default, up to 3 drafts/slot = 8 rows/window)**:
  - Solo: **94.0 tok/s** | Refactor: **84.3 tok/s** | `2×256` Batch Aggregate: **117.2 tok/s** (`59.1 tok/s` per slot)
  - Telemetry: `avg 4.82 rows/window`, `37.1 ms/window`, `~58` unique routed experts/layer (`54.8` VRAM hits + `3.6` CPU experts)
- **`STRATA_BATCH_DF2_MAX_T=10` (up to 4 drafts/slot = 10 rows/window)**:
  - Solo: **93.1 tok/s** | Refactor: **82.2 tok/s** | `2×256` Batch Aggregate: **112.6 tok/s** (`57.5 tok/s` per slot)
- **`STRATA_BATCH_DF2_MAX_T=12` (up to 5 drafts/slot = 12 rows/window)**:
  - Solo: **91.7 tok/s** | Refactor: **82.5 tok/s** | `2×256` Batch Aggregate: **110.6 tok/s** (`57.2 tok/s` per slot)
- **`STRATA_BATCH_DF2_MAX_T=16` (up to 7 drafts/slot = 16 rows/window)**:
  - Solo: **89.7 tok/s** | `2×256` Batch Aggregate: **102.9 tok/s** (`52.7 tok/s` per slot)
  - Telemetry: `avg 5.54 rows/window` (+14.9% yield), `45.8 ms/window` (+23.5% time), `~75–85` unique routed experts/layer (`70.2` VRAM hits + `4.6–5.6` CPU experts)

## 5. Post-Promotion Verification (`strata.service`)

- Keyed `GET /v1/models` -> `200`; unkeyed -> `401`.
- Cold smoke completion (24 tok): **115.1 tok/s** engine decode (`drafts accepted 20 of 21` = 95.2%).
- Multi-turn conversation cache reuse check (`~1,000` prompt tokens):
  - Turn 1: `prompt 968 tokens = 0 reused + 968 read in 1994 ms (485.5 tok/s), 31 generated in 253 ms (122.7 tok/s), drafts accepted 25 of 27` -> `conversation cache: parked 998 tokens in 39.6 ms`.
  - Turn 2: `prompt 1032 tokens = 961 reused + 71 read in 438 ms (162.2 tok/s), 19 generated in 178 ms (106.8 tok/s), drafts accepted 15 of 16`.
