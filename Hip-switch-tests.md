# HIP DFlash2 Switch & A/B Tests

Quick env-var A/B test matrix for the HIP machine, covering the 6 DFlash2 optimizations, the `STRATA_DF2_HEAD=native` default, and raising the batch window cap (`STRATA_BATCH_DF2_MAX_T`) with the tuned settings.

All examples use Linux/bash syntax (`ENV=VAL strata ...`). On Windows PowerShell, set `$env:VAR="VAL"` for the session (or `cmd /c "set VAR=VAL && ..."`). Every variable below takes effect per process launch.

---

## 1. New Defaults Active Automatically (Baseline)

With no env vars set, HIP builds now run with:

| Variable | New Default | Meaning | Opt-out / Revert |
|---|---|---|---|
| `STRATA_DF2_HEAD` | `native` (unset/empty) | Native drafter head (`df2_gemm`) instead of pruned copy | `STRATA_DF2_HEAD=pruned` (or `0`, `off`, `false`) |
| `STRATA_DF2_BEAM` | `4` | Order-1 Viterbi log-probability beam search (`select_slot_walk`) | `STRATA_DF2_BEAM=1` (greedy, old behavior) |
| `STRATA_DF2_ATTN_F16_SMEM` | `1` (HIP) / `0` (CUDA) | FP16 attention score SMEM + parallel `4x2` `Q K^T` tile | `STRATA_DF2_ATTN_F16_SMEM=0` (FP32 SMEM) |
| `STRATA_DF2_FP16_RING` | `1` (`true`) | FP16 K/V ring buffers in DFlash2 attention | `STRATA_DF2_FP16_RING=0` (FP32 ring) |
| `STRATA_DF2_GEMM_32` | `1` | Single-pass 17..32 col drafter GEMM (`33..64` in steps of 32) | `STRATA_DF2_GEMM_32=0` (split by 16 cols) |
| `STRATA_DF2_HEAD_IL` | `1` (HIP, gated on IL-supported head quants) | Interleaved `native_mmvq` head path | `STRATA_DF2_HEAD_IL=0` |
| `STRATA_DF2_HEAD_IL_ROWS` | `4` (override `1` / `2` / `4`) | IL rows per warp on 5..8 col slices | `STRATA_DF2_HEAD_IL_ROWS=1` or `2` |
| `STRATA_DF2_HEAD_CHUNK` | `8` (HIP) / `14` (CUDA) | Max columns per head chunk (`1..14`; chunk-0 Q8_1/IL is fused into `pack_head_rows`) | `STRATA_DF2_HEAD_CHUNK=14` |
| `STRATA_DF2_FORK` | `0` (HIP) / `1` (CUDA) | Single stream on HIP (avoids fork/join sync overhead) | `STRATA_DF2_FORK=1` |
| `STRATA_DF2_PRIO` | `0` (HIP) / `1` (CUDA) | Normal stream priority on HIP | `STRATA_DF2_PRIO=1` |
| `STRATA_DF2_SOLOPIPE` | `1` | Overlap single-slot drafting with next step | `STRATA_DF2_SOLOPIPE=0` |
| `STRATA_DF2_LOOKUP_SKIP` | `0` (HIP/CUDA) / `8` (Metal) | Min lookup suffix match length to skip DFlash2 round | `STRATA_DF2_LOOKUP_SKIP=8` (or `6`, `10`) |
| `STRATA_BATCH_DF2_MAX_T` | `8` (`kVerifyNarrowT`) | Max tokens per batch verify window (`8..16` when built with `STRATA_WINDOW_MAX_T=16`) | `STRATA_BATCH_DF2_MAX_T=16` (or `12`) |
| `STRATA_BATCH_DF2_CONF` | `--draft-conf` else `0.35` | Batch confidence chain cutoff (`0` = off, always keep up to cap) | `STRATA_BATCH_DF2_CONF=0` / `0.25` / `0.45` |
| `STRATA_BATCH_DF2_MAXX` | `99` (clamped by `kDrafts`) | Max draft rows per slot per batch window | e.g. `STRATA_BATCH_DF2_MAXX=6` |
| `STRATA_BATCH_DF2_DEFER` | `1` | Skip drafting for slots without spare rows | `STRATA_BATCH_DF2_DEFER=0` |

---

## 2. Quick Env-Var A/B Matrix (Run Order)

Compare **tok/s**, **draft acceptance rate / mean accepted tokens per step**, and **draft ms per round** across the runs below using the same model, prompt set, `--dflash2` pack, and `-n` / concurrency settings.

### A. Standalone Kernel Sanity & Microbench (Fast First Check)

```sh
# 1. Parity check (beam=4 default vs greedy beam=1)
./dflash2_q4_parity
STRATA_DF2_BEAM=1 ./dflash2_q4_parity

# 2. Attention microbench: FP16 SMEM (HIP default 1) vs FP32 SMEM (0)
./dflash2_attn_bench
STRATA_DF2_ATTN_F16_SMEM=0 ./dflash2_attn_bench
```

### B. Single-Stream (Concurrency 1) A/B Matrix

Pick your standard single-stream `--dflash2` benchmark command as `<SOLO_CMD>`.

| Run | Purpose | Command Prefix |
|---|---|---|
| **S0 (New Default)** | All 6 optimizations + `native` head + `beam=4` | `<SOLO_CMD>` |
| **S1 (Pruned Head)** | Measure acceptance & speed diff of `native` vs `pruned` head (requires vocab map file in pack) | `STRATA_DF2_HEAD=pruned <SOLO_CMD>` |
| **S2 (Greedy Select)** | Measure acceptance gain from `beam=4` Viterbi vs `beam=1` greedy | `STRATA_DF2_BEAM=1 <SOLO_CMD>` |
| **S3 (FP32 Attn SMEM)** | Isolate `STRATA_DF2_ATTN_F16_SMEM=1` vs `0` on Strix Halo / RDNA | `STRATA_DF2_ATTN_F16_SMEM=0 <SOLO_CMD>` |
| **S4 (GEMM 32-col Off)** | Isolate single-pass 17..32 col GEMM (`df2_gemm`) | `STRATA_DF2_GEMM_32=0 <SOLO_CMD>` |
| **S5 (Head IL Off / Chunk 14)** | Isolate IL head + chunk size (`8` vs `14`) | `STRATA_DF2_HEAD_IL=0 <SOLO_CMD>` |
| **S6 (Head IL Rows Sweep)** | Tune warp rows in IL head (`4` vs `2` vs `1`) | `STRATA_DF2_HEAD_IL_ROWS=2 <SOLO_CMD>` (and `=1`) |
| **S7 (Head Chunk 14)** | Test full-window 14-col head launch vs 8-col chunk | `STRATA_DF2_HEAD_CHUNK=14 <SOLO_CMD>` |
| **S8 (Fork / Lookup Skip)** | Test dual-stream fork or skipping DFlash2 on long suffix matches | `STRATA_DF2_FORK=1 <SOLO_CMD>` / `STRATA_DF2_LOOKUP_SKIP=8 <SOLO_CMD>` |

### C. Batch Window Cap & Tuned Batch Matrix (Target Test: Raising `STRATA_BATCH_DF2_MAX_T`)

Pick a steady multi-slot batch benchmark (e.g., `--batch 4` and `--batch 8` with `--dflash2`) as `<BATCH_CMD>`. Up to 8 batch slots are now drafted and cached in a single batched drafter pass (`kBatchMaxSlots = 8`, LRU graph cache = 64).

| Run | Purpose | Command Prefix |
|---|---|---|
| **B0 (Tuned Default, T=8)** | New defaults (`native`, `beam=4`, up to 8-slot batch draft, `MAX_T=8`) | `<BATCH_CMD>` |
| **B1 (Raise Window Cap -> 12)** | Test raising batch window cap with new defaults | `STRATA_BATCH_DF2_MAX_T=12 <BATCH_CMD>` |
| **B2 (Raise Window Cap -> 16)** | Full 16-row batch window with new defaults (`--batch 4` & `--batch 8`) | `STRATA_BATCH_DF2_MAX_T=16 <BATCH_CMD>` |
| **B3 (Old Greedy + T=16)** | Check whether `beam=4` + `native` unlocked `MAX_T=16` vs greedy (`beam=1`) | `STRATA_BATCH_DF2_MAX_T=16 STRATA_DF2_BEAM=1 <BATCH_CMD>` |
| **B4 (Pruned Head + T=16)** | Check `pruned` vs `native` under `MAX_T=16` | `STRATA_BATCH_DF2_MAX_T=16 STRATA_DF2_HEAD=pruned <BATCH_CMD>` |
| **B5 (Confidence Threshold Sweep)** | Tune batch stop threshold under `MAX_T=16` (lower keeps deeper drafts) | `STRATA_BATCH_DF2_MAX_T=16 STRATA_BATCH_DF2_CONF=0.20 <BATCH_CMD>` |
| **B6 (Confidence Off + T=16)** | Unconditional full window draft under `MAX_T=16` | `STRATA_BATCH_DF2_MAX_T=16 STRATA_BATCH_DF2_CONF=0 <BATCH_CMD>` |
| **B7 (GEMM 32-col Off @ Batch 8)** | Confirm win of single-pass `17..32` / chunk-32 GEMM in 8-slot batch | `STRATA_BATCH_DF2_MAX_T=16 STRATA_DF2_GEMM_32=0 <BATCH_CMD>` |
| **B8 (Defer Off)** | Measure impact of `STRATA_BATCH_DF2_DEFER=1` vs `0` | `STRATA_BATCH_DF2_MAX_T=16 STRATA_BATCH_DF2_DEFER=0 <BATCH_CMD>` |

> [!TIP]
> Check `STRATA_BATCH_DF2_MAX_T=12` and `16` at both `--batch 4` (25..32 drafter cols -> tests single-pass 32-col GEMM) and `--batch 8` (33..64 drafter cols -> tests multi-slot batching across 8 slots + async precomp log flush in `draft_batch_slots_finish`).

---

## 3. What to Record per Run

1. **System decode throughput**: aggregate `tok/s` (and per-user `tok/s` in `--batch 4` / `--batch 8`).
2. **Draft acceptance**: mean accepted tokens per verify step and `%` accepted.
3. **Batch window shape** (from stderr timing line when batch finishes): `rows/window` and `tokens/window`.
4. **Regressions check**: ensure `STRATA_DF2_HEAD=native` and `STRATA_DF2_BEAM=4` keep greedy output deterministic across repeats, and watch VRAM headroom at `MAX_T=16` (`head` logits buffer scales from 7 to 14 vocab vectors when batch DFlash2 is enabled).
