# Verify window time reduction: research (2026-10-31)

Desk study (no GPU work). Sources: code at HEAD `82ab07c8`, measured docs (DECODE-TRACE, BATCH-WINDOW-SYNC, AB-FUSED-DECODE-E1), and this session's DF2 batch run.

## 1. What the numbers say (reconciled)

| metric | MTP batch (8-row) | DF2 batch (16-row, depth 3) | DF2 batch (8-row, depth 1, prod) |
|---|---|---|---|
| window ms | 40–43 | **90.8** | 82–84 (AB-FUSED-DECODE-E1) |
| run (GPU) | 35–38 | **79.8** | ~47 CUDA0 + ~20 CPU |
| commit | 2.2–2.3 | **0.02** | — |
| emit | 2.1–2.2 | **10.7** | — |
| adapt wait | 0.4 | **0.3** | — |
| GPU reach wait | — | **44.6 ms** (74% of run) | — |
| CPU experts | ~6.3 ms | **28.1 ms** (35% of run) | ~20 ms |
| routed / window | — | **12.13 avg rows** | 10.8 rows |
| routed entries | — | VRAM 83.2%, CPU 15.1%, PCIe 1.6% | — |
| per layer-window | — | CPU 15.9, VRAM 126, PCIe 2.2 ms | — |
| aggregate tok/s | 118 | **113.4** (28–31/slot) | 94.1 |

**Root cause of the 2× gap (MTP 40 ms vs DF2 90 ms):**
- DF2's `--batch-draft-window 16 --batch-draft-depth 3` → **12.13 rows/window** average vs MTP's fixed 8 rows.
- More rows → **larger expert union** (CPU experts 15.1% of routed vs ~lower in MTP).
- `verify.hpp:262-276`: union grows ~1.75× at 2 rows, 2.4× at 3, 3.05× at 4 (comment). DF2 carries 3.24 proposal rows/slot on average → unions approach 3×.
- **GPU reach wait 44.6 ms**: the GPU is busy computing the union; the host spins on doorbells.
- **CPU experts 28.1 ms**: the 15.1% of routed entries that miss VRAM go to CPU pool (pool workers 22 on 16 cores → contention).

Emit bucket is large (10.7 ms) because DFlash2 co-draft is **4 serialized slot rounds** (each: taps copy + graph launch + full sync), vs MTP's 2.2 ms for 4 serialized MTP co-drafts.

## 2. Code map of the verify window (what runs in the 90 ms)

`run_slot_rows` (`verify.cpp:4924`):
1. `stage_batch` → launches captured graph `exec_bm_[bkey]` (one per row layout, up to 8-row or 16-row).
2. **Per layer** (5 layers in this model): doorbell spin on `*seq < want` → CPU pool call `pool(user, h_x_, h_ids_, S, k, h_ymiss_, l)` → raise flag for next layer.
3. `cudaStreamSynchronize(cs_)` at the end of the layer loop (or immediately if `ar_on()` all-resident).
4. `sample_rows` (greedy: just read `h_out_`; sampled: draw on device).
5. `commit_slot_prefixes` → launch commit graph + optional sync.

The **door bell spin** (`verify.cpp:5029-5053`) is where `ms_wait` accumulates — the host waits for `*seq` to reach `want` per layer. `ms_pool` accumulates the CPU pool call time. For DF2's 12-row windows, the expert union is larger → more layers have CPU-served experts → more doorbell waits + more CPU pool time.

## 3. Ideas, ranked by expected value

### P1 — Reduce window breadth (the #1 lever, measured)
DF2's adaptive depth (`STRATA_BATCH_DF2_DEPTH`, BATCH plan Phase 2) already designed: give deeper chains only to slots with high acceptance EMA, and trim by draft probability. With 4 slots and window 16, a slot with EMA 0.75 gets depth 2, EMA 0.80 gets depth 3, new slots get 1. **Expected**: 12.13 → ~8 rows/window, bringing verify cost near MTP's 40 ms. **Gate: `STRATA_BATCH_DF2_DEPTH`**. Zero kernel changes, reuses `batch_mtp_depth.hpp`. **Do this first.**

### P2 — Cap the window at 8 rows for DF2 batch (immediate config)
`--batch-draft-window 8` forces the planner to fit at most 8 rows. The BATCH plan Phase 2 notes: "at N=1 it pins the serial path to base behaviour". Measured: DF2 at window 8, depth 1 was 82–84 ms (AB-FUSED-DECODE-E1) — still 2× MTP because of lower acceptance (52.8% vs 78%). **Fix acceptance first (P1), then cap breadth.**

### P3 — `STRATA_BATCH_EXPERT_AHEAD=N` (already coded, MTP opt-in)
`verify.cpp:498`: while waiting for layer l's doorbell, call `ahead_(ahead_user_, l)` which does `warm`/`prefetch` for layer l+N's routed experts. **Effect**: overlaps CPU expert load with GPU doorbell wait. Measured in MTP window 2: batch 6, **−1.8% concurrent** — small because CPU pool is the bottleneck, not prefetch latency. With DF2's 28 ms CPU experts, could be larger. **Gate exists, untested on DF2.**

### P4 — Fuse the verify graph (kernel-level)
The batch graph `exec_bm_` is captured per row layout (bkey = rows + S + base). For 8-row window: up to 256 layouts. Inside the graph: dense projections (GEMV), router, MoE combine, residual, attention, head. **Fusion targets**:
- `gr_down_q8_fast` ~51 launches/token (DECODE-TRACE) → fused variant in `perf/launch-reduction`.
- `wait_flag_ge` 26/token → doorbell spin, not a kernel.
- Quantize/interleave 171/token → fused into producer/consumer.
- **Gain**: dispatch count reduction (76% of idle in gaps <20 µs). But the verify graph runs **once per window** (12 rows), not per token. 12 rows × 1096 dispatches/token ≈ 13,000 dispatches/window. Fusion saves launch overhead, not arithmetic. **Est: 5–15% of 44.6 ms GPU wait = 2–7 ms.** Needs kernel work + parity tests.

### P5 — `STRATA_BATCH_DRAFT_GROUP` for DFlash2 (like MTP's group co-draft)
MTP's group co-draft saves 0.8 ms/window (2% of 40 ms) by running one head pass for all slots. DF2's emit is 10.7 ms (4 serialized co-draft rounds). Batching the DFlash2 co-draft (BATCH plan Phase 1) targets this — **already designed, highest draft leverage**.

### P6 — Increase expert cache hit rate (reduce CPU %)
DF2 run: 15.1% CPU, 1.6% PCIe. MTP (BATCH-WINDOW-SYNC follow-up 3): 11184 cached experts, 46.8 GiB experts, 21.3 GiB VRAM. DF2 run: 1098 MiB free VRAM (engine warned below 512 MiB). **VRAM pressure is real**. Options:
- `--vram-reserve-mib 832` (production default) vs lower reserve → trade KV pool for expert cache.
- `STRATA_DF2_RING_SLOTS=N` (BATCH plan Phase 4): shrink slot rings (40 MiB each × 4 = 160 MiB) → recover ~400 expert slots.
- **Ring quantization fp16→fp8**: halves ring VRAM (40→20 MiB/slot) → recover ~800 slots. **Parity-suspect** (changes float bits), gate it.

### P7 — `STRATA_ROUTE_TAIL_SKIP=7` (already default ON)
Skips experts ranked 7+ on every token of a window. Already on in production. Raising the threshold (e.g., skip rank 5+) cuts CPU experts but changes answers (opt-in only).

### P8 — `--pool-workers 15` (match physical cores)
Production warns: "22 CPU expert workers configured, but this CPU has 16 physical cores. Two workers on one core share its memory bandwidth and slow each other down." DF2's 28 ms CPU time includes pool contention. **Free config change: `--pool-workers 15` (or omit, engine picks).**

### P9 — Commit async + draft overlap (revisited)
BATCH-WINDOW-SYNC tried `STRATA_BATCH_COMMIT_ASYNC`: commit host time 2.2→0.02 ms, but emit grew to 3.5–4.3 ms (wait moved). **DF2's commit is already 0.02 ms** (commit graph is tiny, no sync). The wait is in the **draft rounds that depend on commit's KV writes**. Phase 1's event scheduling (wait for window event E, not commit) already addresses this.

### P10 — `STRATA_IQ_MT_MIN=1` (reduce multi-threaded expert combine threshold)
`verify.cpp` comment: CPU pool uses multi-threaded combine when `S >= STRATA_IQ_MT_MIN` (default 1). With 12-row windows, S≥12 always → multi-threaded. Single-threaded combine avoids thread sync overhead for small expert sets. **Unmeasured on gfx1201.**

## 4. Sequencing

| # | work | needs | effect (est) |
|---|---|---|---|
| 1 | **P1 adaptive depth for DF2** (`STRATA_BATCH_DF2_DEPTH`) | code GPU-free, build+measure | 12→8 rows, verify 90→~45 ms, **biggest single lever** |
| 2 | **P8 `--pool-workers 15`** | config only | −10–30% of 28 ms CPU = 3–8 ms |
| 3 | **P6 ring shrink / fp8** (`STRATA_DF2_RING_SLOTS` or fp8) | knob / kernel | recover 400–800 expert slots → −2–5% CPU |
| 4 | **P3 `STRATA_BATCH_EXPERT_AHEAD` on DF2** | test existing gate | overlap CPU load with GPU wait |
| 5 | **P4 verify graph fusion** | kernel work, parity | −2–7 ms GPU wait |
| 6 | **P5 DFlash2 batched co-draft** (BATCH plan Phase 1) | code GPU-free | emit 10.7→~2 ms (but Phase 1 already top draft priority) |

## 5. Parity and gates

- P1/P3/P8/P10: **no output change** (expert placement / scheduling only) — verify via engine decode lines + throughput, no parity probe needed.
- P6 (ring fp8): **parity-suspect** — changes drafter-internal bits; greedy answers identical (verify gates), sampled statistical.
- P4/P5: **bit-parity by construction** (same kernel sequence / same per-element math).

## 6. Corrections / open questions

- **DF2 accept at depth 3 is 73.6%** (this run) vs MTP 78%. P1's depth EMA + prob gate should push it higher. If not, acceptance gap remains a throughput drag.
- **GPU reach wait 44.6 ms** — how much is the graph's actual GPU time vs host spin? `hipEvent` timing inside `run_slot_rows` (reverted earlier) would separate them.
- **`ms_pool` 28.1 ms includes both CPU expert compute and pool contention** — `--pool-workers 15` isolates the contention component.
- **Emit 10.7 ms includes DFlash2 co-draft + output loop** — `STRATA_BATCH_DRAFT_TIMING` only prints for MTP; extend it to DF2 to isolate.

---

**Reproduce** (per `strata-speed-ab-testing`):
```sh
# P1 adaptive depth
~/Strata/.venv/bin/python tools/ab/batch_burst_bench.py $S/cand_df2_depth.json $S/bench.json --rounds 4 --max-tokens 256 --conc 4
# env: STRATA_BATCH_DF2_DEPTH=3 STRATA_BATCH_DF2_DEPTH_FLOOR=0.5

# P8 pool workers
# config arg: --pool-workers 15

# P6 ring shrink
# env: STRATA_DF2_RING_SLOTS=1024  (default 2064)
```

## 7. Measured P1 + P2 results on `gfx1201` (2026-10-31, build `69a14d5e`)

Implemented `STRATA_BATCH_DF2_DEPTH=N` + `STRATA_BATCH_DF2_DEPTH_FLOOR=F` (`include/strata/core/batch_spec_slots.hpp`, `src/program/generate.cpp`, unit tests in `tests/core/batch_spec_slots_test.cpp` and `tests/core/batch_mtp_depth_test.cpp`) and extended `STRATA_BATCH_DRAFT_TIMING=1` to DFlash2 batch co-drafting.

4×256 concurrent greedy coding requests (`parallel: 4`, `swift-1.5-iq3_s`):

| Metric | Baseline (`W=16, depth=3`, fixed) | Arm 1 (`W=16, depth=3, DF2_DEPTH=3`) | Arm 2 (`W=8, depth=1, DF2_DEPTH=3`) |
|---|---|---|---|
| Windows | 85 | 96 | **138** |
| Verified rows / window | ~14.2 | 12.42 | **7.89** |
| Committed tokens / window (`bt_rows`) | 12.13 | 10.61 | **7.38** |
| Offered proposals / slot-window | 3.00 | 2.36 (257 trimmed) | **1.01** (1166 trimmed) |
| Proposal acceptance | 707/960 (73.6%) | 664/837 (**79.3%**) | 478/548 (**87.2%**) |
| Accepted-depth histogram | — | `0:52, 1:91, 2:63, 3:149` | `0:72, 1:464, 2:1, 3:4` |
| **Total window ms** | **90.84 ms** | **82.14 ms** (−9.6%) | **53.49 ms** (**−41.1%**) |
| **`run` (verify pass)** | **79.78 ms** | **72.09 ms** (−9.6%) | **43.61 ms** (**−45.3%**) |
| — `CUDA0 GPU-reach wait` | 44.61 ms | 37.69 ms (−15.5%) | **26.39 ms** (**−40.8%**) |
| — `CPU experts` | 28.11 ms | 23.96 ms (−14.8%) | **14.31 ms** (**−49.1%**) |
| `commit` | 0.02 ms | 0.02 ms | 0.02 ms |
| `emit` (incl. co-draft) | 10.74 ms | 9.55 ms | 9.54 ms |
| — `strata batch draft` (`STRATA_BATCH_DRAFT_TIMING=1`) | — | **9.51 ms** (3.07 calls/win, 3.10 ms/call) | **9.49 ms** (3.19 calls/win, 2.97 ms/call) |
| Routed entries / window | 7268.6 | 5960.0 (−18.0%) | **3787.8** (**−47.9%**) |
| — VRAM hit % | 83.2% | 83.6% | **86.2%** |
| — CPU expert entries / window | 1100.2 (15.1%) | 882.8 (14.8%) | **479.4 (12.7%)** |
| — CPU experts / layer-window | 15.85 | 13.60 | **8.21** |
| **Engine batch decode (`rows/s`)** | ~123 rows/s | **129.2 rows/s** (+5.0%) | **138.0 rows/s** (**+12.2%**) |
| **HTTP aggregate `tok/s`** | 113.4 tok/s | 110.2 tok/s | **114.7 tok/s** (28.7–29.8/slot) |

### Key takeaways from the measurement

1. **`STRATA_BATCH_DF2_DEPTH=3` works as designed on both 16-row and 8-row windows**:
   - On `W=16, depth=3`, it trims 257 low-confidence / post-rejection proposals (`3.00 → 2.36` proposals/slot-window), raising acceptance from **73.6% → 79.3%**, cutting verify `run` time from **79.78 → 72.09 ms** (−7.7 ms) and window time from **90.84 → 82.14 ms**, and lifting engine batch decode speed to **129.2 rows/s**.
   - On `W=8, depth=1` (8-row batched DFlash2 with `STRATA_BATCH_DF2_DEPTH=3`), verified rows average **7.89 rows/window**, acceptance jumps to **87.2%** (`7.38` committed tokens per `7.89` verified rows = **93.5% row efficiency**), routed CPU expert misses drop by **56%** (`1100.2 → 479.4`), verify `run` time drops from **79.78 → 43.61 ms (−45.3%)**, total window drops from **90.84 → 53.49 ms (−41.1%)**, and engine batch decode reaches **138.0 rows/s** (**114.7 tok/s** HTTP wall).
2. **`STRATA_BATCH_DRAFT_TIMING=1` isolates DFlash2 co-draft inside `emit`**:
   - Of the `9.54 ms` `emit` bucket, **`9.49 ms` (99.5%) is serialized DFlash2 co-drafting** (`3.19` calls/window × `2.97 ms/call`).
   - Now that `W=8` brings verify `run` down to `43.61 ms`, serialized DFlash2 co-draft (`9.49 ms`) is **17.7% of the entire 53.49 ms window** — making Phase 1 (`BATCHHEAD`: batched head across slots + event overlap) an immediate win on top of `W=8`.
3. **Measured Phase 1 (`STRATA_DF2_BATCHHEAD=1`, build `a83e4a16`) on top of `STRATA_BATCH_DF2_DEPTH=3`**:
   - Implemented `Dflash2Drafter::draft_batch` + `capture_body` + `df2_pack_head_rows` + `df2_select_multi` + `df2_batch_slot_max_k`.
   - On `W=8, depth=1, STRATA_BATCH_DF2_DEPTH=3`, `STRATA_DF2_BATCHHEAD=1` cuts DFlash2 co-draft time from **`9.49 → 7.63 ms/window` (−19.6%, `2.97 → 2.41 ms/call`)**, cuts `emit` from **`9.54 → 7.66 ms/window`**, cuts total window time from **`53.49 → 50.41 ms/window` (−44.5% vs 90.84 ms baseline)**, and boosts engine batch decode throughput to **`145.4 rows/s` (+18.2% vs baseline)** and 4×256 HTTP wall throughput to **`120.1 tok/s` (30.0–31.4 tok/s per slot)**.
   - On `W=16, depth=3, STRATA_BATCH_DF2_DEPTH=3`, `STRATA_DF2_BATCHHEAD=1` cuts DFlash2 co-draft time from **`9.51 → 8.35 ms/window` (−12.2%)** and lifts engine batch decode from **`129.2 → 131.8 rows/s`**.