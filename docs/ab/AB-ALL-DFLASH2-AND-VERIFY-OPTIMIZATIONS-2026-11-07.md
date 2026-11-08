# DFlash2 & Target Verifier Optimization Campaign (2026-11-07)

- **Worktree**: `/home/soren/Strata-perf-dflash2-on-main-20261103`
- **Branch**: `perf/dflash2-tap-overlap-16row` (`0247ac749`, 31 code commits + docs on top of `fork-main` @ `d80835e40`)
- **Hardware**: AMD Radeon AI PRO R9700 (`gfx1201` / RDNA4, 32 GiB VRAM, ROCm nightly `10.2.0a20260914`)
- **Model**: `qwen3.8-flash-next-unsloth-ud-iq4_xs-v31` + `qwen3.8-flash-next-dflash2.q4_0-selq4.pack` (`--batch 2`, `--spec 8`, `--draft-conf 0.5`, `--resident-experts --resident-budget-gib 34`)

## 1. Atomic Commit Stack (`d80835e40..0247ac749`)

1. `b44d8bc61` `perf(dflash2): support 16-way grouped GEMM dispatch`
2. `62b915718` `perf(dflash2): batch all-layer K/V projections with strided scratch`
3. `4ac0f3992` `perf(verify): extend batched fast paths through 16 rows`
4. `e06154af0` `perf(cpu): parallelize multi-token expert activation quantization`
5. `0cc729d8d` `perf(verify): fuse QSA sigmoid gate and Q8_1 quantization`
6. `a1573b9cd` `perf(verify): fuse grouped GDN output quantization`
7. `b8151728d` `perf(dflash2): precompute batch context and capture full draft graph`
8. `813d0b7ad` `perf(dflash2): overlap precomputed context with target verification`
9. `15255e495` `perf(batch): rank DFlash2 spare verify rows across slots by chain dprob`
10. `a75f82205` `perf(dflash2): incremental warp reduction and 8-per-thread topk_part`
11. `f5dbf9d3d` `perf(dflash2): default interleaved MMVQ and 8-row chunking for HIP draft head`
12. `8086e282b` `perf(dflash2): mapped-sentinel completion spin and deferred ring-log D2H`
13. `a7d10ef0c` `perf(batch): defer Verifier::wait_commit across window bookkeeping`
14. `74f86ef7e` `perf(dflash2): pack active head rows before candidate_selector projection`
15. `11e9c9a12` `perf(dflash2): warp-parallelize selector softmax in select_slot_walk`
16. `69f6951be` `perf(dflash2): elide empty-chunk part_ writes and reads in attention merge`
17. `012759143` `perf(dflash2): specialize exact-column counts in gemm4_rows and gemm_rows kernels`
18. `b0797e8d1` `perf(batch): overlap DFlash2 finish with adapt_run and support STRATA_ADAPT_OVERLAP in step_batch`
19. `84230cd2f` `perf(mmvq): extend fused native_quantize_q8_1_il to 2..8 columns`
20. `ece65d1a4` `perf(verify): fuse interleaved Q8_1 store into native_qsa_gate_quantize_q8_1`
21. `d1c3d7fb3` `perf(verify): fuse step+pos upload and batch GDN/QSA layer commits in capture_commit_batch`
22. `3c075b37f` `perf(batch): enable mid-chunk prefill layer yield for batch_df2 and shrink lend() chunk when slots decode`
23. `ac3b744cd` `perf(dflash2): widen prompt context and log rebuild from 8-row to 16-row GEMM passes and batch D2H log copies`
24. `db4964732` `perf(batch): overlap DFlash2 context with verify in read_windows and yield to active batch slots`
25. `d3711bb3f` `perf(batch): enable precomputed context and captured batch_full graph when 1 slot is active in step_batch`
26. `0d468cf6b` `perf(batch): batch DFlash2 ring copy into direct_slot_copy sync and overlap BADM initial draft with checkpoint setup`
27. `6b09a8335` `perf(cpu): enable parallel activation quantization across cold modes and fuse intermediate H quantization into gate/up phase`
28. `a3aee7b03` `perf(expert-source): move RouterLookahead submit past GPU plan publish, fast-dedupe routed experts, and skip lookahead/claim_todo when all layer experts are in VRAM/RAM`
29. `7e5c0e10c` `perf(dflash2): stage precomputed batch context log copies in precompute_batch_context, flush on CPU during batch_full launch, and elide redundant finish stream syncs`
30. `b8b65f534` `perf(verify): overlap PLE gather with Layer 0, add adapt_gate_wait, and batch penalty staging and multi-row sample_tokens in run_slot_rows`
31. `0247ac749` `perf(snapshot): pre-populate CheckpointPool on background CPU thread, move dropped slot checkpoints in park_slot/park_current, and batch snapshot DMA syncs`

## 2. Microbenchmark & Parity Verification (`gfx1201`, `strata.service` stopped)

- **`dflash2_ring_test` (`DFLASH2_RING_FULL=1`)**:
  - Test 6 (`batch_body` 2-slot): tokens, probs, and K/V rings match single-slot `draft()` **bitwise**.
  - Test 7 (`precompute_batch_context + batch_full_exec_` 16-row): tokens, probs, and K/V rings match single-slot `draft()` **bitwise**.
  - Timing improvements (Test 7):
    - `precomp (16 rows)`: **289.0 µs -> 212.79 µs (−26.4%)**
    - `batch_full_exec_`: **3540.0 µs -> 2052.61 µs (−42.0%)**
    - `batch_body_exec_ + head`: **2221.32 µs**
- **`dflash2_small_bench`**:
  - `topk (7 x 106299)`: **50.50 µs -> 37.63 µs (−25.5%)**
  - `conv_rms_quant (8 x 2560)`: **6.40 µs -> 5.91 µs (−7.7%)**
  - `select (7 rows)`: **13.71 µs**
  - `select_multi (4 x 3 rows)`: **7.73 µs**
  - `pack_head_rows (3 rows)`: **2.98 µs**
- **Parity tests (`19/19` CTest + standalone binaries)**:
  - `quantize_il_parity` (`ncols = 2..8`): **0 differing bytes**
  - `mmvq_il_parity` (`T = 1..8`, rows `1/2/4` + table): **bitwise**
  - `dflash2_gemm_bench` (all matrices, all variants): **bitwise**
  - `gdn_parity`: **0 failures**
  - `qsa_parity`: **0 failures** (`append_steps_layers` & `gate_quantize_q8_1` bitwise)
  - `gr_parity`: **0 failures**
  - `verify_parity`: **PASS**
  - `verify_batch_parity`: **PASS** (all kernels bitwise equal)

## 3. End-to-End A/B Results (`/home/soren/.hermes/cache/scratch/ab_all_opts_20261107/results.json`)

- **Arm 0 (`0_prod_8row`, control `cfb22e5b4`)**:
  - Solo greedy code (`2x256`): mean **86.2 tok/s** wall (`80.1` cold / `92.4` warm; engine decode **87.8** cold / **102.0 tok/s** warm)
  - Solo refactor (`2x192`): mean **85.1 tok/s** wall (`75.3` cold / `94.9` warm; engine decode **114.4** cold / **153.6 tok/s** warm), hash `ece1433a582a4c38`
  - Concurrent `2x256` (`batch2`): mean **122.7 tok/s** aggregate wall (`118.4` R1 / `127.1` R2; per-slot **62.6 tok/s**; engine **127.3 / 148.2 rows/s**; `emit 3.88 / 4.16 ms/window`, `adapt wait 4.14 / 3.91 ms/window`)
- **Arm 1 (`1_cand_8row_all`, candidate `d1c3d7fb3` without adapt overlap)**:
  - Solo greedy code (`2x256`): mean **85.5 tok/s** wall (`80.3` cold / `90.8` warm; engine decode **88.9** cold / **100.1 tok/s** warm) — note: auto expert cache allocated `9888` slots (`206`/layer) vs `9936` slots (`207`/layer) on Arms 0 & 2 due to 30 MiB free-VRAM delta at startup
  - Solo refactor (`2x192`): mean **82.1 tok/s** wall, hash `ece1433a582a4c38` (exact match with control)
  - Concurrent `2x256` (`batch2`): mean **125.6 tok/s** aggregate wall (**+2.4% vs control**, `121.0` R1 / `130.2` R2; per-slot **63.5 tok/s**; engine **130.2 / 153.2 rows/s**; `emit 3.53 / 2.74 ms/window` [**−34.1% warm co-draft time**], acceptance **348/162 = 214.8%** vs `196.5%` control)
- **Arm 2 (`2_cand_8row_adaptoverlap`, candidate `d1c3d7fb3` + `STRATA_ADAPT_OVERLAP=1 STRATA_ADAPT_OVERLAP_RESIDENT=1`, matched `9936` expert cache slots)**:
  - Solo greedy code (`2x256`): mean **90.7 tok/s** wall (**+5.2% vs control**, `83.5` cold / **97.8 tok/s** warm; engine decode **91.8** cold / **108.7 tok/s** warm [**+6.6% warm engine decode**])
  - Solo refactor (`2x192`): mean **83.1 tok/s** wall (`74.5` cold / `91.7` warm; engine decode **115.5** cold / **148.2 tok/s** warm), hash `ece1433a582a4c38` (exact match with control)
  - Concurrent `2x256` (`batch2`): mean **126.0 tok/s** aggregate wall (**+2.7% vs control**, `122.9` R1 / `129.2` R2; per-slot **64.1 tok/s**; engine **133.3 / 151.3 rows/s**; `emit 2.78 / 2.76 ms/window` [**−33.7% warm co-draft time**], `adapt wait 1.06 / 1.07 ms/window` [**−72.6% adapt wait**])

## 4. Documented Negative / Guarded Findings

- **Shrinking `conv_rms_quant_kernel` from 1024 to 256 threads regressed `dflash2_small_bench` (`6.40 µs -> 12.13 µs`)**:
  - Because `conv_rms_quant_kernel` launches only `B = 8` blocks across the GPU (`64` WGPs on `gfx1201`), reducing threads per block from `1024` (`32` warps, `float v[3]`, 3 loop rounds) to `256` (`8` warps, `float v[4][3]`, 12 serial loop rounds) serialized global memory latency and increased per-thread VGPR usage. Restoring `1024` threads brought `conv_rms_quant` to **5.91 µs**.
- **Fusing `x_il` stores into `FusedGrArgs` (`fused_gr_read_multi`) and `gdn_step_norm_multi_kernel` increased verify window time by ~1.5 ms**:
  - Adding `void* q8_il` to `FusedGrArgs` expanded `GrMulti` by 128 bytes across every `gr_down` kernel launch, and adding `bm/pm/dl/cp` to `gdn_step_norm_multi_kernel` (which holds `float s[32]` recurrence state in VGPRs across 512 threads) increased VGPR pressure inside the recurrence loop. By contrast, fusing `x_il` into `native_qsa_gate_quantize_q8_1` (`ece65d1a4`, a 16-VGPR elementwise kernel) cleanly eliminated the 256 KB `attn32_` round-trip and second kernel launch when `use_quant_il_wo` is active for `2..8` rows.

## 5. Single-Card Concurrent Prefill + Decode A/B Results (`/home/soren/.hermes/cache/scratch/ab_concurrent_pf_dec_20261108/results.json`)

- **`dflash2_ring_test` Context & Stash Rebuild (`ac3b744cd`)**:
  - Test 1 (`rewind`, 578 rows rebuilt): **4.1 ms** (down from ~8.5 ms with 8-row passes; bitwise exact)
  - Test 2 (`parked`, 2042 rows rebuilt): **13.0 ms** (down from ~26 ms with 8-row passes; bitwise exact)
- **Paired End-to-End Concurrent Prefill + Decode Benchmark (`0_prod_control` @ `cfb22e5b4` vs `1_cand_concurrent` @ `0d468cf6b`)**:
  - **Solo Refactor Parity (`2x192`, `temp=0`)**:
    - Control: **74.1 tok/s** (`60.5` cold / `87.6` warm), hash `ece1433a582a4c38`
    - Candidate: **74.5 tok/s** (`63.3` cold / `85.7` warm), hash `ece1433a582a4c38` (exact match)
  - **Concurrent Decode (Slot 0: 256 tok) + Long Prefill & Decode (Slot 1: 4,712 prompt tok + 128 generated tok)**:
    - Control (`0_prod_control`):
      - Combined aggregate throughput: **46.5 tok/s** (`45.4` cold R1 / `47.7` warm R2; wall `8.46s` / `8.05s`)
      - Slot 0 decode throughput during co-prefill: **32.6 tok/s** (`31.1` R1 / `34.1` R2)
      - Slot 1 total latency: **7.90s** (`8.11s` R1 / `7.70s` R2); engine 4,712-token prompt read: `6,146 ms (766.7 tok/s)` R1 / `5,886 ms (800.6 tok/s)` R2
      - Slot 1 `copy_to_slot` admission latency: `57.7 ms`
      - Window telemetry (warm R2): `34.11 ms/window` (`emit 3.53 ms`, `adapt wait 1.45 ms`), VRAM hits `84.6%`, CPU `14.6%`, `48.2 rows/s`, file-tier reads `10,845` blobs
    - Candidate (`1_cand_concurrent`):
      - Combined aggregate throughput: **48.2 tok/s** (**+3.7% mean / +5.8% warm**, `46.0` cold R1 / **50.5 tok/s** warm R2; wall `8.35s` / **7.61s**, **−0.44s faster warm**)
      - Slot 0 decode throughput during co-prefill: **33.5 tok/s** (`32.1` R1 / **34.9 tok/s** warm R2)
      - Slot 1 total latency: **7.63s** (`8.00s` R1 / **7.25s** warm R2, **−0.45s faster warm**); engine 4,712-token prompt read: **`5,843 ms (806.4 tok/s)` R1 / `5,606 ms (840.5 tok/s)` R2 (−292 ms avg prompt read)**
      - Slot 1 `copy_to_slot` admission latency: **`5.8 ms` (−90.0%)**
      - Window telemetry (warm R2): **`32.21 ms/window`** (`emit 2.40 ms` [**−32.0%**], `adapt wait 0.45 ms` [**−69.0%**]), VRAM hits **`87.1%`**, CPU **`12.3%`**, **`52.2 rows/s` (+8.3%)**, file-tier reads **`6,849` blobs (−36.8% due to `lend()` chunk shrink)**
  - **2-Slot Concurrent `2x256` Burst (`batch2`)**:
    - Control (`0_prod_control`): mean **125.8 tok/s** aggregate (`125.7` R1 / `125.9` R2; per-slot **63.7 tok/s**; engine **136.4 / 136.3 rows/s**; `emit 3.81 / 3.67 ms`, `adapt wait 3.69 / 3.97 ms`)
    - Candidate (`1_cand_concurrent`): mean **128.9 tok/s** aggregate (**+2.5% mean / +4.4% warm**, `126.5` R1 / **131.4 tok/s** warm R2; per-slot **65.3 tok/s** [`64.0` R1 / **66.6 tok/s** R2]; engine **148.3 / 154.1 rows/s [+13.1% warm engine rows/s]**; `emit 2.79 / 2.76 ms [−25.6%]`, `adapt wait 1.08 / 1.08 ms [−72.8%]`, acceptance **346/164 = 211.0%**)

## 6. CPU-Side Concurrent Batched Processing Optimizations (`6b09a8335..0247ac749`)

- **Mechanisms**:
  1. **Parallel Multi-Token Activation Quantization (`defer_act_quant`) & Fused Intermediate `H` Quantization in `ExpertPool` (`6b09a8335`)**:
     - Raised `kMaxActQuantToks` from `32` to `1024` and removed the `d.cold_mode == 0` restriction in `expert_pool_dispatch_multi` (`src/core/expert_source.cpp`) so Linux doorbell/mapped mode (`d.cold_mode == 4`) and `PREFILL_CPU_SHARE` (`src/prefill/prefill.cpp`) quantize multi-token activations in parallel across `ExpertPool` worker threads during Phase 1 (`mode 5` / `mode 3`) instead of serially on the calling thread.
     - Added per-expert `gu_rows_done` atomic row tracking in `SplitBufMulti`: the worker thread completing the final gate+up row slice of expert `e` immediately quantizes `sb.ff[t]` into `sb.hq[t]` / `sb.a2[t]` while `sb.ff[t]` is hot in L1/L2 cache, eliminating Phase 2 (`mode 6` / `mode 4`) startup contention.
  2. **RouterLookahead Off-Critical-Path Submission, Resident-Layer Fast-Skip, and $O(1)$ Bitmap Deduplication (`a3aee7b03`)**:
     - Moved `RouterLookahead::submit` after `gpu_plan.publish(...)` in `expert_source.cpp` so CPU lookahead router GEMV/top-K runs while the GPU is already executing the current layer's MoE kernel.
     - Added `ExpertSource::has_resident_blob(layer, expert)` and skipped `RouterLookahead::submit` and `FileExpertSource::claim_todo` when all routed cold experts in a layer are already resident in pinned RAM (`complement_ready_`), avoiding pointless background file-pread queueing and mutex contention.
     - Replaced $O(N^2)$ linear-scan expert deduplication in `expert_pool_dispatch_multi` with direct 512-entry lookup tables (`seen_i0[512]`, `miss_seen[512]`).
  3. **CPU-Overlapped DFlash2 KV Ring Log Maintenance & Duplicate Sync Elision (`7e5c0e10c`)**:
     - Staged `pos_d_` and `slot_d_` D2H copies into `h_logq_` / `h_logd_` inside `precompute_batch_context` before recording `bctx_ready_[s]`, then launched `batch_full_exec_` immediately in `draft_batch_slots_launch` and flushed the staged ring logs into CPU `log_pos_` / `log_slot_` while the GPU executes the draft graph.
     - Elided the duplicate `cudaStreamSynchronize(items[0].drafter->cs_)` in `draft_batch_slots_finish` when `n_items >= 2` (`cs_` is already `items[0].drafter->cs_`).
  4. **Verifier CPU/GPU Overlap (`ple_done_ev_`, `adapt_gate_wait`, and Batched Multi-Row `sample_tokens`) (`b8b65f534`)**:
     - Replaced blocking `cudaStreamSynchronize(ple_stream_)` before Layer 0 in `Verifier::run_slot_rows` with `cudaEventRecord(ple_done_ev_, ple_stream_)` + `cudaStreamWaitEvent(cs_, ple_done_ev_, 0)` so CPU Layer-0 setup and GPU Layer-0 execution overlap with PLE gather DMA.
     - Added guarded `adapt_gate_wait` (`adapt_gate_->active && adapt_gate_->has[l] != 0`) in `Verifier::run_slot_rows`.
     - Batched penalty staging and contiguous multi-row `cuda::sample_tokens` calls in `Verifier::run_slot_rows` across rows sharing sampling parameters without per-token grammar state transitions.
  5. **Background CPU `CheckpointPool` Warmup, Zero-Copy `park_slot`/`park_current` Move, and Batched Snapshot DMA Syncs (`0247ac749`)**:
     - Defaulted `STRATA_CKPT_POOL_READY=2` on HIP so `CheckpointPool` pre-allocates and pre-faults two 118 MB `ConvCheckpoint` structs on a background `std::async` CPU thread at startup and after each checkpoint consumption.
     - Enabled `park_drops` move semantics in `park_slot` and `park_current` when `STRATA_SLOT_CKPT_MOVE=1` (default on) so dropped slot checkpoints move into `ConversationSnapshot` in $O(1)$ without 118 MB host-vector reallocation/copy on the engine thread.
     - Queued running-state and KV-snapshot DMA copies (`queued = true`) on `private_stream` in `conversation_state.cpp` and `conversation_snapshot.cpp` with a single `sync_private_stream()` barrier at completion.
- **Paired End-to-End A/B Results (`0_prod_control` @ `cfb22e5b4` vs `1_cand_concurrent` @ `0247ac749`)**:
  - **Solo Refactor Parity (`2x192`, `temp=0`)**:
    - Control (`0_prod_control`): **74.4 tok/s** wall (`60.8` cold / `88.0` warm; engine decode `85.6` cold / **136.0 tok/s** warm), hash `ece1433a582a4c38`
    - Candidate (`1_cand_concurrent` @ `0247ac749`): **78.6 tok/s** wall (**+5.6% vs control**, `64.4` cold / **92.8 tok/s** warm; engine decode `90.4` cold / **144.8 tok/s** warm [**+6.5% warm engine decode**]), hash `ece1433a582a4c38` (exact bitwise match)
  - **Concurrent Decode (Slot 0: 256 tok) + Long Prefill & Decode (Slot 1: 4,712 prompt tok + 128 generated tok)**:
    - Control (`0_prod_control`):
      - Combined aggregate throughput: **46.3 tok/s** (`44.6` R1 / `48.1` R2; wall `8.62s` / `7.98s`)
      - Slot 0 decode throughput during co-prefill: **32.8 tok/s** (`31.2` R1 / `34.4` R2; wall `8.20s` / `7.43s`)
      - Slot 1 total latency: **7.95s** (`8.27s` R1 / `7.63s` R2); engine 4,712-token prompt read: `6,210 ms (758.8 tok/s)` R1 / `5,915 ms (796.6 tok/s)` R2
      - Window telemetry: `33.81 ms/window = run 29.03 + commit 0.01 + emit 3.56 + adapt wait 1.20`
    - Candidate (`1_cand_concurrent` @ `0247ac749`):
      - Combined aggregate throughput: **48.8 tok/s** (**+5.3% vs control**, `47.7` R1 / **49.8 tok/s** R2; wall `8.05s` / `7.71s`)
      - Slot 0 decode throughput during co-prefill: **36.5 tok/s** (**+11.3% vs control**, `35.1` R1 / **38.0 tok/s** R2; wall `7.30s` / **`6.73s` [−0.90s faster]**)
      - Slot 1 total latency: **7.53s** (**−5.3% vs control**, `7.70s` R1 / **`7.36s`** R2); engine 4,712-token prompt read: **`5,879 ms (801.6 tok/s)` R1 / `5,816 ms (810.2 tok/s)` R2**
      - Window telemetry: **`31.57 ms/window` (−6.6% vs control)** = `run 29.10 + commit 0.01 + emit 2.21 (−37.9% vs control) + adapt wait 0.26 (−78.3% vs control)`
  - **2-Slot Concurrent `2x256` Burst (`batch2`)**:
    - Control (`0_prod_control`): mean **126.6 tok/s** aggregate (`127.4` R1 / `125.8` R2; per-slot **63.8 tok/s**; slot admission copies `41.7 ms`, `44.4 ms`, `38.2 ms`; engine **147.5 / 135.1 rows/s**; warm R2 `CPU pool 8.15 ms/window`, `emit 3.75 ms/window`, `adapt wait 3.98 ms/window`)
    - Candidate (`1_cand_concurrent` @ `0247ac749`): mean **132.5 tok/s** aggregate (**+4.7% vs control, +2.8% vs pre-CPU candidate `0d468cf6b`**, **134.0 tok/s** R1 / **131.0 tok/s** R2; per-slot **68.2 tok/s [+6.8% vs control]**; slot admission copies **`17.6 ms`, `10.0 ms`, `14.4 ms`, `14.0 ms` [−67% vs control]**; engine **152.3 / 154.5 rows/s [+14.4% warm engine rows/s]**; warm R2 **`CPU pool 6.91 ms/window [−15.2% vs control, −11.1% vs pre-CPU 7.77 ms]`**, **`emit 2.82 ms/window [−24.8% vs control]`**, **`adapt wait 1.25 ms/window [−68.6% vs control]`**, acceptance **347/163 = 212.9%**)

## 7. Second-Wave CPU Concurrent Optimizations: Barrier-Free `ExpertPool` Pass, Fast-Path Dispatch, Async `sample_rows`, and Early Adaptive Tier Launch (`e18b5c8f5..bc746b98a`)

- **Mechanisms**:
  1. **Single Barrier-Free `ExpertPool` Pass (`STRATA_POOL_SINGLE_PHASE=1`, `e18b5c8f5`)**:
     - Fuses the `gate/up` (`parts_a_`) and `down` (`parts_b_`) phases into a single `run_phase(9..11, n * parts_a_ + n * parts_b_)` work-stealing pass across `run_split_multi_native`, `run_split_multi`, and `run_split` when `numa_n_ == 0`.
     - Uses per-expert `q_state` (`0 = pending`, `1 = quantizing`, `2 = ready`) so workers reaching a `down` task (`t >= n * parts_a_`) for an expert whose `gate/up + H-quantization` is already complete execute `down` immediately without a global thread-pool barrier; if `q_state != 2`, the worker helps quantize `H` (or spins briefly on `q_state == 2`), eliminating 48 extra pool wake/wait barriers per verify window.
  2. **Direct Deduplicated CPU Expert Job Construction & Scan Elision (`STRATA_DISPATCH_FAST_JOBS=1`, `0ba00b12a`)**:
     - Builds `jobs` (`ExpertJobMulti`) directly from the deduplicated `(expert -> token chain)` linked list (`cpu_i0[]`, `head[]`, `next_node[]`, `node_tok[]`, `node_w[]`) during `expert_pool_dispatch_multi`, avoiding the $O(n_{\text{tok}} \times K)$ `job_of[512]` rescan.
     - Tracks `cpu_tok_mask` (`uint64_t` bitmap for `n_tok <= 64`) during the routing classification loop and skips both the `claim_todo` loop and `job_of` construction when all cold experts are resident in RAM (`need_claim_todo == 0`).
  3. **Async `sample_rows` Before Final `Verifier::run_slot_rows` Stream Sync & `post_layers_cb_` Hook (`STRATA_VERIFY_SAMPLE_ASYNC=1`, `cd3a3ded7`)**:
     - Enqueues `sample_rows(..., /*sync=*/false)` on `cs_` immediately after `forward_head` and before the final `cudaStreamSynchronize(cs_)` in `Verifier::run_slot_rows`, using persistent host staging (`slot_hrows_`) and `cudaMemcpyAsync` into `slot_hist_d_` so CPU penalty-staging and `sample_tokens` kernel launch overlap with GPU Layer 47 + `output_norm` + 248,320-row LM-head execution (~3.8 ms).
     - Adds `Verifier::set_post_layers(PostLayersFn, void*)` invoked right after Layer 47's CPU experts finish (`*flag = 48`) and before the final `cudaStreamSynchronize(cs_)`.
  4. **Early Adaptive Tier Worker Launch During Final GPU Layer/LM-Head Execution (`STRATA_ADAPT_EARLY=1`, `bc746b98a`)**:
     - Defaulted `STRATA_ADAPT_OVERLAP=1` and `STRATA_ADAPT_OVERLAP_RESIDENT=1` on HIP (`#if defined(STRATA_USE_HIP) && !defined(_WIN32)`).
     - Uses `ver.set_post_layers(...)` in `step_batch` to launch the pinned `strata-adapt` worker thread (`adapt_impl(&adapt_run.ver_done)`) as soon as Layer 47's CPU expert pool finishes (`*flag = 48`), overlapping candidate/victim sorting across all 48 layers, `resident_stage_swaps` D2H copies, and Layers `0..46` H2D slot updates with the remaining ~3.8 ms of GPU execution (`Layer 47 MoE + output_norm + LM head + sample_rows`). Only Layer 47 slot H2D writes wait on `ver_done`.
     - Keeps `ver.wait_commit(err)` deferred across `adapt_run.th.join()` when `aov` (`STRATA_ADAPT_OVERLAP=1`) is active.
- **Verification & Paired End-to-End A/B Results (`0_prod_control` @ `cfb22e5b4` vs `1_cand_concurrent` @ `bc746b98a`)**:
  - **CTest, `pool_stress`, and `dflash2_ring_test`**:
    - `10/10` CTest parity & pool tests PASS (`mmvq_il_parity`, `quantize_il_parity`, `mmvf_rows_parity`, `shared_expert_parity`, `gr_parity`, `sampler_parity`, `verify_batch_parity`, `pool_tasks_test`, `expert_multi_test`, `pool_plan_test`)
    - `pool_stress`: `39,637` batches across 15 workers + host over 20 s (`649` after sleep), `0` jobs missed (`pool_stress OK`)
    - `dflash2_ring_test`: all 7 tests PASS (`578`-row rewind rebuild in `4.1 ms`, `2042`-row parked rebuild in `13.2 ms`, `precomp (16 rows): 225.91 us | batch_full_exec_: 43.91 us`, bitwise K/V ring and token/prob parity)
  - **Solo Refactor Parity (`2x192`, `temp=0`)**:
    - Control (`0_prod_control`): **73.8 tok/s** wall (`61.3` cold / `86.3` warm; engine decode `86.7` / `137.1 tok/s`, prompt `913` / `823 ms`), hash `ece1433a582a4c38`
    - Candidate (`1_cand_concurrent` @ `bc746b98a`): **77.9 tok/s** wall (**+5.5% vs control**, `64.7` cold / **91.0 tok/s** warm; engine decode `92.3` / **142.4 tok/s**, prompt **`885` / `759 ms`**), hash `ece1433a582a4c38` (exact bitwise match)
  - **Concurrent Decode (Slot 0: 256 tok) + Long Prefill & Decode (Slot 1: 4,712 prompt tok + 128 generated tok)**:
    - Control (`0_prod_control`):
      - Combined aggregate throughput: **46.2 tok/s** (`44.6` R1 / `47.9` R2; wall `8.61s` / `8.02s`)
      - Slot 0 decode throughput during co-prefill: **33.7 tok/s** (`32.4` R1 / `35.0` R2; wall `7.89s` / `7.31s`)
      - Slot 1 total latency: **7.97s** (`8.26s` R1 / `7.67s` R2); engine 4,712-token prompt read: `6,328 ms (744.7 tok/s)` R1 / `6,065 ms (776.9 tok/s)` R2; file-tier reads `12,459` blobs
      - Co-prefill window telemetry (`80 windows`): `34.16 ms/window = run 29.67 (CUDA0 wait 15.47 + CPU experts 13.00) + commit 0.01 + emit 3.56 + adapt wait 0.92`, `46.4 rows/s`
    - Candidate (`1_cand_concurrent` @ `bc746b98a`):
      - Combined aggregate throughput: **49.4 tok/s** (**+6.9% vs control**, `48.2` R1 / **50.6 tok/s** R2; wall `7.96s` / **`7.59s` [−0.54s avg faster]**)
      - Slot 0 decode throughput during co-prefill: **37.3 tok/s** (**+10.7% vs control**, `36.2` R1 / **38.5 tok/s** R2; wall `7.07s` / **`6.65s` [−0.74s avg faster]**)
      - Slot 1 total latency: **7.43s** (**−6.8% vs control**, `7.61s` R1 / **`7.24s`** R2); engine 4,712-token prompt read: **`5,936 ms (793.7 tok/s)` R1 / `5,704 ms (826.1 tok/s)` R2 [−377 ms avg prompt read]**; file-tier reads **`7,380` blobs (−40.8%)**
      - Co-prefill window telemetry (`75 windows`): **`31.07 ms/window` (−9.0% vs control)** = `run 28.66 (CUDA0 wait 16.47 + CPU experts 10.71 [−17.6% vs control]) + commit 0.01 + emit 2.21 [−37.9% vs control] + adapt wait 0.19 [−79.3% vs control]`, **`49.1 rows/s` (+5.8%)**
  - **2-Slot Concurrent `2x256` Burst (`batch2`)**:
    - Control (`0_prod_control`): mean **127.3 tok/s** aggregate (`126.0` R1 / `128.7` R2; per-slot **64.4 tok/s**; slot admission copies `48.7 + 3.3 ms` R1, `37.8 + 42.0 ms` R2; engine **135.5 / 150.7 rows/s**; `CPU experts 11.32 / 7.48 ms/window`, `emit 3.93 / 3.83 ms/window`, `adapt wait 3.85 / 3.61 ms/window`)
    - Candidate (`1_cand_concurrent` @ `bc746b98a`): mean **135.8 tok/s** aggregate (**+6.7% / +8.5 tok/s vs control, +2.5% vs first CPU wave `0247ac749`**, **135.7 tok/s** R1 / **135.9 tok/s** R2; per-slot **68.8 tok/s [+6.7% vs control]**; slot admission copies **`20.9 + 14.1 ms` R1, `12.4 + 12.9 ms` R2 [−68.3% R2 admission latency]**; engine **145.5 / 155.6 rows/s [+7.4% R1 / +3.3% R2]**; **`CPU experts 10.55 / 6.85 ms/window [−8.4% warm vs control, −16.0% vs 8.15 ms pre-CPU]`**, **`emit 2.64 / 2.57 ms/window [−32.9% vs control]`**, **`adapt wait 1.03 / 1.09 ms/window [−71.6% vs control]`**, acceptance **350/160 = 218.8% R1, 345/165 = 209.1% R2**)

- **Third-Wave Co-Prefill Concurrent Decode Optimizations (`b38a58c09..345c2b505`)**:
  - **Root Cause #1 — Host RAM Loan Backup for Unbacked Lent VRAM Cache Slots (`STRATA_LEND_RAM_BACKUP=1`, default on; `STRATA_LEND_PCIE_PINNED=0`, opt-in; commits `b38a58c09`, `345c2b505`)**:
    - When `--resident-budget-gib` (`34 GiB`) does not cover lent GPU cache slots (`lend_from_slot`), lending `N` GPU cache slots to co-prefill previously left those `N` experts with `has_resident == false`, forcing `7,380..12,459` file-tier `pread` blob reads during concurrent decode bursts and post-prefill `refill`.
    - Added `FileExpertSource` loan backup buffers (`reserve_loan_slots`, `ensure_loan_slot`, `loan_buffer`, `loan_tag`, `activate_loan`, `deactivate_loan`, `clear_active_loans`) in `include/strata/core/expert_source.hpp` and `src/core/expert_source.cpp`, backed by 64-byte aligned host RAM by default (`0 B` of GPU/ROCm aperture impact; `STRATA_LEND_PCIE_PINNED=1` opt-in for mapped pinned host memory).
    - In `src/program/generate.cpp` (`lend`, `refill_issue`, `refill_wait`), before `relayout` overwrites lent GPU slots, any unbacked lent GPU slot whose `loan_tag(lid) != expert_idx` is D2H-copied once into `loan_buffer(lid)` (`0 B` copied on subsequent co-prefills when the slot holds the same expert), activated as resident in RAM during co-prefill, and restored via H2D DMA from `loan_buffer(lid)` after co-prefill — **eliminating 100% of file-tier blob reads (`10,964 / 7,380 -> 0 blob reads from the file`)**.
  - **Root Cause #2 & #3 — Prefill `GatherGroup` Flush Fix, Issuer Pause on Yield, and Co-Prefill Ring Cap (`STRATA_BATCH_CO_RING=96`, default 96; commits `b13ffdabd`, `345c2b505`)**:
    - Fixed premature `flush(0)` in `Prefill::run_impl::release_to()` (`src/prefill/prefill.cpp`): when an open `GatherGroup` has fewer than `MMQ_GROUP` (`8`) experts (`gg_nslots > 0 && k - gg_ents[0] + 1 < MMQ_GROUP && k - gg_ents[0] + 1 < m.ring`), `release_to` defers slot release until `flush()` instead of breaking batched MMQ launches into 1-expert kernels.
    - Added `a_pause` atomic gate in `Prefill::run_impl` to pause the background prefill H2D issuer thread (`std::this_thread::yield()`) while `layer_yield_fn_()` runs mid-chunk decode verify bursts.
    - Called `Prefill::set_share_blocked(n_act_lend > 0)` before prefill ring sizing in `src/program/generate.cpp` and capped `ring_slots(T)` to `STRATA_BATCH_CO_RING` (default `96`, down from `112..128`) during co-prefill so `16..32` additional GPU cache slots remain resident for concurrent decode (`VRAM hits` during co-prefill rose from `84.0%` to `85.4%`, and `CPU experts` time per co-prefill window dropped from `12.16 ms` to `9.12 ms`, **−25.0%**).
  - **Root Cause #4 — Single-Slot Batch Verify `gdn_conv_ab_multi` Fusion & `capture_batch` Sync Upload Gate (`STRATA_BATCH_GRAPH_UPLOAD=0`, default off; commit `b6b59a762`)**:
    - Relaxed the `gdn_conv_ab_multi` gate in `VerifyEngine::run_layer_batch_body` (`src/core/verify.cpp`) from `S >= 2` to `S >= 1` so single-active-slot batch verify windows during co-prefill fuse `gdn_conv1d_silu` + `gdn_gate_beta` (`36` launches instead of `72` across 36 GDN layers).
    - Gated synchronous `cudaGraphUpload` + `cudaStreamSynchronize` in `VerifyEngine::capture_batch` behind `STRATA_BATCH_GRAPH_UPLOAD` (default off).
  - **Root Cause #5 — Fast Concurrent-to-Solo Slot Handoff & Delta KV Copy (`STRATA_SOLO_HANDOFF_FAST=1`, default on; commits `69bf365f8`, `345c2b505`)**:
    - Added `int64_t unchanged_tokens = 0` prefix skip to `conversation_kv_copy` (`include/strata/core/conversation_snapshot.hpp`, `src/core/conversation_snapshot.cpp`) so concurrent-to-solo handoff (`slot_source >= 0`) copies only delta KV pages `[park_held, upto)` when a parked snapshot already holds the prefix.
    - Updated `in_batch_slot` detection in `src/program/generate.cpp` (including checkpoint coverage check against `sl.checks`) to skip redundant `park_current` saves when a live conversation is already held in an active/cached batch slot, cutting post-`BGEN` redundant parking (`19.0 ms -> 0 ms`) and concurrent-to-solo handoff (`35–43 ms -> 18–27 ms`).
  - **Paired A/B Results (`0_prod_control` @ `cfb22e5b4` vs `1_cand_coprefill` @ `345c2b505`)**:
    - **File-Tier Blob Reads**: **`10,964` blobs (`0_prod_control`) / `7,380` blobs (`1_cand_concurrent`) -> `0 blob reads from the file` (`1_cand_coprefill`, −100%)**
    - **Solo Greedy Refactor (`316` prompt + `192` decode, bit-identical hash `ece1433a582a4c38`)**:
      - Control (`0_prod_control`): **73.6 tok/s** mean (`61.0` cold / `86.1` warm; engine decode `85.0 / 131.2 tok/s`)
      - Candidate (`1_cand_coprefill`): **80.8 tok/s** mean (**+9.8% vs control**, `67.3` cold / **94.3 tok/s** warm; engine decode **`92.2 / 146.6 tok/s` [+11.8% warm]**)
    - **Concurrent `4,712`-Token Prefill + Continuous Decode (`pf_dec`)**:
      - Control (`0_prod_control`):
        - Combined aggregate throughput: **45.8 tok/s** (`44.3` R1 / `47.4` R2; wall `8.66s` / `8.11s`)
        - Slot 0 continuous decode during co-prefill: **32.3 tok/s** (`31.1` R1 / `33.4` R2; wall `8.22s` / `7.66s`)
        - Slot 1 total latency (`4,712` prompt + `128` decode): **8.04s** (`8.31s` R1 / `7.76s` R2; prompt read `6,242 ms` R1 / `5,934 ms` R2)
        - Warm co-prefill window telemetry (`92 windows`): `34.12 ms/window = run 29.19 (CUDA0 wait 15.49 + CPU experts 12.16) + commit 0.01 + emit 3.61 + adapt wait 1.30; VRAM 84.0%, CPU 15.1%`
      - Candidate (`1_cand_coprefill` @ `345c2b505`):
        - Combined aggregate throughput: **48.7 tok/s** (**+6.3% mean / +7.8% warm R2 vs control**, `46.4` R1 / **51.1 tok/s** R2; wall `8.28s` / **`7.52s` [−0.59s warm]**)
        - Slot 0 continuous decode during co-prefill: **37.4 tok/s** (**+15.9% mean / +17.2% warm R2 vs control**, `35.6` R1 / **39.2 tok/s** R2; wall `7.19s` / **`6.53s` [−1.13s faster during co-prefill]**)
        - Slot 1 total latency (`4,712` prompt + `128` decode): **7.55s** (**−6.1% mean / −7.6% warm R2 vs control**, `7.93s` R1 / **`7.17s`** R2; warm prompt read **`5,700 ms (826.7 tok/s)`**, solo handoff **`18 ms` R1 / `27 ms` R2**)
        - Warm co-prefill window telemetry (`76 windows`): **`29.35 ms/window` (−14.0% vs control, −5.5% vs CPU wave 2)** = `run 27.02 (CUDA0 wait 16.56 + CPU experts 9.12 [−25.0% vs control, −14.8% vs CPU wave 2]) + commit 0.01 + emit 2.21 [−38.8% vs control] + adapt wait 0.12 [−90.8% vs control]; VRAM 85.4% (+1.4 pp), CPU 13.9% (−1.2 pp)`
    - **2-Slot Concurrent `2x256` Burst (`batch2`)**:
      - Control (`0_prod_control`): mean **126.6 tok/s** aggregate (`125.2` R1 / `128.0` R2; per-slot **64.3 tok/s**; engine **135.2 / 148.0 rows/s**)
      - Candidate (`1_cand_coprefill` @ `345c2b505`): mean **135.2 tok/s** aggregate (**+6.8% vs control**, `134.9` R1 / `135.6` R2; per-slot **68.5 tok/s**; slot admission copies **`11.3 + 12.8 ms` R1, `9.8 + 14.0 ms` R2 [−70.0% R2 admission latency]**; engine **154.6 / 155.2 rows/s [+14.3% R1 / +4.9% R2]**; **`CPU experts 9.92 / 6.83 ms/window [−14.4% R1 / −10.2% R2 vs control]`**, **`emit 2.56 / 2.50 ms/window [−33.2% vs control]`**, **`adapt wait 0.92 / 1.03 ms/window [−74.9% vs control]`**)


