# DFlash2 & Target Verifier Optimization Campaign (2026-11-07)

- **Worktree**: `/home/soren/Strata-perf-dflash2-on-main-20261103`
- **Branch**: `perf/dflash2-tap-overlap-16row` (`d1c3d7fb3`, 21 atomic commits on top of `fork-main` @ `d80835e40`)
- **Hardware**: AMD Radeon AI PRO R9700 (`gfx1201` / RDNA4, 32 GiB VRAM, ROCm nightly `10.2.0a20260914`)
- **Model**: `qwen3.8-flash-next-unsloth-ud-iq4_xs-v31` + `qwen3.8-flash-next-dflash2.q4_0-selq4.pack` (`--batch 2`, `--spec 8`, `--draft-conf 0.5`, `--resident-experts --resident-budget-gib 34`)

## 1. Atomic Commit Stack (`d80835e40..d1c3d7fb3`)

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
