# Fork notes displaced by the upstream v0.2.6 doc rewrite — docs/AMD_HIP.md

> **Archival addendum, not maintained.** The 0.2.6 merge (2026-10-25) took upstream's rewritten
> `docs/AMD_HIP.md` wholesale; the fork-side passages that rewrite replaced are kept here verbatim, in the
> order they appeared, so measured results and community validation notes stay at the branch tip.
> They were written against the pre-0.2.6 text and are not updated when upstream's docs move.
> Source: branch `integ/pre-upstream-0.2.6` (commit `cf70f46a`), the file's conflict blocks.
> Recover the whole pre-merge file with `git show cf70f46a:docs/AMD_HIP.md`.

## Quick start

- **Detection:** setup finds the card through the kernel's KFD topology. Integrated Radeon GPUs are listed as not
  supported. On a PC without an NVIDIA card Strata can use, `--backend hip` is chosen automatically.
- **ROCm:** a system ROCm 7 in `/opt/rocm` (or `$ROCM_PATH`) with hipcc and hipBLAS is used when present. Otherwise
  (or when it is older than 7.0) ROCm is installed into `.venv` from AMD's TheRock wheels (~10 GB, no sudo), from the
  card family's index: `gfx110X-dgpu` for gfx1100 / gfx1101, `gfx120X-all` for gfx1200 / gfx1201, `gfx103X-all` for
  gfx1030, `gfx1151` for Strix Halo. The version is chosen per family: 7.10.0a20251120 where that is on the index,
  7.13.0a20260515 for `gfx103X-all` (its index starts at 7.13, #1103) and 7.14.0a20260608 for `gfx1151` (the 7.10 wheel
  segfaults on kernel 7.2.8, #1267). When the index no longer offers the version, setup warns and takes the newest of
  the same 7.x line, else of the same major. `STRATA_ROCM_VERSION` / `STRATA_ROCM_INDEX` override them; the gfx1030
  wheels are unvalidated (a system ROCm 7 is the tested path there).
- **Engine:** compiled on your PC for the card's architecture (10-20 minutes, once; again after a `git pull` that
  changes it, or when you pick a card of another architecture). This needs a C++ compiler and git
  (`sudo apt install build-essential git`).
- **hipBLASLt tuning table:** setup uses `tools/hip/<arch>-hipblaslt-<version>.txt` only when it matches both the
  card's architecture and the installed hipBLASLt version (read from `hipblaslt-version.h`; 1.2.0 is `100200`).
  Otherwise it says so and the prompt's dense matrix products use plain hipBLAS (slower prompts, same answers).
  A table's solution ids are valid only for that pair, and the engine refuses any other table. On gfx103x (RDNA2,
  no hipBLASLt kernels) the same role falls to a rocBLAS table, `tools/hip/<arch>-rocblas-<version>.txt`, which the
  engine picks for its own rocBLAS build when setup points it at the directory ([RDNA2](#rdna2-gfx1030)).
- **Several cards:** setup takes one card (the one with the most VRAM, or `--gpu N`) unless you name more:
  `./setup.sh --backend hip --gpus 1,0` splits the model's layers across them, the faster card goes last and is the Main card (`--gpu-roles 1=main,0=helper` says it yourself; docs/MULTI_GPU.md) (numbers
  as setup lists them; `--gpus all` = every supported card, the most VRAM first). Every chosen card must be one of the
  architectures above; the engine is compiled for each of them (cards of two families, e.g. gfx1100 + gfx1201, need
  a system ROCm 7: AMD's wheels hold one family). A split pays only when no single card holds the model's experts
  (see RDNA4 below).
- **Pictures:** on by default; the reader is the CPU one under 24 GB of VRAM and the GPU one from 24 GB (the same rule on NVIDIA and AMD, `strata_control/vision_reader.py`); `--vision gpu` / `--vision cpu` switch it. The engine package carries two picture readers: `strata-vision-hip`, which reads a picture on the
  card (llama.cpp's HIP backend; about 1.4 GiB of VRAM, measured on an R9700: 0.06 s for a 260-token picture and 0.4 s for 972
  tokens, against 3.1 s and 14 s on the CPU), and `strata-vision`, the CPU reader. If the GPU reader cannot start (the card is
  not one it was built for, the HIP runtime does not load, the card is busy) the server starts the CPU reader by itself and
  prints one line; nothing is asked. `--vision cpu` picks the CPU reader on purpose.
- **Limits for now:** Setup does not offer the tuning
  (calibration) on AMD yet: its controls are being checked on HIP one at a time (#566). Since 0.1.39 a tuning run by
  hand (`./setup.sh --calibrate`) is saved for the AMD card it ran on and reused when setup runs again. The Monitor
  shows the card's load, VRAM, temperature and power from Linux sysfs (0.1.32); on Windows from the driver's ADL
  library (`atiadlxx.dll`, part of Adrenalin: activity, graphics temperature, board power where the card reports it,
  else ASIC power), the OS counter
  `\GPU Adapter Memory(<luid>)\Dedicated Usage` (checked on a Radeon 8060S; an APU's ASIC power is the whole chip),
  and the PCIe link Windows itself keeps for the card (its device properties; the driver's PMLog link channels fill
  in where those are missing, and where the two disagree about width the narrower one is shown).

## Community-validated cards

Run by their owners, not on the maintainers' machines; setup accepts them like gfx1100 / gfx1201. setup ships a
hipBLASLt table for gfx1200 (the numbers below) and for gfx1101 with hipBLASLt 1.2.0; for gfx1101 with another
hipBLASLt, make one with [Tuning table](#tuning-table) and compare the prompt speed with and without it.

## RDNA2 (gfx1030)

- **Build and tests** (engine 0.1.26): a complete HIP build for gfx1030, made by hand with cmake and ROCm's own
  `clang++` (the nixpkgs ROCm is not an `/opt/rocm` tree, so setup's `build_engine_hip` was not exercised; the
  binary was placed in `engine/` for setup to use). All 28 registered ctest tests pass on the card, including
  `hip_device_selftest`. An engine 0.1.30 build of this branch configures, builds and runs clean on the same card;
  the speeds below are measured on it.
- **End to end** (Swift 1.5 IQ3_XXS, `--context 131072` with `--kv-resident 32768 --adapt-every 1
  --vram-reserve-mib 1024`, 200 greedy tokens): 38-42 tok/s decode with the default 15 CPU pool workers (one per
  physical core except the host thread; 8 workers: 36; 24 = every logical core: 27), consistent even with the
  131,072-token context full.
- **Prefill** (the same 15-worker configuration): 246 tok/s on a 2,000-token prompt, 330-339 tok/s at the auto
  8,192-token chunk (7,997 and 15,967 tokens; time to first token 8.9 and 48.3 s), 133 tok/s on a 522-token
  prompt - short prompts are fixed overhead (19-25 tok/s on 34 tokens). Decode after a 16K prefill holds at
  45.5 tok/s.
- **16 GB card:** the expert cache holds 6,310 slots (10.2 GiB) at 16K context and 5,247 slots at 131,072, where
  setup keeps the KV cache in VRAM because IQ3_XXS needs about 60 GB of RAM plus the cache to stream it. The
  `--pcie-frac 0` and `--adapt-every 0` of the 7900 XTX configuration in
  [AMD_HIP_PERFORMANCE.md](AMD_HIP_PERFORMANCE.md) cost 8.6 and 13.6 tok/s here (30 with the defaults): keep the
  defaults on a 16 GB card.
- **hipBLASLt:** ROCm's hipBLASLt ships no gfx1030 kernels, so there is no table and the plain hipBLAS path runs.
- **Prompt GEMMs in FP16 (#835):** rocBLAS on gfx1030 has tuned kernels for FP16-in / FP16-out GEMMs only; the prompt
  path's FP16-in / FP32-out and BF16 products ran generic kernels about 6.6x slower (5.6 and ~5.3 TFLOPS against 37.7 at
  N = 10240, T = 7313, K = 2560). `STRATA_HIP_PROMPT_F16=1` (opt-in, off by default; the engine prints a tip on gfx103x) runs the 16-bit prompt GEMMs FP16 in and out;
  unset is the old path. `STRATA_DBG_NAN=1` also counts the non-finite FP16 outputs (a sum past 65504). The output is not bit-identical to it (an FP16 rounding of each GEMM's output replaces the BF16 rounding of
  the BF16 GEMMs' activations); what the distribution check showed and did not show is in
  [bench/results/2026-10-04-rdna2-fp16-prompt](../bench/results/2026-10-04-rdna2-fp16-prompt/README.md). FP16 ends at
  65504: `STRATA_F16_RANGE=1` prints, per prompt, the largest value that entered or left FP16 and how many were beyond
  the range (466K tokens there: outputs peak at 410, activations at 112, none beyond; the image writers saturate, a GEMM
  output beyond the range would be Inf and is what the counter is for). `STRATA_HIP_PROMPT_F16=1` is the documented
  historical all-device setting; `=all` is a new explicit all-device alias, not a historical option. The parser also
  retains the old first-character `1` check (so `=10` still selects all devices), though only `=1` is documented.
  `=gfx103x` selects FP16 only on gfx103x and leaves other devices on BF16; `=0`, unset, empty, and other unrecognized values
  use BF16 everywhere unless they begin with `1`, which retains the old all-device behavior.
  The filtered mode is intended for mixed gfx103x / gfx12x layer splits. It is a per-device policy, not a diagnosis of
  the reported `hipErrorNoBinaryForGpu`; the rocBLAS/runtime cause has not been isolated. Measured on a
  second community machine - 2x RX 6900 XT 16 GB (one card for these numbers), Ryzen 5 5600X (6 cores, AVX2), 128 GB,
  Ubuntu 26.04, ROCm 10.0.0, engine 0.1.39 built by hand with `-DSTRATA_PREFILL_MMQ=ON` as setup does, GSQ-RCO IQ3_S:
  a 9.4K / 34.7K / 105.8K-token prompt reads at 744 / 915 / 926 tok/s instead of 439 / 466 / 461; decode is unchanged.
  On that machine the HIP ctest passes 60 of 65 with and without this change: 2 skipped (`hip_prompt_attn_wmma`, no
  matrix cores; `hip_prefill_hipblaslt_gemm`, no hipBLASLt table) and 3 that fail for reasons outside the engine
  (`ple_parity` needs a Q2_0 PLE file that is not on that machine, `expert_multi_test` refuses the CPU without AVX-512,
  `platform_memory_test` cannot `mlock` at the shell's default `ulimit -l`).
- **The verifier's hyper-connection read as one persistent kernel:** `STRATA_HC_PERSIST=1` (opt-in, HIP, 32-lane
  waves) runs the read of 1-8 tokens as one kernel of 80 blocks with two grid barriers instead of three launches; a
  start-up run per token count with a barrier timeout decides where it is resident, and one `strata hc:` line per card
  says so. On 2x RX 6900 XT (ROCm 10.0, layer split): the read 1.35-2.10x, ms per verify window -4.2 % / -2.8 % at
  4K / 32K (6 rounds), not bit-identical (as close to a double-accumulation reference as the default read):
  [bench/results/2026-10-12-hc-persist-gfx1030](../bench/results/2026-10-12-hc-persist-gfx1030/README.md). Other
  RDNA cards are untested; the `strata hc:` line and a `STRATA_DECODE_TIMING` run from one are welcome. Not offered on gfx11 and
  gfx12 (RDNA3, RDNA4): the barrier timeout needs `s_memrealtime`, which gfx1100 and gfx1200 / gfx1201 do not have, and an R9700 (#2226 thread) measured the
  read +0.7 % / +1.0 % per window slower than the default one; the start-up line says so. A barrier that times out at run time
  (never seen in a working run) leaves the partial outputs of that read in place and every later persistent read of the
  process aborts at once, without a log line; the opt-in is for measurement on gfx103x until that is closed.
- **Prompt GEMMs as SGEMMs (engine 0.1.39 + `Gemm::rdna2_sgemm`):** the rocBLAS in setup's ROCm (10.2.0a20260930)
  has tuned gfx1030 kernels for FP16 -> FP16, int8 and FP32 -> FP32 only. The prompt path's dense GEMMs take FP16 or
  BF16 inputs and write FP32 (rocBLAS's HS / BS types), which run its fallback kernels at about 5 TFLOPS. With
  `STRATA_RDNA2_SGEMM=1` (opt-in; `=all` forces it on any AMD card) the engine widens both inputs to FP32 (exact) and runs SGEMM,
  which accumulates in FP32 like the native call (`src/prefill/gemm.cu`; N < 64 keeps the native call). The author ran it
  as the gfx103x default; here it is a switch because it is not bit-identical to the native call (below). Measured on an RX 6800 16 GB over OCuLink (PCIe 4.0 x4, 7.1 GB/s), Ryzen 7 8845HS, 28.8 GB RAM, Windows 11,
  Coder IQ1_M with setup's arguments (64K context, MTP), fresh 12K-token prompts, 4 per arm, A/B in one binary:

## RDNA2 (gfx1030) (2)

  The exact calls on the card (opA = T, opB = N, FP32 out): N 10240 x T 8192 x K 2560 86.7 -> 27.7 ms including
  the widening; N 320 x T 8192 x K 10240 (BF16) 10.9 -> 7.5 ms. Against a float64 reference on random inputs in
  [-1, 1], the SGEMM route is as close as or closer than the native call on all 10 FP16 / BF16 shapes checked; the
  two differ by at most 4.3e-6 of the largest output. A 9-task code-repair benchmark (the model patches seeded bugs
  in a browser game until 38 browser checks pass) solved 9/9 on the first try with both, model time 458 -> 374 s.
  With the switch on and `STRATA_HIP_PROMPT_F16` unset the SGEMM route runs; with `STRATA_HIP_PROMPT_F16=1`
  the FP16 route above runs first and the SGEMM route takes the products it leaves (`beta != 0`). Same RX 6800, one
  session, fresh 12K-token prompts: native 310, SGEMM 439, FP16 (#835) 494 tok/s; `STRATA_F16_RANGE=1` on the Coder
  IQ1_M peaked at 105.3 (activations) and 424.8 (FP16 outputs) with nothing beyond 65504, and the repair benchmark
  solved 9/9 on the first try with both (model time 374 s SGEMM, 307 s FP16). The SGEMM route is the slower of the
  two; no value passes through FP16, so it has no 65504 limit to stay under. It is not bit-identical to the native
  call (at most 4.3e-6 of the largest output apart), which can flip a near-tied greedy token like any such change.
- **rocBLAS solution table** (for the `STRATA_HIP_PROMPT_F16=1` route above; with it off, nothing runs through the
  table): rocBLAS picks the kernel for each of those FP16 GEMMs from its own table by shape, and
  on gfx1030 that pick is far from the best kernel the library holds for some shapes: the GDN input projection
  (N = 10240, K = 2560) below about 1,150 tokens runs at 5 TFLOPS where the best of rocBLAS's own 237 solutions for the
  same shape runs at 32-38 (7x; the kernel choice flips at T = 1152). `tools/hip/tune_rocblas` enumerates the library's
  solutions per shape and token bucket, checks each against the default kernel's output and writes the winners as
  `tools/hip/<arch>-rocblas-<version>.txt`; the engine reads the table through `STRATA_ROCBLAS_TUNING` (a file, or
  the directory, from which it takes the file of its own rocBLAS build) and runs those solutions in place of rocBLAS's
  choice, the default kernel for every shape without a row. Solution indices are valid for one architecture and one
  rocBLAS build (the full version string, `5.6.0.8d1ae90e` for ROCm 10.0.0's), and the engine refuses any other
  table. Shipped: `gfx1030-rocblas-5.6.0.8d1ae90e.txt` (ROCm 10.0.0, the `/opt/rocm` setup installs on Ubuntu) and
  `gfx1030-rocblas-5.4.0.ad84daa81c.txt` (TheRock's `gfx103X-all` 7.13.0a20260515 wheels, which setup installs into
  `.venv` when there is no system ROCm 7), both measured on an RX 6900 XT. With the 7.13 table, a 268-395-token
  follow-up on a cached conversation read in 1,635 ms instead of 1,851 (median of 6 each, every pair 10-13% faster),
  a fresh 745-1,569-token prompt in 3,435 instead of 3,554 ms and a fresh 4.9-6.7K-token one in 8,976 instead of
  9,070 ms (UD-Q4_K_XL with `STRATA_HIP_PROMPT_F16=1`, which the table needs; nothing else differed between the two).
  Other gfx1030 cards (RX 6800 / 6800 XT / 6950 XT, Radeon PRO W6800 / V620) run it as well - the
  indices are valid on every gfx1030 - but have fewer CUs or other clocks, so their own table may pick differently;
  `tune_rocblas` makes one in about 5 minutes ([Tuning table](#tuning-table)), and a report of what it changed is welcome. Setup
  points the engine at the directory when a table for the card's architecture exists, and the engine's log says
  `rocBLAS tuning enabled (... rows ...)` or why not. Measured on the 2x RX 6900 XT machine above (one card, #835 +
  this, `STRATA_PREFILL_TIMING=1`, temperature 0, the same answers with and without): a 799-token prompt's GPU time
  3,329 -> 2,985 ms (the GDN projections 443 -> 177 ms, `hc read` 270 -> 77), 495 tokens 2,477 -> 2,139 ms (365 ->
  146), 8,874 tokens 11,978 -> 11,459 ms (the 682-token tail chunk's projections, `hc read` 1,200 -> 898); a
  1,108-token prompt did not change (3,391 -> 3,395 ms): its GDN time fell 598 -> 201 ms and the whole saving
  reappeared as `wait copy`, the experts' PCIe transfer, which bounds that prompt. Per-run lines and the tuner's
  summary: [bench/results/2026-10-05-rdna2-rocblas-table](../bench/results/2026-10-05-rdna2-rocblas-table/README.md). Decode does not use these GEMMs and is unchanged. See [Tuning
  table](#tuning-table) for making a table for another rocBLAS build.
- **QSA block scores on the FP32 tiled scorer (`STRATA_SELECT_SIMT=1`, opt-in, as on CUDA below sm_80):** without
  matrix cores the prompt path's QSA selection scores every (query, KV block) pair on the warp kernel, the one prompt
  term that grows with the context. The switch runs the FP32 tiled scorer instead (FP32 in another summation order:
  not bitwise the warp kernel, so off by default), tiled 32/128/16/8 on gfx103x. `qsa_select_bench`, 256 queries:
  2.69 -> 0.587 ms at 128K (the original 32/64/32/4 tiling: 0.695), 0.67 -> 0.169 ms at 32K; FP64 gate passed,
  selections identical 256/256. Every tiling gives the same bytes (each score is one fmaf chain over k in order), and
  the fastest tiling was the same with 60 and 72 of the card's 80 CUs (`HSA_CU_MASK`). On the 2x RX 6900 XT split
  (0.1.40.3, IQ3_S, `--kv int8 --kv-resident 32768`, two cold starts per arm, ABBA) a 128K-token prompt reads 1,735 ->
  1,861 and 1,727 -> 1,858 tok/s (+7.4%), 32K +0.8%; decode is not affected (the text changes with the scores' order,
  so its speed is not compared). Measurements:
  [bench/results/2026-10-07-rdna2-qsa-select-simt](../bench/results/2026-10-07-rdna2-qsa-select-simt/README.md).
- **File-tier cache rule on gfx906 (`STRATA_FILE_CACHE_OLD_RULE`):** on an MI50 ROCm queues are suspended while the kernel reclaims around the process's HMM-tracked arena (#1752, KevinX8: stalls, prompt 46 -> 360 tok/s once the file tier stops churning the page cache). The gfx906 backend therefore keeps the old rule (the file cache is used only when the room left is at least the bytes read from the files); other HIP cards keep the current rule. `STRATA_FILE_CACHE_OLD_RULE=1` forces the old rule on any Linux build, `=0` the current one.
- **Windows** (the report above): setup's ready-made Windows engine 0.1.39 runs the RX 6800 beside a 780M iGPU
  (HIP numbers the RX 6800 1; setup points the engine at it), and `tools/hip/build_windows.bat` with
  `STRATA_HIP_ARCHS=gfx1030` builds a working engine from the `gfx103X-all` 10.2.0a20260930 wheels.
- **gfx1031** (RX 6700 XT, #524): setup knows it (the `gfx103X-all` wheels, unvalidated); its reporter runs it daily
  on one card. More reports: an RX 6700 XT 12 GB run as gfx1030 on ROCm 7.2.4 (#1027: IQ2_XS, decode 30-32 tok/s, prompt about 300 tok/s, 6 of 6
  needles), and an RX 6800M 12 GB on Windows with a self-built engine (#915, #1078: Q2_0, decode 9-27 tok/s, prompt 50-114 tok/s; the
  experts stream from disk with 31 GB of RAM). The ready-made Windows zip has no gfx1031 code unless it is built with it: `STRATA_HIP_ARCHS=gfx1031`
  in `tools\hip\build_windows.bat` (the default list has it from 0.1.40.2 on).
- **gfx1150** (Radeon 890M, Ryzen AI 9 HX PRO 370, Strix Point, #1217): builds and runs from 0.1.40.2 as an unvalidated target (CMake warns; the
  device code already covers it: the WMMA guards and `gfx_arch_is_gfx11_wmma()` include it). One community machine: MINISFORUM N5 PRO, 96 GB DDR5-5600, GTT
  raised to 64 GiB (`ttm.pages_limit=16777216 ttm.page_pool_size=16777216`), TheRock ROCm 7.14.1 for gfx1150, an unprivileged LXC container, IQ3_XXS: chat and
  tool calls work, decode 15-19 tok/s, the HIP ctest passes apart from tests that need files a public checkout lacks.
  Prompts read at about 125-130 tok/s on plain hipBLAS and 1.5-1.7x faster with `tools/hip/gfx1150-hipblaslt-100401.txt` (ROCm 7.14.1's
  hipBLASLt; 3.6K tokens 124 -> 216 tok/s, 7K 130 -> 226 tok/s, decode unchanged):
  [bench/results/2026-10-07-community-gfx1150](../bench/results/2026-10-07-community-gfx1150/README.md).
  Setup does not install for it yet (an integrated Radeon other than Strix Halo is named and not supported): build by hand with `-DCMAKE_HIP_ARCHITECTURES=gfx1150`
  and point `STRATA_HIPBLASLT_TUNING` at the table yourself.
- **gfx1152** (Radeon 860M / 840M, Ryzen AI 300 "Krackan", #1625): in CMake's unvalidated list like gfx1150, so
  `-DCMAKE_HIP_ARCHITECTURES=gfx1152` builds with a warning. It runs the portable kernels (the matrix-core guards name
  gfx1150 and gfx1151 only), no card has reported on it, and setup does not install for it (an integrated Radeon other
  than Strix Halo is named and not supported). A report from a real 860M is welcome.
- **gfx1032** (RX 6600/6600 XT): unvalidated; built with `STRATA_HIP_ARCHS=gfx1030;gfx1032` and ran under Windows as
  the *helper* expert GPU of an RX 6800 ([SECOND_GPU.md](SECOND_GPU.md)), 3.4-3.8k expert slots in its 8 GB and
  +8-18% decode over the 6800 alone. Its 14 WGP make a helper row about 2-3x the 6800's cost, so it pays as a
  cache tier for the CPU pool, not as a compute peer. No report of it as a primary card.
- **Not validated:** setup's own build path and the
  `gfx103X-all` wheels on gfx1030, images, answer-quality benchmarks. RDNA1 (gfx1012, RX 5500 XT) builds by hand:
  [OLDER_GPUS.md](OLDER_GPUS.md#amd-building-gfx906-and-gfx1012).

## Tuning table

- `gfx1150-hipblaslt-100401.txt`: Radeon 890M (gfx1150, Strix Point APU, 16 CUs), calibrated with AMD's TheRock ROCm 7.14.1 for
  gfx1150 (hipBLASLt 1.4.1 `cd957402`, version number 100401) and the engine's 32 MiB workspace. The 16 dense GEMM geometries a
  Qwen3.8-Flash-Next prompt logs on that card (the same 16 as the gfx1201 100500 table) at T = 64, 128, ..., 16384: 144 rows. Two
  `tune_hipblaslt` passes back to back; each row keeps the solution with the lowest median of all six repetitions, because one
  repetition stalled now and then (a 100 ms call read up to 1.6 s) and the tool's mean-of-three picks differed in 26 rows between
  the passes. Every row beats plain hipBLAS, 1.55-22.6x per GEMM (geometric mean 4.5x), and `STRATA_HIPBLASLT_VERBOSE=1` reported
  no fallbacks over prompts of 28-7,880 tokens. Setup does not use it (it does not install for gfx1150): point
  `STRATA_HIPBLASLT_TUNING` at it. Measurements and the per-row timings:
  [bench/results/2026-10-07-community-gfx1150](../bench/results/2026-10-07-community-gfx1150/README.md).
- `gfx1103-hipblaslt-100401.txt`: Radeon 780M (gfx1103, RDNA3, 12 CUs, unified memory), calibrated with ROCm 10.1
  (`rocm/dev-ubuntu-24.04:10.1.0-full`, hipBLASLt 1.4.1, version number 100401) and the engine's 32 MiB workspace. The
  16 dense GEMM geometries a Qwen3.8-Flash-Next prompt logs on the card at T = 64, 128, ..., 16384: 144 cases, two
  `tune_hipblaslt` passes back to back, each row the solution with the lowest median of all six repetitions. One row
  is left out (`bf16 N=1 K=2560 T=64`: no candidate beat plain hipBLAS at 0.04 ms), so the table has **143 rows**,
  1.05-23.30x per GEMM, geometric mean 4.95x. Without it the engine prefills on plain hipBLAS: a 4K prompt on Q2_0
  read 90 tok/s, 187 with the table and 277 with the WMMA opt-ins (one 780M; 32K context, MTP). The merged table
  prefilled the 4K prompt at 187.2 tok/s. A context ladder to 131K (shipping configuration, fresh prompts) puts the
  full 128K read at 198-221 tok/s, its last ~14K at ~206, and decode at ~10 tok/s at that depth; the numbers are in
  the report folder. `hip_prefill_hipblaslt_gemm` reports `fallbacks=0`;
  `hip_prefill_hcd_exact_parity` skips, because the table picks no 1176 / 1177 for the HC down GEMM (the skip the
  gfx1150 table gets too). Setup installs for gfx1103 only with `STRATA_EXPERIMENTAL_GFX1103=1` and then uses the
  table when the installed hipBLASLt reports 1.4.1. Measurements:
  [bench/results/2026-10-12-community-gfx1103](../bench/results/2026-10-12-community-gfx1103/README.md).
- `gfx1101-hipblaslt-100200.txt`: RX 7800 XT (gfx1101, 16 GB), calibrated with the ROCm 7.10.0a20251120 wheels that
  setup installs for gfx110X (hipBLASLt 1.2.0, version number 100200); setup uses it only when the installed hipBLASLt
  reports 1.2.0. The 43 geometries of `gfx1100-hipblaslt-100401.txt` (the small-T expert GEMM included), calibrated
  with `tune_hipblaslt --workspace-mib 32`. Two passes picked the same solution in all 43 rows; every row beats plain
  hipBLAS, 2.98-16.6x per GEMM (median 4.9-6.0x). `hip_prefill_hipblaslt_gemm` passes with it (`launches=4
  fallbacks=0`); `hip_prefill_hcd_exact_parity` skips (the table sends no tested T to solution 1176 / 1177). Ryzen 7
  5800X3D, 60 GiB RAM, Ubuntu 26.04 (kernel 7.0.0), engine 0.1.44.1 in Docker, Qwen3.8-Flash-Next GSQ-RCO IQ3_S at
  131072 ctx, `--kv q4_0`, three interleaved pairs: a cold prompt of 35,537 tokens reads at 447.0 -> 761.9 tok/s
  (median; 446.7-449.1 -> 758.0-762.1), 91,320 tokens at 443.6 -> 751.9 tok/s (443.4-444.5 -> 751.6-752.5), both
  +70%; decode after the prompt is unchanged (42.0 / 48.5 -> 41.5 / 49.3 tok/s), and `STRATA_HIPBLASLT_VERBOSE=1`
  reported `launches=13328 fallbacks=0` in each run. Measured on a 7800 XT only.

