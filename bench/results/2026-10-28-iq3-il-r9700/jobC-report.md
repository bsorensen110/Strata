# Job C — `mmvq_il_parity --bench --emit-table` on the Radeon AI PRO R9700 (gfx1201)

Repo `/home/soren/Strata-integ`, branch `integ/community-2026-10`, 2026-10-28. Files I touched in the repo:
`src/kernels/cuda/native_mmvq.cu` and `CMakeLists.txt` (nothing else; the patch is `jobC-edits.patch` here,
253 lines, refreshed against the final worktree diff). Everything else in this report is measured on this machine or read from this tree.

**Verdict in one line: the bench runs on the HIP build, parity is `OK` in two independent runs, and the IL win on
gfx1201 is real and large — 65 of 75 cells in the 2-4 column table, 98 of 100 in the 5-8 table, 10 of 25 in the
one-column table come out non-zero, with the Q5_K LM head at 4 tokens going 1713 µs → 879 µs (-48.7%).**

---

## 1. Edits made (file:line)

### `src/kernels/cuda/native_mmvq.cu`

| Line | Change |
|---|---|
| 35 | `#include "q8_1_il.cuh"` is now unconditional (it was wrapped in `#if !defined(__HIPCC__)` at 35-37). HIP has every construct the header uses: `__ldg` for `int2/int4/float2/float4` (`/opt/rocm/include/hip/amd_detail/hip_ldg.h:14-80`) and for `half2` (`amd_hip_fp16.h:605-613`). |
| 1623-1633 | The IL family's opening gate `#if !defined(__HIPCC__)   // CUDA only: the HIP paths keep native_mmvq's kernels` (old 1623) is gone: the region is compiled for every backend, and a new switch `STRATA_IL_PF` is defined there — `0` on `__HIPCC__`, `1` otherwise. |
| 1668 / 1735 | `IlIQ4XS`: its P1/P2 members (`HAS_PF`, `RP`, `rowptrs`, `WR`, `at`, `loadr`, `xb`, `xp`, `applyr`) wrapped in `#if STRATA_IL_PF`. |
| 1740 / 1806 | `IlQ4K`: same members, same guard. |
| 1830 / 1903 | `IlQ5K`: same. |
| 1984 / 2031 | `IlQ6K`: same (`HAS_PF`, `RP`, `rowptrs`, `WR`, `at`, `loadr`, `xb`, `xp`, `applyr`). |
| 2034 / 2190 | `native_mmvq_il_pf_kernel` (the whole P1/P2 kernel template) wrapped in `#if STRATA_IL_PF`, with the reason in the comment: its addresses come from PTX `mad.wide.u32` (`madw`, `I::at`) and its loads are `ld.global.nc` (`ldg_nc_*`, `I::loadr`), which exist only in the CUDA helper block at 653-712 — that block is untouched and stays CUDA-only. |
| 2266 / 2277 | `launch_il_one`: its `if constexpr (I::HAS_PF)` variant dispatch wrapped in `#if STRATA_IL_PF`; a HIP build always launches `native_mmvq_il_kernel`. (This was needed to compile: `launch_il_one` names the pf kernel template, so the name must exist when the region is un-gated.) |
| 2358 | The region's closing `#endif  // !__HIPCC__` → `#undef STRATA_IL_PF`. |
| 3573 | The tables/entry-point region's `#if !defined(__HIPCC__)` (old 3554, the one that started `native_q8_1_il_bytes`) is gone; a heading comment says the region is for every backend. |
| old 3952-3967 | The `#else // HIP: no interleaved path` stubs are deleted: `native_q8_1_il_bytes` (which returned `native_q8_1_bytes * 2`), `native_quantize_q8_1_il`/`native_q8_1_interleave` (which threw "… is CUDA only"), `native_mmvq_il_supported` (`return false`), `native_mmvq_il_rows_for` (`return 0`), `native_mmvq_il` (plain `native_mmvq`), `native_mmvq_il_tune`/`native_mmvq_il_pf` (empty). One implementation now serves both backends; the file ends at 3971. |
| 1929 | Not changed: `IlQ3K`'s `HAS_PF = false` stays as it was. |

No CUDA-only intrinsic had to be ported: the IL kernels use `STRATA_DP4A`, `__vsubss4`, `__shfl_*_sync`,
`__half*`/`__ushort_as_half` and `__int_as_float`, all of which hip_compat already provides
(`include/strata/hip_compat/intrinsics.hpp:146-154`, and the fp16 header for the half conversions). Wave32 is the
right shape for RDNA4 and `WARP = 32` (line 61) is what the IL chunk layout is written against.

### `CMakeLists.txt`

| Line | Change |
|---|---|
| 1335 | `if(STRATA_ENABLE_CUDA)` → `if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)` for the IL parity block: `add_executable(mmvq_il_parity …)` (1338), its include dirs / link libs (1339-1340) and `add_test(NAME mmvq_il_parity COMMAND mmvq_il_parity)` (1341). |
| 1342-1391 | HIP branch: a target-scoped shim header generated into the build dir by `file(GENERATE)` (`build-hip/mmvq_il_parity_hip_l2.h`, 1343) and `-include`d for that target by `target_compile_options` (1390-1391) — the same shape the `ds4_*` HIP targets already use (`SHELL:-include …/dsv41/hip_extra.h`, lines 1061-1091). It maps `cudaDevAttrL2CacheSize`, `cudaDeviceSetLimit`, `cudaDeviceGetLimit`, `cudaStreamSetAttribute`, `cudaStreamAttributeAccessPolicyWindow`, `cudaStreamAttrValue`, `cudaAccessPropertyPersisting/Streaming` to HIP's own names, and the three names HIP does not have (`cudaDevAttrMaxPersistingL2CacheSize`, `cudaDevAttrMaxAccessPolicyWindowSize`, `cudaLimitPersistingL2CacheSize`) to ids HIP answers as `hipErrorInvalidValue` / `hipErrorUnsupportedLimit`. Needed because `mmvq_il_parity.cpp` gates its `--persist`/`--ab` L2 code with `#if !defined(STRATA_HIP_GFX906)` (lines 88, 492), which a wave32 AMD build does not define. |
| 1392-1402 | `if(STRATA_ENABLE_CUDA)` keeps `mmvq_il_parity_89table` (sm_89) and `mmvq_il_parity_pf` CUDA-only — on HIP every variant runs `native_mmvq_il_kernel`, so a `_pf` test there measures nothing. |
| 1403-1409 | New `if(STRATA_ENABLE_CUDA)` around `dequant_wide_parity` (content unchanged; it was inside the old IL block). |

## 2. CUDA behaviour is unchanged — checked, not assumed

`nvcc` is not on this machine, so the CUDA target was **not built** here. What was checked mechanically: the file
before (`git show HEAD:…`) and after were preprocessed with the *CUDA* branch taken (plain `clang++ -E -x c++`, no
`--offload-arch`, so `__HIPCC__` is undefined; `cuda-pre/old.i`, `cuda-pre/new.i`, 80 121 / 80 131 lines after
stripping `#line` markers). **The diff has 0 non-whitespace lines** — every guard I added is true for CUDA
(`STRATA_IL_PF == 1`) and every guard I removed was true for CUDA, so a CUDA compilation sees the same tokens.
The report claims parity of CUDA *behaviour* on that evidence, not on a CUDA build.

## 3. Build (only the bench target)

Plain shells here are capped at 4 GiB, so every compile ran inside a user scope
(`systemd-run --user --collect --scope --property=MemoryMax=24G --setenv=PATH=…/opt/rocm/bin … --working-directory=/home/soren/Strata-integ -- …`).

1. First attempt: `cmake --build build-hip --target mmvq_il_parity -j 20` → `make: *** No rule to make target
   'mmvq_il_parity'`: a new target needs a reconfigure first. Ran `cmake -S . -B build-hip` in the same kind of
   scope (`configure.log`), then the build.
2. First real failure: `launch_il_one` named `native_mmvq_il_pf_kernel`, which the HIP branch does not compile
   (`error: use of undeclared identifier 'native_mmvq_il_pf_kernel'`, 20 errors, `build-target2.log`). Fixed by
   guarding the dispatch at 2266/2277 (section 1).
3. Second failure: the first version of the L2 shim used `target_compile_definitions` with plain integers →
   `error: invalid conversion from 'int' to 'hipDeviceAttribute_t' [-fpermissive]` (4 sites); with casts inline the
   shell mangled the `-D…((type)-1)` strings. Fixed by the generated-header + `-include` form (section 1).
4. Result: `build-target4.log` → `[100%] Built target mmvq_il_parity`, 0 errors,
   `build-hip/mmvq_il_parity` (22 652 896 bytes). `native_mmvq.cu` compiled with **408 warnings per pass**
   (gfx1200, gfx1201, host) — 1 113 `sign-compare` and 102 `nodiscard hipError_t` per pass are pre-existing in
   this file; 3 new ones appear per pass on HIP only: `unused variable 'HAS_PF'` (1929), `unused parameter 'pf'`
   (`launch_il_one`), and `unused function 'q2_0_warp_rows_on'` (which sits at `#if` depth 0 in HEAD too, so it is
   not from this change — checked in `cuda-pre/old.cu`).
5. Only `mmvq_il_parity` was built. No ctest run, no engine, no model.

## 4. What HIP reports for this card (measured, `cc_probe.cu` → `cc-probe.txt`)

`device 0: AMD Radeon AI PRO R9700; major 12 minor 0 -> il_cc (major*10+minor) = 120; multiprocessor count 32,
warp size 32, L2 8 388 608 bytes`.

* **The card answers cc 120 — the same key an RTX 5070 answers.** `il_cc()`'s "12.0 with ≥ 80 SMs → 1201" rule
  (`native_mmvq.cu:3881-3885`) cannot fire on HIP: `hipDeviceAttributeMultiprocessorCount` reports 32 here, not 64.
* Consequence for 2-4 columns and 1 column: 120 is in **neither** `kIlArch` (3775-3786) **nor** `kIl1Arch`
  (3754-3759), so `native_mmvq_il_rows_for(120, …)` falls back to `kIlRowsShared` for 2-4 columns and to 0 for one
  column — the "shared table on HIP" the job asked for, with no hand-authored gfx1201 cells.
* Consequence for 5-8 columns: 120 **is** in `kIl58Arch` (3727-3731), so a HIP engine call at 5-8 columns now takes
  `kIlRows58_120` — cells measured on a 110-SM RTX PRO 5000 Blackwell. `il_arch_ok()` (3906-3919) passes on HIP
  (major 12 ≥ 8), so `native_mmvq_il_supported` can return true.
* `il_pf_table` returns 0 for every cc except 80, so nothing PF-related is selected.

## 5. Parity verdict

The documented command, GPU to ourselves (`strata.service` was already inactive; checked with
`systemctl is-active` → `inactive`, read-only):

```
systemd-run --user --collect --scope --property=MemoryMax=8G \
  --setenv=PATH=/usr/sbin:/usr/bin:/bin:/opt/rocm/bin --setenv=ROCM_PATH=/opt/rocm \
  --setenv=HIP_PATH=/opt/rocm --setenv=STRATA_IQ_STAGE_TABLES=1 \
  --working-directory=/home/soren/Strata-integ \
  -- /home/soren/Strata-integ/build-hip/mmvq_il_parity --bench --emit-table
```

Run 1 `bench-full.log`, run 2 `bench-full-run2.log`, each ~39 s, both ending with

```
mmvq_il_parity: OK (every case, T 1-8, rows 1/2/4 and the table, bitwise)
```

0 `FAIL` lines in either log. Coverage as the tool defines it: all 32 cases (30 timed + 2 parity-only odd-row
shapes), T = 1…8 columns, rows a warp 0 (the card's table), 1, 2, 4 — each compared bit-for-bit against
`native_mmvq`'s multi-column kernels on the same q8_1 bytes, plus `check_tables()`'s pure table checks for
sm_60/70/80/86/89/120. 240 `BENCH` lines per run. Nothing about the parity checks was weakened.

`STRATA_IQ_STAGE_TABLES` was set as instructed and is a **no-op for this target**: the only reader in the tree is
`src/kernels/cuda/iq_kernels.cu:3419-3426`, which gates the IQ3_S/IQ3_XXS (types 18/21) codebook staging inside the
grouped-expert kernels (`iq_kernels.cu:4213`, `:4265`), and the IL bench's types 23/12/13/14/11 never route to
`iq_mmvq` (`native_mmvq.cu:3564-3566` sends 16/17/18/21/22/29 there, the five IL types to their own kernels). This
run measures nothing about that variable's effect; the production unit keeps it for its own IQ3 layers.

## 6. Measured gains (cold arm, 192 MB rotation, `BENCH` lines; run 1, run 2 in brackets)

Best of the three rows counts per call, ratio to `native_mmvq`'s multi-column kernel:

| shape | T=1 | T=2 | T=3 | T=4 | T=5 | T=6 | T=7 | T=8 |
|---|---|---|---|---|---|---|---|---|
| Q5_K head 2560x248320 | -15.5 [-16.4] | -27.6 [-30.5] | -48.1 [-48.1] | **-48.7 [-48.8]** | -40.8 [-40.4] | -35.9 [-35.6] | -38.9 [-39.0] | -36.7 [-36.5] |
| IQ4_XS 2560x10240 (qkv) | +4.8 [+3.7] | -5.4 [-6.2] | -22.0 [-23.4] | -23.6 [-23.0] | -27.3 [-27.2] | -34.8 [-33.8] | -35.4 [-35.3] | -39.0 [-38.7] |
| Q3_K 2560x10240 | -21.3 [-23.7] | -24.6 [-27.2] | -27.7 [-27.5] | -28.3 [-28.5] | -30.7 [-30.8] | -36.6 [-36.3] | -40.0 [-40.1] | -41.5 [-41.4] |
| Q6_K 2560x12288 | +0.7 [+0.2] | -8.5 [-8.7] | -17.6 [-18.6] | -25.6 [-25.6] | -33.1 [-33.3] | -34.8 [-34.5] | -36.5 [-36.1] | -32.4 [-32.3] |
| Q4_K 6144x2560 (out proj) | +3.6 [+3.0] | -2.2 [-2.7] | -0.5 [-1.8] | -6.0 [-7.0] | -9.4 [-9.3] | -6.3 [-6.0] | -14.2 [-14.2] | -17.0 [-17.9] |

Per type and token count, the emitted cell's rule (the worst measured shape of each n_out class; class = the five
`n_out` buckets the table uses: `<2048, 4096, 8192, 12288, more`; `-` = no shape in that class):

```
IQ4_XS  T=1: rows<2048 1:-9.4%  4096 1:+0.5%  8192 1:+4.0%  12288 2:+4.8%  more 2:+2.9%
IQ4_XS  T=2: rows<2048 1:-2.0%  4096 1:-4.5%  8192 1:-7.0%  12288 1:-5.4%  more 1:-17.9%
IQ4_XS  T=3: rows<2048 1:-13.8%  4096 1:-11.7%  8192 1:-16.9%  12288 1:-22.0%  more 1:-18.7%
IQ4_XS  T=4: rows<2048 1:-17.7%  4096 1:-15.8%  8192 1:-22.1%  12288 1:-23.6%  more 1:-22.1%
IQ4_XS  T=5: rows<2048 1:-19.4%  4096 2:-22.2%  8192 1:-24.8%  12288 4:-27.3%  more 4:-27.5%
IQ4_XS  T=6: rows<2048 1:-21.9%  4096 2:-29.9%  8192 2:-33.5%  12288 4:-34.8%  more 4:-34.1%
IQ4_XS  T=7: rows<2048 1:-26.9%  4096 1:-25.5%  8192 1:-34.7%  12288 4:-35.4%  more 4:-36.0%
IQ4_XS  T=8: rows<2048 1:-21.3%  4096 1:-34.2%  8192 1:-38.5%  12288 1:-39.0%  more 1:-38.4%
Q4_K    T=1: rows<2048 1:-4.3%  4096 2:+3.6%  8192 2:-4.2%  12288 2:-3.9%  more   -
Q4_K    T=2: rows<2048 1:-13.2%  4096 2:-2.2%  8192 2:-18.9%  12288 2:-22.7%  more   -
Q4_K    T=3: rows<2048 1:-18.0%  4096 1:-0.5%  8192 1:-32.7%  12288 2:-33.4%  more   -
Q4_K    T=4: rows<2048 1:-22.1%  4096 2:-6.0%  8192 2:-36.8%  12288 4:-39.2%  more   -
Q4_K    T=5: rows<2048 1:-24.0%  4096 4:-9.4%  8192 4:-38.1%  12288 2:-38.7%  more   -
Q4_K    T=6: rows<2048 1:-19.3%  4096 4:-6.3%  8192 4:-36.5%  12288 2:-37.8%  more   -
Q4_K    T=7: rows<2048 1:-31.2%  4096 4:-14.2%  8192 4:-40.5%  12288 4:-40.5%  more   -
Q4_K    T=8: rows<2048 1:-37.0%  4096 4:-17.0%  8192 1:-41.9%  12288 4:-40.2%  more   -
Q5_K    T=1: rows<2048 1:-4.2%  4096 2:+3.6%  8192 2:+0.0%  12288 2:-2.8%  more 4:+0.0%
Q5_K    T=2: rows<2048 1:-17.9%  4096 1:+2.4%  8192 2:-12.5%  12288 2:-18.3%  more 2:-18.7%
Q5_K    T=3: rows<2048 1:-17.5%  4096 4:+4.7%  8192 1:-31.4%  12288 1:-31.0%  more 4:-31.5%
Q5_K    T=4: rows<2048 1:-22.4%  4096 1:-4.3%  8192 1:-33.0%  12288 4:-35.5%  more 4:-36.9%
Q5_K    T=5: rows<2048 1:-24.4%  4096 4:-10.4%  8192 2:-37.6%  12288 4:-38.0%  more 4:-38.2%
Q5_K    T=6: rows<2048 1:-17.6%  4096 4:+1.6%  8192 4:-34.6%  12288 4:-36.3%  more 2:-35.9%
Q5_K    T=7: rows<2048 1:-35.0%  4096 1:-15.2%  8192 2:-40.4%  12288 2:-40.9%  more 2:-38.9%
Q5_K    T=8: rows<2048 1:-36.9%  4096 1:-15.9%  8192 1:-39.3%  12288 1:-38.6%  more 4:-36.7%
Q6_K    T=1: rows<2048 2:-8.9%  4096 2:+4.2%  8192 2:-0.4%  12288 2:-1.1%  more 1:+0.7%
Q6_K    T=2: rows<2048 1:+0.0%  4096 2:-4.9%  8192 2:-11.7%  12288 1:-8.2%  more 1:-8.5%
Q6_K    T=3: rows<2048 1:-7.0%  4096 2:-4.0%  8192 1:-14.4%  12288 1:-16.0%  more 2:-17.6%
Q6_K    T=4: rows<2048 1:-25.7%  4096 1:+3.7%  8192 1:-20.8%  12288 1:-24.9%  more 1:-25.6%
Q6_K    T=5: rows<2048 2:-18.6%  4096 4:-13.7%  8192 4:-27.4%  12288 4:-32.2%  more 4:-33.1%
Q6_K    T=6: rows<2048 1:-26.6%  4096 4:-12.2%  8192 4:-30.7%  12288 4:-33.8%  more 4:-34.8%
Q6_K    T=7: rows<2048 1:-21.3%  4096 4:-21.3%  8192 4:-33.0%  12288 4:-35.6%  more 4:-36.5%
Q6_K    T=8: rows<2048 1:-29.0%  4096 1:-16.5%  8192 1:-30.0%  12288 1:-31.8%  more 1:-32.4%
Q3_K    T=1: rows<2048 1:+4.0%  4096 2:+12.2%  8192 2:-15.3%  12288 2:-21.3%  more 2:-25.6%
Q3_K    T=2: rows<2048 1:-15.9%  4096 2:+11.9%  8192 2:-17.9%  12288 2:-24.6%  more 2:-22.9%
Q3_K    T=3: rows<2048 1:-20.0%  4096 4:+8.5%  8192 2:-21.7%  12288 4:-27.7%  more 4:-29.7%
Q3_K    T=4: rows<2048 1:-20.8%  4096 2:+5.1%  8192 1:-23.7%  12288 4:-28.3%  more 4:-29.8%
Q3_K    T=5: rows<2048 1:-25.0%  4096 1:+13.8%  8192 4:-28.4%  12288 2:-30.7%  more 4:-31.6%
Q3_K    T=6: rows<2048 1:-22.4%  4096 1:-3.2%  8192 4:-34.1%  12288 4:-36.6%  more 4:-36.9%
Q3_K    T=7: rows<2048 1:-28.6%  4096 4:-7.8%  8192 4:-38.3%  12288 2:-40.0%  more 2:-41.0%
Q3_K    T=8: rows<2048 1:-30.5%  4096 4:-15.2%  8192 4:-39.6%  12288 4:-41.5%  more 4:-42.1%
```
(`summarize.py`, produced from `bench-full.log`. The 4096 class = the 6144x2560 output projections: this is where
the IL path is weakest at low token counts, and it is the class where the emitted cells are mostly 0.)

## 7. Pasteable `--emit-table` block (run 1, verbatim, `bench-full.log`)

```
// measured on this card by mmvq_il_parity --bench --emit-table
    {23, {{0, 1, 1, 1, 1}, {1, 1, 1, 1, 2}, {1, 1, 1, 1, 1}}},   // IQ4_XS
    {12, {{1, 0, 2, 2, 2}, {1, 0, 1, 2, 2}, {1, 2, 2, 4, 4}}},   // Q4_K
    {13, {{1, 0, 2, 2, 2}, {1, 0, 1, 1, 4}, {1, 1, 1, 4, 4}}},   // Q5_K
    {14, {{0, 2, 2, 1, 1}, {1, 2, 1, 1, 2}, {1, 0, 1, 1, 1}}},   // Q6_K
    {11, {{1, 0, 2, 2, 2}, {1, 0, 2, 4, 4}, {1, 0, 1, 4, 4}}},   // Q3_K
// 5-8 columns (kIlRows58_<cc>; rows 4 runs as 2 there):
    {23, {{1, 2, 1, 4, 4}, {1, 2, 2, 4, 4}, {1, 1, 1, 4, 4}, {1, 1, 1, 1, 1}}},   // IQ4_XS
    {12, {{1, 4, 4, 2, 2}, {1, 4, 4, 2, 2}, {1, 4, 4, 4, 4}, {1, 4, 1, 4, 4}}},   // Q4_K
    {13, {{1, 4, 4, 4, 4}, {1, 0, 4, 4, 2}, {1, 1, 2, 2, 2}, {1, 1, 1, 1, 4}}},   // Q5_K
    {14, {{4, 4, 4, 4, 4}, {1, 4, 4, 4, 4}, {1, 4, 4, 4, 4}, {1, 1, 1, 1, 1}}},   // Q6_K
    {11, {{1, 0, 4, 2, 4}, {1, 1, 4, 4, 4}, {1, 4, 4, 2, 2}, {1, 4, 4, 4, 4}}},   // Q3_K
// one column (kIl1RowsShared):
    {23, {1, 0, 0, 0, 0}},   // IQ4_XS
    {12, {1, 0, 2, 2, 2}},   // Q4_K
    {13, {1, 0, 0, 0, 0}},   // Q5_K
    {14, {2, 0, 0, 0, 0}},   // Q6_K
    {11, {0, 0, 2, 2, 2}},   // Q3_K
```

## 8. Are gfx1201 cells non-zero — i.e. is the IL win real on this card?

Yes, and it is not marginal. Non-zero cells in run 1's block: **2-4 columns 65 of 75**, **5-8 columns 98 of 100**,
**one column 10 of 25**. Gains grow with the token count as the mechanism says they should (weights read once per
M rows): the worst shape of a class is typically -13% to -24% at 3-4 tokens and -27% to -42% at 6-8 tokens, and the
LM head -48.7% at 4 tokens. `docs/MMVQ_IL_TABLE.md:9` records -45% at 5 tokens on a 110-SM Blackwell head
(899 → 491 µs); our 64-CU card takes **-40.8%** on its own head at 5 tokens (1967.6 → 1164.2 µs), which is the
expected shape of the result (same mechanism, fewer CUs).

Against what this card runs today (measured by comparing the block to the shipped tables):

* 2-4 columns, where HIP answers 120 → `kIlRowsShared`: **21 of 60** cells agree; 18 cells the shared table leaves
  at 0 measure non-zero on this card, 2 cells the shared table turns on measure 0 here.
* 5-8 columns, where HIP answers 120 → `kIlRows58_120` (Blackwell's cells): **29 of 100** agree; 20 of that
  table's zeros measure non-zero on this card, 2 of its non-zeros measure 0.

So a card-specific gfx1201 table would turn the IL path on in many more cells than the card gets today, and it
would also turn off a handful the shared/Blackwell tables turn on.

## 9. Reproducibility (two runs of the same command, `compare_cells.py`)

* 195 of 215 numeric cells identical: 2-4 table **75/80 (93.8%)**, 5-8 table **91/105 (86.7%)**, one column
  **29/30 (96.7%)**.
* Timing noise: over the 240 matched `BENCH` shapes the best-ratio differs by a **median 0.36 pp**; `native_mmvq`'s
  own times agree within a median 0.7%. The 5 largest disagreements (10-14 pp) are all small shapes
  (n_out 512/640 at T ≤ 5, calls of 5-9 µs), where per-call overhead dominates; the same shapes are where the
  emitted cells flip.
* Only 2 of 215 cells flip between "on" and "off" across the two runs (Q3_K 5-8 cell 7: 1 vs 0; Q5_K one-column
  cell 4: 0 vs 2). Every other flip is between neighbouring row counts (1↔2, 2↔4) and the two arms are within a
  few percent of each other.

## 10. Caveats, in the order someone acting on this would hit them

1. **Do not paste this block as `kIlRows_120`.** On CUDA, cc 120 means "sm_120 with fewer than 80 SMs" (RTX 5070
   and smaller, docs' table row 1); our HIP card answers the same number. Adding `{120, kIlRows_<this>, …}` to
   `kIlArch` would change the table those NVIDIA cards get. A gfx1201 table needs an AMD-distinct key first, which
   means a change in `il_cc()` (`native_mmvq.cu:3876-3895`) — e.g. a gfx-arch- or vendor-derived key (the file
   already has `gfx_arch.hpp` includes) — and `kIl58Arch`/`kIl1Arch` keys to match. **I did not change `il_cc`;
   that is the one repo change outside these two files that this job needs from the parent.**
2. **The engine's HIP default path changes with this un-gate** (not just the bench): `verify.cpp:1851, 1893, 2130,
   2820` and `dflash_runtime.cpp:1366` gate on `native_mmvq_il_supported`, and `g_mmvq_il()` (`verify.cpp:187`) is
   on unless `STRATA_MMVQ_IL=0`. With this patch a gfx1201 engine call runs the IL kernels per `kIlRowsShared`
   (2-4 columns) and per `kIlRows58_120` (5-8 columns). Output bits are unchanged — that is exactly what the
   parity line proves for every shape/T/rows choice — and `native_q8_1_il_bytes` now returns the layout's real size
   (smaller than the old stub's `native_q8_1_bytes * 2`, and the callers size their buffers with the same function:
   `verify.cpp:1185`, `dflash_runtime.cpp:586`). **No end-to-end decode A/B was run here** (no engine, no model,
   per the job's limits), so the GEMV numbers above are the only effect sizes I can quote.
3. **Register pressure, measured**: compiling this file for gfx1201 with `-Rpass-analysis=kernel-resource-usage`
   (`resource.log`) gives 50 `native_mmvq_il_kernel` instantiations, **VGPRs max 188, median 107, 0 spills,
   0 scratch bytes/lane**; reported occupancy 8-16 waves/SIMD (17 of the 50 exceed 128 VGPRs → 8-10 waves/SIMD).
   `native_mmvq_multi_kernel` (the plain path) is median 59, max 120. The IL kernels trade registers for the
   weight-traffic saving, as the design intends, and nothing spills on this arch.
4. **P1/P2 (`STRATA_MMVQ_IL_PF`, `kIlPf_80`) stay CUDA-only** by my guard (section 1). `native_mmvq_il_pf(-1)`
   (the bench's default) selects today's kernel on HIP, and `il_pf_table` answers 0 outside cc 80, so nothing was
   hidden by the guard for this measurement. `mmvq_il_parity_pf` remains a CUDA-only test. A HIP PF variant needs
   HIP forms of `madw` (`base + k*stride` as one wide multiply) and of the `ld.global.nc` loads —
   `__builtin_nontemporal_load`/plain loads are the obvious candidates (precedent: `native_mmvq.cu:2507-2511`
   uses `__builtin_nontemporal_load` for `STRATA_Q8P_LOAD` on HIP).
5. **The `--persist` / `--ab` arms are NVIDIA-only**: `mmvq_il_parity.cpp:87-102, 492-505` gate on
   `!STRATA_HIP_GFX906`, which a gfx1201 build does not define, and HIP has no persisting-L2 limit
   (`hipLimit_t`, `/opt/rocm/include/hip/hip_runtime_api.h:872-878`) and no access-policy attributes. My CMake
   shim makes the file compile (with ids HIP rejects, verified in `cc-probe.txt`); the honest fix is a
   `#if defined(__HIPCC__)` arm in `src/kernels/mmvq_il_parity.cpp` — a file this job was not allowed to edit,
   so I am reporting it rather than changing it. Consequence: on this build `--persist`/`--ab` are meaningless
   (they report 0 MiB granted / an unsupported limit); `--bench`, `--emit-table`, `--hot`, `--gap`, `--rounds`
   and `MMVQ_IL_ONLY`/`MMVQ_IL_TMIN`/`MMVQ_IL_TMAX` all work.
6. **Small shapes are noisy** (caveat 9): cells built from `n_out` ≤ 640 at T ≤ 5 flip between runs; treat them as
   0 unless a third run agrees. The docs' own Ada entry (`MMVQ_IL_TABLE.md:18`) reports the same phenomenon for
   the 5-8 table (changed in 10-14 of 20 entries between runs of one card); ours is milder
   (86.7% agreement) but the same kind of thing.
7. **IQ3_S / IQ3_XXS are not measured by this tool** (types 21/18 are not in the IL family; `native_mmvq` sends
   them to `iq_mmvq`, `native_mmvq.cu:3564-3566`, and `native_mmvq_il_supported` admits only
   {23,12,13,14,11} at 3940-3943). Q3_K (11) is measured and wins consistently (-15% to -42%), which supports
   building an IL variant for IQ3 as Job A's F1 proposes; this run does not by itself decide it.
8. **One `IlQ3K` note**: `HAS_PF = false` (1929) is now an unused member on HIP (warning, section 3). Harmless.

## 11. Artifacts in this directory

`bench-full.log` (run 1, the pasteable source), `bench-full-run2.log` (reproducibility), `summary.txt`
(`summarize.py`), `cc-probe.txt` + `cc_probe.cu` (what HIP reports, and HIP's answers for the missing APIs),
`build-target{2,3,4}.log`, `configure.log`, `resource.log` + `native_mmvq_resource.o` (register/spill report),
`jobC-edits.patch` (the whole repo diff), `cuda-pre/` (CUDA-branch preprocessed old/new for the
behaviour-unchanged check), `compare_cells.py`, `summarize.py`, `run_bench.sh`.

## 12. Not run (so the parent does not mistake silence for evidence)

No ctest run, no CUDA build (no `nvcc` here), no engine, no model, no decode A/B, no `--persist`/`--ab`/`--hot`/
`--gap` arms, no `--pf` run on HIP, no IlIQ3S/IlIQ3XXS kernels.

## 13. Post-run production restoration (parent verification)

After both GPU runs completed, the parent restarted `strata.service` with `STRATA_IQ_STAGE_TABLES=1` in its unit.
The service is active; its main process environment contains `STRATA_IQ_STAGE_TABLES=1`, authenticated
`GET /v1/models` returned HTTP 200, and `NRestarts=0`. This confirms the flag is active in production, not that its
IQ3 staging path improves performance: no model-level A/B was run.
