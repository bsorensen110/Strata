# RDNA4 / HIP port candidates from the CUDA path

Date: 2026-11-11. Derived by scanning `fork-main` (`6b0550082`) source, not from docs or memory:
`git grep` for CUDA-only annotations, `STRATA_USE_HIP` / `__HIPCC__` call-site gates, and CMake
backend exclusions. Line references are against that commit and will drift.

Scope note: this file is a gap inventory and triage, **not** a claim that any item below has been
ported, benchmarked, or validated on gfx1201. Items marked "measured here" reference our own A/B
history on an R9700 (gfx1201); everything else is unverified on AMD and must get parity + A/B
gates before anything ships.

## Tier 1 — gate-only or small-code ports (no missing hardware; never validated on HIP)

- **`--vram-elastic` segmented expert cache** (`src/core/expert_cache.cpp:400`: "is CUDA-only for
  now"). Built on the driver VMM API (`cuMem*`). The HIP branch of `src/core/vmm.cpp` is live on
  this tree — `hipMemCreate` / `hipMemMap` / `hipMemAddressReserve` probe and pass — so this is an
  un-port, not a hardware gap. Highest config fit: production runs `--kv-resident` plus a large
  expert cache. The existing HIP CTest exclusion on `expert_cache_segmented_test` (error-string
  mismatch, "CUDA-only for now" vs "virtual memory management") is part of the same work.
- **Per-window commit-graph elision** (`src/core/verify.cpp:418`, `STRATA_ONE_TOKEN_COMMIT`).
  CUDA skips the commit graph after a window; HIP replays it every window. `hipGraph*` is fully
  mapped in `include/strata/platform/hip_compat/strata_hip.h`, so this is call-site gate widening
  plus validation.
- **HostGraphLaunch from a helper thread** (`src/core/verify.cpp:2`, CUDA builds only). The launch
  thread serves doorbells while a helper posts the graph. Host-side plumbing over already-mapped
  graph APIs.
- **MTP shared-expert side-stream branch** (`src/core/mtp.cpp:1236`, `STRATA_MTP_SHARED_BRANCH`
  forced off on HIP — "keeps one stream"). Streams and events exist in HIP; looks like a
  validation gap. Worth a matched A/B (the env var already exists to force the CUDA side to
  single-stream for a fair comparison).
- **Stream/graph priority** (`STRATA_GRAPH_PRIO`, `src/core/verify.cpp:234` "every HIP caller is
  guarded off"; `STRATA_MTP_PRIORITY`). HIP exposes `hipStreamCreateWithPriority`; portability of
  the *effect* (does the ROCm scheduler actually prioritize?) needs an A/B, not just a compile.
- **Multi-block argmax / top-k head ops** (`STRATA_MULTI_BLOCK_ARGMAX`,
  `src/kernels/cuda/verify_kernels.cu:1089+`). Newer than the option docs claim: the gate now
  contains gfx12 arch-detection code (gfx12 is "the only HIP target #3264 was run on"), but docs
  still say the HIP build never uses them. First step is just confirming whether the path engages
  in our current build; if it is dead code, widening it is near-free.

## Tier 2 — real kernel ports, no missing hardware

- **Native MMVQ P1/P2 prefetch variants** (`native_mmvq_il_pf_kernel`,
  `src/kernels/cuda/native_mmvq.cu:655,2258`). CUDA-only because of PTX `mad.wide.u32` and
  `ld.global.nc`; both have HIP-able forms (`__ldg`, widened multiply). Caveat: we A/B-dropped
  `STRATA_HIP_MMVQ_IL` on this box already — keep expectations low and re-measure.
- **IQ staged-table kernels** (`STRATA_IQ_STAGE_GRID`, `src/kernels/cuda/iq_kernels.cu:3481`:
  "the AMD builds keep the old kernels").
- **FP16 tensor-core IQ prefill GEMM** (`include/strata/kernels/iq_kernels.hpp:140`, CUDA only).
  RDNA4 WMMA FP16 is real (this fork ships WMMA kernels on gfx12), so this is translation, not a
  wishlist item.
- **Two-pass QSA select scorer** (`src/kernels/cuda/qsa_select.cu:4043`; parity tool
  `qsa_two_pass_parity.cpp` skips on AMD). The gfx12 WMMA scorer has different arithmetic, so a
  port lands as *differs-in-bits* and must stay behind an opt-in gate with measured gains.
- **Fused int8 prefill experts** (`src/prefill/prefill.cpp:152`) and the **pointer-table prompt
  MMQ** (`STRATA_MMQ_TAB`, `CMakeLists.txt:177`, "CUDA; ~37 MiB VRAM per GPU"). HIP keeps the
  unfused MMQ / FP16 paths. Caveat: `STRATA_DENSE_MMQ` was A/B-dropped from production here.
- **QSA decode-attention alternate-partials variant**
  (`src/kernels/cuda/qsa_decode_attn.cu:1052`, opt-in, CUDA only).
- **GDN tensor-core prompt kernel** (`src/split/src/gdn_tc_prompt.cuh`, `FNX_PFX_GDN_TC`, sm_80+).
  FP16 wmma-based, so WMMA-portable in principle; HIP keeps `k_gdn_ch_prep` / `k_gdn_ch_rec`.

## Tier 3 — genuine hardware / driver gaps (do not schedule)

- **INT8 tensor-core prompt attention** (`FNX_PFX_ATTN_I8`, `src/split/src/kernels.cu:1449`).
  RDNA4 has no INT8 tensor cores; HIP keeps the f16 kernel. No port exists.
- **PDL (programmatic dependent launch, sm_90+)**. No RDNA equivalent. Persistent-kernel tricks
  are the workaround; we tested one on gfx1201 (upstream PR #2226, persistent HC read): parity
  clean but ~neutral on the R9700, so don't expect miracles from that lane.
- **`cp.async`-based kernels** (e.g. upstream PR #413's GDN recurrence kernel, gated
  `#if !defined(__HIPCC__)`). sm_80+ construct with no gfx12 analogue.
- **NVFP4 MMQ instances** (`src/prefill/moe_mmq.cu:556`). No RDNA4 NVFP4 datapath; inert on an
  IQ3_S-type pack anyway.
- **EXL3 v1.1.8 vendor kernels** (`src/dsv41/vendor/exl3.cu:1274`), shared-memory carveout
  percentage control (`hip_compat/strata_hip.h:68`), spin-wait context scheduling
  (`cudaInitDevice`, `src/core/remote_experts.cpp:100`), `cuobjdump`-based build-time checks
  (`CMakeLists.txt:587`). Driver/vendor/tooling gaps, low value.
- **`STRATA_PCIE_AUTO_CANON`** (canonical-pack PcieAuto) — CUDA-only today; a tuning-path
  extension rather than a hardware gap, listed here for completeness.

## HIP / RDNA4 (R9700) opt-ins to benchmark and validate

Date: 2026-11-12. Switches to run through parity selftests + matched A/B benchmarks on the gfx1201
(R9700) box. Everything below stays opt-in (or default-off on gfx12) until measured on hardware;
items that alter summation order or rounding (`PF_WMMA`, `PF_F16`, `ATTN_DAILY`, `GR_V3`, `MMQ_TAB`)
need a KL / greedy check alongside tok/s.

### Newly ported Tier-2 & decode items on `fork-main`

- **MMVQ interleaved prefetch variants (`STRATA_MMVQ_IL_PF=1..6`)** (`src/kernels/cuda/native_mmvq.cu:2275,2487`).
  P1/P2 wave32 HIP port (`8c03a43ed`); sweep `=2..6` on the production quant (e.g. `IQ4_XS`,
  `Q4_K`, `Q6_K`) before adding a `1202` table entry (`il_pf_table`). Note: gfx12 already turns on
  `STRATA_QIL` by default, and `b5d62e0fe` folds the `q8_il` interleave write into
  `fused_gr_read_multi` (bit-exact with `native_q8_1_interleave`).
- **Decode attention `pre75` / Daily merge (`STRATA_ATTN_DAILY=1`, `STRATA_ATTN_PRE75=1`)**
  (`src/kernels/cuda/qsa_decode_attn.cu:1065,1094`). Widened to wave32 HIP (`8c03a43ed`); test with
  `STRATA_ATTN_MERGE_V2=0/1`.
- **TerQuant sub-byte dot (`STRATA_TQ_SUB8=1`)** (`src/kernels/cuda/iq_kernels.cu:3485`).
  Opt-in on HIP (`1376e0d28`).
- **Grouped FP16 WMMA prompt experts (`STRATA_PF_WMMA=1`)** (`include/strata/kernels/iq_kernels.hpp:138`,
  `src/kernels/pf_wmma_parity.cpp`). RDNA4 WMMA tile port (`1376e0d28`); verify `pf_wmma_parity` first.
- **FP16 tensor-core prompt experts (`STRATA_PF_F16=1`) & Q2_0 fused MoE decode** (`e7a34c39f`).
  Test `STRATA_PF_F16=1` against the default dequant+hipBLASLt / `STRATA_MMQ_KQ=1` prompt path, and
  benchmark `Q2_0` packs on `moe_fused`.
- **Pointer-table prompt MMQ (`-DSTRATA_MMQ_TAB=ON` + `STRATA_MMQ_DIRECT=1`)** (`CMakeLists.txt:177`,
  `include/strata/prefill/moe_mmq.hpp:147`, `e7a34c39f`). Build-time opt-in (~37 MiB VRAM/GPU).
- **Multi-slot GDN conv batching (`gdn_conv_l2_groups`)** (`8cd4f9779`, enabled under default
  `STRATA_BATCH_GDN_GROUPS`). Benchmark parallel-slot decode (`c >= 2`) for bit-identical parity
  and launch-overhead savings.
- **Recent default-on decode launch reductions (benchmark vs pre-patch baseline on R9700):**
  - `gdn_step_norm_multi` / `gdn_step_norm_groups` `q8_il` fusion (`76d3e8319`): activates when
    `STRATA_QFUSE=1` and `2 <= n <= 8` (`native_mmvq_il_supported` on wout).
  - Shared-expert `native_mmvq_il` & `native_swiglu_quantize_q8_1_il` (`0dd018de2`): eliminates
    separate `q8_1` interleave launches in `shared_expert_multi` (`2 <= n <= 8`).
  - Paired K/V FWHT (`fwht256_pair_inplace_cuda`) & multi-slot KV append (`kv_append*_groups`)
    (`0b8b0d91f`): halving FWHT + multi-slot KV append launches for Q4/Q8/FP16/K8V4 (`c >= 2`,
    gated by `STRATA_BATCH_ATTN_ROWS` and `STRATA_NO_BATCH_KV_STEP`).
- **Long-context single-read top-k (`STRATA_TOPK_SAMPLE=1`)** (`src/kernels/cuda/qsa_select.cu:3034`,
  `90926ab93`). Opt-in on HIP (`block_topk_sample_kernel` + `block_topk_wide_kernel<true>` past
  28,672 blocks, `nq <= 4096`); verify `qsa_select` parity tests and long-context prompt/decode A/B.

### Existing opt-ins not enabled in `gfx12` defaults (worth A/B testing on R9700)

- **Decode fusion & launch-reduction (expected bit-identical unless noted):**
  - `STRATA_NORM_ROPE=1` (`src/kernels/cuda/native_rope.cu:180`): merges RMSNorm + RoPE (and skips
    Q `copy_rows_strided`). Requires `rope_parity` check 6 pass on gfx1201 first.
  - `STRATA_LFUSE=1` (`src/core/verify.cpp:116`; sub-gates `STRATA_LFUSE_GATE`, `STRATA_LFUSE_PAIR`):
    fuses shared-expert gate into router GEMV, gate/up SwiGLU pair (Q8_0), and `kv_append_q8` steps.
  - `STRATA_QFUSE=1` (`src/core/verify.cpp:299`): adds GDN output norm `q8_1` image fusion on top of
    default-on `STRATA_QFUSE_HC` (test on `gdn_step_norm_multi` / `gdn_step_norm_groups`).
  - `STRATA_MMVQ_TSUM=1` (`src/kernels/cuda/native_mmvq.cu:1255`): transposed warp butterfly for
    multi-column MMVQ (Note: inspect interaction with `native_mmvq_pair_swiglu` if `LFUSE` is on).
  - `STRATA_ONE_TOKEN_COMMIT=1` (`src/core/verify.cpp:556,4073`): opt-in on HIP (elides the
    commit graph on 1-token windows unless `STRATA_QFUSE=1` or `always_publish_` is active).
  - `STRATA_GR_V3=1` (`src/kernels/cuda/fused_gr.cu:837,3025`, + `STRATA_GR_V3_PF`): stream-split HC
    read (different summation order; verify KL + R9700 tok/s).
  - `STRATA_SIDE_MIXER=1` / `2` (`src/core/verify.cpp:266`) and `STRATA_SH_STREAM=1`
    (`include/strata/core/arch_defaults.hpp:25`): side-stream overlap (Note: R9700 has gfx1151-like
    stream-fork sensitivity; measure A/A noise carefully).
  - `STRATA_VERIFY_ASYNC_LAUNCH=1` (`src/core/verify.cpp:3`): async graph launch from helper thread.
  - Known neutral/negative on earlier R9700 A/Bs (re-test only if kernel mix shifts):
    `STRATA_FUSE_CONVAB=1` (`verify.cpp:528`, -0.3% earlier), `STRATA_FUSE_PCIE_WAIT=1`
    (`verify.cpp:517`, -3–4% at c=4 earlier).
- **Batching (`--slots >= 2`) & speculative decoding (MTP / DFlash2):**
  - `STRATA_QSA_SLOTS=1` and `STRATA_BATCH_SELF_COMMIT=1` (`src/core/verify.cpp:329,338`): bit-exact
    batch slot consolidation (`parallel_default_env` enables these on gfx1151 Windows / Linux CUDA).
  - `STRATA_DF_PLE=1` (`src/core/verify.cpp:1614`): batched PLE post-ops for draft windows.
  - `STRATA_MTP_SHARED_BRANCH=1` (`src/core/mtp.cpp:1236`): overlaps MTP shared expert on a side stream.
  - `STRATA_DF2_FORK=1` (`src/core/dflash2.cpp:745`) and `STRATA_DF2_HEAD_CHUNK=14` /
    `STRATA_DF2_HEAD_IL`: DFlash2 side-stream fork and draft LM-head tuning.

## Method notes (for reproducing or refreshing this list)

```
git grep -n -i -E "cuda[- ]only|CUDA only|not (available|supported) on hip|hip (keeps|has no|returns false)|no hip equiv" fork-main -- 'src/*' 'include/*' 'CMakeLists.txt'
git grep -n "STRATA_ENABLE_" fork-main -- CMakeLists.txt   # backend FATAL_ERROR gates: usually validation gaps
```

Triage rules (proven in this fork, see the `strata-hip-install` skill's
`references/hip-porting-gated-features.md`):

1. A gate is usually a *validation* gap, not a hardware gap — grep the kernels for CUDA-only
   constructs (`wmma`, `mma.sync`, `cublas`, `cudaFuncSetAttribute`, `cp.async`, cooperative
   launch) before writing any code.
2. Watch for the silent failure mode where the kernel compiles into the HIP build but the runtime
   *call site* sits behind a CUDA-only `#if` — the fix is widening the `#if`, not new code.
3. HIP `arch-key` routing: gfx12 reports compute capability 12.0 and can silently take NVIDIA
   sm_120 launch tables — give AMD cards their own key before trusting table-driven launches.
4. Every candidate ships only after its parity selftest passes on gfx1201 plus a matched A/B on
   the production config; differs-in-bits variants stay behind opt-in gates with measured gains.
