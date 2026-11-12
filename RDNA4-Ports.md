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
