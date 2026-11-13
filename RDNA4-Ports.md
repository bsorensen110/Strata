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

> **Measured 2026-11-12:** the switches in this section were run through matched A/B arms on an R9700 —
> see [Measured A/B on gfx1201](#measured-ab-on-gfx1201-r9700--switch-campaign-2026-11-12) at the end of
> this file for the per-switch numbers, the measured A/A noise floor, and the corrections this campaign
> makes to the items below.

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

<!-- AB-CAMPAIGN:BEGIN -> generated by bench/ports-ab-2026-11-12/render_doc.py; results are appended after each switch pair -->

## Measured A/B on gfx1201 (R9700) — switch campaign 2026-11-12

Progress: **36/36** pair rows (switches plus the A/A floor pairs). Binary `e9a2fcf8e04ada92`…
(full hash in `bench/ports-ab-2026-11-12/results/`), built from `fork-main` `edc2956c7`, whose last
engine change is `90926ab93`.

**Method.** Every arm is a fresh engine launch of the fork-main binary with the live production argv: a
scratch profile generated by `candidate_config.py` from the production profile, pack
`unsloth-ud-iq4_xs-v31`, DFlash2 `selq4` drafter, `--spec 8 --draft-conf 0.35`, `parallel: 2` (`--batch 2`),
`--expert-cache auto --expert-cache-per-layer`, `--kv-resident 32768`, `--kv-host-grow`,
`--resident-cpu-experts`. The engine argv of every arm was verified from `/proc/<pid>/cmdline` and is
identical to production's except for the scratch paths. Environment = the live production engine
environment (captured from the production engine, key stripped) with the tested switch **pinned explicitly
in both arms**: the control runs production's effective value, the treatment runs the tested value.
Baselines come from the switch's read site plus the production environment, not from this file's prose:
`STRATA_IQ_STAGE_GRID` and `STRATA_MMQ_KQ` are default-ON on gfx12, so they are tested as ablations.

**Fixed workload, identical across arms.** Cold prompt processing on a 3,947-token prompt, two 512-token
solo greedy trials (trial 1 cold, trial 2 warm), then a 2×256 concurrent burst whose aggregate is the
client's `usage.completion_tokens` over shared wall time. Temperature 0, thinking disabled, `[ports-ab …]`
prompts identical across arms. Startup telemetry matched in every arm: `resident RAM: 34.71 GiB` and
`expert cache 9408 slots`, so expert placement is not a confound between arms.

**Noise floor, measured not assumed.** Two A/A pairs ran the *identical* configuration twice
(`aa_floor_1`, `aa_floor_2`). The spread between those launches was **warm solo
±8.0%**, **2×256 burst ±2.0%**, **p4k prefill ±7.6%**,
and the 512-token greedy bytes differed between identical launches. Every verdict below is judged against
those numbers, so a switch has to clear ±8% on solo decode before it counts. The output-bits column is
therefore **not** a parity gate in this harness: bit parity needs the deterministic
`--mmap-experts --pcie-frac 1 --adapt-every 0` or CLI configuration described in the
`strata-speed-ab-testing` skill, and a differing hash only flags "needs a KL / long-greedy gate before
shipping".

**Scope.** One screen pass over every env-settable switch, plus two extra launches per arm
(rounds 2 and 3) for the nine switches whose screen sample moved more than ~5%, and a variant sweep of
`STRATA_MMVQ_IL_PF` (2, 3, 4, 6). Each pair is one control launch followed by one treatment launch
(serial, so minute-scale drift is possible); `n=` in the control column is the number of rounds averaged
and the deltas are medians across them, so a single degraded launch cannot carry a verdict (one was
flagged and excluded that way). This is not a promotion test — a candidate still needs the longer
interleaved run from the skill before it ships.

**Engagement, stated per row.** Each arm is verified to have *its* switch set in the engine process
environment (`/proc/<pid>/environ`), but the engine prints no banner for most of these switches, so for
those rows engagement is **unverified** — an absent delta may mean "no effect" or "path not reached". The
exceptions with positive evidence: `STRATA_MMQ_KQ` prints its gfx12 default banner in every arm
(`arch_defaults.cpp:311`), `STRATA_MULTI_BLOCK_ARGMAX` engages by default on gfx12
(`multi_block_head_ops()`), and the DFlash2 rows run behind the `DFlash2 drafter on` banner. Two rows
cannot engage on this workload: `STRATA_TOPK_SAMPLE` only switches its scorer past 28,672 blocks, which
needs a >30k-token prompt rather than the 3,947-token probe. The prompt-side switches
(`STRATA_PF_WMMA`, `STRATA_PF_F16`, `STRATA_DENSE_MMQ`, `STRATA_IQ_STAGE_GRID`, `STRATA_TQ_SUB8`) do have
the prompt phase in front of them, but with no counter their dispatch is an inference from the measured
prefill delta, not telemetry.

**What it means.** No switch in this file beats the production setting beyond the measured floor, so the
production configuration stands. Two things are worth acting on, and both are already the production
choice: `STRATA_SH_STREAM=1` costs 8.3% of concurrent decode and 7.8% of prompt speed in all three rounds,
and every `STRATA_MMVQ_IL_PF` variant (2, 3, 4, 6) is slower in solo decode, so the prefetch port does not
pay on this IQ4_XS pack. The mildly positive rows (`STRATA_ONE_TOKEN_COMMIT` +3.6% solo / +1.6% burst,
`STRATA_QFUSE` +4.1% / +1.6%, `STRATA_VERIFY_ASYNC_LAUNCH` +1.4% / +1.9%) are all inside the floor and
are candidates only for a much longer interleaved run, not for a config change today.

| # | switch | prod value | arm value | warm solo tok/s (control, median of rounds) | warm solo (arm, Δ) | 2×256 burst agg (arm, Δ) | p4k prefill tok/s (arm, Δ) | cold solo (arm, Δ) | engine decode warm ctl→arm | output bits (cold 512 greedy) | verdict vs A/A floor |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `STRATA_ATTN_PRE75` | 1 | 0 | 132.8 (n=1) | 132.8 → 130.8 (-1.5%) | 131.3 → 135.7 (+3.3%) | 43.1 → 43.6 (+1.2%) | 110.5 → 109.5 (-0.9%) | 134.8 → 133.3 | same | not separable from launch-to-launch noise |
| 2 | `STRATA_ATTN_MERGE_V2` | 1 | 0 | 130.7 (n=3) | 130.8 → 124.2 (-5.0%) | 133.8 → 132.1 (-1.3%) | 42.3 → 43.8 (+3.5%) | 114.2 → 112.1 (-1.9%) | 132.6 → 126.2 | differ (identical launches differ too — see floor) | not separable from launch-to-launch noise (round agreement solo 3/3, burst 2/3; solo direction stable in every round, magnitude sub-floor) |
| 3 | `STRATA_QIL` | 1 | 0 | 131.4 (n=1) | 131.4 → 124.4 (-5.4%) | 132.5 → 132.8 (+0.2%) | 44.7 → 40.7 (-8.9%) | 119.3 → 113.4 (-4.9%) | 133.6 → 126.0 | differ (identical launches differ too — see floor) | slower than the production value beyond the floor on: prefill |
| 4 | `STRATA_MMVQ_IL_PF` | 0 | 2 | 128.6 (n=3) | 128.5 → 123.3 (-4.0%) | 132.0 → 131.2 (-0.6%) | 43.0 → 40.5 (-5.7%) | 114.7 → 107.3 (-6.4%) | 130.6 → 123.7 | differ (identical launches differ too — see floor) | not separable from launch-to-launch noise (round agreement solo 3/3, burst 3/3; solo direction stable in every round, magnitude sub-floor) |
| 5 | `STRATA_DF2_HEAD_IL` | 1 | 0 | 133.0 (n=1) | 133.0 → 128.4 (-3.5%) | 133.8 → 133.6 (-0.1%) | 44.2 → 43.3 (-2.2%) | 118.0 → 111.9 (-5.2%) | 134.9 → 130.5 | differ (identical launches differ too — see floor) | not separable from launch-to-launch noise |
| 6 | `STRATA_DF2_HEAD_CHUNK` | 8 | 14 | 124.3 (n=1) | 124.3 → 125.3 (+0.7%) | 135.3 → 135.7 (+0.3%) | 43.8 → 42.7 (-2.6%) | 113.6 → 114.0 (+0.3%) | 125.9 → 126.9 | differ (identical launches differ too — see floor) | not separable from launch-to-launch noise |
| 7 | `STRATA_DF2_FORK` | 0 | 1 | 123.5 (n=1) | 123.5 → 127.1 (+2.9%) | 134.7 → 132.8 (-1.4%) | 43.5 → 41.3 (-5.0%) | 110.0 → 108.1 (-1.7%) | 125.1 → 128.8 | differ (identical launches differ too — see floor) | not separable from launch-to-launch noise |
| 8 | `STRATA_DF_PLE` | 0 | 1 | 126.1 (n=1) | 126.1 → 127.1 (+0.8%) | 131.7 → 134.9 (+2.5%) | 44.8 → 43.5 (-2.9%) | 111.8 → 111.1 (-0.7%) | 127.9 → 129.2 | differ (identical launches differ too — see floor) | not separable from launch-to-launch noise |
| 9 | `STRATA_LFUSE` | 0 | 1 | 133.4 (n=1) | 133.4 → 128.4 (-3.7%) | 132.7 → 134.3 (+1.2%) | 39.6 → 44.8 (+13.2%) | 110.1 → 115.0 (+4.5%) | 135.3 → 130.3 | differ (identical launches differ too — see floor) | not separable from launch-to-launch noise |
| 10 | `STRATA_QFUSE` | 0 | 1 | 123.6 (n=1) | 123.6 → 128.8 (+4.1%) | 131.1 → 133.2 (+1.6%) | 43.6 → 43.2 (-1.1%) | 111.6 → 108.9 (-2.5%) | 125.3 → 130.6 | differ (identical launches differ too — see floor) | not separable from launch-to-launch noise |
| 11 | `STRATA_MMVQ_TSUM` | 1 | 0 | 129.0 (n=3) | 128.5 → 134.4 (+4.6%) | 133.0 → 129.3 (-2.8%) | 43.1 → 43.5 (+0.9%) | 113.4 → 117.0 (+3.2%) | 130.6 → 136.9 | differ (identical launches differ too — see floor) | slower than the production value beyond the floor on: burst (round agreement solo 3/3, burst 2/3; solo direction stable in every round, magnitude sub-floor) |
| 12 | `STRATA_ONE_TOKEN_COMMIT` | 0 | 1 | 124.3 (n=1) | 124.3 → 128.9 (+3.6%) | 132.9 → 135.0 (+1.6%) | 43.3 → 41.2 (-4.8%) | 112.6 → 117.3 (+4.1%) | 125.9 → 131.0 | differ (identical launches differ too — see floor) | not separable from launch-to-launch noise |
| 13 | `STRATA_GR_V3` | 0 | 1 | 132.6 (n=3) | 130.0 → 128.2 (-1.3%) | 137.4 → 134.4 (-2.2%) | 42.0 → 43.0 (+2.2%) | 115.0 → 113.8 (-1.1%) | 134.6 → 131.7 | differ (expected: rounding-sensitive switch) | slower than the production value beyond the floor on: burst (round agreement solo 1/3, burst 2/3) |
| 14 | `STRATA_SIDE_MIXER` | 0 | 1 | 130.1 (n=3) | 129.1 → 129.4 (+0.2%) | 134.6 → 131.6 (-2.3%) | 43.5 → 40.3 (-7.2%) | 112.5 → 109.9 (-2.3%) | 131.9 → 128.9 | differ (identical launches differ too — see floor) | slower than the production value beyond the floor on: burst (round agreement solo 1/3, burst 2/3) |
| 15 | `STRATA_SH_STREAM` | 0 | 1 | 126.8 (n=3) | 126.8 → 122.5 (-3.4%) | 135.9 → 124.7 (-8.3%) | 42.4 → 39.1 (-7.8%) | 111.6 → 106.3 (-4.7%) | 128.6 → 124.2 | differ (identical launches differ too — see floor) | slower than the production value beyond the floor on: burst, prefill (round agreement solo 3/3, burst 3/3; solo direction stable in every round, magnitude sub-floor) |
| 16 | `STRATA_VERIFY_ASYNC_LAUNCH` | 0 | 1 | 126.7 (n=1) | 126.7 → 128.4 (+1.4%) | 132.1 → 134.6 (+1.9%) | 42.7 → 42.9 (+0.4%) | 112.4 → 111.8 (-0.6%) | 128.4 → 130.1 | differ (identical launches differ too — see floor) | not separable from launch-to-launch noise |
| 17 | `STRATA_FUSE_CONVAB` | 0 | 1 | 128.4 (n=1) | 128.4 → 126.9 (-1.1%) | 136.6 → 135.9 (-0.5%) | 41.5 → 41.8 (+0.7%) | 110.3 → 115.3 (+4.5%) | 130.2 → 128.7 | differ (identical launches differ too — see floor) | not separable from launch-to-launch noise |
| 18 | `STRATA_FUSE_PCIE_WAIT` | 0 | 1 | 127.2 (n=1) | 127.2 → 123.9 (-2.5%) | 135.3 → 133.8 (-1.1%) | 43.6 → 42.4 (-2.9%) | 110.8 → 110.6 (-0.2%) | 129.3 → 125.5 | differ (identical launches differ too — see floor) | not separable from launch-to-launch noise |
| 19 | `STRATA_QSA_SLOTS` | 0 | 1 | 128.0 (n=1) | 128.0 → 131.8 (+3.0%) | 135.8 → 131.5 (-3.2%) | 42.3 → 43.3 (+2.4%) | 110.2 → 111.6 (+1.3%) | 129.9 → 133.5 | differ (identical launches differ too — see floor) | slower than the production value beyond the floor on: burst |
| 20 | `STRATA_BATCH_SELF_COMMIT` | 0 | 1 | 135.9 (n=3) | 137.7 → 127.3 (-7.5%) | 133.8 → 133.4 (-0.3%) | 41.4 → 43.6 (+5.3%) | 114.0 → 113.6 (-0.4%) | 137.9 → 132.4 | differ (identical launches differ too — see floor) | not separable from launch-to-launch noise (round agreement solo 2/3, burst 1/3) |
| 21 | `STRATA_TOPK_SAMPLE` | 0 | 1 | 130.7 (n=1) | 130.7 → 129.9 (-0.7%) | 131.4 → 131.3 (-0.1%) | 42.1 → 43.5 (+3.4%) | 108.1 → 112.8 (+4.4%) | 132.6 → 132.1 | differ (expected: rounding-sensitive switch) | not separable from launch-to-launch noise |
| 22 | `STRATA_TQ_SUB8` | 0 | 1 | 123.5 (n=1) | 123.5 → 128.4 (+4.0%) | 134.2 → 131.0 (-2.3%) | 44.1 → 42.0 (-4.7%) | 112.0 → 111.6 (-0.4%) | 125.4 → 130.6 | differ (expected: rounding-sensitive switch) | slower than the production value beyond the floor on: burst |
| 23 | `STRATA_IQ_STAGE_GRID` | 1 | 0 | 117.7 (n=3) | 125.1 → 130.7 (+4.5%) | 132.7 → 131.7 (-0.8%) | 43.1 → 42.6 (-1.3%) | 111.3 → 111.8 (+0.4%) | 119.2 → 132.5 | differ (identical launches differ too — see floor) | not separable from launch-to-launch noise (round agreement solo 3/3, burst 1/3; solo direction stable in every round, magnitude sub-floor; degraded launch excluded from the median: control round 3: 100.0 tok/s vs 125.1 median) |
| 24 | `STRATA_PF_WMMA` | 0 | 1 | 127.9 (n=1) | 127.9 → 128.6 (+0.6%) | 135.5 → 133.4 (-1.5%) | 44.2 → 42.2 (-4.5%) | 110.3 → 114.4 (+3.8%) | 129.7 → 130.3 | differ (expected: rounding-sensitive switch) | not separable from launch-to-launch noise |
| 25 | `STRATA_PF_F16` | 0 | 1 | 127.6 (n=3) | 127.5 → 127.5 (+0.0%) | 133.8 → 131.8 (-1.5%) | 42.0 → 42.5 (+1.0%) | 111.5 → 111.9 (+0.4%) | 129.7 → 130.4 | differ (expected: rounding-sensitive switch) | not separable from launch-to-launch noise (round agreement solo 1/3, burst 1/3) |
| 26 | `STRATA_MMQ_KQ` | 1 | 0 | 127.5 (n=1) | 127.5 → 130.2 (+2.1%) | 134.1 → 131.1 (-2.3%) | 41.8 → 43.0 (+2.9%) | 112.5 → 110.2 (-2.0%) | 129.1 → 132.0 | differ (identical launches differ too — see floor) | slower than the production value beyond the floor on: burst |
| 27 | `STRATA_DENSE_MMQ` | 0 | 1 | 130.4 (n=1) | 130.4 → 132.5 (+1.6%) | 133.6 → 134.0 (+0.3%) | 42.8 → 42.8 (-0.1%) | 111.5 → 109.8 (-1.5%) | 132.2 → 134.2 | differ (expected: rounding-sensitive switch) | not separable from launch-to-launch noise |
| 28 | `STRATA_MULTI_BLOCK_ARGMAX` | 1 | 0 | 126.2 (n=1) | 126.2 → 122.1 (-3.2%) | 136.2 → 135.8 (-0.3%) | 44.2 → 43.3 (-2.1%) | 110.8 → 113.3 (+2.2%) | 128.1 → 123.7 | same | not separable from launch-to-launch noise |
| 29 | `STRATA_NO_BATCH_KV_STEP` | 0 | 1 | 129.1 (n=1) | 129.1 → 121.0 (-6.3%) | 133.6 → 135.3 (+1.3%) | 43.9 → 42.6 (-3.0%) | 111.2 → 109.4 (-1.7%) | 131.3 → 123.0 | differ (identical launches differ too — see floor) | not separable from launch-to-launch noise |
| 30 | `STRATA_BATCH_ATTN_ROWS` | 1 | 0 | 130.2 (n=1) | 130.2 → 126.4 (-2.9%) | 135.9 → 129.9 (-4.4%) | 42.9 → 43.0 (+0.1%) | 111.5 → 114.9 (+3.1%) | 131.9 → 128.5 | differ (identical launches differ too — see floor) | slower than the production value beyond the floor on: burst |
| 31 | `STRATA_BATCH_GDN_GROUPS` | 1 | 0 | 124.1 (n=1) | 124.1 → 129.0 (+4.0%) | 133.4 → 133.0 (-0.3%) | 42.4 → 44.5 (+5.0%) | 111.0 → 113.5 (+2.2%) | 126.2 → 130.8 | differ (identical launches differ too — see floor) | not separable from launch-to-launch noise |
| 32 | A/A floor (identical config) | 1 | 1 | 138.0 (n=1) | 138.0 → 126.9 (-8.0%) | 136.4 → 134.6 (-1.3%) | 43.8 → 40.5 (-7.6%) | 112.2 → 108.6 (-3.2%) | 140.0 → 129.0 | n/a (identical config) | A/A floor sample (no treatment) |
| 33 | A/A floor (identical config) | 1 | 1 | 129.1 (n=1) | 129.1 → 121.8 (-5.6%) | 131.8 → 132.5 (+0.5%) | 41.0 → 44.1 (+7.4%) | 111.6 → 111.3 (-0.3%) | 130.9 → 124.1 | n/a (identical config) | A/A floor sample (no treatment) |
| 34 | `STRATA_MMVQ_IL_PF` | 0 | 3 | 130.7 (n=1) | 130.7 → 125.4 (-4.1%) | 132.9 → 130.7 (-1.6%) | 42.4 → 41.8 (-1.3%) | 114.3 → 112.1 (-2.0%) | 132.4 → 126.9 | differ (identical launches differ too — see floor) | not separable from launch-to-launch noise |
| 35 | `STRATA_MMVQ_IL_PF` | 0 | 4 | 132.8 (n=1) | 132.8 → 124.9 (-5.9%) | 131.8 → 134.0 (+1.7%) | 43.3 → 42.5 (-1.9%) | 109.4 → 110.6 (+1.1%) | 134.6 → 126.9 | differ (identical launches differ too — see floor) | not separable from launch-to-launch noise |
| 36 | `STRATA_MMVQ_IL_PF` | 0 | 6 | 131.5 (n=1) | 131.5 → 128.2 (-2.5%) | 132.1 → 135.5 (+2.6%) | 42.4 → 43.7 (+3.1%) | 116.3 → 114.6 (-1.5%) | 133.6 → 129.8 | differ (identical launches differ too — see floor) | not separable from launch-to-launch noise |

### Switches in this file that are not settable, not compiled in, or blocked

| switch | why |
|---|---|
| `STRATA_NORM_ROPE` | **blocked by its own parity gate**: `rope_parity` check 6 (`native_qsa_rms_norm_rope hd=128/256`) FAILS on gfx1201 with `STRATA_NORM_ROPE=1` — 3644 of 49152 (hd=128) and 3546 of 98304 (hd=256) elements differ, 7190 failures total (`build-hip/rope_parity`, 2026-11-12). The doc requires that check to pass before enabling this switch on RDNA4, so no throughput A/B was run; the fused kernel needs fixing first. |
| `STRATA_ATTN_DAILY` | name is absent from the built engine: the getenv site sits inside `#if !defined(__HIPCC__)`, so on HIP it is not settable even though docs/commit 8c03a43ed describe a wave32 port |
| `STRATA_GRAPH_PRIO` | not compiled into the engine (every HIP caller guarded off) |
| `STRATA_MTP_PRIORITY` | not compiled into the engine |
| `STRATA_MTP_SHARED_BRANCH` | not compiled into the engine (HIP keeps one stream) |
| `STRATA_PCIE_AUTO_CANON` | not compiled into the engine (CUDA-only tuning path) |
| `STRATA_HIP_MMVQ_IL` | not compiled into this engine; already A/B-dropped from production earlier |
| `STRATA_MMQ_TAB` | build-time CMake option (STRATA_MMQ_TAB=OFF in build-hip); needs a second build before STRATA_MMQ_DIRECT can do anything |

### Corrections this campaign makes to the sections above

- **`STRATA_ATTN_DAILY` is not settable on this backend.** The HIP section lists it as ported to wave32 HIP
  (`8c03a43ed`), but the name is absent from the built engine: its read site sits inside
  `#if !defined(__HIPCC__)` (`qsa_decode_attn.cu:1065`). What gfx1201 actually exposes is
  `STRATA_ATTN_PRE75` and `STRATA_ATTN_MERGE_V2`, both on by default, and both measured here.
- **`STRATA_NORM_ROPE` is blocked by its own precondition.** This file says to pass `rope_parity` check 6 on
  gfx1201 first; that check fails there — with `STRATA_NORM_ROPE=1`, `native_qsa_rms_norm_rope hd=128` differs
  in 3,644 of 49,152 elements and `hd=256` in 3,546 of 98,304 (7,190 failures). The fused RMSNorm+RoPE kernel
  is numerically wrong on RDNA4, so no throughput A/B was run.
- **`STRATA_IQ_STAGE_GRID` and `STRATA_MMQ_KQ` are default-ON on gfx12, not opt-ins.** `g_stage_grid`
  (`iq_kernels.cu:3428`) keeps staged 64-bit codebook tables for any value but `"0"`, and `STRATA_MMQ_KQ`
  comes from `gfx12_value_changing_env` (`arch_defaults.cpp:306`) — its startup banner prints in every arm.
  Production therefore already runs both, and the A/B had to ablate them.
- **`STRATA_MMQ_TAB` cannot be measured in this build.** It is a CMake option
  (`STRATA_MMQ_TAB=OFF` in `build-hip`), so `STRATA_MMQ_DIRECT` is inert until a build with
  `-DSTRATA_MMQ_TAB=ON` exists (that second build is the open follow-up).
- **Tier 1's open question about `STRATA_MULTI_BLOCK_ARGMAX` is answered.** The path does engage:
  `multi_block_head_ops()` returns true on gfx12 by default (`verify_kernels.cu:1141`). It is not dead code —
  ablating it moved nothing beyond the floor, which is the useful result.
- **Five more names are not in the built engine** (`STRATA_GRAPH_PRIO`, `STRATA_MTP_PRIORITY`,
  `STRATA_MTP_SHARED_BRANCH`, `STRATA_PCIE_AUTO_CANON`, `STRATA_HIP_MMVQ_IL`), so no environment change can
  test them: porting those gates is source work, not a benchmark.

### Reproduction

```
cd ~/Strata-latest-20261112-182859
python3 bench/ports-ab-2026-11-12/gen_configs.py                 # arm profiles from the production profile
systemd-run --user --scope -- python3 bench/ports-ab-2026-11-12/driver.py                 # full screen
systemd-run --user --scope -- python3 bench/ports-ab-2026-11-12/driver.py \
    --skip-ctest --round-offset 1 --rounds 2 --only <id>,<id>                              # confirmation
python3 bench/ports-ab-2026-11-12/render_doc.py                  # regenerate this table
```

The driver stops `strata.service` for the window, runs one engine at a time, waits for the GPU to be free
between arms, and restarts production on exit (readiness verified). Per-arm raw records (every request,
engine log slices, hashes, argv, effective environment): `bench/ports-ab-2026-11-12/results/`. Driver
transcript: `bench/ports-ab-2026-11-12/campaign.log`. CTest parity subset for this window:
`bench/ports-ab-2026-11-12/results/ctest.log` (a skip is not a pass).

<!-- AB-CAMPAIGN:END -->
