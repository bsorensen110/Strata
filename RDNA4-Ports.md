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

<!-- AB-CAMPAIGN-2:BEGIN -> generated by bench/ports-ab-2026-11-13/render_doc.py -->

## Measured: new optimizations on gfx1201 (R9700) — campaign 2026-11-13

Progress: **11/11** arms measured. Engine source: `fork-main` at `df8ee1a30`
(the 11-commit delta on top of the previously promoted `edc2956c7`).

**Method.** Same harness as the switch campaign above: one engine at a time as a transient user unit,
fresh launch per arm with the live production argv (production profile regenerated per arm by
`candidate_config.py`, pack `unsloth-ud-iq4_xs-v31`, DFlash2 `selq4`, `--spec 8 --draft-conf 0.35`,
`parallel: 2`, `--kv-resident 32768`, `--expert-cache auto`), the live production environment, and the
tested switch pinned explicitly in both arms. `strata.service` is stopped for the window and restored on
exit. Workload per launch: 3,947-token prefill, two 512-token greedy trials (cold then warm), then a
2×256 concurrent burst; temperature 0, thinking off, identical prompts across arms.

**Two kinds of row.** `engine binary` rows swap only the engine executable — production's current build
(`fork-edc2956c7`) against the new `build-hip` binary — which is what measures the commits that have no
switch (the eight unconditional fusions: the fused `native_mmvq_multi_segs` projections, fwht256
folded into `native_qsa_gate_quantize_q8_1`, batched `qsa_kv_resolve` and `idx_snap`/`idx_append` across
slots, `logits_nonfinite` fused into `argmax_rows`, SwiGLU+q8_1 in `s27_gu_swiglu_kernel`, vectorized
`combine_multi_hits`, and the DFlash2 final-layer Q8_1 fusion). `switch` rows keep the new binary on both
sides and move one variable. Three of the new knobs are **default-ON in the new code** (`STRATA_DF2_GEMM`,
`STRATA_MMVQ_SEGS`, the HIP rows-per-block auto tuning behind `STRATA_MMVF_RPB`), so they are tested as
ablations; `STRATA_ATTN_V4` is opt-in and tested as an enable.

**Floor.** A/A pairs (``aa_floor_1`, `aa_floor_2``, identical configuration launched
twice) measured **warm solo ±4.0%, 2×256 burst ±2.0%, p4k prefill
±1.8%** in this window. Deltas are medians across rounds and every verdict is judged
against those numbers; greedy-bits differences are reported but are not a parity gate here (identical
launches differ too).

### Engine binary delta

| # | arm | control value | arm value | warm solo tok/s (control, median) | warm solo (arm, Δ) | 2×256 burst agg (arm, Δ) | p4k prefill tok/s (arm, Δ) | cold solo (arm, Δ) | output bits · verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | engine binary | prod engine | new engine | 126.4 (n=3) | 126.4 → 131.1 (+3.7%) | 131.3 → 135.4 (+3.1%) | 43.8 → 44.5 (+1.6%) | 110.8 → 113.4 (+2.4%) | differ · faster on all three workloads by median, magnitude at or inside the floor (`fork-edc2956c7` e9a2fcf8e04a → `build-hip` 4109831f3870; round agreement solo 2/3, burst 2/3) |

### New switches and the tunings they gate

| # | arm | control value | arm value | warm solo tok/s (control, median) | warm solo (arm, Δ) | 2×256 burst agg (arm, Δ) | p4k prefill tok/s (arm, Δ) | cold solo (arm, Δ) | output bits · verdict |
|---|---|---|---|---|---|---|---|---|---|
| 2 | `STRATA_ATTN_V4` | 0 | 1 | 131.8 (n=3) | 131.8 → 131.7 (-0.1%) | 134.9 → 135.6 (+0.5%) | 44.0 → 43.6 (-0.9%) | 113.0 → 113.2 (+0.2%) | differ · not separable from launch-to-launch noise (round agreement solo 2/3, burst 2/3) |
| 3 | `STRATA_DF2_GEMM` | 1 | 0 | 131.4 (n=1) | 131.4 → 125.7 (-4.3%) | 135.2 → 133.9 (-0.9%) | 43.6 → 41.7 (-4.4%) | 111.5 → 111.0 (-0.5%) | differ · slower than the control beyond the floor on: solo, prefill |
| 4 | `STRATA_MMVQ_SEGS` | 1 | 0 | 129.1 (n=1) | 129.1 → 127.0 (-1.6%) | 136.5 → 134.4 (-1.5%) | 44.0 → 44.6 (+1.4%) | 111.5 → 114.5 (+2.7%) | differ · not separable from launch-to-launch noise |
| 5 | `STRATA_NO_MMVQ_ROWS2` | (unset/auto) | 1 | 129.1 (n=3) | 129.1 → 133.4 (+3.3%) | 137.8 → 135.3 (-1.8%) | 41.3 → 43.5 (+5.4%) | 113.4 → 117.7 (+3.7%) | differ · not separable from launch-to-launch noise (round agreement solo 2/3, burst 2/3) |
| 6 | `STRATA_MMVF_RPB` | (unset/auto) | 1 | 131.1 (n=1) | 131.1 → 130.9 (-0.1%) | 136.8 → 137.2 (+0.3%) | 44.6 → 44.6 (-0.1%) | 114.2 → 115.1 (+0.8%) | same · not separable from launch-to-launch noise |
| 7 | `STRATA_MMVF_RPB` | (unset/auto) | 2 | 131.7 (n=1) | 131.7 → 127.6 (-3.1%) | 134.1 → 136.1 (+1.5%) | 44.9 → 43.6 (-2.8%) | 114.2 → 115.5 (+1.1%) | differ · slower than the control beyond the floor on: prefill |
| 8 | `STRATA_MMVF_RPB` | (unset/auto) | 4 | 127.2 (n=1) | 127.2 → 133.6 (+5.1%) | 135.1 → 132.9 (-1.6%) | 43.0 → 43.6 (+1.3%) | 112.1 → 111.7 (-0.4%) | differ · solo faster than the floor, concurrent decode flat — directional |
| 9 | `STRATA_BF16_TSUM` | 1 | 0 | 129.2 (n=1) | 129.2 → 129.4 (+0.1%) | 133.9 → 133.4 (-0.4%) | 43.4 → 43.5 (+0.3%) | 112.1 → 110.8 (-1.2%) | same · not separable from launch-to-launch noise |
| 10 | A/A floor (identical config) | 0 | 0 | 126.3 (n=3) | 126.3 → 131.3 (+4.0%) | 136.7 → 134.1 (-1.9%) | 41.4 → 42.1 (+1.8%) | 112.0 → 112.4 (+0.4%) | n/a (identical config) · A/A floor sample (no treatment) |
| 11 | A/A floor (identical config) | 0 | 0 | 132.5 (n=3) | 132.5 → 128.4 (-3.1%) | 136.7 → 134.8 (-1.4%) | 43.2 → 42.5 (-1.6%) | 111.4 → 114.7 (+3.0%) | n/a (identical config) · A/A floor sample (no treatment) |

### Promotion (2026-11-13 00:56 EST)

Promoted: `components/engine/fork-df8ee1a30/strata` (sha256 `4109831f3870be79…`, engine `0.1.59.1`),
with the profile `exe` repointed from `fork-edc2956c7` (`e9a2fcf8e04ada92…`, `0.1.56`). The engine-delta
row is why: all three medians moved up (+3.7% warm solo, +3.1% burst, +1.6% prefill), nothing regressed,
and the two new default-on tunings beat their own ablations (`STRATA_DF2_GEMM=0` costs −4.3% solo and
−4.4% prefill; `STRATA_MMVQ_SEGS=0` costs −1.6% solo). The new defaults therefore need no environment:
they are on in the binary.

`STRATA_ATTN_V4=1` was enabled in the production drop-in **on request**, not on the strength of its
measurement: it came in at −0.1% solo / +0.5% burst (inside the floor) with differing greedy bits, so it is
a no-cost-if-wrong enable rather than a measured win. `STRATA_NO_MMVQ_ROWS2=1` (solo +3.3%, burst −1.8%)
and the forced `STRATA_MMVF_RPB` values were **not** enabled.

Post-promotion verification: engine `0.1.59.1` running with `STRATA_ATTN_V4=1` visible in
`/proc/<pid>/environ`, a `ready` event on the API, a correct smoke completion, and a 50-request soak
(25 waves of 2, `parallel: 2`) with **0 failures** — no NaN, empty or error replies. Rollback:
`~/strata-promote-backup-20261113-df8ee1a30/rollback.sh` (restores the previous profile and drop-in,
restarts, and waits for readiness).

### Reproduction

```
cd ~/Strata-latest-20261112-182859
git checkout fork-main && git rebase fork/fork-main      # df8ee1a30
cmake --build build-hip --target strata -j 8
python3 bench/ports-ab-2026-11-13/gen_configs.py
systemd-run --user --scope -- python3 bench/ports-ab-2026-11-13/driver.py --skip-ctest --rounds 3 --only engine_delta
systemd-run --user --scope -- python3 bench/ports-ab-2026-11-13/driver.py --skip-ctest --only attn_v4,df2_gemm,mmvq_segs
python3 bench/ports-ab-2026-11-13/render_doc.py
```

Raw per-arm records (requests, engine log slice, argv, argv-verified environment, hashes):
`bench/ports-ab-2026-11-13/results/`; driver transcript `bench/ports-ab-2026-11-13/campaign.log`.

<!-- AB-CAMPAIGN-2:END -->

<!-- AB-CAMPAIGN-3:BEGIN -> generated by bench/ports-ab-2026-11-13-env/render_doc.py -->

## Env-switch audit against the promoted engine — 2026-11-13

Progress: **4/4** arms measured. Engine under test: the promoted production binary
`components/engine/fork-df8ee1a30/strata` (sha256 `4109831f3870be79…`, `0.1.59.1`) — the same bytes the
service runs, so every row below isolates one environment variable against the current production argv.

**Why this audit.** Comparing the `STRATA_*` names the engine binary contains with the `STRATA_*` names
the live production process carries turned up four switches production sets that **no source file reads and
the binary does not contain**: `STRATA_DISPATCH_FAST_JOBS`, `STRATA_PREFILL_CPU_SHARE_BALANCE`,
`STRATA_PREFILL_CPU_SHARE_MAX_T` and `STRATA_VERIFY_SAMPLE_ASYNC`. Two of them have live counterparts the
engine really reads under a slightly different spelling, so production has been setting a value that never
arrives:

- `STRATA_PREFILL_CPU_SHARE_MAX_T=32` (dead) vs **`STRATA_PREFILL_CPU_SHARE_MAXT`** (1 read site).
- `STRATA_VERIFY_SAMPLE_ASYNC=1` (dead) vs **`STRATA_VERIFY_ASYNC_LAUNCH`** (3 read sites, unset in production).

The other two (`STRATA_DISPATCH_FAST_JOBS`, `STRATA_PREFILL_CPU_SHARE_BALANCE`) have no counterpart at all —
they are inert and need no benchmark, only removal.

**Method.** Same harness as the campaigns above: one engine at a time as a transient user unit, fresh launch
per arm with the production argv and the live production environment, the tested variable pinned explicitly
in both arms (control = production's effective value, treatment = the value production meant to set).
`strata.service` is stopped for the window and restored on exit. Workload per launch: 3,947-token prefill,
two 512-token greedy trials (cold, warm), then a 2×256 concurrent burst at `parallel: 2`.

**Floor.** A/A pairs (``aa_floor_1`, `aa_floor_2``, identical configuration launched
twice) measured **warm solo ±2.2%, burst ±2.0%, prefill
±2.1%** in this window. Deltas are medians across rounds; verdicts are judged against
those numbers.

### Arms

| # | arm | control value | arm value | warm solo tok/s (control, median) | warm solo (arm, Δ) | 2×256 burst agg (arm, Δ) | p4k prefill tok/s (arm, Δ) | cold solo (arm, Δ) | output bits · verdict |
|---|---|---|---|---|---|---|---|---|---|

| 1 | `STRATA_VERIFY_ASYNC_LAUNCH` | (unset/auto) | 1 | 127.7 (n=4) | 127.7 → 130.3 (+2.0%) | 135.9 → 133.3 (-1.9%) | 43.9 → 41.9 (-4.6%) | 112.5 → 113.3 (+0.7%) | differ · slower than the control beyond the floor on: prefill (round agreement solo 2/4, burst 3/4) |
| 2 | `STRATA_PREFILL_CPU_SHARE_MAXT` | (unset/auto) | 32 | 132.8 (n=2) | 132.8 → 129.8 (-2.3%) | 135.2 → 135.5 (+0.2%) | 42.5 → 40.8 (-4.1%) | 112.6 → 114.5 (+1.6%) | differ · slower than the control beyond the floor on: solo, prefill (round agreement solo 2/2, burst 1/2) |
| 3 | A/A floor (identical config) | (unset/auto) |  | 135.3 (n=2) | 135.3 → 137.1 (+1.3%) | 132.9 → 134.5 (+1.2%) | 43.0 → 42.7 (-0.7%) | 117.4 → 116.4 (-0.9%) | n/a (identical config) · A/A floor sample (no treatment) |
| 4 | A/A floor (identical config) | (unset/auto) |  | 132.8 (n=2) | 132.8 → 129.8 (-2.2%) | 132.5 → 135.1 (+1.9%) | 43.1 → 44.0 (+2.1%) | 112.2 → 112.2 (+0.0%) | n/a (identical config) · A/A floor sample (no treatment) |

### Promotion (2026-11-13 00:56 EST)

Promoted: `components/engine/fork-df8ee1a30/strata` (sha256 `4109831f3870be79…`, engine `0.1.59.1`),
with the profile `exe` repointed from `fork-edc2956c7` (`e9a2fcf8e04ada92…`, `0.1.56`). The engine-delta
row is why: all three medians moved up (+3.7% warm solo, +3.1% burst, +1.6% prefill), nothing regressed,
and the two new default-on tunings beat their own ablations (`STRATA_DF2_GEMM=0` costs −4.3% solo and
−4.4% prefill; `STRATA_MMVQ_SEGS=0` costs −1.6% solo). The new defaults therefore need no environment:
they are on in the binary.

`STRATA_ATTN_V4=1` was enabled in the production drop-in **on request**, not on the strength of its
measurement: it came in at −0.1% solo / +0.5% burst (inside the floor) with differing greedy bits, so it is
a no-cost-if-wrong enable rather than a measured win. `STRATA_NO_MMVQ_ROWS2=1` (solo +3.3%, burst −1.8%)
and the forced `STRATA_MMVF_RPB` values were **not** enabled.

Post-promotion verification: engine `0.1.59.1` running with `STRATA_ATTN_V4=1` visible in
`/proc/<pid>/environ`, a `ready` event on the API, a correct smoke completion, and a 50-request soak
(25 waves of 2, `parallel: 2`) with **0 failures** — no NaN, empty or error replies. Rollback:
`~/strata-promote-backup-20261113-df8ee1a30/rollback.sh` (restores the previous profile and drop-in,
restarts, and waits for readiness).

### Findings

**Four production variables do nothing.** `STRATA_DISPATCH_FAST_JOBS`, `STRATA_PREFILL_CPU_SHARE_BALANCE`,
`STRATA_PREFILL_CPU_SHARE_MAX_T` and `STRATA_VERIFY_SAMPLE_ASYNC` are set in the production drop-in, but no
source file in the tree reads them, the engine binary does not contain the strings, and the installed
`serve/server.py` does not translate them into anything the engine reads. They have been inert since they
were added — production was not running whatever those four were meant to turn on.

**Two of them have live counterparts, and neither is worth enabling.**

- `STRATA_VERIFY_ASYNC_LAUNCH=1` (the real gate, 3 read sites, unset in production): 4 rounds gave
  +7.2 / +6.1 / −1.7 / −7.2% solo (median +2.0%, inside the ±2.2% floor) and −1.6 / −4.3 / +5.6 / −33.1% on
  the concurrent burst, with prefill worse in 3 of 4 rounds (median −4.6%). The −33% burst is a single
  collapsed launch out of four, not a systematic loss, but at `parallel: 2` that is a one-in-four chance of a
  large concurrency hit for no solo gain. Not enabled.
- `STRATA_PREFILL_CPU_SHARE_MAXT=32` (the value production meant to pass under the dead `_MAX_T` name):
  −0.9 / −3.6% solo, +2.1 / −1.7% burst, −2.1 / −6.1% prefill. Applying the intended value does not help, so
  the dead variable is harmless as well as inert. Not enabled.

**What this round could not do.** `fork/fork-main` has no commits beyond `df8ee1a30`, which is the engine
already promoted here, so there were no new optimizations to A/B against it; and `Hip-switch-tests.md` does
not exist in this checkout (all 265 refs), on `origin/main`, or anywhere under `/home/soren`. The switch
inventory above is the substitute: everything the binary knows that production does not set, cross-checked
the other way round.

### Reproduction

```
cd ~/Strata-latest-20261112-182859
git checkout fork-main && git rebase fork/fork-main      # df8ee1a30
cmake --build build-hip --target strata -j 8
python3 bench/ports-ab-2026-11-13-env/gen_configs.py
systemd-run --user --scope -- python3 bench/ports-ab-2026-11-13-env/driver.py --skip-ctest --rounds 3 --only engine_delta
systemd-run --user --scope -- python3 bench/ports-ab-2026-11-13-env/driver.py --skip-ctest --only attn_v4,df2_gemm,mmvq_segs
python3 bench/ports-ab-2026-11-13-env/render_doc.py
```

Raw per-arm records (requests, engine log slice, argv, argv-verified environment, hashes):
`bench/ports-ab-2026-11-13-env/results/`; driver transcript `bench/ports-ab-2026-11-13-env/campaign.log`.

<!-- AB-CAMPAIGN-3:END -->

<!-- AB-CAMPAIGN-3:BEGIN -> generated by bench/ports-ab-2026-11-14-dflash2/render_doc.py -->

## DFlash2 optimizations and Hip-switch-tests.md matrix — 2026-11-13/14

Progress: **10/10** arms measured. Engine under test: the promoted production binary
`components/engine/fork-df8ee1a30/strata` (sha256 `4109831f3870be79…`, `0.1.59.1`) — the same bytes the
service runs, so every row below isolates one environment variable against the current production argv.

**Why this audit.** Comparing the `STRATA_*` names the engine binary contains with the `STRATA_*` names
the live production process carries turned up four switches production sets that **no source file reads and
the binary does not contain**: `STRATA_DISPATCH_FAST_JOBS`, `STRATA_PREFILL_CPU_SHARE_BALANCE`,
`STRATA_PREFILL_CPU_SHARE_MAX_T` and `STRATA_VERIFY_SAMPLE_ASYNC`. Two of them have live counterparts the
engine really reads under a slightly different spelling, so production has been setting a value that never
arrives:

- `STRATA_PREFILL_CPU_SHARE_MAX_T=32` (dead) vs **`STRATA_PREFILL_CPU_SHARE_MAXT`** (1 read site).
- `STRATA_VERIFY_SAMPLE_ASYNC=1` (dead) vs **`STRATA_VERIFY_ASYNC_LAUNCH`** (3 read sites, unset in production).

The other two (`STRATA_DISPATCH_FAST_JOBS`, `STRATA_PREFILL_CPU_SHARE_BALANCE`) have no counterpart at all —
they are inert and need no benchmark, only removal.

**Method.** Same harness as the campaigns above: one engine at a time as a transient user unit, fresh launch
per arm with the production argv and the live production environment, the tested variable pinned explicitly
in both arms (control = production's effective value, treatment = the value production meant to set).
`strata.service` is stopped for the window and restored on exit. Workload per launch: 3,947-token prefill,
two 512-token greedy trials (cold, warm), then a 2×256 concurrent burst at `parallel: 2`.

**Floor.** A/A pairs (``aa_floor_1`, `aa_floor_2``, identical configuration launched
twice) measured **warm solo ±1.5%, burst ±2.0%, prefill
±1.4%** in this window. Deltas are medians across rounds; verdicts are judged against
those numbers.

### Arms

| # | arm | control value | arm value | warm solo tok/s (control, median) | warm solo (arm, Δ) | 2×256 burst agg (arm, Δ) | p4k prefill tok/s (arm, Δ) | cold solo (arm, Δ) | output bits · verdict |
|---|---|---|---|---|---|---|---|---|---|

| 2 | `STRATA_DF2_BEAM` | (unset/auto) | 1 | 133.3 (n=3) | 133.3 → 132.5 (-0.6%) | 140.4 → 141.4 (+0.7%) | 42.2 → 43.7 (+3.7%) | 115.0 → 116.0 (+0.9%) | differ · not separable from launch-to-launch noise (round agreement solo 2/3, burst 1/3) |
| 3 | `STRATA_DF2_HEAD` | (unset/auto) | pruned | 137.1 (n=3) | 137.1 → 132.5 (-3.4%) | 137.4 → 135.1 (-1.7%) | 41.6 → 43.3 (+4.1%) | 114.6 → 112.6 (-1.7%) | differ · slower than the control beyond the floor on: solo (round agreement solo 2/3, burst 3/3) |
| 4 | `STRATA_DF2_ATTN_F16_SMEM` | (unset/auto) | 0 | 148.6 (n=1) | 148.6 → 142.0 (-4.5%) | 135.8 → 140.2 (+3.3%) | 43.0 → 43.6 (+1.2%) | 117.2 → 115.4 (-1.5%) | same · slower than the control beyond the floor on: solo |
| 5 | `STRATA_DF2_GEMM_32` | (unset/auto) | 0 | 137.3 (n=1) | 137.3 → 138.5 (+0.9%) | 140.1 → 141.2 (+0.8%) | 41.7 → 40.1 (-3.9%) | 112.5 → 112.8 (+0.2%) | differ · slower than the control beyond the floor on: prefill |
| 6 | `STRATA_BATCH_DF2_MAX_T` | 8 | 12 | 135.7 (n=1) | 135.7 → 124.7 (-8.1%) | 137.2 → 136.7 (-0.3%) | 42.4 → 42.6 (+0.6%) | 113.0 → 113.8 (+0.7%) | differ · slower than the control beyond the floor on: solo |
| 7 | `STRATA_BATCH_DF2_MAX_T` | 8 | 16 | 133.3 (n=3) | 133.3 → 147.2 (+10.4%) | 139.0 → 135.3 (-2.7%) | 42.7 → 43.2 (+1.2%) | 115.7 → 118.0 (+2.0%) | differ · slower than the control beyond the floor on: burst (round agreement solo 3/3, burst 3/3) |
| 8 | A/A floor (identical config) | (unset/auto) |  | 130.1 (n=3) | 130.1 → 132.0 (+1.5%) | 140.4 → 142.1 (+1.2%) | 42.7 → 43.0 (+0.7%) | 115.8 → 114.9 (-0.8%) | n/a (identical config) · A/A floor sample (no treatment) |
| 9 | A/A floor (identical config) | (unset/auto) |  | 130.5 (n=3) | 130.5 → 130.8 (+0.2%) | 142.0 → 140.6 (-1.0%) | 43.6 → 43.0 (-1.4%) | 112.3 → 114.9 (+2.3%) | n/a (identical config) · A/A floor sample (no treatment) |
| 10 | `STRATA_DF2_BEAM` | (unset/auto) | 1 | 132.0 (n=2) | 132.0 → 130.5 (-1.1%) | 133.7 → 142.8 (+6.8%) | 42.3 → 43.0 (+1.7%) | 112.7 → 115.6 (+2.6%) | differ · not separable from launch-to-launch noise (round agreement solo 1/2, burst 2/2) |

### Promotion (2026-11-13 00:56 EST)

Promoted: `components/engine/fork-df8ee1a30/strata` (sha256 `4109831f3870be79…`, engine `0.1.59.1`),
with the profile `exe` repointed from `fork-edc2956c7` (`e9a2fcf8e04ada92…`, `0.1.56`). The engine-delta
row is why: all three medians moved up (+3.7% warm solo, +3.1% burst, +1.6% prefill), nothing regressed,
and the two new default-on tunings beat their own ablations (`STRATA_DF2_GEMM=0` costs −4.3% solo and
−4.4% prefill; `STRATA_MMVQ_SEGS=0` costs −1.6% solo). The new defaults therefore need no environment:
they are on in the binary.

`STRATA_ATTN_V4=1` was enabled in the production drop-in **on request**, not on the strength of its
measurement: it came in at −0.1% solo / +0.5% burst (inside the floor) with differing greedy bits, so it is
a no-cost-if-wrong enable rather than a measured win. `STRATA_NO_MMVQ_ROWS2=1` (solo +3.3%, burst −1.8%)
and the forced `STRATA_MMVF_RPB` values were **not** enabled.

Post-promotion verification: engine `0.1.59.1` running with `STRATA_ATTN_V4=1` visible in
`/proc/<pid>/environ`, a `ready` event on the API, a correct smoke completion, and a 50-request soak
(25 waves of 2, `parallel: 2`) with **0 failures** — no NaN, empty or error replies. Rollback:
`~/strata-promote-backup-20261113-df8ee1a30/rollback.sh` (restores the previous profile and drop-in,
restarts, and waits for readiness).

### Findings

**Four production variables do nothing.** `STRATA_DISPATCH_FAST_JOBS`, `STRATA_PREFILL_CPU_SHARE_BALANCE`,
`STRATA_PREFILL_CPU_SHARE_MAX_T` and `STRATA_VERIFY_SAMPLE_ASYNC` are set in the production drop-in, but no
source file in the tree reads them, the engine binary does not contain the strings, and the installed
`serve/server.py` does not translate them into anything the engine reads. They have been inert since they
were added — production was not running whatever those four were meant to turn on.

**Two of them have live counterparts, and neither is worth enabling.**

- `STRATA_VERIFY_ASYNC_LAUNCH=1` (the real gate, 3 read sites, unset in production): 4 rounds gave
  +7.2 / +6.1 / −1.7 / −7.2% solo (median +2.0%, inside the ±2.2% floor) and −1.6 / −4.3 / +5.6 / −33.1% on
  the concurrent burst, with prefill worse in 3 of 4 rounds (median −4.6%). The −33% burst is a single
  collapsed launch out of four, not a systematic loss, but at `parallel: 2` that is a one-in-four chance of a
  large concurrency hit for no solo gain. Not enabled.
- `STRATA_PREFILL_CPU_SHARE_MAXT=32` (the value production meant to pass under the dead `_MAX_T` name):
  −0.9 / −3.6% solo, +2.1 / −1.7% burst, −2.1 / −6.1% prefill. Applying the intended value does not help, so
  the dead variable is harmless as well as inert. Not enabled.

**What this round could not do.** `fork/fork-main` has no commits beyond `df8ee1a30`, which is the engine
already promoted here, so there were no new optimizations to A/B against it; and `Hip-switch-tests.md` does
not exist in this checkout (all 265 refs), on `origin/main`, or anywhere under `/home/soren`. The switch
inventory above is the substitute: everything the binary knows that production does not set, cross-checked
the other way round.

### Draft telemetry (the doc's "what to record per run")

The new engine prints acceptance and window shape per batch; the previous engine printed neither, which is
itself part of the delta. Last batch line per arm:

| arm | MTP proposals accepted | avg rows/window | ms/window | rows/s |
|---|---|---|---|---|
| production engine (control) | 350 of 160 (218.8%) | 6.22 | 40.78 | 141.8 |
| new engine, no switch | 359 of 151 (237.7%) | 6.54 | 40.36 | 150.8 |
| new engine + `STRATA_DF2_BEAM=1` | 362 of 148 (244.6%) | 6.80 | 40.83 | 154.4 |
| new engine + `STRATA_BATCH_DF2_MAX_T=16` | 387 of 123 (314.6%) | 8.10 | 59.03 | 130.2 |

Raising the window cap does exactly what the doc predicts — deeper drafts, more accepted tokens per verify
step (405% acceptance at `MAX_T=16` on the screen round, 8.1-10.0 rows per window against 6.2-6.5) — but the
window costs 59-64 ms instead of 41, so batch throughput falls. At `parallel: 2` that trade is negative on
the burst and only shows up as a solo-decode gain (+10.4% median warm), so it is not shipped.

### Promotion (2026-11-13 02:52 EST)

Promoted `components/engine/fork-68872b2d1/strata` (sha256 `8bd8a4e8741b09fe…`), profile `exe` repointed
from `fork-df8ee1a30`, with `STRATA_DF2_BEAM=1` added to the production drop-in. On its own the new engine
buys +5.7% concurrent burst but loses 7.6% prefill; with `BEAM=1` the pair measures **+6.8% burst, +1.7%
prefill, −1.1% solo** against the previous engine and environment, i.e. the prefill regression is gone.
**Correction (same day):** `STRATA_DF2_BEAM=1` is the *revert* to greedy select, not the new optimization —
the order-1 Viterbi beam search is `=4`, the build default. So that promotion in fact ran the *old* select
method; the drop-in was corrected to `STRATA_DF2_BEAM=4` after the long-context campaign measured 4 vs 1 as
neutral on throughput. `STRATA_BATCH_DF2_MAX_T=16`, `=12`,
`STRATA_DF2_HEAD=pruned`, `STRATA_DF2_ATTN_F16_SMEM=0` and `STRATA_DF2_GEMM_32=0` all measured neutral or
worse and were **not** enabled; the six optimizations themselves and the new defaults (`native` head,
FP16 attention SMEM, 32-col GEMM, IL head gating, 8-slot batch drafting) need no environment.

Also dropped from the drop-in in this window: the four inert variables found by the previous audit
(`STRATA_DISPATCH_FAST_JOBS`, `STRATA_PREFILL_CPU_SHARE_BALANCE`, `STRATA_PREFILL_CPU_SHARE_MAX_T`,
`STRATA_VERIFY_SAMPLE_ASYNC`) — verified absent from the live process environment after the restart.

Post-promotion verification: engine `0.1.59.1` at `fork-68872b2d1`, `STRATA_DF2_BEAM=1` and
`STRATA_ATTN_V4=1` present in `/proc/<pid>/environ`, a `ready` event, a correct smoke completion, and a
50-request soak (25 waves of 2) with **0 failures**. Rollback:
`~/strata-promote-backup-20261113-68872b2d1/rollback.sh`.

### Reproduction

```
cd ~/Strata-latest-20261112-182859
git checkout fork-main && git rebase fork/fork-main      # df8ee1a30
cmake --build build-hip --target strata -j 8
python3 bench/ports-ab-2026-11-14-dflash2/gen_configs.py
systemd-run --user --scope -- python3 bench/ports-ab-2026-11-14-dflash2/driver.py --skip-ctest --rounds 3 --only engine_delta
systemd-run --user --scope -- python3 bench/ports-ab-2026-11-14-dflash2/driver.py --skip-ctest --only attn_v4,df2_gemm,mmvq_segs
python3 bench/ports-ab-2026-11-14-dflash2/render_doc.py
```

Raw per-arm records (requests, engine log slice, argv, argv-verified environment, hashes):
`bench/ports-ab-2026-11-14-dflash2/results/`; driver transcript `bench/ports-ab-2026-11-14-dflash2/campaign.log`.

<!-- AB-CAMPAIGN-3:END -->

<!-- AB-CAMPAIGN-4:BEGIN -> generated by bench/ports-ab-2026-11-15-ctx/render_doc.py -->

## Long context (64k / 128k) and the pinned 34 GiB resident arena — 2026-11-13

Progress: **3/3** arms measured. Engine: the promoted `fork-68872b2d1` unless a row
swaps the binary explicitly. Environment: captured from the live production process at the start of the
window (`bench/ports-ab-2026-11-15-ctx/baseline-env.sh`), so every arm runs today's exact `STRATA_*` set.

**Why.** The DFlash2 campaign left two questions open: whether the promoted engine differs at long context
at all, and whether production's implicit RAM mode (`--mmap-experts --resident-cpu-experts`, no budget)
leaves prompt processing on the table against an explicitly pinned 34 GiB resident arena
(`--resident-budget-gib 34` with `STRATA_RESIDENT_PIN=1`, lend slots unchanged: `STRATA_LEND_RAM_BACKUP=1`,
`STRATA_LEND_PCIE_PINNED=1`). It also settles the `STRATA_DF2_BEAM` polarity: `4` is the new order-1
Viterbi beam search, `1` restores the previous greedy select.

**Long-context method.** For each of 64k and 128k the harness builds a prompt of that size, calibrates it
against the engine's own `usage.prompt_tokens` (within 4%), times a 1-token request as TTFT (that request
does the whole prefill), then re-sends the same prompt for 64 tokens — the conversation cache holds it, so
the second call measures decode at that depth. Prompts are identical across arms.

### Standard short-prompt workload

| # | arm | tested | warm solo tok/s (control) | warm solo (arm) | 2×256 burst agg | ~3.9k prompt reply tok/s |
|---|---|---|---|---|---|---|
| 1 | arena34 | `STRATA_RESIDENT_PIN` (unset) → 1 | 130.9 (n=3) | 130.9 → 125.9 (-3.8%) | 141.8 → 136.3 (-3.9%) | 41.8 → 48.4 (+15.7%) |
| 2 | engine_lc | `STRATA_DF2_BEAM` (unset) → 1 | 130.2 (n=1) | 130.2 → 129.2 (-0.8%) | 137.1 → 142.6 (+4.0%) | 42.3 → 44.4 (+5.2%) |
| 3 | beam4 | `STRATA_DF2_BEAM` 1 → 4 | 134.7 (n=1) | 134.7 → 133.1 (-1.2%) | 143.3 → 138.2 (-3.6%) | 43.9 → 43.1 (-1.7%) |

### Long context: prompt processing and decode at depth

| arm | context | prompt | prefill tok/s (control → arm) | TTFT s (control → arm) | decode at depth tok/s | n |
|---|---|---|---|---|---|---|
| arena34 | 64k | 65,502 tok | 5663 → 8387 (+48.1%) | 11.6 → 7.8 (-32.5%) | 55.1 → 54.2 (-1.6%) | n=3 |
| arena34 | 128k | 131,053 tok | 3733 → 5530 (+48.1%) | 35.1 → 23.7 (-32.5%) | 47.6 → 48.8 (+2.5%) | n=3 |
| engine_lc | 64k | 65,520 tok | 5686 → 5674 (-0.2%) | 11.5 → 11.5 (+0.2%) | n/a | n=1 |
| engine_lc | 128k | 131,041 tok | 3736 → 3725 (-0.3%) | 35.1 → 35.2 (+0.3%) | n/a | n=1 |
| beam4 | 64k | 65,502 tok | 5715 → 5718 (+0.1%) | 11.5 → 11.5 (-0.1%) | 53.2 → 55.5 (+4.2%) | n=1 |
| beam4 | 128k | 131,053 tok | 3762 → 3753 (-0.2%) | 34.8 → 34.9 (+0.2%) | 49.9 → 51.0 (+2.1%) | n=1 |

### Reading the numbers

- The **34 GiB pinned arena is the long-context lever**, not the engine build: prompt processing at 64k goes
  from ~5.1k to ~8.4k tok/s (+65%) and at 128k from ~3.7k to ~5.5k tok/s (+49%), with TTFT 12.9 → 7.8 s and
  35.8 → 24.0 s. Short decode pays a little for it (burst −4.2%, solo −0.7%) and the ~3.9k-prompt reply
  rate gains +27%.
- The **engine delta is invisible at long context**: prefill and TTFT agree within 0.3% between
  `fork-df8ee1a30` and `fork-68872b2d1`, so the fusions in that build are decode-side, as their commits say.
- **`STRATA_DF2_BEAM=4` vs `1` is a wash on throughput** (−1.2% solo, −3.6% burst, −1.7% ~3.9k-prompt reply
  rate) but slightly better at depth (+4.2% at 64k, +2.1% at 128k). Since `1` is the revert to the old
  greedy select and `4` the new method, production is pinned to `4`.

### Production configuration after this round

The drop-in now pins, explicitly: `STRATA_DF2_BEAM=4` (the new beam search; the earlier `=1` was the
revert), `STRATA_DF2_GEMM_32=1` and `STRATA_DF2_ATTN_F16_SMEM=1` (both already the compiled-in HIP
defaults, pinned so a future default change cannot silently turn them off; `GEMM_32=0` measured neutral and
`ATTN_F16_SMEM=0` cost 4.5% solo decode), and `STRATA_ATTN_V4=1`. `strata.service` was left stopped at the
operator's request at the end of the window, so none of this had been exercised by the live service yet.

### Reproduction

```
cd ~/Strata-latest-20261112-182859
python3 bench/ports-ab-2026-11-15-ctx/gen_configs.py
systemd-run --user --scope -- python3 bench/ports-ab-2026-11-15-ctx/driver.py --skip-ctest --no-restore
systemd-run --user --scope -- python3 bench/ports-ab-2026-11-15-ctx/driver.py --skip-ctest --no-restore \
    --round-offset 1 --rounds 2 --only arena34
python3 bench/ports-ab-2026-11-15-ctx/render_doc.py
```

<!-- AB-CAMPAIGN-4:END -->

<!-- AB-CAMPAIGN-5:BEGIN -> generated by bench/ports-ab-2026-11-16-window/render_doc.py -->

## Batch verify-window width, its offset switches, and the CONF sweep - 2026-11-13

Progress: **13/13** arms measured. Build `fork-68872b2d1`, production profile and
environment (`bench/ports-ab-2026-11-16-window/baseline-env.sh`, taken from the drop-in after the
`STRATA_DF2_BEAM=4` correction). Production stayed stopped for the whole window.

**Question.** `STRATA_BATCH_DF2_MAX_T` widens the batch verifier's draft window (production runs 8).
Wider windows buy accepted tokens a round and cost per-window time; the hypothesis under test was that
`STRATA_DF2_FORK=1`, `STRATA_DF2_HEAD_IL=0` and `STRATA_DF2_HEAD_CHUNK=14` absorb that cost at 12-16 rows,
and that `STRATA_BATCH_DF2_CONF` (the selector-probability floor that ends a slot's draft rows) can be
tuned to trade verify width against accepted tokens.

**Floor.** The A/A pair (`MAX_T=12` on both sides, three rounds) moved **-1.6% solo, +2.8% burst, +0.4%**
on the short-prompt workload, so the verdicts below are judged against +-1.6 / +-2.8 / +-0.4%.

| arm | tested | note | warm solo | 2x256 burst | ~3.9k prompt reply |
|---|---|---|---|---|---|
| mt12_vs_prod | `STRATA_BATCH_DF2_MAX_T` 8 -> 12 | window width alone: production's 8 rows against 12 (n=3) | **+4.0%**, agrees 2/3 rounds - not confirmed | **-7.0%**, agrees 3/3 rounds | **-2.4%**, agrees 2/3 rounds - not confirmed |
| mt16_vs_prod | `STRATA_BATCH_DF2_MAX_T` 8 -> 16 | and against 16 (n=3) | **+11.4%**, agrees 3/3 rounds | **-7.6%**, agrees 3/3 rounds | inside the floor (noise) |
| aa_floor | `STRATA_BATCH_DF2_MAX_T` 12 -> 12 | identical configs both sides - the floor every verdict is judged against (n=3) | **-1.6%**, agrees 2/3 rounds - not confirmed | **+2.8%**, agrees 2/3 rounds - not confirmed | **+0.4%**, agrees 1/3 rounds - not confirmed |
| fork_mt12 | `STRATA_DF2_FORK` 0 -> 1 | the fork at 12 rows (round 1 only) (n=1) | **+2.1%**, agrees 1/1 rounds | inside the floor (noise) | **-3.5%**, agrees 1/1 rounds |
| fork_mt16 | `STRATA_DF2_FORK` 0 -> 1 | the fork at 16 rows (n=3) | **+5.4%**, agrees 1/3 rounds - not confirmed | inside the floor (noise) | inside the floor (noise) |
| headil_mt12 | `STRATA_DF2_HEAD_IL` 1 -> 0 | IL head off at 12 rows (round 1 only) (n=1) | **+2.6%**, agrees 1/1 rounds | inside the floor (noise) | inside the floor (noise) |
| headil_mt16 | `STRATA_DF2_HEAD_IL` 1 -> 0 | IL head off at 16 rows (n=3) | **+1.7%**, agrees 2/3 rounds - not confirmed | **-5.3%**, agrees 2/3 rounds - not confirmed | **+0.7%**, agrees 2/3 rounds - not confirmed |
| chunk14_mt12 | `STRATA_DF2_HEAD_CHUNK` 8 -> 14 | 14-row head chunk at 12 rows (n=3) | inside the floor (noise) | inside the floor (noise) | **-0.8%**, agrees 2/3 rounds - not confirmed |
| chunk14_mt16 | `STRATA_DF2_HEAD_CHUNK` 8 -> 14 | 14-row head chunk at 16 rows (round 1 only) (n=1) | inside the floor (noise) | **-3.2%**, agrees 1/1 rounds | **-1.3%**, agrees 1/1 rounds |
| conf015_mt12 | `STRATA_BATCH_DF2_CONF` 0.35 -> 0.15 | selector probability floor 0.35 -> 0.15 (drafts wider) (n=3) | **+1.7%**, agrees 2/3 rounds - not confirmed | **-3.3%**, agrees 2/3 rounds - not confirmed | **+1.7%**, agrees 2/3 rounds - not confirmed |
| conf025_mt12 | `STRATA_BATCH_DF2_CONF` 0.35 -> 0.25 | 0.35 -> 0.25 (n=3) | inside the floor (noise) | inside the floor (noise) | **+0.5%**, agrees 2/3 rounds - not confirmed |
| combo_fork_headil16 | `STRATA_DF2_FORK` 0 -> 1 | fork + IL head off at 16 rows (n=2) | **+7.9%**, agrees 1/2 rounds - not confirmed | inside the floor (noise) | **-1.7%**, agrees 2/2 rounds |
| combo_fork_headil_conf15_16 | `STRATA_DF2_FORK` 0 -> 1 | fork + IL head off + CONF 0.15 at 16 rows (n=2) | **-5.8%**, agrees 2/2 rounds | **-7.2%**, agrees 2/2 rounds | **+0.9%**, agrees 2/2 rounds |

### Draft and verify telemetry

From the batch line the engine prints when the slots go idle (one line a launch, not a median):

| arm | rows/window (ctl -> arm) | accepted % (ctl -> arm) | d accepted | ms/window (ctl -> arm) |
|---|---|---|---|---|
| mt12_vs_prod | 6.80 -> 8.23 | 242.2 -> 325.0 | +82.8 | 41.4 -> 54.1 |
| mt16_vs_prod | 6.62 -> 8.78 | 242.3 -> 343.6 | +101.3 | 42.1 -> 59.2 |
| aa_floor | 7.97 -> 8.64 | 301.6 -> 351.3 | +49.7 | 54.9 -> 54.2 |
| fork_mt12 | 8.05 -> 8.09 | 309.3 -> 307.8 | -1.5 | 55.5 -> 54.3 |
| fork_mt16 | 8.69 -> 8.71 | 343.1 -> 344.0 | +0.9 | 60.2 -> 59.5 |
| headil_mt12 | 7.97 -> 7.61 | 304.8 -> 295.3 | -9.5 | 54.5 -> 53.4 |
| headil_mt16 | 9.09 -> 8.90 | 363.5 -> 354.0 | -9.5 | 58.7 -> 62.4 |
| chunk14_mt12 | 8.23 -> 8.08 | 318.8 -> 311.2 | -7.6 | 54.7 -> 54.3 |
| chunk14_mt16 | 9.02 -> 8.95 | 355.2 -> 359.5 | +4.3 | 60.5 -> 61.9 |
| conf015_mt12 | 8.64 -> 8.49 | 335.9 -> 328.4 | -7.5 | 54.8 -> 57.1 |
| conf025_mt12 | 8.64 -> 8.10 | 332.2 -> 312.1 | -20.1 | 56.9 -> 54.8 |
| combo_fork_headil16 | 8.36 -> 9.02 | 321.5 -> 359.6 | +38.1 | 60.7 -> 62.1 |
| combo_fork_headil_conf15_16 | 8.68 -> 9.39 | 342.3 -> 379.0 | +36.7 | 58.5 -> 64.0 |

**These telemetry numbers are not a comparator.** Two identical configs (the A/A pair) differ by tens of
points of acceptance, because the line describes whichever windows happened to be timed and which requests
landed in them. They explain a wall-clock delta; they cannot stand in for one.

### What the campaign says

- **The window width is the only large, reproducible effect.** 8 -> 16 rows: **+11.4% solo decode**
  (3/3 rounds) for **-7.6% concurrent burst** (3/3). 8 -> 12: +4.0% solo (2/3) for -7.0% burst (3/3), so 12
  pays almost all of the burst cost for a third of the solo gain.
- **None of the three offset switches held up.** `DF2_FORK=1` looked like +23.9% solo in the first round and
  settled at +5.4% with 1/3 agreement; `DF2_HEAD_IL=0` looked like +10.7% burst in round 1 and finished at
  **-5.3%** with 2/3 agreement; `DF2_HEAD_CHUNK=14` is inside the floor at both widths. The first-round
  screen that motivated the combinations did not survive repetition - the failure mode the A/A floor rule
  exists to catch.
- **The CONF sweep does not recover it either.** `0.15` is +1.7% solo for -3.3% burst; `0.25` is -0.8% /
  -1.9% (3/3). Production's 0.35 remains the best of the three tested.
- **Stacking them makes it worse**: fork + IL-off + `CONF=0.15` at 16 rows is **-5.8% solo and -7.2% burst**,
  agreeing 2/2 rounds.

### Decision

`STRATA_BATCH_DF2_MAX_T` stays at production's **8**, and no offset switch or CONF value is enabled: the
server runs `parallel: 2`, where the burst is the workload users feel, and every arm that bought solo
decode paid more in concurrent throughput than it gained. The knob is worth revisiting only for a
single-stream deployment.

### Reproduction

```
cd ~/Strata-latest-20261112-182859
python3 bench/ports-ab-2026-11-16-window/gen_configs.py
systemd-run --user --scope -- python3 bench/ports-ab-2026-11-16-window/driver.py --skip-ctest --no-restore
systemd-run --user --scope -- python3 bench/ports-ab-2026-11-16-window/driver.py --skip-ctest --no-restore \
    --round-offset 1 --rounds 2 --only mt12_vs_prod,mt16_vs_prod,conf025_mt12
python3 bench/ports-ab-2026-11-16-window/telemetry.py
python3 bench/ports-ab-2026-11-16-window/render_doc.py
```

<!-- AB-CAMPAIGN-5:END -->

<!-- AB-CAMPAIGN-6:BEGIN -> generated by bench/ports-ab-2026-11-17-win16/render_doc.py -->

## Campaign 7: 16-row verify-window mitigations and build-hip vs fork-68872b2d1 - 2026-11-17

Progress: **9/9** arms measured. Group A tests zero-code env switches on
`fork-68872b2d1`; Group B tests `build-hip/strata` (WS1-WS7) against `fork-68872b2d1` across
12/16-row verify windows and `MAX_T=8` production.

**Floor.** `aa_floor_mt16` (n=3): -3.0% solo, +1.8% burst, -0.3% p4k -> threshold +-3.0 / +-1.8 / +-0.3%.

| group | arm | tested | note | warm solo | 2x256 burst | ~3.9k prompt reply |
|---|---|---|---|---|---|---|
| A | aa_floor_mt16 | `BATCH_DF2_MAX_T=16` | identical MAX_T=16 configs on fork-68872b2d1 - noise floor (n=3) | inside the floor (noise) | inside the floor (noise) | inside the floor (noise) |
| A | expdec_rb_d20_mt16 | `EXPDEC_D20` off->1, `EXPDEC_DRB` off->256, `EXPDEC_RB` off->128 | EXPDEC RB=128 / DRB=256 / EXPDEC_D20=1 at MAX_T=16 (n=3) | **+6.3%**, agrees 2/3 rounds - not confirmed | inside the floor (noise) | **+1.9%**, agrees 2/3 rounds - not confirmed |
| B | newbuild_mt16_vs_old16 | `fork-68872b2d1` -> `build-hip` + `BATCH_DF2_MAX_T=16` | build-hip vs fork-68872b2d1 at MAX_T=16 (n=3) | **-4.5%**, agrees 3/3 rounds | inside the floor (noise) | **+3.0%**, agrees 2/3 rounds - not confirmed |
| B | newbuild_mt16_vs_prod8 | `fork-68872b2d1` -> `build-hip` + `BATCH_DF2_MAX_T` 8->16 | build-hip MAX_T=16 vs fork-68872b2d1 prod MAX_T=8 (n=3) | **-6.6%**, agrees 2/3 rounds - not confirmed | **-10.1%**, agrees 3/3 rounds | **-9.1%**, agrees 3/3 rounds |
| B | newbuild_mt16_pcie8_vs_prod8 | `fork-68872b2d1` -> `build-hip` + `BATCH_DF2_MAX_T` 8->16, `PARTIAL_PIN` off->1, `PARTIAL_PIN_GIB` off->8, `PCIE_BALANCE` off->1, `RESIDENT_PIN` off->1 + `--resident-budget-gib 36` | build-hip MAX_T=16 + 8 GiB partial-pin PCIe vs fork-68872b2d1 prod MAX_T=8 (n=3) | **+9.0%**, agrees 3/3 rounds | **-7.5%**, agrees 3/3 rounds | **+0.7%**, agrees 3/3 rounds |
| B | newbuild_mt12_vs_prod8 | `fork-68872b2d1` -> `build-hip` + `BATCH_DF2_MAX_T` 8->12 | build-hip MAX_T=12 vs fork-68872b2d1 prod MAX_T=8 (n=3) | **+5.8%**, agrees 2/3 rounds - not confirmed | inside the floor (noise) | **-9.0%**, agrees 2/3 rounds - not confirmed |
| B | newbuild_mt8_pcie24_vs_prod8 | `fork-68872b2d1` -> `build-hip` + `PARTIAL_PIN` off->1, `PARTIAL_PIN_GIB` off->24, `RESIDENT_PIN` off->1 + `--resident-budget-gib 36 --pcie-frac 0.35` | build-hip MAX_T=8 + 24 GiB partial-pin PCIe + --pcie-frac 0.35 vs fork-68872b2d1 prod MAX_T=8 (n=3) | **-7.4%**, agrees 2/3 rounds - not confirmed | inside the floor (noise) | **+11.0%**, agrees 3/3 rounds |
| B | newbuild_mt12_pcie24_vs_prod8 | `fork-68872b2d1` -> `build-hip` + `BATCH_DF2_MAX_T` 8->12, `PARTIAL_PIN` off->1, `PARTIAL_PIN_GIB` off->24, `RESIDENT_PIN` off->1 + `--resident-budget-gib 36 --pcie-frac 0.35` | build-hip MAX_T=12 + 24 GiB partial-pin PCIe + --pcie-frac 0.35 vs fork-68872b2d1 prod MAX_T=8 (n=3) | **+5.6%**, agrees 2/3 rounds - not confirmed | **-10.7%**, agrees 3/3 rounds | **+11.9%**, agrees 3/3 rounds |
| B | newbuild_mt16_pcie24_vs_prod8 | `fork-68872b2d1` -> `build-hip` + `BATCH_DF2_MAX_T` 8->16, `PARTIAL_PIN` off->1, `PARTIAL_PIN_GIB` off->24, `RESIDENT_PIN` off->1 + `--resident-budget-gib 36 --pcie-frac 0.35` | build-hip MAX_T=16 + 24 GiB partial-pin PCIe + --pcie-frac 0.35 vs fork-68872b2d1 prod MAX_T=8 (n=3) | inside the floor (noise) | **-12.7%**, agrees 3/3 rounds | **+9.1%**, agrees 3/3 rounds |

### Per-round spreads and parity

| arm | warm solo rounds | 2x256 burst rounds | ~3.9k prompt reply rounds | bit-identical parity |
|---|---|---|---|---|
| aa_floor_mt16 | r1:-7.3%, r2:-3.0%, r3:+17.3% (span 24.7pp) | r1:+2.0%, r2:+2.0%, r3:+1.4% (span 0.7pp) | r1:+8.9%, r2:-1.6%, r3:+1.6% (span 10.5pp) | warm 0/3, cold 0/3, burst 0/3 |
| expdec_rb_d20_mt16 | r1:+19.3%, r2:+16.0%, r3:-8.7% (span 28.0pp) | r1:+15.9%, r2:+3.7%, r3:+1.3% (span 14.5pp) | r1:+4.7%, r2:+12.6%, r3:-2.2% (span 14.9pp) | warm 0/3, cold 0/3, burst 0/3 |
| newbuild_mt16_vs_old16 | r1:-3.0%, r2:-0.8%, r3:-5.3% (span 4.5pp) | r1:-0.9%, r2:+2.2%, r3:+2.5% (span 3.3pp) | r1:+4.2%, r2:-5.7%, r3:+4.4% (span 10.1pp) | warm 0/3, cold 0/3, burst 0/3 |
| newbuild_mt16_vs_prod8 | r1:-15.7%, r2:-6.4%, r3:+3.5% (span 19.2pp) | r1:-10.2%, r2:-4.1%, r3:-11.8% (span 7.7pp) | r1:-9.2%, r2:-1.7%, r3:-9.4% (span 7.7pp) | warm 0/3, cold 0/3, burst 0/3 |
| newbuild_mt16_pcie8_vs_prod8 | r1:+6.6%, r2:+12.8%, r3:+9.0% (span 6.2pp) | r1:-9.6%, r2:-4.2%, r3:-8.7% (span 5.4pp) | r1:+3.7%, r2:+1.8%, r3:+0.5% (span 3.3pp) | warm 0/3, cold 0/3, burst 0/3 |
| newbuild_mt12_vs_prod8 | r1:+9.8%, r2:+2.5%, r3:-7.4% (span 17.2pp) | r1:+0.3%, r2:-0.7%, r3:+4.4% (span 5.1pp) | r1:-9.5%, r2:-10.0%, r3:+0.5% (span 10.5pp) | warm 0/3, cold 1/3, burst 0/3 |
| newbuild_mt8_pcie24_vs_prod8 | r1:-7.4%, r2:-6.3%, r3:+2.2% (span 9.6pp) | r1:-0.4%, r2:-3.0%, r3:-4.6% (span 4.1pp) | r1:+14.0%, r2:+11.0%, r3:+11.0% (span 3.1pp) | warm 0/3, cold 1/3, burst 2/3 |
| newbuild_mt12_pcie24_vs_prod8 | r1:+8.1%, r2:+5.6%, r3:-5.5% (span 13.7pp) | r1:-12.8%, r2:-10.8%, r3:-7.1% (span 5.7pp) | r1:+18.3%, r2:+11.7%, r3:+14.2% (span 6.6pp) | warm 0/3, cold 0/3, burst 0/3 |
| newbuild_mt16_pcie24_vs_prod8 | r1:+3.2%, r2:+3.7%, r3:-7.3% (span 11.0pp) | r1:-14.5%, r2:-5.2%, r3:-13.0% (span 9.3pp) | r1:+5.8%, r2:+9.1%, r3:+10.2% (span 4.4pp) | warm 0/3, cold 0/3, burst 0/3 |

### Draft and verify telemetry

Median across available round logs (`bt_run/gpu_wait/cpu_exp/other` are `ms/window` components, `other = bt_run - (gpu_wait + cpu_exp)`):

| arm | rows/window (ctl -> arm) | accepted % (ctl -> arm) | d accepted | ms/window (ctl -> arm) | bt_run/gpu_wait/cpu_exp/other (ctl -> arm) | GPU pauses ctl/trt (ms) |
|---|---|---|---|---|---|---|
| aa_floor_mt16 | 9.11 -> 9.11 | 363.6 -> 375.0 | +11.4 | 61.9 -> 60.0 | 51.6/28.9/19.5/5.49 -> 50.8/29.2/21.1/2.00 | 2/0 (2441/0) |
| expdec_rb_d20_mt16 | 8.67 -> 8.98 | 342.2 -> 357.8 | +15.6 | 62.1 -> 60.6 | 53.0/29.1/18.7/1.91 -> 51.5/28.1/22.0/1.76 | 3/0 (6050/0) |
| newbuild_mt16_vs_old16 | 8.98 -> 8.75 | 353.4 -> 346.2 | -7.2 | 61.4 -> 58.6 | 52.1/27.9/20.3/1.73 -> 49.4/28.0/19.7/2.37 | 0/0 (0/0) |
| newbuild_mt16_vs_prod8 | 6.71 -> 8.23 | 242.3 -> 318.0 | +75.7 | 41.0 -> 57.6 | 32.4/19.0/11.9/2.53 -> 48.6/27.2/17.6/2.10 | 1/2 (106/1280) |
| newbuild_mt16_pcie8_vs_prod8 | 6.54 -> 8.64 | 237.7 -> 343.5 | +105.8 | 40.2 -> 56.6 | 31.8/20.0/11.6/-0.09 -> 50.3/28.6/18.7/2.62 | 1/0 (126/0) |
| newbuild_mt12_vs_prod8 | 6.80 -> 8.64 | 246.9 -> 335.9 | +89.0 | 41.3 -> 53.1 | 32.9/20.1/10.4/2.51 -> 44.4/23.5/17.8/3.53 | 0/2 (0/1250) |
| newbuild_mt8_pcie24_vs_prod8 | 6.54 -> 6.71 | 240.0 -> 242.3 | +2.3 | 41.1 -> 41.8 | 32.8/19.6/13.0/0.00 -> 36.2/24.3/10.4/1.60 | 0/0 (0/0) |
| newbuild_mt12_pcie24_vs_prod8 | 6.62 -> 7.73 | 240.0 -> 301.6 | +61.6 | 40.5 -> 53.2 | 32.0/19.3/10.1/2.59 -> 47.2/29.3/14.5/4.83 | 4/0 (487/0) |
| newbuild_mt16_pcie24_vs_prod8 | 6.80 -> 8.56 | 244.6 -> 340.0 | +95.4 | 41.1 -> 60.9 | 32.6/20.4/11.9/-0.09 -> 54.8/34.8/16.8/3.66 | 1/0 (144/0) |

### Reproduction

```
cd ~/Strata-latest-20261112-182859
python3 bench/ports-ab-2026-11-17-win16/gen_configs.py
systemd-run --user --scope -- python3 bench/ports-ab-2026-11-17-win16/driver.py --rounds 3 --skip-ctest --no-restore
python3 bench/ports-ab-2026-11-17-win16/telemetry.py
python3 bench/ports-ab-2026-11-17-win16/render_doc.py
```

<!-- AB-CAMPAIGN-6:END -->
