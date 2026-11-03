# Hardware-path stall prevention and the AVX-512 CPU path (research run, 2026-11-02)

Tree `E:\dev\Strata` @ `integ/community-2026-10` (HEAD `045737d2`), strictly read-only. Host: **Ryzen 9 9950X3D**
(Zen 5, 8C/16T) + **RTX 5080**, Windows, live service `strata-swift-iq3_xxs` (engine 0.1.52 = build `045737d2`,
decode 161.9–162.3 tok/s). Method: two parallel code audits (GPU/PCIe overlap; CPU kernels), a compiled CPUID +
intrinsic probe on this host, the live engine log, and llama.cpp (ggml-cpu) as the external reference via Context7.
Every claim carries a provenance tag: `(code file:line)`, `(docs/…)`, `(log …)` = this service's engine log,
`(probe)` = measured on this host by this run, `(rig:gfx1201)` / `(rig:A770)` = the cited docs' machines, not this one.

## 1. What already overlaps

The engine is far from single-stream. The overlap machinery in the decode/prefill paths:

| Mechanism | Where | What it overlaps | Default |
|---|---|---|---|
| Side-stream branches for GDN gates, gate projection, indexer, queries (`side_`, `side2_`, `mjoin_`, `pjoin_`) | `src/core/verify.cpp:1917, 1934, 1981, 2205, 2361` (code) | branch work vs the main verify chain | on when the graph builds them |
| Shared-expert stream fork (`sh_cs_`, `ev_fork_/ev_join_`) | `verify.cpp:2418-2548`; gate `verify.cpp:388`; arch table `src/core/arch_defaults.cpp:83` (code) | shared-expert GEMV vs routed-MoE | `STRATA_SH_STREAM` per-arch: **on** Linux gfx1151/gfx12 (+1.8/+6.7%), off Windows gfx1151, **unmeasured on CUDA** |
| MTP draft pipelining (mapped-pinned `h_out_` + `ev_step_/ev_chain_` spin, one sync at tail) | `src/core/mtp.cpp:2272, 1895` (code) | host token consumption vs draft graph | on |
| Verifier copy stream (`copy_`) vs compute (`cs_`) | `verify.cpp:860, 3786, 4963` (code) | expert uploads vs GEMVs | on; synced at window end |
| PCIe expert staging with per-buffer events (`stage_ev_`, per-card streams) | `src/core/expert_source.cpp:2154, 6818` (code) | pinned staging copies vs compute | on |
| Async adaptation stream (`adapt_stream`, `--adapt-async`) | `src/program/generate.cpp:11955-12093` (code) | expert VRAM/RAM swaps vs decode windows | self-decides on for serve (this host: **blocking**, log 45-46) |
| Prefill↔MTP parity events (`pl_mtp_ev[l&1]`), prefill→slot KV publish (`pm_copied`) | `generate.cpp:15878, 14048` (code) | MTP draft vs next prefill chunk | on |
| Conversation-snapshot side stream (`sn.side`, `pl_snap_overlap`) | `generate.cpp:17062, 17464` (code) | snapshot writes vs decode | on when taken |
| Mapped-pinned doorbell flags (kernel sequence numbers, flag A/B, `wait_flag_ge_kernel`) | `src/core/layer.cpp:139`, `verify.cpp:5020`, `src/core/pinned.cu` (code) | host↔device progress without stream syncs | core, always on in the pool path |
| Cold-start file overlap + I/O prefetch threads (`STRATA_COLD_OVERLAP`, `STRATA_IO_PREFETCH`) | `generate.cpp:6661-6680` (code) | expert blob reads vs first-request decode | on under Linux file defaults (decode +20-34%) |
| Helper-GPU refill event (`refill_ev_`) | `src/core/peer_experts.cpp:448, 490` (code) | helper cache refill vs primary decode | on with peer tiers (experimental) |

## 2. Stall inventory (ranked)

1. **CRITICAL — GPU→CPU-pool expert reach.** The GPU spins (`wait_flag_ge_kernel`, 26 launches/token, 4% of kernel
   time) waiting for CPU-computed experts. On the 4-way batch rig, 80/85 long stalls and 74% of the 44 ms window are
   this reach-wait; `mtp_select → copy_i32_from_mapped` is the attribution `(rig:gfx1201 docs/DECODE-TRACE,
   docs/BATCH-WINDOW-SYNC, commit 5ea9034b)`. This host: pool 50–55 µs per missed expert = 31.8–35.0 GB/s,
   6–15% of routed work lands on the pool after `pcie_frac` auto-refines 0.55→0.45 (log 69, 80). Pool sites:
   `verify.cpp:5020` flag protocol, `pinned.cu:1005`.
2. **HIGH — launch-bound decode.** ~1,100 dispatches/token, median kernel 3.6 µs; traced GPU idle 33.6%, 76.5% of it
   in <20 µs inter-launch gaps `(rig:gfx1201 docs/DECODE-TRACE-2026-10-30.md:30-35)` — a stable engine property
   across two captures; arithmetic is not the constraint. Biggest launch families: quantize/interleave 171/token,
   `gr_down_q8_fast` ~51/token, sync/doorbell kernels ~90/token.
3. **HIGH — serial per-slot co-draft (emit).** Each slot's MTP draft ends in a full `cudaStreamSynchronize(cs_)`
   (`mtp.cpp:1909`, confirmed) → ~0.75 ms/slot ≈ 2.2 ms of a 40-43 ms window at `--batch 4 --batch-mtp` (rig:gfx1201).
4. **HIGH — commit sync per window.** `commit_slot_prefixes` syncs `cs_` and `copy_` once per window
   (`verify.cpp:5028-5030`, confirmed); the next co-draft reads commit state, so the 2.2 ms commit GPU time is on the
   critical path; the reverted `STRATA_BATCH_COMMIT_ASYNC` moved the wait into emit, it did not remove it (311b49d4).
5. **MEDIUM — window-edge flag-B guard** syncs `cs_` then `copy_` between windows (`verify.cpp:3815-3827`, confirmed);
   **helper-refill blocking fallback** `peer_experts.cpp:425`; **remote-stage** serializes by design (one TCP round
   trip per window, `peer_link.cpp:444`).
6. **LOW — stage-buffer refill** (`expert_source.cpp:1989` query-miss hard-sync) — rare by pool sizing.
7. **MEASUREMENT GATE — determinism.** Greedy single requests differ across fresh launches of identical code
   (placement timing + pool reduce order) `(docs/BATCH-WINDOW-SYNC §Determinism)`. **Every async/overlap A/B claim is
   currently vetoed on "bits differ" — fix this before any window-overlap change.**

## 3. Features gated OFF, and why (the gate-measure culture)

| Feature | Gate | Why off |
|---|---|---|
| PCIe-wait fold (fold expert-copy wait into fused quantize) | `STRATA_FUSE_PCIE_WAIT=1`, default off every backend (8f731ff2) | −3-4% at 4-way on gfx1201 (99.2/114.0 → 95.5/109.5 tok/s) |
| Async batch commit | fully reverted (311b49d4) | commit 2.2→0.02 ms but emit +3.5-4.3 ms; window unchanged |
| Batched co-draft group (one graph, 2 rows/item) | `STRATA_BATCH_DRAFT_GROUP=1` (177038db) | ~0.8 ms/window gained, but ~3% drafted bits differ, acceptance 70.2→68.3% |
| Two-phase co-draft | `STRATA_BATCH_DRAFT_TWOPHASE=1` | no wall reduction (2.05-2.53 vs 2.2 ms) |
| Batched DMA (`cudaMemcpyBatchAsync` + PreferOverlap hint) | `STRATA_DMA_BATCH=1|2`, default 0 `(docs/BATCHED_DMA.md)` | submission −82.7%, transfer −1.3%, no tok/s gain (30.3 vs 30.0) |
| Adaptive-tier copy overlap | `STRATA_ADAPT_OVERLAP=1`, opt-in since 0.1.43.1 | gate-measured on Linux CUDA only |
| Disjoint expert slots (2 GPUs) | `STRATA_DISJOINT_ADAPT=1` | historical +15.5% on 2×RTX3080; port is an unvalidated draft PR |
| Exchange-buffer rotation (no host commit memcpy per swap) | `STRATA_EXCHANGE_ROTATE=1` | needs fully pinned+mapped complement; no promotion run |
| PCIe auto-split / measured-frac / per-layer balance | `STRATA_PCIE_AUTO_SPLIT`, `PCIE_FRAC_DEFAULT=auto`, `PCIE_BALANCE` (HIP) | author gate-measured on own rig only (`generate.cpp:2187, 2222-2239, 4314`) |
| Router bias favoring resident experts | `STRATA_ROUTE_RESIDENT_MTP=1` EXPERIMENTAL | changes output (`generate.cpp:10084-10093`) |
| Prefill CPU share | CUDA-only, 1 GPU, no batch, chunk <1024 (`prefill.cpp:218-265`) | **ON here** (log: prompt path borrows 2646 slots / 4.25 GiB) |

## 4. CPU path on Zen 5 (AVX-512) — audit

**Dispatch** `(code, all confirmed in tree)`: runtime CPUID gates in `src/kernels/cpu/expert_layout.cpp:75-312`
(`cpu_avx512_ok` = F+BW+VL+VNNI+VBMI+XCR0; `cpu_avx512bw_ok`; `cpu_avxvnni_ok`; banner `cpu_features()`; refusal
from `generate.cpp:4095`); per-file MSVC flags (`CMakeLists.txt:2088-2123, 2197-2238`; instantiated in
`build-cu130/build.ninja`): `iq_avx512/nvfp4_avx512/tq_avx512/expert.cpp` `/arch:AVX512`; `iq_avx2/q2_avx2`
`/arch:AVX2 -DSTRATA_AVXVNNI=1`; per-expert pick in `native_expert.cpp:352-575`, phase dispatch `pool.cpp:1219-1250`.

**(probe) This host admits everything**: leaf7.0 ECX bit11 AVX512-VNNI=1, VBMI=1, VPOPCNTDQ=1; leaf7.1 EAX bit4
AVX-VNNI=1; leaf7.0 EBX F/DQ/IFMA/CD/BW/VL=1; XCR0=0xE7; `cpu_avx512_ok()`/`cpu_avx5vnni_ok()` TRUE — the
512-bit kernels do run. `cpu_gather_fast()` is **FALSE**: `expert_layout.cpp:260-262` hard-requires `GenuineIntel`,
so the 512-bit **single-token** exact rows (`native_expert.cpp:415-417` "x512_one") and the gathered grid decode
(`iq_avx512.cpp:135,160`) never run on Zen 5. `STRATA_POOL_UMWAIT` unusable: WAITPKG=0, AMD MWAITX present, unused.

**The hot-path truth.** The live pack `(Strata-data/packs/swift-iq3_xxs/native_experts.txt)`: 48 expert layers ×
512 experts, 42.91 GB; gate/up = 13 IQ3_S + 10 IQ2_S + 10 IQ2_XS + 9 IQ2_XXS + 6 IQ3_XXS; down = **30 Q2_0 + 18
IQ4_NL**. A window's expert groups hold **~1.4 tokens** (`native_expert.cpp:432-434`), so `nt==1` is the decode case
— and `#152 mt_min=2` sends `nt==1` IQ groups to **ggml's 256-bit AVX2 one-token dot** (`quants.c:3260/3273`,
`assert(nrc==1)`, scalar `iq3xxs_grid` lookups, **no VNNI**): `GGML_AVX512_VNNI=OFF`, `GGML_AVX_VNNI=OFF`,
`GGML_CPU_ALL_VNNI→CPU_ALL_VARIANTS=OFF` in `build-cu130/CMakeCache.txt`, and MSVC `/arch:AVX512` defines no
`__AVX512VNNI__` macro `(probe)`, so every guarded ggml micro-kernel is compiled out.
512-bit VNNI **is** in-tree — `expert.cpp:608/650/700` (Q2_0, the kernel carrying 30/48 down layers, ~39 GB/s
compute-bound ceiling, header note 144-150), `nvfp4_avx512.cpp:120`, `tq_avx512.cpp:70-73` — **never in the IQ3/IQ2
dot**: `iq_avx512.cpp:208` is `add(madd(maddubs(g,ys),sc))` (confirmed). AVX-VNNI (256-bit) exists in
`iq_avx2_rows.inl:12-27` behind `cpu_avxvnni_ok()`, reached only when `!cpu_avx512_ok()` — i.e. not on this host.

**Pool micro-architecture (this host)**: 7 workers on logical 2,4,…,14 (one per physical core, host on 0) (log 21-24);
workers spin (`_mm_pause`; "no WAITPKG" path, `pool.cpp:38-54, 982-1017`); host spins `wait_parked/wait_done` with the
issue-#29 abort (`pool.cpp:1128-1180`); NUMA split is Linux-only ("the split needs Linux", `pool.cpp:629-640`);
zero OpenMP (`GGML_OPENMP=OFF`, 0 pragma sites); pool streams at 31.8–35.0 GB/s vs the machine's ~52 GB/s —
TLB/page-fault shaped on 4 KB pages (log 69/80; `iq_avx512.cpp:48-49`).

## 5. Opportunities, ranked for this host

1. **Determinism + fixed-composition harness** (gate on all the rest). Site: placement tiers (`generate.cpp`
   `hits.d_res`, `expert_source.cpp` routing) + `tools/ab/batch_burst_bench.py --same-prompts`.
2. **PCIe-miss share rounding** — the pool-vs-PCIe misrouting behind stall 1: the share "rounds to 0 at 1-2 misses",
   so misses default to the CPU pool (~1 ms/expert) instead of PCIe (0.15-0.31 ms/expert). Accumulate the fractional
   share across layers. Site `generate.cpp:2187-2433` (`PcieAuto`, kPoolFactor 1.85) + `expert_source` dispatch;
   `(docs/ARCH-EXPERT-MEMORY-2026-10-19 §1,§4)`. Gate-measure per the PCIe-wait-fold lesson.
3. **Batch co-draft group + async commit as a pair.** The group (2.13→1.30 ms/window measured) plus an async commit
   that syncs only at slot release removes both 2.2 ms buckets; the pair works only if the group draft's read set
   excludes commit writes. Sites `mtp.cpp draft_group/capture_group (2140+)`, `verify.cpp:5028-5030`,
   `generate.cpp batch_step`. Precondition: multi-row kernel byte-parity matrix (router GEMV+top-k, MoE combine,
   residual, head-mix) — `STRATA_NO_MULTI_GR` shows a kernel switch can zero acceptance.
4. **512-bit VNNI for the IQ dot (Zen 5).** Replace `iq_avx512.cpp:208` (and `:387, 449-450`) with
   `_mm512_dpbusd_epi32(acc, g_signed, sc_u8)` — the code already makes the activation unsigned with
   `_mm512_mask_sub_epi8` (199-207), exactly dpbusd's operand shape; keep ggml's 8-lane fold (`fold256` 234-243).
   Integer sums → parity-checkable with the existing harness; gate on the VNNI bit `expert_layout.cpp:84-86` already
   tests; pattern proven in-tree (`nvfp4:120`, `expert.cpp:608`). One uop replaces 3 vector-ALU uops per 64 values;
   raises the ALU ceiling above the measured 30-35 GB/s stream.
5. **Stop routing single-token CPU experts to ggml's 256-bit dot.** (a) Mirror the DSV41 path's measured Zen 5 rule
   — `ds4_gg_cpu.cpp:181-201 want_512()` enables the bit-identical 512-bit exact rows for AMD family ≥ 0x1A with
   per-format one-token gains (IQ2_XXS 35→44 GB/s, IQ2_XS 37→48, ×2.5-3.5 at 4 tokens) — into
   `native_expert.cpp:415` (today: `iq512_gather_on()` = Intel-only) and add the AMD branch to `cpu_gather_fast()`
   (`expert_layout.cpp:258-297`; keep Zen 4's measured −3-5% gather result as negative control, measure Zen 5).
   (b) Relax `#152 mt_min` to 1 for IQ2_XXS/IQ2_XS (20 of 48 gate/up layers are those types) where a
   window-invariant 512-bit row exists (`native_expert.cpp:352-372, 452`).
6. **Wire the down phase into 512-bit + add prefetch.** 18/48 down layers are IQ4_NL at nt==1 → ggml dot;
   `iq512_rows` (`iq_avx512.cpp:497-504`) has **no production caller**; `dot_rows_wide`'s 8×16 tiling + prefetch
   (407-451) is gate/up-only. Add per-row prefetch knobs (file's own `STRATA_IQ_PREFETCH` shows −2-3%: measure, don't
   assume) and none in `tq_avx512.cpp:48-80` / `expert.cpp`. Huge-page/pin coverage (`pin_ranges`,
   `expert_source.cpp:6577-6718`) is the TLB-shaped lever for the 31.8→52 GB/s gap.
7. **Enable ggml's VNNI dots** with a per-target preprocessor define (the `-DSTRATA_AVXVNNI=1` mechanism,
   `CMakeLists.txt:2100-2123`) — the intrinsics compile under MSVC `/arch:AVX512` `(probe)`; only the macro is
   missing. **Whole-build parity risk**: ggml is the reference bits for every native-kernel check
   (`native_expert.cpp:79-141`, `#2215` guard 395-401) — needs reply-level A/B, not just unit parity.
8. **Launch-count levers (GPU)**: fuse `gr_down_q8_fast` variants (~51 launches/token); merge the GPU-side sync
   kernels (`wait_flag_ge`/`wait_copy_i32`/doorbell, ~90/token, 5.3% kernel time); A/B `STRATA_QIL` +
   `STRATA_FUSE_PCIE_WAIT` + `STRATA_SH_STREAM` on CUDA — all three gates were set on AMD data and are
   **unmeasured on this RTX 5080**.
9. **Pool waits on AMD**: MWAITX (present `(probe)`) idle-with-timeout instead of `_mm_pause` spins
   (`pool.cpp:982-1017`); re-measure SMT siblings for the memory-bound IQ rows (siblings idle today; the i7-13700K
   "all cores" −2.4-3× result was hybrid-topology, not this host).
10. **BF16 dots** (`_mm512_dpbf16_ps`, present `(probe)`): `kq_avx2.cpp:228-280` widens bf16→f32 at 256-bit 8 values
    per FMA; `nvfp4_avx512.cpp:82-114` is the in-tree pattern. Not bit-exact vs f32 — opt-in knob, sum order is
    load-bearing (`kq_avx2.cpp:215-217`).

## 6. This host's A/B protocol (short)

Determinism first (§2.7), then one-variable arms on the live config; read the `strata batch:` line + `progress_at`
buckets (the 27% tracer inflation means every trace-derived idle figure is an upper bound — the reverted hipEvent
timers died because serve decode is not the instrumented generate loop; use batch telemetry hooks
`mtp.cpp:1897`). Candidate arms, cheapest first: `STRATA_IQ512_GATHER=1` + x512_one behind an AMD rule;
`STRATA_IQ_MT_MIN=1` for types 16/17; `STRATA_QIL=1`; `STRATA_FUSE_PCIE_WAIT=1`; `STRATA_SH_STREAM=1`;
`STRATA_DMA_BATCH=2`; `pcie_frac` rounding fix; `STRATA_EXCHANGE_ROTATE=1`. Note the live log's #2065 warning
(305 MiB VRAM free at request start): add `--vram-reserve-mib 1472` (or trim context) before perf claims, since
WDDM paging can dominate.

## 7. External references (Context7 → llama.cpp/ggml, fetched 2026-11-02)

- VNNI micro-kernels: `_mm512_dpbusd_epi32` / `_mm256_dpbusd_avx_epi32 / dpbssd` replace the
  `maddubs+madd(+sign)` chains in `ggml/src/ggml-cpu/arch/x86/repack.cpp` and `llamafile/sgemm.cpp`; ggml gates them
  on `__AVX512VNNI__ && __AVX512VL__` — exactly the macros MSVC omits here (§4). VNNI shifts IQ4_NL tinyBLAS ranking.
- Copy/compute overlap: ggml's SYCL backend uses N×8 MiB pinned staging slots so "the host copy of one slot overlaps
  the transfer of the previous" (`GGML_SYCL_UPLOAD_STAGING_SLOTS`), and documents that >2 GiB allocations serialize
  H2D with compute — the same slot-pool discipline Strata already applies in `expert_source` staging.
- Scheduler `op_offload` moves host-weight ops onto the GPU backend when a host buffer feeds them — Strata's
  mapped-flag doorbells are the ahead-of-ggml variant of the same idea.

## 8. Operational notes from the E: relocation (2026-11-02)

- `build/`, `build-cu130/`, `build-pr/` CMake caches were copied and still name `C:\Users\blsor\Strata` — a rebuild
  in any of them needs a **fresh configure** (matches the known stale-CMakeCache pitfall; re-check
  `STRATA_Q6K_EXPERTS` style options after reconfiguring). `build/` dates to 2026-10-11; audit `build-cu130`.
- Worktrees `E:\dev\Strata-wt\pr2318` (feat/kv-q2k-window) and `pr2509` (detached) were repointed
  (`.git`/`gitdir` both sides); `git worktree list` is clean.
- The scheduled task now serves `E:\dev\Strata\serve\server.py`; engine binary + model data stay on
  `C:\Users\blsor\Strata-data`.

Caveats: §2's window anatomy (40-43 ms, 74% reach, 6.3 ms CPU experts) is the gfx1201 R9700 4-way batch rig; this
host runs serial serve at 162 tok/s with 49→87% VRAM cache hits (log 82 and later). The A770 figures
(`waitA/waitB` ≈ 87% of window) are an Intel Arc box. Gate defaults are per-arch tables (`arch_defaults.cpp`) — the
CUDA column of every table in §3 is open experimental territory.
