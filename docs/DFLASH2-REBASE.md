# DFlash2 perf rebase onto upstream main — reconciliation plan

Date: 2026-11-02. Branch: `perf/dflash2-on-main` (worktree `E:\dev\Strata-wt\dflash2-main`, from `origin/main` = `ffa80d731`, 0.2.10 era). Source of the perf work: `integ/community-2026-10` (`a1cca420`); merge-base `870aedd9` (v0.2.8).

## Why this is not a cherry-pick sweep

21 dflash2 perf commits + 3 docs commits are fork-only (ancestry-tested), **and** upstream main rewrote DFlash2 on its own since v0.2.8 (`f-0291-df2-*` series: dflash2.cpp 559→1107 lines, dflash2.cu 794→1171). Main independently built: per-slot drafters sharing pack+head with stage-wise taps and solo↔slot ring handoff (f70368b99), deferred context with spare window rows + graph launch-then-finish (29d55bd71), ring log rebuild (96f522776), park/restore for batch slots, serve mode, Q4_0 selector codebooks + `STRATA_DF2_HEAD=native` (8ec2c8a91), Q4_0 GEMM row-variant tables (b353ce7d4), `--draft-fallback none` (09c407d8f), layer-split DFlash2 solo app rules (ffa80d731). Naive cherry-picks would fight upstream's implementations of the same ideas.

## Evidence grep (symbol → files in origin/main src/)

| symbol | main | verdict |
| --- | --- | --- |
| `batch_body_exec` / `STRATA_DF2_BATCHBODY` | 0 | **absent — port** |
| `STRATA_DF2_BATCHHEAD` | 0 | **absent — port** |
| `copy_slot` (D2D drafter) | 0 | **absent — port** |
| suffix co-draft batch / `draft_group` | 0 | **absent — port** |
| FP16 context K/V ring (`half` in ring) | 0 | **absent — port** |
| PLE async gather (HIP) | 0 | **absent — structural port** |
| `read_windows` | 2 files | main has its own — **reconcile** |
| park / `restore_batch` | 39 / 3 files | **covered — skip** |
| serve mode (generate.cpp) | present | **covered** |
| batch-slot drafters, ring handoff, graphs | f70368b99, 29d55bd71 | **covered — skip** |

## Port list (dependency order, each = one reconcile commit, gated)

Scope (user decision 2026-11-02): **perf/optimization work only**, applied to upstream's dflash2 implementation. Upstream dflash2 is already wired to serve — no infra, no features, no docs commits ride this branch.

EXCLUDED as not-applicable perf: serve-mode wiring (c245a6c30), 4k/8k windows (9e9569e46), park/restore (8b14892c4), n-gram co-draft + BATCHHEAD + BGEN fixups (448c0f836, c983d97d1, 996650eaa, 15d4684cf — fork serve-batch feature work), gfx12 rig defaults, measurement-record docs (556d7c611, 623df1216, 82ab07c89, f57bfc554, 5e463f9bc).

1. **Ring**: FP16 context K/V ring + uint2 register prefetch (`cf70f46ac`).
2. **Batching**: `STRATA_DF2_BATCHBODY` context projection + 5-layer body across slots (`2999c0b18`) + `Dflash2Drafter::copy_slot` (`5928dc613`) + `batch_body_exec_` graph capture (`b0be13a49`) — reconcile against main's 29d55bd71 graph launching in one pass.
3. **Verify overlap**: context projections overlap target verify layers 37–47 (`19e213789`).
4. **Window merges**: 16 trailing prompt tokens into first verify window (`922620781`); 16-row `read_windows` on one-card DFlash2 (`941ad664f`) — reconcile with main's read_windows.
5. **Head micro-perf**: head IL + register top-k from `3f647f45b` **only if** they apply on top of main's native-head Q4_0 path (8ec2c8a91) — A/B on the same card, default keep main; the fork's pruned-head/Q6-pack layout itself is NOT ported.

## Divergent — measure, don't port blind

- Fork pruned-head IL/Q6-pack + register top-k (`3f647f45b`) vs main's native-head Q4_0 codebook path (8ec2c8a91): A/B on the same card; default keep main.

## Disk conversation cache — recommendation: EXCLUDE (for now)

Main has the memory conversation cache (`conversation_cache_test.cpp`, `conversation_copy.hpp`) and the app knows `auto_sessions`; disk-mode is 4 fork commits (`ec95b2bf6` header, `0002309fa` reclaim guard, `a0153a027` generate.cpp, `fcd4a61d0` docs, adapted from community `1aa35f01`). The live service does use it (`keep8`), so the value question is real: quantify cold-restart reload hit-rate and RAM headroom gain first; if it earns, it lands as its own labeled series (cache.hpp + generate.cpp + app config surface), not inside the perf rebase.

## Gates per wave

CUDA build green (fresh configure in the worktree — caches from other dirs are path-stale), `dflash2_parity`, `dflash2_q4_parity`, `draft_cap_test`, `suffix_drafter_test`, then one-card serve smoke `--draft dflash2` on the swift model; bitwise where the harnesses promise bitwise. HIP-only hunks: structural merge now, gfx1201 rig gate recorded as deferred.

## Ship

Rebased branch stays on the fork; the fork-unique waves are PR candidates for upstream (Niko1221) once green — upstream is actively developing this exact area, so contribution beats long-lived divergence.

## Execution results (2026-11-03)

**Ported and gated: wave 1 only.** FP16 context K/V ring + uint2 register prefetch (`cf70f46ac`) merged by hand into main's rewritten kernels (`70f43138d`): fork's `RegTraits<RingT>` templating and all-ring fast path fused with main's `tag`/`epoch` ring-log params; `df2_attention_f16`/`df2_kv_ring_f16` entry points; `STRATA_DF2_FP16_RING=0` off-switch; `copy_state_from` refuses mixed-dtype state; bench parity gate ported.

Gates (RTX 5080, CUDA 13.0, vcvars-wrapped build `build-df2-cu`, 2026-11-03): **4 of 6 genuinely green, 2 need the drafter pack this host lacks.**
- `dflash2_q4_parity` rc 0 — "PASS: 40 of 40 cases match the CPU reference" (self-contained)
- `dflash2_attn_bench` rc 0 — printed `fp16 ring: 37.1 us per call (bitwise = fp32 ring)`: the wave-1 bitwise-equality contract is proven on this GPU; fp16 ring at half the ring bandwidth
- `draft_cap_test` rc 0 — ok; `suffix_drafter_test` rc 0 — OK
- `dflash2_parity` rc 0 but printed only its usage line: needs `<pack> <case.bin>` (the incoai DFlash2 drafter pack for the Qwen3.8-Flash-Next target + torch tap-dump case; tooling paths live on the dev PC per tools/dflash2/README) — NOT actually run here
- `dflash2_ring_test` rc 0 but usage-only: same pack dependency — NOT actually run here
Re-run those two on a rig holding the pack (gfx1201 rig / dev PC). The merge fixes needed to get the CUDA build green: `86ddc13cc` (wave-1 RegTraits/pack-unpack restore + corrupted identifier), `f454895d5` (dedup dev_class/q8_0_rows1 and the superseded 6-param multi_kernel; mmvq_sync_lds; ADAPT_FETCH call sites adapted to main), `67462de10` (launch_pdl row_ids arg), `55b360d59` (df2_kv_ring_f16 declaration; MSVC env helpers in upstream's ring test).

**Wave 2 (BATCHBODY `2999c0b18` + `copy_slot` `5928dc613` + graph `b0be13a49`): NOT applicable as a port — replacement.** `5928dc613` is GDN/kv_pool serve-batch plumbing (main covers drafter D2D copies with `copy_state_from`); BATCHBODY is a kernel-level batched body across fork `BatchItem` slots, while main's `29d55bd71` runs per-slot CUDA graphs launched-then-finished together. Merging them is a design change to upstream's batch scheduler, not an optimization of it → RFC for upstream, not this branch.

**Wave 3 (verify overlap `19e213789`): NOT applicable — consumes wave 2.** `precompute_batch_context` feeds the precomputed projections only into `draft_batch_begin`'s batch body; without wave 2 the optimization has no consumer. The `set_taps_ready` hook grafts onto fork's verify tap layout (conflicts in verify.hpp/generate.cpp). Ships with the wave-2 RFC.

**Wave 4 (window merges `941ad664f`, `922620781`): NOT applicable — generate.cpp call-site commits.** Each touches only `src/program/generate.cpp` (16+/11− and 24+/10−): they widen the fork's serve window calls. Main's `read_windows` (generate.cpp:16221) is main's own structure; the perf must be re-implemented against main's admission/window code if wanted → RFC.

**Wave 5 (head micro-perf `3f647f45b`): NOT applicable — HIP-only.** Head IL + Q6 pack + register top-k are the gfx12/HIP head layout; main's CUDA side already ships its own native-head Q4_0 codebook path (8ec2c8a91, b353ce7d4). No CUDA-side delta to A/B.

**Net:** the fork's CUDA-relevant dflash2 perf that is *applicable to upstream's implementation* is the FP16 ring; the batch-body family is a parallel architecture, best carried to upstream as an RFC with the fork's measurement data (concurrent-serve +24–41%).

## General perf waves picked (2026-11-03, user: "port any general perf optimizations also" + "pull in the hip only stuff also")

All cherry-picked `-x` onto `perf/dflash2-on-main`, each individually conflict-reconciled against main's trees (fork-base drift), verified marker-free after resolve:

1. `b7770864a` perf(verify,experts) bucket MoE plan by M_e — clean
2. `998ae6dee` perf(shared_expert) fused gate+up MMVQ pair — took main's `native_mmvq_pair` guard (main already has `b6_rows_mode`)
3. `f1b34c952` perf(moe) parallel blockIdx.y resident-group chunks — clean
4. `1fcd39f01` perf(verify) zero-CPU-miss pool dispatch fast path — clean
5. `7a5d4584f` perf(mmvq) register-cached activations across ROWS>1 — comment-block merge
6. `a99f65016` perf(shared_expert,moe) SwiGLU fused into `native_mmvq_pair` — kept fork's combine restructure (ours kept both gated paths = double gate-apply)
7. `f87da9044` perf(verify,moe) skip zero-fill hit rows — took fork side (dropped fork's debug `notef`)
8. `a1ac0ce17` perf(verify,pool) float4 doorbell + single-pass plan — clean
9. `734227266` + `b2a6054f2` the shipped Zen 5 CPU pair (`cpu_avx512_ok` family 0x1A gate, dpwssd, `STRATA_X86_ISA=zen5`) — both clean onto main
10. `7513a788d` perf(hip) quantize+interleave one launch, gfx12-gated `qil_on()` — kept qil_on helper (THEIRS)
11. `c057974cb` perf(hip) PCIe-wait fold `fetch_blobs_after` — hpp theirs, stubs both, verify.cpp keep-ours@helpers + theirs@call-site, cu both-theirs
12. `bb7332cac` perf(hip) fold default off everywhere (+ A/B doc) — clean
13. `9d9837ff3` fix(expert-cache) rank-major per-layer slots — clean
14. `20a0f1ae8` perf(hip) PR#1097 decode fusions flag flip — clean

HIP-only hunks ride the CUDA build structurally; gfx1201 rig gate stays deferred (honest wording per plan).

**Wave-1 repair `86ddc13cc`:** the first gate build caught that the hand-merge dropped the fork's `pack/unpack_f16x4` + `RegTraits` template block and injected one corrupted identifier (`kin<U+8C17>`) in `df2_kv_ring`. Lesson: scanning for junk tokens is not enough — the compiler is the corruption detector; grep non-ASCII across merged files (`[^\x00-\x7f]`) and template-use sites, then build.

## HIP/RDNA4/gfx1201 sweep (2026-11-03, user: "port all of our HIP/RDNA4/gfx1201 optimizations. And the mmvq-il-iq3-tables")

All 32 fork-only HIP/RDNA4 candidates from `origin/main..integ/community-2026-10` triaged DAG-order onto this branch. Portability policy applied throughout: `cherry-pick -x` provenance; additive-only adaptations (main params/branches never dropped); fork-only behavior behind `STRATA_*`/`__HIPCC__`/gfx guards so the CUDA default stays bit-identical; commit messages carry Port-note blocks listing every adaptation site.

**Ported with content:** RDNA4 P1/P1b `STRATA_HC_Q8` prefill dequant, gfx12 `gr_upmix`+fused-Q8 `w_up`, wmma_gemm RDNA4 port (`gdn_chunked` half dropped — main's own gfx12 GDN WMMA pass supersedes it), fused_gr tsum+LDS-only barriers, EXPDEC routed experts with gfx12-aware default, wave32 QSA decode + B6 MMVQ fragment reuse, QSA block-scoring/top-k, RDNA2 prompt attention (`STRATA_PROMPT_ATTN_RDNA2` ifdef grafted onto main's gated call path), Q4_0-KV WMMA parity wiring, `STRATA_SELECT_GRID` (merged with main's SPLIT2: `template <bool S2, bool XQ=false>`, 4-way launcher), gfx1101 hipBLASLt table, per-device FP16 prefill selection, grid-padding opt-in T=1 (additive signature `(y,n_in,fast,n_out)`; 9-param multi_kernel superset kept, fork's NCOLS==1 pad guard ported into it), pinned-expert-row HIP helper, `quantize_il_parity` test, **mmvq-il on HIP** (IL region compiles every backend, PF variants behind `STRATA_IL_PF`, HIP stubs removed, `il_cc()` maps 12.0 AMD → 1202, R9700 `kIlRows_1202`/`kIlRows58_1202`/`kIl1Rows_1202` tables), **IQ3_S/IQ3_XXS interleaved mmvq for RDNA4** (`kIlSingleStream`, `kIq3IlMaxNIn=4096`, `iq3_grid.inc.h`), and the applicable slices of the catch-all `3f647f45b` (setup.py GFX12_SWITCHES, verify.cpp batch-commit overlap + `STRATA_DF2_WAITCOMMIT`, native_bf16 `STRATA_BF16_TSUM` split, qsa_select gfx12 default, B6 auto mode with `b6_default_rows_gfx12`, `STRATA_MMVQ_TSUM` split).

**Superseded / abort-empty (main already carries):** `d79fe8e88` fused_gr/verify/s2 wave (PR #3264 upstreamed), route-window (`51d8370a8`), select-scorer tuning (`5dcc80c62`), DFlash2-on-HIP build wiring (`f5e6e2cb9`; main relocated dflash2_parity to src/core), sm_89 IL tables (`17dd3a912` — main has `kIlRows_89` via #3453: **mmvq-il-iq3-tables = nothing to port on the CUDA side**), `a65910327` fused_gr restore (merge-seam; originals picked directly), docs-only and comment-only survivors.

**Deferred to the gfx1201 rig (honest):** all gfx12-only kernels gate-build on HIP there (this host is CUDA); dflash2.cu head tiling (`TK_PER_THR`/`TM_PER_THR`, `launch_gemm8_ncols`) and pruned-head IL/Q6_K pack wiring were NOT ported — main's head path was reworked (f-0291-df2) and needs a bench-driven re-port, not a blind graft. CUDA gate (q4_parity, attn_bench, draft_cap, suffix_drafter, arch_defaults) re-run against the enlarged branch: see the build log marker below when done.

### Post-sweep CUDA gate (2026-06, head 6829727cf, tree E:\\dev\\Strata-wt\\build\\df2-cu)

- Full build: green after four pick-repair commits (marker-tail strip, fused_gr
  graft undo, gr_parity.cpp rebuilt as main + the e4821bc11/b32a5ef5c hunks).
- ctest 14/14: arch_defaults, suffix_drafter, draft_cap, gr_parity, dequant_s2,
  mmvq_multi, mmvq_il, mmvq_il_pf, mmvq_il_89table, quantize_il, pdl,
  iq_multi (+kq_nov2k, kq_swiglu, kq_nosub16).
- The IL suite is CUDA-green for the first time: the 7553a8153/0dcd08151
  every-backend IL path is bit-identical on sm_120.
- gr_parity: main's contract plus the two fork hunks live - "fused GR Q8_0
  dequant (vec8 + pair), finite/Inf/NaN scales: pass" (P1 port verified
  bitwise vs scalar) and the QFUSE STRATA_TSUM A/B section, no diffs.
- dflash2_q4_parity: PASS 40/40 vs CPU reference (matches pre-sweep baseline).
- dflash2_attn_bench: fp16 ring 36.9us @ ctx2048 (baseline 37.1us); the
  "bitwise = fp32 ring" label is a printf constant - ring bitwise equality is
  enforced by the wave-1 ring tests, not this bench.
- Still rig-only: dflash2_parity + dflash2_ring_test (drafter pack), the
  ddebfd3cb gfx12 tsum/barrier re-tune, and the QFUSE counter-free tuning.
