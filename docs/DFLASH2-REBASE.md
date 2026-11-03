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
