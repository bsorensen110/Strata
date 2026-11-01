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

**Ported and gated: wave 1 only.** FP16 context K/V ring + uint2 register prefetch (`cf70f46ac`) merged by hand into main's rewritten kernels (`70f43138d`): fork's `RegTraits<RingT>` templating and all-ring fast path fused with main's `tag`/`epoch` ring-log params; `df2_attention_f16`/`df2_kv_ring_f16` entry points; `STRATA_DF2_FP16_RING=0` off-switch; `copy_state_from` refuses mixed-dtype state; bench parity gate ported. Gates on RTX 5080: `dflash2_parity` rc 0, `dflash2_q4_parity` rc 0, `dflash2_ring_test` rc 0, `dflash2_attn_bench` bitwise-equal fp32 reference + timing line.

**Wave 2 (BATCHBODY `2999c0b18` + `copy_slot` `5928dc613` + graph `b0be13a49`): NOT applicable as a port — replacement.** `5928dc613` is GDN/kv_pool serve-batch plumbing (main covers drafter D2D copies with `copy_state_from`); BATCHBODY is a kernel-level batched body across fork `BatchItem` slots, while main's `29d55bd71` runs per-slot CUDA graphs launched-then-finished together. Merging them is a design change to upstream's batch scheduler, not an optimization of it → RFC for upstream, not this branch.

**Wave 3 (verify overlap `19e213789`): NOT applicable — consumes wave 2.** `precompute_batch_context` feeds the precomputed projections only into `draft_batch_begin`'s batch body; without wave 2 the optimization has no consumer. The `set_taps_ready` hook grafts onto fork's verify tap layout (conflicts in verify.hpp/generate.cpp). Ships with the wave-2 RFC.

**Wave 4 (window merges `941ad664f`, `922620781`): NOT applicable — generate.cpp call-site commits.** Each touches only `src/program/generate.cpp` (16+/11− and 24+/10−): they widen the fork's serve window calls. Main's `read_windows` (generate.cpp:16221) is main's own structure; the perf must be re-implemented against main's admission/window code if wanted → RFC.

**Wave 5 (head micro-perf `3f647f45b`): NOT applicable — HIP-only.** Head IL + Q6 pack + register top-k are the gfx12/HIP head layout; main's CUDA side already ships its own native-head Q4_0 codebook path (8ec2c8a91, b353ce7d4). No CUDA-side delta to A/B.

**Net:** the fork's CUDA-relevant dflash2 perf that is *applicable to upstream's implementation* is the FP16 ring; the batch-body family is a parallel architecture, best carried to upstream as an RFC with the fork's measurement data (concurrent-serve +24–41%).
