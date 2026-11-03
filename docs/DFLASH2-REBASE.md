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

1. **Infra**: serve-mode draft wiring (`c245a6c30` pieces absent from main), 4k/8k attention windows (`9e9569e46`).
2. **Ring**: FP16 context K/V ring + uint2 register prefetch (`cf70f46ac`).
3. **Batching**: `STRATA_DF2_BATCHBODY` context projection + 5-layer body across slots (`2999c0b18`) + `Dflash2Drafter::copy_slot` (`5928dc613`) + `batch_body_exec_` graph capture (`b0be13a49`) — reconcile against main's 29d55bd71 graph launching in one pass.
4. **Verify overlap**: context projections overlap target verify layers 37–47 (`19e213789`).
5. **Window merges**: 16 trailing prompt tokens into first verify window (`922620781`); 16-row `read_windows` on one-card DFlash2 (`941ad664f`) — reconcile with main's read_windows.
6. **Co-draft**: per-slot n-gram co-draft (`448c0f836`) + `STRATA_DF2_BATCHHEAD` (`c983d97d1`) + BGEN admission fixups (`996650eaa`).
7. **HIP multi-row**: multi-row QSA/GDN, batched KV/idx appends, async PLE gather (`15d4684cf`) — structural merge; CUDA side compile-verified here, HIP gate deferred to the gfx1201 rig (honest wording in the commit).
8. **Docs**: wave measurement records (`556d7c611`, `623df1216`, `82ab07c89`, `f57bfc554`, `5e463f9bc`) — port as docs so decision history rides the branch.

## Divergent — measure, don't port blind

- Fork pruned-head IL/Q6-pack + register top-k (`3f647f45b`) vs main's native-head Q4_0 codebook path (8ec2c8a91): A/B on the same card; default keep main.

## Disk conversation cache — recommendation: EXCLUDE (for now)

Main has the memory conversation cache (`conversation_cache_test.cpp`, `conversation_copy.hpp`) and the app knows `auto_sessions`; disk-mode is 4 fork commits (`ec95b2bf6` header, `0002309fa` reclaim guard, `a0153a027` generate.cpp, `fcd4a61d0` docs, adapted from community `1aa35f01`). The live service does use it (`keep8`), so the value question is real: quantify cold-restart reload hit-rate and RAM headroom gain first; if it earns, it lands as its own labeled series (cache.hpp + generate.cpp + app config surface), not inside the perf rebase.

## Gates per wave

CUDA build green (fresh configure in the worktree — caches from other dirs are path-stale), `dflash2_parity`, `dflash2_q4_parity`, `draft_cap_test`, `suffix_drafter_test`, then one-card serve smoke `--draft dflash2` on the swift model; bitwise where the harnesses promise bitwise. HIP-only hunks: structural merge now, gfx1201 rig gate recorded as deferred.

## Ship

Rebased branch stays on the fork; the fork-unique waves are PR candidates for upstream (Niko1221) once green — upstream is actively developing this exact area, so contribution beats long-lived divergence.
