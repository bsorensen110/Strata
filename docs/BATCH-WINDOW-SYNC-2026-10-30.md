# Batch window host stalls: where the 2–3 ms per window goes (2026-10-30)

Scope: the `--serve` batch path with `--batch 4 --batch-mtp` (production-equivalent config), four concurrent
greedy requests. Measured with the `strata batch:` line the engine already prints when a concurrent burst goes
idle. Follows `docs/DECODE-TRACE-2026-10-30.md`, item 3 ("long host stalls").

## Method

- Engine: `build-hip/strata` on `integ/community-2026-10` (no timers in the tree; the hipEvent timers from the
  earlier attempt were reverted). Config: `cand-bmtp.json`, with scratch paths remapped per arm. Each arm has its
  own empty conversation-cache and slot folders.
- Load: `tools/ab/batch_burst_bench.py`, four distinct prompts at once, temperature 0, thinking off, 256 max
  tokens, two rounds per arm. The script stores the SHA-256 of each reply, not the text.
- Arms alternate: base, async, base2, async2. Each arm is a fresh engine process.
- Reproduce: `tools/ab/batch_burst_bench.py <cand.json> <out.json> --rounds 2 --max-tokens 256 --conc 4`, then
  `grep "strata batch:" <engine.log>`.

## Per-window breakdown (baseline arms, base and base2)

Window time 40.1–43.4 ms; run 35.4–38.5 ms; commit 2.2–2.3 ms; emit 2.1–2.2 ms; adapt wait 0.4 ms.

The run bucket is the GPU verify window, and it includes the host wait for the GPU. The commit bucket is
`Verifier::commit_slot_prefixes` (`src/core/verify.cpp`). The emit bucket is everything from the end of the
commit to the start of the next window. That includes the output loop, the finished-slot release, and the
next window's MTP co-draft.

## Findings

1. The emit bucket contains one `MtpDrafter::draft` call per active slot. Each call ends in
   `cudaStreamSynchronize(cs_)` (`src/core/mtp.cpp`, about line 1884), so the four slot drafts run one after
   another, each with its own round trip. This is the likely share of the 2.2 ms emit bucket. Not yet measured
   per slot; the next step is to time the draft loop.
2. `commit_slot_prefixes` synchronizes its stream once (`src/core/verify.cpp`, about line 5037). The commit's GPU
   work is on the critical path: the next co-draft reads state that the commit writes.
3. An opt-in async batch commit was tried (`STRATA_BATCH_COMMIT_ASYNC=1`: no sync after the commit graph, a
   `commit_done_` event, `wait_commit()` at the start of `batch_step` and before any slot release).
   Results:
   - commit host time fell from about 2.2 ms to 0.02 ms per window, as intended;
   - emit grew to 3.5–4.3 ms, so the wait moved rather than went away;
   - window time did not change: 39.7–44.2 ms in the async arms against 40.1–43.4 ms in the base arms.
   The change was reverted and is not in the branch. Conclusion: the commit's GPU time is not hidden by the
   host work that follows it, because the drafts depend on it.
4. The baseline is not reproducible under concurrency. Two baseline runs with the same four prompts gave
   different replies for request 0 of round 0 (`5be1ccfe…` and `47f8b6f8…`). In round 0, three of the four
   requests differed between the two baseline runs (request 3 matched). In round 1, three of four differed
   (request 2 matched). The mix of matches and mismatches points to arrival-order effects. Arrival order decides which slot
   takes which prompt, so the batch composition differs from run to run, and so do the numerics. A hash-based
   parity check under concurrency therefore needs a fixed-composition harness first.
5. Not run: a single-request determinism check. Without it, the claim that single-slot greedy output is stable
   is unverified.

## Next step

- Time the per-slot draft loop directly (launch and sync counts per window) to confirm finding 1.
- Two-phase co-draft: launch every active slot's draft graph, then collect and synchronise once. Expected to
  recover part of the emit bucket. Not implemented.
- Before any parity claim for a concurrent change: build a fixed-composition harness (for example, admit slots
  one at a time with a fixed order), then compare per-request hashes against a baseline run.

## Raw result files

`burst/{base,async,base2,async2}/result.json` and `batch_lines.txt` in the scratch directory
`~/.hermes/cache/scratch/ab-fused-e1-20261029/burst/`. Each arm's reply hashes and the per-window `strata batch:`
lines are there.

## Follow-up 2: co-draft timing, determinism, and the two-phase co-draft

### D1: the emit bucket is the serial co-draft

`STRATA_BATCH_DRAFT_TIMING=1` (default off) adds one stderr line per idle print with the co-draft host wall.
Four-way burst on the base build: co-draft 2.22 and 2.19 ms per window in 2.92 and 2.87 calls per window.
Emit was 2.20 ms in the same windows. The emit bucket is the serial co-drafts, about 0.75 ms each.

### Determinism: open, and not the co-draft

Greedy single-request replies differ between fresh launches of the same code, and the runs below all used one
request per launch with no prompt reuse:

- `--adapt-async 0`, base-equivalent gates, two launches: `d1bcd4790d66…` and `84369e232433…` (differ).
- `--spec 2` with no MTP and no batch-MTP (adaptation off), two launches: `1b23d9cca325…` and `394ef5aa2527…` (differ).

The speculation-off config is not runnable: the native pack needs `--spec T` with T ≥ 2 and refuses to start
otherwise. So the co-draft is ruled out by the `--spec 2` run, and asynchronous adaptation by the
`--adapt-async 0` run. Untested suspects: timing-dependent expert placement (VRAM / PCIe / CPU split of routed
experts) and the reduction order of the CPU expert pool. Consequence: an engine-level bit-parity claim needs
this fixed first. Kernel-level parity tests (below) do not depend on it.

### D2: two-phase co-draft (`STRATA_BATCH_DRAFT_TWOPHASE=1`, default off)

Implemented as `MtpDrafter::draft_begin` / `draft_end` (the fast path of `draft()` split at its wait). The batch
loop launches every active slot's round, then waits for all of them once.

- Check mode (`STRATA_BATCH_DRAFT_CHECK=1`, implies two-phase): each drafted token is re-drafted serially and
  compared. Two concurrent bursts: 0 of 447 and 0 of 417 drafted tokens differ.
- Timing: two-phase co-draft wall 2.05 and 2.53 ms per window, against 2.2 ms serial. Emit 2.54 ms against
  2.20 ms. No reduction.
- Begin and end were not timed separately. So it is not yet established whether the four graphs serialize on
  the GPU or the host setup dominates.

Conclusion: one sync per window does not remove the co-draft cost. The next lever is a single MTP draft graph
over all slots' rows, which needs a kernel change. Ceiling about 2 ms per window (about 5%), unmeasured. Not started.

### Launch-reduction branch (`perf/launch-reduction`, sub-agent)

- `quantize_il_parity` (the test commit `d9fcdd38`): all 45 cases pass on gfx1201 (5 shapes × 3 column counts ×
  3 seeds, byte for byte against the pair). This is the first run of that test.
- The fused quantize-plus-interleave (`585ff5f5`) is on by default on gfx12. Its engine-level effect is not
  measured: burst tok/s stayed within run-to-run noise (93–99) across the arms.
- The PCIe-wait fold (`055025ed`) has no kernel-level parity test yet. The launch count it claims (one fewer
  node per layer per group) is unverified in a trace.
- The branch was cut from `d043ccfd`, two commits behind `integ/community-2026-10`. A plain merge would
  delete the two docs commits. Rebase first.

### Tooling

`tools/ab/batch_burst_bench.py --same-prompts` sends the same first prompts every round (determinism checks).

## Follow-up 3: CPU share off, the profile, and the batched co-draft (`STRATA_BATCH_DRAFT_GROUP=1`, default off)

### "CPU share off" for identical replies: not available in this configuration

- The prefill CPU share is already off on this build: `generate.cpp` enables it only on CUDA builds with one GPU and no
  batch slots. `STRATA_PREFILL_CPU_SHARE=0` changes nothing here.
- The CPU share that does exist is the decode expert pool. The serve log: 46.8 GiB of experts, 21.3 GiB in the GPU
  cache (11184 experts), 30.8 GiB in the RAM copy. The batch line shows CPU experts at about 6.3 ms of each window.
- `--no-pool` (the GPU-only floor) does not start: verify needs the profile-filled VRAM expert tier (`hits.d_res`), and
  that tier is only built with the pool on. Without the pool, the uncached experts would contribute nothing, so the
  model would not be computed. Speculative decoding cannot run with the pool off.
- `--pool-workers 1` (one CPU worker, `pw1a`/`pw1b`): single-request replies still differ (`3dd488f5…` vs `c6bc3155…`).
  Pool worker order is not the cause.
- `--adapt-async 1` is production (the help text says it is not bit-exact run to run). `--adapt-async 0` (`detA`) still
  differed.
- Status: identical replies across fresh launches are NOT reproduced. The open suspect is the CPU/GPU routing of
  individual experts (cache hits vs CPU vs PCIe), which varies with timing. Fixing it means changing the engine's
  expert-routing decision, not a flag.

### Profile of one co-draft (the batched draft's target)

`STRATA_MTP_PROFILE=1` (the fast path now records too; before this only the slow path did), 4-way burst, ms per round
graph over 300 rounds: total 0.768 ms. Head GEMV 0.234 (30%), q+norm-rope 0.079, gate+o_proj 0.050, experts 0.046,
hc-read attn 0.043, k/v+append 0.034, attention 0.035, and the other stages 0.009–0.037 each. Stage costs are
launch-chain sized, so batching the rows across slots pays off.

### The batched co-draft (`STRATA_BATCH_DRAFT_GROUP=1`)

- What it does: one round for all active slots, from one captured graph per active-slot set (bitmask key). Item i's
  two rows sit at rows 2i, 2i+1 of the owner's buffers. The front appends each item's K/V into that item's own state,
  the rest attends each row over its own cells, and one head pass gives each item's draft. Applies only to serial
  single-step co-drafts (max_drafts 1, greedy); anything else falls back to per-slot drafts.
- Code: `MtpDrafter::GroupItem`, `group_applies`, `draft_group`, `capture_group` (`mtp.hpp`, `mtp.cpp`); the batch loop
  in `generate.cpp` collects items, runs the group after the emit loop, and reuses the check mode (serial re-draft per
  slot, token compare).
- Results (`burst2/` arms, 4 concurrent, 2 rounds):

| arm | draft ms/window | MTP acceptance | tok/s | draft tokens differing from serial |
|---|---|---|---|---|
| base (serial co-draft) | 2.13–2.19 | 70.2% | 96–98 | - |
| group, default rest kernels (`grp_multi_time`) | 1.30–1.39 | 68.3% | 98.4–98.6 | 25 of 420 and 0 of 440 (check mode) |
| group + `STRATA_NO_MULTI_GR=1` (`gN_time`, `gA_nomulti`) | 1.26–1.34 | **0.0%** | 66–67 | 2 of 755 and 0 of 705 |

- The group saves about 0.8 ms per window (about 2% of a 40 ms window). That is within noise for throughput, and the
  drafts differ in bits, so the arm is opt-in only. Not promoted.
- Where the bits differ: the group runs the rest on 2 rows per item with the multi-row kernels (router GEMV and top-10,
  MoE combine, residual write, head-mix). Serial runs them on one row. Each of those multi-row kernels is a candidate;
  `STRATA_NO_MULTI_GR=1` changes the single-row residual write as well, and that breaks the drafter for serial too
  (0% acceptance with and without the group). Do not use that switch on this build.
- A bit-exact group needs every rest kernel run one row at a time, which removes most of the saving. Not built.
- Check mode (`STRATA_BATCH_DRAFT_CHECK=1`) hides the state difference: its serial re-draft rewrites the same K/V after
  each group, so acceptance looks normal (67.9%) there. Use the non-check arm to judge the group.
- Pitfall found on the way: `batch_burst_bench.py` reads the port from the config's `port` field (default 8091), not
  from the server command line. An arm whose server listens on another port fails with "connection refused".

### Repro

```
cd ~/Strata-integ/build-hip && make strata -j32
# arms (cand.json per arm, env as listed): base=d1 (STRATA_BATCH_DRAFT_TIMING=1),
#   group=d1 + STRATA_BATCH_DRAFT_GROUP=1, check=group + STRATA_BATCH_DRAFT_CHECK=1
python3 tools/ab/batch_burst_bench.py <cand.json> <out.json> --rounds 2 --max-tokens 256 --conc 4
grep -h "MTP proposals accepted" <engine.log> | tail -1
grep -h "strata batch draft:" <engine.log>
```
