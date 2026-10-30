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
