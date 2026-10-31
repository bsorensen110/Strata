# Batch DFlash2 implementation plan (2026-10-31)

Source: [DECODE-DRAFT-SPEEDUP-IDEAS-2026-10-30.md](DECODE-DRAFT-SPEEDUP-IDEAS-2026-10-30.md) on `fork/reimagined-memory`
(ideas 1, 2-for-DFlash2, 4, 5, 6). This plan covers only the DFlash2 ideas; the batch-MTP switches from that
branch are already coded and have their own campaign in [BATCH-MTP-TEST-RUNS.md](BATCH-MTP-TEST-RUNS.md).

Line references below were re-grepped on this tree (HEAD `8d1c1549`, which is `fork/reimagined-memory` + one docs
commit). The ideas doc cites `c43e882`; the MTP work added ~15 lines per file, so the ideas doc's dflash2.cpp
references land 15 lines early against this tree.

Standing rules (from the ideas doc and the A/B skill): every change is **opt-in behind an env gate** and prints an
engaged/refused banner at readiness; **bit parity** of the target's answers is required (proposals the target did
not pick are never emitted, so drafting changes cannot change output *provided* kernels and row math are
unchanged — variants that change float bits are labelled below); the solo MTP/DEVTOK work already on the branch is
**unbuilt on GPU** — a HIP build is a gate for every phase here.

## Verified code map (current tree)

- Window plan + row build: `src/program/generate.cpp:13728-13800` (`df2_window_plan`, `batch_spec_slot_rows`,
  rows/tok/pos fill). The accept rule + per-round stats: `generate.cpp:14020-14030` (`batch_spec_accept`,
  `Df2DraftStats::note_round`).
- Per-slot draft (the serial loop): `generate.cpp:14235-14300` — per slot: `df2_stage_rows` geometry →
  `cudaMemcpyAsync` taps copy into the **one shared** `df2_stage` → `df2.set_slot(b)` → `df2.draft(...)`
  (graph launch + full `cudaStreamSynchronize` inside) → `set_slot(0)`. Four slots = 4 host round trips, fully
  serialized, after the window's commit.
- Drafter: `src/core/dflash2.cpp` — `record_context` 494 (per-slot ring writes), `record_block` 527 (the 8-row
  block: embed, 5 layers, final norm → `fn_`; tail 598-610: `native_quantize_q8_1(fn_+N_, …, kDrafts)`, head
  `native_mmvq` → `logits_` (7 × 106,299), `df2_topk` 601, `candidate_selector` GEMM 605, `df2_select` 606-608).
  `capture` 621 (per slot, per n=1..8, bakes this slot's rings + the one taps pointer), `prepare` 660,
  `draft` 689 (sync at ~727 in the ideas doc's numbering = ~742-15 here; graph launch 717, `cudaStreamSynchronize`
  ~734, mapped reads of `h_out_` after).
- Kernels: `src/kernels/cuda/dflash2.cu` — `topk_part_kernel` 585, `topk_merge_kernel` 634, `select_kernel` 687
  (one 256-thread block, walks its rows **sequentially** — the chain), `df2_topk` 890, `df2_select` 900 (rank 256).
- Geometry: `include/strata/core/dflash2.hpp` — `kRows=8`, `kDrafts=7` (42-43); N_=2560, L_=5, NKV_=8, HD_=128,
  ring_slots_ = kRing = 2048+slack (181, 47); `fp16_ring_` default true (223) → ring ≈ 2048·5·8·128·2·2 B ≈ 40 MiB
  per slot, matching the ideas doc's "4 rings ≈ 40 MiB".
- Row split laws: `include/strata/core/batch_spec_slots.hpp` — `batch_spec_slot_rows` 33 (uniform `max_drafts`),
  `Df2WindowPlan`/`df2_window_plan` 66-100, `df2_stage_rows` (committed-prefix geometry).
- Depth rules to reuse: `include/strata/core/batch_mtp_depth.hpp` — `MtpDepthConfig` 25, `mtp_depth_for` 54,
  `mtp_depth_trim` 67. These are drafter-agnostic pure functions.
- Overlap hook: `Verifier::set_post_launch` (`include/strata/core/verify.hpp:474`) exists (added for MTP
  EARLYCOPY/DEVACCEPT). `tap_dev()` = `[max_t][n_taps][n_embd]` (verify.hpp:433-436), written **by the window
  graph itself**; "valid until the next window".
- Slot state: `BSlot.draft` is `std::array<int32_t, kVerifyNarrowT>` (generate.cpp:12967), `dflash_k =
  min(spec-1, 7)` (generate.cpp:9595).

Why the batch window costs 82-84 ms vs 45 ms for batch MTP: 4 slots × (taps copy + graph launch + full stream
sync) serialized after the window, each head+topk+select launched per slot, plus the drafter's block GPU time
itself. Ideas 4+5 target the launch/sync overhead; idea 2 targets the acceptance gap.

---

## Phase 0 — why does batch accept less? (idea 1; measurement only, no code)

The acceptance gap (42-53% batch vs 62-75% solo at the same depth) gates how much idea 2/4 pay off. No new code:
`Df2DraftStats` (`generate.cpp:13299, 13761, 14028, 14456-14474`) already prints offered/accepted/depth
histogram/tokens-per-round when slots go idle.

Arms (per the `strata-speed-ab-testing` skill, `parallel: 4` scratch DFlash2 server):
1. **A-solo-batched**: one request against the 4-slot DFlash2 server (three idle slots).
2. **A-solo-true**: same request, `parallel: 1` launch, same cap flags.
3. **A-vram-matched**: arm 1 with the MTP drafter's VRAM headroom (shrink the expert cache so cached experts
   ≈ 11,184, ideas doc's 10,752 − 432 delta).
Read: per-slot acceptance from the stats line, expert-cache hit rate, `KV pool full` events. One request in the
batch server accepting less ⇒ slot-ring priming (C2 H3) / shared-stage effects; equal ⇒ concurrency
(expert-cache pressure, union misses). Runs recorded in the new `DFLASH2-BATCH-TEST-RUNS.md` (Phase 6).

Can run in parallel with coding phases; needs one measurement window.

## Phase 1 — one head pass across slots (idea 4) + staged overlap (idea 5)

**Gate: `STRATA_DF2_BATCHHEAD=1`** (off = today's serial loop, bit-identical). Expected: the head+topk+select
launches go from 4×3 to 1×3 per window, host round trips from 4 to 1, and the whole draft pass overlaps the
commit + next-window queue. Main suspect for the 82→45 ms gap.

Design:

1. **Split `record_block`** into `record_block_body` (embed → final norm, writes `fn_`) and `record_head`
   (quantize rows 1..7 → head GEMV → `df2_topk` → `candidate_selector` GEMM → select). `draft()` composes both,
   so the solo path and the off-gate path are the same kernel sequence in the same order — bit parity by
   construction.
2. **Per-slot `fn_` slices.** Allocate `fn_all_ = (1 + max_slots) × kRows × N_`; `set_slot(b)` points the member
   `fn_` at slice b (slot 0 keeps slice 0 → solo output unchanged). Slot b's captured body bakes slice b.
3. **Per-slot taps staging.** Grow `df2_stage` (generate.cpp:10417-10425) to `max_slots × kRows` rows of taps;
   slot b's graphs capture against `df2_stage + b·stride` (generalizes `Slot::taps_` — `df2_graphs_valid_for`
   already guards a per-slot pointer, just a different one per slot). `prepare()` gains the per-slot stride;
   admission's ring-priming (`generate.cpp:13527-13546`) is untouched (ring state, not taps).
4. **Batched head graph.** One captured graph per slot-count `n_sel = 1..max_slots`:
   `native_quantize_q8_1` over each slice's rows 1..7 → `xq_all` (rows = n_sel·7) → `native_mmvq` (n_rows =
   n_sel·7, one weight pass — the 106k-vocab GEMV reads weights once for all slots' rows) → `df2_topk`
   (grid (n_sel·7, kDf2TopBlocks)) → selector GEMM (n_sel·7 rows) → **new `df2_select_multi`**: grid = n_sel,
   one 256-thread block per slot, each block runs exactly today's `select_kernel` walk over its 7 rows with its
   own anchor. Per-block chain = today's per-row loop → bit parity. Keep `select_kernel` for n_sel = 1.
   Anchors: extend the mapped meta region (h_meta_/d_meta_, 32 ints) to carry up to `max_slots` anchor tokens
   (offsets 32+); each select block reads its anchor from there.
5. **Output.** `h_out_` mapped buffer → per-slot regions (tokens 0..6, probs 16..22 per slot, today's layout
   per slot); the -1 spin init and reads extend per slot.
6. **Window scheduling (folds idea 5 in, off by the same gate):**
   - `Verifier::set_post_launch`: record event E on the verify stream right after the window graph is queued
     (tap_dev is written by the window itself, so E is the true data dependency — **not** the commit).
   - Host builds the next window's rows from the mapped accept output (as today), then on the df2 stream:
     `waitEvent(E)` → enqueue all slots' taps copies → enqueue all slot bodies back-to-back → enqueue the batched
     head graph. **One** `cudaStreamSynchronize` (or a mapped -1 spin) right before the *next* window's row
     build reads drafts — never inside the per-slot loop.
   - Race to gate: the next window overwrites `tap_dev`. Enqueue an event F after the taps copies on the df2
     stream and have the verify launch wait on F (same shape as `Verifier::add_device_rows` in MTP DEVTOK).
     The rest of the draft (bodies + head) overlaps the next window freely.
   - Collection points mirror DEVTOK: rounds are collected before admission, restore, park, and non-BSTOP/BYIELD
     commands.
7. **Refusals** (with a stderr line): non-batch runs, pipelined windows, layer split, PLE, Metal,
   `STRATA_DF2_FORK` interaction (fork only exists off-HIP; keep BATCHHEAD off when fork is on for the first
   cut), `--batch-mtp` co-draft E1 (the MTP co-draft stays serial in the first cut; note it in the banner).

Costs: extra VRAM ≈ slots × (fn 328 KiB + xq + logits 7×106k×4 B ≈ 2.9 MiB + topscratch) ≈ **+15 MiB at 4 slots**
— feed into Phase 4's budget. Buffers size at `max_slots`; head graphs capture at `set_n_slots` time
(n_sel = 1..max_slots), bodies keep today's per-slot `exec_[n]` capture.

CPU-testable parts (host tests, GPU busy): taps-copy geometry per slot (extend `dflash2_tap_test`), row-slice
arithmetic, select_multi's per-slot offset math (host mock like the MTP `mtp_dev_accept` mock).

## Phase 2 — adaptive depth for DFlash2 (idea 2 for DFlash2)

**Gate: `STRATA_BATCH_DF2_DEPTH=N` (N ≥ 2) + `STRATA_BATCH_DF2_DEPTH_FLOOR=F` (default 0.5).** Mirrors the batch-MTP
depth rules exactly, reusing `batch_mtp_depth.hpp` verbatim (the functions are already drafter-agnostic; add a
`spec_depth_for`/`spec_depth_trim` alias if the MTP names bother anyone).

1. **EMA per slot**: update `sl.df2_ema` at `df2_stats.note_round` (`generate.cpp:14028`) with
   `keep−1 ≥ 1 ? 1 : 0`, weight 0.25, start 0.5 — same rule as MTP.
2. **Acceptance gate (before the window)**: in the row builder (13749-13770) each ready slot requests
   `d_b = mtp_depth_for(sl.df2_ema, cfg)`; `batch_spec_slot_rows` gains an overload taking a per-slot
   `max_drafts[a]` (today's uniform `max_drafts` becomes `max_drafts[a] = min(d_b, dflash_k)`; the round-robin
   fairness and window-fit drop-lowest-`ema^k` rules carry over). `df2_window_plan` keeps the breadth decision.
3. **Probability gate (after the draft)**: `df2.draft` already returns `probs[]` per proposal (`out_prob` of the
   selector); apply `sl.n_drafts = min(k_next, mtp_depth_trim(pp, n_drafts, floor))` next to
   `generate.cpp:14268-14271`. The first proposal always keeps its row.
4. A slot's drafted chain is 7 rows from **one** graph launch; depth only trims which rows get window rows, so
   Phase 1 is not a prerequisite — GPU cost of depth is ~0, it buys verify rows.
5. CPU tests: extend `tests/core/batch_mtp_depth_test.cpp` + a `batch_spec_slot_rows` per-slot-cap test.
6. Needs Phase 0's verdict on the acceptance cause: if slot-ring priming is the cause, tune depth after fixing
   priming, not on top of it.

## Phase 3 — commit off the critical path, standalone (rest of idea 5)

Phase 1's scheduling already launches the draft after the window event, not after the commit. Phase 3 A/Bs
the residual question from the ideas doc: does the co-draft need the commit's *data* (KV write of the accepted
prefix) or only the GPU? Facts from the code: the drafter reads `tap_dev` (window output) and its own ring; the
ring is fed by committed taps staged from `tap_dev` — it never reads the verifier's KV. So the data dependency
is the window, not the commit; verify with `STRATA_DF2_WAITCOMMIT=1` as the negative arm (forces sync after
commit launch). If confirmed, keep Phase 1's event scheduling and drop the pre-draft `cudaDeviceSynchronize`
paths; document. No new kernel.

## Phase 4 — VRAM for DFlash2 batch (idea 6)

Order of cheapness:
1. **Measure first**: ring size vs acceptance is testable today with a small knob
   `STRATA_DF2_RING_SLOTS=N` (clamp ≤ kRing; the ring is a circular buffer, smaller = older context evicted
   from the drafter's view, acceptance falls on long chats, nothing breaks). Prints its size in the banner.
   Window 2048 ring slots vs typical slot conversation length: read `avg slots` + conversation lengths from
   telemetry before shrinking anything.
2. **Batch-head buffer budget** (Phase 1, +15 MiB) counted against the 404 MiB free line.
3. **Ring quantization fp16 → fp8** (`fp16_ring_` shows the plumbing: `df2_kv_ring_f16`/`df2_attention_f16`
   have an element-type seam): **changes float bits** → gated opt-in, parity-suspect label, A/B shows
   acceptance delta next to the speed gain. Only after 1-2 measure the headroom value.
4. Shared drafter scratch: mostly already shared (xq_, q_, k_…); Phase 1 adds the only per-slot buffer (fn).
   Skip pre-mature "share the drafter's scratch" work.

## Phase 5 — explicit non-goals

- **Head+topk fusion** ("logits never written out"): `topk (7×106299)` measured **49.5-50.4 µs** and
  `select (7 rows)` **13.6 µs** in `dflash2_small_bench`. Against 82 ms windows this is noise; the logits
  buffer is 2.9 MiB and L2-resident. Do not spend the fusion complexity; the batched head (Phase 1) gets the
  same launch-count win at rows=28 for ~free.
- **Verify launch count** (idea 7): solo-decode trace territory, separate campaign.
- **MTP + DFlash2 in one slot** (idea 8): owner decision, row-layout redesign, not scheduled.
- **Generalizing draft_group to chained drafts**: needs the bit-exact question from BATCH-WINDOW-SYNC answered.

## Phase 6 — build, test, and measurement gates (every phase)

1. **HIP build gate first**: this branch has never been built on GPU (the ideas doc says so about its own MTP
   work). `build-verify` HIP configure + `ctest` incl. `batch_mtp_depth_test`, `batch_mtp_select_test`,
   `dflash2_slots_test`, `dflash2_tap_test`, `arch_defaults_test` + the `dflash2_*_bench` microbenches (one
   new `df2_select_multi` host-mock test + a batched-head row-count bench).
2. **Parity probe before throughput** (skill Core Rule 3): fixed prompt set, temp 0, SHA-256 of replies across
   arms. Phase 1/2 must match the base floor exactly (bit-parity by construction); Phase 4.3 is parity-suspect
   by design and reports deltas instead.
3. **A/B arms, interleaved** (base, treatment, base, treatment), `tools/ab/batch_burst_bench.py`, client-tokens
   over wall-time for concurrency; engine `drafts accepted A of D` + the `Df2DraftStats` line for mechanism.
   Compare targets: batch DFlash2 window 82-84 ms (Phase 1), acceptance 42-53% (Phase 0/2), 404 MiB free /
   10,752 cached experts (Phase 4).
4. Results doc: **new `docs/DFLASH2-BATCH-TEST-RUNS.md`** with arms, env keys, gate banners, parity table,
   speed table with floor spread; method + binary md5 + exact reproduction commands in the doc; unverified
   claims labelled.
5. Production untouched throughout (scratch server on the stopped service, restore per `strata-hip-install`).

## Sequencing

| # | Work | Needs GPU? | Expected effect (estimates, unmeasured) |
|---|---|---|---|
| 1 | Phase 0 acceptance diagnosis | one window | explains 42→62% acceptance gap; gates Phase 2 tuning |
| 2 | HIP build gate + ctest/microbench baseline | yes | unblocks everything; branch is unproven |
| 3 | Phase 1 batch head + staged overlap (`STRATA_DF2_BATCHHEAD`) | build+measure, code GPU-free | −1 launch round, −3 syncs, head weights read once: biggest window-time lever |
| 4 | Phase 2 adaptive depth (`STRATA_BATCH_DF2_DEPTH`) | build+measure, code GPU-free | spare-row windows: ~0 GPU cost for deeper chains |
| 5 | Phase 3 commit-dependency A/B (`STRATA_DF2_WAITCOMMIT` arm) | one window | confirms Phase 1's event choice |
| 6 | Phase 4 ring-size knob + budget | knob is tiny; measure with Phase 1 arms | recovers cached experts (−432 today) |

Phases 1+2 are independent of each other and of GPU access for coding; ship each behind its own gate, measure
interleaved, and only fold defaults into the DFlash2 batch launch line once a window shows the gain.
