# Decode and drafting speed-up ideas: DFlash2 and MTP, batch mode first (2026-10-30)

Status: research notes from reading the code and the earlier measurement docs. Nothing in this file was measured
in this session. Every gain below is an estimate. Every file:line reference is to the tree at `c43e882`.

Hardware the cited numbers come from: one AMD Radeon AI PRO R9700 (gfx1201), `swift-1.5-iq3_s`, MoE experts split
between a VRAM cache and a host RAM arena. Sources: [DFLASH2_BATCH_TUNING_2026-10-29.md](DFLASH2_BATCH_TUNING_2026-10-29.md),
[C2-BATCH-PENALTY-DIAGNOSIS.md](C2-BATCH-PENALTY-DIAGNOSIS.md), [BATCH-WINDOW-SYNC-2026-10-30.md](BATCH-WINDOW-SYNC-2026-10-30.md),
[AB-FUSED-DECODE-E1-2026-10-30.md](AB-FUSED-DECODE-E1-2026-10-30.md), [DECODE-TRACE-2026-10-30.md](DECODE-TRACE-2026-10-30.md),
[LAUNCH-REDUCTION-AB-2026-10-30.md](LAUNCH-REDUCTION-AB-2026-10-30.md).

## What the earlier measurements already say

| Fact | Number | Source |
|---|---|---|
| Batch MTP window (4 slots, `--batch-mtp`) | 40–44 ms; host waits on the GPU for about 74% of it | BATCH-WINDOW-SYNC, DECODE-TRACE |
| Commit / emit buckets per window | 2.2–3.1 ms / 2.1–2.9 ms | BATCH-WINDOW-SYNC |
| Serial MTP co-draft | about 0.75 ms per slot, 2.2 ms per window | BATCH-WINDOW-SYNC D1 |
| Batch DFlash2 window (depth 3) | 82–84 ms vs 45 ms for batch MTP | AB-FUSED-DECODE-E1 |
| Batch DFlash2 acceptance vs solo | 42–53% batch vs 62–75% solo at the same depth | DFLASH2_BATCH_TUNING, C2 |
| Batch aggregate | MTP 107–118 tok/s, plain 97, DFlash2 depth 1 81 | DFLASH2_BATCH_TUNING |
| Solo decode trace | about 1,100 dispatches per token; 76% of GPU idle time in gaps under 20 µs | DECODE-TRACE |
| Long host stalls (solo, traced) | 80 of 85 stalls ≥100 µs follow `mtp_select_kernel` → `copy_i32_from_mapped_kernel`, about 3 ms each | DECODE-TRACE follow-up |
| DFlash2 batch VRAM | 404 MiB free at start, 10,752 cached experts vs 11,184 for MTP | DFLASH2_BATCH_TUNING |

Already tried, do not repeat without a new reason:

- Two-phase MTP co-draft (`STRATA_BATCH_DRAFT_TWOPHASE=1`): no gain (2.05–2.53 ms against 2.2 ms).
- Grouped MTP co-draft (`STRATA_BATCH_DRAFT_GROUP=1`): saves about 0.8 ms per window, drafts differ in bits; opt-in only.
- Async batch commit (`STRATA_BATCH_COMMIT_ASYNC`): the wait moved into emit; reverted.
- PCIe-wait fold (`STRATA_FUSE_PCIE_WAIT=1`): −4% at 4 concurrent; default off.
- Fused decode (`STRATA_FUSE_DECODE=1`): its kernels do not run on the native pack.

## Code map: batch DFlash2 (one window)

1. Plan: `df2_window_plan()` and `batch_spec_slot_rows()` give each selected slot one anchor row, then share the
   rest round-robin among ready slots up to `max_drafts` (`include/strata/core/batch_spec_slots.hpp:24-52,70-151`).
   A positive `--batch-draft-depth` lowers breadth. Allocation does not look at acceptance or draft probability.
2. Draft: per slot, stage the committed taps into one shared `df2_stage` buffer, `set_slot(b)`, `df2.draft()`
   (`src/program/generate.cpp:13558-13593`, solo path near `18190-18211`).
3. `Dflash2Drafter::draft()` launches one captured graph per row count and then calls `cudaStreamSynchronize(cs_)`
   before reading mapped output (`src/core/dflash2.cpp:678-737`, the sync at about line 727). So slots draft one
   after another, each with a host round trip.
4. Head and top-k: head GEMV writes full logits over the 106,299-token pruned vocab; `topk_part_kernel`
   (256 threads per (row, chunk), up to 4,096 logits per chunk) plus `topk_merge_kernel` give the top 16
   (`kDf2TopK`); then `df2_select` runs one 256-thread block that walks the rows in order
   (`src/kernels/cuda/dflash2.cu:578-750,890-904`).
5. Verify: one window over all selected slots' rows; accept the prefix up to the first mismatch
   (`include/strata/core/batch_spec.hpp:73-112`); `commit_slot_prefixes()`.
6. Default-off knobs: `STRATA_DF2_FORK` (off on HIP), `STRATA_DF2_WAITCOMMIT`, `STRATA_DF_EVENTS`,
   `STRATA_DFLASH_CONF_ADAPT` (its policy sets `max_k=1`; review before relying on it), `STRATA_DFLASH_DEPTH_SLOPE`,
   `STRATA_DFLASH_MIN_CUM_P`.

## Code map: batch MTP (one window)

1. Slot selection is round-robin. With `--batch-mtp`, each slot gets exactly two rows (anchor + one proposal), and
   the window is capped at `kVerifyNarrowT = 8` rows (`src/program/generate.cpp:13522-13544`,
   `include/strata/kernels/window_limits.hpp:21`). Four slots fill the 8-row window exactly.
2. `Verifier::run_slot_rows()` runs all rows; dense weights are read once and routed experts are the union over
   rows (`include/strata/core/verify.hpp:10-11,262-276`). The union grows about 1.75× at 2 rows, 2.4× at 3,
   3.05× at 4 (comment in verify.hpp).
3. Greedy accept is `outb[i] == tok[i+1]`; sampled acceptance with `min(1, p/q)` is opt-in (`STRATA_SPEC_PROB`,
   `include/strata/core/spec_prob.hpp`).
4. `commit_slot_prefixes()` syncs once (`src/core/verify.cpp` about line 5037).
5. Per live slot: copy two residual rows into the slot drafter, then `MtpDrafter::draft()`, which ends with
   `cudaStreamSynchronize` (`src/core/mtp.cpp` about line 1884). The drafted token goes into
   `bs[b].draft[0]` on the host and is written into `tok[]` for the next window.
6. MTP draft round profile (`STRATA_MTP_PROFILE=1`): 0.768 ms total, head GEMV 0.234 ms (30%), the rest are
   launch-sized stages.
7. Guard: `--batch-mtp` and `--draft dflash2` are exclusive (`generate.cpp:6379`), because both use the slot's
   second row.

## Ranked proposals

1. **Find why batch DFlash2 accepts less.** No code change. One request on a `parallel=4` DFlash2 server against
   true solo at the same cap; then a second arm with VRAM headroom matched to MTP. If one request in a batch server
   already accepts less, the cause is the slot ring priming (C2 H3) or the shared stage buffer; if not, it is
   concurrency (expert-cache pressure, union misses).
2. **Adaptive depth by draft probability and acceptance (both drafters).** Allocate draft rows by expected payoff:
   give a slot a deeper chain only when its running acceptance rate and the draft's own probability say the extra
   row is likely accepted. For batch MTP this means allowing 3 rows (depth 2) for confident slots and 1 row
   (anchor only) for slots whose proposals keep failing, inside the same 8-row window. Started in this branch; see
   "Implementation status" below.
3. **Keep the MTP draft token on the GPU between draft and verify.** Remove the `mtp_select` → mapped copy → host
   → `tok[]` → H2D path. The verifier reads the draft token straight from the drafter's device output. Started in
   this branch; see below.
4. **Batch the DFlash2 draft head across slots.** One head GEMM over all slots' rows, one top-k launch, and a
   selector with one block per slot (the chain inside a slot stays sequential). Removes per-slot graph launch and
   per-slot sync (`dflash2.cpp:727`).
5. **Take commit off the critical path.** The async-commit test showed the next co-draft waits for the commit.
   Check whether it needs the commit's data or only competes for the GPU. If only the KV write of the accepted
   prefix matters, the drafter can read the residual rows before the commit finishes.
6. **Win back VRAM for DFlash2 batch.** Four slot rings at about 40 MiB each plus a 1.1 GiB drafter left 404 MiB
   free and about 430 fewer cached experts than MTP. Shrink rings, share the drafter's scratch, or quantize the ring.
7. **Cut verify launch count.** `gr_down_q8_fast` is about 51 launches per token; quantize/interleave about 171.
   76% of GPU idle time sits in gaps under 20 µs.
8. **MTP + DFlash2 in the same slot.** Needs a row-layout design to lift the guard at `generate.cpp:6379`. Owner
   decision.

Further ideas from the code reading, lower priority:

- Fuse the DFlash2 head GEMV with top-k so the 106k logits are never written out; or a gfx12-tuned top-k with
  larger chunks.
- Slot selection that prefers slots with ready drafts and high predicted acceptance, or slots whose routed experts
  overlap (smaller expert union per window).
- Start CPU expert work as soon as each layer's route IDs exist, to shrink `ms_wait` / `ms_pool` in verify.
- Generalize `draft_group()` to chained drafts (needs the bit-exact question from BATCH-WINDOW-SYNC answered first).

## Implementation status (this branch)

Ideas 2 and 3 are coded for batch MTP (`--batch-mtp`, serial windows). Six more MTP ideas from the MTP report
followed. Every one is off unless its environment variable is set. See "Adaptive depth and on-device draft handoff"
and "Six more MTP draft switches" below. Nothing here has been built for a GPU or measured yet.

## Adaptive depth and on-device draft handoff

### What was added

| Switch | What it does |
|---|---|
| `STRATA_BATCH_MTP_DEPTH=N` | A slot drafts and checks up to N MTP proposals per window instead of 1 (N is at most `--spec` - 1, and at most 7). |
| `STRATA_BATCH_MTP_DEPTH_FLOOR=F` | The gate for depth (default 0.5). See below. |
| `STRATA_BATCH_MTP_DEVTOK=1` | The draft tokens stay on the GPU between the draft and the next check. The host no longer waits for each slot's draft. |
| `--batch-draft-window 16` | With `STRATA_BATCH_MTP_DEPTH` on one GPU, batch-MTP windows may hold 16 rows instead of 8. |

With none of them set, the batch loop runs the same code and gives the same output as before.

### Adaptive depth: how a slot's depth is chosen

Each slot keeps a running average (EMA, weight 0.25, start 0.5) of "the target picked my first proposal".

1. **Acceptance gate (before the draft).** The slot asks for the deepest d (1 to N) where `average^d >= F`. With
   F = 0.5: average 0.75 gives 2 proposals, 0.80 gives 3, and a new request (0.5) gives 1, as today.
2. **Probability gate (after the draft).** Proposal j (j >= 1) keeps its row only while the product of the draft
   layer's own probabilities up to j is still at least F. The first proposal always keeps its row, as today.
3. **Window fit.** Every slot in the window keeps its anchor row and first proposal (today's two rows). If the
   deeper proposals do not fit, the ones with the lowest expected value (`average^k`) are dropped first.

Answers do not change. A proposal the target did not pick is never emitted (`batch_spec_accept`, the same rule as
today at two rows). The two-phase co-draft (`STRATA_BATCH_DRAFT_TWOPHASE`) is not used with it. The group co-draft
(`STRATA_BATCH_DRAFT_GROUP`) now joins it; see "Six more MTP draft switches" below. The engine prints one more stderr line when the slots go idle: offered proposal rows, accepted rows, and
proposals per slot-window.

Where it should help: windows with spare rows. With 4 busy slots in an 8-row window there are none, so it needs
`--batch-draft-window 16` (one GPU) or fewer busy slots. That is the case to measure first.

Code: `include/strata/core/batch_mtp_depth.hpp` (the rules, CPU-only), `tests/core/batch_mtp_depth_test.cpp`, and the
batch loop in `src/program/generate.cpp` (row build, accept, draft, admission).

### On-device handoff: what changes in a window

Today, after a window each slot runs its MTP draft, and the host waits for it (spin, then `cudaStreamSynchronize`).
The host then writes the token into the next window's rows. With `STRATA_BATCH_MTP_DEVTOK=1`:

1. After the window, each slot's draft round is launched (`MtpDrafter::draft_begin`) and **not** waited for.
2. When the next window's rows are known, a small copy kernel on the drafter's stream copies the slot's draft tokens
   from the drafter's output buffer into the verifier's token rows. An event is recorded after it.
3. The verifier skips those rows when it stages the host tokens, and its launch waits for the events on the GPU
   (`Verifier::add_device_rows`). No host wait sits between the draft and the launch.
4. After the window, the host collects the tokens (`draft_end`, already finished) to judge them and emit.

No new GPU kernel and no kernel signature change: it reuses `copy_i32_from_mapped`. Slots the window does not carry
also make the launch wait for their round, because that round still reads the verifier's residual rows. Rounds still
in flight are collected before admission, restore or park touches that slot's drafter.

A round also reads the verifier's rows and the shared expert cache, so every slot's round is collected before work
outside the batch windows: when a command arrives (anything but `BSTOP` and `BYIELD`), and after the windows that run
between the chunks of a prompt read. That is a host wait, but only when a request comes in; steady decode has none.
A slot with no cell left for a draft launches nothing; its round is collected at once and its proposal is zeros, as
with `draft()`.

It turns off, with a stderr line, when: the pipelined MTP windows are on, there is a layer split, the model has PLE,
a slot drafter has a chain rule (`STRATA_SPEC_COST`), or on Metal. With it, depth comes from the running average
alone, because the draft probabilities arrive after the rows are fixed.

### Not done yet

- No GPU build or run. The `.cpp` files pass a host syntax check (`g++ -fsyntax-only` against the CUDA headers); the
  HIP build has not been run. The depth rules have a CPU unit test, which passes.
- No speed numbers. Whether either switch is faster is unknown until the A/B below is run.
- The solo MTP path (the `mtp_select` to `copy_i32_from_mapped` gap in DECODE-TRACE-2026-10-30) still goes through
  the host.
- DFlash2 batch depth (idea 2 for DFlash2) is not started.

### How to measure (R9700, `strata-speed-ab-testing` method)

Same engine binary, same config, one variable per arm. Stop `strata.service` and use a scratch server, or edit the
service's environment between arms. Report cold trial 1 and warm trials apart, and the engine's `generated in` lines.

| Arm | Environment / flags |
|---|---|
| A (control) | `--batch 4 --batch-mtp`, no new variables |
| B (handoff) | A + `STRATA_BATCH_MTP_DEVTOK=1` |
| C (depth, 8 rows) | A + `STRATA_BATCH_MTP_DEPTH=3` |
| D (depth, 16 rows) | A + `STRATA_BATCH_MTP_DEPTH=3 --batch-draft-window 16` |
| E (both) | D + `STRATA_BATCH_MTP_DEVTOK=1` |

For each arm, run the skill's 4 x 256 concurrent test 3 times, and the solo greedy test once (one busy slot shows
depth best). Keep `STRATA_BATCH_DRAFT_TIMING=1` in every arm for the per-window draft time. Check:

- Output: at `temperature=0`, arms B to E must give the same text as A for the same prompt. A difference is a bug.
- Speed: aggregate tok/s, window ms, and the "MTP proposals accepted" and "MTP depth" stderr lines.
- B should cut the draft time per window (about 0.75 ms per slot today), not the acceptance.

## Six more MTP draft switches

These come from the MTP report's list (its items 2, 3, 5, 6, 7 and 8). Its item 1 (two-phase drafting by default) was
measured before with no gain, and item 4 is the adaptive depth above. All six are off by default. With none set, the
batch loop runs the same code as before.

| Switch | Report item | What it does |
|---|---|---|
| `STRATA_BATCH_DRAFT_GROUP=1` with `STRATA_BATCH_MTP_DEPTH=N` | 2: grouped multi-token drafting | Each slot's first draft comes from the one group pass, as before. A slot that may draft more runs its later steps on its own stream right after it. The host waits once for all of them. |
| `STRATA_BATCH_MTP_EARLYCOPY=1` | 3: early residual copies | Each slot's residual rows are queued for its drafter when the window is queued, not after the host has accepted and committed. The copy still waits for the window on the GPU. |
| `STRATA_BATCH_MTP_SELECT=1` or `=N` | 5: slot choice by acceptance | When a window cannot carry every live slot, it takes those whose proposals are taken most often. No slot waits more than N windows (2 by default). |
| `STRATA_BATCH_MTP_SELECT_OVERLAP=W` | 7: expert-overlap grouping | With `SELECT`: also prefer slots whose last rows used the same CPU-served experts as the slots already chosen (weight W, 0 to 8). |
| `STRATA_BATCH_MTP_DEVACCEPT=1` | 6: accept on the device | Needs `STRATA_BATCH_MTP_DEVTOK=1`. A greedy slot's next draft round is set up and started by the GPU right after its window, before the host has read the window. |
| `STRATA_BATCH_EXPERT_AHEAD=N` | 8: earlier CPU expert loads | While the host waits for layer l, it asks the expert source to start reading layer l+N's experts (N is 1 to 4). It asks for the experts the last window sent to the CPU. |

### How each one works

**Group with depth.** Before, `draft_group` only ran when every slot drafted one token. Now a slot whose drafter
allows more steps also joins. The group pass fills its first draft. Its own `mtp_select` then takes its row of the
group's output, and its step graphs run on its own stream. The group's ids and probabilities are copied out before
any step runs, because the owner's steps reuse those buffers. The owner's own slot goes last, after the other slots
have read its rows. A slot that accepted more than one proposal (a > 1) does not fit the group's two rows a slot, so
it drafts on its own, as with depth alone. Drafts are then cut by probability, as with depth alone.

**Early copy.** A new hook, `Verifier::set_post_launch`, runs right after a window's graph is queued. It records an
event on the verifier's stream. Each carried slot's drafter stream waits for that event and copies the slot's rows.
The copy after the commit is then skipped. A slot that ended in that window drafts nothing, so the host waits for its
copy before the next window.

**Slot choice.** Only matters when there are more live slots than the window holds (two rows a slot). With 4 slots and
8 rows it never acts. It acts with more slots, or with deep slots on a 16-row window. Each slot counts the windows it
waited. A window first takes the longest-waiting slots it must take so that no slot waits more than N windows. If the
window is too small for N, the bound is the smallest one it can keep. The rest are ranked by their acceptance average,
plus the overlap bonus. The rules are in `include/strata/core/batch_mtp_select.hpp` with a CPU test.

**Expert overlap.** With `SELECT_OVERLAP`, the verifier keeps the routed expert ids of each row and layer it serves
from the host (`Verifier::set_route_record`). Each slot keeps a 1024-bit sketch of the experts its rows sent to the
CPU (experts already on the GPU are skipped). Slots that share CPU experts make the CPU's work per window smaller. When
every expert is on the GPU there are no records, and the choice uses acceptance alone.

**Device accept.** In the post-launch hook, the rounds the window waited for are collected first (they are already
done on the GPU). Then, for each greedy slot, a small new kernel (`mtp_dev_accept`) runs on the drafter's stream after
the window. It counts the proposals the target picked, writes the round's tokens, row and cells, and the round and its
steps follow (`MtpDrafter::draft_begin_dev`). Their graphs are captured in the hook before the drafter's stream
waits for the window (`prepare_dev_round`), because a capture waits for that stream. Every row of the slot is caught up, as with `STRATA_MTP_CATCHUP_ALL`.
After the window the host runs its own accept rule. If it keeps the same count, the round stands. If it stops sooner
(an end-of-stream token, a length bound, a cancel), the round is thrown away and the slot drafts as with DEVTOK alone.
A slot with sampling or a bias table is not staged, because its row is drawn again after the graph. The draft tokens
may differ in their last float bits from the host path, because all rows are caught up. The target still checks every
proposal, so the output does not change.

**Experts ahead.** A new hook, `Verifier::set_layer_ahead`, runs once per layer while the host waits for the GPU. It
takes the last window's routes for layer l+N, drops experts already on the GPU, and calls the source's `warm` (if
it reads ahead) or `prefetch` (for each expert). A source held in RAM does nothing with it.

### When they turn off

`EARLYCOPY`, `DEVACCEPT` and `SELECT_OVERLAP` turn off, with a stderr line, without `--batch-mtp`, with the pipelined
MTP windows, with a layer split, or on Metal. `DEVACCEPT` also needs `DEVTOK`, so it is off with PLE and with a chain
rule too. `EXPERT_AHEAD` is off with a layer split. The hooks only act in the serial batch windows
(`run_slot_rows`), not in DFlash2 windows.

New stderr lines when the slots go idle: device-staged rounds kept, redrafted and not staged (`DEVACCEPT`), and the
number of experts asked for ahead (`EXPERT_AHEAD`).

### Not tested

- No GPU build. The `.cpp` files pass a host syntax check. The new kernel's logic passes a host mock. The slot choice
  rules pass a CPU unit test (`tests/core/batch_mtp_select_test.cpp`).
- No speed numbers and no output check on a GPU.
- The early copy can run while the commit runs. The copy today already does that (the commit may be asynchronous,
  `commit_slot_prefixes`), so the switch adds no new overlap. Both rely on the commit graph not writing the residual
  rows, which was read from the code, not tested.

### How to measure

Use the same method as above, with one variable per arm on top of control A (`--batch 4 --batch-mtp`):

| Arm | Adds |
|---|---|
| F | `STRATA_BATCH_MTP_EARLYCOPY=1` |
| G | `STRATA_BATCH_MTP_DEVTOK=1 STRATA_BATCH_MTP_DEVACCEPT=1` (compare with B) |
| H | `STRATA_BATCH_MTP_DEPTH=3 STRATA_BATCH_DRAFT_GROUP=1` (compare with C) |
| I | `STRATA_BATCH_EXPERT_AHEAD=1`, then `=2` |
| J | `--batch 6` with `STRATA_BATCH_MTP_SELECT=1`, then with `STRATA_BATCH_MTP_SELECT_OVERLAP=2` (compare with `--batch 6` alone) |

At `temperature=0` every arm must give the same text as A. `STRATA_BATCH_DRAFT_TIMING=1` shows the draft time per
window. For G, a high "redrafted by the host" count means the round is thrown away too often to pay.
