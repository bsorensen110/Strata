# C2 — Batch DFlash2 acceptance penalty: code-level diagnosis

## Symptom (from docs/DFLASH2_BATCH_TUNING_2026-10-29.md)

At equal proposal depth, batch accepts far less than true-solo:

| depth (proposals) | solo accepted/offered | batch accepted/offered | gap |
|---|---|---|---|
| 1 | (cap-3 is 2-prop) | 832/1239 = 67.15% | — |
| 2 | 331/439 = 75.40% (cap-3) | 1051/1966 = 53.46% | **22.0 pts** |
| 3 | 377/609 = 61.90% (cap-5) | 1138/2709 = 42.01% | **19.9 pts** |

The gap is consistent and large across depths, and tokens/slot-round is lower in
batch (2.056 at depth-2) than the solo equivalent would be (~2.5 at cap-3). This is a
**batch-specific** penalty, not a drafter-quality issue (same drafter, same pack).

## The two draft paths (verified against source)

**True-solo** (`generate.cpp:19402`, `draft_next`):
```cpp
df2.draft(ver.tap_dev(), a_ + 1, p, outv[(size_t) a_], dd, pp, err);
```
- `n_new = a_ + 1` rows (the anchor + the previous window's accepted proposals),
  read directly from `ver.tap_dev()` (the verifier's live tap buffer).
- The drafter is **slot 0**, the single conversation's ring, which has accumulated
  every committed row of the conversation via `context()` (prefill) and `draft()`
  (each window). The ring is **never re-primed** — it is the same ring from start to
  finish.

**Batch** (`generate.cpp:13558-13593`, the per-slot next-draft block):
```cpp
const bool geo = df2_stage_rows(first[t], keep[b], ntok[t], S, &lo, &hi);
// lo = first[t], hi = first[t] + keep[b]  ->  n_new = keep[b] rows
cudaMemcpyAsync(df2_stage.get(), ver.tap_dev() + lo * df2_stage_stride,
                (hi - lo) * df2_stage_stride * sizeof(float), ...);
df2.set_slot(b, de);
df2.draft(df2_stage.get(), hi - lo, pos[first[t]], sl.x, dd, pp, de);
df2.set_slot(0, se);   // back to slot 0
```
- `n_new = keep[b]` rows (the anchor + the slot's accepted proposals this window),
  read from a **shared staging buffer** `df2_stage` (the captured graphs bake that one
  pointer, so every slot drafts into the same rows).
- The drafter is **slot b** (the slot's own ring), which was **primed at admission**
  from the main session's ring (slot 0):
  ```cpp
  // generate.cpp:13128-13146 (batch admission)
  const bool primed = df2.set_slot(0, de) && df2.save_state(blob, de, upto) &&
                      df2.set_slot(b, de) && df2.restore_state(blob, upto, de);
  ```

## Root-cause candidates (ranked by likelihood)

### H1 (most likely): the slot's ring is primed from the main session's ring, which in
serve mode is a **template** that has only seen the **prompt** (and the tail of the
previous turn for multi-turn), **not** the slot's own committed rows. The priming is a
one-time copy at admission. After that, the slot's ring accumulates its own committed
rows via `draft()`. **This is correct for a single-turn conversation** (the prompt is
the full context), so it should not explain the tuning's single-turn penalty. **Ruled
out as the primary cause for the tuning, but a real risk for multi-turn serving.**

### H2 (likely): **ring-position drift from the shared staging buffer.** In batch, every
slot drafts into the same `df2_stage` buffer, and the graphs are baked on that pointer.
The `df2.draft()` call writes the slot's committed rows into the **slot's own ring** at
`pos % ring_slots`. But the **position table** (`ring_pos_`) is a host-side mirror that
is updated by `mirror_cell(sl, pos0 + i)` inside `draft()`. If the slot's ring was
primed from the main session's ring (H1), the position table maps ring slots to the
**prompt's** positions. When the slot then drafts at its own committed positions, the
`mirror_cell` writes update the table. **This is correct**, but only if the slot's
committed positions are **monotonically increasing** and **never collide** with a
prompt position that is still in the window. For a single-turn conversation this holds.
**Ruled out for single-turn, but a real risk if a slot's committed position wraps the
ring and collides with a prompt position.**

### H3 (likely, and the strongest candidate): **the drafter's attention window sees a
shorter or stale context in batch than in solo.** The drafter's ring is a sliding window
of `ring_slots_` (default 2064) cells. In solo, the ring accumulates every committed row
from the start of the conversation, so the drafter attends to the most recent 2048
committed rows **plus the prompt tail**. In batch, the slot's ring is primed at
admission with the prompt tail, and then accumulates the slot's committed rows. **The
difference:** in solo, the ring has been accumulating committed rows **continuously**
from the first window, so by the time the drafter drafts at window N, the ring holds the
most recent 2048 rows of the **entire** conversation (prompt + all committed rows up to
window N). In batch, the slot's ring is primed at admission with the **prompt tail only**
(the main session's ring has not been accumulating the slot's committed rows — the slot
is a different ring). So the slot's ring holds the prompt tail + the slot's committed
rows up to the current window. **These are the same set of rows** (prompt tail + committed
rows), so the drafter should attend to the same context. **Ruled out as a difference in
context content, but the priming is a host round-trip (save_state/restore_state) that
involves a device-to-host and host-to-device copy of the entire ring (L * 2 * ring_slots
* NKV * HD * elem_bytes bytes). If this copy is not perfectly faithful (e.g., a
fp16->fp32->fp16 round-trip loss, or a position-table mismatch), the slot's ring could
be subtly different from the solo ring.** This is the **strongest candidate** and the
one that is **not confirmable without a GPU A/B test**.

### H4 (possible): **the anchor token differs.** In solo, the anchor is `outv[a_]`
(the target's pick for the anchor position, read from the verifier's output). In batch,
the anchor is `sl.x` (the token the slot feeds to the next window). These **should** be
the same (both are the target's pick for the anchor position), but if `sl.x` is set from
a **different** source (e.g., the slot's own draft rather than the target's pick), the
anchor could differ. **Need to verify where `sl.x` is set in the batch loop.** (See
`generate.cpp:13537`: `sl.x = y;` where `y = outb[first[t] + j]` — the target's pick.
**So the anchor is correct. Ruled out.**)

## Verdict

The code-level diagnosis points to **H3 (a subtle ring-state difference from the
admission priming's host round-trip)** as the most likely cause of the batch penalty.
The priming is **correct in content** (prompt tail + committed rows) but involves a
**host round-trip** of the entire ring that could introduce a subtle numerical or
position-table difference. This is **not confirmable without a GPU A/B test** that
compares the slot's ring (after priming) to the solo ring (at the same conversation
point) and measures the drafter's output on both.

**Next steps (require a GPU window):**
1. Add a diagnostic that dumps the slot's ring (after priming) and the solo ring (at the
   same conversation point) and compares them (position table + K/V values). If they
   differ, the priming is the cause.
2. If the rings match, the penalty is not from the priming — look at the drafter's
   output (the draft tokens + probabilities) on both rings and compare. If the outputs
   differ, the drafter is seeing a different context (H2). If the outputs match, the
   penalty is from a different cause (H1 for multi-turn, or a drafter-quality issue).
3. If the rings and outputs match, the penalty is from the **verification** (the target
   model's acceptance of the batch draft vs the solo draft). In that case, the batch
   penalty is not a drafter issue at all — it is a **target-model** issue (the batch
   window's verification is different from the solo window's verification).

**This diagnosis is code-level only. It has not been confirmed by a GPU A/B test.**
The tuning's single-turn penalty is most likely H3, but H1/H2 are real risks for
multi-turn serving and ring-wrap collisions.

## What can be done without a GPU (this session)

- **C1 (n-gram co-draft)**: implement the n-gram co-draft for the batch DFlash2 windows.
  This is independent of the ring-priming diagnosis and can raise the batch acceptance
  on copy-heavy workloads (where the n-gram match is near-certain) without touching the
  ring. It also **reduces the drafter's workload** (the n-gram proposals are free — they
  come from a lookup, not a drafter round), which could partially offset the batch
  penalty.
- **A2 (confidence sweeps)**: the `--spec-min-p` and `STRATA_DFLASH_DEPTH_SLOPE` sweeps
  are config-only and can be run in a short GPU window.
