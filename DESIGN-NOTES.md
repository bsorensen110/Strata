# Structural VRAM Design Notes

## Phase 2 — Safe Boundary for Cache-Backed Late Allocations

The cache-backed transfer of ExpertCache physical chunks to prompt taps, verifier storage, DFlash2/logits, or captured-graph pools is **not implemented**.  The current code has useful idle points, but it does not prove that every consumer has stopped using every address in a tail segment, nor does it have one ownership lock covering source exchange, residency publication, prefill, adaptation, and graph submission.  A device-wide fence proves prior work is complete; without a lock or equivalent submission gate, it does not prevent new work from starting against a segment after that fence.

The safe Phase 2 changes are deliberately limited to:

- report mapped-slot and live-slot counts for segmented ExpertCache at startup;
- reject slot selection and device pointers outside the currently mapped prefix;
- retain the existing serialized `/v1/vram` shrink/grow path and its existing consumer behavior;
- defer shared physical-chunk transfer and graph-backed consumer pools.

`--vram-elastic` remains opt-in.  The default single-allocation cache and its sizing/accounting are unchanged.

### Ordering and ownership evidence

| Consumer / pool | Producer | Consumer and stream | Existing event or fence | Quiesce point / decision |
|---|---|---|---|---|
| ExpertCache slots and residency table | Startup/profile fills and later queued refills into `ExpertCache` slots; `host_res` is copied into `d_res` by `res_put` | Verify kernels, prompt expert kernels, and adaptation; verifier/prefill/adaptation use distinct streams and graph submissions | Refill uses `sync_queued`; the VRAM command calls `ver.wait_commit`, `adapt_quiesce`, then `cudaDeviceSynchronize`; it calls `res_upload` after evict/refill updates | Current command is accepted only between requests and refuses while a prompt loan is out.  Safe for the current serialized resize path, but no single `expert_source` exchange-ownership lock spans every publisher/consumer.  Do not transfer a segment to another pool. |
| Prompt loan / prompt taps | ExpertCache lends a tail region; `Prefill` owns staging and tap buffers | Prompt compute/copy streams (`m.cs` and its copy path), then verifier consumes selected taps | Prompt completion/refill path waits for its queued copies; `Prefill::idle` can drain its streams.  The VRAM command rejects non-empty `p.lent` | Relending occurs after resize, but these checks do not prove all long-lived tap references or any captured consumer graph have been destroyed.  Tap staging is excluded. |
| Verifier arena and verifier taps | `Verifier::init` allocates its arena and tap storage | Captured verifier graph executables replay on verifier streams | `ver.wait_commit` waits for the current committed window; device synchronization drains prior kernels | A commit boundary is not graph destruction.  Captures can retain pointers for later replay, so the arena and verifier taps must remain mapped until every referencing graph executable is destroyed and graph capture is rebuilt.  Excluded. |
| DFlash2 / logits buffers | Drafter bind allocators create logits, state, and scratch buffers | Drafter kernels and any graph-backed verifier/drafter path | No single cache-segment ownership event is tied to these allocations | Their allocation and graph lifetimes are not registered against ExpertCache ranges.  Excluded; graph sharing remains a stretch goal only if a separate lifetime proof is added. |
| Adaptive exchange and refill | `expert_source` stages host blobs; adaptation updates `host_res` and `d_res` | Adaptation stream plus later verify/prefill readers | `adapt_quiesce` and `res_upload` order current resize operations; `res_upload` uses the existing `res_put` copy path | Source exchange synchronization protects source buffers, not a global cache-range ownership transition.  No transfer under the current locks. |

### Required protocol before a shared chunk pool can ship

1. Add one ownership/submission gate spanning `expert_source` exchanges, prompt loan/refill, adaptation, `host_res`/`d_res` updates, and verifier graph submission.  A resize must hold it after stopping new submissions.
2. At shrink, wait for the prompt refill event(s), verifier commit, adaptation completion, and all other stream users.  Destroy or pin every graph executable that can reference the tail.  Only then clear affected residency entries and revoke access.
3. Revoke device access, fence/drain that operation, then unmap and release the physical chunk.  The current `VmmRange::unmap` does not independently establish a consumer quiescence point or explicitly revoke access; callers must not treat it as a cross-pool transfer primitive.
4. At grow, map the chunk, set access, refill every newly live slot, upload the residency table and wait for that publication before making those slots selectable.  Failure at any step must leave slots unavailable and report the failure.
5. Keep graph-baked regions mapped for each graph executable's entire lifetime.  Graph destruction and recapture need to be part of the same ownership gate, not inferred from an idle stream.

The current resize path's `cudaDeviceSynchronize`, loan check, verifier commit wait, and adaptation quiesce are useful precedents but do not replace these missing ownership and graph-lifetime guarantees.

### Host and live-hardware validation boundary

`expert_cache_live_slot_test` exercises the host-side mapped-prefix rule: the mapped prefix is selectable, the unmapped tail and negative indices are rejected, shrink-to-zero exposes none, and a restored prefix remains bounded by its budget.  It does not simulate VMM handle reuse, device access revocation, stream races, or graph replay.

The following assertions remain pending for an approved live gfx1201 A/B window; none were attempted because the production GPU must not be shared with another allocator:

- HIP `expert_cache_segmented_test` eager map, fill, shrink, grow, and byte verification on device memory.
- HIP `vmm_test` allocation/map/access/unmap behavior and allocation-failure behavior.
- Live `--vram-elastic` startup banner and repeated `/v1/vram` shrink/grow, checking refill correctness and stable outputs.
- Captured-graph replay and prompt-tap behavior across resize; all graph references must be inventoried before attempting shared-pool transfers.
- Physical-chunk transfer to each proposed consumer, including access revocation and residency publication ordering.  No such transfer is currently enabled.
- The `STRATA_EC_NO_VMM` compile-out variant; its explicit meaning is preserved, but a separate variant build remains to be run.

## Phase 3 — Startup-Only Quotas

`--expert-cache-quota auto` is opt-in and reads per-expert weights from a `.heat` (`STRH` v1) sidecar paired with `--expert-profile-save` or `--expert-profile`.  It uses each layer's actual `sized_slots` stride for native packs and greedily assigns integer slots by next-expert weight per byte, while preserving the uniform plan's total slot count and staying within the same byte budget.  It keeps at least one slot per layer; with no sidecar or all-zero heat it uses uniform per-layer quotas.  Startup reports the full quota vector.  A checked byte sum and assertion guard both initial planning and cache-allocation retries.  Existing behavior is unchanged unless the new option is used.

Runtime rebalancing is out of scope: it would require stopping routing and graph submissions, draining all consumer streams, evicting and refilling slots across different layer ranges, publishing a new residency table, and preserving the byte cap while accounting for unequal slot strides.  That is a separate state transition and needs its own ordering proof and live A/B validation.

### Heat Sidecar Bootstrap, Precedence, and Provenance

When `--expert-profile-save P` is enabled at startup and no valid `.heat` sidecar is selected yet, the engine bootstraps a deterministic **profile-rank prior** atomically to `P.heat` and `P.heat.meta` (still opt-in for quota selection: only `--expert-cache-quota auto` consumes it).

- **Prior derivation order**: read ranking from `P` (`--expert-profile-save`) if it exists and validates for the current `(n_layers, n_expert)`; otherwise fall back to the loaded `--expert-profile` ranking.  Each first-seen `(layer, expert)` at rank `r` gets finite weight `R - r`; unranked experts get `0.0`.
- **Lookup precedence**:
  1. `<expert-profile-save>.heat` if valid `runtime` or legacy unmarked (`STRH` v1 without `.meta`).  If `<expert-profile-save>.heat` exists and is malformed/incompatible, leave it untouched (warn at startup; fail closed under `--expert-cache-quota auto`).
  2. `<expert-profile>.heat` if valid `runtime` or legacy unmarked.  If `<expert-profile>.heat` exists and is malformed/incompatible, leave it untouched and do not shadow it with a prior (fail closed under `--expert-cache-quota auto`).
  3. `<expert-profile-save>.heat` if valid `profile-rank prior`.
  4. `<expert-profile>.heat` if valid `profile-rank prior`.
- **Provenance and crash safety**: `.heat.meta` records `provenance=profile-rank-prior` vs `provenance=runtime`, the source profile, dimensions, and `heat_fnv1a64`.  Existing valid `.heat` files without `.meta` remain accepted as legacy unmarked measured heat.  Bootstrapping stages `.meta` before `.heat`; runtime saving stages both `.tmp` files first and skips overwrites when zero runtime routing counts were observed, then renames `.heat` before `.meta` so a failure or crash never silently upgrades a prior's provenance to `runtime`.  `--expert-cache-quota auto` logs `profile-rank prior` (never `learned`) when allocating from a prior.
- **Developer note**: Profile-rank priors are synthetic monotonic ordering weights derived from profile rank, **not** true observed route counts.  Actual live traffic and periodic/`QUIT` saves under `--expert-profile-save` overwrite both `P.heat` and `P.heat.meta` with real routing counts and upgrade provenance to `runtime`.

## Fork-main integration (2026-11-14)

The history split matches the production snapshot exactly before integration.
The fork-main merge adds its CUDA DFlash2 head, selector and small-matrix series.
The conflict in `src/core/dflash2.cpp` combines both independent switches:
`STRATA_DF2_PRECOMP_GRAPH` and `STRATA_DF2_SELECT_TABLE`.

Buffer sizing keeps the VRAM stack's solo parent and clone allocations at eight
rows, and its multi-slot parent at 64 rows with 16 logit rows. The fork's sizing
fix calls the Q8 helpers with eight columns, because a default CUDA build rejects
a 16-column helper call. The merged code uses those legal calls, doubles the
ordinary Q8 capacity for batches, and retains the stack's arithmetic 16-column
interleaved capacity for wide-head builds. Allocation pricing uses the same
formulas. The fork's selector table is allocated and freed for each drafter;
its 8192 bytes are included in the allocation breakdown.

The deferred-tap chunking fix in `70e634cd7` remains unchanged: integration
does not change `src/program/generate.cpp`. CUDA selector/top-16/Q4 kernels and
their parity/round-benchmark sources retain the fork's changes. Validation here
is the HIP gfx1201 build and the six CPU-only host tests; no GPU workload is run.
CUDA and SYCL execution are not validated by this integration.
