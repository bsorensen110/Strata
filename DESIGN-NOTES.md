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

`--expert-cache-quota auto` is opt-in and reads per-expert miss counts from the `.heat` sidecar written alongside `--expert-profile-save`.  It uses each layer's actual `sized_slots` stride for native packs and greedily assigns integer slots by next-expert heat per byte, while preserving the uniform plan's total slot count and staying within the same byte budget.  It keeps at least one slot per layer; with no sidecar or all-zero heat it uses uniform per-layer quotas.  Startup reports the full quota vector.  A checked byte sum and assertion guard both initial planning and cache-allocation retries.  Existing behavior is unchanged unless the new option is used.

Runtime rebalancing is out of scope: it would require stopping routing and graph submissions, draining all consumer streams, evicting and refilling slots across different layer ranges, publishing a new residency table, and preserving the byte cap while accounting for unequal slot strides.  That is a separate state transition and needs its own ordering proof and live A/B validation.
