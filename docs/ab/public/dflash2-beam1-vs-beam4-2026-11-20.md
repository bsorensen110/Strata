# DFlash2: Beam 1 vs. Beam 4 diagnostic, 2026-11-20

**Decision: keep Beam 4.** The completed matched window yielded five cold deep-context pairs and two repeat controls. Beam 1 did not improve acceptance or throughput: 88.79 vs. 89.93 pooled decode tok/s (−1.27%), within the observed same-beam spread. Neither beam reproduced its own greedy reply bytes on the cold repeat, so this is not a test of selector-specific output correctness. Production was restored unchanged. No selector or tree was promoted.

## Completed matched comparison

The qualified comparison completed 12/12 measurements on four fresh solo boots in order **4 → 1 → 1 → 4**. Primary pairs used boots 1/2 for CPython, CMake and systemd and boots 4/3 for Go and Git. The extra CPython runs (boots 3 and 4) are A/A controls only; warm-ups and the earlier live reference are not pooled. The 164,601–248,723-token inputs matched byte for byte, with zero tokens reused, temperature 0, thinking off and 1,024 generated tokens per request. Both arms ran the same build/pack, verifier settings, runtime heat pair and initial expert quota vector. All boots logged 10,896 GPU expert slots (24.68 GiB), 35.94 GiB pinned complement and 1,980/1,980 RAM-backed lend slots. This matches startup placement, not every runtime routing or cache decision.

Pooled input tokens: **1,058,536 per arm**. Beam 4 decode rate: **89.93** tok/s; Beam 1: **88.79** tok/s; Beam 1 vs. Beam 4: **−1.27%**.

Pooled decode rate is 5,120 generated tokens divided by summed engine decode time (56.933 s for Beam 4, 57.664 s for Beam 1). Pooled client-wall rate was 14.347 vs. 14.261 generated tok/s (−0.60%); prompt prefill rate was 3,534 vs. 3,517 tok/s (−0.47%). Beam 1 led on 2/5 prompts. The paired aggregate is not separable at this n; the CPython repeat controls are not a workload-wide noise estimate. **Byte comparisons: 0/5 cross-beam pairs and 0/2 same-beam repeats matched.** This establishes that whole-reply bytes are not a stable oracle in this setup; it does not identify the cause or show a selector-specific change in quality.

| DF2 position | Beam 4 accepted / reached | Beam 1 accepted / reached |
|---|---:|---:|
| 1 | 1,203/1,598 | 1,223/1,652 |
| 2 | 752/1,055 | 779/1,086 |
| 3 | 505/652 | 489/672 |
| 4 | 339/438 | 334/422 |
| 5 | 247/302 | 234/295 |
| 6 | 171/208 | 147/189 |
| 7 | 112/142 | 99/121 |

- **Acceptance:** mean conditional rate at positions 2–7 was **78.17% vs. 77.09%**, with minimum reached counts 142 and 121. Both meet the sample floor and fall in the `<80%` steep band on these fixtures, not necessarily on production traffic. Neural DF2 accepted/offered was 3,329/6,229 vs. 3,305/6,239, or **1.928 vs. 1.879 accepted per total verify window**. Suffix/lookup was 48/123 vs. 58/142; chain was 20/108 vs. 9/63. These source counts are not interchangeable with position-wise conditional rates.
- **Work:** Beam 4 ran 1,727 windows, Beam 1 ran 1,759. Logical verifier rows were 8,187 vs. 8,203 (**+0.20%**); the `windows + offered draft rows` ledger reconciled with the independently logged rounded `avg T` for all 12 requests. This is logical work, not a physical GPU-dispatch count. Whole-phase draft time per window was 3.19 vs. 3.20 ms; verifier time was 29.54 vs. 29.35 ms. Beam 1 emitted fewer tokens per window (2.911 vs. 2.965) without a measured drafting-time saving. These are end-to-end window outcomes, not selector-kernel timings or fixed-prefix counterfactuals.
- **Where to look:** 698/1,066 Beam-4 first neural rejections were at positions 1 or 2 (Beam 1: 736/1,132). Raw top-16 recall at the first neural rejection was 861/1,066 and 924/1,132, respectively. The beams diverge onto different prefixes, and these v1 dumps lack predecessor-conditioned transitions and can omit zero-DF2 windows. Neither the higher raw recall in Beam 1 nor the steep curves prove a tree would help.
- **Recovery:** the driver exited 0 with a 12-row completion ledger. The verified window was 1,058.58 s from stop through restore. Production reported the original Beam 4 binary/config/argv, keyed HTTP 200, unkeyed HTTP 401, a correct stop-terminated completion, and no remaining candidate listener or restore timer. This verifies service health, not post-restore throughput.

## Verified Beam-4 reference (not an A/B arm)

Five separate technical corpora from local CPython, CMake, systemd, Go and Git sources. The manifest audit found 354 distinct source paths and 354 distinct included-content hashes, with no overlap across corpora. These are technical fixtures, not a random sample of production traffic. All five completed at temperature 0 with thinking disabled, 164,601–248,723 actual prompt tokens, zero reuse, and 1,024 generated tokens (`finish_reason=length`). The five prompt-file hashes match the corpus and result manifests. An independent reconstruction from the offered width and first-rejection offsets reproduced each histogram.

| DF2 proposal position | Accepted / reached |
|---|---:|
| 1 | 1,234/1,591 |
| 2 | 784/1,080 |
| 3 | 525/672 |
| 4 | 345/452 |
| 5 | 242/304 |
| 6 | 161/209 |
| 7 | 110/136 |

- Mean conditional acceptance at positions 2–7: **77.43%** (minimum 136 reaches). 2.138 accepted DF2 tokens per **observed DF2-bearing round**, not per total verifier round; zero-DF2 windows can be absent. Suffix/lookup 23/45; appended chain 2/25.
- First DF2 rejections: positions 1/2 accounted for **653/1,043**; the verifier token was in raw top 16 on 834/1,043 (325/357 at position 1). At position 1, over 1,591 observed windows, raw top-1 matched 1,238 vs. Beam 4's 1,234 (23 raw-only vs. 19 Beam-4-only matches). Raw top-1 is not Beam 1; these are not counterfactual rollout results.
- Reference throughput: **92.38 decode tok/s** pooled across completed outputs; **14.18 end-to-end generated tok/s** pooled across client wall time. Not a clean prospective control for the instrumented scratch benchmark.

The earlier [archived-corpus pilot](dflash2-p0-live-technical-replay-2026-11-20.md) is not pooled here. Its 77.89% is computed from six trial histograms; a separate aggregate reports 77.36% from five histograms. These are different cohorts, not a rounding disagreement.

## Attempt and recovery

- Production-matched settings used Qwen3.8 Flash Next UD-IQ4_XS v31 and a `selq4` DFlash2 pack on an R9700. Solo, `--spec 8 --draft-conf 0.35 --draft-fallback none --lookup-max-t 16 --lookup-chain 6`, `--max-context 409600`, int8 KV with 32,768 resident cells. `STRATA_LEND_RAM_BACKUP=1`. Operational artifact identifiers are private.
- The `drafting: auto` CLI plan initially named a different Q4 pack than the loaded `selq4` pack. A scratch profile used resolved live inputs with overrides folded into the resolved argv, preserving every effective argument except the two scratch output paths. This was checked against the installed serve resolver, all four CLI plans, and the started scratch engine process. The banner confirmed `selq4` and no loaded MTP. The plan alone can miss a pack substitution.
- The first scratch Beam-4 unit reached readiness, authorized and unauthorized `/v1/models` checks passed, and the short warm-up produced draft/timing telemetry. It booted with 10,896 GPU expert slots and 2,001/2,001 RAM-backed prompt-lend slots. Its complement was 36.26 GiB, versus 35.99 GiB and 1,992 backed lend slots on the restored production boot. Equal argv or expert-cache slot totals do not guarantee equal allocation/placement across launches.
- A replay transport whose non-streaming default was 230 s was replaced with one whose default was 8 s. The client timed out and the engine cancelled the request during prefill. That cancellation is not a drafter failure or measured performance. A delayed loopback test reproduced the short default failing and the explicit long-timeout adapter succeeding, including preservation of the caller's 10 s model listing and 30 s token-count overrides.
- The exclusive driver stopped the candidate, checked GPU, VRAM and port release, restarted the unchanged production unit, and verified the original argv, stable environment and file hashes. Authenticated `/v1/models` returned 200, unauthenticated returned 401, and a generation probe completed with `finish_reason=stop`. Restoration readback confirmed zero candidate or recovery units left running.
- A subsequent production Beam-4 repeat reused the persistent conversation cache across the restart. It is **not** a cold A/A or parity sample. Leave production cache intact; use fresh, empty scratch cache paths for both arms.
- A separate transport-only probe used a new input marker to avoid cache reuse. It validated the 230 s client adapter on a cold request; the modified input was excluded from A/A and A/B scoring.

## Approved retry: startup heat provenance guard (no timing data)

An earlier attempt used the corrected timeout and copied the same startup quota heat to all four arms. The `.heat` payload was staged without its `.heat.meta` file. The first scratch Beam-4 process reached API readiness, loaded `selq4`, and reported a learned 48-layer quota totaling 10,896 expert slots, a 35.99 GiB pinned complement and 1,992/1,992 RAM-backed lend slots. But the exact banner was `retaining existing legacy-unmarked heat sidecar`, not `runtime`. The harness stopped before warm-up or any long request. This was a staging error, not a DFlash2 refusal. It yielded 0/12 qualifying requests. Recovery took 89.04 s from stop through verification: original binary/profile/argv and Beam 4, HTTP 200 keyed / 401 unkeyed, a successful stop-terminated completion, and no active candidate or recovery timer.

The corrected **read-only preflight** staged byte-identical copies of the live runtime `.heat` **and** `.heat.meta` under all four redirected `--expert-profile-save` paths. It verified the STRH-v1 48×512 tensor geometry, finite nonnegative values, runtime metadata and the deployed payload checksum. Missing and mismatched metadata were rejected; the CPU `expert_heat_bootstrap_test` passed 1/1. All four exact `serve --plan` checks and empty-cache checks passed, without a GPU launch or a production change. This fixes the known preflight issue, not the unrun comparison. Revalidate the seed from a stable production snapshot before the next window and require matching loaded quota vectors, cache size, complement and lend counts at startup. Do not reuse staged configs if the running production artifacts or argv drift.

## Corrected provenance; rejected historical allocation anchor

Another boot loaded the correct runtime `.heat` plus `.heat.meta` pair and the intended DFlash2 pack. It was falsely rejected against an older heat seed's allocation: both had 10,896 slots, but the earlier boot used 24.63 GiB VRAM and 1,992 backed lend slots, versus 24.68 GiB and 1,980 slots on the new seed. No measured request had started. Independent readback after the 87.89 s restore confirmed that the latest scratch and restored production boots had **identical** 48-layer quota vectors, cache bytes as reported, KV shape, a 35.94 GiB pinned complement and 1,980/1,980 backed lend slots. The corrected driver compares every arm to the current production boot and the same staged heat pair, not the archive. The corrected CLI plan also preserves the explicit `split_engine_note` bypass: without it the installed resolver substituted the non-`selq4` pack and changed `--spec 8` and confidence `0.35` to `6`/`0.7`; this was caught before any GPU launch.

## Reproduction and next gate

Raw evidence, prompts, candidate traces, logs, environment files, and the campaign harness are private and are not published. Per-pair results and repeat controls were cross-checked by independently parsing each primary request with [the repository acceptance-curve tool](../../../tools/accept_curve.py) and directly comparing reply bytes. The measured build included an uncommitted P0 telemetry diff. The historical public source reference `029f1452c:src/core/dflash2.cpp:51–58,979–982` shows that Beam 1 and Beam 4 select within the same draft dispatch. The private driver, run only under outage approval, checked the actual child environment, argv, loaded pack and startup shape, verified zero reuse and actual counts for each request, and restored production on exit. No engine code was changed for this comparison.

The CPU selector reference still passes all 8 unit tests. It optimizes **uncut** draft prefix survival, then applies the cap/gate. A two-step synthetic check separates that heuristic from directly optimizing the post-cutoff prefix: path A has cumulative probabilities `0.61, 0.305`, path B `0.39, 0.351`; at `0.35`, the current reference prefix heuristic chooses A, but the gated objective prefers B (`0.741` vs. `0.61`). For normalized predecessor scores, prefix masses at each depth sum to 1, hence at most two prefixes can survive a 0.35 threshold; an exact gate-aware enumeration needs at most 208 candidate extensions for seven rows with 16 candidates each, **excluding** transition-score computation/normalization. That's a search bound, not a GPU speed prediction.

Beam 1 did not resolve the loss. The next selector experiment should establish a repeatable target reference at fixed prefixes (including logits and state at the first same-arm divergence), then capture exact predecessor-conditioned transitions, confidence/cap and zero-offer windows. Replay Beam 1/4, exact full-path MAP (pruning control) and the gate-aware prefix objective on the same inputs. Record an independent target continuation on each lineage or explicitly censor unknown continuations. Compare accepted tokens at a matched verifier-row budget before changing the runtime selector. A tree still needs branch-state/device correctness and a repeatable throughput gain; the benchmark does not clear those gates.
