# DFlash2 batch/solo acceptance tuning — 2026-10-29

## Verdict

No DFlash2 batch variant passed the throughput gate. The best batch acceptance result was the shallow one-proposal arm, but it was still 16.2% slower than plain batch and 24.6% slower than batch MTP. Production remains inactive; nothing was promoted or started. The runtime `/dev/null` link is not an effective mask (systemd still reports the unit enabled); see the service-state caveat below.

## Method and verification

- Same target, DFlash2 pack, and production-matched candidate binary for every arm. Candidate SHA-256: `b7af4afdb505ab4671d6f185d846a48afe1d9fb1580f45020fa4b19c90e52cfc`.
- Build cache: Release, HIP enabled, tests enabled, `STRATA_PREFILL_MMQ=ON`, `CMAKE_HIP_ARCHITECTURES=gfx1201;gfx1200`, `GGML_HIP=OFF`.
- Four-way batch trials used temperature 0, up to 256 output tokens, three measured repetitions per arm, a single 8-token warm-up plus a four-slot 64-token warm-up, `--expert-cache auto`, `--prefill 512`, `--spec-min-p 0.5`, and a 16-row DFlash2 window. Solo arms used true `parallel=1`, no batch options, and two warm-ups.
- All configurations had the same target/model/build and production tuning; DFlash2/MTP/plain controls were kept separate. DFlash2 acceptance is weighted `accepted/offered proposals`; tokens per slot-round includes the target anchor. MTP's offer semantics differ, so its acceptance percentage is not compared to DFlash2.
- Live planner banners verified the actual slot/depth plan. None of the batch/acceptance override environment variables was set in the measured process.
- Both run final-state records show `strata.service` inactive, no GPU/candidate PIDs, candidate port free, and unchanged production config/binary hashes (`8f9ba350…` / `18fb5a3f…`). A filesystem link `/run/user/1000/systemd/user/strata.service -> /dev/null` exists, but a direct systemd query reports `UnitFileState=enabled` and `FragmentPath=/home/soren/.config/systemd/user/strata.service`; the runtime link does **not** mask this higher-priority user unit. Production is stopped now, but may autostart when the user manager starts again. No persistent unit change was made.

## Batch controls and confidence sweep

- Plain batch, six trials (A-B-A control): median **96.61 tok/s**, range 95.47–99.25.
- Batch MTP, three trials: median **107.40 tok/s**, range 103.02–109.68.
- DFlash2 cap 5, actual plan four selected slots × three proposal rows: initial default arm **1,144/2,768 = 41.33%**, 2.252 tokens/slot-round, median **77.06 tok/s**. A separate pre/post control around the depth sweep pooled to **2,253/5,472 = 41.17%**, 2.237 tokens/slot-round, median **78.00 tok/s** (75.20–79.54).
- The requested-depth 1/2/3 arms at `--draft-cap 5` all executed the same effective four-slots × three-rows plan. They are not distinct depth measurements.
- `--draft-conf` did not materially raise acceptance at actual depth 3: 0.1 → **40.53%** (median 77.07 tok/s), 0.25 → **40.69%** (77.48 tok/s), 0.5 → **41.19%** (78.42 tok/s), versus 41.33% with no explicit confidence threshold. The rates and throughput spreads overlap.

## Corrected effective-depth sweep — C4, 16-row window

`--draft-cap N` includes the anchor token; a cap of N permits at most N−1 DFlash proposals. Each entry is three measured trials. The planned shape is the live startup plan; average shape is observed during generation as slots finish at different times.

- **Depth 1, cap 2; plan 4×1; average 3.86×1.00:** 832/1,239 = **67.15%**, 1.665 tokens/slot-round; median **80.96 tok/s** (79.45–82.14).
- **Depth 2, cap 3; plan 4×2; average 3.85×2.00:** 1,051/1,966 = **53.46%**, 2.056 tokens/slot-round; median **78.16 tok/s** (77.22–79.59).
- **Depth 3, cap 4; plan 4×3; average 3.88×3.00:** 1,138/2,709 = **42.01%**, 2.244 tokens/slot-round; median **77.23 tok/s** (74.63–78.58).
- **Depth 4, cap 5; plan 3×4; average 2.93×4.00:** 1,180/3,424 = **34.46%**, 2.359 tokens/slot-round; median **69.23 tok/s** (69.14–69.31).
- **Depth 5, cap 6; plan 2×5; average 1.99×5.00:** 1,195/3,990 = **29.95%**, 2.475 tokens/slot-round; median **62.22 tok/s** (60.87–62.72).
- **Depth 6, cap 7; plan 2×6; average 1.99×6.00:** 1,240/4,806 = **25.80%**, 2.525 tokens/slot-round; median **58.29 tok/s** (56.14–58.80).
- **Depth 7, cap 8; plan 2×7; average 1.99×7.00:** 1,220/5,432 = **22.46%**, 2.548 tokens/slot-round; median **54.11 tok/s** (53.34–56.20).

Shallow depth raises the accepted/offered fraction, but the extra acceptance does not make DFlash2 competitive with plain batch or MTP. Deeper plans select fewer of the four active slots and lose throughput sharply. The depth-1 arm is still about 16% below plain batch.

## True-solo cap sweep

Each cap is the verify window in tokens including the anchor; all nine responses ended with `finish_reason=stop`.

- Cap 3 (max 2 proposals): 331/439 = **75.40%**; median **79.12 tok/s** (73.77–79.78).
- Cap 5 (max 4 proposals): 377/609 = **61.90%**; median **89.07 tok/s** (81.61–90.22).
- Cap 7 (max 6 proposals): 378/703 = **53.77%**; median **84.51 tok/s** (82.99–87.77).

Cap 3 is the acceptance choice; cap 5 was fastest in this three-trial DFlash-only solo comparison. There was no solo drafter-off control, so these numbers rank DFlash caps, not solo DFlash versus target-only throughput.

## Caveats and next levers

- DFlash2 C4 loaded 10,752 expert-cache slots and reported 404 MiB free at startup; a measured request fell to 260 MiB. The production-matched config has `--vram-reserve-mib 704`, and the engine warned below its 512 MiB paging threshold. Plain reported 11,280 slots / 642 MiB free; MTP 11,184 / 590 MiB. Memory pressure is part of this production-config result; isolate it with a separate, matched headroom/cache A/B before assigning the entire speed loss to draft computation.
- To raise acceptance percentage immediately, use lower effective depth: batch cap 2/depth 1 or solo cap 3. This raises the ratio partly by offering fewer proposals; track tokens/slot-round and tok/s so the metric is not optimized by denominator reduction alone.
- The confidence sweep changed only the cumulative `--draft-conf` floor while `--spec-min-p 0.5` remained active. Next useful gates are a one-variable per-token `--spec-min-p` sweep and a per-depth `STRATA_DFLASH_DEPTH_SLOPE` sweep, reporting acceptance, yield, VRAM, and end-to-end throughput together.
- For higher match quality rather than fewer offers, test a DFlash2 pack trained or calibrated against the exact target weights/quantization and workload. The active tree has no DFlash2 beam-search CLI knob; a beam/reranker would be new code, not a config tweak. The opt-in `STRATA_DFLASH_CONF_ADAPT` path was not tested; its current policy construction sets `max_k=1`, so review/fix and test that path before relying on it as adaptive.

## Raw evidence

Raw JSONL, logs, sanitized configs, manifests and state records remain in the owner-local archive `~/Strata-ab/results/dflash2-batch-tuning-20261029/`; they are not checked in because they include machine-specific paths and large run logs. The initial runner's summarization step raised `NameError: statistics` after all 27 batch and 9 solo trials had been recorded. Raw data was preserved and aggregated into `recovered_summary.json`; the corrected effective-depth runner completed its summary successfully. No production deployment files or binaries were changed.
