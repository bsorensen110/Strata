# Batch MTP test results (windows 2 and 3)

Engine: `integ/community-2026-10` at c510e2a3. Binary `build-verify/strata`, md5 prefix `ccd1b6351748`, built 2026-10-31 01:12 local, after c510e2a3 (01:10 EDT). `git diff bba6e346..HEAD -- src include tools` is empty. The commit 79c91e77 named in older notes is not in this repo.

Test plan: [BATCH-MTP-TEST-RUNS.md](BATCH-MTP-TEST-RUNS.md). Idea claims: [DECODE-DRAFT-SPEEDUP-IDEAS-2026-10-30.md](DECODE-DRAFT-SPEEDUP-IDEAS-2026-10-30.md).

Labels: **measured** = from engine logs or runner records. **source** = read in the code. **unverified** = not tested.

Hashes (sha256 prefixes, window 3): prod config `b83ac1ec209d7a31` and prod exe `0ae3e1dffcf727c6` (from `snapshot-before-w3.txt`); candidate exe `6303df64c7d87471` (recomputed from `build-verify/strata`).

## 1. The decode-side CPU switch

Decode misses (routed experts not in the VRAM cache) run on the GPU over PCIe or on the CPU. The split is set by `--pcie-frac F`.

- **source:** `pcie_num = round(F*256)` (`src/program/generate.cpp:10189`, set at `:10276`). The PCIe count per layer-window is `m = (nmiss * pcie_num) >> 8` (`src/core/expert_source.cpp:5504`). Prod uses F=0.15 (`pcie_num` 38): by that formula `m` is 0 while `nmiss` is 6 or less (computed, not measured). The comment at `expert_source.cpp:5505` describes the 1-2 miss case.
- **source:** F=1 makes every miss a PCIe candidate.
- **source:** a candidate falls back to the CPU once the layer-window has staged `P.staging_cap` blobs, or 64 (`expert_source.cpp:5538`). `staging_cap` is `kStagingBlobs` = 16 (`include/strata/core/verify.hpp:875`), or 8 per group when two groups run (`src/core/verify.cpp:3949-3951`). No flag changes these.
- **source:** `--adapt-every` is parsed at `generate.cpp:3265`. Setting it to 0 turns the adaptive VRAM tier off; the engine prints `adaptive swaps off` (`generate.cpp:9387`; the window-3 A-1 log shows it).
- **source, other candidates:** `STRATA_BATCH_DECODE_SHARE` (default 0.5) paces prompt reads against slot decode (`generate.cpp:16828`). It does not place experts. `STRATA_CPU_YMM` picks the CPU kernel width (`src/kernels/cpu/expert.cpp:151`). `STRATA_VERIFY_DEFER_CPU` (`=1` or `=check`) and `STRATA_VERIFY_DEFER_CPU_BELOW` are opt-in deferral knobs (`src/core/verify.cpp:1037-1044`); their effect on expert placement is **unverified**. `STRATA_PCIE_BALANCE=1` applies only when `--pcie-frac` is not given and batch serving is off (`generate.cpp:3663`, `:4348-4353`).
- **measured, window 3 parity arm:** VRAM 66.6%, PCIe 28.2%, CPU 5.1% of routed entries (about 193 CPU entries per window).
- **measured, window 2 base (seven launches, recomputed from the engine logs):** CPU expert time 8.4–8.8 ms per window; GPU wait 29.5–29.9 ms per window; routed entries VRAM 92.9–93.0%, PCIe 0.43–0.47%, CPU 6.6–6.7%.

Result: `--pcie-frac 1 --adapt-every 0` is the closest flag set to GPU-only decode. It is not CPU-free in this build: about 5% of routed entries still run on the CPU. Zero CPU needs a change to the staging cap and the 64 limit. That change is not made.

## 2. Window 3: depth 1 plus all switches except EARLYCOPY and DEVTOK

Batch 4, `--batch-mtp`, default 8-row window. Both arms run in parity mode: `--pcie-frac 1 --adapt-every 0`, `STRATA_PREFILL_CPU_SHARE=0`, `STRATA_BATCH_DRAFT_TIMING=1`.

- A (control): parity mode only.
- T (treatment): A plus `STRATA_BATCH_MTP_DEPTH=1 STRATA_BATCH_MTP_DEVACCEPT=1 STRATA_BATCH_MTP_SELECT=1 STRATA_BATCH_MTP_SELECT_OVERLAP=2 STRATA_BATCH_EXPERT_AHEAD=1 STRATA_BATCH_DRAFT_GROUP=1`. T is one combined arm, so no single switch is attributed.
- Schedule A#1, T#1, A#2, T#2. Concurrency: 4 requests of 256 tokens, 3 rounds, with `[round N]` prefixes. Solo: greedy code, 512 tokens, 2 trials. Parity probe: 4 prompts, 384 max tokens, temperature 0.
- Prod stopped 03:15:21, restored 03:23:49 (section 6).

### Gate (T, from engine logs)

| Switch | Result |
|---|---|
| DEPTH=1 | Engaged: `MTP depth: ... 1.00 proposals per slot-window`. No `adaptive depth` readiness banner at N=1 (measured). |
| DEVACCEPT=1 | Refused: `STRATA_BATCH_MTP_DEVACCEPT is off: it needs STRATA_BATCH_MTP_DEVTOK=1` (measured). |
| SELECT=1 | Banner printed: `it only acts when the live slots do not all fit a window`. At batch 4 with the 8-row window, 4 slots of 2 rows fill the window, so it did not act. Inferred from the banner and about 7.4 rows per window. |
| SELECT_OVERLAP=2 | No banner (neither engaged nor refused). No effect is possible while SELECT is inactive. |
| EXPERT_AHEAD=1 | Engaged: `the CPU's experts are asked for 1 layer(s) ahead`. CPU routed entries unchanged (measured). |
| DRAFT_GROUP=1 | Engaged: draft line `one group round per window`, 7.7 co-draft calls per window (measured). |

### Speed (two launches per arm)

| | A#1 | A#2 | T#1 | T#2 |
|---|---|---|---|---|
| solo cold, tok/s | 75.99 | 75.76 | 75.82 | 75.75 |
| solo warm, tok/s | 80.64 | 80.50 | 80.14 | 80.54 |
| concurrent agg, rounds 1/2/3 | 92.66 / 89.39 / 89.48 | 91.24 / 88.69 / 89.16 | 95.33 / 91.43 / 91.60 | 95.00 / 91.09 / 91.68 |
| concurrent mean | 90.51 | 89.70 | 92.79 | 92.59 |
| ms per window | 68.46 | 68.81 | 65.82 | 66.21 |
| draft ms per window (co-draft calls) | 5.80 (3.9) | 5.77 (3.9) | 3.02 (7.7) | 3.04 (7.8) |
| emit ms per window | 5.83 | 5.80 | 3.34 | 3.36 |

Arm means: solo cold 75.88 (A) vs 75.78 (T), −0.1%. Solo warm 80.57 vs 80.34, −0.3%. Concurrent 90.10 vs 92.69, +2.9% (+2.5% and +3.2% per pair). ms per window 68.6 vs 66.0, −3.8%. Draft time 5.8 vs 3.0 ms per window, −48%. Launch-to-launch spread within A: 0.8 tok/s concurrent (0.9%).

Acceptance on the depth counter (T-1 log, 3 lines): 90.5%, 89.8%, 89.8%. The base arm has no depth counter, so no cross-arm acceptance comparison is made.

### Parity (temperature 0)

- Parity probe, 4 prompts: A#1 = A#2 4/4. T#1 = T#2 4/4. T#1 = A#1 4/4. T#2 = A#2 4/4. Tokens 365, 383, 156, 380 in every launch.
- Solo, 2 trials: 2/2 in every comparison above.
- Concurrent, 12 slot texts per comparison: A#1 vs A#2 7/12 match. T#1 vs T#2 5/12. A vs T 13/24 (11 differ). Identical base launches differ in concurrency, so the concurrent T-vs-A differences are not attributable.
- Parity mode is a different numeric path from the prod-like base: 3 of 4 probe texts differ from the window-2 base launch A-1.

### Window 2 for context (non-parity; concurrent aggregate mean; solo cold/warm)

- A base (n=7): 126.6; 91.1 / 109.5.
- C depth 3, default window (n=1): 114.2 (−9.8%).
- D depth 3 with `--batch-draft-window 16` (n=1): 118.2 (−6.6%).
- H depth 3 with group (n=1): 117.1 (−7.5%).
- F EARLYCOPY (n=1): 92.2 (−27%).
- I1 / I2 EXPERT_AHEAD 1 / 2 (n=1 each): 126.2 / 126.4 (−0.3% / −0.2%).
- B, E, G (DEVTOK refused): no switch effect beyond the D arm.
- Batch 6: J0 control (n=2) 123.6; J1 SELECT (n=1) 121.0 (−2.1%); J2 SELECT plus OVERLAP (n=1) 121.4 (−1.8%).

### Parity-mode cost (vs window-2 base)

Concurrent 90.1 vs 126.6 (−29%). Solo warm 80.6 vs 109.5 (−26%). Solo cold 75.9 vs 91.1 (−17%). GPU wait rose from 29.5–29.9 to 52.8–53.0 ms per window. Routed entries streamed over PCIe rose from 0.4–0.5% to 28%, and the VRAM share fell from 93% to 67% with the tier frozen. The two causes (`--pcie-frac 1` and `--adapt-every 0`) are not separated.

## 3. Verdict (window 3)

- Solo: no change.
- Concurrent: +2.9% on two launches per arm. The gap is larger than the within-arm spread, but n=2 does not establish it.
- Draft time: DRAFT_GROUP cuts it by about half (measured). Acceptance is not compared across arms.
- Not measured: SELECT and SELECT_OVERLAP at batch 6, where the plan tests them.

## 4. Unverified

- Why concurrent texts do not reproduce. Candidates: the CPU fallback varies with window composition, and batch composition depends on timing. The solo CPU share was not measured.
- EXPERT_AHEAD: no effect beyond its counters (two launches).
- Emit drop with T (5.8 to 3.4 ms per window): mechanism not established.

## 5. Reproduce

- Runner: `~/Strata-ab/bba6e346-test/ab3_run.py`, a copy of `ab2_run.py` with arms A and T. Run: `cd ~/Strata-ab/bba6e346-test && python3 -u ab3_run.py --tag w3 --schedule A#1,T#1,A#2,T#2`. About 8.5 minutes; it stops and restores prod.
- Output: `runs/results-w3.json`, `runs/<id>/engine.log`, `runs/<id>/parity.json`, `restore-w3.json`.
- Analysis: `python3 analyze_w3.py` writes `summary-w3.json`.
- Window 2: `runs-w2/results-main.json`.
- The usage line in `ab3_run.py` still says `ab2_run.py` (cosmetic).

## 6. Prod

Prod stopped 03:15:21 and restored 03:23:49: ready, unkeyed 401, smoke finish `stop` with 42, config and exe SHA matching the snapshot, unit active, listening on 192.168.1.156:8086.

Prod config at test time (`~/Strata/strata-swift-iq3_s.json`): `parallel` 2 (batch 2) and `--batch-mtp --spec 8 --spec-min-p 0.5 --pcie-frac 0.15 --adapt-async 1 --expert-cache auto --resident-experts --prefill 8192 --vram-reserve-mib 832 --kv int8`. The process argv is only the serve wrapper (`serve.server --engine strata --config ... --port 8086`).
