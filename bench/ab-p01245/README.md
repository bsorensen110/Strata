# P0/P1/P2/P4/P5 compound build A/B — 2026-11-19

Solo engine-only A/B: current production binary (baseline) vs the merged P0/P1/P2/P4/P5
build with the four new opt-in flags enabled (treatment). Matched serve layer (installed
v0.2.10 distribution), matched model/pack/args/env, `parallel: 1`.

## Arms

- **baseline**: `components/engine/fork-58aefafc9-p0/strata`, sha256 `108f83087d2e6f901120010c85918c9807856f4e9dabe17b5d728659a678ce95`. Production effective env (81 switches; P0 accept-hist + cand-dump already on). New flags absent.
- **treatment**: merged `fork-main` `7782928f2a74904b51dc5129d9c747ccec47ad44`, build `build-hip-gfx1201-p01245/strata`, sha256 `b53d8457fc4f60c5dd4fd8831d27d8a857e68496194544d99ace04f30dd0c327`. Same env plus `STRATA_DF2_TAP_OVERLAP_DECODE=1` (P1), `STRATA_DF2_ADAPTIVE_SOLO=1` (P2), `STRATA_DF2_SUFFIX_FILL=1` (P4), `STRATA_DF2_SAMPLE_WALK=1` (P5).

Both arms verified from the engine.log `engine started:` line: exe matches, `--draft-model …q4_0-selq4.pack` (live pack), `--draft dflash2 --spec 8 --draft-conf 0.35`, no `--batch` (solo). 54 gfx1201 code objects only. 17/17 focused host CTest passed on the treatment build (`spec_proposal_test` not registered → Not Run).

This is a **compound build comparison** (new binary + 4 flags), not per-feature attribution.

## Workload

Per launch: warmup (untimed) → greedy parity probe (fixed prompt, T0, max_tokens 256) → 2 greedy decode trials (T0, max_tokens 512, distinct markers) → 1 sampled trial (T0.7, max_tokens 512, exercises P5). 3 interleaved launches per arm (baseline,treatment ×3). Decode tok/s from the engine `N generated in X ms (Y tok/s)` line; acceptance from `drafts accepted A of B`.

Prompts were short (≈22–180 generated tokens; small prompt prefixes). This does **not** exercise suffix-fill (needs repeated history) or deep-context acceptance.

## Results (n=3 launches/arm)

- Greedy decode tok/s: baseline mean **62.8** (54.7–71.9, n6) vs treatment **60.6** (53.4–69.8, n6). Delta **−3.5%**. Ranges overlap → **not separable at this n**.
- Sampled decode tok/s: baseline mean **55.2** (53.8–56.5, n3) vs treatment **51.1** (44.6–57.3, n3). Delta **−7.4%**. Ranges overlap → **not separable**.
- Greedy acceptance ratio: baseline ≈0.44, treatment ≈0.36 (treatment offered more drafts, similar absolute accepts).

## Parity / correctness

Greedy parity probe produced **3 different SHA-256 across the baseline's own 3 launches** for the identical fixed T0 prompt (b412bfaf…, 83ececf4…, 5be464e5…). Greedy output is non-deterministic across fresh launches on this build/hardware (consistent with expert-placement timing; see llm-engine-gap-review). Bit-parity therefore **not assessable** via whole-reply hash; treatment#3 coincidentally matched baseline#1. Not a promotion gate here (user allows differing-bits opt-in variants when faster), but reported honestly.

## Feature dispatch (treatment engine.log)

- P0 accept-hist: **engaged** — `[accept-hist] req=N df2_pos_reached=… df2_pos_accept=…` lines present.
- P4 suffix fill: composition path **runs** (`df2 suffix composition: p=… df2=… suffix=… lookup=…`), but `suffix=0` on this short non-repetitive workload (no suffix matches found).
- P2 adaptive solo: env delivered; distinct P2 dispatch not isolated from the existing `STRATA_ADAPT_OVERLAP` lines — **dispatch unverified by log**.
- P5 sampled walk: env delivered; no distinct log line found — **dispatch unverified by log** (sampled trial T0.7 ran).

## Caveats

- Short workload; suffix-fill and deep-context acceptance not exercised. A longer/repetitive or >150k-token workload may change the result.
- Compound comparison; no per-feature ablation arms run.
- Greedy non-determinism prevents whole-reply bit-parity.
- P2/P5 kernel dispatch not confirmed from telemetry (env-set ≠ engaged).

## Production state

Production restored to the **previous binary/profile** (unchanged), verified: keyed `/v1/models` 200 (x-api-key), unkeyed 401, smoke `17*23=391` finish `stop`, binary + profile sha256 match the pre-window snapshot. **No promotion performed.** The new build is ready at `fork-main 7782928f2` with features verified opt-in; promotion is a separate decision (the build measured not-faster on this workload).

## Reproduce

```
python3 ab_p01245.py --treatment-exe <merged build>/strata --tree <integration worktree> --plan
python3 ab_p01245.py --treatment-exe <merged build>/strata --tree <integration worktree> --run --launches 3
```
Auth: this serve build accepts the key via `x-api-key` (Bearer alone returned 401). Raw records: `records.jsonl`; aggregate: `summary.json`; arm configs/env: `baseline.json/.env`, `treatment.json/.env` (mode 0600, key redacted here).
