# `perf/verify-wide-all` — gfx1201 batch-4 progressive A/B

## Scope and result

This is the five progressive configurations in `perf-verify-wide.md` §4.2, not the separate 15-arm micro-ablation matrix in `tools/bench_r9700_verify_wide.sh`. The branch was re-fetched before the run; `HEAD` and `fork/perf/verify-wide-all` were both `4c977f73e9457f2f18c0f51126bb66b01ebda11c`.

Candidate: AMD Radeon AI PRO R9700 (`gfx1201`), DFlash2 `qwen3.8-flash-next-dflash2.q4_0-selq4.pack`, batch 4; four fixed prompts; 128 generated tokens per slot (512 total); greedy; same wide-window-16 binary and production model/config for each arm. Every engine process's live environment and argv were checked; all 18 measured launches showed the DFlash2 startup banner, batch-4 route, and four 128-token outputs. Each run reached `STRATA_BATCH_DF2_TIMING` at window 32 with four active slots and `S > 8`. DFlash2's accepted-proposal *rate* is unavailable in this build: it emits `acc/win`, not an offered-proposal denominator. Do not interpret the legacy `MTP proposals accepted` line in the batch-DFlash2 run as DFlash2 acceptance; the denominator is not meaningful for that purpose.

| Progressive arm | Measured aggregate tok/s, n=3 (median; range) | Change vs prior progressive arm | Median full batch-window time | Checkpoint wide-window evidence |
|---|---:|---:|---:|---|
| Arm 0 — all new switches off | **96.6** (93.8–101.3), 6 runs including the baseline bracket | control | 85.6 ms/window | S=15–16; 12.34 rows/window at win=32 |
| Arm 1 — safe compute core | **99.1** (94.5–100.3) | +2.6% vs control | 84.4 ms/window | S=12–14; 12.03 rows/window |
| Arm 1+2 — plus gfx1201 WMMA dense/MoE | **93.5** (90.7–94.3) | −5.7% vs Arm 1 | 85.7 ms/window | S=14; 11.94 rows/window |
| Arm 1+2+3 — plus PCIe overlap and CPU miss reduction | **95.8** (92.2–96.1) | +2.5% vs Arm 1+2 | 79.4 ms/window | S=14; 11.94 rows/window |
| Arm 1+2+3+4 — plus adaptive T, tail-skip, route affinity, batch split | **104.1** (102.4–104.8) | +8.7% vs Arm 1+2+3 | 72.2 ms/window | S=9; 11.91 rows/window |

Versus the six-run baseline median (96.6 tok/s), the medians are +2.6%, −3.2%, −0.8%, and +7.8%. For a useful drift check, the three baseline runs before the treatment sequence had a 97.2 median; the three afterward had 96.0 (−1.2%). Thus Arm 1, Arm 1+2, and Arm 1+2+3 remain within observed baseline/run spread. The full Arm 1+2+3+4 combination is the only clear positive signal in this small campaign: +7.8% versus the six-run baseline median and +8.7% versus the preceding combination. Treat it as preliminary, not statistical proof; repeat matched full-stack/control trials before deciding to promote.

## Interpretation and caveats

- The test measures end-to-end throughput of four concurrent 128-token generations, including prompt admission. It is not single-request latency, and it does not validate multi-GPU/layer-pipelined behavior.
- The branch guide's *expected* verify-time reductions were not observed for isolated Arms 1–3: checkpoint `verify` was roughly 83–94 ms for those runs. Arm 4 reduced that checkpoint to about 72 ms while reducing routed CPU misses and PCIe entries. Whole-window timing is the stronger metric here; attribution among Arm 4's four simultaneously added treatments is not possible from this progressive schedule.
- The adaptive-depth combination reports `max_S=9` and still has one observed >8-row checkpoint; wide verify remains engaged, but adaptive scheduling substantially reduced the maximum observed batch window.
- Raw token sequences differ between at least some independent runs and treatments. The batch test has no cross-arm parity assertions, and the workload is model-generated prose/code. These throughput numbers are not an output-equivalence claim; the raw token dumps are retained for later analysis.
- The DFlash2 banner confirms the intended selq4 pack loaded. The emitted `acc/win` counter is positive, but it is accepted extra rows per window; no offered count is printed, so a DFlash2 acceptance percentage cannot be reported.
- A first arm1234 run exceeded the outer orchestration timeout after successfully finishing trial 1; it was not counted as a completed trial. Its partial log was preserved as `arm1234/r2.aborted-timeout/`. Trials 2 and 3 were then run normally and verified.

## Correctness/build gate

- Candidate build cache: `STRATA_WINDOW_MAX_T=16`, `CMAKE_HIP_ARCHITECTURES=gfx1201`, `STRATA_ENABLE_HIP=ON`, `STRATA_BUILD_TESTS=ON`.
- Branch-focused wide CTest: **26/26 passed**; direct `native_wmma_gfx12_test`: passed. Full details: `bench/vwa/official/final_ctest.log` in the candidate worktree.
- Candidate binary for the original progressive campaign SHA-256: `7b2bcd2f266ee8bc2cd3ca03ebb9c2b8f58e04681dbc28583ead4e8edc560c13`; its code objects contain `gfx1201` only. The final post-default-change promotion binary has a different SHA recorded below.
- Build/CTest worktree contains two minimal fixes needed for this toolchain/test data: `src/kernels/pcie_overlap_graph_test.cpp` casts HIP host-device-pointer outputs to `(void **)`; `src/kernels/native_wmma_gfx12_test.cpp` uses `n_in=1024` for the Q8 segmented case so the reference has the required number of blocks. These fixes were included in the fork branch before final promotion (see commits `c79e3543e` and later merge refs), not left unpushed.

## Reproduction and artifacts

Run from this machine's matching worktree after obtaining the same model/data and an authorized GPU window. The production service is stopped only for the campaign. The in-worktree helper captures the real engine process's argv/env and gates every launch on the GPU and available RAM:

```bash
cd /home/soren/scratch/Strata-verify-wide-all
systemd-run --user --scope -p MemoryMax=infinity -p MemorySwapMax=infinity -p TasksMax=infinity -- \
  /home/soren/Strata/.venv/bin/python bench/vwa/run_wide_arm.py baseline 3 1
# Repeat for the progressive treatments:
systemd-run --user --scope -p MemoryMax=infinity -p MemorySwapMax=infinity -p TasksMax=infinity -- \
  /home/soren/Strata/.venv/bin/python bench/vwa/run_wide_arm.py arm1 3 1
# arm12, arm123, arm1234 likewise
```

Every run uses `--batch 4 --n 4 --max-new 128 --skip-solo --extra '--draft-cap 16'`, wide env `STRATA_BATCH_DF2_MAX_T=16`, telemetry `STRATA_BATCH_DF2_TIMING=1`, and separate output/cache files. Each arm JSON line in the preserved raw file includes switches, result, live-env/argv check booleans, binary hash, and per-window telemetry. Raw results (18 progressive trials: baseline x6, four treatments x3) are the first 18 rows of `docs/ab/perf-verify-wide-results-2026-11-19.jsonl`; three supplemental all-on rows were appended later (see follow-up). Candidate logs/token dumps remain in `~/scratch/Strata-verify-wide-all/bench/vwa/official/r9700_ab_logs/`.

## Final machine state

Production is now on the promoted v0.2.10 component binary (`components/engine/fork-58aefafc9/strata`, SHA-256 `4a69eecc771adfd2d672a625e6ef8d026c997d27745fb431ee699ab87632d0a6`, project version `0.1.61`), with the user's explicit approval after an earlier unapproved confirmation timed out. `strata.service` is **active** with `0` restarts, and the production drop-in pins all 25 runtime/performance variables while keeping telemetry flags off. The rollback bundle (including the prior `fork-0161-fc2de6a19` binary, SHA-256 `805c9c66c62bbb7184280b3526ba063a255f388f1f029efc3914ca6bbe2a0e81`, profile, and drop-in) is preserved at `/home/soren/strata-promote-backup-20261119-58aefafc9/`.

## Follow-up: 2026-11-19 candidate confirmation and promotion
- **Promoted candidate:** `fork-main` / `perf/verify-wide-all` commit `029f1452c1bc28b5f6742af7a27062875051e90d`, installed to the `v0.2.10` component/profile install generation at `/home/soren/.local/share/strata-upstream-v0.2.10/components/engine/fork-58aefafc9/strata` (verified engine binary reports project version `0.1.61`, SHA-256 `4a69eecc771adfd2d672a625e6ef8d026c997d27745fb431ee699ab87632d0a6`, sole `gfx1201` code object, `STRATA_WINDOW_MAX_T=16`).
- **Build & CTest gate:** Wide CTest suite passed **26/26** (plus direct `native_wmma_gfx12_test` pass; log `bench/vwa/official_ctest.log` in `/home/soren/scratch/Strata-verify-wide-all`). The profile (`strata-iq3_s.json`) changed `exe` only.
- **Variable inventory & production drop-in:** `perf-verify-wide.md` lists 27 table entries: 25 performance/runtime variables and 2 telemetry-only variables (`STRATA_BATCH_DF2_TIMING`, `STRATA_VERIFY_PROFILE`). Although commit `33f81bfd4` is titled “default-on the 24 ... switch set,” an audit of its source diff against the parent shows fallback/default changes for all 25 runtime variables: this includes `STRATA_BATCH_DF2_TARGET_ROWS` moving from the implicit narrow value 8 to 12 (matching the guide) and `STRATA_MOE_GRID_CAP` moving from unset 0 (uncapped) to 64. The production drop-in (`fork-switches.conf`) explicitly pins all 25 performance/runtime values—including `STRATA_BATCH_DF2_MAX_T=16`, `STRATA_BATCH_DF2_TARGET_ROWS=12`, and `STRATA_MOE_GRID_CAP=64`—while the two telemetry variables are absent/unset in production.
- **Supplemental `all_on_24` A/B qualification:** Before cutover, three supplemental batch-4 trials (`arm == "all_on_24"`, four 128-token output slots, 512 batch tokens) on candidate binary `4a69eecc771adfd2d672a625e6ef8d026c997d27745fb431ee699ab87632d0a6` measured **96.9, 100.7, 99.0 tok/s** (median **99.0 tok/s**; `max_S` 14/12/16, `wide_windows_gt8` 1/1/1, `mean_verify_ms` 82.52/80.59/84.75 ms). Each raw row records all 27 experiment keys in `set_switches` (all 25 runtime keys plus `STRATA_BATCH_DF2_TIMING=1` and `STRATA_VERIFY_PROFILE=1` for A/B measurement only), `live_env_checked=true`, `live_argv_checked=true`, and `runtime_dflash2_banner=true`. Compared with earlier progressive `arm1234` (104.1, 104.8, 102.4 tok/s; median **104.1 tok/s**), `all_on_24` is not the same treatment: `set_switches` has 27 keys vs `arm1234`'s 25, differing in `STRATA_CPU_EXPERT_NO_SPLIT` (`"0"` → `"1"`), `STRATA_BATCH_PCIE_AUTO` (unset → `"1"`), and `STRATA_BATCH_PCIE_BALANCE` (unset → `"1"`), and both runs also used different `profile_sha256` (`cf01d86e5a21e61ed27b9511e2034d7cc83c27da897b9eb22bec154a201c7a0b` vs `24a4b0c76e34f2315b0fa9a237d132588a55732d17ebe9778df3e52e73078105`) and `binary_sha256` (`7b2bcd2f266ee8bc2cd3ca03ebb9c2b8f58e04681dbc28583ead4e8edc560c13` vs `4a69eecc771adfd2d672a625e6ef8d026c997d27745fb431ee699ab87632d0a6`). The throughput gap between `arm1234` and `all_on_24` is unresolved and not attributable to any single switch or change.
- **Post-cutover production verification:** Recorded in `/home/soren/strata-promote-backup-20261119-58aefafc9/promotion-result.json`, `candidate-health-smoke.json`, and `candidate-substantive-smoke.json` (rollback bundle at `/home/soren/strata-promote-backup-20261119-58aefafc9/`):
  - Auth & health: keyed `/v1/models` returned `200`, unkeyed returned `401`; arithmetic smoke returned `42` with `finish_reason=stop`.
  - Substantive smoke: 66 output tokens (`finish_reason=stop`, 33 prompt tokens, 396 chars) with feature telemetry `drafts accepted 38 of 61` (`67.5 tok/s`, `1 checkpoints`).
  - Live process & service: `strata.service` `active` with `0` restarts; live engine PID `283668` verified with all 25 performance pins (`switch_count: 25`, `wrong_switch_values: {}`) and DFlash2 startup banner confirming `qwen3.8-flash-next-dflash2.q4_0-selq4.pack` loaded with windows up to 16 tokens (`--draft-fallback none`).
- **Raw JSONL verification:** `docs/ab/perf-verify-wide-results-2026-11-19.jsonl` contains 21 rows total (the 18 original rows plus the 3 verbatim `all_on_24` rows from `/home/soren/scratch/Strata-verify-wide-all/bench/vwa/official/r9700_ab_logs/results.jsonl`), SHA-256 `b19f705732d28215c0f088e85b725bf944ae2ee5599020b4753d99d5f6ccd2f2`.
