# IQ3 interleaved-mmvq and IL on/off on the Radeon AI PRO R9700 (2026-10-28)

## Summary

- **Kernel level:** the IQ3_S and IQ3_XXS interleaved-mmvq paths gain 3–26% on the best single shape per column count (T=3 to 8). The installed cells are the conservative subset; T=6 is zero.
- **End to end, single-slot MTP decode:** IL on 171.2 tok/s median, IL off 160.95 (+6.4%). Round by round: +10.1% and +4.7%.
- **Batch-mode MTP (the production layout): not measured.** Both batch arms of this session failed to run. Details in section 3.
- **Production is not switched to this build.** The candidate cannot register its 46.8 GiB expert arena with the GPU driver on this machine, and production's memory layout depends on that registration. The end-to-end runs here used `--mmap-experts`, which avoids it. Section 4.

## Environment

- GPU: AMD Radeon AI PRO R9700 (gfx1201, 32 CUs). ROCm 10.2.0 at `/opt/rocm`.
- Host: Linux 7.3.0-2-cachyos-server. MemTotal 65,426,204 kB; the engine's memory guard limit is 62.4 GiB. Model files on local NVMe.
- Code: `integ/community-2026-10`, tree of commit `d10df443` (pre-rewrite SHA). The tree is identical after the message rewrite.
- Candidate engine: `build-hip/strata`, sha256 `9d8e39ef908b…`, 76,960,528 bytes.
- Parity binary: `build-hip/mmvq_il_parity`, sha256 `72ab31cd…`. Result: `parity-HEAD.log`.

## 1. Kernel-level results

Source: `iq3-report.md` (two runs, cold cache, 192 MB rotation). Gain = 1 − (fastest of rows 1/2/4) / (native_mmvq time), best single shape per T.

| T | IQ3_S best shape (rows) | gain run1 / run2 | IQ3_XXS best shape (rows) | gain run1 / run2 |
|---|---|---|---|---|
| 1 | 256x512 (1) | 0.0% / 0.0% | 2560x512 (1) / 2560x640 (1) | +2.2% / +4.1% |
| 2 | 2560x10240 (1) / 2560x12288 (1) | +5.7% / +5.3% | 2560x12288 (1) / 2560x12288 (1) | -0.3% / +0.8% |
| 3 | 4096x640 (1) / 4096x640 (1) | +9.0% / +7.5% | 2560x6144 (1) / 2560x10240 (1) | +3.6% / +3.1% |
| 4 | 4096x640 (1) / 4096x640 (1) | +12.5% / +11.1% | 2560x6144 (1) / 2560x6144 (1) | +8.4% / +9.1% |
| 5 | 2560x12288 (2) / 2560x12288 (2) | +7.4% / +6.6% | 2560x12288 (2) / 2560x10240 (2) | +12.2% / +10.9% |
| 6 | 2560x12288 (2) / 2560x12288 (2) | -1.4% / -1.4% | 2560x12288 (2) / 2560x12288 (2) | +1.8% / +1.6% |
| 7 | 2560x512 (1) / 2560x512 (1) | +25.0% / +23.8% | 2560x512 (1) / 2560x512 (1) | +25.6% / +25.6% |
| 8 | 2560x12288 (2) / 2560x10240 (2) | +23.1% / +23.1% | 2560x512 (1) / 2560x512 (1) | +22.8% / +22.8% |

Cell rules: a cell is kept only where both runs agree (six disagreeing cells zeroed: five IQ3_XXS, one IQ3_S), and a class keeps a value only if every shape in it beats the baseline by at least 3%. T=6 has no kept cell.

**Four-stream accumulator (rejected).** Interleaved microbenchmark, n_in 2560, table-selected rows: 6.9–18.0% slower than single-stream on IQ3_S at T=5, 7 and 8. Its first parity run also failed (337 FAIL lines). Report: `iq3-accumulator-ab-report.md`. The test-only switch is removed from the tree.

## 2. End to end: single-slot MTP decode, IL on vs off

Harness: `ab_decode_ab_v5.py`. Candidate build with the production arguments minus the three fork-only flags (`--conversation-cache-disk-mib`, `--conversation-cache-dir`, `--kv-pool-tokens`), plus `--mmap-experts`. `parallel` 1, drafter MTP. Per round: 3 warm-up requests (128 tokens), then 10 measured greedy requests (512 tokens). Metric: the engine-reported `predicted_per_second`. Order: S0 S1 | S1 S0. Production was stopped for the window and restarted afterward.

| arm | round 1 median | round 2 median | pooled median (n = 20) | range |
|---|---|---|---|---|
| IL off (S0) | 155.65 | 163.5 | 160.95 | 121.7–164.3 |
| IL on (S1) | 171.4 | 171.2 | 171.2 | 169.8–173.0 |

- Median difference +10.25 tok/s (+6.37%). Bootstrap 95% interval of the median difference (4000 resamples): +8.2 to +14.7 tok/s. Welch t = 6.33.
- Round by round: +10.12% and +4.71%. The IL-off arm drifted between rounds (155.65 to 163.5), and its round-1 requests included one 121.7 tok/s outlier. The IL-on arm was stable.
- Consistency: the earlier v4 MTP arm with IL on, same single-slot configuration, measured 171.5 and 173.2 tok/s.
- Caveats: `--mmap-experts` disables the adaptive tier, so these are not production numbers. The prompts are counting lists, where draft acceptance is about 99%; chat text will accept less.

## 3. Batch-mode MTP (production layout): not measured

Arms K0 (IL off) and K1 (IL on): `--batch 4 --batch-mtp`, same `--mmap-experts` mode, same request protocol.

- **K0 round 1:** the engine reported ready after 865 s. The warm-up request then returned HTTP 503 after about 120 s at 0.2 tok/s. The engine was in disk sleep, paging expert files. Its prefill plan wants 31.4 GiB of experts in page cache with 38.7 GiB available to the cache.
- **K1 rounds 1 and 2:** the engine exited on start with `native dense upload blk.7.attn_q.weight: out of memory`. The previous engine process still held VRAM when the next arm started.
- **K0 round 2:** stopped by hand after about 3 minutes of startup, once the earlier batch arms had failed.
- **Harness flaw:** `stop_arm()` waits for the engine process to disappear, but a process in disk sleep kept its VRAM past the timeout, and the runner moved on. The fix is to wait for GPU memory to be released before the next arm, and to warm the page cache until two consecutive requests agree.

Earlier batch runs in the same mmap mode (v4, five requests per round). Exploratory, not used for any decision:

| arm | round medians | per-request outliers (tok/s) |
|---|---|---|
| IL off (B) | 138.4, 154.9 | 37.9, 87.8 |
| IL on (C) | 159.7, 149.2 | 99.6 |

The round order flipped between rounds, so these are inconclusive.

## 4. Blockers and unverified items

- **Candidate expert-arena registration (blocking production).** `cudaHostRegister` of the whole 46.8 GiB expert arena returns `invalid argument`. The engine then pins 41 GiB in slices and falls back to a buffered path for the rest. The production binary registers the same arena in one call (`cudaHostRegister PORTABLE ok`). The candidate also rejects `--kv-pool-tokens`, so its memory layout differs beyond the binary. The root cause is not yet found.
- **Batch-mode IL effect, end to end:** not measured (section 3).
- **CUDA build:** not run on this machine.
- **gfx1200 and other 12.0 AMD parts:** not measured; they share key 1202.
- **Prompt mix:** counting lists. Acceptance here is not representative of chat.

## 5. Files

- `iq3-report.md`: per-shape kernel tables, cell agreement, parity, CUDA preprocessing check.
- `iq3-accumulator-ab-report.md`: four-stream rejection.
- `jobC-report.md`: commit `4c9980cb` tables and HIP enablement.
- `run_bench.sh`: the wrapper used for the kernel bench runs.
- `iq3-bench-run1.log`, `iq3-bench-run2.log`: raw kernel bench logs (240 BENCH lines each).
- `ab4s-*.log`: raw four-stream microbenchmark logs.
- `iq3-parity-final.log`: parity from the earlier binary (before the last rebuild).
- `parity-HEAD.log`, `verify_parity.sh`: parity at this commit.
- `ab_decode_ab_v5.py`: end-to-end harness. Paths are local; edit `D`, `PROD_CFG`, `CAND` and `PY`.
- `e2e-v5-summary.json`, `e2e-v5-arms.jsonl`: end-to-end raw data (per-request timings, engine argv and IL environment).
- `e2e-v5-serve-*.out`: serve logs for the S arms (`done:` lines per request) and the failed K arms.
- `e2e-v4-summary.json`, `e2e-v4-arms.jsonl`: earlier exploratory run (batch and single, mmap mode).

Engine logs are not included (large, per-window timing). Keys and service configs are not included.

## 6. Reproduce

Kernel parity (expect `mmvq_il_parity: OK ...` and 0 FAIL lines):

    systemd-run --user --collect --scope --property=MemoryMax=8G \
      --working-directory=$HOME/Strata-integ -- ./build-hip/mmvq_il_parity

Kernel bench (cold cache): `run_bench.sh` runs `mmvq_il_parity --bench --emit-table` in an 8 GiB scope.

End to end: `python3 ab_decode_ab_v5.py`. The harness stops the production service, runs the arms, writes `arms-v5.jsonl` and `summary-v5.json` into its working directory, and restarts production when every benchmark process has exited.
