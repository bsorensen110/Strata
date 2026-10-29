# IQ3 IL single- vs four-stream accumulator A/B

Date: 2026-10-28. Card: AMD Radeon AI PRO R9700 / gfx1201, ROCm 10.2 nightly. Tree `/home/soren/Strata-integ`, branch `integ/community-2026-10`, IQ3 IL commit `d10df443` (the only baseline kernel code used in the test).

## Question and method

Compared the IQ3 single-stream specialization (`kIlSingleStream<IlIQ3S/XXS> = true`) with an experimental four-stream instantiation (`false`), selected only for that process by `STRATA_MMVQ_IL_IQ3_4S=1`. The experimental instantiation was local test-only code, not committed. The production service was stopped for the GPU window and restored by an EXIT trap; the A/B alternated single, four-stream, single, four-stream. Each arm ran `build-hip/mmvq_il_parity --bench` on the same build and all 128 matched IQ3 type/shape/T cases. The harness timed native_mmvq and rows 1/2/4; deltas below compare the *same row count* between variants. Numbers are medians of the two runs. Run logs: `ab4s-single-{1,2}.log`, `ab4s-4s-{1,2}.log` in this directory.

## Results

Median four-stream speed change over single-stream across shapes, same row count (negative means slower):

| Type | rows | T5 | T6 | T7 | T8 |
|---|---:|---:|---:|---:|---:|
| IQ3_S | 1 | +0.7% | +3.1% | -2.6% | +3.3% |
| IQ3_S | 2 | -15.5% | -0.4% | -8.8% | -9.5% |
| IQ3_S | 4 | -15.8% | -0.3% | -8.1% | -9.3% |
| IQ3_XXS | 1 | -13.2% | +4.6% | -1.7% | -3.8% |
| IQ3_XXS | 2 | -18.8% | +10.0% | -9.1% | -8.5% |
| IQ3_XXS | 4 | -18.0% | +9.7% | -8.8% | -5.3% |

For the production IQ3_S table-selected rows, n_in=2560:

| n_out | T | rows | single (run1, run2) us | four-stream us | delta |
|---:|---:|---:|---:|---:|---:|
| 6144 | 5 | 2 | 27.0, 27.5 | 31.7, 32.3 | -17.4% |
| 6144 | 7 | 4 | 32.3, 32.6 | 34.7, 34.7 | -6.9% |
| 6144 | 8 | 2 | 32.7, 33.0 | 37.1, 37.1 | -12.9% |
| 10240 | 5 | 2 | 42.5, 42.1 | 48.4, 48.6 | -14.7% |
| 10240 | 7 | 0 (table fallback) | — | — | — |
| 10240 | 8 | 2 | 49.5, 49.9 | 57.7, 58.5 | -16.9% |
| 12288 | 5 | 2 | 49.2, 49.4 | 56.2, 56.5 | -14.3% |
| 12288 | 7 | 2 | 56.2, 56.8 | 63.8, 63.0 | -12.2% |
| 12288 | 8 | 2 | 57.6, 58.2 | 68.1, 68.5 | -18.0% |

Across T1–8, best-of-rows deltas (selection independently per variant) were median -1.27%, mean -3.41% over 128 cases. At T6 four-stream sometimes helped vs single, but the shipped table disables IL at T6 because neither is consistently faster than native_mmvq.

## Correctness and decision

- Default (single-stream) `mmvq_il_parity`: PASS, bitwise, every case and table.
- Four-stream opt-in: 377 IQ3 parity comparison failures per run (both runs); all are output-bit differences. One representative: IQ3_XXS 4096x640, T8, rows4: 2,847/5,120 bits differ. This is not a numerically qualified fast path.
- Decision: **do not keep or ship `STRATA_MMVQ_IL_IQ3_4S`**. On the table-selected IQ3_S T5/T7/T8 production shapes, four-stream was 6.9–18.0% slower, while it breaks bitwise parity broadly. Test-only code was removed; source remains at the committed single-accumulator implementation. Rebuilt `mmvq_il_parity` and reran default: PASS.
- Service restoration: `systemctl --user is-active` = active; authenticated `/v1/models` = HTTP 200. Production binary/config were not changed.

## Caveat

This is a microbenchmark of the interleaved GEMV kernel, not end-to-end generation throughput. It is sufficient to reject four-stream for the table-selected dense/LM-head geometries because it consistently loses at the kernel level; no production flag or deployment was made.
