# Job D2 — IL multi-token GEMV for IQ3_S (21) and IQ3_XXS (18), HIP/gfx1201

Verdict: parity PASS (bitwise, every case). Both timing runs done; the emitted IQ3 cells are reported here and NOT
installed in the kernel tables. Two caveats matter for review: the CUDA preprocessed output is not byte-identical
(3 existing lines changed, see below), and the IL path takes IQ3 only when n_in <= 4096.

Card: AMD Radeon AI PRO R9700 (gfx1201, wave32, 32 CUs). Repo: /home/soren/Strata-integ, branch integ/community-2026-10.
Parity and bench binary: build-hip/mmvq_il_parity, sha256 0cab226a7f7f28edf93350f8185f77570df73cd94ebd235ac377b178719f530c
(the same hash before and after the final rebuild, so the bench ran the final source).

## Edits made (file:line, new-file line numbers)

| File | Lines | Change |
|------|-------|--------|
| src/kernels/cuda/native_mmvq.cu | 1965-2079 | new: `#include "iq3_grid.inc.h"`, Iq3XxsBlock (98 B), Iq3SBlock (110 B) with static_assert, `iq3_int_b2` (two u16 loads; the 98/110-byte blocks are 2-byte aligned), Iq3IlDot, unpack_ksigns, Iq3XxsIlTraits (TYPE 18), Iq3SIlTraits (TYPE 21), iq3_cols, IlIQ3XXS, IlIQ3S (PM=false, HAS_PF=false) |
| src/kernels/cuda/native_mmvq.cu | 2307-2310 | new: `kIlSingleStream<I>` (true for IlIQ3XXS and IlIQ3S, false otherwise) |
| src/kernels/cuda/native_mmvq.cu | 2345 | changed: accumulate `acc[kIlSingleStream<I> ? 0 : v][i][c] += o[c]` (non-PF kernel) |
| src/kernels/cuda/native_mmvq.cu | 4099-4100 | changed: `native_mmvq_il_supported` type list adds `ggml_type == 21 \|\| ggml_type == 18` |
| src/kernels/cuda/native_mmvq.cu | 4103-4106 | new: `kIq3IlMaxNIn = 4096` with a comment |
| src/kernels/cuda/native_mmvq.cu | 4109-4110 | changed: `native_mmvq_il` falls back to native_mmvq for IQ3 when n_in > 4096 |
| src/kernels/cuda/native_mmvq.cu | 4128-4129 | new: `case 21: launch_il_rows<IlIQ3S>`, `case 18: launch_il_rows<IlIQ3XXS>` |
| src/kernels/cuda/iq3_grid.inc.h | new file | iq3xxs_grid[256] and iq3s_grid[512], copied token for token from third_party/ggml/ggml-common.h lines 1017-1117 |
| src/kernels/mmvq_il_parity.cpp | 142-160 | new: IQ3 CASES (type 21 at 110 B, type 18 at 98 B, block 256, scale_at {0,-1}) |
| src/kernels/mmvq_il_parity.cpp | 311-312, 340, 354, 369, 383, 398 | changed: emit_table covers 7 types (`ti < 7`) |

Kernel tables (kIlRows_1202, kIlRows58_1202, kIl1Rows_1202): not changed. They have no 18/21 cells, so IQ3 is
unreachable by default. The patch is in iq3-edits.patch (369 lines, includes the untracked iq3_grid.inc.h). Nothing staged or committed.

## CUDA-unchanged evidence

Method: `clang++ -E -x c++ -I /opt/cuda/targets/x86_64-linux/include -I include -I src` on the HEAD file
(iq3-cuda-pre/native_mmvq.before.i) and on the final file (iq3-cuda-pre/native_mmvq.after.i, regenerated after the
last edit, so `__HIPCC__` is undefined). Lines starting with `#` and all whitespace removed, then a multiset compare
(iq3-cuda-pre/cmp-summary.txt).

- Before: 38976 non-# lines. After: 39188 lines.
- Removed: 3 lines. These are the three changed lines in the table above (2345, 4099, 4109 in the old form).
- Added: 215 lines. This is the new IQ3 code and the grid table tokens.

The output is NOT byte-identical, so this is not a clean "unchanged" result. The argument that existing types behave the
same is a reading of the code, not a measurement:

- The accumulate line is `acc[kIlSingleStream<I> ? 0 : v]`. For every type except IQ3 the condition is false and the index is `v`.
- The `|| (iq3 && n_in > kIq3IlMaxNIn)` term is false for every non-IQ3 type.
- The type list gains 18/21. Those types only reach the IL path if `il_rows()` returns a non-zero rows count. The
  shipped tables have no 18/21 cells, so for existing cards it returns 0.
- Not checked: no CUDA build was run, and no disassembly compares the generated code.
- Exception: the `g_tune_rows` override (used by the parity tool) bypasses `il_rows()` for every type, so with that
  override set, 18/21 also reach the IL path on any card.

## Build log summary

- `cmake --build build-hip --target mmvq_il_parity -j 20`, in the systemd-run scope: exit 0, zero errors.
  Full log: iq3-build-final.log.
- The first build had 6 errors (`ldg_nc_b2` is CUDA-only; HIP could not see it). Fixed by the unguarded `iq3_int_b2`.
- Warnings: 1230 lines, all from native_mmvq.cu, which is 410 per target over three targets (gfx1200, gfx1201, host).
  Across the three targets, 1113 are sign-compare and 102 are unused-value (the hipError_t returns); all are on
  lines this job did not change.
- New warnings from this job: 2, both `-Wunused-const-variable` for `HAS_PF` at native_mmvq.cu:2043 (IlIQ3XXS) and
  :2063 (IlIQ3S). The same warning already fires at :1929, in an existing struct. The members stay because
  `native_mmvq_il_kernel` reads `I::HAS_PF` (`if constexpr`) at :2386. Not silenced.
- No other warning falls on a changed line.

## Parity verdict

Command, in the scope: `./build-hip/mmvq_il_parity` (no flags). Exit 0. Final line:

```
mmvq_il_parity: OK (every case, T 1-8, rows 1/2/4 and the table, bitwise)
```

FAIL lines: 0 (iq3-parity-final.log). Both bench runs also end with that same line and have 0 FAIL lines.

Coverage for types 21 and 18: n_out 512, 640, 2051 (odd), 6144, 10240, 12288 at n_in 2560; 4096x640; 256x512; 512x512.
T 1-8, rows 1/2/4, compared against native_mmvq in the same process. The comparison was not loosened.

History: the first parity run had 337 FAIL lines, all IQ3. The first case was IQ3_S 2560x512 T=1 rows=1 (236 of 512 bits
differ). Shapes with 256/512 passed, so the failure starts at 3 or more chunks per lane. The fix was a single accumulator
for IQ3 (`kIlSingleStream`), matching the baseline's one chained sum. Three of the four allowed rebuild cycles were used.

Cause: the fp-contraction explanation (the baseline chains its sum where the four-stream form does not, and fma
rounds differently) is a hypothesis. I did not confirm it by disassembly; the attempt to unbundle the offload object
failed.

## Per-type IQ3 bench summary (best gain per T vs the native_mmvq baseline)

Baseline: the "multi" column, which is native_mmvq for the type. For 18 and 21 that is the iq_mmvq path
(native_mmvq.cu:3637 and 3686). Gain = 1 - (fastest of rows 1/2/4 time) / (multi time), per shape. The table shows the
best shape for each T, with the rows count that shape used. Two runs; both shown as run1 / run2.

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

Reading the table: the best single shape is not what gets installed. A cell is kept only if every shape in its class
beats the baseline by at least 3%, so the emitted values below are much more conservative. Shapes that did not win
at T=6 (IQ3_S, IQ3_XXS) or at T=1 (IQ3_S) get no cell. Wins at the 3%-every-shape rule, per run: IQ3_S T=3 and T=4 were
6 of 8 shapes in both runs, T=7 and T=8 were 8 of 8; T=6 was 0 of 8 for both types.

## Emitted iq3 tables and per-cell agreement

Layout (same as kIlRows): the three-column table is T=2, 3, 4 with 5 n_out classes each (classes: n_out <2048,
<4096, <8192, <12288, else). The 5-8 column table is T=5, 6, 7, 8 (the source comment says rows 4 runs as 2 there). The one-column
table is T=1.

Agreement rule: keep the value where run1 and run2 print the same; otherwise 0. Checked by script (iq3-tables.py output).

Three-column table (T=2, 3, 4):

```
IQ3_S   run1  T2 [0,0,0,1,1]  T3 [0,0,1,1,1]  T4 [0,0,1,1,1]
        run2  identical
        kept  identical
IQ3_XXS run1  T2 [0,0,0,0,0]  T3 [0,0,1,0,0]  T4 [0,0,1,1,1]
        run2  T2 [0,0,0,0,0]  T3 [0,0,0,1,0]  T4 [0,0,1,1,1]
        kept  T2 [0,0,0,0,0]  T3 [0,0,0,0,0]  T4 [0,0,1,1,1]
```

Five-to-eight column table (T=5, 6, 7, 8):

```
IQ3_S   run1  T5 [0,0,2,2,2]  T6 [0,0,0,0,0]  T7 [1,1,4,4,2]  T8 [1,1,2,2,2]
        run2  T5 [0,0,2,2,2]  T6 [0,0,0,0,0]  T7 [1,1,4,2,2]  T8 [1,1,2,2,2]
        kept  T5 [0,0,2,2,2]  T6 [0,0,0,0,0]  T7 [1,1,4,0,2]  T8 [1,1,2,2,2]
IQ3_XXS run1  T5 [0,0,2,2,2]  T6 [0,0,0,0,0]  T7 [1,1,4,2,2]  T8 [1,1,4,4,4]
        run2  T5 [0,0,2,2,2]  T6 [0,0,0,0,0]  T7 [1,1,4,4,2]  T8 [1,1,2,4,2]
        kept  T5 [0,0,2,2,2]  T6 [0,0,0,0,0]  T7 [1,1,4,0,2]  T8 [1,1,0,4,0]
```

One-column table (T=1): all zero in both runs for both types, so kept is zero.

Per-cell agreement: 40 cells per type (15 in the three-column table, 20 in the five-to-eight table, 5 in the one-column table).
- IQ3_S: 1 cell differs, five-to-eight table T=7 class 3 (run1 4, run2 2), kept as 0.
- IQ3_XXS: 5 cells differ. Three-column table T=3 classes 2 (1 vs 0) and 3 (0 vs 1), kept as 0. Five-to-eight table
  T=7 class 3 (2 vs 4), T=8 class 2 (4 vs 2) and class 4 (4 vs 2), kept as 0.
- The one-column table: no cell differs (all zero).

Not installed. The parent reviews and installs.

## Caveats

- Bench arm: cold only (192 MB rotation). Two same-day runs on one card. No other arm was measured.
- The first bench attempt did not run: a shell quoting bug (a systemd-run prefix held in a variable, read by zsh as one word) made both logs
  error out. Both logs in this folder come from the second attempt, which ran in full. The service was stopped and started around it.
- The service is active at the end (`systemctl --user is-active strata.service` returned `active`).
- Parity covers only n_in in {256, 512, 2560, 4096}. For n_in > 4096 the IL call falls back to native_mmvq (kIq3IlMaxNIn), so no IQ3 speed-up is possible there.
- The bitwise match was checked for the shapes above. Other shapes are not covered by the gate.
- The CUDA check is a token argument, not a build or a measurement (see above). The `g_tune_rows` override now also reaches
  IQ3 types.
- The root cause of the first parity failure (contraction-order) is a hypothesis, not confirmed.
- Two new `-Wunused-const-variable` warnings (`HAS_PF` in IlIQ3XXS and IlIQ3S), left in place.
- The emitted tables are not in the source, per the brief.
- No commit, push, or change to ~/Strata or to the unit files.
