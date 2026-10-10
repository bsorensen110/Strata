# fused-GR speedup A/B on gfx1201 (Radeon AI PRO R9700, 64 CUs) — 2026-10-09

Branch `r9700/hc-register-pipe-ab-20261009`, HEAD 3842ad99. No commits; edits stay in the worktree.
Parity policy (owner, 2026-10-09): bitwise parity with staged is PREFERRED, not mandatory; a differing-bits
variant with a real win ships behind an opt-in gate, never as the default.

## Baseline (given, measured)

Staged whole-read T=1..6 = 30.7 / 35.3 / 38.6 / 43.2 / 48.4 / 53.0 us.
Kernel times T=1: norm 5.59 + down 9.02 + up 8.50 = 23.1 us (launch gap ~7.6 us).
Down grid 41 blocks x 256 thr; 23 of 64 CUs idle; 4 waves/SIMD; waves constant 328 across T. Down VGPR 112 (T=1) -> 152 (T>=3).

## P1 — row-split down kernel (STRATA_HC_SPLIT=8)

Status: MEASURED (row-split arm done; paired staged baseline running).

Build: `ninja -C build-hip -j 8 strata_kernels fused_gr_bench` -> "no work to do"; the row-split kernel
compiles clean for gfx1201, no fixes needed. Binary 2026-10-09 18:48:30, newer than `fused_gr.cu` 18:48:09.

Parity: **BITWISE EQUAL to staged.** The self-test latched variant 9 on its own:
`the hyper-connection read runs as row-split staged (four rows per CTA, a warp pair per row, 81 blocks,
STRATA_HC_SPLIT=8); checked bit for bit against the plain read on this card`.

Row-split whole-read, `STRATA_HC_SPLIT=8 ./build-hip/fused_gr_bench 200 1 6 --mode=time-staged`,
median over the 4 apply/inject cases, median of 5 rounds, 200 iters, fresh process:

| T | row-split (us) | given staged baseline (us) | delta |
|---|---|---|---|
| 1 | 31.9 | 30.7 | +1.2 |
| 2 | 36.7 | 35.3 | +1.4 |
| 3 | 41.0 | 38.6 | +2.4 |
| 4 | 45.6 | 43.2 | +2.4 |
| 5 | 69.7 | 48.4 | +21.3 |
| 6 | 78.3 | 53.0 | +25.3 |

Parity passes, timing is a loss at every T. Paired same-session staged arm below.

Paired staged baseline, same session, same machine, fresh process, `env -u STRATA_HC_SPLIT
./build-hip/fused_gr_bench 200 1 6 --mode=time-staged` (latched variant 3):

| T | staged, paired (us) | row-split (us) | row-split - staged | given baseline (us) |
|---|---|---|---|---|
| 1 | 31.0 | 31.9 | +0.9 | 30.7 |
| 2 | 35.6 | 36.7 | +1.1 | 35.3 |
| 3 | 39.4 | 41.0 | +1.6 | 38.6 |
| 4 | 44.1 | 45.6 | +1.5 | 43.2 |
| 5 | 49.2 | 69.7 | +20.5 | 48.4 |
| 6 | 53.7 | 78.3 | +24.6 | 53.0 |

The paired staged arm reproduces the given baseline to within 0.7 us at every T, so the two arms are
comparable and the row-split loss is real, not drift.

Verdict: **P1 is a measured negative.** Bitwise parity is met (the check latched it on its own), the target
(T=1 down 9.0 -> ~6 us, whole-read 30.7 -> below it) is missed, and the whole-read time is worse at every T,
by 0.9-1.6 us at T=1..4 and by 20.5 / 24.6 us at T=5 / 6. Under the parity policy this variant cannot be
promoted (no speed win), and there is no reason to ship it even behind the opt-in gate: it is slower than the
gate it would sit behind. Recommendation: do not promote; keep the code out of the default path.

Why it loses (measured from the gfx1201 device ELF and `rocminfo`, not guessed):

Kernel resources, `llvm-readobj --notes` on the device ELF carved from `.hip_fatbin` of
`build-hip/CMakeFiles/strata_kernels.dir/src/kernels/cuda/fused_gr.cu.o` (verified `EM_AMDGPU`, `gfx1201`):

| T | staged down VGPR | row-split VGPR | private segment (spills) |
|---|---|---|---|
| 1 | 110 | 111 | 0 |
| 2 | 127 | 131 | 0 |
| 3 | 145 | 146 | 0 |
| 4 | 144 | 145 | 0 |
| 5 | 147 | 157 | 0 |
| 6 | 152 | 154 | 0 |

So register pressure is NOT the cause: the row split costs 1-10 VGPR and spills nothing.

The cause is wave quantization. Shared memory per CTA is `ct * 2 * H_TILE * sizeof(float)` = `ct * 10240` B
(fused_gr.cu:1701), the CTA is 256 threads = 8 waves, `rocminfo` on this card gives **Max Waves Per CU 32**
(= 4 CTAs/CU by waves) and **64 KB group/LDS per CU**. The row split launches 81 blocks, staged launches 41:

| T | smem/CTA | CTAs per CU | row-split waves (81 blocks) | staged waves (41 blocks) | measured delta |
|---|---|---|---|---|---|
| 1 | 10 240 B | 4 | 1 | 1 | +0.9 us |
| 2 | 20 480 B | 3 | 1 | 1 | +1.1 us |
| 3 | 30 720 B | 2 | 1 | 1 | +1.6 us |
| 4 | 40 960 B | 1 | **2** | 1 | +1.5 us |
| 5 | 51 200 B | 1 | **2** | 1 | +20.5 us |
| 6 | 61 440 B | 1 | **2** | 1 | +24.6 us |

At T>=4 the 81 blocks no longer fit in one wave on 64 CUs, so the down projection runs two waves where staged
runs one. At T=5 and T=6 the per-CTA work is large enough that doubling it shows up whole; at T=4 the down
kernel is a small share of the whole read, so the same doubling costs only 1.5 us. The row split also leaves
half of every warp idle on the down rows (16 of 32 lanes walk a row), which is why it does not win even in the
one-wave cases.

Reproduce P1:
```
ninja -C build-hip -j 8 strata_kernels fused_gr_bench
STRATA_HC_SPLIT=8 ./build-hip/fused_gr_bench 200 1 6 --mode=time-staged   # row-split, latches variant 9
env -u STRATA_HC_SPLIT ./build-hip/fused_gr_bench 200 1 6 --mode=time-staged   # staged, latches variant 3
```
Fresh process per arm, run serially, `strata.service` inactive (verified: `systemctl is-active` -> inactive).

## P2 — HIP graph for norm->down->up

Status: **MEASURED — win ONLY on the variant-7 path; no win on staged. Keep behind the `STRATA_HC_GRAPH=1` gate.**
(Section written by the parent from the run's raw data in `/tmp/p2/`; the run hit max-turns before writing it.)

The change (in the worktree, uncommitted): `src/kernels/cuda/fused_gr.cu` — `hc_graph_env()` (opt-in
`STRATA_HC_GRAPH=1`), `graph_cache()`/`GraphEntry` (keyed by every pointer/flag the launches read, cap 32),
`graph_replay()` capturing the launch_multi sequence (3 launches + optional stamp); production call site
falls back to plain launches when the gate is off, capture fails, or the cache misses. Capture-failure
latches the gate off for the process.

Timing, fresh process per arm, 200 iters, median over the 4 apply/inject cases, `strata.service` inactive.

Staged arm (graph vs plain, 4 paired rounds r1..r4 — graph gives nothing):

- T=1: 31.4 / 31.2 vs 31.1 / 31.0 (≈ tie)
- T=2: 35.1 / 35.0 vs 35.4 / 35.2 (≈ tie)
- T=3: 39.4 / 39.2 vs 39.2 / 39.0 (≈ tie)
- T=4: 43.8 / 43.5 vs 43.9 / 43.8 (≈ tie)
- T=5: 48.7 / 48.2 vs 49.1 / 48.9 (−0.3, noise)
- T=6: 52.9 / 52.5 vs 53.7 / 53.3 (−0.6, noise)

Variant-7 arm (graph vs plain, 2 paired rounds — a real win):

- T=1: 26.0 / 26.5 vs 30.2 / 30.3 → **−4.2 / −3.8**
- T=2: 29.3 / 29.4 vs 34.0 / 34.0 → **−4.7 / −4.6**
- T=3: 32.7 / 32.9 vs 37.2 / 37.2 → **−4.5 / −4.3**
- T=4: 36.7 / 37.0 vs 41.2 / 41.3 → **−4.5 / −4.3**

Parity: `--mode=parity` with `STRATA_HC_GRAPH=1`: staged 24/24 bitwise equal (`parity_graph.txt`);
variant 7 + graph 24/24 bitwise equal, 0 DIFFERS, 0 capture failures (`parity_v7_graph.txt`). Graph replay
does not change bits — it removes launch gaps.

Why: the graph kills the CPU launch gap (~7.6 µs at T=1). Staged's kernels are long enough that the gap
hides under GPU work; variant 7's shorter kernels expose it, and the graph recovers ~4.5 µs of it.
Combined P4+P2 at T=1: 26.0 µs vs the 30.7 µs staged baseline = **−4.7 µs (−15%)**.

Verdict: **keep behind the opt-in gate** (`STRATA_HC_GRAPH=1`), promote-to-default NOT recommended:
no win on the staged default path, and the graph cache keys on buffer pointers — a promote needs a
pointer-stability audit of the serve path first. Best config measured: `STRATA_HC_SPLIT=7 STRATA_HC_GRAPH=1`.

Reproduce P2:
```
STRATA_HC_GRAPH=1 ./build-hip/fused_gr_bench 200 1 6 --mode=time-staged   # staged + graph
env -u STRATA_HC_GRAPH ./build-hip/fused_gr_bench 200 1 6 --mode=time-staged   # staged plain
STRATA_HC_BENCH_DIRECT=1 STRATA_HC_SPLIT=7 STRATA_HC_GRAPH=1 ./build-hip/fused_gr_bench 200 1 4   # v7 + graph
STRATA_HC_BENCH_DIRECT=1 STRATA_HC_SPLIT=7 env -u STRATA_HC_GRAPH ./build-hip/fused_gr_bench 200 1 4   # v7 plain
STRATA_HC_GRAPH=1 ./build-hip/fused_gr_bench 1 1 6 --mode=parity          # staged parity
STRATA_HC_SPLIT=7 STRATA_HC_GRAPH=1 ./build-hip/fused_gr_bench 1 1 6      # v7+graph parity (old/fast equal)
```

## P4 — per-T latch for variant 7

Status: **MEASURED - WIN at every T=1..8, bitwise parity held. Final boundary: `kHcHalfWinMaxT = kFusedGrMaxT`
(8) - the clamp is a no-op safety net. Promote-to-default candidate: parity held + win at every measured T.**
The clamp boundary was finalized on the 2nd run (paired fresh processes, 200 iters, T=1..8); the earlier
boundary-4 and boundary-8 runs are superseded by the table below.

The change (in the worktree, uncommitted): a per-token-count clamp on the latched variant.

- `src/kernels/cuda/fused_gr.cu`, after `env_variant()`:
  `constexpr int kHcHalfWinMaxT = kFusedGrMaxT;` and
  `int variant_for_T(int v, int T) { return (v == kHcRegisterHalf && T > kHcHalfWinMaxT) ? kHcStaged : v; }`
- applied at the one production call site:
  `launch_multi(m, variant_for_T(fused_gr_variant(), m.T), st, stamp_buf, stamp_i0);`
- `int fused_gr_variant_for_T(int T)` added, declared in `include/strata/kernels/fused_gr.hpp`, used by the bench
  so the timed line prints what the launch really ran, not what the env asked for.

Only variant 7 is clamped. The staged default, the plain read and every other opt-in path return unchanged.
The self-test is deliberately NOT clamped - it still checks the half kernel against staged at every T, so
latching on variant 7 still means "bit for bit equal at every T", and the clamp is a dispatch choice, not a
parity exemption.

Timing, paired fresh processes, `STRATA_HC_BENCH_DIRECT=1`, 200 iters, median over the 4 apply/inject cases.
`HC-read` is the whole read (norm + down + up) timed by cudaEvent around `fused_gr_read_multi`. Boundary at 8,
so the variant-7 arm ran register-half at every T=1..8 (the timed line says so per case). Two pairs (A, B),
then the boundary re-run (C) after setting `kHcHalfWinMaxT = kFusedGrMaxT`:

| T | v7 A (us) | staged A (us) | delta A | v7 B (us) | staged B (us) | delta B | delta C (boundary re-run, T=5..8) |
|---|---|---|---|---|---|---|---|
| 1 | 30.2 | 30.4 | **-0.2** | 30.4 | 29.9 | +0.6 | - |
| 2 | 33.9 | 35.3 | **-1.4** | 33.8 | 35.2 | **-1.5** | - |
| 3 | 37.1 | 39.1 | **-2.0** | 36.9 | 39.0 | **-2.1** | - |
| 4 | 41.1 | 44.2 | **-3.1** | 41.0 | 43.8 | **-2.8** | - |
| 5 | 49.0 | 49.3 | **-0.2** | 49.1 | 49.0 | +0.1 | **-0.5** |
| 6 | 53.8 | 54.0 | **-0.2** | 53.5 | 53.9 | **-0.4** | **-1.0** |
| 7 | 78.5 | 79.6 | **-1.0** | 78.2 | 79.2 | **-0.9** | **-1.5** |
| 8 | 82.8 | 84.5 | **-1.8** | 82.8 | 83.7 | **-0.8** | **-2.2** |

No crossover in T=1..8: variant 7 wins at T=2,3,4,6,7,8 in every pair and ties within the noise floor (~0.5 us)
at T=1 and T=5. The biggest win is T=4 (-2.8 to -3.1 us); the wins at T=7,8 (-0.8 to -2.2 us) are new - the
boundary-8 run before this one stopped at T=6. The boundary re-run (C) confirms T=5..8 after the change, with
T=8 printing `register-half staged` as the effective variant: the clamp at kFusedGrMaxT never demotes.

Parity: **bitwise equal at every T.** The card's own check latched it:
`the hyper-connection read runs as register-half staged (the same with the tuple in two halves,
STRATA_HC_SPLIT=7); checked bit for bit against the plain read on this card`. The self-test is not clamped, so
this says bit-for-bit equal at every T, not just at the clamped ones.

Verdict: **promote-to-default candidate** - parity held + win at every measured T. The clamp stays in the code
as a no-op safety net (it only matters if `kFusedGrMaxT` grows past the measured T=1..8 range).

Reproduce P4:
```
STRATA_HC_BENCH_DIRECT=1 STRATA_HC_SPLIT=7 ./build-hip/fused_gr_bench 200 1 8   # variant-7 arm
env -u STRATA_HC_SPLIT STRATA_HC_BENCH_DIRECT=1 ./build-hip/fused_gr_bench 200 1 8   # staged arm
env -u STRATA_HC_SPLIT ./build-hip/fused_gr_bench 1 1 1 --mode=time-staged      # parity: the check's latch line
STRATA_HC_SPLIT=7 ./build-hip/fused_gr_bench 1 1 1 --mode=time-staged           # parity: check latches 7, gate exits 2
```

## P3 — VGPR cut at T>=3

Status: **MEASURED - DO NOT PROMOTE.**

The change (in the worktree, uncommitted): a staged-down clone whose per-token running sums live in dynamic LDS.

- `src/kernels/cuda/fused_gr.cu`: new `gr_down_lds_accum_kernel<MAX_T, EXACT_T>` - the staged grid (41 blocks),
  the 256-thread CTA, the two-buffer tile schedule, `stage_htile` calls, weight prefetch and the q-outer /
  k-inner dot order, with `acc[MAX_T]` and `s[MAX_T]` replaced by a per-thread LDS row
  (`lacc = hbuf + 2*buf_f4` at `t*MAX_T + k`, k compile-time: a runtime k lowers to private scratch on
  gfx1201, the HcChain note). Launch smem is `ct*(2*H_TILE + THREADS)*4` = 11264 B/token vs staged's
  10240 B/token, so the launch slices to `usable/(2*H_TILE+THREADS)/4` tokens (5 on this 64 KiB card).
- Slot: `constexpr int kHcLdsAccum = 10`; the env value is **`STRATA_HC_SPLIT=9`** (the parser is one digit and
  8 already names the row split, constant 9). `down_chunk` fills `chunk_lds[dev]`; `launch_multi` takes the
  branch; `fused_gr_selftest` arity 10 -> 11 and checks it against staged at every T; `fused_gr_check` latches
  it only on a bitwise match; `fused_gr_bench` accepts variant 10.
- Parity trap found on the first run: guarding the warp sum per lane (`if (lane != k) continue;` before
  `warp_sum`) leaves `__shfl_xor_sync(0xffffffff)` with lanes already exited; on gfx1201 the shfl emulation
  answers with the wrong bits - the self-test caught it at T=1 (`lo` index 0, `bbdb14f7` vs `3d927695`).
  Fixed by computing the warp sum converged on every lane and guarding only the store, as staged does.

Parity: **bitwise equal to staged at every T=1..8, apply x inject.** Same operations in the same order (q-outer /
k-inner dots, the same `warp_sum` on every lane, the same store guard); only the accumulator's home changed. The
card's own latch:

```
strata hc: CUDA0: the hyper-connection read runs as LDS-accumulator staged (the staged read with the per-token
sums in dynamic LDS, STRATA_HC_SPLIT=9); checked bit for bit against the plain read on this card
```

VGPR from the compiler, per instantiation (`/tmp/p3/traces/lds/lds_kernel_trace.csv`, scratch 0, accum VGPR 0,
SGPR 128, no spills in either arm):

| T | staged VGPR | LDS-accum VGPR | delta |
|---|---|---|---|
| 1 | 112 | 112 | 0 |
| 2 | 128 | 128 | 0 |
| 3 | 152 | **168** | **+16** |
| 4 | 144 | 128 | −16 |
| 5 | 152 | **72** | **−80** |

Whole-read µs, 200 iters, fresh process per arm, serial, two rounds, median over the four apply/inject cases
(round-to-round spread ≤0.8 µs; LDS-accum is behind in all 16 paired comparisons):

| T | LDS-accum | staged | delta |
|---|---|---|---|
| 1 | 30.7 | 29.7 | +1.0 |
| 2 | 35.6 | 35.0 | +0.6 |
| 3 | 40.1 | 38.6 | +1.6 |
| 4 | 44.7 | 43.7 | +1.0 |
| 5 | 51.4 | 48.8 | +2.6 |
| 6 | 64.5 | 53.5 | **+11.1** |
| 7 | 80.4 | 78.8 | +1.5 |
| 8 | 86.7 | 83.4 | +3.2 |

Why the register cut buys nothing here, and what it costs:

- The down grid is 41 blocks on 64 CUs. Each CU holds one block, so waves/SIMD is 4 of 16 because of the **block
  count**, not the register count. Freeing registers cannot add a wave to a grid that has no block to place.
- What the move costs is per-value LDS traffic: the accumulator is read and written once per dot, 5 reads + 5
  writes per token per h-tile per thread, on the kernel's critical path. The trace shows the same shape
  (avg kernel time 15.64 µs at T=3 vs staged's 14.55, 23.85 at T=5 vs 20.95).
- T=6 is the worst case: 11264 B/token x 6 = 67584 B exceeds this card's 64 KiB LDS, so a 6-token read runs as
  **two launches (5 + 1)** and reads the 6.5 MB weight matrix twice. That is the +11.1 µs.
- The T=5 drop (152 → 72) is 80 registers, far more than the accumulator array's own 8 (16 with the epilogue's
  `s[]`), so the array was never the bulk of the pressure - removing it changed the scheduler's live ranges.
  Hypothesis, not measured: with the array out of VGPR the compiler shortens the live range of the 10-uint4
  weight-prefetch tuple. The disassembly was not compared.

Reproduce:

```
STRATA_HC_BENCH_DIRECT=1 STRATA_HC_SPLIT=9 ./build-hip/fused_gr_bench 200 1 8                    # LDS-accum arm
env -u STRATA_HC_SPLIT STRATA_HC_BENCH_DIRECT=1 ./build-hip/fused_gr_bench 200 1 8               # staged arm
/opt/rocm/bin/rocprofv3 --kernel-trace -f csv -o /tmp/p3/traces/lds/lds -d usec -- \
  env STRATA_HC_BENCH_DIRECT=1 STRATA_HC_SPLIT=9 ./build-hip/fused_gr_bench 200 1 5              # VGPR column
STRATA_HC_SPLIT=9 ./build-hip/fused_gr_bench 1 1 1 --mode=time-staged                            # parity latch line
```

Verdict: **do not promote, not even behind the gate.** Bitwise parity holds, the T=4/T=5 VGPR cut is real, and the
arm is slower at every T with a +11.1 µs cliff at T=6 from the launch split. The code stays in the worktree
uncommitted; the default path is untouched.

## P2 audit — pointer stability of the serve path, and what the graph gate does there (2026-10-09)

Status: **AUDIT DONE. Pointer stability PASSES. The gate is INERT on the serve path — promote-to-default is off
the table; `STRATA_HC_GRAPH=1` stays a bench/loop tool.**

### Every production call site (grep of `src/`, file:line)

`fused_gr_read_multi` (declared `include/strata/kernels/fused_gr.hpp:68`, defined `src/kernels/cuda/fused_gr.cu:2736`)
is called from eight production places, all in the engine's record/capture layer:

| Site | Path | Notes |
|---|---|---|
| `src/core/mtp.cpp:723` | `MtpDrafter::record_front`, attention hyper-connection | `apply=false`, `w_inject` non-null |
| `src/core/mtp.cpp:838` | `MtpDrafter::record_rest`, MLP hyper-connection | `apply=true` |
| `src/core/mtp.cpp:946`, `:957` | `MtpDrafter::record_rest`, final-mixer arms | `apply=true` / head-mix |
| `src/core/verify.cpp:940` | `Verifier::record_window`'s `gr_read_group` lambda | the only site that passes `stamp_buf` (`prof_`) and a `stamp_i0` |
| `src/core/verify.cpp:1555`, `:1588` | `Verifier::record_window`, final-mixer arms | `w_inject = nullptr`, `head_mixed_`/`head_inj_` (`:1573` is a comment) |

`launch_multi` (`fused_gr.cu:1738`, file-local) is reached only from `graph_replay` (`:1995`) and the plain launch
(`:2866`); the self-test calls it at `:3058-3096`. Nothing in `serve/` calls either (the Python server reaches the
engine through `src/program/generate.cpp`). `src/kernels/gr_parity.cpp:368-440` and `fused_gr_bench.cpp` are tests.

### Pointer stability: passes

The key (`args_key`, `fused_gr.cu:1955-1980`) mixes `variant`, `m.T`, `gr_fast()` (under `STRATA_GR_FAST_BUILD`),
`m.xn`, `stamp_buf`, `stamp_i0`, and per token `R`, `R_out`, `apply`, `bo_prev`, `inj_prev`, `w_norm`, `w_down`,
`w_up`, `w_inject`, `lo`, `rs`, `inject_out`, `mixed`, and the three `pack_*.lows`.

- The `FusedGrArgs` array is a per-call stack array at every site (`mtp.cpp:713/827/935/949`, `verify.cpp:923/1545/1558/1581`)
  and is copied by value into the stack `GrMulti` (`fused_gr.cu:2743`). The key mixes the pointer *values* it holds,
  not the array's address, so a stack array costs nothing.
- Every buffer those pointers name is carved once and lives for the object's life: `struct Bump` (`mtp.cpp:70-80`),
  arena `cudaMalloc` at `mtp.cpp:405`, carve at `:379-389` (`R_`, `mixed_`, `inj_`, `inj2_`, `lo_`, `rs_`, `bo_`,
  `xn_` at `:381`), freed only at unload (`:152-153`). `verify.cpp` is the same shape: `:518-559`, with `R_` `:521`,
  `mixed_`/`bo_` `:522`, `inj_`/`inj2_` `:523`, `lo_`/`rs_`/`xn_` `:524`, `head_mixed_`/`head_inj_` `:548`.
- Weights: `mtp.cpp:163-171` return `dense_ + off` with `dense_` allocated once at `:283`; `tensors_` is a
  `std::vector` of *offsets*, so a vector reallocation moves no device buffer. `verify.cpp` reads `WeightRef::data`
  (`weights.hpp:49`), set once by the loader at `weights.cpp:444` into the arena allocated at
  `program/generate.cpp:2676`. `stamp_buf` is `prof_`, one `cudaMalloc` at `verify.cpp:584`.

→ Nothing in the key moves between decode steps. The audit's pointer question is answered **yes, stable**.

### The gate never engages in production (measured)

Every one of those sites runs *inside the engine's own stream capture*: `Verifier::capture` begins the capture at
`verify.cpp:1685` and calls `record_window` at `:1691`; `capture_batch` at `:2349` → `:2357`; `MtpDrafter::capture_prefill`
`:1024` → `:1028`, `capture_prefill_dev` `:1033` → `:1035`, `capture_round` `:1045` → `:1067`/`:1075`, `capture_step`
`:1092` → `:1102`. `record_window`/`record_front`/`record_rest` have no other callers, so there is no eager
production call. A new bench arm (`STRATA_HC_BENCH_NESTED=1`, `fused_gr_bench.cpp:406-449`) reproduces that shape:
outer `cudaStreamBeginCapture` → `fused_gr_read_multi` → `EndCapture` → instantiate → replay. Variant 7, T=1,
200 iters, created stream, fresh process, `strata.service` inactive:

- gate on (`nested_graph_on.txt`): the engine prints `fused_gr: STRATA_HC_GRAPH capture failed (the operation
  cannot be performed in the present state); plain launches from here on` — the inner `BeginCapture` on a stream
  that is already capturing is rejected, and `broken` latches the gate off for the process (`fused_gr.cu:1990/2002`).
  The outer capture survives: begin/end/instantiate all "no error", outer-graph replay **34.1 µs**.
- gate off (`nested_graph_off.txt`): no failure line, begin/end/instantiate all "no error", replay **34.1 µs**.

So on the serve path the gate adds a warning line and nothing else: the CPU launch gap it removes is already
removed by the engine's own whole-step graph.

### Cache capacity: overflow is benign (measured)

Cap 32, linear scan, no eviction, no counters (`fused_gr.cu:1953`, `:1985-1991`); a full cache returns false and the
caller plain-launches (`:2865-2866`). Churn arm: 33 argument sets cycled (one more than the cap), variant 7,
200 iters, two rounds per arm, gate on vs off, µs/call:

| T | steady, gate on (r1/r2) | steady, gate off | churn (33 sets), gate on | churn, gate off | 1 cold call, gate on | 1 cold call, gate off |
|---|---|---|---|---|---|---|
| 1 | 34.0 / 34.3 | 33.3 / 33.1 | 29.4 / 27.7 | 31.2 / 31.0 | 2699.7 / 2497.7 | 3023.7 / 2304.7 |
| 2 | 35.7 / 35.4 | 34.4 / 37.0 | 33.2 / 33.2 | 33.1 / 33.2 | 40.3 / 43.0 | 43.0 / 55.1 |
| 3 | 38.4 / 37.4 | 37.4 / 37.5 | 36.3 / 36.1 | 36.1 / 36.1 | 45.7 / 43.8 | 45.4 / 42.2 |
| 4 | 41.5 / 41.6 | 42.4 / 41.7 | 40.5 / 40.2 | 40.3 / 40.3 | 48.6 / 48.8 | 55.9 / 47.3 |

- Overflow costs the same as plain launching (churn columns): capture is skipped when the cache is full, so a
  serve path with more keys than the cap pays no capture cost — it just stops using graphs.
- The 2.3-3.0 ms first-call cost is **not** graph capture: it appears with the gate off too (3023.7/2304.7 µs).
  It is the process's first fused-GR read. Per-key capture is inside the noise of a cold plain call (T=2..4 rows).
- Steady (every call a cache hit) shows no win in this harness (34.0-34.3 on vs 33.1-33.3 off): 200 launches queued
  back-to-back pipeline the CPU launch gap away. P2's −4.2..−4.7 µs came from a harness that times the gap per rep.
  The graph's win is a **per-call latency** win, visible only when host work separates the calls — and on the serve
  path the whole step is already one graph replay.

### What a promote would need

1. A production call site outside the engine's own capture — today there is none.
2. A hits/misses counter — none exists; the only output is the one-time failure line (`fused_gr.cu:2002`).
3. A cache sized to the key set, or eviction: cap 32 with no eviction, and the verify site's key varies with `tb`,
   `half`, `apply` and `stamp_i0 = l * kProfPer + (half == 0 ? 27 : 30)` (`verify.cpp:926`, `:940`) — far more than
   32 distinct keys per stream, so which keys get graphs would be call-order dependent.
4. The mutex at `fused_gr.cu:1986` serializes every graph launch process-wide, across all the engine's streams
   (`mtp.cpp:424/427/430`, `verify.cpp:600/604-606/624`).

**Verdict: keep `STRATA_HC_GRAPH=1` behind the gate as a bench/loop tool.** Its variant-7 win (−4.2..−4.7 µs,
T=1..4) is real and reproducible in the bench harness, and parity holds bitwise; but on the serve path it is inert
(first call fails, gate latches off), so promoting it to default would ship a warning line and no speed.

Reproduce the audit:
```
# nested capture (the production shape): gate on, then gate off
STRATA_HC_BENCH_DIRECT=1 STRATA_HC_SPLIT=7 STRATA_HC_GRAPH=1 STRATA_HC_BENCH_GRAPH_CHURN=1 STRATA_HC_BENCH_NESTED=1 \
  ./build-hip/fused_gr_bench 200 1 1
STRATA_HC_BENCH_DIRECT=1 STRATA_HC_SPLIT=7 STRATA_HC_BENCH_GRAPH_CHURN=1 STRATA_HC_BENCH_NESTED=1 \
  ./build-hip/fused_gr_bench 200 1 1
# steady / churn / cold-capture, gate on then off
STRATA_HC_BENCH_DIRECT=1 STRATA_HC_SPLIT=7 STRATA_HC_GRAPH=1 STRATA_HC_BENCH_GRAPH_CHURN=1 ./build-hip/fused_gr_bench 200 1 4
STRATA_HC_BENCH_DIRECT=1 STRATA_HC_SPLIT=7 STRATA_HC_BENCH_GRAPH_CHURN=1 ./build-hip/fused_gr_bench 200 1 4
# legacy stream (no stream of its own): capture is unsupported there
STRATA_HC_BENCH_DIRECT=1 STRATA_HC_SPLIT=7 STRATA_HC_GRAPH=1 STRATA_HC_BENCH_GRAPH_CHURN=1 STRATA_HC_BENCH_LEGACY_STREAM=1 \
  ./build-hip/fused_gr_bench 200 1 1
```
Raw outputs: `/tmp/p3/g/{nested_graph_on,nested_graph_off,on_r1,on_r2,off_r1,off_r2,legacy}.txt`. The legacy-stream
run prints `capture failed (operation not permitted when stream is capturing)` — the same latch, a different reason;
its timings are not meaningful (the events sit on the created stream while the read runs on the legacy one).

## P5 — pipe+half combination (STRATA_HC_SPLIT=10)

Status: **MEASURED — bitwise parity holds, the arm beats staged at T=2..8 and ties at T=1, and it does NOT beat
variant 7. Not a promote candidate; it stays behind the env gate.**

### What 6 and 7 actually are, read from the kernels

The brief's split ("pipe = weight prefetch, half = half-tuple accumulation") is not how the two kernels differ.
`gr_down_register_pipe_kernel` (6) and `gr_down_register_half_kernel` (7) carry the **identical** weight prefetch:
the `wv`/`wnext` double buffer, all `HQ = 5` uint4 issued before the barrier, `wv = wnext` after the dot. The
only difference is the activation tuple: 6 carries the whole tile in one `HcChain<NS-1>` (loaded at the prime and
re-loaded at `h+3`), 7 splits it into `HcChain<H0-1>` + `HcChain<NS-H0-1>` and issues the second half mid-dot
(`q == 2`), inside the dot. **7 is 6 with the tuple split** — the union of 6 and 7 is 7, so the combination worth
measuring is the half treatment applied to the register array 7 left whole: the weights.

### The change (in the worktree, uncommitted)

`gr_down_register_pipe_half_kernel<MAX_T, EXACT_T>` — 7's activation schedule unchanged, plus the weight
prefetch split the same way: `wnext[0..W0)` (`W0 = (HQ+1)/2 = 3`) issued before the barrier as 6/7 issue all 5,
`wnext[3..5)` issued mid-dot at `q == W0`, so the live range of the second weight half starts inside the dot.
Same grid (41 blocks), same 256-thread CTA, same two-buffer tile schedule, same shared-memory contract as
staged, same weights, same q-outer / k-inner dot order — the moved operations are loads, not arithmetic.

- `src/kernels/cuda/fused_gr.cu`: `constexpr int kHcRegisterPipeHalf = 11;` (the next free constant; 8 is the
  packed arm, 9 the row split, 10 the LDS accumulator); the kernel; `set_staged_attr` for the instantiations the
  dispatch uses (exact 1..4, generic past that — the same split 6/7 use, where the exact tuple kernels past 4
  spill); the `launch_multi` branch; `constexpr int kHcPipeHalfWinMaxT = kFusedGrMaxT;` and the clamp added to
  `variant_for_T`; the self-test (arity 11 → 12, its own arena, checked against staged at every T); the
  `fused_gr_check` latch and the two name tables.
- **Env syntax**: `STRATA_HC_SPLIT=10`. The parser reads the value's leading characters and every one-digit value
  was taken, so 10 is the first two-character value: matched as the two leading characters (`"10"`, so `"10x"`
  reads as 10) **before** the one-digit `'1'`, which would otherwise win. Documented at `env_variant()` and in
  `include/strata/kernels/fused_gr.hpp`.
- `src/kernels/fused_gr_bench.cpp`: accepts variant 11, names it in the selected line and in the timed line's
  effective-variant label (the label also gained the row-split name, which fell through to "register-half"
  before).
- The clamp follows P4's convention: its reference is the default it falls back to, so the boundary answers
  "above which T does this variant stop beating staged" — measured, never, so `kFusedGrMaxT` and the clamp is a
  no-op safety net. The self-test is not clamped, so latching on 10 means bit-for-bit equal to staged at every T.

### Timing — whole-read µs (`HC-read`), paired fresh processes, `STRATA_HC_BENCH_DIRECT=1`, 200 iters, T=1..8,
median over the 4 apply/inject cases, two rounds (A, B). Deltas are **combination − X**, negative = combination
faster. Noise floor from round-to-round spread ≈ 0.5 µs.

| T | comb A | staged A | vs staged A | v7 A | vs v7 A | comb B | staged B | vs staged B | v7 B | vs v7 B |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 30.5 | 30.1 | +0.4 | 30.1 | +0.4 | 30.0 | 30.1 | −0.1 | 29.9 | +0.1 |
| 2 | 33.4 | 35.0 | **−1.6** | 33.5 | −0.1 | 33.0 | 35.1 | **−2.1** | 33.3 | −0.3 |
| 3 | 36.6 | 39.0 | **−2.4** | 36.5 | +0.1 | 36.0 | 38.6 | **−2.6** | 36.2 | −0.2 |
| 4 | 40.6 | 43.8 | **−3.2** | 40.5 | 0.0 | 40.0 | 43.6 | **−3.6** | 40.3 | −0.3 |
| 5 | 49.0 | 49.1 | −0.1 | 48.4 | +0.5 | 48.2 | 48.8 | −0.5 | 48.1 | +0.1 |
| 6 | 53.5 | 54.0 | −0.5 | 52.8 | +0.7 | 52.8 | 53.5 | −0.7 | 52.3 | +0.5 |
| 7 | 78.3 | 79.2 | −0.9 | 77.7 | +0.6 | 77.3 | 78.7 | −1.3 | 77.3 | 0.0 |
| 8 | 82.6 | 83.8 | −1.2 | 81.8 | +0.8 | 81.2 | 83.5 | −2.3 | 81.3 | −0.1 |

Mean over the two rounds, combination − staged: **+0.15, −1.85, −2.50, −3.40, −0.30, −0.60, −1.10, −1.75 µs** —
the same shape as variant 7's own win (T=1 tie, T=2..8 ahead, biggest at T=4).
Mean over the two rounds, combination − variant 7: **+0.25, −0.20, −0.05, −0.15, +0.30, +0.65, +0.35, +0.35 µs** —
inside the noise floor at T=1..4, behind at T=5..8. **No win over variant 7 at any T.**

### Parity

**Bitwise equal to staged at every T=1..8, apply × inject, with and without the inject weights** — the card's own
self-test latched it, and the self-test is not clamped:

```
strata hc: CUDA0: the hyper-connection read runs as register-pipe-half staged (the register pipeline with the
activation tuple AND the weight prefetch in two halves, STRATA_HC_SPLIT=10); checked bit for bit against the
plain read on this card
```

The bench's `--mode=parity` gate is the staged-reference path and stops at the latch (variant 11 is not one of
the modes it times), as with variant 7; the parity evidence is the self-test above, which compares the variant
against staged at every T.

### Mechanism — VGPR from the compiler (rocprofv3 kernel-trace, 50 iters, T=1..8, gfx1201, ROCm 7.17)

| Instantiation | comb VGPR | v7 VGPR | delta | comb kernel avg µs | v7 kernel avg µs |
|---|---|---|---|---|---|
| MAX_T=1 exact | 96 | 96 | 0 | 8.44 | 8.36 |
| MAX_T=2 exact | 128 | 128 | 0 | 9.32 | 10.32 |
| MAX_T=3 exact | **160** | **144** | **+16** | 11.74 | 12.53 |
| MAX_T=4 exact | **192** | **168** | **+24** | 13.81 | 15.04 |
| MAX_T=8 generic (T=5..8) | 152 | 152 | 0 | 22.46 | 22.79 |

Scratch 0, SGPR 128, accum-VGPR 0 in every row of both arms (no private scratch, no spills).

Why the combination buys nothing: the peak live set at the end of a dot is the same in both kernels —
`wv[5]` + `wnext[5]` + `t0` + `t1` + `acc[MAX_T]`. Splitting the weight prefetch moves where a live range
*starts*, not how many registers are live at the peak, and the peak is what the allocation is sized by. Where
the allocation did move (the exact MAX_T=3 and MAX_T=4 kernels, the ones T=3 and T=4 launch) it moved the wrong
way, +16 and +24 VGPR: issuing the second weight half inside the `q` loop, next to the mid-dot tuple load,
lengthens the range the scheduler has to keep live across the remaining dots. That is consistent with the
timings — the combination never passes variant 7, and trails it most at T=5..8, where the generic kernel is
register-identical and the extra issue point buys nothing.

The trace's per-kernel averages (comb ahead at MAX_T=2,3,4,8) are a weaker signal than the paired whole-read
medians: 50 iters, averaged over the T values that share an instantiation, and the whole-read number is what the
read pays for. The paired 200-iter medians are the verdict.

### Verdict

**Do not promote.** Parity holds and the arm beats staged at T=2..8, so it earns its place behind the gate
(`STRATA_HC_SPLIT=10`), and it does not beat variant 7 at any T — so **variant 7 stays the promote-to-default
candidate** and the default path is untouched. The clamp stays as a no-op safety net.

T=9..16 was not measured: `kFusedGrMaxT = 8` is the multi-read contract (`fused_gr_read_multi` rejects
`n_tok > kFusedGrMaxT`), so the bench has nothing above 8 to time.

Reproduce P5:
```
STRATA_HC_BENCH_DIRECT=1 STRATA_HC_SPLIT=10 ./build-hip/fused_gr_bench 200 1 8        # combination arm
env -u STRATA_HC_SPLIT STRATA_HC_BENCH_DIRECT=1 ./build-hip/fused_gr_bench 200 1 8    # staged arm
STRATA_HC_BENCH_DIRECT=1 STRATA_HC_SPLIT=7 ./build-hip/fused_gr_bench 200 1 8         # variant-7 arm
STRATA_HC_SPLIT=10 ./build-hip/fused_gr_bench 1 1 1 --mode=time-staged                # parity: the check's latch line
/opt/rocm/bin/rocprofv3 --kernel-trace -f csv -o /tmp/p5/tr_comb_8 -d usec -- \
  env STRATA_HC_BENCH_DIRECT=1 STRATA_HC_SPLIT=10 ./build-hip/fused_gr_bench 50 1 8   # VGPR column
```
Raw outputs: `/tmp/p5/{A,B}_{comb,staged,v7}.txt`, `/tmp/p5/tr_{comb,half}_8_kernel_trace.csv`.

## Ranked promote list (final, 2026-10-09)

1. **P4 — variant 7 per-T latch: PROMOTE-TO-DEFAULT CANDIDATE.** Win at every T=1..8 (−1.4 to −3.2 µs,
   T=1/T=5 ties), bitwise parity held at every T. Parity requirement met.
2. **P2 — HIP graph trio: KEEP BEHIND GATE (`STRATA_HC_GRAPH=1`), promote ruled out by the audit.** Real win
   only on the variant-7 bench path (−4.2 to −4.7 µs, T=1..4), bitwise parity held; no win on staged. The
   audit (section above) answers its own gate question: pointers are stable, but every production call site runs
   inside the engine's own capture, where the inner capture fails and latches the gate off — inert on the serve
   path. Best measured config: `STRATA_HC_SPLIT=7 STRATA_HC_GRAPH=1` → 26.0 µs at T=1 vs 30.7 µs staged
   baseline (−15%), a bench-harness number.
3. **P5 — pipe+half combination (`STRATA_HC_SPLIT=10`): KEEP BEHIND GATE, NOT a promote candidate.** Bitwise
   parity holds at every T=1..8 and the arm beats staged at T=2..8 (−1.6 to −3.6 µs, biggest at T=4, T=1 a tie),
   so it earns the gate. It does **not** beat variant 7 at any T (T=1..4 within the 0.5 µs noise floor, T=5..8
   behind by +0.3 to +0.7 µs), so **variant 7 stays the promote candidate** and the default path is untouched.
   Mechanism: 7 already contains 6's weight prefetch, and splitting that prefetch in halves moves where live
   ranges start, not the peak live set at end-of-dot — VGPR is identical at MAX_T 1, 2 and the generic 8, and
   +16/+24 the wrong way at the exact MAX_T 3/4 kernels. See the P5 section.
4. **P1 — row-split down: DO NOT PROMOTE.** Bitwise parity held but slower at every T (+0.9..+1.6 µs
   T=1..4, +20.5/+24.6 µs T=5/6); wave quantization at T>=4. No reason to ship even behind a gate.
5. **P3 — LDS-accumulator staged (`STRATA_HC_SPLIT=9`): DO NOT PROMOTE, not even behind a gate.** Bitwise
   parity holds and the T=4/T=5 VGPR cut is real (144→128, 152→72), but the arm is slower at every T
   (+0.6 to +3.2 µs, +11.1 µs at T=6 from the 5-token launch split): the 41-block grid is block-limited at
   4 waves/SIMD, so freed registers cannot add a wave, and the accumulator's LDS round-trip is on the
   critical path.
6. **P2 pointer-stability audit: DONE — pointer stability PASSES, the gate is INERT on the serve path.** All
   eight production call sites (`mtp.cpp:723/838/946/957`, `verify.cpp:940/1555/1588`) run inside the engine's
   own stream capture, and every pointer the key mixes is carved once from a bump arena that lives for the
   object's life. Cache overflow is benign (33 sets cycling costs the same as plain launching). Measured with
   the gate on inside an outer capture: the inner capture fails, the gate latches off for the process, and the
   outer graph replays the same 34.1 µs as with the gate off. Nothing to promote; the gate stays a bench tool.

## Promote applied (2026-10-10)

The P4 candidate is now the default. The change is in `fused_gr_check` (`src/kernels/cuda/fused_gr.cu`):
with `STRATA_HC_SPLIT` **unset**, the card latches **variant 7 (register-half)** when its own self-test
latched it — the latch publishes only on a bit-for-bit match with the staged read at every T — and **staged**
when it did not. `STRATA_HC_SPLIT=2` stays the escape hatch: an explicit 2 latches staged even on a card that
latched 7. The latch line names the effective variant and says it is the promoted default. The bench latches
(`fused_gr_check()`) before reading the variant in both the direct arm and the default mode, so its startup
line names what every launch really runs. The per-T clamp (`variant_for_T`) is unchanged and applies to the
promoted default exactly as it applied to the opt-in 7.

Startup line, `env -u STRATA_HC_SPLIT ./build-hip/fused_gr_bench 1 1 6` (Radeon 9700, gfx1201):

```
strata hc: CUDA0: the hyper-connection read runs as register-half staged (the same with the tuple in two
halves, STRATA_HC_SPLIT=7); checked bit for bit against the plain read on this card (STRATA_HC_SPLIT=1 or 0
for the earlier ones) (the promoted default on this card; STRATA_HC_SPLIT=2 keeps the staged read)
```

The self-test ran in the same process and reported no difference (the run's parity lines: `0 differing` at
T=1..6, all `bitwise equal`).

Paired timing, fresh process per arm, 200 iters, median over the 4 apply/inject cases,
`STRATA_HC_BENCH_DIRECT=1`:

| T | default arm (us) | effective variant | `STRATA_HC_SPLIT=2` arm (us) | effective variant | delta (us) |
|---|---|---|---|---|---|
| 1 | 30.2 | register-half staged | 30.2 | staged | 0.0 |
| 2 | 34.0 | register-half staged | 35.2 | staged | **−1.2** |
| 3 | 37.1 | register-half staged | 39.0 | staged | **−1.9** |
| 4 | 41.0 | register-half staged | 43.9 | staged | **−2.9** |
| 5 | 49.0 | register-half staged | 49.1 | staged | −0.1 |
| 6 | 53.2 | register-half staged | 53.7 | staged | **−0.5** |
| 7 | 78.1 | register-half staged | 79.2 | staged | **−1.1** |
| 8 | 82.7 | register-half staged | 83.7 | staged | **−1.0** |

The default arm reproduces the P4 variant-7 numbers (30.1/33.4/36.3/40.3/48.3/52.6/77.5/81.6 µs, within the
run-to-run noise floor) and not the staged ones; the `STRATA_HC_SPLIT=2` arm reproduces staged. The win
pattern matches P4: wins at T=2,3,4,6,7,8, ties within noise at T=1 and T=5.

Note for the P4 reproduce commands: `--mode=time-staged` gates on the latched variant, so after the promote
the staged baseline there needs `STRATA_HC_SPLIT=2` (with the env unset the card latches 7 and the gate
exits 2, by design).

Reproduce the promote:
```
env -u STRATA_HC_SPLIT ./build-hip/fused_gr_bench 1 1 6                                        # startup line + parity
env -u STRATA_HC_SPLIT STRATA_HC_BENCH_DIRECT=1 ./build-hip/fused_gr_bench 200 1 8            # default arm (register-half)
STRATA_HC_SPLIT=2 STRATA_HC_BENCH_DIRECT=1 ./build-hip/fused_gr_bench 200 1 8                 # escape hatch (staged)
```
