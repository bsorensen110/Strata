# Wave 7 Concurrent Decode + Co-Prefill Speedup Report (2026-11-08)

## Summary

Wave 7 builds on Wave 6 (`cd893f82d`) and Wave 5 (`d36dfef63`, production baseline) to accelerate CPU expert sharing and eliminate host-buffer reallocation stalls and probe-layer drops during `--batch` co-prefill on single-card `gfx1201` (AMD Radeon AI PRO R9700, 32 GiB VRAM, AMD Ryzen 9 9950X 16C/32T AVX-512):

1. **8-Token Single-Pass AVX-512 CPU Expert Row-Group Kernels (`kIq512MaxRowGroupTokens = 8`)**:
   - Extended `kIq512MaxRowGroupTokens` (`kKQMaxT`) from `4` to `8` across `include/strata/kernels/cpu/iq_avx512.hpp`, `src/kernels/cpu/iq_avx512.cpp` (`dot_rows_nt_g`, `gu_rows_x_nt_g`, `iq4nl512_down_rows`, `q2_512_down_rows`, `kq512_row`, `kq512_gu_row`), and `src/kernels/cpu/native_expert.cpp`.
   - Each `MAXT = 8` token job slice now reads the expert weight matrix once per `1..8` tokens instead of splitting `5..8` token slices into two 4-token passes, while keeping `MAXT = 8` fixed so `off[e]` and `Dm` reduction boundaries remain bit-identical (`ece1433a582a4c38`).
2. **Disable Redundant CPU-Share Off-Probing on Single-Card `gfx12` (`STRATA_PREFILL_CPU_GATE_PROBE`)**:
   - Added `share_rules::cpu_share_gate_probe_from` and `share_rules::cpu_share_should_run` (`include/strata/prefill/share_rules.hpp`).
   - On single-card `gfx12`, where CPU expert sharing is always a net win, `probe_gate` defaults to `false` (overridable via `STRATA_PREFILL_CPU_GATE_PROBE=1` or `STRATA_DBG_CPU_GATE=1`). This stops dropping `cpu_arm` on calibration layers `i = 2, 4` and periodic `i % 29 == 0` probe layers (`+3` to `+4` more layers per chunk using CPU expert offload) and skips recording 4 `cudaEventRecord(m.cpu_wall[...])` events per layer.
3. **One-Time Max-Chunk Pinned & Host Buffer Sizing + Warm-Bucket Layer-0 Offload**:
   - In `Prefill::run_impl` (`src/prefill/prefill.cpp`), `m.cpu_x`, `m.cpu_rows`, `m.cpu_ones`, `m.cpu_nact`, and `m.cpu_need` are now sized once at `cpu_max_t = std::max<int64_t>(T, stream_all_min() - 1)` (`2560` tokens) on the first `cpu_maybe` call (startup warmup), eliminating mid-session `cudaFreeHost` + `cudaHostAlloc` stalls (`150–349 ms` in `host grouping`) when `T` grows from warmup (`19–21`) to `309`, `609`, or `2048`.
   - When `cpu_buffers_warm` is true (`m.cpu_x_n >= cpu_alloc_want && m.cpu_rows_n >= cpu_rows_cap`) and `!probe_gate`, Layer 0 (`i == 0`) of a new chunk-size bucket (`T = 309, 609, 2048`) immediately runs CPU expert sharing (seeded with `m.cpu_buckets[0].share_now` while keeping `cb.c_ms == 0` so Layer 0's own bucket measurement initializes `cb.c_ms` and `cb.g_ms` on Layer 1) and reuses `m.cpu_need` across layers instead of allocating a temporary `std::vector<char>` per layer.
4. **Opt-In Lead-Resident MMQ Staged Sort Mode 4 (`STRATA_MMQ_STAGED_SORT=4`)**:
   - Added Mode 4 (`src/prefill/mmq_resident_sort.hpp`) as an opt-in mode that extracts a lead pure-resident group from the highest-count resident experts while keeping Mode 3 (`STRATA_MMQ_STAGED_SORT=3`) as the default when `default_on = true` (head-to-head A/B confirmed Mode 3's natural mixing of resident and streamed experts per 8-expert group achieves `-15 to -22 ms` lower `wait copy` per 2,048-token chunk than Mode 4).

## Unit & GPU Parity Gates

All unit and GPU parity tests built and passed (`10/10` CTest targets, `0` skipped, `0` failed):

- `share_rules_test`: Passed (`0.00 s`)
- `mmvf_rows_parity`: Passed (`1.23 s`)
- `shared_expert_parity`: Passed (`0.13 s`)
- `gr_parity`: Passed (`1.59 s`)
- `spec_verify_parity`: Passed (`1.91 s`)
- `verify_batch_parity`: Passed (`0.15 s`)
- `arch_defaults_test`: Passed (`0.00 s`)
- `native_arithmetic_parity` (including `static_assert(kIq512MaxRowGroupTokens == 8)` and `nt = 12, 13, 14` multi-row AVX-512 parity): Passed (`0.09 s`)
- `pool_tasks_test`: Passed (`13.51 s`)
- `copy_events_test`: Passed (`0.03 s`)

## Paired 4-Arm A/B Benchmark Results (`--batch 2`, `qwen3.8-flash-next-unsloth-ud-iq4_xs-v31`)

- **Control binary (Wave 6 `cd893f82d`)**: `/home/soren/.hermes/cache/scratch/strata-wave6-cd893f82d` (`sha256: 71e76c2a3b6aa6f5bde9fbc944fb5769830e58add2e629dcb407e4053067c3a4`)
- **Candidate binary (Wave 7)**: `/home/soren/Strata-perf-dflash2-on-main-20261103/build-hip/strata`
- **Greedy output hash (`PROMPT_REFACTOR`, 192 tokens, `temp=0`)**: **`ece1433a582a4c38`** across all 4 arms and all runs (`100%` bit-identical).

### Arm-by-Arm Results

- **`0_control_wave6` (Wave 6 control launch 1)**:
  - Greedy hash: `ece1433a582a4c38` (`2/2`)
  - 4,712-token co-prefill prompt read: **`3,573.0 ms` mean** (`3,893 ms` R0 [`1,210.3 tok/s`, hit `349 ms` `host grouping` `cudaHostAlloc` stall on first `T=2048` chunk] / `3,253 ms` R1 [`1,448.7 tok/s`])
  - Concurrent `pf+dec` aggregate (`Slot 0 256g + Slot 1 4,712p+128g`): **`65.6 tok/s`** (`61.6 / 69.5 tok/s`)
  - Slot 0 decode during 4,712-token co-prefill: **`43.7 tok/s`** (`41.1 / 46.4 tok/s`)
  - Slot 1 end-to-end latency (`4,712p + 128g`): **`5.33 s`** (`5.59 s / 5.07 s`)
  - Concurrent Batch-2 decode (`c=2`, `2 x 256g`): **`134.1 tok/s`** (`130.9 / 137.3 tok/s`, per-slot `70.6 tok/s`)
- **`1_cand_wave7_default` (Wave 7 default, `STRATA_MMQ_STAGED_SORT=3`)**:
  - Greedy hash: `ece1433a582a4c38` (`2/2`)
  - 4,712-token co-prefill prompt read: **`3,238.0 ms` mean (`1,455.6 tok/s`, `-214.5 ms` / `+6.1%` vs Wave 6 control mean, `-364 ms` / `+11.3%` vs Wave 5 production)**
    - Round 0: **`3,293 ms` (`1,430.8 tok/s`)** — `T=2048` chunks: `920 ms` (`host grouping 24 ms`, `wait copy 50 ms`), `962 ms` (`wait copy 77 ms`); `T=609` chunk: `624 ms` (`wait copy 101 ms`)
    - Round 1: **`3,183 ms` (`1,480.4 tok/s`)** — `T=2048` chunks: `889 ms` (`wait copy 48 ms`), `943 ms` (`wait copy 79 ms`); `T=609` chunk: `588 ms` (`wait copy 92 ms`)
  - Concurrent `pf+dec` aggregate (`Slot 0 256g + Slot 1 4,712p+128g`): **`69.3 tok/s` (`67.1 / 71.6 tok/s`, `+4.6%` vs Wave 6 control mean `66.3 tok/s`, `+6.0%` vs Wave 5 production `65.4 tok/s`)**
  - Slot 0 decode during 4,712-token co-prefill: **`46.5 tok/s` (`44.7 / 48.3 tok/s`, `+5.3%` vs Wave 6 control mean `44.2 tok/s`, `+6.7%` vs Wave 5 production `43.6 tok/s`)**
  - Slot 1 end-to-end latency (`4,712p + 128g`): **`5.05 s` (`5.08 s / 5.01 s`, `-110 ms` vs Wave 6 control mean `5.16 s`)**
  - Concurrent Batch-2 decode (`c=2`, `2 x 256g`): **`137.5 tok/s` (`140.1 / 134.9 tok/s`, per-slot `71.0 tok/s`, `+1.5%` vs Wave 6 control mean `135.5 tok/s`)**
- **`2_cand_wave7_sort4` (Wave 7 + `STRATA_MMQ_STAGED_SORT=4` ablation)**:
  - Greedy hash: `ece1433a582a4c38` (`2/2`)
  - 4,712-token co-prefill prompt read: **`3,287.5 ms` mean** (`3,277 ms` R0 [`1,438.0 tok/s`] / `3,298 ms` R1 [`1,428.7 tok/s`])
  - Concurrent `pf+dec` aggregate: **`67.4 tok/s`** (`65.5 / 69.4 tok/s`)
  - Slot 0 decode during co-prefill: **`45.0 tok/s`** (`43.6 / 46.3 tok/s`)
  - Concurrent Batch-2 decode (`c=2`): **`135.4 tok/s`** (`130.8 / 140.0 tok/s`)
- **`3_control_wave6_repeat` (Wave 6 control launch 2)**:
  - Greedy hash: `ece1433a582a4c38` (`2/2`)
  - 4,712-token co-prefill prompt read: **`3,332.0 ms` mean** (`3,334 ms` R0 [`1,413.1 tok/s`] / `3,330 ms` R1 [`1,415.0 tok/s`])
  - Concurrent `pf+dec` aggregate: **`66.9 tok/s`** (`66.0 / 67.9 tok/s`)
  - Slot 0 decode during co-prefill: **`44.6 tok/s`** (`44.0 / 45.3 tok/s`)
  - Slot 1 end-to-end latency (`4,712p + 128g`): **`4.98 s`** (`5.03 s / 4.93 s`)
  - Concurrent Batch-2 decode (`c=2`): **`136.9 tok/s`** (`139.2 / 134.7 tok/s`, per-slot `70.6 tok/s`)

## Reproduction Command

```bash
trap "systemctl --user start strata.service" EXIT
systemctl --user stop strata.service
while fuser /dev/kfd >/dev/null 2>&1; do sleep 1; done
systemd-run --user --scope -- python3 /home/soren/.hermes/cache/scratch/bench_concurrent_prefill_decode_ab.py
```
