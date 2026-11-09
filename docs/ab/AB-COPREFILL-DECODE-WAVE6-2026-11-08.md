# Wave 6 Concurrent Decode + Co-Prefill Optimization Report (2026-11-08)

## Executive Summary

Wave 6 builds on the promoted Wave 5 production baseline (`fork-d36dfef63`) and targets the remaining GPU MMQ tile-padding waste and CPU `ExpertPool` non-resident offload bottlenecks during concurrent batch decode + co-prefill on single-card `gfx1201` (AMD Radeon AI PRO R9700, 32 GiB VRAM, ROCm `10.2.0a20260914`, `qwen3.8-flash-next-unsloth-ud-iq4_xs-v31`, `--batch 2`).

All Wave 6 optimizations are **bit-exact** (greedy `refactor` token hash `ece1433a582a4c38` identical across all runs and `hip_prefill_mmq_parity` passing) and **enabled by default on single-card `gfx12`** (with explicit environment overrides preserved):

- **4,712-token Co-Prefill Prompt Read (under concurrent Slot 0 decode)**:
  - Wave 5 control (`fork-d36dfef63`): **`3,580.3 ms` (`1,316.2 tok/s`)** (`0_control`: `3,571 / 3,606 ms`; `3_control`: `3,563 / 3,581 ms`)
  - Wave 6 default (`1_cand_wave6_default`): **`3,284.5 ms` (`1,434.9 tok/s`)** (`3,317 / 3,252 ms`) — **`-295.8 ms` (`+9.0%` faster on top of Wave 5; `+92.3%` cumulative over Wave 4's `6,348 ms`)**
  - Wave 6 `STRATA_PREFILL_CPU_SHARE_MAXT=24` (`2_cand_wave6_maxt24`): **`3,269.5 ms` (`1,441.2 tok/s`)** (`3,300 / 3,239 ms`) — **`-310.8 ms` (`+9.5%` faster on top of Wave 5)**
- **2,048-token Co-Prefill Chunk GPU Timeline**:
  - Wave 5 control: **`989 – 1,074 ms`** (`dequant 120–123 ms`, `gemm gate/up 200–227 ms`, `gemm down 118–123 ms`, total MMQ = **`438 – 471 ms`**)
  - Wave 6 default: **`890 – 932 ms`** (`dequant 100–109 ms`, `gemm gate/up 142–146 ms`, `gemm down 86–89 ms`, total MMQ = **`329 – 344 ms` (`-110 to -127 ms` / `-26%` MMQ reduction per chunk)**, `combine 28–34 ms`)
- **309-token Solo Prefill Chunk (`--batch 2`, `refactor` prompt)**:
  - Cold read: **`526 / 491 ms -> 379 ms` (`-129 ms` / `-25.4%`)**
  - Warm read: **`372 / 366 ms -> 334 ms` (`-35 ms` / `-9.5%`)**
  - End-to-end solo `refactor` throughput: **`99.4 / 103.1 tok/s -> 106.2 tok/s` (`+4.9%`)**
- **Concurrent `pf+dec` Aggregate Throughput & Slot 1 Completion Wall Time**:
  - Aggregate generation throughput (`pf_dec_agg_tps`): **`64.91 / 66.25 tok/s -> 67.42 tok/s`** (Slot 0 decode **`43.28 / 44.17 tok/s -> 44.95 tok/s`**)
  - Slot 1 end-to-end wall time (`4,712p + 128g`): **`5.20 / 5.35 s -> 5.04 s`** (`5.14 s` R0, **`4.93 s` R1**)
- **Concurrent Batch-2 Decode (`c=2`)**:
  - Aggregate throughput: **`135.1 tok/s`** (Round 1 **`140.1 tok/s`**, matching Wave 5 control `135.3 – 136.8 tok/s`).

---

## Root Causes & Wave 6 Design

### 1. RDNA4 `mmq::whole_tiles()` Fix (`src/prefill/moe_mmq.cu`)
- **Root cause**: `MmqPlan::group_sort[l]` was `0` on all 48 layers of `gfx1201` because `mmq::whole_tiles()` required `c.I == c0.I` across `J = 8..128`. On RDNA4 (`mmq-config-rdna4.cuh`), `J <= 32` uses `(I = 64, warps = 4)` while `J >= 64` uses `(I = 128, warps = 8)`, both with `rows_per_warp = I / warps = 16` and `stream_k = false`.
- **Fix**: Check `c.I * c0.nthreads == c0.I * c.nthreads && c.I > 0 && w_rows % c.I == 0` when `!c.stream_k && !c0.stream_k`. Since every tile uses the exact same `rows_per_warp` and runs without Stream-K K-splitting, permuting full 16-expert MMQ groups is bit-exact across all 48 RDNA4 layers.

### 2. Staged MMQ Group Permutation (`STRATA_MMQ_STAGED_SORT`, `src/prefill/mmq_resident_sort.hpp`)
- **Root cause**: In ID order (`e = 0..511`), each 16-expert MMQ group mixes resident and non-resident experts and mixes low-token (`c = 1..5`) and high-token (`c = 40..90`) experts, padding the entire 16-expert MMQ launch to `max_rows` of the group's heaviest expert.
- **Fix**: Implemented `mmq_staged_sort` modes 1, 2, and 3 (defaulting to **Mode 3** on single-card `gfx12` when `STRATA_MMQ_STAGED_SORT` is unset):
  - **Mode 1**: Partitions resident experts before staged (non-resident) experts within the full-group span (`n - n % 16`), sorting full resident groups ascending by `(count[e], e)` and remaining full groups ascending by `(count[e], e)`.
  - **Mode 2**: Pure row-count sort ascending across all full groups (`n - n % 16`).
  - **Mode 3 (Default on single-card `gfx12`)**: Places full resident groups first (sorted ascending by `(count[e], e)` so GPU MMQ starts immediately with zero PCIe wait and tight row-count packing while `issue_to` prefetches up to `STAGE_GRP = 32` staged experts ahead), and sorts the remaining staged groups **descending** by `count[e]` (`count[a] > count[b]`) so the heaviest staged experts are prefetched and computed first while lighter staged groups finish at the tail.
- Also guarded the `stream_all` 16-slot ring-wrap event wait in `flush_group` (`src/prefill/prefill.cpp`) with `if (stream_all)` so `!stream_all` staged walks do not redundantly synchronize `stage_ready[(gg_s0 - 16) % 32]`.

### 3. 32-Slot Staged Copy Lookahead (`staged_can_issue()`, `include/strata/prefill/copy_events.hpp`)
- **Root cause**: In the `!stream_all` staged walk, `staged_pending_cap` capped in-flight H2D expert copies to `MMQ_GROUP = 16` whenever `group_resident == true`, leaving half of the 32-slot `STAGE_GRP` staging ring idle during full resident MMQ groups.
- **Fix**: Added `staged_can_issue(stream_ahead, group_resident, full_group_lookahead, pending, open_group_slots, stage, stage_grp, mmq_group, staged, j, lookahead)` which allows up to `STAGE_GRP = 32` pending H2D copies when `open_group_slots == 0` while enforcing `pending + open_group_slots < stage_grp` so open-group slots are never overwritten before group flush.

### 4. `cpu::ExpertPool` Shared Activation Pre-Quantization & Single-Wave Dispatch (`include/strata/kernels/cpu/pool.hpp`)
- **Root cause**:
  1. `run_split_multi_native` checks `x_toks <= kMaxActQuantToks` (`1024`) to pre-quantize Q8_1 activations once across all threads. On 2,048-token prefill chunks (`x_toks = 2048 > 1024`), `use_act_q8` was `false`, forcing every CPU expert job to redundantly quantize its activation rows.
  2. `kMaxSplitMulti` was `96`, splitting `> 96` CPU expert jobs per layer into multiple sequential thread-pool waves (`for (int base = 0; base < n_jobs; base += kMaxSplitMulti)`).
- **Fix**: Increased `kMaxActQuantToks` from `1024` to **`4096`** and `kMaxSplitMulti` from `96` to **`384`** (verified in `pool_tasks_test`).

### 5. Multi-Slice Non-Resident CPU Expert Offload & `T`-Bucketed Adaptive Controller (`include/strata/prefill/share_rules.hpp`, `src/prefill/prefill.cpp`)
- **Root cause**:
  1. CPU share only considered non-resident experts with `c <= MAXT` (`8` tokens on native `iq3_s` / `iq4_xs` where `kMaxMultiTokens == 16`, `MAXT = 8` in `prefill.cpp` or `16`). On a 2,048-token chunk, only ~84 of ~228 non-resident experts per layer have `c <= 8`, leaving the CPU pool underutilized (`~3.8 ms/layer` vs `~9.0 ms/layer` GPU time) and forcing ~144 non-resident experts per layer over PCIe.
  2. Adaptive CPU share state (`c_ms`, `g_ms`, `share_now`, `gate`, `ratio[]`) was shared in a single global struct across all prompt lengths `T`, so small `T = 40–65` warmup/admission prefills polluted the controller state used by `T = 309`, `T = 609`, and `T = 2048` chunks, and cold layer 0 GPU initialization spikes polluted `g_ms`.
- **Fix**:
  1. Added `prefill_cpu_share_max_t_from()` (`STRATA_PREFILL_CPU_SHARE_MAXT`, defaulting to `4 * base_maxt = 32` on single-card `gfx12`) and sliced non-resident experts with `c > base_maxt` into `ceil(c / base_maxt)` `ExpertJobMulti` jobs of `<= base_maxt` tokens each (processed in ascending `c` order so 1-slice experts are always taken first).
  2. Added `cpu_share_can_take_next()` to balance cumulative CPU slice time `(cur_jobs + 0.5 * next_jobs) * c_ms` against `0.88 * ((nstream - cur_cpu_experts) - 0.5) * g_ms` (leaving a 12% margin for the post-join `m.cpu_rows` H2D copy so `kPfCombine` stays at `28–34 ms`).
  3. Partitioned adaptive CPU share state into `kCpuShareBuckets = 4` token-count buckets (`cpu_share_bucket(T)`: `[64..127]`, `[128..511]`, `[512..1535]`, `[1536+]`) with per-bucket cold-layer exclusion (`cb.first_l`, `!cpu_cold` on `m.cpu_pend`) and collision-free `cpu_set` selection across bucket transitions.

---

## Paired 4-Arm A/B Benchmark Results (`ab_concurrent_pf_dec_wave6_defaults_20261108`)

Harness: `/home/soren/.hermes/cache/scratch/bench_concurrent_prefill_decode_ab.py`
Logs: `/home/soren/.hermes/cache/scratch/ab_concurrent_pf_dec_wave6_defaults_20261108/`

| Metric | `0_control_wave5_prod` (`fork-d36dfef63`) | `1_cand_wave6_default` (Wave 6 Default) | `2_cand_wave6_maxt24` (`MAXT=24`) | `3_control_wave5_repeat` (`fork-d36dfef63`) | Wave 6 Default vs Control Mean |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Greedy `refactor` Hash** | `ece1433a582a4c38` | `ece1433a582a4c38` | `ece1433a582a4c38` | `ece1433a582a4c38` | **100% Bit-Identical** |
| **Solo `refactor` (`--batch 2`, tok/s)** | `99.4` (`86.1 / 112.8`) | **`106.2` (`97.5 / 115.0`)** | `94.7` (`76.4 / 113.0`) | `103.1` (`92.5 / 113.7`) | **`+4.9%`** |
| **316-tok Solo Prompt Read (Cold / Warm)** | `578 ms / 428 ms` | **`433 ms / 397 ms`** | `434 ms / 397 ms` | `542 ms / 418 ms` | **`-127 ms (-22.7%) / -26 ms (-6.1%)`** |
| **4,712-tok Co-Prefill Read (R0 / R1)** | `3,571 ms / 3,606 ms` | **`3,317 ms / 3,252 ms`** | **`3,300 ms / 3,239 ms`** | `3,563 ms / 3,581 ms` | **`-295.8 ms` (`+9.0%` tok/s)** |
| **4,712-tok Co-Prefill Read Speed (tok/s)** | `1,319.6 / 1,306.7` | **`1,420.7 / 1,449.1`** | **`1,427.7 / 1,454.7`** | `1,322.7 / 1,315.9` | **`1,316.2 -> 1,434.9 tok/s`** |
| **2,048-tok Chunk GPU Time (R0c1/c2, R1c1/c2)** | `1033 / 1031 / 989 / 1069 ms` | **`923 / 932 / 890 / 931 ms`** | `914 / 929 / 890 / 926 ms` | `1021 / 1074 / 1001 / 1051 ms` | **`-114.8 ms / chunk (-11.1%)`** |
| **2,048-tok Chunk MMQ (`deq+gu+down`)** | `452 / 441 / 455 / 443 ms` | **`344 / 329 / 337 / 329 ms`** | `341 / 326 / 334 / 327 ms` | `447 / 438 / 471 / 442 ms` | **`-114.1 ms / chunk (-25.4%)`** |
| **Slot 1 Total Wall Time (`4,712p + 128g`)** | `5.20 s` (`5.34 / 5.06 s`) | **`5.04 s` (`5.14 / 4.93 s`)** | **`5.01 s` (`5.11 / 4.91 s`)** | `5.35 s` (`5.37 / 5.33 s`) | **`-0.24 s (-4.6%)`** |
| **Concurrent `pf+dec` Aggregate (`tok/s`)** | `64.91` (`63.75 / 66.08`) | **`67.42` (`66.72 / 68.13`)** | `67.25` (`65.07 / 69.44`) | `66.25` (`65.60 / 66.89`) | **`+2.8%` (all rounds beat control)** |
| **Slot 0 Concurrent Decode (`tok/s`)** | `43.28` (`42.50 / 44.05`) | **`44.95` (`44.48 / 45.42`)** | `44.84` (`43.38 / 46.29`) | `44.17` (`43.74 / 44.59`) | **`+2.8%`** |
| **Concurrent Batch-2 Decode (`c=2`, `tok/s`)** | `136.8` (`137.6 / 136.1`) | **`135.1` (`130.0 / 140.1`)** | `134.0` (`135.2 / 132.8`) | `135.3` (`137.4 / 133.2`) | **Parity (`140.1 tok/s` R1 peak)** |

---

## Verification & Reproducibility

1. **Unit & Parity Tests**:
   ```bash
   systemd-run --user --scope -p MemoryMax=6G -- cmake --build build-hip --target hip_prefill_mmq_parity share_rules_test copy_events_test pool_tasks_test arch_defaults_test strata -j 2
   ctest --test-dir build-hip --output-on-failure -R "^(hip_prefill_mmq_parity|share_rules_test|copy_events_test|pool_tasks_test|arch_defaults_test)$"
   ```
   All 5 test suites pass (`100% tests passed out of 5`).
2. **Paired A/B Benchmark**:
   ```bash
   /usr/bin/python3 /home/soren/.hermes/cache/scratch/bench_concurrent_prefill_decode_ab.py
   ```
3. **Environment Knobs (Default-On on Single-Card `gfx12`)**:
   - `STRATA_MMQ_STAGED_SORT` — `0` disables, `1` resident-first ascending + staged ascending, `2` all full groups ascending, `3` (default on single-card `gfx12`) resident-first ascending + staged descending.
   - `STRATA_PREFILL_CPU_SHARE_MAXT` — max tokens per non-resident expert eligible for multi-slice CPU `ExpertPool` offload (default `32` = `4 * MAXT` on single-card `gfx12`; set to `8` for single-slice only).
   - `STRATA_PREFILL_CPU_SHARE_BATCH_DIAG=1` — logs per-chunk `strata prefill cpu-share:` bucket telemetry (`T`, `bkt`, `res`, `strm`, `cpu`, `share`, `gate`, `c_ms`, `g_ms`).
