# Concurrent Decode + Co-Prefill Wave 5 Optimization & A/B Report (2026-11-08)

## Summary of Findings & Speedups

Across three isolated 4-arm paired A/B benchmarks on `gfx1201` (AMD Radeon AI PRO R9700, 32 GiB VRAM, ROCm `10.2.0a20260914`, model `qwen3.8-flash-next-unsloth-ud-iq4_xs-v31`, `--batch 2`), we identified and validated five compounding speedups for concurrent decode + co-prefill and solo `--batch 2` prefill:

1. **Decoupled Serial-Batch CPU `ExpertPool` Sharing (`STRATA_PREFILL_CPU_SHARE_BATCH=1`, `STRATA_PREFILL_CPU_SHARE_MAX=2560`)**:
   - Previously, `Prefill::set_batch_active(true)` simultaneously capped the co-prefill staging ring at 96 slots *and* unconditionally blocked CPU `ExpertPool` sharing (`g_share_blocked = true`), even during serial layer-yield `read_part` chunks where the GPU is running prompt prefill and the 15-worker CPU `ExpertPool` is idle between decode windows.
   - `BatchPrefillPolicy` (`include/strata/prefill/share_rules.hpp`) decouples `set_share_blocked` from `set_co_prefill_active`, keeping CPU share blocked only during `run_conc` / `pr_conc` concurrent-reader threads while allowing serial `read_part` chunks to use the 15-worker CPU `ExpertPool`.
   - Raising `STRATA_PREFILL_CPU_SHARE_MAX=2560` extends the staged MMQ + CPU `ExpertPool` share walk (`!stream_all`) to cover the `2048`-token co-prefill chunks (`eff_c = 2048`), cutting `ring_slots(2048)` from `96` to `STAGE_GRP = 32` (saving 64 staging slots / ~155 MiB of lent VRAM) and offloading the lowest-row-count non-resident experts to the 15-thread CPU pool.

2. **`Q5_0` Down-Projection Grouped MMQ (`-DSTRATA_ORCA_Q4KS_MMQ=ON`) & Fused Int8 WMMA (`STRATA_PF_FUSED_Q50=1`)**:
   - In `unsloth-ud-iq4_xs-v31`, **33 of 48 layers** use `GGML_TYPE_Q5_0` down projections (`d_type = 6`) with `IQ3_XXS` or `IQ4_XS` gate/up.
   - With `STRATA_ORCA_Q4KS_MMQ=OFF` and `STRATA_PF_FUSED_Q50` unset, `mmq::fits(GGML_TYPE_Q5_0, 2048)` returned `false` on all 33 layers, forcing both gate/up and down projections on 68.75% of the model back onto per-expert GPU FP16 dequantization (`iq_dequant_gu_f16` + `iq_dequant_f16`, costing 341 ms/chunk) + per-expert hipBLASLt GEMMs + 10,035 un-deduplicated per-slot event waits, and forcing `fp16_bufs(mp) = true` (borrowing 1,934 expert-cache slots instead of 1,723 slots).
   - Building with `-DSTRATA_ORCA_Q4KS_MMQ=ON` enables exact grouped `MMQ` (`group_resident` / `group_gather`) on all 33 `Q5_0`-down layers with **zero loss of expert-cache slots** (`9,936` slots in VRAM) and **bit-for-bit identical output hash (`ece1433a582a4c38`)**.

3. **Fused Streamed Copy-Event Batching (`STRATA_PREFILL_FUSED_BATCH_RELEASE=1`)**:
   - Records one tail release event per 32-expert fused batch (`copy_release_event_slot`) and maps all ring slots in `[k0, k)` to that tail event (`m.used_of[kk % m.ring] = batch_rel`), allowing `STRATA_PREFILL_COPY_WAIT_DEDUPE=1` to skip up to **96.9%** of `cudaStreamWaitEvent` calls on subsequent ring revolutions (`14,714` of `15,191` waits skipped).

4. **Grouped Staged Transfer Lookahead (`STRATA_PREFILL_STAGE_LOOKAHEAD=1`)**:
   - In the staged walk (`!stream_all`), `group_resident` allocates `STAGE_GRP = 32` staging slots for `MMQ_GROUP = 16`, so up to `STAGE_GRP - MMQ_GROUP = 16` transfers can safely remain pending ahead of the consumer (`staged_pending_cap`). Previously, `stream_ahead ? pending < STAGE : ...` hardcoded `STAGE = 8`, starving `m.copy` mid-group whenever a 16-expert MMQ group contained more than 8 non-resident experts.

5. **Pinned Loan Backup with Startup Co-Prefill Prewarm (`STRATA_LEND_PCIE_PINNED=1`)**:
   - With `STRATA_LEND_RAM_BACKUP=1` alone, `loan_host_` was allocated via `std::malloc` (unpinned), so `src.pinned(l, e)` returned `false` for all ~312 loan-backed experts on a 2,048-token co-prefill chunk, forcing `Prefill::run_impl` to create ~312 `Stager::Job` CPU memcpy tasks per chunk (`gather` 116 ms/chunk, cumulative `host staging` 922 ms).
   - Enabling `STRATA_LEND_PCIE_PINNED=1` and pre-warming the `warm_slots = part_slots(p, 2048)` (`529` tail slots) once at server startup (`kvg_start`) makes 100% of all 24,576 experts page-locked in RAM (`0` `Stager` jobs, `gather` reduced from `116 ms -> 5 ms` per 2,048-token chunk) and eliminates mid-request `cudaHostAllocMapped` reallocations.

---

## Matched 4-Arm Isolated A/B Results (`ab_concurrent_pf_dec_wave5_prewarm_20261108`)

All runs executed with `strata.service` stopped under `trap "systemctl --user start strata.service" EXIT`, identical production environment (`fork-switches.conf`), temperature 0, and 2 rounds per workload:

| Arm | Solo Refactor (`--batch 2`, 316p+192g) | Refactor Prompt Read (316 tok) | Refactor Output Hash | Concurrent `pf+dec` Agg (tok/s) | Slot 0 Decode Under Co-Prefill (tok/s) | Slot 1 4,712p+128g Wall Time (s) | 4,712-Token Co-Prefill Read (ms / tok/s) | Co-Prefill Window Rate (rows/s) | `batch2` (`c=2`) Agg (tok/s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `0_control_wave4` (`fork-17de4f2e4` prod) | 75.1 (`61.8 / 88.4`) | `1032 / 872 ms` (`306 / 362 tok/s`) | `ece1433a582a4c38` | 46.1 (`45.2 / 47.0`) | 33.7 (`32.7 / 34.7`) | 7.99 s (`8.15 / 7.83`) | `6411 / 6255 ms` (`735 / 753 tok/s`) | `45.3 rows/s` | 135.5 (`134.2 / 136.7`) |
| `1_cand_q50_mmq_max2560_la` (`Q5_0` MMQ + `CPU_SHARE_BATCH=1` + `MAX=2560` + `STAGE_LOOKAHEAD=1`) | 101.9 (**+35.7%**) (`92.0 / 111.8`) | `550 / 446 ms` (`575 / 708 tok/s`) | `ece1433a582a4c38` | 61.0 (**+32.4%**) (`59.1 / 62.8`) | 40.6 (**+20.7%**) (`39.4 / 41.9`) | 5.89 s (**-26.2%**) | `4280 / 4040 ms` (`1101 / 1166 tok/s`) | `61.1 / 67.4 rows/s` | 137.9 (**+1.8%**) (`135.4 / 140.3`) |
| **`2_cand_q50_mmq_max2560_pin_la` (`+ STRATA_LEND_PCIE_PINNED=1` prewarmed, exact MMQ)** | **104.9 (+39.7%)** (`96.3 / 113.5`) | **`456 / 413 ms` (`693 / 764 tok/s`, +119%)** | **`ece1433a582a4c38`** | **65.4 (+42.1%)** (`63.9 / 67.0`) | **43.6 (+29.5%)** (`42.6 / 44.7`) | **5.39 s (-32.5% / -2.60 s)** | **`3613 / 3591 ms` (`1304 / 1312 tok/s`, +76%)** | **`66.3 / 72.2 rows/s` (+53%)** | **136.4 (+0.7%)** (`139.2 / 133.6`) |
| `3_cand_q50_fused_pin_la_chunk8k` (`+ STRATA_PF_FUSED_Q50=1` + `STRATA_GFX12_CHUNK=0`) | 81.5 (`67.8 / 95.2`) | `629 / 427 ms` (`503 / 741 tok/s`) | `ece1433a582a4c38` | 64.5 (**+40.0%**) (`63.6 / 65.4`) | 43.0 (**+27.6%**) (`42.4 / 43.6`) | **5.28 s (-33.9%)** | `3564 / 3622 ms` (`1322 / 1301 tok/s`) | `70.5 rows/s` | 132.8 (`132.1 / 133.5`) |

### Per-Chunk GPU Timeline Comparison (`2048`-Token Co-Prefill Piece)

- **`0_control_wave4` (`fork-17de4f2e4`)**:
  `2048 tokens, GPU timeline 1693 ms: wait copy 165 ms (9.7%), dequant 341 ms (20.1%), gemm gate/up 420 ms (24.8%), gemm down 243 ms (14.3%)`
- **`2_cand_q50_mmq_max2560_pin_la`**:
  `2048 tokens, GPU timeline 983 ms (-41.9%): gather 5 ms (0.5%), wait copy 34 ms (3.5%), dequant 123 ms (12.5%), gemm gate/up 208 ms (21.2%), gemm down 123 ms (12.5%)`
- **`609`-Token Tail Piece**:
  Dropped from `1,316 ms` in `0_control_wave4` to `642 ms` (`-51.2%`) in `2_cand_q50_mmq_max2560_pin_la`.

---

## Verification & Parity Tests Passed

- `ctest --test-dir build-hip --output-on-failure -R '^(copy_events_test|share_rules_test|dflash2_ring_test|verify_batch_parity)$'` -> **4/4 PASSED**
- `./build-hip/hip_prefill_mmq_parity` -> **PASSED** (`new-set-GU-Q5_0: rel_L2=0.000516`, `new-set-down-Q5_0: rel_L2=0.000491`, all `all_finite=1`)
- `./build-hip/hip_prefill_fused_iq --only=Q5_0 --chunks=2048` -> **PASSED** (`IQ3_XXS / Q5_0: rel_l2=0.0086`, `IQ4_XS / Q5_0: rel_l2=0.0073`, 3.92x–4.15x faster than reference at `T=2048`)
- Solo Refactor (`--batch 2`) output SHA-256 prefix: `ece1433a582a4c38` across all arms.

## Reproduction

```bash
cmake -B build-hip -DSTRATA_ORCA_Q4KS_MMQ=ON
systemd-run --user --scope -p MemoryMax=6G -- cmake --build build-hip --target copy_events_test share_rules_test hip_prefill_mmq_parity hip_prefill_fused_iq strata -j 2
ctest --test-dir build-hip --output-on-failure -R '^(copy_events_test|share_rules_test)$'
```
