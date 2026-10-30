# Launch-reduction A/B (2026-10-30): which parts stay on for gfx12

Branch `perf/launch-reduction` (3 commits, merged into `integ/community-2026-10`):

- `585ff5f5` quantize + interleave in one launch for 2-4 column inputs (`STRATA_QIL`, default on for gfx12).
- `055025ed` the PCIe-share wait folded into its staging copy (`STRATA_FUSE_PCIE_WAIT`).
- `d9fcdd38` `quantize_il_parity` test (fused quantize+interleave vs the pair, byte for byte).

## Method

- Config: production-shaped solo-batch config, `parallel: 4` (`--batch 4`), MTP, no DFlash2, no group draft,
  `STRATA_BATCH_DRAFT_TIMING=1` in every arm (same across arms). Fresh conversation-cache directory per launch.
- Bench: `tools/ab/batch_burst_bench.py`, 4 and 8 concurrent requests, 3 rounds of 512 max tokens per level. Round 0
  of each level is warmup and is dropped; the mean is over 4 rounds per arm (2 launches x 2 rounds kept per level).
- Arms (each launched twice, pass 2 in reversed order to cancel drift):
  - `base`: integ before the merge (`177038db`-era binary; no launch-reduction code).
  - `on`: merged binary, defaults (both on).
  - `qil_off`: `STRATA_QIL=0`.
  - `pcie_off`: `STRATA_FUSE_PCIE_WAIT=0`.

## Results (aggregate decode tok/s, mean of 4 non-warmup rounds per arm)

| arm | 4 concurrent | vs base | 8 concurrent | vs base |
|---|---:|---:|---:|---:|
| base | 99.9 (98.2-101.8) | - | 113.0 (111.2-114.9) | - |
| on (both on) | 95.5 (92.7-98.1) | -4.4% | 109.5 (108.2-110.4) | -3.1% |
| qil_off (fold on) | 96.1 (94.5-97.7) | -3.8% | 110.7 (109.1-112.2) | -2.1% |
| pcie_off (QIL on) | 99.2 (96.8-101.2) | -0.7% | 114.0 (112.1-115.9) | +0.9% |

Per pass (4 concurrent / 8 concurrent): base 99.8 / 113.6 then 100.0 / 112.5; on 95.7 / 110.2 then 95.4 / 108.8;
qil_off 96.1 / 111.2 then 96.1 / 110.2; pcie_off 99.0 / 115.0 then 99.5 / 113.0.

## Reading

- The PCIe-wait fold is the regression. With the fold off, speed is back to base (within noise) even with the
  quantize change on. With the fold on and quantize off, it is still about 4% down at 4 concurrent.
- Quantize + interleave is neutral in speed (`pcie_off` vs `base`). It is also the only part with a byte-parity
  test (45 of 45 cases identical; re-run on the merged tree, ALL PASS).
- Acceptance is flat across arms (about 71-72% MTP proposals accepted in the logs).

Committed default check (new build, no env overrides, one launch): 4 concurrent 99.4 (98.6, 100.3), 8 concurrent
113.6 (113.7, 113.5). Matches the baseline and the fold-off arm.

## Bit parity

- Quantize + interleave: verified by `quantize_il_parity` (byte for byte).
- PCIe-wait fold: NOT verified bit-for-bit. Whole-reply comparison cannot be used, because replies already differ
  across fresh launches (expert placement is timing-dependent; see `BATCH-WINDOW-SYNC-2026-10-30.md`).

## Decision

- `STRATA_FUSE_PCIE_WAIT` default is now OFF on every backend. `STRATA_FUSE_PCIE_WAIT=1` forces it on (kept for
  future A/B; it is not a production setting).
- `STRATA_QIL` stays default on for gfx12 (neutral speed, parity-tested, fewer launches).
- The fold stays in the tree, opt-in, because it is what the A/B measured and a future kernel change may flip the
  result.

## Repro

```
# arms: baseline binary = integ build before the merge; the others = merged build with the env listed above
ARM=<arm> bash run_lpt.sh     # launches strata with the arm's cand.json, runs
                              # batch_burst_bench.py --conc 4 and --conc 8, 3 rounds x 512 tokens, then stops it
```
