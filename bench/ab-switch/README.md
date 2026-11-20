# Per-switch solo decode ablation (deep-context) — 2026-11-19

Isolates the Nov-19 `perf/verify-wide-all` optimization **env switches** on the **same production
binary** `components/engine/fork-58aefafc9-p0/strata` (sha256 `108f8308…`). Goal: find the residual
deep-context solo decode regression after the 3 already-found culprits (`MOE_LDS_NT8`,
`MOE_GRID_CAP`, `VERIFY_WMMA_DENSE`) were reverted to 0.

Reference: `docs/ab/decode-dropout-diagnosis-2026-11-19.md` (regression 87.5 → ~73 tok/s median,
7–12% per-step verify slowdown + acceptance drop).

## Method

- **Control** = current prod env (3 found culprits at 0; all other Nov-19 switches at promoted values).
- **12 ablation arms**, each reverting ONE solo-active suspect to its pre-promotion value (diagnosis §5).
- The **7 batch-only/no-op switches excluded** (inert at `parallel=1`): `VERIFY_WMMA_MOE` (stub),
  `BATCH_SPLIT`, `BATCH_DF2_ADAPTIVE_T`, `BATCH_DF2_TARGET_ROWS`, `BATCH_ROUTE_AFFINITY`,
  `BATCH_PCIE_AUTO`, `BATCH_PCIE_BALANCE`.
- **Deep-context workload required**: ~222k-token prompt (`p0_deep_request.json`), 1024 generated, T0.
  Short prompts never fire the wide-window/lookup paths, so they cannot reveal this regression.
- n=3 interleaved passes (control + 12 arms ×3 = 39 launches). Every launch verified: exe matches prod
  binary, `--draft-model …q4_0-selq4.pack`, solo, ablated switch value confirmed in `/proc/<engine>/environ`
  (`revert_ok`), `check_engine_env.py` N/N (rc 0). **39/39 passed.**
- Decode tok/s from `N generated in X ms (Y tok/s)`; acceptance from `drafts accepted A of B`.

## Results (n=3/arm)

- **control: 80.2** (75.6–84.7), accept 0.544 — ref
- **pcie_fetch_overlap (`PCIE_FETCH_OVERLAP 1→0`): 86.2** (84.1–87.9), accept 0.521, **+7.5%** ← CULPRIT
- mmvq_il_narrow_acc (`1→0`): 79.8 (78.5–81.8), accept 0.557, −0.5% (flat)
- cpu_expert_no_split (`1→0`): 77.1 (74.8–79.0), −3.9%
- bucket_nosplit (`NOSPLIT 1→0, CHUNK 8→4`): 76.7 (72.5–78.9), −4.4%
- qsa_scores_16 (`1→0`): 76.4 (75.7–77.1), −4.7%
- pcie_group_rows (`32→16`): 76.2 (74.6–78.6), −5.0%
- gdn_groups_split (`1→0`): 76.2 (74.8–78.5), −5.0%
- gdn_ab_wide (`1→0`): 76.2 (69.3–81.6), −5.0%
- staging_blobs (`32→16`): 75.7 (72.6–78.7), −5.6%
- cpu_reduce_miss (`1→0`): 75.6 (74.0–77.1), −5.7%
- hc_q8_16 (`1→0`): 75.6 (74.9–76.9), −5.7%
- draft_tail_skip (`8→0`): 74.1 (72.5–75.2), accept 0.510, −7.6%

## Verdict

**`STRATA_PCIE_FETCH_OVERLAP=1` is the residual decode culprit.** Reverting it to 0 recovers decode
from 80.2 → **86.2 tok/s (+7.5%)**, essentially the pre-promotion ~87.5 level. Mechanism (diagnosis §5):
`verify.cpp:978-1010` overlaps PCIe expert fetch with verify compute; in solo deep-context this overlap
costs per-step throughput. The effect is consistent across all 3 launches (84.1/86.x/87.9, all above
control mean); ranges nearly disjoint (control max 84.7 vs arm min 84.1, ~0.6 touch) — strong signal,
borderline on the strict no-overlap separability rule.

**Every other revert is flat or WORSE than control** (−0.5% to −7.6%): those switches are *beneficial*
in the current config, not culprits. In particular:
- `draft_tail_skip→0` **lowers** acceptance (0.510 vs 0.544) and decode (−7.6%) → **refutes** the
  diagnosis's secondary suspect #4 (`DRAFT_TAIL_SKIP`); `=8` is helping here.
- `mmvq_il_narrow_acc` flat at n=3 (an n=2 +3% hint was noise).

## Caveats

- The single deep prompt fired **few prompt-lookup windows** (`lookup 0/2` every control launch), so the
  wide-window switches (`hc_q8_16`, `qsa_scores_16`, `gdn_*`, `mmvq_il_narrow_acc`) were lightly
  exercised; their flat/negative results are valid for this workload but a repetitive deep context that
  fires many wide lookup windows could differ. The every-window PCIe/CPU/MoE switches are well exercised.
- Control variance is wide (75.6–84.7); the culprit arm is tighter and consistently high.
- Greedy whole-reply non-determinism (expert placement) — parity not assessed by reply hash.
- Compound env (other switches at promoted values); this attributes the single-switch revert, not interactions.

## Recommended fix (NOT applied — user decision)

Set `Environment=STRATA_PCIE_FETCH_OVERLAP=0` in `~/.config/systemd/user/strata.service.d/fork-switches.conf`
(keep an explicit `=0` pin with a dated comment so a future upstream default flip can't silently re-enable it),
`daemon-reload`, restart, verify live env + keyed smoke. Expected: deep-context solo decode ~86 tok/s.

## Production state

Restored to previous binary/profile (unchanged), verified binary+profile sha match, selq4 pack, solo,
keyed 200 / unkeyed 401, smoke `17*23=391` stop. **No promotion / no config change applied.**

## Reproduce

```
python3 ab_switch.py --plan
python3 ab_switch.py --run --launches 3            # all 13 arms
python3 ab_switch.py --run --arms control,pcie_fetch_overlap --launches 5   # confirm culprit
```
Raw: `records.jsonl`; aggregate: `summary.json`; arm configs/env: `<arm>.json/.env` (0600).
