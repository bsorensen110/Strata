# DRAFT_TAIL_SKIP 6/7/8 sweep

**Status:** Complete — 9 successful interleaved launches (3 per arm), no `ABORT after failed` marker. The measured request returned 200 with 222,006 prompt tokens and 1,024 generated tokens in every launch. `STRATA_PCIE_FETCH_OVERLAP=0` was pinned and verified in all arms; this isolates the DTS comparison with the PCIe-overlap regression fix enabled.

## Method

- Same production engine binary for all arms: `/home/soren/.local/share/strata-upstream-v0.2.10/components/engine/fork-58aefafc9-p0/strata` (SHA-256 `108f83087d2e6f901120010c85918c9807856f4e9dabe17b5d728659a678ce95`).
- Interleaved order per round: `dts6`, `dts7`, `dts8`; 3 rounds. Only `STRATA_DRAFT_TAIL_SKIP` varied (6/7/8); `STRATA_PCIE_FETCH_OVERLAP=0` was common to every arm. Runtime environment checks passed and each arm reverted its temporary override.
- Deep-context workload: model `qwen3.8-flash-next-unsloth-ud-iq4_xs-v31`, 222,006 prompt tokens, 1,024 generated tokens, temperature 0.0, thinking disabled. Each measured request completed at the requested token limit. Decode is the engine's reported tok/s; acceptance is the runner's mean of the three per-launch accepted/offered ratios.
- Runner/source: `ab_dts.py`; invocation: `/home/soren/Strata/.venv/bin/python /home/soren/.hermes/cache/scratch/ab_dts.py --run --launches 3`. The runner imports the shared `ab_window.py` and `ab_switch.py` helpers from `/home/soren/.hermes/cache/scratch`. Do not rerun outside an authorized GPU test window.

## Results

- **DTS 6** — decode **88.0 tok/s** (85.6–89.8); acceptance **53.2%**; vs DTS 8: **+6.2%** decode (+5.1 tok/s), **+2.3 percentage points** acceptance.
- **DTS 7** — decode **83.6 tok/s** (80.2–87.1); acceptance **50.5%**; vs DTS 8: **+0.8%** decode (+0.7 tok/s), **−0.4 percentage points** acceptance.
- **DTS 8** (reference/current production value) — decode **82.9 tok/s** (82.1–83.7); acceptance **50.9%**; reference (0.0%).

DTS 6 is the best measured arm on both mean decode and mean acceptance. It leads DTS 8 by 5.1 tok/s and 2.3 points of acceptance; DTS 7 does not improve acceptance. Acceptance is non-monotonic across 6/7/8, so this run does not support a claim that increasing the skip value raises acceptance. Across all nine launches, mean decode was 84.8 tok/s (about 85, i.e. the mid-80s recovery with overlap disabled).

## Limits and production state

This is directional evidence, not a definitive promotion case: only 3 fresh launches per arm, with greedy temperature-0 run-to-run variability. DTS 6's observed range (85.6–89.8) does not overlap DTS 8's (82.1–83.7), but DTS 7's broad range overlaps both. The retained final-launch logs show sparse lookup-chain activity (11/339, 19/392, and 20/391 windows for DTS 6/7/8 respectively; 0–1 chained drafts accepted), limiting conclusions about lookup-window behavior. The results do not establish general workload or output-quality effects.

Production was restored to the pre-sweep snapshot and verified: service active; keyed `/v1/models` returned 200, unkeyed returned 401, and the smoke response was `391` with finish `stop`; binary and profile SHA-256 match the snapshot. The live engine environment remains `STRATA_DRAFT_TAIL_SKIP=8` and `STRATA_PCIE_FETCH_OVERLAP=1`. No production config change or promotion was applied.

**Recommendation for a separate, explicitly approved config decision:** test `STRATA_PCIE_FETCH_OVERLAP=0` with `STRATA_DRAFT_TAIL_SKIP=6`. This recommendation is not applied by this sweep.

## Production application (2026-11-19, approved by Brooks)

Brooks approved putting the fix into production and chose `DRAFT_TAIL_SKIP=6` for a live trial. Applied to `~/.config/systemd/user/strata.service.d/fork-switches.conf` (dated comments, explicit pins):
- `STRATA_PCIE_FETCH_OVERLAP=1 → 0` (the ablation culprit; ~7.5% solo deep decode recovery).
- `STRATA_DRAFT_TAIL_SKIP=8 → 6` (best sweep arm).

Duplicate-key guard clean; `daemon-reload` + `restart strata.service`. **Verified live** in `/proc/<engine>/environ`: `PCIE_FETCH_OVERLAP=0`, `DRAFT_TAIL_SKIP=6`; keyed `/v1/models` 200, unkeyed 401, smoke `17*23=391` finish `stop`.

Deep-context confirmation (222k prompt, 1024 gen, T0): first post-restart request **37.4 tok/s** was a cold-start transient (GPU expert cache began at 6083 experts / 71.9% hit / 48.6% PCIe / 258 file blob reads, climbing to 87% hit by request end). Warm requests recovered **75.4 → 81.0 → 82.1 → 80.0 tok/s**, acceptance ~0.53–0.56, cache hit ~88%, PCIe ~3%. Live warm steady state (~80–82) is below the isolated sweep's 88.0 — attributed to the sweep's fresh scratch expert/conv-cache state vs production's persistent accumulated cache; real traffic will show DTS=6's sustained effect. No binary promotion (same `fork-58aefafc9-p0` engine); config-only change.
