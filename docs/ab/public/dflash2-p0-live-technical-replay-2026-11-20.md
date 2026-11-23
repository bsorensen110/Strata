# DFlash2 P0 — live technical corpus replay (2026-11-20)

## Scope and disposition

This is a **live linear-DFlash2 measurement**, not a candidate-tree test. On the six-request pilot corpus the numerical mean positions 2–7 is *steep*, but the prompts all reuse one 222k-token archived codebase and share the same engineering task family. Treat this as **pilot evidence, not independent-workload clearance** for committing to a tree. The minimum-per-position sample floor passes mechanically but does not make repeated questions over the same source corpus independent examples of coding/system-management traffic. Variation between requests remains material, and later-position counts per request are small. Run a confirmatory sample using at least five **genuinely different long technical task corpora** before making the roadmap decision. Do not extrapolate this six-prompt corpus to production's broad traffic mix.

## Measurement setup

- Existing production service remained active; no restart, configuration edit or tree code was run. All completions were submitted one at a time through its loopback API.
- The service, engine, and profile were pinned during the pilot. Operational identifiers are private.
- Target: Qwen3.8 Flash Next UD-IQ4_XS v31 (`qwen3.8-flash-next-unsloth-ud-iq4_xs-v31`). Drafter: `qwen3.8-flash-next-dflash2.q4_0-selq4.pack`, `--spec 8`, `--draft-conf 0.35`, no MTP fallback; solo mode, `--lookup-chain 6`, `--lookup-max-t 16`. Existing production settings include `STRATA_DRAFT_TAIL_SKIP=6`, `STRATA_PCIE_FETCH_OVERLAP=0`. `STRATA_DF2_ACCEPT_HIST=1`, capped `STRATA_DF2_CAND_DUMP` were verified from the live engine child before replay.
- Six cold independent HTTP requests (`reused=0`), actual target prompt depths 221,897–222,006, each generated all 1,024 requested tokens (`finish_reason=length`). One short prompt family generated normally but was explicitly excluded from P0.
- Each input was first measured with `/v1/messages/count_tokens`. The exact HTTP completion `usage.prompt_tokens` and `completion_tokens`, same-request `[accept-hist]`, and matching DFlash2 candidate rows were cross-checked. Unmatched or below-band attempts do not appear in the report. No #879/NaN appeared during these six replays.

## Source-separated DFlash2 acceptance

Conditional DFlash2 accepted/reached, pooled over the six requests. 95% Wilson intervals are descriptive proposal-level intervals; correlated windows and prompts from one source corpus limit their interpretation.

| position | accepted/reached | rate | 95% Wilson CI |
|---|---:|---:|---:|
| 1 | 1460/1942 | 75.18% | 73.21–77.05% |
| 2 | 929/1263 | 73.55% | 71.05–75.91% |
| 3 | 609/799 | 76.22% | 73.15–79.04% |
| 4 | 397/525 | 75.62% | 71.77–79.10% |
| 5 | 266/335 | 79.40% | 74.75–83.39% |
| 6 | 193/231 | 83.55% | 78.23–87.77% |
| 7 | 124/157 | 78.98% | 71.95–84.62% |

Mean positions 2–7: **77.89%**. The scorer labels this pilot curve *steep* (`<80%`). Every position 2–7 exceeds 100 reached proposals pooled across requests; minimum is 157 at position 7. Task-source variation remains material.

Other proposal sources remain separate: **suffix/lookup 33/97**, **appended chain 6/74**. Neither contributes to the DFlash2 conditional curve.

## Candidate-rank coverage — diagnostic, not realized tree acceptance

For the 1,274 rows where the target's first-rejected token could be compared at a rejection position strictly inside the DFlash2 prefix, the v1 `draft-head logit` dump contains it in the first *k* raw-head candidates:

- raw Recall@1: **79/1,274 = 6.20%**
- raw Recall@2: **468/1,274 = 36.73%**
- raw Recall@16: **979/1,274 = 76.84%**

These ranks are not the predecessor-conditioned selector order. Later-row candidates are evaluated against the already-proposed path. This measurement is not accepted-tree yield, a speed estimate, or proof that extra candidates form a target-consistent path. It is enough to justify *investigating* a realistic, budgeted selector/tree bound after the traffic-representativeness check; it does not justify production integration or promotion.

## Artifacts and reproducibility

Raw campaign evidence, source excerpts, request data, logs, candidate traces and private replay/scoring scripts are not published. The recorded aggregate was checked against each sample's IDs, depth, completion count and recall.

Protocol: production linear service, serial HTTP requests, `temperature=0`, `max_tokens=1024`, thinking disabled; measure exact input tokens, require actual completion in 150,000–250,000-token band and positive generation; match each request's histogram and candidate dump by engine request ID; exclude lookup/chain rejections from candidate recall; do not pool different configurations or depths.

To collect the next confirmatory set, first replace the archived corpus with five truly different coding/system-management tasks or technical-history datasets while preserving the same pinned service and gate. Continue until ≥5 distinct completions and ≥100 reaches at each position 2–7; then use the identical scorer. A confirmatory mean ≥84% argues against a tree, <80% supports budgeting a branch-aware feasibility investigation, and 80–84% or a narrow/non-independent workload leaves P0 unresolved. Any GPU tree verifier still needs independent branch-aware GDN/QSA/PLE state parity and matched speed tests.
