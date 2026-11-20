# Spec-decode audit: draft-tree verification for DFlash2 + engine feature gaps (2026-11-19)

Inputs: Copilot read-only census (copilot_df2_gap_census.md, 12 gaps, key claims
source-verified by Hermes), tree-method research (tree_spec_research.md),
engine feature-gap research (engine_spec_gap_research.md), Context7 vLLM docs,
Hermes source verification of dflash2.hpp/.cpp/.cu/generate.cpp.
Status codes: [V]=verified in our source by Hermes this session, [M]=measured
by us previously (docs/ab/perf-verify-wide-results-2026-11-19.md), [C]=cited
external source, [I]=inference/unmeasured.

## 1. Geometry truth (anchors everything) [V]
- DF2 block: kRows=8 (anchor + 7 drafts), kDrafts=7 cap (dflash2.hpp:41-42,
  dflash2.cpp:1160-1177). kDf2TopK=16 candidates per position (kernels/dflash2.hpp:21).
- Selector walk kernel already stages rows*16*16 candidate-transition scores in LDS
  and has a BEAM template (dflash2.cu:1461-1475); topk_part/merge_rank kernels
  (1346,1371) exist. Branching needs NO retrain, NO extra draft forward:
  candidates + transition table are computed every block today, we only keep 1 path.
- Prod launch: --spec 8 --draft-conf 0.35 --lookup-max-t 16 --lookup-sizes 10,12,14,16
  --lookup-chain 6. Wide windows = DF2's <=7 proposals + appended lookup-chain tokens.
- Published DFlash2 (arXiv:2602.06036, inco.ai/blog/dflash2 [C]): conditional
  acceptance 88.3%->86.5% across positions 1..15, mean tau 4.80 on Qwen3.8-27B.
  Our blended "50-53% accepted of offered" counts DF2+lookup+suffix rows together
  (census gap 12 [V]) on an IQ4_XS UD-quant target at 150k-250k depth; NOT
  comparable to the published per-position conditional number.

## 2. Confirmed retrofit gaps worth acting (from census, dedup) [V unless noted]
1. Solo decode does NOT overlap DF2 context projection into verify tail layers:
   set_taps_ready installed only for admission / merged T>8 / final window
   (generate.cpp:22399-22413). Precompute scratch + subrange validation already
   safe (dflash2.cpp:2047-2111). Gate change, no new kernels. Gain [I]: hides
   8-row projection under verify tail; bench before enabling.
2. No cost-aware width chooser for DF2 (MTP has one; DF2 correctly bypasses it,
   nothing replaced it; window_rule.hpp:30-32,62-75). [I] gain.
3. --draft-cap is a global ceiling, not a DF2 knob; can delete wide lookup sizes
   (generate.cpp:7935-7939). Config-hygiene trap.
4. Accept telemetry source-blind; Prometheus labeled "MTP" (generate.cpp:23191,
   serve/prometheus.py:91-94). Blocks every accept-rate decision we make.
5. DF2 GPU parity tools EXCLUDE_FROM_ALL, parity cosine-only, window_rule_test/
   draft_vocab_test unregistered in CTest (census 6).
6. draft-fallback none: mid-request DF2 failure exits the request (policy, not bug).

## 3. TREE VERIFY (the "tree spanning" question)
Method lineage [C]: SpecInfer (token trees + tree attention), Medusa top-k trees,
EAGLE-2 dynamic trees, Sequoia, DDTree (2604.12989), CaDDTree, D3Tree/TreeFlash/
GRAFT/PCTree/DARTree. Standard mechanics: ancestor-masked causal attention, rows
in topological (ancestor-before-descendant) order, rejected branches' KV simply
not committed, accept = longest accepted root-leaf path + target bonus token.

Engine status today [C, spot-checked vLLM via Context7]:
- SGLang: ONLY mainstream engine with draft-tree verify (EAGLE-2/3 dynamic trees,
  radix-compatible, fixed verify capacity padded; DFLASH algorithm since Aug 2026;
  eagle-harness even has GDN tree verify).
- vLLM: chain-only (Medusa REMOVED #61701); dflash/LiLiCorr method w/ 16-row
  trained blocks, per-step Adaptive Verification cost model, batch-keyed dynamic K.
- llama.cpp: chain-only; tree prototype abandoned 2023 ("no significant
  improvement" #3137); ships draft-dflash2 via #32016; --spec-draft-p-min +8.8%
  net on Qwen3-Next MTP (91->130 tok/s M5 Max).
- TRT-LLM: EAGLE-3/MTP-Eagle dynamic-tree mode + DFlash; Medusa legacy.
=> differentiation window open: nobody chains tree-verify a *block* drafter
   in production; SGLang trees are EAGLE-AR.
Note [C]: 2023 llama.cpp abandonment predates block drafters entirely — with AR
drafters each branch costs draft forwards; with DF2 branches are already computed.

Same-window evidence (our regime, rows<=16 already paid) [C]: D3Tree 1.36-1.59x
over DFlash chain same drafts; DDTree beats 16-chain at B=16; MFD measures
DFlash2's own tree +5-15% over selector chain; inco oracle tau 4.27->6.79 inside
existing top-16 lists (chain takes 4.61). Big-paper tau 6-7.5 numbers are at
64-512 rows, NOT our budget.

Our-engine sizing [I, DP over 15 nodes anchored to measured 2.5 accepted/step,
q~0.72]: rank-2 forks recover 25-40% of residual miss; accept length +13-45%,
solo tok/s +15-30% at 256k; uniform binary depth-3 tree LOSES to chain.

Recommended first shape: front-heavy spine+fork in 16 rows — greedy spine depth
8-10, rank-2 selector forks at depths 1-5 (rank-3 at depth 1 if margin),
spine-first DFS row order (common accept = contiguous prefix, zero gather),
fixed 16-row captured graph + per-step dynamic ancestor-list mask buffer
(EAGLE-2/SGLang pattern).

THE hard problem: GDN linear-attention layers (3/4 of Qwen3.8 layers) have
recurrent state with NO branch semantics — sibling rows at one position need
different state continuations. Options: (a) per-row state snapshots in
topological order with ancestor gather (<=15 sequential updates/step, small
state; llama.cpp Metal MTP snapshot slots precedent); (b) recompute accepted
path's GDN steps after verify (path is short, GDN steps cheap). Softmax layers
just take the ancestor mask; MoE per-step routed tokens unchanged (<=16 rows).
Checkpoints/lookup-chain/conversation-cache untouched at the accept-walk
boundary. This is make-or-break and needs a design doc + spike before any prod
work (spike skill).

## 4. PRIORITIZED PLAN (solo deep-context)
**2026-11-19 UPDATE: superseded by block16-lilicorr-decision-2026-11-19.md —
tree priority (P3) is now GATED on the P0 acceptance-curve measurement; flat
curve => retrain b16 instead of building trees. Original ordering below.**
P0 (cheap, unblocks everything): accept telemetry split by source + per-position
   conditional-acceptance histograms + selector rank-2/3 Recall@k offline probe
   at 150k+ depth. Answers "is 50-53% workload or config loss" and gates P3.
P1: solo verify→DF2 projection overlap gate (census fix 1) + DF2 clamp to real
   7 proposals + decouple width controls (census 2/3). No new kernels. Bench A/B.
P2: dynamic lookup width: SGLang-style EMA tiers over existing 10/12/14/16
   graphs with hysteresis; per-step p-min on MTP fallback path. +5-15% MTP
   [C-analog], +2-8% DF2 [I].
P3: TREE VERIFY per §3 (gated on P0 recall@k + GDN state design spike).
   Expected +15-30% solo. Biggest differentiator; SGLang's DFLASH verify is the
   existence proof that block+tree coexists with radix-style KV.
P4: suffix/ngram co-drafter over conversation cache (vLLM suffix decoding,
   llama.cpp #22838 chaining): zero-forward drafts, additive with P3, strong
   for agent traffic re-emitting file/boilerplate spans at 200k depth.
P5: probabilistic draft sampling + rejection at T>0 — distribution-correctness
   first (we are lossless-greedy today [V via --draft-conf greedy walk]), accept
   length secondary. Check before serving T>0 broadly.
Deferred: DSpark-style trained confidence head (forks us off public checkpoint
   line), FLy lossy verify (breaks lossless story), draft-lm_head reduction
   (<=3% here), heavier drafter/target overlap (bandwidth-bound => thin).

## 5. External claims NOT independently verified here [C]
arXiv IDs, PR numbers, blog numbers come from the research agents' web passes;
headline ones (vLLM dflash method, dynamic SD, ngram configs) were cross-checked
against vLLM docs via Context7 directly. Spot-check any number before quoting
upstream (Niko1221/Strata) in a PR.
