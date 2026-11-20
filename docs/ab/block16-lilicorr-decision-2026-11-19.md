# 8->16-row DFlash2 blocks & LiLiCorr: decision research (2026-11-19)

Supersedes §3/§4 P3 ordering of spec-audit-tree-verify-2026-11-19.md (tree
priority DEMOTED pending P0; see §4 below). Inputs: two research agents
(lilicorr_dflash16_research.md, block16_vs_tree_analysis.md), Hermes
first-hand verification of incoai HF card + AMD Primus/SpecForge ROCm docs +
our pack cfg. Codes: [V]verified by Hermes, [C]cited, [M]ours, [I]inferred.

## 1. What exists (verified)
- Our pack IS the public checkpoint: cfg source
  incoai/Qwen3.8-Flash-Next-DFlash2_a22a1744, cfg block 8, top_k 16,
  conv_kernel 2/conv_group 16, 5 layers, taps 5/15/25/35/45, rank 256 [V].
  Block size rides in the pack index; the ENGINE hardcodes kDf2Rows=8
  (99 sites; LDS table rows*16*16 -> 8KB@b8, 16KB@b16; exec_ arrays,
  kBatchMaxRows, ring slack all sized off it) [V].
- NO b16 drafter for Qwen3.8-Flash-Next anywhere (incoai b8 only + 5
  community quants; nazeshinjite Q4_0 86.4% accept at 131k ctx).
  NO LiLiCorr for the family (community b8 epoch-0 for Qwen3.8-27B only,
  val EAL 3.612, not serving-validated). Unsloth ships no drafter and no
  plan; they consume third-party DFlash2 [C].
- incoai card [V direct read]: official eval ran at BLOCK 4 on GB300
  (QSA verify-width limit!), tau 2.83-3.26, 1.63-1.82x AR @c1, FP8 target.
  => our engine already runs this drafter WIDER (7 DF2 proposals inside
  10-16-row windows) than the vendors' public stack. Adoption risk LOW:
  our fork computes the 4-way hyper-connection stream mean the draft was
  trained on (llama.cpp #27342 cannot; we prod-run it fine [V]) — and the
  repo SHIPS the Flash-Next target-feature capture patch for SGLang
  (patches/qwen38-dflash2-capture.patch) [V] = training-data path for our
  exact model is public.
- LiLiCorr (NVIDIA, arXiv:2608.20530) [C]: DFlash backbone CO-TRAINED with
  a 2-layer 32-37M transformer candidate-lattice reranker (K=8/slot,
  unit-norm in/out vectors, cosine pair scores, 904 scores @b16, parameter-
  free greedy walk; Viterbi is 2.6% WORSE under prefix acceptance).
  +5-9% tau / +4-13% throughput over DFlash2 reimpl; head ~3% of block
  latency. Slot/rank embeddings block-width-TIED: shorter drafts = prefix
  reuse [C, vLLM docs V]. NVIDIA training code NOT released; community
  recipe: vllm-project/speculators lilicorr branch (Qwen3.8-27b example).
- DFlash paper block-size ablation (only one) [C]: b16-trained > b8 by
  tau +1.12/+0.68/+0.21 (Math/HumanEval/MT-Bench); b8-trained at b16
  DEGRADES (hence llama.cpp n-max clamp). A b16 checkpoint serves any
  width 1..15.
- Training pipeline on AMD exists [V direct read]: AMD Primus SpecForge
  recipes (capture + offline/online train, DFlash strategy, b16 filter
  example, AMD ROCm tutorial, MI35X image). Cost estimate for our own b16
  DFlash2 [I, anchored on published runs]: 600-1700 H100-h ~$1.2k-4k spot
  (capture dominates for a 125B-A6B target: regenerate 150-400 GPU-h,
  hidden-state capture 100-300, train 300-1000 on 4-8 GPUs); warm-start
  from incoai b8 claimed supported (SpecForge draft_checkpoint_path —
  VERIFY IN SPECFORGE SOURCE BEFORE SPENDING). Teacher precision: papers
  use BF16/FP8; FP8-target training + IQ4 serving accepted by community
  cards; IQ4-as-teacher untested by anyone.

## 2. Expected gain, computed two ways [C, formula validated to +-0.01 vs inco's published acceptance lengths]
Chain E[tau] = 1 + sum_i prod_{j<=i} p_j.
- On inco's FLAT curve (88.3%->86.5% over pos 14): b8(7 drafts) tau 5.02;
  b12 tau 5.95; b16 tau 6.47 (+29%). Verify rows 9-15 already exist in our
  window (currently filled by weaker lookup-chain tokens) -> marginal verify
  cost ~1-2% cycle => net ~+25-28% decode IF our curve is flat.
- On a STEEP curve (p~0.72, closer to our blended 2.5-accepted/step):
  b16 chain adds only ~+0.2 tau (positions 8-15 = 0.072..0.008 each) but
  TREE BRANCHING wins: fork break-even is marginal fork-hit r > P_15;
  flat curve P_15=0.102 (unbeatable: Recall@16-Recall@1 caps rank-2..16
  mass at ~9-14 points total), steep P_15~0.01 (beatable by 10x).
- Tree-over-8-block at 16 rows: flat regime +0.17..0.24 tau (DO NOT BUILD);
  steep regime plausibly +1..1.8 tau. DDTree's own optimality heap on a flat
  marginal curve returns the pure chain — trees pay only on weak/steep
  drafters (D3Tree agrees). Our prior +13-45% tree estimate rested on a
  25-40% fork-hit rate that is only defensible if OUR curve is steep.

## 3. The decisive unknown = exactly P0
Per-position conditional acceptance histogram for DF2 drafts at our operating
point (solo, 150k-250k ctx, Hermes traffic, --draft-conf 0.35, IQ4 target),
split from lookup/suffix sources (census gap 4 fix), + selector Recall@1/2/16
probes. It chooses the entire plan:
- flat (>=84% tail): retrain b16 => +25-28%. Tree: skip.
- steep (~72%): b16 adds ~+4%; TREE ON EXISTING b8 => the big win, zero
  training cost (GDN state spike required); b16 retrain value mostly
  evaporates; ALSO investigate why acceptance is low (quant? HC taps
  extraction at serving = the inco training distribution mismatch?) —
  a 10-point per-position acceptance recovery is worth more than b16.

## 4. Revised priorities (solo)
P0: source-split accept telemetry + per-position histogram + Recall@k probe.
  1-2 days, prod-safe, decides everything. [was already P0; now it is THE gate]
P-A: if flat => b16 DFlash2 retrain (Primus/SpecForge on rented MI355/H100,
  $1.2-4k, 2-3 weeks wall-clock incl data regen; warm-start verify-first;
  LiLiCorr co-train as a same-run add-on for +5-9% more if budget allows).
  Engine port: kDf2Rows parametrization (compile-time variant like
  STRATA_WINDOW_MAX_T pattern) + cfg block plumbing + b16 pack.
P-B: if steep => tree verify over existing b8 (design in
  tree_spec_research.md §5 still valid; GDN snapshot spike first) + hunt the
  acceptance-loss cause first (may be free).
P1/P2 from the audit unchanged (overlap gate, dynamic tiers; a b16 ckpt makes
  tiers strictly better since one checkpoint serves any width <=15).
LiLiCorr-as-separate-serving-path: DEAD END (no checkpoint, co-trained
  design, b8-only NVIDIA releases). Only viable inside a co-train run.

## 5. Open verification items before money moves
1. SpecForge draft_checkpoint_path warm-start b8->b16: read the source
   (sgl-project/SpecForge) — agent claim, UNVERIFIED.
2. b16 loss-decay gamma 7 setting: from DFlash paper, UNVERIFIED directly.
3. Our per-position curve: P0, above.
4. Verify b16 selector-walk cost on gfx1201 after port (16KB LDS table).
