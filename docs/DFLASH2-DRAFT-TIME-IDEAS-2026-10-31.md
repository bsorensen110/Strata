# DFlash2 draft-time reduction: fuse, tile, graph (2026-10-31)

Desk study (no GPU work: production held `/dev/kfd`). Source for facts: code at HEAD `556d7c61`, and the measured
docs cited below. **Every expected effect is an estimate and labelled as such**; the byte model in §2 is arithmetic
from pack sizes and quant types, not a measurement. The one true source for per-stage draft time exists
(`STRATA_DF2_PROFILE=1`) and has **never been run on gfx1201** — experiment E0 (§6) closes that gap for free.

Companion plan (batch scheduling, adaptive depth, VRAM): [DFLASH2-BATCH-IMPL-PLAN-2026-10-31.md](DFLASH2-BATCH-IMPL-PLAN-2026-10-31.md).
This file focuses on the **draft pass itself** — the drafter round in solo and the co-draft loop in batch.

## 1. What one draft round is (code map, re-grepped on this tree)

`Dflash2Drafter::draft()` (`src/core/dflash2.cpp:689`): fill `h_meta_` → launch one captured graph
(`capture()` 621 bakes `record_context` + `record_block` per row count n=1..8, per slot) →
**full `cudaStreamSynchronize`** (727) → read 7 tokens + 7 probs from the pinned `h_out_` (the −1 spin-init
sentinels are already there, lines 711/728 — the spin-read shape exists, it is not used).

- Solo serve path: `generate.cpp:20270` — `df2.draft(ver.tap_dev(), a+1, ...)`, one round per window, on the
  window's heels; the next window's rows are built from the drafts, so the round sits on the critical path.
- Batch path: `generate.cpp:14235-14300` — per slot: taps `cudaMemcpyAsync` into the **one shared** `df2_stage`
  → `set_slot` → `draft()` (graph + full sync) → `set_slot(0)`. Four slots = four serialized host round trips.
  The data dependency is the **window** (taps = window output), not the commit (BATCH plan Phase 3, facts from
  `record_context`/`record_block`: the drafter reads taps + its own ring, never the verifier KV).

Kernel inventory per round (single stream, `record_context` + `record_block`, IL head on):

| part | kernels |
|---|---|
| meta copy | 1 |
| context (n rows) | rope_table, quant_rows, **fc GEMV**, rms_quant, 5×(kv GEMV + kv_ring) = 14 |
| block front | rope_table, embed, rms_quant = 3 |
| per layer ×5 | **akp GEMV**, conv_quant, **qkv GEMV**, norm_rope_qk, **attention**, **o GEMV**, conv_rms_quant, **mkp GEMV**, conv_quant, **gate+up GEMV**, swiglu_quant, **down GEMV**, conv_rms_quant = 12 → 60 |
| tail | quantize_q8_1, q8_interleave, **head GEMV** (7×106,299), topk_part, topk_merge, selector GEMV, **df2_select** = 7 |

≈ **85 in-graph launches**, of which ~15 are elementwise 6–10 µs kernels (measured: `conv_rms_quant` 6.3–6.5 µs)
and 12 are GEMVs. DFlash2 is a block drafter: the 7 proposals come out of **one** 8-row block pass — there is no
7-step chain to graph; the serial "steps" that exist are (a) the ~85 kernels inside the round — already one graph —
and (b) in batch, the **4 slots** drafted serially.

## 2. Byte model of a round (the round is weight-streaming bound)

From the pack (760 MB, `~/Strata-data/dflash2/qwen3.8-flash-next-dflash2.pack`) and geometry
(`dflash2.hpp:181`: N_=2560, L_=5, NH_=32, NKV_=8, HD_=128, I_=7680; header: taps 5×2560/row → fc 12800→2560;
head type 14 = Q6_K, 106,299 rows):

| stream | bytes/round | µs @ 460 GB/s (est) |
|---|---|---|
| fc GEMV (Q8_1, 32.8M params) | ~35 MB | ~76 |
| 5 layers Q8_1 (75.2M params/layer) | ~400 MB | ~870 |
| ring reads (window_ = 2048, fp16 K+V, 5 layers) | ~42 MB | ~92 |
| head GEMV Q6_K (272M params) | ~221 MB | ~480 |
| logits write + topk read | ~6 MB | ~13 |
| elementwise (~35 kernels) | ~3 MB traffic | ~250 (launch/latency bound, not bytes) |
| topk (measured 49.5–50.4 µs) / select (measured 13.6 µs) | — | ~63 |

**Round ≈ 1.6–1.9 ms GPU** at a 2048-token window (estimate from the byte model; anchors: MTP's whole 7-step
chain round is 0.768 ms with a 0.234 ms head GEMV → ~110 MB, per `MTP_PROFILE` in the ideas doc; gfx12 effective
GEMV bandwidth taken at 430–480 GB/s against 576 GB/s peak). Unvalidated until E0.

Where that puts draft time (estimates, from the doc numbers):
- **Solo**: window ~37–45 ms at 85–89 tok/s (`DFLASH2_BATCH_TUNING` cap sweep; MTP solo ~33 ms at 118 tok/s);
  the draft round + sync + staged read adds **~2–2.6 ms/window ≈ 5–7%** of solo window time.
- **Batch**: 82–84 ms/window vs 45 for batch MTP (`AB-FUSED-DECODE-E1`). Split: a 16-row window carrying ~10.8 rows
  (2.99 slots × 3.60 rows) makes the **verify** window itself cost more (CUDA0 47 ms + CPU experts ~20 ms), and
  **4 serialized co-draft rounds** (4 × ~2.2 ms GPU + 4 × (memcpy + graph launch + full sync) host round trips,
  plus a per-slot head GEMV that re-reads 221 MB of head weights each time) sit on the critical path.
  Co-draft serial cost: **~8–12 ms/window** (est). Not separable from the wider-window cost without the E0 arms.

## 3. Ideas, ranked by expected value

Parity classes (BATCH plan header): proposals the target did not pick are never emitted, so **drafter-internal**
changes keep greedy output bit-identical even when they change draft bits; sampled output is distribution-level,
not bit-identical, when proposals change. Kernel-sequence-preserving changes are bit-identical everywhere.

### P1 — Batched head across slots (`STRATA_DF2_BATCHHEAD`, BATCH plan Phase 1) — biggest batch lever
Already designed; the byte math strengthens the case: today each slot's round reads the **whole 221 MB head**
weights — 4 × 221 MB per window. Stacking 4 slots × 7 rows = 28 rows into one `native_mmvq` pass cuts that to
221 MB: **−~1.3 ms GPU/window (est)**, 12 head/topk/select launches → 3, 4 syncs → 1, and with the Phase 1.6
event scheduling (`set_post_launch` → `waitEvent(E(window))` → taps copies → bodies → one batched head graph)
the whole co-draft overlaps the commit and the next-window queue. Compare target: 82–84 ms windows.

### P2 — Solo: take the sync off the critical path (solo DEVTOK shape)
`draft()`'s full `cudaStreamSynchronize` (dflash2.cpp:727) makes the solo critical path
window → accept → **full draft round, host blocked** → rows. Two code-supported moves:
1. **Spin the mapped sentinels** — the −1 init + volatile reads at 711/728 already implement the safe read shape;
   replace the stream sync with a spin (bounded, fall back to sync on timeout). Removes the sync's wake latency.
2. **Event-chain the draft behind the window graph** on `cs_` (solo: no slot loop, so this is Phase 1.6 in
   miniature): enqueue the round right after the window's event, and do host accept / emit / next-row build while
   it runs; wait once, immediately before the rows need `dd[]`. `tap_dev` is written by the window and valid
   "until the next window" (verify.hpp:433–436), so the window event is the true data dependency — the next
   window's launch must gate on an event recorded after its taps are consumed (DEVTOK race pattern).
   Estimated **−0.5–1.5 ms/window (est, ~2–4% solo)**; the MTP analogue removed 3 ms/slot-class stalls
   (`DECODE-TRACE` follow-up: 80 of 85 long stalls were MTP select → host copy).
   Gate: `STRATA_DF2_SOLOPIPE=1`, off = today's blocking draft.

### P3 — Quantize the drafter's weights (bit-safe byte lever, cheap for the head)
The round is ~650 MB of weights. Cheapest tier first:
- **Head Q6_K → Q4_K/Q4_0**: `native_mmvq` already supports Q4_0/Q5_0/Q4_K rows (`dflash.cpp:126`), so the head
  tier is a **repack** (`tools/dflash_head_quantize.py` + `dflash2_pack.py`) — 221 → ~153/136 MB, **−0.15–0.19 ms
  per round (est)**. Gated: `--draft-model` pack choice is already per-config; no engine change needed. Greedy
  bit-safe (verify gates proposals); sampled = distribution-level, label per skill.
- **Layers Q8_1 → Q6_K/int4 planar**: −100–200 MB/round, **−0.22–0.43 ms/round (est)**, but `gemm8_kernel` is
  int8-planar-only (`dflash2.cu:331`) — needs a new planar format + GEMV variant (real kernel work; parity of
  quantization to be proven against the pack tools). Do after the head tier proves the acceptance cost is small:
  the acceptance lever is cap/conf (measured: quant tier's acceptance cost **unmeasured**; run the cap sweep arms
  with a repacked pack).

### P4 — Fuse the elementwise steps into the GEMMs (bit parity by construction)
The ~15 quant/norm/conv elementwise kernels per round (~250–350 µs + ~35 launch gaps) are epilogues of their GEMMs:
- `df2_swiglu_quant` → epilogue of the gate+up GEMM (the 32-element quant blocks align with the warp's output
  pattern in `gemm8_kernel`; `quant_block` already reduces via shuffles).
- `df2_conv_rms_quant` → epilogues of o_proj / down GEMMs (reads h_ residual + dyn, writes xn_/fn_ + planar).
- `df2_conv_quant` → epilogue of akp/mkp (dyn) GEMMs.
- `df2_norm_rope_qk` → attention prologue; `df2_kv_ring` → the ctx K/V GEMM epilogue; rope tables → embed/quant.
Elementwise math and per-element order are unchanged → **bit parity**; removes ~30 launches and their gaps.
Estimate **−0.2–0.3 ms/round (12–17% of the round)**; in batch it multiplies across slots' bodies. Ship behind
`STRATA_DF2_FUSE_EPI=1`; `dflash2_stage_parity.py` exists for the stage-by-stage parity proof.
Sub-item: **fuse topk into the head GEMV epilogue** (running top-16 per row in smem): −50 µs topk pass + 3 MB
logits traffic. The BATCH plan defers this as batch noise (topk 50 µs vs 82 ms windows) — against a ~1.7 ms
**solo round** it is ~3%: keep it low priority, solo-relevant only.

### P5 — Tile tuning (measured, cheap targets)
- **`topk_part_kernel`** (`dflash2.cu:585`): extracts the 16 winners **sequentially**, each extraction = a full
  7-value register scan per thread + 2 block barriers, 16 times, 64 blocks/row. 49.5–50.4 µs to read 3 MB ≈ 7×
  the bytes bound. Per-thread sorted top-16 + tournament merge gives the identical selection sequence (same
  `better` comparator: each step still takes the global max) → target ~10–15 µs. Pure kernel, bit-parity provable
  against today's kernel with a parity test over logits buffers.
- **`select_kernel`** (687): one 256-thread block walks 7 rows serially (13.6 µs). The inter-row chain is real
  (row j conditions on row j−1's pick); the **intra-row** scoring (16 candidates × rank 256 codebook) can go
  parallel. ~5–8 µs target (est).
- **`gemm8_kernel` occupancy**: grid = rows/(WARPS·8); on 54 CUs the ctx GEMVs (2560 rows → 40 blocks at
  WARPS=8) leave 14 CUs idle. `df2_gemm_bench` + `df2_gemm_variant` exist — sweep WARPS/R per shape on gfx1201
  and pick the winner per matrix (est: 5–10% of GEMV time on the small matrices, unmeasured).
- Attention: ring reads are ~42 MB/round at window 2048 — `dflash2_attn_bench` exists; check the part/merge
  split hits bandwidth, tune `kDf2TopBlocks` alongside the topk rewrite.

### P6 — Batch co-draft as ONE graph launch (graph the multiple slots, not the steps)
The steps are already in a graph; in batch the serial chain is slots. After P1's batched head, capture the whole
co-draft per window (taps copies + 4 bodies + batched head) as one `cudaGraphExec` per `n_sel = 1..max_slots`
(child-graph nodes, or one stream capture) → 1 graph launch + 1 sync per window. Launch-side win is small
(≈10–40 µs, `dflash2_launch_bench` measures the floor on this card); the win is the **host round-trip count**,
which P1/P2 already fix. Fold into P1's gate; do not schedule separately. The super-graph **cannot** span
accept → row-build (host decisions) — events are the right mechanism, as BATCH plan Phase 1.6 chose.

### Non-goals (measured dead, reasons on record)
- MTP two-phase co-draft (no gain), async commit (wait moved to emit), `STRATA_FUSE_DECODE` (inert on native
  pack — `AB-FUSED-DECODE-E1` verdict 1), head+topk fusion as a *batch* priority (noise there), verify launch
  count (separate solo-trace campaign). Do not re-litigate without a new measurement.

## 4. Sequencing (estimates, unmeasured)

| # | work | needs | effect (est) |
|---|---|---|---|
| E0 | profile + microbenches (below) | one measurement window | turns §2 estimates into a stage budget |
| 1 | P1 batch head + events (BATCH plan Phase 1) | code GPU-free, build+measure | batch: −1.3 ms head bytes + −6–9 ms serial sync tail |
| 2 | P2 solo spin + event chain | small code | solo −0.5–1.5 ms/window |
| 3 | P3 head Q4 repack | pack tool run | −0.15–0.19 ms/round, solo+batch, no engine code |
| 4 | P5 topk/select/occupancy tuning | kernel work | −40–70 µs/round, bit-parity testable |
| 5 | P4 epilogue fusions | kernel work, parity gates | −0.2–0.3 ms/round |
| 6 | P3 layers int4/int6 planar | kernel work | −0.22–0.43 ms/round |
| 7 | P6 whole-co-draft graph | folds into P1 | host launches 6→1 |

P1+P2+P3(head) touch no kernels, are the three biggest levers, and gate cleanly.

## 5. Parity and gates (per skill + BATCH plan)

Every arm: env gate + engaged/refused banner; parity probe (temp 0, fixed prompts, SHA-256) against a two-launch
base floor **before** throughput; interleaved A/B; solo numbers from the engine `generated in` line, concurrent
from client tokens / wall time. P1/P2/P4 are bit-parity by construction (same kernel sequence per slot / same
per-element math); P3 and P5-topk change drafter-internal bits → greedy answers identical (verify gates), sampled
statistical; P3-layers ships behind the repack + its own label.

## 6. Experiment E0 — the per-stage draft budget (first GPU window)

The knobs exist; nobody has run them on this GPU (no `strata dflash2 profile` line exists in `docs/`/`bench/`):
- `STRATA_DF2_PROFILE=1` (dflash2.cpp:241) — per-stage ms/round table at exit (`gemm:qp`, `df2_attention`,
  `native_mmvq`, `df2_topk`, ...); runs the round **without** its graph, so treat it as a stage mix, not the
  production round time.
- `STRATA_DF2_TIMING=1` (242) — `ms_gpu`/`ms_draft`/rounds with events around real graph rounds.
- `STRATA_BATCH_DRAFT_TIMING=1` — `strata batch draft: <ms> ms/window in <n> co-draft calls/window` (the batch
  "draft time ms" metric named in the ask; set it in every arm).
- Microbenches (service stopped, `systemd-run --user --scope`): `dflash2_gemm_bench`, `dflash2_attn_bench`,
  `dflash2_small_bench`, `dflash2_bw_bench`, **`dflash2_launch_bench`** (in-graph per-kernel floor + idle-launch
  wall on gfx1201 — never run here; sizes the fusion/graph-launch ceiling).

Arms: solo DF2 (`--draft dflash2 --draft-cap 6 --spec 8 ...`, per the skill's DFlash2 flag set) and 4-slot batch
(`--batch-draft-window 16 --batch-draft-depth 1|3`), 2+ trials, cold separated. Deliverable: fill in §2's table
with measured µs/stage, then re-rank P4–P6 against the measured launch floor (if in-graph gaps are ~1–2 µs,
fusion is mostly about the 50 µs topk + 250 µs elementwise, not gaps; if gaps are ~5 µs, fusion/occupancy jumps).

Reproduce (per `strata-speed-ab-testing` + `strata-hip-install/scripts/candidate_config.py`, scratch server on
the stopped service, restore after):
```sh
~/Strata/.venv/bin/python tools/ab/greedy_bench.py $S/cand_df2.json $S/bench_df2.json DF2-PROF
~/Strata/.venv/bin/python tools/ab/batch_burst_bench.py $S/cand_df2b.json $S/bench_df2b.json --rounds 4 --max-tokens 256 --conc 4
cmake --build build-verify --target dflash2_launch_bench dflash2_bw_bench   # EXCLUDE_FROM_ALL (CMakeLists.txt:954,957)
LD_LIBRARY_PATH=/opt/rocm/lib $HOME/Strata-integ/build-verify/dflash2_launch_bench 100
LD_LIBRARY_PATH=/opt/rocm/lib $HOME/Strata-integ/build-verify/dflash2_gemm_bench
```

## Corrections / open questions
- §2's window-split (verify vs co-draft share of the 82 ms) is not separable from existing runs; E0's
  `STRATA_BATCH_DRAFT_TIMING` arm settles it.
- Acceptance cost of P3 (quant tier on draft quality → cap re-tune) is unmeasured; run it with the cap sweep.
- The solo-window budget in §1 (37–45 ms) is inferred from the cap-sweep medians, not a direct timer; E0's
  `STRATA_DF2_TIMING` gives ms_draft/round directly.
