# 2-bit K / 1.5-bit V KV storage (`--kv q2`), and why the FP16 window is what makes it usable

Design for the second KV-compression step. The first one — the FP16 sliding window + attention sinks
(`docs/kv-window.md`, `bench/results/2026-10-11-kv-window/`) — is the prerequisite, not a sibling: at 4 bits the
bulk quantization error never reached the model's output, and at 2 bits it does. The overlay is what keeps a 2-bit
bulk from collapsing.

## What the literature says, and what it means for this engine

* Plain 2-bit per-token KV collapses. Every 2024-2026 paper agrees (KIVI 2402.02750, KVQuant 2401.18079,
  RotateKV 2501.16383, SKVQ 2405.06219). 2-bit is only viable as **rotation + asymmetric codebook + recent/sink
  protection**.
* **Rotation first.** Strata already does this for `q4_0`: K and V are rotated by the orthonormal 256-point
  Walsh-Hadamard matrix before quantization, the query is rotated the same way so `<Hq,Hk> = <q,k>`, and the
  attention output is rotated back (`kv_q4.hpp`). That removes the outlier channels that kill naive 2-bit, so the
  2-bit work starts from a rotated space and reuses `fwht256_cuda` unchanged.
* **KIVI's asymmetry**: keys have persistent per-channel outliers, values do not, so K wants per-channel scales and
  V per-token scales. After a Hadamard rotation the channel outliers are gone, so per-token block scales are
  adequate for both — measured, not assumed, in the parity test.
* **SKVQ's shape** (2405.06219): 2-bit K, 1.5-bit V, plus a sliding FP16 window for recent tokens. The 1.5 bits is
  *1.5 bits stored*, which requires entropy coding; a 3-level codebook packed 2-bits-per-value is 2 bits stored.
  Stated here as a difference, not glossed over.

## Layout (head_dim 256, 2 KV heads — the Swift/Qwen3 shape `q4_0` already serves)

Bytes per cell (one token, one layer = K and V of both heads). Measured layouts, not targets:

| format | per head per cell | per cell | vs fp16 |
| --- | --- | --- | --- |
| fp16 | 512 B | 2048 B | 1.0x |
| int8 | 264 B | 1056 B | 1.9x |
| q4_0 (in use) | 144 B | 576 B | 3.6x |
| **q2: 4-level K, 32-value blocks** | 80 B | 320 B | 6.4x |
| **q2 + 1.5-bit V (entropy-coded)** | 80 B K + 64 B V | 288 B | 7.1x |

Block layout for the 2-bit tier, mirroring `block_q4_0` so the append/gather kernels keep their shape:

```
#pragma pack(push, 1)
struct block_q2_0 {           // 32 values, 10 bytes: 2.5 bits/value stored, 2 bits of code
    uint16_t d;               // fp16 scale (the block's RMS for the shipping Lloyd-Max codebook)
    uint8_t  qs[8];           // 32 2-bit codes; element j in bits [2*(j%4), +2) of qs[j/4]
};
#pragma pack(pop)             // 10 bytes -> 8 blocks per head per cell = 64 B of codes + 16 B of scales = 80 B
```

With 32-value blocks the scale bytes are 25% of the payload, which is why the table says 80 B and not 64 B:
`8 x (8 B codes + 2 B scale) = 80 B` per head per cell. A 64-value block (16 B codes + 2 B scale, 4 blocks) gives
72 B and is the variant to try first if the scale bytes dominate the error.

**Codebook.** K uses the symmetric 4-level set `{-3, -1, +1, +3} * d` (no zero point, so no min byte). V uses the
3-level set `{-1, 0, +1} * d`, stored in 2 bits; the 4th code is reserved, which is what makes entropy coding
possible later without changing the layout.

**The overlay is not optional here.** `kv_window.hpp` keeps the `sinks` leading cells and the `window` most recent
cells in FP16. At 4 bits that was a precision improvement with no measurable end-to-end effect; at 2 bits it is the
difference term: the bulk's relative RMS roughly doubles from q4_0's 8.57%, and everything the model reads in the
last 128 tokens and the first 8 is exact.

## Scope of the first implementation

In: `block_q2_0` layout; append (quantize) and gather (dequant) kernels in `src/kernels/cuda/kv_q2.cu`, reusing
`fwht256_cuda` and the `KvHostPools` granule so `--kv-resident` streaming keeps working; the decode-attention read
path; `--kv q2` in `generate.cpp`; `qsa_kv_format` and the `kv_stream.hpp` format constant so block moves know the
granule; a `kv_q2_parity` test with the per-bucket relative-RMS table and an attention-output error check; the
overlay combined.

Out of the first implementation, stated as open: entropy coding for the 1.5-bit V (it changes the layout's
read path, not just the writer); per-channel K scales (only if per-token blocks measure worse than expected);
the prompt-attention read path (the window work left it reading the bulk — same limitation, same reason).

## How "near lossless" gets proven

Same harness that measured q4_0 and the window, so the numbers are comparable:

1. `kv_q2_parity` — deterministic: relative RMS of K and V against an FP16 cache, by bucket (sinks / window /
   bulk), plus the attention-output error against a full-precision reference. This is where a 2-bit scheme is
   shown to be too lossy *before* any GPU time is spent on end-to-end runs.
2. `tools/kv_pr_report.py` — exact-copy of 12 five-digit codes (6 in the body, 6 inside the window) at 4K, 16K,
   32K, 64K, 131K, plus the reasoning-loop rate, for four arms: `q4_0` (reference), `q2` alone, `q2` + window,
   `fp16` (upper bound). The claim to earn is `q2 + window` matching `fp16` on all of it.
3. Speed, same runs: prefill and decode t/s. 2-bit is 1.8x smaller than q4_0, so a regression here is a bug.

Build with CUDA 13.0 (the `build-cu130` recipe in bench/results/2026-10-11-kv-window/README.md) — a 13.2 build is
numerically broken for this model and cannot be used
for any of these measurements.

## State of the implementation

Done:

* `include/strata/kernels/kv_q2.hpp` — `block_q2_0` (10 B: 2 scale + 8 codes), the byte math (80 B per cell and
  head, 320 B per cell for 2 KV heads), and the host codebook helpers `q2_code_4level` / `q2_value_4level` /
  `q2_code_3level` / `q2_value_3level` that the kernel and the parity test share.
* `src/kernels/cuda/kv_q2.cu` — append (per step, batched decode, prompt + staging pool) and gather, on the same
  page-table addressing, host-copy write and warp-wide scale rule as `kv_q4.cu`, with the codebook chosen per
  plane.
* `include/strata/kernels/kv_stream.hpp` — `KvHostPools.k_q2/v_q2` and `kKvQ2 = 4`, so streaming and snapshots
  have a format number to pass.
* `src/kernels/kv_q2_parity.cpp` — the gate described above: bitwise writer/reader parity, the per-format RMS
  table for K and V, the attention-output error, both V codebooks, and the byte-math assertions.

Remaining, in order:

1. **Done.** `src/kernels/cuda/kv_q2.cu` is registered, `kv_q2_parity` is a test target, and the gate passes on
   CUDA 13.0. The kernel ships the Lloyd-Max codebook (RMS scale), and the gate's Lloyd-Max rows are the device
   pools read back through the real gather kernel — they match the host reference bitwise.
2. `layer.hpp` / `layer.cpp`: a `kv_q2` flag on `QsaState` (`qsa_set_kv_q2` / `qsa_kv_q2`), the pools, the append
   sites, and `qsa_kv_format` returning `kKvQ2`.
3. `qsa_decode_attn.hpp` / `.cu`: `k_q2`/`v_q2` in `QsaAttnPools` and the read path in the fused decode attention,
   combined with the window overlay's `k_win`/`v_win` exactly as q4_0 is.
4. `verify.cpp` (draft verification), `prefill.cpp` (the prompt path), `conversation_snapshot.cpp` (the format
   number and restore), `generate.cpp` (`--kv q2` in the accepted set at the `--kv` validation).
5. Measure: `tools/kv_pr_report.py` over `q4_0` / `q2` / `q2 + window` / `fp16` at 4K-131K.

Open, stated as open: entropy coding for a true 1.5-bit V (it changes the reader, not just the writer);
per-channel K scales (only if per-token blocks measure worse than the parity test predicts); the prompt-attention
read path, which the window work also left reading the bulk.

## Measured (RTX 5080, CUDA 13.0, `kv_q2_parity`)

Synthetic rotated cache, 1024 cells, 2 KV heads, head_dim 256, mild per-cell magnitude spread. Error against an
FP16 cache, relative RMS:

| format | K | V | attention output |
|---|---|---|---|
| `q4_0` (in use) | 8.574% | 8.588% | 2.052% |
| `q2`, V 4-level `{-3,-1,+1,+3}` | 60.110% | 60.088% | 53.427% |
| `q2`, V 3-level `{-1,0,+1}` | 60.110% | 65.590% | 25.502% |

Writer/reader parity is bitwise, 6/6. Bytes: 80 per cell and head, 320 per cell for 2 KV heads against q4_0's 576
— the 1.8x saving is real and delivered.

**The uniform codebook is not near lossless, and the reason is the codebook, not the bit count.** A 4-level
uniform grid over `[-amax, +amax]` puts its inner level at `amax/3` and its outer at `amax`. On rotated KV,
`amax ~ 2.7 sigma`, so the nearest level to a typical value (one sigma) sits 0.9 sigma away and the next sits 2.7
sigma away: most of the mass of the distribution lands on a level far from where it lives. That is where the 7x
error over q4_0 comes from, and it is why the earlier note here said the bulk's error "roughly doubles" — it does
not, it multiplies by seven.

Two things the gate settled:

- The 3-level V is lossier per element (65.6% vs 60.1%) and cheaper **only if entropy coded**, yet on attention
  output it is *better* (25.5% vs 53.4%): a codebook that maps small values to zero leaves them alone, while a
  uniform 4-level grid pushes every small value out to `amax/3` and inflates it. The codebook choice is a real
  decision made from data, not a formality.
- 2-bit K at 60% relative RMS is not shippable. The FP16 window and sinks protect the positions attention leans
  on hardest, but the bulk needs a better codebook before the format is usable at all.

## Next: the non-uniform codebook (SKVQ's actual lever) — measured

SKVQ's gain over uniform 2-bit is a Lloyd-Max codebook fitted to the KV distribution. Measured in the same gate
(host-side, same blocks, scaled by block RMS instead of amax):

| 2-bit format | K | V | attention output |
|---|---|---|---|
| uniform `{-3,-1,+1,+3}` over amax | 60.110% | 60.088% | 53.427% |
| Lloyd-Max `±0.459, ±1.510` over RMS | 33.381% | 33.413% | 19.920% |
| Lloyd-Max 3-level `0, ±1.454` over RMS | 33.381% | 44.750% | 21.666% |

The codebook is worth 1.80x on both K and V for free — same 10-byte block, same reader. It also settles the V
codebook cleanly: with a uniform grid the 3-level V looked better on attention output (25.5% vs 53.4%) while
being lossier per element, which was an artifact of the uniform grid inflating small values; with Lloyd-Max the
4-level V wins on both measures (33.4% / 19.9% against 44.8% / 21.7%). Ship 4-level Lloyd-Max for K and V.

What is left before "near lossless": 33% per-element RMS is 3.9x q4_0's 8.57%, and 19.9% attention output is
9.7x q4_0's 2.05%. The codebook closed the shape of the distribution; the remaining gap is scale granularity —
one fp16 scale per 32 values is coarse, and KIVI closes this by splitting K into per-channel and per-token
scales. That is the next lever, with the FP16 window and sinks carrying the recent and sink positions exactly.

Order from here:

1. Lloyd-Max 4-level K and V in the kernel (RMS scale, fixed level table); re-measure. **Done.**
2. Finer scale granularity: 16-value blocks, then KIVI's per-channel + per-token split for K. **Measured, and the
   first half is wrong** — see the next section.
3. Wire `--kv q2` through `layer.cpp` / `qsa_decode_attn` / `verify.cpp` / `conversation_snapshot.cpp` with the
   FP16 window and sinks on by default. **Done** (mode 4 in the fused attention, streaming movers, drafter ring,
   snapshot restore).

## Scale placement — measured, and it splits the two metrics

Same 2 bits, same Lloyd-Max codebook, same 80 B per cell and head; only WHAT the fp16 scale measures changes:

| scale over | K | V | attention output |
|---|---|---|---|
| 32 values of one cell (shipping) | 33.381% | 33.413% | 19.928% |
| 16 values of one cell (12 B per 32 values, not 10) | 32.586% | 32.604% | **37.228%** |
| 32 cells of one dimension (KIVI-style, same 80 B) | 44.414% | 44.524% | **7.150%** |

Two findings, and they point in opposite directions:

**Halving the block does not pay.** A 16-value scale moves K and V by 2% and makes the attention output 1.87x
WORSE. A shorter block estimates the scale from 16 samples instead of 32, and estimator noise lands on every code
in the block; nothing in the data pays for it. The gate pins this (`the 16-value scale does not beat 32 on the
attention output`) so nobody adopts a finer block because finer sounds better.

**Per-channel scaling is the lever, and the two metrics disagree on purpose.** Per-channel is 33% WORSE per
element (44.4% against 33.4%) and 2.8x BETTER on the attention output (7.15% against 19.93%) at the same byte
cost. The reason is what attention does with the error: a scale shared across cells makes K's quantization error
largely common-mode across cells, and softmax is invariant to a common-mode shift of the scores, so that error
cancels where it is measured. A per-cell scale produces errors that differ cell to cell, which move the RELATIVE
scores, and the relative scores are the whole output. Per-element RMS counts error that cancels and ignores error
that does not.

So the metric for this feature is the attention output (and, downstream, the logits) — not per-element RMS. The
gate's per-element rows stay in the table as diagnostics, labelled as such.

Caveat to keep: this gate's cache is i.i.d. Gaussian with a 3x per-cell magnitude spread. That spread is exactly
what penalizes a shared channel scale per element, and the data has no channel-aligned outliers, the structure
KIVI's gain rests on. The 2.8x is therefore a floor on the win, measured on data that is unkind to it — and the
real number needs a real cache.

Order from here:

1. Per-channel 2-bit layout (codes packed per dimension across 32 cells, one fp16 scale per dimension per group,
   80 B per cell and head) in the kernel and the fused attention read path; re-measure end to end. **Done** —
   `block_q2c_0` (10 B per dimension per 32-cell group), group-granular append with an atomic word-masked path
   for partial groups, mode 4 in the fused attention, window restore, snapshot format guard. The gate's device
   rows read the new pools through the real gather kernel and match the host reference bit for bit:
   **attention output 7.150% against the per-token layout's 19.913% (2.78x better), per-element 44.414% against
   33.381% (1.33x worse)** — the split the table above predicts.
2. Real cache statistics: dump a layer's FP16 K/V from a served prompt and run the same comparison on it, so the
   per-channel number is measured on data with the outlier structure the method targets.
3. Keep the FP16 window and sinks carrying the recent and sink positions exactly.
4. End-to-end suite on the real model: `q4_0` vs `q2` vs `q2`+window, exact-copy probes at 4K-131K.
5. Only after 1-4: the entropy-coded 1.5-bit V, which buys 288 B/cell (7.1x) on top.

## Why the published "near lossless at 2 bits" does not transfer to our numbers

Checked against the sources (2026-10-12), not against summaries. What the papers actually specify:

- **KIVI** (Liu et al., arXiv 2402.02750): key cache quantized **per-channel**, value cache **per-token**,
  **asymmetric** quantization (`z = min(X)`, `s = (max-min)/(2^B-1)`), group size 32, and a **residual of 128
  recent tokens kept in FP16**. Their "near-lossless" is perplexity/QA at 512-1024-token contexts with that
  residual in place. The paper's own text says the fake 2-bit (K per-channel, V per-token) result is
  "significantly worse" on hard generation tasks such as GSM8K.
- **SKVQ** (Duanmu et al., arXiv **2405.06219**, COLM 2024 - not 2505.22301): **channel reordering** (a
  transform-invariant permutation that groups channels with similar statistics), **clipped dynamic
  quantization** (per-group min/max, dynamic, i.e. asymmetric with a zero-point), and a **sliding window of
  recent tokens kept in high precision**. Headline: K 2-bit, V 1.5-bit, "almost no accuracy drop".
- The 2-bit-K / 1-bit-V split (1.5-bit average) is **AsymKV** (Tao et al., arXiv 2410.13212), not KIVI's
  main configuration.

Five reasons our measurement is worse than their claim, in the order they bite:

1. **Our gate data cannot show the gain per-channel scaling exists to capture.** KIVI's whole per-channel
   result rests on K having a few channels with far larger magnitude than the rest. The gate's cache is i.i.d.
   Gaussian: every channel's RMS is the same number, so a per-channel scale has nothing to exploit. Our 2.8x
   is a floor measured on data that is structurally blind to the mechanism.
2. **We quantize a Hadamard-rotated cache; they do not.** Rotation equalizes channel variance - that is its
   job for weight quantization - and it removes exactly the channel outliers KIVI's per-channel scaling
   exploits. SKVQ's answer to the same problem is the opposite: reorder channels so similar ones group
   together, keeping the structure. We are the only one of the three with the structure rotated away.
3. **We are symmetric with no zero-point and no clipping.** Our block stores an fp16 RMS scale and a symmetric
   Lloyd-Max codebook. KIVI and SKVQ both use asymmetric min/max with a zero-point, and SKVQ clips. On
   outlier-heavy data, an RMS scale over a group is dominated by the outliers, so the non-outlier values get
   the coarse end of a 4-level codebook.
4. **Our scales are never joint.** KIVI applies per-channel to K and per-token to V *at the same time*. Our
   formats pick one axis per tensor. Measured in the gate: K per-channel + V per-token at 2 bits is
   **6.226% against the shipping symmetric per-channel 7.150%** (1.15x, same 160 B per cell and head).
5. **Their bar is a window bar and a perplexity bar.** KIVI's 128-token FP16 residual and SKVQ's FP16 sliding
   window carry the positions that dominate short-context perplexity, and perplexity hides the loss that
   top-1 agreement and KL show. Our bar is 511 teacher-forced positions of KL/top-1 plus a 5K buried-fact
   probe, which is stricter than what "near lossless" claims in those papers.

What the asymmetric experiment settled (gate, same probe, bytes per cell and KV head):

| arm | attention output | B |
|---|---:|---:|
| K 4-bit + V 2-bit per-channel | 4.768% | 224 |
| K 2-bit per-channel + V 4-bit | 5.710% | 224 |
| K per-channel + V per-token, both 2b | 6.226% | 160 |
| K 2-bit + V 1-bit per-token | 7.057% | 114 |
| K per-token + V per-channel, both 2b | 10.134% | 160 |
| 1-bit both, per-channel | 8.106% | 96 |
| q2 per-channel (ships) | 7.150% | 160 |
| q4_0 | 2.052% | 288 |

KIVI's placement findings reproduce on our rotated cache (K wants per-channel: per-token K is 1.42x worse; V
wants per-token: 1.15x better than per-channel V). Asymmetry narrows the gap and does not close it - the best
224 B arm is still 2.3x q4_0's error.

Order from here, by expected gain per unit of work:

1. **Real-cache statistics** (step 2 above) - without it every per-channel number stays a floor measured on
   blind data, and the rotation question cannot be answered at all.
2. **Asymmetric quantizer with a zero-point and clipping** in the gate, KIVI-style, measured against the
   symmetric Lloyd-Max on that real data.
3. **K per-channel + V per-token** as a format (measured 1.15x at identical bytes).
4. **Window policy at the logit level**: sweep the FP16 window (128 -> 1024) in `kv_precision_compare.py`;
   the papers' near-lossless is largely this device, and we have never measured its size.
5. Channel reordering (SKVQ) as the alternative to rotation, if the real-cache dump shows the outliers.

## Real cache: the dump, the statistics, and six rankings that flipped

`--dump-kv PATH` (engine) writes the stored KV cache of every QSA layer after the prefill - rotated,
post-RoPE, exactly the tensors the attention kernels read (`st.k_pool`/`st.v_pool`, the same pointers
`kv_gather_step` and `qsa_attn_pools` use). `tools/kv_cache_dump.py` reads it and prints the statistics;
`STRATA_KV_PROBE_CACHE` + `STRATA_KV_PROBE_LAYER` feed it to the parity gate, which now runs the whole
table on real data.

Statistics from an 890-token document prompt, layer 3, per (kv head, dimension):

| tensor | channel RMS min/median/max | max/median | top-8 channel variance share | cell RMS max/median |
|---|---|---:|---:|---:|
| K | 0.029 / 1.424 / 8.310 | **5.84x** | 5.4% | 1.03x |
| V | 0.043 / 0.159 / 2.164 | **13.60x** | 9.6% | 1.22x |

**Channel structure survives the Hadamard rotation** - K channels span 5.8x and V channels 13.6x in RMS -
while K's per-cell spread is flat (1.03x). So per-channel scaling has real structure to exploit on our
cache, and a per-token K scale has almost nothing, which is why per-token K measured 1.42x worse.

The same gate, run on that cache (attention output, relative RMS vs the FP16 cache):

| arm | synthetic | real cache | B/cell/head |
|---|---:|---:|---:|
| q4_0 | 2.052% | **0.888%** | 288 |
| q2 Lloyd-Max per-channel (ships) | 7.150% | **3.508%** | 160 |
| K 2-bit per-channel + V 4-bit | 5.710% | **1.911%** | 224 |
| K 4-bit + V 2-bit per-channel | 4.768% | 3.171% | 224 |
| KIVI's split (K per-channel, V per-token) | 6.226% | 5.188% | 160 |
| K per-token + V per-channel | 10.134% | 4.234% | 160 |
| 1-bit V per-channel | 7.773% | 9.682% | 128 |
| 1-bit V per-token (KIVI's V) | 7.057% | 14.427% | 114 |
| asymmetric K+V (zero-point, per-channel) | 3.777% | 3.227% | 192 |
| joint channel x token scales, K+V | 36.947% | **3.219%** | 164 |
| 16-value scale (vs 32, per-token) | 37.228% | 5.329% (beats 5.524%) | - |

Six of the synthetic-pinned rankings reverse on real data, and the gate now pins each claim with its data
source (`check_data`) so neither can be quoted as the other's law:

1. **Our shipping q2 is 2.0x better on real data than the synthetic floor said** (3.508% vs 7.150%). The
   gate was pessimistic in exactly the predicted direction.
2. **KIVI's premise is confirmed on K**: coarse K costs more than coarse V (K4+V2c 3.171% vs K2c+V4
   1.911%) - K is the tensor that needs the bits.
3. **KIVI's V finding is refuted on our cache**: V wants the per-channel scale, not per-token (V2c 3.508%
   vs V2t 5.188%; 1-bit V 9.682% per-channel vs 14.427% per-token). The reason is in the statistics -
   rotation left V with the strongest channel structure in the cache (13.6x).
4. **Joint channel x token scales, terrible on synthetic data (36.9%), are competitive on real data
   (3.219%)** - on flat data the channel factor is estimator noise, on real data it is signal.
5. **The zero-point lever shrinks to 1.09x** on real data (3.227% vs 3.508%) - real, modest, not the 1.89x
   the synthetic probe showed.
6. **The 16-value scale beats 32 on real data** (5.329% vs 5.524% on the per-token arm): block size is a
   data question, not a law.

Best measured point: **K 2-bit per-channel + V 4-bit = 1.911% at 224 B per cell and head (448 B per cell
for 2 KV heads, 22% under q4_0) at 2.2x q4_0's attention error.** That is the shape of the next format
candidate: asymmetric bits, not a symmetric 2-bit block.

## SKVQ's channel reordering, and the 2.5-bit K / 1.5-bit V widths it was published with

SKVQ (arXiv 2405.06219) is the paper whose mechanism the published "near lossless at 2 bits" rests on:
sort each tensor's channels by RMS so a per-token group of 32 consecutive channels is homogeneous, then
quantize K per-channel at 2 bits and V per-token at 1.5. The gate now implements the reorder
(`channel_order` + `apply_channel_order`, host-side, computed from the cache under test) and the
fractional-bit codebooks (generic Lloyd-Max at any level count; 6 levels ~ 2.58 bits, 3 levels ~ 1.58
bits; a 2.5-bit average arm spends 6 levels on the loud half of the channels and 4 on the quiet half).

**The mechanism is real on our cache - and it does not make the scheme competitive.**

Reordering V's 3-level per-token quantization (attention output, fp16 K):

| cache | natural | reordered | gain |
|---|---:|---:|---:|
| synthetic | 4.031% | 4.143% | 0.97x (noise; i.i.d. channels have no RMS order to sort) |
| real L3 | 4.939% | 3.471% | **1.42x** |
| real L11 | 6.550% | 3.540% | **1.85x** |
| real L23 | 8.254% | 5.609% | **1.47x** |
| real L47 | 13.232% | 12.926% | 1.02x |

The reorder exploits exactly the structure the real-cache dump showed (V channel RMS max/median up to
13.6x surviving the rotation), and it is a provable no-op for per-channel K - the gate pins it
bit-for-bit: quantizing a reordered cache per-channel and unpermuting gives the identical array, because
a channel permutation relabels scales, it never re-estimates them.

But the schemes built on it lose to what we already ship, at equal or MORE bytes:

| arm | B/cell/head | L3 | L11 | L23 | L47 |
|---|---:|---:|---:|---:|---:|
| q4_0 | 288 | 0.888 | 0.740 | 0.941 | 1.040 |
| **q2c (shipping 2-bit)** | **160** | **3.508** | **3.471** | **3.655** | **2.948** |
| **q2k (shipping mixed)** | **224** | **1.910** | **2.002** | **2.211** | **1.648** |
| K2c + V3t-reordered (SKVQ's own budget, 160 B) | 160 | 3.887 | 4.109 | 6.273 | 13.920 |
| K6c + V3t-reordered (2.58-bit K) | 192 | 3.993 | - | 6.228 | 12.560 |
| K2.5-mixed + V3t-reordered (2.5-bit avg) | 176 | 4.261 | - | 6.770 | 13.250 |
| K6ac + V3t-reordered (zero-point 2.58-bit K) | 208 | 3.676 | 3.597 | 6.097 | 12.775 |
| K2ac + V4 (zero-point K on q2k) | 240 | 2.165 | 2.161 | 2.938 | **1.305** |

Three findings:

1. **SKVQ's byte-tied split loses to q2c on every real layer** (3.887 vs 3.508, 4.109 vs 3.471, 6.273 vs
   3.655, 13.920 vs 2.948). The reason is the K side: at 2 bits per-channel K is the binding constraint
   (the layers where K's channel RMS spans 5.8-74.9x), and no V-side trick recovers it. On the synthetic
   cache the split actually beats q2c (6.365% vs 7.150%) - the synthetic cache has no K channel structure
   either, so nothing binds, and V3t's per-token washing wins. The synthetic gate flattered SKVQ; the
   real cache is where the ranking was decided.
2. **The fractional-bit widths do not buy back their bytes.** 6-level K is 1.7x better per element than
   4-level (19.2% vs 32.0% on L3) but the attention output barely moves (3.429% vs 3.508% at 192 B):
   K's quantization error under the softmax is largely common-mode, so finer codebooks buy less than the
   per-element RMS says. Every K6c/K2.5-mix/K6ac combo lands behind q2c at fewer bytes.
3. **Zero-point on K is LAYER-dependent**: the min/max grid loses to Lloyd-Max on the well-behaved layers
   (2.165 vs 1.910 on L3, 2.161 vs 2.002 on L11, 2.938 vs 2.211 on L23) and wins on the extreme-outlier
   layer L47 (1.305 vs 1.648, 1.26x), where min/max's range-fitting pays for itself. Not a format change;
   a per-layer choice if we ever want it.

So the honest answer to "SKVQ was the mechanism for near lossless": the mechanism transfers (the reorder
genuinely reorganizes our rotated cache's V structure), the published widths do not - our per-channel
placements dominate every SKVQ-family arm at equal or fewer bytes on every real layer measured, and the
shipping q2k remains the best point under q4_0.

The window overlay is *written* by the prefill (`prefill.cpp:1504`, `2785`) and *read* by the decode
attention. It therefore cannot change teacher-forced prompt-position logits, and the logit harness
measures the prefill path. Measured, 890-token document, logit level:

| arm | top-1 vs fp16 | mean KL | nats/token vs fp16 |
|---|---:|---:|---:|
| q2 | 79.2% | 0.2329 | +0.111 +- 0.025 |
| q2 + window 128 | 77.6% | 0.2335 | +0.143 +- 0.026 |
| q2 + window 512 | 78.3% | 0.2317 | +0.132 +- 0.028 |
| q4_0 + window 512 | 92.5% | 0.0295 | -0.001 +- 0.009 |

No window effect at 128 or 512 - by construction, not by luck. The window's benefit is a decode-path
effect (the 5K recall probe: 2/3 short answers plain, 3/3 with the window). Measuring it at the logit
level needs decode-path logits: greedy-generate N tokens per arm with `--dump-logits` and compare the
per-step distributions. That harness is the next measurement step.

## Decode-path logits: the window's real domain, and what it measures

`tools/kv_precision_compare.py --decode N --repeat R` is that measurement. Each arm greedy-generates N
tokens with `--dump-logits`; row i is the distribution that produced generated token i. Arms are
compared over the **agreement prefix** (the steps where both emitted the same token - after a greedy
divergence the two arms are conditioned on different contexts), and **KL @ 0** is reported separately
because it is always comparable: same context, the last prompt position, decode path.

**Greedy decode on this engine is not reproducible run to run.** The same command gave q4_0 a 6/64
identical prefix in one run and 1/64 in the next; across 5 repeats the prefix spread is 0-17. QSA
top-k tie-breaks and the atomic partial-group KV writes are order-sensitive, so a single run per arm
is a sample, not a measurement. Every decode-path number below is a mean over 5 runs with its spread.

890-token document, 64 generated tokens, 5 runs per arm, reference `--kv fp16`:

| arm | identical prefix (mean, min-max) | mean KL on prefix | KL @ 0 (mean, min-max) |
|---|---|---:|---|
| q4_0 | 7.4 (0-17) | 0.0462 | 0.2032 (0.105-0.307) |
| q4_0 + window 128 | 5.0 (0-9) | 0.0607 | 0.1956 (0.123-0.370) |
| q2 | 2.0 (0-10) | 0.0074 | 0.4087 (0.304-0.630) |
| q2 + window 128 | 0.0 (0-0) | 0.0000 | 0.4713 (0.334-0.605) |
| q2 + window 512 | 0.2 (0-1) | 0.1289 | 0.4764 (0.362-0.645) |

Two findings:

1. **The window has no logit-level effect on the decode path either.** q4_0 0.2032 vs 0.1956, q2 0.4087
   vs 0.4713 - inside the run-to-run spread, and the prefix means (7.4 vs 5.0, 2.0 vs 0.0) are noise at
   n=5. The single-run 42/64 that looked like a window win in the first sweep did not survive repeats.
   The window's measured benefit stays behavioral: the 5K recall probe's short answers, 2/3 plain, 3/3
   with the window.
2. **The decode path is where KV quantization actually bites.** q4_0's KL at the last prompt position
   through the decode path is 0.2032 nats - 7x its teacher-forced mean KL of 0.0276. Prefill runs full
   chunk attention; the first decode step runs sparse QSA attention over the quantized bulk, and that
   is where the quantized cache is read. A format that looks like the noise floor in prefill is not at
   the noise floor in decode, which is why the recall probe - not prefill perplexity - is the bar that
   tracks what users see.

## `--kv q2k`: the asymmetric point, implemented (K 2-bit per-channel + V 4-bit)

The table above ended with "the shape of the next format candidate: asymmetric bits, not a symmetric
2-bit block". `--kv q2k` is that shape, wired to the two formats that already exist: K goes through the
q2 per-channel path (kv_q2.hpp - the 10-byte slot per (32-cell group, KV head, dimension), 4-level
Lloyd-Max) and V through the q4_0 path (kv_q4.hpp - the rotated 18-byte block per 32 values). No new
quantizer, no new kernel: a mixed-width dispatch, the same shape `--kv k8v4` is (INT8 K + rotated Q4_0 V).

**Bytes.** 80 (K) + 144 (V) = 224 B per cell and KV head, **448 B per cell for 2 KV heads** - against
q2's 320 and q4_0's 576. 40% more than q2, 22% less than q4_0.

**Attention output** (the gate's probe, relative RMS vs the FP16 cache): **1.911% on the real dump,
5.710% on the synthetic cache**, against the shipping q2's 3.508% / 7.150% and q4_0's 0.888% / 2.052%.
Spending the bits on V is the win because the K side is the sensitive one on a real cache (K4+V2c
measured 3.171%, worse than K2c+V4's 1.911%).

**What the format is, stated as a mixed format.** `QsaState::kv_q2k` is its own flag - not `kv_q2`, not
`kv_q4` - and `qsa_kv_format()` answers `kKvQ2K` (5). The block movers, the snapshot marker and the
attention readers all take that answer. Naming it "q2" would tell a mover V's 18-byte q4_0 blocks are
10-byte per-channel slots; naming it "q4" would tell it K's slots are q4_0 blocks. Both are silent.
The snapshot marker is 13 (5 + the +8 per-channel-layout marker the q2 format carries), so a q2k image
never restores into a q2, q4_0 or K8V4 state, nor the reverse.

**The attention reads K and V by two modes.** The fused decode kernel took one `KV_MODE` for the whole
cache; a mixed cache needs one per side, so the template is now `<K_MODE, V_MODE>` and the pairs are the
dispatch key: (0,0) fp16, (1,1) int8, (2,2) q4_0, (4,4) q2, (1,2) K8V4, (4,2) q2k. A pair with both
halves equal compiles to exactly the kernel it ran before; K8V4, which used to be the single mode 3, is
now the pair it always was. A pair the engine does not build (K q4 + V q2, say) is refused by name
rather than dispatched to the nearest one.

**Streaming and rings.** The residency unit is the q2 group (32/page_size = 8 pages), because the K
side's per-channel scale is shared by the whole group and the group is resident together or not at all.
The V side's q4_0 blocks are self-contained and get a coarser residency than they need, which is
harmless - the unit is set by the most granular half. The drafter's ring keeps q2k (unlike k8v4, whose
drafter falls back to plain INT8), so its ring table is the group-aligned one.

**The gate's device arm.** `kv_q2_parity` fills K through the q2 group kernels and V through the q4
kernels, reads each half back with its own gather kernel, and measures the attention output off those
device pools - the engine's own calls, not a host composition. It pins three things: the arm lands on
the host K2c + V4 row (within 2%), q2k's K is bit for bit the q2 cache's K, and q2k's V is bit for bit
the q4_0 cache's V. A mixed dispatch that re-derived either side fails there and only there.

**Measured end to end** (built with CUDA 13.0, served on the RTX 5080, six parity suites green, gate
green on the synthetic cache and on real layers 3/11/23/47 - 31 checks each):

The device arm lands on the host row exactly: **1.910% on the real dump against the host's 1.911%**,
5.710% on synthetic, and both halves are bit-for-bit the formats they claim (K = the q2 cache's K,
V = the q4_0 cache's V).

Logit level, 512 tokens, teacher-forced, reference `--kv fp16` (top-1 / mean KL / nats per token):

| text | q4_0 | q2 | **q2k** | q2k + window |
|---|---|---|---|---|
| code agent | 88.7% / 0.0522 / +0.011 | 72.7% / 0.3111 / -0.024 | **79.3% / 0.2378 / -0.055** | 79.9% / 0.2316 |
| document | 94.1% / 0.0276 / -0.017 | 79.3% / 0.1963 / +0.054 | **84.0% / 0.1346 / +0.033** | 83.6% / 0.1335 |
| chat | 94.7% / 0.0268 / +0.009 | 82.0% / 0.1759 / +0.111 | **85.5% / 0.1290 / +0.073** | 84.2% / 0.1283 |

q2k is **1.35x closer to fp16 than q2 on mean KL** on all three texts and gains 4 to 6.5 points of top-1
agreement, for 40% more bytes than q2. It is still 2.5-5x q4_0's KL: **the 2-bit K side is the reason,
and no arrangement of the bits removes that** - the logit bar is 7x stricter than the softmax probe that
ranks the formats.

5K buried-fact recall probe: mid-document 4/4, window tail 4/4, **short answers 3/3 with no window**
(plain q2 gets 2/3), 0 loops, prompt 2362 t/s, decode **137.4 t/s** (q2: 135.4, q4_0: 157.6). So q2k
buys back the behavioral loss the 2-bit cache had, at 448 B per cell against q4_0's 576 and at 87% of
q4_0's decode speed.

### q2k under the engine's append patterns (after the PR #2318 review fixes)

The device arm above appends all cells in one call - whole groups - which never runs the partial-group scale rule.
The engine does: decode appends into the group a prompt ended in, speculative verify appends 2-5-cell windows
through the batch path (each new group first written from 1-3 cells), and after a reused conversation prefix the
2048-cell prefill chunk boundaries fall mid-group. `kv_q2_parity` now runs those patterns on the real dump
(attention output vs FP16). Three scale rules for a group first written by a partial batch: **old** = the pre-fix
kernel (the n cells' sum divided by 32), **/n** = the RMS of the n cells, **shrink** = the shipping rule, the
full-group RMS with the previous group's mean square standing in for the 32 - n cells not yet written:

| layer | q4_0 | q2k one-shot | chunked (100-cell chunks) old / /n / shrink | spec verify (3-cell windows) old / /n / shrink | prompt + decode |
|---|---:|---:|---:|---:|---:|
| L3 | 0.888% | 1.910% | 3.631% / 2.076% / 2.186% | 1.978% / 1.912% / **1.892%** | 1.910% |
| L11 | 0.740% | 2.002% | 3.617% / 2.727% / **2.626%** | 2.030% / 2.011% / 2.012% | 2.002% |
| L23 | 0.941% | 2.211% | 3.767% / 2.509% / **2.161%** | 2.895% / 2.667% / **2.245%** | 2.211% |
| L47 | 1.040% | 1.648% | 1.438% / 1.346% / 1.803% | 1.457% / 1.799% / **1.720%** | 1.648% |
| synthetic | 2.052% | 5.710% | 6.164% / 5.859% / 5.823% | 5.710% / 5.710% / 5.710% | 5.710% |

Summed over the four real layers, the verify pattern's excess over one-shot is +8.0% with /n and **+1.3% with
shrink**; the chunked stress pattern is 1.4% worse with shrink than /n (mixed by layer) and 30% better than old.

- Prompt + decode lands exactly on the one-shot row (pinned).
- The fix removes most of the chunk-boundary damage (L3 3.63% -> 2.08%, about 4.1x -> 2.3x q4_0's error) and cuts
  the verify-window cost on L23 (+31% -> +21% over one-shot).
- The /n rule alone left a group first written from a few cells with a noisy small-sample RMS (verify up to +21%
  on L23); the shrinkage rule closes most of that (L23 verify 2.245% vs 2.211% one-shot). On the extreme-outlier
  layer 47 an undersized scale acts as clipping and measures better, which is why the old kernel looks better there.
- Real prefill chunks are 2048 cells, aligned from position 0, so the 100-cell chunked row is a stress case; the
  verify row is what every speculative decode does at each group boundary.

## `--kv q2t`: the top-q lever shipped, and what it bought

q2t gives the loudest quarter of K's channels a 16-level Lloyd-Max codebook (4-bit codes, 18 B per
32-cell group and channel) and keeps the rest at q2's 4-level slot - 96 B per cell and head for K,
240 with q4_0's V, **480 B per cell**. The level bitmap freezes on the first prefill batch and travels
in the snapshot; the loud channels live in their own plane, addressed by the bitmap's prefix sum.

The gate verified every piece (69 checks synthetic, 65 per real layer, all green): the freeze's loud
set is the host reference's top-1/4 bit for bit, append placement is exact to 0.0090%, the gather's
slot math is the append's placement bit-exact, the atomic single-cell path survives both planes
(0.014-0.018% fp16 threshold flips), and the lm16 codebook is the converged Gaussian Lloyd-Max fit
(0.049%). Two tables failed that fit before the third was right - one invented, one misremembered as
"the literature values" that was a different quantizer's grid; the converged fit is the authority.

**The honest gate verdict, layer-scoped**: q2t beats q2k on the typical real layers (1.02x L3, 1.23x
L11, 1.09x L23) and loses on the extreme-outlier layer 47 (0.60x, where K's channel RMS spans 74.9x
and no fixed grid tracks it); synthetic favors q2t (1.21x). The per-group-fit rows that motivated the
format (1.228% L3, 0.805% L47) are an ENCODER CEILING - 16 centroids fitted to a group's 32 samples is
degenerate - not a format; the fixed grid is what ships and what the device measures (the gate now
separates the two rows and pins the device to the fixed grid, 1.872 = 1.872%).

**End to end** (5K buried-fact probe): q2t matches release q4_0 exactly - mid 4/4, tail 4/4, short
answers 3/3 with no window, 0 loops - at 480 B/cell, decode **139.4 t/s** (q2k 137.9, q2 135.6, q4_0
157.3). The behavioral A/B is a tie with q2k; the gate's softmax probe is where the difference lives.

**Shipping call**: `--kv q2k` remains the default mixed format (same behavior, 7% fewer bytes, no
outlier-layer regression). `--kv q2t` ships as the measured alternative for typical-layer deployments
and as the base the outlier layers would need to beat - per-group codebooks reach 1.037% on layer 47,
but at a byte cost no layout here pays.

### Per-layer K clip factor (`--kv-k-clip`)

`--kv-k-clip SPEC` puts a per-layer CLIP FACTOR alpha on the K side of the q2-family per-channel scale
(`--kv q2` and `--kv q2k`): the stored K scale becomes `alpha * (the scale the rule computes)`, and the
codes are then chosen against that clipped scale (the unrounded float, as for every q2 code; the reader
dequantizes with the stored fp16 bits). alpha < 1 clips harder, alpha > 1 widens the grid. SPEC is a comma list; an item is a bare float
(the default for every QSA layer) or `L:A` (model layer L - 3, 7, ... 47, the numbering `--dump-kv` and
STRATA_KV_PROBE_LAYER use). Examples: `0.9`, `3:0.85,47:0.7`, `0.95,47:0.8`. 0.25 <= A <= 4.0; a
non-1.0 alpha is refused unless `--kv` is q2 or q2k. V is never clipped, and the MTP drafter's ring
state keeps 1.0.

The exact rule, in `kv_append_q2_group_kernel`: the rule itself runs in the RAW domain. Every stored
previous scale it reads (the pool's scale of group g-1, or the in-thread recompute of a complete g-1 in
the same launch - which produces exactly what that block stores, `f16(alpha * sqrt(ss_prev/32))`) is
converted first: `prev_raw = prev_stored * (1.0f / alpha)`. The rule computes s from ss, n and prev_raw
exactly as above - complete group, fresh partial with prev, fresh partial without prev - and the scale
that gets stored is `f16(alpha * s)`. A partial group that already carries a scale keeps it as stored
(no second alpha). With alpha = 1.0 every multiply by 1.0f is exact, so the stored bytes are
**bit-identical** to the rule without the flag; the parity gate's bitwise checks run at 1.0.

alpha is calibrated per layer with `tools/kv_realq_probe.py`: the model's own queries (`STRATA_QSA_QDUMP`),
its own QSA selections (`STRATA_QSA_DUMP`) and the fp16 cache (`--dump-kv`), the probe reproducing the
engine's attention output to ~1e-7 before any arm is scored. The random-query gate probe is NOT a
calibration source for this: on real queries the best alpha is above 1 (widen the grid), the opposite of
what the random query suggested on layer 47. Pinned in kv_q2_parity: "--kv-k-clip 0.8: the stored K
scale is alpha x the rule ...", bit for bit (mutation-tested).

**Calibration** (`tools/kv_realq_probe.py`; real queries over the engine's own selections; three texts:
an 890-token document, 6144 tokens of C++ (`kv_q2.cu`), 6144 tokens of markdown (`docs/kv-q2.md`); 256
query positions x 24 heads each; e = attention-output error / rms(V)). The optimum is interior - error
falls to alpha 1.20-1.25 and rises again from 1.30 (L7, window on: 1.00 6.00%, 1.25 4.34%, 1.50 5.39%).
Picked per layer by **leave-one-text-out** (choose on two texts, score the third) in the decode view
(8 sinks + 128-cell FP16 window); a layer gets an alpha only if the held-out gain holds on all three:

| layer | alpha | held-out gain (document / code / prose) |
|---|---:|---|
| L3 | 1.20 | -14.5% / -15.6% / -10.2% |
| L7 | 1.25 | -27.7% / -20.9% / -18.1% |
| L11 | 1.10 | -8.3% / -4.7% / -4.3% |
| L23 | 1.20 | -6.3% / -7.1% / -1.5% |
| L47 | 1.20 | -7.5% / -7.3% / -8.9% |
| L15, L19, L27, L31, L35, L39, L43 | 1.00 | not robust: a held-out pick can hurt (L31: +19.5%) |

`--kv-k-clip 3:1.20,7:1.25,11:1.10,23:1.20,47:1.20`. Without the FP16 window the gains are smaller and
only L47 (1.25: -16..-21%) and L3 (1.10: -5..-7%) hold out. These are attention-output numbers; the
logit-level (KL) and recall validation in the engine is the next gate before any default changes.

### Layer-wise mixed precision (`--kv-layer`)

`--kv-layer SPEC` overrides the `--kv` format for named QSA layers: `--kv q2k --kv-layer 7:q4_0,39:q4_0`
keeps q2k everywhere except model layers 7 and 39, which store q4_0. SPEC is a comma list of `L:FMT` -
model layer L (3, 7, ... 47, the numbering `--dump-kv` and `--kv-k-clip` use) and FMT in {q4_0, q2, q2k};
a layer listed twice takes the last. The base `--kv` must ALSO be one of q4_0 / q2 / q2k: all three are
Hadamard-rotated (`QsaState::kv_rot`), so the rotation, the FP16 window overlay and the query rotation
stay uniform across a mixed session - `int8`, `k8v4`, `q2t` and `fp16` are refused by name, on both sides
of the flag. The MTP drafter's ring state keeps the GLOBAL `--kv` format (it is not a main layer;
`mtp.cpp` passes -1). `--kv-k-clip` follows the layer: a non-1.0 alpha is allowed only on layers whose
EFFECTIVE format is q2 or q2k, and the error names the layer and its format.

The plumbing: `qsa_set_kv_layer_format(model_layer, fmt)` fills a sparse table in `layer.cpp` (mirroring
`qsa_set_kv_k_clip`), and `qsa_state_bytes` / `qsa_state_init` take the model layer, resolving the
effective format once per state. Every per-state size - the VRAM pools, the host copy, the residency
unit, the FP16 window - follows the state's format, and the session arena is a running sum of per-layer
sizes (with no override every size equals the global one, so the default path is byte-identical). The
streaming staging pool is sized by the WIDEST streamed layer, since one buffer is reused layer by layer -
and every reader gets a per-layer VIEW of it (`stage_view` in `prefill.cpp`) carrying only that layer's
format planes. The readers infer the format from WHICH pointers are set (the tensor-core prompt attention
takes q4_0 mode whenever `k_q4` is set; the decode attention checks k_q2t, k_q2, k_q4, k_q in that order),
so a pool with every format's pointer aliased to one buffer reads a q2k layer as q4_0. Measured: with the
raw aliased pool, `--kv q2k` on a 40K-token prompt (streamed, every chunk staged) generates token 0 sixteen
times; with the view it generates exactly what the pre-`--kv-layer` binary generates, for q4_0 and q2k.
Conversation snapshots already carry a PER-LAYER format marker (`qsa_kv_format(st)` + rotated +
q2_layout, `conversation_snapshot.cpp`), and `conversation_kv_validate` compares it layer by layer: a
snapshot taken with one `--kv-layer` spec refuses to restore into a session with a different spec
("incompatible K/V geometry" on the layer whose marker moved).

The measured motivation: the real-query probe (`tools/kv_realq_probe.py`) finds q2k's excess attention
error over q4_0 differs ~2.2x between layers (L7/L39 worst, L47 best), so promoting the 1-3 worst layers
to q4_0 buys the most accuracy per byte: promoting 2 of the 12 QSA layers raises the average from 448
to 469 B per cell (+4.7% of the cache). The engine's logit-level and decode-path numbers follow below.

### Engine-level results for the three levers (logit KL and decode attention)

Logit KL vs the fp16 arm, teacher-forced over the prompt (`tools/kv_precision_compare.py`, `--dump-prompt-logits`,
served args, build of this branch). Texts: the three built-in prompts (731 / 890 / 761 tokens) and two files the
clip calibration never saw, `src/kernels/cuda/kv_stream.cu` and `README.md` (6144 tokens each). Noise floor: three
separate runs of the q2k arm give identical KL on the two 6144-token texts and vary by up to 5% on the short ones.

| Text | q4_0 | q2k | +clip | +clip (no-window spec) | L7+L39 at q4_0 | L15+L47 at q4_0 | top 4 at q4_0 |
|---|---|---|---|---|---|---|---|
| code agent   | 0.0418 | 0.2079 | 0.2226 | 0.2090 | 0.1950 (-6.2%)  | 0.2033 (-2.2%)  | 0.1772 (-14.8%) |
| document     | 0.0294 | 0.1814 | 0.1824 | 0.1819 | 0.1711 (-5.7%)  | 0.1588 (-12.5%) | 0.1425 (-21.4%) |
| chat         | 0.0277 | 0.1603 | 0.1516 | 0.1604 | 0.1350 (-15.8%) | 0.1440 (-10.2%) | 0.1148 (-28.4%) |
| kv_stream.cu | 0.0231 | 0.1753 | 0.1694 | 0.1757 | 0.1511 (-13.8%) | 0.1587 (-9.5%)  | 0.1346 (-23.2%) |
| README.md    | 0.0238 | 0.1516 | 0.1548 | 0.1540 | 0.1386 (-8.6%)  | 0.1413 (-6.8%)  | 0.1182 (-22.0%) |
| mean change  |        |        | -0.2%  | +0.2%  | **-10.0%**      | -8.2%           | **-22.0%**      |

(The q2k column and the per-layer arms are one run; the clip columns another, whose q2k row matches to the noise
floor. q2k's KL here is 5-7.6x q4_0's.)

- **The clip factor buys nothing at the logit level.** Its calibrated -5% to -28% attention-error gains (the table
  above) do not reach the logits in this view. The caveat: this view runs the PREFILL attention, which never reads
  the FP16 window, so it cannot score the window-on spec at its best; the decode-path check below is where that
  shows up, and it needs the teacher-forced decode below to be conclusive.
- **Layer-wise precision is a linear dial, not a free lunch.** Promoting the two layers the real-query probe ranks
  MOST sensitive (7, 39) cuts KL 10.0%; the two it ranks LEAST sensitive (15, 47) cut 8.2% - within the noise of
  each other. Four layers (7, 39, 23, 35) cut 22.0%. Each promoted layer buys ~5% KL for +2.4% of the cache
  (q4_0 everywhere: ~-80% for +28.6%), so the per-layer ranking does not concentrate the benefit at the logit
  level. `--kv-layer` is still the cheapest bytes-for-quality knob there is; it is not a way around the trade.

Decode attention (`STRATA_QSA_DDUMP`, the first post-prompt query, its input token identical across arms; relative
error of the attention output vs the fp16 arm):

- **The FP16 overlay is exact for q2k.** With a window covering the whole prompt, the first QSA layer (L3) - the
  only layer whose inputs do not depend on the KV format - falls from 0.185 to 0.004 (731-token prompt) and from
  0.196 to 0.014 (761 tokens). A 128-cell window cuts L3's q2k error 12% / 13% / 17% on three prompts.
- **Deeper layers are not a per-layer measurement in this view**: each inherits the format's prefill differences
  through the residual stream, and past 2051 cells the indexer's top-k selects different cells per arm (single-
  query errors of 0.4-0.75 at L23+ on a 16K prompt, swinging either way). Averaged over layers and nine prompts:
  q4_0 0.087 -> 0.065 with the window; q2k 0.251 -> 0.257; q2k+L7L39 0.211 (-16%); window+clip+L7L39 0.201 (-20%).
- What is missing is a teacher-forced DECODE (every arm fed the same continuation through the decode path), which
  would give decode-path KL for the window, the clip spec and an FP16 overlay of salient cells; the engine does not
  have one yet.
