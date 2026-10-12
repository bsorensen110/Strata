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
    uint16_t d;               // fp16 scale
    uint8_t  qs[4];           // 32 2-bit codes; element j in bits [2*(j%4), +2) of qs[j/4]
};
#pragma pack(pop)             // 6 bytes -> 8 blocks per head per cell = 48 B of codes + 16 B of scales = 64 B
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

Build with CUDA 13.0 (`build-cu130.bat`) — a 13.2 build is numerically broken for this model and cannot be used
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

1. Lloyd-Max 4-level K and V in the kernel (RMS scale, fixed level table); re-measure.
2. Finer scale granularity: 16-value blocks, then KIVI's per-channel + per-token split for K.
3. Wire `--kv q2` through `layer.cpp` / `qsa_decode_attn` / `verify.cpp` / `conversation_snapshot.cpp` with the
   FP16 window and sinks on by default.
4. End-to-end suite on the real model: `q4_0` vs `q2` vs `q2`+window, exact-copy probes at 4K-131K.
5. Only after 1-4: the entropy-coded 1.5-bit V, which buys 288 B/cell (7.1x) on top.
