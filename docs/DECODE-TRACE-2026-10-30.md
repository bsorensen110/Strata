# Decode-only kernel trace, warm engine — 2026-10-30

Build `build-hip/strata` (branch `integ/community-2026-10`), production-equivalent args
(`cand-bmtp.json`: MTP + `--batch 4` + `--batch-mtp`, `STRATA_FUSE_DECODE=0`), single stream, temperature 0,
`enable_thinking: false`, RDNA4 R9700 (gfx1201). Production service stayed inactive; the GPU was freed afterwards.

## Method

- Engine executable wrapped by `rocprofv3 --kernel-trace` (wrapper script, same pattern as
  `~/Strata-ab-qsa/rocprof-stats.sh`). Wrapping the Python server instead dropped most of the engine's
  records (captures of 4 MB and 5 MB against 92 MB here; the decode window had 0 dispatches in both). Those
  captures are discarded.
- Warm-up: one 16-token request. Measured: two 128-token decode requests, client-side timestamps
  (`CLOCK_MONOTONIC`) around each request.
- Window alignment: the client clock whose windows contain the dispatches (`mono`: 280,571 starts inside
  the windows; `mono_raw`: 190,655). The script picks automatically.
- Window contents: the decode request includes its own prompt prefill (about 20 tokens in one pass). This
  is small against 128 generated tokens and is not removed.
- Tools: `tools/ab/decode_window_profile.py` (per-family and per-kernel kernel-time mix),
  `tools/ab/decode_gap_profile.py` (idle-gap buckets, small-kernel share).

Results from the run: decode_a 63.9 tok/s, decode_b 63.8 tok/s **under the tracer**. Untraced, the same
config measures about 88 tok/s single (`bench.py`), so tracing costs about 27% of throughput. Every
number below is traced, and treat it as an upper bound on overhead.

## Results (256 decode tokens, 2 windows)

- Dispatches: 1,069 and 1,123 per token (mean about 1,096). The earlier engine (2026-10-02 reference)
  measured 1,094 per token.
- Median kernel duration 3.6 µs; 74% of dispatches under 10 µs (813 per token).
- GPU busy (merged kernel intervals): 66.4% of decode wall time; idle 33.6%.
- Idle decomposition:
  - Gaps under 20 µs: 1,031 ms, 76.5% of idle, 279,903 gaps (about 1,093 per token, about 3.7 µs each).
  - 20–100 µs: 16 ms, 1.2%.
  - Over 100 µs: 302 ms, 22.4%, 89 gaps (about 0.35 per token, average 3.4 ms each).
- Kernel-time mix (share of GPU kernel time; per-token dispatch count in brackets):
  - MoE expert GEMV: 39.7%, 4.13 ms/token [167]
  - Dense GEMV (mmvq/mmvf/bf16): 19.2%, 2.00 ms/token [178]
  - GDN/QSA/norm/rope: 13.1%, 1.36 ms/token [149]
  - Other: 11.8%, 1.23 ms/token [252]; of this, `gr_down_q8_fast` (all template variants) is about 41%
    (about 51 launches per token), `attn_chunk_kernel` 13.6% (8 per token), `wait_flag_ge_kernel` 4%
    (26 per token)
  - Host/device sync + copy: 5.3%, 0.56 ms/token [63]
  - Expert-row copy (mapped): 4.7%, 0.49 ms/token [26]
  - Router/gather/combine: 2.4%, 0.25 ms/token [60]
  - Activation quantize/interleave: 2.2%, 0.23 ms/token [171]
  - Expert blob fetch: 1.5%, 0.16 ms/token [26]
  - Runtime copy/fill: 0.1%

## Corrections to earlier notes

- The partial trace from the first capture showed expert-row copies at about 38% of decode kernel time.
  That capture was missing most of the records, and the figure is wrong. The complete trace puts expert
  copies and blob fetch at about 6% combined.
- Last turn's "fusing quantize is worth about 2–3%" is correct for GPU time. It is incomplete for
  launch overhead: quantize/interleave is 15% of dispatches (171 per token), so each removed launch
  matters more than its kernel time suggests.
- The "GPU idle 95%" figure from the partial capture is withdrawn. The complete trace gives 33.6% idle
  (traced).

## Interpretation

1. Decode is a chain of about 1,100 small kernels per token, most of them under 10 µs. The GPU is busy
   about two thirds of the traced time, and most idle time sits in gaps under 20 µs between launches.
   That is launch and dependency overhead, not arithmetic. The 2026-10-02 reference shows the same
   structure (42.7% idle, 52% of it in gaps under 20 µs; here 33.6% idle, 76.5% in gaps under 20 µs).
   It is a stable property of the engine, not a one-off.
2. The high-value levers are dispatch count and host round trips, not the arithmetic inside the big GEMVs:
   - `gr_down_q8_fast`: about 51 launches per token across template variants. One fused variant would
     cut most of them.
   - `wait_flag_ge` (26 per token) and `wait_copy_i32` / `doorbell` (sync and copy family, 63 per
     token): both are host-device synchronisation points. Each one is a candidate for merging.
   - Quantize/interleave (171 per token): fusing these into their producer or consumer removes launches
     even though it removes little kernel time.
   - The 89 long gaps (over 100 µs, 3.4 ms average) are host stalls, about 0.35 per token. The kernel
     before each one should be identified, as the 2026-10-02 analysis did.
3. Fused-kernel setup for the native pack (last turn) would target only the quantize/interleave part of
   this list, and only for the canonical path. The launch-count argument makes that path somewhat more
   attractive than the kernel-time share suggested, but the canonical path is not the production path.

## Caveats

- Tracing inflates wall time (about 27% throughput loss) and per-launch overhead, so both the gap total
  and the GPU-busy fraction are upper bounds. The 2026-10-02 reference found the same caveat. The true
  untraced idle share is not measured here.
- One card, one prompt, 2 windows of 128 tokens. The two windows agree closely (1,069 and 1,123 dispatches
  per token), and the earlier engine's 1,094 per token supports the count.
- The window includes prompt prefill (about 20 tokens).

## Recommendations

1. Measure the untraced split before choosing a lever. Two options: engine-side per-step timers
   (`predicted_ms`-style, as in the 2026-10-02 reference), or `hipEvent` timing around the decode step.
   Either gives the real GPU-busy share without tracer overhead.
2. Target launch count. Priority order by expected value:
   a. Fuse `gr_down_q8_fast` variants (about 51 launches per token).
   b. Merge the `wait_flag_ge` / `wait_copy_i32` / `doorbell` sync points (about 90 launches per token).
   c. Fuse activation quantize/interleave into producers or consumers (171 launches per token).
   d. Investigate the 89 long host stalls (about 300 ms per 256 tokens in the trace).
3. Do not prioritise MoE GEMV arithmetic optimisation on the strength of this trace alone. It is the
   largest kernel-time share (about 40%), but the GEMVs are already few-launch-per-expert, and the idle
   time sits between kernels.
4. Keep the tracing setup. For future captures use the engine-wrapper pattern and check dispatches per
   token against about 1,100 before trusting a capture.

## Reproduce

```sh
# 1. wrapper: rocprofv3 wraps the ENGINE binary (not the Python server)
cat > decode3/rocprof-wrap.sh <<'SH'
#!/usr/bin/env bash
exec /opt/rocm/bin/rocprofv3 --kernel-trace --output-format csv -d "$OUT" -o trace -- "$ENGINE" "$@"
SH
# 2. candidate config with exe = wrapper (mode 0600), serve.server as usual
# 3. launch in a scope, wait for /v1/models 200
# 4. warm-up request (16 tokens), then two 128-token requests; record CLOCK_MONOTONIC around each
# 5. stop the scope; wait until /dev/kfd is free
python3 tools/ab/decode_window_profile.py decode3/prof/trace_kernel_trace.csv decode3/windows.json
python3 tools/ab/decode_gap_profile.py  decode3/prof/trace_kernel_trace.csv decode3/windows.json
```

## Follow-up (same day): long-stall attribution and the production split

### Item 3: what precedes the long host stalls

From the same decode3 trace: 85 gaps of at least 100 µs inside the decode windows, 286.7 ms total
(`tools/ab/long_stall_attribution.py`).

- 80 gaps (240.5 ms, 84% of the long-gap time, about 3.0 ms each): `mtp_select_kernel` followed by
  `copy_i32_from_mapped_kernel`. The MTP selection result goes to the host through a mapped copy, so each
  speculative round has a host round trip.
- 1 gap (44.2 ms): `native_q8_1_interleave_kernel` then the IlQ6K GEMV.
- 2 gaps (1.7 ms): `kv_append_q8_kernel<true>` then the mapped copy.
- 2 gaps (0.3 ms): `copy_indexed_kernel` then a runtime copy.

Consequence for recommendation 3: the target is the MTP select to host copy round trip, not launch count.
A device-side handoff for the next draft, with the host reading ids only when PLE or the n-gram table needs
them, would remove most of the 240 ms. This is unmeasured until implemented.

Correction to recommendation 2b: `wait_flag_ge_kernel` is a one-thread spin on a flag the host writes
(`src/kernels/cuda/verify_kernels.cu`). It makes the GPU wait for the CPU expert pool. Merging it is a
pipeline change, not a fusion. The sub-agent's item (b) is scoped to that.

### Item 1: the untraced production split

The `--serve` batch path already prints a host-side split in its `strata batch:` line. It appears when a
concurrent burst goes idle, and it needs at least two concurrent requests. A single request did not print
it in my runs.

Measured on the production-equivalent config, four concurrent 256-token requests, no tracer, timed burst:

- 161 windows, 6.32 rows per window, 44.07 ms per window.
- Run 37.84 ms = GPU-reach wait 32.59 ms (74% of the window) + CPU experts 2.57 ms (overlapped).
- Commit 3.07 ms, emit 2.93 ms, adapt wait 0.23 ms.
- MTP proposals accepted: 391 of 627 (62.4%).
- Warm burst for comparison: 48.06 ms per window, GPU wait 30.01 ms, CPU experts 8.63 ms, MTP 61.0%.

Reading: the host is blocked on the GPU for about three quarters of each window, so the window is
GPU-bound from the host's side. The GPU-reach wait is an upper bound on GPU busy time inside the window,
because the GPU can spin on pool flags within it. It is not the same measure as the trace's busy share.
A per-kernel busy measure without the tracer still needs event timing inside `batch_step`, which runs on
several streams.

### Event-timer attempt: reverted

I added hipEvent GPU-span timers (`STRATA_STEP_TIMING=1`) to the generate loop and the speculative loop.
The flag reached the engine (checked in `/proc/<pid>/environ`), but no timing lines printed in either the
batch configuration or a solo configuration (`--batch` and `--batch-mtp` removed). Serve decode does not
run those loops. I reverted the change, so `integ/community-2026-10` carries no timer. Open item:
instrument `batch_step` if a true GPU span is needed.
