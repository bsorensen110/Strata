# Fused-decode A/B and batch MTP / DFlash2 co-draft test — 2026-10-30

Branch `integ/community-2026-10`, build `build-hip/strata` (built 2026-10-29 23:38, includes commit `31be5813`).
Host: RDNA4 R9700 (gfx1201), one card. Production (`strata.service`) stayed inactive. Paiton
(`paiton-qwen38` container, Qwen3.8 on :18982) was stopped for the window at the user's request and is
**left stopped**.

## Verdicts

1. **Fused decode (`STRATA_FUSE_DECODE=1`): no measurable effect, and the fused kernels never run for
   this config.** Kernel trace over a full request: 0 of 12,181 dispatches are `quantize_images_kernel`
   or `bf16_gemv_pair_warp_kernel`. The fused branches only fire for projections that are *not* in
   native form (`layer.cpp`: `!w_qkv->native_data`, `!w_attnk->native_data`, …). The production pack is
   native, so the change in `31be5813` is inert here. The earlier claim that it closes a fusion gap for
   production was wrong.
2. **MTP co-draft (E1, `--batch-mtp` + `--draft dflash2`): not testable as built.** The engine refuses
   DFlash2 whenever `--batch-mtp` is set (`generate.cpp:6379`: "`--batch-mtp` takes the slot's second row
   for its own drafter ... the MTP drafts instead"). The DF2MTP arm therefore ran MTP-only and measured
   nothing new. Lifting that guard is a design decision (row layout), so it is left for the user.
3. **Batch DFlash2 (no co-draft) is slower than batch MTP on this workload.** 4-way aggregate 94.1 vs
   118.1 tok/s (−20%). Single-request is flat (89.7 vs 87.7, within noise). DFlash2 accepts 52.8–54.2%
   of offered proposals; MTP accepts 76.8–78.0%.

## Arms (same binary, same production args, scratch paths, port 8091)

| Arm | Difference from production args | Effective path (from engine log) |
|---|---|---|
| FUSE0 | `STRATA_FUSE_DECODE=0` | MTP + `--batch 4` + `--batch-mtp` |
| FUSE1 | `STRATA_FUSE_DECODE=1` | same as FUSE0 |
| BMTP | none (production) | MTP + `--batch 4` + 4 slot drafters |
| DF2 | `--batch-mtp` removed; `--draft dflash2 --draft-model <pack> --draft-cap 5 --stats --batch-draft-window 16 --batch-draft-depth 3` | "DFlash2 drafter on"; 4 slots × 3 proposal rows |
| DF2MTP | DF2 + `--batch-mtp` | **DFlash2 refused → MTP only** (not a co-draft test) |

Workload: `tools/ab/greedy_bench.py`. Four fixed prompts, `temperature 0`, `max_tokens 384`,
`enable_thinking: false` (without it the model spends the budget in `reasoning_content` and returns
empty `content`). Warm-up excluded. Two single-request runs, three 4-way concurrent runs. Each arm ran on
a fresh engine with empty conversation-cache and slot folders (a first pass without the reset let the
prefix cache leak across arms; those numbers were discarded).

## Results (medians; per-run values in the bench JSON)

| Arm | single tok/s | 4-way agg tok/s | 4-way wall s | acceptance |
|---|---|---|---|---|
| FUSE0 | 88.20 | 118.73 | 10.38 | — |
| FUSE1 | 87.70 | 117.94 | 10.34 | — |
| BMTP | 87.66 | 118.07 | 10.46 | MTP 499/640 (78.0%) |
| DF2 | 89.71 | 94.11 | 13.08 | DFlash2 766/1452 (52.8%), 2.82 tok/slot-round |
| DF2MTP | 87.90 | 115.96 | 10.74 | MTP 514/669 (76.8%) (DFlash2 off) |

FUSE1 − FUSE0: −0.6% single, −0.7% 4-way. Run-to-run spread within one arm is about ±3%, so this is noise.

### Output comparison

- Single requests: FUSE0 and FUSE1 produced identical text for both requests (1766 and 1989 chars).
- 4-way concurrent: outputs differ across arms, but they also differ between repeat runs of FUSE0
  alone (first divergence at chars 347, 417, 154, 369 in one pair). The engine is not bit-deterministic
  under concurrent batching at temperature 0, so the cross-arm concurrent differences are not evidence
  about fusion.

## Fused-decode dispatch evidence

`rocprofv3 --kernel-trace` on the FUSE1 engine, one 48-token request. 12,181 dispatches, 180 distinct
kernels. Fused names: `quantize_images_kernel` 0, `bf16_gemv_pair_warp_kernel` 0. The decode path runs
`native_quantize_q8_1_kernel` (545), `native_swiglu_quantize_q8_1_kernel` (263),
`native_q8_1_interleave_kernel` (406), `native_down_multi_kernel` (410), and `bf16_f32_mmvf_*`.
Counter script: `tools/ab/count_kernel_dispatch.py`. Raw CSV is in the scratch folder.

## DFlash2 batch detail (DF2 arm)

- Engine lines: `DFlash2 windows run up to 4 slots of 4 at 3 proposal rows each, of 16 rows`;
  `4 drafter slots (40.3 MiB of K/V ring each)`; drafter 1107 MiB VRAM.
- Mean window: 2.99 slots × 3.60 proposal rows; 1672 stored, 220 not offered.
- Window cost: 82–84 ms/window (CUDA0 wait 47 ms + CPU experts ~20 ms) vs 45 ms/window in BMTP.
  Wider windows cost more than the extra accepted tokens recover at 4 slots.
- VRAM was 32.4 of 32.6 GB in use during DF2, and the engine warned below 512 MiB free. That is a
  real risk for this arm on this card; the throughput numbers are measured under that pressure.

## Caveats

- One card, one binary, one prompt set, 3 concurrent runs per arm. Good enough for the verdicts above,
  not for a publishable figure.
- Greedy, `enable_thinking: false`. Thinking-on workloads were not measured.
- The E1 co-draft could not be measured, so no statement about its acceptance lift is made.
- `fusions_parity` (cited in `31be5813`) is not a build target in `build-hip` and was not run.
  The bitwise claim rests on `quantize_act_parity` (0 failures) and on the singles A/B above.

## Corrections to earlier notes

- `31be5813` and `HANDOFF-INTEG-SWEEP.md` ("PR #1097 decode operator fusions") say the change closes a
  production fusion gap. That is wrong for the native production pack (see verdict 1).
- `STRATA-VS-PAITON-SPEED-REVIEW.md` section 3 presents the same claim as a win. Superseded by this doc.
- `STRATA_FUSE_DECODE: "1"` was added to the production config in the previous turn without A/B. It has
  been reverted; the production config matches its pre-session state.

## Environment note

`docker stop paiton-qwen38` removed the container (no `docker rm` was run; the container was most likely
started with `--rm`). The image still exists by digest (`ghcr.io/eliovp/paiton-vllm-plugin@sha256:245c71d5…`)
but has lost its tag. Restarting Paiton needs its original `docker run` flags, which are not on this box.

## Reproduce

```sh
# 1. build (no GPU needed)
cd build-hip && make strata -j8
# 2. candidate configs (scratch only, mode 0600)
S=/path/to/scratch
python3 ~/.hermes/skills/software-development/strata-hip-install/scripts/candidate_config.py \
  ~/Strata/strata-swift-iq3_s.json $S/cand.json --exe $PWD/strata --port 8091 --scratch $S \
  "--set=--conversation-cache-dir=$S/conv" "--set=--expert-profile-save=$S/profile.bin"
#    then edit cand.json: env.STRATA_FUSE_DECODE "0"|"1"; for DF2 remove "--batch-mtp" from args and add
#    "--draft dflash2 --draft-model /home/soren/Strata-data/dflash2/qwen3.8-flash-next-dflash2.pack
#     --draft-cap 5 --stats --batch-draft-window 16 --batch-draft-depth 3"
# 3. launch in a scope (empty conversation/slot folders per arm)
systemd-run --user --scope --unit=arm -p MemoryMax=infinity -p MemorySwapMax=infinity -p TasksMax=infinity -- \
  bash -c "cd ~/Strata-integ && exec ~/Strata/.venv/bin/python -u -m serve.server --engine strata --config $S/cand.json --host 127.0.0.1 --port 8091 > $S/serve.out 2>&1"
# 4. bench and check the engine log for the path that actually ran
~/Strata/.venv/bin/python tools/ab/greedy_bench.py $S/cand.json $S/bench.json LABEL
grep -h "DFlash2 accepted\|MTP proposals accepted\|draft dflash2 is off" $S/engine.log $S/serve.out
# 5. kernel dispatch check (short request under rocprofv3)
/opt/rocm/bin/rocprofv3 --kernel-trace --output-format csv -d $S/prof -o trace -- <serve.server command>
~/Strata/.venv/bin/python tools/ab/count_kernel_dispatch.py $S/prof/trace_kernel_trace.csv
# 6. stop and confirm
systemctl --user stop arm.scope; fuser /dev/kfd    # no output = free
```
