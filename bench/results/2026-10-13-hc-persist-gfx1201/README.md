# `STRATA_HC_PERSIST=1`: gfx12 timing compatibility and RDNA4 test

**Card:** AMD Radeon AI PRO R9700 (gfx1201, RDNA4, 32 multiprocessors), ROCm 10.2.0a20260914 nightly. Single-GPU Strata 0.1.44 run, Swift-Qwen3.8-Flash-Next IQ3_S. The engine built for gfx1201 and gfx1200.

## Compatibility fix

The unmodified PR source does not compile for gfx1200/gfx1201: `__builtin_amdgcn_s_memrealtime` needs target feature `s-memrealtime`, and raw `s_memrealtime` assembly is rejected by ld.lld. `s_memtime` is not available on these targets either.

The local fix probes `__builtin_amdgcn_is_invocable(__builtin_amdgcn_s_memrealtime)` on-device. Timer-capable cards retain the PR's `s_memrealtime` path. On gfx12, the host measures the barrier spin-loop body with 80 blocks using HIP events, applies a conservative per-iteration margin, converts the logical 100 MHz timeout units to a spin budget, then runs 79-block forced-timeout probes for the nominal 50 ms startup and 2 s runtime budgets. It corrects the event estimate from those observed durations and repeats both checks. If capability probing, calibration, or either check fails, the opt-in persistent path stays disabled.

On the R9700, calibration was 0.30–0.31 us/iteration. Across the six cold on-arms, the corrected forced-timeout probes fired after **40.2–45.2 ms** for the 50 ms check and **1.60–1.83 s** for the 2 s check. These are conservative early timeouts (about 10–20% early), not an exact hardware clock; this proxy has only been validated on the R9700. Each real startup residency check passed T=1..8 with zero timeouts, max relative difference `1.8e-7`, and the persistent kernel active.

## A/B protocol and result

One HIP Release binary, built with `CMAKE_HIP_ARCHITECTURES=gfx1201;gfx1200`, `STRATA_ENABLE_HIP=ON`, `STRATA_PREFILL_MMQ=ON`, `STRATA_MMQ_KQUANTS=ON`, `STRATA_BUILD_TESTS=ON`, `GGML_HIP=OFF`. The A/B used six alternating rounds and a cold server per arm; each arm received two ~4K and two ~32K prompts, each capped at 256 generated tokens. `STRATA_DECODE_TIMING=1` was enabled in both arms; identical salted prompt payloads were used. The persistent switch was the sole treatment variable.

The control config set `STRATA_HC_PERSIST=""` to prevent inherited environment from enabling it (`atoi("") == 0`); the systemd user manager had no ambient value. The separate `hc_persist_parity --off` test exercised the genuinely-unset default path.

| Prompt | Default, ms/verify window (median) | Persistent, ms/verify window (median) | Paired median change | Persistent faster |
|---|---:|---:|---:|---:|
| ~4K | 25.04 | 25.17 | +0.72% | 2/6 rounds |
| ~32K | 26.33 | 26.58 | +1.02% | 1/6 rounds |

Run-to-run differences were mixed and within roughly the noise seen for this workload; this RDNA4 run shows **no robust speedup** (and a small median slowdown). It does not reproduce the gfx1030 gains. Paired prompt-rate medians were -1.06% at 4K and +4.12% at 32K with wide per-round ranges; prompt throughput is not a reliable discriminator here. Average tokens/window were 2.013→2.008 (4K) and 2.236→2.277 (32K).

## Correctness and artifact provenance

- `hc_persist_parity` passed T=1..8: finite outputs, below the `1e-4` bound (maximum observed relative difference `2.46e-7`), two persistent runs bitwise equal.
- `hc_persist_parity --off` passed with the switch unset; `gr_multi_parity` passed 32/32 cases.
- All six A/B enabled arms logged the persistent kernel selected (80 blocks on 32 multiprocessors, up to T=8); all six controls did not. All arms loaded; every response reached 256 tokens; exactly four decode-timing lines per arm.
- Tested Strata binary SHA-256: `37588b5a54260a57a757b0411f0d0dbe900fb76c254a925f07c4c91af9c001f1`.
- Raw per-round logs, configs, and machine-readable results: `/home/soren/Strata-ab-hcpersist/` (`results.json`, `summary.json`, `r*-engine.log`). Retest runner: `/home/soren/Strata-ab-hcpersist/run_fix_test.sh`.

The production config was not changed. `strata.service` was restarted and `/health` returned `loaded: true` after the test. The patch is opt-in and was tested only on this gfx1201 card; smaller gfx12 devices remain unverified.
