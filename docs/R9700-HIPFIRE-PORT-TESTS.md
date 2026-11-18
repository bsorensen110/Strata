# R9700 (gfx1201, Linux) — hipfire-port build & test checklist

Landed on `fork-main` 2026-11-17 by the hipfire-port program (audit: `warpfront/hipfire`,
Apache-2.0/MIT, attributions in `third_party/hipfire/NOTICE` and each commit's `Port note:`).
Every route is **opt-in and default-off**; a plain build/run behaves exactly as before.

## 1. Build (device-compile gate for everything Windows never saw)

```sh
cmake -G Ninja -S . -B build/hip-port -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
      -DSTRATA_BUILD_TESTS=ON -DSTRATA_PREFILL_MMQ=ON -DSTRATA_WITH_HSA=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DCMAKE_BUILD_TYPE=Release
cmake --build build/hip-port -j
```

This is the `hip-compile` preset plus tests plus HSA. Expect first-real-compilation of:
`src/hip/hsa_linux.cpp`, `tests/hip/*.cu` (device arms), `src/prefill/moe_iu8_gfx.cu`,
`moe_iu4_gfx.cu`, `src/kernels/cuda/qsa_decode_fp8_probe.cu`. Report any error verbatim —
these TUs are written-but-uncompiled on the Windows side by necessity.

## 2. Device tests (skip-77 arms now live)

```sh
ctest --test-dir build/hip-port -R "pm4_test|hip_buffer_load|hip_moe_iu8|hip_moe_iu4|hip_projection|hip_fp8|arch_defaults|moe_iu8_parity|moe_iu4_parity|hip_projection_parity|fp8_attn_parity" --output-on-failure
```

## 3. Known gfx1201 / ROCm footguns (carry from the audit)

- ROCm ≥7.14 gfx1201 VA-reuse: mapped→unmapped→remapped VA can keep FIRST backing pages.
  Strata calls `hipMemAddressFree` (`src/core/vmm.cpp`) — watch for corruption on model reload.
- Never set `ROC_GLOBAL_CU_MASK` — hangs the R9700 MES.
- Verify `hipcc` resolves to ROCm's, not a System32-style shadow DLL (Windows-only, noted for completeness).

## 4. Per-route A/B protocol (env flags; every flag ON vs OFF, same model, greedy ids byte-equal)

| Route | Env | What to measure | Success bar |
|---|---|---|---|
| P4 buffer loads | `STRATA_HIP_BUFLOAD=1` | decode tok/s, dense/QSA kernel µs | bitwise-clean; kernel-time win; log declines |
| P1 MoE IU8 W4A8 | `STRATA_HIP_MOE_IU8=1` | pp512/2048/4096 prefill tok/s | hipfire saw +11–18% prefill on their models; ours TBD; KLD-neutral |
| P2 verify tiles + IU4 | `STRATA_HIP_MOE_IU4=1` | verify-window µs at 9..=64 rows | narrow-tile lane declines logged; KLD check later |
| P5 x4 fusion + dot4 | `STRATA_HIP_MOE_X4=1`, `STRATA_HIP_DOT4_GEMV=1` | µs at 24–48 verify rows; GEMV decode | hipfire: −31–38% µs fused, bitwise per member |
| P3 FP8 attn probe | `STRATA_HIP_FP8_ATTN=1` | per-launch attn µs; `python tools/quant_quality.py run --arm current/fp8-attn` then `compare` | NULL RESULT COUNTS: declined route + logged reason is a complete answer |
| P0 PM4-IB replay | `STRATA_HIP_PM4=1` | decode tok/s vs hipGraph; capture/latch counters in stderr | fail-closed: ANY suspicion latches hipGraph fallback; if it never activates, the eligibility/latch log names why |

## 5. Engine sanity per ON state

- `pm4_test` + all parity exes green, then one greedy generation byte-identical to OFF, then KLD.
- PM4: confirm quiescence proofs (never see `Invalidated*` in steady state; a steady `LatchedFallback`
  with the printed reason is normal-but-inert, and means the +10–16% decode claim is unreachable).
- Occupancy: `hipcc --offload-arch=gfx1201 -Rpass-analysis=kernel-resource-usage` on the new
  MoE/verify kernels (36,864 B LDS tiles) — spills would change the economics.

## 6. Windows-only known-failures (not R9700 bugs, already ticketed)

`native_dense_ple_key_test` env-latch ordering under Windows; cpu-check preset needs
`-DSTRATA_ORCA_Q4KS_MMQ=OFF` (CMakeLists guard gap) — both pre-date the program.
