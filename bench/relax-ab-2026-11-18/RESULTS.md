# hipfire-port A/B campaign — 2026-11-17 (window 20:51–21:32 EST)

Binary under test: `build-hip-port/strata` @ fork-main `0e9a96158`, sha recorded in every
`results/*.json` (`binary_sha256`), gfx1201-only code objects, `STRATA_WITH_HSA=ON`+`STRATA_MMQ=ON`
(engine link line verified identical to the production `build-hip` config).
Control (all route pairs): the same binary, route env pinned off. Baseline pair control: the
deployed production engine `fork-0161-fc2de6a19` (sha 805c9c66…, verified). Model/pack/drafter/parallel/args
all generated from the live production profile; expert-cache slots matched (10944) in every arm — no VRAM confound.
3 rounds per pair, fresh launches, interleaved; medians reported.

## First-real-Linux-compile build repairs (were blocking any build with WITH_HSA/tests)
1. `cmake`: `STRATA_PREFILL_MMQ=ON` alone leaves `strata_mmq` absent (needs `STRATA_MMQ=ON`);
   the hipfire test targets hard-link it → configure with MMQ=ON (docs §1 already specified it).
2. `hip_prefill_decode_golden` unlinkable: `moe_fused_iq.cu` under `STRATA_FUSED_DECODE_ONLY`
   referenced `iu8::launch_sym_i8_dot4` (defined only in strata_mmq).
3. Duplicate `strata::prefill::iu8::strata_gemv_sym_i8_dot4` at `strata` link: the P5-T3 iu8 block
   sits outside the q36 TU's `fused->fused_q36` namespace rename, so both copies emitted it.
4. `STRATA_FUSED_Q36_TU` was `#define`d *after* `#include "moe_fused_iq.cu"` in
   `moe_fused_iq_q36.cu`, so any compile gate keyed on it silently fails.
Fixes (local, uncommitted at time of writing): iu8 block guarded
`#if defined(__HIPCC__) && !defined(STRATA_FUSED_DECODE_ONLY) && !defined(STRATA_FUSED_Q36_TU)`;
marker defined before the include and undefined after `moe_fused_q36.hpp` (which also keys off it);
`size_bytes()` accessor added to `Gfx12Pm4CommandBuffer` (mtp.cpp/verify.cpp called a member that
never made it into `pm4.hpp`).

## A/A floor (same binary, same env, both sides) — the bar every delta below must clear
- engine decode cold |Δ| med 6.5% max 9.5% · warm med 5.9% max 7.2%
- 2×256 burst med 3.1% max 6.4% · p4k prefill med 2.1% max 4.8%
- Greedy warm hashes differ between **every** launch, including control-vs-control: bit parity is
  unattainable in this adaptive-tier configuration (known rig property); output parity unverified per rule 9.

## Results (median of 3 rounds; sign = per-round agreement)
- **baseline** (new binary, routes off, vs production): decode warm +2.9% (3/3 +), cold −0.3% mixed,
  burst −1.4% mixed, prefill −1.5% mixed → merge is production-safe; no workload regresses past floor.
- **BUFLOAD** (`STRATA_HIP_BUFLOAD=1`): all four metrics inside floor, mixed signs; **no engagement
  evidence** in engine log or manifest → "env set / unverified"; not separable from noise.
- **PM4** (`STRATA_HIP_PM4=1`, pacing auto): all metrics inside floor; **zero PM4 log lines** — no
  activation, no latch/eligibility print. Consistent with the doc's eligibility (all-routed-experts-
  resident `ar_on`), which this pack fails (31.5 GiB of experts live outside VRAM). Declined route;
  the decline reason is inferred from the eligibility rule, **not printed by the engine** — worth an
  upstream line naming why a refused window never prints.
- **IU8** (`STRATA_HIP_MOE_IU8=1`): **all 48 MoE layers refused**: "format is not an admitted
  IQ4/Q4_0/Q6_K source" — the UD-IQ4_XS pack's expert quants aren't admitted (the route targets
  Q4/IQ4/Q6_K sources). Decode warm +3.1% (3/3 +) is inside floor on a route that never ran.
- **IU4** (`STRATA_HIP_MOE_IU4=1`): inside floor, mixed, no engagement evidence.
- **X4** (`STRATA_HIP_MOE_X4=1` + `DOT4_GEMV=1`): **156 projection refusals**: "dequant bits are not
  symmetric-exact at row 0, K128 block" — this model's weights don't meet the exact-fold contract;
  prefill −7.6% (3/3 −) is the one signal that looks systematic, but with the route refusing every
  inspected projection the mechanism is unclear — treat as unresolved (candidate: probe overhead when
  the route is armed but declining).

## Verdict
No route is adoptable on this pack/config today: IU8 and X4 are **hard-declined by format gates with
logged reasons** (a complete answer per the port protocol), PM4 never engages on a hybrid pack whose
experts are not VRAM-resident, and BUFLOAD/IU4 show no engagement evidence at all. The new binary
itself is neutral-to-slightly-positive (warm decode +2.9% 3/3) and is what production should keep
after the next promotion of `0e9a96158` (build fixes included).

Reproduce: `python3 bench/relax-ab-2026-11-18/driver.py --rounds 3` (stops production, restores +
readiness-checks on exit); per-arm JSON + texts + ctest log under `results/`.

## Open items
- Commit + push the four build fixes to fork-main (copilot handoff).
- Upstream worth asking: why PM4 refusal never prints; whether UD pack quants (IQ4_XS gate/up +
  Q5_K/IQ4_NL down) can ever be IU8-admitted; whether X4's "symmetric-exact" contract can cover our
  attention/PLE tensors.
