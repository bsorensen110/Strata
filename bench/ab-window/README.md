# Verify-window width A/B: 8 vs 12 vs 16 rows — 2026-11-19

Solo build-isolation A/B of `STRATA_WINDOW_MAX_T` = 8 / 12 / 16 on the **same source**
(merged `fork-main` `7782928f2a74904b51dc5129d9c747ccec47ad44`). Same prod env (new
P1/P2/P4/P5 flags **OFF**), same model/pack/args, `parallel: 1`, matched installed serve layer.

## Builds (gfx1201-only, 54 code objects each)

| arm | STRATA_WINDOW_MAX_T | binary | sha256 | size |
|---|---|---|---|---|
| w8  | 8  | `build-hip-gfx1201-w8/strata`     | `a52e7da2a5486adc008b98c871d37fe19f13f43b58256ea0b7891ee368704bae` | 61,735,680 |
| w12 | 12 | `build-hip-gfx1201-w12/strata`    | `e03b26c7717f90eb902841733dee7a3e6e3cc6667b4e18024b46fa673d589a1e` | 76,926,728 |
| w16 | 16 | `build-hip-gfx1201-p01245/strata` | `b53d8457fc4f60c5dd4fd8831d27d8a857e68496194544d99ace04f30dd0c327` | 94,943,824 |

Size grows with the cap (more wide-window kernel instantiations). All other CMake options identical to the validated w16 cache.

## Window math (source)

`kVerifyMaxT = kWindowMaxT` (build cap). `kVerifyLongT = kVerifyMaxT>8 ? kVerifyMaxT : 2*kVerifyMaxT`
→ **8-build: 16, 12-build: 12, 16-build: 16**. In solo, speculative windows hold ≤8 rows;
prompt-lookup windows (`--lookup-max-t`) reach up to `kVerifyLongT`. Past `kVerifyMaxT` the
canonical head runs **per token** (8-build for rows 9–16); ≤`kVerifyMaxT` uses the batched head.

## Arm runtime config

- w8 and w16: identical prod `user_args` (`--lookup-max-t 16 --lookup-sizes 10,12,14,16 --lookup-chain 6 --spec 8`). Both report ladder `1,2,3,4,5,6,7,8,10,12,14,16` ("windows up to 16"). This isolates the **build kernel path** (w8 per-token head for 9–16 vs w16 batched wide head) at identical runtime request.
- w12: clamped to its cap (`--lookup-max-t 12 --lookup-sizes 10,11,12`); ladder `1..8,10,11,12` ("windows up to 12"). Necessary because kVerifyLongT=12.

Every launch verified from engine.log `engine started:`: exe matches arm binary, `--draft-model …q4_0-selq4.pack`, solo (no `--batch`), `check_engine_env.py` N/N (rc 0). Env identical across arms except per-arm isolated `STRATA_DF2_CAND_DUMP`.

## Workload

Per launch: warmup → greedy parity probe (T0, 256) → 2 greedy decode trials (T0, 512) → 1 sampled trial (T0.7, 512). 3 interleaved passes (w8,w12,w16 ×3 = 9 launches). Short prompts (≈100–180 generated tokens). Decode tok/s from `N generated in X ms (Y tok/s)`; acceptance from `drafts accepted A of B`.

## Results (n=3 launches/arm)

Greedy decode tok/s (n6):
- w8: mean **63.8** (55.7–74.6), accept 0.455, **+3.9%** vs w16
- w12: mean **63.6** (55.5–72.0), accept 0.441, **+3.6%** vs w16
- w16: mean **61.4** (52.1–70.1), accept 0.434, ref

Sampled decode tok/s (n3):
- w8: mean **46.1** (40.3–49.5), **−11.2%** vs w16
- w12: mean **57.7** (47.1–65.3), **+11.2%** vs w16
- w16: mean **51.9** (39.7–58.4), ref

## Verdict

- Greedy: narrow builds (w8/w12) point-estimate ~4% faster than wide (w16), but **ranges overlap → not separable at n=3**.
- Sampled: w12 fastest (+11%), w8 slowest (−11%), w16 middle; **ranges overlap → not separable**.
- No separable winner on this workload. The narrow-build greedy edge is within noise.

## Caveats

- **Short prompts**: wide prompt-lookup windows (10–16) fire only on repetitive/long context; on short prompts they rarely fire, so this mostly measures narrow-window kernels + build overhead, not the wide-window path under load. A longer/repetitive workload that actually fills 10–16-row windows is needed to separably judge the wide build.
- Greedy whole-reply hashes non-deterministic across fresh launches within each arm (expert-placement timing); bit-parity not assessable by reply hash.
- w12 differs in runtime `--lookup-max-t` (12 vs 16) by necessity (its cap), so w12 is not a pure build-only contrast like w8-vs-w16.
- n=3/arm; overlapping ranges. More passes (n≥5–6) on a representative workload would tighten this.

## Production state

Restored to previous binary/profile (unchanged), verified keyed 200 / unkeyed 401, smoke `17*23=391` stop, binary+profile sha256 match snapshot. **No promotion.**

## Reproduce

```
# build w8/w12 (w16 already built): see build_w8_w12.sh option set, flip STRATA_WINDOW_MAX_T
python3 ab_window.py --tree <integration worktree> --plan
python3 ab_window.py --tree <integration worktree> --run --launches 3
```
Auth via `x-api-key`. Raw: `records.jsonl`; aggregate: `summary.json`; arm configs/env: `w8/w12/w16.json/.env` (0600).
