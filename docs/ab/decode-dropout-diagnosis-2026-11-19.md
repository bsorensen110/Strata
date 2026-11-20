# Production Decode Dropout Diagnosis (2026-11-19)

**Machine / GPU:** AMD Radeon AI PRO R9700 (`gfx1201`), solo production profile (`parallel=1`, `--spec 8`, `--draft dflash2`, `--lookup-max-t 16`, `--lookup-sizes 10,12,14,16`, `--lookup-chain 6`, `--draft-conf 0.35`, `--kv int8`, `--kv-resident 32768`, `--resident-budget-gib 40`).  
**Authoritative log:** `/home/soren/.local/share/strata-upstream-v0.2.10/data/logs/qwen-ud-iq4_xs-256k.production.log` (snapshot at 54,365+ lines; Era 200 `n=42` requests).  
**Machine-readable summary:** [`decode-dropout-diagnosis-2026-11-19.json`](/home/soren/Strata-upstream/docs/ab/decode-dropout-diagnosis-2026-11-19.json).

---

## 1. Executive Summary

Production decode speed dropped from **85.5–87.5 tok/s median** (Eras 197–198, `fork-0161-fc2de6a19`) to **72.8 tok/s median** (Era 200, `fork-58aefafc9`), with long-generation requests at 42–46% draft acceptance falling to **58.9–63.1 tok/s** (p10 **62.1 tok/s** vs **75.5–75.9 tok/s** pre-promotion).

### What the logs rule out (log-evidenced)
1. **VRAM expert-cache shrinkage / PCIe thrash:** Expert cache slots moved only from `10944` (`24.69–24.77 GiB`) in Eras 197–198 to `10896` (`24.60 GiB`, `-0.44%`) in Era 200 due to verify arena growing `+43.8 MiB` (`140.1 -> 183.9 MiB`). Decode expert-cache hit rate actually **improved** slightly (`94.8% / 94.7% -> 95.4%` median), PCIe routed share stayed **0.30%**, and file blob reads stayed **0**.
2. **Thermal throttling or external GPU contention:** `amd-smi` shows `strata` (`pid 283668`) running at **2985 MHz** (max clock), **238 W**, **71 °C edge / 85 °C hotspot / 79 °C mem**, with no other GPU processes in `ps aux`.

### Root cause: two compounding effects
1. **7–12% per-step GPU verify slowdown (primary):** Binning requests by `drafts accepted` ratio shows Era 200 is **7–12% slower at every single acceptance level** (e.g., `[50%, 55%)` drops from `85.2 -> 75.3 tok/s`, `-11.6%`; `[46%, 50%)` drops from `78.2 -> 70.4 tok/s`, `-10.0%`). Matched requests at `170k–225k` prompt and `42–52%` draft acceptance drop from `78.2` to `69.1 tok/s` (`-11.6%`).
2. **Lower draft acceptance + deeper context (secondary):** Draft acceptance median moved from `53.4–54.0%` (at `122k–133k` median prompt) to `50.0%` (at `195.7k` median prompt). Even within `[170k, 200k)` prompts, draft acceptance moved from `54.2%` (`n=21`, `85.1 tok/s`) to `49.9%` (`n=19`, `75.5 tok/s`).

---

## 2. Era-by-Era Log Statistics

Parsed from `qwen-ud-iq4_xs-256k.production.log` (last 8 non-empty eras; `generated in` capture used for decode tok/s):

| Era | Start (EST) | Engine | n | Dec min | Dec p10 | Dec med | Dec mean | Dec max | Draft med (p10) | Lookup win med | EC hit% (PCIe%) | Prompt med | Reused med | Gen med | Prefill med |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 193 | 2026-11-18 03:26:41 | `fork-0161-fc2de6a19` | 4 | 51.6 | 60.9 | 84.3 | 78.8 | 94.9 | 52.8% (25.2%) | 6.9% | 87.8% (1.05%) | 224,330 | 156,310 | 318 | 425.2 |
| 194 | 2026-11-18 04:28:56 | `fork-0161-fc2de6a19` | 34 | 58.6 | 71.8 | 82.7 | 85.3 | 134.7 | 50.6% (45.4%) | 3.9% | 95.0% (0.20%) | 95,136 | 94,080 | 1,130 | 812.0 |
| 195 | 2026-11-18 07:04:27 | `fork-0161-fc2de6a19` | 77 | 34.5 | 71.6 | 83.3 | 94.0 | 208.6 | 52.2% (44.2%) | 3.8% | 94.7% (0.20%) | 146,431 | 141,533 | 1,278 | 867.3 |
| 196 | 2026-11-18 16:31:52 | `fork-0161-fc2de6a19` | 9 | 69.4 | 71.6 | 76.7 | 78.6 | 101.4 | 48.9% (47.2%) | 2.5% | 95.7% (0.20%) | 239,237 | 234,737 | 1,235 | 1313.6 |
| 197 | 2026-11-18 17:37:38 | `fork-0161-fc2de6a19` | 112 | 64.3 | 75.5 | **87.5** | 92.1 | 181.7 | **54.0%** (47.6%) | 4.0% | 94.8% (0.30%) | 133,346 | 132,381 | 408 | 706.8 |
| 198 | 2026-11-19 00:26:43 | `fork-0161-fc2de6a19` | 34 | 70.9 | 75.9 | **85.5** | 94.7 | 200.0 | **53.4%** (45.8%) | 4.1% | 94.7% (0.30%) | 122,190 | 120,715 | 755 | 938.2 |
| 199 | 2026-11-19 01:28:29 | `fork-0161-fc2de6a19` | 1 | 62.1 | 62.1 | 62.1 | 62.1 | 62.1 | 33.3% (33.3%) | 0.0% | 67.9% (4.10%) | 28 | 0 | 3 | 209.7 |
| **200** | **2026-11-19 01:34:51** | **`fork-58aefafc9`** | **42** | **58.9** | **62.1** | **72.8** | **77.4** | **173.1** | **50.0%** (43.9%) | **7.7%** | **95.4%** (0.30%) | **195,727** | **194,472** | **476** | **719.3** |

---

## 3. Stage-Level & Controlled Comparison (Old vs New)

Per-window verify/CPU-expert stage timing lines (`strata batch:` / `STRATA_VERIFY_PROFILE`) are **absent** in solo production logs (telemetry flags are unset in production). Falling back to draft-acceptance-binned and prompt-matched comparisons (`gen >= 50`):

### Decode tok/s by draft acceptance ratio (`gen >= 50`, Eras 197+198 vs Era 200)

| Draft accept bin | Old n | Old dec med (mean) | New n | New dec med (mean) | Delta (med) |
|---|---:|---:|---:|---:|---:|
| `[40%, 46%)` | 12 | 71.6 (71.8) | 8 | 66.0 (67.0) | **-7.8%** |
| `[46%, 50%)` | 23 | 78.2 (78.0) | 7 | 70.4 (71.2) | **-10.0%** |
| `[50%, 55%)` | 45 | 85.2 (86.3) | 7 | 75.3 (74.0) | **-11.6%** |
| `[55%, 65%)` | 43 | 94.7 (95.6) | 10 | 88.2 (87.6) | **-6.9%** |
| `[65%, 101%)` | 22 | 117.0 (124.9) | 1 | 106.3 (106.3) | **-9.1%** |

Matched-prompt slice (`170k–225k` prompt, `gen >= 200`, `draft 42–52%`):
- **Old (`n=5`, `178k–191k`, `draft med 49.1%`):** `dec med = 78.2 tok/s`
- **New (`n=19`, `178k–220k`, `draft med 48.9%`):** `dec med = 69.1 tok/s` (**-11.6%**)

---

## 4. Era 200 Dropouts (`< 65 tok/s`, 9 of 42 requests)

Timestamps matched 1:1 from `journalctl --user -u strata.service --since "2026-11-19 01:34:51"`. No `WARNING`, `ERROR`, `evict`, `FileExpertSource: WARNING`, or non-zero `blob reads from the file` occurred in any of these windows.

1. **Req #1 (L53906, `Nov 19 01:35:39`) — 62.4 tok/s** (cold 3-token smoke): `prompt=28 (reused=0, read=28)`, `gen=3 in 48 ms`, `draft=1/3 (33.3%)`, `lookup=0/2 win`, `ec_hit=67.4% (pcie=3.6%, 76)`, `kv_vram=94.44%`.
   - `L53906: strata serve: prompt 28 tokens = 0 reused + 28 read in 134 ms (209.7 tok/s), 3 generated in 48 ms (62.4 tok/s), drafts accepted 1 of 3, 6 checkpoints`
2. **Req #2 (L53916, `Nov 19 01:35:52`) — 62.1 tok/s** (3-token smoke): `prompt=28 (reused=21, read=7)`, `gen=3 in 48 ms`, `draft=1/2 (50.0%)`, `lookup=0/2 win`, `ec_hit=66.0% (pcie=3.8%, 63)`, `kv_vram=96.26%`.
   - `L53916: strata serve: prompt 28 tokens = 21 reused + 7 read in 60 ms (116.1 tok/s), 3 generated in 48 ms (62.1 tok/s), drafts accepted 1 of 2, 6 checkpoints`
3. **Req #17 (L54090, `Nov 19 01:57:06`) — 62.2 tok/s**: `prompt=188342 (reused=188051, read=291)`, `gen=224 in 3599 ms`, `draft=126/285 (44.2%)`, `lookup=4/98 win (4.1%), 0/10 tok`, `ec_hit=96.5% (pcie=0.1%, 218)`, `kv_vram=95.52%`.
   - `L54090: strata serve: prompt 188342 tokens = 188051 reused + 291 read in 446 ms (652.0 tok/s), 224 generated in 3599 ms (62.2 tok/s), drafts accepted 126 of 285, 6 checkpoints`
   - `L54092: strata serve: decode expert cache hit rate: 96.5% (165757 hits / 171758 lookups); 218 more read by the GPU over PCIe or from another GPU (0.1% of all 171976 routed)`
   - `L54094: strata serve: resident RAM: 36.03 GiB of experts in RAM, 43997 exchanged with the VRAM tier, 0 blob reads from the file`
4. **Req #25 (L54180, `Nov 19 02:06:09`) — 60.1 tok/s**: `prompt=204395 (reused=204363, read=32)`, `gen=1864 in 31006 ms`, `draft=907/2139 (42.4%)`, `lookup=47/957 win (4.9%), 8/60 tok`, `ec_hit=95.5% (pcie=0.4%, 6295)`, `kv_vram=96.50%`.
   - `L54180: strata serve: prompt 204395 tokens = 204363 reused + 32 read in 196 ms (163.0 tok/s), 1864 generated in 31006 ms (60.1 tok/s), drafts accepted 907 of 2139, 6 checkpoints`
   - `L54182: strata serve: decode expert cache hit rate: 95.5% (1375118 hits / 1440120 lookups); 6295 more read by the GPU over PCIe or from another GPU (0.4% of all 1446415 routed)`
   - `L54189: strata serve: lookup chain: 47 of 957 windows, 8 of 60 chained drafts accepted`
5. **Req #26 (L54192, `Nov 19 02:06:40`) — 61.7 tok/s**: `prompt=207600 (reused=204390, read=3210)`, `gen=1804 in 29216 ms`, `draft=879/1945 (45.2%)`, `lookup=55/925 win (5.9%), 9/82 tok`, `ec_hit=96.8% (pcie=0.1%, 1627)`, `kv_vram=96.55%`.
   - `L54192: strata serve: prompt 207600 tokens = 204390 reused + 3210 read in 1458 ms (2201.8 tok/s), 1804 generated in 29216 ms (61.7 tok/s), drafts accepted 879 of 1945, 6 checkpoints`
   - `L54194: strata serve: decode expert cache hit rate: 96.8% (1302893 hits / 1345531 lookups); 1627 more read by the GPU over PCIe or from another GPU (0.1% of all 1347158 routed)`
6. **Req #34 (L54283, `Nov 19 02:09:27`) — 63.1 tok/s**: `prompt=220660 (reused=220342, read=318)`, `gen=1232 in 19540 ms`, `draft=673/1469 (45.8%)`, `lookup=43/559 win (7.7%), 7/63 tok`, `ec_hit=95.4% (pcie=0.3%, 2992)`, `kv_vram=96.23%`.
   - `L54283: strata serve: prompt 220660 tokens = 220342 reused + 318 read in 426 ms (747.0 tok/s), 1232 generated in 19540 ms (63.1 tok/s), drafts accepted 673 of 1469, 6 checkpoints`
   - `L54285: strata serve: decode expert cache hit rate: 95.4% (898831 hits / 941794 lookups); 2992 more read by the GPU over PCIe or from another GPU (0.3% of all 944786 routed)`
7. **Req #36 (L54306, `Nov 19 02:10:26`) — 61.7 tok/s**: `prompt=222220 (reused=222075, read=145)`, `gen=344 in 5572 ms`, `draft=168/399 (42.1%)`, `lookup=15/180 win (8.3%), 3/23 tok`, `ec_hit=96.5% (pcie=0.2%, 442)`, `kv_vram=95.52%`.
   - `L54306: strata serve: prompt 222220 tokens = 222075 reused + 145 read in 275 ms (528.2 tok/s), 344 generated in 5572 ms (61.7 tok/s), drafts accepted 168 of 399, 6 checkpoints`
8. **Req #38 (L54327, `Nov 19 02:11:02`) — 58.9 tok/s**: `prompt=225077 (reused=222548, read=2529)`, `gen=1741 in 29547 ms`, `draft=828/1868 (44.3%)`, `lookup=35/913 win (3.8%), 8/44 tok`, `ec_hit=97.5% (pcie=0.2%, 2349)`, `kv_vram=96.20%`.
   - `L54327: strata serve: prompt 225077 tokens = 222548 reused + 2529 read in 1303 ms (1940.7 tok/s), 1741 generated in 29547 ms (58.9 tok/s), drafts accepted 828 of 1868, 6 checkpoints`
9. **Req #39 (L54339, `Nov 19 02:11:47`) — 62.4 tok/s**: `prompt=235172 (reused=226817, read=8355)`, `gen=2566 in 41138 ms`, `draft=1334/2911 (45.8%)`, `lookup=88/1232 win (7.1%), 19/122 tok`, `ec_hit=97.0% (pcie=0.1%, 2701)`, `kv_vram=96.19%`.
   - `L54339: strata serve: prompt 235172 tokens = 226817 reused + 8355 read in 3960 ms (2110.1 tok/s), 2566 generated in 41138 ms (62.4 tok/s), drafts accepted 1334 of 2911, 6 checkpoints`

---

## 5. Effective Env Diff (Old vs New) & Solo-Mode Applicability

All 25 changed switches (old `fork-0161-fc2de6a19` + backup `fork-switches.conf` vs new `fork-58aefafc9` + live `fork-switches.conf` + `33f81bfd4` defaults):

- **Solo-active, every `T=2..16` window:**
  - `STRATA_MOE_LDS_NT8`: `0 -> 1` (`iq_kernels.cu:6777,6918`)
  - `STRATA_MOE_GRID_CAP`: `0 -> 64` (`iq_kernels.cu:6642,6752,6778,6919`)
  - `STRATA_MOE_BUCKET_NOSPLIT`: `0 -> 1` (`expert_source.cpp:5358`)
  - `STRATA_MOE_BUCKET_CHUNK`: `4 -> 8` (overridden to `kMaxWindowEntries` by `NOSPLIT=1`)
  - `STRATA_DRAFT_TAIL_SKIP`: `0 -> 6` (`expert_source.cpp:5182`)
  - `STRATA_BATCH_TAIL_SKIP`: `0 -> 7` (redundant with `STRATA_ROUTE_TAIL_SKIP=7`)
  - `STRATA_PCIE_FETCH_OVERLAP`: `0 -> 1`, `STRATA_STAGING_BLOBS`: `16 -> 32`, `STRATA_PCIE_GROUP_ROWS`: `16 -> 32` (`verify.cpp:978-1010`)
  - `STRATA_CPU_REDUCE_MISS`: `0 -> 1`, `STRATA_CPU_EXPERT_NO_SPLIT`: `0 -> 1` (`verify.cpp:986`, `expert_source.cpp:5723`)
- **Solo-active, wide `T=10,12,14,16` lookup-chain windows (`ncols >= 9`):**
  - `STRATA_VERIFY_WMMA_DENSE`: `0 -> 1` (`native_mmvq.cu:3841,4020,4731`)
  - `STRATA_MMVQ_IL_NARROW_ACC`: `0 -> 1` (`native_mmvq.cu:2701,4548`)
  - `STRATA_HC_Q8_16`: `0 -> 1` (`verify.cpp:3080`, `fused_gr.cu:2951`)
  - `STRATA_QSA_SCORES_16`: `0 -> 1` (`verify.cpp:2520`, `qsa_select.cu:2320`)
  - `STRATA_GDN_GROUPS_SPLIT`: `0 -> 1`, `STRATA_GDN_AB_WIDE`: `0 -> 1` (`verify_kernels.cu:1485,1535`)
- **No-op / Batch-only in solo (`parallel=1`):**
  - `STRATA_VERIFY_WMMA_MOE`: `0 -> 1` (stub `native_wmma_expert_grouped` returns `false` in `native_wmma_gfx12.cu:343-348`)
  - `STRATA_BATCH_SPLIT`: `0 -> 1` (gated on `batch_rec_`)
  - `STRATA_BATCH_DF2_ADAPTIVE_T`: `0 -> 1`, `STRATA_BATCH_DF2_TARGET_ROWS`: `8 -> 12`, `STRATA_BATCH_DF2_MAX_T`: `8 -> 16` (gated on `batch_df2`)
  - `STRATA_BATCH_ROUTE_AFFINITY`: `0 -> 1` (batch slot reordering)
  - `STRATA_BATCH_PCIE_AUTO`: `0 -> 1`, `STRATA_BATCH_PCIE_BALANCE`: `0 -> 1` (already active when `o.batch == 0`)

---

## 6. Ranked Top-3 Culprit Switches & Mitigation Plan

### #1: `STRATA_MOE_LDS_NT8=1` (`0 -> 1`)
- **Mechanism (code-verified):** In `src/kernels/cuda/iq_kernels.cu:6777,6918`, `STRATA_MOE_LDS_NT8=1` switches `native_gu_fused_kernel<TG, 8>` and `native_down_lds_kernel<TG, 8>` on **every `T=2..16` window across all 48 MoE layers** (production model uses `gu_type=18,22` and `d_type=20,42`). It doubles dynamic LDS (`sh8` = `8 * (n_embd/32) * 36` B + `res[8][64]`) and accumulator registers (`s0[8], s1[8]`).
- **Evidence & inference:** In solo `T=2..8` DFlash2 windows, most routed experts receive only 1–3 tokens (`e1 - e0 <= 3`), so `NT=8` pays double LDS/VGPR occupancy cost on `gfx1201` on 100% of layers without utilizing the extra token slots (inferred mechanism for the uniform 7–12% per-step slowdown across all acceptance bins).
- **Counter-evidence:** In batch-4 `Arm 1` (`perf-verify-wide-results-2026-11-19.md`), `MOE_LDS_NT8=1` was bundled with `HC_Q8_16`/`QSA_SCORES_16`/`GDN_*` and scored `+2.6%` at `T=12..16` (where experts have more tokens per group).
- **Mitigation:** `Environment=STRATA_MOE_LDS_NT8=0`

### #2: `STRATA_VERIFY_WMMA_DENSE=1` (`0 -> 1`)
- **Mechanism (code-verified):** In `src/kernels/cuda/native_mmvq.cu:3841,4020,4731`, when `ncols >= 9` (`T=10,12,14,16` lookup-chain windows, which execute 40–1,232 times per request in Era 200), Q8_0 and Q6_K dense projections dispatch `wmma_q8_0_kernel` / `wmma_q8_0_segs_kernel` / `wmma_q6_k_kernel` instead of `native_mmvq` / `native_mmvq_il`.
- **Evidence (A/B measured):** In `docs/ab/perf-verify-wide-results-2026-11-19.md`, `Arm 1+2` (adding `STRATA_VERIFY_WMMA_DENSE=1` + no-op `STRATA_VERIFY_WMMA_MOE=1`) **regressed R9700 throughput by -5.7% vs Arm 1** (`99.1 -> 93.5 tok/s`).
- **Counter-evidence:** Only fires on `ncols >= 9` (`T=10,12,14,16` lookup windows), not on `T=1..8` DFlash2 windows.
- **Mitigation:** `Environment=STRATA_VERIFY_WMMA_DENSE=0`

### #3: `STRATA_MOE_GRID_CAP=64` (`0 -> 64`)
- **Mechanism (code-verified):** In `src/kernels/cuda/iq_kernels.cu:6752,6778,6919`, caps `gridDim.y` at `64` instead of `cap_groups`. When `n_groups > 64` (common on `T=8..16` windows with 10 experts/token), blocks `0..(n_groups - 65)` execute a second serial loop iteration with `__syncthreads()` while remaining blocks exit, creating tail-wave imbalance.
- **Evidence & counter-evidence:** Fires on every MoE layer with `n_groups > 64`; however, when `n_groups <= 64` (many `T=2..7` windows), `gcap=64` has no effect.
- **Mitigation:** `Environment=STRATA_MOE_GRID_CAP=0`

*(Secondary draft-side suspect #4: `STRATA_DRAFT_TAIL_SKIP=6` (`0 -> 6`, `expert_source.cpp:5182`) skips rank-6 missed experts on draft-only rows, consistent with draft acceptance dropping from `53.4–54.0%` to `50.0%`. Mitigation: `Environment=STRATA_DRAFT_TAIL_SKIP=0`.)*

### 1-Request-Per-Arm Confirmation Plan (Do Not Execute Here)
1. **Control (current Era 200 env):** Run 1 long-context solo request (`~190k` reused prompt, `max_tokens=1024`, fixed prompt/seed) and record `generated in` tok/s and `drafts accepted` ratio.
2. **Arm A (`STRATA_MOE_LDS_NT8=0`):** Add `Environment=STRATA_MOE_LDS_NT8=0` to `/home/soren/.config/systemd/user/strata.service.d/fork-switches.conf`, restart `strata.service`, repeat the same 1-request probe.
3. **Arm B (`+ STRATA_VERIFY_WMMA_DENSE=0`):** Add `Environment=STRATA_VERIFY_WMMA_DENSE=0`, restart, repeat.
4. **Arm C (`+ STRATA_MOE_GRID_CAP=0`):** Add `Environment=STRATA_MOE_GRID_CAP=0`, restart, repeat.
5. **Arm D (if draft acceptance remains ~50%, `+ STRATA_DRAFT_TAIL_SKIP=0`):** Add `Environment=STRATA_DRAFT_TAIL_SKIP=0`, restart, repeat.
