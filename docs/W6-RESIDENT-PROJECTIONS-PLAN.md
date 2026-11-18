# W6-resident projections: census and design plan

Status: source-only design; no implementation or GPU validation has been done. SUPERSEDED 2026-11-18 by "Resolution of open questions" at the end: for this release model the residency path is a dead end (all 156 spans are ~2.1-2.6 bpw; W6 cannot fit and the route's regression is read-byte cost, not VRAM pressure).

## Verdict

Do not overwrite native projection bytes yet. The current HIP A8 path does **not** cover every reader: `core::layer`/`session_token`, prompt GEMM, batched and side-stream verifier branches, and several solo verifier projections can still decode `native_data` as its original GGML type. Making residency safe requires W6-aware paths for those readers or a proven startup gate that excludes each one. The verifier's current X4/DOT4 dispatch alone is insufficient.

The desired transaction belongs in `NativeDense::load`: create all 156 relaxed W6 packs from the mapped GGUF host bytes before uploading any targeted tensor; only after all packs and capacity checks succeed, upload W6 to the corresponding existing native arena spans and publish W6 metadata. A failed pack leaves all 156 native uploads unchanged. Existing separate `DeviceWeight` uploads remain the off-mode behavior and the fallback behavior.

## Census

This model has 36 GDN layers and 12 QSA layers. `src/core/layout.cpp:65-75` defines the seven target suffixes and their `ne0`/`ne1`; those are 108 GDN tensors plus 48 QSA tensors, 156 total. `ne0` is K (columns), `ne1` is output rows.

| Reader / source path (verified lines) | Target tensors | Layout read or expected | W6-ready? / required action |
|---|---|---|---|
| `src/core/native_dense.cpp:164-168, 460-475, 581-608, 665-675` | All 156 | The suffix filter selects these GGUF matrices. Pass 1 sizes 256-byte-aligned slices in one contiguous device arena; `plan_arena` assigns each tensor an offset. Pass 2 currently copies the raw GGUF native blocks to that span. | **Uploader, not inference reader.** This is the intended host-pack point: pack the GGUF's host tensor bytes before the upload and put W6 bytes at the same `dst`. Publish metadata only after the full set is ready. `NativeDense` owns and frees the arena (`native_dense.cpp:296-311, 694-700`). |
| `src/core/verify.cpp:1553-1617` | All 156 when X4 and DOT4 are enabled | Current init reads each `native_data` span D2H (`1593-1595`), calls `pack_exact`, then `iu8::upload` allocates separate device storage (`1600-1604`). The loop can succeed for some tensors and refuse others. | **Not residency-safe as written.** Replace with construction of non-owning W6 views from published `WeightRef` metadata; do not D2H-repack or allocate another W6 copy. Stage the complete set transactionally. |
| `src/core/verify.cpp:2327-2378, 2426-2428, 2471-2478, 2614, 2662-2665, 2863-2891, 3190` | All GDN and QSA projections | `mm` falls through to `native_mmvq_il` or `native_mmvq` using `native_type/native_data`. Some paths also pass raw pointers to `native_mmvq` or `native_mmvq_multi_segs`. X4 currently fuses only GDN `qkv|gate` and QSA `q|k|v`; the output projections still use `mm`. | **Partial only. REQUIRED-NEW-KERNEL / dispatch changes.** DOT4 eligibility is limited to `n<=8` (`2327-2330`). X4 requires `9<=n<=64` (`2332-2342`) and only the two groups above call it. GDN `ssm_out` (`2614`) and QSA `attn_output` (`3190`) therefore fall back for widths 9-16. `hip_projection_lane` also declines batch, side, and profiling paths (`2305-2307`), which then read native bytes. Add a W6-aware single-projection path covering every target, width 1-16, and any supported graph/eager/batch/side path, or refuse residency whenever such a path can run. |
| `src/core/layer.cpp:183-199, 264-386, 1544-1730`; callers `src/core/session.cpp:218-252, 804-835, 870-892`; `src/program/generate.cpp:10770, 23708` | All GDN and QSA projections | `gemv_quantized` chooses `native_mmvq(w.native_type, w.native_data, ...)`; `gdn_layer` and `qsa_layer` call it for their native projections. `session_capture_token` captures `block_layer_pre`; `session_token` and `session_loop` are live generation paths. | **Not W6-ready; REQUIRED-NEW-KERNEL or explicit path gate.** Add a W6 branch to the shared `gemv_quantized` consumer and ensure the captured session path uses it. Verifier-only metadata does not cover this reader. |
| `src/prefill/prefill.cpp:3865-3872, 5067-5107, 5128-5131, 5476, 9034-9050, 9064-9066, 9183` and `src/prefill/gemm.cu:1287-1379, 1385-1524` | All 156 | `native_proj` passes the native GGML type and block pointer to `Gemm::native`. On HIP, that uses the type-based native MMQ when admitted, otherwise device `dequant_f16` followed by FP16 GEMM; it does not decode on the host. GDN and QSA each have two prompt/run implementations. | **Not W6-ready; REQUIRED-NEW-KERNEL.** Add a W6-aware `Gemm::native`/`native_proj` branch. Proposed smallest route: a W6-to-FP16 row-slab dequant kernel into the existing GEMM scratch, then the existing FP16 GEMM. Largest projection is 31,457,280 elements; source comments say the engine GEMM scratch holds this shape. This avoids another persistent weight copy but adds a full W6 read and FP16 staging traffic per GEMM. |
| QSA prompt attention/scoring: `src/prefill/prefill.cpp:5128-5131, 5420-5476` and the `qsa_*` attention kernels | QSA `attn_q/k/v/output` | The projections are `native_proj` calls; the scorer/attention stage consumes projected Q/K/V and attention buffers, not a `WeightRef` or native matrix pointer. The output projection is another `native_proj` after attention. | No separate raw-weight reader in QSA scorer/WMMA attention. Covered by the prompt `native_proj` row above; do not mistake attention-kernel coverage for projection coverage. |
| `src/split/src/model.cpp:815-832` (main split loader; alternate `FNX_27B` branch `678-713`) | Same suffixes, in the standalone split engine | Uses its own `Tensor`/`Stager`/`Up` objects and `up.dense(up.rows/cols/rows2)` to create stage-local matrices. It is not a core `WeightRef` path. `dense` only rearranges Q8_0 blocks into split `q`/`d` planes (`model.cpp:426-445`); the `q8()` assertion at `667-713` is in the `FNX_27B` branch, not the normal Qwen3.8 path. | **Out of scope only when gated out.** The split engine is a separate `split_fnx_run` target (`src/split/CMakeLists.txt:222-229`) with its own `--layer-split` mode (`src/split/tools/sfnx.cpp:93`). Residency must refuse `--layer-split`; do not assume these matrices share the core native arena. |
| `src/core/dflash2.cpp:227-310, 610-630` | None of the 156 target projections | DFlash2 loads its own indexed DF2P draft pack into `blob_`; `bind` reads the target embedding and native head, not target layer projections. | No target projection reader. No change needed. |
| Other repository matches: `src/qwen35/q36_engine.cu:1047-1060`, `src/qwen35/q36ref.cu:619-650`; `sycl/src/core/native_dense.cpp`; split sources; tooling/converters | Similar suffix spellings, but separate engines/backends/tools | The Qwen3.6 engine has its own `Mat` loader; SYCL has its own native loader; scripts inspect or convert GGUF. These are not the HIP gfx1201 core `WeightRef` route. | Annotate, do not modify. Keep Windows, CUDA, SYCL, Metal and separate split-engine behavior unchanged. |

Repository-wide searches for the seven suffixes plus `native_data`/`native_type` found no additional HIP core inference consumer beyond the rows above. `include/strata/core/weights.hpp:90-92` confirms the native pointer/type pair is the common metadata read by the core paths.

### Coverage conclusion for decode widths 1-16

The current verifier map is assembled in `Verifier::init` before graph capture (`verify.cpp:1553-1617`; capture is a later operation at `3897-3926`), so initialization order is suitable, but it does not establish full reader coverage. In the solo verifier lane, DOT4 can cover members up to width 8. X4 can cover the GDN input pair and QSA Q/K/V group at widths 9-16. It does **not** cover `ssm_out` or `attn_output` at widths 9-16; those call `mm` and reach native MMVQ. Batch/branch/profile paths are expressly declined by `hip_projection_lane`. The core session path has no W6 dispatch at all. These are hard safety gaps, not just missing counters.

## Tensor shapes and byte arithmetic

Geometry comes from `include/strata/core/geometry.hpp:24-41`; projection orientation comes from `src/core/layout.cpp:65-75`. For this model, GDN uses `n_embd=2560`, `ssm_conv_channels=10240`, `ssm_value_dim=6144`; QSA uses `n_head=24`, `n_head_kv=2`, `head_dim=256`. `pack_exact` stores 6 packed code bits per weight and one fp16 scale per 128 values (`src/prefill/moe_iu8_pack.cpp:121-147, 174-219`):

- W6 bytes = `elements * 6/8 + elements/128 * 2` = `elements * 49/64` = 6.125 bpw.
- `native_mmvq_weight_bytes` identifies Q6_K as 256 elements / 210 bytes (`src/kernels/cuda/native_mmvq.cu:4147-4165`), i.e. 6.5625 bpw. Its exact margin is `210/256 - 49/64 = 7/128` bytes/weight (0.4375 bit/weight), not merely the nominal 6.5-bpw shorthand.
- Every extent below is divisible by 256 and its W6 codes are 2-byte aligned, so no per-tensor scale-offset padding is needed for these particular shapes. `NativeDense` rounds each arena slice to 256 bytes (`include/strata/core/native_dense.hpp:20-55`).

| Tensor suffix | Count | Shape (`rows x K`) | W6 bytes / tensor | Q6_K bytes / tensor | Margin / tensor |
|---|---:|---:|---:|---:|---:|
| `attn_qkv.weight` | 36 | 10,240 x 2,560 | 20,070,400 (19.140625 MiB) | 21,504,000 (20.507812 MiB) | 1,433,600 (1.367188 MiB) |
| `attn_gate.weight` | 36 | 6,144 x 2,560 | 12,042,240 (11.484375 MiB) | 12,902,400 (12.304688 MiB) | 860,160 (0.820312 MiB) |
| `ssm_out.weight` | 36 | 2,560 x 6,144 | 12,042,240 (11.484375 MiB) | 12,902,400 (12.304688 MiB) | 860,160 (0.820312 MiB) |
| `attn_q.weight` | 12 | 12,288 x 2,560 | 24,084,480 (22.968750 MiB) | 25,804,800 (24.609375 MiB) | 1,720,320 (1.640625 MiB) |
| `attn_k.weight` | 12 | 512 x 2,560 | 1,003,520 (0.957031 MiB) | 1,075,200 (1.025391 MiB) | 71,680 (0.068359 MiB) |
| `attn_v.weight` | 12 | 512 x 2,560 | 1,003,520 (0.957031 MiB) | 1,075,200 (1.025391 MiB) | 71,680 (0.068359 MiB) |
| `attn_output.weight` | 12 | 2,560 x 6,144 | 12,042,240 (11.484375 MiB) | 12,902,400 (12.304688 MiB) | 860,160 (0.820312 MiB) |
| **156 tensors total** | **156** | **2,673,868,800 elements** | **2,047,180,800 (1,952.344 MiB)** | **2,193,408,000 (2,091.797 MiB)** | **146,227,200 (139.453 MiB)** |

This is the capacity proof **if** every original span is Q6_K-sized. `NativeDense` sizes each span from its actual GGUF type, not a Q6_K assumption (`native_dense.cpp:253-261`). A 4-bit native tensor's span is smaller than the 6.125-bpw W6 payload; preflight must check actual type/bytes per tensor and refuse without changing any span if it does not fit. Do not write past a source-sized span. The source GGUF type of every one of the 156 tensors was not available from repository geometry alone.

The raw host staging needed to retain all 156 W6 `Pack`s for all-or-nothing packing is about **1,952.344 MiB** (1.91 GiB) plus vector/object overhead and a small per-call decode row. Existing native GGUF source is mapped separately. This avoids a persistent device duplicate. The supplied observed VRAM delta is 2,040 MiB, which is 87.656 MiB greater than the code-derived W6 payload; report the actual summed allocation footprint and reconcile the difference before claiming the measured saving.

## Activation and refusal policy

Default off: without `STRATA_HIP_W6_RESIDENT=1`, preserve the existing load, native `WeightRef` metadata and separate `hip_projection_weights_` uploads byte-for-byte.

When requested, activate only when **all** conditions hold:

1. HIP build with `STRATA_PREFILL_MMQ`; resolved `STRATA_HIP_MOE_X4=1`, `STRATA_HIP_DOT4_GEMV=1`, and exact `STRATA_HIP_RELAX_PACK=1`.
2. Device reports warp size 32 and exact gfx1201 (`hip_projection_capable`, `include/strata/core/arch_defaults.hpp:39-46`).
3. One full-model stage only: `lb_==0 && le_==n_layers`; no `--layer-split`, multi-GPU stage, or alternate split engine.
4. Exactly 156 matching, native-backed target `WeightRef`s and host GGUF tensors; every pack is verified W6; every packed extent fits its planned native arena span.
5. Every inference reader listed in the census has a W6 path or is provably unavailable in this run. Until session, prompt, batch/side and output-projection coverage is implemented, refuse activation rather than partially replacing weights.

For any refusal, print one and only one residency-refusal line, exactly:

`strata: HIP W6 residency refused: <reason>; keeping the current resident layout`

Aggregate the first concrete reason (for example unsupported device, split stage, missing tensor, un-packable source, or insufficient span) and do not emit one refusal per tensor. Then perform today's native uploads and allow today's separate projection-pack behavior. For non-HIP builds, the new code is compiled out; no Windows/CUDA/SYCL/Metal behavior changes.

## Brief B implementation design

### Load order, spans and ownership

1. Resolve the switch and required mode flags before `NativeDense::load`; pass an explicit residency request into that loader rather than infer it later in `Verifier::init`.
2. Use `NativeDense::load`'s existing read-only GGUF mapping (`include/strata/artifact/gguf_reader.hpp:397-475`) and tensor specs (`native_dense.cpp:340-475`). Before any targeted device upload, call `pack_exact` on all 156 host `tensor_data` buffers with relaxed packing. Validate each source extent, dimensions, `bits==6`, `symmetric_verified`, `codes + scales` layout, and W6 bytes `<=` the allocated native span. The current loader already holds the GGUF objects through upload.
3. If any of those checks/pack calls fails, discard staging and execute the existing native upload for **all** targets. No device span has been changed. If all succeed, upload each pack to `arena_base + plan.offsets[data_slice]` instead of uploading the raw GGUF bytes there. Use the existing layout: codes at the span start, scale offset `(codes_bytes+1)&~1`, fp16 scales after the code plane. Keep other slices/scratch and one-arena allocation unchanged.
4. Publish W6 metadata only after the entire set of H2D copies succeeds. An H2D failure after copies begin is a startup error (never serve a partially rewritten set); the all-or-nothing *fallback* guarantee applies to packing/capacity refusal, before writes.
5. Add explicit per-`WeightRef` layout metadata (e.g. `native_w6_resident`, scale pointer, rows/cols, and bit width 6). Keep `native_type` as the original source type only for diagnostics; a reader must test the W6 marker before interpreting `native_data` by that type. Do not rely only on `Verifier`'s pointer-keyed map: `core::layer` and prefill need the metadata too.
6. Change `hip_projection_weights_` to hold non-owning `WeightView` descriptors (codes and scales point inside the native arena), not owning `DeviceWeight`s. `NativeDense` remains the sole owner and frees the single arena; no `hipMalloc` or double-free. Keep the 156-entry view/map complete before any verifier/session graph capture. The existing per-entry init loop at `verify.cpp:1553-1617` is before capture, but it is incremental and must be replaced with a complete-set publication.

### Reader changes needed

| Reader | Minimal change shape |
|---|---|
| `src/core/verify.cpp:2327-2378, 2426-2428, 2614, 2662-2665, 2863-2891, 3190` | Add one W6 dispatch decision in the common projection helper and route every W6-marked target before any native `mmv`/`mmv_il`/segmented fallback. Keep X4 as an optimization for the current GDN/QSA input groups, not as the correctness path. Add a W6 single-projection GEMV kernel/path for output projections and uncovered 9-16 windows; batch/side/profiling branches need equivalent W6 support or an explicit refusal gate. |
| `src/core/layer.cpp:183-199` and session callers | Add W6 marker handling to `gemv_quantized`; ensure eager `session_token` and captured `session_capture_token` use the W6 route. If this path cannot be supported in Brief B, residency must be refused whenever `session_loop`/token-graph generation can be selected. |
| `src/prefill/prefill.cpp:3865-3872` and all listed GDN/QSA sites | `native_proj` branches on W6 metadata and calls a new `Gemm::w6`/equivalent. A W6-to-FP16 row-slab kernel into existing GEMM scratch is the smallest proposal; use the existing FP16 GEMM. This is a REQUIRED NEW DEVICE KERNEL, not a host-side `ggml` traits-table change. |
| QSA scorer and prompt attention kernels | No additional WeightRef change; they consume projected tensors. Their upstream and output `native_proj` calls are covered by the GEMM change. |
| Split engine / DFlash2 / alternate backend files | None for this mode. Startup gate excludes the split engine; DFlash2 uses its own draft pack; keep other backends untouched. |

**Required-kernel verdict:** a W6-resident plan is feasible only with these additions. Existing X4/DOT4 does not prove complete decode coverage, and prompt GEMM plus the core session path cannot consume W6 today. This is not a safe metadata-only/loader-only change.

### Accounting, route evidence and later GPU A/B

- On successful activation, print one startup accounting line with `156/156`, actual bytes/MiB of native spans replaced, actual W6 payload/VRAM saved, and `separate_projection_uploads=0`. On refusal, print only the specified refusal line for residency and report the existing behavior normally. Compute reported savings from actual packed byte extents and device allocation accounting, not the 2,040-MiB estimate alone.
- Add dispatch counters for W6 DOT4, W6 X4, and W6 single-member/batch paths. `STRATA_DISPATCH_REPORT` already scopes capture counts by window width and graph kind and records eager calls (`include/strata/core/dispatch_stats.hpp:1-7, 51-52, 94-100, 133-146`). A later campaign must show width bins 1 through 16 with nonzero W6 route counts for each required solo/batch graph/eager case; zero native fallback count for all 156 target matrices.
- Later, in an explicitly authorized GPU window only: compare `STRATA_HIP_W6_RESIDENT=0` (current duplicate baseline) against `=1` with identical campaign-9 harness/configuration, pinned cache slots, prompts and decode windows. Record route reports and process startup free-VRAM line. The expected direction is that the baseline's approximately 2,040-MiB projection allocation delta approaches zero, but reconcile this with the current source-derived 1,952.344-MiB W6 payload before treating the exact number as an acceptance threshold. Do not run that campaign during this source-only task.

## Risk register

| Risk | Mitigation / stop condition |
|---|---|
| One tensor is un-packable or absent | Host-pack and validate all 156 before writing any span; one refusal line and current-layout fallback. |
| W6 payload exceeds a native span | Check actual GGUF type and bytes per tensor. Refuse before writes; Q6_K arithmetic is not proof for Q4/IQ4 spans. |
| Host RAM peak | Retaining all packs for transactionality costs about 1.91 GiB plus overhead. Measure/approve RAM headroom or choose a separate transactional staging design before implementation. |
| Graph captures stale or partial metadata | Publish all W6 descriptors before any graph/session capture; no lazy map fill or allocation during capture. |
| Arena lifetime / double-free | W6 views are non-owning; `NativeDense` owns the arena for the full lifetime of all Verifiers and Sessions. |
| Decode/prompt numerical behavior | Relaxed W6 is intentionally lossy within the pack's block error bound. Require parity/quality acceptance separately; this task makes no quality claim. |
| Other operating systems/backends | Keep the switch and layout changes HIP-gfx1201-only; do not touch CUDA, SYCL, Metal, or Windows behavior. |

## Open questions (for Brief B)

1. What are the actual GGUF source types and native byte extents for all 156 tensors in the target release model? In particular, do any Q4/IQ4 tensors have spans too small for 6.125-bpw W6?
2. What accounts for the observed 2,040-MiB VRAM delta versus the calculated 1,952.344-MiB raw W6 payload (87.656 MiB difference)?
3. Is the roughly 1.91-GiB temporary host-pack peak acceptable on the supported gfx1201 machines, or is bounded/disk-backed transactional staging required?
4. Which batch, side-stream and profiling decode modes must remain enabled under residency? The current lane declines these paths; Brief B must either cover them or define and test concrete refusal gates.
5. What parity/quality tolerance and performance acceptance criteria will authorize the relaxed W6 prompt kernel and the new single-projection decode kernel?

## Resolution of open questions (2026-11-18, host-only GGUF scan) — the residency path is a dead end for this model

### Q1 answered: actual native types/spans (measured, this model)

Method: stdlib `struct` parse of the GGUF headers of all four shards under `data/models/unsloth-UD-IQ4_XS-v31/` (magic/version, `n_tensors`/`n_kv`, KV skip, then tensor-info `name / n_dims / dims(u64) / type(u32) / offset(u64)`), filtered to `blk.N.` names ending in the seven census suffixes. Cross-checks: exactly 156 matches (36 GDN x3 + 12 QSA x4), and every `dims` pair agrees with the `layout.cpp`/`geometry.hpp` shapes used above, so the offset arithmetic is verified by construction.

Result — none of the 156 projections is Q6_K or any 6-bpw type:

| ggml type id | name | bpw | count | elements | bytes |
|---:|---|---:|---:|---:|---:|
| 14 | IQ2_XS | 2.0625 | 118 | 2,065,694,720 | 532,561,920 (507.9 MiB) |
| 8 | Q2_K | 2.625 | 38 | 608,174,080 | 199,557,120 (190.3 MiB) |
| **total** | | | **156** | **2,673,868,800** | **732,119,040 (698.2 MiB)** |

(`ssm_out.weight` x36 and one each of `attn_qkv`/`attn_gate` are Q2_K; the rest, including all QSA `attn_q/k/v/output`, are IQ2_XS.)

Consequences:

- **In-span replacement is impossible for all 156 tensors.** W6 is 6.125 bpw vs 2.0625/2.625 bpw native spans: 2.80x payload-to-span, per tensor (worst examples: `attn_k` W6 = 2.97x its span; a Q2_K `attn_qkv` span is 8.20 MiB vs 19.14 MiB W6). The Q6_K capacity table above is superseded; no subset of the 156 fits, so there is no partial transactional variant either.
- "Replace native with W6 in the arena" would *shrink* nothing — keeping W6-only still grows the resident set from 698.2 MiB to 1,952.3 MiB (+1,254 MiB vs X4-off baseline). It only beats today's dual-copy X4 mode (-698 MiB), i.e. it makes the losing route cheaper rather than enabling a win.

### Q2 answered: the 2,040 vs 1,952 MiB delta

The measured VRAM delta is the 1,952.3 MiB W6 payload plus ~88 MiB of allocation granularity/overhead across the 312 separate `hipMalloc`s (codes + scales per tensor) and capture-buffer variance. No unexplained component; reconcile exactly at implementation time only if the route is ever revived.

### Q-evidence: the regression is byte cost, not VRAM pressure

Campaign 9 pinned expert residency identically on both arms (`expected 8736; actual [8736, 8736, 8736]` control and treatment, RDNA4-Ports.md campaign-9 table), so expert eviction is excluded as the cause of the X4 arm's engine cold -11.8% / warm -10.1% / 2x256 burst -5.6% (diagnostics) regression. The route covers solo decode (DOT4 at widths <=8; X4 widths 9-16 for the two fused groups), so per decode pass it reads 1,952 MiB where native MMVQ reads 698 MiB: +1,254 MiB of weight traffic/token on a bandwidth-bound path reproduces a ~10% solo-decode loss without any VRAM-pressure hypothesis. A W4 uniform layout would still read 1,314.8 MiB (1.88x native). Any layout-based route must therefore beat native MMVQ by >1.9-2.8x in compute efficiency just to break even on decode bytes, and campaign 9 measured the opposite.

### Decision

Do not implement Brief B for this model. The census's REQUIRED-NEW-KERNEL reader work (session path, prompt `native_proj`, `ssm_out`/`attn_output` 9-16, batch/side lanes) is moot because the economics fail before coverage matters: on Unsloth-UD-style quants the projections are already ~2 bpw, so a dot-friendly uniform layout is strictly a bytes-for-compute trade the decode path cannot win. Keep this census as the reusable artifact: revive the residency idea only for a model whose native projection types are >=6 bpw (e.g. Q6_K/Q5_K projections), where W6 replacement both fits spans and reduces resident bytes. The same expansion penalty predicts against the MOE_IU8 expert-repack idea on this model (experts are IQ4_XS 4.25 bpw -> 8-bit is ~1.9x expert read bytes).
