# T3 Hook-Point Audit & Seam Analysis (`docs/pm4-t3-findings.md`)

Port note: from `warpfront/hipfire` (`Apache-2.0`) — `crates/rdna-compute/src/replay.rs` and `crates/railgun/src/plan.rs`.

## 1. Summary of Findings

During T3 implementation, static inspection of Strata's kernel launch and stream-capture architecture revealed two structural facts that affect the `launch_pdl` capture seam assumed by the initial plan:

1. **`launch_pdl` in `include/strata/kernels/pdl.hpp` wraps only 3 kernel launch sites across the entire repository**, whereas `Verifier::record_window` (`src/core/verify.cpp`) and `MtpDrafter` (`src/core/mtp.cpp`) launch ~28 distinct kernel families via raw `<<<grid, block, smem, stream>>>(...)` syntax across 15+ `.cu` files.
2. **Stream capture (`cudaStreamBeginCapture` / `cudaStreamEndCapture`) occurs in multiple recorder classes and methods**, and non-all-resident (`!ar_on()`) verify windows interleave GPU spin-wait kernels (`wait_flag_ge`) with host-thread doorbell servicing (`h_seq_`, `h_flagA_`, `h_flagB_`, `h_flag_`).

To keep T3–T5 100% fail-closed without improvising invasive edits across 20+ CUDA-visible `.cu` files:
- We wired `pm4_scope()` and `detail::record_pdl_kernargs(...)` in `include/strata/kernels/pdl.hpp` and `src/kernels/cuda/verify_kernels.cu` (compiled only under `#if defined(__linux__) && defined(__HIPCC__) && defined(STRATA_WITH_HSA)`).
- We wired `Pm4WindowCaptureSlot` and `Pm4TapeScopeGuard` at the window capture boundaries in `src/core/verify.cpp` (`Verifier::capture`) and `src/core/mtp.cpp` (`MtpDrafter::capture_round`, `MtpDrafter::capture_step`), strictly guarded by `#if defined(__linux__) && defined(__HIPCC__) && defined(STRATA_WITH_HSA)`.
- At `cudaStreamEndCapture`, `pm4_end_verifier_capture` / `pm4_end_mtp_capture` queries `hipGraphGetNodes` + `hipGraphNodeGetType(nd, &ty)` on the captured `hipGraph_t` and passes the exact count of `hipGraphNodeTypeKernel` nodes to `Pm4WindowCaptureSlot::end_capture(expected_graph_kernel_nodes)`.
- Whenever any kernel in the window is launched via raw `<<<...>>>` (bypassing `launch_pdl`), `tape.launch_count() != expected_graph_kernel_nodes` triggers `TapeStatus::InvalidatedIncompleteCapture`, **fails closed**, logs once, and **latches the window slot to `hipGraph` fallback** (`Pm4WindowState::LatchedFallback`) so there is no partial-tape execution and no retry storm.

---

## 2. Exact Call-Site Inventory

### 2.1 `launch_pdl` Call Sites (`include/strata/kernels/pdl.hpp:79`)

Only three kernel launch sites in the repository call `launch_pdl`:

1. `src/kernels/cuda/verify_kernels.cu:1934` — `gpu_stamp_kernel<true>` inside `gpu_stamp(...)`.
2. `src/kernels/cuda/native_mmvq.cu:3387` — `native_quantize_q8_1_kernel` inside `native_quantize_q8_1(...)` (single-input path; `native_quantize_q8_1_pair` at line 3403 and `native_quantize_q8_1_il` at line 3426 use raw `<<<...>>>`).
3. `src/kernels/cuda/native_mmvq.cu:1703` & `1710` — `native_mmvq_multi_kernel` inside `launch_mmvq_multi(...)`, **only** when `ncols > 1` and neither `b6_launch` (`b6_default_rows_gfx12`, lines 1689–1694), nor `native_mmvq_il` (`kIlRows_1202` / `kIl1Rows_1202` / `kIl58_1202`), nor `q8_packed` / `q6_packed` is taken. For `ncols == 1` (`T == 1` plain AR decode), `native_mmvq` dispatches `native_q4_k_mmvq_kernel<<<...>>>`, `native_small_mmvq_kernel<<<...>>>`, etc. via raw `<<<...>>>`.

### 2.2 Raw `<<<...>>>` Kernel Families in `Verifier::record_window` (`src/core/verify.cpp:1417–2943`)

All other kernels recorded during a solo verify window launch via raw `<<<...>>>`:

| Stage | Wrapper Call Site in `src/core/verify.cpp` | Kernel Definition (`.cu` file) | Launch Mechanism |
| --- | --- | --- | --- |
| Input staging | `verify.cpp:1512–1517` | `copy_i32_from_mapped`, `copy_step_pos_from_mapped` (`verify_kernels.cu`) | Raw `<<<...>>>` |
| Embedding | `verify.cpp:1524` | `embedding_gather_dev` (`verify_kernels.cu`) | Raw `<<<...>>>` |
| Stream broadcast | `verify.cpp:1525` | `broadcast_streams` (`fused_gr.cu`) | Raw `<<<...>>>` |
| Hyper-connection read | `verify.cpp:1580–1647`, `2444`, `2845–2889` | `fused_gr_read_multi` (`fused_gr.cu`) | Raw `<<<...>>>` |
| Interleaved Q8_1 quant | `verify.cpp:1735`, `2906` | `native_quantize_q8_1_il`, `native_q8_1_interleave` (`native_mmvq.cu`) | Raw `<<<...>>>` |
| Projections (MMVQ / IL / B6) | `verify.cpp:1742`, `2907`, `2914`, `2917` | `native_mmvq_il`, `native_mmvq` (`native_mmvq.cu`, `iq_kernels.cu`) | Raw `<<<...>>>` (except un-interleaved `ncols > 1` fallback) |
| GDN mixer | `verify.cpp:1932–2120` | `gdn_conv_l2_multi`, `gdn_ab_multi`, `gdn_step_norm_multi` (`verify_kernels.cu`) | Raw `<<<...>>>` |
| QSA norm + RoPE + FWHT | `verify.cpp:2255–2310`, `2411` | `native_qsa_rms_norm_rope` (`native_qsa.cu`), `fwht256_pair_inplace_cuda`, `fwht256_inplace_cuda` (`verify_kernels.cu`) | Raw `<<<...>>>` |
| QSA KV + indexer append | `verify.cpp:2320–2365` | `kv_append_int8_dev_multi` (`verify_kernels.cu`), `native_qsa_indexer_append_steps` (`native_qsa_indexer.cu`) | Raw `<<<...>>>` |
| QSA sparse select + attn | `verify.cpp:2370–2408` | `qsa_block_scores`, `qsa_block_topk`, `qsa_kv_resolve` (`qsa_select.cu`), `qsa_decode_attn_batch` (`qsa_decode_attn.cu`) | Raw `<<<...>>>` |
| QSA output gate | `verify.cpp:2420–2434` | `native_qsa_gate_quantize_q8_1` (`native_qsa.cu`) | Raw `<<<...>>>` |
| MoE router + plan | `verify.cpp:2472–2513` | `bf16_gemv_fp32_mmvf_multi` (`bf16_gemv.cu`), `native_router_top10_multi` (`native_router.cu`), `resident_plan` (`verify_kernels.cu`) | Raw `<<<...>>>` |
| Shared expert | `verify.cpp:2572` | `shared_expert_multi` (`shared_expert.cu`) | Raw `<<<...>>>` |
| Routed experts + combine | `verify.cpp:2594`, `2645`, `2739–2750` | `quantize_q8_1_rows`, `native_expert_grouped`, `native_moe_combine_multi` (`native_moe.cu`) | Raw `<<<...>>>` |
| Final GR write + argmax | `verify.cpp:2778`, `2930–2937` | `gr_write_multi` (`fused_gr.cu`), `argmax_rows` (`verify_kernels.cu`), `sample_tokens` (`sampler.cu`) | Raw `<<<...>>>` |

### 2.3 Stream-Capture Recorders Across the Codebase

Four distinct components capture `cudaGraph_t` / `hipGraph_t` instances:

1. **`Verifier` (`src/core/verify.cpp`)**:
   - `Verifier::capture(int T, std::string& err)` (`verify.cpp:3050`): captures solo window graphs `exec_[sel_long_][T]` (all-resident `ar_on()`) and `exec_nr_[sel_long_][T]` (doorbell graph with CPU/PCIe expert waits `wait_flag_ge`).
   - `Verifier::capture_commit(std::string& err)` (`verify.cpp:3216`): captures `commit_exec_`, launched in `Verifier::commit()`.
   - `Verifier::capture_batch` / `capture_commit_batch` / `capture_deferred` (`verify.cpp:4303–4714`): captures multi-sequence slot-batch graphs `exec_bm_`, `commit_bm_`, `save_bm_`, `dcommit_bm_`.
2. **`MtpDrafter` (`src/core/mtp.cpp`)**:
   - `MtpDrafter::capture_prefill(int T)` (`mtp.cpp:1522`) and `capture_prefill_dev(int T)` (`mtp.cpp:1533`).
   - `MtpDrafter::capture_round(int T, bool coupled)` (`mtp.cpp:1540`) and `capture_step(int j, bool coupled)` (`mtp.cpp:1581`), both funneling through `finish_capture(...)` (`mtp.cpp:1498`).
3. **`CapturedGraph` / `GraphRegistry` (`include/strata/core/graph.hpp`, `src/core/graph.cpp`)**:
   - Used by single-layer `LayerExecutor` (`src/core/layer.cpp`) and microbenchmarks.
4. **`Q36Engine` (`src/qwen35/q36_engine.cu`)**:
   - Standalone dense/MoE Qwen3.6 engine with its own decode graph capture.

### 2.4 Host-Doorbell Spin-Wait Constraint (`ar_on()` + `early_ple_flag_mode() == 1`)

In `Verifier::run` (`verify.cpp:3711–3760`), a window launched with `!ar_on()` (or without `flag_raised` before launch) contains device-side `wait_flag_ge` kernels that spin on mapped host memory (`m_flagA_`, `m_flagB_`, `m_flag_`) while the host thread services `h_seq_` doorbells layer by layer after `cudaGraphLaunch` returns asynchronously. Because `HsaAqlQueue::submit_and_wait_pm4_ib` is a synchronous submit-and-wait on the completion signal, PM4 replay **must only admit** `ar_on()` windows where `flag_raised` is set prior to launch (`early_ple_flag_mode() == 1`, the default on Linux). The T5 eligibility gate enforces `all_experts_resident = true` (`ar_on()`) and refuses any CPU-offload / doorbell window.

---

## 3. Recommended Seam for Full Kernel-Node Extraction

Rather than rewriting ~150 raw `<<<grid, block, smem, stream>>>` call sites across 15+ CUDA-visible `.cu` files to call `launch_pdl`, the cleanest, single-point HIP seam is **post-capture graph-node extraction** inside `pm4_end_verifier_capture` (`src/core/verify.cpp`) and `pm4_end_mtp_capture` (`src/core/mtp.cpp`):

1. `cudaStreamBeginCapture` / `cudaStreamEndCapture` already intercepts **every** `<<<...>>>` launch on `cs_` into an ordered linear chain of `hipGraphNode_t` kernel nodes (when side-stream forks `sh_fork`, `sidem`, `df_branch_` are off, which the T5 eligibility gate requires).
2. `Verifier::capture` (`verify.cpp:3084–3119`) already walks `hipGraphGetNodes(graph, nodes.data(), &nn)` and calls `hipGraphKernelNodeGetParams(nd, &kp)` + `hipKernelNameRefByPtr(kp.func, nullptr)`.
3. On ROCm/HIP, `hipKernelNodeParams` provides:
   - `kp.func`: the host function pointer (`hipKernelNameRefByPtr(kp.func, nullptr)` yields the mangled/unmangled symbol name, and `HsaExecutableCache` resolves `KernelMetadata` + `KernelPm4Metadata`).
   - `kp.gridDim`, `kp.blockDim`, `kp.dynamicSmemBytes`: the exact HIP workgroup geometry and dynamic LDS size.
   - `kp.kernelParams` / `kp.extra`: the argument pointers or packed kernarg buffer, which can be serialized using the `KernelContract`'s formal layout (or via `hipActivity`/`launch_pdl` for kernels routed through `pdl.hpp`).

Until either (a) post-capture `hipKernelNodeParams` extraction is enabled on a live ROCm 7 `gfx1201` host or (b) the remaining decode kernel launches in `record_window` are routed through `launch_pdl`, the `expected_graph_kernel_nodes` check in `Pm4Tape::finalize_recording` guarantees that any incomplete capture fails closed and latches cleanly to `hipGraph`.
