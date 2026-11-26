# HIP prefill dequant-side route (P2)

`STRATA_PF_DEQSIDE=1` opts HIP prefill into the existing double-buffered dequant-side route. It is off when unset or when
set to any value other than `1`, preserving the serial HIP default. CUDA keeps its existing `STRATA_DEQUANT_SIDE`
setting unchanged.

The prefill allocator already reserves a second `GEMM_SCRATCH` region when `gemm_dequant_side()` is enabled, both at
initialization and relayout (`engine/src/prefill/prefill.cpp`). `Gemm::set_dequant_side()` creates a nonblocking stream
and readiness/reuse events; `engine/include/strata/hip_compat/cuda_runtime.h` maps those CUDA-named calls to HIP stream
and event APIs. The same calls are already used by the HIP-side cuBLAS prefill helper in
`engine/src/hy4flash/run/sk_cublas_ctx.hip.h`. Static inspection therefore shows no apparent HIP stream-shim blocker.

The route is expected to preserve dequant/GEMM values and accumulation order; overlap is opt-in. The parity executable
was built only. No GPU execution or parity run was performed, so HIP bitwise parity and runtime behavior remain
unverified.
