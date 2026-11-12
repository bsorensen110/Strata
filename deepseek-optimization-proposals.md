# DeepSeek V4.1 (`ds4`) Decode & CPU-Lane Optimization Proposals

This document captures optimization opportunities found while auditing `src/dsv41/ds4_model.cu` and the shared CPU worker pool (`src/kernels/cpu/pool.cpp`). None of these changes have been implemented yet; each requires benchmarking on DeepSeek V4.1 workloads (CUDA / HIP / CPU hybrid offload) to confirm end-to-end gains and parity.

## 1. Eliminate per-layer, per-step heap allocations in MoE routing (`src/dsv41/ds4_model.cu:14313-14336`)

### Observation
Inside the per-layer MoE routing loop (`src/dsv41/ds4_model.cu:14313-14335`), every layer and decode window heap-allocates three vectors, including a vector of vectors sized by `l.n_exp`:

- `std::vector<Ent> ents;` (`L14313`)
- `std::vector<std::pair<int, float>> cpu_rows;` (`L14314`)
- `std::vector<std::vector<std::pair<int, int>>> per((size_t) l.n_exp);` (`L14315`)

Then for each token `t` in `0..T-1` and expert pick `i` in `0..l.k_exp-1`, `per[(size_t) e].push_back({t, i})` (`L14334`) triggers dynamic inner-vector growth across active experts on every layer step, followed by destruction at scope exit.

### Proposal
Replace the per-step heap containers with reusable scratch storage held on the workspace/session state (or fixed-capacity stack/scratch buffers bounded by `kWin * kPicks` and `l.n_exp`):
- Maintain a flat scratch bucket array indexed by expert (or active-expert head/next linked list of size `kWin * kPicks` + `head[n_exp]` cleared only for active experts via a compact `active_experts` list) so routing does **zero heap allocations** per layer/step and touches only `O(T * k_exp)` cache lines instead of `O(n_exp)` vector headers.
- Reuse capacity-reserved scratch for `ents` and `cpu_rows` across layers and steps.

---

## 2. Vectorize or GPU-offload the CPU-lane `h_part` reduction (`src/dsv41/ds4_model.cu:14417-14432`)

### Observation
After `cpu_experts(l, h_x, ents)` finishes computing CPU-routed expert outputs into `c_yd` (`L14412`), `ds4_model.cu:14417-14423` zeroes `h_part` (`T * D` floats) and reduces all `nslots` CPU expert outputs sequentially in a scalar loop on the orchestrating thread before uploading `h_part` via `cudaMemcpyAsync` (`L14430`) and running GPU `combine` (`L14432`):

```cpp
std::fill(h_part, h_part + (size_t) T * D, 0.0f);
for (int sl = 0; sl < nslots; ++sl) {
    float* o = h_part + (size_t) cpu_rows[(size_t) sl].first * D;
    const float wv = cpu_rows[(size_t) sl].second;
    const float* y = c_yd.data() + (size_t) sl * D;
    for (int e = 0; e < D; ++e) o[e] += wv * y[e];
}
```

### Proposal
Depending on `nslots * D` and PCIe / H2D cost:
- **Option A (Host SIMD / multi-threaded or token-partitioned combine):** Partition the reduction by token row (or vectorize with explicit AVX2/AVX-512 FMA intrinsics + multi-threaded worker fan-out when `nslots` is large, or accumulate directly into the first slot in-place when a token has multiple CPU slots) so `T * D` stays hot in L1/L2 cache instead of single-thread scalar `o[e] += wv * y[e]`. Note: keep summation order identical when bit-exact parity is gated, or guard any reassociation behind a benchmarked opt-in.
- **Option B (Upload + fold into GPU `combine` when H2D bandwidth permits vs H2D savings):** If `nslots <= T` (few CPU hits), copying `(nslots, D)` vs `(T, D)` differs; only offload if H2D bytes do not increase on high-`nslots` steps. Profile both `cpu_ms` (`L14428`) and `cudaMemcpyAsync` time first.

---

## 3. Overlap `d.cpu_part` H2D copy with GPU work on CUDA/HIP (`src/dsv41/ds4_model.cu:14380-14432`)

### Observation
At `L14407-14416`, GPU-parallel kick/busy helpers (`ds4_metal_kick()`, `ds4_busy_begin()`, `ds4_busy_end()`) are guarded by `#if defined(STRATA_METAL)`. On CUDA/HIP, GPU VRAM experts (`mv_n` + `swiglu` + `mv_n`, `L14380-14403`) enqueue on stream `s` before `cpu_experts(l, h_x, ents)` (`L14412`) runs synchronously on host, and `cudaMemcpyAsync(d.cpu_part, ...)` (`L14430`) uses the same stream `s`.

### Proposal
- Verify with profiler traces whether VRAM expert kernels finish before `cpu_experts` on typical offload splits; if `cpu_experts` finishes earlier or overlaps only partially, using a separate H2D copy stream (`h2d_s`) with an event wait before `combine` (`L14432`) allows `d.cpu_part` upload to overlap the tail of GPU `mv_n` instead of serializing behind it on stream `s`.

---

## 4. Enable `WAITPKG` (`_umonitor` / `_umwait`) on Clang-cl and Linux x86 builds (`src/kernels/cpu/pool.cpp:996-1015`)

### Observation
In the CPU worker park loop (`src/kernels/cpu/pool.cpp:995-1015`), hardware-monitored sleep/wake (`_umonitor` / `_umwait` under `STRATA_POOL_UMWAIT`) is gated by:

```cpp
#if defined(_MSC_VER) && defined(_M_X64) && !defined(__clang__)   // MSVC proper: clang-cl (the Windows HIP build) has no WAITPKG intrinsics by default
```

Consequently, both Windows HIP (`clang-cl`) builds and Linux GCC/Clang builds fall back to `strata_cpu_pause()` (`L1013`), burning spin power and sibling-hyperthread front-end bandwidth while waiting on `epoch_` between layers.

### Proposal
- Add a clean CMake feature test / inline-asm (or `<immintrin.h>` gated by `__WAITPKG__` / runtime CPUID check) path for `clang-cl` on Windows and Clang/GCC on Linux x86_64, while keeping `STRATA_POOL_UMWAIT` runtime-gated and CPUID-checked so CPUs without `WAITPKG` never fault with `SIGILL`.
