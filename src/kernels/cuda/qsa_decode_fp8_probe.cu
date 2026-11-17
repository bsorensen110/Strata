// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Adapted from hipfire attention_flash_fp8_e4m3_tile_gqa.gfx1201.hip.
// Strata changes: FP16 paged reads converted on the fly, GQA8/12, chunk64
// planar partials and ordinary loads. No native-FP8 cache or FP8 WMMA.
// See third_party/hipfire/{NOTICE,LICENSE-APACHE}.
#if defined(STRATA_USE_HIP)
#include "strata/kernels/qsa_decode_fp8_probe.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <atomic>
#include <cfloat>
#include <cstdio>
#include <cstdlib>

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__) && \
    __has_builtin(__builtin_amdgcn_cvt_pk_fp8_f32) && __has_builtin(__builtin_amdgcn_cvt_pk_f32_fp8)
#define STRATA_FP8_PAIR_IMAGE 1
#else
#define STRATA_FP8_PAIR_IMAGE 0
#endif

namespace strata::kernels::fp8_probe {
namespace {
constexpr int kThreads = 256;
constexpr int kMaxDevices = 64;
bool initialized = false;
bool requested = false;
Support devices[kMaxDevices];
int device_count = 0;
std::atomic<unsigned> reported[kMaxDevices]{};

void check(hipError_t e, const char* operation) {
    if (e == hipSuccess) return;
    std::fprintf(stderr, "strata fp8-attn probe: %s: %s\n", operation, hipGetErrorString(e));
    std::exit(1);
}

__global__ void pair_image(int* out) {
    *out = STRATA_FP8_PAIR_IMAGE;
}

#if STRATA_FP8_PAIR_IMAGE
using Pair = float __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float clamp_e4m3(float v) {
    return v > 448.0f ? 448.0f : v < -448.0f ? -448.0f : v;
}

__device__ __forceinline__ unsigned encode4(const uint16_t* p) {
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(p));
    const float2 b = __half22float2(*reinterpret_cast<const __half2*>(p + 2));
    unsigned u = __builtin_amdgcn_cvt_pk_fp8_f32(clamp_e4m3(a.x), clamp_e4m3(a.y), 0, false);
    return __builtin_amdgcn_cvt_pk_fp8_f32(clamp_e4m3(b.x), clamp_e4m3(b.y), u, true);
}

__device__ __forceinline__ Pair decode2(unsigned u, bool high) {
    if (high) return __builtin_amdgcn_cvt_pk_f32_fp8(u, true);
    return __builtin_amdgcn_cvt_pk_f32_fp8(u, false);
}

template <int Group, int Dim, int Tile>
__device__ void tile_body(const float* q, QsaAttnPools pools, const int32_t* ids,
                         const int32_t* step, int cap, int kv_heads, int page_size, float* scratch) {
    static_assert(Dim == 256 && Tile == 128);
    constexpr int Passes = Tile / 8 / 2;
    const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
    const int row = lane / 16, rl = lane % 16;
    const int tile = blockIdx.x, kv = blockIdx.y;
    const int chunks = (cap + kChunk - 1) / kChunk;
    const int width = min(max(step[kStepWidth], 0), cap);
    __shared__ long long rows[Tile];
    __shared__ float query[Group * Dim];
    __shared__ float scores[Group][Tile];
    __shared__ unsigned values[Tile * Dim / 4];

    if (tid < Tile) {
        const int c = tile * Tile + tid;
        long long r = -1;
        if (c < width) {
            const int cell = ids[c];
            if (cell >= 0) {
                const int page = pools.page_table[cell / page_size];
                if (page >= 0) r = (static_cast<long long>(page) * kv_heads + kv) * page_size + cell % page_size;
            }
        }
        rows[tid] = r;
    }
    for (int i = tid; i < Group * Dim; i += kThreads)
        query[i] = q[static_cast<size_t>(kv) * Group * Dim + i];
    __syncthreads();

    // One 16-lane row per key. Keep K in registers; only V needs a byte LDS plane.
    unsigned keys[Passes][Dim / 16 / 4];
#pragma unroll
    for (int p = 0; p < Passes; ++p) {
        const int t = wave * (2 * Passes) + 2 * p + row;
#pragma unroll
        for (int j = 0; j < Dim / 16 / 4; ++j)
            keys[p][j] = rows[t] >= 0 ? encode4(pools.k_pool + rows[t] * Dim + 16 * rl + 4 * j) : 0;
    }
    for (int i = tid; i < Tile * Dim / 4; i += kThreads) {
        const int t = i / (Dim / 4), d = 4 * (i % (Dim / 4));
        values[i] = rows[t] >= 0 ? encode4(pools.v_pool + rows[t] * Dim + d) : 0;
    }
    __syncthreads();

#pragma unroll
    for (int p = 0; p < Passes; ++p) {
        const int t = wave * (2 * Passes) + 2 * p + row;
        for (int h = 0; h < Group; ++h) {
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < Dim / 16 / 4; ++j) {
                const Pair lo = decode2(keys[p][j], false), hi = decode2(keys[p][j], true);
                const float* qr = query + h * Dim + 16 * rl + 4 * j;
                sum = fmaf(qr[0], lo.x, sum);
                sum = fmaf(qr[1], lo.y, sum);
                sum = fmaf(qr[2], hi.x, sum);
                sum = fmaf(qr[3], hi.y, sum);
            }
#pragma unroll
            for (int off = 8; off; off >>= 1) sum += __shfl_xor(sum, off, 16);
            if (rl == 0) scores[h][t] = rows[t] >= 0 ? sum * 0.0625f : -FLT_MAX;
        }
    }
    __syncthreads();

    float* pm = scratch + static_cast<size_t>(chunks) * kv_heads * Group * Dim;
    float* pl = pm + static_cast<size_t>(chunks) * kv_heads * Group;
    for (int half = 0; half < 2; ++half) {
        const int chunk = 2 * tile + half;
        if (chunk >= chunks) continue;
        for (int h = wave; h < Group; h += 8) {
            const int a = half * kChunk + lane, b = a + 32;
            float m = fmaxf(scores[h][a], scores[h][b]);
#pragma unroll
            for (int off = 16; off; off >>= 1) m = fmaxf(m, __shfl_xor(m, off));
            const float ea = rows[a] >= 0 && m != -FLT_MAX ? expf(scores[h][a] - m) : 0.0f;
            const float eb = rows[b] >= 0 && m != -FLT_MAX ? expf(scores[h][b] - m) : 0.0f;
            scores[h][a] = ea;
            scores[h][b] = eb;
            float l = ea + eb;
#pragma unroll
            for (int off = 16; off; off >>= 1) l += __shfl_xor(l, off);
            if (lane == 0) {
                const size_t r = (static_cast<size_t>(kv) * chunks + chunk) * Group + h;
                pm[r] = m;
                pl[r] = l;
            }
        }
    }
    __syncthreads();

    // Thread d owns output dimension d, sharing each decoded V across all heads.
    for (int half = 0; half < 2; ++half) {
        const int chunk = 2 * tile + half;
        if (chunk >= chunks) continue;
        float acc[Group] = {};
        for (int t = half * kChunk; t < (half + 1) * kChunk; ++t) {
            const Pair v = decode2(values[t * (Dim / 4) + tid / 4], (tid & 2) != 0);
            const float vd = (tid & 1) ? v.y : v.x;
#pragma unroll
            for (int h = 0; h < Group; ++h) acc[h] = fmaf(scores[h][t], vd, acc[h]);
        }
#pragma unroll
        for (int h = 0; h < Group; ++h) {
            const size_t r = (static_cast<size_t>(kv) * chunks + chunk) * Group + h;
            scratch[r * Dim + tid] = acc[h];
        }
    }
}
#endif

template <int Group>
__global__ __launch_bounds__(kThreads, 1)
void decode_tile(const float* q, QsaAttnPools pools, const int32_t* ids,
                 const int32_t* step, int cap, int kv_heads, int page_size, float* scratch) {
#if STRATA_FP8_PAIR_IMAGE
    tile_body<Group, kDim, kTile>(q, pools, ids, step, cap, kv_heads, page_size, scratch);
#else
    // The preflight image query must refuse before this body can be dispatched.
    __builtin_trap();
#endif
}
}  // namespace

bool initialize(const char* setting, std::string& error) {
    bool on = false;
    if (!resolve(setting, on)) { error = "FP8 attention probe environment requires exact 0 or 1"; return false; }
    if (initialized) {
        if (requested == on) return true;
        error = "FP8 attention probe was already frozen; restart to change it";
        return false;
    }
    requested = on;
    if (!on) { initialized = true; return true; }
    check(hipGetDeviceCount(&device_count), "device count");
    if (device_count > kMaxDevices) { error = "FP8 attention probe supports at most 64 devices"; return false; }
    int previous = 0;
    check(hipGetDevice(&previous), "current device");
    for (int d = 0; d < device_count; ++d) {
        hipDeviceProp_t prop{};
        check(hipGetDeviceProperties(&prop, d), "device properties");
        std::snprintf(devices[d].arch, sizeof(devices[d].arch), "%s", prop.gcnArchName);
        devices[d].wave = prop.warpSize;
        if (!exact_arch(prop.gcnArchName)) continue;
        check(hipSetDevice(d), "select preflight device");
        int* result = nullptr;
        int host_result = 0;
        check(hipMalloc(&result, sizeof(int)), "preflight allocation");
        pair_image<<<1, 1>>>(result);
        check(hipGetLastError(), "preflight launch");
        check(hipMemcpy(&host_result, result, sizeof(int), hipMemcpyDeviceToHost), "preflight image query");
        check(hipFree(result), "preflight free");
        devices[d].conversions = host_result == 1;
    }
    check(hipSetDevice(previous), "restore device");
    initialized = true;
    return true;
}

bool enabled() { return requested; }

Support support() {
    int d = -1;
    check(hipGetDevice(&d), "route device");
    if (!initialized || d < 0 || d >= device_count) {
        std::fputs("strata fp8-attn probe: enabled route used without pre-capture initialization\n", stderr);
        std::exit(1);
    }
    return devices[d];
}

Refusal launch_chunks(bool on, const float* q, const QsaAttnPools& pools,
                      const int32_t* ids, const int32_t* step, int64_t cap,
                      const QsaShapes& s, float* scratch, int64_t queries, void* stream) {
    if (!on) return Refusal::disabled;
    const auto device = support();
    const bool fp16 = !pools.k_q && !pools.v_q && !pools.k_q4 && !pools.v_q4;
    const bool buffers = q && ids && step && scratch && pools.page_table && pools.k_pool && pools.v_pool;
    const Refusal r = admit(on, device.arch, device.conversions, device.wave,
                           s.n_head, s.n_head_kv, s.head_dim, cap, s.page_size, queries, fp16, buffers);
    int d = -1;
    check(hipGetDevice(&d), "report device");
    const unsigned bit = 1u << static_cast<unsigned>(r);
    if (!(reported[d].fetch_or(bit) & bit))
        std::fprintf(stderr, "strata fp8-attn probe: device %d %s: %s%s\n", d, device.arch,
                     r == Refusal::none ? "selected: " : "declined: ", reason(r));
    if (r != Refusal::none) return r;
    const dim3 grid(static_cast<unsigned>((cap + kTile - 1) / kTile), static_cast<unsigned>(s.n_head_kv));
    hipStream_t st = static_cast<hipStream_t>(stream);
    if (s.n_head / s.n_head_kv == 8)
        decode_tile<8><<<grid, kThreads, 0, st>>>(q, pools, ids, step, static_cast<int>(cap),
                                                static_cast<int>(s.n_head_kv), static_cast<int>(s.page_size), scratch);
    else
        decode_tile<12><<<grid, kThreads, 0, st>>>(q, pools, ids, step, static_cast<int>(cap),
                                                 static_cast<int>(s.n_head_kv), static_cast<int>(s.page_size), scratch);
    check(hipGetLastError(), "tile launch");
    return Refusal::none;
}
}  // namespace strata::kernels::fp8_probe
#undef STRATA_FP8_PAIR_IMAGE
#endif
