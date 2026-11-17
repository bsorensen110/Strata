// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Daniil Markevich
// Port note: hipfire gemm_mq4g256v2_residual_mmq_iu4.gfx12.hip:168-320,563-649
// and its gfx11 K16 sibling:5-30. Planar exact W4 packs replace MQ4V2 headers.
#include "strata/prefill/moe_iu4_gfx.hpp"
#if defined(__HIPCC__)
#include "strata/core/arch_defaults.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::prefill::iu4 {
namespace {
void check(hipError_t e, const char* message) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "HIP MoE IU4: %s: %s\n", message, hipGetErrorString(e)); std::exit(1);
    }
}
[[noreturn]] void invalid(const char* message) {
    std::fprintf(stderr, "HIP MoE IU4: invalid %s\n", message); std::exit(1);
}
__device__ __forceinline__ float half(uint16_t h) { return __half2float(__ushort_as_half(h)); }
__device__ __forceinline__ float fold(float acc, float sc, float zp, float d, int c, int s, bool symmetric) {
    const float t1 = __fmul_rn(sc, d);
    const float p = __fmul_rn(t1, (float) c);
    if (symmetric) return __fadd_rn(acc, p);
    const float t2 = __fmul_rn(zp, d);
    const float term = __fmaf_rn(t2, (float) s, p);
    return __fadd_rn(acc, term);
}
struct Tile { int row, token, begin, end; WeightView weight; };
__device__ __forceinline__ Tile tile(const Product& p) {
    const int mt = (p.rows + 127) / 128, nt = (p.tokens + 127) / 128;
    const int local = blockIdx.x % (mt * nt), group = blockIdx.x / (mt * nt);
    Tile t{};
    if (p.count && group >= *p.count) { t.end = -1; return t; }
    t.begin = p.count ? p.starts[group] : 0;
    t.end = p.count ? p.starts[group + 1] : p.tokens;
    if (p.banded) {
        const int base = (local / (8 * nt)) * 8;
        const int width = min(8, mt - base), in = local - base * nt;
        t.row = (base + in % width) * 128; t.token = t.begin + (in / width) * 128;
    } else {
        t.row = (local % mt) * 128; t.token = t.begin + (local / mt) * 128;
    }
    if (t.token >= t.end) return t;
    t.weight = p.count ? p.table[p.route[p.dst[t.begin]]] : p.weight;
    return t;
}
__device__ __forceinline__ int source(const Product& p, int token) {
    return p.count && !p.sorted_x ? p.tok[token] : token;
}
__device__ __forceinline__ int destination(const Product& p, int token) {
    return p.count && !p.sorted_y ? p.dst[token] : token;
}
namespace gfx12 {
// K32 is deliberately unavailable, including on gfx1200, outside this exact
// predicate. Never substitute the gfx11 builtin under an operand-width alias.
#if defined(__gfx1201__)
using OperandK32 = int __attribute__((ext_vector_type(2)));
using Acc = int __attribute__((ext_vector_type(8)));
using F8 = float __attribute__((ext_vector_type(8)));
static_assert(sizeof(OperandK32) == 8);
struct Payload { OperandK32 a[2], w[2]; unsigned ds; float sz; };
template<bool Metadata>
__device__ __forceinline__ Payload fetch(const Product& p, const Tile& t, int kb, int slab) {
    Payload v{};
    const int tid = threadIdx.x;
    for (int j = 0; j < 2; ++j) {
        const int i = tid + j * 256, r = i / 4, quad = i % 4;
        const int token = t.token + r, row = t.row + r;
        if (token < t.end)
            v.a[j] = *(const OperandK32*) (p.x[(int64_t) kb * p.x_rows + source(p, token)].qs + slab * 32 + quad * 8);
        if (row < p.rows) {
            v.w[j] = *(const OperandK32*) (t.weight.codes + (size_t) row * (p.cols / 2) + kb * 64 + slab * 32 + quad * 8);
            if (p.symmetric) {
                v.w[j][0] ^= (int) 0x88888888u; v.w[j][1] ^= (int) 0x88888888u;
            }
        }
    }
    if constexpr (Metadata) {
        const int token = t.token + tid / 2, row = t.row + tid / 2;
        if (token < t.end)
            v.ds = ((const unsigned*) (p.x + (int64_t) kb * p.x_rows + source(p, token)))[tid & 1];
        if (row < p.rows) {
            const size_t h = (size_t) row * (p.cols / 128) + kb;
            const float sc = half(t.weight.scales[h]);
            v.sz = (tid & 1) ? (t.weight.zero_points ? half(t.weight.zero_points[h]) : __fmul_rn(-8.0f, sc)) : sc;
        }
    }
    return v;
}
template<bool Metadata>
__device__ __forceinline__ void publish(char* lds, const Payload& v, int slot, int kb) {
    const int tid = threadIdx.x;
    const int aoff = slot ? 16384 : 0, woff = slot ? 8192 : 4096;
    for (int j = 0; j < 2; ++j) {
        const int i = tid + j * 256, r = i / 4, quad = i % 4;
        const int fragment = (r / 16) * 2 + quad / 2;
        const int lane = r % 16 + 16 * (quad & 1);
        *(OperandK32*) (lds + aoff + fragment * 256 + lane * 8) = v.a[j];
        *(OperandK32*) (lds + woff + fragment * 256 + lane * 8) = v.w[j];
    }
    if constexpr (Metadata) {
        ((unsigned*) (lds + 12288 + (kb & 1) * 1024))[tid] = v.ds;
        ((float*) (lds + 14336 + (kb & 1) * 1024))[tid] = v.sz;
    }
}
__device__ __forceinline__ void bundle(const char* lds, int slot, bool symmetric, Acc (&c)[4][2]) {
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int aoff = slot ? 16384 : 0, woff = slot ? 8192 : 4096;
    for (int step = 0; step < 2; ++step)
    for (int rg = 0; rg < 2; ++rg) {
        const OperandK32 a = *(const OperandK32*) (lds + woff + ((wave / 2 * 2 + rg) * 2 + step) * 256 + lane * 8);
        for (int nb = 0; nb < 4; ++nb) {
            const OperandK32 b = *(const OperandK32*) (lds + aoff + ((wave % 2 * 4 + nb) * 2 + step) * 256 + lane * 8);
            // signedness is an immediate operand; the two modes remain distinct.
            if (symmetric) c[nb][rg] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, a, true, b, c[nb][rg], false);
            else c[nb][rg] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false, a, true, b, c[nb][rg], false);
        }
    }
}
#else
template<class... T> __device__ void bundle(T...) = delete;
#endif
}  // namespace gfx12

namespace gfx11 {
#if defined(__gfx1100__) || defined(__gfx1151__)
using OperandK16 = int __attribute__((ext_vector_type(2)));
using Acc = int __attribute__((ext_vector_type(8)));
using F8 = float __attribute__((ext_vector_type(8)));
static_assert(sizeof(OperandK16) == 8);
__device__ __forceinline__ void bundle(const char* ws, const Block* xs, int kb, bool symmetric, Acc (&c)[4][2]) {
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, lo = lane & 15;
    // Both lane halves load IDENTICAL K16 operands. No K32 lane split here.
    for (int step = 0; step < 8; ++step)
    for (int rg = 0; rg < 2; ++rg) {
        const int row = wave / 2 * 32 + rg * 16 + lo;
        const OperandK16 a = *(const OperandK16*) (ws + row * 168 + (kb & 1) * 64 + step * 8);
        for (int nb = 0; nb < 4; ++nb) {
            const OperandK16 b = *(const OperandK16*) (xs[wave % 2 * 64 + nb * 16 + lo].qs + step * 8);
            if (symmetric) c[nb][rg] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c[nb][rg], false);
            else c[nb][rg] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, a, true, b, c[nb][rg], false);
        }
    }
}
#else
template<class... T> __device__ void bundle(T...) = delete;
#endif
}  // namespace gfx11
}  // namespace

extern "C" __global__ __launch_bounds__(256) void strata_gemm_i4_k32_gfx1201(Product p) {
#if defined(__gfx1201__)
    const Tile t = tile(p);
    if (t.token >= t.end) return;
    __shared__ __align__(16) char lds[kGfx12Lds];
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, lo = lane & 15, hi = lane >> 4;
    gfx12::F8 acc[4][2] = {};
    gfx12::publish<true>(lds, gfx12::fetch<true>(p, t, 0, 0), 0, 0);
    __syncthreads();
    for (int kb = 0; kb < p.cols / 128; ++kb) {
        gfx12::Acc c[4][2] = {};
        auto next = gfx12::fetch<false>(p, t, kb, 1);
        gfx12::bundle(lds, 0, p.symmetric, c);
        gfx12::publish<false>(lds, next, 1, kb);
        __syncthreads();  // slot-0 retirement + slot-1 publication
        const bool more = kb + 1 < p.cols / 128;
        if (more) next = gfx12::fetch<true>(p, t, kb + 1, 0);
        gfx12::bundle(lds, 1, p.symmetric, c);
        for (int nb = 0; nb < 4; ++nb) {
            const int token = wave % 2 * 64 + nb * 16 + lo;
            const char* header = lds + 12288 + (kb & 1) * 1024 + token * 8;
            const float d = *(const float*) header;
            const int s = *(const int*) (header + 4);
            for (int rg = 0; rg < 2; ++rg)
            for (int j = 0; j < 8; ++j) {
                const int row = wave / 2 * 32 + rg * 16 + 8 * hi + j;
                const float* sz = (const float*) (lds + 14336 + (kb & 1) * 1024 + row * 8);
                acc[nb][rg][j] = fold(acc[nb][rg][j], sz[0], sz[1], d, c[nb][rg][j], s, p.symmetric);
            }
        }
        if (more) {
            gfx12::publish<true>(lds, next, 0, kb + 1);
            __syncthreads();  // slot-1 retirement + next slot-0 publication
        }
    }
    for (int nb = 0; nb < 4; ++nb)
    for (int rg = 0; rg < 2; ++rg)
    for (int j = 0; j < 8; ++j) {
        const int token = t.token + wave % 2 * 64 + nb * 16 + lo;
        const int row = t.row + wave / 2 * 32 + rg * 16 + 8 * hi + j;
        if (token < t.end && row < p.rows)
            p.y[(int64_t) destination(p, token) * p.ld + row] = acc[nb][rg][j];
    }
#elif defined(__HIP_DEVICE_COMPILE__)
    __builtin_trap();
#endif
}

extern "C" __global__ __launch_bounds__(256) void strata_gemm_i4_k16_gfx11(Product p) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    const Tile t = tile(p);
    if (t.token >= t.end) return;
    __shared__ __align__(16) char lds[kGfx11Lds];
    auto* ws = lds;
    auto* xs = (Block*) (lds + 128 * 168);
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, lo = lane & 15, hi = lane >> 4;
    gfx11::F8 acc[4][2] = {};
    for (int kb = 0; kb < p.cols / 128; ++kb) {
        for (int i = tid; i < 128 * 16; i += 256) {
            const int r = i / 16, word = i % 16, row = t.row + r;
            unsigned v = row < p.rows
                ? ((const unsigned*) (t.weight.codes + (size_t) row * (p.cols / 2) + kb * 64))[word] : 0;
            if (p.symmetric) v ^= 0x88888888u;
            ((unsigned*) (ws + r * 168 + (kb & 1) * 64))[word] = v;
        }
        for (int i = tid; i < 128 * 18; i += 256) {
            const int token = t.token + i / 18;
            ((unsigned*) xs)[i] = token < t.end
                ? ((const unsigned*) (p.x + (int64_t) kb * p.x_rows + source(p, token)))[i % 18] : 0;
        }
        const int row = t.row + tid / 2;
        float metadata = 0.0f;
        if (row < p.rows) {
            const size_t h = (size_t) row * (p.cols / 128) + kb;
            const float sc = half(t.weight.scales[h]);
            metadata = tid & 1 ? (t.weight.zero_points ? half(t.weight.zero_points[h]) : __fmul_rn(-8.0f, sc)) : sc;
        }
        ((float*) (ws + (tid / 2) * 168 + 128 + (kb & 1) * 8))[tid & 1] = metadata;
        __syncthreads();
        gfx11::Acc c[4][2] = {};
        gfx11::bundle(ws, xs, kb, p.symmetric, c);
        for (int nb = 0; nb < 4; ++nb) {
            const auto& x = xs[wave % 2 * 64 + nb * 16 + lo];
            for (int rg = 0; rg < 2; ++rg)
            for (int j = 0; j < 8; ++j) {
                const int r = wave / 2 * 32 + rg * 16 + 2 * j + hi;
                const float* sz = (const float*) (ws + r * 168 + 128 + (kb & 1) * 8);
                acc[nb][rg][j] = fold(acc[nb][rg][j], sz[0], sz[1], x.d, c[nb][rg][j], x.s, p.symmetric);
            }
        }
        __syncthreads();
    }
    for (int nb = 0; nb < 4; ++nb)
    for (int rg = 0; rg < 2; ++rg)
    for (int j = 0; j < 8; ++j) {
        const int token = t.token + wave % 2 * 64 + nb * 16 + lo;
        const int row = t.row + wave / 2 * 32 + rg * 16 + 2 * j + hi;
        if (token < t.end && row < p.rows)
            p.y[(int64_t) destination(p, token) * p.ld + row] = acc[nb][rg][j];
    }
#elif defined(__HIP_DEVICE_COMPILE__)
    __builtin_trap();
#endif
}

bool usable() {
    const char* env = std::getenv(strata::core::kHipMoeIu4Env);
    if (!env || std::strcmp(env, "1") != 0) return false;
    int device = -1;
    check(hipGetDevice(&device), "device");
    hipDeviceProp_t prop{};
    check(hipGetDeviceProperties(&prop, device), "properties");
    return prop.warpSize == 32 && strata::core::hip_moe_iu4_capable(prop.gcnArchName);
}
void run(const Product& p, void* stream) {
    if (!usable() || !p.x || !p.y || p.rows <= 0 || p.rows > INT_MAX - 127 ||
        p.cols <= 0 || p.cols % 128 || p.tokens <= 0 || p.tokens > INT_MAX - 127 ||
        p.groups <= 0 || p.x_rows <= 0 || p.ld < p.rows) invalid("shape/admission");
    if (p.count ? (!p.table || !p.starts || !p.tok || !p.dst || !p.route)
                : (!p.weight.codes || !p.weight.scales || p.weight.rows != p.rows || p.weight.cols != p.cols ||
                   p.x_rows != p.tokens || (p.symmetric && (!p.weight.symmetric_verified || p.weight.zero_points))))
        invalid("weight proof/route tables");
    const int64_t grid = ((int64_t) p.rows + 127) / 128 * (((int64_t) p.tokens + 127) / 128) * p.groups;
    if (grid <= 0 || grid > INT_MAX) invalid("folded x grid");
    int device = -1;
    check(hipGetDevice(&device), "launch device");
    if (!p.count && p.weight.device != device) invalid("weight device");
    hipDeviceProp_t prop{};
    check(hipGetDeviceProperties(&prop, device), "launch properties");
    if (strata::kernels::gfx_arch_is(prop.gcnArchName, "gfx1201"))
        strata_gemm_i4_k32_gfx1201<<<(unsigned) grid, 256, 0, (hipStream_t) stream>>>(p);
    else strata_gemm_i4_k16_gfx11<<<(unsigned) grid, 256, 0, (hipStream_t) stream>>>(p);
    check(hipGetLastError(), "GEMM launch");
}
}  // namespace strata::prefill::iu4
#endif
