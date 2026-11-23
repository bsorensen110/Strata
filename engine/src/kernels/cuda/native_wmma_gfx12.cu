// src/kernels/cuda/native_wmma_gfx12.cu - RDNA4 (gfx12 wave32) INT8 WMMA kernels for wide verify windows
// and CPU software emulator for cross-platform parity validation.

#include "strata/kernels/native_wmma_gfx12.hpp"
#include "strata/kernels/gfx_arch.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace strata::kernels {
namespace {

struct WmmaQ80Block {
    half d;
    int8_t qs[32];
};
struct WmmaQ81Block {
    half2 ds;
    int8_t qs[32];
};
struct WmmaQ6KBlock {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};
static_assert(sizeof(WmmaQ80Block) == 34);
static_assert(sizeof(WmmaQ81Block) == 36);
static_assert(sizeof(WmmaQ6KBlock) == 210);

float half_to_float_host(uint16_t h) {
    const uint32_t sign = (uint32_t(h & 0x8000u)) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x03FFu;
    uint32_t fbits = 0;
    if (exp == 0) {
        if (mant == 0) {
            fbits = sign;
        } else {
            uint32_t m = mant;
            int e = -14;
            while ((m & 0x0400u) == 0) { m <<= 1; --e; }
            m &= 0x03FFu;
            fbits = sign | (uint32_t(e + 127) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        fbits = sign | 0x7F800000u | (mant << 13);
    } else {
        fbits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &fbits, sizeof(out));
    return out;
}

#if defined(STRATA_USE_HIP) && defined(STRATA_WMMA_GFX11)
#define STRATA_BUILD_GFX12_WMMA 1
typedef int wmma_i8x8 __attribute__((ext_vector_type(8)));
typedef int wmma_i32x2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ int load_i32_u16(const int8_t* p) {
    const auto* u = reinterpret_cast<const uint16_t*>(p);
    return int(uint32_t(u[0]) | (uint32_t(u[1]) << 16));
}

__device__ __forceinline__ wmma_i8x8 gfx12_wmma_i8(wmma_i32x2 a, wmma_i32x2 b, wmma_i8x8 c) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1200__) || defined(__gfx1201__))
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
#else
    (void) a; (void) b;
    __builtin_trap();
    return c;
#endif
}

// Each wave computes 16 output rows x up to 16 columns. 4 waves per block -> 64 rows per CTA.
template <bool CHECK_BOUNDS>
__launch_bounds__(128)
__global__ void wmma_q8_0_kernel(const WmmaQ80Block* __restrict__ w, const WmmaQ81Block* __restrict__ x,
                                 float* __restrict__ y, int n_in, int n_out, int ncols) {
    const int tid = int(threadIdx.x);
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    const int row0 = (int(blockIdx.x) * 4 + wave) * 16;
    if (row0 >= n_out) return;

    const int nblk = n_in / 32;
    const int wrow = row0 + l16;
    const bool valid_wrow = !CHECK_BOUNDS || (wrow < n_out);
    const WmmaQ80Block* const wp = valid_wrow ? (w + size_t(wrow) * nblk) : nullptr;
    const bool valid_col = l16 < ncols;
    const WmmaQ81Block* const xp = valid_col ? (x + size_t(l16) * nblk) : nullptr;

    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int b = 0; b < nblk; ++b) {
        const float my_dw = wp ? __half2float(wp[b].d) : 0.0f;
        const float dx_col = xp ? __low2float(xp[b].ds) : 0.0f;
        const int8_t* wqs = wp ? wp[b].qs : nullptr;
        const int* xqs = xp ? reinterpret_cast<const int*>(xp[b].qs) : nullptr;
        wmma_i8x8 d_tile{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
            const int i32 = sub * 4 + hi * 2;
            const wmma_i32x2 a_frag = wqs
                ? wmma_i32x2{load_i32_u16(wqs + i32 * 4), load_i32_u16(wqs + (i32 + 1) * 4)}
                : wmma_i32x2{0, 0};
            const wmma_i32x2 b_frag = xqs ? wmma_i32x2{xqs[i32], xqs[i32 + 1]} : wmma_i32x2{0, 0};
            d_tile = gfx12_wmma_i8(a_frag, b_frag, d_tile);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float dw = __shfl_sync(0xffffffffu, my_dw, 8 * hi + i);
            acc[i] = fmaf(float(d_tile[i]), dw * dx_col, acc[i]);
        }
    }
    if (valid_col) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = row0 + 8 * hi + i;
            if (!CHECK_BOUNDS || r < n_out) y[size_t(l16) * n_out + r] = acc[i];
        }
    }
}

struct WmmaQ80Seg {
    const WmmaQ80Block* w;
    float* y;
    int n_out;
    int blk_end;
};
struct WmmaQ80SegArray {
    WmmaQ80Seg seg[3];
    int n_segs;
};

__launch_bounds__(128)
__global__ void wmma_q8_0_segs_kernel(WmmaQ80SegArray segs, const WmmaQ81Block* __restrict__ x,
                                      int n_in, int ncols) {
    const int gblk = int(blockIdx.x);
    int si = 0;
    if (gblk >= segs.seg[0].blk_end) si = (segs.n_segs > 2 && gblk >= segs.seg[1].blk_end) ? 2 : 1;
    const int blk_base = si == 0 ? 0 : segs.seg[si - 1].blk_end;
    const auto& seg = segs.seg[si];
    const int tid = int(threadIdx.x);
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    const int row0 = ((gblk - blk_base) * 4 + wave) * 16;
    if (row0 >= seg.n_out) return;

    const int nblk = n_in / 32;
    const int wrow = row0 + l16;
    const WmmaQ80Block* const wp = (wrow < seg.n_out) ? (seg.w + size_t(wrow) * nblk) : nullptr;
    const bool valid_col = l16 < ncols;
    const WmmaQ81Block* const xp = valid_col ? (x + size_t(l16) * nblk) : nullptr;

    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int b = 0; b < nblk; ++b) {
        const float my_dw = wp ? __half2float(wp[b].d) : 0.0f;
        const float dx_col = xp ? __low2float(xp[b].ds) : 0.0f;
        const int8_t* wqs = wp ? wp[b].qs : nullptr;
        const int* xqs = xp ? reinterpret_cast<const int*>(xp[b].qs) : nullptr;
        wmma_i8x8 d_tile{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
            const int i32 = sub * 4 + hi * 2;
            const wmma_i32x2 a_frag = wqs
                ? wmma_i32x2{load_i32_u16(wqs + i32 * 4), load_i32_u16(wqs + (i32 + 1) * 4)}
                : wmma_i32x2{0, 0};
            const wmma_i32x2 b_frag = xqs ? wmma_i32x2{xqs[i32], xqs[i32 + 1]} : wmma_i32x2{0, 0};
            d_tile = gfx12_wmma_i8(a_frag, b_frag, d_tile);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float dw = __shfl_sync(0xffffffffu, my_dw, 8 * hi + i);
            acc[i] = fmaf(float(d_tile[i]), dw * dx_col, acc[i]);
        }
    }
    if (valid_col) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = row0 + 8 * hi + i;
            if (r < seg.n_out) seg.y[size_t(l16) * seg.n_out + r] = acc[i];
        }
    }
}

__launch_bounds__(128)
__global__ void wmma_q6_k_kernel(const WmmaQ6KBlock* __restrict__ w, const WmmaQ81Block* __restrict__ x,
                                 float* __restrict__ y, int n_in, int n_out, int ncols) {
    const int tid = int(threadIdx.x);
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    const int row0 = (int(blockIdx.x) * 4 + wave) * 16;
    if (row0 >= n_out) return;

    const int nblk = n_in / 256;
    const int nx8 = n_in / 32;
    const int wrow = row0 + l16;
    const WmmaQ6KBlock* const wp = (wrow < n_out) ? (w + size_t(wrow) * nblk) : nullptr;
    const bool valid_col = l16 < ncols;
    const WmmaQ81Block* const xp = valid_col ? (x + size_t(l16) * nx8) : nullptr;

    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int b = 0; b < nblk; ++b) {
        const float my_d = wp ? __half2float(wp[b].d) : 0.0f;
        for (int sb = 0; sb < 16; ++sb) {
            const int qb = sb >> 1;
            const int half16 = sb & 1;
            const int half = qb >> 2;
            const int ni = (qb >> 1) & 1;
            const int quad = qb & 1;
            const float my_sc = wp ? (my_d * float(wp[b].scales[sb])) : 0.0f;
            const float dx_col = xp ? __low2float(xp[b * 8 + qb].ds) : 0.0f;
            const int base32 = half16 * 16 + hi * 8;
            int aw[2] = {0, 0};
            if (wp) {
                for (int word = 0; word < 2; ++word) {
                    int packed = 0;
                    for (int t = 0; t < 4; ++t) {
                        const int idx32 = base32 + word * 4 + t;
                        const uint8_t qlb = wp[b].ql[64 * half + 32 * quad + idx32];
                        const uint8_t qhb = wp[b].qh[32 * half + idx32];
                        const int q6 = ((qlb >> (4 * ni)) & 0x0F) | (((qhb >> (2 * quad + 4 * ni)) & 0x03) << 4);
                        const int8_t val = int8_t(q6 - 32);
                        packed |= (int(uint8_t(val)) << (8 * t));
                    }
                    aw[word] = packed;
                }
            }
            const int* xqs = xp ? reinterpret_cast<const int*>(xp[b * 8 + qb].qs + base32) : nullptr;
            const wmma_i32x2 a_frag{aw[0], aw[1]};
            const wmma_i32x2 b_frag = xqs ? wmma_i32x2{xqs[0], xqs[1]} : wmma_i32x2{0, 0};
            const wmma_i8x8 d_tile = gfx12_wmma_i8(a_frag, b_frag, wmma_i8x8{0, 0, 0, 0, 0, 0, 0, 0});
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float sw = __shfl_sync(0xffffffffu, my_sc, 8 * hi + i);
                acc[i] = fmaf(float(d_tile[i]), sw * dx_col, acc[i]);
            }
        }
    }
    if (valid_col) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = row0 + 8 * hi + i;
            if (r < n_out) y[size_t(l16) * n_out + r] = acc[i];
        }
    }
}
#else
#define STRATA_BUILD_GFX12_WMMA 0
#endif

}  // namespace

bool native_wmma_gfx12_available() noexcept {
#if STRATA_BUILD_GFX12_WMMA
    static int cached[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    int v = cached[dev];
    if (v == 0) {
        cudaDeviceProp p{};
        v = (cudaGetDeviceProperties(&p, dev) == cudaSuccess &&
             strata::kernels::gfx_arch_is_gfx12_wmma(p.gcnArchName)) ? 1 : 2;
        cached[dev] = v;
    }
    return v == 1;
#else
    return false;
#endif
}

bool native_wmma_dense_q8_0(const void* weights, const void* x_q8_1, float* y,
                            int n_in, int n_out, int ncols, void* stream) {
#if STRATA_BUILD_GFX12_WMMA
    if (!native_wmma_gfx12_available()) return false;
    if (!weights || !x_q8_1 || !y || !stream || n_in <= 0 || n_in % 32 != 0 || n_out <= 0 || ncols < 2 || ncols > 16)
        return false;
    const unsigned blocks = unsigned((n_out + 63) / 64);
    const auto* w = static_cast<const WmmaQ80Block*>(weights);
    const auto* x = static_cast<const WmmaQ81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    if (n_out % 64 == 0) wmma_q8_0_kernel<false><<<blocks, 128, 0, s>>>(w, x, y, n_in, n_out, ncols);
    else wmma_q8_0_kernel<true><<<blocks, 128, 0, s>>>(w, x, y, n_in, n_out, ncols);
    return cudaGetLastError() == cudaSuccess;
#else
    (void) weights; (void) x_q8_1; (void) y; (void) n_in; (void) n_out; (void) ncols; (void) stream;
    return false;
#endif
}

bool native_wmma_dense_q8_0_segs(const NativeMmvqSeg* segs, int n_segs, const void* x_q8_1,
                                 int n_in, int ncols, void* stream) {
#if STRATA_BUILD_GFX12_WMMA
    if (!native_wmma_gfx12_available()) return false;
    if (!segs || n_segs < 2 || n_segs > 3 || !x_q8_1 || !stream || n_in <= 0 || n_in % 32 != 0 || ncols < 2 || ncols > 16)
        return false;
    WmmaQ80SegArray sa{};
    sa.n_segs = n_segs;
    int total_blk = 0;
    for (int i = 0; i < n_segs; ++i) {
        if (!segs[i].weights || !segs[i].y || segs[i].n_out <= 0) return false;
        total_blk += (segs[i].n_out + 63) / 64;
        sa.seg[i] = {static_cast<const WmmaQ80Block*>(segs[i].weights), segs[i].y, segs[i].n_out, total_blk};
    }
    wmma_q8_0_segs_kernel<<<unsigned(total_blk), 128, 0, static_cast<cudaStream_t>(stream)>>>(
        sa, static_cast<const WmmaQ81Block*>(x_q8_1), n_in, ncols);
    return cudaGetLastError() == cudaSuccess;
#else
    (void) segs; (void) n_segs; (void) x_q8_1; (void) n_in; (void) ncols; (void) stream;
    return false;
#endif
}

bool native_wmma_dense_q6_k(const void* weights, const void* x_q8_1, float* y,
                            int n_in, int n_out, int ncols, void* stream) {
#if STRATA_BUILD_GFX12_WMMA
    if (!native_wmma_gfx12_available()) return false;
    if (!weights || !x_q8_1 || !y || !stream || n_in <= 0 || n_in % 256 != 0 || n_out <= 0 || ncols < 2 || ncols > 16)
        return false;
    const unsigned blocks = unsigned((n_out + 63) / 64);
    wmma_q6_k_kernel<<<blocks, 128, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const WmmaQ6KBlock*>(weights), static_cast<const WmmaQ81Block*>(x_q8_1), y, n_in, n_out, ncols);
    return cudaGetLastError() == cudaSuccess;
#else
    (void) weights; (void) x_q8_1; (void) y; (void) n_in; (void) n_out; (void) ncols; (void) stream;
    return false;
#endif
}

bool native_wmma_expert_grouped(const NativeExpertLayout&, const unsigned long long*, const int32_t*,
                                const int32_t*, const int32_t*, const int32_t*, int64_t, int64_t,
                                const void*, void*, float*, void*) {
    // MoE INT8 WMMA dispatch point for Q8_0/Q6_K grouped experts (opt-in STRATA_VERIFY_WMMA_MOE=1).
    return false;
}

void native_wmma_gfx12_emulate_16x16x16_i8(const int8_t a_lane[32][8], const int8_t b_lane[32][8],
                                           int32_t d_lane[32][8]) {
    int8_t A[16][16]{};
    int8_t B[16][16]{};
    for (int l = 0; l < 32; ++l) {
        const int l16 = l & 15;
        const int hi = l >> 4;
        for (int j = 0; j < 8; ++j) {
            A[l16][8 * hi + j] = a_lane[l][j];
            B[8 * hi + j][l16] = b_lane[l][j];
        }
    }
    int32_t D[16][16]{};
    for (int r = 0; r < 16; ++r) {
        for (int c = 0; c < 16; ++c) {
            int32_t acc = 0;
            for (int k = 0; k < 16; ++k) acc += int32_t(A[r][k]) * int32_t(B[k][c]);
            D[r][c] = acc;
        }
    }
    for (int l = 0; l < 32; ++l) {
        const int l16 = l & 15;
        const int hi = l >> 4;
        for (int i = 0; i < 8; ++i) d_lane[l][i] = D[8 * hi + i][l16];
    }
}

void native_wmma_gfx12_q8_0_cpu_ref_tiled(const void* weights, const void* x_q8_1, float* y,
                                          int n_in, int n_out, int ncols) {
    const auto* w = static_cast<const uint8_t*>(weights);
    const auto* x = static_cast<const uint8_t*>(x_q8_1);
    const int nblk = n_in / 32;
    for (int row0 = 0; row0 < n_out; row0 += 16) {
        float acc[32][8]{};
        for (int b = 0; b < nblk; ++b) {
            float dw_lane[32]{}, dx_lane[32]{};
            for (int l = 0; l < 32; ++l) {
                const int l16 = l & 15;
                if (row0 + l16 < n_out) {
                    const uint8_t* wp = w + (size_t(row0 + l16) * nblk + b) * 34;
                    uint16_t dh = 0;
                    std::memcpy(&dh, wp, 2);
                    dw_lane[l] = half_to_float_host(dh);
                }
                if (l16 < ncols) {
                    const uint8_t* xp = x + (size_t(l16) * nblk + b) * 36;
                    uint16_t dxh = 0;
                    std::memcpy(&dxh, xp, 2);
                    dx_lane[l] = half_to_float_host(dxh);
                }
            }
            int32_t d_sum[32][8]{};
            for (int sub = 0; sub < 2; ++sub) {
                int8_t a_lane[32][8]{}, b_lane[32][8]{};
                int32_t d_lane[32][8]{};
                for (int l = 0; l < 32; ++l) {
                    const int l16 = l & 15;
                    const int hi = l >> 4;
                    if (row0 + l16 < n_out) {
                        const uint8_t* wp = w + (size_t(row0 + l16) * nblk + b) * 34 + 2 + sub * 16 + hi * 8;
                        std::memcpy(a_lane[l], wp, 8);
                    }
                    if (l16 < ncols) {
                        const uint8_t* xp = x + (size_t(l16) * nblk + b) * 36 + 4 + sub * 16 + hi * 8;
                        std::memcpy(b_lane[l], xp, 8);
                    }
                }
                native_wmma_gfx12_emulate_16x16x16_i8(a_lane, b_lane, d_lane);
                for (int l = 0; l < 32; ++l) {
                    for (int i = 0; i < 8; ++i) d_sum[l][i] += d_lane[l][i];
                }
            }
            for (int l = 0; l < 32; ++l) {
                const int hi = l >> 4;
                for (int i = 0; i < 8; ++i) {
                    const float dw = dw_lane[8 * hi + i];
                    acc[l][i] = std::fmaf(float(d_sum[l][i]), dw * dx_lane[l], acc[l][i]);
                }
            }
        }
        for (int l = 0; l < 32; ++l) {
            const int l16 = l & 15;
            const int hi = l >> 4;
            if (l16 < ncols) {
                for (int i = 0; i < 8; ++i) {
                    const int r = row0 + 8 * hi + i;
                    if (r < n_out) y[size_t(l16) * n_out + r] = acc[l][i];
                }
            }
        }
    }
}

void native_wmma_gfx12_q6_k_cpu_ref_tiled(const void* weights, const void* x_q8_1, float* y,
                                          int n_in, int n_out, int ncols) {
    const auto* w = static_cast<const uint8_t*>(weights);
    const auto* x = static_cast<const uint8_t*>(x_q8_1);
    const int nblk = n_in / 256;
    const int nx8 = n_in / 32;
    for (int row0 = 0; row0 < n_out; row0 += 16) {
        float acc[32][8]{};
        for (int b = 0; b < nblk; ++b) {
            for (int sb = 0; sb < 16; ++sb) {
                const int qb = sb >> 1;
                const int half16 = sb & 1;
                const int half = qb >> 2;
                const int ni = (qb >> 1) & 1;
                const int quad = qb & 1;
                float sw_lane[32]{}, dx_lane[32]{};
                int8_t a_lane[32][8]{}, b_lane[32][8]{};
                int32_t d_lane[32][8]{};
                for (int l = 0; l < 32; ++l) {
                    const int l16 = l & 15;
                    const int hi = l >> 4;
                    if (row0 + l16 < n_out) {
                        const uint8_t* wp = w + (size_t(row0 + l16) * nblk + b) * 210;
                        uint16_t dh = 0;
                        std::memcpy(&dh, wp + 208, 2);
                        const int8_t sc = static_cast<int8_t>(wp[192 + sb]);
                        sw_lane[l] = half_to_float_host(dh) * float(sc);
                        for (int j = 0; j < 8; ++j) {
                            const int idx32 = half16 * 16 + hi * 8 + j;
                            const uint8_t qlb = wp[64 * half + 32 * quad + idx32];
                            const uint8_t qhb = wp[128 + 32 * half + idx32];
                            const int q6 = ((qlb >> (4 * ni)) & 0x0F) | (((qhb >> (2 * quad + 4 * ni)) & 0x03) << 4);
                            a_lane[l][j] = int8_t(q6 - 32);
                        }
                    }
                    if (l16 < ncols) {
                        const uint8_t* xp = x + (size_t(l16) * nx8 + b * 8 + qb) * 36;
                        uint16_t dxh = 0;
                        std::memcpy(&dxh, xp, 2);
                        dx_lane[l] = half_to_float_host(dxh);
                        std::memcpy(b_lane[l], xp + 4 + half16 * 16 + hi * 8, 8);
                    }
                }
                native_wmma_gfx12_emulate_16x16x16_i8(a_lane, b_lane, d_lane);
                for (int l = 0; l < 32; ++l) {
                    const int hi = l >> 4;
                    for (int i = 0; i < 8; ++i) {
                        const float sw = sw_lane[8 * hi + i];
                        acc[l][i] = std::fmaf(float(d_lane[l][i]), sw * dx_lane[l], acc[l][i]);
                    }
                }
            }
        }
        for (int l = 0; l < 32; ++l) {
            const int l16 = l & 15;
            const int hi = l >> 4;
            if (l16 < ncols) {
                for (int i = 0; i < 8; ++i) {
                    const int r = row0 + 8 * hi + i;
                    if (r < n_out) y[size_t(l16) * n_out + r] = acc[l][i];
                }
            }
        }
    }
}

}  // namespace strata::kernels
