// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Buffer descriptor/cache encodings adapted from hipfire's
// kernels/src/gfx12_weight_cache_policy.inc.
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>

// Match gfx_arch_is_gfx11_wmma / gfx_arch_is_gfx12, including HIP's host pass.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__)
#define STRATA_HIP_BUFFER_RDNA 1
#else
#define STRATA_HIP_BUFFER_RDNA 0
#endif

namespace strata::hip_compat {

using Buf128 = unsigned int __attribute__((ext_vector_type(4)));
using Buf64 = unsigned int __attribute__((ext_vector_type(2)));

#if defined(__gfx1200__) || defined(__gfx1201__)
// gfx12 TH_RT + SCOPE_DEV. 18 is HT/dev, 20 and 22 are the NT variants.
inline constexpr int kBufferCachePolicy = 16;
#else
inline constexpr int kBufferCachePolicy = 0;
#endif

struct BufRsrc {
#if STRATA_HIP_BUFFER_RDNA
    __amdgpu_buffer_rsrc_t value;
#else
    const unsigned char* base;
#endif
};

__device__ __forceinline__ BufRsrc strata_hip_make_rsrc(const void* p, std::size_t bound_bytes) {
#if STRATA_HIP_BUFFER_RDNA
    const unsigned int bound = bound_bytes > UINT32_MAX ? UINT32_MAX : static_cast<unsigned int>(bound_bytes);
    return {__builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(p), 0, bound, 0x31004000u)};
#else
    (void) bound_bytes;
    return {static_cast<const unsigned char*>(p)};
#endif
}

// The cache policy must be an ISA immediate, hence the template override rather than a runtime argument.
// `soff` is separate so a wave-uniform row offset can stay in an SGPR.
template <int Ctrl = kBufferCachePolicy>
__device__ __forceinline__ Buf128 strata_hip_buf_ld_b128(BufRsrc rsrc, unsigned int off, unsigned int soff = 0) {
#if STRATA_HIP_BUFFER_RDNA
    return __builtin_amdgcn_raw_buffer_load_b128(rsrc.value, off, soff, Ctrl);
#else
    Buf128 v;
    __builtin_memcpy(&v, rsrc.base + off + soff, sizeof(v));
    return v;
#endif
}

template <int Ctrl = kBufferCachePolicy>
__device__ __forceinline__ Buf64 strata_hip_buf_ld_b64(BufRsrc rsrc, unsigned int off, unsigned int soff = 0) {
#if STRATA_HIP_BUFFER_RDNA
    return __builtin_amdgcn_raw_buffer_load_b64(rsrc.value, off, soff, Ctrl);
#else
    Buf64 v;
    __builtin_memcpy(&v, rsrc.base + off + soff, sizeof(v));
    return v;
#endif
}

template <int Ctrl = kBufferCachePolicy>
__device__ __forceinline__ uint32_t strata_hip_buf_ld_b32(BufRsrc rsrc, unsigned int off, unsigned int soff = 0) {
#if STRATA_HIP_BUFFER_RDNA
    return __builtin_amdgcn_raw_buffer_load_b32(rsrc.value, off, soff, Ctrl);
#else
    uint32_t v;
    __builtin_memcpy(&v, rsrc.base + off + soff, sizeof(v));
    return v;
#endif
}

__device__ __forceinline__ void strata_hip_buffer_sched_barrier() {
#if STRATA_HIP_BUFFER_RDNA
    __builtin_amdgcn_sched_barrier(0);
#endif
}

}  // namespace strata::hip_compat

#else
namespace strata::hip_compat {

struct BufRsrc {};
inline void strata_hip_make_rsrc(const void*, std::size_t) {}
template <int Ctrl = 0> inline void strata_hip_buf_ld_b128(BufRsrc, unsigned int, unsigned int = 0) {}
template <int Ctrl = 0> inline void strata_hip_buf_ld_b64(BufRsrc, unsigned int, unsigned int = 0) {}
template <int Ctrl = 0> inline void strata_hip_buf_ld_b32(BufRsrc, unsigned int, unsigned int = 0) {}

}  // namespace strata::hip_compat
#endif
