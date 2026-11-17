#include <cuda_runtime.h>
#include "strata/hip_compat/buffer_load.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <type_traits>

#if defined(__HIPCC__)
#include "strata/kernels/gfx_arch.hpp"

static void ck(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "hip_buffer_load_test: %s: %s\n", what, hipGetErrorString(e));
        std::exit(2);
    }
}

__global__ void copy_chunk(const unsigned char* src, unsigned char* dst, unsigned offset, unsigned bytes) {
    using namespace strata::hip_compat;
    __shared__ unsigned char tile[128] __attribute__((aligned(16)));
    const auto rsrc = strata_hip_make_rsrc(src, offset + bytes);
    strata_hip_buffer_copy_wave(tile, src, rsrc, offset, bytes, threadIdx.x);
    __syncwarp();
    for (unsigned i = threadIdx.x; i < bytes; i += 32) dst[i] = tile[i];
}

__global__ void widths(const unsigned char* src, unsigned char* dst) {
    using namespace strata::hip_compat;
    if (threadIdx.x != 0) return;
    const auto rsrc = strata_hip_make_rsrc(src, 32);
    const Buf128 a = strata_hip_buf_ld_b128(rsrc, 8, 8);
    const Buf64 b = strata_hip_buf_ld_b64(rsrc, 4, 4);
    const uint32_t c = strata_hip_buf_ld_b32(rsrc, 0, 4);
    __builtin_memcpy(dst, &a, 16);
    __builtin_memcpy(dst + 16, &b, 8);
    __builtin_memcpy(dst + 24, &c, 4);
}

int main() {
    int count = 0;
    const hipError_t e = hipGetDeviceCount(&count);
    if (e == hipErrorNoDevice || e == hipErrorInsufficientDriver || (e == hipSuccess && count == 0)) {
        std::puts("SKIP hip_buffer_load_test: no HIP GPU");
        return 77;
    }
    ck(e, "device count");
    hipDeviceProp_t prop{};
    ck(hipGetDeviceProperties(&prop, 0), "properties");
    if (prop.warpSize != 32 || (!strata::kernels::gfx_arch_is_gfx11_wmma(prop.gcnArchName) &&
                              !strata::kernels::gfx_arch_is_gfx12(prop.gcnArchName))) {
        std::puts("SKIP hip_buffer_load_test: requires supported gfx11/gfx12 wave32");
        return 77;
    }
    ck(hipSetDevice(0), "set device");
    unsigned char input[160], result[128];
    for (unsigned i = 0; i < sizeof(input); ++i) input[i] = (unsigned char) (i * 37 + 11);
    unsigned char *src = nullptr, *dst = nullptr;
    ck(hipMalloc((void**) &src, sizeof(input)), "source allocation");
    ck(hipMalloc((void**) &dst, sizeof(result)), "destination allocation");
    ck(hipMemcpy(src, input, sizeof(input), hipMemcpyHostToDevice), "upload");
    int cases = 0;
    for (unsigned bytes : {0u, 1u, 2u, 3u, 4u, 7u, 8u, 15u, 16u, 17u, 31u, 32u, 33u, 63u, 64u, 65u, 127u})
    for (unsigned off : {0u, 1u, 2u, 3u, 15u}) {
        ck(hipMemset(dst, 0xa5, sizeof(result)), "poison output");
        copy_chunk<<<1, 32>>>(src, dst, off, bytes);
        ck(hipGetLastError(), "copy launch");
        ck(hipMemcpy(result, dst, sizeof(result), hipMemcpyDeviceToHost), "copy output");
        if (std::memcmp(result, input + off, bytes) != 0) {
            std::fprintf(stderr, "FAIL buffer copy offset=%u bytes=%u\n", off, bytes);
            return 1;
        }
        for (unsigned i = bytes; i < sizeof(result); ++i) {
            if (result[i] != 0xa5) {
                std::fprintf(stderr, "FAIL buffer copy overwrote tail offset=%u bytes=%u at=%u\n", off, bytes, i);
                return 1;
            }
        }
        ++cases;
    }
    widths<<<1, 32>>>(src, dst);
    ck(hipGetLastError(), "widths launch");
    ck(hipMemcpy(result, dst, sizeof(result), hipMemcpyDeviceToHost), "widths output");
    if (std::memcmp(result, input + 16, 16) || std::memcmp(result + 16, input + 8, 8) ||
        std::memcmp(result + 24, input + 4, 4)) {
        std::puts("FAIL b128/b64/b32 or split scalar offset");
        return 1;
    }
    ck(hipFree(dst), "free destination");
    ck(hipFree(src), "free source");
    std::printf("PASS hip_buffer_load_test: %d bounded copy cases and b128/b64/b32\n", cases);
}
#else
static_assert(std::is_same_v<decltype(strata::hip_compat::strata_hip_make_rsrc(nullptr, 0)), void>);
static_assert(std::is_same_v<decltype(strata::hip_compat::strata_hip_buf_ld_b128({}, 0)), void>);
static_assert(std::is_same_v<decltype(strata::hip_compat::strata_hip_buf_ld_b64({}, 0)), void>);
static_assert(std::is_same_v<decltype(strata::hip_compat::strata_hip_buf_ld_b32({}, 0)), void>);
int main() { std::puts("PASS hip_buffer_load_test: CUDA has only void stubs"); }
#endif
