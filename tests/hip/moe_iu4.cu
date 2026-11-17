// SPDX-License-Identifier: MIT
// Port note: hipfire block_i4_128_quant.hip producer parity.
#include "strata/prefill/moe_iu4_gfx.hpp"
#include <cstdio>
#include <type_traits>
#if defined(__HIPCC__) && defined(STRATA_MOE_IU4_CPU_ORACLE)
#include "strata/prefill/moe_iu4_reference.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/core/arch_defaults.hpp"
#include <hip/hip_runtime.h>
#include <cfenv>
namespace i4 = strata::prefill::iu4;
namespace r4 = i4::reference;
namespace r8 = strata::prefill::iu8::reference;
namespace {
void check(hipError_t e, const char* message) {
    if (e != hipSuccess) throw std::runtime_error(std::string(message) + ": " + hipGetErrorString(e));
}
struct Buffer {
    void* p = nullptr;
    explicit Buffer(size_t bytes) { check(hipMalloc(&p, bytes), "allocate"); }
    ~Buffer() {
        if (p && hipFree(p) != hipSuccess) { std::fputs("FAIL IU4 buffer free\n", stderr); std::abort(); }
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
template<class T> void upload(Buffer& b, const std::vector<T>& x) {
    check(hipMemcpy(b.p, x.data(), x.size() * sizeof(T), hipMemcpyHostToDevice), "upload");
}
template<class T> void download(const Buffer& b, std::vector<T>& x) {
    check(hipMemcpy(x.data(), b.p, x.size() * sizeof(T), hipMemcpyDeviceToHost), "download");
}
void producers(hipStream_t stream) {
    constexpr int k = 256, n = 37;
    auto x = r8::activations(k, n);
    for (int j = 0; j < 128; ++j) {
        x[k+j] = j == 127 ? 7.0f : (float) ((j % 15) - 7) + 0.5f;
        x[2*k+j] = std::ldexp((float) (j - 64), -125);
        x[3*k+j] = (j & 1 ? -1.0f : 1.0f) * 0x1p100f;
    }
    const auto want = r4::quantize(x.data(), k, n);
    Buffer dx(x.size()*4), dq((want.size()+1)*sizeof(i4::Block));
    upload(dx, x);
    check(hipMemset(dq.p, 0xa5, (want.size()+1)*sizeof(i4::Block)), "poison A4");
    i4::quantize((const float*) dx.p, (i4::Block*) dq.p, k, n, stream);
    check(hipStreamSynchronize(stream), "quantize");
    std::vector<i4::Block> got(want.size()+1);
    download(dq, got);
    r8::require(std::memcmp(want.data(), got.data(), want.size()*sizeof(i4::Block)) == 0,
                "A4 producer differs from full eight-candidate CPU reference");
    for (auto v : got.back().qs) r8::require(v == 0xa5, "A4 producer tail overwrite");
    const auto* tail = (const uint8_t*) &got.back();
    for (size_t j = 0; j < sizeof(i4::Block); ++j) r8::require(tail[j] == 0xa5, "A4 header tail overwrite");

    auto gu = r8::activations(2*k, n, 67);
    Buffer dg(gu.size()*4), dh((size_t) n*k*4), bounds(2*sizeof(int32_t)), count(sizeof(int32_t));
    upload(dg, gu); upload(bounds, std::vector<int32_t>{3,n-2}); upload(count, std::vector<int32_t>{1});
    check(hipMemset(dq.p, 0xa5, (want.size()+1)*sizeof(i4::Block)), "poison fused A4");
    i4::swiglu_window((const float*) dg.p, (float*) dh.p, (i4::Block*) dq.p, k, n,
                      (const int32_t*) bounds.p, (const int32_t*) count.p, stream);
    check(hipStreamSynchronize(stream), "fused A4");
    std::vector<float> h((size_t) n*k);
    download(dh, h); download(dq, got);
    for (int kb = 0; kb < k/128; ++kb)
    for (int t = 0; t < n; ++t) {
        const auto& b = got[kb*n+t];
        if (t >= 3 && t < n-2) {
            const auto ref = r4::quantize_block(h.data()+(size_t)t*k+kb*128);
            r8::require(std::memcmp(&b, &ref, sizeof(b)) == 0, "fused A4 producer changed candidate/sum/packing");
        } else {
            const auto* bytes = (const uint8_t*) &b;
            for (size_t j = 0; j < sizeof(b); ++j) r8::require(bytes[j] == 0xa5, "masked A4 producer wrote inactive row");
        }
    }
}
}  // namespace
int main() {
    try {
        r8::require(std::fesetround(FE_TONEAREST) == 0, "cannot set RNE");
        int count = 0;
        const auto e = hipGetDeviceCount(&count);
        if (e == hipErrorNoDevice || e == hipErrorInsufficientDriver || (e == hipSuccess && !count)) {
            std::puts("SKIP hip_moe_iu4_test: no HIP GPU"); return 77;
        }
        check(e, "device count");
        int tested = 0;
        for (int device = 0; device < count; ++device) {
            check(hipSetDevice(device), "select device");
            hipDeviceProp_t prop{};
            check(hipGetDeviceProperties(&prop, device), "properties");
            if (prop.warpSize != 32 || !strata::core::hip_moe_iu8_capable(prop.gcnArchName)) continue;
            hipStream_t stream = nullptr;
            check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
            producers(stream);
            check(hipStreamDestroy(stream), "free stream"); ++tested;
        }
        if (!tested) { std::puts("SKIP hip_moe_iu4_test: exact gfx wave32 required"); return 77; }
        std::puts("PASS hip_moe_iu4_test: A4 producer bits; quality requires separate KLD");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL hip_moe_iu4_test: %s\n", e.what()); return 1;
    }
}
#else
#if !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
static_assert(std::is_void_v<strata::prefill::iu4::Block>);
#endif
int main() { std::puts("SKIP hip_moe_iu4_test: HIP MMQ build required"); return 77; }
#endif
