// SPDX-License-Identifier: MIT
// Port note: hipfire block_i8_128_quant.hip, gemm_mq6sym_i8.gfx1201.hip
// and CHANGELOG.md:856. Exact CPU-fold parity, not a speed/KLD receipt.
#include "strata/prefill/moe_iu8_gfx.hpp"
#include <cstdio>
#include <type_traits>

#if defined(__HIPCC__) && defined(STRATA_MOE_IU8_CPU_ORACLE)
#include "strata/prefill/moe_iu8_reference.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/core/arch_defaults.hpp"
#include "mmq_resident_sort.hpp"
#include <hip/hip_runtime.h>
#include <cfenv>
#include <numeric>
#include <utility>

namespace iu8 = strata::prefill::iu8;
namespace ref = iu8::reference;
namespace detail = strata::prefill::detail;

namespace {
void check(hipError_t e, const char* message) {
    if (e != hipSuccess) throw std::runtime_error(std::string(message) + ": " + hipGetErrorString(e));
}
struct Buffer {
    void* p = nullptr;
    explicit Buffer(size_t bytes) { check(hipMalloc(&p, bytes), "allocate test buffer"); }
    ~Buffer() {
        if (p && hipFree(p) != hipSuccess) {
            std::fputs("FAIL hip_moe_iu8_test: free buffer\n", stderr); std::abort();
        }
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
template<class T>
void upload(Buffer& b, const std::vector<T>& x) {
    check(hipMemcpy(b.p, x.data(), x.size() * sizeof(T), hipMemcpyHostToDevice), "upload test data");
}
template<class T>
void download(const Buffer& b, std::vector<T>& x) {
    check(hipMemcpy(x.data(), b.p, x.size() * sizeof(T), hipMemcpyDeviceToHost), "download test data");
}

void producers(hipStream_t stream) {
    constexpr int k = 256, n = 65, topk = 2;
    auto x = ref::activations(k, n);
    // RNE ties and the tiny-value divide fallback, in separate K128 blocks.
    for (int j = 0; j < 128; ++j) x[k + j] = j == 127 ? 127.0f : (float) j - 63.5f;
    for (int j = 0; j < 128; ++j) x[k + 128 + j] = std::ldexp((float) (j - 64), -125);
    const auto expected = ref::quantize(x.data(), k, n);
    Buffer dx(x.size() * sizeof(float)), dq((expected.size() + 1) * sizeof(iu8::Block));
    upload(dx, x);
    check(hipMemset(dq.p, 0xa5, (expected.size() + 1) * sizeof(iu8::Block)), "quantizer poison");
    iu8::quantize((const float*) dx.p, nullptr, (iu8::Block*) dq.p, k, k, n, stream);
    check(hipStreamSynchronize(stream), "quantizer completion");
    std::vector<iu8::Block> got(expected.size() + 1);
    download(dq, got);
    ref::require(std::memcmp(expected.data(), got.data(), expected.size() * sizeof(iu8::Block)) == 0,
                 "producer differs from IEEE reference (zero/ties/tiny/tail)");
    const auto* tail = (const uint8_t*) &got.back();
    for (size_t j = 0; j < sizeof(iu8::Block); ++j) ref::require(tail[j] == 0xa5, "producer overwrote tail");

    std::vector<int32_t> slots(n * topk);
    for (int i = 0; i < n * topk; ++i) slots[i] = n * topk - 1 - i;
    Buffer ds(slots.size() * sizeof(int32_t)), scattered(expected.size() * topk * sizeof(iu8::Block));
    upload(ds, slots);
    iu8::quantize_scatter((const float*) dx.p, (const int32_t*) ds.p, (iu8::Block*) scattered.p,
                          k, k, n, topk, stream);
    check(hipStreamSynchronize(stream), "scatter completion");
    std::vector<iu8::Block> routed(expected.size() * topk);
    download(scattered, routed);
    for (int kb = 0; kb < k / 128; ++kb)
    for (int token = 0; token < n; ++token)
    for (int rank = 0; rank < topk; ++rank)
        ref::require(std::memcmp(&routed[kb * n * topk + slots[token * topk + rank]],
                                &expected[kb * n + token], sizeof(iu8::Block)) == 0,
                     "scatter sidecar differs by route rank");

    auto gu = ref::activations(2 * k, n, 67);
    Buffer dg(gu.size() * sizeof(float)), dh((size_t) k * n * sizeof(float));
    Buffer dh_ref((size_t) k * n * sizeof(float)), q_ref(expected.size() * sizeof(iu8::Block));
    upload(dg, gu);
    iu8::swiglu_quantize((const float*) dg.p, (float*) dh.p, (iu8::Block*) dq.p, k, n, stream);
    strata::prefill::mmq::swiglu((const float*) dg.p, (float*) dh_ref.p, n, k, false, stream);
    iu8::quantize((const float*) dh_ref.p, nullptr, (iu8::Block*) q_ref.p, k, k, n, stream);
    check(hipStreamSynchronize(stream), "SwiGLU completion");
    std::vector<float> h((size_t) k * n), h_ref(h.size());
    download(dh, h); download(dh_ref, h_ref);
    ref::require(std::memcmp(h.data(), h_ref.data(), h.size() * sizeof(float)) == 0,
                 "fused SwiGLU changed incumbent arithmetic");
    std::vector<iu8::Block> split(expected.size());
    download(q_ref, split); download(dq, got);
    ref::require(std::memcmp(split.data(), got.data(), split.size() * sizeof(iu8::Block)) == 0,
                 "producer-owned SwiGLU sidecar differs from standalone quantizer");
}

void grouped(int type, int m, int k, hipStream_t stream, strata::prefill::mmq::Context& context) {
    constexpr int ne = 6, prefix = 7;
    const int counts[ne] = {0, 1, 127, 128, 129, 0};
    int32_t bounds[ne + 1] = {prefix};
    for (int e = 0; e < ne; ++e) bounds[e + 1] = bounds[e] + counts[e];
    const int n = bounds[ne], ld = m + 3;
    auto x = ref::activations(k, n);
    const auto blocks = ref::quantize(x.data(), k, n);
    std::vector<iu8::Pack> packs(ne);
    std::vector<iu8::DeviceWeight> weights(ne);
    iu8::Product p;
    ref::require(detail::mmq_iu8_tile_ranges(bounds, ne, p), "valid tile prefix refused");
    iu8::Product bad;
    const int32_t invalid_bounds[] = {2, 1};
    ref::require(!detail::mmq_iu8_tile_ranges(invalid_bounds, 1, bad), "descending tile bounds admitted");
    for (int e = 0; e < ne; ++e) {
        const auto src = ref::source(type, m, k, (uint32_t) (e + 29));
        std::string why;
        const bool packed = iu8::pack_exact(type, src.data(), src.size(), m, k, packs[e], why);
        ref::require(packed, why.c_str());
        const bool uploaded = iu8::upload(packs[e], weights[e], why);
        ref::require(uploaded, why.c_str());
        p.weights[e] = iu8::view(weights[e]);
    }
    void narrow_tiles(hipStream_t stream) {
        for (int type : {GGML_TYPE_Q4_0, GGML_TYPE_Q6_K}) {
            constexpr int m = 65, k = 512, ld = 68;
            const auto src = ref::source(type, m, k);
            iu8::Pack w;
            iu8::DeviceWeight dw;
            std::string why;
            ref::require(iu8::pack_exact(type, src.data(), src.size(), m, k, w, why), why.c_str());
            ref::require(iu8::upload(w, dw, why), why.c_str());
            for (int n : {1, 5, 9, 16, 17, 24, 48}) {
                const auto x = ref::quantize(ref::activations(k, n).data(), k, n);
                Buffer dx(x.size() * sizeof(iu8::Block));
                const size_t words = (size_t) n * ld + 17;
                Buffer dy(words * sizeof(float)), baseline(words * sizeof(float));
                upload(dx, x);
                iu8::Product pre;
                pre.weights[0] = iu8::view(dw); pre.ranges[0] = {0,n,0};
                pre.n = pre.tiles = 1; pre.total_rows = n; pre.tokens = 64;
                pre.x = (const iu8::Block*) dx.p; pre.dst = (float*) baseline.p; pre.ld = ld;
                check(hipMemsetAsync(baseline.p, 0xa5, words * sizeof(float), stream), "prefill poison");
                iu8::run(pre, stream);
                check(hipStreamSynchronize(stream), "prefill cross-arm");
                std::vector<float> expected(words), got(words);
                download(baseline, expected);
                for (int t = 0; t < n; ++t)
                for (int r = 0; r < m; ++r) {
                    const float f = ref::fold(w, x.data(), n, t, r);
                    ref::require(std::memcmp(&f, &expected[(size_t) t * ld + r], 4) == 0, "prefill cross-arm CPU mismatch");
                }
                for (int entry = 0; entry < 24; ++entry) {
                    const auto tile = iu8::kNarrowTiles[entry];
                    if (!tile.split && n > tile.tokens) continue;
                    iu8::Window p;
                    p.weight = iu8::view(dw); p.x = pre.x; p.y = dy.p;
                    p.rows = m; p.cols = k; p.tokens = p.x_rows = n; p.ld = ld;
                    check(hipMemsetAsync(dy.p, 0xa5, words * sizeof(float), stream), "narrow poison");
                    hipGraph_t graph = nullptr;
                    check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "narrow capture");
                    iu8::run_narrow(p, entry, stream);
                    check(hipStreamEndCapture(stream, &graph), "end narrow capture");
                    hipGraphExec_t exec = nullptr;
                    check(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate narrow");
                    check(hipGraphLaunch(exec, stream), "replay narrow");
                    check(hipStreamSynchronize(stream), "narrow completion");
                    if (!tile.bf16) {
                        download(dy, got);
                        ref::require(std::memcmp(got.data(), expected.data(), words * sizeof(float)) == 0,
                                     "narrow/pre-fill GPU bits or tail canaries differ");
                    } else {
                        std::vector<uint16_t> bits(words * 2, 0xa5a5), want(bits);
                        for (int t = 0; t < n; ++t)
                        for (int r = 0; r < m; ++r) {
                            const float f = 0.0f + expected[(size_t) t * ld + r];
                            uint32_t u; std::memcpy(&u, &f, 4);
                            want[(size_t) t * ld + r] = (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
                        }
                        download(dy, bits);
                        ref::require(bits == want, "narrow bf16 bits/canaries differ");
                    }
                    check(hipGraphExecDestroy(exec), "free narrow exec");
                    check(hipGraphDestroy(graph), "free narrow graph");
                }
            }
        }
    }
    std::vector<int32_t> ids(n);
    for (int i = 0; i < n; ++i) ids[i] = n - 1 - i;
    Buffer dx(x.size() * sizeof(float)), dq((blocks.size() + 1) * sizeof(iu8::Block));
    Buffer dy(((size_t) n * ld + 32) * sizeof(float)), did(ids.size() * sizeof(int32_t));
    upload(dx, x); upload(did, ids);
    iu8::quantize((const float*) dx.p, nullptr, (iu8::Block*) dq.p, k, k, n, stream);
    p.total_rows = n; p.tokens = 65; p.x = (const iu8::Block*) dq.p;
    p.dst = (float*) dy.p; p.ld = ld; p.ids = (const int32_t*) did.p;
    std::vector<float> expected((size_t) n * ld + 32), first(expected.size()), second(expected.size());
    std::memset(expected.data(), 0xa5, expected.size() * sizeof(float));
    for (int e = 0; e < ne; ++e)
    for (int token = bounds[e]; token < bounds[e + 1]; ++token)
    for (int r = 0; r < m; ++r)
        expected[(size_t) ids[token] * ld + r] = ref::fold(packs[e], blocks.data(), n, token, r);
    for (auto* result : {&first, &second}) {
        check(hipMemsetAsync(dy.p, 0xa5, expected.size() * sizeof(float), stream), "group poison");
        strata::prefill::mmq::Product route;
        route.hip_iu8 = &p;
        context.run(route, stream);
        check(hipStreamSynchronize(stream), "group completion");
        download(dy, *result);
        ref::require(std::memcmp(result->data(), expected.data(), expected.size() * sizeof(float)) == 0,
                     "GPU grouped fold/scatter/canaries differ from CPU bits");
    }
    ref::require(std::memcmp(first.data(), second.data(), first.size() * sizeof(float)) == 0,
                 "GPU grouped run is nondeterministic");
    std::printf("PASS grouped W%d M=%d K=%d: empty/1/127/128/129 slots, offset, ids, tails, repeat\n",
                packs[0].bits, m, k);
}
}  // namespace

int main() {
    try {
        ref::require(std::fesetround(FE_TONEAREST) == 0, "cannot set CPU RNE");
        int count = 0;
        const hipError_t e = hipGetDeviceCount(&count);
        if (e == hipErrorNoDevice || e == hipErrorInsufficientDriver || (e == hipSuccess && count == 0)) {
            std::puts("SKIP hip_moe_iu8_test: no HIP GPU"); return 77;
        }
        check(e, "device count");
        int tested = 0;
        for (int device = 0; device < count; ++device) {
            check(hipSetDevice(device), "select device");
            if (!iu8::usable()) continue;
            hipStream_t stream = nullptr;
            check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "test stream");
            ref::require(!iu8::admitted(63, 128, stream) && !iu8::admitted(64, 1088, stream) &&
                         iu8::admitted(64, 128, stream), "prefill admission mismatch");
            check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "begin admission capture");
            ref::require(!iu8::admitted(64, 128, stream), "capture must decline IU8");
            hipGraph_t graph = nullptr;
            check(hipStreamEndCapture(stream, &graph), "end admission capture");
            check(hipGraphDestroy(graph), "free capture graph");
            producers(stream);
            narrow_tiles(stream);
            strata::prefill::mmq::Context context;
            for (const auto& shape : {std::pair{17, 128}, std::pair{129, 256},
                                      std::pair{128, 512}, std::pair{33, 1280}}) {
                grouped(GGML_TYPE_Q4_0, shape.first, shape.second, stream, context);
                if (shape.second % 256 == 0) grouped(GGML_TYPE_Q6_K, shape.first, shape.second, stream, context);
            }
            check(hipStreamDestroy(stream), "free test stream");
            ++tested;
        }
        if (!tested) { std::puts("SKIP hip_moe_iu8_test: needs flag=1 and exact supported gfx"); return 77; }
        std::puts("PASS hip_moe_iu8_test: GPU bits equal CPU fold; no performance/KLD claim");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL hip_moe_iu8_test: %s\n", e.what()); return 1;
    }
}
#else
#if !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
static_assert(std::is_void_v<strata::prefill::iu8::Block>);
static_assert(std::is_void_v<strata::prefill::iu8::Product>);
#endif
int main() { std::puts("SKIP hip_moe_iu8_test: HIP MMQ build required"); return 77; }
#endif
