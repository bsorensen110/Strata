// SPDX-License-Identifier: MIT
#include <cstdio>
#if defined(STRATA_USE_HIP)
#include "strata/kernels/qsa_decode_fp8_probe.hpp"
#include "strata/kernels/qsa_decode_fp8_reference.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <cstring>

namespace fp = strata::kernels::fp8_probe;
namespace ref = fp::reference;
namespace {
void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
struct Buffer {
    void* p = nullptr;
    explicit Buffer(size_t bytes) { check(hipMalloc(&p, bytes), "allocate"); }
    ~Buffer() {
        if (p && hipFree(p) != hipSuccess) { std::fputs("FAIL fp8-attn free\n", stderr); std::abort(); }
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
void upload(void* dst, const void* src, size_t bytes) {
    check(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice), "upload");
}
void upload_half(void* dst, const std::vector<float>& src) {
    std::vector<__half> half(src.size());
    for (size_t i = 0; i < src.size(); ++i) half[i] = __float2half(src[i]);
    upload(dst, half.data(), half.size() * sizeof(__half));
}
struct Case {
    ref::Fixture f;
    strata::kernels::QsaShapes s{};
    size_t scratch_floats;
    Buffer q, k, v, ids, pages, step, scratch, out;
    strata::kernels::QsaAttnPools pools{};
    Case(int group, int cap, bool missing)
        : f(group, cap, cap, 17, missing), scratch_floats(static_cast<size_t>((cap + 63) / 64) * f.kv_heads * group * 258 + 64),
          q(f.q.size() * 4), k(f.k.size() * 2), v(f.v.size() * 2),
          ids(f.ids.size() * 4), pages(f.pages.size() * 4), step(strata::kernels::kStepCount * 4),
          scratch((scratch_floats + 16) * 4), out((f.q.size() + 16) * 4) {
        s.n_head = group * f.kv_heads; s.n_head_kv = f.kv_heads; s.head_dim = 256; s.page_size = f.page_size;
        ref::require(scratch_floats == strata::kernels::qsa_decode_attn_scratch_floats(cap, s), "scratch ABI changed");
        upload(q.p, f.q.data(), f.q.size() * 4);
        upload_half(k.p, f.k); upload_half(v.p, f.v);
        upload(ids.p, f.ids.data(), f.ids.size() * 4); upload(pages.p, f.pages.data(), f.pages.size() * 4);
        pools.k_pool = static_cast<const uint16_t*>(k.p); pools.v_pool = static_cast<const uint16_t*>(v.p);
        pools.page_table = static_cast<const int32_t*>(pages.p);
        width(cap);
    }
    void width(int width) {
        f.width = width;
        int32_t state[strata::kernels::kStepCount] = {};
        state[strata::kernels::kStepWidth] = width;
        upload(step.p, state, sizeof(state));
    }
    fp::Refusal probe(bool on, hipStream_t stream) {
        return fp::launch_chunks(on, static_cast<const float*>(q.p), pools, static_cast<const int32_t*>(ids.p),
                                static_cast<const int32_t*>(step.p), f.cap, s, static_cast<float*>(scratch.p), 1, stream);
    }
    void merge(int v2, hipStream_t stream) {
        strata::kernels::qsa_decode_attn_merge_only(static_cast<const int32_t*>(step.p), f.cap, s,
            static_cast<const float*>(scratch.p), static_cast<float*>(out.p), 1, v2, stream);
    }
};

template <class Launch>
void refuse(Launch launch, fp::Refusal expected, hipStream_t stream) {
    hipGraph_t graph = nullptr;
    check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "refusal capture");
    const auto got = launch();
    check(hipStreamEndCapture(stream, &graph), "refusal end capture");
    size_t nodes = 0;
    check(hipGraphGetNodes(graph, nullptr, &nodes), "refusal nodes");
    check(hipGraphDestroy(graph), "refusal graph free");
    ref::require(got == expected && nodes == 0, "route did not decline without launching");
}

void parity(int group, hipStream_t stream) {
    for (int cap : {1, 63, 64, 65, 127, 128, 129, 257})
    for (bool missing : {false, true}) {
        Case c(group, cap, missing);
        for (int v2 : {0, 1}) {
            hipGraph_t graph = nullptr;
            check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "tile capture");
            ref::require(c.probe(true, stream) == fp::Refusal::none, "qualified tile declined");
            c.merge(v2, stream);
            check(hipStreamEndCapture(stream, &graph), "tile end capture");
            size_t nodes = 0;
            check(hipGraphGetNodes(graph, nullptr, &nodes), "tile nodes");
            ref::require(nodes == 2, "expected exactly tile + incumbent merge");
            hipGraphExec_t exec = nullptr;
            check(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
            for (int width : {0, 1, cap - 1, cap, cap + 1}) {
                c.width(width);  // replay must read the live device width, including a shrinking width
                check(hipMemset(c.scratch.p, 0xa5, (c.scratch_floats + 16) * 4), "scratch poison");
                check(hipMemset(c.out.p, 0xa5, (c.f.q.size() + 16) * 4), "output poison");
                check(hipGraphLaunch(exec, stream), "replay");
                check(hipStreamSynchronize(stream), "replay complete");
                std::vector<float> got(c.f.q.size() + 16);
                std::vector<uint32_t> scratch(c.scratch_floats + 16);
                check(hipMemcpy(got.data(), c.out.p, got.size() * 4, hipMemcpyDeviceToHost), "download output");
                check(hipMemcpy(scratch.data(), c.scratch.p, scratch.size() * 4, hipMemcpyDeviceToHost), "download scratch");
                const auto want = ref::attention(c.f);
                for (size_t i = 0; i < want.size(); ++i)
                    ref::require(std::isfinite(got[i]) && std::fabs(got[i] - want[i]) <= 5e-5f * std::max(1.0f, std::fabs(want[i])),
                                 "FP8 output differs from scalar paged FP64 reference");
                for (size_t i = c.scratch_floats - 64; i < scratch.size(); ++i)
                    ref::require(scratch[i] == 0xa5a5a5a5u, "scratch overrun");
                for (size_t i = want.size(); i < got.size(); ++i) {
                    uint32_t bits = 0; std::memcpy(&bits, &got[i], 4);
                    ref::require(bits == 0xa5a5a5a5u, "output overrun");
                }
            }
            check(hipGraphExecDestroy(exec), "free exec"); check(hipGraphDestroy(graph), "free graph");
        }
    }
}

void dispatch(Case& c, hipStream_t stream, bool admitted) {
    void* probe_function = nullptr;
    if (admitted) {
        hipGraph_t tile = nullptr;
        check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "dispatch reference capture");
        ref::require(c.probe(true, stream) == fp::Refusal::none, "dispatch reference declined");
        check(hipStreamEndCapture(stream, &tile), "dispatch reference end");
        hipGraphNode_t node = nullptr;
        size_t n = 1;
        check(hipGraphGetNodes(tile, &node, &n), "dispatch reference node");
        hipKernelNodeParams params{};
        check(hipGraphKernelNodeGetParams(node, &params), "dispatch reference function");
        probe_function = params.func;
        check(hipGraphDestroy(tile), "dispatch reference free");
    }
    for (int path = 0; path < 3; ++path) {
        hipGraph_t graph = nullptr;
        check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "dispatch capture");
        const auto* q = static_cast<const float*>(c.q.p);
        const auto* ids = static_cast<const int32_t*>(c.ids.p);
        const auto* step = static_cast<const int32_t*>(c.step.p);
        auto* scratch = static_cast<float*>(c.scratch.p);
        auto* out = static_cast<float*>(c.out.p);
        if (path == 0) strata::kernels::qsa_decode_attn_batch(q, c.pools, ids, step, c.f.cap, c.s, scratch, out, 1, stream);
        else if (path == 1) strata::kernels::qsa_decode_attn_step(q, c.pools, ids, step, c.f.cap, c.s, scratch, out, stream);
        else {
            strata::kernels::QsaAttnPoolsRows rows{};
            rows.p[0] = c.pools; rows.n = 1;
            strata::kernels::qsa_decode_attn_batch_rows(q, rows, ids, step, c.f.cap, c.s, scratch, out, 1, stream);
        }
        check(hipStreamEndCapture(stream, &graph), "dispatch end capture");
        size_t n = 0;
        check(hipGraphGetNodes(graph, nullptr, &n), "dispatch node count");
        ref::require(n == 2, "dispatch must capture chunk + merge");
        if (admitted) {
            std::vector<hipGraphNode_t> nodes(n);
            check(hipGraphGetNodes(graph, nodes.data(), &n), "dispatch nodes");
            int matches = 0;
            for (auto node : nodes) {
                hipKernelNodeParams params{};
                check(hipGraphKernelNodeGetParams(node, &params), "dispatch function");
                matches += params.func == probe_function;
            }
            ref::require(matches == 1, "production chooser bypassed the probe");
        }
        hipGraphExec_t exec = nullptr;
        check(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "dispatch instantiate");
        check(hipGraphLaunch(exec, stream), "dispatch replay");
        check(hipStreamSynchronize(stream), "dispatch finish");
        std::vector<float> got(c.f.q.size());
        check(hipMemcpy(got.data(), c.out.p, got.size() * 4, hipMemcpyDeviceToHost), "dispatch output");
        const auto want = ref::attention(c.f, admitted);
        for (size_t i = 0; i < want.size(); ++i)
            ref::require(std::isfinite(got[i]) && std::fabs(got[i] - want[i]) <= 5e-5f * std::max(1.0f, std::fabs(want[i])),
                         "production chooser output differs from admitted/fallback arithmetic");
        check(hipGraphExecDestroy(exec), "dispatch free exec"); check(hipGraphDestroy(graph), "dispatch free graph");
    }
}

void bench(int group, bool probe, hipStream_t stream) {
    Case c(group, 2048, false);
    const auto launch = [&] {
        if (probe) ref::require(c.probe(true, stream) == fp::Refusal::none, "benchmark probe declined");
        else strata::kernels::qsa_decode_attn_chunks_only(static_cast<const float*>(c.q.p), c.pools,
            static_cast<const int32_t*>(c.ids.p), static_cast<const int32_t*>(c.step.p), c.f.cap, c.s,
            static_cast<float*>(c.scratch.p), 1, 0, stream);
    };
    launch(); check(hipStreamSynchronize(stream), "warm");
    constexpr int burst = 128, replays = 32;
    hipGraph_t graph = nullptr;
    check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "bench capture");
    for (int i = 0; i < burst; ++i) launch();
    check(hipStreamEndCapture(stream, &graph), "bench end capture");
    size_t nodes = 0;
    check(hipGraphGetNodes(graph, nullptr, &nodes), "bench nodes");
    ref::require(nodes == burst, "per-launch benchmark contains unexpected work");
    hipGraphExec_t exec = nullptr;
    check(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "bench instantiate");
    check(hipGraphLaunch(exec, stream), "bench warm graph");
    check(hipStreamSynchronize(stream), "bench warm complete");
    hipEvent_t start = nullptr, end = nullptr;
    check(hipEventCreate(&start), "event start"); check(hipEventCreate(&end), "event end");
    check(hipEventRecord(start, stream), "record start");
    for (int i = 0; i < replays; ++i) check(hipGraphLaunch(exec, stream), "bench replay");
    check(hipEventRecord(end, stream), "record end"); check(hipEventSynchronize(end), "bench finish");
    float ms = 0;
    check(hipEventElapsedTime(&ms, start, end), "bench elapsed");
    std::printf("fp8-attn mode=%s cap=2048 group=%d launches=%d us_per_launch=%.6f (chunk pass only; no merge)\n",
                probe ? "probe" : "reference-kernel-0", group, burst * replays, ms * 1000 / (burst * replays));
    check(hipEventDestroy(start), "free start"); check(hipEventDestroy(end), "free end");
    check(hipGraphExecDestroy(exec), "bench free exec"); check(hipGraphDestroy(graph), "bench free graph");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const bool timing = argc == 2 && std::strcmp(argv[1], "--bench") == 0;
        ref::require(argc == 1 || timing, "usage: hip_fp8_attn_test [--bench]");
        bool on = false;
        ref::require(fp::resolve(std::getenv("STRATA_HIP_FP8_ATTN"), on), "environment requires 0 or 1");
        int count = 0;
        const auto e = hipGetDeviceCount(&count);
        if (e == hipErrorNoDevice || (e == hipSuccess && count == 0)) { std::puts("SKIP no HIP device"); return 77; }
        check(e, "device count");
        std::string error;
        if (!fp::initialize("1", error)) throw std::runtime_error(error);
        ref::require(!fp::initialize("0", error) && !fp::initialize("1junk", error) && fp::enabled(),
                     "probe request changed after pre-capture freeze");
        const auto support = fp::support();
        const auto& g = strata::core::kCompiledGeometry;
        const int group = static_cast<int>(g.n_head / g.n_head_kv);
        hipStream_t stream = nullptr;
        check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
        const auto expected = !fp::exact_arch(support.arch) ? fp::Refusal::arch
                            : !support.conversions ? fp::Refusal::conversion
                            : support.wave != 32 ? fp::Refusal::wave
                            : group != 8 && group != 12 ? fp::Refusal::shape : fp::Refusal::none;
        Case c(group, 129, false);
        refuse([&] { return c.probe(false, stream); }, fp::Refusal::disabled, stream);
        if (expected != fp::Refusal::none) {
            check(hipMemset(c.scratch.p, 0xa5, (c.scratch_floats + 16) * 4), "refusal poison");
            refuse([&] { return c.probe(true, stream); }, expected, stream);
            std::vector<uint32_t> untouched(c.scratch_floats + 16);
            check(hipMemcpy(untouched.data(), c.scratch.p, untouched.size() * 4, hipMemcpyDeviceToHost), "refusal download");
            for (auto bits : untouched) ref::require(bits == 0xa5a5a5a5u, "declined route wrote scratch");
            if (!timing) dispatch(c, stream, false);
            std::printf("PASS fp8-attn declined: %s (%s); no timing or quality claim\n", fp::reason(expected), support.arch);
        } else {
            c.pools.k_q = static_cast<const int8_t*>(c.k.p);
            refuse([&] { return c.probe(true, stream); }, fp::Refusal::format, stream);
            c.pools.k_q = nullptr;
            c.s.n_head = c.s.n_head_kv * 16;
            refuse([&] { return c.probe(true, stream); }, fp::Refusal::shape, stream);
            c.s.n_head = c.s.n_head_kv * group;
            refuse([&] {
                return fp::launch_chunks(true, static_cast<const float*>(c.q.p), c.pools,
                    static_cast<const int32_t*>(c.ids.p), static_cast<const int32_t*>(c.step.p), c.f.cap,
                    c.s, static_cast<float*>(c.scratch.p), 2, stream);
            }, fp::Refusal::queries, stream);
            if (timing) bench(group, on, stream);
            else {
                dispatch(c, stream, true);
                parity(group, stream);
                std::puts("PASS fp8-attn GPU: production choosers, scalar reference, both merges, live-width graph replay, canaries and refusals");
            }
        }
        check(hipStreamDestroy(stream), "free stream");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL fp8-attn: %s\n", e.what()); return 1;
    }
}
#else
int main() { std::puts("SKIP HIP-only FP8 attention probe"); return 77; }
#endif
