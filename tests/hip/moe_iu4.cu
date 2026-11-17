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
    void products(hipStream_t stream) {
        namespace i8 = strata::prefill::iu8;
        for (const auto& shape : {std::array{17,128,5}, std::array{129,384,129}, std::array{1153,128,129}}) {
            const int m = shape[0], k = shape[1], n = shape[2], ld = m+3;
            const auto src = r8::source(GGML_TYPE_Q4_0,m,k);
            i8::Pack w;
            i8::DeviceWeight dw;
            std::string why;
            r8::require(i8::pack_exact(GGML_TYPE_Q4_0,src.data(),src.size(),m,k,w,why), why.c_str());
            r8::require(i8::upload(w,dw,why), why.c_str());
            const auto x = r4::quantize(r8::activations(k,n).data(),k,n);
            std::vector<uint16_t> zeros(w.scales.size(),0xb000);
            Buffer dx(x.size()*sizeof(i4::Block)), dz(zeros.size()*2);
            const size_t words = (size_t)n*ld+17;
            Buffer dy(words*4);
            upload(dx,x); upload(dz,zeros);
            for (int mode = 0; mode < 3; ++mode) {
                std::vector<float> want(words),got(words);
                std::memset(want.data(),0xa5,words*4);
                const bool symmetric = mode == 0;
                for (int t = 0; t < n; ++t)
                for (int r = 0; r < m; ++r)
                    want[(size_t)t*ld+r] = r4::fold(w,x.data(),n,t,r,symmetric,mode==2?zeros.data():nullptr);
                i4::Product p;
                p.weight.codes=dw.codes; p.weight.scales=dw.scales;
                p.weight.zero_points=mode==2?(const uint16_t*)dz.p:nullptr;
                p.weight.rows=m; p.weight.cols=k; p.weight.device=dw.device; p.weight.symmetric_verified=true;
                p.x=(const i4::Block*)dx.p; p.y=(float*)dy.p;
                p.rows=m; p.cols=k; p.tokens=p.x_rows=n; p.ld=ld; p.symmetric=symmetric;
                for (bool banded : {false,true}) {
                    p.banded=banded;
                    check(hipMemsetAsync(dy.p,0xa5,words*4,stream),"IU4 output poison");
                    hipGraph_t graph=nullptr;
                    check(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal),"IU4 capture");
                    i4::run(p,stream);
                    check(hipStreamEndCapture(stream,&graph),"IU4 end capture");
                    hipGraphExec_t exec=nullptr;
                    check(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0),"IU4 instantiate");
                    for (int repeat=0;repeat<2;++repeat) {
                        check(hipGraphLaunch(exec,stream),"IU4 replay");
                        check(hipStreamSynchronize(stream),"IU4 completion");
                        download(dy,got);
                        r8::require(std::memcmp(want.data(),got.data(),words*4)==0,
                                    "IU4 device differs from own exact-fold CPU/canaries");
                    }
                    void window_groups(hipStream_t stream) {
                        namespace i8=strata::prefill::iu8;
                        constexpr int n=24,k=256,m=65,ld=68,cap=48,prefix=2,ne=3;
                        const int counts[ne]={1,5,24};
                        std::vector<int32_t> starts{prefix},tok(cap),dst(cap),route(cap);
                        for (int e=0;e<ne;++e) {
                            for (int j=0;j<counts[e];++j) {
                                const int s=starts.back()+j;
                                tok[s]=(j*5+e)%n; dst[s]=cap-1-(s-prefix); route[dst[s]]=e;
                            }
                            starts.push_back(starts.back()+counts[e]);
                        }
                        i8::Layer layer;
                        std::vector<i8::Pack> packs(ne);
                        for (int e=0;e<ne;++e) {
                            const auto src=r8::source(GGML_TYPE_Q4_0,m,k,71+e);
                            std::string why;
                            r8::require(i8::pack_exact(GGML_TYPE_Q4_0,src.data(),src.size(),m,k,packs[e],why),why.c_str());
                            i8::Expert expert;
                            r8::require(i8::upload(packs[e],expert.gu,why),why.c_str());
                            r8::require(i8::upload(packs[e],expert.down,why),why.c_str());
                            layer.experts.push_back(std::move(expert));
                        }
                        i4::WindowWeights table; r8::require(table.upload(layer),"IU4 exact W4 table refused");
                        const auto x=r4::quantize(r8::activations(k,n).data(),k,n);
                        Buffer dtok(cap*4),ddst(cap*4),droute(cap*4),dstart(starts.size()*4),dcount(4);
                        upload(dtok,tok); upload(ddst,dst); upload(droute,route); upload(dstart,starts); upload(dcount,std::vector<int32_t>{ne});
                        strata::prefill::mmq::Context context;
                        for (bool sorted_x : {false,true})
                        for (bool sorted_y : {false,true}) {
                            std::vector<i4::Block> input=sorted_x?std::vector<i4::Block>((k/128)*cap):x;
                            if (sorted_x)
                                for (int kb=0;kb<k/128;++kb)
                                for (int s=prefix;s<starts.back();++s) input[kb*cap+s]=x[kb*n+tok[s]];
                            Buffer dx(input.size()*sizeof(i4::Block)),dy(((size_t)cap*ld+17)*4);
                            upload(dx,input);
                            std::vector<float> want((size_t)cap*ld+17),got(want.size());
                            std::memset(want.data(),0xa5,want.size()*4);
                            for (int e=0;e<ne;++e)
                            for (int s=starts[e];s<starts[e+1];++s)
                            for (int r=0;r<m;++r) want[(size_t)(sorted_y?s:dst[s])*ld+r]=r4::fold(packs[e],x.data(),n,tok[s],r,true);
                            i4::Product p;
                            p.table=table.gu; p.count=(const int32_t*)dcount.p; p.starts=(const int32_t*)dstart.p;
                            p.tok=(const int32_t*)dtok.p; p.dst=(const int32_t*)ddst.p; p.route=(const int32_t*)droute.p;
                            p.x=(const i4::Block*)dx.p; p.y=(float*)dy.p; p.rows=m; p.cols=k; p.tokens=n;
                            p.x_rows=sorted_x?cap:n; p.groups=cap; p.ld=ld; p.sorted_x=sorted_x; p.sorted_y=sorted_y;
                            check(hipMemsetAsync(dy.p,0xa5,want.size()*4,stream),"IU4 group poison");
                            hipGraph_t graph=nullptr;
                            check(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal),"IU4 group capture");
                            strata::prefill::mmq::Product call; call.hip_iu4=&p; context.run(call,stream);
                            check(hipStreamEndCapture(stream,&graph),"IU4 group end capture");
                            hipGraphExec_t exec=nullptr;
                            check(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0),"IU4 group instantiate");
                            check(hipGraphLaunch(exec,stream),"IU4 group replay"); check(hipStreamSynchronize(stream),"IU4 group done");
                            download(dy,got);
                            r8::require(std::memcmp(want.data(),got.data(),want.size()*4)==0,"IU4 dynamic expert/token/destination map drift");
                            check(hipGraphExecDestroy(exec),"IU4 group free exec"); check(hipGraphDestroy(graph),"IU4 group free graph");
                        }
                    }
                    check(hipGraphExecDestroy(exec),"IU4 exec free");
                    check(hipGraphDestroy(graph),"IU4 graph free");
                }
            }
        }
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
            if (!i4::usable()) continue;
            hipStream_t stream = nullptr;
            check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
            producers(stream);
            products(stream);
            window_groups(stream);
            check(hipStreamDestroy(stream), "free stream"); ++tested;
        }
        if (!tested) { std::puts("SKIP hip_moe_iu4_test: exact gfx wave32 required"); return 77; }
        std::puts("PASS hip_moe_iu4_test: own A4/fold bits, affine/XOR/raster/tails/capture; KLD separate");
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
