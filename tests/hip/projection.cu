// SPDX-License-Identifier: MIT
// Port note: hipfire gemm_mq6sym_i8.gfx1201.hip bitwise per-member x4.
#include <cstdio>
#if defined(__HIPCC__) && defined(STRATA_MOE_IU8_CPU_ORACLE)
#include "strata/prefill/hip_projection_reference.hpp"
#include <hip/hip_runtime.h>
#include <cfenv>
#include <memory>
#include "strata/kernels/gfx_arch.hpp"
namespace i8=strata::prefill::iu8;
namespace ref=i8::reference;
namespace {
void check(hipError_t e, const char* what) {
    if (e!=hipSuccess) throw std::runtime_error(std::string(what)+": "+hipGetErrorString(e));
}
struct Buffer {
    void* p=nullptr;
    explicit Buffer(size_t size) { check(hipMalloc(&p,size),"allocate"); }
    ~Buffer() {
        if (p && hipFree(p)!=hipSuccess) { std::fputs("FAIL projection free\n",stderr); std::abort(); }
    }
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
};
void fusion(hipStream_t stream) {
    for (int n : {9,16,24,32,48,49,64})
    for (int bits : {4,6}) {
        constexpr int k=2560;
        std::array<i8::Pack,4> packs;
        std::array<i8::DeviceWeight,4> weights;
        std::array<i8::Window,4> p{};
        std::array<std::vector<float>,4> want,single,got;
        const auto x=ref::quantize(ref::activations(k,n).data(),k,n);
        Buffer dx(x.size()*sizeof(i8::Block));
        check(hipMemcpy(dx.p,x.data(),x.size()*sizeof(i8::Block),hipMemcpyHostToDevice),"upload A8");
        std::vector<std::unique_ptr<Buffer>> out;
        for (int i=0;i<4;++i) {
            const int m=i==0?65:i==1?33:i==2?17:1;
            packs[i]=ref::projection_pack(m,k,bits,141+i);
            std::string why;
            ref::require(i8::upload(packs[i],weights[i],why),why.c_str());
            want[i].resize((size_t)n*m+19);
            std::memset(want[i].data(),0xa5,want[i].size()*4);
            single[i]=got[i]=want[i];
            for (int t=0;t<n;++t)
            for (int r=0;r<m;++r) want[i][(size_t)t*m+r]=ref::fold(packs[i],x.data(),n,t,r);
            out.push_back(std::make_unique<Buffer>(want[i].size()*4));
            p[i].weight=i8::view(weights[i]); p[i].x=(const i8::Block*)dx.p;
            p[i].y=out.back()->p; p[i].rows=m; p[i].cols=k; p[i].tokens=p[i].x_rows=n; p[i].ld=m;
        }
        for (int count : {2,3,4}) {
            auto members=p;
            for (int i=count;i<4;++i) members[i]={};
            std::array<int,4> rows{};
            for (int i=0;i<count;++i) rows[i]=p[i].rows;
            const auto plan=i8::x4_plan(rows,count,k,n);
            for (int i=0;i<count;++i) {
                check(hipMemsetAsync(out[i]->p,0xa5,want[i].size()*4,stream),"poison standalone");
                i8::launch_sym_i8_member(p[i],plan.entry,stream);
            }
            check(hipStreamSynchronize(stream),"standalone finish");
            for (int i=0;i<count;++i) {
                check(hipMemcpy(single[i].data(),out[i]->p,want[i].size()*4,hipMemcpyDeviceToHost),"download standalone");
                ref::require(std::memcmp(want[i].data(),single[i].data(),want[i].size()*4)==0,"standalone/CPU bits");
                check(hipMemsetAsync(out[i]->p,0xa5,want[i].size()*4,stream),"poison fused");
            }
            hipGraph_t graph=nullptr;
            check(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal),"capture");
            ref::require(i8::strata_gemm_sym_i8_x4(members,count,true,stream),"x4 refused exact GPU");
            check(hipStreamEndCapture(stream,&graph),"end capture");
            hipGraphExec_t exec=nullptr;
            check(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0),"instantiate");
            for (int repeat=0;repeat<2;++repeat) {
                check(hipGraphLaunch(exec,stream),"replay");
                check(hipStreamSynchronize(stream),"finish");
                for (int i=0;i<count;++i) {
                    check(hipMemcpy(got[i].data(),out[i]->p,want[i].size()*4,hipMemcpyDeviceToHost),"download fused");
                    ref::require(std::memcmp(single[i].data(),got[i].data(),want[i].size()*4)==0,"x4/standalone bits/canaries");
                }
            }
            check(hipGraphExecDestroy(exec),"free exec"); check(hipGraphDestroy(graph),"free graph");
            std::reverse(members.begin(),members.begin()+count);
            ref::require(i8::strata_gemm_sym_i8_x4(members,count,true,stream),"reversed x4 refused");
            check(hipStreamSynchronize(stream),"reversed finish");
            for (int i=0;i<count;++i) {
                check(hipMemcpy(got[i].data(),out[i]->p,want[i].size()*4,hipMemcpyDeviceToHost),"reversed download");
                ref::require(std::memcmp(single[i].data(),got[i].data(),want[i].size()*4)==0,"reversed x4 bits");
            }
        }
        ref::require(!i8::strata_gemm_sym_i8_x4(p,4,false,stream),"disabled x4 launched");
        auto bad=p; bad[1].x=(const i8::Block*)((const char*)dx.p+sizeof(i8::Block));
        bool refused=false;
        try { i8::strata_gemm_sym_i8_x4(bad,4,true,stream); } catch (const std::invalid_argument&) { refused=true; }
        ref::require(refused,"mismatched prepared X admitted");
    }
}
}  // namespace
int main() {
    try {
        ref::require(std::fesetround(FE_TONEAREST)==0,"RNE");
        int count=0;
        const auto e=hipGetDeviceCount(&count);
        if (e==hipErrorNoDevice || e==hipErrorInsufficientDriver || (e==hipSuccess && count==0)) return 77;
        check(e,"device count");
        int tested=0;
        for (int d=0;d<count;++d) {
            check(hipSetDevice(d),"device");
            hipDeviceProp_t prop{}; check(hipGetDeviceProperties(&prop,d),"properties");
            if (prop.warpSize!=32 || !strata::kernels::gfx_arch_is(prop.gcnArchName,"gfx1201")) continue;
            hipStream_t stream=nullptr; check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking),"stream");
            fusion(stream);
            check(hipStreamDestroy(stream),"free stream"); ++tested;
        }
        if (!tested) { std::puts("SKIP hip_projection_test: exact gfx1201 required"); return 77; }
        std::puts("PASS hip_projection_test: fused/standalone/CPU bits, tails, graphs"); return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"FAIL hip_projection_test: %s\n",e.what()); return 1; }
}
#else
int main() { std::puts("SKIP hip_projection_test: HIP MMQ build required"); return 77; }
#endif
