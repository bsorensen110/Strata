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
void producer(hipStream_t stream) {
    for (int n : {1,8,9,16,24,32,48,64}) {
        constexpr int k=2560;
        const auto x=ref::activations(k,n);
        const auto want=ref::quantize(x.data(),k,n);
        Buffer dx(x.size()*4),dq((want.size()+1)*sizeof(i8::Block));
        check(hipMemcpy(dx.p,x.data(),x.size()*4,hipMemcpyHostToDevice),"producer upload");
        check(hipMemset(dq.p,0xa5,(want.size()+1)*sizeof(i8::Block)),"producer canary");
        hipGraph_t graph=nullptr;
        check(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal),"producer capture");
        i8::quantize_projection((const float*)dx.p,(i8::Block*)dq.p,k,n,stream);
        check(hipStreamEndCapture(stream,&graph),"producer end capture");
        hipGraphExec_t exec=nullptr;
        check(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0),"producer instantiate");
        check(hipGraphLaunch(exec,stream),"producer replay");
        check(hipStreamSynchronize(stream),"producer complete");
        std::vector<i8::Block> got(want.size()+1);
        check(hipMemcpy(got.data(),dq.p,got.size()*sizeof(i8::Block),hipMemcpyDeviceToHost),"producer download");
        ref::require(std::memcmp(want.data(),got.data(),want.size()*sizeof(i8::Block))==0,"projection producer/A8 CPU bits");
        const auto* tail=(const uint8_t*)&got.back();
        for (size_t j=0;j<sizeof(i8::Block);++j) ref::require(tail[j]==0xa5,"projection producer overrun");
        check(hipGraphExecDestroy(exec),"producer free exec"); check(hipGraphDestroy(graph),"producer free graph");
    }
}
void fusion(hipStream_t stream) {
    for (int n : {9,16,24,32,48,49,64})
    for (int bits : {4,6})
    for (bool trunk_tiles : {false,true}) {
        const int k=trunk_tiles?128:2560;
        std::array<i8::Pack,4> packs;
        std::array<i8::DeviceWeight,4> weights;
        std::array<i8::Window,4> p{};
        std::array<std::vector<float>,4> want,single,got;
        const auto x=ref::quantize(ref::activations(k,n).data(),k,n);
        Buffer dx(x.size()*sizeof(i8::Block));
        check(hipMemcpy(dx.p,x.data(),x.size()*sizeof(i8::Block),hipMemcpyHostToDevice),"upload A8");
        std::vector<std::unique_ptr<Buffer>> out;
        for (int i=0;i<4;++i) {
            const int m=i==0?(trunk_tiles?(n>16 && n<=32?6145:4097):65):i==1?33:i==2?17:1;
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
            size_t nodes=0;
            check(hipGraphGetNodes(graph,nullptr,&nodes),"x4 node count");
            ref::require(nodes==1,"x4 must capture one kernel node");
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
void dot4(hipStream_t stream) {
    for (int n=1;n<=8;++n) {
        constexpr int k=2560,m=33,ld=36;
        const auto w=ref::projection_pack(m,k,6,191);
        const auto x=ref::quantize(ref::activations(k,n).data(),k,n);
        i8::DeviceWeight dw;
        std::string why;
        ref::require(i8::upload(w,dw,why),why.c_str());
        std::vector<float> want((size_t)n*ld+19,123.0f),got=want;
        for (int t=0;t<n;++t)
        for (int r=0;r<m;++r) want[(size_t)t*ld+r]=ref::fold(w,x.data(),n,t,r);
        Buffer dx(x.size()*sizeof(i8::Block)),dy(want.size()*4);
        check(hipMemcpy(dx.p,x.data(),x.size()*sizeof(i8::Block),hipMemcpyHostToDevice),"dot4 upload");
        for (int t=0;t<n;++t)
            std::fill_n(got.data()+(size_t)t*ld,m,0.0f);
        check(hipMemcpy(dy.p,got.data(),got.size()*4,hipMemcpyHostToDevice),"dot4 poison");
        i8::Window p;
        p.weight=i8::view(dw); p.x=(const i8::Block*)dx.p; p.y=dy.p;
        p.rows=m; p.cols=k; p.tokens=p.x_rows=n; p.ld=ld;
        ref::require(!i8::strata_gemv_sym_i8_dot4(p,false,stream),"disabled dot4 launched");
        hipGraph_t graph=nullptr;
        check(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal),"dot4 capture");
        ref::require(i8::strata_gemv_sym_i8_dot4(p,true,stream),"dot4 refused");
        check(hipStreamEndCapture(stream,&graph),"dot4 end capture");
        hipGraphExec_t exec=nullptr;
        check(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0),"dot4 instantiate");
        for (int replay=0;replay<2;++replay) {
            check(hipGraphLaunch(exec,stream),"dot4 replay");
            check(hipStreamSynchronize(stream),"dot4 complete");
            check(hipMemcpy(got.data(),dy.p,got.size()*4,hipMemcpyDeviceToHost),"dot4 download");
            ref::require(std::memcmp(want.data(),got.data(),want.size()*4)==0,"dot4 device/CPU bits and padding");
        }
        check(hipGraphExecDestroy(exec),"dot4 free exec"); check(hipGraphDestroy(graph),"dot4 free graph");
        i8::launch_sym_i8_member(p,3,stream);
        check(hipStreamSynchronize(stream),"dot4 WMMA twin finish");
        std::vector<float> twin(want.size());
        check(hipMemcpy(twin.data(),dy.p,twin.size()*4,hipMemcpyDeviceToHost),"dot4 WMMA twin download");
        ref::require(std::memcmp(twin.data(),got.data(),got.size()*4)==0,"dot4/standalone WMMA bits");
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
            producer(stream);
            fusion(stream);
            dot4(stream);
            check(hipStreamDestroy(stream),"free stream"); ++tested;
        }
        if (!tested) { std::puts("SKIP hip_projection_test: exact gfx1201 required"); return 77; }
        std::puts("PASS hip_projection_test: x4/dot4/standalone/CPU bits, tails, graphs"); return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"FAIL hip_projection_test: %s\n",e.what()); return 1; }
}
#else
int main() { std::puts("SKIP hip_projection_test: HIP MMQ build required"); return 77; }
#endif
