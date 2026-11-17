// SPDX-License-Identifier: MIT
// Port note: hipfire gemm_mq6sym_i8.gfx1201.hip per-member x4 contract.
#include "strata/prefill/hip_projection_reference.hpp"
#include <cfenv>
#include <cstdio>

namespace i8=strata::prefill::iu8;
namespace ref=i8::reference;
namespace {
void fusion() {
    // Keep K=2560, scale only the output rows for fast host CI. Exercise actual
    // trunk M sizes in the planner, without billions of redundant scalar MACs.
    for (int n : {9,16,24,32,48,49,64}) {
        for (const auto& rows : {std::array{10240,6144,48,48},std::array{12288,512,512,640}}) {
            const auto plan=i8::x4_plan(rows,4,2560,n);
            const int expected=n<=16?3:n<=32?1:n<=48?5:9;
            ref::require(plan.entry==expected,"trunk largest-member/token-split pick");
        }
        const auto x=ref::quantize(ref::activations(2560,n).data(),2560,n);
        for (int bits : {4,6}) {
            std::array<i8::Pack,4> w{};
            for (int i=0;i<4;++i) w[i]=ref::projection_pack(i==0?17:i==1?5:i==2?3:1,2560,bits,101+i);
            for (int count : {2,3,4}) {
                if (count>2 && n!=24) continue;
                std::array<std::vector<float>,4> want,got;
                for (int i=0;i<count;++i) {
                    want[i].resize((size_t)n*w[i].rows+19);
                    std::memset(want[i].data(),0xa5,want[i].size()*4);
                    got[i]=want[i];
                    for (int t=0;t<n;++t)
                    for (int r=0;r<w[i].rows;++r)
                        want[i][(size_t)t*w[i].rows+r]=ref::fold(w[i],x.data(),n,t,r);
                }
                ref::projection_x4(w,count,x.data(),n,got);
                for (int i=0;i<count;++i)
                    ref::require(std::memcmp(want[i].data(),got[i].data(),want[i].size()*4)==0,
                                 "x4 per-member bits or sentinel tail differ");
                // A wrong member or output offset must be observable, not all-zero fixtures.
                auto torn=got[1];
                torn[1]+=1.0f;
                ref::require(std::memcmp(want[1].data(),torn.data(),torn.size()*4)!=0,"x4 negative control");
            }
            if (n!=24) continue;
            std::swap(w[0],w[3]); std::swap(w[1],w[2]);
            std::array<std::vector<float>,4> reversed;
            for (int i=0;i<4;++i) reversed[i].resize((size_t)n*w[i].rows+7,123.0f);
            ref::projection_x4(w,4,x.data(),n,reversed);
            for (int i=0;i<4;++i) {
                for (int t=0;t<n;++t)
                for (int r=0;r<w[i].rows;++r) {
                    const float value=ref::fold(w[i],x.data(),n,t,r);
                    ref::require(std::memcmp(&value,&reversed[i][(size_t)t*w[i].rows+r],4)==0,"reversed x4 member");
                }
                for (size_t j=(size_t)n*w[i].rows;j<reversed[i].size();++j)
                    ref::require(reversed[i][j]==123.0f,"reversed tail");
            }
        }
    }
    ref::require(i8::x4_plan({INT_MAX,INT_MAX,INT_MAX,INT_MAX},4,2560,64).entry<0,"row-rounding overflow admitted");
    for (auto p : {i8::x4_plan({1,1,0,0},2,2559,24),i8::x4_plan({1,0,0,0},2,2560,24),
                   i8::x4_plan({1,1,0,0},2,2560,8),i8::x4_plan({1,1,0,0},2,2560,65),
                   i8::x4_plan({1,1,1,0},2,2560,24)})
        ref::require(p.entry<0,"invalid x4 shape admitted");
}
}  // namespace
int main() {
    try {
        ref::require(std::fesetround(FE_TONEAREST)==0,"RNE unavailable");
        fusion();
        std::puts("PASS hip_projection_parity: CPU x4 member bits, mappings and canaries; GPU untested");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr,"FAIL hip_projection_parity: %s\n",e.what()); return 1;
    }
}
