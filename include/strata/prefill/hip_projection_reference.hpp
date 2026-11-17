// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// Port note: hipfire gemm_mq6sym_i8.gfx1201.hip shared-body x4 oracle.
// Modified for Strata planar packs. Test-only; no engine dependency.
#pragma once
#if defined(STRATA_MOE_IU8_CPU_ORACLE)
#include "strata/prefill/hip_projection.hpp"
#include "strata/prefill/moe_iu8_reference.hpp"

namespace strata::prefill::iu8::reference {
inline Pack projection_pack(int m, int k, int bits, uint32_t seed) {
    Pack p;
    p.rows=m; p.cols=k; p.bits=bits; p.symmetric_verified=true;
    p.codes.resize((size_t)m*k*bits/8);
    p.scales.resize((size_t)m*k/128);
    std::mt19937 rng(seed);
    for (auto& b : p.codes) b=(uint8_t)rng();
    for (size_t i=0;i<p.scales.size();++i)
        p.scales[i]=(uint16_t)((i%3==0?0xa000:0x2400)+(i%29)*13);
    return p;
}

// Each CTA owns only one matrix. Reconstruct the flattened prefix mapping,
// token slices and WMMA lane halves, independently of the scalar fold oracle.
inline void projection_x4(const std::array<Pack,4>& w, int count, const Block* x, int n,
                          std::array<std::vector<float>,4>& out) {
    std::array<int,4> rows{};
    for (int i=0;i<count;++i) rows[i]=w[i].rows;
    const auto plan=x4_plan(rows,count,w[0].cols,n);
    require(plan.entry>=0,"x4 oracle plan refused");
    const auto tile=kNarrowTiles[plan.entry];
    for (uint32_t by=0;by<plan.y;++by)
    for (uint32_t bx=0;bx<plan.x;++bx) {
        int part=0,local=(int)bx;
        while (part<count && local>=(w[part].rows+tile.rows-1)/tile.rows) {
            local-=(w[part].rows+tile.rows-1)/tile.rows; ++part;
        }
        require(part<count,"x4 oracle prefix escaped");
        const auto& p=w[part];
        for (int wave=0;wave<tile.rows/16;++wave)
        for (int lane=0;lane<32;++lane)
        for (int tg=0;tg<tile.tokens/16;++tg)
        for (int j=0;j<8;++j) {
            const int row=local*tile.rows+wave*16+(lane&15);
            const int token=(int)by*tile.tokens+tg*16+8*(lane>>4)+j;
            if (row>=p.rows || token>=n) continue;
            float acc=0.0f;
            for (int kb=0;kb<p.cols/128;++kb) {
                const auto& b=x[kb*n+token];
                int32_t c=0;
                for (int step=0;step<8;++step)
                for (int h=0;h<2;++h)
                for (int q=0;q<8;++q) {
                    const int kk=64*h+8*step+q;
                    c+=signed_code(p,(size_t)row*p.cols+kb*128+kk)*b.qs[byte_index(kk)];
                }
                acc=std::fma(mul(half_value(p.scales[(size_t)row*(p.cols/128)+kb]),b.d),(float)c,acc);
            }
            out[part][(size_t)token*p.rows+row]=acc;
        }
    }
}
}  // namespace strata::prefill::iu8::reference
#endif
