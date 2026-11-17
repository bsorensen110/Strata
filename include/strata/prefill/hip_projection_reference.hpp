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

inline int signed_dot4(uint32_t a, uint32_t b, int acc) {
    for (int j=0;j<4;++j) {
        int av=(a>>(8*j))&255, bv=(b>>(8*j))&255;
        if (av>=128) av-=256;
        if (bv>=128) bv-=256;
        acc+=av*bv;
    }
    return acc;
}
inline int dot4_slice(const Pack& w, const Block& x, int row, int group, int first, int length) {
    int sum=0;
    for (int j=0;j<length;j+=8) {
        const auto* code=w.codes.data()+((size_t)row*w.cols+group*256+first+j)*3/4;
        const unsigned a=code[0]|((unsigned)code[1]<<8)|((unsigned)code[2]<<16);
        const unsigned b=code[3]|((unsigned)code[4]<<8)|((unsigned)code[5]<<16);
        uint32_t xe=0,xo=0;
        for (int q=0;q<4;++q) {
            xe|=(uint32_t)(uint8_t)x.qs[(first&127)+j+q]<<(8*q);
            xo|=(uint32_t)(uint8_t)x.qs[(first&127)+j+4+q]<<(8*q);
        }
        sum=signed_dot4(dot4_word6(a,b,0),xe,sum);
        sum=signed_dot4(dot4_word6(a,b,1),xo,sum);
    }
    return sum;
}
inline void projection_dot4(const Pack& w, const Block* x, int n, float* out, int ld) {
    require(w.bits==6 && w.cols%256==0 && n>=1 && n<=8,"dot4 oracle shape");
    const int half=(n+1)/2;
    for (int row=0;row<w.rows;++row)
    for (int token=0;token<n;++token) {
        float acc=0.0f;
        for (int group=0;group<w.cols/256;++group) {
            int c[2]={};
            if (n<=2) {
                int lane[4];
                for (int q=0;q<4;++q)
                    lane[q]=dot4_slice(w,x[(group*2+(q>>1))*n+token],row,group,q*64,64);
                c[0]=(lane[0]+lane[1])/4; c[1]=(lane[2]+lane[3])/4;
            } else {
                const int tp=token/half,local=token%half;
                const int writer=2*tp+(local&1);
                require(writer>=0 && writer<4,"dot4 token owner");
                for (int kh=0;kh<2;++kh)
                    c[kh]=dot4_slice(w,x[(group*2+kh)*n+token],row,group,kh*128,128)/4;
            }
            for (int kh=0;kh<2;++kh)
                acc=std::fma(mul(half_value(w.scales[(size_t)row*(w.cols/128)+2*group+kh]),
                                 x[(2*group+kh)*n+token].d),(float)c[kh],acc);
        }
        out[(size_t)token*ld+row]=acc;
    }
}
}  // namespace strata::prefill::iu8::reference
#endif
