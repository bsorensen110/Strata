// src/core/dflash2_q4_parity.cpp - the drafter's Q4_0 GEMM (kernels/dflash2.hpp df2_gemm with Df2Mat::q4, after df2_q4_tile) against a CPU reference.
//
//     dflash2_q4_parity
//
// Random float matrices are quantized to ggml Q4_0 blocks on the CPU (the reference of tools/dflash2_pack.py quant_q4_0), tiled on the device, and multiplied by
// random planar-int8 activations (df2_quant_rows).  The reference reads the same blocks and the same activation codes back and sums per block in integers,
// so the only difference to the kernel is the float order of the sum over blocks.  Shapes: the drafter's (n_in 2560, 4096, 7680, 12800), one, two and three
// matrices to a launch, 1..8 columns.  Exit 0 when every output is within 2e-5 of the largest value (relative), 1 otherwise.
#include "strata/kernels/dflash2.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

uint16_t f2h(float f) {   // IEEE half, round to nearest even (the d of a block is stored this way)
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0) {
        if (e < -10) return (uint16_t) sign;
        m |= 0x800000u;
        const int sh = 14 - e;
        uint32_t r = m >> sh;
        const uint32_t rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1))) ++r;
        return (uint16_t) (sign | r);
    }
    if (e >= 31) return (uint16_t) (sign | 0x7c00u);
    uint32_t r = ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) ++r;
    return (uint16_t) (sign | r);
}

float h2f(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t x;
    if (e == 0) {
        if (m == 0) x = sign;
        else {
            int s = 0;
            uint32_t mm = m;
            while (!(mm & 0x400)) { mm <<= 1; ++s; }
            x = sign | ((uint32_t) (127 - 15 - s + 1) << 23) | ((mm & 0x3ff) << 13);
        }
    } else if (e == 31) x = sign | 0x7f800000u | (m << 13);
    else x = sign | ((e - 15 + 127) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

// ggml quantize_row_q4_0_ref: 18 bytes a block
void quant_q4_0(const float* x, int n, uint8_t* out) {
    for (int b = 0; b < n / 32; ++b) {
        float amax = 0, mx = 0;
        for (int i = 0; i < 32; ++i)
            if (std::fabs(x[b * 32 + i]) > amax) { amax = std::fabs(x[b * 32 + i]); mx = x[b * 32 + i]; }
        const float d = mx / -8.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        const uint16_t dh = f2h(d);
        out[b * 18] = (uint8_t) (dh & 0xff);
        out[b * 18 + 1] = (uint8_t) (dh >> 8);
        for (int i = 0; i < 16; ++i) {
            const int q0 = std::min(15, (int) (x[b * 32 + i] * id + 8.5f)), q1 = std::min(15, (int) (x[b * 32 + 16 + i] * id + 8.5f));
            out[b * 18 + 2 + i] = (uint8_t) (q0 | (q1 << 4));
        }
    }
}

struct Case { int n_in; std::vector<int> n_out; };

}  // namespace

int main() {
    std::mt19937 rng(20261031);
    std::normal_distribution<float> nd(0.0f, 0.1f);
    const Case cases[] = {{2560, {64}}, {2560, {64, 32}}, {2560, {4096, 1024, 1024}}, {4096, {2560}}, {7680, {2560}}, {12800, {2560}}, {2560, {7680, 7680}}, {2560, {16}}};
    int bad = 0, total = 0;
    for (const Case& cs : cases) {
        const int n_in = cs.n_in, nb = n_in / 32, nm = (int) cs.n_out.size();
        std::vector<std::vector<uint8_t>> blocks(nm);
        std::vector<uint8_t*> dev_w(nm);
        std::vector<float*> dev_y(nm);
        for (int m = 0; m < nm; ++m) {
            const int n_out = cs.n_out[m];
            std::vector<float> w((size_t) n_out * n_in);
            for (float& v : w) v = nd(rng);
            blocks[m].resize((size_t) n_out * nb * 18);
            for (int r = 0; r < n_out; ++r) quant_q4_0(&w[(size_t) r * n_in], n_in, &blocks[m][(size_t) r * nb * 18]);
            uint8_t* tmp;
            cudaMalloc((void**) &tmp, blocks[m].size());
            cudaMemcpy(tmp, blocks[m].data(), blocks[m].size(), cudaMemcpyHostToDevice);
            cudaMalloc((void**) &dev_w[m], blocks[m].size());
            strata::kernels::df2_q4_tile(tmp, dev_w[m], dev_w[m] + (size_t) n_out * n_in / 2, n_in, (int64_t) n_out * nb, nullptr);
            cudaDeviceSynchronize();
            cudaFree(tmp);
            cudaMalloc((void**) &dev_y[m], (size_t) 32 * n_out * 4);
        }
        std::vector<float> x((size_t) 32 * n_in);
        for (float& v : x) v = nd(rng) * 4.0f;
        float* dx;
        int8_t* dxq;
        float* dxd;
        cudaMalloc((void**) &dx, x.size() * 4);
        cudaMalloc((void**) &dxq, (size_t) 32 * n_in + 4096);
        cudaMalloc((void**) &dxd, (size_t) 32 * nb * 4 + 64);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        strata::kernels::df2_quant_rows(dx, n_in, 32, dxq, dxd, nullptr);
        cudaDeviceSynchronize();
        std::vector<int8_t> xq((size_t) 32 * n_in);
        std::vector<float> xd((size_t) 32 * nb);
        cudaMemcpy(xq.data(), dxq, xq.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(xd.data(), dxd, xd.size() * 4, cudaMemcpyDeviceToHost);
        for (int ncols : {1, 2, 3, 5, 8, 12, 16, 24}) {
            strata::kernels::Df2Mat mats[3];
            for (int m = 0; m < nm; ++m) mats[m] = strata::kernels::Df2Mat{dev_w[m], dev_y[m], cs.n_out[m], true};
            strata::kernels::df2_gemm(mats, nm, dxq, dxd, n_in, ncols, nullptr);
            cudaDeviceSynchronize();
            double worst = 0, big = 0;
            for (int m = 0; m < nm; ++m) {
                const int n_out = cs.n_out[m];
                std::vector<float> y((size_t) ncols * n_out);
                cudaMemcpy(y.data(), dev_y[m], y.size() * 4, cudaMemcpyDeviceToHost);
                for (int c = 0; c < ncols; ++c)
                    for (int r = 0; r < n_out; ++r) {
                        double ref = 0;
                        for (int b = 0; b < nb; ++b) {
                            const uint8_t* blk = &blocks[m][((size_t) r * nb + b) * 18];
                            const uint16_t dh = (uint16_t) blk[0] | ((uint16_t) blk[1] << 8);
                            long isum = 0;
                            for (int i = 0; i < 16; ++i) {
                                isum += (long) ((blk[2 + i] & 15) - 8) * xq[(size_t) c * n_in + b * 32 + i];
                                isum += (long) ((blk[2 + i] >> 4) - 8) * xq[(size_t) c * n_in + b * 32 + 16 + i];
                            }
                            ref += (double) h2f(dh) * (double) xd[(size_t) c * nb + b] * (double) isum;
                        }
                        const double got = y[(size_t) c * n_out + r];
                        worst = std::max(worst, std::fabs(got - ref));
                        big = std::max(big, std::fabs(ref));
                    }
            }
            const bool ok = worst <= 2e-5 * std::max(big, 1e-9);
            ++total;
            if (!ok) ++bad;
            std::printf("n_in %5d mats %d rows %5d cols %d: max |kernel - cpu| %.3g of %.3g (%.2g)  %s\n", n_in, nm, cs.n_out[0], ncols, worst, big, worst / std::max(big, 1e-9), ok ? "ok" : "MISMATCH");
        }
        for (int m = 0; m < nm; ++m) { cudaFree(dev_w[m]); cudaFree(dev_y[m]); }
        cudaFree(dx); cudaFree(dxq); cudaFree(dxd);
    }
    // 2. Q8_0 df2_gemm (NCOLS-specialized gemm8_kernel) + df2_topk + df2_pack_head_rows + df2_select_multi parity against per-slot df2_select.
    {
        const int n_in = 2560, nb = n_in / 32, n_out = 64;
        std::vector<uint8_t> q8_blocks((size_t) n_out * nb * 34);
        std::vector<int8_t> wq((size_t) n_out * n_in);
        std::vector<uint16_t> wd((size_t) n_out * nb);
        for (int r = 0; r < n_out; ++r)
            for (int b = 0; b < nb; ++b) {
                const uint16_t dh = f2h(std::fabs(nd(rng)) + 0.01f);
                wd[(size_t) r * nb + b] = dh;
                uint8_t* blk = &q8_blocks[((size_t) r * nb + b) * 34];
                blk[0] = (uint8_t) (dh & 0xff);
                blk[1] = (uint8_t) (dh >> 8);
                for (int i = 0; i < 32; ++i) {
                    const int8_t qv = (int8_t) std::clamp((int) std::lrint(nd(rng) * 40.0f), -127, 127);
                    wq[(size_t) r * n_in + b * 32 + i] = qv;
                    blk[2 + i] = (uint8_t) qv;
                }
            }
        uint8_t* raw_q8;
        uint8_t* tiled_q8;
        cudaMalloc((void**) &raw_q8, q8_blocks.size());
        cudaMalloc((void**) &tiled_q8, q8_blocks.size());
        cudaMemcpy(raw_q8, q8_blocks.data(), q8_blocks.size(), cudaMemcpyHostToDevice);
        strata::kernels::df2_q8_tile(raw_q8, tiled_q8, tiled_q8 + wq.size(), n_in, (int64_t) n_out * nb, nullptr);
        cudaFree(raw_q8);
        std::vector<float> x((size_t) 32 * n_in);
        for (float& v : x) v = nd(rng) * 4.0f;
        float *dx, *dxd, *dy;
        int8_t* dxq;
        cudaMalloc((void**) &dx, x.size() * 4);
        cudaMalloc((void**) &dxq, (size_t) 32 * n_in + 4096);
        cudaMalloc((void**) &dxd, (size_t) 32 * nb * 4 + 64);
        cudaMalloc((void**) &dy, (size_t) 32 * n_out * 4);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        strata::kernels::df2_quant_rows(dx, n_in, 32, dxq, dxd, nullptr);
        cudaDeviceSynchronize();
        std::vector<int8_t> xq((size_t) 32 * n_in);
        std::vector<float> xd((size_t) 32 * nb);
        cudaMemcpy(xq.data(), dxq, xq.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(xd.data(), dxd, xd.size() * 4, cudaMemcpyDeviceToHost);
        for (int ncols : {1, 2, 3, 5, 8, 12, 16, 24}) {
            strata::kernels::Df2Mat mat{tiled_q8, dy, n_out, false};
            strata::kernels::df2_gemm(&mat, 1, dxq, dxd, n_in, ncols, nullptr);
            cudaDeviceSynchronize();
            std::vector<float> y((size_t) ncols * n_out);
            cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost);
            double worst = 0, big = 0;
            for (int c = 0; c < ncols; ++c)
                for (int r = 0; r < n_out; ++r) {
                    double ref = 0;
                    for (int b = 0; b < nb; ++b) {
                        long isum = 0;
                        for (int i = 0; i < 32; ++i)
                            isum += (long) wq[(size_t) r * n_in + b * 32 + i] * xq[(size_t) c * n_in + b * 32 + i];
                        ref += (double) h2f(wd[(size_t) r * nb + b]) * (double) xd[(size_t) c * nb + b] * (double) isum;
                    }
                    const double got = y[(size_t) c * n_out + r];
                    worst = std::max(worst, std::fabs(got - ref));
                    big = std::max(big, std::fabs(ref));
                }
            const bool ok = worst <= 2e-5 * std::max(big, 1e-9);
            ++total;
            if (!ok) ++bad;
            std::printf("Q8_0 n_in %5d cols %d: max |kernel - cpu| %.3g of %.3g  %s\n", n_in, ncols, worst, big, ok ? "ok" : "MISMATCH");
        }
        cudaFree(tiled_q8); cudaFree(dx); cudaFree(dxq); cudaFree(dxd); cudaFree(dy);

        // 3. df2_topk + df2_pack_head_rows + df2_select_multi vs per-slot df2_select
        const int n_vocab = 4096, rank = 256, n_slots = 4;
        const int kw[4] = {4, 7, 2, 5};
        const int prev0[4] = {11, 22, 33, 44};
        int total_rows = 0;
        for (int s = 0; s < n_slots; ++s) total_rows += kw[s];
        std::vector<float> h_logits((size_t) total_rows * n_vocab);
        for (float& v : h_logits) v = nd(rng) * 5.0f;
        std::vector<uint16_t> h_pcb((size_t) n_vocab * rank), h_scb((size_t) n_vocab * rank);
        for (uint16_t& u : h_pcb) u = f2h(nd(rng));
        for (uint16_t& u : h_scb) u = f2h(nd(rng));
        std::vector<int32_t> h_dv(n_vocab);
        for (int i = 0; i < n_vocab; ++i) h_dv[i] = (i * 7) % n_vocab;
        float *d_logits, *d_tv;
        int32_t *d_ti, *d_dv;
        uint32_t* d_scratch;
        uint16_t *d_pcb, *d_scb;
        cudaMalloc((void**) &d_logits, h_logits.size() * 4);
        cudaMalloc((void**) &d_tv, (size_t) total_rows * 16 * 4);
        cudaMalloc((void**) &d_ti, (size_t) total_rows * 16 * 4);
        cudaMalloc((void**) &d_scratch, (size_t) total_rows * strata::kernels::kDf2TopBlocks * 16 * 2 * 4);
        cudaMalloc((void**) &d_pcb, h_pcb.size() * 2);
        cudaMalloc((void**) &d_scb, h_scb.size() * 2);
        cudaMalloc((void**) &d_dv, h_dv.size() * 4);
        cudaMemcpy(d_logits, h_logits.data(), h_logits.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_pcb, h_pcb.data(), h_pcb.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_scb, h_scb.data(), h_scb.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_dv, h_dv.data(), h_dv.size() * 4, cudaMemcpyHostToDevice);
        strata::kernels::df2_topk(d_logits, n_vocab, d_dv, total_rows, d_scratch, d_tv, d_ti, nullptr);

        // Slot fn and hsel buffers, packed via df2_pack_head_rows
        float *slot_fn[4], *slot_hsel[4], *fn_batch, *hsel_batch;
        cudaMalloc((void**) &fn_batch, (size_t) total_rows * n_in * 4);
        cudaMalloc((void**) &hsel_batch, (size_t) total_rows * rank * 4);
        int off = 0;
        std::vector<int32_t> bmeta(n_slots * 4, 0);
        for (int s = 0; s < n_slots; ++s) {
            cudaMalloc((void**) &slot_fn[s], (size_t) 8 * n_in * 4);
            cudaMalloc((void**) &slot_hsel[s], (size_t) 8 * rank * 4);
            std::vector<float> h_fn((size_t) 8 * n_in), h_hs((size_t) 8 * rank);
            for (float& v : h_fn) v = nd(rng);
            for (float& v : h_hs) v = nd(rng);
            cudaMemcpy(slot_fn[s], h_fn.data(), h_fn.size() * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(slot_hsel[s], h_hs.data(), h_hs.size() * 4, cudaMemcpyHostToDevice);
            strata::kernels::df2_pack_head_rows(slot_fn[s] + n_in, slot_hsel[s], fn_batch + (size_t) off * n_in,
                                              hsel_batch + (size_t) off * rank, kw[s], n_in, rank, nullptr);
            bmeta[s * 4 + 0] = off;
            bmeta[s * 4 + 1] = kw[s];
            bmeta[s * 4 + 2] = prev0[s];
            off += kw[s];
        }
        int32_t *d_bmeta, *d_btok, *d_stok, *d_anchor;
        float *d_bprob, *d_sprob;
        cudaMalloc((void**) &d_bmeta, bmeta.size() * 4);
        cudaMalloc((void**) &d_btok, (size_t) n_slots * 8 * 4);
        cudaMalloc((void**) &d_bprob, (size_t) n_slots * 8 * 4);
        cudaMalloc((void**) &d_stok, 8 * 4);
        cudaMalloc((void**) &d_sprob, 8 * 4);
        cudaMalloc((void**) &d_anchor, 4);
        cudaMemcpy(d_bmeta, bmeta.data(), bmeta.size() * 4, cudaMemcpyHostToDevice);
        strata::kernels::df2_select_multi(hsel_batch, d_tv, d_ti, d_pcb, d_scb, rank, d_bmeta, n_slots, d_btok, d_bprob, nullptr, false);
        cudaDeviceSynchronize();
        std::vector<int32_t> btok(n_slots * 8);
        std::vector<float> bprob(n_slots * 8);
        cudaMemcpy(btok.data(), d_btok, btok.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(bprob.data(), d_bprob, bprob.size() * 4, cudaMemcpyDeviceToHost);
        bool sel_ok = true;
        off = 0;
        for (int s = 0; s < n_slots; ++s) {
            cudaMemcpy(d_anchor, &prev0[s], 4, cudaMemcpyHostToDevice);
            strata::kernels::df2_select(slot_hsel[s], d_tv + (size_t) off * 16, d_ti + (size_t) off * 16,
                                      d_pcb, d_scb, rank, kw[s], d_anchor, d_stok, d_sprob, nullptr, false);
            cudaDeviceSynchronize();
            int32_t stok[8] = {};
            float sprob[8] = {};
            cudaMemcpy(stok, d_stok, 8 * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(sprob, d_sprob, 8 * 4, cudaMemcpyDeviceToHost);
            for (int i = 0; i < kw[s]; ++i)
                if (stok[i] != btok[s * 8 + i] || std::memcmp(&sprob[i], &bprob[s * 8 + i], 4) != 0) sel_ok = false;
            off += kw[s];
            cudaFree(slot_fn[s]); cudaFree(slot_hsel[s]);
        }
        ++total;
        if (!sel_ok) ++bad;
        std::printf("df2_pack_head_rows + df2_select_multi (4 slots, 18 total rows): %s\n", sel_ok ? "bitwise equal" : "MISMATCH");
        cudaFree(d_logits); cudaFree(d_tv); cudaFree(d_ti); cudaFree(d_scratch);
        cudaFree(d_pcb); cudaFree(d_scb); cudaFree(d_dv); cudaFree(fn_batch); cudaFree(hsel_batch);
        cudaFree(d_bmeta); cudaFree(d_btok); cudaFree(d_bprob); cudaFree(d_stok); cudaFree(d_sprob); cudaFree(d_anchor);
    }
    std::printf("%s: %d of %d cases match the CPU reference\n", bad == 0 ? "PASS" : "FAIL", total - bad, total);
    return bad == 0 ? 0 : 1;
}
