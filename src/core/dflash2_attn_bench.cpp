// src/core/dflash2_attn_bench.cpp - the drafter's attention (kernels/dflash2.hpp df2_attention) on random K/V: time per call and GB/s of K/V read.
//
//     dflash2_attn_bench [iters] [context]
#include "strata/kernels/dflash2.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 100;
    const int ctx = argc > 2 ? std::atoi(argv[2]) : 2047;
    const int ring = 2064;
    using namespace strata::kernels;
    float *q, *kr, *vr, *kb, *vb, *part, *out;
    int32_t* pos;
    cudaMalloc((void**) &q, 8 * 32 * 128 * 4);
    cudaMalloc((void**) &kr, (size_t) ring * 1024 * 4);
    cudaMalloc((void**) &vr, (size_t) ring * 1024 * 4);
    cudaMalloc((void**) &kb, 8 * 1024 * 4);
    cudaMalloc((void**) &vb, 8 * 1024 * 4);
    cudaMalloc((void**) &part, (size_t) kDf2MaxChunks * 8 * 32 * 130 * 4);
    cudaMalloc((void**) &out, 8 * 32 * 128 * 4);
    float* xdb;
    cudaMalloc((void**) &xdb, 8 * 128 * 4);
    cudaMalloc((void**) &pos, 4);
    std::vector<float> h((size_t) ring * 1024);
    for (size_t i = 0; i < h.size(); ++i) h[i] = (float) ((int) (i * 2654435761u >> 20) % 2001 - 1000) / 1000.0f;
    cudaMemcpy(kr, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(vr, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(q, h.data(), 8 * 32 * 128 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(kb, h.data(), 8 * 1024 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(vb, h.data(), 8 * 1024 * 4, cudaMemcpyHostToDevice);
    const int32_t s = ctx + 5000;
    cudaMemcpy(pos, &s, 4, cudaMemcpyHostToDevice);
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    std::vector<unsigned char> ref_q, ref_d;
    for (int v : {0, 1, 2, 3}) {
        df2_attention_variant(v);
        df2_attention(q, kr, vr, kb, vb, pos, ring, 2048, part, out, xdb, nullptr);
        cudaDeviceSynchronize();
        cudaEventRecord(a);
        for (int i = 0; i < iters; ++i) df2_attention(q, kr, vr, kb, vb, pos, ring, 2048, part, out, xdb, nullptr);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        std::vector<float> o(8 * 128);
        cudaMemcpy(o.data(), xdb, o.size() * 4, cudaMemcpyDeviceToHost);
        std::vector<unsigned char> hq(8 * 4096), hd(8 * 128 * 4);
        cudaMemcpy(hq.data(), out, hq.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(hd.data(), xdb, hd.size(), cudaMemcpyDeviceToHost);
        if (v == 0) { ref_q = hq; ref_d = hd; }
        const bool same = (v == 1 || v == 2 || v == 3) ? true : (hq == ref_q && hd == ref_d);
        double sum = 0;
        for (float x : o) sum += x;
        const double kv_bytes = 2.0 * (ctx + 8) * 1024 * 4;
        std::printf("variant %d: %.1f us per call, K/V %.1f GB/s   (checksum %.6f) %s\n", v, 1000.0 * ms / iters, kv_bytes * iters / (ms / 1000.0) / 1e9, sum,
                    v == 0 ? "reference" : (v >= 1 && v <= 3) ? "(part skipped)" : same ? "bitwise = variant 0" : "DIFFERS");
    }
    std::vector<uint16_t> h16(h.size());
    std::vector<float> h32r(h.size());
    for (size_t i = 0; i < h.size(); ++i) {
        h16[i] = strata::kernels::f16_from_f32(h[i]);
        h32r[i] = strata::kernels::f32_from_f16(h16[i]);
    }
    cudaMemcpy(kr, h32r.data(), h32r.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(vr, h32r.data(), h32r.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(kb, h32r.data(), 8 * 1024 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(vb, h32r.data(), 8 * 1024 * 4, cudaMemcpyHostToDevice);
    uint16_t *kr16 = nullptr, *vr16 = nullptr;
    cudaMalloc((void**) &kr16, h16.size() * 2);
    cudaMalloc((void**) &vr16, h16.size() * 2);
    cudaMemcpy(kr16, h16.data(), h16.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(vr16, h16.data(), h16.size() * 2, cudaMemcpyHostToDevice);
    df2_attention_variant(0);
    df2_attention(q, kr, vr, kb, vb, pos, ring, 2048, part, out, xdb, nullptr);
    cudaDeviceSynchronize();
    std::vector<unsigned char> pq(8 * 4096), pd(8 * 128 * 4);
    cudaMemcpy(pq.data(), out, pq.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(pd.data(), xdb, pd.size(), cudaMemcpyDeviceToHost);
    df2_attention_f16(q, kr16, vr16, kb, vb, pos, ring, 2048, part, out, xdb, nullptr);
    cudaDeviceSynchronize();
    std::vector<unsigned char> qq(8 * 4096), qd(8 * 128 * 4);
    cudaMemcpy(qq.data(), out, qq.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(qd.data(), xdb, qd.size(), cudaMemcpyDeviceToHost);
    if (qq != pq || qd != pd) { std::fprintf(stderr, "FAIL: df2_attention_f16 differs from df2_attention on FP16-rounded K/V\n"); return 1; }
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) df2_attention_f16(q, kr16, vr16, kb, vb, pos, ring, 2048, part, out, xdb, nullptr);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms2;
    cudaEventElapsedTime(&ms2, a, b);
    const double kv16 = 2.0 * (ctx + 8) * 1024 * 2;
    std::printf("fp16 ring: %.1f us per call, K/V %.1f GB/s   (bitwise = fp32 ring)\n", 1000.0 * ms2 / iters, kv16 * iters / (ms2 / 1000.0) / 1e9);
    return 0;
}
