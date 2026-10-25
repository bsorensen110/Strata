// src/core/dflash2_attn_bench.cpp - the drafter's attention (kernels/dflash2.hpp df2_attention) on random K/V: time per call and GB/s of K/V read.
//
//     dflash2_attn_bench [iters] [window]
#include "strata/kernels/dflash2.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 100;
    const int window = argc > 2 ? std::atoi(argv[2]) : 2048;
    if (window < 1 || window > strata::kernels::kDf2MaxWindow) {
        std::fprintf(stderr, "window must be in 1..%d\n", strata::kernels::kDf2MaxWindow);
        return 2;
    }
    const int ring = window + 16;
    using namespace strata::kernels;
    const int chunks = df2_attention_chunks(window);
    const int ctx = window - 1;
    float *q, *kr, *vr, *kb, *vb, *part, *out;
    int32_t* pos;
    cudaMalloc((void**) &q, 8 * 32 * 128 * 4);
    cudaMalloc((void**) &kr, (size_t) ring * 1024 * 4);
    cudaMalloc((void**) &vr, (size_t) ring * 1024 * 4);
    cudaMalloc((void**) &kb, 8 * 1024 * 4);
    cudaMalloc((void**) &vb, 8 * 1024 * 4);
    cudaMalloc((void**) &part, (size_t) chunks * 8 * 32 * 130 * 4);
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
    for (int v : {0, 1, 2, 3}) {
        df2_attention_variant(v);
        df2_attention(q, kr, vr, kb, vb, pos, ring, window, part, out, xdb, nullptr);
        cudaDeviceSynchronize();
        cudaEventRecord(a);
        for (int i = 0; i < iters; ++i) df2_attention(q, kr, vr, kb, vb, pos, ring, window, part, out, xdb, nullptr);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        std::vector<float> o(8 * 128);
        cudaMemcpy(o.data(), xdb, o.size() * 4, cudaMemcpyDeviceToHost);
        double sum = 0;
        for (float x : o) sum += x;
        const double kv_bytes = 2.0 * (ctx + 8) * 1024 * 4;
        std::printf("window %d variant %d (fp32): %.1f us per call, K/V %.1f GB/s   (checksum %.6f)\n", window, v, 1000.0 * ms / iters, kv_bytes * iters / (ms / 1000.0) / 1e9, sum);
    }
    // FP16 ring parity + timing against the FP32 ring on identical rounded K/V
    std::vector<uint16_t> h16(h.size());
    std::vector<float> h32_rounded(h.size());
    for (size_t i = 0; i < h.size(); ++i) {
        h16[i] = strata::kernels::f16_from_f32(h[i]);
        h32_rounded[i] = strata::kernels::f32_from_f16(h16[i]);
    }
    cudaMemcpy(kr, h32_rounded.data(), h32_rounded.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(vr, h32_rounded.data(), h32_rounded.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(kb, h32_rounded.data(), 8 * 1024 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(vb, h32_rounded.data(), 8 * 1024 * 4, cudaMemcpyHostToDevice);
    uint16_t *kr16 = nullptr, *vr16 = nullptr;
    cudaMalloc((void**) &kr16, h16.size() * 2);
    cudaMalloc((void**) &vr16, h16.size() * 2);
    cudaMemcpy(kr16, h16.data(), h16.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(vr16, h16.data(), h16.size() * 2, cudaMemcpyHostToDevice);
    df2_attention_variant(0);
    df2_attention(q, kr, vr, kb, vb, pos, ring, window, part, out, xdb, nullptr);
    cudaDeviceSynchronize();
    std::vector<int8_t> xq_ref(8 * 32 * 128), xq_got(8 * 32 * 128);
    std::vector<float> xd_ref(8 * 128), xd_got(8 * 128);
    cudaMemcpy(xq_ref.data(), out, xq_ref.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(xd_ref.data(), xdb, xd_ref.size() * 4, cudaMemcpyDeviceToHost);

    df2_attention_f16(q, kr16, vr16, kb, vb, pos, ring, window, part, out, xdb, nullptr);
    cudaDeviceSynchronize();
    cudaMemcpy(xq_got.data(), out, xq_got.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(xd_got.data(), xdb, xd_got.size() * 4, cudaMemcpyDeviceToHost);
    if (xq_got != xq_ref || xd_got != xd_ref) {
        std::fprintf(stderr, "FAIL: df2_attention_f16 output differs from df2_attention on FP16-rounded K/V\n");
        return 1;
    }
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) df2_attention_f16(q, kr16, vr16, kb, vb, pos, ring, window, part, out, xdb, nullptr);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms16 = 0;
    cudaEventElapsedTime(&ms16, a, b);
    double sum16 = 0;
    for (float x : xd_got) sum16 += x;
    const double kv_bytes16 = 2.0 * (ctx + 8) * 1024 * 2;
    std::printf("window %d variant 0 (fp16): %.1f us per call, K/V %.1f GB/s   (checksum %.6f, bitwise equal to rounded fp32)\n",
                window, 1000.0 * ms16 / iters, kv_bytes16 * iters / (ms16 / 1000.0) / 1e9, sum16);
    cudaFree(kr16);
    cudaFree(vr16);
    return 0;
}
