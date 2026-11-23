// src/core/dflash2_wmma_gfx12_parity.cpp - gfx12 FP16 WMMA DFlash2 attention vs the production FP16-ring path, on fixed random data.
//
// Run each arm as its own process (the gate reads the environment once per process), then compare the dumps:
//   STRATA_DF2_ATTN_F16_SMEM=1 dflash2_wmma_gfx12_parity ctl.bin  [anchor_offset=2047] [iters=50]   # production dispatch
//   STRATA_DFLASH_WMMA=1       dflash2_wmma_gfx12_parity cand.bin [anchor_offset=2047] [iters=50]   # gfx12 WMMA branch (opt-in)
//   python3 dflash2_wmma_gfx12_parity_compare.py ctl.bin cand.bin
//
// Not in CMake yet. It needs a free R9700: production owns ~31 GiB of VRAM, so run only inside an authorized GPU window.
// Dump layout: u64 n_part, n_xq, n_xd; then float[n_part] partials, uint8[n_xq] merged int8 activations, float[n_xd] scales.
#include "strata/kernels/dflash2.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace strata::kernels;

static void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(3);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s out.bin [anchor_offset=2047] [iters=50]\n", argv[0]);
        return 2;
    }
    const int anchor_offset = argc > 2 ? std::atoi(argv[2]) : 2047;
    const int iters = argc > 3 ? std::atoi(argv[3]) : 50;
    const int ring = 2064;   // ring slots, as dflash2_attn_bench uses them
    const int window = 2048;
    const size_t n_q = (size_t) kDf2Rows * 32 * kDf2Hd;        // [row][head][HD]
    const size_t n_ring = (size_t) ring * kDf2Kv * kDf2Hd;     // [kvh][slot][HD]
    const size_t n_blk = (size_t) kDf2Rows * kDf2Kv * kDf2Hd;  // [row][kvh][HD]
    const size_t n_part = (size_t) kDf2MaxChunks * kDf2Kv * 32 * 130;
    const size_t n_xq = (size_t) kDf2Kv * 32 * kDf2Hd;
    const size_t n_xd = (size_t) kDf2Kv * kDf2Hd;

    // Deterministic inputs: xorshift64, so both arms see identical bits.
    uint64_t st = 0x9E3779B97F4A7C15ull;
    auto rnd = [&]() {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        return (float) ((st >> 11) & 0xFFFFFF) / 16777216.0f * 2.0f - 1.0f;
    };
    std::vector<float> hq(n_q), hkb(n_blk), hvb(n_blk);
    std::vector<__half> hkr(n_ring), hvr(n_ring);
    for (auto& x : hq) x = rnd() * 2.0f;
    for (auto& x : hkb) x = rnd();
    for (auto& x : hvb) x = rnd();
    for (auto& x : hkr) x = __float2half(rnd());
    for (auto& x : hvr) x = __float2half(rnd());

    float *q, *kb, *vb, *part, *xd;
    int8_t* xq;
    __half *kr, *vr;
    int32_t* pos;
    check_cuda(cudaMalloc((void**) &q, n_q * 4), "malloc q");
    check_cuda(cudaMalloc((void**) &kb, n_blk * 4), "malloc kb");
    check_cuda(cudaMalloc((void**) &vb, n_blk * 4), "malloc vb");
    check_cuda(cudaMalloc((void**) &kr, n_ring * 2), "malloc kr");
    check_cuda(cudaMalloc((void**) &vr, n_ring * 2), "malloc vr");
    check_cuda(cudaMalloc((void**) &part, n_part * 4), "malloc part");
    check_cuda(cudaMalloc((void**) &xq, n_xq), "malloc xq");
    check_cuda(cudaMalloc((void**) &xd, n_xd * 4), "malloc xd");
    check_cuda(cudaMalloc((void**) &pos, 4), "malloc pos");
    check_cuda(cudaMemcpy(q, hq.data(), n_q * 4, cudaMemcpyHostToDevice), "copy q");
    check_cuda(cudaMemcpy(kb, hkb.data(), n_blk * 4, cudaMemcpyHostToDevice), "copy kb");
    check_cuda(cudaMemcpy(vb, hvb.data(), n_blk * 4, cudaMemcpyHostToDevice), "copy vb");
    check_cuda(cudaMemcpy(kr, hkr.data(), n_ring * 2, cudaMemcpyHostToDevice), "copy kr");
    check_cuda(cudaMemcpy(vr, hvr.data(), n_ring * 2, cudaMemcpyHostToDevice), "copy vr");
    const int32_t s = 5000 + anchor_offset;
    check_cuda(cudaMemcpy(pos, &s, 4, cudaMemcpyHostToDevice), "copy pos");
    check_cuda(cudaMemset(part, 0, n_part * 4), "memset part");
    check_cuda(cudaMemset(xq, 0, n_xq), "memset xq");

    auto run = [&]() {
        df2_attention_f16(q, kr, vr, kb, vb, pos, ring, window, part, xq, xd, nullptr, nullptr, nullptr);
    };
    run();
    check_cuda(cudaDeviceSynchronize(), "warmup");

    cudaEvent_t a, b;
    check_cuda(cudaEventCreate(&a), "event");
    check_cuda(cudaEventCreate(&b), "event");
    check_cuda(cudaEventRecord(a), "record");
    for (int i = 0; i < iters; ++i) run();
    check_cuda(cudaEventRecord(b), "record");
    check_cuda(cudaEventSynchronize(b), "sync");
    float ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&ms, a, b), "elapsed");

    std::vector<float> hpart(n_part), hxd(n_xd);
    std::vector<int8_t> hxq(n_xq);
    check_cuda(cudaMemcpy(hpart.data(), part, n_part * 4, cudaMemcpyDeviceToHost), "dump part");
    check_cuda(cudaMemcpy(hxq.data(), xq, n_xq, cudaMemcpyDeviceToHost), "dump xq");
    check_cuda(cudaMemcpy(hxd.data(), xd, n_xd * 4, cudaMemcpyDeviceToHost), "dump xd");

    FILE* f = std::fopen(argv[1], "wb");
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 4;
    }
    const uint64_t hdr[3] = {n_part, n_xq, n_xd};
    std::fwrite(hdr, sizeof(hdr), 1, f);
    std::fwrite(hpart.data(), 4, n_part, f);
    std::fwrite(hxq.data(), 1, n_xq, f);
    std::fwrite(hxd.data(), 4, n_xd, f);
    std::fclose(f);

    std::printf("df2_attention_f16: %.1f us per call (anchor=%d, window=%d, iters=%d) F16_SMEM=%s DFLASH_WMMA=%s -> %s\n",
                1000.0 * ms / iters, s, window, iters, std::getenv("STRATA_DF2_ATTN_F16_SMEM") ? std::getenv("STRATA_DF2_ATTN_F16_SMEM") : "unset",
                std::getenv("STRATA_DFLASH_WMMA") ? std::getenv("STRATA_DFLASH_WMMA") : "unset", argv[1]);
    return 0;
}
