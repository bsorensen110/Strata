// src/kernels/native_wmma_gfx12_test.cpp - tests gfx12 wave32 INT8 WMMA lane-mapping oracle
// and compares tiled Q8_0 / Q6_K CPU WMMA oracle against GPU native_mmvq for ncols = 9..16.
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_wmma_gfx12.hpp"
#include "strata/kernels/window_limits.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 256), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "memset");
    return p;
}

bool test_16x16x16_oracle() {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-127, 127);
    int8_t A[16][16]{}, B[16][16]{};
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c) {
            A[r][c] = int8_t(dist(rng));
            B[r][c] = int8_t(dist(rng));
        }

    int8_t a_lane[32][8]{}, b_lane[32][8]{};
    int32_t d_lane[32][8]{};
    for (int l = 0; l < 32; ++l) {
        const int l16 = l & 15;
        const int hi = l >> 4;
        for (int j = 0; j < 8; ++j) {
            a_lane[l][j] = A[l16][8 * hi + j];
            b_lane[l][j] = B[8 * hi + j][l16];
        }
    }
    k::native_wmma_gfx12_emulate_16x16x16_i8(a_lane, b_lane, d_lane);

    for (int l = 0; l < 32; ++l) {
        const int l16 = l & 15;
        const int hi = l >> 4;
        for (int i = 0; i < 8; ++i) {
            const int r = 8 * hi + i;
            const int c = l16;
            int32_t ref = 0;
            for (int kk = 0; kk < 16; ++kk) ref += int32_t(A[r][kk]) * int32_t(B[kk][c]);
            if (d_lane[l][i] != ref) {
                std::fprintf(stderr, "16x16x16 oracle mismatch at (%d,%d): %d != %d\n", r, c, d_lane[l][i], ref);
                return false;
            }
        }
    }
    std::printf("  emulate_16x16x16_i8: exact\n");
    return true;
}

bool test_tiled_type(int type, const char* name, int n_in, int n_out, cudaStream_t s) {
    std::mt19937 rng(120 + type * 13 + n_out);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> byte(0, 255), ex(2, 8), man(0, 1023), sgn(0, 3);
    const size_t row_bytes = k::native_mmvq_weight_bytes(type, n_in, 1);
    const size_t bs = k::native_mmvq_weight_bytes(type, type == 8 ? 32 : 256, 1);
    std::vector<uint8_t> w((size_t) n_out * row_bytes);
    for (auto& b : w) b = uint8_t(byte(rng));
    for (size_t o = 0; o < w.size(); o += bs) {
        const uint16_t h = uint16_t((sgn(rng) == 0 ? 0x8000 : 0) | (ex(rng) << 10) | man(rng));
        std::memcpy(&w[o + (type == 14 ? 208 : 0)], &h, 2);
    }

    const int max_c = k::kWindowMaxT;
    std::vector<float> x((size_t) max_c * n_in);
    for (auto& v : x) v = nd(rng);

    uint8_t* dw = dalloc<uint8_t>(w.size());
    float* dx = dalloc<float>(x.size());
    const size_t xq_bytes = k::native_q8_1_bytes(n_in, max_c);
    uint8_t* dxq = dalloc<uint8_t>(xq_bytes);
    float* dy = dalloc<float>((size_t) max_c * n_out);
    ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
    ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    k::native_quantize_q8_1(dx, dxq, n_in, max_c, s);
    ck(cudaStreamSynchronize(s), "q8_1");
    std::vector<uint8_t> xq(xq_bytes);
    ck(cudaMemcpy(xq.data(), dxq, xq_bytes, cudaMemcpyDeviceToHost), "xq");

    bool ok = true;
    for (int nc : {4, 8, 9, 12, 16}) {
        if (nc > k::kWindowMaxT) continue;
        k::native_mmvq(type, dw, dxq, dy, n_in, n_out, nc, s);
        ck(cudaStreamSynchronize(s), "mmvq");
        std::vector<float> y_gpu((size_t) nc * n_out), y_wmma((size_t) nc * n_out);
        ck(cudaMemcpy(y_gpu.data(), dy, y_gpu.size() * 4, cudaMemcpyDeviceToHost), "y_gpu");
        if (type == 8) k::native_wmma_gfx12_q8_0_cpu_ref_tiled(w.data(), xq.data(), y_wmma.data(), n_in, n_out, nc);
        else k::native_wmma_gfx12_q6_k_cpu_ref_tiled(w.data(), xq.data(), y_wmma.data(), n_in, n_out, nc);

        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < y_gpu.size(); ++i) {
            if (!std::isfinite(y_gpu[i]) || !std::isfinite(y_wmma[i])) ok = false;
            num += std::fabs((double) y_gpu[i] - (double) y_wmma[i]);
            den += std::fabs((double) y_gpu[i]);
        }
        const double rel = num / (den + 1e-30);
        if (!(rel < 1e-4)) ok = false;
        std::printf("  %-5s %4d x %4d nc=%2d cpu-wmma vs mmvq rel=%.2e\n", name, n_out, n_in, nc, rel);

        // Verify direct GPU call returns false cleanly when non-gfx12, or matches when gfx12.
        std::vector<float> y_dev((size_t) nc * n_out);
        const bool launched = (type == 8)
            ? k::native_wmma_dense_q8_0(dw, dxq, dy, n_in, n_out, nc, s)
            : k::native_wmma_dense_q6_k(dw, dxq, dy, n_in, n_out, nc, s);
        if (launched != k::native_wmma_gfx12_available()) {
            std::fprintf(stderr, "unexpected wmma launch status %d vs available %d\n",
                         (int) launched, (int) k::native_wmma_gfx12_available());
            ok = false;
        }
        if (launched) {
            ck(cudaStreamSynchronize(s), "wmma");
            ck(cudaMemcpy(y_dev.data(), dy, y_dev.size() * 4, cudaMemcpyDeviceToHost), "y_dev");
            double wnum = 0.0, wden = 0.0;
            for (size_t i = 0; i < y_dev.size(); ++i) {
                wnum += std::fabs((double) y_dev[i] - (double) y_wmma[i]);
                wden += std::fabs((double) y_wmma[i]);
            }
            if (!(wnum / (wden + 1e-30) < 1e-5)) ok = false;
        }
    }

    cudaFree(dw); cudaFree(dx); cudaFree(dxq); cudaFree(dy);
    return ok;
}

bool test_q8_0_segs(int n_in, cudaStream_t s) {
    const int n_out0 = 128, n_out1 = 192;
    const size_t rb = k::native_mmvq_weight_bytes(8, n_in, 1);
    const size_t bs = k::native_mmvq_weight_bytes(8, 32, 1);
    std::mt19937 rng(777 + n_in);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> byte(0, 255), ex(2, 8), man(0, 1023), sgn(0, 3);
    auto make_w = [&](int n_out) {
        std::vector<uint8_t> w((size_t) n_out * rb);
        for (auto& b : w) b = uint8_t(byte(rng));
        for (size_t o = 0; o < w.size(); o += bs) {
            const uint16_t h = uint16_t((sgn(rng) == 0 ? 0x8000 : 0) | (ex(rng) << 10) | man(rng));
            std::memcpy(&w[o], &h, 2);
        }
        return w;
    };
    const std::vector<uint8_t> w0 = make_w(n_out0), w1 = make_w(n_out1);
    const int max_c = k::kWindowMaxT;
    std::vector<float> x((size_t) max_c * n_in);
    for (auto& v : x) v = nd(rng);

    uint8_t* dw0 = dalloc<uint8_t>(w0.size());
    uint8_t* dw1 = dalloc<uint8_t>(w1.size());
    float* dx = dalloc<float>(x.size());
    uint8_t* dxq = dalloc<uint8_t>(k::native_q8_1_bytes(n_in, max_c));
    float* dy0 = dalloc<float>((size_t) max_c * n_out0);
    float* dy1 = dalloc<float>((size_t) max_c * n_out1);
    float* r0 = dalloc<float>((size_t) max_c * n_out0);
    float* r1 = dalloc<float>((size_t) max_c * n_out1);
    ck(cudaMemcpy(dw0, w0.data(), w0.size(), cudaMemcpyHostToDevice), "w0");
    ck(cudaMemcpy(dw1, w1.data(), w1.size(), cudaMemcpyHostToDevice), "w1");
    ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "xs");
    k::native_quantize_q8_1(dx, dxq, n_in, max_c, s);
    ck(cudaStreamSynchronize(s), "q8_1_segs");

    bool ok = true;
    for (int nc : {4, 8, 9, 16}) {
        if (nc > k::kWindowMaxT) continue;
        const k::NativeMmvqSeg segs_wmma[2] = {{dw0, dy0, n_out0}, {dw1, dy1, n_out1}};
        const k::NativeMmvqSeg segs_ref[2] = {{dw0, r0, n_out0}, {dw1, r1, n_out1}};
        const bool launched = k::native_wmma_dense_q8_0_segs(segs_wmma, 2, dxq, n_in, nc, s);
        if (launched != k::native_wmma_gfx12_available()) {
            std::fprintf(stderr, "unexpected wmma segs launch status %d vs available %d\n",
                         (int) launched, (int) k::native_wmma_gfx12_available());
            ok = false;
        }
        if (launched) {
            if (!k::native_mmvq_multi_segs(8, segs_ref, 2, dxq, nullptr, n_in, nc, s, false)) {
                std::fprintf(stderr, "native_mmvq_multi_segs ref failed at nc=%d\n", nc);
                ok = false;
            }
            ck(cudaStreamSynchronize(s), "wmma_segs");
            auto check_seg = [&](const float* dg, const float* dr, int n_out) {
                std::vector<float> hg((size_t) nc * n_out), hr((size_t) nc * n_out);
                ck(cudaMemcpy(hg.data(), dg, hg.size() * 4, cudaMemcpyDeviceToHost), "hg");
                ck(cudaMemcpy(hr.data(), dr, hr.size() * 4, cudaMemcpyDeviceToHost), "hr");
                double num = 0.0, den = 0.0;
                for (size_t i = 0; i < hg.size(); ++i) {
                    if (!std::isfinite(hg[i]) || !std::isfinite(hr[i])) ok = false;
                    num += std::fabs((double) hg[i] - (double) hr[i]);
                    den += std::fabs((double) hr[i]);
                }
                const double rel = num / (den + 1e-30);
                if (!(rel < 1e-4)) ok = false;
                std::printf("  Q8_0 seg(%d) x %d nc=%2d wmma vs mmvq_segs rel=%.2e\n", n_out, n_in, nc, rel);
            };
            check_seg(dy0, r0, n_out0);
            check_seg(dy1, r1, n_out1);
        }
    }

    cudaFree(dw0); cudaFree(dw1); cudaFree(dx); cudaFree(dxq);
    cudaFree(dy0); cudaFree(dy1); cudaFree(r0); cudaFree(r1);
    return ok;
}

}  // namespace

int main() {
    std::printf("native_wmma_gfx12_test (available=%d)\n", (int) k::native_wmma_gfx12_available());
    if (!test_16x16x16_oracle()) return 1;

    cudaStream_t s = nullptr;
    ck(cudaStreamCreate(&s), "stream");

    bool ok = true;
    ok = test_tiled_type(8, "Q8_0", 512, 133, s) && ok;
    ok = test_tiled_type(14, "Q6_K", 512, 133, s) && ok;
    // native_mmvq_multi_segs requires n_in / 32 >= SmallTraits<Q80>::BPI (32); 512 only has 16 blocks.
    ok = test_q8_0_segs(1024, s) && ok;

    cudaStreamDestroy(s);
    std::printf("native_wmma_gfx12_test: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
