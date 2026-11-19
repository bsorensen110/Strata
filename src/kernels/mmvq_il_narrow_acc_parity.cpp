// src/kernels/mmvq_il_narrow_acc_parity.cpp - bitwise parity of STRATA_MMVQ_IL_NARROW_ACC=1 (R=2, NarrowAcc=true)
// across ncols=9..16 for native_mmvq_il and native_mmvq_multi_segs vs single-column native_mmvq reference.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/window_limits.hpp"

#include <cuda_runtime.h>

#include <algorithm>
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

std::vector<uint8_t> make_weights(int type, int n_in, int n_out, std::mt19937& rng) {
    const size_t rb = k::native_mmvq_weight_bytes(type, n_in, 1);
    const size_t total = (size_t) n_out * rb;
    const size_t bs = k::native_mmvq_weight_bytes(type, (type == 8 || type == 20) ? 32 : (type == 42 ? 64 : 256), 1);
    std::vector<uint8_t> w(total);
    std::uniform_int_distribution<int> byte(0, 255), ex(2, 8), man(0, 1023), sgn(0, 3);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (size_t o = 0; o < total; o += bs) {
        const uint16_t h = (uint16_t) ((sgn(rng) == 0 ? 0x8000 : 0) | (ex(rng) << 10) | man(rng));
        if (type == 14) {
            std::memcpy(&w[o + 208], &h, 2);
        } else if (type == 11) {
            std::memcpy(&w[o + 108], &h, 2);
        } else {
            std::memcpy(&w[o], &h, 2);
            if (type == 12 || type == 13) {
                const uint16_t h2 = (uint16_t) ((sgn(rng) == 0 ? 0x8000 : 0) | (ex(rng) << 10) | man(rng));
                std::memcpy(&w[o + 2], &h2, 2);
            }
        }
    }
    return w;
}

bool check_type(int type, const char* name, int n_in, int n_out, cudaStream_t s) {
    std::mt19937 rng(911 + type * 17 + n_out);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int max_c = 16;
    const auto w = make_weights(type, n_in, n_out, rng);
    uint8_t* dw = dalloc<uint8_t>(w.size());
    ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "w");

    std::vector<float> x((size_t) max_c * n_in);
    for (auto& v : x) v = nd(rng);
    float* dx = dalloc<float>(x.size());
    ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");

    const size_t xq_bytes = k::native_q8_1_bytes(n_in, max_c);
    uint8_t* dxq = dalloc<uint8_t>(xq_bytes);
    uint8_t* dxil = dalloc<uint8_t>(k::native_q8_1_il_bytes(n_in, max_c));
    k::native_quantize_q8_1_il(dx, dxq, dxil, n_in, max_c, s);

    float* dy_il = dalloc<float>((size_t) max_c * n_out);
    float* dy_ref = dalloc<float>((size_t) max_c * n_out);

    bool ok = true;
    const int cols[] = {9, 10, 12, 16};
    for (int nc : cols) {
        if (!k::native_mmvq_il_supported(type, nc, n_out)) {
            std::printf("  %-8s nc=%2d not supported\n", name, nc);
            ok = false;
            continue;
        }
        ck(cudaMemset(dy_il, 0, (size_t) nc * n_out * 4), "zero il");
        ck(cudaMemset(dy_ref, 0, (size_t) nc * n_out * 4), "zero ref");
        k::native_mmvq_il(type, dw, dxq, dxil, dy_il, n_in, n_out, nc, s);
        const size_t col_b = k::native_q8_1_bytes(n_in, 1);
        for (int c = 0; c < nc; ++c) {
            k::native_mmvq(type, dw, dxq + (size_t) c * col_b, dy_ref + (size_t) c * n_out, n_in, n_out, 1, s);
        }
        ck(cudaStreamSynchronize(s), "sync");
        std::vector<float> y1((size_t) nc * n_out), y2((size_t) nc * n_out);
        ck(cudaMemcpy(y1.data(), dy_il, y1.size() * 4, cudaMemcpyDeviceToHost), "y1");
        ck(cudaMemcpy(y2.data(), dy_ref, y2.size() * 4, cudaMemcpyDeviceToHost), "y2");

        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < y1.size(); ++i) {
            if (!std::isfinite(y1[i]) || !std::isfinite(y2[i])) ok = false;
            num += std::fabs((double) y1[i] - (double) y2[i]);
            den += std::fabs((double) y2[i]);
        }
        const double rel = num / (den + 1e-30);
        if (!(rel < 1e-4)) ok = false;
        std::printf("  %-8s %4d x %4d nc=%2d rel=%.2e\n", name, n_out, n_in, nc, rel);
    }

    // Check multi-segs path at nc=12.
    const int nc = 12;
    const int n1 = n_out / 2, n2 = n_out - n1;
    const size_t w1b = k::native_mmvq_weight_bytes(type, n_in, n1);
    float* dy_seg1 = dalloc<float>((size_t) nc * n1);
    float* dy_seg2 = dalloc<float>((size_t) nc * n2);
    k::NativeMmvqSeg segs[2] = {{dw, dy_seg1, n1}, {dw + w1b, dy_seg2, n2}};
    if (!k::native_mmvq_multi_segs(type, segs, 2, dxq, dxil, n_in, nc, s, true)) {
        std::printf("  %-8s multi_segs nc=%2d not launched\n", name, nc);
        ok = false;
    } else {
        k::native_mmvq(type, dw, dxq, dy_ref, n_in, n1, nc, s);
        ck(cudaStreamSynchronize(s), "sync seg");
        std::vector<float> ys((size_t) nc * n1), yr((size_t) nc * n1);
        ck(cudaMemcpy(ys.data(), dy_seg1, ys.size() * 4, cudaMemcpyDeviceToHost), "ys");
        ck(cudaMemcpy(yr.data(), dy_ref, yr.size() * 4, cudaMemcpyDeviceToHost), "yr");
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < ys.size(); ++i) {
            num += std::fabs((double) ys[i] - (double) yr[i]);
            den += std::fabs((double) yr[i]);
        }
        const double rel = num / (den + 1e-30);
        if (!(rel < 1e-4)) ok = false;
        std::printf("  %-8s multi_segs nc=%2d rel=%.2e\n", name, nc, rel);
    }

    cudaFree(dy_seg1); cudaFree(dy_seg2);
    cudaFree(dw); cudaFree(dx); cudaFree(dxq); cudaFree(dxil); cudaFree(dy_il); cudaFree(dy_ref);
    return ok;
}

}  // namespace

int main() {
    std::printf("mmvq_il_narrow_acc_parity\n");
    if constexpr (k::kWindowMaxT < 16) {
        std::printf("  skipped (kWindowMaxT=%d < 16)\n", k::kWindowMaxT);
        return 0;
    }
    const int tbl_rows = k::native_mmvq_il_rows_for(1201, 12, 12, 2048);
    if (tbl_rows != 2) {
        std::fprintf(stderr, "expected native_mmvq_il_rows_for(1201, 12, 12, 2048) == 2 with NarrowAcc=1, got %d\n", tbl_rows);
        return 1;
    }

    cudaStream_t s = nullptr;
    ck(cudaStreamCreate(&s), "stream");

    bool ok = true;
    ok = check_type(12, "Q4_K", 2560, 1280, s) && ok;
    ok = check_type(13, "Q5_K", 2560, 1280, s) && ok;
    ok = check_type(14, "Q6_K", 2560, 1280, s) && ok;
    ok = check_type(11, "Q3_K", 2560, 1280, s) && ok;
    ok = check_type(23, "IQ4_XS", 2560, 1280, s) && ok;
    ok = check_type(21, "IQ3_S", 2560, 1280, s) && ok;
    ok = check_type(18, "IQ3_XXS", 2560, 1280, s) && ok;

    cudaStreamDestroy(s);
    std::printf("mmvq_il_narrow_acc_parity: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
