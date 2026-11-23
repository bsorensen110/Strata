// src/kernels/moe_wide_bucket_parity.cpp - bitwise parity for wide MoE LDS_NT=8 + STRATA_MOE_GRID_CAP
// across groups of size 0..16 vs mode-0 reference.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
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

int block_values(int t) { return (t == 20 || t == 7 || t == 8) ? 32 : t == 42 ? 64 : 256; }

std::vector<uint8_t> random_rows(int t, int64_t rows, int64_t n, std::mt19937& rng) {
    const size_t rb = k::iq_row_bytes(t, n), bs = k::iq_row_bytes(t, block_values(t));
    std::vector<uint8_t> w((size_t) rows * rb);
    std::uniform_int_distribution<int> byte(0, 255), ex(2, 8), man(0, 1023), sgn(0, 3);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (size_t o = 0; o < w.size(); o += bs) {
        const uint16_t h = (uint16_t) ((sgn(rng) == 0 ? 0x8000 : 0) | (ex(rng) << 10) | man(rng));
        std::memcpy(&w[o], &h, 2);
        if (t == 12 || t == 13 || t == 7) {
            const uint16_t h2 = (uint16_t) ((sgn(rng) == 0 ? 0x8000 : 0) | (ex(rng) << 10) | man(rng));
            std::memcpy(&w[o + 2], &h2, 2);
        }
    }
    return w;
}

bool check_case(int gu, int dt, int64_t H, int64_t FF, const std::vector<int>& counts, int tokens, cudaStream_t s) {
    std::mt19937 rng(701 + gu * 17 + dt * 31 + (int) counts.size());
    k::NativeExpertLayout L = k::native_expert_layout(gu, dt, H, FF);
    const int n_groups = (int) counts.size();
    const int cap_groups = n_groups + 2;
    std::vector<int32_t> start(1, 0), tok, dst;
    for (int c : counts) start.push_back(start.back() + c);
    const int n_ent = start.back();
    const int cap_ent = n_ent + 3;
    std::uniform_int_distribution<int> tk(0, tokens - 1);
    for (int e = 0; e < n_ent; ++e) tok.push_back(tk(rng));
    dst.resize(n_ent);
    std::iota(dst.begin(), dst.end(), 0);
    std::shuffle(dst.begin(), dst.end(), rng);

    std::vector<uint8_t*> blobs;
    std::vector<unsigned long long> ptr;
    for (int g = 0; g < n_groups; ++g) {
        auto b = random_rows(gu, 2 * FF, H, rng);
        const auto d = random_rows(dt, H, FF, rng);
        b.insert(b.end(), d.begin(), d.end());
        uint8_t* db = dalloc<uint8_t>(b.size());
        ck(cudaMemcpy(db, b.data(), b.size(), cudaMemcpyHostToDevice), "blob");
        blobs.push_back(db);
        ptr.push_back((unsigned long long) db);
    }

    auto* dptr = dalloc<unsigned long long>(cap_groups);
    auto* dstart = dalloc<int32_t>(cap_groups + 1);
    auto* dn = dalloc<int32_t>(1);
    auto* ddst = dalloc<int32_t>(cap_ent);
    auto* dtok = dalloc<int32_t>(cap_ent);
    ck(cudaMemcpy(dptr, ptr.data(), ptr.size() * 8, cudaMemcpyHostToDevice), "ptr");
    ck(cudaMemcpy(dstart, start.data(), start.size() * 4, cudaMemcpyHostToDevice), "start");
    ck(cudaMemcpy(dn, &n_groups, 4, cudaMemcpyHostToDevice), "n");
    if (n_ent) {
        ck(cudaMemcpy(ddst, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice), "dst");
        ck(cudaMemcpy(dtok, tok.data(), tok.size() * 4, cudaMemcpyHostToDevice), "tok");
    }

    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) tokens * H);
    for (auto& v : x) v = nd(rng);
    float* dx = dalloc<float>(x.size());
    ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    uint8_t* dxq = dalloc<uint8_t>((size_t) tokens * (H / 32) * 36);
    uint8_t* dscr = dalloc<uint8_t>(k::native_expert_scratch_bytes(cap_ent, FF));
    const size_t out_floats = (size_t) cap_ent * H;
    float* dout = dalloc<float>(out_floats);

    auto run_mode = [&](int mode) {
        ck(cudaMemset(dout, 0xFF, out_floats * 4), "memset");
        k::native_expert_set_mode(mode, 0);
        k::quantize_q8_1_rows(dx, tokens, H, dxq, s);
        k::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, cap_groups, cap_ent, dxq, dscr, dout, s);
        ck(cudaStreamSynchronize(s), "sync");
        std::vector<float> out(out_floats);
        ck(cudaMemcpy(out.data(), dout, out_floats * 4, cudaMemcpyDeviceToHost), "d2h");
        k::native_expert_set_mode(-1, 0);
        return out;
    };

    const auto y_ref = run_mode(0);
    const auto y_lds8 = run_mode(8);

    size_t diff = 0;
    for (size_t i = 0; i < out_floats; ++i) {
        if (std::memcmp(&y_ref[i], &y_lds8[i], 4) != 0) ++diff;
    }
    std::printf("  gu=%d down=%d groups=%d entries=%d: %zu diffs\n", gu, dt, n_groups, n_ent, diff);

    for (auto* b : blobs) cudaFree(b);
    cudaFree(dptr); cudaFree(dstart); cudaFree(dn); cudaFree(ddst); cudaFree(dtok);
    cudaFree(dx); cudaFree(dxq); cudaFree(dscr); cudaFree(dout);
    return diff == 0;
}

}  // namespace

int main() {
    std::printf("moe_wide_bucket_parity\n");
    cudaStream_t s = nullptr;
    ck(cudaStreamCreate(&s), "stream");

    const std::vector<int> counts = {16, 12, 8, 7, 5, 3, 1, 0};
    bool ok = true;
    ok = check_case(21, 20, 2560, 640, counts, 16, s) && ok;   // IQ3_S + IQ4_NL
    ok = check_case(22, 42, 2560, 640, counts, 16, s) && ok;   // IQ2_S + Q2_0
    ok = check_case(18, 20, 2560, 640, counts, 16, s) && ok;   // IQ3_XXS + IQ4_NL
    ok = check_case(16, 42, 2560, 640, counts, 16, s) && ok;   // IQ2_XXS + Q2_0
    ok = check_case(23, 20, 2560, 640, counts, 16, s) && ok;   // IQ4_XS + IQ4_NL

    cudaStreamDestroy(s);
    std::printf("moe_wide_bucket_parity: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
