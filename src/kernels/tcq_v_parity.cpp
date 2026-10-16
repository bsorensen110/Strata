// src/kernels/tcq_v_parity.cpp - device parity gate for the trellis-coded 2-bit V (tcq_v.hpp, tcq_v_cuda.hpp). GPU, no model.
//
// The host reference is the same header compiled for the host (tcq_quantize_block / tcq_decode_block). Checked for
// each trellis (4, 8, 16 states), on Qwen-shaped rows (2 kv heads, head_dim 256) with per-row scales spanning 3
// decades, zero rows, and one spike row:
//   1. writer: device blocks equal host blocks bit for bit (f16 scale bits and all 8 code bytes);
//   2. reader: device output equals the host decode of the same blocks, bit for bit;
//   3. row selection: the reader with an ids array (repeats allowed) equals the host decode of those rows, bit for bit;
//   4. zero rows decode to exact zeros with scale bits 0;
//   5. accuracy: relative RMS error of the round trip sits in the band the prototype predicts for Gaussian data
//      (0.313 for TCQ-4 at 2 bits; the scalar LM4 baseline on the same data, printed for context, is about 0.344).
//
// Build (CUDA 13.0, see build-cu130-tests.bat): add_executable(tcq_v_parity ...) in CMakeLists.txt.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/tcq_v.hpp"
#include "strata/kernels/tcq_v_cuda.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_fail = 0;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_fail;
}

template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "cudaMalloc");
    return p;
}

// The engine's scalar 2-bit codebook on the same blocks (per-token RMS scale, f16), for context only.
const float kLM4[4] = {-1.5097f, -0.4591f, 0.4591f, 1.5097f};

double scalar_lm4_rel_rms(const std::vector<float>& x) {
    double num = 0.0, den = 0.0;
    for (size_t b = 0; b + 32 <= x.size(); b += 32) {
        float ss = 0.0f;
        for (int t = 0; t < 32; ++t) ss += x[b + t] * x[b + t];
        const float s = k::f32_from_f16(k::f16_from_f32(std::sqrt(ss / 32.0f)));
        for (int t = 0; t < 32; ++t) {
            const float u = s > 0 ? x[b + t] / s : 0.0f;
            int best = 0;
            for (int c = 1; c < 4; ++c)
                if (std::fabs(u - kLM4[c]) < std::fabs(u - kLM4[best])) best = c;
            const float rec = s * kLM4[best];
            num += (double) (rec - x[b + t]) * (double) (rec - x[b + t]);
            den += (double) x[b + t] * (double) x[b + t];
        }
    }
    return std::sqrt(num / den);
}

}  // namespace

int main() {
    const int n_rows = 512, n_head_kv = 2, head_dim = 256;
    const int nb = head_dim / 32;
    const size_t n_vals = (size_t) n_rows * n_head_kv * head_dim;
    const size_t n_blk = (size_t) n_rows * n_head_kv * nb;

    // data: per-row scale exp(U(-3,3)), zero rows every 37th, one spike row
    std::mt19937 rng(20261015);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(-3.0f, 3.0f);
    std::vector<float> x(n_vals);
    for (int r = 0; r < n_rows; ++r) {
        const float sc = std::exp(ud(rng));
        for (int i = 0; i < n_head_kv * head_dim; ++i) {
            float v = sc * nd(rng);
            if (r % 37 == 0) v = 0.0f;
            x[(size_t) r * n_head_kv * head_dim + i] = v;
        }
    }
    for (int i = 0; i < head_dim; ++i) x[(size_t) 5 * n_head_kv * head_dim + i] *= 50.0f;  // spike row

    // row selection with repeats
    const int n_sel = 300;
    std::vector<int32_t> ids(n_sel);
    std::uniform_int_distribution<int> rd(0, n_rows - 1);
    for (auto& v : ids) v = rd(rng);

    static_assert(sizeof(k::block_tcq2_0) == 10, "block size");
    std::printf("block_tcq2_0 = %zu bytes; %d rows x %d kv heads x %d dims; %zu blocks\n", sizeof(k::block_tcq2_0),
                n_rows, n_head_kv, head_dim, n_blk);

    const int trellis[3] = {4, 8, 16};
    for (int states : trellis) {
        std::printf("--- %d states\n", states);
        // host reference
        std::vector<k::block_tcq2_0> hb(n_blk);
        for (size_t i = 0; i < n_blk; ++i) k::tcq_quantize_block(states, &x[i * 32], &hb[i]);
        std::vector<float> host_dec(n_vals);
        for (size_t i = 0; i < n_blk; ++i)
            k::tcq_decode_block(states, k::f32_from_f16(hb[i].d), hb[i].qs, &host_dec[i * 32]);

        // device writer
        float* dX = dalloc<float>(n_vals);
        k::block_tcq2_0* dB = dalloc<k::block_tcq2_0>(n_blk);
        ck(cudaMemcpy(dX, x.data(), n_vals * sizeof(float), cudaMemcpyHostToDevice), "x");
        k::tcq_v_quantize_rows(states, dX, dB, n_rows, n_head_kv, head_dim, nullptr);
        ck(cudaGetLastError(), "quantize launch");
        ck(cudaDeviceSynchronize(), "quantize sync");
        std::vector<k::block_tcq2_0> db(n_blk);
        ck(cudaMemcpy(db.data(), dB, n_blk * sizeof(k::block_tcq2_0), cudaMemcpyDeviceToHost), "blocks");
        size_t scale_bad = 0, code_bad = 0;
        for (size_t i = 0; i < n_blk; ++i) {
            if (db[i].d != hb[i].d) ++scale_bad;
            if (std::memcmp(db[i].qs, hb[i].qs, 8) != 0) ++code_bad;
        }
        std::printf("     writer: scale mismatches %zu, block code mismatches %zu / %zu\n", scale_bad, code_bad, n_blk);

        // device reader on the device-written blocks, bit-exact vs host decode of the same blocks
        float* dO = dalloc<float>(n_vals);
        k::tcq_v_dequantize_rows(states, dB, dO, nullptr, n_rows, n_head_kv, head_dim, nullptr);
        ck(cudaGetLastError(), "dequantize launch");
        ck(cudaDeviceSynchronize(), "dequantize sync");
        std::vector<float> dev_dec(n_vals);
        ck(cudaMemcpy(dev_dec.data(), dO, n_vals * sizeof(float), cudaMemcpyDeviceToHost), "dec");
        size_t dec_bad = 0;
        for (size_t i = 0; i < n_vals; ++i)
            if (std::memcmp(&dev_dec[i], &host_dec[i], sizeof(float)) != 0) ++dec_bad;
        std::printf("     reader: values differing from host decode %zu / %zu\n", dec_bad, n_vals);

        // row selection: device reader with ids vs host decode of the selected rows
        k::block_tcq2_0* dBh = dalloc<k::block_tcq2_0>(n_blk);
        ck(cudaMemcpy(dBh, hb.data(), n_blk * sizeof(k::block_tcq2_0), cudaMemcpyHostToDevice), "hb");
        int32_t* dIds = dalloc<int32_t>(n_sel);
        ck(cudaMemcpy(dIds, ids.data(), n_sel * sizeof(int32_t), cudaMemcpyHostToDevice), "ids");
        const size_t n_sel_vals = (size_t) n_sel * n_head_kv * head_dim;
        float* dS = dalloc<float>(n_sel_vals);
        k::tcq_v_dequantize_rows(states, dBh, dS, dIds, n_sel, n_head_kv, head_dim, nullptr);
        ck(cudaGetLastError(), "gather launch");
        ck(cudaDeviceSynchronize(), "gather sync");
        std::vector<float> dev_sel(n_sel_vals);
        ck(cudaMemcpy(dev_sel.data(), dS, n_sel_vals * sizeof(float), cudaMemcpyDeviceToHost), "sel");
        size_t sel_bad = 0;
        const size_t row_vals = (size_t) n_head_kv * head_dim;
        for (int r = 0; r < n_sel; ++r)
            for (size_t i = 0; i < row_vals; ++i)
                if (std::memcmp(&dev_sel[r * row_vals + i], &host_dec[(size_t) ids[r] * row_vals + i], sizeof(float)))
                    ++sel_bad;
        std::printf("     gather(ids): values differing from host decode %zu / %zu\n", sel_bad, n_sel_vals);

        // fp16 gather (the engine's scratch format): bitwise vs f16_from_f32 of the host decode of the selected rows
        uint16_t* dF = dalloc<uint16_t>(n_sel_vals);
        k::tcq_v_gather_f16(states, dBh, dF, dIds, n_sel, n_head_kv, head_dim, nullptr);
        ck(cudaGetLastError(), "gather16 launch");
        ck(cudaDeviceSynchronize(), "gather16 sync");
        std::vector<uint16_t> dev_f16(n_sel_vals);
        ck(cudaMemcpy(dev_f16.data(), dF, n_sel_vals * sizeof(uint16_t), cudaMemcpyDeviceToHost), "f16");
        size_t f16_bad = 0;
        for (int r = 0; r < n_sel; ++r)
            for (size_t i = 0; i < row_vals; ++i)
                if (dev_f16[(size_t) r * row_vals + i] != k::f16_from_f32(host_dec[(size_t) ids[r] * row_vals + i]))
                    ++f16_bad;
        std::printf("     gather fp16 (ids): bits differing from f16(host decode) %zu / %zu\n", f16_bad, n_sel_vals);
        {
            char m2[128];
            std::snprintf(m2, sizeof m2, "%d states: fp16 gather bit-exact vs f16(host decode)", states);
            check(f16_bad == 0, m2);
        }
        cudaFree(dF);

        // informational timing (no pass/fail) on the 4-state pool: decode-step shape, 2048 selected rows
        if (states == 4) {
            const int T_rows = 2048, iters = 200;
            std::vector<int32_t> tids(T_rows);
            std::uniform_int_distribution<int> trd(0, n_rows - 1);
            for (auto& v : tids) v = trd(rng);
            int32_t* dT = dalloc<int32_t>(T_rows);
            ck(cudaMemcpy(dT, tids.data(), T_rows * sizeof(int32_t), cudaMemcpyHostToDevice), "tids");
            const size_t out_vals = (size_t) T_rows * n_head_kv * head_dim;
            uint16_t* dO16 = dalloc<uint16_t>(out_vals);
            float* dOF = dalloc<float>(out_vals);
            cudaEvent_t e0, e1;
            ck(cudaEventCreate(&e0), "ev");
            ck(cudaEventCreate(&e1), "ev");
            for (int pass = 0; pass < 2; ++pass) {
                for (int i = 0; i < 10; ++i) {
                    if (pass == 0) k::tcq_v_gather_f16(states, dBh, dO16, dT, T_rows, n_head_kv, head_dim, nullptr);
                    else k::tcq_v_dequantize_rows(states, dBh, dOF, dT, T_rows, n_head_kv, head_dim, nullptr);
                }
                ck(cudaDeviceSynchronize(), "warm");
                ck(cudaEventRecord(e0), "rec");
                for (int i = 0; i < iters; ++i) {
                    if (pass == 0) k::tcq_v_gather_f16(states, dBh, dO16, dT, T_rows, n_head_kv, head_dim, nullptr);
                    else k::tcq_v_dequantize_rows(states, dBh, dOF, dT, T_rows, n_head_kv, head_dim, nullptr);
                }
                ck(cudaEventRecord(e1), "rec");
                ck(cudaEventSynchronize(e1), "sync");
                float ms = 0.0f;
                ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
                const double us = ms * 1000.0 / iters;
                const double blocks = (double) T_rows * n_head_kv * nb;
                std::printf("     timing (informational, %s): %d rows x %d heads x %d blocks, %.1f us per call, "
                            "%.2f ns per block\n",
                            pass == 0 ? "gather to fp16" : "dequantize to fp32", T_rows, n_head_kv, nb, us,
                            us * 1000.0 / blocks);
            }
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
            cudaFree(dT);
            cudaFree(dO16);
            cudaFree(dOF);
        }

        // zero rows: scale bits 0 and exact zeros
        size_t zero_bad = 0;
        for (int r = 0; r < n_rows; r += 37) {
            for (int b = 0; b < n_head_kv * nb; ++b) {
                const size_t bi = (size_t) r * n_head_kv * nb + b;
                if (db[bi].d != 0) ++zero_bad;
            }
            for (int i = 0; i < n_head_kv * head_dim; ++i)
                if (dev_dec[(size_t) r * n_head_kv * head_dim + i] != 0.0f) ++zero_bad;
        }

        // accuracy of the round trip vs the fp32 source, and the scalar baseline on the same data
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < n_vals; ++i) {
            num += (double) (dev_dec[i] - x[i]) * (double) (dev_dec[i] - x[i]);
            den += (double) x[i] * (double) x[i];
        }
        const double rel = std::sqrt(num / den);
        const double base = scalar_lm4_rel_rms(x);
        std::printf("     round trip relative RMS %.4f (scalar 2-bit LM4 on the same data: %.4f)\n", rel, base);

        char msg[160];
        std::snprintf(msg, sizeof msg, "%d states: writer bit-exact (scales and codes)", states);
        check(scale_bad == 0 && code_bad == 0, msg);
        std::snprintf(msg, sizeof msg, "%d states: reader bit-exact vs host decode", states);
        check(dec_bad == 0, msg);
        std::snprintf(msg, sizeof msg, "%d states: row-selected reader bit-exact vs host decode", states);
        check(sel_bad == 0, msg);
        std::snprintf(msg, sizeof msg, "%d states: zero rows decode to exact zeros with scale 0", states);
        check(zero_bad == 0, msg);
        std::snprintf(msg, sizeof msg, "%d states: round-trip relative RMS in [0.29, 0.34]", states);
        check(rel > 0.29 && rel < 0.34, msg);

        cudaFree(dX);
        cudaFree(dB);
        cudaFree(dO);
        cudaFree(dBh);
        cudaFree(dIds);
        cudaFree(dS);
    }
    std::printf(g_fail ? "FAILED: %d check(s)\n" : "PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
