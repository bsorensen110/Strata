// src/kernels/kv_q2_parity.cpp - the 2-bit KV format (kv_q2.hpp) against a full-precision reference (GPU, no
// model). Design and the byte math: docs/kv-q2.md.
//
// The pools are built by the real append kernels from rotated data, then read back by the real gather kernel:
//   1. the device quantizer reproduces the host codebook bit for bit (the parity the parity test is named for -
//      the engine's writer and reader cannot drift apart);
//   2. the relative RMS of K and V against an FP16 cache, for q2 and for q4_0 side by side, which is the cost of
//      going from 4 bits to 2 stated in the same units as the window work measures it;
//   3. the attention-output error that follows from it: the same query, the same cells, softmax over the reference
//      K/V and over the quantized ones, so a number that reaches the model is reported, not just a per-element one;
//   4. both V codebooks (the 3-level {-1,0,+1} and the 4-level {-3,-1,+1,+3}) measured, so the codebook is chosen
//      from data. The 3-level form is the one SKVQ's 1.5-bit V needs; if it is too lossy, that is said here.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q2.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/qsa.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
int g_fail = 0;
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_fail;
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}

/// The host reference quantizer: the same warp rule as the kernel, run serially. d is the block's scale from the
/// largest |x| in the block; ties resolve to the larger value, as they do in every lane on the device.
void quant_block(const float* x, int n, bool three_level, float* out) {
    float amax = 0.f, mval = 0.f;
    for (int i = 0; i < n; ++i) {
        const float a = std::fabs(x[i]);
        if (a > amax || (a == amax && x[i] > mval)) { amax = a; mval = x[i]; }
    }
    const float d = three_level ? amax : amax / 3.0f;
    // Codes are chosen with the float scale, exactly as the kernel does, but the value the kernel writes back is
    // built from the scale as STORED: d lives in the block as fp16, so the host has to round it the same way
    // before multiplying, or the two disagree on every block whose scale is not exactly representable.
    const float ds = k::f32_from_f16(k::f16_from_f32(d));
    for (int i = 0; i < n; ++i) {
        const int c = three_level ? k::q2_code_3level(x[i], d) : k::q2_code_4level(x[i], d);
        out[i] = three_level ? k::q2_value_3level(c, ds) : k::q2_value_4level(c, ds);
    }
}

/// Lloyd-Max levels for a Gaussian source: the codebook SKVQ's gain rests on. Scaled by the block's RMS rather
/// than its max, so the inner level sits at 0.46 sigma where the mass is, not at 0.9 sigma where amax/3 puts it.
void quant_lm(const float* x, int n, bool three_level, float* out) {
    // The warp sums x*x in a xor-butterfly (offsets 16,8,4,2,1), which every lane finishes with the same value
    // but in a fixed addition order. Summing in double, or left to right, gives an RMS one ulp away, and an ulp
    // in the scale is enough to flip a code at a threshold - so the host replays the butterfly.
    float ss[k::Q2_0], prev[k::Q2_0];
    for (int i = 0; i < n; ++i) ss[i] = x[i] * x[i];
    for (int o = 16; o > 0; o >>= 1) {
        for (int i = 0; i < n; ++i) prev[i] = ss[i];
        for (int i = 0; i < n; ++i) ss[i] = prev[i] + prev[i ^ o];
    }
    const float s = sqrtf(ss[0] * (1.0f / (float) k::Q2_0));
    const float ds = k::f32_from_f16(k::f16_from_f32(s));               // the scale as the block stores it
    for (int i = 0; i < n; ++i)
        out[i] = three_level ? k::q2_value_lm3(k::q2_code_lm3(x[i], s), ds)
                             : k::q2_value_lm4(k::q2_code_lm4(x[i], s), ds);
}

/// Relative RMS of `got` against `ref`, over the whole vector: the same units the window work reports in.
double rel_rms(const std::vector<float>& got, const std::vector<float>& ref) {
    double se = 0.0, ss = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double e = (double) got[i] - ref[i];
        se += e * e;
        ss += (double) ref[i] * ref[i];
    }
    return ss > 0.0 ? std::sqrt(se / ss) : 0.0;
}

/// The attention output's error measured against the cache's own value scale. Dividing by the norm of the
/// reference output is the obvious choice and is wrong: attention averages zero-mean values, so the reference
/// output is small, and any error reads as hundreds of percent. Against sqrt(mean(V^2)) the number says what it
/// looks like next to the data the model is mixing.
double out_err(const std::vector<float>& got, const std::vector<float>& ref, const std::vector<float>& scale_of) {
    double se = 0.0, ss = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double e = (double) got[i] - ref[i];
        se += e * e;
    }
    for (const float x : scale_of) ss += (double) x * x;
    return (ss > 0.0 && !got.empty()) ? std::sqrt(se / got.size()) / std::sqrt(ss / scale_of.size()) : 0.0;
}

/// Softmax attention with the given K/V, one head, over `cells` cell indices: the score a query would get, and the
/// output it would return. Reference and quantized run through this the same way, so the difference is the error
/// the model would see. The 1/sqrt(D) score scaling is not optional: without it the scores of a 256-dim head span
/// hundreds of logits, the softmax degenerates to an argmax, and a 4-bit cache measures worse than a 2-bit one
/// because flipping the single winning cell moves the whole output.
void attention(const std::vector<float>& q, const std::vector<float>& K, const std::vector<float>& V, int D,
               const std::vector<int>& cells, std::vector<float>& out) {
    out.assign(q.size(), 0.f);
    const double temp = 1.0 / std::sqrt((double) D);
    std::vector<double> score(cells.size());
    double mx = -1e300, sum = 0.0;
    for (size_t c = 0; c < cells.size(); ++c) {
        double s = 0.0;
        for (int d = 0; d < D; ++d) s += (double) q[d] * K[(size_t) cells[c] * D + d];
        score[c] = s * temp;
        if (s * temp > mx) mx = s * temp;
    }
    for (double& s : score) { s = std::exp(s - mx); sum += s; }
    for (double& s : score) s /= sum;
    for (size_t c = 0; c < cells.size(); ++c)
        for (int d = 0; d < D; ++d) out[d] += (float) (score[c] * V[(size_t) cells[c] * D + d]);
}
}  // namespace

int main() {
    k::QsaShapes s = k::qsa_real_shapes();
    s.page_size = 4;
    const int H = (int) s.n_head_kv, D = (int) s.head_dim, P = (int) s.page_size;
    const int N = 1024;                                  // cells in the cache
    const int pages = N / P;
    const int rows = N * H;                              // identity layout: one row per (cell, kv head)
    const int groups = D / k::Q2_0;                      // 8 blocks of 32 per head

    std::mt19937 rng(23);
    std::normal_distribution<float> nd(0.f, 1.f);
    // Mild per-cell magnitude spread on purpose: a cache where every cell has the same scale flatters every
    // quantizer, because the softmax weights barely move. 25x spikes were tried first and made the probe useless -
    // they hand the whole softmax to a handful of cells, so flipping which one wins (which 4 bits can do) moves
    // the entire output, and q4_0 measured worse than a 2-bit cache. 3x is the regime a real rotated cache has.
    std::vector<float> K((size_t) N * H * D), V((size_t) N * H * D);
    for (int cell = 0; cell < N; ++cell) {
        const float sc = (cell % 17 == 0) ? 3.0f : (cell % 23 == 0 ? 0.1f : 1.0f);
        for (int i = 0; i < H * D; ++i) {
            K[(size_t) cell * H * D + i] = nd(rng) * sc;
            V[(size_t) cell * H * D + i] = nd(rng) * sc;
        }
    }

    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = i;
    int32_t* d_table = dalloc<int32_t>(pages);
    ck(cudaMemcpy(d_table, table.data(), pages * 4, cudaMemcpyHostToDevice), "table");

    const size_t q2_bytes = (size_t) rows * k::kv_q2_bytes_per_head(D);
    const size_t q4_bytes = (size_t) rows * k::kv_q4_bytes_per_head(D);
    uint8_t *kq2 = dalloc<uint8_t>(q2_bytes), *vq2 = dalloc<uint8_t>(q2_bytes);
    uint8_t* vq2_4 = dalloc<uint8_t>(q2_bytes);                 // V written with the 4-level codebook, its own pool
    uint8_t *kq4 = dalloc<uint8_t>(q4_bytes), *vq4 = dalloc<uint8_t>(q4_bytes);
    uint16_t* k_scr = dalloc<uint16_t>((size_t) rows * D);
    uint16_t* v_scr = dalloc<uint16_t>((size_t) rows * D);
    uint16_t* v_scr4 = dalloc<uint16_t>((size_t) rows * D);      // V read back with the 4-level codebook
    uint16_t* k_scr_q4 = dalloc<uint16_t>((size_t) rows * D);    // q4_0 gathers into their own scratch, so they
    uint16_t* v_scr_q4 = dalloc<uint16_t>((size_t) rows * D);    // cannot overwrite the q2 results already there
    float* dK = dalloc<float>(K.size());
    float* dV = dalloc<float>(V.size());
    ck(cudaMemcpy(dK, K.data(), K.size() * 4, cudaMemcpyHostToDevice), "K");
    ck(cudaMemcpy(dV, V.data(), V.size() * 4, cudaMemcpyHostToDevice), "V");

    // Rotate, exactly as the engine does before it quantizes: the reference below is the ROTATED cache, because
    // that is what the codes represent, and the query is rotated the same way.
    k::fwht256_cuda(dK, dK, (int64_t) rows, nullptr);
    k::fwht256_cuda(dV, dV, (int64_t) rows, nullptr);
    ck(cudaDeviceSynchronize(), "rotate");
    std::vector<float> Kref(K.size()), Vref(V.size());
    ck(cudaMemcpy(Kref.data(), dK, K.size() * 4, cudaMemcpyDeviceToHost), "Kref");
    ck(cudaMemcpy(Vref.data(), dV, V.size() * 4, cudaMemcpyDeviceToHost), "Vref");

    // Append through the real kernels: the prompt path, T = N cells from pos 0. The pools are the destination;
    // dK/dV are the rotated source tensors.
    k::kv_append_q2(kq2, vq2, d_table, 0, N, dK, dV, s, nullptr, nullptr, nullptr, /*v3level*/ true);
    k::kv_append_q2(kq2, vq2_4, d_table, 0, N, dK, dV, s, nullptr, nullptr, nullptr, /*v3level*/ false);
    k::kv_append_q4(kq4, vq4, d_table, 0, N, dK, dV, s, nullptr, nullptr);
    ck(cudaDeviceSynchronize(), "append");

    // Read every cell back with the real gather kernel.
    std::vector<int32_t> ids(N);
    for (int i = 0; i < N; ++i) ids[i] = i;
    int32_t* d_ids = dalloc<int32_t>(N);
    ck(cudaMemcpy(d_ids, ids.data(), N * 4, cudaMemcpyHostToDevice), "ids");
    int32_t hstep[k::kStepCount] = {N - 1, N, 0, N};
    int32_t* d_step = dalloc<int32_t>(k::kStepCount);
    ck(cudaMemcpy(d_step, hstep, sizeof hstep, cudaMemcpyHostToDevice), "step");
    k::kv_gather_q2_step(kq2, vq2, d_table, d_ids, d_step, N, s, k_scr, v_scr, nullptr, /*v3level*/ true);
    k::kv_gather_q2_step(kq2, vq2_4, d_table, d_ids, d_step, N, s, k_scr, v_scr4, nullptr, /*v3level*/ false);
    ck(cudaDeviceSynchronize(), "gather");

    std::vector<uint16_t> h_k(rows * D), h_v(rows * D), h_v4(rows * D);
    ck(cudaMemcpy(h_k.data(), k_scr, h_k.size() * 2, cudaMemcpyDeviceToHost), "k back");
    ck(cudaMemcpy(h_v.data(), v_scr, h_v.size() * 2, cudaMemcpyDeviceToHost), "v back");
    ck(cudaMemcpy(h_v4.data(), v_scr4, h_v4.size() * 2, cudaMemcpyDeviceToHost), "v4 back");

    // 1. the device quantizer equals the host codebook, bit for bit, over every block
    bool bit_exact = true;
    for (int cell = 0; cell < N && bit_exact; ++cell)
        for (int h = 0; h < H && bit_exact; ++h)
            for (int g = 0; g < groups && bit_exact; ++g) {
                const size_t base = ((size_t) cell * H + h) * D + (size_t) g * k::Q2_0;
                float ref_k4[k::Q2_0], ref_v3[k::Q2_0], ref_v4[k::Q2_0];
                quant_lm(Kref.data() + base, k::Q2_0, false, ref_k4);   // K: the 4-level Lloyd-Max codebook
                quant_lm(Vref.data() + base, k::Q2_0, true, ref_v3);    // V: the 3-level Lloyd-Max codebook
                quant_lm(Vref.data() + base, k::Q2_0, false, ref_v4);   // V: the 4-level, same block
                for (int i = 0; i < k::Q2_0; ++i) {
                    // The gathered values are fp16 bits straight off the device; the reference is quantized on the
                    // host and rounded the same way, so an unequal bit pattern is a real writer/reader drift.
                    if (h_k[base + i] != k::f16_from_f32(ref_k4[i])) bit_exact = false;
                    if (h_v[base + i] != k::f16_from_f32(ref_v3[i])) bit_exact = false;
                    if (h_v4[base + i] != k::f16_from_f32(ref_v4[i])) bit_exact = false;
                }
            }
    check(bit_exact, "the device quantizer reproduces the host codebook bitwise (K 4-level, V 3-level and 4-level)");

    // 2. the cost of 2 bits, in the same units as the window work
    std::vector<float> h_k2(h_k.size()), h_v2_3(h_v.size()), h_v2_4(h_v4.size()), h_k4(rows * D), h_vq4(rows * D);
    for (size_t i = 0; i < h_k.size(); ++i) h_k2[i] = k::f32_from_f16(h_k[i]);
    for (size_t i = 0; i < h_v.size(); ++i) h_v2_3[i] = k::f32_from_f16(h_v[i]);
    for (size_t i = 0; i < h_v4.size(); ++i) h_v2_4[i] = k::f32_from_f16(h_v4[i]);

    std::vector<uint16_t> w_k4(rows * D), w_v4(rows * D);
    k::kv_gather_q4_step(kq4, vq4, d_table, d_ids, d_step, N, s, k_scr_q4, v_scr_q4, nullptr);
    ck(cudaDeviceSynchronize(), "gather q4");
    ck(cudaMemcpy(w_k4.data(), k_scr_q4, w_k4.size() * 2, cudaMemcpyDeviceToHost), "k q4");
    ck(cudaMemcpy(w_v4.data(), v_scr_q4, w_v4.size() * 2, cudaMemcpyDeviceToHost), "v q4");
    for (size_t i = 0; i < w_k4.size(); ++i) h_k4[i] = k::f32_from_f16(w_k4[i]);
    for (size_t i = 0; i < w_v4.size(); ++i) h_vq4[i] = k::f32_from_f16(w_v4[i]);

    const double r_k2 = rel_rms(h_k2, Kref), r_v2_3 = rel_rms(h_v2_3, Vref);
    const double r_v2_4 = rel_rms(h_v2_4, Vref), r_k4 = rel_rms(h_k4, Kref), r_v4 = rel_rms(h_vq4, Vref);

    // The uniform grid, quantized on the host: the device now ships Lloyd-Max, so the uniform rows are the
    // comparison, not a second read of the same pools.
    std::vector<float> h_k_uni(Kref.size()), h_v_uni4(Vref.size()), h_v_uni3(Vref.size());
    for (size_t row = 0; row < rows; ++row)
        for (int b = 0; b < D / k::Q2_0; ++b) {
            const size_t base = row * D + (size_t) b * k::Q2_0;
            quant_block(Kref.data() + base, k::Q2_0, false, h_k_uni.data() + base);
            quant_block(Vref.data() + base, k::Q2_0, false, h_v_uni4.data() + base);
            quant_block(Vref.data() + base, k::Q2_0, true, h_v_uni3.data() + base);
        }
    const double r_k_uni = rel_rms(h_k_uni, Kref), r_v_uni4 = rel_rms(h_v_uni4, Vref);
    const double r_v_uni3 = rel_rms(h_v_uni3, Vref);
    // h_k2 / h_v2_4 / h_v2_3 are the device's Lloyd-Max pools, read back through the real gather kernel.
    const double r_k_lm = r_k2, r_v_lm4 = r_v2_4, r_v_lm3 = r_v2_3;
    std::printf("\nError against an FP16 cache (rotated), relative RMS:\n");
    std::printf("  %-26s %8s %8s\n", "format", "K", "V");
    std::printf("  %-26s %7.3f%% %7.3f%%\n", "q4_0 (the format in use)", 100 * r_k4, 100 * r_v4);
    std::printf("  %-26s %7.3f%% %7.3f%%\n", "q2 uniform, V 4-level", 100 * r_k_uni, 100 * r_v_uni4);
    std::printf("  %-26s %7.3f%% %7.3f%%\n", "q2 uniform, V 3-level", 100 * r_k_uni, 100 * r_v_uni3);
    std::printf("  %-26s %7.3f%% %7.3f%%\n", "q2 Lloyd-Max, V 4-level", 100 * r_k_lm, 100 * r_v_lm4);
    std::printf("  %-26s %7.3f%% %7.3f%%\n", "q2 Lloyd-Max, V 3-level", 100 * r_k_lm, 100 * r_v_lm3);

    check(r_k_lm > r_k4, "2-bit K is worse than 4-bit K (the table is not measuring the same thing twice)");
    check(r_v_lm3 > r_v_lm4, "the 3-level V is lossier than the 4-level V (so the codebook is a real choice)");
    std::printf("  ratio 2-bit K / 4-bit K: %.2fx\n", r_k_lm / r_k4);
    check(r_k_lm < 8.0 * r_k4, "2-bit K's error stays under 8x the 4-bit error (the per-token block scale is not collapsing)");
    check(r_k_lm < r_k_uni && r_v_lm4 < r_v_uni4, "the Lloyd-Max codebook beats the uniform one at the same 2 bits");
    std::printf("  Lloyd-Max gain over uniform at 2 bits: K %.2fx, V %.2fx\n", r_k_uni / r_k_lm, r_v_uni4 / r_v_lm4);

    // 3. the attention-output error, one head, over the whole cache
    std::vector<float> q(D);
    for (auto& x : q) x = nd(rng);
    std::vector<int> cells(N);
    for (int i = 0; i < N; ++i) cells[i] = i;
    std::vector<float> out_ref, out_lm3, out_lm4, out_q4, out_uni4, out_uni3;
    attention(q, Kref, Vref, D, cells, out_ref);
    attention(q, h_k2, h_v2_3, D, cells, out_lm3);       // device Lloyd-Max K, Lloyd-Max 3-level V
    attention(q, h_k2, h_v2_4, D, cells, out_lm4);       // device Lloyd-Max K, Lloyd-Max 4-level V
    attention(q, h_k4, h_vq4, D, cells, out_q4);
    attention(q, h_k_uni, h_v_uni4, D, cells, out_uni4);
    attention(q, h_k_uni, h_v_uni3, D, cells, out_uni3);
    const double e_lm3 = out_err(out_lm3, out_ref, Vref), e_lm4 = out_err(out_lm4, out_ref, Vref);
    const double e_q4 = out_err(out_q4, out_ref, Vref);
    const double e_uni4 = out_err(out_uni4, out_ref, Vref), e_uni3 = out_err(out_uni3, out_ref, Vref);
    std::printf("\nAttention output against the FP16 cache, relative RMS:\n");
    std::printf("  %-26s %7.3f%%\n", "q4_0 (the format in use)", 100 * e_q4);
    std::printf("  %-26s %7.3f%%\n", "q2 uniform, V 4-level", 100 * e_uni4);
    std::printf("  %-26s %7.3f%%\n", "q2 uniform, V 3-level", 100 * e_uni3);
    std::printf("  %-26s %7.3f%%\n", "q2 Lloyd-Max, V 4-level", 100 * e_lm4);
    std::printf("  %-26s %7.3f%%\n", "q2 Lloyd-Max, V 3-level", 100 * e_lm3);
    check(e_lm4 > 0.0 && e_q4 > 0.0, "both quantized caches move the attention output (the probe is live)");
    check(e_lm4 < e_uni4 && e_lm3 < e_uni3, "the Lloyd-Max codebook moves the attention output less than the grid");

    // 4. the byte math the design doc quotes
    std::printf("\nBytes per cell and KV head: fp16 %d, q4_0 %llu, q2 %llu (head_dim %d)\n", 2 * D,
                (unsigned long long) k::kv_q4_bytes_per_head(D), (unsigned long long) k::kv_q2_bytes_per_head(D), D);
    check(k::kv_q2_bytes_per_head(D) == 80, "q2 is 80 B per cell and head (8 blocks x 10 B)");
    check(k::kv_q2_bytes_per_cell(s) == 2 * 2 * 80, "q2 is 320 B per cell for 2 KV heads (vs 576 for q4_0)");

    std::printf("\n%s\n", g_fail == 0 ? "kv_q2_parity: all checks passed" : "kv_q2_parity: FAILURES");
    return g_fail == 0 ? 0 : 1;
}
