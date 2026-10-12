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

/// The same 4-level Lloyd-Max codebook with a scale every 16 values instead of every 32: 12 B per 32 values
/// (two fp16 scales + the same 8 code bytes) against 10 B, bought for a scale that tracks the data twice as
/// closely.  Host-side only - this is the measurement that decides whether the layout changes.
void quant_lm16(const float* x, int n, float* out) {
    for (int b = 0; b < n; b += 16) {
        const float* p = x + b;
        float ss[16], prev[16];
        for (int i = 0; i < 16; ++i) ss[i] = p[i] * p[i];
        for (int o = 8; o > 0; o >>= 1) {                       // the same butterfly the device runs on 16 lanes
            for (int i = 0; i < 16; ++i) prev[i] = ss[i];
            for (int i = 0; i < 16; ++i) ss[i] = prev[i] + prev[i ^ o];
        }
        const float s = sqrtf(ss[0] * (1.0f / 16.0f));
        const float ds = k::f32_from_f16(k::f16_from_f32(s));   // the scale as the block would store it
        for (int i = 0; i < 16; ++i) out[b + i] = k::q2_value_lm4(k::q2_code_lm4(p[i], s), ds);
    }
}

/// KIVI-style per-channel scaling (arXiv 2402.02750): one scale per (dimension, 32-cell group) instead of one per
/// (cell, 32-value block).  It costs the same 80 B per cell and head - 256 dims x 2 bits = 64 B of codes, plus
/// 256 dims x 2 B of scale per 32 cells = 16 B - so this is a like-for-like swap of WHAT the scale measures, not
/// of how many bits are spent.
///
/// THIS IS THE REFERENCE THE DEVICE HAS TO MATCH BIT FOR BIT.  kv_q2.cu's group kernel sums `ss += v*v` over the
/// group's 32 cells in cell order and takes `sqrtf(ss / 32)`; this sums the same 32 numbers in the same order and
/// takes the same square root, then rounds the scale to fp16 and quantizes with the full-precision scale while
/// writing back with the stored one.  Every one of those choices is the kernel's, because an ulp in the scale
/// flips a code at a threshold.
///
/// Caveat the reader must keep: this gate's cache is i.i.d. Gaussian, so every channel's RMS is the same number;
/// per-channel scaling cannot show its gain here beyond the 2.8x on the attention output that it does show.  The
/// structure KIVI's gain rests on - outliers aligned to dimensions - has to come from a real cache.
void quant_kivi(const float* x, int n_cells, int n_heads, int D, bool three_level, float* out) {
    const int stride = n_heads * D;
    for (int h = 0; h < n_heads; ++h)
        for (int d = 0; d < D; ++d)
            for (int g = 0; g < n_cells; g += 32) {
                float ss = 0.0f;
                for (int i = 0; i < 32; ++i) {
                    const float v = x[(size_t) (g + i) * stride + h * D + d];
                    ss += v * v;
                }
                const float s = sqrtf(ss * (1.0f / 32.0f));
                const float ds = k::f32_from_f16(k::f16_from_f32(s));
                for (int i = 0; i < 32; ++i) {
                    const size_t idx = (size_t) (g + i) * stride + h * D + d;
                    out[idx] = three_level ? k::q2_value_lm3(k::q2_code_lm3(x[idx], s), ds)
                                           : k::q2_value_lm4(k::q2_code_lm4(x[idx], s), ds);
                }
            }
}

/// 1-bit (two-level) quantization: the levels are the Lloyd-Max optimal pair for a Gaussian, +-sqrt(2/pi)*sigma,
/// and the code is the sign.  `per_channel` puts the scale on a (dimension, 32-cell) group like quant_kivi;
/// `per_token` puts one fp16 scale on the whole head_dim of a cell.  KIVI's V arm is the per-token one, and the
/// point of the arm is that V is quantized differently from K, not harder than K.
void quant_1bit(const float* x, int n_cells, int n_heads, int D, bool per_channel, float* out) {
    const float lm2 = 0.7978845608028654f;                       // sqrt(2/pi)
    const int stride = n_heads * D;
    if (per_channel) {
        for (int h = 0; h < n_heads; ++h)
            for (int d = 0; d < D; ++d)
                for (int g = 0; g < n_cells; g += 32) {
                    float ss = 0.0f;
                    for (int i = 0; i < 32; ++i) {
                        const float v = x[(size_t) (g + i) * stride + h * D + d];
                        ss += v * v;
                    }
                    const float ds = k::f32_from_f16(k::f16_from_f32(sqrtf(ss * (1.0f / 32.0f))));
                    for (int i = 0; i < 32; ++i) {
                        const size_t idx = (size_t) (g + i) * stride + h * D + d;
                        out[idx] = x[idx] >= 0.0f ? lm2 * ds : -lm2 * ds;
                    }
                }
    } else {
        for (int c = 0; c < n_cells; ++c)
            for (int h = 0; h < n_heads; ++h) {
                const size_t base = (size_t) c * stride + h * D;
                float ss = 0.0f;
                for (int i = 0; i < D; ++i) ss += x[base + i] * x[base + i];
                const float ds = k::f32_from_f16(k::f16_from_f32(sqrtf(ss * (1.0f / (float) D))));
                for (int i = 0; i < D; ++i) out[base + i] = x[base + i] >= 0.0f ? lm2 * ds : -lm2 * ds;
            }
    }
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

    // 1. the device quantizer equals the PER-CHANNEL host reference, bit for bit, over every (dimension, group).
    // quant_kivi is the reference; the per-token Lloyd-Max arm (quant_lm) is a comparison now, not the parity
    // target - the device no longer writes per-token blocks.
    std::vector<float> ref_k(Kref.size()), ref_v3(Vref.size()), ref_v4(Vref.size());
    quant_kivi(Kref.data(), N, H, D, false, ref_k.data());     // K: 4-level Lloyd-Max, per-channel scale
    quant_kivi(Vref.data(), N, H, D, true, ref_v3.data());     // V: the 3-level codebook, same scale
    quant_kivi(Vref.data(), N, H, D, false, ref_v4.data());    // V: the 4-level, same scale
    bool bit_exact = true;
    for (size_t i = 0; i < h_k.size() && bit_exact; ++i) {
        // The gathered values are fp16 bits straight off the device; the reference is quantized on the host and
        // rounded the same way, so an unequal bit pattern is a real writer/reader drift.
        if (h_k[i] != k::f16_from_f32(ref_k[i])) bit_exact = false;
        if (h_v[i] != k::f16_from_f32(ref_v3[i])) bit_exact = false;
        if (h_v4[i] != k::f16_from_f32(ref_v4[i])) bit_exact = false;
    }
    check(bit_exact, "the device quantizer reproduces the per-channel host reference bitwise (K 4-level, "
                     "V 3-level and 4-level)");

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
    // Scale granularity: the shipping codebook with a scale every 16 values.  Host-side, so the number costs
    // nothing to obtain; the layout only changes if it pays for the extra 2 bytes per 32 values.
    std::vector<float> h_k_lm16(Kref.size()), h_v_lm16(Vref.size());
    for (size_t row = 0; row < rows; ++row) {
        quant_lm16(Kref.data() + row * D, D, h_k_lm16.data() + row * D);
        quant_lm16(Vref.data() + row * D, D, h_v_lm16.data() + row * D);
    }
    const double r_k_lm16 = rel_rms(h_k_lm16, Kref), r_v_lm16 = rel_rms(h_v_lm16, Vref);
    // KIVI per-channel scaling, same 2 bits and the same 80 B per cell and head: the host arm of the row the
    // device fills, so the table's per-channel row and the device's Lloyd-Max rows are the same measurement by
    // two paths (and check 1 says they agree bit for bit).
    std::vector<float> h_k_kivi(Kref.size()), h_v_kivi(Vref.size());
    quant_kivi(Kref.data(), N, H, D, false, h_k_kivi.data());
    quant_kivi(Vref.data(), N, H, D, false, h_v_kivi.data());
    const double r_k_kivi = rel_rms(h_k_kivi, Kref), r_v_kivi = rel_rms(h_v_kivi, Vref);
    // h_k2 / h_v2_4 / h_v2_3 are the device's PER-CHANNEL pools, read back through the real gather kernel.
    const double r_k_lm = r_k2, r_v_lm4 = r_v2_4, r_v_lm3 = r_v2_3;
    // The per-token Lloyd-Max layout this format had before: host-only now (the device writes per-channel
    // groups), kept because the scale-placement measurement is stated against it - 33.4% per element against
    // 44.4%, and 19.9% of the attention output against 7.15%.
    std::vector<float> h_k_lmt(Kref.size()), h_v_lmt4(Vref.size());
    for (size_t row = 0; row < rows; ++row)
        for (int b = 0; b < D / k::Q2_0; ++b) {
            const size_t base = row * D + (size_t) b * k::Q2_0;
            quant_lm(Kref.data() + base, k::Q2_0, false, h_k_lmt.data() + base);
            quant_lm(Vref.data() + base, k::Q2_0, false, h_v_lmt4.data() + base);
        }
    const double r_k_lmt = rel_rms(h_k_lmt, Kref), r_v_lmt4 = rel_rms(h_v_lmt4, Vref);
    // ---- Asymmetric K/V bit widths (the KIVI line of work: K and V are not equally sensitive, so they are
    // quantized DIFFERENTLY, not at the same bit width).  Host-side arms composed from the same dequantized
    // arrays the device pools produce, so each arm's byte cost is stated and the comparison is like for like.
    std::vector<float> h_v_1c(Vref.size()), h_v_1t(Vref.size()), h_k_1c(Kref.size());
    quant_1bit(Vref.data(), N, H, D, true, h_v_1c.data());    // V 1-bit, per-channel scale
    quant_1bit(Vref.data(), N, H, D, false, h_v_1t.data());   // V 1-bit, per-token scale (KIVI's V arm)
    quant_1bit(Kref.data(), N, H, D, true, h_k_1c.data());    // K 1-bit, per-channel scale
    const double r_v_1c = rel_rms(h_v_1c, Vref), r_v_1t = rel_rms(h_v_1t, Vref), r_k_1c = rel_rms(h_k_1c, Kref);
    std::printf("\nError against an FP16 cache (rotated), relative RMS:\n");
    std::printf("  %-26s %8s %8s\n", "format", "K", "V");
    std::printf("  (per-element diagnostics - the decision metric is the attention output table below)\n");
    std::printf("  %-26s %7.3f%% %7.3f%%\n", "q4_0 (the format in use)", 100 * r_k4, 100 * r_v4);
    std::printf("  %-26s %7.3f%% %7.3f%%\n", "q2 uniform, V 4-level", 100 * r_k_uni, 100 * r_v_uni4);
    std::printf("  %-26s %7.3f%% %7.3f%%\n", "q2 uniform, V 3-level", 100 * r_k_uni, 100 * r_v_uni3);
    std::printf("  %-26s %7.3f%% %7.3f%%\n", "q2 Lloyd-Max, V 4-level", 100 * r_k_lm, 100 * r_v_lm4);
    std::printf("  %-26s %7.3f%% %7.3f%%\n", "q2 Lloyd-Max, V 3-level", 100 * r_k_lm, 100 * r_v_lm3);
    std::printf("  %-26s %7.3f%% %7.3f%%\n", "q2 Lloyd-Max, 16-value scale", 100 * r_k_lm16, 100 * r_v_lm16);
    std::printf("  %-26s %7.3f%% %7.3f%%\n", "q2 Lloyd-Max, per-token (host)", 100 * r_k_lmt, 100 * r_v_lmt4);
    std::printf("  %-26s %7.3f%% %7.3f%%\n", "q2 Lloyd-Max, per-channel", 100 * r_k_kivi, 100 * r_v_kivi);
    std::printf("  (the two Lloyd-Max V rows are the DEVICE pools; the per-channel row is the same pools quantized\n");
    std::printf("   on the host, which check 1 says agree bit for bit. Per-channel is the WORSE per-element row\n");
    std::printf("   and the better attention row - see the note under the second table.)\n");

    check(r_k_lm > r_k4, "2-bit K is worse than 4-bit K (the table is not measuring the same thing twice)");
    check(r_v_lm3 > r_v_lm4, "the 3-level V is lossier than the 4-level V (so the codebook is a real choice)");
    std::printf("  ratio 2-bit K / 4-bit K: %.2fx\n", r_k_lm / r_k4);
    check(r_k_lm < 8.0 * r_k4, "2-bit K's error stays under 8x the 4-bit error (the 2-bit cache is not collapsing)");
    check(r_k_lm < r_k_uni && r_v_lm4 < r_v_uni4, "the Lloyd-Max codebook beats the uniform one at the same 2 bits");
    std::printf("  Lloyd-Max gain over uniform at 2 bits: K %.2fx, V %.2fx\n", r_k_uni / r_k_lm, r_v_uni4 / r_v_lm4);
    // The two metrics split, and the split is the finding: per-channel is lossier per element than the per-token
    // block scale and better on the attention output. Both halves are pinned so neither can be quoted alone.
    check(r_k_kivi > r_k_lmt, "per-channel is the lossier per-element row (stated, not glossed)");
    std::printf("  per-channel vs per-token, per element: K %.2fx worse, V %.2fx worse "
                "(the attention table below is where it wins)\n", r_k_kivi / r_k_lmt, r_v_kivi / r_v_lmt4);

    // 3. the attention-output error, one head, over the whole cache
    std::vector<float> q(D);
    for (auto& x : q) x = nd(rng);
    std::vector<int> cells(N);
    for (int i = 0; i < N; ++i) cells[i] = i;
    std::vector<float> out_ref, out_lm3, out_lm4, out_q4, out_uni4, out_uni3, out_lm16, out_kivi, out_lmt,
                       out_k2v4, out_k4v2, out_k2v1c, out_k2v1t, out_k2c2t, out_k2t2c, out_k1v1;
    attention(q, Kref, Vref, D, cells, out_ref);
    attention(q, h_k2, h_v2_3, D, cells, out_lm3);       // device pools (per-channel), Lloyd-Max 3-level V
    attention(q, h_k2, h_v2_4, D, cells, out_lm4);       // device pools (per-channel), Lloyd-Max 4-level V
    attention(q, h_k4, h_vq4, D, cells, out_q4);
    attention(q, h_k_uni, h_v_uni4, D, cells, out_uni4);
    attention(q, h_k_uni, h_v_uni3, D, cells, out_uni3);
    attention(q, h_k_lm16, h_v_lm16, D, cells, out_lm16);
    attention(q, h_k_kivi, h_v_kivi, D, cells, out_kivi);   // the host per-channel arm: out_lm4's own number
    attention(q, h_k_lmt, h_v_lmt4, D, cells, out_lmt);     // the per-token layout, host-only
    // The asymmetric arms: same attention probe, K and V quantized at different widths / scale placements.
    attention(q, h_k_kivi, h_vq4, D, cells, out_k2v4);      // K 2-bit per-channel, V 4-bit
    attention(q, h_k4, h_v_kivi, D, cells, out_k4v2);       // K 4-bit, V 2-bit per-channel
    attention(q, h_k_kivi, h_v_1c, D, cells, out_k2v1c);    // K 2-bit, V 1-bit per-channel
    attention(q, h_k_kivi, h_v_1t, D, cells, out_k2v1t);    // K 2-bit, V 1-bit per-token (KIVI's split)
    attention(q, h_k_kivi, h_v_lmt4, D, cells, out_k2c2t);  // K per-channel 2-bit, V per-token 2-bit
    attention(q, h_k_lmt, h_v_kivi, D, cells, out_k2t2c);   // K per-token 2-bit, V per-channel 2-bit
    attention(q, h_k_1c, h_v_1c, D, cells, out_k1v1);       // 1-bit both, per-channel
    const double e_k2v4 = out_err(out_k2v4, out_ref, Vref), e_k4v2 = out_err(out_k4v2, out_ref, Vref);
    const double e_k2v1c = out_err(out_k2v1c, out_ref, Vref), e_k2v1t = out_err(out_k2v1t, out_ref, Vref);
    const double e_k2c2t = out_err(out_k2c2t, out_ref, Vref), e_k2t2c = out_err(out_k2t2c, out_ref, Vref);
    const double e_k1v1 = out_err(out_k1v1, out_ref, Vref);
    const double e_lm3 = out_err(out_lm3, out_ref, Vref), e_lm4 = out_err(out_lm4, out_ref, Vref);
    const double e_q4 = out_err(out_q4, out_ref, Vref);
    const double e_uni4 = out_err(out_uni4, out_ref, Vref), e_uni3 = out_err(out_uni3, out_ref, Vref);
    const double e_lm16 = out_err(out_lm16, out_ref, Vref);
    const double e_kivi = out_err(out_kivi, out_ref, Vref);
    const double e_lmt = out_err(out_lmt, out_ref, Vref);
    std::printf("\nAttention output against the FP16 cache, relative RMS:\n");
    std::printf("  %-26s %7.3f%%\n", "q4_0 (the format in use)", 100 * e_q4);
    std::printf("  %-26s %7.3f%%\n", "q2 uniform, V 4-level", 100 * e_uni4);
    std::printf("  %-26s %7.3f%%\n", "q2 uniform, V 3-level", 100 * e_uni3);
    std::printf("  %-26s %7.3f%%\n", "q2 Lloyd-Max, V 4-level", 100 * e_lm4);
    std::printf("  %-26s %7.3f%%\n", "q2 Lloyd-Max, V 3-level", 100 * e_lm3);
    std::printf("  %-26s %7.3f%%\n", "q2 Lloyd-Max, 16-value scale", 100 * e_lm16);
    std::printf("  %-26s %7.3f%%\n", "q2 Lloyd-Max, per-token (host)", 100 * e_lmt);
    std::printf("  %-26s %7.3f%%\n", "q2 Lloyd-Max, per-channel", 100 * e_kivi);
    std::printf("  (the V 4-level / V 3-level rows are the DEVICE pools read through the real gather kernel; the\n");
    std::printf("   per-channel row is the host reference for the same pools. THIS IS THE DECISION METRIC.)\n");
    // ---- Asymmetric K/V.  The published 2-bit schemes do not spend the same bits on K and V, so the arms differ
    // in WHICH tensor is coarse, not only in how many bits.  Byte cost per cell and KV head at head_dim 256:
    // q4_0 288, q2 per-channel 80, 1-bit per-channel 48, 1-bit per-token 34.
    const int B4 = (int) k::kv_q4_bytes_per_head(D), B2 = (int) k::kv_q2_bytes_per_head(D);
    const int B1c = D / 8 + 2 * D / 32, B1t = D / 8 + 2;
    std::printf("\n  asymmetric K/V arms (attention output; bytes per cell and KV head):\n");
    std::printf("  %-34s %7.3f%%  %4d B   K 2b per-channel, V 4-bit\n", "K2c + V4", 100 * e_k2v4, B2 + B4);
    std::printf("  %-34s %7.3f%%  %4d B   K 4-bit, V 2b per-channel\n", "K4 + V2c", 100 * e_k4v2, B4 + B2);
    std::printf("  %-34s %7.3f%%  %4d B   K 2b per-channel, V 1b per-channel\n", "K2c + V1c", 100 * e_k2v1c, B2 + B1c);
    std::printf("  %-34s %7.3f%%  %4d B   K 2b per-channel, V 1b per-token (KIVI's split)\n", "K2c + V1t", 100 * e_k2v1t, B2 + B1t);
    std::printf("  %-34s %7.3f%%  %4d B   K per-channel, V per-token, both 2b\n", "K2c + V2t", 100 * e_k2c2t, 2 * B2);
    std::printf("  %-34s %7.3f%%  %4d B   K per-token, V per-channel, both 2b\n", "K2t + V2c", 100 * e_k2t2c, 2 * B2);
    std::printf("  %-34s %7.3f%%  %4d B   both 1-bit per-channel\n", "K1c + V1c", 100 * e_k1v1, 2 * B1c);
    std::printf("  reference rows: symmetric q2c %7.3f%% (%4d B), q4_0 %7.3f%% (%4d B)\n",
                100 * e_lm4, 2 * B2, 100 * e_q4, 2 * B4);
    check(e_k2v4 > 0.0 && e_k1v1 > 0.0, "the asymmetric arms are live (the probe responds to K and V separately)");
    // Measured, and it reproduces KIVI's two placement findings (arXiv 2402.02750) on our rotated cache:
    // K wants a per-CHANNEL scale, V wants a per-TOKEN one.  The gain is real and it is small.
    check(e_k2c2t < e_lm4, "KIVI's split (K per-channel, V per-token, both 2b) beats symmetric per-channel (measured)");
    check(e_k2t2c > e_lm4, "a per-TOKEN K scale is the lossy side - K is the tensor that needs per-channel scaling");
    check(e_k2v1t < e_k2v1c, "1-bit V wants a per-token scale, not a per-channel one (KIVI's V finding)");
    check(e_k4v2 < e_k2v4, "at 2 bits the V side is the sensitive one: coarse V costs less than coarse K here");
    std::printf("  asymmetry: KIVI's split %.3f%% vs symmetric per-channel %.3f%% (%.2fx better, same 160 B); "
                "the best 224 B arm is %.3f%% against q4_0's %.3f%% - asymmetry narrows the gap, it does not close it\n",
                100 * e_k2c2t, 100 * e_lm4, e_lm4 / e_k2c2t, 100 * e_k4v2, 100 * e_q4);
    // The probe's own limit, stated where it can be seen: 1-bit on both tensors at 96 B measures only 13% worse
    // than 2-bit on both at 160 B.  Averaging the output over 1024 cells with a diffuse softmax washes coarse
    // quantization out; this table ranks schemes against each other, it is not an absolute quality bar.  The
    // absolute bar is the logit-level harness (bench/results/2026-10-12-kv-precision).
    std::printf("  probe limit: 1-bit both (%d B) measures %.3f%% against 2-bit both (%d B) at %.3f%% - a diffuse\n"
                "  softmax over %d cells averages coarse quantization away, so this table RANKS schemes; the\n"
                "  absolute bar is the logit-level KL/top-1 table.\n",
                2 * B1c, 100 * e_k1v1, 2 * B2, 100 * e_lm4, N);
    // On i.i.d. Gaussian data every channel's RMS is the same number, so per-channel scaling can only show that
    // it does no harm here.  The gain KIVI claims comes from outliers aligned to dimensions, which this probe's
    // data does not have - the number that decides it has to come from a real cache.
    check(e_kivi < 1.10 * e_lm4, "per-channel scaling does no harm on this data (its gain needs real cache statistics)");
    // The two arms of the scale-placement measurement, on the metric that decides it. Per-channel is 33% worse
    // per element and 2.8x better here; the reason is in the header of kv_q2.hpp.
    check(e_lm4 < e_lmt, "per-channel beats the per-token block scale on the attention output (the shipping choice)");
    std::printf("  per-channel over per-token, attention output: %.2fx better (%.3f%% against %.3f%%), while being\n"
                "  %.2fx WORSE per element - the metrics disagree on purpose, see docs/kv-q2.md\n",
                e_lmt / e_lm4, 100 * e_lm4, 100 * e_lmt, r_k_kivi / r_k_lmt);
    std::printf("  16-value scale gain over 32-value: K %.2fx, V %.2fx, attention %.2fx "
                "(12 B per 32 values, not 10)\n", r_k_lmt / r_k_lm16, r_v_lmt4 / r_v_lm16, e_lmt / e_lm16);
    // Measured, not assumed: a 16-value scale moves K and V by 2% and makes the attention output 1.87x WORSE
    // (37.2% against 19.9%).  A shorter block estimates the scale from 16 samples instead of 32, and that
    // estimator noise lands on every code in the block; nothing here pays for it.  The gate pins the finding so a
    // future change cannot quietly adopt a finer block because finer sounds better.  Pinned against both the
    // per-token arm it was measured against and the shipping per-channel pools.
    check(e_lm16 > e_lmt, "the 16-value scale does not beat 32 on the attention output (measured; the block stays 32)");
    check(e_lm16 > e_lm4, "the 16-value scale does not beat the shipping per-channel pools either");
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
