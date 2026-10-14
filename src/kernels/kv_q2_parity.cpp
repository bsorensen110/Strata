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
#include "strata/kernels/kv_stream.hpp"   // kKvQ2K, kv_stream_unit_blocks, kv_block_bytes (the q2k byte math)
#include "strata/kernels/qsa.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
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
/// A directional finding is a property of the DATA as much as of the scheme. Running this gate on a real
/// cache dump (STRATA_KV_PROBE_CACHE) REVERSED six of the rankings pinned below, so each one is pinned in
/// BOTH directions with its data source named. Quoting any of these numbers without saying which cache it
/// was measured on is exactly how this table becomes a lie.
void check_data(bool real_cache, bool synthetic_holds, bool real_holds, const char* synthetic_claim,
                const char* real_claim) {
    if (real_cache) check(real_holds, real_claim);
    else check(synthetic_holds, synthetic_claim);
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}

/// A REAL KV cache replaces the synthetic one when STRATA_KV_PROBE_CACHE names a file - the SDKV dump
/// `--dump-kv` writes (header: magic, version, n_layers, n_cells, n_head_kv, head_dim, layer ids, then per
/// layer K then V as fp16 in (cell, head, dim) order). That element order is the order these arrays use, so
/// a real cache drops straight in, and it is the STORED cache - rotated, post-RoPE, post-norm - which is
/// exactly what the quantizers are being asked about. STRATA_KV_PROBE_LAYER picks the layer.
///
/// This exists because the synthetic cache is structurally blind to per-channel scaling: i.i.d. Gaussian
/// noise gives every channel the same RMS, and KIVI's per-channel result (arXiv 2402.02750) exists to
/// exploit a handful of channels being far larger than the rest. Every per-channel number measured on the
/// synthetic cache is a floor, not the number.
bool load_real_cache(const char* path, int want_layer, int H, int D, int max_cells, int& n_out,
                     std::vector<float>& K, std::vector<float>& V, int& layer_out, std::string& err) {
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) { err = std::string("probe cache: cannot open ") + path; return false; }
    int hdr[6] = {0, 0, 0, 0, 0, 0};
    if (std::fread(hdr, 4, 6, f) != 6 || hdr[0] != 0x564B4453 || hdr[1] != 1) {
        std::fclose(f);
        err = "probe cache: not an SDKV v1 dump (write one with --dump-kv)";
        return false;
    }
    const int n_layers = hdr[2], n_cells = hdr[3], n_head_kv = hdr[4], head_dim = hdr[5];
    if (n_head_kv != H || head_dim != D) {
        std::fclose(f);
        err = "probe cache: " + std::to_string(n_head_kv) + " heads x " + std::to_string(head_dim) +
              " dims does not match this model's " + std::to_string(H) + " x " + std::to_string(D);
        return false;
    }
    std::vector<int> ids(n_layers, -1);
    if (n_layers <= 0 || n_cells <= 0 || std::fread(ids.data(), 4, n_layers, f) != (size_t) n_layers) {
        std::fclose(f);
        err = "probe cache: short layer id table";
        return false;
    }
    int slot = -1;
    for (int i = 0; i < n_layers; ++i)
        if (ids[i] == want_layer) slot = i;
    if (slot < 0) {
        std::fclose(f);
        err = "probe cache: layer " + std::to_string(want_layer) + " is not in the dump";
        return false;
    }
    const size_t plane = (size_t) n_cells * (size_t) H * (size_t) D;
    std::vector<uint16_t> buf(plane);
    for (int i = 0; i < slot; ++i)                    // skip the layers before the one we want
        if (std::fread(buf.data(), 2, plane, f) != plane || std::fread(buf.data(), 2, plane, f) != plane) {
            std::fclose(f);
            err = "probe cache: file is shorter than its header claims";
            return false;
        }
    std::vector<float> k((size_t) n_cells * H * D), v((size_t) n_cells * H * D);
    if (std::fread(buf.data(), 2, plane, f) != plane) { std::fclose(f); err = "probe cache: short K plane"; return false; }
    for (size_t i = 0; i < plane; ++i) k[i] = k::f32_from_f16(buf[i]);
    if (std::fread(buf.data(), 2, plane, f) != plane) { std::fclose(f); err = "probe cache: short V plane"; return false; }
    std::fclose(f);
    for (size_t i = 0; i < plane; ++i) v[i] = k::f32_from_f16(buf[i]);
    for (size_t i = 0; i < plane; ++i)
        if (!std::isfinite(k[i]) || !std::isfinite(v[i])) {
            err = "probe cache: a non-finite value in the dump";
            return false;
        }
    // Whole 32-cell groups only: the group is the per-channel scale's unit, so a tail that does not fill one
    // would be quantized half a scale at a time.
    const int usable = (n_cells / 32) * 32;
    n_out = usable < max_cells ? usable : max_cells;
    K.assign(k.begin(), k.begin() + (size_t) n_out * H * D);
    V.assign(v.begin(), v.begin() + (size_t) n_out * H * D);
    layer_out = want_layer;
    return true;
}

/// The statistics the synthetic cache cannot have, printed so the reader of the table knows what the data is.
/// Per-channel scaling exists to exploit the first ratio (a channel's RMS varying across the cache) and the
/// second (a few channels carrying most of the variance); if both are flat, per-channel scaling has nothing
/// to exploit and the per-token scale is the honest choice.
void channel_stats(const std::vector<float>& X, int n_cells, int H, int D, const char* name) {
    const int nd = H * D;
    std::vector<double> chan((size_t) nd, 0.0), cell((size_t) n_cells, 0.0);
    for (int c = 0; c < n_cells; ++c)
        for (int i = 0; i < nd; ++i) {
            const double x = X[(size_t) c * nd + i];
            chan[i] += x * x;
            cell[c] += x * x;
        }
    std::vector<double> cv((size_t) nd);
    double total = 0.0;
    for (int i = 0; i < nd; ++i) { chan[i] = std::sqrt(chan[i] / n_cells); cv[i] = chan[i] * chan[i]; total += cv[i]; }
    for (int c = 0; c < n_cells; ++c) cell[c] = std::sqrt(cell[c] / nd);
    std::sort(cv.begin(), cv.end());
    std::sort(cell.begin(), cell.end());
    double top8 = 0.0;
    for (size_t i = cv.size() - 8; i < cv.size(); ++i) top8 += cv[i];
    const double cmed = cv[cv.size() / 2], cmax = cv.back(), cmin = cv.front();
    const double lmed = cell[cell.size() / 2], lmax = cell.back(), lmin = cell.front();
    std::printf("  %s: channel RMS min %.4f median %.4f max %.4f (max/median %.2fx), top-8 channels carry %.1f%% "
                "of the variance; cell RMS min %.4f median %.4f max %.4f (max/median %.2fx)\n",
                name, cmin, cmed, cmax, cmed > 0.0 ? cmax / cmed : 0.0, total > 0.0 ? 100.0 * top8 / total : 0.0,
                lmin, lmed, lmax, lmed > 0.0 ? lmax / lmed : 0.0);
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

/// KIVI-style ASYMMETRIC quantization (arXiv 2402.02750): scale AND A ZERO-POINT from the group's extremes,
/// `z = min`, `s = (max - min) / 3`, levels z, z+s, z+2s, z+3s. SKVQ (arXiv 2405.06219) clips those extremes
/// first; `clip_sigma > 0` does the same (mean +- clip_sigma * sd). `per_channel` groups 32 cells of one
/// dimension, `per_token` groups one cell's head_dim.
///
/// This is the scheme both published 2-bit methods use and ours does not: we store a symmetric RMS scale with
/// no zero-point. It costs two fp16 per group (scale + zero-point) - 32 B per cell and head per-channel,
/// 4 B per cell and head per-token - so 96 B per cell and head for a per-channel 2-bit K, against 80 B for
/// the symmetric block.
void quant_asym(const float* x, int n_cells, int n_heads, int D, bool per_channel, float clip_sigma, float* out) {
    const int stride = n_heads * D;
    auto qz = [&](const float* v, int n, float* o) {
        float lo = v[0], hi = v[0], ss = 0.f, s1 = 0.f;
        for (int i = 0; i < n; ++i) {
            if (v[i] < lo) lo = v[i];
            if (v[i] > hi) hi = v[i];
            ss += v[i] * v[i];
            s1 += v[i];
        }
        const float mean = s1 / (float) n;
        const float sd = std::sqrt(std::max(0.0f, ss / (float) n - mean * mean));
        if (clip_sigma > 0.0f) {
            lo = std::max(lo, mean - clip_sigma * sd);
            hi = std::min(hi, mean + clip_sigma * sd);
        }
        float s = (hi - lo) / 3.0f;
        if (!(s > 0.0f)) s = 1e-8f;
        const float ds = k::f32_from_f16(k::f16_from_f32(s));       // stored as the device would store it
        const float dz = k::f32_from_f16(k::f16_from_f32(lo));
        for (int i = 0; i < n; ++i) {
            float q = std::nearbyintf((v[i] - dz) / ds);
            q = q < 0.f ? 0.f : (q > 3.f ? 3.f : q);
            o[i] = dz + q * ds;
        }
    };
    std::vector<float> tmp(512), tmpo(512);
    if (per_channel) {
        for (int h = 0; h < n_heads; ++h)
            for (int d = 0; d < D; ++d)
                for (int g = 0; g < n_cells; g += 32) {
                    for (int i = 0; i < 32; ++i) tmp[i] = x[(size_t) (g + i) * stride + h * D + d];
                    qz(tmp.data(), 32, tmpo.data());
                    for (int i = 0; i < 32; ++i) out[(size_t) (g + i) * stride + h * D + d] = tmpo[i];
                }
    } else {
        for (int c = 0; c < n_cells; ++c)
            for (int h = 0; h < n_heads; ++h) {
                const size_t base = (size_t) c * stride + h * D;
                qz(x + base, D, out + base);
            }
    }
}

/// CHANNEL REORDERING, SKVQ's first mechanism (arXiv 2405.06219): sort a tensor's channels by RMS so that a
/// per-token group of 32 CONSECUTIVE channels is homogeneous - one scale per group then fits all 32 channels
/// instead of being set by the group's outlier. The permutation is per KV head (channel d of head h reads the
/// old channel perm[h*D+d]), computed from the cache being tested - at runtime it would be fixed offline from
/// calibration data and folded into the query (K) and the output projection's columns (V), which is why it
/// costs nothing at attention time.
void channel_order(const std::vector<float>& X, int n_cells, int n_heads, int D, std::vector<int>& perm) {
    const int stride = n_heads * D;
    perm.assign((size_t) n_heads * D, 0);
    std::vector<float> rms(D);
    for (int h = 0; h < n_heads; ++h) {
        for (int d = 0; d < D; ++d) {
            double ss = 0.0;
            for (int c = 0; c < n_cells; ++c) {
                const float v = X[(size_t) c * stride + h * D + d];
                ss += (double) v * v;
            }
            rms[d] = (float) std::sqrt(ss / (double) n_cells);
        }
        std::vector<int> idx(D);
        for (int d = 0; d < D; ++d) idx[d] = d;
        std::stable_sort(idx.begin(), idx.end(),
                         [&](int a, int b) { return rms[a] < rms[b]; });
        for (int d = 0; d < D; ++d) perm[(size_t) h * D + d] = idx[d];
    }
}

/// Apply (or undo, with the inverse permutation) a channel reorder: new channel d of head h is old channel
/// perm[h*D+d], every cell.
void apply_channel_order(const std::vector<float>& X, const std::vector<int>& perm, int n_cells,
                         int n_heads, int D, std::vector<float>& out) {
    const int stride = n_heads * D;
    out.assign(X.size(), 0.f);
    for (int h = 0; h < n_heads; ++h)
        for (int d = 0; d < D; ++d) {
            const int src = perm[(size_t) h * D + d];
            for (int c = 0; c < n_cells; ++c)
                out[(size_t) c * stride + h * D + d] = X[(size_t) c * stride + h * D + src];
        }
}

/// GENERIC Lloyd-Max on one 32-value group, any level count - the fractional-bit arms (2.5-bit K = 6 levels
/// at ~2.58 bits of codebook, 1.5-bit V = 3 levels) need codebooks the device kernels do not have yet, and
/// the gate exists to measure BEFORE the kernels are written. Uniform init on +-1 RMS, 12 iterations, which
/// converges for L <= 8 on 32 Gaussian samples. The write-back rounds to fp16 like every stored value here.
void lloyd_max_group(const float* x, int n, int L, float* out) {
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    const float s = (float) std::sqrt(ss / (double) n);
    std::vector<float> cb(L);
    for (int j = 0; j < L; ++j)
        cb[j] = (L == 1) ? 0.f : -s + (2.f * s * (float) j) / (float) (L - 1);
    for (int it = 0; it < 12; ++it) {
        std::vector<float> sum(L, 0.f);
        std::vector<int> cnt(L, 0);
        for (int i = 0; i < n; ++i) {
            int best = 0;
            float bd = 1e30f;
            for (int j = 0; j < L; ++j) {
                const float d = std::fabs(x[i] - cb[j]);
                if (d < bd) { bd = d; best = j; }
            }
            sum[best] += x[i];
            cnt[best]++;
        }
        for (int j = 0; j < L; ++j)
            if (cnt[j]) cb[j] = sum[j] / (float) cnt[j];
    }
    for (int i = 0; i < n; ++i) {
        int best = 0;
        float bd = 1e30f;
        for (int j = 0; j < L; ++j) {
            const float d = std::fabs(x[i] - cb[j]);
            if (d < bd) { bd = d; best = j; }
        }
        out[i] = k::f32_from_f16(k::f16_from_f32(cb[best]));
    }
}

/// Per-channel L-level Lloyd-Max (the quant_kivi shape, arbitrary codebook size): one scale per
/// (dimension, 32-cell group). Host-only - this is the measurement that decides whether a 2.5-bit K
/// format exists, before any kernel is written for it.
void quant_kivi_l(const float* x, int n_cells, int n_heads, int D, int levels, float* out) {
    const int stride = n_heads * D;
    std::vector<float> tmp(32), tmpo(32);
    for (int h = 0; h < n_heads; ++h)
        for (int d = 0; d < D; ++d)
            for (int g = 0; g < n_cells; g += 32) {
                for (int i = 0; i < 32; ++i) tmp[i] = x[(size_t) (g + i) * stride + h * D + d];
                lloyd_max_group(tmp.data(), 32, levels, tmpo.data());
                for (int i = 0; i < 32; ++i) out[(size_t) (g + i) * stride + h * D + d] = tmpo[i];
            }
}

/// MIXED-LEVEL per channel, the 2.5-bit average: channels sorted by RMS (the SKVQ permutation's own order),
/// the loud half at 6 levels (~2.58 bits of codebook), the quiet half at 4 (2 bits). Codes store at 3 bits
/// for the loud half and 2 for the quiet half, so the AVERAGE is 2.5 bits of code - 80 B of code bytes per
/// cell and head at D=256, the same 16 B of scales, plus a per-channel level bitmap (D/8 = 32 B per cache,
/// negligible per cell). SKVQ's mixed precision along channels, applied to K.
void quant_kivi_mixed(const float* x, int n_cells, int n_heads, int D, float* out) {
    const int stride = n_heads * D;
    std::vector<float> rms((size_t) n_heads * D, 0.f);
    for (int h = 0; h < n_heads; ++h)
        for (int d = 0; d < D; ++d) {
            double ss = 0.0;
            for (int c = 0; c < n_cells; ++c) {
                const float v = x[(size_t) c * stride + h * D + d];
                ss += (double) v * v;
            }
            rms[(size_t) h * D + d] = (float) std::sqrt(ss / (double) n_cells);
        }
    std::vector<float> sorted(rms);
    std::sort(sorted.begin(), sorted.end());
    const float med = sorted[sorted.size() / 2];
    std::vector<float> tmp(32), tmpo(32);
    for (int h = 0; h < n_heads; ++h)
        for (int d = 0; d < D; ++d) {
            const int L = rms[(size_t) h * D + d] >= med ? 6 : 4;
            for (int g = 0; g < n_cells; g += 32) {
                for (int i = 0; i < 32; ++i) tmp[i] = x[(size_t) (g + i) * stride + h * D + d];
                lloyd_max_group(tmp.data(), 32, L, tmpo.data());
                for (int i = 0; i < 32; ++i) out[(size_t) (g + i) * stride + h * D + d] = tmpo[i];
            }
        }
}

/// TOP-FRACTION bit allocation for K - the lever the real-cache dump points at: K's channel RMS spans
/// 5.8x (L3) to 74.9x (L47), so the LOUD channels are where resolution is worth anything. The loudest
/// top_frac of each KV head's channels get `loud_levels` (up to 16), the rest stay at 4 - per head, because
/// the format's block is per head and that is the split the device's freeze can and does make. Unlike the 2.5-bit mixed arm
/// (6/4 levels, 3-bit codes), loud channels can take 4-bit codes - 16 levels for the outliers that were
/// eating the whole per-group scale. Average bits = 2 + top_frac * (bits(loud) - 2); bytes per cell and
/// head = 32 * avg_bits + 16 (D=256). Host-only, measurement before kernels.
void quant_kivi_topq(const float* x, int n_cells, int n_heads, int D, float top_frac, int loud_levels,
                     float* out) {
    const int stride = n_heads * D;
    std::vector<float> rms((size_t) n_heads * D, 0.f);
    for (int h = 0; h < n_heads; ++h)
        for (int d = 0; d < D; ++d) {
            double ss = 0.0;
            for (int c = 0; c < n_cells; ++c) {
                const float v = x[(size_t) c * stride + h * D + d];
                ss += (double) v * v;
            }
            rms[(size_t) h * D + d] = (float) std::sqrt(ss / (double) n_cells);
        }
    std::vector<float> tmp(32), tmpo(32);
    const int nloud = (int) std::lround(top_frac * (double) D);
    for (int h = 0; h < n_heads; ++h) {
        // PER HEAD, and this is the whole point of the line: the loud set is the top nloud channels OF THIS
        // HEAD, so the threshold is this head's own sorted RMS. Sorting all n_heads * D RMS values into one
        // array and taking `sorted[D - nloud]` is a global threshold read as a per-head one - it marks
        // ~n_heads * nloud channels spread over the heads, not nloud in each, and the device's freeze (which
        // ranks per head, as the format's block is per head) can never land on it.
        std::vector<float> sorted(rms.begin() + (size_t) h * D, rms.begin() + (size_t) (h + 1) * D);
        std::sort(sorted.begin(), sorted.end());
        const float thr = nloud > 0 ? sorted[(size_t) D - nloud] : 1e30f;   // smallest RMS still in this head's loud set
        for (int d = 0; d < D; ++d) {
            const int L = rms[(size_t) h * D + d] >= thr ? loud_levels : 4;
            for (int g = 0; g < n_cells; g += 32) {
                for (int i = 0; i < 32; ++i) tmp[i] = x[(size_t) (g + i) * stride + h * D + d];
                lloyd_max_group(tmp.data(), 32, L, tmpo.data());
                for (int i = 0; i < 32; ++i) out[(size_t) (g + i) * stride + h * D + d] = tmpo[i];
            }
        }
    }
}

/// The FIXED-GRID twin of quant_kivi_topq - the honest device reference. The per-group fit above cannot be
/// what a 16-level format ships: 16 centroids fitted to a group's 32 SAMPLES is degenerate (the centroids
/// land on the data, roughly halving the error against any fixed grid), so the per-group row is an encoder
/// ceiling, not a format. This quantizes against the SAME constants the device kernels call - the header's
/// q2_code_lm16/q2_lm16_level for loud channels, q2_code_lm4 for quiet, the scale fp16-rounded - so the
/// device arm and this row measure the same object and the within-2% pin means something.
void quant_kivi_topq_fixed(const float* x, int n_cells, int n_heads, int D, float top_frac, float* out) {
    const int stride = n_heads * D;
    std::vector<float> rms((size_t) n_heads * D, 0.f);
    for (int h = 0; h < n_heads; ++h)
        for (int d = 0; d < D; ++d) {
            double ss = 0.0;
            for (int c = 0; c < n_cells; ++c) {
                const float v = x[(size_t) c * stride + h * D + d];
                ss += (double) v * v;
            }
            rms[(size_t) h * D + d] = (float) std::sqrt(ss / (double) n_cells);
        }
    const int nloud = (int) std::lround(top_frac * (double) D);
    for (int h = 0; h < n_heads; ++h) {
        std::vector<float> sorted(rms.begin() + (size_t) h * D, rms.begin() + (size_t) (h + 1) * D);
        std::sort(sorted.begin(), sorted.end());
        const float thr = nloud > 0 ? sorted[(size_t) D - nloud] : 1e30f;
        for (int d = 0; d < D; ++d) {
            const bool loud = rms[(size_t) h * D + d] >= thr;
            for (int g = 0; g < n_cells; g += 32) {
                double ss = 0.0;
                for (int i = 0; i < 32; ++i) {
                    const float v = x[(size_t) (g + i) * stride + h * D + d];
                    ss += (double) v * v;
                }
                const float s = (float) std::sqrt(ss / 32.0);
                const float ds = k::f32_from_f16(k::f16_from_f32(s));   // the scale as the slot stores it
                for (int i = 0; i < 32; ++i) {
                    const float v = x[(size_t) (g + i) * stride + h * D + d];
                    const int code = loud ? k::q2_code_lm16(v, s) : k::q2_code_lm4(v, s);
                    const float lv = loud ? k::q2_lm16_level(code) : k::q2_lm4_level(code);
                    out[(size_t) (g + i) * stride + h * D + d] = lv * ds;
                }
            }
        }
    }
}

/// A FIXED Lloyd-Max codebook for L levels, fitted once from 1M Gaussian samples and shared by every group -
/// the shippable form of an L-level channel (a per-group fit is an encoder ceiling when L approaches the
/// sample count). Self-seeded and deterministic; runs to convergence (last move < 1e-5). cb ascending,
/// thr the L-1 decision boundaries.
void fit_lm_grid(int L, std::vector<float>& cb, std::vector<float>& thr) {
    std::mt19937 rng(20260713u);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> samp((size_t) 1 << 20);
    for (auto& x : samp) x = nd(rng);
    std::sort(samp.begin(), samp.end());
    cb.assign(L, 0.f);
    for (int j = 0; j < L; ++j) cb[j] = -4.f + 8.f * (float) j / (float) (L - 1);
    for (int it = 0; it < 500; ++it) {
        double sum[64] = {0};
        int cnt[64] = {0};
        int j = 0;
        for (const float x : samp) {
            while (j + 1 < L && x > 0.5f * (cb[j] + cb[j + 1])) ++j;
            sum[j] += (double) x;
            cnt[j]++;
        }
        float delta = 0.f;
        for (int m = 0; m < L; ++m) {
            if (!cnt[m]) continue;
            const float n = (float) (sum[m] / (double) cnt[m]);
            delta = std::max(delta, std::fabs(n - cb[m]));
            cb[m] = n;
        }
        if (delta < 1e-5f) break;
    }
    thr.assign(L - 1, 0.f);
    for (int j = 0; j + 1 < L; ++j) thr[j] = 0.5f * (cb[j] + cb[j + 1]);
}

/// Per-channel FIXED-codebook quantization: every (channel, 32-cell group) quantizes against the same L-level
/// grid scaled by the group's fp16-rounded RMS - the form a device kernel burns in. cb ascending, thr the
/// L-1 boundaries between consecutive levels.
void quant_per_channel_grid(const float* x, int n_cells, int n_heads, int D,
                            const std::vector<float>& cb, const std::vector<float>& thr, float* out) {
    const int stride = n_heads * D;
    const int nthr = (int) thr.size();
    for (int h = 0; h < n_heads; ++h)
        for (int d = 0; d < D; ++d)
            for (int g = 0; g < n_cells; g += 32) {
                double ss = 0.0;
                for (int i = 0; i < 32; ++i) {
                    const float v = x[(size_t) (g + i) * stride + h * D + d];
                    ss += (double) v * v;
                }
                const float s = (float) std::sqrt(ss / 32.0);
                const float ds = k::f32_from_f16(k::f16_from_f32(s));
                const float inv = s > 0.f ? 1.0f / s : 0.f;
                for (int i = 0; i < 32; ++i) {
                    const float a = x[(size_t) (g + i) * stride + h * D + d] * inv;
                    int code = 0;
                    while (code < nthr && a >= thr[code]) ++code;
                    out[(size_t) (g + i) * stride + h * D + d] = cb[code] * ds;
                }
            }
}

/// The top-q K arm against FIXED grids: the loudest top_frac of channels per head quantize against loud_cb
/// (loud_thr), the rest against the quiet grid - the shippable twin of quant_kivi_topq, whose per-group fit
/// is an encoder ceiling. Both grids scale by the group's fp16-rounded RMS.
void quant_topq_grid(const float* x, int n_cells, int n_heads, int D, float top_frac,
                     const std::vector<float>& loud_cb, const std::vector<float>& loud_thr, float* out) {
    const int stride = n_heads * D;
    std::vector<float> rms((size_t) n_heads * D, 0.f);
    for (int h = 0; h < n_heads; ++h)
        for (int d = 0; d < D; ++d) {
            double ss = 0.0;
            for (int c = 0; c < n_cells; ++c) {
                const float v = x[(size_t) c * stride + h * D + d];
                ss += (double) v * v;
            }
            rms[(size_t) h * D + d] = (float) std::sqrt(ss / (double) n_cells);
        }
    const int nloud = (int) std::lround(top_frac * (double) D);
    std::vector<float> tmpo(32);
    for (int h = 0; h < n_heads; ++h) {
        std::vector<float> sorted(rms.begin() + (size_t) h * D, rms.begin() + (size_t) (h + 1) * D);
        std::sort(sorted.begin(), sorted.end());
        const float thr = nloud > 0 ? sorted[(size_t) D - nloud] : 1e30f;
        for (int d = 0; d < D; ++d) {
            const bool loud = rms[(size_t) h * D + d] >= thr;
            for (int g = 0; g < n_cells; g += 32) {
                double ss = 0.0;
                for (int i = 0; i < 32; ++i) {
                    const float v = x[(size_t) (g + i) * stride + h * D + d];
                    ss += (double) v * v;
                }
                const float s = (float) std::sqrt(ss / 32.0);
                const float ds = k::f32_from_f16(k::f16_from_f32(s));
                const float inv = s > 0.f ? 1.0f / s : 0.f;
                for (int i = 0; i < 32; ++i) {
                    const float a = x[(size_t) (g + i) * stride + h * D + d] * inv;
                    if (loud) {
                        int code = 0;
                        while (code < (int) loud_thr.size() && a >= loud_thr[code]) ++code;
                        out[(size_t) (g + i) * stride + h * D + d] = loud_cb[code] * ds;
                    } else {
                        tmpo[i] = 0.f;   // quiet path below uses the device lm4 via the existing helper
                    }
                }
                if (!loud) {
                    const float* src = x + (size_t) g * stride + h * D + d;
                    for (int i = 0; i < 32; ++i) {
                        const float v = src[(size_t) i * stride];
                        const int code = k::q2_code_lm4(v, s);
                        out[(size_t) (g + i) * stride + h * D + d] = k::q2_lm4_level(code) * ds;
                    }
                }
            }
        }
    }
}

/// The level bitmap quant_kivi_topq's rule IMPLIES, in the layout the device stores it in: per (KV head,
/// dimension), bit set = loud, the loud set being the top `nloud` channels by RMS over `n_cells` cells with the
/// threshold `sorted[D - nloud]` - the same ranking, the same `>=`, the same per-head split. The device's
/// freeze kernel computes the same set from the append's cells by counting, for each channel, how many are
/// louder; with continuous data the two are the same set, and the gate pins that bit for bit.
std::vector<uint32_t> host_topq_bits(const std::vector<float>& x, int n_cells, int n_heads, int D, int nloud) {
    const int stride = n_heads * D;
    std::vector<float> rms((size_t) n_heads * D, 0.f);
    for (int h = 0; h < n_heads; ++h)
        for (int d = 0; d < D; ++d) {
            double ss = 0.0;
            for (int c = 0; c < n_cells; ++c) {
                const float v = x[(size_t) c * stride + h * D + d];
                ss += (double) v * v;
            }
            rms[(size_t) h * D + d] = (float) std::sqrt(ss / (double) n_cells);
        }
    const int words = (D + 31) / 32;
    std::vector<uint32_t> bits((size_t) n_heads * words, 0u);
    for (int h = 0; h < n_heads; ++h) {
        // Per head, exactly as quant_kivi_topq and as the device's freeze: the loud set is this head's top
        // nloud channels. A threshold taken from a sort of ALL heads' RMS is a global cut read as a per-head
        // one, and it selects a different set than the format's per-head block can ever produce.
        std::vector<float> sorted(rms.begin() + (size_t) h * D, rms.begin() + (size_t) (h + 1) * D);
        std::sort(sorted.begin(), sorted.end());
        const float thr = nloud > 0 ? sorted[(size_t) D - nloud] : 1e30f;
        for (int d = 0; d < D; ++d)
            if (rms[(size_t) h * D + d] >= thr) bits[(size_t) h * words + (d >> 5)] |= (1u << (d & 31));
    }
    return bits;
}

/// ASYMMETRIC min/max with a zero-point, arbitrary level count - quant_asym's grid with L levels instead of
/// 4. This is the user's second ask measured on its own: zero-point on the K side at 2.5 bits (6 uniform
/// levels over [min, max]) against symmetric 6-level Lloyd-Max at the same codebook size.
void quant_asym_l(const float* x, int n_cells, int n_heads, int D, int levels, bool per_channel, float* out) {
    const int stride = n_heads * D;
    const float hi_code = (float) (levels - 1);
    auto qz = [&](const float* v, int n, float* o) {
        float lo = v[0], hi = v[0];
        for (int i = 0; i < n; ++i) {
            if (v[i] < lo) lo = v[i];
            if (v[i] > hi) hi = v[i];
        }
        float s = (hi - lo) / hi_code;
        if (!(s > 0.0f)) s = 1e-8f;
        const float ds = k::f32_from_f16(k::f16_from_f32(s));
        const float dz = k::f32_from_f16(k::f16_from_f32(lo));
        for (int i = 0; i < n; ++i) {
            float q = std::nearbyintf((v[i] - dz) / ds);
            q = q < 0.f ? 0.f : (q > hi_code ? hi_code : q);
            o[i] = dz + q * ds;
        }
    };
    std::vector<float> tmp(32), tmpo(32);
    if (per_channel) {
        for (int h = 0; h < n_heads; ++h)
            for (int d = 0; d < D; ++d)
                for (int g = 0; g < n_cells; g += 32) {
                    for (int i = 0; i < 32; ++i) tmp[i] = x[(size_t) (g + i) * stride + h * D + d];
                    qz(tmp.data(), 32, tmpo.data());
                    for (int i = 0; i < 32; ++i) out[(size_t) (g + i) * stride + h * D + d] = tmpo[i];
                }
    } else {
        for (int c = 0; c < n_cells; ++c)
            for (int h = 0; h < n_heads; ++h) {
                const size_t base = (size_t) c * stride + h * D;
                qz(x + base, D, out + base);
            }
    }
}

/// JOINT per-channel x per-token scaling (the Coupled-Quantization / SKVQ direction): a value's scale is the
/// PRODUCT of a channel scale and a token scale, so one 2-bit code carries both structures at once instead of
/// choosing one axis. Costs the per-channel fp16 (16 B per cell and head) plus one fp16 per cell and head
/// (2 B): 82 B per cell and head at 2 bits, against 80 B for the single-axis per-channel block.
void quant_joint(const float* x, int n_cells, int n_heads, int D, float* out) {
    const int stride = n_heads * D;
    const int ngroups = n_cells / 32;
    std::vector<float> a((size_t) ngroups * stride, 1.0f), b((size_t) n_cells * n_heads, 1.0f);
    for (int h = 0; h < n_heads; ++h)
        for (int d = 0; d < D; ++d)
            for (int g = 0; g < ngroups; ++g) {
                float ss = 0.f;
                for (int i = 0; i < 32; ++i) {
                    const float v = x[(size_t) (g * 32 + i) * stride + h * D + d];
                    ss += v * v;
                }
                a[(size_t) g * stride + h * D + d] = std::sqrt(ss / 32.0f);
            }
    // The token scale is measured with the channel scale divided out, then normalized to mean 1 so the two
    // factors cannot both claim the same variance.
    std::vector<float> tr((size_t) n_cells * n_heads, 0.f);
    double acc = 0.0;
    for (int c = 0; c < n_cells; ++c)
        for (int h = 0; h < n_heads; ++h) {
            float ss = 0.f;
            for (int d = 0; d < D; ++d) {
                const size_t idx = (size_t) c * stride + h * D + d;
                const float g = a[(size_t) (c / 32) * stride + h * D + d];
                const float r = g > 0.f ? x[idx] / g : 0.f;
                ss += r * r;
            }
            tr[(size_t) c * n_heads + h] = std::sqrt(ss / (float) D);
            acc += tr[(size_t) c * n_heads + h];
        }
    const double mean = acc / ((double) n_cells * n_heads);
    for (size_t i = 0; i < tr.size(); ++i) b[i] = mean > 0.0 ? (float) (tr[i] / mean) : 1.0f;
    for (int c = 0; c < n_cells; ++c)
        for (int h = 0; h < n_heads; ++h) {
            const float bs = k::f32_from_f16(k::f16_from_f32(b[(size_t) c * n_heads + h]));
            for (int d = 0; d < D; ++d) {
                const size_t idx = (size_t) c * stride + h * D + d;
                const float as = k::f32_from_f16(k::f16_from_f32(a[(size_t) (c / 32) * stride + h * D + d]));
                const float s = as * bs;
                out[idx] = s > 0.f ? k::q2_value_lm4(k::q2_code_lm4(x[idx], s), s) : 0.f;
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
    int N = 1024;                                        // cells in the cache (the loader may set a smaller one)
    int pages = 0, rows = 0;                             // derived from N below, once N is final

    std::mt19937 rng(23);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> K, V;
    bool real_cache = false;
    int probe_layer = -1;                                 // the layer the real cache came from (-1 = synthetic)
    if (const char* e = std::getenv("STRATA_KV_PROBE_CACHE")) {
        int want = 0, layer_out = 0;
        if (const char* l = std::getenv("STRATA_KV_PROBE_LAYER")) want = std::atoi(l);
        std::string cerr;
        if (!load_real_cache(e, want, H, D, 1024, N, K, V, layer_out, cerr)) {
            std::fprintf(stderr, "strata kv_q2_parity: %s\n", cerr.c_str());
            return 2;
        }
        real_cache = true;
        probe_layer = layer_out;
        std::printf("REAL CACHE (stored, rotated): %s layer %d - %d cells x %d kv heads x %d dims\n",
                    e, layer_out, N, H, D);
        channel_stats(K, N, H, D, "K");
        channel_stats(V, N, H, D, "V");
    }
    pages = N / P;
    rows = N * H;                                        // identity layout: one row per (cell, kv head)
    if (!real_cache) {
        K.assign((size_t) N * H * D, 0.f);
        V.assign((size_t) N * H * D, 0.f);
        // Mild per-cell magnitude spread on purpose: a cache where every cell has the same scale flatters every
        // quantizer, because the softmax weights barely move. 25x spikes were tried first and made the probe
        // useless - they hand the whole softmax to a handful of cells, so flipping which one wins (which 4 bits
        // can do) moves the entire output, and q4_0 measured worse than a 2-bit cache. 3x is the regime a real
        // rotated cache has.
        for (int cell = 0; cell < N; ++cell) {
            const float sc = (cell % 17 == 0) ? 3.0f : (cell % 23 == 0 ? 0.1f : 1.0f);
            for (int i = 0; i < H * D; ++i) {
                K[(size_t) cell * H * D + i] = nd(rng) * sc;
                V[(size_t) cell * H * D + i] = nd(rng) * sc;
            }
        }
        std::printf("SYNTHETIC CACHE: i.i.d. Gaussian, 3x per-cell spread - blind to channel structure, so "
                    "every per-channel number here is a floor\n");
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

    // ---- THE q2k DEVICE ARM (`--kv q2k`): K filled by the q2 group kernels, V by the q4 kernels, each half
    // read back by its own gather kernel. These are the engine's own calls, in the shape the engine makes them:
    // the q2 append called with its pool and tensor twice (its K plane only - the fold layer.cpp/verify.cpp use),
    // the q4 append called the same way for V. Nothing here is composed on the host: the bytes are the pools a
    // q2k state carves (K at 80 B per cell and head, V at 144) and the numbers are what its readers return.
    uint8_t* kq2_dev = dalloc<uint8_t>(q2_bytes);
    uint8_t* vq4_dev = dalloc<uint8_t>(q4_bytes);
    uint16_t* k_scr_2k = dalloc<uint16_t>((size_t) rows * D);
    uint16_t* v_scr_2k = dalloc<uint16_t>((size_t) rows * D);
    k::kv_append_q2(kq2_dev, kq2_dev, d_table, 0, N, dK, dK, s, nullptr, nullptr, nullptr, /*v3level*/ false);
    k::kv_append_q4(vq4_dev, vq4_dev, d_table, 0, N, dV, dV, s, nullptr, nullptr);
    ck(cudaDeviceSynchronize(), "q2k append");
    k::kv_gather_q2_step(kq2_dev, kq2_dev, d_table, d_ids, d_step, N, s, k_scr_2k, k_scr_2k, nullptr,
                         /*v3level*/ false);
    k::kv_gather_q4_step(vq4_dev, vq4_dev, d_table, d_ids, d_step, N, s, v_scr_2k, v_scr_2k, nullptr);
    ck(cudaDeviceSynchronize(), "q2k gather");
    std::vector<uint16_t> w_k2k(rows * D), w_v2k(rows * D);
    ck(cudaMemcpy(w_k2k.data(), k_scr_2k, w_k2k.size() * 2, cudaMemcpyDeviceToHost), "k q2k");
    ck(cudaMemcpy(w_v2k.data(), v_scr_2k, w_v2k.size() * 2, cudaMemcpyDeviceToHost), "v q2k");
    std::vector<float> h_k2k(rows * D), h_v2k(rows * D);
    for (size_t i = 0; i < w_k2k.size(); ++i) h_k2k[i] = k::f32_from_f16(w_k2k[i]);
    for (size_t i = 0; i < w_v2k.size(); ++i) h_v2k[i] = k::f32_from_f16(w_v2k[i]);
    // The mixed format must not move either half's bytes: K is bit for bit what the q2 cache holds, V bit for
    // bit what the q4_0 cache holds. A mixed dispatch that re-derived either side would fail here, and only here.
    bool k2k_is_q2 = true, v2k_is_q4 = true;
    for (size_t i = 0; i < w_k2k.size(); ++i) {
        if (w_k2k[i] != h_k[i]) k2k_is_q2 = false;
        if (w_v2k[i] != w_v4[i]) v2k_is_q4 = false;
    }
    check(k2k_is_q2, "q2k's K is bit for bit the q2 cache's K (the mixed dispatch re-quantizes nothing)");
    check(v2k_is_q4, "q2k's V is bit for bit the q4_0 cache's V (the mixed dispatch re-quantizes nothing)");
    const double r_k2k = rel_rms(h_k2k, Kref), r_v2k = rel_rms(h_v2k, Vref);

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
    // ---- The published schemes' QUANTIZER, not ours: asymmetric min/max with a zero-point (KIVI,
    // arXiv 2402.02750), with SKVQ's clipping (arXiv 2405.06219), and the joint per-channel x per-token scale.
    // Same 2 bits, a different scale geometry - this is the arm the "near lossless" claim rests on.
    std::vector<float> h_k_ac(Kref.size()), h_v_ac(Vref.size()), h_v_at(Vref.size()), h_k_ac3(Kref.size()),
                       h_k_j(Kref.size()), h_v_j(Vref.size());
    quant_asym(Kref.data(), N, H, D, true, 0.0f, h_k_ac.data());    // KIVI: per-channel, z = min
    quant_asym(Kref.data(), N, H, D, true, 3.0f, h_k_ac3.data());   // the same, SKVQ-clipped
    quant_asym(Vref.data(), N, H, D, true, 0.0f, h_v_ac.data());
    quant_asym(Vref.data(), N, H, D, false, 0.0f, h_v_at.data());   // KIVI's V axis: per-token
    quant_joint(Kref.data(), N, H, D, h_k_j.data());
    quant_joint(Vref.data(), N, H, D, h_v_j.data());
    const double r_k_ac = rel_rms(h_k_ac, Kref), r_v_ac = rel_rms(h_v_ac, Vref), r_v_at = rel_rms(h_v_at, Vref);
    const double r_k_ac3 = rel_rms(h_k_ac3, Kref), r_k_j = rel_rms(h_k_j, Kref), r_v_j = rel_rms(h_v_j, Vref);
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
    check_data(real_cache, r_k_kivi > r_k_lmt, r_k_kivi <= r_k_lmt,
               "per-element, SYNTHETIC: per-channel is the lossier row (no channel outliers for it to exploit)",
               "per-element, REAL CACHE: per-channel is the better row (the channel outliers are what it exploits)");
    std::printf("  per-channel vs per-token, per element: K %.2fx %s, V %.2fx %s\n",
                r_k_kivi > r_k_lmt ? r_k_kivi / r_k_lmt : r_k_lmt / r_k_kivi,
                r_k_kivi > r_k_lmt ? "worse" : "better",
                r_v_kivi > r_v_lmt4 ? r_v_kivi / r_v_lmt4 : r_v_lmt4 / r_v_kivi,
                r_v_kivi > r_v_lmt4 ? "worse" : "better");

    // 3. the attention-output error, one head, over the whole cache
    std::vector<float> q(D);
    for (auto& x : q) x = nd(rng);
    std::vector<int> cells(N);
    for (int i = 0; i < N; ++i) cells[i] = i;
    std::vector<float> out_ref, out_lm3, out_lm4, out_q4, out_uni4, out_uni3, out_lm16, out_kivi, out_lmt,
                       out_k2v4, out_k2v4_dev, out_k4v2, out_k2v1c, out_k2v1t, out_k2c2t, out_k2t2c, out_k1v1,
                       out_ac, out_acat, out_acatc, out_jj, out_jat;
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
    attention(q, h_k_kivi, h_vq4, D, cells, out_k2v4);      // K 2-bit per-channel, V 4-bit (host K reference)
    attention(q, h_k2k, h_v2k, D, cells, out_k2v4_dev);     // the SAME arm off the DEVICE pools: `--kv q2k`
    attention(q, h_k4, h_v_kivi, D, cells, out_k4v2);       // K 4-bit, V 2-bit per-channel
    attention(q, h_k_kivi, h_v_1c, D, cells, out_k2v1c);    // K 2-bit, V 1-bit per-channel
    attention(q, h_k_kivi, h_v_1t, D, cells, out_k2v1t);    // K 2-bit, V 1-bit per-token (KIVI's split)
    attention(q, h_k_kivi, h_v_lmt4, D, cells, out_k2c2t);  // K per-channel 2-bit, V per-token 2-bit
    attention(q, h_k_lmt, h_v_kivi, D, cells, out_k2t2c);   // K per-token 2-bit, V per-channel 2-bit
    attention(q, h_k_1c, h_v_1c, D, cells, out_k1v1);       // 1-bit both, per-channel
    attention(q, h_k_ac, h_v_ac, D, cells, out_ac);         // KIVI's quantizer, per-channel on both
    attention(q, h_k_ac, h_v_at, D, cells, out_acat);       // KIVI's exact split: K per-channel, V per-token
    attention(q, h_k_ac3, h_v_at, D, cells, out_acatc);     // the same with SKVQ clipping on K
    attention(q, h_k_j, h_v_j, D, cells, out_jj);           // joint channel x token scales on both
    attention(q, h_k_j, h_v_at, D, cells, out_jat);         // joint K, per-token asymmetric V
    const double e_ac = out_err(out_ac, out_ref, Vref), e_acat = out_err(out_acat, out_ref, Vref);
    const double e_acatc = out_err(out_acatc, out_ref, Vref), e_jj = out_err(out_jj, out_ref, Vref);
    const double e_jat = out_err(out_jat, out_ref, Vref);
    const double e_k2v4 = out_err(out_k2v4, out_ref, Vref), e_k4v2 = out_err(out_k4v2, out_ref, Vref);
    const double e_k2v4_dev = out_err(out_k2v4_dev, out_ref, Vref);   // the device arm: `--kv q2k` as it runs
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
    std::printf("  %-34s %7.3f%%  %4d B   the SAME arm off the device pools: --kv q2k (q2 append for K, q4 for V)\n",
                "K2c + V4 (device)", 100 * e_k2v4_dev, B2 + B4);
    // ---- q2k UNDER THE ENGINE'S APPEND PATTERNS. The device arm above appends all N cells in ONE call - whole
    //    groups only - so it never runs the partial-group scale rule. The engine does, every request: a prompt is
    //    prefilled in chunks whose boundaries need not be 32-aligned (a fresh group is first written with only
    //    the cells before the boundary), and decode then appends one cell at a time into the group the prompt
    //    ended in (keeping the scale the prompt tail wrote). Same K/V, same readers, two engine-shaped patterns:
    //      chunked: K appended in 100-cell prefill chunks (every boundary splits a group)
    //      decode:  K prompt of N - 14 cells (it ends mid-group), then 14 single-cell decode appends
    //    V stays q4_0 written in one call - its blocks are per cell, so no pattern can change them.
    {
        const auto q2k_k_after = [&](const std::vector<std::pair<int, int>>& prefill, int decode_from) {
            uint8_t* kp = dalloc<uint8_t>(q2_bytes);
            for (const auto& c : prefill)   // (pos0, T): a prefill chunk, rows relative to the chunk
                k::kv_append_q2(kp, kp, d_table, c.first, c.second, dK + (size_t) c.first * H * D,
                                dK + (size_t) c.first * H * D, s, nullptr, nullptr, nullptr, /*v3level*/ false);
            for (int c = decode_from; c < N; ++c) {   // the decode shape: one cell per launch, step-addressed
                int32_t hs[k::kStepCount] = {c, c + 1, 0, 0};
                int32_t* ds = dalloc<int32_t>(k::kStepCount);
                ck(cudaMemcpy(ds, hs, sizeof hs, cudaMemcpyHostToDevice), "q2k decode step");
                k::kv_append_q2_step(kp, kp, d_table, ds, dK + (size_t) c * H * D, dK + (size_t) c * H * D, s,
                                     nullptr, nullptr, /*v3level*/ false);
                ck(cudaDeviceSynchronize(), "q2k decode append");
                cudaFree(ds);
            }
            ck(cudaDeviceSynchronize(), "q2k pattern append");
            k::kv_gather_q2_step(kp, kp, d_table, d_ids, d_step, N, s, k_scr_2k, k_scr_2k, nullptr,
                                 /*v3level*/ false);
            ck(cudaDeviceSynchronize(), "q2k pattern gather");
            std::vector<uint16_t> w(rows * D);
            ck(cudaMemcpy(w.data(), k_scr_2k, w.size() * 2, cudaMemcpyDeviceToHost), "q2k pattern K");
            cudaFree(kp);
            std::vector<float> hk(rows * D);
            for (size_t i = 0; i < w.size(); ++i) hk[i] = k::f32_from_f16(w[i]);
            return hk;
        };
        std::vector<std::pair<int, int>> chunks;
        for (int p0 = 0; p0 < N; p0 += 100) chunks.push_back({p0, std::min(100, N - p0)});
        const int prompt = N - 14;
        // speculative decode: an aligned prompt, then 3-cell accepted verify windows through the batch append
        // (no previous-group borrow there), so each new group is first written from 1-3 cells
        const int vbase = (N / 2) / k::Q2_0 * k::Q2_0;
        std::vector<std::pair<int, int>> verify{{0, vbase}};
        for (int p0 = vbase; p0 < N; p0 += 3) verify.push_back({p0, std::min(3, N - p0)});
        const std::vector<float> k_chunk = q2k_k_after(chunks, N);
        const std::vector<float> k_decode = q2k_k_after({{0, prompt}}, prompt);
        const std::vector<float> k_verify = q2k_k_after(verify, N);
        std::vector<float> out_chunk, out_decode, out_verify;
        attention(q, k_chunk, h_v2k, D, cells, out_chunk);
        attention(q, k_decode, h_v2k, D, cells, out_decode);
        attention(q, k_verify, h_v2k, D, cells, out_verify);
        const double e_chunk = out_err(out_chunk, out_ref, Vref), e_decode = out_err(out_decode, out_ref, Vref);
        const double e_verify = out_err(out_verify, out_ref, Vref);
        std::printf("  %-34s %7.3f%%  %4d B   --kv q2k, prompt prefilled in 100-cell chunks (boundaries split groups)\n",
                    "K2c + V4 (device, chunked)", 100 * e_chunk, B2 + B4);
        std::printf("  %-34s %7.3f%%  %4d B   --kv q2k, a %d-cell prompt (ends mid-group) + 14 decode appends\n",
                    "K2c + V4 (device, prompt+decode)", 100 * e_decode, B2 + B4, prompt);
        std::printf("  %-34s %7.3f%%  %4d B   --kv q2k, a %d-cell prompt + 3-cell speculative verify windows\n",
                    "K2c + V4 (device, spec verify)", 100 * e_verify, B2 + B4, vbase);
        std::printf("  q2k vs q4_0 (%.3f%%): one-shot %.2fx, chunked %.2fx, prompt+decode %.2fx, spec verify %.2fx the "
                    "q4_0 error\n", 100 * e_q4, e_k2v4_dev / e_q4, e_chunk / e_q4, e_decode / e_q4, e_verify / e_q4);
        check(std::fabs(e_decode - e_k2v4_dev) <= e_k2v4_dev * 0.01,
              "q2k prompt + decode (the prompt ends mid-group, decode appends keep its scale) lands on the one-shot row");
        // Measured, not pinned: a group FIRST written from a few cells (a 100-cell chunk boundary leaves 4-28;
        // a 3-cell verify window 1-3) fixes its scale from them for life. With the shrinkage rule (the full-group
        // RMS, the previous group's mean square standing in for the missing cells) the verify row sits within
        // ~1-4% of one-shot on the real layers (it was up to +21% on layer 23 with the plain RMS of the n cells);
        // the chunked row is mixed by layer (layers 11/23 better, 3/47 worse than the plain RMS). Real prefill
        // chunks are 2048 cells (aligned) - the chunked row is a stress case; the verify row is what every
        // speculative decode does at each group boundary.
        std::printf("  (measured, not pinned) chunked %+.1f%%, spec verify %+.1f%% vs one-shot: the first write of a "
                    "group fixes its scale from %s cells\n", 100 * (e_chunk / e_k2v4_dev - 1),
                    100 * (e_verify / e_k2v4_dev - 1), "the few");
    }
    std::printf("  %-34s %7.3f%%  %4d B   K 4-bit, V 2b per-channel\n", "K4 + V2c", 100 * e_k4v2, B4 + B2);
    std::printf("  %-34s %7.3f%%  %4d B   K 2b per-channel, V 1b per-channel\n", "K2c + V1c", 100 * e_k2v1c, B2 + B1c);
    std::printf("  %-34s %7.3f%%  %4d B   K 2b per-channel, V 1b per-token (KIVI's split)\n", "K2c + V1t", 100 * e_k2v1t, B2 + B1t);
    std::printf("  %-34s %7.3f%%  %4d B   K per-channel, V per-token, both 2b\n", "K2c + V2t", 100 * e_k2c2t, 2 * B2);
    std::printf("  %-34s %7.3f%%  %4d B   K per-token, V per-channel, both 2b\n", "K2t + V2c", 100 * e_k2t2c, 2 * B2);
    std::printf("  %-34s %7.3f%%  %4d B   both 1-bit per-channel\n", "K1c + V1c", 100 * e_k1v1, 2 * B1c);
    std::printf("  reference rows: symmetric q2c %7.3f%% (%4d B), q4_0 %7.3f%% (%4d B)\n",
                100 * e_lm4, 2 * B2, 100 * e_q4, 2 * B4);
    // The published quantizers, at their own byte costs (per-channel asymmetric: 64 B codes + 2 fp16 per
    // dimension per 32 cells = 96 B; per-token asymmetric: 64 B codes + 2 fp16 = 68 B; joint: 64 + 16 + 2 = 82).
    std::printf("\n  the published quantizers (asymmetric min/max + zero-point, joint scales):\n");
    std::printf("  %-34s %7.3f%%  %4d B   KIVI's quantizer, per-channel on K and V\n", "asym-c K+V", 100 * e_ac, 2 * 96);
    std::printf("  %-34s %7.3f%%  %4d B   KIVI's split: K per-channel, V per-token\n", "asym-c K + asym-t V", 100 * e_acat, 96 + 68);
    std::printf("  %-34s %7.3f%%  %4d B   the same, K clipped at 3 sd (SKVQ)\n", "asym-c3 K + asym-t V", 100 * e_acatc, 96 + 68);
    std::printf("  %-34s %7.3f%%  %4d B   joint channel x token scales on K and V\n", "joint K+V", 100 * e_jj, 2 * 82);
    std::printf("  %-34s %7.3f%%  %4d B   joint K, per-token asymmetric V\n", "joint K + asym-t V", 100 * e_jat, 82 + 68);
    std::printf("  per-element: asym-c K %.3f%% (symmetric %.3f%%), asym-t V %.3f%%, joint K %.3f%%, joint V %.3f%%\n",
                100 * r_k_ac, 100 * r_k_kivi, 100 * r_v_at, 100 * r_k_j, 100 * r_v_j);

    // ---- SKVQ's CHANNEL REORDERING (arXiv 2405.06219) and the fractional-bit widths it was published with
    // (K 2.5-bit, V 1.5-bit). The reorder sorts channels by RMS so a per-token group of 32 CONSECUTIVE
    // channels is homogeneous. For K's per-channel scale the order is a relabeling - proven below, not
    // assumed - so the reorder can only act through V's per-token groups and the mixed-precision placement.
    // Every reordered arm is unpermuted before its attention run, so the rows sit on the same scale as the
    // tables above.
    std::vector<int> perm_k, perm_v, iperm_k((size_t) H * D), iperm_v((size_t) H * D);
    channel_order(Kref, N, H, D, perm_k);
    channel_order(Vref, N, H, D, perm_v);
    for (int h = 0; h < H; ++h)
        for (int d = 0; d < D; ++d) {
            iperm_k[(size_t) h * D + perm_k[(size_t) h * D + d]] = d;
            iperm_v[(size_t) h * D + perm_v[(size_t) h * D + d]] = d;
        }
    std::vector<float> Kr, Vr;
    apply_channel_order(Kref, perm_k, N, H, D, Kr);
    apply_channel_order(Vref, perm_v, N, H, D, Vr);
    // PROOF of the K claim: per-channel quantization of the reordered cache, unpermuted back, must be
    // bit-for-bit the quantization of the original - a channel permutation changes which LABEL a scale
    // carries, never the 32 cells the scale spans.
    std::vector<float> h_k_kivi_r(Kref.size()), h_k_kivi_ru(Kref.size());
    quant_kivi(Kr.data(), N, H, D, false, h_k_kivi_r.data());
    apply_channel_order(h_k_kivi_r, iperm_k, N, H, D, h_k_kivi_ru);
    bool k_reorder_bitexact = true;
    for (size_t i = 0; i < Kref.size(); ++i)
        if (h_k_kivi_ru[i] != h_k_kivi[i]) { k_reorder_bitexact = false; break; }
    check(k_reorder_bitexact,
          "channel reordering does not move per-channel K quantization at all (relabeling, not re-estimation)");

    // Fractional-bit K arms, per-channel: 6-level Lloyd-Max (~2.58 bits), 6-level asymmetric min/max
    // (zero-point at the same codebook size - the user's zero-point-on-K ask), and the 2.5-bit average
    // (loud half 6 levels, quiet half 4).
    std::vector<float> h_k_6c(Kref.size()), h_k_6ac(Kref.size()), h_k_mix(Kref.size());
    quant_kivi_l(Kref.data(), N, H, D, 6, h_k_6c.data());
    quant_asym_l(Kref.data(), N, H, D, 6, true, h_k_6ac.data());
    quant_kivi_mixed(Kref.data(), N, H, D, h_k_mix.data());
    const double r_k_6c = rel_rms(h_k_6c, Kref), r_k_6ac = rel_rms(h_k_6ac, Kref), r_k_mix = rel_rms(h_k_mix, Kref);

    // Fractional-bit V arms, per-token, natural order AND reordered: 3-level LM (~1.58 bits) and 4-level
    // asymmetric min/max. The reorder's whole effect is here.
    std::vector<float> h_v_3t(Vref.size()), h_v_3t_ru(Vref.size()), h_v_3t_ru_u(Vref.size()),
                       h_v_4t_ru(Vref.size()), h_v_4t_ru_u(Vref.size());
    for (size_t row = 0; row < rows; ++row)
        for (int b = 0; b < D / k::Q2_0; ++b) {
            const size_t base = row * D + (size_t) b * k::Q2_0;
            quant_lm(Vref.data() + base, k::Q2_0, true, h_v_3t.data() + base);
            quant_lm(Vr.data() + base, k::Q2_0, true, h_v_3t_ru.data() + base);
        }
    quant_asym(Vr.data(), N, H, D, false, 0.0f, h_v_4t_ru.data());
    apply_channel_order(h_v_3t_ru, iperm_v, N, H, D, h_v_3t_ru_u); // 3-level LM, reordered, back in natural order
    apply_channel_order(h_v_4t_ru, iperm_v, N, H, D, h_v_4t_ru_u); // 4-level asym, reordered, back in natural order
    const double r_v_3t = rel_rms(h_v_3t, Vref), r_v_3t_r = rel_rms(h_v_3t_ru_u, Vref),
                 r_v_4t_r = rel_rms(h_v_4t_ru_u, Vref);

    std::vector<float> out_v3t, out_v3t_r, out_v4t_r, out_k6c, out_k6ac, out_kmix,
                       out_k6c_v3tr, out_kmix_v3tr, out_k6ac_v3tr, out_kac_v4, out_k2c_v3tr;
    attention(q, Kref, h_v_3t, D, cells, out_v3t);
    attention(q, Kref, h_v_3t_ru_u, D, cells, out_v3t_r);
    attention(q, Kref, h_v_4t_ru_u, D, cells, out_v4t_r);
    attention(q, h_k_6c, h_v_kivi, D, cells, out_k6c);
    attention(q, h_k_6ac, h_v_kivi, D, cells, out_k6ac);
    attention(q, h_k_mix, h_v_kivi, D, cells, out_kmix);
    attention(q, h_k_6c, h_v_3t_ru_u, D, cells, out_k6c_v3tr);
    attention(q, h_k_mix, h_v_3t_ru_u, D, cells, out_kmix_v3tr);
    attention(q, h_k_6ac, h_v_3t_ru_u, D, cells, out_k6ac_v3tr);
    attention(q, h_k_ac, h_vq4, D, cells, out_kac_v4);   // zero-point K at 2 bits on q2k's V side
    attention(q, h_k_kivi, h_v_3t_ru_u, D, cells, out_k2c_v3tr);  // SKVQ's own budget: K 2b per-channel + V 1.5b reordered, byte-tied with q2c
    const double e_v3t = out_err(out_v3t, out_ref, Vref), e_v3t_r = out_err(out_v3t_r, out_ref, Vref);
    const double e_v4t_r = out_err(out_v4t_r, out_ref, Vref);
    const double e_k6c = out_err(out_k6c, out_ref, Vref), e_k6ac = out_err(out_k6ac, out_ref, Vref);
    const double e_kmix = out_err(out_kmix, out_ref, Vref);
    const double e_k6c_v3tr = out_err(out_k6c_v3tr, out_ref, Vref);
    const double e_kmix_v3tr = out_err(out_kmix_v3tr, out_ref, Vref);
    const double e_k6ac_v3tr = out_err(out_k6ac_v3tr, out_ref, Vref);
    const double e_kac_v4 = out_err(out_kac_v4, out_ref, Vref);
    const double e_k2c_v3tr = out_err(out_k2c_v3tr, out_ref, Vref);

    check(r_k_6c < r_k_kivi && r_k_6ac < r_k_kivi,
          "a 6-level K codebook beats the 4-level one per element, symmetric or zero-point (2.5-bit K has a floor to stand on)");
    std::printf("\n  SKVQ reorder + fractional bits (attention output; bytes per cell and KV head):\n");
    std::printf("  %-30s %7.3f%%  %4d B   V 3-level per-token, natural order (1.5-bit V, no reorder)\n",
                "V3t only", 100 * e_v3t, B2);
    std::printf("  %-30s %7.3f%%  %4d B   the same V arm with SKVQ's channel reordering\n",
                "V3t + reorder", 100 * e_v3t_r, B2);
    std::printf("  %-30s %7.3f%%  %4d B   V 4-level asym per-token, reordered\n",
                "V4t-asym + reorder", 100 * e_v4t_r, B2);
    std::printf("  %-30s %7.3f%%  %4d B   K 6-level per-channel (~2.58-bit K), V q2\n",
                "K6c only", 100 * e_k6c, 112 + B2);
    std::printf("  %-30s %7.3f%%  %4d B   K 6-level asymmetric (zero-point), V q2\n",
                "K6ac only", 100 * e_k6ac, 128 + B2);
    std::printf("  %-30s %7.3f%%  %4d B   K 2.5-bit mixed (loud half 6-level), V q2\n",
                "K2.5-mixed only", 100 * e_kmix, 96 + B2);
    std::printf("  %-30s %7.3f%%  %4d B   SKVQ's own budget: K 2b per-channel, V 1.5b reordered\n",
                "K2c + V3t-reord", 100 * e_k2c_v3tr, B2 + B2);
    std::printf("  %-30s %7.3f%%  %4d B   SKVQ's own split, reordered V\n",
                "K6c + V3t-reord", 100 * e_k6c_v3tr, 112 + B2);
    std::printf("  %-30s %7.3f%%  %4d B   the 2.5-bit K / 1.5-bit V average\n",
                "K2.5-mix + V3t-reord", 100 * e_kmix_v3tr, 96 + B2);
    std::printf("  %-30s %7.3f%%  %4d B   zero-point 2.58-bit K + reordered 1.5-bit V\n",
                "K6ac + V3t-reord", 100 * e_k6ac_v3tr, 128 + B2);
    std::printf("  %-30s %7.3f%%  %4d B   zero-point 2-bit K on q2k's V side\n",
                "K2ac + V4 (= q2k, K zero-point)", 100 * e_kac_v4, 96 + B4);
    std::printf("  per element: K6c %.3f%% (4-level %.3f%%), K6ac %.3f%%, K2.5-mix %.3f%%; "
                "V3t %.3f%% -> reordered %.3f%% (4-level asym reordered %.3f%%)\n",
                100 * r_k_6c, 100 * r_k_kivi, 100 * r_k_6ac, 100 * r_k_mix,
                100 * r_v_3t, 100 * r_v_3t_r, 100 * r_v_4t_r);
    std::printf("  reference rows: q2c %.3f%% (160 B), q2k %.3f%% (%d B), q4_0 %.3f%% (%d B)\n",
                100 * e_lm4, 100 * e_k2v4_dev, B2 + B4, 100 * e_q4, 2 * B4);
    // The reorder's verdict is data-scoped by construction: on i.i.d. noise every channel's SAMPLED RMS is a
    // draw from the same distribution, so the permutation is an arbitrary relabeling and the error can only
    // move by quantizer noise - pinned at 10%, measured and printed everywhere.
    if (!real_cache)
        check(e_v3t_r <= e_v3t * 1.10,
              "SYNTHETIC: reordering changes nothing beyond noise (i.i.d. channels have no RMS order to sort)");
    std::printf("  reorder on V 3-level per-token (layer %d): %.3f%% -> %.3f%% (%.2fx) - %s\n", probe_layer,
                100 * e_v3t, 100 * e_v3t_r, e_v3t / e_v3t_r,
                e_v3t_r < e_v3t ? "the reorder helps" : "the reorder does not help");
    std::printf("  SKVQ's byte-tied split (K2c + V3t-reord) vs our q2c at the same 160 B: %.3f%% vs %.3f%%\n",
                100 * e_k2c_v3tr, 100 * e_lm4);
    std::printf("  verdict vs q4_0: best fractional arm %.3f%% against q4_0's %.3f%% and q2k's %.3f%%\n",
                100 * std::min(std::min(e_k6c_v3tr, e_kmix_v3tr), std::min(e_k6ac_v3tr, e_kac_v4)),
                100 * e_q4, 100 * e_k2v4_dev);

    // ---- TOP-FRACTION K bit allocation: the loudest channels get the bits, everyone else stays at 2 bits.
    // V stays at the shipping q2 per-channel, so every row is directly comparable to q2c (K2c + V2c, 160 B)
    // and to its own q2k-shaped twin (V q4_0). Bytes: 32 * (2 + top_frac * (bits_loud - 2)) + 16 per cell/head.
    std::vector<float> h_k_t8_16(Kref.size()), h_k_t4_16(Kref.size()), h_k_t8_6(Kref.size()), h_k_t4_6(Kref.size());
    quant_kivi_topq(Kref.data(), N, H, D, 1.f / 8.f, 16, h_k_t8_16.data());
    quant_kivi_topq(Kref.data(), N, H, D, 1.f / 4.f, 16, h_k_t4_16.data());
    quant_kivi_topq(Kref.data(), N, H, D, 1.f / 8.f, 6, h_k_t8_6.data());
    quant_kivi_topq(Kref.data(), N, H, D, 1.f / 4.f, 6, h_k_t4_6.data());
    const double r_k_t8_16 = rel_rms(h_k_t8_16, Kref), r_k_t4_16 = rel_rms(h_k_t4_16, Kref);
    const double r_k_t8_6 = rel_rms(h_k_t8_6, Kref), r_k_t4_6 = rel_rms(h_k_t4_6, Kref);
    std::vector<float> out_t8_16, out_t4_16, out_t8_6, out_t4_6, out_t4_16_k;
    attention(q, h_k_t8_16, h_v_kivi, D, cells, out_t8_16);
    attention(q, h_k_t4_16, h_v_kivi, D, cells, out_t4_16);
    attention(q, h_k_t8_6, h_v_kivi, D, cells, out_t8_6);
    attention(q, h_k_t4_6, h_v_kivi, D, cells, out_t4_6);
    attention(q, h_k_t4_16, h_vq4, D, cells, out_t4_16_k);   // the winner's q2k-shaped twin, if it wins
    const double e_t8_16 = out_err(out_t8_16, out_ref, Vref), e_t4_16 = out_err(out_t4_16, out_ref, Vref);
    const double e_t8_6 = out_err(out_t8_6, out_ref, Vref), e_t4_6 = out_err(out_t4_6, out_ref, Vref);
    const double e_t4_16_k = out_err(out_t4_16_k, out_ref, Vref);
    check(r_k_t8_16 < r_k_kivi && r_k_t4_16 < r_k_kivi,
          "spending the extra bits on the loud channels beats spreading them evenly (per element)");
    std::printf("\n  TOP-FRACTION K bit allocation (attention output; V at the shipping q2 per-channel):\n");
    std::printf("  %-34s %7.3f%%  %4d B   K: loudest 1/8 at 16 levels, rest 4 (2.25 b avg)\n",
                "K-top1/8@16b", 100 * e_t8_16, 88);
    std::printf("  %-34s %7.3f%%  %4d B   K: loudest 1/4 at 16 levels, rest 4 (2.5 b avg)\n",
                "K-top1/4@16b", 100 * e_t4_16, 96);
    std::printf("  %-34s %7.3f%%  %4d B   K: loudest 1/8 at 6 levels, rest 4 (2.125 b avg)\n",
                "K-top1/8@6b", 100 * e_t8_6, 84);
    std::printf("  %-34s %7.3f%%  %4d B   K: loudest 1/4 at 6 levels, rest 4 (2.25 b avg)\n",
                "K-top1/4@6b", 100 * e_t4_6, 88);
    std::printf("  %-34s %7.3f%%  %4d B   K-top1/4@16b's q2k-shaped twin (V q4_0)\n",
                "K-top1/4@16b + V4", 100 * e_t4_16_k, 96 + B4);
    std::printf("  per element: top1/8@16 %.3f%%, top1/4@16 %.3f%%, top1/8@6 %.3f%%, top1/4@6 %.3f%% "
                "(uniform 4-level %.3f%%, uniform 6-level %.3f%%)\n",
                100 * r_k_t8_16, 100 * r_k_t4_16, 100 * r_k_t8_6, 100 * r_k_t4_6, 100 * r_k_kivi, 100 * r_k_6c);
    std::printf("  reference rows: q2c %.3f%% (160 B), q2k %.3f%% (%d B)\n",
                100 * e_lm4, 100 * e_k2v4_dev, B2 + B4);

    // ---- THE LOW-BYTE COMBOS: K2V2 is the shipping q2 (320 B/cell); the question is what K2.5 and V1.5 buy.
    // Every arm here uses FIXED codebooks (the form a format ships): a converged 1M-sample Lloyd-Max grid per
    // level count, scaled by the group's fp16-rounded RMS. No per-group fits - the top-q section showed those
    // are an encoder ceiling. V1.5's byte accounting is stated both ways: 3 levels at 2-bit storage (80 B per
    // cell and head, same bytes as 4-level) and tight 5-trits-per-byte packing (3^5 = 243 <= 256, 7 B of codes
    // per 32 values, 72 B).
    std::vector<float> cb6, thr6, cb3, thr3;
    fit_lm_grid(6, cb6, thr6);
    fit_lm_grid(3, cb3, thr3);
    std::vector<float> h_k25c(Kref.size()), h_v3c(Vref.size());
    quant_topq_grid(Kref.data(), N, H, D, 0.25f, cb6, thr6, h_k25c.data());    // K 2.5-bit: loud 1/4 at 6 levels
    quant_per_channel_grid(Vref.data(), N, H, D, cb3, thr3, h_v3c.data());     // V 1.5-bit: 3 levels per-channel
    const double r_k25c = rel_rms(h_k25c, Kref), r_v3c = rel_rms(h_v3c, Vref);
    std::vector<float> out_k25_v3, out_k2_v3, out_k25_v2;
    attention(q, h_k25c, h_v3c, D, cells, out_k25_v3);
    attention(q, h_k_kivi, h_v3c, D, cells, out_k2_v3);
    attention(q, h_k25c, h_v_kivi, D, cells, out_k25_v2);
    const double e_k25_v3 = out_err(out_k25_v3, out_ref, Vref);
    const double e_k2_v3 = out_err(out_k2_v3, out_ref, Vref);
    const double e_k25_v2 = out_err(out_k25_v2, out_ref, Vref);
    std::printf("\n  low-byte combos (attention output; FIXED grids; bytes per cell and KV head):\n");
    std::printf("  %-30s %7.3f%%  %4d B   K2c + V3c (V1.5 at 2-bit storage / 72 B tight-trit)\n",
                "K2c + V3c", 100 * e_k2_v3, 160);
    std::printf("  %-30s %7.3f%%  %4d B   K 2.5-bit (loud 1/4 at 6 levels) + V3c\n",
                "K2.5c + V3c", 100 * e_k25_v3, 168);
    std::printf("  %-30s %7.3f%%  %4d B   K 2.5-bit + V 2-bit per-channel\n",
                "K2.5c + V2c", 100 * e_k25_v2, 168);
    std::printf("  per element: K2.5c %.3f%% (4-level %.3f%%), V3c %.3f%% (4-level %.3f%%)\n",
                100 * r_k25c, 100 * r_k_kivi, 100 * r_v3c, 100 * r_v_kivi);
    std::printf("  reference rows: q2c %.3f%% (160 B), q2k %.3f%% (%d B), q4_0 %.3f%% (%d B)\n",
                100 * e_lm4, 100 * e_k2v4_dev, B2 + B4, 100 * e_q4, 2 * B4);

    // ==================== THE q2t DEVICE ARM (`--kv q2t`) ====================
    //
    // K filled by the q2t group kernel (the freeze runs on this append, so the classes come from the same cells
    // the host reference ranks), V by the q4 kernels, each half read back by its own gather kernel. The host row
    // it has to land on is the winner of the block above: K-top1/4@16b (quant_kivi_topq) on q2k's V side.
    //
    // The two paths are NOT expected to agree bit for bit, and that is the honest difference from the q2k arm:
    // the host reference fits a 16-level Lloyd-Max codebook per 32-value group (12 iterations on 32 samples),
    // the device quantizes against the FIXED Gaussian grid scaled by the group's RMS. The gate's claim is that
    // the fixed grid is the right object - the fit on 32 samples is estimator noise around it - so the two
    // rows land on top of each other within 2%, and the check below says so.
    const int nloud = D / 4;
    const int lvl_words = k::q2t_level_words(H, D);
    const size_t q2t_bytes = (size_t) rows * k::kv_q2t_bytes_per_head(D);
    uint8_t* kq2t = dalloc<uint8_t>(q2t_bytes);
    uint8_t* kq2t_ring = dalloc<uint8_t>(q2t_bytes);     // the same cache appended cell by cell (the atomic path)
    uint8_t* vq4_t = dalloc<uint8_t>(q4_bytes);
    uint32_t* d_bits = dalloc<uint32_t>(lvl_words);
    int32_t* d_off = dalloc<int32_t>((size_t) H * D);
    int32_t* d_frozen = dalloc<int32_t>(1);
    uint32_t* d_bits2 = dalloc<uint32_t>(lvl_words);     // the "restored" map: bitmap in, offsets derived
    int32_t* d_off2 = dalloc<int32_t>((size_t) H * D);
    int32_t* d_frozen2 = dalloc<int32_t>(1);
    k::kv_q2t_levels_init(d_bits, d_off, d_frozen, s, nullptr);
    k::kv_q2t_levels_init(d_bits2, d_off2, d_frozen2, s, nullptr);
    const k::KvLevelMap lm{d_bits, d_off, d_frozen};
    const k::KvLevelMap lm2{d_bits2, d_off2, d_frozen2};
    k::kv_append_q2t(kq2t, kq2t, d_table, 0, N, dK, dK, s, nullptr, nullptr, nullptr, lm, /*freeze*/ true);
    k::kv_append_q4(vq4_t, vq4_t, d_table, 0, N, dV, dV, s, nullptr, nullptr);
    // THE RING ARM'S STARTING STATE - stated, not silently picked.
    //
    // Which case the engine is: the prompt append (kv_append_q2t, group-granular: prefill.cpp's prompt copy,
    // mtp.cpp's drafter-ring prefill) runs FIRST and writes every group fully inside its range WHOLE - the
    // grid is over groups, so a complete group takes the whole-slot path and stores its own 32-cell RMS as
    // its scale. The decode append (kv_append_q2t_step: layer.cpp) only ever runs after that. So a
    // single-cell append lands on a group that ALREADY CARRIES a scale set by a whole group, and the branch
    // the scale rule takes is "a partial group keeps the scale its slot already carries". The "no scale yet,
    // take the previous group's" branch fires for exactly ONE group per session - the one right after the
    // prompt's last whole group - and its previous group is a real 32-cell RMS.
    //
    // So: the engine CAN hit an empty-group single-cell append, but never a CHAIN of them and never with a
    // single-sample previous scale. Appending the whole cache cell by cell into a ZEROED pool - which is what
    // this arm used to do - is a state the engine never reaches, and it is not a test of the atomics at all.
    // In that state group 0's scale is the single-cell append's "RMS of the cells the call has" =
    // sqrt(v*v/32) = |v|/sqrt(32), a scale 5.66x too small for a one-sample group; and that number then
    // propagates forever, because group 1 takes it as its previous group's scale, group 2 takes group 1's, and
    // so on - every group of every channel is quantized against ONE single-sample scale, every code
    // saturates, and the gathered K collapses to sign * constant. The ring row then reads as a clobbered
    // atomic when nothing was clobbered. Pre-fill with the prompt append first: that is the state a decode
    // append actually runs in, and the single-cell phase re-writes bytes that are already there, which is
    // exactly what the masked atomics have to survive.
    k::kv_append_q2t(kq2t_ring, kq2t_ring, d_table, 0, N, dK, dK, s, nullptr, nullptr, nullptr, lm,
                     /*freeze*/ false);          // the map is already frozen by this arm's prompt append above
    ck(cudaDeviceSynchronize(), "q2t ring prefill");
    for (int c = 0; c < N; ++c) {                        // the decode/verify shape: one cell per launch, the
        int32_t hstep1[k::kStepCount] = {c, c + 1, 0, 0};  // word-masked atomic path, 2 bits quiet / 4 bits loud
        int32_t* d_step1 = dalloc<int32_t>(k::kStepCount);
        ck(cudaMemcpy(d_step1, hstep1, sizeof hstep1, cudaMemcpyHostToDevice), "q2t step");
        k::kv_append_q2t_step(kq2t_ring, kq2t_ring, d_table, d_step1, dK + (size_t) c * H * D,
                              dK + (size_t) c * H * D, s, nullptr, nullptr, lm);
        ck(cudaDeviceSynchronize(), "q2t step append");
    }
    ck(cudaDeviceSynchronize(), "q2t append");
    uint16_t* k_scr_t = dalloc<uint16_t>((size_t) rows * D);
    uint16_t* v_scr_t = dalloc<uint16_t>((size_t) rows * D);
    uint16_t* k_scr_ring = dalloc<uint16_t>((size_t) rows * D);
    k::kv_gather_q2t_step(kq2t, kq2t, d_table, d_ids, d_step, N, s, k_scr_t, k_scr_t, nullptr, lm);
    k::kv_gather_q2t_step(kq2t_ring, kq2t_ring, d_table, d_ids, d_step, N, s, k_scr_ring, k_scr_ring, nullptr, lm);
    k::kv_gather_q4_step(vq4_t, vq4_t, d_table, d_ids, d_step, N, s, v_scr_t, v_scr_t, nullptr);
    ck(cudaDeviceSynchronize(), "q2t gather");
    std::vector<uint16_t> w_kt(rows * D), w_vt(rows * D), w_kt_ring(rows * D);
    ck(cudaMemcpy(w_kt.data(), k_scr_t, w_kt.size() * 2, cudaMemcpyDeviceToHost), "k q2t");
    ck(cudaMemcpy(w_vt.data(), v_scr_t, w_vt.size() * 2, cudaMemcpyDeviceToHost), "v q2t");
    ck(cudaMemcpy(w_kt_ring.data(), k_scr_ring, w_kt_ring.size() * 2, cudaMemcpyDeviceToHost), "k q2t ring");
    std::vector<float> h_kt(rows * D), h_vt(rows * D), h_kt_ring(rows * D);
    bool kt_finite = true;
    for (size_t i = 0; i < w_kt.size(); ++i) {
        h_kt[i] = k::f32_from_f16(w_kt[i]);
        h_vt[i] = k::f32_from_f16(w_vt[i]);
        h_kt_ring[i] = k::f32_from_f16(w_kt_ring[i]);
        if (!std::isfinite(h_kt[i]) || !std::isfinite(h_vt[i]) || !std::isfinite(h_kt_ring[i])) kt_finite = false;
    }
    check(kt_finite, "the q2t pools dequantize to finite values everywhere (no plane read out of its slot)");
    std::vector<float> out_t_dev;
    attention(q, h_kt, h_vt, D, cells, out_t_dev);
    const double e_t_dev = out_err(out_t_dev, out_ref, Vref);
    const double r_k_t = rel_rms(h_kt, Kref), r_v_t = rel_rms(h_vt, Vref);
    const double r_k_t_ring = rel_rms(h_kt_ring, Kref);
    check(e_t_dev > 0.0, "the q2t device arm is live (the engine's appends filled both halves)");
    // A K that dequantizes to ZEROS is not "lossy", it is a reader that reads bytes the writer never wrote -
    // a plane boundary, a block stride, an offset table that the append and the gather do not agree on. It
    // costs 100% per element exactly (rel_rms of a zero vector against the data is 1.0), it leaves the
    // attention output at a plausible mid-range number (softmax over zero scores is the mean of V), and every
    // check below reads as a quality miss. Named here, where it is unambiguous.
    check(r_k_t < 0.5, "the q2t K gather carries signal (a K that dequantizes to zeros is a plane/offset read "
                       "the append never wrote, not a 2-bit codebook)");
    // The HONEST host reference: the fixed-grid twin (quant_kivi_topq_fixed), the same constants the device
    // kernels call. The per-group-fit row (e_t4_16_k) is an encoder ceiling - 16 centroids fitted to a
    // group's 32 samples is degenerate - so it is printed as a diagnostic and the pin runs against the
    // fixed grid, which is what the format ships.
    std::vector<float> h_k_tfix(Kref.size());
    quant_kivi_topq_fixed(Kref.data(), N, H, D, 0.25f, h_k_tfix.data());
    const double r_k_tfix = rel_rms(h_k_tfix, Kref);
    std::vector<float> out_tfix_k;
    attention(q, h_k_tfix, h_vq4, D, cells, out_tfix_k);
    const double e_tfix_k = out_err(out_tfix_k, out_ref, Vref);
    std::printf("  q2t host references: fixed grid %.3f%% (the format's own quantizer), per-group fit "
                "%.3f%% (16 centroids on 32 samples - an encoder ceiling, not a format)\n",
                100 * e_tfix_k, 100 * e_t4_16_k);
    check(e_t_dev < 1.02 * e_tfix_k && e_t_dev > 0.98 * e_tfix_k,
          "the q2t device arm lands on the fixed-grid host row (within 2% - same constants, same scale rule)");
    // q2t vs q2k is LAYER-scoped on the real cache: with the fit-authoritative grid q2t wins on the typical
    // layers (1.02x L3, 1.23x L11, 1.09x L23) and loses only on the extreme-outlier layer 47 (0.60x), where
    // K's channel RMS spans 74.9x and the fixed 16-level grid still cannot track it. The real-side pin says
    // exactly that: never lose by more than 1.7x (the observed worst case), and the format decision follows
    // the measurement - q2k remains the shipping mixed format unless the outlier layers get their own answer.
    check_data(real_cache, e_t_dev < e_k2v4_dev, e_t_dev < 1.7 * e_k2v4_dev,
               "SYNTHETIC: q2t beats q2k on the attention output (no channel outliers to amplify the grid's error)",
               "REAL CACHE: q2t wins on typical layers and loses only on the extreme-outlier layer - q2k stays the shipping mixed format");
    // The decode/verify shape (one cell per launch, the word-masked atomic path) must survive the plane split:
    // its scales differ from the prompt path's by the scale rule (a partial group keeps the stored scale, a
    // single-cell append takes the previous group's), so the pin is that it is the same cache to within that
    // rule, not a clobbered one. A lost nibble shows up as a K error far above the scale rule's cost.
    check(r_k_t_ring < 1.5 * r_k_t && r_k_t_ring > 0.0,
          "the q2t cell-by-cell append (partial groups, atomics, both planes) lands within the scale rule of the "
          "prompt append (a clobbered 4-bit code would not)");
    // The byte-level version of the same statement, because the RMS ratio is a soft pin. The ring is NOT
    // expected to be bit-identical to the prompt append, and the reason is the scale rule, not the atomics:
    // a complete group quantizes its codes against the FULL-PRECISION RMS and stores the fp16 rounding of it,
    // while a partial group re-quantizes against the STORED (fp16-rounded) scale, so a value sitting within
    // half an fp16 ulp of a decision threshold (2^-11 = 0.049% in the scale) flips its code. Measured over
    // the 7 loud and 1 quiet thresholds at Gaussian density that is ~0.1% of the cells; a clobbered nibble
    // (loud) or a clobbered 2-bit code (quiet) is a whole-byte event and lands orders of magnitude above it.
    size_t ring_diff = 0;
    for (size_t i = 0; i < w_kt_ring.size(); ++i)
        if (w_kt_ring[i] != w_kt[i]) ++ring_diff;
    const double ring_diff_frac = (double) ring_diff / (double) w_kt_ring.size();
    check(ring_diff_frac < 0.01, "the q2t cell-by-cell append over a prefilled pool reproduces the prompt "
                                 "append's K bytes (only scale-rounding threshold flips differ, not lost codes)");
    std::printf("  q2t ring: %zu of %zu gathered cells differ from the prompt append (%.4f%%) - the fp16 scale "
                "rounding's threshold flips; a clobbered nibble would be a whole byte per event\n",
                ring_diff, w_kt_ring.size(), 100.0 * ring_diff_frac);

    // ---- THE BITMAP: the device's classes are the host reference's, the offsets are its prefix sum, and a
    // save/restore round-trip reproduces the same classes (the snapshot stores the bitmap, not the offsets).
    std::vector<uint32_t> h_bits(lvl_words);
    ck(cudaMemcpy(h_bits.data(), d_bits, lvl_words * 4, cudaMemcpyDeviceToHost), "q2t bitmap");
    std::vector<int32_t> h_off((size_t) H * D);
    ck(cudaMemcpy(h_off.data(), d_off, (size_t) H * D * 4, cudaMemcpyDeviceToHost), "q2t offsets");
    int32_t h_frozen = -1;
    ck(cudaMemcpy(&h_frozen, d_frozen, 4, cudaMemcpyDeviceToHost), "q2t frozen flag");
    check(h_frozen == 1, "the first prompt batch froze the bitmap (every later append reads it, never re-ranks)");
    const std::vector<uint32_t> want_bits = host_topq_bits(Kref, N, H, D, nloud);
    bool bits_match = true;
    for (size_t i = 0; i < want_bits.size(); ++i)
        if (want_bits[i] != h_bits[i]) bits_match = false;
    check(bits_match, "the device's loud set is the host reference's top 1/4 by channel RMS, bit for bit");
    int loud_count = 0, off_ok = 0;
    for (int h = 0; h < H; ++h)
        for (int d = 0; d < D; ++d)
            if ((h_bits[h * (D / 32) + (d >> 5)] >> (d & 31)) & 1u) ++loud_count;
    check(loud_count == H * nloud, "the loud set is head_dim/4 channels per KV head (the gate's winning fraction)");
    {   // the offsets: a valid placement - quiet ranks at 10 B steps below the plane boundary, loud ranks at
        // 18 B steps above it, no two channels of a class sharing a slot
        const int qb = (int) k::q2t_quiet_bytes(D), lb = (int) k::q2t_loud_bytes(D);
        std::vector<int> seen_q((size_t) H * (qb / 10), 0), seen_l((size_t) H * (lb / 18), 0);
        off_ok = 1;
        for (int h = 0; h < H; ++h)
            for (int d = 0; d < D; ++d) {
                const int off = h_off[h * D + d];
                const bool loud = ((h_bits[h * (D / 32) + (d >> 5)] >> (d & 31)) & 1u) != 0u;
                if (loud) {
                    if (off < qb || off >= qb + lb || ((off - qb) % 18) != 0) off_ok = 0;
                    else if (++seen_l[(size_t) h * (lb / 18) + (off - qb) / 18] > 1) off_ok = 0;
                } else {
                    if (off < 0 || off >= qb || (off % 10) != 0) off_ok = 0;
                    else if (++seen_q[(size_t) h * (qb / 10) + off / 10] > 1) off_ok = 0;
                }
            }
        check(off_ok == 1, "the per-channel offsets are the bitmap's prefix sum: one slot per channel, 10 B steps "
                           "in the quiet plane, 18 B in the loud one, no collisions");
    }
    {   // The offsets, checked against the header's OWN placement rule rather than against a description of
        // it: q2t_slot_offset(loud, rank, head_dim) is the function the freeze kernel, the restore kernel, the
        // append and every reader use to place and find a channel's slot. A valid prefix sum is not enough -
        // a table that is a valid placement of a DIFFERENT layout (a boundary at 256*10, a loud stride of 10,
        // ranks counted in the other plane) passes the validity check above and sends the writer and the
        // reader to different bytes. This compares the numbers, not the shape.
        bool off_exact = true;
        for (int h = 0; h < H; ++h) {
            int qr = 0, lr = 0;
            for (int d = 0; d < D; ++d) {
                const bool loud = ((h_bits[h * (D / 32) + (d >> 5)] >> (d & 31)) & 1u) != 0u;
                const int want = k::q2t_slot_offset(loud, loud ? lr : qr, D);
                if (loud) ++lr; else ++qr;
                if (h_off[h * D + d] != want) off_exact = false;
            }
        }
        check(off_exact, "the offsets table IS q2t_slot_offset(loud, rank, head_dim) applied to the bitmap - the "
                         "number the append adds to the block base and the number the gather adds are the same "
                         "function, not two layouts that happen to look valid");
    }
    {   // The bytes, read where the reader says they are: every (group, head, channel) slot the gather
        // computes must carry that channel's group RMS as its fp16 scale. This is the check that a plane
        // boundary, a block stride or an offset table that the append and the gather disagree on CANNOT pass,
        // and it names the failure as a placement failure instead of leaving it to show up as a K of zeros.
        std::vector<uint8_t> pool(q2t_bytes);
        ck(cudaMemcpy(pool.data(), kq2t, q2t_bytes, cudaMemcpyDeviceToHost), "q2t pool");
        const long long blk = (long long) k::q2t_block_bytes(D);
        const long long nblocks = (long long) ((long long) q2t_bytes / blk);
        double worst_scale = 0.0;
        long long worst_at = -1;
        bool slots_live = true, slots_inrange = true;
        for (int h = 0; h < H && slots_live; ++h)
            for (int d = 0; d < D && slots_live; ++d)
                for (int g = 0; g + 32 <= N; g += 32) {
                    // g is a CELL index; the group it belongs to is g / 32 (pos0 = 0, identity page table). The
                    // pool holds (N/32) * H = 64 blocks of 3072 B, so a walk that feeds the CELL index to the
                    // block formula lands on block 0 for g = 0 and then on blocks 65, 130, ... 1985 - past the
                    // end of the pool from the SECOND group on. That is what a "dead slot" was: heap bytes
                    // beyond the vector, not a slot the gather reads. The group index is g / 32, and the
                    // address is the header's own q2t_slot() - the function the gather calls - so this walk
                    // tests the gather's addressing instead of a hand-written copy of it.
                    const long long grp = (long long) g / 32;
                    const uint8_t* slot = k::q2t_slot(pool.data(), grp, h, h_off[h * D + d], H, D);
                    const long long rel = (long long) (slot - pool.data());
                    if (rel < 0 || rel + (long long) sizeof(k::block_q2l_0) > (long long) q2t_bytes) {
                        slots_inrange = false;
                        slots_live = false;
                        break;
                    }
                    double ss = 0.0;
                    for (int i = 0; i < 32; ++i) {
                        const double v = Kref[(size_t) (g + i) * H * D + h * D + d];
                        ss += v * v;
                    }
                    const float want = (float) std::sqrt(ss / 32.0);
                    const float got = k::q2t_scale_at(slot);
                    if (!(got > 0.0f)) { slots_live = false; worst_at = rel; break; }
                    const double rel_err = want > 0.0 ? std::fabs((double) got - want) / want : 0.0;
                    if (rel_err > worst_scale) { worst_scale = rel_err; worst_at = rel; }
                }
        check(slots_inrange, "every slot address the walk computes is INSIDE the pool: the group index is "
                             "cell/32, and a walk that feeds the cell index to the block formula reads past the "
                             "end and reports heap bytes as a dead slot");
        check(slots_live, "every q2t slot the gather computes carries a scale: no channel is read out of the "
                          "plane the append wrote it in");
        check(worst_scale < 0.002, "the q2t append wrote each channel's slot exactly where the reader looks: the "
                                   "fp16 scale at block_base(group,head) + off[head,dim] is that channel's RMS "
                                   "over the group's 32 cells");
        std::printf("  q2t placement: the scale at base+off matches the channel's group RMS to %.4f%% worst "
                    "case (byte %lld of %lld, %lld blocks of %lld B = %d quiet slots of 10 + %d loud of %zu)\n",
                    100.0 * worst_scale, worst_at, (long long) q2t_bytes, nblocks, blk, (int) k::q2t_quiet(D),
                    (int) k::q2t_loud(D), sizeof(k::block_q2l_0));
    }
    {   // THE GATHER'S OWN SLOT MATH, pinned against the bytes the append wrote. The check above proves the
        // APPEND placed the scales where the header says they are; it says nothing about what the GATHER
        // kernel returns. This one dequantizes the pool on the host at the address the gather computes -
        // q2t_slot(group, head, off[head,dim]), the class from bitmap word h*(D/32)+(d>>5) bit d&31, the
        // plane's code position - and every gathered fp16 bit has to match, for every (cell, head, dim).
        // A gather that lands on a different byte than the append wrote fails here and only here.
        std::vector<uint8_t> pool(q2t_bytes);
        ck(cudaMemcpy(pool.data(), kq2t, q2t_bytes, cudaMemcpyDeviceToHost), "q2t pool (gather arm)");
        bool gather_reads_the_slots = true;
        size_t gather_first_bad = 0;
        for (int c = 0; c < N && gather_reads_the_slots; ++c)
            for (int h = 0; h < H && gather_reads_the_slots; ++h)
                for (int d = 0; d < D && gather_reads_the_slots; ++d) {
                    const long long grp = (long long) c / 32;
                    const int idx = c % 32;
                    const bool loud = ((h_bits[h * (D / 32) + (d >> 5)] >> (d & 31)) & 1u) != 0u;
                    const uint8_t* slot = k::q2t_slot(pool.data(), grp, h, h_off[h * D + d], H, D);
                    const float sc = k::q2t_scale_at(slot);
                    const int code = loud ? ((slot[2 + (idx >> 1)] >> k::q2t_loud_shift(idx)) & 0xF)
                                          : ((slot[2 + (idx >> 2)] >> k::q2t_quiet_shift(idx)) & 3);
                    const uint16_t want = k::f16_from_f32(loud ? k::q2_value_lm16(code, sc)
                                                               : k::q2_value_lm4(code, sc));
                    if (w_kt[(size_t) c * H * D + h * D + d] != want) {
                        gather_reads_the_slots = false;
                        gather_first_bad = (size_t) c * H * D + (size_t) h * D + d;
                    }
                }
        check(gather_reads_the_slots, "the gather's slot math IS the append's placement: dequantizing the pool at "
                                      "q2t_slot(group, head, off[head,dim]) with the bitmap's class reproduces "
                                      "every gathered fp16 bit");
        if (!gather_reads_the_slots)
            std::printf("  q2t gather slot: first mismatch at cell*H*D + h*D + d = %zu (cell %zu, head %zu, dim "
                        "%zu)\n", gather_first_bad, gather_first_bad / ((size_t) H * D),
                        (gather_first_bad / (size_t) D) % (size_t) H, gather_first_bad % (size_t) D);
    }
    {   // the round-trip: the bitmap out to the host (the snapshot's save) and back into a fresh map (the
        // restore's kv_q2t_levels_from_bits), then the same append through the restored map - the bytes have to
        // come back identical, which is what a resumed session needs.
        //
        // THE ENGINE'S ORDER, WHICH THE ARM HAS TO FOLLOW: a snapshot stores the bitmap in the image's scales
        // slot, and a restore writes those bits into st.kv_levels and ONLY THEN calls levels_from_bits to
        // derive the offsets from them (conversation_snapshot.cpp). Deriving the restored offsets from the
        // SAVED map's bits while the restored map's own bits are still the fresh all-quiet ones does not
        // restore anything: the append below reads classes that were never saved, and the check compares a
        // frozen bitmap against an empty one.
        ck(cudaMemcpy(d_bits2, h_bits.data(), (size_t) lvl_words * 4, cudaMemcpyHostToDevice), "q2t restore bitmap");
        k::kv_q2t_levels_from_bits(d_bits2, d_off2, d_frozen2, s, nullptr);
        ck(cudaDeviceSynchronize(), "q2t restore");
        std::vector<uint32_t> h_bits2(lvl_words);
        ck(cudaMemcpy(h_bits2.data(), d_bits2, lvl_words * 4, cudaMemcpyDeviceToHost), "q2t bitmap 2");
        std::vector<int32_t> h_off2((size_t) H * D);
        ck(cudaMemcpy(h_off2.data(), d_off2, (size_t) H * D * 4, cudaMemcpyDeviceToHost), "q2t offsets 2");
        bool rt_same = true;
        for (size_t i = 0; i < h_off.size(); ++i)
            if (h_off2[i] != h_off[i]) rt_same = false;
        for (size_t i = 0; i < h_bits.size(); ++i)
            if (h_bits2[i] != h_bits[i]) rt_same = false;
        check(rt_same, "the bitmap round-trip reproduces the same classes and the same offsets (save stores the "
                       "bitmap, restore derives the offsets from it)");
        uint8_t* kq2t_rt = dalloc<uint8_t>(q2t_bytes);
        // freeze = FALSE. The map below (lm2) was seeded from the SAVED bits and its offsets derived from them,
        // and levels_from_bits set its frozen flag - so passing freeze = true here asks the freeze kernel to
        // re-rank the classes from this call's cells. It happens to be a no-op today only because the frozen
        // flag is already 1 and the kernel returns on the first line; as a statement of intent it is the op-
        // posite of what this arm is testing. A resumed session never re-ranks: it reads the classes it saved.
        // Say that in the call.
        k::kv_append_q2t(kq2t_rt, kq2t_rt, d_table, 0, N, dK, dK, s, nullptr, nullptr, nullptr, lm2,
                         /*freeze*/ false);
        ck(cudaDeviceSynchronize(), "q2t restore append");
        uint16_t* k_scr_rt = dalloc<uint16_t>((size_t) rows * D);
        k::kv_gather_q2t_step(kq2t_rt, kq2t_rt, d_table, d_ids, d_step, N, s, k_scr_rt, k_scr_rt, nullptr, lm2);
        ck(cudaDeviceSynchronize(), "q2t restore gather");
        std::vector<uint16_t> w_kt_rt(rows * D);
        ck(cudaMemcpy(w_kt_rt.data(), k_scr_rt, w_kt_rt.size() * 2, cudaMemcpyDeviceToHost), "k q2t restore");
        bool rt_bytes = true;
        for (size_t i = 0; i < w_kt_rt.size(); ++i)
            if (w_kt_rt[i] != w_kt[i]) rt_bytes = false;
        check(rt_bytes, "a session resumed on the restored bitmap writes the same K bytes (the classes are frozen "
                        "state, not a re-rank)");
    }
    // The plane split is live: a loud channel's gathered values are one of 16 magnitudes per group, a quiet
    // one's one of 4. A dispatch that quietly read every channel through the 4-level path would pass every
    // error measure above and fail here.
    {
        int max_loud = 0, max_quiet = 0, over16 = 0, over4 = 0;
        for (int h = 0; h < H; ++h)
            for (int d = 0; d < D; ++d) {
                const bool loud = ((h_bits[h * (D / 32) + (d >> 5)] >> (d & 31)) & 1u) != 0u;
                for (int g = 0; g + 32 <= N; g += 32) {
                    std::vector<float> vals;
                    for (int i = 0; i < 32; ++i)
                        vals.push_back(h_kt[(size_t) (g + i) * H * D + h * D + d]);
                    std::sort(vals.begin(), vals.end());
                    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
                    if (loud) { max_loud = std::max(max_loud, (int) vals.size()); if ((int) vals.size() > 16) ++over16; }
                    else { max_quiet = std::max(max_quiet, (int) vals.size()); if ((int) vals.size() > 4) ++over4; }
                }
            }
        check(over16 == 0 && over4 == 0, "every channel takes its own codebook: 16 magnitudes per group for a "
                                         "loud one, 4 for a quiet one (no plane reads the other's codes)");
        check(max_loud > 4, "the 16-level plane is live (a loud channel uses more than the 4-level grid's "
                            "magnitudes - the split is not a label)");
        std::printf("  q2t planes: a loud channel takes up to %d magnitudes per group, a quiet one up to %d "
                    "(16 and 4 are the codebooks)\n", max_loud, max_quiet);
    }
    // The fixed 16-level grid against a large-sample Lloyd-Max fit: the table in kv_q2.hpp is a constant, so
    // the gate is where it is checked. 1<<20 Gaussian samples, Max-Lloyd on the samples, up to 60 iterations,
    // stopped when the levels stop moving - a fit that has not converged is not the grid, and the gap would be
    // the fit's, not the table's, which is exactly the reading this check must not allow.
    {
        // THE FIT GETS ITS OWN SEEDED GENERATOR. It used to draw from the main cache rng, whose state at this
        // point depends on how many draws the cache generator consumed before it - which is different between
        // the synthetic arm (N = 1024 cells) and a real-cache dump (N = whatever the dump has). So the SAME
        // table was judged against a DIFFERENT 1M-sample fit in the two runs, and the same table read 0.100%
        // on synthetic and 4.349% on layer 3. A check whose sample changes between runs is not a check: the
        // number moves and nothing about the table moved. Seeded constant, so "the table is the fit" is
        // judged against the same fit every run, on every cache.
        std::mt19937 fit_rng(20260713u);
        std::normal_distribution<float> fit_nd(0.f, 1.f);
        std::vector<float> samp((size_t) 1 << 20);
        for (auto& x : samp) x = fit_nd(fit_rng);
        std::sort(samp.begin(), samp.end());
        const int L = 16;
        std::vector<float> lv(L);
        for (int j = 0; j < L; ++j)
            lv[j] = -4.0f + 8.0f * (float) j / (float) (L - 1);
        int it = 0;
        float delta = 1e30f;
        for (; it < 500; ++it) {
            double sum[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            int cnt[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            int j = 0;
            for (const float x : samp) {
                while (j + 1 < L && x > 0.5f * (lv[j] + lv[j + 1])) ++j;
                sum[j] += (double) x;
                cnt[j]++;
            }
            delta = 0.0f;
            for (int m = 0; m < L; ++m) {
                if (!cnt[m]) continue;
                const float n = (float) (sum[m] / (double) cnt[m]);
                delta = std::max(delta, std::fabs(n - lv[m]));
                lv[m] = n;
            }
            if (delta < 1e-5f) break;   // converged: the fit is the fixed point, not a walk towards it
        }
        double worst = 0.0;
        int worst_m = 0;
        for (int m = 0; m < 8; ++m) {
            const float fit = std::fabs(lv[8 + m]);
            const float tab = k::q2_lm16_pos(m);
            const double rel = (double) (std::fabs(fit - tab) / std::max(tab, 1e-6f));
            if (rel > worst) { worst = rel; worst_m = m; }
        }
        const bool table_is_fit = worst < 0.01;
        check(table_is_fit, "the fixed 16-level table IS the Gaussian Lloyd-Max grid (a 1M-sample fit lands on "
                            "it within 1%)");
        // Printed with the verdict and the two rows side by side, levels AND boundaries: a bare "worst gap
        // 27.783%" line says nothing about WHICH level is off, and the check that owns this number is the one
        // above - it is wired to the failure counter, so a gap here fails the run.
        std::printf("  q2t codebook: the fixed grid vs a 1M-sample Lloyd-Max fit (%d iterations, last level move "
                    "%.5f), worst relative gap %.3f%% at level %d - %s\n", it, delta, 100 * worst, worst_m,
                    table_is_fit ? "the table is the fit" : "THE TABLE IS NOT THE FIT");
        std::printf("  q2t levels (sigma)  table:");
        for (int m = 0; m < 8; ++m) std::printf(" %7.4f", k::q2_lm16_pos(m));
        std::printf("\n  q2t levels (sigma)    fit :");
        for (int m = 0; m < 8; ++m) std::printf(" %7.4f", std::fabs(lv[8 + m]));
        std::printf("\n  q2t boundaries (sigma) table:");
        const float tab_t[7] = {k::kQ2Lm16T1, k::kQ2Lm16T2, k::kQ2Lm16T3, k::kQ2Lm16T4,
                                k::kQ2Lm16T5, k::kQ2Lm16T6, k::kQ2Lm16T7};
        for (int m = 0; m < 7; ++m) std::printf(" %7.4f", tab_t[m]);
        std::printf("\n  q2t boundaries (sigma) fit :");
        for (int m = 0; m < 7; ++m) std::printf(" %7.4f", 0.5f * (std::fabs(lv[8 + m]) + std::fabs(lv[9 + m])));
        std::printf("\n");
        // The thresholds, pinned against the FIT's boundaries - which is the property that matters and the one
        // the header claims. A Lloyd-Max fixed point has its boundaries at the midpoints of its own levels, so
        // the fit's midpoints ARE its boundaries; the table's thresholds are judged against THOSE. The pin is
        // 2%, not 0.1%: the table's levels and thresholds were read off the grid at different precisions, and
        // the two densest boundaries (between levels 4/5 and 5/6, where the Voronoi cells are narrowest) sit
        // ~0.5% and ~1% off the midpoints of the table's own levels. A boundary error of 1% costs a fraction of
        // a percent of MSE - it moves the level a value nearest to for values within 1% of the boundary - so
        // the honest claim is "the thresholds are the fit's boundaries to a percent", and the number is
        // printed, not hidden behind a pin tuned to it.
        double worst_t = 0.0, worst_mid = 0.0;
        int worst_tm = 0;
        for (int m = 0; m < 7; ++m) {
            const float mid = 0.5f * (k::q2_lm16_pos(m) + k::q2_lm16_pos(m + 1));
            const float fitb = 0.5f * (std::fabs(lv[8 + m]) + std::fabs(lv[9 + m]));
            const double g_fit = (double) (std::fabs(fitb - tab_t[m]) / std::max(fitb, 1e-6f));
            const double g_mid = (double) (std::fabs(mid - tab_t[m]) / std::max(mid, 1e-6f));
            if (g_fit > worst_t) { worst_t = g_fit; worst_tm = m; }
            worst_mid = std::max(worst_mid, g_mid);
        }
        check(worst_t < 0.02, "the 16-level thresholds are the FIT's decision boundaries (within 2%) - the "
                              "codebook codes values into the level they are nearest to");
        std::printf("  q2t boundaries: table vs the fit's boundaries, worst %.3f%% at T%d; vs the midpoints of "
                    "the table's own levels, worst %.3f%% (a Lloyd-Max fixed point has them equal; this grid was "
                    "read off at finite precision, and the gap is printed, not pinned away)\n",
                    100.0 * worst_t, worst_tm + 1, 100.0 * worst_mid);
    }
    const int B2T = (int) k::kv_q2t_bytes_per_head(D);
    std::printf("  %-34s %7.3f%%  %4d B   K-top1/4@16b + V4 off the DEVICE pools: --kv q2t (q2t append for K, q4 for V)\n",
                "K-top1/4@16b + V4 (device)", 100 * e_t_dev, B2T + B4);
    std::printf("  q2t (device): %.3f%% against q2k's %.3f%% (%.2fx better) and q4_0's %.3f%% (%.2fx worse), at %d B "
                "per cell and head; per element K %.3f%% (q2c's %.3f%%), V %.3f%% (q4_0's %.3f%%)\n",
                100 * e_t_dev, 100 * e_k2v4_dev, e_k2v4_dev / e_t_dev, 100 * e_q4, e_t_dev / e_q4, B2T + B4,
                100 * r_k_t, 100 * r_k_kivi, 100 * r_v_t, 100 * r_v4);
    std::printf("  q2t byte math: K %d B per cell and head (192 quiet slots of 10 B + 64 loud slots of 18 B per "
                "32-cell group) + V %d = %d; %d per cell for %d KV heads (q2k %d, q4_0 %d)\n",
                B2T, B4, B2T + B4, (B2T + B4) * (int) s.n_head_kv, (int) s.n_head_kv, B2 + B4, 2 * B4);
    check(e_ac > 0.0 && e_jj > 0.0, "the published quantizers are live in this probe");
    // Measured, and it is the largest single lever found in this work: the ZERO-POINT, not the bit count.
    // Same 2 bits, same per-channel group; a min/max scale with an offset instead of a symmetric RMS scale
    // moves the attention output 7.150% -> 3.777% for 12% more bytes (96 B against 80 B per cell and head).
    check(e_ac < e_lm4, "the asymmetric zero-point quantizer beats the symmetric RMS block on the attention output");
    // Clipping is LAYER-dependent on real data: it helps on layers 11 and 47, not on 3 and 23. Pinned as a
    // synthetic-only check; printed with the layer named so the reversal is visible instead of fatal.
    if (!real_cache)
        check(e_acatc < e_acat, "SYNTHETIC: clipping the group extremes helps (SKVQ's clipping)");
    std::printf("  clipping (layer %d): clipped %.3f%% vs unclipped %.3f%% -> %s\n", probe_layer,
                100 * e_acatc, 100 * e_acat, e_acatc < e_acat ? "clipping helps" : "clipping does not help");
    // Joint scales: on i.i.d. Gaussian noise a channel factor is estimator noise from 32 samples, multiplied
    // into every code, so it loses. On a real cache it depends on the layer: it ties the best arm on layer 3
    // and loses on layer 47. Same treatment - synthetic-only check, measured print everywhere.
    if (!real_cache)
        check(e_jj > e_lm4, "SYNTHETIC: joint scales lose - with no channel structure the channel factor is estimator noise");
    std::printf("  joint scales (layer %d): joint %.3f%% vs per-channel %.3f%% -> %s\n", probe_layer,
                100 * e_jj, 100 * e_lm4, e_jj <= e_lm4 ? "joint does not lose" : "joint loses");
    std::printf("  zero-point lever: %.3f%% against the shipping symmetric block's %.3f%% (%.2fx, 96 B vs 80 B); "
                "clipping %.3f%%; joint scales %.3f%%\n",
                100 * e_ac, 100 * e_lm4, e_lm4 / e_ac, 100 * e_acatc, 100 * e_jj);
    check(e_k2v4 > 0.0 && e_k1v1 > 0.0, "the asymmetric arms are live (the probe responds to K and V separately)");
    // The device arm IS the host arm's format, run through the engine's own writer and reader. The two differ
    // only in which path quantized K - the host reference against the q2 group kernel - and check 1 says those
    // two agree bit for bit, so the rows have to land on top of each other. A gap here means the q2k dispatch
    // is not the pair of formats it claims to be (a V read out of the q2 pool, a K read out of the q4 one).
    check(e_k2v4_dev > 0.0, "the q2k device arm is live (the engine's appends filled both halves)");
    check(e_k2v4_dev < 1.02 * e_k2v4 && e_k2v4_dev > 0.98 * e_k2v4,
          "the q2k device arm lands on the host K2c + V4 row (within 2% - same bytes, same readers)");
    check(e_k2v4_dev < e_lm4, "q2k beats the shipping symmetric 2-bit cache on the attention output (its reason to exist)");
    check(e_k2v4_dev > e_q4, "q2k is lossier than q4_0 (it is the cheaper format, not the better one)");
    // The device-pool twin of the host-side claim above, and it is data-scoped for the same reason: on the
    // synthetic cache the K side is the cheap one to coarsen, on a real cache the V side is.
    check_data(real_cache, e_k4v2 < e_k2v4_dev, e_k2v4_dev < e_k4v2,
               "SYNTHETIC: spending the bits on K beats spending them on V, on the device pools too",
               "REAL CACHE: spending the bits on V beats spending them on K, on the device pools too");
    std::printf("  q2k (device): %.3f%% against q2's %.3f%% (%.2fx better) and q4_0's %.3f%% (%.2fx worse), at %d B "
                "per cell and head; per element K %.3f%% (q2's %.3f%%), V %.3f%% (q4_0's %.3f%%)\n",
                100 * e_k2v4_dev, 100 * e_lm4, e_lm4 / e_k2v4_dev, 100 * e_q4, e_k2v4_dev / e_q4, B2 + B4,
                100 * r_k2k, 100 * r_k2, 100 * r_v2k, 100 * r_v4);
    // Measured, and the two data sources disagree on three of these four - which is the finding. KIVI's
    // placement claims (arXiv 2402.02750) rest on channel outliers; the synthetic cache has none, the real
    // rotated cache has them (K max/median channel RMS 5.8x, V 13.6x), and the V-side claims flip.
    check_data(real_cache, e_k2c2t < e_lm4, e_k2c2t >= e_lm4,
               "SYNTHETIC: KIVI's split (K per-channel, V per-token) beats symmetric per-channel",
               "REAL CACHE: KIVI's split loses - rotation left V's channel structure in place, so V wants a per-channel scale too");
    // K's scale axis, measured with V held per-channel in BOTH arms (the earlier phrasing compared
    // K2t+V2c against K2c+V2t, which moves both tensors' scale axes at once and measures V's channel
    // structure, not K's). Per-channel K wins on the synthetic cache and on real layers 3/11/23; on layer
    // 47, where K's channel RMS spans 74.9x, per-token K ties it - K's error is largely common-mode under
    // the softmax, so a per-channel scale buys less as the channel spread grows.
    if (!real_cache)
        check(e_k2t2c > e_lm4, "SYNTHETIC: a per-TOKEN K scale is the lossy side - K is the tensor that needs per-channel scaling");
    std::printf("  K scale axis (V per-channel in both arms, layer %d): per-token K %.3f%% vs per-channel K %.3f%% -> %s\n",
                probe_layer, 100 * e_k2t2c, 100 * e_lm4,
                e_k2t2c > e_lm4 ? "per-channel K wins" : "per-token K ties or wins");
    check_data(real_cache, e_k2v1t < e_k2v1c, e_k2v1t >= e_k2v1c,
               "SYNTHETIC: 1-bit V wants a per-token scale (KIVI's V finding)",
               "REAL CACHE: 1-bit V wants the per-channel scale (V's channel outliers survive the rotation)");
    // The condition is "K4 + V2c beats K2c + V4" (synthetic) / "K2c + V4 beats K4 + V2c" (real): whichever
    // tensor's coarse arm costs MORE is the sensitive one. On the synthetic cache that is K; on the real
    // rotated cache it is V - the ranking --kv q2k is built on.
    check_data(real_cache, e_k4v2 < e_k2v4, e_k4v2 >= e_k2v4,
               "SYNTHETIC: the K side is the sensitive one (coarse K costs more than coarse V: K4+V2c beats K2c+V4)",
               "REAL CACHE: the V side is the sensitive one (coarse V costs more than coarse K: K2c+V4 beats K4+V2c) - the reverse of the synthetic ranking");
    std::printf("  asymmetry: KIVI's split %.3f%% vs symmetric per-channel %.3f%% (%.2fx, same 160 B); "
                "the best 224 B arm is %.3f%% against q4_0's %.3f%% - asymmetry narrows the gap, it does not close it\n",
                100 * e_k2c2t, 100 * e_lm4, e_lm4 / e_k2c2t, 100 * std::min(e_k4v2, e_k2v4), 100 * e_q4);
    // The probe's own limit, stated where it can be seen: the gap between 1-bit both and 2-bit both is small
    // on the synthetic cache and large on a real one, because a diffuse softmax averaging over N cells washes
    // coarse quantization away. This table RANKS schemes against each other; it is not an absolute quality
    // bar. The absolute bar is the logit-level harness (bench/results/2026-10-12-kv-precision).
    std::printf("  probe limit: 1-bit both (%d B) %.3f%% vs 2-bit both (%d B) at %.3f%% (%.2fx) over %d cells\n"
                "  - a diffuse softmax averages coarse quantization away, so this table RANKS schemes;\n"
                "  the absolute bar is the logit-level KL/top-1 table.\n",
                2 * B1c, 100 * e_k1v1, 2 * B2, 100 * e_lm4, e_k1v1 / e_lm4, N);
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
    check_data(real_cache, e_lm16 > e_lmt, e_lm16 <= e_lmt,
               "SYNTHETIC: the 16-value scale does not beat 32 on the attention output (estimator noise from 16 samples)",
               "REAL CACHE: the 16-value scale does beat 32 (a finer scale tracks the channel structure) - the block size is a data question, not a law");
    check(e_lm16 > e_lm4, "the 16-value scale does not beat the shipping per-channel pools either");
    check(e_lm4 > 0.0 && e_q4 > 0.0, "both quantized caches move the attention output (the probe is live)");
    check(e_lm4 < e_uni4 && e_lm3 < e_uni3, "the Lloyd-Max codebook moves the attention output less than the grid");

    // 4. the byte math the design doc quotes
    std::printf("\nBytes per cell and KV head: fp16 %d, q4_0 %llu, q2 %llu (head_dim %d)\n", 2 * D,
                (unsigned long long) k::kv_q4_bytes_per_head(D), (unsigned long long) k::kv_q2_bytes_per_head(D), D);
    check(k::kv_q2_bytes_per_head(D) == 80, "q2 is 80 B per cell and head (8 blocks x 10 B)");
    check(k::kv_q2_bytes_per_cell(s) == 2 * 2 * 80, "q2 is 320 B per cell for 2 KV heads (vs 576 for q4_0)");
    // q2k's byte math, pinned from the two widths the format is made of: 80 (K, q2 per-channel) + 144 (V, q4_0)
    // = 224 per cell and KV head, 448 per cell for 2 KV heads - 40% over q2's 320, 22% under q4_0's 576.
    const int B2K = B2 + B4;
    std::printf("Bytes per cell and KV head: q2k %d (q2 K %d + q4_0 V %d); per cell for %d KV heads: q2k %d, "
                "q2 %d, q4_0 %d\n",
                B2K, B2, B4, (int) s.n_head_kv, B2K * (int) s.n_head_kv, (int) k::kv_q2_bytes_per_cell(s),
                (int) k::kv_q4_bytes_per_cell(s));
    check(B2K == 224, "q2k is 224 B per cell and KV head (80 q2 + 144 q4_0)");
    check(B2K * (int) s.n_head_kv == 448, "q2k is 448 B per cell for 2 KV heads (q2: 320, q4_0: 576)");
    check(k::kv_stream_unit_blocks(s, k::kKvQ2K) == (int) (32 / s.page_size),
          "q2k's residency unit is the q2 group (32/page_size blocks): its K scale spans the group");
    check(k::kv_block_bytes(s, k::kKvQ2K) ==
              (uint64_t) (s.n_head_kv * s.page_size) * (k::kv_q2_bytes_per_head(D) + k::kv_q4_bytes_per_head(D)),
          "the q2k block mover moves K's q2 run and V's q4_0 run (224 B per cell and head)");
    // q2t's byte math, pinned from the two planes the K side is made of: (head_dim - head_dim/4) quiet slots of
    // 10 B + head_dim/4 loud slots of 18 B per 32-cell group and head = 96 B per cell and head at D=256, plus
    // V's 144 = 240, 480 per cell for 2 KV heads (q2k 224/448, q4_0 288/576). B2T is already declared above.
    std::printf("Bytes per cell and KV head: q2t %d (q2t K %d + q4_0 V %d); per cell for %d KV heads: q2t %d, "
                "q2k %d, q4_0 %d\n",
                B2T + B4, B2T, B4, (int) s.n_head_kv, (B2T + B4) * (int) s.n_head_kv, B2K * (int) s.n_head_kv,
                (int) k::kv_q4_bytes_per_cell(s));
    check(B2T == (int) ((k::q2t_quiet(D) * sizeof(k::block_q2c_0) + k::q2t_loud(D) * sizeof(k::block_q2l_0)) / 32),
          "q2t's K width is the two planes' slot bytes over a 32-cell group (10 B quiet, 18 B loud)");
    check(B2T == 96, "q2t's K is 96 B per cell and head at head_dim 256 (192 quiet slots + 64 loud ones)");
    check(B2T + B4 == 240, "q2t is 240 B per cell and KV head (96 K + 144 q4_0 V)");
    check((B2T + B4) * (int) s.n_head_kv == 480, "q2t is 480 B per cell for 2 KV heads (q2k: 448, q4_0: 576)");
    check(k::q2t_loud(D) == D / 4 && k::q2t_quiet(D) == 3 * D / 4,
          "the loud set is head_dim/4 channels per KV head, the quiet set the rest (the gate's winning fraction)");
    check(k::q2t_level_words(H, D) * 4 == (size_t) H * D / 8,
          "the level bitmap is n_head_kv * head_dim bits = 64 B per layer, stored once, not per cell");
    check(k::kv_stream_unit_blocks(s, k::kKvQ2T) == (int) (32 / s.page_size),
          "q2t's residency unit is the q2 group (32/page_size blocks): the loud plane's scale spans the group too");
    check(k::kv_block_bytes(s, k::kKvQ2T) ==
              (uint64_t) (s.n_head_kv * s.page_size) * (k::kv_q2t_bytes_per_head(D) + k::kv_q4_bytes_per_head(D)),
          "the q2t block mover moves K's mixed-level run and V's q4_0 run (240 B per cell and head)");
    check(k::kv_stream_unit_blocks(s, k::kKvQ2) == k::kv_stream_unit_blocks(s, k::kKvQ2T),
          "q2t streams with q2's unit, so a q2t ring is rounded to whole groups exactly like q2's and q2k's");

    // 5. A FRESH PARTIAL GROUP'S SCALE. A group whose first cells arrive in a batch shorter than 32 has no
    //    stored scale to keep, so the kernel takes the RMS of the cells the call has - of the 12 it has, not of
    //    32. Dividing a 12-cell sum by 32 under-scales the group by sqrt(32/12) = 1.63x, and every code in it
    //    is then quantized against a scale 1.63x too small. This is the real prompt-path append (use_prev = 0,
    //    a zeroed pool, so the group is fresh and the kernel writes its own scale), read back as stored bits;
    //    the expected bits are the host sum in cell order with the multiply and the add kept separate, for the
    //    reason quant_kivi states - an ulp in the scale is a different code at a threshold.
    if (N >= 12) {
        const int n12 = 12;
        const size_t pool12 = (size_t) H * D * sizeof(k::block_q2c_0);   // group 0's slots, every head and dim
        uint8_t* kq2_12 = dalloc<uint8_t>(pool12);                       // zeroed: a scale of 0 bits = no scale
        std::vector<int32_t> tab12(P);                                    // cells 0..11 = pages 0..2, identity
        for (int i = 0; i < P; ++i) tab12[i] = i;
        int32_t* d_tab12 = dalloc<int32_t>(P);
        ck(cudaMemcpy(d_tab12, tab12.data(), P * 4, cudaMemcpyHostToDevice), "12-cell table");
        k::kv_append_q2(kq2_12, kq2_12, d_tab12, 0, n12, dK, dK, s, nullptr, nullptr, nullptr, /*v3level*/ false);
        ck(cudaDeviceSynchronize(), "12-cell append");
        std::vector<uint8_t> h12(pool12);
        ck(cudaMemcpy(h12.data(), kq2_12, pool12, cudaMemcpyDeviceToHost), "12-cell pool");
        bool scale_ok = true;
        for (int h = 0; h < H && scale_ok; ++h)
            for (int d = 0; d < D && scale_ok; ++d) {
                float ss = 0.0f;
                for (int i = 0; i < n12; ++i) {
                    const float v = Kref[(size_t) i * H * D + (size_t) h * D + d];
                    ss += v * v;
                }
                if (k::q2c_slot(h12.data(), 0, h, d, H, D)->d !=
                    k::f16_from_f32(sqrtf(ss * (1.0f / (float) n12))))
                    scale_ok = false;
            }
        check(scale_ok, "fresh partial group scale: a 12-cell append into an empty group stores the RMS of the 12 "
                        "cells the call has, bit for bit (not of 32)");
    } else {
        std::printf("  fresh partial group scale: NOT RUN - the probe cache has %d cells, fewer than the 12 the\n"
                    "  check appends\n", N);
    }

    // 5b. A FRESH PARTIAL GROUP AFTER A COMPLETE ONE. A batch append that starts a group fresh has no stored
    //     scale to keep, and in a batch the previous group may be written by THIS launch, so the kernel builds
    //     the previous group's scale race-free and shrinks the new one toward it: the full-group RMS with the
    //     previous group's mean square standing in for the cells not yet written,
    //     s = sqrtf((ss1 + (32 - 12) * prev * prev) / 32).
    //     (a) appends 32 cells and then 12: group 1 is fresh in the SECOND call, and group 0's scale is the
    //     stored one from the first call. (b) appends all 44 in ONE call: group 0 is complete in the SAME
    //     launch, so the kernel computes its f16 scale from the call's cells instead of reading the pool. The
    //     two arms must store the same bits, and those bits must be the host value built in the kernel's op
    //     order (separate multiply and add, as the check above).
    if (N >= 44) {
        const int n44 = 44;
        const int pages48 = (48 + P - 1) / P;                          // 48 cells, identity
        std::vector<int32_t> tab48(pages48);
        for (int i = 0; i < pages48; ++i) tab48[i] = i;
        int32_t* d_tab48 = dalloc<int32_t>(pages48);
        ck(cudaMemcpy(d_tab48, tab48.data(), pages48 * 4, cudaMemcpyHostToDevice), "44-cell table");
        const size_t pool48 = (size_t) 2 * H * D * sizeof(k::block_q2c_0);   // groups 0 and 1, every head and dim
        uint8_t* kq2_a = dalloc<uint8_t>(pool48);                     // zeroed: a scale of 0 bits = no scale
        uint8_t* kq2_b = dalloc<uint8_t>(pool48);
        k::kv_append_q2(kq2_a, kq2_a, d_tab48, 0, 32, dK, dK, s, nullptr, nullptr, nullptr, /*v3level*/ false);
        k::kv_append_q2(kq2_a, kq2_a, d_tab48, 32, 12, dK + (size_t) 32 * H * D, dK + (size_t) 32 * H * D, s,
                        nullptr, nullptr, nullptr, /*v3level*/ false);
        k::kv_append_q2(kq2_b, kq2_b, d_tab48, 0, n44, dK, dK, s, nullptr, nullptr, nullptr, /*v3level*/ false);
        ck(cudaDeviceSynchronize(), "44-cell appends");
        std::vector<uint8_t> ha(pool48), hb(pool48);
        ck(cudaMemcpy(ha.data(), kq2_a, pool48, cudaMemcpyDeviceToHost), "44-cell pool a");
        ck(cudaMemcpy(hb.data(), kq2_b, pool48, cudaMemcpyDeviceToHost), "44-cell pool b");
        bool shrink_ok = true;
        for (int h = 0; h < H && shrink_ok; ++h)
            for (int d = 0; d < D && shrink_ok; ++d) {
                float ss0 = 0.0f, ss1 = 0.0f;
                for (int i = 0; i < 32; ++i) {
                    const float v = Kref[(size_t) i * H * D + (size_t) h * D + d];
                    ss0 += v * v;
                }
                for (int i = 32; i < n44; ++i) {
                    const float v = Kref[(size_t) i * H * D + (size_t) h * D + d];
                    ss1 += v * v;
                }
                const float prev = k::f32_from_f16(k::f16_from_f32(sqrtf(ss0 * (1.0f / 32.0f))));
                const float pp = prev * prev;
                const float t = 20.0f * pp;
                const float u = ss1 + t;
                const float w = u * (1.0f / 32.0f);
                const uint16_t want = k::f16_from_f32(sqrtf(w));
                const uint16_t bits_a = k::q2c_slot(ha.data(), 1, h, d, H, D)->d;
                if (bits_a != want || k::q2c_slot(hb.data(), 1, h, d, H, D)->d != bits_a) shrink_ok = false;
            }
        check(shrink_ok, "fresh partial group after a complete one shrinks toward the previous scale: a 12-cell "
                         "batch append into a fresh group stores the full-group RMS with the previous group's "
                         "mean square for the 20 cells not yet written, bit for bit, and the same bits whether "
                         "the previous group was in this launch or the last");
        // --kv-k-clip: the same two appends with alpha = 0.8 on K. The stored scale is alpha x the rule, the rule
        // runs in the raw domain (the stored previous scale divided by alpha first), and the codes are chosen
        // against the stored (clipped) scale. Host mirror: separate float ops in the kernel's order.
        {
            const float a = 0.8f, inv_a = 1.0f / a;
            uint8_t* kq2_c = dalloc<uint8_t>(pool48);
            uint8_t* kq2_d = dalloc<uint8_t>(pool48);
            k::kv_append_q2(kq2_c, kq2_c, d_tab48, 0, 32, dK, dK, s, nullptr, nullptr, nullptr, false, a);
            k::kv_append_q2(kq2_c, kq2_c, d_tab48, 32, 12, dK + (size_t) 32 * H * D, dK + (size_t) 32 * H * D, s,
                            nullptr, nullptr, nullptr, false, a);
            k::kv_append_q2(kq2_d, kq2_d, d_tab48, 0, n44, dK, dK, s, nullptr, nullptr, nullptr, false, a);
            ck(cudaDeviceSynchronize(), "clipped 44-cell appends");
            std::vector<uint8_t> hc(pool48), hd(pool48);
            ck(cudaMemcpy(hc.data(), kq2_c, pool48, cudaMemcpyDeviceToHost), "clipped pool c");
            ck(cudaMemcpy(hd.data(), kq2_d, pool48, cudaMemcpyDeviceToHost), "clipped pool d");
            bool clip_ok = true, codes_ok = true, differs = false;
            for (int h = 0; h < H && clip_ok; ++h)
                for (int d = 0; d < D && clip_ok; ++d) {
                    float ss0 = 0.0f, ss1 = 0.0f;
                    for (int i = 0; i < 32; ++i) {
                        const float v = Kref[(size_t) i * H * D + (size_t) h * D + d];
                        ss0 += v * v;
                    }
                    for (int i = 32; i < n44; ++i) {
                        const float v = Kref[(size_t) i * H * D + (size_t) h * D + d];
                        ss1 += v * v;
                    }
                    const float s0 = sqrtf(ss0 * (1.0f / 32.0f));
                    const uint16_t want0 = k::f16_from_f32(a * s0);
                    const float prev_raw = k::f32_from_f16(want0) * inv_a;
                    const float pp = prev_raw * prev_raw;
                    const float t = 20.0f * pp;
                    const float u = ss1 + t;
                    const float w = u * (1.0f / 32.0f);
                    const uint16_t want1 = k::f16_from_f32(a * sqrtf(w));
                    const k::block_q2c_0* c0 = k::q2c_slot(hc.data(), 0, h, d, H, D);
                    const k::block_q2c_0* c1 = k::q2c_slot(hc.data(), 1, h, d, H, D);
                    if (c0->d != want0 || c1->d != want1 || k::q2c_slot(hd.data(), 0, h, d, H, D)->d != want0 ||
                        k::q2c_slot(hd.data(), 1, h, d, H, D)->d != want1)
                        clip_ok = false;
                    if (want0 != k::f16_from_f32(s0)) differs = true;   // the clip actually moved the scale
                    for (int i = 0; i < 32; ++i) {                     // group 0's codes vs the CLIPPED scale
                        // the kernel picks codes against the unrounded float scale (a x the rule); the reader
                        // dequantizes with the stored fp16 bits - as for every q2 code
                        const int want_code = k::q2_code_lm4(Kref[(size_t) i * H * D + (size_t) h * D + d],
                                                             a * s0);
                        if (((c0->qs[i / 4] >> (2 * (i % 4))) & 3) != want_code) codes_ok = false;
                    }
                }
            check(clip_ok && codes_ok && differs,
                  "--kv-k-clip 0.8: the stored K scale is alpha x the rule (complete group) and alpha x the raw-domain "
                  "shrink (fresh partial group), bit for bit, the same bits in one call or two, and the codes are "
                  "chosen against the clipped scale");
            cudaFree(kq2_c);
            cudaFree(kq2_d);
        }
        // K reconstruction gain (a NEGATIVE k_clip, beta = 1.1): the same two appends. The stored scale is beta x
        // the rule exactly as the clip computes it (raw-domain shrink for the fresh partial), but the codes are
        // chosen against sc = (beta x rule) x (1/beta) - the code assignment of an UN-gained append, with only the
        // reconstruction scaled up. Host mirror: separate float ops in the kernel's order.
        {
            const float a = 1.1f, inv_a = 1.0f / a;
            uint8_t* kq2_e = dalloc<uint8_t>(pool48);
            uint8_t* kq2_f = dalloc<uint8_t>(pool48);
            k::kv_append_q2(kq2_e, kq2_e, d_tab48, 0, 32, dK, dK, s, nullptr, nullptr, nullptr, false, -1.1f);
            k::kv_append_q2(kq2_e, kq2_e, d_tab48, 32, 12, dK + (size_t) 32 * H * D, dK + (size_t) 32 * H * D, s,
                            nullptr, nullptr, nullptr, false, -1.1f);
            k::kv_append_q2(kq2_f, kq2_f, d_tab48, 0, n44, dK, dK, s, nullptr, nullptr, nullptr, false, -1.1f);
            ck(cudaDeviceSynchronize(), "gain 44-cell appends");
            std::vector<uint8_t> he(pool48), hf(pool48);
            ck(cudaMemcpy(he.data(), kq2_e, pool48, cudaMemcpyDeviceToHost), "gain pool e");
            ck(cudaMemcpy(hf.data(), kq2_f, pool48, cudaMemcpyDeviceToHost), "gain pool f");
            bool gain_ok = true, gcodes_ok = true, differs = false;
            int ties = 0;
            for (int h = 0; h < H && gain_ok; ++h)
                for (int d = 0; d < D && gain_ok; ++d) {
                    float ss0 = 0.0f, ss1 = 0.0f;
                    for (int i = 0; i < 32; ++i) {
                        const float v = Kref[(size_t) i * H * D + (size_t) h * D + d];
                        ss0 += v * v;
                    }
                    for (int i = 32; i < n44; ++i) {
                        const float v = Kref[(size_t) i * H * D + (size_t) h * D + d];
                        ss1 += v * v;
                    }
                    const float s0 = sqrtf(ss0 * (1.0f / 32.0f));
                    const uint16_t want0 = k::f16_from_f32(a * s0);
                    const float prev_raw = k::f32_from_f16(want0) * inv_a;
                    const float pp = prev_raw * prev_raw;
                    const float t = 20.0f * pp;
                    const float u = ss1 + t;
                    const float w = u * (1.0f / 32.0f);
                    const uint16_t want1 = k::f16_from_f32(a * sqrtf(w));
                    const k::block_q2c_0* c0 = k::q2c_slot(he.data(), 0, h, d, H, D);
                    const k::block_q2c_0* c1 = k::q2c_slot(he.data(), 1, h, d, H, D);
                    if (c0->d != want0 || c1->d != want1 || k::q2c_slot(hf.data(), 0, h, d, H, D)->d != want0 ||
                        k::q2c_slot(hf.data(), 1, h, d, H, D)->d != want1)
                        gain_ok = false;
                    if (want0 != k::f16_from_f32(s0)) differs = true;   // the gain moved the stored scale
                    const float sc0 = (a * s0) * inv_a;                 // the UN-gained scale the codes use
                    for (int i = 0; i < 32; ++i) {
                        const float v = Kref[(size_t) i * H * D + (size_t) h * D + d];
                        const int want_code = k::q2_code_lm4(v, sc0);
                        if (((c0->qs[i / 4] >> (2 * (i % 4))) & 3) != want_code) gcodes_ok = false;
                        if (k::q2_code_lm4(v, s0) != want_code) ++ties;   // exact-threshold ties, printed only
                    }
                    const float sc1 = (a * sqrtf(w)) * inv_a;
                    for (int i = 0; i < 12; ++i) {
                        const float v = Kref[(size_t) (32 + i) * H * D + (size_t) h * D + d];
                        if (((c1->qs[i / 4] >> (2 * (i % 4))) & 3) != k::q2_code_lm4(v, sc1)) gcodes_ok = false;
                    }
                }
            check(gain_ok && gcodes_ok && differs,
                  "K reconstruction gain 1.1 (negative k_clip): the stored K scale is beta x the rule (complete "
                  "group) and beta x the raw-domain shrink (fresh partial group), bit for bit, in one call or two, "
                  "and the codes are chosen against the UN-gained scale");
            std::printf("  K reconstruction gain 1.1: %d group-0 codes differ from the un-gained choice "
                        "(exact-threshold ties)\n", ties);
            cudaFree(kq2_e);
            cudaFree(kq2_f);
        }
    } else {
        std::printf("  fresh partial group after a complete one: NOT RUN - the probe cache has %d cells, fewer\n"
                    "  than the 44 the check appends\n", N);
    }

    // 6. STAGING MOVES WHOLE GROUPS. A q2 group's bytes are group-major (one slot per (group, head, dim)), so a
    //    PAGE is not a byte range: 3 of a group's 8 pages are spread over every dim slot of that group. Staging a
    //    block count that stops mid-group must copy the whole group (else most of its dim slots stay stale in the
    //    staging pool the prompt attention reads), and nothing past it; unstaging [b0, b1) inside a group must
    //    write the whole group back. Pinned on q2k, the format whose unit is set by its q2 K half.
    {
        const int64_t unit = k::kv_stream_unit_blocks(s, k::kKvQ2K);       // pages per 32-cell group
        const size_t kb = (size_t) (2 * unit) * s.n_head_kv * s.page_size * k::kv_q2_bytes_per_head(D);
        const size_t vb = (size_t) (2 * unit) * s.n_head_kv * s.page_size * k::kv_q4_bytes_per_head(D);
        uint8_t* hk = dalloc<uint8_t>(kb);
        uint8_t* hv = dalloc<uint8_t>(vb);
        uint8_t* sk = dalloc<uint8_t>(kb);
        uint8_t* sv = dalloc<uint8_t>(vb);
        std::vector<uint8_t> pk(kb), pv(vb), gk(kb), gv(vb);
        for (size_t i = 0; i < kb; ++i) pk[i] = (uint8_t) (i * 131u + 7u);
        for (size_t i = 0; i < vb; ++i) pv[i] = (uint8_t) (i * 197u + 3u);
        const auto zero_from = [](const std::vector<uint8_t>& b, size_t at) {
            return std::all_of(b.begin() + (std::ptrdiff_t) at, b.end(), [](uint8_t x) { return x == 0; });
        };
        k::KvHostPools host{};
        host.k_q2 = hk;
        host.v_q4 = hv;
        k::QsaAttnPools stage{};
        stage.k_q2 = sk;
        stage.v_q4 = sv;
        // stage 3 pages (mid-group) from a host copy holding a pattern; the staging pool starts zeroed
        ck(cudaMemcpy(hk, pk.data(), kb, cudaMemcpyHostToDevice), "stage: host K");
        ck(cudaMemcpy(hv, pv.data(), vb, cudaMemcpyHostToDevice), "stage: host V");
        k::kv_stage_from_host(stage, host, k::kKvQ2K, 3, s, nullptr);
        ck(cudaDeviceSynchronize(), "stage");
        ck(cudaMemcpy(gk.data(), sk, kb, cudaMemcpyDeviceToHost), "stage: read K");
        ck(cudaMemcpy(gv.data(), sv, vb, cudaMemcpyDeviceToHost), "stage: read V");
        const bool staged = std::memcmp(gk.data(), pk.data(), kb / 2) == 0 &&
                            std::memcmp(gv.data(), pv.data(), vb / 2) == 0 && zero_from(gk, kb / 2) &&
                            zero_from(gv, vb / 2);
        check(staged, "staging 3 of a group's 8 pages copies the WHOLE q2k group (every dim slot, K and V) and "
                      "nothing past it");
        // unstage pages [3, 5) - inside group 0 - from a staging pool holding the pattern into a zeroed host copy
        ck(cudaMemset(hk, 0, kb), "unstage: zero host K");
        ck(cudaMemset(hv, 0, vb), "unstage: zero host V");
        ck(cudaMemcpy(sk, pk.data(), kb, cudaMemcpyHostToDevice), "unstage: stage K");
        ck(cudaMemcpy(sv, pv.data(), vb, cudaMemcpyHostToDevice), "unstage: stage V");
        k::kv_unstage_to_host(stage, host, k::kKvQ2K, 3, 5, s, nullptr);
        ck(cudaDeviceSynchronize(), "unstage");
        ck(cudaMemcpy(gk.data(), hk, kb, cudaMemcpyDeviceToHost), "unstage: read K");
        ck(cudaMemcpy(gv.data(), hv, vb, cudaMemcpyDeviceToHost), "unstage: read V");
        const bool unstaged = std::memcmp(gk.data(), pk.data(), kb / 2) == 0 &&
                              std::memcmp(gv.data(), pv.data(), vb / 2) == 0 && zero_from(gk, kb / 2) &&
                              zero_from(gv, vb / 2);
        check(unstaged, "unstaging pages [3, 5) writes the WHOLE q2k group back to the host copy and nothing past "
                        "it");
    }

    std::printf("\n%s\n", g_fail == 0 ? "kv_q2_parity: all checks passed" : "kv_q2_parity: FAILURES");
    return g_fail == 0 ? 0 : 1;
}
