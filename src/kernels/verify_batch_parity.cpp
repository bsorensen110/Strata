// src/kernels/verify_batch_parity.cpp - the verify window's one-launch forms (the hyper-connection write, the PLE post-ops) against the per-token launches they
// replace, bitwise.
//
//     build/verify_batch_parity
//
// 1. native_gr_post_multi against n native_gr_post calls (the hyper-connection write before the PLE block).
// 2. native_ple_postops_batch_snap (STRATA_DF_PLE) against the per-token loop of the verify window: per token
//    native_ple_postops, ple_history_advance and a copy of the history (the commit's snapshot); the results, the final
//    history and every snapshot, for windows of 2..8 tokens.
// GPU, synthetic, no model.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/native_gr_postops.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
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
        std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

// The set-up copies and fills run on the legacy default stream, which does NOT order work on the non-blocking stream
// the kernels run on (and cudaMemset / a pageable cudaMemcpy may return before the device has finished them): each
// one is completed here, or a kernel could read its inputs - or have its outputs overwritten - while they land.
template <typename T>
T* dalloc(size_t n, int fill = 0) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, fill, n * sizeof(T) + 64), "memset");
    ck(cudaDeviceSynchronize(), "memset sync");
    return p;
}

template <typename T>
std::vector<T> read(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "read");
    return h;
}

template <typename T>
T* upload(const std::vector<T>& h) {
    T* d = dalloc<T>(h.size());
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    ck(cudaDeviceSynchronize(), "upload sync");
    return d;
}

void report(const char* what, bool ok) {
    std::printf("%-64s %s\n", what, ok ? "bitwise equal" : "DIFFERENT");
    if (!ok) ++g_fail;
}

// ---- 1. the hyper-connection write
void gr_post(cudaStream_t s) {
    const int N = 2560, HC = 4;
    std::mt19937 rng(12);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int n : {2, 4, 8}) {
        std::vector<float> r((size_t) n * HC * N), bo((size_t) n * N), inj((size_t) n * HC);
        for (auto& x : r) x = nd(rng);
        for (auto& x : bo) x = nd(rng);
        for (auto& x : inj) x = 3.f * nd(rng);
        float* r0 = upload(r);
        float* r1 = upload(r);
        float* d_bo = upload(bo);
        float* d_inj = upload(inj);
        for (int t = 0; t < n; ++t)
            k::native_gr_post(r0 + (size_t) t * HC * N, d_bo + (size_t) t * N, d_inj + (size_t) t * HC,
                              r0 + (size_t) t * HC * N, N, HC, s);
        k::native_gr_post_multi(r1, d_bo, d_inj, r1, N, HC, n, (long long) HC * N, N, HC, s);
        ck(cudaStreamSynchronize(s), "gr post");
        char what[128];
        std::snprintf(what, sizeof what, "native_gr_post_multi, %d tokens (in place)", n);
        report(what, read(r0, r.size()) == read(r1, r.size()));
        cudaFree(r0); cudaFree(r1); cudaFree(d_bo); cudaFree(d_inj);
    }
}

// ---- 2. the PLE post-ops of a window, with the commit's history snapshots
void ple(cudaStream_t s) {
    constexpr int N = k::NG_N_EMBD, D = k::NG_HC_DIM, HS = k::NG_HIST * k::NG_HC_DIM;
    std::mt19937 rng(13);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> nk(D), nq(D), nc(D);
    for (auto* v : {&nk, &nq, &nc})
        for (auto& x : *v) x = 1.f + 0.1f * nd(rng);
    std::vector<uint16_t> conv((size_t) 4 * D);
    for (auto& x : conv) x = k::f16_from_f32(0.3f * nd(rng));
    k::PleWeights w;
    w.norm_key = upload(nk);
    w.norm_query = upload(nq);
    w.norm_conv = upload(nc);
    w.conv1d_f16 = upload(conv);
    std::vector<float> hist0((size_t) HS);
    for (auto& x : hist0) x = nd(rng);
    for (int n : {2, 3, 4, 8}) {
        std::vector<float> key((size_t) n * D), hid((size_t) n * D), val((size_t) n * N);
        for (auto& x : key) x = 4.f * nd(rng);
        for (auto& x : hid) x = nd(rng);
        for (auto& x : val) x = nd(rng);
        // per token, as the verify window's loop: native_ple_postops (ple_block_projected's buffers), the history
        // advance and the snapshot copy
        float* key0 = upload(key);
        float* hid0 = upload(hid);
        float* val0 = upload(val);
        float* hist_a = upload(hist0);
        float* snap_a = dalloc<float>((size_t) n * HS);
        float *q = dalloc<float>(D), *nrm = dalloc<float>(D), *g = dalloc<float>(4), *gd = dalloc<float>(D);
        float* cv = dalloc<float>(D);
        for (int t = 0; t < n; ++t) {
            k::NativePlePostopsBuffers b{q, nrm, g, gd, nrm, cv, hid0 + (size_t) t * D};
            k::native_ple_postops(key0 + (size_t) t * D, hid0 + (size_t) t * D, val0 + (size_t) t * N, hist_a, w, b, s);
            k::ple_history_advance(hist_a, nrm, s);
            ck(cudaMemcpyAsync(snap_a + (size_t) t * HS, hist_a, (size_t) HS * 4, cudaMemcpyDeviceToDevice, s), "snap");
        }
        // the window's one pass
        float* key1 = upload(key);
        float* hid1 = upload(hid);
        float* val1 = upload(val);
        float* hist_b = upload(hist0);
        float* snap_b = dalloc<float>((size_t) n * HS);
        float *qn = dalloc<float>((size_t) n * D), *gated = dalloc<float>((size_t) n * D), *gate = dalloc<float>((size_t) n * 4);
        k::native_ple_postops_batch_snap(key1, hid1, val1, hist_b, w, qn, gated, gate, n, snap_b, s);
        ck(cudaStreamSynchronize(s), "ple");
        char what[128];
        std::snprintf(what, sizeof what, "native_ple_postops_batch_snap, %d tokens: results", n);
        report(what, read(hid0, hid.size()) == read(hid1, hid.size()));
        std::snprintf(what, sizeof what, "native_ple_postops_batch_snap, %d tokens: history", n);
        report(what, read(hist_a, (size_t) HS) == read(hist_b, (size_t) HS));
        std::snprintf(what, sizeof what, "native_ple_postops_batch_snap, %d tokens: snapshots", n);
        report(what, read(snap_a, (size_t) n * HS) == read(snap_b, (size_t) n * HS));
        for (float* p : {key0, hid0, val0, hist_a, snap_a, q, nrm, g, gd, cv, key1, hid1, val1, hist_b, snap_b, qn, gated,
                         gate})
            cudaFree(p);
    }
}

// ---- 3. fused QSA gate + Q8_1 quantize (native_qsa_gate_quantize_q8_1 vs native_qsa_gate_apply + native_quantize_q8_1)
void qsa_gate_q8(cudaStream_t s) {
    const int NH = 16, HD = 256;
    std::mt19937 rng(17);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int n : {1, 4, 8, 9, 12, 16}) {
        const size_t elems = (size_t) n * NH * HD;
        const size_t q8_bytes = (elems / 32) * 36;
        const size_t il_bytes = n >= 2 ? k::native_q8_1_il_bytes(NH * HD, n) : 0;
        std::vector<float> attn(elems), qfull(elems * 2);
        for (auto& x : attn) x = nd(rng);
        for (auto& x : qfull) x = 2.f * nd(rng);
        float* d_attn = upload(attn);
        float* d_qfull = upload(qfull);
        float* d_out = dalloc<float>(elems);
        uint8_t* xq0 = dalloc<uint8_t>(q8_bytes);
        uint8_t* xq1 = dalloc<uint8_t>(q8_bytes);
        uint8_t* xil0 = n >= 2 ? dalloc<uint8_t>(il_bytes) : nullptr;
        uint8_t* xil1 = n >= 2 ? dalloc<uint8_t>(il_bytes) : nullptr;
        if (n >= 2) {
            ck(cudaMemsetAsync(xil0, 0, il_bytes, s), "qsa il0 zero");
            ck(cudaMemsetAsync(xil1, 0, il_bytes, s), "qsa il1 zero");
        }
        k::native_qsa_gate_apply(d_attn, d_qfull, d_out, n * NH, HD, s);
        if (n >= 2) k::native_quantize_q8_1_il(d_out, xq0, xil0, NH * HD, n, s);
        else k::native_quantize_q8_1(d_out, xq0, NH * HD, n, s);
        k::native_qsa_gate_quantize_q8_1(d_attn, d_qfull, xq1, n * NH, HD, s, xil1, n >= 2 ? n : 0, false);
        ck(cudaStreamSynchronize(s), "qsa_gate_q8");
        char what[128];
        std::snprintf(what, sizeof what, "native_qsa_gate_quantize_q8_1, %d tokens", n);
        bool ok = read(xq0, q8_bytes) == read(xq1, q8_bytes);
        if (n >= 2) ok = ok && (read(xil0, il_bytes) == read(xil1, il_bytes));
        report(what, ok);
        if (n >= 2) {
            ck(cudaMemsetAsync(xil0, 0, il_bytes, s), "qsa fwht il0 zero");
            ck(cudaMemsetAsync(xil1, 0, il_bytes, s), "qsa fwht il1 zero");
            float* d_rot = dalloc<float>(elems);
            ck(cudaMemcpyAsync(d_rot, d_attn, elems * sizeof(float), cudaMemcpyDeviceToDevice, s), "qsa rot copy");
            k::fwht256_inplace_cuda(d_rot, (int64_t) n * NH, s);
            k::native_qsa_gate_apply(d_rot, d_qfull, d_out, n * NH, HD, s);
            k::native_quantize_q8_1_il(d_out, xq0, xil0, NH * HD, n, s);
            k::native_qsa_gate_quantize_q8_1(d_attn, d_qfull, xq1, n * NH, HD, s, xil1, n, true);
            ck(cudaStreamSynchronize(s), "qsa_gate_q8_fwht");
            std::snprintf(what, sizeof what, "native_qsa_gate_quantize_q8_1 fwht+il, %d tokens", n);
            report(what, read(xq0, q8_bytes) == read(xq1, q8_bytes) && read(xil0, il_bytes) == read(xil1, il_bytes));
            cudaFree(d_rot);
        }
        cudaFree(d_attn); cudaFree(d_qfull); cudaFree(d_out); cudaFree(xq0); cudaFree(xq1);
        cudaFree(xil0); cudaFree(xil1);
    }
}

// ---- 4. fused GDN multi-group step + norm + Q8_1 quantize (gdn_step_norm_groups with xq_out vs separate native_quantize_q8_1)
void gdn_groups_q8(cudaStream_t s) {
    const int HK = 16, HV = 32, S = 128, ZV = HV * S, C = (2 * HK + HV) * S;
    std::mt19937 rng(23);
    std::normal_distribution<float> nd(0.f, 0.5f);
    for (int n : {8, 9, 12, 16}) {
        const int n0 = n / 2, n1 = n - n0;
        const size_t st_floats = (size_t) HV * S * S;
        const size_t q8_bytes = ((size_t) n * ZV / 32) * 36;
        const size_t il_bytes = k::native_q8_1_il_bytes(ZV, n);
        std::vector<float> st0(st_floats), st1(st_floats), h((size_t) n * C), gate((size_t) n * HV), beta((size_t) n * HV),
            z((size_t) n * ZV), gam(S);
        for (auto* v : {&st0, &st1, &h, &gate, &beta, &z})
            for (auto& x : *v) x = nd(rng);
        for (auto& x : gam) x = 1.f + 0.1f * nd(rng);
        float *s0a = upload(st0), *s1a = upload(st1), *s0b = upload(st0), *s1b = upload(st1);
        float *dh = upload(h), *dg = upload(gate), *db = upload(beta), *dz = upload(z), *dgam = upload(gam);
        float *ya = dalloc<float>((size_t) n * ZV), *yb = dalloc<float>((size_t) n * ZV);
        uint8_t *xqa = dalloc<uint8_t>(q8_bytes), *xqb = dalloc<uint8_t>(q8_bytes);
        uint8_t *xila = dalloc<uint8_t>(il_bytes), *xilb = dalloc<uint8_t>(il_bytes);
        ck(cudaMemsetAsync(xila, 0, il_bytes, s), "gdn il_a zero");
        ck(cudaMemsetAsync(xilb, 0, il_bytes, s), "gdn il_b zero");
        k::GdnGroups gga{}, ggb{};
        gga.n = 2; gga.state[0] = s0a; gga.first[0] = 0; gga.ntok[0] = n0; gga.state[1] = s1a; gga.first[1] = n0; gga.ntok[1] = n1;
        ggb.n = 2; ggb.state[0] = s0b; ggb.first[0] = 0; ggb.ntok[0] = n0; ggb.state[1] = s1b; ggb.first[1] = n0; ggb.ntok[1] = n1;
        k::gdn_step_norm_groups(gga, dh, C, dg, db, dz, dgam, 1e-6f, ya, HK, HV, false, s, 0, nullptr);
        k::native_quantize_q8_1_il(ya, xqa, xila, ZV, n, s);
        k::gdn_step_norm_groups(ggb, dh, C, dg, db, dz, dgam, 1e-6f, yb, HK, HV, false, s, 0, xqb, xilb);
        ck(cudaStreamSynchronize(s), "gdn_groups_q8");
        char what[128];
        std::snprintf(what, sizeof what, "gdn_step_norm_groups fused q8_1+il, %d tokens (2 groups)", n);
        report(what, read(xqa, q8_bytes) == read(xqb, q8_bytes) && read(xila, il_bytes) == read(xilb, il_bytes));
        for (float* p : {s0a, s1a, s0b, s1b, dh, dg, db, dz, dgam, ya, yb}) cudaFree(p);
        cudaFree(xqa); cudaFree(xqb); cudaFree(xila); cudaFree(xilb);
    }
}

}  // namespace

int main() {
    cudaDeviceProp prop{};
    int dev = 0;
    ck(cudaGetDevice(&dev), "device");
    ck(cudaGetDeviceProperties(&prop, dev), "props");
    std::printf("verify_batch_parity: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    cudaStream_t s = nullptr;
    ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    gr_post(s);
    ple(s);
    qsa_gate_q8(s);
    gdn_groups_q8(s);
    std::printf("verify_batch_parity: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
