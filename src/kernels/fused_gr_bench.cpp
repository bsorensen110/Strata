// src/kernels/fused_gr_bench.cpp - fused_gr_read_multi and the isolated down-kernel arm timing.
//
//     build/fused_gr_bench [iters] [T min] [T max]                      the timing and old/fast parity this file
//                                                                        has always done (unchanged, no --mode)
//     build/fused_gr_bench [iters] [T min] [T max] --mode=parity        packed staged vs staged, bitwise
//     build/fused_gr_bench [iters] [T min] [T max] --mode=time          staged and packed, paired rounds
//     build/fused_gr_bench [iters] [T min] [T max] --mode=time-staged   one arm, fresh process
//     build/fused_gr_bench [iters] [T min] [T max] --mode=time-packed   one arm, fresh process
//
// The packed arms (S27, STRATA_HC_PACK; include/strata/kernels/hc_pack.hpp) build the 12-bit image of the SAME
// wd/wu/wi bytes this bench generates, validate it on the CPU, upload it whole, and hand it to
// fused_gr_read_multi through FusedGrArgs::pack_down / pack_up / pack_inject.  No production call site fills
// those fields (the loader never sets them), so this bench and the fused-GR self-test in
// src/kernels/cuda/fused_gr.cu:2658 are the only callers that do.
//
// Which arm a launch takes is the pack pointers, not the environment: with the packed arm latched on by
// fused_gr_check, a launch whose pack_* are null takes the plain staged read and one whose pack_* are set takes
// the packed staged read (src/kernels/cuda/fused_gr.cu:1496), so ONE process can run both arms on the same
// buffers, stream, weights and settings, and the reps can be paired round by round.  --mode=time-staged /
// time-packed are for running one arm per fresh process with identical settings instead.
//
// Every mode that reads packed weights runs fused_gr_check() first - it latches the variant and the packed arm
// for this device - and stops before timing anything if the card rejects the packed read.
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/hc_pack.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace K = strata::kernels;

namespace {

enum BenchMode { kModeDefault = 0, kModeParity, kModeTime, kModeTimeStaged, kModeTimePacked };

/// One matrix's packed image: built and validated on the CPU from the bench's own BF16 bytes, uploaded whole,
/// and the kernel's view of it.  `view` is a HOST object the kernel dereferences (exactly as the fused-GR
/// self-test's pk_down / pk_up / pk_inj are), so an arena must outlive every launch that reads it.
struct PackedArena {
    std::vector<uint8_t> host;
    uint8_t* dev = nullptr;
    K::HcPackedMatrix pm;
    K::HcPackedWeights view;
};

/// count -> build -> validate on the CPU, then allocate and upload.  False with the reason in `err`; the
/// validator is the gate the packed arm is built on, so a bench never launches an image it has not checked.
bool make_arena(const char* label, const uint16_t* src, const K::HcPackGeometry& g, PackedArena& a,
                std::string& err) {
    const size_t esc = K::hc_pack_count_escapes(src, g);
    const size_t need = K::hc_pack_bytes(g, esc);
    a.host.assign(need, 0);
    if (!K::hc_pack_build(src, g, a.host.data(), a.host.size(), &a.pm, &err)) return false;
    if (!K::hc_pack_validate(src, g, a.pm, a.host.data(), &err)) return false;
    if (cudaMalloc((void**) &a.dev, need) != cudaSuccess) {
        err = std::string(label) + ": no room for the packed image (" + std::to_string(need) + " bytes)";
        return false;
    }
    if (cudaMemcpy(a.dev, a.host.data(), need, cudaMemcpyHostToDevice) != cudaSuccess) {
        err = std::string(label) + ": uploading the packed image failed";
        return false;
    }
    a.view = K::hc_pack_device_view(a.pm, a.dev);
    return true;
}

double median_of(const double* v, int n) {
    std::vector<double> s(v, v + n);
    std::sort(s.begin(), s.end());
    return n % 2 ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}

}  // namespace

static uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    // The positional arguments are unchanged: argv[1] iters, argv[2] T min, argv[3] T max, in order.  A
    // --mode=... flag is picked out of the list and does not shift them.
    int iters = 500, t_lo = 1, t_hi = 6;
    BenchMode mode = kModeDefault;
    const char* mode_name = "default";
    for (int i = 1, pos = 0; i < argc; ++i) {
        if (std::strncmp(argv[i], "--mode=", 7) == 0) {
            const char* m = argv[i] + 7;
            if (std::strcmp(m, "parity") == 0) mode = kModeParity;
            else if (std::strcmp(m, "time") == 0) mode = kModeTime;
            else if (std::strcmp(m, "time-staged") == 0) mode = kModeTimeStaged;
            else if (std::strcmp(m, "time-packed") == 0) mode = kModeTimePacked;
            else {
                std::fprintf(stderr, "fused_gr_bench: unknown --mode=%s (parity, time, time-staged, time-packed)\n",
                             m);
                return 2;
            }
            mode_name = m;
            continue;
        }
        if (pos == 0) iters = std::atoi(argv[i]);
        else if (pos == 1) t_lo = std::atoi(argv[i]);
        else if (pos == 2) t_hi = std::atoi(argv[i]);
        ++pos;
    }
    const int N = 2560, HC = 4, D = N * HC, LR = 320, TM = K::kFusedGrMaxT;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto fill = [&](std::vector<float>& v, float sc) { for (auto& x : v) x = sc * nd(rng); };
    std::vector<float> R((size_t) TM * D), bo((size_t) TM * N), inj((size_t) TM * HC), wn(D);
    fill(R, 1.0f); fill(bo, 0.5f); fill(inj, 1.0f); fill(wn, 0.3f);
    for (auto& x : wn) x += 1.0f;
    std::vector<uint16_t> wd((size_t) LR * D), wu((size_t) D * LR), wi((size_t) HC * D);
    for (auto& x : wd) x = bf16(0.02f * nd(rng));
    for (auto& x : wu) x = bf16(0.05f * nd(rng));
    for (auto& x : wi) x = bf16(0.02f * nd(rng));
    float *dR, *dbo, *dinj, *dwn, *dxn, *dlo, *drs, *dio, *dmix;
    uint16_t *dwd, *dwu, *dwi;
    cudaMalloc((void**) &dR, R.size() * 4);
    cudaMalloc((void**) &dbo, bo.size() * 4);
    cudaMalloc((void**) &dinj, inj.size() * 4);
    cudaMalloc((void**) &dwn, wn.size() * 4);
    cudaMalloc((void**) &dxn, (size_t) TM * D * 4);
    cudaMalloc((void**) &dlo, (size_t) TM * LR * 4);
    cudaMalloc((void**) &drs, (size_t) TM * HC * 4);
    cudaMalloc((void**) &dio, (size_t) TM * HC * 4);
    cudaMalloc((void**) &dmix, (size_t) TM * N * 4);
    cudaMalloc((void**) &dwd, wd.size() * 2);
    cudaMalloc((void**) &dwu, wu.size() * 2);
    cudaMalloc((void**) &dwi, wi.size() * 2);
    cudaMemcpy(dbo, bo.data(), bo.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dinj, inj.data(), inj.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dwn, wn.data(), wn.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dwd, wd.data(), wd.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dwu, wu.data(), wu.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dwi, wi.data(), wi.size() * 2, cudaMemcpyHostToDevice);
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    int failures = 0;
    const bool hc_variant_bench = std::getenv("STRATA_HC_BENCH_DIRECT") != nullptr;
    const char* expect_reuse_env = std::getenv("STRATA_HC_EXPECT_REUSE");
    const bool expect_reuse_variant = expect_reuse_env != nullptr && expect_reuse_env[0] != '0';
    const char* expect_variant_env = std::getenv("STRATA_HC_EXPECT_VARIANT");
    const int expect_variant = expect_variant_env != nullptr ? std::atoi(expect_variant_env) : -1;
    // With the promote-to-default, the arm a direct bench runs is the latched one, not the env's bare number:
    // latch first (idempotent - a checked card returns at once), then read what the card picked.
    if (hc_variant_bench) K::fused_gr_check();
    const int selected_hc_variant = hc_variant_bench ? K::fused_gr_variant() : 0;
    if (hc_variant_bench && selected_hc_variant != 3 && selected_hc_variant != 4 && selected_hc_variant != 5 &&
        selected_hc_variant != 6 && selected_hc_variant != 7 && selected_hc_variant != 9 &&
        selected_hc_variant != 10 && selected_hc_variant != 11)
        return 2;
    if (expect_reuse_variant && selected_hc_variant != 5) {
        std::fprintf(stderr, "fused_gr_bench: expected two-row reuse variant 5, selected %d\n", selected_hc_variant);
        return 2;
    }
    if (expect_variant >= 0 && selected_hc_variant != expect_variant) {
        std::fprintf(stderr, "fused_gr_bench: expected HC variant %d, selected %d\n", expect_variant,
                     selected_hc_variant);
        return 2;
    }
    if (hc_variant_bench) {
        const char* variant_name = selected_hc_variant == 3    ? "staged"
                                   : selected_hc_variant == 4  ? "small-CTA staged"
                                   : selected_hc_variant == 5  ? "2-row-per-warp staged"
                                   : selected_hc_variant == 6  ? "register-pipe staged"
                                   : selected_hc_variant == 9  ? "row-split staged"
                                   : selected_hc_variant == 10 ? "LDS-accumulator staged"
                                   : selected_hc_variant == 11 ? "register-pipe-half staged"
                                                               : "register-half staged";
        std::fprintf(stderr, "fused_gr_bench: selected HC variant %d (%s)\n", selected_hc_variant, variant_name);
        K::fused_gr_set_fast(0);
    }

    // ==================== S27 STRATA_HC_PACK arms (--mode=..., see the file header) ====================
    if (mode != kModeDefault) {
        // Every allocation this bench makes is freed on every exit from this block, the arenas included.
        auto free_base = [&] {
            cudaFree(dR); cudaFree(dbo); cudaFree(dinj); cudaFree(dwn); cudaFree(dxn);
            cudaFree(dlo); cudaFree(drs); cudaFree(dio); cudaFree(dmix);
            cudaFree(dwd); cudaFree(dwu); cudaFree(dwi);
            cudaEventDestroy(e0); cudaEventDestroy(e1); cudaStreamDestroy(s);
        };
        // The gate first, before any launch: fused_gr_check runs the card's self-test once and latches BOTH the
        // variant and the packed arm for this device, so the packed mode has to be asked for (STRATA_HC_PACK=1)
        // before it runs.  Every mode below needs the staged read; the modes that read packed weights need the
        // latch to say packed, and stop here - nothing is timed, nothing is printed - if it says otherwise.
        K::fused_gr_check();
        const int variant = K::fused_gr_variant();
        if (variant != 3 && variant != 9 && variant != 10) {
            std::fprintf(stderr, "fused_gr_bench: --mode=%s needs the staged read (variant 3, STRATA_HC_SPLIT=2), "
                                 "the row split (variant 9, STRATA_HC_SPLIT=8) or the LDS-accumulator read (variant "
                                 "10, STRATA_HC_SPLIT=9); this card selected %d\n",
                         mode_name, variant);
            free_base();
            return 2;
        }
        const bool wants_packed = mode == kModeParity || mode == kModeTime || mode == kModeTimePacked;
        const int pack_latched = K::fused_gr_hc_pack();
        if (wants_packed && pack_latched != 1) {
            std::fprintf(stderr, "fused_gr_bench: --mode=%s needs fused_gr_hc_pack() == 1 and it is %d: the check's "
                                 "own line above says why this card rejected the packed read.  Nothing is timed.\n",
                         mode_name, pack_latched);
            free_base();
            return 2;
        }
        // The three matrices, packed from the bytes already uploaded to dwd / dwu / dwi and validated on the CPU.
        PackedArena adown, aup, ainj;
        std::string perr;
        if (!make_arena("hc_down", wd.data(), K::hc_pack_geometry_down(), adown, perr) ||
            !make_arena("hc_up", wu.data(), K::hc_pack_geometry_up(), aup, perr) ||
            !make_arena("hc_inject", wi.data(), K::hc_pack_geometry_inject(), ainj, perr)) {
            std::fprintf(stderr, "fused_gr_bench: packing the bench's weights: %s\n", perr.c_str());
            cudaFree(adown.dev); cudaFree(aup.dev); cudaFree(ainj.dev);   // cudaFree(null) is legal
            free_base();
            return 2;
        }
        // T 1..6: the packed kernels instantiate 1..6 exactly (src/kernels/cuda/fused_gr.cu:1570, 1603); 7 and 8
        // take the generic instantiation, which the fused-GR self-test covers but this bench does not time.
        const int plo = t_lo < 1 ? 1 : t_lo, phi = t_hi > 6 ? 6 : t_hi;
        if (phi != t_hi || plo != t_lo)
            std::fprintf(stderr, "fused_gr_bench: --mode=%s runs T %d..%d (asked for %d..%d)\n", mode_name, plo, phi,
                         t_lo, t_hi);
        std::printf("fused_gr_bench: mode %s | %s read (variant %d) | packed arm %s | iters %d | T %d..%d\n",
                    mode_name, variant == 9 ? "row-split" : variant == 10 ? "LDS-accumulator" : "staged", variant,
                    pack_latched == 1 ? "latched on (fused_gr_hc_pack 1)" : "off (fused_gr_hc_pack 0)",
                    iters, plo, phi);
        std::printf("fused_gr_bench: packed image hc_down %zu escapes / %zu B, hc_up %zu escapes / %zu B, "
                    "hc_inject %zu escapes / %zu B; BF16 beside it %zu + %zu + %zu B\n",
                    adown.pm.escapes, adown.pm.total_bytes, aup.pm.escapes, aup.pm.total_bytes, ainj.pm.escapes,
                    ainj.pm.total_bytes, wd.size() * 2, wu.size() * 2, wi.size() * 2);

        // One token set's arguments.  The two arms differ ONLY in the three pack pointers: null is the plain
        // staged read of dwd / dwu / dwi, set is the packed staged read of the same values.  With w_inject null
        // (the final mixer) pack_inject stays null, which is what the packed kernel's injection CTA exits on.
        auto fill_args = [&](K::FusedGrArgs* a, int T, int apply, int inject, bool packed) {
            for (int t = 0; t < T; ++t) {
                a[t] = K::FusedGrArgs{};
                a[t].R = dR + (size_t) t * D; a[t].R_out = dR + (size_t) t * D; a[t].apply = apply;
                a[t].bo_prev = dbo + (size_t) t * N; a[t].inj_prev = dinj + (size_t) t * HC;
                a[t].w_norm = dwn; a[t].w_down = dwd; a[t].w_up = dwu; a[t].w_inject = inject ? dwi : nullptr;
                if (packed) {
                    a[t].pack_down = adown.view;
                    a[t].pack_up = aup.view;
                    if (inject) a[t].pack_inject = ainj.view;
                }
                a[t].eps = 1e-6f;
                a[t].lo = dlo + (size_t) t * LR; a[t].rs = drs + (size_t) t * HC;
                a[t].inject_out = dio + (size_t) t * HC; a[t].mixed = dmix + (size_t) t * N;
            }
        };
        auto free_all = [&] {
            cudaFree(adown.dev); cudaFree(aup.dev); cudaFree(ainj.dev);
            free_base();
        };

        if (mode == kModeParity) {
            // Every observable output of the packed read, bit for bit, against the staged read of the same
            // bytes: R (updated in place), lo, rs, inject_out, mixed, over the whole TM-token span - the
            // tokens past T are memset to 0xff and compared too, so a kernel that writes past T is caught.
            // fused_gr_read_multi compares nothing itself; the memcmp below is the whole gate.
            std::vector<float> o[2];
            const size_t nfloat = (size_t) TM * (D + LR + HC + HC + N);
            int differs = 0;
            for (int T = plo; T <= phi; ++T)
                for (int apply = 0; apply < 2; ++apply)
                    for (int inject = 0; inject < 2; ++inject) {
                        K::FusedGrArgs a[TM];
                        for (int f = 0; f < 2; ++f) {
                            fill_args(a, T, apply, inject, f == 1);
                            K::fused_gr_set_fast(0);
                            cudaMemcpy(dR, R.data(), R.size() * 4, cudaMemcpyHostToDevice);
                            cudaMemset(dlo, 0xff, (size_t) TM * LR * 4);
                            cudaMemset(drs, 0xff, (size_t) TM * HC * 4);
                            cudaMemset(dio, 0xff, (size_t) TM * HC * 4);
                            cudaMemset(dmix, 0xff, (size_t) TM * N * 4);
                            K::fused_gr_read_multi(a, T, dxn, s);
                            cudaStreamSynchronize(s);
                            auto& v = o[f];
                            v.resize(nfloat);
                            float* p = v.data();
                            cudaMemcpy(p, dR, (size_t) TM * D * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * D;
                            cudaMemcpy(p, dlo, (size_t) TM * LR * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * LR;
                            cudaMemcpy(p, drs, (size_t) TM * HC * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * HC;
                            cudaMemcpy(p, dio, (size_t) TM * HC * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * HC;
                            cudaMemcpy(p, dmix, (size_t) TM * N * 4, cudaMemcpyDeviceToHost);
                        }
                        const bool same = std::memcmp(o[0].data(), o[1].data(), nfloat * 4) == 0;
                        std::printf("T %d apply %d inject %d | staged vs packed | %s\n", T, apply, inject,
                                    same ? "bitwise equal" : "DIFFERS");
                        if (!same) ++differs;
                    }
            std::printf("fused_gr_bench: %d differing (packed vs staged, %d cases)\n", differs, (phi - plo + 1) * 4);
            free_all();
            return differs ? 1 : 0;
        }

        // Timing.  No output is read back here - the arms are timed on the buffers, not measured by them.
        // --mode=time runs BOTH arms in one process, paired: round r measures staged then packed back to back
        // on the same buffers, so drift and clock state hit the two arms of a pair alike.  --mode=time-staged /
        // time-packed measure one arm each, for a fresh process per arm with the same settings.
        const int rounds = 5;
        const bool both = mode == kModeTime;
        const int arm = mode == kModeTimePacked ? 1 : 0;
        const char* arm_name = arm == 1 ? "packed" : "staged";
        int cases = 0;
        for (int T = plo; T <= phi; ++T) {
            double per_t0[4], per_t1[4];
            for (int apply = 0; apply < 2; ++apply)
                for (int inject = 0; inject < 2; ++inject, ++cases) {
                    K::FusedGrArgs a[TM];
                    double t[2][16];
                    for (int f = 0; f < 2; ++f)
                        for (int r = 0; r < 16; ++r) t[f][r] = 1e30;
                    for (int r = 0; r < rounds; ++r)
                        for (int f = 0; f < 2; ++f) {
                            if (!both && f != arm) continue;
                            fill_args(a, T, apply, inject, f == 1);
                            K::fused_gr_set_fast(0);
                            for (int i = 0; i < 20; ++i) K::fused_gr_read_multi(a, T, dxn, s);
                            cudaEventRecord(e0, s);
                            for (int i = 0; i < iters; ++i) K::fused_gr_read_multi(a, T, dxn, s);
                            cudaEventRecord(e1, s);
                            cudaEventSynchronize(e1);
                            float ms = 0.0f;
                            cudaEventElapsedTime(&ms, e0, e1);
                            t[f][r] = 1e3 * ms / iters;
                        }
                    const double m0 = (both || arm == 0) ? median_of(t[0], rounds) : 1e30;
                    const double m1 = (both || arm == 1) ? median_of(t[1], rounds) : 1e30;
                    per_t0[apply * 2 + inject] = m0;
                    per_t1[apply * 2 + inject] = m1;
                    if (both)
                        std::printf("T %d apply %d inject %d | staged %6.1f us  packed %6.1f us  packed/staged %5.3f"
                                    " | median of %d paired rounds\n", T, apply, inject, m0, m1, m1 / m0, rounds);
                    else
                        std::printf("T %d apply %d inject %d | %s %6.1f us | median of %d rounds\n", T, apply, inject,
                                    arm_name, arm == 1 ? m1 : m0, rounds);
                }
            const double s0 = median_of(per_t0, 4), s1 = median_of(per_t1, 4);
            if (both)
                std::printf("T %d | median over the 4 cases: staged %6.1f us  packed %6.1f us  packed/staged %5.3f\n",
                            T, s0, s1, s1 / s0);
            else
                std::printf("T %d | median over the 4 cases: %s %6.1f us\n", T, arm_name, arm == 1 ? s1 : s0);
        }
        std::printf("fused_gr_bench: %d cases timed, %d rounds each, %d iterations per rep\n", cases, rounds, iters);
        free_all();
        return 0;
    }

    // STRATA_HC_BENCH_GRAPH_CHURN=1 (the STRATA_HC_GRAPH promote audit, opt-in, nothing in the engine changes):
    // the graph cache's three costs, one T per process.  The cache keys on every pointer and flag in the argument
    // set, so a call whose pointers differ from every cached set captures a new graph.  `steady` is the hit path,
    // `churn` cycles NSETS sets - one more than the cache's 32 entries - so every call misses, and `capture` is
    // one cold call on a set the cache has never seen.  STRATA_HC_BENCH_LEGACY_STREAM=1 passes stream nullptr, which
    // is what a call site with no stream of its own passes (verify.cpp:605 falls back to nullptr when its stream
    // creation fails), so the capture runs on the legacy stream.
    if (std::getenv("STRATA_HC_BENCH_GRAPH_CHURN") != nullptr) {
        constexpr int NSETS = 33;  // one more than graph_cache()'s 32 entries
        const bool legacy_stream = std::getenv("STRATA_HC_BENCH_LEGACY_STREAM") != nullptr;
        void* const gs = legacy_stream ? (void*) nullptr : (void*) s;
        struct GSet { float *R, *bo, *inj, *xn, *lo, *rs, *io, *mix; };
        auto alloc_set = [&](GSet& g) {
            cudaMalloc((void**) &g.R, (size_t) TM * D * 4);
            cudaMalloc((void**) &g.bo, (size_t) TM * N * 4);
            cudaMalloc((void**) &g.inj, (size_t) TM * HC * 4);
            cudaMalloc((void**) &g.xn, (size_t) TM * D * 4);
            cudaMalloc((void**) &g.lo, (size_t) TM * LR * 4);
            cudaMalloc((void**) &g.rs, (size_t) TM * HC * 4);
            cudaMalloc((void**) &g.io, (size_t) TM * HC * 4);
            cudaMalloc((void**) &g.mix, (size_t) TM * N * 4);
            // The inputs get the same values the default path copies.  Uninitialized R/bo/inj make the kernels
            // run on denormals and NaN, which costs a few us of GPU time and hides the launch gap the graph removes.
            cudaMemcpy(g.R, R.data(), (size_t) TM * D * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(g.bo, bo.data(), (size_t) TM * N * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(g.inj, inj.data(), (size_t) TM * HC * 4, cudaMemcpyHostToDevice);
        };
        auto free_set = [&](GSet& g) {
            cudaFree(g.R); cudaFree(g.bo); cudaFree(g.inj); cudaFree(g.xn);
            cudaFree(g.lo); cudaFree(g.rs); cudaFree(g.io); cudaFree(g.mix);
        };
        auto fill_set = [&](K::FusedGrArgs* a, const GSet& g, int T, int apply, int inject) {
            for (int t = 0; t < T; ++t) {
                a[t] = K::FusedGrArgs{};
                a[t].R = g.R + (size_t) t * D; a[t].R_out = g.R + (size_t) t * D; a[t].apply = apply;
                a[t].bo_prev = g.bo + (size_t) t * N; a[t].inj_prev = g.inj + (size_t) t * HC;
                a[t].w_norm = dwn; a[t].w_down = dwd; a[t].w_up = dwu; a[t].w_inject = inject ? dwi : nullptr;
                a[t].lo = g.lo + (size_t) t * LR; a[t].rs = g.rs + (size_t) t * HC;
                a[t].inject_out = g.io + (size_t) t * HC; a[t].mixed = g.mix + (size_t) t * N;
            }
        };
        K::fused_gr_set_fast(0);
        std::printf("fused_gr_bench: graph-churn arm, variant %d, NSETS %d, iters %d, stream %s\n",
                    K::fused_gr_variant(), NSETS, iters, legacy_stream ? "legacy (nullptr)" : "created");
        // STRATA_HC_BENCH_NESTED=1: the production shape, at t_lo.  Every production call site runs inside the
        // engine's own stream capture - Verifier::capture (verify.cpp:1685) and MtpDrafter::capture_round/step/
        // prefill (mtp.cpp:1045/1092/1024/1033) call cudaStreamBeginCapture, then record_window/record_front/
        // record_rest, which are where fused_gr_read_multi is called.  So the read is captured into a whole-step
        // graph, and with STRATA_HC_GRAPH=1 the inner graph_replay would capture a stream that is already
        // capturing.  This arm does that outer capture, instantiate and replay, and prints each step's status.
        if (std::getenv("STRATA_HC_BENCH_NESTED") != nullptr) {
            const int T = t_lo;
            const cudaStream_t ns = (cudaStream_t) gs;
            K::FusedGrArgs a[TM];
            GSet fresh{};
            alloc_set(fresh);
            fill_set(a, fresh, T, 1, 1);
            cudaGraph_t g = nullptr;
            cudaGraphExec_t ex = nullptr;
            const cudaError_t b = cudaStreamBeginCapture(ns, cudaStreamCaptureModeThreadLocal);
            K::fused_gr_read_multi(a, T, fresh.xn, gs);
            const cudaError_t en = cudaStreamEndCapture(ns, &g);
            cudaError_t in = cudaSuccess;   // left as success when there is nothing to instantiate
            if (en == cudaSuccess && g != nullptr) in = cudaGraphInstantiate(&ex, g, nullptr, nullptr, 0);
            std::printf("T %d | nested: begin %s | end %s | instantiate %s\n", T,
                        cudaGetErrorString(b), cudaGetErrorString(en), cudaGetErrorString(in));
            double replay_us = -1.0;
            if (ex != nullptr) {
                cudaGraphLaunch(ex, ns);
                cudaStreamSynchronize(ns);
                float ms2 = 0.0f;
                cudaEventRecord(e0, s);
                for (int i = 0; i < iters; ++i) cudaGraphLaunch(ex, ns);
                cudaEventRecord(e1, s);
                cudaEventSynchronize(e1);
                cudaEventElapsedTime(&ms2, e0, e1);
                replay_us = 1e3 * ms2 / iters;
                std::printf("T %d | nested: outer-graph replay %6.1f us\n", T, replay_us);
                (void) replay_us;
            }
            if (g) cudaGraphDestroy(g);
            free_set(fresh);
            cudaEventDestroy(e0); cudaEventDestroy(e1); cudaStreamDestroy(s);
            cudaFree(dR); cudaFree(dbo); cudaFree(dinj); cudaFree(dwn); cudaFree(dxn);
            cudaFree(dlo); cudaFree(drs); cudaFree(dio); cudaFree(dmix);
            cudaFree(dwd); cudaFree(dwu); cudaFree(dwi);
            return 0;
        }
        for (int T = t_lo; T <= t_hi; ++T) {
            K::FusedGrArgs a[TM];
            float ms = 0.0f;
            // capture: one cold call on a set the cache has never seen, no warmup.
            GSet fresh{};
            alloc_set(fresh);
            fill_set(a, fresh, T, 1, 1);
            cudaEventRecord(e0, s);
            K::fused_gr_read_multi(a, T, fresh.xn, gs);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            cudaEventElapsedTime(&ms, e0, e1);
            const double capture_us = 1e3 * ms;
            // steady: that same set warmed, so every call below is a cache hit.
            for (int i = 0; i < 20; ++i) K::fused_gr_read_multi(a, T, fresh.xn, gs);
            cudaEventRecord(e0, s);
            for (int i = 0; i < iters; ++i) K::fused_gr_read_multi(a, T, fresh.xn, gs);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            cudaEventElapsedTime(&ms, e0, e1);
            const double steady_us = 1e3 * ms / iters;
            free_set(fresh);
            // churn: NSETS sets cycled, so every call's key is one the cache has no room for.
            std::vector<GSet> sets((size_t) NSETS);
            for (int i = 0; i < NSETS; ++i) {
                alloc_set(sets[(size_t) i]);
                fill_set(a, sets[(size_t) i], T, 1, 1);
                for (int w = 0; w < 2; ++w) K::fused_gr_read_multi(a, T, sets[(size_t) i].xn, gs);
            }
            cudaEventRecord(e0, s);
            for (int i = 0; i < iters; ++i) {
                const GSet& g = sets[(size_t) (i % NSETS)];
                fill_set(a, g, T, 1, 1);
                K::fused_gr_read_multi(a, T, g.xn, gs);
            }
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            cudaEventElapsedTime(&ms, e0, e1);
            const double churn_us = 1e3 * ms / iters;
            for (auto& g : sets) free_set(g);
            std::printf("T %d | steady %6.1f us  churn %6.1f us  capture (1 cold call) %8.1f us\n",
                        T, steady_us, churn_us, capture_us);
        }
        cudaEventDestroy(e0); cudaEventDestroy(e1); cudaStreamDestroy(s);
        cudaFree(dR); cudaFree(dbo); cudaFree(dinj); cudaFree(dwn); cudaFree(dxn);
        cudaFree(dlo); cudaFree(drs); cudaFree(dio); cudaFree(dmix);
        cudaFree(dwd); cudaFree(dwu); cudaFree(dwi);
        return 0;
    }

    // The default arm is the promoted default: latch it before timing (the check prints the card's parity
    // line and the effective variant), so the startup names the variant every launch below really takes.
    K::fused_gr_check();

    for (int T = t_lo; T <= t_hi; ++T)
        for (int apply = 0; apply < 2; ++apply)
            for (int inject = 0; inject < 2; ++inject) {
                K::FusedGrArgs a[TM];
                for (int t = 0; t < T; ++t) {
                    a[t].R = dR + (size_t) t * D; a[t].R_out = dR + (size_t) t * D; a[t].apply = apply;
                    a[t].bo_prev = dbo + (size_t) t * N; a[t].inj_prev = dinj + (size_t) t * HC;
                    a[t].w_norm = dwn; a[t].w_down = dwd; a[t].w_up = dwu; a[t].w_inject = inject ? dwi : nullptr;
                    a[t].lo = dlo + (size_t) t * LR; a[t].rs = drs + (size_t) t * HC;
                    a[t].inject_out = dio + (size_t) t * HC; a[t].mixed = dmix + (size_t) t * N;
                }
                std::vector<float> out[2];
                const int measurement_arms = hc_variant_bench ? 1 : 2;
                for (int f = 0; f < measurement_arms; ++f) {
                    if (hc_variant_bench && K::fused_gr_variant() != selected_hc_variant) return 2;
                    cudaMemcpy(dR, R.data(), R.size() * 4, cudaMemcpyHostToDevice);
                    cudaMemset(dlo, 0xff, (size_t) TM * LR * 4);
                    cudaMemset(drs, 0xff, (size_t) TM * HC * 4);
                    cudaMemset(dio, 0xff, (size_t) TM * HC * 4);
                    cudaMemset(dmix, 0xff, (size_t) TM * N * 4);
                    K::fused_gr_set_fast(hc_variant_bench ? 0 : f);
                    K::fused_gr_read_multi(a, T, dxn, s);
                    cudaStreamSynchronize(s);
                    auto& o = out[f];
                    o.resize((size_t) TM * (D + LR + HC + HC + N));
                    if (!hc_variant_bench) {
                        float* p = o.data();
                        cudaMemcpy(p, dR, (size_t) TM * D * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * D;
                        cudaMemcpy(p, dlo, (size_t) TM * LR * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * LR;
                        cudaMemcpy(p, drs, (size_t) TM * HC * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * HC;
                        cudaMemcpy(p, dio, (size_t) TM * HC * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * HC;
                        cudaMemcpy(p, dmix, (size_t) TM * N * 4, cudaMemcpyDeviceToHost);
                    }
                }
                const bool same = hc_variant_bench ? true : std::memcmp(out[0].data(), out[1].data(), out[0].size() * 4) == 0;
                if (hc_variant_bench) {
                    double kernel_us = 1e30;
                    for (int round = 0; round < 3; ++round) {
                        K::fused_gr_set_fast(0);
                        for (int i = 0; i < 30; ++i) K::fused_gr_read_multi(a, T, dxn, s);
                        cudaEventRecord(e0, s);
                        for (int i = 0; i < iters; ++i) K::fused_gr_read_multi(a, T, dxn, s);
                        cudaEventRecord(e1, s);
                        cudaEventSynchronize(e1);
                        float ms = 0.0f;
                        cudaEventElapsedTime(&ms, e0, e1);
                        kernel_us = std::fmin(kernel_us, 1e3 * ms / iters);
                    }
                    const int eff_variant = K::fused_gr_variant_for_T(T);   // what this launch really ran
                    std::printf("T %d apply %d inject %d | HC-read %6.1f us (%s) | %s\n", T, apply, inject, kernel_us,
                                eff_variant == 3    ? "staged"
                                : eff_variant == 4  ? "small-CTA staged"
                                : eff_variant == 5  ? "2-row-per-warp staged"
                                : eff_variant == 6  ? "register-pipe staged"
                                : eff_variant == 9  ? "row-split staged"
                                : eff_variant == 10 ? "LDS-accumulator staged"
                                : eff_variant == 11 ? "register-pipe-half staged"
                                                    : "register-half staged",
                                hc_variant_bench ? "timed-only; parity is a separate gate"
                                                 : (same ? "bitwise equal" : "DIFFERS"));
                    if (!same) ++failures;
                    continue;
                }
                double us[2] = {1e30, 1e30};
                for (int round = 0; round < 3; ++round)
                    for (int f = 0; f < 2; ++f) {
                        K::fused_gr_set_fast(f);
                        for (int i = 0; i < 20; ++i) K::fused_gr_read_multi(a, T, dxn, s);
                        cudaEventRecord(e0, s);
                        for (int i = 0; i < iters; ++i) K::fused_gr_read_multi(a, T, dxn, s);
                        cudaEventRecord(e1, s);
                        cudaEventSynchronize(e1);
                        float ms = 0;
                        cudaEventElapsedTime(&ms, e0, e1);
                        us[f] = std::fmin(us[f], 1e3 * ms / iters);
                    }
                std::printf("T %d apply %d inject %d | old %6.1f us  fast %6.1f us | %s\n", T, apply, inject, us[0], us[1],
                            same ? "bitwise equal" : "DIFFERS");
                if (!same) ++failures;
            }
    std::printf("fused_gr_bench: %d differing\n", failures);
    return failures ? 1 : 0;
}
