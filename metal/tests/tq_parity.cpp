// metal/tests/tq_parity.cpp - Gyro's trellis experts (TQ2_T / TQK6 / TQK7, ggml 144-146) on Metal against the CPU reference.
//
//   tq_parity [--rows R] [--seed S]
//
// For each type and both expert shapes (gate/up: 2560 -> 640 rows, down: 640 -> 2560 rows) random valid blocks and random
// activations x ~ U(-1,1): the GEMV is moe_down (moe.metal) of one entry of weight 1, i.e. ye = W . h with h = x, at the
// engine's default tile and at every tile the window can pick.  Checked, as max |err| / max |ref| and relative L2:
//   exact   Metal against a double dot of the exact fp16 codebook values (ref::dequant: an independent bit-by-bit
//           decoder of the format) - the kernel's own error, float accumulation only;
//   cpu     Metal fed the Q8_0-rounded x against src/kernels/cpu/tq_cpu.cpp's scalar reference (tq_rows): the same
//           inputs, so what is left is the CPU's int16 (2^-13) codebook rounding and its Q8 integer arithmetic;
//   q8      the CPU reference against `exact` - the Q8_0 activation quantization the CPU path adds (information).
// And the Hadamard rotation (hadamard_rows of glue.metal) against hadamard.cpp's hadamard_rotate: bit-identical or not.
// Exit 0 when every check passes (exact: rel L2 < 1e-5; cpu: rel L2 < 2e-3; Hadamard: max diff <= 1e-6 of the max).
#include "mtl_rt.h"
#include "ref_quant.h"

#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/kernels/cpu/tq_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// plan layout (moe.metal)
constexpr int P_NG = 0, P_IGU = 4, P_IDN = 8, P_MAXE = 128, P_GEXP = 16, P_GSTART = P_GEXP + P_MAXE,
              P_ETOK = P_GSTART + P_MAXE + 4, P_EW = P_ETOK + P_MAXE, P_TOKENT = P_EW + P_MAXE, P_INTS = P_TOKENT + P_MAXE;
struct MoeArgs {
    uint64_t gate_off, up_off, down_off, gu_stride, d_stride;
    uint32_t gu_row, d_row, n_embd, n_ff, n_tok, topk, n_expert, gx_gu, gx_dn, pad;
};
struct HadArgs { uint32_t width, has_signs; float scale, pad; };

struct Err {
    double max_abs = 0, max_ref = 0, se = 0, sr = 0;
    void add(double got, double ref) {
        max_abs = std::max(max_abs, std::fabs(got - ref));
        max_ref = std::max(max_ref, std::fabs(ref));
        se += (got - ref) * (got - ref);
        sr += ref * ref;
    }
    double rel_max() const { return max_ref > 0 ? max_abs / max_ref : max_abs; }
    double rel_l2() const { return sr > 0 ? std::sqrt(se / sr) : std::sqrt(se); }
};

struct Tile { int L, R, S; };

}  // namespace

int main(int argc, char** argv) {
    int rows_arg = 0;
    uint64_t seed = 7;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--rows" && i + 1 < argc) rows_arg = std::atoi(argv[++i]);
        else if (a == "--seed" && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    mtl::Context cx;
    const int lib = cx.compile({"quant.metal", "units.metal", "moe.metal", "hadamard.metal"});
    std::printf("device %s\n", cx.device_name().c_str());
    bool all_ok = true;
    ref::Rng rng(seed);

    const std::vector<Tile> tiles = {{4, 2, 4}, {4, 1, 4}, {8, 1, 4}, {8, 2, 4}, {16, 1, 4}, {2, 2, 4}};
    struct Shape { const char* name; int k, rows; };
    const Shape shapes[2] = {{"gate/up 2560->640", 2560, 640}, {"down 640->2560", 640, 2560}};
    for (int t : {ref::TQ2_T, ref::TQK6, ref::TQK7}) {
        for (const Shape& sh : shapes) {
            const int K = sh.k, R = rows_arg ? rows_arg : sh.rows;
            const size_t rb = ref::row_bytes(t, K);
            mtl::Buffer W = cx.alloc(rb * R + 256), x = cx.alloc(4 * K), ye = cx.alloc(4 * R), plan = cx.alloc(4 * P_INTS);
            ref::random_blocks(t, W.as<uint8_t>(), R, K, seed * 1000 + t);
            std::vector<float> xf(K), xq(K);
            for (auto& v : xf) v = rng.sym();
            // the Q8_0 activation of the CPU path, and its values back as floats
            std::vector<uint8_t> q8((size_t)K / 32 * 34);
            ref::quantize_q8_0(xf.data(), K, q8.data());
            ref::dequant(ref::Q8_0, q8.data(), xq.data(), K);
            // exact (double) reference on xf, and on xq
            std::vector<float> wd((size_t)R * K);
            ref::dequant(t, W.as<uint8_t>(), wd.data(), wd.size());
            std::vector<double> ex(R, 0.0), exq(R, 0.0);
            for (int r = 0; r < R; ++r)
                for (int k = 0; k < K; ++k) {
                    ex[r] += (double)wd[(size_t)r * K + k] * xf[k];
                    exq[r] += (double)wd[(size_t)r * K + k] * xq[k];
                }
            // the CPU kernel (scalar: the reference every ISA reproduces)
            std::vector<float> cpu(R);
            {
                const void* act[1] = {q8.data()};
                float* out[1] = {cpu.data()};
                strata::kernels::cpu::tq_rows(strata::kernels::cpu::TqIsa::scalar, t, W.as<uint8_t>(), rb, K, act, 1, out, 0, R);
            }
            // the plan: one group (expert 0) of one entry (token 0, weight 1)
            int* P = plan.as<int>();
            P[P_NG] = 1;
            P[P_GEXP] = 0;
            P[P_GSTART] = 0;
            P[P_GSTART + 1] = 1;
            P[P_ETOK] = 0;
            const float one = 1.0f;
            std::memcpy(&P[P_EW], &one, 4);
            P[P_TOKENT] = 0;
            for (int pass = 0; pass < 2; ++pass) {   // pass 0: x; pass 1: the Q8-rounded x against the CPU kernel
                std::memcpy(x.ptr, pass ? xq.data() : xf.data(), 4 * K);
                for (const Tile& tl : tiles) {
                    const int rpt = tl.S * (32 / tl.L) * tl.R;
                    if (R % rpt) continue;
                    MoeArgs a{};
                    a.d_stride = rb * R;
                    a.d_row = (uint32_t)rb;
                    a.n_embd = R;
                    a.n_ff = K;
                    a.n_tok = 1;
                    a.topk = 1;
                    a.n_expert = 1;
                    a.gx_dn = R / rpt;
                    const std::string name = "moe_down_" + std::to_string(t) + "_" + std::to_string(tl.L) + "_" +
                                             std::to_string(tl.R) + "_" + std::to_string(tl.S);
                    const mtl::Pipeline p = cx.pipeline(lib, name);
                    mtl::Dispatch d;
                    d.p = &p;
                    d.args = {mtl::Arg::B(W), mtl::Arg::B(x), mtl::Arg::B(ye), mtl::Arg::B(plan), mtl::Arg::V(a), mtl::Arg::B(x)};
                    d.tg[0] = tl.S * 32;
                    d.grid[0] = a.gx_dn;
                    cx.run({d});
                    Err e;
                    for (int r = 0; r < R; ++r) e.add(ye.as<float>()[r], pass ? (double)cpu[r] : ex[r]);
                    const bool ok = pass ? e.rel_l2() < 2e-3 : e.rel_l2() < 1e-5;
                    all_ok &= ok;
                    std::printf("%-5s %-18s tile %2d,%d,%d vs %-5s: max abs %.3e  max rel %.3e  rel L2 %.3e  %s\n",
                                ref::type_name(t), sh.name, tl.L, tl.R, tl.S, pass ? "cpu" : "exact", e.max_abs, e.rel_max(),
                                e.rel_l2(), ok ? "PASS" : "FAIL");
                }
            }
            Err q;
            for (int r = 0; r < R; ++r) q.add(cpu[r], ex[r]);
            std::printf("%-5s %-18s cpu (Q8_0 x, int16 codebook) vs exact: max abs %.3e  max rel %.3e  rel L2 %.3e  (information)\n",
                        ref::type_name(t), sh.name, q.max_abs, q.rel_max(), q.rel_l2());
        }
    }

    // ---- the Hadamard rotation
    for (int width : {2560, 640}) {
        for (int block : {128, 64}) {
            const int rows = 80;
            std::vector<float> x((size_t)rows * width), sg(width), want;
            for (auto& v : x) v = rng.sym();
            for (auto& v : sg) v = (rng.next() & 1) ? 1.0f : -1.0f;
            want = x;
            for (int r = 0; r < rows; ++r) strata::kernels::cpu::hadamard_rotate(want.data() + (size_t)r * width, width, block, sg.data());
            for (int use_signs = 0; use_signs < 2; ++use_signs) {
                std::vector<float> w2 = x;
                if (!use_signs)
                    for (int r = 0; r < rows; ++r) strata::kernels::cpu::hadamard_rotate(w2.data() + (size_t)r * width, width, block, nullptr);
                const std::vector<float>& ref_out = use_signs ? want : w2;
                mtl::Buffer bx = cx.alloc(4 * x.size()), bs = cx.alloc(4 * width);
                std::memcpy(bx.ptr, x.data(), 4 * x.size());
                std::memcpy(bs.ptr, sg.data(), 4 * width);
                const mtl::Pipeline p = cx.pipeline(lib, "hadamard_rows");
                HadArgs ha{(uint32_t)width, (uint32_t)use_signs, 1.0f / std::sqrt((float)block), 0};
                mtl::Dispatch d;
                d.p = &p;
                d.args = {mtl::Arg::B(bx), mtl::Arg::B(bx), mtl::Arg::B(bs), mtl::Arg::V(ha)};   // in place, as the window runs it
                d.grid[0] = width / block;
                d.grid[1] = rows;
                d.tg[0] = block;
                cx.run({d});
                double md = 0, mr = 0;
                size_t differ = 0;
                for (size_t i = 0; i < x.size(); ++i) {
                    md = std::max(md, (double)std::fabs(bx.as<float>()[i] - ref_out[i]));
                    mr = std::max(mr, (double)std::fabs(ref_out[i]));
                    differ += bx.as<float>()[i] != ref_out[i];
                }
                const bool ok = md <= 1e-6 * mr;
                all_ok &= ok;
                std::printf("hadamard width %4d block %3d signs %d, %d rows: max abs diff %.3e, %zu of %zu values differ from hadamard_rotate  %s\n",
                            width, block, use_signs, rows, md, differ, x.size(), ok ? "PASS" : "FAIL");
            }
        }
    }
    std::printf("%s\n", all_ok ? "ALL PASS" : "FAILED");
    return all_ok ? 0 : 1;
}
