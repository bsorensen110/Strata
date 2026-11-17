// SPDX-License-Identifier: MIT
// Port note: hipfire gemm_mq6sym_i8.gfx1201.hip frozen CPU fold;
// CHANGELOG.md:856 measured ~0.65% RMS is a fixture gate, not a universal bound.
// CPU only: moe_iu8_parity; or --real WEIGHTS.ggml TYPE M K N [ACTIVATIONS.f32].
// The dump is exactly one raw GGML expert matrix, no inferred headers or offsets.
#include "strata/prefill/moe_iu8_reference.hpp"
#include <cfenv>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>

namespace iu8 = strata::prefill::iu8;
namespace ref = iu8::reference;

namespace {
int number(const char* s) {
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    ref::require(end != s && *end == '\0' && v > 0 && v <= INT_MAX, "invalid positive integer");
    return (int) v;
}
template<class T>
std::vector<T> read(const char* path, size_t count) {
    ref::require(count <= (size_t) (256 << 20) / sizeof(T), "dump exceeds 256 MiB oracle limit");
    std::ifstream f(path, std::ios::binary);
    std::vector<T> out(count);
    ref::require((bool) f.read((char*) out.data(), (std::streamsize) (count * sizeof(T))), "truncated dump");
    ref::require(f.peek() == std::char_traits<char>::eof(), "dump has trailing bytes");
    return out;
}

void check_adapter() {
    iu8::Pack p;
    std::string why;
    std::vector<uint8_t> iq(4 * 18);
    const uint16_t scale = 0x3000;
    for (int b = 0; b < 4; ++b) {
        std::memcpy(iq.data() + b * 18, &scale, 2);
        for (int j = 0; j < 16; ++j) iq[b * 18 + 2 + j] = (uint8_t) (j | ((15 - j) << 4));
    }
    ref::require(!iu8::pack_exact(GGML_TYPE_IQ4_NL, iq.data(), iq.size(), 1, 128, p, why) &&
                 !p.symmetric_verified && p.codes.empty() && !why.empty(), "nonlinear IQ4 must refuse");
    for (int b = 0; b < 4; ++b) std::fill_n(iq.data() + b * 18 + 2, 16, uint8_t{0x88});
    ref::require(iu8::pack_exact(GGML_TYPE_IQ4_NL, iq.data(), iq.size(), 1, 128, p, why),
                 "exact sparse IQ4 subset must admit");
    ref::require(!iu8::pack_exact(GGML_TYPE_IQ4_NL, iq.data(), iq.size() - 1, 1, 128, p, why) &&
                 !p.symmetric_verified, "truncated tensor must clear proof");
    ref::require(!iu8::pack_exact(GGML_TYPE_IQ4_NL, iq.data(), iq.size(), 1, 1088, p, why),
                 "K1088 must refuse instead of reading padding");
    auto q = ref::source(GGML_TYPE_Q4_0, 1, 128);
    const uint16_t inf = 0x7c00;
    std::memcpy(q.data(), &inf, 2);
    ref::require(!iu8::pack_exact(GGML_TYPE_Q4_0, q.data(), q.size(), 1, 128, p, why),
                 "nonfinite source must refuse");
    q = ref::source(GGML_TYPE_Q4_0, 1, 128);
    std::fill(q.begin(), q.end(), 0);
    ref::require(!iu8::pack_exact(GGML_TYPE_Q4_0, q.data(), q.size(), 1, 128, p, why),
                 "negative zero dequant must not silently become positive zero");
}

void product(const iu8::Pack& p, int n, const std::vector<float>& x) {
    const auto blocks = ref::quantize(x.data(), p.cols, n);
    const auto repeat_blocks = ref::quantize(x.data(), p.cols, n);
    ref::require(std::memcmp(blocks.data(), repeat_blocks.data(), blocks.size() * sizeof(iu8::Block)) == 0,
                 "activation quantization is nondeterministic");
    std::vector<float> a((size_t) n * p.rows), b(a.size());
    for (int t = 0; t < n; ++t)
    for (int r = 0; r < p.rows; ++r) a[(size_t) t * p.rows + r] = ref::fold(p, blocks.data(), n, t, r);
    for (int t = 0; t < n; ++t)
    for (int r = 0; r < p.rows; ++r) b[(size_t) t * p.rows + r] = ref::fold(p, repeat_blocks.data(), n, t, r);
    ref::require(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0, "fold is nondeterministic");
    const double rms = ref::rms_f16(p, x, n, a);
    std::printf("W%d M=%d K=%d N=%d: relative RMS vs F16 %.6f; deterministic bits\n",
                p.bits, p.rows, p.cols, n, rms);
    ref::require(std::isfinite(rms) && rms <= 0.0065, "RMS exceeds 0.65% fixture gate");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        ref::require(std::fesetround(FE_TONEAREST) == 0, "cannot set RNE");
        check_adapter();
        iu8::Pack p;
        std::string why;
        if (argc == 1) {
            for (int type : {GGML_TYPE_Q4_0, GGML_TYPE_Q6_K}) {
                const auto bytes = ref::source(type, 129, 256);
                const bool packed = iu8::pack_exact(type, bytes.data(), bytes.size(), 129, 256, p, why);
                ref::require(packed, why.c_str());
                product(p, 65, ref::activations(256, 65));
            }
        } else {
            ref::require((argc == 7 || argc == 8) && std::string(argv[1]) == "--real",
                         "usage: moe_iu8_parity [--real WEIGHTS TYPE M K N [ACTIVATIONS.f32]]");
            const int type = number(argv[3]), m = number(argv[4]), k = number(argv[5]), n = number(argv[6]);
            ref::require(type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_IQ4_XS ||
                         type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q6_K, "unsupported dump type");
            ref::require(k % 128 == 0 && k % ggml_blck_size((ggml_type) type) == 0, "dump needs K128/source blocks");
            const auto bytes = read<uint8_t>(argv[2], (size_t) m * ggml_row_size((ggml_type) type, k));
            if (!iu8::pack_exact(type, bytes.data(), bytes.size(), m, k, p, why)) {
                std::printf("REFUSED real tensor: %s (incumbent required)\n", why.c_str()); return 77;
            }
            ref::require((size_t) k * n <= (256 << 20) / sizeof(float), "activation extent exceeds oracle limit");
            product(p, n, argc == 8 ? read<float>(argv[7], (size_t) k * n) : ref::activations(k, n));
        }
        std::puts("PASS moe_iu8_parity (CPU oracle only; no GPU or performance claim)");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL moe_iu8_parity: %s\n", e.what()); return 1;
    }
}
