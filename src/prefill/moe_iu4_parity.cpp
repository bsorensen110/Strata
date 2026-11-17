// SPDX-License-Identifier: MIT
// Port note: hipfire block_i4_128_quant.hip eight-point reference search.
#include "strata/prefill/moe_iu4_reference.hpp"
#include <cfenv>
#include <cstdio>
#include <limits>

namespace i4 = strata::prefill::iu4;
namespace r4 = i4::reference;
namespace r8 = strata::prefill::iu8::reference;

int main() {
    try {
        r8::require(std::fesetround(FE_TONEAREST) == 0, "cannot set RNE");
        auto x = r8::activations(128, 37);
        for (int j = 0; j < 128; ++j) {
            x[128 + j] = j == 127 ? 7.0f : (float) ((j % 15) - 7) + 0.5f;
            x[256 + j] = std::ldexp((float) (j - 64), -125);
            x[384 + j] = (j & 1 ? -1.0f : 1.0f) * 0x1p100f;
        }
        const auto blocks = r4::quantize(x.data(), 128, 37);
        for (int t = 0; t < 37; ++t) {
            const auto& b = blocks[t];
            const auto owned = r4::quantize_block(x.data() + t * 128, true);
            r8::require(std::memcmp(&b, &owned, sizeof(b)) == 0, "A4 producer-ownership MSE order drift");
            int s = 0;
            for (int k = 0; k < 128; ++k) {
                const int q = i4::code(b, k); s += q;
                const double v = x[t * 128 + k], d = b.d;
                // Clipped endpoints are asymmetric (-8,7); remaining error <= d/2.
                const double bound = std::max({0.0, v - 7*d, -v - 8*d}) + d*0.5;
                r8::require(std::fabs(v - d*q) <= bound + std::fabs(v)*1e-7 + 1e-44,
                            "A4 round-trip exceeds clipped rounding bound");
            }
            r8::require(s == b.s, "A4 exact-sum header differs from signed nibble sum");
        }
        r8::require(blocks[0].d == 1.0f && blocks[0].s == 0, "A4 zero block reference policy");
        std::puts("PASS moe_iu4_parity (CPU A4 eight-candidate oracle; no KLD/performance claim)");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL moe_iu4_parity: %s\n", e.what()); return 1;
    }
}
