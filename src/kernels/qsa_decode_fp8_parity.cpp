// SPDX-License-Identifier: MIT
#include "strata/kernels/qsa_decode_fp8_reference.hpp"
#include <cstdio>
#include <cstring>

namespace fp = strata::kernels::fp8_probe;
namespace ref = fp::reference;

int main() {
    try {
        bool on = true;
        ref::require(fp::resolve(nullptr, on) && !on && fp::resolve("0", on) && !on &&
                     fp::resolve("1", on) && on, "probe default/strict modes");
        for (const char* bad : {"", "true", "01", "1x", "-1", " 1"})
            ref::require(!fp::resolve(bad, on), "invalid probe mode admitted");
        for (const char* arch : {static_cast<const char*>(nullptr), "gfx1200", "gfx12010", "gfx1202", "gfx1250", "gfx1151", "gfx1100"}) {
            ref::require(!fp::exact_arch(arch), "unqualified FP8 target admitted");
            ref::require(fp::admit(true, arch, true, 32, 24, 2, 256, 129, 4, 1, true, true) ==
                         fp::Refusal::arch, "target refusal hidden");
        }
        for (const char* arch : {"gfx1201", "gfx1201:sramecc-:xnack-"})
            ref::require(fp::admit(true, arch, true, 32, 24, 2, 256, 129, 4, 1, true, true) ==
                         fp::Refusal::none, "qualified packed-conversion target declined");
        ref::require(fp::admit(true, "gfx1201", false, 32, 24, 2, 256, 129, 4, 1, true, true) ==
                     fp::Refusal::conversion, "image without conversions admitted");
        ref::require(fp::admit(true, "gfx1201", true, 64, 24, 2, 256, 129, 4, 1, true, true) ==
                     fp::Refusal::wave, "wave64 admitted");
        ref::require(fp::admit(true, "gfx1201", true, 32, 32, 2, 256, 129, 4, 1, true, true) ==
                     fp::Refusal::shape, "uninstantiated GQA16 admitted");
        ref::require(fp::admit(true, "gfx1201", true, 32, 24, 2, 256, 129, 4, 1, false, true) ==
                     fp::Refusal::format, "quantized pool admitted as FP8");
        ref::require(fp::admit(true, "gfx1201", true, 32, 24, 2, 256, 129, 4, 2, true, true) ==
                     fp::Refusal::queries, "prefill admitted");
        for (int64_t cap : {int64_t{0}, int64_t{INT_MAX}, int64_t{1} << 32})
            ref::require(fp::admit(true, "gfx1201", true, 32, 24, 2, 256, cap, 4, 1, true, true) ==
                         fp::Refusal::bounds, "narrowing overflow admitted");
        ref::require(fp::admit(true, "gfx1201", true, 32, 12 * int64_t{65536}, 65536, 256, 129, 4, 1, true, true) ==
                     fp::Refusal::bounds, "grid.y wrap admitted");
        ref::require(fp::admit(true, "gfx1201", true, 32, 24, 2, 256, 129, 4, 1, true, false) ==
                     fp::Refusal::buffers, "missing buffers admitted");

        for (unsigned c = 0; c < 256; ++c) {
            if ((c & 127) == 127) { ref::require(std::isnan(ref::decode(c)), "E4M3FN NaN"); continue; }
            ref::require(ref::encode(ref::decode(c)) == c, "E4M3 representable code round trip");
        }
        for (unsigned c = 0; c < 126; ++c) {
            const float mid = (ref::decode(c) + ref::decode(c + 1)) * 0.5f;
            const unsigned even = c & 1 ? c + 1 : c;
            ref::require(ref::encode(mid) == even && ref::encode(-mid) == (128 | even), "E4M3 RNE tie");
            ref::require(ref::encode(std::nextafter(mid, 0.0f)) == c &&
                         ref::encode(std::nextafter(mid, INFINITY)) == c + 1, "E4M3 tie neighbors");
        }
        ref::require(ref::encode(65504) == 126 && ref::encode(-65504) == 254, "finite E4M3 saturation");
        ref::require(ref::decode(1) == 1.0f / 512 && ref::decode(56) == 1 &&
                     ref::decode(126) == 448, "OCP E4M3FN (not FNUZ) encoding");
        for (int group : {8, 12}) {
            const auto a = ref::attention(ref::Fixture(group, 65, 65, 4, false));
            const auto b = ref::attention(ref::Fixture(group, 65, 65, 4, false), false);
            ref::require(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) != 0, "FP8 negative control unchanged");
            const auto dead = ref::attention(ref::Fixture(group, 1, 1, 4, true));
            for (float x : dead) ref::require(x == 0, "missing page produced attention");
            // Scatter both halves of tile128: every incumbent record exactly once,
            // including the odd final chunk and a second KV head.
            constexpr int chunks = 5, kv_heads = 2;
            std::vector<int> seen(kv_heads * chunks * group);
            for (int kv = 0; kv < kv_heads; ++kv)
            for (int tile = 0; tile < (chunks + 1) / 2; ++tile)
            for (int half = 0; half < 2; ++half)
            for (int h = 0; h < group; ++h)
                if (2 * tile + half < chunks) ++seen[fp::record(kv, 2 * tile + half, h, chunks, group)];
            for (int count : seen) ref::require(count == 1, "tile128/chunk64 partial layout alias");
        }
        std::puts("PASS qsa_decode_fp8_parity: scalar E4M3, refusal policy, paged reference and planar layout; GPU untested");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL qsa_decode_fp8_parity: %s\n", e.what());
        return 1;
    }
}
