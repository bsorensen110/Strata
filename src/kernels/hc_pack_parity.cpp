// src/kernels/hc_pack_parity.cpp - the CPU gate for the 12-bit packed hc weights (STRATA_HC_PACK).
//
// No CUDA, no GPU, no downloads: it packs on the CPU and checks the two things the fused-GR packed arm needs
// to be bitwise identical to the plain read - unpack(pack(x)) == x for every BF16 bit pattern, and every
// escape's slot is exactly where the kernel's rank formula (warp prefix + row-group prefix + earlier lanes +
// earlier values in the chunk) puts it.
//
//     build/hip_hc_pack_parity
#include "strata/kernels/hc_pack.hpp"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace K = strata::kernels;
using K::HcPackGeometry;
using K::HcPackedMatrix;

namespace {

int g_fail = 0, g_checks = 0;
constexpr uint32_t kNoRow = 0xFFFFFFFFu;

void check(bool ok, const char* what) {
    ++g_checks;
    if (!ok) { ++g_fail; std::printf("FAIL: %s\n", what); }
}

// A synthetic geometry: one CTA, 8 warps, rows split evenly over the warps.  Used for the exhaustive sweeps,
// where the point is the pattern, not the fused-GR tiling.
struct Synth {
    uint32_t rows, row_len;
};
uint32_t synth_slot_row(const void* ctx, uint32_t block, uint32_t warp, uint32_t j) {
    const Synth* s = (const Synth*)ctx;
    const uint32_t per = s->rows / 8u;
    return (block == 0u && warp < 8u && j < per) ? warp * per + j : kNoRow;
}
bool synth_row_slot(const void* ctx, uint32_t row, uint32_t* b, uint32_t* w, uint32_t* j) {
    const Synth* s = (const Synth*)ctx;
    if (row >= s->rows) return false;
    const uint32_t per = s->rows / 8u;
    *b = 0u; *w = row / per; *j = row % per;
    return true;
}
HcPackGeometry synth(uint32_t rows, uint32_t row_len, Synth* ctx) {
    HcPackGeometry g;
    g.name = "synth"; g.rows = rows; g.row_len = row_len; g.warps_per_block = 8; g.blocks = 1;
    g.rows_per_warp = rows / 8u;
    g.groups_per_row = (row_len / K::HC_PACK_VALUES + K::HC_PACK_GROUP_CHUNKS - 1) / K::HC_PACK_GROUP_CHUNKS;
    g.slot_row = synth_slot_row; g.row_slot = synth_row_slot; g.ctx = ctx;
    return g;
}

// Pack + validate one matrix; reports the escape count it saw.
bool round_trip(const char* label, const uint16_t* src, const HcPackGeometry& g, size_t* escapes_out,
                double max_escape_fraction = 1.0) {
    const size_t esc = K::hc_pack_count_escapes(src, g);
    const size_t need = K::hc_pack_bytes(g, esc);
    std::vector<uint8_t> arena(need + 64, 0xA5);   // slack: a builder that overruns the arena is caught
    HcPackedMatrix p;
    std::string err;
    if (!K::hc_pack_build(src, g, arena.data(), arena.size(), &p, &err)) {
        std::printf("FAIL: %s: build: %s\n", label, err.c_str());
        ++g_fail;
        return false;
    }
    if (p.total_bytes > need) {
        std::printf("FAIL: %s: total_bytes %zu exceeds hc_pack_bytes %zu\n", label, p.total_bytes, need);
        ++g_fail;
        return false;
    }
    const size_t plane[6] = {p.off_lows, p.off_nibbles, p.off_esc_flag, p.off_esc_index, p.off_esc_value,
                             p.off_esc_warp_row};
    for (size_t i = 0; i < 6; ++i)
        check((plane[i] & 7u) == 0u, "every plane starts on a multiple of 8");
    if (!K::hc_pack_validate(src, g, p, arena.data(), &err)) {
        std::printf("FAIL: %s: validate: %s\n", label, err.c_str());
        ++g_fail;
        return false;
    }
    // The synthetic all-escape/exhaustive sweeps validate losslessness and may expand; production compression
    // is asserted by the weighted BF16-distribution fixture below.
    const size_t values = size_t(g.rows) * g.row_len;
    check(double(esc) / double(values) <= max_escape_fraction, "escape fraction meets this fixture's bound");
    ++g_checks;
    std::printf("ok: %-22s %u x %u  escapes %7zu (%5.3f%%)  packed %8zu B vs bf16 %8zu B  (%4.2f B/value)\n", label,
                g.rows, g.row_len, esc, 100.0 * double(esc) / double(values), p.total_bytes, values * 2u,
                double(p.total_bytes) / double(values));
    if (escapes_out) *escapes_out = esc;
    return true;
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("hc_pack_parity: window [%d,%d], %d+%d+%d bytes per 8 values\n", K::HC_PACK_EXP_BASE,
                K::HC_PACK_EXP_BASE + K::HC_PACK_EXP_SPAN - 1, K::HC_PACK_LOW_BYTES, K::HC_PACK_NIBBLE_BYTES,
                K::HC_PACK_FLAG_BYTES);

    // 1. The window's edges, and the patterns the pack must never mangle.  The exponent is bits[14:7]:
    //    exponent 111 is 0x3780, 126 is 0x3F00, 110 is 0x3700, 127 is 0x3F80.
    check(K::hc_pack_in_window(0x3780u), "exponent 111 is in the window");
    check(K::hc_pack_in_window(0x3F00u), "exponent 126 is in the window");
    check(!K::hc_pack_in_window(0x3700u), "exponent 110 is an escape");
    check(!K::hc_pack_in_window(0x3F80u), "exponent 127 is an escape");
    check(!K::hc_pack_in_window(0x0000u), "zero is an escape");
    check(!K::hc_pack_in_window(0x8000u), "-0 is an escape");
    check(!K::hc_pack_in_window(0x0001u), "the smallest subnormal is an escape");
    check(!K::hc_pack_in_window(0x7F80u), "+inf is an escape");
    check(!K::hc_pack_in_window(0xFF80u), "-inf is an escape");
    check(!K::hc_pack_in_window(0x7FFFu), "a NaN is an escape");
    check(!K::hc_pack_in_window(0xFFFFu), "a NaN with the sign bit is an escape");
    check(K::hc_pack_in_window(0xBF00u), "a negative exponent-126 value is in the window");

    // 2. Every one of the 65536 BF16 bit patterns, in order, through pack + validate: the in-window ones
    //    compose back bit-exactly, the 0x10000 - 16*256 = 61440 out-of-window ones get an escape slot (the window
    //    holds 16 exponents x 2 signs x 128 mantissas = 4096 patterns).
    {
        Synth s{2048u, 32u};
        const HcPackGeometry g = synth(s.rows, s.row_len, &s);
        std::vector<uint16_t> src(65536);
        for (uint32_t p = 0; p < 65536u; ++p) src[p] = (uint16_t)p;
        size_t esc = 0;
        round_trip("all 65536 patterns", src.data(), g, &esc);
        check(esc == 65536u - 16u * 256u, "the exhaustive sweep's escape count is the out-of-window patterns");
    }

    // 3. Worst case for the prefix tables: every value is an escape (zeros), so every lane, every group, every
    //    warp and every row contributes to the counts, and the list is as long as it can be.
    {
        Synth s{512u, 512u};
        const HcPackGeometry g = synth(s.rows, s.row_len, &s);
        std::vector<uint16_t> src(size_t(s.rows) * s.row_len, 0x0000u);
        size_t esc = 0;
        round_trip("every value an escape", src.data(), g, &esc);
        check(esc == size_t(s.rows) * s.row_len, "the all-escape sweep escapes every value");
    }

    // 4. Escapes pinned to the boundaries the rank formula adds across: the first and last value of a chunk,
    //    the first and last chunk of a group (lane 0 / lane 31), and one row per warp.
    {
        Synth s{64u, 512u};
        const HcPackGeometry g = synth(s.rows, s.row_len, &s);
        const uint32_t cpr = s.row_len / K::HC_PACK_VALUES;
        std::vector<uint16_t> src(size_t(s.rows) * s.row_len, 0x3780u);   // exponent 111, in the window
        for (uint32_t row = 0; row < s.rows; ++row)
            for (uint32_t c = 0; c < cpr; ++c) {
                const uint32_t base = row * s.row_len + c * K::HC_PACK_VALUES;
                src[base] = 0x0000u;                       // first value of the chunk
                src[base + 7] = 0x7F80u;                   // last value: +inf
                if (c % K::HC_PACK_GROUP_CHUNKS == 0u) src[base + 3] = 0x8000u;      // lane 0 of the group
                if (c % K::HC_PACK_GROUP_CHUNKS == K::HC_PACK_GROUP_CHUNKS - 1u)
                    src[base + 4] = 0xFFFFu;               // lane 31 of the group
            }
        round_trip("escapes on the boundaries", src.data(), g, nullptr);
    }

    // 5. The three real fused-GR geometries, filled with a rotating sweep of all 65536 patterns so escapes land
    //    in every (row, group, lane, value) position the rank formula walks.
    {
        const HcPackGeometry& gd = K::hc_pack_geometry_down();
        std::vector<uint16_t> src(size_t(gd.rows) * gd.row_len);
        for (uint32_t row = 0; row < gd.rows; ++row)
            for (uint32_t v = 0; v < gd.row_len; ++v)
                src[size_t(row) * gd.row_len + v] = (uint16_t)((row * 1031u + v) & 0xFFFFu);
        round_trip("hc_down 320 x 10240", src.data(), gd, nullptr);

        const HcPackGeometry& gu = K::hc_pack_geometry_up();
        src.assign(size_t(gu.rows) * gu.row_len, 0);
        for (uint32_t row = 0; row < gu.rows; ++row)
            for (uint32_t v = 0; v < gu.row_len; ++v)
                src[size_t(row) * gu.row_len + v] = (uint16_t)((row * 7u + v * 17u) & 0xFFFFu);
        round_trip("hc_up 10240 x 320", src.data(), gu, nullptr);

        const HcPackGeometry& gi = K::hc_pack_geometry_inject();
        src.assign(size_t(gi.rows) * gi.row_len, 0);
        for (uint32_t row = 0; row < gi.rows; ++row)
            for (uint32_t v = 0; v < gi.row_len; ++v)
                src[size_t(row) * gi.row_len + v] = (uint16_t)((row * 4093u + v) & 0xFFFFu);
        round_trip("hc_inject 4 x 10240", src.data(), gi, nullptr);

        // The model's BF16 weights concentrate exponents near [111,126]; this seeded bell-shaped sample
        // creates a representative mix with rare out-of-window escapes and tests the compression claim.
        std::mt19937 rng(0x12B17u);
        std::normal_distribution<float> normal(0.0f, 0.02f);
        auto to_bf16 = [](float x) {
            uint32_t bits;
            std::memcpy(&bits, &x, sizeof(bits));
            return uint16_t((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
        };
        const size_t n = size_t(gd.rows) * gd.row_len;
        std::vector<uint16_t> gaussian(n);
        for (uint16_t& x : gaussian) x = to_bf16(normal(rng));
        round_trip("hc_down BF16 normal", gaussian.data(), gd, nullptr, 0.01);
        check(K::hc_pack_bytes(gd, K::hc_pack_count_escapes(gaussian.data(), gd)) < n * 2u,
              "the representative BF16 pack compresses below BF16");

        // The geometries must tile the matrices exactly: whole 8-value chunks, and every row owned by a CTA.
        check(gd.row_len % K::HC_PACK_VALUES == 0u && gu.row_len % K::HC_PACK_VALUES == 0u &&
                  gi.row_len % K::HC_PACK_VALUES == 0u,
              "every row is a whole number of 8-value chunks (D = 10240, LR = 320, so 8 divides both)");
        uint32_t b, w, j;
        check(gd.row_slot(gd.ctx, gd.rows - 1, &b, &w, &j) && b == 39u && w == 7u, "the last hc_down row is on the last row CTA");
        check(gu.row_slot(gu.ctx, gu.rows - 1, &b, &w, &j) && b == 159u && w == 7u && j == 7u,
              "the last hc_up row is on the last CTA, warp 7, its 8th row");
        check(gi.slot_row(gi.ctx, 40u, 3u, 0u) == 3u && gi.slot_row(gi.ctx, 40u, 4u, 0u) == kNoRow,
              "the injection CTA walks rows 0..3 and its other warps walk none");
        // hc_down / hc_inject rows are 10240 values = 1280 chunks = 40 groups of 32; hc_up rows are 320 = 40
        // chunks = 2 groups.
        check(gd.groups_per_row == 40u && gu.groups_per_row == 2u && gi.groups_per_row == 40u,
              "the escape-prefix group counts match the tiles");
    }

    std::printf("hc_pack_parity: %d checks, %d failing\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
