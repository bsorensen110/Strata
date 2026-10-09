// src/kernels/hc_pack.cpp - the host builder and validator for the 12-bit packed hc weights
// (include/strata/kernels/hc_pack.hpp).  Plain C++: no CUDA, no GPU, so it unit-tests on a CPU.
//
// The builder walks the matrix in the ORDER THE KERNEL DISCOVERS VALUES - CTA ascending, warp ascending, row
// ascending within the warp, chunk ascending within the row, value ascending within the chunk - because that
// is the order the escape slots are numbered in.  The kernel reaches a slot with adds and popcounts only, so
// the ordering is the whole contract, and hc_pack_validate checks it by recomputing the kernel's formula.
#include "strata/kernels/hc_pack.hpp"

#include <cstdio>
#include <cstring>

namespace strata::kernels {
namespace {

// The fused-GR read's geometry, repeated here so the pack does not depend on the kernel file.
constexpr int kN = 2560, kHc = 4, kD = kN * kHc, kLr = 320;
constexpr int kThreads = 256, kWarps = kThreads / 32;
constexpr int kDownBlocks = kLr / kWarps;              // 40, plus the injection CTA
constexpr int kUpCols = 32 / 2;                        // UPM_COLS: 16 columns of n_embd per up CTA
constexpr int kUpBlocks = kN / kUpCols;                // 160
constexpr uint32_t kNoRow = 0xFFFFFFFFu;

size_t align8(size_t b) { return (b + 7u) & ~size_t(7u); }

uint32_t slot_row_down(const void*, uint32_t block, uint32_t warp, uint32_t j) {
    return (j == 0u && block < uint32_t(kDownBlocks) && warp < uint32_t(kWarps)) ? block * kWarps + warp : kNoRow;
}
bool row_slot_down(const void*, uint32_t row, uint32_t* block, uint32_t* warp, uint32_t* j) {
    if (row >= uint32_t(kLr)) return false;
    *block = row / kWarps; *warp = row % kWarps; *j = 0u;
    return true;
}
// The injection CTA (blockIdx.x == kDownBlocks) walks rows 0..hc on its warps 0..hc-1; its other warps are
// idle, and the pack gives them no row, so they contribute no escape and no prefix.
uint32_t slot_row_inject(const void*, uint32_t block, uint32_t warp, uint32_t j) {
    return (j == 0u && block == uint32_t(kDownBlocks) && warp < uint32_t(kHc)) ? warp : kNoRow;
}
bool row_slot_inject(const void*, uint32_t row, uint32_t* block, uint32_t* warp, uint32_t* j) {
    if (row >= uint32_t(kHc)) return false;
    *block = uint32_t(kDownBlocks); *warp = row; *j = 0u;
    return true;
}
// The up read's CTA b owns columns [b*UPM_COLS, b*UPM_COLS+UPM_COLS) of every stream; warp w walks the rows
// r = w, w+WARPS, ... of its 4*UPM_COLS rows, and row c*n_embd + d has r = c*UPM_COLS + (d % UPM_COLS).
uint32_t slot_row_up(const void*, uint32_t block, uint32_t warp, uint32_t j) {
    if (block >= uint32_t(kUpBlocks) || warp >= uint32_t(kWarps) || j >= 8u) return kNoRow;
    const uint32_t c = j / 2u, dd = block * kUpCols + (j % 2u) * kWarps + warp;
    return c * uint32_t(kN) + dd;
}
bool row_slot_up(const void*, uint32_t row, uint32_t* block, uint32_t* warp, uint32_t* j) {
    if (row >= uint32_t(kD)) return false;
    const uint32_t c = row / uint32_t(kN), dd = row % uint32_t(kN);
    // slot_row_up's inverse: dd = block*UPM_COLS + half*WARPS + warp, so the half is (dd / WARPS) % 2 - dd /
    // UPM_COLS is the block, not the half.
    *block = dd / kUpCols; *warp = dd % kWarps; *j = 2u * c + (dd / kWarps) % 2u;
    return true;
}

HcPackGeometry make_geometry(const char* name, uint32_t rows, uint32_t row_len, uint32_t blocks,
                             uint32_t rows_per_warp, uint32_t (*sr)(const void*, uint32_t, uint32_t, uint32_t),
                             bool (*rs)(const void*, uint32_t, uint32_t*, uint32_t*, uint32_t*)) {
    HcPackGeometry g;
    g.name = name; g.rows = rows; g.row_len = row_len; g.warps_per_block = kWarps; g.blocks = blocks;
    g.rows_per_warp = rows_per_warp;
    g.groups_per_row = (row_len / HC_PACK_VALUES + HC_PACK_GROUP_CHUNKS - 1) / HC_PACK_GROUP_CHUNKS;
    g.slot_row = sr; g.row_slot = rs;
    return g;
}

}  // namespace

const HcPackGeometry& hc_pack_geometry_down() {
    static const HcPackGeometry g = make_geometry("hc_down", kLr, kD, kDownBlocks, 1, slot_row_down, row_slot_down);
    return g;
}
const HcPackGeometry& hc_pack_geometry_inject() {
    static const HcPackGeometry g =
        make_geometry("hc_inject", kHc, kD, kDownBlocks + 1, 1, slot_row_inject, row_slot_inject);
    return g;
}
const HcPackGeometry& hc_pack_geometry_up() {
    static const HcPackGeometry g =
        make_geometry("hc_up", kD, kLr, kUpBlocks, 8, slot_row_up, row_slot_up);
    return g;
}

size_t hc_pack_bytes(const HcPackGeometry& g, size_t escapes) {
    const size_t chunks = size_t(g.rows) * g.row_len / HC_PACK_VALUES;
    const size_t slots = size_t(g.blocks) * g.warps_per_block * g.rows_per_warp;
    return align8(chunks * HC_PACK_LOW_BYTES) + align8(chunks * HC_PACK_NIBBLE_BYTES) +
           align8(chunks * HC_PACK_FLAG_BYTES) + align8(escapes * 4) + align8(escapes * 2) + align8(slots * 4) +
           align8(size_t(g.rows) * g.groups_per_row * 4);
}

size_t hc_pack_count_escapes(const uint16_t* src, const HcPackGeometry& g) {
    // Count in discovery order, not storage order, and only rows the geometry actually visits. The three
    // production layouts cover every row once, but the generic CPU gate also tests synthetic geometries.
    size_t n = 0;
    if (g.slot_row == nullptr) return 0;
    for (uint32_t block = 0; block < g.blocks; ++block)
        for (uint32_t warp = 0; warp < g.warps_per_block; ++warp)
            for (uint32_t j = 0; j < g.rows_per_warp; ++j) {
                const uint32_t row = g.slot_row(g.ctx, block, warp, j);
                if (row >= g.rows) continue;
                for (uint32_t v = 0; v < g.row_len; ++v)
                    if (!hc_pack_in_window(src[size_t(row) * g.row_len + v])) ++n;
            }
    return n;
}

bool hc_pack_build(const uint16_t* src, const HcPackGeometry& g, uint8_t* arena, size_t capacity,
                   HcPackedMatrix* out, std::string* err) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
    if (g.row_len % HC_PACK_VALUES != 0u)
        return fail(std::string(g.name) + ": a row is not a whole number of 8-value chunks");
    if (g.slot_row == nullptr || g.row_slot == nullptr)
        return fail(std::string(g.name) + ": no CTA/warp walk");
    const size_t chunks = size_t(g.rows) * g.row_len / HC_PACK_VALUES;
    const size_t slots = size_t(g.blocks) * g.warps_per_block * g.rows_per_warp;
    const size_t escapes = hc_pack_count_escapes(src, g);
    const size_t need = hc_pack_bytes(g, escapes);
    if (capacity < need)
        return fail(std::string(g.name) + ": arena is short (" + std::to_string(capacity) + " < " +
                    std::to_string(need) + " bytes)");

    size_t off = 0;
    out->rows = g.rows; out->row_len = g.row_len; out->groups_per_row = g.groups_per_row;
    out->warps_per_block = g.warps_per_block; out->rows_per_warp = g.rows_per_warp;
    out->chunks = chunks; out->escapes = escapes;
    out->off_lows = off; off += align8(chunks * HC_PACK_LOW_BYTES);
    out->off_nibbles = off; off += align8(chunks * HC_PACK_NIBBLE_BYTES);
    out->off_esc_flag = off; off += align8(chunks * HC_PACK_FLAG_BYTES);
    out->off_esc_index = off; off += align8(escapes * 4);
    out->off_esc_value = off; off += align8(escapes * 2);
    out->off_esc_warp_row = off; off += align8(slots * 4);
    out->off_esc_row_group = off; off += align8(size_t(g.rows) * g.groups_per_row * 4);
    out->total_bytes = off;

    uint8_t* lows = arena + out->off_lows;
    uint8_t* nib = arena + out->off_nibbles;
    uint8_t* flags = arena + out->off_esc_flag;
    uint32_t* esc_index = (uint32_t*)(arena + out->off_esc_index);
    uint16_t* esc_value = (uint16_t*)(arena + out->off_esc_value);
    uint32_t* warp_row = (uint32_t*)(arena + out->off_esc_warp_row);
    uint32_t* row_group = (uint32_t*)(arena + out->off_esc_row_group);
    const uint32_t chunks_per_row = g.row_len / HC_PACK_VALUES;

    // 1. The two planes, in memory order: low byte = (sign << 7) | mantissa, nibble i = exponent - 111
    //    (an escape's nibble stays 0; its low byte is written the same way, and the kernel ignores both).
    for (size_t i = 0; i < chunks; ++i) {
        uint8_t lo[HC_PACK_LOW_BYTES], nb[HC_PACK_NIBBLE_BYTES] = {0, 0, 0, 0};
        uint8_t flag = 0;
        for (int v = 0; v < HC_PACK_VALUES; ++v) {
            const uint16_t bits = src[i * HC_PACK_VALUES + v];
            lo[v] = (uint8_t)(((bits >> 15u) & 1u) << 7u) | (uint8_t)(bits & 0x7Fu);
            const uint32_t exp = (bits >> 7u) & 0xFFu;
            if (hc_pack_in_window(bits))
                nb[v / 2] |= (uint8_t)((exp - uint32_t(HC_PACK_EXP_BASE)) << (4u * (v % 2u)));
            else
                flag |= (uint8_t)(1u << v);
        }
        std::memcpy(lows + i * HC_PACK_LOW_BYTES, lo, HC_PACK_LOW_BYTES);
        std::memcpy(nib + i * HC_PACK_NIBBLE_BYTES, nb, HC_PACK_NIBBLE_BYTES);
        flags[i] = flag;
    }

    // 2. The escape list and the two prefix tables, in the kernel's discovery order.
    std::memset(warp_row, 0, slots * 4);
    std::memset(row_group, 0, size_t(g.rows) * g.groups_per_row * 4);
    size_t slot = 0;
    for (uint32_t block = 0; block < g.blocks; ++block) {
        for (uint32_t warp = 0; warp < g.warps_per_block; ++warp) {
            for (uint32_t j = 0; j < g.rows_per_warp; ++j) {
                const size_t s = (size_t(block) * g.warps_per_block + warp) * g.rows_per_warp + j;
                warp_row[s] = uint32_t(slot);          // escapes this warp found before its j-th row
                const uint32_t row = g.slot_row(g.ctx, block, warp, j);
                if (row == kNoRow) continue;
                uint32_t seen = 0;   // escapes this row has found, in all groups up to and including this one
                for (uint32_t c = 0; c < chunks_per_row; ++c) {
                    const uint32_t group = c / HC_PACK_GROUP_CHUNKS;
                    for (int v = 0; v < HC_PACK_VALUES; ++v) {
                        const uint16_t bits = src[size_t(row) * g.row_len + size_t(c) * HC_PACK_VALUES + v];
                        if (hc_pack_in_window(bits)) continue;
                        esc_index[slot] = uint32_t(size_t(row) * g.row_len + size_t(c) * HC_PACK_VALUES + v);
                        esc_value[slot] = bits;
                        ++slot;
                        ++seen;
                    }
                    if (c % HC_PACK_GROUP_CHUNKS == HC_PACK_GROUP_CHUNKS - 1u || c + 1u == chunks_per_row) {
                        // the group just finished: the next group's prefix is everything found so far
                        const uint32_t next = group + 1u;
                        if (next < g.groups_per_row)
                            row_group[size_t(row) * g.groups_per_row + next] = uint32_t(seen);
                    }
                }
            }
        }
    }
    if (slot != escapes) return fail(std::string(g.name) + ": the escape walk found " + std::to_string(slot) +
                                     " of " + std::to_string(escapes) + " escapes");
    return true;
}

bool hc_pack_validate(const uint16_t* src, const HcPackGeometry& g, const HcPackedMatrix& p, const uint8_t* arena,
                      std::string* err) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
    const uint8_t* lows = arena + p.off_lows;
    const uint8_t* nib = arena + p.off_nibbles;
    const uint8_t* flags = arena + p.off_esc_flag;
    const uint16_t* esc_value = (const uint16_t*)(arena + p.off_esc_value);
    const uint32_t* warp_row = (const uint32_t*)(arena + p.off_esc_warp_row);
    const uint32_t* row_group = (const uint32_t*)(arena + p.off_esc_row_group);
    const uint32_t chunks_per_row = g.row_len / HC_PACK_VALUES;
    uint32_t block, warp, j;
    size_t escapes_seen = 0;

    for (uint32_t row = 0; row < g.rows; ++row) {
        if (!g.row_slot(g.ctx, row, &block, &warp, &j))
            return fail(std::string(g.name) + ": row " + std::to_string(row) + " is walked by no CTA");
        const uint32_t warp_base = warp_row[(size_t(block) * g.warps_per_block + warp) * g.rows_per_warp + j];
        // The kernel's per-group, per-lane and per-value counts, replayed in the kernel's order.
        for (uint32_t c = 0; c < chunks_per_row; ++c) {
            const uint32_t group = c / HC_PACK_GROUP_CHUNKS, lane = c % HC_PACK_GROUP_CHUNKS;
            uint32_t lane_before = 0, value_before = 0;
            for (uint32_t cl = 0; cl < lane; ++cl)
                for (int v = 0; v < HC_PACK_VALUES; ++v)
                    if (!hc_pack_in_window(src[size_t(row) * g.row_len + size_t(group) * HC_PACK_GROUP_CHUNKS *
                                                  HC_PACK_VALUES + size_t(cl) * HC_PACK_VALUES + v]))
                        ++lane_before;
            uint8_t chunk_flag = 0;
            for (int v = 0; v < HC_PACK_VALUES; ++v) {
                const uint16_t bits = src[size_t(row) * g.row_len + size_t(c) * HC_PACK_VALUES + v];
                if (!hc_pack_in_window(bits)) chunk_flag |= (uint8_t)(1u << v);
                // Compose the 12 bits back the way the kernel does.
                uint8_t lo = 0, nb = 0;
                std::memcpy(&lo, lows + (size_t(row) * chunks_per_row + c) * HC_PACK_LOW_BYTES + v, 1);
                std::memcpy(&nb, nib + (size_t(row) * chunks_per_row + c) * HC_PACK_NIBBLE_BYTES + v / 2, 1);
                const uint32_t nib_v = (nb >> (4u * (v % 2u))) & 0xFu;
                const uint16_t got = (uint16_t)((((lo >> 7u) & 1u) << 15u) | ((uint32_t(HC_PACK_EXP_BASE) + nib_v) << 7u) |
                                                (lo & 0x7Fu));
                if (hc_pack_in_window(bits)) {
                    if (got != bits) {
                        char b[160];
                        std::snprintf(b, sizeof b, "%s: row %u value %u unpacks to %04x, not %04x", g.name, row,
                                      unsigned(size_t(c) * HC_PACK_VALUES + v), unsigned(got), unsigned(bits));
                        return fail(b);
                    }
                    continue;
                }
                ++escapes_seen;
                const uint32_t slot = warp_base + row_group[size_t(row) * g.groups_per_row + group] + lane_before +
                                       value_before;
                if (slot >= p.escapes) return fail(std::string(g.name) + ": escape slot out of range");
                if (esc_value[slot] != bits) {
                    char b[192];
                    std::snprintf(b, sizeof b,
                                  "%s: row %u value %u is an escape (exponent %u) at slot %u, which holds %04x, "
                                  "not %04x",
                                  g.name, row, unsigned(size_t(c) * HC_PACK_VALUES + v), unsigned((bits >> 7) & 0xFF),
                                  unsigned(slot), unsigned(esc_value[slot]), unsigned(bits));
                    return fail(b);
                }
                ++value_before;
            }
            if (flags[size_t(row) * chunks_per_row + c] != chunk_flag) {
                char b[128];
                std::snprintf(b, sizeof b, "%s: row %u chunk %u flag byte %02x, not %02x", g.name, row,
                              unsigned(c), unsigned(flags[size_t(row) * chunks_per_row + c]), unsigned(chunk_flag));
                return fail(b);
            }
        }
    }
    if (escapes_seen != p.escapes)
        return fail(std::string(g.name) + ": " + std::to_string(escapes_seen) + " escapes reached, " +
                    std::to_string(p.escapes) + " packed");
    return true;
}

HcPackedWeights hc_pack_device_view(const HcPackedMatrix& p, uint8_t* device_base) {
    HcPackedWeights w;
    w.lows = device_base + p.off_lows;
    w.nibbles = device_base + p.off_nibbles;
    w.flags = device_base + p.off_esc_flag;
    w.esc_value = (const uint16_t*)(device_base + p.off_esc_value);
    w.esc_warp_row = (const uint32_t*)(device_base + p.off_esc_warp_row);
    w.esc_row_group = (const uint32_t*)(device_base + p.off_esc_row_group);
    w.groups_per_row = p.groups_per_row; w.warps_per_block = p.warps_per_block; w.rows_per_warp = p.rows_per_warp;
    return w;
}

}  // namespace strata::kernels
