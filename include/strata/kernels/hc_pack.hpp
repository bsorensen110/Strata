// include/strata/kernels/hc_pack.hpp - the 12-bit packed image of the hyper-connection BF16 weights
// (STRATA_HC_PACK=1; hc-rdna4-proposals-20261009 section 3.2).  Host side: build it from the artifact's BF16
// bytes and validate the round trip.  Device side: src/kernels/cuda/fused_gr.cu composes it back to the
// identical BF16 bits in registers, inside the fused-GR read.
//
// The pack is 12 bytes per 8 values - 25% below BF16 - split into two planes so a warp's loads stay coalesced:
// per 8-value chunk, 8 LOW bytes and 4 NIBBLE bytes.
//
//   low byte i  : bits[6:0] = value i's 7-bit mantissa, bit 7 = value i's sign
//   nibble  i   : value i's exponent INDEX, in nibble i of byte i/2 (the low nibble is the even i);
//                 exponent = HC_PACK_EXP_BASE + index
//   composition: bits = (sign << 15) | ((111 + index) << 7) | mantissa
//
// A lane reads one chunk: 8 bytes of lows (one uint2) and 4 bytes of nibbles (one u32), so a warp reads a
// contiguous 256-byte run of lows and a contiguous 128-byte run of nibbles.  12 bytes per 8 values is not
// 16-byte aligned, and no cp.async / global_load_async_to_lds exists on gfx1201; the plane split is what
// keeps the loads coalesced.
//
// A third plane, HC_PACK_FLAG_BYTES per chunk, carries the escape FLAGS: bit i of the chunk's flag byte says
// value i is an escape.  It is needed and cannot be folded into the 12 bytes: the window [111,126] uses all 16
// nibble values and the low byte uses all 8 bits, so no in-band marker can say "this value escaped" - an
// escape's nibble 0 composes to exponent 111, a legal value.  With the flag plane the pack is 13 bytes per 8
// values (1.625 B/value, 18.75% below BF16's 2 B); the 8+4 planes are exactly the layout above.  The flag
// plane is a warp-coalesced 32-byte run per group (one sector).
//
// Values whose exponent falls outside [111,126] - 0.041% of the pack's hc weights, including exponent 0
// (zeros, subnormals, -0) and 255 (inf, NaN) - are ESCAPES: their nibble is left at 0 and their exact 16 bits
// go to a side list.  The list is ordered by the order the kernel discovers values, so the prefix counts below
// give a discovered escape its slot with three adds and two popcounts - no search, and the window boundary is
// exact on both sides.
//
//   slot = esc_warp_row[(block*warps_per_block + warp)*rows_per_warp + j]  // this warp's earlier rows
//        + esc_row_group[row*groups_per_row + group]                        // this row's earlier groups
//        + popc(ballot of the warp's lanes before this one)                // this group's earlier lanes
//        + popc(this chunk's escape bits below this value)                 // this chunk's earlier values
//
// `group` is chunk / HC_PACK_GROUP_CHUNKS, `lane` is chunk % HC_PACK_GROUP_CHUNKS: a warp's lanes take the 32
// chunks of one group in ascending chunk order, which is the order the kernels read them.
//
// Host-only: no CUDA here, so the builder and its validator unit-test on a CPU with no GPU (the
// hip_hc_pack_parity target).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::kernels {

constexpr int HC_PACK_EXP_BASE = 111;      // the exponent window's base: exponent = 111 + index
constexpr int HC_PACK_EXP_SPAN = 16;       // the window is [111,126]; every other exponent is an escape
constexpr int HC_PACK_VALUES = 8;          // values per chunk
constexpr int HC_PACK_LOW_BYTES = 8;       // lows per chunk: one uint2 per lane
constexpr int HC_PACK_NIBBLE_BYTES = 4;    // nibbles per chunk: one u32 per lane
constexpr int HC_PACK_FLAG_BYTES = 1;      // escape flags per chunk: bit i = value i is an escape
constexpr int HC_PACK_CHUNK_BYTES = 13;    // lows + nibbles + escape flag byte; 18.75% below 8 x BF16
constexpr int HC_PACK_GROUP_CHUNKS = 32;   // chunks per escape-prefix group: one warp's lanes

/// True when a BF16 bit pattern's exponent is inside the window (so the pack can carry it in 12 bits).
constexpr bool hc_pack_in_window(uint16_t bits) {
    const uint32_t exp = (bits >> 7u) & 0xFFu;
    return exp - uint32_t(HC_PACK_EXP_BASE) < uint32_t(HC_PACK_EXP_SPAN);
}

/// The escape-prefix geometry one matrix is packed for: the CTA/warp walk the fused-GR kernel uses.  The
/// builder emits the escape list in that walk's discovery order, and the validator recomputes a value's slot
/// with the kernel's own formula, so a geometry that does not match the kernel is caught on the CPU.
struct HcPackGeometry {
    const char* name = nullptr;
    uint32_t rows = 0;             // rows in the matrix (w_down: hc_lr; w_up: hc*n_embd; w_inject: hc)
    uint32_t row_len = 0;          // values per row, a multiple of HC_PACK_VALUES
    uint32_t warps_per_block = 8;  // the CTA's warps (THREADS/32 on the staged read)
    uint32_t blocks = 0;           // CTAs that own rows (the staged grid's, injection CTA included)
    uint32_t rows_per_warp = 1;    // rows one warp walks, in the kernel's order
    uint32_t groups_per_row = 0;   // ceil(row_len / 8 / HC_PACK_GROUP_CHUNKS)
    /// The row this (block, warp, j) walks, in the kernel's discovery order; 0xFFFFFFFF when it walks none
    /// (the injection CTA's idle warps).  The builder walks slots in ascending (block, warp, j) order.
    uint32_t (*slot_row)(const void* ctx, uint32_t block, uint32_t warp, uint32_t j) = nullptr;
    /// The inverse: where the kernel discovers this row.  False when no CTA walks it.  The validator uses it
    /// to recompute a value's slot with the kernel's formula, walking rows in memory order.
    bool (*row_slot)(const void* ctx, uint32_t row, uint32_t* block, uint32_t* warp, uint32_t* j) = nullptr;
    const void* ctx = nullptr;
};

/// The three fused-GR weight matrices, for the staged read's geometry (256 threads, 8 warps per CTA,
/// LR/8 + 1 CTAs for `down` and the injection CTA, UPM_BLOCKS CTAs for `up`).
const HcPackGeometry& hc_pack_geometry_down();     // w_down: 320 rows x 10240
const HcPackGeometry& hc_pack_geometry_inject();    // w_inject: 4 rows x 10240, on the injection CTA
const HcPackGeometry& hc_pack_geometry_up();        // w_up: 10240 rows x 320

/// The packed image of one matrix, as byte offsets in one arena: the arena is copied to the device whole and
/// the device view is built by adding the base.  Every plane starts at a multiple of 8.
struct HcPackedMatrix {
    uint32_t rows = 0, row_len = 0, groups_per_row = 0, warps_per_block = 0, rows_per_warp = 0;
    size_t chunks = 0;         // rows * row_len / HC_PACK_VALUES
    size_t escapes = 0;        // values whose exponent left the window
    size_t off_lows = 0;       // chunks * HC_PACK_LOW_BYTES
    size_t off_nibbles = 0;    // chunks * HC_PACK_NIBBLE_BYTES
    size_t off_esc_flag = 0;   // chunks * HC_PACK_FLAG_BYTES
    size_t off_esc_index = 0;  // escapes * 4: the value's row*row_len + column (host validation only)
    size_t off_esc_value = 0;  // escapes * 2: the value's exact BF16 bits, in discovery order
    size_t off_esc_warp_row = 0;    // blocks*warps_per_block*rows_per_warp * 4
    size_t off_esc_row_group = 0;   // rows*groups_per_row * 4
    size_t total_bytes = 0;
};

/// The device-side view the kernel reads: every pointer is a device pointer.
struct HcPackedWeights {
    const uint8_t* lows = nullptr;         // chunks * HC_PACK_LOW_BYTES
    const uint8_t* nibbles = nullptr;      // chunks * HC_PACK_NIBBLE_BYTES
    const uint8_t* flags = nullptr;        // chunks * HC_PACK_FLAG_BYTES
    const uint16_t* esc_value = nullptr;   // escapes, in discovery order
    const uint32_t* esc_warp_row = nullptr;
    const uint32_t* esc_row_group = nullptr;
    uint32_t groups_per_row = 0, warps_per_block = 0, rows_per_warp = 0;
};

/// Bytes the packed image of `g` with `escapes` escapes needs (call hc_pack_count_escapes first).
size_t hc_pack_bytes(const HcPackGeometry& g, size_t escapes);
/// Values of `src` (rows*row_len BF16 bits, row-major) whose exponent leaves the window.
size_t hc_pack_count_escapes(const uint16_t* src, const HcPackGeometry& g);
/// Build the image into `arena` (hc_pack_bytes(g, hc_pack_count_escapes(...)) bytes).  False, with the reason
/// in `err`, when the geometry does not tile the matrix or the arena is short.
bool hc_pack_build(const uint16_t* src, const HcPackGeometry& g, uint8_t* arena, size_t capacity,
                   HcPackedMatrix* out, std::string* err);

/// unpack(pack(x)) == x for every value, and every escape's slot is where the kernel's rank formula lands.
/// This is the gate: a single differing bit pattern, a mis-sorted escape or a prefix off by one fails it.
bool hc_pack_validate(const uint16_t* src, const HcPackGeometry& g, const HcPackedMatrix& p,
                      const uint8_t* arena, std::string* err);

/// The kernel's view of a built matrix, given the device base of the arena it was built into.
HcPackedWeights hc_pack_device_view(const HcPackedMatrix& p, uint8_t* device_base);

}  // namespace strata::kernels
