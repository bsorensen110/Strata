// include/strata/kernels/kv_q2.hpp - 2-bit KV storage with Walsh-Hadamard rotation for the QSA layers
// (`--kv q2`). Design and the measurements: docs/kv-q2.md.
//
// The same shape as kv_q4.hpp, at half the code width: K and V are rotated by the orthonormal 256-point Hadamard
// matrix (fwht256_cuda, reused unchanged) before quantization, the query is rotated the same way so <Hq,Hk> =
// <q,k>, and the attention output is rotated back. What is lossy is the 2-bit rounding, and at 2 bits that
// rounding is large enough to reach the model's output - which is why this format is meant to be used with the
// FP16 sliding window + attention sinks of kv_window.hpp: the `sinks` leading cells and the `window` most recent
// cells stay exact, and only the bulk is 2-bit.
//
// ========================== THE SCALE IS PER CHANNEL, NOT PER TOKEN ==========================
//
// One fp16 scale per (KV head, dimension) for each group of 32 CONSECUTIVE CELLS - KIVI's split
// (arXiv 2402.02750) - instead of one scale per (cell, 32-value block). The reason is measured, in kv_q2_parity,
// on the same 2 bits, the same Lloyd-Max codebook and the same 80 B per cell and head; only WHAT the scale
// measures changes (docs/kv-q2.md, "Scale placement"):
//
//     scale over                              K        V        attention output
//     32 values of one cell (the old layout)  33.381%  33.413%  19.928%
//     32 cells of one dimension (this layout) 44.414%  44.524%   7.150%
//
// That is NOT an unqualified win, and it is not written as one: per-element RMS gets 33% WORSE (44.4% against
// 33.4%) while the attention output gets 2.8x BETTER (7.15% against 19.93%). The two metrics disagree on purpose.
// A scale shared across cells makes K's quantization error largely common-mode across cells, and softmax is
// invariant to a common-mode shift of the scores, so that error cancels where per-element RMS measures it and
// barely moves the output. A per-cell scale produces errors that differ cell to cell, which move the RELATIVE
// scores - and the relative scores are the whole output. The metric for this format is the attention output; the
// gate keeps printing the per-element rows, labelled as diagnostics.
//
// The gate's cache is i.i.d. Gaussian with a 3x per-cell magnitude spread and has no channel-aligned outliers -
// the structure KIVI's gain rests on. That spread is exactly what penalizes a shared channel scale per element, so
// the 2.8x is a floor, measured on data that is unkind to it; the real number needs a real cache.
//
// ========================== LAYOUT ==========================
//
// Codes are 2 bits per value, packed per DIMENSION across 32 cells. One slot per (group, head, dimension):
//
//     slot = ((cell / 32) * n_head_kv + head) * head_dim + d        // 10 bytes each
//     struct block_q2c_0 { uint16_t d; uint8_t qs[8]; };           // same 10-byte shape as block_q4_0
//     cell c of the group: its code is at bits [2*(c%4), +2) of qs[c/4]
//
// `cell` is the cell index in the pool being addressed: the logical cell for the host copy and for a fully
// resident pool (identity page table), and the RESIDENT cell `slot_page * page_size + (pos % page_size)` for a
// streamed pool's VRAM slots - so a group is 32 consecutive cells OF THE POOL, i.e. 32/page_size consecutive
// pages, and the page table's indirection never splits a group.
//
// Size: 256 dims x 10 B per 32 cells = 80 B per cell and head - byte for byte what the per-token layout costed,
// so pool sizing, kv_q2_bytes_per_head/cell, the streaming movers' byte math and the window overlay's byte math
// are unchanged. What is NOT unchanged is the movers' granularity: a group spans 32/page_size = 8 pages, so a
// streamed q2 layer grants residency a group at a time (kv_stream.cu), and its per-block copy is a byte scatter
// (a page's codes are one byte inside each of the 512 slots of its group), not one contiguous run.
//
// The fused attention reads 8 consecutive dims of one cell: with this layout those are 8 slots 10 bytes apart, and
// the lanes walk d0 = 0, 8, 16, ... so a warp reads consecutive slots and stays coalesced.
//
// The shipping codebook is the 4-level Lloyd-Max grid (q2_lm4_level) scaled by the group's RMS, for K AND V: the
// parity gate measured it 1.80x better than a uniform grid over the block max on both, and better than the 3-level
// grid on both per-element RMS and attention output. The 3-level grid (q2_lm3_level) is kept because the gate
// compares the two and because the entropy-coded "1.5-bit V" of SKVQ (arXiv 2405.06219) is a writer/reader change
// later, not a layout change now. 2 bits stored is not 1.5 bits stored, and nothing here claims otherwise.
//
// With KV streaming (--kv-resident) the codes live in the host copy and the resident slots like q4_0's (KvHostPools
// k_q2/v_q2, the same per-cell granule), so both options combine.
#pragma once

#include "strata/kernels/f16_bits.hpp"      // for STRATA_HD: __host__ __device__ under nvcc, empty on host-only
#include "strata/kernels/qsa.hpp"

#include <cmath>
#include <cstdint>

namespace strata::kernels {

inline constexpr int Q2_0 = 32;      ///< cells per group (and values per block for the codebook helpers)

#pragma pack(push, 1)
struct block_q2c_0 {
    uint16_t d;             // scale (fp16 bits): the RMS of the 32 cells of ONE dimension in this group
    uint8_t qs[Q2_0 / 4];   // 32 2-bit codes; cell c of the group at bits [2*(c%4), +2) of qs[c/4]
};
#pragma pack(pop)

static_assert(sizeof(block_q2c_0) == 10, "block_q2c_0 must be 10 bytes: 2 of scale + 8 of codes");

/// Bytes per cell and KV head: 256 dims x 10 B per 32 cells -> 80, the same 80 the per-token layout costed.
inline uint64_t kv_q2_bytes_per_head(int head_dim) { return (uint64_t) (head_dim / Q2_0) * sizeof(block_q2c_0); }

/// Bytes per cell (one token, one layer): K and V of every KV head.
inline uint64_t kv_q2_bytes_per_cell(const QsaShapes& s) {
    return (uint64_t) s.n_head_kv * kv_q2_bytes_per_head((int) s.head_dim) * 2;
}

/// The slot of (group, head, dim) in a q2 pool, as a pointer. Host and device share it, so the parity test reads
/// a pool exactly the way the kernels write it.
STRATA_HD inline block_q2c_0* q2c_slot(uint8_t* pool, long long group, int head, int dim, int n_head_kv,
                                       int head_dim) {
    return reinterpret_cast<block_q2c_0*>(pool) + (((group * n_head_kv + head) * (long long) head_dim) + dim);
}
STRATA_HD inline const block_q2c_0* q2c_slot(const uint8_t* pool, long long group, int head, int dim, int n_head_kv,
                                             int head_dim) {
    return reinterpret_cast<const block_q2c_0*>(pool) + (((group * n_head_kv + head) * (long long) head_dim) + dim);
}
/// A cell's group, and its index inside the group: cell c is at byte qs[(c%32)/4], bits [2*((c%32)%4), +2).
STRATA_HD inline long long q2c_group(long long cell) { return cell / Q2_0; }
STRATA_HD inline int q2c_index(long long cell) { return (int) (cell % Q2_0); }
/// The scale of one slot, as the pool stores it (fp16-rounded, read back exactly).
STRATA_HD inline float q2c_scale(const uint8_t* pool, long long group, int head, int dim, int n_head_kv,
                                 int head_dim) {
    return f32_from_f16(q2c_slot(pool, group, head, dim, n_head_kv, head_dim)->d);
}

/// The 4-level codebook K uses: the code in {0,1,2,3} nearest to x / d, mapped so 0 -> -3, 1 -> -1, 2 -> +1,
/// 3 -> +3. Device and host share this mapping so the parity test can reproduce the kernel bit for bit.
/// d is the block's scale; d == 0 (an all-zero block) takes the code for 0.
inline int q2_code_4level(float x, float d) {
    if (!(d > 0.f)) return 1;                              // 1 -> -1 * d == 0 when d is 0
    const float q = x * (1.0f / d);                        // the device multiplies by the reciprocal; x / d is a
    int c = (int) std::nearbyintf(q);                      // different float, so the host has to do it the same way
    if (c < -3) c = -3;                                    // (nearbyintf = nearest, ties to even = __float2int_rn)
    if (c > 3) c = 3;
    return (c + 3) >> 1;                                   // -3 -> 0, -1 -> 1, +1 -> 2, +3 -> 3
}
inline float q2_value_4level(int code, float d) { return d * (float) (2 * code - 3); }   // 0..3 -> -3,-1,+1,+3

/// The 3-level codebook V uses: 0 -> -1, 1 -> 0, 2 -> +1, and 3 is never written.
inline int q2_code_3level(float x, float d) {
    if (!(d > 0.f)) return 1;                              // 1 -> 0 * d == 0
    const float q = x * (1.0f / d);                        // reciprocal multiply, as on the device
    if (q >= 0.5f) return 2;
    if (q <= -0.5f) return 0;
    return 1;
}
inline float q2_value_3level(int code, float d) { return d * (float) (code - 1); }         // 0..2 -> -1, 0, +1

/// Lloyd-Max levels for a Gaussian source: the codebook SKVQ's gain rests on. A uniform grid over the block's
/// max puts its inner level at amax/3 ~ 0.9 sigma, so a typical value (one sigma) is quantized 0.9 sigma away
/// from where it lives; these put the inner level at 0.459 sigma, measured 1.80x better on both K and V at the
/// same 2 bits. s is the group's RMS, not its max. The levels are functions, not an array: nvcc will read a
/// namespace-scope constexpr scalar in device code, but not a constexpr array.
STRATA_HD inline float q2_lm4_level(int code) {
    return code == 0 ? -1.5097f : code == 1 ? -0.4591f : code == 2 ? 0.4591f : 1.5097f;
}
STRATA_HD inline float q2_lm3_level(int code) {
    return code == 0 ? -1.4544f : code == 2 ? 1.4544f : 0.0f;
}
constexpr float kQ2Lm4T = 0.9844f;                            // midpoint of 0.4591 and 1.5097
constexpr float kQ2Lm3T = 0.7272f;                            // midpoint of 0 and 1.4544

/// Written to mirror the device exactly: reciprocal multiply, same thresholds, and s == 0 falls through to the
/// same code the warp takes (2 for the 4-level, 1 for the 3-level), because -0.0 and +0.0 are not the same bits.
STRATA_HD inline int q2_code_lm4(float x, float s) {
    const float q = s > 0.f ? x * (1.0f / s) : 0.f;
    const float a = q < 0.f ? -q : q;
    if (a < kQ2Lm4T) return q < 0.f ? 1 : 2;
    return q < 0.f ? 0 : 3;
}
STRATA_HD inline float q2_value_lm4(int code, float s) { return s * q2_lm4_level(code); }
STRATA_HD inline int q2_code_lm3(float x, float s) {
    const float q = s > 0.f ? x * (1.0f / s) : 0.f;
    if (q >= kQ2Lm3T) return 2;
    if (q <= -kQ2Lm3T) return 0;
    return 1;
}
STRATA_HD inline float q2_value_lm3(int code, float s) { return s * q2_lm3_level(code); }

// ========================== `--kv q2t`: THE LOUD CHANNELS AT 16 LEVELS ==========================
//
// K's channel RMS spans 5.8x (layer 3) to 74.9x (layer 47) after the Hadamard rotation, so one per-channel
// scale is shared by channels that differ by more than an order of magnitude, and the loud ones eat the whole
// group scale (measured in the parity gate, commit 0fa969a5). `--kv q2t` spends the bits where they are worth
// anything: the loudest 1/4 of the channels of each KV head get a 16-level Lloyd-Max codebook at 4-bit codes,
// the rest keep the 4-level codebook at 2-bit codes they already have. V is the rotated q4_0 path (kv_q4.hpp),
// exactly as q2k's V is - the shape is q2k's, the new thing is only the K side's per-channel level split.
//
// THE CODEBOOK IS A CONSTANT, NOT A FIT. q2_lm4_level is the 4-level Lloyd-Max grid for a Gaussian source,
// scaled by the group's RMS; q2_lm16_level is the same object at 16 levels - the standard Gaussian Lloyd-Max
// points (Max, arXiv 00118), the same ones the gate's generic lloyd_max_group converges to when it is given
// enough samples. Nothing is fitted per group: the scale carries the group's magnitude, the table its shape.
// The gate pins the table against a large-sample fit (kv_q2_parity), so these constants are a checked number,
// not a remembered one.
//
// The levels are functions of the code, not an array, for the reason stated above (nvcc reads a namespace-scope
// constexpr scalar in device code, not a constexpr array). The code is 0..15, 0 the most negative level; the
// grid is symmetric, so the positive half is stated once and the sign comes from the code.
// The 16-level Gaussian Lloyd-Max grid, positive half, in units of sigma (Max, 1960; the fixed point of
// "every level is the centroid of its Voronoi cell, every boundary the midpoint of its two levels"). The
// gate fits it from 1M samples and pins this table against the fit (kv_q2_parity, "q2t codebook").
STRATA_HD inline float q2_lm16_pos(int m) {   // m = 0..7: the positive half's levels, in units of sigma
    // The Gaussian Lloyd-Max 16-level grid - the gate's CONVERGED 1M-sample fit (268 iterations, last move
    // 1e-5), which is the authority this table is pinned against. Two earlier tables failed that pin: the
    // first was invented, the second was read off from memory as "the literature values" and turned out to
    // be a different quantizer's grid entirely (up to 9% off at level 1). The fit is the grid.
    return m == 0 ? 0.13103f : m == 1 ? 0.39188f : m == 2 ? 0.66181f : m == 3 ? 0.94713f
         : m == 4 ? 1.25982f : m == 5 ? 1.62138f : m == 6 ? 2.06995f : 2.73361f;
}
STRATA_HD inline float q2_lm16_level(int code) {
    return code < 8 ? -q2_lm16_pos(7 - code) : q2_lm16_pos(code - 8);   // 0 -> -2.7336 ... 15 -> +2.7336
}
/// The 7 positive thresholds - the fit's decision boundaries, NOT the midpoints of adjacent levels (a
/// Lloyd-Max fixed point's boundaries sit below the midpoint on the side of the denser levels; the gate
/// prints both and the difference is real: worst 1.4% at T5 on the first table's midpoints).
constexpr float kQ2Lm16T1 = 0.26141f, kQ2Lm16T2 = 0.52685f, kQ2Lm16T3 = 0.80441f, kQ2Lm16T4 = 1.10354f;
constexpr float kQ2Lm16T5 = 1.44062f, kQ2Lm16T6 = 1.84568f, kQ2Lm16T7 = 2.40181f;

/// Written to mirror the device exactly, as q2_code_lm4 is: reciprocal multiply, then the 7 comparisons on
/// |q|, then the sign. s == 0 falls through to code 8 (the smallest positive level), the same code the device
/// takes, because -0.0 and +0.0 are not the same bits.
STRATA_HD inline int q2_code_lm16(float x, float s) {
    const float q = s > 0.f ? x * (1.0f / s) : 0.f;
    const float a = q < 0.f ? -q : q;
    int m = 0;
    if (a >= kQ2Lm16T1) ++m;
    if (a >= kQ2Lm16T2) ++m;
    if (a >= kQ2Lm16T3) ++m;
    if (a >= kQ2Lm16T4) ++m;
    if (a >= kQ2Lm16T5) ++m;
    if (a >= kQ2Lm16T6) ++m;
    if (a >= kQ2Lm16T7) ++m;
    return q < 0.f ? 7 - m : 8 + m;
}
STRATA_HD inline float q2_value_lm16(int code, float s) { return s * q2_lm16_level(code); }

#pragma pack(push, 1)
struct block_q2l_0 {
    uint16_t d;              // scale (fp16 bits): the RMS of the 32 cells of ONE loud dimension in this group
    uint8_t qs[Q2_0 / 2];    // 32 4-bit codes; cell c of the group at bits [4*(c%2), +4) of qs[c/2]
};
#pragma pack(pop)

static_assert(sizeof(block_q2l_0) == 18, "block_q2l_0 must be 18 bytes: 2 of scale + 16 of codes");

/// Channels per KV head: the loud set is 1/4 of head_dim (the gate's winner - top 1/4 at 16 levels), the rest
/// stay on the 4-level path. head_dim must be a multiple of 4 (256 is).
///
/// STRATA_HD, all of them, deliberately: these five numbers ARE the plane layout, and the freeze kernel, the
/// offset-restore kernel, the append, the gather, the fused attention and the window reader all have to agree
/// on them. Any one of them computing the boundary from `head_dim / 4` and a `sizeof` locally is a second
/// copy of the layout that can drift from the first, and a drift here is silent - the writer and the reader
/// land on different bytes of the same block. They call these instead.
STRATA_HD inline int q2t_loud(int head_dim) { return head_dim / 4; }
STRATA_HD inline int q2t_quiet(int head_dim) { return head_dim - head_dim / 4; }
/// Bytes of one (32-cell group, KV head) block: the quiet plane's slots first, then the loud plane's. The
/// order is fixed and the plane boundary is a constant, so a channel's slot is one addition away from the
/// block's base - which is what the frozen per-channel offset table stores.
STRATA_HD inline uint64_t q2t_quiet_bytes(int head_dim) { return (uint64_t) q2t_quiet(head_dim) * sizeof(block_q2c_0); }
STRATA_HD inline uint64_t q2t_loud_bytes(int head_dim) { return (uint64_t) q2t_loud(head_dim) * sizeof(block_q2l_0); }
STRATA_HD inline uint64_t q2t_block_bytes(int head_dim) { return q2t_quiet_bytes(head_dim) + q2t_loud_bytes(head_dim); }
/// Bytes per cell and KV head: 256 dims -> 192 quiet slots of 10 B + 64 loud slots of 18 B per 32-cell group
/// = 3,072 B per group and head = 96 B per cell and head. With V's q4_0 144 that is 240 per cell and head,
/// 480 per cell for 2 KV heads (q2k: 224 / 448, q4_0: 288 / 576).
inline uint64_t kv_q2t_bytes_per_head(int head_dim) { return q2t_block_bytes(head_dim) / Q2_0; }
inline uint64_t kv_q2t_bytes_per_cell(const QsaShapes& s) {
    return (uint64_t) s.n_head_kv * kv_q2t_bytes_per_head((int) s.head_dim) * 2;
}
/// The bitmap's size: one bit per (KV head, dimension), stored in 32-bit words. 64 B per layer at 2 heads.
inline int q2t_level_words(int n_head_kv, int head_dim) { return n_head_kv * ((head_dim + 31) / 32); }

/// A channel's slot offset inside its (group, head) block, from its class and its rank within that class:
/// `rank * 10` in the quiet plane, the plane boundary + `rank * 18` in the loud one. The freeze kernel, the
/// restore kernel and the gate's host reference all place channels with THIS and nothing else, so the offset
/// table the append and every reader add to the block's base is one function, not three that happen to agree.
STRATA_HD inline int q2t_slot_offset(bool loud, int rank, int head_dim) {
    return loud ? (int) q2t_quiet_bytes(head_dim) + rank * (int) sizeof(block_q2l_0)
                : rank * (int) sizeof(block_q2c_0);
}

/// A channel's slot, as a byte pointer: the (group, head) block's base plus the channel's frozen offset. The
/// offset table is derived from the bitmap once (kv_q2t_levels_init / the freeze kernel), so neither the
/// append kernel nor a reader scans bits to find a slot.
STRATA_HD inline uint8_t* q2t_slot(uint8_t* pool, long long group, int head, int off, int n_head_kv, int head_dim) {
    return pool + (size_t) (((long long) group * n_head_kv + head) * (long long) q2t_block_bytes(head_dim) + off);
}
STRATA_HD inline const uint8_t* q2t_slot(const uint8_t* pool, long long group, int head, int off, int n_head_kv,
                                         int head_dim) {
    return pool + (size_t) (((long long) group * n_head_kv + head) * (long long) q2t_block_bytes(head_dim) + off);
}
/// The scale of a slot, read byte-wise: the offsets are multiples of 2 (10 B and 18 B strides), so a 2-byte
/// load is aligned, but reading two bytes is alignment-proof and identical on both sides of the parity gate.
STRATA_HD inline float q2t_scale_at(const uint8_t* slot) {
    const uint16_t b = (uint16_t) ((uint16_t) slot[0] | ((uint16_t) slot[1] << 8));
    return f32_from_f16(b);
}
STRATA_HD inline void q2t_put_scale(uint8_t* slot, uint16_t sbits) {
    slot[0] = (uint8_t) (sbits & 0xFFu);
    slot[1] = (uint8_t) (sbits >> 8);
}
/// A cell's code position in each plane: the quiet slot holds four cells per byte (2 bits), the loud one two
/// cells per byte (4 bits).
STRATA_HD inline int q2t_quiet_shift(int idx) { return 2 * (idx & 3); }
STRATA_HD inline int q2t_loud_shift(int idx) { return 4 * (idx & 1); }

/// THE FROZEN LEVEL MAP (device memory, at fixed addresses - the graphs bake them in).
/// `bits` is the per-(head, dim) level bitmap: bit set = loud (16 levels, 4-bit codes), clear = quiet (the
/// existing 4-level, 2-bit q2 slot). `off` is the per-(head, dim) byte offset of that channel's slot inside
/// its (group, head) block - the prefix sum of the bitmap, precomputed so no reader scans bits. `frozen` is
/// the flag the freeze kernel tests: 0 = the bitmap has not been computed, and then every channel is quiet,
/// which is exactly the q2 layout - an unfrozen state reads and writes the same bytes it would as q2.
struct KvLevelMap {
    uint32_t* bits = nullptr;
    int32_t* off = nullptr;
    int32_t* frozen = nullptr;
    bool present() const { return bits != nullptr && off != nullptr && frozen != nullptr; }
};

/// The all-quiet starting state (offsets = d * 10, bitmap zero, frozen = 0): what a fresh state gets, and what
/// a restore of an all-zero bitmap reproduces. Called from qsa_state_init.
void kv_q2t_levels_init(uint32_t* bits, int32_t* off, int32_t* frozen, const QsaShapes& s, void* stream);

/// The offsets DERIVED FROM A BITMAP, and the flag set: the restore path's call. A conversation snapshot
/// stores the bitmap (the classes are frozen state, part of what the bytes mean) and not the offsets, so a
/// restored state re-derives the prefix sum from the bitmap it just read - the offsets are a function of the
/// bitmap, never a second copy of it that could disagree.
void kv_q2t_levels_from_bits(const uint32_t* bits, int32_t* off, int32_t* frozen, const QsaShapes& s,
                             void* stream);

/// The appends, in the same three shapes as the q2 ones (see the note above), with the per-channel class
/// dispatch: a quiet channel goes through the existing lm4 path into its q2 slot, a loud one through lm16 into
/// the loud plane. The scale rule is q2's, unchanged - a complete group's scale is the RMS of its 32 values,
/// a partial group keeps the scale its slot already carries.
/// `levels` is the frozen map; `freeze` (the prompt path only) computes it from THIS call's cells first, on
/// the same stream, so the append below sees it. See the header comment of src/kernels/cuda/kv_q2.cu for when
/// the bitmap is taken and what that assumes.
void kv_append_q2t_step(uint8_t* k_q2t, uint8_t* v_q2t, const int32_t* page_table, const int32_t* step,
                        const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                        const KvHostPools* host, const KvLevelMap& levels);
void kv_append_q2t_steps(uint8_t* k_q2t, uint8_t* v_q2t, const int32_t* page_table, const int32_t* step,
                         int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                         void* stream, const KvHostPools* host, const KvLevelMap& levels);
void kv_append_q2t(uint8_t* k_q2t, uint8_t* v_q2t, const int32_t* page_table, int64_t pos0, int64_t T,
                   const float* K, const float* V, const QsaShapes& s, void* stream, const KvHostPools* host,
                   const KvHostPools* stage, const KvLevelMap& levels, bool freeze);

/// Gather step[kStepWidth] cells named by `ids` into FP16 scratch [id][kv_head][head_dim] (still rotated):
/// K through the class dispatch, V through the plane it was written to (the q2 slot layout, which is what the
/// folded V calls of a q2t state use - the engine's q2t V is the q4_0 pool, gathered by kv_gather_q4_step).
void kv_gather_q2t_step(const uint8_t* k_q2t, const uint8_t* v_q2t, const int32_t* page_table, const int32_t* ids,
                        const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                        uint16_t* v_scratch, void* stream, const KvLevelMap& levels);

// THE APPEND comes in three shapes, because the format's unit is a GROUP of 32 cells and the callers append at
// three granularities. All three write the same bytes; they differ in how much of a group one call sees, and that
// is what decides the scale and whether a code byte needs an atomic. The rule, and which call sites take the tail
// path, is spelled out in the header comment of src/kernels/cuda/kv_q2.cu.
//
/// Append the (already rotated) cell at step[kStepPos]. With a host copy (KV streaming) it is written there too,
/// and to VRAM only if its block is resident. `v3level` selects V's codebook; K is always the 4-level one. The
/// engine appends BOTH pools in one call with v3level = false, the shipping setting (4-level Lloyd-Max for K and
/// V). True is used only by the parity gate, which measures the 3-level grid against it.
///
/// One cell is a PARTIAL group: its code byte is written with an atomic (the group's other cells own the other
/// bits of that byte), and the group keeps the scale it already carries. A group whose first cell is appended
/// here takes the PREVIOUS group's scale; group 0 takes the cell's own magnitude. Decode appends one cell per
/// launch on one stream, so these appends are sequential and the scale read is deterministic.
void kv_append_q2_step(uint8_t* k_q2, uint8_t* v_q2, const int32_t* page_table, const int32_t* step,
                       const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host = nullptr, bool v3level = true);

/// Many cells at once (a verification batch, the drafter's ring): GROUP-GRANULAR. The kernel walks the steps'
/// positions and, for every group they touch, quantizes all the cells of that group the call has, so each
/// dimension's code byte is written once per group. A group the call covers only PARTLY (a batch's tail) takes
/// the atomic path, exactly as a single-cell append does.
void kv_append_q2_steps(uint8_t* k_q2, uint8_t* v_q2, const int32_t* page_table, const int32_t* step,
                        int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                        void* stream, const KvHostPools* host = nullptr, bool v3level = true);

/// The prompt path: T consecutive (rotated) cells from pos0, K/V [T, n_head_kv, 256]; also into `stage` (identity
/// layout, the one-layer staging pool of a streamed session) when given. Group-granular: the grid is over the
/// groups the range touches, so a group fully inside the range writes its 10-byte slot once, with no atomics. The
/// range's first and last group are partial (a prompt chunk is not a multiple of 32 cells) and take the atomic
/// path.
void kv_append_q2(uint8_t* k_q2, uint8_t* v_q2, const int32_t* page_table, int64_t pos0, int64_t T, const float* K,
                  const float* V, const QsaShapes& s, void* stream, const KvHostPools* host = nullptr,
                  const KvHostPools* stage = nullptr, bool v3level = true);

/// Gather step[kStepWidth] cells named by `ids` into FP16 scratch [id][kv_head][head_dim] (still rotated).
void kv_gather_q2_step(const uint8_t* k_q2, const uint8_t* v_q2, const int32_t* page_table, const int32_t* ids,
                       const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                       uint16_t* v_scratch, void* stream, bool v3level = true);

}  // namespace strata::kernels
