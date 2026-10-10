// include/strata/kernels/fused_gr.hpp - plan v0.3 P3: the hyper-connection read in TWO kernels, with the
// previous half's write folded in.
//
// The native path spends six kernels per `gr_read` (norm, down MMVF, silu, up MMVF, gate+mean, inject MMVF) and
// one per `gr_write`, 96 + 96 times per token, and its up projection (10240 rows of 320) runs one 160-thread
// block per row: 20.6 us for 6.5 MB.  Here:
//
//   fused_gr_down : R' = R + bo_prev * 2 sigmoid(inj_prev / hc)  (only when `apply`, computed on the fly)
//                   rs[c] = rsqrt(mean(R'[c]^2) + eps),  xn = R' * w_norm * rs
//                   lo[k] = silu((w_down[k] . xn) / hc)          k < hc_lr
//                   inject[c] = w_inject[c] . xn                 when w_inject is given
//   fused_gr_up   : R <- R' in place for this block's columns (when `apply`)
//                   mixed[d] = mean_c  xn[c,d] * sigmoid(w_up[c*n_embd + d] . lo)
//
// FP32 activations and BF16 weights, like the native MMVF contract; the summation order differs from it (G-C
// judges the result).  Geometry is the artifact's: n_embd 2560, hc 4, hc_lr 320.  `inj_prev` and `inject_out`
// must be different buffers (every block reads the former while one block writes the latter).
#pragma once

#include "strata/kernels/hc_pack.hpp"   // S27 STRATA_HC_PACK: the packed hc weight image (HcPackedWeights)

#include <cstdint>

namespace strata::kernels {

struct FusedGrArgs {
    const float* R = nullptr;          ///< (hc, n_embd), read by `down`; `up` updates it in place when apply
    float* R_out = nullptr;            ///< == R for the in-place update
    bool apply = false;                ///< fold the previous half's gr_write
    const float* bo_prev = nullptr;    ///< that half's block output, n_embd
    const float* inj_prev = nullptr;   ///< that half's injection, hc
    const float* w_norm = nullptr;     ///< (hc * n_embd) f32
    const uint16_t* w_down = nullptr;  ///< bf16 [hc_lr][hc*n_embd]
    const uint16_t* w_up = nullptr;    ///< bf16 [hc*n_embd][hc_lr]
    const uint16_t* w_inject = nullptr;///< bf16 [hc][hc*n_embd], or null (the final mixer)
    /// S27 STRATA_HC_PACK=1 (hc-rdna4-proposals-20261009 3.2): device-view descriptors for the 12-bit packed
    /// images. These are embedded by value in FusedGrArgs (the descriptor's plane pointers are device pointers),
    /// never host pointers to descriptor structs: GrMulti is passed by value to a GPU kernel. A null `lows` keeps
    /// that matrix's plain BF16 read; pack_inject stays empty when w_inject is null.
    HcPackedWeights pack_down{};    ///< packed w_down; `lows == nullptr` means plain BF16
    HcPackedWeights pack_up{};      ///< packed w_up; `lows == nullptr` means plain BF16
    HcPackedWeights pack_inject{};  ///< packed w_inject, or empty with w_inject null
    float eps = 1e-6f;
    float* lo = nullptr;               ///< workspace, hc_lr floats
    float* rs = nullptr;               ///< workspace, hc floats
    float* inject_out = nullptr;       ///< hc floats (when w_inject)
    float* mixed = nullptr;            ///< n_embd
    /// S23 experiment (STRATA_HC_Q8=1): the GGUF's Q8_0 projections (null: the BF16 ones above); fused_gr_read_multi
    /// only
    const uint8_t* q8_down = nullptr;  ///< Q8_0 [hc_lr][hc*n_embd]
    const uint8_t* q8_up = nullptr;    ///< Q8_0 [hc*n_embd][hc_lr]
    const uint8_t* q8_inject = nullptr;///< Q8_0 [hc][hc*n_embd]
    /// S26 STRATA_QFUSE=1 (fused_gr_read_multi, the default and Q8_0 reads): also write `mixed`'s q8_1 image here (the
    /// bytes native_quantize_q8_1 would write); q8_cnt = n_embd / 32 zeroed counters owned by the caller (token 0's)
    uint8_t* q8_mixed = nullptr;
    unsigned* q8_cnt = nullptr;
};

bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr);
void fused_gr_read(const FusedGrArgs& a, void* stream);

/// Plan v0.3 P6: the same read for up to 8 tokens that share the weights (a verify window): the weights are read
/// once for all of them.  `a[t]` is token t's arguments (its own R, pending write, lo, rs, inject, mixed; the four
/// weight pointers and eps must be the same for every t); `xn_scratch` is n_tok * hc * n_embd floats.  Every
/// token's outputs are bitwise `fused_gr_read(a[t])`.
constexpr int kFusedGrMaxT = 8;
/// Returns true when it also wrote the q8_1 images (every token's q8_mixed set and this read supports it).
bool fused_gr_read_multi(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream,
                         unsigned long long* stamp_buf = nullptr, int stamp_i0 = 0);

/// The bench only: the AMD latency-hidden kernels on (1) or off (0); -1 = STRATA_GR_FAST.
void fused_gr_set_fast(int on);
/// The multi read's variants (#315; not main's opt-in STRATA_GR_V3 read, which sums in another order), all computing
/// every output with the plain read's operations in its order, so bitwise the plain read's and the single-token
/// read's: plain (0.1.31's default: the norm one block per token, the down projection on 41 blocks), split (the norm
/// one block per token and stream, then the plain down projection), staged (split's norm, with the staged down
/// projection), small-CTA staged (same staged kernel using four warps per block), two rows per warp on a
/// four-warp CTA (same eight rows per CTA as the baseline), and two pipelined staged reads that carry the next
/// activation tile in a register tuple (the full tuple, and the tuple split in two halves). `fused_gr_check`
/// compares the variants against the plain read on the current card with random weights and inputs (1..8 tokens,
/// with and without the pending write, with and without the inject weights) - the tuple variants against the
/// staged read - before using one. Leaving STRATA_HC_SPLIT unset takes the promoted default: the register-half
/// read (=7) on a card whose check latched it (bit-for-bit equal to staged at every T), staged on a card that
/// did not; =2 forces staged on every card (the escape hatch). =0 keeps plain, =1 stops at split, =2 uses staged, =3 opts into
/// the four-warp staged experiment, =5 opts into the four-warp/two-row-per-warp reuse experiment, =6 opts into the
/// register-tuple pipeline and =7 into its half-tuple form, =8 into the row split (four rows per CTA, a warp
/// pair per row) and =9 into the staged read with its per-token accumulators in dynamic LDS (the same sums in
/// the same order; a launch carries the accumulator row on top of the two staged tiles, so it slices to fewer
/// tokens per launch than staged), and =10 into the register pipeline with both register arrays in halves (the
/// 6/7 combination: the half-tuple activation schedule of 7, with the weight prefetch split the same way); any
/// other value takes staged, and a tuple variant that fails its check falls back to staged. 10 is the first
/// two-character value the parser reads - see `env_variant` in the source. The check runs once per card
/// (Verifier::init calls it) and prints the selected path. On an unchecked card, `fused_gr_variant` is plain
/// unless STRATA_HC_SPLIT explicitly names a variant.
void fused_gr_check();
int fused_gr_variant();
/// The path a launch of `T` tokens actually takes: the latched variant, except that the half-tuple read (7) and
/// the pipe+half read (10) are clamped to staged above their own measured boundary
/// (`kHcHalfWinMaxT` / `kHcPipeHalfWinMaxT`, both kFusedGrMaxT on the card they were measured on).  Every other
/// variant is returned unchanged.
int fused_gr_variant_for_T(int T);
/// The bench and the loader: 1 when the packed staged arm is selected (STRATA_HC_PACK=1 on a card whose self-test
/// passed it), 0 otherwise.  Latched per device with the variant, and logged by fused_gr_check.
int fused_gr_hc_pack();

}  // namespace strata::kernels
