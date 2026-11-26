# RDNA IQ4_XS shared packed-word loads (P1)

`STRATA_RDNA_KQ_SHR=1` opts the RDNA K-quant IQ4_XS (`FMT == 23`) route into the shared-load variant. It is off when
unset or when set to any value other than `1`. The route remains gated by the existing `STRATA_RDNA_Q6K=1` or
`STRATA_HIP_FAST_DEC=1` enable; this setting does not activate the RDNA path on its own.

Each four-lane group has duplicate packed-word addresses in lane pairs 0/2 and 1/3. In the opt-in variant, lanes 0/1
of each group load the two 32-bit words once and `__shfl` broadcasts each word to its paired lane (width 4). The
existing nibble shift, LUT decode, dot products, accumulation, and reduction are otherwise unchanged. The intended
contract is bitwise-identical output; only packed-word loads are shared.

The HIP targets were compiled and linked only. No GPU execution or parity run was performed, so bitwise parity and
performance of the shared-load variant remain unverified.
