#!/usr/bin/env python3
"""Compare two dumps from dflash2_wmma_gfx12_parity (control = production FP16-ring path, cand = gfx12 WMMA branch).

Usage: python3 dflash2_wmma_gfx12_parity_compare.py ctl.bin cand.bin

Reports error metrics only. The WMMA branch rounds Q and P to FP16, so bit equality is not expected (differing-bits
variant, opt-in). The 2e-2 threshold below is a provisional tolerance for that rounding, not a verdict.
"""
import math
import struct
import sys


def load(path):
    with open(path, "rb") as f:
        blob = f.read()
    n_part, n_xq, n_xd = struct.unpack_from("<QQQ", blob, 0)
    off = 24
    part = struct.unpack_from(f"<{n_part}f", blob, off)
    off += 4 * n_part
    xq = blob[off:off + n_xq]
    off += n_xq
    xd = struct.unpack_from(f"<{n_xd}f", blob, off)
    return part, xq, xd


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    pc, xc, dc = load(sys.argv[1])
    pk, xk, dk = load(sys.argv[2])
    if len(pc) != len(pk) or len(xc) != len(xk) or len(dc) != len(dk):
        print("size mismatch")
        return 1

    # Partials: per (chunk, kvh, query) row there are 130 floats: [m, l, O[128]]. Compare finite pairs only.
    max_abs = 0.0
    max_scale = 0.0
    inf_mismatch = 0
    for i, (a, b) in enumerate(zip(pc, pk)):
        a_fin, b_fin = math.isfinite(a), math.isfinite(b)
        if a_fin != b_fin:
            inf_mismatch += 1
            continue
        if not a_fin:
            continue
        max_abs = max(max_abs, abs(a - b))
        max_scale = max(max_scale, abs(a))
    rel = max_abs / max_scale if max_scale > 0 else float("inf")

    # Merged int8 activations and per-block scales (what the next GEMV consumes).
    xq_diff = [abs(a - b) for a, b in zip(xc, xk)]
    xq_nonzero_diff = sum(1 for d in xq_diff if d)
    xq_max = max(xq_diff) if xq_diff else 0
    dmax = 0.0
    dscale = 0.0
    for a, b in zip(dc, dk):
        if math.isfinite(a) and math.isfinite(b):
            dmax = max(dmax, abs(a - b))
            dscale = max(dscale, abs(a))
    d_rel = dmax / dscale if dscale > 0 else float("inf")

    print(f"partials: n={len(pc)} finite-mismatch={inf_mismatch} max|diff|={max_abs:.6g} "
          f"max|ctl|={max_scale:.6g} rel={rel:.3e}")
    print(f"xq (int8): n={len(xc)} differing bytes={xq_nonzero_diff} max |diff|={xq_max}")
    print(f"xd (scales): max|diff|={dmax:.6g} rel={d_rel:.3e}")
    ok = inf_mismatch == 0 and rel < 2e-2 and d_rel < 2e-2
    print("provisional: within 2e-2 relative tolerance" if ok else "provisional: OUTSIDE 2e-2 relative tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
