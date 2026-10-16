#!/usr/bin/env python3
"""Golden vectors for include/strata/kernels/tcq_v.hpp (host parity test src/kernels/tcq_v_host_parity.cpp).

Writes tests/data/tcq_v_golden.txt, one section per trellis (4, 8, 16 states), from the numpy reference
(tools/tcq_trellis_gen.py + tools/tcq_v_multistate.py). The C++ side must reproduce the trellis tables exactly,
decode the golden codes to the golden values, and encode the golden inputs to (almost) the same codes.

Format (whitespace separated):
  TCQ_GOLDEN 1
  per trellis:  STATES <states> DELTA <delta> BLOCKS <n> MSE <numpy mse>
                NEXT <ns*4 ints>            (NEXT[s][k], -1 = subset not allowed in state s)
                per block:  U <32 floats: normalized inputs>  Q <8 ints: packed codes>  R <32 floats: recon, scale 1>
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from tcq_v_multistate import DELTA, TRELLIS, from_index, to_index, viterbi  # noqa: E402

DELTA_FIXED = {4: 0.4945, 8: 0.4935, 16: 0.4940}  # the constants the header carries (fitted in tcq_v_multistate)
BLOCKS = 256


def pack(idx_row):
    qs = [0] * 8
    for t, c in enumerate(idx_row):
        qs[t >> 2] |= int(c) << (2 * (t & 3))
    return qs


def main():
    out = ["TCQ_GOLDEN 1"]
    rng = np.random.default_rng(20261015)
    for states in (4, 8, 16):
        NEXT, UNION = TRELLIS[states]
        delta = DELTA_FIXED[states]
        u = rng.standard_normal((BLOCKS, 32)).astype(np.float32)
        j = viterbi(u.astype(np.float64), delta, NEXT)
        idx = to_index(j, NEXT, UNION)
        rec = from_index(idx, delta, NEXT, UNION)
        mse = float(np.mean((rec - u) ** 2))
        out.append("STATES %d DELTA %.6f BLOCKS %d MSE %.8f" % (states, delta, BLOCKS, mse))
        out.append("NEXT " + " ".join(str(int(x)) for x in NEXT.ravel()))
        for b in range(BLOCKS):
            out.append("U " + " ".join("%.9g" % x for x in u[b]))
            out.append("Q " + " ".join(str(x) for x in pack(idx[b])))
            out.append("R " + " ".join("%.9g" % x for x in rec[b]))
        print("states %2d: delta %.4f, numpy mse %.5f, index hist %s"
              % (states, delta, mse, np.bincount(idx.ravel(), minlength=4)))
    dst = ROOT / "tests" / "data" / "tcq_v_golden.txt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out) + "\n")
    print("wrote %s (%d bytes)" % (dst, dst.stat().st_size))


if __name__ == "__main__":
    main()
