"""Generic Ungerboeck trellis generator for 1-D TCQ (4 subsets, rate 1/2 systematic feedback encoder).

Parity check h0(D), h1(D) (octal, Ungerboeck 1987 table for 1-D / amplitude modulation). Observer-form state
r = (r1..r_nu) over GF(2):  z0 = r1 (selects the union: subsets k with k % 2 == z0);  input u (free bit) gives
subset k = z0 + 2u;  next state:  r_i' = r_{i+1} ^ h0_i * z0 ^ h1_i * u   (r_{nu+1} = 0).
Returns NEXT[s, k] (-1 when subset k is not allowed in state s) and UNION[s] (the 4 grid indices of the union,
in ascending order), same shape as the 4-state table in tcq_v_prototype.py.
"""
import numpy as np


def make_trellis(nu: int, h0: int, h1: int):
    ns = 1 << nu
    NEXT = -np.ones((ns, 4), dtype=np.int64)
    UNION = np.zeros((ns, 4), dtype=np.int64)
    for s in range(ns):
        r = [(s >> i) & 1 for i in range(nu)]           # r[0] = r1
        z0 = r[0]
        UNION[s] = sorted(j for j in range(8) if j % 4 in (z0, z0 + 2))
        for u in (0, 1):
            k = z0 + 2 * u
            new = []
            for i in range(nu):
                nxt = r[i + 1] if i + 1 < nu else 0
                den = (h0 >> (i + 1)) & 1                  # h0_{i+1}
                num = (h1 >> (i + 1)) & 1                  # h1_{i+1}
                new.append(nxt ^ (den & z0) ^ (num & u))
            NEXT[s, k] = sum(b << i for i, b in enumerate(new))
    return NEXT, UNION


def check(NEXT, UNION):
    ns = NEXT.shape[0]
    indeg = np.zeros(ns, int)
    for s in range(ns):
        allowed = [k for k in range(4) if NEXT[s, k] >= 0]
        assert len(allowed) == 2, (s, allowed)
        for k in allowed:
            indeg[NEXT[s, k]] += 1
    assert np.all(indeg == 2), indeg
    return True


if __name__ == "__main__":
    for nu, h0, h1 in ((2, 0o5, 0o2), (3, 0o13, 0o04), (4, 0o23, 0o04)):
        N, U = make_trellis(nu, h0, h1)
        print(nu, check(N, U))
        print(N)
