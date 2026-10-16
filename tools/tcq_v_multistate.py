#!/usr/bin/env python3
"""Multi-state trellis-coded quantization for the 2-bit V path (4, 8, 16 states, Ungerboeck 1-D trellises).

Grid      y_j = (j - 3.5) * delta,  j = 0..7;  subsets D_k = {j : j % 4 == k};  a state's union is
          A0 = D0 u D2 or A1 = D1 u D3 (chosen by the state);  NEXT[s, k] = next state after subset k.
Encoder   Viterbi (min total squared error) over the trellis; one point per value from the state's union.
Index     2 bits per value = the point's position inside the state's union; the decoder runs the same state
          machine, so the index stream alone is the code. Each block starts in state 0; no termination.
Storage   identical to the scalar 2-bit path: 2 bits per value + one f16 scale per block.
Trellises come from tcq_trellis_gen.make_trellis(nu, h0, h1).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tcq_trellis_gen import check, make_trellis  # noqa: E402

TRELLIS = {4: make_trellis(2, 0o5, 0o2), 8: make_trellis(3, 0o13, 0o04), 16: make_trellis(4, 0o23, 0o04)}
for _t in TRELLIS.values():
    check(*_t)
DELTA: dict[int, float] = {}  # filled by fit_delta / gaussian_report


def f16(x):
    return np.asarray(x, np.float32).astype(np.float16).astype(np.float32)


def grid(delta: float) -> np.ndarray:
    return (np.arange(8) - 3.5) * delta


def viterbi(u: np.ndarray, delta: float, NEXT: np.ndarray) -> np.ndarray:
    """u (M, L) -> chosen grid index j (M, L) in 0..7, minimizing total squared error over the trellis."""
    M, L = u.shape
    NS = NEXT.shape[0]
    y = grid(delta)
    cost = np.full((M, NS), np.inf)
    cost[:, 0] = 0.0
    bs = np.zeros((L, M, NS), np.int64)
    bj = np.zeros((L, M, NS), np.int64)
    rows = np.arange(M)
    for t in range(L):
        x = u[:, t]
        e = np.empty((M, 4))
        jb = np.empty((M, 4), np.int64)
        for k in range(4):  # best point inside each subset {k, k+4}
            d0 = (x - y[k]) ** 2
            d1 = (x - y[k + 4]) ** 2
            p1 = d1 < d0
            e[:, k] = np.where(p1, d1, d0)
            jb[:, k] = np.where(p1, k + 4, k)
        new = np.full((M, NS), np.inf)
        nbs = np.zeros((M, NS), np.int64)
        nbj = np.zeros((M, NS), np.int64)
        for s in range(NS):
            for k in range(4):
                ns = NEXT[s, k]
                if ns < 0:
                    continue
                c = cost[:, s] + e[:, k]
                better = c < new[:, ns]
                new[:, ns] = np.where(better, c, new[:, ns])
                nbs[:, ns] = np.where(better, s, nbs[:, ns])
                nbj[:, ns] = np.where(better, jb[:, k], nbj[:, ns])
        cost = new
        bs[t] = nbs
        bj[t] = nbj
    cur = np.argmin(cost, axis=1)
    j_out = np.zeros((M, L), np.int64)
    for t in range(L - 1, -1, -1):
        j_out[:, t] = bj[t][rows, cur]
        cur = bs[t][rows, cur]
    return j_out


def to_index(j: np.ndarray, NEXT: np.ndarray, UNION: np.ndarray) -> np.ndarray:
    """Grid points (M, L) -> 2-bit indices (position inside the state's union), walking the state machine."""
    M, L = j.shape
    s = np.zeros(M, np.int64)
    idx = np.zeros((M, L), np.int64)
    for t in range(L):
        idx[:, t] = np.argmax(UNION[s] == j[:, t, None], axis=1)
        s = NEXT[s, j[:, t] % 4]
    return idx


def from_index(idx: np.ndarray, delta: float, NEXT: np.ndarray, UNION: np.ndarray) -> np.ndarray:
    """Decoder: 2-bit indices (M, L) -> reconstructed values (M, L). This is the whole decode path."""
    M, L = idx.shape
    y = grid(delta)
    s = np.zeros(M, np.int64)
    out = np.empty((M, L))
    for t in range(L):
        j = UNION[s, idx[:, t]]
        out[:, t] = y[j]
        s = NEXT[s, j % 4]
    return out


def tcq_quantize(u: np.ndarray, delta: float, states: int = 4):
    """u (M, L) normalized values -> (reconstruction, 2-bit indices). Asserts encoder/decoder agreement."""
    NEXT, UNION = TRELLIS[states]
    j = viterbi(u, delta, NEXT)
    idx = to_index(j, NEXT, UNION)
    rec = from_index(idx, delta, NEXT, UNION)
    assert np.allclose(rec, grid(delta)[j]), "decoder does not reproduce the encoder's path"
    return rec, idx


def tcq_qt(x: np.ndarray, perm: np.ndarray | None, block: int, delta: float, states: int) -> np.ndarray:
    """Drop-in for kv_k3v2_probe.qt(x, lv, perm, block): per-token blocks, f16 RMS scale, TCQ codes."""
    N, Hh, Dd = x.shape
    y = x if perm is None else np.take_along_axis(x, np.broadcast_to(perm[None], x.shape), axis=2)
    b = y.reshape(-1, block).astype(np.float64)
    s = f16(np.sqrt(np.sum(b * b, axis=1) / block)).astype(np.float64)
    u = np.where(s[:, None] > 0, b / np.where(s > 0, s, 1)[:, None], 0.0)
    rec, _ = tcq_quantize(u, delta, states)
    out = (rec * s[:, None]).reshape(N, Hh, Dd)
    if perm is None:
        return out.astype(np.float32)
    inv = np.argsort(perm, axis=1)
    return np.take_along_axis(out, np.broadcast_to(inv[None], out.shape), axis=2).astype(np.float32)


def snr_db(mse: float, var: float = 1.0) -> float:
    return 10 * np.log10(var / mse)


def fit_delta(states: int, n: int = 400_000, seed: int = 0) -> float:
    u = np.random.default_rng(seed).standard_normal((n // 32, 32))
    best = (np.inf, None)
    for d in np.linspace(0.15, 0.70, 111):
        rec, _ = tcq_quantize(u, d, states)
        m = float(np.mean((rec - u) ** 2))
        if m < best[0]:
            best = (m, d)
    for d in np.linspace(best[1] - 0.01, best[1] + 0.01, 41):
        rec, _ = tcq_quantize(u, d, states)
        m = float(np.mean((rec - u) ** 2))
        if m < best[0]:
            best = (m, d)
    return float(best[1])


def gaussian_report(states_list) -> None:
    LM4 = np.array([-1.5097, -0.4591, 0.4591, 1.5097])
    u = np.random.default_rng(1).standard_normal((200_000 // 32, 32))
    th = (LM4[:-1] + LM4[1:]) / 2
    mse_s = float(np.mean((LM4[np.searchsorted(th, u)] - u) ** 2))
    print("Gaussian, 2 bits/value, 200k samples, blocks of 32 (MSE, SNR dB):")
    print("  scalar LM4 (engine)        mse %.4f  snr %.2f dB" % (mse_s, snr_db(mse_s)))
    for S in states_list:
        d = DELTA.get(S) or fit_delta(S)
        DELTA[S] = d
        rec, idx = tcq_quantize(u, d, S)
        m = float(np.mean((rec - u) ** 2))
        print("  TCQ-%-2d delta=%.4f         mse %.4f  snr %.2f dB  (%+.2f dB vs scalar)"
              % (S, d, m, snr_db(m), snr_db(m) - snr_db(mse_s)))
        print("         index histogram %s" % np.bincount(idx.ravel(), minlength=4))
    print("  Gaussian R(D) bound at 2 bits: 12.04 dB")


def main() -> int:
    ap = argparse.ArgumentParser(description="multi-state TCQ prototype for the 2-bit V path")
    ap.add_argument("--states", default="4,8,16")
    a = ap.parse_args()
    t0 = time.time()
    gaussian_report([int(x) for x in a.states.split(",")])
    print("time %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
