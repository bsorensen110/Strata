#!/usr/bin/env python3
"""Trellis-coded quantization (TCQ) prototype for the 2-bit V path.

4-state Ungerboeck-style trellis (Marcellin & Fischer), 2 bits per value:
  reproduction grid   y_j = (j - 3.5) * DELTA,  j = 0..7
  subsets             D_k = {j : j % 4 == k}, k = 0..3
  union per state     A0 = D0 u D2 for states 0 and 2, A1 = D1 u D3 for states 1 and 3
  next state          NEXT[s, k], k = subset of the chosen point
The encoder (Viterbi, min total squared error) picks one point per value from the state's union, so each value costs
exactly 2 index bits: the point's position inside the union. The decoder runs the same state machine, so the index
stream alone is the code. Each block starts in state 0; no termination bits (the best end state is free).

Storage matches the scalar 2-bit path: 2 bits per value + one f16 scale per block. Decode is a 4x4 table lookup.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

DELTA = 0.4945  # fitted on unit Gaussian (see fit_delta); a fixed constant in a real format, no calibration needed
NS = 4
UNION = np.array([[0, 2, 4, 6], [1, 3, 5, 7], [0, 2, 4, 6], [1, 3, 5, 7]], dtype=np.int64)
NEXT = np.array([[0, -1, 1, -1], [-1, 2, -1, 3], [1, -1, 0, -1], [-1, 3, -1, 2]], dtype=np.int64)


def f16(x):
    return np.asarray(x, np.float32).astype(np.float16).astype(np.float32)


def grid(delta: float) -> np.ndarray:
    return (np.arange(8) - 3.5) * delta


def viterbi(u: np.ndarray, delta: float) -> np.ndarray:
    """u (M, L) -> chosen grid index j (M, L) in 0..7, minimizing total squared error over the trellis."""
    M, L = u.shape
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


def to_index(j: np.ndarray) -> np.ndarray:
    """Grid points (M, L) -> 2-bit indices (position inside the state's union), walking the state machine."""
    M, L = j.shape
    s = np.zeros(M, np.int64)
    idx = np.zeros((M, L), np.int64)
    for t in range(L):
        idx[:, t] = np.argmax(UNION[s] == j[:, t, None], axis=1)
        s = NEXT[s, j[:, t] % 4]
    return idx


def from_index(idx: np.ndarray, delta: float) -> np.ndarray:
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


def tcq_quantize(u: np.ndarray, delta: float = DELTA):
    """u (M, L) normalized values -> (reconstruction, 2-bit indices). Checks encoder/decoder agreement."""
    j = viterbi(u, delta)
    idx = to_index(j)
    rec = from_index(idx, delta)
    assert np.allclose(rec, grid(delta)[j]), "decoder does not reproduce the encoder's path"
    return rec, idx


def tcq_qt(x: np.ndarray, perm: np.ndarray | None = None, block: int = 32, delta: float = DELTA) -> np.ndarray:
    """Drop-in for kv_k3v2_probe.qt(x, lv, perm, block): per-token blocks, f16 RMS scale, TCQ-4 codes."""
    N, Hh, Dd = x.shape
    y = x if perm is None else np.take_along_axis(x, np.broadcast_to(perm[None], x.shape), axis=2)
    b = y.reshape(-1, block).astype(np.float64)
    s = f16(np.sqrt(np.sum(b * b, axis=1) / block)).astype(np.float64)
    u = np.where(s[:, None] > 0, b / np.where(s > 0, s, 1)[:, None], 0.0)
    rec, _ = tcq_quantize(u, delta)
    out = (rec * s[:, None]).reshape(N, Hh, Dd)
    if perm is None:
        return out.astype(np.float32)
    inv = np.argsort(perm, axis=1)
    return np.take_along_axis(out, np.broadcast_to(inv[None], out.shape), axis=2).astype(np.float32)


def snr_db(mse: float, var: float = 1.0) -> float:
    return 10 * np.log10(var / mse)


def fit_delta(n: int = 400_000, seed: int = 0) -> float:
    u = np.random.default_rng(seed).standard_normal((n // 32, 32))
    best = (np.inf, None)
    for d in np.linspace(0.15, 0.60, 91):
        rec, _ = tcq_quantize(u, d)
        m = float(np.mean((rec - u) ** 2))
        if m < best[0]:
            best = (m, d)
    lo, hi = best[1] - 0.01, best[1] + 0.01
    for d in np.linspace(lo, hi, 41):
        rec, _ = tcq_quantize(u, d)
        m = float(np.mean((rec - u) ** 2))
        if m < best[0]:
            best = (m, d)
    return float(best[1])


def gaussian_report(delta: float) -> None:
    LM4 = np.array([-1.5097, -0.4591, 0.4591, 1.5097])
    u = np.random.default_rng(1).standard_normal((200_000 // 32, 32))
    th = (LM4[:-1] + LM4[1:]) / 2
    scal = LM4[np.searchsorted(th, u)]
    mse_s = float(np.mean((scal - u) ** 2))
    print("Gaussian, 2 bits/value, 200k samples, blocks of 32 (MSE, SNR dB):")
    print("  scalar LM4 (engine)     mse %.4f  snr %.2f dB" % (mse_s, snr_db(mse_s)))
    for d in (delta,):
        rec, idx = tcq_quantize(u, d)
        m = float(np.mean((rec - u) ** 2))
        print("  TCQ-4 delta=%.3f        mse %.4f  snr %.2f dB  (%.2f dB over scalar)"
              % (d, m, snr_db(m), snr_db(m) - snr_db(mse_s)))
        print("  index histogram: %s" % np.bincount(idx.ravel(), minlength=4))
    print("  Gaussian R(D) bound at 2 bits: 12.04 dB")


def main() -> int:
    ap = argparse.ArgumentParser(description="TCQ-4 prototype for the 2-bit V path")
    ap.add_argument("--fit", action="store_true", help="refit DELTA on a unit Gaussian and print it")
    ap.add_argument("--delta", type=float, default=DELTA)
    a = ap.parse_args()
    t0 = time.time()
    if a.fit:
        print("fitted DELTA = %.4f" % fit_delta())
    gaussian_report(a.delta)
    # round trip on structured data: a block that is all zeros, and one with a single spike
    z = np.zeros((4, 32)); z[1, 5] = 3.0
    rec, _ = tcq_quantize(z, a.delta)
    print("round trip ok on zeros/spike (max |rec|: %.3f)" % np.abs(rec).max())
    print("time %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
