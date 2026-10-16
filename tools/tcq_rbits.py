#!/usr/bin/env python3
"""R-bit trellis-coded quantization along a sequence (any R: 2 bits = 8-point grid, 3 bits = 16-point grid).

Grid      y_j = (j - (G-1)/2) * delta,  G = 2^(R+1);  subset k = j % 4 has G/4 points  {k, k+4, ...}
Union     state s with z0 = s & 1 allows {j : j % 2 == z0}: G/2 points, index idx = j >> 1 (R bits).
Trellis   NEXT[s, k] from tcq_trellis_gen.make_trellis (the subset structure does not depend on G).
Encoder   Viterbi: per subset, the nearest of its G/4 points; path cost summed over the sequence.
Decoder   idx -> j = 2*idx + z0 -> y_j, state advance NEXT[s, j % 4].

Used for the K side (sequence = the 32 cells of one channel group) and checked against the 2-bit numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tcq_trellis_gen import make_trellis  # noqa: E402

TRELLIS = {4: make_trellis(2, 0o5, 0o2), 8: make_trellis(3, 0o13, 0o04)}


def union_table(states: int, G: int) -> np.ndarray:
    NS = states
    U = np.zeros((NS, G // 2), dtype=np.int64)
    for s in range(NS):
        z0 = s & 1
        U[s] = [j for j in range(G) if j % 2 == z0]
    return U


def grid(R: int, delta: float) -> np.ndarray:
    G = 2 ** (R + 1)
    return (np.arange(G) - (G - 1) / 2) * delta


def viterbi(u: np.ndarray, delta: float, states: int, R: int) -> np.ndarray:
    M, L = u.shape
    G = 2 ** (R + 1)
    y = grid(R, delta)
    NEXT, _ = TRELLIS[states]
    NS = states
    cost = np.full((M, NS), np.inf)
    cost[:, 0] = 0.0
    bs = np.zeros((L, M, NS), np.int64)
    bj = np.zeros((L, M, NS), np.int64)
    rows = np.arange(M)
    for t in range(L):
        x = u[:, t]
        e = np.empty((M, 4))
        jb = np.empty((M, 4), np.int64)
        for k in range(4):
            cand = y[k::4]                              # G/4 points of subset k, in order j = k + 4m
            d = (x[:, None] - cand[None, :]) ** 2
            m = np.argmin(d, axis=1)
            e[:, k] = d[rows, m]
            jb[:, k] = k + 4 * m
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


def quantize(u: np.ndarray, delta: float, states: int = 4, R: int = 2):
    """u (M, L) normalized -> (reconstruction in units of the scale, R-bit indices). Asserts decoder agreement."""
    NEXT, _ = TRELLIS[states]
    U = union_table(states, 2 ** (R + 1))
    y = grid(R, delta)
    j = viterbi(u, delta, states, R)
    M, L = j.shape
    s = np.zeros(M, np.int64)
    idx = np.zeros((M, L), np.int64)
    for t in range(L):
        idx[:, t] = np.argmax(U[s] == j[:, t, None], axis=1)
        s = NEXT[s, j[:, t] % 4]
    # decode the indices with the decoder path and check agreement with the encoder's points
    s = np.zeros(M, np.int64)
    rec = np.empty((M, L))
    for t in range(L):
        jt = U[s, idx[:, t]]
        rec[:, t] = y[jt]
        s = NEXT[s, jt % 4]
    assert np.allclose(rec, y[j]), "decoder does not reproduce the encoder path"
    return rec, idx


def qc_tcq(x: np.ndarray, delta: float, states: int, R: int, group: int = 32, seg: int | None = None) -> np.ndarray:
    """Per-channel TCQ: for each (group of `group` consecutive cells, head, dim) the sequence runs over the cells.
    Scale per (group, head, dim) = f16(RMS over the group), as the per-channel K layout. x (N, H, D).
    seg: restart the trellis every `seg` cells inside a group (scale unchanged) - the read-path variant."""
    from kv_realq_probe import f16  # the engine's fp16 rounding

    N, Hh, Dd = x.shape
    assert N % group == 0, "cells must fill whole groups for this probe"
    g = x.reshape(N // group, group, Hh, Dd).transpose(0, 2, 3, 1).reshape(-1, group).astype(np.float64)
    s = f16(np.sqrt(np.mean(g * g, axis=1))).astype(np.float64)
    u = np.where(s[:, None] > 0, g / np.where(s > 0, s, 1)[:, None], 0.0)
    if seg is None or seg >= group:
        rec, _ = quantize(u, delta, states, R)
    else:
        assert group % seg == 0
        rec, _ = quantize(u.reshape(-1, seg), delta, states, R)
        rec = rec.reshape(u.shape)
    out = (rec * s[:, None]).reshape(N // group, Hh, Dd, group).transpose(0, 3, 1, 2).reshape(N, Hh, Dd)
    return out.astype(np.float32)


def fit_delta(states: int, R: int, n: int = 400_000, seed: int = 0) -> float:
    u = np.random.default_rng(seed).standard_normal((n // 32, 32))
    best = (np.inf, None)
    for d in np.linspace(0.15, 0.80, 131):
        rec, _ = quantize(u, d, states, R)
        m = float(np.mean((rec - u) ** 2))
        if m < best[0]:
            best = (m, d)
    for d in np.linspace(best[1] - 0.01, best[1] + 0.01, 41):
        rec, _ = quantize(u, d, states, R)
        m = float(np.mean((rec - u) ** 2))
        if m < best[0]:
            best = (m, d)
    return float(best[1])


if __name__ == "__main__":
    u = np.random.default_rng(1).standard_normal((200_000 // 32, 32))
    for R, states, d in ((2, 4, 0.4945), (3, 4, None), (3, 8, None)):
        if d is None:
            d = fit_delta(states, R)
        rec, idx = quantize(u, d, states, R)
        m = float(np.mean((rec - u) ** 2))
        print("R=%d TCQ-%d delta=%.4f mse %.5f snr %.2f dB (idx hist %s)"
              % (R, states, d, m, -10 * np.log10(m), np.bincount(idx.ravel(), minlength=2 ** R)))
