#!/usr/bin/env python3
"""Bitshift trellis-coded scalar quantizer (QTIP-style), R bits per sample, O(1) decode. Planning item P2.

State s in [0, 2^L). One step with R new bits b:  s' = ((s << R) | b) & (2^L - 1).
Reconstruction of step t is C[s_t], where C is a codebook with 2^L entries.
Decode has no trellis walk: s_t is the last L bits of the stream, so each value is one table lookup
(a sliding bit window), which is what the fused decode-attention reader needs.
Encoder: Viterbi over the 2^L states, per block (blocks start in state 0).
Codebook C: generalized Lloyd (GLA) on unit-Gaussian blocks, then reused on normalized real blocks.

Gaussian check (sanity): L = R = 2 has 4 states that each hold one value, i.e. scalar 4-level.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def predecessors(L: int, R: int) -> np.ndarray:
    """P[s', j] = predecessor state of s' for the j-th of the 2^R candidates (top R bits free)."""
    sp = np.arange(1 << L)
    j = np.arange(1 << R)
    return (j[None, :] << (L - R)) | (sp[:, None] >> R)


def viterbi(u: np.ndarray, C: np.ndarray, L: int, R: int, chunk: int = 2048) -> np.ndarray:
    """u (M, T) -> state path (M, T). Blocks start in state 0. Chunked over M to bound memory."""
    M, T = u.shape
    NS = 1 << L
    P = predecessors(L, R)
    out = np.empty((M, T), np.int64)
    for c0 in range(0, M, chunk):
        uc = u[c0:c0 + chunk]
        m = uc.shape[0]
        cost = np.full((m, NS), np.inf)
        cost[:, 0] = 0.0
        back = np.empty((T, m, NS), np.uint8)
        for t in range(T):
            cand = cost[:, P]                                   # (m, NS, 2^R)
            j = np.argmin(cand, axis=2)
            best = np.take_along_axis(cand, j[..., None], axis=2)[..., 0]
            cost = best + (uc[:, t, None] - C[None, :]) ** 2
            back[t] = j
        s = np.argmin(cost, axis=1)
        states = np.empty((m, T), np.int64)
        ar = np.arange(m)
        for t in range(T - 1, -1, -1):
            states[:, t] = s
            s = P[s, back[t][ar, s]]
        out[c0:c0 + m] = states
    return out


def quantize(u: np.ndarray, C: np.ndarray, L: int, R: int):
    """u (M, T) normalized -> (reconstruction (M, T), state path (M, T)). Checks the O(1) window decoder."""
    st = viterbi(u, C, L, R)
    rec = C[st]
    mask = (1 << L) - 1
    s = np.zeros(u.shape[0], np.int64)
    for t in range(u.shape[1]):
        b = st[:, t] & ((1 << R) - 1)                           # the R bits this step emits
        s = ((s << R) | b) & mask
        assert np.array_equal(s, st[:, t]), "window decoder disagrees with the encoder path"
    return rec, st


def train_codebook(L: int, R: int, n: int = 200_000, iters: int = 10, seed: int = 0, T: int = 32) -> np.ndarray:
    """GLA: alternate Viterbi assignment and per-state centroid update on unit-Gaussian blocks."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n // T, T))
    NS = 1 << L
    # i.i.d. Gaussian init (QTIP-style). Sorted-quantile init makes each state's successors a contiguous run of
    # levels and lands in a poor optimum (measured: 8.9-9.2 dB at L=6..10, below scalar LM4).
    C = np.random.default_rng(seed + 1).standard_normal(NS)
    for _ in range(iters):
        st = viterbi(u, C, L, R).ravel()
        cnt = np.bincount(st, minlength=NS)
        sm = np.bincount(st, weights=u.ravel(), minlength=NS)
        C = np.where(cnt > 0, sm / np.maximum(cnt, 1), C)
    return C.astype(np.float64)


def snr_db(C: np.ndarray, L: int, R: int, n: int = 200_000, seed: int = 123, T: int = 32) -> float:
    u = np.random.default_rng(seed).standard_normal((n // T, T))
    rec, _ = quantize(u, C, L, R)
    return float(-10 * np.log10(np.mean((rec - u) ** 2)))


# ---- real-data entry points (same layouts as the probe's qc_tcq / tcq_qt; codebook C trained on Gaussian) ----

def qc_bs(x: np.ndarray, C: np.ndarray, L: int, R: int, group: int = 32) -> np.ndarray:
    """K side, per channel: sequence = 32 consecutive cells of one (head, dim); scale = f16 RMS of the group."""
    from kv_realq_probe import f16  # the engine's fp16 rounding

    N, Hh, Dd = x.shape
    assert N % group == 0
    g = x.reshape(N // group, group, Hh, Dd).transpose(0, 2, 3, 1).reshape(-1, group).astype(np.float64)
    s = f16(np.sqrt(np.mean(g * g, axis=1))).astype(np.float64)
    u = np.where(s[:, None] > 0, g / np.where(s > 0, s, 1)[:, None], 0.0)
    rec, _ = quantize(u, C, L, R)
    out = (rec * s[:, None]).reshape(N // group, Hh, Dd, group).transpose(0, 3, 1, 2).reshape(N, Hh, Dd)
    return out.astype(np.float32)


def qt_bs(x: np.ndarray, C: np.ndarray, L: int, R: int, perm: np.ndarray | None = None, block: int = 32) -> np.ndarray:
    """V side, per token: sequence = `block` consecutive (optionally permuted) dims of one (cell, head)."""
    from kv_realq_probe import f16

    N, Hh, Dd = x.shape
    y = x if perm is None else np.take_along_axis(x, np.broadcast_to(perm[None], x.shape), axis=2)
    b = y.reshape(N, Hh, Dd // block, block).astype(np.float64)
    s = f16(np.sqrt(np.sum(b * b, axis=-1) / block)).astype(np.float64)[..., None]
    u = np.where(s > 0, b / np.where(s > 0, s, 1), 0.0).reshape(-1, block)
    rec, _ = quantize(u, C, L, R)
    out = (rec.reshape(N, Hh, Dd // block, block) * s).reshape(N, Hh, Dd)
    if perm is not None:
        inv = np.argsort(perm, axis=1)
        out = np.take_along_axis(out, np.broadcast_to(inv[None], out.shape), axis=2)
    return out.astype(np.float32)


if __name__ == "__main__":
    import time

    from kv_k3v2_probe import fit_lloyd_max  # scalar 4-level reference (same fitter as the probe)

    LM4 = fit_lloyd_max(4)
    u = np.random.default_rng(123).standard_normal((200_000 // 32, 32))
    thr = (LM4[:-1] + LM4[1:]) / 2
    lm_rec = LM4[np.searchsorted(thr, u)]
    print("scalar LM4 reference: %.2f dB" % (-10 * np.log10(np.mean((lm_rec - u) ** 2))), flush=True)
    for L in (2, 6, 8, 10):
        t0 = time.time()
        C = train_codebook(L, 2)
        print("L=%2d (%4d states) R=2: %.2f dB  [train+eval %.0fs]" % (L, 1 << L, snr_db(C, L, 2), time.time() - t0),
              flush=True)
        np.save(Path(__file__).resolve().parent.parent / "tmp" / ("bitshift_L%d_R2.npy" % L), C)
