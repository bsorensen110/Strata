#!/usr/bin/env python3
"""K 3-bit per-channel + V 2-bit per-token (SKVQ-style channel reorder), measured on REAL queries.

Reads the same dumps as kv_realq_probe.py (PREFIX.kv.bin from --dump-kv, PREFIX.q.bin from
STRATA_QSA_QDUMP, PREFIX.sel.bin from STRATA_QSA_DUMP) and reproduces the engine's attention output
(scale 1/16, query head h reads KV head h // 12, causal selections) before scoring any arm.

Codecs (all on the Hadamard-rotated cache, as the engine stores it):
  K2c : 4-level Lloyd-Max (engine LM4), per-channel: 32 consecutive cells per (head, dim), scale = group RMS, f16.
  K3c : 8-level Lloyd-Max (fitted here on N(0,1), 1M samples), same per-channel scale rule.
  V4  : engine q4_0 (32-dim blocks, per token).
  V2c : 4-level LM4, per-channel (same layout as K2c).
  V2t : 4-level LM4, per-token: 32 consecutive dims per (cell, head), scale = block RMS, f16.
        natural = dims in order; reord = dims permuted per kv head by descending channel RMS, so each
        32-dim block holds channels of similar magnitude (SKVQ's reorder). The permutation is computed
        from the first --cal cells (one engine prefill chunk); reord-oracle uses all cells (in-sample bound).
Per-channel K is invariant to reordering (a relabeling of channels), so the reorder only acts through V2t.

Bytes are per (kv head, cell): codes + 2 B scale per 32-value group; q4_0 = 18 B per 32 values.
Metric (as kv_realq_probe): e = rms(O_arm - O_fp16) / rms(V), over the sampled query rows; two views:
  nowin : every cell read from the quantized cache
  win   : sinks (first 8 cells) and the last --window cells read at FP16 (the shipping overlay)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tcq_v_prototype import DELTA as TCQ_DELTA, tcq_qt  # noqa: E402  (trellis-coded 2-bit V, same bytes)
from tcq_v_multistate import tcq_qt as tcq_qt_ms  # noqa: E402  (4/8/16-state trellises, generic)
import tcq_rbits as TQ  # noqa: E402  (R-bit trellis along a sequence: used for the K side)
from tcq_bitshift import qc_bs, qt_bs  # noqa: E402  (P2: bitshift trellis, O(1) decode)
from kv_realq_probe import (  # noqa: E402  (reuse the probe's loaders and engine constants)
    D, H_KV, H_Q, LM4, SINKS, TOPK, BLOCK, f16, q4_0, q_payload, read_records, read_sdkv, rot, sel_payload,
)

SCALE = np.float32(1.0 / 16.0)
ROOT_TMP = Path(__file__).resolve().parent.parent / "tmp"
GQA = H_Q // H_KV  # 12 query heads per KV head


def fit_lloyd_max(levels: int, n: int = 1_000_000, iters: int = 2000, seed: int = 0) -> np.ndarray:
    x = np.random.default_rng(seed).standard_normal(n)
    lv = np.quantile(x, (np.arange(levels) + 0.5) / levels)
    for _ in range(iters):
        idx = np.searchsorted((lv[:-1] + lv[1:]) / 2, x)
        cnt = np.bincount(idx, minlength=levels)
        sm = np.bincount(idx, weights=x, minlength=levels)
        new = sm / np.maximum(cnt, 1)
        done = np.max(np.abs(new - lv)) < 1e-7
        lv = new
        if done:
            break
    return np.sort(lv).astype(np.float32)


def to_levels(u: np.ndarray, lv: np.ndarray) -> np.ndarray:
    """Nearest level by midpoint thresholds (the engine's |q| < T rule for the 4-level grid)."""
    th = ((lv[:-1] + lv[1:]) / 2).astype(np.float32)
    return lv[np.searchsorted(th, u)]


def qc(x: np.ndarray, lv: np.ndarray) -> np.ndarray:
    """Per-channel: groups of 32 consecutive cells per (head, dim); one-shot append rule as kv_realq_probe.q2c."""
    n = x.shape[0]
    out = np.empty_like(x)
    prev = None
    for g0 in range(0, n, 32):
        blk = x[g0:g0 + 32]
        m = blk.shape[0]
        ss = np.sum(blk * blk, axis=0, dtype=np.float32)
        if m == 32 or prev is None:
            s_rule = np.sqrt(ss * np.float32(1.0 / m))
        else:
            s_rule = np.sqrt((ss + np.float32(32 - m) * prev * prev) * np.float32(1.0 / 32))
        s = f16(s_rule)
        qn = np.where(s > 0, blk / np.where(s > 0, s, 1), 0).astype(np.float32)
        out[g0:g0 + m] = to_levels(qn, lv) * s
        prev = s
    return out


def qt(x: np.ndarray, lv: np.ndarray, perm: np.ndarray | None = None, block: int = 32) -> np.ndarray:
    """Per-token: blocks of `block` consecutive dims (after an optional per-head channel permutation)."""
    N, Hh, Dd = x.shape
    y = x if perm is None else np.take_along_axis(x, np.broadcast_to(perm[None], x.shape), axis=2)
    b = y.reshape(N, Hh, Dd // block, block)
    ss = np.sum(b * b, axis=-1, dtype=np.float32)
    s = f16(np.sqrt(ss * np.float32(1.0 / block)))[..., None]
    qn = np.where(s > 0, b / np.where(s > 0, s, 1), 0).astype(np.float32)
    out = (to_levels(qn, lv) * s).reshape(N, Hh, Dd)
    if perm is not None:
        inv = np.argsort(perm, axis=1)
        out = np.take_along_axis(out, np.broadcast_to(inv[None], out.shape), axis=2)
    return out.astype(np.float32)


def channel_perm(v_rot: np.ndarray, n_cal: int) -> np.ndarray:
    """Per kv head, dims sorted by descending RMS over the first n_cal cells (loudest first)."""
    rms = np.sqrt(np.mean(v_rot[:n_cal].astype(np.float64) ** 2, axis=0))  # (H, D)
    return np.argsort(-rms, axis=1, kind="stable")


def qt_out(x: np.ndarray, lv: np.ndarray, perm: np.ndarray, n_exact: int, block: int) -> np.ndarray:
    """Dense-and-sparse V: the n_exact loudest (reordered) dims stay FP16; the rest are per-token 2-bit."""
    N, Hh, Dd = x.shape
    y = np.take_along_axis(x, np.broadcast_to(perm[None], x.shape), axis=2)
    ex, rest = y[..., :n_exact], y[..., n_exact:]
    q = qt(rest, lv, None, block=block)
    out = np.concatenate([ex, q], axis=-1)
    inv = np.argsort(perm, axis=1)
    return np.take_along_axis(out, np.broadcast_to(inv[None], out.shape), axis=2).astype(np.float32)


def klt_basis(x: np.ndarray, n_cal: int) -> np.ndarray:
    """Per kv head PCA basis (H, D, D) from the first n_cal cells, columns by descending variance."""
    out = []
    for h in range(x.shape[1]):
        c = x[:n_cal, h].astype(np.float64)
        _, V = np.linalg.eigh(c.T @ c / c.shape[0])
        out.append(V[:, ::-1])
    return np.stack(out).astype(np.float32)


def rot_h(x: np.ndarray, B: np.ndarray, inverse: bool = False) -> np.ndarray:
    M = np.transpose(B, (0, 2, 1)) if inverse else B
    return np.einsum("nhd,hde->nhe", x, M).astype(np.float32)


def block_norm_samples(x: np.ndarray, perm: np.ndarray, block: int = 32) -> np.ndarray:
    """Normalized (x / block RMS) values of the reordered per-token blocks, as the 2-bit codebook sees them."""
    N, Hh, Dd = x.shape
    y = np.take_along_axis(x, np.broadcast_to(perm[None], x.shape), axis=2)
    b = y.reshape(N, Hh, Dd // block, block)
    s = np.sqrt(np.sum(b * b, axis=-1, dtype=np.float64) / block)[..., None]
    u = (b / np.where(s > 0, s, 1)).ravel()
    return u


def lloyd(x: np.ndarray, lv0: np.ndarray, iters: int = 300) -> np.ndarray:
    """Lloyd-Max refit of a level set on a sample (bincount means, fixed iteration cap)."""
    lv = lv0.astype(np.float64).copy()
    L = len(lv)
    for _ in range(iters):
        idx = np.searchsorted((lv[:-1] + lv[1:]) / 2, x)
        cnt = np.bincount(idx, minlength=L)
        sm = np.bincount(idx, weights=x, minlength=L)
        new = np.where(cnt > 0, sm / np.maximum(cnt, 1), lv)
        done = np.max(np.abs(new - lv)) < 1e-7
        lv = new
        if done:
            break
    return np.sort(lv).astype(np.float32)


def qc_var(x: np.ndarray, bits: np.ndarray, lv_by_bits: dict) -> np.ndarray:
    """Per-channel, variable bit width per (kv head, dim). 0 bits reconstructs zero. Scale rule as qc()."""
    out = np.zeros(x.shape, np.float32)
    for b in np.unique(bits):
        if int(b) == 0:
            continue
        out = np.where((bits == b)[None], qc(x, lv_by_bits[int(b)]), out)
    return out.astype(np.float32)


def distortion_table(lv_by_bits: dict) -> np.ndarray:
    """Normalized MSE on N(0,1) per bit width: D[0] = 1 (zero bits), D[b] = MSE of the b-bit level set."""
    x = np.random.default_rng(7).standard_normal(1_000_000).astype(np.float32)
    Dt = [1.0]
    for b in range(1, max(lv_by_bits) + 1):
        Dt.append(float(np.mean((to_levels(x, lv_by_bits[b]) - x) ** 2)))
    return np.array(Dt)


def waterfill(c: np.ndarray, Dt: np.ndarray, avg_bits: float, max_bits: int = 4) -> np.ndarray:
    """Greedy reverse water-filling over integer bits per (kv head, dim). Error model: sum_i c_i * D[b_i].
    Each step gives one bit to the channel with the largest gain c_i * (D[b_i] - D[b_i + 1]); per kv head."""
    import heapq

    H, Dd = c.shape
    total = int(round(avg_bits * Dd))
    bits = np.zeros((H, Dd), np.int64)
    for h in range(H):
        b = np.zeros(Dd, np.int64)
        heap = [(-float(c[h, d]) * float(Dt[0] - Dt[1]), d) for d in range(Dd)]
        heapq.heapify(heap)
        for _ in range(total):
            _, d = heapq.heappop(heap)
            b[d] += 1
            if b[d] < max_bits:
                heapq.heappush(heap, (-float(c[h, d]) * float(Dt[b[d]] - Dt[b[d] + 1]), d))
        bits[h] = b
    return bits


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a.astype(np.float64) - b) ** 2) / np.mean(b.astype(np.float64) ** 2)))


def main() -> int:
    ap = argparse.ArgumentParser(description="K3c/V2t real-query probe")
    ap.add_argument("--prefix", required=True, help="dump prefix, e.g. tmp/kv-realq/code")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--stride", type=int, default=4, help="score every Nth query row (1 = all)")
    ap.add_argument("--cal", type=int, default=2048, help="cells used to fit the V channel reorder")
    ap.add_argument("--window", type=int, default=256, help="FP16 window (live config: 256)")
    ap.add_argument("--json", required=True)
    ap.add_argument("--arms", default="", help="'|'-separated arm names to evaluate (default: all)")
    ap.add_argument("--holdout", action="store_true",
                    help="calibrate query moments and P4 importance on the first half of the dumped queries; score "
                         "only the second half (the dumps hold just the last 256 queries per layer)")
    ap.add_argument("--holdout-swap", action="store_true",
                    help="as --holdout with the halves swapped: calibrate on the second half, score the first")
    a = ap.parse_args()
    t0 = time.time()

    LM8 = fit_lloyd_max(8)
    LM4f = fit_lloyd_max(4)
    LMV = {1: fit_lloyd_max(2), 2: LM4, 3: LM8, 4: fit_lloyd_max(16)}   # per-channel level sets by bit width (P1)
    DT = distortion_table(LMV)
    print("distortion D[b] (normalized Gaussian MSE):", np.round(DT, 4), flush=True)
    C8_path = Path(__file__).resolve().parent.parent / "tmp" / "bitshift_L8_R2.npy"
    C8 = np.load(C8_path) if C8_path.exists() else None   # P2 codebook (trained by tcq_bitshift.py)
    print("bitshift L=8 codebook:", "loaded" if C8 is not None else "MISSING, bitshift arms skipped", flush=True)
    real_K = ROOT_TMP / "bitshift_realK_L8.npy"
    real_V = ROOT_TMP / "bitshift_realV_L8.npy"
    C8K = np.load(real_K) if real_K.exists() else None   # real-data refit (tools/tcq_bitshift_refit.py)
    C8V = np.load(real_V) if real_V.exists() else None
    print("bitshift real-data refit K/V codebooks:", "loaded" if C8K is not None and C8V is not None else "MISSING",
          flush=True)
    print("LM4 engine  :", np.round(LM4, 4))
    print("LM4 refit   :", np.round(LM4f, 4))
    print("LM8 fit     :", np.round(LM8, 4), flush=True)

    ids, n_cells, layers = read_sdkv(Path(a.prefix + ".kv.bin"))
    qrec = read_records(Path(a.prefix + ".q.bin"), q_payload)
    srec = read_records(Path(a.prefix + ".sel.bin"), sel_payload)
    print("cache: %d cells, layers %s (load %.0fs)" % (n_cells, ids, time.time() - t0), flush=True)
    want = ids if a.layers == "all" else [int(x) for x in a.layers.split(",")]

    arm_bytes = {  # per (kv head, cell): K bytes + V bytes
        "q4_0 (K4+V4)": 288, "q2c (K2c+V2c)": 160, "q2k (K2c+V4)": 224,
        "K3c+V4": 112 + 144, "K3c+V2c": 112 + 80, "K3c+V2t natural": 112 + 80,
        "K3c+V2t reord": 112 + 80, "K3c+V2t reord (oracle perm)": 112 + 80,
        "K3c*1.10+V2t reord": 112 + 80, "K2c+V2t reord": 80 + 80,
        "K3c+V2t reord blk16": 112 + 96, "K3c+V2t reord blk64": 112 + 72,
        "KLT K3c+V2c": 112 + 80, "KLT K3c+V2t reord": 112 + 80, "KLT K2c+V4 (q2k-KLT)": 80 + 144,
        "K3c+V2t reord +8fp16ch": 112 + 94, "K3c+V2t reord +16fp16ch": 112 + 108,
        "K3c+V2t reord, V gain 1.05": 112 + 80, "K3c+V2t reord, V gain 1.10": 112 + 80,
        "K3c+V2t reord, V gain 1.15": 112 + 80, "K3c+V2t reord, V LM4 refit": 112 + 80,
        "K3c+V2c, V gain 1.10": 112 + 80,
        "K3c+V2t reord TCQ-4": 112 + 80, "K3c+V2t reord TCQ-8": 112 + 80,
        "K3c+V2t reord TCQ-16": 112 + 80, "K3c+V2t natural TCQ-4": 112 + 80,
        "K3 TCQ-8 3b + V2t reord TCQ-4": 112 + 80, "K3 TCQ-4 3b + V2t reord TCQ-4": 112 + 80,
        "K3 TCQ-8 3b reset16 + V2t reord TCQ-4": 112 + 80,
        "K2c TCQ-4 + V4": 80 + 144, "K2c TCQ-4 reset16 + V4": 80 + 144, "K2c TCQ-4 reset8 + V4": 80 + 144,
        "K3c TCQ-4 3b + V4": 112 + 144, "K3c TCQ-8 3b + V4": 112 + 144,
        "K mixed 4/2 (RMS sel) + V4": 112 + 144, "K mixed 4/2 (query sel) + V4": 112 + 144,
        "K2c+V2tcq": 80 + 80, "K2tc+V2tcq": 80 + 80,
        "K2bs+V4": 80 + 144, "K2bs+V2bs": 80 + 80,
        "P1 raw uni2": 80 + 144, "P1 raw wf": 80 + 144, "P1 QEIG uni2": 80 + 144, "P1 QEIG wf": 80 + 144,
        "P1 WHITE uni2": 80 + 144, "P1 WHITE wf": 80 + 144,
        "P4 recent p10": 80 + 86.4, "P4 recent p20": 80 + 92.8, "P4 random p20": 80 + 92.8, "P4 oracle p20": 80 + 92.8,
        "K2bs x1.05+V4": 80 + 144, "K2bs x1.10+V4": 80 + 144, "K2bs x1.10+V2bs": 80 + 80, "K2tc x1.05+V4": 80 + 144,
        "P5 f10 query-energy": 86.5 + 144, "P5 f15 query-energy": 89.5 + 144, "P5 f20 query-energy": 92.75 + 144,
        "P5 f10 mean-absq": 86.5 + 144, "P5 f20 mean-absq": 92.75 + 144, "P5 f20 random": 92.75 + 144,
        "P1 raw wf+TCQ": 80 + 144, "K2bsR+V4": 80 + 144, "K2bsR+V2bsR": 80 + 80,
        "P5 K f20 + V2tcq": 92.75 + 80, "P5+P4 combined (K f20 + V p20)": 92.75 + 92.8,
    }
    results = {"prefix": a.prefix, "cells": n_cells, "stride": a.stride, "cal": a.cal, "window": a.window,
               "bytes_per_head_cell": arm_bytes, "layers": {}}

    for lid in want:
        qi = ids.index(lid)
        k, v = layers[lid]
        Ph = [h for (h, r) in qrec if h[0] == qi]
        Qa = np.concatenate([r[0] for (h, r) in qrec if h[0] == qi])
        Aa = np.concatenate([r[1] for (h, r) in qrec if h[0] == qi])
        P = np.concatenate([np.arange(h[1], h[1] + h[2]) for h in Ph])
        sel_by_pos = {}
        for (h, s) in srec:
            if h[0] != qi:
                continue
            for t in range(h[2]):
                p = h[1] + t
                sel_by_pos[p] = np.array(s[t, :min(p + 1, TOPK + BLOCK - 1)], dtype=np.int64)
        sels = [sel_by_pos.get(int(p), np.arange(int(p) + 1)) for p in P]
        rows = list(range(0, len(P), a.stride))
        # Query dumps hold only the last 256 positions per layer. Calibration rows (query moments, P4 importance)
        # and scored rows are split on the position order; --holdout scores only the rows calibration did not see.
        calib_mask = np.ones(len(P), bool)
        if a.holdout or a.holdout_swap:
            split = int(P.min()) + len(P) // 2
            calib_mask = (P >= split) if a.holdout_swap else (P < split)
            rows = [r for r in rows if not calib_mask[r]]
        print("L%-2d query rows %d (calibration %d, scored %d)" % (lid, len(P), int(calib_mask.sum()), len(rows)),
              flush=True)
        vrms = float(np.sqrt(np.mean(v.astype(np.float64) ** 2)))

        def attend(Kc, Vc, window=None):
            out = np.empty((len(rows), H_Q, D), np.float32)
            for i, r in enumerate(rows):
                ids_r = sels[r]
                kk = Kc[ids_r]
                vv = Vc[ids_r]
                if window is not None:
                    m = (ids_r < SINKS) | (ids_r > P[r] - window)
                    if m.any():
                        kk = np.where(m[:, None, None], k[ids_r], kk)
                        vv = np.where(m[:, None, None], v[ids_r], vv)
                for g in range(H_KV):
                    hs = slice(g * GQA, (g + 1) * GQA)
                    sc = (kk[:, g, :] @ Qa[r, hs].T) * SCALE
                    sc -= sc.max(axis=0, keepdims=True)
                    w = np.exp(sc)
                    w /= w.sum(axis=0, keepdims=True)
                    out[i, hs] = w.T @ vv[:, g, :]
            return out

        def err(O, Oref):
            return float(np.sqrt(np.mean((O.astype(np.float64) - Oref) ** 2))) / vrms

        # reproduce the engine's attention from fp16 K/V before scoring anything
        Oref = attend(k, v)
        vs_engine = float(np.sqrt(np.mean((Oref - Aa[rows]) ** 2)) / np.sqrt(np.mean(Aa[rows] ** 2)))

        kr, vr = rot(k), rot(v)
        perm_cal = channel_perm(vr, min(a.cal, n_cells))
        perm_all = channel_perm(vr, n_cells)
        K3 = rot(qc(kr, LM8))
        K2 = rot(qc(kr, LM4))
        V2c = rot(qc(vr, LM4))
        V2t_nat = rot(qt(vr, LM4, None))
        V2t_re = rot(qt(vr, LM4, perm_cal))
        V2t_or = rot(qt(vr, LM4, perm_all))
        V2t_re16 = rot(qt(vr, LM4, perm_cal, block=16))
        V2t_re64 = rot(qt(vr, LM4, perm_cal, block=64))
        V4 = rot(q4_0(vr))
        K4 = rot(q4_0(kr))
        # PCA (KLT) basis per kv head instead of the Hadamard
        BK, BV = klt_basis(k, min(a.cal, n_cells)), klt_basis(v, min(a.cal, n_cells))
        K3k = rot_h(qc(rot_h(k, BK), LM8), BK, inverse=True)
        K2k = rot_h(qc(rot_h(k, BK), LM4), BK, inverse=True)
        V2ck = rot_h(qc(rot_h(v, BV), LM4), BV, inverse=True)
        V4k = rot_h(q4_0(rot_h(v, BV)), BV, inverse=True)
        V2tk_re = rot_h(qt(rot_h(v, BV), LM4, channel_perm(rot_h(v, BV), min(a.cal, n_cells))), BV, inverse=True)
        V8 = rot(qt_out(vr, LM4, perm_cal, 8, block=31))
        V16 = rot(qt_out(vr, LM4, perm_cal, 16, block=30))
        # knobs on the 2-bit V codebook: reconstruction gain (= scale clip), and a refit of the 4 levels on real V
        LMv = lloyd(block_norm_samples(vr, perm_cal, 32), LM4)
        V2t_refit = rot(qt(vr, LMv, perm_cal))
        print("L%-2d LM4 refit on real V-blocks: %s" % (lid, np.round(LMv, 4)), flush=True)
        # ---- K-side candidates (V stays q4_0 here, so the arms differ only in K) ----
        K2tc = rot(TQ.qc_tcq(kr, 0.4945, 4, 2))                 # 2-bit TCQ-4 along cells, per channel
        K2tc_r16 = rot(TQ.qc_tcq(kr, 0.4945, 4, 2, seg=16))     # same, trellis restarted every 16 cells
        K2tc_r8 = rot(TQ.qc_tcq(kr, 0.4945, 4, 2, seg=8))       # restarted every 8 cells
        K3tc4 = rot(TQ.qc_tcq(kr, 0.2930, 4, 3))                # 3-bit TCQ-4, 16-point grid
        K3tc8 = rot(TQ.qc_tcq(kr, 0.2935, 8, 3))                # 3-bit TCQ-8
        K3tc8_r16 = rot(TQ.qc_tcq(kr, 0.2935, 8, 3, seg=16))    # 3-bit TCQ-8, trellis restarted every 16 cells
        cal_n = min(a.cal, n_cells)
        var_k = np.mean(kr[:cal_n].astype(np.float64) ** 2, axis=0)            # (H, D) K channel energy (rotated)
        Qr = rot(Qa)
        calr = calib_mask  # all dumped queries (or the first half under --holdout); the old P < cal selected none
        wq = np.stack([np.mean(Qr[calr][:, g * GQA:(g + 1) * GQA, :].astype(np.float64) ** 2, axis=(0, 1))
                       for g in range(H_KV)])                                   # (H, D) query energy per kv head
        LM16 = fit_lloyd_max(16)

        def mixed_k(score):
            """Equal bytes (4 + 2 bits averaged = 3): per kv head the top half of channels by `score` at 16 levels,
            the rest at the engine's 4 levels, both per-channel (32-cell groups)."""
            out = np.empty_like(kr)
            for h in range(H_KV):
                top = np.argsort(-score[h])[: D // 2]
                rest = np.setdiff1d(np.arange(D), top)
                out[:, h, top] = qc(kr[:, h:h + 1, top], LM16)[:, 0, :]
                out[:, h, rest] = qc(kr[:, h:h + 1, rest], LM4)[:, 0, :]
            return rot(out)

        K_mix_rms = mixed_k(var_k)
        K_mix_q = mixed_k(wq * var_k)
        # ---- P1: attention-metric basis + water-filling (V stays q4_0, K budget = 2 bits/dim avg) ----
        cal_n = min(a.cal, n_cells)
        Qcal = Qa[calib_mask]
        assert len(Qcal) > 0, "no calibration queries"
        M_q = np.zeros((H_KV, D, D)); Mi_q = np.zeros((H_KV, D, D)); c_q = np.zeros((H_KV, D))
        M_w = np.zeros((H_KV, D, D)); Mi_w = np.zeros((H_KV, D, D)); c_w = np.zeros((H_KV, D))
        M_r = np.broadcast_to(np.eye(D), (H_KV, D, D)).copy(); Mi_r = M_r.copy(); c_r = np.zeros((H_KV, D))
        for g in range(H_KV):
            qg = Qcal[:, g * GQA:(g + 1) * GQA, :].reshape(-1, D).astype(np.float64)
            Sq = qg.T @ qg / qg.shape[0]                                  # query second moment (score metric)
            kg = k[:cal_n, g, :].astype(np.float64)
            Ck = kg.T @ kg / cal_n                                        # key second moment
            lam, Vq = np.linalg.eigh(Sq)
            o = np.argsort(-lam)
            lam = np.maximum(lam[o], 1e-12)
            Vq = Vq[:, o]
            M_q[g], Mi_q[g] = Vq, Vq.T                                    # QEIG: rotate onto query eigenbasis
            c_q[g] = lam * np.diag(Vq.T @ Ck @ Vq)                       # error weight lam_i * key variance_i
            Tm = (Vq * np.sqrt(lam)) @ Vq.T                               # Sq^(1/2): whitens the score metric
            Tmi = (Vq / np.sqrt(lam)) @ Vq.T
            uh = kg @ Tm
            mu, R = np.linalg.eigh(uh.T @ uh / cal_n)                     # KLT of the whitened keys
            o2 = np.argsort(-mu)
            mu, R = mu[o2], R[:, o2]
            M_w[g], Mi_w[g] = Tm @ R, R.T @ Tmi                           # WHITE: non-orthogonal, error Euclidean
            c_w[g] = np.maximum(mu, 0)
            c_r[g] = np.diag(Sq) * np.diag(Ck)                            # RAW control: diag weights, identity basis

        def p1_arm(M, Mi, c, wf):
            def run():
                z = np.einsum("nhd,hde->nhe", k, M).astype(np.float32)
                bits = waterfill(c, DT, 2.0) if wf else np.full((H_KV, D), 2, np.int64)
                print("     bits histogram 0..4 (per kv head x dims): %s" % np.bincount(bits.ravel(), minlength=5),
                      flush=True)
                zh = qc_var(z, bits, LMV)
                return np.einsum("nhd,hde->nhe", zh, Mi).astype(np.float32), V4
            return run

        chk = float(np.max(np.abs(qc_var(k, np.full((H_KV, D), 2), LMV) - qc(k, LM4))))
        print("     P1 check: qc_var(uniform 2 bits) vs qc(LM4), max|diff| = %.2e" % chk, flush=True)

        # ---- P4: attention-mass importance per (cell, kv head); top fraction gets V4, the rest V2-TCQ ----
        V2tcq = rot(tcq_qt(vr, perm_cal, 32, TCQ_DELTA))

        def importance(row_list):
            imp_sum = np.zeros((n_cells, H_KV)); cnt = np.zeros((n_cells, H_KV))
            for r in row_list:
                ids_r = sels[r]
                kk = k[ids_r]
                for g in range(H_KV):
                    hs = slice(g * GQA, (g + 1) * GQA)
                    sc = (kk[:, g, :] @ Qa[r, hs].T) * SCALE
                    sc -= sc.max(axis=0, keepdims=True)
                    w = np.exp(sc)
                    w /= w.sum(axis=0, keepdims=True)
                    imp_sum[ids_r, g] += w.mean(axis=1)
                    cnt[ids_r, g] += 1
            return imp_sum / np.maximum(cnt, 1)

        rows_recent = [r for r in range(len(P)) if calib_mask[r]]   # prefill-end window (the dumped queries)
        imp_memo = {}

        def imp_of(key):
            if key not in imp_memo:
                imp_memo[key] = importance(rows_recent)
            return imp_memo[key]

        def v_mix(mode, frac):
            def run():
                nsel = int(round(frac * n_cells))
                sel = np.zeros((n_cells, H_KV), bool)
                if mode == "random":
                    rng = np.random.default_rng(0)
                    for g in range(H_KV):
                        sel[rng.choice(n_cells, nsel, replace=False), g] = True
                else:
                    order = np.argsort(-imp_of("recent"), axis=0)
                    for g in range(H_KV):
                        sel[order[:nsel, g], g] = True
                return K2tc, np.where(sel[:, :, None], V4, V2tcq).astype(np.float32)
            return run

        arms = {
            "q4_0 (K4+V4)": (K4, V4),
            "q2c (K2c+V2c)": (K2, V2c),
            "q2k (K2c+V4)": (K2, V4),
            "K3c+V4": (K3, V4),
            "K3c+V2c": (K3, V2c),
            "K3c+V2t natural": (K3, V2t_nat),
            "K3c+V2t reord": (K3, V2t_re),
            "K3c+V2t reord (oracle perm)": (K3, V2t_or),
            "K3c*1.10+V2t reord": (K3 * np.float32(1.10), V2t_re),
            "K2c+V2t reord": (K2, V2t_re),
            "K3c+V2t reord blk16": (K3, V2t_re16),
            "K3c+V2t reord blk64": (K3, V2t_re64),
            "KLT K3c+V2c": (K3k, V2ck),
            "KLT K3c+V2t reord": (K3k, V2tk_re),
            "KLT K2c+V4 (q2k-KLT)": (K2k, V4k),
            "K3c+V2t reord +8fp16ch": (K3, V8),
            "K3c+V2t reord +16fp16ch": (K3, V16),
            "K3c+V2t reord, V gain 1.05": (K3, V2t_re * np.float32(1.05)),
            "K3c+V2t reord, V gain 1.10": (K3, V2t_re * np.float32(1.10)),
            "K3c+V2t reord, V gain 1.15": (K3, V2t_re * np.float32(1.15)),
            "K3c+V2t reord, V LM4 refit": (K3, V2t_refit),
            "K3c+V2c, V gain 1.10": (K3, V2c * np.float32(1.10)),
            "K3c+V2t reord TCQ-4": (K3, rot(tcq_qt(vr, perm_cal, 32, TCQ_DELTA))),
            "K3 TCQ-8 3b + V2t reord TCQ-4": (K3tc8, rot(tcq_qt(vr, perm_cal, 32, TCQ_DELTA))),
            "K3 TCQ-4 3b + V2t reord TCQ-4": (K3tc4, rot(tcq_qt(vr, perm_cal, 32, TCQ_DELTA))),
            "K3 TCQ-8 3b reset16 + V2t reord TCQ-4": (K3tc8_r16, rot(tcq_qt(vr, perm_cal, 32, TCQ_DELTA))),
            "K2c TCQ-4 + V4": (K2tc, V4),
            "K2c TCQ-4 reset16 + V4": (K2tc_r16, V4),
            "K2c TCQ-4 reset8 + V4": (K2tc_r8, V4),
            "K3c TCQ-4 3b + V4": (K3tc4, V4),
            "K3c TCQ-8 3b + V4": (K3tc8, V4),
            "K mixed 4/2 (RMS sel) + V4": (K_mix_rms, V4),
            "K mixed 4/2 (query sel) + V4": (K_mix_q, V4),
            "K3c+V2t reord TCQ-8": (K3, rot(tcq_qt_ms(vr, perm_cal, 32, 0.4935, 8))),
            "K3c+V2t reord TCQ-16": (K3, rot(tcq_qt_ms(vr, perm_cal, 32, 0.4940, 16))),
            "K3c+V2t natural TCQ-4": (K3, rot(tcq_qt_ms(vr, None, 32, 0.4945, 4))),
        }
        lazy = {
            "K2c+V2tcq": (K2, V2tcq),
            "K2tc+V2tcq": (K2tc, V2tcq),
            "P1 raw uni2": p1_arm(M_r, Mi_r, c_r, False),
            "P1 raw wf": p1_arm(M_r, Mi_r, c_r, True),
            "P1 QEIG uni2": p1_arm(M_q, Mi_q, c_q, False),
            "P1 QEIG wf": p1_arm(M_q, Mi_q, c_q, True),
            "P1 WHITE uni2": p1_arm(M_w, Mi_w, c_w, False),
            "P1 WHITE wf": p1_arm(M_w, Mi_w, c_w, True),
            "P4 recent p10": v_mix("recent", 0.10),
            "P4 recent p20": v_mix("recent", 0.20),
            "P4 random p20": v_mix("random", 0.20),
        }
        if C8 is not None:
            lazy["K2bs+V4"] = lambda: (rot(qc_bs(kr, C8, 8, 2)), V4)
            lazy["K2bs+V2bs"] = lambda: (rot(qc_bs(kr, C8, 8, 2)), rot(qt_bs(vr, C8, 8, 2, perm_cal)))
            lazy["K2bs x1.05+V4"] = lambda: (rot(qc_bs(kr, C8, 8, 2)) * np.float32(1.05), V4)
            lazy["K2bs x1.10+V4"] = lambda: (rot(qc_bs(kr, C8, 8, 2)) * np.float32(1.10), V4)
            lazy["K2bs x1.10+V2bs"] = lambda: (rot(qc_bs(kr, C8, 8, 2)) * np.float32(1.10),
                                               rot(qt_bs(vr, C8, 8, 2, perm_cal)))
        absq = np.stack([np.mean(np.abs(Qr[calr][:, g * GQA:(g + 1) * GQA, :].astype(np.float64)), axis=(0, 1))
                         for g in range(H_KV)])                                   # (H, D) mean |q| per kv head
        score_rand = np.random.default_rng(1).random((H_KV, D))

        def p5_arm(score, f):
            """Kitty-style: top fraction f of rotated channels per kv head at 4 bits (scalar), rest at 2-bit TCQ."""
            n_hi = int(round(f * D))

            def run():
                base_r = TQ.qc_tcq(kr, 0.4945, 4, 2)
                hi_r = qc(kr, LMV[4])
                out_r = base_r.copy()
                for h in range(H_KV):
                    top = np.argsort(-score[h], kind="stable")[:n_hi]
                    out_r[:, h, top] = hi_r[:, h, top]
                return rot(out_r), V4
            return run

        def p1_tcq_arm():
            """Water-filled bits in the raw basis, realized per channel: 1 -> LM2, 2 -> TCQ-4, 3 -> TCQ-8 (3-bit),
            4 -> LM16, 0 -> zero. Same 2-bit average budget as the other P1 arms."""
            def run():
                bits = waterfill(c_r, DT, 2.0)[None]
                T2 = TQ.qc_tcq(k, 0.4945, 4, 2)
                T3 = TQ.qc_tcq(k, 0.2935, 8, 3)
                S4 = qc(k, LMV[4])
                S1 = qc(k, LMV[1])
                out = np.zeros(k.shape, np.float32)
                out = np.where(bits == 1, S1, out)
                out = np.where(bits == 2, T2, out)
                out = np.where(bits == 3, T3, out)
                out = np.where(bits == 4, S4, out)
                return out.astype(np.float32), V4
            return run

        lazy["P5 f10 query-energy"] = p5_arm(wq * var_k, 0.10)
        lazy["P5 f15 query-energy"] = p5_arm(wq * var_k, 0.15)
        lazy["P5 f20 query-energy"] = p5_arm(wq * var_k, 0.20)
        lazy["P5 f10 mean-absq"] = p5_arm(absq, 0.10)
        lazy["P5 f20 mean-absq"] = p5_arm(absq, 0.20)
        lazy["P5 f20 random"] = p5_arm(score_rand, 0.20)
        lazy["P1 raw wf+TCQ"] = p1_tcq_arm()
        lazy["P5 K f20 + V2tcq"] = lambda: (p5_arm(wq * var_k, 0.20)()[0], V2tcq)
        lazy["P5+P4 combined (K f20 + V p20)"] = lambda: (p5_arm(wq * var_k, 0.20)()[0], v_mix("recent", 0.20)()[1])
        if C8K is not None and C8V is not None:
            lazy["K2bsR+V4"] = lambda: (rot(qc_bs(kr, C8K, 8, 2)), V4)
            lazy["K2bsR+V2bsR"] = lambda: (rot(qc_bs(kr, C8K, 8, 2)), rot(qt_bs(vr, C8V, 8, 2, perm_cal)))
        lazy["K2tc x1.05+V4"] = (K2tc * np.float32(1.05), V4)
        arms.update(lazy)
        if a.arms:
            keep = set(a.arms.split("|"))
            arms = {n: x for n, x in arms.items() if n in keep}
        r_out = {"layer": lid, "positions": [int(P[0]), int(P[-1])], "rows_scored": len(rows),
                 "vs_engine": vs_engine, "arms": {}}
        for name, spec in arms.items():
            Kc, Vc = spec() if callable(spec) else spec
            Ov = attend(Kc, Vc)
            Ow = attend(Kc, Vc, window=a.window)
            r_out["arms"][name] = {
                "bytes": arm_bytes[name],
                "elem_K": rel(Kc, k), "elem_V": rel(Vc, v),
                "nowin": err(Ov, Oref), "win": err(Ow, Oref),
            }
        results["layers"][str(lid)] = r_out
        print("L%-2d vs engine %.1e | elem K3 %.1f%% K2 %.1f%% V2t-nat %.1f%% V2t-reord %.1f%% V2c %.1f%%"
              % (lid, vs_engine, 100 * rel(K3, k), 100 * rel(K2, k), 100 * rel(V2t_nat, v),
                 100 * rel(V2t_re, v), 100 * rel(V2c, v)), flush=True)
        for name, x in r_out["arms"].items():
            print("     %-30s %4d B  nowin %6.3f%%  win%d %6.3f%%" % (name, x["bytes"], 100 * x["nowin"],
                                                                 a.window, 100 * x["win"]), flush=True)
        Path(a.json).write_text(json.dumps(results, indent=1))
        print("     (%.0fs elapsed)" % (time.time() - t0), flush=True)

    done = list(results["layers"].values())
    print("\nSUMMARY over %d layers (mean per-layer relative attention error, %%):" % len(done))
    print("%-30s %5s %9s %9s %9s %9s" % ("arm", "B/h/c", "nowin", "win", "elemK%", "elemV%"))
    results["summary"] = {}
    for nm in arm_bytes:
        if nm not in done[0]["arms"]:
            continue
        nw = float(np.mean([d["arms"][nm]["nowin"] for d in done]))
        wi = float(np.mean([d["arms"][nm]["win"] for d in done]))
        ek = float(np.mean([d["arms"][nm]["elem_K"] for d in done]))
        ev = float(np.mean([d["arms"][nm]["elem_V"] for d in done]))
        results["summary"][nm] = {"nowin": nw, "win": wi}
        print("%-30s %5d %9.3f %9.3f %9.2f %9.2f" % (nm, arm_bytes[nm], 100 * nw, 100 * wi, 100 * ek, 100 * ev))
    Path(a.json).write_text(json.dumps(results, indent=1))
    print("total %.0fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
