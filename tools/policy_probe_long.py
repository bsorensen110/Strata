#!/usr/bin/env python3
"""Eviction-time V-flag policy probe, long-prompt full-row capture (tmp/kv-long/longfull.*), QSA layers 7,15,39,47.

Scope (stated, not hidden): the selection dump covers only the last 4096 prompt positions (STRATA_QSA_DUMP_LAST=4096),
so the policy is measured on the selection region R = [sel_start, n_cells). Cells outside R are V2-TCQ under every
policy, so the policies differ only inside R.

Combined scheme under test (per kv head, per cell):
  K: 2-bit TCQ-4 along cells (per channel, 32-cell groups); top 20% channels by query energy x key variance at 4 bit.
  V: 2-bit TCQ-4 per token (reordered channels); 20% of R's cells per head at V4 (q4_0), rest V2-TCQ.
Policies for WHICH cells get V4 (budget = 20% of R per head):
  evict   : importance from queries at positions c+1..c+256 (the cell's own FP16-window time). Valid online policy.
  prefill : importance from the last 256 prompt queries (the P4 policy).
  oracle  : importance from all selecting queries (in-sample upper bound).
  random  : seeded random 20% (control).
  uniform : V2-TCQ everywhere (no V4 cells).

Scoring (on every 8th selecting query row):
  nowin  : every selected cell read from the quantized cache.
  win256 : sinks (first 8 cells) and the query's last 256 cells read exact (the shipped overlay).

Gate: before any arm is scored, the probe reproduces the engine's attention output (the query dump's outputs) from the
raw fp16 K/V and the dumped selections; it aborts if the relative error is above 1e-3.

Not scored: decode queries (STRATA_QSA_DQDUMP). The decode indexer's selections are not in the dumps.
"""
from __future__ import annotations

import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from kv_realq_probe import D, H_KV, H_Q, q4_0, rot  # noqa: E402
from kv_k3v2_probe import channel_perm, fit_lloyd_max, qc  # noqa: E402
from tcq_v_prototype import DELTA as TCQ_DELTA, tcq_qt  # noqa: E402
import tcq_rbits as TQ  # noqa: E402

GQA = H_Q // H_KV
SCALE = np.float32(1.0 / 16.0)
WIN = 256
SINKS = 8
FRAC = 0.20
CHUNK = 8192            # cells per quantizer call (multiple of the 32-cell K group)
OUT = HERE.parent / "tmp" / "kv-long"
TAG = "longfull"
LAYERS = [7, 15, 39, 47]
SCORE_STRIDE = 8


def read_kv_layers(path: Path, want: list[int]):
    """SDKV v1: header 24 B, layer ids, then per layer fp16 K and V (nc, nh, hd). Reads only the wanted layers."""
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    magic, ver, nl, nc, nh, hd = struct.unpack_from("<6i", mm, 0)
    assert magic == 0x564B4453 and ver == 1, "not an SDKV v1 file"
    ids = list(struct.unpack_from("<%di" % nl, mm, 24))
    per = nc * nh * hd
    off = 24 + 4 * nl
    out = {}
    for lid in ids:
        if lid in want:
            k = np.frombuffer(mm, np.float16, per, off).astype(np.float32).reshape(nc, nh, hd)
            v = np.frombuffer(mm, np.float16, per, off + per * 2).astype(np.float32).reshape(nc, nh, hd)
            out[lid] = (k, v)
        off += per * 4
    return ids, nc, out


def stream_query_layer(path: Path, lid: int):
    """Query dump: records {qsa layer, pos0, rows, zv} + rows*zv queries + rows*zv outputs (float32)."""
    pos_l, Q_l, A_l = [], [], []
    with open(path, "rb") as f:
        while True:
            h = f.read(16)
            if len(h) < 16:
                break
            q_idx, pos0, rows, zv = struct.unpack("<4i", h)
            nb = rows * zv * 4
            if q_idx == lid:
                pos_l.append(np.arange(pos0, pos0 + rows))
                Q_l.append(np.frombuffer(f.read(nb), np.float32).reshape(rows, H_Q, D))
                A_l.append(np.frombuffer(f.read(nb), np.float32).reshape(rows, H_Q, D))
            else:
                f.seek(2 * nb, 1)
    pos = np.concatenate(pos_l)
    order = np.argsort(pos)
    return pos[order], np.concatenate(Q_l)[order], np.concatenate(A_l)[order]


def stream_sel_layer(path: Path, lid: int) -> dict:
    """Selection dump: records {qsa layer, pos0, T, cap} + T*cap int32. Returns {position: selected cells}."""
    sel = {}
    with open(path, "rb") as f:
        while True:
            h = f.read(16)
            if len(h) < 16:
                break
            q_idx, pos0, T, cap = struct.unpack("<4i", h)
            nb = T * cap * 4
            if q_idx == lid:
                s = np.frombuffer(f.read(nb), np.int32).reshape(T, cap)
                for t in range(T):
                    sel[pos0 + t] = s[t].astype(np.int64)
            else:
                f.seek(nb, 1)
    return sel


def attend(Kc, Vc, Q, row_pos, row_sel, win: bool, Kex=None, Vex=None):
    """Engine attention for the given query rows. Kc/Vc: cache for quantized cells; Kex/Vex: exact cache for the
    window overlay (sinks + last WIN cells of each query). Returns (rows, H_Q, D)."""
    out = np.empty((len(row_pos), H_Q, D), np.float32)
    for i, p in enumerate(row_pos):
        ids = row_sel[i]
        kk, vv = Kc[ids], Vc[ids]
        if win:
            m = (ids < SINKS) | (ids > p - WIN)
            if m.any():
                kk = np.where(m[:, None, None], Kex[ids], kk)
                vv = np.where(m[:, None, None], Vex[ids], vv)
        for g in range(H_KV):
            hs = slice(g * GQA, (g + 1) * GQA)
            sc = (kk[:, g, :] @ Q[i, hs].T) * SCALE
            sc -= sc.max(axis=0, keepdims=True)
            w = np.exp(sc)
            w /= w.sum(axis=0, keepdims=True)
            out[i, hs] = w.T @ vv[:, g, :]
    return out


def chunked(fn, X):
    out = np.empty_like(X)
    for i in range(0, len(X), CHUNK):
        out[i:i + CHUNK] = fn(X[i:i + CHUNK])
    return out


def accumulate_importance(rows, pos, Q, sel, K, n_cells, prompt_end):
    """One pass over the selecting queries. Returns (win, end, all): mean attention weight per (cell, kv head).
    win : the query at position p counts for cells c with c < p <= c + WIN (the cell's own window)
    end : the queries in the last WIN prompt positions (the prefill-end policy)
    all : every selecting query (oracle)"""
    shp = (n_cells, H_KV)
    w_win, c_win = np.zeros(shp), np.zeros(shp)
    w_end, c_end = np.zeros(shp), np.zeros(shp)
    w_all, c_all = np.zeros(shp), np.zeros(shp)
    for r in rows:
        p = int(pos[r])
        ids = sel[p]
        ids = ids[ids <= p]
        for g in range(H_KV):
            hs = slice(g * GQA, (g + 1) * GQA)
            sc = (K[ids, g, :] @ Q[r, hs].T) * SCALE
            sc -= sc.max(axis=0, keepdims=True)
            w = np.exp(sc)
            w /= w.sum(axis=0, keepdims=True)
            wm = w.mean(axis=1)
            w_all[ids, g] += wm
            c_all[ids, g] += 1
            if p >= prompt_end - WIN:
                w_end[ids, g] += wm
                c_end[ids, g] += 1
            m = (ids >= p - WIN) & (ids < p)
            if m.any():
                w_win[ids[m], g] += wm[m]
                c_win[ids[m], g] += 1
    div = lambda a, c: a / np.maximum(c, 1)
    return div(w_win, c_win), div(w_end, c_end), div(w_all, c_all)


def v4_mask(score, region, frac):
    """Per kv head: the top `frac` of the region's cells by score get V4. Returns (n_cells, H_KV) bool."""
    n_v4 = int(round(frac * len(region)))
    mask = np.zeros((score.shape[0], H_KV), bool)
    s = score[region]
    for h in range(H_KV):
        top = np.argsort(-s[:, h], kind="stable")[:n_v4]
        mask[region[top], h] = True
    return mask


def main() -> int:
    t0 = time.time()
    lids = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else LAYERS
    _, n_cells, _ = read_kv_layers(OUT / f"{TAG}.kv.bin", [])
    sel_any = stream_sel_layer(OUT / f"{TAG}.sel.bin", lids[0])
    sel_start, prompt_end = min(sel_any), max(sel_any) + 1
    region = np.arange(sel_start, n_cells)
    print("cells %d, selection region %d..%d (%d cells), prompt end %d (%.0fs)"
          % (n_cells, sel_start, n_cells - 1, len(region), prompt_end, time.time() - t0), flush=True)
    del sel_any

    results = {"region": [sel_start, n_cells], "frac": FRAC, "win": WIN, "layers": {}}
    for lid in lids:
        _, _, kv = read_kv_layers(OUT / f"{TAG}.kv.bin", [lid])
        k, v = kv[lid]
        pos, Q, A = stream_query_layer(OUT / f"{TAG}.q.bin", lid)
        sel = stream_sel_layer(OUT / f"{TAG}.sel.bin", lid)
        rows = [r for r in range(len(pos)) if int(pos[r]) in sel]
        score_rows = rows[::SCORE_STRIDE]
        sp = pos[score_rows]
        ssel = [sel[int(p)] for p in sp]
        print("L%d: %d query rows, %d selecting, %d scored (%.0fs)"
              % (lid, len(pos), len(rows), len(score_rows), time.time() - t0), flush=True)

        # ---- engine-match gate: raw fp16 K/V and raw queries reproduce the dumped attention output ----
        Oref = attend(k, v, Q[score_rows], sp, ssel, win=False)
        Aref = A[score_rows]
        gate = float(np.sqrt(np.mean((Oref - Aref) ** 2)) / np.sqrt(np.mean(Aref ** 2)))
        print("L%d engine match: rel %.2e" % (lid, gate), flush=True)
        if gate > 1e-3:
            print("ABORT: engine reproduction failed for L%d" % lid, flush=True)
            return 1
        vrms = float(np.sqrt(np.mean(v.astype(np.float64) ** 2)))

        # ---- importance: one pass over the selecting queries ----
        imp_win, imp_end, imp_all = accumulate_importance(rows, pos, Q, sel, k, n_cells, prompt_end)
        cov = int(np.sum(imp_win[region].sum(axis=1) > 0))
        print("L%d importance pass: %d / %d region cells have a decision window (%.0fs)"
              % (lid, cov, len(region), time.time() - t0), flush=True)

        # ---- quantized K (combined: 2-bit TCQ, top 20% channels at 4-bit) and V (2-bit TCQ, plus V4 choice) ----
        kr, vr = rot(k), rot(v)
        cal = min(2048, n_cells)
        var_k = np.mean(kr[:cal].astype(np.float64) ** 2, axis=0)
        early = pos < prompt_end - 2 * WIN
        Qr = rot(Q[early])
        wq = np.stack([np.mean(Qr[:, g * GQA:(g + 1) * GQA, :].astype(np.float64) ** 2, axis=(0, 1))
                       for g in range(H_KV)])
        score_k = wq * var_k
        LM16 = fit_lloyd_max(16)
        k2 = chunked(lambda x: TQ.qc_tcq(x, 0.4945, 4, 2), kr)
        k4 = chunked(lambda x: qc(x, LM16), kr)
        n_hi = int(round(FRAC * D))
        for h in range(H_KV):
            top = np.argsort(-score_k[h], kind="stable")[:n_hi]
            k2[:, h, top] = k4[:, h, top]
        Kc = rot(k2)
        del k2, k4
        perm = channel_perm(vr, cal)
        V2 = rot(chunked(lambda x: tcq_qt(x, perm, 32, TCQ_DELTA), vr))
        V4 = rot(chunked(q4_0, vr))
        print("L%d quantized K (combined) and V (%.0fs)" % (lid, time.time() - t0), flush=True)

        policies = {
            "uniform V2 (no V4)": None,
            "evict window": imp_win,
            "prefill-end (P4)": imp_end,
            "oracle (all queries)": imp_all,
            "random 20%": np.random.default_rng(0).random((n_cells, H_KV)),
        }
        arm_res = {}
        for name, score in policies.items():
            if score is None:
                Vc = V2
            else:
                m = v4_mask(score, region, FRAC)
                Vc = np.where(m[:, :, None], V4, V2)
            res = {}
            for view, win in (("nowin", False), ("win256", True)):
                O = attend(Kc, Vc, Q[score_rows], sp, ssel, win=win, Kex=k, Vex=v)
                res[view] = float(np.sqrt(np.mean((O - Oref) ** 2)) / vrms)
            arm_res[name] = res
            print("L%d  %-22s nowin %6.2f%%  win256 %6.2f%%"
                  % (lid, name, 100 * res["nowin"], 100 * res["win256"]), flush=True)
        results["layers"][str(lid)] = {"engine_match": gate, "covered_region_cells": cov, "arms": arm_res}
        (OUT / "policy_probe.json").write_text(json.dumps(results, indent=1))
        del Kc, V2, V4, kr, vr, Q, A, k, v

    print("total %.0fs" % (time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
