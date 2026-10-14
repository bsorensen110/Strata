"""Real-query KV quantization probe: attention error with the MODEL'S OWN queries, causal, over the engine's own
QSA selections - the measurement the per-layer clip factor, per-layer precision and salient-cell work calibrate on.

Inputs (one prompt), written by the engine with `--kv fp16`:
    --kv   NAME.kv.bin   `--dump-kv`            SDKV v1: fp16 K/V of every QSA layer, (cell, head, dim)
    --q    NAME.q.bin    STRATA_QSA_QDUMP       per QSA layer and chunk: int32 {qsa layer, first pos, rows, ZV}
                                                + rows*ZV f32 queries + rows*ZV f32 attention outputs (pre-gate)
    --sel  NAME.sel.bin  STRATA_QSA_DUMP        per QSA layer and chunk: int32 {qsa layer, pos0, T, cap}
                                                + T*cap int32 selected cells (the first `width` are valid)

The probe first REPRODUCES the engine's attention output from the dumped fp16 K/V, Q and selections (that pins
the softmax scale, the GQA head mapping and the causal/selection mask), then quantizes K/V with the engine's
codebooks (q4_0 = ggml rule after the orthonormal 256-point Hadamard; q2 = per-channel 32-cell groups, 4-level
Lloyd-Max levels x the group RMS, one-shot append rules incl. the fresh-partial shrink) and reports
    e = rms(O_arm - O_ref) / rms(V)     over every recorded (position, head, dim)
for each arm. Decode-time view: sinks (first 8 cells) and the window (cells > p - 128) can be held FP16.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

H_Q, H_KV, D = 24, 2, 256
TOPK, BLOCK = 2048, 4
SINKS, WINDOW = 8, 128
LM4 = np.array([-1.5097, -0.4591, 0.4591, 1.5097], dtype=np.float32)
LM4_T = np.float32(0.9844)


def hadamard(n: int) -> np.ndarray:
    h = np.array([[1.0]], dtype=np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return (h / np.sqrt(n)).astype(np.float32)


HN = hadamard(D)


def f16(x):
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)


# ------------------------------------------------------------------ readers
def read_sdkv(path: Path):
    b = path.read_bytes()
    magic, ver, nl, nc, nh, hd = struct.unpack_from("<6i", b, 0)
    assert magic == 0x564B4453 and ver == 1, "not an SDKV v1 file"
    ids = list(struct.unpack_from("<%di" % nl, b, 24))
    off, per = 24 + 4 * nl, nc * nh * hd
    layers = {}
    for lid in ids:
        k = np.frombuffer(b, np.float16, per, off).astype(np.float32).reshape(nc, nh, hd); off += per * 2
        v = np.frombuffer(b, np.float16, per, off).astype(np.float32).reshape(nc, nh, hd); off += per * 2
        layers[lid] = (k, v)
    return ids, nc, layers


def read_records(path: Path, payload):
    b, off, out = path.read_bytes(), 0, []
    while off < len(b):
        hdr = struct.unpack_from("<4i", b, off); off += 16
        rec, off = payload(b, off, hdr)
        out.append((hdr, rec))
    return out


def q_payload(b, off, hdr):
    rows, zv = hdr[2], hdr[3]
    q = np.frombuffer(b, np.float32, rows * zv, off).reshape(rows, H_Q, D); off += rows * zv * 4
    a = np.frombuffer(b, np.float32, rows * zv, off).reshape(rows, H_Q, D); off += rows * zv * 4
    return (q, a), off


def sel_payload(b, off, hdr):
    T, cap = hdr[2], hdr[3]
    s = np.frombuffer(b, np.int32, T * cap, off).reshape(T, cap); off += T * cap * 4
    return s, off


# ------------------------------------------------------------------ quantizers (engine codebooks)
def rot(x):           # (..., D) orthonormal Hadamard, self-inverse
    return x @ HN


def q4_0(x):          # x (N, H, D) rotated; ggml q4_0 over 32-value blocks along D
    blk = x.reshape(-1, 32)
    ia = np.argmax(np.abs(blk) + (blk > 0) * 1e-30, axis=1)          # largest |x|, positive on a tie
    mval = blk[np.arange(blk.shape[0]), ia]
    d = (mval / np.float32(-8.0)).astype(np.float32)
    idv = np.where(d != 0, np.float32(1.0) / np.where(d != 0, d, 1), 0).astype(np.float32)
    q = np.trunc(blk * idv[:, None] + np.float32(8.5))
    q = np.clip(q, 0, 15)
    return ((q - 8) * f16(d)[:, None]).reshape(x.shape).astype(np.float32)


def q2c(x, alpha: float = 1.0):
    """Per-channel 2-bit: 32-cell groups per (head, dim), stored scale f16(alpha * rule), one-shot append (every
    group complete except a possible last one, which is fresh-partial with group g-1 complete in the same call:
    s = sqrt((ss + (32 - n) * prev_raw^2) / 32), prev_raw = f16(alpha * sqrt(ss_prev / 32)) / alpha)."""
    n_cells = x.shape[0]
    out = np.empty_like(x)
    a = np.float32(alpha)
    prev_stored = None
    for g0 in range(0, n_cells, 32):
        blk = x[g0:g0 + 32]                                               # (n, H, D)
        n = blk.shape[0]
        ss = np.sum(blk.astype(np.float32) ** 2, axis=0, dtype=np.float32)
        if n == 32 or prev_stored is None:
            s_rule = np.sqrt(ss * np.float32(1.0 / n))
        else:
            prev_raw = prev_stored * (np.float32(1.0) / a)
            s_rule = np.sqrt((ss + np.float32(32 - n) * prev_raw * prev_raw) * np.float32(1.0 / 32))
        s = f16(a * s_rule)
        qn = np.where(s > 0, blk * (np.float32(1.0) / np.where(s > 0, s, 1)), 0)
        code = np.where(np.abs(qn) < LM4_T, np.where(qn < 0, 1, 2), np.where(qn < 0, 0, 3))
        out[g0:g0 + n] = LM4[code] * s
        prev_stored = s
    return out


# ------------------------------------------------------------------ attention
class Layer:
    def __init__(self, k, v, Q, A, P, sels, scale, gmap):
        self.k, self.v, self.Q, self.A, self.P, self.sels = k, v, Q, A, P, sels
        self.scale, self.gmap = scale, gmap
        self.vrms = float(np.sqrt(np.mean(v.astype(np.float64) ** 2)))

    def attend(self, K, V, rows=None, keep=None, weights_out=None):
        """O (len(rows), H_Q, D). keep: optional (N,) bool - cells read at FP16 (the reference K/V) instead of K/V."""
        rows = range(len(self.P)) if rows is None else rows
        out = []
        for r in rows:
            ids = self.sels[r]
            kk, vv = K[ids], V[ids]
            if keep is not None:
                m = keep[ids]
                if m.any():
                    kk = kk.copy(); vv = vv.copy()
                    kk[m] = self.k[ids[m]]; vv[m] = self.v[ids[m]]
            o = np.empty((H_Q, D), np.float32)
            for h in range(H_Q):
                g = self.gmap(h)
                sc = (kk[:, g, :] @ self.Q[r, h]) * self.scale
                sc -= sc.max()
                w = np.exp(sc); w /= w.sum()
                o[h] = w @ vv[:, g, :]
                if weights_out is not None:
                    np.add.at(weights_out, ids, w)
            out.append(o)
        return np.stack(out)

    def err(self, O, Oref):
        return float(np.sqrt(np.mean((O.astype(np.float64) - Oref) ** 2))) / self.vrms


def window_keep(n_cells, p, sinks=SINKS, window=WINDOW):
    keep = np.zeros(n_cells, bool)
    keep[:min(sinks, n_cells)] = True
    keep[max(0, p - window + 1):p + 1] = True
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kv", required=True)
    ap.add_argument("--q", required=True)
    ap.add_argument("--sel", required=True)
    ap.add_argument("--layers", default="all", help="comma list of model layer ids, or all")
    ap.add_argument("--alphas", default="0.70,0.75,0.80,0.85,0.90,0.95,1.00,1.05,1.10,1.20")
    ap.add_argument("--salient", default="64,256,1024")
    ap.add_argument("--json", default="")
    ap.add_argument("--sections", default="alpha,salient,windows",
                    help="which arm groups to run besides the baselines: alpha, salient, windows")
    ap.add_argument("--windows", default="256,384,512,1152", help="FP16 window sizes for the windows section")
    a = ap.parse_args()

    ids, n_cells, layers = read_sdkv(Path(a.kv))
    qrec = read_records(Path(a.q), q_payload)
    srec = read_records(Path(a.sel), sel_payload)
    want = ids if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    alphas = [float(x) for x in a.alphas.split(",")]
    salient = [int(x) for x in a.salient.split(",")]
    print("cache: %d cells, layers %s" % (n_cells, ids))
    results = {}
    for lid in want:
        qi = ids.index(lid)                                   # QSA ordinal = position in the dump's layer list
        k, v = layers[lid]
        Q = [r for (h, r) in qrec if h[0] == qi]
        Ph = [h for (h, r) in qrec if h[0] == qi]
        P = np.concatenate([np.arange(h[1], h[1] + h[2]) for h in Ph])
        Qa = np.concatenate([r[0] for r in Q]); Aa = np.concatenate([r[1] for r in Q])
        sel_by_pos = {}
        for (h, s) in srec:
            if h[0] != qi:
                continue
            for t in range(h[2]):
                p = h[1] + t
                width = min(p + 1, TOPK + BLOCK - 1)
                sel_by_pos[p] = np.array(s[t, :width], dtype=np.int64)
        sels = []
        for p in P:
            ids_p = sel_by_pos.get(int(p))
            if ids_p is None:
                ids_p = np.arange(int(p) + 1)                     # no selection record: dense causal
            sels.append(ids_p)
        bad = sum(1 for p, s in zip(P, sels) if (s.max() > p) or np.any(np.diff(s) <= 0))
        # pin scale + GQA mapping against the engine's own output
        best = None
        for scale in (1.0 / 16.0, 1.0):
            for name, gm in (("h//12", lambda h: h // 12), ("h%2", lambda h: h % 2)):
                L = Layer(k, v, Qa, Aa, P, sels, np.float32(scale), gm)
                rows = range(0, len(P), max(1, len(P) // 16))
                O = L.attend(k, v, rows=rows)
                e = float(np.sqrt(np.mean((O - Aa[list(rows)]) ** 2)) / np.sqrt(np.mean(Aa[list(rows)] ** 2)))
                if best is None or e < best[0]:
                    best = (e, scale, name, gm)
        e_eng, scale, gname, gm = best
        L = Layer(k, v, Qa, Aa, P, sels, np.float32(scale), gm)
        Oref = L.attend(k, v)
        e_vs_engine = float(np.sqrt(np.mean((Oref - Aa) ** 2)) / np.sqrt(np.mean(Aa ** 2)))
        kr, vr = rot(k), rot(v)
        K4, V4 = rot(q4_0(kr)), rot(q4_0(vr))
        r = {"positions": [int(P[0]), int(P[-1])], "sel_width_last": int(len(sels[-1])),
             "bad_selection_rows": bad, "vs_engine": e_vs_engine, "scale": scale, "gqa": gname}
        # per-element (cross-check against the C++ gate)
        Kc1 = rot(q2c(kr, 1.0))
        r["elem_k_q2c"] = float(np.sqrt(np.mean((Kc1 - k) ** 2) / np.mean(k ** 2)))
        r["elem_v_q4"] = float(np.sqrt(np.mean((V4 - v) ** 2) / np.mean(v ** 2)))
        r["q4_0"] = L.err(L.attend(K4, V4), Oref)
        r["q2k"] = L.err(L.attend(Kc1, V4), Oref)
        # decode view: sinks + window FP16
        def win_err(K, V, extra=None, rows=None, window=WINDOW):
            rows = range(len(P)) if rows is None else rows
            O = []
            for rr in rows:
                keep = window_keep(n_cells, int(P[rr]), window=window)
                if extra is not None:
                    keep |= extra
                O.append(L.attend(K, V, rows=[rr], keep=keep)[0])
            return L.err(np.stack(O), Oref[list(rows)])
        r["q4_0+win"] = win_err(K4, V4)
        r["q2k+win"] = win_err(Kc1, V4)
        # clip sweep (K only)
        r["alpha"], r["alpha_win"] = {}, {}
        for al in (alphas if "alpha" in a.sections else []):
            Ka = rot(q2c(kr, al))
            r["alpha"]["%.2f" % al] = L.err(L.attend(Ka, V4), Oref)
            r["alpha_win"]["%.2f" % al] = win_err(Ka, V4)
        # salient cells: oracle (all recorded queries' attention mass) and causal (first half -> second half)
        mass = np.zeros(n_cells, np.float64)
        L.attend(k, v, weights_out=mass)
        excl = np.zeros(n_cells, bool); excl[:SINKS] = True
        half = len(P) // 2
        mass1 = np.zeros(n_cells, np.float64)
        L.attend(k, v, rows=range(half), weights_out=mass1)
        cnt1 = np.zeros(n_cells, np.float64)
        for rr in range(half):
            cnt1[sels[rr]] += 1
        r["salient"] = {}
        second = range(half, len(P))
        r["q2k+win(2nd half)"] = win_err(Kc1, V4, rows=second)
        tot = mass[~excl].sum()
        r["windows"] = {}
        for W in ([int(x) for x in a.windows.split(",")] if "windows" in a.sections else []):
            r["windows"][str(W)] = {"all": win_err(Kc1, V4, window=W), "2nd": win_err(Kc1, V4, rows=second, window=W)}
        for M in (salient if "salient" in a.sections else []):
            def top(score):
                sc = score.copy(); sc[excl] = -np.inf
                order = np.argsort(-sc)[:min(M, int((~excl).sum()))]
                keep = np.zeros(n_cells, bool); keep[order] = True
                return keep
            ko = top(mass)
            r["salient"][str(M)] = {
                "oracle": win_err(Kc1, V4, extra=ko),
                "causal_mass": win_err(Kc1, V4, extra=top(mass1), rows=second),
                "causal_count": win_err(Kc1, V4, extra=top(cnt1), rows=second),
                "mass_share": float(mass[ko].sum() / tot) if tot > 0 else 0.0,
            }
        results[lid] = r
        if not r["alpha"]:
            r["alpha"] = {"1.00": r["q2k"]}; r["alpha_win"] = {"1.00": r["q2k+win"]}
        best_a = min(r["alpha"], key=lambda x: r["alpha"][x])
        best_w = min(r["alpha_win"], key=lambda x: r["alpha_win"][x])
        print("L%-2d pos %d-%d width %d | vs engine %.2e (scale %s, %s, bad rows %d) | elem K %.1f%% V %.1f%%"
              % (lid, P[0], P[-1], len(sels[-1]), e_vs_engine, scale, gname, bad, 100 * r["elem_k_q2c"],
                 100 * r["elem_v_q4"]))
        print("     q4_0 %.3f%%  q2k %.3f%% (%.2fx) | +win: q4_0 %.3f%%  q2k %.3f%% | best alpha %s: %.3f%% (%+.1f%%)"
              % (100 * r["q4_0"], 100 * r["q2k"], r["q2k"] / r["q4_0"], 100 * r["q4_0+win"], 100 * r["q2k+win"],
                 best_a, 100 * r["alpha"][best_a], 100 * (r["alpha"][best_a] / r["alpha"]["1.00"] - 1)))
        print("     alpha sweep (+win): " + " ".join("%s:%.3f" % (k_, 100 * v_) for k_, v_ in r["alpha_win"].items())
              + " -> best %s (%+.1f%%)" % (best_w, 100 * (r["alpha_win"][best_w] / r["alpha_win"]["1.00"] - 1)))
        print("     salient (q2k+win, 2nd-half base %.3f%%): " % (100 * r["q2k+win(2nd half)"]) + "  ".join(
            "M=%s oracle %.3f%% (mass %.0f%%) causal-mass %.3f%% causal-count %.3f%%"
            % (m, 100 * x["oracle"], 100 * x["mass_share"], 100 * x["causal_mass"], 100 * x["causal_count"])
            for m, x in r["salient"].items()), flush=True)
        if r["windows"]:
            print("     windows (q2k, sinks 8): " + "  ".join("W=%s %.3f%% (2nd %.3f%%)" % (w, 100 * x["all"], 100 * x["2nd"])
                                                       for w, x in r["windows"].items()), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    sys.exit(main())
