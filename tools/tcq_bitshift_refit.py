#!/usr/bin/env python3
"""Real-data codebook refit for the bitshift trellis (P2).

The Gaussian-trained codebook shrinks real keys (reconstruction energy 0.91 vs 0.96 for TCQ-4). Here the codebook is
refit by GLA, warm-started from the Gaussian one, on real normalized sequences exactly as the codec sees them:
  K: rotated keys, 32 consecutive cells per (head, dim), scale = f16 RMS of the group   (as qc_bs)
  V: rotated values, 32 channel-reordered dims per (cell, head), scale = f16 RMS         (as qt_bs)
Training uses the CODE dump only (layers 3, 7, 11, 15). The check uses the PROSE dump (layers 3, 7), so the
refit is never scored on the data it was trained on.

Check metric: SNR (dB) of the normalized sequences, plus the reconstruction energy ratio (shrinkage).
Compared: scalar LM4, TCQ-4 (tcq_rbits, 2 bits), bitshift with the Gaussian codebook, bitshift with the refit codebook.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from kv_realq_probe import f16, read_sdkv, rot  # noqa: E402
from tcq_bitshift import viterbi, quantize  # noqa: E402
import tcq_rbits as TQ  # noqa: E402

L, R = 8, 2
NS = 1 << L
TRAIN_LAYERS = [3, 7, 11, 15]
EVAL_LAYERS = [3, 7]
MAX_SEQ = 60_000
ROOT = HERE.parent


def k_seqs(kr: np.ndarray, group: int = 32) -> np.ndarray:
    N, H, D = kr.shape
    g = kr.reshape(N // group, group, H, D).transpose(0, 2, 3, 1).reshape(-1, group).astype(np.float64)
    s = f16(np.sqrt(np.mean(g * g, axis=1))).astype(np.float64)
    return np.where(s[:, None] > 0, g / np.where(s > 0, s, 1)[:, None], 0.0)


def v_perm(vr: np.ndarray, n_cal: int = 2048) -> np.ndarray:
    rms = np.sqrt(np.mean(vr[:n_cal].astype(np.float64) ** 2, axis=0))
    return np.argsort(-rms, axis=1, kind="stable")


def v_seqs(vr: np.ndarray, perm: np.ndarray, block: int = 32) -> np.ndarray:
    N, H, D = vr.shape
    y = np.take_along_axis(vr, np.broadcast_to(perm[None], vr.shape), axis=2)
    b = y.reshape(N, H, D // block, block).astype(np.float64)
    s = f16(np.sqrt(np.sum(b * b, axis=-1) / block)).astype(np.float64)[..., None]
    return np.where(s > 0, b / np.where(s > 0, s, 1), 0.0).reshape(-1, block)


def gla(u: np.ndarray, C0: np.ndarray, iters: int) -> np.ndarray:
    C = C0.astype(np.float64).copy()
    for it in range(iters):
        st = viterbi(u, C, L, R).ravel()
        cnt = np.bincount(st, minlength=NS)
        sm = np.bincount(st, weights=u.ravel(), minlength=NS)
        C = np.where(cnt > 0, sm / np.maximum(cnt, 1), C)
        print("   gla iter %d done" % it, flush=True)
    return C


def snr_and_gain(u: np.ndarray, rec: np.ndarray):
    e = np.mean((rec - u) ** 2)
    gain = float(np.sum(rec * u) / np.sum(u * u))
    return float(-10 * np.log10(e)), gain


def load_layers(prefix: str, layers: list[int]):
    ids, _, lay = read_sdkv(Path(prefix + ".kv.bin"))
    out = {}
    for lid in layers:
        k, v = lay[lid]
        kr, vr = rot(k), rot(v)
        out[lid] = (kr, vr)
    return out


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(0)
    C_gauss = np.load(ROOT / "tmp" / "bitshift_L8_R2.npy")

    # ---- training sequences (code dump) ----
    tr = load_layers(str(ROOT / "tmp/kv-realq/code"), TRAIN_LAYERS)
    Ku, Vu = [], []
    for lid, (kr, vr) in tr.items():
        Ku.append(k_seqs(kr))
        Vu.append(v_seqs(vr, v_perm(vr)))
    Ku = np.concatenate(Ku)
    Vu = np.concatenate(Vu)
    Ku = Ku[rng.choice(len(Ku), min(MAX_SEQ, len(Ku)), replace=False)]
    Vu = Vu[rng.choice(len(Vu), min(MAX_SEQ, len(Vu)), replace=False)]
    print("train sequences: K %d, V %d  (load %.0fs)" % (len(Ku), len(Vu), time.time() - t0), flush=True)

    print("refit K codebook:", flush=True)
    C_K = gla(Ku, C_gauss, 6)
    print("refit V codebook:", flush=True)
    C_V = gla(Vu, C_gauss, 6)
    np.save(ROOT / "tmp" / "bitshift_realK_L8.npy", C_K)
    np.save(ROOT / "tmp" / "bitshift_realV_L8.npy", C_V)

    # ---- held-out check on the prose dump ----
    te = load_layers(str(ROOT / "tmp/kv-realq/prose"), EVAL_LAYERS)
    rows = []
    LM4 = np.array([-1.5097, -0.4591, 0.4591, 1.5097])
    for lid, (kr, vr) in te.items():
        for side, u in (("K", k_seqs(kr)), ("V", v_seqs(vr, v_perm(vr)))):
            C_side = C_K if side == "K" else C_V
            cands = {}
            th = (LM4[:-1] + LM4[1:]) / 2
            cands["scalar LM4 (engine levels)"] = LM4[np.searchsorted(th, u)]
            cands["TCQ-4 (tcq_rbits, 2 bits)"] = TQ.quantize(u, 0.4945, 4, 2)[0]
            cands["bitshift Gaussian codebook"] = quantize(u, C_gauss, L, R)[0]
            cands["bitshift refit codebook (code-trained)"] = quantize(u, C_side, L, R)[0]
            for name, rec in cands.items():
                snr, gain = snr_and_gain(u, rec)
                rows.append((lid, side, name, snr, gain))
    print("\nprose held-out, normalized sequences (layers %s):" % EVAL_LAYERS, flush=True)
    for side in ("K", "V"):
        print("  %s" % side, flush=True)
        names = []
        for lid, s, name, snr, gain in rows:
            if s == side and name not in names:
                names.append(name)
        for name in names:
            vals = [(snr, gain) for lid, s, nm, snr, gain in rows if s == side and nm == name]
            print("    %-42s SNR %.2f dB   gain %.4f" % (name, np.mean([v[0] for v in vals]),
                                                     np.mean([v[1] for v in vals])), flush=True)
    print("total %.0fs" % (time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
