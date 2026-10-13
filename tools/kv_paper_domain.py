"""kv_paper_domain.py - the research papers' KV schemes in THEIR domain: un-rotated.

The engine's stored cache is rotated (orthonormal Walsh-Hadamard H_256, scale 1/16, applied before
quantization - include/strata/kernels/kv_q4.hpp). KIVI (2402.02750) and SKVQ (2405.06219) assume the RAW
cache: K quantized per-channel, V per-token, SKVQ reordering channels by RMS. Every number this repo has
published for those schemes was measured on the ROTATED cache - their design domain was never tested.

H is symmetric and orthogonal ((H/16)^2 = I), so applying fwht256 again un-rotates exactly. This tool
un-rotates a dumped layer (SDKV from --dump-kv), validates the round trip, and runs the same quantizer
arms on the raw and the rotated cache with the gate's own attention probe (softmax over all cells,
1/sqrt(D) scores, relative-RMS output error), so the two domains are directly comparable.

Arms (same quantizer code in both domains, only the cache differs):
  kivi      K per-channel 2-bit LM4 + V per-token 2-bit asymmetric   (KIVI's config)
  skvq      K per-channel 2-bit LM4 + V per-token 3-level LM, channels reordered by RMS (SKVQ's config)
  ours      K per-channel 2-bit LM4 + V per-channel 2-bit LM4        (the shipping q2, rotation-designed)
  ours-topq K loudest 1/4 at 16-level LM, rest 4-level + V q2        (the top-q lever)
  q4_0      per-32-block 4-bit uniform (the engine's current format)

Usage: python tools/kv_paper_domain.py tmp/kv-precision/kv890.bin --layers 3,47
"""
import argparse
import struct
import sys

import numpy as np

MAGIC = 0x564B4453
HEADER = 6

# Gaussian Lloyd-Max codebooks in sigma units (the gate's k::q2_lm4_level / lm3 constants).
LM4 = np.array([-1.5104, -0.4517, 0.4517, 1.5104], dtype=np.float64)
LM3 = np.array([-1.2247, 0.0, 1.2247], dtype=np.float64)   # sqrt(4/3)*{-1,0,1}
T4 = np.array([-0.9816, 0.0, 0.9816])                      # the midpoints the codes read through


def walsh_hadamard(n):
    """Orthonormal Walsh-Hadamard of order n (n a power of 2): H/np.sqrt(n), H symmetric."""
    H = np.ones((1, 1))
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def fwht_rows(x, H):
    """Apply H to every row: x is (..., n) with the transform along the last axis."""
    return x @ H.T


class Dump:
    def __init__(self, path):
        with open(path, "rb") as f:
            raw = f.read()
        magic, version, n_layers, n_cells, n_head_kv, head_dim = struct.unpack("<6i", raw[:HEADER * 4])
        if magic != MAGIC:
            raise SystemExit("not an SDKV dump (write it with --dump-kv)")
        self.layer_ids = list(struct.unpack("<%di" % n_layers, raw[HEADER * 4:HEADER * 4 + 4 * n_layers]))
        self.n_cells, self.n_head_kv, self.head_dim = n_cells, n_head_kv, head_dim
        plane = n_cells * n_head_kv * head_dim
        self.body = raw[HEADER * 4 + 4 * n_layers:]
        self.plane = plane

    def layer(self, i):
        off = i * 2 * self.plane * 2
        k = np.frombuffer(self.body, dtype="<u2", count=self.plane, offset=off)
        v = np.frombuffer(self.body, dtype="<u2", count=self.plane, offset=off + self.plane * 2)
        kf = k.view(np.float16).astype(np.float64).reshape(self.n_cells, self.n_head_kv, self.head_dim)
        vf = v.view(np.float16).astype(np.float64).reshape(self.n_cells, self.n_head_kv, self.head_dim)
        return kf, vf


# ------------------------------------------------------------- quantizers (group-of-32 along an axis)
def q_per_channel_lm(x, codebook, thresholds):
    """Per (head, dim): groups of 32 CELLS, scale = RMS of the group, LM codebook."""
    n_cells, n_heads, D = x.shape
    g = x.reshape(n_cells // 32, 32, n_heads, D)
    scale = np.sqrt((g * g).mean(axis=1, keepdims=True))                       # (n/32, 1, H, D)
    idx = (g[..., None] > thresholds * scale[..., None]).sum(-1)
    out = codebook[idx] * scale
    err = out - g
    return out.reshape(x.shape), np.sqrt((err * err).mean()) / np.sqrt((g * g).mean())


def q_per_token_asym(x, levels):
    """Per (cell, head): groups of 32 consecutive DIMS, min/max grid with a zero-point (KIVI's V)."""
    n_cells, n_heads, D = x.shape
    g = x.reshape(n_cells, n_heads, D // 32, 32)
    lo = g.min(axis=-1, keepdims=True)
    hi = g.max(axis=-1, keepdims=True)
    step = (hi - lo) / (levels - 1)
    step = np.where(step > 0, step, 1e-8)
    q = np.clip(np.rint((g - lo) / step), 0, levels - 1)
    out = lo + q * step
    err = out - g
    return out.reshape(x.shape), np.sqrt((err * err).mean()) / np.sqrt((g * g).mean())


def q_per_token_lm(x, codebook, thresholds, perm=None):
    """Per (cell, head) LM codebook over groups of 32 dims; perm reorders channels first (SKVQ)."""
    n_cells, n_heads, D = x.shape
    src = x if perm is None else np.take_along_axis(x, perm[None, :, :], axis=2)
    g = src.reshape(n_cells, n_heads, D // 32, 32)
    scale = np.sqrt((g * g).mean(axis=-1, keepdims=True))
    idx = (g[..., None] > thresholds * scale[..., None]).sum(-1)
    out = codebook[idx] * scale
    out = out.reshape(src.shape)
    if perm is not None:
        inv = np.argsort(perm, axis=1)
        out = np.take_along_axis(out, inv[None, :, :], axis=2)
    err = (out - x)
    return out, np.sqrt((err * err).mean()) / np.sqrt((x * x).mean())


def q_per_channel_topq(x, frac=0.25, loud=16, quiet=4):
    """The top-q lever: loudest frac of channels per head at `loud` levels, rest at `quiet` (LM fit)."""
    n_cells, n_heads, D = x.shape
    rms = np.sqrt((x * x).mean(axis=0))                                        # (H, D)
    out = np.empty_like(x)
    thr_i = int(round((1.0 - frac) * D))
    errs_q, errs_n = 0.0, 0.0
    for h in range(n_heads):
        order = np.argsort(rms[h])
        loud_set = order[thr_i:]
        quiet_set = order[:thr_i]
        for d in range(D):
            col = x[:, h, d].reshape(-1, 32)
            s = np.sqrt((col * col).mean(axis=1, keepdims=True))
            levels = loud if d in set(loud_set.tolist()) else quiet
            cb = _lm_levels(levels, s)                                # (n_groups, L)
            idx = np.argmin(np.abs(col[:, :, None] - cb[:, None, :]), axis=2)
            rec = np.take_along_axis(cb, idx, axis=1)                 # (n_groups, 32) dequantized
            out[:, h, d] = rec.ravel()
            e = rec - col
            errs_q += (e * e).sum()
            errs_n += (col * col).sum()
    return out, np.sqrt(errs_q / errs_n)


_LM_CACHE = {}


def _lm_levels(L, scales):
    """L-level Lloyd-Max on a Gaussian, in sigma units, scaled per group. Cached per L."""
    if L not in _LM_CACHE:
        rng = np.random.default_rng(7)
        x = rng.standard_normal(200_000)
        cb = np.linspace(-1, 1, L)
        for _ in range(30):
            idx = np.argmin(np.abs(x[:, None] - cb[None, :]), axis=1)
            for j in range(L):
                if (idx == j).any():
                    cb[j] = x[idx == j].mean()
        _LM_CACHE[L] = np.sort(cb)
    cb = _LM_CACHE[L]
    return cb[None, :] * scales                                   # (n_groups, L)


def q4_0(x):
    """The engine's current format: per-32 block (dims here), 4-bit uniform, scale = amax/7."""
    n_cells, n_heads, D = x.shape
    g = x.reshape(n_cells, n_heads, D // 32, 32)
    d = np.abs(g).max(axis=-1, keepdims=True) / 7.0
    d = np.where(d > 0, d, 1e-8)
    q = np.clip(np.rint(g / d), -8, 7)
    out = q * d
    err = out - g
    return out.reshape(x.shape), np.sqrt((err * err).mean()) / np.sqrt((g * g).mean())


# ------------------------------------------------------------- the gate's attention probe
def attention_probe(q, K, V):
    """Exactly the gate's probe: softmax(q.K^T/sqrt(D)) . V, one query, all cells; relative-RMS error."""
    D = q.shape[0]
    scores = (K @ q) / np.sqrt(D)
    scores -= scores.max()
    a = np.exp(scores)
    a /= a.sum()
    return a @ V


def out_err(got, ref, scale_of):
    return np.sqrt(((got - ref) ** 2).mean()) / np.sqrt((scale_of ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--layers", default="3,47")
    a = ap.parse_args()
    dump = Dump(a.dump)
    H = walsh_hadamard(dump.head_dim)
    rng = np.random.default_rng(23)
    q_rot = rng.standard_normal(dump.head_dim)

    for want in [int(s) for s in a.layers.split(",") if s.strip()]:
        if want in dump.layer_ids:
            li = dump.layer_ids.index(want)
        else:
            li = want if 0 <= want < dump.n_layers else -1
        if li < 0:
            raise SystemExit("layer %d is not in the dump (ids: %s)" % (want, dump.layer_ids))
        K, V = dump.layer(li)
        n_use = (K.shape[0] // 32) * 32                 # whole 32-cell groups, like the gate's loader
        K, V = K[:n_use], V[:n_use]
        # un-rotate: H is its own inverse
        K_raw = fwht_rows(K.reshape(-1, dump.head_dim), H).reshape(K.shape)
        V_raw = fwht_rows(V.reshape(-1, dump.head_dim), H).reshape(V.shape)
        # validate the round trip: rotating twice is the identity
        rt = fwht_rows(K_raw.reshape(-1, dump.head_dim), H).reshape(K.shape)
        rt_err = np.abs(rt - K).max() / np.abs(K).max()
        q_raw = H @ q_rot

        def channel_spread(x):
            rms = np.sqrt((x * x).mean(axis=0))
            return rms.max() / np.median(rms)

        print("layer %d (id %d): %d cells x %d heads x %d dims; un-rotate round trip %.2e"
              % (li, dump.layer_ids[li], dump.n_cells, dump.n_head_kv, dump.head_dim, rt_err))
        print("  channel RMS max/median: K raw %.2fx -> rotated %.2fx | V raw %.2fx -> rotated %.2fx"
              % (channel_spread(K_raw), channel_spread(K), channel_spread(V_raw), channel_spread(V)))

        def probe_all(q, Kd, Vd):
            """The gate's probe per (cell-row) head slice: one query per head, outputs concatenated."""
            outs = [attention_probe(q, Kd[:, h, :], Vd[:, h, :]) for h in range(Kd.shape[1])]
            return np.concatenate(outs)

        for name, dom, Kd, Vd, q in (("raw (the papers' domain)", "raw", K_raw, V_raw, q_raw),
                                     ("rotated (our engine's)", "rot", K, V, q_rot)):
            n_cells = Kd.shape[0]
            Kf = Kd.reshape(n_cells, -1)
            Vf = Vd.reshape(n_cells, -1)
            out_ref = probe_all(q, Kd, Vd)
            rows = []
            # KIVI: K per-channel 2-bit LM4 + V per-token 2-bit asym
            Kq, e_k = q_per_channel_lm(Kd, LM4, T4)
            Vq, e_v = q_per_token_asym(Vd, 4)
            rows.append(("kivi  K2c + V2t-asym", e_k, e_v, out_err(probe_all(q, Kq, Vq), out_ref, Vf)))
            # SKVQ: K2c + V per-token 3-level LM with channel reorder
            rms_v = np.sqrt((Vd * Vd).mean(axis=0))
            perm = np.argsort(rms_v, axis=1)
            Vq, e_v = q_per_token_lm(Vd, LM3, np.array([-0.6124, 0.6124]), perm=perm)
            rows.append(("skvq  K2c + V1.5t-reord", e_k, e_v, out_err(probe_all(q, Kq, Vq), out_ref, Vf)))
            # ours: q2 per-channel LM4 both
            Kq2, ek2 = q_per_channel_lm(Kd, LM4, T4)
            Vq2, ev2 = q_per_channel_lm(Vd, LM4, T4)
            rows.append(("ours  K2c + V2c", ek2, ev2, out_err(probe_all(q, Kq2, Vq2), out_ref, Vf)))
            # ours top-q: K loudest 1/4 at 16 levels + V q2 per-channel
            Kq3, ek3 = q_per_channel_topq(Kd)
            rows.append(("ours  Ktopq + V2c", ek3, ev2, out_err(probe_all(q, Kq3, Vq2), out_ref, Vf)))
            # q4_0
            Kq4, ek4 = q4_0(Kd)
            Vq4, ev4 = q4_0(Vd)
            rows.append(("q4_0 (4-bit uniform)", ek4, ev4, out_err(probe_all(q, Kq4, Vq4), out_ref, Vf)))
            print("  %s:" % name)
            print("    %-24s %9s %9s %12s" % ("scheme", "K rms", "V rms", "attention"))
            for n, ek, ev, ee in rows:
                print("    %-24s %8.3f%% %8.3f%% %11.3f%%" % (n, 100 * ek, 100 * ev, 100 * ee))


if __name__ == "__main__":
    main()
