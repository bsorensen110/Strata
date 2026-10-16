#!/usr/bin/env python3
"""Per-(cell, kv head) V format flag: layout reference + round-trip gate (planning item P4, engine-facing).

Each (cell, head) holds one V vector (256 dims) in exactly one of two record kinds:
  V4  record : q4_0, 8 blocks x 18 B = 144 B   (flag = 1)
  V2  record : TCQ-2bit, 8 blocks x 10 B =  80 B (flag = 0)
A fixed-stride cell cannot hold both kinds, so the layout uses one pool per kv head per kind:

  bitmap[h]      : 1 bit per cell, packed (np.packbits), flag of (cell, h)              0.125 B / (cell, h)
  base4[blk, h]  : uint32, number of V4 cells of head h before block blk (blk = 32 cells) 0.125 B / (cell, h)
  pool4[h]       : V4 records of head h in cell order                                    144 B per flagged cell
  pool2[h]       : V2 records of head h in cell order                                     80 B per other cell

Address of (c, h):  blk = c // 32, w = c % 32, n4 = V4 cells of head h in [blk*32, c)
  flag = 1 : pool4[h][base4[blk,h] + n4]
  flag = 0 : pool2[h][(blk*32 - base4[blk,h]) + (w - n4)]
Overhead = 0.25 B per (cell, head); no per-cell pointer. Flags are fixed at write time, so the reader needs only
the bitmap and base4 for one 32-cell block.

This file is the host reference and the gate. Record bytes here are random: the codecs are gated elsewhere
(tcq_v_host_parity, kv_q2 parity); this gate covers addressing, pool packing, and overhead.
"""
from __future__ import annotations

import sys

import numpy as np

BLK = 32
R4 = 144  # bytes per V4 record (q4_0, 256 dims)
R2 = 80   # bytes per V2 record (TCQ 2-bit, 256 dims)


def encode(flags: np.ndarray, rec4: np.ndarray, rec2: np.ndarray) -> dict:
    """flags (N, H) bool; rec4 (N, H, R4) uint8; rec2 (N, H, R2) uint8."""
    N, H = flags.shape
    assert N % BLK == 0 and N % 8 == 0
    f = flags.astype(np.int64)
    fb = f.reshape(N // BLK, BLK, H)
    blk_count = fb.sum(axis=1)                                   # (nb, H) V4 cells per block
    base4 = np.cumsum(blk_count, axis=0) - blk_count             # V4 cells before each block
    return {
        "N": N, "H": H,
        "bitmap": np.packbits(flags.T.astype(np.uint8), axis=1),  # (H, N // 8)
        "base4": base4.astype(np.uint32),
        "pool4": [rec4[flags[:, h], h] for h in range(H)],
        "pool2": [rec2[~flags[:, h], h] for h in range(H)],
    }


def flags_of(enc: dict) -> np.ndarray:
    return np.stack([np.unpackbits(enc["bitmap"][h])[: enc["N"]] for h in range(enc["H"])], axis=1).astype(bool)


def decode_one(enc: dict, c: int, h: int) -> np.ndarray:
    """O(1) address: one bitmap read, one base read, one in-block popcount (<= 31 bits)."""
    blk, w = c // BLK, c % BLK
    fh = np.unpackbits(enc["bitmap"][h])
    n4 = int(fh[blk * BLK: c].sum())
    if fh[c]:
        return enc["pool4"][h][int(enc["base4"][blk, h]) + n4]
    return enc["pool2"][h][(blk * BLK - int(enc["base4"][blk, h])) + (w - n4)]


def decode_all(enc: dict, H: int | None = None) -> tuple[list, list]:
    """Vectorized reference of the whole cache: returns per head the (N, R) records in cell order."""
    N, H_ = enc["N"], enc["H"]
    fl = flags_of(enc)                                           # (N, H)
    out4, out2 = [], []
    for h in range(H_):
        fb = fl[:, h].reshape(N // BLK, BLK).astype(np.int64)
        n4_in_blk = np.cumsum(fb, axis=1) - fb                   # V4 before c within block
        base = enc["base4"][:, h].astype(np.int64)[:, None]
        slot4 = (base + n4_in_blk).reshape(-1)
        w = np.tile(np.arange(BLK), N // BLK)
        slot2 = (np.arange(N // BLK).repeat(BLK) * BLK - base.repeat(BLK, axis=1).reshape(-1)) + (w - n4_in_blk.reshape(-1))
        rec = np.empty((N, R4), np.uint8)
        f1 = fl[:, h]
        rec[f1] = enc["pool4"][h][slot4[f1]]
        rec2 = np.empty((N, R2), np.uint8)
        rec2[~f1] = enc["pool2"][h][slot2[~f1]]
        out4.append(rec)
        out2.append(rec2)
    return out4, out2


def footprint(enc: dict) -> dict:
    """Bytes per kind, and overhead per (cell, head) of the flag metadata."""
    N, H = enc["N"], enc["H"]
    p4 = sum(p.size for p in enc["pool4"])
    p2 = sum(p.size for p in enc["pool2"])
    meta = enc["bitmap"].size + enc["base4"].size * 4
    return {"pool4_bytes": p4, "pool2_bytes": p2, "meta_bytes": meta,
            "meta_per_cell_head": meta / (N * H), "payload_per_cell_head": (p4 + p2) / (N * H)}


def _gate(verbose: bool = True) -> bool:
    rng = np.random.default_rng(42)
    N, H = 512, 2
    ok = True
    for density in (0.0, 0.1, 0.2, 0.5, 1.0):
        flags = rng.random((N, H)) < density
        rec4 = rng.integers(0, 256, size=(N, H, R4), dtype=np.uint8)
        rec2 = rng.integers(0, 256, size=(N, H, R2), dtype=np.uint8)
        enc = encode(flags, rec4, rec2)
        all4, all2 = decode_all(enc)
        # exact expectation per (cell, head)
        bad = 0
        for h in range(H):
            for c in range(N):
                want = rec4[c, h] if flags[c, h] else rec2[c, h]
                got = decode_one(enc, c, h)
                if not np.array_equal(got, want):
                    bad += 1
                if flags[c, h]:
                    if not np.array_equal(all4[h][c], rec4[c, h]):
                        bad += 1
                elif not np.array_equal(all2[h][c], rec2[c, h]):
                    bad += 1
        fp = footprint(enc)
        n4 = int(flags.sum())
        pass_ = bad == 0 and fp["pool4_bytes"] == n4 * R4 and fp["pool2_bytes"] == (N * H - n4) * R2
        ok &= pass_
        if verbose:
            print("density %.1f: V4 cells %6d / %d  mismatches %d  meta %.4f B/(cell,head)  payload %.1f B/(cell,head)  %s"
                  % (density, n4, N * H, bad, fp["meta_per_cell_head"], fp["payload_per_cell_head"],
                     "PASS" if pass_ else "FAIL"), flush=True)
    return ok


def _mutation_check() -> bool:
    """The gate must fail on a corrupted base table. Returns True if the corruption was detected."""
    rng = np.random.default_rng(7)
    N, H = 256, 2
    flags = rng.random((N, H)) < 0.2
    rec4 = rng.integers(0, 256, size=(N, H, R4), dtype=np.uint8)
    rec2 = rng.integers(0, 256, size=(N, H, R2), dtype=np.uint8)
    enc = encode(flags, rec4, rec2)
    enc["base4"] = enc["base4"].copy()
    enc["base4"][1, 0] += 1                                       # off-by-one in one block base
    bad = 0
    for h in range(H):
        for c in range(N):
            want = rec4[c, h] if flags[c, h] else rec2[c, h]
            if not np.array_equal(decode_one(enc, c, h), want):
                bad += 1
    detected = bad > 0
    print("mutation (base4 off by one): %d mismatches -> %s" % (bad, "DETECTED" if detected else "MISSED"), flush=True)
    return detected


if __name__ == "__main__":
    ok = _gate()
    mut = _mutation_check()
    print("GATE", "PASS" if ok and mut else "FAIL", flush=True)
    sys.exit(0 if ok and mut else 1)
