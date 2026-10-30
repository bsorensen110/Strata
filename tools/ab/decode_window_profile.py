#!/usr/bin/env python3
"""Decode-only kernel profile from a rocprofv3 kernel trace + client-side windows.

Usage:
  decode_window_profile.py TRACE_CSV WINDOWS_JSON [--clock mono|mono_raw]

WINDOWS_JSON is written by the driver: {"warmups": [...], "decode_a": {...}, "decode_b": {...}}
where each request has "mono" / "mono_raw" = [t_start_ns, t_end_ns] taken on the client.

Clock alignment is checked, not assumed: for each clock the script reports how many
kernel dispatches fall inside the decode windows versus the idle gaps between them.
The clock whose windows are dense with dispatches is the one the trace uses.
"""
import csv
import json
import re
import sys
import collections

CLASSES = [
    ("runtime copy/fill (rocclr)", lambda n: "__amd_rocclr" in n),
    ("rocBLAS GEMM (Cijk)", lambda n: n.startswith("Cijk_")),
    ("expert-row copy (mapped)", lambda n: "copy_rows_from_mapped" in n),
    ("expert blob fetch", lambda n: "fetch_blobs" in n),
    ("host/device sync + copy", lambda n: any(s in n for s in ("doorbell", "wait_copy", "copy_i32_from_mapped", "copy_from_mapped"))),
    ("activation quantize/interleave", lambda n: any(s in n for s in ("native_quantize", "quantize_q8", "quantize_act", "interleave", "swiglu_quantize", "swiglu_q8_1"))),
    ("MoE expert GEMV", lambda n: any(s in n for s in ("expdec", "native_down", "IlQ", "Q40Block"))),
    ("dense GEMV (mmvq/mmvf/bf16)", lambda n: any(s in n for s in ("mmvq", "mmvf", "gemv", "project_"))),
    ("GDN/QSA/norm/rope", lambda n: any(s in n for s in ("gdn_", "gr_up", "qsa", "rope", "norm"))),
    ("router/gather/combine", lambda n: any(s in n for s in ("route", "gather", "combine", "topk"))),
]


def classify(name):
    for label, pred in CLASSES:
        if pred(name):
            return label
    return "other"


def short_name(name):
    s = name.replace("(anonymous namespace)::", "").replace("void ", "")
    if name.startswith("Cijk_"):
        return "Cijk_* (rocBLAS)"
    return (s.split("(")[0].split("::")[-1].strip() or s[:60])[:60]


def merged_busy_ns(intervals):
    """Union length of [start, end) intervals: GPU-busy time with overlaps counted once."""
    total = 0
    cur_s = cur_e = None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def main():
    trace_path, windows_path = sys.argv[1], sys.argv[2]
    win = json.load(open(windows_path))
    decode = [win["decode_a"], win["decode_b"]]

    rows = []
    with open(trace_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
    rows.sort()

    # ---- clock alignment: pick the client clock whose decode windows hold the dispatches ----
    print(f"trace: {len(rows)} dispatches, span {(rows[-1][1]-rows[0][0])/1e9:.2f} s")
    inside_by_clock = {}
    for ck in ("mono", "mono_raw"):
        inside = 0
        for w in decode:
            s0, s1 = w[ck]
            inside += sum(1 for s, _, _ in rows if s0 <= s < s1)
        span_ns = sum(w[ck][1] - w[ck][0] for w in decode)
        inside_by_clock[ck] = inside
        print(f"  clock {ck:8s}: {inside:6d} dispatch starts inside decode windows "
              f"({inside/(span_ns/1e6):7.1f} per ms of window)")
    clock = max(inside_by_clock.items(), key=lambda kv: kv[1])[0]
    if "--clock" in sys.argv:
        clock = sys.argv[sys.argv.index("--clock") + 1]
    print(f"using clock: {clock}")

    # ---- per-window profile ----
    all_dec = []
    for label, w in zip(("decode_a", "decode_b"), decode):
        s0, s1 = w[clock]
        sel = [(s, e, n) for s, e, n in rows if s0 <= s < s1]
        wall_ns = s1 - s0
        busy = merged_busy_ns([(s, e) for s, e, _ in sel])
        ksum = sum(e - s for s, e, _ in sel)
        toks = w["completion_tokens"]
        print(f"\n[{label}] tokens={toks} wall={wall_ns/1e6:.1f} ms  "
              f"wall/token={wall_ns/1e6/toks:.2f} ms  tok/s(traced)={toks/(wall_ns/1e9):.1f}")
        print(f"  dispatches={len(sel)}  per-token={len(sel)/toks:.0f}  "
              f"kernel-time sum={ksum/1e6:.1f} ms ({ksum/1e6/toks:.2f} ms/token)")
        print(f"  GPU-busy union={busy/1e6:.1f} ms ({busy/1e6/toks:.2f} ms/token) = "
              f"{100*busy/wall_ns:.1f}% of wall; idle={100*(wall_ns-busy)/wall_ns:.1f}%")
        all_dec.extend(sel)

    # ---- family breakdown over both decode windows ----
    fam_t = collections.Counter()
    fam_n = collections.Counter()
    name_t = collections.Counter()
    name_fam = {}
    for s, e, n in all_dec:
        fam = classify(n)
        fam_t[fam] += e - s
        fam_n[fam] += 1
        sn = short_name(n)
        name_t[sn] += e - s
        name_fam[sn] = fam
    total = sum(fam_t.values())
    toks = sum(w["completion_tokens"] for w in decode)
    print(f"\nFamily breakdown, both decode windows ({toks} tokens, kernel-time sum {total/1e6:.1f} ms)")
    for fam, t in fam_t.most_common():
        print(f"  {100*t/total:5.1f}%  {t/1e6:8.2f} ms  {t/1e6/toks:6.3f} ms/token  n={fam_n[fam]:6d}  {fam}")
    print("\nTop kernels by time, both decode windows")
    for nm, t in name_t.most_common(15):
        print(f"  {100*t/total:5.1f}%  {t/1e6/toks:6.3f} ms/token  {nm}  [{name_fam[nm]}]")


if __name__ == "__main__":
    main()
