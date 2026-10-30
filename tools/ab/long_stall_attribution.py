#!/usr/bin/env python3
"""Attribute long GPU idle gaps (> threshold) in the decode windows to the kernels on either side.

Usage: long_stall_attribution.py TRACE_CSV WINDOWS_JSON [--clock mono|mono_raw] [--min-us 100]
For every gap between merged kernel intervals inside a decode window that is longer than the threshold,
the kernel that ends last before the gap ("before") and the kernel that starts first after it ("after")
are recorded. Prints totals per (before, after) pair.
"""
import csv, json, sys, collections

def short(name):
    s = name.replace("(anonymous namespace)::", "").replace("void ", "")
    if name.startswith("__amd_rocclr"):
        return name.split("(")[0][:48]
    return (s.split("(")[0].split("::")[-1].strip() or s[:48])[:48]


def main():
    trace, wpath = sys.argv[1], sys.argv[2]
    clock = sys.argv[sys.argv.index("--clock") + 1] if "--clock" in sys.argv else "mono"
    min_ns = int(float(sys.argv[sys.argv.index("--min-us") + 1]) * 1000) if "--min-us" in sys.argv else 100_000
    W = json.load(open(wpath))
    wins = [W["decode_a"], W["decode_b"]]
    rows = []
    with open(trace, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
    rows.sort()
    pair_t = collections.Counter()
    pair_n = collections.Counter()
    total = 0
    count = 0
    for w in wins:
        s0, s1 = w[clock]
        sel = [r for r in rows if s0 <= r[0] < s1]
        cur_end, cur_name = None, None
        for s, e, name in sel:
            if cur_end is None:
                cur_end, cur_name = e, name
                continue
            if s > cur_end:
                gap = s - cur_end
                if gap >= min_ns:
                    key = (short(cur_name), short(name))
                    pair_t[key] += gap
                    pair_n[key] += 1
                    total += gap
                    count += 1
                cur_end, cur_name = e, name
            elif e > cur_end:
                cur_end, cur_name = e, name
    print(f"gaps >= {min_ns/1000:.0f} us in decode windows: {count}, total {total/1e6:.1f} ms")
    for (b, a), t in pair_t.most_common(15):
        print(f"  {t/1e6:8.2f} ms  n={pair_n[(b, a)]:4d}  before: {b:40s} after: {a}")

if __name__ == "__main__":
    main()
