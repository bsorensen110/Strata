#!/usr/bin/env python3
"""Idle-gap and small-kernel profile for the decode windows of a rocprofv3 kernel trace.

Usage: decode_gap_profile.py TRACE_CSV WINDOWS_JSON [--clock mono|mono_raw]
Idle = wall time inside a decode window not covered by any kernel (merged intervals).
Gaps are bucketed as launch-scale (<20 us), medium (20-100 us) and long (>100 us), following
docs/decode-idle-profile-2026-10-02 in the skill references.
"""
import csv, json, sys, statistics

def main():
    trace, wpath = sys.argv[1], sys.argv[2]
    clock = sys.argv[sys.argv.index("--clock") + 1] if "--clock" in sys.argv else "mono"
    W = json.load(open(wpath))
    wins = [W["decode_a"], W["decode_b"]]
    rows = []
    with open(trace, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
    rows.sort()
    buckets = {"<20us": 0, "20-100us": 0, ">100us": 0}
    counts = {"<20us": 0, "20-100us": 0, ">100us": 0}
    busy_total = wall_total = 0
    durs = []
    small_per_tok = 0
    toks_total = 0
    for w in wins:
        s0, s1 = w[clock]
        sel = [(s, e, n) for s, e, n in rows if s0 <= s < s1]
        wall = s1 - s0
        wall_total += wall
        toks_total += w["completion_tokens"]
        durs += [e - s for s, e, _ in sel]
        small_per_tok += sum(1 for s, e, _ in sel if e - s < 10_000)
        cur_e = None
        busy = 0
        gaps = []
        for s, e, _ in sel:
            if cur_e is None:
                cur_s, cur_e = s, e
                gaps.append(s - s0)                      # lead-in before first kernel
            elif s > cur_e:
                busy += cur_e - cur_s
                gaps.append(s - cur_e)
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        if cur_e is not None:
            busy += cur_e - cur_s
            gaps.append(s1 - cur_e)
        busy_total += busy
        for g in gaps:
            if g <= 0:
                continue
            k = "<20us" if g < 20_000 else ("20-100us" if g <= 100_000 else ">100us")
            buckets[k] += g
            counts[k] += 1
    idle_total = wall_total - busy_total
    print(f"decode windows: {toks_total} tokens, wall {wall_total/1e6:.0f} ms, GPU busy {100*busy_total/wall_total:.1f}%, idle {100*idle_total/wall_total:.1f}%")
    for k in buckets:
        print(f"  idle {k:9s}: {buckets[k]/1e6:8.1f} ms  ({100*buckets[k]/max(idle_total,1):5.1f}% of idle)  gaps={counts[k]}")
    med = statistics.median(durs)
    print(f"kernel durations: median {med/1e3:.1f} us, dispatches <10 us: {100*small_per_tok/len(durs):.1f}% of all ({small_per_tok/toks_total:.0f} per token)")

if __name__ == "__main__":
    main()
