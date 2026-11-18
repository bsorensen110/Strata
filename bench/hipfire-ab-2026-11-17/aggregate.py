#!/usr/bin/env python3
"""Aggregate hipfire-ab-2026-11-17 results: medians per pair, A/A floor, verdicts. (schema-fixed)"""
import json, statistics, os

ROOT = os.path.dirname(os.path.abspath(__file__))

def load(sid, role, rnd):
    name = f'{role}_{sid}' + ('' if rnd == 1 else f'_r{rnd}')
    p = f'{ROOT}/results/{name}.json'
    return json.load(open(p)) if os.path.exists(p) else None

def metric(d, m):
    if not d: return None
    if m == 'engine_cold':
        e = d.get('engine_decode_512') or []
        return float(e[0]) if e else None
    if m == 'engine_warm':
        e = d.get('engine_decode_512') or []
        return float(e[1]) if len(e) > 1 else None
    if m == 'burst':
        return (d.get('burst') or {}).get('aggregate_tok_s')
    if m == 'prefill':
        return (d.get('prefill') or {}).get('wall_tok_s')

PAIRS = ['baseline', 'bufload', 'pm4', 'iu8', 'iu4', 'x4', 'aa_floor']
METRICS = ['engine_cold', 'engine_warm', 'burst', 'prefill']

def slots(d):
    s = (d or {}).get('expert_cache') or []
    import re
    for line in s:
        m = re.search(r'expert cache (\d+) slots', line)
        if m: return int(m.group(1))
    return None

print(f'{"pair":9s} {"metric":11s} {"ctl":>7s} {"trt":>7s} {"Δmed":>7s} {"round Δs":>22s} sign')
res = {}
for sid in PAIRS:
    res[sid] = {}
    for m in METRICS:
        ct, tt, ds = [], [], []
        for r in (1, 2, 3):
            c, t = load(sid, 'ctl', r), load(sid, 'trt', r)
            cv, tv = metric(c, m), metric(t, m)
            if cv and tv:
                ct.append(cv); tt.append(tv); ds.append(100 * (tv - cv) / cv)
        if not ds: continue
        mc, mt = statistics.median(ct), statistics.median(tt)
        d = 100 * (mt - mc) / mc
        sign = 'all+' if all(x > 0 for x in ds) else ('all-' if all(x < 0 for x in ds) else 'mixed')
        print(f'{sid:9s} {m:11s} {mc:7.1f} {mt:7.1f} {d:+6.1f}% {", ".join(f"{x:+.1f}" for x in ds):>22s} {sign}')
        res[sid][m] = (d, ds)
    sc = [slots(load(sid, 'ctl', r)) for r in (1, 2, 3)]
    st = [slots(load(sid, 'trt', r)) for r in (1, 2, 3)]
    print(f'{"":9s} expert slots ctl={sc} trt={st}')
    print()

print('A/A floor (aa_floor, |Δ|):')
for m in METRICS:
    ds = [abs(x) for x in res['aa_floor'][m][1]]
    print(f'  {m:11s} med {statistics.median(ds):4.1f}%  max {max(ds):4.1f}%')

print('\nParity (greedy_warm hash, trt vs ctl, and within-arm round stability):')
for sid in PAIRS:
    eq = [load(sid, 'ctl', r)['parity']['greedy_warm'] == load(sid, 'trt', r)['parity']['greedy_warm'] for r in (1, 2, 3)]
    base = [load(sid, 'ctl', r)['parity']['greedy_warm'] for r in (1, 2, 3)]
    print(f'  {sid:9s} pair-match {sum(eq)}/3, ctl round-stable {len(set(base)) == 1}')
