#!/usr/bin/env python3
"""Recompute a matched replay from raw records, rejecting incomplete arms."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--output', type=Path)
a = p.parse_args()
root = a.root.resolve()
work = root / 'hipblaslt-benchmark'
arms = {}
for name in ('baseline', 'tuned'):
    folder = work / name
    record = json.loads((folder / 'arm-results.json').read_text())
    decode = json.loads((folder / 'decode-results.json').read_text())
    prefill = json.loads((folder / 'prefill-results.json').read_text())
    assert record['passed'] and decode['passed'] and len(prefill) == 15
    assert record['engine_pids_before'] == record['engine_pids_after'] and not record['remaining_engine_pids']
    fresh = [row for row in prefill if row['kind'] == 'fresh']
    assert len(fresh) == 6 and all(row['metrics']['reused'] == 0 for row in fresh)
    groups = defaultdict(list)
    for row in fresh:
        groups[int(row['metrics']['prompt_tokens'])].append(row)
    rates = {}
    for size, rows in groups.items():
        assert len(rows) == 3 and all(row['metrics']['prefill_ms'] > 0 for row in rows)
        rates[size] = sum(row['metrics']['fresh'] for row in rows) / (sum(row['metrics']['prefill_ms'] for row in rows) / 1000)
    trials = decode['trials']
    assert len(trials) == 3 and all(t['usage']['completion_tokens'] == 512 and t['finish_reason'] == 'length'
                                  and t['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0) == 0
                                  and t['timings']['predicted_ms'] > 0 for t in trials)
    tokens = sum(t['usage']['completion_tokens'] for t in trials)
    seconds = sum(t['timings']['predicted_ms'] for t in trials) / 1000
    assert tokens == 1536
    arms[name] = dict(record=record, rates=rates, decode=decode, tokens=tokens, seconds=seconds, decode_tps=tokens / seconds)
b, t = arms['baseline'], arms['tuned']
assert b['record']['engine_sha256'] == t['record']['engine_sha256']
assert b['decode']['sampling'] == t['decode']['sampling']
assert [r['prompt'] for r in b['decode']['trials']] == [r['prompt'] for r in t['decode']['trials']]
normalized = json.loads(json.dumps(b['record']['configuration']))
normalized['log'] = t['record']['configuration']['log']
normalized['env']['STRATA_HIPBLASLT_TUNING'] = t['record']['configuration']['env']['STRATA_HIPBLASLT_TUNING']
assert normalized == t['record']['configuration']
assert any('using hipBLASEx' in line for line in b['record']['engine_diagnostics'])
assert any('tuning enabled' in line for line in t['record']['engine_diagnostics'])
summary = [line for line in t['record']['engine_diagnostics'] if 'summary launches=' in line]
assert len(summary) == 1
counters = re.search(r'launches=(\d+) fallbacks=(\d+) unique_fallback_shapes=(\d+)', summary[0])
assert counters and int(counters[1]) > 0 and int(counters[2]) == int(counters[3]) == 0
assert b['rates'].keys() == t['rates'].keys()
report = dict(passed=True, prefill_comparison=[], runtime_lt_launches=int(counters[1]), runtime_lt_fallbacks=0)
for size in sorted(b['rates']):
    baseline, tuned = b['rates'][size], t['rates'][size]
    report['prefill_comparison'].append(dict(prompt_tokens=size, baseline_tps=baseline, tuned_tps=tuned,
                                             throughput_change_percent=(tuned / baseline - 1) * 100))
report['decode_comparison'] = dict(baseline_tps=b['decode_tps'], tuned_tps=t['decode_tps'],
                                  observed_change_percent=(t['decode_tps'] / b['decode_tps'] - 1) * 100)
profile = Path(t['record']['configuration']['env']['STRATA_HIPBLASLT_TUNING'])
rows = [line.split() for line in profile.read_text().splitlines() if line.split() and line.split()[0] in ('bf16', 'f16')]
shapes = json.loads((root / 'hipblaslt-shapes.json').read_text())
expected = {(s['dtype'], s['N'], s['K'], s['ldy'], bucket) for s in shapes for bucket in (4096, 8192)}
assert len(rows) == len(expected) and {(r[0], *map(int, r[1:5])) for r in rows} == expected
observed = {(d, int(n), int(k), int(ld)) for d, n, k, ld in re.findall(r'dtype=(bf16|f16) T=\d+ N=(\d+) K=(\d+) ldy=(\d+)', (work / 'tuned/engine.log').read_text())}
assert observed and observed <= {(s['dtype'], s['N'], s['K'], s['ldy']) for s in shapes}
verification = t['record']['verification']
assert verification['passed'] and all(c['passed'] for c in verification['checks'])
assert {c['name'] for c in verification['checks']} == {'arithmetic', 'system_marker_recall', 'multi_turn_recall', 'streaming'}
report['profile'] = dict(path=str(profile), rows=len(rows), geometries=len(shapes), sha256=hashlib.sha256(profile.read_bytes()).hexdigest())
report['arms'] = {name: dict(generated_tokens=data['tokens'], decode_seconds=data['seconds'],
                            engine_sha256=data['record']['engine_sha256']) for name, data in arms.items()}
if a.output:
    a.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
