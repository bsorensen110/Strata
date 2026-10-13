#!/usr/bin/env python3
"""Check current-library calibration coverage and production-Gemm parity logs."""
import argparse
from collections import Counter
import ctypes
import json
from pathlib import Path
import re

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--table', type=Path, required=True)
p.add_argument('--rocm', type=Path, default=Path('/opt/rocm'))
a = p.parse_args()
root = a.root.resolve()
lib = ctypes.CDLL(str(a.rocm / 'lib/libhipblaslt.so'))
handle, version = ctypes.c_void_p(), ctypes.c_int()
assert lib.hipblasLtCreate(ctypes.byref(handle)) == 0
try:
    assert lib.hipblasLtGetVersion(handle, ctypes.byref(version)) == 0
finally:
    assert lib.hipblasLtDestroy(handle) == 0
lines = a.table.read_text().splitlines()
header = next(line.split() for line in lines if line.startswith('STRATA_HIPBLASLT_TUNING_V1 '))
assert len(header) == 3 and int(header[2]) == version.value
shapes = json.loads((root / 'hipblaslt-shapes.json').read_text())
expected = {(s['dtype'], s['N'], s['K'], s['ldy'], t) for s in shapes for t in (4096, 8192)}
rows = [line.split() for line in lines if line.split() and line.split()[0] in ('bf16', 'f16')]
keys = [(r[0], *map(int, r[1:5])) for r in rows]
assert len(keys) == len(set(keys)) == len(expected) and set(keys) == expected
calibration = (root / 'hipblaslt-calibration.log').read_text().splitlines()
meta = next(line for line in calibration if line.startswith('meta '))
assert f'device_arch={header[1]} ' in meta and f'hipblaslt_version={version.value} ' in meta
best = [json.loads(line.split('=', 1)[1]) for line in calibration if line.startswith('best_json=')]
assert len(best) == len(rows) and {(b['dtype'], b['N'], b['K'], b['ldy'], b['T']) for b in best} == expected
assert all(b['finite'] and b['padding_writes'] == 0 and b['relative_l2'] <= 1e-4 and b['max_abs'] <= 1e-2 for b in best)
checks = re.findall(r'case dtype=(bf16|f16) T=(\d+) N=(\d+) K=(\d+) ldy=(\d+) output_offset=(\d+) beta=([0-9.]+) relative_l2=([0-9.e+-]+) max_abs=([0-9.e+-]+) (PASS|FAIL)',
                    (root / 'hipblaslt-full-parity.log').read_text())
assert len(checks) == len(rows) * 2
assert all(c[-1] == 'PASS' and float(c[-3]) <= 1e-4 and float(c[-2]) <= 5e-3 for c in checks)
required = Counter()
for dtype, n, k, ld, t in expected:
    required[(dtype, t, n, k, ld, 0, 0.0)] += 1
    required[(dtype, 37, n, k, ld, 7, 1.0)] += 1
assert Counter((c[0], *map(int, c[1:6]), float(c[6])) for c in checks) == required
assert f'summary launches={len(checks)} fallbacks=0 unique_fallback_shapes=0' in (root / 'hipblaslt-full-parity-engine.log').read_text()
print(json.dumps(dict(architecture=header[1], hipblaslt_version=version.value,
                      calibrated_rows=len(rows), parity_checks=len(checks), fallbacks=0,
                      maximum_calibration_error=max(b['max_abs'] for b in best)), indent=2))
