#!/bin/bash
# Per-batch verify for integ sweep: CUDA 13.0 build + engine tests + serve tests.
# Usage: bash tmp/check_sweep.sh <batchNN>
set -u
cd /c/Users/blsor/Strata
B="$1"
LOG="build-verify.log.$B"
{
echo "=== $B $(date -Is) HEAD=$(git rev-parse --short HEAD) ==="
cmake -B build-cu130 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER="C:/Users/blsor/cuda130/nvidia/cu13/bin/nvcc.exe" \
  -DSTRATA_Q6K_EXPERTS=ON 2>&1 | tail -5
echo "--- cache option check (new options must match, not stale) ---"
grep -E 'STRATA_[A-Z0-9_]*:BOOL' build-cu130/CMakeCache.txt
ninja -C build-cu130 strata 2>&1 | tail -15
echo "BUILD_STRATA_RC=$?"
ninja -C build-cu130 2>&1 | tail -8
echo "BUILD_ALL_RC=$?"
ctest --test-dir build-cu130 --output-on-failure -j 6 2>&1 | tail -25
echo "CTEST_RC=$?"
./.venv/Scripts/python.exe - <<'EOF'
import unittest, glob, importlib, sys
fails=[]
for f in sorted(glob.glob('serve/test_*.py')):
    mod = f[:-3].replace('/', '.')
    try:
        suite = unittest.defaultTestLoader.loadTestsFromName(mod)
        res = unittest.TextTestRunner(verbosity=1).run(suite)
        print(f"{mod}: run={res.testsRun} ok={res.wasSuccessful()}")
        if not res.wasSuccessful(): fails.append(mod)
    except Exception as e:
        print(f"{mod}: LOAD-FAIL {e}"); fails.append(mod)
print("SERVE_FAILS:", fails)
sys.exit(1 if fails else 0)
EOF
echo "SERVE_RC=$?"
echo "=== $B DONE $(date -Is) ==="
} > "$LOG" 2>&1
tail -6 "$LOG"
