#!/bin/bash
# Verify the HEAD parity binary before pushing: rebuild only if older than its sources, run the
# parity tool in a bounded scope, record sha256 and the final lines.
set -u
cd /home/soren/Strata-integ || exit 1
LOG=/home/soren/Strata-ab/commit-msgs/parity-HEAD.log
BIN=build-hip/mmvq_il_parity
: > "$LOG"
if [ "$BIN" -ot src/kernels/cuda/native_mmvq.cu ] || [ "$BIN" -ot src/kernels/cuda/iq3_grid.inc.h ] || [ "$BIN" -ot src/kernels/mmvq_il_parity.cpp ]; then
  echo "binary older than sources: rebuilding" | tee -a "$LOG"
  systemd-run --user --collect --scope --property=MemoryMax=24G \
    --setenv=PATH=/usr/sbin:/usr/bin:/bin:/opt/rocm/bin --setenv=ROCM_PATH=/opt/rocm --setenv=HIP_PATH=/opt/rocm \
    --working-directory=/home/soren/Strata-integ -- cmake --build build-hip --target mmvq_il_parity -j 8 >> "$LOG" 2>&1 \
    || { echo "BUILD FAILED" | tee -a "$LOG"; exit 1; }
fi
git rev-parse --short HEAD | sed 's/^/HEAD /' | tee -a "$LOG"
sha256sum "$BIN" | tee -a "$LOG"
systemd-run --user --collect --scope --property=MemoryMax=8G \
  --setenv=PATH=/usr/sbin:/usr/bin:/bin:/opt/rocm/bin --setenv=ROCM_PATH=/opt/rocm --setenv=HIP_PATH=/opt/rocm \
  --working-directory=/home/soren/Strata-integ -- "$BIN" >> "$LOG" 2>&1
echo "parity exit=$?" | tee -a "$LOG"
echo "FAIL lines: $(grep -c '^FAIL' "$LOG")" | tee -a "$LOG"
tail -n 1 "$LOG" | tee -a "$LOG"
