#!/bin/bash
# run_bench.sh - wait for the HIP build of mmvq_il_parity, then run the documented bench in its own scope.
set -u
BIN=/home/soren/Strata-integ/build-hip/mmvq_il_parity
LOG=/home/soren/Strata-ab/ilbench-20261028/bench-full.log

for _ in $(seq 1 120); do
    [ -x "$BIN" ] && break
    sleep 30
done
if [ ! -x "$BIN" ]; then echo "mmvq_il_parity not built after 60 min"; exit 2; fi

systemd-run --user --collect --scope --property=MemoryMax=8G \
    --setenv=PATH=/usr/sbin:/usr/bin:/bin:/opt/rocm/bin --setenv=ROCM_PATH=/opt/rocm \
    --setenv=HIP_PATH=/opt/rocm --setenv=STRATA_IQ_STAGE_TABLES=1 \
    --working-directory=/home/soren/Strata-integ \
    -- "$BIN" --bench --emit-table 2>&1 | tee "$LOG"
echo "bench exit=${PIPESTATUS[0]} log=$LOG"
