#!/bin/bash
# Wave 24 of the integ sweep (integ/community-2026-10), 2026-10-22.
# Resolves nothing: stops on first conflict, records PR; fix + `git commit` then rerun.
set -u
cd /c/Users/blsor/Strata
STATE=tools/sweep/done_wave24.txt
BR=integ/community-2026-10
git checkout -q "$BR"

declare -A D
D[3241]="serve: keepawake primes its daemon before the first acquire"
D[3305]="fix(setup AMD): engine probes get the engine's lib_dirs on LD_LIBRARY_PATH on Linux"
D[3521]="tests: stop the wizard-discovery test from reading the checkout's own run configs"
D[3522]="calibrate: keep the PCIe/draft-floor winner only when it beats this run's noise"
D[3557]="setup.sh: fail fast with a clear message on non-x86_64 (ARM)"
D[3686]="Metal: honor a client cancel during the prompt read (stop at the next chunk)"
D[3235]="strata_bench: both Windows start-up paths crash (NameError port, GBK decode)"
D[3414]="bench: fix startup crashes and Windows version detection"
D[3220]="perf(hip): vectorize HC_Q8 prefill dequant, gfx12 Q8_0 gr_upmix"
D[3225]="perf(hip): fused_gr tsum on gfx12, fix q8_inject guard, LDS-only barriers"
D[3222]="perf(prefill): single LDS-only barrier per stage in native_w11_kernel"
D[3223]="perf(prefill): port wmma_gemm and gdn_chunked WMMA kernels to RDNA4 gfx12"
D[3224]="perf(hip): tune RDNA4 QSA prompt WMMA kernel (qsa_prompt_attn.cu)"
D[3230]="perf(hip): exact wave32 QSA decode kernels, B6 MMVQ fragment reuse (gfx12)"
D[3233]="perf(hip): exact QSA block scoring and top-k selection on gfx12"
D[3226]="perf(hip): optimize GDN verify kernels with LDS-only barriers, batched commits"
D[3259]="perf(hip): optimize EXPDEC routed-expert kernels, default on gfx12"
D[3264]="perf(hip): MTP draft, shared expert, verify/sampler head kernels on RDNA4"
D[3281]="add request-scoped prefill VRAM loans"
D[3392]="add opt-in advisory VRAM plan"
D[3388]="rebase Qwen3.6 bounded low-memory expert cache onto main"
D[3564]="Metal: working session cache, and the 12 GiB its arena reserved for nothing"
D[3678]="q36: quantized KV caches behind --kv (int8 / fp8 / rk4v4-e8 q4)"
D[2079]="native Windows support for Intel SYCL GPUs"
D[3729]="SYCL: flag Intel iGPUs as integrated so the expert cache sizes from MemAvailable"
D[2114]="sycl: run on an integrated Arc (Arc 140T, xe) with free memory from /proc/meminfo"
D[2115]="setup: the SYCL setup on an integrated Arc (xe): native launcher, RAM fit rule"
D[2240]="Remote stage: a worker for the SYCL build (Intel Arc)"
D[3299]="Metal: run Gyro (trellis TQ2_T/TQK6/TQK7) models on Apple Silicon"
D[3457]="Metal: opt-in --dflash DFlash2 drafter in place of the MTP layer"
D[3662]="support Qwen3.8-Flash-Next Coder (256 experts) on Apple Silicon Metal"
D[1887]="Remote stage: conversation checkpoints across the link (--prompt-cache)"
D[3466]="tools: the busbar gateway, one OpenAI address in front of several providers"
D[3643]="Jev System One decision endpoint (/v1/systemone), opt-in"
D[3663]="TUI: preserve log config and add indexed log rotation on launch"

BATCHES=(
"24a 3241 3305 3521 3522 3557 3686 3235 3414"
"24b 3220 3225 3222 3223 3224 3230 3233 3259 3226 3264"
"24c 3281 3392 3388 3564 3678"
"24d 2079 3729 2114 2115 2240"
"24e 3299 3457 3662 1887 3466 3643 3663"
)

done_list() { grep -qx "$1" "$STATE" 2>/dev/null; }
mark_done() { echo "$1" >> "$STATE"; }

for BATCH in "${BATCHES[@]}"; do
  read -r K REST <<< "$BATCH"
  for N in $REST; do
    done_list "$N" && continue
    if ! git rev-parse -q --verify "refs/heads/new/pr$N" >/dev/null; then
      echo "SKIP $N (no fetched head)"; mark_done "$N"; continue
    fi
    echo "=== merging #$N batch $K ==="
    if git merge --no-ff -q -m "merge #$N (${D[$N]}) — integ wave 24 batch $K" "new/pr$N"; then
      mark_done "$N"; echo "OK $N $(git rev-parse --short HEAD)"
    else
      echo "CONFLICT $N — conflicted files:"; git diff --name-only --diff-filter=U | head -20
      exit 10
    fi
  done
done
echo "ALL WAVE-24 MERGES DONE"
