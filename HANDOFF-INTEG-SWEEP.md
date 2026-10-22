# Integ sweep — handoff (2026-10-22)

Working notes for whoever picks up the upstream-PR sweep on `integ/community-2026-10`
(fork remote `fork` = bsorensen110/Strata; `origin` = Niko1221/Strata, read-only).

## State when this was written

- Everything through **wave 24 batch 24c first PR (#3281)** is merged and pushed:
  waves 18-23 (fixes, GPU perf, prefix/KV reuse, API, web UI), 24a (8 fixes),
  24b (HIP RDNA4 perf chain #3220/#3225/#3222/#3223/#3224/#3230/#3233/#3259/#3226/#3264),
  24c #3281 (request-scoped prefill VRAM loans).
- The **fork's DFlash2 line** (group-4 attention, fused conv, sampled speculation,
  Q4_K_M HIP artifacts — 3 commits that only existed on the fork) is merged in
  (`git log --oneline` top merge with "merge fork/integ/community-2026-10").
  Its `fuse_conv` uses were guarded for DFLASH2-off builds in that same commit.
- Verified at push time: `build-cu130.bat` green (0 FAILED), 170 tests green across
  serve/test_mcp.py, test_mcp_media.py, test_vision_elastic.py, test_vision_cache_wake.py,
  test_tracing.py, test_monitor.py, test_kv_pool.py, tools/test_strata_mcp.py;
  serve/test_web_history.py 18/18 (node harness); tools/test_calibrate.py 49/49;
  tests/terminal/test_discovery.py 25/25; tools/test_mtp_fetch.py 19/19.
- Working tree is clean (ignore `M serve/web/app.js` — pure EOL noise; git
  normalizes it to nothing).

## Remaining queue (wave 24, batches 24c-24e)

24c rest: **3392** (advisory VRAM plan), **3388** (low-memory expert cache rebase),
**3564** (Metal session cache + arena), **3678** (quantized KV caches, `--kv` int8/fp8).
24d SYCL: **2079, 3729, 2114, 2115, 2240** (Intel GPU / Arc stack).
24e: **3299, 3457, 3662** (Metal features), **1887** (remote-stage conversation
checkpoints), **3466** (busbar gateway tool), **3643** (/v1/systemone endpoint),
**3663** (TUI log rotation).
PR heads are fetched as `new/prNNNN`. Classification report of the full 359-PR open
list + what's already in: `tools/sweep/sweep_report.txt` (+ `tools/sweep/merged_nums.txt`).
Skip benchmark-only / docs-only / translation PRs — the queue above is the filtered set.

## How to resume

From the repo root:

    bash tools/sweep/merge_sweep24.sh

- Resumable: skips PRs listed in `tools/sweep/done_wave24.txt` (only appends on
  success). Stops on the first conflict (exit 10) and names the PR + files.
- Resolve, `git add` the files, `git commit` (never a partial commit mid-merge),
  then rerun the script.
- Any PR number you add to a batch **needs a `D[NNNN]="..."` message entry** in the
  script, or bash dies with `unbound variable`.
- After a batch: `cmd /c build-cu130.bat` (CUDA 13.0, quality build — the system
  CUDA 13.2 toolkit produces numerically broken IQ3_XXS kernels, do not use it),
  then `ctest --test-dir build-cu130 -j 6`, then the pytest clusters above.
  Full serve suite: ~9 min (`pytest serve/ tools/test_strata_mcp.py -q`).
- Engine-level quality probe before shipping a perf wave: 5K recall probe, release
  baseline scores 8/8.

## Known landmines

- `src/program/generate.cpp` is the hot file. When merging waves that touch
  `generateReply`-equivalents: keep a single `parked` Match (the #3584
  best_with_partial + #3615 read-gate version wins; `slot_source=-1` move goes
  after it), `carry_chain` must be declared before #1614's chain-carry in the
  move_cks block, one splice-aware `read_from`, `moved` hoisted to the trace's scope.
- `serve/server.py` tool loop (`run_with_mcp`) is merged from #2473 final: round-2
  requests must pass integ's `Service.run(..., declared_only=True)` graft — the
  fix is in `f6c9bfab`-era code; don't undo it.
- Web: `serve/test_web_history.js` VM stubs `paintLive/ws/searchPhase/settings/
  window/setSearchPhase/endSearchPhase/setSearchBubble/clearSearchBubble` — integ's
  Chat slice borrows those from the Monitor section; the VM slice can't see them.
- CMake: the WIN32 `ws2_32` link for `stage_link.cpp` must stay OUTSIDE the
  `WIN32 AND STRATA_ENABLE_HIP` guard (stranded there once; CUDA link broke).
- Preserve line endings when rewriting files via Python (`newline=''`, detect
  per-file: C++ sources here are CRLF, tools/*.py vary; the web harness matches
  on LF banner strings inside app.js).
- `feat/kv-q2k-window`: the fork's copy is ahead of this box's local branch —
  do not force-push it; the dev box owns that line (PR #2318 sibling).
- Never commit `build-cu130/`, `tmp/`, the recall/AB JSON dumps, logs, `.dmp`s,
  `engine.bak-*/`, `data/expert-profile-*.bin` — all local experiment litter.

## Sweep kit in-tree

- `tools/sweep/merge_sweep24.sh` — driver (batches, messages, state path)
- `tools/sweep/check_sweep.sh` — per-batch verify: CUDA build + ctest + serve tests
- `tools/sweep/done_wave24.txt` — what's merged (driver state)
- `tools/sweep/sweep_report.txt` / `merged_nums.txt` — classification + coverage
- `build-cu130.bat` — the quality-build recipe (paths are box-local; adjust if needed)
