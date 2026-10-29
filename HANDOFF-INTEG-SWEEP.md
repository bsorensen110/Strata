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

## Wave 24 status — Linux sweep pass (2026-10-24)

All of 24c/24d/24e are merged on Linux; `tools/sweep/done_wave24.txt` holds 33 entries.
Merged this pass: 3564, 3678, 2079, 3729, 2114, 2115 (24d), 3299, 3457, 3662, 1887,
3466 (clean), 3643, 3663 (24e).

**PARKED — #2240** (Remote stage: a worker for the SYCL build): 211 conflicts over 19
files (`src/net/stage_link.cpp` 40, `src/net/stage_link_test.cpp` 46,
`src/program/generate.cpp` 53) against an 11.8k-line PR. Needs a dedicated session.

Resolution policy that worked, reuse it:
- integ already carries a newer version of the PR's feature -> take HEAD
  (2079 Windows/SYCL setup, 3299 Gyro Metal, 3662 conflicts).
- both sides add distinct things -> union (2114 doc sections, 2115 setup igpu helpers,
  3457 Metal drafter members, 1887 struct fields, 3643 routes/imports).
- After ANY union, grep for a duplicated call at the seam: #3457's union duplicated
  `gr_read`/`hc_read` of `hc_attn`, which would have applied the mixer read twice.

Follow-ups owed:
- #3662: `constexpr int NEXP = 512` in `metal/src/engine/window.cpp` needs the runtime
  `n_expert_` port before 256-expert Coder models work on Metal.
- `metal/CMakeLists.txt` carries a stray `>` line (pre-existing integ artifact; Metal is
  not build-gated on Linux).

Linux build gate (the Windows `build-cu130.bat` recipe does not run here):
`cmake -B build-verify -S . -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON
 -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++
 -DSTRATA_BUILD_TESTS=OFF -DSTRATA_Q6K_EXPERTS=ON`  (Unix Makefiles, do not add -G Ninja)
then `systemd-run --user --scope -- make -C build-verify -j16 strata` — the agent's shell
cgroup caps RAM at 4 GiB, so the scope wrapper is required.

### Auditing a union for duplicated statements
`git diff --name-only <pre-sweep-base> HEAD | python3 tools/sweep/dup_audit.py`
reports ADJ (identical line back-to-back), NEAR (within 4 lines), FN (identical function
definition line twice) and any surviving conflict markers, over just the swept files.
Expect false positives: `#if defined(_WIN32)` / `#else` pairs define the same function twice
legitimately, and tests sometimes call a function twice on purpose. Duplicates arrive as whole
SETS — the Metal trellis block carried two `deq_tq` variants plus two identical
`deq<T_TQ2T/TQK6/TQK7>` specialisation sets.

## Upstream main merged (8183c6b6, v0.2.4.1) — 2026-10-24

Merge commit `bca6d251` (parents `6f35c688` ours, `8183c6b6` upstream; 391 upstream commits,
476 files, +68k lines). 26 conflicted files, 55 markers; HIP build of `strata` is GREEN
(`build-verify`, md5 recorded in the commit messages).

**DFlash2 decision (Brooks, this session):** upstream's drafter is adopted (`--draft dflash2`,
pack format, `tools/dflash2_pack.py`); the fork's own `--dflash` GGUF drafter is retired. Its
implementation is preserved on branch `integ/pre-upstream-0.2.4` (== `6f35c688`). `--dflash` and
`--dflash-*` now exit 2 with a pointer to the replacement instead of being silently accepted.

What the excision cost, so the next merge is cheaper:
- `verify.hpp/cpp` + `prefill.hpp/cpp`: the fork's tap API is gone (`set_tap_layers`, `n_taps()`,
  `taps()`, `tap_stride()`, `tap_layers_[8]`, `n_taps_`, `taps_`, `taps` Impl member, `g_n_taps`,
  the second uint16_t `on_taps`). Upstream's `set_taps` / `tap_dev_` / `tap_on_` / float `on_taps`
  is what stays — keep those, they are live.
- `generate.cpp`: ~45 sites. `has_dflash` is now `const bool = false`, `dflash_k`/`use_dflash`
  survive as inert locals; the dead `if (has_dflash)`/`if (use_dflash)` blocks still compile and
  are harmless but should be deleted on a quiet pass.
- CMake: `strata_engine` needed the fork's `peer_link.cpp`, `kv_pool.cpp`, `kv_chunks.cpp`,
  `rope_cache_migration.cpp` added back — upstream's source list lacks them while the merged
  `generate.cpp` still calls them. Two duplicate targets had to be unpicked (`dflash2_parity`,
  and `strata` itself from a keep-both hunk).

Still owed:
1. Metal side of `--dflash`: `#3457`'s `df_*` members in `metal/src/engine/window.cpp` and
   `prompt_chain.inc`, and the `STRATA_DFLASH` block in `metal/CMakeLists.txt`. Metal is not built
   on Linux, so this rots silently — remove or keep deliberately, do not leave it half-there.
2. ~14 fork dflash sources are still tracked. `include/strata/spec/dflash_policy.hpp` is LIVE
   (upstream calls `dflash_proposal_fits`/`dflash_truncate_prefix` from it) — keep it. The rest
   (`core/dflash.hpp`, `core/dflash.cpp`, `core/dflash_runtime.cpp`, `spec/dflash_drafter*.{hpp,cpp}`,
   `spec/dflash_backend.hpp`, `spec/dflash_ref.hpp`, `kernels/dflash2_parity.cpp`,
   `kernels/dflash_taps_test.cpp`) appear dead; delete in the same quiet pass.
3. `serve/` + `tools/` pytest not yet run against the merged tree (system python lacks pytest;
   use `/home/soren/Strata/.venv/bin/python -m pytest serve/ -q`, excluding the `sys.argv[1]` scripts).

## Upstream main merged (4b9e05a2, v0.2.6) — 2026-10-25

Merge of `origin/main` **4b9e05a2** (tag v0.2.6; v0.2.4.2 → v0.2.5 → v0.2.6 = 186 commits,
249 files) into `integ/community-2026-10`. Parents: `cf70f46a` (ours) + `4b9e05a2`.
**Rollback ref: branch `integ/pre-upstream-0.2.6` = `cf70f46a`** (branch, not tag — a plain tag
on the live branch poisons `git describe`, see the 0.2.4 notes).

25 conflicted files, 74 markers: 5 code + 20 docs. HIP build of `strata` was **GREEN on the
first pass** (`build-verify`, gfx1201, md5 `98535f90dda80ef3bf0cf2b1b0d22b51`, `strata --version`
= engine 0.1.51).

Code resolutions — all unions; policy = upstream wins its own refactors, fork keeps live grafts:
- `generate.cpp`: took upstream's `--adapt-async` help text **and** its `adapt_async_given`
  setter (the arena auto-trial; read at the `adapt_async_default` gate, ~line 10214); kept the
  fork's `--numa-hot-node` help entry (live, 16 refs) and `spec_min_p_set`. Upstream's new help
  literal lines carry **no trailing `\n`**, so that paragraph prints as one long `--help` line —
  upstream's own text, kept verbatim, don't "fix" it.
- `serve/server.py`: the conflict is ONLY the `serve.frontend`/`serve.mcp` import lines —
  upstream's moved import block (runconfig, engine_component, `tls.install()`, hipblaslt_tuning,
  vision_reader, draft_layer, drafting, user_args, gpu_roles, ready, owner_guard,
  launch_identity) **auto-merged below the block**. A context-printing script that shows lines
  after `>>>>>>>` makes that block look like part of "theirs" and invites a duplicate import.
  Union: `assistant_prefix_kw`, `require_messages`, `videos_of`, `MAX_RESULT_IMAGES`, plus the
  fork's `serve.tracing` import.
- `serve/frontend.py`: fork's `assistant_prefix_kw` + `_order_chat_tool_results`, then upstream's
  `require_messages`, ahead of the shared `openai_to_messages`.
- `strata_control/runtime.py`: one `restart()` signature carrying upstream's `MAX_START_WAIT`
  bound AND the fork's `refuse_busy` (live: `restart_if_idle` passes it; the RPC name switches
  to `restart_if_idle`). Taking either side alone breaks the other's caller.
- `apps/host/src/components/SettingsUpdates.tsx`: single `native-job-diagnostics` import
  (`jobErrorText` + `nativeJobDiagnostics`), fork's `RecoveredUpdateHistory`, upstream's
  `shownEngineVersion`. Pre-existing fork litter: `RecoveredUpdateHistory` is imported and never
  rendered, and `diagnostics` is a dead local — on BOTH sides, so not a merge seam; apps/host is
  not built on Linux. Clean on a quiet pass.

Docs — **policy chosen by Brooks: hybrid.** Upstream's rewritten prose taken wholesale in all 20
doc/README/packaging files. The fork's displaced passages from `AMD_HIP.md` (215 lines),
`DETAILS.md` (2,167) and `INTEL.md` (928) are extracted verbatim into
`docs/FORK-AMD_HIP.md`, `docs/FORK-DETAILS.md`, `docs/FORK-INTEL.md` (archival addenda, each
linked from a pointer line under its page's H1). Fork doc content dropped and NOT extracted,
recoverable with `git show cf70f46a:<file>` (ours-only lines inside the conflict blocks):
`MULTI_GPU.md` 377, `README.md` 272, `README.fr.md` 258, `TROUBLESHOOTING.md` 158, `MAC.md` 152,
`INTEL_ARC.md` 143, `AI_SETUP.md` 82, `SECOND_GPU.md` 43, `OLDER_GPUS.md` 39, the five other
README translations 28 each, `UNSLOTH_Q4.md` 16, `packaging/README.md` 11, `STRIX_HALO.md` 9.

Verified after resolving: 0 conflict markers in the tree; `py_compile` clean over all 71 changed
`.py`; `import serve.server`, `serve.frontend`, `serve.tracing`, `serve.runconfig`, `serve.mcp`,
`strata_control.runtime` clean under `/home/soren/Strata/.venv/bin/python`; `dup_audit.py` over
the hand-resolved files reports NEAR noise only, nothing at a seam. Two `ADJ` hits in the wide
audit (`strata_cli/launch/screens.py:1037` duplicated `return out`,
`tests/terminal/test_flow2_022.py:311`) are byte-identical to `origin/main` — upstream's own,
worth a PR upstream. Fork work confirmed at the tip: the staged tree vs `origin/main` shows the
fork residual in all 16 files the DFlash2 serve-mode / park-restore / FP16-ring / 4k-window
commits touch.

Not run in this session (owed):
1. `serve/` + `tools/` pytest clusters and `ctest` (`build-verify` is `STRATA_BUILD_TESTS=OFF`).
2. The 5K recall probe before shipping a perf wave (release baseline 8/8) — the merge moved
   `dsv41`, `iq_avx512` and CPU kernels, so run it before promotion.
3. Promotion. **Prod is untouched**, still the 0.2.4-era engine md5 `49ec4c9f`; `~/promote-integ.sh`
   copies `build-verify/strata` + the python layer and rolls back on health failure.
4. Wave 24's parked **#2240** (Remote stage worker, 211 conflicts) and the 0.2.4 follow-ups
   above (Metal `--dflash` side, dead fork dflash sources, dead `if (has_dflash)` blocks).

### Post-merge test pass (2026-10-25, on 2a9befe8)
- `serve/test_frontend.py` 26 passed + 58 subtests — the frontend union (`require_messages`,
  `assistant_prefix_kw`, `_order_chat_tool_results`).
- `serve/test_mcp.py` 29 passed + 8 subtests — imports `serve.server`, so it exercises the
  import union (`tls.install()`, `MAX_RESULT_IMAGES`, `serve.tracing`).
- `tests/terminal/test_runtime.py` + `test_runtime_idle.py` 59 passed, 4 skipped — the
  `restart()` signature union (`refuse_busy` + `MAX_START_WAIT`).
Still owed: the rest of `serve/`, `ctest`, and the 5K recall probe before promotion.

### Defect fixes + production promotion (2026-10-25)

Fixed and committed as **`9d4d0a59`** (`fix two real defects that came in with upstream 0.2.6`):
- `strata_cli/launch/screens.py`: removed one unreachable duplicate `return out`.
- `src/program/generate.cpp`: added the missing `\\n` escapes to three `--adapt-async` help
  string literals. Verified the built binary prints the paragraph across separate lines.
- The apparent duplicate in `tests/terminal/test_flow2_022.py` is intentional: the test emits
  30s/60s/60s and asserts two visible heartbeat lines (deduplication of the repeated 60s event).
  Left unchanged; `pytest tests/terminal/test_flow2_022.py -q` = 30 passed.

Rebuilt after both fixes: HIP gfx1201 `strata` green, md5
`21739e622be44f82651855bf62e7053b`.

Before promotion, the deployed release binary (`49ec4c9f`) passed the 5K needle probe **8/8**
(depths 5/15/25/35/50/65/80/95); raw output `recall-baseline-49ec4c9f.json`.
Then ran `~/promote-integ.sh` with the repaired v0.2.6 candidate. It created rollback artifacts:
`/home/soren/Strata/engine/strata.pre-integ-0.2.4-20261025-154155.bak` and
`/home/soren/Strata-backup-python-20261025-154155.tgz`; synced the Python layer and started the
service. Post-promotion verification:
- `strata.service`: active/running, NRestarts=0.
- live `/home/soren/Strata/engine/strata` md5 matches candidate `21739e622be44f82651855bf62e7053b`.
- authenticated `/v1/models`: HTTP 200; `swift-1.5-iq3_s` loaded at context 262144.
- real arithmetic inference: 19 x 23 -> `437`, finish_reason=stop, 36 completion tokens.
- **Post-promotion 5K needle probe: 8/8** at the same eight depths; raw JSON and stdout in
  `recall-prod-v0.2.6-21739e62.json` / `.log`.
- promotion script's initial health and model-list checks passed. Its exact `ready` smoke request
  ended `finish_reason=length` after 16 reasoning tokens and the engine correctly reported no
  final answer; the explicit arithmetic request above subsequently completed normally.

Current prod = v0.2.6 + two defect fixes, build hash above. Rollback artifacts retained; the
service has not restarted unexpectedly. Prod config (2026-10-28): env aligned to upstream gfx12 defaults
(see `~/Strata-ab/gfx12align-batch-20261028/METHOD.md` — four 0.2.7 decode-half switches set explicitly
because 0.1.51 predates their auto-enable; fork-era ATTN_PRE75/PA_FAST/DENSE_MMQ and dead HIP_MMVQ_IL
removed). IQ_STAGE_TABLES is OFF after neutral speed A/B. Current serving uses MTP (`--mtp --spec 8`)
with batch on (`"parallel": 4` + `--batch-mtp`), no DFlash2.

Batch decision: the first comparison was confounded because batch disables DFlash2. Controlled MTP-only
A/B with same four prompts and `--batch-mtp` on batch arm: four-way aggregate 166.46 vs 140.85 tok/s
(+18.2%); single-request median 165.95 vs 169.40 tok/s (-2.0%). User approved enabling `parallel: 4`
plus `--batch-mtp`. Live engine cmdline verified `--batch 4`, `--batch-mtp`, `--mtp` (no DFlash2); the
engine allocated 4 slot drafters (51 MiB each). The service is active, NRestarts=0, authenticated
`/v1/models` HTTP 200. Remaining QA not done: full serve/pytest suite and ctest
(`STRATA_BUILD_TESTS=OFF` in the build); these were not required for focused QA.

## Upstream main merged (5e8a4cca, v0.2.7) — 2026-10-28

Merge commit `377bd6aa` (parents `aae61677` ours, `5e8a4cca` = tag v0.2.7; **367 upstream
commits**, 477 files, +44,936 / −21,684) into `integ/community-2026-10`.
**Rollback ref: branch `integ/pre-upstream-0.2.7` = `aae61677`** (pushed to `fork`, so the
0.2.6 line is recoverable server-side too). 32 repair commits sit on top of the merge;
tip at write time is `529bec00` (`git log --oneline 377bd6aa..HEAD`).

### The failure mode that dominated this merge: one-sided hunks

The merge's CMakeLists.txt hunks applied **without conflict** and silently deleted fork-only
lines (fork `CMakeLists.txt:639` source tail vs merged `:701`). A HIP-only configure
(`STRATA_BUILD_CONVERSATION_TESTS=OFF`) hides most of the fallout, so the first build looks
fine and the link errors show up weeks later. What was lost and had to be re-grafted:
- `src/core/rope_cache_migration.cpp`, `kv_chunks.cpp`, `kv_pool.cpp` dropped from
  `strata_engine`'s source list while `rope_cache_migration_test` still registered and called
  `migrate_rope_cache_to_yarn4` — the exact same trio the 0.2.4 merge had to add back
  (see the 0.2.4 notes). **This is a recurring upstream-side gap; check it every merge.**
- `native_mmvq_indexed`: the header declaration and both call sites survived, the definition
  and the `launch_indexed` chain did not; `native_mmvq_multi_kernel` also lost its `INDEXED`
  template axis and `row_ids` argument. Restored as `INDEXED=false` default, which folds the
  row index back to `row0 + i` so the dense (production decode) instantiations compile
  unchanged.
- Fork CMake test/target registrations: the HIP ctest family, `ple_reader_*` lifecycle
  harness, PeerLink/`numa_policy` targets, the dflash2 family, the fork's Usage-panel web
  slice, `expert_source.hpp` as a 3-way union (fork dst2 DMA fields + upstream `stage_ev_`).

Method that worked (reuse it next merge): census the **whole tree** — function definitions and
the CMake source/test lists — and diff those sets against `git show <rollback-ref>:...`,
instead of chasing link errors one at a time. Anchor grep/replace regexes at **column 0**;
unanchored patterns matched indented call sites and corrupted argument lists.

### Repairs worth remembering

- `launch_pdl` forwards **every** kernel parameter explicitly, so a defaulted trailing kernel
  parameter still needs an argument at each `launch_pdl` site. Five dense sites now pass
  `(const int32_t*) nullptr` (`d0c833d8`). Forgetting this is a build break, not a warning.
- cgroup v2 parsing (`src/core/expert_source.cpp`, `679c077f`): `/proc` limit lines carry an
  **absolute** path relative to the mount root, so `root / path` threw the root away. Hybrid
  hosts mount v2 under `<mount>/unified`; some ancestors are not visible from this namespace.
  Resolve as `relative_path()` + `lexically_normal()`, reject `..`, walk to the deepest
  **visible** ancestor (ancestor limits still constrain descendants), and if no mount is
  visible at all, `continue` and use host `MemAvailable`. `file_expert_source_test` now
  exercises namespace/hybrid/partial-path/no-mount/traversal cases.
- `conversation_copy_streams_test` (#2409): HIP's lazy memcpy staging path is initialised by
  the **first** async copy, so the first private stream/event measurement saw 148 MiB of
  runtime init. Warm the runtime with a blocking `cudaMemcpy` + sync before the measured
  section, and keep the test `RUN_SERIAL` — it races other GPU tests at `-j 8`.
- Environment-gated tests now exit **77** (ctest SKIP) rather than fail: `hip_multigpu_*`
  (one R9700 here), `platform_memory_test` (`ulimit -l` = 8192 KB vs a 256 MiB
  lock_resident probe). Enforced both in CMake and in-code.
- `prefill_mmq_parity` dereferenced a null `from_float_ref` for quant types this ggml has no
  host reference quantizer for — prints and skips the type now. `check_isa_guard.py` needed
  the `iq512_reference_gu_rows` AVX-512 allowlist entry.
- Optional `STRATA_BUILD_CONVERSATION_TESTS=ON` exposed two more seams (`529bec00`): a
  duplicated `suffix_reuse_dense_tail_test` registration (removed) and the fork's
  `suffix_reuse_partial_prefix_test`, which needs the removed v0.2.7
  `ConversationCache::Match::partial` / `best_with_partial` API — left unregistered until
  ported. `conversation_validation_test.cpp`'s fork-added no-draft/explicit-length
  assertions also target absent APIs; the fixture follows upstream now.

### Verification (all on this box, gfx1201)

Configure (note `STRATA_BUILD_CONVERSATION_TESTS=ON` — that is what surfaces the census
link-breakers; keep it ON for merge work):

    cmake -S . -B build-hip -DSTRATA_ENABLE_HIP=ON -DSTRATA_BUILD_TESTS=ON \
      -DSTRATA_BUILD_CONVERSATION_TESTS=ON -DSTRATA_Q6K_EXPERTS=ON \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx1201\;gfx1200 \
      -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++
    systemd-run --user --collect --scope --property=MemoryMax=24G \
      --setenv=PATH=/usr/sbin:/usr/bin:/bin:/opt/rocm/bin --setenv=ROCM_PATH=/opt/rocm \
      --setenv=HIP_PATH=/opt/rocm --working-directory=/home/soren/Strata-integ \
      -- cmake --build build-hip -j 20

- Build after the PR cherry-picks: **exit 0** (`~/Strata-ab/cherry-20261028/build-hip33.log`).
- CTest after picks: 226 registered, 6 fixture exclusions, **220 run: 205 passed, 15 skipped, 0 failed**
  (`ctest-merge-027.xml`, 219.45 s). `q2g_rows_bench_validate` passed (bit-exact sweep).
  One earlier parallel run flaked `stage_link_test`'s 0.3 s stop-flag assertion under `-j8`; isolated run
  passed. Set `RUN_SERIAL TRUE` (`70bbacce`) and reran the complete gate green. The 15 skips are listed in
  `~/Strata-ab/cherry-20261028/gate-cherry-serial.log`.
- Cherry-picked test: `serve/test_responses.py` = 43 passed, 1 skipped, 3 subtests. Full serve suite from
  before these picks had 5 `serve/test_jev_calib.py` failures (404s); isolated pre-merge `test_jev_calib.py`
  also fails `OpusPrereviewFixes.test_g3_log_flush_splits_by_tuple` (5 rows vs 3), so there is a pre-existing
  test-state problem. Full-suite baseline on rollback `aae61677` completed: it has the same five
  `SystemOneCalibrated` failures as the pre-pick tip (11 failed, 1,645 passed, 18 skipped, 11,176 subtests
  versus 5 failed, 1,747 passed, 20 skipped, 11,274 subtests). The five JEV 404s therefore predate #4210;
  the rollback run also had six other failures (one lifeline and five literal-think-guard cases), which remain
  separately unresolved. `serve/test_web_history.js` passed 17 cases on the final merge tip; the cherry-picks do not touch it.
- Prod after the full gate: active, authenticated `/v1/models` HTTP 200, NRestarts=0. **The merge binary
  was NOT promoted.**

### Upstream PRs cherry-picked after the v0.2.7 merge (2026-10-28)

| PR | Upstream commit(s) | Local commit(s) | Summary |
|---|---|---|---|
| #3946 | `16c857f0`, `635dc657`, `3b46bce7`, `0e217b6b` | `9c69f7f5`, `da1f4fd8`, `a2ec84d0`, `edbd4b53` | bit-exact AVX-512 Q2_0 CPU expert rows, benchmark + CTest + recorded Ryzen results |
| #4152 | `05768037` | `ded9c429` | check both HIP/CUDA f16-range reset symbol writes |
| #4210 | `a00a4be7` | `0dd6a5cc` | accept Codex `agent_message` items in `/v1/responses` |
| #4258 | `e09a687a` | `ff71203c` | opt-in `--drop-file-cache` after startup reads |

PR #3946 is a **CPU-side** Q2_0 kernel, not the GPU IQ3_S/IQ3_XXS decode path. The current Ryzen 9 9950X
has AVX-512; the PR's `q2g_rows_bench_validate` ran and passed its seeded memcmp sweep. The benchmark
numbers committed by upstream are from a Ryzen 7 9800X3D, not this R9700; no local speedup measurement
was made. #4258's flag is off by default, and the call site refuses to drop the cache when any expert
still reads from files; current production config has no full complementary RAM-resident mode, so we did
not enable it.

Cherry-pick ledger + reasoning: `~/Strata-ab/cherry-20261028/CHERRY-4X.md`. Full `git cherry-pick -x`
provenance is preserved in the commits. Worktree is expected to be ahead of `fork/integ/community-2026-10`
until the user asks to push this follow-up batch.

### Still owed before promotion / open fork-only work

1. **Promotion.** Prod is still the v0.2.6 engine (`md5 21739e622be44f82651855bf62e7053b`)
   now serving the **Turbo-Brilliance** pack (`model_name turbo-brilliance-iq3_s`, 12 parked
   slots, kv int8, 262144 ctx, bound to 192.168.1.156:8086). Before promoting a v0.2.7 build,
   run the 8-depth (5/15/25/35/50/65/80/95) needle gate on it; release baseline is 8/8.
   `~/promote-integ.sh` + `.bak`/tgz rollback artifacts are the established path.
2. `STRATA_HYBRID` and `cmake/hybrid.cmake` are unwired at top level. Half-restoring the
   fork's 4 gates against the merged file's 14 `if(STRATA_ENABLE_HIP)` gates is guesswork and
   this box has no CUDA toolkit — needs a CUDA+HIP host to configure and test.
3. DFlash v1 drafter (`src/core/dflash_runtime.cpp`) does not compile against the merged tree:
   the merge dropped the fork's BF16 opt-in in `NativeHead::load`; same root cause orphans
   `tests/core/dflash_owned_kv_test.cpp` and `dflash_head_bf16_test.cpp`. Upstream dflash2
   (`STRATA_HAVE_DFLASH2`) is not a replacement. Probe detail:
   `~/Strata-ab/merge-repair/job1/report.md`. Docs still tell users to pass the retired
   `-DSTRATA_ENABLE_DFLASH2` option.
4. Fork-only partial-prefix suffix reuse and the explicit-length / no-draft snapshot
   behaviour need a deliberate port to the v0.2.7 conversation/snapshot APIs.
5. gfx906's specialized INDEXED wave-kernel path is not restored (generic INDEXED MMVQ builds
   and passes on gfx1201; the gfx906 path is untested here).
6. Older queues still open: wave 24's parked **#2240** (Remote stage worker, 211 conflicts)
   and the 0.2.4 follow-ups (Metal `--dflash` side, dead fork dflash sources, the dead
   `if (has_dflash)` blocks in `generate.cpp`).

## PR #4114 review (2026-10-28, read-only; nothing applied)

Two Claude Code reviews of upstream draft PR #4114 (`Native RDNA4 gfx1201 NVFP4 engine`, head `b5e1887e`,
base v0.2.6). Full reports + probe dumps: `~/Strata-ab/pr-4114/jobA/report.md` (IQ3_S/IQ3_XXS optimization
mining, findings F1-F18, decline list N1-N13, measurement plan) and `jobB/report.md` (merge-candidate review).

Verdict: **do not merge as-is.** Ungated CPU/spec sources and an unconditional `-mavx512bf16` land in every
build; the `generate.cpp` argv hook re-routes `--model/--native/--pack` at any safetensors dir into the new
engine on HIP builds; `serve/server.py` gains an ungated, untested prefix-reuse memo that bypasses our `#567`
shared-prefix encoder; the 1.16 MB / 102-kernel embedded hsaco blob has no source for 71 kernels and its
generator hardcodes a private desktop path (not reproducible); the `gfx120` arch-prefix gate accepts
gfx1200/1250 and then `std::exit(1)`s at module load; the two CMake conflicts land exactly on the seams this
handoff exists to protect; and most body numbers are author-only (per-expert 618,240 B contradicts the PR's own
2,764,800 B stride). Only cleanly-takeable pieces: the `check_isa_guard.py` registration hunk, the
`nvfp4_modelopt_{avx512,avx2,pool}` CPU set as an opt-in unit **with a parity test we write**, and
`safetensors_reader.hpp` as tooling.

Best speedup leads and gfx1201 measurements (IL enablement + gfx12 key/table fix **committed on the integ
branch as `4c9980cb`**, 2026-10-28):
1. `native_mmvq_il` (multi-token GEMV: weights read once per verify batch) was compiled out on HIP and its type list
   omits IQ3_S(21)/IQ3_XXS(18); `docs/MMVQ_IL_TABLE.md` records -45% at 5 tokens for the same mechanism on a 110-SM
   card. Claude Code widened the existing non-PF IL kernels/entry points to HIP, added the HIP bench target, and ran
   `mmvq_il_parity --bench --emit-table` twice on the R9700. Both runs had 240 `BENCH` lines, zero `FAIL`, and ended
   `mmvq_il_parity: OK` (bitwise parity across T=1..8 and rows 1/2/4). First-run table has nonzero cells in 65/75
   (2-4 columns), 98/100 (5-8), and 10/25 (one-column); repeat counts are 65/75, 97/100, and 11/25, with two
   on/off flips (Q3_K 5-8 cell 7; Q5_K one-column cell 4), attributable to small-shape timing noise.
   Example: Q5_K LM head at T=4 is 1,713.2 → 878.6 µs (-48.7%); T=5 is 1,967.6 → 1,164.2 µs (-40.8%). Full
   measurements and emitted table: `~/Strata-ab/ilbench-20261028/jobC-report.md` and `bench-full*.log`.
   **Arch-key collision resolved in `4c9980cb`:** `il_cc()` now gives every 12.0 AMD card its own key **1202**, and
   `kIlRows_1202`/`kIlRows58_1202`/`kIl1Rows_1202` carry this card's cells — kept only where the two bench runs
   agree, else 0 (the kIlRows_89 method; 20 flips zeroed). `mmvq_il_parity check_tables()` spot-checks 1202.
   Post-install parity on the card: OK (T 1-8, rows 1/2/4 and the table, bitwise). Still pending: the bench's
   types do **not** include IQ3_S(21)/IQ3_XXS(18), and the turbo-brilliance dense side is all IQ3_S, so the IL
   path does nothing for our model yet; the IQ3 IL extension (bitwise parity vs `iq_mmvq`) is running as job D
   (`~/Strata-ab/ilbench-20261028/jobD-brief.md`); model/decode A/B follows once iq3 table cells are installed.
2. `STRATA_IQ_STAGE_TABLES` stages IQ3 codebooks in shared ("bitwise the same either way",
   `iq_kernels.cu:3403-3426`) but measured negative on other cards. **Ran the model-level A/B (2026-10-28):
   decode 97.2 (ON) vs 94.8 (OFF) tok/s, +2.5% but within noise (Welch t=0.47), medians identical 98.2/98.7 —
   neutral, matching upstream (P100 0.97-1.03x noise, 3090 Ti forced-on 3-13% slower). Left OFF (upstream
   default); unit file keeps `STRATA_IQ_STAGE_TABLES=0` (explicit) with an inline A/B comment.** Method and
   raw data: `~/Strata-ab/gfx12align-batch-20261028/` (`iqstage_ab.py`, `iqstage-{ON,OFF}.json`).
   The IL benchmark does not read/use this flag—it routes none of its types through `iq_mmvq`.
   `ds4_iq.cu:1344` (`kCanStage = TAB == 0`) disables activation staging for i-quants outright.
3. IQ3 decode's byte dot is already one instruction on this card (`__dp4a` -> `v_dot4_i32_iu8` for gfx1201,
   `hip_compat/intrinsics.hpp:19-25`) — no port needed, do not spend effort there.
4. The int8 wave32 WMMA already default-on for IQ3_S **prefill** (`moe_fused_iq.cu` `nw_wmma`) could be moved
   to decode shapes (M=2..8 rows per weight read) — rounding-changing (int8 activations), so opt-in gate.
5. Drafter-cost cuts that are parity-safe under greedy acceptance: pruned draft vocab (`native_mmvq_indexed`
   already exists), sink+window bounded draft attention, draft-expert truncation.

## Promotion 0.1.52 to production, cache limits, RAM read, DFlash2 state (2026-10-29)

### What is live
- Engine: `strata 0.1.52`, md5 `203258b8…`, built clean in `build-verify` from integ `dd84b2a5` (+ this docs commit). Options copied from `build-hip`, the tree the GPU CTest gate ran on: `STRATA_PREFILL_MMQ=ON`, `gfx1201;gfx1200`, tests ON. 0 build errors. Host subset 8/8 (`fork_runtime_options`, `conversation_*`, `kv_chunks_test`).
- Promotion: `~/promote-integ.sh`. 5K needle gate 8/8, smoke ok, config diff none. Record: `~/Strata-ab/promotion-20261029-001958.json`.
- Rollback set: `~/Strata/engine/strata.pre-promote-20261029-001958.bak` (0.1.51), `~/Strata/strata-swift-iq3_s.json.pre-promote-20261029-001958.bak`, `~/Strata-backup-python-20261029-001958.tgz`. Note: the rollback restores binary and Python layer, not config. The new args are accepted by 0.1.51 (checked with `--help`).
- Python layer synced from integ (`serve`, `strata_control`, `strata_cli`, `tools`, `chat.py`). Prod's web UI is now integ's version. Prod-only `serve/test_chat_sidebar.{py,js}` remain. The chat UI itself is not browser-tested; `/` returns 200.
- Restore after restart, prod: A (464-token transcript) parked to disk; `systemctl --user restart`; take-back of 464 tokens in 104 ms; 415 of 453 prompt tokens reused; `PELICAN-42` returned. Script: `~/.hermes/cache/scratch/gpu-window/prod_restart_test.py`.
- No like-for-like throughput benchmark against 0.1.51 was run. The promotion gate was needle plus smoke.

### Cache files from 0.1.51 were removed at first start, by design
- `session identity = model + config fingerprint`, and the config fingerprint includes `STRATA_VERSION` (`c.engine_version`). A version bump makes every 0.1.51 file "foreign"; `conversation_cache_index` deletes foreign files at startup. The log says `0 conversation(s) kept from earlier runs`.
- Backup: `/home/soren/Strata-data/conv-cache-backup-0.1.51-20261029` (8 files, 8,650,109,916 bytes, including 3.1 GB and 3.3 GB conversations). Usable only by 0.1.51.

### Prod cache settings (config changed 2026-10-29)
- `--conversation-cache-min-tokens 256` (was unset = 0). Basis, from logs: park 97-127 ms per file; take-back 104-280 ms warm and about 1.0 s cold right after restart; prefill 80-125 tok/s warm. Recompute at 256 tokens costs about 2-3 s, so 256 keeps short side requests off disk. Each park writes 240 MB or more. Not an A/B.
- `--conversation-cache-slots 12` to `2147483647` (count cap removed).
- `--conversation-cache-disk-mib 402429` to `1073741824` (1 PiB; size cap removed).
- `--conversation-cache-days 7` now explicit (default 7; the only eviction left).
- Guards that remain: `--session-min-free-mib` default 4096 (parks keep at least 8 GiB free on the cache disk); `--conversation-cache-min-free-mib` default 2560 (RAM floor).
- Park size fit from logs: bytes ~= 236 MB + 15.3 KB x tokens (4,421 tokens = 303.7 MB; 464 = 243.2 MB). A 200K-token conversation is ~3.3 GB.
- Prod config file mode is now 600 (it holds the API key). Pre-change backup: `strata-swift-iq3_s.json.pre-limits-20261029.bak`.

### Why the restore reads the conversation into RAM (open, not fixed)
- The disk take-back calls `session_file_read` (`src/program/generate.cpp` ~14733). The whole session file, K/V included, is parsed into host RAM before upload. Parks also build the whole image in host RAM first; `SavedConversation` holds the K/V vectors.
- The streamed two-pass reader is in `src/core/conversation_file.cpp` (`session_file_read_streamed`, `session_file_apply_streamed_df2`, `SessionKvBlock` in `conversation_file.hpp`). Nothing calls it. It is upstream `d438ca51`, fork `69fb7912` (8 files, +561/-50). The `generate.cpp` glue must refill each layer's KV residency (`kv_stream_reset` / `kv_ring_restore`) under `--kv-resident`.
- Upstream measured a 3.05 GiB restore in 3.4-3.5 s, with the next turn 99.99% reused.
- Effect today: the 2.5 GiB RAM floor refuses restores and parks that do not fit. Under prod's load (MemAvailable about 5.6 GB while running), conversations above about 3 GB (about 200K tokens) cannot be restored and recompute instead. Not tested at that size.
- Plan: wire streamed apply into the take-back and RESTORE paths, streamed write into park. Verify on GPU: bit-exact versus whole-file restore with `--kv-resident` and `--kv-pool-tokens`, then a multi-GB conversation.

### DFlash2 drafter context persistence: state
- Already in HEAD: `Dflash2Drafter::save_state/restore_state` (`src/core/dflash2.cpp`), payload v4 tail (`kVersionDf2`, `conversation_file.cpp`), park hook (`generate.cpp` ~11006), restore hook (~15106).
- Not verified on GPU in this tree. `dflash2_state_test` is `EXCLUDE_FROM_ALL` and not in CTest. Prod does not enable DFlash2.
- Before enabling: build and run `dflash2_state_test`; run a GPU park/restore round trip with DFlash2 on; check whether the superseded-leak fix from `8b14892c` is present (not compared line by line).

### Still open
- `--kv-pool-tokens` with a layer split: the refusal was removed in the port and the path is untested. Prod is single-GPU, so it does not affect prod.
- CTest: `ple_parity`, `expert_parity`, `pool_test` fail on missing local fixture files. Not yet confirmed against the pre-port tree.
- `dflash_head_bf16_test` and `dflash_owned_kv_test` are not registered. `suffix_reuse_partial_prefix_test` does not compile (pre-existing).
- `~/.hermes/cache/scratch/strata-test-4972d47c` worktree has uncommitted changes; left alone.

### Housekeeping done (2026-10-29)
- Removed worktrees `Strata-port-kv` and `Strata-port-disk`, and branches `port/kv-pool-tokens` and `port/disk-cache`. The six pre-squash `port/disk-cache` commits are represented in HEAD.
- Deleted gpu-window scratch: four candidate configs holding the API key, `restore_prod.sh` (old chained restart), scratch session files.
- Pruned 10 engine binaries dated 2026-10-20 to 10-24 (~245 MB) from `~/Strata/engine`.

### Web page stuck on "Connecting…" (2026-10-29, fixed)
- Cause: `serve/web/app.js` assigns handlers to the web-search drawer controls at startup (`$("s-ws-readmax").oninput = ...`). `index.html` had lost those controls (input, reader select, Jina note, page-limit switch, characters input) in the re-apply of the web-search slice (`a46b6a97`): the label and the JS arrived, the inputs did not. `$()` returned null, startup threw before the first metrics poll, and the header pill stayed on "Connecting…".
- Fix: the markup was restored from the pre-promotion prod `index.html` (same ids). Regression test `serve/test_web_element_ids.py` fails on the broken page (10 ids named) and passes on the fixed one. Run: `python3 serve/test_web_element_ids.py`.
- Verified in a browser on prod: no JS errors. Without a key the pill reads "API key needed" (prod requires a key; enter it under About > Settings). With a key: Idle, and a chat message returned 200 with the expected reply.
