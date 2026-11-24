# DF2 Tree Drafting — Phase 2 Status

Branch: `feat/df2-tree-drafts` (parent Phase-1 commit `99ec266fb`, trunk base `3087c8e57`).
Runtime gate: `STRATA_DF2_TREE=0|1` (`0` / unset by default, host `getenv` cached once via `strata::spec::df2_tree_on()`).

Note on GPU testing: production is live on this host, so **no GPU binary or server was executed**. All host-side logic and telemetry formatting were verified on CPU via `df2_tree_test` and `accept_hist_test`; all `.cu` kernels, `Verifier` tree paths, and serve-loop wiring are **compile- and link-verified only** against HIP `gfx1201`.

## 1. Per-subsystem status

| Subsystem | Status | Scope & Notes |
| --- | --- | --- |
| Env gate & `PrefixFanout` selector builder | **implemented** | `df2_tree_on()`, `df2_tree_build_from_selector()` (`kDf2TreeDefaultMinBranchP = 0.10f`, `max_nodes <= 8`), `df2_tree_caterpillar_gdn_order()`, `df2_tree_ple_conv_taps()`. Host unit-tested in `df2_tree_test`. |
| DF2 runner-up extraction & 7.6 KiB D2H copy | **implemented (compile-verified, GPU untested)** | `Dflash2Drafter` enables `seltab_` under `df2_tree_on()`, lazily allocates `h_tree_tok_` (`7*16*4 B`) + `h_tree_seltab_` (`7*16*16*4 B`) in solo `alloc()`, copies async in `draft_finish()` into those ~7.6 KiB pinned host buffers. |
| Solo greedy serve-loop, longest path, non-contiguous commit | **implemented (compile-verified, GPU untested)** | `generate.cpp`: solo DF2, `o.batch == 0`, `n_stages <= 1`, `!o.draft_auto`, `head_native()`, greedy (`!df2_sample_active`, `temp <= 0`, `hist_n <= 0`, no logit bias/pipe/logprobs/follow), `df2_tree_greedy_accept()`, `commit_tree()`, `compact_taps_tree()`. |
| `[accept-hist]` telemetry extension | **implemented** | `Df2AcceptHist::emit(..., &tree_accept_hist)` stays byte-identical when `tree_rounds == 0`; appends `tree_rounds=... tree_branch_hits=... tree_node_reached=... tree_node_accept=...` when `tree_rounds > 0`. Host unit-tested in `accept_hist_test`. |
| GDN conv + linear-attention caterpillar kernels | **implemented (compile-verified, GPU untested)** | `verify_kernels.cu` + `verify.cpp`: `gdn_conv_l2_tree`, `gdn_conv_ab_tree`, `gdn_step_norm_tree` evaluate leaves read-only against state before spine overwrites (zero checkpoint VRAM), followed by `gdn_commit_tree_layers` on `path[0..n_keep-1]`. Non-caterpillar checkpoints not added. |
| QSA indexer, block selection & KV compaction | **implemented (compile-verified, GPU untested)** | `native_qsa_indexer_append_tree_layers`, `native_qsa_tree_patch_scores` (up to 3 active blocks `b0..n_bid`, cell-0 dead key, YaRN/mrope/RopeTab, F16-rounded pooling), `qsa_tree_remap_ids`, and `kv_compact_tree` (F16, Int8, Q4, Hybrid/K8V4 + flat/chunked host mirror). |
| PLE dilated conv + non-mutating snapshot | **implemented (compile-verified, GPU untested)** | `native_ple_postops_tree_snap` (`conv_residual_tree_kernel` + `history_tree_snap_kernel` read-only on `history`), guarded alongside `!tree_mode` on `multi_write` to avoid double layer-0 residual write. |
| DF2 tap extraction gather | **implemented (compile-verified, GPU untested)** | `df2_gather_taps` gathers `tap_dev_[path[k]]` through `df2.landing()` scratch and copies back to `tap_dev_[0..n_keep-1]` (`compact_taps_tree`, preserving `graph_taps_ == tap_dev_`). |
| Parameterized `T=8` tree verify & commit HIP graphs | **implemented (compile-verified, GPU untested)** | `exec_tree_[2]`, `exec_tree_nr_[2]`, `commit_tree_exec_` lazily captured on first `run_tree` / `commit_tree` when `df2_tree_on()`. Includes `STRATA_DF2_TREE_NOGRAPH=1` debug fallback when `ar_on()`. |
| Sampled tree rejection sampling (SpecInfer) | **stubbed / out of scope** | Requests with sampling fall back to linear DF2 (`tree_req_eligible == false`). |
| `Df2AdaptiveSolo` tree arm / `PrefixFork2Tail` | **deferred** | Not modified in Phase 2 (`generate.cpp` gates tree mode out of `df2_compose_round`, suffix, chain, and `win_forced` rounds). |

## 2. Spec reconciliation notes (`DESIGN.md` & `df2_tree.hpp` over brief)

Where the prompt brief conflicted with `docs/df2-tree/DESIGN.md` and Phase-1 `engine/include/strata/spec/df2_tree.hpp`, Phase 2 followed `DESIGN.md` and `df2_tree.hpp`:

1. **Suffix / Lookup / chain interaction (`DESIGN.md §2.2` vs brief):** Instead of appending Suffix/Lookup/chain rows onto the tree spine tail, `generate.cpp` keeps clean per-round mutual exclusion: when `from_sfx`, `df2_suffix_n > 0`, `chain_n > 0`, `df2_handoff_round`, `merged`, `first_window`, or `win_forced` applies, the round runs the existing linear / `DraftComposition` path unchanged; pure DF2 solo rounds with `tree_cap >= 4` and a non-linear valid caterpillar run tree verify in one narrow group (`G = 1`, padded to `T = 8`).
2. **`[accept-hist]` field names (`DESIGN.md §5.2` & `df2_tree.hpp` vs brief):** Kept `Df2TreeAcceptHist::format_accept_hist_fields()` (`tree_rounds=... tree_branch_hits=... tree_node_reached=... tree_node_accept=...`), matching Phase-1 `parse_accept_hist_fields` rather than renaming to `tree_nodes=... tree_accept=...`. When `tree_rounds == 0`, output matches baseline byte-for-byte.
3. **HIP graph capture (`DESIGN.md §6.1, §6.3` vs brief):** Added lazily-captured parameterized `T=8` verify/commit HIP graphs (`capture_tree()`, `capture_commit_tree()`, `STRATA_DF2_TREE_NOGRAPH=1` uncaptured debug path under `ar_on()`) rather than deferring capture code completely; live GPU validation of captured graphs remains in Phase 3.

## 3. Build summary & compiled modified TUs

Configured inside `/home/soren/scratch/wt-df2-tree-20261124/build-hip-wt` from source `/home/soren/scratch/wt-df2-tree-20261124/engine` matching `/home/soren/scratch/Strata-merge-p01245/build-hip-gfx1201-w8/CMakeCache.txt` (`STRATA_USE_HIP=ON`, `GPU_TARGETS=gfx1201`, `STRATA_WINDOW_MAX_T=8`, `CMAKE_BUILD_TYPE=Release`, `CMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++`), with `-DSTRATA_BUILD_TESTS=ON`.

Build command (memory-capped under `systemd-run`, `-j 2`):

```sh
systemd-run --user --scope -p MemoryMax=6G -- cmake --build build-hip-wt -j 2 --target strata df2_tree_test accept_hist_test
```

Exit code: `0` (`[95/96] Linking HIP executable strata` and test binaries built cleanly; temporary build dir removed before commit).

Compiled modified translation units observed in the build log:
- `CMakeFiles/strata_kernels.dir/src/kernels/cuda/verify_kernels.cu.o`
- `CMakeFiles/strata_kernels.dir/src/kernels/cuda/kv_stream.cu.o`
- `CMakeFiles/strata_kernels.dir/src/kernels/cuda/native_qsa_indexer.cu.o`
- `CMakeFiles/strata_kernels.dir/src/kernels/cuda/qsa_select.cu.o`
- `CMakeFiles/strata_kernels.dir/src/kernels/cuda/native_ple_postops.cu.o`
- `CMakeFiles/strata_kernels.dir/src/kernels/cuda/dflash2.cu.o` (unchanged source, rebuilt against headers)
- `CMakeFiles/strata_engine.dir/src/core/dflash2.cpp.o`
- `CMakeFiles/strata_engine.dir/src/core/verify.cpp.o`
- `CMakeFiles/strata.dir/src/program/generate.cpp.o`
- `CMakeFiles/df2_tree_test.dir/src/spec/df2_tree_test.cpp.o`
- `CMakeFiles/accept_hist_test.dir/src/spec/accept_hist_test.cpp.o`

Host HIP guard check (`git diff HEAD -- 'engine/*.cpp' 'engine/*.hpp' | grep -n '^+.*__HIPCC__'`): zero occurrences added in host files.

## 4. Binary architecture check (`gfx1201`-only)

```text
$ strings -n 6 build-hip-wt/strata | grep -oE 'amdgcn-amd-amdhsa--gfx[0-9a-f]+' | sort | uniq -c
     61 amdgcn-amd-amdhsa--gfx1201
```

## 5. Host unit test outputs

```text
$ ./build-hip-wt/df2_tree_test
df2_tree_test: OK
```

```text
$ ./build-hip-wt/accept_hist_test
(exited with code 0)
```

## 6. Programmatic gate-off audit (`git diff HEAD --unified=0`)

Scope vs Phase-1 `HEAD` (`99ec266fb`): **19 files changed (+2060 / -75), 81 hunks**. (Cumulative vs trunk `3087c8e57`: 21 files including `docs/df2-tree/DESIGN.md`, `engine/CMakeLists.txt`, `engine/include/strata/spec/df2_tree.hpp`, `engine/src/spec/df2_tree_test.cpp`, plus this `PHASE2-STATUS.md`.)

Every hunk falls into one of two categories: (a) new tree-only declarations/functions/tests unreachable unless called under `df2_tree_on()`, or (b) guarded branches inside existing paths that evaluate to no-op / original expressions when `STRATA_DF2_TREE=0`:

| File | Hunks | Hunk lines | Guard when `STRATA_DF2_TREE=0` |
| --- | --- | --- | --- |
| `engine/include/strata/core/dflash2.hpp` | 2 | L189, L340 | New accessors + nullptr-init fields; unreferenced when off |
| `engine/include/strata/core/verify.hpp` | 3 | L33, L246, L736 | New declarations + null-init tree state (`tree_rec_=false`, `last_tree_=false`) |
| `engine/include/strata/kernels/kv_stream.hpp` | 1 | L213 | New kernel decl `kv_compact_tree` |
| `engine/include/strata/kernels/native_ple_postops.hpp` | 1 | L45 | New kernel decl `native_ple_postops_tree_snap` |
| `engine/include/strata/kernels/native_qsa_indexer.hpp` | 2 | L7, L89 | New kernel decls `native_qsa_tree_patch_scores`, `native_qsa_indexer_append_tree_layers` |
| `engine/include/strata/kernels/qsa_select.hpp` | 2 | L27, L145 | New kernel decl `qsa_tree_remap_ids` |
| `engine/include/strata/kernels/verify_kernels.hpp` | 1 | L275 | `VerifyTreeMeta` + new tree kernel decls |
| `engine/include/strata/spec/accept_hist.hpp` | 4 | L3, L47, L55, L58 | `emit(..., tree=nullptr)` prints identically unless `tree != nullptr && tree->tree_rounds > 0` |
| `engine/include/strata/spec/df2_tree.hpp` | 9 | L11, L23, L267, L271, L282, L296, L322, L335, L626 | Header-only helpers (`df2_tree_on()`, builders, GDN/PLE helpers); inert when off |
| `engine/src/core/dflash2.cpp` | 9 | L4, L73, L172, L638, L1411, L2088, L2103, L2127, L2130 | Every hook checks `df2_tree_on()` (or `tree_copy`, which requires `df2_tree_on()`) |
| `engine/src/core/verify.cpp` | 19 | L836, L1689, L1694, L1697, L1713, L1737, L1820, L1846, L1968, L1970, L2136, L2239, L2339, L2647, L2905, L4936, L4971, L5817, L7037 | `tree_mode = df2_tree_on() && tree_rec_` is `false` in L1689–L2905 (`multi_write` also checks `!tree_mode`); L836/L4936/L4971/L5817 guard on `df2_tree_on()`; L7037+ methods require `df2_tree_on()` |
| `engine/src/kernels/cuda/kv_stream.cu` | 1 | L477 | New `kv_compact_tree` kernel only |
| `engine/src/kernels/cuda/native_ple_postops.cu` | 2 | L212, L338 | New `conv_residual_tree_kernel` / `history_tree_snap_kernel` / `native_ple_postops_tree_snap` |
| `engine/src/kernels/cuda/native_qsa_indexer.cu` | 2 | L25, L425 | New `native_qsa_tree_patch_scores` / `native_qsa_indexer_append_tree_layers` |
| `engine/src/kernels/cuda/qsa_select.cu` | 2 | L10, L4196 | New `qsa_tree_remap_ids` |
| `engine/src/kernels/cuda/verify_kernels.cu` | 1 | L2274 | New tree kernels (`copy_tree_step_pos_from_mapped`, `gdn_*_tree`, `gdn_commit_tree_layers`, `df2_gather_taps`) |
| `engine/src/program/generate.cpp` | 17 | L158, L23489, L24631, L24654, L24657, L24906, L24910, L24952, L24963, L24968, L24970, L24973, L25066, L25068, L25124, L25140, L25900 | `tree_req_eligible` checks `df2_tree_on()`; `tree_round` stays `false` when off (all branch/commit/tap/telemetry hooks are `false`/baseline) |
| `engine/src/spec/accept_hist_test.cpp` | 1 | L40 | Host unit test additions |
| `engine/src/spec/df2_tree_test.cpp` | 2 | L253, L371 | Host unit test additions |

## 7. Deferred items & Phase-3 launch plan (10 lines)

Deferred to Phase 3: live GPU functional/parity/perf validation, captured tree HIP graph timing & verification on hardware, `Df2AdaptiveSolo` tree arm, sampled (SpecInfer) tree verification, `min_branch_p` empirical tuning, and non-caterpillar (`PrefixFork2Tail` + 2 MiB GDN checkpoint) evaluation if data warrants it.

**Phase-3 Launch Plan:**
1. Deploy Phase-2 build in an offline GPU maintenance window; confirm `STRATA_DF2_TREE=0` gives zero behavioral/latency regression and unchanged `[accept-hist]` lines.
2. Force degenerate single-spine tree (`STRATA_DF2_TREE=1` with temporary high threshold) and verify bit-exact logits/tokens vs `STRATA_DF2_TREE=0` across F16/Int8/Q4/K8V4 KV, QSA, GDN, PLE, and DF2 taps.
3. Run greedy determinism suite under `STRATA_DF2_TREE=1 STRATA_DF2_TREE_NOGRAPH=1` on Qwen3.8-Flash-Next and Qwen3.6-35B-A3B, asserting 100% token identity against plain sequential (`--draft none`) greedy decode.
4. Re-enable captured tree graphs (`STRATA_DF2_TREE_NOGRAPH=0`), re-run deterministic parity suite, and verify zero HIP graph errors or non-finite activations across 1k+ rounds.
5. Collect `[accept-hist]` before/after (`STRATA_DF2_ACCEPT_HIST=1 STRATA_DF2_TREE=0` vs `=1`) across code, math, and chat prompts.
6. Watch `tree_branch_hits / tree_rounds` (target >10% branch save rate) and `tree_node_accept / tree_node_reached` per fan-out node `0..7`.
7. Track net request `mean` accepted tokens (`accepted/round`) vs any T=8 tree-verify latency delta so net decode `tok/s` improves over linear DF2.
8. Sweep `min_branch_p` (`0.05..0.20`) and fan-out width (`2` vs `3`) offline/online using `STRATA_DF2_CAND_DUMP` traces.
9. Validate `STRATA_DF2_TREE=1` gracefully falls back to linear on batched requests, `--draft auto`, sampled decoding, and Suffix/Lookup composition rounds.
10. Gate `Df2AdaptiveSolo` tree policy and SpecInfer sampled tree rejection on positive net `tok/s` from steps 5–8.
