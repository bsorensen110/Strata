# DFlash2 (DF2) Tree-Shaped Speculative Drafting — Solo Path Design (gfx1201 / HIP)

**Status:** Phase 1 Design (Inert; no runtime wiring yet)  
**Target:** `feat/df2-tree-drafts` (base `3087c8e57`), Qwen3.8-Flash-Next solo serve/CLI path on gfx1201 (HIP)  
**Scope:** Greedy tree drafting & verification in Phase 2 (`STRATA_DF2_TREE=0/1`, default off); probabilistic SpecInfer tree rejection sampling explicitly deferred out of Phase 1/2.

---

## 0. Executive Summary & Architectural Context

DFlash2 (`engine/include/strata/core/dflash2.hpp`, `engine/src/core/dflash2.cpp`) is a 5-layer block drafter with hard compile-time block dimensions `Dflash2Drafter::kRows = 8` (1 committed anchor token at row `0`, plus `kDrafts = 7` mask rows at rows `1..7`). On Qwen3.8-Flash-Next with the default `STRATA_DF2_HEAD=native` (`Dflash2Drafter::head_native()`, `dflash2.hpp:71-75`, ignoring `draft_vocab.bin`), each draft round computes full-vocabulary logits for the active draft rows, reduces top-16 candidates per row (`df2_topk`), projects `hsel_` (`rank = 256`), materializes the full `[7][16][16]` order-1 bilinear transition table `seltab_` via `select_table_kernel` (`engine/src/kernels/cuda/dflash2.cu:2407`), and runs a `beam = 4` Viterbi walk (`df2_beam()`, `dflash2.cpp:52-59`; `select_walk_beam_table_kernel<4>`, `dflash2.cu:2594`).

### Key Structural Advantage Over `STRATA_MTP_TOP2`
In the MTP drafter (`engine/src/core/mtp.cpp:699-722`), probing a runner-up (`STRATA_MTP_TOP2=1`) requires copying full-vocabulary `head_logits_` to the host per step because MTP only emits a single greedy argmax token per draft step. By contrast, DF2 **already materializes** `toptok_[7][16]` (`448 B`) and `seltab_[7][16][16]` (`7,168 B`) in device memory on every round—the exact buffers already copied to host by `df2_sample_selector_walk` (`dflash2.cpp:2108-2124`) and `STRATA_DF2_CAND_DUMP` (`dflash2.cpp:2075-2087`). Consequently:
- Extracting runner-up tokens (`rank 1`, `rank 2`, ...) at any cut depth `c`, **and**
- Conditionally selecting next-step continuations conditioned on a runner-up candidate slot `j2` via `seltab_[c+1][j2][*]`

costs **~7.6 KiB D2H** and **zero extra drafter backbone layers or LM-head GEMMs**.

### Key Difference from Prior Art (`hy4flash`)
Prior tree machinery in this repository (`engine/src/hy4flash/run/hy4_dfly2.hip.h:1121`, `engine/src/hy4flash/metal/hy4_metal_spec.h:421,770`, `engine/src/hy4flash/run/mla_attn.hip.h:44-111`, `engine/src/hy4flash/run/hy4_engram.h:49-51`) targets Hy4-Flash, which has only MLA attention plus optional Engram modules. Qwen3.8-Flash-Next verification (`engine/src/core/verify.cpp`) has **four** cross-token state subsystems that must all become tree-ancestor-aware:
1. **36 GDN layers**: causal 4-tap Conv1D (`gdn_conv_l2_multi` / `gdn_conv_ab_multi`) + linear-attention delta-rule recurrence (`gdn_step_norm_multi`).
2. **12 QSA layers**: block-4 pooled indexer (`native_qsa_indexer_append_steps`, `qsa_block_scores`, `qsa_block_topk`) + KV cache append (`kv_append_*`) + sparse decode attention (`qsa_decode_attn_batch`).
3. **Layer-1 PLE**: host n-gram hashing (`ngram_rows`) + device dilated Conv1D history (`NG_HIST = 9`, `NG_HC_DIM = 10240`, `native_ple_postops_batch_snap` / `ple_block_projected`).
4. **DF2 tap extraction**: `tap_dev_` rows gathered by `Verifier::gather_taps` (`verify.hpp:442`) for the next draft round.

---

## 1. Tree Shapes Fitting 8 Verify Rows & Confidence-Cut Branching

### 1.1 Hard 8-Row Budget (`M <= 8`)
A verify window always places the anchor (last committed token) at node `0` (logical position `pos0`, depth `0`) and up to `7` drafted tokens at nodes `1..7` (`M <= kVerifyNarrowT = 8`). Because DF2 produces at most `kDrafts = 7` rows (`dflash2.hpp:40-41`), any pure-DF2 tree with `M <= 8` fits natively in one narrow verify window.

### 1.2 Why `--draft-conf` Cut Depth Is the Natural Branch Point
Today, `choose_window` (`engine/include/strata/spec/window_rule.hpp:70-74`) and the CLI/serve loops (`engine/src/program/generate.cpp:27053`, `24464`) truncate DF2's linear chain at the first draft index `c` (`0 <= c < kDrafts`) where the cumulative product of selector probabilities falls below `--draft-conf P`:

$$\prod_{i=0}^{c} p_i < P.$$

- **Before `c` (`0 <= i < c`)**: cumulative path survival is high ($\ge P$). Allocating sibling rows here wastes verify slots on low-probability alternatives.
- **At `c`**: primary confidence drops (either `dprob[c]` itself is low, e.g. bimodal token choice, or cumulative survival crosses `P`). Linear drafting with `--draft-conf` simply stops at `T = c + 1` (leaving `8 - (c + 1)` verify rows idle) or forces extra low-confidence linear tokens whose expected survival drops exponentially.
- **Branching at `c`**: spending 1 or 2 unused rows on the runner-up(s) at `c` (from `toptok_[c]` / `seltab_[c][prev_slot]`) converts an uncertain single branch into parallel verification of the top modes while keeping every node within `M <= 8`.

If `--draft-conf` is `0` (disabled) or `c >= 6` (only 1 draft slot left), a tree either picks the lowest-probability position `argmin_i dprob[i]` (for `Caterpillar`) or stays linear (`T = 8`).

### 1.3 Candidate Topologies (`M <= 8`)

| Shape | `Df2TreeShape` | Example Layout (`M = 8`) | Max Depth | GDN Checkpoint Need | Primary Use Case |
|---|---|---|---|---|---|
| **A1. Prefix + Fan-out (no tail)** | `PrefixFanout` | `p=4` prefix + `F=3` fan-out (`1+4+3=8`), or `p=5` + `F=2` (`1+5+2=8`) | `p+1` | **Zero** (caterpillar) | High `--draft-conf` cut at `c=4` or `c=5` |
| **A2. Prefix + Fan-out + Primary Tail** | `PrefixFanout` | `p=3` prefix + `F=2` at `c=3` + `2` primary tail (`1+3+2+2=8`), or `p=2` + `F=2` + `3` tail | `6` | **Zero** (caterpillar) | Early cut `c=2..3` where primary path still merits speculation |
| **B. Prefix + 2-Way Fork w/ Both Tails** | `PrefixFork2Tail` | `p=3` prefix (`0->1->2->3`), `3->4->6` (primary), `3->5->7` (runner-up) (`1+3+2+2=8`) | `5` | **1 checkpoint (2 MiB)** | Cut `c=3` when runner-up's conditioned continuation `seltab_[4][j2][*]` is also strong |
| **C. Multi-Point Caterpillar** | `Caterpillar` | Spine of 5 (`0->1->2->3->4->5`) + runner-up leaves at two lowest-conf spine nodes | `5` | **Zero** (caterpillar) | `--draft-conf 0` or diffused uncertainty across 2 positions |

### 1.4 The Caterpillar GDN Property & Phase-2 Recommendation
Notice that **Shapes A1, A2, and C are all caterpillar trees**: at every depth, at most one node has children (the primary spine), and every runner-up branch is a **leaf**. In a caterpillar tree, `gdn_step_norm_tree` can evaluate every leaf sibling at depth `d` **read-only against the current register state `s[r]`** (before overwriting `s[r]` with the primary spine node's recurrence step) — requiring **0 bytes of GDN state checkpoint VRAM** and **0 extra VGPRs** for state backup on gfx1201.

**Recommended Phase-2 Default Shape:** `PrefixFanout` (caterpillar family: linear prefix of length `p = cut_depth`, `F = 2` or `3` fan-out at `p` from `seltab_[p][prev_slot]` when runner-up conditional probability `p2 >= min_branch_p`, and any remaining nodes up to `M <= 8` allocated to the primary spine tail). `PrefixFork2Tail` (`3 + 2x2`) is supported by the scaffolding and queued for Phase-3 A/B once the 2 MiB global-memory GDN checkpoint path is validated.

---

## 2. Narrow Build Two-Group `<= 8` Verify & `DraftComposition` Interaction

### 2.1 Why Trees Stay Strictly Within One `<= 8` Group (`G = 1`)
In this production narrow build (`engine/include/strata/kernels/window_limits.hpp:21-25`, `verify_kernels.hpp:25-31`):
- `kVerifyNarrowT = 8`, `kVerifyMaxT = 8`, `kVerifyLongT = 16`.
- `Verifier::record_window` (`engine/src/core/verify.cpp:1684`) sets:
  ```cpp
  const int G = batch_rec_ ? ... : (((split_ && T >= 2) || T > kVerifyMaxT) ? 2 : 1);
  ```
- When `G = 2`, the window splits tokens into `[0, half_b)` and `[half_b, T)` (`verify.cpp:1719-1720`). Crucially, in `G = 2` mode, group 1's GDN recurrence (`gdn_step_norm_multi`, `verify.cpp:2294`) **re-runs tokens `0..te-1` linearly from `state`** (`t_out_begin = tb`) while reading `qkv`/`hb` sequentially! If a tree were split across `G = 2`, group 1 would either pay redundant tree-traversal state replays across all 36 GDN layers or force `gdn_commit_layers` (`verify.cpp:4036`) for `MT > 8`.
- Furthermore, all high-efficiency kernels (`fused_gr_read_multi`, `native_mmvq_il`, `qsa_decode_attn_batch`, `gdn_step_norm_multi`) are compiled and tuned for `<= kVerifyNarrowT = 8` columns.

**Design Decision:** Every DF2 tree window satisfies `M <= 8` and executes with **`G = 1` (single group)** in `Verifier::record_window`.

### 2.2 Interaction with Suffix / Lookup & `DraftComposition::append`
In `generate.cpp` (`24526-24608` serve, `27153-27200` CLI), linear rounds compose tokens via `DraftComposition` (`engine/include/strata/spec/draft_composition.hpp:10-38`) in strict non-decreasing rank order:

`DraftCompositionSource::Dflash2` (0) $\rightarrow$ `Suffix` (1) $\rightarrow$ `Lookup` (2).

`Suffix` (`sfx.propose_after`) and `Lookup` (`propose_from_sources` / `Df2AdaptiveSolo` tiers `9..16`) are inherently **linear** continuations (and often extend `T` into `9..16` across two `<= 8` groups). Mixing a branched `Dflash2` tree with a linear `Suffix`/`Lookup` tail in `9..16` would break `G = 1` and create ambiguous attachment semantics if Suffix only extends the primary branch.

**Phase-2 Policy (Clean Mutual Exclusion per Round):**
1. Run DF2 draft + check `df2_suffix_fill_enabled()` / `--lookup-chain` / `Df2AdaptiveSolo` **before** committing to tree mode.
2. If a valid `Suffix` fill (`sfx.last_match() >= df2_suffix_min_match()`) or `Lookup` chain (`chain_n > 0`) extends the round, execute the existing linear `DraftComposition` path (`Dflash2 -> Suffix -> Lookup`, up to `T <= 16`) **unchanged**.
3. Otherwise (no Suffix/Lookup extension), if `STRATA_DF2_TREE=1` and the request is greedy and DF2's `--draft-conf` cut leaves room (or a runner-up crosses the branch threshold), build a `Df2TreeSpec` (`M <= 8`, pure `DraftCompositionSource::Dflash2`) and dispatch the `G = 1` tree verify graph.

*(Phase-3 option)*: If Suffix/Lookup later wants to coexist inside `M <= 8`, it can append `Suffix` / `Lookup` nodes linearly off the **primary spine leaf** (`parent = spine_leaf`) while keeping total `M <= 8` and `DraftComposition` rank `Dflash2 -> Suffix -> Lookup` intact.

---

## 3. Attention Masking & Required Kernel Work on HIP (`gfx1201`)

### 3.1 Comparison: `hy4flash` Tree-8 vs Current DF2 Verification

| Mechanism | `hy4flash` Prior Art (`mla_attn.hip.h`, `hy4_graph.hip.h`, `hy4_engram.h`) | Current DF2 Linear Verify (`verify.cpp`, `verify_kernels.hpp`) | Required DF2 Tree Support (HIP / gfx1201) |
|---|---|---|---|
| **Positions & RoPE** | `TreePos` (`depth[8]`), `mla_prep_q_dec_d` / `mla_write_kv_d` write KV to physical `p0 + t` while applying RoPE at logical `p0 + depth[t]` | `copy_step_pos_from_mapped` (`verify_kernels.hpp:115`) broadcasts `step[t].pos = pos0 + t` to `pos_`, `pos_k`, `pos_i` | Add `copy_tree_step_pos_from_mapped`: write logical `pos0 + depth[t]` to `pos_`, `pos_k`, `pos_i` while keeping physical `pos0 + t` in `step_[t].pos` |
| **Main Attention** | `MlaAttnArgs::TreeDev` (`tree_depth[8]`, `tree_anc[8][8]`); `mla_attend` remaps keys `>= p0` via `tree_anc[t][d]` | `qsa_decode_attn_batch` (`qsa_decode_attn.cu:196,218`) attends **only** explicit cell list `ids[t * cap + 0..width-1]` (no internal causal mask) | **Attention kernel itself is unchanged!** Tree masking happens in selection (`ids` remapping) |
| **QSA Indexer & Top-K** | `IdxTree` (`hy4_graph.hip.h:134-142`) | `native_qsa_indexer_append_steps` (`native_qsa_indexer.cu:50`) pools blocks of `R=4`; `qsa_block_scores` / `qsa_block_topk` (`qsa_select.cu:114,158`) | Path-exact indexer block scoring & cell remapping (`qsa_block_scores_tree` + `qsa_block_topk_tree` or `qsa_tree_remap_ids`) |
| **Conv1D + Recurrence** | None (MLA-only model) | `gdn_conv_l2_multi` / `gdn_conv_ab_multi` + `gdn_step_norm_multi` (`verify_kernels.cu:60-101,371-483`) | `gdn_conv_l2_tree` / `gdn_conv_ab_tree` (via `anc3[8][3]`) + `gdn_step_norm_tree` |
| **N-gram / Conv1D** | `engram_hash_tree` (`hy4_engram.h:51`) | Host `ngram_rows` (`verify.cpp:4381`) + `native_ple_postops_batch_snap` / `ple_block_projected` (`verify.cpp:1965-1992`) | Host parent-chain `ngram_rows` + device `ple_postops_tree_snap` reading `hist_snap_[parent[t]]` |

### 3.2 Detailed HIP Kernel Inventory & Design

1. **Mapped Tree Descriptor (`TreeMeta`)**  
   To avoid per-shape graph explosion (Section 6), add a small device-copied struct `TreeMeta` (staged in mapped host memory and copied at graph entry via `copy_i32_from_mapped`):
   - `int32_t M` (`2..8`), `int32_t pos0`
   - `int8_t parent[8]`, `uint8_t depth[8]`, `uint8_t is_spine[8]`
   - `uint8_t anc[8][8]` (`anc[t][d]` = physical row index of node `t`'s ancestor at relative depth `d <= depth[t]`)
   - `int8_t anc3[8][3]` (the 3 preceding taps `[p3, p2, p1]` for GDN Conv1D; negative `-1..-3` index `hist[2], hist[1], hist[0]`, `>= 0` index `qkv[idx]`)

2. **QSA Indexer & Sparse Selection (`qsa_select.cu` / `native_qsa_indexer.cu`) — Critical Subtlety**  
   In QSA, `R = 4` consecutive logical positions form one pooled block `b = pos / 4` (`native_qsa_indexer.cu:65-115`), and `qsa_block_topk` returns cell indices `b * 4 + c` (`qsa_select.cu:235-238`).
   - Naive `native_qsa_indexer_append_steps` across `t = 0..M-1` fails on a tree for two reasons: (a) sibling nodes at the same depth would overwrite or pollute the same 4-cell block, and (b) `n_kv = pos0 + t + 1` in `step_[t]` would miscount logical sequence length `pos0 + depth[t] + 1`.
   - **Exact Tree Solution (2 small kernels, zero changes to `qsa_decode_attn_batch`):**
     - **Step A (`step_` setup):** Set `step_[t].n_kv = pos0 + depth[t] + 1`, `n_bid = n_kv / 4`, `width = qsa_selection_width(n_kv, s)` for selection, while `kv_append_*` receives physical cell `pos0 + t` via `kv_pos[t] = pos0 + t` (from `TreeMeta`).
     - **Step B (Committed + Tree Block Scoring):** Do **not** mutate `st.idx_pooled` during the window; instead, read `tail_snap_` (snapshot of `pos0`'s incomplete block `[pos0 - (pos0 % 4), pos0)`) and `idx_raw[0..M-1]` (already computed for every tree row by `idx_key` in `verify.cpp:2381`). For each node `t`:
       - Blocks `0 .. (pos0 / 4) - 1` are **committed** and already reside in `st.idx_pooled`: score them with `qsa_block_scores` as usual (`n_bid_committed = pos0 / 4`).
       - The at most **2 active blocks** covering `[pos0 - (pos0 % 4), pos0 + depth[t]]` (specifically `b_start = pos0 / 4` and, if `(pos0 % 4) + depth[t] >= 4`, `b_start + 1`) depend only on `tail_snap_` and `idx_raw[anc[t][0..depth[t]]]`. A lightweight kernel `qsa_tree_patch_scores` (1 block per node `t`, 128 threads, running right after `qsa_block_scores`) computes the exact RMSNorm + RoPE pooled score of those 1–2 blocks for node `t` and writes `scores_[t * max_blocks_ + b]` (and handles the incomplete tail `+1e9f` when `n_kv % 4 != 0`).
       - `qsa_block_topk` then runs **unchanged** on `scores_` using `n_kv = pos0 + depth[t] + 1`, selecting logical cell IDs in `[0, pos0 + depth[t]]`.
       - A tiny 1-launch kernel `qsa_tree_remap_ids` (`M` blocks, `cap_` threads, only touching trailing entries `c >= pos0` in `sel_[t]`, which are sorted ascending!) remaps logical cell `pos0 + d` $\rightarrow$ physical cell `pos0 + anc[t][d]`.
       - `qsa_kv_resolve` and `qsa_decode_attn_batch` then execute **100% unmodified**!

3. **GDN Conv1D (`verify_kernels.cu:60-101`, `verify_kernels.hpp:36-45`)**  
   Add `gdn_conv_l2_tree` and HIP-fused `gdn_conv_ab_tree` (`verify_kernels.hpp`): identical arithmetic to `gdn_conv_l2_body` (`verify_kernels.cu:60-85`), replacing `src = t + j` with `anc3[t][j]` (`anc3[t][j] < 0 ? hist[c*3 + (3 + anc3[t][j])] : qkv[anc3[t][j]*C + c]`).

4. **GDN Recurrence (`verify_kernels.cu:371-483`)**  
   Add `gdn_step_norm_tree`:
   - **Caterpillar Fast Path (Phase 2 default):** `TreeMeta` marks spine nodes vs leaf branches. Iterate depth-by-depth (or topological node order with spine-last at each depth):
     - For any **leaf sibling** at depth `d` whose parent is the current spine state in registers `s[r]`, compute `kv_col`, `delta`, temporary `s_tmp = fmaf(g, s[r], k * delta)` (without mutating `s[r]`!), `o`, RMSNorm, and SiLU gate $\rightarrow$ write `y[t]`.
     - Then step the **spine node** at depth `d`, updating `s[r]` in-place and writing `y[t_spine]`.
     - **Zero** extra global memory, **zero** extra VGPR state arrays, bitwise identical per-path arithmetic to `gdn_step_norm_multi_kernel`.
   - **General `M <= 8` Path (Phase 2 optional / Phase 3 `PrefixFork2Tail`):** When a node has $\ge 2$ non-leaf children (at most 1 fork in `M <= 8` with tails), save `s[r]` once into a reusable 2 MiB scratch buffer (`gdn_fork_state_`, `32 heads * 128 * 128 * 4 B = 2 MiB`, shared across all 36 GDN layers) at the fork node, execute arm 1, restore `s[r]` from `gdn_fork_state_`, and execute arm 2.

5. **Layer-1 PLE (`verify.cpp:4378-4386`, `1965-1992`, `native_ple_postops.hpp`)**  
   - **Host hashing (`Verifier::run`):** Instead of shifting a single `prev[2]` linearly across `t = 0..T-1`, compute `prev[2]` for node `t` from `parent[t]` (matching `hy4_engram.h:51` `engram_hash_tree`):
     - `p1 = parent[t] < 0 ? ss.ple_prev[1] : tokens[parent[t]]`
     - `p2 = parent[t] < 0 ? ss.ple_prev[0] : (parent[parent[t]] < 0 ? ss.ple_prev[1] : tokens[parent[parent[t]]])`
   - **Device dilated Conv1D (`native_ple_postops_tree_snap`):** Loop `t = 0..M-1` in topological order; read history from `t == 0 ? ss.ple.hist : hist_snap_ + parent[t] * HS` (without mutating `ss.ple.hist` in-place during verify) and write the shifted history to `hist_snap_ + t * HS`.

---

## 4. Acceptance, Bit-Losslessness, Sampling, and Non-Contiguous Commit

### 4.1 Longest-Accepted-Path Rule
Given `Df2TreeSpec` in topological order (`parent[i] < i`, root `0` = anchor at `pos0`) and verifier greedy picks `verifier_argmax[0..M-1]`:
1. Initialize `path = [0]`, `u = 0`.
2. Loop: scan children `c` of `u` in ascending node index (`c > u`, `parent[c] == u`). If `token[c] == verifier_argmax[u]`, append `c` to `path`, set `u = c`, and continue.
3. Stop when no child of `u` matches `verifier_argmax[u]`.
4. Set `n_keep = path.size()` (`1 <= n_keep <= M`) and `bonus_token = verifier_argmax[u]`.

For general predicate-driven walks (`df2_tree_longest_accepted_path(tree, node_accepted)`), a node `i > 0` is **reachable-accepted** if and only if `node_accepted[i] != 0` **and** its parent `parent[i]` is reachable-accepted (**strict branch-kill**: rejecting an interior node kills its entire subtree). Ties in accepted depth break in favor of the lowest node index (primary branch first).

### 4.2 Why Greedy Acceptance Stays Bit-Lossless
Plain 1-token greedy decode generates tokens $y_0, y_1, \dots$ sequentially where $y_k = \arg\max \text{logits}(x_{\le p_0 + k})$. Tree verification preserves **bit-exact** equivalence to 1-token greedy decode via four invariants:
1. **Ancestor-only forward semantics:** Every node `t` in the tree receives identical inputs to running 1-token decode along `root_to_node(t)`: RoPE position `pos0 + depth[t]`, QSA keys strictly from `< pos0` plus `anc[t][0..depth[t]]`, path-exact QSA indexer block pooling, GDN Conv1D over `anc3[t]`, GDN recurrence from `S_{parent[t]}`, and PLE n-grams/history over `anc[t]`.
2. **Row-wise arithmetic parity:** Embeddings, hyper-connections, dense MMVQs, MoE router/experts, and LM head are row-independent (note: CPU routed experts under `pool` keep the exact same multi-token rounding policy as linear `verify.hpp:5-8` / `STRATA_IQ_MT_MIN`).
3. **Unique child token per parent:** Builders enforce `has_unique_siblings()` (no two children of the same parent `u` share a token ID). Hence at most one child of `u` can equal `verifier_argmax[u]`.
4. **Path-only state commit:** All rejected nodes (both siblings and descendants beyond the cut) never enter the committed state; only `path[0..n_keep-1]` is compacted/replayed into positions `pos0 .. pos0 + n_keep - 1`. Thus `verifier_argmax[path[k]]` equals $y_k$ bit-for-bit, and post-commit state equals 1-token decode after $k = n_{\text{keep}}$ steps.

### 4.3 Sampled-Token Handling (Out of Scope for Phase 1 & Phase 2)
Exact probabilistic tree verification (SpecInfer, as implemented in `hy4_dfly2.hip.h:1203-1219` and `hy4_metal_spec.h:850+`) requires:
1. Drawing child 1 from $q_1$ (`seltab_`) and sibling 2 **without replacement** from the residual $q_2(x) = q_1(x) / (1 - q_1(y_1))$,
2. Running path-specific rejection sampling ($u < p(y) / q(y)$, residual renormalization $(p - q)_+$ on rejection) sequentially down the chosen tree path on host/device,
3. Constructing per-node penalty histories (`penalty_rows` in `generate.cpp:24797`) along each tree branch rather than linearly over `window[0..T-1]`.

**Decision:** Probabilistic tree sampling is **explicitly out of scope for Phase 1 and Phase 2**. In Phase 2, any non-greedy request (`!sampling_.greedy && sampling_.temperature > 0`) or active `STRATA_DF2_SAMPLE_WALK` automatically stays on the linear DF2 path.

### 4.4 Non-Contiguous Commit & Compaction (`path[k] != k`)
When a runner-up branch is accepted (e.g., `path = [0, 1, 2, 4, 6]`, skipping rejected node `3` and node `5`), `n_keep = 5` accepted tokens occupy non-contiguous physical window rows. Because the tree is in topological order (`parent[i] < i`), `path[k] >= k` holds for every `k = 0..n_keep-1` (the same invariant documented in `mla_attn.hip.h:108-110`). Commit handles all 5 subsystems as follows:

1. **GDN Commit (36 layers):** Extend `h_commit_` (`verify.cpp:5071-5073`) with `path[0..n_keep-1]` (8 ints at `h_commit_[2 + MT ..]`). Add `gdn_commit_tree_layers` (`gdn_conv_commit_tree_layers` + `gdn_step_norm_tree_commit`) reading `qkv[path[k]]`, `h[path[k]]`, `gate[path[k]]`, `beta[path[k]]` in ascending `k = 0..n_keep-1` to update conv history and recurrent state.
2. **QSA KV Cache Compaction (12 layers):** Add `kv_compact_tree` supporting F16, Q8, Q4, and K8V4 (plus streamed host mirror if `st.host.present()`): for `k = 0..n_keep-1`, if `path[k] != k`, copy cell `pos0 + path[k]` $\rightarrow$ `pos0 + k`. Because `path[k] >= k` and `k` ascends, in-place sequential compaction per block is race-free (identical invariant to `mla_compact_kv`). Cells `>= pos0 + n_keep` are stale and overwritten on the next window before any query reads them.
3. **QSA Indexer Commit (12 layers):** Restore `st.idx_tail` from `tail_snap_` and append `idx_raw[path[k]]` at logical position `pos0 + k` for `k = 0..n_keep-1` (via `native_qsa_indexer_append_steps_layers` with an optional `row_idx` map `path[k]`).
4. **PLE Commit (Layer 1):** Device `copy_indexed(ss.ple.hist, hist_snap_, HS, commit_ + 1, HS, cs_)` (`verify.cpp:4071`) already indexes an arbitrary row! Setting `h_commit_[1] = path[n_keep - 1]` commits the exact accepted leaf history with **zero new PLE commit kernels**. Host `ss_->ple_prev` advances through `last_tokens_[path[0..n_keep-1]]`.
5. **DF2 Taps Compaction:** When `df2_tree_commit` has `path[k] != k`, copy `tap_dev_[path[k]]` $\rightarrow$ `df2.landing()[k]` for `k = 0..n_keep-1` (at most `8 * 5 * 2560 * 4 B = 400 KiB` via a 1-launch `df2_gather_taps` kernel on `df2.stream()`). The DF2 drafter's own `context()` / `draft()` graphs remain **100% untouched**.

---

## 5. Policy (`Df2AdaptiveSolo`) & Telemetry (`[accept-hist]`)

### 5.1 Policy Integration (`engine/include/strata/spec/df2_adaptive_solo.hpp`)
`Df2AdaptiveSolo` currently updates `df2_ok_` / `df2_seen_` and `lookup_ok_` / `lookup_seen_` (0.95 EMA) only on `df2_full_prefix` linear rounds, selecting `tier_` in `[base_width_, 16]` that maximizes `(1 + expected_accepted(w)) / costs[w]` with hysteresis (`tier_delta >= 0.10 * tier_`, `rounds_since_switch_ >= 8`, `desired_score >= current_score * 0.90`).

**Decision:** Do **not** pollute `df2_ok_` / `lookup_ok_` with tree-round observations (whose node indices do not correspond to linear positions). Instead, treat tree mode as a **separate policy arm**:
- Keep linear DF2 + adaptive lookup (`tier_ = base_width_ .. 16`) tracked via `df2_ok_` / `lookup_ok_`.
- Track tree rounds with a parallel 0.95 EMA of `(tree_accepted_ok, tree_rounds_seen)` and measured tree verify cost `tree_cost_ms` (vs linear `costs[tier_]`).
- Select the tree arm when `(1 + tree_expected_accepted) / tree_cost_ms` beats the active linear tier by the same 10% hysteresis / 8-round dwell guard (and no Suffix/Lookup match fired).

### 5.2 `[accept-hist]` Telemetry Extension (`engine/include/strata/spec/accept_hist.hpp`)
Current format (`accept_hist.hpp:47-55`, tested by `accept_hist_test.cpp:36-38`):
```text
[accept-hist] req=9 df2_pos_reached=3,3,1,0,0,0,0 df2_pos_accept=3,2,1,0,0,0,0 lookup_rows=0 lookup_accept=0 chain_rows=1 chain_accept=0
```
**Compatibility Rule:**
- When `tree_rounds == 0` (the default / linear path), emit the **exact** existing line byte-for-byte.
- When `tree_rounds > 0`, append space-separated `key=value` fields with comma-separated 8-element integer lists (preserving existing comma-list parsers):
  ```text
  ... tree_rounds=4 tree_branch_hits=1 tree_node_reached=4,4,4,3,3,2,0,0 tree_node_accept=4,4,3,2,1,0,0,0
  ```
- **Reachability Semantics (`Df2TreeAcceptHist::observe`):** Node `0` (root) is always reached if `count > 0`; node `i > 0` is counted in `node_reached[i]` if and only if its parent `parent[i]` was accepted on the winning path (or `i == 0` for node `0`'s direct children). Thus all siblings at a reached branch point count as reached, while nodes behind a rejected ancestor do **not** count as reached (matching the conditional-probability invariant in `accept_hist.hpp:31-32`).

---

## 6. HIP Graph & VRAM Budget (`gfx1201`)

### 6.1 Graph Capture Count: Parameterized Topology via `TreeMeta`
Capturing a separate 48-layer verify graph per `(M, branch_depth, fanout)` combination would multiply graph count and VRAM overhead. Instead, follow the repo's established device-memory staging pattern (`MlaAttnArgs::tree_dev` in `mla_attn.hip.h:67-75` and `QsaStep` in `qsa.hpp:142-166`):
- Capture **1 parameterized tree verify graph** at fixed launch width `T = 8` (`exec_tree_[sel_long_]`, or 2 variants for `sel_long_ in {0, 1}` if long-context selection toggles), reading `TreeMeta` from mapped host memory via `copy_i32_from_mapped` at the start of the graph. (For trees with `M < 8`, pad unused rows `M..7` with dummy `parent = 0` or capture at most `M in {5,6,7,8}` = 4 graphs if profiling shows padding rows cost > 0.2 ms/round.)
- Capture **1 tree commit graph** (`commit_tree_exec_`), reading `h_commit_` (`n_keep`, `leaf_row`, `path[0..7]`).
- **Recommended Phase-2 graph count:** **2 new HIP graphs total** (`exec_tree_` at `M = 8` + `commit_tree_exec_`; 3 if `sel_long_` splits).

### 6.2 Incremental VRAM Budget

| Buffer / Resource | Size | Scope |
|---|---|---|
| `TreeMeta` device + mapped host staging | `< 1 KiB` | 1x per `Verifier` |
| GDN caterpillar state | **`0 B`** | Uses register state `s[r]` directly |
| GDN general-tree fork checkpoint (`gdn_fork_state_`, Phase 3 / `PrefixFork2Tail`) | `2.0 MiB` (`32 * 128 * 128 * 4 B`) | Reused across all 36 GDN layers |
| PLE `hist_snap_` | `0 B` incremental | Already allocated (`max_t_ * 9 * 10240 * 4 B`) |
| 2–3 extra HIP `hipGraphExec_t` objects | `~2–5 MiB` | Graph node/arg metadata |
| **Total incremental VRAM (Phase 2 caterpillar)** | **`~2–5 MiB`** | (`~4–7 MiB` with general fork scratch) |

### 6.3 Phase-2 Bring-Up No-Graph Fallback & Expected Cost
During Phase-2 kernel bring-up, an opt-in debug flag `STRATA_DF2_TREE_NOGRAPH=1` will bypass HIP graph capture and launch the tree verify + commit kernels directly on `cs_` for rapid parity debugging against 1-token decode.
- **Measured cost basis (`verify_kernels.hpp:110`):** A dependent kernel node in a gfx1201 HIP graph costs `~2.4 us`; uncaptured host kernel launches across a 48-layer verify window (`~1,500–2,000` launches including hyper-connections, norms, GDN/QSA, MoE dispatch, and commit) add **~3.5–5.0 ms/round** of host launch/driver overhead.
- Therefore, **Phase 2 verification may use `STRATA_DF2_TREE_NOGRAPH=1` for bit-parity bring-up, but HIP graph capture is mandatory before Phase 3 A/B benchmarking**.

---

## 7. Risks, Open Questions, & Phased Implementation Plan

### 7.1 Key Risks & Mitigations
1. **MoE Routed-Expert Union Expansion (Biggest Throughput Risk):** Adding 1–2 diverse sibling tokens at the branch point can pull in distinct routed experts on the CPU/PCIe path (`verify.hpp:10-11` notes `T=2 -> 1.75x`, `T=3 -> 2.4x`, `T=4 -> 3.05x` one token's CPU misses on linear prefixes, where adjacent positions share more experts than divergent siblings). If the CPU miss union grows too much, extra verify latency could erase the accepted-token gain. **Mitigation:** Measure `STRATA_ROUTE_STATS=1` (`verify.hpp:485`) and `STRATA_DF2_CAND_DUMP` hit rates before tuning `min_branch_p` and fan-out.
2. **QSA Indexer 4-Cell Block Boundary Parity:** Off-by-one errors at `pos0 % 4 in {0, 1, 2, 3}` or across the `2048`-cell sparse selection threshold (`qsa.hpp:135`) would silently corrupt attention selection. **Mitigation:** Dedicated `qsa_tree_select_parity` test + 1-token oracle bit-parity check crossing `pos0 = 2045..2060`.
3. **GDN Kernel Register Pressure on gfx1201:** Saving `s[r]` (`32` floats per thread) in extra VGPRs would halve wave occupancy. **Mitigation:** Caterpillar fast path needs 0 extra VGPR state arrays; general fork uses a single 2 MiB global scratch buffer.
4. **DF2 Runner-Up Quality Without Re-Running the Drafter Backbone:** Because DF2 predicts all 7 mask rows in one forward pass, `seltab_[c+1][j2][*]` conditions on `j2` only through the rank-256 bilinear head, not the 5-layer drafter transformer. **Mitigation:** Audit `STRATA_DF2_CAND_DUMP` traces offline first to quantify runner-up top-1 recall at `--draft-conf` cut points.

### 7.2 Open Questions to Answer Before / During Phase 2
1. On Qwen3.8-Flash-Next coding/reasoning workloads with `--draft-conf 0.35`, how often does `toptok_[c]` rank 2 or rank 3 match the target verifier's argmax when rank 1 is rejected? (Measurable offline from `STRATA_DF2_CAND_DUMP` JSONL logs.)
2. On gfx1201, what is the marginal verify-time delta (`ms`) of `M=8` with 1–2 diverse sibling tokens vs a linear `T=5..7` window?
3. Does `PrefixFork2Tail` (`3 + 2x2`) earn enough extra accepted tokens on rank-2 branches to pay for the 2 MiB GDN checkpoint + restore?

### 7.3 Phase 2 Plan — Kernel + Wiring (`STRATA_DF2_TREE=0/1`, default `0`)
1. **GDN tree kernels (`verify_kernels.hpp`, `verify_kernels.cu`):** Implement `gdn_conv_l2_tree`, `gdn_conv_ab_tree`, `gdn_step_norm_tree` (caterpillar first), and `gdn_commit_tree_layers` + standalone parity test against 1-token sequential execution.
2. **QSA tree kernels (`qsa_select.hpp/.cu`, `kv_stream.hpp`, `native_qsa_indexer.cu`):** Implement `qsa_tree_patch_scores`, `qsa_tree_remap_ids`, `kv_compact_tree` (F16/Q8/Q4/K8V4), and `row_idx` indirection in indexer commit + parity test across all `pos0 % 4` phases and `n_kv > 2051`.
3. **PLE & Verifier tree graph (`verify.hpp`, `verify.cpp`):** Add `TreeMeta`, `copy_tree_step_pos_from_mapped`, `native_ple_postops_tree_snap`, `Verifier::run_tree` (`G=1`), `Verifier::commit_tree`, and `df2_gather_taps`.
4. **DF2 candidate extraction & solo loop wiring (`dflash2.hpp/.cpp`, `generate.cpp`):** Expose `toptok_` / `seltab_` runner-ups on `Dflash2Drafter` when `STRATA_DF2_TREE=1`, wire `Df2TreeSpec` + `df2_tree_greedy_accept` + `Df2TreeAcceptHist` behind `STRATA_DF2_TREE=1` (greedy-only, non-Suffix/Lookup rounds).
5. **Phase-2 Exit Gates:**
   - **Gate 2a (Inert Default):** With `STRATA_DF2_TREE=0` (or unset), output tokens and decode `tok/s` are unchanged from base `3087c8e57`.
   - **Gate 2b (Bit-Exact Greedy Parity):** With `STRATA_DF2_TREE=1`, generated tokens match 1-token plain greedy decode **bit-for-bit** across short prompts, long contexts (`> 2051`), forced branch-kill/runner-up acceptance, and all KV formats.

### 7.4 Phase 3 Plan — gfx1201 Solo A/B & Tuning
1. Enable HIP graph capture (`exec_tree_` + `commit_tree_exec_`) and integrate the tree policy arm into `Df2AdaptiveSolo`.
2. Run controlled same-day A/B benchmarks (`STRATA_DF2_TREE=0` vs `1`) on gfx1201 solo serve across diverse non-lookup prompts (code, math, multi-turn chat) with `--draft-conf` tuned.
3. **Phase-3 Exit Gates (ordered):**
   - **Gate 3a (Acceptance):** Mean accepted tokens per verify round increases by **$\ge +0.25$ tokens/round** (or $\ge +5\%$ relative) on non-lookup prompts.
   - **Gate 3b (End-to-End Speed):** Net solo decode `tok/s` on gfx1201 improves by **$\ge +3\%$** over linear DF2 after accounting for MoE miss union and `kv_compact_tree` commit cost.
