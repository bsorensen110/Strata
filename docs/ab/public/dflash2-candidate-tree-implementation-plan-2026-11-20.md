# DFlash2 candidate-tree implementation plan (2026-11-20)

**Status: gated plan.** This slice adds a host-only regression and does not implement a runtime
tree, a parent-node GPU kernel, or a new draft API. The existing linear path is unchanged. No
target-tree acceptance, speedup, or runtime memory use has been measured.

## What the spike established

The earlier scratch spike passed on the CPU. The reported compiler command was
`g++ -std=c++17 -O2 -Wall -Wextra -Werror` (scratch source, output and run paths were not recorded
here). Two siblings cloned both recurrence and convolution history from the same parent. They
matched independently gathered FP64 serial paths in either order, left the parent unchanged, and
published the selected leaf's state. An intentionally shared-state control exposed leakage: max
recurrence error `0.107667083`. This is the reported scratch result, not a run of the checked-in
regression.

This establishes the state-copy idea for one GDN layer in host FP64 arithmetic. It does not show GPU
kernel parity, branch-safe QSA or PLE, end-to-end target correctness, or throughput. The checked-in
test extends the reference to multiple levels and heads, zero and random initial state, and
histories shorter and longer than the three carried rows. It checks every path against the fixture
oracle, swaps sibling execution order, changes one branch's input, and checks that only a selected
leaf is published. It was not run in this slice.

## P0: decide whether to build

Follow §§3-5 of the
[block-16 decision note](../block16-lilicorr-decision-2026-11-19.md#3-the-decisive-unknown--exactly-p0).
At the solo operating point in that note (150k-250k context, Hermes traffic, IQ4 target,
`--draft-conf 0.35`), measure **conditional per-position acceptance for DFlash2 separately from
lookup and suffix**, plus selector **Recall@1/2/16**. Keep the reached and accepted counts per
source and position, and the sample counts for recall. Compare selector ranks with the target's next
token on the same prefix; stop or label the probe when a proposed prefix diverges. Do not mix later
rows from a diverged path into recall on the matched prefix.

Recall measures whether the target token is among the candidates. It is not achieved acceptance: the
tree has a finite node budget and only one target-consistent ancestry can win. **Flat acceptance
(the note's example is a tail at or above 84%): stop tree work.** A steep curve (the note's example
is around 72%) with useful incremental recall under a realistic node budget justifies considering
the next stage, not shipping it. If the evidence is inconclusive, or extra candidates rarely help
where the chain fails, hold. Investigate the acceptance loss before assuming branching is the best
fix. P0 data collection is separate work and is not authorized by this plan.

## Design and acceptance stages

1. **CPU lineage regression (this slice).** Start with a fixed tree and the existing FP64 fixture.
   Check all outputs, final recurrence and exact convolution history against fresh serial paths.
   Require finite values, parent and sibling isolation, order invariance, a perturbation that cannot
   cross siblings, and rejection of the deliberately shared-state control's result. The host test is
   not a GPU or target oracle.
2. **CUDA/HIP GDN parity, separately approved after P0.** Prototype parent-indexed input and output
   state for each node: read the parent's recurrent matrix and raw convolution tail; write only the
   child's buffers. Pin FP32 recurrence and output tolerances against the fixture and the
   same-backend one-token path, and require exact tail copies and unchanged live state. Cover a
   width-one chain, fork and deep fork, sibling reordering, zero/random state, rejection, and the
   first serial step after commit. Measure scratch and synchronization or copy costs on each backend
   before integrating it. Do not infer parity on one backend from the other.
3. **Opt-in solo deterministic-greedy verification.** Keep the linear path as the default. Start
   with a small, bounded tree over the existing pack-trained DFlash2 block. Use its top-16
   candidates and predecessor/candidate transition scores as proposal inputs, with stable
   tie-breaking, sibling-token deduplication and a hard depth/node budget. This is reuse of one
   block's lattice, **not** generation of fresh draft logits conditioned on each full branch prefix.
   Assign every node a token, parent ID, depth, logical position, and separate physical storage ID.
   Enforce parent-first order and one logical-position step per edge. Use logical positions for RoPE
   and other positional rules, never physical node slots. Use separate graph/cache keys and scratch
   ownership for tree and linear paths; key or validate topology, capacities and layout as required
   by graph capture. Fall back only from an unchanged live state.
4. **Full target lineage and publication.** Read the committed prefix as shared, immutable input.
   Keep speculative QSA K/V and indexer state private by lineage: tail, spare key, pooled keys and
   the block-position metadata must follow the parent. Siblings at one logical position cannot share
   a canonical K/V destination. Restrict both indexer selection and attention to the committed
   prefix plus that node's ancestry (including the current node where the serial causal rule
   includes it); mask before top-k, not just at the attention read. Keep PLE convolution/hash state,
   token and penalty histories, and draft tap/context inputs on the same lineage. Walk from the root
   using the target's own greedy token at each prefix: follow a matching child only, and use the
   target token at the first missing child or exhausted leaf, with the same EOS and length semantics
   as the serial path. Publish only the winning root-to-leaf path, or replay just that path into
   canonical state; never pass a count over flattened tree rows to `commit(n_keep)`. Publish only
   consumed input rows: don't commit a target fallback token before it has been run. Verify no
   rejected row appears in committed GDN, QSA/indexer, PLE, histories or draft context. Require the
   exact greedy token stream and next-step parity versus serial target decode on the same backend,
   including no-match, early EOS/limit, same-position siblings, QSA block boundaries, cache/graph
   reuse, and long-context cases.
5. **Matched A/B, only after parity.** Compare opt-in tree against the existing linear DFlash2 path
   with the same target, pack, cache mode, context, prompts, device and settings. Track attained
   accepted drafts per step (by source), verified rows, draft/verify/commit times, decode
   throughput, tail latency and peak host/device memory. Include the same-day linear baseline and
   warm/cold cases as appropriate. Stop if exact greedy parity fails, a discarded branch leaks,
   memory does not fit the supported configuration, or the tree has no repeatable end-to-end gain
   against the matched baseline. Acceptance and speed remain unknown until measured.

**Deferred:** sampling (the tree proposal/acceptance law is unproven), batching, lookup/suffix
composition, other backends, and any default change or promotion. Fail closed on unsupported modes;
don't silently use a tree for sampling. Beyond the host test, every stage needs separate approval.

## GDN state budget (estimate, not measured allocation)

Flash-Next uses `S=128`, `HV=48`, `C=10240`, `d_conv=4`, and 36 GDN layers. With four bytes per
float, one layer's recurrence plus raw convolution history costs
`(S * HV * S + C * (d_conv - 1)) * 4` bytes. A naive strategy retaining one complete FP32 copy for
each live lineage costs:

| Copies | Bytes | MiB | GiB |
| --- | ---: | ---: | ---: |
| One GDN layer | 3,268,608 | 3.117188 | 0.003044 |
| One 36-layer lineage | 117,669,888 | 112.218750 | 0.109589 |
| 15 extra lineages | 1,765,048,320 | 1,683.281250 | 1.643829 |
| 16 total lineages (base plus 15 extra) | 1,882,718,208 | 1,795.500000 | 1.753418 |

MiB and GiB use powers of two. This counts neither QSA K/V/indexer, PLE, token histories, replay
rows, taps, logits, draft buffers, activations, graph workspace nor alignment or allocator overhead.
Full snapshots are a reference, not a proposed allocation policy. If the budget doesn't fit, test
layer-local scratch reuse or replay before increasing the node cap. Extra memory traffic,
synchronization, QSA selection at long context, graph recapture, and less efficient expert routing
could outweigh even useful recall; none is measured here.

## Source map

These are static source observations, not runtime verification.

- P0 and the decision gate: `docs/ab/block16-lilicorr-decision-2026-11-19.md:70`,
  `docs/ab/block16-lilicorr-decision-2026-11-19.md:82`,
  `docs/ab/block16-lilicorr-decision-2026-11-19.md:98`.
- Fixture: one-head FP64 step and independently gathered serial-path oracle entry points in
  `engine/tests/gdn/gdn_fixture.hpp`. Flash-Next GDN geometry and the 36-layer GDN count are in
  `engine/include/strata/core/geometry.hpp`. FP32 state cost is computed in
  `engine/include/strata/plan/plan.hpp`.
- Linear verifier: `engine/include/strata/core/verify.hpp` and `engine/src/core/verify.cpp` commit the
  first rows by count. `engine/include/strata/kernels/verify_kernels.hpp` documents non-writing multirow
  recurrence; `engine/src/core/verify.cpp` supplies no keep count for solo multirow verification,
  gates live convolution writes on self-commit, and starts the per-layer GDN commit replay. The existing kernels take
  linear rows or independent slot groups, not parent-node IDs.
- QSA and histories: `engine/src/core/verify.cpp` appends to the session's canonical K/V;
  `engine/src/kernels/cuda/qsa.cu` keys the F16 write by position. `engine/include/strata/kernels/qsa.hpp`
  documents indexer state; `engine/src/core/verify.cpp` restores its tail and
  appends accepted rows. `engine/src/core/verify.cpp` advances the
  host-side PLE hash-token history; `engine/include/strata/core/verify.hpp` describes per-row penalty
  history and `engine/src/core/verify.cpp` shows how linear graph variants are keyed.
- DFlash2: `engine/include/strata/kernels/dflash2.hpp` fixes the block width. `engine/src/core/dflash2.cpp`
  generates top candidates and launches the selector; `engine/src/kernels/cuda/dflash2.cu` describes
  the optional transition table over predecessor and candidate slots.
  `engine/include/strata/core/dflash2.hpp` describes the normal single-path result and the opt-in
  sample-walk candidate output; neither is a target tree nor proves a tree sampling law.
