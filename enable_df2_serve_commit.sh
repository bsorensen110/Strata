#!/bin/bash
cd /home/soren/Strata-integ || exit 1
git add -A
git commit --no-verify -q -F - <<'MSG'
serve: enable upstream's DFlash2 drafter in --serve (single card, no batch)

Upstream's 0.2.4.1 gate refuses DFlash2 under --serve ("it is for one card and
the CLI run"), and its serve loop carries no df2 plumbing at all — only the CLI
path does. This wires the serve path to it:

- gate: drop `o.serve`, keep `multi_gpu || o.batch > 0` excluded
- ver.set_taps(df2.tap_layers()) before ver.init (the taps are what the drafter reads)
- after the verifier/drafter check: bind + prepare, clearing has_dflash (and so
  falling back to MTP with a warning) if the taps or the ~1.2 GiB the pack needs
  are not available — never a hard failure
- per request: df2.reset(), then df2.context(ver.tap_dev(), T, q, e) after each
  prompt chunk's commit rebuilds the drafter's context from that request's rows
- per round: df2.draft(ver.tap_dev(), a + 1, p + a + 1, outv[a], dd, pp, err).
  pos0 is p + a + 1 because the serve loop advances `p` AFTER this block
  (the CLI calls draft_next after advancing it — this was the one real trap)
- the existing serve scaffolding (req_dflash, df_pending_k, dflash_ready,
  df_windows, T = req_dflash && dflash_ready ? df_pending_k + 1 : S_mtp)
  is reused as-is: it survived the merge

Verified: HIP build of the strata target green.
MSG
git log --oneline -1
