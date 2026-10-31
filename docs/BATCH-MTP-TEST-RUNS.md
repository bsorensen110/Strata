# Batch MTP test runs

The runs that measure the opt-in batch MTP switches on branch `reimagined-memory`. None has been run yet.
Background and how each switch works: [DECODE-DRAFT-SPEEDUP-IDEAS-2026-10-30.md](DECODE-DRAFT-SPEEDUP-IDEAS-2026-10-30.md).

## Setup

- One engine binary built from this branch, one config. Change one thing per arm.
- Stop `strata.service` and use a scratch server (under `systemd-run --user --scope`), or edit the service's
  environment between arms.
- MTP, not DFlash2: drop `--draft dflash2` and `--dflash-no-mtp`, pass `--mtp ...`.
- Set `STRATA_BATCH_DRAFT_TIMING=1` in every arm.

## Arms

| Arm | Flags and environment | Compare with |
|---|---|---|
| A | `--batch 4 --batch-mtp` (control) | |
| B | A + `STRATA_BATCH_MTP_DEVTOK=1` | A |
| C | A + `STRATA_BATCH_MTP_DEPTH=3` | A |
| D | A + `STRATA_BATCH_MTP_DEPTH=3 --batch-draft-window 16` | A |
| E | D + `STRATA_BATCH_MTP_DEVTOK=1` | D |
| F | A + `STRATA_BATCH_MTP_EARLYCOPY=1` | A |
| G | A + `STRATA_BATCH_MTP_DEVTOK=1 STRATA_BATCH_MTP_DEVACCEPT=1` | B |
| H | A + `STRATA_BATCH_MTP_DEPTH=3 STRATA_BATCH_DRAFT_GROUP=1` | C |
| I1 | A + `STRATA_BATCH_EXPERT_AHEAD=1` | A |
| I2 | A + `STRATA_BATCH_EXPERT_AHEAD=2` | A |
| J0 | `--batch 6 --batch-mtp` (control for J) | |
| J1 | J0 + `STRATA_BATCH_MTP_SELECT=1` | J0 |
| J2 | J0 + `STRATA_BATCH_MTP_SELECT=1 STRATA_BATCH_MTP_SELECT_OVERLAP=2` | J1 |

## Each arm

1. The `strata-speed-ab-testing` 4 x 256 concurrent test, 3 times (for J, 6 prompts).
2. The solo greedy code test once (2 trials). One busy slot shows depth best.
3. Report cold trial 1 and warm trials apart.

## What to record

- Aggregate tok/s and per-slot tok/s.
- The engine's `generated in` lines.
- Draft time per window (`STRATA_BATCH_DRAFT_TIMING`).
- The stderr lines printed when the slots go idle: "MTP proposals accepted", "MTP depth", and for G
  "device-staged MTP rounds" (kept, redrafted by the host, not staged).

## Pass rules

- At `temperature=0`, every arm gives the same text as its control for the same prompt. A difference is a bug.
- B should cut the draft time per window (about 0.75 ms per slot today), not the acceptance.
- G: a high "redrafted by the host" count means the staged round is thrown away too often to pay.
