# Strata promotion record — heat bootstrap + elastic expert cache

- Deployed: 2026-11-14 18:43 EST
- Production profile: `/home/soren/.local/share/strata-upstream-v0.2.10/components/strata-iq3_s.json`
- Production engine: `/home/soren/.local/share/strata-upstream-v0.2.10/components/engine/fork-vram-structural/strata`
- Candidate branch/worktree: `feat/vram-structural`, `/home/soren/scratch/Strata-vram-structural`; source changes remain uncommitted and scratch-only.
- Engine version: `0.1.59.1`
- Candidate SHA-256: `159069cfb24c698df8b5245e610aedcf551402e3c1caff6e1be15fe55cc967ef`
- Promoted source HEAD: `b1ca1a33371c7d3213b7feee06ec192a21ab7877`
- Preserved known-good engine: `/home/soren/.local/share/strata-upstream-v0.2.10/components/engine/fork-70e634cd7/strata`, SHA-256 `7a86fdb689914f6d6aa0685fb8e557eae7e9b390d6f16cb742b973cd1c8f341f`
- Rollback snapshot directory: `/home/soren/strata-prod-backup-20261114-184122-heatquota` (mode 0700; `manifest.json` contains checksums for original profile, engine and systemd snippets). To roll back: stop `strata.service`; restore the saved original `strata-iq3_s.json`; point its `exe` to `fork-70e634cd7/strata`; daemon-reload and start service; verify authenticated API readiness and a completion.

## Exact production delta

Only the profile's `exe` and the following runtime flags/settings changed; existing arguments, environment, endpoint, DFlash2 pack, cache directories, model and context were preserved:

- `--expert-cache-quota auto`
- `--vram-elastic --vram-segment-mib 512`
- `vram_elastic=true`, `vram_segment_mib=512` in profile metadata for serve `/v1/vram` integration.

## Verification

- Before promotion: focused tests passed 6/6 (`expert_profile_save_test`, `expert_profile_version_test`, `expert_cache_quota_test`, `expert_heat_bootstrap_test`, `expert_cache_live_slot_test`, `auto_cache_test`); build was HIP gfx1201 and contained gfx1201 code objects. No GPU behavior was validated in the candidate A/B before deployment.
- Production restart: ready event at 18:43:53 EST; service active with `NRestarts=0`, `Result=success`.
- Auth: keyed `GET /v1/models` returned HTTP 200; unkeyed returned 401.
- Completion smoke: “What is 19 times 23?” returned `437`, `finish_reason=stop`; 4 completion tokens.
- Runtime command line confirms the candidate path, quota mode `auto`, VMM flag and 512 MiB segment size; DFlash2 loaded the existing `q4_0-selq4.pack`.
- The runtime created `learned.bin.heat` (196,624 bytes, mode 0600) and `.heat.meta` (268 bytes, mode 0600) on first startup. Provenance is `profile-rank-prior`, from the saved profile (`48x512`, 24,576 ranked pairs); this is a synthetic prior, not request-derived routing data.
- Startup used the prior for quotas and printed the 48-layer quota vector: `[231,243,239,234,188,185,233,288,359,286,371,258,229,248,256,144,165,187,193,248,242,284,236,263,244,237,238,170,184,187,203,144,197,186,208,199,239,182,241,234,236,232,257,234,241,233,233,175]` (10,944 total slots).
- Elastic ExpertCache initialized `10,944 mapped / 10,944 live` slots (`25,004 of 25,003 MiB mapped`). No resize or pressure-triggered unmap/remap was exercised. The smoke request log showed 2/3 drafts accepted.
- Engine environment confirms `STRATA_PF_HC_REGION=1`, `STRATA_PF_GROUP_SCRATCH` unset, and preserved production switches. No startup warning/refusal was found for the newly enabled flags.
- Service remained active after smoke; 0 restarts. GPU used VRAM ~31.3 GiB, ~1.3 GiB free at idle after test. Host MemAvailable ~15 GiB, SwapFree ~94 GiB.
- Full workload quality, runtime update from prior to measured routing heat, extended soak, and VMM live shrink/grow remain unverified.
