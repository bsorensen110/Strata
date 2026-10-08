# Local Strata installation — AMD R9700

## Service and endpoints

The existing user service is enabled for boot and configured for LAN access. Manage it with:

```bash
systemctl --user start strata.service
systemctl --user stop strata.service
systemctl --user restart strata.service
systemctl --user status strata.service
journalctl --user -u strata.service -f
```

- Browser: http://192.168.1.151:8086/
- OpenAI API base: http://192.168.1.151:8086/v1
- Anthropic endpoint: http://192.168.1.151:8086/v1/messages
- Model ID: `swift-1.5-iq3_xxs`.
- Bind: private IPv4 `192.168.1.151` only, not loopback/wildcard/public IPv6. Use this address from the server too.
- Authentication remains off. Existing UFW configuration permits TCP/UDP 8086 from anywhere; no firewall rule was added or widened. Do not forward this unauthenticated service to the internet. Tightening existing rules needs administrator access.
- Unit: `/home/soren/.config/systemd/user/strata.service`, enabled in `default.target.wants`.
- `loginctl show-user soren -p Linger` is `yes`, so the user manager starts at boot without an interactive login. No system-level duplicate unit was created.
- The unit reads the JSON host rather than overriding it with `--host 127.0.0.1`. The manual launcher reads the same configuration.
- `Restart=on-failure`, `RestartSec=5`, `StartLimitIntervalSec=0` retry if DHCP has not assigned the private address yet. Reserve `192.168.1.151` in the router: the interface currently uses DHCP.
- LocalAI is stopped temporarily, still enabled; its service/packages were not removed.
- Boot enablement/linger are verified configuration, not a claim that a reboot was performed. Start/stop, web UI, health, model listing and real arithmetic were tested at the LAN address; the user confirmed that the page loads from the other computer.

The alternative manual launcher is `./run-swift-iq3_xxs.sh` from this directory. Do not run it or the benchmark runner beside the live service on port 8086.

## Current source and configuration

- Strata **0.1.30**, HIP, AMD Radeon AI PRO R9700 (`gfx1201`), native upstream hardware support.
- Tag commit: `30ec18ec7094550fcc594fd948220d511d80464e`; local branch `r9700-v0.1.30`.
- Approved local fast-intrinsics overlay: hardware byte permutation plus packed-byte wrapping/saturating subtraction and comparison. Upstream architecture guards and native support remain intact.
- Two host BF16 tuner/test input-encoding fixes remain necessary on this SDK. They use the existing `bf16_from_f32()` helper.
- The complete old experimental hardware-support patch is NOT applied to v0.1.30. The three tracked local modifications are preserved and must not be reset blindly.
- Canonical engine: `engine/strata`; build metadata: `engine/BUILD.json`, source fingerprint `53d27c706d92cdfd`.
- Tested canonical engine SHA-256: `6c6eaa659033683e717b29e4e2b6029735fdf6961e22ac9cf40373d080440241`.
- Existing ROCm at `/opt/rocm` and Python venv `.venv`; no package/driver changes.
- Swift 1.5 IQ3_XXS, 131072-token context, INT8 KV, text-only, memory-mapped experts, prepared speculative draft runtime. Experimental speed projection remains off.
- Prepared model/data directory: `/home/soren/Strata-data`. No model redownload or repacking was needed.
- Active config: `strata-swift-iq3_xxs.json`; model settings preserved. After upgrade, only `host` changed from loopback to `192.168.1.151` for LAN serving.
- hipBLASLt profile: `tools/hip/gfx1201-hipblaslt-100500.txt`, unchanged SHA-256 `e587f30aaab7033aee336e9e0ac60ca37074ce2ad8cd5f4ac019633b34497363`.

## Verified upgrade

- Canonical clean rebuild completed; metadata confirms version 0.1.30, HIP, gfx1201.
- 43 selected CTests passed, including intrinsics and handoff under the normal deadline. Four exclusions: `ple_parity`, `platform_memory_test`, `expert_parity`, `pool_test`. The last two require an absent legacy `pack/full/experts.bin` fixture; they are not counted as passes.
- Additional GPU intrinsics parity passed 262144 cases / 1048576 operation results, all 65536 byte pairs, 4096 selectors and 16 guards.
- Real GGUF expert parity passed on Swift shard 1 layer 0 and shard 2 layer 47, covering mixed native quantization and CPU/GPU/reference comparisons.
- Both BF16 encoder regressions passed. All 64 production-Gemm checks passed against the new canonical libraries, with 64 Lt launches and zero fallbacks.
- Real-model arithmetic, system-marker recall, multi-turn recall and streaming passed, both in the dedicated release benchmark and in the final running service.
- Benchmark runtime: 16480 Lt launches, zero fallbacks, all 16 observed projection geometries covered by the unchanged 32-row profile. This was reuse with fresh validation, not a new calibration run.

## Recorded performance

All release trials used the saved long-prefill warmups and three fresh, deterministic, length-capped 512-token generation requests. These were sequential same-session runs, not an interleaved replicated causal experiment.

- Fresh v0.1.29 reference: 69.14 generation tokens/sec.
- Stock native v0.1.30: 55.01 tokens/sec.
- Approved v0.1.30 fast-intrinsics candidate: 67.52 tokens/sec.
- Actual canonical v0.1.30 build: **67.50 tokens/sec**, trial range 61.3–72.4; 1536 outputs / 22.7556 engine decode seconds. HTTP wall time was 23.9222 seconds.
- Canonical prefill: 1591.08 input tokens/sec at 4210 tokens; 1554.12 at 8830 tokens, three fresh trials per size.

Current release evidence: `upgrade-v0.1.30/upgrade-report.json`, `tuned-arm/`, `ctest.xml`, `full-parity.log`, `full-parity-engine.log`, and `production-verification.log`.

The original v0.1.29 tuning A/B evidence remains under `hipblaslt-benchmark/`; its earlier 64.73 → 67.65 decode and small prefill improvements are historical, not new-release speed claims. Both original GGUF hashes were verified during installation; those records remain in `download-verification.json`.

## Replay and rollback

The Hermes `strata-hip-install` skill is version 0.3.0. Read its `references/upstream-upgrades.md` for native-release building, the approved overlay, tests, performance gate, promotion and rollback. Read `references/hipblaslt-replay.md` for complete calibration and matched testing commands. Re-enable `STRATA_BUILD_TESTS=ON` after each engine-helper build; the helper resets it.

Rollback inputs: `upgrade-backups/v0.1.30-dda723cd/`, including `state.json`, source patch, old binaries/metadata, config/profile/unit, previous build tree and fresh reference trials. The immutable pre-upgrade stash is `77a80c08cb8a11f80218085057a508b80910c45e`. Its source replay and old engine checksum were verified; no destructive rollback was performed on the final working service.

To repeat LAN API/UI and real-model smoke checks while the service is ready: `.venv/bin/python verify-lan-service.py`. It saves `lan-service-verification.json`. Historical upgrade verification/benchmark helpers assume localhost; their prior raw evidence remains unchanged and should not be used unchanged against this private-IP listener.

Full-window 131072-token inference and broad answer quality remain untested. Recalibrate if the actual hipBLASLt version or relevant matrix routes change; never relabel a profile header to bypass its version guard.

## gfx12 WMMA prompt attention (2026-09-30)

The live engine includes the int8-KV QSA prompt attention on RDNA4 matrix cores (upstream PR
https://github.com/Niko1221/Strata/pull/329, branch bsorensen110:gfx12-wmma-prompt-attn, commit e591767).
Measured on this R9700 (A/B, same binary, old kernel via STRATA_PROMPT_ATTN_OLD=1): prompt processing 1.35x at 31K
tokens, 1.26x at 108K; decode unchanged. Config (strata-swift-iq3_xxs.json) and the systemd unit were not changed.

- Engine: engine/strata sha256 acba57d14f70db80a0b47429cb4ac169192f66b191d7fb3c10ae3078f9273b3b
- Rollback: service-backups/pre-wmma-20260930-222329/ (previous strata sha256 6c6eaa659033683e..., BUILD.json, config, unit, wmma-prompt-attn.patch).
  Stop the unit, copy strata and BUILD.json back into engine/, git apply -R the patch, start the unit.
- Escape hatch without a rebuild: add "STRATA_PROMPT_ATTN_OLD": "1" to "env" in the config and restart.

## gfx12 QSA select (2026-10-01)

- PR https://github.com/Niko1221/Strata/pull/337: WMMA block scorer plus a top-k dispatch that follows the prompt length, not the `--max-context` capacity.
- Canonical branch `r9700-v0.1.30` = v0.1.30 + local overlay + PR #329 + PR #337 (merge `498836b`), pushed to `bsorensen110/Strata`.
- Live engine sha256 `2e1f25b482c2d54c4e8bcbaa20459c4184dcdc4b76c21eaebb8887627c209453`, BUILD.json `src` `2921351f51b8c456`.
- Rollback: stop the unit, copy `strata` and `BUILD.json` from `service-backups/pre-p4-20261001-000119/` into `engine/`, `git apply -R` that directory's `p4-select.patch` (or `git revert -m 1 498836b`), start the unit.
- Escape hatches (config `env`, then restart): `"STRATA_SELECT_OLD": "1"` (warp scorer) and `"STRATA_TOPK_OLD": "1"` (reference top-k).
- Prefill at 188K-262K tokens: about 1.5-1.7x faster than with only the #329 kernel (1,020 -> 1,748 tok/s at 245K); under ~10K tokens about 1% (not worth noticing).

## Larger --prefill chunk (2026-10-01)

**Result:** `--prefill auto` stops at 8192-token chunks. Explicit `--prefill 32768` (or the `prefill-auto-chunks` change that lets auto reach it) raises prompt processing by about **20-23%** at 8K-188K tokens, with decode unchanged. The live service config still has `--prefill auto` (not switched yet; see "To apply").

Full numbers, reproduction and checks: `/home/soren/Strata-ab-qsa/results-chunk/SUMMARY.md`.

## Upgrade to v0.1.31 (2026-10-01)

Live service now runs engine 0.1.31 built from branch `r9700-v0.1.31` @ `5064923` (v0.1.31 + our PRs #337, #339, #329 + the larger `--prefill auto` chunk, fork `bsorensen110/Strata`). Engine sha256 `fd0353d271d69a2e...`; `--prefill auto` now picks 32768. Config and unit unchanged. Decode 66.5 tok/s (3x512), was 61.5. Rollback: `service-backups/pre-v0.1.31-20261001-024625/` + `git checkout r9700-v0.1.30`. Details: skill `strata-hip-install`, section "Upgrade to v0.1.31".
