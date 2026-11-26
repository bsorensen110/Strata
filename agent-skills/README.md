# AMD ROCm Agent Skills (vendor drop)

Official agent skills shipped by AMD inside the ROCm install, copied verbatim
into this repo so agents working on Strata's HIP/RDNA4 kernels can use them.
AMD publishes these to teach agents how to profile and optimize AMD GPU kernels
with `rocprof-compute` (kernel-level bottleneck/roofline analysis) and
`rocprofiler-systems` (full-stack sampling/instrumentation).

- Provenance: AUR package `rocm-nightly-gfx120x-all-bin` v10.3.0a20261115,
  copied from `/opt/rocm/share/rocprofiler-compute/{skills,docs}` and
  `/opt/rocm/share/rocprofiler-systems/skills` on 2026-11-25.
- Upstream: [ROCm/rocm-systems](https://github.com/ROCm/rocm-systems/)
- License: MIT (per AMD's `install-skills.sh`; ROCm docs carry their own
  license file under `rocprofiler-compute/docs/license.rst`).
- AMD's `evals/` test fixtures are excluded (their own installer strips them too).

## Layout

The vendor tree layout is preserved so the skills' `../../docs/...` relative
links resolve against the bundled docs:

```
rocprofiler-compute/skills/
  rocprof-compute-kernel-bottleneck/   # per-kernel bottleneck triage
  rocprof-compute-roofline/            # place a kernel vs peak compute/BW ceilings
  rocprof-compute-speed-of-light/      # SoL headroom overview
  rocprof-compute-memory/              # memory-hierarchy analysis
  rocprof-compute-pc-sampling/         # PC-sampling guided optimization
  rocprof-compute-torch-trace/         # PyTorch trace integration
rocprofiler-systems/skills/
  choosing-a-profiling-preset/
  discovering-profiling-options/
  instrumenting-binaries/
```

Each skill dir contains `SKILL.md` (instructions) and `skill-card.md`
(description/owner/license).

## Installing into an agent

Use AMD's per-project installer scripts:

```sh
# from this directory
./rocprofiler-compute/skills/install-skills.sh --agent claude   # ~/.claude/skills
./rocprofiler-compute/skills/install-skills.sh --agent codex    # ~/.agents/skills
./rocprofiler-systems/skills/install-skills.sh --agent claude
```

Note: the scripts rewrite the skills' docs links to point at the docs tree
next to the script they were run from — running them from a machine's own
`/opt/rocm` install keeps links valid there; running from this repo copy
points them at this repo's `rocprofiler-compute/docs/`.

## Refreshing

Re-copy from a newer ROCm install the same way (keep `evals/` excluded), or
pull from upstream `ROCm/rocm-systems` (`projects/rocprofiler-compute/skills`,
`projects/rocprofiler-systems/skills`).
