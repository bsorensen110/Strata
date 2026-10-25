# Fork notes displaced by the upstream v0.2.6 doc rewrite — docs/INTEL.md

> **Archival addendum, not maintained.** The 0.2.6 merge (2026-10-25) took upstream's rewritten
> `docs/INTEL.md` wholesale; the fork-side passages that rewrite replaced are kept here verbatim, in the
> order they appeared, so measured results and community validation notes stay at the branch tip.
> They were written against the pre-0.2.6 text and are not updated when upstream's docs move.
> Source: branch `integ/pre-upstream-0.2.6` (commit `cf70f46a`), the file's conflict blocks.
> Recover the whole pre-merge file with `git show cf70f46a:docs/INTEL.md`.

## How to build it

        sudo apt install docker.io git
        sudo usermod -aG docker,render $USER

   Log out and in again (or reboot) so the groups apply. `docker ps` must work without `sudo`; the `render` group lets
   the container open the GPU.
2. **Get Strata and build the engine** ("How to build it" below, about 10 minutes the first time).
3. **Run setup:**

        ./setup.sh --backend sycl [setup.py's options, e.g. --model IQ3_S --context 32768 --port 8085]

   `setup.sh` makes its own Python environment in `.venv`, so Ubuntu's "externally managed" system Python is not
   touched. **Do not run `python3 sycl/setup_intel.py` yourself:** it installs setup's packages into whichever Python runs
   it, and Ubuntu's refuses (`externally-managed-environment`). `./setup.sh --backend sycl` is the one command, for the
   first install and every later start.

Setup is upstream's `setup.py` with the Intel steps swapped in (`sycl/setup_intel.py` imports it and replaces those steps;
setup.py itself is unchanged). The model choice, download, pack, tokenizer, MTP draft layer and the context and KV
questions are setup's own. What changes:

- **GPU check:** the Arc is found in sysfs (vendor 8086 under `xe` or `i915`) and offered through setup's AMD
  path. That is the path that builds locally and has no images.
- **Engine step:** it uses the SYCL build (`build-sycl-aot/strata` or `build-sycl/strata`, run in the `strata-sycl-dev`
  image by `sycl/serve/strata-sycl.sh`) instead of compiling CUDA or HIP.
- **RAM rule:** this does not apply on an `xe` card. The CUDA engine keeps every expert in RAM; the port streams them from
  the GGUF into VRAM (`--stream-experts`), so RAM only decides the KV streaming. `--check` lists what fits by VRAM.
- **The config:** it uses the container's paths and `"backend": "sycl"`. The VRAM reserve is 1,024 MiB up to
  32K and 2,048 MiB with 4,096-token prompt chunks above that, 300 MiB on a card under 12 GB; a
  `--vram-reserve-mib N` you give is kept as given. KV streaming (`--kv-resident 32768`) is on from
  64K up when the RAM holds the KV. A `model_switcher` or `sampling` block from an earlier config is kept.
  `run-<model>.sh` starts `sycl/serve/server_intel.py`.
- **Arc A-series (`i915`, e.g. the A750 with 8 GB):** a different config, see "Arc A750 and the other Alchemist cards" below.
- **An integrated Arc (`xe`, e.g. the Arc 140T):** no Docker, the engine runs on the PC's own oneAPI; see "Integrated
  Arc" below.
- **Engine settings:** the config's `"env"` block reaches the engine. `strata-sycl.sh` forwards every variable starting with
  `STRATA_`, `ONEAPI_`, `UR_`, `IGC_`, `SYCL_` or `ZES_` into the container and sets `STRATA_VERIFY_DEVICE_PLAN=1`,
  `STRATA_VERIFY_NO_HOST=1` and `STRATA_STAGER_THREADS=12` unless the environment says otherwise (for example
  `"env": {"STRATA_VERIFY_NO_HOST": "0"}`).

If a future `setup.py` drops a step this relies on, it stops with a message instead of writing a wrong config.
The models, packs and checkout must sit under the folder `strata-sycl.sh` mounts at `/work` (the one above the
checkout, or `STRATA_SYCL_ROOT`). An earlier Strata install on the same PC keeps its data folder in
`~/.config/strata/settings.json`; pass `--data-dir <folder above the checkout>/Strata-data` to put this one under the
mount. A symlink to a folder outside the mount does not work in the container; a hard link does.

Then `run-<model>.sh` (or `./setup.sh --backend sycl` again) starts the model. The first start of a 30 GB model
takes about two minutes. `--port N` and `--host 0.0.0.0` work as in upstream's setup.

## How to build it

The engine is built in Docker, in the `strata-sycl-dev` image (oneAPI 2026.1 compiler, oneMKL and Level Zero, from a
community llama.cpp SYCL image), so the PC needs no oneAPI. From the checkout (its folder is called `Strata` here; the
folder above it is what the container mounts at `/work`):

    cd Strata
    docker build --pull -t strata-sycl-dev sycl/tools   # once: about 6 minutes, a 13 GB image
    H=$(basename "$PWD")

**Arc B-series** (Pro B70: `bmg-g31`; B580, B570 and Pro B60: `bmg-g21`; `ocloc ids bmg-g21` in the image lists the
device ids). Ahead-of-time code, no compile at the first start:

    docker run --rm -u $(id -u):$(id -g) -v "$PWD/..:/work" -e AOT=bmg-g31 -e REPO=/work/$H \
        -e BUILD_DIR=/work/$H/build-sycl-aot strata-sycl-dev "cd /work/$H && bash sycl/tools/build.sh strata"

**Arc A-series** (Alchemist: A750, A770, A580, A380): no `AOT=`. The code is compiled from SPIR-V at the first start
(JIT, about a minute). Nobody has run an AOT build for these cards.

    docker run --rm -u $(id -u):$(id -g) -v "$PWD/..:/work" -e REPO=/work/$H \
        -e BUILD_DIR=/work/$H/build-sycl strata-sycl-dev "cd /work/$H && bash sycl/tools/build.sh strata"

**Podman instead of Docker** (e.g. Fedora Silverblue, rootless): `podman build -t strata-sycl-dev sycl/tools`, then the
same `run` lines with `podman` and `--security-opt label=disable` (a data disk without SELinux labels is otherwise
unreadable in the container). To serve or bench that way, set `STRATA_SYCL_RUNNER=podman` for `strata-sycl.sh`;
`STRATA_SYCL_RUNNER=native` runs the engine directly when the server already runs inside a toolbox with oneAPI (the
data root at `/work`), and `distrobox:<name>` enters that distrobox from the host.
`--pull` matters: the image starts from a community image whose tag moves, and `docker build` reuses a copy of it from an earlier
build unless it is told to fetch the current one. An old copy can hold a device compiler (ocloc from Compute Runtime
26.22.38646, IGC 2.36.3) that crashes on the Battlemage kernels ("IGC: Internal Compiler Error: Floating point exception", 77 device
images, then `gen compiler command failed with exit code 245` at the last link, #3219). `sycl/tools/build.sh` checks the
version before the compile and stops with the fix. If you cannot update the image, `-e I8_ESIMD=OFF` builds without the fused int8
kernels (stubs: the prompt's experts take the slower XMX path, the answers do not change); `-e ALLOW_BAD_OCLOC=1` builds anyway.

The build takes about 3 minutes and ends with `BUILD EXIT 0` and `errors: 0`. The binary is `build-sycl-aot/strata`
(B-series) or `build-sycl/strata` (A-series); setup finds either one. Run the `docker build` again after pulling a
Strata update that changes `sycl/tools/Dockerfile`, and the `docker run` after every update of the engine (setup
says when the engine is older than it needs).

## Things that matter on this GPU

- **`SYCL_CACHE_PERSISTENT` must be 0.** The persistent JIT cache segfaults on Xe2 during the first
  compile. The start script sets it; if you run llama-server by hand, do too.
- **The whole model goes on the card** (`--n-gpu-layers 999`), except `per_layer_token_embd.weight`,
  the single 28.8 GB tensor of shard 2. `--override-tensor per_layer_token_embd=CPU` keeps it in host
  memory, mmapped and paged from the SSD by row, which is how the model's authors serve it. Host RSS
  stays around 2 GB.
- **Thinking.** The model reasons before it answers. Strata's web app has the setting; for an API
  client, put `{"reasoning_effort": "none"}` in `strata-<model>.shared-settings.json` next to the
  config, or send `chat_template_kwargs: {"enable_thinking": false}` per request. Without one of
  those a short `max_tokens` is spent entirely inside the think block and the answer looks empty.
- **`/health` says 503 while loading**; the server polls `/props` instead.

## How much context fits

The architecture keeps this small: only every fourth layer is full attention (12 of 48); the other
36 are gated-delta-net layers with a fixed-size recurrent state that does not grow with context.
Per token, the 12 attention layers keep 2 KV heads x 256 x (K + V) at q8_0, plus the
sparse-attention indexer's keys. **Measured: about 20 KiB per token** (VRAM grows 0.6 GB per 32k
of context with everything else unchanged).

| context | VRAM in use, model loaded | headroom on 32 GB | status |
|---|---|---|---|
| 32,768 | 28.4 GB | 3.5 GB | measured, the default |
| 65,536 | 29.0 GB | 2.9 GB | measured, loads |
| 98,304 | 29.6 GB | 2.3 GB | measured, loads |
| 131,072 | 30.3 GB | 1.5 GB | **measured, the practical ceiling**: a 104,798-token prompt (79% of the window) read in 247 s at 424 tok/s, then answered |
| 163,840 | ~31.0 GB | ~0.9 GB | not attempted: under the 1.2 GB safety margin |
| 262,144 | ~32.6 GB | none | **does not fit - asking for it took the host down** |

**Do not ask for more than fits.** On this driver a GPU allocation past VRAM does not fail: the xe
driver evicts buffers into host RAM, the kernel runs out of memory, and the machine livelocks until
its hardware watchdog resets it. A 262K request did exactly that on 2026-09-29, and llama.cpp's own
"failed to fit params" check fired too late to prevent it. Compute the KV size first and leave
1.5 GB free.

Prompt reading gets faster with size, because the work batches better: ~150 tok/s on a 2.7k-token
prompt, 424 tok/s on a 105k-token one. So a full 128k window costs about five minutes to read,
not fifteen.

## The engine itself on Intel: the SYCL port (`sycl/`)

Everything above runs the model through llama.cpp. This is the other half: Strata's own engine - the 50 CUDA
kernels and the host code that drives them (streams, events, graph capture, pinned memory) - built for the
Arc with oneAPI. It is a migration of the tree, not a new backend: the engine has no backend seam to slot
into.

**Layout.** `sycl/src` and `sycl/include` mirror the tree and hold only the files the port changes; the
SYCL build (`sycl/CMakeLists.txt`) takes every other source from the original location. No upstream file
is edited. `sycl/include/dpct/` is the vendored SYCLomatic helper library, so the port builds without the
migration tool.

**Expert-parallel prompt row reduction (opt-in).** With `--expert-parallel-device`, the peer card returns one
FP32 row per routed expert of every prompt token, and the primary combines them. `STRATA_EP_PREFILL_REDUCE=1` sums
the peer-owned routed experts of each token on the peer first and returns one FP32 row per token; the primary adds
its own routed sum and the shared expert. Decode is unchanged. Unset keeps the per-expert rows and the original
combine. The FP32 addition order changes, so generated text can change. `STRATA_EP_PREFILL_PUSH=1` (with the
reduction) moves the sum region to the primary and has the peer write it through the IPC mapping, each producer
work item fencing at system scope; the host peer wait stays, and the primary still copies the sum uncached before
using it. PUSH changes the transport, not the arithmetic: its outputs are bitwise those of the reduction alone.

Measured on 0.2.0-beta, two B70s, IQ3_XXS, `--expert-parallel-device 1 --prefill 8192 --max-context 65536 --kv int8
--spec 4 --mtp`, one slot, fresh prompts, one engine per arm (prompt tok/s at 4K / 32K):

| | 4K | 32K | decode tok/s, two prompts |
|---|---:|---:|---|
| unset | 635 | 720 | 80.0 / 81.3 |
| `STRATA_EP_PREFILL_REDUCE=1` | 993 | 1,264 | 80.6 / 83.8 |
| `+ STRATA_EP_PREFILL_PUSH=1` | 1,148 | 1,514 | 80.2 / 83.4 |

A one-off posted write of 256,235,520 bytes takes 22.8 ms (11.2 GB/s) against 77.2 ms (3.3 GB/s) for an uncached
peer read, measured on 0.1.x; it excludes the primary's later uncached copy. These are bounded screens at one slot,
not a stability or 200K-context qualification.

`sycl/probe/ep_prefill_reduce.cpp` checks two-card IPC updates against an independent FP32 FMA oracle in the
original router order: all-primary, all-peer and mixed ownership, up to 4,096 tokens, repeated updates and a cached
GPU consumer. Run it without arguments and with `--push`; both 26-case suites pass with a largest error of 9.54e-7
(tolerance scaled by contribution magnitude). Build it in the oneAPI development image and run it with both cards;
Docker needs `--cap-add SYS_PTRACE` for the Level Zero IPC import under its default seccomp profile:

```sh
icpx -O2 -fsycl -fsycl-targets=spir64_gen \
  -Xsycl-target-backend=spir64_gen "-device bmg-g31 -options -cl-fp32-correctly-rounded-divide-sqrt" \
  -I sycl/include -I include sycl/probe/ep_prefill_reduce.cpp \
  sycl/src/kernels/cuda/ep_kernels.dp.cpp -lze_loader -o ep-prefill-reduce-probe
ONEAPI_DEVICE_SELECTOR='level_zero:*' SYCL_CACHE_PERSISTENT=0 ./ep-prefill-reduce-probe
ONEAPI_DEVICE_SELECTOR='level_zero:*' SYCL_CACHE_PERSISTENT=0 ./ep-prefill-reduce-probe --push
```

**How it was made, so it can be redone.**

1. `sycl/tools/Dockerfile` - the dev image: the llama.cpp SYCL image plus SYCLomatic (`dpct` 2025.3),
   ninja, and the CUDA 12.8 headers that `sycl/tools/get-cuda-headers.sh` pulls out of NVIDIA's pip wheels
   (dpct parses CUDA; it needs the headers, not the toolkit).
2. `sycl/tools/migrate.sh` - writes a compilation database for the 86 CUDA-touching translation units and
   runs dpct over them. 85 migrate; dpct reports no line it could not migrate, and about 1,400 advisory
   notes.
3. `sycl/tools/fixups.py` - what dpct got wrong or could not do, as an idempotent script with a reason per
   item. The ones that mattered:
   - CUDA's null stream means the default stream; dpct turned it into a null `sycl::queue*`. Every
     stream cast now goes through `strata::q_of()` (`sycl/include/strata/sycl_queue.hpp`).
   - `__ldg((const float*) p)` came out as `*p`, reading one byte of a float scale (two sites, s_gemv).
   - `__fadd_rn(a, b ? c : d)` lost its parentheses (two sites).
   - the ggml lookup tables were threaded through kernel parameters with the wrong table per template;
     they are plain `static const` arrays read from device code, as ggml-sycl does.
   - helper headers renamed since dpct 2025.3 (`entangle`, `chunked_partition`), graph introspection and
     `cudaGraphUpload` (no SYCL equivalents), `%globaltimer` (the stage profiler reads zeros).
4. `sycl/tools/build.sh` - configure + build with icpx inside the image. Two compiler flags are load-bearing:
   `-fp-model=precise` (icpx defaults to a fast FP model) and `-cl-fp32-correctly-rounded-divide-sqrt`
   for the device compiler - the Arc's fp32 divide is not correctly rounded by default (OpenCL allows
   2.5 ulp), and Strata's quantizers are byte-exact against ggml through `amax / 127`. Measured: without
   it `quantize_act_parity` has 303k mismatches, with it none.

**Where it stands (2026-09-29).** The whole tree builds and links: the `strata` binary and 22 kernel
parity tests. On the B70, 19 of the 22 pass byte-for-byte or within their tolerances; the other three
need model fixtures (`iq_parity`, `ple_parity`) or the tensor-core kernel below (`qsa_prompt_attn`).

| parity test | result |
|---|---|
| bf16_gemv, cvec, dequant_s2, elementwise, gdn, gr, kv_q4, kv_q8, kv_stream, qsa, quantize_act, rope, router_top10, s2_gemv, s2_gemv_q8, s_gemv, s_gemv_q8k, sampler, shared_expert | pass |
| iq_parity, ple_parity | `ctest` generates the `iq_parity` fixtures into the build tree (`iq_parity_fixtures`, needs python3 with numpy; skipped without). `ple_parity` needs the Q2_0 shard 2 GGUF and the dense pack: configure with `-DSTRATA_PLE_GGUF=<shard 2 .gguf>` (and `-DSTRATA_PLE_FIXTURE_DIR=<dir with ple_in.bin and ple_out.bin>`); a missing input is a hard error, as in the CUDA build (#1795) |
| qsa_prompt_attn_parity | needs the tensor-core kernel (below) |
| kv_hybrid_parity (0.1.27, hybrid K8V4 KV) | appends, gathers and attention pass; its last step is that same tensor-core kernel |

**The engine end to end (2026-09-29, later the same day).** It generates. Prompt "Write a Python function
that returns the n-th Fibonacci number", greedy, 64 tokens:

> `<think>` The user wants a Python function that returns the n-th Fibonacci number. Let me consider the
> options: 1. **Recursive approach** - Simple but O(2^n) time complexity, very slow for large n. 2.
> **Iterative approach** - O(n) time, O(1) ...

Measured on the same card, same 2,185-token prompt, 64 greedy tokens (AOT build, `--spec 2`,
`--no-prefill-borrow`), against llama.cpp's SYCL build serving the same GGUF:

| | llama.cpp SYCL | Strata SYCL port |
|---|---|---|
| prompt reading | 138-319 tok/s | **560 tok/s** (693 at 2,000 tokens, 180 at 300) |
| decode, suffix drafter only (`--spec 2`) | 24.2-26.0 tok/s | 20.2-20.8 tok/s |
| decode with the MTP draft layer (`--spec 4 --mtp`) | - | **42.7 tok/s** at the 2,185-token context (81% accepted, 2.9 tokens/round); 45.3 on a short prompt |

Prompt reading is where Strata's design pays (oneMKL GEMM over dequantised experts, the whole model on
the card). Decode is at ~80%: a speculative round costs ~53 ms whatever its size and the suffix drafter
is accepted 9% of the time, so nearly every round yields one token; `--spec 4` (16.9 tok/s) was worse
than `--spec 2` (20.8) for that reason. The kernels themselves run at 130-160 GB/s of weights on a card
that streams 600 GB/s (`sycl/probe/bw.cpp`, incompressible data) - the same class llama.cpp reaches -
and five variants of the hot Q6_K matvec (lanes per row, rows per warp, unroll, 16-byte loads, software
pipelining; `mmvq_bench`) all landed in that band. The draft source was the lever: the base
Qwen3.8-Flash-Next checkpoint's MTP layer (`tools/mtp_fetch.py fetch`, `mtp_pack.py --experts q2_0`,
`mtp_rt.py`; 4.9 GB downloaded, 809 MiB of VRAM) drafts for the Coder fine-tune at 87% acceptance:
3.4 tokens per 53 ms round, 45 tok/s, the same greedy tokens. The `mtp.cpp` path ran on the first try
under the same flags.

Without AOT the runtime JIT-compiles every kernel on first use, ~47 s the first time a process runs;
`SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=<dir>` keeps that across runs (15 MB).

How to run it by hand (a greedy test run, the way every number in this section was measured; inside the
`strata-sycl-dev` image, AOT build in `build-sycl-aot/`):

## How to run it by hand

- **Host-mapped flags.** Strata's decode is a GPU/CPU handshake through host-mapped memory. `volatile`
  device loads do not bypass the caches on Intel; system-scope atomics do, and they are the only thing
  measured to work (`sycl/probe/doorbell.cpp`, six variants). Host-to-device visibility *during* a kernel
  stays unreliable on this platform, which is why the all-resident path waits for the window instead.
- **Bounded spins.** A device spin that never sees its flag is not a hang of one process: the xe driver
  times the queue out and resets the GT node by node - a window graph has 2,400 - and the card stays
  wedged until a reboot (twice). Every spin is capped (`kSpinMax`).
- **32-lane sub-groups.** The kernels are written for warps; dpct pinned 134 of 289 launches, the rest
  would run at Xe2's default 16 (`-fsycl-default-sub-group-size=32`).
- **Synchronous copies.** `cudaMemcpy` blocks; dpct's default-queue `memcpy` did not wait. With the
  streaming ring that filled the expert cache from overwritten buffers: non-deterministic residuals, NaN by
  layer 5. Found with the per-layer residual ladder (`STRATA_VERIFY_DEBUG=1` prints R after every layer).
- `native_expert_parity` hand-ported: the GPU native expert kernel matches ggml's float reference on real
  IQ1_M rows (rel 1.1e-2, the same class as the CPU path).

Profiling: `sycl/tools/Dockerfile.unitrace` builds the dev image with Intel's unitrace; `unitrace -d` around
the engine plus `sycl/rank_kernels.py` gives device time per kernel. `mmvq_bench` and `native_expert_parity
NATIVE_BENCH=1` time the two hot kernel families in isolation (warm the clocks first: a 5 ms run measures
the ramp, not the kernel).

**After the 0.1.25-0.1.27 merge (2026-09-30, same card, same flags, AOT).** Fibonacci prompt (19 tokens),
64 greedy tokens: decode 46.5 tok/s (85% of drafts accepted). The 2,184-token prompt: 566 tok/s prompt,
38.3 tok/s decode (77% accepted on that text); the draft layer's prompt pass is now upstream's batched one
(38.6 ms for 2,184 tokens, from 105 ms for 19 before it). Parity: 19 of 22, byte-exact `quantize_act` back.

**Speed work, 2026-09-30 (after the 0.1.27 merge).** What moved the numbers, measured on the B70 with the
same prompts as above (outputs identical before and after each change):

| change | effect |
|---|---|
| every window graph and the drafter's graphs captured at load (`STRATA_WARM_GRAPHS=0`: as before) | the first request no longer pays 250-290 ms of captures |
| the commit graph left running while the drafter's round runs on its own queue (`Verifier::commit(wait=false)` + `commit_finish`) | decode 38.3 -> 43.0 tok/s at the 2,184-token context, 46.5 -> 49.8 short |
| upstream 0.1.29 merged (GDN recurrence pipelining, the block-scores read, sampler) | 2,184-token prompt 765 -> 792 tok/s, decode at that context 43.1 -> 45.2 |
| the draft layer's batched prompt pass takes prompts under 64 rows (it fell back to per-6-token graphs) | its prompt cost on 19 tokens 106 -> 16 ms |
| one device module per kernel (`-fsycl-device-code-split=per_kernel`) | the first launch in the prompt path (the embedding gather) 245 -> 1 ms |
| the expert dequant writes each thread's run of FP16 values as one vector store (it wrote them one 2-byte store at a time) | dequant per expert 0.085 -> 0.030 ms; prompt 496 -> 575 tok/s at 2,184 tokens, 720 -> 841 at 8,000 |
| SWAR sign compare/subtract in the expert dots, a local-memory resident-plan kernel, a split-K fused down kernel (bit-identical to the single-token one) | parity-clean, no measurable decode change; kept |
| the PLE row reader's default queue depth 16 -> 64 blocking O_DIRECT threads (`STRATA_IO_THREADS`; 128 and 256 change nothing: the drive tops out near 85k IOPS on the 27k random 4 KB reads a 2,184-token prompt needs) | the rows of that prompt 466 -> 330 ms |
| a short first prompt chunk (`STRATA_PREFILL_FIRST`, default 256 tokens) so the GPU starts while the remaining rows are still being read | 2,184 tokens: 711 -> 765 tok/s, 8,000: 857 -> 986; time to first token about 150 ms less. The chunk boundary moves rounding in the expert GEMMs, so a long greedy continuation can diverge late (token 45 of 64 on the test prompt) |

Draft policy sweep (2,184-token prompt, 128 tokens): windows of 6 (`--spec 4`) at 41.6-43.5 tok/s; `--spec 2`
33.6, `--spec 6` 35-37; `--spec-min-p` 0.3-0.7 within noise. Profiles (unitrace, `strata-sycl-dev:metrics` with
Intel's metrics libraries and `dev.xe.observation_paranoid=0`): the expert dot kernels are ALU-bound (77% XVE
active), the dense projections memory-latency bound (92-128 GB/s at 84-93% occupancy), a decode round is 80-85%
kernel time and ~5 us of launch gap per node over 2,400-2,600 nodes.

**XMX.** oneMKL's FP16 GEMMs already run on the XMX units (30-60 TFLOP/s in `xmx_gemm_bench`); the prompt path is
bound by the dequant that feeds them, not by the products. Two joint_matrix kernels were written and are correct
but lose to the existing paths on this card, so both stay opt-in: `xmx_gemm_iq` (a fused dequant + GEMM straight
from the quantized rows, 4-5x slower than dequant + oneMKL) and `qsa_prompt_attn_xmx` (the port of the mma.sync
prompt attention; `qsa_prompt_attn_parity` passes at 1e-6 of scale, 3x slower than the FP32 fallback per chunk;
`STRATA_PROMPT_ATTN_XMX=1`). Where the prompt time goes now (2,184 / 8,000 tokens): dequant 28% / 27%, GEMMs
22% / 20%, the per-layer host grouping 9.5% / 9%, attention 7% / 14%, embeddings + PLE rows 13% / 5%.

A trap worth knowing: with a native pack `--prefill-until N` does not feed the rest of the prompt through the
token loop (it is skipped for native packs), so the tokens after N are dropped and the model free-runs. Compare
output tokens between paths, never only timings.

**An 80,000-token prompt (2026-09-30).** Where the time goes changes completely at this scale:

- **The VRAM plan.** `--expert-cache auto` fills the card down to `--vram-reserve-mib` *before* the KV state
  (about 1 GB in INT8 at 81,920 cells) and the prompt chunk buffers (about 2 GB at `--prefill 8192`) exist. With
  the 1,536 MiB default that put 30.95 of the B70's 32.6 GB in use, the driver started migrating buffers and the
  run never finished. `--max-context 81920 --kv int8 --vram-reserve-mib 3072 --prefill 4096` peaks at 28.8 GB.
  (80k ids also exceed Linux's 128 KB single-argument limit: `--tokens-file`.)
- **Streamed experts.** The reserve costs cache slots (10,348 of 12,288 at 3 GB), and the prompt path streams the
  missing experts from the GGUF for every chunk. Two port bugs sat on this path and are fixed: the stream plan
  held `GgufExpertSource::blob()` pointers across ~1,900 reads of a 512-slot ring (so blobs were overwritten
  before they were copied: the 75 s / 1,065 tok/s measured first was computed partly with the wrong experts), and
  the stream-all walk hangs on this card in its first large chunk (the copy engine stops on a barrier). The stager
  threads now read the blobs themselves, and the port uses the per-layer routed-only walk (`STRATA_PREFILL_RING=8`
  upstream, the port's default; `STRATA_PREFILL_STREAM_ALL=1` restores the other for debugging).
- **Correct result: 80,000 tokens in 101 s = 790 tok/s.** GPU time: expert down GEMM 26.0 s, attention 15.4 s,
  dequant 12.7 s, QSA block selection 9.3 s (0.2 s at 8k: it scans every block of the context per query), gate/up
  GEMM 8.9 s, gather 6.1 s, the per-layer grouping sync 5.2 s, GDN recurrence 4.8 s. The PLE rows are free at this
  scale (97.6% row-cache hits).
- **Decode after such a prompt, and the fix.** Without lending cache slots to the prompt path (`--no-prefill-borrow`,
  which the port used from the start) the reserve evicts ~1,900 experts for good and decode after the prompt runs at
  2 tok/s. Borrowing lends ~940 slots to the prompt path and refills them in about a second afterwards: 80,000
  tokens in 75 s = **1,062 tok/s, decode 35-40 tok/s** after it, at `--vram-reserve-mib 2048 --prefill 4096`.
  It costs ~1 s on short prompts (2,184 tokens: 610 vs 792 tok/s), so the port borrows by default only above a
  32K context (`--prefill-borrow` / `--no-prefill-borrow` decide explicitly).

**Long contexts (2026-10-01, engine 0.1.32-sycl, setup's flags: `--kv-resident 32768 --vram-reserve-mib 2048
--prefill 4096`, the prompt path borrowing cache slots - the default above 32K).** Same text repeated to length,
then 256 greedy tokens:

| context | KV | prompt | decode after | peak VRAM | experts in the host mirror |
|---|---|---|---|---|---|
| 128K | int8 | 888 tok/s (144 s) | **66.6 tok/s** | 29.9 GB | none |
| 128K | q4_0 | 848 tok/s (151 s) | 62.8 tok/s | 29.7 GB | none |
| 128K | k8v4, KV in VRAM | **1,006 tok/s** (127 s) | **66.8 tok/s** | 30.7 GB | 103 (0.2 GB) |
| 256K | int8 | 757 tok/s (338 s) | **55.9 tok/s** | 30.2 GB | none |
| 256K | q4_0 | 737 tok/s (347 s) | 51.7 tok/s | 30.0 GB | none |
| 256K | k8v4, KV in VRAM | 823 tok/s (311 s) | 52.2 tok/s | 30.9 GB | 865 (1.7 GB) |

Borrowing keeps every expert in VRAM (the prompt buffers live in lent cache slots, refilled after the prompt).
Without it (`--no-prefill-borrow`) the buffers keep their own VRAM and ~1,000-2,300 experts move to the pinned host
mirror: the prompt is 2-5% faster, decode after it 10-23% slower (0.1.31 without borrowing: 128K int8 934 / 59.5,
q4_0 887 / 53.4, k8v4 1,026 / 54.4; 256K int8 784 / 48.7, q4_0 763 / 43.2, k8v4 838 / 42.3 tok/s).

KV streaming (`--kv-resident`) keeps the whole KV in pinned host memory and only the attended window in VRAM, so
the KV pushes no experts out; setup.py turns it on from 64K up and keeps INT8 (the faster of the
two at every size here). Without it the KV pushes experts out and decode after the prompt fell to 4-9 tok/s at
128K-256K (2026-09-30). `--kv k8v4` does not support streaming yet: setup keeps its KV in VRAM, which pushes up to
more experts out; those come from the pinned host mirror over PCIe (planned item 2), so decode holds at 42 tok/s at
256K (it was 3.8 before the mirror). k8v4 reads prompts fastest (no streaming copies) and works through the FP32
attention fallback (its dedicated prompt kernel is the XMX one below).

**XMX prompt attention v2.** 64-cell chunks, vector-packed K^T and V, hi+lo Q in one accumulator per scale group,
one accumulator update per chunk: 1.4-1.5x faster than v1 and correct in all modes (`qsa_prompt_attn_parity`,
and `kv_hybrid_parity` passes completely with it), but still ~2x slower than the FP32 fallback (13.4 vs 5.5 ms per
chunk, INT8, 32K context). This attention is gather-bound: each query position selects its own ~2,000 cells, so
the K/V fetch dominates and only 12 of the 16 matrix rows are real heads. Opt-in: `STRATA_PROMPT_ATTN_XMX=1`
(64-cell chunks, 120 KB of local memory) or `=32`.

**XMX v2 in the full matrix, and why a grouped kernel is not next (2026-09-30).** With `STRATA_PROMPT_ATTN_XMX=1`
(stage timing on, ~8-10% overhead) the long prompts run 23-33% slower than with the FP32 attention: 128K int8 666 vs
960 tok/s, k8v4 596 vs 983; 256K k8v4 506 vs 752, int8 536 vs 718 (q4_0 does not use the XMX kernel). The obvious
fix, gathering the union of neighbouring positions' cells once, depends on how much their selections overlap.
Measured on the last chunk of an 80K prompt (first QSA layer, `STRATA_DUMP_SEL=<file>`): the union of 8 consecutive
positions is 3.3x one position's 2,051 cells (16: 5.1x), and only 12% of a selection is shared by all 8. Grouping
would cut the K/V gather to ~40% but multiply the arithmetic by 3-5x: at best 5-8 s of an 80K prompt. Not built.

**The grouped kernel, built: prompt attention over 8-query selection unions (`STRATA_PROMPT_ATTN_UNION=1`,
2026-10-16).** `qsa_prompt_attn_union` (`sycl/src/kernels/cuda/qsa_prompt_attn_union.dp.cpp`). For every 8
neighbouring prompt positions a first kernel builds the sorted union of their selected cells, each cell marked with
the positions that selected it. The attention kernel then walks that union in 64-cell chunks per KV head: the INT8 K
and V rows are dequantized once to FP16, S^T = K Q^T and O += P V run on joint_matrix (8x16x16, FP16 inputs, FP32
sums) for all 96 rows of the group (8 positions x 12 heads, every matrix row a real head), and the online softmax
gives a cell -inf in the rows of the positions that did not select it, so each position still attends to exactly
its own cells. The extra arithmetic of the union goes to the matrix units; the K/V gather is done once per group.

- INT8 KV (`--kv int8`) on a Battlemage card. Any other KV format or card, or a failed scratch allocation, keeps
  the FP32 attention (one stderr line says why). The selection (block scores, top-k) and decode are not touched.
- Not bitwise: Q, K, V and the probabilities are rounded to FP16 once each, and the sums run in another order.
  Each position attends to the same cells as with the FP32 kernel, with the same masking of non-resident pages.
- It pays only when neighbouring selections overlap: the union of 8 is 8 selections' work when they share nothing.
- Scratch: about 80 MB of device memory per card that reads a prompt, allocated at the first prompt; it comes out
  of the `--vram-reserve-mib` headroom.
- Check on a card: `qsa_prompt_attn_union_parity --selftest` (also `ctest -R qsa_prompt_attn_union_parity`; skipped
  off Battlemage): against FP64 and the FP32 kernel on random INT8 pools at 1.5K-40K contexts, ragged last groups,
  pages that are not resident, and a bitwise rerun. `qsa_prompt_attn_union_parity <context> <queries> [reps]
  [--drift F] [--sel <STRATA_DUMP_SEL file>]` times one chunk.

Measured 2026-10-16 on 2x Arc Pro B70 (32 GB, `xe`, kernel 7.1, Compute Runtime 26.22.38646.4, oneAPI 2026.1, AOT
`bmg-g31`), Qwen3.8-Flash-Next ISTA GSQ-RCO IQ3_XXS, `--layer-split 30 --prefill 8192 --max-context 65536 --kv int8
--spec 4 --mtp <BF16 draft layer> --mtp-max-t 4 --spec-min-p 0 --stream-experts --expert-cache auto`, one slot,
greedy, fixed prompts (at `--max-context 200000` not every expert fits on a plain two-card layer split):

| | upstream main | `STRATA_PROMPT_ATTN_UNION=1` |
| --- | ---: | ---: |
| prompt, 4K tokens (two fresh prompts) | 993 / 1,189 tok/s | 1,127 / 1,382 tok/s |
| prompt, 32K tokens (two fresh prompts) | 2,004 / 1,956 tok/s | 2,314 / 2,246 tok/s |
| first token of a turn with 40,796 tokens of context | 23.28 s | 20.39 s |
| decode, 512 tokens (3 runs) | 78.7-81.8 tok/s | 79.1-82.9 tok/s |
| one verify window (3 runs) | 39.4-39.7 ms | 39.3-40.0 ms |

With the switch unset the outputs and draft counts are identical to upstream main's. With it set the outputs differ
(rounding), so decode tok/s moves with the draft acceptance; the verify window takes the same time.

- One chunk of 8,192 positions at 24K-32K context (`qsa_prompt_attn_union_parity 32768 8192 5 --drift D`): the FP32
  kernel 137-152 ms, the union kernel 38.7 / 82.3 / 146.6 ms when the union of 8 neighbouring selections is 1.15x /
  2.67x / 5.22x one position's cells. 8-16K: 137-141 ms against 38.1 / 74.3 / 117.3 ms (1.14x / 2.40x / 4.07x).
- `--selftest`: error against FP64 2.6e-3 - 5.0e-3 at an output scale of 3.3-3.6 (the FP32 kernel: 1.7e-6 - 2.6e-6);
  union against FP32 on average 7e-5 of the scale.
- Teacher-forced NLL (`STRATA_LOGPOS`; 8 chats of an ~8,400-token context read by the prompt path and a 1,617-token
  follow-up scored through the decode windows, 12,737 tokens): 2.6234 with the switch unset, 2.5506 with it, 2.6321
  for a control that only changes rounding (`--prefill 4096`, switch unset). Single chats move by up to 0.2
  nats/token in the control as well (the spread over the 8 chats is 0.12 in both), so 8 chats do not resolve a
  difference of this size. The most likely next token agrees with the unset run at 93.2% of the positions (control:
  94.1%).

**Serving the port (2026-09-30).** `serve/server.py --engine strata` runs the SYCL engine unchanged through
`sycl/serve/strata-sycl.sh`, an `exe` that starts the binary inside the oneAPI runtime image with the serve pipes
attached (paths in the config's `args` are the container's, the data root mounted at `/work`). A config:

## Two cards computing each layer's experts together: `--expert-parallel-device` (2026-10-08)

Checked: the 2,744-token repro finished 3 of 3 times after the fix (0 of 6 before); the first-token logits and the tokens are
identical old against new on the A750 (1.4K, 2.5K and 4K prompts) and on the Arc Pro B70 (stock config, and without
`--stream-experts`); speed is unchanged where it did not hang (A750 4K prompt 79.4 against 79.3 tok/s, B70 stock 1,246
against 1,240), and on the B70 without `--stream-experts` the prompt is faster (680-740 against 77-456 tok/s). The
regression test is `sycl/tools/test_prompt_deadlock.sh` (the 2,744-token prompt; it times out on the old build and passes on
the fix).

## Two cards computing each layer's experts together: `--expert-parallel-device` (2026-10-08)

Measured on 2x Arc Pro B70 (32 GB each, `xe`), Flash-Next IQ3_XXS, Ryzen 9 9950X3D, 60 GB RAM, through the server
(`sycl/serve/server_intel.py`, 131,072 context, INT8 KV with `--kv-resident 32768`, `--vram-reserve-mib 2048`, the
default drafting), greedy, thinking off, the same request 3 times (the table: the 2nd and 3rd, the 1st reads the prompt):

| | `--layer-split auto` | `--expert-parallel-device 1` |
|---|---|---|
| where the experts live | layers 0-28 on card 0, 29-47 on card 1, every expert in VRAM | card 0: 283 per layer, card 1: the other 229, every expert in VRAM |
| decode, a BST class (600 tokens) | 94.3 tok/s | **107.9 tok/s** (+14.4%) |
| decode, a 300-word explanation (384 tokens) | 64.2 tok/s | **72.6 tok/s** (+13.1%) |
| output | | byte-identical to the layer split, all 6 requests of each |

`--expert-parallel-device N` uses two cards, card 0 and card N (N is a card number, not a count). Card 0 runs every
layer and card N holds the experts card 0 does not; for each layer the two cards compute their own experts at the same
time, card N writes its rows into card 0's memory, and card 0 combines them as on one card. Only the experts run on
both cards; the rest of each layer runs on card 0 alone. Each layer's experts are dealt to the two cards hottest
first, in pairs alternating 0,1 / 1,0, so both serve about the same share of the routing (in a unitrace of 96 decoded
tokens each card's expert kernels take 160-170 ms of the trace's 1.1 s). The sets are fixed at the start (no prompt
loans, no adaptive swaps).

- Needs `--stream-experts` and every expert in one of the two cards' VRAM; not with `--layer-split`, `--peer-device`
  or `--pipeline-windows`.
- In a server config: `"gpu": [0, 1]`, `"--expert-parallel-device", "1"` in `"args"` and
  `"ONEAPI_DEVICE_SELECTOR": "level_zero:*"` in `"env"` (setup writes the last for a list of cards). The server adds
  no `--layer-split` beside the option. With every expert in VRAM nothing is mirrored in RAM.
- `--batch N` works with it: 4 requests at once give the same outputs as one at a time.
- The prompt path computes card N's experts on card N, with card 0's FP16 expert path (byte-identical rows).
  `STRATA_EP_PREFILL=0` reads them from the GGUF on card 0 instead; `STRATA_EP_MIRROR=1` then mirrors them in pinned
  RAM (18 GB for IQ3_XXS).
- Checks: `STRATA_EP_VERIFY=1` (with `STRATA_EP_MIRROR=1`) has card 0 compute every row itself and compare card N's
  rows with them word for word: no row differed in the 448 windows the three runs reported. `STRATA_EP_SELF=1` runs
  the window on card 0's own rows only; its output is byte-identical to the expert-parallel one.
- Through `--tokens` (no server, 32K context), 512 tokens of the BST prompt: 107.3 tok/s; one card with the other
  experts in the RAM mirror: 50.8 (1,024 tokens).

## Integrated Arc: the Arc 140T (`xe`, 2026-10-12, re-run 2026-10-22)

Measured on the Arc 140T (PCI id 7d51, Xe-LPG+) of a Core Ultra 9 285H with 62 GiB of RAM, Fedora 44 (kernel 7.2.9,
`xe` driver), oneAPI 2026.0 installed on the PC, with the Coder IQ1_M, engine 0.1.43 and again with engine 0.1.47
(Strata 0.2.2) on 2026-10-22. The GPU's memory is the system
RAM: it reports 58 GiB of global memory and no free-memory query.

| | Arc Pro B70 (`xe`, 32 GB) | Arc 140T (`xe`, integrated) |
|---|---|---|
| where the engine runs | in the `strata-sycl-dev` image (`sycl/serve/strata-sycl.sh`) | on the PC's own oneAPI (`sycl/serve/strata-sycl-native.sh`), no Docker |
| build | AOT `bmg-g31` | JIT (`BUILD_DIR=build-sycl bash sycl/tools/build.sh`) |
| experts | `--stream-experts`: from the GGUF into VRAM, the rest in a pinned RAM mirror | the same flags; the cache and the mirror are both in the RAM |
| how big the cache is | the card's free VRAM | `MemAvailable` from `/proc/meminfo` (`STRATA_SYCL_FREE_MIB` overrides it) |
| prompt-slot borrowing | from 32K up | off |
| `SYCL_CACHE_PERSISTENT` | 0 (it crashed on Xe2) | 1 works: the start takes 7 s instead of 79 s from the second start on (0.1.40) |

- **The driver.** Only `xe` was tested. On a kernel that binds the iGPU to `i915`, `i915.force_probe=!7d51 xe.force_probe=7d51`
  on the kernel command line moves it to `xe`.
- **Build.** Install the oneAPI Base Toolkit (2026.0 here, Intel's installer, into `/opt/intel/oneapi`) and the GPU compute runtime of the
  distribution (Fedora 44: `intel-compute-runtime`, `intel-level-zero`, `oneapi-level-zero`, and `cmake`, `ninja-build`). Then, in the checkout,
  `BUILD_DIR=build-sycl bash sycl/tools/build.sh` (it sources `setvars.sh` itself). The build ends with `BUILD EXIT 0`.
- **Free memory.** The 140T has no `ext_intel_free_memory`, also with `ZES_ENABLE_SYSMAN=1`, so the engine got 0 and the
  expert cache refused to start. On a device with `host_unified_memory` it now reads `MemAvailable`. The `xe` driver keeps the
  memory a GPU program freed in its own pool, which `MemAvailable` does not count until a reboot: on the first start after a
  boot all 12,288 experts were cached, on later starts 10,496 to 10,870 (the rest mirrored). `STRATA_SYCL_FREE_MIB` sets the
  figure by hand.
- **No prompt-slot borrowing.** With it, the answer after a prompt that borrowed the cache's slots was token 0 (every expert
  activation NaN from the second prompt chunk; 0.1.40, any context). The prompt now gets its own buffers on a device with
  `host_unified_memory`. Borrowing saved about 7% of prompt time in a test without the bug (112 vs 105 tok/s, 0.1.40).
- **The first decode window stalled** on every request until `STRATA_WINDOW_SYNC=1` (#2110): 10 of 10 in 0.1.43,
  6 of 6 in 0.1.47.
- **What fits.** The experts' file (shard 1) has to fit in the RAM beside everything else. The Coder IQ1_M and the IQ2_XS
  (shard 1 36.5 GiB) ran with 10 to 13 GiB of RAM left; the IQ3_XXS (43.8 GiB) left 3 to 4 GiB and did not run well (0.1.40).
  Programs that use the GPU's memory (another model server) take from the same RAM: stop them first.
- **The launcher.** `sycl/serve/strata-sycl-native.sh` is the config's `exe`. It sources oneAPI, sets the port's defaults
  (`STRATA_VERIFY_DEVICE_PLAN=1`, `STRATA_VERIFY_NO_HOST=1`, `STRATA_STAGER_THREADS=12`, `STRATA_NO_MULTI_GR=1`), each one
  overridden by the config's `"env"` (`0` turns one off), and gives the engine an OOM score of 1000, so that a PC out of
  memory loses the engine and not the desktop. `STRATA_SYCL_BIN` names another binary (relative to the checkout or absolute).

**With setup.** `./setup.sh --backend sycl` finds the integrated GPU in sysfs (an Intel GPU at PCI `00:02.0` on `xe`) and needs
oneAPI in `/opt/intel/oneapi` (or `ONEAPI_ROOT`) and the engine in `build-sycl` (or `build-sycl-aot`); Docker is not used. What
changes from a card:

- **Which size fits.** A model needs its shard 1 plus 20 GiB of RAM (the 14 GiB the measured runs used beside shard 1, and
  6 GiB more). The size menu marks the sizes that need more, recommends the first one that fits, and asks before it installs
  one that does not (`--model X --yes` installs it anyway). `--check` says the same per model.
- **The config.** As the one below: the launcher as `exe`, the PC's own paths, `--stream-experts`, no KV streaming (the KV is in
  the same RAM either way), the VRAM reserve as on a card (1,024 MiB up to 32K, 2,048 above), `--draft-vocab en`, and
  `SYCL_CACHE_PERSISTENT=1` on the 140T only (it crashed on Xe2). An AOT build adds `STRATA_SYCL_BIN`.
- **On `i915`** setup stops and names the kernel command line that moves the GPU to `xe`.

On the 140T, `./setup.sh --backend sycl --family coder --model IQ1_M --context 98304 --no-start --yes` wrote the config below
(less `"fit_max_tokens"`) and `run-coder-iq1_m.sh`.

**Running it by hand.** A config like this, started with
`.venv/bin/python sycl/serve/server_intel.py --engine strata --config strata-coder-iq1_m.json --port 8080`:

    {
     "exe": "<checkout>/sycl/serve/strata-sycl-native.sh",
     "args": ["--pack", "<data>/packs/coder-iq1_m", "--native", "<data>/models/coder-IQ1_M/<shard 1>.gguf",
              "--ple-gguf", "<data>/models/coder-IQ1_M/<shard 2>.gguf", "--expert-profile", "<checkout>/data/expert-profile-coder.bin",
              "--expert-cache", "auto", "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5", "--mtp", "<data>/mtp/rt",
              "--max-context", "98304", "--kv", "int8", "--stream-experts", "--vram-reserve-mib", "2048"],
     "cwd": "<checkout>", "tokenizer": "<data>/packs/coder-iq1_m/tokenizer", "model_name": "qwen3.8-flash-next-coder-iq1_m",
     "log": "<checkout>/strata-coder-iq1_m.log", "port": 8080, "backend": "sycl", "draft_vocab": "en", "fit_max_tokens": true,
     "env": {"SYCL_CACHE_PERSISTENT": "1"}
    }

**Speed** (through the server, greedy, thinking off, with #2110):

| | |
|---|---|
| decode, 100-token answers to short questions, 0.1.43 | 12.0 to 17.4 tok/s all experts cached, 11.5 to 17.2 with 1,792 of them mirrored |
| the same, 0.1.47 (four runs of 10, 1,915 to 2,660 experts mirrored) | 8.4 to 15.2 tok/s; two runs of the same config gave 8.5-13.3 and 10.6-15.2 |
| a 72,648-token prompt, 0.1.43 | read at 156.5 tok/s (7 min 44 s), the fact at its start recalled |
| a 74,939-token prompt, 0.1.47 | read at 149.9 tok/s (8 min 20 s), the fact at its start recalled |
| a 24-token prompt | 1.2 s (0.1.43), 2.3 s (0.1.47) |

The 140T reads 90.5 GB/s from device memory and 76.1 GB/s from host memory (a read probe, 2026-10-05). Where a decode round's
time goes is not profiled yet.

## Not done

- Images on the llama.cpp config: not wired. On the SYCL port they are wired, see "Pictures on an Arc" below. Open there: a
  nobody has run the SYCL picture reader on a card of 24 GB or more. Pictures are on for a Battlemage card too since 2026-10-22 (see "Default" below),
  with the CPU reader.
- Speculative decoding on the llama.cpp path (the GGUF carries no draft layer llama.cpp can use). The SYCL
  port has it (MTP draft layer, `--mtp`).

## Pictures on an Arc (2026-10-19, branch f-021-arcvis)

Pictures follow the rule of every other card ([MODELS.md](MODELS.md#the-versions)): a picture reader on the CPU under 24 GB of VRAM, on
the card from 24 GB. `strata_control/vision_reader.py` is the one place for it. On an Arc:

- **The CPU reader** is `strata-vision` (llama.cpp's mtmd). Setup compiles it into `engine-sycl/`, the server runs it on the host, and the
  engine in the container reads the picture's embeddings from the host's `/dev/shm`, which `strata-sycl.sh` shares
  (`STRATA_SYCL_SHM=0` keeps a private one). On the A750 a 448x448 picture takes 1.73 s.
- **The SYCL reader** is `strata-vision-sycl`, the same program on llama.cpp's SYCL backend (`tools/vision`, `STRATA_VISION_SYCL=ON`).
  `setup --backend sycl --vision gpu` compiles it on the host when the oneAPI compiler (`icx`, `icpx`, `intel-oneapi-mkl-devel`) is
  installed; without it setup says so and keeps the CPU reader. It needs the PC's oneAPI runtime, not the container. If it does not
  start, the server falls back to the CPU reader with one log line. On the A750: 0.16 s per picture instead of 1.73 s (after a 30 s
  first start that compiles the kernels), and the embeddings match the CPU reader's (cosine 0.9959 to 0.9999 over 8 pictures). It takes
  VRAM the experts would use, which is why the rule gives it to 24 GB cards only.
- **Default.** On for every Arc card with the CPU reader, a B-series (Battlemage) card included. An Arc of 24 GB or more also keeps the CPU reader by
  default: the SYCL reader has not run on such a card, so `--vision gpu` asks for it and `vision_reader.INTEL_GPU_READER_BY_DEFAULT` is the switch to
  flip once one has. **B-series cards are supported and untested by us** (the owner decision of 2026-10-22: users will report failures). One known
  risk: the first picture on the Arc Pro B70 (engine 0.1.45, IQ3_S, 2026-10-19, CPU reader) logged an xe engine reset (ccs + bcs) and
  `Fault response: Unsuccessful -EINVAL`, and the request ended 503. The CPU reader was in use, so the fault is in the engine on xe, not in a GPU
  reader; it has not been looked at since. It is not a refusal: setup and `strata launch` print one plain line that the card is new to Strata and
  how to report a problem, and `--vision no` (setup) or `--vision off` (`strata launch`) turns pictures off.

Tested on the A750 (IQ3_XXS, 8K context, engine from f-021-int, JIT build, 2026-10-19): a picture with a red circle, a blue square and the
words "STRATA 42", 8 variants, answered correctly by the CPU reader 10 of 10 (a first set with smaller text read "42" as "412"), and
by the SYCL reader 10 of 10; two pictures in one request, a picture kept in the history of a second turn, and a long text prompt after the
pictures all answered; a text request after every picture answered; 47 picture requests, outside and inside the container (inside, one
answer was cut off by the 200-token limit of the test), no GPU fault in the kernel log, no 503.

## Gyro on an Arc (2026-10-20, branch f-022-arcgyro)

Gyro-S and Gyro-M (trellis-coded experts, [GYRO.md](GYRO.md#intel-arc-sycl)) run on the SYCL engine. Tested on the Arc A750 only (i915, 8 GB,
Ryzen 5 5600X, 62 GB of RAM). Setup writes the usual A750 config for them (32K context, 300 MiB VRAM reserve), without `--ple-io ram`: the
28 GB n-gram table stays on the SSD, because the expert arena already takes 24 GB (Gyro-S) of the RAM.

| A750, Gyro-S | |

## Arc Pro B70: cheaper IQ decodes in the grouped expert kernels (`STRATA_IQ_DECODE_FAST=1`, opt-in, 2026-10-16)

## Arc Pro B70: cheaper IQ decodes in the grouped expert kernels (`STRATA_IQ_DECODE_FAST=1`, opt-in, 2026-10-16)

The decode kernels that read the routed experts (`native_gu_kernel` / `native_down_kernel` in `sycl/src/kernels/cuda/iq_kernels.dp.cpp`,
8 lanes a row) are ALU-bound on this card, not bandwidth-bound: an ISA dump (IGC, AOT `bmg-g31`) of the IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S,
IQ2_S gate/up loops and the Q2_0 / IQ4_NL down loops shows 280-480 instructions per 32-weight part for the part's 8 dp4a, with the kernels
reading experts at 19-44% of DRAM bandwidth. Most of it is the dpct per-byte emulation of `byte_level_permute` in the Q2_0 dot (476 of the
instructions, 157 of them `mov`), the compare/select chain of the IQ4_NL codebook lookup, the popcount/multiply ksigns unpacking and the
64-bit `kbx * sizeof(block)` address products each iteration.

`STRATA_IQ_DECODE_FAST=1` switches the seven formats to kernels that produce the same bits with less of that: the ksigns masks from a
2 KB table (IQ2_XXS, IQ2_XS), the signed grid byte as `(g ^ m) + (m & 0x01010101)` (exact because every byte of the IQ grids is 1..62),
the Q2_0 fields spread by shifts and masks and biased to {-1, 0, 1, 2} in one add and xor, the IQ4_NL codebook through four 256-entry
position tables, a loop that walks block pointers, two rows per lane in 16-lane sub-groups (the one-row SIMD32 form spilled registers and
measured slower), and a multi-entry form for IQ2_XXS (the current kernel decodes it once per entry). Every part yields the same integers to
the same `finish` in the same float order as the current `Multi<TY>` / `vec_dot_*` dots and keeps each entry's lane-strided k order, so the
outputs are bit for bit the current kernels': `native_grouped_parity --iq` (ctest `native_grouped_parity_iq_decode`) compares the two on
random blobs, every IQ gate/up format with both IQ down formats, groups of 1..6 entries, at 512 x 256 and at Flash-Next's 2560 x 640.
`STRATA_IQ_DECODE_FAST_FMTS=16,42` (ggml type ids) restricts it to some formats for an A/B. Off by default; nothing changes without the flag.

Measured 2026-10-16 on a B70 (card 1 of two, xe, oneAPI 2026.1 AOT) with a grouped-call microbenchmark (48 layers per graph, each its
own experts streaming from VRAM, us per call, median of 5 x 3 replays; the shapes are the per-card expert calls of a 2-card layer-split
serve at 1, 2 and 4 concurrent streams, 10 to 32 groups of 1 to 4 entries), new kernels against the current ones, bitwise equal in every row:

| gate/up / down | 1 stream, card 0 / card 1 | 2 streams | 4 streams |
|---|---|---|---|
| IQ2_XXS / Q2_0 | 114.8 -> 59.8 / 135.1 -> 83.8 | 158.7 -> 85.8 / 192.3 -> 121.5 | 213.1 -> 112.3 / 248.7 -> 147.3 |
| IQ3_S / Q2_0 | 105.1 -> 70.3 / 147.9 -> 100.1 | 147.4 -> 99.3 / 215.2 -> 147.2 | 194.8 -> 130.8 / 256.9 -> 175.1 |
| IQ2_S / Q2_0 | 97.0 -> 67.2 / 135.5 -> 91.5 | 136.4 -> 94.5 / 195.5 -> 133.1 | 181.1 -> 123.1 / 237.3 -> 161.9 |
| IQ2_XS / IQ4_NL | (not run) / 135.6 -> 119.6 | 138.1 -> 117.7 / 196.7 -> 169.9 | (not run) / 242.6 -> 205.4 |
| IQ3_XXS / IQ4_NL | (not run) / 146.6 -> 127.9 | 142.7 -> 126.0 / 208.2 -> 183.9 | (not run) / 253.4 -> 221.3 |

The Q2_0 down product alone went 63.7 -> 26.4 us (card 0) and 92.3 -> 35.2 (card 1) at 2 streams, IQ4_NL down 64.9 -> 55.8 and
95.9 -> 80.3. Those numbers were taken on a development tree with other decode changes stacked on it.

End to end on this tree, 2026-10-16: two Arc Pro B70 (32 GB, kernel 7.1, Compute Runtime 26.22.38646.4, oneAPI 2026.1, AOT `bmg-g31`),
Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS, `--layer-split 30 --prefill 8192 --max-context 65536 --kv int8 --spec 4 --mtp <BF16 draft layer>
--mtp-max-t 4 --spec-min-p 0 --stream-experts --expert-cache auto`, one slot, greedy, a fixed prompt with a 512-token coding reply
(`--max-context 200000` does not fit all 24,576 experts on a plain two-card layer split, so 65536). Decode: main 81.8 / 79.5 / 78.7 tok/s
(3 runs, median 79.5); this branch with the flag unset 81.7 / 79.3; with `STRATA_IQ_DECODE_FAST=1` 88.1 / 85.6 / 84.7 (median 85.6,
+7.7%). Output tokens and draft acceptance counts identical in every run, flag on or off, so the gain is kernel time alone. A turn at a
40,796-token context: 76.0 -> 81.4 tok/s (+7.1%), identical output. Prompt reading unchanged (4K 1189 -> 1192 tok/s, 32K 1956-2004 ->
1964-2009). `native_grouped_parity --iq`: 0 failures (15 cases bitwise equal), plain `native_grouped_parity`: 0 failures; per-card health
after every stop passed. Two streams were not tested (main does not take `--batch-mtp` on a layer split).

## Arc Pro B70: KREG decode kernels for Q4_K, Q5_K, Q6_K, IQ4_XS (opt-in `STRATA_MMVQ_KREG=1`, 2026-10-16)

`STRATA_MMVQ_KREG=1` runs `native_mmvq` for Q4_K, Q5_K, Q6_K and IQ4_XS at 1-8 columns (the dense projections, the
verify window and the output head) through KREG kernels. One work-group covers a row's whole K: each lane loads its
32-value slice of every column's activations once, into registers, then the group streams tiles of 2-8 rows (one or
two 16-byte weight loads per lane and row), a sub-group butterfly and a sum across sub-groups in local memory. IQ4_XS
looks its codebook up in a 256-entry local-memory table. The wide kernels it replaces re-read every column's
activations for every row and are bound by instruction issue, not by memory: Q4_K 2560 x 10240 at 6 columns took 82 us
with the matrix held in L2 and 94 us from DRAM.

Not bitwise equal to the default kernels (another summation order). A T-column call is bitwise equal to T one-column
calls, as with the default kernels (`native_mmvq_set_multi_exact`). Unset, nothing on the default path changes: the
flag is the first check of the new branch. Matrices that are not 16-byte (Q4_K, Q5_K) or 8-byte (IQ4_XS) aligned and
rows over 32,768 values keep the default kernels. `mmvq_kreg_parity --selftest` checks it against a float64 reference,
the default kernels and itself one column at a time: on the B70 all 160 cases pass, KREG and the default kernels both
within 5e-8 of sum |w x| of the reference, every multi-column output bitwise equal to the one-column calls.
`mmvq_kreg_parity --bench` times it.

Kernel time, us per call, default -> KREG (`mmvq_kreg_parity --bench`: B70, random weights, distinct copies per graph
so L2 never holds a matrix, median of 7-11 graph replays):

| format, shape (n_in x n_out) | 1 column | 2 | 4 | 6 |
|---|---|---|---|---|
| Q6_K 2560 x 10240 | 55.7 -> 41.0 | 63.0 -> 41.5 | 73.4 -> 46.0 | 88.9 -> 57.7 |
| Q6_K 6144 x 2560 | 47.3 -> 28.1 | 50.3 -> 27.8 | 56.1 -> 30.7 | 66.7 -> 37.6 |
| Q5_K 2560 x 10240 | 49.2 -> 34.7 | 61.3 -> 35.9 | 85.3 -> 40.1 | 113.2 -> 52.0 |
| Q5_K head 2560 x 248320 | 1063 -> 759 | 1374 -> 775 | 2013 -> 893 | 2691 -> 1175 |
| Q4_K 2560 x 10240 | 44.5 -> 28.8 | 55.6 -> 29.6 | 78.8 -> 31.9 | 107.9 -> 40.5 |
| IQ4_XS 2560 x 10240 | 42.9 -> 33.5 | 47.9 -> 37.8 | 55.9 -> 41.2 | 67.6 -> 49.3 |

Over the large shapes (2560 x 6144 / 10240 / 12288, 6144 x 2560, the 248,320-row head) the K-quant kernels stream
74-97% of the card's 608 GB/s at 1-2 columns and 47-66% at 6 (default kernels 31-72% and 18-41%); IQ4_XS 53-81% and
40-50% (default 41-54% and 30-34%), ALU-bound in the codebook decode. The small matrices (2560 x 512 / 640) sit near
the 4-6 us launch floor: K-quants 1.2-2.2x faster, IQ4_XS 0.90-0.98x at 1 column and 1.14-1.34x at 2-6 (KREG takes
every column count so that a multi-column call stays bitwise equal to one-column calls).

Engine, upstream main 296b59d2 against this change: two B70s, kernel 7.1, Compute Runtime 26.22.38646.4, oneAPI
2026.1, AOT bmg-g31; Qwen3.8-Flash-Next ISTA GSQ-RCO IQ3_XXS, `--layer-split 30 --prefill 8192 --max-context 65536
--kv int8 --spec 4 --mtp <BF16 draft layer> --mtp-max-t 4 --spec-min-p 0 --stream-experts --expert-cache auto`, one
slot, greedy, a fixed prompt with a 512-token coding reply, 3 runs each:

| | decode tok/s | tokens per verify window | ms per window |
|---|---|---|---|
| main | 81.8 / 79.5 / 78.7 | 3.22 / 3.14 / 3.12 | 39.4 / 39.5 / 39.7 |
| `STRATA_MMVQ_KREG` unset | 81.9 / 79.4 | the same as main | - |
| `STRATA_MMVQ_KREG=1` | 95.0 / 93.1 / 96.3 | 3.35 / 3.20 / 3.33 | 35.2 / 34.4 / 34.5 |

Median decode 79.5 -> 95.0 tok/s (+19.5%). A verify window costs about 12% less (39.5 -> 34.5 ms), the part that does
not depend on the drafts; the rest is the changed output drafting a little better on this prompt. With the flag unset
the output and the draft counts are identical to main (both runs, and a turn at a 40,796-token context); with it the
output differs from main. At a 40,796-token context: 76.0 -> 83.4 tok/s. Prompt reading is unchanged (4K 1,189 ->
1,191 tok/s, 32K 1,956-2,004 -> 1,961-2,008). Not measured: two streams (main refuses `--batch-mtp` on a layer split),
`--max-context 200000` (a plain two-card layer split of main does not fit all 24,576 experts there: 710 refused).
Measured on the B70 only; the A-series cards are not tested.

## Arc Pro B70: decode QSA selection and attention in fewer, shorter launches (opt-in, 2026-10-16)

Four switches, off by default, each giving the same bits as the path it replaces. Kernel times: `qsa_verify_batch_parity --bench`
on one B70 (synthetic layers: 24 q heads, INT8 K/V, page size 4, a 200,000-cell capacity; graph replay, us per call or per attention
layer), switch off -> on, at 1K / 16K / 64K of context.

- `STRATA_BATCH_QSA_ATTN=1`: a batch window (`--batch`) ran the QSA scores, top-k and attention one row at a time, because each slot has its
  own K/V. Now the scores run once per slot, the top-k once for the window, and the attention once for the window over every slot's pools
  (`sycl/include/strata/qsa_decode_attn_multi.hpp`). Selection + attention per layer, 2 slots x 1 row: 102.8 -> 62.3 / 184.0 -> 107.9 /
  252.6 -> 156.2; 2 slots x 2 rows: 183.6 -> 78.3 / 365.0 -> 138.2 / 498.2 -> 205.0. On a layer split a window holds one row per slot
  (`--batch-mtp` runs on one GPU only), and more than one slot only with `--batch-groups 1`. A streamed K/V keeps the per-row path.
- `STRATA_QSA_TOPK_SCAN=1`: the wide top-k kernel (a capacity above ~135K cells, e.g. `--max-context 200000`) searched each radix pass's
  digit with one thread over 256 bins; now a work-group scan. Integers only, the same ids. One query at 16K / 64K: 36.3 -> 19.5 /
  57.9 -> 42.1 (at 1K every cell is selected and there is no search).
- `STRATA_QSA_SCORES_TRANSPOSED=1`: the decode block scores reduce a block's (query, head) sums together instead of one butterfly each
  (16-31 shuffles per block instead of 20 per query); the same pairs are added in the same order. Four queries: 5.5 -> 4.6 / 18.7 -> 14.1 /
  63.1 -> 43.2.
- `STRATA_ATTN_MERGE_ACTIVE=1`: the attention merge of a call of 8 rows or fewer (verify windows, the MTP drafter) folds only the chunks
  its selection reaches, its loads issued 8 at a time; the chunks past it hold no cells and were skipped anyway. The MTP drafter's call
  (one row, 32K window = 512 chunks): 86.9 -> 41.0 / 223.7 -> 157.2 / 460.9 -> 335.9.

End to end, main 296b59d2 against this change with all four on, 2x B70 (Compute Runtime 26.22, oneAPI 2026.1, AOT bmg-g31),
Flash-Next IQ3_XXS, `--layer-split 30 --kv int8 --spec 4 --mtp` (BF16 draft layer), greedy, fixed prompts: one stream at
`--max-context 65536` 78.7-81.8 -> 79.1-82.2 tok/s (+0.6%), a turn on a 40,796-token context 76.0 -> 77.1 (+1.4%); two streams
(`"parallel": 2`, `--batch-groups 1`, `--max-context 16384`; no MTP in the slots on a layer split) 57.8 -> 59.0 tok/s together (+2%,
medians of 3). The outputs (and the draft counts) are identical to main in every arm, and identical to main with the switches unset;
prompt reading is unchanged. At 65536 cells the wide top-k is not used, so `STRATA_QSA_TOPK_SCAN` did not act in these runs.

Check on a card: `ctest -R qsa_verify_batch_parity` (runs it without the switches, then with the three kernel switches against the first
run's output hashes), or `qsa_verify_batch_parity --bench --ctx 1024,16384,65536` for the times above.

## `STRATA_HC_Q8=1` on SYCL: the Q8_0 hyper-connection read (2026-10-16)

`STRATA_HC_Q8=1` (docs/QUANTS.md: the hyper-connection down / up projections read as Q8_0 instead of BF16, an opt-in
on CUDA, the default on AMD gfx12) did nothing on the SYCL port: the loader kept a Q8_0 copy only when the GGUF
stored the projections as Q8_0 (the Unsloth files; the ISTA files store them as BF16, so nothing was copied and the
log said `0.00 GiB`), and the port's `fused_gr_read_multi` never had the Q8_0 read - it ignored `q8_down` / `q8_up`
and read the BF16 copies either way. The flag cost VRAM on the Unsloth files and changed nothing on any file.

Now, with the flag, the loader quantizes BF16 projections to Q8_0 at load (ggml's rounding, `d = amax / 127` as fp16,
on the host; a GGUF's own Q8_0 blocks are taken as they are) and the read uses them: two kernels, one row per
16-lane sub-group lane, the per-stream norm and the partial sums folded into the down kernel (the last work-group of
a row group to arrive finishes it, so no co-residency is assumed), K split over the sub-groups in the up kernel.
The inject rows stay BF16 (`STRATA_HC_Q8_INJECT` has no effect here). The default (flag unset) is the same code as
before: the Q8_0 pointers are null and the hook returns to the BF16 read; outputs and draft counts were identical to
`main` on the fixed prompts below.

**VRAM.** The BF16 copies stay in the arena (the prompt path's GEMMs read them), so on this port the flag ADDS the
Q8_0 bytes instead of halving them as on a single CUDA GPU: 0.63 GiB for Flash-Next (48 layers x 4 projections + the
final mixer, 3.32 MiB each; on the two-card layer split below 0.40 + 0.24 GiB), which comes out of the expert cache.
On 2x B70 with `--layer-split 30` the model that ran at `--max-context 65536` without the flag refused 310 experts
with it (132 at 32768 with the default 1,024 MiB reserve); the comparison below ran at `--max-context 32768
--vram-reserve-mib 512`. Dropping the BF16 copies (CUDA's `add_hc_skip` / `hc_bf16_view`) is not ported.

**What changes in the output.** The Q8_0 read is NOT bitwise the BF16 read - it reads Q8_0-rounded weights and sums in
another order, as `STRATA_HC_Q8=1` does on CUDA (`fused_gr.cu`, "an output-changing step"). `gr_q8_parity
--selftest` checks it against an FP64 reference computed from the very Q8_0 bytes it reads (T = 1..8, with and
without the pending write and the inject rows): lo / rs / inject / mixed within FP32 summation rounding (1e-4 of the
largest value), R_out bitwise the BF16 read's, a captured graph replayed twice bitwise the direct call, arguments not
at the callers' fixed strides fall back to the BF16 read bitwise; 33 of 33 cases pass on the B70. On the model, the
teacher-forced NLL of 4,762 tokens of three texts through the decode windows (same targets, same binary) was 3.1974
with the flag unset and 3.1699 with it set: no degradation measured on this gate (the gate's own spread between
rounding-only variants is of the order of 0.01-0.1 nats per token, so this is not a measured improvement either).

**Speed** (2x Arc Pro B70 32 GB, kernel 7.1, Compute Runtime 26.22.38646.4, oneAPI 2026.1, AOT bmg-g31; Flash-Next
ISTA GSQ-RCO IQ3_XXS, `--layer-split 30 --prefill 8192 --kv int8 --spec 4 --mtp <BF16 draft layer> --mtp-max-t 4
--spec-min-p 0 --stream-experts --expert-cache auto --max-context 32768 --vram-reserve-mib 512`, one slot, greedy,
a fixed prompt with a 512-token coding reply, 3 runs, same binary, flag unset vs set): the verify window 39.4 / 39.6 /
39.8 -> 37.4 / 37.5 / 37.4 ms (-5.3%, independent of draft acceptance); decode 81.8 / 79.4 / 78.5 -> 85.0 / 88.0 /
86.0 tok/s (median 79.4 -> 86.0, +8.3%, part of it a different acceptance of the changed output: drafts accepted
353/474, 349/489, 348/492 -> 351/489, 357/471, 353/483); prompt reading unchanged (4K 1,093 / 1,190 -> 1,093 / 1,188
tok/s, 16K 1,774 / 1,749 -> 1,769 / 1,750). Kernel level (one card, graph replay, random weights not L2-resident, us per
read, BF16 -> Q8_0): T = 1 38.7 -> 24.7, T = 4 50.8 -> 29.3, T = 6 62.0 -> 34.4. The down kernel's shape (4 rows per
lane x 32 columns per sub-group) was the fastest of (1..8 rows) x (32..160 columns) tried.

Switches: `STRATA_HC_Q8=1` on (unset or `0`: off, the BF16 read as before).

## `STRATA_GR_FUSED=1` on SYCL: the two-kernel BF16 hyper-connection read (2026-10-16, opt-in)

The port's default hyper-connection read is four launches per layer half (the norm per token and stream, the sliced
down projection, its reduce, the up projection); at T = 6 rows they take 5.8 + 22.7 + 2.7 + 30.2 us on the B70, and
the down / up kernels grow ~1.7 / 3 us per extra row - latency and issue, not DRAM (the weights are 13.2 MB a read).
`STRATA_GR_FUSED=1` runs the BF16 weights through the structure of the Q8_0 read above: one row per 16-lane
sub-group lane with the activations uniform across the sub-group, the norm and the partial reduction folded into the
down kernel (the last work-group of a row group to arrive finishes it; nothing waits), K split over the sub-groups in
the up kernel. Two launches instead of four. The weights are the same BF16 bytes; no extra VRAM. With `STRATA_HC_Q8=1`
the Q8_0 read runs instead (it has the pointers; this flag then changes nothing). Unset, the default read runs
unchanged: outputs and draft counts identical to `main` on the fixed prompts below.

**What changes in the output.** NOT bitwise the default read. The per-element arithmetic is the default read's (the
same fma chains, the same epilogue: R_out, the pending write, IS bitwise), but the down projection and the up
projection are summed in another order (slabs of 1,280 columns, 16 sub-group partials; K quarters), so lo, rs and
mixed differ at FP32 rounding level - the same kind of change as upstream's opt-in `STRATA_GR_V3` read.
`gr_fused_parity --selftest` checks it (T = 1..8, with and without the pending write and the inject rows): both reads
within 1e-4 of the largest value of an FP64 reference and of each other, R_out bitwise, a captured graph replayed
twice bitwise the direct call, arguments not at the callers' fixed strides fall back to the default read bitwise; 33
of 33 cases pass on the B70. Greedy outputs on the model differ from the default read's after a near-tie, as every
summation-order change does.

**Speed** (2x Arc Pro B70 32 GB, kernel 7.1, Compute Runtime 26.22.38646.4, oneAPI 2026.1, AOT bmg-g31; Flash-Next
ISTA GSQ-RCO IQ3_XXS, `--layer-split 30 --prefill 8192 --kv int8 --spec 4 --mtp <BF16 draft layer> --mtp-max-t 4
--spec-min-p 0 --stream-experts --expert-cache auto --max-context 65536`, one slot, greedy, a fixed prompt with a
512-token coding reply, 3 runs, `main` vs this read): the verify window 39.4 / 39.5 / 39.7 -> 38.8 / 38.1 / 38.7 ms
(-2.5%, independent of draft acceptance - the effect of the read); decode 81.8 / 79.5 / 78.7 -> 86.8 / 79.0 / 88.3
tok/s, whose spread is mostly a different draft acceptance of the changed output (drafts accepted 360/462, 342/510,
362/456); a turn at a 40,796-token context 76.0 -> 79.4 tok/s (output differs); prompt reading unchanged (4K 993 ->
986 tok/s, first run). Kernel level (one card, graph replay, random weights not L2-resident, us per read, default ->
two-kernel): T = 1 38.7 -> 34.8 (-10%), T = 4 50.8 -> 41.5 (-18%), T = 6 62.0 -> 47.1 (-24%). Of the down kernel's
shapes (rows per lane x columns per sub-group) at T = 6, (1, 80) 30.6, (2, 40) 26.9, (1, 40) 30.2, (2, 80) 23.6 us:
(2, 80) is the one built.

Switches: `STRATA_GR_FUSED=1` on (unset or `0`: off, the default read, byte-identical).

## Measured, 2026-10-01, Arc Pro B70, Coder IQ1_M, 32K context, INT8 KV: the SYCL port (engine 0.1.31-sycl)

## Arc Pro B70: QSA prompt attention per cell and block scores as GEMM tiles (2026-10-19)

The B70's int8 expert kernels also build and pass the parity test as a SPIR-V JIT build and as an AOT build for `bmg-g21` (compile only: no
G21 card here); an AOT build for another device (Alchemist, Xe-HPC) gets stubs (`STRATA_I8_NO_ESIMD`).

## Arc Pro B70: QSA prompt attention per cell and block scores as GEMM tiles (2026-10-19)

Two prompt-path kernels that grow with the context. On a 40K prompt (Coder IQ1_M, 256K config) the QSA attention was 27.6%
of the GPU timeline (7.3 s) and the block selection 8.0% (2.1 s).

- **Attention scores one work-item per cell** (`qsa_prompt_attn_percell`, prompt path only; `STRATA_ATTN_PERCELL=0` off): a
  work-item holds one cell's key row and all 12 heads' dot products instead of a sub-group splitting the 256 dimensions with a
  shuffle tree. 2.0-2.2x the warp kernel in `qsa_prompt_attn_parity` (int8 / FP16 pools; 1.2-1.4x on Q4_0).
- **Block scores as oneMKL FP32 GEMM tiles** (pooled keys x the batch's indexer queries, then a relu-sum kernel;
  `qsa_block_scores_reduce`). The tile lives in the attention scratch, so it costs no VRAM. It loses to the warp kernel below
  ~8K cells (oneMKL's launch: -2% at 2K, even at 8K), so it runs only for a batch reaching 2,048 blocks; `STRATA_SELECT_GEMM=0`
  never, `=1` always. Selection at 40K 2.1 -> 0.7 s.

Both are FP32 in another summation order, not bitwise. First-token logits against the old kernels (whole vocabulary): KL
3.9e-10 nats at 8K and 1.3e-11 at 40K, top-1 equal (upstream's int8 against FP16 prompt experts, the same prompts: 8.4e-10 and
5.1e-12); the 256 greedy tokens after a 128K prompt are identical.

`--pcie-frac 0 --adapt-every 0 STRATA_PREFILL_MMQ=1`, 256 greedy tokens, interleaved runs, prefill tok/s (old -> new):

| model | 2,185 | 40,000 | 128,000 |
|---|---|---|---|
| Coder IQ1_M, 256K config | 1,329 -> 1,468 (+10%) | 1,562 -> 1,885 (+21%) | 1,336 -> 1,798 (+35%) |
| Flash-Next IQ2_XS, 128K config | | 1,373 -> 1,613 (+18%) | |

## Two cards computing each layer's experts together: `--expert-parallel-device` (2026-10-08)

Measured on 2x Arc Pro B70 (32 GB each, `xe`), Flash-Next IQ3_XXS, Ryzen 9 9950X3D, 60 GB RAM, through the server
(`sycl/serve/server_intel.py`, 131,072 context, INT8 KV with `--kv-resident 32768`, `--vram-reserve-mib 2048`, the
default drafting), greedy, thinking off, the same request 3 times (the table: the 2nd and 3rd, the 1st reads the prompt):

| | `--layer-split auto` | `--expert-parallel-device 1` |
|---|---|---|
| where the experts live | layers 0-28 on card 0, 29-47 on card 1, every expert in VRAM | card 0: 283 per layer, card 1: the other 229, every expert in VRAM |
| decode, a BST class (600 tokens) | 94.3 tok/s | **107.9 tok/s** (+14.4%) |
| decode, a 300-word explanation (384 tokens) | 64.2 tok/s | **72.6 tok/s** (+13.1%) |
| output | | byte-identical to the layer split, all 6 requests of each |

`--expert-parallel-device N` uses two cards, card 0 and card N (N is a card number, not a count). Card 0 runs every
layer and card N holds the experts card 0 does not; for each layer the two cards compute their own experts at the same
time, card N writes its rows into card 0's memory, and card 0 combines them as on one card. Only the experts run on
both cards; the rest of each layer runs on card 0 alone. Each layer's experts are dealt to the two cards hottest
first, in pairs alternating 0,1 / 1,0, so both serve about the same share of the routing (in a unitrace of 96 decoded
tokens each card's expert kernels take 160-170 ms of the trace's 1.1 s). The sets are fixed at the start (no prompt
loans, no adaptive swaps).

- Needs `--stream-experts` and every expert in one of the two cards' VRAM; not with `--layer-split`, `--peer-device`
  or `--pipeline-windows`.
- In a server config: `"gpu": [0, 1]`, `"--expert-parallel-device", "1"` in `"args"` and
  `"ONEAPI_DEVICE_SELECTOR": "level_zero:*"` in `"env"` (setup writes the last for a list of cards). The server adds
  no `--layer-split` beside the option. With every expert in VRAM nothing is mirrored in RAM.
- `--batch N` works with it: 4 requests at once give the same outputs as one at a time.
- The prompt path computes card N's experts on card N, with card 0's FP16 expert path (byte-identical rows).
  `STRATA_EP_PREFILL=0` reads them from the GGUF on card 0 instead; `STRATA_EP_MIRROR=1` then mirrors them in pinned
  RAM (18 GB for IQ3_XXS).
- Checks: `STRATA_EP_VERIFY=1` (with `STRATA_EP_MIRROR=1`) has card 0 compute every row itself and compare card N's
  rows with them word for word: no row differed in the 448 windows the three runs reported. `STRATA_EP_SELF=1` runs
  the window on card 0's own rows only; its output is byte-identical to the expert-parallel one.
- Through `--tokens` (no server, 32K context), 512 tokens of the BST prompt: 107.3 tok/s; one card with the other
  experts in the RAM mirror: 50.8 (1,024 tokens).

## Integrated Arc: the Arc 140T (`xe`, 2026-10-12, re-run 2026-10-22)

Measured on the Arc 140T (PCI id 7d51, Xe-LPG+) of a Core Ultra 9 285H with 62 GiB of RAM, Fedora 44 (kernel 7.2.9,
`xe` driver), oneAPI 2026.0 installed on the PC, with the Coder IQ1_M, engine 0.1.43 and again with engine 0.1.47
(Strata 0.2.2) on 2026-10-22. The GPU's memory is the system
RAM: it reports 58 GiB of global memory and no free-memory query.

| | Arc Pro B70 (`xe`, 32 GB) | Arc 140T (`xe`, integrated) |
|---|---|---|
| where the engine runs | in the `strata-sycl-dev` image (`sycl/serve/strata-sycl.sh`) | on the PC's own oneAPI (`sycl/serve/strata-sycl-native.sh`), no Docker |
| build | AOT `bmg-g31` | JIT (`BUILD_DIR=build-sycl bash sycl/tools/build.sh`) |
| experts | `--stream-experts`: from the GGUF into VRAM, the rest in a pinned RAM mirror | the same flags; the cache and the mirror are both in the RAM |
| how big the cache is | the card's free VRAM | `MemAvailable` from `/proc/meminfo` (`STRATA_SYCL_FREE_MIB` overrides it) |
| prompt-slot borrowing | from 32K up | off |
| `SYCL_CACHE_PERSISTENT` | 0 (it crashed on Xe2) | 1 works: the start takes 7 s instead of 79 s from the second start on (0.1.40) |

- **The driver.** Only `xe` was tested. On a kernel that binds the iGPU to `i915`, `i915.force_probe=!7d51 xe.force_probe=7d51`
  on the kernel command line moves it to `xe`.
- **Build.** Install the oneAPI Base Toolkit (2026.0 here, Intel's installer, into `/opt/intel/oneapi`) and the GPU compute runtime of the
  distribution (Fedora 44: `intel-compute-runtime`, `intel-level-zero`, `oneapi-level-zero`, and `cmake`, `ninja-build`). Then, in the checkout,
  `BUILD_DIR=build-sycl bash sycl/tools/build.sh` (it sources `setvars.sh` itself). The build ends with `BUILD EXIT 0`.
- **Free memory.** The 140T has no `ext_intel_free_memory`, also with `ZES_ENABLE_SYSMAN=1`, so the engine got 0 and the
  expert cache refused to start. On a device with `host_unified_memory` it now reads `MemAvailable`. The `xe` driver keeps the
  memory a GPU program freed in its own pool, which `MemAvailable` does not count until a reboot: on the first start after a
  boot all 12,288 experts were cached, on later starts 10,496 to 10,870 (the rest mirrored). `STRATA_SYCL_FREE_MIB` sets the
  figure by hand.
- **No prompt-slot borrowing.** With it, the answer after a prompt that borrowed the cache's slots was token 0 (every expert
  activation NaN from the second prompt chunk; 0.1.40, any context). The prompt now gets its own buffers on a device with
  `host_unified_memory`. Borrowing saved about 7% of prompt time in a test without the bug (112 vs 105 tok/s, 0.1.40).
- **The first decode window stalled** on every request until `STRATA_WINDOW_SYNC=1` (#2110): 10 of 10 in 0.1.43,
  6 of 6 in 0.1.47.
- **What fits.** The experts' file (shard 1) has to fit in the RAM beside everything else. The Coder IQ1_M and the IQ2_XS
  (shard 1 36.5 GiB) ran with 10 to 13 GiB of RAM left; the IQ3_XXS (43.8 GiB) left 3 to 4 GiB and did not run well (0.1.40).
  Programs that use the GPU's memory (another model server) take from the same RAM: stop them first.
- **The launcher.** `sycl/serve/strata-sycl-native.sh` is the config's `exe`. It sources oneAPI, sets the port's defaults
  (`STRATA_VERIFY_DEVICE_PLAN=1`, `STRATA_VERIFY_NO_HOST=1`, `STRATA_STAGER_THREADS=12`, `STRATA_NO_MULTI_GR=1`), each one
  overridden by the config's `"env"` (`0` turns one off), and gives the engine an OOM score of 1000, so that a PC out of
  memory loses the engine and not the desktop. `STRATA_SYCL_BIN` names another binary (relative to the checkout or absolute).

**Running it by hand.** A config like this, started with
`.venv/bin/python sycl/serve/server_intel.py --engine strata --config strata-coder-iq1_m.json --port 8080`:

    {
     "exe": "<checkout>/sycl/serve/strata-sycl-native.sh",
     "args": ["--pack", "<data>/packs/coder-iq1_m", "--native", "<data>/models/coder-IQ1_M/<shard 1>.gguf",
              "--ple-gguf", "<data>/models/coder-IQ1_M/<shard 2>.gguf", "--expert-profile", "<checkout>/data/expert-profile-coder.bin",
              "--expert-cache", "auto", "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5", "--mtp", "<data>/mtp/rt",
              "--max-context", "98304", "--kv", "int8", "--stream-experts", "--vram-reserve-mib", "2048"],
     "cwd": "<checkout>", "tokenizer": "<data>/packs/coder-iq1_m/tokenizer", "model_name": "qwen3.8-flash-next-coder-iq1_m",
     "log": "<checkout>/strata-coder-iq1_m.log", "port": 8080, "backend": "sycl", "draft_vocab": "en", "fit_max_tokens": true,
     "env": {"SYCL_CACHE_PERSISTENT": "1"}
    }

**Speed** (through the server, greedy, thinking off, with #2110):

| | |
|---|---|
| decode, 100-token answers to short questions, 0.1.43 | 12.0 to 17.4 tok/s all experts cached, 11.5 to 17.2 with 1,792 of them mirrored |
| the same, 0.1.47 (four runs of 10, 1,915 to 2,660 experts mirrored) | 8.4 to 15.2 tok/s; two runs of the same config gave 8.5-13.3 and 10.6-15.2 |
| a 72,648-token prompt, 0.1.43 | read at 156.5 tok/s (7 min 44 s), the fact at its start recalled |
| a 74,939-token prompt, 0.1.47 | read at 149.9 tok/s (8 min 20 s), the fact at its start recalled |
| a 24-token prompt | 1.2 s (0.1.43), 2.3 s (0.1.47) |

The 140T reads 90.5 GB/s from device memory and 76.1 GB/s from host memory (a read probe, 2026-10-05). Where a decode round's
time goes is not profiled yet.

