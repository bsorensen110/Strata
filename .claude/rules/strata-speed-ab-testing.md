---
name: strata-speed-ab-testing
description: "Use when running Strata A/B or decode/prefill speed tests."
version: 0.1.0
author: Brooks (bsorensen110), Hermes Agent
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [strata, benchmarking, ab-testing, dflash2, mtp, hip, rdna4, gfx1201]
---

# Strata Speed & A/B Testing Methodology

Standardized methodology for running decode, prefill, batch/concurrent, drafter (DFlash2 vs. MTP), and kernel microbenchmark A/B tests on Strata (`gfx1201` / AMD Radeon AI PRO R9700). Use this exact protocol in both Hermes and Claude Code so numbers are directly comparable across sessions.

## When to Use

- Benchmarking Strata decode (`tok/s`) or prefill (`tok/s`) after a config, env, kernel, or drafter change.
- Running A/B comparisons between two engine binaries (`$HOME/Strata-integ/build-verify/strata` vs. `$HOME/Strata/engine/strata`), env flags, or drafter modes (`dflash2` vs. `mtp`).
- Measuring solo greedy code, solo sampled prose, or 4-slot batch/concurrent throughput.

## Core Rules (Non-Negotiable)

1. **Single-GPU Exclusivity (`/dev/kfd` & RAM):** One engine instance owns ~31 GiB VRAM and the ~47 GiB host expert arena. Never start a second full-config engine while `strata.service` is active. Check `systemctl --user is-active strata.service` and `fuser -v /dev/kfd` before standalone runs.
2. **Escape the 4 GiB Cgroup Cap:** Hermes and subagent background shells cap `memory.max=4 GiB`. Always launch standalone engines, CMake builds, or scratch servers under `systemd-run --user --scope -- ...`.
3. **Separate Cold Trial 1 from Warm Trial 2+:** With `--resident-experts` and `--expert-cache auto`, Trial 1 after engine start swaps ~1,200–1,400 working-set experts from RAM into VRAM (~98.8% hit). Trials 2+ run at steady state (~99.7% hit, +5–15 tok/s faster). Always run **at least 2 trials** for greedy code and **3 distinct prompts** for sampled prose; report Cold Trial 1 and Warm Trial 2+ separately.
4. **Authoritative Metric = Engine Log Line:** HTTP wall time includes prefill and HTTP framing. Always parse the engine's own log line from `~/Strata/strata-swift-iq3_s.log` (or the scratch log) starting from the byte offset recorded before the run:
   `strata serve: prompt P tokens = R reused + W read in Tp ms (Sp tok/s), G generated in Tg ms (Sg tok/s), drafts accepted A of D`
   Report **both** engine decode `Sg tok/s` and HTTP wall `tok/s`, plus `drafts accepted A of D` (`A/D` % and `1 + A / windows` tokens/window).
5. **Disable Reasoning on Throughput Probes:** Always pass `"chat_template_kwargs": {"enable_thinking": False}` so thinking tokens do not consume `max_tokens` or alter stop behavior.
6. **Secret Hygiene:** Read `api_key` in Python via `json.load(open(os.path.expanduser('~/Strata/strata-swift-iq3_s.json')))['api_key']`. Never echo or log the key. Any scratch config containing `api_key` must be `chmod 600`.

## Environment Flags & Drafter Modes (`gfx1201` / RDNA4)

### Production Fast-Path Env Set (`GFX12_SWITCHES`)
Keep these identical across control and treatment arms unless one is the variable under test:
```json
{
  "STRATA_ATTN_LANECELL": "1",
  "STRATA_ATTN_MERGE_V2": "1",
  "STRATA_DFLASH_HEAD_IL": "1",
  "STRATA_DF_TELEMETRY": "1",
  "STRATA_EXPDEC": "1",
  "STRATA_GR_TSUM": "1",
  "STRATA_TSUM": "0",
  "STRATA_HIPBLASLT_TUNING": "~/Strata/tools/hip/gfx1201-hipblaslt-100500.txt",
  "STRATA_HIP_WMMA": "1",
  "STRATA_PF_FUSED": "1",
  "STRATA_SELECT_WMMA": "1"
}
```
- **Why `STRATA_GR_TSUM=1` + `STRATA_TSUM=0`:** `STRATA_TSUM=1` globally enables tree-sum in `native_mmvq.cu` and `native_bf16.cu`, which regresses dense GEMVs on `gfx1201`. Keep `STRATA_TSUM=0` (or use `STRATA_MMVQ_TSUM=1` only when testing MMVQ specifically) and scope tree-sum to `fused_gr.cu` via `STRATA_GR_TSUM=1`.
- **Opt-in A/B switches:**
  - `STRATA_DFLASH_HEAD_IL=0|1`: toggles interleaved `Q8_1` (`native_mmvq_il`) for the DFlash2 draft head (auto-enabled on supported quant types).
  - `STRATA_Q6_PACKED=0|1`: packs `Q6_K` draft-head rows into 32-byte aligned `Q6KPacked` blocks at bind time.
  - `STRATA_DF2_WAITCOMMIT=1`: forces synchronous `cudaStreamSynchronize` after batch commit launch (use to A/B async commit overlap vs. sync commit).
  - `STRATA_DECODE_TIMING=1` / `STRATA_PREFILL_TIMING=1`: per-phase GPU timing tables in the engine log. Keep identical across A and B arms.

### Drafter Configurations

- **DFlash2 (Production Default on `swift-1.5-iq3_s`):**
  - Required flags: `--draft dflash2 --draft-model $HOME/Strata-data/dflash2/qwen3.8-flash-next-dflash2.pack --draft-cap 6 --spec 8 --spec-min-p 0.5 --dflash-no-mtp --mtp-draft-vocab $HOME/Strata-data/mtp/rt/draft_vocab.bin --expert-profile $HOME/Strata/data/expert-profile-learned.bin`
  - Batch flags (`"parallel": 4`): `--batch-draft-window 16 --batch-draft-depth 1` (gives 4 slots × 1 anchor + 3 proposal rows = 16 rows/window; `--draft-cap 6` gives 5 proposals/window in solo and up to 3 in 4-slot batch).
  - **Critical pitfall:** Never omit `--mtp-draft-vocab` when `--dflash-no-mtp` is set (though newer builds auto-discover sibling `draft_vocab.bin`): falling back to the unpruned 248,320-token head instead of the 106,299-token pruned head adds ~1.9 ms/window and costs ~25 tok/s.
- **MTP (Comparison Arms Only):**
  - Solo MTP: remove `--draft dflash2` and `--dflash-no-mtp`; pass `--mtp $HOME/Strata-data/mtp/rt/...`.
  - Batch MTP (`--batch 4`): add `--batch-mtp` (never combine `--batch-mtp` with `--draft dflash2`). Optional batch MTP A/B env knobs: `STRATA_BATCH_DRAFT_TWOPHASE=1`, `STRATA_BATCH_DRAFT_GROUP=1`.

## Canonical Benchmark Script (Greedy Code + 3-Prompt Sampled Prose)

Run against the live server or a scratch server port using `$HOME/Strata/.venv/bin/python`:

```bash
$HOME/Strata/.venv/bin/python - <<'PY'
import json, os, time, urllib.request

cfg_path = os.path.expanduser('~/Strata/strata-swift-iq3_s.json')
cfg = json.load(open(cfg_path))
key = cfg['api_key']
url = f"http://{cfg.get('host', '192.168.1.156')}:{cfg.get('port', 8086)}"
log_path = cfg.get('log_file', os.path.expanduser('~/Strata/strata-swift-iq3_s.log'))
log_off = os.path.getsize(log_path) if os.path.exists(log_path) else 0

def chat(prompt, max_tokens=512, temperature=0.0):
    body = json.dumps({
        'model': cfg.get('model_id', 'swift-1.5-iq3_s'),
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens,
        'temperature': temperature,
        'chat_template_kwargs': {'enable_thinking': False},
    }).encode()
    req = urllib.request.Request(
        f'{url}/v1/chat/completions',
        data=body,
        headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read().decode())
    dt = time.monotonic() - t0
    tok = data.get('usage', {}).get('completion_tokens', 0)
    return tok, dt

# 1. Solo Greedy Code (temp=0.0, 512 tokens, 2 trials: Trial 1 cold, Trial 2 warm)
prompt_greedy = (
    "Write a complete, self-contained Python 3 module that implements a B+ tree "
    "supporting insert, point lookup, range scan, and delete, with docstrings and "
    "a deterministic unit test suite at the bottom. Output only the Python code."
)
for i in range(1, 3):
    tok, dt = chat(prompt_greedy, max_tokens=512, temperature=0.0)
    print(f"Greedy code trial {i}: {tok} tok in {dt:.2f}s (wall {tok/dt:.1f} tok/s)")

# 2. Solo Sampled Prose (temp=0.7, 512 tokens, 3 prompts)
prose_prompts = [
    "Write a vivid, detailed historical narrative about the construction of the transcontinental telegraph and how it transformed nineteenth-century journalism.",
    "Explain the thermodynamics and fluid mechanics of deep-sea hydrothermal vents and how chemosynthetic ecosystems thrive without sunlight.",
    "Describe the architectural evolution of Gothic cathedrals from twelfth-century France to the late English Perpendicular style, focusing on structural engineering."
]
for idx, p in enumerate(prose_prompts, 1):
    tok, dt = chat(p, max_tokens=512, temperature=0.7)
    print(f"Prose prompt {idx}: {tok} tok in {dt:.2f}s (wall {tok/dt:.1f} tok/s)")

# 3. Parse authoritative engine decode lines added during this run
if os.path.exists(log_path):
    with open(log_path, 'rb') as f:
        f.seek(log_off)
        new_log = f.read().decode('utf-8', errors='replace')
    print("\n--- Engine Log Lines ---")
    for line in new_log.splitlines():
        if 'generated in' in line:
            print(line)
PY
```

## Batch / 4-Way Concurrent Throughput Test

To test `--batch 4` (`"parallel": 4`) concurrent slots on a live or scratch server:

```bash
$HOME/Strata/.venv/bin/python - <<'PY'
import concurrent.futures, json, os, time, urllib.request

cfg = json.load(open(os.path.expanduser('~/Strata/strata-swift-iq3_s.json')))
key = cfg['api_key']
url = f"http://{cfg.get('host', '192.168.1.156')}:{cfg.get('port', 8086)}"
log_path = cfg.get('log_file', os.path.expanduser('~/Strata/strata-swift-iq3_s.log'))
log_off = os.path.getsize(log_path)

prompts = [
    "Write a Python 3 implementation of Dijkstra's algorithm using heapq with unit tests. Output only code.",
    "Write a Python 3 implementation of an LRU cache using a doubly linked list and dict with unit tests. Output only code.",
    "Write a Python 3 implementation of a trie supporting prefix search and deletion with unit tests. Output only code.",
    "Write a Python 3 implementation of Kahn's topological sort with cycle detection and unit tests. Output only code.",
]

def one_req(p):
    body = json.dumps({
        'model': cfg.get('model_id', 'swift-1.5-iq3_s'),
        'messages': [{'role': 'user', 'content': p}],
        'max_tokens': 256,
        'temperature': 0.0,
        'chat_template_kwargs': {'enable_thinking': False},
    }).encode()
    req = urllib.request.Request(f'{url}/v1/chat/completions', data=body,
                                 headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read().decode())
    dt = time.monotonic() - t0
    tok = d.get('usage', {}).get('completion_tokens', 0)
    return tok, dt

t_start = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    results = list(ex.map(one_req, prompts))
t_total = time.monotonic() - t_start
tot_tok = sum(r[0] for r in results)
print(f"4x256 concurrent: total {tot_tok} tok in {t_total:.2f}s -> aggregate {tot_tok/t_total:.1f} tok/s")
for i, (tok, dt) in enumerate(results, 1):
    print(f"  slot {i}: {tok} tok in {dt:.2f}s ({tok/dt:.1f} tok/s)")

with open(log_path, 'rb') as f:
    f.seek(log_off)
    for line in f.read().decode('utf-8', errors='replace').splitlines():
        if 'generated in' in line:
            print(line)
PY
```
For standalone CLI batch burst testing (when `strata.service` is stopped), use `$HOME/Strata-integ/tools/batch_burst_bench.py` under `systemd-run --user --scope` (add `--same-prompts` to check slot determinism).

## Standalone Kernel Microbenches & Parity Suite (Service Stopped)

Before promoting any kernel/drafter code change, stop `strata.service` and run the microbenches + CTest parity gate:

```bash
systemctl --user stop strata.service && sleep 2
cd "$HOME/Strata-integ"
PYTHONPATH="$HOME/Strata-integ/third_party/llama.cpp/gguf-py" LD_LIBRARY_PATH=/opt/rocm/lib "$HOME/Strata-integ/build-verify/dflash2_small_bench"
PYTHONPATH="$HOME/Strata-integ/third_party/llama.cpp/gguf-py" LD_LIBRARY_PATH=/opt/rocm/lib "$HOME/Strata-integ/build-verify/dflash2_gemm_bench"
PYTHONPATH="$HOME/Strata-integ/third_party/llama.cpp/gguf-py" LD_LIBRARY_PATH=/opt/rocm/lib "$HOME/Strata-integ/build-verify/dflash2_attn_bench"
PYTHONPATH="$HOME/Strata-integ/third_party/llama.cpp/gguf-py" LD_LIBRARY_PATH=/opt/rocm/lib ctest --test-dir "$HOME/Strata-integ/build-verify" --output-on-failure -R '^(dflash2_slots_test|dflash2_tap_test|arch_defaults_test|qsa_select_bench|qsa_select_bench_fp32_tiled|native_mmvq_wide_parity|mmvq_il_parity|quantize_il_parity|iq_multi_parity|quantize_act_parity|bf16_gemv_parity)$'
```
Inspect `$HOME/Strata-integ/build-verify/Testing/Temporary/LastTest.log` to confirm `0 failures` / bitwise parity on tests with `PASS_REGULAR_EXPRESSION`.

## 5K Needle Quality & Prefill Gate

```bash
umask 077 && python3 -c "import json,os; open('/tmp/.strata_key','w').write(json.load(open(os.path.expanduser('~/Strata/strata-swift-iq3_s.json')))['api_key'])" && chmod 600 /tmp/.strata_key
"$HOME/Strata/.venv/bin/python" "$HOME/Strata-integ/tools/needle_bench.py" --url http://192.168.1.156:8086 --depths 5,15,25,35,50,65,80,95 --context 5000; rc=$?; rm -f /tmp/.strata_key; exit $rc
```
Require **8/8 found** (`rc=0`) and verify warm prefill throughput (~2,420+ tok/s on 4.4K-token prompts) in the engine log.

## Reference Baselines (`swift-1.5-iq3_s`, `strata 0.1.54`, `md5 dceeee39`, Oct 30 2026)

- **Solo Greedy Code (`PROMPT_GREEDY`, 512 tok, `temp=0`):** Cold Trial 1 **117.9 tok/s**, Warm Trial 2 **133.3 tok/s** (`drafts accepted 387/521` = 74.3%).
- **Solo Sampled Prose (3 prompts, 512 tok, `temp=0.7`):** **66.7, 84.7, 78.9 tok/s** (mean **76.8 tok/s**, `0.1.52` run mean **79.9 tok/s**).
- **`dflash2_small_bench`:** `topk (7 x 106299)` **49.5–50.4 µs**, `select (7 rows)` **13.56 µs**, `conv_rms_quant` **6.3–6.5 µs**.
- **5K Needle Prefill (depths 5..95):** **2,422–2,430 tok/s** warm prefill, **8/8 PASS**.
