# Elastic host K/V copy for streamed (mode-1) layers — design

Status: designed, not implemented. Measured on the R9700 production box, 2026-11-11, engine fork-a223b54e5.

## The RAM the box pays today

KV streaming (`--kv-resident`) keeps the authoritative K/V of every cell in a **pinned,
device-mapped host copy** allocated at boot, sized by `--max-context`, one pool array set
per QSA layer per session. Production (262144 cells, int8 KV, 12 QSA layers, `--batch 2`):

    276,825,088 B (264 MiB) x 12 layers x 3 sessions (main + 2 slots) = 9.28 GiB

pinned at startup and never sized by what the conversations actually use. Verified by an
LD_PRELOAD shim on `hipHostMalloc` (the census method is in the ops skill): 36 identical
allocations, `qsa_state_init` frames — 24 from the slot sessions, 12 from main.

## Why `--kv-grow` does not help here today

`--kv-grow`'s elastic machinery (layer.cpp `ElasticPool`, `g_kv_elastic`) runs only for
`p.mode == 0` — the whole-K/V-in-VRAM layout. A streamed session (`mode == 1`) allocates the
host copy through `kv_host_alloc()` unconditionally and elastically-sized VRAM slots are
unrelated. The startup guard additionally requires `!o.resident_cpu_experts`,
`o.kv_resident <= 0` and `o.batch == 0`. Removing the batch clause alone changes nothing on
any streaming config: `--kv-grow stays off` either way. Mode 0 is not an option on a 34 GiB
card either: the full 262k K/V in VRAM is ~3.1 GiB and the expert cache leaves <1 GiB.

## The fix: chunked elastic host copies

Make the mode-1 host copy grow per state like `ElasticPool` grows mode-0 pools: reserve the
full range in address space, commit physical pages per chunk as the conversation grows.

Layout. Chunk = a whole number of KV blocks, a power of two in blocks, so
`cell -> (chunk, offset)` is shift+mask. Chunks per state per array:
`pages / blocks_per_chunk`. Choose blocks_per_chunk so a chunk is 8-32 MiB; page_size is
already block-aligned and no block may span chunks.

Allocation per chunk on demand:
1. `mmap` the **full** host range once with `MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE` so
   every array keeps one stable host VA range (the existing `Cursor hc{d}` carving of
   `st.host.*` stays valid — array bases are unchanged addresses).
2. Per chunk, on grow: touch the pages (or `MAP_FIXED` `MAP_POPULATE` remap), then
   `cudaHostRegister(Portable|Mapped)` **that chunk's range** and
   `cudaHostGetDevicePointer` for it. Note: one device VA is NOT assumed to cover the whole
   range — device pointers come per chunk.

Two access paths to retrofit (the census of both, current code):
- **DMA builders** (`kv_stream.cu` `r.src[i] = host.X + off`, and `kv_stage_from_host`,
  `kv_unstage_to_host`, `kv_ring_restore` `[b0,b1)` ranges): pure host-side code — replace
  the single base with `chunk_base[b >> shift] + (b & mask) * block_bytes`, splitting the
  [b0,b1) loop at chunk boundaries. No ABI change.
- **Zero-copy kernel reads** (the prompt-path attention pools built from
  `h ? st.host.X : st.X` at the generate.cpp read sites): kernels index one base pointer.
  Give these kernels a fixed-size device-resident pointer table (one `void*` per chunk,
  written at commit time) and one level of indirection at the array base — offsets within a
  chunk are identical to today, so outputs are bit-identical by construction. Count the read
  sites before starting: they are all the same shape (`cond ? host : vram` pool structs).

Grow hook: where the conversation extends past the committed cells — prompt admission
(per chunk of the prompt) and the per-token step (a relaxed compare against the committed
target, like `qsa_kv_elastic_need`/`grow` at generate.cpp:10218/10304, which today grow ALL
mode-0 pools to the max target; mirror that conservatism, or key pools per state for
tighter accounting). Shrink at conversation end can reuse the same tables (unregister +
`MADV_DONTNEED`).

Guard: relax `o.batch == 0` for the new path behind its own flag (`--kv-host-grow`, or
`STRATA_KV_HOST_GROW=1`), keep mode-0 `--kv-grow` guard as-is. Slot states each get their
own chunk tables — nothing about the slots' session_init layout changes.

## Expected effect

Measured active conversations on this box park at 3k-132k cells; the 262144-cell bodies are
mostly unbacked pages today and would be ~committed-on-use. Boot-time pinned drops from
9.28 GiB to `3 x 12 x init_chunk` (16384-cell init: ~1.1 GiB), growing to the working set.

## Test plan

- Bit parity: full prompt+decode replay vs the current engine at `--kv-host-grow` off (flag
  off must be byte-identical argv-for-argv) and on (chunking must not reorder any byte).
- Candidate window (`candidate-test-window.md`): 262k single request, two concurrent 60k
  requests (batch-2), a parked-conversation restore, decode tok/s A/B (the indirection is
  one load per array base per kernel, expected noise), boot RAM (`pmap` census) before/after.
- Failure modes to prove safe: kernel touching an uncommitted chunk must be impossible by
  table lookup (null chunk entry -> the existing illegal-access path, prefer an explicit
  abort with a clear message), and RAM admission must respect `conversation_memory_admit`
  when chunks grow at runtime (grow refused -> the conversation must stop cleanly, not die).

## Effort

~1-2 dev-days for the implementation + A/B, gated behind its own flag. Nothing here should
ride into production unmeasured.
