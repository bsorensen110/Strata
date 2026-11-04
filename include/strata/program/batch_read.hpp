// include/strata/program/batch_read.hpp - how --batch reads a prompt in pieces (header-only: it is arithmetic).
//
// A prompt read with --batch stops between two pieces to (a) let the slots decode and (b) look for a BYIELD (a
// shorter request is waiting).  The pieces always end on a multiple of the prompt chunk, so they read exactly the
// chunks a single run would.
#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::program {

/// Tokens of the next piece of a --batch prompt read, or 0 for "the whole rest in one run".
///   beside_slots: some slot is decoding (its windows run between the pieces)
///   layer_split / piped: the read goes through a layer split's stages / the pipelined groups
///   chunks_per_piece: for a layer split with no slot decoding, the pieces' length in chunks (0: one run)
/// A layer split reads its stages as a pipeline over one run's chunks, so a piece costs a pipeline fill; with slots
/// decoding the pieces are single chunks anyway.  With no slot decoding a piece is `chunks_per_piece` chunks: long
/// enough to keep the pipeline full, short enough that a BYIELD is taken within a few chunks (it used to wait for the
/// whole prompt, and the long prompt then held its slot: "ERR BGEN: no such free slot" for the request that should
/// have taken over).
inline int64_t batch_read_piece(bool batch, bool piped, bool layer_split, bool beside_slots, int64_t chunk,
                                int64_t chunks_per_piece) {
    if (!batch || piped) return 0;
    chunk = std::max<int64_t>(chunk, 1);
    if (!layer_split || beside_slots) return chunk;
    return chunks_per_piece > 0 ? chunk * chunks_per_piece : 0;
}

/// Default prefill chunk cap on a single GPU when active batch decode slots are running alongside a prompt read.
/// An 8192-token chunk blocks active decode slots for ~2.5-3.5 s on a single card; capping the chunk to 2048 tokens
/// (aligned to the 2048-token Q2/Q4 KV group and checkpoint grid) yields to active decode slots 4x more often.
///   full_chunk: the prefill engine's configured max chunk (sp.chunk())
///   active_slots: number of currently decoding batch slots
///   env_chunk: STRATA_BATCH_PREFILL_CHUNK override (> 0: explicit cap; 0: disable cap; < 0: use default 2048)
inline constexpr int64_t kDefaultBatchPrefillChunk = 2048;
inline int64_t batch_prefill_chunk(int64_t full_chunk, int active_slots, int64_t env_chunk = -1) {
    full_chunk = std::max<int64_t>(full_chunk, 1);
    if (active_slots <= 0 || env_chunk == 0) return full_chunk;
    const int64_t cap = env_chunk > 0 ? env_chunk : kDefaultBatchPrefillChunk;
    return std::min(full_chunk, std::max<int64_t>(cap, 256));
}

/// Scales the decode budget share between prompt chunks by the number of active decoding slots so 2-3 concurrent
/// decoding slots get proportionally more GPU time between prefill chunks without starving incoming prompt reads:
///   1 slot  -> base_share (default 0.50)
///   2 slots -> min(1.0, base_share * 1.5) (0.75 at default 0.50)
///   3+ slots -> min(1.5, base_share * 2.0) (1.00 at default 0.50)
inline double batch_decode_share(double base_share, int active_slots) {
    if (base_share <= 0.0 || active_slots <= 0) return 0.0;
    if (active_slots == 1) return base_share;
    if (active_slots == 2) return std::min(std::max(base_share, 1.0), base_share * 1.5);
    return std::min(std::max(base_share, 1.5), base_share * 2.0);
}

/// What a configuration has that keeps the concurrent read (STRATA_PIPED_CONCURRENT=1) off.
struct ConcurrentReadSetup {
    bool prefill_help = false;      ///< STRATA_PREFILL_HELP: the idle stage's help allocates on the read's thread
    bool split_mtp_batch = false;   ///< STRATA_SPLIT_MTP_BATCH: the batched draft K/V frees there (a free waits)
    bool fs_slots = false;          ///< STRATA_FS_SLOTS (Foresight swap space): serving a doorbell makes a CUDA call
    bool pcie_dma = false;          ///< --pcie-mode dma: the same (its copies)
    bool helper_caches = false;     ///< --expert-cache-device1..3 (and --remote-expert-opt on them), a prompt source of
                                    ///< its own; --remote-expert-opt alone is inert on a plain layer split
    bool staged_source = false;     ///< the GGUF source stages its blobs (in place, unbuffered, STRATA_IO_PREFETCH):
                                    ///< it recycles the buffers it hands out, one thread at a time
    bool other_source = false;      ///< an expert source that is neither the GGUF source nor the arena
};

/// Why the concurrent read stays off with `s` (the first reason, for the log line), or nullptr when it can run.
inline const char* concurrent_read_off(const ConcurrentReadSetup& s) {
    if (s.prefill_help) return "STRATA_PREFILL_HELP";
    if (s.split_mtp_batch) return "STRATA_SPLIT_MTP_BATCH";
    if (s.fs_slots) return "STRATA_FS_SLOTS";
    if (s.pcie_dma) return "--pcie-mode dma";
    if (s.helper_caches) return "the helper caches (--expert-cache-device1..3)";
    if (s.staged_source)
        return "an expert source that stages its blobs (the GGUF in place, unbuffered, STRATA_IO_PREFETCH)";
    if (s.other_source) return "this expert source";
    return nullptr;
}

/// Tokens of the next piece of a prompt read on the pipelined slot groups (--batch-groups > 1 on a layer split, #1906),
/// or 0 for "the whole rest in one run" (the opt-in is off).
///   piece_tokens: STRATA_PIPED_READ_PIECE, rounded down to whole chunks (at least one, at most kPipedMaxChunks)
/// A piece is one pipelined run over its chunks, so the stages stay overlapped inside it and only its end drains the
/// pipeline.  At a piece's end the decoding slots decode, and a waiting shorter request can take over (BYIELD), as
/// between the pieces of a read off the pipeline.
inline constexpr int64_t kPipedMaxChunks = 32;
inline int64_t piped_read_piece(int64_t piece_tokens, int64_t chunk) {
    if (piece_tokens <= 0) return 0;
    chunk = std::max<int64_t>(chunk, 1);
    return std::clamp<int64_t>(piece_tokens / chunk, 1, kPipedMaxChunks) * chunk;
}

/// The slots' decode time between two pieces: share x the piece's time (STRATA_BATCH_DECODE_SHARE, the rule the
/// one-GPU path uses between chunks), at most cap_ms (STRATA_PIPED_DECODE_MAX_MS; <= 0: no cap), at least floor_ms
/// (STRATA_PIPED_DECODE_MIN_MS).
inline double piped_decode_budget_ms(double piece_ms, double share, double cap_ms, double floor_ms) {
    double b = std::max(0.0, share) * std::max(0.0, piece_ms);
    if (cap_ms > 0.0) b = std::min(b, cap_ms);
    return std::max(b, std::max(0.0, floor_ms));
}

/// Prompts arriving back to back: a piecewise read that starts within kPipedLeadInWindowMs + budget_ms of the
/// previous piecewise read's end first gives the slots one decode budget (otherwise they stop for both reads).
///   ms_since_prev_end: < 0 when there was no piecewise read yet
inline constexpr double kPipedLeadInWindowMs = 1000.0;
inline bool piped_read_lead_in(bool enabled, bool beside_slots, double ms_since_prev_end, double budget_ms) {
    return enabled && beside_slots && ms_since_prev_end >= 0.0 &&
           ms_since_prev_end <= kPipedLeadInWindowMs + std::max(0.0, budget_ms);
}

}  // namespace strata::program
