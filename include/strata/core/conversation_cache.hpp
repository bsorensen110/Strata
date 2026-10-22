// CPU-only ownership and matching policy for --serve's parked conversations.
// Token equality, image identity, and steering mode are all required for reuse.
#pragma once

#include "strata/core/conversation_buffer.hpp"
#include "strata/core/conversation_memory.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>   // [SFX-DBG] probe: fprintf in best_with_partial's candidate scan
#include <deque>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace strata::core {

struct ConversationImageKey {
    int64_t start = 0;
    uint64_t hash = 0;
    bool operator==(const ConversationImageKey&) const = default;
};

struct ConversationCheckpoint {
    std::vector<int32_t> ids;
    std::vector<ConversationImageKey> imgs;
    std::vector<uint8_t> gdn, ple, tails, dead, block_pos;
    uint64_t used = 0; // upstream root-pinned/LRU checkpoint retention
    // A shared-prefix pin (the request key pin=N): this checkpoint is the read-only prefix many suffix queries branch
    // from, so retention never evicts it (conv_cache.hpp) and the parked cache keeps at least one holder.  A run-time
    // mark only: it is not in the session file, a request that pins the same prefix again sets it.
    bool pinned = false;
    // STRATA_BRANCH_CHECKPOINT: taken where the prompt left a conversation the engine held - the point a client that
    // rewrites the end of its history continues from next, so a batch slot keeps its deepest one.  Run-time only.
    bool branch = false;
    // --prefill-pipe on a layer split: the plan a prompt segment [from, to) is read in - its chunk (0: none), the loan
    // its buffers were laid out for (which experts the prompt reads in place and which it streams: that sets bits
    // too), the pipe value it was planned with, whether it is one chunk and whether the plan changed the buffers'
    // chunks.  A periodic checkpoint taken inside such a segment keeps it, so a read that goes on from there reads the
    // rest in the same chunks with the same loan.  Run-time only, like `pinned` (a layer split saves no session files).
    struct PromptPlan {
        int64_t chunk = 0, from = -1, to = -1, lend = 0;
        double pipe = 0.0;
        bool single = false, chunked = false;
    } plan;
    // Ordinary layer-split checkpoints retain each device's running state.
    // Whole-session parking is currently single-GPU and rejects these parts.
    std::vector<ConversationCheckpoint> stage_parts;

    size_t bytes() const {
        size_t n = ids.capacity() * sizeof(int32_t) + imgs.capacity() * sizeof(ConversationImageKey) +
               gdn.capacity() + ple.capacity() + tails.capacity() + dead.capacity() + block_pos.capacity() +
               stage_parts.capacity() * sizeof(ConversationCheckpoint);
        for (const auto& part : stage_parts) n += part.bytes();
        return n;
    }
};

// Identity-layout K/V pages and completed indexer rows. For streamed layers the
// source is the authoritative host pool, NOT the replaceable VRAM slots.
struct ConversationKv {
    int format = 0;
    int64_t cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0;
    ConversationBuffer k, v, k_scale, v_scale, pooled;
    size_t bytes() const {
        return k.bytes() + v.bytes() + k_scale.bytes() + v_scale.bytes() + pooled.bytes();
    }
};

struct ConversationKvReuse {
    std::vector<ConversationKv> kv;
    // Original image extent for validation, and the earliest subsequent rewrite.
    int64_t captured_tokens = 0, unchanged_tokens = 0;
    // The conversation the image was captured from: a park reuses it only for the prefix it shares with the
    // conversation then in the main session (a slot may have replaced that session's K/V since).
    std::vector<int32_t> ids;
    // With a layer split: the later stages' own retained K/V, one per stage, same extents (empty: none)
    std::vector<ConversationKvReuse> stages;
    size_t bytes() const {
        size_t n = kv.capacity() * sizeof(ConversationKv) + stages.capacity() * sizeof(ConversationKvReuse) +
                   ids.capacity() * sizeof(int32_t);
        for (const auto& layer : kv) n += layer.bytes();
        for (const auto& s : stages) n += s.bytes();
        return n;
    }
};

/// Layer-split parking: each running checkpoint holds the first stage's state and, in `stage_parts`, one part per
/// later stage; a parked conversation keeps them in one image per stage.  `split` MOVES the checkpoints apart (no
/// running state is copied): the complete ones (a part for each of `stages` stages) into `stage0` and `parts[k]`,
/// every part carrying the checkpoint's ids / images / LRU stamp, the others aside; `merge` puts them back in their
/// original order.  merge(split(x)) == x.
struct ConversationCheckpointSplit {
    std::vector<ConversationCheckpoint> stage0;
    std::vector<std::vector<ConversationCheckpoint>> parts;   ///< [stage][checkpoint]
    std::vector<ConversationCheckpoint> rest;                 ///< not split (no complete set of stage parts)
    std::vector<bool> complete;                               ///< per original position: in stage0/parts or in rest
};

inline ConversationCheckpointSplit conversation_checkpoints_split(std::vector<ConversationCheckpoint>&& checks,
                                                                  size_t stages) {
    ConversationCheckpointSplit out;
    out.parts.resize(stages);
    out.complete.reserve(checks.size());
    size_t whole = 0;
    for (const auto& c : checks) {
        const bool complete = c.stage_parts.size() == stages;
        out.complete.push_back(complete);
        whole += complete ? 1 : 0;
    }
    out.stage0.reserve(whole);
    out.rest.reserve(checks.size() - whole);
    for (auto& parts : out.parts) parts.reserve(whole);
    // Allocate and copy identity metadata before moving any running state.
    // If allocation fails, every input checkpoint remains usable by the caller.
    for (const auto& c : checks) {
        if (c.stage_parts.size() != stages) continue;
        for (size_t k = 0; k < stages; ++k) {
            ConversationCheckpoint part;
            part.ids = c.ids; part.imgs = c.imgs; part.used = c.used; part.pinned = c.pinned;
            out.parts[k].push_back(std::move(part));
        }
    }
    size_t w = 0;
    for (auto& c : checks) {
        if (c.stage_parts.size() != stages) { out.rest.push_back(std::move(c)); continue; }
        for (size_t k = 0; k < stages; ++k) {
            auto& dst = out.parts[k][w];
            auto& src = c.stage_parts[k];
            dst.gdn = std::move(src.gdn); dst.ple = std::move(src.ple);
            dst.tails = std::move(src.tails); dst.dead = std::move(src.dead);
            dst.block_pos = std::move(src.block_pos);
        }
        c.stage_parts.clear();
        out.stage0.push_back(std::move(c));
        ++w;
    }
    checks.clear();
    return out;
}

/// false (and `out` empty) when the parts do not line up: a stage with another number of checkpoints, or a part
/// whose ids differ from its checkpoint's.
inline bool conversation_checkpoints_merge(ConversationCheckpointSplit&& split, std::vector<ConversationCheckpoint>& out) {
    out.clear();
    for (const auto& p : split.parts)
        if (p.size() != split.stage0.size()) return false;
    size_t whole = 0, other = 0;
    for (bool c : split.complete) (c ? whole : other) += 1;
    if (!split.complete.empty() && (whole != split.stage0.size() || other != split.rest.size())) return false;
    if (split.complete.empty()) split.complete.assign(split.stage0.size(), true);   // a restored image: all whole
    for (size_t j = 0; j < split.stage0.size(); ++j)
        for (const auto& p : split.parts)
            if (p[j].ids != split.stage0[j].ids) return false;
    size_t w = 0, r = 0;
    for (bool c : split.complete) {
        if (!c) { out.push_back(std::move(split.rest[r++])); continue; }
        ConversationCheckpoint cp = std::move(split.stage0[w]);
        for (auto& p : split.parts) cp.stage_parts.push_back(std::move(p[w]));
        ++w;
        out.push_back(std::move(cp));
    }
    return true;
}

/// --kv-pool-tokens: a conversation changes lane (the main session <-> a batch slot) by moving its K/V chunks, so its
/// running-state checkpoints must change lane with them - a checkpoint is only usable beside the K/V cells below it.
/// Moves the checkpoints of `from` whose tokens are a prefix of `ids` onto the back of `to`, in their order; the others
/// stay in `from` (the caller drops them).  Text only: a checkpoint with a picture is not carried (a slot never caches
/// a conversation with pictures).  Returns how many moved.
inline size_t conversation_checkpoints_carry(std::vector<ConversationCheckpoint>& from, const std::vector<int32_t>& ids,
                                             std::vector<ConversationCheckpoint>& to) {
    size_t moved = 0;
    std::vector<ConversationCheckpoint> rest;
    for (auto& c : from) {
        const bool prefix = !c.ids.empty() && c.ids.size() <= ids.size() && c.imgs.empty() &&
                            std::equal(c.ids.begin(), c.ids.end(), ids.begin());
        if (prefix) { to.push_back(std::move(c)); ++moved; }
        else rest.push_back(std::move(c));
    }
    from = std::move(rest);
    return moved;
}

struct SavedConversation {
    // Experimental approximate ordinary->YaRN history: version, source/target
    // execution fingerprints, original prefix length. Zero means no migration.
    std::array<uint64_t, 4> rope_migration{};
    // Runtime compatibility only; NOT a model/weights identity or disk schema.
    std::array<int64_t, 18> geometry{};
    // The session's layer carve the image was captured from ([0, n_layers) on one GPU); restore requires the same.
    int64_t layer_lo = 0, layer_hi = 0;
    ConversationCheckpoint live;
    std::vector<ConversationCheckpoint> checkpoints;
    std::vector<ConversationKv> kv; // main layers followed by the draft layer
    // In-memory DFlash context, separate from the target/MTP disk-session schema.
    std::vector<ConversationKv> dflash_kv;
    bool cvec = true;
    bool lora = true;   // the LoRA adapter's setting (--lora; true when none is loaded)
    // with a layer split, the later stages' own images, one per stage, in stage order
    std::vector<SavedConversation> stage_images;
    // Disk-mode only, runtime only: NOT serialized (conversation_file's put_payload/get_payload never see these).
    // Non-empty disk_path: the payload lives in this session file and RAM holds the index entry only; the budget
    // accounts disk_bytes instead of bytes().
    std::string disk_path;
    size_t disk_bytes = 0;

    /// Holds a pinned shared prefix (see ConversationCheckpoint::pinned): the parked-conversation budget keeps it.
    bool pinned() const {
        for (const auto& c : checkpoints) if (c.pinned) return true;
        return false;
    }

    size_t bytes() const {
        size_t n = live.bytes() + checkpoints.capacity() * sizeof(ConversationCheckpoint) +
                   kv.capacity() * sizeof(ConversationKv) + dflash_kv.capacity() * sizeof(ConversationKv);
        for (const auto& k : dflash_kv) n += k.bytes();
        for (const auto& s : stage_images) n += s.bytes();
        for (const auto& c : checkpoints) n += c.bytes();
        for (const auto& k : kv) n += k.bytes();
        return n;
    }
};

/// `c` holds the state `ids` (with `images`) describes, or a continuation of it: its tokens start with `ids` and
/// its pictures below ids.size() are exactly `images`.  What an idle batch slot holds is then parked already (or
/// a longer continuation is), so the slot need not park it again before an admission overwrites it.
template<class Token>
bool conversation_covers(const ConversationCheckpoint& c, const std::vector<Token>& ids,
                         const std::vector<ConversationImageKey>& images) {
    const size_t n = ids.size();
    if (n == 0 || c.ids.size() < n) return false;
    for (size_t i = 0; i < n; ++i)
        if ((int64_t) c.ids[i] != (int64_t) ids[i]) return false;
    size_t j = 0;
    for (const auto& image : c.imgs) {
        if (image.start >= (int64_t) n) continue;
        if (j == images.size() || !(images[j++] == image)) return false;
    }
    return j == images.size();
}

/// The matching index of a conversation without its state: ids/imgs/used/pinned of live and of each checkpoint,
/// cvec and the geometry / layer range; empty state vectors, empty K/V, no stage parts.
inline SavedConversation conversation_meta_only(const SavedConversation& s) {
    SavedConversation m;
    m.geometry = s.geometry;
    m.layer_lo = s.layer_lo; m.layer_hi = s.layer_hi;
    m.cvec = s.cvec;
    m.live.ids = s.live.ids; m.live.imgs = s.live.imgs; m.live.used = s.live.used; m.live.pinned = s.live.pinned;
    for (const auto& c : s.checkpoints) {
        ConversationCheckpoint mc;
        mc.ids = c.ids; mc.imgs = c.imgs; mc.used = c.used; mc.pinned = c.pinned;
        m.checkpoints.push_back(std::move(mc));
    }
    return m;
}

template<class Token>
int64_t conversation_prefix(const ConversationCheckpoint& c, const std::vector<Token>& prompt,
                            const std::vector<ConversationImageKey>& images) {
    const size_t n = c.ids.size();
    // The last prompt token always starts the next verify window.
    if (n == 0 || n >= prompt.size() || !std::equal(c.ids.begin(), c.ids.end(), prompt.begin())) return 0;
    size_t j = 0;
    for (const auto& image : images) {
        if (image.start >= (int64_t) n) continue;
        if (j == c.imgs.size() || !(c.imgs[j++] == image)) return 0;
    }
    if (j != c.imgs.size()) return 0;
    return (int64_t) n;
}

/// The leading cells two conversations provably share: equal token ids, and equal image keys wherever either
/// carries one below that length (an image changes the K/V of every cell from its start on). The K/V bytes of
/// the cells below the result are the same in both conversations of one model - what neither rewrote stays
/// interchangeable storage. Both image lists are in `start` order, as the engine keeps them.
inline int64_t conversation_shared_prefix(const std::vector<int32_t>& a, const std::vector<ConversationImageKey>& a_imgs,
                                          const std::vector<int32_t>& b, const std::vector<ConversationImageKey>& b_imgs) {
    int64_t n = (int64_t) std::min(a.size(), b.size());
    int64_t i = 0;
    while (i < n && a[(size_t) i] == b[(size_t) i]) ++i;
    n = i;
    // the images below the token bound must agree; the first disagreement cuts the prefix at its start
    size_t ja = 0, jb = 0;
    while (ja < a_imgs.size() || jb < b_imgs.size()) {
        const bool below_a = ja < a_imgs.size() && a_imgs[ja].start < n;
        const bool below_b = jb < b_imgs.size() && b_imgs[jb].start < n;
        if (!below_a && !below_b) break;
        if (below_a && below_b && a_imgs[ja] == b_imgs[jb]) { ++ja; ++jb; continue; }
        if (below_a) n = std::min(n, a_imgs[ja].start);
        if (below_b) n = std::min(n, b_imgs[jb].start);
    }
    return n;
}

/// Phase 3b: longest common PREFIX of a checkpoint and a prompt, up to the checkpoint's length.
/// Returns:
///   0:  no LCP (token 0 mismatches, or image key mismatch below the LCP, or empty checkpoint)
///   n:  the checkpoint is a strict prefix of the prompt (== conversation_prefix, the "full match")
///   p:  partial match - 0 < p < n (the LCP length; the checkpoint's tail [p, n) diverges)
/// Image check mirrors conversation_prefix but bounded by the LCP (not the full checkpoint):
/// request images at positions < p must match the checkpoint's images at the same positions; the
/// checkpoint's images at positions >= p are irrelevant (the partial mount discards [p, n) of the
/// checkpoint).  This is the helper the flag-gated partial-prefix mount in
/// `best_with_partial` walks to find the longest LCP when no strict match exists.
template<class Token>
int64_t conversation_lcp(const ConversationCheckpoint& c, const std::vector<Token>& prompt,
                         const std::vector<ConversationImageKey>& images) {
    const size_t n = c.ids.size();
    if (n < 1) return 0;                       // empty: no LCP
    // T17 (2026-10-13): a candidate LONGER than the prompt is a valid partial
    // candidate - the parked LIVE (prompt+response) is almost always slightly
    // longer than the next turn's prompt, and the old `n > prompt.size()` early
    // out here (plus the two guards beside it) refused the single most valuable
    // anchor in every edit chain (probe: e0.live:L=5786:skip, prompt 5736).
    // The walk is bounded by min(n, prompt.size()): tokens past the prompt are
    // irrelevant (the partial mount discards them), and the p == n < prompt.size
    // case reports the full length (the strict-prefix shape, unchanged).
    int64_t p = 0;
    while (p < (int64_t) n && p < (int64_t) prompt.size() &&
           c.ids[(size_t) p] == (Token) prompt[(size_t) p]) ++p;
    if (p == 0) return 0;                                // token 0 mismatches
    // Image check: request images at positions < p must match the checkpoint's images.  The
    // checkpoint's images at positions >= p are irrelevant (we re-prefill [p, n) on the new
    // prompt).  The strict `conversation_prefix` check is the same logic with n -> p.
    size_t j = 0;
    for (const auto& image : images) {
        if (image.start >= p) continue;
        if (j == c.imgs.size() || !(c.imgs[j++] == image)) return 0;
    }
    for (size_t k = j; k < c.imgs.size(); ++k)
        if (c.imgs[k].start < p) return 0;               // checkpoint has an image below p not in the request
    return p;
}

class ConversationCache {
public:
    struct Match {
        size_t index = 0;
        int64_t tokens = 0;
        bool live = false;
        uint64_t serial = 0;   // the entry's identity; its index shifts when entries before it go
        // Phase 3b: flag-on partial-prefix match.  `partial == true` means the longest LCP of any
        // checkpoint with the request is strictly less than the checkpoint's length (a mid-prompt
        // edit past the LCP).  In that case `tokens` is the LCP (the resume length the splice hook
        // narrows to) and `checkpoint_length` is the full checkpoint's length (the restore anchor).
        // Strict match (== the existing behaviour): `partial == false`, `tokens == checkpoint_length`.
        // No match: `partial == false`, `tokens == 0`.  Off-flag (`flag <= 0`) and on-flag-with-
        // strict-match returns are byte-identical to `best()`.
        bool partial = false;
        int64_t checkpoint_length = 0;   ///< the full checkpoint's length (== tokens when !partial)
    };

    ConversationCache(size_t budget, size_t slots, int64_t min_tokens = 0)
        : budget_(budget), slots_(slots), min_tokens_(min_tokens) {}
    bool enabled() const { return budget_ != 0 && slots_ != 0; }
    // the whole budget in bytes (disk mode: the budget across the parked files)
    size_t budget() const { return budget_; }
    // Disk mode: entries hold a session-file path and the budget counts its bytes, not the RAM index's.
    // Unlink rule: evict_oldest, drop_superseded and clear DELETE the entry's disk_path (they created it, on the
    // park path); take() and remove_at() do NOT - the caller owns the file until its restore succeeds.
    void enable_disk() { disk_ = true; }
    bool disk() const { return disk_; }
    static size_t entry_bytes(const SavedConversation& e) { return e.disk_path.empty() ? e.bytes() : e.disk_bytes; }
    /// `--conversation-cache-min-tokens`: whether a conversation of this length is worth one of `slots`.  Every
    /// parked conversation costs a whole slot whatever its length, so a client that interleaves short side requests
    /// with one long conversation spends the cache on the side requests and evicts the long one (oldest first) -
    /// exactly the reuse it wanted.  0 parks every conversation, whatever its size.
    bool wants(int64_t tokens) const { return min_tokens_ <= 0 || tokens >= min_tokens_; }
    size_t bytes() const { return bytes_ + reuse_.bytes() + reuse_identity_bytes(); }
    size_t size() const { return entries_.size(); }
    size_t evictions() const { return evictions_; }
    // the longest parked conversation, in tokens (--kv-grow keeps the K/V that long while it could be restored)
    int64_t longest_tokens() const {
        int64_t n = 0;
        for (const auto& e : entries_) n = std::max<int64_t>(n, (int64_t) e.live.ids.size());
        return n;
    }

    // Retain only the restored K/V buffers, not duplicate running checkpoints.
    // This optimization never evicts a parked conversation to make itself fit.
    // `stage_kv`: with a layer split, the later stages' restored K/V (one per stage), retained with the first's.
    // `ids`/`imgs`: the conversation these bytes belong to (the session's live tokens and image keys at the
    // restore).  take_reuse() then only ever hands the buffers back for a conversation sharing their prefix, and
    // take_reuse_for() refuses them for any other (none: never reused), so the bytes of a DIFFERENT conversation
    // can never be parked as this one's, whatever the caller's own bookkeeping did to the session in between.
    // Retained without an identity, the caller owns that instead.
    void retain(std::vector<ConversationKv>&& kv, int64_t tokens,
                std::vector<std::vector<ConversationKv>>&& stage_kv = {},
                std::vector<int32_t> ids = {}, std::vector<ConversationImageKey> imgs = {}) {
        drop_reuse();
        ConversationKvReuse candidate{std::move(kv), tokens, tokens, ids, {}};
        for (auto& k : stage_kv) candidate.stages.push_back(ConversationKvReuse{std::move(k), tokens, tokens, {}, {}});
        if (enabled() && candidate.bytes() <= budget_ - bytes_) {
            reuse_ = std::move(candidate);
            reuse_ids_ = std::move(ids);
            reuse_imgs_ = std::move(imgs);
        }
    }
    void limit_reuse(int64_t first_dirty) {
        reuse_.unchanged_tokens = std::min(reuse_.unchanged_tokens, first_dirty);
        for (auto& s : reuse_.stages) s.unchanged_tokens = std::min(s.unchanged_tokens, first_dirty);
        if (reuse_.unchanged_tokens <= 0) drop_reuse();
    }
    // The retained K/V, bounded by the first rewrite (limit_reuse) AND - when it was retained with the identity
    // of the conversation it was restored from - by the prefix that conversation and `ids`/`imgs` actually
    // share: equal ids and equal image keys below it are what make those cells' bytes interchangeable.  Called
    // with no conversation, it only drops the reuse (an image of one conversation is no image of another).
    ConversationKvReuse take_reuse(const std::vector<int32_t>& ids = {},
                                   const std::vector<ConversationImageKey>& imgs = {}) {
        ConversationKvReuse out = std::exchange(reuse_, {});
        const std::vector<int32_t> were = std::exchange(reuse_ids_, {});
        const std::vector<ConversationImageKey> were_imgs = std::exchange(reuse_imgs_, {});
        if (!were.empty()) {
            const int64_t shared = conversation_shared_prefix(were, were_imgs, ids, imgs);
            out.unchanged_tokens = std::min(out.unchanged_tokens, shared);
            for (auto& s : out.stages) s.unchanged_tokens = std::min(s.unchanged_tokens, shared);
            if (out.unchanged_tokens <= 0) out = {};
        }
        return out;
    }
    // The retained K/V for parking `live`: limited to the prefix `live` shares with the conversation it was captured
    // from, and nothing when they share none.  Without this a park after a slot gave its conversation back to the main
    // session wrote the RESTORED conversation's K/V into the new one's image (2026-10-08 cross-conversation leak).
    template<class Token>
    ConversationKvReuse take_reuse_for(const std::vector<Token>& live) {
        ConversationKvReuse r = std::exchange(reuse_, {});
        const int64_t n = std::min<int64_t>((int64_t) r.ids.size(), (int64_t) live.size());
        int64_t same = 0;
        while (same < n && (int64_t) r.ids[(size_t) same] == (int64_t) live[(size_t) same]) ++same;
        r.unchanged_tokens = std::min(r.unchanged_tokens, same);
        for (auto& st : r.stages) st.unchanged_tokens = std::min(st.unchanged_tokens, same);
        if (r.unchanged_tokens <= 0) return {};
        return r;
    }
    size_t retained_bytes() const { return reuse_.bytes(); }
    int64_t retained_tokens() const { return reuse_.kv.empty() ? 0 : reuse_.unchanged_tokens; }
    bool can_fit(size_t incoming, size_t held = 0) const {
        return enabled() && held <= budget_ && incoming <= budget_ - held &&
               entries_.size() < slots_ && bytes() <= budget_ - held - incoming;
    }

    template<class Token>
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec,
               bool lora = true) const {
        Match best;
        // Ties prefer the most recently parked branch. The caller prefers its
        // already-active state when that offers the same prefix length.
        for (size_t i = entries_.size(); i-- > 0;) {
            const auto& e = entries_[i];
            if (e.cvec != cvec || e.lora != lora) continue;
            auto consider = [&](const ConversationCheckpoint& c, bool live) {
                const int64_t n = conversation_prefix(c, prompt, images);
                if (n > best.tokens) best = {i, n, live, serials_[i], false, n};
            };
            consider(e.live, true);
            for (const auto& c : e.checkpoints) consider(c, false);
        }
        return best;
    }

    /// A parked conversation holds this state already (its live tokens or one of its checkpoints start with `ids`,
    /// same pictures, same steering): see conversation_covers.  Read-only; no LRU effect.
    template<class Token>
    bool holds(const std::vector<Token>& ids, const std::vector<ConversationImageKey>& images, bool cvec) const {
        for (const auto& e : entries_) {
            if (e.cvec != cvec) continue;
            if (conversation_covers(e.live, ids, images)) return true;
            for (const auto& c : e.checkpoints)
                if (conversation_covers(c, ids, images)) return true;
        }
        return false;
}

    /// Phase 3b: flag-gated partial-prefix match.  At `flag <= 0` (and on-flag when a strict
    /// match exists) this is byte-identical to `best()`: same loop, same `conversation_prefix`,
    /// same tie-break by most recently parked.  At `flag > 0` with NO strict match, falls back
    /// to the longest LCP `p` such that `0 < p < c.ids.size()` and the image check at positions
    /// < p matches; the returned Match has `partial = true, tokens = p, checkpoint_length = L`.
    /// Ties: longest LCP wins, then longest checkpoint (a deeper anchor gives the splice more
    /// K rows to rotate for the unchanged tail), then most recently parked (same as `best()`).
    template<class Token>
    Match best_with_partial(int64_t flag, const std::vector<Token>& prompt,
                            const std::vector<ConversationImageKey>& images, bool cvec) const {
        if (flag <= 0) return best(prompt, images, cvec);   // flag off: byte-identical
        Match m;
        // strict first (same as best()).
        for (size_t i = entries_.size(); i-- > 0;) {
            const auto& e = entries_[i];
            if (e.cvec != cvec) continue;
            auto consider = [&](const ConversationCheckpoint& c, bool live) {
                const int64_t n = conversation_prefix(c, prompt, images);
                if (n > m.tokens) m = {i, n, live, serials_[i], false, n};
            };
            consider(e.live, true);
            for (const auto& c : e.checkpoints) consider(c, false);
        }
        // FW-1 splice engagement (2026-10-13): do NOT return at the first strict
        // match.  One conversation's checkpoints are nested prefixes of ONE history,
        // so that history diverges from this request at a single point D and every
        // strict anchor s <= D equals the partial LCP p of the SAME conversation's
        // deeper checkpoints.  A partial mount therefore never reads from a
        // shallower point than the strict match would (p >= s), and its splice may
        // recover K/V up to the anchor length L.  Keep the partial candidate when
        // its checkpoint covers strictly more than the strict anchor (L > s);
        // otherwise the strict match stands (no splice cost when it cannot recover
        // more).  Without this, a mid-file edit always strict-mounts the anchor
        // just below the edit and the splice never engages (partial-branch
        // enter=0 in every steady-state trace - measured 2026-10-13).
        const int64_t strict_tokens = m.tokens;
        // partial pass: longest LCP > 0 and < c.ids.size(); tie-break by deeper anchor
        // (longer checkpoint) then by most recently parked (the loop iterates newest-first).
        Match pm;   // the best partial candidate (tokens = LCP, checkpoint_length = L)
        // [SFX-DBG] probe: per-candidate partial scan visibility.  The steady-state chain showed
        // parked-search reporting match{tokens=2598 partial=0} where the source says a partial
        // candidate (LCP 3424, anchor 5702) should dominate strict 2598 - log every candidate's
        // L/idx/LCP so the steady-state trace can attribute the miss to the candidate set, the
        // LCP guard, or the size guard.  Gated `flag > 0` (probe on the sfx path only).
        const bool dbg_partial = flag > 0;
        if (dbg_partial)
            std::fprintf(stderr, "[SFX-DBG] parked-scan: entries=%zu prompt_len=%zu strict_tokens=%lld:",
                         entries_.size(), prompt.size(), (long long) strict_tokens);
        for (size_t i = entries_.size(); i-- > 0;) {
            const auto& e = entries_[i];
            if (e.cvec != cvec) { if (dbg_partial) std::fprintf(stderr, " e%zu:cvec", i); continue; }
            auto consider = [&](const ConversationCheckpoint& c, bool live) {
                const int64_t L = (int64_t) c.ids.size();
                // T17: the size guard is lifted for the PARTIAL pass - a longer-than-prompt
                // candidate (the parked live) is the most valuable partial anchor.  The strict
                // pass above keeps its own guard (conversation_prefix refuses longer), so this
                // scan alone can surface it.  L < 1 stays refused (empty).
                if (L < 1) {
                    if (dbg_partial) std::fprintf(stderr, " e%zu.%s:L=%lld:skip", i, live ? "live" : "ck", (long long) L);
                    return;
                }
                const int64_t p = conversation_lcp(c, prompt, images);
                if (dbg_partial)
                    std::fprintf(stderr, " e%zu.%s:L=%lld:lcp=%lld", i, live ? "live" : "ck", (long long) L, (long long) p);
                if (p <= 0 || p >= L) return;                         // 0 LCP or strict (already tried)
                if (p > pm.tokens || (p == pm.tokens && L > pm.checkpoint_length))
                    pm = {i, p, live, serials_[i], true, L};
            };
            consider(e.live, true);
            for (const auto& c : e.checkpoints) consider(c, false);
        }
        if (dbg_partial) std::fprintf(stderr, " -> pm{tokens=%lld L=%lld} strict=%lld\n",
                                      (long long) pm.tokens, (long long) pm.checkpoint_length,
                                      (long long) strict_tokens);
        // Cross-conversation safety (review r1 finding 1, 2026-10-13): the nesting
        // argument above is SAME-conversation only.  A DIFFERENT conversation's
        // partial can have a deeper anchor (L > s) with a shallower LCP (p < s);
        // mounting it would read from a shallower point than the strict match.
        // Require p >= s as well as L > s.  (Single-conversation behavior is
        // unchanged: there p >= s always holds, so the added term never binds.)
        if (pm.tokens > 0 && pm.tokens >= strict_tokens && pm.checkpoint_length > strict_tokens) return pm;
        return m;                                            // strict match wins; partial ignored
    }

    SavedConversation take(size_t index) {
        SavedConversation out = std::move(entries_.at(index));
        bytes_ -= entry_bytes(out);
        erase(index);
        return out;
    }

    // The entry without taking it; take()'s index, same bounds check.
    const SavedConversation& at(size_t index) const { return entries_.at(index); }

    // Erase the entry and its budget; does NOT delete its file (the caller owns it until its restore succeeds).
    void remove_at(size_t index) {
        bytes_ -= entry_bytes(entries_.at(index));
        erase(index);
    }

    // A checkpoint inside a parked conversation that a NEW conversation starts with - a sibling subagent sharing the
    // system prompt and tool list, a compacted history - is borrowed: restored from the entry, which stays parked.
    // take() would move the whole conversation into the session and the new request would overwrite everything after
    // that checkpoint, so the other conversation's next turn would read its history again.  A match on the entry's
    // live state or on its deepest checkpoint is the same conversation going on (the deepest one: its client
    // re-rendered the last reply), whose tail is stale: that is still taken.  So is a match that reaches a pinned
    // shared prefix (pin=N): a borrow copies the checkpoints up to the match, and taking it instead keeps one
    // conversation holding the pin, as pin=N expects.
    bool borrows(const Match& m) const {
        if (m.tokens <= 0 || m.live || m.index >= entries_.size() || serials_[m.index] != m.serial) return false;
        size_t deepest = 0;
        for (const auto& c : entries_[m.index].checkpoints) {
            if (c.pinned && (int64_t) c.ids.size() <= m.tokens) return false;
            deepest = std::max(deepest, c.ids.size());
        }
        return (size_t) m.tokens < deepest;
    }
    // The entry with this identity: its index (SIZE_MAX: gone), or the entry itself (nullptr: gone; the pointer is
    // valid until the cache changes).
    size_t index_of(uint64_t serial) const {
        for (size_t i = 0; i < entries_.size(); ++i)
            if (serials_[i] == serial) return i;
        return SIZE_MAX;
    }
    const SavedConversation* find(uint64_t serial) const {
        const size_t i = index_of(serial);
        return i == SIZE_MAX ? nullptr : &entries_[i];
    }
    // While a borrowed entry (the donor) is being restored, parking the outgoing conversation must not evict or drop
    // it: guard() it.  When the outgoing image fits only without it, make_room refuses before evicting anything and
    // guard_blocked() says so: the caller then takes the entry as before, so the conversation that was active a
    // moment ago keeps its place.
    void guard(uint64_t serial) { guarded_ = serial; guard_blocked_ = false; }
    void unguard() { guarded_ = 0; }
    bool guard_blocked() const { return guard_blocked_; }
    size_t borrowed() const { return borrowed_; }
    void count_borrow() { ++borrowed_; }

    // Reserve before allocating a snapshot. held is an incoming image removed
    // with take() but still alive during the exchange; count it against RAM too.
    bool make_room(size_t incoming, size_t held = 0) {
        if (!enabled() || held > budget_ || incoming > budget_ - held) return false;
        if (guarded_ != 0 && index_of(guarded_) != SIZE_MAX) {
            // with everything evict_oldest() can remove gone, would it fit beside the entries that stay (the guarded
            // donor and the last holder of each pinned shared prefix)?  If not, evict nothing
            size_t kept = entries_.size(), kept_bytes = 0;
            for (const auto& e : entries_) kept_bytes += e.bytes();
            each_evictable([&](size_t i) { --kept; kept_bytes -= entries_[i].bytes(); });
            if (kept >= slots_ || kept_bytes > budget_ - held - incoming) { guard_blocked_ = true; return false; }
        }
        if (bytes() > budget_ - held - incoming) drop_reuse();
        while (!entries_.empty() && (entries_.size() >= slots_ || bytes_ > budget_ - held - incoming)) {
            // the oldest entry without a pinned shared prefix leaves, then the oldest pinned one whose prefixes another
            // entry still holds; when each entry left is the last holder of a prefix the new image does not fit (the
            // caller skips parking it - the pinned prefix is what the queries come back to)
            if (!evict_oldest()) return false;
        }
        return true;
    }

    // The parked conversation that has gone unused the longest.  This is make_room()'s loop body, so
    // the parking path can also free RAM one entry at a time on demand (see the physical-RAM admission
    // gate in generate.cpp).  A ConversationBuffer is a list of 16 MiB segments and each segment is its
    // own allocation. Whether freeing it reduces RSS depends on the allocator and platform;
    // the admission caller must resample available memory after eviction. glibc's mmap threshold is dynamic: once
    // it has risen past 16 MiB, dropping an entry only moves the bytes onto the allocator's free list - RSS, and
    // with it /proc/meminfo, does not move until the heap is handed back (see admit_ram's release).
    // A pin keeps its shared prefix parked, not every conversation that branched from it (#2255): the oldest entry
    // without a pin goes first; with only pinned ones left, the oldest whose every pinned prefix another entry
    // still holds goes, so every prefix keeps at least one holder.  A client that pins its system prompt on every
    // request fills every slot with holders of that one prefix, and refusing them all refused every later park.
    // False when none can go (empty, or each entry left is the last holder of a pinned prefix).
    bool evict_oldest() {
        // the oldest entry that holds no pinned shared prefix, then the oldest pinned one whose every prefix another
        // entry still holds; the guarded (borrowed) donor never goes, though it counts as a holder (it stays parked)
        size_t victim = SIZE_MAX;
        for (size_t i = 0; i < entries_.size() && victim == SIZE_MAX; ++i)
            if (!guarded(i) && !entries_[i].pinned()) victim = i;
        for (size_t i = 0; i < entries_.size() && victim == SIZE_MAX; ++i)
            if (!guarded(i) && pins_held_elsewhere(i)) victim = i;
        if (victim == SIZE_MAX) return false;
        if (!entries_[victim].disk_path.empty()) { std::error_code ec; std::filesystem::remove(entries_[victim].disk_path, ec); }
        bytes_ -= entry_bytes(entries_[victim]);
        erase(victim);
        ++evictions_;
        return true;
    }

    // Bytes released by repeated evict_oldest(), excluding retained K/V, the guarded donor and every pin's final holder.
    // This is logical snapshot storage, not a guarantee of RSS reduction: admission must still resample memory.
    // Simulate removals using metadata only; neither the cache nor snapshot payloads are copied/modified.
    size_t evictable_bytes() const {
        size_t total = 0;
        each_evictable([&](size_t i) { total += entries_[i].bytes(); });
        return total;
    }

    // The physical-RAM gate of the parking path: may a snapshot of `allocation` bytes be built while the host keeps
    // `floor` bytes free?  `available()` reads the host's free RAM and `release()` hands what the allocator still
    // holds of freed memory back to the kernel.  Where the first reading refuses, it releases once (make_room may
    // have dropped a conversation just before, and glibc keeps a freed 16 MiB segment - ConversationBuffer's grain -
    // in its heap once its mmap threshold has risen) and reads again.  If that is not enough, parked conversations
    // leave, least recently active first, each followed by a release and a new reading, until the snapshot fits.
    // Nothing leaves if there is no figure to measure by or if what may leave holds less than is missing, as the
    // snapshot would be refused anyway.  `evicted` counts the ones that left.
    template<class Available, class Release>
    bool admit_ram(Available&& available, Release&& release, uint64_t allocation, uint64_t floor, size_t& evicted) {
        auto have = available();
        if (conversation_memory_admit(have, allocation, floor)) return true;
        if (!have) return false;
        release();
        have = available();
        if (conversation_memory_admit(have, allocation, floor)) return true;
        if (!have) return false;
        const uint64_t short_by = *have < floor ? floor - *have + allocation : allocation - (*have - floor);
        if (short_by > evictable_bytes()) return false;
        while (evict_oldest()) {
            ++evicted;
            release();
            have = available();
            if (conversation_memory_admit(have, allocation, floor)) return true;
            if (!have) return false;
        }
        return false;
    }

    // #879: every parked conversation (pinned ones too) and the retained K/V go - after non-finite logits any of them
    // may carry the poison.  Returns how many were parked.
    size_t clear() {
        const size_t n = entries_.size();
        // a cleared cache's parked files are gone with it; error_code ignores a path that does not exist
        for (const auto& e : entries_) if (!e.disk_path.empty()) { std::error_code ec; std::filesystem::remove(e.disk_path, ec); }
        entries_.clear();
        serials_.clear();
        bytes_ = 0;
        drop_reuse();
        return n;
    }

    // The slot count, so a caller that evicts in a loop has a bound it did not invent.
    size_t slots() const { return slots_; }

    // #342: drop the parked entries an outgoing conversation (its live tokens and checkpoint chain) supersedes:
    // the same conversation a turn back, whose DEEPEST checkpoint the outgoing chain still holds, so all it adds
    // is the tail the client rewrote (the reply as it was generated, before the next request re-rendered it) and
    // checkpoints older than that one.  A subagent's successive turns parked one such copy each, and make_room's
    // oldest-first eviction then pushed the parent conversation out after `slots` turns.  Also superseded: an older,
    // shorter state of this conversation with at most a tenth of its tokens not in the outgoing ones (rewritten turns
    // back, when the chain's spread checkpoints no longer hold one near its end).  An entry without checkpoints, or
    // another conversation that only shares the system prompt's root with it, is kept.  Returns how many were dropped.
    size_t drop_superseded(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                           const std::vector<ConversationCheckpoint>& checkpoints, bool cvec,
                           bool lora = true) {
        size_t dropped = 0;
        for (size_t i = 0; i < entries_.size();) {
            const auto& e = entries_[i];
            if (e.lora == lora && superseded_at(e, ids, images, checkpoints, cvec) > 0 &&
                !(guarded_ != 0 && serials_[i] == guarded_)) {
                if (!e.disk_path.empty()) { std::error_code ec; std::filesystem::remove(e.disk_path, ec); }
                bytes_ -= entry_bytes(e);
                erase(i);
                ++dropped;
                continue;
            }
            ++i;
        }
        superseded_ += dropped;
        return dropped;
    }
    size_t superseded() const { return superseded_; }
    /// --conversation-cache-park-gain N: the longest parked entry this conversation's outgoing state supersedes, in
    /// tokens, 0 when it has none.  It is the entry drop_superseded() would remove, so the caller can tell whether a
    /// park now would only rewrite what a turn from now drops: the prefix such an entry holds is this conversation's
    /// prefix, and the conversation only appends, so the copy stays usable for the reuse it was made for.
    size_t parked_prefix_tokens(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                                const std::vector<ConversationCheckpoint>& checkpoints, bool cvec) const {
        size_t longest = 0;
        for (const auto& e : entries_) longest = std::max(longest, superseded_at(e, ids, images, checkpoints, cvec));
        return longest;
    }

    bool put(SavedConversation&& image, size_t held = 0) {
        const size_t n = entry_bytes(image);
        if (!enabled() || held > budget_ || n > budget_ - held) return false;   // make_room's refusal, first
        drop_superseded(image.live.ids, image.live.imgs, image.checkpoints, image.cvec, image.lora);
        if (!make_room(n, held)) return false;
        entries_.push_back(std::move(image));
        serials_.push_back(next_serial_++);
        bytes_ += n;
        return true;
    }

    // Park a disk-backed entry: `meta` is the RAM index (see conversation_meta_only), the payload is the
    // session file at `path` with size `file_bytes` - the budget counts the file, not the index.
    bool put_disk(SavedConversation&& meta, std::string path, size_t file_bytes) {
        meta.disk_path = std::move(path);
        meta.disk_bytes = file_bytes;
        return put(std::move(meta));
    }

private:
    // How many of `e`'s tokens the outgoing conversation holds when it supersedes `e` (drop_superseded), else 0.
    static size_t superseded_at(const SavedConversation& e, const std::vector<int32_t>& ids,
                                const std::vector<ConversationImageKey>& images,
                                const std::vector<ConversationCheckpoint>& checkpoints, bool cvec) {
        const ConversationCheckpoint* deepest = nullptr;
        for (const auto& c : e.checkpoints)
            if (!deepest || c.ids.size() > deepest->ids.size()) deepest = &c;
        if (e.cvec != cvec || !deepest || deepest->ids.empty()) return 0;
        bool held = deepest->ids == ids && deepest->imgs == images;
        for (const auto& k : checkpoints)
            if (!held && k.ids == deepest->ids && k.imgs == deepest->imgs) held = true;
        if (held) return deepest->ids.size();
        const auto& own = e.live.ids;
        const size_t shared = std::mismatch(own.begin(), own.end(), ids.begin(), ids.end()).first - own.begin();
        return ids.size() > own.size() && own.size() - shared <= own.size() / 10 ? shared : 0;
    }
    void erase(size_t index) {
        entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
        serials_.erase(serials_.begin() + (std::ptrdiff_t) index);
    }
    bool guarded(size_t i) const { return guarded_ != 0 && serials_[i] == guarded_; }
    // Every pinned prefix entries_[i] holds is held (same tokens, images and steering mode, pinned) by another parked
    // entry that `removed` does not mark.  The guarded donor counts: it stays parked.
    bool pins_held_elsewhere(size_t i, const uint8_t* removed = nullptr) const {
        const SavedConversation& e = entries_[i];
        for (const auto& c : e.checkpoints) {
            if (!c.pinned) continue;
            bool held = false;
            for (size_t j = 0; j < entries_.size() && !held; ++j) {
                if (j == i || (removed != nullptr && removed[j]) || entries_[j].cvec != e.cvec) continue;
                for (const auto& k : entries_[j].checkpoints)
                    if (k.pinned && k.ids == c.ids && k.imgs == c.imgs) { held = true; break; }
            }
            if (!held) return false;
        }
        return true;
    }
    // f(i) for every entry repeated evict_oldest() would remove, using metadata only (nothing is copied or changed).
    // Removing an entry can only take a holder away, never make another entry evictable, so one pass in age order
    // marks the same set as the repeated calls (which take unpinned entries first; those hold no pins).
    template <class F> void each_evictable(F&& f) const {
        std::array<uint8_t, 64> local{};
        std::vector<uint8_t> overflow;
        uint8_t* removed = local.data();
        if (entries_.size() > local.size()) {
            overflow.assign(entries_.size(), 0);
            removed = overflow.data();
        }
        for (size_t i = 0; i < entries_.size(); ++i)
            if (!guarded(i) && (!entries_[i].pinned() || pins_held_elsewhere(i, removed))) {
                removed[i] = 1;
                f(i);
            }
    }
    void drop_reuse() { reuse_ = {}; reuse_ids_ = {}; reuse_imgs_ = {}; }
    size_t reuse_identity_bytes() const {
        return reuse_ids_.capacity() * sizeof(int32_t) + reuse_imgs_.capacity() * sizeof(ConversationImageKey);
    }

    size_t budget_ = 0, slots_ = 0, bytes_ = 0, evictions_ = 0, superseded_ = 0, borrowed_ = 0;
    bool disk_ = false;   // enable_disk(): budget counts entries' disk_bytes, see the unlink rule at enable_disk
    int64_t min_tokens_ = 0;   // --conversation-cache-min-tokens: 0 parks every conversation, whatever its size
    std::deque<SavedConversation> entries_; // least recently active first
    std::deque<uint64_t> serials_;          // entries_[i]'s identity, never reused
    uint64_t next_serial_ = 1, guarded_ = 0;
    bool guard_blocked_ = false;
    ConversationKvReuse reuse_;
    std::vector<int32_t> reuse_ids_;                        ///< the conversation `reuse_`'s bytes belong to
    std::vector<ConversationImageKey> reuse_imgs_;          ///< (empty: retained without an identity)
};

} // namespace strata::core
