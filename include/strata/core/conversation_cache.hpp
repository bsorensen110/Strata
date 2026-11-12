// CPU-only ownership and matching policy for --serve's parked conversations.
// Token equality, image identity, and steering mode are all required for reuse.
#pragma once

#include "strata/core/conversation_buffer.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace strata::core {

/// #5697: hands the memory this process freed back to the system where the C library keeps it in its heap (glibc:
/// malloc_trim, every arena; freed pages inside the heap are released, not only its top).  MemAvailable, which the
/// RAM gates read, counts only what was handed back.  False where there is no such call (Windows, macOS, musl: their
/// allocators return large blocks on free).  Costs a walk over the heap's free chunks: call it when RAM is short.
inline bool release_free_heap() {
#if defined(__GLIBC__)
    ::malloc_trim(0);
    return true;
#else
    return false;
#endif
}

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
    // STRATA_CKPT_VRAM_SLOTS: the tag of the device copy of `gdn` (0: none).  Set by conversation_checkpoint_save, the
    // only writer of a checkpoint's GDN bytes besides a whole copy (which keeps them equal, so it keeps the tag).
    uint64_t dev_tag = 0;
    // A shared-prefix pin (the request key pin=N): this checkpoint is the read-only prefix many suffix queries branch
    // from, so retention never evicts it (conv_cache.hpp) and a parked conversation holding it stays parked.  A run-time
    // mark only: it is not in the session file, a request that pins the same prefix again sets it.
    bool pinned = false;
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
    // STRATA_PARK_VRAM_MIB: the tag this layer's parts were recorded under in the park's VRAM slot (0: none).  Set by
    // conversation_kv_save; a restore that finds the slot copies from it instead of from the buffers above.
    uint64_t dev_tag = 0;
    // STRATA_PARK_VRAM_MIB: set while a background copy fills the buffers above from the slot; its deleter waits for
    // that copy and releases the slot.  This layer's own token (not shared): the layers outlive their image when they
    // are retained for the next park (ConversationCache::retain), and go in any order.  Last member: it goes before
    // any buffer.
    std::shared_ptr<void> pending;
    size_t bytes() const {
        return k.bytes() + v.bytes() + k_scale.bytes() + v_scale.bytes() + pooled.bytes();
    }
};

struct ConversationKvReuse {
    std::vector<ConversationKv> kv;
    // Original image extent for validation, and the earliest subsequent rewrite.
    int64_t captured_tokens = 0, unchanged_tokens = 0;
    // With a layer split: the later stages' own retained K/V, one per stage, same extents (empty: none)
    std::vector<ConversationKvReuse> stages;
    // The conversation the image was captured from (the first stage's only): a park reuses it only for the prefix it
    // shares with the conversation then in the main session (a batch slot may have replaced that session's K/V since).
    std::vector<int32_t> ids;
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

struct SavedConversation {
    // Runtime compatibility only; NOT a model/weights identity or disk schema.
    std::array<int64_t, 18> geometry{};
    // The session's layer carve the image was captured from ([0, n_layers) on one GPU); restore requires the same.
    int64_t layer_lo = 0, layer_hi = 0;
    ConversationCheckpoint live;
    std::vector<ConversationCheckpoint> checkpoints;
    std::vector<ConversationKv> kv; // main layers followed by the draft layer
    bool cvec = true;
    // with a layer split, the later stages' own images, one per stage, in stage order
    std::vector<SavedConversation> stage_images;
    // DFlash2 drafter ring-log lineage (serialized by Dflash2Drafter::save_lineage); empty when DFlash2 is off.
    std::vector<uint8_t> df2_state;
    // Disk-backed conversation cache (--conversation-cache-disk-mib): the session file holding this entry's running
    // state and K/V on disk, and its size in bytes.  Runtime-only: session_file_write / session_file_read do not
    // serialize these fields, and a RAM-cached entry leaves them empty / 0.
    std::string disk_path;
    size_t disk_bytes = 0;
    // STRATA_PARK_ASYNC: set while a background copy fills this image's K/V; its deleter waits for that copy, and as
    // one of the last members it runs before any buffer is freed (whichever copy of the image goes last)
    std::shared_ptr<void> pending;
    // STRATA_CKPT_ASYNC: the same for the background copy that fills `live.gdn` (the park's running state)
    std::shared_ptr<void> live_pending;

    /// Holds a pinned shared prefix (see ConversationCheckpoint::pinned): the parked-conversation budget keeps it.
    bool pinned() const {
        for (const auto& c : checkpoints) if (c.pinned) return true;
        return false;
    }

    size_t bytes() const {
        size_t n = live.bytes() + checkpoints.capacity() * sizeof(ConversationCheckpoint) +
                   kv.capacity() * sizeof(ConversationKv) + df2_state.capacity();
        for (const auto& s : stage_images) n += s.bytes();
        for (const auto& c : checkpoints) n += c.bytes();
        for (const auto& k : kv) n += k.bytes();
        return n;
    }
};

/// A slim copy of `s` for the disk-backed conversation cache: keeps only the tokens, image keys, checkpoint tokens /
/// image keys / pins, geometry, layer range and steering flag that prefix matching (`ConversationCache::best` /
/// `holds` / `drop_superseded` / `borrows`) reads, and drops all running-state byte arrays, K/V buffers, and df2_state.
inline SavedConversation conversation_meta_only(const SavedConversation& s) {
    SavedConversation m;
    m.geometry = s.geometry;
    m.layer_lo = s.layer_lo;
    m.layer_hi = s.layer_hi;
    m.cvec = s.cvec;
    m.live.ids = s.live.ids;
    m.live.imgs = s.live.imgs;
    m.live.used = s.live.used;
    m.live.pinned = s.live.pinned;
    m.checkpoints.reserve(s.checkpoints.size());
    for (const auto& c : s.checkpoints) {
        ConversationCheckpoint mc;
        mc.ids = c.ids;
        mc.imgs = c.imgs;
        mc.used = c.used;
        mc.pinned = c.pinned;
        m.checkpoints.push_back(std::move(mc));
    }
    return m;
}

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

class ConversationCache {
public:
    struct Match {
        size_t index = 0;
        int64_t tokens = 0;
        bool live = false;
        uint64_t serial = 0;   // the entry's identity; its index shifts when entries before it go
    };

    ConversationCache(size_t budget, size_t slots, int64_t min_tokens = 0)
        : budget_(budget), slots_(slots), min_tokens_(min_tokens) {}
    bool enabled() const { return budget_ != 0 && slots_ != 0; }
    size_t budget() const { return budget_; }
    /// Disk-only mode (--conversation-cache-disk-mib with --conversation-cache-mib 0): entries are session files on
    /// disk (`e.disk_path`, `e.disk_bytes`), `budget_` is the folder's byte cap, and eviction / supersede / clear
    /// unlink the entry's file.
    void enable_disk() { disk_ = true; }
    bool disk() const { return disk_; }
    /// Two-tier L1 RAM + L2 Disk mode (--conversation-cache-mib > 0 and --conversation-cache-disk-mib > 0):
    /// L1 RAM entries (`e.disk_path.empty()`) count toward `bytes_` (under `budget_`), while L2 Disk entries
    /// (`!e.disk_path.empty()`) count toward `l2_disk_bytes_` (under `l2_disk_budget_`).  When an L1 RAM entry is
    /// evicted, `spill_` writes it to a `.sess` file and converts the entry in-place into an L2 Disk entry.
    using SpillFn = std::function<bool(const SavedConversation& image, std::string& path_out, size_t& file_bytes_out)>;
    void set_l2_disk(size_t disk_budget, SpillFn spill = {}) {
        l2_disk_budget_ = disk_budget;
        spill_ = std::move(spill);
    }
    bool has_l2_disk() const { return !disk_ && l2_disk_budget_ > 0; }
    size_t l2_disk_budget() const { return l2_disk_budget_; }
    size_t l2_disk_bytes() const { return l2_disk_bytes_; }
    static size_t entry_bytes(const SavedConversation& e) { return e.disk_path.empty() ? e.bytes() : e.disk_bytes; }
    /// `--conversation-cache-min-tokens`: whether a conversation of this length is worth one of `slots`.  Every
    /// parked conversation costs a whole slot whatever its length, so a client that interleaves short side requests
    /// with one long conversation spends the cache on the side requests and evicts the long one (oldest first) -
    /// exactly the reuse it wanted.  0 parks every conversation, whatever its size.
    bool wants(int64_t tokens) const { return min_tokens_ <= 0 || tokens >= min_tokens_; }
    size_t bytes() const { return bytes_ + reuse_.bytes(); }
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
    // `ids`: the conversation the K/V belongs to (take_reuse_for refuses it for any other; none: never reused).
    void retain(std::vector<ConversationKv>&& kv, int64_t tokens,
                std::vector<std::vector<ConversationKv>>&& stage_kv = {}, std::vector<int32_t> ids = {}) {
        reuse_ = {};
        ConversationKvReuse candidate{std::move(kv), tokens, tokens, {}, std::move(ids)};
        for (auto& k : stage_kv) candidate.stages.push_back(ConversationKvReuse{std::move(k), tokens, tokens, {}, {}});
        if (enabled() && candidate.bytes() <= budget_ - bytes_) reuse_ = std::move(candidate);
    }
    void limit_reuse(int64_t first_dirty) {
        reuse_.unchanged_tokens = std::min(reuse_.unchanged_tokens, first_dirty);
        for (auto& s : reuse_.stages) s.unchanged_tokens = std::min(s.unchanged_tokens, first_dirty);
        if (reuse_.unchanged_tokens <= 0) reuse_ = {};
    }
    ConversationKvReuse take_reuse() { return std::exchange(reuse_, {}); }
    // The retained K/V for parking `live`: limited to the prefix `live` shares with the conversation it was captured
    // from, and nothing when they share none.  Without this a park after a batch slot gave its conversation back to the
    // main session wrote the RESTORED conversation's K/V into the new one's image (#1503, a cross-conversation leak).
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
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const {
        Match best;
        // Ties prefer the most recently parked branch. The caller prefers its
        // already-active state when that offers the same prefix length.
        for (size_t i = entries_.size(); i-- > 0;) {
            const auto& e = entries_[i];
            if (e.cvec != cvec) continue;
            auto consider = [&](const ConversationCheckpoint& c, bool live) {
                const int64_t n = conversation_prefix(c, prompt, images);
                if (n > best.tokens) best = {i, n, live, serials_[i]};
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

    SavedConversation take(size_t index) {
        SavedConversation out = std::move(entries_.at(index));
        debit_entry(out);
        erase(index);
        return out;
    }

    const SavedConversation& at(size_t index) const { return entries_.at(index); }

    /// Removes the entry at `index` and debits its budget without unlinking `disk_path` (the caller owns the file:
    /// either it just restored from it and will unlink it after `conversation_snapshot_restore`, or the read failed on
    /// transient host memory pressure and the file is left on disk for a later run).
    void remove_at(size_t index) {
        debit_entry(entries_.at(index));
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
        auto stays = [&](size_t i) { return entries_[i].pinned() || (guarded_ != 0 && serials_[i] == guarded_); };
        if (guarded_ != 0 && index_of(guarded_) != SIZE_MAX) {
            // with everything else gone, would it fit beside the entries that stay (the guarded donor and any pinned
            // shared prefix)?  If not, evict nothing
            size_t kept = 0, kept_bytes = 0;
            for (size_t i = 0; i < entries_.size(); ++i)
                if (stays(i)) {
                    ++kept;
                    if (disk_ || entries_[i].disk_path.empty()) kept_bytes += entry_bytes(entries_[i]);
                }
            if (kept >= slots_ || kept_bytes > budget_ - held - incoming) { guard_blocked_ = true; return false; }
        }
        if (bytes() > budget_ - held - incoming) reuse_ = {};
        while (!entries_.empty() && bytes_ > budget_ - held - incoming) {
            if (has_l2_disk() ? !evict_oldest_l1() : !evict_oldest_slot()) return false;
        }
        while (!entries_.empty() && entries_.size() >= slots_) {
            if (!evict_oldest_slot()) return false;
        }
        return true;
    }

    // The parked conversation that has gone unused the longest, for the callers that free RAM one entry at a time
    // and read MemAvailable after each (the physical-RAM admission gate in generate.cpp, session_save_reclaim).
    // #5697: a ConversationBuffer's segments are plain allocations of up to 16 MiB; once glibc's dynamic mmap
    // threshold has risen (the first freed segment raises it), they come from the heap, and free() hands back only
    // the heap's top - an entry below a newer one stayed resident and the gate read no change.  So the freed memory
    // is handed back here (release_free_heap).  make_room() drops entries without it: its room is for the image
    // that is allocated right after, which reuses that memory.
    // False when none can go (empty, or only entries that hold a pinned shared prefix are left).
    bool evict_oldest() {
        if (has_l2_disk()) {
            if (!evict_oldest_l1()) return false;
        } else if (!evict_oldest_slot()) {
            return false;
        }
        release_free_heap();
        return true;
    }

    // #879: every parked conversation (pinned ones too) and the retained K/V go - after non-finite logits any of them
    // may carry the poison.  Returns how many were parked.
    size_t clear() {
        const size_t n = entries_.size();
        for (const auto& e : entries_) unlink_if_disk(e);
        entries_.clear();
        serials_.clear();
        bytes_ = 0;
        l2_disk_bytes_ = 0;
        reuse_ = {};
        return n;
    }

    // The slot count, so a caller that evicts in a loop has a bound it did not invent.
    size_t slots() const { return slots_; }

    // #342: drop the parked entries an outgoing conversation (its live tokens and checkpoint chain) supersedes:
    // the same conversation a turn back, whose DEEPEST checkpoint the outgoing chain still holds, so all it adds
    // is the tail the client rewrote (the reply as it was generated, before the next request re-rendered it) and
    // checkpoints older than that one.  A subagent's successive turns parked one such copy each, and make_room's
    // oldest-first eviction then pushed the parent conversation out after `slots` turns.  An entry without
    // checkpoints, or whose deepest checkpoint the outgoing chain does not hold (another conversation that only
    // shares the system prompt's root with it), is kept.  Returns how many were dropped.
    size_t drop_superseded(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                           const std::vector<ConversationCheckpoint>& checkpoints, bool cvec) {
        auto held = [&](const ConversationCheckpoint& c) {
            if (c.ids == ids && c.imgs == images) return true;
            for (const auto& k : checkpoints)
                if (k.ids == c.ids && k.imgs == c.imgs) return true;
            return false;
        };
        size_t dropped = 0;
        for (size_t i = 0; i < entries_.size();) {
            const auto& e = entries_[i];
            const ConversationCheckpoint* deepest = nullptr;
            for (const auto& c : e.checkpoints)
                if (!deepest || c.ids.size() > deepest->ids.size()) deepest = &c;
            if (e.cvec == cvec && deepest && !deepest->ids.empty() && held(*deepest) &&
                !(guarded_ != 0 && serials_[i] == guarded_)) {
                debit_entry(e);
                unlink_if_disk(e);
                erase(i);
                ++dropped;
                continue;
            }
            ++i;
        }
        superseded_ += dropped;
        return dropped;
    }
    // A batch slot's chain holds one checkpoint (the one its admission copied in), while the main session's parked copy of the same
    // conversation keeps the whole chain - the root checkpoint at the end of an agent's system prompt and tools among them, the one its
    // next conversations borrow (borrows()).  Parking the slot (#2526) would drop that copy as superseded (#342: the slot's chain holds its
    // deepest checkpoint) and the root with it: every sibling then reads the shared prefix again (#4420, 0.2.6 -> 0.2.7: 323 ms -> 4130 ms
    // for a 5 K-token prefix).  True when drop_superseded(ids, images, checkpoints, cvec) would drop an entry that has a checkpoint the
    // given chain does not hold.  Nothing is changed.
    bool supersedes_richer(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                           const std::vector<ConversationCheckpoint>& checkpoints, bool cvec) const {
        auto held = [&](const ConversationCheckpoint& c) {
            if (c.ids == ids && c.imgs == images) return true;
            for (const auto& k : checkpoints)
                if (k.ids == c.ids && k.imgs == c.imgs) return true;
            return false;
        };
        for (size_t i = 0; i < entries_.size(); ++i) {
            const auto& e = entries_[i];
            const ConversationCheckpoint* deepest = nullptr;
            for (const auto& c : e.checkpoints)
                if (!deepest || c.ids.size() > deepest->ids.size()) deepest = &c;
            if (e.cvec != cvec || !deepest || deepest->ids.empty() || !held(*deepest) ||
                (guarded_ != 0 && serials_[i] == guarded_))
                continue;
            for (const auto& c : e.checkpoints)
                if (!held(c)) return true;
        }
        return false;
    }
    size_t superseded() const { return superseded_; }
    /// --conversation-cache-park-gain N: the longest parked entry this conversation's outgoing state supersedes, in
    /// tokens, 0 when it has none.  It is the entry drop_superseded() would remove, so the caller can tell whether a
    /// park now would only rewrite what a turn from now drops: the prefix such an entry holds is this conversation's
    /// prefix, and the conversation only appends, so the copy stays usable for the reuse it was made for.
    size_t parked_prefix_tokens(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                                const std::vector<ConversationCheckpoint>& checkpoints, bool cvec) const {
        size_t longest = 0;
        for (const auto& e : entries_) {
            if (e.cvec != cvec) continue;
            const ConversationCheckpoint* deepest = nullptr;
            for (const auto& c : e.checkpoints)
                if (!deepest || c.ids.size() > deepest->ids.size()) deepest = &c;
            if (!deepest || deepest->ids.empty()) continue;
            bool holds = deepest->ids == ids && deepest->imgs == images;
            for (const auto& k : checkpoints)
                if (!holds && k.ids == deepest->ids && k.imgs == deepest->imgs) holds = true;
            if (holds) longest = std::max(longest, deepest->ids.size());
        }
        return longest;
    }

    bool put(SavedConversation&& image, size_t held = 0) {
        const size_t n = entry_bytes(image);
        if (!enabled() || held > budget_ || n > budget_ - held) return false;   // make_room's refusal, first
        drop_superseded(image.live.ids, image.live.imgs, image.checkpoints, image.cvec);
        if (!make_room(n, held)) return false;
        entries_.push_back(std::move(image));
        serials_.push_back(next_serial_++);
        bytes_ += n;
        return true;
    }

    /// Indexes a disk-backed entry (`meta` holds only the slim prefix-matching metadata; `path` is its `.sess` file
    /// of `file_bytes` on disk).  Drops any superseded entries (unlinking their files) and evicts oldest unpinned
    /// entries if needed to fit the disk budget and slot limit.
    bool put_disk(SavedConversation&& meta, std::string path, size_t file_bytes) {
        if (path.empty() || file_bytes == 0) return false;
        meta.disk_path = std::move(path);
        meta.disk_bytes = file_bytes;
        if (disk_) return put(std::move(meta), 0);
        if (!has_l2_disk() || slots_ == 0 || file_bytes > l2_disk_budget_) return false;
        drop_superseded(meta.live.ids, meta.live.imgs, meta.checkpoints, meta.cvec);
        auto stays = [&](size_t i) { return entries_[i].pinned() || (guarded_ != 0 && serials_[i] == guarded_); };
        if (guarded_ != 0 && index_of(guarded_) != SIZE_MAX) {
            size_t kept = 0, kept_l2 = 0;
            for (size_t i = 0; i < entries_.size(); ++i)
                if (stays(i)) {
                    ++kept;
                    if (!entries_[i].disk_path.empty()) kept_l2 += entries_[i].disk_bytes;
                }
            if (kept >= slots_ || kept_l2 > l2_disk_budget_ - file_bytes) { guard_blocked_ = true; return false; }
        }
        while (l2_disk_bytes_ > l2_disk_budget_ - file_bytes) {
            if (!evict_oldest_l2_disk(SIZE_MAX)) return false;
        }
        while (entries_.size() >= slots_) {
            if (!evict_oldest_slot()) return false;
        }
        entries_.push_back(std::move(meta));
        serials_.push_back(next_serial_++);
        l2_disk_bytes_ += file_bytes;
        return true;
    }

private:
    static void unlink_if_disk(const SavedConversation& e) {
        if (!e.disk_path.empty()) {
            std::error_code ec;
            std::filesystem::remove(e.disk_path, ec);
        }
    }
    void debit_entry(const SavedConversation& e) {
        if (!disk_ && !e.disk_path.empty()) l2_disk_bytes_ -= std::min(l2_disk_bytes_, e.disk_bytes);
        else bytes_ -= std::min(bytes_, entry_bytes(e));
    }
    bool evict_oldest_l1() {
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (!entries_[i].disk_path.empty()) continue;
            if (entries_[i].pinned() || (guarded_ != 0 && serials_[i] == guarded_)) continue;
            bytes_ -= entries_[i].bytes();
            ++evictions_;
            std::string path;
            size_t fb = 0;
            if (has_l2_disk() && spill_ && spill_(entries_[i], path, fb) && fb > 0 && fb <= l2_disk_budget_) {
                while (l2_disk_bytes_ + fb > l2_disk_budget_) {
                    if (!evict_oldest_l2_disk(i)) break;
                }
                if (l2_disk_bytes_ + fb <= l2_disk_budget_) {
                    entries_[i] = conversation_meta_only(entries_[i]);
                    entries_[i].disk_path = std::move(path);
                    entries_[i].disk_bytes = fb;
                    l2_disk_bytes_ += fb;
                    return true;
                }
                std::error_code ec;
                std::filesystem::remove(path, ec);
            }
            erase(i);
            return true;
        }
        return false;
    }
    bool evict_oldest_l2_disk(size_t skip_index) {
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (i == skip_index || entries_[i].disk_path.empty()) continue;
            if (entries_[i].pinned() || (guarded_ != 0 && serials_[i] == guarded_)) continue;
            debit_entry(entries_[i]);
            unlink_if_disk(entries_[i]);
            erase(i);
            ++evictions_;
            return true;
        }
        return false;
    }
    bool evict_oldest_slot() {
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (entries_[i].pinned() || (guarded_ != 0 && serials_[i] == guarded_)) continue;
            debit_entry(entries_[i]);
            unlink_if_disk(entries_[i]);
            erase(i);
            ++evictions_;
            return true;
        }
        return false;
    }
    void erase(size_t index) {
        entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
        serials_.erase(serials_.begin() + (std::ptrdiff_t) index);
    }
    size_t budget_ = 0, slots_ = 0, bytes_ = 0, evictions_ = 0, superseded_ = 0, borrowed_ = 0;
    size_t l2_disk_budget_ = 0, l2_disk_bytes_ = 0;
    int64_t min_tokens_ = 0;   // --conversation-cache-min-tokens: 0 parks every conversation, whatever its size
    bool disk_ = false;        // --conversation-cache-disk-mib (disk-only): entries are session files on disk
    SpillFn spill_;
    std::deque<SavedConversation> entries_; // least recently active first
    std::deque<uint64_t> serials_;          // entries_[i]'s identity, never reused
    uint64_t next_serial_ = 1, guarded_ = 0;
    bool guard_blocked_ = false;
    ConversationKvReuse reuse_;
};

} // namespace strata::core
