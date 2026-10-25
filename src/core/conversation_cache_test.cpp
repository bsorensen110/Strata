#include "strata/core/conversation_cache.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>

using namespace strata::core;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", description); std::exit(1); }
}
// A host whose free RAM is `base` plus the share (num/den) of the bytes the cache has dropped that the allocator has
// handed back so far: release() is what hands them back.
struct Host {
    ConversationCache& cache;
    size_t start;
    uint64_t base;
    unsigned num, den;
    size_t released = 0;
    uint64_t returned = 0;
    uint64_t cap = UINT64_MAX;   // the most the allocator will hand back in all
    uint64_t noise = 0;          // free RAM another process takes from the second release on
    std::optional<uint64_t> available() const { return base + returned - (released >= 2 ? noise : 0); }
    auto available_fn() { return [this] { return available(); }; }
    auto release_fn() { return [this] { ++released; returned = std::min<uint64_t>((start - cache.bytes()) * num / den, cap); }; }
};
SavedConversation image(std::initializer_list<int32_t> ids, bool cvec = true) {
    SavedConversation s;
    s.live.ids = ids;
    s.live.gdn.resize(64, 7);
    s.cvec = cvec;
    return s;
}
// a conversation `ids` that branched from the pinned shared prefix `pin`: its chain holds the pin and, deeper, a
// checkpoint of its own (one token short of `ids`), so another holder does not supersede it (drop_superseded)
SavedConversation holder(std::initializer_list<int32_t> ids, std::initializer_list<int32_t> pin) {
    SavedConversation s = image(ids);
    s.checkpoints.push_back(ConversationCheckpoint{});
    s.checkpoints.back().ids = pin;
    s.checkpoints.back().pinned = true;
    s.checkpoints.back().gdn.resize(64, 1);
    s.checkpoints.push_back(ConversationCheckpoint{});
    s.checkpoints.back().ids.assign(ids.begin(), ids.end() - 1);
    s.checkpoints.back().gdn.resize(64, 2);
    return s;
}
}

namespace {
void test_evictable_bytes() {
    // All 4,096 assignments of three possible pins to four distinct conversations.
    for (unsigned pattern = 0; pattern < 4096; ++pattern) {
        ConversationCache cache(1u << 24, 8);
        unsigned pins = 0;
        for (unsigned i = 0; i < 4; ++i) {
            auto s = image({80 + (int) i, 90, 100, 110});
            const unsigned mask = (pattern >> (3 * i)) & 7;
            pins |= mask;
            for (unsigned p = 0; p < 3; ++p) if (mask & (1u << p)) {
                ConversationCheckpoint c;
                c.ids = {1, (int) p + 2}; c.pinned = true; c.gdn.resize(17 + i);
                s.checkpoints.push_back(std::move(c));
            }
            ConversationCheckpoint unique;
            unique.ids = {80 + (int) i, 90, 100};
            s.checkpoints.push_back(std::move(unique));
            check(cache.put(std::move(s)), "accounting fixture parks distinct branches");
        }
        ConversationKv retained;
        retained.k.resize(100);
        std::vector<ConversationKv> layers;
        layers.push_back(std::move(retained));
        cache.retain(std::move(layers), 10);
        const size_t bytes = cache.bytes(), entries = cache.size(), evictions = cache.evictions();
        const size_t preview = cache.evictable_bytes(), reuse = cache.retained_bytes();
        check(cache.bytes() == bytes && cache.size() == entries && cache.evictions() == evictions,
              "preview is non-mutating");
        size_t freed = 0;
        while (cache.evict_oldest()) freed = bytes - cache.bytes();
        check(preview == freed, "preview exactly matches repeated eviction");
        check(cache.evictable_bytes() == 0 && cache.retained_bytes() == reuse,
              "last holders and retained KV are not evictable");
        for (unsigned p = 0; p < 3; ++p) if (pins & (1u << p))
            check(cache.best(std::vector<int32_t>{1, (int) p + 2, 999}, {}, true).tokens == 2,
                  "every initial pinned prefix survives the drain");
    }
    for (size_t n : {size_t(0), size_t(1), size_t(3), size_t(64), size_t(65)}) {
        ConversationCache cache(1u << 24, 128);
        for (size_t i = 0; i < n; ++i)
            check(cache.put(holder({1,2,(int)i+10,0}, {1,2})), "park duplicate pin holders across scratch boundary");
        const size_t before = cache.bytes(), preview = cache.evictable_bytes();
        const size_t one = n ? before / n : 0;
        check(preview == (n ? (n - 1) * one : 0), "only redundant holders count, including large caches");
        while (cache.evict_oldest()) {}
        check(preview == before - cache.bytes(), "scratch boundary preview matches actual drain");
    }
    for (bool images : {false, true}) {
        ConversationCache cache(1u << 24, 8);
        auto a = holder({1,2,3,4}, {1,2}), b = holder({1,2,5,6}, {1,2});
        if (images) { a.checkpoints[0].imgs = {{0,1}}; b.checkpoints[0].imgs = {{0,2}}; }
        else b.cvec = false;
        check(cache.put(std::move(a)) && cache.put(std::move(b)), "park different pin identities");
        check(cache.evictable_bytes() == 0 && !cache.evict_oldest(), "image and steering identity keep sole holders");
    }
}
}

int main() {
    test_evictable_bytes();
    {
        ConversationBuffer bytes;
        const size_t first = ConversationBuffer::segment_bytes + 17;
        const size_t peak = bytes.allocation_peak(first);
        bytes.resize(first, 7);
        check(bytes.bytes() <= peak, "segmented payload and directory fit admitted bytes");
        uint8_t* original = nullptr;
        bytes.visit(0, 1, [&](uint8_t* p, size_t, size_t) { original = p; return true; });
        const size_t grown_peak = bytes.allocation_peak(first + 99);
        bytes.resize(first + 99, 9);
        bytes.visit(0, 1, [&](uint8_t* p, size_t, size_t) {
            check(p == original, "appending preserves existing payload addresses"); return true;
        });
        check(bytes.bytes() <= grown_peak, "growth including transient directory fits admission");
        std::vector<uint8_t> tail(116);
        check(bytes.read(tail.data(), ConversationBuffer::segment_bytes, tail.size()), "read spans segment boundaries");
        check(std::all_of(tail.begin(), tail.begin()+17, [](auto v){return v==7;}) &&
              std::all_of(tail.begin()+17, tail.end(), [](auto v){return v==9;}), "growth preserves prefix and initializes only suffix");
        auto copy = bytes;
        bytes.resize(ConversationBuffer::segment_bytes + 10);
        bytes.resize(first, 7); bytes.resize(first + 99, 9);
        check(bytes == copy, "equality ignores differing segmentation after rewind and regrowth");
        check(!bytes.read(tail.data(), bytes.size()-1, 2), "range check rejects a truncated payload");
        check(bytes.allocation_peak(SIZE_MAX) == SIZE_MAX, "allocation estimate rejects overflow");
    }
    {
        ConversationBuffer bytes;
        for (size_t n=1;n<=4096;++n) {
            const size_t peak = bytes.allocation_peak(n*17);
            bytes.resize(n*17,7);
            check(bytes.bytes() <= peak,"small append stays within the predicted allocation peak");
        }
        size_t segments = 0;
        bytes.visit(0,bytes.size(),[&](const uint8_t*,size_t,size_t){++segments;return true;});
        check(segments <= 4,"thousands of small turns do not create thousands of restore transfers");
    }
    {
        // a parked conversation's K/V buffer (16 MiB, as captured) grows by one turn: the reserve for later turns is
        // an eighth of the buffer at most, not another 16 MiB segment
        ConversationBuffer bytes;
        bytes.resize(ConversationBuffer::segment_bytes, 3);
        const size_t captured = bytes.bytes();
        check(captured - bytes.size() < 1024, "a fresh capture reserves nothing");
        const size_t turn = 1 << 20;
        const size_t peak = bytes.allocation_peak(bytes.size() + turn);
        bytes.resize(bytes.size() + turn, 4);
        check(bytes.bytes() <= peak, "a turn's growth stays within the predicted allocation peak");
        check(bytes.bytes() - captured <= turn + ConversationBuffer::segment_bytes / 8 + 1024,
              "the reserve for later turns is at most an eighth of the buffer");
        const size_t reserved = bytes.bytes();
        bytes.resize(bytes.size() + turn / 2, 5);
        check(bytes.bytes() == reserved, "the next small turn fills the reserve instead of allocating");
        size_t segments = 0;
        bytes.visit(0, bytes.size(), [&](const uint8_t*, size_t, size_t) { ++segments; return true; });
        check(segments == 2, "the reserve extends the turn's segment");
    }
    {
        // many turns of 1 MiB onto a 16 MiB buffer: the reserve grows with the buffer, so the segments do not
        // multiply one per turn, and the reserve never exceeds an eighth of the buffer
        ConversationBuffer bytes;
        bytes.resize(ConversationBuffer::segment_bytes, 1);
        for (int k = 0; k < 256; ++k) {
            const size_t before = bytes.size();
            const size_t peak = bytes.allocation_peak(before + (1 << 20));
            bytes.resize(before + (1 << 20), 2);
            check(bytes.bytes() <= peak, "each turn stays within the predicted allocation peak");
            check(bytes.bytes() - bytes.size() <= before / 8 + 4096, "the reserve stays within an eighth");
        }
        size_t segments = 0;
        bytes.visit(0, bytes.size(), [&](const uint8_t*, size_t, size_t) { ++segments; return true; });
        check(segments <= 64, "256 turns of 1 MiB take far fewer than 256 segments");
    }
    {
        ConversationKv layer;
        layer.k.resize(400);
        std::vector<ConversationKv> layers;
        layers.push_back(std::move(layer));
        const size_t retained = layers.capacity()*sizeof(ConversationKv) + layers[0].bytes();
        const size_t parked = image({1,2,3}).bytes();
        ConversationCache cache(retained + parked, 4);
        check(cache.put(image({1,2,3})), "park before retaining active storage");
        cache.retain(std::move(layers), 16);
        check(cache.bytes() == retained + parked && cache.size() == 1, "active retained storage consumes bytes but no parked slot");
        cache.limit_reuse(9); cache.limit_reuse(12);
        auto reuse = cache.take_reuse();
        check(reuse.unchanged_tokens == 9, "a later continuation cannot undo a rewind's dirty boundary");
        check(cache.bytes() == parked, "taking retained buffers releases their budget accounting");
        cache.retain(std::move(reuse.kv), 16);
        check(!cache.can_fit(parked),"retained storage is counted when checking a capture without eviction");
        check(cache.retained_bytes() == retained && cache.size() == 1 && cache.evictions() == 0,
              "optional reuse admission does not evict or release anything");
        check(!cache.can_fit(SIZE_MAX) && !cache.can_fit(1,SIZE_MAX),"non-mutating reservation rejects overflow");
        check(cache.make_room(parked), "reservation can discard optional active buffers");
        check(cache.retained_bytes() == 0 && cache.size() == 1 && cache.evictions() == 0,
              "pressure drops retained storage before evicting parked conversations");
    }
    const std::vector<int64_t> a = {1, 2, 3, 4}, b = {9, 8, 7, 6};
    {
        ConversationCache cache(1024, 2);
        check(cache.put(image({1, 2, 3})), "park A");
        check(cache.put(image({9, 8, 7})), "park B");
        auto match = cache.best(a, {}, true);
        check(match.tokens == 3 && match.live, "A/B/A: recover A");
        auto restored = cache.take(match.index);
        check(restored.live.ids == std::vector<int32_t>({1, 2, 3}), "taking selected A preserves identity");
        check(cache.size() == 1 && cache.best(b, {}, true).tokens == 3, "B remains parked");
        check(cache.bytes() == image({9,8,7}).bytes(), "byte accounting after take");
        check(cache.put(std::move(restored)), "park returned A as newest");
        check(cache.put(image({5, 6})), "evict oldest by slot limit");
        check(cache.best(b, {}, true).tokens == 0 && cache.best(a, {}, true).tokens == 3, "B evicted before A");
        check(cache.evictions() == 1, "eviction counter");
    }
    {   // the physical-RAM gate of the parking path: a full cache gives RAM back until the snapshot fits
        const size_t one = image({1,2,3}).bytes();
        const uint64_t floor = 1000;
        ConversationCache cache(one * 8, 8);
        for (int32_t k = 1; k <= 4; ++k) check(cache.put(image({k,2,3})), "parked");
        Host host{cache, cache.bytes(), floor + one, 1, 1};   // one conversation's worth of room beside the floor
        size_t evicted = 0;
        check(!conversation_memory_admit(host.available(), one * 3, floor), "the plain check refuses a snapshot of three");
        check(cache.admit_ram(host.available_fn(), host.release_fn(), one, floor, evicted) && evicted == 0 &&
              host.released == 0 && cache.size() == 4, "a snapshot that fits beside the floor evicts nothing");
        check(cache.admit_ram(host.available_fn(), host.release_fn(), one * 3, floor, evicted) && evicted == 2 &&
              host.released == 3 && cache.size() == 2 && cache.evictions() == 2, "two conversations give back what is missing");
        check(cache.best(std::vector<int32_t>{1,2,3,0}, {}, true).tokens == 0 &&
              cache.best(std::vector<int32_t>{2,2,3,0}, {}, true).tokens == 0, "the oldest two are the ones gone");
        check(cache.best(std::vector<int32_t>{3,2,3,0}, {}, true).tokens == 3 &&
              cache.best(std::vector<int32_t>{4,2,3,0}, {}, true).tokens == 3, "the newest two stay parked");
        ConversationCache edge(one * 8, 8);
        for (int32_t k = 1; k <= 4; ++k) check(edge.put(image({k,2,3})), "parked");
        Host edge_host{edge, edge.bytes(), floor, 1, 1};
        evicted = 0;
        check(edge.admit_ram(edge_host.available_fn(), edge_host.release_fn(), one + 1, floor, evicted) && evicted == 2,
              "one byte past a conversation takes the next one");
        ConversationCache low(one * 8, 8);
        for (int32_t k = 1; k <= 4; ++k) check(low.put(image({k,2,3})), "parked");
        Host low_host{low, low.bytes(), floor - one / 2, 1, 1};   // already under the floor, and the figure is read again
        evicted = 0;
        check(low.admit_ram(low_host.available_fn(), low_host.release_fn(), one, floor, evicted) && evicted == 2,
              "RAM given back can lift a host that is under the floor");
    }
    {   // the RAM a conversation held does not all come back at once: the gate goes on until the snapshot fits
        const size_t one = image({1,2,3}).bytes();
        const uint64_t floor = 1000;
        ConversationCache half(one * 8, 8);
        for (int32_t k = 1; k <= 6; ++k) check(half.put(image({k,2,3})), "parked");
        Host half_host{half, half.bytes(), floor + one, 1, 2};   // half of each conversation's bytes come back
        size_t evicted = 0;
        check(half.admit_ram(half_host.available_fn(), half_host.release_fn(), one * 3, floor, evicted) && evicted == 4 &&
              half.size() == 2, "conversations whose RAM half comes back take four to cover two");
        ConversationCache slow(one * 8, 8);
        for (int32_t k = 1; k <= 7; ++k) check(slow.put(image({k,2,3})), "parked");
        Host slow_host{slow, slow.bytes(), floor + one, 1, 5};   // a fifth of each conversation's bytes come back
        evicted = 0;
        check(slow.admit_ram(slow_host.available_fn(), slow_host.release_fn(), one * 2, floor, evicted) && evicted == 5 &&
              slow.size() == 2, "a conversation that gives back little is still worth another when it helps");
        ConversationCache noisy(one * 8, 8);
        for (int32_t k = 1; k <= 6; ++k) check(noisy.put(image({k,2,3})), "parked");
        Host noisy_host{noisy, noisy.bytes(), floor + one, 1, 1, 0, 0, UINT64_MAX, one + one / 2};   // another process takes RAM meanwhile
        evicted = 0;
        check(!conversation_memory_admit(noisy_host.available(), one * 3, floor), "the plain check refuses a snapshot of three");
        check(noisy.admit_ram(noisy_host.available_fn(), noisy_host.release_fn(), one * 3, floor, evicted) && evicted == 4 &&
              noisy.size() == 2, "free RAM that falls after an eviction does not end the loop while more can leave");
        ConversationCache kept(one * 8, 8);
        for (int32_t k = 1; k <= 4; ++k) check(kept.put(image({k,2,3})), "parked");
        Host kept_host{kept, kept.bytes(), floor + one, 0, 1};   // the allocator keeps it all
        evicted = 0;
        check(!kept.admit_ram(kept_host.available_fn(), kept_host.release_fn(), one * 3, floor, evicted) && evicted == 4 &&
              kept.size() == 0, "when no RAM comes back the conversations that may leave are all it can cost");
    }
    {   // make_room dropped the last parked conversation just before the gate and the allocator still holds it
        const size_t one = image({1,2,3}).bytes();
        const uint64_t floor = 1000;
        ConversationCache cache(one * 8, 8);
        check(cache.put(image({1,2,3})), "parked");
        Host host{cache, cache.bytes(), floor, 1, 1};
        check(cache.evict_oldest() && cache.size() == 0, "make_room drops it");
        size_t evicted = 0;
        check(cache.admit_ram(host.available_fn(), host.release_fn(), one, floor, evicted) && evicted == 0 && host.released == 1,
              "one release before any further eviction shows the RAM");
    }
    {   // when eviction cannot help, nothing is evicted and the snapshot is refused as before
        const size_t one = image({1,2,3}).bytes();
        const uint64_t floor = 1000;
        ConversationCache cache(one * 8, 8);
        for (int32_t k = 1; k <= 3; ++k) check(cache.put(image({k,2,3})), "parked");
        Host host{cache, cache.bytes(), floor, 1, 1};
        size_t evicted = 0;
        check(!cache.admit_ram(host.available_fn(), host.release_fn(), one * 3 + 1, floor, evicted) && evicted == 0 &&
              host.released == 1 && cache.size() == 3 && cache.evictions() == 0, "a shortfall beyond what is parked evicts nothing");
        check(!cache.admit_ram([] { return std::optional<uint64_t>(); }, host.release_fn(), one, floor, evicted) && evicted == 0 &&
              cache.size() == 3, "no free-RAM figure: nothing is evicted");
        check(cache.admit_ram(host.available_fn(), host.release_fn(), one * 3, floor, evicted) && evicted == 3 && cache.size() == 0,
              "a shortfall of everything takes everything");
        Host empty{cache, cache.bytes(), floor, 1, 1};
        evicted = 0;
        check(!cache.admit_ram(empty.available_fn(), empty.release_fn(), one, floor, evicted) && evicted == 0 && !cache.evict_oldest(),
              "an empty cache has nothing to give");
    }
    {
        // evict_oldest(): the physical-RAM admission gate in generate.cpp frees parked conversations
        // one at a time and re-checks the host's free memory, so this has to be exactly the oldest-first
        // step make_room() takes - same accounting, same counters - and it has to stop at empty.
        ConversationCache cache(4096, 4);
        cache.put(image({1, 2, 3}));
        cache.put(image({9, 8, 7}));
        check(cache.size() == 2 && cache.slots() == 4, "slots() reports the configured limit");
        const size_t both = cache.bytes();
        check(cache.evict_oldest(), "evict_oldest reports an eviction");
        check(cache.size() == 1 && cache.evictions() == 1, "evict_oldest drops exactly one, oldest first");
        check(cache.best(a, {}, true).tokens == 0 && cache.best(b, {}, true).tokens == 3,
              "the oldest conversation went, the newest stayed");
        check(cache.bytes() == both - image({1, 2, 3}).bytes(), "evict_oldest releases the entry's bytes");
        cache.evict_oldest();
        check(cache.size() == 0 && cache.bytes() == 0, "evicting the last parked conversation empties the cache");
        check(!cache.evict_oldest(), "the gate's loop can reach an empty cache: false");
        check(cache.size() == 0 && cache.evictions() == 2, "evict_oldest on an empty cache is a no-op");
    }
    {   // the RAM gate's eviction skips a conversation that holds a pinned shared prefix, and says so (false)
        ConversationCache cache(1 << 20, 4);
        auto pinned = image({1, 2, 3});
        pinned.checkpoints.push_back(ConversationCheckpoint{});
        pinned.checkpoints.back().ids = {1, 2};
        pinned.checkpoints.back().pinned = true;
        pinned.checkpoints.back().gdn.resize(64, 1);
        cache.put(std::move(pinned));
        cache.put(image({9, 8, 7}));
        check(cache.evict_oldest() && cache.size() == 1, "the oldest unpinned conversation went (the pinned one is older)");
        check(!cache.evict_oldest() && cache.size() == 1, "only a pinned one is left: nothing more can go");
        // #879: after non-finite logits every parked conversation goes, the pinned one too
        cache.put(image({4, 5, 6}));
        std::vector<ConversationKv> kv(1);
        kv[0].cells = 3;
        kv[0].k.resize(64, 1);
        cache.retain(std::move(kv), 3);
        check(cache.retained_bytes() > 0, "a retained K/V before clear()");
        check(cache.clear() == 2, "clear() reports how many were parked");
        check(cache.size() == 0 && cache.bytes() == 0 && cache.retained_bytes() == 0, "clear() leaves nothing parked");
        check(cache.best(std::vector<int64_t>{1, 2, 3, 4}, {}, true).tokens == 0, "a cleared cache matches nothing");
        check(cache.clear() == 0, "clear() on an empty cache");
    }
    {
        auto s = image({1, 2, 3});
        ConversationCheckpoint cp;
        cp.ids = {1, 2};
        s.checkpoints.push_back(cp);
        ConversationCache cache(4096, 4);
        cache.put(std::move(s));
        auto match = cache.best(std::vector<int64_t>{1, 2, 9, 4}, {}, true);
        check(match.tokens == 2 && !match.live, "edited suffix falls back to parked checkpoint");
        check(cache.best(a, {}, true).tokens == 3, "live prefix beats shorter checkpoint");
        check(cache.best(a, {}, false).tokens == 0, "steering mode is isolated");
        check(cache.best(std::vector<int64_t>{1, 2}, {}, true).tokens == 0, "equal-length checkpoint cannot consume last token");
        check(cache.best(std::vector<int64_t>{1}, {}, true).tokens == 0, "short prompt cannot match");
        check(cache.best(std::vector<int64_t>{}, {}, true).tokens == 0, "empty prompt cannot match");
        cache.put(image({1, 2, 3}));
        check(cache.best(a, {}, true).index == 1, "ties prefer most recently parked");
    }
    {
        auto s = image({1, 2, 3});
        s.live.imgs = {{1, 123}};
        ConversationCache cache(1024, 3);
        cache.put(std::move(s));
        check(cache.best(a, {{1,123}}, true).tokens == 3, "same image can resume");
        check(cache.best(a, {{1,124}}, true).tokens == 0, "different image pixels invalidate same pad tokens");
        check(cache.best(a, {}, true).tokens == 0, "missing image invalidates prefix");
        check(cache.best(a, {{1,123},{2,45}}, true).tokens == 0, "additional image in cached prefix invalidates");
        check(cache.best(a, {{1,123},{3,45}}, true).tokens == 3, "image after cached prefix does not invalidate");
    }
    {
        const size_t one = image({1,2,3}).bytes();
        ConversationCache cache(one*2, 8);
        cache.put(image({1,2,3})); cache.put(image({9,8,7}));
        check(cache.bytes() == one*2, "budget holds two exact-sized images");
        auto held = cache.take(cache.best(a, {}, true).index);
        check(cache.make_room(one, held.bytes()), "count in-flight image when reserving outgoing snapshot");
        check(cache.size() == 0, "in-flight reservation evicts otherwise fitting B");
        check(cache.put(image({5,6,7}), held.bytes()), "insert with in-flight accounting");
        check(cache.bytes()+held.bytes() <= one*2, "exchange obeys byte budget");
        check(!cache.make_room(one+1, one), "oversized exchange rejected");
        check(cache.size() == 1, "oversized snapshot does not evict useful entries");
        auto huge = image({1}); huge.live.gdn.resize(one*3);
        check(!cache.put(std::move(huge)), "oversized image rejected");
        check(cache.size() == 1, "oversized put leaves cache unchanged");
        check(!cache.make_room(0, one*2+1), "held larger than budget cannot underflow");
    }
    {
        // #342: MAIN -> SUB (9 turns) -> MAIN with 4 slots.  Every SUB turn parks the previous turn's live state
        // (its reply as generated, which the next request re-rendered) with a chain that holds that turn's
        // boundary checkpoint; the copy a turn back adds only that stale tail and is dropped, so MAIN survives.
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        auto with = [](std::vector<int32_t> prefix, std::initializer_list<int32_t> more) {
            prefix.insert(prefix.end(), more); return prefix;
        };
        const std::vector<int32_t> root = {1, 2, 3, 4};
        SavedConversation main = image({});
        main.live.ids = with(with(root, {10, 11, 12}), {13, 14});
        main.checkpoints = {cp(root), cp(with(root, {10, 11, 12}))};
        const size_t big = 1 << 20;
        ConversationCache cache(big, 4);
        check(cache.put(std::move(main)), "park MAIN");
        std::vector<int32_t> history = with(root, {20});         // SUB's conversation so far, re-rendered
        std::vector<ConversationCheckpoint> chain = {cp(root)};
        for (int turn = 1; turn <= 9; ++turn) {
            chain.push_back(cp(history));                        // the turn boundary the next request resumes from
            SavedConversation sub = image({});
            sub.live.ids = with(history, {900, (int32_t) turn});   // + the reply with its thinking (stale tail)
            sub.checkpoints = chain;
            check(cache.put(std::move(sub)), "park SUB turn");
            check(cache.size() <= 2, "one parked copy of SUB at a time");
            history = with(history, {30, (int32_t) turn});       // the reply as the next request renders it
        }
        check(cache.evictions() == 0 && cache.superseded() == 8, "8 superseded copies dropped, nothing evicted");
        const auto m = cache.best(with(with(root, {10, 11, 12}), {13, 14, 15}), {}, true);
        check(m.tokens == 9 && m.live, "MAIN still restores in full");
        const auto s = cache.best(with(history, {40}), {}, true);
        check(s.tokens == (int64_t) history.size() - 2, "SUB resumes from its last turn boundary");
    }
    {
        // holds(): what an idle batch slot holds is parked already (or a continuation of it is)
        auto s = image({1, 2, 3, 4, 5});
        ConversationCheckpoint cp;
        cp.ids = {1, 2, 3};
        s.checkpoints.push_back(cp);
        ConversationCache cache(4096, 4);
        cache.put(std::move(s));
        check(cache.holds(std::vector<int32_t>{1, 2, 3, 4, 5}, {}, true), "the parked live state itself");
        check(cache.holds(std::vector<int32_t>{1, 2, 3, 4}, {}, true), "a prefix of the parked live state");
        check(cache.holds(std::vector<int32_t>{1, 2}, {}, true), "a prefix of a parked checkpoint");
        check(!cache.holds(std::vector<int32_t>{1, 2, 3, 4, 5, 6}, {}, true), "a longer state is not held");
        check(!cache.holds(std::vector<int32_t>{1, 2, 9}, {}, true), "a diverging state is not held");
        check(!cache.holds(std::vector<int32_t>{}, {}, true), "an empty state is never held");
        check(!cache.holds(std::vector<int32_t>{1, 2, 3}, {}, false), "steering mode is isolated");
        check(!cache.holds(std::vector<int32_t>{1, 2, 3}, {{1, 77}}, true), "a slot with pictures is not held by a text image");
        auto p = image({1, 2, 3, 4});
        p.live.imgs = {{1, 77}, {3, 78}};
        cache.put(std::move(p));
        check(cache.holds(std::vector<int32_t>{1, 2, 3}, {{1, 77}}, true), "pictures below the state must match");
        check(!cache.holds(std::vector<int32_t>{1, 2, 3, 4}, {{1, 77}}, true), "a picture inside the state that the slot lacks");
    }
    {
        // The 2026-10-14 four-conversation measurement replayed at policy level: two batch slots, four parked
        // entries, conversations A..D.  Each has a prompt P (8 tokens; the read's periodic checkpoint at 4), a first
        // reply R1 (3 tokens; the slot holds P + 2 of them), the next turn as the client renders it (P + R1 + end +
        // header: 14 tokens, the read's turn checkpoint at 12), a second reply R2 (4 tokens; the slot holds 17) and
        // a third turn (21 tokens).  The engine parks the MAIN session (the prompt as read, plus the admission's
        // token) whenever another conversation displaces it; the replies are decoded in the slots.  Measured:
        // A's third turn found 199584 of 199839 held tokens (its prompt-read state), B found nothing.
        //   slot_parks:   an idle slot parks what it holds before an admission overwrites it
        //   checks_back:  a slot that gives "all it holds" back to the main session gives its checkpoints with it,
        //                 so the chain still supersedes (#342) the parked copies a turn back
        auto replay = [&](bool slot_parks, bool checks_back) {
            std::vector<std::vector<int32_t>> full(4);
            for (int c = 0; c < 4; ++c) {
                for (int32_t t = 1; t <= 8; ++t) full[(size_t) c].push_back(c * 100 + t);
                for (int32_t t : {21, 22, 23, 9, 31, 32, 41, 42, 43, 44, 9, 51, 52}) full[(size_t) c].push_back(c * 100 + t);
            }
            auto st = [&](int c, size_t n) { return std::vector<int32_t>(full[(size_t) c].begin(), full[(size_t) c].begin() + (std::ptrdiff_t) n); };
            auto cp = [&](int c, size_t n) { ConversationCheckpoint k; k.ids = st(c, n); return k; };
            auto park = [&](ConversationCache& cache, int c, size_t n, const std::vector<size_t>& cks) {
                SavedConversation s = image({});
                s.live.ids = st(c, n);
                for (size_t k : cks) s.checkpoints.push_back(cp(c, k));
                check(cache.put(std::move(s)), "the replay's parks all fit the budget");
            };
            auto restore = [&](ConversationCache& cache, int c, size_t prompt) -> int64_t {
                const auto m = cache.best(st(c, prompt), {}, true);
                if (m.tokens > 0) cache.take(m.index);
                return m.tokens;
            };
            // the main chain after a slot gave "all it holds" back: its checkpoints, or none
            auto back = [&](const std::vector<size_t>& slot_cks) { return checks_back ? slot_cks : std::vector<size_t>{}; };
            auto slot_park = [&](ConversationCache& cache, int c, size_t n, const std::vector<size_t>& cks) {
                if (slot_parks && !cache.holds(st(c, n), {}, true)) park(cache, c, n, cks);
            };
            enum { A, B, C, D };
            ConversationCache cache(1 << 20, 4);
            // initial-AB: B solo then admitted (its prompt + the first token parked when A's request came), A
            // admitted directly, then alone: solo again from its slot (A's prompt-read state parked first)
            park(cache, B, 9, {4});
            park(cache, A, 8, {4});
            std::vector<size_t> chain_a = back({4});           // slot 1 gave back A's 10 tokens
            // resident1-AB: B's request displaces A's solo state; B's slot gives back 10; B admitted with the turn
            // checkpoint; A's request restores A's solo state from RAM and is admitted beside B
            park(cache, A, 12, chain_a);
            std::vector<size_t> chain_b = back({4});
            chain_b.push_back(12);
            // (a request takes its parked conversation before it parks the one it displaces)
            check(restore(cache, A, 14) == 12, "A's second turn resumes from its parked solo state");
            park(cache, B, 15, chain_b);
            chain_a.push_back(12);
            // both decode: slot 0 holds B's 17, slot 1 holds A's 17
            // initial-CD: C's request displaces A's main state; D admitted into slot 0 (B's reply), C into slot 1
            park(cache, A, 14, chain_a);
            slot_park(cache, B, 17, {12});
            park(cache, D, 9, {4});
            slot_park(cache, A, 17, {12});
            park(cache, C, 8, {4});                              // C alone: solo again from its slot
            std::vector<size_t> chain_c = back({4});
            // resident1-CD: C continues from the main session (no park), D's request displaces it (slot 1 held C's
            // own earlier state: covered, no slot park), D's slot gives back 10 (covered as well)
            chain_c.push_back(12);
            park(cache, C, 15, chain_c);
            std::vector<size_t> chain_d = back({4});
            chain_d.push_back(12);
            // both decode: slot 0 holds D's 17, slot 1 holds C's 17
            // exchange1-AB: A's request takes A's parked state, displaces D's main state, is admitted into slot 1
            // (C's reply); B's request displaces A's state
            const int64_t a_found = restore(cache, A, 21);
            park(cache, D, 14, chain_d);
            slot_park(cache, C, 17, {12});
            park(cache, A, 22, {12, 20});
            const int64_t b_found = restore(cache, B, 21);
            return std::pair<int64_t, int64_t>(a_found, b_found);
        };
        const auto baseline = replay(false, false);
        check(baseline.first == 14 && baseline.second == 0, "baseline: A returns to its prompt-read state, B to nothing (as measured)");
        const auto parks_only = replay(true, false);
        check(parks_only.first == 17 && parks_only.second == 0, "slot parks alone: A whole, B's copy evicted by a stale initial image");
        const auto both = replay(true, true);
        check(both.first == 17 && both.second == 17, "slot parks + checkpoints back: both return whole");
    }
    {
        // what is NOT superseded: another conversation sharing only the root, an entry without checkpoints, the
        // other steering mode, a branch whose deepest checkpoint the new chain does not hold
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        ConversationCache cache(1 << 20, 8);
        SavedConversation other = image({1, 2, 3, 4, 50, 51, 52});
        other.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 50, 51})};
        cache.put(std::move(other));
        cache.put(image({1, 2, 3, 4, 7, 7}));                    // no checkpoints
        SavedConversation steered = image({1, 2, 3, 4, 60, 61}, false);
        steered.checkpoints = {cp({1, 2, 3, 4, 60})};
        cache.put(std::move(steered));
        SavedConversation branch = image({1, 2, 3, 4, 60, 70, 71});
        branch.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 60, 70})};
        cache.put(std::move(branch));
        SavedConversation incoming = image({1, 2, 3, 4, 60, 80, 81});
        incoming.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 60})};
        check(cache.put(std::move(incoming)), "park a conversation sharing roots with all of them");
        check(cache.size() == 5 && cache.superseded() == 0, "none of them is superseded");
        SavedConversation same_state = image({1, 2, 3, 4, 60, 70});   // live equal to the branch's deepest point
        same_state.checkpoints = {cp({1, 2, 3, 4})};
        check(cache.put(std::move(same_state)) && cache.superseded() == 1 && cache.size() == 5,
              "the live state equal to an entry's deepest checkpoint supersedes it");
        SavedConversation huge = image({1, 2, 3, 4, 60, 80});
        huge.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 60})};
        huge.live.gdn.resize(2 << 20);
        check(!cache.put(std::move(huge)) && cache.size() == 5 && cache.superseded() == 1,
              "an oversized put drops nothing");
    }
    {
        // an older, shorter state of this conversation, rewritten from token 36 on; the outgoing chain holds only the
        // root (spread checkpoints thinned out).  At most a tenth of its tokens not shared: superseded; more: kept.
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        auto first = [](int32_t n) { std::vector<int32_t> v(n); std::iota(v.begin(), v.end(), 1); return v; };
        std::vector<int32_t> rewritten = first(36);   // and grown past the old state since
        rewritten.insert(rewritten.end(), {900, 901, 902, 903, 904, 905, 906, 907, 908, 909});
        const std::vector<ConversationCheckpoint> chain = {cp(first(4))};
        size_t reported = 0;
        auto superseded_after_rewrite = [&](int32_t tokens) {
            ConversationCache cache(1 << 20, 8);
            SavedConversation old = image({});
            old.live.ids = first(tokens);
            old.checkpoints = {cp(first(4)), cp(first(36)), cp(first(tokens - 1))};
            cache.put(std::move(old));
            reported = cache.parked_prefix_tokens(rewritten, {}, chain, true);
            SavedConversation now = image({});
            now.live.ids = rewritten;
            now.checkpoints = chain;
            cache.put(std::move(now));
            return cache.superseded();
        };
        check(superseded_after_rewrite(40) == 1 && reported == 36, "4 of 40 tokens not shared: dropped");
        check(superseded_after_rewrite(41) == 0 && reported == 0, "5 of 41 tokens not shared: kept");
    }
    {
        // --conversation-cache-park-gain N: parked_prefix_tokens reports what a park now would supersede, so a
        // conversation that advanced fewer than N tokens since that copy keeps it instead of writing the whole
        // snapshot again for the next turn to drop.  It must name exactly the entries drop_superseded() would remove.
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        ConversationCache cache(1 << 20, 8);
        check(cache.parked_prefix_tokens({1, 2, 3}, {}, {}, true) == 0, "an empty cache supersedes nothing");

        const std::vector<int32_t> live_ids = {1, 2, 3, 4, 60, 70};
        const std::vector<ConversationCheckpoint> chain = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 60})};
        SavedConversation same = image({1, 2, 3, 4, 60, 70});
        same.checkpoints = chain;
        check(cache.put(std::move(same)), "park a conversation with a checkpoint chain");
        check(cache.parked_prefix_tokens(live_ids, {}, chain, true) == 5,
              "the entry the outgoing chain supersedes is reported by its deepest checkpoint");

        const std::vector<ConversationCheckpoint> other_chain = {cp({1, 2, 3})};
        check(cache.parked_prefix_tokens({1, 2, 3, 40, 50}, {}, other_chain, true) == 0,
              "a conversation sharing only the root supersedes nothing");
        check(cache.parked_prefix_tokens(live_ids, {}, chain, false) == 0,
              "the other steering mode is a different entry");
        check(cache.parked_prefix_tokens({1, 2, 3, 4, 60, 70, 71}, {}, chain, true) == 5,
              "an appended tail keeps the reported depth at the parked one");
        SavedConversation no_cp = image({1, 2, 3, 9});
        check(cache.put(std::move(no_cp)) && cache.parked_prefix_tokens({1, 2, 3, 9}, {}, {}, true) == 0,
              "an entry without checkpoints is never reported");
    }
    {
        // borrowing: a NEW conversation that starts with a checkpoint inside a parked one (a sibling subagent with
        // the same system prompt and tools) restores from it and leaves it parked; the same conversation going on
        // (its live state, or its deepest checkpoint after the client re-rendered the last reply) is still taken
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        SavedConversation a = image({});
        a.live.ids = {1, 2, 3, 4, 10, 11, 12, 13};
        a.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 10, 11})};
        ConversationCache cache(1 << 20, 2);
        check(cache.put(std::move(a)), "park subagent A");
        const auto sib = cache.best(std::vector<int64_t>{1, 2, 3, 4, 20, 21}, {}, true);
        check(sib.tokens == 4 && !sib.live && sib.serial != 0 && cache.borrows(sib), "a sibling on A's root borrows");
        const auto again = cache.best(std::vector<int64_t>{1, 2, 3, 4, 10, 11, 30}, {}, true);
        check(again.tokens == 6 && !again.live && !cache.borrows(again),
              "A re-rendered at its deepest checkpoint is taken");
        const auto cont = cache.best(std::vector<int64_t>{1, 2, 3, 4, 10, 11, 12, 13, 14}, {}, true);
        check(cont.live && !cache.borrows(cont), "A going on is taken");
        check(!cache.borrows(cache.best(std::vector<int64_t>{5, 6, 7}, {}, true)), "no match, nothing to borrow");
        // while the outgoing conversations are parked, the guarded donor is neither evicted nor dropped
        cache.guard(sib.serial);
        check(cache.put(image({7, 7, 7})) && cache.put(image({8, 8, 8})), "two outgoing conversations park");
        check(cache.find(sib.serial) != nullptr && cache.size() == 2 && cache.evictions() == 1 && !cache.guard_blocked(),
              "the slot limit evicts the other entry, not the guarded donor");
        check(cache.best(std::vector<int64_t>{7, 7, 7, 1}, {}, true).tokens == 0, "the other one went");
        SavedConversation rewritten = image({1, 2, 3, 4, 10, 11, 40});   // holds A's deepest checkpoint: supersedes A
        rewritten.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 10, 11})};
        check(cache.put(std::move(rewritten)) && cache.find(sib.serial) != nullptr && cache.superseded() == 0,
              "a guarded donor is not dropped as superseded");
        cache.unguard();
        // identity survives index shifts: take an older entry, the donor's serial still finds it
        const auto donor_index = cache.index_of(sib.serial);
        check(donor_index != SIZE_MAX, "donor indexed by its serial");
        check(cache.best(std::vector<int64_t>{1, 2, 3, 4, 10, 11, 12, 13, 14}, {}, true).serial == sib.serial,
              "matches report the entry's serial");
    }
    {
        // the outgoing conversation fits only in the guarded donor's room: make_room refuses before evicting anything
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        SavedConversation a = image({1, 2, 3, 4, 10, 11});
        a.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 10})};
        const size_t a_bytes = a.bytes();
        const size_t out_bytes = image({9, 9, 9}).bytes();
        ConversationCache tight(a_bytes + out_bytes - 1, 4);
        check(tight.put(std::move(a)), "park the donor");
        const auto m = tight.best(std::vector<int64_t>{1, 2, 3, 4, 50}, {}, true);
        check(tight.borrows(m), "a sibling would borrow");
        tight.guard(m.serial);
        check(!tight.put(image({9, 9, 9})) && tight.guard_blocked(), "no room beside the guarded donor: refused");
        check(tight.size() == 1 && tight.evictions() == 0, "nothing evicted by the refusal");
        tight.unguard();
        auto taken = tight.take(tight.index_of(m.serial));   // what the caller does then: take it, as before
        check(taken.live.ids.size() == 6 && taken.checkpoints.size() == 2 && tight.size() == 0,
              "the taken donor is whole");
        ConversationCache one_slot(1 << 20, 1);
        SavedConversation b = image({1, 2, 3, 4, 10, 11});
        b.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 10})};
        check(one_slot.put(std::move(b)), "park in a single slot");
        const auto mb = one_slot.best(std::vector<int64_t>{1, 2, 3, 4, 50}, {}, true);
        one_slot.guard(mb.serial);
        check(!one_slot.put(image({9, 9, 9})) && one_slot.guard_blocked() && one_slot.size() == 1,
              "one slot: the guarded donor blocks, nothing evicted");
        one_slot.unguard();
    }
    {
        // pin=N beside a borrow: the guarded donor and a conversation holding a pinned shared prefix both stay; the
        // oldest other entry is the one evicted (the case midhatn tested on #1164)
        auto cp = [](std::vector<int32_t> ids, bool pinned = false) {
            ConversationCheckpoint c; c.ids = std::move(ids); c.pinned = pinned; return c;
        };
        SavedConversation doc = image({1, 2, 3, 4, 5, 6, 30, 31});
        doc.checkpoints = {cp({1, 2, 3, 4, 5, 6}, true), cp({1, 2, 3, 4, 5, 6, 30})};
        SavedConversation donor = image({1, 2, 10, 11, 12});
        donor.checkpoints = {cp({1, 2}), cp({1, 2, 10, 11})};
        ConversationCache cache(1 << 20, 3);
        check(cache.put(std::move(doc)) && cache.put(std::move(donor)) && cache.put(image({7, 7, 7})),
              "pinned document, donor and an ordinary conversation parked");
        const auto m = cache.best(std::vector<int64_t>{1, 2, 40, 41}, {}, true);
        check(cache.borrows(m), "a sibling on the donor's root borrows");
        cache.guard(m.serial);
        check(cache.put(image({8, 8, 8})) && !cache.guard_blocked() && cache.size() == 3 && cache.evictions() == 1,
              "the outgoing conversation parks");
        check(cache.find(m.serial) != nullptr &&
              cache.best(std::vector<int64_t>{1, 2, 3, 4, 5, 6, 9}, {}, true).tokens == 6 &&
              cache.best(std::vector<int64_t>{7, 7, 7, 1}, {}, true).tokens == 0,
              "only the ordinary entry was evicted: the donor and the pinned document stay");
        check(cache.put(image({9, 9, 9})) && cache.size() == 3 && cache.evictions() == 2 &&
              cache.find(m.serial) != nullptr && cache.best(std::vector<int64_t>{8, 8, 8, 1}, {}, true).tokens == 0,
              "the next outgoing one evicts the previous outgoing one, still not the donor or the document");
        cache.unguard();
        ConversationCache two(1 << 20, 2);
        SavedConversation doc2 = image({1, 2, 3, 4, 5, 6, 30});
        doc2.checkpoints = {cp({1, 2, 3, 4, 5, 6}, true)};
        SavedConversation donor2 = image({1, 2, 10, 11, 12});
        donor2.checkpoints = {cp({1, 2}), cp({1, 2, 10, 11})};
        check(two.put(std::move(doc2)) && two.put(std::move(donor2)), "pinned document and donor fill two slots");
        const auto m2 = two.best(std::vector<int64_t>{1, 2, 40}, {}, true);
        two.guard(m2.serial);
        check(!two.put(image({9, 9, 9})) && two.guard_blocked() && two.size() == 2 && two.evictions() == 0,
              "no slot beside the donor and the pinned document: refused before evicting anything");
        two.unguard();
        // a match that reaches the pinned prefix is taken, not borrowed: one conversation keeps holding the pin
        const auto on_pin = two.best(std::vector<int64_t>{1, 2, 3, 4, 5, 6, 77}, {}, true);
        check(on_pin.tokens == 6 && !on_pin.live && !two.borrows(on_pin), "a pinned prefix is taken, not borrowed");
        ConversationCache three(1 << 20, 3);
        SavedConversation doc3 = image({1, 2, 3, 4, 5, 6, 30, 31, 32});
        doc3.checkpoints = {cp({1, 2, 3, 4, 5, 6}, true), cp({1, 2, 3, 4, 5, 6, 30}), cp({1, 2, 3, 4, 5, 6, 30, 31})};
        check(three.put(std::move(doc3)), "a conversation past its pinned prefix is parked");
        const auto past_pin = three.best(std::vector<int64_t>{1, 2, 3, 4, 5, 6, 30, 99}, {}, true);
        check(past_pin.tokens == 7 && !three.borrows(past_pin),
              "a match past the pinned prefix (the copy would carry the pin) is taken, not borrowed");
    }
    {
        ConversationCache disabled(0,4), no_slots(1024,0);
        check(!disabled.enabled() && !no_slots.enabled(), "both disable switches");
        check(!disabled.put(image({1,2,3})) && !no_slots.put(image({1,2,3})), "disabled cache stores nothing");
        check(disabled.best(a,{},true).tokens == 0, "disabled cache has no matches");
    }
    {
        // --conversation-cache-min-tokens: a conversation shorter than the minimum is not worth one of the slots.
        // Parking is a switch-time decision, so without this a client's short side requests park themselves and,
        // once the cache is full, evict the long conversation they were interleaved with (oldest first).
        ConversationCache any(1024, 4, 0), floor(1024, 4, 12288);
        check(any.enabled() && floor.enabled(), "the minimum does not disable the cache");
        check(any.wants(0) && any.wants(1) && any.wants(1 << 20), "0 parks every length, as before the flag");
        check(!floor.wants(0) && !floor.wants(1) && !floor.wants(12287), "below the minimum is not parked");
        check(floor.wants(12288) && floor.wants(12289), "the minimum itself and above are parked");
        // the CLI rejects a negative, but the predicate must not invert for one that reaches it another way
        ConversationCache negative(1024, 4, -1);
        check(negative.wants(0) && negative.wants(1), "a negative minimum parks everything, like 0");
        // wants() answers for the length alone; the gate is enabled() && wants(), so a disabled cache parks
        // nothing whatever the length - keep the two independent rather than folding one into the other
        ConversationCache disabled_min(0, 4, 12288);
        check(!disabled_min.enabled() && disabled_min.wants(1 << 20), "a disabled cache still answers for length");
    }
    {
        // --conversation-cache-min-tokens WRITE[,READ]: a single value keeps the old meaning (WRITE,0)
        ConversationCacheTokenLimits limits;
        check(parse_conversation_cache_min_tokens("6000", limits) && limits.write_min_tokens == 6000 &&
              limits.read_min_tokens == 0, "a single value is WRITE,0");
        check(parse_conversation_cache_min_tokens("6000,0", limits) && limits.write_min_tokens == 6000 &&
              limits.read_min_tokens == 0, "WRITE,0 is the single value spelled out");
        check(parse_conversation_cache_min_tokens("6000,5000", limits) && limits.write_min_tokens == 6000 &&
              limits.read_min_tokens == 5000, "WRITE,READ are the two limits");
        check(parse_conversation_cache_min_tokens("2147483647,2147483647", limits) &&
              limits.write_min_tokens == INT32_MAX && limits.read_min_tokens == INT32_MAX,
              "both limits accept the maximum");
        for (const char* bad : {"", "-1,2", "1,-2", ",2", "1,", "1,2,3", "6000x", "6000, 5000", "2147483648"})
            check(!parse_conversation_cache_min_tokens(bad, limits), "a malformed WRITE,READ is refused");
        // READ=0 has no limit; READ>0 measures the COMPLETE prompt, cached history included
        check(!conversation_cache_should_read_parked(4999, 5000), "below READ the parked conversations are not searched");
        check(conversation_cache_should_read_parked(5000, 5000), "READ itself is searched");
        check(conversation_cache_should_read_parked(4999, 0) && conversation_cache_should_read_parked(0, 0),
              "READ=0 searches every prompt, as before the flag");
        std::vector<int32_t> history(25967, 3);
        std::vector<int32_t> prompt = history;
        prompt.insert(prompt.end(), 33, 4);
        check(!conversation_cache_should_read_parked(33, 26000), "the 33 new tokens alone are below READ");
        check(conversation_cache_should_read_parked((int64_t) prompt.size(), 26000),
              "25967 cached tokens plus 33 new ones reach READ=26000");
        // the gate skips the parked search only: the active session's prefix reuse is conversation_prefix()
        SavedConversation active;
        active.live.ids = history;
        active.live.gdn.resize(64, 7);
        check(!conversation_cache_should_read_parked((int64_t) prompt.size(), 30000),
              "the full prompt stays below a higher READ");
        check(conversation_prefix(active.live, prompt, {}) == (int64_t) history.size(),
              "below READ the active session still reuses its whole prefix");
    }
    {
        // layer-split parking: checkpoints moved apart into stage parts and put back, no running state copied
        auto running = [](std::initializer_list<int32_t> ids, uint8_t salt, size_t parts) {
            ConversationCheckpoint c;
            c.ids = ids;
            c.gdn.assign(256, salt);
            c.tails.assign(32, uint8_t(salt + 1));
            c.used = salt;
            for (size_t k = 0; k < parts; ++k) {
                ConversationCheckpoint p;
                p.gdn.assign(128, uint8_t(salt + 10 + k));
                c.stage_parts.push_back(std::move(p));
            }
            return c;
        };
        std::vector<ConversationCheckpoint> checks;
        checks.push_back(running({1, 2}, 3, 2));
        checks.push_back(running({1, 2, 3}, 5, 1));       // an incomplete set: kept aside, put back in place
        checks.push_back(running({1, 2, 3, 4}, 7, 2));
        const auto original = checks;
        const uint8_t* gdn0 = checks[0].gdn.data();
        const uint8_t* part1 = checks[2].stage_parts[1].gdn.data();
        auto split = conversation_checkpoints_split(std::move(checks), 2);
        check(checks.empty() && split.stage0.size() == 2 && split.rest.size() == 1 && split.parts.size() == 2 &&
              split.parts[0].size() == 2 && split.parts[1].size() == 2, "split: complete checkpoints apart, the other aside");
        check(split.stage0[0].stage_parts.empty() && split.parts[1][1].ids == original[2].ids &&
              split.parts[1][1].used == original[2].used && split.parts[0][0].gdn == original[0].stage_parts[0].gdn,
              "split: every stage part carries its checkpoint's ids and LRU stamp");
        check(split.stage0[0].gdn.data() == gdn0 && split.parts[1][1].gdn.data() == part1,
              "split moves the running state (same buffers, no copy)");
        std::vector<ConversationCheckpoint> back;
        check(conversation_checkpoints_merge(std::move(split), back) && back.size() == 3, "merge puts them back");
        bool same = true;
        for (size_t j = 0; j < back.size(); ++j) {
            same = same && back[j].ids == original[j].ids && back[j].gdn == original[j].gdn &&
                   back[j].tails == original[j].tails && back[j].used == original[j].used &&
                   back[j].stage_parts.size() == original[j].stage_parts.size();
            for (size_t k = 0; same && k < back[j].stage_parts.size(); ++k)
                same = back[j].stage_parts[k].gdn == original[j].stage_parts[k].gdn;
        }
        check(same, "merge(split(x)) == x, in the original order");
        check(back[0].gdn.data() == gdn0 && back[2].stage_parts[1].gdn.data() == part1, "merge moves them back too");

        auto torn = conversation_checkpoints_split(std::vector<ConversationCheckpoint>(original), 2);
        torn.parts[1].pop_back();
        check(!conversation_checkpoints_merge(std::move(torn), back) && back.empty(), "a stage short of a part is refused");
        auto mixed = conversation_checkpoints_split(std::vector<ConversationCheckpoint>(original), 2);
        mixed.parts[0][0].ids = {9, 9};
        check(!conversation_checkpoints_merge(std::move(mixed), back) && back.empty(), "a part of another checkpoint is refused");
        // a restored image: stage 0's checkpoints and each stage's, with no record of an incomplete one
        ConversationCheckpointSplit restored;
        restored.stage0.push_back(running({4, 5}, 11, 0));
        restored.parts.resize(2);
        for (auto& p : restored.parts) { auto c = running({4, 5}, 12, 0); p.push_back(std::move(c)); }
        check(conversation_checkpoints_merge(std::move(restored), back) && back.size() == 1 &&
              back[0].stage_parts.size() == 2, "a restored image's parts are all whole");
        auto none = conversation_checkpoints_split({}, 3);
        check(conversation_checkpoints_merge(std::move(none), back) && back.empty(), "no checkpoints: nothing to do");
    }
    {
        // 2026-10-08 cross-conversation leak: the retained K/V of a restored conversation must never be reused to
        // park ANOTHER conversation the main session later holds (a slot gave its conversation back)
        auto kv = [](size_t n) {
            std::vector<ConversationKv> v(2);
            for (auto& l : v) l.k.resize(n, 1);
            return v;
        };
        ConversationCache cache(1 << 20, 4);
        const std::vector<int32_t> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        std::vector<std::vector<ConversationKv>> st;
        st.push_back(kv(100));
        cache.retain(kv(500), 16, std::move(st), a);
        std::vector<int32_t> b(16, 99);
        auto other = cache.take_reuse_for(b);
        check(other.kv.empty() && other.stages.empty() && other.unchanged_tokens == 0,
              "another conversation never gets the retained K/V");
        check(cache.retained_bytes() == 0, "the retained K/V is dropped once refused");
        cache.retain(kv(500), 16, {}, a);
        check(cache.retained_tokens() == 16, "the retained extent is reported");
        std::vector<int32_t> cont(a.begin(), a.begin() + 10);
        cont.push_back(42);
        cont.push_back(43);
        auto partly = cache.take_reuse_for(cont);
        check(partly.unchanged_tokens == 10, "a conversation that diverges reuses only the shared prefix");
        std::vector<std::vector<ConversationKv>> st2;
        st2.push_back(kv(100));
        cache.retain(kv(500), 16, std::move(st2), a);
        cache.limit_reuse(9);
        auto same = cache.take_reuse_for(a);
        check(same.unchanged_tokens == 9 && same.stages.size() == 1 && same.stages[0].unchanged_tokens == 9,
              "the same conversation keeps the first rewrite's limit, on every stage");
        cache.retain(kv(500), 16);
        check(cache.take_reuse_for(a).kv.empty(), "retained K/V of unknown identity is never reused");
        // --kv-pool-tokens: the chain moves with the K/V - every checkpoint that is a prefix of the conversation
        // goes, in order; one of another conversation (or one past it) stays behind for the caller to drop
        auto ck = [](std::vector<int32_t> ids, uint64_t used) {
            ConversationCheckpoint c;
            c.ids = std::move(ids);
            c.used = used;
            c.gdn.assign(8, 7);
            return c;
        };
        std::vector<ConversationCheckpoint> from;
        from.push_back(ck({1, 2}, 1));
        from.push_back(ck({1, 2, 3, 4}, 2));
        from.push_back(ck({1, 2, 9, 9}, 3));         // another branch
        from.push_back(ck({1, 2, 3, 4, 5, 6, 7}, 4));  // past the conversation
        from.push_back(ck({1, 2, 3, 4, 5}, 5));        // the whole conversation: a prefix
        std::vector<ConversationCheckpoint> to;
        to.push_back(ck({1}, 0));                      // what the receiver already holds stays first
        check(conversation_checkpoints_carry(from, {1, 2, 3, 4, 5}, to) == 3, "carry: three checkpoints are prefixes");
        check(to.size() == 4 && to[0].used == 0 && to[1].used == 1 && to[2].used == 2 && to[3].used == 5,
              "carry: they arrive in order after what was there");
        check(from.size() == 2 && from[0].used == 3 && from[1].used == 4, "carry: the others stay");
        check(to[3].gdn.size() == 8 && to[3].gdn[0] == 7, "carry: the running state moved with it");
        ConversationCheckpoint pic = ck({1, 2}, 6);
        pic.imgs.push_back({1, 77});
        std::vector<ConversationCheckpoint> with_picture, out;
        with_picture.push_back(std::move(pic));
        check(conversation_checkpoints_carry(with_picture, {1, 2, 3}, out) == 0 && out.empty(),
              "carry: a checkpoint with a picture is not carried");
        std::vector<ConversationCheckpoint> none, sink;
        check(conversation_checkpoints_carry(none, {1, 2}, sink) == 0 && sink.empty(), "carry: nothing to move");
    }
    {
        // the later stages' retained K/V, kept with the first stage's and limited together
        auto kv = [](size_t n) {
            std::vector<ConversationKv> v(2);
            for (auto& l : v) l.k.resize(n, 1);
            return v;
        };
        ConversationCache cache(1 << 20, 4);
        std::vector<std::vector<ConversationKv>> stage_kv;
        stage_kv.push_back(kv(1000));
        stage_kv.push_back(kv(2000));
        cache.retain(kv(500), 40, std::move(stage_kv));
        check(cache.retained_bytes() >= 2 * (500 + 1000 + 2000), "retained bytes count every stage");
        cache.limit_reuse(30);
        auto r = cache.take_reuse();
        check(r.kv.size() == 2 && r.stages.size() == 2 && r.unchanged_tokens == 30 &&
              r.stages[0].unchanged_tokens == 30 && r.stages[1].unchanged_tokens == 30 &&
              r.stages[1].captured_tokens == 40, "every stage's reuse is limited to the first rewrite");
        check(cache.retained_bytes() == 0, "taken once");
        std::vector<std::vector<ConversationKv>> big;
        big.push_back(kv(1 << 20));
        cache.retain(kv(500), 40, std::move(big));
        check(cache.retained_bytes() == 0, "a stage's K/V over the budget drops the whole reuse");
        std::vector<std::vector<ConversationKv>> one;
        one.push_back(kv(100));
        cache.retain(kv(500), 40, std::move(one));
        cache.limit_reuse(0);
        check(cache.retained_bytes() == 0, "a rewrite from the start drops every stage's reuse");
        cache.retain(kv(500), 40);
        r = cache.take_reuse();
        check(r.kv.size() == 2 && r.stages.empty(), "no layer split: no stage reuse, as before");
    }
    {   // pin=N: a parked conversation that holds a pinned shared prefix is not the one evicted
        const size_t one = image({1,2,3}).bytes();
        auto pinned = image({1,2,3});
        pinned.checkpoints.push_back(ConversationCheckpoint{});
        pinned.checkpoints.back().ids = {1,2};
        pinned.checkpoints.back().pinned = true;
        pinned.checkpoints.back().gdn.resize(64, 1);
        const size_t pinned_bytes = pinned.bytes();
        ConversationCache big(pinned_bytes + one * 2 + 64, 3);
        check(pinned.pinned() && !image({1,2,3}).pinned(), "a conversation is pinned when one of its checkpoints is");
        check(big.put(std::move(pinned)), "the pinned conversation is parked first (the oldest)");
        check(big.put(image({7,8,9})) && big.put(image({4,5,6})) && big.size() == 3, "two more fill the slots");
        check(big.put(image({10,11,12})) && big.size() == 3 && big.evictions() == 1,
              "a fourth evicts the oldest unpinned one");
        check(big.best(std::vector<int32_t>{1,2,9,9}, {}, true).tokens == 2, "the pinned prefix is still found");
        ConversationCache tiny(pinned_bytes + 8, 2);
        auto p2 = image({1,2,3});
        p2.checkpoints.push_back(ConversationCheckpoint{});
        p2.checkpoints.back().ids = {1,2};
        p2.checkpoints.back().pinned = true;
        p2.checkpoints.back().gdn.resize(64, 1);
        check(tiny.put(std::move(p2)), "a pinned one fits an empty cache");
        check(!tiny.put(image({7,8,9})) && tiny.size() == 1, "nothing else fits beside it: parking is refused, the pin stays");
    }
    {   // #2255: a client that pins one prefix on every request fills every slot with its holders; parking goes on
        const size_t one = holder({1,2,3,4}, {1,2}).bytes();
        ConversationCache cache(one * 8, 3);
        check(cache.put(holder({1,2,3,4}, {1,2})) && cache.put(holder({1,2,5,6}, {1,2})) &&
              cache.put(holder({1,2,7,8}, {1,2})) && cache.size() == 3, "three holders of one pin fill the slots");
        check(cache.put(holder({1,2,9,10}, {1,2})) && cache.size() == 3 && cache.evictions() == 1,
              "a fourth holder is parked: the oldest holder goes (#2255)");
        check(cache.best(std::vector<int32_t>{1,2,3,4,0}, {}, true).tokens == 2,
              "the oldest holder's own tail is gone");
        check(cache.best(std::vector<int32_t>{1,2,9,10,0}, {}, true).tokens == 4, "the newest holder is kept");
        check(cache.put(image({20,21,22})) && cache.best(std::vector<int32_t>{1,2,0}, {}, true).tokens == 2,
              "an unpinned conversation is parked too, and the pinned prefix is still found");
    }
    {   // #2255: two pins side by side - the oldest entry is the only holder of pin A, so it stays and a B holder goes
        const size_t one = holder({1,2,3,4}, {1,2}).bytes();
        ConversationCache cache(one * 8, 3);
        cache.put(holder({1,2,3,4}, {1,2}));                              // A: oldest and its only holder
        cache.put(holder({5,6,7,8}, {5,6}));
        cache.put(holder({5,6,9,10}, {5,6}));
        check(cache.put(holder({5,6,11,12}, {5,6})) && cache.size() == 3, "a third B holder is parked");
        check(cache.best(std::vector<int32_t>{1,2,0}, {}, true).tokens == 2, "pin A (held once) survives");
        check(cache.best(std::vector<int32_t>{5,6,7,8,0}, {}, true).tokens == 2, "the oldest B holder went instead");
        check(cache.best(std::vector<int32_t>{5,6,0}, {}, true).tokens == 2, "pin B is still found");
    }
    {   // #2255: each entry the last holder of its own pin - nothing can go, parking is refused, every pin stays
        const size_t one = holder({1,2,3,4}, {1,2}).bytes();
        ConversationCache cache(one * 8, 2);
        cache.put(holder({1,2,3,4}, {1,2}));
        cache.put(holder({5,6,7,8}, {5,6}));
        check(!cache.evict_oldest() && cache.size() == 2, "the last holder of each pin is never evicted");
        check(!cache.put(image({9,10,11})) && cache.size() == 2 && cache.evictions() == 0,
              "parking is refused while every entry is a pin's last holder");
    }
    {   // Identical tokens with different steering modes are different pins.
        auto a = holder({1,2,3,4}, {1,2});
        auto b = holder({1,2,5,6}, {1,2}); b.cvec = false;
        ConversationCache cache((a.bytes() + b.bytes()) * 4, 2);
        check(cache.put(std::move(a)) && cache.put(std::move(b)), "different cvec holders parked");
        const size_t before = cache.bytes();
        check(!cache.evict_oldest() && cache.bytes() == before, "different cvec cannot replace the last holder");
    }
    {   // Identical tokens with different images are different pins.
        auto a = holder({1,2,3,4}, {1,2}); a.checkpoints[0].imgs = {{0, 100}};
        auto b = holder({1,2,5,6}, {1,2}); b.checkpoints[0].imgs = {{0, 200}};
        ConversationCache cache((a.bytes() + b.bytes()) * 4, 2);
        check(cache.put(std::move(a)) && cache.put(std::move(b)), "different image holders parked");
        check(!cache.evict_oldest() && cache.size() == 2, "image identity protects both pins");
    }
    {   // Every pin in a multi-pin entry must have a surviving holder; one duplicate is not enough.
        auto a = holder({1,2,3,4}, {1,2});
        auto cp = a.checkpoints[0]; cp.ids = {1}; a.checkpoints.push_back(cp);
        auto b = holder({1,2,5,6}, {1,2});
        const size_t kept = a.bytes();
        ConversationCache cache((a.bytes() + b.bytes()) * 4, 2);
        check(cache.put(std::move(a)) && cache.put(std::move(b)), "multi-pin and newer single-pin holders parked");
        check(cache.evict_oldest() && cache.size() == 1, "newer redundant holder goes when oldest has a unique pin");
        check(cache.bytes() == kept && cache.evictions() == 1, "eviction updates exact byte and entry accounting");
        check(cache.best(std::vector<int32_t>{1,0}, {}, true).tokens == 1, "unique short pin survives");
        check(cache.best(std::vector<int32_t>{1,2,0}, {}, true).tokens == 2, "shared pin also survives");
    }
    {   // Redundant image pins can make room under a byte budget, not only a slot limit.
        auto a = holder({1,2,3,4}, {1,2}); a.checkpoints[0].imgs = {{0, 100}};
        auto b = holder({1,2,5,6}, {1,2}); b.checkpoints[0].imgs = {{0, 100}};
        const size_t budget = a.bytes() + b.bytes(), kept = b.bytes();
        ConversationCache cache(budget, 8);
        check(cache.put(std::move(a)) && cache.put(std::move(b)), "image pin holders fill byte budget");
        check(cache.make_room(1) && cache.size() == 1 && cache.bytes() == kept, "redundant holder evicted for byte pressure");
        check(cache.best(std::vector<int32_t>{1,2,0}, {{0,100}}, true).tokens == 2, "same image pin remains usable");
    }
    {   // #2255 beside a borrow: the guarded donor stays and counts as a holder; redundant holders still make room
        auto cp = [](std::vector<int32_t> ids, bool pinned = false) {
            ConversationCheckpoint c; c.ids = std::move(ids); c.pinned = pinned; return c;
        };
        // two holders of one pin and an unpinned donor fill three slots: the precheck must not count both holders as
        // staying, or the borrow is refused although the oldest holder can go
        SavedConversation donor = image({7, 8, 9, 10, 11});
        donor.checkpoints = {cp({7, 8}), cp({7, 8, 9, 10})};
        ConversationCache cache(1 << 20, 3);
        check(cache.put(holder({1,2,3,4}, {1,2})) && cache.put(holder({1,2,5,6}, {1,2})) && cache.put(std::move(donor)),
              "two holders of one pin and a donor fill the slots");
        const auto m = cache.best(std::vector<int64_t>{7, 8, 40}, {}, true);
        check(cache.borrows(m), "a sibling on the donor's root borrows");
        cache.guard(m.serial);
        check(cache.put(image({9, 9, 9})) && !cache.guard_blocked() && cache.size() == 3 && cache.evictions() == 1,
              "the outgoing conversation parks beside the donor: a redundant holder makes room");
        check(cache.find(m.serial) != nullptr && cache.best(std::vector<int32_t>{1,2,3,4,0}, {}, true).tokens == 2 &&
              cache.best(std::vector<int32_t>{1,2,5,6,0}, {}, true).tokens == 4,
              "the oldest holder went; the newer holder and the donor stay");
        check(cache.put(image({6, 6, 6})) && !cache.guard_blocked() && cache.evictions() == 2 &&
              cache.best(std::vector<int64_t>{9, 9, 9, 1}, {}, true).tokens == 0,
              "the next outgoing one evicts the previous outgoing one, not the pin's last holder");
        check(cache.best(std::vector<int32_t>{1,2,0}, {}, true).tokens == 2 && cache.find(m.serial) != nullptr,
              "the pin and the donor survive further parks");
        cache.unguard();

        // the guarded donor holds the pin past the borrowed root: it keeps the pin parked, so an older holder can go,
        // and it is never the one evicted even when it is a redundant holder itself
        SavedConversation doc = image({1, 2, 3, 4, 5, 6, 30});
        doc.checkpoints = {cp({1, 2}), cp({1, 2, 3, 4, 5, 6}, true), cp({1, 2, 3, 4, 5, 6, 30})};
        ConversationCache two(1 << 20, 2);
        check(two.put(holder({1,2,3,4,5,6,50,51}, {1,2,3,4,5,6})) && two.put(std::move(doc)) && two.size() == 2,
              "an older holder and a donor holding the same pin fill two slots");
        const auto md = two.best(std::vector<int64_t>{1, 2, 40}, {}, true);
        check(md.tokens == 2 && two.borrows(md), "a match short of the pin borrows the donor");
        two.guard(md.serial);
        check(two.put(image({9, 9, 9})) && !two.guard_blocked() && two.size() == 2 && two.evictions() == 1,
              "the older holder goes: the guarded donor still holds the pin");
        check(two.find(md.serial) != nullptr &&
              two.best(std::vector<int32_t>{1,2,3,4,5,6,50,51,0}, {}, true).tokens == 6,
              "the donor stays and the pin is still found");
        two.unguard();
        ConversationCache young(1 << 20, 2);
        SavedConversation doc2 = image({1, 2, 3, 4, 5, 6, 30});
        doc2.checkpoints = {cp({1, 2}), cp({1, 2, 3, 4, 5, 6}, true), cp({1, 2, 3, 4, 5, 6, 30})};
        check(young.put(std::move(doc2)) && young.put(holder({1,2,3,4,5,6,50,51}, {1,2,3,4,5,6})),
              "a donor and a newer holder of the same pin fill two slots");
        const auto my = young.best(std::vector<int64_t>{1, 2, 40}, {}, true);
        check(young.borrows(my) && young.index_of(my.serial) == 0, "the oldest entry is the donor");
        young.guard(my.serial);
        check(young.put(image({9, 9, 9})) && !young.guard_blocked() && young.evictions() == 1 &&
              young.size() == 2 && young.find(my.serial) != nullptr &&
              young.best(std::vector<int32_t>{1,2,3,4,5,6,50,51,0}, {}, true).tokens == 6,
              "the oldest redundant holder is the guarded donor: the newer holder goes instead");
        young.unguard();
    }
    {   // evictable_bytes() during a borrow: the guarded donor is not reclaimable but keeps its pin held
        auto cp = [](std::vector<int32_t> ids, bool pinned = false) {
            ConversationCheckpoint c; c.ids = std::move(ids); c.pinned = pinned; return c;
        };
        SavedConversation doc = image({1, 2, 3, 4, 5, 6, 30});
        doc.checkpoints = {cp({1, 2}), cp({1, 2, 3, 4, 5, 6}, true), cp({1, 2, 3, 4, 5, 6, 30})};
        ConversationCache cache(1 << 20, 8);
        check(cache.put(std::move(doc)) && cache.put(holder({1,2,3,4,5,6,50,51}, {1,2,3,4,5,6})) &&
              cache.put(holder({1,2,3,4,5,6,60,61}, {1,2,3,4,5,6})) && cache.put(image({9, 9, 9})),
              "a donor, two holders of its pin and an ordinary conversation parked");
        const auto m = cache.best(std::vector<int64_t>{1, 2, 40}, {}, true);
        check(cache.borrows(m) && cache.index_of(m.serial) == 0, "the oldest entry is the donor");
        const size_t donor_bytes = cache.find(m.serial)->bytes(), before = cache.bytes();
        const auto newest = cache.best(std::vector<int64_t>{1, 2, 3, 4, 5, 6, 60, 61, 0}, {}, true);
        check(newest.tokens == 8 && cache.find(newest.serial) != nullptr, "the newest holder is found");
        check(cache.evictable_bytes() == before - cache.find(newest.serial)->bytes(),
              "unguarded: everything but the pin's last holder (the newest) is reclaimable");
        cache.guard(m.serial);
        const size_t preview = cache.evictable_bytes();
        check(preview == before - donor_bytes, "guarded: everything but the donor, which keeps the pin held");
        while (cache.evict_oldest()) {}
        check(cache.size() == 1 && cache.find(m.serial) != nullptr && preview == before - cache.bytes(),
              "the guarded drain matches the preview and leaves only the donor");
        cache.unguard();
    }
    {
        // the retained reuse image belongs to the conversation it was restored from.  The production bug: with
        // --kv-pool-tokens a solo request's conversation moved into a batch slot and the session forgot it, the
        // retained K/V stayed behind, the next request resumed ANOTHER conversation from a different slot, and
        // that one's next park reused the first conversation's bytes as its own unchanged prefix.
        auto kv = [](size_t n) { std::vector<ConversationKv> v(1); v[0].k.resize(n, 1); return v; };
        ConversationCache cache(1 << 20, 4);
        const std::vector<int32_t> root = {1, 2, 3, 4};              // the system prompt both share
        const std::vector<int32_t> conv_a = {1, 2, 3, 4, 5, 6, 7, 8};
        const std::vector<int32_t> conv_b = {1, 2, 3, 4, 9, 10};
        cache.retain(kv(100), int64_t(conv_a.size()), {}, conv_a);   // A restored, then parked again
        cache.limit_reuse(int64_t(conv_b.size()));                   // the session now holds B: limit_reuse alone
        auto r = cache.take_reuse(conv_b, {});                       // would keep A's bytes for B's first 6 cells
        check(!r.kv.empty() && r.unchanged_tokens == 4,
              "a park of another conversation reuses only the prefix the two share");
        cache.retain(kv(100), int64_t(conv_a.size()), {}, conv_a);
        auto gone = cache.take_reuse();   // the park early-returned: the session holds no conversation at all
        check(gone.kv.empty() && cache.retained_bytes() == 0, "asking for the reuse with no conversation drops it");
        cache.retain(kv(100), int64_t(conv_a.size()), {}, conv_a);
        std::vector<int32_t> grown = conv_a;
        grown.push_back(9);
        r = cache.take_reuse(grown, {});
        check(r.unchanged_tokens == 8, "the same conversation, continued, keeps its retained prefix");
        cache.retain(kv(100), 8, {}, conv_a);
        cache.limit_reuse(5);
        r = cache.take_reuse(conv_a, {});
        check(r.unchanged_tokens == 5, "the first rewrite still bounds the reuse");
        cache.retain(kv(100), 8, {}, conv_a);
        cache.retain(kv(100), 6, {}, conv_b);   // B restored: its bytes, B's identity
        r = cache.take_reuse(conv_a, {});
        check(r.unchanged_tokens == 4, "retaining another conversation replaces the identity with it");
        std::vector<std::vector<ConversationKv>> stage_kv;
        stage_kv.push_back(kv(50));
        cache.retain(kv(100), 8, std::move(stage_kv), conv_a);
        r = cache.take_reuse(conv_b, {});
        check(r.stages.size() == 1 && r.stages[0].unchanged_tokens == 4,
              "a later stage's reuse is bounded by the shared prefix too");
        // the pictures: equal placeholder tokens over different pixels share only the cells before the picture
        const std::vector<ConversationImageKey> img_a{{2, 111}}, img_b{{2, 222}};
        cache.retain(kv(100), 4, {}, root, img_a);
        r = cache.take_reuse(root, img_b);
        check(r.unchanged_tokens == 2, "the same tokens under different pixels stop at the picture");
        cache.retain(kv(100), 4, {}, root, img_a);
        r = cache.take_reuse(root, {});
        check(r.unchanged_tokens == 2, "a conversation without the picture reuses nothing past its start");
        cache.retain(kv(100), 4, {}, root, img_a);
        r = cache.take_reuse(root, img_a);
        check(r.unchanged_tokens == 4, "the same tokens and the same pixels keep everything");
        check(conversation_shared_prefix({1, 2, 3}, {}, {1, 2}, {}) == 2, "shared prefix: the shorter prefix");
        check(conversation_shared_prefix({1, 2, 3}, {}, {1, 9, 3}, {}) == 1, "shared prefix: to the first difference");
        check(conversation_shared_prefix({1, 2}, {{1, 5}}, {1, 2}, {{1, 5}}) == 2, "shared prefix: equal pictures pass");
        ConversationCache budgeted(1 << 20, 4);
        budgeted.retain(kv(100), 8, {}, conv_a);
        check(budgeted.bytes() > budgeted.retained_bytes(), "the retained identity counts against the budget");
    }
    {   // disk mode: budget counts file bytes, eviction removes entries (caller deletes files via at())
        ConversationCache c(1000, 2);
        c.enable_disk();
        SavedConversation a = image({1, 2, 3});           // the file's existing slim builder (RAM payload irrelevant here)
        a.live.gdn.clear();
        check(c.put_disk(std::move(a), "/tmp/a.sess", 600), "disk put accepts under budget");
        SavedConversation b = image({4, 5}); b.live.gdn.clear();
        check(c.put_disk(std::move(b), "/tmp/b.sess", 300), "second disk entry");
        SavedConversation d = image({6}); d.live.gdn.clear();
        check(c.put_disk(std::move(d), "/tmp/d.sess", 300), "disk budget evicts oldest (600+300+300 > 1000)");
        check(c.size() == 2 && c.bytes() == 600, "bytes counts disk sizes, oldest first evicted");
        check(c.at(0).disk_path == "/tmp/b.sess", "survivors keep their paths");
        check(c.best(std::vector<int32_t>{4, 5, 9}, {}, true).tokens == 2, "matching works on the slim index");
        c.remove_at(1);
        check(c.size() == 1 && c.bytes() == 300, "remove_at erases and unbudgets");
        SavedConversation over = image({9}); over.live.gdn.clear();
        check(!c.put_disk(std::move(over), "/tmp/over.sess", 1500) && c.size() == 1 && c.bytes() == 300,
              "one disk entry bigger than the whole budget is refused, survivors untouched");
        ConversationCache r(1000, 2);                     // RAM mode unchanged
        check(r.put(image({1, 2, 3})), "RAM mode still works");
    }
    {   // the unlink rule with real files: evict_oldest, drop_superseded and clear DELETE the entries' files
        namespace fs = std::filesystem;
        const fs::path dir = fs::temp_directory_path() / "strata_conv_cache_unlink_test";
        std::error_code ec;
        fs::remove_all(dir, ec);
        fs::create_directories(dir);
        auto real_file = [&](const char* n) {
            const fs::path p = dir / n;
            std::FILE* h = std::fopen(p.string().c_str(), "w");
            std::fputs("x", h);
            std::fclose(h);
            return p.string();
        };
        const std::string pa = real_file("a.sess"), pb = real_file("b.sess"), pd = real_file("d.sess"),
                        pe = real_file("e.sess");
        ConversationCache c(1000, 3);
        c.enable_disk();
        SavedConversation a = image({1, 2, 3}); a.live.gdn.clear();
        SavedConversation b = image({4, 5}); b.live.gdn.clear();
        SavedConversation d = image({6}); d.live.gdn.clear();
        SavedConversation e = image({7, 8}); e.live.gdn.clear();
        e.checkpoints.push_back(ConversationCheckpoint{});
        e.checkpoints.back().ids = {4, 5};
        check(c.put_disk(std::move(a), pa, 600) && c.put_disk(std::move(b), pb, 300), "two real files parked");
        check(c.put_disk(std::move(d), pd, 300), "the third evicts the oldest (600+300+300 > 1000)");
        check(!fs::exists(pa) && fs::exists(pb) && fs::exists(pd), "evicting unlinks the victim's file, not the survivors'");
        check(c.put_disk(std::move(e), pe, 100), "fourth entry, superseded shape");
        std::vector<ConversationCheckpoint> held_by_new_chain;
        ConversationCheckpoint cp; cp.ids = {4, 5};
        held_by_new_chain.push_back(cp);
        // take_superseded_reuse must leave disk entries untouched for drop_superseded to unlink and debit disk_bytes.
        auto disk_harvest = c.take_superseded_reuse({9}, {}, held_by_new_chain, true);
        check(disk_harvest.dropped == 0 && c.size() == 3 && c.bytes() == 700 && fs::exists(pe),
              "take_superseded_reuse does not steal disk entries without unlinking them");
        check(c.drop_superseded({9}, {}, held_by_new_chain, true) == 1 && c.bytes() == 600,
              "the superseded entry drops and debits disk_bytes");
        check(!fs::exists(pe) && fs::exists(pb) && fs::exists(pd), "drop_superseded unlinks the dropped entry's file");
        c.clear();
        check(c.size() == 0 && !fs::exists(pb) && !fs::exists(pd), "clear unlinks every parked file");
        fs::remove_all(dir, ec);
    }
    {   // meta_only copy keeps matching working without state
        SavedConversation s = image({1, 2, 3});
        s.checkpoints.push_back(ConversationCheckpoint{});
        s.checkpoints.back().ids = {1, 2};
        s.checkpoints.back().gdn.assign(64, 7);
        SavedConversation slim = conversation_meta_only(s);
        check(slim.live.ids == s.live.ids && slim.checkpoints.back().ids == std::vector<int32_t>({1, 2})
              && slim.checkpoints.back().gdn.empty() && slim.kv.empty() && slim.live.gdn.empty(),
              "meta_only keeps ids/imgs, drops state and K/V");
    {
        ConversationBuffer bytes;
        bytes.resize(ConversationBuffer::segment_bytes, 1);
        uint8_t* first = nullptr;
        bytes.visit(0, 1, [&](uint8_t* p, size_t, size_t) { first = p; return true; });
        bytes.resize(ConversationBuffer::segment_bytes, 2);
        uint8_t* second = nullptr;
        bytes.visit(0, 1, [&](uint8_t* p, size_t, size_t) { second = p; return true; });
        check(first != nullptr && first == second, "a same-length resize keeps the segment pages");
    }
    {
        // #3321: a superseded park is moved, not freed, so the next capture can write the warm pages.
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        auto with_kv = [](std::vector<int32_t> ids) {
            SavedConversation s = image({});
            s.live.ids = std::move(ids);
            ConversationKv layer;
            layer.cells = 8;
            layer.k.resize(ConversationBuffer::segment_bytes, 9);
            s.kv.push_back(std::move(layer));
            ConversationKv stage_layer;
            stage_layer.k.resize(1 << 20, 3);
            SavedConversation stage;
            stage.kv.push_back(std::move(stage_layer));
            s.stage_images.push_back(std::move(stage));
            return s;
        };
        ConversationCache cache(64 << 20, 4);
        SavedConversation old = with_kv({1, 2, 3, 4, 10, 11});
        old.checkpoints = {cp({1, 2, 3, 4})};
        uint8_t* page = nullptr;
        old.kv[0].k.visit(0, 1, [&](uint8_t* p, size_t, size_t) { page = p; return true; });
        const size_t kept = old.kv[0].k.bytes() + old.stage_images[0].kv[0].k.bytes();
        check(cache.put(std::move(old)), "park the copy a turn back");
        SavedConversation other = image({9, 9, 9});
        other.checkpoints = {cp({9, 9})};
        check(cache.put(std::move(other)), "park an unrelated conversation");
        const std::vector<int32_t> live = {1, 2, 3, 4, 10, 12, 13};
        const std::vector<ConversationCheckpoint> chain = {cp({1, 2, 3, 4})};
        const auto donor = cache.best(std::vector<int32_t>{1, 2, 3, 4, 10, 11, 99}, {}, true).serial;
        check(donor != 0, "the copy a turn back is still parked");
        cache.guard(donor);
        auto blocked = cache.take_superseded_reuse(live, {}, chain, true);
        check(blocked.dropped == 0 && blocked.reuse.kv.empty() && cache.size() == 2,
              "a guarded donor is not harvested");
        cache.unguard();
        auto got = cache.take_superseded_reuse(live, {}, chain, true);
        check(got.dropped == 1 && cache.size() == 1 && cache.superseded() == 1, "the superseded copy is removed");
        check(cache.bytes() < (8 << 20), "harvested pages are no longer charged to the cache");
        check(got.reuse.unchanged_tokens == 0 && got.reuse.captured_tokens == 6, "pages are recopied, not trusted as a prefix");
        check(got.reuse.kv.size() == 1 && got.reuse.stages.size() == 1 && got.reuse.stages[0].kv.size() == 1,
              "main and stage K/V both move");
        uint8_t* moved = nullptr;
        got.reuse.kv[0].k.visit(0, 1, [&](uint8_t* p, size_t, size_t) { moved = p; return true; });
        check(moved == page && got.reuse.bytes() >= kept, "the segment pages themselves are kept");
        auto none = cache.take_superseded_reuse(live, {}, chain, true);
        check(none.dropped == 0 && none.reuse.kv.empty(), "a second take finds nothing");
        SavedConversation sibling = with_kv({1, 2, 3, 4, 80});
        sibling.checkpoints = {cp({1, 2, 3, 4, 80})};
        check(cache.put(std::move(sibling)), "park a sibling that only shares the root");
        auto kept_sibling = cache.take_superseded_reuse(live, {}, chain, true);
        check(kept_sibling.dropped == 0 && cache.size() == 2, "a different deepest checkpoint is not harvested");
    }
    std::printf("conversation_cache_test: %d checks passed\n", checks);
}
}
