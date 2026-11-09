// Host test for include/strata/prefill/copy_events.hpp (#4005): the copied-event rules of the streamed prompt walk.
#include "strata/prefill/copy_events.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace strata::prefill;

static int g_fail = 0;
static void check(bool ok, const char* what) {
    if (!ok) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

int main() {
    check(copy_event_every_from(0) == 1 && copy_event_every_from(1) == 1 && copy_event_every_from(-5) == 1, "unset / 0 / 1 / negative: every entry");
    check(copy_event_every_from(2) == 2 && copy_event_every_from(16) == 16 && copy_event_every_from(64) == 64, "2..64 as given");
    check(copy_event_every_from(65) == 64 && copy_event_every_from(100000) == 64, "above 64 is 64");
    check(!copy_wait_dedupe_from(nullptr, false) && !copy_wait_dedupe_from("", false), "dedupe unset/empty uses default_on=false");
    check(copy_wait_dedupe_from(nullptr, true) && copy_wait_dedupe_from("", true), "dedupe unset/empty uses default_on=true");
    check(!copy_wait_dedupe_from("0", true) && copy_wait_dedupe_from("1", false), "dedupe explicit 0/1 overrides default_on");

    check(!fused_release_batch_from(nullptr, false) && !fused_release_batch_from("", false), "fused batch release defaults off");
    check(fused_release_batch_from("1", false) && !fused_release_batch_from("0", true), "fused batch release env overrides default");

    check(!stage_lookahead_from(nullptr, false) && !stage_lookahead_from("", false), "stage lookahead defaults off");
    check(stage_lookahead_from("1", false) && !stage_lookahead_from("0", true), "stage lookahead env overrides default");
    check(staged_pending_cap(false, false, 8, 32, 16) == 8 && staged_pending_cap(true, false, 8, 32, 16) == 8,
          "stage lookahead off keeps STAGE=8 pending cap");
    check(staged_pending_cap(false, true, 8, 32, 16) == 8,
          "non-grouped staged walk keeps STAGE=8 pending cap");
    check(staged_pending_cap(true, true, 8, 32, 16) == 16,
          "grouped staged walk with lookahead on keeps STAGE_GRP - MMQ_GROUP = 16 transfers ahead");
    check(staged_pending_cap(true, true, 8, 16, 16) == 8,
          "grouped staged walk falls back to STAGE when stage_grp <= mmq_group");

    check(copy_release_event_slot(0, 32, 96) == 31, "fused batch releases at its last ring slot");
    check(copy_release_event_slot(32, 64, 96) == 63, "next fused batch gets its own release slot");
    check(copy_release_event_slot(96, 128, 96) == 31, "fused release slot wraps with the ring");
    check(copy_release_event_slot(8, 8, 96) == -1 && copy_release_event_slot(0, 1, 0) == -1,
          "empty or zero-ring fused batch has no release event");
    {
        // Simulate a 96-slot co-prefill ring released in 32-slot fused batches [0,32), [32,64), [64,96):
        // on the second revolution [96, 192), only the first slot of each batch waits; the other 31 dedupe.
        constexpr int ring = 96;
        constexpr int group = 16;
        int used_of[ring] = {};
        for (size_t b0 = 0; b0 < (size_t) ring; b0 += 32) {
            const int rel = copy_release_event_slot(b0, b0 + 32, ring);
            for (size_t k = b0; k < b0 + 32; ++k) used_of[k % (size_t) ring] = rel;
        }
        int prev_rel = used_of[ring - 1];   // entry 95's wait at the end of the first revolution
        int waits = 0, dedup = 0;
        for (size_t idx = (size_t) ring; idx < (size_t) (2 * ring); ++idx) {
            const int rel = used_of[idx % (size_t) ring];
            if (copy_wait_dedupe_ok(idx, ring, group, rel, prev_rel)) ++dedup;
            else ++waits;
            prev_rel = rel;
        }
        check(waits == 3 && dedup == 93, "fused batch release deduplicates 93 of 96 waits on second ring revolution");
    }

    // default: every entry records (the walk then waits on copied[slot], as before the switch)
    for (size_t total : {1u, 2u, 7u, 100u})
        for (size_t i = 0; i < total; ++i) check(copy_event_due(i, total, 1), "every == 1 records every entry");

    // sparse: the walk's issuer/consumer as a model.  The issuer records per copy_event_due (and, when it must wait for
    // the consumer, the entry before the one it is about to issue); the consumer for entry k waits on
    // copy_event_entry(k, recorded, ...).  Every wait must name a recorded entry at or after k, and the ring must
    // never need an event that a later entry has overwritten.
    std::srand(4005);
    for (int every : {2, 3, 5, 16, 64})
        for (int ring : {2, 4, 8, 96})
            for (size_t total : {1u, 5u, 33u, 300u}) {
                std::vector<long long> rec_of((size_t) ring, -1);
                size_t recorded = 0, issued = 0, consumed = 0;
                long long last_rec = -1;
                bool ok = true;
                auto record = [&](size_t i) {
                    if (rec_of[i % (size_t) ring] >= 0 && (size_t) rec_of[i % (size_t) ring] >= consumed) ok = false;   // overwrote a live event
                    rec_of[i % (size_t) ring] = (long long) i;
                    recorded = i + 1;
                    last_rec = (long long) i;
                };
                auto cover = [&](size_t idx) { if (idx > 0 && last_rec < (long long) idx - 1) record(idx - 1); };
                size_t guard = 0;
                while (consumed < total && ok && ++guard < 100000) {
                    const bool can_issue = issued < total && issued < consumed + (size_t) ring;
                    const bool can_consume = consumed < issued;
                    const bool pick_issue = can_issue && (!can_consume || std::rand() % 2);
                    if (pick_issue) {
                        const size_t idx = issued;
                        if (std::rand() % 4 == 0) cover(idx);   // the issuer had to wait (ring full or stager)
                        if (copy_event_due(idx, total, every)) record(idx);
                        ++issued;
                    } else if (can_consume) {
                        if (issued == total || true) {
                            // a consumer needs recorded > k; when the issuer is blocked it has covered what it issued
                            if (recorded <= consumed) {
                                if (issued < total && !(issued < consumed + (size_t) ring)) cover(issued);
                                if (issued == total) cover(issued);
                            }
                            if (recorded <= consumed) { continue; }   // would spin until the issuer records
                            const size_t r = copy_event_entry(consumed, recorded, [&](size_t e) { return rec_of[e % (size_t) ring] == (long long) e; });
                            if (r < consumed || r >= recorded || rec_of[r % (size_t) ring] != (long long) r) ok = false;
                            ++consumed;
                        }
                    } else {
                        break;
                    }
                }
                char what[128];
                std::snprintf(what, sizeof what, "sparse every=%d ring=%d total=%zu: every wait names a live recorded entry", every, ring, total);
                check(ok && consumed == total, what);
            }

    // copy_wait_dedupe_ok: skipping the copy stream's wait on a slot's release event must never leave a slot's refill
    // without a wait on a record at or after the flush that released it.  Two walks of random length with random group
    // sizes (1..group), the issuer and the consumer interleaved at random.  A "record" is a counter tick of the event.
    auto simulate = [&](int ring, int group, bool original_rule, unsigned seed) -> bool {
        std::srand(seed);
        std::vector<int> used_of((size_t) ring);
        std::vector<long long> rec_time((size_t) ring, 0), last_flush((size_t) ring, 0);
        for (int i = 0; i < ring; ++i) used_of[(size_t) i] = i;
        long long tick = 0;
        for (int walk = 0; walk < 3; ++walk) {
            const size_t n = (size_t) ring * (size_t) (2 + std::rand() % 3) + (size_t) (std::rand() % (ring + 1));
            std::vector<size_t> group_end;   // the last entry of each group, in order
            for (size_t s = 0; s < n;) {
                size_t len = (size_t) (1 + std::rand() % group);
                if (s + len > n) len = n - s;
                s += len;
                group_end.push_back(s - 1);
            }
            size_t issued = 0, consumed = 0, g = 0;
            const int issue_pct = std::vector<int>{50, 90, 99}[(size_t) (std::rand() % 3)];   // how far ahead the issuer runs
            int prev_rel = -1;
            long long prev_cap = 0;
            std::vector<bool> live((size_t) ring, walk > 0);   // the first walk starts with empty slots
            size_t guard = 0;
            while (consumed < n && ++guard < 1000000) {
                const bool can_issue = issued < n && issued < consumed + (size_t) ring;
                const bool can_flush = g < group_end.size() && group_end[g] < issued;
                if (can_issue && (!can_flush || std::rand() % 100 < issue_pct)) {
                    const size_t idx = issued++;
                    const int sl = (int) (idx % (size_t) ring);
                    if (live[(size_t) sl]) {
                        const int rel = used_of[(size_t) sl];
                        const bool skip = original_rule ? (idx >= (size_t) ring && rel == prev_rel)
                                                        : copy_wait_dedupe_ok(idx, ring, group, rel, prev_rel);
                        const long long cap = skip ? prev_cap : rec_time[(size_t) rel];
                        if (cap < last_flush[(size_t) sl]) return false;   // the refill would not wait for its release
                        prev_cap = cap;
                        prev_rel = rel;
                    } else {
                        prev_rel = -1;
                    }
                    live[(size_t) sl] = true;
                } else if (can_flush) {
                    size_t first = consumed;
                    const size_t last = group_end[g++];
                    const int rel = (int) (last % (size_t) ring);
                    rec_time[(size_t) rel] = ++tick;
                    for (size_t e = first; e <= last; ++e) {
                        used_of[e % (size_t) ring] = rel;
                        last_flush[e % (size_t) ring] = tick;
                    }
                    consumed = last + 1;
                } else {
                    return false;   // stuck: the model is wrong
                }
            }
        }
        return true;
    };
    for (int ring : {17, 18, 24, 32, 96, 1024})
        for (unsigned seed = 1; seed <= 300; ++seed) {
            char what[96];
            std::snprintf(what, sizeof what, "dedupe never skips a needed wait (ring %d, seed %u)", ring, seed);
            if (!simulate(ring, 16, false, seed)) { check(false, what); break; }
        }
    {
        bool rule_ok = true;   // the rule refuses a ring that a group can fill
        for (size_t idx : {20u, 1000u}) rule_ok = rule_ok && !copy_wait_dedupe_ok(idx, 16, 16, 3, 3);
        check(rule_ok, "a ring of one group never dedupes");
        check(!copy_wait_dedupe_ok(32, 32, 16, 3, 3) && copy_wait_dedupe_ok(33, 32, 16, 3, 3), "the first round never dedupes");
        check(!copy_wait_dedupe_ok(40, 32, 16, 3, 4) && !copy_wait_dedupe_ok(40, 32, 16, 3, -1), "different events or none: wait");
        // the control: the PR's first rule (idx >= ring) is caught by this model
        bool caught = false;
        for (int ring : {17, 18, 20, 24, 32}) for (unsigned seed = 1; seed <= 600 && !caught; ++seed) caught = !simulate(ring, 16, true, seed);
        check(caught, "the model finds the stale-event race in the rule 'idx >= ring'");
    }

    // staged_can_issue: in grouped staged walk (STAGE_GRP = 32, MMQ_GROUP = 16), up to 32 transfers (2 full groups)
    // can be pending when gg_nslots == 0, while pending + gg_nslots <= 32 prevents reusing an open group's slot before flush.
    check(strata::prefill::staged_can_issue(true, true, true, 31, 0, 8, 32, 16, 0, 0, 15),
          "grouped lookahead: allows 31st transfer when open group has 0 slots");
    check(!strata::prefill::staged_can_issue(true, true, true, 32, 0, 8, 32, 16, 0, 0, 15),
          "grouped lookahead: blocks 32nd transfer when 32 are already pending");
    check(strata::prefill::staged_can_issue(true, true, true, 16, 15, 8, 32, 16, 0, 0, 15),
          "grouped lookahead: allows 16 pending when open group holds 15 slots");
    check(!strata::prefill::staged_can_issue(true, true, true, 17, 15, 8, 32, 16, 0, 0, 15),
          "grouped lookahead: blocks 17th pending when open group holds 15 slots (prevents slot overwrite before flush)");
    check(strata::prefill::staged_can_issue(true, true, false, 7, 0, 8, 32, 16, 0, 0, 15) &&
          !strata::prefill::staged_can_issue(true, true, false, 8, 0, 8, 32, 16, 0, 0, 15),
          "grouped lookahead off: keeps legacy STAGE=8 cap");
    {
        // Simulate a grouped staged walk over 32 resident + 64 staged experts with STAGE_GRP=32, MMQ_GROUP=16
        constexpr size_t n_exp = 96, stage_grp = 32, mmq_group = 16;
        std::vector<int> stage_of(n_exp, -1), used_of(stage_grp, -1);
        std::vector<long long> rec_time(stage_grp, 0), slot_flush(stage_grp, 0);
        std::vector<bool> live(stage_grp, false);
        int stage_next = 0, gg_nslots = 0;
        int gg_slots[mmq_group] = {};
        size_t staged = 0, pending = 0, max_pending_at_start = 0;
        long long tick = 0;
        bool walk_ok = true;
        for (size_t j = 0; j < n_exp; ++j) {
            while (staged < n_exp &&
                   strata::prefill::staged_can_issue(true, true, true, pending, (size_t) gg_nslots, 8, stage_grp, mmq_group,
                                                     staged, j, stage_grp - mmq_group - 1)) {
                const bool resident = staged < 32;
                if (!resident) {
                    const int sl = stage_next;
                    stage_next = (stage_next + 1) % (int) stage_grp;
                    if (live[(size_t) sl]) {
                        const int rel = used_of[(size_t) sl];
                        if (rel < 0 || rec_time[(size_t) rel] < slot_flush[(size_t) sl]) walk_ok = false;
                        // Also ensure sl is not currently sitting in the open group's gg_slots
                        for (int k = 0; k < gg_nslots; ++k)
                            if (gg_slots[k] == sl) walk_ok = false;
                    }
                    live[(size_t) sl] = true;
                    slot_flush[(size_t) sl] = ++tick;   // cannot be reused until a flush after this tick
                    stage_of[staged] = sl;
                    ++pending;
                }
                ++staged;
            }
            if (j == 0) max_pending_at_start = pending;
            const size_t q = j % mmq_group;
            const bool last_of_group = q + 1 == mmq_group || j + 1 == n_exp;
            if (stage_of[j] >= 0) {
                gg_slots[gg_nslots++] = stage_of[j];
            }
            if (last_of_group) {
                if (gg_nslots > 0) {
                    const int rel = gg_slots[gg_nslots - 1];
                    rec_time[(size_t) rel] = ++tick;
                    for (int i = 0; i < gg_nslots; ++i) {
                        used_of[(size_t) gg_slots[i]] = rel;
                        slot_flush[(size_t) gg_slots[i]] = tick;
                    }
                    gg_nslots = 0;
                }
            }
            if (stage_of[j] >= 0) --pending;
        }
        check(walk_ok && max_pending_at_start == 32 && pending == 0 && gg_nslots == 0,
              "grouped staged walk: pre-stages 32 transfers across resident prefix without clobbering open group slots");
    }

    if (g_fail == 0) std::printf("copy_events_test: all checks passed\n");
    return g_fail == 0 ? 0 : 1;
}
