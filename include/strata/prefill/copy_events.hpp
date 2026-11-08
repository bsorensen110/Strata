// include/strata/prefill/copy_events.hpp - which streamed expert copies get a "copied" event (#4005).
//
// The streamed prompt walk issues one host-to-device copy per expert entry on the copy stream and the compute stream
// waits for it.  By default every entry records its own event (every == 1) and the compute stream waits on the event of
// the entry it needs.  STRATA_PREFILL_COPY_EVENT_EVERY=N (tester switch) records every N-th entry only, and the
// compute stream waits on the first recorded entry at or after the one it needs: the copy stream is in order, so that
// event covers it.  The two rules are pure functions so a host test can check them without a GPU.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace strata::prefill {

/// STRATA_PREFILL_FUSED_BATCH_RELEASE: enable one tail release event for each fused streamed batch.
inline bool fused_release_batch_from(const char* v, bool default_on = false) {
    if (v == nullptr || v[0] == '\0') return default_on;
    return std::atoi(v) != 0;
}

/// STRATA_PREFILL_STAGE_LOOKAHEAD: enable full STAGE_GRP - MMQ_GROUP pending transfer lookahead in grouped staged walk.
inline bool stage_lookahead_from(const char* v, bool default_on = false) {
    if (v == nullptr || v[0] == '\0') return default_on;
    return std::atoi(v) != 0;
}

/// Maximum in-flight non-resident transfers ahead of the consumer in the staged walk (`!stream_all`).
/// When `group_resident` is active, an open MMQ group holds at most `mmq_group - 1` already-decremented slots
/// in the `stage_grp`-slot ring before flush, so up to `stage_grp - mmq_group` transfers can safely remain pending.
inline size_t staged_pending_cap(bool group_resident, bool full_group_lookahead,
                                 size_t stage, size_t stage_grp, size_t mmq_group) {
    if (group_resident && full_group_lookahead && stage_grp > mmq_group) {
        const size_t cap = stage_grp - mmq_group;
        return cap > stage ? cap : stage;
    }
    return stage;
}

/// Event slot that releases a non-empty fused streamed batch [first, end) in the ring.
inline int copy_release_event_slot(size_t first, size_t end, int ring) {
    if (ring <= 0 || end <= first) return -1;
    return (int) ((end - 1) % (size_t) ring);
}

/// The value of STRATA_PREFILL_COPY_EVENT_EVERY: 1 (every entry) unless the text is a number above 1; at most 64.
inline int copy_event_every_from(int parsed) { return parsed > 1 ? (parsed < 64 ? parsed : 64) : 1; }

/// STRATA_PREFILL_COPY_WAIT_DEDUPE: unset or empty uses `default_on`; any non-zero number is on.
inline bool copy_wait_dedupe_from(const char* v, bool default_on = false) {
    if (v == nullptr || v[0] == '\0') return default_on;
    return std::atoi(v) != 0;
}

/// Entry `idx` of `total` records its copied event right after its copy: always with every == 1; else every N-th entry
/// and the last one.
inline bool copy_event_due(size_t idx, size_t total, int every) {
    return every <= 1 || (idx + 1) % (size_t) every == 0 || idx + 1 == total;
}

/// The entry whose recorded event a consumer waits on for entry `k`, when the entries [0, recorded) have had their
/// events decided (`recorded` = 1 + the last recorded entry) and `is_recorded(r)` says whether entry r has one.  The
/// first recorded entry at or after k; the last recorded entry (recorded - 1) when none is found, which covers k as
/// well.  Precondition: k < recorded.
template <class IsRecorded>
inline size_t copy_event_entry(size_t k, size_t recorded, IsRecorded is_recorded) {
    for (size_t r = k; r < recorded; ++r)
        if (is_recorded(r)) return r;
    return recorded - 1;
}

/// STRATA_PREFILL_COPY_WAIT_DEDUPE: skip a wait if the previous entry in this walk already waited on the same release event.
/// This is safe only when both slots were released in this walk (idx - 1 >= ring) and ring > group (a full-ring flush
/// would re-record the event after the previous wait).
inline bool copy_wait_dedupe_ok(size_t idx, int ring, int group, int rel, int prev_rel) {
    return ring > group && idx > (size_t) ring && rel == prev_rel;
}

}  // namespace strata::prefill
