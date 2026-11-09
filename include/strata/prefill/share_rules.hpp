// include/strata/prefill/share_rules.hpp - when the default CPU prefill share is worth arming (no GPU needed to check).
//
// The share hands the idle CPU pool experts the GPU would otherwise stream over PCIe.  With every expert resident in
// the GPU's cache nothing is streamed, so there is nothing to take; arming the default only moves the staged-chunk
// limit and costs ~5% at 1,000 tokens (#1595: V100, 3 rounds x 3 runs).  An explicit STRATA_PREFILL_CPU_SHARE is
// honoured by Prefill::arm_cpu_share whatever this says.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace strata::prefill::share_rules {

constexpr bool batch_cpu_share_blocked(bool share_enabled, bool concurrent_reader) {
    return share_enabled && concurrent_reader;
}

/// Whether active single-card batch slots need the co-prefill ring policy, regardless of CPU-pool ownership.
constexpr bool batch_co_prefill_active(bool single_card_batch, bool active_slots) {
    return single_card_batch && active_slots;
}

/// Whether serial batch prefill shares the CPU pool (STRATA_PREFILL_CPU_SHARE_BATCH).
/// Defaults ON for single-card gfx12 when batch is enabled by the engine/user (`single_card_gfx12 && batch_enabled`).
inline bool batch_cpu_share_enabled_from(const char* v, bool single_card_gfx12, bool batch_enabled) {
    if (v != nullptr && v[0] != '\0') return std::atoi(v) != 0;
    return single_card_gfx12 && batch_enabled;
}

/// Whether the loan backup uses page-locked PCIe memory (STRATA_LEND_PCIE_PINNED).
/// Defaults ON for single-card gfx12 when batch is enabled by the engine/user (`single_card_gfx12 && batch_enabled`).
inline bool batch_lend_pinned_from(const char* v, bool single_card_gfx12, bool batch_enabled) {
    if (v != nullptr && v[0] != '\0') return std::atoi(v) != 0;
    return single_card_gfx12 && batch_enabled;
}

struct BatchPrefillPolicy {
    bool block_cpu_share = false;
    bool co_prefill_ring = false;
};

constexpr BatchPrefillPolicy batch_prefill_policy(bool batch_enabled, bool concurrent_reader,
                                                  bool share_enabled, bool active_slots) {
    return {batch_cpu_share_blocked(share_enabled, concurrent_reader),
            batch_co_prefill_active(batch_enabled, active_slots)};
}

// The DEFAULT share applies to staged chunks up to this many tokens.  0.1.41 armed it below 1,024 tokens (RTX 3060, P100, RTX 5070: -20..-35% at 512 and
// 1,000 tokens).  A chunk of 1-3K tokens routes to nearly every expert of every layer, so with half the experts resident one pass over the rest at the link's
// speed is its floor (RTX PRO 5000 with a 28 GB/s link and 12,437 of 24,576 experts resident: 0.97-1.04 s for any 1K-4K prompt) and the CPU share takes
// part of that pass: prompt speed +151% at 1K, +45% at 2K (TTFT 1.07 -> 0.74 s), +99% / +32% with 8 pool workers; 4K and above unchanged, and a limit
// of 8,192 loses 1-6% at 4K-8K.  First-token KL after a 1.1-2.8K prompt: mean 0.011, top-1 100%.  STRATA_PREFILL_CPU_SHARE_MAX=1024 restores the old limit.
constexpr int64_t kShareLimitOld = 1024;
constexpr int64_t kShareLimitNew = 3072;
/// `raise`: the card class may take the new limit (a class whose A/B lost at 1-3K keeps 1,024, automatically).
constexpr int64_t default_share_limit(bool raise = true) { return raise ? kShareLimitNew : kShareLimitOld; }
/// AMD-Q: a gfx12 card whose fused IQ2_S / IQ4_XS gate/up kernels are on (fused::gfx12_iq2s_default): chunks below this are staged and take the CPU share, a chunk
/// from it up streams every expert and runs the fused kernels.  R9700 (p620, IQ3_S, 54% resident), prompt speed against the share-only default (limit 3,072):
/// limit 1,536: 1.3K 1.00, 1.5K 1.00, 1.8K +7.1%, 2K +10.0%, 2.6K +16.2%, 3K +11.1% (two pairs each); limit 2,048: 1.8K +1.4%, 2K +9.7%; limit 3,072 (fused only above): no gain at 1.5K-2.6K.
/// When batch slots are used on single-card gfx12 (`batch_active`), co-prefill chunks are capped at 2,048 tokens, so raising the staged CPU-share ceiling to 2,560
/// keeps 2,048-token co-prefill pieces on the staged MMQ + CPU share path (`STAGE_GRP = 32` slots) instead of streaming all non-resident experts with the CPU pool idle.
constexpr int64_t kShareLimitGfx12Fused = 1536;
constexpr int64_t kShareLimitGfx12Batch = 2560;
constexpr int64_t gfx12_share_limit(bool fused_iq2s, bool batch_active = false) {
    return batch_active ? kShareLimitGfx12Batch : (fused_iq2s ? kShareLimitGfx12Fused : kShareLimitNew);
}

/// resident_slots: the expert cache's live slots; total_pairs: layers x routed experts per layer (<= 0: unknown, arm).
constexpr bool default_share_has_work(int64_t resident_slots, int64_t total_pairs) {
    return total_pairs <= 0 || resident_slots < total_pairs;
}

/// Maximum routed tokens per expert eligible for the CPU ExpertPool during prefill (STRATA_PREFILL_CPU_SHARE_MAXT).
/// Experts with `c > base_maxt` are sliced into `ceil(c / base_maxt)` ExpertJobMulti jobs of up to `base_maxt` tokens each.
inline int32_t prefill_cpu_share_max_t_from(const char* v, int32_t base_maxt, bool multi_slice_default) {
    if (v != nullptr && v[0] != '\0') return std::max<int32_t>(base_maxt, std::atoi(v));
    return multi_slice_default ? 4 * base_maxt : base_maxt;
}

/// Chunk-size bucket (`0..kCpuShareBuckets-1`) for isolating adaptive CPU-share EMAs and gate ratios across prompt scales.
constexpr int kCpuShareBuckets = 4;
constexpr int kCpuShareRatios = 5;
constexpr int cpu_share_bucket(int64_t T) {
    return T < 128 ? 0 : (T < 512 ? 1 : (T < 1536 ? 2 : 3));
}

struct CpuShareBucket {
    double c_ms = 0, g_ms = 0, share_now = 0.5;
    int64_t layers = 0, first_l = -1;
    int last_arm = 0, last_l = 0;
    double last_w = 0;
    double ratio[kCpuShareRatios] = {};
    int nratio = 0;
    bool gate = false;
};

/// Whether the runtime should alternate/probe `cpu_arm = false` layers to test if CPU share regresses wall time.
/// On single-card gfx12 (`single_card_gfx12 == true`), CPU share is always a net win, so off-probing is disabled by
/// default unless `STRATA_PREFILL_CPU_GATE_PROBE=1` or `STRATA_DBG_CPU_GATE` is set.
inline bool cpu_share_gate_probe_from(const char* v, bool dbg_gate, bool single_card_gfx12) {
    if (v != nullptr && v[0] != '\0') return std::atoi(v) != 0;
    if (dbg_gate) return true;
    return !single_card_gfx12;
}

/// Decides whether layer call `i` (0-indexed within its chunk-size bucket) should enable CPU expert sharing (`cpu_arm`).
/// When `probe_gate` is false (default on single-card gfx12), only a cold layer (`i == 0 && !buffers_warm`) skips CPU
/// share while buffers warm up; once buffers are pre-allocated (`buffers_warm == true`), even `i == 0` runs CPU share
/// and calibrates `c_ms`/`g_ms` immediately for `i == 1`.
constexpr bool cpu_share_should_run(int64_t i, int nratio, int min_ratios, int64_t probe_period,
                                    bool gate, bool probe_gate, bool buffers_warm = false) {
    if (!probe_gate) return gate && (i > 0 || buffers_warm);
    return nratio < min_ratios ? (i & 1) != 0 : (i % probe_period == 0) != gate;
}

/// Whether the CPU pool should take the next candidate non-resident expert (`next_jobs` slices of `<= MAXT` tokens).
/// When `cpu_c_ms > 0` (per job slice) and `cpu_g_ms > 0` (per GPU-streamed expert), balances total CPU job time against
/// 0.88 * remaining GPU/PCIe stream time (leaving 12% margin for H2D row DMA); reduces identically to
/// `cur_cpu_experts < llround((g / (c + g)) * nstream)` for 1-slice experts.
inline bool cpu_share_can_take_next(int64_t cur_jobs, int64_t next_jobs, int64_t nstream,
                                    int64_t cur_cpu_experts, double share,
                                    double cpu_c_ms = 0.0, double cpu_g_ms = 0.0) {
    if (nstream <= 0 || next_jobs <= 0 || cur_cpu_experts >= nstream) return false;
    if (next_jobs > 1 && cpu_c_ms > 0.0 && cpu_g_ms > 0.0) {
        const int64_t max_exp = (int64_t) std::llround(0.9 * (double) nstream);
        if (cur_cpu_experts >= max_exp) return false;
        const double lhs = ((double) cur_jobs + 0.5 * (double) next_jobs) * cpu_c_ms;
        const double rhs = 0.88 * ((double) (nstream - cur_cpu_experts) - 0.5) * cpu_g_ms;
        return lhs <= rhs;
    }
    const int64_t take = (int64_t) std::llround(share * (double) nstream);
    return cur_jobs + next_jobs <= take;
}


// ---- memory (0.1.41 report, RTX 5090, tight RAM: chats failed and the engine restarted until STRATA_PREFILL_CPU_SHARE=0) ----
// The default share reads experts with the CPU and keeps two pinned buffers on the host.  When the host has little
// memory left, or the experts it would read are file pages the RAM cannot keep (each is a page-in on a worker thread),
// the default is not armed.  An explicit STRATA_PREFILL_CPU_SHARE is the user's: it is armed and the engine only warns.
constexpr uint64_t kShareRamMargin = 3ull << 30;     // RAM kept free beside the share's own buffers and the file pages
constexpr uint64_t kShareCommitFloor = 4ull << 30;   // Windows commit kept free (the #1607 floor)
constexpr uint64_t kUnknown = ~uint64_t{0};

struct ShareMemory {
    uint64_t avail_ram = kUnknown;       ///< RAM this process can get (cgroup-aware); kUnknown: not known
    uint64_t avail_commit = kUnknown;    ///< Windows commit left (RAM + page file); kUnknown elsewhere
    uint64_t buffer_bytes = 0;           ///< host bytes the share itself allocates (pinned buffers, scratch)
    uint64_t inplace_bytes = 0;          ///< expert bytes the CPU would read as file pages (not in the GPU cache, not in RAM)
};

struct ShareVerdict {
    bool arm = true;
    std::string why;   ///< when !arm: what the engine says, in one line
};

inline ShareVerdict default_share_memory_verdict(const ShareMemory& m) {
    ShareVerdict v;
    const double gib = 1073741824.0;
    char buf[300];
    if (m.avail_ram != kUnknown && m.avail_ram < m.buffer_bytes + kShareRamMargin) {
        std::snprintf(buf, sizeof buf, "only %.1f GiB of RAM is free (the share needs %.2f GiB and %.1f GiB left over)",
                      (double) m.avail_ram / gib, (double) m.buffer_bytes / gib, (double) kShareRamMargin / gib);
        v.arm = false;
    } else if (m.avail_commit != kUnknown && m.avail_commit < m.buffer_bytes + kShareCommitFloor) {
        std::snprintf(buf, sizeof buf, "only %.1f GiB of the Windows commit limit is left (the share needs %.2f GiB and %.1f GiB left over)",
                      (double) m.avail_commit / gib, (double) m.buffer_bytes / gib, (double) kShareCommitFloor / gib);
        v.arm = false;
    } else if (m.inplace_bytes > 0 && m.avail_ram != kUnknown && m.inplace_bytes + kShareRamMargin > m.avail_ram) {
        std::snprintf(buf, sizeof buf, "the experts it would read are %.1f GiB of file pages and only %.1f GiB of RAM is free to keep them",
                      (double) m.inplace_bytes / gib, (double) m.avail_ram / gib);
        v.arm = false;
    }
    if (!v.arm) v.why = buf;
    return v;
}

}  // namespace strata::prefill::share_rules
