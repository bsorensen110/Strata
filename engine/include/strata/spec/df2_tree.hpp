// include/strata/spec/df2_tree.hpp - host-only DFlash2 tree-drafting topology, walk, and telemetry scaffolding.
//
// Inert in Phase 1: describes M <= 8 verification trees rooted at anchor node 0, builders for confidence-cut
// branching topologies, topological ordering/canonicalization, strict branch-kill longest-accepted-path commit walks,
// and [accept-hist] comma-list serialization without any device types or runtime wiring.
#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace strata::spec {

inline constexpr int kDf2TreeMaxNodes = 8;   ///< matches kVerifyNarrowT (anchor + up to 7 drafts)
inline constexpr int kDf2TreeMaxDrafts = 7;  ///< matches Dflash2Drafter::kDrafts
inline constexpr int8_t kDf2TreeNoParent = -1;
inline constexpr int kDf2TreeSelectorTopK = 16;
inline constexpr float kDf2TreeDefaultMinBranchP = 0.10f;

/// Parses STRATA_DF2_TREE value ("1" enables tree drafting; anything else keeps default off).
inline bool df2_tree_env_enabled(const char* value) noexcept {
    return value != nullptr && std::strcmp(value, "1") == 0;
}

/// Process-wide host-side runtime gate read once from STRATA_DF2_TREE (default OFF).
inline bool df2_tree_on() noexcept {
    static const bool on = df2_tree_env_enabled(std::getenv("STRATA_DF2_TREE"));
    return on;
}

enum class Df2TreeShape : uint8_t {
    Linear = 0,
    PrefixFanout = 1,
    PrefixFork2Tail = 2,
    Caterpillar = 3,
};

inline const char* df2_tree_shape_name(Df2TreeShape shape) noexcept {
    switch (shape) {
        case Df2TreeShape::Linear: return "linear";
        case Df2TreeShape::PrefixFanout: return "prefix_fanout";
        case Df2TreeShape::PrefixFork2Tail: return "prefix_fork2_tail";
        case Df2TreeShape::Caterpillar: return "caterpillar";
    }
    return "unknown";
}

/// Tree-8 speculative verification descriptor rooted at node 0 (the committed anchor token).
/// In canonical topological order, parent[0] == -1 and 0 <= parent[i] < i for 1 <= i < count.
struct Df2TreeSpec {
    int count = 0;
    std::array<int32_t, kDf2TreeMaxNodes> token{};
    std::array<int8_t, kDf2TreeMaxNodes> parent{
        kDf2TreeNoParent, kDf2TreeNoParent, kDf2TreeNoParent, kDf2TreeNoParent,
        kDf2TreeNoParent, kDf2TreeNoParent, kDf2TreeNoParent, kDf2TreeNoParent};
    std::array<uint8_t, kDf2TreeMaxNodes> depth{};
    /// Candidate rank within the parent's selector distribution (0 = primary pick, 1 = runner-up, ...).
    std::array<uint8_t, kDf2TreeMaxNodes> slot_rank{};
    /// anc[i][d] is the node index of i's ancestor at logical depth d (0 <= d <= depth[i]).
    std::array<std::array<uint8_t, kDf2TreeMaxNodes>, kDf2TreeMaxNodes> anc{};
    Df2TreeShape shape = Df2TreeShape::Linear;
    int8_t branch_node = -1;  ///< primary branch point (-1 if linear or multi-branch)

    bool empty() const noexcept { return count <= 0; }

    int max_depth() const noexcept {
        int md = 0;
        for (int i = 0; i < count && i < kDf2TreeMaxNodes; ++i) {
            md = std::max(md, static_cast<int>(depth[static_cast<size_t>(i)]));
        }
        return md;
    }

    bool is_topological() const noexcept {
        if (count < 1 || count > kDf2TreeMaxNodes) return false;
        if (parent[0] != kDf2TreeNoParent || depth[0] != 0) return false;
        for (int i = 1; i < count; ++i) {
            const int p = static_cast<int>(parent[static_cast<size_t>(i)]);
            if (p < 0 || p >= i) return false;
            if (depth[static_cast<size_t>(i)] != depth[static_cast<size_t>(p)] + 1) return false;
        }
        return true;
    }

    bool is_linear() const noexcept {
        if (!is_topological()) return false;
        for (int i = 1; i < count; ++i) {
            if (parent[static_cast<size_t>(i)] != i - 1) return false;
        }
        return true;
    }

    /// True when at most one node at each depth has children (so every runner-up branch is a leaf).
    bool is_caterpillar() const noexcept {
        if (!is_topological()) return false;
        std::array<uint8_t, kDf2TreeMaxNodes> child_count{};
        for (int i = 1; i < count; ++i) {
            ++child_count[static_cast<size_t>(parent[static_cast<size_t>(i)])];
        }
        std::array<uint8_t, kDf2TreeMaxNodes> non_leaf_at_depth{};
        for (int i = 0; i < count; ++i) {
            if (child_count[static_cast<size_t>(i)] > 0) {
                const uint8_t d = depth[static_cast<size_t>(i)];
                if (d >= kDf2TreeMaxNodes) return false;
                if (++non_leaf_at_depth[d] > 1) return false;
            }
        }
        return true;
    }

    /// Unique child token ids under every parent guarantee at most one child matches greedy argmax.
    bool has_unique_siblings() const noexcept {
        if (count < 1 || count > kDf2TreeMaxNodes) return false;
        for (int i = 1; i < count; ++i) {
            for (int j = i + 1; j < count; ++j) {
                if (parent[static_cast<size_t>(i)] == parent[static_cast<size_t>(j)] &&
                    token[static_cast<size_t>(i)] == token[static_cast<size_t>(j)]) {
                    return false;
                }
            }
        }
        return true;
    }

    bool valid() const noexcept { return is_topological() && has_unique_siblings(); }

    void build_anc() noexcept {
        anc = {};
        if (!is_topological()) return;
        for (int i = 0; i < count; ++i) {
            int cur = i;
            for (int d = static_cast<int>(depth[static_cast<size_t>(i)]); d >= 0; --d) {
                anc[static_cast<size_t>(i)][static_cast<size_t>(d)] = static_cast<uint8_t>(cur);
                cur = cur > 0 ? static_cast<int>(parent[static_cast<size_t>(cur)]) : -1;
            }
        }
    }
};

struct Df2TreeOrder {
    bool ok = false;
    int count = 0;
    std::array<uint8_t, kDf2TreeMaxNodes> order{};
};

/// Computes a valid topological ordering (parent before child, tie-breaking by lowest node index).
/// Accepts arbitrary node permutations as long as node 0 is the root and there are no cycles or dangling parents.
inline Df2TreeOrder df2_tree_topological_order(const Df2TreeSpec& tree) noexcept {
    Df2TreeOrder res;
    if (tree.count < 1 || tree.count > kDf2TreeMaxNodes) return res;
    if (tree.parent[0] != kDf2TreeNoParent) return res;

    std::array<uint8_t, kDf2TreeMaxNodes> indegree{};
    for (int i = 1; i < tree.count; ++i) {
        const int p = static_cast<int>(tree.parent[static_cast<size_t>(i)]);
        if (p < 0 || p >= tree.count || p == i) return res;
        indegree[static_cast<size_t>(i)] = 1;
    }

    std::array<uint8_t, kDf2TreeMaxNodes> emitted{};
    int out_n = 0;
    while (out_n < tree.count) {
        int pick = -1;
        for (int i = 0; i < tree.count; ++i) {
            if (!emitted[static_cast<size_t>(i)] && indegree[static_cast<size_t>(i)] == 0) {
                pick = i;
                break;
            }
        }
        if (pick < 0) return Df2TreeOrder{};  // cycle or disconnected component
        emitted[static_cast<size_t>(pick)] = 1;
        res.order[static_cast<size_t>(out_n++)] = static_cast<uint8_t>(pick);
        for (int c = 1; c < tree.count; ++c) {
            if (!emitted[static_cast<size_t>(c)] &&
                static_cast<int>(tree.parent[static_cast<size_t>(c)]) == pick) {
                indegree[static_cast<size_t>(c)] = 0;
            }
        }
    }
    res.ok = true;
    res.count = out_n;
    return res;
}

/// Reorders nodes into canonical topological order, recomputes depths and ancestor table, and validates uniqueness.
inline bool df2_tree_canonicalize(const Df2TreeSpec& in, Df2TreeSpec& out) noexcept {
    const Df2TreeOrder ord = df2_tree_topological_order(in);
    if (!ord.ok) return false;

    std::array<int8_t, kDf2TreeMaxNodes> remap{};
    remap.fill(-1);
    for (int i = 0; i < ord.count; ++i) {
        remap[ord.order[static_cast<size_t>(i)]] = static_cast<int8_t>(i);
    }

    Df2TreeSpec tmp;
    tmp.count = ord.count;
    tmp.shape = in.shape;
    tmp.branch_node = in.branch_node >= 0 && in.branch_node < in.count
                          ? remap[static_cast<size_t>(in.branch_node)]
                          : -1;
    for (int i = 0; i < ord.count; ++i) {
        const int src = static_cast<int>(ord.order[static_cast<size_t>(i)]);
        tmp.token[static_cast<size_t>(i)] = in.token[static_cast<size_t>(src)];
        tmp.slot_rank[static_cast<size_t>(i)] = in.slot_rank[static_cast<size_t>(src)];
        if (i == 0) {
            tmp.parent[0] = kDf2TreeNoParent;
            tmp.depth[0] = 0;
        } else {
            const int p_src = static_cast<int>(in.parent[static_cast<size_t>(src)]);
            const int8_t p_new = remap[static_cast<size_t>(p_src)];
            tmp.parent[static_cast<size_t>(i)] = p_new;
            tmp.depth[static_cast<size_t>(i)] = static_cast<uint8_t>(tmp.depth[static_cast<size_t>(p_new)] + 1);
        }
    }
    if (!tmp.valid()) return false;
    tmp.build_anc();
    out = tmp;
    return true;
}

/// Returns the --draft-conf product-gate cut depth in [0, n_drafts] matching window_rule.hpp::choose_window.
/// A return value of c means drafts[0..c-1] pass the product threshold and drafts[c] is the first uncertain draft.
inline int df2_tree_cut_depth(const float* probs, int n_drafts, double draft_conf) noexcept {
    const int n = std::clamp(n_drafts, 0, kDf2TreeMaxDrafts);
    if (draft_conf <= 0.0 || probs == nullptr) return n;
    int c = 0;
    double prod = 1.0;
    while (c < n && (prod *= static_cast<double>(probs[c])) >= draft_conf) {
        ++c;
    }
    return c;
}

/// Builds a linear chain of up to 7 drafts under `anchor` (total <= 8 nodes).
inline Df2TreeSpec make_linear_tree(int32_t anchor, const int32_t* drafts, int n_drafts) noexcept {
    Df2TreeSpec t;
    t.count = 1;
    t.token[0] = anchor;
    t.parent[0] = kDf2TreeNoParent;
    t.depth[0] = 0;
    t.slot_rank[0] = 0;
    t.shape = Df2TreeShape::Linear;
    t.branch_node = -1;
    const int n = (drafts != nullptr) ? std::clamp(n_drafts, 0, kDf2TreeMaxDrafts) : 0;
    for (int i = 0; i < n; ++i) {
        const int idx = t.count++;
        t.token[static_cast<size_t>(idx)] = drafts[i];
        t.parent[static_cast<size_t>(idx)] = static_cast<int8_t>(idx - 1);
        t.depth[static_cast<size_t>(idx)] = static_cast<uint8_t>(idx);
        t.slot_rank[static_cast<size_t>(idx)] = 0;
    }
    t.build_anc();
    return t;
}

/// Builds Shape A (PrefixFanout):
///   - linear prefix of `prefix_len` drafts (`0 -> 1 -> ... -> prefix_len`)
///   - `fanout`-way branch at node `prefix_len` using distinct tokens from `branch_tokens[0..fanout-1]`
///     (where `branch_tokens[0]` is the primary continuation and `1..fanout-1` are runner-ups)
///   - optional linear tail `primary_tail[0..primary_tail_len-1]` off `branch_tokens[0]` while nodes < max_nodes.
/// Resulting tree is always a caterpillar (0 GDN checkpoint memory).
inline Df2TreeSpec make_prefix_fanout_tree(int32_t anchor, const int32_t* prefix_drafts, int prefix_len,
                                           const int32_t* branch_tokens, int fanout,
                                           const int32_t* primary_tail = nullptr, int primary_tail_len = 0,
                                           int max_nodes = kDf2TreeMaxNodes) noexcept {
    const int node_cap = std::clamp(max_nodes, 1, kDf2TreeMaxNodes);
    Df2TreeSpec t;
    t.count = 1;
    t.token[0] = anchor;
    t.parent[0] = kDf2TreeNoParent;
    t.depth[0] = 0;
    t.slot_rank[0] = 0;
    t.shape = Df2TreeShape::PrefixFanout;

    const int p_len = (prefix_drafts != nullptr) ? std::clamp(prefix_len, 0, std::max(0, node_cap - 2)) : 0;
    for (int i = 0; i < p_len && t.count < node_cap; ++i) {
        const int idx = t.count++;
        t.token[static_cast<size_t>(idx)] = prefix_drafts[i];
        t.parent[static_cast<size_t>(idx)] = static_cast<int8_t>(idx - 1);
        t.depth[static_cast<size_t>(idx)] = static_cast<uint8_t>(idx);
        t.slot_rank[static_cast<size_t>(idx)] = 0;
    }

    const int branch_parent = t.count - 1;
    const uint8_t branch_depth = static_cast<uint8_t>(t.depth[static_cast<size_t>(branch_parent)] + 1);
    int primary_child = -1;
    int added_branches = 0;
    if (branch_tokens != nullptr && fanout > 0) {
        const int f_cap = std::min(fanout, node_cap - t.count);
        for (int b = 0; b < f_cap; ++b) {
            const int32_t tok = branch_tokens[b];
            bool dup = false;
            for (int k = branch_parent + 1; k < t.count; ++k) {
                if (t.parent[static_cast<size_t>(k)] == branch_parent &&
                    t.token[static_cast<size_t>(k)] == tok) {
                    dup = true;
                    break;
                }
            }
            if (dup) continue;
            const int idx = t.count++;
            t.token[static_cast<size_t>(idx)] = tok;
            t.parent[static_cast<size_t>(idx)] = static_cast<int8_t>(branch_parent);
            t.depth[static_cast<size_t>(idx)] = branch_depth;
            t.slot_rank[static_cast<size_t>(idx)] = static_cast<uint8_t>(b);
            if (primary_child < 0) primary_child = idx;
            ++added_branches;
        }
    }
    t.branch_node = added_branches > 1 ? static_cast<int8_t>(branch_parent) : -1;
    if (added_branches <= 1) t.shape = Df2TreeShape::Linear;

    if (primary_child >= 0 && primary_tail != nullptr && primary_tail_len > 0) {
        int cur_parent = primary_child;
        for (int i = 0; i < primary_tail_len && t.count < node_cap; ++i) {
            const int idx = t.count++;
            t.token[static_cast<size_t>(idx)] = primary_tail[i];
            t.parent[static_cast<size_t>(idx)] = static_cast<int8_t>(cur_parent);
            t.depth[static_cast<size_t>(idx)] = static_cast<uint8_t>(t.depth[static_cast<size_t>(cur_parent)] + 1);
            t.slot_rank[static_cast<size_t>(idx)] = 0;
            cur_parent = idx;
        }
    }
    t.build_anc();
    return t;
}

/// Builds a PrefixFanout caterpillar tree directly from DF2's primary path (`drafts`, `probs`) and
/// selector tables (`toptok[n_drafts][16]`, `seltab[n_drafts][16][16]`), fanning out 2..max_fanout
/// candidates at the confidence-cut depth when runner-up conditional probability >= min_branch_p.
inline Df2TreeSpec df2_tree_build_from_selector(int32_t anchor, const int32_t* drafts, const float* probs,
                                                int n_drafts, const int32_t* toptok, const float* seltab,
                                                double draft_conf,
                                                float min_branch_p = kDf2TreeDefaultMinBranchP,
                                                int max_fanout = 3,
                                                int max_nodes = kDf2TreeMaxNodes) noexcept {
    const int cap = std::clamp(max_nodes, 1, kDf2TreeMaxNodes);
    const int nd = (drafts != nullptr) ? std::clamp(n_drafts, 0, kDf2TreeMaxDrafts) : 0;
    const int lin_n = std::min(nd, cap - 1);
    if (nd < 2 || cap < 4 || drafts == nullptr || probs == nullptr || toptok == nullptr || seltab == nullptr) {
        return make_linear_tree(anchor, drafts, lin_n);
    }

    int cut = -1;
    if (draft_conf > 0.0) {
        cut = df2_tree_cut_depth(probs, nd, draft_conf);
        if (cut > cap - 3 || cut >= nd) {
            return make_linear_tree(anchor, drafts, lin_n);
        }
    } else {
        const int search_lim = std::min(nd, cap - 2);
        float best_p = probs[0];
        cut = 0;
        for (int i = 1; i < search_lim; ++i) {
            if (probs[i] < best_p) {
                best_p = probs[i];
                cut = i;
            }
        }
    }

    int prev_slot = 0;
    if (cut > 0) {
        prev_slot = -1;
        const int32_t* prev_toks = toptok + static_cast<size_t>(cut - 1) * kDf2TreeSelectorTopK;
        const int32_t prev_pick = drafts[cut - 1];
        for (int j = 0; j < kDf2TreeSelectorTopK; ++j) {
            if (prev_toks[j] == prev_pick) {
                prev_slot = j;
                break;
            }
        }
        if (prev_slot < 0) return make_linear_tree(anchor, drafts, lin_n);
    }

    const int32_t* row_toks = toptok + static_cast<size_t>(cut) * kDf2TreeSelectorTopK;
    const float* row_scores = seltab + (static_cast<size_t>(cut) * kDf2TreeSelectorTopK + static_cast<size_t>(prev_slot)) *
                                           kDf2TreeSelectorTopK;

    float max_score = -INFINITY;
    for (int j = 0; j < kDf2TreeSelectorTopK; ++j) {
        if (row_toks[j] >= 0 && std::isfinite(row_scores[j])) {
            max_score = std::max(max_score, row_scores[j]);
        }
    }
    if (!std::isfinite(max_score)) return make_linear_tree(anchor, drafts, lin_n);

    std::array<double, kDf2TreeSelectorTopK> weights{};
    double total = 0.0;
    for (int j = 0; j < kDf2TreeSelectorTopK; ++j) {
        if (row_toks[j] < 0 || !std::isfinite(row_scores[j])) continue;
        weights[static_cast<size_t>(j)] = std::exp(static_cast<double>(row_scores[j] - max_score));
        total += weights[static_cast<size_t>(j)];
    }
    if (!(total > 0.0) || !std::isfinite(total)) return make_linear_tree(anchor, drafts, lin_n);

    struct Cand {
        int slot;
        int32_t tok;
        float p;
    };
    std::array<Cand, kDf2TreeSelectorTopK> runners{};
    int n_runners = 0;
    const int32_t primary_tok = drafts[cut];
    for (int j = 0; j < kDf2TreeSelectorTopK; ++j) {
        const int32_t tok = row_toks[j];
        if (tok < 0 || tok == primary_tok) continue;
        const float p = static_cast<float>(weights[static_cast<size_t>(j)] / total);
        if (p < min_branch_p) continue;
        bool dup = false;
        for (int k = 0; k < n_runners; ++k) {
            if (runners[static_cast<size_t>(k)].tok == tok) {
                dup = true;
                break;
            }
        }
        if (!dup) runners[static_cast<size_t>(n_runners++)] = Cand{j, tok, p};
    }
    if (n_runners == 0) return make_linear_tree(anchor, drafts, lin_n);

    std::sort(runners.begin(), runners.begin() + n_runners, [](const Cand& a, const Cand& b) {
        if (a.p != b.p) return a.p > b.p;
        return a.slot < b.slot;
    });

    const int fan_budget = std::clamp(max_fanout, 2, std::min(kDf2TreeMaxNodes - (1 + cut), cap - (1 + cut)));
    std::array<int32_t, kDf2TreeMaxNodes> branch_tokens{};
    branch_tokens[0] = primary_tok;
    int fanout = 1;
    for (int k = 0; k < n_runners && fanout < fan_budget; ++k) {
        branch_tokens[static_cast<size_t>(fanout++)] = runners[static_cast<size_t>(k)].tok;
    }
    if (fanout < 2) return make_linear_tree(anchor, drafts, lin_n);

    const int tail_len = std::max(0, nd - (cut + 1));
    return make_prefix_fanout_tree(anchor, drafts, cut, branch_tokens.data(), fanout,
                                   tail_len > 0 ? drafts + cut + 1 : nullptr, tail_len, cap);
}

/// Builds Shape B (PrefixFork2Tail):
///   - linear prefix `0 -> ... -> prefix_len`
///   - 2-way fork at `prefix_len` (`primary_tok`, `runner_up_tok` if distinct)
///   - interleaved tails off both arms up to 8 total nodes (e.g., 3 prefix + 2-way fork + 1+1 tails = 8 nodes).
inline Df2TreeSpec make_prefix_fork2_tail_tree(int32_t anchor, const int32_t* prefix_drafts, int prefix_len,
                                               int32_t primary_tok, int32_t runner_up_tok,
                                               const int32_t* primary_tail, int primary_tail_len,
                                               const int32_t* runner_up_tail, int runner_up_tail_len) noexcept {
    Df2TreeSpec t;
    t.count = 1;
    t.token[0] = anchor;
    t.parent[0] = kDf2TreeNoParent;
    t.depth[0] = 0;
    t.slot_rank[0] = 0;

    const int p_len = (prefix_drafts != nullptr) ? std::clamp(prefix_len, 0, kDf2TreeMaxDrafts - 1) : 0;
    for (int i = 0; i < p_len && t.count < kDf2TreeMaxNodes; ++i) {
        const int idx = t.count++;
        t.token[static_cast<size_t>(idx)] = prefix_drafts[i];
        t.parent[static_cast<size_t>(idx)] = static_cast<int8_t>(idx - 1);
        t.depth[static_cast<size_t>(idx)] = static_cast<uint8_t>(idx);
        t.slot_rank[static_cast<size_t>(idx)] = 0;
    }

    const int fork_parent = t.count - 1;
    const uint8_t fork_depth = static_cast<uint8_t>(t.depth[static_cast<size_t>(fork_parent)] + 1);
    int arm_p = -1;
    int arm_r = -1;
    if (t.count < kDf2TreeMaxNodes) {
        arm_p = t.count++;
        t.token[static_cast<size_t>(arm_p)] = primary_tok;
        t.parent[static_cast<size_t>(arm_p)] = static_cast<int8_t>(fork_parent);
        t.depth[static_cast<size_t>(arm_p)] = fork_depth;
        t.slot_rank[static_cast<size_t>(arm_p)] = 0;
    }
    if (t.count < kDf2TreeMaxNodes && runner_up_tok != primary_tok) {
        arm_r = t.count++;
        t.token[static_cast<size_t>(arm_r)] = runner_up_tok;
        t.parent[static_cast<size_t>(arm_r)] = static_cast<int8_t>(fork_parent);
        t.depth[static_cast<size_t>(arm_r)] = fork_depth;
        t.slot_rank[static_cast<size_t>(arm_r)] = 1;
    }

    const int p_tail_n = (arm_p >= 0 && primary_tail != nullptr) ? std::max(0, primary_tail_len) : 0;
    const int r_tail_n = (arm_r >= 0 && runner_up_tail != nullptr) ? std::max(0, runner_up_tail_len) : 0;
    int cur_p = arm_p;
    int cur_r = arm_r;
    int ip = 0;
    int ir = 0;
    bool runner_has_tail = false;
    while (t.count < kDf2TreeMaxNodes && (ip < p_tail_n || ir < r_tail_n)) {
        if (ip < p_tail_n && t.count < kDf2TreeMaxNodes) {
            const int idx = t.count++;
            t.token[static_cast<size_t>(idx)] = primary_tail[ip++];
            t.parent[static_cast<size_t>(idx)] = static_cast<int8_t>(cur_p);
            t.depth[static_cast<size_t>(idx)] = static_cast<uint8_t>(t.depth[static_cast<size_t>(cur_p)] + 1);
            t.slot_rank[static_cast<size_t>(idx)] = 0;
            cur_p = idx;
        }
        if (ir < r_tail_n && t.count < kDf2TreeMaxNodes) {
            const int idx = t.count++;
            t.token[static_cast<size_t>(idx)] = runner_up_tail[ir++];
            t.parent[static_cast<size_t>(idx)] = static_cast<int8_t>(cur_r);
            t.depth[static_cast<size_t>(idx)] = static_cast<uint8_t>(t.depth[static_cast<size_t>(cur_r)] + 1);
            t.slot_rank[static_cast<size_t>(idx)] = 0;
            cur_r = idx;
            runner_has_tail = true;
        }
    }

    if (arm_r < 0) {
        t.shape = Df2TreeShape::Linear;
        t.branch_node = -1;
    } else {
        t.shape = runner_has_tail ? Df2TreeShape::PrefixFork2Tail : Df2TreeShape::PrefixFanout;
        t.branch_node = static_cast<int8_t>(fork_parent);
    }
    t.build_anc();
    return t;
}

/// Builds Shape C (Caterpillar):
///   - primary spine of `spine_len` drafts (`0 -> 1 -> ... -> spine_len`)
///   - for each `i` in `0..spine_len-1` where `branch_mask[i] != 0` and `runner_ups[i] != spine_drafts[i]`,
///     attaches a runner-up leaf under parent `i` while total nodes < 8.
inline Df2TreeSpec make_caterpillar_tree(int32_t anchor, const int32_t* spine_drafts, int spine_len,
                                         const int32_t* runner_ups, const uint8_t* branch_mask) noexcept {
    Df2TreeSpec t;
    t.count = 1;
    t.token[0] = anchor;
    t.parent[0] = kDf2TreeNoParent;
    t.depth[0] = 0;
    t.slot_rank[0] = 0;

    const int s_len = (spine_drafts != nullptr) ? std::clamp(spine_len, 0, kDf2TreeMaxDrafts) : 0;
    for (int i = 0; i < s_len && t.count < kDf2TreeMaxNodes; ++i) {
        const int idx = t.count++;
        t.token[static_cast<size_t>(idx)] = spine_drafts[i];
        t.parent[static_cast<size_t>(idx)] = static_cast<int8_t>(idx - 1);
        t.depth[static_cast<size_t>(idx)] = static_cast<uint8_t>(idx);
        t.slot_rank[static_cast<size_t>(idx)] = 0;
    }

    int branches = 0;
    int first_branch_parent = -1;
    if (runner_ups != nullptr && branch_mask != nullptr) {
        for (int i = 0; i < s_len && t.count < kDf2TreeMaxNodes; ++i) {
            if (branch_mask[i] == 0) continue;
            if (runner_ups[i] == spine_drafts[i]) continue;
            const int parent_idx = i;  // spine step i predicts child at depth i + 1 under node i
            const int idx = t.count++;
            t.token[static_cast<size_t>(idx)] = runner_ups[i];
            t.parent[static_cast<size_t>(idx)] = static_cast<int8_t>(parent_idx);
            t.depth[static_cast<size_t>(idx)] = static_cast<uint8_t>(t.depth[static_cast<size_t>(parent_idx)] + 1);
            t.slot_rank[static_cast<size_t>(idx)] = 1;
            if (first_branch_parent < 0) first_branch_parent = parent_idx;
            ++branches;
        }
    }

    if (branches == 0) {
        t.shape = Df2TreeShape::Linear;
        t.branch_node = -1;
    } else if (branches == 1) {
        t.shape = Df2TreeShape::PrefixFanout;
        t.branch_node = static_cast<int8_t>(first_branch_parent);
    } else {
        t.shape = Df2TreeShape::Caterpillar;
        t.branch_node = -1;
    }
    t.build_anc();
    return t;
}

struct Df2TreePath {
    std::array<uint8_t, kDf2TreeMaxNodes> nodes{};
    int length = 0;
};

/// Returns the inclusive path `[node, parent[node], ..., 0]` from `node` up to root `0`.
inline Df2TreePath df2_tree_path_to_root(const Df2TreeSpec& tree, int node) noexcept {
    Df2TreePath p;
    if (tree.count < 1 || tree.count > kDf2TreeMaxNodes || node < 0 || node >= tree.count) return p;
    int cur = node;
    for (int step = 0; step < kDf2TreeMaxNodes && cur >= 0 && cur < tree.count; ++step) {
        p.nodes[static_cast<size_t>(p.length++)] = static_cast<uint8_t>(cur);
        if (cur == 0) return p;
        cur = static_cast<int>(tree.parent[static_cast<size_t>(cur)]);
    }
    return Df2TreePath{};  // cycle or did not reach root 0
}

/// Returns the inclusive path `[0, ..., node]` from root `0` down to `node`.
inline Df2TreePath df2_tree_root_to_node(const Df2TreeSpec& tree, int node) noexcept {
    Df2TreePath rev = df2_tree_path_to_root(tree, node);
    if (rev.length <= 0) return rev;
    std::reverse(rev.nodes.begin(), rev.nodes.begin() + rev.length);
    return rev;
}

/// Returns the 3 preceding Conv1D taps `[t-3, t-2, t-1]` for `node` along its ancestor chain.
/// Negative indices (-3, -2, -1) denote the committed GDN Conv1D history slots preceding node 0.
inline std::array<int8_t, 3> df2_tree_conv_ancestors(const Df2TreeSpec& tree, int node) noexcept {
    std::array<int8_t, 3> taps{-3, -2, -1};
    const Df2TreePath path = df2_tree_path_to_root(tree, node);
    if (path.length <= 0) return taps;
    // path.nodes[0] is `node`, path.nodes[1] is t-1, path.nodes[2] is t-2, path.nodes[3] is t-3.
    for (int j = 0; j < 3; ++j) {
        const int dist = 3 - j;  // j=0 -> t-3, j=1 -> t-2, j=2 -> t-1
        if (dist < path.length) {
            taps[static_cast<size_t>(j)] = static_cast<int8_t>(path.nodes[static_cast<size_t>(dist)]);
        } else {
            // Slots before root 0: dist == path.length -> -1, dist == path.length+1 -> -2, etc.
            taps[static_cast<size_t>(j)] = static_cast<int8_t>(-(dist - path.length + 1));
        }
    }
    return taps;
}

/// Computes the caterpillar GDN execution order and spine flags for `pad_to` rows (`<= 8`).
/// Nodes `tree.count .. pad_to-1` are treated as dummy leaves at depth 1 under root 0.
/// At each depth `d = 0..max_depth`, all leaf rows (`is_spine == 0`) appear before the depth's spine row (`is_spine == 1`),
/// so the GDN recurrence kernel evaluates leaves read-only against `s[r]` before the spine step updates `s[r]`.
inline bool df2_tree_caterpillar_gdn_order(const Df2TreeSpec& tree, int32_t out_order[kDf2TreeMaxNodes],
                                           int32_t out_is_spine[kDf2TreeMaxNodes],
                                           int pad_to = kDf2TreeMaxNodes) noexcept {
    if (!tree.is_caterpillar() || out_order == nullptr || out_is_spine == nullptr) return false;
    const int n_pad = std::clamp(pad_to, tree.count, kDf2TreeMaxNodes);
    std::array<uint8_t, kDf2TreeMaxNodes> has_child{};
    for (int i = 1; i < tree.count; ++i) {
        has_child[static_cast<size_t>(tree.parent[static_cast<size_t>(i)])] = 1;
    }
    for (int i = 0; i < kDf2TreeMaxNodes; ++i) {
        out_is_spine[i] = (i < tree.count && has_child[static_cast<size_t>(i)] != 0) ? 1 : 0;
        out_order[i] = i;
    }
    const int max_d = std::max(tree.max_depth(), n_pad > tree.count ? 1 : 0);
    int out_idx = 0;
    for (int d = 0; d <= max_d; ++d) {
        int spine_at_d = -1;
        for (int i = 0; i < n_pad; ++i) {
            const int node_d = i < tree.count ? static_cast<int>(tree.depth[static_cast<size_t>(i)]) : 1;
            if (node_d != d) continue;
            if (out_is_spine[i] != 0) spine_at_d = i;
            else out_order[out_idx++] = i;
        }
        if (spine_at_d >= 0) out_order[out_idx++] = spine_at_d;
    }
    return out_idx == n_pad;
}

/// Returns the 3 history/normalized tap indices `[d-9, d-6, d-3]` along `node`'s ancestor chain for Layer-1 PLE Conv1D.
/// Negative values `(-9..-1)` index `history[9 + tap]`; non-negative values index `normalized[tap]`.
inline std::array<int32_t, 3> df2_tree_ple_conv_taps(const Df2TreeSpec& tree, int node) noexcept {
    std::array<int32_t, 3> taps{-9, -6, -3};
    if (!tree.is_topological() || node < 0 || node >= tree.count) return taps;
    const int d = static_cast<int>(tree.depth[static_cast<size_t>(node)]);
    for (int k = 0; k < 3; ++k) {
        const int dist = 9 - 3 * k;  // 9, 6, 3
        if (d >= dist) {
            taps[static_cast<size_t>(k)] = static_cast<int32_t>(tree.anc[static_cast<size_t>(node)][static_cast<size_t>(d - dist)]);
        } else {
            taps[static_cast<size_t>(k)] = static_cast<int32_t>(d - dist);
        }
    }
    return taps;
}

struct Df2TreeCommitWalk {
    std::array<uint8_t, kDf2TreeMaxNodes> path{};  ///< accepted physical node indices from root 0 to deepest kept node
    int n_keep = 0;                                ///< number of nodes to commit (1 = root anchor only)
    bool branch_hit = false;                       ///< true if any node on `path` had `slot_rank > 0` or `path[k] != k`

    int accepted_drafts() const noexcept { return n_keep > 0 ? n_keep - 1 : 0; }
    int deepest_node() const noexcept { return n_keep > 0 ? static_cast<int>(path[static_cast<size_t>(n_keep - 1)]) : -1; }
    bool needs_compaction() const noexcept {
        for (int k = 0; k < n_keep; ++k) {
            if (static_cast<int>(path[static_cast<size_t>(k)]) != k) return true;
        }
        return false;
    }
};

/// Computes the longest root-to-node path where every node on the path has `node_accepted[i] != 0`.
/// Enforces strict branch-kill: if a parent is rejected, none of its descendants can be kept.
/// Tie-breaks equal-depth accepted paths by lowest node index (in topological traversal order).
inline Df2TreeCommitWalk df2_tree_longest_accepted_path(const Df2TreeSpec& tree, const uint8_t* node_accepted) noexcept {
    Df2TreeCommitWalk walk;
    if (node_accepted == nullptr) return walk;
    const Df2TreeOrder ord = df2_tree_topological_order(tree);
    if (!ord.ok || ord.count <= 0) return walk;
    if (node_accepted[0] == 0) return walk;

    std::array<uint8_t, kDf2TreeMaxNodes> reachable_ok{};
    std::array<uint8_t, kDf2TreeMaxNodes> depth_of{};
    reachable_ok[0] = 1;
    depth_of[0] = 0;
    int best_node = 0;
    int best_depth = 0;

    for (int k = 1; k < ord.count; ++k) {
        const int u = static_cast<int>(ord.order[static_cast<size_t>(k)]);
        const int p = static_cast<int>(tree.parent[static_cast<size_t>(u)]);
        if (p >= 0 && reachable_ok[static_cast<size_t>(p)] != 0 && node_accepted[u] != 0) {
            reachable_ok[static_cast<size_t>(u)] = 1;
            const int d = static_cast<int>(depth_of[static_cast<size_t>(p)]) + 1;
            depth_of[static_cast<size_t>(u)] = static_cast<uint8_t>(d);
            if (d > best_depth || (d == best_depth && u < best_node)) {
                best_depth = d;
                best_node = u;
            }
        }
    }

    const Df2TreePath r2n = df2_tree_root_to_node(tree, best_node);
    walk.n_keep = r2n.length;
    for (int k = 0; k < r2n.length; ++k) {
        const uint8_t node = r2n.nodes[static_cast<size_t>(k)];
        walk.path[static_cast<size_t>(k)] = node;
        if (static_cast<int>(node) != k || tree.slot_rank[static_cast<size_t>(node)] > 0) {
            walk.branch_hit = true;
        }
    }
    return walk;
}

/// Exact greedy tree acceptance walk: starts at root `0` and repeatedly follows the unique child whose token
/// equals `verifier_argmax[u]`. Optionally returns the next token (`verifier_argmax[deepest_node]`).
inline Df2TreeCommitWalk df2_tree_greedy_accept(const Df2TreeSpec& tree, const int32_t* verifier_argmax,
                                                int32_t* bonus_token = nullptr) noexcept {
    Df2TreeCommitWalk walk;
    if (verifier_argmax == nullptr || !tree.valid()) return walk;

    int u = 0;
    walk.path[0] = 0;
    walk.n_keep = 1;
    while (walk.n_keep < kDf2TreeMaxNodes) {
        const int32_t target = verifier_argmax[u];
        int hit = -1;
        for (int c = u + 1; c < tree.count; ++c) {
            if (static_cast<int>(tree.parent[static_cast<size_t>(c)]) == u &&
                tree.token[static_cast<size_t>(c)] == target) {
                hit = c;
                break;
            }
        }
        if (hit < 0) break;
        const int k = walk.n_keep++;
        walk.path[static_cast<size_t>(k)] = static_cast<uint8_t>(hit);
        if (hit != k || tree.slot_rank[static_cast<size_t>(hit)] > 0) {
            walk.branch_hit = true;
        }
        u = hit;
    }
    if (bonus_token != nullptr) *bonus_token = verifier_argmax[u];
    return walk;
}

/// Tree telemetry compatible with [accept-hist] comma-list logging.
struct Df2TreeAcceptHist {
    std::array<int64_t, kDf2TreeMaxNodes> node_reached{};
    std::array<int64_t, kDf2TreeMaxNodes> node_accepted{};
    int64_t tree_rounds = 0;
    int64_t branch_hits = 0;
    int64_t total_accepted = 0;

    /// Records one verified tree round. Node `i` is counted as reached iff `i == 0` or its parent was accepted
    /// along `walk.path` (conditional reachability with branch-kill).
    void observe(const Df2TreeSpec& tree, const Df2TreeCommitWalk& walk) noexcept {
        if (!tree.valid() || walk.n_keep <= 0) return;
        ++tree_rounds;
        if (walk.branch_hit) ++branch_hits;
        total_accepted += walk.accepted_drafts();

        std::array<uint8_t, kDf2TreeMaxNodes> on_path{};
        for (int k = 0; k < walk.n_keep; ++k) {
            on_path[walk.path[static_cast<size_t>(k)]] = 1;
        }

        // Root is always evaluated/reached; a child i > 0 is reached iff its parent was accepted
        // (i.e. parent is on the accepted path and is not the rejected leaf where verification stopped,
        // OR wait: the deepest accepted node STILL predicted its children via verifier_argmax[deepest],
        // so all children of every node in walk.path[0..n_keep-1] were reached/tested!).
        ++node_reached[0];
        if (on_path[0]) ++node_accepted[0];
        for (int i = 1; i < tree.count; ++i) {
            const int p = static_cast<int>(tree.parent[static_cast<size_t>(i)]);
            if (p >= 0 && on_path[static_cast<size_t>(p)] != 0) {
                ++node_reached[static_cast<size_t>(i)];
                if (on_path[static_cast<size_t>(i)] != 0) {
                    ++node_accepted[static_cast<size_t>(i)];
                }
            }
        }
    }

    /// Formats optional suffix fields to append to [accept-hist] while preserving comma-list format.
    std::string format_accept_hist_fields() const {
        std::string s = "tree_rounds=" + std::to_string(tree_rounds) +
                        " tree_branch_hits=" + std::to_string(branch_hits) +
                        " tree_node_reached=";
        for (int i = 0; i < kDf2TreeMaxNodes; ++i) {
            if (i > 0) s += ',';
            s += std::to_string(node_reached[static_cast<size_t>(i)]);
        }
        s += " tree_node_accept=";
        for (int i = 0; i < kDf2TreeMaxNodes; ++i) {
            if (i > 0) s += ',';
            s += std::to_string(node_accepted[static_cast<size_t>(i)]);
        }
        return s;
    }
};

namespace detail {

inline bool parse_i64_token(const char*& p, int64_t& out) noexcept {
    if (p == nullptr) return false;
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == '\0') return false;
    char* end = nullptr;
    const long long val = std::strtoll(p, &end, 10);
    if (end == p) return false;
    out = static_cast<int64_t>(val);
    p = end;
    return true;
}

template <size_t N>
inline bool parse_csv_ints(const std::string& line, const char* key, std::array<int64_t, N>& out, int expected_n) noexcept {
    const size_t pos = line.find(key);
    if (pos == std::string::npos) return false;
    const char* p = line.c_str() + pos + std::strlen(key);
    for (int i = 0; i < expected_n; ++i) {
        if (i > 0) {
            if (*p != ',') return false;
            ++p;
        }
        int64_t v = 0;
        if (!parse_i64_token(p, v)) return false;
        out[static_cast<size_t>(i)] = v;
    }
    return *p == '\0' || *p == ' ' || *p == '\t' || *p == '\n' || *p == '\r';
}

inline bool parse_kv_i64(const std::string& line, const char* key, int64_t& out) noexcept {
    const size_t pos = line.find(key);
    if (pos == std::string::npos) return false;
    const char* p = line.c_str() + pos + std::strlen(key);
    if (!parse_i64_token(p, out)) return false;
    return *p == '\0' || *p == ' ' || *p == '\t' || *p == '\n' || *p == '\r';
}

}  // namespace detail

/// Parses `tree_rounds=... tree_branch_hits=... tree_node_reached=... tree_node_accept=...`
/// either standalone or appended to a full `[accept-hist]` log line.
inline bool parse_accept_hist_fields(const std::string& line, Df2TreeAcceptHist& out) noexcept {
    Df2TreeAcceptHist tmp;
    if (!detail::parse_kv_i64(line, "tree_rounds=", tmp.tree_rounds)) return false;
    if (!detail::parse_kv_i64(line, "tree_branch_hits=", tmp.branch_hits)) return false;
    if (!detail::parse_csv_ints(line, "tree_node_reached=", tmp.node_reached, kDf2TreeMaxNodes)) return false;
    if (!detail::parse_csv_ints(line, "tree_node_accept=", tmp.node_accepted, kDf2TreeMaxNodes)) return false;
    out = tmp;
    return true;
}

/// Serializes a Df2TreeSpec into a compact comma-list key=value string for telemetry/tests.
inline std::string serialize_tree_spec(const Df2TreeSpec& tree) {
    if (tree.count <= 0 || tree.count > kDf2TreeMaxNodes) return "";
    std::string s = "shape=" + std::to_string(static_cast<int>(tree.shape)) +
                    " count=" + std::to_string(tree.count) +
                    " branch=" + std::to_string(static_cast<int>(tree.branch_node)) +
                    " tok=";
    for (int i = 0; i < tree.count; ++i) {
        if (i > 0) s += ',';
        s += std::to_string(tree.token[static_cast<size_t>(i)]);
    }
    s += " par=";
    for (int i = 0; i < tree.count; ++i) {
        if (i > 0) s += ',';
        s += std::to_string(static_cast<int>(tree.parent[static_cast<size_t>(i)]));
    }
    s += " rank=";
    for (int i = 0; i < tree.count; ++i) {
        if (i > 0) s += ',';
        s += std::to_string(static_cast<int>(tree.slot_rank[static_cast<size_t>(i)]));
    }
    return s;
}

/// Parses a string produced by `serialize_tree_spec`, rebuilding depths and ancestor tables and validating.
inline bool parse_tree_spec(const std::string& text, Df2TreeSpec& out) noexcept {
    int64_t shape_v = 0, count_v = 0, branch_v = -1;
    if (!detail::parse_kv_i64(text, "shape=", shape_v)) return false;
    if (!detail::parse_kv_i64(text, "count=", count_v)) return false;
    if (!detail::parse_kv_i64(text, "branch=", branch_v)) return false;
    if (shape_v < 0 || shape_v > static_cast<int64_t>(Df2TreeShape::Caterpillar)) return false;
    if (count_v < 1 || count_v > kDf2TreeMaxNodes) return false;
    if (branch_v < -1 || branch_v >= count_v) return false;

    const int n = static_cast<int>(count_v);
    std::array<int64_t, kDf2TreeMaxNodes> toks{}, pars{}, ranks{};
    if (!detail::parse_csv_ints(text, "tok=", toks, n)) return false;
    if (!detail::parse_csv_ints(text, "par=", pars, n)) return false;
    if (!detail::parse_csv_ints(text, "rank=", ranks, n)) return false;

    Df2TreeSpec tmp;
    tmp.count = n;
    tmp.shape = static_cast<Df2TreeShape>(shape_v);
    tmp.branch_node = static_cast<int8_t>(branch_v);
    for (int i = 0; i < n; ++i) {
        if (pars[static_cast<size_t>(i)] < -1 || pars[static_cast<size_t>(i)] >= n) return false;
        if (ranks[static_cast<size_t>(i)] < 0 || ranks[static_cast<size_t>(i)] > 255) return false;
        tmp.token[static_cast<size_t>(i)] = static_cast<int32_t>(toks[static_cast<size_t>(i)]);
        tmp.parent[static_cast<size_t>(i)] = static_cast<int8_t>(pars[static_cast<size_t>(i)]);
        tmp.slot_rank[static_cast<size_t>(i)] = static_cast<uint8_t>(ranks[static_cast<size_t>(i)]);
    }
    return df2_tree_canonicalize(tmp, out);
}

}  // namespace strata::spec
