// src/spec/df2_tree_test.cpp - unit tests for host-only DFlash2 tree scaffolding (spec/df2_tree.hpp).
#include "strata/spec/df2_tree.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>

namespace {

int g_failures = 0;

void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++g_failures;
    }
}

using namespace strata::spec;

void test_cut_depth() {
    const float probs[7] = {0.95f, 0.90f, 0.85f, 0.50f, 0.80f, 0.90f, 0.90f};
    check(df2_tree_cut_depth(probs, 7, 0.0) == 7, "disabled draft_conf keeps all 7 drafts");
    check(df2_tree_cut_depth(nullptr, 7, 0.5) == 7, "null probs returns clamped count");
    // 0.95 >= 0.5; 0.95*0.90 = 0.855 >= 0.5; *0.85 = 0.72675 >= 0.5; *0.50 = 0.363375 < 0.5 -> cut at 3
    check(df2_tree_cut_depth(probs, 7, 0.5) == 3, "draft_conf=0.5 cuts at index 3");
    check(df2_tree_cut_depth(probs, 7, 0.98) == 0, "draft_conf=0.98 cuts at index 0");
    check(df2_tree_cut_depth(probs, 4, 0.1) == 4, "loose draft_conf keeps all provided drafts");
}

void test_builders_and_invariants() {
    // 1. Linear builder + clamping past 7 drafts
    const int32_t lin_drafts[9] = {10, 11, 12, 13, 14, 15, 16, 17, 18};
    const Df2TreeSpec lin = make_linear_tree(100, lin_drafts, 9);
    check(lin.count == 8, "linear tree clamps to 8 nodes");
    check(lin.valid() && lin.is_linear() && lin.is_caterpillar(), "linear tree satisfies validity invariants");
    check(lin.max_depth() == 7 && lin.branch_node == -1, "linear tree has depth 7 and no branch node");
    check(lin.token[0] == 100 && lin.token[7] == 16, "linear tree preserves anchor and first 7 drafts");

    // 2. PrefixFanout (p=3 prefix, 2-way branch at node 3, + 2 primary tail = 8 nodes)
    const int32_t prefix[3] = {10, 11, 12};
    const int32_t br2[3] = {20, 21, 22};
    const int32_t ptail[4] = {30, 31, 32, 33};
    const Df2TreeSpec pf = make_prefix_fanout_tree(1, prefix, 3, br2, 2, ptail, 4);
    // Nodes: 0(1)->1(10)->2(11)->3(12); 3->4(20), 3->5(21); 4->6(30)->7(31)
    check(pf.count == 8, "prefix_fanout fills 8 nodes");
    check(pf.valid() && !pf.is_linear() && pf.is_caterpillar(), "prefix_fanout with primary tail is a caterpillar");
    check(pf.branch_node == 3 && pf.shape == Df2TreeShape::PrefixFanout, "branch_node and shape recorded");
    check(pf.parent[4] == 3 && pf.parent[5] == 3 && pf.slot_rank[5] == 1, "branch siblings attach to node 3");
    check(pf.parent[6] == 4 && pf.parent[7] == 6 && pf.depth[7] == 6, "primary tail chains under primary child");
    check(pf.anc[7][0] == 0 && pf.anc[7][3] == 3 && pf.anc[7][4] == 4 && pf.anc[7][6] == 7,
          "anc table matches root-to-node 7");
    check(pf.anc[5][4] == 5 && pf.anc[5][3] == 3, "anc table matches root-to-runner-up 5");

    // 3. PrefixFanout deduplicates identical sibling tokens
    const int32_t dup_br[3] = {20, 20, 21};
    const Df2TreeSpec pf_dedup = make_prefix_fanout_tree(1, prefix, 3, dup_br, 3, nullptr, 0);
    check(pf_dedup.count == 6 && pf_dedup.has_unique_siblings(), "duplicate sibling token is deduplicated");
    check(pf_dedup.token[4] == 20 && pf_dedup.token[5] == 21, "deduplication keeps primary and distinct runner-up");

    // 4. PrefixFork2Tail (3+2x2: 8 nodes, non-caterpillar)
    const int32_t rtail[2] = {40, 41};
    const Df2TreeSpec f2 = make_prefix_fork2_tail_tree(1, prefix, 3, 20, 21, ptail, 2, rtail, 2);
    // Nodes: 0,1,2,3 (prefix), 4(20,p=3), 5(21,p=3), 6(30,p=4), 7(40,p=5)
    check(f2.count == 8 && f2.valid() && !f2.is_caterpillar(), "prefix_fork2_tail is valid and non-caterpillar");
    check(f2.shape == Df2TreeShape::PrefixFork2Tail && f2.branch_node == 3, "fork2_tail shape and branch_node");
    check(f2.parent[6] == 4 && f2.parent[7] == 5 && f2.max_depth() == 5, "both arms extend to depth 5");

    // 5. Multi-point Caterpillar
    const int32_t spine[5] = {10, 11, 12, 13, 14};
    const int32_t rups[5] = {90, 91, 92, 93, 94};
    const uint8_t mask[5] = {0, 1, 0, 1, 0};
    const Df2TreeSpec cat = make_caterpillar_tree(1, spine, 5, rups, mask);
    // Nodes: 0..5 spine, 6(91, parent=1, depth=2), 7(93, parent=3, depth=4)
    check(cat.count == 8 && cat.valid() && cat.is_caterpillar(), "multi-point caterpillar valid");
    check(cat.shape == Df2TreeShape::Caterpillar && cat.branch_node == -1, "caterpillar shape");
    check(cat.parent[6] == 1 && cat.depth[6] == 2 && cat.parent[7] == 3 && cat.depth[7] == 4,
          "caterpillar runner-up leaves attach at masked spine nodes");
}

void test_paths_and_conv_ancestors() {
    const int32_t prefix[2] = {10, 11};
    const int32_t ptail[2] = {30, 31};
    const int32_t rtail[1] = {40};
    const Df2TreeSpec t = make_prefix_fork2_tail_tree(5, prefix, 2, 20, 21, ptail, 2, rtail, 1);
    // 0(5)->1(10)->2(11); 2->3(20), 2->4(21); 3->5(30), 4->6(40), 5->7(31)
    const Df2TreePath p6_root = df2_tree_path_to_root(t, 6);
    check(p6_root.length == 5 && p6_root.nodes[0] == 6 && p6_root.nodes[1] == 4 &&
              p6_root.nodes[2] == 2 && p6_root.nodes[3] == 1 && p6_root.nodes[4] == 0,
          "path_to_root on runner-up arm");
    const Df2TreePath r2n6 = df2_tree_root_to_node(t, 6);
    check(r2n6.length == 5 && r2n6.nodes[0] == 0 && r2n6.nodes[4] == 6, "root_to_node reverses path_to_root");
    check(df2_tree_path_to_root(t, -1).length == 0 && df2_tree_path_to_root(t, 8).length == 0,
          "out-of-range node returns empty path");

    const auto c0 = df2_tree_conv_ancestors(t, 0);
    const auto c1 = df2_tree_conv_ancestors(t, 1);
    const auto c4 = df2_tree_conv_ancestors(t, 4);
    const auto c6 = df2_tree_conv_ancestors(t, 6);
    check((c0 == std::array<int8_t, 3>{-3, -2, -1}), "root conv ancestors read 3 history slots");
    check((c1 == std::array<int8_t, 3>{-2, -1, 0}), "depth-1 conv ancestors read 2 history + root");
    check((c4 == std::array<int8_t, 3>{0, 1, 2}), "runner-up node 4 conv ancestors are [0,1,2]");
    check((c6 == std::array<int8_t, 3>{1, 2, 4}), "runner-up tail node 6 conv ancestors skip primary node 3");
}

void test_topological_order_and_canonicalize() {
    // Valid non-topological tree: root=0, node 1's parent is 3, node 3's parent is 2, node 2's parent is 0.
    Df2TreeSpec scrambled;
    scrambled.count = 5;
    scrambled.shape = Df2TreeShape::PrefixFanout;
    scrambled.branch_node = 2;
    scrambled.token = {100, 103, 101, 102, 104};
    scrambled.parent = {-1, 3, 0, 2, 2};
    scrambled.slot_rank = {0, 0, 0, 0, 1};
    check(!scrambled.is_topological(), "scrambled tree is not topological");

    const Df2TreeOrder ord = df2_tree_topological_order(scrambled);
    check(ord.ok && ord.count == 5, "topological sort succeeds on scrambled valid tree");
    check(ord.order[0] == 0 && ord.order[1] == 2 && ord.order[2] == 3 && ord.order[3] == 1 && ord.order[4] == 4,
          "topological order breaks ties by lowest index");

    Df2TreeSpec canon;
    check(df2_tree_canonicalize(scrambled, canon) && canon.valid(), "canonicalize produces valid topological tree");
    check(canon.token[0] == 100 && canon.token[1] == 101 && canon.token[2] == 102 &&
              canon.token[3] == 103 && canon.token[4] == 104,
          "canonicalized tokens follow topological order");
    check(canon.branch_node == 1, "canonicalized branch_node remapped from 2 to 1");

    // Invalid topologies
    Df2TreeSpec cyc = scrambled;
    cyc.parent[2] = 1;  // cycle 1 -> 3 -> 2 -> 1
    check(!df2_tree_topological_order(cyc).ok, "cycle rejected");

    Df2TreeSpec self_p = scrambled;
    self_p.parent[4] = 4;
    check(!df2_tree_topological_order(self_p).ok, "self-parent rejected");

    Df2TreeSpec bad_root = scrambled;
    bad_root.parent[0] = 1;
    check(!df2_tree_topological_order(bad_root).ok, "non-root node 0 rejected");
}

void test_acceptance_walks_and_branch_kill() {
    // Fork2Tail: 0(1)->1(10)->2(11)->3(12); 3->4(20), 3->5(21); 4->6(30), 5->7(40)
    const int32_t prefix[3] = {10, 11, 12};
    const int32_t ptail[1] = {30};
    const int32_t rtail[1] = {40};
    const Df2TreeSpec t = make_prefix_fork2_tail_tree(1, prefix, 3, 20, 21, ptail, 1, rtail, 1);

    // Case 1: Root only
    {
        const uint8_t acc[8] = {1, 0, 1, 1, 1, 1, 1, 1};
        const Df2TreeCommitWalk w = df2_tree_longest_accepted_path(t, acc);
        check(w.n_keep == 1 && w.path[0] == 0 && !w.branch_hit && !w.needs_compaction(),
              "branch-kill at node 1 stops walk at root despite accepted descendants");
    }

    // Case 2: Primary branch rejected at node 4, runner-up branch 5->7 accepted!
    {
        const uint8_t acc[8] = {1, 1, 1, 1, 0, 1, 1, 1};
        const Df2TreeCommitWalk w = df2_tree_longest_accepted_path(t, acc);
        check(w.n_keep == 6 && w.accepted_drafts() == 5, "runner-up arm commits 6 nodes (5 drafts)");
        check(w.path[0] == 0 && w.path[1] == 1 && w.path[2] == 2 && w.path[3] == 3 &&
                  w.path[4] == 5 && w.path[5] == 7,
              "commit path skips rejected primary nodes 4 and 6");
        check(w.branch_hit && w.needs_compaction(), "runner-up arm flags branch_hit and needs_compaction");
    }

    // Case 3: Tie-breaking when both arms fully accepted -> primary (lowest index 4,6) wins
    {
        const uint8_t acc[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        const Df2TreeCommitWalk w = df2_tree_longest_accepted_path(t, acc);
        check(w.n_keep == 6 && w.path[4] == 4 && w.path[5] == 6 && w.needs_compaction(),
              "equal-depth tie breaks to primary branch (4,6) and compacts over skipped node 5");
    }

    // Case 4: Root rejected -> n_keep == 0
    {
        const uint8_t acc[8] = {0, 1, 1, 1, 1, 1, 1, 1};
        const Df2TreeCommitWalk w = df2_tree_longest_accepted_path(t, acc);
        check(w.n_keep == 0, "rejected root yields n_keep=0");
    }

    // Case 5: Greedy accept walk with runner-up continuation + bonus token
    {
        // argmax[0]=10 (hits 1), argmax[1]=11 (hits 2), argmax[2]=12 (hits 3),
        // argmax[3]=21 (hits runner-up 5!), argmax[5]=40 (hits 7!), argmax[7]=999 (no child -> bonus=999)
        const int32_t argmax[8] = {10, 11, 12, 21, 999, 40, 999, 777};
        int32_t bonus = -1;
        const Df2TreeCommitWalk w = df2_tree_greedy_accept(t, argmax, &bonus);
        check(w.n_keep == 6 && w.path[4] == 5 && w.path[5] == 7 && bonus == 777,
              "greedy walk selects runner-up arm and returns bonus token from node 7");
    }

    // Case 6: Greedy accept immediate mismatch at root
    {
        const int32_t argmax[8] = {555, 11, 12, 20, 30, 40, 0, 0};
        int32_t bonus = -1;
        const Df2TreeCommitWalk w = df2_tree_greedy_accept(t, argmax, &bonus);
        check(w.n_keep == 1 && w.path[0] == 0 && bonus == 555 && !w.needs_compaction(),
              "greedy immediate mismatch commits root only and emits argmax[0]");
    }
}

void test_serialization_and_telemetry_roundtrip() {
    const int32_t prefix[3] = {10, 11, 12};
    const int32_t br[3] = {20, 21, 22};
    const int32_t ptail[1] = {30};
    const Df2TreeSpec orig = make_prefix_fanout_tree(42, prefix, 3, br, 3, ptail, 1);
    // 0(42)->1(10)->2(11)->3(12); 3->4(20), 3->5(21), 3->6(22); 4->7(30)
    const std::string encoded = serialize_tree_spec(orig);
    Df2TreeSpec decoded;
    check(parse_tree_spec(encoded, decoded), "serialize/parse tree spec round-trip succeeds");
    check(decoded.count == orig.count && decoded.shape == orig.shape && decoded.branch_node == orig.branch_node,
          "decoded spec preserves metadata");
    check(decoded.token == orig.token && decoded.parent == orig.parent &&
              decoded.depth == orig.depth && decoded.slot_rank == orig.slot_rank && decoded.anc == orig.anc,
          "decoded spec preserves tokens, parents, depths, ranks, and anc");

    Df2TreeSpec bad;
    check(!parse_tree_spec("shape=1 count=9 branch=-1 tok=1 par=-1 rank=0", bad), "rejects out-of-range count");
    check(!parse_tree_spec("shape=1 count=2 branch=-1 tok=1,2 par=0,0 rank=0,0", bad), "rejects non-root node 0");

    // Telemetry reachability & [accept-hist] comma-list compatibility
    Df2TreeAcceptHist hist;
    // Round 1: stopped at node 2 (nodes 0,1,2 accepted; node 3 reached as child of 2, rejected; nodes 4..7 NOT reached!)
    const int32_t arg1[8] = {10, 11, 999, 20, 30, 0, 0, 0};
    hist.observe(orig, df2_tree_greedy_accept(orig, arg1));
    // Round 2: runner-up node 5 accepted and stopped there (nodes 0,1,2,3,5 accepted; 4,5,6 reached; 7 NOT reached)
    const int32_t arg2[8] = {10, 11, 12, 21, 0, 888, 0, 0};
    hist.observe(orig, df2_tree_greedy_accept(orig, arg2));

    check(hist.tree_rounds == 2 && hist.branch_hits == 1 && hist.total_accepted == 6,
          "telemetry counts rounds, branch_hits, and total_accepted");
    check((hist.node_reached == std::array<int64_t, 8>{2, 2, 2, 2, 1, 1, 1, 0}),
          "branch-kill excludes nodes behind rejected ancestors from node_reached");
    check((hist.node_accepted == std::array<int64_t, 8>{2, 2, 2, 1, 0, 1, 0, 0}),
          "node_accepted records winning path nodes only");

    const std::string full_line =
        "[accept-hist] req=9 df2_pos_reached=3,3,1,0,0,0,0 df2_pos_accept=3,2,1,0,0,0,0 "
        "lookup_rows=0 lookup_accept=0 chain_rows=1 chain_accept=0 " +
        hist.format_accept_hist_fields();

    Df2TreeAcceptHist parsed;
    check(parse_accept_hist_fields(full_line, parsed), "parses tree fields embedded in [accept-hist] line");
    check(parsed.tree_rounds == hist.tree_rounds && parsed.branch_hits == hist.branch_hits &&
              parsed.node_reached == hist.node_reached && parsed.node_accepted == hist.node_accepted,
          "telemetry round-trip matches exactly");
}

void test_env_gate_and_selector_builder() {
    check(!df2_tree_env_enabled(nullptr), "null env keeps tree off");
    check(!df2_tree_env_enabled(""), "empty env keeps tree off");
    check(!df2_tree_env_enabled("0"), "0 env keeps tree off");
    check(!df2_tree_env_enabled("2"), "non-1 env keeps tree off");
    check(df2_tree_env_enabled("1"), "1 env enables tree");

    std::array<int32_t, 7> drafts{101, 102, 103, 104, 105, 106, 107};
    std::array<float, 7> probs{0.95f, 0.90f, 0.85f, 0.45f, 0.80f, 0.75f, 0.70f};
    std::array<int32_t, 7 * kDf2TreeSelectorTopK> toptok{};
    std::array<float, 7 * kDf2TreeSelectorTopK * kDf2TreeSelectorTopK> seltab{};
    toptok.fill(-1);
    seltab.fill(-100.0f);

    for (int r = 0; r < 7; ++r) {
        toptok[static_cast<size_t>(r) * kDf2TreeSelectorTopK + 0] = drafts[static_cast<size_t>(r)];
        toptok[static_cast<size_t>(r) * kDf2TreeSelectorTopK + 1] = 200 + r;
        toptok[static_cast<size_t>(r) * kDf2TreeSelectorTopK + 2] = 300 + r;
        seltab[(static_cast<size_t>(r) * kDf2TreeSelectorTopK + 0) * kDf2TreeSelectorTopK + 0] = 0.0f;
    }

    // At cut depth c=3 (0.95*0.9*0.85=0.72675 >= 0.5, *0.45 < 0.5), prev_slot=0 (drafts[2] at slot 0):
    // give runner-up 1 conditional p ~ 0.25 and runner-up 2 conditional p ~ 0.15 (>= 0.10).
    const size_t base3 = (3u * kDf2TreeSelectorTopK + 0u) * kDf2TreeSelectorTopK;
    seltab[base3 + 0] = 1.0f;    // exp(1.0) = 2.718
    seltab[base3 + 1] = 0.0f;    // exp(0.0) = 1.000 -> p ~ 0.22
    seltab[base3 + 2] = -0.3f;   // exp(-0.3) = 0.741 -> p ~ 0.166
    const Df2TreeSpec tree = df2_tree_build_from_selector(99, drafts.data(), probs.data(), 7,
                                                          toptok.data(), seltab.data(), 0.5, 0.10f, 3, 8);
    // Nodes: 0(99)->1(101)->2(102)->3(103); 3->4(104), 3->5(203), 3->6(303); 4->7(105)
    check(tree.count == 8 && tree.valid() && tree.is_caterpillar() && !tree.is_linear(),
          "selector builder produces 8-node PrefixFanout caterpillar");
    check(tree.shape == Df2TreeShape::PrefixFanout && tree.branch_node == 3, "branch_node at cut depth 3");
    check(tree.token[4] == 104 && tree.token[5] == 203 && tree.token[6] == 303 && tree.token[7] == 105,
          "selector builder keeps primary continuation, 2 runner-ups, and primary tail");
    check(tree.parent[4] == 3 && tree.parent[5] == 3 && tree.parent[6] == 3 && tree.parent[7] == 4,
          "selector builder attaches fanout and tail to right parents");

    // Below min_branch_p -> falls back to linear
    const Df2TreeSpec lin_fallback = df2_tree_build_from_selector(99, drafts.data(), probs.data(), 7,
                                                                  toptok.data(), seltab.data(), 0.5, 0.50f, 3, 8);
    check(lin_fallback.is_linear() && lin_fallback.count == 8,
          "high min_branch_p falls back to linear tree");

    // Cut depth >= 6 (no budget for 2-way fanout) -> linear
    const Df2TreeSpec late_cut = df2_tree_build_from_selector(99, drafts.data(), probs.data(), 7,
                                                              toptok.data(), seltab.data(), 0.01, 0.10f, 3, 8);
    check(late_cut.is_linear() && late_cut.count == 8, "late confidence cut falls back to linear");

    // draft_conf == 0.0 picks argmin probs[i] within cap-2 (index 3 here)
    const Df2TreeSpec no_conf = df2_tree_build_from_selector(99, drafts.data(), probs.data(), 7,
                                                             toptok.data(), seltab.data(), 0.0, 0.10f, 3, 8);
    check(no_conf.shape == Df2TreeShape::PrefixFanout && no_conf.branch_node == 3,
          "draft_conf=0 picks lowest-prob spine slot for branching");

    // Respects max_nodes < 8
    const Df2TreeSpec capped = df2_tree_build_from_selector(99, drafts.data(), probs.data(), 7,
                                                            toptok.data(), seltab.data(), 0.5, 0.10f, 3, 6);
    check(capped.count == 6 && capped.valid() && capped.is_caterpillar(), "respects max_nodes=6 cap");
}

void test_ev_budget_ordering_and_cut_drop() {
    check(df2_tree_parse_ev_cut(nullptr) == 0.0f, "null ev_cut defaults to 0");
    check(df2_tree_parse_ev_cut("") == 0.0f, "empty ev_cut defaults to 0");
    check(df2_tree_parse_ev_cut("abc") == 0.0f, "invalid ev_cut defaults to 0");
    check(df2_tree_parse_ev_cut("-0.1") == 0.0f, "negative ev_cut defaults to 0");
    check(df2_tree_parse_ev_cut("1.2") == 0.0f, "out-of-range ev_cut defaults to 0");
    check(df2_tree_parse_ev_cut("0.15") > 0.149f && df2_tree_parse_ev_cut("0.15") < 0.151f, "valid ev_cut parsed");

    std::array<int32_t, 7> drafts{101, 102, 103, 104, 105, 106, 107};
    // Survival: S1=0.90, S2=0.81, S3=0.729, S4=0.2916, S5=0.1166, S6=0.0350, S7=0.0105
    std::array<float, 7> probs{0.90f, 0.90f, 0.90f, 0.40f, 0.40f, 0.30f, 0.30f};
    std::array<int32_t, 7 * kDf2TreeSelectorTopK> toptok{};
    std::array<float, 7 * kDf2TreeSelectorTopK * kDf2TreeSelectorTopK> seltab{};
    toptok.fill(-1);
    seltab.fill(-100.0f);
    for (int r = 0; r < 7; ++r) {
        toptok[static_cast<size_t>(r) * kDf2TreeSelectorTopK + 0] = drafts[static_cast<size_t>(r)];
        toptok[static_cast<size_t>(r) * kDf2TreeSelectorTopK + 1] = 200 + r;
        toptok[static_cast<size_t>(r) * kDf2TreeSelectorTopK + 2] = 300 + r;
        seltab[(static_cast<size_t>(r) * kDf2TreeSelectorTopK + 0) * kDf2TreeSelectorTopK + 0] = 0.0f;
    }
    // Depth 2 runner-up (under node 2): S2*p_run = 0.81 * 0.25 = 0.2025
    const size_t base2 = (2u * kDf2TreeSelectorTopK + 0u) * kDf2TreeSelectorTopK;
    seltab[base2 + 0] = 1.1f;   // exp(1.1)=3.004
    seltab[base2 + 1] = 0.0f;   // 1/(3.004+1) ~ 0.250
    // Depth 3 runner-up (under node 3): S3*p_run = 0.729 * 0.27 = 0.1968
    const size_t base3 = (3u * kDf2TreeSelectorTopK + 0u) * kDf2TreeSelectorTopK;
    seltab[base3 + 0] = 1.0f;   // exp(1.0)=2.718
    seltab[base3 + 1] = 0.0f;   // 1/3.718 = 0.2689

    // Default ev_cut=0 reproduces df2_tree_build_from_selector identically.
    Df2TreeBuildOptions legacy_opts;
    legacy_opts.draft_conf = 0.5;
    legacy_opts.min_branch_p = 0.10f;
    legacy_opts.max_fanout = 2;
    legacy_opts.max_nodes = 8;
    const Df2TreeSpec ex_def = df2_tree_build_from_selector_ex(99, drafts.data(), probs.data(), 7,
                                                               toptok.data(), seltab.data(), legacy_opts);
    const Df2TreeSpec orig = df2_tree_build_from_selector(99, drafts.data(), probs.data(), 7,
                                                          toptok.data(), seltab.data(), 0.5, 0.10f, 2, 8);
    check(ex_def.count == orig.count && ex_def.token == orig.token && ex_def.parent == orig.parent &&
              ex_def.shape == orig.shape && ex_def.branch_node == orig.branch_node,
          "default options reproduce legacy df2_tree_build_from_selector identically");

    // EV budget (ev_cut=0.05): drops low-EV tail drafts[5], drafts[6] (0.035, 0.0105) in favor of
    // multi-point caterpillar runner-ups 202 (under node 2, EV~0.202) and 203 (under node 3, EV~0.196).
    Df2TreeBuildOptions ev_opts = legacy_opts;
    ev_opts.ev_cut = 0.05f;
    const Df2TreeSpec ev_tree = df2_tree_build_from_selector_ex(99, drafts.data(), probs.data(), 7,
                                                                toptok.data(), seltab.data(), ev_opts);
    // Expected kept: anchor(0), spine 1..5 (101..105), leaves 202 (parent=2), 203 (parent=3) => 8 nodes Caterpillar
    check(ev_tree.count == 8 && ev_tree.valid() && ev_tree.is_caterpillar(), "EV builder produces valid 8-node caterpillar");
    check(ev_tree.shape == Df2TreeShape::Caterpillar && ev_tree.branch_node == -1, "multi-point EV tree is Caterpillar");
    check(ev_tree.token[5] == 105 && ev_tree.token[6] == 202 && ev_tree.parent[6] == 2 &&
              ev_tree.token[7] == 203 && ev_tree.parent[7] == 3,
          "EV builder replaces dead tail nodes with higher-EV runner-ups at nodes 2 and 3");

    // Primary-first topological indexing avoids compaction on primary spine wins and flags runner-up hits cleanly.
    const int32_t arg_prim[8] = {101, 102, 103, 104, 105, 999, 0, 0};
    const Df2TreeCommitWalk w_prim = df2_tree_greedy_accept(ev_tree, arg_prim);
    check(w_prim.n_keep == 6 && !w_prim.needs_compaction() && !w_prim.branch_hit && !w_prim.runner_up_hit,
          "primary spine win on EV caterpillar needs no compaction");
    const int32_t arg_run[8] = {101, 102, 202, 0, 0, 0, 999, 0};
    const Df2TreeCommitWalk w_run = df2_tree_greedy_accept(ev_tree, arg_run);
    check(w_run.n_keep == 4 && w_run.path[3] == 6 && w_run.branch_hit && w_run.runner_up_hit,
          "runner-up leaf win on EV caterpillar sets branch_hit and runner_up_hit");

    // Higher ev_cut=0.25 prunes nodes < 0.25 (S4=0.2916 kept; S5=0.1166 and leaves ~0.20 dropped -> falls back to linear)
    // Wait: check ev_cut=0.18 prunes S5 (0.1166) while keeping leaves 202 (0.202) and 203 (0.196) -> count=7!
    ev_opts.ev_cut = 0.18f;
    const Df2TreeSpec pruned = df2_tree_build_from_selector_ex(99, drafts.data(), probs.data(), 7,
                                                               toptok.data(), seltab.data(), ev_opts);
    check(pruned.count == 7 && pruned.valid() && pruned.shape == Df2TreeShape::Caterpillar,
          "ev_cut=0.18 prunes low-EV spine node 105 to 7 nodes while keeping both higher-EV leaves");

    // High ev_cut=0.30 drops all runner-ups -> falls back to linear
    ev_opts.ev_cut = 0.30f;
    const Df2TreeSpec lin_drop = df2_tree_build_from_selector_ex(99, drafts.data(), probs.data(), 7,
                                                                 toptok.data(), seltab.data(), ev_opts);
    check(lin_drop.is_linear() && lin_drop.count == 8, "high ev_cut dropping all runner-ups falls back to linear");
}

void test_adaptive_cut_choice() {
    check(!df2_tree_adaptive_env_enabled(nullptr) && !df2_tree_adaptive_env_enabled("0") &&
              df2_tree_adaptive_env_enabled("1"),
          "adaptive env gate parser");

    std::array<int32_t, 7> drafts{101, 102, 103, 104, 105, 106, 107};
    std::array<float, 7> probs{0.85f, 0.85f, 0.80f, 0.35f, 0.35f, 0.35f, 0.35f};
    std::array<int32_t, 7 * kDf2TreeSelectorTopK> toptok{};
    std::array<float, 7 * kDf2TreeSelectorTopK * kDf2TreeSelectorTopK> seltab{};
    toptok.fill(-1);
    seltab.fill(-100.0f);
    for (int r = 0; r < 7; ++r) {
        toptok[static_cast<size_t>(r) * kDf2TreeSelectorTopK + 0] = drafts[static_cast<size_t>(r)];
        toptok[static_cast<size_t>(r) * kDf2TreeSelectorTopK + 1] = 200 + r;
        seltab[(static_cast<size_t>(r) * kDf2TreeSelectorTopK + 0) * kDf2TreeSelectorTopK + 0] = 0.0f;
    }
    // Give depth 1 runner-up p=0.28 and depth 2 runner-up p=0.45.
    const size_t base1 = (1u * kDf2TreeSelectorTopK + 0u) * kDf2TreeSelectorTopK;
    seltab[base1 + 0] = 0.944f;  // exp(0.944)=2.570 -> runner 1/(3.570) ~ 0.280
    seltab[base1 + 1] = 0.0f;
    const size_t base2 = (2u * kDf2TreeSelectorTopK + 0u) * kDf2TreeSelectorTopK;
    seltab[base2 + 0] = 0.20f;   // exp(0.20)=1.2214 -> runner 1/(2.2214) ~ 0.450
    seltab[base2 + 1] = 0.0f;

    Df2TreeBuildOptions opts;
    opts.draft_conf = 0.7;
    opts.min_branch_p = 0.10f;
    opts.max_fanout = 2;
    opts.max_nodes = 8;
    opts.adaptive_cut = true;

    // 1. High empirical acceptance at depths 0..2 -> cut=2 beats cut=1 and linear.
    Df2TreeAdaptivePolicy high_pol;
    for (int i = 0; i < 30; ++i) high_pol.observe_linear(7, 7);
    const Df2TreeSpec t_high = df2_tree_build_from_selector_ex(99, drafts.data(), probs.data(), 7,
                                                               toptok.data(), seltab.data(), opts, &high_pol);
    check(t_high.shape == Df2TreeShape::PrefixFanout && t_high.branch_node == 2,
          "high EWMA survival picks higher-yield cut=2");

    // 2. Collapsed depth-1 acceptance (only 1 draft accepted repeatedly) lowers S[2] -> shifts cut to 1.
    Df2TreeAdaptivePolicy low_pol;
    for (int i = 0; i < 30; ++i) low_pol.observe_linear(7, 1);
    const Df2TreeSpec t_low = df2_tree_build_from_selector_ex(99, drafts.data(), probs.data(), 7,
                                                              toptok.data(), seltab.data(), opts, &low_pol);
    check(t_low.shape == Df2TreeShape::PrefixFanout && t_low.branch_node == 1,
          "low depth-1 EWMA survival shifts adaptive cut earlier to 1");

    // 3. observe_tree updates primary spine nodes only (ignores runner-up slots).
    Df2TreeAdaptivePolicy tree_pol;
    const int32_t arg_run[8] = {101, 201, 0, 0, 0, 0, 0, 0};  // depth-0 accepted, depth-1 primary rejected, runner-up 201 accepted
    tree_pol.observe_tree(t_low, df2_tree_greedy_accept(t_low, arg_run));
    check(tree_pol.seen_weight(0) > 0.99 && tree_pol.seen_weight(1) > 0.99 && tree_pol.seen_weight(2) == 0.0,
          "observe_tree updates reached primary depths only");
    check(tree_pol.empirical_rate(0) > 0.6 && tree_pol.empirical_rate(1) < 0.4,
          "observe_tree records depth-0 primary accept and depth-1 primary reject");

    // 4. High primary probs everywhere + tiny runner-up -> linear beats branching (no-gain fallback).
    const std::array<float, 7> strong_probs{0.96f, 0.96f, 0.96f, 0.96f, 0.96f, 0.96f, 0.96f};
    seltab[base1 + 0] = 2.1f;   // p_run ~ 0.11
    seltab[base2 + 0] = 2.1f;   // p_run ~ 0.11
    const Df2TreeSpec t_nogain = df2_tree_build_from_selector_ex(99, drafts.data(), strong_probs.data(), 7,
                                                                 toptok.data(), seltab.data(), opts, &high_pol);
    check(t_nogain.is_linear() && t_nogain.count == 8,
          "adaptive builder falls back to linear when branching lowers expected accepted drafts");
}

void test_sibling_dedup() {
    check(!df2_tree_dedup_env_enabled(nullptr) && !df2_tree_dedup_env_enabled("0") &&
              df2_tree_dedup_env_enabled("1"),
          "dedup env gate parser");

    std::array<int32_t, 7> drafts{101, 102, 103, 104, 105, 106, 107};
    std::array<float, 7> probs{0.95f, 0.90f, 0.40f, 0.80f, 0.80f, 0.80f, 0.80f};
    std::array<int32_t, 7 * kDf2TreeSelectorTopK> toptok{};
    std::array<float, 7 * kDf2TreeSelectorTopK * kDf2TreeSelectorTopK> seltab{};
    toptok.fill(-1);
    seltab.fill(-100.0f);
    for (int r = 0; r < 7; ++r) {
        toptok[static_cast<size_t>(r) * kDf2TreeSelectorTopK + 0] = drafts[static_cast<size_t>(r)];
        seltab[(static_cast<size_t>(r) * kDf2TreeSelectorTopK + 0) * kDf2TreeSelectorTopK + 0] = 0.0f;
    }
    // Cut at c=2: slot 1 collides with prefix token drafts[1]=102, slot 2 collides with anchor=99,
    // slot 3 is future spine token drafts[5]=106 (must NOT be excluded by P1), slot 4 collides with drafts[0]=101,
    // slot 5 is distinct token 303 (with duplicate at slot 6).
    const size_t base2 = (2u * kDf2TreeSelectorTopK + 0u) * kDf2TreeSelectorTopK;
    toptok[2 * kDf2TreeSelectorTopK + 1] = 102;
    toptok[2 * kDf2TreeSelectorTopK + 2] = 99;
    toptok[2 * kDf2TreeSelectorTopK + 3] = 106;
    toptok[2 * kDf2TreeSelectorTopK + 4] = 101;
    toptok[2 * kDf2TreeSelectorTopK + 5] = 303;
    toptok[2 * kDf2TreeSelectorTopK + 6] = 303;
    seltab[base2 + 0] = 1.0f;
    seltab[base2 + 1] = 0.8f;
    seltab[base2 + 2] = 0.7f;
    seltab[base2 + 3] = 0.6f;   // exp(0.6)=1.822 > 2*exp(-0.5)=1.213 -> 106 beats merged 303
    seltab[base2 + 4] = 0.5f;
    seltab[base2 + 5] = -0.5f;
    seltab[base2 + 6] = -0.5f;

    Df2TreeBuildOptions opts;
    opts.draft_conf = 0.5;
    opts.min_branch_p = 0.10f;
    opts.max_fanout = 2;
    opts.max_nodes = 8;

    opts.dedup_siblings = false;
    const Df2TreeSpec no_dedup = df2_tree_build_from_selector_ex(99, drafts.data(), probs.data(), 7,
                                                                 toptok.data(), seltab.data(), opts);
    check(no_dedup.token[4] == 102, "without dedup, prefix-colliding runner-up 102 is selected");

    opts.dedup_siblings = true;
    const Df2TreeSpec with_dedup = df2_tree_build_from_selector_ex(99, drafts.data(), probs.data(), 7,
                                                                   toptok.data(), seltab.data(), opts);
    check(with_dedup.valid() && with_dedup.branch_node == 2 && with_dedup.token[4] == 106,
          "with dedup, anchor/prefix drafts[0..2] (102, 99) are skipped while future drafts[5]=106 is NOT excluded");

    // When slot 3 collides with prefix drafts[0]=101, distinct merged 303 wins.
    toptok[2 * kDf2TreeSelectorTopK + 3] = 101;
    const Df2TreeSpec with_distinct = df2_tree_build_from_selector_ex(99, drafts.data(), probs.data(), 7,
                                                                      toptok.data(), seltab.data(), opts);
    check(with_distinct.valid() && with_distinct.branch_node == 2 && with_distinct.token[4] == 303,
          "with dedup, colliding prefix candidates (102, 99, 101) are skipped in favor of distinct 303");

    toptok[2 * kDf2TreeSelectorTopK + 5] = 101;
    toptok[2 * kDf2TreeSelectorTopK + 6] = 102;
    const Df2TreeSpec all_coll = df2_tree_build_from_selector_ex(99, drafts.data(), probs.data(), 7,
                                                                 toptok.data(), seltab.data(), opts);
    check(all_coll.is_linear() && all_coll.count == 8, "all-colliding runner-ups fall back to linear");

    // P1 sample_walk=true topology invariance: changing drafts[cut] must NOT alter runner selection or topology.
    toptok[2 * kDf2TreeSelectorTopK + 3] = 303;
    toptok[2 * kDf2TreeSelectorTopK + 5] = 404;
    toptok[2 * kDf2TreeSelectorTopK + 6] = -1;
    seltab[base2 + 3] = 0.85f;  // 0.855 * (exp(0.85)/sum) > 0.15 ev_cut
    seltab[base2 + 5] = -0.5f;
    Df2TreeBuildOptions sw_opts = opts;
    sw_opts.sample_walk = true;
    sw_opts.ev_cut = 0.15f;
    std::array<int32_t, 7> drafts_alt = drafts;
    drafts_alt[2] = 303;  // non-argmax draw at cut=2
    const Df2TreeSpec sw_a = df2_tree_build_from_selector_ex(99, drafts.data(), probs.data(), 7,
                                                             toptok.data(), seltab.data(), sw_opts);
    const Df2TreeSpec sw_b = df2_tree_build_from_selector_ex(99, drafts_alt.data(), probs.data(), 7,
                                                             toptok.data(), seltab.data(), sw_opts);
    check(!sw_a.is_linear() && sw_a.count == sw_b.count && sw_a.parent == sw_b.parent &&
              sw_a.slot_rank == sw_b.slot_rank && sw_a.shape == sw_b.shape && sw_a.branch_node == sw_b.branch_node,
          "sample_walk=true keeps tree topology invariant to which token was drawn at cut depth");
}

void test_extended_telemetry_and_runner_hit() {
    // PrefixFanout at cut=1, F=2, plus primary tail: 0->1(10); 1->2(20), 1->3(21); 2->4(30)->5(31)
    const int32_t prefix[1] = {10};
    const int32_t br[2] = {20, 21};
    const int32_t tail[2] = {30, 31};
    const Df2TreeSpec pf = make_prefix_fanout_tree(1, prefix, 1, br, 2, tail, 2);

    // Primary tail win (0->1->2->4->5): path[2]=2, path[3]=4 != 3 => branch_hit=true, runner_up_hit=false!
    const int32_t arg_tail[6] = {10, 20, 30, 0, 31, 999};
    const Df2TreeCommitWalk w_tail = df2_tree_greedy_accept(pf, arg_tail);
    check(w_tail.branch_hit && !w_tail.runner_up_hit && w_tail.accepted_drafts() == 4,
          "primary-tail win separates branch_hit=true from runner_up_hit=false");

    // Runner-up win (0->1->3): runner_up_hit=true
    const int32_t arg_run[6] = {10, 21, 0, 888, 0, 0};
    const Df2TreeCommitWalk w_run = df2_tree_greedy_accept(pf, arg_run);
    check(w_run.branch_hit && w_run.runner_up_hit && w_run.accepted_drafts() == 2,
          "runner-up win sets runner_up_hit=true");

    Df2TreeAcceptHist hist;
    hist.observe(pf, w_tail);
    hist.observe(pf, w_run);
    check(hist.tree_rounds == 2 && hist.branch_hits == 2 && hist.runner_hits == 1 && hist.total_offered == 10,
          "extended telemetry tracks runner_hits separately from branch_hits and sums total_offered");
    // Round 1 (w_tail): node 3 (runner) reached (parent 1 accepted), rejected; Round 2 (w_run): node 3 reached, accepted -> 1/2
    check(hist.runner_reached == 2 && hist.runner_accepted == 1,
          "runner_reached and runner_accepted isolate sibling recovery rate");
    // Depth 1 (node 1): 2/2; Depth 2 (nodes 2,3): 4 reached, 2 accepted; Depth 3 (node 4): 1/1; Depth 4 (node 5): 1/1
    check(hist.depth_reached[0] == 2 && hist.depth_accepted[0] == 2 &&
              hist.depth_reached[1] == 4 && hist.depth_accepted[1] == 2 &&
              hist.depth_reached[2] == 1 && hist.depth_accepted[2] == 1 &&
              hist.depth_reached[3] == 1 && hist.depth_accepted[3] == 1,
          "depth_reached/depth_accepted track true depth-indexed acceptance");
    check(hist.cut_hist[1] == 2, "cut_hist records branch_node=1");

    const std::string formatted = hist.format_accept_hist_fields();
    Df2TreeAcceptHist parsed;
    check(parse_accept_hist_fields(formatted, parsed), "extended accept-hist fields parse cleanly");
    check(parsed.runner_hits == 1 && parsed.total_offered == 10 &&
              parsed.runner_reached == 2 && parsed.runner_accepted == 1 &&
              parsed.depth_reached == hist.depth_reached &&
              parsed.depth_accepted == hist.depth_accepted &&
              parsed.cut_hist == hist.cut_hist,
          "extended accept-hist round-trip preserves all fields");

    // Legacy 4-field log format still parses cleanly with zero-initialized new fields.
    Df2TreeAcceptHist legacy;
    const std::string legacy_line =
        "[accept-hist] req=1 df2_pos_reached=1,0,0,0,0,0,0 df2_pos_accept=1,0,0,0,0,0,0 "
        "lookup_rows=0 lookup_accept=0 chain_rows=0 chain_accept=0 "
        "tree_rounds=5 tree_branch_hits=2 tree_node_reached=5,5,4,2,0,0,0,0 tree_node_accept=5,3,2,1,0,0,0,0";
    check(parse_accept_hist_fields(legacy_line, legacy) &&
              legacy.tree_rounds == 5 && legacy.branch_hits == 2 && legacy.runner_hits == 0,
          "legacy 4-field accept-hist line remains backward-compatible");
}

void test_gdn_order_ple_taps_and_remap_invariants() {
    const int32_t prefix[2] = {10, 11};
    const int32_t br[3] = {20, 21, 22};
    const int32_t ptail[1] = {30};
    // 0->1->2; 2->3(20), 2->4(21), 2->5(22); 3->6(30) => count = 7, padded row 7 at depth 1 under 0
    const Df2TreeSpec t = make_prefix_fanout_tree(1, prefix, 2, br, 3, ptail, 1);
    check(t.count == 7 && t.is_caterpillar(), "7-node fanout tree is caterpillar");

    int32_t order[8] = {}, is_spine[8] = {};
    check(df2_tree_caterpillar_gdn_order(t, order, is_spine, 8), "gdn_order succeeds for 7-node tree padded to 8");
    // Depth 0: 0(spine)
    // Depth 1: 7(padded leaf) before 1(spine)
    // Depth 2: 2(spine)
    // Depth 3: 4(leaf), 5(leaf) before 3(spine)
    // Depth 4: 6(leaf)
    check(order[0] == 0 && is_spine[0] == 1, "root spine first");
    check(order[1] == 7 && order[2] == 1 && is_spine[7] == 0 && is_spine[1] == 1,
          "padded dummy leaf runs before depth-1 spine");
    check(order[3] == 2 && is_spine[2] == 1, "depth-2 spine");
    check(order[4] == 4 && order[5] == 5 && order[6] == 3 && is_spine[3] == 1,
          "depth-3 runner-up leaves run before depth-3 spine");
    check(order[7] == 6 && is_spine[6] == 0, "depth-4 tail leaf last");

    // PLE dilated conv taps [d-9, d-6, d-3]
    check((df2_tree_ple_conv_taps(t, 0) == std::array<int32_t, 3>{-9, -6, -3}), "root ple taps");
    check((df2_tree_ple_conv_taps(t, 3) == std::array<int32_t, 3>{-6, -3, 0}), "depth-3 primary ple taps");
    check((df2_tree_ple_conv_taps(t, 5) == std::array<int32_t, 3>{-6, -3, 0}), "depth-3 runner-up ple taps");
    check((df2_tree_ple_conv_taps(t, 6) == std::array<int32_t, 3>{-5, -2, 1}), "depth-4 tail ple taps");

    // Verify QSA ancestor remap preserves strictly ascending order and non-contiguous commit satisfies path[k] >= k
    for (int pos0 = 0; pos0 < 16; ++pos0) {
        for (int n = 0; n < t.count; ++n) {
            int prev_cell = -1;
            for (int d = 0; d <= static_cast<int>(t.depth[static_cast<size_t>(n)]); ++d) {
                const int phys = pos0 + static_cast<int>(t.anc[static_cast<size_t>(n)][static_cast<size_t>(d)]);
                check(phys > prev_cell, "remapped tree QSA cells stay strictly ascending");
                prev_cell = phys;
            }
        }
    }
    const int32_t argmax[7] = {10, 11, 22, 0, 0, 999, 0};
    const Df2TreeCommitWalk walk = df2_tree_greedy_accept(t, argmax);
    check(walk.n_keep == 4 && walk.path[3] == 5 && walk.needs_compaction(), "runner-up branch 5 selected");
    for (int k = 0; k < walk.n_keep; ++k) {
        check(static_cast<int>(walk.path[static_cast<size_t>(k)]) >= k, "commit path satisfies path[k] >= k");
    }
}

void test_var_t_width_ladder() {
    check(!df2_tree_var_t_env_enabled(nullptr), "var_t env null is off");
    check(!df2_tree_var_t_env_enabled(""), "var_t env empty is off");
    check(!df2_tree_var_t_env_enabled("0"), "var_t env 0 is off");
    check(!df2_tree_var_t_env_enabled("true"), "var_t env non-1 is off");
    check(df2_tree_var_t_env_enabled("1"), "var_t env 1 is on");

    // Default-OFF knob preserves fixed T=8 for every tree node count M in 2..8.
    for (int m = 1; m <= 10; ++m) {
        check(df2_tree_select_width(m, false) == kDf2TreeMaxNodes,
              "var_t=off always selects fixed T=8");
    }

    // Dense ladder (T in 3..8): M=2 clamps to T=3, M=3..8 map identically.
    check(df2_tree_select_width(2, true) == 3, "dense var_t maps M=2 -> T=3");
    for (int m = 3; m <= 8; ++m) {
        check(df2_tree_select_width(m, true) == m, "dense var_t maps M -> T=M for 3..8");
    }
    check(df2_tree_select_width(0, true) == 3, "var_t clamps below-min M to 3");
    check(df2_tree_select_width(12, true) == 8, "var_t clamps above-max M to 8");

    // Sparse ladder e.g. {4, 6, 8} (bits 4, 6, 8 => 0x150): M->next enabled step >= M.
    constexpr unsigned kSparseMask = (1u << 4) | (1u << 6) | (1u << 8);
    check(df2_tree_select_width(2, true, kSparseMask) == 4, "sparse ladder M=2 -> T=4");
    check(df2_tree_select_width(3, true, kSparseMask) == 4, "sparse ladder M=3 -> T=4");
    check(df2_tree_select_width(4, true, kSparseMask) == 4, "sparse ladder M=4 -> T=4");
    check(df2_tree_select_width(5, true, kSparseMask) == 6, "sparse ladder M=5 -> T=6");
    check(df2_tree_select_width(6, true, kSparseMask) == 6, "sparse ladder M=6 -> T=6");
    check(df2_tree_select_width(7, true, kSparseMask) == 8, "sparse ladder M=7 -> T=8");
    check(df2_tree_select_width(8, true, kSparseMask) == 8, "sparse ladder M=8 -> T=8");
    check(df2_tree_select_width(4, true, 0u) == 8, "empty mask falls back to T=8");

    // Verify gdn_order with pad_to=T keeps all real nodes in [0, T) without dummy leaves bumping spine nodes.
    const int32_t prefix[1] = {10};
    const int32_t br[2] = {20, 21};
    // Nodes: 0->1(10)->2(20), 1->3(21) => M=4 (leaves at depth 2 emitted in index order 2, 3).
    const Df2TreeSpec t4 = make_prefix_fanout_tree(1, prefix, 1, br, 2, nullptr, 0);
    check(t4.count == 4, "4-node fanout tree count");
    const int w4 = df2_tree_select_width(t4.count, true);
    check(w4 == 4, "4-node tree selects width T=4");
    int32_t order[8] = {}, is_spine[8] = {};
    check(df2_tree_caterpillar_gdn_order(t4, order, is_spine, w4), "gdn_order succeeds for pad_to=T=4");
    check(order[0] == 0 && order[1] == 1 && order[2] == 2 && order[3] == 3,
          "gdn_order(T=4) schedules all 4 real nodes in [0,4)");

    // With a primary tail child off branch 0 (M=5: 0->1->2->4, 1->3), depth-2 runner-up leaf 3 precedes
    // depth-2 spine 2, and no dummy leaves precede depth-1 spine 1 when pad_to=T=5.
    const int32_t ptail[1] = {30};
    const Df2TreeSpec t5 = make_prefix_fanout_tree(1, prefix, 1, br, 2, ptail, 1);
    check(t5.count == 5, "5-node fanout tree count");
    const int w5 = df2_tree_select_width(t5.count, true);
    check(w5 == 5, "5-node tree selects width T=5");
    check(df2_tree_caterpillar_gdn_order(t5, order, is_spine, w5), "gdn_order succeeds for pad_to=T=5");
    check(order[0] == 0 && order[1] == 1 && order[2] == 3 && order[3] == 2 && order[4] == 4,
          "gdn_order(T=5) keeps leaf-before-spine without dummy leaves in [0,5)");
}

void test_spine_skip_predicate() {
    check(!df2_tree_spine_skip_env_enabled(nullptr), "spine_skip env null is off");
    check(!df2_tree_spine_skip_env_enabled(""), "spine_skip env empty is off");
    check(!df2_tree_spine_skip_env_enabled("0"), "spine_skip env 0 is off");
    check(!df2_tree_spine_skip_env_enabled("true"), "spine_skip env non-1 is off");
    check(df2_tree_spine_skip_env_enabled("1"), "spine_skip env 1 is on");

    const uint8_t contig[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    check(!df2_tree_is_contiguous_spine(nullptr, 1), "null path rejected");
    check(!df2_tree_is_contiguous_spine(contig, 0), "n_keep=0 rejected");
    check(!df2_tree_is_contiguous_spine(contig, 9), "n_keep>8 rejected");
    for (int k = 1; k <= 8; ++k) {
        check(df2_tree_is_contiguous_spine(contig, k), "contiguous spine prefix detected");
        check(!df2_tree_can_skip_kv_compact(false, contig, k), "knob off never skips kv_compact_tree");
        check(df2_tree_can_skip_kv_compact(true, contig, k), "knob on + contiguous spine skips kv_compact_tree");
    }

    const uint8_t bad_root[2] = {1, 2};
    const uint8_t branched2[2] = {0, 2};
    const uint8_t branched4[4] = {0, 1, 2, 5};
    const uint8_t branched5[5] = {0, 1, 3, 4, 5};
    check(!df2_tree_is_contiguous_spine(bad_root, 1), "non-zero root rejected");
    check(!df2_tree_is_contiguous_spine(branched2, 2), "depth-1 branch rejected");
    check(!df2_tree_is_contiguous_spine(branched4, 4), "deep runner-up branch rejected");
    check(!df2_tree_is_contiguous_spine(branched5, 5), "shifted branch tail rejected");
    check(!df2_tree_can_skip_kv_compact(true, branched4, 4), "knob on never skips kv_compact_tree when branched");

    // Cross-check against df2_tree_greedy_accept walks on a 7-node fanout tree.
    const int32_t prefix[2] = {10, 11};
    const int32_t br[3] = {20, 21, 22};
    const int32_t ptail[1] = {30};
    const Df2TreeSpec t = make_prefix_fanout_tree(1, prefix, 2, br, 3, ptail, 1);

    // Contiguous spine acceptance (0->1->2->3->6 => n_keep=5, path={0,1,2,3,6}? Wait: node 6!=4!).
    // Primary prefix+branch up to node 3 is contiguous {0,1,2,3}; tail node 6 has index 6 != depth 4!
    const int32_t argmax_spine3[7] = {10, 11, 20, 999, 0, 0, 0};
    const Df2TreeCommitWalk walk_s3 = df2_tree_greedy_accept(t, argmax_spine3);
    check(walk_s3.n_keep == 4 && !walk_s3.needs_compaction(), "primary walk 0->1->2->3 needs no compaction");
    check(df2_tree_can_skip_kv_compact(true, walk_s3.path.data(), walk_s3.n_keep),
          "primary walk 0->1->2->3 qualifies for spine KV skip");

    // Primary walk continuing to ptail node 6 skips over runner-up leaves 4,5 -> path[4]==6 != 4!
    const int32_t argmax_tail[7] = {10, 11, 20, 30, 0, 0, 999};
    const Df2TreeCommitWalk walk_tail = df2_tree_greedy_accept(t, argmax_tail);
    check(walk_tail.n_keep == 5 && walk_tail.path[4] == 6 && walk_tail.needs_compaction(),
          "tail past fanout branch shifts physical index (6 != 4) and requires compaction");
    check(!df2_tree_can_skip_kv_compact(true, walk_tail.path.data(), walk_tail.n_keep),
          "non-contiguous tail after fanout does NOT skip kv_compact_tree");
}

void test_composable_ev_adaptive_matrix_and_no_inflation() {
    check(!df2_tree_adaptive_inflate_env_enabled(nullptr) &&
              !df2_tree_adaptive_inflate_env_enabled("") &&
              !df2_tree_adaptive_inflate_env_enabled("0") &&
              df2_tree_adaptive_inflate_env_enabled("1"),
          "adaptive_inflate env gate parser");

    Df2TreeAdaptivePolicy high_pol;
    for (int i = 0; i < 30; ++i) high_pol.observe_linear(7, 7);
    // Non-inflating effective_prob must never inflate low selector_p above raw p unless opted in.
    check(high_pol.effective_prob(0, 0.20f) == static_cast<double>(0.20f),
          "default effective_prob clamps upward inflation to raw selector_p");
    check(high_pol.effective_prob(0, 0.20f, true) > 0.45,
          "opt-in allow_inflate=true reproduces upward EMA blend");

    Df2TreeAdaptivePolicy low_pol;
    for (int i = 0; i < 30; ++i) low_pol.observe_linear(7, 1);
    check(low_pol.effective_prob(1, 0.85f) < 0.60,
          "default effective_prob still discounts overconfident collapsed depth");

    // 1. 2x2 selection matrix (ev_cut x adaptive) on a standard profile with collapsed depth-1 history.
    std::array<int32_t, 7> drafts{101, 102, 103, 104, 105, 106, 107};
    std::array<float, 7> probs{0.90f, 0.85f, 0.80f, 0.35f, 0.35f, 0.30f, 0.30f};
    std::array<int32_t, 7 * kDf2TreeSelectorTopK> toptok{};
    std::array<float, 7 * kDf2TreeSelectorTopK * kDf2TreeSelectorTopK> seltab{};
    toptok.fill(-1);
    seltab.fill(-100.0f);
    for (int r = 0; r < 7; ++r) {
        toptok[static_cast<size_t>(r) * kDf2TreeSelectorTopK + 0] = drafts[static_cast<size_t>(r)];
        toptok[static_cast<size_t>(r) * kDf2TreeSelectorTopK + 1] = 200 + r;
        seltab[(static_cast<size_t>(r) * kDf2TreeSelectorTopK + 0) * kDf2TreeSelectorTopK + 0] = 0.0f;
    }
    // Depth-2 runner-up has p_run ~ 0.40 -> raw leaf_ev = 0.90*0.85*0.40 = 0.306 (>= 0.25).
    // Under low_pol (emp(1) ~ 0.068 -> p_cal[1] < 0.56), cal leaf_ev = 0.90*0.54*0.40 < 0.25.
    const size_t base2 = (2u * kDf2TreeSelectorTopK + 0u) * kDf2TreeSelectorTopK;
    seltab[base2 + 0] = 0.4055f;  // exp(0.4055) = 1.50 -> runner = 1/2.5 = 0.40
    seltab[base2 + 1] = 0.0f;

    // Cell (0,0): ev_cut=0, adaptive=false => legacy draft_conf cut at c=2 (0.90*0.85=0.765 >= 0.7, *0.80=0.612 < 0.7).
    Df2TreeBuildOptions opts;
    opts.draft_conf = 0.7;
    opts.min_branch_p = 0.10f;
    opts.max_fanout = 2;
    opts.max_nodes = 8;
    const Df2TreeSpec cell00 = df2_tree_build_from_selector_ex(99, drafts.data(), probs.data(), 7,
                                                               toptok.data(), seltab.data(), opts, &low_pol);
    const Df2TreeSpec cell00_leg = df2_tree_build_from_selector(99, drafts.data(), probs.data(), 7,
                                                                toptok.data(), seltab.data(), 0.7, 0.10f, 2, 8);
    check(cell00.count == 8 && cell00.shape == Df2TreeShape::PrefixFanout && cell00.branch_node == 2 &&
              cell00.token == cell00_leg.token && cell00.parent == cell00_leg.parent,
          "matrix (ev=0, adaptive=0): bit-identical legacy PrefixFanout regardless of adaptive pointer");

    // Cell (1,0): ev_cut=0.25, adaptive=false => pure EV ignores low_pol and keeps depth-2 leaf.
    opts.ev_cut = 0.25f;
    opts.adaptive_cut = false;
    const Df2TreeSpec cell10_null = df2_tree_build_from_selector_ex(99, drafts.data(), probs.data(), 7,
                                                                    toptok.data(), seltab.data(), opts, nullptr);
    const Df2TreeSpec cell10_low = df2_tree_build_from_selector_ex(99, drafts.data(), probs.data(), 7,
                                                                   toptok.data(), seltab.data(), opts, &low_pol);
    check(!cell10_null.is_linear() && cell10_null.count == cell10_low.count &&
              cell10_null.token == cell10_low.token && cell10_null.parent == cell10_low.parent,
          "matrix (ev>0, adaptive=0): pure EV ignores adaptive state identically to nullptr");

    // Cell (0,1): ev_cut=0, adaptive=true => Policy 2 uses calibrated survival, detects low_pol collapse.
    opts.ev_cut = 0.0f;
    opts.adaptive_cut = true;
    const Df2TreeSpec cell01_high = df2_tree_build_from_selector_ex(99, drafts.data(), probs.data(), 7,
                                                                    toptok.data(), seltab.data(), opts, &high_pol);
    check(cell01_high.shape == Df2TreeShape::PrefixFanout && cell01_high.branch_node == 2,
          "matrix (ev=0, adaptive=1): Policy 2 builds branch at depth 2 under high_pol");

    // Cell (1,1): ev_cut=0.25, adaptive=true => Policy 2 is consulted on top of EV:
    // - under high_pol: calibrated leaf_ev stays >= 0.25 and net EV beats linear -> keeps pruned tree (count < 8).
    // - under low_pol: calibrated leaf_ev drops below 0.25 -> Policy 2 suppresses collapsed tree -> linear!
    opts.ev_cut = 0.25f;
    opts.adaptive_cut = true;
    const Df2TreeSpec cell11_high = df2_tree_build_from_selector_ex(99, drafts.data(), probs.data(), 7,
                                                                    toptok.data(), seltab.data(), opts, &high_pol);
    const Df2TreeSpec cell11_low = df2_tree_build_from_selector_ex(99, drafts.data(), probs.data(), 7,
                                                                   toptok.data(), seltab.data(), opts, &low_pol);
    check(!cell11_high.is_linear() && cell11_high.count == 5,
          "matrix (ev=0.25, adaptive=1): high_pol keeps EV-pruned 5-node tree (0,101,102,103,202)");
    check(cell11_low.is_linear() && cell11_low.count == 8,
          "matrix (ev=0.25, adaptive=1): Policy 2 suppresses collapsed depth-2 EV tree to linear");

    // 2. EA-style low-confidence inflation regression test:
    // Low selector confidence probs={0.20, 0.85, 0.85, ...} with high_pol and depth-1 runner-up p=0.33:
    // - Legacy EA (adaptive_inflate=true, ev_use_adaptive_probs=true) inflates 0.20 -> ~0.52, making
    //   depth-1 leaf_ev = 0.52 * 0.33 = 0.172 >= 0.15 and firing a low-yield tree!
    // - Fixed EA (default adaptive_inflate=false, ev_use_adaptive_probs=false) prices raw 0.20:
    //   depth-1 leaf_ev = 0.20 * 0.33 = 0.066 < 0.15 -> stays linear (no double-counted EMA inflation!).
    const std::array<float, 7> low_start_probs{0.20f, 0.85f, 0.85f, 0.80f, 0.30f, 0.20f, 0.20f};
    std::array<float, 7 * kDf2TreeSelectorTopK * kDf2TreeSelectorTopK> ea_seltab{};
    ea_seltab.fill(-100.0f);
    for (int r = 0; r < 7; ++r) {
        ea_seltab[(static_cast<size_t>(r) * kDf2TreeSelectorTopK + 0) * kDf2TreeSelectorTopK + 0] = 0.0f;
    }
    const size_t ea_base1 = (1u * kDf2TreeSelectorTopK + 0u) * kDf2TreeSelectorTopK;
    ea_seltab[ea_base1 + 0] = 0.708f;  // exp(0.708) = 2.030 -> runner = 1/3.030 = 0.330
    ea_seltab[ea_base1 + 1] = 0.0f;

    Df2TreeBuildOptions ea_fixed;
    ea_fixed.draft_conf = 0.7;
    ea_fixed.min_branch_p = 0.20f;
    ea_fixed.max_fanout = 2;
    ea_fixed.max_nodes = 8;
    ea_fixed.ev_cut = 0.15f;
    ea_fixed.adaptive_cut = true;
    const Df2TreeSpec t_ea_fixed = df2_tree_build_from_selector_ex(99, drafts.data(), low_start_probs.data(), 7,
                                                                   toptok.data(), ea_seltab.data(), ea_fixed, &high_pol);
    check(t_ea_fixed.is_linear(),
          "fixed EA combo prices raw p=0.20 without inflating depth-1 leaf EV >= 0.15");

    Df2TreeBuildOptions ea_legacy = ea_fixed;
    ea_legacy.adaptive_inflate = true;
    ea_legacy.ev_use_adaptive_probs = true;
    const Df2TreeSpec t_ea_legacy = df2_tree_build_from_selector_ex(99, drafts.data(), low_start_probs.data(), 7,
                                                                    toptok.data(), ea_seltab.data(), ea_legacy, &high_pol);
    check(!t_ea_legacy.is_linear() && t_ea_legacy.count >= 4,
          "opt-in legacy inflation reproduces old EA low-yield tree trigger");
}

void test_max_cut_and_row_ema_cost_gates() {
    // 1. Env parsers
    check(df2_tree_parse_max_cut(nullptr) == -1 && df2_tree_parse_max_cut("") == -1 &&
              df2_tree_parse_max_cut("abc") == -1 && df2_tree_parse_max_cut("-1") == -1,
          "max_cut parser defaults invalid/unset to -1 (off)");
    check(df2_tree_parse_max_cut("0") == 0 && df2_tree_parse_max_cut("3") == 3 &&
              df2_tree_parse_max_cut("9") == kDf2TreeMaxDrafts,
          "max_cut parser accepts non-negative integers and clamps upper bound");

    check(df2_tree_parse_row_ema(nullptr) == 0.0f && df2_tree_parse_row_ema("") == 0.0f &&
              df2_tree_parse_row_ema("0") == 0.0f && df2_tree_parse_row_ema("-0.2") == 0.0f &&
              df2_tree_parse_row_ema("1.5") == 0.0f && df2_tree_parse_row_ema("bad") == 0.0f,
          "row_ema parser defaults invalid/non-positive/out-of-range to 0 (off)");
    check(df2_tree_parse_row_ema("0.35") > 0.349f && df2_tree_parse_row_ema("0.35") < 0.351f,
          "row_ema parser parses 0.35 cleanly");

    // 2. Cut-depth guard (max_cut = 3): cut=4 stands down across Legacy, EV, and Adaptive; cut=2 stays active.
    std::array<int32_t, 7> drafts{101, 102, 103, 104, 105, 106, 107};
    // 0.95^4 = 0.8145 >= 0.7, *0.45 = 0.3665 < 0.7 => draft_conf cut = 4
    const std::array<float, 7> probs_cut4{0.95f, 0.95f, 0.95f, 0.95f, 0.45f, 0.40f, 0.30f};
    const std::array<float, 7> probs_cut2{0.90f, 0.90f, 0.55f, 0.45f, 0.40f, 0.30f, 0.30f};
    std::array<int32_t, 7 * kDf2TreeSelectorTopK> toptok{};
    std::array<float, 7 * kDf2TreeSelectorTopK * kDf2TreeSelectorTopK> seltab{};
    toptok.fill(-1);
    seltab.fill(-100.0f);
    for (int r = 0; r < 7; ++r) {
        toptok[static_cast<size_t>(r) * kDf2TreeSelectorTopK + 0] = drafts[static_cast<size_t>(r)];
        toptok[static_cast<size_t>(r) * kDf2TreeSelectorTopK + 1] = 200 + r;
        const size_t base = (static_cast<size_t>(r) * kDf2TreeSelectorTopK + 0u) * kDf2TreeSelectorTopK;
        seltab[base + 0] = 0.0f;
        if (r == 2 || r == 4) {
            seltab[base + 0] = 0.20f;  // exp(0.20)=1.2214 -> runner p ~ 0.450
            seltab[base + 1] = 0.0f;
        }
    }

    Df2TreeBuildOptions opts;
    opts.draft_conf = 0.7;
    opts.min_branch_p = 0.35f;
    opts.max_fanout = 2;
    opts.max_nodes = 8;

    // Unset (max_cut = -1): cut=4 enters tree mode in legacy and EV.
    check(!df2_tree_build_from_selector_ex(99, drafts.data(), probs_cut4.data(), 7,
                                           toptok.data(), seltab.data(), opts).is_linear(),
          "max_cut=-1 (off) allows cut=4 legacy tree");
    opts.ev_cut = 0.35f;
    check(!df2_tree_build_from_selector_ex(99, drafts.data(), probs_cut4.data(), 7,
                                           toptok.data(), seltab.data(), opts).is_linear(),
          "max_cut=-1 (off) allows cut=4 EV tree");

    // max_cut = 3: cut=4 stands down in Legacy, EV, and Adaptive; cut=2 still enters!
    opts.max_cut = 3;
    opts.ev_cut = 0.0f;
    check(df2_tree_build_from_selector_ex(99, drafts.data(), probs_cut4.data(), 7,
                                          toptok.data(), seltab.data(), opts).is_linear(),
          "max_cut=3 stands down on cut=4 in legacy mode");
    opts.ev_cut = 0.35f;
    check(df2_tree_build_from_selector_ex(99, drafts.data(), probs_cut4.data(), 7,
                                          toptok.data(), seltab.data(), opts).is_linear(),
          "max_cut=3 stands down on cut=4 in EV mode");
    opts.ev_cut = 0.0f;
    opts.adaptive_cut = true;
    Df2TreeAdaptivePolicy high_pol;
    for (int i = 0; i < 30; ++i) high_pol.observe_linear(7, 7);
    check(df2_tree_build_from_selector_ex(99, drafts.data(), probs_cut4.data(), 7,
                                          toptok.data(), seltab.data(), opts, &high_pol).is_linear(),
          "max_cut=3 stands down on cut=4 in adaptive mode");

    opts.adaptive_cut = false;
    opts.ev_cut = 0.35f;
    const Df2TreeSpec t_cut2 = df2_tree_build_from_selector_ex(99, drafts.data(), probs_cut2.data(), 7,
                                                               toptok.data(), seltab.data(), opts);
    check(!t_cut2.is_linear() && t_cut2.branch_node == 2,
          "max_cut=3 allows cut=2 EV tree entry");

    // 3. Counterfactual linear accept helper
    const int32_t prefix[2] = {10, 11};
    const int32_t br[2] = {20, 21};
    const int32_t tail[1] = {30};
    // 0->1(10)->2(11)->3(20)->5(30), 2->4(21)
    const Df2TreeSpec cf_tree = make_prefix_fanout_tree(1, prefix, 2, br, 2, tail, 1);
    {
        const int32_t arg_tail[6] = {10, 11, 20, 30, 0, 999};  // primary walk 0->1->2->3->5 (4 drafts)
        const Df2TreeCommitWalk w = df2_tree_greedy_accept(cf_tree, arg_tail);
        check(df2_tree_counterfactual_linear_accept(cf_tree, w, 2) == 2 &&
                  df2_tree_counterfactual_linear_accept(cf_tree, w, 5) == 4,
              "counterfactual linear accept caps primary spine acceptance at lin_cf_drafts");
    }
    {
        const int32_t arg_run[6] = {10, 11, 21, 0, 999, 0};  // runner-up hit at node 4 (depth 3, rank 1)
        const Df2TreeCommitWalk w = df2_tree_greedy_accept(cf_tree, arg_run);
        check(df2_tree_counterfactual_linear_accept(cf_tree, w, 3) == 2 &&
                  df2_tree_counterfactual_linear_accept(cf_tree, w, 1) == 1,
              "counterfactual linear accept stops before runner-up node");
    }

    // 4. Df2TreeRowEmaPolicy suppression + probe recovery (reusing Df2AdaptiveSolo cadence)
    Df2TreeRowEmaPolicy ema_pol;
    check(ema_pol.allow_tree(0.0), "row_ema=0 never suppresses");
    for (int i = 0; i < 7; ++i) {
        ema_pol.observe(0, 3, 0.35);
        check(!ema_pol.suppressed() && ema_pol.allow_tree(0.35),
              "row_ema stays active during 8-round warmup");
    }
    ema_pol.observe(0, 3, 0.35);  // 8th round -> rate 0.0 < 0.35 -> suppressed!
    check(ema_pol.suppressed(), "row_ema suppresses after 8 low-yield rounds");
    for (int i = 0; i < Df2TreeRowEmaPolicy::kProbeInterval; ++i) {
        check(!ema_pol.allow_tree(0.35), "suppressed row_ema skips tree during probe cooldown");
    }
    check(ema_pol.allow_tree(0.35), "suppressed row_ema admits one probe after cooldown");
    // Simulate probe recovery with high-yield tree rounds (d_acc=2, d_row=3 => 0.667 >= 0.35)
    for (int p = 0; p < 4 && ema_pol.suppressed(); ++p) {
        ema_pol.observe(2, 3, 0.35);
        if (ema_pol.suppressed()) {
            for (int i = 0; i < Df2TreeRowEmaPolicy::kProbeInterval; ++i) {
                (void) ema_pol.allow_tree(0.35);
            }
            check(ema_pol.allow_tree(0.35), "cooldown probe fires");
        }
    }
    check(!ema_pol.suppressed() && ema_pol.marginal_rate() >= 0.35,
          "row_ema recovers from suppression once probe dAcc/dRow clears 0.35");

    // 5. Cost-normalized predicate (VAR_T=1 vs T=8 padding)
    // Suppose T_lin=1 (cost 20.0ms, e_lin=0), M=5 tree adds e_tree=1.60:
    // - Under VAR_T=1: T_tree=5 (cost 31.4ms, +4 rows => 0.40 dAcc/dRow >= 0.35) -> PASS!
    // - Under T=8 pad: T_tree=8 (cost 40.0ms, +7 rows => 0.229 dAcc/dRow < 0.35) -> FAIL!
    check(df2_tree_cost_normalized_beats(1.60, 5, 31.4, 0.0, 1, 20.0, 0.35),
          "cost-normalized gate passes M=5 unpadded tree clearing 0.35 dAcc/dRow");
    check(!df2_tree_cost_normalized_beats(1.60, 8, 40.0, 0.0, 1, 20.0, 0.35),
          "cost-normalized gate rejects padded T=8 tree below 0.35 dAcc/dRow");
    check(!df2_tree_cost_normalized_beats(0.80, 5, 31.4, 0.0, 1, 20.0, 0.332),
          "cost-normalized gate catches low dAcc/dRow=0.20 < 0.332 even when local ratio 1.8/31.4 > 1/20");
}

void test_engagement_shaping_envelope() {
    // 1. Safety guard parser & VAR_T=0 clamp semantics
    check(!df2_tree_branch_guard_env_enabled(nullptr) &&
              !df2_tree_branch_guard_env_enabled("") &&
              !df2_tree_branch_guard_env_enabled("0") &&
              !df2_tree_branch_guard_env_enabled("true") &&
              df2_tree_branch_guard_env_enabled("1"),
          "branch guard env gate parser");
    check(df2_tree_effective_min_branch_p(0.10f, false, false) == 0.10f &&
              df2_tree_effective_min_branch_p(0.30f, false, false) == 0.30f,
          "guard off keeps min_branch_p < 0.45 unchanged when VAR_T=0 (bit-identical HEAD / Run 1)");
    check(df2_tree_effective_min_branch_p(0.10f, false, true) == kDf2TreeSafeMinBranchP &&
              df2_tree_effective_min_branch_p(0.30f, false, true) == kDf2TreeSafeMinBranchP &&
              df2_tree_effective_min_branch_p(0.50f, false, true) == 0.50f,
          "guard on clamps min_branch_p < 0.45 to 0.45 when VAR_T=0");
    check(df2_tree_effective_min_branch_p(0.30f, true, true) == 0.30f &&
              df2_tree_effective_min_branch_p(0.35f, true, true) == 0.35f,
          "guard on preserves min_branch_p=0.30..0.35 when VAR_T=1");

    // 2. Variable-T ON engagement ladder: MIN_BRANCH_P=0.30..0.35 + MAX_FANOUT=2 + DEDUP=1 + EV_CUT=0.35
    // produces M in [4, 6] across cut=0, 1, 2 so exec_tree_[4..6] get used without T=8 padding.
    constexpr int32_t kAnchor = 99;
    std::array<int32_t, 7> drafts{101, 102, 103, 104, 105, 106, 107};
    std::array<int32_t, 7 * kDf2TreeSelectorTopK> toptok{};
    std::array<float, 7 * kDf2TreeSelectorTopK * kDf2TreeSelectorTopK> seltab{};
    auto init_tables = [&](int cut, float runner_p, bool add_anchor_collision) {
        toptok.fill(-1);
        seltab.fill(-100.0f);
        for (int r = 0; r < 7; ++r) {
            toptok[static_cast<size_t>(r) * kDf2TreeSelectorTopK + 0] = drafts[static_cast<size_t>(r)];
            toptok[static_cast<size_t>(r) * kDf2TreeSelectorTopK + 1] = 200 + r;
            const size_t base = (static_cast<size_t>(r) * kDf2TreeSelectorTopK + 0u) * kDf2TreeSelectorTopK;
            seltab[base + 0] = 0.0f;
            if (r == cut) {
                if (add_anchor_collision) {
                    // Put colliding anchor token at slot 1 (higher logit) and distinct 200+r at slot 2.
                    toptok[static_cast<size_t>(r) * kDf2TreeSelectorTopK + 1] = kAnchor;
                    toptok[static_cast<size_t>(r) * kDf2TreeSelectorTopK + 2] = 200 + r;
                    const float rem = 1.0f - 2.0f * runner_p;
                    seltab[base + 0] = std::log(std::max(0.05f, rem) / runner_p);
                    seltab[base + 1] = 0.1f;  // colliding anchor slightly higher score
                    seltab[base + 2] = 0.0f;  // distinct runner at p ~= runner_p / (rem+exp(0.1)*runner_p+runner_p)
                } else {
                    const float p0 = 1.0f - runner_p;
                    seltab[base + 0] = std::log(p0 / runner_p);
                    seltab[base + 1] = 0.0f;
                }
            }
        }
    };

    auto check_var_t_band = [&](int cut, const std::array<float, 7>& probs, float runner_p, int expected_m,
                                bool add_collision, const char* label) {
        init_tables(cut, runner_p, add_collision);
        for (float min_bp : {0.30f, 0.35f}) {
            Df2TreeBuildOptions opts;
            opts.draft_conf = 0.7;
            opts.min_branch_p = df2_tree_effective_min_branch_p(min_bp, /*var_t_on=*/true, /*guard_on=*/true);
            opts.max_fanout = 2;
            opts.max_nodes = 8;
            opts.ev_cut = 0.35f;
            opts.dedup_siblings = true;
            opts.max_cut = 3;
            const Df2TreeSpec tree = df2_tree_build_from_selector_ex(kAnchor, drafts.data(), probs.data(), 7,
                                                                     toptok.data(), seltab.data(), opts);
            check(tree.valid() && !tree.is_linear() && tree.is_caterpillar(), label);
            check(tree.count == expected_m && tree.count >= 4 && tree.count <= 6, label);
            const int exec_t = df2_tree_select_width(tree.count, /*var_t_on=*/true);
            check(exec_t == expected_m && exec_t >= 4 && exec_t <= 6, label);
            int32_t gdn_order[8] = {}, is_spine[8] = {};
            check(df2_tree_caterpillar_gdn_order(tree, gdn_order, is_spine, exec_t), label);
        }
    };

    // cut=0: S1=0.55, S2=0.385, S3=0.173(<0.35), leaf(0)=0.36 -> M=4 (T=4), with colliding anchor stripped by DEDUP=1
    check_var_t_band(0, {0.55f, 0.70f, 0.45f, 0.35f, 0.30f, 0.30f, 0.30f}, 0.38f, 4, true,
                     "engagement ladder cut=0 with dedup=1 + ev_cut=0.35 lands at M=4 (T=4)");
    // cut=1: S1=0.90, S2=0.585, S3=0.468, S4=0.187(<0.35), leaf(1)=0.90*0.40=0.36 -> M=5 (T=5)
    check_var_t_band(1, {0.90f, 0.65f, 0.80f, 0.40f, 0.30f, 0.30f, 0.30f}, 0.40f, 5, false,
                     "engagement ladder cut=1 with ev_cut=0.35 lands at M=5 (T=5)");
    // cut=2: S1=0.92, S2=0.828, S3=0.538, S4=0.430, S5=0.172(<0.35), leaf(2)=0.828*0.43=0.356 -> M=6 (T=6)
    check_var_t_band(2, {0.92f, 0.90f, 0.65f, 0.80f, 0.40f, 0.30f, 0.30f}, 0.43f, 6, false,
                     "engagement ladder cut=2 with ev_cut=0.35 lands at M=6 (T=6)");

    // 3. Run 1 -12.3% trap check: VAR_T=0 with min_branch_p < 0.45 (and guard off) reproduces legacy behavior
    // bit-exactly and executes padded T=8; enabling the guard clamps to 0.45 and prevents the trap.
    init_tables(1, 0.38f, false);
    const std::array<float, 7> run1_probs{0.90f, 0.65f, 0.80f, 0.40f, 0.30f, 0.30f, 0.30f};
    Df2TreeBuildOptions trap_opts;
    trap_opts.draft_conf = 0.7;
    trap_opts.min_branch_p = df2_tree_effective_min_branch_p(0.30f, /*var_t_on=*/false, /*guard_on=*/false);
    trap_opts.max_fanout = 2;
    trap_opts.max_nodes = 8;
    const Df2TreeSpec t_trap = df2_tree_build_from_selector_ex(kAnchor, drafts.data(), run1_probs.data(), 7,
                                                               toptok.data(), seltab.data(), trap_opts);
    const Df2TreeSpec t_legacy = df2_tree_build_from_selector(kAnchor, drafts.data(), run1_probs.data(), 7,
                                                              toptok.data(), seltab.data(), 0.7, 0.30f, 2, 8);
    check(!t_trap.is_linear() && t_trap.count == t_legacy.count && t_trap.token == t_legacy.token &&
              t_trap.parent == t_legacy.parent &&
              df2_tree_select_width(t_trap.count, /*var_t_on=*/false) == 8,
          "VAR_T=0 with min_branch_p=0.30 (guard off) reproduces legacy tree bit-exactly at fixed T=8");

    Df2TreeBuildOptions guarded_opts = trap_opts;
    guarded_opts.min_branch_p = df2_tree_effective_min_branch_p(0.30f, /*var_t_on=*/false, /*guard_on=*/true);
    const Df2TreeSpec t_guarded = df2_tree_build_from_selector_ex(kAnchor, drafts.data(), run1_probs.data(), 7,
                                                                  toptok.data(), seltab.data(), guarded_opts);
    check(t_guarded.is_linear(),
          "VAR_T=0 with guard on clamps min_branch_p=0.30 -> 0.45 and suppresses sub-0.45 branch");
}

void test_sampled_tree_gate_penalties_and_q_helpers() {
    // 1. STRATA_DF2_TREE_SAMPLED env gate parser + default off
    check(!df2_tree_sampled_env_enabled(nullptr) && !df2_tree_sampled_env_enabled("0") &&
              !df2_tree_sampled_env_enabled("true") && df2_tree_sampled_env_enabled("1"),
          "sampled tree env gate parser");
    check(!df2_tree_sampled_on(), "sampled tree gate defaults OFF");

    // 2. penalty_tree_rows mirrors linear penalty_rows along root-to-node ancestor chain
    // Tree: 0(99) -> 1(10) -> 2(11), fork 0(99) -> 3(20), 1(10) -> 4(30)
    Df2TreeSpec tree;
    tree.count = 5;
    tree.token = {99, 10, 11, 20, 30, 0, 0, 0};
    tree.parent = {kDf2TreeNoParent, 0, 1, 0, 1, kDf2TreeNoParent, kDf2TreeNoParent, kDf2TreeNoParent};
    tree.depth = {0, 1, 2, 1, 2, 0, 0, 0};
    tree.slot_rank = {0, 0, 0, 1, 1, 0, 0, 0};
    tree.shape = Df2TreeShape::Caterpillar;
    tree.build_anc();

    const int32_t tail[3] = {7, 8, 9};
    constexpr int kT = 6;
    constexpr int kH = 4;
    std::array<int32_t, kT * kH> rows{};
    penalty_tree_rows(tail, 3, tree, kT, kH, rows.data());
    // row 0: [7, 8, 9, 99]
    // row 1: [8, 9, 99, 10]
    // row 2: [9, 99, 10, 11]
    // row 3: [8, 9, 99, 20]
    // row 4: [9, 99, 10, 30]
    // row 5 (padding): [-1, -1, -1, -1]
    const std::array<int32_t, kT * kH> expected{
        7, 8, 9, 99,
        8, 9, 99, 10,
        9, 99, 10, 11,
        8, 9, 99, 20,
        9, 99, 10, 30,
        -1, -1, -1, -1,
    };
    check(rows == expected, "penalty_tree_rows follows exact per-node ancestor chain");

    // 3. df2_tree_populate_sampled_q handles sibling exhaustion by pruning cleanly while preserving validity.
    const int32_t toptok[7 * kDf2TreeSelectorTopK] = {
        10, 99, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        11, 21, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    };
    std::array<float, 7 * kDf2TreeSelectorTopK * kDf2TreeSelectorTopK> seltab{};
    seltab.fill(-INFINITY);
    seltab[0] = 0.0f;
    seltab[1] = -0.2f;
    seltab[(1 * kDf2TreeSelectorTopK + 0) * kDf2TreeSelectorTopK + 0] = 0.0f;
    seltab[(1 * kDf2TreeSelectorTopK + 0) * kDf2TreeSelectorTopK + 1] = -0.5f;
    const int32_t drafts[2] = {10, 11};
    // Build a fork tree whose runner-up at root collides with anchor when dedup=true.
    Df2TreeSpec t_prune;
    t_prune.count = 4;
    t_prune.token = {99, 10, 11, 99};
    t_prune.parent = {kDf2TreeNoParent, 0, 1, 0};
    t_prune.depth = {0, 1, 2, 1};
    t_prune.slot_rank = {0, 0, 0, 1};
    t_prune.shape = Df2TreeShape::PrefixFanout;
    t_prune.branch_node = 0;
    t_prune.build_anc();
    // Mark duplicate root child? Wait: unique siblings check passes if child 3 is token 11 (`drafts[1]`, valid for dedup test!).
    t_prune.token[3] = 11;
    Df2TreeSampleQ sq;
    check(df2_tree_populate_sampled_q(t_prune, 99, drafts, 2, toptok, seltab.data(), /*dedup=*/true,
                                      0x1234ULL, 5ULL, sq) &&
              t_prune.valid() && t_prune.is_linear() && t_prune.count == 3,
          "populate_sampled_q prunes sibling with 0 non-excluded candidates under dedup=true");

    // 4. Zero-residual rounding guard (fixes hy4_spec.hip.h:430 bug): if residual p_res <= q everywhere
    //    when testing child c_k, accept c_k cleanly instead of resetting residual mass.
    const int32_t fork_b[2] = {10, 20};
    const Df2TreeSpec t_guard = make_prefix_fanout_tree(99, nullptr, 0, fork_b, 2);
    const int32_t p_ids[2] = {10, 20};
    const double p0_vec[2] = {0.5, 0.5};
    const double p_child[2] = {0.5, 0.5};
    const strata::core::SpecP pv_guard[3] = {
        {p_ids, p0_vec, 2}, {p_ids, p_child, 2}, {p_ids, p_child, 2}};
    Df2TreeSampleQ sq_guard;
    sq_guard.node[1].n = 2;
    sq_guard.node[1].id[0] = 10; sq_guard.node[1].id[1] = 20;
    sq_guard.node[1].q[0] = 0.99f; sq_guard.node[1].q[1] = 0.99f;  // covers all residual mass -> next_z == 0
    for (int pos = 0; pos < 16; ++pos) {
        int32_t bonus = -1;
        int64_t resamp = 0;
        const auto w_g = df2_tree_sample_accept(t_guard, pv_guard, &sq_guard, 0x777ULL, (uint64_t) pos,
                                                &bonus, &resamp);
        check(w_g.n_keep == 2 && w_g.path[1] == 1 && resamp == 0,
              "zero-residual rounding guard accepts covered child instead of resetting residual");
    }
}

}  // namespace

int main() {
    test_cut_depth();
    test_builders_and_invariants();
    test_paths_and_conv_ancestors();
    test_topological_order_and_canonicalize();
    test_acceptance_walks_and_branch_kill();
    test_serialization_and_telemetry_roundtrip();
    test_env_gate_and_selector_builder();
    test_ev_budget_ordering_and_cut_drop();
    test_adaptive_cut_choice();
    test_sibling_dedup();
    test_extended_telemetry_and_runner_hit();
    test_gdn_order_ple_taps_and_remap_invariants();
    test_var_t_width_ladder();
    test_spine_skip_predicate();
    test_composable_ev_adaptive_matrix_and_no_inflation();
    test_max_cut_and_row_ema_cost_gates();
    test_engagement_shaping_envelope();
    test_sampled_tree_gate_penalties_and_q_helpers();

    // P4: q_T temperature scaling, logit-shift invariance, logit_bias, and STRATA_DF2_QT_RAW helper tests.
    {
        check(!df2_qt_raw_env_enabled(nullptr) && !df2_qt_raw_env_enabled("") &&
                  !df2_qt_raw_env_enabled("0") && df2_qt_raw_env_enabled("1"),
              "STRATA_DF2_QT_RAW parser defaults OFF and enables on 1");
        check(df2_effective_q_temp(0.6f, 0.5f, false) > 0.299f && df2_effective_q_temp(0.6f, 0.5f, false) < 0.301f,
              "df2_effective_q_temp scales by spec_draft_temp_scale");
        check(df2_effective_q_temp(0.6f, 0.5f, true) == 1.0f &&
                  df2_effective_q_temp(0.0f, 1.0f, false) == 1.0f &&
                  df2_effective_q_temp(-0.5f, 1.0f, false) == 1.0f,
              "df2_effective_q_temp guards raw_qt=true and non-positive temperature to 1.0f");

        const int32_t toptok[7 * kDf2TreeSelectorTopK] = {
            10, 20, 30, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            11, 21, 31, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        };
        std::array<float, 7 * kDf2TreeSelectorTopK * kDf2TreeSelectorTopK> base_s{}, shifted_s{};
        base_s.fill(-INFINITY);
        shifted_s.fill(-INFINITY);
        const float raw0[3] = {2.0f, 1.4f, 0.1f};
        const float raw1[3] = {1.5f, 1.0f, -0.5f};
        for (int j = 0; j < 3; ++j) {
            base_s[j] = raw0[j];
            shifted_s[j] = raw0[j] + 17.25f;
            for (int prev = 0; prev < 3; ++prev) {
                base_s[(1 * kDf2TreeSelectorTopK + prev) * kDf2TreeSelectorTopK + j] = raw1[j];
                shifted_s[(1 * kDf2TreeSelectorTopK + prev) * kDf2TreeSelectorTopK + j] = raw1[j] - 42.5f;
            }
        }
        const int32_t drafts[2] = {10, 11};
        const Df2TreeSpec tree = make_prefix_fanout_tree(99, nullptr, 0, toptok, 2);

        for (float T : {0.3f, 1.0f, 1.5f}) {
            Df2TreeSpec t_base = tree, t_shift = tree;
            Df2TreeSampleQ q_base, q_shift;
            check(df2_tree_populate_sampled_q(t_base, 99, drafts, 2, toptok, base_s.data(), false, 0x11ULL, 3ULL,
                                              q_base, T),
                  "populate_sampled_q base succeeds");
            check(df2_tree_populate_sampled_q(t_shift, 99, drafts, 2, toptok, shifted_s.data(), false, 0x11ULL, 3ULL,
                                              q_shift, T),
                  "populate_sampled_q shifted succeeds");
            check(q_base.node[1].n == 3 && q_shift.node[1].n == 3, "q_T preserves support count");
            for (int k = 0; k < 3; ++k) {
                check(std::fabs(q_base.node[1].q[k] - q_shift.node[1].q[k]) < 1e-6f,
                      "df2_tree_populate_sampled_q q_T is invariant under constant logit shifts");
            }
            std::array<detail::Df2SelectorCand, kDf2TreeSelectorTopK> r_base{}, r_shift{};
            const int nb = detail::extract_selector_runners(0, 99, drafts, 2, toptok, base_s.data(), 0.01f, false,
                                                            r_base, true, T);
            const int ns = detail::extract_selector_runners(0, 99, drafts, 2, toptok, shifted_s.data(), 0.01f, false,
                                                            r_shift, true, T);
            check(nb == ns && nb > 0 && std::fabs(r_base[0].p - r_shift[0].p) < 1e-6f,
                  "extract_selector_runners q_T is invariant under constant logit shifts");
        }

        // Verify temperature sharpening and logit_bias (-inf ban).
        Df2TreeSpec t_hot = tree, t_cold = tree, t_bias = tree;
        Df2TreeSampleQ q_hot, q_cold, q_bias;
        df2_tree_populate_sampled_q(t_cold, 99, drafts, 2, toptok, base_s.data(), false, 0x11ULL, 3ULL, q_cold, 0.3f);
        df2_tree_populate_sampled_q(t_hot, 99, drafts, 2, toptok, base_s.data(), false, 0x11ULL, 3ULL, q_hot, 1.5f);
        check(q_cold.node[1].q[0] > q_hot.node[1].q[0], "lower temperature sharpens q_T primary probability");

        std::array<float, 64> bias{};
        bias[20] = -INFINITY;  // ban runner-up token 20
        df2_tree_populate_sampled_q(t_bias, 99, drafts, 2, toptok, base_s.data(), false, 0x11ULL, 3ULL, q_bias, 0.7f,
                                    bias.data(), static_cast<int>(bias.size()));
        check(q_bias.node[1].n == 2 && q_bias.node[1].id[0] == 10 && q_bias.node[1].id[1] == 30,
              "logit_bias -inf ban removes candidate from q_T support");
    }

    // P5: Rejection-reason classification in df2_tree_build_from_selector_ex (Cut, MinP, Ev, Dedup).
    {
        std::array<int32_t, 7> drafts{101, 102, 103, 104, 105, 106, 107};
        std::array<float, 7> probs{0.95f, 0.90f, 0.85f, 0.40f, 0.40f, 0.40f, 0.40f};
        std::array<int32_t, 7 * kDf2TreeSelectorTopK> toptok{};
        std::array<float, 7 * kDf2TreeSelectorTopK * kDf2TreeSelectorTopK> seltab{};
        toptok.fill(-1);
        seltab.fill(-100.0f);
        for (int r = 0; r < 7; ++r) {
            toptok[static_cast<size_t>(r) * kDf2TreeSelectorTopK + 0] = drafts[static_cast<size_t>(r)];
            toptok[static_cast<size_t>(r) * kDf2TreeSelectorTopK + 1] = 200 + r;
            seltab[(static_cast<size_t>(r) * kDf2TreeSelectorTopK + 0) * kDf2TreeSelectorTopK + 0] = 2.0f;
            seltab[(static_cast<size_t>(r) * kDf2TreeSelectorTopK + 0) * kDf2TreeSelectorTopK + 1] = 0.0f;
        }
        Df2TreeBuildOptions ropts;
        ropts.draft_conf = 0.7;
        ropts.min_branch_p = 0.25f;
        ropts.max_fanout = 2;
        ropts.max_nodes = 8;
        ropts.max_cut = 1;  // conf_cut=3 > max_cut=1 -> Cut
        Df2TreeRejReason rej = Df2TreeRejReason::None;
        (void) df2_tree_build_from_selector_ex(99, drafts.data(), probs.data(), 7, toptok.data(), seltab.data(),
                                               ropts, nullptr, &rej);
        check(rej == Df2TreeRejReason::Cut, "max_cut exceeded records Cut rejection reason");

        ropts.max_cut = -1;
        // At cut=3, exp(2.0)/(exp(2.0)+1) ~ 0.88 vs runner ~ 0.12 < min_branch_p=0.25 -> MinP
        (void) df2_tree_build_from_selector_ex(99, drafts.data(), probs.data(), 7, toptok.data(), seltab.data(),
                                               ropts, nullptr, &rej);
        check(rej == Df2TreeRejReason::MinP, "runner below min_branch_p records MinP rejection reason");

        // Boost runner at cut=3 to p ~ 0.45 >= 0.25, then apply ev_cut=0.50 -> Ev
        const size_t base3 = (3u * kDf2TreeSelectorTopK + 0u) * kDf2TreeSelectorTopK;
        seltab[base3 + 0] = 0.2f;
        seltab[base3 + 1] = 0.0f;
        ropts.ev_cut = 0.50f;
        (void) df2_tree_build_from_selector_ex(99, drafts.data(), probs.data(), 7, toptok.data(), seltab.data(),
                                               ropts, nullptr, &rej);
        check(rej == Df2TreeRejReason::Ev, "runners surviving min_branch_p but dropped by ev_cut record Ev");

        // Colliding runner with dedup_siblings=true -> Dedup
        ropts.ev_cut = 0.0f;
        ropts.dedup_siblings = true;
        toptok[3 * kDf2TreeSelectorTopK + 1] = 99;  // anchor collision
        (void) df2_tree_build_from_selector_ex(99, drafts.data(), probs.data(), 7, toptok.data(), seltab.data(),
                                               ropts, nullptr, &rej);
        check(rej == Df2TreeRejReason::Dedup, "runner stripped by dedup_siblings records Dedup");
    }

    // P8: Hygiene bundle tests (56-bit counter bounds, invariant pool env, Gumbel conflict guard).
    {
        using strata::core::kTreeBranchMaxNodeIdx;
        using strata::core::kTreeBranchMaxRounds;
        using strata::core::tree_branch_accept_counter;
        using strata::core::tree_branch_counter_in_bounds;
        using strata::core::tree_branch_draft_counter_checked;
        using strata::core::tree_branch_round_in_bounds;

        check(tree_branch_round_in_bounds(0ull, 0ull), "round=0, pos=0 is in bounds");
        check(tree_branch_round_in_bounds((1ull << 32) - 1ull, kTreeBranchMaxRounds - 1ull),
              "max 24-bit round and 32-bit pos are in bounds");
        check(!tree_branch_round_in_bounds(0ull, kTreeBranchMaxRounds), "round=1<<24 wraps 56-bit key");
        check(!tree_branch_round_in_bounds(0ull, ~0ull), "UINT64_MAX round rejected");
        check(!tree_branch_round_in_bounds(1ull << 56, 0ull), "pos >= 1<<56 rejected");

        check(tree_branch_counter_in_bounds((1ull << 56) - 1ull, 0) &&
                  tree_branch_counter_in_bounds((1ull << 56) - 1ull, kTreeBranchMaxNodeIdx),
              "56-bit max key and 5-bit node_idx [0..31] are in bounds");
        check(!tree_branch_counter_in_bounds(1ull << 56, 1) &&
                  !tree_branch_counter_in_bounds(0ull, -1) &&
                  !tree_branch_counter_in_bounds(0ull, 32),
              "56-bit key overflow and out-of-range node_idx rejected");

        uint64_t ctr = 0;
        check(tree_branch_draft_counter_checked(0x123456ull, 5, ctr) && (ctr >> 61) == 0x5ull &&
                  ((ctr >> 56) & 0x1Full) == 5ull &&
                  ctr != tree_branch_accept_counter(0x123456ull, 5),
              "checked draft counter encodes tag 0x5 + node_idx disjoint from accept tag 0x3");
        check(!tree_branch_draft_counter_checked(1ull << 56, 1, ctr),
              "checked draft counter refuses 56-bit key wrap");

        // df2_tree_populate_sampled_q refuses 56-bit key overflow rather than wrapping.
        const int32_t toptok[7 * kDf2TreeSelectorTopK] = {
            10, 20, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        };
        std::array<float, 7 * kDf2TreeSelectorTopK * kDf2TreeSelectorTopK> seltab{};
        seltab.fill(-INFINITY);
        seltab[0] = 0.0f;
        seltab[1] = -0.2f;
        const int32_t drafts[1] = {10};
        const int32_t br[2] = {10, 20};
        Df2TreeSpec t_wrap = make_prefix_fanout_tree(99, nullptr, 0, br, 2);
        Df2TreeSampleQ sq_wrap;
        check(!df2_tree_populate_sampled_q(t_wrap, 99, drafts, 1, toptok, seltab.data(), false, 0x1ULL,
                                           (kTreeBranchMaxRounds << 32), sq_wrap),
              "populate_sampled_q fails safely on 56-bit counter wrap");

        check(!df2_tree_invariant_pool_env_enabled(nullptr) &&
                  !df2_tree_invariant_pool_env_enabled("") &&
                  !df2_tree_invariant_pool_env_enabled("0") &&
                  df2_tree_invariant_pool_env_enabled("1"),
              "STRATA_DF2_TREE_INVARIANT_POOL defaults OFF and enables on 1");
        check(!spec_gumbel_env_enabled(nullptr) && !spec_gumbel_env_enabled("") &&
                  !spec_gumbel_env_enabled("0") && spec_gumbel_env_enabled("1"),
              "STRATA_SPEC_GUMBEL env parser matches sampler.cu env_flag");
        check(!df2_tree_sampled_conflict_with_gumbel(false, false) &&
                  !df2_tree_sampled_conflict_with_gumbel(true, false) &&
                  !df2_tree_sampled_conflict_with_gumbel(false, true) &&
                  df2_tree_sampled_conflict_with_gumbel(true, true) &&
                  df2_tree_sampled_conflict_with_gumbel(true, true, false) &&
                  !df2_tree_sampled_conflict_with_gumbel(true, true, true),
              "Gumbel + sampled-tree conflict predicate triggers iff both are enabled and TREE_GM is off");
    }

    // P2: Zero-sync GPU top-2 pre-gate & direct top-2 builder parity tests.
    {
        check(!df2_tree_pregrid_env_enabled(nullptr) &&
                  !df2_tree_pregrid_env_enabled("") &&
                  !df2_tree_pregrid_env_enabled("0") &&
                  !df2_tree_pregrid_env_enabled("true") &&
                  df2_tree_pregrid_env_enabled("1"),
              "STRATA_DF2_TREE_PREGRID parser defaults OFF and enables on 1");
        const char* pre_env = std::getenv("STRATA_DF2_TREE_PREGRID");
        check(df2_tree_pregrid_on() == df2_tree_pregrid_env_enabled(pre_env),
              "df2_tree_pregrid_on matches STRATA_DF2_TREE_PREGRID env");

        const int32_t top2_tok[7] = {200, 201, 202, 203, 204, 205, 206};
        const float top2_prob[7] = {0.10f, 0.20f, 0.34f, 0.48f, 0.15f, 0.05f, 0.05f};
        check(!df2_tree_pregrid_passes(top2_tok, top2_prob, 7, 0.50f, -1, 0.0f),
              "pregrid rejects when all runner-up probs < min_branch_p");
        check(df2_tree_pregrid_passes(top2_tok, top2_prob, 7, 0.35f, -1, 0.0f),
              "pregrid passes when step 3 prob 0.48 >= 0.35");
        check(!df2_tree_pregrid_passes(top2_tok, top2_prob, 7, 0.35f, 2, 0.0f),
              "pregrid respects max_cut=2 and ignores step 3 prob 0.48");
        check(df2_tree_pregrid_passes(top2_tok, top2_prob, 7, 0.34005f, 2, kDf2TreePregridFetchEps),
              "pregrid fetch eps prevents boundary false negative");
        const int32_t neg_tok[7] = {-1, -1, -1, -1, -1, -1, -1};
        check(!df2_tree_pregrid_passes(neg_tok, top2_prob, 7, 0.05f, -1, 0.0f),
              "pregrid rejects invalid (-1) runner-up tokens");

        const std::array<int32_t, 7> drafts{101, 102, 103, 104, 105, 106, 107};
        Df2TreeBuildOptions opts;
        opts.max_fanout = 2;
        opts.min_branch_p = 0.30f;
        opts.dedup_siblings = true;
        check(df2_tree_pregrid_can_use_direct_top2(99, drafts.data(), 7, top2_tok, top2_prob, opts),
              "direct top-2 builder eligible for max_fanout=2 non-walk without dedup collision");
        opts.max_fanout = 3;
        check(!df2_tree_pregrid_can_use_direct_top2(99, drafts.data(), 7, top2_tok, top2_prob, opts),
              "direct top-2 builder defers to seltab when max_fanout > 2");
        opts.max_fanout = 2;
        opts.sample_walk = true;
        check(!df2_tree_pregrid_can_use_direct_top2(99, drafts.data(), 7, top2_tok, top2_prob, opts),
              "direct top-2 builder defers when sample_walk=true");
        opts.sample_walk = false;
        opts.q_temperature = 0.7f;
        check(!df2_tree_pregrid_can_use_direct_top2(99, drafts.data(), 7, top2_tok, top2_prob, opts),
              "direct top-2 builder defers when q_temperature != 1.0f");
        opts.q_temperature = 1.0f;
        const int32_t coll_tok[7] = {200, 99, 202, 203, 204, 205, 206};
        const float coll_prob[7] = {0.10f, 0.40f, 0.10f, 0.10f, 0.10f, 0.10f, 0.10f};
        check(!df2_tree_pregrid_can_use_direct_top2(99, drafts.data(), 7, coll_tok, coll_prob, opts),
              "direct top-2 builder defers when qualifying rank-2 collides under dedup_siblings");

        // Parity between df2_tree_build_from_top2_ex and df2_tree_build_from_selector_ex when max_fanout == 2.
        std::array<int32_t, 7 * kDf2TreeSelectorTopK> toptok{};
        std::array<float, 7 * kDf2TreeSelectorTopK * kDf2TreeSelectorTopK> seltab{};
        toptok.fill(-1);
        seltab.fill(-100.0f);
        const std::array<float, 7> probs{0.90f, 0.65f, 0.80f, 0.45f, 0.40f, 0.40f, 0.40f};
        std::array<int32_t, 7> exact_tok2{};
        std::array<float, 7> exact_p2{};
        for (int r = 0; r < 7; ++r) {
            toptok[static_cast<size_t>(r) * kDf2TreeSelectorTopK + 0] = drafts[static_cast<size_t>(r)];
            toptok[static_cast<size_t>(r) * kDf2TreeSelectorTopK + 1] = 300 + r;
            toptok[static_cast<size_t>(r) * kDf2TreeSelectorTopK + 2] = 400 + r;
            const size_t base = (static_cast<size_t>(r) * kDf2TreeSelectorTopK + 0u) * kDf2TreeSelectorTopK;
            const float p2 = (r == 1) ? 0.40f : (r == 2 ? 0.36f : 0.12f);
            const float p3 = 0.05f;
            const float p1 = 1.0f - p2 - p3;
            seltab[base + 0] = std::log(p1 / p2);
            seltab[base + 1] = 0.0f;
            seltab[base + 2] = std::log(p3 / p2);
            exact_tok2[static_cast<size_t>(r)] = 300 + r;
            exact_p2[static_cast<size_t>(r)] = p2;
        }
        Df2TreeAdaptivePolicy adapt;
        adapt.observe_linear(5, 3);
        for (float ev : {0.0f, 0.35f}) {
            for (bool ac : {false, true}) {
                for (int mc : {-1, 1, 3}) {
                    Df2TreeBuildOptions popts;
                    popts.draft_conf = 0.7;
                    popts.min_branch_p = 0.30f;
                    popts.max_fanout = 2;
                    popts.max_nodes = 8;
                    popts.ev_cut = ev;
                    popts.adaptive_cut = ac;
                    popts.max_cut = mc;
                    Df2TreeRejReason rej_sel = Df2TreeRejReason::None;
                    Df2TreeRejReason rej_top2 = Df2TreeRejReason::None;
                    const Df2TreeSpec t_sel = df2_tree_build_from_selector_ex(
                        99, drafts.data(), probs.data(), 7, toptok.data(), seltab.data(), popts,
                        ac ? &adapt : nullptr, &rej_sel);
                    const Df2TreeSpec t_top2 = df2_tree_build_from_top2_ex(
                        99, drafts.data(), probs.data(), 7, exact_tok2.data(), exact_p2.data(), popts,
                        ac ? &adapt : nullptr, &rej_top2);
                    check(rej_sel == rej_top2 && t_sel.count == t_top2.count &&
                              t_sel.token == t_top2.token && t_sel.parent == t_top2.parent &&
                              t_sel.depth == t_top2.depth && t_sel.slot_rank == t_top2.slot_rank &&
                              t_sel.shape == t_top2.shape && t_sel.branch_node == t_top2.branch_node,
                          "df2_tree_build_from_top2_ex matches df2_tree_build_from_selector_ex across policy combos");
                }
            }
        }
    }

    // P6: Temperature- and mode-aware tree gating (MAX_TEMP, NO_WALK_TREES, EV_CUT_SAMP, MIN_BRANCH_GUARD interplay).
    {
        check(df2_tree_parse_max_temp(nullptr) == 0.0f &&
                  df2_tree_parse_max_temp("") == 0.0f &&
                  df2_tree_parse_max_temp("0") == 0.0f &&
                  df2_tree_parse_max_temp("-0.5") == 0.0f &&
                  df2_tree_parse_max_temp("bad") == 0.0f &&
                  std::fabs(df2_tree_parse_max_temp("0.7") - 0.7f) < 1e-6f,
              "STRATA_DF2_TREE_MAX_TEMP parser validates positive finite thresholds");
        check(!df2_tree_no_walk_trees_env_enabled(nullptr) &&
                  !df2_tree_no_walk_trees_env_enabled("0") &&
                  df2_tree_no_walk_trees_env_enabled("1"),
              "STRATA_DF2_TREE_NO_WALK_TREES parser defaults OFF and enables on 1");
        check(df2_tree_parse_ev_cut_samp(nullptr) == 0.0f &&
                  df2_tree_parse_ev_cut_samp("0") == 0.0f &&
                  df2_tree_parse_ev_cut_samp("1.5") == 0.0f &&
                  std::fabs(df2_tree_parse_ev_cut_samp("0.25") - 0.25f) < 1e-6f,
              "STRATA_DF2_TREE_EV_CUT_SAMP parser validates [0, 1]");
        check(std::fabs(df2_tree_max_temp() - df2_tree_parse_max_temp(std::getenv("STRATA_DF2_TREE_MAX_TEMP"))) < 1e-6f,
              "df2_tree_max_temp matches env");
        check(df2_tree_no_walk_trees_on() == df2_tree_no_walk_trees_env_enabled(std::getenv("STRATA_DF2_TREE_NO_WALK_TREES")),
              "df2_tree_no_walk_trees_on matches env");
        check(std::fabs(df2_tree_ev_cut_samp() -
                        df2_tree_parse_ev_cut_samp(std::getenv("STRATA_DF2_TREE_EV_CUT_SAMP"))) < 1e-6f,
              "df2_tree_ev_cut_samp matches env");

        check(df2_tree_sampled_request_gate(false, false, 0.7f, false, 0.7f, false) ==
                  Df2TreeSampledGateStatus::Disabled,
              "sampled gate returns Disabled when STRATA_DF2_TREE_SAMPLED=0");
        check(df2_tree_sampled_request_gate(true, true, 0.0f, false, 0.7f, false) ==
                  Df2TreeSampledGateStatus::Greedy,
              "sampled gate returns Greedy for greedy requests");
        check(df2_tree_sampled_request_gate(true, false, 0.7f, false, 0.7f, true) ==
                  Df2TreeSampledGateStatus::Allowed,
              "sampled gate allows point-mass T <= max_temp even with NO_WALK_TREES=1");
        check(df2_tree_sampled_request_gate(true, false, 0.9f, false, 0.7f, false) ==
                  Df2TreeSampledGateStatus::MaxTempExceeded,
              "sampled gate skips sampled-tree entry when temperature > MAX_TEMP");
        check(df2_tree_sampled_request_gate(true, false, 0.7f, true, 0.7f, true) ==
                  Df2TreeSampledGateStatus::WalkTreesDisabled,
              "sampled gate disables trees under DF2_SAMPLE_WALK=1 when NO_WALK_TREES=1");
        check(df2_tree_sampled_request_gate(true, false, 0.9f, true, 0.0f, false) ==
                  Df2TreeSampledGateStatus::Allowed,
              "sampled gate allows DF2_SAMPLE_WALK=1 trees when NO_WALK_TREES=0 and MAX_TEMP=0");

        // EV_CUT_SAMP inheritance + explicit MIN_BRANCH_GUARD / ev_cut interplay.
        const auto env_inherit = df2_tree_resolve_sampled_envelope(0.30f, 0.35f, 0.0f, true, false);
        check(std::fabs(env_inherit.ev_cut - 0.35f) < 1e-6f &&
                  std::fabs(env_inherit.min_branch_p - 0.35f) < 1e-6f &&
                  env_inherit.ev_shadow_clamped && !env_inherit.guard_clamped,
              "ev_cut_samp=0 inherits ev_cut and explicitly raises min_branch_p to prevent silent shadowing");

        const auto env_samp = df2_tree_resolve_sampled_envelope(0.10f, 0.35f, 0.25f, true, true);
        check(std::fabs(env_samp.ev_cut - 0.25f) < 1e-6f &&
                  std::fabs(env_samp.min_branch_p - 0.25f) < 1e-6f &&
                  env_samp.ev_shadow_clamped && !env_samp.guard_clamped,
              "ev_cut_samp=0.25 overrides ev_cut=0.35 and prevents silent shadowing at VAR_T=1");

        const auto env_guard = df2_tree_resolve_sampled_envelope(0.20f, 0.35f, 0.25f, false, true);
        check(std::fabs(env_guard.ev_cut - 0.25f) < 1e-6f &&
                  std::fabs(env_guard.min_branch_p - kDf2TreeSafeMinBranchP) < 1e-6f &&
                  env_guard.guard_clamped && env_guard.ev_below_guard && !env_guard.ev_shadow_clamped,
              "VAR_T=0 with MIN_BRANCH_GUARD=1 clamps sampled min_branch_p=0.45 and flags ev_below_guard");

        const auto env_ok = df2_tree_resolve_sampled_envelope(0.30f, 0.35f, 0.25f, true, true);
        check(std::fabs(env_ok.ev_cut - 0.25f) < 1e-6f &&
                  std::fabs(env_ok.min_branch_p - 0.30f) < 1e-6f &&
                  !env_ok.ev_shadow_clamped && !env_ok.guard_clamped && !env_ok.ev_below_guard,
              "min_branch_p >= ev_cut_samp with VAR_T=1 leaves both thresholds untouched");
    }

    // P7: GPU accept staging/unpacking helpers & STRATA_DF2_TREE_GPU_ACCEPT gate tests.
    {
        check(!df2_tree_gpu_accept_env_enabled(nullptr) &&
                  !df2_tree_gpu_accept_env_enabled("") &&
                  !df2_tree_gpu_accept_env_enabled("0") &&
                  !df2_tree_gpu_accept_env_enabled("true") &&
                  df2_tree_gpu_accept_env_enabled("1"),
              "STRATA_DF2_TREE_GPU_ACCEPT parser defaults OFF and enables on 1");
        check(df2_tree_gpu_accept_on() == df2_tree_gpu_accept_env_enabled(std::getenv("STRATA_DF2_TREE_GPU_ACCEPT")),
              "df2_tree_gpu_accept_on matches env");
        static_assert(sizeof(strata::core::TreeSampleAcceptResult) == 16,
                      "TreeSampleAcceptResult must stay packed 16 bytes");

        const int32_t prefix[2] = {10, 11};
        const int32_t br[2] = {20, 21};
        const int32_t tail[1] = {30};
        const Df2TreeSpec tree = make_prefix_fanout_tree(99, prefix, 2, br, 2, tail, 1);
        Df2TreeSampleQ sq;
        sq.node[3].n = 2;
        sq.node[3].id[0] = 20;
        sq.node[3].id[1] = 21;
        sq.node[3].q[0] = 0.7f;
        sq.node[3].q[1] = 0.3f;

        strata::core::TreeSampleAcceptDesc desc{};
        strata::core::TreeSampleAcceptQ q_pod{};
        check(df2_tree_pack_gpu_accept_input(tree, &sq, 0xCAFEBABEULL, 1234ULL, desc, &q_pod),
              "df2_tree_pack_gpu_accept_input succeeds for valid tree");
        check(desc.seed == 0xCAFEBABEULL && desc.pos0 == 1234ULL && desc.count == tree.count &&
                  desc.token[0] == 99 && desc.parent[0] == kDf2TreeNoParent &&
                  desc.parent[4] == 2 && desc.slot_rank[4] == 1 &&
                  q_pod.n[0] == -1 && q_pod.n[3] == 2 && q_pod.id[3][1] == 21 &&
                  std::fabs(q_pod.q[3][1] - 0.3f) < 1e-6f,
              "df2_tree_pack_gpu_accept_input copies tree topology and Q rows faithfully");

        strata::core::TreeSampleAcceptResult res{};
        res.path[0] = 0;
        res.path[1] = 1;
        res.path[2] = 2;
        res.path[3] = 4;
        res.n_keep = 4;
        res.flags = 3u;
        res.resamples = 1;
        res.bonus_token = 777;
        int32_t bonus = -1;
        int64_t resamp = -1;
        const Df2TreeCommitWalk walk = df2_tree_walk_from_gpu_result(res, &bonus, &resamp);
        check(walk.n_keep == 4 && walk.path[0] == 0 && walk.path[3] == 4 &&
                  walk.branch_hit && walk.runner_up_hit && bonus == 777 && resamp == 1,
              "df2_tree_walk_from_gpu_result unpacks 16-byte result into Df2TreeCommitWalk");
    }

    // P3 Commit 1: Host gumbel_exp_host + q_T top-2 Gumbel score helper + photo-finish branch-gate tests.
    {
        std::array<int32_t, kDf2TreeSelectorTopK> toks{};
        toks.fill(-1);
        std::array<float, kDf2TreeSelectorTopK> scores{};
        std::array<float, kDf2TreeSelectorTopK> shifted{};
        scores.fill(-INFINITY);
        shifted.fill(-INFINITY);
        toks[0] = 10; toks[1] = 20; toks[2] = 30;
        scores[0] = std::log(0.45f); scores[1] = std::log(0.40f); scores[2] = std::log(0.15f);
        for (int j = 0; j < 3; ++j) shifted[static_cast<size_t>(j)] = scores[static_cast<size_t>(j)] + 17.5f;

        const auto r0 = df2_gm_row_top2(toks.data(), scores.data(), 0x51A7ULL, 11ULL, 0.7f);
        const auto r1 = df2_gm_row_top2(toks.data(), shifted.data(), 0x51A7ULL, 11ULL, 0.7f);
        check(r0.valid && r1.valid && r0.has_runner && r0.tok0 != r0.tok1 &&
                  r0.tok0 == r1.tok0 && r0.tok1 == r1.tok1 &&
                  std::fabs(r0.delta_g - r1.delta_g) < 1e-6 &&
                  std::fabs(r0.q0 - r1.q0) < 1e-6f &&
                  std::fabs(r0.max_q - r1.max_q) < 1e-6f,
              "df2_gm_row_top2 is invariant to constant logit shifts");

        const float dg = static_cast<float>(r0.delta_g);
        check(df2_gm_photo_finish_fires(r0, dg + 1e-3f) &&
                  !df2_gm_photo_finish_fires(r0, std::max(0.0f, dg - 1e-3f)) &&
                  !df2_gm_photo_finish_fires(r0, 0.0f),
              "df2_gm_photo_finish_fires enforces strict Δg < delta boundary");

        // Single-candidate row has no runner-up and never fires branch gate.
        std::array<int32_t, kDf2TreeSelectorTopK> single_tok{};
        single_tok.fill(-1);
        single_tok[0] = 77;
        const auto rs = df2_gm_row_top2(single_tok.data(), scores.data(), 0x51A7ULL, 11ULL, 1.0f);
        check(rs.valid && !rs.has_runner && !df2_gm_photo_finish_fires(rs, 10.0f),
              "single-candidate row never fires photo-finish gate");
    }

    // P3 Commit 2: GM knobs, guard relaxation, delta histogram binning, and photo-finish caterpillar builder.
    {
        check(!df2_tree_gm_env_enabled(nullptr) &&
                  !df2_tree_gm_env_enabled("") &&
                  !df2_tree_gm_env_enabled("0") &&
                  !df2_tree_gm_env_enabled("true") &&
                  df2_tree_gm_env_enabled("1"),
              "STRATA_DF2_TREE_GM parser defaults OFF and enables on 1");
        check(df2_tree_gm_on() == df2_tree_gm_env_enabled(std::getenv("STRATA_DF2_TREE_GM")),
              "df2_tree_gm_on matches STRATA_DF2_TREE_GM env");
        check(std::fabs(df2_tree_parse_gm_delta(nullptr) - kDf2TreeDefaultGmDelta) < 1e-6f &&
                  std::fabs(df2_tree_parse_gm_delta("") - kDf2TreeDefaultGmDelta) < 1e-6f &&
                  std::fabs(df2_tree_parse_gm_delta("0") - kDf2TreeDefaultGmDelta) < 1e-6f &&
                  std::fabs(df2_tree_parse_gm_delta("-0.5") - kDf2TreeDefaultGmDelta) < 1e-6f &&
                  std::fabs(df2_tree_parse_gm_delta("bad") - kDf2TreeDefaultGmDelta) < 1e-6f &&
                  std::fabs(df2_tree_parse_gm_delta("1.5") - 1.5f) < 1e-6f,
              "STRATA_DF2_GM_DELTA parser validates positive finite thresholds");
        check(std::fabs(df2_tree_gm_delta() - df2_tree_parse_gm_delta(std::getenv("STRATA_DF2_GM_DELTA"))) < 1e-6f,
              "df2_tree_gm_delta matches STRATA_DF2_GM_DELTA env");

        check(df2_tree_gm_delta_bin(0.0) == 0 && df2_tree_gm_delta_bin(0.1499) == 0 &&
                  df2_tree_gm_delta_bin(0.15) == 1 && df2_tree_gm_delta_bin(0.3499) == 1 &&
                  df2_tree_gm_delta_bin(0.35) == 2 && df2_tree_gm_delta_bin(0.6999) == 2 &&
                  df2_tree_gm_delta_bin(0.70) == 3 && df2_tree_gm_delta_bin(1.4999) == 3 &&
                  df2_tree_gm_delta_bin(1.50) == 4 && df2_tree_gm_delta_bin(INFINITY) == 4,
              "df2_tree_gm_delta_bin buckets [0,0.15,0.35,0.7,1.5,inf) accurately");

        // Build GM photo-finish tree and verify depth-d topology never depends on depth > d scores.
        std::array<int32_t, 7 * kDf2TreeSelectorTopK> toptok{};
        toptok.fill(-1);
        std::array<float, 7 * kDf2TreeSelectorTopK * kDf2TreeSelectorTopK> seltab_a{};
        seltab_a.fill(-INFINITY);
        for (int r = 0; r < 4; ++r) {
            for (int j = 0; j < 4; ++j) {
                toptok[static_cast<size_t>(r) * kDf2TreeSelectorTopK + static_cast<size_t>(j)] = 100 + r * 10 + j;
            }
            for (int prev = 0; prev < 4; ++prev) {
                const size_t base = (static_cast<size_t>(r) * kDf2TreeSelectorTopK + static_cast<size_t>(prev)) *
                                    kDf2TreeSelectorTopK;
                seltab_a[base + 0] = std::log(0.42f);
                seltab_a[base + 1] = std::log(0.38f);
                seltab_a[base + 2] = std::log(0.12f);
                seltab_a[base + 3] = std::log(0.08f);
            }
        }
        auto seltab_b = seltab_a;
        for (int prev = 0; prev < 4; ++prev) {
            const size_t base = (static_cast<size_t>(1) * kDf2TreeSelectorTopK + static_cast<size_t>(prev)) *
                                kDf2TreeSelectorTopK;
            seltab_b[base + 0] = std::log(0.90f);
            seltab_b[base + 1] = std::log(0.05f);
            seltab_b[base + 2] = std::log(0.03f);
            seltab_b[base + 3] = std::log(0.02f);
        }

        Df2GmBuildOptions opts;
        opts.delta = 2.0f;
        opts.max_nodes = 8;
        opts.q_temperature = 0.8f;
        Df2GmBuildStats st_a, st_b;
        const Df2TreeSpec tree_a = df2_tree_build_gm_photo_finish(
            50, 4, toptok.data(), seltab_a.data(), 0xFACEULL, 100ULL, opts, &st_a);
        const Df2TreeSpec tree_b = df2_tree_build_gm_photo_finish(
            50, 4, toptok.data(), seltab_b.data(), 0xFACEULL, 100ULL, opts, &st_b);
        check(tree_a.valid() && tree_a.is_caterpillar() && tree_a.count <= 8 &&
                  tree_b.valid() && tree_b.is_caterpillar() && tree_b.count <= 8,
              "df2_tree_build_gm_photo_finish builds valid caterpillars within max_nodes");
        check(st_a.spine_tokens[0] == st_b.spine_tokens[0] &&
                  std::fabs(st_a.decision_deltas[0] - st_b.decision_deltas[0]) < 1e-12,
              "depth-0 GM decision is independent of depth >= 1 scores");
    }

    std::printf("df2_tree_test: %s\n", g_failures == 0 ? "OK" : "FAILED");
    return g_failures == 0 ? 0 : 1;
}
