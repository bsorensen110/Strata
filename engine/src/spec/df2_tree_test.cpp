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
    // slot 3 collides with later top-1 drafts[5]=106, slot 4 is distinct token 303 (with duplicate at slot 5).
    const size_t base2 = (2u * kDf2TreeSelectorTopK + 0u) * kDf2TreeSelectorTopK;
    toptok[2 * kDf2TreeSelectorTopK + 1] = 102;
    toptok[2 * kDf2TreeSelectorTopK + 2] = 99;
    toptok[2 * kDf2TreeSelectorTopK + 3] = 106;
    toptok[2 * kDf2TreeSelectorTopK + 4] = 303;
    toptok[2 * kDf2TreeSelectorTopK + 5] = 303;
    seltab[base2 + 0] = 1.0f;
    seltab[base2 + 1] = 0.5f;
    seltab[base2 + 2] = 0.4f;
    seltab[base2 + 3] = 0.3f;
    seltab[base2 + 4] = -0.3f;
    seltab[base2 + 5] = -0.3f;

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
    check(with_dedup.valid() && with_dedup.branch_node == 2 && with_dedup.token[4] == 303,
          "with dedup, colliding candidates (102, 99, 106) are skipped in favor of distinct 303");

    // If every runner-up collides -> linear fallback
    toptok[2 * kDf2TreeSelectorTopK + 4] = 101;
    toptok[2 * kDf2TreeSelectorTopK + 5] = 104;
    const Df2TreeSpec all_coll = df2_tree_build_from_selector_ex(99, drafts.data(), probs.data(), 7,
                                                                 toptok.data(), seltab.data(), opts);
    check(all_coll.is_linear() && all_coll.count == 8, "all-colliding runner-ups fall back to linear");
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
    std::printf("df2_tree_test: %s\n", g_failures == 0 ? "OK" : "FAILED");
    return g_failures == 0 ? 0 : 1;
}
