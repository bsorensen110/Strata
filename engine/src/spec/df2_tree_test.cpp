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

}  // namespace

int main() {
    test_cut_depth();
    test_builders_and_invariants();
    test_paths_and_conv_ancestors();
    test_topological_order_and_canonicalize();
    test_acceptance_walks_and_branch_kill();
    test_serialization_and_telemetry_roundtrip();
    std::printf("df2_tree_test: %s\n", g_failures == 0 ? "OK" : "FAILED");
    return g_failures == 0 ? 0 : 1;
}
