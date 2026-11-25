#include "strata/spec/accept_hist.hpp"

#include <cstdio>
#include <cstring>

int main() {
    if (strata::spec::df2_accept_hist_enabled(nullptr) ||
        strata::spec::df2_accept_hist_enabled("0") ||
        !strata::spec::df2_accept_hist_enabled("1"))
        return 5;

    strata::spec::Df2AcceptHist h;
    h.observe(3, 0, 0, 3, 3);  // all three DF2 proposals accepted
    h.observe(3, 0, 0, 3, 1);  // position 1 accepted, position 2 rejected
    h.observe(2, 0, 1, 2, 2);  // chain row follows the two DF2 proposals and is rejected
    if (h.reached != std::array<int64_t, 7>{3, 3, 1, 0, 0, 0, 0} ||
        h.accepted != std::array<int64_t, 7>{3, 2, 1, 0, 0, 0, 0} ||
        h.lookup_rows != 0 || h.lookup_accept != 0 || h.chain_rows != 1 || h.chain_accept != 0)
        return 1;
    strata::spec::Df2AcceptHist suffix;
    suffix.observe(0, 2, 0, 0, 1);
    suffix.observe(3, 2, 1, 5, 4);  // DF2 prefix and one suffix row accepted
    suffix.observe(3, 2, 1, 5, 6);  // DF2, suffix, and chained lookup all accepted
    if (suffix.lookup_rows != 6 || suffix.lookup_accept != 4 ||
        suffix.chain_rows != 2 || suffix.chain_accept != 1)
        return 6;

    FILE* f = std::tmpfile();
    if (f == nullptr) return 2;
    h.emit(f, strata::spec::df2_accept_hist_enabled(nullptr), 1);
    if (std::ftell(f) != 0) { std::fclose(f); return 3; }
    h.emit(f, true, 9);
    std::rewind(f);
    char line[512] = {};
    const bool ok = std::fgets(line, sizeof(line), f) != nullptr &&
                    std::strcmp(line, "[accept-hist] req=9 df2_pos_reached=3,3,1,0,0,0,0 "
                                     "df2_pos_accept=3,2,1,0,0,0,0 lookup_rows=0 lookup_accept=0 "
                                     "chain_rows=1 chain_accept=0\n") == 0;
    std::fclose(f);
    if (!ok) return 4;

    // Empty tree histogram must keep the exact byte-identical linear line
    strata::spec::Df2TreeAcceptHist empty_tree;
    FILE* f_empty = std::tmpfile();
    if (f_empty == nullptr) return 7;
    h.emit(f_empty, true, 9, &empty_tree);
    std::rewind(f_empty);
    char line_empty[512] = {};
    const bool ok_empty = std::fgets(line_empty, sizeof(line_empty), f_empty) != nullptr &&
                          std::strcmp(line_empty, line) == 0;
    std::fclose(f_empty);
    if (!ok_empty) return 8;

    // Non-empty tree histogram appends tree_rounds/tree_branch_hits/tree_node_reached/tree_node_accept cleanly
    const int32_t prefix[3] = {10, 11, 12};
    const int32_t br[2] = {20, 21};
    const strata::spec::Df2TreeSpec tree = strata::spec::make_prefix_fanout_tree(1, prefix, 3, br, 2);
    const int32_t arg[8] = {10, 11, 12, 21, 0, 999, 0, 0};
    strata::spec::Df2TreeAcceptHist tree_hist;
    tree_hist.observe(tree, strata::spec::df2_tree_greedy_accept(tree, arg));
    FILE* f_tree = std::tmpfile();
    if (f_tree == nullptr) return 9;
    h.emit(f_tree, true, 9, &tree_hist);
    std::rewind(f_tree);
    char line_tree[1024] = {};
    const bool read_tree = std::fgets(line_tree, sizeof(line_tree), f_tree) != nullptr;
    std::fclose(f_tree);
    if (!read_tree) return 10;
    strata::spec::Df2TreeAcceptHist parsed_tree;
    if (!strata::spec::parse_accept_hist_fields(line_tree, parsed_tree) ||
        parsed_tree.tree_rounds != 1 || parsed_tree.branch_hits != 1 ||
        parsed_tree.runner_hits != 1 || parsed_tree.total_offered != 5 ||
        parsed_tree.runner_reached != 1 || parsed_tree.runner_accepted != 1 ||
        parsed_tree.cut_hist[3] != 1 ||
        parsed_tree.sampled_tree_rounds != 0 || parsed_tree.reject_resamples != 0 ||
        std::strstr(line_tree, "tree_sampled_rounds=") != nullptr ||
        std::strstr(line_tree, "df2_pos_reached=3,3,1,0,0,0,0") == nullptr) {
        return 11;
    }

    // Sampled tree round appends tree_sampled_rounds and tree_reject_resamples and parses back.
    tree_hist.observe(tree, strata::spec::df2_tree_greedy_accept(tree, arg), true, 2);
    FILE* f_sampled = std::tmpfile();
    if (f_sampled == nullptr) return 12;
    h.emit(f_sampled, true, 9, &tree_hist);
    std::rewind(f_sampled);
    char line_sampled[1024] = {};
    const bool read_sampled = std::fgets(line_sampled, sizeof(line_sampled), f_sampled) != nullptr;
    std::fclose(f_sampled);
    if (!read_sampled) return 13;
    strata::spec::Df2TreeAcceptHist parsed_sampled;
    if (!strata::spec::parse_accept_hist_fields(line_sampled, parsed_sampled) ||
        parsed_sampled.tree_rounds != 2 || parsed_sampled.sampled_tree_rounds != 1 ||
        parsed_sampled.reject_resamples != 2 ||
        std::strstr(line_sampled, "tree_sampled_rounds=1 tree_reject_resamples=2") == nullptr) {
        return 14;
    }

    // P5: When sampled_req_active is set, unconditional telemetry prints even when tree_rounds == 0 and round-trips.
    {
        strata::spec::Df2TreeAcceptHist zero_sampled;
        zero_sampled.sampled_req_active = true;
        for (int i = 0; i < 12; ++i) zero_sampled.observe_eligible();
        for (int i = 0; i < 4; ++i) zero_sampled.observe_reject(strata::spec::Df2TreeRejReason::MinP);
        for (int i = 0; i < 3; ++i) zero_sampled.observe_reject(strata::spec::Df2TreeRejReason::Ev);
        for (int i = 0; i < 2; ++i) zero_sampled.observe_reject(strata::spec::Df2TreeRejReason::Cut);
        zero_sampled.observe_reject(strata::spec::Df2TreeRejReason::Ema);
        for (int i = 0; i < 2; ++i) zero_sampled.observe_reject(strata::spec::Df2TreeRejReason::Dedup);

        FILE* f_zero = std::tmpfile();
        if (f_zero == nullptr) return 15;
        h.emit(f_zero, true, 9, &zero_sampled);
        std::rewind(f_zero);
        char line_zero[1024] = {};
        const bool read_zero = std::fgets(line_zero, sizeof(line_zero), f_zero) != nullptr;
        std::fclose(f_zero);
        if (!read_zero) return 16;

        strata::spec::Df2TreeAcceptHist parsed_zero;
        if (!strata::spec::parse_accept_hist_fields(line_zero, parsed_zero) ||
            !parsed_zero.sampled_req_active || parsed_zero.tree_rounds != 0 ||
            parsed_zero.sampled_tree_rounds != 0 || parsed_zero.reject_resamples != 0 ||
            parsed_zero.elig_rounds != 12 ||
            parsed_zero.rej_reason != std::array<int64_t, 5>{4, 3, 2, 1, 2} ||
            parsed_zero.width_hist != std::array<int64_t, 6>{0, 0, 0, 0, 0, 0} ||
            std::strstr(line_zero, "tree_sampled_rounds=0 tree_reject_resamples=0 tree_elig_rounds=12 "
                                   "tree_rej_reason=4,3,2,1,2 tree_width_hist=0,0,0,0,0,0") == nullptr) {
            return 17;
        }
    }

    // P5: Executed graph widths (T=3..8) accumulate into tree_width_hist and round-trip cleanly.
    {
        strata::spec::Df2TreeAcceptHist active_sampled;
        active_sampled.sampled_req_active = true;
        active_sampled.observe_eligible();
        active_sampled.observe(tree, strata::spec::df2_tree_greedy_accept(tree, arg), true, 0, 3);
        active_sampled.observe_eligible();
        active_sampled.observe(tree, strata::spec::df2_tree_greedy_accept(tree, arg), true, 1, 5);
        active_sampled.observe_eligible();
        active_sampled.observe(tree, strata::spec::df2_tree_greedy_accept(tree, arg), true, 2, 8);

        FILE* f_act = std::tmpfile();
        if (f_act == nullptr) return 18;
        h.emit(f_act, true, 9, &active_sampled);
        std::rewind(f_act);
        char line_act[1024] = {};
        const bool read_act = std::fgets(line_act, sizeof(line_act), f_act) != nullptr;
        std::fclose(f_act);
        if (!read_act) return 19;

        strata::spec::Df2TreeAcceptHist parsed_act;
        if (!strata::spec::parse_accept_hist_fields(line_act, parsed_act) ||
            !parsed_act.sampled_req_active || parsed_act.tree_rounds != 3 ||
            parsed_act.sampled_tree_rounds != 3 || parsed_act.reject_resamples != 3 ||
            parsed_act.elig_rounds != 3 ||
            parsed_act.width_hist != std::array<int64_t, 6>{1, 0, 1, 0, 0, 1} ||
            std::strstr(line_act, "tree_width_hist=1,0,1,0,0,1") == nullptr) {
            return 20;
        }
    }

    // P6: Applied sampled-tree envelope round-trips cleanly inside [accept-hist].
    {
        strata::spec::Df2TreeAcceptHist env_hist;
        env_hist.sampled_req_active = true;
        env_hist.set_sampled_envelope(strata::spec::Df2TreeSampledGateStatus::WalkTreesDisabled,
                                      0.90f, 0.25f, 0.25f, 0.70f, true, true);
        FILE* f_env = std::tmpfile();
        if (f_env == nullptr) return 21;
        h.emit(f_env, true, 9, &env_hist);
        std::rewind(f_env);
        char line_env[1024] = {};
        const bool read_env = std::fgets(line_env, sizeof(line_env), f_env) != nullptr;
        std::fclose(f_env);
        if (!read_env) return 22;

        strata::spec::Df2TreeAcceptHist parsed_env;
        if (!strata::spec::parse_accept_hist_fields(line_env, parsed_env) ||
            !parsed_env.has_envelope ||
            parsed_env.env_gate != strata::spec::Df2TreeSampledGateStatus::WalkTreesDisabled ||
            parsed_env.env_req_temp < 0.89f || parsed_env.env_req_temp > 0.91f ||
            parsed_env.env_min_branch_p < 0.24f || parsed_env.env_min_branch_p > 0.26f ||
            parsed_env.env_ev_cut < 0.24f || parsed_env.env_ev_cut > 0.26f ||
            parsed_env.env_max_temp < 0.69f || parsed_env.env_max_temp > 0.71f ||
            !parsed_env.env_no_walk_trees || !parsed_env.env_sample_walk ||
            std::strstr(line_env, "tree_env=gate:no_walk_trees,temp:0.90,minp:0.25,ev:0.25,maxt:0.70,nowalk:1,walk:1") ==
                nullptr) {
            return 23;
        }
    }
    return 0;
}
