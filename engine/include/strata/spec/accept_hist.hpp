#pragma once

#include "strata/spec/df2_tree.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace strata::spec {

inline bool df2_accept_hist_enabled(const char* value) {
    return value != nullptr && std::strcmp(value, "1") == 0;
}

struct Df2AcceptHist {
    std::array<int64_t, 7> reached{};
    std::array<int64_t, 7> accepted{};
    // lookup_* counts suffix rows; chain_* counts rows appended after the base window.
    int64_t lookup_rows = 0;
    int64_t lookup_accept = 0;
    int64_t chain_rows = 0;
    int64_t chain_accept = 0;

    void observe(int df2_rows, int lookup_n, int chain_n, int chain_offset, int verify_accept) {
        const int drows = std::clamp(df2_rows, 0, 7);
        const int lrows = std::max(0, lookup_n);
        const int crows = std::max(0, chain_n);
        const int cstart = std::max(0, chain_offset);
        const int a = std::max(0, verify_accept);

        // Position i is "reached" only when every earlier proposal survived; this makes
        // accepted/reached the conditional chain probability, not unconditional row accuracy.
        for (int i = 0; i < drows; ++i) {
            if (a >= i) {
                ++reached[(size_t) i];
                if (a > i) ++accepted[(size_t) i];
            }
        }
        lookup_rows += lrows;
        lookup_accept += std::clamp(a - drows, 0, lrows);
        chain_rows += crows;
        chain_accept += std::clamp(a - cstart, 0, crows);
    }

    void emit(FILE* out, bool enabled, int64_t request_id, const Df2TreeAcceptHist* tree = nullptr) const {
        if (!enabled || out == nullptr) return;
        std::fprintf(out, "[accept-hist] req=%lld df2_pos_reached=", (long long) request_id);
        for (size_t i = 0; i < reached.size(); ++i)
            std::fprintf(out, "%s%lld", i == 0 ? "" : ",", (long long) reached[i]);
        std::fprintf(out, " df2_pos_accept=");
        for (size_t i = 0; i < accepted.size(); ++i)
            std::fprintf(out, "%s%lld", i == 0 ? "" : ",", (long long) accepted[i]);
        std::fprintf(out, " lookup_rows=%lld lookup_accept=%lld chain_rows=%lld chain_accept=%lld",
                     (long long) lookup_rows, (long long) lookup_accept,
                     (long long) chain_rows, (long long) chain_accept);
        if (tree != nullptr && (tree->tree_rounds > 0 || tree->sampled_req_active)) {
            std::fprintf(out, " %s", tree->format_accept_hist_fields().c_str());
        }
        std::fprintf(out, "\n");
    }
};

}  // namespace strata::spec
