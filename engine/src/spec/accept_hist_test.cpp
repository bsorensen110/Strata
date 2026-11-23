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
    return ok ? 0 : 4;
}
