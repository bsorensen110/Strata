#include "strata/spec/draft_composition.hpp"
#include "strata/spec/suffix_drafter.hpp"

#include <cstdio>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}
}

int main() {
    strata::spec::SuffixDrafter sfx(3, 32, 256);
    const int32_t history[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 50, 51, 52, 1, 2};
    const int32_t df2[] = {3, 4};
    sfx.append(history, sizeof(history) / sizeof(history[0]));
    int32_t suffix[4] = {};
    check(sfx.propose_after(df2, 2, 4, 2, suffix) == 4 && suffix[0] == 5,
          "the configured two-token minimum accepts this longer source match");
    const int n = sfx.propose_after(df2, 2, 4, 4, suffix);
    check(n == 4 && suffix[0] == 5 && suffix[3] == 8, "suffix follows the DF2 prefix after a four-token match");
    check(sfx.last_match() == 4, "suffix match length is measured across the DF2 prefix");
    check(sfx.propose_after(df2, 2, 4, 5, suffix) == 0, "per-call minimum match rejects a shorter continuation");
    check(sfx.size() == sizeof(history) / sizeof(history[0]), "suffix proposal does not commit DF2 drafts");

    strata::spec::SuffixDrafter short_sfx(3, 32, 64);
    short_sfx.enable_short_matches();
    const int32_t short_history[] = {1, 2, 9, 1};
    const int32_t short_df2[] = {2};
    short_sfx.append(short_history, sizeof(short_history) / sizeof(short_history[0]));
    check(short_sfx.propose_after(short_df2, 1, 2, 2, suffix) == 2 && suffix[0] == 9,
          "the opt-in bigram index supports the configured two-token minimum");
    check(short_sfx.propose_after(short_df2, 1, 2, 3, suffix) == 0,
          "the ordinary trigram index still rejects a two-token-only match");

    const int32_t overlapping_suffix[] = {3, 4, 5};
    const int32_t lookup[] = {6, 7};
    strata::spec::DraftComposition composition(7);
    check(composition.append(strata::spec::DraftCompositionSource::Dflash2, df2, 2), "DF2 rows append first");
    check(composition.append(strata::spec::DraftCompositionSource::Suffix, overlapping_suffix, 3), "suffix rows append second");
    check(composition.append(strata::spec::DraftCompositionSource::Lookup, lookup, 2), "lookup rows append last");
    check(composition.tokens() == std::vector<int32_t>({3, 4, 3, 4, 5, 6, 7}),
          "overlap never replaces the higher-priority DF2 rows");
    check(composition.source_at(0) == strata::spec::DraftCompositionSource::Dflash2 &&
              composition.source_at(2) == strata::spec::DraftCompositionSource::Suffix &&
              composition.source_at(5) == strata::spec::DraftCompositionSource::Lookup,
          "composition records the DF2, suffix, lookup arbitration order");
    check(!composition.append(strata::spec::DraftCompositionSource::Dflash2, df2, 1), "lower-priority rows cannot be appended later");

    std::printf("df2_suffix_fill_test: %s\n", failures == 0 ? "OK" : "FAILED");
    return failures == 0 ? 0 : 1;
}
