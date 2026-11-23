// src/core/batch_tail_skip_isolation_test.cpp - unit test for per-row anchor/draft tail-skip isolation (6B).
#include "strata/core/tail_skip_policy.hpp"

#include <cassert>
#include <cstdio>

int main() {
    using strata::core::should_tail_skip_expert;

    // All switches off -> never skip.
    assert(!should_tail_skip_expert(9, false, 0, 0, 0));
    assert(!should_tail_skip_expert(9, true, 0, 0, 0));

    // Legacy STRATA_ROUTE_TAIL_SKIP applies to both anchor and draft rows at min_rank >= base_tail_rank.
    assert(should_tail_skip_expert(7, true, 7, 0, 0));
    assert(!should_tail_skip_expert(6, true, 7, 0, 0));

    // Draft-only / batch tail-skip NEVER skips an expert if any anchor row routes it.
    assert(!should_tail_skip_expert(9, true, 0, 7, 0));
    assert(!should_tail_skip_expert(9, true, 0, 0, 6));
    assert(!should_tail_skip_expert(9, true, 0, 7, 6));

    // Draft-only missed experts beyond threshold ARE skipped when no anchor row touches them.
    assert(should_tail_skip_expert(7, false, 0, 7, 0));
    assert(!should_tail_skip_expert(6, false, 0, 7, 0));
    assert(should_tail_skip_expert(6, false, 0, 0, 6));
    assert(!should_tail_skip_expert(5, false, 0, 0, 6));

    std::printf("batch_tail_skip_isolation_test: ok\n");
    return 0;
}
