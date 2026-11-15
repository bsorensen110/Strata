#include "strata/core/expert_cache.hpp"

#include <cstdio>

int main() {
    using strata::core::ExpertCache;
    int failed = 0;
    auto check = [&](bool ok, const char* name) {
        std::printf("%s: %s\n", ok ? "PASS" : "FAIL", name);
        failed += !ok;
    };

    check(ExpertCache::is_live_slot(0, 8), "first mapped slot is selectable");
    check(ExpertCache::is_live_slot(7, 8), "last mapped slot is selectable");
    check(!ExpertCache::is_live_slot(8, 8), "unmapped tail is not selectable");
    check(!ExpertCache::is_live_slot(-1, 8), "negative slots are not selectable");
    check(!ExpertCache::is_live_slot(0, 0), "shrink to zero exposes no slots");
    check(ExpertCache::is_live_slot(7, 16), "growing restores slots within the budget");
    check(!ExpertCache::is_live_slot(16, 16), "growth beyond the budget remains unavailable");
    return failed == 0 ? 0 : 1;
}
