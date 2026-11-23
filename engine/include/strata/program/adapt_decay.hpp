#pragma once
// --adapt-decay: the factor the adaptive tier's usage counts are multiplied by after each adaptation (#1284).
// Inside 0..1 it forgets old routing; above 1 the counts grow without end (inf after a few hundred adaptations) and the
// swap gains turn NaN, which breaks the sort that orders the swaps.  An unusable value is not refused: the engine says
// so and runs with the default.
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace strata::program {

constexpr float kAdaptDecayDefault = 0.7f;

inline bool adapt_coverage_enabled(const char* value) {
    return value != nullptr && value[0] == '1';
}

inline int adapt_exchange_limit(bool coverage_enabled) {
    return coverage_enabled ? 192 : 96;
}

inline void adapt_coverage_defaults(bool enabled, int64_t live, int64_t total, int& every, int& swaps, float& decay,
                                    bool every_given, bool swaps_given, bool decay_given) {
    if (!enabled || every <= 0 || swaps <= 0 || live <= 0 || live >= total) return;
    const double coverage = static_cast<double>(live) / static_cast<double>(total);
    if (coverage < 0.15 || coverage > 0.65) return;
    if (!every_given) every = 2;
    if (!swaps_given) swaps = 192;
    if (!decay_given) decay = 0.92f;
}

/// `v` itself when it is finite and in 0..1 (0 forgets everything at once, 1 never forgets: both work), else the default
/// with a line on stderr.
inline float adapt_decay_checked(double v, std::FILE* log = stderr) {
    if (std::isfinite(v) && v >= 0.0 && v <= 1.0) return (float) v;
    if (log != nullptr)
        std::fprintf(log, "strata: --adapt-decay %g is not between 0 and 1; using %g (counts above 1 grow without end and the swap gains turn NaN)\n",
                     v, (double) kAdaptDecayDefault);
    return kAdaptDecayDefault;
}

}  // namespace strata::program
