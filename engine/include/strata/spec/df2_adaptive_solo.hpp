// include/strata/spec/df2_adaptive_solo.hpp - host-only solo DFlash2 width selection.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace strata::spec {

inline constexpr int df2_solo_prefix_width(int selected_width, int produced_drafts) {
    return std::min(selected_width, 1 + produced_drafts);
}

class Df2AdaptiveSolo {
public:
    static constexpr int kMaxWidth = 16;
    static constexpr int kPositions = kMaxWidth - 1;

    void reset(int base_width) {
        base_width_ = std::clamp(base_width, 1, kMaxWidth);
        tier_ = base_width_;
        rounds_since_switch_ = 0;
        df2_ok_.fill(0.0);
        df2_seen_.fill(0.0);
        lookup_ok_.fill(0.0);
        lookup_seen_.fill(0.0);
    }

    // Only rounds with the full DF2 prefix are used, so each lookup position has the same offset.
    void observe(int accepted, int lookup_rows) {
        const int df2_rows = base_width_ - 1;
        const int offered = df2_rows + std::clamp(lookup_rows, 0, kMaxWidth - base_width_);
        accepted = std::clamp(accepted, 0, offered);
        constexpr double decay = 0.95;
        auto update = [decay](std::array<double, kPositions>& ok, std::array<double, kPositions>& seen,
                              int position, bool success) {
            const size_t i = (size_t) position - 1;
            ok[i] = decay * ok[i] + (success ? 1.0 : 0.0);
            seen[i] = decay * seen[i] + 1.0;
        };
        for (int position = 1; position <= std::min(offered, accepted + 1); ++position) {
            const bool success = position <= accepted;
            if (position <= df2_rows) {
                update(df2_ok_, df2_seen_, position, success);
            } else {
                update(lookup_ok_, lookup_seen_, position - df2_rows, success);
            }
        }
        ++rounds_since_switch_;
    }

    double acceptance(int position) const {
        if (position < 1 || position > kPositions) return 0.5;
        if (position < base_width_)
            return rate(df2_ok_, df2_seen_, position);
        return rate(lookup_ok_, lookup_seen_, position - base_width_ + 1);
    }

    double expected_accepted(int width) const {
        width = std::clamp(width, 1, kMaxWidth);
        double survival = 1.0;
        double expected = 0.0;
        for (int i = 1; i < width; ++i) {
            survival *= acceptance(i);
            expected += survival;
        }
        return expected;
    }

    double expected_needed_rows() const {
        return 1.0 + expected_accepted(kMaxWidth);
    }

    int choose_chain(int lookup_available, int lookup_kept, unsigned allowed_sizes, const double* costs) {
        return std::max(lookup_kept, choose(lookup_available, allowed_sizes, costs) - base_width_);
    }

    int tier() const { return tier_; }

private:
    int choose(int lookup_available, unsigned allowed_sizes, const double* costs) {
        const int reachable = base_width_ + std::clamp(lookup_available, 0, kMaxWidth - base_width_);
        allowed_sizes |= 1u << base_width_;
        allowed_sizes &= (1u << (reachable + 1)) - 1u;
        int desired = base_width_;
        const double needed = expected_needed_rows();
        for (int width = base_width_; width <= reachable; ++width) {
            if (((allowed_sizes >> width) & 1u) == 0) continue;
            desired = width;
            if ((double) width >= needed) break;
        }

        if (((allowed_sizes >> tier_) & 1u) == 0) {
            for (int width = std::min(tier_, kMaxWidth); width >= base_width_; --width) {
                if ((allowed_sizes >> width) & 1u) {
                    tier_ = width;
                    rounds_since_switch_ = 0;
                    break;
                }
            }
        }
        const double tier_delta = std::abs(needed - (double) tier_);
        if (desired != tier_ && tier_delta >= 0.10 * tier_ && rounds_since_switch_ >= 8) {
            const double current_cost = std::max(1e-9, costs[tier_]);
            const double desired_cost = std::max(1e-9, costs[desired]);
            const double current_score = (1.0 + expected_accepted(tier_)) / current_cost;
            const double desired_score = (1.0 + expected_accepted(desired)) / desired_cost;
            if (desired_score >= current_score * 0.90) {
                tier_ = desired;
                rounds_since_switch_ = 0;
            }
        }
        return tier_;
    }

    static double rate(const std::array<double, kPositions>& ok, const std::array<double, kPositions>& seen, int position) {
        if (position < 1 || position > kPositions) return 0.5;
        const size_t i = (size_t) position - 1;
        return (ok[i] + 1.0) / (seen[i] + 2.0);
    }

    int base_width_ = 1;
    int tier_ = 1;
    int rounds_since_switch_ = 0;
    std::array<double, kPositions> df2_ok_{};
    std::array<double, kPositions> df2_seen_{};
    std::array<double, kPositions> lookup_ok_{};
    std::array<double, kPositions> lookup_seen_{};
};

}  // namespace strata::spec
