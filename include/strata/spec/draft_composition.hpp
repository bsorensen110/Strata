#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::spec {

enum class DraftCompositionSource : uint8_t { Dflash2, Suffix, Lookup };

class DraftComposition {
public:
    explicit DraftComposition(size_t budget) : budget_(budget) {}

    bool append(DraftCompositionSource source, const int32_t* tokens, int n) {
        const int rank = static_cast<int>(source);
        if (rank < last_rank_ || n < 0 || (n > 0 && tokens == nullptr)) return false;
        last_rank_ = rank;
        const size_t count = std::min(static_cast<size_t>(n), budget_ - tokens_.size());
        if (count > 0) tokens_.insert(tokens_.end(), tokens, tokens + count);
        sources_.insert(sources_.end(), count, source);
        return true;
    }

    size_t size() const { return tokens_.size(); }
    const std::vector<int32_t>& tokens() const { return tokens_; }
    DraftCompositionSource source_at(size_t i) const { return sources_[i]; }
    int count(DraftCompositionSource source) const {
        return static_cast<int>(std::count(sources_.begin(), sources_.end(), source));
    }

private:
    size_t budget_;
    int last_rank_ = -1;
    std::vector<int32_t> tokens_;
    std::vector<DraftCompositionSource> sources_;
};

}  // namespace strata::spec
