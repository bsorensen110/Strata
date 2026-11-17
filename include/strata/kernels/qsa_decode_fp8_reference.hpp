// SPDX-License-Identifier: MIT
#pragma once

#include "strata/kernels/qsa_decode_fp8_policy.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace strata::kernels::fp8_probe::reference {
inline void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

inline float decode(unsigned code) {
    const unsigned e = (code >> 3) & 15, m = code & 7;
    const float v = e == 0 ? std::ldexp(static_cast<float>(m), -9)
                           : e == 15 && m == 7 ? NAN
                           : std::ldexp(1.0f + static_cast<float>(m) / 8.0f, static_cast<int>(e) - 7);
    return code & 128 ? -v : v;
}

// Independent scalar nearest-neighbor oracle, including even-code ties and
// finite saturation. No HIP conversion headers or device arithmetic reused.
inline unsigned encode(float value) {
    const unsigned sign = std::signbit(value) ? 128 : 0;
    if (std::isnan(value)) return sign | 127;
    const float x = std::min(std::fabs(value), 448.0f);
    unsigned lo = 0, hi = 126;
    while (lo < hi) {
        const unsigned mid = (lo + hi) / 2;
        if (decode(mid) < x) lo = mid + 1;
        else hi = mid;
    }
    if (lo > 0) {
        const float down = x - decode(lo - 1), up = decode(lo) - x;
        if (down < up || (down == up && (lo & 1))) --lo;
    }
    return sign | lo;
}

struct Fixture {
    int group, kv_heads = 2, cap, width, page_size;
    std::vector<float> q, k, v;
    std::vector<int32_t> ids, pages;

    Fixture(int g, int c, int w, int ps, bool missing)
        : group(g), cap(c), width(w), page_size(ps), q(static_cast<size_t>(g * kv_heads) * kDim), ids(c) {
        const int np = (c + ps - 1) / ps;
        pages.resize(np);
        for (int p = 0; p < np; ++p) pages[p] = missing && p == np / 2 ? -1 : np - 1 - p;
        for (int i = 0; i < c; ++i) ids[i] = c - 1 - i;
        k.resize(static_cast<size_t>(np) * kv_heads * ps * kDim);
        v.resize(k.size());
        for (size_t i = 0; i < k.size(); ++i) {
            k[i] = (static_cast<int>(i * 3 % 113) - 56) / 32.0f;
            v[i] = (static_cast<int>(i * 5 % 97) - 48) / 16.0f;
        }
        for (size_t i = 0; i < q.size(); ++i) q[i] = (static_cast<int>(i * 11 % 101) - 50) / 64.0f;
    }
    long long row(int cell, int kv) const {
        if (cell < 0 || pages[cell / page_size] < 0) return -1;
        return (static_cast<long long>(pages[cell / page_size]) * kv_heads + kv) * page_size + cell % page_size;
    }
};

inline std::vector<float> attention(const Fixture& f, bool quantized = true) {
    std::vector<float> k = f.k, v = f.v, out(f.q.size());
    if (quantized) {
        for (auto& x : k) x = decode(encode(x));
        for (auto& x : v) x = decode(encode(x));
    }
    const int width = std::clamp(f.width, 0, f.cap);
    std::vector<double> scores(width);
    for (int h = 0; h < f.group * f.kv_heads; ++h) {
        double m = -INFINITY, l = 0;
        for (int t = 0; t < width; ++t) {
            const auto row = f.row(f.ids[t], h / f.group);
            double s = -INFINITY;
            if (row >= 0) {
                s = 0;
                for (int d = 0; d < kDim; ++d) s += static_cast<double>(f.q[h * kDim + d]) * k[row * kDim + d];
                s *= 0.0625;
            }
            scores[t] = s;
            m = std::max(m, s);
        }
        if (m == -INFINITY) continue;
        for (auto& s : scores) { s = std::exp(s - m); l += s; }
        for (int d = 0; d < kDim; ++d) {
            double sum = 0;
            for (int t = 0; t < width; ++t) {
                const auto row = f.row(f.ids[t], h / f.group);
                if (row >= 0) sum += scores[t] * v[row * kDim + d];
            }
            out[h * kDim + d] = static_cast<float>(sum / l);
        }
    }
    return out;
}
}  // namespace strata::kernels::fp8_probe::reference
