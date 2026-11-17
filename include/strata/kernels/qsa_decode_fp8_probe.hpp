// SPDX-License-Identifier: MIT
#pragma once

#if defined(STRATA_USE_HIP)
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_decode_fp8_policy.hpp"
#include <string>

namespace strata::kernels::fp8_probe {

struct Support {
    char arch[256] = {};
    int wave = 0;
    bool conversions = false;
};

// Once, before any capture or worker threads. A tiny device query checks the
// actual image, not the host compiler's builtin table. No work when disabled.
bool initialize(const char* setting, std::string& error);
bool enabled();
Support support();

// No allocation, synchronization, environment reads or discovery in this path.
// On refusal nothing is launched or written; the caller retains its merge.
Refusal launch_chunks(bool requested, const float* q, const QsaAttnPools& pools,
                      const int32_t* ids, const int32_t* step, int64_t cap,
                      const QsaShapes& s, float* scratch, int64_t queries, void* stream);

}  // namespace strata::kernels::fp8_probe
#endif
