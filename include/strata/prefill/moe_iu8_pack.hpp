// SPDX-License-Identifier: MIT
// Port note: hipfire CHANGELOG.md:840,856 and gemm_mq6sym_i8.gfx1201.hip.
// Only an exact, bit-checked change of weight representation is admitted.
#pragma once

#if defined(__HIPCC__) || defined(STRATA_USE_HIP) || defined(STRATA_MOE_IU8_CPU_ORACLE)
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace strata::prefill::iu8 {

struct Pack {
    int rows = 0, cols = 0, bits = 0;
    bool symmetric_verified = false;
    std::vector<uint8_t> codes;
    std::vector<uint16_t> scales;
};

// Failure clears out and names the first refused block. No approximate conversion.
bool pack_exact(int type, const void* source, size_t bytes, int rows, int cols, Pack& out, std::string& why);
int signed_code(const Pack& p, size_t i);
float half_value(uint16_t h);

#if defined(__HIPCC__) || defined(STRATA_USE_HIP)
bool usable();
struct WeightView;

struct DeviceWeight {
    void* allocation = nullptr;
    const uint8_t* codes = nullptr;
    const uint16_t* scales = nullptr;
    int rows = 0, cols = 0, bits = 0, device = -1;
    bool symmetric_verified = false;
    DeviceWeight() = default;
    ~DeviceWeight();
    DeviceWeight(DeviceWeight&& other) noexcept;
    DeviceWeight& operator=(DeviceWeight&& other) noexcept;
    DeviceWeight(const DeviceWeight&) = delete;
    DeviceWeight& operator=(const DeviceWeight&) = delete;
};
struct Expert { DeviceWeight gu, down; };
struct Layer { std::vector<Expert> experts; };

bool upload(const Pack& pack, DeviceWeight& out, std::string& why);
WeightView view(const DeviceWeight& w);
bool load_layer(int gu_type, int down_type, int embd, int ff, int experts, size_t blob_bytes,
                size_t up_offset, size_t down_offset,
                const std::function<const uint8_t*(int)>& blob, Layer& out, std::string& why);
#endif
}  // namespace strata::prefill::iu8
#else
namespace strata::prefill::iu8 {
inline void pack_exact(...) = delete;
}
#endif
