#pragma once

#if defined(__HIPCC__)
#include "strata/core/arch_defaults.hpp"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace strata::hip_compat {

inline bool hip_bufload_usable() {
#if defined(STRATA_HIP_GFX906)
    return false;
#else
    static const bool requested = [] {
        const char* v = std::getenv(strata::core::kHipBufloadEnv);
        return v ? v[0] == '1' && v[1] == '\0' : strata::core::kHipBufloadDefaultOn;
    }();
    if (!requested) return false;
    int device = -1;
    hipError_t e = hipGetDevice(&device);
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: hipGetDevice: %s\n", strata::core::kHipBufloadEnv, hipGetErrorString(e));
        std::exit(1);
    }
    static thread_local int cached_device = -1;
    static thread_local bool supported = false;
    if (device != cached_device) {
        hipDeviceProp_t prop{};
        e = hipGetDeviceProperties(&prop, device);
        if (e != hipSuccess) {
            std::fprintf(stderr, "%s: device %d properties: %s\n",
                         strata::core::kHipBufloadEnv, device, hipGetErrorString(e));
            std::exit(1);
        }
        supported = prop.warpSize == 32 && strata::core::hip_bufload_capable(prop.gcnArchName);
        cached_device = device;
    }
    return supported;
#endif
}

}  // namespace strata::hip_compat
#endif
