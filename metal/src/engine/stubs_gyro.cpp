// metal/src/engine/stubs_gyro.cpp - the Gyro (agentionai trellis experts) entry points of src/kernels/cuda/iq_kernels.cu that
// the Metal engine does not implement as functions: the GPU Hadamard rotations.  The window and the prompt chain
// (window.cpp, prompt_chain.inc) encode the rotation themselves (hadamard_rows / hadamard_rows_h of kernels/hadamard.metal)
// into their command buffers, so the shared host code's calls are never reached on Metal.  Each stops the engine with its
// name, like stubs.cpp.

#include <cstdint>

#include "strata/kernels/iq_kernels.hpp"

namespace strata::metal_engine {
void missing(const char* name);
}

namespace strata::kernels {
void hadamard_rows(const float*, float*, int64_t, int64_t, int, const float*, void*) {
    strata::metal_engine::missing("strata::kernels::hadamard_rows");
}
void hadamard_rows_f16(uint16_t*, int64_t, int64_t, int, const float*, void*) {
    strata::metal_engine::missing("strata::kernels::hadamard_rows_f16");
}
const float* hadamard_device_signs(const float*, int) {
    strata::metal_engine::missing("strata::kernels::hadamard_device_signs");
    return nullptr;
}
}  // namespace strata::kernels
