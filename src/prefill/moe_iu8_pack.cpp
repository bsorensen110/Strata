// SPDX-License-Identifier: MIT
// Port note: hipfire CHANGELOG.md:840,856: verify symmetry at load, never requantize.
// gemm_mq6sym_i8.gfx1201.hip: per-K128 f16 scale and signed q-32 (q-8 for W4).
// IQ4's nonlinear table is decoded by ggml, not mistaken for symmetric nibbles.
#include "strata/prefill/moe_iu8_pack.hpp"

#if defined(__HIPCC__) || defined(STRATA_USE_HIP) || defined(STRATA_MOE_IU8_CPU_ORACLE)
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>

namespace strata::prefill::iu8 {
namespace {
bool equal_bits(float a, float b) { return std::memcmp(&a, &b, sizeof(a)) == 0; }

bool block(const float* x, int bits, uint16_t& scale, uint8_t* codes) {
    int anchor = 0;
    for (int i = 0; i < 128; ++i) {
        if (!std::isfinite(x[i])) return false;
        if (std::fabs(x[i]) > std::fabs(x[anchor])) anchor = i;
    }
    const int mid = 1 << (bits - 1);
    // Enumerate the anchor's possible integer code. This is an exact encoding search,
    // not a quantizer: EVERY reconstructed float, including signed zero, must match.
    for (int c = -mid; c < mid; ++c) {
        if (c == 0) continue;
        const uint16_t h = ggml_fp32_to_fp16(x[anchor] == 0.0f ? 1.0f : x[anchor] / (float) c);
        const float d = ggml_fp16_to_fp32(h);
        if (d == 0.0f || !std::isfinite(d)) continue;
        bool exact = true;
        for (int k = 0; k < 128; ++k) {
            const float q = std::nearbyint(x[k] / d);
            if (q < -mid || q >= mid || !equal_bits(d * q, x[k])) { exact = false; break; }
            codes[k] = (uint8_t) ((int) q + mid);
            // Conversion of -0 to integer drops its sign; check the ACTUAL consumer.
            if (!equal_bits(d * (float) ((int) codes[k] - mid), x[k])) { exact = false; break; }
        }
        if (exact) { scale = h; return true; }
    }
    return false;
}
}  // namespace

float half_value(uint16_t h) { return ggml_fp16_to_fp32(h); }

int signed_code(const Pack& p, size_t i) {
    const size_t bit = i * (size_t) p.bits, byte = bit / 8;
    unsigned word = p.codes[byte];
    if (bit % 8 + p.bits > 8) word |= (unsigned) p.codes[byte + 1] << 8;
    return (int) ((word >> (bit % 8)) & ((1u << p.bits) - 1)) - (1 << (p.bits - 1));
}

bool pack_exact(int type, const void* source, size_t bytes, int rows, int cols, Pack& out, std::string& why) {
    out = {};
    why.clear();
    if (!source || rows <= 0 || cols <= 0 || cols % 128 != 0) {
        why = "symmetric IU8 weights need a nonempty tensor and K % 128 == 0"; return false;
    }
    if (type != GGML_TYPE_IQ4_NL && type != GGML_TYPE_IQ4_XS &&
        type != GGML_TYPE_Q4_0 && type != GGML_TYPE_Q6_K) {
        why = "format is not an admitted IQ4/Q4_0/Q6_K source"; return false;
    }
    const auto t = (ggml_type) type;
    const auto* traits = ggml_get_type_traits(t);
    const size_t qk = (size_t) ggml_blck_size(t);
    if ((size_t) cols % qk || !traits->to_float ||
        (size_t) rows > std::numeric_limits<size_t>::max() / (size_t) cols) {
        why = "invalid source block geometry"; return false;
    }
    const size_t row_bytes = (size_t) cols / qk * ggml_type_size(t);
    if ((size_t) rows > std::numeric_limits<size_t>::max() / row_bytes ||
        bytes != (size_t) rows * row_bytes) { why = "source tensor byte extent differs"; return false; }
    Pack p;
    p.rows = rows; p.cols = cols; p.bits = type == GGML_TYPE_Q6_K ? 6 : 4;
    const size_t count = (size_t) rows * cols;
    if (count > std::numeric_limits<size_t>::max() / (size_t) p.bits) {
        why = "packed extent overflow"; return false;
    }
    p.codes.resize(count * p.bits / 8, 0);
    p.scales.resize(count / 128);
    std::vector<float> decoded((size_t) cols);
    for (int r = 0; r < rows; ++r) {
        traits->to_float((const uint8_t*) source + (size_t) r * row_bytes, decoded.data(), cols);
        for (int kb = 0; kb < cols / 128; ++kb) {
            uint8_t codes[128];
            const size_t bi = (size_t) r * (cols / 128) + kb;
            if (!block(decoded.data() + kb * 128, p.bits, p.scales[bi], codes)) {
                why = "dequant bits are not symmetric-exact at row " + std::to_string(r) +
                      ", K128 block " + std::to_string(kb);
                return false;
            }
            for (size_t k = 0; k < 128; ++k) {
                const size_t bit = (bi * 128 + k) * p.bits, byte = bit / 8;
                const unsigned word = (unsigned) codes[k] << (bit % 8);
                p.codes[byte] |= (uint8_t) word;
                if (bit % 8 + p.bits > 8) p.codes[byte + 1] |= (uint8_t) (word >> 8);
            }
        }
        // Independent readback through the packed consumer, not the candidate scratch.
        for (int k = 0; k < cols; ++k) {
            const size_t i = (size_t) r * cols + k;
            if (!equal_bits(half_value(p.scales[i / 128]) * (float) signed_code(p, i), decoded[k])) {
                why = "packed dequant readback differs"; return false;
            }
        }
    }
    p.symmetric_verified = true;
    out = std::move(p);
    return true;
}
}  // namespace strata::prefill::iu8
#endif

#if defined(__HIPCC__) || defined(STRATA_USE_HIP)
#include "strata/core/arch_defaults.hpp"
#include "strata/prefill/moe_iu8_gfx.hpp"
#include <hip/hip_runtime.h>
#include <cstdlib>

namespace strata::prefill::iu8 {
namespace {
void check(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "HIP MoE IU8: %s: %s\n", what, hipGetErrorString(e)); std::exit(1);
    }
}
void release(DeviceWeight& p) {
    if (!p.allocation) return;
    int prev = -1;
    check(hipGetDevice(&prev), "release device");
    check(hipSetDevice(p.device), "release owner");
    check(hipFree(p.allocation), "free exact weights");
    check(hipSetDevice(prev), "restore device");
    p.allocation = nullptr;
}
}  // namespace

bool usable() {
    static const bool requested = [] {
        const char* v = std::getenv(strata::core::kHipMoeIu8Env);
        return v && std::strcmp(v, "1") == 0;
    }();
    if (!requested) return false;
    int device = -1;
    check(hipGetDevice(&device), "current device");
    hipDeviceProp_t prop{};
    check(hipGetDeviceProperties(&prop, device), "device properties");
    return prop.warpSize == 32 && strata::core::hip_moe_iu8_capable(prop.gcnArchName);
}

DeviceWeight::~DeviceWeight() { release(*this); }
DeviceWeight::DeviceWeight(DeviceWeight&& other) noexcept { *this = std::move(other); }
DeviceWeight& DeviceWeight::operator=(DeviceWeight&& other) noexcept {
    if (this == &other) return *this;
    release(*this);
    allocation = other.allocation; codes = other.codes; scales = other.scales;
    rows = other.rows; cols = other.cols; bits = other.bits; device = other.device;
    symmetric_verified = other.symmetric_verified;
    other.allocation = nullptr; other.codes = nullptr; other.scales = nullptr; other.symmetric_verified = false;
    return *this;
}

WeightView view(const DeviceWeight& w) {
    return {w.codes, w.scales, w.rows, w.cols, w.bits, w.device, w.symmetric_verified};
}

WindowWeights::~WindowWeights() {
    if (!allocation) return;
    int previous = -1;
    check(hipGetDevice(&previous), "window table device");
    check(hipSetDevice(device), "window table owner");
    check(hipFree(allocation), "free window table");
    check(hipSetDevice(previous), "restore window table device");
}
WindowWeights::WindowWeights(WindowWeights&& other) noexcept { *this = std::move(other); }
WindowWeights& WindowWeights::operator=(WindowWeights&& other) noexcept {
    if (this == &other) return *this;
    // Tables are uploaded once, before capture; vector relocation only moves empty destinations.
    if (allocation) { std::fputs("HIP MoE IU8: replacing a live window table\n", stderr); std::abort(); }
    allocation = other.allocation; gu = other.gu; down = other.down; device = other.device;
    other.allocation = nullptr; other.gu = nullptr; other.down = nullptr;
    return *this;
}
void WindowWeights::upload(const Layer& layer) {
    if (allocation || layer.experts.empty()) {
        std::fputs("HIP MoE IU8: invalid window table upload\n", stderr); std::abort();
    }
    std::vector<WeightView> views;
    for (const auto& e : layer.experts) views.push_back(view(e.gu));
    for (const auto& e : layer.experts) views.push_back(view(e.down));
    check(hipGetDevice(&device), "window table upload device");
    check(hipMalloc(&allocation, views.size() * sizeof(WeightView)), "allocate window table");
    check(hipMemcpy(allocation, views.data(), views.size() * sizeof(WeightView), hipMemcpyHostToDevice),
          "upload window table");
    gu = (const WeightView*) allocation; down = gu + layer.experts.size();
}

bool upload(const Pack& p, DeviceWeight& out, std::string& why) {
    if (!p.symmetric_verified || p.rows <= 0 || p.cols <= 0 || p.cols % 128 ||
        (p.bits != 4 && p.bits != 6) ||
        p.codes.size() != (size_t) p.rows * p.cols * p.bits / 8 ||
        p.scales.size() != (size_t) p.rows * p.cols / 128) {
        why = "weight upload needs a verified complete pack"; return false;
    }
    DeviceWeight w;
    check(hipGetDevice(&w.device), "upload device");
    const size_t scale_offset = (p.codes.size() + 1) & ~size_t{1};
    const hipError_t e = hipMalloc(&w.allocation, scale_offset + p.scales.size() * sizeof(uint16_t));
    if (e == hipErrorOutOfMemory) {
        (void) hipGetLastError();
        why = "not enough VRAM for an additional exact weight pack"; return false;
    }
    check(e, "allocate exact weights");
    check(hipMemcpy(w.allocation, p.codes.data(), p.codes.size(), hipMemcpyHostToDevice), "upload codes");
    w.codes = (const uint8_t*) w.allocation;
    w.scales = (const uint16_t*) (w.codes + scale_offset);
    check(hipMemcpy((void*) w.scales, p.scales.data(), p.scales.size() * sizeof(uint16_t),
                    hipMemcpyHostToDevice), "upload scales");
    w.rows = p.rows; w.cols = p.cols; w.bits = p.bits; w.symmetric_verified = true;
    out = std::move(w);
    return true;
}

bool load_layer(int gt, int dt, int embd, int ff, int ne, size_t bytes, size_t up, size_t down,
                const std::function<const uint8_t*(int)>& blob, Layer& out, std::string& why) {
    out.experts.clear();
    if (ne <= 0 || embd <= 0 || ff <= 0 || embd % 128 || ff % 128 || up == 0 ||
        down != 2 * up || down >= bytes) {
        why = "layer needs split gate/up, K128 geometry and complete source extents"; return false;
    }
    Layer layer;
    for (int e = 0; e < ne; ++e) {
        const uint8_t* b = blob(e);
        if (!b) { why = "source could not supply expert " + std::to_string(e); return false; }
        Pack gu, dn;
        if (!pack_exact(gt, b, down, 2 * ff, embd, gu, why) ||
            !pack_exact(dt, b + down, bytes - down, embd, ff, dn, why)) {
            why = "expert " + std::to_string(e) + ": " + why; return false;
        }
        Expert expert;
        if (!upload(gu, expert.gu, why) || !upload(dn, expert.down, why)) return false;
        layer.experts.push_back(std::move(expert));
    }
    out = std::move(layer);
    return true;
}
}  // namespace strata::prefill::iu8
#endif
