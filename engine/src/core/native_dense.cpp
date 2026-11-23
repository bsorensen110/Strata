#include "strata/core/native_dense.hpp"
#include "strata/core/gfx11_extra.hpp"
#include "strata/core/run_manifest.hpp"
#include <cstdlib>
#include "strata/core/weights.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/platform/memory.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <set>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

namespace strata::core {
namespace {
// `load` packs a stage's native projections (dense, packed copies, HC Q8_0, and scratch) at 256B alignment
// into one contiguous device arena, so the driver's 2 MiB granule rounding hits ONCE per arena instead of once
// per tensor (previously 300x 6.73 MiB cost 2400 vs 2019 MiB, and 192x 3.32 MiB HC tensors lost ~130 MiB).
int g_layer_lb = -1, g_layer_le = -1;   // set_layer_range; -1: every layer
bool in_range(const std::string& name) {
    if (g_layer_lb < 0 || name.rfind("blk.", 0) != 0) return true;
    const int l = std::atoi(name.c_str() + 4);
    return l >= g_layer_lb && l < g_layer_le;
}
// S23 experiment (STRATA_HC_Q8=1): the hyper-connection projections' Q8_0 bytes for the verify read
bool hc_q8_env() {
    // STRATA_HC_Q8: 1 = on (the GGUF's own Q8_0 projections, or - a bf16 pack - a Q8_0 copy quantized at load); 0 = off;
    // unset / auto = on for gfx12 (RDNA4: the R9700 decode +2.5 to +4%, KL 0.0003-0.0010 over its noise floor) unless
    // STRATA_GFX12_DEFAULTS=0; also on for gfx1151 (Strix Halo, STRATA_HC_Q8=0 turns it off), off elsewhere
    static const bool on = [] {
        const char* v = std::getenv("STRATA_HC_Q8");
        if (v != nullptr && v[0] == '1') return true;
        if (v != nullptr && v[0] != '\0' && std::strcmp(v, "auto") != 0) return false;
#if defined(STRATA_USE_HIP)
        {   // Strix Halo (gfx1151): on by default (decode +9.5%, json/code/tool/chat/agent +7.9..+10.7%); the answers differ in
            // near-tie tokens by about the size of the engine's own CPU-vs-GPU expert rounding (docs/AMD_HIP.md);
            // STRATA_HC_Q8=0 keeps the bf16 read
            int dev1151 = 0;
            cudaDeviceProp p1151;
            const bool g1151 = cudaGetDevice(&dev1151) == cudaSuccess && cudaGetDeviceProperties(&p1151, dev1151) == cudaSuccess &&
                               strata::kernels::gfx_arch_is_gfx1151(p1151.gcnArchName);
            (void) cudaGetLastError();
            if (g1151) {
                std::fprintf(stderr, "strata: Strix Halo (gfx1151): 8-bit hyper-connection weights are on (decode about 9%% faster, "
                                     "answers can differ in near-tie tokens); STRATA_HC_Q8=0 keeps the bf16 read\n");
                return true;
            }
        }
        {   // Discrete RDNA3 (gfx1100 / gfx1101 / gfx1102): Q8_0 hyper-connection projections halve verify HC memory traffic
            if (!strata::core::gfx11_wmma_defaults_disabled(strata::qwen_defaults_026())) {
                int dev11 = 0;
                cudaDeviceProp p11;
                const bool g11 = cudaGetDevice(&dev11) == cudaSuccess && cudaGetDeviceProperties(&p11, dev11) == cudaSuccess &&
                                 !p11.integrated && strata::kernels::gfx_arch_is_discrete_rdna3(p11.gcnArchName);
                (void) cudaGetLastError();
                if (g11) {
                    std::fprintf(stderr, "strata: gfx11 (RDNA3): 8-bit hyper-connection weights are on (quantized at load, "
                                         "decode faster, answers can differ slightly); STRATA_HC_Q8=0 or STRATA_GFX11_DEFAULTS=0 keeps the bf16 read\n");
                    return true;
                }
            }
        }
        const char* d = std::getenv("STRATA_GFX12_DEFAULTS");
        if (d != nullptr && d[0] == '0') return false;
        int dev = 0;
        cudaDeviceProp p;
        const bool g12 = cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&p, dev) == cudaSuccess &&
                         strata::kernels::gfx_arch_is_gfx12(p.gcnArchName);
        (void) cudaGetLastError();
        if (g12) std::fprintf(stderr, "strata: gfx12 (RDNA4): the hyper-connection projections are read as Q8_0 (native GGUF Q8_0 when present, "
                                      "else quantized at load); STRATA_HC_Q8=0 keeps the bf16 read\n");
        return g12;
#else
        return false;
#endif
    }();
    return on;
}
// STRATA_HC_Q8_ALL_STAGES=1: the Q8_0 hyper-connection rows of every layer are copied to every stage (the behaviour before #3559)
bool hc_q8_stage_local() {
    static const bool on = [] { const char* v = std::getenv("STRATA_HC_Q8_ALL_STAGES"); return !(v != nullptr && v[0] == '1'); }();
    return on;
}
// STRATA_HC_Q8_INJECT=1: the F32 inject rows get a Q8_0 copy too (S25)
bool hc_q8_inject_env() {
    static const bool on = [] { const char* v = std::getenv("STRATA_HC_Q8_INJECT"); return v && v[0] == '1'; }();
    return on;
}
bool hc_q8_name(const std::string& name) {
    if (name == "output_hc_down.weight" || name == "output_hc_up.weight") return true;   // S25: the final mixer too
    if (name.rfind("blk.", 0) != 0) return false;
    static const char* suffixes[] = {".hc_attn_down.weight", ".hc_attn_up.weight", ".hc_attn_inject.weight",
                                     ".hc_ffn_down.weight", ".hc_ffn_up.weight", ".hc_ffn_inject.weight"};
    for (const char* suffix : suffixes) if (name.ends_with(suffix)) return true;
    return false;
}
// S25: the GGUF keeps the inject rows as F32 - quantized here to Q8_0 (ggml's reference rounding) so the Q8_0 read
// covers every hc projection
uint16_t half_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0) {                                   // subnormal or zero
        if (e < -10) return (uint16_t) sign;
        m |= 0x800000u;
        const uint32_t shift = (uint32_t) (14 - e);
        uint32_t h = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1u))) ++h;
        return (uint16_t) (sign | h);
    }
    if (e >= 31) return (uint16_t) (sign | 0x7c00u);
    uint32_t h = ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) (sign | h);
}
std::vector<uint8_t> q8_0_of(const float* x, uint64_t n) {
    std::vector<uint8_t> out((size_t) (n / 32 * 34));
    for (uint64_t b = 0; b < n / 32; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = (std::max)(amax, std::fabs(x[b * 32 + j]));
        const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        uint8_t* o = out.data() + b * 34;
        const uint16_t hb = half_bits(d);
        std::memcpy(o, &hb, 2);
        for (int j = 0; j < 32; ++j) o[2 + j] = (uint8_t) (int8_t) std::lround(x[b * 32 + j] * id);
    }
    return out;
}
// A bf16 matrix (the pack file's values) as Q8_0, ggml's reference rounding; the blocks are independent, so a few threads
std::vector<uint8_t> q8_0_of_bf16(const uint16_t* x, uint64_t n) {
    std::vector<uint8_t> out((size_t) (n / 32 * 34));
    const uint64_t nb = n / 32;
    auto work = [&](uint64_t b0, uint64_t b1) {
        for (uint64_t b = b0; b < b1; ++b) {
            float v[32], amax = 0.0f;
            for (int j = 0; j < 32; ++j) {
                const uint32_t u = (uint32_t) x[b * 32 + j] << 16;
                std::memcpy(&v[j], &u, 4);
                amax = (std::max)(amax, std::fabs(v[j]));
            }
            const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
            uint8_t* o = out.data() + b * 34;
            const uint16_t hb = half_bits(d);
            std::memcpy(o, &hb, 2);
            for (int j = 0; j < 32; ++j) o[2 + j] = (uint8_t) (int8_t) std::lround(v[j] * id);
        }
    };
    const unsigned nt = (std::min)(8u, (std::max)(1u, std::thread::hardware_concurrency()));
    std::vector<std::thread> pool;
    for (unsigned i = 0; i < nt; ++i) pool.emplace_back(work, nb * i / nt, nb * (i + 1) / nt);
    for (auto& th : pool) th.join();
    return out;
}
bool eligible(const strata::TensorInfo& tensor, bool include_ple_key) {
    const auto& name = tensor.name;
    if (name.rfind("blk.", 0) != 0) return false;
    // Match the native PLE kernel: Q2_0, IQ3_XXS, IQ4_XS and Q8_0 (UD-Q4_K_XL). Other keys retain the packed BF16
    // fallback.
    if (name == "blk.1.ple_key.weight")
        return include_ple_key && (tensor.type == 42 || tensor.type == 18 || tensor.type == 23 || tensor.type == 8);
    static const char* suffixes[] = {".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight",
        ".attn_q.weight", ".attn_k.weight", ".attn_v.weight", ".attn_output.weight",
        ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight", ".ffn_down_shexp.weight"};
    for (const char* suffix : suffixes) if (name.ends_with(suffix)) return true;
    return false;
}
// #4741: a native-eligible projection a pack keeps unquantized (BF16, GGUF type 30; Heretic v2 IQ4_NL has its layer-0 attn_gate so)
// has no native kernel.  Its siblings are Q8_0, so it is served as Q8_0 made from the file's BF16 values (ggml's rounding), and
// the dispatchers that need a native projection find one.  The pack's BF16 copy stays where it is.
bool bf16_as_q8_0(const strata::TensorInfo& tensor, const WeightRef& ref) {
    return tensor.type == 30 && tensor.name != "blk.1.ple_key.weight" && tensor.shape.size() == 2 &&
           !ref.quantized() && ref.kind == WeightKind::Bf16InF32 &&
           ref.ne0 > 0 && ref.ne0 <= INT_MAX && ref.ne1 > 0 && ref.ne1 <= INT_MAX && ref.ne0 % 32 == 0 &&
           tensor.shape[0] == (uint64_t) ref.ne0 && tensor.shape[1] == (uint64_t) ref.ne1;
}
struct DeviceFree { void operator()(void* p) const { if (p) (void) cudaFree(p); } };
using DevicePtr = std::unique_ptr<void, DeviceFree>;
bool dense_arena_on() {
    const char* v = std::getenv("STRATA_NATIVE_DENSE_ARENA");
    return v != nullptr && v[0] == '1';
}
}

bool NativeDense::served_names(const std::vector<std::string>& shards, bool include_ple_key,
                               std::set<std::string>& out, std::string& err) {
    try {
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            for (const auto& tensor : gguf.tensors())
                if (eligible(tensor, include_ple_key) && strata::kernels::native_mmvq_supported(tensor.type) &&
                    tensor.shape.size() == 2)
                    out.insert(tensor.name);
        }
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}

void NativeDense::set_layer_range(int lb, int le) { g_layer_lb = lb; g_layer_le = le; }
// The byte walk `load` does for the layers [lb, le), without the allocations: the same filters in the same order,
// reading the GGUF headers and the canonical table only, so it needs no device.  `load` packs a stage's tensors at
// 256B alignment into one contiguous device arena, rounded up ONCE to the 2 MiB granule (#1238, Audit Finding 10).
// A repeated name is skipped rather than refused (`load` refuses it).
bool NativeDense::weight_bytes_for(const std::vector<std::string>& shards, WeightTable& table, bool include_ple_key,
                                   int64_t lb, int64_t le, uint64_t& out, std::string& err) {
    return weight_bytes_for(shards, table.all(), include_ple_key, lb, le, out, err);
}

bool NativeDense::weight_bytes_for(const std::vector<std::string>& shards, const std::map<std::string, WeightRef>& table,
                                   bool include_ple_key, int64_t lb, int64_t le, uint64_t& out, std::string& err) {
    out = 0;
    try {
        std::set<std::string> seen, hc_seen;
        uint64_t payload = 0, separate = 0;
        auto add = [&](uint64_t bytes) { payload += arena_align(bytes); separate += alloc_bytes(bytes); };
        auto in_stage = [&](const std::string& name) {   // a blk.<l>. tensor of layers [lb, le); the rest belongs to every stage
            if (name.rfind("blk.", 0) != 0) return true;
            const long l = std::strtol(name.c_str() + 4, nullptr, 10);
            return l >= lb && l < le;
        };
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            for (const auto& tensor : gguf.tensors()) {
                // STRATA_HC_Q8: `load` copies the GGUF's Q8_0 hyper-connection rows of its layers (and the final mixer's)
                // to the card, one cudaMalloc each, and the pack's bf16 rows get a Q8_0 copy below: priced as `load` makes them
                if (hc_q8_env() && hc_q8_name(tensor.name) && tensor.shape.size() == 2) {
                    const bool f32_inject = hc_q8_inject_env() && tensor.type == 0 && tensor.name.ends_with("_inject.weight");
                    if ((tensor.type == 8 || f32_inject) && (!hc_q8_stage_local() || in_stage(tensor.name)) &&
                        hc_seen.insert(tensor.name).second) {
                        const auto found = table.find(tensor.name);
                        if (found != table.end() && found->second.ne0 % 32 == 0 &&
                            tensor.shape[0] == (uint64_t) found->second.ne0 && tensor.shape[1] == (uint64_t) found->second.ne1)
                            add((uint64_t) found->second.ne0 / 32 * 34 * (uint64_t) found->second.ne1);
                    }
                }
                if (!eligible(tensor, include_ple_key)) continue;
                if (tensor.name.rfind("blk.", 0) == 0 && tensor.name.find("ple") == std::string::npos &&
                    !in_stage(tensor.name)) {
                    const long l = std::strtol(tensor.name.c_str() + 4, nullptr, 10);
                    if (l < lb || l >= le) continue;
                }
                if (!seen.insert(tensor.name).second) continue;
                auto found = table.find(tensor.name);
                if (found == table.end()) {
                    err = "native dense: tensor absent from canonical table: " + tensor.name;
                    return false;
                }
                const auto& ref = found->second;
                if (bf16_as_q8_0(tensor, ref)) {   // #4741
                    add((uint64_t) ref.ne0 / 32 * 34 * (uint64_t) ref.ne1);
                    continue;
                }
                if (!strata::kernels::native_mmvq_supported(tensor.type)) continue;
                if (tensor.name == "blk.1.ple_key.weight" && !ref.quantized()) continue;
                if (!ref.quantized() || tensor.shape.size() != 2 ||
                    ref.ne0 <= 0 || ref.ne0 > INT_MAX || ref.ne1 <= 0 || ref.ne1 > INT_MAX ||
                    tensor.shape[0] != (uint64_t) ref.ne0 || tensor.shape[1] != (uint64_t) ref.ne1) {
                    err = "native dense: incompatible matrix " + tensor.name;
                    return false;
                }
                add(strata::kernels::native_mmvq_weight_bytes(tensor.type, (int) ref.ne0, (int) ref.ne1));
            }
        }
        if (hc_q8_env())   // a bf16 pack: `load` quantizes the layers' down / up projections that the GGUF does not carry as Q8_0
            for (const auto& [name, ref] : table) {
                if (name.rfind("blk.", 0) != 0 || !in_stage(name) || !hc_q8_name(name) || hc_seen.count(name) ||
                    !(name.ends_with("_down.weight") || name.ends_with("_up.weight")))
                    continue;
                if (ref.ne0 % 32 == 0 && ref.elements == ref.ne0 * ref.ne1)
                    add((uint64_t) ref.elements / 32 * 34);
            }
        out = dense_arena_on() ? arena_alloc_bytes(payload) : separate;
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}
bool NativeDense::keep_unquantized_ple_key(const std::string& pack_dir, std::set<std::string>& skip,
                                           std::string& err) {
    const std::string key = "blk.1.ple_key.weight";
    if (!skip.count(key)) return true;
    int code_bits = -1;
    if (!WeightTable::index_code_bits(pack_dir, key, code_bits, err)) return false;
    if (code_bits == 0) skip.erase(key);
    return true;
}

// STRATA_HC_Q8 without a bf16 copy: the two dequantization scratches (one bf16 matrix each) of `hc_bf16_view`, per device,
// allocated by `load` - never on first use, so a graph capture cannot meet the allocation
static void* g_hc_scratch[64][2][2] = {};     // [device][class][slot]
static cudaEvent_t g_hc_ev[64][2] = {};        // [device][slot]: the prompt path's last reader of the slot
static bool g_hc_ev_set[64][2] = {};
static void* g_hc_ev_stream[64][2] = {};       // [device][slot]: stream that recorded g_hc_ev (same-stream wait elision)

NativeDense::~NativeDense() {
    release();
}

void NativeDense::release() {
    const auto base = reinterpret_cast<uintptr_t>(arena_);
    for (auto& dev : g_hc_scratch)
        for (auto& cl : dev)
            for (void*& b : cl) {
                if (b == nullptr) continue;
                const auto addr = reinterpret_cast<uintptr_t>(b);
                if ((arena_ != nullptr && addr >= base && (uint64_t) (addr - base) < arena_size_) ||
                    std::find(allocations_.begin(), allocations_.end(), b) != allocations_.end()) b = nullptr;
            }
    for (const void* p : packed_keys_) strata::kernels::native_q8_0_packed_unregister(p);
    if (arena_) (void) cudaFree(arena_);
    for (void* p : allocations_) (void) cudaFree(p);
    allocations_.clear();
    arena_ = nullptr;
    arena_size_ = 0;
    scratch_ = nullptr;
    packed_keys_.clear();
    layer_bytes_.clear();
    bytes_ = 0;
    allocated_ = 0;
    tensor_count_ = 0;
}

uint64_t NativeDense::bytes_in(int64_t lo, int64_t hi) const {
    uint64_t n = 0;
    for (int64_t l = std::max<int64_t>(lo, 0); l < hi && l < (int64_t) layer_bytes_.size(); ++l) n += layer_bytes_[(size_t) l];
    return n;
}

bool NativeDense::load(const std::vector<std::string>& shards, WeightTable& table, std::string& err,
                       bool include_ple_key, int64_t layer_lo, int64_t layer_hi) {
    auto outside = [&](const std::string& name) {   // a blk.<l>. tensor of another stage's layers
        if (layer_hi < 0 || name.rfind("blk.", 0) != 0) return false;
        const long l = std::strtol(name.c_str() + 4, nullptr, 10);
        return l < layer_lo || l >= layer_hi;
    };
    auto layer_of = [&](const std::string& name) -> int64_t {   // the layer of a blk.<l>. tensor, -1 otherwise
        if (name.rfind("blk.", 0) != 0) return -1;
        return std::strtoll(name.c_str() + 4, nullptr, 10);
    };
    if (arena_ || scratch_ || !allocations_.empty() || !packed_keys_.empty()) { err = "native dense: already loaded"; return false; }
    if (shards.empty()) { err = "native dense: at least one GGUF shard is required"; return false; }
    try {
        struct DenseSpec {
            size_t shard_idx = 0;
            size_t tensor_idx = 0;
            WeightRef* ref = nullptr;
            int type = 0;
            uint64_t bytes = 0;
            bool bf16_to_q8 = false;
            bool has_packed = false;
            size_t data_slice = 0;
            size_t packed_slice = 0;
        };
        struct HcGgufSpec {
            size_t shard_idx = 0;
            size_t tensor_idx = 0;
            WeightRef* ref = nullptr;
            uint64_t bytes = 0;
            bool from_gguf = false;
            bool f32_inject = false;
            size_t slice = 0;
        };
        struct HcPackSpec {
            std::string name;
            WeightRef* ref = nullptr;
            uint64_t nb = 0;
            uint64_t bytes = 0;
            bool skipped = false;
            size_t slice = 0;
        };

        std::vector<std::unique_ptr<strata::GgufFile>> ggufs;
        ggufs.reserve(shards.size());
        std::vector<DenseSpec> dense_specs;
        std::vector<HcGgufSpec> hc_gguf_specs;
        std::vector<HcPackSpec> hc_pack_specs;
        std::vector<uint64_t> slice_sizes;
        auto add_slice = [&](uint64_t bytes) -> size_t {
            const size_t id = slice_sizes.size();
            slice_sizes.push_back(bytes);
            return id;
        };

        std::set<std::string> seen;
        std::set<std::string> hc_q8_seen;
        int max_in = 0;
        uint64_t total = 0, hc_q8_bytes = 0, hc_q8_skipped_bytes = 0, hc_q8_skipped = 0;
        std::vector<uint64_t> per_layer;
        uint64_t split_count = 0, split_tensors = 0;
        std::set<uint64_t> split_numbers;
        bool have_architecture = false;
        const bool use_hc_q8 = hc_q8_env();

        // Pass 1: validate every shard/tensor and collect contiguous arena slices before allocating any VRAM.
        for (size_t sidx = 0; sidx < shards.size(); ++sidx) {
            ggufs.push_back(std::make_unique<strata::GgufFile>(shards[sidx]));
            const strata::GgufFile& gguf = *ggufs.back();
            const auto* count = gguf.get("split.count");
            const auto* number = gguf.get("split.no");
            const auto* tensors = gguf.get("split.tensors.count");
            if (gguf.get("general.architecture") && (!number || number->u == 0)) {
                // splitter may copy general.architecture into every shard (Huihui abliterated re-split)
                err = strata::check_engine_architecture(gguf);
                if (!err.empty()) return false;
                have_architecture = true;
                if (count && number && tensors && number->u == 0 && count->u > 1) {
                    split_count = count->u;
                    split_tensors = tensors->u;
                }
            } else if (!have_architecture || !split_count || !count || !number || !tensors ||
                       count->u != split_count || number->u == 0 || number->u >= split_count ||
                       tensors->u != split_tensors) {
                err = "native dense: additional shard must match the architecture-validated first shard's split metadata";
                return false;
            }
            if (number && !split_numbers.insert(number->u).second) {
                err = "native dense: duplicate split shard number"; return false;
            }
            std::vector<uint64_t> offsets;
            offsets.reserve(gguf.tensors().size());
            for (const auto& tensor : gguf.tensors()) offsets.push_back(tensor.offset);
            std::sort(offsets.begin(), offsets.end());
            if (std::adjacent_find(offsets.begin(), offsets.end()) != offsets.end()) {
                err = "native dense: tensor payload offsets overlap"; return false;
            }
            // Validate every directory span, including tensors we do not upload:
            // an ignored tensor must not overlap the native matrix that follows it.
            const uint64_t payload = gguf.file_size() - gguf.data_start();
            for (const auto& tensor : gguf.tensors()) {
                int block_elements = 0, block_bytes = 0;
                uint64_t elements = 1;
                if (tensor.shape.empty() || !strata::block_geometry(tensor.type, block_elements, block_bytes) ||
                    tensor.shape[0] % (uint64_t) block_elements != 0) {
                    err = "native dense: invalid block geometry " + tensor.name; return false;
                }
                for (uint64_t dimension : tensor.shape) {
                    if (!dimension || elements > (std::numeric_limits<uint64_t>::max)() / dimension) {
                        err = "native dense: invalid tensor extent " + tensor.name; return false;
                    }
                    elements *= dimension;
                }
                const uint64_t blocks = elements / (uint64_t) block_elements;
                if (blocks > (std::numeric_limits<uint64_t>::max)() / (uint64_t) block_bytes) {
                    err = "native dense: tensor byte count overflow " + tensor.name; return false;
                }
                const uint64_t bytes = blocks * (uint64_t) block_bytes;
                if (tensor.offset > payload || bytes > payload - tensor.offset) {
                    err = "native dense: truncated payload " + tensor.name; return false;
                }
                const auto next = std::upper_bound(offsets.begin(), offsets.end(), tensor.offset);
                if (next != offsets.end() && bytes > *next - tensor.offset) {
                    err = "native dense: overlapping payload " + tensor.name; return false;
                }
                // the uploads below read these: ask for them now so the reads overlap
                if (eligible(tensor, include_ple_key) && !outside(tensor.name) &&
                    strata::kernels::native_mmvq_supported(tensor.type))
                    strata::platform::advise_willneed(gguf.tensor_data(tensor), bytes);
            }
            const auto& gtensors = gguf.tensors();
            for (size_t tidx = 0; tidx < gtensors.size(); ++tidx) {
                const auto& tensor = gtensors[tidx];
                if (!eligible(tensor, include_ple_key) || outside(tensor.name)) continue;
                if (!in_range(tensor.name) && tensor.name.find("ple") == std::string::npos) continue;
                if (!seen.insert(tensor.name).second) {
                    err = "native dense: duplicate tensor " + tensor.name; return false;
                }
                auto found = table.table_.find(tensor.name);
                if (found == table.table_.end()) {
                    err = "native dense: tensor absent from canonical table: " + tensor.name; return false;
                }
                auto& ref = found->second;
                if (ref.native_data) { err = "native dense: override already attached"; return false; }
                if (bf16_as_q8_0(tensor, ref)) {   // #4741: a BF16 projection of the pack, served as Q8_0
                    const uint64_t bytes = (uint64_t) ref.ne0 / 32 * 34 * (uint64_t) ref.ne1;
                    max_in = (std::max)(max_in, (int) ref.ne0);
                    total += bytes;
                    dense_specs.push_back(DenseSpec{sidx, tidx, &ref, 8, bytes, true, false, add_slice(bytes), 0});
                    continue;
                }
                if (!strata::kernels::native_mmvq_supported(tensor.type)) continue;
                // #326: the pack keeps an unquantized (--compat-bf16) key, which the PLE reads from the arena
                if (tensor.name == "blk.1.ple_key.weight" && !ref.quantized()) continue;
                if (!ref.quantized() || tensor.shape.size() != 2 ||
                    ref.ne0 <= 0 || ref.ne0 > INT_MAX || ref.ne1 <= 0 || ref.ne1 > INT_MAX ||
                    tensor.shape[0] != (uint64_t) ref.ne0 || tensor.shape[1] != (uint64_t) ref.ne1) {
                    err = "native dense: incompatible matrix " + tensor.name; return false;
                }
                const auto bytes = strata::kernels::native_mmvq_weight_bytes(tensor.type, (int) ref.ne0, (int) ref.ne1);
                const bool has_packed = tensor.type == 8 && tensor.name != "blk.1.ple_key.weight" &&
                                        strata::kernels::native_q8_0_packed_enabled() &&
                                        strata::kernels::native_q8_0_packed_eligible((int) ref.ne0, (int) ref.ne1);
                max_in = (std::max)(max_in, (int) ref.ne0);
                total += bytes;
                const int64_t layer = layer_of(tensor.name);
                if (layer >= 0) {
                    if (per_layer.size() <= (size_t) layer) per_layer.resize((size_t) layer + 1, 0);
                    per_layer[(size_t) layer] += bytes;
                }
                const size_t data_slice = add_slice(bytes);
                const size_t packed_slice = has_packed ? add_slice(bytes) : 0;
                dense_specs.push_back(DenseSpec{
                    sidx, tidx, &ref, (int) tensor.type, (uint64_t) bytes, false, has_packed, data_slice, packed_slice});
            }
            if (strata::core::manifest::enabled()) {
                const char* v = std::getenv("STRATA_HC_Q8");
                strata::core::manifest::instance().path("dense", "hc_q8", use_hc_q8,
                    use_hc_q8 ? (v != nullptr && v[0] == '1' ? "STRATA_HC_Q8=1" : "architecture default (gfx12 / gfx1151 HIP)")
                              : (v != nullptr && v[0] != '\0' && std::strcmp(v, "auto") != 0 ? "STRATA_HC_Q8 turned it off"
                                                                                             : "off by default on this backend"));
            }
            if (use_hc_q8) {
                for (size_t tidx = 0; tidx < gtensors.size(); ++tidx) {
                    const auto& tensor = gtensors[tidx];
                    // S25: the F32 inject rows hold BF16-exact values: the read takes the pack's BF16 rows unless
                    // STRATA_HC_Q8_INJECT=1 asks for a Q8_0 copy
                    const bool f32_inject = hc_q8_inject_env() && tensor.type == 0 && tensor.name.ends_with("_inject.weight");
                    if ((tensor.type != 8 && !f32_inject) || !hc_q8_name(tensor.name) || tensor.shape.size() != 2) continue;
                    // a trimmed layer split: another stage's layers are never read here, so their rows are not copied (the upload
                    // loop above and the bf16 pack path below already filter this way; PR #3559, havok2-htwo).
                    // STRATA_HC_Q8_ALL_STAGES=1 copies them as before.
                    if (hc_q8_stage_local() && (outside(tensor.name) || !in_range(tensor.name))) {
                        hc_q8_skipped_bytes += (uint64_t) tensor.shape[0] / 32 * 34 * (uint64_t) tensor.shape[1];
                        ++hc_q8_skipped;
                        continue;
                    }
                    auto found = table.table_.find(tensor.name);
                    if (found == table.table_.end() || found->second.hc_q8 != nullptr ||
                        hc_q8_seen.count(tensor.name) != 0)
                        continue;
                    auto& ref = found->second;
                    if (tensor.shape[0] != (uint64_t) ref.ne0 || tensor.shape[1] != (uint64_t) ref.ne1 || ref.ne0 % 32 != 0) {
                        err = "native dense (STRATA_HC_Q8): incompatible matrix " + tensor.name; return false;
                    }
                    const uint64_t bytes = (uint64_t) ref.ne0 / 32 * 34 * (uint64_t) ref.ne1;
                    hc_q8_seen.insert(tensor.name);
                    hc_q8_bytes += bytes;
                    strata::core::manifest::instance().tensors("dense_hc_q8_copy", f32_inject ? "F32->Q8_0" : "Q8_0", 1, bytes);
                    hc_gguf_specs.push_back(
                        HcGgufSpec{sidx, tidx, &ref, bytes, tensor.type == 8, f32_inject, add_slice(bytes)});
                }
            }
        }
        if (dense_specs.empty()) { err = "native dense: no supported GDN/QSA matrices in supplied shards"; return false; }

        if (use_hc_q8) {
            // a bf16 pack stores no Q8_0 copy: quantize the layers' and final mixer's down / up projections. A row the
            // arena kept (--keep-canonical, a layer split) is quantized on the GPU from the arena's bf16; a skipped row
            // (the single-GPU default: add_hc_skip) from the pack file on the host, and has no bf16 copy at all. The
            // inject rows stay bf16 (82 KB).
            for (auto& [name, ref] : table.table_) {
                if (ref.hc_q8 != nullptr || hc_q8_seen.count(name) != 0 || outside(name) ||
                    !hc_q8_name(name) || !(name.ends_with("_down.weight") || name.ends_with("_up.weight"))) continue;
                const bool skipped = ref.data == nullptr;
                if (skipped && ref.resident) continue;
                if (ref.ne0 % 32 != 0 || ref.elements != ref.ne0 * ref.ne1 || (!skipped && ref.bytes != (uint64_t) ref.elements * 2)) {
                    err = "native dense (STRATA_HC_Q8): " + name + " is not a bf16 matrix"; return false;
                }
                const uint64_t nb = (uint64_t) ref.elements / 32, bytes = nb * 34;
                hc_q8_seen.insert(name);
                hc_q8_bytes += bytes;
                hc_pack_specs.push_back(HcPackSpec{name, &ref, nb, bytes, skipped, add_slice(bytes)});
            }
        }

        int hc_dev = 0;
        bool need_hc_scratch = false;
        size_t hc_scratch_slice[2][2] = {};
        if (use_hc_q8) {   // the scratches of hc_bf16_view (rows without a bf16 copy), here and not on first use
            uint64_t elems = 0;
            for (const auto& [name, ref] : table.table_)
                if ((ref.hc_q8 != nullptr || hc_q8_seen.count(name) != 0) && ref.data == nullptr)
                    elems = (std::max)(elems, (uint64_t) ref.elements);
            if (elems != 0 && cudaGetDevice(&hc_dev) == cudaSuccess && hc_dev >= 0 && hc_dev < 64 &&
                g_hc_scratch[hc_dev][0][0] == nullptr) {
                need_hc_scratch = true;
                for (int c = 0; c < 2; ++c)
                    for (int i = 0; i < 2; ++i)
                        hc_scratch_slice[c][i] = add_slice(elems * 2);
            }
        }
        const size_t scratch_bytes = strata::kernels::native_q8_1_bytes(max_in);
        const size_t scratch_slice = add_slice((uint64_t) scratch_bytes);
        const ArenaPlan plan = plan_arena(slice_sizes);

        // Allocate each slice separately unless the arena is explicitly requested.
        void* raw_arena = nullptr;
        DevicePtr arena;
        std::vector<DevicePtr> owned_slices;
        std::vector<void*> slices;
        uint64_t allocated = 0;
        if (dense_arena_on()) {
            const auto st = cudaMalloc(&raw_arena, (size_t) plan.payload_bytes);
            if (st != cudaSuccess) { err = std::string("native dense arena: ") + cudaGetErrorString(st); return false; }
            arena.reset(raw_arena);
            for (size_t i = 0; i < slice_sizes.size(); ++i)
                slices.push_back(static_cast<uint8_t*>(raw_arena) + plan.offsets[i]);
            allocated = plan.allocated_bytes;
        } else {
            owned_slices.reserve(slice_sizes.size());
            slices.reserve(slice_sizes.size());
            for (uint64_t bytes : slice_sizes) {
                void* p = nullptr;
                const auto st = cudaMalloc(&p, (size_t) bytes);
                if (st != cudaSuccess) { err = std::string("native dense allocation: ") + cudaGetErrorString(st); return false; }
                owned_slices.emplace_back(p);
                slices.push_back(p);
                allocated += alloc_bytes(bytes);
            }
        }
        uint64_t packed_bytes = 0;
        size_t packed_count = 0;
        for (const auto& d : dense_specs) {
            const strata::GgufFile& gguf = *ggufs[d.shard_idx];
            const auto& tensor = gguf.tensors()[d.tensor_idx];
            void* dst = slices[d.data_slice];
            if (d.bf16_to_q8) {
                const uint64_t n = (uint64_t) d.ref->ne0 * (uint64_t) d.ref->ne1;
                const std::vector<uint8_t> q8 = q8_0_of_bf16((const uint16_t*) gguf.tensor_data(tensor), n);
                const auto st = cudaMemcpy(dst, q8.data(), (size_t) d.bytes, cudaMemcpyHostToDevice);
                if (st != cudaSuccess) { err = "native dense upload " + tensor.name + ": " + cudaGetErrorString(st); return false; }
                std::fprintf(stderr, "native dense: %s is BF16 in the file: served as Q8_0 like its siblings (#4741)\n",
                             tensor.name.c_str());
                continue;
            }
            auto st = cudaMemcpy(dst, gguf.tensor_data(tensor), (size_t) d.bytes, cudaMemcpyHostToDevice);
            if (st != cudaSuccess) { err = "native dense upload " + tensor.name + ": " + cudaGetErrorString(st); return false; }
            if (d.has_packed) {
                std::vector<uint8_t> host((size_t) d.bytes);
                strata::kernels::native_q8_0_pack_host(gguf.tensor_data(tensor), host.data(), (int) d.ref->ne0, (int) d.ref->ne1);
                void* packed_dst = slices[d.packed_slice];
                st = cudaMemcpy(packed_dst, host.data(), (size_t) d.bytes, cudaMemcpyHostToDevice);
                if (st != cudaSuccess) {
                    err = "native dense packed upload " + tensor.name + ": " + cudaGetErrorString(st);
                    return false;
                }
                packed_bytes += d.bytes;
                ++packed_count;
            }
            if (strata::core::manifest::enabled()) {   // STRATA_MANIFEST: the dense formats as held, and the packed copies
                strata::core::manifest::instance().tensors("dense", strata::ggml_type_name((uint32_t) tensor.type), 1, d.bytes);
                if (d.has_packed) strata::core::manifest::instance().tensors("dense_packed_copy", "Q8_0", 1, d.bytes);
            }
        }

        for (const auto& h : hc_gguf_specs) {
            const strata::GgufFile& gguf = *ggufs[h.shard_idx];
            const auto& tensor = gguf.tensors()[h.tensor_idx];
            void* dst = slices[h.slice];
            std::vector<uint8_t> q8;
            if (h.f32_inject) q8 = q8_0_of((const float*) gguf.tensor_data(tensor), (uint64_t) h.ref->ne0 * (uint64_t) h.ref->ne1);
            if (cudaMemcpy(dst, h.f32_inject ? (const void*) q8.data() : gguf.tensor_data(tensor),
                           (size_t) h.bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
                err = "native dense (STRATA_HC_Q8) upload " + tensor.name; return false;
            }
            strata::core::manifest::instance().tensors("dense_hc_q8_copy", h.f32_inject ? "F32->Q8_0" : "Q8_0", 1, h.bytes);
        }

        uint64_t made = 0;
        for (const auto& h : hc_pack_specs) {
            void* dst = slices[h.slice];
            if (h.skipped) {
                std::vector<uint16_t> host;
                if (!table.read_bf16_host(h.name, host, err)) return false;
                const std::vector<uint8_t> q8 = q8_0_of_bf16(host.data(), (uint64_t) host.size());
                if (cudaMemcpy(dst, q8.data(), (size_t) h.bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
                    err = "native dense (STRATA_HC_Q8) upload " + h.name; return false;
                }
            } else {
                strata::kernels::fused_gr_quantize_q8_0((const uint16_t*) h.ref->data, (uint8_t*) dst, (int64_t) h.nb, nullptr);
                ++made;
            }
        }
        if (made && (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)) {
            err = "native dense (STRATA_HC_Q8): quantizing the projections failed"; return false;
        }
        if (need_hc_scratch) {
            for (int i = 0; i < 2; ++i)
                if (g_hc_ev[hc_dev][i] == nullptr && cudaEventCreateWithFlags(&g_hc_ev[hc_dev][i], cudaEventDisableTiming) != cudaSuccess) {
                    err = "native dense (STRATA_HC_Q8): scratch event"; return false;
                }
        }

        // Reserve before publication so a host allocation failure cannot leave dangling references.
        allocations_.reserve(owned_slices.size());
        packed_keys_.reserve(packed_count);
        if (hc_q8_skipped != 0 && strata::core::manifest::enabled())
            strata::core::manifest::instance().tensors("dense_hc_q8_out_of_stage_not_copied", "Q8_0", hc_q8_skipped, hc_q8_skipped_bytes);
        try {
            for (const auto& d : dense_specs) {
                if (!d.has_packed) continue;
                void* data_ptr = slices[d.data_slice];
                strata::kernels::native_q8_0_packed_register(data_ptr, slices[d.packed_slice], (int) d.ref->ne0, (int) d.ref->ne1);
                packed_keys_.push_back(data_ptr);
            }
        } catch (...) {
            for (const void* p : packed_keys_) strata::kernels::native_q8_0_packed_unregister(p);
            packed_keys_.clear();
            throw;
        }

        // All checks, uploads, events, and registrations finished before publishing references.
        void* const scratch_ptr = slices[scratch_slice];
        for (const auto& d : dense_specs) {
            void* data_ptr = slices[d.data_slice];
            d.ref->native_data = data_ptr;
            d.ref->native_type = d.type;
            d.ref->native_bytes = d.bytes;
            d.ref->native_q8_1 = scratch_ptr;
        }
        for (const auto& h : hc_gguf_specs) {
            h.ref->hc_q8 = slices[h.slice];
            h.ref->hc_q8_from_gguf = h.from_gguf;
        }
        for (const auto& h : hc_pack_specs) {
            h.ref->hc_q8 = slices[h.slice];
            h.ref->hc_q8_from_gguf = false;
        }
        if (need_hc_scratch) {
            for (int c = 0; c < 2; ++c)
                for (int i = 0; i < 2; ++i)
                    g_hc_scratch[hc_dev][c][i] = slices[hc_scratch_slice[c][i]];
        }
        if (packed_count)
            std::fprintf(stderr, "native dense: STRATA_Q8_PACKED=1 packed %zu Q8_0 matrices for decode (+%.1f MiB)\n",
                         packed_count, packed_bytes / 1048576.0);
        arena_ = arena.release();
        arena_size_ = arena_ ? plan.payload_bytes : 0;
        for (auto& p : owned_slices) allocations_.push_back(p.release());
        scratch_ = scratch_ptr;
        tensor_count_ = dense_specs.size() + hc_gguf_specs.size() + hc_pack_specs.size();
        bytes_ = total + hc_q8_bytes;
        layer_bytes_ = std::move(per_layer);
        allocated_ = allocated;
        if (use_hc_q8) {
            uint64_t gguf_q8_bytes = 0, bf16_pack_bytes = 0;
            for (const auto& [name, ref] : table.table_) {
                if (!hc_q8_name(name) || ref.hc_q8 == nullptr) continue;
                const uint64_t bytes = (uint64_t) ref.ne0 / 32 * 34 * (uint64_t) ref.ne1;
                if (ref.hc_q8_from_gguf) gguf_q8_bytes += bytes;
                else bf16_pack_bytes += bytes;
            }
            std::fprintf(stderr,
                         "strata: STRATA_HC_Q8=1: %.2f GiB native GGUF Q8_0, %.2f GiB quantized from pack "
                         "hyper-connection projections\n",
                         (double) gguf_q8_bytes / 1073741824.0,
                         (double) bf16_pack_bytes / 1073741824.0);
        }
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}
bool NativeDense::gguf_carries_q8_hc(const std::vector<std::string>& shards) {
    try {
        for (const auto& path : shards) {
            const strata::GgufFile g(path);
            for (const auto& tensor : g.tensors())
                if (tensor.type == 8 && tensor.shape.size() == 2 && hc_q8_name(tensor.name) && !tensor.name.ends_with("_inject.weight")) return true;
        }
    } catch (const std::exception&) {}
    return false;
}

bool NativeDense::apply_hc_q8_default(const std::vector<std::string>& shards) {
    if (std::getenv("STRATA_HC_Q8") != nullptr || !gguf_carries_q8_hc(shards)) return false;
#if defined(_WIN32)
    _putenv_s("STRATA_HC_Q8", "1");
#else
    setenv("STRATA_HC_Q8", "1", 0);
#endif
    return true;
}

bool NativeDense::hc_q8_requested() { return hc_q8_env(); }

bool NativeDense::add_hc_skip(const std::string& pack_dir, std::set<std::string>& skip) {
    std::FILE* f = std::fopen((pack_dir + "/index.txt").c_str(), "rb");
    if (!f) return false;
    char line[1024], name[256];
    while (std::fgets(line, sizeof line, f)) {
        if (line[0] == '#' || std::sscanf(line, "%255s", name) != 1) continue;
        const std::string n = name;
        if (hc_q8_name(n) && (n.ends_with("_down.weight") || n.ends_with("_up.weight"))) skip.insert(n);
    }
    std::fclose(f);
    return true;
}

const uint16_t* NativeDense::hc_bf16_view(const WeightRef& w, void* stream, int slot, int cls) {
    if (w.data != nullptr) return (const uint16_t*) w.data;
    if (w.hc_q8 == nullptr) return nullptr;
    // the device's scratch (allocated at load, one pair per device): the prompt path and the single-token read run one
    // after the other on a device, and each use is stream-ordered after the previous GEMM that read the slot
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64 || slot < 0 || slot > 1 || cls < 0 || cls > 1) return nullptr;
    void* buf = g_hc_scratch[dev][cls][slot];
    if (buf == nullptr) return nullptr;
    // the prompt path: the scratch may still be read by a GEMM another stream enqueued (batch serving: several slots' prompts).
    // When the previous reader ran on the exact same stream, stream program order already serializes the reuse without an HSA barrier wait.
    if (cls == 0 && g_hc_ev_set[dev][slot] && g_hc_ev_stream[dev][slot] != stream) {
        cudaStreamWaitEvent((cudaStream_t) stream, g_hc_ev[dev][slot], 0);
        g_hc_ev_stream[dev][slot] = stream;
    }
    strata::kernels::fused_gr_dequant_q8_0((const uint8_t*) w.hc_q8, (uint16_t*) buf, w.elements / 32, stream);
    return (const uint16_t*) buf;
}

bool NativeDense::hc_bf16_view_pair(const WeightRef& w0, const WeightRef& w1, void* stream, const uint16_t*& out0,
                                    const uint16_t*& out1, int cls) {
    if (w0.data == nullptr && w0.hc_q8 != nullptr && w1.data == nullptr && w1.hc_q8 != nullptr &&
        w0.elements == w1.elements) {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64 || cls < 0 || cls > 1) {
            out0 = nullptr; out1 = nullptr; return false;
        }
        void* buf0 = g_hc_scratch[dev][cls][0];
        void* buf1 = g_hc_scratch[dev][cls][1];
        if (buf0 == nullptr || buf1 == nullptr) {
            out0 = nullptr; out1 = nullptr; return false;
        }
        if (cls == 0) {
            for (int slot = 0; slot < 2; ++slot) {
                if (g_hc_ev_set[dev][slot] && g_hc_ev_stream[dev][slot] != stream) {
                    cudaStreamWaitEvent((cudaStream_t) stream, g_hc_ev[dev][slot], 0);
                    g_hc_ev_stream[dev][slot] = stream;
                }
            }
        }
        strata::kernels::fused_gr_dequant_q8_0_pair((const uint8_t*) w0.hc_q8, (uint16_t*) buf0,
                                                    (const uint8_t*) w1.hc_q8, (uint16_t*) buf1,
                                                    w0.elements / 32, stream);
        out0 = (const uint16_t*) buf0;
        out1 = (const uint16_t*) buf1;
        return true;
    }
    out0 = hc_bf16_view(w0, stream, 0, cls);
    out1 = hc_bf16_view(w1, stream, 1, cls);
    return out0 != nullptr && out1 != nullptr;
}

void NativeDense::hc_bf16_release(void* stream, int slot, int cls) {
    int dev = 0;
    if (cls != 0 || slot < 0 || slot > 1 || cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64 || g_hc_ev[dev][slot] == nullptr) return;
    cudaEventRecord(g_hc_ev[dev][slot], (cudaStream_t) stream);
    g_hc_ev_set[dev][slot] = true;
    g_hc_ev_stream[dev][slot] = stream;
}

void NativeDense::hc_bf16_release_pair(void* stream, int cls) {
    int dev = 0;
    if (cls != 0 || cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return;
    for (int slot = 0; slot < 2; ++slot) {
        if (g_hc_ev[dev][slot] != nullptr) {
            cudaEventRecord(g_hc_ev[dev][slot], (cudaStream_t) stream);
            g_hc_ev_set[dev][slot] = true;
            g_hc_ev_stream[dev][slot] = stream;
        }
    }
}
} // namespace strata::core
