// src/core/expert_cache.cpp - R4's slot storage and residency table.  Read the header first.
#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_cache_refuse.hpp"

// The gfx906 compatibility layer and Metal do not expose the VMM driver API. STRATA_EC_NO_VMM remains
// an explicit build-time kill switch; HIP VMM-capable devices use the shared Vmm implementation.
#if (defined(STRATA_HIP_GFX906) || defined(STRATA_GPU_METAL) || defined(SYCL_LANGUAGE_VERSION)) && \
    !defined(STRATA_EC_NO_VMM)
#define STRATA_EC_NO_VMM 1
#endif

#include <thread>
#include <mutex>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <filesystem>
#include <cstdlib>
#include <utility>
#include <cstring>
#include <cmath>
#include <functional>
#include <limits>

// (not in a SYCL build: its CUDA names are a compatibility layer without the PCI ids NVML is matched by)
#if defined(_WIN32) && !defined(STRATA_EC_NO_VMM) && !defined(SYCL_LANGUAGE_VERSION)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#define STRATA_EC_NVML_FREE 1
#endif

namespace strata::core {

#if defined(STRATA_EC_NVML_FREE)
namespace {

// STRATA_VRAM_FREE=device (opt-in, Windows + NVIDIA).  Under WDDM cudaMemGetInfo answers for THIS process: memory
// another process holds on the card (the image encoder, a browser, another engine) is not in its free figure.
// Measured on an RTX 5090, driver 610.88: with 6 GiB written by another process, cudaMemGetInfo in a second process
// still read 30,994 MiB free where NVML read 24,348.  NVML (nvml.dll, installed with every driver) counts the whole
// card.  The library is loaded from the system directory only, at first use; false when it cannot answer.
bool nvml_device_free(size_t& out) {
    struct Memory { unsigned long long total, free, used; };
    typedef int (*InitFn)(void);
    typedef int (*ByBusFn)(const char*, void**);
    typedef int (*MemFn)(void*, Memory*);
    struct Lib {
        ByBusFn by_bus = nullptr;
        MemFn mem = nullptr;
        Lib() {
            HMODULE h = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
            if (h == nullptr) return;
            const auto init = (InitFn) (void*) GetProcAddress(h, "nvmlInit_v2");
            const auto bus = (ByBusFn) (void*) GetProcAddress(h, "nvmlDeviceGetHandleByPciBusId_v2");
            const auto m = (MemFn) (void*) GetProcAddress(h, "nvmlDeviceGetMemoryInfo");
            if (init == nullptr || bus == nullptr || m == nullptr || init() != 0) return;
            by_bus = bus;
            mem = m;
        }
    };
    static const Lib lib;
    if (lib.mem == nullptr) return false;
    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return false;
    char bus[32];
    std::snprintf(bus, sizeof bus, "%08X:%02X:%02X.0", (unsigned) prop.pciDomainID, (unsigned) prop.pciBusID,
                  (unsigned) prop.pciDeviceID);
    void* handle = nullptr;
    Memory m{};
    if (lib.by_bus(bus, &handle) != 0 || lib.mem(handle, &m) != 0) return false;
    out = (size_t) m.free;
    return true;
}

}  // namespace
#endif

namespace {
// STRATA_EXPERT_NIBBLE: a write into a sealed cache (see ExpertCache::seal).  Says which call and stops: the slots hold
// 4-bit codes the GGUF bytes would overwrite, or a GGUF expert where a 4-bit one is expected - a wrong reply, not an error.
[[noreturn]] void sealed_write(const char* call, int32_t slot) {
    std::fprintf(stderr, "strata: STRATA_EXPERT_NIBBLE: ExpertCache::%s(slot %d) after the cache was sealed - a GGUF "
                         "blob would be written into the 4-bit expert slots: stopping\n", call, (int) slot);
    std::fflush(stderr);
    std::abort();
}
}  // namespace

// See the header.  NVIDIA's DGX Spark Porting Guide (section 5.5, "Memory reporting on UMA systems") recommends the same:
// not to rely on cudaMemGetInfo alone but to count the memory the OS can reclaim.  Swap is not counted here (unlike
// NVIDIA's reference snippet): an expert cache that pushes the system into swap would be far slower than a smaller one.
size_t device_free_bytes() {
    size_t free_b = 0, total_b = 0;
    if (const cudaError_t e = cudaMemGetInfo(&free_b, &total_b); e != cudaSuccess) {
        // #5708: a failed query is not a full card.  It still counts as nothing free (the callers size from it), but the
        // engine says why, once per kind of error, so a "0.00 GiB free" after it is not read as a measurement; the error
        // is taken off the thread's last-error state so a later check does not report it as its own.
        free_b = 0;
        (void) cudaGetLastError();
        static std::atomic<int> said{(int) cudaSuccess};
        if (said.exchange((int) e) != (int) e)
            // the error's text and number: cudaGetErrorName is not in every build's runtime layer (the Mac's
            // metal_compat, the HIP shim)
            std::fprintf(stderr, "strata: the free graphics memory could not be read (%s, error %d); the expert cache "
                                 "counts it as none free, so it may start with no GPU slots. If this repeats, restart "
                                 "the computer.\n", cudaGetErrorString(e), (int) e);
    }
#if defined(__linux__) && !defined(STRATA_HIP_GFX906)   // (the gfx906 compat layer has no cudaDevAttrIntegrated; that GPU is discrete)
    // per device: a box can mix an integrated GPU (an APU) with a discrete one, and the answer is the CURRENT device's
    static std::atomic<int> uma_cache[64];   // 0 unknown, 1 integrated (unified memory), 2 discrete
    bool unified_memory = false;
    {
        int dev = 0, v = 0;
        if (cudaGetDevice(&dev) == cudaSuccess && dev >= 0 && dev < 64) {
            int c = uma_cache[dev].load(std::memory_order_acquire);
            if (c == 0) {
                c = cudaDeviceGetAttribute(&v, cudaDevAttrIntegrated, dev) == cudaSuccess && v ? 1 : 2;
                uma_cache[dev].store(c, std::memory_order_release);
            }
            unified_memory = c == 1;
        }
    }
    if (unified_memory) {
        if (FILE* m = std::fopen("/proc/meminfo", "r")) {
            char line[256];
            unsigned long long kb = 0;
            while (std::fgets(line, sizeof line, m))
                if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) break;
            std::fclose(m);
            // STRATA_UMA_HEADROOM_GIB: a whole number of GiB, 0..1024; anything else keeps the default 6 (said once)
            static const long gib = [] {
                const char* h = std::getenv("STRATA_UMA_HEADROOM_GIB");
                if (h == nullptr) return 6L;
                char* end = nullptr;
                const long v = std::strtol(h, &end, 10);
                if (end != h && *end == '\0' && v >= 0 && v <= 1024) return v;
                std::fprintf(stderr, "strata: STRATA_UMA_HEADROOM_GIB=%s is not a whole number of GiB (0-1024): using 6\n", h);
                return 6L;
            }();
            const unsigned long long head = (unsigned long long) gib << 30;
            const unsigned long long avail = kb << 10;
            if (avail > head && avail - head > free_b) free_b = (size_t) (avail - head);
        }
    }
#endif
#if defined(STRATA_EC_NVML_FREE)
    static const bool device_wide = [] {
        const char* v = std::getenv("STRATA_VRAM_FREE");
        return v != nullptr && std::strcmp(v, "device") == 0;
    }();
    if (device_wide) {
        size_t nvml_b = 0;
        const bool got = nvml_device_free(nvml_b);
        static std::atomic<bool> said{false};
        if (!said.exchange(true)) {
            if (got)
                std::fprintf(stderr, "strata generate: STRATA_VRAM_FREE=device: free VRAM is read for the whole card "
                                     "(NVML: %lld MiB now; CUDA's figure for this process alone: %lld MiB)\n",
                             (long long) (nvml_b >> 20), (long long) (free_b >> 20));
            else
                std::fprintf(stderr, "strata generate: STRATA_VRAM_FREE=device: NVML did not answer (nvml.dll); free "
                                     "VRAM stays CUDA's figure for this process\n");
        }
        if (got) free_b = std::min(free_b, nvml_b);
    }
#endif
    return free_b;
}

bool read_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                         std::vector<std::pair<int32_t, int32_t>>& ranked, int64_t& slots, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "read_expert_profile: cannot open " + path;
        return false;
    }
    char magic[4] = {0, 0, 0, 0};
    uint32_t hdr[5] = {0, 0, 0, 0, 0};
    if (std::fread(magic, 1, 4, f) != 4 || std::fread(hdr, 4, 5, f) != 5) {
        std::fclose(f);
        err = "read_expert_profile: " + path + " is too short to hold a header";
        return false;
    }
    if (std::memcmp(magic, "STRP", 4) != 0) {
        std::fclose(f);
        err = "read_expert_profile: " + path + " does not start with STRP";
        return false;
    }
    const uint32_t version = hdr[0], nl = hdr[1], ne = hdr[2], want = hdr[3], n_ranked = hdr[4];
    // the version comes first: the fields after it are only known to be these in a version 1 file, so a later
    // format is refused by its number rather than read as if it were this one
    if (version != 1) {
        std::fclose(f);
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "read_expert_profile: %s is a version %u profile but this engine reads version 1 - it was "
                      "written by a different release's tools/make_profile.py or --expert-profile-save",
                      path.c_str(), version);
        err = buf;
        return false;
    }
    if ((int64_t) nl != n_layers || (int64_t) ne != n_expert) {
        std::fclose(f);
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "read_expert_profile: %s is %ux%u but this model is %lldx%lld - it is a profile for a "
                      "different artifact", path.c_str(), nl, ne, (long long) n_layers, (long long) n_expert);
        err = buf;
        return false;
    }
    if (n_ranked > want) {
        std::fclose(f);
        err = "read_expert_profile: the header claims more ranked pairs than slots";
        return false;
    }
    ranked.assign(n_ranked, {0, 0});
    std::vector<uint16_t> raw((size_t) n_ranked * 2);
    if (n_ranked > 0 && std::fread(raw.data(), 2, (size_t) n_ranked * 2, f) != (size_t) n_ranked * 2) {
        std::fclose(f);
        err = "read_expert_profile: the ranked list is truncated";
        return false;
    }
    std::fclose(f);
    for (uint32_t i = 0; i < n_ranked; ++i) {
        const int32_t l = (int32_t) raw[(size_t) i * 2], e = (int32_t) raw[(size_t) i * 2 + 1];
        if (l < 0 || l >= n_layers || e < 0 || e >= n_expert) {
            char buf[256];
            std::snprintf(buf, sizeof buf, "read_expert_profile: pair %u is (layer %d, expert %d), out of range",
                          i, l, e);
            err = buf;
            return false;
        }
        ranked[(size_t) i] = {l, e};
    }
    slots = (int64_t) want;
    return true;
}

std::vector<std::pair<int32_t, int32_t>> rank_learned_profile(int64_t n_layers, int64_t n_expert,
                                                              const std::vector<uint8_t>& resident,
                                                              const std::vector<double>& heat,
                                                              const std::vector<std::pair<int32_t, int32_t>>& prior) {
    const size_t n = (size_t) (n_layers * n_expert);
    std::vector<int64_t> prior_rank(n, INT64_MAX);
    for (size_t r = 0; r < prior.size(); ++r) {
        const auto [l, e] = prior[r];
        if (l >= 0 && l < n_layers && e >= 0 && e < n_expert) {
            int64_t& pr = prior_rank[(size_t) (l * n_expert + e)];
            if (pr == INT64_MAX) pr = (int64_t) r;
        }
    }
    std::vector<int64_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = (int64_t) i;
    auto res = [&](int64_t i) { return (size_t) i < resident.size() && resident[(size_t) i] != 0; };
    auto ht = [&](int64_t i) { return (size_t) i < heat.size() ? heat[(size_t) i] : 0.0; };
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        if (res(a) != res(b)) return res(a);
        if (ht(a) != ht(b)) return ht(a) > ht(b);
        if (prior_rank[(size_t) a] != prior_rank[(size_t) b]) return prior_rank[(size_t) a] < prior_rank[(size_t) b];
        return a < b;
    });
    std::vector<std::pair<int32_t, int32_t>> ranked(n);
    for (size_t r = 0; r < n; ++r)
        ranked[r] = {(int32_t) (order[r] / n_expert), (int32_t) (order[r] % n_expert)};
    return ranked;
}

bool write_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                          const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err) {
    if (n_layers <= 0 || n_expert <= 0 || n_layers > 65535 || n_expert > 65535) {
        err = "write_expert_profile: the model's layout does not fit the format";
        return false;
    }
    std::vector<int32_t> table((size_t) (n_layers * n_expert), -1);
    std::vector<uint16_t> pairs;
    pairs.reserve(ranked.size() * 2);
    for (size_t r = 0; r < ranked.size(); ++r) {
        const auto [l, e] = ranked[r];
        if (l < 0 || l >= n_layers || e < 0 || e >= n_expert) {
            err = "write_expert_profile: a ranked pair is out of range";
            return false;
        }
        table[(size_t) (l * n_expert + e)] = (int32_t) r;
        pairs.push_back((uint16_t) l);
        pairs.push_back((uint16_t) e);
    }
    const uint32_t hdr[5] = {1u, (uint32_t) n_layers, (uint32_t) n_expert, (uint32_t) ranked.size(),
                             (uint32_t) ranked.size()};
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
        err = "write_expert_profile: cannot create " + tmp;
        return false;
    }
    // the format is little-endian (make_profile.py's "<"): so is every machine this engine runs on
    bool ok = std::fwrite("STRP", 1, 4, f) == 4 && std::fwrite(hdr, 4, 5, f) == 5 &&
              (pairs.empty() || std::fwrite(pairs.data(), 2, pairs.size(), f) == pairs.size()) &&
              std::fwrite(table.data(), 4, table.size(), f) == table.size();
    ok = (std::fclose(f) == 0) && ok;
    std::error_code ec;
    if (ok) std::filesystem::rename(tmp, path, ec);   // replaces an existing file (MoveFileEx / rename(2))
    if (!ok || ec) {
        std::filesystem::remove(tmp, ec);
        err = "write_expert_profile: cannot write " + path;
        return false;
    }
    return true;
}

bool read_expert_profile_heat(const std::string& path, int64_t n_layers, int64_t n_expert,
                              std::vector<double>& heat, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "read_expert_profile_heat: cannot open " + path;
        return false;
    }
    char magic[4] = {};
    uint32_t hdr[3] = {};
    if (std::fread(magic, 1, 4, f) != 4 || std::fread(hdr, 4, 3, f) != 3 ||
        std::memcmp(magic, "STRH", 4) != 0 || hdr[0] != 1 ||
        (int64_t) hdr[1] != n_layers || (int64_t) hdr[2] != n_expert ||
        n_layers <= 0 || n_expert <= 0 || n_layers > 65535 || n_expert > 65535 ||
        (size_t) n_layers > std::numeric_limits<size_t>::max() / (size_t) n_expert ||
        (size_t) n_layers * (size_t) n_expert > std::numeric_limits<size_t>::max() / sizeof(double)) {
        std::fclose(f);
        err = "read_expert_profile_heat: invalid header or model dimensions in " + path;
        return false;
    }
    const size_t count = (size_t) n_layers * (size_t) n_expert;
    std::vector<double> values(count);
    if (std::fread(values.data(), sizeof(double), count, f) != count || std::fgetc(f) != EOF || std::ferror(f)) {
        std::fclose(f);
        err = "read_expert_profile_heat: truncated or oversized heat file " + path;
        return false;
    }
    std::fclose(f);
    if (std::any_of(values.begin(), values.end(), [](double v) { return !std::isfinite(v) || v < 0.0; })) {
        err = "read_expert_profile_heat: invalid heat value in " + path;
        return false;
    }
    heat = std::move(values);
    return true;
}

namespace ec_test {   // strata/core/expert_cache_refuse.hpp: tests only, null in the engine
bool (*refuse_segment_call)(const char* call, int64_t seg) = nullptr;
}  // namespace ec_test

bool write_expert_profile_heat(const std::string& path, int64_t n_layers, int64_t n_expert,
                               const std::vector<double>& heat, std::string& err) {
    if (n_layers <= 0 || n_expert <= 0 || n_layers > 65535 || n_expert > 65535 ||
        (size_t) n_layers > std::numeric_limits<size_t>::max() / (size_t) n_expert ||
        (size_t) n_layers * (size_t) n_expert > std::numeric_limits<size_t>::max() / sizeof(double) ||
        heat.size() != (size_t) n_layers * (size_t) n_expert ||
        std::any_of(heat.begin(), heat.end(), [](double v) { return !std::isfinite(v) || v < 0.0; })) {
        err = "write_expert_profile_heat: invalid model dimensions or heat";
        return false;
    }
    const uint32_t hdr[3] = {1u, (uint32_t) n_layers, (uint32_t) n_expert};
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
        err = "write_expert_profile_heat: cannot create " + tmp;
        return false;
    }
    bool ok = std::fwrite("STRH", 1, 4, f) == 4 && std::fwrite(hdr, 4, 3, f) == 3 &&
              std::fwrite(heat.data(), sizeof(double), heat.size(), f) == heat.size();
    ok = (std::fclose(f) == 0) && ok;
    std::error_code ec;
    if (ok) std::filesystem::rename(tmp, path, ec);
    if (!ok || ec) {
        std::filesystem::remove(tmp, ec);
        err = "write_expert_profile_heat: cannot write " + path;
        return false;
    }
    return true;
}

std::vector<int64_t> uniform_expert_cache_quotas(const std::vector<int64_t>& slot_bytes, int64_t n_expert,
                                                uint64_t byte_budget) {
    if (slot_bytes.empty() || n_expert <= 0) return {};
    uint64_t bytes_per_round = 0;
    for (const int64_t bytes : slot_bytes) {
        if (bytes <= 0 || (uint64_t) bytes > std::numeric_limits<uint64_t>::max() - bytes_per_round) return {};
        bytes_per_round += (uint64_t) bytes;
    }
    const int64_t q = std::min<int64_t>(n_expert, (int64_t) std::min<uint64_t>(
                                                           byte_budget / bytes_per_round,
                                                           (uint64_t) std::numeric_limits<int64_t>::max()));
    return q > 0 ? std::vector<int64_t>(slot_bytes.size(), q) : std::vector<int64_t>{};
}

std::vector<int64_t> learned_expert_cache_quotas(const std::vector<double>& heat,
                                                 const std::vector<int64_t>& slot_bytes, int64_t n_expert,
                                                 uint64_t byte_budget) {
    if (slot_bytes.empty() || n_expert <= 0 ||
        (size_t) n_expert > std::numeric_limits<size_t>::max() / slot_bytes.size() ||
        heat.size() != slot_bytes.size() * (size_t) n_expert ||
        std::any_of(slot_bytes.begin(), slot_bytes.end(), [](int64_t b) { return b <= 0; }) ||
        std::any_of(heat.begin(), heat.end(), [](double v) { return !std::isfinite(v) || v < 0.0; }))
        return {};
    const std::vector<int64_t> uniform = uniform_expert_cache_quotas(slot_bytes, n_expert, byte_budget);
    if (uniform.empty() || std::none_of(heat.begin(), heat.end(), [](double v) { return v > 0.0; })) return uniform;
    int64_t slot_limit = 0;
    for (const int64_t quota : uniform) {
        if (quota > std::numeric_limits<int64_t>::max() - slot_limit) return {};
        slot_limit += quota;
    }

    std::vector<std::vector<double>> ranked(slot_bytes.size());
    uint64_t used = 0;
    for (size_t layer = 0; layer < slot_bytes.size(); ++layer) {
        const uint64_t cost = (uint64_t) slot_bytes[layer];
        if (cost > std::numeric_limits<uint64_t>::max() - used) return {};
        used += cost;   // floor each layer at one slot
        ranked[layer].assign(heat.begin() + (ptrdiff_t) (layer * (size_t) n_expert),
                             heat.begin() + (ptrdiff_t) ((layer + 1) * (size_t) n_expert));
        std::sort(ranked[layer].begin(), ranked[layer].end(), std::greater<double>());
    }
    if (used > byte_budget) return {};
    std::vector<int64_t> quotas(slot_bytes.size(), 1);
    int64_t assigned = (int64_t) quotas.size();
    for (;;) {
        if (assigned >= slot_limit) break;
        size_t best = slot_bytes.size();
        long double best_value = 0.0L;
        for (size_t layer = 0; layer < slot_bytes.size(); ++layer) {
            const int64_t next = quotas[layer];
            if (next >= n_expert || (uint64_t) slot_bytes[layer] > byte_budget - used) continue;
            const long double value = (long double) ranked[layer][(size_t) next] / (long double) slot_bytes[layer];
            if (value > best_value ||
                (value == best_value && best != slot_bytes.size() && quotas[layer] < quotas[best])) {
                best = layer;
                best_value = value;
            }
        }
        if (best == slot_bytes.size()) break;
        ++quotas[best];
        ++assigned;
        used += (uint64_t) slot_bytes[best];
    }
    return quotas;
}

const char* expert_heat_provenance_name(ExpertHeatProvenance p) {
    switch (p) {
        case ExpertHeatProvenance::None: return "none";
        case ExpertHeatProvenance::Runtime: return "runtime";
        case ExpertHeatProvenance::LegacyUnmarked: return "legacy-unmarked";
        case ExpertHeatProvenance::ProfileRankPrior: return "profile-rank prior";
        case ExpertHeatProvenance::Malformed: return "malformed";
    }
    return "unknown";
}

std::string expert_profile_heat_meta_path(const std::string& heat_path) {
    return heat_path.empty() ? std::string{} : heat_path + ".meta";
}

namespace {

constexpr uint64_t kHeatFnvOffset = 14695981039346656037ull;
constexpr uint64_t kHeatFnvPrime  = 1099511628211ull;

bool valid_heat_dimensions(int64_t n_layers, int64_t n_expert) {
    return n_layers > 0 && n_expert > 0 && n_layers <= 65535 && n_expert <= 65535 &&
           (size_t) n_layers <= std::numeric_limits<size_t>::max() / (size_t) n_expert &&
           (size_t) n_layers * (size_t) n_expert <= std::numeric_limits<size_t>::max() / sizeof(double);
}

bool valid_heat_values(const std::vector<double>& heat, int64_t n_layers, int64_t n_expert) {
    return valid_heat_dimensions(n_layers, n_expert) &&
           heat.size() == (size_t) n_layers * (size_t) n_expert &&
           std::none_of(heat.begin(), heat.end(), [](double v) { return !std::isfinite(v) || v < 0.0; });
}

uint64_t heat_payload_fnv1a64(const std::vector<double>& heat) {
    uint64_t h = kHeatFnvOffset;
    const auto* bytes = reinterpret_cast<const uint8_t*>(heat.data());
    const size_t nbytes = heat.size() * sizeof(double);
    for (size_t i = 0; i < nbytes; ++i) {
        h ^= (uint64_t) bytes[i];
        h *= kHeatFnvPrime;
    }
    return h;
}

std::string format_hex64(uint64_t v) {
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", (unsigned long long) v);
    return std::string(buf, 16);
}

std::string sanitize_meta_value(std::string s) {
    for (char& c : s) {
        if (c == '\r' || c == '\n' || c == '\0') c = ' ';
    }
    return s;
}

bool parse_strict_i64(const std::string& s, int64_t& out) {
    if (s.empty()) return false;
    size_t idx = 0;
    if (s[0] == '+' || s[0] == '-') {
        if (s[0] == '+' || s.size() == 1) return false;
    }
    char* end = nullptr;
    errno = 0;
    const long long val = std::strtoll(s.c_str(), &end, 10);
    if (errno != 0 || end == s.c_str() || *end != '\0') return false;
    (void) idx;
    out = (int64_t) val;
    return true;
}

bool parse_strict_hex64(const std::string& s, uint64_t& out) {
    if (s.size() != 16) return false;
    uint64_t acc = 0;
    for (unsigned char ch : s) {
        acc <<= 4;
        if (ch >= '0' && ch <= '9') acc |= (uint64_t) (ch - '0');
        else if (ch >= 'a' && ch <= 'f') acc |= (uint64_t) (10 + (ch - 'a'));
        else if (ch >= 'A' && ch <= 'F') acc |= (uint64_t) (10 + (ch - 'A'));
        else return false;
    }
    out = acc;
    return true;
}

bool stage_heat_tmp(const std::string& tmp_path, int64_t n_layers, int64_t n_expert,
                    const std::vector<double>& heat, std::string& err) {
    const uint32_t hdr[3] = {1u, (uint32_t) n_layers, (uint32_t) n_expert};
    std::FILE* f = std::fopen(tmp_path.c_str(), "wb");
    if (f == nullptr) {
        err = "write_expert_profile_heat_with_provenance: cannot create " + tmp_path;
        return false;
    }
    bool ok = std::fwrite("STRH", 1, 4, f) == 4 && std::fwrite(hdr, 4, 3, f) == 3 &&
              (heat.empty() || std::fwrite(heat.data(), sizeof(double), heat.size(), f) == heat.size());
    ok = (std::fclose(f) == 0) && ok;
    if (!ok) {
        std::error_code ec;
        std::filesystem::remove(tmp_path, ec);
        err = "write_expert_profile_heat_with_provenance: cannot write " + tmp_path;
        return false;
    }
    return true;
}

bool stage_meta_tmp(const std::string& tmp_path, int64_t n_layers, int64_t n_expert,
                    ExpertHeatProvenance provenance, const std::string& source_profile,
                    const std::string& source_kind, int64_t ranked_pairs, uint64_t heat_hash,
                    std::string& err) {
    const char* prov_str = provenance == ExpertHeatProvenance::Runtime ? "runtime" : "profile-rank-prior";
    const std::string body =
        "strata_heat_meta=1\n"
        "provenance=" + std::string(prov_str) + "\n"
        "source_kind=" + sanitize_meta_value(source_kind) + "\n"
        "source_profile=" + sanitize_meta_value(source_profile) + "\n"
        "n_layers=" + std::to_string((long long) n_layers) + "\n"
        "n_expert=" + std::to_string((long long) n_expert) + "\n"
        "ranked_pairs=" + std::to_string((long long) ranked_pairs) + "\n"
        "heat_fnv1a64=" + format_hex64(heat_hash) + "\n";

    std::FILE* f = std::fopen(tmp_path.c_str(), "wb");
    if (f == nullptr) {
        err = "write_expert_profile_heat_with_provenance: cannot create " + tmp_path;
        return false;
    }
    bool ok = std::fwrite(body.data(), 1, body.size(), f) == body.size();
    ok = (std::fclose(f) == 0) && ok;
    if (!ok) {
        std::error_code ec;
        std::filesystem::remove(tmp_path, ec);
        err = "write_expert_profile_heat_with_provenance: cannot write " + tmp_path;
        return false;
    }
    return true;
}

bool read_heat_meta_file(const std::string& meta_path, int64_t n_layers, int64_t n_expert,
                         ExpertHeatProvenance& provenance, std::string& source_profile,
                         uint64_t& recorded_hash, std::string& err) {
    std::error_code ec;
    if (std::filesystem::is_directory(meta_path, ec) || ec) {
        err = "read_expert_profile_heat_with_provenance: invalid metadata path " + meta_path;
        return false;
    }
    const uintmax_t fsize = std::filesystem::file_size(meta_path, ec);
    if (ec || fsize > 8192) {
        err = "read_expert_profile_heat_with_provenance: unreadable or oversized metadata file " + meta_path;
        return false;
    }
    std::FILE* f = std::fopen(meta_path.c_str(), "rb");
    if (f == nullptr) {
        err = "read_expert_profile_heat_with_provenance: cannot open metadata file " + meta_path;
        return false;
    }
    std::string content((size_t) fsize, '\0');
    const size_t nread = content.empty() ? 0 : std::fread(content.data(), 1, content.size(), f);
    const int extra = std::fgetc(f);
    const bool io_err = std::ferror(f) != 0;
    std::fclose(f);
    if (nread != content.size() || extra != EOF || io_err) {
        err = "read_expert_profile_heat_with_provenance: truncated metadata file " + meta_path;
        return false;
    }

    bool have_ver = false;
    bool have_prov = false;
    bool have_nl = false;
    bool have_ne = false;
    bool have_hash = false;
    ExpertHeatProvenance parsed_prov = ExpertHeatProvenance::None;
    std::string parsed_src;
    uint64_t parsed_hash = 0;

    size_t pos = 0;
    while (pos < content.size()) {
        const size_t nl = content.find('\n', pos);
        std::string line = (nl == std::string::npos) ? content.substr(pos) : content.substr(pos, nl - pos);
        pos = (nl == std::string::npos) ? content.size() : nl + 1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        const size_t eq = line.find('=');
        if (eq == std::string::npos || eq == 0) {
            err = "read_expert_profile_heat_with_provenance: malformed metadata line in " + meta_path;
            return false;
        }
        const std::string key = line.substr(0, eq);
        const std::string val = line.substr(eq + 1);
        if (key == "strata_heat_meta") {
            if (have_ver || val != "1") {
                err = "read_expert_profile_heat_with_provenance: unsupported metadata version in " + meta_path;
                return false;
            }
            have_ver = true;
        } else if (key == "provenance") {
            if (have_prov) {
                err = "read_expert_profile_heat_with_provenance: duplicate provenance in " + meta_path;
                return false;
            }
            if (val == "runtime") parsed_prov = ExpertHeatProvenance::Runtime;
            else if (val == "profile-rank-prior") parsed_prov = ExpertHeatProvenance::ProfileRankPrior;
            else {
                err = "read_expert_profile_heat_with_provenance: unknown provenance '" + val + "' in " + meta_path;
                return false;
            }
            have_prov = true;
        } else if (key == "n_layers") {
            int64_t v = 0;
            if (have_nl || !parse_strict_i64(val, v) || v != n_layers) {
                err = "read_expert_profile_heat_with_provenance: metadata n_layers mismatch in " + meta_path;
                return false;
            }
            have_nl = true;
        } else if (key == "n_expert") {
            int64_t v = 0;
            if (have_ne || !parse_strict_i64(val, v) || v != n_expert) {
                err = "read_expert_profile_heat_with_provenance: metadata n_expert mismatch in " + meta_path;
                return false;
            }
            have_ne = true;
        } else if (key == "heat_fnv1a64") {
            if (have_hash || !parse_strict_hex64(val, parsed_hash)) {
                err = "read_expert_profile_heat_with_provenance: invalid heat checksum in " + meta_path;
                return false;
            }
            have_hash = true;
        } else if (key == "source_profile") {
            parsed_src = val;
        } else if (key == "source_kind" || key == "ranked_pairs") {
            if (key == "ranked_pairs") {
                int64_t rp = -1;
                if (!parse_strict_i64(val, rp) || rp < 0) {
                    err = "read_expert_profile_heat_with_provenance: invalid ranked_pairs in " + meta_path;
                    return false;
                }
            }
        }
    }

    if (!have_ver || !have_prov || !have_nl || !have_ne || !have_hash) {
        err = "read_expert_profile_heat_with_provenance: incomplete metadata in " + meta_path;
        return false;
    }
    provenance = parsed_prov;
    source_profile = std::move(parsed_src);
    recorded_hash = parsed_hash;
    return true;
}

bool same_profile_target(const std::string& a, const std::string& b) {
    if (a.empty() || b.empty()) return false;
    const std::filesystem::path pa = std::filesystem::path(a).lexically_normal();
    const std::filesystem::path pb = std::filesystem::path(b).lexically_normal();
    if (pa == pb) return true;
    std::error_code ec;
    const std::filesystem::path ca = std::filesystem::weakly_canonical(pa, ec);
    if (!ec) {
        std::error_code ec2;
        const std::filesystem::path cb = std::filesystem::weakly_canonical(pb, ec2);
        if (!ec2 && ca == cb) return true;
    }
    const std::filesystem::path ha(a + ".heat");
    const std::filesystem::path hb(b + ".heat");
    if (std::filesystem::exists(ha, ec) && !ec && std::filesystem::exists(hb, ec) && !ec) {
        const bool eq = std::filesystem::equivalent(ha, hb, ec);
        if (!ec && eq) return true;
    }
    return false;
}

ExpertHeatResolution inspect_heat_candidate(const std::string& profile_path, int64_t n_layers, int64_t n_expert) {
    ExpertHeatResolution res;
    if (profile_path.empty()) return res;
    res.path = profile_path + ".heat";
    std::error_code ec;
    const bool exists = std::filesystem::exists(res.path, ec);
    if (ec) {
        res.found = true;
        res.valid = false;
        res.provenance = ExpertHeatProvenance::Malformed;
        res.detail = "cannot check expert heat file " + res.path + ": " + ec.message();
        return res;
    }
    if (!exists) return res;

    res.found = true;
    std::string err;
    if (!read_expert_profile_heat_with_provenance(res.path, n_layers, n_expert, res.heat, res.provenance,
                                                  res.source_profile, err)) {
        res.valid = false;
        res.provenance = ExpertHeatProvenance::Malformed;
        res.detail = err;
        res.heat.clear();
        return res;
    }
    res.valid = true;
    return res;
}

bool is_measured_provenance(ExpertHeatProvenance p) {
    return p == ExpertHeatProvenance::Runtime || p == ExpertHeatProvenance::LegacyUnmarked;
}

}  // namespace

bool profile_rank_prior_heat(int64_t n_layers, int64_t n_expert,
                             const std::vector<std::pair<int32_t, int32_t>>& ranked,
                             std::vector<double>& heat, std::string& err) {
    if (!valid_heat_dimensions(n_layers, n_expert)) {
        err = "profile_rank_prior_heat: invalid model dimensions";
        return false;
    }
    if (ranked.empty()) {
        err = "profile_rank_prior_heat: ranked profile list is empty";
        return false;
    }
    const size_t total = (size_t) n_layers * (size_t) n_expert;
    std::vector<double> weights(total, 0.0);
    const size_t n_ranked = ranked.size();
    for (size_t r = 0; r < n_ranked; ++r) {
        const auto [l, e] = ranked[r];
        if (l < 0 || l >= n_layers || e < 0 || e >= n_expert) {
            err = "profile_rank_prior_heat: ranked pair is out of range for model dimensions";
            return false;
        }
        const size_t idx = (size_t) l * (size_t) n_expert + (size_t) e;
        if (weights[idx] == 0.0) weights[idx] = (double) (n_ranked - r);
    }
    heat = std::move(weights);
    return true;
}

bool read_expert_profile_heat_with_provenance(const std::string& heat_path, int64_t n_layers, int64_t n_expert,
                                              std::vector<double>& heat, ExpertHeatProvenance& provenance,
                                              std::string& source_profile, std::string& err) {
    heat.clear();
    source_profile.clear();
    if (heat_path.empty()) {
        provenance = ExpertHeatProvenance::None;
        err = "read_expert_profile_heat_with_provenance: empty path";
        return false;
    }
    std::error_code ec;
    const bool exists = std::filesystem::exists(heat_path, ec);
    if (ec) {
        provenance = ExpertHeatProvenance::Malformed;
        err = "read_expert_profile_heat_with_provenance: cannot stat " + heat_path + ": " + ec.message();
        return false;
    }
    if (!exists) {
        provenance = ExpertHeatProvenance::None;
        err = "read_expert_profile_heat_with_provenance: missing " + heat_path;
        return false;
    }
    if (std::filesystem::is_directory(heat_path, ec) || ec) {
        provenance = ExpertHeatProvenance::Malformed;
        err = "read_expert_profile_heat_with_provenance: invalid heat file path " + heat_path;
        return false;
    }

    std::vector<double> values;
    if (!read_expert_profile_heat(heat_path, n_layers, n_expert, values, err)) {
        provenance = ExpertHeatProvenance::Malformed;
        return false;
    }

    const std::string meta_path = expert_profile_heat_meta_path(heat_path);
    const bool has_meta = std::filesystem::exists(meta_path, ec);
    if (ec) {
        provenance = ExpertHeatProvenance::Malformed;
        err = "read_expert_profile_heat_with_provenance: cannot stat metadata " + meta_path + ": " + ec.message();
        return false;
    }
    if (!has_meta) {
        provenance = ExpertHeatProvenance::LegacyUnmarked;
        heat = std::move(values);
        return true;
    }

    ExpertHeatProvenance meta_prov = ExpertHeatProvenance::None;
    std::string meta_src;
    uint64_t meta_hash = 0;
    if (!read_heat_meta_file(meta_path, n_layers, n_expert, meta_prov, meta_src, meta_hash, err)) {
        provenance = ExpertHeatProvenance::Malformed;
        return false;
    }
    const uint64_t actual_hash = heat_payload_fnv1a64(values);
    if (meta_prov == ExpertHeatProvenance::Runtime && meta_hash != actual_hash) {
        provenance = ExpertHeatProvenance::Malformed;
        err = "read_expert_profile_heat_with_provenance: runtime heat checksum mismatch with " + meta_path;
        return false;
    }
    // Note: if meta_prov == ProfileRankPrior and meta_hash != actual_hash (e.g. a crash occurred after
    // renaming runtime .heat before renaming .meta), provenance intentionally stays ProfileRankPrior so a
    // failed/interrupted write never silently upgrades a prior to Runtime.
    provenance = meta_prov;
    source_profile = std::move(meta_src);
    heat = std::move(values);
    return true;
}

bool write_expert_profile_heat_with_provenance(const std::string& heat_path, int64_t n_layers, int64_t n_expert,
                                               const std::vector<double>& heat, ExpertHeatProvenance provenance,
                                               const std::string& source_profile, const std::string& source_kind,
                                               int64_t ranked_pairs, std::string& err) {
    if (heat_path.empty() || !valid_heat_values(heat, n_layers, n_expert) || ranked_pairs < 0) {
        err = "write_expert_profile_heat_with_provenance: invalid arguments, dimensions, or heat values";
        return false;
    }
    if (provenance != ExpertHeatProvenance::Runtime && provenance != ExpertHeatProvenance::ProfileRankPrior) {
        err = "write_expert_profile_heat_with_provenance: provenance must be Runtime or ProfileRankPrior";
        return false;
    }

    const std::string meta_path = expert_profile_heat_meta_path(heat_path);
    const std::string heat_tmp = heat_path + ".tmp";
    const std::string meta_tmp = meta_path + ".tmp";
    const uint64_t heat_hash = heat_payload_fnv1a64(heat);

    std::error_code ec;
    const bool had_heat = std::filesystem::exists(heat_path, ec);
    if (ec) {
        err = "write_expert_profile_heat_with_provenance: cannot inspect " + heat_path;
        return false;
    }
    const bool had_meta = std::filesystem::exists(meta_path, ec);
    if (ec) {
        err = "write_expert_profile_heat_with_provenance: cannot inspect " + meta_path;
        return false;
    }

    auto cleanup_tmps = [&] {
        std::error_code rm_ec;
        std::filesystem::remove(heat_tmp, rm_ec);
        std::filesystem::remove(meta_tmp, rm_ec);
    };

    if (provenance == ExpertHeatProvenance::ProfileRankPrior) {
        if (!stage_meta_tmp(meta_tmp, n_layers, n_expert, provenance, source_profile, source_kind, ranked_pairs,
                            heat_hash, err) ||
            !stage_heat_tmp(heat_tmp, n_layers, n_expert, heat, err)) {
            cleanup_tmps();
            return false;
        }
        std::filesystem::rename(meta_tmp, meta_path, ec);
        if (ec) {
            cleanup_tmps();
            err = "write_expert_profile_heat_with_provenance: cannot commit metadata " + meta_path + ": " + ec.message();
            return false;
        }
        std::filesystem::rename(heat_tmp, heat_path, ec);
        if (ec) {
            cleanup_tmps();
            if (!had_heat && !had_meta) {
                std::error_code rm_ec;
                std::filesystem::remove(meta_path, rm_ec);
            }
            err = "write_expert_profile_heat_with_provenance: cannot commit heat " + heat_path + ": " + ec.message();
            return false;
        }
        return true;
    }

    // Runtime write: stage BOTH temp files first so staging failures leave existing heat & metadata intact,
    // then commit .heat before .meta so a failure or crash between renames never upgrades a prior to Runtime.
    if (!stage_heat_tmp(heat_tmp, n_layers, n_expert, heat, err) ||
        !stage_meta_tmp(meta_tmp, n_layers, n_expert, provenance, source_profile, source_kind, ranked_pairs,
                        heat_hash, err)) {
        cleanup_tmps();
        return false;
    }
    std::filesystem::rename(heat_tmp, heat_path, ec);
    if (ec) {
        cleanup_tmps();
        err = "write_expert_profile_heat_with_provenance: cannot commit runtime heat " + heat_path + ": " +
              ec.message();
        return false;
    }
    std::filesystem::rename(meta_tmp, meta_path, ec);
    if (ec) {
        cleanup_tmps();
        err = "write_expert_profile_heat_with_provenance: cannot commit runtime metadata " + meta_path + ": " +
              ec.message();
        return false;
    }
    return true;
}

ExpertHeatResolution resolve_expert_profile_heat(const std::string& save_profile_path,
                                                 const std::string& input_profile_path,
                                                 int64_t n_layers, int64_t n_expert) {
    if (!valid_heat_dimensions(n_layers, n_expert)) {
        ExpertHeatResolution bad;
        bad.found = false;
        bad.valid = false;
        bad.detail = "invalid model dimensions";
        return bad;
    }
    ExpertHeatResolution save = inspect_heat_candidate(save_profile_path, n_layers, n_expert);
    ExpertHeatResolution input;
    if (!same_profile_target(save_profile_path, input_profile_path)) {
        input = inspect_heat_candidate(input_profile_path, n_layers, n_expert);
    }

    if (save.found && !save.valid) return save;
    if (save.valid && is_measured_provenance(save.provenance)) return save;
    if (input.found && !input.valid) return input;
    if (input.valid && is_measured_provenance(input.provenance)) return input;
    if (save.valid) return save;
    if (input.found) return input;
    return {};
}

ExpertHeatBootstrapResult bootstrap_expert_profile_heat(
    const std::string& save_profile_path, const std::string& input_profile_path,
    const std::vector<std::pair<int32_t, int32_t>>& loaded_input_profile,
    int64_t n_layers, int64_t n_expert) {
    ExpertHeatBootstrapResult out;
    if (save_profile_path.empty()) {
        out.action = ExpertHeatBootstrapResult::Action::SkippedNoProfile;
        out.detail = "no --expert-profile-save path configured";
        return out;
    }
    out.heat_path = save_profile_path + ".heat";
    if (!valid_heat_dimensions(n_layers, n_expert)) {
        out.action = ExpertHeatBootstrapResult::Action::Failed;
        out.detail = "invalid model dimensions";
        return out;
    }

    const ExpertHeatResolution save = inspect_heat_candidate(save_profile_path, n_layers, n_expert);
    ExpertHeatResolution input;
    if (!same_profile_target(save_profile_path, input_profile_path)) {
        input = inspect_heat_candidate(input_profile_path, n_layers, n_expert);
    }

    if (save.found && !save.valid) {
        out.action = ExpertHeatBootstrapResult::Action::MalformedLeftUntouched;
        out.heat_path = save.path;
        out.provenance = ExpertHeatProvenance::Malformed;
        out.detail = save.detail;
        return out;
    }
    if (save.valid && is_measured_provenance(save.provenance)) {
        out.action = ExpertHeatBootstrapResult::Action::RetainedValid;
        out.heat_path = save.path;
        out.provenance = save.provenance;
        out.source_profile = save.source_profile;
        return out;
    }
    if (input.found && !input.valid) {
        out.action = ExpertHeatBootstrapResult::Action::MalformedLeftUntouched;
        out.heat_path = input.path;
        out.provenance = ExpertHeatProvenance::Malformed;
        out.detail = input.detail;
        return out;
    }
    if (input.valid && is_measured_provenance(input.provenance)) {
        out.action = ExpertHeatBootstrapResult::Action::RetainedValid;
        out.heat_path = input.path;
        out.provenance = input.provenance;
        out.source_profile = input.source_profile;
        return out;
    }
    if (save.valid) {
        out.action = ExpertHeatBootstrapResult::Action::RetainedValid;
        out.heat_path = save.path;
        out.provenance = save.provenance;
        out.source_profile = save.source_profile;
        return out;
    }

    std::vector<std::pair<int32_t, int32_t>> chosen_ranked;
    std::string chosen_source_profile;
    std::string chosen_source_kind;
    std::string fallback_note;

    {
        std::vector<std::pair<int32_t, int32_t>> saved_ranked;
        int64_t saved_slots = 0;
        std::string read_err;
        if (read_expert_profile(save_profile_path, n_layers, n_expert, saved_ranked, saved_slots, read_err) &&
            !saved_ranked.empty()) {
            chosen_ranked = std::move(saved_ranked);
            chosen_source_profile = save_profile_path;
            chosen_source_kind = "saved-profile";
        } else {
            std::error_code ec;
            if (std::filesystem::exists(save_profile_path, ec)) {
                fallback_note = read_err.empty() ? "saved profile has empty ranking" : read_err;
            }
        }
    }

    if (chosen_ranked.empty()) {
        if (!loaded_input_profile.empty()) {
            chosen_ranked = loaded_input_profile;
            chosen_source_profile = input_profile_path.empty() ? "<input-profile>" : input_profile_path;
            chosen_source_kind = "input-profile";
        } else if (!input_profile_path.empty()) {
            std::vector<std::pair<int32_t, int32_t>> input_ranked;
            int64_t input_slots = 0;
            std::string read_err;
            if (read_expert_profile(input_profile_path, n_layers, n_expert, input_ranked, input_slots, read_err) &&
                !input_ranked.empty()) {
                chosen_ranked = std::move(input_ranked);
                chosen_source_profile = input_profile_path;
                chosen_source_kind = "input-profile";
            }
        }
    }

    if (chosen_ranked.empty()) {
        out.action = ExpertHeatBootstrapResult::Action::SkippedNoProfile;
        out.detail = fallback_note.empty() ? "no valid saved or input expert profile available for heat prior"
                                           : fallback_note;
        return out;
    }

    std::vector<double> prior_heat;
    std::string prior_err;
    if (!profile_rank_prior_heat(n_layers, n_expert, chosen_ranked, prior_heat, prior_err)) {
        out.action = ExpertHeatBootstrapResult::Action::Failed;
        out.detail = prior_err;
        return out;
    }

    std::string write_err;
    if (!write_expert_profile_heat_with_provenance(out.heat_path, n_layers, n_expert, prior_heat,
                                                   ExpertHeatProvenance::ProfileRankPrior,
                                                   chosen_source_profile, chosen_source_kind,
                                                   (int64_t) chosen_ranked.size(), write_err)) {
        out.action = ExpertHeatBootstrapResult::Action::Failed;
        out.detail = write_err;
        return out;
    }

    out.action = ExpertHeatBootstrapResult::Action::DerivedPrior;
    out.provenance = ExpertHeatProvenance::ProfileRankPrior;
    out.source_profile = std::move(chosen_source_profile);
    out.source_kind = std::move(chosen_source_kind);
    out.ranked_pairs = chosen_ranked.size();
    out.detail = std::move(fallback_note);
    return out;
}

bool save_runtime_expert_profile_heat(const std::string& save_profile_path, int64_t n_layers, int64_t n_expert,
                                      const std::vector<double>& heat, int64_t ranked_pairs,
                                      bool& skipped_zero_heat, std::string& err) {
    skipped_zero_heat = false;
    if (save_profile_path.empty() || !valid_heat_values(heat, n_layers, n_expert)) {
        err = "save_runtime_expert_profile_heat: invalid path, dimensions, or heat values";
        return false;
    }
    const bool has_observed_routing = std::any_of(heat.begin(), heat.end(), [](double v) { return v > 0.0; });
    if (!has_observed_routing) {
        skipped_zero_heat = true;
        return true;
    }
    const std::string heat_path = save_profile_path + ".heat";
    return write_expert_profile_heat_with_provenance(heat_path, n_layers, n_expert, heat,
                                                     ExpertHeatProvenance::Runtime, save_profile_path,
                                                     "runtime-routing", ranked_pairs, err);
}

ExpertCache::~ExpertCache() { close(); }

#if !defined(STRATA_EC_NO_VMM)
static bool segmented_supported() { return vmm_available() && vmm_granularity() > 0; }
#endif

bool ExpertCache::open_segmented(uint64_t want, std::string& err) {
#if defined(STRATA_EC_NO_VMM)
    (void) want;
    err = "ExpertCache: --vram-elastic needs the driver's virtual memory management, which this build does not offer";
    return false;
#else
    const uint64_t gran = vmm_granularity();
    if (!vmm_available() || gran == 0) {
        err = "ExpertCache: --vram-elastic needs the driver's virtual memory management, which this GPU or driver "
              "does not offer";
        return false;
    }
    seg_ = (int64_t) (((uint64_t) seg_req_ + gran - 1) / gran * gran);
    auto range = std::make_unique<VmmRange>();
    if (!range->reserve(want)) {
        err = "ExpertCache: cannot reserve the address range of the segmented expert cache";
        seg_ = 0;
        return false;
    }
    if (!range->map_range(0, range->chunks(), [] { return (VmmChunk) 0; })) {
        err = "ExpertCache: cudaMalloc failed: a physical segment of the segmented cache could not be allocated";
        range.reset();
        seg_ = 0;
        return false;
    }
    base_ = range->base();
    reserved_ = (uint64_t) range->chunks() * gran;
    segmented_vmm_ = std::move(range);
    return true;
#endif
}

void ExpertCache::release_segmented() {
#if !defined(STRATA_EC_NO_VMM)
    if (base_ != nullptr) cudaDeviceSynchronize();
    for (VmmChunk h : unmapped_vmm_chunks_)
        if (h != 0) vmm_chunk_free(h);
    unmapped_vmm_chunks_.clear();
#endif
    segmented_vmm_.reset();
    reserved_ = 0;
    seg_ = 0;
    base_ = nullptr;
}

int64_t ExpertCache::mapped_bytes() const {
    if (segmented_vmm_)
        return segmented_vmm_->mapped_count() * (int64_t) vmm_granularity();
    return base_ != nullptr ? full_bytes() : 0;
}

int64_t ExpertCache::slots_within(int64_t bytes) const {
    if (bytes >= full_bytes()) return slots_;
    if (bytes <= 0) return 0;
    if (off_.empty()) return blob_ > 0 ? bytes / blob_ : 0;
    // off_[i + 1] is slot i's end: count the slots whose end is at most `bytes`
    return (int64_t) (std::upper_bound(off_.begin() + 1, off_.end(), (uint64_t) bytes) - (off_.begin() + 1));
}

bool ExpertCache::shrink(int64_t keep_bytes, std::string& err) {
    if (!segmented_vmm_) {
        err = "the expert cache is not segmented (the engine needs --vram-elastic)";
        return false;
    }
#if defined(STRATA_EC_NO_VMM)
    (void) keep_bytes;
    return false;
#else
    const uint64_t gran = vmm_granularity();
    const int64_t total_chunks = segmented_vmm_->chunks();
    int64_t keep_chunks = 0;
    if (keep_bytes > 0) {
        const uint64_t segment_count = ((uint64_t) keep_bytes + (uint64_t) seg_ - 1) / (uint64_t) seg_;
        const uint64_t total_segments = ((uint64_t) reserved_ + (uint64_t) seg_ - 1) / (uint64_t) seg_;
        keep_chunks = segment_count >= total_segments
                          ? total_chunks
                          : (int64_t) (segment_count * (uint64_t) seg_ / gran);
    }
    const int64_t mapped_chunks = segmented_vmm_->mapped_count();
    if (keep_chunks >= mapped_chunks) return true;
    if (cudaDeviceSynchronize() != cudaSuccess) {
        err = std::string("the device failed before the cache shrank: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    auto refused = [](const char* call, int64_t seg) {
        return ec_test::refuse_segment_call != nullptr && ec_test::refuse_segment_call(call, seg);
    };
    for (int64_t i = mapped_chunks - 1; i >= keep_chunks; --i) {
        if (refused("unmap", i)) {
            err = "the driver refused to release an expert-cache segment";
            live_slots_ = slots_within(mapped_bytes());
            return false;
        }
        const VmmChunk h = segmented_vmm_->unmap(i);
        if (h == 0) {
            err = "the driver refused to release an expert-cache segment";
            live_slots_ = slots_within(mapped_bytes());   // segment i is still mapped: [0, i] stay
            return false;
        }
        if (refused("release", i)) {
            unmapped_vmm_chunks_.push_back(h);
            err = "the driver refused to release an expert-cache segment";
            live_slots_ = slots_within(mapped_bytes());
            return false;
        }
        vmm_chunk_free(h);
    }
    live_slots_ = slots_within(mapped_bytes());
    return true;
#endif
}

bool ExpertCache::grow(int64_t want_bytes, std::string& err) {
    if (!segmented_vmm_) {
        err = "the expert cache is not segmented (the engine needs --vram-elastic)";
        return false;
    }
#if defined(STRATA_EC_NO_VMM)
    (void) want_bytes;
    return false;
#else
    const uint64_t gran = vmm_granularity();
    const uint64_t capped = want_bytes > 0 ? std::min<uint64_t>((uint64_t) want_bytes, reserved_) : 0;
    const int64_t target_chunks = capped >= reserved_
                                      ? segmented_vmm_->chunks()
                                      : (int64_t) ((capped / (uint64_t) seg_) * (uint64_t) seg_ / gran);
    const int64_t mapped_chunks = segmented_vmm_->mapped_count();
    if (target_chunks > mapped_chunks &&
        !segmented_vmm_->map_range(mapped_chunks, target_chunks, [this] {
            if (unmapped_vmm_chunks_.empty()) return (VmmChunk) 0;
            const VmmChunk h = unmapped_vmm_chunks_.back();
            unmapped_vmm_chunks_.pop_back();
            return h;
        })) {
        (void) cudaGetLastError();
        err = "the driver has no VRAM for another expert-cache segment";
        live_slots_ = slots_within(mapped_bytes());
        return false;
    }
    live_slots_ = slots_within(mapped_bytes());
    return true;
#endif
}

#if defined(STRATA_USE_HIP)
bool ExpertCache::ensure_blocking_staging(std::size_t bytes, std::string& err) {
    if (bytes <= blocking_staging_bytes_) return true;
    void* next = nullptr;
    const cudaError_t status = cudaHostAlloc(&next, bytes, cudaHostAllocDefault);
    if (status != cudaSuccess) {
        err = std::string("ExpertCache: HIP blocking staging allocation: ") + cudaGetErrorString(status);
        return false;
    }
    if (blocking_staging_) (void) cudaFreeHost(blocking_staging_);
    blocking_staging_ = static_cast<uint8_t*>(next);
    blocking_staging_bytes_ = bytes;
    return true;
}
#endif

namespace {
bool g_cache_vmm = false;   // set_vmm
}  // namespace

void ExpertCache::set_vmm(bool enabled) { g_cache_vmm = enabled; }

bool ExpertCache::open(int64_t n_slots, int64_t n_layers, int64_t n_expert, int64_t blob_bytes,
                       std::string& err) {
    close();
    if (n_slots <= 0) {
        err = "ExpertCache: n_slots must be positive";
        return false;
    }
    if (n_layers <= 0 || n_expert <= 0 || blob_bytes <= 0) {
        err = "ExpertCache: n_layers, n_expert and blob_bytes must all be positive";
        return false;
    }

    const uint64_t want = (uint64_t) n_slots * (uint64_t) blob_bytes;

    // ---- **THE ALLOCATION IS CHECKED AGAINST THE CARD, NOT AGAINST THE REQUEST.**
    //
    // `cudaMalloc` failing is the easy case. The one that matters is a machine where the weights already own
    // most of VRAM: the cache then takes what is left and `slots()` would report the number ASKED FOR while
    // `device_slot()` walks off the end. So the free-VRAM figure is read and compared BEFORE the allocation,
    // and the two numbers are named in the refusal.
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
        free_b = device_free_bytes();   // unified memory: what the OS can give back counts (see the header)
        if ((uint64_t) free_b < want) {
            char buf[320];
            std::snprintf(buf, sizeof buf,
                          "ExpertCache: %lld slots x %lld B = %.2f GiB, but only %.2f GiB of VRAM is free "
                          "(%.2f GiB of %.2f GiB total). Lower --expert-cache.",
                          (long long) n_slots, (long long) blob_bytes, (double) want / 1073741824.0,
                          (double) free_b / 1073741824.0, (double) (total_b - free_b) / 1073741824.0,
                          (double) total_b / 1073741824.0);
            err = buf;
            return false;
        }
    }

#if !defined(STRATA_EC_NO_VMM)
    if (seg_req_ > 0 && !segmented_supported()) {   // a driver without them: one block, the engine starts all the same
        std::fprintf(stderr, "ExpertCache: --vram-elastic is off: the driver or this GPU offers no virtual memory management; "
                             "the cache is one block\n");
        seg_req_ = 0;
    }
#endif
    if (seg_req_ > 0) {   // #533: --vram-elastic: physical segments behind one address range (zeroed below)
        if (!open_segmented(want, err)) return false;
    } else if (g_cache_vmm && vmm_available()) {
        // the elastic K/V: every chunk mapped now; the K/V may later take some of them (and give them back)
        auto r = std::make_unique<VmmRange>();
        if (!r->reserve(want) || !r->map_range(0, r->chunks(), [] { return (VmmChunk) 0; })) {
            char buf[256];
            std::snprintf(buf, sizeof buf, "ExpertCache: mapping %.2f GiB of VRAM failed: out of memory",
                          (double) want / 1073741824.0);
            err = buf;
            return false;
        }
        base_ = r->base();
        vmm_ = std::move(r);
    } else if (cudaMalloc((void**) &base_, (size_t) want) != cudaSuccess) {
        base_ = nullptr;
        char buf[256];
        std::snprintf(buf, sizeof buf, "ExpertCache: cudaMalloc(%.2f GiB) failed: %s",
                      (double) want / 1073741824.0, cudaGetErrorString(cudaGetLastError()));
        err = buf;
        return false;
    }
    // Zeroed so a slot read before it is filled is a DETERMINISTIC wrong answer rather than whatever the
    // allocator handed back.  A stale block of a previous process's memory would still sum to finite floats.
    if (cudaMemset(base_, 0, (size_t) want) != cudaSuccess) {
        err = "ExpertCache: cudaMemset of the slot arena failed";
        close();
        return false;
    }

    residency_.assign((size_t) (n_layers * n_expert), kNotResident);
    slots_ = n_slots;
    live_slots_ = n_slots;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    blob_ = blob_bytes;
#if defined(STRATA_USE_HIP)
    if (!ensure_blocking_staging((std::size_t) blob_, err)) {
        close();
        return false;
    }
#endif
    next_free_ = 0;
    fills_ = 0;
    admitted_ = 0;
    // R4.2g: each layer starts at the bottom of its own range.  Built here rather than lazily so `admit`
    // stays allocation-free on the token path.
    layer_next_.assign((size_t) (n_layers > 0 ? n_layers : 0), 0);
    layer_count_.assign((size_t) (n_layers > 0 ? n_layers : 0), 0);
    for (int64_t l = 0; l < n_layers; ++l) {
        int64_t lo = 0, hi = 0;
        layer_slot_range(l, lo, hi);
        layer_next_[(size_t) l] = (int32_t) lo;
    }
    return true;
}

bool ExpertCache::open_sized(const std::vector<int64_t>& slot_bytes, int64_t n_layers, int64_t n_expert,
                             std::string& err) {
    if (slot_bytes.empty()) { err = "ExpertCache: no slots"; return false; }
    int64_t mx = 0;
    std::vector<uint64_t> off(slot_bytes.size() + 1, 0);
    for (size_t i = 0; i < slot_bytes.size(); ++i) {
        // 256-byte aligned slots, so every blob starts where the kernels' vector loads expect it (and the
        // STRATA_MMQ_DIRECT guard after each blob, 0 by default)
        off[i + 1] = off[i] + (uint64_t) slot_stride(slot_bytes[i], guard_req_);
        mx = slot_bytes[i] > mx ? slot_bytes[i] : mx;
    }
    // one allocation of the summed size, through the uniform path's checks: n "slots" of 1 byte
    if (!open((int64_t) off.back(), n_layers, n_expert, 1, err)) return false;
    slots_ = (int64_t) slot_bytes.size();
    live_slots_ = slots_;
    blob_ = mx;
    off_ = std::move(off);
    guard_ = guard_req_;
#if defined(STRATA_USE_HIP)
    if (!ensure_blocking_staging((std::size_t) blob_, err)) {
        close();
        return false;
    }
#endif
    // #369: each layer's cursor at the bottom of its own range, as open() seeds it - open() above ran on byte-sized
    // "slots", so its seeds are not slot indices
    layer_next_.assign((size_t) (n_layers > 0 ? n_layers : 0), 0);
    layer_count_.assign((size_t) (n_layers > 0 ? n_layers : 0), 0);
    for (int64_t l = 0; l < n_layers; ++l) {
        int64_t lo = 0, hi = 0;
        layer_slot_range(l, lo, hi);
        layer_next_[(size_t) l] = (int32_t) lo;
    }
    return true;
}

void ExpertCache::close() {
#if defined(STRATA_USE_HIP)
    if (blocking_staging_) (void) cudaFreeHost(blocking_staging_);
    blocking_staging_ = nullptr;
    blocking_staging_bytes_ = 0;
#endif
    off_.clear();
    guard_ = 0;
    if (segmented_vmm_) {
        release_segmented();
    } else if (vmm_) {
        vmm_.reset();   // unmaps and frees every chunk it still holds
        base_ = nullptr;
    } else if (base_ != nullptr) {
        cudaFree(base_);
        base_ = nullptr;
    }
    residency_.clear();
    slots_ = 0;
    live_slots_ = 0;
    n_layers_ = 0;
    n_expert_ = 0;
    blob_ = 0;
    next_free_ = 0;
    fills_ = 0;
    admitted_ = 0;
    sealed_ = false;
    layer_next_.clear();
    layer_count_.clear();
    layer_quotas_.clear();
    layer_begin_.clear();
}

/// R4.2g.  Layer `l` owns `[l*q, (l+1)*q)` with `q = slots_ / n_layers_`; the LAST layer takes whatever is
/// left over, so the ranges always cover `0..slots_` exactly and no slot is orphaned by the division.
void ExpertCache::layer_slot_range(int64_t layer, int64_t& lo, int64_t& hi) const {
    lo = 0;
    hi = 0;
    if (n_layers_ <= 0 || slots_ <= 0 || layer < 0 || layer >= n_layers_) return;
    if (layer_quotas_.size() == (size_t) n_layers_ && layer_begin_.size() == (size_t) n_layers_ + 1) {
        lo = layer_begin_[(size_t) layer];
        hi = layer_begin_[(size_t) layer + 1];
        return;
    }
    const int64_t q = slots_ / n_layers_;
    lo = layer * q;
    hi = (layer == n_layers_ - 1) ? slots_ : (layer + 1) * q;
}

bool ExpertCache::set_layer_quotas(const std::vector<int64_t>& quotas, std::string& err) {
    if (quotas.size() != (size_t) n_layers_ || n_layers_ <= 0 || slots_ <= 0) {
        err = "ExpertCache::set_layer_quotas: quota count does not match the opened cache";
        return false;
    }
    std::vector<int64_t> begins(quotas.size() + 1, 0);
    for (size_t i = 0; i < quotas.size(); ++i) {
        if (quotas[i] <= 0 || quotas[i] > n_expert_ || quotas[i] > slots_ - begins[i]) {
            err = "ExpertCache::set_layer_quotas: a quota is out of range";
            return false;
        }
        begins[i + 1] = begins[i] + quotas[i];
    }
    if (begins.back() != slots_) {
        err = "ExpertCache::set_layer_quotas: quotas do not cover the opened cache";
        return false;
    }
    layer_quotas_ = quotas;
    layer_begin_ = std::move(begins);
    layer_next_.resize(quotas.size());
    layer_count_.assign(quotas.size(), 0);
    for (size_t i = 0; i < quotas.size(); ++i) layer_next_[i] = (int32_t) layer_begin_[i];
    return true;
}

int32_t ExpertCache::slot_of(int64_t layer, int64_t expert) const {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return kNotResident;
    const int32_t slot = residency_[(size_t) (layer * n_expert_ + expert)];
    return is_live_slot(slot, live_slots_) ? slot : kNotResident;
}

int32_t ExpertCache::admit(int64_t layer, int64_t expert) {
    if (sealed_) return kNotResident;   // STRATA_EXPERT_NIBBLE: nothing is admitted (and so filled) after the startup fill
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return kNotResident;
    const size_t at = (size_t) (layer * n_expert_ + expert);
    if (residency_[at] != kNotResident)
        return is_live_slot(residency_[at], live_slots_) ? residency_[at] : kNotResident;
    // R4.2g: THE PER-LAYER PATH.  Same "no eviction" rule, but the ceiling is this layer's own range rather
    // than one counter shared by all 48 - which is what confined the measured hit rate to 2.97%.
    if (per_layer_) {
        if (per_layer_rank_major_) {
            if (layer_count_.empty() || n_layers_ <= 0 || slots_ <= 0) return kNotResident;
            const int64_t q = slots_ / n_layers_;
            const int64_t r = (int64_t) layer_count_[(size_t) layer];
            int64_t slot = -1;
            if (r < q) {
                slot = r * n_layers_ + layer;
            } else if (layer == n_layers_ - 1 && q * n_layers_ + (r - q) < slots_) {
                slot = q * n_layers_ + (r - q);
            } else {
                return kNotResident;
            }
            if (!is_live_slot(slot, live_slots_)) return kNotResident;
            ++layer_count_[(size_t) layer];
            residency_[at] = (int32_t) slot;
            ++admitted_;
            return residency_[at];
        }
        if (layer_next_.empty()) return kNotResident;
        int64_t lo = 0, hi = 0;
        layer_slot_range(layer, lo, hi);
        if ((int64_t) layer_next_[(size_t) layer] >= hi) return kNotResident;   // this layer's quota is full
        const int32_t slot = layer_next_[(size_t) layer];
        if (!is_live_slot(slot, live_slots_)) return kNotResident;
        residency_[at] = layer_next_[(size_t) layer]++;
        ++admitted_;
        return residency_[at];
    }
    if (next_free_ >= live_slots_) return kNotResident;   // full: no eviction, deliberately - see the header
    residency_[at] = (int32_t) next_free_;
    return (int32_t) next_free_++;
}

uint8_t* ExpertCache::device_slot(int32_t slot) {
    if (!is_live_slot(slot, live_slots_)) return nullptr;
    if (!off_.empty()) return base_ + off_[(size_t) slot];
    return base_ + (size_t) slot * (size_t) blob_;
}

const uint8_t* ExpertCache::device_slot(int32_t slot) const {
    if (!is_live_slot(slot, live_slots_)) return nullptr;
    if (!off_.empty()) return base_ + off_[(size_t) slot];
    return base_ + (size_t) slot * (size_t) blob_;
}

bool ExpertCache::fill_slot(int32_t slot, const uint8_t* host_blob, void* stream, std::string& err, int64_t bytes) {
    if (sealed_) sealed_write("fill_slot", slot);
    const size_t n = (size_t) (bytes > 0 && bytes <= blob_ ? bytes : blob_);
    uint8_t* dst = device_slot(slot);
    if (dst == nullptr) {
        err = "ExpertCache::fill_slot: slot " + std::to_string(slot) + " is outside 0.." +
              std::to_string(slots_ - 1);
        return false;
    }
    if (host_blob == nullptr) {
        err = "ExpertCache::fill_slot: the host blob is null";
        return false;
    }
    const cudaError_t e = cudaMemcpyAsync(dst, host_blob, n, cudaMemcpyHostToDevice,
                                          (cudaStream_t) stream);
    if (e != cudaSuccess) {
        err = std::string("ExpertCache::fill_slot: ") + cudaGetErrorString(e);
        return false;
    }
    ++fills_;
    return true;
}

bool ExpertCache::fill_slot_blocking(int32_t slot, const uint8_t* host_blob, std::string& err, int64_t bytes) {
    if (sealed_) sealed_write("fill_slot_blocking", slot);
    const size_t n = (size_t) (bytes > 0 && bytes <= blob_ ? bytes : blob_);
    uint8_t* dst = device_slot(slot);
    if (dst == nullptr) {
        err = "ExpertCache::fill_slot_blocking: slot outside the arena";
        return false;
    }
    if (host_blob == nullptr) {
        err = "ExpertCache::fill_slot_blocking: the host blob is null";
        return false;
    }
#if defined(STRATA_USE_HIP)
    // Bound HIP's pageable-source staging to one expert instead of repeatedly
    // registering regions of the mmap. The blocking copy completes before reuse.
    if (!blocking_staging_ || n > blocking_staging_bytes_) {
        err = "ExpertCache::fill_slot_blocking: HIP staging buffer is too small";
        return false;
    }
    std::memcpy(blocking_staging_, host_blob, n);
    const cudaError_t e = cudaMemcpy(dst, blocking_staging_, n, cudaMemcpyHostToDevice);
#else
    const cudaError_t e = cudaMemcpy(dst, host_blob, n, cudaMemcpyHostToDevice);
#endif
    if (e != cudaSuccess) {
        err = std::string("ExpertCache::fill_slot_blocking: ") + cudaGetErrorString(e);
        return false;
    }
    ++fills_;
    return true;
}

bool ExpertCache::fill_slot_queued(int32_t slot, const uint8_t* host_blob, std::string& err, int64_t bytes) {
    if (sealed_) sealed_write("fill_slot_queued", slot);
    const size_t n = (size_t) (bytes > 0 && bytes <= blob_ ? bytes : blob_);
    uint8_t* dst = device_slot(slot);
    if (dst == nullptr || host_blob == nullptr) {
        err = dst == nullptr ? "ExpertCache::fill_slot_queued: slot outside the arena"
                             : "ExpertCache::fill_slot_queued: the host blob is null";
        return false;
    }
    const cudaError_t e = cudaMemcpyAsync(dst, host_blob, n, cudaMemcpyHostToDevice, (cudaStream_t) 0);
    if (e != cudaSuccess) {
        err = std::string("ExpertCache::fill_slot_queued: ") + cudaGetErrorString(e);
        return false;
    }
    ++fills_;
    return true;
}

bool ExpertCache::fill_slot_queued_parts(int32_t slot, const BlobPart* parts, int n, std::string& err) {
    if (sealed_) sealed_write("fill_slot_queued_parts", slot);
    uint8_t* dst = device_slot(slot);
    if (dst == nullptr || parts == nullptr || n <= 0) {
        err = "ExpertCache::fill_slot_queued_parts: slot outside the arena or no pieces";
        return false;
    }
    uint64_t total = 0;
    for (int i = 0; i < n; ++i) total += parts[i].n;
    if (total > (uint64_t) blob_) {
        err = "ExpertCache::fill_slot_queued_parts: the pieces are larger than a slot";
        return false;
    }
    for (int i = 0; i < n; ++i) {
        const cudaError_t e = cudaMemcpyAsync(dst + parts[i].at, parts[i].p, (size_t) parts[i].n, cudaMemcpyHostToDevice,
                                              (cudaStream_t) 0);
        if (e != cudaSuccess) {
            err = std::string("ExpertCache::fill_slot_queued_parts: ") + cudaGetErrorString(e);
            return false;
        }
    }
    ++fills_;
    return true;
}

bool ExpertCache::sync_queued(std::string& err) {
    const cudaError_t e = cudaStreamSynchronize((cudaStream_t) 0);
    if (e != cudaSuccess) {
        err = std::string("ExpertCache::sync_queued: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

bool ExpertCache::verify_slot(int32_t slot, const uint8_t* host_blob, std::string& err, int64_t bytes) {
    const int64_t nb = bytes > 0 && bytes <= blob_ ? bytes : blob_;
    const uint8_t* src = device_slot(slot);
    if (src == nullptr) {
        err = "ExpertCache::verify_slot: slot outside the arena";
        return false;
    }
    // `cudaMemcpy` and not `cudaMemcpyAsync`: this is a startup check, and a check that can be read before it
    // has happened is not a check.  It also synchronises the fills queued before it, which is what makes the
    // comparison meaningful.
    std::vector<uint8_t> got((size_t) nb);
    const cudaError_t e = cudaMemcpy(got.data(), src, (size_t) nb, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        err = std::string("ExpertCache::verify_slot: ") + cudaGetErrorString(e);
        return false;
    }
    if (std::memcmp(got.data(), host_blob, (size_t) nb) != 0) {
        size_t first = 0;
        while (first < (size_t) nb && got[first] == host_blob[first]) ++first;
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "ExpertCache::verify_slot: slot %d differs from the arena at byte %llu (of %lld)",
                      (int) slot, (unsigned long long) first, (long long) blob_);
        err = buf;
        return false;
    }
    return true;
}

}  // namespace strata::core
