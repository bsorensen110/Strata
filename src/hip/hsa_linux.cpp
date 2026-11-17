// src/hip/hsa_linux.cpp - Dynamic ROCr/HSA runtime shim implementation (Linux only).
//
// Port note: from warpfront/hipfire redline-rocr (Apache-2.0)
// Ported from warpfront/hipfire crates/redline-rocr/src/{abi.rs,runtime.rs,lib.rs},
// crates/hsa-bridge/src/{ffi.rs,lib.rs}, and crates/rdna-compute/src/loaded_code_check.rs.

#if defined(__linux__) && defined(STRATA_WITH_HSA)

#include "strata/hip/hsa.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
#include <utility>

namespace strata::hip::hsa {

namespace {

inline void bounded_poll_pause(uint32_t& polls) {
    ++polls;
    if ((polls & 0x3fu) == 0u) {
        std::this_thread::yield();
    } else {
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#elif defined(__aarch64__)
        asm volatile("yield" ::: "memory");
#endif
    }
}

extern "C" void hsa_queue_error_callback(HsaStatus status, HsaQueue* /*source*/, void* data) {
    if (data == nullptr) return;
    static_cast<HsaQueueFaultState*>(data)->record(status);
}

bool is_power_of_two_u32(uint32_t v) {
    return v != 0u && (v & (v - 1u)) == 0u;
}

bool is_power_of_two_size(size_t v) {
    return v != 0u && (v & (v - 1u)) == 0u;
}

bool parse_pci_bus_id_string(const char* str, PciBusId& out) {
    if (str == nullptr) return false;
    unsigned int domain = 0, bus = 0, dev = 0, func = 0;
    if (std::sscanf(str, "%x:%x:%x.%x", &domain, &bus, &dev, &func) != 4) {
        return false;
    }
    if (bus > 0xffu || dev > 0x1fu || func > 0x7u) return false;
    out.domain = static_cast<uint32_t>(domain);
    out.bus = static_cast<uint8_t>(bus);
    out.device = static_cast<uint8_t>(dev);
    out.function = static_cast<uint8_t>(func);
    return true;
}

std::vector<std::string> rocm_library_candidates(const char* soname) {
    std::vector<std::string> out;
    for (const char* env_name : {"ROCM_PATH", "HIP_PATH"}) {
        const char* root = std::getenv(env_name);
        if (root != nullptr && root[0] != '\0') {
            out.push_back(std::string(root) + "/lib/" + soname);
            out.push_back(std::string(root) + "/lib64/" + soname);
        }
    }
    out.push_back(std::string("/opt/rocm/lib/") + soname);
    out.push_back(std::string("/opt/rocm/lib64/") + soname);
    out.push_back(soname);
    return out;
}

}  // namespace

// ---- Dynamically resolved ROCr and HIP function pointers ----

struct HsaRuntime::Impl {
    void* lib_handle = nullptr;
    bool initialized = false;

    // Core HSA lifecycle & status
    HsaStatus (*hsa_init)() = nullptr;
    HsaStatus (*hsa_shut_down)() = nullptr;
    HsaStatus (*hsa_status_string)(HsaStatus, const char**) = nullptr;
    HsaStatus (*hsa_system_get_info)(uint32_t, void*) = nullptr;
    HsaStatus (*hsa_system_major_extension_supported)(uint16_t, uint16_t, uint16_t*, bool*) = nullptr;
    HsaStatus (*hsa_system_get_major_extension_table)(uint16_t, uint16_t, size_t, void*) = nullptr;

    // Agents
    HsaStatus (*hsa_iterate_agents)(HsaStatus (*)(HsaAgent, void*), void*) = nullptr;
    HsaStatus (*hsa_agent_get_info)(HsaAgent, uint32_t, void*) = nullptr;

    // Signals
    HsaStatus (*hsa_signal_create)(HsaSignalValue, uint32_t, const HsaAgent*, HsaSignal*) = nullptr;
    HsaStatus (*hsa_signal_destroy)(HsaSignal) = nullptr;
    HsaSignalValue (*hsa_signal_load_relaxed)(HsaSignal) = nullptr;
    HsaSignalValue (*hsa_signal_load_scacquire)(HsaSignal) = nullptr;
    void (*hsa_signal_store_relaxed)(HsaSignal, HsaSignalValue) = nullptr;
    void (*hsa_signal_store_screlease)(HsaSignal, HsaSignalValue) = nullptr;
    HsaSignalValue (*hsa_signal_wait_scacquire)(
        HsaSignal, uint32_t, HsaSignalValue, uint64_t, uint32_t) = nullptr;

    // Queues
    HsaStatus (*hsa_queue_create)(
        HsaAgent, uint32_t, uint32_t,
        void (*)(HsaStatus, HsaQueue*, void*), void*,
        uint32_t, uint32_t, HsaQueue**) = nullptr;
    HsaStatus (*hsa_queue_destroy)(HsaQueue*) = nullptr;
    HsaStatus (*hsa_queue_inactivate)(HsaQueue*) = nullptr;
    uint64_t (*hsa_queue_load_read_index_relaxed)(const HsaQueue*) = nullptr;
    uint64_t (*hsa_queue_load_read_index_scacquire)(const HsaQueue*) = nullptr;
    uint64_t (*hsa_queue_load_write_index_relaxed)(const HsaQueue*) = nullptr;
    uint64_t (*hsa_queue_add_write_index_relaxed)(const HsaQueue*, uint64_t) = nullptr;

    // Memory registration & AMD memory pools
    HsaStatus (*hsa_memory_register)(void*, size_t) = nullptr;
    HsaStatus (*hsa_memory_deregister)(void*, size_t) = nullptr;
    HsaStatus (*hsa_amd_agent_iterate_memory_pools)(
        HsaAgent, HsaStatus (*)(HsaMemoryPool, void*), void*) = nullptr;
    HsaStatus (*hsa_amd_memory_pool_get_info)(HsaMemoryPool, uint32_t, void*) = nullptr;
    HsaStatus (*hsa_amd_agent_memory_pool_get_info)(
        HsaAgent, HsaMemoryPool, uint32_t, void*) = nullptr;
    HsaStatus (*hsa_amd_memory_pool_allocate)(
        HsaMemoryPool, size_t, uint32_t, void**) = nullptr;
    HsaStatus (*hsa_amd_memory_pool_free)(void*) = nullptr;
    HsaStatus (*hsa_amd_agents_allow_access)(
        uint32_t, const HsaAgent*, const uint32_t*, const void*) = nullptr;
    HsaStatus (*hsa_amd_pointer_info)(
        const void*, HsaAmdPointerInfo*, void* (*)(size_t), uint32_t*, HsaAgent**) = nullptr;

    // Executables & code objects
    HsaStatus (*hsa_code_object_reader_create_from_memory)(
        const void*, size_t, HsaCodeObjectReader*) = nullptr;
    HsaStatus (*hsa_code_object_reader_destroy)(HsaCodeObjectReader) = nullptr;
    HsaStatus (*hsa_executable_create_alt)(
        uint32_t, uint32_t, const char*, HsaExecutable*) = nullptr;
    HsaStatus (*hsa_executable_destroy)(HsaExecutable) = nullptr;
    HsaStatus (*hsa_executable_load_agent_code_object)(
        HsaExecutable, HsaAgent, HsaCodeObjectReader, const char*, HsaLoadedCodeObject*) = nullptr;
    HsaStatus (*hsa_executable_freeze)(HsaExecutable, const char*) = nullptr;
    HsaStatus (*hsa_executable_get_info)(HsaExecutable, uint32_t, void*) = nullptr;
    HsaStatus (*hsa_executable_get_symbol_by_name)(
        HsaExecutable, const char*, const HsaAgent*, HsaExecutableSymbol*) = nullptr;
    HsaStatus (*hsa_executable_symbol_get_info)(HsaExecutableSymbol, uint32_t, void*) = nullptr;

    // AMD Loader extension 1.03 table
    HsaVenAmdLoaderTable103 loader = {};

    // Optional HIP runtime helpers resolved from RTLD_DEFAULT / libamdhip64.so
    const char* (*hip_kernel_name_ref_by_ptr)(const void*, void*) = nullptr;
    int (*hip_get_func_by_symbol)(void**, const void*) = nullptr;
    const char* (*hip_kernel_name_ref)(const void*) = nullptr;
    int (*hip_func_get_attributes)(void*, const void*) = nullptr;
    int (*hip_device_get_pci_bus_id)(char*, int, int) = nullptr;

    ~Impl() {
        if (initialized && hsa_shut_down != nullptr) {
            (void) hsa_shut_down();
        }
        if (lib_handle != nullptr) {
            (void) dlclose(lib_handle);
        }
    }

    bool load_symbols(std::string& err) {
        // Prefer an already-loaded ROCr image in-process first (RTLD_NOLOAD), then normal dlopen.
        for (const char* name : {"libhsa-runtime64.so.1", "libhsa-runtime64.so"}) {
            lib_handle = dlopen(name, RTLD_NOW | RTLD_NOLOAD);
            if (lib_handle != nullptr) break;
        }
        if (lib_handle == nullptr) {
            std::vector<std::string> candidates = rocm_library_candidates("libhsa-runtime64.so.1");
            const auto extra = rocm_library_candidates("libhsa-runtime64.so");
            candidates.insert(candidates.end(), extra.begin(), extra.end());
            for (const auto& candidate : candidates) {
                lib_handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
                if (lib_handle != nullptr) break;
            }
        }
        if (lib_handle == nullptr) {
            const char* dl_err = dlerror();
            err = std::string("failed to dlopen libhsa-runtime64.so.1: ") +
                  (dl_err ? dl_err : "unknown dlopen error");
            return false;
        }

        auto req = [&](const char* sym_name, auto& fn_ptr) -> bool {
            void* raw = dlsym(lib_handle, sym_name);
            if (raw == nullptr) {
                err = std::string("libhsa-runtime64 is missing required symbol ") + sym_name;
                return false;
            }
            std::memcpy(&fn_ptr, &raw, sizeof(raw));
            return true;
        };
        auto opt = [&](const char* sym_name, auto& fn_ptr) {
            void* raw = dlsym(lib_handle, sym_name);
            if (raw != nullptr) {
                std::memcpy(&fn_ptr, &raw, sizeof(raw));
            }
        };

        if (!req("hsa_init", hsa_init) ||
            !req("hsa_shut_down", hsa_shut_down) ||
            !req("hsa_status_string", hsa_status_string) ||
            !req("hsa_system_get_info", hsa_system_get_info) ||
            !req("hsa_system_major_extension_supported", hsa_system_major_extension_supported) ||
            !req("hsa_system_get_major_extension_table", hsa_system_get_major_extension_table) ||
            !req("hsa_iterate_agents", hsa_iterate_agents) ||
            !req("hsa_agent_get_info", hsa_agent_get_info) ||
            !req("hsa_signal_create", hsa_signal_create) ||
            !req("hsa_signal_destroy", hsa_signal_destroy) ||
            !req("hsa_signal_load_relaxed", hsa_signal_load_relaxed) ||
            !req("hsa_signal_load_scacquire", hsa_signal_load_scacquire) ||
            !req("hsa_signal_store_relaxed", hsa_signal_store_relaxed) ||
            !req("hsa_signal_store_screlease", hsa_signal_store_screlease) ||
            !req("hsa_signal_wait_scacquire", hsa_signal_wait_scacquire) ||
            !req("hsa_queue_create", hsa_queue_create) ||
            !req("hsa_queue_destroy", hsa_queue_destroy) ||
            !req("hsa_queue_inactivate", hsa_queue_inactivate) ||
            !req("hsa_queue_load_read_index_relaxed", hsa_queue_load_read_index_relaxed) ||
            !req("hsa_queue_load_read_index_scacquire", hsa_queue_load_read_index_scacquire) ||
            !req("hsa_queue_load_write_index_relaxed", hsa_queue_load_write_index_relaxed) ||
            !req("hsa_queue_add_write_index_relaxed", hsa_queue_add_write_index_relaxed) ||
            !req("hsa_memory_register", hsa_memory_register) ||
            !req("hsa_memory_deregister", hsa_memory_deregister) ||
            !req("hsa_amd_agent_iterate_memory_pools", hsa_amd_agent_iterate_memory_pools) ||
            !req("hsa_amd_memory_pool_get_info", hsa_amd_memory_pool_get_info) ||
            !req("hsa_amd_memory_pool_allocate", hsa_amd_memory_pool_allocate) ||
            !req("hsa_amd_memory_pool_free", hsa_amd_memory_pool_free) ||
            !req("hsa_amd_agents_allow_access", hsa_amd_agents_allow_access) ||
            !req("hsa_amd_pointer_info", hsa_amd_pointer_info) ||
            !req("hsa_code_object_reader_create_from_memory", hsa_code_object_reader_create_from_memory) ||
            !req("hsa_code_object_reader_destroy", hsa_code_object_reader_destroy) ||
            !req("hsa_executable_create_alt", hsa_executable_create_alt) ||
            !req("hsa_executable_destroy", hsa_executable_destroy) ||
            !req("hsa_executable_load_agent_code_object", hsa_executable_load_agent_code_object) ||
            !req("hsa_executable_freeze", hsa_executable_freeze) ||
            !req("hsa_executable_get_info", hsa_executable_get_info) ||
            !req("hsa_executable_get_symbol_by_name", hsa_executable_get_symbol_by_name) ||
            !req("hsa_executable_symbol_get_info", hsa_executable_symbol_get_info)) {
            return false;
        }
        opt("hsa_amd_agent_memory_pool_get_info", hsa_amd_agent_memory_pool_get_info);

        // Optional HIP runtime symbol helpers from RTLD_DEFAULT (already loaded by HIP host binary)
        auto opt_hip = [&](const char* sym_name, auto& fn_ptr) {
            void* raw = dlsym(RTLD_DEFAULT, sym_name);
            if (raw != nullptr) {
                std::memcpy(&fn_ptr, &raw, sizeof(raw));
            }
        };
        opt_hip("hipKernelNameRefByPtr", hip_kernel_name_ref_by_ptr);
        opt_hip("hipGetFuncBySymbol", hip_get_func_by_symbol);
        opt_hip("hipKernelNameRef", hip_kernel_name_ref);
        opt_hip("hipFuncGetAttributes", hip_func_get_attributes);
        opt_hip("hipDeviceGetPCIBusId", hip_device_get_pci_bus_id);

        return true;
    }

    bool resolve_loader_extension(std::string& err) {
        uint16_t minor = 0;
        bool supported = false;
        const HsaStatus st =
            hsa_system_major_extension_supported(kHsaExtensionAmdLoader, 1u, &minor, &supported);
        if (st != kHsaStatusSuccess || !supported) {
            err = "HSA_EXTENSION_AMD_LOADER (0x201) major 1 is not supported by this ROCr runtime";
            return false;
        }
        (void) minor;
        loader = HsaVenAmdLoaderTable103{};
        const HsaStatus tbl_st = hsa_system_get_major_extension_table(
            kHsaExtensionAmdLoader, 1u, sizeof(HsaVenAmdLoaderTable103), &loader);
        if (tbl_st != kHsaStatusSuccess ||
            loader.iterate_executables == nullptr ||
            loader.executable_iterate_loaded_code_objects == nullptr ||
            loader.loaded_code_object_get_info == nullptr) {
            err = "HSA_EXTENSION_AMD_LOADER 1.03 function table is incomplete";
            return false;
        }
        return true;
    }
};

// ---- HsaBuffer ----

HsaBuffer::~HsaBuffer() {
    reset();
}

HsaBuffer::HsaBuffer(HsaBuffer&& other) noexcept
    : runtime_(std::move(other.runtime_)),
      ptr_(std::exchange(other.ptr_, nullptr)),
      length_(std::exchange(other.length_, 0)),
      allocation_size_(std::exchange(other.allocation_size_, 0)),
      device_local_(std::exchange(other.device_local_, false)) {}

HsaBuffer& HsaBuffer::operator=(HsaBuffer&& other) noexcept {
    if (this != &other) {
        reset();
        runtime_ = std::move(other.runtime_);
        ptr_ = std::exchange(other.ptr_, nullptr);
        length_ = std::exchange(other.length_, 0);
        allocation_size_ = std::exchange(other.allocation_size_, 0);
        device_local_ = std::exchange(other.device_local_, false);
    }
    return *this;
}

bool HsaBuffer::write_exact(const void* src, size_t n) {
    if (ptr_ == nullptr || src == nullptr || n != length_) return false;
    std::memcpy(ptr_, src, n);
    return true;
}

void HsaBuffer::publish_host_writes() const {
    if (!device_local_ || ptr_ == nullptr || length_ == 0) return;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    volatile const uint8_t* tail = static_cast<volatile const uint8_t*>(ptr_) + (length_ - 1u);
    (void) *tail;
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

void HsaBuffer::reset() {
    if (ptr_ != nullptr && runtime_ != nullptr && runtime_->impl_ != nullptr) {
        (void) runtime_->impl_->hsa_amd_memory_pool_free(ptr_);
    }
    ptr_ = nullptr;
    length_ = 0;
    allocation_size_ = 0;
    device_local_ = false;
    runtime_.reset();
}

// ---- HsaCompletionSignal ----

HsaCompletionSignal::~HsaCompletionSignal() {
    if (signal_.handle != 0 && runtime_ != nullptr && runtime_->impl_ != nullptr) {
        (void) runtime_->impl_->hsa_signal_destroy(signal_);
    }
}

HsaCompletionSignal::HsaCompletionSignal(HsaCompletionSignal&& other) noexcept
    : runtime_(std::move(other.runtime_)),
      signal_(std::exchange(other.signal_, HsaSignal{0})) {}

HsaCompletionSignal& HsaCompletionSignal::operator=(HsaCompletionSignal&& other) noexcept {
    if (this != &other) {
        if (signal_.handle != 0 && runtime_ != nullptr && runtime_->impl_ != nullptr) {
            (void) runtime_->impl_->hsa_signal_destroy(signal_);
        }
        runtime_ = std::move(other.runtime_);
        signal_ = std::exchange(other.signal_, HsaSignal{0});
    }
    return *this;
}

void HsaCompletionSignal::reset() {
    if (signal_.handle != 0 && runtime_ != nullptr) {
        runtime_->impl_->hsa_signal_store_screlease(signal_, 1);
    }
}

bool HsaCompletionSignal::is_complete() const {
    if (signal_.handle == 0 || runtime_ == nullptr) return false;
    return runtime_->impl_->hsa_signal_load_scacquire(signal_) < 1;
}

HsaSignalValue HsaCompletionSignal::value_scacquire() const {
    if (signal_.handle == 0 || runtime_ == nullptr) return -1;
    return runtime_->impl_->hsa_signal_load_scacquire(signal_);
}

bool HsaCompletionSignal::wait_timeout(uint64_t timeout_ns, std::string& err) const {
    if (signal_.handle == 0 || runtime_ == nullptr) {
        err = "cannot wait on an uninitialized HSA completion signal";
        return false;
    }
    const auto started = std::chrono::steady_clock::now();
    uint32_t polls = 0;
    for (;;) {
        if (is_complete()) return true;
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started);
        if (static_cast<uint64_t>(elapsed.count()) >= timeout_ns) {
            err = "HSA completion signal timed out";
            return false;
        }
        bounded_poll_pause(polls);
    }
}

// ---- HsaAqlQueue ----

HsaAqlQueue::~HsaAqlQueue() {
    if (raw_ != nullptr && runtime_ != nullptr && runtime_->impl_ != nullptr) {
        if (active_) {
            (void) runtime_->impl_->hsa_queue_inactivate(raw_);
            active_ = false;
        }
        const HsaStatus st = runtime_->impl_->hsa_queue_destroy(raw_);
        if (st != kHsaStatusSuccess) {
            // Mirror redline-rocr runtime.rs:1642-1653: if queue_destroy fails, leak fault state
            // so a late ROCr callback cannot dereference freed memory.
            (void) fault_.release();
        }
        raw_ = nullptr;
    }
}

HsaAqlQueue::HsaAqlQueue(HsaAqlQueue&& other) noexcept
    : runtime_(std::move(other.runtime_)),
      raw_(std::exchange(other.raw_, nullptr)),
      fault_(std::move(other.fault_)),
      active_(std::exchange(other.active_, false)) {}

HsaAqlQueue& HsaAqlQueue::operator=(HsaAqlQueue&& other) noexcept {
    if (this != &other) {
        std::string ignore;
        (void) inactivate(ignore);
        if (raw_ != nullptr && runtime_ != nullptr && runtime_->impl_ != nullptr) {
            if (runtime_->impl_->hsa_queue_destroy(raw_) != kHsaStatusSuccess) {
                (void) fault_.release();
            }
        }
        runtime_ = std::move(other.runtime_);
        raw_ = std::exchange(other.raw_, nullptr);
        fault_ = std::move(other.fault_);
        active_ = std::exchange(other.active_, false);
    }
    return *this;
}

bool HsaAqlQueue::check_fault(std::string& err) const {
    if (raw_ == nullptr || !active_) {
        err = "HSA queue is inactive";
        return false;
    }
    HsaStatus fault_status = kHsaStatusSuccess;
    if (fault_ != nullptr && fault_->has_fault(fault_status)) {
        err = "HSA queue " + std::to_string(raw_->id) + " reported asynchronous fault: " +
              (runtime_ ? runtime_->status_string(fault_status) : std::to_string(fault_status));
        return false;
    }
    return true;
}

bool HsaAqlQueue::inactivate(std::string& err) {
    if (!active_ || raw_ == nullptr || runtime_ == nullptr) return true;
    const HsaStatus st = runtime_->impl_->hsa_queue_inactivate(raw_);
    active_ = false;
    if (st != kHsaStatusSuccess) {
        err = "hsa_queue_inactivate failed: " + runtime_->status_string(st);
        return false;
    }
    return true;
}

bool HsaAqlQueue::wait_for_capacity(
    uint64_t packet_count,
    uint64_t timeout_ns,
    std::string& err) const {
    const auto started = std::chrono::steady_clock::now();
    uint32_t polls = 0;
    for (;;) {
        if (!check_fault(err)) return false;
        const uint64_t write_idx = runtime_->impl_->hsa_queue_load_write_index_relaxed(raw_);
        const uint64_t read_idx = runtime_->impl_->hsa_queue_load_read_index_scacquire(raw_);
        const uint64_t last_id = write_idx + packet_count - 1u;
        if (last_id - read_idx < static_cast<uint64_t>(raw_->size)) {
            return true;
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started);
        if (static_cast<uint64_t>(elapsed.count()) >= timeout_ns) {
            err = "HSA queue capacity timed out";
            return false;
        }
        bounded_poll_pause(polls);
    }
}

bool HsaAqlQueue::submit_packets(
    const pm4::PacketImage* packets,
    size_t packet_count,
    uint64_t timeout_ns,
    std::string& err) {
    if (packet_count == 0) return true;
    if (!check_fault(err)) return false;
    if (packet_count > static_cast<size_t>(raw_->size)) {
        err = "packet batch exceeds HSA queue ring size";
        return false;
    }
    if (!wait_for_capacity(static_cast<uint64_t>(packet_count), timeout_ns, err)) {
        std::string ignore;
        (void) inactivate(ignore);
        return false;
    }

    const uint64_t count = static_cast<uint64_t>(packet_count);
    const uint64_t first_id = runtime_->impl_->hsa_queue_add_write_index_relaxed(raw_, count);
    const uint64_t last_id = first_id + count - 1u;
    const uint64_t mask = static_cast<uint64_t>(raw_->size - 1u);
    auto* base = static_cast<uint8_t*>(raw_->base_address);

    for (size_t i = 0; i < packet_count; ++i) {
        const uint64_t packet_id = first_id + static_cast<uint64_t>(i);
        const size_t slot_index = static_cast<size_t>(packet_id & mask);
        uint8_t* slot = base + slot_index * pm4::kAqlPacketBytes;

        // Copy payload bytes 4..64 before publishing the header word at 0..4 (runtime.rs:1589-1598).
        std::memcpy(slot + 4u, packets[i].bytes.data() + 4u, pm4::kAqlPacketBytes - 4u);
        __atomic_store_n(
            reinterpret_cast<uint32_t*>(slot), packets[i].header_word, __ATOMIC_RELEASE);
    }

    // Ring the doorbell with the final packet ID (runtime.rs:1603-1615).
    runtime_->impl_->hsa_signal_store_relaxed(
        raw_->doorbell_signal, static_cast<HsaSignalValue>(last_id));
    return true;
}

bool HsaAqlQueue::wait_signal(
    const HsaCompletionSignal& signal,
    uint64_t timeout_ns,
    std::string& err) {
    const auto started = std::chrono::steady_clock::now();
    uint32_t polls = 0;
    for (;;) {
        if (!check_fault(err)) {
            std::string ignore;
            (void) inactivate(ignore);
            return false;
        }
        if (signal.is_complete()) {
            return true;
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started);
        if (static_cast<uint64_t>(elapsed.count()) >= timeout_ns) {
            err = "HSA queue " + std::to_string(id()) + " timed out waiting for completion signal";
            std::string ignore;
            (void) inactivate(ignore);
            return false;
        }
        bounded_poll_pause(polls);
    }
}

bool HsaAqlQueue::submit_and_wait_pm4_ib(
    const HsaBuffer& ib_buffer,
    uint32_t ib_dwords,
    HsaCompletionSignal& completion,
    uint64_t timeout_ns,
    std::string& err) {
    if (!ib_buffer.valid()) {
        err = "PM4 indirect buffer is not allocated";
        return false;
    }
    pm4::PacketImage pkt{};
    const pm4::PacketErrorCode perr = pm4::PacketImage::pm4_indirect_buffer(
        ib_buffer.gpu_address(), ib_dwords, completion.handle(), pkt);
    if (perr != pm4::PacketErrorCode::Ok) {
        err = "failed to build AQL vendor PM4-IB packet";
        return false;
    }
    ib_buffer.publish_host_writes();
    completion.reset();
    if (!submit_packets(&pkt, 1u, timeout_ns, err)) {
        return false;
    }
    return wait_signal(completion, timeout_ns, err);
}

// ---- HsaStandaloneExecutable ----

HsaStandaloneExecutable::~HsaStandaloneExecutable() {
    if (runtime_ != nullptr && runtime_->impl_ != nullptr) {
        if (executable_.handle != 0) {
            (void) runtime_->impl_->hsa_executable_destroy(executable_);
        }
        if (reader_.handle != 0) {
            (void) runtime_->impl_->hsa_code_object_reader_destroy(reader_);
        }
    }
}

HsaStandaloneExecutable::HsaStandaloneExecutable(HsaStandaloneExecutable&& other) noexcept
    : runtime_(std::move(other.runtime_)),
      gpu_agent_(std::exchange(other.gpu_agent_, HsaAgent{0})),
      executable_(std::exchange(other.executable_, HsaExecutable{0})),
      reader_(std::exchange(other.reader_, HsaCodeObjectReader{0})),
      code_object_bytes_(std::move(other.code_object_bytes_)) {}

HsaStandaloneExecutable& HsaStandaloneExecutable::operator=(HsaStandaloneExecutable&& other) noexcept {
    if (this != &other) {
        if (runtime_ != nullptr && runtime_->impl_ != nullptr) {
            if (executable_.handle != 0) {
                (void) runtime_->impl_->hsa_executable_destroy(executable_);
            }
            if (reader_.handle != 0) {
                (void) runtime_->impl_->hsa_code_object_reader_destroy(reader_);
            }
        }
        runtime_ = std::move(other.runtime_);
        gpu_agent_ = std::exchange(other.gpu_agent_, HsaAgent{0});
        executable_ = std::exchange(other.executable_, HsaExecutable{0});
        reader_ = std::exchange(other.reader_, HsaCodeObjectReader{0});
        code_object_bytes_ = std::move(other.code_object_bytes_);
    }
    return *this;
}

bool HsaStandaloneExecutable::resolve_kernel(
    const char* symbol_name,
    HsaResolvedKernel& out,
    std::string& err) const {
    if (!valid() || runtime_ == nullptr || symbol_name == nullptr || symbol_name[0] == '\0') {
        err = "invalid standalone HSA executable or kernel symbol name";
        return false;
    }
    std::string base(symbol_name);
    if (base.size() > 3u && base.compare(base.size() - 3u, 3u, ".kd") == 0) {
        base.resize(base.size() - 3u);
    }
    const std::string kd_name = base + ".kd";
    if (!runtime_->query_symbol_and_pm4_metadata(executable_, base, kd_name, out, err)) {
        // Fallback to raw ELF bytes stored in code_object_bytes_
        if (out.loader.kernel_object != 0 &&
            pm4::parse_kernel_pm4_metadata_from_elf(
                code_object_bytes_.data(),
                code_object_bytes_.size(),
                kd_name.c_str(),
                out.loader.kernel_object,
                out.pm4)) {
            err.clear();
            return true;
        }
        return false;
    }
    return true;
}

// ---- HsaRuntime ----

HsaRuntime::HsaRuntime() : impl_(std::make_unique<Impl>()) {}

HsaRuntime::~HsaRuntime() = default;

std::string HsaRuntime::status_string(HsaStatus status) const {
    if (impl_ != nullptr && impl_->hsa_status_string != nullptr) {
        const char* str = nullptr;
        if (impl_->hsa_status_string(status, &str) == kHsaStatusSuccess && str != nullptr) {
            return std::string(str);
        }
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "HSA status 0x%x", static_cast<unsigned int>(status));
    return std::string(buf);
}

std::shared_ptr<HsaRuntime> HsaRuntime::initialize(int hip_device_ordinal, std::string& err) {
    // Standing note: ROC_GLOBAL_CU_MASK / HSA_CU_MASK is banned on RDNA4/gfx12 PM4 queues (MES hang).
    for (const char* env_var : {"ROC_GLOBAL_CU_MASK", "HSA_CU_MASK"}) {
        const char* val = std::getenv(env_var);
        if (val != nullptr && val[0] != '\0') {
            err = std::string(env_var) +
                  " is set in the environment; refused on RDNA4 PM4 replay (MES hang hazard)";
            return nullptr;
        }
    }

    std::shared_ptr<HsaRuntime> rt(new HsaRuntime());
    if (!rt->impl_->load_symbols(err)) {
        return nullptr;
    }

    const HsaStatus init_st = rt->impl_->hsa_init();
    if (init_st != kHsaStatusSuccess) {
        err = "hsa_init failed: " + rt->status_string(init_st);
        return nullptr;
    }
    rt->impl_->initialized = true;

    if (!rt->impl_->resolve_loader_extension(err)) {
        return nullptr;
    }

    // Enumerate CPU and GPU agents (runtime.rs:198-343).
    std::vector<HsaAgent> raw_agents;
    auto collect_agent = +[](HsaAgent agent, void* data) -> HsaStatus {
        static_cast<std::vector<HsaAgent>*>(data)->push_back(agent);
        return kHsaStatusSuccess;
    };
    const HsaStatus iter_st = rt->impl_->hsa_iterate_agents(collect_agent, &raw_agents);
    if (iter_st != kHsaStatusSuccess) {
        err = "hsa_iterate_agents failed: " + rt->status_string(iter_st);
        return nullptr;
    }

    std::vector<HsaAgentInfo> gpu_agents;
    bool have_cpu = false;
    for (const HsaAgent agent : raw_agents) {
        HsaAgentInfo info{};
        info.handle = agent;
        char name_buf[64] = {};
        if (rt->impl_->hsa_agent_get_info(agent, kHsaAgentInfoName, name_buf) != kHsaStatusSuccess ||
            rt->impl_->hsa_agent_get_info(agent, kHsaAgentInfoDevice, &info.device_type) != kHsaStatusSuccess) {
            err = "hsa_agent_get_info(NAME/DEVICE) failed";
            return nullptr;
        }
        name_buf[sizeof(name_buf) - 1u] = '\0';
        info.name = name_buf;
        const size_t colon = info.name.find(':');
        if (colon != std::string::npos) {
            info.name.resize(colon);
        }

        if (info.device_type == kHsaDeviceTypeCpu) {
            if (!have_cpu) {
                rt->cpu_ = info;
                have_cpu = true;
            }
            continue;
        }
        if (info.device_type != kHsaDeviceTypeGpu) continue;

        uint32_t pci_domain = 0, pci_bdfid = 0;
        if (rt->impl_->hsa_agent_get_info(agent, kHsaAgentInfoProfile, &info.profile) != kHsaStatusSuccess ||
            rt->impl_->hsa_agent_get_info(agent, kHsaAgentInfoDefaultFloatRoundingMode, &info.rounding) != kHsaStatusSuccess ||
            rt->impl_->hsa_agent_get_info(agent, kHsaAgentInfoQueueMinSize, &info.queue_min_size) != kHsaStatusSuccess ||
            rt->impl_->hsa_agent_get_info(agent, kHsaAgentInfoQueueMaxSize, &info.queue_max_size) != kHsaStatusSuccess ||
            rt->impl_->hsa_agent_get_info(agent, kHsaAgentInfoQueueType, &info.queue_type) != kHsaStatusSuccess ||
            rt->impl_->hsa_agent_get_info(agent, kHsaAgentInfoWorkgroupMaxDim, info.workgroup_max_dim.data()) != kHsaStatusSuccess ||
            rt->impl_->hsa_agent_get_info(agent, kHsaAgentInfoWorkgroupMaxSize, &info.workgroup_max_size) != kHsaStatusSuccess ||
            rt->impl_->hsa_agent_get_info(agent, kHsaAgentInfoGridMaxDim, info.grid_max_dim.data()) != kHsaStatusSuccess ||
            rt->impl_->hsa_agent_get_info(agent, kHsaAgentInfoGridMaxSize, &info.grid_max_size) != kHsaStatusSuccess ||
            rt->impl_->hsa_agent_get_info(agent, kHsaAmdAgentInfoDomain, &pci_domain) != kHsaStatusSuccess ||
            rt->impl_->hsa_agent_get_info(agent, kHsaAmdAgentInfoBdfid, &pci_bdfid) != kHsaStatusSuccess) {
            err = "hsa_agent_get_info(GPU attributes) failed";
            return nullptr;
        }
        info.pci_bus_id = PciBusId::from_hsa_location(pci_domain, pci_bdfid);
        gpu_agents.push_back(std::move(info));
    }

    if (!have_cpu) {
        err = "ROCr runtime exposed no CPU agent";
        return nullptr;
    }
    if (gpu_agents.empty()) {
        err = "ROCr runtime exposed no GPU agent";
        return nullptr;
    }

    // Match the active HIP device by PCI bus ID first, falling back to ordinal index.
    bool matched_gpu = false;
    if (rt->impl_->hip_device_get_pci_bus_id != nullptr && hip_device_ordinal >= 0) {
        char pci_str[64] = {};
        if (rt->impl_->hip_device_get_pci_bus_id(pci_str, sizeof(pci_str), hip_device_ordinal) == 0) {
            PciBusId target_pci{};
            if (parse_pci_bus_id_string(pci_str, target_pci)) {
                for (const auto& candidate : gpu_agents) {
                    if (candidate.pci_bus_id == target_pci) {
                        rt->gpu_ = candidate;
                        matched_gpu = true;
                        break;
                    }
                }
            }
        }
    }
    if (!matched_gpu) {
        const size_t idx = (hip_device_ordinal >= 0) ? static_cast<size_t>(hip_device_ordinal) : 0u;
        if (idx >= gpu_agents.size()) {
            err = "HIP device ordinal " + std::to_string(hip_device_ordinal) +
                  " is out of range for HSA GPU agents (" + std::to_string(gpu_agents.size()) + ")";
            return nullptr;
        }
        rt->gpu_ = gpu_agents[idx];
    }

    // Discover the CPU-agent fine-grained kernarg memory pool (runtime.rs:1986-2109).
    std::vector<HsaMemoryPool> pools;
    auto collect_pool = +[](HsaMemoryPool pool, void* data) -> HsaStatus {
        static_cast<std::vector<HsaMemoryPool>*>(data)->push_back(pool);
        return kHsaStatusSuccess;
    };
    const HsaStatus pool_st =
        rt->impl_->hsa_amd_agent_iterate_memory_pools(rt->cpu_.handle, collect_pool, &pools);
    if (pool_st != kHsaStatusSuccess) {
        err = "hsa_amd_agent_iterate_memory_pools failed: " + rt->status_string(pool_st);
        return nullptr;
    }

    for (const HsaMemoryPool pool : pools) {
        uint32_t segment = 0, flags = 0;
        bool alloc_allowed = false;
        size_t granule = 0, alignment = 0;
        if (rt->impl_->hsa_amd_memory_pool_get_info(pool, kHsaAmdMemoryPoolInfoSegment, &segment) != kHsaStatusSuccess ||
            segment != kHsaAmdSegmentGlobal) {
            continue;
        }
        if (rt->impl_->hsa_amd_memory_pool_get_info(pool, kHsaAmdMemoryPoolInfoGlobalFlags, &flags) != kHsaStatusSuccess ||
            rt->impl_->hsa_amd_memory_pool_get_info(pool, kHsaAmdMemoryPoolInfoRuntimeAllocAllowed, &alloc_allowed) != kHsaStatusSuccess) {
            continue;
        }
        constexpr uint32_t kRequiredPoolFlags =
            kHsaAmdMemoryPoolGlobalFlagKernargInit | kHsaAmdMemoryPoolGlobalFlagFineGrained;
        if (!alloc_allowed || (flags & kRequiredPoolFlags) != kRequiredPoolFlags) {
            continue;
        }
        if (rt->impl_->hsa_amd_memory_pool_get_info(pool, kHsaAmdMemoryPoolInfoRuntimeAllocGranule, &granule) != kHsaStatusSuccess ||
            rt->impl_->hsa_amd_memory_pool_get_info(pool, kHsaAmdMemoryPoolInfoRuntimeAllocAlignment, &alignment) != kHsaStatusSuccess) {
            continue;
        }
        if (granule == 0u || !is_power_of_two_size(alignment)) {
            continue;
        }
        rt->kernarg_pool_ = pool;
        rt->kernarg_granule_ = granule;
        rt->kernarg_alignment_ = alignment;
        break;
    }

    if (rt->kernarg_pool_.handle == 0) {
        err = "no allocatable fine-grained HSA kernarg memory pool found";
        return nullptr;
    }

    return rt;
}

bool HsaRuntime::create_queue(
    uint32_t requested_size,
    HsaAqlQueue& out_queue,
    std::string& err) const {
    if (gpu_.queue_type != kHsaQueueTypeMulti) {
        err = "HSA GPU agent does not support HSA_QUEUE_TYPE_MULTI";
        return false;
    }
    const uint32_t qsize = (requested_size == 0u)
                               ? std::max(gpu_.queue_min_size, 64u)
                               : requested_size;
    if (!is_power_of_two_u32(qsize) || qsize < gpu_.queue_min_size || qsize > gpu_.queue_max_size) {
        err = "requested HSA queue size " + std::to_string(qsize) + " is outside [" +
              std::to_string(gpu_.queue_min_size) + ", " + std::to_string(gpu_.queue_max_size) + "]";
        return false;
    }

    auto fault = std::make_unique<HsaQueueFaultState>();
    HsaQueue* raw = nullptr;
    const HsaStatus st = impl_->hsa_queue_create(
        gpu_.handle,
        qsize,
        kHsaQueueTypeMulti,
        hsa_queue_error_callback,
        fault.get(),
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<uint32_t>::max(),
        &raw);
    if (st != kHsaStatusSuccess || raw == nullptr) {
        err = "hsa_queue_create failed: " + status_string(st);
        return false;
    }
    if (raw->base_address == nullptr || raw->queue_type != kHsaQueueTypeMulti ||
        raw->size != qsize || (raw->features & kHsaQueueFeatureKernelDispatch) == 0u) {
        (void) impl_->hsa_queue_destroy(raw);
        err = "HSA queue descriptor failed post-creation validation";
        return false;
    }

    out_queue = HsaAqlQueue{};
    out_queue.runtime_ = shared_from_this();
    out_queue.raw_ = raw;
    out_queue.fault_ = std::move(fault);
    out_queue.active_ = true;
    return true;
}

bool HsaRuntime::create_signal(
    HsaCompletionSignal& out_signal,
    std::string& err) const {
    HsaSignal raw = {0};
    const HsaStatus st = impl_->hsa_signal_create(1, 0u, nullptr, &raw);
    if (st != kHsaStatusSuccess || raw.handle == 0) {
        err = "hsa_signal_create failed: " + status_string(st);
        return false;
    }
    out_signal = HsaCompletionSignal{};
    out_signal.runtime_ = shared_from_this();
    out_signal.signal_ = raw;
    return true;
}

bool HsaRuntime::allocate_kernarg(
    size_t bytes,
    size_t alignment,
    HsaBuffer& out_buffer,
    std::string& err) const {
    const size_t req_align = std::max<size_t>(alignment, 16u);
    if (!is_power_of_two_size(req_align)) {
        err = "kernarg alignment must be a power of two";
        return false;
    }
    if (bytes == 0u) {
        out_buffer = HsaBuffer{};
        return true;
    }
    if (bytes > std::numeric_limits<size_t>::max() - (kernarg_granule_ - 1u)) {
        err = "kernarg allocation size overflowed";
        return false;
    }
    const size_t alloc_size =
        ((bytes + kernarg_granule_ - 1u) / kernarg_granule_) * kernarg_granule_;

    void* ptr = nullptr;
    const HsaStatus st = impl_->hsa_amd_memory_pool_allocate(
        kernarg_pool_, alloc_size, kHsaAmdMemoryPoolStandardFlag, &ptr);
    if (st != kHsaStatusSuccess || ptr == nullptr) {
        err = "hsa_amd_memory_pool_allocate(kernarg) failed: " + status_string(st);
        return false;
    }
    const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    if ((addr % req_align) != 0u || (addr % kernarg_alignment_) != 0u) {
        (void) impl_->hsa_amd_memory_pool_free(ptr);
        err = "HSA kernarg pool allocation did not satisfy alignment";
        return false;
    }
    const HsaStatus access_st =
        impl_->hsa_amd_agents_allow_access(1u, &gpu_.handle, nullptr, ptr);
    if (access_st != kHsaStatusSuccess) {
        (void) impl_->hsa_amd_memory_pool_free(ptr);
        err = "hsa_amd_agents_allow_access(kernarg) failed: " + status_string(access_st);
        return false;
    }
    std::memset(ptr, 0, alloc_size);

    out_buffer = HsaBuffer{};
    out_buffer.runtime_ = shared_from_this();
    out_buffer.ptr_ = ptr;
    out_buffer.length_ = bytes;
    out_buffer.allocation_size_ = alloc_size;
    out_buffer.device_local_ = false;
    return true;
}

bool HsaRuntime::allocate_executable_ib(
    size_t bytes,
    HsaBuffer& out_buffer,
    std::string& err) const {
    if (bytes == 0u) {
        err = "PM4 indirect buffer size must be non-zero";
        return false;
    }
    if (bytes > std::numeric_limits<size_t>::max() - (kernarg_granule_ - 1u)) {
        err = "PM4 indirect buffer allocation size overflowed";
        return false;
    }
    const size_t alloc_size =
        ((bytes + kernarg_granule_ - 1u) / kernarg_granule_) * kernarg_granule_;

    void* ptr = nullptr;
    const HsaStatus st = impl_->hsa_amd_memory_pool_allocate(
        kernarg_pool_, alloc_size, kHsaAmdMemoryPoolExecutableFlag, &ptr);
    if (st != kHsaStatusSuccess || ptr == nullptr) {
        err = "hsa_amd_memory_pool_allocate(executable IB) failed: " + status_string(st);
        return false;
    }
    if ((reinterpret_cast<uintptr_t>(ptr) & 15u) != 0u) {
        (void) impl_->hsa_amd_memory_pool_free(ptr);
        err = "PM4 indirect buffer allocation is not 16-byte aligned";
        return false;
    }
    const HsaStatus access_st =
        impl_->hsa_amd_agents_allow_access(1u, &gpu_.handle, nullptr, ptr);
    if (access_st != kHsaStatusSuccess) {
        (void) impl_->hsa_amd_memory_pool_free(ptr);
        err = "hsa_amd_agents_allow_access(executable IB) failed: " + status_string(access_st);
        return false;
    }
    std::memset(ptr, 0, alloc_size);

    out_buffer = HsaBuffer{};
    out_buffer.runtime_ = shared_from_this();
    out_buffer.ptr_ = ptr;
    out_buffer.length_ = bytes;
    out_buffer.allocation_size_ = alloc_size;
    out_buffer.device_local_ = false;
    return true;
}

bool HsaRuntime::register_memory(void* ptr, size_t bytes, std::string& err) const {
    if (ptr == nullptr || bytes == 0u) {
        err = "cannot register empty or null memory range";
        return false;
    }
    const HsaStatus st = impl_->hsa_memory_register(ptr, bytes);
    if (st != kHsaStatusSuccess) {
        err = "hsa_memory_register failed: " + status_string(st);
        return false;
    }
    return true;
}

bool HsaRuntime::deregister_memory(void* ptr, std::string& err) const {
    if (ptr == nullptr) return true;
    const HsaStatus st = impl_->hsa_memory_deregister(ptr, 0u);
    if (st != kHsaStatusSuccess) {
        err = "hsa_memory_deregister failed: " + status_string(st);
        return false;
    }
    return true;
}

bool HsaRuntime::query_symbol_and_pm4_metadata(
    HsaExecutable exec,
    const std::string& base_name,
    const std::string& kd_name,
    HsaResolvedKernel& out,
    std::string& err) const {
    HsaExecutableSymbol sym = {0};
    HsaStatus st =
        impl_->hsa_executable_get_symbol_by_name(exec, kd_name.c_str(), &gpu_.handle, &sym);
    if (st != kHsaStatusSuccess || sym.handle == 0) {
        return false;
    }

    uint32_t kind = 0;
    if (impl_->hsa_executable_symbol_get_info(sym, kHsaExecutableSymbolInfoType, &kind) != kHsaStatusSuccess ||
        kind != kHsaSymbolKindKernel) {
        err = "HSA symbol " + kd_name + " is not a kernel symbol";
        return false;
    }

    pm4::KernelMetadata meta{};
    if (impl_->hsa_executable_symbol_get_info(sym, kHsaExecutableSymbolInfoKernelObject, &meta.kernel_object) != kHsaStatusSuccess ||
        impl_->hsa_executable_symbol_get_info(sym, kHsaExecutableSymbolInfoKernelKernargSegmentSize, &meta.kernarg_segment_size) != kHsaStatusSuccess ||
        impl_->hsa_executable_symbol_get_info(sym, kHsaExecutableSymbolInfoKernelKernargSegmentAlignment, &meta.kernarg_segment_alignment) != kHsaStatusSuccess ||
        impl_->hsa_executable_symbol_get_info(sym, kHsaExecutableSymbolInfoKernelGroupSegmentSize, &meta.group_segment_size) != kHsaStatusSuccess ||
        impl_->hsa_executable_symbol_get_info(sym, kHsaExecutableSymbolInfoKernelPrivateSegmentSize, &meta.private_segment_size) != kHsaStatusSuccess ||
        impl_->hsa_executable_symbol_get_info(sym, kHsaExecutableSymbolInfoKernelDynamicCallstack, &meta.dynamic_callstack) != kHsaStatusSuccess ||
        meta.kernel_object == 0u) {
        err = "hsa_executable_symbol_get_info failed for " + kd_name;
        return false;
    }

    out = HsaResolvedKernel{};
    out.symbol_name = base_name;
    out.kd_name = kd_name;
    out.executable = exec;
    out.loader = meta;

    // Verify owning executable / LOAD_BASE..LOAD_SIZE containment when loader supports it.
    if (impl_->loader.query_executable != nullptr) {
        HsaExecutable owner = {0};
        if (impl_->loader.query_executable(
                reinterpret_cast<const void*>(meta.kernel_object), &owner) == kHsaStatusSuccess &&
            owner.handle != 0 && owner.handle != exec.handle) {
            err = "kernel_object owning executable mismatch for " + kd_name;
            return false;
        }
    }

    auto validate_and_parse_kd = [&](const uint8_t* kd_bytes) -> bool {
        uint32_t kd_group = 0, kd_priv = 0;
        pm4::KernelPm4Metadata pm4_meta{};
        if (!pm4::parse_kernel_descriptor_bytes(
                kd_bytes, 64u, meta.kernel_object, pm4_meta, &kd_group, &kd_priv)) {
            return false;
        }
        if (kd_group != meta.group_segment_size || kd_priv != meta.private_segment_size) {
            return false;
        }
        if ((pm4_meta.code_entry & 0xffu) != 0u) {
            return false;
        }
        out.pm4 = pm4_meta;
        return true;
    };

    // Path 1 (Primary): Validated CPU alias from hsa_amd_pointer_info
    // (ported from hipfire/crates/rdna-compute/src/loaded_code_check.rs:506-542).
    {
        HsaAmdPointerInfo ptr_info{};
        ptr_info.size = static_cast<uint32_t>(sizeof(HsaAmdPointerInfo));
        const HsaStatus pi_st = impl_->hsa_amd_pointer_info(
            reinterpret_cast<const void*>(meta.kernel_object),
            &ptr_info,
            nullptr,
            nullptr,
            nullptr);
        if (pi_st == kHsaStatusSuccess &&
            (ptr_info.ptr_type == kHsaPointerTypeHsa || ptr_info.ptr_type == kHsaPointerTypeHsaVmem) &&
            ptr_info.agent_base_address != nullptr &&
            ptr_info.host_base_address != nullptr) {
            const bool host_access_ok =
                (ptr_info.size < offsetof(HsaAmdPointerInfo, alloc_flags) + sizeof(uint32_t)) ||
                (ptr_info.alloc_flags == 0u) ||
                ((ptr_info.alloc_flags & kHsaPointerInfoAllocFlagHostAccess) != 0u);
            const uint64_t agent_base = reinterpret_cast<uint64_t>(ptr_info.agent_base_address);
            const uint64_t host_base = reinterpret_cast<uint64_t>(ptr_info.host_base_address);
            if (host_access_ok && meta.kernel_object >= agent_base) {
                const uint64_t offset = meta.kernel_object - agent_base;
                if (offset <= ptr_info.size_in_bytes && ptr_info.size_in_bytes - offset >= 64u) {
                    const auto* host_kd = reinterpret_cast<const uint8_t*>(host_base + offset);
                    if (validate_and_parse_kd(host_kd)) {
                        return true;
                    }
                }
            }
        }
    }

    // Path 2 (Fallback A): Loader host-address query for non-CPU-mapped device code objects.
    if (impl_->loader.query_host_address != nullptr) {
        const void* host_kd = nullptr;
        if (impl_->loader.query_host_address(
                reinterpret_cast<const void*>(meta.kernel_object), &host_kd) == kHsaStatusSuccess &&
            host_kd != nullptr) {
            if (validate_and_parse_kd(static_cast<const uint8_t*>(host_kd))) {
                return true;
            }
        }
    }

    // Path 3 (Fallback B): Parse in-memory ELF bytes from the loaded code object's storage.
    std::vector<HsaLoadedCodeObject> lcos;
    auto collect_lco = +[](HsaExecutable /*e*/, HsaLoadedCodeObject lco, void* data) -> HsaStatus {
        static_cast<std::vector<HsaLoadedCodeObject>*>(data)->push_back(lco);
        return kHsaStatusSuccess;
    };
    if (impl_->loader.executable_iterate_loaded_code_objects(exec, collect_lco, &lcos) == kHsaStatusSuccess) {
        for (const HsaLoadedCodeObject lco : lcos) {
            uint32_t storage_type = 0;
            const void* mem_base = nullptr;
            size_t mem_size = 0;
            if (impl_->loader.loaded_code_object_get_info(
                    lco, kHsaVenAmdLoaderLoadedCodeObjectInfoStorageType, &storage_type) == kHsaStatusSuccess &&
                storage_type == kHsaVenAmdLoaderCodeObjectStorageTypeMemory &&
                impl_->loader.loaded_code_object_get_info(
                    lco, kHsaVenAmdLoaderLoadedCodeObjectInfoMemoryBase, &mem_base) == kHsaStatusSuccess &&
                impl_->loader.loaded_code_object_get_info(
                    lco, kHsaVenAmdLoaderLoadedCodeObjectInfoMemorySize, &mem_size) == kHsaStatusSuccess &&
                mem_base != nullptr && mem_size >= 64u) {
                pm4::KernelPm4Metadata pm4_meta{};
                if (pm4::parse_kernel_pm4_metadata_from_elf(
                        static_cast<const uint8_t*>(mem_base),
                        mem_size,
                        kd_name.c_str(),
                        meta.kernel_object,
                        pm4_meta) &&
                    (pm4_meta.code_entry & 0xffu) == 0u) {
                    out.pm4 = pm4_meta;
                    return true;
                }
            }
        }
    }

    err = "resolved HSA symbol " + kd_name +
          " but could not read/validate its 64-byte .kd PM4 descriptor";
    return false;
}

bool HsaRuntime::resolve_hip_kernel(
    const void* host_function,
    HsaResolvedKernel& out,
    std::string& err,
    const char* explicit_symbol_name,
    const void* module_device_ptr) {
    {
        std::lock_guard<std::mutex> lock(kernel_cache_mu_);
        if (host_function != nullptr) {
            const auto it = kernel_by_ptr_.find(host_function);
            if (it != kernel_by_ptr_.end()) {
                out = it->second;
                return true;
            }
        }
        if (explicit_symbol_name != nullptr && explicit_symbol_name[0] != '\0') {
            const auto it = kernel_by_name_.find(explicit_symbol_name);
            if (it != kernel_by_name_.end()) {
                out = it->second;
                return true;
            }
        }
    }

    // 1. Trigger ROCclr's lazy fatbinary loading for `host_function` on the active HIP device.
    if (host_function != nullptr && impl_->hip_func_get_attributes != nullptr) {
        alignas(8) uint8_t attr_storage[128] = {};
        (void) impl_->hip_func_get_attributes(attr_storage, host_function);
    }

    // 2. Resolve the mangled kernel symbol name from the HIP runtime or dynamic symbol table.
    std::string symbol_name;
    if (explicit_symbol_name != nullptr && explicit_symbol_name[0] != '\0') {
        symbol_name = explicit_symbol_name;
    } else if (host_function != nullptr) {
        if (impl_->hip_kernel_name_ref_by_ptr != nullptr) {
            const char* ref = impl_->hip_kernel_name_ref_by_ptr(host_function, nullptr);
            if (ref != nullptr && ref[0] != '\0') {
                symbol_name = ref;
            }
        }
        if (symbol_name.empty() &&
            impl_->hip_get_func_by_symbol != nullptr &&
            impl_->hip_kernel_name_ref != nullptr) {
            void* hip_fn = nullptr;
            if (impl_->hip_get_func_by_symbol(&hip_fn, host_function) == 0 && hip_fn != nullptr) {
                const char* ref = impl_->hip_kernel_name_ref(hip_fn);
                if (ref != nullptr && ref[0] != '\0') {
                    symbol_name = ref;
                }
            }
        }
        if (symbol_name.empty()) {
            Dl_info dl_info{};
            if (dladdr(host_function, &dl_info) != 0 &&
                dl_info.dli_sname != nullptr &&
                dl_info.dli_sname[0] != '\0') {
                symbol_name = dl_info.dli_sname;
            }
        }
    }

    if (symbol_name.empty()) {
        err = "could not resolve device kernel symbol name from HIP host function pointer";
        return false;
    }

    std::string base_name = symbol_name;
    if (base_name.size() > 3u && base_name.compare(base_name.size() - 3u, 3u, ".kd") == 0) {
        base_name.resize(base_name.size() - 3u);
    }
    const std::string kd_name = base_name + ".kd";

    // 3. If a device pointer from the owning module is supplied, query its executable directly first.
    if (module_device_ptr != nullptr && impl_->loader.query_executable != nullptr) {
        HsaExecutable direct_exec = {0};
        if (impl_->loader.query_executable(module_device_ptr, &direct_exec) == kHsaStatusSuccess &&
            direct_exec.handle != 0) {
            uint32_t state = 0;
            if (impl_->hsa_executable_get_info(direct_exec, kHsaExecutableInfoState, &state) == kHsaStatusSuccess &&
                state == kHsaExecutableStateFrozen &&
                query_symbol_and_pm4_metadata(direct_exec, base_name, kd_name, out, err)) {
                std::lock_guard<std::mutex> lock(kernel_cache_mu_);
                if (host_function != nullptr) kernel_by_ptr_[host_function] = out;
                kernel_by_name_[base_name] = out;
                return true;
            }
        }
    }

    // 4. Enumerate all frozen HSA executables in-process (including HIP-loaded fatbinaries).
    std::vector<HsaExecutable> execs;
    auto collect_exec = +[](HsaExecutable exec, void* data) -> HsaStatus {
        static_cast<std::vector<HsaExecutable>*>(data)->push_back(exec);
        return kHsaStatusSuccess;
    };
    const HsaStatus iter_st = impl_->loader.iterate_executables(collect_exec, &execs);
    if (iter_st != kHsaStatusSuccess) {
        err = "hsa_ven_amd_loader_iterate_executables failed: " + status_string(iter_st);
        return false;
    }

    std::string last_err;
    for (const HsaExecutable exec : execs) {
        uint32_t state = 0;
        if (impl_->hsa_executable_get_info(exec, kHsaExecutableInfoState, &state) != kHsaStatusSuccess ||
            state != kHsaExecutableStateFrozen) {
            continue;
        }
        std::string sym_err;
        if (query_symbol_and_pm4_metadata(exec, base_name, kd_name, out, sym_err)) {
            std::lock_guard<std::mutex> lock(kernel_cache_mu_);
            if (host_function != nullptr) kernel_by_ptr_[host_function] = out;
            kernel_by_name_[base_name] = out;
            return true;
        }
        if (!sym_err.empty()) {
            last_err = sym_err;
        }
    }

    err = last_err.empty()
              ? ("kernel symbol " + kd_name + " not found in any frozen HSA executable")
              : last_err;
    return false;
}

bool HsaRuntime::load_executable_from_memory(
    const uint8_t* bytes,
    size_t len,
    HsaStandaloneExecutable& out_exec,
    std::string& err) const {
    size_t elf_offset = 0, elf_size = 0;
    if (!pm4::unwrap_clang_offload_bundle(bytes, len, elf_offset, elf_size, err)) {
        return false;
    }
    if (elf_size == 0u) {
        err = "unwrapped AMDGPU ELF is empty";
        return false;
    }

    HsaStandaloneExecutable result{};
    result.runtime_ = shared_from_this();
    result.gpu_agent_ = gpu_.handle;
    result.code_object_bytes_.assign(bytes + elf_offset, bytes + elf_offset + elf_size);

    const HsaStatus rdr_st = impl_->hsa_code_object_reader_create_from_memory(
        result.code_object_bytes_.data(),
        result.code_object_bytes_.size(),
        &result.reader_);
    if (rdr_st != kHsaStatusSuccess) {
        err = "hsa_code_object_reader_create_from_memory failed: " + status_string(rdr_st);
        return false;
    }

    const HsaStatus ex_st = impl_->hsa_executable_create_alt(
        gpu_.profile, gpu_.rounding, nullptr, &result.executable_);
    if (ex_st != kHsaStatusSuccess) {
        err = "hsa_executable_create_alt failed: " + status_string(ex_st);
        return false;
    }

    HsaLoadedCodeObject loaded = {0};
    const HsaStatus load_st = impl_->hsa_executable_load_agent_code_object(
        result.executable_, gpu_.handle, result.reader_, nullptr, &loaded);
    if (load_st != kHsaStatusSuccess) {
        err = "hsa_executable_load_agent_code_object failed: " + status_string(load_st);
        return false;
    }

    const HsaStatus frz_st = impl_->hsa_executable_freeze(result.executable_, nullptr);
    if (frz_st != kHsaStatusSuccess) {
        err = "hsa_executable_freeze failed: " + status_string(frz_st);
        return false;
    }

    out_exec = std::move(result);
    return true;
}

}  // namespace strata::hip::hsa

#endif  // defined(__linux__) && defined(STRATA_WITH_HSA)
