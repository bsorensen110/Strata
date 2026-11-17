// include/strata/hip/hsa.hpp - Dynamic ROCr/HSA runtime shim for Linux GFX12 PM4-IB replay.
//
// Port note: from warpfront/hipfire redline-rocr (Apache-2.0)
// Ported from warpfront/hipfire crates/redline-rocr/src/{abi.rs,runtime.rs,lib.rs},
// crates/hsa-bridge/src/{ffi.rs,lib.rs}, and crates/rdna-compute/src/loaded_code_check.rs.
#pragma once

#if defined(__linux__) && defined(STRATA_WITH_HSA)

#include "strata/hip/pm4.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::hip::hsa {

// ---- Public ROCr/HSA ABI types and constants (ROCm 7.2 hsa.h / hsa_ext_amd.h / hsa_ven_amd_loader.h) ----
// Port note: from warpfront/hipfire redline-rocr (Apache-2.0)

using HsaStatus = int32_t;
using HsaSignalValue = int64_t;

inline constexpr HsaStatus kHsaStatusSuccess = 0;
inline constexpr HsaStatus kHsaStatusInfoBreak = 1;

inline constexpr uint32_t kHsaSystemInfoTimestampFrequency = 3u;

inline constexpr uint32_t kHsaDeviceTypeCpu = 0u;
inline constexpr uint32_t kHsaDeviceTypeGpu = 1u;

inline constexpr uint32_t kHsaAgentInfoName = 0u;
inline constexpr uint32_t kHsaAgentInfoProfile = 4u;
inline constexpr uint32_t kHsaAgentInfoDefaultFloatRoundingMode = 5u;
inline constexpr uint32_t kHsaAgentInfoWorkgroupMaxDim = 7u;
inline constexpr uint32_t kHsaAgentInfoWorkgroupMaxSize = 8u;
inline constexpr uint32_t kHsaAgentInfoGridMaxDim = 9u;
inline constexpr uint32_t kHsaAgentInfoGridMaxSize = 10u;
inline constexpr uint32_t kHsaAgentInfoQueueMinSize = 13u;
inline constexpr uint32_t kHsaAgentInfoQueueMaxSize = 14u;
inline constexpr uint32_t kHsaAgentInfoQueueType = 15u;
inline constexpr uint32_t kHsaAgentInfoDevice = 17u;

inline constexpr uint32_t kHsaAmdAgentInfoComputeUnitCount = 0xA002u;
inline constexpr uint32_t kHsaAmdAgentInfoBdfid = 0xA006u;
inline constexpr uint32_t kHsaAmdAgentInfoDomain = 0xA00Fu;
inline constexpr uint32_t kHsaAmdAgentInfoTimestampFrequency = 0xA016u;

inline constexpr uint32_t kHsaQueueTypeMulti = 0u;
inline constexpr uint32_t kHsaQueueTypeSingle = 1u;
inline constexpr uint32_t kHsaQueueFeatureKernelDispatch = 1u;

inline constexpr uint32_t kHsaSignalConditionEq = 0u;
inline constexpr uint32_t kHsaSignalConditionNe = 1u;
inline constexpr uint32_t kHsaSignalConditionLt = 2u;
inline constexpr uint32_t kHsaSignalConditionGte = 3u;

inline constexpr uint32_t kHsaWaitStateBlocked = 0u;
inline constexpr uint32_t kHsaWaitStateActive = 1u;

inline constexpr uint32_t kHsaAmdSegmentGlobal = 0u;
inline constexpr uint32_t kHsaAmdMemoryPoolGlobalFlagKernargInit = 1u;
inline constexpr uint32_t kHsaAmdMemoryPoolGlobalFlagFineGrained = 2u;
inline constexpr uint32_t kHsaAmdMemoryPoolGlobalFlagCoarseGrained = 4u;

inline constexpr uint32_t kHsaAmdMemoryPoolInfoSegment = 0u;
inline constexpr uint32_t kHsaAmdMemoryPoolInfoGlobalFlags = 1u;
inline constexpr uint32_t kHsaAmdMemoryPoolInfoSize = 2u;
inline constexpr uint32_t kHsaAmdMemoryPoolInfoRuntimeAllocAllowed = 5u;
inline constexpr uint32_t kHsaAmdMemoryPoolInfoRuntimeAllocGranule = 6u;
inline constexpr uint32_t kHsaAmdMemoryPoolInfoRuntimeAllocAlignment = 7u;

inline constexpr uint32_t kHsaAmdMemoryPoolStandardFlag = 0u;
inline constexpr uint32_t kHsaAmdMemoryPoolExecutableFlag = 1u << 2;

inline constexpr uint32_t kHsaAmdAgentMemoryPoolInfoAccess = 0u;
inline constexpr uint32_t kHsaAmdMemoryPoolAccessNeverAllowed = 0u;
inline constexpr uint32_t kHsaAmdMemoryPoolAccessAllowedByDefault = 1u;
inline constexpr uint32_t kHsaAmdMemoryPoolAccessDisallowedByDefault = 2u;

inline constexpr uint32_t kHsaExecutableInfoState = 2u;
inline constexpr uint32_t kHsaExecutableStateFrozen = 1u;

inline constexpr uint32_t kHsaExecutableSymbolInfoType = 0u;
inline constexpr uint32_t kHsaSymbolKindKernel = 1u;
inline constexpr uint32_t kHsaExecutableSymbolInfoKernelKernargSegmentSize = 11u;
inline constexpr uint32_t kHsaExecutableSymbolInfoKernelKernargSegmentAlignment = 12u;
inline constexpr uint32_t kHsaExecutableSymbolInfoKernelGroupSegmentSize = 13u;
inline constexpr uint32_t kHsaExecutableSymbolInfoKernelPrivateSegmentSize = 14u;
inline constexpr uint32_t kHsaExecutableSymbolInfoKernelDynamicCallstack = 15u;
inline constexpr uint32_t kHsaExecutableSymbolInfoKernelObject = 22u;

// AMD Loader extension (hsa_ven_amd_loader.h, loaded_code_check.rs:994-1003)
inline constexpr uint16_t kHsaExtensionAmdLoader = 0x0201u;
inline constexpr uint32_t kHsaVenAmdLoaderCodeObjectStorageTypeMemory = 2u;
inline constexpr uint32_t kHsaVenAmdLoaderLoadedCodeObjectInfoStorageType = 2u;
inline constexpr uint32_t kHsaVenAmdLoaderLoadedCodeObjectInfoMemoryBase = 3u;
inline constexpr uint32_t kHsaVenAmdLoaderLoadedCodeObjectInfoMemorySize = 4u;
inline constexpr uint32_t kHsaVenAmdLoaderLoadedCodeObjectInfoLoadDelta = 8u;
inline constexpr uint32_t kHsaVenAmdLoaderLoadedCodeObjectInfoLoadBase = 9u;
inline constexpr uint32_t kHsaVenAmdLoaderLoadedCodeObjectInfoLoadSize = 10u;

// hsa_amd_pointer_info_t constants (hsa_ext_amd.h:3019-3076, loaded_code_check.rs:1000-1002)
inline constexpr uint32_t kHsaPointerTypeHsa = 1u;
inline constexpr uint32_t kHsaPointerTypeHsaVmem = 6u;
inline constexpr uint32_t kHsaPointerInfoAllocFlagHostAccess = 1u << 4;

// Default finite host-side completion and capacity timeout (5 seconds, runtime.rs:24).
inline constexpr uint64_t kDefaultWaitTimeoutNs = 5'000'000'000ULL;

struct HsaAgent { uint64_t handle = 0; };
struct HsaSignal { uint64_t handle = 0; };
struct HsaMemoryPool { uint64_t handle = 0; };
struct HsaCodeObjectReader { uint64_t handle = 0; };
struct HsaExecutable { uint64_t handle = 0; };
struct HsaLoadedCodeObject { uint64_t handle = 0; };
struct HsaExecutableSymbol { uint64_t handle = 0; };

// Public hsa_queue_t layout for 64-bit little-endian HSA hosts (abi.rs:128-137).
struct alignas(8) HsaQueue {
    uint32_t queue_type = 0;
    uint32_t features = 0;
    void* base_address = nullptr;
    HsaSignal doorbell_signal = {0};
    uint32_t size = 0;
    uint32_t reserved1 = 0;
    uint64_t id = 0;
};

static_assert(sizeof(HsaQueue) == 40, "HsaQueue must match 40-byte hsa_queue_t ABI");
static_assert(alignof(HsaQueue) == 8, "HsaQueue alignment must be 8 bytes");

// Public hsa_amd_pointer_info_t layout (hsa_ext_amd.h:3096-3151, loaded_code_check.rs:1024-1035).
struct HsaAmdPointerInfo {
    uint32_t size = 0;
    uint32_t ptr_type = 0;
    void* agent_base_address = nullptr;
    void* host_base_address = nullptr;
    size_t size_in_bytes = 0;
    void* user_data = nullptr;
    HsaAgent agent_owner = {0};
    uint32_t global_flags = 0;
    bool registered = false;
    uint32_t alloc_flags = 0;
};

// Public hsa_ven_amd_loader_1_03_pfn_t layout (hsa_ven_amd_loader.h:622-661, loaded_code_check.rs:1060-1068).
struct HsaVenAmdLoaderTable103 {
    HsaStatus (*query_host_address)(const void* device_address, const void** host_address) = nullptr;
    void* query_segment_descriptors = nullptr;
    HsaStatus (*query_executable)(const void* device_address, HsaExecutable* executable) = nullptr;
    HsaStatus (*executable_iterate_loaded_code_objects)(
        HsaExecutable executable,
        HsaStatus (*callback)(HsaExecutable, HsaLoadedCodeObject, void*),
        void* data) = nullptr;
    HsaStatus (*loaded_code_object_get_info)(
        HsaLoadedCodeObject loaded_code_object,
        uint32_t attribute,
        void* value) = nullptr;
    void* reader_create_from_file_with_offset_size = nullptr;
    HsaStatus (*iterate_executables)(
        HsaStatus (*callback)(HsaExecutable, void*),
        void* data) = nullptr;
};

// Normalized PCI domain:bus:device.function identity (runtime.rs:384-420).
struct PciBusId {
    uint32_t domain = 0;
    uint8_t bus = 0;
    uint8_t device = 0;
    uint8_t function = 0;

    static PciBusId from_hsa_location(uint32_t domain, uint32_t bdfid) {
        return PciBusId{
            domain,
            static_cast<uint8_t>((bdfid >> 8) & 0xffu),
            static_cast<uint8_t>((bdfid >> 3) & 0x1fu),
            static_cast<uint8_t>(bdfid & 0x7u),
        };
    }

    bool operator==(const PciBusId& o) const {
        return domain == o.domain && bus == o.bus && device == o.device && function == o.function;
    }
    bool operator!=(const PciBusId& o) const { return !(*this == o); }
};

struct HsaAgentInfo {
    HsaAgent handle = {0};
    std::string name;
    uint32_t device_type = 0;
    uint32_t profile = 0;
    uint32_t rounding = 0;
    uint32_t queue_min_size = 0;
    uint32_t queue_max_size = 0;
    uint32_t queue_type = 0;
    std::array<uint16_t, 3> workgroup_max_dim = {0, 0, 0};
    uint32_t workgroup_max_size = 0;
    std::array<uint32_t, 3> grid_max_dim = {0, 0, 0};
    uint32_t grid_max_size = 0;
    PciBusId pci_bus_id = {};
};

// Resolved kernel metadata for PM4 direct dispatch on a HIP-loaded or HSA-loaded code object.
struct HsaResolvedKernel {
    std::string symbol_name;  // Mangled kernel name without ".kd"
    std::string kd_name;      // Symbol name with ".kd" suffix
    HsaExecutable executable = {0};
    pm4::KernelMetadata loader = {};
    pm4::KernelPm4Metadata pm4 = {};
};

class HsaRuntime;

// Fine-grained or host-writable device-local HSA memory allocation (kernarg arena / PM4 IB).
class HsaBuffer {
public:
    HsaBuffer() = default;
    ~HsaBuffer();

    HsaBuffer(const HsaBuffer&) = delete;
    HsaBuffer& operator=(const HsaBuffer&) = delete;
    HsaBuffer(HsaBuffer&& other) noexcept;
    HsaBuffer& operator=(HsaBuffer&& other) noexcept;

    void* address() const { return ptr_; }
    uint8_t* bytes() const { return static_cast<uint8_t*>(ptr_); }
    uint64_t gpu_address() const { return reinterpret_cast<uint64_t>(ptr_); }
    size_t size() const { return length_; }
    size_t allocation_size() const { return allocation_size_; }
    bool is_device_local() const { return device_local_; }
    bool valid() const { return ptr_ != nullptr; }

    // Write exact bytes into the buffer; returns false on size mismatch.
    bool write_exact(const void* src, size_t n);

    // Drain write-combining buffers with a 1-byte BAR readback when device_local_ is true (runtime.rs:2757).
    void publish_host_writes() const;

    void reset();

private:
    friend class HsaRuntime;
    std::shared_ptr<const HsaRuntime> runtime_;
    void* ptr_ = nullptr;
    size_t length_ = 0;
    size_t allocation_size_ = 0;
    bool device_local_ = false;
};

// Persistent HSA completion signal initialized to 1 and decremented to 0 by AQL packet completion.
class HsaCompletionSignal {
public:
    HsaCompletionSignal() = default;
    ~HsaCompletionSignal();

    HsaCompletionSignal(const HsaCompletionSignal&) = delete;
    HsaCompletionSignal& operator=(const HsaCompletionSignal&) = delete;
    HsaCompletionSignal(HsaCompletionSignal&& other) noexcept;
    HsaCompletionSignal& operator=(HsaCompletionSignal&& other) noexcept;

    HsaSignal raw() const { return signal_; }
    uint64_t handle() const { return signal_.handle; }
    bool valid() const { return signal_.handle != 0; }

    // Reset signal to 1 with screlease ordering before submitting a new packet.
    void reset();

    // True when hsa_signal_load_scacquire(signal_) < 1.
    bool is_complete() const;

    // System-scope acquire load of the signal value.
    HsaSignalValue value_scacquire() const;

    // Poll for completion up to timeout_ns.
    bool wait_timeout(uint64_t timeout_ns, std::string& err) const;

private:
    friend class HsaRuntime;
    std::shared_ptr<const HsaRuntime> runtime_;
    HsaSignal signal_ = {0};
};

struct HsaQueueFaultState {
    std::atomic<uint64_t> encoded{0};

    void record(HsaStatus status) noexcept {
        const uint64_t val = (1ULL << 63) | static_cast<uint64_t>(static_cast<uint32_t>(status));
        uint64_t expected = 0;
        (void) encoded.compare_exchange_strong(
            expected, val, std::memory_order_release, std::memory_order_relaxed);
    }

    bool has_fault(HsaStatus& out_status) const noexcept {
        const uint64_t val = encoded.load(std::memory_order_acquire);
        if (val == 0) return false;
        out_status = static_cast<HsaStatus>(static_cast<uint32_t>(val));
        return true;
    }
};

// Distinct user-mode HSA MULTI queue for retained PM4-IB submission.
class HsaAqlQueue {
public:
    HsaAqlQueue() = default;
    ~HsaAqlQueue();

    HsaAqlQueue(const HsaAqlQueue&) = delete;
    HsaAqlQueue& operator=(const HsaAqlQueue&) = delete;
    HsaAqlQueue(HsaAqlQueue&& other) noexcept;
    HsaAqlQueue& operator=(HsaAqlQueue&& other) noexcept;

    bool valid() const { return raw_ != nullptr && active_; }
    uint64_t id() const { return raw_ ? raw_->id : 0; }
    uint32_t size() const { return raw_ ? raw_->size : 0; }

    bool check_fault(std::string& err) const;
    bool inactivate(std::string& err);

    // Reserve slots, copy packet payload bytes[4..64], release-store header_word at bytes[0..4],
    // and ring the queue doorbell with hsa_signal_store_relaxed.
    bool submit_packets(
        const pm4::PacketImage* packets,
        size_t packet_count,
        uint64_t timeout_ns,
        std::string& err);

    // Wait for `signal` to reach < 1 while checking this queue for asynchronous faults.
    // On fault or timeout, inactivates the queue before returning false.
    bool wait_signal(
        const HsaCompletionSignal& signal,
        uint64_t timeout_ns,
        std::string& err);

    // hipStreamSynchronize-compatible PM4-IB submission and completion wait:
    //
    // Why host-side signal polling instead of hipStreamWaitValue64:
    //   1. hipStreamWaitValue64 enqueues a WAIT_REG_MEM packet onto a HIP stream against a memory
    //      address (not an hsa_signal_t) and requires hipStreamSynchronize, which re-incurs the
    //      ~22.1 us HIP host-sync driver tax on gfx1201 and holds a MEC queue slot spinning.
    //   2. The vendor PM4-IB AQL packet carries acquire=System, release=System, barrier=1, and
    //      decrements `completion` to 0 after system-scope cache release completes. Polling
    //      `completion` via hsa_signal_load_scacquire directly proves system-scope completion for
    //      both host consumers and subsequent HIP stream work without touching HIP's stream layer.
    bool submit_and_wait_pm4_ib(
        const HsaBuffer& ib_buffer,
        uint32_t ib_dwords,
        HsaCompletionSignal& completion,
        uint64_t timeout_ns,
        std::string& err);

private:
    friend class HsaRuntime;
    bool wait_for_capacity(uint64_t packet_count, uint64_t timeout_ns, std::string& err) const;

    std::shared_ptr<const HsaRuntime> runtime_;
    HsaQueue* raw_ = nullptr;
    std::unique_ptr<HsaQueueFaultState> fault_;
    bool active_ = false;
};

// Standalone HSA executable loaded from raw .hsaco / clang-offload-bundle bytes (fallback path).
class HsaStandaloneExecutable {
public:
    HsaStandaloneExecutable() = default;
    ~HsaStandaloneExecutable();

    HsaStandaloneExecutable(const HsaStandaloneExecutable&) = delete;
    HsaStandaloneExecutable& operator=(const HsaStandaloneExecutable&) = delete;
    HsaStandaloneExecutable(HsaStandaloneExecutable&& other) noexcept;
    HsaStandaloneExecutable& operator=(HsaStandaloneExecutable&& other) noexcept;

    bool valid() const { return executable_.handle != 0; }
    bool resolve_kernel(
        const char* symbol_name,
        HsaResolvedKernel& out,
        std::string& err) const;

private:
    friend class HsaRuntime;
    std::shared_ptr<const HsaRuntime> runtime_;
    HsaAgent gpu_agent_ = {0};
    HsaExecutable executable_ = {0};
    HsaCodeObjectReader reader_ = {0};
    std::vector<uint8_t> code_object_bytes_;
};

// Dynamically loaded ROCr (`libhsa-runtime64.so.1`) runtime shim.
class HsaRuntime : public std::enable_shared_from_this<HsaRuntime> {
public:
    ~HsaRuntime();

    HsaRuntime(const HsaRuntime&) = delete;
    HsaRuntime& operator=(const HsaRuntime&) = delete;

    // Dlopen libhsa-runtime64.so.1, resolve all required symbols and the AMD loader extension,
    // verify ROC_GLOBAL_CU_MASK / HSA_CU_MASK is unset, call hsa_init(), and discover agents/pools.
    // Fail-closed: any missing symbol or unsupported extension returns nullptr and populates `err`.
    static std::shared_ptr<HsaRuntime> initialize(
        int hip_device_ordinal,
        std::string& err);

    const HsaAgentInfo& gpu_agent() const { return gpu_; }
    const HsaAgentInfo& cpu_agent() const { return cpu_; }
    bool is_gfx12() const {
        return gpu_.name == "gfx1200" || gpu_.name == "gfx1201";
    }

    // Create a user-mode HSA MULTI queue on the selected GPU agent.
    // If requested_size == 0, defaults to gpu_agent().queue_min_size (or 64).
    bool create_queue(
        uint32_t requested_size,
        HsaAqlQueue& out_queue,
        std::string& err) const;

    // Create a completion signal initialized to 1.
    bool create_signal(
        HsaCompletionSignal& out_signal,
        std::string& err) const;

    // Allocate fine-grained CPU/GPU-accessible kernarg memory (zero-initialized).
    bool allocate_kernarg(
        size_t bytes,
        size_t alignment,
        HsaBuffer& out_buffer,
        std::string& err) const;

    // Allocate executable command memory for a PM4 indirect buffer (HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG).
    bool allocate_executable_ib(
        size_t bytes,
        HsaBuffer& out_buffer,
        std::string& err) const;

    // Register / deregister existing host memory with ROCr.
    bool register_memory(void* ptr, size_t bytes, std::string& err) const;
    bool deregister_memory(void* ptr, std::string& err) const;

    // Interrogate the default (HIP-loaded) code object for a kernel without loading .hsaco outside HIP:
    //   1. Calls hipFuncGetAttributes(host_function) to ensure ROCclr has lazily loaded and frozen
    //      the owning fatbinary's hsa_executable_t.
    //   2. Resolves the mangled kernel symbol via explicit_symbol_name, hipKernelNameRefByPtr,
    //      hipGetFuncBySymbol + hipKernelNameRef, or dladdr, and forms "<symbol>.kd".
    //   3. Locates the frozen hsa_executable_t (via hsa_ven_amd_loader_query_executable when
    //      module_device_ptr is supplied, or hsa_ven_amd_loader_iterate_executables) and queries
    //      KernelMetadata via hsa_executable_get_symbol_by_name + hsa_executable_symbol_get_info.
    //   4. Reads the 64-byte AMDGPU kernel_descriptor_t at `kernel_object` via a validated
    //      hsa_amd_pointer_info CPU alias (with hsa_ven_amd_loader_query_host_address and
    //      loaded_code_object in-memory ELF fallbacks) and cross-checks group/private segment
    //      sizes and 256-byte code_entry alignment.
    bool resolve_hip_kernel(
        const void* host_function,
        HsaResolvedKernel& out,
        std::string& err,
        const char* explicit_symbol_name = nullptr,
        const void* module_device_ptr = nullptr);

    // Fallback: load a standalone .hsaco or uncompressed clang offload bundle from memory via HSA.
    bool load_executable_from_memory(
        const uint8_t* bytes,
        size_t len,
        HsaStandaloneExecutable& out_exec,
        std::string& err) const;

    std::string status_string(HsaStatus status) const;

    struct Impl;

private:
    HsaRuntime();

    // Resource classes manipulate the dlopen'd runtime and its pools directly
    // (their destructors free into the owning pools; resolve_kernel() needs the
    // symbol query). Friendship is intentionally narrow: these four only.
    friend class HsaBuffer;
    friend class HsaCompletionSignal;
    friend class HsaAqlQueue;
    friend class HsaStandaloneExecutable;

    bool query_symbol_and_pm4_metadata(
        HsaExecutable exec,
        const std::string& base_name,
        const std::string& kd_name,
        HsaResolvedKernel& out,
        std::string& err) const;

    std::unique_ptr<Impl> impl_;
    HsaAgentInfo gpu_ = {};
    HsaAgentInfo cpu_ = {};
    HsaMemoryPool kernarg_pool_ = {0};
    size_t kernarg_granule_ = 0;
    size_t kernarg_alignment_ = 0;

    mutable std::mutex kernel_cache_mu_;
    std::unordered_map<const void*, HsaResolvedKernel> kernel_by_ptr_;
    std::unordered_map<std::string, HsaResolvedKernel> kernel_by_name_;
};

}  // namespace strata::hip::hsa

#endif  // defined(__linux__) && defined(STRATA_WITH_HSA)
