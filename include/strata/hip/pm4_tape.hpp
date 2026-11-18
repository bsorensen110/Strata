// include/strata/hip/pm4_tape.hpp - Tape recorder, static per-kernel-family effect table,
// fixed-capacity kernarg arena, and per-window fail-closed latch for GFX12 PM4 replay.
//
// Port note: from warpfront/hipfire rdna-compute + railgun (Apache-2.0)
// Ported from warpfront/hipfire crates/rdna-compute/src/replay.rs (pointer_effects,
// expected_kernarg_bytes, recorded_resource_accesses, ReplayQuiescence) and
// crates/railgun/src/plan.rs (ResourceId, Access, Effect).
//
// Guarded to compile only on Linux with STRATA_WITH_HSA enabled or under the
// host-runnable unit test target (STRATA_PM4_HOST_TEST).
#pragma once

#if (defined(__linux__) && defined(STRATA_WITH_HSA)) || defined(STRATA_PM4_HOST_TEST)

#include "strata/hip/pm4.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::hip::pm4 {

/// Dense allocation-level resource identifier within a recorded tape.
using ResourceId = uint32_t;
inline constexpr ResourceId kInvalidResourceId = 0xffffffffu;

/// Default per-window kernarg arena capacity (256 KiB) and 64-byte slot alignment.
inline constexpr size_t kDefaultKernargArenaBytes = 256u * 1024u;
inline constexpr size_t kKernargSlotAlignment = 64u;
inline constexpr size_t kKernargTailAlignment = 16u;

/// Sentinel addresses used exclusively by host-only unit tests (STRATA_PM4_HOST_TEST).
/// Production lowering rejects these sentinels and requires real ROCr/HSA bindings.
inline constexpr uint64_t kSyntheticHostTestKernelObject = 0x7f000000ULL;
inline constexpr uint64_t kSyntheticHostTestCodeEntry = 0x7f000100ULL;
inline constexpr uint64_t kSyntheticHostTestArenaBaseVa = 0x100000000ULL;

/// Quiescence proof for a retained window slot before re-capture or replay.
/// Ported from hipfire crates/rdna-compute/src/replay.rs (ReplayQuiescence).
enum class ReplayQuiescence : uint8_t {
    Proven = 0,
    Unknown = 1,
};

/// MemAccessKind (the A1 memory class of a pointer effect) lives in
/// strata/hip/pm4.hpp alongside the hazard vocabulary that consumes it.

/// Fail-closed status of a PM4 tape recording.
enum class TapeStatus : uint8_t {
    Ok = 0,
    InvalidatedUnknownKernel = 1,
    InvalidatedKernargSizeMismatch = 2,
    InvalidatedPointerOffsetOutOfBounds = 3,
    InvalidatedUnalignedPointerOffset = 4,
    InvalidatedUnresolvedPointer = 5,
    InvalidatedArenaOverflow = 6,
    InvalidatedInvalidGeometry = 7,
    InvalidatedIncompleteCapture = 8,
    InvalidatedNonQuiescent = 9,
    InvalidatedEmptyTape = 10,
};

const char* tape_status_to_string(TapeStatus status) noexcept;

/// One pointer argument's byte offset in the naturally aligned HIP kernarg block
/// and its conservative memory access classification.
/// Non-const or read-modify-write pointers are classified as VmemStore.
struct KernelPointerEffect {
    uint16_t offset = 0;
    MemAccessKind access = MemAccessKind::VmemLoad;
};

/// Static contract for a kernel family admitted to PM4 window replay.
/// Any kernel absent from the static table returns nullptr from lookup_kernel_contract()
/// and immediately fails closed.
struct KernelContract {
    const char* family_name = nullptr;
    uint32_t expected_kernarg_bytes = 0;
    const KernelPointerEffect* pointer_effects = nullptr;
    size_t pointer_effect_count = 0;
    bool supports_dynamic_grid = false;
};

/// Look up a kernel family in Strata's static decode/verify effect table.
/// Returns nullptr if the kernel is unknown or not admitted (fail-closed).
const KernelContract* lookup_kernel_contract(const char* family_name) noexcept;

/// Look up a kernel family in a caller-supplied contract table (or the default table if custom_table is null).
const KernelContract* lookup_kernel_contract_in(
    const char* family_name,
    const KernelContract* custom_table,
    size_t custom_count) noexcept;

/// Inspect the built-in static kernel contract table (for audit and unit tests).
const KernelContract* registered_kernel_contracts(size_t& out_count) noexcept;

/// Explicit mapping from a concrete kernel symbol name in Strata sources
/// (e.g. "gpu_stamp_kernel<true>", "bf16_f32_mmvf_multi_kernel<256, 4, true, false>",
/// or a `__global__` kernel identifier) to its canonical `KernelContract::family_name`.
struct KernelSymbolAlias {
    const char* symbol_name = nullptr;
    const char* canonical_family = nullptr;
};

/// Resolve a concrete kernel symbol name to its canonical contract family name using an
/// explicit, source-verified lookup table (never a substring or name-mangling guess).
/// Returns nullptr if `symbol_name` is not in the explicit table or contract list.
const char* canonicalize_kernel_symbol(const char* symbol_name) noexcept;

/// Inspect the built-in static kernel symbol canonicalization table (for audit and unit tests).
const KernelSymbolAlias* registered_kernel_symbol_aliases(size_t& out_count) noexcept;

/// One live device allocation range `[base, base + size)`.
struct AllocationRange {
    uint64_t base = 0;
    uint64_t size = 0;

    constexpr uint64_t end() const noexcept { return base + size; }
    constexpr bool contains(uint64_t ptr) const noexcept {
        return base != 0 && size != 0 && ptr >= base && ptr < (base + size);
    }
    constexpr bool operator==(const AllocationRange& o) const noexcept {
        return base == o.base && size == o.size;
    }
};

/// Callback that resolves a device pointer to its owning allocation `[base, base + size)`.
/// Returns true on success, false if the pointer is unknown / not a device allocation.
using AddressRangeResolver = bool (*)(uint64_t device_ptr, AllocationRange& out_range, void* user_ctx);

/// Deterministic host-side table of known device allocations, usable both as a test fixture
/// and as a pre-registered arena lookup without driver round-trips.
class AllocationTable {
public:
    bool insert(uint64_t base, uint64_t size) noexcept;
    void clear() noexcept { ranges_.clear(); }
    bool resolve(uint64_t device_ptr, AllocationRange& out_range) const noexcept;
    size_t size() const noexcept { return ranges_.size(); }

    static bool resolver_thunk(uint64_t device_ptr, AllocationRange& out_range, void* user_ctx) noexcept;

private:
    std::vector<AllocationRange> ranges_;
};

/// Resolved allocation-granularity memory effect of one pointer slot in a recorded dispatch.
struct ResourceEffect {
    ResourceId resource = kInvalidResourceId;
    uint64_t allocation_base = 0;
    uint64_t allocation_bytes = 0;
    uint64_t access_ptr = 0;
    uint16_t kernarg_offset = 0;
    MemAccessKind access = MemAccessKind::VmemLoad;

    constexpr bool is_read() const noexcept {
        return access == MemAccessKind::VmemLoad || access == MemAccessKind::SmemLoad;
    }
    constexpr bool is_write() const noexcept {
        return access == MemAccessKind::VmemStore;
    }
};

/// Fixed-capacity per-window kernarg byte arena.
/// Pre-sized once before capture; never reallocates during capture or replay.
/// Overflow fails closed so pointers into the arena never dangle or move.
class KernargArena {
public:
    explicit KernargArena(size_t capacity_bytes = kDefaultKernargArenaBytes);

    void reset() noexcept { used_ = 0; }
    void reinitialize(size_t capacity_bytes);

    size_t capacity() const noexcept { return storage_.size(); }
    size_t used() const noexcept { return used_; }

    /// Aligns cursor to 64 bytes, copies `size` bytes from `src`, pads tail to 16 bytes,
    /// and writes the 64-byte-aligned byte offset to `out_offset`.
    /// Returns false on overflow without mutating `storage_` or reallocating.
    bool allocate_and_copy(const void* src, size_t size, uint32_t& out_offset) noexcept;

    uint8_t* data() noexcept { return storage_.data(); }
    const uint8_t* data() const noexcept { return storage_.data(); }

    uint8_t* slot_ptr(uint32_t offset) noexcept { return storage_.data() + offset; }
    const uint8_t* slot_ptr(uint32_t offset) const noexcept { return storage_.data() + offset; }

    /// Base GPU/HSA virtual address of the arena when backed by or mirrored into fine-grained memory.
    void set_device_base_va(uint64_t va) noexcept { device_base_va_ = va; }
    uint64_t device_base_va() const noexcept { return device_base_va_; }
    uint64_t slot_device_va(uint32_t offset) const noexcept { return device_base_va_ + offset; }

private:
    std::vector<uint8_t> storage_;
    size_t used_ = 0;
#if defined(STRATA_PM4_HOST_TEST)
    uint64_t device_base_va_ = kSyntheticHostTestArenaBaseVa;
#else
    uint64_t device_base_va_ = 0ULL;
#endif
};

/// One kernel launch recorded into a Pm4Tape.
struct RecordedDispatch {
    std::string kernel_family;
    const void* host_function = nullptr;
    std::array<uint32_t, 3> hip_grid_workgroups = {1, 1, 1};
    std::array<uint32_t, 3> hip_workgroup_size = {1, 1, 1};
    LaunchGeometry geometry{};
    uint32_t dynamic_lds_bytes = 0;
    uint32_t kernarg_arena_offset = 0;
    uint32_t kernarg_bytes = 0;
    uint32_t kernarg_padded_bytes = 0;
    KernelMetadata loader_meta{};
    KernelPm4Metadata pm4_meta{};
    bool has_hsa_metadata = false;
    std::vector<ResourceEffect> effects;
};

/// Recorded sequence of kernel dispatches and allocation-level effects for one verify window.
class Pm4Tape {
public:
    explicit Pm4Tape(size_t arena_capacity_bytes = kDefaultKernargArenaBytes);

    void set_resolver(AddressRangeResolver resolver, void* user_ctx) noexcept {
        resolver_ = resolver;
        resolver_ctx_ = user_ctx;
    }

    void set_custom_contracts(const KernelContract* table, size_t count) noexcept {
        custom_contracts_ = table;
        custom_contract_count_ = count;
    }

    /// Begin recording a window. Fails closed with InvalidatedNonQuiescent if the previous
    /// replay of this window has not proven quiescence.
    bool begin_recording(ReplayQuiescence prev_quiescence = ReplayQuiescence::Proven) noexcept;

    /// Record one kernel launch into the tape.
    /// Any contract violation (unknown kernel, kernarg size mismatch, grid.y/z > 65536,
    /// arena overflow, unaligned or unresolved pointer) invalidates the tape and fails closed.
    bool record_launch(
        const char* kernel_family,
        const void* host_function,
        std::array<uint32_t, 3> hip_grid_workgroups,
        std::array<uint32_t, 3> hip_workgroup_size,
        uint32_t dynamic_lds_bytes,
        const void* kernarg_data,
        size_t kernarg_bytes,
        const KernelMetadata* loader_meta = nullptr,
        const KernelPm4Metadata* pm4_meta = nullptr) noexcept;

    /// Finalize recording and verify that the number of recorded launches matches
    /// the captured hipGraph kernel node count (when `expected_graph_kernel_nodes > 0`).
    /// Any mismatch means a raw `<<<...>>>` launch bypassed `launch_pdl`, failing closed.
    TapeStatus finalize_recording(size_t expected_graph_kernel_nodes = 0) noexcept;

    /// Bind validated ROCr/HSA kernel descriptor metadata onto `dispatches_[dispatch_index]`.
    /// Fails closed if `kernel_object == 0`, `code_entry == 0`, `code_entry` is not 256B-aligned,
    /// or `kernarg_segment_size < kernarg_bytes`.
    bool bind_hsa_metadata(
        size_t dispatch_index,
        const KernelMetadata& loader_meta,
        const KernelPm4Metadata& pm4_meta) noexcept;

    /// Explicitly invalidate the tape with a reason.
    void invalidate(TapeStatus status, const char* reason) noexcept;

    bool valid() const noexcept { return finalized_ && status_ == TapeStatus::Ok; }
    TapeStatus status() const noexcept { return status_; }
    const std::string& failure_reason() const noexcept { return failure_reason_; }

    size_t launch_count() const noexcept { return dispatches_.size(); }
    const std::vector<RecordedDispatch>& dispatches() const noexcept { return dispatches_; }
    std::vector<RecordedDispatch>& dispatches_mut() noexcept { return dispatches_; }

    size_t resource_count() const noexcept { return resources_.size(); }
    const std::vector<AllocationRange>& resources() const noexcept { return resources_; }

    KernargArena& arena() noexcept { return arena_; }
    const KernargArena& arena() const noexcept { return arena_; }

private:
    ResourceId intern_resource(const AllocationRange& range);

    KernargArena arena_;
    AddressRangeResolver resolver_ = nullptr;
    void* resolver_ctx_ = nullptr;
    const KernelContract* custom_contracts_ = nullptr;
    size_t custom_contract_count_ = 0;

    std::vector<RecordedDispatch> dispatches_;
    std::vector<AllocationRange> resources_;
    TapeStatus status_ = TapeStatus::InvalidatedEmptyTape;
    std::string failure_reason_;
    bool recording_ = false;
    bool finalized_ = false;
};

/// Lifecycle state of one verify-window PM4 slot (keyed by window token count T).
enum class Pm4WindowState : uint8_t {
    Uninitialized = 0,
    Recording = 1,
    Captured = 2,
    LatchedFallback = 3,
};

/// Per-window capture slot that enforces the fail-closed per-window latch (no retry storm)
/// and tracks replay quiescence across window executions.
class Pm4WindowCaptureSlot {
public:
    explicit Pm4WindowCaptureSlot(size_t arena_capacity_bytes = kDefaultKernargArenaBytes)
        : tape_(arena_capacity_bytes) {}

    Pm4WindowState state() const noexcept { return state_; }
    bool is_latched_fallback() const noexcept { return state_ == Pm4WindowState::LatchedFallback; }
    bool is_captured() const noexcept { return state_ == Pm4WindowState::Captured && tape_.valid(); }
    bool can_attempt_capture() const noexcept { return state_ == Pm4WindowState::Uninitialized; }

    ReplayQuiescence quiescence() const noexcept { return quiescence_; }
    void set_quiescence(ReplayQuiescence q) noexcept {
        quiescence_ = q;
        if (q != ReplayQuiescence::Proven) {
            latch_fallback(TapeStatus::InvalidatedNonQuiescent, "previous window replay did not prove quiescence");
        }
    }

    /// Start capturing this window slot. If already LatchedFallback or non-quiescent,
    /// returns false immediately without resetting or retrying (no retry storm).
    bool begin_capture() noexcept;

    /// Complete capture and validate against the hipGraph kernel node count.
    /// On any failure, transitions permanently to LatchedFallback until reset().
    bool end_capture(size_t expected_graph_kernel_nodes = 0) noexcept;

    /// Permanently latch this window slot to hipGraph fallback until reset().
    void latch_fallback(TapeStatus status, const char* reason) noexcept;

    /// Reset slot (only on explicit model/graph teardown).
    void reset() noexcept;

    uint32_t capture_attempts() const noexcept { return capture_attempts_; }
    Pm4Tape& tape() noexcept { return tape_; }
    const Pm4Tape& tape() const noexcept { return tape_; }

private:
    Pm4Tape tape_;
    Pm4WindowState state_ = Pm4WindowState::Uninitialized;
    ReplayQuiescence quiescence_ = ReplayQuiescence::Proven;
    uint32_t capture_attempts_ = 0;
};

/// Map a host kernel function pointer to its canonical family name for `launch_pdl` capture.
void register_host_kernel_family(const void* host_function, const char* family_name);
const char* lookup_host_kernel_family(const void* host_function) noexcept;
void clear_host_kernel_families() noexcept;

/// Thread-local active PM4 tape for `launch_pdl` capture scope.
Pm4Tape* active_pm4_tape() noexcept;
bool is_pm4_capture_scope_active() noexcept;

/// RAII scope guard that binds a `Pm4Tape` as the current thread's active recorder.
class Pm4TapeScopeGuard {
public:
    explicit Pm4TapeScopeGuard(Pm4Tape* tape) noexcept;
    ~Pm4TapeScopeGuard() noexcept;
    Pm4TapeScopeGuard(const Pm4TapeScopeGuard&) = delete;
    Pm4TapeScopeGuard& operator=(const Pm4TapeScopeGuard&) = delete;

private:
    Pm4Tape* prev_ = nullptr;
};

}  // namespace strata::hip::pm4

#endif  // (defined(__linux__) && defined(STRATA_WITH_HSA)) || defined(STRATA_PM4_HOST_TEST)
