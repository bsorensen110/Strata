// include/strata/hip/pm4_plan.hpp - Allocation-granularity boundary planner,
// GFX12 CacheTable lowering, in-place patch points, and PM4 IB replay executor.
//
// Port note: from warpfront/hipfire railgun + rdna-compute (Apache-2.0)
// Ported from warpfront/hipfire crates/railgun/src/plan.rs (CacheTable, Hazard,
// plan(), ArchLowering::for_arch) and crates/rdna-compute/src/replay.rs
// (PreparedPm4Replay, gfx1201_pm4_pacing, patch_dispatch_dimensions, patch_kernarg_u32).
//
// Guarded to compile only on Linux with STRATA_WITH_HSA enabled or under the
// host-runnable unit test target (STRATA_PM4_HOST_TEST).
#pragma once

#if (defined(__linux__) && defined(STRATA_WITH_HSA)) || defined(STRATA_PM4_HOST_TEST)

#include "strata/hip/pm4.hpp"
#include "strata/hip/pm4_tape.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::hip::pm4 {

/// Target GFX12 sub-architecture for PM4 lowering.
enum class Gfx12Arch : uint8_t {
    Gfx1200 = 0,
    Gfx1201 = 1,
};

/// Parse a device architecture string (e.g. "gfx1201", "gfx1201:sramecc-:xnack-", "gfx1200")
/// into `Gfx12Arch`. Returns false for any non-GFX12 architecture.
bool parse_gfx12_arch(const char* arch_name, Gfx12Arch& out_arch) noexcept;

/// Parse `STRATA_GFX1201_PM4_PACING` ("", "auto", "0", "off", "nop:N") for the target arch.
/// On gfx1201, unset/"auto" returns `kGfx1201PostDispatchNopDwords` (64).
/// On gfx1200, unset/"auto" returns 0.
/// Ported from hipfire crates/rdna-compute/src/replay.rs:4834-4850.
bool resolve_gfx12_post_dispatch_nop(
    Gfx12Arch arch,
    const char* env_value,
    uint32_t& out_nop_dwords) noexcept;

/// Parse `STRATA_HIP_PM4` kill-switch.
/// Default OFF (`nullptr`, `""`, `"0"`, `"off"`, `"false"`, or unknown values return false).
/// Returns true only for `"1"`, `"on"`, `"true"` (case-insensitive).
bool parse_strata_hip_pm4_env(const char* env_value) noexcept;

/// Reason code returned by the T5 PM4 eligibility gate.
enum class Pm4EligibilityReason : uint8_t {
    Admitted = 0,
    RefusedDisabledByEnv = 1,
    RefusedNotLinux = 2,
    RefusedHsaUnavailable = 3,
    RefusedUnsupportedArch = 4,
    RefusedPrefill = 5,
    RefusedBatchedWindow = 6,
    RefusedLayerSplit = 7,
    RefusedCpuOffloadOrDoorbells = 8,
    RefusedLatePleFlagWait = 9,
    RefusedSideStreamBranches = 10,
    RefusedInvalidWindowSize = 11,
};

const char* pm4_eligibility_reason_to_string(Pm4EligibilityReason reason) noexcept;

/// Inputs to the T5 PM4 eligibility gate.
struct Pm4EligibilityInput {
    const char* env_strata_hip_pm4 = nullptr;
    bool is_linux = false;
    bool hsa_available = false;
    const char* arch_name = nullptr;
    bool is_prefill = false;
    bool is_batched = false;
    bool is_layer_split = false;
    bool all_experts_resident = false;
    bool early_ple_flag = true;
    bool has_side_streams = false;
    int window_tokens = 1;
};

/// Result of evaluating the T5 PM4 eligibility gate.
struct Pm4EligibilityDecision {
    bool admitted = false;
    Pm4EligibilityReason reason = Pm4EligibilityReason::RefusedDisabledByEnv;
    Gfx12Arch arch = Gfx12Arch::Gfx1201;
    const char* message = "";
};

/// Evaluate whether a window is eligible for retained PM4 indirect-buffer replay.
/// Admits only Linux + ROCr HSA + exact gfx1201/gfx1200 + STRATA_HIP_PM4=1 +
/// plain AR decode (T=1) or solo verify windows (1 <= T <= 16) with 100% resident
/// experts, single-GPU stage, early PLE flag, and no side-stream forks.
/// Refuses prefill, batched exec_bm_ graphs, layer-split stages, and CPU/PCIe offload.
Pm4EligibilityDecision evaluate_pm4_eligibility(const Pm4EligibilityInput& input) noexcept;

/// Derived boundary plan produced by walking a `Pm4Tape`'s allocation-level effects.
struct Pm4BoundaryPlan {
    Gfx12Arch arch = Gfx12Arch::Gfx1201;
    uint32_t post_dispatch_nop_dwords = kGfx1201PostDispatchNopDwords;
    bool entry_system_acquire = true;
    bool trailing_system_release = true;
    /// Per-dispatch pre-boundary visibility (`boundaries[0]` is `{false, None, false}` because
    /// the IB entry emits `entry_system_acquire` unconditionally).
    std::vector<BoundaryVisibility> boundaries;
    uint32_t wait_idle_count = 0;
    uint32_t inter_node_acquire_count = 0;
    uint32_t system_acquire_count = 0;
};

/// Derive the minimal inter-dispatch barriers and cache acquires for `tape`
/// by walking its allocation-level effects in launch order and joining hazards
/// (`Raw`, `War`, `Waw` over the open epoch since the last `wait_idle`, plus
/// `WarPreWriter` across epochs since the last `AcquireRung::System`) through
/// `gfx12_cache_table_lookup`.
/// Ported from hipfire crates/railgun/src/plan.rs:463-544.
bool plan_pm4_boundaries(
    const Pm4Tape& tape,
    Gfx12Arch arch,
    Pm4BoundaryPlan& out_plan,
    const char* pacing_env_override = nullptr) noexcept;

/// In-place patch descriptor for one lowered dispatch inside a `PreparedPm4Window`.
struct Pm4DispatchPatchPoint {
    uint32_t dispatch_index = 0;
    uint32_t kernarg_arena_offset = 0;
    uint32_t kernarg_bytes = 0;
    std::array<uint32_t, 3> workgroup_size = {1, 1, 1};
    std::array<uint32_t, 3> recorded_grid_workgroups = {1, 1, 1};
    uint32_t sh_reg_base_dword = 0;
    uint32_t dim_x_dword = 0;
    uint32_t dispatch_initiator_dword = 0;
};

/// Callback for submitting a lowered PM4 indirect buffer to a hardware or mock queue.
/// Must set `out_quiescence = ReplayQuiescence::Proven` only when the completion signal
/// reaches zero before timeout.
using SubmitPm4IbFn = bool (*)(
    const uint32_t* ib_dwords,
    uint32_t ib_len_dwords,
    uint64_t ib_device_va,
    ReplayQuiescence& out_quiescence,
    void* user_ctx);

/// Sentinel IB virtual address used exclusively by host-only unit tests (STRATA_PM4_HOST_TEST).
/// Production lowering rejects this sentinel and requires a bound HSA executable buffer VA.
inline constexpr uint64_t kSyntheticHostTestIbVa = 0x200000000ULL;

/// Prepared single-IB PM4 window ready for zero-allocation in-place patching and replay.
class PreparedPm4Window {
public:
    PreparedPm4Window() = default;

    /// Enforce production metadata and GPU buffer binding validation during `prepare()`.
    /// In production builds (`!defined(STRATA_PM4_HOST_TEST)`), this is unconditionally
    /// enabled and cannot be disabled; in `pm4_test` (`STRATA_PM4_HOST_TEST`), callers
    /// can enable it via `set_require_production_metadata(true)` to test production refusal
    /// of synthetic host-test metadata on a host machine.
    void set_require_production_metadata(bool require) noexcept {
#if defined(STRATA_PM4_HOST_TEST)
        require_production_metadata_ = require;
#else
        (void) require;
        require_production_metadata_ = true;
#endif
    }
    bool requires_production_metadata() const noexcept { return require_production_metadata_; }

    /// Validate that `tape` and `ib_device_va` carry bound, non-synthetic ROCr/HSA
    /// kernel metadata (`has_hsa_metadata`, non-zero/non-synthetic `kernel_object` and
    /// 256B-aligned `code_entry`, `kernarg_segment_size >= kernarg_bytes`) and bound
    /// GPU virtual addresses for both the kernarg arena and indirect buffer.
    static bool validate_production_bindings(
        const Pm4Tape& tape,
        uint64_t ib_device_va,
        std::string* out_err = nullptr) noexcept;

    /// Lower a validated `Pm4Tape` into a single GFX12 PM4 command buffer with
    /// minimal barriers, `post_dispatch_nop` pacing, and recorded patch points.
    /// Fails closed if `tape` is invalid or any geometry violates `grid.y/z <= 65536`.
    bool prepare(
        Pm4Tape& tape,
        Gfx12Arch arch,
        const char* pacing_env_override = nullptr,
        std::string* out_err = nullptr) noexcept;

    bool ready() const noexcept {
        return prepared_ && !latched_fallback_ && quiescence_ == ReplayQuiescence::Proven;
    }
    bool is_latched_fallback() const noexcept { return latched_fallback_; }
    ReplayQuiescence quiescence() const noexcept { return quiescence_; }

    /// Patch a 32-bit scalar in `dispatch_index`'s kernarg arena slot in place.
    /// Fails closed if the window is not quiescent or `byte_offset` is out of bounds / unaligned.
    bool patch_kernarg_u32(
        size_t dispatch_index,
        size_t byte_offset,
        uint32_t value) noexcept;

    /// Patch a 64-bit scalar or pointer in `dispatch_index`'s kernarg arena slot in place.
    /// Fails closed if the window is not quiescent or `byte_offset` is out of bounds / unaligned.
    bool patch_kernarg_u64(
        size_t dispatch_index,
        size_t byte_offset,
        uint64_t value) noexcept;

    /// Patch `dispatch_index`'s `DISPATCH_DIRECT` `NUM_THREAD_X/Y/Z` dwords in place
    /// from HIP workgroup counts, reusing `LaunchGeometry::from_hip_workgroups`
    /// (`grid.y/z <= 65536` wrap guard in `src/hip/pm4_packet.cpp`).
    bool patch_dispatch_dimensions(
        size_t dispatch_index,
        std::array<uint32_t, 3> hip_grid_workgroups) noexcept;

    /// Execute the prepared PM4 IB via `submit_fn`.
    /// If the previous execution did not prove quiescence, or `submit_fn` fails / times out,
    /// latches fallback permanently for this window so no retry storm or in-flight mutation occurs.
    bool replay(SubmitPm4IbFn submit_fn, void* user_ctx = nullptr) noexcept;

    void set_ib_device_va(uint64_t va) noexcept { ib_device_va_ = va; }
    uint64_t ib_device_va() const noexcept { return ib_device_va_; }

    const Gfx12Pm4CommandBuffer& command_buffer() const noexcept { return cb_; }
    const Pm4BoundaryPlan& plan() const noexcept { return plan_; }
    const std::vector<Pm4DispatchPatchPoint>& patch_points() const noexcept { return patch_points_; }
    uint32_t replay_count() const noexcept { return replay_count_; }

private:
    Pm4Tape* tape_ = nullptr;
    Pm4BoundaryPlan plan_{};
    Gfx12Pm4CommandBuffer cb_{};
    std::vector<Pm4DispatchPatchPoint> patch_points_;
#if defined(STRATA_PM4_HOST_TEST)
    uint64_t ib_device_va_ = kSyntheticHostTestIbVa;
    bool require_production_metadata_ = false;
#else
    uint64_t ib_device_va_ = 0ULL;
    bool require_production_metadata_ = true;
#endif
    ReplayQuiescence quiescence_ = ReplayQuiescence::Proven;
    uint32_t replay_count_ = 0;
    bool prepared_ = false;
    bool latched_fallback_ = false;
};

/// Execution backend selected for a verify or MTP window launch.
enum class Pm4ExecutionMode : uint8_t {
    HipGraphLaunch = 0,
    Pm4IndirectBufferReplay = 1,
};

/// Reason code for the step-execution backend decision.
enum class Pm4ExecutionReason : uint8_t {
    UsePm4Replay = 0,
    FallbackEligibilityRefused = 1,
    FallbackNoSlot = 2,
    FallbackSlotNotCaptured = 3,
    FallbackSlotLatched = 4,
    FallbackSlotNonQuiescent = 5,
    FallbackWindowNotPrepared = 6,
    FallbackWindowLatched = 7,
    FallbackHsaUnavailable = 8,
};

const char* pm4_execution_reason_to_string(Pm4ExecutionReason reason) noexcept;

/// Combined eligibility + slot-state + prepared-window execution decision.
struct Pm4ExecutionDecision {
    Pm4ExecutionMode mode = Pm4ExecutionMode::HipGraphLaunch;
    Pm4ExecutionReason reason = Pm4ExecutionReason::FallbackNoSlot;
    Pm4EligibilityDecision eligibility{};
};

/// Fail-closed execution selector for verify/MTP step execution.
/// Selects `Pm4IndirectBufferReplay` ONLY when:
///   1. `evaluate_pm4_eligibility(elig_input)` admits the window, AND
///   2. `slot != nullptr` is `Captured`, not latched, and `ReplayQuiescence::Proven`, AND
///   3. `prepared_window != nullptr` is `ready()` and not latched, AND
///   4. `hsa_executor_ready` is true.
/// Otherwise selects `HipGraphLaunch`.
Pm4ExecutionDecision select_pm4_execution_mode(
    const Pm4EligibilityInput& elig_input,
    const Pm4WindowCaptureSlot* slot,
    const PreparedPm4Window* prepared_window,
    bool hsa_executor_ready = true) noexcept;

}  // namespace strata::hip::pm4

#endif  // (defined(__linux__) && defined(STRATA_WITH_HSA)) || defined(STRATA_PM4_HOST_TEST)
