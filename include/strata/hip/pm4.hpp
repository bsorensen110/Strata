// include/strata/hip/pm4.hpp - AQL packet and GFX12 PM4 command codec for retained verify replay.
//
// Port note: from warpfront/hipfire redline-rocr (Apache-2.0)
// Ported from warpfront/hipfire crates/redline-rocr/src/{abi.rs,packet.rs,pm4.rs,pm4_decode.rs,runtime.rs}
// and crates/railgun/src/plan.rs (CacheTable lines 121-201).
#pragma once

#if (defined(__linux__) && defined(STRATA_WITH_HSA)) || defined(STRATA_PM4_HOST_TEST)

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace strata::hip::pm4 {

// ---- ROCr / AQL ABI constants (from redline-rocr/src/abi.rs & packet.rs) ----

inline constexpr size_t kAqlPacketBytes = 64;
inline constexpr size_t kBarrierDependencyCapacity = 5;

inline constexpr uint16_t kPacketTypeVendorSpecific = 0;
inline constexpr uint16_t kPacketTypeInvalid = 1;
inline constexpr uint16_t kPacketTypeKernelDispatch = 2;
inline constexpr uint16_t kPacketTypeBarrierAnd = 3;

inline constexpr uint16_t kPacketHeaderTypeShift = 0;
inline constexpr uint16_t kPacketHeaderBarrierShift = 8;
inline constexpr uint16_t kPacketHeaderScAcquireFenceScopeShift = 9;
inline constexpr uint16_t kPacketHeaderScReleaseFenceScopeShift = 11;

inline constexpr uint16_t kKernelDispatchSetupDimensionsShift = 0;

enum class FenceScope : uint16_t {
    None = 0,
    Agent = 1,
    System = 2,
};

struct HeaderPolicy {
    bool barrier = true;
    FenceScope acquire = FenceScope::System;
    FenceScope release = FenceScope::System;

    static constexpr HeaderPolicy recorded_dispatch() {
        return {true, FenceScope::System, FenceScope::System};
    }
    static constexpr HeaderPolicy cross_queue_barrier() {
        return {true, FenceScope::System, FenceScope::System};
    }
    static constexpr HeaderPolicy two_queue_dispatch() {
        return {true, FenceScope::System, FenceScope::Agent};
    }
    static constexpr HeaderPolicy same_agent_dispatch() {
        return {true, FenceScope::Agent, FenceScope::Agent};
    }
    static constexpr HeaderPolicy two_queue_dependency() {
        return {true, FenceScope::None, FenceScope::None};
    }
    static constexpr HeaderPolicy two_queue_host_terminal() {
        return {true, FenceScope::None, FenceScope::System};
    }
    static constexpr HeaderPolicy batch_boundary_first_serial() {
        return {true, FenceScope::System, FenceScope::None};
    }
    static constexpr HeaderPolicy batch_boundary_internal_serial() {
        return {true, FenceScope::None, FenceScope::None};
    }
    static constexpr HeaderPolicy batch_boundary_first_independent() {
        return {false, FenceScope::System, FenceScope::None};
    }
    static constexpr HeaderPolicy batch_boundary_internal_independent() {
        return {false, FenceScope::None, FenceScope::None};
    }
};

constexpr uint16_t encode_aql_header(uint16_t packet_type, HeaderPolicy policy) {
    uint16_t header = static_cast<uint16_t>(packet_type << kPacketHeaderTypeShift);
    if (policy.barrier) {
        header = static_cast<uint16_t>(header | (1u << kPacketHeaderBarrierShift));
    }
    header = static_cast<uint16_t>(
        header | (static_cast<uint16_t>(policy.acquire) << kPacketHeaderScAcquireFenceScopeShift));
    header = static_cast<uint16_t>(
        header | (static_cast<uint16_t>(policy.release) << kPacketHeaderScReleaseFenceScopeShift));
    return header;
}

constexpr uint32_t encode_aql_dispatch_header_word(
    uint16_t dimensions,
    HeaderPolicy policy = HeaderPolicy::recorded_dispatch()) {
    const uint16_t header = encode_aql_header(kPacketTypeKernelDispatch, policy);
    const uint16_t setup = static_cast<uint16_t>(dimensions << kKernelDispatchSetupDimensionsShift);
    return static_cast<uint32_t>(header) | (static_cast<uint32_t>(setup) << 16);
}

// Vendor-specific AQL PM4-IB header word:
// type=0, barrier=1, acquire=System(2), release=System(2) -> header 0x1500, format=1 -> 0x00011500
inline constexpr uint16_t kVendorPacketFormatPm4Ib = 1;
inline constexpr uint32_t kVendorPm4IbHeaderWord =
    static_cast<uint32_t>(encode_aql_header(kPacketTypeVendorSpecific, HeaderPolicy::recorded_dispatch())) |
    (static_cast<uint32_t>(kVendorPacketFormatPm4Ib) << 16);
static_assert(kVendorPm4IbHeaderWord == 0x00011500u, "AQL PM4-IB vendor header must equal 0x00011500");

// GFX12 grid dimension wrap guard (Standing note: grid.y/z <= 65536 on gfx1201).
inline constexpr uint32_t kGfx12MaxGridYZWorkgroups = 65536u;

enum class PacketErrorCode {
    Ok = 0,
    ZeroDimension,
    WorkgroupExceedsGrid,
    InactiveDimensionNotOne,
    GridDimensionOverflow,
    GridYZWrapExceedsLimit,
    NullKernelObject,
    NullKernarg,
    DynamicCallstackNeedsPrivateSize,
    DynamicGroupSegmentOverflow,
    TooManyBarrierDependencies,
    InvalidIndirectBufferAddress,
    InvalidIndirectBufferSize,
};

struct LaunchGeometry {
    std::array<uint32_t, 3> grid_workitems = {1, 1, 1};
    std::array<uint16_t, 3> workgroup = {1, 1, 1};
    uint16_t dimensions = 1;

    static PacketErrorCode from_workitems(
        std::array<uint32_t, 3> grid_items,
        std::array<uint16_t, 3> wg,
        LaunchGeometry& out);

    static PacketErrorCode from_workgroups(
        std::array<uint32_t, 3> workgroups,
        std::array<uint16_t, 3> wg,
        LaunchGeometry& out);

    // HIP supplies all three workgroup IDs even when Y and Z have extent 1;
    // also enforces the gfx1201 grid.y/z <= 65536 wrap guard.
    static PacketErrorCode from_hip_workgroups(
        std::array<uint32_t, 3> workgroups,
        std::array<uint32_t, 3> wg,
        LaunchGeometry& out);
};

// Loader-derived fields consumed by an AQL kernel-dispatch packet.
struct KernelMetadata {
    uint64_t kernel_object = 0;
    uint32_t kernarg_segment_size = 0;
    uint32_t kernarg_segment_alignment = 16;
    uint32_t group_segment_size = 0;
    uint32_t private_segment_size = 0;
    bool dynamic_callstack = false;
};

// Descriptor fields read from the 64-byte AMDGPU .kd symbol for PM4 direct dispatch.
struct KernelPm4Metadata {
    uint64_t code_entry = 0;
    uint32_t compute_pgm_rsrc1 = 0;
    uint32_t compute_pgm_rsrc2 = 0;
    uint32_t compute_pgm_rsrc3 = 0;
    uint16_t kernel_code_properties = 0;

    bool operator==(const KernelPm4Metadata& o) const {
        return code_entry == o.code_entry &&
               compute_pgm_rsrc1 == o.compute_pgm_rsrc1 &&
               compute_pgm_rsrc2 == o.compute_pgm_rsrc2 &&
               compute_pgm_rsrc3 == o.compute_pgm_rsrc3 &&
               kernel_code_properties == o.kernel_code_properties;
    }
    bool operator!=(const KernelPm4Metadata& o) const { return !(*this == o); }
};

// ABI-identical to hsa_kernel_dispatch_packet_t on a 64-bit little-endian HSA host.
struct alignas(8) KernelDispatchPacket {
    uint32_t full_header = 0;
    uint16_t workgroup_size_x = 0;
    uint16_t workgroup_size_y = 0;
    uint16_t workgroup_size_z = 0;
    uint16_t reserved0 = 0;
    uint32_t grid_size_x = 0;
    uint32_t grid_size_y = 0;
    uint32_t grid_size_z = 0;
    uint32_t private_segment_size = 0;
    uint32_t group_segment_size = 0;
    uint64_t kernel_object = 0;
    uint64_t kernarg_address = 0;
    uint64_t reserved2 = 0;
    uint64_t completion_signal = 0;

    static PacketErrorCode build(
        const KernelMetadata& metadata,
        const LaunchGeometry& geometry,
        uint32_t dynamic_group_bytes,
        uint64_t kernarg_address,
        uint64_t completion_signal,
        KernelDispatchPacket& out,
        HeaderPolicy policy = HeaderPolicy::recorded_dispatch());

    uint32_t published_header_word() const { return full_header; }
};

static_assert(sizeof(KernelDispatchPacket) == kAqlPacketBytes, "KernelDispatchPacket must be 64 bytes");
static_assert(alignof(KernelDispatchPacket) == 8, "KernelDispatchPacket alignment must be 8 bytes");
static_assert(offsetof(KernelDispatchPacket, kernel_object) == 32, "kernel_object offset must be 32");
static_assert(offsetof(KernelDispatchPacket, kernarg_address) == 40, "kernarg_address offset must be 40");
static_assert(offsetof(KernelDispatchPacket, completion_signal) == 56, "completion_signal offset must be 56");

// ABI-identical to hsa_barrier_and_packet_t on a 64-bit little-endian HSA host.
struct alignas(8) BarrierAndPacket {
    uint32_t full_header = 0;
    uint32_t reserved1 = 0;
    std::array<uint64_t, kBarrierDependencyCapacity> dependency_signals = {0, 0, 0, 0, 0};
    uint64_t reserved2 = 0;
    uint64_t completion_signal = 0;

    static PacketErrorCode build(
        const uint64_t* dependencies,
        size_t dependency_count,
        uint64_t completion_signal,
        BarrierAndPacket& out,
        HeaderPolicy policy = HeaderPolicy::cross_queue_barrier());

    uint32_t published_header_word() const { return full_header; }
};

static_assert(sizeof(BarrierAndPacket) == kAqlPacketBytes, "BarrierAndPacket must be 64 bytes");
static_assert(alignof(BarrierAndPacket) == 8, "BarrierAndPacket alignment must be 8 bytes");
static_assert(offsetof(BarrierAndPacket, dependency_signals) == 8, "dependency_signals offset must be 8");
static_assert(offsetof(BarrierAndPacket, completion_signal) == 56, "completion_signal offset must be 56");

// Fully prepared, reusable 64-byte AQL packet image with the publication word kept out of band.
// Queue publication copies bytes[4..64] first and release-stores header_word at bytes[0..4] last.
struct PacketImage {
    std::array<uint8_t, kAqlPacketBytes> bytes = {};
    uint32_t header_word = 0;

    static PacketImage from_kernel(const KernelDispatchPacket& packet);
    static PacketImage from_barrier(const BarrierAndPacket& packet);

    // Build AMD's vendor-specific AQL packet carrying one PM4 INDIRECT_BUFFER command
    // (rocm-systems aqlprofile/src/core/amd_aql_pm4_ib_packet.h; packet.rs:418-462).
    static PacketErrorCode pm4_indirect_buffer(
        uint64_t ib_address,
        uint32_t ib_dwords,
        uint64_t completion_signal,
        PacketImage& out);
};

// ---- GFX12 PM4 constants & encodings (from redline-rocr/src/pm4.rs:14-199) ----
// Port note: ported from warpfront/hipfire redline-rocr (Apache-2.0)

inline constexpr uint32_t kPacket3Nop = 0x10u;
inline constexpr uint32_t kPacket3DispatchDirect = 0x15u;
inline constexpr uint32_t kPacket3IndirectBuffer = 0x3fu;
inline constexpr uint32_t kPacket3CopyData = 0x40u;
inline constexpr uint32_t kPacket3EventWrite = 0x46u;
inline constexpr uint32_t kPacket3ReleaseMem = 0x49u;
inline constexpr uint32_t kPacket3AcquireMem = 0x58u;
inline constexpr uint32_t kPacket3SetShReg = 0x76u;
inline constexpr uint32_t kPacket3MaxBodyDwords = 0x4000u;

// Vendor PM4-IB payload constants (packet.rs:431-456)
inline constexpr uint32_t kIbValidBit = 1u << 23;
inline constexpr uint32_t kIbTemporalLu = 3u << 28;
inline constexpr uint32_t kIbMaxDwords = 0x000fffffu;
inline constexpr uint32_t kVendorPm4IbTrailerDw5 = 10u;

// GC 12.0.x SET_SH_REG offsets used by gfx1200/gfx1201 (SH window 0x200..0x2ff).
inline constexpr uint32_t kShRegFirstGfx12 = 0x200u;
inline constexpr uint32_t kShRegLastGfx12 = 0x2ffu;
inline constexpr uint32_t kComputeNumThreadX = 0x207u;
inline constexpr uint32_t kComputePgmLo = 0x20cu;
inline constexpr uint32_t kComputePgmRsrc1 = 0x212u;
inline constexpr uint32_t kComputeResourceLimits = 0x215u;
inline constexpr uint32_t kComputePgmRsrc3Gfx12 = 0x228u;
inline constexpr uint32_t kComputeUserData0 = 0x240u;

inline constexpr uint32_t kLdsSizeMask = 0x00ff8000u;
inline constexpr uint32_t kLdsSizeShift = 15u;
inline constexpr uint32_t kGfx12LdsGranule = 512u;

inline constexpr uint16_t kEnableSgprKernargSegmentPtr = 1u << 3;
inline constexpr uint16_t kEnableWavefrontSize32 = 1u << 10;
inline constexpr uint16_t kSupportedKernelProperties =
    static_cast<uint16_t>(kEnableSgprKernargSegmentPtr | kEnableWavefrontSize32);

inline constexpr uint32_t kDispatchInitiatorBase = (1u << 0) | (1u << 2) | (1u << 5);  // 0x25
inline constexpr uint32_t kDispatchInitiatorCsW32En = 1u << 15;                         // 0x8000
inline constexpr uint32_t kComputeResourceLimitsDefault = 0x3ffu;
inline constexpr uint32_t kEventWriteCsPartialFlush = 0x407u;

// GFX12 GCR bit definitions and rung values (pm4.rs:44-89, 198)
inline constexpr uint32_t kGcrGlkInv = 1u << 7;
inline constexpr uint32_t kGcrGlvInv = 1u << 8;
inline constexpr uint32_t kGcrGl1Inv = 1u << 9;
inline constexpr uint32_t kGcrGl2Inv = 1u << 14;
inline constexpr uint32_t kGcrGl2Wb = 1u << 15;
inline constexpr uint32_t kGcrSeqForward = 1u << 16;

// Primary GCR values on gfx12 (railgun/src/plan.rs:121-201 & redline-rocr/src/pm4.rs)
inline constexpr uint32_t kGcrInterNodeGfx12 = kGcrGlkInv | kGcrGlvInv | kGcrSeqForward;  // 0x10180
inline constexpr uint32_t kGcrSystemGfx12 = 0x0000c3b1u;                                  // 0xc3b1
inline constexpr uint32_t kGcrLegacyAcquireSystem = 0x0001c3f1u;                          // 0x1c3f1

// Default post-dispatch NOP body dwords on gfx1201 (railgun ArchLowering: post_dispatch_nop=64).
inline constexpr uint32_t kGfx1201PostDispatchNopDwords = 64u;

constexpr uint32_t packet3(uint32_t opcode, uint32_t body_dwords, bool compute) {
    return (3u << 30) | ((body_dwords - 1u) << 16) | (opcode << 8) | (compute ? (1u << 1) : 0u);
}

constexpr uint32_t dispatch_initiator(bool wave32) {
    return kDispatchInitiatorBase | (wave32 ? kDispatchInitiatorCsW32En : 0u);
}

// Cache actions paired with a compute-idle wait between dependent gfx12 dispatches (pm4.rs:58-89).
enum class Gfx12RmwAcquirePolicy : uint32_t {
    CurrentSequential = kGcrInterNodeGfx12,                                                  // 0x10180
    RadvGlobal = kGcrGlkInv | kGcrGlvInv | kGcrGl1Inv | kGcrGl2Inv | kGcrGl2Wb,              // 0xc380
    SameAgentParallelL1 = kGcrGlkInv | kGcrGlvInv | kGcrGl1Inv,                              // 0x0380
    SameAgentParallelL0 = kGcrGlkInv | kGcrGlvInv,                                           // 0x0180
    HipLlvmVmemL1 = kGcrGlvInv | kGcrGl1Inv,                                                 // 0x0300
};

constexpr uint32_t gcr_cntl_for_policy(Gfx12RmwAcquirePolicy policy) {
    return static_cast<uint32_t>(policy);
}

// Hazard model ported from hipfire's gfx12 CacheTable (railgun/src/plan.rs:121-201).
enum class HazardKind : uint8_t {
    Raw = 0,
    War = 1,
    Waw = 2,
    WarPreWriter = 3,
};

/// A1 memory class of one pointer effect. Ported from warpfront/hipfire
/// crates/railgun/src/kernel.rs (MemClass), trimmed to the classes this
/// planner reasons about: VmemLoad/VmemStore drive the RAW/WAR/WAW edges and
/// SmemLoad is workgroup-private (never crosses a dispatch boundary). The
/// atomic/LDS-DMA classes hipfire defines are intentionally absent: no
/// admitted kernel uses them, and an effect naming one fails closed.
/// Port note: from warpfront/hipfire crates/railgun/src/kernel.rs:30-48 (Apache-2.0).
enum class MemAccessKind : uint8_t {
    VmemLoad = 0,
    VmemStore = 1,
    SmemLoad = 2,
};

/// AMD kernel-code-object ABI `kernel_code_properties` bit: the dispatch
/// passes the kernarg segment via an SGPR pointer (always true for HIP
/// kernels). Port note: from warpfront/hipfire crates/redline-rocr/src/pm4.rs:39 (Apache-2.0).
inline constexpr uint16_t kAmdKernelCodePropertyEnableSgprKernargSegmentPtr = 1u << 3;

enum class AcquireRung : uint8_t {
    None = 0,
    InterNode = 1,  // GCR 0x10180 (scalar + vector L0 + seq forward)
    System = 2,     // GCR 0xc3b1  (GL1 + GL2 wb/inv + scalar/vector/inst)
};

struct BoundaryVisibility {
    bool wait_idle = false;
    AcquireRung acquire = AcquireRung::None;
    bool pre_writer = false;
};

// Returns the gfx12 CacheTable row for a given hazard (railgun/src/plan.rs:126-160).
constexpr BoundaryVisibility gfx12_cache_table_lookup(HazardKind hazard) {
    switch (hazard) {
        case HazardKind::Raw:
        case HazardKind::War:
        case HazardKind::Waw:
            return {true, AcquireRung::InterNode, false};
        case HazardKind::WarPreWriter:
            return {true, AcquireRung::System, true};
    }
    return {true, AcquireRung::System, false};
}

// Access-class-aware wrapper: SMEM effects are workgroup-private and never
// cross a dispatch boundary, so a hazard pair involving one needs no
// boundary at all. Everything else is keyed by the hazard alone, matching
// hipfire railgun/src/plan.rs CacheTable::for_arch("gfx12xx").
// Port note: from warpfront/hipfire crates/railgun/src/plan.rs (Apache-2.0).
constexpr BoundaryVisibility gfx12_cache_table_lookup(HazardKind hazard, MemAccessKind src,
                                                      MemAccessKind dst) {
    if (hazard == HazardKind::WarPreWriter) return gfx12_cache_table_lookup(hazard);
    if (src == MemAccessKind::SmemLoad || dst == MemAccessKind::SmemLoad) return {};
    return gfx12_cache_table_lookup(hazard);
}

// Join two boundary visibilities: OR the waits, take the stronger acquire
// rung (None < InterNode < System). Port note: from warpfront/hipfire
// crates/railgun/src/plan.rs Visibility::join (Apache-2.0).
constexpr BoundaryVisibility join_visibility(BoundaryVisibility a, BoundaryVisibility b) {
    return {a.wait_idle || b.wait_idle,
            a.acquire > b.acquire ? a.acquire : b.acquire,
            a.pre_writer || b.pre_writer};
}

const char* packet_error_to_string(PacketErrorCode code) noexcept;

/// In-IB dword positions recorded when a dispatch is lowered, so a prepared
/// window can patch it in place. DISPATCH_DIRECT layout: header at +0,
/// COMPUTE_DIMENSION_X/Y/Z at +1/+2/+3, DISPATCH_INITIATOR at +4 — hence
/// dim_x == dispatch_initiator_dword - 3. `sh_reg_base_dword` is where the
/// SET_SH_REG emission began (stateful cache mode may elide runs, so treat
/// it as a locator, not a fixed-size header).
struct Pm4DispatchOffsets {
    uint32_t sh_reg_base_dword = 0;
    uint32_t dispatch_initiator_dword = 0;
};

enum class Pm4BuildError {
    Ok = 0,
    MissingKernelDescriptor,
    ScratchUnsupported,
    UnsupportedKernelProperties,
    NullKernarg,
    GroupSegmentOverflow,
    GroupSegmentTooLarge,
    MalformedStream,
};

enum class Gfx12RegisterCacheMode : uint8_t {
    Stateless = 0,
    StatefulAll = 1,
    StatefulStaticOnly = 2,
};

// Retained GFX12 PM4 command words suitable for one PM4 indirect buffer (pm4.rs:106-568).
class Gfx12Pm4CommandBuffer {
public:
    Gfx12Pm4CommandBuffer() = default;
    explicit Gfx12Pm4CommandBuffer(Gfx12RegisterCacheMode mode) : cache_mode_(mode) {}

    static Gfx12Pm4CommandBuffer new_stateful() {
        return Gfx12Pm4CommandBuffer(Gfx12RegisterCacheMode::StatefulAll);
    }

    static Gfx12Pm4CommandBuffer new_static_stateful() {
        return Gfx12Pm4CommandBuffer(Gfx12RegisterCacheMode::StatefulStaticOnly);
    }

    void set_post_dispatch_nop(uint32_t body_dwords) { post_dispatch_nop_body_ = body_dwords; }
    uint32_t post_dispatch_nop() const { return post_dispatch_nop_body_; }
    uint64_t pacing_dwords() const { return pacing_dwords_; }

    // Emit one NOP packet with `body_dwords` zero body words (`1 + body_dwords` total dwords).
    void emit_nop_body(uint32_t body_dwords);

    // Legacy ROCr AcquireMemTemplate (8 dwords, GCR 0x1c3f1).
    void acquire_system();

    // GFX12 ownership-boundary / pre-writer system acquire (8 dwords, GCR 0xc3b1).
    // With `wait_idle` a CS_PARTIAL_FLUSH event precedes the ACQUIRE_MEM.
    void acquire_system_gfx12(bool wait_idle = false);

    // Same-agent inter-node acquire for one retained gfx12 tape (8 dwords, GCR 0x10180).
    void acquire_inter_node_gfx12();

    // Emit ACQUIRE_MEM with the specified Gfx12RmwAcquirePolicy rung.
    void acquire_rmw_gfx12(Gfx12RmwAcquirePolicy policy);

    // Emit ACQUIRE_MEM with an explicit GCR control word (0x10180 or 0xc3b1).
    void emit_acquire_gcr(uint32_t gcr_cntl);

    // Emit EVENT_WRITE CS_PARTIAL_FLUSH (0x407) + RMW acquire boundary.
    void dependency_rmw_same_agent_gfx12();
    void dependency_rmw_hip_llvm_vmem_gfx12();

    // Emit boundary operations prescribed by a BoundaryVisibility entry.
    void emit_boundary(BoundaryVisibility vis);

    // Emit EVENT_WRITE CS_PARTIAL_FLUSH (2 dwords: 0xc0004600, 0x407).
    void wait_compute_idle();

    // Emit SET_SH_REG for contiguous registers starting at `first_reg` (0x200..0x2ff).
    void set_sh_regs(uint32_t first_reg, const uint32_t* values, size_t count);
    void set_sh_regs(uint32_t first_reg, std::initializer_list<uint32_t> values) {
        set_sh_regs(first_reg, values.begin(), values.size());
    }

    // Emit DISPATCH_DIRECT (5 dwords) + optional post-dispatch NOP pacing.
    void emit_dispatch_direct(std::array<uint32_t, 3> grid_workitems, bool wave32);

    // Full zero-scratch kernel dispatch lowering (SET_SH_REG sequence + DISPATCH_DIRECT + pacing).
    // When out_offsets is non-null, receives the in-IB dword positions of the
    // emitted sequence (see Pm4DispatchOffsets) for later in-place patching.
    Pm4BuildError dispatch(
        const KernelMetadata& loader,
        const KernelPm4Metadata& pm4,
        const LaunchGeometry& geometry,
        uint32_t dynamic_group_bytes,
        uint64_t kernarg_address,
        Pm4DispatchOffsets* out_offsets = nullptr);

    uint32_t len_dwords() const { return static_cast<uint32_t>(dwords_.size()); }
    bool is_empty() const { return dwords_.empty(); }
    // Mutable view for in-place patching of an already-lowered IB (see
    // PreparedPm4Window::patch_dispatch_dimensions). Prefer the targeted
    // patch helpers; this is the raw seam they are written against.
    std::vector<uint32_t>& dwords_mut() { return dwords_; }
    // Reset for reuse across captures: dwords AND the SH-register cache, so a
    // reused buffer never elides a SET_SH_REG against a previous IB's state.
    void clear() { dwords_.clear(); register_state_.clear(); }
    const std::vector<uint32_t>& dwords() const { return dwords_; }
    std::vector<uint8_t> as_bytes() const;

private:
    void emit_set_sh_regs_raw(uint32_t first_reg, const uint32_t* values, size_t count);
    void pace_after_dispatch();

    std::vector<uint32_t> dwords_;
    std::map<uint32_t, uint32_t> register_state_;
    Gfx12RegisterCacheMode cache_mode_ = Gfx12RegisterCacheMode::Stateless;
    uint32_t post_dispatch_nop_body_ = 0;
    uint64_t pacing_dwords_ = 0;
};

// ---- CPU-only GFX12 PM4 stream decoder / encoder (from pm4_decode.rs) ----

enum class Pm4PacketKind : uint8_t {
    SetShRegs = 0,
    DispatchDirect = 1,
    EventComputeIdle = 2,
    Acquire = 3,
    Nop = 4,
};

struct DecodedPm4Packet {
    Pm4PacketKind kind = Pm4PacketKind::EventComputeIdle;
    uint32_t first_reg = 0;                       // SetShRegs: 0x200..0x2ff
    std::vector<uint32_t> reg_values;             // SetShRegs
    std::array<uint32_t, 3> grid_workitems = {};  // DispatchDirect
    uint32_t dispatch_initiator = 0;              // DispatchDirect (0x25 or 0x8025)
    AcquireRung acquire_scope = AcquireRung::None;
    uint32_t acquire_gcr = 0;                     // Acquire (0xc3b1 or 0x10180)
    uint32_t nop_body_dwords = 0;                 // Nop

    bool operator==(const DecodedPm4Packet& o) const;
    bool operator!=(const DecodedPm4Packet& o) const { return !(*this == o); }
};

struct Pm4ParseError {
    bool ok = true;
    size_t dword = 0;
    const char* reason = "";
};

// Decode a GFX12 PM4 stream into typed packets; fails closed on any unknown or malformed packet.
Pm4ParseError parse_pm4_gfx12(
    const uint32_t* words,
    size_t word_count,
    std::vector<DecodedPm4Packet>& out_packets,
    bool allow_nop = false);

// Re-encode decoded GFX12 PM4 packets back into dwords; fails closed on invalid packet fields.
Pm4BuildError encode_pm4_gfx12(
    const std::vector<DecodedPm4Packet>& packets,
    std::vector<uint32_t>& out_words,
    bool allow_nop = false);

// ---- AMDGPU ELF64 .kd descriptor & clang offload bundle helpers (from runtime.rs:3084-3310) ----

// Parse a 64-byte AMDGPU kernel descriptor (.kd) in memory and rebase `code_entry` from `loaded_descriptor_va`.
bool parse_kernel_descriptor_bytes(
    const uint8_t* descriptor_64b,
    size_t len,
    uint64_t loaded_descriptor_va,
    KernelPm4Metadata& out,
    uint32_t* out_group_fixed_size = nullptr,
    uint32_t* out_private_fixed_size = nullptr);

// Parse an AMDGPU ELF64 image to locate `symbol_name` (e.g. "kernel.kd") and extract its KernelPm4Metadata.
bool parse_kernel_pm4_metadata_from_elf(
    const uint8_t* elf_bytes,
    size_t elf_len,
    const char* symbol_name,
    uint64_t loaded_descriptor_va,
    KernelPm4Metadata& out);

// Unwrap an uncompressed __CLANG_OFFLOAD_BUNDLE__ to its embedded amdgcn-amd-amdhsa ELF subrange.
// If `image` is already a raw ELF64, returns [0, image_len). Fails closed on CCOB or malformed bundles.
bool unwrap_clang_offload_bundle(
    const uint8_t* image,
    size_t image_len,
    size_t& out_offset,
    size_t& out_size,
    std::string& out_error);

}  // namespace strata::hip::pm4

#endif  // (__linux__ && STRATA_WITH_HSA) || STRATA_PM4_HOST_TEST
