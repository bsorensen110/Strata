// src/hip/pm4_packet.cpp - AQL packet and GFX12 PM4 command codec implementation.
//
// Port note: from warpfront/hipfire redline-rocr (Apache-2.0)
// Ported from warpfront/hipfire crates/redline-rocr/src/{packet.rs,pm4.rs,pm4_decode.rs,runtime.rs}.

#include "strata/hip/pm4.hpp"

#if (defined(__linux__) && defined(STRATA_WITH_HSA)) || defined(STRATA_PM4_HOST_TEST)

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace strata::hip::pm4 {

namespace {

inline void write_le16(uint8_t* dst, uint16_t v) {
    dst[0] = static_cast<uint8_t>(v & 0xffu);
    dst[1] = static_cast<uint8_t>((v >> 8) & 0xffu);
}

inline void write_le32(uint8_t* dst, uint32_t v) {
    dst[0] = static_cast<uint8_t>(v & 0xffu);
    dst[1] = static_cast<uint8_t>((v >> 8) & 0xffu);
    dst[2] = static_cast<uint8_t>((v >> 16) & 0xffu);
    dst[3] = static_cast<uint8_t>((v >> 24) & 0xffu);
}

inline void write_le64(uint8_t* dst, uint64_t v) {
    write_le32(dst, static_cast<uint32_t>(v & 0xffffffffu));
    write_le32(dst + 4, static_cast<uint32_t>(v >> 32));
}

inline bool read_le16(const uint8_t* bytes, size_t len, size_t offset, uint16_t& out) {
    if (offset > len || len - offset < 2) return false;
    out = static_cast<uint16_t>(bytes[offset]) |
          static_cast<uint16_t>(static_cast<uint16_t>(bytes[offset + 1]) << 8);
    return true;
}

inline bool read_le32(const uint8_t* bytes, size_t len, size_t offset, uint32_t& out) {
    if (offset > len || len - offset < 4) return false;
    out = static_cast<uint32_t>(bytes[offset]) |
          (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
          (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
          (static_cast<uint32_t>(bytes[offset + 3]) << 24);
    return true;
}

inline bool read_le64(const uint8_t* bytes, size_t len, size_t offset, uint64_t& out) {
    uint32_t lo = 0, hi = 0;
    if (!read_le32(bytes, len, offset, lo) || !read_le32(bytes, len, offset + 4, hi)) return false;
    out = static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
    return true;
}

constexpr std::array<uint32_t, 6> kAcquirePrefixGfx12 = {
    0u, 0xffffffffu, 0x00ffffffu, 0u, 0u, 0x0000000au
};

}  // namespace

// ---- LaunchGeometry (packet.rs:141-209) ----

PacketErrorCode LaunchGeometry::from_workitems(
    std::array<uint32_t, 3> grid_items,
    std::array<uint16_t, 3> wg,
    LaunchGeometry& out) {
    for (size_t axis = 0; axis < 3; ++axis) {
        if (grid_items[axis] == 0 || wg[axis] == 0) {
            return PacketErrorCode::ZeroDimension;
        }
        if (static_cast<uint32_t>(wg[axis]) > grid_items[axis]) {
            return PacketErrorCode::WorkgroupExceedsGrid;
        }
    }
    const uint16_t dims = (grid_items[2] != 1 || wg[2] != 1)   ? 3
                          : (grid_items[1] != 1 || wg[1] != 1) ? 2
                                                               : 1;
    if (dims < 3 && (grid_items[2] != 1 || wg[2] != 1)) {
        return PacketErrorCode::InactiveDimensionNotOne;
    }
    if (dims < 2 && (grid_items[1] != 1 || wg[1] != 1)) {
        return PacketErrorCode::InactiveDimensionNotOne;
    }
    out.grid_workitems = grid_items;
    out.workgroup = wg;
    out.dimensions = dims;
    return PacketErrorCode::Ok;
}

PacketErrorCode LaunchGeometry::from_workgroups(
    std::array<uint32_t, 3> workgroups,
    std::array<uint16_t, 3> wg,
    LaunchGeometry& out) {
    std::array<uint32_t, 3> grid_items = {0, 0, 0};
    for (size_t axis = 0; axis < 3; ++axis) {
        const uint64_t prod = static_cast<uint64_t>(workgroups[axis]) * static_cast<uint64_t>(wg[axis]);
        if (prod > std::numeric_limits<uint32_t>::max()) {
            return PacketErrorCode::GridDimensionOverflow;
        }
        grid_items[axis] = static_cast<uint32_t>(prod);
    }
    return from_workitems(grid_items, wg, out);
}

PacketErrorCode LaunchGeometry::from_hip_workgroups(
    std::array<uint32_t, 3> workgroups,
    std::array<uint32_t, 3> wg,
    LaunchGeometry& out) {
    if (workgroups[1] > kGfx12MaxGridYZWorkgroups || workgroups[2] > kGfx12MaxGridYZWorkgroups) {
        return PacketErrorCode::GridYZWrapExceedsLimit;
    }
    if (wg[0] > 0xffffu || wg[1] > 0xffffu || wg[2] > 0xffffu) {
        return PacketErrorCode::WorkgroupExceedsGrid;
    }
    const PacketErrorCode err =
        from_workgroups(workgroups, {static_cast<uint16_t>(wg[0]), static_cast<uint16_t>(wg[1]),
                                     static_cast<uint16_t>(wg[2])}, out);
    if (err != PacketErrorCode::Ok) return err;
    out.dimensions = 3;
    return PacketErrorCode::Ok;
}

const char* packet_error_to_string(PacketErrorCode code) noexcept {
    switch (code) {
        case PacketErrorCode::Ok: return "ok";
        case PacketErrorCode::ZeroDimension: return "zero-dimension";
        case PacketErrorCode::WorkgroupExceedsGrid: return "workgroup-exceeds-grid";
        case PacketErrorCode::InactiveDimensionNotOne: return "inactive-dimension-not-one";
        case PacketErrorCode::GridDimensionOverflow: return "grid-dimension-overflow";
        case PacketErrorCode::GridYZWrapExceedsLimit: return "grid-yz-wrap-exceeds-limit";
        case PacketErrorCode::NullKernelObject: return "null-kernel-object";
        case PacketErrorCode::NullKernarg: return "null-kernarg";
        case PacketErrorCode::DynamicCallstackNeedsPrivateSize: return "dynamic-callstack-needs-private-size";
        case PacketErrorCode::DynamicGroupSegmentOverflow: return "dynamic-group-segment-overflow";
        case PacketErrorCode::TooManyBarrierDependencies: return "too-many-barrier-dependencies";
        case PacketErrorCode::InvalidIndirectBufferAddress: return "invalid-indirect-buffer-address";
        case PacketErrorCode::InvalidIndirectBufferSize: return "invalid-indirect-buffer-size";
    }
    return "unknown";
}

// ---- KernelDispatchPacket & BarrierAndPacket (packet.rs:242-396) ----

PacketErrorCode KernelDispatchPacket::build(
    const KernelMetadata& metadata,
    const LaunchGeometry& geometry,
    uint32_t dynamic_group_bytes,
    uint64_t kernarg_address,
    uint64_t completion_signal,
    KernelDispatchPacket& out,
    HeaderPolicy policy) {
    if (metadata.kernel_object == 0) {
        return PacketErrorCode::NullKernelObject;
    }
    if (metadata.dynamic_callstack) {
        return PacketErrorCode::DynamicCallstackNeedsPrivateSize;
    }
    if (metadata.kernarg_segment_size != 0 && kernarg_address == 0) {
        return PacketErrorCode::NullKernarg;
    }
    const uint64_t group_total =
        static_cast<uint64_t>(metadata.group_segment_size) + static_cast<uint64_t>(dynamic_group_bytes);
    if (group_total > std::numeric_limits<uint32_t>::max()) {
        return PacketErrorCode::DynamicGroupSegmentOverflow;
    }
    out = KernelDispatchPacket{};
    out.full_header = encode_aql_dispatch_header_word(geometry.dimensions, policy);
    out.workgroup_size_x = geometry.workgroup[0];
    out.workgroup_size_y = geometry.workgroup[1];
    out.workgroup_size_z = geometry.workgroup[2];
    out.reserved0 = 0;
    out.grid_size_x = geometry.grid_workitems[0];
    out.grid_size_y = geometry.grid_workitems[1];
    out.grid_size_z = geometry.grid_workitems[2];
    out.private_segment_size = metadata.private_segment_size;
    out.group_segment_size = static_cast<uint32_t>(group_total);
    out.kernel_object = metadata.kernel_object;
    out.kernarg_address = kernarg_address;
    out.reserved2 = 0;
    out.completion_signal = completion_signal;
    return PacketErrorCode::Ok;
}

PacketErrorCode BarrierAndPacket::build(
    const uint64_t* dependencies,
    size_t dependency_count,
    uint64_t completion_signal,
    BarrierAndPacket& out,
    HeaderPolicy policy) {
    if (dependency_count > kBarrierDependencyCapacity) {
        return PacketErrorCode::TooManyBarrierDependencies;
    }
    out = BarrierAndPacket{};
    out.full_header = static_cast<uint32_t>(encode_aql_header(kPacketTypeBarrierAnd, policy));
    out.reserved1 = 0;
    for (size_t i = 0; i < dependency_count; ++i) {
        out.dependency_signals[i] = dependencies[i];
    }
    out.reserved2 = 0;
    out.completion_signal = completion_signal;
    return PacketErrorCode::Ok;
}

PacketImage PacketImage::from_kernel(const KernelDispatchPacket& packet) {
    PacketImage img{};
    std::memcpy(img.bytes.data(), &packet, kAqlPacketBytes);
    img.header_word = packet.published_header_word();
    return img;
}

PacketImage PacketImage::from_barrier(const BarrierAndPacket& packet) {
    PacketImage img{};
    std::memcpy(img.bytes.data(), &packet, kAqlPacketBytes);
    img.header_word = packet.published_header_word();
    return img;
}

// Port note: ported from warpfront/hipfire redline-rocr/src/packet.rs:418-462 (Apache-2.0)
PacketErrorCode PacketImage::pm4_indirect_buffer(
    uint64_t ib_address,
    uint32_t ib_dwords,
    uint64_t completion_signal,
    PacketImage& out) {
    if (ib_address == 0 || (ib_address & 3u) != 0) {
        return PacketErrorCode::InvalidIndirectBufferAddress;
    }
    if (ib_dwords == 0 || ib_dwords > kIbMaxDwords) {
        return PacketErrorCode::InvalidIndirectBufferSize;
    }
    const uint32_t pm4_header = (3u << 30) | (2u << 16) | (kPacket3IndirectBuffer << 8);
    const uint16_t aql_header = encode_aql_header(
        kPacketTypeVendorSpecific,
        HeaderPolicy{true, FenceScope::System, FenceScope::System});

    out = PacketImage{};
    write_le16(out.bytes.data() + 0, aql_header);
    write_le16(out.bytes.data() + 2, kVendorPacketFormatPm4Ib);
    write_le32(out.bytes.data() + 4, pm4_header);
    write_le32(out.bytes.data() + 8, static_cast<uint32_t>(ib_address) & 0xfffffffcu);
    write_le32(out.bytes.data() + 12, static_cast<uint32_t>(ib_address >> 32));
    write_le32(out.bytes.data() + 16, ib_dwords | kIbValidBit | kIbTemporalLu);
    write_le32(out.bytes.data() + 20, kVendorPm4IbTrailerDw5);
    write_le64(out.bytes.data() + 56, completion_signal);
    out.header_word = static_cast<uint32_t>(aql_header) |
                      (static_cast<uint32_t>(kVendorPacketFormatPm4Ib) << 16);
    return PacketErrorCode::Ok;
}

// ---- Gfx12Pm4CommandBuffer (pm4.rs:106-568) ----
// Port note: ported from warpfront/hipfire redline-rocr (Apache-2.0)

void Gfx12Pm4CommandBuffer::emit_nop_body(uint32_t body_dwords) {
    if (body_dwords == 0) return;
    const uint32_t clamped = std::min(body_dwords, kPacket3MaxBodyDwords);
    dwords_.push_back(packet3(kPacket3Nop, clamped, false));
    dwords_.insert(dwords_.end(), clamped, 0u);
    pacing_dwords_ += static_cast<uint64_t>(clamped) + 1u;
}

void Gfx12Pm4CommandBuffer::pace_after_dispatch() {
    if (post_dispatch_nop_body_ != 0) {
        emit_nop_body(post_dispatch_nop_body_);
    }
}

void Gfx12Pm4CommandBuffer::acquire_system() {
    const uint32_t words[8] = {
        packet3(kPacket3AcquireMem, 7, false),
        0u,
        0xffffffffu,
        0x000000ffu,
        0u,
        0u,
        4u,
        kGcrLegacyAcquireSystem,
    };
    dwords_.insert(dwords_.end(), std::begin(words), std::end(words));
}

void Gfx12Pm4CommandBuffer::acquire_system_gfx12(bool wait_idle) {
    if (wait_idle) wait_compute_idle();
    emit_acquire_gcr(kGcrSystemGfx12);
}

void Gfx12Pm4CommandBuffer::acquire_inter_node_gfx12() {
    acquire_rmw_gfx12(Gfx12RmwAcquirePolicy::CurrentSequential);
}

void Gfx12Pm4CommandBuffer::acquire_rmw_gfx12(Gfx12RmwAcquirePolicy policy) {
    emit_acquire_gcr(gcr_cntl_for_policy(policy));
}

void Gfx12Pm4CommandBuffer::emit_acquire_gcr(uint32_t gcr_cntl) {
    const uint32_t words[8] = {
        packet3(kPacket3AcquireMem, 7, false),
        kAcquirePrefixGfx12[0],
        kAcquirePrefixGfx12[1],
        kAcquirePrefixGfx12[2],
        kAcquirePrefixGfx12[3],
        kAcquirePrefixGfx12[4],
        kAcquirePrefixGfx12[5],
        gcr_cntl,
    };
    dwords_.insert(dwords_.end(), std::begin(words), std::end(words));
}

void Gfx12Pm4CommandBuffer::dependency_rmw_same_agent_gfx12() {
    wait_compute_idle();
    acquire_rmw_gfx12(Gfx12RmwAcquirePolicy::SameAgentParallelL1);
}

void Gfx12Pm4CommandBuffer::dependency_rmw_hip_llvm_vmem_gfx12() {
    wait_compute_idle();
    acquire_rmw_gfx12(Gfx12RmwAcquirePolicy::HipLlvmVmemL1);
}

void Gfx12Pm4CommandBuffer::emit_boundary(BoundaryVisibility vis) {
    if (vis.wait_idle) {
        wait_compute_idle();
    }
    switch (vis.acquire) {
        case AcquireRung::None:
            break;
        case AcquireRung::InterNode:
            acquire_inter_node_gfx12();
            break;
        case AcquireRung::System:
            acquire_system_gfx12();
            break;
    }
}

void Gfx12Pm4CommandBuffer::wait_compute_idle() {
    dwords_.push_back(packet3(kPacket3EventWrite, 1, false));
    dwords_.push_back(kEventWriteCsPartialFlush);
}

void Gfx12Pm4CommandBuffer::emit_set_sh_regs_raw(
    uint32_t first_reg,
    const uint32_t* values,
    size_t count) {
    if (count == 0) return;
    dwords_.push_back(packet3(kPacket3SetShReg, static_cast<uint32_t>(1 + count), true));
    dwords_.push_back(first_reg);
    dwords_.insert(dwords_.end(), values, values + count);
}

void Gfx12Pm4CommandBuffer::set_sh_regs(
    uint32_t first_reg,
    const uint32_t* values,
    size_t count) {
    if (count == 0) return;
    const bool is_static_reg = (first_reg == kComputeResourceLimits);
    if (cache_mode_ == Gfx12RegisterCacheMode::Stateless ||
        (cache_mode_ == Gfx12RegisterCacheMode::StatefulStaticOnly && !is_static_reg)) {
        emit_set_sh_regs_raw(first_reg, values, count);
        return;
    }

    std::vector<std::pair<uint32_t, std::vector<uint32_t>>> changed_runs;
    bool has_run = false;
    uint32_t run_first = 0;
    std::vector<uint32_t> run_values;

    for (size_t offset = 0; offset < count; ++offset) {
        const uint32_t reg = first_reg + static_cast<uint32_t>(offset);
        const uint32_t val = values[offset];
        const auto it = register_state_.find(reg);
        if (it != register_state_.end() && it->second == val) {
            if (has_run) {
                changed_runs.emplace_back(run_first, std::move(run_values));
                run_values.clear();
                has_run = false;
            }
            continue;
        }
        register_state_[reg] = val;
        if (!has_run) {
            run_first = reg;
            has_run = true;
        }
        run_values.push_back(val);
    }
    if (has_run) {
        changed_runs.emplace_back(run_first, std::move(run_values));
    }
    for (const auto& run : changed_runs) {
        emit_set_sh_regs_raw(run.first, run.second.data(), run.second.size());
    }
}

void Gfx12Pm4CommandBuffer::emit_dispatch_direct(
    std::array<uint32_t, 3> grid_workitems,
    bool wave32) {
    dwords_.push_back(packet3(kPacket3DispatchDirect, 4, true));
    dwords_.push_back(grid_workitems[0]);
    dwords_.push_back(grid_workitems[1]);
    dwords_.push_back(grid_workitems[2]);
    dwords_.push_back(dispatch_initiator(wave32));
    pace_after_dispatch();
}

Pm4BuildError Gfx12Pm4CommandBuffer::dispatch(
    const KernelMetadata& loader,
    const KernelPm4Metadata& pm4,
    const LaunchGeometry& geometry,
    uint32_t dynamic_group_bytes,
    uint64_t kernarg_address) {
    if (loader.private_segment_size != 0 || loader.dynamic_callstack) {
        return Pm4BuildError::ScratchUnsupported;
    }
    if (pm4.code_entry == 0) {
        return Pm4BuildError::MissingKernelDescriptor;
    }
    const uint16_t unsupported =
        static_cast<uint16_t>(pm4.kernel_code_properties & ~kSupportedKernelProperties);
    if (unsupported != 0) {
        return Pm4BuildError::UnsupportedKernelProperties;
    }
    const bool wave32 = (pm4.kernel_code_properties & kEnableWavefrontSize32) != 0;
    const bool needs_kernarg = (pm4.kernel_code_properties & kEnableSgprKernargSegmentPtr) != 0;
    if (needs_kernarg && kernarg_address == 0) {
        return Pm4BuildError::NullKernarg;
    }

    const uint64_t total_group_bytes64 =
        static_cast<uint64_t>(loader.group_segment_size) + static_cast<uint64_t>(dynamic_group_bytes);
    if (total_group_bytes64 > std::numeric_limits<uint32_t>::max()) {
        return Pm4BuildError::GroupSegmentOverflow;
    }
    const uint32_t total_group_bytes = static_cast<uint32_t>(total_group_bytes64);
    const uint32_t lds_blocks =
        (total_group_bytes + kGfx12LdsGranule - 1u) / kGfx12LdsGranule;
    if (lds_blocks > (kLdsSizeMask >> kLdsSizeShift)) {
        return Pm4BuildError::GroupSegmentTooLarge;
    }
    const uint32_t rsrc2 =
        (pm4.compute_pgm_rsrc2 & ~kLdsSizeMask) | (lds_blocks << kLdsSizeShift);

    const uint32_t pgm_words[2] = {
        static_cast<uint32_t>(pm4.code_entry >> 8),
        static_cast<uint32_t>(pm4.code_entry >> 40),
    };
    set_sh_regs(kComputePgmLo, pgm_words, 2);

    const uint32_t rsrc12_words[2] = {pm4.compute_pgm_rsrc1, rsrc2};
    set_sh_regs(kComputePgmRsrc1, rsrc12_words, 2);

    const uint32_t rsrc3_word[1] = {pm4.compute_pgm_rsrc3};
    set_sh_regs(kComputePgmRsrc3Gfx12, rsrc3_word, 1);

    const uint32_t num_thread_words[3] = {
        static_cast<uint32_t>(geometry.workgroup[0]),
        static_cast<uint32_t>(geometry.workgroup[1]),
        static_cast<uint32_t>(geometry.workgroup[2]),
    };
    set_sh_regs(kComputeNumThreadX, num_thread_words, 3);

    const uint32_t limits_word[1] = {kComputeResourceLimitsDefault};
    set_sh_regs(kComputeResourceLimits, limits_word, 1);

    if (needs_kernarg) {
        const uint32_t kernarg_words[2] = {
            static_cast<uint32_t>(kernarg_address),
            static_cast<uint32_t>(kernarg_address >> 32),
        };
        set_sh_regs(kComputeUserData0, kernarg_words, 2);
    }

    emit_dispatch_direct(geometry.grid_workitems, wave32);
    return Pm4BuildError::Ok;
}

std::vector<uint8_t> Gfx12Pm4CommandBuffer::as_bytes() const {
    std::vector<uint8_t> out(dwords_.size() * 4u);
    for (size_t i = 0; i < dwords_.size(); ++i) {
        write_le32(out.data() + i * 4u, dwords_[i]);
    }
    return out;
}

// ---- CPU-only GFX12 PM4 stream decoder / encoder (pm4_decode.rs) ----

bool DecodedPm4Packet::operator==(const DecodedPm4Packet& o) const {
    if (kind != o.kind) return false;
    switch (kind) {
        case Pm4PacketKind::SetShRegs:
            return first_reg == o.first_reg && reg_values == o.reg_values;
        case Pm4PacketKind::DispatchDirect:
            return grid_workitems == o.grid_workitems && dispatch_initiator == o.dispatch_initiator;
        case Pm4PacketKind::EventComputeIdle:
            return true;
        case Pm4PacketKind::Acquire:
            return acquire_scope == o.acquire_scope && acquire_gcr == o.acquire_gcr;
        case Pm4PacketKind::Nop:
            return nop_body_dwords == o.nop_body_dwords;
    }
    return false;
}

Pm4ParseError parse_pm4_gfx12(
    const uint32_t* words,
    size_t word_count,
    std::vector<DecodedPm4Packet>& out_packets,
    bool allow_nop) {
    out_packets.clear();
    size_t at = 0;
    auto reject = [](size_t dword, const char* reason) -> Pm4ParseError {
        return Pm4ParseError{false, dword, reason};
    };

    while (at < word_count) {
        const uint32_t header = words[at];
        if ((header >> 30) != 3u) {
            return reject(at, "not a PACKET3 header");
        }
        const size_t count = static_cast<size_t>((header >> 16) & 0x3fffu) + 1u;
        const uint32_t opcode = (header >> 8) & 0xffu;
        uint32_t expected_compute = 0u;
        if (opcode == kPacket3SetShReg || opcode == kPacket3DispatchDirect) {
            expected_compute = 2u;
        } else if (opcode == kPacket3EventWrite || opcode == kPacket3AcquireMem ||
                   (allow_nop && opcode == kPacket3Nop)) {
            expected_compute = 0u;
        } else {
            return reject(at, "unsupported PM4 opcode");
        }
        if ((header & 0xffu) != expected_compute) {
            return reject(at, "reserved header bits or wrong compute bit");
        }
        if (count > word_count - at - 1u) {
            return reject(at, "truncated packet");
        }
        const uint32_t* body = words + at + 1u;
        const size_t end = at + 1u + count;

        DecodedPm4Packet pkt{};
        if (opcode == kPacket3SetShReg) {
            if (count < 2u) {
                return reject(at, "SET_SH_REG needs a register and values");
            }
            const uint32_t first = body[0];
            if (first < kShRegFirstGfx12 || first > kShRegLastGfx12) {
                return reject(at + 1u, "register outside gfx12 SH window");
            }
            if (first + static_cast<uint32_t>(count - 2u) > kShRegLastGfx12) {
                return reject(at, "register run exceeds gfx12 SH window");
            }
            pkt.kind = Pm4PacketKind::SetShRegs;
            pkt.first_reg = first;
            pkt.reg_values.assign(body + 1u, body + count);
        } else if (opcode == kPacket3DispatchDirect) {
            if (count != 4u) {
                return reject(at, "DISPATCH_DIRECT requires four body dwords");
            }
            if (body[0] == 0u || body[1] == 0u || body[2] == 0u) {
                return reject(at, "zero dispatch dimension");
            }
            const uint32_t init = body[3];
            if (init != kDispatchInitiatorBase &&
                init != (kDispatchInitiatorBase | kDispatchInitiatorCsW32En)) {
                return reject(at + 4u, "unsupported dispatch initiator");
            }
            pkt.kind = Pm4PacketKind::DispatchDirect;
            pkt.grid_workitems = {body[0], body[1], body[2]};
            pkt.dispatch_initiator = init;
        } else if (opcode == kPacket3EventWrite) {
            if (count != 1u || body[0] != kEventWriteCsPartialFlush) {
                return reject(at, "not a CS_PARTIAL_FLUSH event");
            }
            pkt.kind = Pm4PacketKind::EventComputeIdle;
        } else if (opcode == kPacket3AcquireMem) {
            if (count != 7u) {
                return reject(at, "unsupported ACQUIRE_MEM shape");
            }
            for (size_t i = 0; i < 6u; ++i) {
                if (body[i] != kAcquirePrefixGfx12[i]) {
                    return reject(at, "unsupported ACQUIRE_MEM prefix");
                }
            }
            AcquireRung scope = AcquireRung::None;
            if (body[6] == kGcrSystemGfx12) {
                scope = AcquireRung::System;
            } else if (body[6] == kGcrInterNodeGfx12) {
                scope = AcquireRung::InterNode;
            } else {
                return reject(at + 7u, "unsupported ACQUIRE_MEM GCR scope");
            }
            pkt.kind = Pm4PacketKind::Acquire;
            pkt.acquire_scope = scope;
            pkt.acquire_gcr = body[6];
        } else if (opcode == kPacket3Nop) {
            for (size_t i = 0; i < count; ++i) {
                if (body[i] != 0u) {
                    return reject(at + 1u + i, "non-zero NOP body dword");
                }
            }
            pkt.kind = Pm4PacketKind::Nop;
            pkt.nop_body_dwords = static_cast<uint32_t>(count);
        }

        out_packets.push_back(std::move(pkt));
        at = end;
    }
    return Pm4ParseError{true, 0, ""};
}

Pm4BuildError encode_pm4_gfx12(
    const std::vector<DecodedPm4Packet>& packets,
    std::vector<uint32_t>& out_words,
    bool allow_nop) {
    out_words.clear();
    for (const auto& pkt : packets) {
        switch (pkt.kind) {
            case Pm4PacketKind::SetShRegs: {
                if (pkt.first_reg < kShRegFirstGfx12 || pkt.first_reg > kShRegLastGfx12 ||
                    pkt.reg_values.empty() ||
                    pkt.reg_values.size() > static_cast<size_t>(kShRegLastGfx12 - pkt.first_reg + 1u)) {
                    return Pm4BuildError::MalformedStream;
                }
                out_words.push_back(
                    packet3(kPacket3SetShReg, static_cast<uint32_t>(1u + pkt.reg_values.size()), true));
                out_words.push_back(pkt.first_reg);
                out_words.insert(out_words.end(), pkt.reg_values.begin(), pkt.reg_values.end());
                break;
            }
            case Pm4PacketKind::DispatchDirect: {
                if (pkt.grid_workitems[0] == 0u || pkt.grid_workitems[1] == 0u ||
                    pkt.grid_workitems[2] == 0u) {
                    return Pm4BuildError::MalformedStream;
                }
                if (pkt.dispatch_initiator != kDispatchInitiatorBase &&
                    pkt.dispatch_initiator != (kDispatchInitiatorBase | kDispatchInitiatorCsW32En)) {
                    return Pm4BuildError::MalformedStream;
                }
                out_words.push_back(packet3(kPacket3DispatchDirect, 4u, true));
                out_words.push_back(pkt.grid_workitems[0]);
                out_words.push_back(pkt.grid_workitems[1]);
                out_words.push_back(pkt.grid_workitems[2]);
                out_words.push_back(pkt.dispatch_initiator);
                break;
            }
            case Pm4PacketKind::EventComputeIdle: {
                out_words.push_back(packet3(kPacket3EventWrite, 1u, false));
                out_words.push_back(kEventWriteCsPartialFlush);
                break;
            }
            case Pm4PacketKind::Acquire: {
                const uint32_t expected_gcr =
                    (pkt.acquire_scope == AcquireRung::System)      ? kGcrSystemGfx12
                    : (pkt.acquire_scope == AcquireRung::InterNode) ? kGcrInterNodeGfx12
                                                                    : 0u;
                if (expected_gcr == 0u || pkt.acquire_gcr != expected_gcr) {
                    return Pm4BuildError::MalformedStream;
                }
                out_words.push_back(packet3(kPacket3AcquireMem, 7u, false));
                out_words.insert(
                    out_words.end(), kAcquirePrefixGfx12.begin(), kAcquirePrefixGfx12.end());
                out_words.push_back(pkt.acquire_gcr);
                break;
            }
            case Pm4PacketKind::Nop: {
                if (!allow_nop || pkt.nop_body_dwords == 0u ||
                    pkt.nop_body_dwords > kPacket3MaxBodyDwords) {
                    return Pm4BuildError::MalformedStream;
                }
                out_words.push_back(packet3(kPacket3Nop, pkt.nop_body_dwords, false));
                out_words.insert(out_words.end(), pkt.nop_body_dwords, 0u);
                break;
            }
        }
    }
    return Pm4BuildError::Ok;
}

// ---- AMDGPU ELF64 .kd descriptor & clang offload bundle helpers (runtime.rs:3084-3310) ----

bool parse_kernel_descriptor_bytes(
    const uint8_t* descriptor_64b,
    size_t len,
    uint64_t loaded_descriptor_va,
    KernelPm4Metadata& out,
    uint32_t* out_group_fixed_size,
    uint32_t* out_private_fixed_size) {
    if (descriptor_64b == nullptr || len < 64u || loaded_descriptor_va == 0u) {
        return false;
    }
    uint32_t group_fixed = 0, private_fixed = 0;
    uint64_t entry_offset_u64 = 0;
    uint32_t rsrc3 = 0, rsrc1 = 0, rsrc2 = 0;
    uint16_t props = 0;
    if (!read_le32(descriptor_64b, len, 0, group_fixed) ||
        !read_le32(descriptor_64b, len, 4, private_fixed) ||
        !read_le64(descriptor_64b, len, 16, entry_offset_u64) ||
        !read_le32(descriptor_64b, len, 44, rsrc3) ||
        !read_le32(descriptor_64b, len, 48, rsrc1) ||
        !read_le32(descriptor_64b, len, 52, rsrc2) ||
        !read_le16(descriptor_64b, len, 56, props)) {
        return false;
    }
    const int64_t entry_offset = static_cast<int64_t>(entry_offset_u64);
    uint64_t code_entry = 0;
    if (entry_offset >= 0) {
        const uint64_t delta = static_cast<uint64_t>(entry_offset);
        if (loaded_descriptor_va > std::numeric_limits<uint64_t>::max() - delta) return false;
        code_entry = loaded_descriptor_va + delta;
    } else {
        const uint64_t abs_delta = static_cast<uint64_t>(-(entry_offset + 1)) + 1u;
        if (loaded_descriptor_va < abs_delta) return false;
        code_entry = loaded_descriptor_va - abs_delta;
    }
    if (code_entry == 0u) return false;

    out.code_entry = code_entry;
    out.compute_pgm_rsrc3 = rsrc3;
    out.compute_pgm_rsrc1 = rsrc1;
    out.compute_pgm_rsrc2 = rsrc2;
    out.kernel_code_properties = props;
    if (out_group_fixed_size != nullptr) *out_group_fixed_size = group_fixed;
    if (out_private_fixed_size != nullptr) *out_private_fixed_size = private_fixed;
    return true;
}

bool parse_kernel_pm4_metadata_from_elf(
    const uint8_t* elf_bytes,
    size_t elf_len,
    const char* symbol_name,
    uint64_t loaded_descriptor_va,
    KernelPm4Metadata& out) {
    constexpr size_t kElf64HeaderBytes = 64u;
    constexpr size_t kElf64SymbolBytes = 24u;
    constexpr uint32_t kPtLoad = 1u;
    constexpr uint32_t kShtSymtab = 2u;
    constexpr uint32_t kShtDynsym = 11u;

    if (elf_bytes == nullptr || symbol_name == nullptr || elf_len < kElf64HeaderBytes) {
        return false;
    }
    if (elf_bytes[0] != 0x7fu || elf_bytes[1] != 'E' || elf_bytes[2] != 'L' || elf_bytes[3] != 'F') {
        return false;
    }
    uint64_t phoff64 = 0, shoff64 = 0;
    uint16_t phentsize16 = 0, phnum16 = 0, shentsize16 = 0, shnum16 = 0;
    if (!read_le64(elf_bytes, elf_len, 32, phoff64) ||
        !read_le64(elf_bytes, elf_len, 40, shoff64) ||
        !read_le16(elf_bytes, elf_len, 54, phentsize16) ||
        !read_le16(elf_bytes, elf_len, 56, phnum16) ||
        !read_le16(elf_bytes, elf_len, 58, shentsize16) ||
        !read_le16(elf_bytes, elf_len, 60, shnum16)) {
        return false;
    }
    if (phentsize16 < 56u || shentsize16 < 64u) return false;
    const size_t phoff = static_cast<size_t>(phoff64);
    const size_t shoff = static_cast<size_t>(shoff64);
    const size_t phentsize = static_cast<size_t>(phentsize16);
    const size_t phnum = static_cast<size_t>(phnum16);
    const size_t shentsize = static_cast<size_t>(shentsize16);
    const size_t shnum = static_cast<size_t>(shnum16);

    auto va_to_offset = [&](uint64_t va, size_t& file_off_out) -> bool {
        for (size_t i = 0; i < phnum; ++i) {
            if (i > (elf_len - phoff) / phentsize) return false;
            const size_t base = phoff + i * phentsize;
            uint32_t p_type = 0;
            if (!read_le32(elf_bytes, elf_len, base, p_type)) return false;
            if (p_type != kPtLoad) continue;
            uint64_t offset = 0, vaddr = 0, filesz = 0;
            if (!read_le64(elf_bytes, elf_len, base + 8, offset) ||
                !read_le64(elf_bytes, elf_len, base + 16, vaddr) ||
                !read_le64(elf_bytes, elf_len, base + 32, filesz)) {
                return false;
            }
            if (filesz > std::numeric_limits<uint64_t>::max() - vaddr) return false;
            if (va >= vaddr && va < vaddr + filesz) {
                const uint64_t rel = va - vaddr;
                if (rel > std::numeric_limits<uint64_t>::max() - offset) return false;
                const uint64_t total = offset + rel;
                if (total > static_cast<uint64_t>(elf_len)) return false;
                file_off_out = static_cast<size_t>(total);
                return true;
            }
        }
        return false;
    };

    const size_t sym_name_len = std::strlen(symbol_name);
    bool found_descriptor_va = false;
    uint64_t descriptor_va = 0;

    for (size_t section = 0; section < shnum && !found_descriptor_va; ++section) {
        if (shoff > elf_len || section > (elf_len - shoff) / shentsize) return false;
        const size_t base = shoff + section * shentsize;
        uint32_t section_type = 0;
        if (!read_le32(elf_bytes, elf_len, base + 4, section_type)) return false;
        if (section_type != kShtSymtab && section_type != kShtDynsym) continue;

        uint64_t symbols_offset64 = 0, symbols_size64 = 0, entry_size64 = 0;
        uint32_t string_section32 = 0;
        if (!read_le64(elf_bytes, elf_len, base + 24, symbols_offset64) ||
            !read_le64(elf_bytes, elf_len, base + 32, symbols_size64) ||
            !read_le32(elf_bytes, elf_len, base + 40, string_section32) ||
            !read_le64(elf_bytes, elf_len, base + 56, entry_size64)) {
            return false;
        }
        if (entry_size64 < kElf64SymbolBytes || string_section32 >= shnum) continue;
        const size_t string_base = shoff + static_cast<size_t>(string_section32) * shentsize;
        uint64_t strings_offset64 = 0, strings_size64 = 0;
        if (!read_le64(elf_bytes, elf_len, string_base + 24, strings_offset64) ||
            !read_le64(elf_bytes, elf_len, string_base + 32, strings_size64)) {
            return false;
        }
        if (strings_offset64 > elf_len || strings_size64 > elf_len - strings_offset64) return false;
        const size_t strings_offset = static_cast<size_t>(strings_offset64);
        const size_t strings_size = static_cast<size_t>(strings_size64);

        const size_t symbols_offset = static_cast<size_t>(symbols_offset64);
        const size_t symbols_size = static_cast<size_t>(symbols_size64);
        const size_t entry_size = static_cast<size_t>(entry_size64);
        if (symbols_offset > elf_len || symbols_size > elf_len - symbols_offset) return false;

        const size_t num_syms = symbols_size / entry_size;
        for (size_t idx = 0; idx < num_syms; ++idx) {
            const size_t sym_pos = symbols_offset + idx * entry_size;
            uint32_t name_off32 = 0;
            if (!read_le32(elf_bytes, elf_len, sym_pos, name_off32)) return false;
            const size_t name_off = static_cast<size_t>(name_off32);
            if (name_off >= strings_size) return false;
            const uint8_t* str_start = elf_bytes + strings_offset + name_off;
            const size_t max_rem = strings_size - name_off;
            const void* nul_ptr = std::memchr(str_start, 0, max_rem);
            if (nul_ptr == nullptr) return false;
            const size_t candidate_len =
                static_cast<size_t>( static_cast<const uint8_t*>(nul_ptr) - str_start );
            if (candidate_len == sym_name_len &&
                std::memcmp(str_start, symbol_name, sym_name_len) == 0) {
                if (!read_le64(elf_bytes, elf_len, sym_pos + 8, descriptor_va)) return false;
                found_descriptor_va = true;
                break;
            }
        }
    }

    if (!found_descriptor_va) return false;
    size_t descriptor_offset = 0;
    if (!va_to_offset(descriptor_va, descriptor_offset)) return false;
    if (descriptor_offset > elf_len || elf_len - descriptor_offset < 64u) return false;
    return parse_kernel_descriptor_bytes(
        elf_bytes + descriptor_offset, 64u, loaded_descriptor_va, out);
}

bool unwrap_clang_offload_bundle(
    const uint8_t* image,
    size_t image_len,
    size_t& out_offset,
    size_t& out_size,
    std::string& out_error) {
    constexpr const char kClangOffloadBundleMagic[] = "__CLANG_OFFLOAD_BUNDLE__";
    constexpr size_t kMagicLen = sizeof(kClangOffloadBundleMagic) - 1u;
    constexpr const char kCompressedMagic[] = "CCOB";
    constexpr const char kAmdgpuTriple[] = "amdgcn-amd-amdhsa";
    constexpr size_t kTripleLen = sizeof(kAmdgpuTriple) - 1u;

    if (image == nullptr || image_len == 0u) {
        out_error = "HSACO code object is empty";
        return false;
    }
    if (image_len >= 4u && std::memcmp(image, kCompressedMagic, 4u) == 0) {
        out_error = "compressed bundle (CCOB) is unsupported by public HSA reader";
        return false;
    }
    if (image_len < kMagicLen || std::memcmp(image, kClangOffloadBundleMagic, kMagicLen) != 0) {
        out_offset = 0u;
        out_size = image_len;
        return true;
    }

    size_t cursor = kMagicLen;
    uint64_t bundle_count = 0;
    if (!read_le64(image, image_len, cursor, bundle_count)) {
        out_error = "truncated clang offload bundle header";
        return false;
    }
    cursor += 8u;

    bool found_amdgpu = false;
    size_t amdgpu_start = 0;
    size_t amdgpu_end = 0;

    for (uint64_t i = 0; i < bundle_count; ++i) {
        uint64_t offset64 = 0, size64 = 0, id_len64 = 0;
        if (!read_le64(image, image_len, cursor, offset64) ||
            !read_le64(image, image_len, cursor + 8u, size64) ||
            !read_le64(image, image_len, cursor + 16u, id_len64)) {
            out_error = "truncated clang offload bundle entry header";
            return false;
        }
        cursor += 24u;
        if (id_len64 > image_len - cursor) {
            out_error = "truncated clang offload bundle entry ID";
            return false;
        }
        const uint8_t* id_ptr = image + cursor;
        const size_t id_len = static_cast<size_t>(id_len64);
        cursor += id_len;

        if (offset64 > image_len || size64 > image_len - offset64) {
            out_error = "bundle entry payload exceeds bundle";
            return false;
        }
        bool has_triple = false;
        if (id_len >= kTripleLen) {
            for (size_t pos = 0; pos + kTripleLen <= id_len; ++pos) {
                if (std::memcmp(id_ptr + pos, kAmdgpuTriple, kTripleLen) == 0) {
                    has_triple = true;
                    break;
                }
            }
        }
        if (has_triple) {
            if (found_amdgpu) {
                out_error = "multiple AMDGPU code objects in bundle require an explicit target";
                return false;
            }
            found_amdgpu = true;
            amdgpu_start = static_cast<size_t>(offset64);
            amdgpu_end = static_cast<size_t>(offset64 + size64);
        }
    }

    if (!found_amdgpu) {
        out_error = "bundle contains no AMDGPU code object";
        return false;
    }
    out_offset = amdgpu_start;
    out_size = amdgpu_end - amdgpu_start;
    return true;
}

}  // namespace strata::hip::pm4

#endif  // (__linux__ && STRATA_WITH_HSA) || STRATA_PM4_HOST_TEST
