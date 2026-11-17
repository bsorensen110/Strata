// src/hip/pm4_test.cpp - Host-side unit test for AQL packet and GFX12 PM4 command codec.
//
// Port note: from warpfront/hipfire redline-rocr (Apache-2.0)
// Golden-bytes and decode round-trip tests ported from warpfront/hipfire
// crates/redline-rocr/src/{packet.rs,pm4.rs,pm4_decode.rs,runtime.rs}.

#ifndef STRATA_PM4_HOST_TEST
#define STRATA_PM4_HOST_TEST 1
#endif

#include "strata/hip/pm4.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

static int g_fails = 0;

#define CHECK(c)                                                           \
    do {                                                                   \
        if (!(c)) {                                                        \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);       \
            ++g_fails;                                                     \
        }                                                                  \
    } while (0)

namespace {

using namespace strata::hip::pm4;

std::string resolve_fixture_path(const char* rel_path, const char* arg_dir) {
    std::vector<std::string> candidates;
    if (arg_dir != nullptr && arg_dir[0] != '\0') {
        candidates.push_back(std::string(arg_dir) + "/" + rel_path);
    }
    candidates.push_back(std::string("data/pm4/") + rel_path);
    candidates.push_back(std::string("../data/pm4/") + rel_path);
    candidates.push_back(std::string("../../data/pm4/") + rel_path);
    for (const auto& path : candidates) {
        std::ifstream f(path);
        if (f.good()) return path;
    }
    return std::string("data/pm4/") + rel_path;
}

bool load_golden_sections(
    const std::string& path,
    std::map<std::string, std::vector<uint32_t>>& out) {
    std::ifstream in(path);
    if (!in.is_open()) return false;
    std::string current_section;
    std::string line;
    while (std::getline(in, line)) {
        const size_t hash_pos = line.find('#');
        if (hash_pos != std::string::npos) {
            line.resize(hash_pos);
        }
        std::istringstream iss(line);
        std::string tok;
        while (iss >> tok) {
            if (tok.size() >= 2 && tok.front() == '[' && tok.back() == ']') {
                current_section = tok.substr(1, tok.size() - 2);
                out[current_section];
            } else if (!current_section.empty()) {
                char* endp = nullptr;
                const unsigned long val = std::strtoul(tok.c_str(), &endp, 0);
                if (endp == tok.c_str() || *endp != '\0') return false;
                out[current_section].push_back(static_cast<uint32_t>(val));
            }
        }
    }
    return !out.empty();
}

std::vector<uint32_t> bytes_to_dwords(const uint8_t* bytes, size_t byte_len) {
    std::vector<uint32_t> out(byte_len / 4u, 0u);
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<uint32_t>(bytes[i * 4u + 0u]) |
                 (static_cast<uint32_t>(bytes[i * 4u + 1u]) << 8) |
                 (static_cast<uint32_t>(bytes[i * 4u + 2u]) << 16) |
                 (static_cast<uint32_t>(bytes[i * 4u + 3u]) << 24);
    }
    return out;
}

void test_aql_packets(const std::map<std::string, std::vector<uint32_t>>& aql_golden) {
    // 1. Vendor PM4-IB AQL packet golden comparison (packet.rs:669-688)
    PacketImage vendor{};
    CHECK(PacketImage::pm4_indirect_buffer(
              0x123456789000ULL, 0x321u, 0x5566778899aabbccULL, vendor) ==
          PacketErrorCode::Ok);
    CHECK(vendor.header_word == 0x00011500u);
    const auto vendor_dwords = bytes_to_dwords(vendor.bytes.data(), vendor.bytes.size());
    auto it_vendor = aql_golden.find("pm4_ib_vendor_packet");
    CHECK(it_vendor != aql_golden.end());
    if (it_vendor != aql_golden.end()) {
        CHECK(vendor_dwords == it_vendor->second);
    }

    // Invalid IB address / size must fail closed
    PacketImage bad_vendor{};
    CHECK(PacketImage::pm4_indirect_buffer(0ULL, 16u, 1ULL, bad_vendor) ==
          PacketErrorCode::InvalidIndirectBufferAddress);
    CHECK(PacketImage::pm4_indirect_buffer(0x1002ULL, 16u, 1ULL, bad_vendor) ==
          PacketErrorCode::InvalidIndirectBufferAddress);
    CHECK(PacketImage::pm4_indirect_buffer(0x1000ULL, 0u, 1ULL, bad_vendor) ==
          PacketErrorCode::InvalidIndirectBufferSize);
    CHECK(PacketImage::pm4_indirect_buffer(0x1000ULL, 0x00100000u, 1ULL, bad_vendor) ==
          PacketErrorCode::InvalidIndirectBufferSize);

    // 2. KernelDispatchPacket golden comparison & headers (packet.rs:629-649)
    LaunchGeometry geom1d{};
    CHECK(LaunchGeometry::from_workitems({1024, 1, 1}, {64, 1, 1}, geom1d) ==
          PacketErrorCode::Ok);
    CHECK(geom1d.dimensions == 1);

    const KernelMetadata meta = {
        /*kernel_object=*/7ULL,
        /*kernarg_segment_size=*/0u,
        /*kernarg_segment_alignment=*/16u,
        /*group_segment_size=*/32u,
        /*private_segment_size=*/64u,
        /*dynamic_callstack=*/false,
    };
    KernelDispatchPacket kpkt{};
    CHECK(KernelDispatchPacket::build(meta, geom1d, 0u, 0ULL, 9ULL, kpkt) ==
          PacketErrorCode::Ok);
    CHECK((kpkt.published_header_word() & 0xffffu) == 0x1502u);
    CHECK((kpkt.published_header_word() >> 16) == 1u);
    CHECK(kpkt.group_segment_size == 32u);

    KernelDispatchPacket kpkt_dyn{};
    CHECK(KernelDispatchPacket::build(meta, geom1d, 16u, 0ULL, 9ULL, kpkt_dyn) ==
          PacketErrorCode::Ok);
    CHECK(kpkt_dyn.group_segment_size == 48u);
    const PacketImage kimg = PacketImage::from_kernel(kpkt_dyn);
    const auto kdwords = bytes_to_dwords(kimg.bytes.data(), kimg.bytes.size());
    auto it_kd = aql_golden.find("kernel_dispatch_1d_dynamic_lds");
    CHECK(it_kd != aql_golden.end());
    if (it_kd != aql_golden.end()) {
        CHECK(kdwords == it_kd->second);
    }

    // 3. BarrierAndPacket golden comparison (packet.rs:651-666)
    const uint64_t deps[2] = {11ULL, 12ULL};
    BarrierAndPacket bpkt{};
    CHECK(BarrierAndPacket::build(deps, 2, 13ULL, bpkt) == PacketErrorCode::Ok);
    CHECK(bpkt.published_header_word() == 0x1503u);
    const PacketImage bimg = PacketImage::from_barrier(bpkt);
    const auto bdwords = bytes_to_dwords(bimg.bytes.data(), bimg.bytes.size());
    auto it_bar = aql_golden.find("barrier_and_packet");
    CHECK(it_bar != aql_golden.end());
    if (it_bar != aql_golden.end()) {
        CHECK(bdwords == it_bar->second);
    }

    // 4. Header policies & HIP geometry wrap guard (packet.rs:691-773)
    CHECK(encode_aql_header(kPacketTypeKernelDispatch, HeaderPolicy::recorded_dispatch()) == 0x1502u);
    CHECK(encode_aql_header(kPacketTypeKernelDispatch, HeaderPolicy::two_queue_dispatch()) == 0x0d02u);
    CHECK(encode_aql_header(kPacketTypeKernelDispatch, HeaderPolicy::batch_boundary_first_serial()) == 0x0502u);
    CHECK(encode_aql_header(kPacketTypeKernelDispatch, HeaderPolicy::batch_boundary_internal_serial()) == 0x0102u);
    CHECK(encode_aql_header(kPacketTypeKernelDispatch, HeaderPolicy::batch_boundary_first_independent()) == 0x0402u);
    CHECK(encode_aql_header(kPacketTypeKernelDispatch, HeaderPolicy::batch_boundary_internal_independent()) == 0x0002u);
    CHECK(encode_aql_header(kPacketTypeBarrierAnd, HeaderPolicy::two_queue_dependency()) == 0x0103u);
    CHECK(encode_aql_header(kPacketTypeBarrierAnd, HeaderPolicy::two_queue_host_terminal()) == 0x1103u);

    LaunchGeometry hip_geom{};
    CHECK(LaunchGeometry::from_hip_workgroups({16, 1, 1}, {256, 1, 1}, hip_geom) ==
          PacketErrorCode::Ok);
    CHECK(hip_geom.dimensions == 3);
    CHECK(hip_geom.grid_workitems[0] == 4096u);
    // gfx1201 grid.y/z <= 65536 wrap guard
    CHECK(LaunchGeometry::from_hip_workgroups({1, 65536, 1}, {64, 1, 1}, hip_geom) ==
          PacketErrorCode::Ok);
    CHECK(LaunchGeometry::from_hip_workgroups({1, 65537, 1}, {64, 1, 1}, hip_geom) ==
          PacketErrorCode::GridYZWrapExceedsLimit);
    CHECK(LaunchGeometry::from_hip_workgroups({1, 1, 65537}, {64, 1, 1}, hip_geom) ==
          PacketErrorCode::GridYZWrapExceedsLimit);
}

void test_pm4_commands_and_roundtrip(
    const std::map<std::string, std::vector<uint32_t>>& pm4_golden) {
    // 1. PACKET3 header and register constants (pm4.rs:817-833)
    CHECK(packet3(kPacket3SetShReg, 3, true) == 0xc0027602u);
    CHECK(packet3(kPacket3DispatchDirect, 4, true) == 0xc0031502u);
    CHECK(packet3(kPacket3EventWrite, 1, false) == 0xc0004600u);
    CHECK(packet3(kPacket3AcquireMem, 7, false) == 0xc0065800u);
    CHECK(kComputePgmRsrc3Gfx12 == 0x228u);
    CHECK(dispatch_initiator(false) == 0x25u);
    CHECK(dispatch_initiator(true) == 0x8025u);

    // 2. Acquire and compute-idle golden check (pm4.rs:836-853)
    Gfx12Pm4CommandBuffer cmd_acq;
    cmd_acq.acquire_system();
    cmd_acq.acquire_system_gfx12();
    cmd_acq.acquire_inter_node_gfx12();
    cmd_acq.wait_compute_idle();
    auto it_acq = pm4_golden.find("acquire_and_compute_idle");
    CHECK(it_acq != pm4_golden.end());
    if (it_acq != pm4_golden.end()) {
        CHECK(cmd_acq.dwords() == it_acq->second);
    }

    // 3. RMW acquire ladder and CacheTable hazard rules (pm4.rs:856-898, railgun/src/plan.rs:121-201)
    CHECK(gcr_cntl_for_policy(Gfx12RmwAcquirePolicy::CurrentSequential) == 0x10180u);
    CHECK(gcr_cntl_for_policy(Gfx12RmwAcquirePolicy::RadvGlobal) ==
          ((1u << 7) | (1u << 8) | (1u << 9) | (1u << 14) | (1u << 15)));
    CHECK(gcr_cntl_for_policy(Gfx12RmwAcquirePolicy::SameAgentParallelL1) ==
          ((1u << 7) | (1u << 8) | (1u << 9)));
    CHECK(gcr_cntl_for_policy(Gfx12RmwAcquirePolicy::SameAgentParallelL0) ==
          ((1u << 7) | (1u << 8)));
    CHECK(gcr_cntl_for_policy(Gfx12RmwAcquirePolicy::HipLlvmVmemL1) ==
          ((1u << 8) | (1u << 9)));

    const BoundaryVisibility raw_vis = gfx12_cache_table_lookup(HazardKind::Raw);
    const BoundaryVisibility war_vis = gfx12_cache_table_lookup(HazardKind::War);
    const BoundaryVisibility waw_vis = gfx12_cache_table_lookup(HazardKind::Waw);
    const BoundaryVisibility pre_vis = gfx12_cache_table_lookup(HazardKind::WarPreWriter);
    CHECK(raw_vis.wait_idle && raw_vis.acquire == AcquireRung::InterNode && !raw_vis.pre_writer);
    CHECK(war_vis.wait_idle && war_vis.acquire == AcquireRung::InterNode && !war_vis.pre_writer);
    CHECK(waw_vis.wait_idle && waw_vis.acquire == AcquireRung::InterNode && !waw_vis.pre_writer);
    CHECK(pre_vis.wait_idle && pre_vis.acquire == AcquireRung::System && pre_vis.pre_writer);

    // 4. Decode round-trip fixture (pm4_decode.rs:291-392)
    Gfx12Pm4CommandBuffer rt_builder;
    rt_builder.acquire_system_gfx12();
    rt_builder.set_sh_regs(0x20cu, {0x1234u, 0x5678u});
    rt_builder.emit_dispatch_direct({256u, 2u, 1u}, true);
    rt_builder.wait_compute_idle();
    rt_builder.acquire_inter_node_gfx12();

    auto it_rt = pm4_golden.find("decode_roundtrip_fixture");
    CHECK(it_rt != pm4_golden.end());
    if (it_rt != pm4_golden.end()) {
        CHECK(rt_builder.dwords() == it_rt->second);
    }

    std::vector<DecodedPm4Packet> decoded;
    const Pm4ParseError perr =
        parse_pm4_gfx12(rt_builder.dwords().data(), rt_builder.dwords().size(), decoded);
    CHECK(perr.ok);
    CHECK(decoded.size() == 5u);
    if (decoded.size() == 5u) {
        CHECK(decoded[0].kind == Pm4PacketKind::Acquire);
        CHECK(decoded[0].acquire_scope == AcquireRung::System);
        CHECK(decoded[0].acquire_gcr == kGcrSystemGfx12);
        CHECK(decoded[1].kind == Pm4PacketKind::SetShRegs);
        CHECK(decoded[1].first_reg == 0x20cu);
        CHECK(decoded[1].reg_values == std::vector<uint32_t>({0x1234u, 0x5678u}));
        CHECK(decoded[2].kind == Pm4PacketKind::DispatchDirect);
        CHECK(decoded[2].grid_workitems == std::array<uint32_t, 3>({256u, 2u, 1u}));
        CHECK(decoded[2].dispatch_initiator == 0x8025u);
        CHECK(decoded[3].kind == Pm4PacketKind::EventComputeIdle);
        CHECK(decoded[4].kind == Pm4PacketKind::Acquire);
        CHECK(decoded[4].acquire_scope == AcquireRung::InterNode);
        CHECK(decoded[4].acquire_gcr == kGcrInterNodeGfx12);
    }

    std::vector<uint32_t> reencoded;
    CHECK(encode_pm4_gfx12(decoded, reencoded) == Pm4BuildError::Ok);
    CHECK(reencoded == rt_builder.dwords());
    const auto rt_bytes = rt_builder.as_bytes();
    CHECK(bytes_to_dwords(rt_bytes.data(), rt_bytes.size()) == rt_builder.dwords());

    // Negative decode checks (pm4_decode.rs:345-392)
    const size_t set_at = 8u;
    const size_t dispatch_at = 12u;
    auto expect_decode_err = [&](const std::vector<uint32_t>& bad_words) {
        std::vector<DecodedPm4Packet> tmp;
        CHECK(!parse_pm4_gfx12(bad_words.data(), bad_words.size(), tmp).ok);
    };
    {
        auto bad = rt_builder.dwords();
        bad[set_at] += (1u << 16);
        expect_decode_err(bad);
    }
    {
        auto bad = rt_builder.dwords();
        bad[set_at] = packet3(kPacket3EventWrite, 3u, false);
        expect_decode_err(bad);
    }
    {
        auto bad = rt_builder.dwords();
        bad[set_at + 1u] = 0x300u;
        expect_decode_err(bad);
    }
    {
        auto bad = rt_builder.dwords();
        bad[set_at + 1u] = 0x2ffu;
        expect_decode_err(bad);
    }
    {
        auto bad = rt_builder.dwords();
        bad[dispatch_at] = packet3(kPacket3DispatchDirect, 3u, true);
        expect_decode_err(bad);
    }
    {
        auto bad = rt_builder.dwords();
        bad[dispatch_at + 1u] = 0u;
        expect_decode_err(bad);
    }
    {
        auto bad = rt_builder.dwords();
        bad[dispatch_at + 4u] = 0x8027u;
        expect_decode_err(bad);
    }
    {
        auto bad = rt_builder.dwords();
        bad[set_at] &= ~2u;
        expect_decode_err(bad);
    }
    {
        auto bad = rt_builder.dwords();
        bad[0] &= ~(3u << 30);
        expect_decode_err(bad);
    }
    {
        auto bad = rt_builder.dwords();
        bad[3] = 0u;
        expect_decode_err(bad);
    }
    {
        auto bad = rt_builder.dwords();
        bad[7] = 0u;
        expect_decode_err(bad);
    }
    for (size_t trunc : {size_t(1), set_at + 1u, dispatch_at + 2u, rt_builder.dwords().size() - 1u}) {
        std::vector<uint32_t> bad(
            rt_builder.dwords().begin(),
            rt_builder.dwords().begin() + static_cast<std::ptrdiff_t>(trunc));
        expect_decode_err(bad);
    }

    // 5. Full kernel dispatch lowering & ELF .kd descriptor parsing (runtime.rs:3918-3974)
    constexpr size_t kPhoff = 64u;
    constexpr size_t kShoff = 120u;
    constexpr size_t kSymtab = 312u;
    constexpr size_t kStrtab = 336u;
    constexpr size_t kDescriptor = 384u;
    std::vector<uint8_t> elf(512u, 0u);
    std::memcpy(elf.data(), "\x7f" "ELF", 4u);
    elf[4] = 2u;  // ELFCLASS64
    elf[5] = 1u;  // ELFDATA2LSB
    auto put_u16 = [&](size_t off, uint16_t v) {
        elf[off] = static_cast<uint8_t>(v & 0xffu);
        elf[off + 1u] = static_cast<uint8_t>((v >> 8) & 0xffu);
    };
    auto put_u32 = [&](size_t off, uint32_t v) {
        for (size_t b = 0; b < 4u; ++b) elf[off + b] = static_cast<uint8_t>((v >> (b * 8u)) & 0xffu);
    };
    auto put_u64 = [&](size_t off, uint64_t v) {
        for (size_t b = 0; b < 8u; ++b) elf[off + b] = static_cast<uint8_t>((v >> (b * 8u)) & 0xffu);
    };
    put_u64(32, kPhoff);
    put_u64(40, kShoff);
    put_u16(54, 56u);
    put_u16(56, 1u);
    put_u16(58, 64u);
    put_u16(60, 3u);
    put_u32(kPhoff, 1u);  // PT_LOAD
    put_u64(kPhoff + 32u, elf.size());

    const size_t sym_section = kShoff + 64u;
    put_u32(sym_section + 4u, 2u);  // SHT_SYMTAB
    put_u64(sym_section + 24u, kSymtab);
    put_u64(sym_section + 32u, 24u);
    put_u32(sym_section + 40u, 2u);
    put_u64(sym_section + 56u, 24u);

    const size_t str_section = kShoff + 128u;
    put_u32(str_section + 4u, 3u);  // SHT_STRTAB
    put_u64(str_section + 24u, kStrtab);
    const char kSymName[] = "\0kernel.kd";
    put_u64(str_section + 32u, sizeof(kSymName));
    std::memcpy(elf.data() + kStrtab, kSymName, sizeof(kSymName));

    put_u32(kSymtab, 1u);
    put_u64(kSymtab + 8u, kDescriptor);

    put_u32(kDescriptor + 0u, 1024u);          // group_segment_fixed_size
    put_u32(kDescriptor + 4u, 0u);             // private_segment_fixed_size
    put_u64(kDescriptor + 16u, 0x120ULL);      // kernel_code_entry_byte_offset
    put_u32(kDescriptor + 44u, 0x150u);        // compute_pgm_rsrc3
    put_u32(kDescriptor + 48u, 0xe00f0004u);   // compute_pgm_rsrc1
    put_u32(kDescriptor + 52u, 0x84u);         // compute_pgm_rsrc2
    put_u16(kDescriptor + 56u, 0x408u);        // kernel_code_properties

    KernelPm4Metadata parsed_pm4{};
    CHECK(parse_kernel_pm4_metadata_from_elf(
        elf.data(), elf.size(), "kernel.kd", 0x7f000000ULL, parsed_pm4));
    const KernelPm4Metadata expected_pm4 = {
        /*code_entry=*/0x7f000120ULL,
        /*compute_pgm_rsrc1=*/0xe00f0004u,
        /*compute_pgm_rsrc2=*/0x84u,
        /*compute_pgm_rsrc3=*/0x150u,
        /*kernel_code_properties=*/0x408u,
    };
    CHECK(parsed_pm4 == expected_pm4);

    const KernelMetadata loader_meta = {
        /*kernel_object=*/0x7f000000ULL,
        /*kernarg_segment_size=*/64u,
        /*kernarg_segment_alignment=*/16u,
        /*group_segment_size=*/1024u,
        /*private_segment_size=*/0u,
        /*dynamic_callstack=*/false,
    };
    LaunchGeometry full_geom{};
    CHECK(LaunchGeometry::from_workitems({16384u, 2u, 1u}, {256u, 1u, 1u}, full_geom) ==
          PacketErrorCode::Ok);

    Gfx12Pm4CommandBuffer full_cmd;
    CHECK(full_cmd.dispatch(
              loader_meta, parsed_pm4, full_geom, 512u, 0x00007fff12345680ULL) ==
          Pm4BuildError::Ok);
    auto it_full = pm4_golden.find("full_kernel_dispatch_gfx12");
    CHECK(it_full != pm4_golden.end());
    if (it_full != pm4_golden.end()) {
        CHECK(full_cmd.dwords() == it_full->second);
    }

    // Round-trip decode/encode on the full dispatch stream
    std::vector<DecodedPm4Packet> full_decoded;
    CHECK(parse_pm4_gfx12(full_cmd.dwords().data(), full_cmd.dwords().size(), full_decoded).ok);
    std::vector<uint32_t> full_reencoded;
    CHECK(encode_pm4_gfx12(full_decoded, full_reencoded) == Pm4BuildError::Ok);
    CHECK(full_reencoded == full_cmd.dwords());

    // Post-dispatch NOP 64 pacing (gfx1201 ArchLowering)
    Gfx12Pm4CommandBuffer paced_cmd;
    paced_cmd.set_post_dispatch_nop(kGfx1201PostDispatchNopDwords);
    CHECK(paced_cmd.dispatch(
              loader_meta, parsed_pm4, full_geom, 512u, 0x00007fff12345680ULL) ==
          Pm4BuildError::Ok);
    CHECK(paced_cmd.pacing_dwords() == kGfx1201PostDispatchNopDwords + 1u);
    CHECK(paced_cmd.len_dwords() == full_cmd.len_dwords() + kGfx1201PostDispatchNopDwords + 1u);

    std::vector<DecodedPm4Packet> paced_decoded;
    CHECK(parse_pm4_gfx12(
              paced_cmd.dwords().data(), paced_cmd.dwords().size(), paced_decoded, /*allow_nop=*/true)
              .ok);
    std::vector<uint32_t> paced_reencoded;
    CHECK(encode_pm4_gfx12(paced_decoded, paced_reencoded, /*allow_nop=*/true) ==
          Pm4BuildError::Ok);
    CHECK(paced_reencoded == paced_cmd.dwords());

    // Stateful register caching (pm4.rs:901-947)
    Gfx12Pm4CommandBuffer stateful = Gfx12Pm4CommandBuffer::new_stateful();
    stateful.set_sh_regs(0x210u, {1u, 2u, 3u, 4u});
    const uint32_t first_len = stateful.len_dwords();
    stateful.set_sh_regs(0x210u, {1u, 2u, 3u, 4u});
    CHECK(stateful.len_dwords() == first_len);
    stateful.set_sh_regs(0x210u, {5u, 2u, 6u, 4u});
    const std::vector<uint32_t> tail(
        stateful.dwords().begin() + static_cast<std::ptrdiff_t>(first_len),
        stateful.dwords().end());
    const std::vector<uint32_t> expected_tail = {
        packet3(kPacket3SetShReg, 2u, true), 0x210u, 5u,
        packet3(kPacket3SetShReg, 2u, true), 0x212u, 6u,
    };
    CHECK(tail == expected_tail);
}

}  // namespace

int main(int argc, char** argv) {
    const char* arg_dir = (argc > 1) ? argv[1] : nullptr;
    const std::string aql_path = resolve_fixture_path("aql_packets_golden.txt", arg_dir);
    const std::string pm4_path = resolve_fixture_path("gfx12_pm4_stream_golden.txt", arg_dir);

    std::map<std::string, std::vector<uint32_t>> aql_golden;
    std::map<std::string, std::vector<uint32_t>> pm4_golden;
    CHECK(load_golden_sections(aql_path, aql_golden));
    CHECK(load_golden_sections(pm4_path, pm4_golden));

    test_aql_packets(aql_golden);
    test_pm4_commands_and_roundtrip(pm4_golden);

    if (g_fails != 0) {
        std::printf("pm4_test: %d failure(s)\n", g_fails);
        return 1;
    }
    std::printf("pm4_test: ok\n");
    return 0;
}
