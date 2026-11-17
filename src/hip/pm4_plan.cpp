// src/hip/pm4_plan.cpp - Allocation-granularity boundary planner,
// GFX12 CacheTable lowering, in-place patch points, and PM4 IB replay executor.
//
// Port note: from warpfront/hipfire railgun + rdna-compute (Apache-2.0)
// Ported from warpfront/hipfire crates/railgun/src/plan.rs (CacheTable, Hazard,
// plan(), ArchLowering::for_arch) and crates/rdna-compute/src/replay.rs
// (PreparedPm4Replay, gfx1201_pm4_pacing, patch_dispatch_dimensions, patch_kernarg_u32).

#include "strata/hip/pm4_plan.hpp"

#if (defined(__linux__) && defined(STRATA_WITH_HSA)) || defined(STRATA_PM4_HOST_TEST)

#include <cstdlib>
#include <cstring>

namespace strata::hip::pm4 {

namespace {

void write_u32_le_inplace(uint8_t* dst, uint32_t v) noexcept {
    dst[0] = static_cast<uint8_t>(v & 0xffu);
    dst[1] = static_cast<uint8_t>((v >> 8) & 0xffu);
    dst[2] = static_cast<uint8_t>((v >> 16) & 0xffu);
    dst[3] = static_cast<uint8_t>((v >> 24) & 0xffu);
}

void write_u64_le_inplace(uint8_t* dst, uint64_t v) noexcept {
    for (size_t b = 0; b < 8u; ++b) {
        dst[b] = static_cast<uint8_t>((v >> (b * 8u)) & 0xffu);
    }
}

bool equals_ignore_case(const char* a, const char* b, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) {
        char ca = a[i];
        char cb = b[i];
        if (ca >= 'A' && ca <= 'Z') ca = static_cast<char>(ca - 'A' + 'a');
        if (cb >= 'A' && cb <= 'Z') cb = static_cast<char>(cb - 'A' + 'a');
        if (ca != cb) return false;
    }
    return true;
}

}  // namespace

bool parse_gfx12_arch(const char* arch_name, Gfx12Arch& out_arch) noexcept {
    if (arch_name == nullptr) {
        return false;
    }
    while (*arch_name == ' ' || *arch_name == '\t') {
        ++arch_name;
    }
    // Strip optional ROCm target feature suffixes (e.g. "gfx1201:sramecc-:xnack-")
    size_t len = 0;
    while (arch_name[len] != '\0' && arch_name[len] != ':' &&
           arch_name[len] != ' ' && arch_name[len] != '\t') {
        ++len;
    }
    if (len == 7u && equals_ignore_case(arch_name, "gfx1201", 7u)) {
        out_arch = Gfx12Arch::Gfx1201;
        return true;
    }
    if (len == 7u && equals_ignore_case(arch_name, "gfx1200", 7u)) {
        out_arch = Gfx12Arch::Gfx1200;
        return true;
    }
    return false;
}

bool resolve_gfx12_post_dispatch_nop(
    Gfx12Arch arch,
    const char* env_value,
    uint32_t& out_nop_dwords) noexcept {
    const uint32_t default_nop =
        (arch == Gfx12Arch::Gfx1201) ? kGfx1201PostDispatchNopDwords : 0u;
    if (env_value == nullptr) {
        out_nop_dwords = default_nop;
        return true;
    }
    while (*env_value == ' ' || *env_value == '\t') {
        ++env_value;
    }
    size_t len = std::strlen(env_value);
    while (len > 0 && (env_value[len - 1] == ' ' || env_value[len - 1] == '\t')) {
        --len;
    }
    if (len == 0 || (len == 4u && equals_ignore_case(env_value, "auto", 4u))) {
        out_nop_dwords = default_nop;
        return true;
    }
    if ((len == 1u && env_value[0] == '0') ||
        (len == 3u && equals_ignore_case(env_value, "off", 3u)) ||
        (len == 4u && equals_ignore_case(env_value, "none", 4u)) ||
        (len == 5u && equals_ignore_case(env_value, "false", 5u))) {
        out_nop_dwords = 0u;
        return true;
    }
    if (len > 4u && equals_ignore_case(env_value, "nop:", 4u)) {
        char* endp = nullptr;
        const unsigned long val = std::strtoul(env_value + 4u, &endp, 10);
        if (endp != (env_value + 4u) &&
            static_cast<size_t>(endp - env_value) == len &&
            val >= 1UL && val <= 16384UL) {
            out_nop_dwords = static_cast<uint32_t>(val);
            return true;
        }
    }
    out_nop_dwords = default_nop;
    return false;
}

bool plan_pm4_boundaries(
    const Pm4Tape& tape,
    Gfx12Arch arch,
    Pm4BoundaryPlan& out_plan,
    const char* pacing_env_override) noexcept {
    if (!tape.valid() || tape.launch_count() == 0) {
        return false;
    }

    const auto& dispatches = tape.dispatches();
    const size_t n = dispatches.size();
    const size_t num_resources = tape.resource_count();

    out_plan = Pm4BoundaryPlan{};
    out_plan.arch = arch;
    uint32_t nop_dwords = 0;
    (void) resolve_gfx12_post_dispatch_nop(arch, pacing_env_override, nop_dwords);
    out_plan.post_dispatch_nop_dwords = nop_dwords;
    out_plan.entry_system_acquire = true;
    out_plan.trailing_system_release = true;
    out_plan.boundaries.assign(n, BoundaryVisibility{false, AcquireRung::None, false});

    // Forward scan ported from warpfront/hipfire crates/railgun/src/plan.rs:463-544:
    // - `open`: node indices since the last `wait_idle` boundary.
    // - `last_vmem_load`: most recent node index that performed a VmemLoad on each resource.
    // - `last_system`: node index of the last AcquireRung::System boundary (0 at entry because
    //   the IB entry emits acquire_system_gfx12() before dispatch 0).
    std::vector<uint32_t> open;
    open.reserve(n);
    std::vector<uint32_t> last_vmem_load(num_resources, UINT32_MAX);
    uint32_t last_system = 0u;

    for (uint32_t j = 0; j < static_cast<uint32_t>(n); ++j) {
        const auto& dj = dispatches[j];
        if (j > 0u) {
            BoundaryVisibility vis{false, AcquireRung::None, false};

            // 1. Check Raw, War, Waw hazards against all nodes in the current open epoch.
            for (uint32_t i : open) {
                const auto& di = dispatches[i];
                for (const ResourceEffect& ea : di.effects) {
                    for (const ResourceEffect& eb : dj.effects) {
                        if (ea.resource != eb.resource) {
                            continue;
                        }
                        if (ea.is_write() && eb.is_read()) {
                            vis = join_visibility(
                                vis, gfx12_cache_table_lookup(HazardKind::Raw, ea.access, eb.access));
                        }
                        if (ea.is_read() && eb.is_write()) {
                            vis = join_visibility(
                                vis, gfx12_cache_table_lookup(HazardKind::War, ea.access, eb.access));
                        }
                        if (ea.is_write() && eb.is_write()) {
                            vis = join_visibility(
                                vis, gfx12_cache_table_lookup(HazardKind::Waw, ea.access, eb.access));
                        }
                    }
                }
            }

            // 2. Check WarPreWriter across epochs: if node j writes (VmemStore) a resource that
            //    had a VmemLoad at node i >= last_system (not yet covered by a System acquire),
            //    GFX12 requires AcquireRung::System (GCR 0xc3b1, PWS_ENA2=1) before the writer.
            for (const ResourceEffect& eb : dj.effects) {
                if (eb.access == MemAccessKind::VmemStore && eb.resource < num_resources) {
                    const uint32_t reader_idx = last_vmem_load[eb.resource];
                    if (reader_idx != UINT32_MAX && reader_idx >= last_system && reader_idx < j) {
                        vis = join_visibility(
                            vis,
                            gfx12_cache_table_lookup(
                                HazardKind::WarPreWriter,
                                MemAccessKind::VmemLoad,
                                MemAccessKind::VmemStore));
                    }
                }
            }

            out_plan.boundaries[j] = vis;
            if (vis.wait_idle) {
                ++out_plan.wait_idle_count;
                open.clear();
            }
            if (vis.acquire == AcquireRung::InterNode) {
                ++out_plan.inter_node_acquire_count;
            } else if (vis.acquire == AcquireRung::System) {
                ++out_plan.system_acquire_count;
                last_system = j;
            }
        }

        for (const ResourceEffect& eb : dj.effects) {
            if (eb.access == MemAccessKind::VmemLoad && eb.resource < num_resources) {
                last_vmem_load[eb.resource] = j;
            }
        }
        open.push_back(j);
    }

    return true;
}

// ---- PreparedPm4Window ----

bool PreparedPm4Window::prepare(
    Pm4Tape& tape,
    Gfx12Arch arch,
    const char* pacing_env_override,
    std::string* out_err) noexcept {
    prepared_ = false;
    latched_fallback_ = false;
    patch_points_.clear();
    cb_.clear();

    if (quiescence_ != ReplayQuiescence::Proven) {
        latched_fallback_ = true;
        if (out_err) *out_err = "window is not quiescent";
        return false;
    }
    if (!tape.valid() || tape.launch_count() == 0) {
        latched_fallback_ = true;
        if (out_err) *out_err = tape.failure_reason().empty() ? "invalid or empty tape" : tape.failure_reason();
        return false;
    }
    if (!plan_pm4_boundaries(tape, arch, plan_, pacing_env_override)) {
        latched_fallback_ = true;
        if (out_err) *out_err = "boundary planner failed";
        return false;
    }

    cb_.set_post_dispatch_nop(plan_.post_dispatch_nop_dwords);
    if (plan_.entry_system_acquire) {
        cb_.acquire_system_gfx12(false);
    }

    const auto& dispatches = tape.dispatches();
    patch_points_.reserve(dispatches.size());

    for (size_t j = 0; j < dispatches.size(); ++j) {
        const RecordedDispatch& disp = dispatches[j];
        if (j > 0u) {
            cb_.emit_boundary(plan_.boundaries[j]);
        }

        // Re-validate geometry via LaunchGeometry::from_hip_workgroups (grid.y/z <= 65536 guard).
        LaunchGeometry geom{};
        const PacketErrorCode gerr = LaunchGeometry::from_hip_workgroups(
            disp.hip_grid_workgroups, disp.hip_workgroup_size, geom);
        if (gerr != PacketErrorCode::Ok) {
            latched_fallback_ = true;
            if (out_err) *out_err = packet_error_to_string(gerr);
            return false;
        }

        const uint64_t kernarg_va = tape.arena().slot_device_va(disp.kernarg_arena_offset);
        Pm4DispatchOffsets offsets{};
        const Pm4BuildError berr = cb_.dispatch(
            disp.loader_meta, disp.pm4_meta, geom, disp.dynamic_lds_bytes, kernarg_va, &offsets);
        if (berr != Pm4BuildError::Ok) {
            latched_fallback_ = true;
            if (out_err) *out_err = "PM4 dispatch packet build failed";
            return false;
        }

        Pm4DispatchPatchPoint pp{};
        pp.dispatch_index = static_cast<uint32_t>(j);
        pp.kernarg_arena_offset = disp.kernarg_arena_offset;
        pp.kernarg_bytes = disp.kernarg_bytes;
        pp.workgroup_size = disp.hip_workgroup_size;
        pp.recorded_grid_workgroups = disp.hip_grid_workgroups;
        pp.sh_reg_base_dword = offsets.sh_reg_base_dword;
        pp.dispatch_initiator_dword = offsets.dispatch_initiator_dword;
        pp.dim_x_dword = offsets.dispatch_initiator_dword - 3u;
        patch_points_.push_back(pp);
    }

    if (plan_.trailing_system_release) {
        cb_.wait_compute_idle();
        cb_.acquire_system_gfx12(false);
    }

    tape_ = &tape;
    prepared_ = true;
    return true;
}

bool PreparedPm4Window::patch_kernarg_u32(
    size_t dispatch_index,
    size_t byte_offset,
    uint32_t value) noexcept {
    if (!ready() || tape_ == nullptr || dispatch_index >= patch_points_.size()) {
        return false;
    }
    const Pm4DispatchPatchPoint& pp = patch_points_[dispatch_index];
    if ((byte_offset % 4u) != 0u || byte_offset + 4u > pp.kernarg_bytes) {
        return false;
    }
    uint8_t* slot = tape_->arena().slot_ptr(pp.kernarg_arena_offset);
    write_u32_le_inplace(slot + byte_offset, value);
    return true;
}

bool PreparedPm4Window::patch_kernarg_u64(
    size_t dispatch_index,
    size_t byte_offset,
    uint64_t value) noexcept {
    if (!ready() || tape_ == nullptr || dispatch_index >= patch_points_.size()) {
        return false;
    }
    const Pm4DispatchPatchPoint& pp = patch_points_[dispatch_index];
    if ((byte_offset % 8u) != 0u || byte_offset + 8u > pp.kernarg_bytes) {
        return false;
    }
    uint8_t* slot = tape_->arena().slot_ptr(pp.kernarg_arena_offset);
    write_u64_le_inplace(slot + byte_offset, value);
    return true;
}

bool PreparedPm4Window::patch_dispatch_dimensions(
    size_t dispatch_index,
    std::array<uint32_t, 3> hip_grid_workgroups) noexcept {
    if (!ready() || dispatch_index >= patch_points_.size()) {
        return false;
    }
    const Pm4DispatchPatchPoint& pp = patch_points_[dispatch_index];
    LaunchGeometry geom{};
    // Reuse LaunchGeometry::from_hip_workgroups (including grid.y/z <= 65536 guard)
    // from src/hip/pm4_packet.cpp.
    if (LaunchGeometry::from_hip_workgroups(hip_grid_workgroups, pp.workgroup_size, geom) !=
        PacketErrorCode::Ok) {
        return false;
    }
    auto& dwords = cb_.dwords_mut();
    if (static_cast<size_t>(pp.dim_x_dword) + 3u > dwords.size()) {
        return false;
    }
    dwords[pp.dim_x_dword + 0u] = geom.grid_workitems[0];
    dwords[pp.dim_x_dword + 1u] = geom.grid_workitems[1];
    dwords[pp.dim_x_dword + 2u] = geom.grid_workitems[2];
    return true;
}

bool PreparedPm4Window::replay(SubmitPm4IbFn submit_fn, void* user_ctx) noexcept {
    if (!ready() || submit_fn == nullptr || cb_.len_dwords() == 0) {
        latched_fallback_ = true;
        return false;
    }
    ReplayQuiescence post_q = ReplayQuiescence::Unknown;
    const bool ok = submit_fn(
        cb_.dwords().data(),
        cb_.len_dwords(),
        ib_device_va_,
        post_q,
        user_ctx);
    quiescence_ = post_q;
    if (!ok || post_q != ReplayQuiescence::Proven) {
        latched_fallback_ = true;
        return false;
    }
    ++replay_count_;
    return true;
}

}  // namespace strata::hip::pm4

#endif  // (defined(__linux__) && defined(STRATA_WITH_HSA)) || defined(STRATA_PM4_HOST_TEST)
