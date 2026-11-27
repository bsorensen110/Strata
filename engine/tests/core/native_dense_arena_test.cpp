// Audit finding 10: verify NativeDense single-arena 256B-aligned planning, single 2 MiB granule rounding,
// host-side weight_bytes_for arithmetic, and bit-identical carved views over a synthetic GGUF fixture (no GPU required).
#include "gguf_fixture.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using strata::core::NativeDense;
using strata::core::WeightKind;
using strata::core::WeightRef;

namespace {

int g_fail = 0;

void check(bool cond, const char* what) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    }
}

void check_eq_u64(uint64_t got, uint64_t want, const char* what) {
    if (got != want) {
        std::fprintf(stderr, "FAIL: %s: got=%llu want=%llu\n",
                     what, (unsigned long long) got, (unsigned long long) want);
        ++g_fail;
    }
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-nd-arena-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

std::vector<fixture::Kv> arch_keys() {
    return {
        fixture::str("general.architecture", "qwen4exp"),
        fixture::u32("qwen4exp.block_count", 48),
        fixture::u32("qwen4exp.embedding_length", 2560),
        fixture::u32("qwen4exp.expert_count", 512),
        fixture::u32("qwen4exp.expert_used_count", 10),
        fixture::u32("qwen4exp.attention.head_count", 24),
        fixture::u32("qwen4exp.attention.head_count_kv", 2),
    };
}

} // namespace

int main() {
#if defined(_WIN32)
    _putenv_s("CUDA_VISIBLE_DEVICES", "-1");
    _putenv_s("HIP_VISIBLE_DEVICES", "-1");
    _putenv_s("STRATA_NATIVE_DENSE_ARENA", "1");
#else
    setenv("CUDA_VISIBLE_DEVICES", "-1", 1);
    setenv("HIP_VISIBLE_DEVICES", "-1", 1);
    setenv("STRATA_NATIVE_DENSE_ARENA", "1", 0);
#endif

    constexpr uint64_t MiB = 1ull << 20;

    // 1. Arena alignment & single-granule accounting vs old per-matrix granule rounding.
    check_eq_u64(NativeDense::arena_align(0), 0, "align 0");
    check_eq_u64(NativeDense::arena_align(1), 256, "align 1 -> 256");
    check_eq_u64(NativeDense::arena_align(256), 256, "align 256 -> 256");
    check_eq_u64(NativeDense::arena_align(257), 512, "align 257 -> 512");

    check_eq_u64(NativeDense::alloc_bytes((1ull << 20) - 1), (1ull << 20) - 1, "<1 MiB unrounded");
    check_eq_u64(NativeDense::alloc_bytes(1ull << 20), 2 * MiB, "1 MiB -> 2 MiB granule");
    check_eq_u64(NativeDense::alloc_bytes(2 * MiB), 2 * MiB, "2 MiB exact");
    check_eq_u64(NativeDense::alloc_bytes(2 * MiB + 1), 4 * MiB, "2 MiB + 1 -> 4 MiB");

    // 192 HC Q8_0 matrices at ne0=2560, ne1=1280 -> 3,481,600 B (~3.32 MiB) each:
    const uint64_t hc_sz = 2560ull / 32ull * 34ull * 1280ull;
    std::vector<uint64_t> hc_sizes(192, hc_sz);
    const auto hc_plan = NativeDense::plan_arena(hc_sizes);
    uint64_t hc_per_matrix = 0;
    for (uint64_t s : hc_sizes) hc_per_matrix += NativeDense::alloc_bytes(s);
    const uint64_t hc_saved = hc_per_matrix - hc_plan.allocated_bytes;
    check_eq_u64(hc_per_matrix, 192ull * 4 * MiB, "old per-matrix HC = 768 MiB");
    check(hc_saved >= 128 * MiB, "192x 3.32 MiB HC arena saves >= 128 MiB of 2 MiB granules");

    // 300 dense matrices of ~6.73 MiB (per native_dense comment: 300 x 6.73 MiB = 2019 MiB vs 2400 MiB per-matrix):
    const uint64_t dense_sz = 7057408ull;   // 6.7305 MiB, divisible by 256 and 32
    std::vector<uint64_t> d_sizes(300, dense_sz);
    const auto d_plan = NativeDense::plan_arena(d_sizes);
    uint64_t d_per_matrix = 0;
    for (uint64_t s : d_sizes) d_per_matrix += NativeDense::alloc_bytes(s);
    const uint64_t d_saved = d_per_matrix - d_plan.allocated_bytes;
    check_eq_u64(d_per_matrix, 300ull * 8 * MiB, "old per-matrix 6.73 MiB = 2400 MiB");
    check_eq_u64(d_plan.allocated_bytes, 2020ull * MiB, "arena 300x 6.73 MiB = 2020 MiB");
    check_eq_u64(d_saved, 380ull * MiB, "300x 6.73 MiB arena saves 380 MiB");

    std::printf(
        "native_dense_arena_test: hc192_per_matrix=%.2f MiB vs arena=%.2f MiB (saved %.2f MiB); "
        "dense300_per_matrix=%.2f MiB vs arena=%.2f MiB (saved %.2f MiB)\n",
        (double) hc_per_matrix / (double) MiB, (double) hc_plan.allocated_bytes / (double) MiB, (double) hc_saved / (double) MiB,
        (double) d_per_matrix / (double) MiB, (double) d_plan.allocated_bytes / (double) MiB, (double) d_saved / (double) MiB);

    // 2. Synthetic GGUF fixture with multiple 1.5 MiB-class Q8_0 and BF16-as-Q8_0 matrices:
    // Verify NativeDense::weight_bytes_for matches single-arena sizing and carved host arena views are bit-identical.
    TempDir tmp;
    const fs::path gguf_path = tmp.path / "fixture.gguf";
    // Q8_0 matrix shape [ne0=2048, ne1=720] -> (2048/32)*34*720 = 1,566,720 B (~1.494 MiB).
    // BF16 matrix shape [ne0=2048, ne1=720] -> BF16 in GGUF (type 30), bf16_as_q8_0 prices Q8_0 bytes = 1,566,720 B.
    constexpr uint64_t ne0 = 2048, ne1 = 720;
    const uint64_t q8_bytes = (uint64_t) strata::kernels::native_mmvq_weight_bytes(8, (int) ne0, (int) ne1);
    check_eq_u64(q8_bytes, (ne0 / 32) * 34 * ne1, "q8_0 fixture matrix bytes");

    const std::vector<fixture::Tensor> ts = {
        {"blk.0.attn_qkv.weight", {ne0, ne1}, 8, 11},
        {"blk.0.attn_gate.weight", {ne0, ne1}, 8, 22},
        {"blk.1.ffn_gate_shexp.weight", {ne0, ne1}, 8, 33},
        {"blk.1.ffn_up_shexp.weight", {ne0, ne1}, 30, 44}, // #4741: bf16_as_q8_0
        {"blk.0.attn_norm.weight", {2560}, 0, 55},         // ignored by eligible()
    };
    fixture::write(gguf_path, arch_keys(), ts);

    std::map<std::string, WeightRef> table;
    auto add_ref = [&](const std::string& name, WeightKind kind, int code_bits) {
        WeightRef r;
        r.kind = kind;
        r.ne0 = (int64_t) ne0;
        r.ne1 = (int64_t) ne1;
        r.elements = (int64_t) (ne0 * ne1);
        r.code_bits = code_bits;
        table[name] = r;
    };
    add_ref("blk.0.attn_qkv.weight", WeightKind::Verbatim, 8);
    add_ref("blk.0.attn_gate.weight", WeightKind::Verbatim, 8);
    add_ref("blk.1.ffn_gate_shexp.weight", WeightKind::Verbatim, 8);
    add_ref("blk.1.ffn_up_shexp.weight", WeightKind::Bf16InF32, 0);

    std::set<std::string> served;
    std::string err;
    check(NativeDense::served_names({gguf_path.string()}, /*include_ple_key=*/false, served, err), "served_names ok");
    check_eq_u64(served.size(), 3, "served_names excludes BF16 (#4741) and 1D norm");

    uint64_t all_layers_bytes = 0;
    check(NativeDense::weight_bytes_for({gguf_path.string()}, table, false, 0, 48, all_layers_bytes, err),
          "weight_bytes_for [0,48) ok");
    const uint64_t expected_payload4 = 4 * NativeDense::arena_align(q8_bytes);   // 4 * 1,566,720 = 6,266,880 B (~5.977 MiB)
    check_eq_u64(all_layers_bytes, NativeDense::alloc_bytes(expected_payload4),
                 "weight_bytes_for [0,48) = ceil_2M(5.98 MiB) = 6 MiB instead of 8 MiB");
    check_eq_u64(all_layers_bytes, 6 * MiB, "single-arena rounds 4x1.494 MiB to 6 MiB, saving 2 MiB over 4x2 MiB");

    uint64_t layer0_bytes = 0;
    check(NativeDense::weight_bytes_for({gguf_path.string()}, table, false, 0, 1, layer0_bytes, err),
          "weight_bytes_for [0,1) ok");
    const uint64_t expected_payload2 = 2 * NativeDense::arena_align(q8_bytes);   // 2 * 1,566,720 = 3,133,440 B (~2.988 MiB)
    check_eq_u64(layer0_bytes, 4 * MiB, "weight_bytes_for [0,1) = ceil_2M(2.99 MiB) = 4 MiB");
    check_eq_u64(expected_payload2, 2 * q8_bytes, "aligned sum matches");

    // 3. Carve a host arena using NativeDense::plan_arena and verify alignment, disjointness, and bit-exact payloads.
    const strata::GgufFile gguf(gguf_path.string());
    std::vector<uint64_t> slice_sizes;
    std::vector<const strata::TensorInfo*> q8_infos;
    for (const auto& t : gguf.tensors()) {
        if (t.type == 8) {
            slice_sizes.push_back((uint64_t) strata::kernels::native_mmvq_weight_bytes(t.type, (int) t.shape[0], (int) t.shape[1]));
            q8_infos.push_back(&t);
        }
    }
    const auto plan = NativeDense::plan_arena(slice_sizes);
    std::vector<uint8_t> arena_buf((size_t) plan.payload_bytes, 0xA5);
    for (size_t i = 0; i < q8_infos.size(); ++i) {
        check((plan.offsets[i] % NativeDense::kArenaAlign) == 0, "slice offset 256B aligned");
        if (i > 0) {
            check(plan.offsets[i] >= plan.offsets[i - 1] + slice_sizes[i - 1], "slices non-overlapping");
        }
        std::memcpy(arena_buf.data() + plan.offsets[i], gguf.tensor_data(*q8_infos[i]), (size_t) slice_sizes[i]);
    }
    for (size_t i = 0; i < q8_infos.size(); ++i) {
        const uint8_t* got = arena_buf.data() + plan.offsets[i];
        const uint8_t* want = (const uint8_t*) gguf.tensor_data(*q8_infos[i]);
        check(std::memcmp(got, want, (size_t) slice_sizes[i]) == 0, "carved arena slice bit-identical to GGUF payload");
    }

    if (g_fail != 0) {
        std::printf("native_dense_arena_test: %d FAILED\n", g_fail);
        return 1;
    }
    std::printf("native_dense_arena_test: all passed\n");
    return 0;
}
