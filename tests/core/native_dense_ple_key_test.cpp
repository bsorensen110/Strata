// tests/core/native_dense_ple_key_test.cpp - #326: a native pack whose PLE key row is BF16 (iq_pack --compat-bf16 of
// OrcaRouter's IQ3_XXS key) loads with that row in the arena, and NativeDense does not upload the GGUF key over it.
//
//   1. BF16 key row + IQ3_XXS GGUF key: served_names lists the key, keep_unquantized_ple_key takes it out of the
//      skip set, the arena holds the BF16 bytes, and NativeDense::load attaches no native key (0.1.25-0.1.31 refused
//      the load with "incompatible matrix blk.1.ple_key.weight").  The other matrix still gets its native upload.
//   2. A quantized key row (the Q2_0 / UD-Q4_K_XL packs: a shape-only row) + a Q8_0 GGUF key: unchanged - the key
//      stays skipped and is served natively.
//   3. index_code_bits on a missing row and a missing index.
// Synthetic files at small dimensions; needs a CUDA device for the arena (exits 77 without one).
#include "gguf_fixture.hpp"

#include "strata/core/layer.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/weights.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using strata::core::NativeDense;
using strata::core::WeightTable;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-84s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-ple-key-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

const char* KEY = "blk.1.ple_key.weight";
const char* QKV = "blk.0.attn_qkv.weight";
constexpr uint64_t NE0 = 256, NE1 = 4;

std::vector<fixture::Kv> arch_keys() {
    return {fixture::str("general.architecture", "qwen4exp"), fixture::u32("qwen4exp.block_count", 48),
            fixture::u32("qwen4exp.embedding_length", 2560), fixture::u32("qwen4exp.expert_count", 512),
            fixture::u32("qwen4exp.expert_used_count", 10), fixture::u32("qwen4exp.attention.head_count", 24),
            fixture::u32("qwen4exp.attention.head_count_kv", 2)};
}

// A shape-only quantized row (what iq_pack writes for a tensor the GGUF serves) and, with `bf16_key`, the key as a
// raw BF16 row (index kind 4) in dense.bin.
void write_pack(const fs::path& dir, bool bf16_key) {
    fs::create_directories(dir);
    const uint64_t key_bytes = NE0 * NE1 * 2;
    std::string idx = "# align 256 pool " + std::to_string(bf16_key ? 2048 + 256 : 256) + " tensors 2\n";
    idx += std::string(QKV) + " 0 0 0 0 0 0 256 4 8 0 32 0 0 0 0 0 0 0\n";
    if (bf16_key)
        idx += std::string(KEY) + " 0 4 0 " + std::to_string(key_bytes) + " 256 " + std::to_string(key_bytes) +
               " 256 4 0 0 1 0 0 0 0 0 0 0\n";
    else
        idx += std::string(KEY) + " 0 0 0 0 256 0 256 4 8 0 32 0 0 0 0 0 0 0\n";
    std::ofstream(dir / "index.txt", std::ios::binary) << idx;
    std::vector<char> bytes(bf16_key ? key_bytes : 16);
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = (char) fixture::pattern(9, i);
    std::ofstream(dir / "dense.bin", std::ios::binary).write(bytes.data(), (std::streamsize) bytes.size());
}

struct Loaded {
    bool ok = false;
    std::string err;
    std::set<std::string> skip;
    WeightTable wt;
    NativeDense dense;
    void* arena = nullptr;
    ~Loaded() { if (arena) cudaFree(arena); }
};

// generate.cpp's order: served_names, keep_unquantized_ple_key (native packs), pool_bytes, load, NativeDense::load.
void load(const fs::path& pack, const std::string& shard, Loaded& l) {
    const std::vector<std::string> shards{shard};
    if (!NativeDense::served_names(shards, true, l.skip, l.err)) return;
    if (!NativeDense::keep_unquantized_ple_key(pack.string(), l.skip, l.err)) return;
    uint64_t pool = 0;
    if (!WeightTable::pool_bytes(pack.string(), pool, l.err, &l.skip)) return;
    if (cudaMalloc(&l.arena, pool ? pool : 256) != cudaSuccess) { l.err = "cudaMalloc"; return; }
    if (!l.wt.load(pack.string(), l.arena, pool ? pool : 256, l.err, &l.skip)) return;
    l.ok = l.dense.load(shards, l.wt, l.err, true);
}
}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("native_dense_ple_key_test: no CUDA device, skipped\n");
        return 77;
    }
    TempDir tmp;

    std::printf("1. BF16 key row (--compat-bf16), IQ3_XXS GGUF key\n");
    {
        const fs::path pack = tmp.path / "orca";
        write_pack(pack, true);
        const std::string shard = (tmp.path / "orca.gguf").string();
        fixture::write(shard, arch_keys(), {{QKV, {NE0, NE1}, 8, 1}, {KEY, {NE0, NE1}, 18, 2}});
        int bits = 99;
        std::string err;
        check(WeightTable::index_code_bits(pack.string(), KEY, bits, err) && bits == 0, "index_code_bits: key row is 0");
        check(WeightTable::index_code_bits(pack.string(), QKV, bits, err) && bits == 8, "index_code_bits: qkv row is 8");
        Loaded l;
        load(pack, shard, l);
        check(l.ok, "the pack loads (was: incompatible matrix) " + l.err);
        check(!l.skip.count(KEY) && l.skip.count(QKV), "the key is not skipped, qkv is");
        const auto* k = l.wt.find(KEY);
        check(k && k->resident && k->data && !k->quantized(), "the key is resident and BF16");
        check(k && !k->native_data, "no native key attached");
        if (k && k->data) {
            std::vector<unsigned char> back(NE0 * NE1 * 2);
            cudaMemcpy(back.data(), k->data, back.size(), cudaMemcpyDeviceToHost);
            bool same = true;
            for (size_t i = 0; i < back.size(); ++i) same = same && back[i] == fixture::pattern(9, i);
            check(same, "the arena holds the pack's BF16 bytes");
        }
        const auto* q = l.wt.find(QKV);
        check(q && q->native_data && q->native_type == 8, "qkv still served natively (Q8_0)");
        check(l.dense.tensor_count() == 1, "one native matrix");
    }

    std::printf("2. quantized key row (Q2_0 / UD-Q4_K_XL packs), Q8_0 GGUF key: unchanged\n");
    {
        const fs::path pack = tmp.path / "q8";
        write_pack(pack, false);
        const std::string shard = (tmp.path / "q8.gguf").string();
        fixture::write(shard, arch_keys(), {{QKV, {NE0, NE1}, 8, 1}, {KEY, {NE0, NE1}, 8, 2}});
        Loaded l;
        load(pack, shard, l);
        check(l.ok, "the pack loads " + l.err);
        check(l.skip.count(KEY) == 1, "the key stays skipped");
        const auto* k = l.wt.find(KEY);
        check(k && !k->resident && k->native_data && k->native_type == 8, "the key is served natively");
        check(l.dense.tensor_count() == 2, "two native matrices");
    }

    std::printf("3. index_code_bits edge cases\n");
    {
        int bits = 0;
        std::string err;
        check(WeightTable::index_code_bits((tmp.path / "orca").string(), "blk.9.nothing.weight", bits, err) &&
                  bits == -1, "a missing row is -1");
        check(!WeightTable::index_code_bits((tmp.path / "none").string(), KEY, bits, err) && !err.empty(),
              "a missing index is an error");
        std::set<std::string> skip{QKV};
        check(NativeDense::keep_unquantized_ple_key((tmp.path / "none").string(), skip, err) && skip.size() == 1,
              "no key in the skip set: the index is not read");
    }

    std::printf("4. STRATA_HC_Q8: add_hc_skip includes output_hc_{down,up}.weight and provenance flag\n");
    {
        const fs::path hc_dir = tmp.path / "hc_skip";
        fs::create_directories(hc_dir);
        std::string idx = "# align 256 pool 35328 tensors 7\n"
                          + std::string(QKV) + " 0 0 0 0 0 0 256 4 8 0 32 0 0 0 0 0 0 0\n"
                          "blk.0.hc_attn_down.weight 0 4 0 512 0 512 256 1 0 0 1 0 0 0 0 0 0 0\n"
                          "blk.0.hc_attn_up.weight 0 4 512 512 512 512 256 1 0 0 1 0 0 0 0 0 0 0\n"
                          "blk.0.hc_attn_inject.weight 0 4 1024 512 1024 512 256 1 0 0 1 0 0 0 0 0 0 0\n"
                          "output_hc_norm.weight 0 2 1536 1024 1536 1024 256 1 0 0 1 0 0 0 0 0 0 0\n"
                          "output_hc_down.weight 0 4 2560 16384 2560 16384 256 32 0 0 1 0 0 0 0 0 0 0\n"
                          "output_hc_up.weight 0 4 18944 16384 18944 16384 32 256 0 0 1 0 0 0 0 0 0 0\n";
        std::ofstream(hc_dir / "index.txt", std::ios::binary) << idx;
        std::set<std::string> skip;
        check(NativeDense::add_hc_skip(hc_dir.string(), skip), "add_hc_skip reads index.txt");
        check(skip.count("blk.0.hc_attn_down.weight") == 1 && skip.count("blk.0.hc_attn_up.weight") == 1,
              "add_hc_skip skips blk.0.hc_attn_{down,up}.weight");
        check(skip.count("output_hc_down.weight") == 1 && skip.count("output_hc_up.weight") == 1,
              "add_hc_skip skips output_hc_{down,up}.weight");
        check(skip.count("blk.0.hc_attn_inject.weight") == 0,
              "add_hc_skip keeps blk.0.hc_attn_inject.weight in the BF16 arena");
        strata::core::WeightRef ref;
        check(!ref.hc_q8_from_gguf, "WeightRef::hc_q8_from_gguf defaults to false");
        setenv("STRATA_HC_Q8", "1", 1);
        std::vector<char> hc_dense(35328, 0);
        // Set non-zero norm and BF16 weights so lm_head_mix produces a non-zero output when reading hc_q8
        for (size_t i = 0; i < 256; ++i) {
            const float one = 1.0f;
            std::memcpy(hc_dense.data() + 1536 + i * 4, &one, 4);
        }
        for (size_t i = 0; i < 256 * 32; ++i) {
            const uint16_t half_bf16 = 0x3E80u; // 0.25f in BF16
            std::memcpy(hc_dense.data() + 2560 + i * 2, &half_bf16, 2);
            std::memcpy(hc_dense.data() + 18944 + i * 2, &half_bf16, 2);
        }
        std::ofstream(hc_dir / "dense.bin", std::ios::binary).write(hc_dense.data(), (std::streamsize) hc_dense.size());
        {
            const std::string shard_q8 = (tmp.path / "hc_gguf_q8.gguf").string();
            const auto winfo = fixture::write(shard_q8, arch_keys(),
                                              {{QKV, {NE0, NE1}, 8, 1},
                                               {"blk.0.hc_attn_down.weight", {NE0, 1}, 8, 2},
                                               {"output_hc_down.weight", {256, 32}, 8, 3}});
            // Overwrite output_hc_down.weight's Q8_0 block scales with finite 0.25f (0x3400) so dequant has no random FP16 NaN
            {
                std::fstream gf(shard_q8, std::ios::in | std::ios::out | std::ios::binary);
                const uint64_t off = winfo.data_start + winfo.offsets[2];
                for (uint64_t b = 0; b < (256 * 32) / 32; ++b) {
                    const uint16_t d_fp16 = 0x3400u;
                    gf.seekp((std::streamoff) (off + b * 34));
                    gf.write((const char*) &d_fp16, 2);
                }
            }
            Loaded l;
            l.skip = skip;
            l.skip.insert(QKV);
            uint64_t pool = 0;
            check(WeightTable::pool_bytes(hc_dir.string(), pool, l.err, &l.skip), "hc_q8 gguf pool_bytes");
            (void) cudaMalloc(&l.arena, pool ? pool : 256);
            check(l.wt.load(hc_dir.string(), l.arena, pool ? pool : 256, l.err, &l.skip), "hc_q8 gguf wt.load");
            check(l.dense.load({shard_q8}, l.wt, l.err, false), "hc_q8 gguf dense.load " + l.err);
            const auto* b0d = l.wt.find("blk.0.hc_attn_down.weight");
            const auto* ohd = l.wt.find("output_hc_down.weight");
            const auto* ohu = l.wt.find("output_hc_up.weight");
            check(b0d && b0d->hc_q8 && b0d->hc_q8_from_gguf,
                  "blk.0.hc_attn_down.weight loaded from GGUF Q8_0 has hc_q8_from_gguf=true");
            check(ohd && ohd->hc_q8 && ohd->hc_q8_from_gguf && !ohd->resident && ohd->data == nullptr,
                  "output_hc_down.weight loaded from GGUF Q8_0 is skipped in arena and has hc_q8_from_gguf=true");
            check(ohu && ohu->hc_q8 && !ohu->hc_q8_from_gguf && !ohu->resident && ohu->data == nullptr,
                  "output_hc_up.weight absent from GGUF is quantized from pack with hc_q8_from_gguf=false");

            strata::core::ModelGeometry g{};
            g.n_embd = 64; g.hc = 4; g.hc_lr = 32;
            strata::core::BlockBuffers bb{};
            void* bb_mem = nullptr;
            const uint64_t bb_bytes = strata::core::block_buffers_bytes(g);
            (void) cudaMalloc(&bb_mem, bb_bytes);
            strata::core::block_buffers_init(g, bb_mem, bb);
            std::vector<float> h_R(256, 1.0f), h_mixed(64, 0.0f);
            (void) cudaMemcpy(bb.R, h_R.data(), h_R.size() * sizeof(float), cudaMemcpyHostToDevice);
            check(strata::core::lm_head_mix(l.wt, g, bb, nullptr, l.err), "lm_head_mix succeeds with skipped output_hc_*");
            check(cudaDeviceSynchronize() == cudaSuccess, "lm_head_mix kernel completes without fault on skipped output_hc_*");
            (void) cudaMemcpy(h_mixed.data(), bb.mixed, h_mixed.size() * sizeof(float), cudaMemcpyDeviceToHost);
            float abs_sum = 0.0f;
            for (float v : h_mixed) abs_sum += std::abs(v);
            check(abs_sum > 0.0f, "lm_head_mix produces non-zero mixed output from hc_q8");
            (void) cudaFree(bb_mem);
        }
    }

    std::printf(g_fail ? "native_dense_ple_key_test: %d FAILED\n" : "native_dense_ple_key_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
