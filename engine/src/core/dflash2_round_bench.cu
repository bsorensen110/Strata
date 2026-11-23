// src/core/dflash2_round_bench.cu - the time of a solo DFlash2 round (Dflash2Drafter::draft) on the real pack, no model.
//
//     dflash2_round_bench PACK [rounds] [head_type] [head_rows]
//
// The head is synthetic (random blocks of ggml type head_type, default 14 = Q6_K over 248320 rows: the native head of Qwen3.8-Flash-Next), the
// embedding a hash of the token, the taps a hash of the position.  The ring is filled to 2040 rows, then `rounds` rounds run with n_new cycling
// through 1..8 as a decode would (the position advances by n_new).  Prints the wall time of a round per n_new and a digest of every round's drafts,
// probabilities and the last round's final rows, so a change can be checked to leave the drafts bitwise the same.
// STRATA_DF2_PROFILE=1 adds the per-stage breakdown; STRATA_DF2_TIMING=1 the GPU time of the graph.
#include "strata/core/dflash2.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using strata::core::Dflash2Drafter;

namespace {
constexpr int H = 2560;
constexpr int kTap = 5 * H;

__device__ __host__ inline float hashf(uint32_t a, uint32_t b) {
    uint32_t x = a * 2654435761u ^ b * 40503u;
    x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
    return ((float) (x & 0xFFFF) / 32768.0f - 1.0f);
}

__global__ void embed_k(const int32_t* tok, float* out) {
    const int r = blockIdx.x;
    const uint32_t t = (uint32_t) tok[r];
    for (int c = threadIdx.x; c < H; c += blockDim.x) out[(size_t) r * H + c] = 0.05f * hashf(t + 7u, (uint32_t) c);
}

__global__ void taps_k(float* out, int64_t pos0, int n) {
    const int r = blockIdx.x;
    if (r >= n) return;
    for (int c = threadIdx.x; c < kTap; c += blockDim.x)
        out[(size_t) r * kTap + c] = hashf((uint32_t) (pos0 + r), (uint32_t) c) * (1.0f + 0.25f * (float) (c % 7));
}

size_t row_bytes(int type) {
    switch (type) {
        case 8: return (size_t) (H / 32) * 34;     // Q8_0
        case 12: return (size_t) (H / 256) * 144;  // Q4_K
        case 13: return (size_t) (H / 256) * 176;  // Q5_K
        case 14: return (size_t) (H / 256) * 210;  // Q6_K
        default: return 0;
    }
}

// random blocks with small, finite fp16 scales
void fill_head(std::vector<uint8_t>& h, int type, int64_t rows) {
    const size_t rb = row_bytes(type);
    uint32_t s = 12345u;
    auto rnd = [&] { s = s * 1664525u + 1013904223u; return (uint8_t) (s >> 24); };
    for (auto& b : h) b = rnd();
    const uint16_t dsc = 0x1400, dmin = 0x0c00;   // ~0.00098, ~0.00024
    for (int64_t r = 0; r < rows; ++r) {
        uint8_t* row = h.data() + (size_t) r * rb;
        if (type == 8) {
            for (int b = 0; b < H / 32; ++b) std::memcpy(row + b * 34, &dsc, 2);
        } else if (type == 12 || type == 13) {
            const int bs = type == 12 ? 144 : 176;
            for (int b = 0; b < H / 256; ++b) { std::memcpy(row + b * bs, &dsc, 2); std::memcpy(row + b * bs + 2, &dmin, 2); }
        } else if (type == 14) {
            for (int b = 0; b < H / 256; ++b) std::memcpy(row + b * 210 + 208, &dsc, 2);
        }
    }
}

uint64_t fnv(uint64_t h, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: dflash2_round_bench PACK [rounds] [head_type] [head_rows]\n"); return 2; }
    const int rounds = argc > 2 ? std::atoi(argv[2]) : 400;
    const int head_type = argc > 3 ? std::atoi(argv[3]) : 14;
    const int64_t head_rows = argc > 4 ? std::atoll(argv[4]) : 248320;
    if (row_bytes(head_type) == 0) { std::fprintf(stderr, "head_type %d: use 8, 12, 13 or 14\n", head_type); return 2; }
    cudaSetDevice(0);
    std::string err;
    Dflash2Drafter d;
    if (!d.load(argv[1], 0, err)) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }
    std::vector<uint8_t> head((size_t) head_rows * row_bytes(head_type));
    fill_head(head, head_type, head_rows);
    uint8_t* d_head = nullptr;
    if (cudaMalloc((void**) &d_head, head.size()) != cudaSuccess) { std::fprintf(stderr, "head alloc\n"); return 1; }
    cudaMemcpy(d_head, head.data(), head.size(), cudaMemcpyHostToDevice);
    auto embed = [](const int32_t* tok, int n, float* out, cudaStream_t cs) { embed_k<<<n, 256, 0, cs>>>(tok, out); };
    if (!d.bind_raw(embed, d_head, head_type, head_rows, {}, err)) { std::fprintf(stderr, "bind: %s\n", err.c_str()); return 1; }

    float* d_ctx = nullptr;
    float* d_taps = nullptr;
    cudaMalloc((void**) &d_ctx, (size_t) 256 * kTap * 4);
    cudaMalloc((void**) &d_taps, (size_t) Dflash2Drafter::kRows * kTap * 4);
    int64_t pos = 0;
    const int64_t P = 2040;
    for (; pos < P; pos += 256) {
        const int n = (int) std::min<int64_t>(256, P - pos);
        taps_k<<<n, 256>>>(d_ctx, pos, n);
        cudaDeviceSynchronize();
        if (!d.context(d_ctx, n, pos, err)) { std::fprintf(stderr, "context: %s\n", err.c_str()); return 1; }
    }
    pos = P;
    int32_t drafts[Dflash2Drafter::kDrafts];
    float probs[Dflash2Drafter::kDrafts];
    double ms_n[9] = {}, cnt_n[9] = {};
    uint64_t dig = 1469598103934665603ull;
    const int warm = 16;
    int32_t anchor = 1000;
    for (int i = 0; i < rounds + warm; ++i) {
        const int n_new = 1 + (i * 5 + i / 8) % 8;
        taps_k<<<n_new, 256>>>(d_taps, pos, n_new);
        cudaDeviceSynchronize();
        const auto t0 = std::chrono::steady_clock::now();
        if (!d.draft(d_taps, n_new, pos, anchor, drafts, probs, err)) { std::fprintf(stderr, "draft: %s\n", err.c_str()); return 1; }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (i >= warm) { ms_n[n_new] += ms; cnt_n[n_new] += 1; }
        dig = fnv(dig, drafts, sizeof drafts);
        dig = fnv(dig, probs, sizeof probs);
        pos += n_new;
        anchor = drafts[0] >= 0 ? drafts[0] : anchor + 1;
    }
    std::vector<float> rows;
    if (d.debug_rows(rows, err)) dig = fnv(dig, rows.data(), rows.size() * 4);
    double tot = 0, cn = 0;
    std::printf("dflash2_round_bench: head type %d x %lld rows, %d rounds\n  ms per round by n_new:", head_type, (long long) head_rows, rounds);
    for (int n = 1; n <= 8; ++n)
        if (cnt_n[n] > 0) { std::printf(" %d:%.3f", n, ms_n[n] / cnt_n[n]); tot += ms_n[n]; cn += cnt_n[n]; }
    std::printf("\n  mean %.4f ms per round (wall, launch to drafts on the host)", tot / cn);
    if (d.rounds > 0 && d.ms_gpu > 0) std::printf(", GPU %.4f ms", d.ms_gpu / (double) d.rounds);
    std::printf("\n  digest %016llx (drafts, probabilities, final rows)\n", (unsigned long long) dig);
    return 0;
}
