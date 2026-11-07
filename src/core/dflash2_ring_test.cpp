// src/core/dflash2_ring_test.cpp - the drafter's ring log and rebuild (Dflash2Drafter::begin_request), against the ring an uninterrupted update sequence left.
//
//     dflash2_ring_test PACK [device]
//
// Context rows are fed with deterministic pseudo-random taps (a function of the position, so any lineage can be refed), for lineage A up to a point P, the ring is
// copied out, then the update goes on far enough to wrap the ring (and, in the other scenarios, another lineage takes over).  begin_request must then rebuild the
// window the first draft reads, [n - 2047, resume), and the K/V bytes of every rebuilt slot must equal, bit for bit, those of the copy taken at P:
//   1 rewind     : A fed to P + 600, rewound to P (a repeated or rewound prefix)             -> rebuild from the live log
//   2 parked     : A fed to P, B fed (different tokens and taps), A resumed at P             -> swap from the stash, rebuild
//   3 branch     : A fed to P + 700, a request branches at P - 300 with other tokens, then A's old branch resumes at P -> stash copy (shared chunks, copy on write)
//   4 off        : STRATA_DF2_RING_LOG=0 (env set before the run): the same rewind finds the window blank (the old behaviour), reported not asserted
//   5 evicted    : a stash too small to keep A: the resume is cold (missing rows reported, *cold set)
// Exit 0 when every assertion holds.
#include "strata/core/dflash2.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void set_env(const char* k, const char* v) {
#ifdef _WIN32
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}
static void unset_env(const char* k) {
#ifdef _WIN32
    _putenv_s(k, "");
#else
    unsetenv(k);
#endif
}

using strata::core::Dflash2Drafter;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-100s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

constexpr int kTap = 12800;   // 5 taps x 2560

float tap_value(uint32_t lineage, int64_t pos, int i) {
    uint32_t x = (uint32_t) pos * 2654435761u ^ (uint32_t) i * 40503u ^ lineage * 97531u;
    x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
    return ((float) (x & 0xFFFF) / 32768.0f - 1.0f) * (1.0f + 0.25f * (float) (i % 7));
}

// feeds the rows [from, to) of `lineage` (tokens: token_of(lineage, pos)) as context updates, 256 rows a call, and notes the tokens
int32_t token_of(uint32_t lineage, int64_t pos) { return (int32_t) (1000 + ((uint32_t) pos * 7919u + lineage * 104729u) % 50000u); }

bool feed(Dflash2Drafter& d, float* dev, uint32_t lineage, int64_t from, int64_t to, std::string& err) {
    std::vector<float> h((size_t) 256 * kTap);
    for (int64_t p0 = from; p0 < to; p0 += 256) {
        const int64_t n = std::min<int64_t>(256, to - p0);
        for (int64_t r = 0; r < n; ++r)
            for (int i = 0; i < kTap; ++i) h[(size_t) r * kTap + (size_t) i] = tap_value(lineage, p0 + r, i);
        if (cudaMemcpy(dev, h.data(), (size_t) n * kTap * 4, cudaMemcpyHostToDevice) != cudaSuccess) { err = "upload"; return false; }
        if (!d.context(dev, n, p0, err)) return false;
        std::vector<int32_t> tk((size_t) n);
        for (int64_t r = 0; r < n; ++r) tk[(size_t) r] = token_of(lineage, p0 + r);
        d.note_rows(tk.data(), n, p0);
    }
    return true;
}

std::vector<int32_t> ids_of(uint32_t lineage, int64_t n) {
    std::vector<int32_t> v((size_t) n);
    for (int64_t p = 0; p < n; ++p) v[(size_t) p] = token_of(lineage, p);
    return v;
}

// bytes of slot s of layer l: K and V, 8 heads x 128 floats
bool slot_equal(const std::vector<float>& a, const std::vector<float>& b, int slot) {
    const size_t per = (size_t) Dflash2Drafter::kRing * 8 * 128;
    for (int l = 0; l < 5; ++l)
        for (int kv = 0; kv < 2; ++kv)
            for (int h = 0; h < 8; ++h) {
                const size_t off = ((size_t) l * 2 + (size_t) kv) * per + ((size_t) h * Dflash2Drafter::kRing + (size_t) slot) * 128;
                if (std::memcmp(&a[off], &b[off], 128 * 4) != 0) return false;
            }
    return true;
}

// every slot of the positions [lo, hi) holds the row `ref` has (same bytes), tagged with the drafter's current epoch and that position
int count_bad(Dflash2Drafter& d, const std::vector<float>& ref, int64_t lo, int64_t hi, int* wrong_tag) {
    std::vector<float> now;
    std::vector<int64_t> tags;
    std::string err;
    if (!d.debug_ring(now, tags, err)) { std::printf("debug_ring: %s\n", err.c_str()); return 1 << 20; }
    int bad = 0;
    *wrong_tag = 0;
    for (int64_t p = lo; p < hi; ++p) {
        const int slot = (int) (p % Dflash2Drafter::kRing);
        if ((tags[(size_t) slot] & 0xFFFFFFFFll) != p || tags[(size_t) slot] < 0) { ++*wrong_tag; ++bad; continue; }
        if (!slot_equal(now, ref, slot)) ++bad;
    }
    return bad;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: dflash2_ring_test PACK [device]\n"); return 2; }
    const int device = argc > 2 ? std::atoi(argv[2]) : 0;
    const bool log_off_run = std::getenv("STRATA_DF2_RING_LOG") != nullptr && std::getenv("STRATA_DF2_RING_LOG")[0] == '0';
    cudaSetDevice(device);
    std::string err;
    const int64_t P = 4227, N = 4232;   // the resumed prefix and the prompt (the agent prompt's numbers)
    float* dev = nullptr;
    cudaMalloc((void**) &dev, (size_t) 256 * kTap * 4);
    std::printf("dflash2_ring_test%s\n", log_off_run ? " (STRATA_DF2_RING_LOG=0)" : "");

    auto fresh = [&](Dflash2Drafter& d) -> bool {
        if (!d.load(argv[1], device, err)) { std::printf("load: %s\n", err.c_str()); return false; }
        return true;
    };
    std::vector<float> ref;
    std::vector<int64_t> reftags;
    {   // ---- 1 rewind
        std::printf("1 rewind (fed to %lld + 600, rewound to %lld, prompt %lld)\n", (long long) P, (long long) P, (long long) N);
        Dflash2Drafter d;
        if (!fresh(d)) return 2;
        if (!feed(d, dev, 1, 0, P, err)) { std::printf("feed: %s\n", err.c_str()); return 2; }
        if (!d.debug_ring(ref, reftags, err)) return 2;
        if (!feed(d, dev, 1, P, P + 600, err)) return 2;
        const std::vector<int32_t> ids = ids_of(1, N);
        bool cold = false;
        d.begin_request(ids.data(), N, P, &cold);
        int wt = 0;
        const int bad = count_bad(d, ref, N - 2047, P, &wt);
        std::printf("    rebuilt %lld rows in %.1f ms, missing %lld, slots wrong %d (wrong tag %d), cold %d\n", (long long) d.last_rebuilt, d.last_ms, (long long) d.last_missing, bad, wt, (int) cold);
        if (log_off_run) {
            check(d.last_rebuilt == 0, "log off: nothing is rebuilt (the pre-log behaviour)");
            check(bad > 0, "log off: the overwritten part of the window is blank (the defect this fixes)");
        } else {
            check(bad == 0, "every slot of [n - 2047, resume) holds its row, bitwise the original update's");
            check(d.last_rebuilt == (P + 600 - Dflash2Drafter::kRing) - (N - 2047), "only the overwritten slots were rebuilt (the others were already right)");
            check(!cold, "not cold");
        }
        // the ring must go on working: feed the prompt's remaining rows and a decode row, no stray tags
        if (!feed(d, dev, 1, P, N, err)) return 2;
    }
    if (log_off_run) {
        std::printf(g_fail == 0 ? "dflash2_ring_test: ok\n" : "dflash2_ring_test: FAILED\n");
        return g_fail == 0 ? 0 : 1;
    }
    {   // ---- 2 parked
        std::printf("2 parked (A to %lld, B to 3000, A resumed at %lld)\n", (long long) P, (long long) P);
        Dflash2Drafter d;
        if (!fresh(d)) return 2;
        if (!feed(d, dev, 1, 0, P, err)) return 2;
        {   // B takes over properly: a request for B from scratch
            const std::vector<int32_t> idsB = ids_of(2, 3000);
            bool cold = false;
            d.begin_request(idsB.data(), 3000, 0, &cold);
            if (!feed(d, dev, 2, 0, 3000, err)) return 2;
        }
        const std::vector<int32_t> ids = ids_of(1, N);
        bool cold = true;
        d.begin_request(ids.data(), N, P, &cold);
        int wt = 0;
        const int bad = count_bad(d, ref, N - 2047, P, &wt);
        std::printf("    stash hits %lld, rebuilt %lld rows in %.1f ms (swap %d), slots wrong %d (wrong tag %d), stash %.1f MiB\n", (long long) d.ring_stats.stash_hits,
                    (long long) d.last_rebuilt, d.last_ms, (int) d.last_swapped, bad, wt, (double) d.ring_stash_bytes() / 1048576.0);
        check(d.ring_stats.stash_hits == 1 && d.last_swapped, "the parked lineage came back from the stash");
        check(bad == 0, "every slot of [n - 2047, resume) holds its row, bitwise the original update's");
        check(d.last_rebuilt == P - (N - 2047), "the whole window below resume was rebuilt (a new epoch)");
        check(!cold, "not cold");
    }
    {   // ---- 3 branch
        std::printf("3 branch (A to %lld, a request at %lld with other tokens, the old branch resumed at %lld)\n", (long long) (P + 700), (long long) (P - 300), (long long) P);
        Dflash2Drafter d;
        if (!fresh(d)) return 2;
        if (!feed(d, dev, 1, 0, P + 700, err)) return 2;
        // reference at P: the ring of an uninterrupted feed to P is `ref` (same lineage, same taps)
        {   // a request that shares A up to P - 300 and then goes its own way (lineage 3 after that point)
            std::vector<int32_t> ids = ids_of(1, P - 300);
            for (int64_t p = P - 300; p < P - 300 + 400; ++p) ids.push_back(token_of(3, p));
            bool cold = false;
            d.begin_request(ids.data(), (int64_t) ids.size(), P - 300, &cold);
            if (!feed(d, dev, 3, P - 300, P - 300 + 400, err)) return 2;   // (rows after the branch point: lineage 3's taps)
        }
        const std::vector<int32_t> ids = ids_of(1, N);
        bool cold = true;
        d.begin_request(ids.data(), N, P, &cold);
        int wt = 0;
        const int bad = count_bad(d, ref, N - 2047, P, &wt);
        std::printf("    stash hits %lld, rebuilt %lld rows, missing %lld, slots wrong %d (wrong tag %d)\n", (long long) d.ring_stats.stash_hits, (long long) d.last_rebuilt,
                    (long long) d.last_missing, bad, wt);
        check(d.ring_stats.stash_hits >= 1, "the old branch was kept (copy on write) and came back");
        check(bad == 0, "the old branch's rows are bitwise the original update's, not the new branch's");
        check(!cold, "not cold");
    }
    {   // ---- 5 evicted
        std::printf("5 evicted (stash budget 0)\n");
        set_env("STRATA_DF2_LOG_MIB", "0");
        Dflash2Drafter d;
        if (!fresh(d)) return 2;
        unset_env("STRATA_DF2_LOG_MIB");
        if (!feed(d, dev, 1, 0, P, err)) return 2;
        const std::vector<int32_t> idsB = ids_of(2, 3000);
        bool cold = false;
        d.begin_request(idsB.data(), 3000, 0, &cold);
        if (!feed(d, dev, 2, 0, 3000, err)) return 2;
        const std::vector<int32_t> ids = ids_of(1, N);
        cold = false;
        d.begin_request(ids.data(), N, P, &cold);
        std::printf("    stash hits %lld, rebuilt %lld, missing %lld, cold %d\n", (long long) d.ring_stats.stash_hits, (long long) d.last_rebuilt, (long long) d.last_missing, (int) cold);
        check(cold && d.last_missing > 0 && d.last_rebuilt == 0, "no log of the lineage: cold, the missing rows are reported");
    }
    {   // ---- 6 batch_body vs single-slot draft parity
        std::printf("6 batch_body (2 slots, n_new=(5, 8) and (8, 3), STRATA_DF2_BATCHBODY=1 vs single-slot draft)\n");
        Dflash2Drafter d, sl0, sl1, ref0, ref1;
        if (!fresh(d)) return 2;
        const int n_vocab = 256;
        std::vector<uint8_t> h_head((size_t) n_vocab * (2560 / 32) * 34, 0);
        for (int r = 0; r < n_vocab; ++r) {
            for (int b = 0; b < 2560 / 32; ++b) {
                uint8_t* blk = &h_head[((size_t) r * (2560 / 32) + b) * 34];
                blk[0] = 0x00; blk[1] = 0x30;   // small fp16 scale
                for (int i = 0; i < 32; ++i)
                    blk[2 + i] = (uint8_t) (int8_t) (((r * 131 + b * 17 + i * 7) % 63) - 31);
            }
        }
        uint8_t* d_head = nullptr;
        cudaMalloc((void**) &d_head, h_head.size());
        cudaMemcpy(d_head, h_head.data(), h_head.size(), cudaMemcpyHostToDevice);
        const int n_emb = std::max(60000, d.mask_token() + 16);
        std::vector<float> h_emb((size_t) n_emb * 2560);
        for (size_t i = 0; i < h_emb.size(); ++i)
            h_emb[i] = 0.01f * (float) (((i * 31 + 13) % 101) - 50);
        float* d_emb = nullptr;
        cudaMalloc((void**) &d_emb, h_emb.size() * 4);
        cudaMemcpy(d_emb, h_emb.data(), h_emb.size() * 4, cudaMemcpyHostToDevice);
        std::vector<int32_t> vocab(n_vocab);
        for (int i = 0; i < n_vocab; ++i) vocab[i] = i;
        auto embed_fn = [d_emb](const int32_t* tok, int n, float* out, cudaStream_t cs) {
            strata::kernels::gather_rows((const uint8_t*) d_emb, 2560 * 4, tok, n, (uint8_t*) out, cs);
        };
        if (!sl0.clone_alloc(d, err) || !sl1.clone_alloc(d, err) ||
            !ref0.clone_alloc(d, err) || !ref1.clone_alloc(d, err)) {
            std::printf("clone_alloc: %s\n", err.c_str());
            return 2;
        }
        if (!d.bind_raw(embed_fn, d_head, 8, n_vocab, vocab, err)) {
            std::printf("bind_raw: %s\n", err.c_str());
            return 2;
        }
        if (!sl0.clone_bind(d, err) || !sl1.clone_bind(d, err) ||
            !ref0.clone_bind(d, err) || !ref1.clone_bind(d, err)) {
            std::printf("clone_bind: %s\n", err.c_str());
            return 2;
        }
        float *taps0 = nullptr, *taps1 = nullptr;
        cudaMalloc((void**) &taps0, (size_t) 16 * kTap * 4);
        cudaMalloc((void**) &taps1, (size_t) 16 * kTap * 4);
        if (!feed(sl0, dev, 1, 0, 256, err) || !feed(ref0, dev, 1, 0, 256, err) ||
            !feed(sl1, dev, 2, 0, 192, err) || !feed(ref1, dev, 2, 0, 192, err)) {
            std::printf("feed: %s\n", err.c_str());
            return 2;
        }
        bool tok_ok = true, ring_ok = true;
        const int n0_cases[2] = {5, 8};
        const int n1_cases[2] = {8, 3};
        int64_t pos0 = 256, pos1 = 192;
        for (int c = 0; c < 2; ++c) {
            const int n0 = n0_cases[c], n1 = n1_cases[c];
            std::vector<float> h0((size_t) n0 * kTap), h1((size_t) n1 * kTap);
            for (int r = 0; r < n0; ++r)
                for (int i = 0; i < kTap; ++i) h0[(size_t) r * kTap + i] = tap_value(1, pos0 + r, i);
            for (int r = 0; r < n1; ++r)
                for (int i = 0; i < kTap; ++i) h1[(size_t) r * kTap + i] = tap_value(2, pos1 + r, i);
            cudaMemcpy(taps0, h0.data(), h0.size() * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(taps1, h1.data(), h1.size() * 4, cudaMemcpyHostToDevice);
            int32_t r_tok0[8] = {}, r_tok1[8] = {}, b_tok0[8] = {}, b_tok1[8] = {};
            float r_prob0[8] = {}, r_prob1[8] = {}, b_prob0[8] = {}, b_prob1[8] = {};
            if (!ref0.draft(taps0, n0, pos0, 17 + c, r_tok0, r_prob0, err) ||
                !ref1.draft(taps1, n1, pos1, 42 + c, r_tok1, r_prob1, err)) {
                std::printf("ref draft: %s\n", err.c_str());
                return 2;
            }
            Dflash2Drafter::BatchSlotItem items[2] = {
                {&sl0, taps0, n0, pos0, 17 + c, 7, b_tok0, b_prob0},
                {&sl1, taps1, n1, pos1, 42 + c, 7, b_tok1, b_prob1},
            };
            if (!d.draft_batch_slots_launch(items, 2, err) ||
                !d.draft_batch_slots_finish(items, 2, err)) {
                std::printf("batch draft: %s\n", err.c_str());
                return 2;
            }
            for (int i = 0; i < 7; ++i) {
                if (r_tok0[i] != b_tok0[i] || r_tok1[i] != b_tok1[i] ||
                    std::fabs(r_prob0[i] - b_prob0[i]) > 1e-5f ||
                    std::fabs(r_prob1[i] - b_prob1[i]) > 1e-5f) {
                    std::printf("    case %d i=%d: sl0 ref=(%d,%.6f) bat=(%d,%.6f) | sl1 ref=(%d,%.6f) bat=(%d,%.6f)\n",
                                c, i, r_tok0[i], r_prob0[i], b_tok0[i], b_prob0[i], r_tok1[i], r_prob1[i], b_tok1[i], b_prob1[i]);
                    tok_ok = false;
                }
            }
            pos0 += n0;
            pos1 += n1;
        }
        std::vector<float> r_ring0, r_ring1;
        std::vector<int64_t> r_tag0, r_tag1;
        ref0.debug_ring(r_ring0, r_tag0, err);
        ref1.debug_ring(r_ring1, r_tag1, err);
        int wt0 = 0, wt1 = 0;
        if (count_bad(sl0, r_ring0, 0, pos0, &wt0) != 0 || count_bad(sl1, r_ring1, 0, pos1, &wt1) != 0)
            ring_ok = false;
        check(tok_ok, "batch_body 2-slot draft tokens and probs match single-slot draft()");
        check(ring_ok, "batch_body 2-slot K/V rings match single-slot draft() bitwise");
        {   // ---- 7 precompute_batch_context + batch_full_exec_ parity
            std::printf("7 precompute_batch_context + batch_full_exec_ (2 slots, n_new=(8, 8))\n");
            Dflash2Drafter p0, p1;
            if (!p0.clone_alloc(d, err) || !p1.clone_alloc(d, err) ||
                !p0.clone_bind(d, err) || !p1.clone_bind(d, err) ||
                !feed(p0, dev, 1, 0, 256, err) || !feed(p1, dev, 2, 0, 192, err)) {
                std::printf("precomp setup: %s\n", err.c_str());
                return 2;
            }
            const int n0 = 8, n1 = 8;
            std::vector<float> h_both((size_t) (n0 + n1) * kTap);
            int64_t pos_both[16] = {};
            for (int r = 0; r < n0; ++r) {
                pos_both[r] = 256 + r;
                for (int i = 0; i < kTap; ++i) h_both[(size_t) r * kTap + i] = tap_value(1, 256 + r, i);
            }
            for (int r = 0; r < n1; ++r) {
                pos_both[n0 + r] = 192 + r;
                for (int i = 0; i < kTap; ++i) h_both[(size_t) (n0 + r) * kTap + i] = tap_value(2, 192 + r, i);
            }
            float* taps_both = nullptr;
            cudaMalloc((void**) &taps_both, h_both.size() * 4);
            cudaMemcpy(taps_both, h_both.data(), h_both.size() * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(taps0, h_both.data(), (size_t) n0 * kTap * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(taps1, h_both.data() + (size_t) n0 * kTap, (size_t) n1 * kTap * 4, cudaMemcpyHostToDevice);
            int32_t r_tok0[8] = {}, r_tok1[8] = {}, b_tok0[8] = {}, b_tok1[8] = {};
            float r_prob0[8] = {}, r_prob1[8] = {}, b_prob0[8] = {}, b_prob1[8] = {};
            ref0.begin_request(nullptr, 0, 0, nullptr);
            ref1.begin_request(nullptr, 0, 0, nullptr);
            feed(ref0, dev, 1, 0, 256, err);
            feed(ref1, dev, 2, 0, 192, err);
            if (!ref0.draft(taps0, n0, 256, 33, r_tok0, r_prob0, err) ||
                !ref1.draft(taps1, n1, 192, 44, r_tok1, r_prob1, err)) {
                std::printf("ref draft (precomp): %s\n", err.c_str());
                return 2;
            }
            d.precompute_batch_context(taps_both, n0 + n1, pos_both);
            Dflash2Drafter::BatchSlotItem pitems[2] = {
                {&p0, taps_both, n0, 256, 33, 7, b_tok0, b_prob0},
                {&p1, taps_both + (size_t) n0 * kTap, n1, 192, 44, 7, b_tok1, b_prob1},
            };
            bool p_tok_ok = true, p_ring_ok = true;
            for (int rep = 0; rep < 2; ++rep) {
                if (rep == 1) {
                    feed(p0, dev, 1, 0, 256, err);
                    feed(p1, dev, 2, 0, 192, err);
                    d.precompute_batch_context(taps_both, n0 + n1, pos_both);
                }
                if (!d.draft_batch_slots_launch(pitems, 2, err) ||
                    !d.draft_batch_slots_finish(pitems, 2, err)) {
                    std::printf("precomp batch draft: %s\n", err.c_str());
                    return 2;
                }
                for (int i = 0; i < 7; ++i) {
                    if (r_tok0[i] != b_tok0[i] || r_tok1[i] != b_tok1[i] ||
                        std::fabs(r_prob0[i] - b_prob0[i]) > 1e-5f ||
                        std::fabs(r_prob1[i] - b_prob1[i]) > 1e-5f) {
                        std::printf("    rep %d i=%d: p0 ref=(%d,%.6f) bat=(%d,%.6f) | p1 ref=(%d,%.6f) bat=(%d,%.6f)\n",
                                    rep, i, r_tok0[i], r_prob0[i], b_tok0[i], b_prob0[i], r_tok1[i], r_prob1[i], b_tok1[i], b_prob1[i]);
                        p_tok_ok = false;
                    }
                }
            }
            ref0.debug_ring(r_ring0, r_tag0, err);
            ref1.debug_ring(r_ring1, r_tag1, err);
            if (count_bad(p0, r_ring0, 0, 256 + n0, &wt0) != 0 || count_bad(p1, r_ring1, 0, 192 + n1, &wt1) != 0)
                p_ring_ok = false;
            check(p_tok_ok, "precompute_batch_context + batch_full_exec_ 16-row draft tokens/probs match single-slot draft()");
            check(p_ring_ok, "precompute_batch_context + batch_full_exec_ 16-row K/V rings match single-slot draft() bitwise");
            {
                cudaEvent_t ev0 = nullptr, ev1 = nullptr, ev2 = nullptr;
                cudaEventCreate(&ev0); cudaEventCreate(&ev1); cudaEventCreate(&ev2);
                const int reps = 50;
                float ms_pre = 0.0f, ms_full = 0.0f, ms_npre = 0.0f;
                for (int r = 0; r < reps; ++r) {
                    cudaEventRecord(ev0, d.stream());
                    d.precompute_batch_context(taps_both, n0 + n1, pos_both);
                    cudaEventRecord(ev1, d.stream());
                    d.draft_batch_slots_launch(pitems, 2, err);
                    cudaEventRecord(ev2, d.stream());
                    d.draft_batch_slots_finish(pitems, 2, err);
                    float a = 0.0f, b = 0.0f;
                    cudaEventElapsedTime(&a, ev0, ev1);
                    cudaEventElapsedTime(&b, ev1, ev2);
                    ms_pre += a;
                    ms_full += b;
                }
                for (int r = 0; r < reps; ++r) {
                    d.clear_precomputed_context();
                    cudaEventRecord(ev0, d.stream());
                    d.draft_batch_slots_launch(pitems, 2, err);
                    cudaEventRecord(ev1, d.stream());
                    d.draft_batch_slots_finish(pitems, 2, err);
                    float a = 0.0f;
                    cudaEventElapsedTime(&a, ev0, ev1);
                    ms_npre += a;
                }
                std::printf("  precomp (16 rows): %.2f us | batch_full_exec_: %.2f us (sum %.2f us) | batch_body_exec_+head: %.2f us\n",
                            ms_pre * 1000.0f / reps, ms_full * 1000.0f / reps, (ms_pre + ms_full) * 1000.0f / reps, ms_npre * 1000.0f / reps);
                cudaEventDestroy(ev0); cudaEventDestroy(ev1); cudaEventDestroy(ev2);
            }
            cudaFree(taps_both);
        }
        cudaFree(taps0); cudaFree(taps1); cudaFree(d_head); cudaFree(d_emb);
    }
    std::printf(g_fail == 0 ? "dflash2_ring_test: ok\n" : "dflash2_ring_test: FAILED\n");
    return g_fail == 0 ? 0 : 1;
}
