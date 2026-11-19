// src/kernels/pcie_overlap_graph_test.cpp - tests CUDA graph fork/join capture of fetch_blobs_after + rebase_ptrs
// on a secondary pcie_cs_ stream (STRATA_PCIE_FETCH_OVERLAP=1) across multi-blob dynamic staging capacities.
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

namespace k = strata::kernels;

namespace {

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA %s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}

}  // namespace

int main() {
    cudaStream_t cs = nullptr, pcie_cs = nullptr;
    cudaEvent_t ev_fork = nullptr, ev_join = nullptr;
    ck(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "cs");
    ck(cudaStreamCreateWithFlags(&pcie_cs, cudaStreamNonBlocking), "pcie_cs");
    ck(cudaEventCreateWithFlags(&ev_fork, cudaEventDisableTiming), "ev_fork");
    ck(cudaEventCreateWithFlags(&ev_join, cudaEventDisableTiming), "ev_join");

    constexpr int64_t blob_bytes = 64 * 1024;
    constexpr int staging_cap = 32;
    constexpr int n_blobs = 24;

    uint32_t *h_flag = nullptr, *m_flag = nullptr;
    ck(cudaHostAlloc(&h_flag, 64, cudaHostAllocMapped), "alloc flag");
    *h_flag = 0;
    ck(cudaHostGetDevicePointer((void **) &m_flag, h_flag, 0), "ptr flag");

    uint8_t *h_src_blobs = nullptr, *m_src_blobs = nullptr;
    ck(cudaHostAlloc(&h_src_blobs, (size_t) n_blobs * blob_bytes, cudaHostAllocMapped), "alloc src blobs");
    ck(cudaHostGetDevicePointer((void **) &m_src_blobs, h_src_blobs, 0), "ptr src blobs");

    uint8_t *d_stage_par = nullptr, *d_stage_ref = nullptr;
    ck(cudaMalloc(&d_stage_par, (size_t) staging_cap * blob_bytes), "stage par");
    ck(cudaMalloc(&d_stage_ref, (size_t) staging_cap * blob_bytes), "stage ref");

    int32_t *d_cnt = nullptr;
    unsigned long long *d_ptr_par = nullptr, *d_ptr_ref = nullptr, *d_dst2 = nullptr;
    ck(cudaMalloc(&d_cnt, 16), "cnt");
    ck(cudaMalloc(&d_ptr_par, staging_cap * 8), "ptr par");
    ck(cudaMalloc(&d_ptr_ref, staging_cap * 8), "ptr ref");
    ck(cudaMalloc(&d_dst2, staging_cap * 8), "dst2");

    std::vector<unsigned long long> h_ptrs(staging_cap, 0), h_dst2(staging_cap, 0);
    for (int i = 0; i < n_blobs; ++i)
        h_ptrs[i] = (unsigned long long) (uintptr_t) (m_src_blobs + (size_t) i * blob_bytes);
    ck(cudaMemcpy(d_cnt, &n_blobs, 4, cudaMemcpyHostToDevice), "copy cnt");
    ck(cudaMemcpy(d_dst2, h_dst2.data(), staging_cap * 8, cudaMemcpyHostToDevice), "copy dst2");

    std::mt19937 rng(88);
    int bad = 0;

    // Capture fork/join overlap graph on cs -> pcie_cs -> cs
    cudaGraph_t g = nullptr;
    cudaGraphExec_t gx = nullptr;
    ck(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal), "cap begin");
    ck(cudaEventRecord(ev_fork, cs), "fork rec");
    ck(cudaStreamWaitEvent(pcie_cs, ev_fork, 0), "fork wait");
    k::fetch_blobs_after(m_flag, 1u, d_ptr_par, d_cnt, d_stage_par, blob_bytes, staging_cap, pcie_cs, d_dst2);
    k::rebase_ptrs(d_ptr_par, d_cnt, d_stage_par, blob_bytes, pcie_cs, d_dst2);
    ck(cudaEventRecord(ev_join, pcie_cs), "join rec");
    ck(cudaStreamWaitEvent(cs, ev_join, 0), "join wait");
    ck(cudaStreamEndCapture(cs, &g), "cap end");
    ck(cudaGraphInstantiate(&gx, g, nullptr, nullptr, 0), "inst");

    for (int rep = 0; rep < 3; ++rep) {
        for (size_t i = 0; i < (size_t) n_blobs * blob_bytes / 4; ++i)
            reinterpret_cast<uint32_t*>(h_src_blobs)[i] = rng();
        *h_flag = 0;
        ck(cudaMemcpy(d_ptr_par, h_ptrs.data(), staging_cap * 8, cudaMemcpyHostToDevice), "reset ptr par");
        ck(cudaMemcpy(d_ptr_ref, h_ptrs.data(), staging_cap * 8, cudaMemcpyHostToDevice), "reset ptr ref");
        ck(cudaMemset(d_stage_par, 0, (size_t) staging_cap * blob_bytes), "zero stage par");
        ck(cudaMemset(d_stage_ref, 0, (size_t) staging_cap * blob_bytes), "zero stage ref");

        std::thread raiser([&]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            *(volatile uint32_t*) h_flag = 1u;
        });
        ck(cudaGraphLaunch(gx, cs), "launch");
        ck(cudaStreamSynchronize(cs), "sync par");
        raiser.join();

        k::fetch_blobs_after(m_flag, 1u, d_ptr_ref, d_cnt, d_stage_ref, blob_bytes, staging_cap, cs, d_dst2);
        k::rebase_ptrs(d_ptr_ref, d_cnt, d_stage_ref, blob_bytes, cs, d_dst2);
        ck(cudaStreamSynchronize(cs), "sync ref");

        std::vector<uint8_t> a((size_t) n_blobs * blob_bytes), b((size_t) n_blobs * blob_bytes);
        ck(cudaMemcpy(a.data(), d_stage_par, a.size(), cudaMemcpyDeviceToHost), "d2h a");
        ck(cudaMemcpy(b.data(), d_stage_ref, b.size(), cudaMemcpyDeviceToHost), "d2h b");
        if (std::memcmp(a.data(), b.data(), a.size()) != 0) {
            std::fprintf(stderr, "rep %d: staged blobs differ\n", rep);
            ++bad;
        }
        std::vector<unsigned long long> pa(n_blobs), pb(n_blobs);
        ck(cudaMemcpy(pa.data(), d_ptr_par, n_blobs * 8, cudaMemcpyDeviceToHost), "d2h pa");
        ck(cudaMemcpy(pb.data(), d_ptr_ref, n_blobs * 8, cudaMemcpyDeviceToHost), "d2h pb");
        for (int i = 0; i < n_blobs; ++i) {
            const unsigned long long want_a = (unsigned long long) (uintptr_t) (d_stage_par + (size_t) i * blob_bytes);
            const unsigned long long want_b = (unsigned long long) (uintptr_t) (d_stage_ref + (size_t) i * blob_bytes);
            if (pa[i] != want_a || pb[i] != want_b) {
                std::fprintf(stderr, "rep %d blob %d: rebased pointer mismatch\n", rep, i);
                ++bad;
                break;
            }
        }
    }

    cudaGraphExecDestroy(gx);
    cudaGraphDestroy(g);
    cudaFree(d_stage_par); cudaFree(d_stage_ref); cudaFree(d_cnt);
    cudaFree(d_ptr_par); cudaFree(d_ptr_ref); cudaFree(d_dst2);
    cudaFreeHost(h_flag); cudaFreeHost(h_src_blobs);
    cudaEventDestroy(ev_fork); cudaEventDestroy(ev_join);
    cudaStreamDestroy(pcie_cs); cudaStreamDestroy(cs);

    std::printf("pcie_overlap_graph_test: %s\n", bad ? "FAILED" : "ok");
    return bad ? 1 : 0;
}
