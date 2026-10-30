#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <vector>

namespace {

constexpr int kTypeQ4_0 = 2;
constexpr int kNIn = 32;
constexpr int kNOut = 8;
constexpr int kNCols = 16;
constexpr uint32_t kGuard = 0xa5a5a5a5u;

bool ck(cudaError_t status, const char* what) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "native_mmvq_wide_parity: %s: %s\n", what, cudaGetErrorString(status));
    return false;
}

long long bit_differences(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
    if (a.size() != b.size()) return (long long) (a.size() > b.size() ? a.size() - b.size() : b.size() - a.size());
    long long diff = 0;
    for (size_t i = 0; i < a.size(); ++i) diff += a[i] != b[i];
    return diff;
}

}  // namespace

int main() {
    cudaStream_t stream = nullptr;
    void* dweights = nullptr;
    float* dx = nullptr;
    float* dgate = nullptr;
    float* dup = nullptr;
    void* dq8 = nullptr;
    float* dmulti = nullptr;
    float* dsingle = nullptr;
    int32_t* dids = nullptr;
    float* dindexed = nullptr;
    bool ok = true;
    long long full_diff = 0, indexed_diff = 0, swiglu_diff = 0;

    try {
        const size_t qcol = strata::kernels::native_q8_1_bytes(kNIn, 1);
        const size_t qbytes = strata::kernels::native_q8_1_bytes(kNIn, kNCols);
        if (qbytes != qcol * kNCols) {
            std::fprintf(stderr, "native_mmvq_wide_parity: Q8_1 byte count is not contiguous across 16 columns\n");
            ok = false;
        }

        const size_t wbytes = strata::kernels::native_mmvq_weight_bytes(kTypeQ4_0, kNIn, kNOut);
        std::vector<uint8_t> weights(wbytes, 0);
        for (int row = 0; row < kNOut; ++row) {
            const uint16_t scale_one = 0x3c00; // fp16 1.0
            uint8_t* block = weights.data() + (size_t) row * 18;
            std::memcpy(block, &scale_one, sizeof(scale_one));
            for (int j = 0; j < 16; ++j) {
                const uint8_t lo = (uint8_t) ((row + j * 3) & 0x0f);
                const uint8_t hi = (uint8_t) ((row * 5 + j + 7) & 0x0f);
                block[2 + j] = (uint8_t) (lo | (hi << 4));
            }
        }
        std::vector<float> input((size_t) kNCols * kNIn);
        std::vector<float> gate(input.size()), up(input.size());
        for (int t = 0; t < kNCols; ++t) {
            for (int i = 0; i < kNIn; ++i) {
                input[(size_t) t * kNIn + i] = (float) (((t * 11 + i * 7) % 31) - 15) * 0.03125f;
                gate[(size_t) t * kNIn + i] = (float) (((t * 5 + i * 3) % 23) - 11) * 0.125f;
                up[(size_t) t * kNIn + i] = (float) (((t * 7 + i * 2) % 19) - 9) * 0.125f;
            }
        }

        if (ok && !ck(cudaStreamCreate(&stream), "stream create")) ok = false;
        if (ok && !ck(cudaMalloc(&dweights, wbytes), "weights allocation")) ok = false;
        if (ok && !ck(cudaMalloc(&dx, input.size() * sizeof(float)), "input allocation")) ok = false;
        if (ok && !ck(cudaMalloc(&dgate, gate.size() * sizeof(float)), "gate allocation")) ok = false;
        if (ok && !ck(cudaMalloc(&dup, up.size() * sizeof(float)), "up allocation")) ok = false;
        if (ok && !ck(cudaMalloc(&dq8, qbytes), "Q8_1 allocation")) ok = false;
        if (ok && !ck(cudaMalloc(&dmulti, (size_t) kNCols * kNOut * sizeof(float)), "multi output allocation")) ok = false;
        if (ok && !ck(cudaMalloc(&dsingle, (size_t) kNCols * kNOut * sizeof(float)), "single output allocation")) ok = false;
        const int32_t ids_host[] = {kNOut - 1, 0, 3, 3};
        constexpr size_t kNIds = sizeof(ids_host) / sizeof(ids_host[0]);
        if (ok && !ck(cudaMalloc(&dids, sizeof(ids_host)), "indexed ids allocation")) ok = false;
        if (ok && !ck(cudaMalloc(&dindexed, (kNCols * kNIds + 4) * sizeof(float)), "indexed output allocation")) ok = false;
        if (ok && !ck(cudaMemcpy(dweights, weights.data(), wbytes, cudaMemcpyHostToDevice), "weights upload")) ok = false;
        if (ok && !ck(cudaMemcpy(dx, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice), "input upload")) ok = false;
        if (ok && !ck(cudaMemcpy(dgate, gate.data(), gate.size() * sizeof(float), cudaMemcpyHostToDevice), "gate upload")) ok = false;
        if (ok && !ck(cudaMemcpy(dup, up.data(), up.size() * sizeof(float), cudaMemcpyHostToDevice), "up upload")) ok = false;
        if (ok && !ck(cudaMemcpy(dids, ids_host, sizeof(ids_host), cudaMemcpyHostToDevice), "indexed ids upload")) ok = false;

        if (ok) {
            strata::kernels::native_mmvq_set_multi_exact(true);
            strata::kernels::native_quantize_q8_1(dx, dq8, kNIn, kNCols, stream);
            strata::kernels::native_mmvq(kTypeQ4_0, dweights, dq8, dmulti, kNIn, kNOut, kNCols, stream);
            for (int t = 0; t < kNCols; ++t)
                strata::kernels::native_mmvq(kTypeQ4_0, dweights, (const uint8_t*) dq8 + (size_t) t * qcol,
                                             dsingle + (size_t) t * kNOut, kNIn, kNOut, 1, stream);
            if (!ck(cudaMemset(dindexed, 0xa5, (kNCols * kNIds + 4) * sizeof(float)), "indexed guard")) ok = false;
            if (ok) strata::kernels::native_mmvq_indexed(kTypeQ4_0, dweights, dq8, dindexed, kNIn, (int) kNIds,
                                                         kNCols, dids, stream);
            if (ok && !ck(cudaStreamSynchronize(stream), "full and indexed projections")) ok = false;

            std::vector<uint32_t> multi((size_t) kNCols * kNOut), single(multi.size());
            std::vector<uint32_t> indexed((size_t) kNCols * kNIds + 4);
            if (ok && !ck(cudaMemcpy(multi.data(), dmulti, multi.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                          "full output download")) ok = false;
            if (ok && !ck(cudaMemcpy(single.data(), dsingle, single.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                          "single output download")) ok = false;
            if (ok && !ck(cudaMemcpy(indexed.data(), dindexed, indexed.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                          "indexed output download")) ok = false;
            if (ok) {
                full_diff = bit_differences(multi, single);
                for (int t = 0; t < kNCols; ++t)
                    for (size_t r = 0; r < kNIds; ++r)
                        indexed_diff += indexed[(size_t) t * kNIds + r] != single[(size_t) t * kNOut + ids_host[r]];
                for (size_t i = (size_t) kNCols * kNIds; i < indexed.size(); ++i)
                    indexed_diff += indexed[i] != kGuard;
            }

            if (ok) {
                strata::kernels::native_swiglu_quantize_q8_1(dgate, dup, dq8, kNIn, kNCols, stream);
                strata::kernels::native_mmvq(kTypeQ4_0, dweights, dq8, dmulti, kNIn, kNOut, kNCols, stream);
                for (int t = 0; t < kNCols; ++t)
                    strata::kernels::native_mmvq(kTypeQ4_0, dweights, (const uint8_t*) dq8 + (size_t) t * qcol,
                                                 dsingle + (size_t) t * kNOut, kNIn, kNOut, 1, stream);
                if (!ck(cudaStreamSynchronize(stream), "SwiGLU-quantized projection")) ok = false;
                if (ok) {
                    if (!ck(cudaMemcpy(multi.data(), dmulti, multi.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                            "SwiGLU multi output download")) ok = false;
                    if (ok && !ck(cudaMemcpy(single.data(), dsingle, single.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                                  "SwiGLU single output download")) ok = false;
                    if (ok) swiglu_diff = bit_differences(multi, single);
                }
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "native_mmvq_wide_parity: %s\n", e.what());
        ok = false;
    }

    if (stream) (void) cudaStreamSynchronize(stream);
    if (dindexed) (void) cudaFree(dindexed);
    if (dids) (void) cudaFree(dids);
    if (dsingle) (void) cudaFree(dsingle);
    if (dmulti) (void) cudaFree(dmulti);
    if (dq8) (void) cudaFree(dq8);
    if (dup) (void) cudaFree(dup);
    if (dgate) (void) cudaFree(dgate);
    if (dx) (void) cudaFree(dx);
    if (dweights) (void) cudaFree(dweights);
    if (stream) (void) cudaStreamDestroy(stream);

    if (!ok) return 1;
    if (full_diff != 0 || indexed_diff != 0 || swiglu_diff != 0) {
        std::fprintf(stderr, "native_mmvq_wide_parity: mismatches full=%lld indexed/guard=%lld swiglu=%lld\n",
                     full_diff, indexed_diff, swiglu_diff);
        return 1;
    }
    std::printf("native_mmvq_wide_parity: 16-column Q8_1, full/indexed MMVQ, and SwiGLU projections match 1-column calls bitwise\n");
    return 0;
}
