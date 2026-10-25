// Context ring save/restore smoke test. Requires a DFlash2 pack and the supported GPU backend.
#include "strata/core/dflash2.hpp"
#include "strata/core/on_device.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: dflash2_state_test PACK\n");
        return 2;
    }
    strata::core::Dflash2Drafter df;
    std::string err;
    if (!df.load(argv[1], 0, err)) {
        std::fprintf(stderr, "load: %s\n", err.c_str());
        return 1;
    }
    const size_t n = (size_t) df.tap_layers().size() * (size_t) df.n_embd() * 2;
    std::vector<float> host(n);
    for (size_t i = 0; i < n; ++i) host[i] = (float)((int)(i % 23) - 11) * 0.01f;
    float* taps = nullptr;
    if (cudaMalloc((void**)&taps, n * sizeof(float)) != cudaSuccess ||
        cudaMemcpy(taps, host.data(), n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "allocate/copy taps failed\n");
        if (taps) cudaFree(taps);
        return 1;
    }
    if (!df.context(taps, 2, 0, err)) {
        std::fprintf(stderr, "context: %s\n", err.c_str());
        cudaFree(taps);
        return 1;
    }
    std::vector<uint8_t> state;
    if (!df.save_state(state, err, 2) || state.size() != df.state_bytes()) {
        std::fprintf(stderr, "save: %s (bytes=%zu expected=%zu)\n", err.c_str(), state.size(), df.state_bytes());
        cudaFree(taps);
        return 1;
    }
    df.reset();
    if (!df.restore_state(state, 2, err)) {
        std::fprintf(stderr, "restore: %s\n", err.c_str());
        cudaFree(taps);
        return 1;
    }
    std::vector<uint8_t> roundtrip;
    if (!df.save_state(roundtrip, err, 2) || roundtrip != state) {
        std::fprintf(stderr, "roundtrip did not preserve the DFlash2 ring: %s\n", err.c_str());
        cudaFree(taps);
        return 1;
    }
    // A checkpoint rewind removes rows beyond the resumed prefix.
    if (!df.restore_state(state, 1, err)) {
        std::fprintf(stderr, "rewind restore: %s\n", err.c_str());
        cudaFree(taps);
        return 1;
    }
    std::vector<uint8_t> rewound;
    if (!df.save_state(rewound, err, 1)) {
        std::fprintf(stderr, "rewound save: %s\n", err.c_str());
        cudaFree(taps);
        return 1;
    }
    int32_t pos_slot1 = -2;
    std::memcpy(&pos_slot1, rewound.data() + 128 + sizeof(int32_t), sizeof(pos_slot1));
    if (pos_slot1 != -1) {
        std::fprintf(stderr, "rewind left position 1 in the ring slot table (%d)\n", pos_slot1);
        cudaFree(taps);
        return 1;
    }
    std::vector<uint8_t> corrupt = state;
    const int32_t wrong_position = 1; // slot 0 must contain position 0 (or -1), never position 1
    std::memcpy(corrupt.data() + 128, &wrong_position, sizeof(wrong_position));
    if (df.restore_state(corrupt, 1, err)) {
        std::fprintf(stderr, "accepted a corrupt ring slot table\n");
        cudaFree(taps);
        return 1;
    }
    cudaFree(taps);
    std::printf("dflash2_state_test: save/restore/rewind passed (%zu bytes)\n", state.size());
    return 0;
}
