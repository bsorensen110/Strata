// Context ring save/restore smoke test. Requires a DFlash2 pack and the supported GPU backend.
#include "strata/core/dflash2.hpp"
#include "strata/core/on_device.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        std::fprintf(stderr, "usage: dflash2_state_test PACK [expected-window]\n");
        return 2;
    }
    const int expected_window = argc == 3 ? std::atoi(argv[2]) : 2048;
    strata::core::Dflash2Drafter df;
    std::string err;
    if (!df.load(argv[1], 0, err)) {
        std::fprintf(stderr, "load: %s\n", err.c_str());
        return 1;
    }
    const size_t expected_ring = (size_t) expected_window + 16;
    const size_t expected_elem = 2;   // default FP16 ring storage
    const size_t expected_state = 128 + expected_ring * 4 + df.tap_layers().size() * 2 * expected_ring * 8 * 128 * expected_elem;
    if (df.attention_window() != expected_window || df.ring_slots() != (int) expected_ring ||
        df.ring_elem_bytes() != expected_elem || df.state_bytes() != expected_state) {
        std::fprintf(stderr, "window %d: loaded window=%d ring=%d elem=%zu state=%zu, expected ring=%zu elem=%zu state=%zu\n",
                     expected_window, df.attention_window(), df.ring_slots(), df.ring_elem_bytes(), df.state_bytes(),
                     expected_ring, expected_elem, expected_state);
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
    // Live in-place rewind (continuing a conversation from an earlier checkpoint without parking):
    // must produce the exact same ring state as restore_state(..., 1).
    if (!df.restore_state(state, 2, err)) {
        std::fprintf(stderr, "restore before live rewind: %s\n", err.c_str());
        cudaFree(taps);
        return 1;
    }
    df.rewind(1);
    std::vector<uint8_t> rewound_live;
    if (!df.save_state(rewound_live, err, 1) || rewound_live != rewound) {
        std::fprintf(stderr, "live rewind(1) did not match restore_state(..., 1): %s\n", err.c_str());
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

    // The batch drafter grows after load(), and a grown slot owns a distinct K/V ring plus its own host position
    // mirror. On an 8k-window pack this covers the 8208-entry mirror (the former 2064-entry default overflowed it).
    if (!df.set_n_slots(2, err)) {
        std::fprintf(stderr, "grow to two drafter slots: %s\n", err.c_str());
        cudaFree(taps);
        return 1;
    }
    if (!df.set_slot(0, err) || !df.restore_state(state, 2, err)) {
        std::fprintf(stderr, "restore slot 0 before interleave: %s\n", err.c_str());
        cudaFree(taps);
        return 1;
    }
    std::vector<uint8_t> slot0_before;
    if (!df.save_state(slot0_before, err, 2)) {
        std::fprintf(stderr, "save slot 0 before interleave: %s\n", err.c_str());
        cudaFree(taps);
        return 1;
    }
    std::vector<float> host_alt(n);
    for (size_t i = 0; i < n; ++i) host_alt[i] = (float)((int)(i % 29) - 14) * 0.037f;
    float* taps_alt = nullptr;
    if (cudaMalloc((void**)&taps_alt, n * sizeof(float)) != cudaSuccess ||
        cudaMemcpy(taps_alt, host_alt.data(), n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "allocate/copy slot 1 taps failed\n");
        if (taps_alt) cudaFree(taps_alt);
        cudaFree(taps);
        return 1;
    }
    if (!df.set_slot(1, err) || !df.context(taps_alt, 2, 0, err)) {
        std::fprintf(stderr, "slot 1 context: %s\n", err.c_str());
        cudaFree(taps_alt);
        cudaFree(taps);
        return 1;
    }
    std::vector<uint8_t> slot1_state;
    if (!df.save_state(slot1_state, err, 2) || slot1_state.size() != df.state_bytes()) {
        std::fprintf(stderr, "save slot 1: %s (bytes=%zu expected=%zu)\n", err.c_str(), slot1_state.size(), df.state_bytes());
        cudaFree(taps_alt);
        cudaFree(taps);
        return 1;
    }
    if (!df.set_slot(0, err)) {
        std::fprintf(stderr, "switch back to slot 0: %s\n", err.c_str());
        cudaFree(taps_alt);
        cudaFree(taps);
        return 1;
    }
    std::vector<uint8_t> slot0_after;
    if (!df.save_state(slot0_after, err, 2) || slot0_after != slot0_before) {
        std::fprintf(stderr, "slot 1 context changed slot 0's ring: %s\n", err.c_str());
        cudaFree(taps_alt);
        cudaFree(taps);
        return 1;
    }
    if (!df.set_slot(1, err)) {
        std::fprintf(stderr, "switch to slot 1: %s\n", err.c_str());
        cudaFree(taps_alt);
        cudaFree(taps);
        return 1;
    }
    df.reset();
    if (!df.restore_state(slot1_state, 2, err)) {
        std::fprintf(stderr, "restore slot 1 after reset: %s\n", err.c_str());
        cudaFree(taps_alt);
        cudaFree(taps);
        return 1;
    }
    std::vector<uint8_t> slot1_roundtrip;
    if (!df.save_state(slot1_roundtrip, err, 2) || slot1_roundtrip != slot1_state) {
        std::fprintf(stderr, "slot 1 save/restore did not preserve its ring: %s\n", err.c_str());
        cudaFree(taps_alt);
        cudaFree(taps);
        return 1;
    }
    if (!df.set_slot(0, err)) {
        std::fprintf(stderr, "switch to slot 0 after slot 1 restore: %s\n", err.c_str());
        cudaFree(taps_alt);
        cudaFree(taps);
        return 1;
    }
    std::vector<uint8_t> slot0_final;
    if (!df.save_state(slot0_final, err, 2) || slot0_final != slot0_before) {
        std::fprintf(stderr, "slot 1 restore changed slot 0's ring: %s\n", err.c_str());
        cudaFree(taps_alt);
        cudaFree(taps);
        return 1;
    }
    cudaFree(taps_alt);
    cudaFree(taps);
    std::printf("dflash2_state_test: save/restore/rewind and two-slot isolation passed (%zu bytes per slot)\n", state.size());
    return 0;
}
