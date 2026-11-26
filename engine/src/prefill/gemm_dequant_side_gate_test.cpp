// The HIP dequant-side opt-in accepts only the exact documented value "1".
#include "strata/prefill/gemm.hpp"

#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    if (argc != 2 || (std::strcmp(argv[1], "--expect-on") != 0 && std::strcmp(argv[1], "--expect-off") != 0)) {
        std::fprintf(stderr, "usage: gemm_dequant_side_gate_test --expect-on|--expect-off\n");
        return 2;
    }
    const bool expected = std::strcmp(argv[1], "--expect-on") == 0;
    const bool actual = strata::prefill::gemm_dequant_side();
    std::printf("gemm_dequant_side gate: expected=%d actual=%d\n", expected, actual);
    return actual == expected ? 0 : 1;
}
