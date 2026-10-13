// Include the real host-only calibration/test encoder selected at compile time.
#include <cstdint>
#include <cstdio>
#include <limits>
#define main included_fixture_main
#include BF16_ENCODER_SOURCE
#undef main

int main() {
    struct Sample { float value; uint16_t expected; };
    const Sample samples[] = {
        {0.0f, 0x0000}, {-0.0f, 0x8000}, {0.5f, 0x3f00}, {-0.5f, 0xbf00},
        {1.0f, 0x3f80}, {-1.0f, 0xbf80}, {1.00390625f, 0x3f80},
        {1.01171875f, 0x3f82}, {std::numeric_limits<float>::infinity(), 0x7f80},
    };
    for (const auto& sample : samples) {
        const uint16_t actual = BF16_ENCODER_FUNCTION(sample.value, true);
        if (actual != sample.expected) {
            std::fprintf(stderr, "BF16 host encoding failure value=%.9g actual=0x%04x expected=0x%04x\n",
                         sample.value, actual, sample.expected);
            return 1;
        }
    }
    const uint16_t nan = BF16_ENCODER_FUNCTION(std::numeric_limits<float>::quiet_NaN(), true);
    if ((nan & 0x7f80) != 0x7f80 || !(nan & 0x007f)) return 1;
    std::puts("BF16 host encoder: signed values, RNE ties, infinity and NaN passed");
    return 0;
}
