// The hipBLASLt tuning table's token window, synthetic tables: no model, no GPU.  A row answers a call only
// within a factor of two of the bucket it was measured at, so a bucket the tuner dropped cannot be answered by a
// row measured 25x away (that measured slower than hipBLASEx).  Two knobs change it, both here:
// STRATA_HIPBLASLT_WINDOW=<factor> (0 / off: no window) and the forced-off case for a KV cache below int8/q8
// (--kv q4_0, --kv k8v4), which must win over the variable.  The window is read per call, so a process that sets
// the variable after startup - this test, a bench harness - sees it.
#include "hipblaslt_tuning.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#if defined(_WIN32)
#include <stdlib.h>
#endif

namespace fs = std::filesystem;

using strata::prefill::hipblaslt::InputType;
using strata::prefill::hipblaslt::TuningTable;
using strata::prefill::hipblaslt::kv_is_low_precision;
using strata::prefill::hipblaslt::set_low_precision_kv;
using strata::prefill::hipblaslt::token_window;
using strata::prefill::hipblaslt::token_window_source;

namespace {

// The shape used by the arch-family test, written to a scratch directory (portable: no /tmp, no getpid()).
struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-tuning-window-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value == nullptr ? "" : value);
#else
    if (value == nullptr) unsetenv(name);
    else setenv(name, value, 1);
#endif
}

// A row answers `want` bucket (0: no row).  The table below holds the bucket families a real one does: a lone
// 4096 row, a lone 16384 row, and two rows of one shape (T=512 and T=8192) so the nearest bucket is observable.
long long resolved(const TuningTable& table, InputType type, int n, int k, int ldy, int t) {
    const auto* row = table.closest(type, n, k, ldy, t);
    return row == nullptr ? 0 : row->t_bucket;
}

struct Case {
    const char* env;      // STRATA_HIPBLASLT_WINDOW, or nullptr for unset
    bool low_precision;   // the KV cache is below int8/q8
    InputType type;
    int n, k, ldy, t;
    long long want;
    const char* why;
};

}  // namespace

int main() {
    TempDir dir;
    const fs::path path = dir.path / "window.txt";
    {
        std::ofstream out(path);
        out << "# solution IDs are scoped to this hipBLASLt version and device architecture\n"
            << "STRATA_HIPBLASLT_TUNING_V1 gfx1201 100500\n"
            << "bf16 48 2560 96 4096 110508\n"     // the lone row the window cases probe
            << "f16 512 2560 512 16384 135629\n"
            << "f16 12288 2560 12288 32768 135625\n"
            << "f16 640 2560 640 512 135740\n"     // two buckets of one shape: nearest wins, and only one may be
            << "f16 640 2560 640 8192 135358\n";   //   in the window at a time for a given token count
        if (!out) {
            std::printf("FAIL cannot write the synthetic table at %s\n", path.string().c_str());
            return 1;
        }
    }
    TuningTable table;
    std::string err;
    if (!table.load(path.string(), "gfx1201", 100500, err)) {
        std::printf("FAIL table load: %s\n", err.c_str());
        return 1;
    }

    const InputType bf16 = InputType::bf16;
    const InputType f16 = InputType::f16;
    const Case cases[] = {
        // The measured default: the 4096 row's window is [2048, 8192], inclusive at both ends.
        {nullptr, false, bf16, 48, 2560, 96, 4096, 4096, "exact bucket"},
        {nullptr, false, bf16, 48, 2560, 96, 2048, 4096, "the low edge, inclusive"},
        {nullptr, false, bf16, 48, 2560, 96, 2047, 0, "one below the low edge: no row"},
        {nullptr, false, bf16, 48, 2560, 96, 8192, 4096, "the high edge, inclusive"},
        {nullptr, false, bf16, 48, 2560, 96, 8193, 0, "one above the high edge: no row"},
        {nullptr, false, bf16, 48, 2560, 96, 162, 0, "the 25x extrapolation the window exists to stop"},
        // A factor of one: only the exact bucket answers.
        {"1", false, bf16, 48, 2560, 96, 4096, 4096, "1x keeps the exact bucket"},
        {"1", false, bf16, 48, 2560, 96, 4097, 0, "1x above"},
        {"1", false, bf16, 48, 2560, 96, 4095, 0, "1x below"},
        // A wider factor.
        {"4", false, bf16, 48, 2560, 96, 1024, 4096, "4x reaches 1024"},
        {"4", false, bf16, 48, 2560, 96, 1023, 0, "4x stops one below"},
        {"4", false, bf16, 48, 2560, 96, 16384, 4096, "4x reaches 16384"},
        {"4", false, bf16, 48, 2560, 96, 16385, 0, "4x stops one above"},
        // No window at all, however it is spelled.
        {"0", false, bf16, 48, 2560, 96, 1, 4096, "0: a 1-token call takes the 4096 row"},
        {"0", false, bf16, 48, 2560, 96, 1000000, 4096, "0: and a 1M-token call the same one"},
        {"off", false, bf16, 48, 2560, 96, 1, 4096, "\"off\""},
        {"none", false, bf16, 48, 2560, 96, 1, 4096, "\"none\""},
        {"-3", false, bf16, 48, 2560, 96, 1, 4096, "a negative factor"},
        // An unparsable value keeps the measured default rather than opening the table wide.
        {"two", false, bf16, 48, 2560, 96, 2047, 0, "junk keeps 2x: below the edge stays out"},
        {"two", false, bf16, 48, 2560, 96, 8192, 4096, "junk keeps 2x: the edge still answers"},
        // Nearest bucket wins, and the window is what stops it choosing one too far away.
        {nullptr, false, f16, 640, 2560, 640, 1500, 0, "1500 is outside both buckets' windows"},
        {"0", false, f16, 640, 2560, 640, 1500, 512, "no window: the nearer 512 row answers 1500"},
        {"0", false, f16, 640, 2560, 640, 5000, 8192, "no window: 5000 is nearest to 8192"},
        // The forced case: a KV cache below int8/q8 (q4_0, k8v4) turns the window off and the variable cannot
        // reopen it, because extrapolating a row is what that cache's cell size changes the cost of.
        {nullptr, true, bf16, 48, 2560, 96, 1, 4096, "forced off with the variable unset"},
        {"1", true, bf16, 48, 2560, 96, 1, 4096, "forced off beats an exact-match window"},
        {"4", true, bf16, 48, 2560, 96, 1000000, 4096, "forced off beats a wide window"},
        {"two", true, bf16, 48, 2560, 96, 1000000, 4096, "forced off beats a junk value"},
    };
    int failures = 0;
    for (const auto& c : cases) {
        set_env("STRATA_HIPBLASLT_WINDOW", c.env);
        set_low_precision_kv(c.low_precision);
        const long long got = resolved(table, c.type, c.n, c.k, c.ldy, c.t);
        const bool good = got == c.want;
        if (!good) ++failures;
        std::printf("%s %-4s %-5s T=%-8d -> %-6lld want %-6lld  %s\n", good ? "ok  " : "FAIL",
                    c.type == InputType::bf16 ? "bf16" : "f16", c.env == nullptr ? "(2x)" : c.env, c.t, got, c.want,
                    c.why);
    }

    // The reported factor and its source, for the one startup notice the engine prints.
    struct WindowCase {
        const char* env;
        bool low_precision;
        double want;
        const char* want_source;
    };
    const WindowCase window_cases[] = {
        {nullptr, false, 2.0, "default"},
        {"4", false, 4.0, "STRATA_HIPBLASLT_WINDOW"},
        {"0.5", false, 0.5, "STRATA_HIPBLASLT_WINDOW"},
        {"0", false, 0.0, "STRATA_HIPBLASLT_WINDOW"},
        {"off", false, 0.0, "STRATA_HIPBLASLT_WINDOW"},
        {"two", false, 2.0, "STRATA_HIPBLASLT_WINDOW"},
        {nullptr, true, 0.0, "KV cache below int8/q8"},
        {"8", true, 0.0, "KV cache below int8/q8"},
    };
    for (const auto& c : window_cases) {
        set_env("STRATA_HIPBLASLT_WINDOW", c.env);
        set_low_precision_kv(c.low_precision);
        const double factor = token_window();
        const std::string source = token_window_source();
        const bool good = factor == c.want && source == c.want_source;
        if (!good) ++failures;
        std::printf("%s window %-5s low_precision=%d -> %.3g (%s), want %.3g (%s)\n", good ? "ok  " : "FAIL",
                    c.env == nullptr ? "(unset)" : c.env, c.low_precision ? 1 : 0, factor, source.c_str(), c.want,
                    c.want_source);
    }

    // A caller that never touches it gets the measured default back.
    set_env("STRATA_HIPBLASLT_WINDOW", nullptr);
    set_low_precision_kv(false);
    const bool restored = token_window() == 2.0 && !kv_is_low_precision() &&
                          resolved(table, bf16, 48, 2560, 96, 2047) == 0;
    if (!restored) ++failures;
    std::printf("%s default restored after the forced cases\n", restored ? "ok  " : "FAIL");

    const int total = int(sizeof(cases) / sizeof(cases[0])) + int(sizeof(window_cases) / sizeof(window_cases[0])) + 1;
    std::printf("hipblaslt_window_test: %d cases, %d failures\n", total, failures);
    return failures == 0 ? 0 : 1;
}
