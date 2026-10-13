#pragma once

// The hipBLASLt prefill tuning table answers a call with the nearest row of the same (dtype, N, K, ldy).  A
// solution measured at one token count is not necessarily the fastest at a very different one: with rows only at
// 4096 and 8192, a 162-token prompt ran the kernel measured at 4096 - a 25x extrapolation that measured slower
// than hipBLASEx.  So a row only answers a token count inside a window around its bucket: a factor of two unless
// overridden here.
//
//   STRATA_HIPBLASLT_WINDOW=<factor>   the window's factor.  Unset: 2, the measured default.  0, a negative
//                                      value, or "off"/"none"/"no": no window at all - the nearest row of the
//                                      shape answers whatever the token count.
//   set_low_precision_kv(true)         the engine's KV cache is below int8/q8 (`--kv q4_0`, `--kv k8v4`: those
//                                      cells carry 4-bit codes, 816 B/cell for k8v4 against int8's 1056).  The
//                                      window is then forced OFF whatever the variable says: the extrapolation it
//                                      guards against was measured on the wider caches.
//
// A table stays a list of solution IDs: these knobs belong to the process, never to the file, so one table means
// the same thing whatever the KV cache is.

#include <cstdlib>
#include <cstring>

namespace strata::prefill::hipblaslt {

/// Set once by the engine when it parses `--kv` (program/generate.cpp).  The default keeps every other entry
/// point - the tests, tune_hipblaslt, the parity helpers - on the measured 2x behavior.
inline bool& low_precision_kv() {
    static bool enabled = false;
    return enabled;
}

inline void set_low_precision_kv(bool enabled) { low_precision_kv() = enabled; }

inline bool kv_is_low_precision() { return low_precision_kv(); }

/// The window's factor; <= 0 means no window.  Read per call (a prefill chunk resolves a handful of calls, and
/// gemm.cu caches each resolved call): a caller that sets the variable after startup still sees it, and the
/// forced case above is never cached.
inline double token_window() {
    if (low_precision_kv()) return 0.0;
    const char* env = std::getenv("STRATA_HIPBLASLT_WINDOW");
    if (env == nullptr) return 2.0;
    while (*env == ' ' || *env == '\t') ++env;
    if (*env == '\0') return 2.0;
    if (std::strcmp(env, "off") == 0 || std::strcmp(env, "none") == 0 || std::strcmp(env, "no") == 0) return 0.0;
    char* end = nullptr;
    const double factor = std::strtod(env, &end);
    if (end == env || !(factor == factor)) return 2.0;   // unparsable or NaN: keep the default, never a surprise
    return factor > 0.0 ? factor : 0.0;
}

/// Where the current window came from, for the one startup notice gemm.cu prints.
inline const char* token_window_source() {
    if (low_precision_kv()) return "KV cache below int8/q8";
    const char* env = std::getenv("STRATA_HIPBLASLT_WINDOW");
    return env != nullptr && *env != '\0' ? "STRATA_HIPBLASLT_WINDOW" : "default";
}

}  // namespace strata::prefill::hipblaslt
