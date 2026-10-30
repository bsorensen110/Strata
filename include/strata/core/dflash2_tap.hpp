// include/strata/core/dflash2_tap.hpp - the rule record_window applies when it decides whether a layer's completed
// output goes into the DFlash2 drafter's device tap buffer (Verifier::tap_dev_, src/core/verify.cpp).
#pragma once

#include <cstdint>
#include <vector>

namespace strata::core {

/// Whether this build compiles the DFlash2 tap kernel (strata::kernels::df2_tap_gr): the CUDA and HIP builds do,
/// the Metal and SYCL builds do not (the same test as the include of strata/kernels/dflash2.hpp).
inline constexpr bool dflash2_kernels_in_build =
#if defined(STRATA_HAVE_DFLASH2)
    true;
#else
    false;
#endif

/// `taps_on`: Verifier::tap_on_ (set_taps or STRATA_TAP_FILE), `batch_rec`: the recorded window is a batch window
/// (capture_batch).  The layer's completed output is captured when taps are on and the layer is one of the selected
/// tap layers (tap_slot[l] is its slot in the tapped layers, >= 0).  A batch record is not a reason to suppress the
/// capture: its rows are the window's rows in the same row-major order the tap_dev_ allocation sizes for
/// ((row * n_taps + slot) * n_embd).  Only where the build has no DFlash2 tap kernel does a batch record capture
/// nothing: its per-row fallback feeds the STRATA_TAP_FILE research dump, which stays serial-only.
inline bool dflash2_tap_capture(bool taps_on, bool batch_rec, int64_t l, const std::vector<int>& tap_slot,
                                bool kernels_in_build = dflash2_kernels_in_build) {
    if (!taps_on) return false;
    if (l < 0 || l >= (int64_t) tap_slot.size()) return false;
    if (tap_slot[(size_t) l] < 0) return false;
    return !batch_rec || kernels_in_build;
}

}  // namespace strata::core
