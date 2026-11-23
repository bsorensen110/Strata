#include "strata/core/expert_cache.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <random>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

struct TempDirGuard {
    std::filesystem::path path;
    ~TempDirGuard() {
        if (!path.empty()) {
            std::error_code ignored;
            std::filesystem::remove_all(path, ignored);
        }
    }
};

std::filesystem::path create_unique_temp_dir(std::error_code& ec) {
    const std::filesystem::path base = std::filesystem::temp_directory_path(ec);
    if (ec || base.empty()) {
        if (!ec) ec = std::make_error_code(std::errc::no_such_file_or_directory);
        return {};
    }

    const auto stamp = static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    std::random_device rd;
    const std::uint64_t seed =
        stamp ^ (static_cast<std::uint64_t>(rd()) << 32) ^ static_cast<std::uint64_t>(rd());
    std::mt19937_64 rng(seed);

    std::error_code last_ec = std::make_error_code(std::errc::file_exists);
    for (int attempt = 0; attempt < 64; ++attempt) {
        const std::filesystem::path candidate =
            base / ("strata-expert-heat-bootstrap-test-" + std::to_string(stamp) + "-" + std::to_string(rng()));
        std::error_code mk_ec;
        if (std::filesystem::create_directories(candidate, mk_ec) && !mk_ec) {
            ec.clear();
            return candidate;
        }
        if (mk_ec) last_ec = mk_ec;
    }
    ec = last_ec;
    return {};
}

std::string read_bytes(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) return {};
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

bool write_bytes(const std::filesystem::path& p, const std::string& data) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out.write(data.data(), (std::streamsize) data.size());
    return (bool) out;
}

}  // namespace

int main() {
    using namespace strata::core;
    int failed = 0;
    auto check = [&](bool ok, const char* name) {
        std::printf("%s: %s\n", ok ? "PASS" : "FAIL", name);
        failed += !ok;
    };

    std::error_code ec;
    TempDirGuard tmp_guard{create_unique_temp_dir(ec)};
    const std::filesystem::path& tmp = tmp_guard.path;
    check(!ec && !tmp.empty() && std::filesystem::is_directory(tmp, ec),
          "creates unique temporary test directory");
    if (ec || tmp.empty() || failed != 0) return 1;

    // 1. profile_rank_prior_heat: validation, unranked=0, determinism, duplicate handling, quota non-uniformity
    {
        std::vector<double> heat;
        std::string err;
        const std::vector<std::pair<int32_t, int32_t>> skewed_ranks{
            {0, 0}, {0, 1}, {0, 2}, {0, 3}, {1, 0}, {1, 1}, {2, 0}, {3, 0}, {0, 0}};
        std::vector<double> heat2;
        check(profile_rank_prior_heat(4, 4, skewed_ranks, heat, err) &&
                  profile_rank_prior_heat(4, 4, skewed_ranks, heat2, err) && heat == heat2,
              "identical ranks produce identical finite prior weights");
        const bool all_valid = !heat.empty() && heat.size() == 16;
        bool values_ok = all_valid;
        for (double v : heat) {
            if (!std::isfinite(v) || v < 0.0) values_ok = false;
        }
        check(values_ok, "prior weights are all finite and nonnegative");
        check(all_valid && heat[0] > heat[1] && heat[1] > heat[4] && heat[4] > heat[8] && heat[8] > 0.0,
              "prior weights strictly decrease with profile rank and keep first duplicate");
        check(all_valid && heat[5] > 0.0 && heat[6] == 0.0 && heat[7] == 0.0 && heat[15] == 0.0,
              "unranked experts receive exact zero prior weight");

        std::vector<double> bad;
        check(!profile_rank_prior_heat(0, 4, skewed_ranks, bad, err), "rejects non-positive layer count");
        check(!profile_rank_prior_heat(4, 65536, skewed_ranks, bad, err), "rejects dimensions over format limit");
        check(!profile_rank_prior_heat(4, 4, {}, bad, err), "rejects empty ranking");
        check(!profile_rank_prior_heat(4, 4, {{4, 0}}, bad, err), "rejects out-of-range layer index");
        check(!profile_rank_prior_heat(4, 4, {{0, -1}}, bad, err), "rejects negative expert index");

        const std::vector<int64_t> costs{10, 10, 10, 10};
        const std::vector<int64_t> uniform = uniform_expert_cache_quotas(costs, 4, 80);
        const std::vector<int64_t> quotas = learned_expert_cache_quotas(heat, costs, 4, 80);
        check(uniform == std::vector<int64_t>({2, 2, 2, 2}) && quotas == std::vector<int64_t>({4, 2, 1, 1}),
              "quota selector produces strictly non-uniform layer quotas from profile-rank prior");
    }

    // 2. Missing heat derives atomically from saved .learned.bin first and records profile-rank prior provenance
    {
        const auto dir = tmp / "case_saved_profile";
        std::filesystem::create_directories(dir, ec);
        const auto input_profile = dir / "base.bin";
        const auto save_profile = dir / "base.learned.bin";
        const auto save_heat = dir / "base.learned.bin.heat";
        const auto save_meta = std::filesystem::path(expert_profile_heat_meta_path(save_heat.string()));
        const std::vector<std::pair<int32_t, int32_t>> input_ranks{{2, 0}, {2, 1}, {1, 0}, {0, 0}};
        const std::vector<std::pair<int32_t, int32_t>> saved_ranks{{0, 0}, {0, 1}, {0, 2}, {1, 0}};
        std::string err;
        check(write_expert_profile(input_profile.string(), 3, 4, input_ranks, err) &&
                  write_expert_profile(save_profile.string(), 3, 4, saved_ranks, err),
              "test fixture writes input and saved profiles");

        const ExpertHeatBootstrapResult boot = bootstrap_expert_profile_heat(
            save_profile.string(), input_profile.string(), input_ranks, 3, 4);
        check(boot.action == ExpertHeatBootstrapResult::Action::DerivedPrior &&
                  boot.provenance == ExpertHeatProvenance::ProfileRankPrior &&
                  boot.source_profile == save_profile.string() && boot.source_kind == "saved-profile",
              "bootstrap prefers saved .learned.bin over input profile when heat sidecar is missing");
        check(std::filesystem::exists(save_heat) && std::filesystem::exists(save_meta) &&
                  !std::filesystem::exists(save_heat.string() + ".tmp") &&
                  !std::filesystem::exists(save_meta.string() + ".tmp"),
              "bootstrap writes STRH and meta sidecars atomically with no leftover .tmp files");

        const std::string meta_txt = read_bytes(save_meta);
        check(meta_txt.find("provenance=profile-rank-prior") != std::string::npos &&
                  meta_txt.find("source_kind=saved-profile") != std::string::npos,
              "meta sidecar marks provenance as profile-rank prior");

        const ExpertHeatResolution res =
            resolve_expert_profile_heat(save_profile.string(), input_profile.string(), 3, 4);
        check(res.found && res.valid && res.path == save_heat.string() &&
                  res.provenance == ExpertHeatProvenance::ProfileRankPrior && res.heat.size() == 12 &&
                  res.heat[0] > res.heat[1] && res.heat[2] > 0.0 && res.heat[8] == 0.0,
              "subsequent quota resolution selects bootstrapped saved-profile prior with unranked zero");

        // Determinism: identical profile produces identical STRH bytes
        const auto dir2 = tmp / "case_saved_profile_det";
        std::filesystem::create_directories(dir2, ec);
        const auto save_profile2 = dir2 / "base.learned.bin";
        const auto save_heat2 = dir2 / "base.learned.bin.heat";
        check(write_expert_profile(save_profile2.string(), 3, 4, saved_ranks, err),
              "determinism fixture writes second saved profile");
        const ExpertHeatBootstrapResult boot2 =
            bootstrap_expert_profile_heat(save_profile2.string(), "", {}, 3, 4);
        check(boot2.action == ExpertHeatBootstrapResult::Action::DerivedPrior &&
                  read_bytes(save_heat) == read_bytes(save_heat2),
              "identical ranks materialize byte-for-byte identical STRH sidecars");
    }

    // 3. Fallback to input profile when saved .learned.bin is absent, wrong dims, or unreadable
    {
        const auto dir = tmp / "case_fallback";
        std::filesystem::create_directories(dir, ec);
        const auto input_profile = dir / "base.bin";
        const auto save_absent = dir / "absent.learned.bin";
        const std::vector<std::pair<int32_t, int32_t>> input_ranks{{1, 0}, {1, 1}, {0, 0}};
        std::string err;
        check(write_expert_profile(input_profile.string(), 2, 3, input_ranks, err),
              "fallback fixture writes input profile");

        const ExpertHeatBootstrapResult boot_absent = bootstrap_expert_profile_heat(
            save_absent.string(), input_profile.string(), input_ranks, 2, 3);
        check(boot_absent.action == ExpertHeatBootstrapResult::Action::DerivedPrior &&
                  boot_absent.source_kind == "input-profile" &&
                  boot_absent.source_profile == input_profile.string(),
              "falls back to input profile when save profile does not exist");

        const auto save_wrong_dims = dir / "wrongdims.learned.bin";
        check(write_expert_profile(save_wrong_dims.string(), 3, 3, {{0, 0}}, err),
              "writes wrong-dimension save profile");
        const ExpertHeatBootstrapResult boot_wrong = bootstrap_expert_profile_heat(
            save_wrong_dims.string(), input_profile.string(), input_ranks, 2, 3);
        check(boot_wrong.action == ExpertHeatBootstrapResult::Action::DerivedPrior &&
                  boot_wrong.source_kind == "input-profile",
              "falls back to input profile when save profile has mismatched dimensions");

        const auto save_corrupt = dir / "corrupt.learned.bin";
        check(write_bytes(save_corrupt, "BAD_PROFILE"), "writes unreadable corrupt save profile");
        const ExpertHeatBootstrapResult boot_corrupt = bootstrap_expert_profile_heat(
            save_corrupt.string(), input_profile.string(), input_ranks, 2, 3);
        check(boot_corrupt.action == ExpertHeatBootstrapResult::Action::DerivedPrior &&
                  boot_corrupt.source_kind == "input-profile",
              "falls back to input profile when save profile is corrupt/unreadable");

        const ExpertHeatBootstrapResult boot_none =
            bootstrap_expert_profile_heat((dir / "no_save.bin").string(), "", {}, 2, 3);
        check(boot_none.action == ExpertHeatBootstrapResult::Action::SkippedNoProfile,
              "skips cleanly when neither save nor input profile is available");
    }

    // 4. Existing valid runtime heat is retained & never overwritten; zero-heat exit does not clobber prior;
    //    runtime save upgrades provenance cleanly
    {
        const auto dir = tmp / "case_retention_and_upgrade";
        std::filesystem::create_directories(dir, ec);
        const auto save_profile = dir / "learned.bin";
        const auto save_heat = dir / "learned.bin.heat";
        const auto save_meta = std::filesystem::path(expert_profile_heat_meta_path(save_heat.string()));
        std::string err;
        check(write_expert_profile(save_profile.string(), 2, 2, {{0, 0}, {0, 1}}, err),
              "writes profile for lifecycle test");

        // Bootstrap prior
        const auto b0 = bootstrap_expert_profile_heat(save_profile.string(), "", {}, 2, 2);
        const std::string prior_bytes = read_bytes(save_heat);
        const std::string prior_meta = read_bytes(save_meta);
        check(b0.action == ExpertHeatBootstrapResult::Action::DerivedPrior && !prior_bytes.empty(),
              "lifecycle starts with bootstrapped prior");

        // Exit with zero runtime routing must not destroy prior or upgrade provenance
        bool skipped_zero = false;
        check(save_runtime_expert_profile_heat(save_profile.string(), 2, 2, {0.0, 0.0, 0.0, 0.0}, 4,
                                               skipped_zero, err) &&
                  skipped_zero,
              "zero-heat runtime save skips overwrite");
        check(read_bytes(save_heat) == prior_bytes && read_bytes(save_meta) == prior_meta,
              "zero-heat exit preserves prior STRH bytes and profile-rank prior metadata");

        // Real runtime routing save replaces prior and marks runtime
        const std::vector<double> observed{0.0, 1.0, 42.0, 17.0};
        skipped_zero = true;
        check(save_runtime_expert_profile_heat(save_profile.string(), 2, 2, observed, 4, skipped_zero, err) &&
                  !skipped_zero,
              "positive runtime heat saves cleanly");

        std::vector<double> loaded_heat;
        ExpertHeatProvenance prov = ExpertHeatProvenance::None;
        std::string src;
        check(read_expert_profile_heat_with_provenance(save_heat.string(), 2, 2, loaded_heat, prov, src, err) &&
                  prov == ExpertHeatProvenance::Runtime && loaded_heat == observed,
              "provenance upgrades to Runtime only after real routing heat save");

        // Next startup retains runtime heat and does not overwrite it even with a newer profile ranking
        check(write_expert_profile(save_profile.string(), 2, 2, {{1, 1}, {1, 0}}, err), "updates saved profile");
        const std::string runtime_bytes = read_bytes(save_heat);
        const auto b1 = bootstrap_expert_profile_heat(save_profile.string(), "", {}, 2, 2);
        const auto res = resolve_expert_profile_heat(save_profile.string(), "", 2, 2);
        const auto res_same = resolve_expert_profile_heat(save_profile.string(), save_profile.string(), 2, 2);
        check(b1.action == ExpertHeatBootstrapResult::Action::RetainedValid &&
                  read_bytes(save_heat) == runtime_bytes && res.found && res.valid &&
                  res.provenance == ExpertHeatProvenance::Runtime && res.heat == observed &&
                  res_same.valid && res_same.path == save_heat.string() &&
                  res_same.provenance == ExpertHeatProvenance::Runtime,
              "existing valid runtime heat is retained byte-for-byte and preferred on restart (save-only and same-path)");
    }

    // 5. Legacy unmarked <expert-profile>.heat compatibility & deterministic precedence
    {
        const auto dir = tmp / "case_precedence";
        std::filesystem::create_directories(dir, ec);
        const auto input_profile = dir / "base.bin";
        const auto save_profile = dir / "learned.bin";
        const auto input_heat = dir / "base.bin.heat";
        const auto save_heat = dir / "learned.bin.heat";
        std::string err;
        check(write_expert_profile(input_profile.string(), 2, 2, {{0, 0}, {0, 1}}, err) &&
                  write_expert_profile(save_profile.string(), 2, 2, {{1, 0}, {1, 1}}, err),
              "precedence fixture writes both profiles");

        const std::vector<double> legacy_heat{9.0, 3.0, 1.0, 0.5};
        check(write_expert_profile_heat(input_heat.string(), 2, 2, legacy_heat, err),
              "writes legacy unmarked input-profile heat without .meta");

        // Legacy unmarked input heat is preserved and selected without shadowing it with a prior
        const auto b_leg = bootstrap_expert_profile_heat(save_profile.string(), input_profile.string(), {}, 2, 2);
        const auto r_leg = resolve_expert_profile_heat(save_profile.string(), input_profile.string(), 2, 2);
        check(b_leg.action == ExpertHeatBootstrapResult::Action::RetainedValid && !std::filesystem::exists(save_heat) &&
                  r_leg.found && r_leg.valid && r_leg.path == input_heat.string() &&
                  r_leg.provenance == ExpertHeatProvenance::LegacyUnmarked && r_leg.heat == legacy_heat,
              "legacy unmarked input heat is retained and selected without being shadowed by a prior");

        // Even if a prior exists at save_heat, valid runtime/legacy input heat takes precedence over a synthetic prior
        std::vector<double> prior_h;
        check(profile_rank_prior_heat(2, 2, {{1, 0}}, prior_h, err) &&
                  write_expert_profile_heat_with_provenance(save_heat.string(), 2, 2, prior_h,
                                                            ExpertHeatProvenance::ProfileRankPrior,
                                                            save_profile.string(), "saved-profile", 1, err),
              "writes synthetic prior alongside legacy input heat");
        const auto r_prior_vs_legacy =
            resolve_expert_profile_heat(save_profile.string(), input_profile.string(), 2, 2);
        check(r_prior_vs_legacy.valid && r_prior_vs_legacy.path == input_heat.string() &&
                  r_prior_vs_legacy.provenance == ExpertHeatProvenance::LegacyUnmarked,
              "valid measured/legacy input heat takes precedence over saved prior");

        // Runtime heat at save_heat overrides legacy/runtime input heat
        bool skipped = false;
        const std::vector<double> saved_rt{0.0, 2.0, 80.0, 40.0};
        check(save_runtime_expert_profile_heat(save_profile.string(), 2, 2, saved_rt, 2, skipped, err) && !skipped,
              "writes runtime heat at save_profile.heat");
        const auto r_rt_wins = resolve_expert_profile_heat(save_profile.string(), input_profile.string(), 2, 2);
        check(r_rt_wins.valid && r_rt_wins.path == save_heat.string() &&
                  r_rt_wins.provenance == ExpertHeatProvenance::Runtime && r_rt_wins.heat == saved_rt,
              "saved runtime heat takes highest precedence over input heat");
    }

    // 6. Malformed/incompatible heat is NEVER overwritten and fails closed
    {
        const auto dir = tmp / "case_malformed";
        std::filesystem::create_directories(dir, ec);
        const auto input_profile = dir / "base.bin";
        const auto save_profile = dir / "learned.bin";
        const auto save_heat = dir / "learned.bin.heat";
        std::string err;
        check(write_expert_profile(input_profile.string(), 2, 2, {{0, 0}}, err) &&
                  write_expert_profile(save_profile.string(), 2, 2, {{1, 0}}, err),
              "malformed fixture writes profiles");

        const std::string garbage = "CORRUPT_STRH_PAYLOAD";
        check(write_bytes(save_heat, garbage), "writes malformed save heat file");
        const auto b_mal = bootstrap_expert_profile_heat(save_profile.string(), input_profile.string(), {}, 2, 2);
        const auto r_mal = resolve_expert_profile_heat(save_profile.string(), input_profile.string(), 2, 2);
        check(b_mal.action == ExpertHeatBootstrapResult::Action::MalformedLeftUntouched &&
                  read_bytes(save_heat) == garbage && r_mal.found && !r_mal.valid &&
                  r_mal.provenance == ExpertHeatProvenance::Malformed,
              "malformed save heat is left untouched and fails closed");

        // Wrong-dimension heat file is also left untouched and fails closed
        std::filesystem::remove(save_heat, ec);
        check(write_expert_profile_heat(save_heat.string(), 3, 2, std::vector<double>(6, 1.0), err),
              "writes wrong-dimension STRH file");
        const std::string wrong_dim_bytes = read_bytes(save_heat);
        const auto b_dim = bootstrap_expert_profile_heat(save_profile.string(), input_profile.string(), {}, 2, 2);
        const auto r_dim = resolve_expert_profile_heat(save_profile.string(), input_profile.string(), 2, 2);
        check(b_dim.action == ExpertHeatBootstrapResult::Action::MalformedLeftUntouched &&
                  read_bytes(save_heat) == wrong_dim_bytes && r_dim.found && !r_dim.valid &&
                  r_dim.provenance == ExpertHeatProvenance::Malformed,
              "incompatible dimension heat is left untouched and fails closed");

        // Corrupt .meta on valid STRH is also treated as malformed and left untouched
        std::filesystem::remove(save_heat, ec);
        const auto input_heat = dir / "base.bin.heat";
        const auto input_meta = std::filesystem::path(expert_profile_heat_meta_path(input_heat.string()));
        check(write_expert_profile_heat(input_heat.string(), 2, 2, {1.0, 2.0, 3.0, 4.0}, err) &&
                  write_bytes(input_meta, "strata_heat_meta=1\nprovenance=bogus\n"),
              "writes valid STRH with corrupt meta sidecar on input profile");
        const std::string corrupt_meta_bytes = read_bytes(input_meta);
        const auto b_cmeta = bootstrap_expert_profile_heat(save_profile.string(), input_profile.string(), {}, 2, 2);
        const auto r_cmeta = resolve_expert_profile_heat(save_profile.string(), input_profile.string(), 2, 2);
        check(b_cmeta.action == ExpertHeatBootstrapResult::Action::MalformedLeftUntouched &&
                  !std::filesystem::exists(save_heat) && read_bytes(input_meta) == corrupt_meta_bytes &&
                  r_cmeta.found && !r_cmeta.valid && r_cmeta.provenance == ExpertHeatProvenance::Malformed,
              "malformed input heat metadata is left untouched, not shadowed, and fails closed");
    }

    // 7. Atomic write failure & simulated crash between heat and meta preserve prior provenance and old data
    {
        const auto dir = tmp / "case_atomic_failure";
        std::filesystem::create_directories(dir, ec);
        const auto save_profile = dir / "learned.bin";
        const auto save_heat = dir / "learned.bin.heat";
        const auto save_meta = std::filesystem::path(expert_profile_heat_meta_path(save_heat.string()));
        std::string err;
        check(write_expert_profile(save_profile.string(), 2, 2, {{0, 0}, {1, 0}}, err),
              "atomic failure fixture writes profile");

        // 7a. Failure during bootstrap leaves neither .heat nor orphan .meta
        const auto heat_tmp_blocker = std::filesystem::path(save_heat.string() + ".tmp");
        std::filesystem::create_directories(heat_tmp_blocker, ec);
        write_bytes(heat_tmp_blocker / "keep", "x");
        const auto b_fail = bootstrap_expert_profile_heat(save_profile.string(), "", {}, 2, 2);
        check(b_fail.action == ExpertHeatBootstrapResult::Action::Failed && !std::filesystem::exists(save_heat) &&
                  !std::filesystem::exists(save_meta),
              "atomic failure during prior bootstrap leaves no partial heat or orphan meta");
        std::filesystem::remove_all(heat_tmp_blocker, ec);

        // 7b. Successful prior bootstrap, followed by runtime save blocked on meta.tmp -> preserves old heat & meta!
        const auto b_ok = bootstrap_expert_profile_heat(save_profile.string(), "", {}, 2, 2);
        const std::string prior_heat_bytes = read_bytes(save_heat);
        const std::string prior_meta_bytes = read_bytes(save_meta);
        check(b_ok.action == ExpertHeatBootstrapResult::Action::DerivedPrior, "prior established for atomic fail test");

        const auto meta_tmp_blocker = std::filesystem::path(save_meta.string() + ".tmp");
        std::filesystem::create_directories(meta_tmp_blocker, ec);
        write_bytes(meta_tmp_blocker / "keep", "x");
        bool skipped = false;
        check(!save_runtime_expert_profile_heat(save_profile.string(), 2, 2, {5.0, 6.0, 7.0, 8.0}, 2, skipped, err),
              "runtime save fails cleanly when meta.tmp staging is blocked");
        check(read_bytes(save_heat) == prior_heat_bytes && read_bytes(save_meta) == prior_meta_bytes &&
                  !std::filesystem::exists(heat_tmp_blocker),
              "failed runtime save preserves prior heat, prior meta, and cleans temp file");
        std::filesystem::remove_all(meta_tmp_blocker, ec);

        // 7c. Simulated mid-commit crash where .heat replaced prior but .meta still says profile-rank-prior:
        // must NEVER silently upgrade provenance to Runtime!
        check(write_expert_profile_heat(save_heat.string(), 2, 2, {9.0, 8.0, 7.0, 6.0}, err),
              "simulates interrupted commit where .heat updated before .meta");
        std::vector<double> loaded;
        ExpertHeatProvenance prov = ExpertHeatProvenance::None;
        std::string src;
        check(read_expert_profile_heat_with_provenance(save_heat.string(), 2, 2, loaded, prov, src, err) &&
                  prov == ExpertHeatProvenance::ProfileRankPrior,
              "crash between heat rename and meta rename keeps provenance as profile-rank prior (never runtime)");

        // 7d. Clean runtime write after recovery restores matching checksum and marks Runtime
        check(save_runtime_expert_profile_heat(save_profile.string(), 2, 2, {9.0, 8.0, 7.0, 6.0}, 2, skipped, err) &&
                  read_expert_profile_heat_with_provenance(save_heat.string(), 2, 2, loaded, prov, src, err) &&
                  prov == ExpertHeatProvenance::Runtime,
              "clean runtime save recovers and updates provenance to Runtime");

        // 7e. Tampered STRH payload under Runtime metadata is caught via checksum mismatch (Malformed)
        check(write_expert_profile_heat(save_heat.string(), 2, 2, {1.0, 2.0, 3.0, 4.0}, err),
              "rewrites STRH payload without updating Runtime meta");
        check(!read_expert_profile_heat_with_provenance(save_heat.string(), 2, 2, loaded, prov, src, err) &&
                  prov == ExpertHeatProvenance::Malformed,
              "runtime metadata checksum mismatch is rejected as Malformed");
    }

    return failed == 0 ? 0 : 1;
}
