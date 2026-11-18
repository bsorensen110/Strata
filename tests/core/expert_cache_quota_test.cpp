#include "strata/core/expert_cache.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

int main() {
  using namespace strata::core;
  int failed = 0;
  auto check = [&](bool ok, const char *name) {
    std::printf("%s: %s\n", ok ? "PASS" : "FAIL", name);
    failed += !ok;
  };
  auto bytes_used = [](const std::vector<int64_t> &quotas,
                       const std::vector<int64_t> &costs) {
    uint64_t bytes = 0;
    for (size_t i = 0; i < quotas.size(); ++i)
      bytes += (uint64_t)quotas[i] * (uint64_t)costs[i];
    return bytes;
  };
  auto slots_used = [](const std::vector<int64_t> &quotas) {
    int64_t slots = 0;
    for (const int64_t q : quotas)
      slots += q;
    return slots;
  };

  const std::vector<int64_t> equal_costs{10, 10, 10};
  const std::vector<int64_t> uniform =
      uniform_expert_cache_quotas(equal_costs, 8, 90);
  std::vector<double> flat_heat(3 * 8, 1.0);
  const std::vector<int64_t> flat =
      learned_expert_cache_quotas(flat_heat, equal_costs, 8, 90);
  check(uniform == std::vector<int64_t>({3, 3, 3}) && flat == uniform &&
            slots_used(flat) == slots_used(uniform),
        "flat synthetic heat matches uniform quotas and slot sum");
  check(bytes_used(flat, equal_costs) == 90,
        "flat heat consumes exactly the byte budget");

  const std::vector<int64_t> unequal_costs{10, 20, 10};
  std::vector<double> unequal_heat(3 * 8, 0.1);
  for (int e = 0; e < 8; ++e)
    unequal_heat[(size_t)e] = 100.0 - e;
  const std::vector<int64_t> unequal =
      learned_expert_cache_quotas(unequal_heat, unequal_costs, 8, 80);
  const std::vector<int64_t> unequal_uniform =
      uniform_expert_cache_quotas(unequal_costs, 8, 80);
  check(!unequal.empty() && bytes_used(unequal, unequal_costs) <= 80,
        "unequal per-layer blob sizes stay inside the byte budget");
  check(slots_used(unequal) == slots_used(unequal_uniform),
        "learned quotas preserve the uniform total slot count");

  std::vector<double> concentrated_heat(3 * 8, 0.0);
  for (int e = 0; e < 8; ++e)
    concentrated_heat[(size_t)e] = 100.0 - e;
  const std::vector<int64_t> concentrated =
      learned_expert_cache_quotas(concentrated_heat, equal_costs, 8, 90);
  check(concentrated == std::vector<int64_t>({7, 1, 1}),
        "extreme heat concentrates slots but preserves every layer's floor");
  check(learned_expert_cache_quotas(flat_heat, equal_costs, 8, 29).empty(),
        "budget below the one-slot-per-layer floor is refused");
  check(learned_expert_cache_quotas(std::vector<double>(24, 0.0), equal_costs,
                                    8, 90) == uniform,
        "zero heat falls back to uniform quotas");

  const auto heat_path = std::filesystem::temp_directory_path() /
                         "strata-expert-cache-quota-test.heat";
  std::error_code ec;
  std::filesystem::remove(heat_path, ec);
  const std::vector<double> saved_heat{1.25, 0.0, 3.5, 2.0, 4.0, 1.0};
  std::string err;
  check(write_expert_profile_heat(heat_path.string(), 2, 3, saved_heat, err),
        "heat sidecar writes");
  std::vector<double> loaded_heat;
  check(read_expert_profile_heat(heat_path.string(), 2, 3, loaded_heat, err) &&
            loaded_heat == saved_heat,
        "heat sidecar round-trips exact per-expert counts");
  check(!read_expert_profile_heat(heat_path.string(), 3, 2, loaded_heat, err),
        "heat sidecar rejects dimensions from another model");
  std::filesystem::remove(heat_path, ec);
  return failed == 0 ? 0 : 1;
}
