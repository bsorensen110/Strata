#include "gdn_fixture.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using gdnfx::C;
using gdnfx::HV;
using gdnfx::S;

constexpr std::array<int, 4> kHeads = {0, 17, 32, 47};
constexpr int kNodeCount = 8, kFork = 1, kLeft = 2, kRight = 3, kLeaf = 7;
constexpr std::array<int, kNodeCount> kParent = {-1, 0, kFork, kFork, kLeft, kRight, 4, 6};
constexpr std::array<int, kNodeCount> kForward = {0, 1, 2, 4, 6, 7, 3, 5};
constexpr std::array<int, kNodeCount> kReverse = {0, 1, 3, 5, 2, 4, 6, 7};
constexpr double kTolerance = 1e-12;

using HeadRows = std::array<std::vector<double>, kHeads.size()>;

struct Lineage {
    HeadRows recurrence;
    std::vector<float> hist;
};

struct NodeResult {
    Lineage lineage;
    HeadRows output;
};

using Tree = std::array<NodeResult, kNodeCount>;

template <typename T>
void compare(const std::vector<T>& got, const std::vector<T>& want, const std::string& where,
             double tolerance = 0.0) {
    if (got.size() != want.size()) throw std::runtime_error(where + ": size mismatch");
    for (size_t i = 0; i < got.size(); ++i) {
        const bool finite = std::isfinite(got[i]) && std::isfinite(want[i]);
        const bool match = tolerance == 0.0 ? std::memcmp(&got[i], &want[i], sizeof(T)) == 0
                                            : std::abs(got[i] - want[i]) <= tolerance;
        if (finite && match) continue;
        std::ostringstream msg;
        msg << where << '[' << i << "]: " << (finite ? "mismatch" : "non-finite")
            << std::setprecision(17) << ", got " << got[i] << ", expected " << want[i];
        throw std::runtime_error(msg.str());
    }
}

std::string head_label(size_t h) { return " head " + std::to_string(kHeads[h]); }

void compare_lineage(const Lineage& got, const Lineage& want, const std::string& where) {
    compare(got.hist, want.hist, where + " conv history");
    for (size_t h = 0; h < kHeads.size(); ++h)
        compare(got.recurrence[h], want.recurrence[h], where + head_label(h) + " recurrence");
}

void compare_node(const NodeResult& got, const NodeResult& want, const std::string& where) {
    compare_lineage(got.lineage, want.lineage, where);
    for (size_t h = 0; h < kHeads.size(); ++h)
        compare(got.output[h], want.output[h], where + head_label(h) + " output");
}

Lineage initial_lineage(const gdnfx::Fixture& raw) {
    Lineage result;
    result.hist = raw.hist;
    for (size_t h = 0; h < kHeads.size(); ++h) {
        auto& state = result.recurrence[h];
        state.resize(S * S);
        for (size_t i = 0; i < S; ++i)
            std::copy_n(raw.state0.data() + (i * HV + kHeads[h]) * S, S, state.data() + i * S);
    }
    return result;
}

gdnfx::Fixture inputs(const gdnfx::Fixture& raw) {
    gdnfx::Fixture f;
    f.dt = raw.dt;
    f.ssm_a = raw.ssm_a;
    f.conv_w = raw.conv_w;
    f.gamma = raw.gamma;
    return f;
}

void resize_rows(gdnfx::Fixture& f, size_t count) {
    f.T = static_cast<int64_t>(count);
    f.qkv.resize(count * C);
    f.ab.resize(count * 2 * HV);
    f.z.resize(count * HV * S);
}

void copy_row(const gdnfx::Fixture& raw, int node, gdnfx::Fixture& f, size_t row) {
    std::copy_n(raw.qkv.data() + node * C, C, f.qkv.data() + row * C);
    std::copy_n(raw.ab.data() + node * 2 * HV, 2 * HV, f.ab.data() + row * 2 * HV);
    std::copy_n(raw.z.data() + node * HV * S, HV * S, f.z.data() + row * HV * S);
}

void advance(const gdnfx::Fixture& raw, int node, gdnfx::Fixture& row, Lineage& state, HeadRows& output) {
    copy_row(raw, node, row, 0);
    row.hist = state.hist;
    for (size_t h = 0; h < kHeads.size(); ++h)
        gdnfx::ref_head(row, kHeads[h], state.recurrence[h], output[h]);
    for (int c = 0; c < C; ++c) {
        float* history = state.hist.data() + c * 3;
        history[0] = history[1];
        history[1] = history[2];
        history[2] = row.qkv[c];
    }
}

Tree walk(const gdnfx::Fixture& raw, const Lineage& live, const std::array<int, kNodeCount>& order) {
    Tree tree, snapshots;
    std::array<bool, kNodeCount> seen{};
    auto row = inputs(raw);
    resize_rows(row, 1);
    for (int node : order) {
        const std::string label = "node " + std::to_string(node);
        if (node < 0 || node >= kNodeCount || seen[node])
            throw std::runtime_error(label + ": invalid or repeated node ID");
        const int parent = kParent[node];
        if (parent < -1 || parent >= kNodeCount || (parent == -1) != (node == 0) ||
            (parent >= 0 && !seen[parent]))
            throw std::runtime_error(label + ": invalid parent or parent not visited first");
        auto& child = tree[node];
        child.lineage = parent == -1 ? live : tree[parent].lineage;
        advance(raw, node, row, child.lineage, child.output);
        snapshots[node] = child;
        seen[node] = true;
    }
    for (int node : order)
        compare_node(tree[node], snapshots[node], "parent/sibling immutability at node " + std::to_string(node));
    return tree;
}

void expect_walk_rejection(const gdnfx::Fixture& raw, const Lineage& live,
                           const std::array<int, kNodeCount>& order, const char* expected) {
    try {
        walk(raw, live, order);
    } catch (const std::runtime_error& error) {
        if (std::strcmp(error.what(), expected) == 0) return;
        throw std::runtime_error(std::string("walk: expected ") + expected + ", got " + error.what());
    }
    throw std::runtime_error(std::string("walk did not throw: ") + expected);
}

void test_topology_rejections(const gdnfx::Fixture& raw, const Lineage& live) {
    static_assert(kParent[0] == -1 && kParent[1] == 0);
    static_assert(kForward[0] == 0 && kForward[1] == 1);
    static_assert(kNodeCount == 8);

    auto child_first = kForward;
    std::swap(child_first[0], child_first[1]);
    expect_walk_rejection(raw, live, child_first, "node 1: invalid parent or parent not visited first");

    auto out_of_range = kForward;
    out_of_range[0] = 8;
    expect_walk_rejection(raw, live, out_of_range, "node 8: invalid or repeated node ID");

    auto repeated = kForward;
    repeated[2] = repeated[1];
    expect_walk_rejection(raw, live, repeated, "node 1: invalid or repeated node ID");
}

gdnfx::RefResult compare_serial_paths(const gdnfx::Fixture& raw, const Tree& tree, const std::string& check) {
    auto path = inputs(raw);
    path.hist = raw.hist;
    path.state0 = raw.state0;
    const std::vector<int> heads(kHeads.begin(), kHeads.end());
    gdnfx::RefResult selected;
    for (int node = 0; node < kNodeCount; ++node) {
        std::vector<int> nodes;
        for (int n = node; n != -1; n = kParent[n]) nodes.push_back(n);
        std::reverse(nodes.begin(), nodes.end());
        resize_rows(path, nodes.size());
        for (size_t t = 0; t < nodes.size(); ++t) copy_row(raw, nodes[t], path, t);
        auto serial = gdnfx::ref_run(path, heads);
        const auto label = check + " node " + std::to_string(node);
        compare(tree[node].lineage.hist, serial.hist, label + " conv history");
        for (size_t h = 0; h < heads.size(); ++h) {
            std::vector<double> output;
            output.reserve(nodes.size() * S);
            for (int n : nodes) {
                const auto& y = tree[n].output[h];
                output.insert(output.end(), y.begin(), y.end());
            }
            compare(output, serial.y[h], label + head_label(h) + " path output", kTolerance);
            compare(tree[node].lineage.recurrence[h], serial.state[h], label + head_label(h) + " recurrence", kTolerance);
        }
        if (node == kLeaf) selected = std::move(serial);
    }
    return selected;
}

double recurrence_error(const Lineage& got, const Lineage& want, const std::string& where) {
    gdnfx::ErrStat error;
    for (size_t h = 0; h < kHeads.size(); ++h)
        for (size_t i = 0; i < got.recurrence[h].size(); ++i)
            error.add(got.recurrence[h][i], want.recurrence[h][i]);
    if (error.nonfinite != 0) throw std::runtime_error(where + ": non-finite recurrence");
    return error.max_abs;
}

void test_case(const char* initial_state) {
    gdnfx::FixtureSpec spec;
    spec.T = kNodeCount;
    spec.seed = 123;
    spec.state = initial_state;
    auto raw = gdnfx::make_fixture(spec);
    if (std::strcmp(initial_state, "zero") == 0) std::fill(raw.hist.begin(), raw.hist.end(), 0.0f);
    const Lineage initial = initial_lineage(raw);
    Lineage live = initial;
    test_topology_rejections(raw, live);
    const Tree forward = walk(raw, live, kForward);
    const auto selected = compare_serial_paths(raw, forward, "serial parity");
    {
        const Tree reverse = walk(raw, live, kReverse);
        for (int node = 0; node < kNodeCount; ++node)
            compare_node(reverse[node], forward[node], "sibling order at node " + std::to_string(node));
    }

    // Deliberately advance both siblings in the same mutable state instead of cloning their parent.
    auto row = inputs(raw);
    resize_rows(row, 1);
    NodeResult shared = forward[kFork];
    advance(raw, kLeft, row, shared.lineage, shared.output);
    compare_node(shared, forward[kLeft], "negative control first sibling");
    advance(raw, kRight, row, shared.lineage, shared.output);
    if (recurrence_error(shared.lineage, forward[kRight].lineage, "negative control") <= kTolerance)
        throw std::runtime_error("negative control: shared recurrence did not expose sibling leakage");
    if (shared.lineage.hist == forward[kRight].lineage.hist)
        throw std::runtime_error("negative control: shared conv history did not expose sibling leakage");

    for (int c = 0; c < C; ++c) raw.qkv[kLeft * C + c] += 0.75f;
    const Tree perturbed = walk(raw, live, kForward);
    compare_serial_paths(raw, perturbed, "perturbed serial parity");
    for (int node : {0, kFork, kRight, 5})
        compare_node(perturbed[node], forward[node], "perturbation isolation at node " + std::to_string(node));
    if (recurrence_error(perturbed[kLeaf].lineage, forward[kLeaf].lineage, "perturbation") <= kTolerance)
        throw std::runtime_error("perturbation did not reach the selected leaf");

    compare_lineage(live, initial, "live state before publication");
    live = forward[kLeaf].lineage;
    compare(live.hist, selected.hist, "published leaf conv history");
    for (size_t h = 0; h < kHeads.size(); ++h)
        compare(live.recurrence[h], selected.state[h], "published leaf" + head_label(h) + " recurrence", kTolerance);
}

}  // namespace

int main() {
    for (const char* state : {"random", "zero"}) {
        try {
            test_case(state);
        } catch (const std::runtime_error& error) {
            std::fprintf(stderr, "gdn_tree_ref_test (%s initial state): %s\n", state, error.what());
            return 1;
        }
    }
    std::puts("gdn_tree_ref_test: lineage parity, isolation, and publication checks passed");
    return 0;
}
