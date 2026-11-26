// Host-only source-contract and boundary checks for QSA attention chunk capacity.
// This test never includes a GPU runtime or launches a kernel.
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

bool fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return false;
}

std::string read_file(const char* path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error(std::string("cannot open ") + path);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

size_t matching_delimiter(std::string_view text, size_t open, char left, char right) {
    int depth = 0;
    bool line_comment = false, block_comment = false, string_literal = false, char_literal = false, escaped = false;
    for (size_t i = open; i < text.size(); ++i) {
        const char c = text[i], next = i + 1 < text.size() ? text[i + 1] : '\0';
        if (line_comment) {
            if (c == '\n') line_comment = false;
            continue;
        }
        if (block_comment) {
            if (c == '*' && next == '/') { block_comment = false; ++i; }
            continue;
        }
        if (string_literal || char_literal) {
            if (escaped) { escaped = false; continue; }
            if (c == '\\') { escaped = true; continue; }
            if (string_literal && c == '"') string_literal = false;
            if (char_literal && c == '\'') char_literal = false;
            continue;
        }
        if (c == '/' && next == '/') { line_comment = true; ++i; continue; }
        if (c == '/' && next == '*') { block_comment = true; ++i; continue; }
        if (c == '"') { string_literal = true; continue; }
        if (c == '\'') { char_literal = true; continue; }
        if (c == left) ++depth;
        else if (c == right && --depth == 0) return i;
    }
    throw std::runtime_error("unmatched delimiter in source");
}

std::string function_body(const std::string& source, std::string_view signature) {
    const size_t start = source.find(signature);
    if (start == std::string::npos) throw std::runtime_error("function signature not found: " + std::string(signature));
    const size_t open = source.find('{', start);
    if (open == std::string::npos) throw std::runtime_error("function body not found: " + std::string(signature));
    const size_t close = matching_delimiter(source, open, '{', '}');
    return source.substr(open + 1, close - open - 1);
}

std::string compact(std::string_view text) {
    std::string result;
    for (const unsigned char c : text) if (!std::isspace(c)) result.push_back((char) c);
    return result;
}

bool check_launches(std::string_view body, const char* path_name, bool step_path) {
    // Parse each launch directly from CUDA/HIP's <<<...>>>(...) syntax.
    std::vector<std::vector<std::string>> parsed;
    size_t search = 0;
    while ((search = body.find("<<<", search)) != std::string_view::npos) {
        const size_t config_end = body.find(">>>", search + 3);
        if (config_end == std::string_view::npos) return fail(std::string(path_name) + ": unterminated launch configuration");
        size_t open = config_end + 3;
        while (open < body.size() && std::isspace((unsigned char) body[open])) ++open;
        if (open == body.size() || body[open] != '(') return fail(std::string(path_name) + ": launch has no argument list");
        const size_t close = matching_delimiter(body, open, '(', ')');
        const std::string_view args(body.data() + open + 1, close - open - 1);
        std::vector<std::string> one;
        int parens = 0, brackets = 0, braces = 0;
        size_t item = 0;
        for (size_t i = 0; i <= args.size(); ++i) {
            const char c = i < args.size() ? args[i] : ',';
            if (c == '(') ++parens;
            else if (c == ')') --parens;
            else if (c == '[') ++brackets;
            else if (c == ']') --brackets;
            else if (c == '{') ++braces;
            else if (c == '}') --braces;
            if (c == ',' && parens == 0 && brackets == 0 && braces == 0) {
                one.push_back(compact(args.substr(item, i - item)));
                item = i + 1;
            }
        }
        parsed.push_back(std::move(one));
        search = close + 1;
    }
    if (step_path && parsed.size() != 6)
        return fail(std::string(path_name) + ": expected six single-step dispatch variants, found " + std::to_string(parsed.size()));
    if (!step_path && parsed.empty()) return fail(std::string(path_name) + ": no batch dispatch launches found");

    for (size_t i = 0; i < parsed.size(); ++i) {
        const auto& args = parsed[i];
        const bool stride_ok = args.size() == 13 && args[12] == (step_path ? "0" : "stride");
        if (args.size() != 13 || args[10] != "n_chunks" || !stride_ok) {
            std::string details;
            for (const auto& arg : args) details += (details.empty() ? "" : ",") + arg;
            return fail(std::string(path_name) + ": launch " + std::to_string(i + 1) +
                        " has unexpected args [" + details + "]");
        }
        if (args[11] != "(int)cap")
            return fail(std::string(path_name) + ": launch " + std::to_string(i + 1) + " passes '" + args[11] +
                        "' as kernel capacity; expected the positive selection cap (int) cap, not a zero sentinel");
    }
    return true;
}

bool check_chunk_count_clamps(const std::string& source) {
    const std::pair<const char*, const char*> kernels[] = {
        {"attn_chunk_body", "void attn_chunk_body("},
        {"attn_chunk_q8_v3_kernel", "__global__ void __launch_bounds__(THREADS) attn_chunk_q8_v3_kernel("},
        {"attn_chunk_pre75_body", "void attn_chunk_pre75_body("},
    };
    for (const auto& [name, signature] : kernels) {
        const std::string body = function_body(source, signature);
        if (body.find("min(CHUNK_, min(n_ids - c0, cap - c0))") == std::string::npos)
            return fail(std::string(name) + ": chunk count no longer clamps against n_ids and the positive selection cap");
    }
    return true;
}

int chunk_cells(int chunk_width, int n_ids, int chunk_start, int cap) {
    return std::min(chunk_width, std::min(n_ids - chunk_start, cap - chunk_start));
}

bool check_boundary_contract() {
    struct Case { int n_ids, cap, start, expected; const char* label; };
    const Case cases[] = {
        {80, 100, 0, 64, "cap > 0 and IDs below cap: full first chunk"},
        {80, 100, 64, 16, "IDs below cap: final partial chunk"},
        {100, 70, 0, 64, "IDs above cap: first chunk bounded by width"},
        {100, 70, 64, 6, "IDs above cap: final chunk bounded by cap"},
        {131, 200, 128, 3, "final partial chunk bounded by n_ids"},
    };
    for (const auto& c : cases) {
        const int actual = chunk_cells(64, c.n_ids, c.start, c.cap);
        if (actual != c.expected)
            return fail(std::string("chunk-boundary contract failed for ") + c.label + ": got " + std::to_string(actual));
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: qsa_attn_cap_contract_test <qsa_decode_attn.cu>\n";
        return 2;
    }
    try {
        const std::string source = read_file(argv[1]);
        const std::string step = function_body(source, "void qsa_decode_attn_step_impl(");
        const std::string batch = function_body(source, "void launch_chunks_batch_impl(");
        if (!check_boundary_contract()) return 1;
        if (!check_chunk_count_clamps(source)) return 1;
        if (!check_launches(step, "single-step path", true)) return 1;
        if (!check_launches(batch, "multi-query batch path", false)) return 1;
        std::cout << "PASS: positive capacity reaches all single-step and batch chunk launches; boundary cases cover full and partial chunks\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
