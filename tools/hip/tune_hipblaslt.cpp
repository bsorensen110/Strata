#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include "../../include/strata/kernels/bf16_bits.hpp"
#include <hip/hip_version.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace {
#define HIP_CHECK(call) do { const hipError_t e = (call); if (e != hipSuccess) { \
    std::fprintf(stderr, "HIP %s:%d: %s: %s\n", __FILE__, __LINE__, #call, hipGetErrorString(e)); std::exit(2); } } while (0)
#define BLAS_CHECK(call) do { const hipblasStatus_t e = (call); if (e != HIPBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "hipBLAS %s:%d: %s: status=%d\n", __FILE__, __LINE__, #call, (int)e); std::exit(2); } } while (0)
#define LT_CHECK(call) do { const hipblasStatus_t e = (call); if (e != HIPBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "hipBLASLt %s:%d: %s: status=%d\n", __FILE__, __LINE__, #call, (int)e); std::exit(2); } } while (0)

struct Buffer {
    void *p = nullptr;
    explicit Buffer(size_t bytes) { if (bytes) HIP_CHECK(hipMalloc(&p, bytes)); }
    ~Buffer() { if (p) (void)hipFree(p); }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
};
struct Events {
    hipEvent_t a = nullptr, b = nullptr;
    Events() { HIP_CHECK(hipEventCreate(&a)); HIP_CHECK(hipEventCreate(&b)); }
    ~Events() { if (a) (void)hipEventDestroy(a); if (b) (void)hipEventDestroy(b); }
};

struct Shape { int t, n, k, ldy; bool bf16; };

// The token ladder the table is written for.  The engine resolves a call to the row whose bucket is NEAREST
// to the real T (TuningTable::closest, ties to the smaller bucket), so a bucket b serves the T range
// ((b_prev+b)/2, (b+b_next)/2].  On this ladder every bucket but the last serves a ~2x span, and the buckets
// 128..1024 carry most of a production session's GEMM calls.
std::vector<int> default_tokens() { return {32, 128, 256, 512, 1024, 2048, 4096, 8192}; }

// The T range a row with bucket b answers, matching the engine's own acceptance window in
// TuningTable::closest ([ceil(b/2), 2b]).  A row is only reached for t within a factor of two of the bucket it
// was measured at, so that - and not the ladder spacing - is the span the row has to hold across.
struct Span { int lo, hi; };
Span span_of(int b) {
    Span s{};
    s.lo = std::max(1, (b + 1) / 2);
    s.hi = 2 * b;
    return s;
}

struct Options {
    size_t workspace = 32U * 1024U * 1024U;
    std::vector<Shape> shapes;
    std::vector<int> tokens = default_tokens();
    std::string shapes_file, tuning_out, hist_file;
    bool explicit_shapes = false;
    // A row is only worth writing when the tuned solution beats the call it replaces - hipBLASEx's
    // hipblasGemmEx - by this factor at EVERY sampled T of the span, not just at the bucket's own T.
    double min_gain = 1.05;
    // Timing rounds.  The per-candidate time is the MEDIAN of these: at a 0.5% run-to-run spread the mean of
    // three let 26% of winners flip between two identical sweeps.
    int reps = 5;
    // Candidates carried from the screen at the bucket point into span validation.
    int top_k = 4;
    // Real production T values taken per span from --t-histogram, most frequent first.
    int hist_points = 3;
    // The winner's worst-case ratio must beat the runner-up's by this much, or the bucket has no clear
    // winner and the row is dropped (the engine then runs hipBLASEx, which measured better here).
    double margin = 1.005;
    // Screen only, at the bucket's own T: the previous behaviour, kept for A/B runs.
    bool no_span = false;
};
struct Err { double rel_l2, max_abs; bool finite; size_t padding_writes; };
struct Best {
    Shape s;
    int heuristic_index, solution_id;
    size_t required_workspace, algo_workspace;
    float mean_ms;
    float baseline_ms;      // hipblasGemmEx at the bucket's own T
    double worst_ratio;     // max over the span's sample points of candidate_ms / baseline_ms (1.0 = parity)
    Err error;
    std::string solution, kernel, config;
};

constexpr int WARMUPS = 2;
constexpr int MAX_ALGOS = 16;
constexpr double REL_L2_TOL = 1e-4;
constexpr double MAX_ABS_TOL = 1e-2;
constexpr float PADDING_CANARY = 123456.25f;

const char *dtype(bool bf16) { return bf16 ? "bf16" : "f16"; }
const char *status_name(hipblasStatus_t s) {
    switch (s) {
        case HIPBLAS_STATUS_SUCCESS: return "success";
        case HIPBLAS_STATUS_NOT_SUPPORTED: return "not_supported";
        case HIPBLAS_STATUS_INVALID_VALUE: return "invalid_value";
        case HIPBLAS_STATUS_ARCH_MISMATCH: return "arch_mismatch";
        default: return "other";
    }
}
std::string json(const std::string &s) {
    std::ostringstream o; o << '"';
    for (unsigned char c : s) {
        if (c == '"') o << "\\\"";
        else if (c == '\\') o << "\\\\";
        else if (c == '\n') o << "\\n";
        else if (c == '\r') o << "\\r";
        else if (c == '\t') o << "\\t";
        else if (c < 0x20) o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (unsigned)c << std::dec;
        else o << (char)c;
    }
    o << '"'; return o.str();
}
std::string config_hex(const hipblasLtMatmulAlgo_t &algo) {
    std::ostringstream o; o << std::hex << std::setfill('0');
    for (uint8_t b : algo.data) o << std::setw(2) << (unsigned)b;
    return o.str();
}
uint16_t input_bits(float x, bool bf16) {
    if (bf16) return strata::kernels::bf16_from_f32(x);
    const __half half = __float2half_rn(x);
    uint16_t bits = 0;
    std::memcpy(&bits, &half, sizeof(bits));
    return bits;
}
void fill(std::vector<uint16_t> &v, uint32_t seed, bool bf16) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : v) x = input_bits(dist(rng), bf16);
}
float time_call(hipStream_t stream, const std::function<void()> &f) {
    Events ev;
    HIP_CHECK(hipEventRecord(ev.a, stream));
    f();
    HIP_CHECK(hipEventRecord(ev.b, stream));
    HIP_CHECK(hipEventSynchronize(ev.b));
    HIP_CHECK(hipDeviceSynchronize());
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, ev.a, ev.b));
    return ms;
}
Err compare(const std::vector<float> &ref, const std::vector<float> &got, int n, int t, int ld) {
    long double d2 = 0, r2 = 0; double max_abs = 0; bool finite = true;
    for (int col = 0; col < t; ++col) for (int row = 0; row < n; ++row) {
        const size_t i = (size_t)col * ld + row;
        const double r = ref[i], g = got[i];
        if (!std::isfinite(r) || !std::isfinite(g)) { finite = false; continue; }
        const double d = g - r;
        if (!std::isfinite(d)) { finite = false; continue; }
        d2 += (long double)d * d; r2 += (long double)r * r;
        max_abs = std::max(max_abs, std::abs(d));
    }
    if (!finite) return {std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::infinity(), false, 0};
    return {std::sqrt((double)(d2 / std::max(r2, 1e-300L))), max_abs, true, 0};
}
size_t padding_writes(const std::vector<float> &got, int n, int t, int ld) {
    size_t writes = 0;
    for (int col = 0; col < t; ++col) for (int row = n; row < ld; ++row)
        if (got[(size_t)col * ld + row] != PADDING_CANARY) ++writes;
    return writes;
}
bool accuracy_ok(const Err &e) {
    return e.finite && e.padding_writes == 0 &&
           e.rel_l2 <= REL_L2_TOL && e.max_abs <= MAX_ABS_TOL;
}

// Median, not mean: at a 0.5% run-to-run spread the mean of three let a quarter of the winners flip between
// two identical sweeps, because a single slow sample moved the mean more than the gap between candidates.
float median_of(std::vector<float> v) {
    if (v.empty()) return std::numeric_limits<float>::infinity();
    std::sort(v.begin(), v.end());
    const size_t m = v.size() / 2;
    return (v.size() % 2) ? v[m] : 0.5f * (v[m-1] + v[m]);
}

// --t-histogram: one "T count" a line, '#' comments.  The engine's own log carries the T values it ran at, so
// the span samples the T values a row will actually serve instead of a synthetic ladder of one point.
bool load_histogram(const std::string &path, std::vector<std::pair<int,long>> &out, std::string &err) {
    std::ifstream f(path);
    if (!f) { err = "cannot open histogram file: " + path; return false; }
    std::string line;
    while (std::getline(f, line)) {
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') continue;
        std::istringstream row(line.substr(first));
        int t = 0; long count = 1;
        if (!(row >> t)) continue;
        if (!(row >> count)) count = 1;
        if (t >= 1) out.emplace_back(t, count);
    }
    if (out.empty()) { err = "no T entries in " + path; return false; }
    return true;
}

// The T values a row with bucket b actually answers: the span's two ends and its middle, plus the most
// frequent real T values inside it.  A ~2x span sampled only at its middle is the failure this exists to fix.
std::vector<int> test_points(int b, const std::vector<std::pair<int,long>> &hist, int hist_points) {
    const Span sp = span_of(b);
    std::vector<int> pts{sp.lo, b, sp.hi};
    std::vector<std::pair<int,long>> inside;
    for (const auto &h : hist) if (h.first >= sp.lo && h.first <= sp.hi) inside.push_back(h);
    std::stable_sort(inside.begin(), inside.end(),
                     [](const std::pair<int,long>&a, const std::pair<int,long>&b2){ return a.second > b2.second; });
    for (const auto &h : inside) {
        if ((int) pts.size() >= 3 + hist_points) break;
        pts.push_back(h.first);
    }
    for (auto &p : pts) p = std::max(1, p);
    std::sort(pts.begin(), pts.end());
    pts.erase(std::unique(pts.begin(), pts.end()), pts.end());
    return pts;
}

// Matrix layouts for one token count.  A candidate is measured at several T values, so they are built per T.
struct Layouts {
    hipblasLtMatrixLayout_t ad = nullptr, bd = nullptr, cd = nullptr;
    bool init(hipDataType it, int n, int k, int ldy, int tt) {
        if (hipblasLtMatrixLayoutCreate(&ad, it, k, n, k) != HIPBLAS_STATUS_SUCCESS) return false;
        if (hipblasLtMatrixLayoutCreate(&bd, it, k, tt, k) != HIPBLAS_STATUS_SUCCESS) return false;
        if (hipblasLtMatrixLayoutCreate(&cd, HIP_R_32F, n, tt, ldy) != HIPBLAS_STATUS_SUCCESS) return false;
        return true;
    }
    ~Layouts() {
        if (ad) (void) hipblasLtMatrixLayoutDestroy(ad);
        if (bd) (void) hipblasLtMatrixLayoutDestroy(bd);
        if (cd) (void) hipblasLtMatrixLayoutDestroy(cd);
    }
};

struct CaseCtx {
    hipblasHandle_t blas; hipblasLtHandle_t lt; hipblasLtMatmulDesc_t op;
    hipDataType it;
    void *da, *db, *dc, *dy, *dws;
    size_t ws_bytes;
    hipStream_t stream;
    int n, k, ldy, reps;
};

// One timed call of a candidate with layouts already built; false if it refuses this shape.
bool one_call(const CaseCtx &c, const Layouts &L, const hipblasLtMatmulAlgo_t &algo, float &ms) {
    const float alpha = 1.f, beta = 0.f;
    auto f = [&] {
        return hipblasLtMatmul(c.lt, c.op, &alpha, c.da, L.ad, c.db, L.bd, &beta, c.dy, L.cd, c.dy, L.cd,
                               &algo, c.dws, c.ws_bytes, c.stream);
    };
    if (f() != HIPBLAS_STATUS_SUCCESS) { (void) hipStreamSynchronize(c.stream); return false; }
    HIP_CHECK(hipStreamSynchronize(c.stream));
    ms = time_call(c.stream, [&] { (void) f(); });
    return true;
}

// hipblasGemmEx at tt: the call the engine makes when it has no usable row, so the row's real alternative.
float baseline_at(const CaseCtx &c, int tt) {
    const float alpha = 1.f, beta = 0.f;
    auto f = [&] {
        BLAS_CHECK(hipblasGemmEx(c.blas, HIPBLAS_OP_T, HIPBLAS_OP_N, c.n, tt, c.k, &alpha, c.da, c.it, c.k,
                                 c.db, c.it, c.k, &beta, c.dc, HIP_R_32F, c.ldy, HIPBLAS_COMPUTE_32F,
                                 HIPBLAS_GEMM_DEFAULT));
    };
    for (int i = 0; i < WARMUPS; ++i) { f(); HIP_CHECK(hipDeviceSynchronize()); }
    std::vector<float> v; v.reserve(c.reps);
    for (int i = 0; i < c.reps; ++i) v.push_back(time_call(c.stream, f));
    return median_of(v);
}

struct Cand {
    int h = -1, id = -1;
    size_t req_ws = 0, max_ws = 0;
    std::string sol, kernel, config;
    hipblasLtMatmulAlgo_t algo{};
    Err error{};
    std::vector<float> times;      // bucket-point rounds
    std::vector<double> ratios;    // candidate_ms / baseline_ms, one per sampled T
    double worst_ratio = 0.0;
    bool usable = true;
    float median_ms() const { return median_of(times); }
};
std::vector<std::string> csv(const std::string &s) {
    std::vector<std::string> out; size_t b = 0;
    for (;;) {
        const size_t p = s.find(',', b);
        out.push_back(s.substr(b, p == std::string::npos ? p : p - b));
        if (p == std::string::npos) return out;
        b = p + 1;
    }
}
int positive(const std::string &s, const char *label) {
    char *e = nullptr; const long v = std::strtol(s.c_str(), &e, 10);
    if (s.empty() || !e || *e || v < 1 || v > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "invalid %s: %s\n", label, s.c_str()); std::exit(2);
    }
    return (int)v;
}
double parse_factor(const std::string &s, const char *label, double minimum) {
    char *e = nullptr; const double v = std::strtod(s.c_str(), &e);
    if (s.empty() || !e || *e || !(v >= minimum)) {
        std::fprintf(stderr, "invalid %s: %s (need >= %g)\n", label, s.c_str(), minimum);
        std::exit(2);
    }
    return v;
}
void add(Options &o, bool bf16, int t, int n, int k, int ldy) {
    if (ldy < n) { std::fprintf(stderr, "ldy must be >= N\n"); std::exit(2); }
    o.shapes.push_back({t,n,k,ldy,bf16});
}
void add_shape(Options &o, const std::string &v) {
    const auto f = csv(v);
    if (f.size() != 3) { std::fprintf(stderr, "--shape expects T,N,K\n"); std::exit(2); }
    int t=positive(f[0],"T"), n=positive(f[1],"N"), k=positive(f[2],"K");
    add(o,false,t,n,k,n); add(o,true,t,n,k,n); o.explicit_shapes=true;
}
void add_case(Options &o, const std::string &v) {
    const auto f = csv(v);
    if (f.size()!=5 || (f[0]!="f16" && f[0]!="bf16")) {
        std::fprintf(stderr, "--case expects dtype,T,N,K,ldy\n"); std::exit(2);
    }
    add(o,f[0]=="bf16",positive(f[1],"T"),positive(f[2],"N"),positive(f[3],"K"),positive(f[4],"ldy"));
    o.explicit_shapes=true;
}
std::vector<int> parse_tokens(const std::string &s) {
    std::vector<int> out;
    for (const auto &x : csv(s)) out.push_back(positive(x,"token bucket"));
    return out;
}
void load_shapes(Options &o) {
    std::ifstream f(o.shapes_file);
    if (!f) { std::fprintf(stderr, "cannot open shapes file: %s\n",o.shapes_file.c_str()); std::exit(2); }
    const std::string text((std::istreambuf_iterator<char>(f)),{});
    const std::regex obj(R"(\{[^{}]*\})"), dtype_re(R"re("dtype"\s*:\s*"(f16|bf16)")re"),
        n_re(R"("N"\s*:\s*([0-9]+))"), k_re(R"("K"\s*:\s*([0-9]+))"),
        ld_re(R"("ldy"\s*:\s*([0-9]+))");
    int count=0;
    for (auto i=std::sregex_iterator(text.begin(),text.end(),obj); i!=std::sregex_iterator(); ++i) {
        const std::string item=i->str(); std::smatch d,n,k,ld;
        if (!std::regex_search(item,d,dtype_re) || !std::regex_search(item,n,n_re) ||
            !std::regex_search(item,k,k_re) || !std::regex_search(item,ld,ld_re)) continue;
        for (int t:o.tokens) add(o,d[1]=="bf16",t,positive(n[1],"N"),positive(k[1],"K"),positive(ld[1],"ldy"));
        ++count;
    }
    if (!count) { std::fprintf(stderr,"no dtype,N,K,ldy entries in %s\n",o.shapes_file.c_str()); std::exit(2); }
    o.explicit_shapes=true;
}
void usage(const char *p) {
    std::printf("Usage: %s [--workspace-mib N] [--shape T,N,K]... [--case dtype,T,N,K,ldy]...\n",p);
    std::printf("       [--shapes-file PATH [--tokens T1,T2,...]] [--tuning-out PATH]\n");
    std::printf("       [--min-gain F] [--reps N] [--top-k N] [--span-points N] [--margin F] [--no-span]\n");
    std::printf("       [--t-histogram PATH]\n");
    std::printf("Defaults: original three shapes, f16+bf16, workspace=32 MiB.\n");
    std::printf("Buckets default to the ladder the engine's nearest-bucket rule turns into ~2x spans:\n");
    std::printf("  32,128,256,512,1024,2048,4096,8192.  A bucket b answers T in ((b_prev+b)/2,(b+b_next)/2].\n");
    std::printf("--t-histogram FILE samples the real T values inside each span (one \"T count\" a line, most\n");
    std::printf("  frequent first, --span-points of them); without it a span is sampled at its two ends and\n");
    std::printf("  its middle only.\n");
    std::printf("A row is kept only when the tuned solution beats hipBLASEx by --min-gain at EVERY sampled T\n");
    std::printf("  of its span (worst case, not the mean), and beats the runner-up by --margin.  A dropped\n");
    std::printf("  row is #981's `default`: the engine has no row and runs hipBLASEx.  --reps rounds are timed\n");
    std::printf("  interleaved and reduced by median.  --no-span ranks at the bucket's own T only (the old\n");
    std::printf("  behaviour), for A/B.\n");
}
Options options(int argc,char **argv) {
    Options o;
    for (int i=1;i<argc;++i) {
        std::string a=argv[i];
        auto next=[&]() -> std::string { if(i+1>=argc){std::fprintf(stderr,"missing option value\n");std::exit(2);} return argv[++i]; };
        if(a=="-h"||a=="--help"){usage(argv[0]);std::exit(0);}
        else if(a=="--workspace-mib"){
            std::string v=next(); char *e=nullptr; unsigned long long x=std::strtoull(v.c_str(),&e,10);
            if(v.empty()||!e||*e||x>std::numeric_limits<size_t>::max()/(1024ULL*1024ULL)){std::fprintf(stderr,"bad workspace MiB\n");std::exit(2);}
            o.workspace=(size_t)x*1024U*1024U;
        } else if(a=="--shape") add_shape(o,next());
        else if(a=="--case") add_case(o,next());
        else if(a=="--shapes-file"){o.shapes_file=next();o.explicit_shapes=true;}
        else if(a=="--tokens") o.tokens=parse_tokens(next());
        else if(a=="--tuning-out") o.tuning_out=next();
        else if(a=="--t-histogram") o.hist_file=next();
        else if(a=="--no-span") o.no_span=true;
        else if(a=="--min-gain") o.min_gain=parse_factor(next(),"--min-gain",1.0);
        else if(a=="--margin") o.margin=parse_factor(next(),"--margin",1.0);
        else if(a=="--reps") o.reps=positive(next(),"reps");
        else if(a=="--top-k") o.top_k=positive(next(),"top-k");
        else if(a=="--span-points") o.hist_points=positive(next(),"span-points");
        else {std::fprintf(stderr,"unknown option: %s\n",a.c_str());usage(argv[0]);std::exit(2);}
    }
    if(!o.shapes_file.empty()) load_shapes(o);
    if(!o.explicit_shapes) {
        struct DefaultShape { int n,k,ldy; bool bf16; };
        const DefaultShape defs[]={{10240,2560,10240,false},{320,10240,320,false},{2560,320,2560,false},
                                {10240,2560,10240,true},{320,10240,320,true},{2560,320,2560,true}};
        for(const auto &s:defs) for(int t:o.tokens) add(o,s.bf16,t,s.n,s.k,s.ldy);
    }
    if(o.shapes.empty()){std::fprintf(stderr,"no shapes requested\n");std::exit(2);}
    return o;
}
std::string arch_name(const char *s) {
    std::string a=s?s:"unknown"; size_t p=a.find(':'); if(p!=std::string::npos)a.resize(p); return a;
}
void emit_candidate(const Shape &s,size_t workspace,int h,int id,size_t req,size_t algows,float ms,const Err &error,
                    const std::string &arch,int version,const std::string &sol,const std::string &kernel,const std::string &config) {
    std::cout<<"candidate_json={\"device_arch\":"<<json(arch)<<",\"hipblaslt_version\":"<<version
      <<",\"dtype\":"<<json(dtype(s.bf16))<<",\"T\":"<<s.t<<",\"N\":"<<s.n<<",\"K\":"<<s.k<<",\"ldy\":"<<s.ldy
      <<",\"workspace_limit_bytes\":"<<workspace<<",\"heuristic_index\":"<<h<<",\"solution_id\":"<<id
      <<",\"required_workspace_bytes\":"<<req<<",\"algo_max_workspace_bytes\":"<<algows
      <<",\"finite\":"<<(error.finite?"true":"false")<<",\"padding_writes\":"<<error.padding_writes
      <<",\"relative_l2\":"<<std::setprecision(12)<<error.rel_l2<<",\"max_abs\":"<<error.max_abs
      <<",\"relative_l2_tolerance\":"<<REL_L2_TOL<<",\"max_abs_tolerance\":"<<MAX_ABS_TOL
      <<",\"mean_ms\":"<<std::setprecision(9)<<ms<<",\"solution_name\":"<<json(sol)
      <<",\"kernel_name\":"<<json(kernel)<<",\"algo_config_hex\":"<<json(config)<<"}\n";
}
void emit_best(const Best &b,size_t workspace,const std::string &arch,int version,int hipver) {
    const auto&s=b.s;
    std::cout<<"best_json={\"device_arch\":"<<json(arch)<<",\"hipblaslt_version\":"<<version
      <<",\"hip_runtime_version\":"<<hipver<<",\"dtype\":"<<json(dtype(s.bf16))
      <<",\"T\":"<<s.t<<",\"N\":"<<s.n<<",\"K\":"<<s.k<<",\"ldy\":"<<s.ldy
      <<",\"workspace_limit_bytes\":"<<workspace<<",\"heuristic_index\":"<<b.heuristic_index
      <<",\"solution_id\":"<<b.solution_id<<",\"required_workspace_bytes\":"<<b.required_workspace
      <<",\"algo_max_workspace_bytes\":"<<b.algo_workspace<<",\"mean_ms\":"<<std::setprecision(9)<<b.mean_ms
      <<",\"finite\":"<<(b.error.finite?"true":"false")<<",\"padding_writes\":"<<b.error.padding_writes
      <<",\"relative_l2\":"<<std::setprecision(12)<<b.error.rel_l2<<",\"max_abs\":"<<b.error.max_abs
      <<",\"relative_l2_tolerance\":"<<REL_L2_TOL<<",\"max_abs_tolerance\":"<<MAX_ABS_TOL
      <<",\"solution_name\":"<<json(b.solution)<<",\"kernel_name\":"<<json(b.kernel)
      <<",\"algo_config_hex\":"<<json(b.config)<<"}\n";
}

// A candidate measured with layouts for one token count: false when it refuses that shape there.
bool candidate_at(const CaseCtx &c, const hipblasLtMatmulAlgo_t &algo, int tt, float &ms) {
    Layouts L;
    if (!L.init(c.it, c.n, c.k, c.ldy, tt)) return false;
    std::vector<float> v; v.reserve(c.reps);
    for (int i = 0; i < c.reps; ++i) {
        float m = 0.f;
        if (!one_call(c, L, algo, m)) return false;
        v.push_back(m);
    }
    ms = median_of(v);
    return true;
}

void run_case(hipblasHandle_t blas,hipblasLtHandle_t lt,const Shape&s,int case_id,size_t ws,hipStream_t stream,
              const std::string&arch,int version,int hipver,const Options&opt,
              const std::vector<std::pair<int,long>>&hist,std::vector<Best>&bests) {
    const int bucket=s.t;                                   // one case is one bucket of the ladder
    const std::vector<int> pts=test_points(bucket,hist,opt.hist_points);
    const int tmax=*std::max_element(pts.begin(),pts.end());  // the span reaches past the bucket
    const size_t na=(size_t)s.k*s.n, nb=(size_t)s.k*tmax, out_elems=(size_t)s.ldy*tmax;
    const hipDataType it=s.bf16?HIP_R_16BF:HIP_R_16F;
    const hipblasOperation_t ta=HIPBLAS_OP_T,tb=HIPBLAS_OP_N;
    std::vector<uint16_t> ha(na),hb(nb);
    uint32_t seed=0x5A17C3U+(uint32_t)case_id*0x10001U+(s.bf16?0xB16U:0xF16U);
    fill(ha,seed,s.bf16);fill(hb,seed^0x9E3779B9U,s.bf16);
    Buffer da(na*sizeof(uint16_t)),db(nb*sizeof(uint16_t)),dc(out_elems*sizeof(float)),
           dy(out_elems*sizeof(float)),dref(out_elems*sizeof(float)),dws(ws);
    const std::vector<float> canary(out_elems,PADDING_CANARY);
    HIP_CHECK(hipMemcpyAsync(da.p,ha.data(),na*sizeof(uint16_t),hipMemcpyHostToDevice,stream));
    HIP_CHECK(hipMemcpyAsync(db.p,hb.data(),nb*sizeof(uint16_t),hipMemcpyHostToDevice,stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    hipblasLtMatmulDesc_t op=nullptr;
    hipblasLtMatmulPreference_t pref=nullptr;
    LT_CHECK(hipblasLtMatmulDescCreate(&op,HIPBLAS_COMPUTE_32F,HIP_R_32F));
    LT_CHECK(hipblasLtMatmulDescSetAttribute(op,HIPBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
    LT_CHECK(hipblasLtMatmulDescSetAttribute(op,HIPBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)));
    LT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
    const uint64_t limit=ws;
    LT_CHECK(hipblasLtMatmulPreferenceSetAttribute(pref,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&limit,sizeof(limit)));
    BLAS_CHECK(hipblasSetStream(blas,stream));

    CaseCtx ctx{blas,lt,op,it,da.p,db.p,dc.p,dy.p,dws.p,ws,stream,s.n,s.k,s.ldy,opt.reps};

    // The reference is hipblasGemmEx at the bucket's own T: that is where the accuracy gate is applied.
    {
        const float alpha=1.f,beta=0.f;
        BLAS_CHECK(hipblasGemmEx(blas,ta,tb,s.n,bucket,s.k,&alpha,da.p,it,s.k,db.p,it,s.k,&beta,dref.p,
                                 HIP_R_32F,s.ldy,HIPBLAS_COMPUTE_32F,HIPBLAS_GEMM_DEFAULT));
        HIP_CHECK(hipStreamSynchronize(stream));
    }
    std::vector<float> ref(out_elems);
    HIP_CHECK(hipMemcpyAsync(ref.data(),dref.p,out_elems*sizeof(float),hipMemcpyDeviceToHost,stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    const size_t baseline_padding=padding_writes(ref,s.n,bucket,s.ldy);
    std::printf("baseline dtype=%s T=%d N=%d K=%d ldy=%d padding_writes=%zu\n",
                dtype(s.bf16),bucket,s.n,s.k,s.ldy,baseline_padding);

    // hipblasGemmEx at every sampled T of the span: each ratio is taken against the call the row replaces.
    std::vector<float> base_ms(pts.size(),0.f);
    for(size_t i=0;i<pts.size();++i)base_ms[i]=baseline_at(ctx,pts[i]);
    const size_t bi=(size_t)(std::find(pts.begin(),pts.end(),bucket)-pts.begin());
    const float base_bucket=base_ms[bi];
    std::printf("case dtype=%s T=%d N=%d K=%d ldy=%d workspace_limit_bytes=%zu baseline=hipblasGemmEx bucket_median_ms=%.4f span_points=",
                dtype(s.bf16),bucket,s.n,s.k,s.ldy,ws,base_bucket);
    for(size_t i=0;i<pts.size();++i)std::printf("%s%d:%.4f",i?",":"",pts[i],base_ms[i]);
    std::printf("\n");

    Layouts Lb;
    bool layouts_ok=Lb.init(it,s.n,s.k,s.ldy,bucket);
    if(!layouts_ok){std::printf("lt_unsupported dtype=%s T=%d N=%d K=%d ldy=%d status=layout_error\n",
                                dtype(s.bf16),bucket,s.n,s.k,s.ldy);}
    hipblasLtMatmulHeuristicResult_t hs[MAX_ALGOS]{};
    int count=0;
    const hipblasStatus_t query=layouts_ok
        ? hipblasLtMatmulAlgoGetHeuristic(lt,op,Lb.ad,Lb.bd,Lb.cd,Lb.cd,pref,MAX_ALGOS,hs,&count)
        : HIPBLAS_STATUS_INVALID_VALUE;
    if(!layouts_ok){ /* nothing to do */ }
    else if(query!=HIPBLAS_STATUS_SUCCESS||count==0){
        std::printf("lt_unsupported dtype=%s T=%d N=%d K=%d ldy=%d status=%s(%d) heuristic_count=%d\n",
          dtype(s.bf16),bucket,s.n,s.k,s.ldy,status_name(query),(int)query,count);
    } else {
        // ---- stage 1: accuracy at the bucket's T, then interleaved rounds for the timing
        std::vector<Cand> cands;
        for(int h=0;h<count;++h){
            if(hs[h].state!=HIPBLAS_STATUS_SUCCESS||hs[h].workspaceSize>ws){
                std::printf("lt_skip dtype=%s T=%d N=%d K=%d ldy=%d heuristic_index=%d status=%s(%d) required_workspace_bytes=%zu\n",
                  dtype(s.bf16),bucket,s.n,s.k,s.ldy,h,status_name(hs[h].state),(int)hs[h].state,hs[h].workspaceSize);
                continue;
            }
            Cand c;
            c.h=h; c.algo=hs[h].algo; c.req_ws=hs[h].workspaceSize; c.max_ws=c.algo.max_workspace_bytes;
            c.id=hipblaslt_ext::getIndexFromAlgo(c.algo);
            c.sol=hipblaslt_ext::getSolutionNameFromAlgo(lt,c.algo);
            c.kernel=hipblaslt_ext::getKernelNameFromAlgo(lt,c.algo);
            c.config=config_hex(c.algo);

            HIP_CHECK(hipMemcpyAsync(dy.p,canary.data(),out_elems*sizeof(float),hipMemcpyHostToDevice,stream));
            HIP_CHECK(hipStreamSynchronize(stream));
            const float alpha=1.f,beta=0.f;
            const hipblasStatus_t check_status=hipblasLtMatmul(lt,op,&alpha,da.p,Lb.ad,db.p,Lb.bd,&beta,dy.p,Lb.cd,dy.p,Lb.cd,&c.algo,dws.p,ws,stream);
            if(check_status!=HIPBLAS_STATUS_SUCCESS){
                std::printf("lt_unsupported dtype=%s T=%d N=%d K=%d ldy=%d heuristic_index=%d solution_id=%d status=%s(%d) phase=accuracy_check\n",
                  dtype(s.bf16),bucket,s.n,s.k,s.ldy,h,c.id,status_name(check_status),(int)check_status);
                continue;
            }
            HIP_CHECK(hipStreamSynchronize(stream));
            std::vector<float> checked_y(out_elems);
            HIP_CHECK(hipMemcpyAsync(checked_y.data(),dy.p,out_elems*sizeof(float),hipMemcpyDeviceToHost,stream));
            HIP_CHECK(hipStreamSynchronize(stream));
            c.error=compare(ref,checked_y,s.n,bucket,s.ldy);
            c.error.padding_writes=padding_writes(checked_y,s.n,bucket,s.ldy);
            if(!accuracy_ok(c.error)){
                std::printf("lt_reject dtype=%s T=%d N=%d K=%d ldy=%d heuristic_index=%d solution_id=%d finite=%s relative_l2=%.12g relative_l2_tolerance=%.3g max_abs=%.12g max_abs_tolerance=%.3g padding_writes=%zu reason=accuracy_gate\n",
                  dtype(s.bf16),bucket,s.n,s.k,s.ldy,h,c.id,c.error.finite?"true":"false",
                  c.error.rel_l2,REL_L2_TOL,c.error.max_abs,MAX_ABS_TOL,c.error.padding_writes);
                continue;
            }
            cands.push_back(c);
        }

        // Every survivor is timed once per round in a rotating order, so a clock or thermal drift lands on
        // all of them alike instead of on whoever happened to be measured last.
        for(int r=0;r<opt.reps && !cands.empty();++r){
            for(size_t j=0;j<cands.size();++j){
                const size_t idx=(j+(size_t)r)%cands.size();
                float ms=0.f;
                if(!one_call(ctx,Lb,cands[idx].algo,ms)){ cands[idx].usable=false; continue; }
                cands[idx].times.push_back(ms);
            }
        }
        std::vector<Cand*> live;
        for(auto&cc:cands) if(cc.usable&&(int)cc.times.size()>=std::max(1,opt.reps/2)) live.push_back(&cc);
        std::sort(live.begin(),live.end(),[](const Cand*a,const Cand*b2){return a->median_ms()<b2->median_ms();});
        for(auto*cc:live)
            std::printf("lt dtype=%s T=%d N=%d K=%d ldy=%d heuristic_index=%d solution_id=%d solution=%s kernel=%s algo_config_hex=%s required_workspace_bytes=%zu finite=%s relative_l2=%.12g max_abs=%.12g median_ms=%.4f rounds=%zu\n",
              dtype(s.bf16),bucket,s.n,s.k,s.ldy,cc->h,cc->id,cc->sol.c_str(),cc->kernel.c_str(),cc->config.c_str(),
              cc->req_ws,cc->error.finite?"true":"false",cc->error.rel_l2,cc->error.max_abs,cc->median_ms(),cc->times.size());

        if(live.empty()){
            std::printf("lt_unsupported dtype=%s T=%d N=%d K=%d ldy=%d status=no_valid_heuristics heuristic_count=%d\n",
                        dtype(s.bf16),bucket,s.n,s.k,s.ldy,count);
        } else {
            std::vector<Cand*> top(live.begin(),live.end());
            if((int)top.size()>opt.top_k)top.resize((size_t)opt.top_k);
            const size_t P=pts.size();

            if(opt.no_span){
                // The previous behaviour, for A/B: rank at the bucket's own T and ignore the span.
                for(auto*cc:top){ cc->ratios.assign(P,0.0); cc->ratios[bi]=(double)cc->median_ms()/(double)base_ms[bi];
                                  cc->worst_ratio=cc->ratios[bi]; }
            } else {
                // ---- stage 2: measure the survivors at every sampled T of the span they will answer
                std::vector<std::unique_ptr<Layouts>> PL(P);
                bool ok=true;
                for(size_t i=0;i<P;++i){
                    auto L=std::unique_ptr<Layouts>(new Layouts());
                    if(!L->init(it,s.n,s.k,s.ldy,pts[i])){ok=false;break;}
                    PL[i]=std::move(L);
                }
                if(!ok){ std::printf("lt_unsupported dtype=%s T=%d N=%d K=%d ldy=%d status=span_layout_error\n",
                                     dtype(s.bf16),bucket,s.n,s.k,s.ldy); top.clear(); }
                else{
                    std::vector<std::vector<float>> acc(top.size()*P);
                    std::vector<char> bad(top.size()*P,0);
                    for(int r=0;r<opt.reps;++r){
                        for(size_t k2=0;k2<top.size()*P;++k2){
                            const size_t idx=(k2+(size_t)r)%(top.size()*P);
                            const size_t a=idx/P, i=idx%P;
                            if(pts[i]==bucket||bad[idx])continue;
                            float ms=0.f;
                            if(!one_call(ctx,*PL[i],top[a]->algo,ms)){bad[idx]=1;continue;}
                            acc[idx].push_back(ms);
                        }
                    }
                    for(size_t a=0;a<top.size();++a){
                        top[a]->usable=true; top[a]->ratios.assign(P,0.0); top[a]->worst_ratio=0.0;
                        for(size_t i=0;i<P;++i){
                            double ratio=0.0;
                            if(pts[i]==bucket)ratio=(double)top[a]->median_ms()/(double)base_ms[i];
                            else{
                                if(bad[a*P+i]||acc[a*P+i].empty()){top[a]->usable=false;break;}
                                ratio=(double)median_of(acc[a*P+i])/(double)base_ms[i];
                            }
                            top[a]->ratios[i]=ratio;
                            top[a]->worst_ratio=std::max(top[a]->worst_ratio,ratio);
                        }
                        if(top[a]->usable){
                            std::printf("lt_span dtype=%s T=%d N=%d K=%d ldy=%d solution_id=%d worst_ratio=%.4f ratios=",
                                        dtype(s.bf16),bucket,s.n,s.k,s.ldy,top[a]->id,top[a]->worst_ratio);
                            for(size_t i=0;i<P;++i)std::printf("%s%d:%.4f",i?",":"",pts[i],top[a]->ratios[i]);
                            std::printf("\n");
                        }
                    }
                }
            }

            Cand* win=nullptr;
            for(auto*cc:top) if(cc->usable&&(!win||cc->worst_ratio<win->worst_ratio)) win=cc;
            double second=std::numeric_limits<double>::infinity();
            for(auto*cc:top) if(cc->usable&&cc!=win) second=std::min(second,cc->worst_ratio);

            if(!win){
                std::printf("lt_drop dtype=%s T=%d N=%d K=%d ldy=%d reason=no_candidate_survives_the_span\n",
                            dtype(s.bf16),bucket,s.n,s.k,s.ldy);
            } else {
                // worst_ratio is candidate/baseline, so the gain the row guarantees at its worst sampled T is
                // its reciprocal.  Both the "beats hipBLASEx everywhere" gate and the "beats the runner-up
                // clearly" gate are conditions on that worst case.
                const double gain=1.0/win->worst_ratio;
                const bool gain_ok=gain>=opt.min_gain;
                const bool clear=!(second<std::numeric_limits<double>::infinity())||
                                  (second/win->worst_ratio>=opt.margin);
                const bool keep=gain_ok&&clear;
                Best b{s,win->h,win->id,win->req_ws,win->max_ws,win->median_ms(),base_bucket,
                       win->worst_ratio,win->error,win->sol,win->kernel,win->config};
                std::printf("lt_best dtype=%s T=%d N=%d K=%d ldy=%d heuristic_index=%d solution_id=%d solution=%s kernel=%s algo_config_hex=%s required_workspace_bytes=%zu algo_max_workspace_bytes=%zu baseline_median_ms=%.4f median_ms=%.4f speedup_vs_blas_at_bucket=%.4fx worst_ratio=%.4f worst_gain=%.4fx finite=%s relative_l2=%.9g max_abs=%.9g row=%s reason=%s\n",
                  dtype(s.bf16),bucket,s.n,s.k,s.ldy,win->h,win->id,win->sol.c_str(),win->kernel.c_str(),
                  win->config.c_str(),win->req_ws,win->max_ws,base_bucket,win->median_ms(),
                  base_bucket/win->median_ms(),win->worst_ratio,gain,win->error.finite?"true":"false",
                  win->error.rel_l2,win->error.max_abs,keep?"kept":"dropped",
                  keep?"ok":(gain_ok?"no_clear_winner":"span_gain_below_min_gain"));
                emit_best(b,ws,arch,version,hipver);
                if(keep)bests.push_back(b);
            }
        }
    }
    (void)hipblasLtMatmulPreferenceDestroy(pref);
    (void)hipblasLtMatmulDescDestroy(op);
}
void tuning_file(const std::string &path,const std::string &arch,int version,const std::vector<Best>&rows){
    if(path.empty())return;
    std::ofstream f(path);
    if(!f){std::fprintf(stderr,"cannot write tuning output: %s\n",path.c_str());std::exit(2);}
    f<<"# solution IDs are scoped to this hipBLASLt version and device architecture\n";
    f<<"STRATA_HIPBLASLT_TUNING_V1 "<<arch<<" "<<version<<"\n";
    for(const auto &b:rows)f<<dtype(b.s.bf16)<<" "<<b.s.n<<" "<<b.s.k<<" "<<b.s.ldy<<" "<<b.s.t<<" "<<b.solution_id<<"\n";
}
} // namespace

int main(int argc,char **argv){
    Options o=options(argc,argv);
    hipblasHandle_t blas=nullptr;hipblasLtHandle_t lt=nullptr;
    BLAS_CHECK(hipblasCreate(&blas));LT_CHECK(hipblasLtCreate(&lt));
    int device=0,hipver=0,version=0;hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDevice(&device));HIP_CHECK(hipGetDeviceProperties(&prop,device));HIP_CHECK(hipRuntimeGetVersion(&hipver));
    const hipblasStatus_t vs=hipblasLtGetVersion(lt,&version);
    if(vs!=HIPBLAS_STATUS_SUCCESS){std::fprintf(stderr,"hipblasLtGetVersion failed: %d\n",(int)vs);return 2;}
    const std::string arch=arch_name(prop.gcnArchName);
    std::printf("meta device_arch=%s hipblaslt_version=%d hipblaslt_header_version=%d.%d.%d hip_runtime_version=%d\n",
      arch.c_str(),version,HIPBLASLT_VERSION_MAJOR,HIPBLASLT_VERSION_MINOR,HIPBLASLT_VERSION_PATCH,hipver);
    hipStream_t stream=nullptr;HIP_CHECK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    std::vector<std::pair<int,long>> hist;
    if(!o.hist_file.empty()){
        std::string err;
        if(!load_histogram(o.hist_file,hist,err)){std::fprintf(stderr,"%s\n",err.c_str());return 2;}
        std::printf("histogram entries=%zu\n",hist.size());
    }
    std::vector<Best>bests;
    for(size_t i=0;i<o.shapes.size();++i)run_case(blas,lt,o.shapes[i],(int)i,o.workspace,stream,arch,version,hipver,o,hist,bests);
    tuning_file(o.tuning_out,arch,version,bests);
    HIP_CHECK(hipStreamSynchronize(stream));HIP_CHECK(hipStreamDestroy(stream));
    LT_CHECK(hipblasLtDestroy(lt));BLAS_CHECK(hipblasDestroy(blas));
    return 0;
}
