// src/core/dflash2.cpp - see include/strata/core/dflash2.hpp.
#include "strata/core/dflash2.hpp"

#include "strata/core/batch_admit_merge.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/dflash2.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/platform/cpu_relax.hpp"

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <atomic>
#include <initializer_list>
#include <stdexcept>

namespace strata::core {
namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

bool mapped(size_t bytes, void** h, void** d) {
    if (cudaHostAlloc(h, bytes, cudaHostAllocMapped) != cudaSuccess) return false;
    std::memset(*h, 0, bytes);
    return cudaHostGetDevicePointer(d, *h, 0) == cudaSuccess;
}

constexpr int GGML_Q8_0 = 8;

// Batch d_meta_ layout (ints, non-colliding for up to kBatchMaxSlots = 8 slots / 64 rows; solo paths use [0..64] sequentially):
constexpr int kMetaCtx        = 32;    // [32..95]   64 ctx positions (contiguous with kMetaBlkPos for 128-row rope)
constexpr int kMetaBlkPos     = 96;    // [96..159]  64 block positions
constexpr int kMetaBlkTok     = 160;   // [160..223] 64 block tokens
constexpr int kMetaAnchor     = 224;   // [224..231] 8 slot anchors
constexpr int kMetaEpoch      = 232;   // [232..239] 8 slot epochs
constexpr int kMetaSlotInfo   = 240;   // [240..271] 8 * 4 (slot_idx, c0, n_new, anchor)
constexpr int kMetaPrecompPos = 272;   // [272..335] 64 precomp positions
constexpr int kMetaInts       = 512;

int df2_beam() {
    static const int b = [] {
        const char* e = std::getenv("STRATA_DF2_BEAM");
        if (e && (e[0] == '0' || e[0] == '1')) return 1;
        return 4;
    }();
    return b;
}

// STRATA_DF2_SELECT_TABLE=0: the selector walks with its one-block steps (the walk's scores computed in one parallel launch first is the default: bitwise the same drafts)
bool df2_select_table() {
    static const bool on = [] { const char* e = std::getenv("STRATA_DF2_SELECT_TABLE"); return !(e && e[0] == '0'); }();
    return on;
}

int df2_head_il_mode() {
    static const int mode = [] {
        const char* e = std::getenv("STRATA_DF2_HEAD_IL");
        if (!e || !e[0]) e = std::getenv("STRATA_DFLASH_HEAD_IL");
        if (e && e[0]) return e[0] != '0' ? 1 : 0;
#if defined(STRATA_USE_HIP)
        return 1;
#else
        return -1;
#endif
    }();
    return mode;
}

int df2_head_il_rows() {
    static const int r = [] {
        const char* e = std::getenv("STRATA_DF2_HEAD_IL_ROWS");
        if (e && e[0]) {
            const int v = std::atoi(e);
            if (v == 1 || v == 2 || v == 4) return v;
        }
        return 4;
    }();
    return r;
}

int df2_head_chunk() {
    static const int c = [] {
        const char* e = std::getenv("STRATA_DF2_HEAD_CHUNK");
        if (e && e[0]) return std::max(1, std::min(14, std::atoi(e)));
#if defined(STRATA_USE_HIP)
        return 8;
#else
        return (int) (2 * Dflash2Drafter::kDrafts);
#endif
    }();
    return c;
}

bool df2_head_use_il(int head_type, int nc, int n_dvocab, int head_il) {
    if (nc < 2 || nc > 8) return false;
    if (!kernels::native_mmvq_il_supported(head_type, 2, n_dvocab)) return false;   // gate out non-IL quant types (Q8_0, F16, ...)
    return head_il == 1 || (head_il < 0 && kernels::native_mmvq_il_supported(head_type, nc, n_dvocab));
}

template <class Entry>
void destroy_graph_cache(std::vector<Entry>& cache) {
    for (Entry& e : cache) {
        if (e.exec) cudaGraphExecDestroy(e.exec);
    }
    cache.clear();
}

}  // namespace

Dflash2Drafter::~Dflash2Drafter() {
    if (device_ < 0) return;
    prof_report();
    prof_free();
    const OnDevice on_device(device_);
    if (cs_) cudaStreamSynchronize(cs_);
    for (cudaGraphExec_t& e : exec_)
        if (e) cudaGraphExecDestroy(e);
    for (cudaGraphExec_t& e : body_exec_)
        if (e) cudaGraphExecDestroy(e);
    destroy_graph_cache(batch_body_graphs_);
    destroy_graph_cache(batch_full_graphs_);
    auto fr = [](void* p) { if (p) cudaFree(p); };
    if (h_logq_) cudaFreeHost(h_logq_);
    if (h_logd_) cudaFreeHost(h_logd_);
    fr(rb_q_); fr(rb_d_); fr(rb_pos_);
    if (!blob_shared_) fr(blob_);
    fr(landing_); if (dpend_) cudaFreeHost(dpend_); fr(tag_); fr(d_meta_); fr(xq_); fr(xil_); fr(xqp_); fr(xdp_); fr(xqc_); fr(xdc_); fr(rt_ctx_); fr(rt_bctx_); fr(rope_inv_);
    for (int l = 0; l < 8; ++l) { fr(kring_[l]); fr(vring_[l]); }
    fr(fcy_); fr(hctx_); fr(kc_); fr(vc_); fr(h_); fr(xn_); fr(xc_); fr(dyn_); fr(q_); fr(qn_); fr(k_); fr(v_); fr(kblk_); fr(ao_); fr(o_);
    fr(gate_); fr(up_); fr(mlp_); fr(part_); fr(fn_); fr(fn_batch_); fr(hsel_batch_); fr(logits_); fr(topval_); fr(hsel_); fr(toptok_); fr(topscratch_); fr(seltab_);
    if (!dhead_shared_ && !blob_shared_) {
        if (dhead_ != nullptr && head_type_ == 14) kernels::native_q6_k_unpack(dhead_);
        fr(dvocab_);
        fr(dhead_);
    }
    if (h_meta_) cudaFreeHost(h_meta_);
    if (h_out_) cudaFreeHost(h_out_);
    if (h_bmeta_) cudaFreeHost(h_bmeta_);
    if (h_bout_) cudaFreeHost(h_bout_);
    if (cs_) cudaStreamDestroy(cs_);
    if (cs2_) cudaStreamDestroy(cs2_);
    if (ev_fork_) cudaEventDestroy(ev_fork_);
    if (ev_join_) cudaEventDestroy(ev_join_);
    for (cudaEvent_t e : ev_ctx_) if (e) cudaEventDestroy(e);
    if (ev_a_) cudaEventDestroy(ev_a_);
    if (ev_b_) cudaEventDestroy(ev_b_);
}

// STRATA_DF2_PROFILE=1 (measurement only): the round runs without its graph and an event follows every stage; the per-stage averages print at exit
struct Dflash2Drafter::Prof {
    std::vector<cudaEvent_t> ev;
    std::vector<std::string> name;
    std::vector<std::pair<std::string, double>> total;
    int64_t rounds = 0;
};

void Dflash2Drafter::prof_free() {
    delete prof_;
    prof_ = nullptr;
}

// STRATA_DF2_SKIP=attn,head,ctx,mlp,qkvo (measurement only): leaves those parts out of the round, so the graph's GPU time without them can be read
static bool skip_part(const char* what) {
    static const std::string list = [] { const char* e = std::getenv("STRATA_DF2_SKIP"); return std::string(e ? e : ""); }();
    return !list.empty() && list.find(what) != std::string::npos;
}

void Dflash2Drafter::stamp(const char* name) {
    if (!prof_) return;
    cudaEvent_t e;
    cudaEventCreate(&e);
    cudaEventRecord(e, cs_);
    prof_->ev.push_back(e);
    prof_->name.push_back(name);
}

void Dflash2Drafter::prof_collect() {
    if (!prof_ || prof_->ev.size() < 2) return;
    cudaStreamSynchronize(cs_);
    for (size_t i = 1; i < prof_->ev.size(); ++i) {
        float ms = 0;
        cudaEventElapsedTime(&ms, prof_->ev[i - 1], prof_->ev[i]);
        auto it = std::find_if(prof_->total.begin(), prof_->total.end(), [&](const auto& t) { return t.first == prof_->name[i]; });
        if (it == prof_->total.end()) prof_->total.emplace_back(prof_->name[i], ms); else it->second += ms;
    }
    for (cudaEvent_t e : prof_->ev) cudaEventDestroy(e);
    prof_->ev.clear();
    prof_->name.clear();
    ++prof_->rounds;
}

void Dflash2Drafter::prof_report() const {
    if (!prof_ || prof_->rounds == 0) return;
    double sum = 0;
    for (const auto& t : prof_->total) sum += t.second;
    std::fprintf(stderr, "strata dflash2 profile (%lld rounds, ms per round, total %.3f):", (long long) prof_->rounds, sum / (double) prof_->rounds);
    for (const auto& t : prof_->total) std::fprintf(stderr, " %s %.3f", t.first.c_str(), t.second / (double) prof_->rounds);
    std::fputc(10, stderr);
}

const Dflash2Drafter::Tensor& Dflash2Drafter::find(const std::string& name) const {
    for (const Tensor& t : tensors_)
        if (t.name == name) return t;
    throw std::runtime_error("dflash2: tensor " + name + " is missing from the pack");
}

bool Dflash2Drafter::load(const std::string& path, int device, std::string& err) {
    device_ = device;
    const OnDevice on_device(device_);
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { err = "dflash2: cannot open " + path; return false; }
    struct Closer { std::FILE* f; ~Closer() { std::fclose(f); } } closer{f};
    uint32_t hdr[3];
    if (std::fread(hdr, 4, 3, f) != 3 || std::memcmp(hdr, "DF2P", 4) != 0 || hdr[1] != 1) { err = "dflash2: " + path + " is not a DF2P v1 pack"; return false; }
    std::string index(hdr[2], ' ');
    if (std::fread(index.data(), 1, hdr[2], f) != hdr[2]) { err = "dflash2: the pack's index is cut short"; return false; }
    const long data_off = 12 + (long) hdr[2];
    uint64_t total = 0;
    for (size_t i = 0; i < index.size();) {
        const size_t j = index.find('\n', i);
        if (j == std::string::npos) break;
        const std::string line = index.substr(i, j - i);
        i = j + 1;
        char a[256], b[64], c[256];
        long long v1, v2, v3, v4;
        if (std::sscanf(line.c_str(), "cfg %63s %255s", b, c) == 2) {
            const std::string key = b;
            if (key == "hidden") N_ = std::atoi(c);
            else if (key == "layers") L_ = std::atoi(c);
            else if (key == "heads") NH_ = std::atoi(c);
            else if (key == "kv_heads") NKV_ = std::atoi(c);
            else if (key == "head_dim") HD_ = std::atoi(c);
            else if (key == "inter") I_ = std::atoi(c);
            else if (key == "window") window_ = std::atoi(c);
            else if (key == "rope_theta") theta_ = (float) std::atof(c);
            else if (key == "eps") eps_ = (float) std::atof(c);
            else if (key == "conv_group") group_ = std::atoi(c);
            else if (key == "rank") rank_ = std::atoi(c);
            else if (key == "mask_token") mask_token_ = std::atoi(c);
            else if (key == "vocab") vocab_ = std::atoll(c);
            else if (key == "taps") {
                taps_.clear();
                for (const char* q = c; *q;) { taps_.push_back(std::atoi(q)); while (*q && *q != ',') ++q; if (*q == ',') ++q; }
            }
        } else if (std::sscanf(line.c_str(), "t %255s %63s %lld %lld %lld %lld", a, b, &v1, &v2, &v3, &v4) == 6) {
            Tensor t;
            t.name = a; t.dtype = b; t.rows = v1; t.cols = v2; t.off = (uint64_t) v3; t.bytes = (uint64_t) v4;
            tensors_.push_back(t);
            total = std::max(total, t.off + t.bytes);
        }
    }
    if (tensors_.empty() || NH_ != 32 || NKV_ != 8 || HD_ != 128 || rank_ != 256 || N_ != 2560 || L_ > 8 || window_ + 16 > kRing) {
        err = "dflash2: the pack's shape is not the Flash-Next drafter this engine was written for";
        return false;
    }
    if (cudaMalloc((void**) &blob_, total) != cudaSuccess) { err = "dflash2: the pack does not fit in VRAM (" + std::to_string(total >> 20) + " MiB)"; return false; }
    vram_ += total;
    uint64_t max_bytes = 0;
    for (const Tensor& t : tensors_) max_bytes = std::max(max_bytes, t.bytes);
    std::vector<uint8_t> stage(max_bytes);
    uint8_t* tmp = nullptr;
    if (cudaMalloc((void**) &tmp, max_bytes) != cudaSuccess) { err = "dflash2: upload scratch"; return false; }
    struct TmpFree { uint8_t* p; ~TmpFree() { cudaFree(p); } } tmp_free{tmp};
    for (const Tensor& t : tensors_) {
#if defined(_WIN32)
        _fseeki64(f, data_off + (long long) t.off, SEEK_SET);
#else
        fseeko(f, data_off + (off_t) t.off, SEEK_SET);
#endif
        if (std::fread(stage.data(), 1, t.bytes, f) != t.bytes) { err = "dflash2: the pack is shorter than its index"; return false; }
        if (t.dtype == "q8_0") {   // GGUF blocks -> the tiled qs plane at the tensor's offset, then the d plane (same total bytes)
            if (cudaMemcpy(tmp, stage.data(), t.bytes, cudaMemcpyHostToDevice) != cudaSuccess) { err = "dflash2: weight upload failed"; return false; }
            kernels::df2_q8_tile(tmp, blob_ + t.off, blob_ + t.off + (uint64_t) t.rows * (uint64_t) t.cols, (int) t.cols, t.rows * t.cols / 32, nullptr);
            if (cudaDeviceSynchronize() != cudaSuccess) { err = "dflash2: weight conversion failed"; return false; }
        } else if (t.dtype == "q4_0" && t.name.find("_codebook") != std::string::npos) {   // the selector's Q4_0 rows stay in block order (read row by row)
            if (cudaMemcpy(blob_ + t.off, stage.data(), t.bytes, cudaMemcpyHostToDevice) != cudaSuccess) { err = "dflash2: weight upload failed"; return false; }
        } else if (t.dtype == "q4_0") {   // GGML Q4_0 blocks -> the tiled nibble plane at the tensor's offset, then the d plane (same total bytes)
            if (cudaMemcpy(tmp, stage.data(), t.bytes, cudaMemcpyHostToDevice) != cudaSuccess) { err = "dflash2: weight upload failed"; return false; }
            kernels::df2_q4_tile(tmp, blob_ + t.off, blob_ + t.off + (uint64_t) t.rows * (uint64_t) t.cols / 2, (int) t.cols, t.rows * t.cols / 32, nullptr);
            if (cudaDeviceSynchronize() != cudaSuccess) { err = "dflash2: weight conversion failed"; return false; }
            q4_ = true;
        } else if (cudaMemcpy(blob_ + t.off, stage.data(), t.bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "dflash2: weight upload failed";
            return false;
        }
    }
    if (!make_streams(err)) return false;
    return alloc(err);
}

bool Dflash2Drafter::make_streams(std::string& err) {
    // the draft round runs while the verifier's async commit still has work on the card: a high-priority stream keeps it from queueing behind that
    // (CUDA by default; on HIP only with STRATA_DF2_PRIO=1, where the engine's other streams keep the default priority)
    int prio_lo = 0, prio_hi = 0;
#if defined(STRATA_USE_HIP)
    static const bool hi_prio = [] { const char* e = std::getenv("STRATA_DF2_PRIO"); return e && e[0] == '1'; }();
    if (hi_prio) hipDeviceGetStreamPriorityRange(&prio_lo, &prio_hi);
    if (hipStreamCreateWithPriority(&cs_, hipStreamNonBlocking, hi_prio ? prio_hi : 0) != cudaSuccess ||
        hipStreamCreateWithPriority(&cs2_, hipStreamNonBlocking, hi_prio ? prio_hi : 0) != cudaSuccess) {
#else
    static const bool hi_prio = [] { const char* e = std::getenv("STRATA_DF2_PRIO"); return !(e && e[0] == '0'); }();
    cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi);
    if (cudaStreamCreateWithPriority(&cs_, cudaStreamNonBlocking, hi_prio ? prio_hi : 0) != cudaSuccess ||
        cudaStreamCreateWithPriority(&cs2_, cudaStreamNonBlocking, hi_prio ? prio_hi : 0) != cudaSuccess) {
#endif
        err = "dflash2: stream";
        return false;
    }
    for (auto& e : ev_hs_) cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
    if (cudaEventCreateWithFlags(&ev_fork_, cudaEventDisableTiming) != cudaSuccess || cudaEventCreateWithFlags(&ev_join_, cudaEventDisableTiming) != cudaSuccess) {
        err = "dflash2: events";
        return false;
    }
    for (int l = 0; l < 8; ++l)
        if (cudaEventCreateWithFlags(&ev_ctx_[l], cudaEventDisableTiming) != cudaSuccess) { err = "dflash2: events"; return false; }
    if (const char* e = std::getenv("STRATA_DF2_FP16_RING"); e && e[0]) fp16_ring_ = e[0] != '0';
    if (const char* e = std::getenv("STRATA_DF2_PROFILE"); e && e[0] == '1') prof_ = new Prof();
    if (const char* e = std::getenv("STRATA_DF2_TIMING"); e && e[0] == '1') {
        timing_ = true;
        cudaEventCreate(&ev_a_);
        cudaEventCreate(&ev_b_);
    }
    return true;
}

bool Dflash2Drafter::alloc(std::string& err) {
    auto dev = [&](auto** p, size_t n) {
        using T = std::remove_pointer_t<std::remove_reference_t<decltype(*p)>>;
        if (cudaMalloc((void**) p, n * sizeof(T)) != cudaSuccess) return false;
        vram_ += n * sizeof(T);
        return true;
    };
    const size_t B = blob_shared_ ? (size_t) kRows : (size_t) (kBatchMaxSlots * kRows);
    const size_t HMax = blob_shared_ ? (size_t) kDrafts : (size_t) (2 * kDrafts);
    // 16 head columns for the batch path, sized from 8-column calls: the helpers validate ncols against the build's window (8 on default CUDA builds)
    const size_t xq_bytes = kernels::native_q8_1_bytes(N_, kRows) * (blob_shared_ ? 1 : 2);
    const size_t xil_bytes = kernels::native_q8_1_il_bytes(N_, kRows);   // the interleaved layout holds at most 8 columns
    bool ok = dev(&d_meta_, kMetaInts) && dev(&xq_, xq_bytes + 4096) && dev(&xil_, xil_bytes + 4096) && dev(&xqp_, (size_t) 5 * N_ * B + 4096) && dev(&xdp_, (size_t) 5 * N_ / 32 * B + 64) && dev(&xqc_, (size_t) 5 * N_ * B + 4096) && dev(&xdc_, (size_t) 5 * N_ / 32 * B + 64) && dev(&rt_ctx_, 2 * kRows * 128) && (blob_shared_ || dev(&rt_bctx_, 2 * kBatchMaxSlots * kRows * 128)) && dev(&rope_inv_, 64) &&
              dev(&fcy_, B * N_) && dev(&hctx_, B * N_) && dev(&kc_, (size_t) L_ * B * NKV_ * HD_) && dev(&vc_, (size_t) L_ * B * NKV_ * HD_) && dev(&h_, B * N_) && dev(&xn_, B * N_) &&
              dev(&xc_, B * N_) && dev(&dyn_, B * 640) && dev(&q_, B * NH_ * HD_) && dev(&qn_, B * NH_ * HD_) && dev(&k_, B * NKV_ * HD_) && dev(&v_, B * NKV_ * HD_) &&
              dev(&kblk_, B * NKV_ * HD_) && dev(&ao_, B * NH_ * HD_) && dev(&o_, B * N_) && dev(&gate_, B * I_) && dev(&up_, B * I_) && dev(&mlp_, B * N_) &&
              dev(&part_, (B / kRows) * (size_t) kernels::kDf2MaxChunks * 8 * 32 * 130) && dev(&fn_, B * N_) &&
              dev(&topval_, (size_t) (blob_shared_ ? kDrafts : kBatchMaxRows) * 16) &&
              dev(&toptok_, (size_t) (blob_shared_ ? kDrafts : kBatchMaxRows) * 16) &&
              dev(&hsel_, B * rank_) && dev(&topscratch_, HMax * kernels::kDf2TopBlocks * kernels::kDf2TopK * 2) &&
              dev(&seltab_, (size_t) kRows * kernels::kDf2TopK * kernels::kDf2TopK);
    if (!blob_shared_) {
        ok = ok && dev(&fn_batch_, (size_t) kBatchMaxRows * N_) && dev(&hsel_batch_, (size_t) kBatchMaxRows * rank_);
    }
    ok = ok && dev(&landing_, (size_t) kRows * std::max<size_t>(taps_.size(), 1) * (size_t) N_);
    const size_t ring_bytes = (size_t) kRing * NKV_ * HD_ * ring_elem_bytes();
    for (int l = 0; ok && l < L_; ++l) {
        if (cudaMalloc((void**) &kring_[l], ring_bytes) != cudaSuccess ||
            cudaMalloc((void**) &vring_[l], ring_bytes) != cudaSuccess) {
            ok = false;
            break;
        }
        vram_ += 2 * ring_bytes;
    }
    if (!ok) { err = "dflash2: the drafter's buffers do not fit in VRAM"; return false; }
    void* hd = nullptr;
    void* dd = nullptr;
    if (!mapped(kMetaInts * 4, (void**) &hd, &dd)) { err = "dflash2: mapped staging"; return false; }
    h_meta_ = (int32_t*) hd;
    m_meta_ = (int32_t*) dd;
    if (!mapped(64 * 4, &hd, &dd)) { err = "dflash2: mapped output"; return false; }
    h_out_ = (int32_t*) hd;
    out_tok_ = (int32_t*) dd;            // the select kernel writes the 7 picks here (host-visible) ...
    out_prob_ = (float*) dd + 16;        // ... and their probabilities at +16 ints
    if (!blob_shared_) {
        if (!mapped((kBatchMaxSlots * 4) * 4, &hd, &dd)) { err = "dflash2: mapped batch staging"; return false; }
        h_bmeta_ = (int32_t*) hd;
        m_bmeta_ = (int32_t*) dd;
        if (!mapped(128 * 4, &hd, &dd)) { err = "dflash2: mapped batch output"; return false; }
        h_bout_ = (int32_t*) hd;
        out_btok_ = (int32_t*) dd;
        out_bprob_ = (float*) dd + 64;
    }
    rt_blk_ = rt_ctx_ + kRows * 128;
    cudaMemset(rt_ctx_, 0, 2 * kRows * 128 * 4);
    if (rt_bctx_) {
        rt_bblk_ = rt_bctx_ + (kBatchMaxSlots * kRows) * 128;
        cudaMemset(rt_bctx_, 0, 2 * kBatchMaxSlots * kRows * 128 * 4);
    }
    kernels::df2_rope_inv(theta_, rope_inv_, nullptr);
    if (cudaMalloc((void**) &tag_, (size_t) kRing * 8) != cudaSuccess) { err = "dflash2: ring tags"; return false; }
    vram_ += (uint64_t) kRing * 8;
    cudaMemset(tag_, 0xff, (size_t) kRing * 8);   // -1: no slot holds a context row yet
    for (int64_t& t : slot_pos_) t = -1;
    init_ring_log();
    cudaMemset(xqp_, 0, (size_t) 5 * N_ * B + 4096);
    cudaMemset(xdp_, 0, ((size_t) 5 * N_ / 32 * B + 64) * 4);
    cudaMemset(xqc_, 0, (size_t) 5 * N_ * B + 4096);
    cudaMemset(xdc_, 0, ((size_t) 5 * N_ / 32 * B + 64) * 4);
    for (int l = 0; l < L_; ++l) { cudaMemset(kring_[l], 0, ring_bytes); cudaMemset(vring_[l], 0, ring_bytes); }
    return cudaDeviceSynchronize() == cudaSuccess;
}

bool Dflash2Drafter::bind_raw(EmbedFn embed, const void* head_w, int head_type, int64_t head_rows, const std::vector<int32_t>& vocab, std::string& err) {
    const OnDevice on_device(device_);
    embed_ = std::move(embed);
    head_w_ = head_w;
    head_type_ = head_type;
    n_dvocab_ = head_rows;
    if (!vocab.empty()) {
        n_dvocab_ = (int64_t) vocab.size();
        if (dhead_shared_) {   // the MTP drafter's copy of the same vocabulary (checked in bind)
            dvocab_ = const_cast<int32_t*>(shared_.vocab_dev);
        } else {
            if (cudaMalloc((void**) &dvocab_, vocab.size() * 4) != cudaSuccess) { err = "dflash2: draft vocabulary"; return false; }
            vram_ += vocab.size() * 4;
            cudaMemcpy(dvocab_, vocab.data(), vocab.size() * 4, cudaMemcpyHostToDevice);
        }
    }
    if ((n_dvocab_ + kernels::kDf2TopBlocks - 1) / kernels::kDf2TopBlocks > 4096) { err = "dflash2: the head has too many rows for the top-k"; return false; }
    const size_t hmax = blob_shared_ ? (size_t) kDrafts : (size_t) (2 * kDrafts);
    if (!logits_ && cudaMalloc((void**) &logits_, hmax * (size_t) n_dvocab_ * 4) != cudaSuccess) { err = "dflash2: the draft logits do not fit"; return false; }
    vram_ += (uint64_t) hmax * (uint64_t) n_dvocab_ * 4;
    bound_ = true;
    return true;
}

#ifndef STRATA_DF2_NO_BIND   // the split engine (src/split) binds through bind_raw only: bind() copies head rows between cards with the peer API
bool Dflash2Drafter::bind(const WeightTable& wt, const NativeHead* head, const std::string& vocab_file, std::string& err, int head_device) {
    const OnDevice on_device(device_);
    if (head == nullptr || !head->loaded()) { err = "dflash2: the drafter needs the native head (--native)"; return false; }
    const WeightRef* we = wt.find("token_embd.weight");
    if (we == nullptr && native_embed() == nullptr) { err = "dflash2: token_embd.weight is missing"; return false; }   // (a later stage of a layer split may not carry it: the native table serves)
    EmbedFn embed = [we](const int32_t* tok, int n, float* out, cudaStream_t cs) {
        if (const NativeEmbed* ne = native_embed()) {
            ne->gather_dev(tok, n, out, cs);
        } else if (we != nullptr) {
            const auto* codes = (const uint8_t*) we->data;
            const auto* scales = (const float*) (codes + we->codes_bytes);
            const auto* offsets = we->has_offset ? (const float*) (codes + we->codes_bytes + we->scales_bytes) : nullptr;
            kernels::embedding_gather_dev(codes, scales, offsets, tok, n, we->ne0, we->code_bits, we->code_bias, we->group_elems,
                                          (uint64_t) (we->ne0 / (8 / we->code_bits)), (uint64_t) (we->ne0 / we->group_elems), out, cs);
        }
    };
    std::vector<int32_t> vocab;
    // STRATA_DF2_HEAD=native: the drafter runs its head on the target's own resident head (every vocabulary row, no 213 MiB pruned copy, no vocabulary file read)
    // (a head on another card, a layer split's last stage, is not read in place: the pruned copy is gathered there and copied over, as before)
    const bool head_native = Dflash2Drafter::head_native() && (head_device < 0 || head_device == device_);
    if (!vocab_file.empty() && !head_native) {
        if (std::FILE* vf = std::fopen(vocab_file.c_str(), "rb")) {
            std::fseek(vf, 0, SEEK_END);
            const long size = std::ftell(vf);
            std::fseek(vf, 0, SEEK_SET);
            if (size >= 4 && size % 4 == 0) {
                vocab.resize((size_t) size / 4);
                if (std::fread(vocab.data(), 4, vocab.size(), vf) != vocab.size()) vocab.clear();
            }
            std::fclose(vf);
        }
    }
    const void* hw = head->weights();
    const int64_t row_bytes = (int64_t) head->row_bytes();
    int64_t head_rows = row_bytes > 0 ? (int64_t) (head->weight_bytes() / (uint64_t) row_bytes) : 0;
    if (!vocab.empty() && shared_.rows != nullptr && !std::getenv("STRATA_DF2_SHARE_HEAD_OFF") && shared_.type == head->type() &&
        shared_.n == (int64_t) vocab.size() && shared_.vocab_host != nullptr && *shared_.vocab_host == vocab && shared_.vocab_dev != nullptr) {
        // DF-05: the MTP drafter gathered these very rows (same vocabulary ids in the same order, the main head's own layout and type): use them
        dhead_ = const_cast<uint8_t*>(shared_.rows);
        dhead_shared_ = true;
        hw = dhead_;
        head_rows = (int64_t) vocab.size();
    } else if (!vocab.empty() && head_device >= 0 && head_device != device_) {
        // the head is on another card: gather the subset's rows there, copy them over once
        const size_t bytes = (size_t) ((int64_t) vocab.size() * row_bytes);
        uint8_t* tmp = nullptr;
        int32_t* dv = nullptr;
        {
            const OnDevice on_head(head_device);
            if (cudaMalloc((void**) &dv, vocab.size() * 4) != cudaSuccess || cudaMalloc((void**) &tmp, bytes) != cudaSuccess) {
                if (dv) cudaFree(dv);
                err = "dflash2: the draft vocabulary's head rows do not fit on the head's card";
                return false;
            }
            cudaMemcpy(dv, vocab.data(), vocab.size() * 4, cudaMemcpyHostToDevice);
            kernels::gather_rows((const uint8_t*) hw, row_bytes, dv, (int64_t) vocab.size(), tmp, nullptr);
            cudaDeviceSynchronize();
            cudaFree(dv);
        }
        if (cudaMalloc((void**) &dhead_, bytes) != cudaSuccess) {
            { const OnDevice on_head(head_device); cudaFree(tmp); }
            err = "dflash2: the draft vocabulary's head rows do not fit";
            return false;
        }
        const cudaError_t pe = cudaMemcpyPeer(dhead_, device_, tmp, head_device, bytes);
        { const OnDevice on_head(head_device); cudaFree(tmp); }
        if (pe != cudaSuccess) { err = std::string("dflash2: the head rows' copy to the drafter's card: ") + cudaGetErrorString(pe); return false; }
        vram_ += bytes;
        hw = dhead_;
        head_rows = (int64_t) vocab.size();
    } else if (!vocab.empty()) {
        int32_t* dv = nullptr;
        if (cudaMalloc((void**) &dv, vocab.size() * 4) != cudaSuccess || cudaMalloc((void**) &dhead_, (size_t) ((int64_t) vocab.size() * row_bytes)) != cudaSuccess) {
            if (dv) cudaFree(dv);
            err = "dflash2: the draft vocabulary's head rows do not fit";
            return false;
        }
        cudaMemcpy(dv, vocab.data(), vocab.size() * 4, cudaMemcpyHostToDevice);
        kernels::gather_rows((const uint8_t*) hw, row_bytes, dv, (int64_t) vocab.size(), dhead_, nullptr);
        cudaDeviceSynchronize();
        cudaFree(dv);
        vram_ += (uint64_t) ((int64_t) vocab.size() * row_bytes);
        hw = dhead_;
        head_rows = (int64_t) vocab.size();
    }
    if (!bind_raw(embed, hw, head->type(), head_rows, vocab, err)) return false;
    if (dhead_ != nullptr && !dhead_shared_ && !blob_shared_ && head_type_ == 14 && kernels::native_q6_k_packed_enabled()) {
        kernels::native_q6_k_pack(dhead_, N_, (int) n_dvocab_, "DFlash2 draft head");
    }
    std::fprintf(stderr, "strata dflash2: %lld-token draft head (%.1f MiB%s), drafter %.0f MiB of VRAM (%s weights)\n", (long long) n_dvocab_,
                 (double) (n_dvocab_ * row_bytes) / 1048576.0, dhead_shared_ ? ", the MTP drafter's copy, no second one" : "", (double) vram_ / 1048576.0, q4_ ? "Q4_0" : "Q8_0");
    // the pack's selector codebooks as Q4_0 rows (tools/dflash2_pack.py --q4_0-selector; strata_control/drafting.py reads this text from the program to know it can use that pack)
    if (find("candidate_selector.predecessor_codebook").dtype == "q4_0")
        std::fprintf(stderr, "strata dflash2: selector codebooks: Q4_0 rows (about 174 MiB less than bf16)\n");
    return true;
}

#endif  // STRATA_DF2_NO_BIND
void Dflash2Drafter::gemm(std::initializer_list<const char*> names, std::initializer_list<float*> outs, int ncols, cudaStream_t cs, const uint8_t* xq, const float* xd) {
    kernels::Df2Mat mats[16];
    int n = 0, n_in = 0;
    auto o = outs.begin();
    for (const char* name : names) {
        const Tensor& t = find(name);
        mats[n] = kernels::Df2Mat{blob_ + t.off, *o++, (int) t.rows, t.dtype == "q4_0"};
        if (n == 0) n_in = (int) t.cols;
        ++n;
    }
    kernels::df2_gemm(mats, n, xq, xd, n_in, ncols, cs);
}

// The part of a context update after the activations are quantized: per layer the keys and values (projection, k-norm, rope, ring write).  Also what a ring
// rebuild runs over logged rows: `xq` / `xd` are then the log's codes and scales, `pos` the rows' positions (device), rt_ctx_ their rope table.
bool Dflash2Drafter::record_kv(int n, cudaStream_t cs, const uint8_t* xq, const float* xd, const int32_t* pos, std::string& err,
                               const float* rt_override, const int32_t* epoch_override) {
    using namespace kernels;
    try {
        const float* rt = rt_override != nullptr ? rt_override : rt_ctx_;
        const int32_t* ep = epoch_override != nullptr ? epoch_override : (d_meta_ + 18);
        const size_t B = blob_shared_ ? (size_t) kRows : (size_t) (kBatchMaxSlots * kRows);
        const size_t l_stride = B * NKV_ * HD_;
        if (!fork_ && (L_ == 5 || L_ == 8)) {
            if (L_ == 5) {
                gemm({"layers.0.self_attn.k_proj.weight", "layers.0.self_attn.v_proj.weight",
                      "layers.1.self_attn.k_proj.weight", "layers.1.self_attn.v_proj.weight",
                      "layers.2.self_attn.k_proj.weight", "layers.2.self_attn.v_proj.weight",
                      "layers.3.self_attn.k_proj.weight", "layers.3.self_attn.v_proj.weight",
                      "layers.4.self_attn.k_proj.weight", "layers.4.self_attn.v_proj.weight"},
                     {kc_ + 0 * l_stride, vc_ + 0 * l_stride,
                      kc_ + 1 * l_stride, vc_ + 1 * l_stride,
                      kc_ + 2 * l_stride, vc_ + 2 * l_stride,
                      kc_ + 3 * l_stride, vc_ + 3 * l_stride,
                      kc_ + 4 * l_stride, vc_ + 4 * l_stride},
                     n, cs, xq, xd);
            } else {
                gemm({"layers.0.self_attn.k_proj.weight", "layers.0.self_attn.v_proj.weight",
                      "layers.1.self_attn.k_proj.weight", "layers.1.self_attn.v_proj.weight",
                      "layers.2.self_attn.k_proj.weight", "layers.2.self_attn.v_proj.weight",
                      "layers.3.self_attn.k_proj.weight", "layers.3.self_attn.v_proj.weight",
                      "layers.4.self_attn.k_proj.weight", "layers.4.self_attn.v_proj.weight",
                      "layers.5.self_attn.k_proj.weight", "layers.5.self_attn.v_proj.weight",
                      "layers.6.self_attn.k_proj.weight", "layers.6.self_attn.v_proj.weight",
                      "layers.7.self_attn.k_proj.weight", "layers.7.self_attn.v_proj.weight"},
                     {kc_ + 0 * l_stride, vc_ + 0 * l_stride,
                      kc_ + 1 * l_stride, vc_ + 1 * l_stride,
                      kc_ + 2 * l_stride, vc_ + 2 * l_stride,
                      kc_ + 3 * l_stride, vc_ + 3 * l_stride,
                      kc_ + 4 * l_stride, vc_ + 4 * l_stride,
                      kc_ + 5 * l_stride, vc_ + 5 * l_stride,
                      kc_ + 6 * l_stride, vc_ + 6 * l_stride,
                      kc_ + 7 * l_stride, vc_ + 7 * l_stride},
                     n, cs, xq, xd);
            }
            stamp("gemm:kp");
            for (int l = 0; l < L_; ++l) {
                const std::string p = "layers." + std::to_string(l) + ".";
                if (fp16_ring_)
                    df2_kv_ring_f16(kc_ + l * l_stride, vc_ + l * l_stride, f32(p + "self_attn.k_norm.weight"), pos, rt, kring_[l], vring_[l], kRing, n, NKV_, eps_, cs, tag_, ep);
                else
                    df2_kv_ring(kc_ + l * l_stride, vc_ + l * l_stride, f32(p + "self_attn.k_norm.weight"), pos, rt, kring_[l], vring_[l], kRing, n, NKV_, eps_, cs, tag_, ep);
                stamp("df2_kv_ring");
            }
            return true;
        }
        for (int l = 0; l < L_; ++l) {
            const std::string p = "layers." + std::to_string(l) + ".";
            const std::string kp = p + "self_attn.k_proj.weight", vp = p + "self_attn.v_proj.weight";
            gemm({kp.c_str(), vp.c_str()}, {kc_, vc_}, n, cs, xq, xd);
            stamp("gemm:kp");
            if (fp16_ring_)
                df2_kv_ring_f16(kc_, vc_, f32(p + "self_attn.k_norm.weight"), pos, rt, kring_[l], vring_[l], kRing, n, NKV_, eps_, cs, tag_, ep);
            else
                df2_kv_ring(kc_, vc_, f32(p + "self_attn.k_norm.weight"), pos, rt, kring_[l], vring_[l], kRing, n, NKV_, eps_, cs, tag_, ep);
            stamp("df2_kv_ring");
            if (fork_) cudaEventRecord(ev_ctx_[l], cs);   // layer l's keys and values are in the ring
        }
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

bool Dflash2Drafter::record_context(const float* taps, int n, cudaStream_t cs, std::string& err, bool rope, int log_off) {
    using namespace kernels;
    if (skip_part("ctx")) return true;
    try {
        const int tap_n = (int) taps_.size() * N_;
        if (rope) df2_rope_table_inv(d_meta_ + 0, n, rope_inv_, rt_ctx_, cs);
        stamp("df2_rope_table");
        df2_quant_rows(taps, tap_n, n, xqc_, xdc_, cs);
        stamp("df2_quant_rows");
        gemm({"fc.weight"}, {fcy_}, n, cs, xqc_, xdc_);
        stamp("gemm:fc");
        df2_rms_quant(fcy_, f32("hidden_norm.weight"), nullptr, n, N_, eps_, xqc_, xdc_, cs);
        stamp("df2_rms_quant");
        if (!record_kv(n, cs, xqc_, xdc_, d_meta_ + 0, err)) return false;
        if (log_on_) {   // the ring log: queued after K/V projection + ring update so D2H DMA never delays the K/V GEMMs
            cudaMemcpyAsync(h_logq_ + (size_t) log_off * N_, xqc_, (size_t) n * N_, cudaMemcpyDeviceToHost, cs);
            cudaMemcpyAsync(h_logd_ + (size_t) log_off * (N_ / 32), xdc_, (size_t) n * (N_ / 32) * 4, cudaMemcpyDeviceToHost, cs);
        }
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

bool Dflash2Drafter::record_block_body(cudaStream_t cs, std::string& err, bool rope) {
    using namespace kernels;
    try {
        const int B = kRows;
        if (rope) df2_rope_table_inv(d_meta_ + 8, B, rope_inv_, rt_blk_, cs);
        stamp("df2_rope_table");
        embed_(d_meta_ + 24, B, h_, cs);
        stamp("embed_");
        df2_rms_quant(h_, f32("layers.0.input_layernorm.weight"), xn_, B, N_, eps_, xqp_, xdp_, cs);
        stamp("df2_rms_quant");
        for (int l = 0; l < L_; ++l) {
            const std::string p = "layers." + std::to_string(l) + ".";
            const float* base_a = f32(p + "attention_conv.base_kernel");
            const float* base_m = f32(p + "mlp_conv.base_kernel");
            const std::string akp = p + "attention_conv.kernel_projection.weight", mkp = p + "mlp_conv.kernel_projection.weight";
            const std::string qp = p + "self_attn.q_proj.weight", kp = p + "self_attn.k_proj.weight", vp = p + "self_attn.v_proj.weight";
            const std::string gp = p + "mlp.gate_proj.weight", up = p + "mlp.up_proj.weight";
            const std::string op = p + "self_attn.o_proj.weight";
            if (!skip_part("tiny")) gemm({akp.c_str()}, {dyn_}, B, cs, xqp_, xdp_);
            stamp("gemm:akp");
            if (!skip_part("small")) df2_conv_quant(xn_, dyn_, 640, base_a, B, N_, group_, xqp_, xdp_, cs);
            stamp("df2_conv_quant");
            if (!skip_part("qkvo")) gemm({qp.c_str(), kp.c_str(), vp.c_str()}, {q_, k_, v_}, B, cs, xqp_, xdp_);
            stamp("gemm:qp");
            if (!skip_part("small")) df2_norm_rope_qk(q_, k_, f32(p + "self_attn.q_norm.weight"), f32(p + "self_attn.k_norm.weight"), rt_blk_, qn_, kblk_, B, NH_, NKV_, eps_, cs);
            stamp("df2_norm_rope_qk");
            if (fork_) cudaStreamWaitEvent(cs, ev_ctx_[l], 0);   // the context update has written this layer's ring
            if (!skip_part("attn")) {
                if (fp16_ring_)
                    df2_attention_f16(qn_, kring_[l], vring_[l], kblk_, v_, d_meta_ + 17, kRing, window_, part_, xqp_, xdp_, cs, tag_, d_meta_ + 18);
                else
                    df2_attention(qn_, kring_[l], vring_[l], kblk_, v_, d_meta_ + 17, kRing, window_, part_, xqp_, xdp_, cs, tag_, d_meta_ + 18);
            }
            stamp("df2_attention");
            if (!skip_part("qkvo")) gemm({op.c_str()}, {o_}, B, cs, xqp_, xdp_);
            stamp("gemm:o_proj");
            if (!skip_part("small")) df2_conv_rms_quant(o_, dyn_ + 320, 640, base_a + 2 * N_, h_, f32(p + "post_attention_layernorm.weight"), xn_, B, N_, group_, eps_, xqp_, xdp_, cs);
            stamp("df2_conv_rms_quant");
            if (!skip_part("tiny")) gemm({mkp.c_str()}, {dyn_}, B, cs, xqp_, xdp_);
            stamp("gemm:mkp");
            if (!skip_part("small")) df2_conv_quant(xn_, dyn_, 640, base_m, B, N_, group_, xqp_, xdp_, cs);
            stamp("df2_conv_quant");
            if (!skip_part("mlp")) gemm({gp.c_str(), up.c_str()}, {gate_, up_}, B, cs, xqp_, xdp_);
            stamp("gemm:gp");
            if (!skip_part("small")) df2_swiglu_quant(gate_, up_, B, I_, xqp_, xdp_, cs);
            stamp("df2_swiglu_quant");
            if (!skip_part("mlp")) gemm({(p + "mlp.down_proj.weight").c_str()}, {mlp_}, B, cs, xqp_, xdp_);
            stamp("gemm:mlp.down_proj");
            // the next layer's input norm, or after the last layer the final norm (-> fn_, the rows the head and the selector read)
            const std::string next_gamma = l + 1 < L_ ? "layers." + std::to_string(l + 1) + ".input_layernorm.weight" : std::string("norm.weight");
            if (!skip_part("small")) df2_conv_rms_quant(mlp_, dyn_ + 320, 640, base_m + 2 * N_, h_, f32(next_gamma), l + 1 < L_ ? xn_ : fn_, B, N_, group_, eps_, xqp_, xdp_, cs);
            stamp("df2_conv_rms_quant");
        }
        if (skip_part("small")) df2_quant_rows(fn_ + N_, N_, ndraft_, xqp_ + N_, xdp_ + N_ / 32, cs);
        gemm({"candidate_selector.hidden_projection.weight"}, {hsel_}, ndraft_, cs, xqp_ + N_, xdp_ + N_ / 32);
        stamp("gemm:candidate_selector.hidden_projection");
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

bool Dflash2Drafter::record_block(cudaStream_t cs, std::string& err, bool rope) {
    using namespace kernels;
    try {
        const int B = kRows;
        if (rope) df2_rope_table_inv(d_meta_ + 8, B, rope_inv_, rt_blk_, cs);
        stamp("df2_rope_table");
        embed_(d_meta_ + 24, B, h_, cs);
        stamp("embed_");
        df2_rms_quant(h_, f32("layers.0.input_layernorm.weight"), xn_, B, N_, eps_, xqp_, xdp_, cs);
        stamp("df2_rms_quant");
        const int head_il = df2_head_il_mode();
        const bool use_il = !skip_part("head") && df2_head_use_il(head_type_, ndraft_, (int) n_dvocab_, head_il);
        const bool want_final_q8 = !skip_part("head") && ndraft_ >= 1 && ndraft_ <= 8;
        const bool want_final_il = want_final_q8 && use_il && ndraft_ >= 2;
        bool did_fuse_q8 = false;
        for (int l = 0; l < L_; ++l) {
            const std::string p = "layers." + std::to_string(l) + ".";
            const float* base_a = f32(p + "attention_conv.base_kernel");
            const float* base_m = f32(p + "mlp_conv.base_kernel");
            const std::string akp = p + "attention_conv.kernel_projection.weight", mkp = p + "mlp_conv.kernel_projection.weight";
            const std::string qp = p + "self_attn.q_proj.weight", kp = p + "self_attn.k_proj.weight", vp = p + "self_attn.v_proj.weight";
            const std::string gp = p + "mlp.gate_proj.weight", up = p + "mlp.up_proj.weight";
            const std::string op = p + "self_attn.o_proj.weight";
            if (!skip_part("tiny")) gemm({akp.c_str()}, {dyn_}, B, cs, xqp_, xdp_);
            stamp("gemm:akp");
            if (!skip_part("small")) df2_conv_quant(xn_, dyn_, 640, base_a, B, N_, group_, xqp_, xdp_, cs);
            stamp("df2_conv_quant");
            if (!skip_part("qkvo")) gemm({qp.c_str(), kp.c_str(), vp.c_str()}, {q_, k_, v_}, B, cs, xqp_, xdp_);
            stamp("gemm:qp");
            if (!skip_part("small")) df2_norm_rope_qk(q_, k_, f32(p + "self_attn.q_norm.weight"), f32(p + "self_attn.k_norm.weight"), rt_blk_, qn_, kblk_, B, NH_, NKV_, eps_, cs);
            stamp("df2_norm_rope_qk");
            if (fork_) cudaStreamWaitEvent(cs, ev_ctx_[l], 0);   // the context update has written this layer's ring
            if (!skip_part("attn")) {
                if (fp16_ring_)
                    df2_attention_f16(qn_, kring_[l], vring_[l], kblk_, v_, d_meta_ + 17, kRing, window_, part_, xqp_, xdp_, cs, tag_, d_meta_ + 18);
                else
                    df2_attention(qn_, kring_[l], vring_[l], kblk_, v_, d_meta_ + 17, kRing, window_, part_, xqp_, xdp_, cs, tag_, d_meta_ + 18);
            }
            stamp("df2_attention");
            if (!skip_part("qkvo")) gemm({op.c_str()}, {o_}, B, cs, xqp_, xdp_);
            stamp("gemm:o_proj");
            if (!skip_part("small")) df2_conv_rms_quant(o_, dyn_ + 320, 640, base_a + 2 * N_, h_, f32(p + "post_attention_layernorm.weight"), xn_, B, N_, group_, eps_, xqp_, xdp_, cs);
            stamp("df2_conv_rms_quant");
            if (!skip_part("tiny")) gemm({mkp.c_str()}, {dyn_}, B, cs, xqp_, xdp_);
            stamp("gemm:mkp");
            if (!skip_part("small")) df2_conv_quant(xn_, dyn_, 640, base_m, B, N_, group_, xqp_, xdp_, cs);
            stamp("df2_conv_quant");
            if (!skip_part("mlp")) gemm({gp.c_str(), up.c_str()}, {gate_, up_}, B, cs, xqp_, xdp_);
            stamp("gemm:gp");
            if (!skip_part("small")) df2_swiglu_quant(gate_, up_, B, I_, xqp_, xdp_, cs);
            stamp("df2_swiglu_quant");
            if (!skip_part("mlp")) gemm({(p + "mlp.down_proj.weight").c_str()}, {mlp_}, B, cs, xqp_, xdp_);
            stamp("gemm:mlp.down_proj");
            // the next layer's input norm, or after the last layer the final norm (-> fn_, the rows the head and the selector read)
            const std::string next_gamma = l + 1 < L_ ? "layers." + std::to_string(l + 1) + ".input_layernorm.weight" : std::string("norm.weight");
            if (!skip_part("small")) {
                const bool last = l + 1 == L_;
                df2_conv_rms_quant(mlp_, dyn_ + 320, 640, base_m + 2 * N_, h_, f32(next_gamma), last ? fn_ : xn_,
                                   B, N_, group_, eps_, xqp_, xdp_, cs,
                                   (last && want_final_q8) ? xq_ : nullptr,
                                   (last && want_final_il) ? xil_ : nullptr, 1, ndraft_);
                if (last && want_final_q8) did_fuse_q8 = true;
            }
            stamp("df2_conv_rms_quant");
        }
        if (!did_fuse_q8) {
            if (use_il && ndraft_ >= 2 && ndraft_ <= 8)
                native_quantize_q8_1_il(fn_ + N_, xq_, xil_, N_, ndraft_, cs);
            else {
                native_quantize_q8_1(fn_ + N_, xq_, N_, ndraft_, cs);   // rows 1..7, the head's format
                if (use_il && ndraft_ > 8)
                    native_q8_1_interleave(xq_, xil_, N_, ndraft_, cs);
            }
        }
        // df2_conv_rms_quant on the final layer has already quantized fn_ rows 1..7 into xqp_ + N_ and xdp_ + N_ / 32
        if (skip_part("small")) df2_quant_rows(fn_ + N_, N_, ndraft_, xqp_ + N_, xdp_ + N_ / 32, cs);
        const bool hs_fork = fork_ && cs2_ != nullptr && ev_hs_[0] != nullptr;   // the selector's projection beside the head, the top-k and their launches
        if (hs_fork) {
            cudaEventRecord(ev_hs_[0], cs);
            cudaStreamWaitEvent(cs2_, ev_hs_[0], 0);
            gemm({"candidate_selector.hidden_projection.weight"}, {hsel_}, ndraft_, cs2_, xqp_ + N_, xdp_ + N_ / 32);
            cudaEventRecord(ev_hs_[1], cs2_);
        }
        if (!skip_part("head")) {
            // the verify window's head kernel (interleaved activations, rows a warp by the card's table): bitwise native_mmvq's output, so the drafts are the same
            if (use_il) {
                const bool force_rows = (std::getenv("STRATA_DF2_HEAD_IL_ROWS") != nullptr) || !native_mmvq_il_supported(head_type_, ndraft_, (int) n_dvocab_);
                if (force_rows) native_mmvq_il_tune(df2_head_il_rows());
                native_mmvq_il(head_type_, head_w_, xq_, xil_, logits_, N_, (int) n_dvocab_, ndraft_, cs);
                if (force_rows) native_mmvq_il_tune(0);
            } else
                native_mmvq(head_type_, head_w_, xq_, logits_, N_, (int) n_dvocab_, ndraft_, cs);
        }
        stamp("native_mmvq");
        df2_topk(logits_, (int) n_dvocab_, dvocab_, ndraft_, topscratch_, topval_, toptok_, cs);
        stamp("df2_topk");
        if (hs_fork) cudaStreamWaitEvent(cs, ev_hs_[1], 0);
        else gemm({"candidate_selector.hidden_projection.weight"}, {hsel_}, ndraft_, cs, xqp_ + N_, xdp_ + N_ / 32);
        stamp("gemm:candidate_selector.hidden_projection");
        df2_select(hsel_, topval_, toptok_, reinterpret_cast<const uint16_t*>(blob_ + find("candidate_selector.predecessor_codebook").off),
                   reinterpret_cast<const uint16_t*>(blob_ + find("candidate_selector.successor_codebook").off), rank_, ndraft_, d_meta_ + 16, out_tok_,
                   out_prob_, cs, find("candidate_selector.predecessor_codebook").dtype == "q4_0", df2_beam(), df2_select_table() ? seltab_ : nullptr);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

bool Dflash2Drafter::capture(int n, const float* taps, std::string& err) {
    if (exec_[n]) return true;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "dflash2: begin capture"; return false; }
    kernels::copy_i32_from_mapped(d_meta_, m_meta_, 32, cs_);
    kernels::df2_rope_table_inv(d_meta_, 16, rope_inv_, rt_ctx_, cs_);   // the context rows' and the block rows' tables in one launch, before the branches split
    // the context update (fc, the keys and values) and the block's front run side by side: they meet before each layer's attention
    // measured: -0.3 ms a round on an RTX 3060, +0.0..0.7 on a Strix Halo (gfx1151), so the HIP builds keep one stream (STRATA_DF2_FORK=1 / 0 overrides)
#if defined(STRATA_USE_HIP)
    static const bool fork_default = false;
#else
    static const bool fork_default = true;
#endif
    static const bool fork_on = [] { const char* e = std::getenv("STRATA_DF2_FORK"); return e && e[0] ? e[0] == '1' : fork_default; }();
    fork_ = fork_on && prof_ == nullptr && cs2_ != nullptr;
    if (fork_) {
        cudaEventRecord(ev_fork_, cs_);
        cudaStreamWaitEvent(cs2_, ev_fork_, 0);
    }
    bool ok = record_context(taps, n, fork_ ? cs2_ : cs_, err, false);
    if (fork_) cudaEventRecord(ev_join_, cs2_);
    if (ok) ok = record_block(cs_, err, false);
    if (fork_) cudaStreamWaitEvent(cs_, ev_join_, 0);
    fork_ = false;
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) { if (graph) cudaGraphDestroy(graph); return false; }
    if (ce != cudaSuccess || cudaGraphInstantiate(&exec_[n], graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("dflash2: capture: ") + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    cudaGraphUpload(exec_[n], cs_);
    cudaStreamSynchronize(cs_);
    return true;
}

bool Dflash2Drafter::capture_body(int n, const float* taps, std::string& err) {
    if (body_exec_[n]) return true;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "dflash2: begin capture_body"; return false; }
    kernels::copy_i32_from_mapped(d_meta_, m_meta_, 32, cs_);
    kernels::df2_rope_table_inv(d_meta_, 16, rope_inv_, rt_ctx_, cs_);
#if defined(STRATA_USE_HIP)
    static const bool fork_default = false;
#else
    static const bool fork_default = true;
#endif
    static const bool fork_on = [] { const char* e = std::getenv("STRATA_DF2_FORK"); return e && e[0] ? e[0] == '1' : fork_default; }();
    fork_ = fork_on && prof_ == nullptr && cs2_ != nullptr;
    if (fork_) {
        cudaEventRecord(ev_fork_, cs_);
        cudaStreamWaitEvent(cs2_, ev_fork_, 0);
    }
    bool ok = record_context(taps, n, fork_ ? cs2_ : cs_, err, false);
    if (fork_) cudaEventRecord(ev_join_, cs2_);
    if (ok) ok = record_block_body(cs_, err, false);
    if (fork_) cudaStreamWaitEvent(cs_, ev_join_, 0);
    fork_ = false;
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) { if (graph) cudaGraphDestroy(graph); return false; }
    if (ce != cudaSuccess || cudaGraphInstantiate(&body_exec_[n], graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("dflash2: capture_body: ") + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    cudaGraphUpload(body_exec_[n], cs_);
    cudaStreamSynchronize(cs_);
    return true;
}

bool Dflash2Drafter::clone_alloc(const Dflash2Drafter& parent, std::string& err) {
    if (parent.blob_ == nullptr) { err = "dflash2: the parent drafter has no pack loaded"; return false; }
    device_ = parent.device_;
    const OnDevice on_device(device_);
    N_ = parent.N_; L_ = parent.L_; NH_ = parent.NH_; NKV_ = parent.NKV_; HD_ = parent.HD_; I_ = parent.I_; window_ = parent.window_;
    group_ = parent.group_; rank_ = parent.rank_; mask_token_ = parent.mask_token_; theta_ = parent.theta_; eps_ = parent.eps_;
    taps_ = parent.taps_; vocab_ = parent.vocab_; q4_ = parent.q4_; tensors_ = parent.tensors_;
    blob_ = parent.blob_; blob_shared_ = true;
    return make_streams(err) && alloc(err);
}

bool Dflash2Drafter::clone_bind(const Dflash2Drafter& parent, std::string& err) {
    if (!parent.bound_ || !blob_shared_) { err = "dflash2: the parent drafter is not bound"; return false; }
    const OnDevice on_device(device_);
    embed_ = parent.embed_; head_w_ = parent.head_w_; head_type_ = parent.head_type_; n_dvocab_ = parent.n_dvocab_;
    dvocab_ = parent.dvocab_; dhead_ = parent.dhead_; dhead_shared_ = true;
    ndraft_ = parent.ndraft_;
    if (cudaMalloc((void**) &logits_, (size_t) kDrafts * (size_t) n_dvocab_ * 4) != cudaSuccess) { err = "dflash2: the slot drafter's draft logits do not fit"; return false; }
    vram_ += (uint64_t) kDrafts * (uint64_t) n_dvocab_ * 4;
    bound_ = true;
    return true;
}

bool Dflash2Drafter::copy_state_from(Dflash2Drafter& src, std::string& err, bool already_synced, bool sync_after) {
    if (src.device_ != device_ || src.L_ != L_ || src.NKV_ != NKV_ || src.fp16_ring_ != fp16_ring_) { err = "dflash2: the two drafters are not alike"; return false; }
    if (!src.context_wait(err) || !context_wait(err)) return false;
    const OnDevice on_device(device_);
    if (!already_synced) {
        if (cudaStreamSynchronize(src.cs_) != cudaSuccess || cudaStreamSynchronize(cs_) != cudaSuccess) { err = "dflash2: sync before a state copy"; return false; }
    }
    const int n_slots = dflash2_ring_copy_slots(src.ring_hi_, kRing);
    if (n_slots >= kRing) {
        const size_t ring_bytes = (size_t) kRing * NKV_ * HD_ * ring_elem_bytes();
        for (int l = 0; l < L_; ++l)
            if (cudaMemcpyAsync(kring_[l], src.kring_[l], ring_bytes, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
                cudaMemcpyAsync(vring_[l], src.vring_[l], ring_bytes, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) { err = "dflash2: ring copy"; return false; }
    } else if (n_slots > 0) {
        const size_t pitch = (size_t) kRing * HD_ * ring_elem_bytes();
        const size_t width = (size_t) n_slots * HD_ * ring_elem_bytes();
        for (int l = 0; l < L_; ++l)
            if (cudaMemcpy2DAsync(kring_[l], pitch, src.kring_[l], pitch, width, (size_t) NKV_, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
                cudaMemcpy2DAsync(vring_[l], pitch, src.vring_[l], pitch, width, (size_t) NKV_, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) { err = "dflash2: ring copy"; return false; }
    }
    if (cudaMemcpyAsync(tag_, src.tag_, (size_t) kRing * 8, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) { err = "dflash2: tag copy"; return false; }
    if (sync_after) {
        if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = std::string("dflash2: state copy: ") + cudaGetErrorString(cudaGetLastError()); return false; }
    }
    copy_log_state_from(src);   // history, the log (shared copy-on-write), the tag shadow, the epoch, ring_hi_ and have_ctx_
    return true;
}

float* Dflash2Drafter::defer_rows(int64_t pos0, int n, std::string& err) {
    if (dpend_ == nullptr) {
        const OnDevice on_device(device_);
        if (cudaHostAlloc((void**) &dpend_, (size_t) kRing2048 * taps_.size() * (size_t) N_ * 4, cudaHostAllocPortable) != cudaSuccess) {
            (void) cudaGetLastError();
            dpend_ = nullptr;
            err = "dflash2: the deferred-rows buffer does not fit in pinned memory";
            return nullptr;
        }
    }
    if (!has_deferred() || pos0 != dpend_hi_) { dpend_lo_ = pos0; dpend_hi_ = pos0; }
    dpend_hi_ = pos0 + n;
    if (dpend_hi_ - dpend_lo_ > kRing2048) dpend_lo_ = dpend_hi_ - kRing2048;
    n_deferred_rows += n;
    return dpend_ + (size_t) (pos0 % kRing2048) * (size_t) (taps_.size() * (size_t) N_);
}

bool Dflash2Drafter::catch_up(int64_t upto, std::string& err) {
    if (!has_deferred()) return true;
    const Clock::time_point t0 = Clock::now();
    const OnDevice on_device(device_);
    const size_t row = taps_.size() * (size_t) N_;
    int64_t from = std::max<int64_t>(dpend_lo_, upto - 2047);
    upto = std::min<int64_t>(upto, dpend_hi_);
    int64_t replayed = 0;
    while (from < upto) {
        const int m = (int) std::min<int64_t>(kRows, upto - from);
        if (!context_wait(err)) return false;
        for (int i = 0; i < m; ++i)   // (the held run wraps at 2048 positions: row by row)
            if (cudaMemcpyAsync(landing_ + (size_t) i * row, dpend_ + (size_t) ((from + i) % kRing2048) * row, row * 4, cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
                err = "dflash2: the deferred rows' upload";
                return false;
            }
        if (!context(landing_, m, from, err)) return false;   // (waits for the stream, so the next chunk may reuse the landing)
        from += m;
        replayed += m;
    }
    dpend_lo_ = std::max<int64_t>(upto, dpend_lo_);
    if (dpend_lo_ >= dpend_hi_) dpend_lo_ = dpend_hi_ = 0;
    ++n_catchups;
    n_catchup_rows += replayed;
    ms_catchup += ms_since(t0);
    return true;
}

void Dflash2Drafter::set_max_drafts(int n) {
    n = std::max(1, std::min(n, (int) kDrafts));
    if (n == ndraft_) return;
    const OnDevice on_device(device_);
    cudaDeviceSynchronize();
    for (auto& e : exec_)
        if (e) { cudaGraphExecDestroy(e); e = nullptr; }
    for (auto& e : body_exec_)
        if (e) { cudaGraphExecDestroy(e); e = nullptr; }
    destroy_graph_cache(batch_body_graphs_);
    destroy_graph_cache(batch_full_graphs_);
    clear_precomputed_context();
    graph_taps_ = nullptr;
    ndraft_ = n;
}

bool Dflash2Drafter::prepare(const float* taps, std::string& err) {
    if (!bound_) { err = "dflash2: not bound"; return false; }
    if (prof_ != nullptr) return true;   // the profile runs without graphs
    static const bool batch_body_on = [] {
        const char* e = std::getenv("STRATA_DF2_BATCHBODY");
        return !e || !e[0] || e[0] != '0';
    }();
    if (!dflash2_clone_eager_single_graphs(blob_shared_, batch_body_on)) return true;
    const OnDevice on_device(device_);
    const Clock::time_point t0 = Clock::now();
    for (int i = 0; i < 8; ++i) h_meta_[i] = i;
    for (int n = 1; n <= kRows; ++n) {
        if (!capture(n, taps, err)) return false;
        if (blob_shared_ && !capture_body(n, taps, err)) return false;
    }
    graph_taps_ = taps;
    ms_capture = ms_since(t0);
    return true;
}

bool Dflash2Drafter::prewarm_batch_slots(Dflash2Drafter* const* slots, int n_slots, std::string& err) {
    if (!bound_ || blob_shared_) { err = "dflash2: prewarm_batch_slots requires bound parent drafter"; return false; }
    if (prof_ != nullptr || slots == nullptr || n_slots < 1 || rt_bctx_ == nullptr) return true;
    static const bool batch_body_on = [] {
        const char* e = std::getenv("STRATA_DF2_BATCHBODY");
        return !e || !e[0] || e[0] != '0';
    }();
    if (!batch_body_on) return true;
    const OnDevice on_device(device_);
    const Clock::time_point t0 = Clock::now();
    std::memset(h_meta_, 0, kMetaInts * sizeof(int32_t));
    for (int i = 0; i < (int) kBatchMaxSlots * 4; ++i) h_bmeta_[i] = 0;
    const int n1 = std::min(n_slots, (int) kBatchMaxSlots);
    for (int b = 0; b < n1; ++b) {
        if (slots[b] == nullptr || !slots[b]->bound_ || slots[b]->device_ != device_) continue;
        const int kw = std::max(1, std::min((int) kDrafts, slots[b]->max_drafts()));
        BatchSlotItem it{};
        it.slot = slots[b];
        it.n_new = 1;
        it.k_want = kw;
        h_bmeta_[0] = 0;
        h_bmeta_[1] = kw;
        cudaGraphExec_t exec = nullptr;
        if (!capture_batch_full(&it, 1, kw, exec, err)) return false;
    }
    ms_capture += ms_since(t0);
    return true;
}

// ---- ring log -----------------------------------------------------------------------------------------------------------------------------------------
// The ring holds the last 2064 positions of the lineage it was filled for.  Resuming a prompt below the highest position written (a repeated or rewound
// prefix, a parked conversation brought back) leaves the part of the first draft's window [n - 2047, resume) either overwritten by the later positions
// (the slots are position % 2064) or never written for this lineage.  The drafts are then poor until the window refills, and the engine said nothing for the
// first kind.  The K/V rows are a function of the quantized hidden rows (codes and block scales after hidden_norm) alone, so those rows are logged as the
// updates make them (2.9 KB a position; the K/V is 40 KB and the taps 51 KB), per lineage, and a rebuild runs record_kv over them: the same kernels on the
// same bytes, so the rebuilt K/V rows are bitwise those of the original update.
bool Dflash2Drafter::Lineage::logged(int64_t pos) const {
    const size_t c = (size_t) (pos / kLogChunk);
    return pos >= 0 && c < chunks.size() && chunks[c] && chunks[c]->ok[(size_t) (pos % kLogChunk)] != 0;
}

uint64_t Dflash2Drafter::Lineage::bytes() const {
    uint64_t b = (uint64_t) hist.capacity() * 4;
    for (const auto& c : chunks)
        if (c) b += c->q.size() + c->d.size() * 4 + c->ok.size();
    return b;
}

Dflash2Drafter::LogChunk& Dflash2Drafter::Lineage::writable(size_t chunk) {
    if (chunks.size() <= chunk) chunks.resize(chunk + 1);
    if (!chunks[chunk]) {
        auto c = std::make_shared<LogChunk>();
        c->q.assign((size_t) kLogChunk * 2560, 0);
        c->d.assign((size_t) kLogChunk * 80, 0.0f);
        c->ok.assign((size_t) kLogChunk, 0);
        chunks[chunk] = std::move(c);
    } else if (chunks[chunk].use_count() > 1) {
        chunks[chunk] = std::make_shared<LogChunk>(*chunks[chunk]);   // copy on write: a stashed copy keeps the old rows
    }
    return *chunks[chunk];
}

void Dflash2Drafter::Lineage::truncate(int64_t keep) {
    keep = std::max<int64_t>(0, keep);
    const size_t c0 = (size_t) (keep / kLogChunk);
    for (size_t c = c0 + 1; c < chunks.size(); ++c) chunks[c].reset();
    if (chunks.size() > c0 + 1) chunks.resize(c0 + 1);
    if (c0 < chunks.size() && chunks[c0]) {
        if (keep % kLogChunk == 0) {
            chunks[c0].reset();
        } else {
            LogChunk& ch = writable(c0);
            for (int i = (int) (keep % kLogChunk); i < kLogChunk; ++i) ch.ok[(size_t) i] = 0;
        }
    }
}

uint64_t Dflash2Drafter::ring_log_bytes() const { return cur_.bytes(); }
uint64_t Dflash2Drafter::ring_stash_bytes() const {
    uint64_t b = 0;
    for (const Lineage& l : stash_) b += l.bytes();
    return b;
}

// The log is OFF until this has allocated its pinned staging: a drafter that never ran alloc() (a --parallel slot's clone) keeps the old behaviour instead of
// copying into a null staging buffer. Callable again by a clone's allocation, on the drafter's device.
void Dflash2Drafter::init_ring_log() {
    const OnDevice on_device(device_);
    for (int64_t& t : slot_pos_) t = -1;
    const char* off = std::getenv("STRATA_DF2_RING_LOG");
    if (off && off[0] == '0') { log_on_ = false; return; }
    if (const char* e = std::getenv("STRATA_DF2_LOG_MIB"); e && e[0]) stash_budget_ = (uint64_t) std::max(0, std::atoi(e)) << 20;
    const size_t log_rows = (size_t) kRows * 8;
    if (h_logq_ == nullptr && cudaHostAlloc((void**) &h_logq_, log_rows * N_, cudaHostAllocDefault) != cudaSuccess) h_logq_ = nullptr;
    if (h_logd_ == nullptr && cudaHostAlloc((void**) &h_logd_, log_rows * (N_ / 32) * 4, cudaHostAllocDefault) != cudaSuccess) h_logd_ = nullptr;
    log_on_ = h_logq_ != nullptr && h_logd_ != nullptr;
    if (!log_on_) {
        (void) cudaGetLastError();
        std::fprintf(stderr, "strata dflash2: the ring log's staging does not fit in pinned memory: the log is off\n");
    }
}

void Dflash2Drafter::reset() {
    have_ctx_ = 0;
    cur_ = Lineage{};
    stash_.clear();
    invalidate_ring();
}

void Dflash2Drafter::copy_log_state_from(const Dflash2Drafter& src) {
    if (this != &src && !cur_.hist.empty()) stash_push(std::move(cur_));
    cur_.hist = src.cur_.hist;
    cur_.chunks = src.cur_.chunks;
    std::memcpy(slot_pos_, src.slot_pos_, sizeof(slot_pos_));
    epoch_ = src.epoch_;
    ring_hi_ = src.ring_hi_;
    have_ctx_ = src.have_ctx_;
}

size_t Dflash2Drafter::lineage_bytes(int64_t upto) const {
    if (!log_on_ || N_ != 2560) return 0;
    const int64_t L = upto < 0 ? (int64_t) cur_.hist.size() : std::min<int64_t>(upto, (int64_t) cur_.hist.size());
    if (L <= 0) return 0;
    size_t n_runs = 0, n_rows = 0;
    int64_t p = 0;
    while (p < L) {
        if (!cur_.logged(p)) { ++p; continue; }
        int64_t q = p + 1;
        while (q < L && cur_.logged(q)) ++q;
        ++n_runs;
        n_rows += (size_t) (q - p);
        p = q;
    }
    if (n_rows == 0) return 0;
    return 32 + (size_t) L * sizeof(int32_t) + n_runs * 8 + n_rows * (2560 + 80 * sizeof(float));
}

bool Dflash2Drafter::save_lineage(std::vector<uint8_t>& out, std::string& err, int64_t upto) const {
    const size_t total = lineage_bytes(upto);
    if (total == 0) { out.clear(); return true; }
    const int64_t L = upto < 0 ? (int64_t) cur_.hist.size() : std::min<int64_t>(upto, (int64_t) cur_.hist.size());
    struct Run { int32_t pos0, count; };
    std::vector<Run> runs;
    int64_t p = 0;
    while (p < L) {
        if (!cur_.logged(p)) { ++p; continue; }
        int64_t q = p + 1;
        while (q < L && cur_.logged(q)) ++q;
        runs.push_back({(int32_t) p, (int32_t) (q - p)});
        p = q;
    }
    try { out.resize(total); } catch (const std::bad_alloc&) { err = "dflash2: out of host memory saving lineage"; return false; }
    uint8_t* dst = out.data();
    auto put = [&](const void* s, size_t n) { std::memcpy(dst, s, n); dst += n; };
    const uint32_t magic = kLineageMagic, ver = kLineageVersion;
    const int32_t n_embd = N_, win = window_;
    const uint64_t hist_len = (uint64_t) L, n_runs = (uint64_t) runs.size();
    put(&magic, 4); put(&ver, 4); put(&n_embd, 4); put(&win, 4);
    put(&hist_len, 8); put(&n_runs, 8);
    put(cur_.hist.data(), (size_t) L * sizeof(int32_t));
    for (const Run& r : runs) {
        put(&r.pos0, 4); put(&r.count, 4);
        for (int32_t i = 0; i < r.count; ++i) {
            const int64_t pos = (int64_t) r.pos0 + i;
            const LogChunk& ch = *cur_.chunks[(size_t) (pos / kLogChunk)];
            put(&ch.q[(size_t) (pos % kLogChunk) * 2560], 2560);
        }
        for (int32_t i = 0; i < r.count; ++i) {
            const int64_t pos = (int64_t) r.pos0 + i;
            const LogChunk& ch = *cur_.chunks[(size_t) (pos / kLogChunk)];
            put(&ch.d[(size_t) (pos % kLogChunk) * 80], 80 * sizeof(float));
        }
    }
    return true;
}

bool Dflash2Drafter::restore_lineage(const std::vector<uint8_t>& in, std::string& err) {
    if (in.empty()) return true;
    if (in.size() < 32) { err = "dflash2: truncated lineage header"; return false; }
    const uint8_t* src = in.data();
    const uint8_t* end = src + in.size();
    auto get = [&](void* d, size_t n) -> bool {
        if ((size_t) (end - src) < n) return false;
        std::memcpy(d, src, n); src += n; return true;
    };
    uint32_t magic = 0, ver = 0;
    int32_t n_embd = 0, win = 0;
    uint64_t hist_len = 0, n_runs = 0;
    get(&magic, 4); get(&ver, 4); get(&n_embd, 4); get(&win, 4);
    get(&hist_len, 8); get(&n_runs, 8);
    if (magic != kLineageMagic || ver != kLineageVersion) { err = "dflash2: incompatible lineage header"; return false; }
    if (n_embd != 2560 || (N_ != 0 && n_embd != N_)) { err = "dflash2: lineage embedding dimension mismatch"; return false; }
    if (hist_len == 0 || hist_len > (1ull << 24) || n_runs > hist_len) { err = "dflash2: invalid lineage bounds"; return false; }
    Lineage l;
    try {
        l.hist.resize((size_t) hist_len);
        if (!get(l.hist.data(), (size_t) hist_len * sizeof(int32_t))) { err = "dflash2: truncated lineage token history"; return false; }
        int64_t prev_end = 0;
        for (uint64_t r = 0; r < n_runs; ++r) {
            int32_t pos0 = 0, count = 0;
            if (!get(&pos0, 4) || !get(&count, 4)) { err = "dflash2: truncated lineage run header"; return false; }
            if (pos0 < prev_end || count <= 0 || (uint64_t) pos0 + (uint64_t) count > hist_len) {
                err = "dflash2: out-of-range lineage run"; return false;
            }
            prev_end = (int64_t) pos0 + count;
            const size_t q_bytes = (size_t) count * 2560;
            const size_t d_bytes = (size_t) count * 80 * sizeof(float);
            if ((size_t) (end - src) < q_bytes + d_bytes) { err = "dflash2: truncated lineage run payload"; return false; }
            const uint8_t* q_src = src;
            const uint8_t* d_src = src + q_bytes;
            src += q_bytes + d_bytes;
            for (int32_t i = 0; i < count; ++i) {
                const int64_t pos = (int64_t) pos0 + i;
                LogChunk& ch = l.writable((size_t) (pos / kLogChunk));
                const size_t row = (size_t) (pos % kLogChunk);
                std::memcpy(&ch.q[row * 2560], q_src + (size_t) i * 2560, 2560);
                std::memcpy(&ch.d[row * 80], d_src + (size_t) i * 80 * sizeof(float), 80 * sizeof(float));
                ch.ok[row] = 1;
            }
        }
    } catch (const std::bad_alloc&) {
        err = "dflash2: out of host memory restoring lineage";
        return false;
    }
    if (src != end) { err = "dflash2: trailing bytes in lineage blob"; return false; }
    if (!log_on_) return true;
    if (!cur_.hist.empty()) {
        stash_push(std::move(cur_));
        cur_ = Lineage{};
        invalidate_ring();
    }
    for (size_t i = stash_.size(); i-- > 0;) {
        const size_t m = std::min(stash_[i].hist.size(), l.hist.size());
        if (m > 0 && std::memcmp(stash_[i].hist.data(), l.hist.data(), m * sizeof(int32_t)) == 0) {
            if (stash_[i].hist.size() <= l.hist.size()) stash_.erase(stash_.begin() + (std::ptrdiff_t) i);
        }
    }
    stash_.push_back(std::move(l));
    ++ring_stats.stash_stores;
    if (stash_budget_ > 0) {
        uint64_t total = ring_stash_bytes();
        while (stash_.size() > 1 && total > stash_budget_) {
            total -= std::min(total, stash_.front().bytes());
            stash_.pop_front();
            ++ring_stats.stash_evictions;
        }
    }
    return true;
}

void Dflash2Drafter::harvest_slot(const Dflash2Drafter& slot) {
    if (!log_on_ || slot.cur_.hist.empty() || slot.cur_.chunks.empty()) return;
    bool any_chunk = false;
    for (const auto& c : slot.cur_.chunks) if (c) { any_chunk = true; break; }
    if (!any_chunk) return;
    Lineage l = slot.cur_;
    for (size_t i = stash_.size(); i-- > 0;) {
        const size_t m = std::min(stash_[i].hist.size(), l.hist.size());
        if (m > 0 && std::memcmp(stash_[i].hist.data(), l.hist.data(), m * sizeof(int32_t)) == 0) {
            if (stash_[i].hist.size() <= l.hist.size()) stash_.erase(stash_.begin() + (std::ptrdiff_t) i);
            else return;
        }
    }
    stash_.push_back(std::move(l));
    ++ring_stats.stash_stores;
    if (stash_budget_ > 0) {
        uint64_t total = ring_stash_bytes();
        while (stash_.size() > 1 && total > stash_budget_) {
            total -= std::min(total, stash_.front().bytes());
            stash_.pop_front();
            ++ring_stats.stash_evictions;
        }
    }
}

void Dflash2Drafter::invalidate_ring() {
    ++epoch_;
    ring_hi_ = 0;
    for (int64_t& t : slot_pos_) t = -1;
}

void Dflash2Drafter::stash_push(Lineage&& l) {
    if (!log_on_ || stash_budget_ == 0 || l.hist.size() < 128 || l.chunks.empty()) return;
    stash_.push_back(std::move(l));
    ++ring_stats.stash_stores;
    uint64_t total = ring_stash_bytes();
    while (!stash_.empty() && total > stash_budget_) {
        total -= std::min(total, stash_.front().bytes());
        stash_.pop_front();
        ++ring_stats.stash_evictions;
    }
}

void Dflash2Drafter::log_stage(int64_t pos0, int n) {
    if (!log_on_) return;
    pend_pos0_ = pos0;
    pend_n_ = n;
    for (int i = 0; i < n; ++i) slot_pos_[(size_t) ((pos0 + i) % kRing)] = ((int64_t) epoch_ << 32) | (pos0 + i);
}

void Dflash2Drafter::log_flush(int row_off, const int8_t* q_src, const float* d_src) {
    if (!log_on_ || pend_pos0_ < 0) return;
    const int8_t* qp = q_src != nullptr ? q_src : h_logq_;
    const float* dp = d_src != nullptr ? d_src : h_logd_;
    for (int i = 0; i < pend_n_; ++i) {
        const int64_t pos = pend_pos0_ + i;
        LogChunk& ch = cur_.writable((size_t) (pos / kLogChunk));
        const size_t r = (size_t) (pos % kLogChunk);
        std::memcpy(&ch.q[r * 2560], qp + (size_t) (row_off + i) * 2560, 2560);
        std::memcpy(&ch.d[r * 80], dp + (size_t) (row_off + i) * 80, 80 * sizeof(float));
        ch.ok[r] = 1;
    }
    pend_pos0_ = -1;
}

bool Dflash2Drafter::rebuild(int64_t lo, int64_t hi, int64_t* rebuilt, int64_t* missing, std::string& err) {
    *rebuilt = 0;
    *missing = 0;
    std::vector<int32_t> pos;
    for (int64_t p = lo; p < hi; ++p) {
        if (slot_pos_[(size_t) (p % kRing)] == (((int64_t) epoch_ << 32) | p)) continue;   // the ring already holds this row
        if (cur_.logged(p)) pos.push_back((int32_t) p);
        else {
            if (++*missing <= 8 && std::getenv("STRATA_DF2_RING_STATS") != nullptr) std::fprintf(stderr, "strata dflash2 ring: position %lld was never logged (hist %zu)\n", (long long) p, cur_.hist.size());
        }
    }
    if (pos.empty()) return true;
    const OnDevice on_device(device_);
    const size_t rows = pos.size();
    const size_t cap = (size_t) kRing;   // a window is at most 2048 rows
    if (rb_q_ == nullptr) {
        if (cudaMalloc((void**) &rb_q_, (cap + kRows) * 2560) != cudaSuccess || cudaMalloc((void**) &rb_d_, (cap + kRows) * 80 * sizeof(float)) != cudaSuccess ||
            cudaMalloc((void**) &rb_pos_, (cap + kRows) * sizeof(int32_t)) != cudaSuccess) {
            if (rb_q_) cudaFree(rb_q_);
            if (rb_d_) cudaFree(rb_d_);
            rb_q_ = nullptr; rb_d_ = nullptr; rb_pos_ = nullptr;
            (void) cudaGetLastError();
            err = "dflash2: the ring rebuild's scratch does not fit";
            return false;
        }
        cudaMemset(rb_q_, 0, (cap + kRows) * 2560);
        cudaMemset(rb_d_, 0, (cap + kRows) * 80 * sizeof(float));
        rb_bytes_ = (cap + kRows) * (2560 + 80 * sizeof(float) + sizeof(int32_t));
        vram_ += rb_bytes_;
    }
    std::vector<int8_t> q(rows * 2560);
    std::vector<float> d(rows * 80);
    for (size_t i = 0; i < rows; ++i) {
        const int64_t p = pos[i];
        const LogChunk& ch = *cur_.chunks[(size_t) (p / kLogChunk)];
        const size_t r = (size_t) (p % kLogChunk);
        std::memcpy(&q[i * 2560], &ch.q[r * 2560], 2560);
        std::memcpy(&d[i * 80], &ch.d[r * 80], 80 * sizeof(float));
    }
    h_meta_[18] = epoch_;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    kernels::copy_i32_from_mapped(d_meta_, m_meta_, 32, cs_);
    cudaMemcpyAsync(rb_q_, q.data(), q.size(), cudaMemcpyHostToDevice, cs_);
    cudaMemcpyAsync(rb_d_, d.data(), d.size() * sizeof(float), cudaMemcpyHostToDevice, cs_);
    cudaMemcpyAsync(rb_pos_, pos.data(), rows * sizeof(int32_t), cudaMemcpyHostToDevice, cs_);
    // (the host vectors die with this scope: wait for the copies)
    if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = std::string("dflash2: ring rebuild: ") + cudaGetErrorString(cudaGetLastError()); return false; }
    const bool was_fork = fork_;
    fork_ = false;
    bool ok = true;
    if (!blob_shared_ && rt_bctx_ != nullptr && !prof_) {
        for (size_t b0 = 0; ok && b0 < rows; b0 += 64) {
            const int batch_n = (int) std::min<size_t>(64, rows - b0);
            try { kernels::df2_rope_table_inv(rb_pos_ + b0, batch_n, rope_inv_, rt_bctx_, cs_); } catch (const std::exception& e) { err = e.what(); ok = false; break; }
            for (int o = 0; ok && o < batch_n; o += 16) {
                const int m = std::min(16, batch_n - o);
                ok = record_kv(m, cs_, reinterpret_cast<const uint8_t*>(rb_q_) + (b0 + (size_t) o) * 2560,
                               rb_d_ + (b0 + (size_t) o) * 80, rb_pos_ + b0 + o, err, rt_bctx_ + (size_t) o * 128);
            }
        }
    } else {
        for (size_t r0 = 0; ok && r0 < rows; r0 += kRows) {
            const int m = (int) std::min<size_t>(kRows, rows - r0);
            try { kernels::df2_rope_table_inv(rb_pos_ + r0, m, rope_inv_, rt_ctx_, cs_); } catch (const std::exception& e) { err = e.what(); ok = false; break; }
            ok = record_kv(m, cs_, reinterpret_cast<const uint8_t*>(rb_q_) + r0 * 2560, rb_d_ + r0 * 80, rb_pos_ + r0, err);
        }
    }
    fork_ = was_fork;
    if (!ok) return false;
    if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = std::string("dflash2: ring rebuild: ") + cudaGetErrorString(cudaGetLastError()); return false; }
    for (int32_t p : pos) slot_pos_[(size_t) (p % kRing)] = ((int64_t) epoch_ << 32) | p;
    ring_hi_ = std::max<int64_t>(ring_hi_, (int64_t) pos.back() + 1);
    *rebuilt = (int64_t) rows;
    return true;
}

bool Dflash2Drafter::begin_request(const int32_t* ids, int64_t n, int64_t resume, bool* cold) {
    resume = std::max<int64_t>(0, std::min(resume, n));
    if (cold) *cold = false;
    ++ring_stats.requests;
    last_rebuilt = last_missing = 0;
    last_ms = 0;
    last_swapped = false;
    std::string err;
    (void) context_wait(err);
    const Clock::time_point t0 = Clock::now();
    auto lcp_of = [&](const std::vector<int32_t>& h) {
        int64_t l = 0;
        const int64_t lim = std::min<int64_t>((int64_t) h.size(), resume);
        while (l < lim && h[(size_t) l] == ids[l]) ++l;
        return l;
    };
    if (!log_on_) {   // the pre-log rule: a history that does not cover the resumed prefix starts the ring over
        const int64_t lcp = lcp_of(cur_.hist);
        const bool stale = (resume <= 0 && ring_hi_ > 0) || (ring_hi_ > lcp && lcp < resume);
        if (stale) { ++epoch_; ring_hi_ = 0; }
        if (cold) *cold = stale;
        cur_.hist.assign(ids, ids + resume);
        return true;
    }
    bool covered = dflash2_begin_request_covered(lcp_of(cur_.hist), resume, cur_.hist.size());
    bool swapped = false;
    if (!covered && resume > 0) {   // a parked lineage may be the one this request resumes
        for (size_t i = stash_.size(); i-- > 0;) {
            if (lcp_of(stash_[i].hist) < resume) continue;
            Lineage got = std::move(stash_[i]);
            stash_.erase(stash_.begin() + (std::ptrdiff_t) i);
            stash_push(std::move(cur_));
            cur_ = std::move(got);
            invalidate_ring();
            covered = swapped = true;
            ++ring_stats.stash_hits;
            break;
        }
    }
    if (!covered) {   // another conversation: the live lineage is parked, this one starts empty
        stash_push(std::move(cur_));
        cur_ = Lineage{};
        invalidate_ring();
        ++ring_stats.cold_starts;
    } else if ((int64_t) cur_.hist.size() >= resume + 128) {
        // the rows past `resume` are about to be rewritten (a rewound or branching request): keep the old branch as it is, sharing the log's chunks
        Lineage copy;
        copy.hist = cur_.hist;
        copy.chunks = cur_.chunks;
        stash_push(std::move(copy));
    }
    if ((int64_t) cur_.hist.size() > resume) {
        cur_.hist.resize((size_t) resume);
        cur_.truncate(resume);
        if (covered) {
            const int64_t clear_hi = std::min<int64_t>(ring_hi_, resume + kRing);
            for (int64_t p = std::max<int64_t>(0, resume); p < clear_hi; ++p) {
                const size_t sl = (size_t) (p % kRing);
                if ((slot_pos_[sl] >> 32) == (int64_t) epoch_ && (slot_pos_[sl] & 0xFFFFFFFFll) >= resume)
                    slot_pos_[sl] = -1;
            }
            ring_hi_ = dflash2_rewind_ring_hi(ring_hi_, resume);
        }
    }
    if (!covered) cur_.hist.assign(ids, ids + resume);
    // the window the first draft reads, below `resume` (the prompt's own taps cover [resume, n)): every slot that does not hold its row is rebuilt from the log
    int64_t rebuilt = 0, missing = 0;
    const int64_t lo = std::max<int64_t>(0, n - (window_ - 1)), hi = resume;
    if (hi > lo) {
        if (!rebuild(lo, hi, &rebuilt, &missing, err)) {
            std::fprintf(stderr, "strata dflash2: %s: the ring starts over for this request\n", err.c_str());
            invalidate_ring();
            missing = hi - lo;
            rebuilt = 0;
        }
    }
    static const bool verbose = std::getenv("STRATA_DF2_RING_STATS") != nullptr;
    if (verbose)
        std::fprintf(stderr, "strata dflash2 ring: request %lld resume %lld of %lld: live log %.1f MiB (%lld positions), parked %zu lineages %.1f MiB, rebuilt %lld rows (missing %lld) in %.2f ms%s\n",
                     (long long) ring_stats.requests, (long long) resume, (long long) n, (double) cur_.bytes() / 1048576.0, (long long) cur_.hist.size(), stash_.size(),
                     (double) ring_stash_bytes() / 1048576.0, (long long) rebuilt, (long long) missing, ms_since(t0), swapped ? " (swapped in)" : "");
    ring_stats.rows_rebuilt += rebuilt;
    ring_stats.rows_missing += missing;
    if (rebuilt > 0) ++ring_stats.rebuilds;
    const double ms = ms_since(t0);
    if (rebuilt > 0) ring_stats.ms_rebuild += ms;
    if (swapped) ring_stats.ms_swap += ms;
    if (cold) *cold = missing > 0;
    last_rebuilt = rebuilt;
    last_missing = missing;
    last_ms = ms;
    last_swapped = swapped;
    return true;
}

void Dflash2Drafter::note_rows(const int32_t* toks, int64_t n, int64_t pos0) {
    if (pos0 < 0) return;
    if ((int64_t) cur_.hist.size() > pos0) { cur_.hist.resize((size_t) pos0); if (log_on_) cur_.truncate(pos0); }
    if ((int64_t) cur_.hist.size() < pos0) return;   // a gap: the history stops being a prefix, so it stays where it is (the next begin_request starts over)
    cur_.hist.insert(cur_.hist.end(), toks, toks + n);
}

bool Dflash2Drafter::context(const float* taps, int64_t n, int64_t pos0, std::string& err) {
    if (!context_wait(err)) return false;
    const OnDevice on_device(device_);
    const int tap_n = (int) taps_.size() * N_;
    const int64_t stage_cap = (int64_t) kRows * 8;
    int precomp_off = 0;
    if (!blob_shared_ && rt_bctx_ != nullptr && !prof_ && !skip_part("ctx") && n >= 1 && n <= kBatchMaxSlots * kRows &&
        has_precomputed_context(taps, (int) n, pos0, &precomp_off)) {
        if (precomp_off + (int) n >= precomp_rows_) clear_precomputed_context();
        try {
            log_stage(pos0, (int) n);
            h_meta_[18] = epoch_;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            kernels::copy_i32_from_mapped(d_meta_ + 18, m_meta_ + 18, 1, cs_);
            const size_t l_stride = (size_t) (kBatchMaxSlots * kRows) * NKV_ * HD_;
            const size_t row_kv = (size_t) precomp_off * NKV_ * HD_;
            for (int l = 0; l < L_; ++l) {
                const std::string p = "layers." + std::to_string(l) + ".";
                if (fp16_ring_)
                    kernels::df2_kv_ring_f16(kc_ + l * l_stride + row_kv, vc_ + l * l_stride + row_kv,
                                             f32(p + "self_attn.k_norm.weight"), d_meta_ + kMetaPrecompPos + precomp_off,
                                             rt_bctx_ + (size_t) precomp_off * 128, kring_[l], vring_[l],
                                             kRing, (int) n, NKV_, eps_, cs_, tag_, d_meta_ + 18);
                else
                    kernels::df2_kv_ring(kc_ + l * l_stride + row_kv, vc_ + l * l_stride + row_kv,
                                         f32(p + "self_attn.k_norm.weight"), d_meta_ + kMetaPrecompPos + precomp_off,
                                         rt_bctx_ + (size_t) precomp_off * 128, kring_[l], vring_[l],
                                         kRing, (int) n, NKV_, eps_, cs_, tag_, d_meta_ + 18);
            }
        } catch (const std::exception& e) {
            err = e.what();
            return false;
        }
        if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = std::string("dflash2: ") + cudaGetErrorString(cudaGetLastError()); return false; }
        log_flush(precomp_off);
        have_ctx_ = std::max<int64_t>(have_ctx_, pos0 + n);
        ring_hi_ = std::max<int64_t>(ring_hi_, pos0 + n);
        return true;
    }
    clear_precomputed_context();
    if (!blob_shared_ && rt_bctx_ != nullptr && !prof_ && !skip_part("ctx")) {
        try {
            uint8_t* xqc_log = xqc_ + (size_t) 80 * N_;
            float* xdc_log = xdc_ + (size_t) 80 * (N_ / 32);
            for (int64_t b0 = 0; b0 < n; b0 += stage_cap) {
                const int batch_n = (int) std::min<int64_t>(stage_cap, n - b0);
                log_stage(pos0 + b0, batch_n);
                for (int i = 0; i < batch_n; ++i) h_meta_[i] = (int32_t) (pos0 + b0 + i);
                h_meta_[64] = epoch_;
                std::atomic_thread_fence(std::memory_order_seq_cst);
                kernels::copy_i32_from_mapped(d_meta_, m_meta_, 65, cs_);
                kernels::df2_rope_table_inv(d_meta_, batch_n, rope_inv_, rt_bctx_, cs_);
                for (int o = 0; o < batch_n; o += 16) {
                    const int m = std::min(16, batch_n - o);
                    kernels::df2_quant_rows(taps + (size_t) (b0 + o) * (size_t) tap_n, tap_n, m, xqc_, xdc_, cs_);
                    gemm({"fc.weight"}, {fcy_}, m, cs_, xqc_, xdc_);
                    kernels::df2_rms_quant(fcy_, f32("hidden_norm.weight"), nullptr, m, N_, eps_,
                                           xqc_log + (size_t) o * N_, xdc_log + (size_t) o * (N_ / 32), cs_);
                    if (!record_kv(m, cs_, xqc_log + (size_t) o * N_, xdc_log + (size_t) o * (N_ / 32),
                                   d_meta_ + o, err, rt_bctx_ + (size_t) o * 128, d_meta_ + 64))
                        return false;
                }
                if (log_on_) {
                    cudaMemcpyAsync(h_logq_, xqc_log, (size_t) batch_n * N_, cudaMemcpyDeviceToHost, cs_);
                    cudaMemcpyAsync(h_logd_, xdc_log, (size_t) batch_n * (N_ / 32) * 4, cudaMemcpyDeviceToHost, cs_);
                }
                if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = std::string("dflash2: ") + cudaGetErrorString(cudaGetLastError()); return false; }
                log_flush();
            }
        } catch (const std::exception& e) {
            err = e.what();
            return false;
        }
        have_ctx_ = std::max<int64_t>(have_ctx_, pos0 + n);
        ring_hi_ = std::max<int64_t>(ring_hi_, pos0 + n);
        return true;
    }
    for (int64_t b0 = 0; b0 < n; b0 += stage_cap) {
        const int batch_n = (int) std::min<int64_t>(stage_cap, n - b0);
        log_stage(pos0 + b0, batch_n);
        for (int o = 0; o < batch_n; o += kRows) {
            const int s = o / kRows;
            const int m = std::min((int) kRows, batch_n - o);
            for (int i = 0; i < m; ++i) h_meta_[s * 32 + i] = (int32_t) (pos0 + b0 + o + i);
            h_meta_[s * 32 + 18] = epoch_;
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        for (int o = 0; o < batch_n; o += kRows) {
            const int s = o / kRows;
            const int m = std::min((int) kRows, batch_n - o);
            kernels::copy_i32_from_mapped(d_meta_, m_meta_ + s * 32, 32, cs_);
            if (!record_context(taps + (size_t) (b0 + o) * (size_t) tap_n, m, cs_, err, true, o)) return false;
        }
        if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = std::string("dflash2: ") + cudaGetErrorString(cudaGetLastError()); return false; }
        log_flush();
    }
    have_ctx_ = std::max<int64_t>(have_ctx_, pos0 + n);
    ring_hi_ = std::max<int64_t>(ring_hi_, pos0 + n);
    return true;
}

bool Dflash2Drafter::context_async(const float* taps, int64_t n, int64_t pos0, std::string& err) {
    if (n < 1 || n > kRows) { err = "dflash2: context_async rows out of range"; return false; }
    if (!context_wait(err)) return false;   // h_meta_ is read by the queued copy: the previous call must be done
    const OnDevice on_device(device_);
    for (int i = 0; i < (int) n; ++i) h_meta_[i] = (int32_t) (pos0 + i);
    h_meta_[18] = epoch_;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    kernels::copy_i32_from_mapped(d_meta_, m_meta_, 32, cs_);
    log_stage(pos0, (int) n);
    if (!record_context(taps, (int) n, cs_, err)) return false;
    ctx_pending_ = true;
    have_ctx_ = std::max<int64_t>(have_ctx_, pos0 + n);
    ring_hi_ = std::max<int64_t>(ring_hi_, pos0 + n);
    return true;
}

bool Dflash2Drafter::context_wait(std::string& err) {
    if (!ctx_pending_) return true;
    ctx_pending_ = false;
    const OnDevice on_device(device_);
    if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = std::string("dflash2: ") + cudaGetErrorString(cudaGetLastError()); return false; }
    log_flush();
    return true;
}

bool Dflash2Drafter::draft(const float* taps, int n_new, int64_t pos0, int32_t anchor, int32_t* drafts, float* probs, std::string& err) {
    return draft_launch(taps, n_new, pos0, anchor, err) && draft_finish(drafts, probs, err);
}

bool Dflash2Drafter::draft_launch(const float* taps, int n_new, int64_t pos0, int32_t anchor, std::string& err) {
    if (!bound_) { err = "dflash2: not bound"; return false; }
    if (!context_wait(err)) return false;
    if (n_new < 1 || n_new > kRows) { err = "dflash2: draft rows out of range"; return false; }
    const OnDevice on_device(device_);
    const Clock::time_point t0 = Clock::now();
    if (graph_taps_ != nullptr && graph_taps_ != taps) { err = "dflash2: the taps buffer moved after capture"; return false; }
    if (!prof_ && !exec_[n_new]) {
        // the staging must hold valid positions while the capture runs the (empty) kernels' parameter checks
        for (int i = 0; i < 8; ++i) h_meta_[i] = (int32_t) (pos0 + i);
        if (!capture(n_new, taps, err)) return false;
        graph_taps_ = taps;
    }
    const int64_t s = pos0 + n_new;
    for (int i = 0; i < 8; ++i) { h_meta_[i] = (int32_t) (pos0 + std::min(i, n_new - 1)); h_meta_[8 + i] = (int32_t) (s + i); }
    h_meta_[16] = anchor;
    h_meta_[18] = epoch_;
    h_meta_[17] = (int32_t) s;
    h_meta_[24] = anchor;
    for (int i = 1; i < 8; ++i) h_meta_[24 + i] = mask_token_;
    for (int i = 0; i < kDrafts; ++i) { ((volatile int32_t*) h_out_)[i] = -1; }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    log_stage(pos0, n_new);
    if (timing_) cudaEventRecord(ev_a_, cs_);
    if (prof_) {   // the same round without its graph, an event after every stage
        cudaStreamSynchronize(cs_);
        for (cudaEvent_t e : prof_->ev) cudaEventDestroy(e);   // the stamps of context() calls are not a round
        prof_->ev.clear();
        prof_->name.clear();
        kernels::copy_i32_from_mapped(d_meta_, m_meta_, 32, cs_);
        stamp("start");
        if (!record_context(taps, n_new, cs_, err) || !record_block(cs_, err)) return false;
        prof_collect();
    } else
    if (cudaGraphLaunch(exec_[n_new], cs_) != cudaSuccess) { err = std::string("dflash2 draft: ") + cudaGetErrorString(cudaGetLastError()); return false; }
    if (timing_) cudaEventRecord(ev_b_, cs_);
    (void) cudaStreamQuery(cs_);
    launch_t0_ = t0;
    launch_n_ = n_new;
    launch_pos0_ = pos0;
    launched_ = true;
    return true;
}

bool Dflash2Drafter::draft_finish(int32_t* drafts, float* probs, std::string& err) {
    if (!launched_) { err = "dflash2: draft_finish without draft_launch"; return false; }
    launched_ = false;
    const OnDevice on_device(device_);
    const int n_new = launch_n_;
    const int64_t pos0 = launch_pos0_, s = pos0 + n_new;
    const Clock::time_point t0 = launch_t0_;
    static const bool solopipe = [] {
        const char* e = std::getenv("STRATA_DF2_SOLOPIPE");
        return !e || e[0] != '0';
    }();
    if (solopipe && !timing_ && !prof_ && ndraft_ > 0) {
        volatile int32_t* const sentinel = (volatile int32_t*) h_out_ + (ndraft_ - 1);
        uint32_t spins = 0;
        while (*sentinel < 0) {
            strata_cpu_pause();
            if ((++spins & 1023u) == 0 && cudaStreamQuery(cs_) != cudaErrorNotReady) break;
        }
        std::atomic_thread_fence(std::memory_order_acquire);
    }
    if (!solopipe || timing_ || prof_ || ((volatile int32_t*) h_out_)[std::max(0, ndraft_ - 1)] < 0) {
        if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = std::string("dflash2 draft: ") + cudaGetErrorString(cudaGetLastError()); return false; }
    }
    log_flush();
    for (int i = 0; i < kDrafts; ++i) {
        const bool made = i < ndraft_;
        drafts[i] = made ? ((volatile int32_t*) h_out_)[i] : 0;
        if (probs) probs[i] = made ? ((volatile float*) h_out_)[16 + i] : 0.0f;
    }
    if (timing_) {
        float ms = 0;
        if (cudaEventElapsedTime(&ms, ev_a_, ev_b_) == cudaSuccess) ms_gpu += ms;
    }
    have_ctx_ = std::max<int64_t>(have_ctx_, s);
    ring_hi_ = std::max<int64_t>(ring_hi_, pos0 + n_new);
    ms_draft += ms_since(t0);
    ++rounds;
    return true;
}

void Dflash2Drafter::precompute_batch_context(const float* taps, int n_rows, const int64_t* pos) noexcept {
    if (!bound_ || blob_shared_ || skip_part("ctx") || taps == nullptr || pos == nullptr ||
        n_rows < 1 || n_rows > kBatchMaxSlots * kRows || rt_bctx_ == nullptr) return;
    try {
        using namespace kernels;
        const OnDevice on_device(device_);
        for (int r = 0; r < n_rows; ++r) {
            h_meta_[kMetaPrecompPos + r] = (int32_t) pos[r];
            precomp_pos_[r] = (int32_t) pos[r];
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        copy_i32_from_mapped(d_meta_ + kMetaPrecompPos, m_meta_ + kMetaPrecompPos, n_rows, cs_);
        df2_rope_table_inv(d_meta_ + kMetaPrecompPos, n_rows, rope_inv_, rt_bctx_, cs_);
        const int tap_n = (int) taps_.size() * N_;
        df2_quant_rows(taps, tap_n, n_rows, xqc_, xdc_, cs_);
        gemm({"fc.weight"}, {fcy_}, n_rows, cs_, xqc_, xdc_);
        df2_rms_quant(fcy_, f32("hidden_norm.weight"), nullptr, n_rows, N_, eps_, xqc_, xdc_, cs_);
        const size_t l_stride = (size_t) (kBatchMaxSlots * kRows) * NKV_ * HD_;
        if (L_ == 5) {
            gemm({"layers.0.self_attn.k_proj.weight", "layers.0.self_attn.v_proj.weight",
                  "layers.1.self_attn.k_proj.weight", "layers.1.self_attn.v_proj.weight",
                  "layers.2.self_attn.k_proj.weight", "layers.2.self_attn.v_proj.weight",
                  "layers.3.self_attn.k_proj.weight", "layers.3.self_attn.v_proj.weight",
                  "layers.4.self_attn.k_proj.weight", "layers.4.self_attn.v_proj.weight"},
                 {kc_ + 0 * l_stride, vc_ + 0 * l_stride,
                  kc_ + 1 * l_stride, vc_ + 1 * l_stride,
                  kc_ + 2 * l_stride, vc_ + 2 * l_stride,
                  kc_ + 3 * l_stride, vc_ + 3 * l_stride,
                  kc_ + 4 * l_stride, vc_ + 4 * l_stride},
                 n_rows, cs_, xqc_, xdc_);
        } else if (L_ == 8) {
            gemm({"layers.0.self_attn.k_proj.weight", "layers.0.self_attn.v_proj.weight",
                  "layers.1.self_attn.k_proj.weight", "layers.1.self_attn.v_proj.weight",
                  "layers.2.self_attn.k_proj.weight", "layers.2.self_attn.v_proj.weight",
                  "layers.3.self_attn.k_proj.weight", "layers.3.self_attn.v_proj.weight",
                  "layers.4.self_attn.k_proj.weight", "layers.4.self_attn.v_proj.weight",
                  "layers.5.self_attn.k_proj.weight", "layers.5.self_attn.v_proj.weight",
                  "layers.6.self_attn.k_proj.weight", "layers.6.self_attn.v_proj.weight",
                  "layers.7.self_attn.k_proj.weight", "layers.7.self_attn.v_proj.weight"},
                 {kc_ + 0 * l_stride, vc_ + 0 * l_stride,
                  kc_ + 1 * l_stride, vc_ + 1 * l_stride,
                  kc_ + 2 * l_stride, vc_ + 2 * l_stride,
                  kc_ + 3 * l_stride, vc_ + 3 * l_stride,
                  kc_ + 4 * l_stride, vc_ + 4 * l_stride,
                  kc_ + 5 * l_stride, vc_ + 5 * l_stride,
                  kc_ + 6 * l_stride, vc_ + 6 * l_stride,
                  kc_ + 7 * l_stride, vc_ + 7 * l_stride},
                 n_rows, cs_, xqc_, xdc_);
        } else {
            for (int l = 0; l < L_; ++l) {
                const std::string p = "layers." + std::to_string(l) + ".";
                const std::string kp = p + "self_attn.k_proj.weight", vp = p + "self_attn.v_proj.weight";
                gemm({kp.c_str(), vp.c_str()}, {kc_ + l * l_stride, vc_ + l * l_stride}, n_rows, cs_, xqc_, xdc_);
            }
        }
        if (log_on_ && h_logq_ != nullptr && h_logd_ != nullptr) {
            cudaMemcpyAsync(h_logq_, xqc_, (size_t) n_rows * N_, cudaMemcpyDeviceToHost, cs_);
            cudaMemcpyAsync(h_logd_, xdc_, (size_t) n_rows * (N_ / 32) * 4, cudaMemcpyDeviceToHost, cs_);
            if (ev_join_ != nullptr) cudaEventRecord(ev_join_, cs_);
        }
        (void) cudaStreamQuery(cs_);
        precomp_taps_ = taps;
        precomp_rows_ = n_rows;
    } catch (...) {
        precomp_taps_ = nullptr;
        precomp_rows_ = 0;
    }
}

bool Dflash2Drafter::has_precomputed_context(const float* taps, int n_rows, int64_t pos0, int* out_off) const noexcept {
    if (precomp_taps_ == nullptr || taps == nullptr || n_rows < 1 || precomp_rows_ < n_rows) return false;
    const ptrdiff_t diff = taps - precomp_taps_;
    const ptrdiff_t tap_n = (ptrdiff_t) taps_.size() * N_;
    if (tap_n <= 0 || diff < 0 || (diff % tap_n) != 0) return false;
    const ptrdiff_t row0 = diff / tap_n;
    if (row0 < 0 || row0 + n_rows > precomp_rows_) return false;
    for (int r = 0; r < n_rows; ++r) {
        if (precomp_pos_[row0 + r] != (int32_t) (pos0 + r)) return false;
    }
    if (out_off) *out_off = (int) row0;
    return true;
}

bool Dflash2Drafter::record_batch_body(const BatchSlotItem* items, int n_items, cudaStream_t cs, std::string& err) {
    using namespace kernels;
    try {
        int total_ctx = 0;
        for (int idx = 0; idx < n_items; ++idx) total_ctx += items[idx].n_new;
        const int total_B = n_items * kRows;
        df2_rope_table_inv(d_meta_ + kMetaCtx, (kBatchMaxSlots * kRows) + total_B, rope_inv_, rt_bctx_, cs);
        if (!skip_part("ctx") && total_ctx > 0) {
            const int tap_n = (int) taps_.size() * N_;
            int c0 = 0;
            for (int idx = 0; idx < n_items; ++idx) {
                const int n = items[idx].n_new;
                df2_quant_rows(items[idx].taps, tap_n, n,
                               xqc_ + (size_t) c0 * tap_n, xdc_ + (size_t) c0 * (tap_n / 32), cs);
                c0 += n;
            }
            gemm({"fc.weight"}, {fcy_}, total_ctx, cs, xqc_, xdc_);
            df2_rms_quant(fcy_, f32("hidden_norm.weight"), nullptr, total_ctx, N_, eps_, xqc_, xdc_, cs);
            Df2BatchKvRing bkv{};
            c0 = 0;
            for (int idx = 0; idx < n_items; ++idx) {
                Dflash2Drafter& sl = *items[idx].slot;
                const int n = items[idx].n_new;
                bkv.tag[idx] = sl.tag_;
                for (int r = 0; r < n; ++r) bkv.row_slot[c0 + r] = (uint8_t) idx;
                c0 += n;
            }
            const size_t l_stride = (size_t) (kBatchMaxSlots * kRows) * NKV_ * HD_;
            if (L_ == 5) {
                gemm({"layers.0.self_attn.k_proj.weight", "layers.0.self_attn.v_proj.weight",
                      "layers.1.self_attn.k_proj.weight", "layers.1.self_attn.v_proj.weight",
                      "layers.2.self_attn.k_proj.weight", "layers.2.self_attn.v_proj.weight",
                      "layers.3.self_attn.k_proj.weight", "layers.3.self_attn.v_proj.weight",
                      "layers.4.self_attn.k_proj.weight", "layers.4.self_attn.v_proj.weight"},
                     {kc_ + 0 * l_stride, vc_ + 0 * l_stride,
                      kc_ + 1 * l_stride, vc_ + 1 * l_stride,
                      kc_ + 2 * l_stride, vc_ + 2 * l_stride,
                      kc_ + 3 * l_stride, vc_ + 3 * l_stride,
                      kc_ + 4 * l_stride, vc_ + 4 * l_stride},
                     total_ctx, cs, xqc_, xdc_);
            } else if (L_ == 8) {
                gemm({"layers.0.self_attn.k_proj.weight", "layers.0.self_attn.v_proj.weight",
                      "layers.1.self_attn.k_proj.weight", "layers.1.self_attn.v_proj.weight",
                      "layers.2.self_attn.k_proj.weight", "layers.2.self_attn.v_proj.weight",
                      "layers.3.self_attn.k_proj.weight", "layers.3.self_attn.v_proj.weight",
                      "layers.4.self_attn.k_proj.weight", "layers.4.self_attn.v_proj.weight",
                      "layers.5.self_attn.k_proj.weight", "layers.5.self_attn.v_proj.weight",
                      "layers.6.self_attn.k_proj.weight", "layers.6.self_attn.v_proj.weight",
                      "layers.7.self_attn.k_proj.weight", "layers.7.self_attn.v_proj.weight"},
                     {kc_ + 0 * l_stride, vc_ + 0 * l_stride,
                      kc_ + 1 * l_stride, vc_ + 1 * l_stride,
                      kc_ + 2 * l_stride, vc_ + 2 * l_stride,
                      kc_ + 3 * l_stride, vc_ + 3 * l_stride,
                      kc_ + 4 * l_stride, vc_ + 4 * l_stride,
                      kc_ + 5 * l_stride, vc_ + 5 * l_stride,
                      kc_ + 6 * l_stride, vc_ + 6 * l_stride,
                      kc_ + 7 * l_stride, vc_ + 7 * l_stride},
                     total_ctx, cs, xqc_, xdc_);
            }
            for (int l = 0; l < L_; ++l) {
                const std::string p = "layers." + std::to_string(l) + ".";
                if (L_ != 5 && L_ != 8) {
                    const std::string kp = p + "self_attn.k_proj.weight", vp = p + "self_attn.v_proj.weight";
                    gemm({kp.c_str(), vp.c_str()}, {kc_ + l * l_stride, vc_ + l * l_stride}, total_ctx, cs, xqc_, xdc_);
                }
                for (int idx = 0; idx < n_items; ++idx) {
                    bkv.kring[idx] = items[idx].slot->kring_[l];
                    bkv.vring[idx] = items[idx].slot->vring_[l];
                }
                if (fp16_ring_)
                    df2_kv_ring_batch_f16(kc_ + l * l_stride, vc_ + l * l_stride, f32(p + "self_attn.k_norm.weight"), d_meta_ + kMetaCtx, rt_bctx_,
                                          bkv, kRing, total_ctx, NKV_, eps_, cs, d_meta_ + kMetaEpoch);
                else
                    df2_kv_ring_batch(kc_ + l * l_stride, vc_ + l * l_stride, f32(p + "self_attn.k_norm.weight"), d_meta_ + kMetaCtx, rt_bctx_,
                                      bkv, kRing, total_ctx, NKV_, eps_, cs, d_meta_ + kMetaEpoch);
            }
            c0 = 0;
            for (int idx = 0; idx < n_items; ++idx) {
                Dflash2Drafter& sl = *items[idx].slot;
                const int n = items[idx].n_new;
                if (sl.log_on_) {
                    cudaMemcpyAsync(sl.h_logq_, xqc_ + (size_t) c0 * N_, (size_t) n * N_, cudaMemcpyDeviceToHost, cs);
                    cudaMemcpyAsync(sl.h_logd_, xdc_ + (size_t) c0 * (N_ / 32), (size_t) n * (N_ / 32) * 4, cudaMemcpyDeviceToHost, cs);
                }
                c0 += n;
            }
        }
        embed_(d_meta_ + kMetaBlkTok, total_B, h_, cs);
        df2_rms_quant(h_, f32("layers.0.input_layernorm.weight"), xn_, total_B, N_, eps_, xqp_, xdp_, cs);
        for (int l = 0; l < L_; ++l) {
            const std::string p = "layers." + std::to_string(l) + ".";
            const float* base_a = f32(p + "attention_conv.base_kernel");
            const float* base_m = f32(p + "mlp_conv.base_kernel");
            const std::string akp = p + "attention_conv.kernel_projection.weight";
            const std::string qp = p + "self_attn.q_proj.weight", kp = p + "self_attn.k_proj.weight", vp = p + "self_attn.v_proj.weight";
            const std::string op = p + "self_attn.o_proj.weight";
            const std::string mkp = p + "mlp_conv.kernel_projection.weight";
            const std::string gp = p + "mlp.gate_proj.weight", up = p + "mlp.up_proj.weight";
            if (!skip_part("tiny")) gemm({akp.c_str()}, {dyn_}, total_B, cs, xqp_, xdp_);
            if (!skip_part("small")) df2_conv_quant(xn_, dyn_, 640, base_a, total_B, N_, group_, xqp_, xdp_, cs);
            if (!skip_part("qkvo")) gemm({qp.c_str(), kp.c_str(), vp.c_str()}, {q_, k_, v_}, total_B, cs, xqp_, xdp_);
            if (!skip_part("small")) df2_norm_rope_qk(q_, k_, f32(p + "self_attn.q_norm.weight"), f32(p + "self_attn.k_norm.weight"), rt_bblk_, qn_, kblk_, total_B, NH_, NKV_, eps_, cs);
            Df2BatchAttn ba{};
            for (int idx = 0; idx < n_items; ++idx) {
                ba.kring[idx] = items[idx].slot->kring_[l];
                ba.vring[idx] = items[idx].slot->vring_[l];
                ba.tag[idx] = items[idx].slot->tag_;
            }
            if (!skip_part("attn")) {
                if (fp16_ring_)
                    df2_attention_batch_f16(qn_, ba, kblk_, v_, d_meta_ + kMetaAnchor, kRing, window_, part_, xqp_, xdp_, n_items, cs, d_meta_ + kMetaEpoch);
                else
                    df2_attention_batch(qn_, ba, kblk_, v_, d_meta_ + kMetaAnchor, kRing, window_, part_, xqp_, xdp_, n_items, cs, d_meta_ + kMetaEpoch);
            }
            if (!skip_part("qkvo")) gemm({op.c_str()}, {o_}, total_B, cs, xqp_, xdp_);
            if (!skip_part("small")) df2_conv_rms_quant(o_, dyn_ + 320, 640, base_a + 2 * N_, h_, f32(p + "post_attention_layernorm.weight"), xn_, total_B, N_, group_, eps_, xqp_, xdp_, cs);
            if (!skip_part("tiny")) gemm({mkp.c_str()}, {dyn_}, total_B, cs, xqp_, xdp_);
            if (!skip_part("small")) df2_conv_quant(xn_, dyn_, 640, base_m, total_B, N_, group_, xqp_, xdp_, cs);
            if (!skip_part("mlp")) gemm({gp.c_str(), up.c_str()}, {gate_, up_}, total_B, cs, xqp_, xdp_);
            if (!skip_part("small")) df2_swiglu_quant(gate_, up_, total_B, I_, xqp_, xdp_, cs);
            if (!skip_part("mlp")) gemm({(p + "mlp.down_proj.weight").c_str()}, {mlp_}, total_B, cs, xqp_, xdp_);
            const std::string next_gamma = l + 1 < L_ ? "layers." + std::to_string(l + 1) + ".input_layernorm.weight" : std::string("norm.weight");
            if (!skip_part("small")) df2_conv_rms_quant(mlp_, dyn_ + 320, 640, base_m + 2 * N_, h_, f32(next_gamma), l + 1 < L_ ? xn_ : fn_, total_B, N_, group_, eps_, xqp_, xdp_, cs);
        }
        if (skip_part("small")) df2_quant_rows(fn_, N_, total_B, xqp_, xdp_, cs);
        gemm({"candidate_selector.hidden_projection.weight"}, {hsel_}, total_B, cs, xqp_, xdp_);
        df2_pack_head_rows_batch(fn_, hsel_, fn_batch_, hsel_batch_, m_bmeta_, n_items, N_, rank_, cs);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

bool Dflash2Drafter::record_batch_body_precomp(const BatchSlotItem* items, int n_items, cudaStream_t cs, std::string& err, int total_rows) {
    using namespace kernels;
    try {
        const int total_B = n_items * kRows;
        Df2BatchKvRingPrecomp bkv{};
        for (int idx = 0; idx < n_items; ++idx) {
            for (int l = 0; l < L_; ++l) {
                bkv.kring[idx][l] = items[idx].slot->kring_[l];
                bkv.vring[idx][l] = items[idx].slot->vring_[l];
            }
            bkv.tag[idx] = items[idx].slot->tag_;
        }
        for (int l = 0; l < L_; ++l) {
            bkv.gk[l] = f32("layers." + std::to_string(l) + ".self_attn.k_norm.weight");
        }
        const size_t l_stride = (size_t) (kBatchMaxSlots * kRows) * NKV_ * HD_;
        if (!skip_part("ctx")) {
            if (fp16_ring_)
                df2_kv_ring_batch_precomp_f16(kc_, vc_, l_stride, d_meta_ + kMetaPrecompPos, rt_bctx_, bkv, d_meta_ + kMetaSlotInfo, d_meta_ + kMetaEpoch,
                                              n_items, L_, kRing, NKV_, eps_, cs, rope_inv_, d_meta_ + kMetaBlkPos, rt_bblk_, total_B);
            else
                df2_kv_ring_batch_precomp(kc_, vc_, l_stride, d_meta_ + kMetaPrecompPos, rt_bctx_, bkv, d_meta_ + kMetaSlotInfo, d_meta_ + kMetaEpoch,
                                          n_items, L_, kRing, NKV_, eps_, cs, rope_inv_, d_meta_ + kMetaBlkPos, rt_bblk_, total_B);
        } else {
            df2_rope_table_inv(d_meta_ + kMetaBlkPos, total_B, rope_inv_, rt_bblk_, cs);
        }
        embed_(d_meta_ + kMetaBlkTok, total_B, h_, cs);
        df2_rms_quant(h_, f32("layers.0.input_layernorm.weight"), xn_, total_B, N_, eps_, xqp_, xdp_, cs);
        for (int l = 0; l < L_; ++l) {
            const std::string p = "layers." + std::to_string(l) + ".";
            const float* base_a = f32(p + "attention_conv.base_kernel");
            const float* base_m = f32(p + "mlp_conv.base_kernel");
            const std::string akp = p + "attention_conv.kernel_projection.weight";
            const std::string qp = p + "self_attn.q_proj.weight", kp = p + "self_attn.k_proj.weight", vp = p + "self_attn.v_proj.weight";
            const std::string op = p + "self_attn.o_proj.weight";
            const std::string mkp = p + "mlp_conv.kernel_projection.weight";
            const std::string gp = p + "mlp.gate_proj.weight", up = p + "mlp.up_proj.weight";
            if (!skip_part("tiny")) gemm({akp.c_str()}, {dyn_}, total_B, cs, xqp_, xdp_);
            if (!skip_part("small")) df2_conv_quant(xn_, dyn_, 640, base_a, total_B, N_, group_, xqp_, xdp_, cs);
            if (!skip_part("qkvo")) gemm({qp.c_str(), kp.c_str(), vp.c_str()}, {q_, k_, v_}, total_B, cs, xqp_, xdp_);
            if (!skip_part("small")) df2_norm_rope_qk(q_, k_, f32(p + "self_attn.q_norm.weight"), f32(p + "self_attn.k_norm.weight"), rt_bblk_, qn_, kblk_, total_B, NH_, NKV_, eps_, cs);
            Df2BatchAttn ba{};
            for (int idx = 0; idx < n_items; ++idx) {
                ba.kring[idx] = items[idx].slot->kring_[l];
                ba.vring[idx] = items[idx].slot->vring_[l];
                ba.tag[idx] = items[idx].slot->tag_;
            }
            if (!skip_part("attn")) {
                if (fp16_ring_)
                    df2_attention_batch_f16(qn_, ba, kblk_, v_, d_meta_ + kMetaAnchor, kRing, window_, part_, xqp_, xdp_, n_items, cs, d_meta_ + kMetaEpoch);
                else
                    df2_attention_batch(qn_, ba, kblk_, v_, d_meta_ + kMetaAnchor, kRing, window_, part_, xqp_, xdp_, n_items, cs, d_meta_ + kMetaEpoch);
            }
            if (!skip_part("qkvo")) gemm({op.c_str()}, {o_}, total_B, cs, xqp_, xdp_);
            if (!skip_part("small")) df2_conv_rms_quant(o_, dyn_ + 320, 640, base_a + 2 * N_, h_, f32(p + "post_attention_layernorm.weight"), xn_, total_B, N_, group_, eps_, xqp_, xdp_, cs);
            if (!skip_part("tiny")) gemm({mkp.c_str()}, {dyn_}, total_B, cs, xqp_, xdp_);
            if (!skip_part("small")) df2_conv_quant(xn_, dyn_, 640, base_m, total_B, N_, group_, xqp_, xdp_, cs);
            if (!skip_part("mlp")) gemm({gp.c_str(), up.c_str()}, {gate_, up_}, total_B, cs, xqp_, xdp_);
            if (!skip_part("small")) df2_swiglu_quant(gate_, up_, total_B, I_, xqp_, xdp_, cs);
            if (!skip_part("mlp")) gemm({(p + "mlp.down_proj.weight").c_str()}, {mlp_}, total_B, cs, xqp_, xdp_);
            const std::string next_gamma = l + 1 < L_ ? "layers." + std::to_string(l + 1) + ".input_layernorm.weight" : std::string("norm.weight");
            if (!skip_part("small")) df2_conv_rms_quant(mlp_, dyn_ + 320, 640, base_m + 2 * N_, h_, f32(next_gamma), l + 1 < L_ ? xn_ : fn_, total_B, N_, group_, eps_, xqp_, xdp_, cs);
        }
        if (skip_part("small")) df2_quant_rows(fn_, N_, total_B, xqp_, xdp_, cs);
        if (total_rows > 0 && total_rows <= total_B) {
            uint8_t* xqp_pack = xqp_ + (size_t) total_B * N_;
            float* xdp_pack = xdp_ + (size_t) total_B * (N_ / 32);
            const int nc0 = std::min(df2_head_chunk(), total_rows);
            const bool fuse_q8 = !skip_part("head") && xq_ != nullptr;
            const bool use_il0 = fuse_q8 && df2_head_use_il(head_type_, nc0, (int) n_dvocab_, df2_head_il_mode());
            df2_pack_head_rows_batch(fn_, nullptr, fn_batch_, nullptr, m_bmeta_, n_items, N_, rank_, cs,
                                     xqp_, xdp_, xqp_pack, xdp_pack,
                                     fuse_q8 ? xq_ : nullptr,
                                     (use_il0 && nc0 >= 2) ? xil_ : nullptr,
                                     (use_il0 && nc0 >= 2) ? nc0 : 0,
                                     fuse_q8 ? nc0 : 0);
            gemm({"candidate_selector.hidden_projection.weight"}, {hsel_batch_}, total_rows, cs, xqp_pack, xdp_pack);
        } else {
            gemm({"candidate_selector.hidden_projection.weight"}, {hsel_}, total_B, cs, xqp_, xdp_);
            df2_pack_head_rows_batch(fn_, hsel_, fn_batch_, hsel_batch_, m_bmeta_, n_items, N_, rank_, cs);
        }
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

bool Dflash2Drafter::record_batch_head(int n_items, int total_rows, cudaStream_t cs, std::string& err, bool q81_prepacked) {
    try {
        using namespace kernels;
        const int head_il = df2_head_il_mode();
        const int head_chunk = df2_head_chunk();
        const bool force_il_rows_env = std::getenv("STRATA_DF2_HEAD_IL_ROWS") != nullptr;
        for (int c0 = 0; c0 < total_rows; c0 += head_chunk) {
            const int nc = std::min(head_chunk, total_rows - c0);
            const bool use_il = !skip_part("head") && df2_head_use_il(head_type_, nc, (int) n_dvocab_, head_il);
            if (!(q81_prepacked && c0 == 0)) {
                if (use_il && nc >= 2 && nc <= 8)
                    native_quantize_q8_1_il(fn_batch_ + (size_t) c0 * N_, xq_, xil_, N_, nc, cs);
                else {
                    native_quantize_q8_1(fn_batch_ + (size_t) c0 * N_, xq_, N_, nc, cs);
                    if (use_il && nc > 8)
                        native_q8_1_interleave(xq_, xil_, N_, nc, cs);
                }
            }
            if (!skip_part("head")) {
                if (use_il) {
                    const bool force_rows = force_il_rows_env || !native_mmvq_il_supported(head_type_, nc, (int) n_dvocab_);
                    if (force_rows) native_mmvq_il_tune(df2_head_il_rows());
                    native_mmvq_il(head_type_, head_w_, xq_, xil_, logits_, N_, (int) n_dvocab_, nc, cs);
                    if (force_rows) native_mmvq_il_tune(0);
                } else {
                    native_mmvq(head_type_, head_w_, xq_, logits_, N_, (int) n_dvocab_, nc, cs);
                }
            }
            df2_topk(logits_, (int) n_dvocab_, dvocab_, nc, topscratch_,
                     topval_ + (size_t) c0 * 16, toptok_ + (size_t) c0 * 16, cs);
        }
        df2_select_multi(hsel_batch_, topval_, toptok_,
                         reinterpret_cast<const uint16_t*>(blob_ + find("candidate_selector.predecessor_codebook").off),
                         reinterpret_cast<const uint16_t*>(blob_ + find("candidate_selector.successor_codebook").off),
                         rank_, m_bmeta_, n_items, out_btok_, out_bprob_, cs,
                         find("candidate_selector.predecessor_codebook").dtype == "q4_0", df2_beam());
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

bool Dflash2Drafter::capture_batch_body(const BatchSlotItem* items, int n_items, cudaGraphExec_t& exec_out, std::string& err) {
    if (n_items < 2 || n_items > kBatchMaxSlots) return false;
    BatchBodyGraphKey key{};
    key.n_items = n_items;
    for (int idx = 0; idx < n_items; ++idx) {
        key.slots[idx] = items[idx].slot;
        key.taps[idx] = items[idx].taps;
        key.n_new[idx] = items[idx].n_new;
    }
    ++batch_graph_tick_;
    for (BatchBodyGraphEntry& e : batch_body_graphs_) {
        if (e.key == key) {
            e.last_use = batch_graph_tick_;
            exec_out = e.exec;
            return true;
        }
    }
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "dflash2: begin batch body capture"; return false; }
    kernels::copy_i32_from_mapped(d_meta_ + kMetaCtx, m_meta_ + kMetaCtx, kMetaSlotInfo - kMetaCtx, cs_);
    const bool ok = record_batch_body(items, n_items, cs_, err);
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) { if (graph) cudaGraphDestroy(graph); return false; }
    cudaGraphExec_t new_exec = nullptr;
    if (ce != cudaSuccess || cudaGraphInstantiate(&new_exec, graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("dflash2: capture_batch_body: ") + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    cudaGraphUpload(new_exec, cs_);
    cudaStreamSynchronize(cs_);
    if (batch_body_graphs_.size() >= kBatchGraphCacheMax) {
        auto victim = std::min_element(batch_body_graphs_.begin(), batch_body_graphs_.end(),
                                       [](const BatchBodyGraphEntry& a, const BatchBodyGraphEntry& b) { return a.last_use < b.last_use; });
        if (victim->exec) cudaGraphExecDestroy(victim->exec);
        *victim = BatchBodyGraphEntry{key, new_exec, batch_graph_tick_};
    } else {
        batch_body_graphs_.push_back(BatchBodyGraphEntry{key, new_exec, batch_graph_tick_});
    }
    exec_out = new_exec;
    return true;
}

bool Dflash2Drafter::capture_batch_full(const BatchSlotItem* items, int n_items, int total_rows, cudaGraphExec_t& exec_out, std::string& err) {
    if (n_items < 1 || n_items > kBatchMaxSlots || total_rows < 1 || total_rows > n_items * kDrafts) return false;
    BatchFullGraphKey key{};
    key.n_items = n_items;
    key.total_rows = total_rows;
    for (int idx = 0; idx < n_items; ++idx) key.slots[idx] = items[idx].slot;
    ++batch_graph_tick_;
    for (BatchFullGraphEntry& e : batch_full_graphs_) {
        if (e.key == key) {
            e.last_use = batch_graph_tick_;
            exec_out = e.exec;
            return true;
        }
    }
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "dflash2: begin batch full capture"; return false; }
    kernels::copy_i32_from_mapped(d_meta_ + kMetaBlkPos, m_meta_ + kMetaBlkPos, kMetaPrecompPos - kMetaBlkPos, cs_);
    const bool ok = record_batch_body_precomp(items, n_items, cs_, err, total_rows) &&
                    record_batch_head(n_items, total_rows, cs_, err, /*q81_prepacked=*/!skip_part("head"));
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) { if (graph) cudaGraphDestroy(graph); return false; }
    cudaGraphExec_t new_exec = nullptr;
    if (ce != cudaSuccess || cudaGraphInstantiate(&new_exec, graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("dflash2: capture_batch_full: ") + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    cudaGraphUpload(new_exec, cs_);
    cudaStreamSynchronize(cs_);
    if (batch_full_graphs_.size() >= kBatchGraphCacheMax) {
        auto victim = std::min_element(batch_full_graphs_.begin(), batch_full_graphs_.end(),
                                       [](const BatchFullGraphEntry& a, const BatchFullGraphEntry& b) { return a.last_use < b.last_use; });
        if (victim->exec) cudaGraphExecDestroy(victim->exec);
        *victim = BatchFullGraphEntry{key, new_exec, batch_graph_tick_};
    } else {
        batch_full_graphs_.push_back(BatchFullGraphEntry{key, new_exec, batch_graph_tick_});
    }
    exec_out = new_exec;
    return true;
}

bool Dflash2Drafter::draft_batch_slots_launch(const BatchSlotItem* items, int n_items, std::string& err) {
    if (!bound_ || blob_shared_) { err = "dflash2: draft_batch_slots requires bound parent drafter"; return false; }
    if (n_items == 0) return true;
    if (items == nullptr || n_items < 0 || n_items > kBatchMaxSlots) { err = "dflash2: invalid batch slot items"; return false; }
    if (!context_wait(err)) return false;
    const OnDevice on_device(device_);
    const Clock::time_point t0 = Clock::now();
    static const bool batch_body_on = [] {
        const char* e = std::getenv("STRATA_DF2_BATCHBODY");
        return !e || !e[0] || e[0] != '0';
    }();
    const bool use_batch_body = batch_body_on && (n_items >= 2 || (n_items == 1 && precomp_taps_ != nullptr && precomp_rows_ > 0)) && rt_bctx_ != nullptr;

    int precomp_offs[kBatchMaxSlots] = {};
    bool can_use_precomp = use_batch_body && precomp_taps_ != nullptr && precomp_rows_ > 0;
    int total_rows = 0;
    for (int idx = 0; idx < n_items; ++idx) {
        const BatchSlotItem& it = items[idx];
        if (it.slot == nullptr || it.slot->device_ != device_ || !it.slot->bound_) {
            err = "dflash2: batch slot drafter is invalid or on another device";
            return false;
        }
        if (it.n_new < 1 || it.n_new > kRows) { err = "dflash2: batch draft rows out of range"; return false; }
        if (it.k_want < 1 || it.k_want > it.slot->ndraft_) { err = "dflash2: batch k_want out of range"; return false; }
        Dflash2Drafter& sl = *it.slot;
        if (!sl.context_wait(err)) return false;
        if (can_use_precomp && !has_precomputed_context(it.taps, it.n_new, it.pos0, &precomp_offs[idx])) {
            can_use_precomp = false;
        }
        if (!use_batch_body) {
            if (sl.graph_taps_ != nullptr && sl.graph_taps_ != it.taps) {
                err = "dflash2: the taps buffer moved after capture";
                return false;
            }
            if (!prof_ && !sl.body_exec_[it.n_new]) {
                for (int i = 0; i < 8; ++i) sl.h_meta_[i] = (int32_t) (it.pos0 + i);
                if (!sl.capture_body(it.n_new, it.taps, err)) return false;
                sl.graph_taps_ = it.taps;
            }
        }
        total_rows += it.k_want;
    }

    int row_off = 0;
    int c0_meta = 0;
    for (int idx = 0; idx < n_items; ++idx) {
        const BatchSlotItem& it = items[idx];
        Dflash2Drafter& sl = *it.slot;
        const int64_t s = it.pos0 + it.n_new;
        if (use_batch_body) {
            const int c0_slot = can_use_precomp ? precomp_offs[idx] : c0_meta;
            if (!can_use_precomp) {
                for (int i = 0; i < it.n_new; ++i) h_meta_[kMetaCtx + c0_meta + i] = (int32_t) (it.pos0 + i);
            }
            c0_meta += it.n_new;
            for (int i = 0; i < 8; ++i) h_meta_[kMetaBlkPos + idx * 8 + i] = (int32_t) (s + i);
            h_meta_[kMetaBlkTok + idx * 8 + 0] = it.anchor;
            for (int i = 1; i < 8; ++i) h_meta_[kMetaBlkTok + idx * 8 + i] = sl.mask_token_;
            h_meta_[kMetaAnchor + idx] = (int32_t) s;
            h_meta_[kMetaEpoch + idx] = sl.epoch_;
            h_meta_[kMetaSlotInfo + idx * 4 + 0] = idx;
            h_meta_[kMetaSlotInfo + idx * 4 + 1] = c0_slot;
            h_meta_[kMetaSlotInfo + idx * 4 + 2] = it.n_new;
            h_meta_[kMetaSlotInfo + idx * 4 + 3] = (int32_t) s;
        } else {
            for (int i = 0; i < 8; ++i) {
                sl.h_meta_[i] = (int32_t) (it.pos0 + std::min(i, it.n_new - 1));
                sl.h_meta_[8 + i] = (int32_t) (s + i);
            }
            sl.h_meta_[16] = it.anchor;
            sl.h_meta_[17] = (int32_t) s;
            sl.h_meta_[18] = sl.epoch_;
            sl.h_meta_[24] = it.anchor;
            for (int i = 1; i < 8; ++i) sl.h_meta_[24 + i] = sl.mask_token_;
        }
        h_bmeta_[idx * 4 + 0] = row_off;
        h_bmeta_[idx * 4 + 1] = it.k_want;
        h_bmeta_[idx * 4 + 2] = it.anchor;
        h_bmeta_[idx * 4 + 3] = 0;
        for (int i = 0; i < it.k_want; ++i) ((volatile int32_t*) h_bout_)[idx * 8 + i] = -1;
        row_off += it.k_want;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);

    if (timing_) cudaEventRecord(ev_a_, cs_);
    bool head_recorded_in_graph = false;
    pend_precomp_logs_.clear();
    if (use_batch_body) {
        if (can_use_precomp) {
            if (!prof_) {
                cudaGraphExec_t g_exec = nullptr;
                if (!capture_batch_full(items, n_items, total_rows, g_exec, err)) { clear_precomputed_context(); return false; }
                clear_precomputed_context();
                if (cudaGraphLaunch(g_exec, cs_) != cudaSuccess) {
                    err = std::string("dflash2 draft_batch_slots (batch_full): ") + cudaGetErrorString(cudaGetLastError());
                    return false;
                }
                head_recorded_in_graph = true;
                (void) cudaStreamQuery(cs_);
            } else {
                clear_precomputed_context();
                kernels::copy_i32_from_mapped(d_meta_ + kMetaBlkPos, m_meta_ + kMetaBlkPos, kMetaPrecompPos - kMetaBlkPos, cs_);
                if (!record_batch_body_precomp(items, n_items, cs_, err)) return false;
            }
        } else if (!prof_ && n_items >= 2) {
            clear_precomputed_context();
            cudaGraphExec_t g_exec = nullptr;
            if (!capture_batch_body(items, n_items, g_exec, err)) return false;
            if (cudaGraphLaunch(g_exec, cs_) != cudaSuccess) {
                err = std::string("dflash2 draft_batch_slots (batch_body): ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
        } else {
            clear_precomputed_context();
            kernels::copy_i32_from_mapped(d_meta_ + kMetaCtx, m_meta_ + kMetaCtx, kMetaSlotInfo - kMetaCtx, cs_);
            if (!record_batch_body(items, n_items, cs_, err)) return false;
        }
        for (int idx = 0; idx < n_items; ++idx) {
            const BatchSlotItem& it = items[idx];
            Dflash2Drafter& sl = *it.slot;
            if (can_use_precomp && sl.log_on_) {
                const int c0 = precomp_offs[idx];
                if (log_on_ && h_logq_ != nullptr && h_logd_ != nullptr) {
                    // Defer ev_join_ sync + ring log flush to draft_batch_slots_finish() so launch() stays 100% async.
                    sl.log_stage(it.pos0, it.n_new);
                    pend_precomp_logs_.push_back(PendingPrecompLog{&sl, it.pos0, it.n_new, c0});
                } else {
                    cudaMemcpyAsync(sl.h_logq_, xqc_ + (size_t) c0 * N_, (size_t) it.n_new * N_, cudaMemcpyDeviceToHost, cs_);
                    cudaMemcpyAsync(sl.h_logd_, xdc_ + (size_t) c0 * (N_ / 32), (size_t) it.n_new * (N_ / 32) * 4, cudaMemcpyDeviceToHost, cs_);
                    sl.log_stage(it.pos0, it.n_new);
                }
            } else {
                sl.log_stage(it.pos0, it.n_new);
            }
            sl.launch_t0_ = t0;
            sl.launch_n_ = it.n_new;
            sl.launch_pos0_ = it.pos0;
            sl.launched_ = true;
        }
    } else {
        clear_precomputed_context();
        row_off = 0;
        for (int idx = 0; idx < n_items; ++idx) {
            const BatchSlotItem& it = items[idx];
            Dflash2Drafter& sl = *it.slot;
            sl.log_stage(it.pos0, it.n_new);
            if (prof_) {
                kernels::copy_i32_from_mapped(sl.d_meta_, sl.m_meta_, 32, sl.cs_);
                if (!sl.record_context(it.taps, it.n_new, sl.cs_, err) || !sl.record_block_body(sl.cs_, err)) return false;
            } else if (cudaGraphLaunch(sl.body_exec_[it.n_new], sl.cs_) != cudaSuccess) {
                err = std::string("dflash2 draft_batch_slots: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            try {
                kernels::df2_pack_head_rows(sl.fn_ + N_, sl.hsel_, fn_batch_ + (size_t) row_off * N_,
                                            hsel_batch_ + (size_t) row_off * rank_, it.k_want, N_, rank_, sl.cs_);
            } catch (const std::exception& e) {
                err = e.what();
                return false;
            }
            cudaEventRecord(sl.ev_join_, sl.cs_);
            cudaStreamWaitEvent(cs_, sl.ev_join_, 0);
            sl.launch_t0_ = t0;
            sl.launch_n_ = it.n_new;
            sl.launch_pos0_ = it.pos0;
            sl.launched_ = true;
            row_off += it.k_want;
        }
    }

    if (!head_recorded_in_graph && !record_batch_head(n_items, total_rows, cs_, err)) return false;
    if (timing_) cudaEventRecord(ev_b_, cs_);
    (void) cudaStreamQuery(cs_);
    launch_t0_ = t0;
    launched_ = true;
    return true;
}

bool Dflash2Drafter::draft_batch_slots_finish(const BatchSlotItem* items, int n_items, std::string& err) {
    if (n_items == 0) return true;
    if (!launched_) { err = "dflash2: draft_batch_slots_finish without launch"; return false; }
    launched_ = false;
    const OnDevice on_device(device_);
    const Clock::time_point t0 = launch_t0_;
    static const bool solopipe = [] {
        const char* e = std::getenv("STRATA_DF2_SOLOPIPE");
        return !e || e[0] != '0';
    }();
    if (solopipe && !timing_ && !prof_) {
        uint32_t spins = 0;
        for (;;) {
            bool done = true;
            for (int idx = 0; idx < n_items; ++idx) {
                const int kw = items[idx].k_want;
                if (kw > 0 && ((volatile int32_t*) h_bout_)[idx * 8 + kw - 1] < 0) { done = false; break; }
            }
            if (done) break;
            strata_cpu_pause();
            if ((++spins & 1023u) == 0 && cudaStreamQuery(cs_) != cudaErrorNotReady) break;
        }
        std::atomic_thread_fence(std::memory_order_acquire);
    }
    bool all_ready = true;
    for (int idx = 0; idx < n_items; ++idx) {
        const int kw = items[idx].k_want;
        if (kw > 0 && ((volatile int32_t*) h_bout_)[idx * 8 + kw - 1] < 0) all_ready = false;
    }
    if (!solopipe || timing_ || prof_ || !all_ready) {
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("dflash2 draft_batch_slots: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
    }
    if (!pend_precomp_logs_.empty()) {
        if (ev_join_ != nullptr) cudaEventSynchronize(ev_join_);
        for (const PendingPrecompLog& pl : pend_precomp_logs_) {
            pl.slot->log_flush(pl.off, h_logq_, h_logd_);
        }
        pend_precomp_logs_.clear();
    }
    for (int idx = 0; idx < n_items; ++idx) {
        const BatchSlotItem& it = items[idx];
        Dflash2Drafter& sl = *it.slot;
        sl.launched_ = false;
        sl.log_flush();
        for (int i = 0; i < kDrafts; ++i) {
            const bool made = i < it.k_want;
            it.drafts[i] = made ? ((volatile int32_t*) h_bout_)[idx * 8 + i] : 0;
            if (it.probs) it.probs[i] = made ? ((volatile float*) h_bout_)[64 + idx * 8 + i] : 0.0f;
        }
        const int64_t s = it.pos0 + it.n_new;
        sl.have_ctx_ = std::max<int64_t>(sl.have_ctx_, s);
        sl.ring_hi_ = std::max<int64_t>(sl.ring_hi_, s);
        ++sl.rounds;
    }
    if (timing_) {
        float ms = 0;
        if (cudaEventElapsedTime(&ms, ev_a_, ev_b_) == cudaSuccess) ms_gpu += ms;
    }
    ms_draft += ms_since(t0);
    rounds += n_items;
    return true;
}

bool Dflash2Drafter::debug_ring(std::vector<float>& kv, std::vector<int64_t>& tags, std::string& err) {
    const OnDevice on_device(device_);
    if (!context_wait(err)) return false;
    const size_t per = (size_t) kRing * NKV_ * HD_;
    kv.resize((size_t) L_ * 2 * per);
    tags.resize((size_t) kRing);
    if (cudaDeviceSynchronize() != cudaSuccess) { err = "dflash2: debug ring sync"; return false; }
    std::vector<uint16_t> raw;
    const size_t elem = ring_elem_bytes();
    if (elem == 2) raw.resize(per);
    for (int l = 0; l < L_; ++l) {
        if (elem == 2) {
            for (int kvh = 0; kvh < 2; ++kvh) {
                void* rng = kvh ? vring_[l] : kring_[l];
                if (cudaMemcpy(raw.data(), rng, per * 2, cudaMemcpyDeviceToHost) != cudaSuccess) { err = "dflash2: debug ring copy"; return false; }
                float* d = kv.data() + ((size_t) l * 2 + kvh) * per;
                for (size_t i = 0; i < per; ++i) d[i] = strata::kernels::f32_from_f16(raw[i]);
            }
        } else {
            if (cudaMemcpy(kv.data() + ((size_t) l * 2) * per, kring_[l], per * 4, cudaMemcpyDeviceToHost) != cudaSuccess ||
                cudaMemcpy(kv.data() + ((size_t) l * 2 + 1) * per, vring_[l], per * 4, cudaMemcpyDeviceToHost) != cudaSuccess) { err = "dflash2: debug ring copy"; return false; }
        }
    }
    if (cudaMemcpy(tags.data(), tag_, (size_t) kRing * 8, cudaMemcpyDeviceToHost) != cudaSuccess) { err = "dflash2: debug tag copy"; return false; }
    return true;
}

bool Dflash2Drafter::debug_rows(std::vector<float>& rows, std::string& err) {
    rows.resize((size_t) kRows * (size_t) N_);
    if (cudaMemcpy(rows.data(), fn_, rows.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess) { err = "dflash2: debug copy"; return false; }
    return true;
}

}  // namespace strata::core
