// DFlash2 equations, independently expressed as direct CUDA/HIP kernels.
// Reference: llama.cpp src/models/dflash.cpp (a9d27ac), z-lab/dflash (MIT, 07ebd93).
#include "strata/kernels/dflash2.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace strata::kernels {
namespace {
__global__ void conv_kernel(const float* x, const float* dynamic, const float* base,
                            float* out, uint16_t* out_bf16, float* accum,
                            int rows, int hidden, int kernel, int group, int side, int block_rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * hidden) return;
    const int row = i / hidden, channel = i % hidden, groups = hidden / group;
    float value = 0;
    const int limit = min(kernel, row % block_rows + 1);
    for (int tap = 0; tap < limit; ++tap) {
        const float weight = base[(side * kernel + tap) * hidden + channel] +
            dynamic[((row * 2 + side) * kernel + tap) * groups + channel / group];
        value += weight * x[(row - tap) * hidden + channel];
    }
    if (out) out[i] = value;
    if (out_bf16) out_bf16[i] = bf16_from_f32(value);
    if (accum) accum[i] += value;
}
// Retained for same-input timing and parity (STRATA_DFLASH2_SERIAL_TOPK=1).
__global__ void topk_kernel(const float* logits, int32_t* ids, float* unary, int vocab, int top_k) {
    if (threadIdx.x != 0) return;
    const int row = blockIdx.x;
    ids += row * top_k; unary += row * top_k; logits += (size_t) row * vocab;
    for (int k = 0; k < top_k; ++k) { ids[k] = -1; unary[k] = -INFINITY; }
    for (int token = 0; token < vocab; ++token) {
        const float score = logits[token];
        // Non-finite model outputs must not manufacture a token ID.
        if (!isfinite(score)) continue;
        int at = top_k;
        while (at > 0 && score > unary[at - 1]) --at;
        if (at == top_k) continue;
        for (int k = top_k - 1; k > at; --k) { ids[k] = ids[k - 1]; unary[k] = unary[k - 1]; }
        ids[at] = token; unary[at] = score;
    }
}
// One CTA per row/tile. A second pass merges the tiles' winners using the same
// reduction; only tile_count * K candidates need temporary storage. IDs break ties.
// When tile_width <= kTopkTile (both pass 1 and pass 2 of tiled top-k), each of the
// 256 threads owns at most 16 elements and caches them in registers so global memory
// is read once per tile instead of top_k times.
constexpr int kTopkTile = 4096;
__global__ void topk_parallel_kernel(const float* logits, const int32_t* candidates, int32_t* ids, float* unary,
                                    int vocab, int top_k, int tile_width, int tiles) {
    const int row = blockIdx.x, tile = blockIdx.y, t = threadIdx.x;
    logits += (size_t)row * vocab;
    if (candidates) candidates += (size_t) row * vocab;
    ids += ((size_t) row * tiles + tile) * top_k;
    unary += ((size_t) row * tiles + tile) * top_k;
    const int begin = tile * tile_width, end = begin + min(vocab - begin, tile_width);
    __shared__ float score[256];
    __shared__ int token[256];
    float upper = INFINITY;
    int upper_id = -1;
    if (end - begin <= kTopkTile) {
        float reg_v[16];
        int reg_c[16];
        int reg_n = 0;
        for (int64_t i = (int64_t) begin + t; i < end; i += 256) {
            reg_v[reg_n] = logits[i];
            reg_c[reg_n] = candidates ? candidates[i] : (int) i;
            ++reg_n;
        }
        for (int k = 0; k < top_k; ++k) {
            float best = -INFINITY;
            int id = -1;
            for (int m = 0; m < reg_n; ++m) {
                const float v = reg_v[m];
                const int candidate = reg_c[m];
                if (candidate < 0 || !isfinite(v) || v > upper || (v == upper && candidate <= upper_id)) continue;
                if (id < 0 || v > best || (v == best && candidate < id)) { best = v; id = candidate; }
            }
            score[t] = best; token[t] = id;
            __syncthreads();
            for (int offset = 128; offset; offset >>= 1) {
                if (t < offset) {
                    const int other = token[t + offset];
                    const float value = score[t + offset];
                    if (other >= 0 && (token[t] < 0 || value > score[t] ||
                                      (value == score[t] && other < token[t]))) {
                        score[t] = value; token[t] = other;
                    }
                }
                __syncthreads();
            }
            upper = score[0]; upper_id = token[0];
            if (!t) { ids[k] = upper_id; unary[k] = upper; }
            __syncthreads();
        }
        return;
    }
    for (int k = 0; k < top_k; ++k) {
        float best = -INFINITY;
        int id = -1;
        for (int64_t i = (int64_t) begin + t; i < end; i += 256) {
            const float v = logits[i];
            const int candidate = candidates ? candidates[i] : i;
            if (candidate < 0 || !isfinite(v) || v > upper || (v == upper && candidate <= upper_id)) continue;
            if (id < 0 || v > best || (v == best && candidate < id)) { best = v; id = candidate; }
        }
        score[t] = best; token[t] = id;
        __syncthreads();
        for (int offset = 128; offset; offset >>= 1) {
            if (t < offset) {
                const int other = token[t + offset];
                const float value = score[t + offset];
                if (other >= 0 && (token[t] < 0 || value > score[t] ||
                                  (value == score[t] && other < token[t]))) {
                    score[t] = value; token[t] = other;
                }
            }
            __syncthreads();
        }
        upper = score[0]; upper_id = token[0];
        if (!t) { ids[k] = upper_id; unary[k] = upper; }
        __syncthreads();
    }
}
__device__ float codebook_value(const float* p, size_t i) { return p[i]; }
__device__ float codebook_value(const uint16_t* p, size_t i) { return f32_from_bf16(p[i]); }

// Philox 4x32-10 matching sampler.cu for position-keyed draft sampling (counter | 1ull<<61).
__device__ __forceinline__ float df_philox_uniform(uint64_t seed, uint64_t counter) {
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32);
    uint32_t c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    for (int i = 0; i < 10; ++i) {
        const uint32_t hi0 = __umulhi(0x9E3779B9u, c0);
        const uint32_t hi1 = __umulhi(0xBB67AE85u, c2);
        const uint32_t lo0 = 0x9E3779B9u * c0;
        const uint32_t lo1 = 0xBB67AE85u * c2;
        const uint32_t n0 = hi1 ^ c1 ^ (uint32_t) i;
        const uint32_t n1 = lo1;
        const uint32_t n2 = hi0 ^ c3;
        const uint32_t n3 = lo0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    }
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}

template<class T>
__global__ void select_kernel(const int32_t* ids, const float* unary, const float* hidden,
                              const T* prev, const T* next, const int32_t* anchor,
                              int32_t* out, float* confidence, int rows, int top_k, int rank) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int predecessor = anchor[0];
    float scores[64];
    bool valid = predecessor >= 0;
    for (int row = 1; row < rows; ++row) {
        int best = -1;
        bool row_valid = valid;
        for (int k = 0; k < top_k; ++k) {
            const int token = ids[row * top_k + k];
            float score = 0;
            if (!valid || token < 0) { scores[k] = -INFINITY; continue; }
            for (int j = 0; j < rank; ++j)
                score += (codebook_value(prev, (size_t) predecessor * rank + j) * hidden[row * rank + j]) *
                         codebook_value(next, (size_t) token * rank + j);
            score += unary[row * top_k + k];
            scores[k] = score;
            if (!isfinite(score)) row_valid = false;
            if (isfinite(score) && (best < 0 || score > scores[best])) best = k;
        }
        valid = row_valid && best >= 0;
        if (!valid) { out[row - 1] = -1; if (confidence) confidence[row - 1] = 0; continue; }
        predecessor = ids[row * top_k + best];
        out[row - 1] = predecessor;
        if (confidence) {
            float sum = 0;
            for (int k = 0; k < top_k; ++k) sum += expf(scores[k] - scores[best]);
            confidence[row - 1] = 1.0f / sum;
        }
    }
}

// Candidates are independent within a row. Keep each candidate's rank sum in
// its original serial order, and let lane 0 retain the original tie handling
// and confidence sum. Only the work between candidates is parallelized.
// Optionally exports the row's q distribution (kSpecQStride = 128 int32s: 64 IDs + 64 floats)
// and/or samples the draft token from q via Philox(sample_seed, (sample_pos + row - 1) | 1<<61).
__global__ void select_bf16_parallel(const int32_t* ids, const float* unary, const float* hidden,
                                    const uint16_t* prev, const uint16_t* next, const int32_t* anchor,
                                    int32_t* out, float* confidence, int32_t* q_out,
                                    int rows, int top_k, int rank,
                                    uint64_t sample_seed, int64_t sample_pos) {
    const int t = threadIdx.x;
    __shared__ float s_ph[1024];
    __shared__ float scores[64];
    __shared__ float probs[64];
    __shared__ int predecessor, valid;
    if (!t) { predecessor = anchor[0]; valid = predecessor >= 0; }
    __syncthreads();
    for (int row = 1; row < rows; ++row) {
        if (valid) {
            for (int j = t; j < rank; j += 64)
                s_ph[j] = codebook_value(prev, (size_t) predecessor * rank + j) * hidden[row * rank + j];
        }
        __syncthreads();
        if (t < top_k) {
            const int token = ids[row * top_k + t];
            float score = -INFINITY;
            if (valid && token >= 0) {
                score = 0;
                const uint16_t* next_row = next + (size_t) token * rank;
                #pragma unroll 8
                for (int j = 0; j < rank; ++j)
                    score += s_ph[j] * codebook_value(next_row, (size_t) j);
                score += unary[row * top_k + t];
            }
            scores[t] = score;
        }
        __syncthreads();
        if (!t) {
            bool row_valid = valid;
            int best = -1;
            for (int k = 0; k < top_k; ++k) {
                if (!valid || ids[row * top_k + k] < 0) continue;
                const float score = scores[k];
                if (!isfinite(score)) row_valid = false;
                if (isfinite(score) && (best < 0 || score > scores[best])) best = k;
            }
            valid = row_valid && best >= 0;
            if (!valid) {
                out[row - 1] = -1;
                if (confidence) confidence[row - 1] = 0;
            } else {
                float sum = 0;
                if (confidence || q_out || sample_pos >= 0) {
                    for (int k = 0; k < top_k; ++k) {
                        const float e = expf(scores[k] - scores[best]);
                        probs[k] = e;
                        sum += e;
                    }
                    const float inv_sum = 1.0f / sum;
                    for (int k = 0; k < top_k; ++k) probs[k] *= inv_sum;
                }
                int pick = best;
                if (sample_pos >= 0) {
                    const float u = df_philox_uniform(sample_seed, ((uint64_t)(sample_pos + row - 1)) | (1ull << 61));
                    float cum = 0.0f;
                    pick = top_k - 1;
                    for (int k = 0; k < top_k; ++k) {
                        cum += probs[k];
                        if (u < cum) { pick = k; break; }
                    }
                }
                predecessor = ids[row * top_k + pick];
                out[row - 1] = predecessor;
                if (confidence) confidence[row - 1] = (sample_pos >= 0 ? probs[pick] : 1.0f / sum);
            }
        }
        __syncthreads();
        if (q_out) {
            int32_t* qrow = q_out + (row - 1) * 128;
            float* qprob = reinterpret_cast<float*>(qrow + 64);
            if (valid && t < top_k) {
                qrow[t] = ids[row * top_k + t];
                qprob[t] = probs[t];
            } else if (t < 64) {
                qrow[t] = -1;
                qprob[t] = 0.0f;
            }
            __syncthreads();
        }
    }
}

// Beam search (B in 2..4) over the selector's conditional log-softmax chain.
__global__ void select_bf16_beam_parallel(const int32_t* ids, const float* unary, const float* hidden,
                                         const uint16_t* prev, const uint16_t* next, const int32_t* anchor,
                                         int32_t* out, float* confidence, int32_t* q_out,
                                         int rows, int top_k, int rank, int beam_width) {
    const int t = threadIdx.x;
    __shared__ float scores[64];
    __shared__ float step_probs[4][64];
    __shared__ float cand_logp[4][64];
    __shared__ int32_t beam_path[4][8];
    __shared__ float beam_conf[4][8];
    __shared__ float beam_logp[4];
    __shared__ int n_beams, valid;
    if (!t) {
        const int a0 = anchor[0];
        valid = a0 >= 0;
        n_beams = valid ? 1 : 0;
        beam_path[0][0] = a0;
        beam_logp[0] = 0.0f;
    }
    __syncthreads();
    for (int row = 1; row < rows; ++row) {
        for (int b = 0; b < n_beams; ++b) {
            const int pred = beam_path[b][row - 1];
            if (t < top_k) {
                const int token = ids[row * top_k + t];
                float score = -INFINITY;
                if (valid && pred >= 0 && token >= 0) {
                    score = 0;
                    for (int j = 0; j < rank; ++j)
                        score += (codebook_value(prev, (size_t) pred * rank + j) * hidden[row * rank + j]) *
                                 codebook_value(next, (size_t) token * rank + j);
                    score += unary[row * top_k + t];
                }
                scores[t] = score;
            }
            __syncthreads();
            if (!t) {
                bool row_valid = valid;
                int best = -1;
                for (int k = 0; k < top_k; ++k) {
                    if (!valid || ids[row * top_k + k] < 0) continue;
                    const float sc = scores[k];
                    if (!isfinite(sc)) row_valid = false;
                    if (isfinite(sc) && (best < 0 || sc > scores[best])) best = k;
                }
                if (!row_valid || best < 0) {
                    valid = 0;
                } else {
                    float sum = 0;
                    for (int k = 0; k < top_k; ++k) sum += expf(scores[k] - scores[best]);
                    const float inv_sum = 1.0f / sum;
                    for (int k = 0; k < top_k; ++k) {
                        step_probs[b][k] = expf(scores[k] - scores[best]) * inv_sum;
                        cand_logp[b][k] = beam_logp[b] + scores[k];
                    }
                }
            }
            __syncthreads();
        }
        if (!t) {
            if (!valid) {
                n_beams = 0;
            } else {
                const int max_b = min(beam_width, n_beams * top_k);
                int32_t next_path[4][8];
                float next_conf[4][8];
                float next_logp[4];
                int picked = 0;
                for (int slot = 0; slot < max_b; ++slot) {
                    int best_b = -1, best_k = -1;
                    float best_lp = -INFINITY;
                    for (int b = 0; b < n_beams; ++b) {
                        for (int k = 0; k < top_k; ++k) {
                            const float lp = cand_logp[b][k];
                            if (lp > best_lp) { best_lp = lp; best_b = b; best_k = k; }
                        }
                    }
                    if (best_b < 0) break;
                    cand_logp[best_b][best_k] = -INFINITY;
                    for (int r = 0; r < row; ++r) {
                        next_path[picked][r] = beam_path[best_b][r];
                        next_conf[picked][r] = beam_conf[best_b][r];
                    }
                    next_path[picked][row] = ids[row * top_k + best_k];
                    next_conf[picked][row] = step_probs[best_b][best_k];
                    next_logp[picked] = best_lp;
                    ++picked;
                }
                n_beams = picked;
                if (n_beams == 0) valid = 0;
                for (int b = 0; b < n_beams; ++b) {
                    for (int r = 0; r <= row; ++r) {
                        beam_path[b][r] = next_path[b][r];
                        beam_conf[b][r] = next_conf[b][r];
                    }
                    beam_logp[b] = next_logp[b];
                }
            }
        }
        __syncthreads();
    }
    if (!t) {
        for (int r = 1; r < rows; ++r) {
            out[r - 1] = valid ? beam_path[0][r] : -1;
            if (confidence) confidence[r - 1] = valid ? beam_conf[0][r] : 0.0f;
        }
    }
    if (q_out && t < 64) {
        for (int r = 1; r < rows; ++r) {
            int32_t* qrow = q_out + (r - 1) * 128;
            qrow[t] = -1;
            reinterpret_cast<float*>(qrow + 64)[t] = 0.0f;
        }
    }
}
}  // namespace

void dflash2_convolve_fused(const float* x, const float* dynamic, const float* base,
                            float* out, uint16_t* out_bf16, float* accum,
                            int rows, int hidden, int kernel, int group, int side, int block_rows, void* stream) {
    if (!x || !dynamic || !base || (!out && !out_bf16 && !accum) || rows <= 0 ||
        (int64_t)rows * hidden > INT32_MAX - 255 || hidden <= 0 || kernel <= 0 || kernel > 8 ||
        group <= 0 || hidden % group || (int64_t)rows * (hidden / group) * 2 * kernel > INT32_MAX ||
        block_rows <= 0 || block_rows > 8 || rows % block_rows || side < 0 || side > 1 ||
        x == out || dynamic == out || base == out || x == accum || dynamic == accum || base == accum)
        throw std::invalid_argument("dflash2_convolve_fused: invalid geometry or aliased output");
    conv_kernel<<<(rows * hidden + 255) / 256, 256, 0, (cudaStream_t) stream>>>(
        x, dynamic, base, out, out_bf16, accum, rows, hidden, kernel, group, side, block_rows);
}

void dflash2_convolve(const float* x, const float* dynamic, const float* base, float* out,
                     int rows, int hidden, int kernel, int group, int side, int block_rows, void* stream) {
    if (!x || !dynamic || !base || !out || rows <= 0 || (int64_t)rows * hidden > INT32_MAX - 255 || hidden <= 0 || kernel <= 0 || kernel > 8 || group <= 0 || hidden % group ||
        (int64_t)rows * (hidden / group) * 2 * kernel > INT32_MAX ||
        block_rows <= 0 || block_rows > 8 || rows % block_rows || side < 0 || side > 1 ||
        x == out || dynamic == out || base == out)
        throw std::invalid_argument("dflash2_convolve: invalid geometry or aliased output");
    conv_kernel<<<(rows * hidden + 255) / 256, 256, 0, (cudaStream_t) stream>>>(
        x, dynamic, base, out, nullptr, nullptr, rows, hidden, kernel, group, side, block_rows);
}
std::size_t dflash2_topk_scratch_bytes(int rows, int vocab, int top_k) {
    if (rows <= 0 || vocab <= 0 || (int64_t) rows * vocab > INT32_MAX ||
        top_k < 1 || top_k > 64 || top_k > vocab)
        throw std::invalid_argument("dflash2_topk: invalid workspace geometry");
    return (std::size_t) rows * ((vocab + kTopkTile - 1LL) / kTopkTile) * top_k *
           (sizeof(int32_t) + sizeof(float));
}
void dflash2_topk(const float* logits, int32_t* ids, float* unary, int rows, int vocab, int top_k, void* stream,
                  void* scratch, std::size_t scratch_bytes) {
    if (!logits || !ids || !unary || rows <= 0 || (int64_t)rows * vocab > INT32_MAX || top_k < 1 || top_k > 64 || top_k > vocab)
        throw std::invalid_argument("dflash2_topk: invalid geometry");
    if (scratch && scratch_bytes < dflash2_topk_scratch_bytes(rows, vocab, top_k))
        throw std::invalid_argument("dflash2_topk: insufficient workspace");
    static const bool serial = [] {
        const char* value = std::getenv("STRATA_DFLASH2_SERIAL_TOPK");
        return value && value[0] == '1';
    }();
    static const bool tiled = [] {
        const char* value = std::getenv("STRATA_DFLASH2_TILED_TOPK");
        return !value || value[0] != '0';
    }();
    if (serial) topk_kernel<<<rows, 1, 0, (cudaStream_t) stream>>>(logits, ids, unary, vocab, top_k);
    else if (tiled && scratch && vocab > kTopkTile && rows <= 65535 &&
             (vocab + kTopkTile - 1LL) / kTopkTile <= 65535) {
        const int tiles = (vocab + kTopkTile - 1LL) / kTopkTile;
        int32_t* local_ids = (int32_t*) scratch;
        float* local_scores = (float*) (local_ids + (size_t) rows * tiles * top_k);
        topk_parallel_kernel<<<dim3(rows, tiles), 256, 0, (cudaStream_t) stream>>>(
            logits, nullptr, local_ids, local_scores, vocab, top_k, kTopkTile, tiles);
        topk_parallel_kernel<<<rows, 256, 0, (cudaStream_t) stream>>>(
            local_scores, local_ids, ids, unary, tiles * top_k, top_k, tiles * top_k, 1);
    } else topk_parallel_kernel<<<rows, 256, 0, (cudaStream_t) stream>>>(
        logits, nullptr, ids, unary, vocab, top_k, vocab, 1);
}
void dflash2_select(const int32_t* ids, const float* unary, const float* hidden,
                    const float* prev, const float* next, const int32_t* anchor,
                    int32_t* out, float* confidence, int rows, int top_k, int rank, void* stream) {
    if (!ids || !unary || !hidden || !prev || !next || !anchor || !out ||
        rows < 2 || rows > 8 || top_k < 1 || top_k > 64 || rank < 1 || rank > 1024)
        throw std::invalid_argument("dflash2_select: invalid geometry");
    select_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(
        ids, unary, hidden, prev, next, anchor, out, confidence, rows, top_k, rank);
}
void dflash2_select_bf16_ex(const int32_t* ids, const float* unary, const float* hidden,
                            const uint16_t* prev, const uint16_t* next, const int32_t* anchor,
                            int32_t* out, float* confidence, int32_t* q_out,
                            int rows, int top_k, int rank, int beam_width,
                            uint64_t sample_seed, int64_t sample_pos,
                            void* stream, bool reference) {
    if (!ids || !unary || !hidden || !prev || !next || !anchor || !out ||
        rows < 2 || rows > 8 || top_k < 1 || top_k > 64 || rank < 1 || rank > 1024)
        throw std::invalid_argument("dflash2_select_bf16: invalid geometry");
    static const bool serial = [] {
        const char* value = std::getenv("STRATA_DFLASH2_SERIAL_SELECT");
        return value && value[0] == '1';
    }();
    static const int env_beam = [] {
        const char* value = std::getenv("STRATA_DFLASH2_BEAM");
        return value ? std::atoi(value) : 1;
    }();
    const int eff_beam = beam_width > 1 ? beam_width : env_beam;
    if ((serial || reference) && !q_out && sample_pos < 0 && eff_beam <= 1) {
        select_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(
            ids, unary, hidden, prev, next, anchor, out, confidence, rows, top_k, rank);
    } else if (!reference && eff_beam >= 2 && sample_pos < 0) {
        select_bf16_beam_parallel<<<1, 64, 0, (cudaStream_t)stream>>>(
            ids, unary, hidden, prev, next, anchor, out, confidence, q_out,
            rows, top_k, rank, std::min(eff_beam, 4));
    } else {
        select_bf16_parallel<<<1, 64, 0, (cudaStream_t)stream>>>(
            ids, unary, hidden, prev, next, anchor, out, confidence, q_out,
            rows, top_k, rank, sample_seed, sample_pos);
    }
}
void dflash2_select_bf16(const int32_t* ids, const float* unary, const float* hidden,
                         const uint16_t* prev, const uint16_t* next, const int32_t* anchor,
                         int32_t* out, float* confidence, int rows, int top_k, int rank, void* stream, bool reference) {
    dflash2_select_bf16_ex(ids, unary, hidden, prev, next, anchor, out, confidence, nullptr,
                           rows, top_k, rank, 1, 0, -1, stream, reference);
}
namespace {
__global__ void hc_mean_kernel(const float* residual, float* out, uint16_t* out_bf16,
                               int rows, int hidden, int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * hidden) return;
    const int row = i / hidden, c = i % hidden;
    float sum = 0;
    for (int h = 0; h < hc; ++h) sum += residual[(row * hc + h) * hidden + c];
    sum /= hc;
    if (out) out[i] = sum;
    if (out_bf16) out_bf16[i] = bf16_from_f32(sum);
}
template<bool Canonical, bool Paired>
__global__ void attention128_kernel(const float* q, const uint16_t* k, const uint16_t* v,
                                    const int32_t* table, const int32_t* steps, float* scratch,
                                    int rows, int heads, int kv_heads, int page_size, int chunks, int window) {
    const int ch = blockIdx.x, head = blockIdx.y, row = blockIdx.z, t = threadIdx.x;
    const int lane = t & 31, warp = t / 32;
    const int end = steps[row * 4 + 1], position = end - rows + row;
    const int first = window ? max(0, end - rows - window + 1) : 0;
    const int begin = first + ch * 64;
    __shared__ float scores[64];
    __shared__ float probabilities[64];
    __shared__ size_t poolrows[64];
    const int kv_head = Canonical ? head >> 2 : head / (heads / kv_heads);
    const float* q_row = q + (row * heads + head) * 128;
    const float q0 = q_row[lane];
    const float q1 = q_row[lane + 32];
    const float q2 = q_row[lane + 64];
    const float q3 = q_row[lane + 96];
    for (int c = warp; c < 64; c += (Paired ? 2 : 4)) {
        const int cell = begin + c;
        float dot = 0;
        const bool visible = cell < end && (!window || abs(cell - position) < window);
        size_t poolrow = 0;
        if (visible) {
            const int page = Canonical ? cell >> 6 : cell / page_size;
            const int offset = Canonical ? cell & 63 : cell % page_size;
            poolrow = ((size_t)table[page] * (Canonical ? 8 : kv_heads) + kv_head) *
                      (Canonical ? 64 : page_size) + offset;
            const uint16_t* k_row = k + poolrow * 128;
            dot += q0 * __half2float(__ushort_as_half(k_row[lane]));
            dot += q1 * __half2float(__ushort_as_half(k_row[lane + 32]));
            dot += q2 * __half2float(__ushort_as_half(k_row[lane + 64]));
            dot += q3 * __half2float(__ushort_as_half(k_row[lane + 96]));
            for (int offset = 16; offset; offset >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, offset);
        }
        if (!lane) {
            scores[c] = visible ? dot * 0.08838834764831845f : -INFINITY;
            poolrows[c] = poolrow;
        }
    }
    __syncthreads();
    float m = -INFINITY;
    for (int c = 0; c < 64; ++c) m = fmaxf(m, scores[c]);
    // Value components share their softmax weights and the address staged
    // by QK. Compute expf once per cell, keeping the original ordered sums
    // and masked-cell skip.
    if (t < 64 && scores[t] != -INFINITY) {
        probabilities[t] = expf(scores[t] - m);
    }
    __syncthreads();
    float sum = 0, acc = 0, acc1 = 0;
    for (int c = 0; c < 64; ++c) {
        if (scores[c] == -INFINITY) continue;
        const float p = probabilities[c];
        const size_t poolrow = poolrows[c];
        sum += p;
        if constexpr (Paired) {
            const float2 value = __half22float2(reinterpret_cast<const __half2*>(v + poolrow * 128)[t]);
            acc += p * value.x;
            acc1 += p * value.y;
        } else {
            acc += p * __half2float(__ushort_as_half(v[poolrow * 128 + t]));
        }
    }
    float* dst = scratch + ((row * heads + head) * chunks + ch) * 130;
    if constexpr (Paired) { dst[2 * t] = acc; dst[2 * t + 1] = acc1; }
    else dst[t] = acc;
    if (!t) { dst[128] = m; dst[129] = sum; }
}
// Group-4 GQA kernel for the official 32Q/8KV x 128, page_size=64 geometry:
// one CTA per (chunk, kv_head, row) serves all 4 Q heads in the KV group simultaneously.
// All 4 warps cooperate on the 64 K cells (loading each K cell once into registers for all 4 Q heads),
// then each warp accumulates V for one Q head in parallel, with only 2.6 KB shared memory.
__global__ void __launch_bounds__(128)
attention128_group4_kernel(const float* q, const uint16_t* k, const uint16_t* v,
                           const int32_t* table, const int32_t* steps, float* scratch,
                           int rows, int chunks, int window) {
    const int ch = blockIdx.x, kv_head = blockIdx.y, row = blockIdx.z, t = threadIdx.x;
    const int lane = t & 31, warp = t >> 5;
    const int end = steps[row * 4 + 1], position = end - rows + row;
    const int first = window ? max(0, end - rows - window + 1) : 0;
    const int begin = first + ch * 64;

    __shared__ float scores[4][64];
    __shared__ float probabilities[4][64];
    __shared__ size_t poolrows[64];
    __shared__ float s_max[4];

    float q0[4], q1[4], q2[4], q3[4];
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        const float* qr = q + (row * 32 + kv_head * 4 + g) * 128;
        q0[g] = qr[lane];
        q1[g] = qr[lane + 32];
        q2[g] = qr[lane + 64];
        q3[g] = qr[lane + 96];
    }

    for (int c = warp; c < 64; c += 4) {
        const int cell = begin + c;
        const bool visible = cell < end && (!window || abs(cell - position) < window);
        size_t poolrow = 0;
        float dot[4] = {0, 0, 0, 0};
        if (visible) {
            const int page = cell >> 6;
            const int offset = cell & 63;
            poolrow = ((size_t)table[page] * 8 + kv_head) * 64 + offset;
            const uint16_t* kr = k + poolrow * 128;
            const float k0 = __half2float(__ushort_as_half(kr[lane]));
            const float k1 = __half2float(__ushort_as_half(kr[lane + 32]));
            const float k2 = __half2float(__ushort_as_half(kr[lane + 64]));
            const float k3 = __half2float(__ushort_as_half(kr[lane + 96]));
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                float d = 0;
                d += q0[g] * k0;
                d += q1[g] * k1;
                d += q2[g] * k2;
                d += q3[g] * k3;
                for (int offset_s = 16; offset_s; offset_s >>= 1)
                    d += __shfl_xor_sync(0xffffffffu, d, offset_s);
                dot[g] = d;
            }
        }
        if (!lane) {
            poolrows[c] = poolrow;
            #pragma unroll
            for (int g = 0; g < 4; ++g)
                scores[g][c] = visible ? dot[g] * 0.08838834764831845f : -INFINITY;
        }
    }
    __syncthreads();

    const int g = warp;
    float m = -INFINITY;
    for (int c = 0; c < 64; ++c) m = fmaxf(m, scores[g][c]);
    for (int c = lane; c < 64; c += 32) {
        if (scores[g][c] != -INFINITY)
            probabilities[g][c] = expf(scores[g][c] - m);
    }
    if (!lane) s_max[g] = m;
    __syncthreads();

    float sum = 0, acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    for (int c = 0; c < 64; ++c) {
        if (scores[g][c] == -INFINITY) continue;
        const float p = probabilities[g][c];
        const size_t poolrow = poolrows[c];
        const uint16_t* vr = v + poolrow * 128;
        sum += p;
        acc0 += p * __half2float(__ushort_as_half(vr[lane]));
        acc1 += p * __half2float(__ushort_as_half(vr[lane + 32]));
        acc2 += p * __half2float(__ushort_as_half(vr[lane + 64]));
        acc3 += p * __half2float(__ushort_as_half(vr[lane + 96]));
    }
    float* dst = scratch + ((row * 32 + kv_head * 4 + g) * chunks + ch) * 130;
    dst[lane] = acc0;
    dst[lane + 32] = acc1;
    dst[lane + 64] = acc2;
    dst[lane + 96] = acc3;
    if (!lane) { dst[128] = s_max[g]; dst[129] = sum; }
}
__global__ void attention128_merge(const float* scratch, float* out, int chunks) {
    const int h = blockIdx.x, d = threadIdx.x;
    const float* src = scratch + h * chunks * 130;
    float m = -INFINITY;
    for (int c = 0; c < chunks; ++c) m = fmaxf(m, src[c * 130 + 128]);
    float sum = 0, acc = 0;
    for (int c = 0; c < chunks; ++c) {
        const float l = src[c * 130 + 129];
        if (!l) continue;
        const float scale = expf(src[c * 130 + 128] - m);
        sum += scale * l; acc += scale * src[c * 130 + d];
    }
    out[h * 128 + d] = sum ? acc / sum : 0;
}
}
void dflash2_hc_mean(const float* residual, float* out, uint16_t* out_bf16,
                     int rows, int hidden, int hc, void* stream) {
    if (!residual || (!out && !out_bf16) || rows < 1 || hidden < 1 || hc < 1 ||
        (int64_t)rows * hidden > INT32_MAX - 255 || (int64_t)rows * hidden * hc > INT32_MAX)
        throw std::invalid_argument("dflash2_hc_mean: invalid geometry or missing buffers");
    hc_mean_kernel<<<(rows * hidden + 255) / 256, 256, 0, (cudaStream_t)stream>>>(residual, out, out_bf16, rows, hidden, hc);
}
void dflash2_attention(const float* q, const uint16_t* k, const uint16_t* v, const int32_t* table,
                       const int32_t* steps, float* out, float* scratch, int rows, int heads,
                       int kv_heads, int page_size, int active_cells, int window, void* stream) {
    if (!q || !k || !v || !table || !steps || !out || !scratch || rows < 1 || rows > 8 ||
        heads < 1 || heads > 65535 || kv_heads < 1 || heads % kv_heads || page_size < 1 || active_cells < 1 || active_cells > INT32_MAX - 63 || window < 0 ||
        (int64_t)rows * heads * ((active_cells + 63LL) / 64) * 130 > INT32_MAX)
        throw std::invalid_argument("dflash2_attention: invalid geometry or missing buffers");
    const int chunks = (active_cells + 63) / 64;
    // The official geometry uses power-of-two pages and four Q heads per KV
    // head. Its addresses need shifts, not runtime integer divisions. Other
    // geometries keep the general indexing path.
    if (heads == 32 && kv_heads == 8 && page_size == 64) {
        static const bool legacy_attn = [] {
            const char* value = std::getenv("STRATA_DFLASH2_LEGACY_ATTN");
            return value && value[0] == '1';
        }();
        if (!legacy_attn) {
            attention128_group4_kernel<<<dim3(chunks, kv_heads, rows), 128, 0, (cudaStream_t)stream>>>(
                q, k, v, table, steps, scratch, rows, chunks, window);
        } else if (active_cells >= 512 && reinterpret_cast<std::uintptr_t>(v) % alignof(__half2) == 0)
            attention128_kernel<true, true><<<dim3(chunks, heads, rows), 64, 0, (cudaStream_t)stream>>>(q,k,v,table,steps,scratch,rows,heads,kv_heads,page_size,chunks,window);
        else
            attention128_kernel<true, false><<<dim3(chunks, heads, rows), 128, 0, (cudaStream_t)stream>>>(q,k,v,table,steps,scratch,rows,heads,kv_heads,page_size,chunks,window);
    } else
        attention128_kernel<false, false><<<dim3(chunks, heads, rows), 128, 0, (cudaStream_t)stream>>>(q,k,v,table,steps,scratch,rows,heads,kv_heads,page_size,chunks,window);
    attention128_merge<<<rows * heads,128,0,(cudaStream_t)stream>>>(scratch,out,chunks);
}

}  // namespace strata::kernels
