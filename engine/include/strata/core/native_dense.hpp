#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace strata::core {
class WeightTable;
struct WeightRef;

// Experimental GDN/QSA/shared-expert projection overrides. Upload unchanged native GGUF
// blocks once, then attach them to the matching canonical WeightRef. Unsupported types retain their
// canonical paths. Owns one Q8_1 scratch vector, so use one ordered session stream and keep this object alive until all
// graphs that reference it have been destroyed and synchronized.
class NativeDense {
public:
    static constexpr uint64_t kArenaAlign = 256;
    static constexpr uint64_t kAllocGranule = 2ull << 20;
    static constexpr uint64_t kSmallAlloc = 1ull << 20;

    struct ArenaPlan {
        std::vector<uint64_t> offsets;
        uint64_t payload_bytes = 0;
        uint64_t allocated_bytes = 0;
    };

    static uint64_t arena_align(uint64_t bytes) noexcept {
        return (bytes + kArenaAlign - 1) / kArenaAlign * kArenaAlign;
    }
    /// Driver backing footprint of one device allocation (`< 1 MiB` unrounded; `>= 1 MiB` rounded up once to 2 MiB granules).
    static uint64_t alloc_bytes(uint64_t bytes) noexcept {
        return bytes < kSmallAlloc ? bytes : (bytes + kAllocGranule - 1) / kAllocGranule * kAllocGranule;
    }
    static uint64_t arena_payload_bytes(const std::vector<uint64_t>& sizes) noexcept {
        uint64_t payload = 0;
        for (uint64_t sz : sizes) payload += arena_align(sz);
        return payload;
    }
    static uint64_t arena_alloc_bytes(uint64_t payload_bytes) noexcept { return alloc_bytes(payload_bytes); }
    static uint64_t arena_alloc_bytes(const std::vector<uint64_t>& sizes) noexcept {
        return alloc_bytes(arena_payload_bytes(sizes));
    }
    static ArenaPlan plan_arena(const std::vector<uint64_t>& sizes) {
        ArenaPlan p;
        p.offsets.reserve(sizes.size());
        uint64_t cursor = 0;
        for (uint64_t sz : sizes) {
            p.offsets.push_back(cursor);
            cursor += arena_align(sz);
        }
        p.payload_bytes = cursor;
        p.allocated_bytes = alloc_bytes(cursor);
        return p;
    }

    NativeDense() = default;
    ~NativeDense();
    NativeDense(const NativeDense&) = delete;
    NativeDense& operator=(const NativeDense&) = delete;
    /// With layer_hi >= 0, only the `blk.<l>.` matrices with layer_lo <= l < layer_hi are uploaded (a layer
    /// split's stage holds its own layers' projections, not the whole model's); the others keep data == nullptr.
    bool load(const std::vector<std::string>& shards, WeightTable& table, std::string& err,
              bool include_ple_key = false, int64_t layer_lo = 0, int64_t layer_hi = -1);
    /// Frees the uploaded matrices and the scratch, so `load` can run again.  The WeightRefs that pointed at them
    /// must not be used afterwards (the caller reloads their table too).
    void release();
    /// The uploaded bytes of layers [lo, hi).
    uint64_t bytes_in(int64_t lo, int64_t hi) const;
    /// Plan v0.3 P1: the canonical tensor names `load` would serve natively from these shards (eligible name,
    /// supported type, 2-D), read from the GGUF headers only - so the canonical arena can skip them.
    static bool served_names(const std::vector<std::string>& shards, bool include_ple_key,
                             std::set<std::string>& out, std::string& err);
    /// Layer split: load only blocks [lb, le) (every other `blk.N.` projection belongs to another GPU's stage; the
    /// PLE tensors are loaded everywhere).  Process-wide, read by the next `load`; (-1, -1) = all layers.
    static void set_layer_range(int lb, int le);
    /// Layer split: the device bytes `load` would allocate for the matrices of layers [lb, le) - the same walk and
    /// filters, each matrix rounded up to the granule cudaMalloc backs it with (with STRATA_HC_Q8 the Q8_0
    /// hyper-connection rows `load` copies too), reading the GGUF headers and `table` only, so it allocates nothing
    /// and needs no device (#1238).
    /// `table` must hold every tensor the shards carry.
    static bool weight_bytes_for(const std::vector<std::string>& shards, WeightTable& table, bool include_ple_key,
                                 int64_t lb, int64_t le, uint64_t& out, std::string& err);
    static bool weight_bytes_for(const std::vector<std::string>& shards, const std::map<std::string, WeightRef>& table,
                                 bool include_ple_key, int64_t lb, int64_t le, uint64_t& out, std::string& err);
    /// #326: a native pack whose `blk.1.ple_key.weight` row is unquantized (iq_pack --compat-bf16 of a GGUF key
    /// the native kernel also reads, e.g. OrcaRouter's IQ3_XXS) serves the PLE from that row, so it is taken out
    /// of `skip` and `load` does not upload the GGUF key over it.  A quantized row leaves `skip` unchanged.
    static bool keep_unquantized_ple_key(const std::string& pack_dir, std::set<std::string>& skip, std::string& err);
    /// STRATA_HC_Q8=1: the hyper-connection down / up projections of the layers live only as Q8_0 (the bf16 copy is not
    /// loaded: `add_hc_skip` puts their rows in the arena's skip set, `load` quantizes them from the pack file).
    /// Whether the GGUF shards carry the hyper-connection down / up projections as Q8_0 (an Unsloth UD-* file; the GSQ-RCO packs keep them
    /// BF16).  Reads the tensor directories only.  generate.cpp turns STRATA_HC_Q8 on by default on CUDA when this is true (owner yes 2026-11-07).
    static bool gguf_carries_q8_hc(const std::vector<std::string>& shards);
    /// The default rule itself: when STRATA_HC_Q8 is not set at all and the shards carry the projections as Q8_0, sets it to 1 (a value the
    /// user set, also 0, is kept).  Returns whether it set it.  Must run before the first read of the switch (`hc_q8_requested`).
    static bool apply_hc_q8_default(const std::vector<std::string>& shards);
    static bool hc_q8_requested();
    static bool add_hc_skip(const std::string& pack_dir, std::set<std::string>& skip);
    /// A projection's bf16 values for a path that has no Q8 kernel (the prompt path's GEMMs, the single-token read): the
    /// resident copy, or - when the row was skipped - the Q8_0 copy dequantized into a per-stream scratch (`slot` 0 / 1:
    /// two matrices alive at once).  nullptr if there is neither.
    /// `cls` 0 = the prompt path (a scratch pair shared by its streams, ordered by events: `hc_bf16_release` after the
    /// last GEMM that reads the slot), 1 = the layer / session read (its own pair: it is captured into graphs, where an
    /// event wait on an outside event is illegal). The scratches are allocated by `load`, never on first use.
    static const uint16_t* hc_bf16_view(const WeightRef& w, void* stream, int slot, int cls);
    static bool hc_bf16_view_pair(const WeightRef& w0, const WeightRef& w1, void* stream, const uint16_t*& out0,
                                  const uint16_t*& out1, int cls);
    static void hc_bf16_release(void* stream, int slot, int cls);
    static void hc_bf16_release_pair(void* stream, int cls);
    uint64_t weight_bytes() const { return bytes_; }
    /// Estimated device backing footprint of the allocations.
    uint64_t allocated_bytes() const { return allocated_; }
    size_t tensor_count() const { return tensor_count_; }

private:
    void* arena_ = nullptr;
    uint64_t arena_size_ = 0;
    std::vector<void*> allocations_;
    size_t tensor_count_ = 0;
    std::vector<const void*> packed_keys_;   // GGUF-layout pointers registered with STRATA_Q8_PACKED=1
    void* scratch_ = nullptr;
    uint64_t bytes_ = 0;
    uint64_t allocated_ = 0;
    std::vector<uint64_t> layer_bytes_;   ///< uploaded bytes per layer
};
} // namespace strata::core
