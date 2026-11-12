#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/conversation_file.hpp"
#include "strata/kernels/kv_q4.hpp"
#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>

using namespace strata::core;
using namespace strata::kernels;

namespace {
// STRATA_CKPT_ASYNC=1 (the ctest entries conversation_snapshot_test_ckpt_async / _vram_slots): the checkpoint's GDN bytes land in
// the background; the test reads them at once, so it joins after each save (the engine's readers join before they read)
template <class... A> bool ckpt_save(A&&... a) {
    const bool ok = conversation_checkpoint_save(std::forward<A>(a)...);
    conversation_checkpoint_join(nullptr);
    return ok;
}
template <class... A> bool snap_save(A&&... a) {
    const bool ok = conversation_snapshot_save(std::forward<A>(a)...);
    conversation_checkpoint_join(nullptr);
    return ok;
}
}  // namespace

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); std::exit(1); }
}
struct Fixture {
    ModelGeometry g;
    QsaState state;
    std::vector<void*> device, host;
    std::array<void*,5> sources{};
    std::array<size_t,5> sizes{};

    template<class T> void alloc(T*& p, size_t n, bool pinned = false) {
        if (!n) return;
        void* raw = nullptr;
        if (pinned) {
            cuda_check(cudaHostAlloc(&raw, n, cudaHostAllocMapped));
            host.push_back(raw);
            void* mapped = nullptr;
            cuda_check(cudaHostGetDevicePointer(&mapped, raw, 0));
            p = static_cast<T*>(mapped);
        } else {
            cuda_check(cudaMalloc(&raw, n)); device.push_back(raw); p = static_cast<T*>(raw);
        }
    }
    Fixture(int fmt, int mode) {
        auto& st = state;
        st.kv_mode = mode; st.kv_int8 = fmt==kKvInt8; st.kv_q4 = fmt==kKvQ4; st.kv_hybrid = fmt==3;
        st.n_pages = 24; st.n_slots = mode ? 4 : st.n_pages;
        st.max_cells = st.n_pages * 4; st.idx_pooled_rows = mode==2 ? 2 : st.max_cells/4+2;
        const size_t per = fmt==kKvQ4 ? kv_q4_bytes_per_head((int)g.head_dim) : g.head_dim*((st.kv_int8 || st.kv_hybrid) ? 1:2);
        const size_t rows = st.max_cells*g.n_head_kv, slot_rows = st.n_slots*4*g.n_head_kv;
        sizes = {rows*per, rows*per, fmt==kKvInt8 ? rows*(g.head_dim/64)*2:0,
                 fmt==kKvInt8 ? rows*(g.head_dim/64)*2:0, (size_t)st.idx_pooled_rows*g.idx_key_dim*4};
        if (fmt==3) {
            sizes = {rows*per, rows*kv_q4_bytes_per_head((int)g.head_dim), rows*(g.head_dim/64)*2, 0,
                     (size_t)st.idx_pooled_rows*g.idx_key_dim*4};
            alloc(st.k_q,sizes[0]); alloc(st.v_q4,sizes[1]); alloc(st.k_scale,sizes[2]);
            sources[0]=st.k_q; sources[1]=st.v_q4; sources[2]=st.k_scale;
        } else if (fmt==kKvQ4) {
            alloc(st.k_q4,slot_rows*per); alloc(st.v_q4,slot_rows*per);
            if (mode) {alloc(st.host.k_q4,sizes[0],true); alloc(st.host.v_q4,sizes[1],true);}
            sources[0]=mode?st.host.k_q4:st.k_q4; sources[1]=mode?st.host.v_q4:st.v_q4;
        } else if (fmt==kKvInt8) {
            alloc(st.k_q,slot_rows*per); alloc(st.v_q,slot_rows*per);
            alloc(st.k_scale,slot_rows*(g.head_dim/64)*2); alloc(st.v_scale,slot_rows*(g.head_dim/64)*2);
            if (mode) {
                alloc(st.host.k_q,sizes[0],true); alloc(st.host.v_q,sizes[1],true);
                alloc(st.host.k_scale,sizes[2],true); alloc(st.host.v_scale,sizes[3],true);
            }
            sources[0]=mode?st.host.k_q:st.k_q; sources[1]=mode?st.host.v_q:st.v_q;
            sources[2]=mode?st.host.k_scale:st.k_scale; sources[3]=mode?st.host.v_scale:st.v_scale;
        } else {
            alloc(st.k_pool,slot_rows*per); alloc(st.v_pool,slot_rows*per);
            if (mode) {alloc(st.host.k_pool,sizes[0],true); alloc(st.host.v_pool,sizes[1],true);}
            sources[0]=mode?st.host.k_pool:st.k_pool; sources[1]=mode?st.host.v_pool:st.v_pool;
        }
        alloc(st.idx_pooled,sizes[4]); sources[4]=st.idx_pooled;
        alloc(st.page_table,st.n_pages*4);
        if (mode==1) {
            auto& m=st.map;
            m.page_table=st.page_table; m.n_blocks=st.n_pages; m.n_slots=st.n_slots;
            alloc(m.slot_block,st.n_slots*4); alloc(m.slot_stamp,st.n_slots*4); alloc(m.slot_ref,st.n_slots*4);
            alloc(m.miss_block,st.n_slots*4); alloc(m.miss_slot,st.n_slots*4); alloc(m.ctl,kKvCtlInts*4);
        } else if (mode==2) {
            kv_ring_table(st.page_table,st.n_pages,st.n_slots,nullptr);
        }
    }
    void fill(uint8_t salt) {
        for (size_t i=0;i<sources.size();++i) {
            std::vector<uint8_t> data(sizes[i]);
            for (size_t j=0;j<data.size();++j) data[j]=(uint8_t)(salt+i*31+j*7+j/257);
            if (!data.empty()) cuda_check(cudaMemcpy(sources[i],data.data(),data.size(),cudaMemcpyDefault));
        }
    }
    void fill_after(uint8_t salt, int64_t first_dirty) {
        for (size_t i=0;i<sources.size();++i) {
            const size_t offset = i == 4 ? size_t(first_dirty/4)*g.idx_key_dim*4
                                         : (sizes[i]/size_t(state.max_cells))*size_t((first_dirty/4)*4);
            if (sizes[i] > offset)
                cuda_check(cudaMemset(static_cast<uint8_t*>(sources[i])+offset, salt, sizes[i]-offset));
        }
    }
    ~Fixture() { for (void* p:device) cudaFree(p); for (void* p:host) cudaFreeHost(p); }
};
bool equal(const ConversationKv& a,const ConversationKv& b) {
    return a.k==b.k && a.v==b.v && a.k_scale==b.k_scale && a.v_scale==b.v_scale && a.pooled==b.pooled;
}

void full_session(int fmt, int mode, int experts) {
    Fixture main(fmt,mode), draft(fmt==3?kKvInt8:fmt,2);
    auto& g=main.g;
    g.n_layers=4; g.n_expert=experts;
    g.ssm_state_size=2; g.ssm_v_heads=2; g.ssm_conv_channels=8;
    if (std::getenv("STRATA_TEST_BIG_GDN")) { g.ssm_state_size=64; g.ssm_v_heads=64; }   // a GDN state over 1 MiB: the async checkpoint path
    SessionState ss;
    ss.max_cells=96; ss.qsa_states=&main.state;
    ss.layer_hi=g.n_layers; ss.gdn_alloc=g.n_gdn_layers(); ss.qsa_alloc=g.n_qsa_layers();   // the whole-model carve
    std::string err;
    ConversationStateSizes sizes;
    check(conversation_state_sizes(g,sizes,err),"whole-session geometry sizes");
    main.alloc(ss.gdn_state,sizes.gdn); main.alloc(ss.ple_hist,sizes.ple);
    main.alloc(main.state.idx_tail,sizes.tail); main.alloc(main.state.idx_dead,sizes.dead);
    main.alloc(main.state.idx_block_pos,sizes.block_pos);
    std::vector<int32_t> ids(65);
    for (size_t i=0;i<ids.size();++i) ids[i]=(int32_t)i+1;
    std::vector<ConversationImageKey> images;
    std::vector<ConversationCheckpoint> checkpoints;
    auto fill=[&](uint8_t salt) {
        main.fill(salt); draft.fill(salt);
        for (const auto& [p,n] : std::vector<std::pair<void*,size_t>>{
                 {ss.gdn_state,sizes.gdn},{ss.ple_hist,sizes.ple},{main.state.idx_tail,sizes.tail},
                 {main.state.idx_dead,sizes.dead},{main.state.idx_block_pos,sizes.block_pos}})
            cuda_check(cudaMemset(p,salt,n));
        // A real indexer maintains its spare pooled row as a copy of idx_dead.
        cuda_check(cudaMemcpy(main.state.idx_pooled+(ids.size()/4)*g.idx_key_dim,
                              main.state.idx_dead,sizes.dead,cudaMemcpyDeviceToDevice));
        cuda_check(cudaDeviceSynchronize());
    };
    fill(13);
    ConversationCheckpoint checkpoint;
    checkpoint.ids.assign(ids.begin(),ids.begin()+3);
    check(ckpt_save(checkpoint,ss,g,err),"save complete running checkpoint");
    checkpoint.used = 17;
    checkpoints.push_back(std::move(checkpoint));
    const ConversationView view{ids,images,checkpoints,true};
    SavedConversation a,b,restored;
    check(snap_save(a,view,ss,g,draft.state,err),"capture complete A");
    check(a.checkpoints[0].used == 17,"upstream checkpoint LRU stamp survives capture");
    fill(177);
    check(snap_save(b,view,ss,g,draft.state,err),"capture complete B");
    {
        // disk save path: metadata + streamed K/V give the same file as the captured image
        SavedConversation meta;
        std::vector<SessionKvSource> sources;
        check(conversation_snapshot_sources(meta,sources,view,ss,g,draft.state,err),"disk-save sources");
        check(meta.kv.empty() && sources.size()==b.kv.size(),"one source per K/V layer, none captured");
        const SessionFileIdentity id{1,2};
        namespace fs=std::filesystem;
        const fs::path dir=fs::temp_directory_path()/("strata-snap-test-"+std::to_string(experts)+"-"+
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(dir);
        const std::string p1=(dir/"streamed.bin").string(), p2=(dir/"captured.bin").string();
        auto slurp=[](const std::string& p){ std::ifstream f(p,std::ios::binary);
            return std::vector<char>((std::istreambuf_iterator<char>(f)),std::istreambuf_iterator<char>()); };
        size_t n1=0,n2=0;
        check(session_file_write(p1,meta,sources,id,n1,err),"streamed session write");
        check(session_file_write(p2,b,id,n2,err),"captured session write");
        SavedConversation x,y;
        check(session_file_read(p1,id,x,n1,err) && session_file_read(p2,id,y,n2,err) && n1==n2,"both files read");
        check(x.live.gdn==y.live.gdn && x.live.dead==y.live.dead && x.live.ids==y.live.ids &&
              x.checkpoints.size()==y.checkpoints.size() && equal(x.kv[0],y.kv[0]) && equal(x.kv.back(),y.kv.back()),
              "streamed file equals captured file");
        check(slurp(p1)==slurp(p2),"streamed file is byte-identical to the captured file");
        std::error_code ec;
        fs::remove_all(dir,ec);
        auto empty_ids=std::vector<int32_t>{};
        const ConversationView none{empty_ids,images,checkpoints,true};
        check(!conversation_snapshot_sources(meta,sources,none,ss,g,draft.state,err),"empty session refused");
    }
    check(a.live.dead!=b.live.dead,"different opening-state spare keys in regression fixture");
    auto bad=a; bad.kv.back().k.pop_back();
    check(conversation_snapshot_restore(bad,ss,g,draft.state,err)==ConversationRestore::invalid,
          "reject invalid late draft before any main-layer write");
    check(snap_save(restored,view,ss,g,draft.state,err),"capture B after refused restore");
    check(restored.live.gdn==b.live.gdn && restored.live.dead==b.live.dead &&
          equal(restored.kv[0],b.kv[0]) && equal(restored.kv[1],b.kv[1]),"refusal preserves all tested state");
    check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,"restore complete A");
    check(snap_save(restored,view,ss,g,draft.state,err),"capture restored A");
    check(restored.live.gdn==a.live.gdn && restored.live.ple==a.live.ple && restored.live.tails==a.live.tails &&
          restored.live.dead==a.live.dead && restored.live.block_pos==a.live.block_pos &&
          equal(restored.kv[0],a.kv[0]) && equal(restored.kv[1],a.kv[1]),"whole-session A/B/A exactness including spare key");
    check(ss.ple_prev[0]==64 && ss.ple_prev[1]==65,"PLE token window reconstructed");
    {
        // #2409: a restore writes into the session's own buffers. After the copy stream exists (the restores above made
        // it) a restore of either image allocates nothing on the card: free memory does not move (2 MiB allocation
        // granules; 4 MiB of slack for a desktop card's other programs).
        check(conversation_copy_stream_ready(0),"this thread has its copy stream after the first restore");
        cudaDeviceSynchronize();
        size_t before=0,after=0,total=0;
        check(cudaMemGetInfo(&before,&total)==cudaSuccess,"free memory before the repeated restores");
        for(int i=0;i<6;++i)
            check(conversation_snapshot_restore(i%2?b:a,ss,g,draft.state,err)==ConversationRestore::restored,"repeated restore");
        cudaDeviceSynchronize();
        check(cudaMemGetInfo(&after,&total)==cudaSuccess,"free memory after the repeated restores");
        check((long long)after+(4ll<<20)>=(long long)before,"six more restores allocate nothing on the card");
        check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,"A is back");
    }
    {
        // a borrowed prefix (ConversationCache::borrows): the K/V of A's first tokens and A's checkpoint there, A
        // untouched - the same bytes as restoring A whole and rewinding to that checkpoint
        const int64_t at = (int64_t) a.checkpoints[0].ids.size();
        check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,"restore A whole");
        check(conversation_checkpoint_restore(a.checkpoints[0],ss,g,err),"rewind it to its checkpoint");
        ConversationKv want_main,want_draft,got_main,got_draft;
        ConversationCheckpoint want_state,got_state;
        want_state.ids=got_state.ids=a.checkpoints[0].ids;
        check(conversation_kv_save(want_main,main.state,g,at,true,err) &&
              conversation_kv_save(want_draft,draft.state,g,at,false,err) &&
              ckpt_save(want_state,ss,g,err),"capture the rewound prefix");
        fill(201);   // another conversation in the session
        const SavedConversation untouched=a;
        check(conversation_snapshot_restore_prefix(a,(int64_t)a.live.ids.size()+1,ss,g,draft.state,err)==
                  ConversationRestore::invalid &&
              conversation_snapshot_restore_prefix(a,0,ss,g,draft.state,err)==ConversationRestore::invalid,
              "a prefix outside the image is refused");
        check(conversation_snapshot_restore_prefix(a,at,ss,g,draft.state,err)==ConversationRestore::restored,
              "restore A's prefix only");
        check(conversation_checkpoint_restore(a.checkpoints[0],ss,g,err),"mount the checkpoint over it");
        check(conversation_kv_save(got_main,main.state,g,at,true,err) &&
              conversation_kv_save(got_draft,draft.state,g,at,false,err) &&
              ckpt_save(got_state,ss,g,err),"capture the borrowed prefix");
        check(equal(got_main,want_main) && equal(got_draft,want_draft),
              "borrowed K/V equals the whole restore's prefix");
        check(got_state.gdn==want_state.gdn && got_state.ple==want_state.ple && got_state.tails==want_state.tails &&
              got_state.dead==want_state.dead && got_state.block_pos==want_state.block_pos,"the same running state");
        check(untouched.live.ids==a.live.ids && equal(untouched.kv[0],a.kv[0]) && equal(untouched.kv[1],a.kv[1]),
              "the parked image is unchanged");
    }
    for (int64_t dirty : {65, 3, 0}) {
        check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,"restore growth fixture base");
        ConversationKvReuse reuse{a.kv,65,dirty};
        const uint8_t* original = nullptr;
        reuse.kv[0].k.visit(0,1,[&](const uint8_t* p,size_t,size_t){original=p;return true;});
        main.fill_after(91,dirty); draft.fill_after(91,std::max<int64_t>(0,dirty-1));
        cuda_check(cudaMemset(ss.gdn_state,91,sizes.gdn));
        ids.resize(70);
        for (size_t i=65;i<ids.size();++i) ids[i]=int32_t(i+1);
        cuda_check(cudaMemcpy(main.state.idx_pooled+(ids.size()/4)*g.idx_key_dim,
                              main.state.idx_dead,sizes.dead,cudaMemcpyDeviceToDevice));
        SavedConversation fresh,incremental;
        check(snap_save(fresh,view,ss,g,draft.state,err),"full capture reference after growth or rewind");
        size_t peak=0,reused=0;
        check(conversation_snapshot_capture_bytes(reuse,view,ss,g,draft.state,peak,err),"admit incremental capture peak");
        check(snap_save(incremental,view,ss,g,draft.state,err,std::move(reuse),&reused),"capture with retained pages");
        check(incremental.bytes() <= peak,"incremental allocation stays within admitted bound");
        check(incremental.live.gdn==fresh.live.gdn && incremental.live.ple==fresh.live.ple &&
              incremental.live.dead==fresh.live.dead && equal(incremental.kv[0],fresh.kv[0]) &&
              equal(incremental.kv[1],fresh.kv[1]),"incremental capture equals full capture after growth or rewind");
        check((dirty>=4)==(reused>0),"only complete unchanged pages or rows are retained");
        incremental.kv[0].k.visit(0,1,[&](const uint8_t* p,size_t,size_t){
            check(p==original,"growth never reallocates the retained payload");return true;
        });
        check(conversation_snapshot_restore(incremental,ss,g,draft.state,err)==ConversationRestore::restored,"restore segmented incremental snapshot");
        uint64_t fingerprint=0;
        check(conversation_kv_verify(incremental.kv.back(),draft.state,g,70,false,fingerprint,err),"incremental draft authoritative and ring read-back");
        ids.resize(65);
    }
    {
        // a layer split's later stage: the same image WITHOUT the draft layer's K/V (draft == nullptr)
        check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,"restore A before stage images");
        size_t with_draft=0,without=0;
        check(conversation_snapshot_bytes(view,ss,g,draft.state,with_draft,err) &&
              conversation_snapshot_bytes(view,ss,g,nullptr,without,err) && without<with_draft,
              "a stage image's estimate leaves the draft out");
        SavedConversation stage,stage_b,stage_back;
        check(snap_save(stage,view,ss,g,nullptr,err),"capture a stage image");
        check(stage.kv.size()==a.kv.size()-1 && equal(stage.kv[0],a.kv[0]) && stage.live.gdn==a.live.gdn,
              "a stage image holds the session's own layers, no draft");
        check(stage.bytes()<a.bytes(),"a stage image is smaller by the draft ring");
        check(conversation_snapshot_validate(stage,ss,g,nullptr,err),"validate a stage image without a draft");
        check(!conversation_snapshot_validate(stage,ss,g,draft.state,err),"a stage image is refused where a draft is expected");
        check(!conversation_snapshot_validate(a,ss,g,nullptr,err),"an image with a draft is refused where none is expected");
        fill(201);
        check(snap_save(stage_b,view,ss,g,nullptr,err),"capture another stage state");
        check(conversation_snapshot_restore(a,ss,g,nullptr,err)==ConversationRestore::invalid,
              "restoring a draft image as a stage image is refused before any write");
        check(snap_save(stage_back,view,ss,g,nullptr,err) && stage_back.live.gdn==stage_b.live.gdn &&
              equal(stage_back.kv[0],stage_b.kv[0]),"the refusal left the stage untouched");
        check(conversation_snapshot_restore(stage,ss,g,nullptr,err)==ConversationRestore::restored,"restore a stage image");
        check(snap_save(stage_back,view,ss,g,nullptr,err) && stage_back.live.gdn==stage.live.gdn &&
              stage_back.live.dead==stage.live.dead && equal(stage_back.kv[0],stage.kv[0]),"stage A/B/A exactness");
        // its retained K/V: the next park copies only what changed, and equals a full capture
        ConversationKvReuse reuse{stage.kv,65,65,{}};
        SavedConversation fresh,incremental;
        size_t peak=0,reused=0;
        check(snap_save(fresh,view,ss,g,nullptr,err),"full stage capture reference");
        check(conversation_snapshot_capture_bytes(reuse,view,ss,g,nullptr,peak,err),"admit an incremental stage capture");
        check(snap_save(incremental,view,ss,g,nullptr,err,std::move(reuse),&reused),"incremental stage capture");
        check(reused>0 && incremental.bytes()<=peak && equal(incremental.kv[0],fresh.kv[0]) &&
              incremental.live.gdn==fresh.live.gdn,"an incremental stage capture reuses pages and equals a full one");
        SavedConversation split_image=stage;
        split_image.stage_images.push_back(stage);
        check(!conversation_snapshot_validate(split_image,ss,g,draft.state,err),
              "a layer split's image is refused by the whole-session form");
        ConversationKvReuse wrong{a.kv,65,65,{}};
        check(!conversation_snapshot_capture_bytes(wrong,view,ss,g,nullptr,peak,err),"a draft image's K/V is not a stage's reuse");
    }
    check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,"restore A after stage images");
    check(conversation_checkpoint_restore(a.checkpoints[0],ss,g,err),"restore early running checkpoint");
    std::vector<uint8_t> spare(sizes.dead);
    cuda_check(cudaMemcpy(spare.data(),main.state.idx_pooled,sizes.dead,cudaMemcpyDeviceToHost));
    check(spare==a.checkpoints[0].dead,"checkpoint rebuilds spare row over a later completed block");
    check(ss.ple_prev[0]==2 && ss.ple_prev[1]==3,"checkpoint PLE token window");
}

void elastic_host_kv_test() {
    ModelGeometry g;
    const int64_t max_cells = 12288, init_cells = 4096, step_cells = 4096;
    qsa_set_kv_resident(4096);
    qsa_set_kv_int8(true);
    qsa_set_kv_host_elastic(true, init_cells, step_cells);
    check(qsa_kv_host_elastic(), "elastic host KV enabled");
    const size_t dev_bytes = (size_t) qsa_state_bytes(g, max_cells, true);
    void *dev_a = nullptr, *dev_b = nullptr;
    cuda_check(cudaMalloc(&dev_a, dev_bytes));
    cuda_check(cudaMalloc(&dev_b, dev_bytes));
    QsaState a{}, b{};
    check(qsa_state_init(g, max_cells, dev_a, a) > 0, "init elastic state a");
    check(qsa_state_init(g, max_cells, dev_b, b) > 0, "init elastic state b");
    check(a.host.chunked() && b.host.chunked() && a.host.block_shift == 10, "elastic states use 1024-block chunks");
    check(qsa_kv_host_elastic_cells() == init_cells, "initial committed extent matches init_cells");
    const uint64_t map0 = qsa_kv_host_elastic_mapped_bytes(), full = qsa_kv_host_elastic_full_bytes();
    check(map0 > 0 && full == map0 * 3 && qsa_kv_host_bytes() == map0, "initial mapped bytes are 1/3 of full ceiling");

    // Build a 10000-cell ConversationKv image (spans 3 chunks: 0..4095, 4096..8191, 8192..9999).
    check(qsa_kv_host_elastic_need_bytes(10000) > 0, "need_bytes reports positive delta before grow");
    check(qsa_kv_host_elastic_grow_state(a, 10000), "grow state a to cover 10000 cells (3 chunks)");
    check(qsa_kv_host_elastic_mapped_bytes() == map0 * 2, "state a (3 chunks) + state b (1 chunk) = 4 chunks mapped");

    // Populate a's 3 chunks and pooled indexer with deterministic bytes via conversation_kv_restore from a synthetic image.
    const int64_t upto = 10000;
    std::string err;
    ConversationKv img;
    check(conversation_kv_save(img, a, g, upto, true, err), "save initial 10000-cell image from grown state a");
    auto fill_buf = [](ConversationBuffer& buf, uint8_t seed) {
        buf.visit(0, buf.size(), [&](uint8_t* p, size_t n, size_t at) {
            for (size_t i = 0; i < n; ++i) p[i] = (uint8_t) (seed + (at + i) * 13 + (at + i) / 251);
            return true;
        });
    };
    fill_buf(img.k, 17);
    fill_buf(img.v, 53);
    fill_buf(img.k_scale, 89);
    fill_buf(img.v_scale, 131);
    fill_buf(img.pooled, 193);

    // Shrink `a` back to 1 chunk (4096 cells), then restore the 10000-cell image into `a`: restore must auto-grow `a`!
    check(qsa_kv_host_elastic_shrink(init_cells) > 0, "shrink elastic pools back to init_cells");
    check(qsa_kv_host_elastic_cells() == init_cells && qsa_kv_host_elastic_mapped_bytes() == map0,
          "shrink decommits chunks above init_cells");
    check(conversation_kv_restore(img, a, g, upto, true, err), "restore 10000-cell image auto-grows chunked state a");
    check(qsa_kv_host_elastic_mapped_bytes() == map0 * 2, "restore grew state a back to 3 chunks");

    ConversationKv saved;
    check(conversation_kv_save(saved, a, g, upto, true, err), "save 10000-cell image back from chunked state a");
    check(equal(img, saved), "save/restore round-trip across 3 chunks is byte-identical");

    uint64_t fp = 0;
    check(conversation_kv_verify(img, a, g, upto, true, fp, err), "conversation_kv_verify succeeds across chunks");

    // Disk-save streaming source across chunks.
    SessionKvSource src;
    check(conversation_kv_source(src, a, g, upto, true, err), "conversation_kv_source on chunked state");
    const std::array<const ConversationBuffer*, 5> parts = {&img.k, &img.v, &img.k_scale, &img.v_scale, &img.pooled};
    for (size_t i = 0; i < 5; ++i) {
        std::vector<uint8_t> got(src.sizes[i]), want(src.sizes[i]);
        const size_t cut = src.sizes[i] / 2;   // split right across chunk 1
        check(src.read(i, 0, got.data(), cut) && src.read(i, cut, got.data() + cut, src.sizes[i] - cut),
              "chunked SessionKvSource split read");
        check(parts[i]->read(want.data(), 0, want.size()) && got == want,
              "chunked SessionKvSource matches image bytes");
    }

    // Copy from `a` (10000 cells, 3 chunks) to `b` (still at 4096 cells, 1 chunk):
    // first 4000 cells fit in chunk 0 without growing `b`; then copying to 10000 with unchanged_tokens=2000 auto-grows `b` to 3 chunks!
    check(qsa_kv_host_elastic_cells() == init_cells, "min mapped cells across pools is init_cells before b grows");
    check(conversation_kv_copy(a, b, g, 4000, true, err, false, 0), "conversation_kv_copy within chunk 0 keeps b at 1 chunk");
    check(qsa_kv_host_elastic_cells() == init_cells, "state b stays at 1 chunk for 4000 cells");
    check(conversation_kv_copy(a, b, g, upto, true, err, false, 2000), "conversation_kv_copy across chunks auto-grows b");
    check(qsa_kv_host_elastic_cells() == max_cells && qsa_kv_host_elastic_mapped_bytes() == full,
          "state b grew to 3 chunks on copy");
    ConversationKv saved_b;
    check(conversation_kv_save(saved_b, b, g, upto, true, err), "save from copied state b");
    check(equal(img, saved_b), "conversation_kv_copy across chunks is byte-identical");

    // Prefix restore (5000 cells -> 2 chunks) into shrunk `b`.
    check(qsa_kv_host_elastic_shrink(init_cells) > 0, "shrink both states to 1 chunk");
    check(conversation_kv_restore(img, a, g, upto, true, err), "re-restore a to 3 chunks");
    check(conversation_kv_restore_prefix(img, b, g, upto, 5000, true, err), "restore 5000-cell prefix into b");
    check(qsa_kv_host_elastic_cells() == 8192, "prefix restore grew b to 2 chunks (8192 cells), not 3");
    ConversationKv pref_a, pref_b;
    check(conversation_kv_save(pref_a, a, g, 5000, true, err) &&
          conversation_kv_save(pref_b, b, g, 5000, true, err) && equal(pref_a, pref_b),
          "prefix restore across 2 chunks matches full state prefix");

    qsa_kv_host_elastic_release_all();
    qsa_set_kv_host_elastic(false);
    qsa_set_kv_resident(0);
    qsa_set_kv_int8(false);
    cudaFree(dev_a);
    cudaFree(dev_b);
}
}

int main() {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    for (int fmt : std::array<int,4>{kKvF16,kKvInt8,kKvQ4,3}) for (int mode : {0,1,2}) {
        if (fmt==3 && mode!=0) continue;
        Fixture f(fmt,mode);
        if (fmt==3) {
            std::string err;
            ConversationKv invalid;
            for (int unsupported : {1,2}) {
                f.state.kv_mode=unsupported;
                check(conversation_kv_bytes(f.state,f.g,9,true)==0,"hybrid streaming/ring rejected without block-mover abort");
                check(!conversation_kv_save(invalid,f.state,f.g,9,true,err),"hybrid streaming/ring capture fails closed");
            }
            f.state.kv_mode=0;
        }
        {
            std::string err;
            ConversationKv invalid;
            for (const int64_t bad : {-1LL, 97LL, (long long) std::numeric_limits<int64_t>::max()}) {
                check(!conversation_kv_save(invalid,f.state,f.g,bad,mode!=2,err),"reject invalid save extent before allocation");
                check(!conversation_kv_restore(invalid,f.state,f.g,bad,mode!=2,err),"reject invalid restore extent before copying");
                check(conversation_kv_bytes(f.state,f.g,bad,mode!=2)==0,"invalid extent cannot overflow byte estimate");
            }
        }
        for (int64_t upto : {0,1,3,4,5,63,64,65,96}) {
            const bool index=mode!=2;
            std::string err;
            f.fill(13);
            ConversationKv a,b,restored;
            cuda_check(cudaDeviceSynchronize());
            check(conversation_kv_save(a,f.state,f.g,upto,index,err),"save A");
            check(a.bytes()==conversation_kv_bytes(f.state,f.g,upto,index),"size estimate equals snapshot payload");
            f.fill(177);
            check(conversation_kv_save(b,f.state,f.g,upto,index,err),"save B");
            check(upto==0 || !equal(a,b),"fixture changes state");
            check(conversation_kv_restore(a,f.state,f.g,upto,index,err),"restore A over B");
            cuda_check(cudaDeviceSynchronize());
            check(conversation_kv_save(restored,f.state,f.g,upto,index,err),"read back restored A");
            check(equal(a,restored),"A/B/A byte-exact K/V and indexer state");
            check(!conversation_kv_copy(f.state,f.state,f.g,upto,index,err,false,-1) &&
                  !conversation_kv_copy(f.state,f.state,f.g,upto,index,err,false,upto+1),
                  "conversation_kv_copy rejects out-of-range unchanged_tokens");
            check(conversation_kv_copy(f.state,f.state,f.g,upto,index,err,false,upto/2),
                  "conversation_kv_copy accepts valid unchanged_tokens");
            uint64_t fingerprint = 0;
            check(conversation_kv_verify(a,f.state,f.g,upto,index,fingerprint,err),"verify restored authoritative and resident bytes without rewriting them");
            {
                // disk save streams the same bytes as a captured image, without the host copy
                SessionKvSource src;
                check(conversation_kv_source(src,f.state,f.g,upto,index,err),"K/V source for a disk save");
                check(src.format==a.format && src.cells==a.cells && src.heads==a.heads && src.head_dim==a.head_dim &&
                      src.page_size==a.page_size && src.pooled_rows==a.pooled_rows && src.idx_dim==a.idx_dim,
                      "K/V source metadata equals the captured image");
                const std::array<const ConversationBuffer*,5> parts={&a.k,&a.v,&a.k_scale,&a.v_scale,&a.pooled};
                for (size_t i=0;i<5;++i) {
                    check(src.sizes[i]==parts[i]->size(),"K/V source part size");
                    std::vector<uint8_t> got(src.sizes[i]), want(src.sizes[i]);
                    const size_t half=src.sizes[i]/2;   // two reads: offsets are honoured
                    check(src.read(i,0,got.data(),half) && src.read(i,half,got.data()+half,src.sizes[i]-half),
                          "K/V source reads");
                    check(parts[i]->read(want.data(),0,want.size()) && got==want,"K/V source bytes equal the image");
                }
                SessionKvSource none;
                check(!conversation_kv_source(none,f.state,f.g,-1,index,err),"K/V source rejects an invalid extent");
            }
            if (mode==1) {
                std::vector<int32_t> table((size_t)f.state.n_pages);
                cuda_check(cudaMemcpy(table.data(),f.state.page_table,table.size()*4,cudaMemcpyDeviceToHost));
                check(std::all_of(table.begin(),table.end(),[](int32_t x){return x==-1;}),"stale VRAM pages invalidated");
            }
            if (mode==2 && upto>0) {
                const auto s=qsa_real_shapes();
                const int64_t b1=(upto+s.page_size-1)/s.page_size,b0=std::max<int64_t>(0,b1-f.state.n_slots);
                const size_t block=kv_block_bytes(s,fmt)/2;
                // Codes and scales are separate in INT8; check code bytes here.
                const size_t code_block=fmt==kKvInt8 ? s.page_size*s.n_head_kv*s.head_dim : block;
                const void* slots=fmt==kKvQ4?(void*)f.state.k_q4:fmt==kKvInt8?(void*)f.state.k_q:(void*)f.state.k_pool;
                std::vector<uint8_t> got(code_block);
                for (int64_t page=b0;page<b1;++page) {
                    cuda_check(cudaMemcpy(got.data(),(const uint8_t*)slots+(page%f.state.n_slots)*code_block,code_block,cudaMemcpyDeviceToHost));
                    std::vector<uint8_t> expected(code_block);
                    check(a.k.read(expected.data(), page*code_block, code_block),"read segmented expected draft page");
                    check(got == expected,"draft ring contains restored pages");
                }
            }
            auto bad=a; bad.head_dim++;
            check(!conversation_kv_restore(bad,f.state,f.g,upto,index,err),"reject incompatible geometry");
            if (fmt==kKvInt8) {
                // #293: INT8 K/V saved through the Hadamard rotation (STRATA_KV_ROT=1) never restores into a state
                // without it, nor the reverse
                f.state.kv_rot=true;
                ConversationKv rotated;
                check(conversation_kv_save(rotated,f.state,f.g,upto,index,err),"save rotated INT8");
                check(!conversation_kv_restore(a,f.state,f.g,upto,index,err),"unrotated image into a rotated state rejected");
                f.state.kv_rot=false;
                check(!conversation_kv_restore(rotated,f.state,f.g,upto,index,err),"rotated image into an unrotated state rejected");
                check(conversation_kv_restore(a,f.state,f.g,upto,index,err),"unrotated image still restores");
            }
            if (!a.k.empty()) {
                bad=a; bad.k.pop_back();
                check(!conversation_kv_restore(bad,f.state,f.g,upto,index,err),"reject malformed payload");
            }
        }
    }
    for (int fmt : {kKvF16,kKvInt8,kKvQ4}) for (int mode : {0,1}) for (int experts : {256,512})
        full_session(fmt,mode,experts);
    for (int experts : {256,512}) full_session(3,0,experts);
    elastic_host_kv_test();
    std::printf("conversation_snapshot_test: %d checks passed\n",checks);
}
