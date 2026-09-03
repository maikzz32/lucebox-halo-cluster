// DeepSeek4Backend — ModelBackend for DeepSeek V4 Flash MLA+MoE models.
//
// Architecture: Multi-head Latent Attention (MLA), KV compression with
// learned compressors, Hierarchical Controller (HC), MoE with hash routing
// (first 3 layers) + top-k routing + shared expert.

#pragma once

#include "common/model_backend.h"
#include "common/moe_expert_compute.h"
#include "common/sampler.h"
#include "../common/moe_hybrid_placement.h"
#include "../common/moe_hybrid_routing_stats.h"
#include "../common/moe_hybrid_storage.h"
#include "../common/moe_hybrid_stream.h"
#include "deepseek4_internal.h"
#include "deepseek4_dspark.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <memory>
#include <random>
#include <string>
#include <vector>

namespace dflash::cluster {
struct Ds4ClusterHooks;  // cluster/cluster_decision_hooks.h
struct ClusterConfig;    // cluster/cluster_config.h
class IClusterComm;      // cluster/cluster_comm.h
}

namespace dflash::common {

struct Ds4ClusterRuntime;  // deepseek4_cluster.h

// Bounds the sparse heterogeneous prefill arena once accumulated attention
// context dominates its memory footprint. Decode batching is unaffected.
int deepseek4_hybrid_prefill_chunk_tokens(
    int requested_chunk,
    int context_end,
    int current_cap = 0);

// Selects the next sparse heterogeneous prefill batch. Large batches retain
// their throughput through the memory-light part of the prompt, then shrink
// at the late-context boundary where one attention arena would otherwise
// exhaust a tightly packed discrete GPU.
int deepseek4_hybrid_prefill_step_tokens(
    int configured_chunk,
    int position,
    int remaining_tokens);
class DeepSeek4Backend : public ModelBackend {
public:
    explicit DeepSeek4Backend(const DeepSeek4BackendConfig & cfg);
    ~DeepSeek4Backend() override;

    DeepSeek4Backend(const DeepSeek4Backend &) = delete;
    DeepSeek4Backend & operator=(const DeepSeek4Backend &) = delete;

    bool init();

    // ModelBackend interface
    void print_ready_banner() const override;

    bool park(ParkTarget target) override;
    bool unpark(ParkTarget target) override;
    bool is_target_parked() const override { return parked_; }

    GenerateResult generate_impl(const GenerateRequest & req,
                                 const DaemonIO & io) override;

    bool snapshot_save(int slot) override;
    void snapshot_free(int slot) override;
    bool snapshot_used(int slot) const override;
    int  snapshot_cur_pos(int slot) const override;

    GenerateResult restore_and_generate_impl(int slot,
                                             const GenerateRequest & req,
                                             const DaemonIO & io) override;

    bool handle_compress(const std::string & line,
                         const DaemonIO & io) override;
    void free_drafter() override;

    void shutdown() override;

    const MoeHybridRoutingStats * get_routing_stats() const override {
        return routing_stats_.get();
    }

    // Cluster lockstep decisions (lucebox-halo-cluster, WP2). nullptr (the
    // default) keeps upstream single-node behaviour byte-identical. The
    // pointer is borrowed: the head decorator / worker main loop owns the
    // hooks and must outlive every generate() call.
    void set_cluster_hooks(cluster::Ds4ClusterHooks * hooks) { hooks_ = hooks; }
    cluster::Ds4ClusterHooks * cluster_hooks() const { return hooks_; }

    // Cluster expert-parallel (WP3). The backend loads only this rank's expert
    // shard (deepseek4_cluster.h) and runs the per-layer forward with one
    // all-reduce per MoE layer; fused whole-model graphs are disabled.
    //
    // Preferred order: set_cluster(&cfg, nullptr) BEFORE init() (the
    // communicator does not exist yet at factory time), then
    // set_cluster(&cfg, comm) once RCCL is up: the second call only attaches
    // the communicator. Calling it for the first time AFTER init() still
    // works but reloads the model (park + unpark) with sharding, i.e. the
    // full expert set is loaded once for nothing. cfg is copied; comm is
    // borrowed and must outlive every forward. Returns false on a config
    // mismatch or a failed reload.
    bool set_cluster(const cluster::ClusterConfig * cfg, cluster::IClusterComm * comm);
    const Ds4ClusterRuntime * cluster_runtime() const { return cluster_.get(); }

    // True when a request with a greedy sampler and no budget hook would
    // decode with DSpark rather than AR. The cluster head asks this before it
    // broadcasts the request: every rank must enter the same decode loop, so
    // the head's answer, not each worker's own configuration, decides.
    bool spec_decode_ready() const { return spec_enabled_ && spec_drafter_ != nullptr; }

private:
    cluster::Ds4ClusterHooks * hooks_ = nullptr;
    // True only when hooks_ belongs to a real cluster (head or worker).
    bool cluster_active() const;
    bool cluster_worker() const;

    DeepSeek4BackendConfig cfg_;
    ggml_backend_t         backend_      = nullptr;
    ggml_backend_t         snap_backend_ = nullptr;
    ggml_backend_t         expert_backend_ = nullptr;
    DeepSeek4Weights       w_;
    DeepSeek4Cache         cache_;
    bool                   parked_       = false;

    // Sampler
    SamplerCfg             sampler_;
    std::mt19937_64        sampler_rng_{std::random_device{}()};

    // Snapshots
    static constexpr int PREFIX_SLOTS = 64;
    struct SnapshotAux {
        std::vector<float> last_logits;
        std::vector<float> spec_feat_window;
        bool used = false;
    };
    DeepSeek4Snapshot      snapshots_[PREFIX_SLOTS];
    SnapshotAux            snapshot_aux_[PREFIX_SLOTS];
    std::vector<float>     last_logits_;
    // Absolute cache position represented by last_logits_. A snapshot is
    // safe only when this matches cache_.cur_pos.
    int                    last_logits_pos_ = -1;

    // DSpark speculative decode (opt-in: DFLASH_DS4_SPEC=1 + DFLASH_DS4_DRAFT=<gguf>).
    bool                           spec_enabled_ = false;
    bool                           spec_drafter_parked_ = false;
    std::string                    spec_draft_path_;
    ggml_backend_t                 spec_backend_ = nullptr;
    std::unique_ptr<DSparkDrafter> spec_drafter_;
    std::vector<float>             spec_feat_window_;
    // Once a long prompt selects the fragmentation-safe prefill shape, retain
    // it for later requests so the HIP arenas never switch back under load.
    int                            hybrid_prefill_chunk_cap_ = 0;

    bool load_spec_drafter();
    void release_spec_drafter(bool mark_parked);
    void keep_spec_feature_tail(std::vector<float> & features,
                                size_t max_rows) const;
    // True when a wide prefill path returns per-token DSpark features and the
    // caller can retain only the requested capture window without splitting.
    static bool supports_batched_spec_feature_capture(
        bool hybrid,
        PrefillAttentionMode mode,
        int n_tokens);
    // Limit a prefill batch to a region with a uniform DSpark capture policy.
    // Wide GPU paths can capture a subrange without splitting the final
    // feature window; other paths still stop exactly at capture boundaries.
    static int capture_safe_prefill_tokens(int token_offset,
                                           int requested_tokens,
                                           int final_capture_from,
                                           bool batch_final_capture,
                                           bool snapshot_pending,
                                           int snapshot_capture_from,
                                           int snapshot_capture_to);

    // Prefill prompt tokens in chunks, return absolute committed position.
    int do_prefill(const std::vector<int32_t> & tokens, const DaemonIO & io,
                   int kv_offset = 0, int snap_slot = -1, int snap_pos = -1);

    // Generate after either a fresh prefill or a restored prefix. kv_offset is
    // the number of prompt tokens already represented by cache_ and the
    // auxiliary logits/speculative state.
    GenerateResult generate_from_state(const GenerateRequest & req,
                                       const DaemonIO & io,
                                       int kv_offset);
    bool snapshot_restore(int slot);

    // Autoregressive decode loop.
    bool do_decode(int committed, int n_gen,
                   const std::vector<int32_t> & history_prefix,
                   std::vector<int32_t> & out_tokens,
                   const DaemonIO & io,
                   const BudgetHook & budget_hook = {},
                   bool * forced_close_out = nullptr);

    bool load_model();
    bool init_hybrid_model();
    // Cluster rank: dense weights + this rank's expert shard (cold owner None).
    bool init_cluster_model();
    bool requires_monolithic_model() const;
    bool validate_prefill_mode() const;
    bool init_moe_tensor_parallel();
    bool compute_uniform_hybrid_placement(const DeepSeek4Weights & w,
                                          int max_ctx,
                                          MoeHybridPlacement & out,
                                          MoeHybridPlacement * decode_out,
                                          std::string * err) const;
    void maybe_save_routing_stats();

    std::shared_ptr<MoeHybridStorage> moe_hybrid_;
    MoeHybridPlacement                moe_placement_;
    MoeHybridPlacement                moe_decode_placement_;
    MoeHybridStreamEngine             stream_engine_;
    MoeExpertComputeRuntime            expert_runtime_;
    std::shared_ptr<MoeHybridRoutingStats> routing_stats_;
    std::string                       routing_stats_out_path_;

    // Non-null only when set_cluster() was called (one rank of a cluster).
    std::unique_ptr<Ds4ClusterRuntime> cluster_;
};

}  // namespace dflash::common
