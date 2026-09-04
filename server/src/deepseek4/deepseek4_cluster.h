// deepseek4_cluster.h - DeepSeek4 expert-parallel cluster runtime (WP3, path 3a).
//
// Every rank loads the full dense model and only its shard of routed experts
// (owned + replicated, see cluster_expert_placement.h). Per MoE layer the
// rank evaluates its resident experts as an owner partial sum through the
// existing hybrid machinery with cold owner MoeHybridColdBackend::None, then
// ONE all-reduce (IClusterComm::allreduce_sum_f32 on the backend stream)
// makes the routed partial identical on every rank before the shared expert
// is added and HC-post runs.
//
// Path 3a: the all-reduce is a host-driven call between per-layer graph
// computes inside deepseek4_step_layer_range / eval_ds4_layer_range_hybrid_ffn.
// The fused whole-model verify/decode graphs are disabled in cluster mode
// (they cannot host a collective between layers); path 3b (in-graph op) is a
// later milestone.
//
// Shared expert: SharedExpertMode::Replicate only. Every rank computes it
// locally from the same normalized input and adds it after the all-reduce;
// the routed partial is evaluated with the shared-expert tensors removed
// from the MoeLayerDesc (the per-layer equivalent of include_shared=false).
//
// Replicated experts: expert e at batch slot t is evaluated only by rank
// (e + t) % N. Slot = token index inside the current batch (0..n_tokens-1),
// identical on every rank because routing is replicated; for decode
// (n_tokens == 1) this pins replicated expert e to rank e % N.

#pragma once

#include "deepseek4_internal.h"

#include "cluster/cluster_comm.h"
#include "cluster/cluster_config.h"
#include "cluster/cluster_expert_placement.h"
#include "cluster/cluster_telemetry.h"
#include "common/moe_hybrid_placement.h"
#include "common/moe_hybrid_storage.h"
#include "common/moe_hybrid_types.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct Ds4ClusterRuntime {
    // cfg points at cfg_storage (a copy taken by DeepSeek4Backend::set_cluster)
    // so the caller's ClusterConfig need not outlive the backend. comm is
    // borrowed and may be attached after the model was loaded; it is required
    // by the first forward.
    cluster::ClusterConfig         cfg_storage;
    const cluster::ClusterConfig * cfg  = nullptr;
    cluster::IClusterComm *        comm = nullptr;
    cluster::ClusterExpertPlacement placement;
    // Rank-local hot set (owned + replicated experts) for the hybrid storage.
    MoeHybridPlacement rank_placement;
    // [n_layer][slot_validity_n_slots][n_expert] 0/1 from
    // ClusterExpertPlacement::slot_validity for this rank; consulted for
    // slots below slot_validity_n_slots, larger slots fall back to
    // placement.slot_owner().
    std::vector<uint8_t> slot_validity_cache;
    int slot_validity_n_slots = 0;
    cluster::ClusterStepTelemetry telemetry;
    bool trace = false;
    // Path 3b: a graph node cannot return an error, so the in-graph
    // all-reduce records the first failure here and the forward fails after
    // the compute. Cleared before every graph that contains such a node.
    std::string node_error;

    // Device staging for host-resident partials (path 3a): one F32 [n] and
    // one bf16 [n] (stored as I16) tensor in a single backend buffer, grown
    // on demand. Owned here, freed in the destructor / free_scratch().
    ggml_backend_t        scratch_backend = nullptr;
    ggml_context *        scratch_ctx = nullptr;
    ggml_backend_buffer_t scratch_buf = nullptr;
    ggml_tensor *         scratch_f32 = nullptr;
    ggml_tensor *         scratch_bf16 = nullptr;
    size_t                scratch_elems = 0;

    // Resident routed-expert bytes on this rank (sum of hot buffers), for logs.
    uint64_t resident_expert_bytes = 0;

    Ds4ClusterRuntime() = default;
    ~Ds4ClusterRuntime();
    Ds4ClusterRuntime(const Ds4ClusterRuntime &) = delete;
    Ds4ClusterRuntime & operator=(const Ds4ClusterRuntime &) = delete;

    int rank() const { return cfg ? cfg->rank : 0; }
    int size() const { return cfg ? cfg->size : 1; }

    // True when this rank evaluates (layer, expert) for batch slot `slot`.
    bool evaluates(int layer, int expert, int slot) const;

    bool ensure_scratch(ggml_backend_t backend, size_t n_elems, std::string * err);
    void free_scratch();
};

// MoeHybridConfig for cluster expert storage: cold owner None, cold experts
// never materialized, hot experts materialized on the local GPU.
MoeHybridConfig ds4_cluster_moe_config(const DeepSeek4Weights & w);

// Build the N-rank placement from cfg (uniform | balanced | file) and derive
// this rank's MoeHybridPlacement and slot-validity cache. hotness_csv_path
// (from DFLASH_DS4_HOTNESS_CSV) is required for `balanced` and used by
// `uniform` when replicate_hot > 0; may be nullptr otherwise.
bool ds4_cluster_build_placement(const cluster::ClusterConfig & cfg,
                                 const DeepSeek4Weights & w,
                                 const std::string * hotness_csv_path,
                                 Ds4ClusterRuntime & rt,
                                 std::string * err);

// Build this rank's expert storage: hot set = rt.rank_placement, cold owner
// None. `w` must already be loaded with TargetLoadPlan.skip_expert_tensors
// (expert tensors present as metadata only). Logs resident expert bytes.
bool ds4_cluster_init_experts(const std::string & model_path,
                              ggml_backend_t backend,
                              const DeepSeek4Weights & w,
                              Ds4ClusterRuntime & rt,
                              MoeHybridStorage & storage,
                              std::string * err);

// Zero out every route this rank does not evaluate: selected id -> -1,
// weight -> 0. selected/weights are [route_width, n_tokens] row-major
// (token-major), slot = token index. Routing statistics must be observed
// before this call.
void ds4_cluster_mask_routes(const Ds4ClusterRuntime & rt,
                             int layer,
                             int32_t * selected,
                             float * weights,
                             int route_width,
                             int n_tokens);

// Diagnostics: sum and sum of |x| over n floats (DFLASH_CLUSTER_TRACE lines).
void ds4_cluster_checksum(const float * data, size_t n, double * sum, double * sum_abs);

// DFLASH_CLUSTER_PREFILL_SINGLE_TOKEN=1: run cluster prefill token by token
// (n_tokens == 1 chunks through the decode path) to bisect multi-token
// prefill numerics. Cached after the first call.
bool ds4_cluster_env_prefill_single_token();

// All-reduce a device-resident F32 partial [n_embd, n_tokens] in place on the
// backend stream. Stays stream-ordered (no host sync) unless rt.trace is set,
// in which case it waits with cfg->timeout_ms and logs the layer. Updates
// rt.telemetry and, when given, telemetry->cluster_allreduce_*.
bool ds4_cluster_allreduce_layer(Ds4ClusterRuntime & rt,
                                 ggml_backend_t backend,
                                 ggml_tensor * partial,
                                 int layer,
                                 DeepSeek4StepTelemetry * telemetry,
                                 std::string * err);

// Path 3b: an all-reduce as an ordinary graph node. `partial` is summed
// across the ranks on the executing backend's stream, so no host round trip
// separates the kernels that produce it from those that consume the sum.
// Returns `partial` unchanged for a single-rank runtime and nullptr when the
// tensor cannot be reduced (non-contiguous or not F32). Errors during the
// compute land in rt.node_error.
ggml_tensor * ds4_cluster_allreduce_node(ggml_context * ctx,
                                         ggml_tensor * partial,
                                         Ds4ClusterRuntime & rt);

// True when this runtime may use the fused whole-model graph (path 3b): the
// opt-in is set, a real multi-rank communicator is attached, the shared
// expert is replicated, and expert ownership is a function of the expert id
// alone. Replicated experts are owned per token slot, which the fused graph's
// per-expert owner LUT cannot express, so they fall back to path 3a.
bool ds4_cluster_fused_graph_available(const Ds4ClusterRuntime * rt);

// Path 3b (default). DFLASH_CLUSTER_NO_INGRAPH_ALLREDUCE=1 falls back to the
// host-enqueued per-layer all-reduce of path 3a, which also gives up the
// fused whole-model graph. Cached after the first call.
bool ds4_cluster_ingraph_allreduce_enabled();

// Host-resident variant used by the per-layer path: uploads to the runtime's
// device scratch, all-reduces on the backend stream, waits (deadline
// cfg->timeout_ms) and downloads the sum back into partial_host.
bool ds4_cluster_allreduce_layer_host(Ds4ClusterRuntime & rt,
                                      ggml_backend_t backend,
                                      float * partial_host,
                                      int n_embd,
                                      int n_tokens,
                                      int layer,
                                      DeepSeek4StepTelemetry * telemetry,
                                      std::string * err);

}  // namespace dflash::common
