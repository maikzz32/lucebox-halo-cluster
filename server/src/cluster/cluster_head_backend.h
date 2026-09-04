// cluster_head_backend.h - rank-0 ModelBackend decorator for cluster runs.
//
// The HTTP server talks to one ModelBackend. In a cluster that backend is
// this decorator: it owns the real DeepSeek4Backend, the control channel to
// the workers, the RCCL communicator and the HeadHooks. Before every
// generate() it broadcasts a RequestMsg so the workers run the identical
// call; every state-changing ModelBackend virtual (snapshot save/free,
// park/unpark, free_drafter) is broadcast as a BackendOpMsg before it is
// executed locally; after generate() it broadcasts RequestEnd and gathers
// one RequestReport per worker into `last_report()` for WP6.
//
// Failure policy (M1): any control-channel or collective failure broadcasts
// Abort, aborts the communicator and, unless exit_on_abort is false, ends
// the process with a non-zero status so a supervisor restarts all ranks
// together. Communicator re-initialization is M4.

#pragma once

#include "cluster/cluster_comm.h"
#include "cluster/cluster_config.h"
#include "cluster/cluster_control.h"
#include "cluster/cluster_decision_hooks.h"
#include "cluster/cluster_protocol.h"
#include "common/cluster_view.h"
#include "common/model_backend.h"
#include "deepseek4/deepseek4_backend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dflash::cluster {

// Per-request cluster telemetry merged on the head. ranks[0] is the head
// itself (only steps / ctrl_wait_us / allreduce_* from its own comm stats are
// filled), ranks[r] for r >= 1 is the worker's RequestReportMsg.
struct ClusterRequestReport {
    uint64_t request_id = 0;
    int      size = 0;
    uint32_t steps = 0;
    bool     complete = false;         // every worker reported
    std::vector<RequestReportMsg> ranks;
    uint64_t head_ctrl_wait_us = 0;
    uint64_t hash_probes = 0;
    uint64_t hash_mismatches = 0;
    int      first_mismatch_rank = -1;
    uint32_t first_mismatch_step = 0;
    std::string error;                 // non-empty when gathering failed
};

// Placement hash for HelloMsg: the inner backend's built placement
// (Ds4ClusterRuntime::placement.hash()) or 0 when no cluster runtime exists.
uint64_t backend_placement_hash(const common::DeepSeek4Backend & backend);

class ClusterHeadBackend final : public common::ModelBackend {
public:
    // `device` is the local HIP ordinal the inner backend runs on
    // (BackendArgs::device.gpu); RCCL binds the communicator to it.
    ClusterHeadBackend(std::unique_ptr<common::DeepSeek4Backend> inner,
                       const ClusterConfig & cfg,
                       std::string model_path,
                       int device);
    ~ClusterHeadBackend() override;

    ClusterHeadBackend(const ClusterHeadBackend &) = delete;
    ClusterHeadBackend & operator=(const ClusterHeadBackend &) = delete;

    // Bootstrap: listen, accept size-1 workers, distribute the RCCL unique
    // id, create the communicator, attach it to the inner backend
    // (set_cluster second call) and install the hooks. The inner backend
    // must already be init()ed with set_cluster(&cfg, nullptr) called before
    // its init() (backend_factory does both) so the Hello carries the real
    // placement hash. When cfg.selftest is set this runs run_cluster_selftest
    // and exits the process with its status (the operator asked for exactly
    // that).
    bool init();

    // ── ModelBackend ─────────────────────────────────────────────────
    void print_ready_banner() const override;
    bool park(common::ParkTarget target) override;
    bool unpark(common::ParkTarget target) override;
    bool is_target_parked() const override;
    common::GenerateResult generate_impl(const common::GenerateRequest & req,
                                         const common::DaemonIO & io) override;
    common::SeqEngine * seq_engine() override { return nullptr; }
    bool snapshot_save(int slot) override;
    void snapshot_free(int slot) override;
    bool snapshot_used(int slot) const override;
    int  snapshot_cur_pos(int slot) const override;
    common::GenerateResult restore_and_generate_impl(int slot,
                                                     const common::GenerateRequest & req,
                                                     const common::DaemonIO & io) override;
    SnapshotRef snapshot_ref(int slot) const override;
    bool snapshot_adopt(int slot, ggml_context * ctx, ggml_backend_buffer_t buf, int cur_pos,
                        int32_t last_tok) override;
    CompressResult compress(const CompressRequest & req) override;
    std::vector<CompressResult> compress_batch(const std::vector<CompressRequest> & requests) override;
    bool handle_compress(const std::string & line, const common::DaemonIO & io) override;
    void free_drafter() override;
    bool try_handle_command(const std::string & line, const common::DaemonIO & io) override;
    bool supports_dflash_spec_decode() const override;
    common::DFlashTarget * dflash_target() override;
    void release_scratch() override;
    bool supports_remote_draft() const override;
    bool supports_kvflash() const override;
    bool supports_mixed_backend_layer_split() const override;
    bool set_routing_collector(common::MoeRoutingCollector * collector) override;
    const common::MoeHybridRoutingStats * get_routing_stats() const override;
    bool spark_wants_bootstrap() const override;
    bool spark_bootstrap_finalize(const std::string & profile_path) override;
    void shutdown() override;

    // ── Cluster introspection (WP6) ──────────────────────────────────
    // WP6: plain snapshots for the HTTP layer (common/cluster_view.h).
    // telemetry describes the request that just finished and is read on the
    // request thread; props is constant after init().
    bool cluster_request_telemetry(common::ClusterTelemetryView & out) const override;
    bool cluster_props(common::ClusterPropsView & out) const override;

    const ClusterRequestReport & last_report() const { return last_report_; }
    const ClusterConfig & config() const { return cfg_; }
    const std::vector<HelloMsg> & worker_identities() const { return hellos_; }
    IClusterComm * comm() { return comm_.get(); }
    bool fatal() const { return fatal_; }
    void set_exit_on_abort(bool v) { exit_on_abort_ = v; }

    // Maps a GenerateRequest (plus optional restore slot) onto the wire
    // descriptor. Exposed for the unit test; see the .cpp for the exact
    // field mapping and the fields the protocol cannot carry yet.
    static RequestMsg build_request_msg(uint64_t request_id, const common::GenerateRequest & req,
                                        int restore_slot, int restore_kv_offset,
                                        bool spec_available);

private:
    common::GenerateResult run_request(const common::GenerateRequest & req,
                                       const common::DaemonIO & io,
                                       int restore_slot);
    bool broadcast_op(BackendOpKind kind, std::vector<int64_t> args);
    bool gather_reports(uint64_t request_id, uint32_t steps);
    // Broadcast Abort, abort the communicator, mark fatal, maybe exit.
    void fail_cluster(const std::string & reason, int code, uint64_t request_id);
    bool spec_available() const;

    std::unique_ptr<common::DeepSeek4Backend> inner_;
    ClusterConfig                     cfg_;
    std::string                       model_path_;
    int                               device_ = 0;
    // High-water mark of device memory in use on rank 0, sampled at request
    // ends; the workers keep the same mark and ship it in RequestReport.
    uint64_t                          peak_device_bytes_ = 0;
    uint64_t                          head_compute_us_ = 0;
    ClusterHeadControl                control_;
    std::unique_ptr<IClusterComm>     comm_;
    std::unique_ptr<HeadHooks>        hooks_;
    std::vector<HelloMsg>             hellos_;
    HelloMsg                          identity_;
    uint64_t                          next_request_id_ = 1;
    ClusterRequestReport              last_report_;
    bool                              initialized_ = false;
    bool                              fatal_ = false;
    bool                              exit_on_abort_ = true;
    bool                              shut_down_ = false;
};

}  // namespace dflash::cluster
