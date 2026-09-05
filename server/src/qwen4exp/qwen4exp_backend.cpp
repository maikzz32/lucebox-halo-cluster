#include "qwen4exp/qwen4exp_backend.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#include "qwen4exp/qwen4exp_internal.h"

#include <cstdio>

namespace dflash::common {
namespace {

Qwen35Config make_qwen_runtime_config(const Qwen4ExpConfig & cfg) {
    Qwen35Config runtime;
    runtime.target_path = cfg.model_path;
    runtime.device = cfg.device;
    runtime.stream_fd = cfg.stream_fd;
    // First version: the ordinary contiguous KV cache and the single-sequence
    // autoregressive loop, with nothing layered on top that would have to be
    // re-validated before the numerics are trusted.
    runtime.paged_attention = false;
    runtime.max_concurrency = 1;
    return runtime;
}

}  // namespace

Qwen4ExpBackend::Qwen4ExpBackend(const Qwen4ExpConfig & cfg)
    : Qwen35Backend(make_qwen_runtime_config(cfg)) {}

bool Qwen4ExpBackend::load_target_model(ggml_backend_t backend, TargetWeights & out) {
    // Only when this rank actually holds a slice: a single-rank run must build
    // the same graph it always did.
    out.cluster = cluster_.sharded() ? &cluster_ : nullptr;
    if (!load_qwen4exp_gguf(cfg_.target_path, backend, out, &cluster_)) {
        return false;
    }
    // The head is opened against the weights that were just read, so its
    // shapes are checked against this target rather than a configuration.
    if (!qwen4exp_mtp_open(out, backend, out.n_vocab, mtp_)) {
        return false;
    }
    return true;
}



void Qwen4ExpBackend::print_ready_banner() const {
    const TargetWeights & w = target_weights();
    const int full = w.full_attention_interval > 0
                         ? w.n_layer / w.full_attention_interval : 0;
    std::printf(
        "[qwen4exp-daemon] ready layers=%d attn=%d dn=%d experts=%d/%d "
        "hc=%d/%d ple_layer=%d ctx=%d\n",
        w.n_layer, full, w.n_layer - full,
        w.n_expert_used, w.n_expert,
        w.n_hc, w.hc_low_rank, w.ple_layer,
        cfg_.device.max_ctx);
    std::fflush(stdout);
}

void Qwen4ExpBackend::cluster_set_hooks(cluster::Ds4ClusterHooks * hooks) {
    // Stored rather than forwarded: the lockstep decisions are consumed by the
    // generate path once rank-0-decides is wired, and qwen35 has no hook slot
    // of its own to forward them to.
    cluster_.hooks = hooks;
}

bool Qwen4ExpBackend::cluster_attach(const cluster::ClusterConfig * cfg,
                                     cluster::IClusterComm * comm) {
    if (!cfg || !cfg->enabled()) {
        return true;                       // not a cluster run
    }
    const std::string err = cfg->validate();
    if (!err.empty()) {
        std::fprintf(stderr, "[qwen4exp-cluster] %s\n", err.c_str());
        return false;
    }
    // The first call defines the placement and must happen before init(); the
    // second only attaches the transport, so its placement-defining fields
    // have to match what the rank already loaded.
    if (cluster_.cfg && (cluster_.cfg_storage.size != cfg->size ||
                         cluster_.cfg_storage.rank != cfg->rank)) {
        std::fprintf(stderr,
                     "[qwen4exp-cluster] refusing to change rank %d/%d to %d/%d "
                     "after the shard was loaded\n",
                     cluster_.cfg_storage.rank, cluster_.cfg_storage.size,
                     cfg->rank, cfg->size);
        return false;
    }
    cluster_.cfg_storage = *cfg;
    cluster_.cfg  = &cluster_.cfg_storage;
    cluster_.comm = comm;
    cluster_.trace = std::getenv("DFLASH_CLUSTER_TRACE") != nullptr;
    std::fprintf(stderr, "[qwen4exp-cluster] rank %d/%d %s\n",
                 cfg->rank, cfg->size,
                 comm ? "communicator attached" : "placement only");

    // Opt-in, and it fails soft: a fabric that cannot come up leaves the RCCL
    // path exactly as it was rather than taking the run down.
    if (comm && cfg->size > 1 &&
        std::getenv("DFLASH_CLUSTER_FAST_REDUCE") != nullptr && !cluster_.fast) {
        cluster::FastReduce::Config fc;
        fc.rank = cfg->rank;
        fc.size = cfg->size;
        fc.hca  = cfg->ib_hca;
        fc.gid_index = cfg->gid_index > 0 ? cfg->gid_index : 1;
        fc.bootstrap_host = cfg->head_host;
        fc.bootstrap_port = cfg->head_port + 100;
        auto fast = std::make_unique<cluster::FastReduce>();
        std::string ferr;
        if (fast->init(fc, &ferr)) {
            cluster_.fast = std::move(fast);
            std::fprintf(stderr,
                         "[qwen4exp-cluster] fast reduce up on %s (gid %d), "
                         "bootstrap :%d\n",
                         fc.hca.c_str(), fc.gid_index, fc.bootstrap_port);
        } else {
            std::fprintf(stderr,
                         "[qwen4exp-cluster] fast reduce unavailable, keeping RCCL: %s\n",
                         ferr.c_str());
        }
    }
    return true;
}

}  // namespace dflash::common
