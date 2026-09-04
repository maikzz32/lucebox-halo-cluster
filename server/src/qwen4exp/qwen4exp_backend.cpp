#include "qwen4exp/qwen4exp_backend.h"

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
    return load_qwen4exp_gguf(cfg_.target_path, backend, out);
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

}  // namespace dflash::common
