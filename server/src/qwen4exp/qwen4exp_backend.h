#pragma once

#include "qwen35_backend.h"

namespace dflash::common {

// qwen4exp reuses the qwen35 runtime wholesale: the gated delta net, the full
// attention block, m-RoPE and the 512-expert MoE with a sigmoid-gated shared
// expert are all the same code. What it adds is the residual carrier around
// them -- four hyper-connection streams instead of one -- and that lives in
// the layer builder, not here.
//
// The configuration exposes only what the first version implements. No
// speculative decode, no expert offload, no paged serving; those come after
// the autoregressive path has a baseline to be measured against, which is the
// same order bailingmoe3 followed.
struct Qwen4ExpConfig {
    const char * model_path = nullptr;
    DevicePlacement device;
    int stream_fd = -1;
};

class Qwen4ExpBackend final : public Qwen35Backend {
public:
    explicit Qwen4ExpBackend(const Qwen4ExpConfig & cfg);

    void print_ready_banner() const override;
    bool supports_dflash_spec_decode() const override { return false; }
    bool supports_remote_draft() const override { return false; }

protected:
    bool load_target_model(ggml_backend_t backend, TargetWeights & out) override;
};

}  // namespace dflash::common
