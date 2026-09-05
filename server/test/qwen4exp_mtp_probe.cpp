// qwen4exp_mtp_probe - does this MTP module belong to this target?
//
// The module's wiring had to be inferred: upstream llama.cpp does not
// implement qwen4exp's multi-token-prediction path, and the converter only
// names the tensors. Every shape in that inference is checked against the
// target's hyperparameters when the module loads, so this tool is the first
// thing to run against a new pair -- before any draft graph is written on top
// of an assumption that a single number could have refuted.
//
//   qwen4exp_mtp_probe <target.gguf> <mtp.gguf>
//
// Exits non-zero when the two do not agree.

#include "common/gguf_shards.h"
#include "qwen4exp/qwen4exp_internal.h"
#include "qwen4exp/qwen4exp_mtp.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <string>

using namespace dflash::common;

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <target.gguf> <mtp.gguf>\n", argv[0]);
        return 2;
    }

    GgufShardSet target_shards;
    std::string err;
    if (!target_shards.open(argv[1], err)) {
        std::fprintf(stderr, "target: %s\n", err.c_str());
        return 1;
    }

    TargetWeights t;
    if (!read_qwen4exp_hparams(target_shards, t, err)) {
        std::fprintf(stderr, "target hparams: %s\n", err.c_str());
        return 1;
    }
    std::printf("target: n_embd=%d n_hc=%d hc_low_rank=%d n_expert=%d n_ff_exp=%d\n",
                t.n_embd, t.n_hc, t.hc_low_rank, t.n_expert, t.n_ff_exp);

    // CPU is enough: this checks names and shapes, and the module is small
    // enough that where its weights land does not matter.
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        std::fprintf(stderr, "no cpu backend\n");
        return 1;
    }

    Qwen4ExpMtpWeights mtp;
    const bool ok = load_qwen4exp_mtp(argv[2], t, backend, mtp);
    if (ok) {
        std::printf("  mtp layer index %d\n", mtp.layer_index);
        std::printf("  eh_proj [%lld, %lld]  enorm [%lld]  hnorm [%lld]\n",
                    (long long) mtp.eh_proj->ne[0], (long long) mtp.eh_proj->ne[1],
                    (long long) mtp.enorm->ne[0], (long long) mtp.hnorm->ne[0]);
        std::printf("  head mixer down [%lld, %lld] up [%lld, %lld]\n",
                    (long long) mtp.head_down->ne[0], (long long) mtp.head_down->ne[1],
                    (long long) mtp.head_up->ne[0], (long long) mtp.head_up->ne[1]);
        std::printf("  embedding %lld rows of %lld\n",
                    (long long) mtp.embedder.n_vocab, (long long) mtp.embedder.n_embd);
        std::printf("=> the module agrees with this target\n");
    }

    ggml_backend_free(backend);
    return ok ? 0 : 3;
}
