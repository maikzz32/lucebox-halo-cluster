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
#include "ggml-cuda.h"

#include <cstdio>
#include <string>
#include <vector>

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

    // The shape check runs anywhere. The draft below does not: the same graph
    // computes on the CPU and crashes on HIP inside the server, so the probe
    // has to be able to reproduce that outside it. A third argument of "hip"
    // puts the module and the graph on the GPU.
    const bool want_hip = argc > 3 && std::string(argv[3]) == "hip";
    ggml_backend_t backend = want_hip ? ggml_backend_cuda_init(0)
                                      : ggml_backend_cpu_init();
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

    if (!ok) {
        ggml_backend_free(backend);
        return 3;
    }

    // Run one draft on the CPU. The server crashes inside the compute of this
    // same graph on HIP, and the two possibilities -- a graph this tree builds
    // wrongly, or one the GPU backend cannot run -- are told apart by whether
    // the CPU manages it. A carrier of ones is enough: what is being checked
    // is that the ops compose, not what they produce.
    {
        TargetCache cache{};
        struct ggml_init_params cip = { ggml_tensor_overhead() * 8, nullptr, true };
        cache.base_ctx = ggml_init(cip);
        cache.backend  = backend;
        cache.max_ctx  = 1;
        cache.kv_k_type = GGML_TYPE_F16;
        cache.kv_v_type = GGML_TYPE_F16;
        cache.n_seq_slots = 1;
        cache.attn_k.assign(1, ggml_new_tensor_3d(cache.base_ctx, GGML_TYPE_F16,
                                                  t.n_embd_head_k, 1, t.n_head_kv));
        cache.attn_v.assign(1, ggml_new_tensor_3d(cache.base_ctx, GGML_TYPE_F16,
                                                  t.n_embd_head_k, 1, t.n_head_kv));
        cache.base_buf = ggml_backend_alloc_ctx_tensors(cache.base_ctx, backend);

        struct ggml_init_params ip = { (size_t) 64 * 1024 * 1024, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);
        ggml_tensor * carrier = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, t.n_embd, t.n_hc, 1);
        ggml_tensor * embed   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, t.n_embd, 1);
        ggml_tensor * pos     = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
        ggml_set_input(carrier);
        ggml_set_input(embed);
        ggml_set_input(pos);

        std::printf("building the draft graph...\n");
        ggml_tensor * draft = build_qwen4exp_mtp_draft(ctx, gf, mtp, t, carrier, embed,
                                                       pos, nullptr, 0, cache);
        std::printf("  %d nodes\n", ggml_graph_n_nodes(gf));

        ggml_gallocr_t alloc =
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(alloc, gf)) {
            std::fprintf(stderr, "graph allocation failed\n");
            return 4;
        }
        std::vector<float> ones((size_t) t.n_embd * t.n_hc, 1.0f);
        ggml_backend_tensor_set(carrier, ones.data(), 0,
                                sizeof(float) * (size_t) t.n_embd * t.n_hc);
        ggml_backend_tensor_set(embed, ones.data(), 0, sizeof(float) * (size_t) t.n_embd);
        const int32_t p4[4] = {0, 0, 0, 0};
        ggml_backend_tensor_set(pos, p4, 0, sizeof(p4));

        std::printf("computing on %s...\n", want_hip ? "hip" : "cpu");
        const ggml_status st = ggml_backend_graph_compute(backend, gf);
        std::printf("  status %d\n", (int) st);
        if (st == GGML_STATUS_SUCCESS && draft) {
            int32_t id = -1;
            ggml_backend_tensor_get(draft, &id, 0, sizeof(int32_t));
            std::printf("=> the draft graph runs; it produced token %d\n", id);
        }
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
    }

    ggml_backend_free(backend);
    return 0;
}
