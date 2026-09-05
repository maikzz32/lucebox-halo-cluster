#include "qwen4exp/qwen4exp_mtp_runtime.h"

#include "qwen4exp/qwen4exp_graph.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace dflash::common {

namespace {

// The head's block is a full-attention one, so it needs somewhere to put K and
// V. One token's worth: the measurement resets it every step (see the header
// for why that is enough to tell a correct wiring from a wrong one).
bool alloc_one_token_cache(const TargetWeights & t, ggml_backend_t backend,
                           TargetCache & c) {
    const int64_t head_dim = t.n_embd_head_k;
    struct ggml_init_params ip = { ggml_tensor_overhead() * 8, nullptr, true };
    c.base_ctx = ggml_init(ip);
    if (!c.base_ctx) return false;
    c.backend    = backend;
    c.max_ctx    = 1;
    c.cur_pos    = 0;
    c.kv_k_type  = GGML_TYPE_F16;
    c.kv_v_type  = GGML_TYPE_F16;
    c.n_seq_slots = 1;

    c.attn_k.assign(1, nullptr);
    c.attn_v.assign(1, nullptr);
    c.attn_k[0] = ggml_new_tensor_3d(c.base_ctx, GGML_TYPE_F16, head_dim, 1, t.n_head_kv);
    c.attn_v[0] = ggml_new_tensor_3d(c.base_ctx, GGML_TYPE_F16, head_dim, 1, t.n_head_kv);
    if (!c.attn_k[0] || !c.attn_v[0]) return false;
    ggml_set_name(c.attn_k[0], "mtp_cache_k");
    ggml_set_name(c.attn_v[0], "mtp_cache_v");

    c.base_buf = ggml_backend_alloc_ctx_tensors(c.base_ctx, backend);
    return c.base_buf != nullptr;
}

}  // namespace

bool qwen4exp_mtp_open(const TargetWeights & target,
                       ggml_backend_t backend,
                       int max_ctx,
                       Qwen4ExpMtpRuntime & rt) {
    (void) max_ctx;
    const char * path = std::getenv("DFLASH_QWEN4EXP_MTP");
    if (!path || !*path) {
        return true;                    // nothing asked for
    }
    if (target.n_hc <= 1) {
        std::fprintf(stderr, "[qwen4exp-mtp] the target is not hyper-connected\n");
        return false;
    }
    if (!load_qwen4exp_mtp(path, target, backend, rt.w)) {
        return false;
    }
    if (!alloc_one_token_cache(target, backend, rt.cache)) {
        std::fprintf(stderr, "[qwen4exp-mtp] cache allocation failed\n");
        return false;
    }
    rt.backend = backend;
    std::fprintf(stderr,
                 "[qwen4exp-mtp] measuring acceptance; the head drafts the token "
                 "after next and the next step scores it\n");
    return true;
}

void qwen4exp_mtp_draft_step(Qwen4ExpMtpRuntime & rt,
                             const TargetWeights & t,
                             const float * carrier,
                             int32_t next_token) {
    if (!rt.ready() || !carrier) return;
    // Checkpoints, because a segfault inside a container leaves no trace and
    // guessing at which of eight steps failed is slower than printing them.
    static const bool trace = std::getenv("DFLASH_QWEN4EXP_MTP_TRACE") != nullptr;
    auto mark = [&](const char * where) {
        if (trace) { std::fprintf(stderr, "[qwen4exp-mtp] at %s\n", where); std::fflush(stderr); }
    };
    mark("enter");

    const int64_t n_embd = t.n_embd;
    const int64_t n_hc   = t.n_hc;

    // A graph per step. The head is one block, so building it costs far less
    // than the target step it rides along with, and a measurement that
    // rebuilds is easier to trust than one that caches.
    const size_t mem = (size_t) 64 * 1024 * 1024;
    struct ggml_init_params ip = { mem, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return;
    mark("graph");
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    ggml_tensor * carrier_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_hc, 1);
    ggml_tensor * embed_t   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);
    ggml_tensor * pos_t     = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
    ggml_set_input(carrier_t);
    ggml_set_input(embed_t);
    ggml_set_input(pos_t);

    mark("build");
    ggml_tensor * draft = build_qwen4exp_mtp_draft(ctx, gf, rt.w, t, carrier_t,
                                                   embed_t, pos_t,
                                                   /*attn_mask=*/nullptr,
                                                   /*kv_start=*/0, rt.cache);
    if (!draft) { ggml_free(ctx); return; }

    mark("alloc");
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(rt.backend));
    if (!alloc || !ggml_gallocr_alloc_graph(alloc, gf)) {
        if (alloc) ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return;
    }

    mark("embed");
    std::vector<float> embed((size_t) n_embd);
    if (!rt.w.embedder.embed(&next_token, 1, embed.data())) {
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return;
    }
    ggml_backend_tensor_set(carrier_t, carrier, 0, sizeof(float) * (size_t) (n_embd * n_hc));
    ggml_backend_tensor_set(embed_t, embed.data(), 0, sizeof(float) * (size_t) n_embd);
    const int32_t pos4[4] = { 0, 0, 0, 0 };
    ggml_backend_tensor_set(pos_t, pos4, 0, sizeof(pos4));

    mark("compute");
    if (ggml_backend_graph_compute(rt.backend, gf) == GGML_STATUS_SUCCESS) {
        int32_t id = -1;
        ggml_backend_tensor_get(draft, &id, 0, sizeof(int32_t));
        rt.pending = id;
    }
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
}

void qwen4exp_mtp_score(Qwen4ExpMtpRuntime & rt, int32_t actual_token) {
    if (!rt.ready() || rt.pending < 0) return;
    rt.drafted++;
    rt.matched += (rt.pending == actual_token) ? 1 : 0;
    rt.pending = -1;
}

void qwen4exp_mtp_report(Qwen4ExpMtpRuntime & rt) {
    if (!rt.ready() || rt.drafted == 0) return;
    std::fprintf(stderr,
                 "[qwen4exp-mtp] acceptance %llu/%llu = %.1f%%\n",
                 (unsigned long long) rt.matched, (unsigned long long) rt.drafted,
                 100.0 * rt.acceptance());
    rt.drafted = 0;
    rt.matched = 0;
    rt.pending = -1;
}

}  // namespace dflash::common
