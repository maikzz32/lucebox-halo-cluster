#include "qwen4exp/qwen4exp_mtp_runtime.h"

#include "qwen4exp/qwen4exp_graph.h"
#include "qwen4exp/qwen4exp_probe.h"

#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <csignal>
#include <cstdio>
#include <unistd.h>
#include <cstdlib>
#include <vector>

namespace dflash::common {

extern "C" void ggml_print_backtrace(void);

namespace {

// A segfault inside a release container leaves nothing behind, and this one
// only appears when the head is on. ggml already knows how to print a
// backtrace -- it does it for its own assertions -- so borrowing that is
// cheaper than a debug build, and the handler is installed only when the head
// is, so nothing else in the process changes.
void install_backtrace_on_segv() {
    static bool done = false;
    if (done) return;
    done = true;
    struct sigaction sa {};
    sa.sa_handler = [](int sig) {
        std::fprintf(stderr, "[qwen4exp-mtp] signal %d, backtrace follows\n", sig);
        std::fflush(stderr);
        ggml_print_backtrace();
        std::fflush(stderr);
        _exit(139);
    };
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESETHAND;
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGBUS,  &sa, nullptr);
    sigaction(SIGABRT, &sa, nullptr);
}


// The head's block is a full-attention one, so it needs somewhere to put K and
// V -- its own, over the whole context, advanced once per draft. A single
// visible token was enough to show the wiring is not noise; it is not enough
// to show what the head can do.
bool alloc_cache(const TargetWeights & t, ggml_backend_t backend,
                 int max_ctx, TargetCache & c) {
    const int64_t head_dim = t.n_embd_head_k;
    struct ggml_init_params ip = { ggml_tensor_overhead() * 8, nullptr, true };
    c.base_ctx = ggml_init(ip);
    if (!c.base_ctx) return false;
    c.backend    = backend;
    c.max_ctx    = max_ctx;
    c.cur_pos    = 0;
    c.kv_k_type  = GGML_TYPE_F16;
    c.kv_v_type  = GGML_TYPE_F16;
    c.n_seq_slots = 1;

    c.attn_k.assign(1, nullptr);
    c.attn_v.assign(1, nullptr);
    c.attn_k[0] = ggml_new_tensor_3d(c.base_ctx, GGML_TYPE_F16, head_dim, max_ctx, t.n_head_kv);
    c.attn_v[0] = ggml_new_tensor_3d(c.base_ctx, GGML_TYPE_F16, head_dim, max_ctx, t.n_head_kv);
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
    const char * path = std::getenv("DFLASH_QWEN4EXP_MTP");
    if (!path || !*path) {
        return true;                    // nothing asked for
    }
    if (target.n_hc <= 1) {
        std::fprintf(stderr, "[qwen4exp-mtp] the target is not hyper-connected\n");
        return false;
    }
    // Its own backend, not the target's.
    //
    // The same draft graph computes standalone on both CPU and HIP and dies
    // inside the server, which leaves the target's backend state -- its
    // context, its memory pool, its stream -- as the difference. A second
    // context removes all three at once, and the head only ever needs the
    // carrier, which crosses through host memory anyway.
    // DFLASH_QWEN4EXP_MTP_CPU=1 puts it on the CPU instead. That is slow --
    // one block and a 248320-row head per token -- and it is the right trade
    // for what this stage is: the acceptance rate over a hundred tokens takes
    // seconds either way, and the CPU path is the one that demonstrably
    // computes this graph. A head that has to earn its place in the decode
    // loop can move to the GPU once the number says it is worth the move.
    const char * cpu_env = std::getenv("DFLASH_QWEN4EXP_MTP_CPU");
    const bool on_cpu = cpu_env && std::atoi(cpu_env) == 1;
    ggml_backend_t own = on_cpu ? ggml_backend_cpu_init() : ggml_backend_cuda_init(0);
    if (!own) {
        std::fprintf(stderr, "[qwen4exp-mtp] could not open a backend of its own\n");
        return false;
    }
    if (!load_qwen4exp_mtp(path, target, own, rt.w)) {
        ggml_backend_free(own);
        return false;
    }
    rt.max_ctx = max_ctx > 0 ? max_ctx : 4096;
    if (!alloc_cache(target, own, rt.max_ctx, rt.cache)) {
        std::fprintf(stderr, "[qwen4exp-mtp] cache allocation failed\n");
        ggml_backend_free(own);
        return false;
    }
    install_backtrace_on_segv();
    rt.backend = own;
    rt.owns_backend = true;
    std::fprintf(stderr,
                 "[qwen4exp-mtp] measuring acceptance; the head drafts the token "
                 "after next and the next step scores it\n");
    return true;
}

void qwen4exp_mtp_draft_step(Qwen4ExpMtpRuntime & rt,
                             const TargetWeights & t,
                             const float * carrier,
                             int32_t next_token,
                             int abs_pos) {
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

    // Built once. The shape never changes -- one token, one carrier -- so
    // there is nothing to rebuild, and reusing the allocator keeps the decode
    // loop from creating and freeing a backend buffer beside the target's.
    if (rt.pos >= rt.max_ctx) return;      // the head has run out of its own cache
    if (rt.ctx) {                          // kv_start is baked in; rebuild for it
        ggml_free(rt.ctx);
        rt.ctx = nullptr;
    }
    {
        mark("build");
        struct ggml_init_params ip = { (size_t) 64 * 1024 * 1024, nullptr, true };
        rt.ctx = ggml_init(ip);
        if (!rt.ctx) return;
        rt.gf = ggml_new_graph_custom(rt.ctx, 16384, false);
        rt.in_carrier = ggml_new_tensor_3d(rt.ctx, GGML_TYPE_F32, n_embd, n_hc, 1);
        rt.in_embed   = ggml_new_tensor_2d(rt.ctx, GGML_TYPE_F32, n_embd, 1);
        rt.in_pos     = ggml_new_tensor_1d(rt.ctx, GGML_TYPE_I32, 4);
        ggml_set_input(rt.in_carrier);
        ggml_set_input(rt.in_embed);
        ggml_set_input(rt.in_pos);
        rt.out_draft = build_qwen4exp_mtp_draft(rt.ctx, rt.gf, rt.w, t,
                                                rt.in_carrier, rt.in_embed, rt.in_pos,
                                                /*attn_mask=*/nullptr, rt.pos,
                                                rt.cache);
        if (!rt.out_draft) { ggml_free(rt.ctx); rt.ctx = nullptr; return; }
        if (!rt.alloc) {
            rt.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(rt.backend));
        }
        if (!rt.alloc || !ggml_gallocr_alloc_graph(rt.alloc, rt.gf)) {
            std::fprintf(stderr, "[qwen4exp-mtp] draft graph allocation failed\n");
            if (rt.alloc) ggml_gallocr_free(rt.alloc);
            rt.alloc = nullptr;
            ggml_free(rt.ctx);
            rt.ctx = nullptr;
            return;
        }
        if (rt.pos == 0) {
            std::fprintf(stderr, "[qwen4exp-mtp] draft graph: %d nodes\n",
                         ggml_graph_n_nodes(rt.gf));
        }
    }

    mark("embed");
    std::vector<float> embed((size_t) n_embd);
    if (!rt.w.embedder.embed(&next_token, 1, embed.data())) return;
    ggml_backend_tensor_set(rt.in_carrier, carrier, 0,
                            sizeof(float) * (size_t) (n_embd * n_hc));
    ggml_backend_tensor_set(rt.in_embed, embed.data(), 0, sizeof(float) * (size_t) n_embd);
    // The head models the same sequence as the target, so it rotates at the
    // target's absolute position. Numbering its own drafts from zero shifts
    // every rotary phase and quietly degrades its attention.
    const int32_t pos4[4] = { abs_pos, abs_pos, abs_pos, 0 };
    ggml_backend_tensor_set(rt.in_pos, pos4, 0, sizeof(pos4));

    mark("compute");
    if (ggml_backend_graph_compute(rt.backend, rt.gf) == GGML_STATUS_SUCCESS) {
        int32_t id = -1;
        ggml_backend_tensor_get(rt.out_draft, &id, 0, sizeof(int32_t));
        rt.pending = id;
    }
    qwen4exp_probe_report();
    rt.pos++;
    mark("done");
}

void qwen4exp_mtp_score(Qwen4ExpMtpRuntime & rt, int32_t actual_token) {
    if (!rt.ready() || rt.pending < 0) return;
    rt.drafted++;
    rt.matched += (rt.pending == actual_token) ? 1 : 0;
    rt.pending = -1;
    // Every 32, so the number is visible while a request is still running
    // rather than only at its end.
    if (rt.drafted % 32 == 0) {
        std::fprintf(stderr, "[qwen4exp-mtp] acceptance %llu/%llu = %.1f%%\n",
                     (unsigned long long) rt.matched, (unsigned long long) rt.drafted,
                     100.0 * rt.acceptance());
    }
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
