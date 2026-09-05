#include "qwen4exp/qwen4exp_mtp.h"

#include "common/gguf_shards.h"
#include "qwen4exp/qwen4exp_graph.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace dflash::common {

Qwen4ExpMtpWeights::~Qwen4ExpMtpWeights() {
    if (buf) ggml_backend_buffer_free(buf);
    if (ctx) ggml_free(ctx);
}

namespace {

struct Bind {
    ggml_tensor *   dst = nullptr;
    GgufShardTensor src;
};

// Bind by name and check the shape against what the target's hyperparameters
// require. A module built for a different configuration has to fail here: its
// only other symptom is a draft that is never accepted, which looks exactly
// like a wiring mistake and would send the search in the wrong direction.
struct MtpBinder {
    const GgufShardSet & shards;
    ggml_context *       ctx;
    std::vector<Bind> &  out;
    std::string &        err;
    bool ok = true;

    ggml_tensor * take(const std::string & name, bool required,
                       int64_t want0 = -1, int64_t want1 = -1) {
        if (!ok) return nullptr;
        GgufShardTensor src;
        std::string e;
        if (!shards.find(name.c_str(), src, e) || !src.meta) {
            if (!required) return nullptr;
            err = "qwen4exp-mtp: " + e;
            ok = false;
            return nullptr;
        }
        if ((want0 >= 0 && src.meta->ne[0] != want0) ||
            (want1 >= 0 && src.meta->ne[1] != want1)) {
            char b[256];
            std::snprintf(b, sizeof(b),
                          "qwen4exp-mtp: %s is [%lld,%lld], expected [%lld,%lld]",
                          name.c_str(), (long long) src.meta->ne[0],
                          (long long) src.meta->ne[1], (long long) want0,
                          (long long) want1);
            err = b;
            ok = false;
            return nullptr;
        }
        ggml_tensor * dst = ggml_dup_tensor(ctx, src.meta);
        if (!dst) {
            err = "qwen4exp-mtp: metadata context exhausted";
            ok = false;
            return nullptr;
        }
        ggml_set_name(dst, name.c_str());
        out.push_back(Bind{dst, src});
        return dst;
    }
};

size_t align256(size_t v) { return (v + 255) & ~(size_t) 255; }

}  // namespace

bool load_qwen4exp_mtp(const std::string & path,
                       const TargetWeights & t,
                       ggml_backend_t backend,
                       Qwen4ExpMtpWeights & out) {
    GgufShardSet shards;
    std::string err;
    if (!shards.open(path, err)) {
        std::fprintf(stderr, "[qwen4exp-mtp] %s\n", err.c_str());
        return false;
    }

    // The module counts the layers it predicts for and itself, so its own
    // layer is the last index.
    const gguf_context * g = shards.meta();
    const int64_t bc_id = gguf_find_key(g, "qwen4exp.block_count");
    if (bc_id < 0) {
        std::fprintf(stderr, "[qwen4exp-mtp] no qwen4exp.block_count\n");
        return false;
    }
    out.layer_index = (int) gguf_get_val_u32(g, bc_id) - 1;

    struct ggml_init_params ip = { ggml_tensor_overhead() * 64, nullptr, true };
    out.ctx = ggml_init(ip);
    if (!out.ctx) {
        std::fprintf(stderr, "[qwen4exp-mtp] metadata context\n");
        return false;
    }

    std::vector<Bind> binds;
    MtpBinder b{shards, out.ctx, binds, err};
    const std::string p = "blk." + std::to_string(out.layer_index) + ".";
    const int64_t hc_dim = (int64_t) t.n_hc * t.n_embd;

    out.enorm     = b.take(p + "nextn.enorm.weight",        true, t.n_embd);
    out.hnorm     = b.take(p + "nextn.hnorm.weight",        true, hc_dim);
    out.eh_proj   = b.take(p + "nextn.eh_proj.weight",      true, 2 * t.n_embd, t.n_embd);
    out.head_norm = b.take(p + "nextn.hc_head_norm.weight", true, hc_dim);
    out.head_down = b.take(p + "nextn.hc_head_down.weight", true, hc_dim, t.hc_low_rank);
    out.head_up   = b.take(p + "nextn.hc_head_up.weight",   true, t.hc_low_rank, hc_dim);
    out.output    = b.take("output.weight",                 true, t.n_embd);

    TargetLayer & L = out.layer;
    L.hc_attn_norm   = b.take(p + "hc_attn_norm.weight",   true, hc_dim);
    L.hc_attn_down   = b.take(p + "hc_attn_down.weight",   true, hc_dim, t.hc_low_rank);
    L.hc_attn_up     = b.take(p + "hc_attn_up.weight",     true, t.hc_low_rank, hc_dim);
    L.hc_attn_inject = b.take(p + "hc_attn_inject.weight", true, hc_dim, t.n_hc);
    L.hc_ffn_norm    = b.take(p + "hc_ffn_norm.weight",    true, hc_dim);
    L.hc_ffn_down    = b.take(p + "hc_ffn_down.weight",    true, hc_dim, t.hc_low_rank);
    L.hc_ffn_up      = b.take(p + "hc_ffn_up.weight",      true, t.hc_low_rank, hc_dim);
    L.hc_ffn_inject  = b.take(p + "hc_ffn_inject.weight",  true, hc_dim, t.n_hc);

    // The module's block is a full-attention one: it carries attn_q/k/v/output
    // rather than the delta net's fused projection.
    L.wq     = b.take(p + "attn_q.weight",      true, t.n_embd);
    L.wk     = b.take(p + "attn_k.weight",      true, t.n_embd);
    L.wv     = b.take(p + "attn_v.weight",      true, t.n_embd);
    L.wo     = b.take(p + "attn_output.weight", true);
    L.q_norm = b.take(p + "attn_q_norm.weight", true);
    L.k_norm = b.take(p + "attn_k_norm.weight", true);

    L.ffn_gate_inp       = b.take(p + "ffn_gate_inp.weight",       true, t.n_embd, t.n_expert);
    L.ffn_gate_inp_shexp = b.take(p + "ffn_gate_inp_shexp.weight", false);
    L.ffn_gate_exps      = b.take(p + "ffn_gate_exps.weight",      true);
    L.ffn_up_exps        = b.take(p + "ffn_up_exps.weight",        true);
    L.ffn_down_exps      = b.take(p + "ffn_down_exps.weight",      true);
    L.ffn_gate_shexp     = b.take(p + "ffn_gate_shexp.weight",     false);
    L.ffn_up_shexp       = b.take(p + "ffn_up_shexp.weight",       false);
    L.ffn_down_shexp     = b.take(p + "ffn_down_shexp.weight",     false);

    if (!b.ok) {
        std::fprintf(stderr, "[qwen4exp-mtp] %s\n", err.c_str());
        ggml_free(out.ctx);
        out.ctx = nullptr;
        return false;
    }

    // The embedding stays on the host, as the target's does: one row per token
    // is a rounding error of bandwidth against a table of over a gigabyte.
    GgufShardTensor tok;
    if (!shards.find("token_embd.weight", tok, err) || !tok.meta) {
        std::fprintf(stderr, "[qwen4exp-mtp] token_embd: %s\n", err.c_str());
        ggml_free(out.ctx);
        out.ctx = nullptr;
        return false;
    }
    out.embedder.tok_embd_owned.resize(tok.size);
    std::memcpy(out.embedder.tok_embd_owned.data(), tok.data, tok.size);
    out.embedder.tok_embd_bytes = out.embedder.tok_embd_owned.data();
    out.embedder.tok_embd_type  = tok.meta->type;
    out.embedder.n_embd         = t.n_embd;
    out.embedder.n_vocab        = tok.meta->ne[1];
    out.embedder.row_bytes      = tok.size / (size_t) out.embedder.n_vocab;

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    size_t total = 0;
    for (const Bind & bd : binds) {
        total = align256(total) + ggml_backend_buft_get_alloc_size(buft, bd.dst);
    }
    out.buf = ggml_backend_alloc_buffer(backend, total);
    if (!out.buf) {
        std::fprintf(stderr, "[qwen4exp-mtp] buffer of %.2f GiB refused\n",
                     (double) total / (1024.0 * 1024.0 * 1024.0));
        ggml_free(out.ctx);
        out.ctx = nullptr;
        return false;
    }
    ggml_backend_buffer_set_usage(out.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    char * base = (char *) ggml_backend_buffer_get_base(out.buf);
    size_t off = 0;
    for (const Bind & bd : binds) {
        off = align256(off);
        if (ggml_backend_tensor_alloc(out.buf, bd.dst, base + off) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "[qwen4exp-mtp] allocation failed for %s\n",
                         ggml_get_name(bd.dst));
            return false;
        }
        ggml_backend_tensor_set(bd.dst, bd.src.data, 0, bd.src.size);
        off += ggml_backend_buft_get_alloc_size(buft, bd.dst);
    }

    std::fprintf(stderr,
                 "[qwen4exp-mtp] layer %d loaded, %zu tensors, %.2f GiB on device\n",
                 out.layer_index, binds.size(),
                 (double) total / (1024.0 * 1024.0 * 1024.0));
    return true;
}

ggml_tensor * build_qwen4exp_mtp_draft(ggml_context * ctx,
                                       ggml_cgraph *  gf,
                                       const Qwen4ExpMtpWeights & w,
                                       const TargetWeights & t,
                                       ggml_tensor * carrier,
                                       ggml_tensor * embed_next,
                                       ggml_tensor * positions,
                                       ggml_tensor * attn_mask,
                                       int           kv_start,
                                       TargetCache & cache) {
    const int64_t nt     = carrier->ne[2];
    const int64_t hc_dim = (int64_t) t.n_hc * t.n_embd;

    static const int variant = []() {
        const char * v = std::getenv("DFLASH_QWEN4EXP_MTP_VARIANT");
        return v ? std::atoi(v) : 0;
    }();

    // DFLASH_QWEN4EXP_MTP_UNFOLDED=1 reads enorm and hnorm as (1 + w) rather
    // than as the gamma itself. The target's hyper-connection norms average
    // about one, which is what a converter that folded the plus-one leaves
    // behind; these two average 0.24 and 0.67, which is what it leaves behind
    // when it did not. Getting this wrong scales the projection's input by a
    // factor of five, and the acceptance rate is what says which it is.
    static const bool unfolded = []() {
        const char * v = std::getenv("DFLASH_QWEN4EXP_MTP_UNFOLDED");
        return v && std::atoi(v) == 1;
    }();
    auto gamma = [&](ggml_tensor * x, ggml_tensor * g) {
        return unfolded ? ggml_add(ctx, x, ggml_mul(ctx, x, g))
                        : ggml_mul(ctx, x, g);
    };

    // The token that was just sampled, normalised on its own.
    ggml_tensor * e = gamma(ggml_rms_norm(ctx, embed_next, t.rms_eps), w.enorm);

    // The carrier, normalised the way every qwen4exp norm treats it: the
    // reduction runs over one stream and the flat gamma scales all of them.
    // Variant 3 reduces over the flat n_hc*n_embd instead of over one stream.
    // The per-stream reading is what the carrier's own norms use, proven from
    // the weights; hnorm belongs to the head's glue rather than to that
    // convention, so it does not have to follow it.
    ggml_tensor * h = variant == 3
        ? ggml_rms_norm(ctx, ggml_reshape_2d(ctx, carrier, hc_dim, nt), t.rms_eps)
        : ggml_reshape_2d(ctx, ggml_rms_norm(ctx, carrier, t.rms_eps), hc_dim, nt);
    h = gamma(h, w.hnorm);
    h = ggml_reshape_3d(ctx, h, t.n_embd, t.n_hc, nt);

    // eh_proj is [2*n_embd, n_embd], so it cannot see the whole carrier at
    // once. It sees one stream at a time with the embedding alongside, and its
    // output is that stream of the carrier the layer runs on. Broadcasting the
    // embedding across the streams and concatenating on the feature axis does
    // all four in one matmul.
    // (variant is declared above, next to the norms it also selects between)
    // Which of these is right is decided by the acceptance rate, not by
    // reading: upstream llama.cpp has no qwen4exp MTP and vLLM's Qwen3-Next
    // one has a single-stream hidden state where this has four. All three
    // agree with the shapes; only one agrees with the target.
    ggml_tensor * state = nullptr;
    if (variant == 2) {
        // The carrier collapsed to one vector first, the projection applied
        // once, and the result seeded across the streams -- the reading in
        // which eh_proj really is the reference's fc over one hidden state.
        ggml_tensor * hm = ggml_cont(ctx, ggml_mean(ctx, ggml_cont(ctx,
            ggml_permute(ctx, h, 1, 0, 2, 3))));            // [1, n_embd, T]
        hm = ggml_reshape_2d(ctx, hm, t.n_embd, nt);
        ggml_tensor * eh = ggml_concat(ctx, e, hm, 0);      // [2*n_embd, T]
        ggml_tensor * z  = ggml_cont(ctx, ggml_mul_mat(ctx, w.eh_proj, eh));
        state = qwen4exp_hc_init(ctx, z, t.n_embd, t.n_hc);
    } else {
        ggml_tensor * e3 = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, e, t.n_embd, 1, nt),
                                          t.n_embd, t.n_hc, nt, 1);
        // variant 1 swaps the halves: the reference concatenates the embedding
        // first, but the reference is a different architecture's module.
        ggml_tensor * eh = variant == 1 ? ggml_concat(ctx, h, e3, 0)
                                        : ggml_concat(ctx, e3, h, 0);
        state = ggml_cont(ctx, ggml_mul_mat(ctx, w.eh_proj, eh));
        // Variant 4 treats the projection as a residual on the carrier rather
        // than as a replacement for it.
        if (variant == 4) {
            state = ggml_add(ctx, state, carrier);
        }
    }

    // One ordinary qwen4exp block, then this head's own output mixer -- which
    // stands in the same place output_hc_* does for the target.
    //
    // The block builder reads its weights out of TargetWeights::layers, so the
    // head is presented as a one-layer model: the target's hyperparameters,
    // this module's single block, and a full-attention interval of one so that
    // block is treated as an attention layer, which is what it is.
    TargetWeights mw = t;

    // TargetWeights carries two CpuEmbedders, and their destructor calls
    // munmap and close. Copying the struct copies those handles, so when this
    // copy goes out of scope it unmaps the token table and the 36 GiB PLE
    // table that the TARGET is still reading from -- and the next target step
    // dies in dequantize_row_q4_0 on memory that is no longer there. Which is
    // exactly what it did. Drop the copy's claim on both; assignment
    // overwrites the handles rather than releasing them, which is what is
    // wanted here because the original still owns them.
    mw.embedder  = CpuEmbedder{};
    mw.ple_table = CpuEmbedder{};

    mw.layers.assign(1, w.layer);
    mw.n_layer = 1;
    mw.full_attention_interval = 1;
    mw.cluster = nullptr;              // the head is replicated on every rank

    // DFLASH_QWEN4EXP_MTP_NO_BLOCK=1 leaves the block out, so a failure in the
    // glue can be told apart from one inside the block. The draft is then
    // nonsense, which is exactly what makes it a bisection and not a fallback.
    static const bool with_block = [] {
        const char * e = std::getenv("DFLASH_QWEN4EXP_MTP_NO_BLOCK");
        return !(e && std::atoi(e) == 1);
    }();
    if (with_block) {
        build_qwen4exp_layer(ctx, gf, mw, cache, /*layer_idx=*/0, &state,
                             positions, attn_mask, kv_start, (int) nt,
                             /*fa_window=*/0, /*kv_write_rows=*/nullptr,
                             /*parent_ids=*/nullptr);
    }

    ggml_tensor * mixed = qwen4exp_hc_mix(ctx, gf, /*cluster=*/nullptr, state,
                                          w.head_norm, w.head_down, w.head_up,
                                          /*w_inject=*/nullptr, /*inject_out=*/nullptr,
                                          t.n_embd, t.n_hc, t.rms_eps);
    ggml_tensor * logits = ggml_mul_mat(ctx, w.output, mixed);
    ggml_tensor * draft  = ggml_argmax(ctx, logits);
    ggml_set_output(draft);
    ggml_build_forward_expand(gf, draft);
    return draft;
}

}  // namespace dflash::common
