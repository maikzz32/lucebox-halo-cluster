#include "qwen4exp/qwen4exp_internal.h"
#include "qwen4exp/qwen4exp_cluster.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <set>
#include <vector>

namespace dflash::common {

namespace {

// ── metadata readers ──────────────────────────────────────────────────────
// Every getter reports the key it wanted, because a missing hyperparameter in
// a 66-key header is otherwise indistinguishable from a wrong one.

bool kv_i64(gguf_context * g, const std::string & key, int64_t & out, std::string & err) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) { err = "qwen4exp: missing metadata key '" + key + "'"; return false; }
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT8:   out = gguf_get_val_u8(g, id);  return true;
        case GGUF_TYPE_INT8:    out = gguf_get_val_i8(g, id);  return true;
        case GGUF_TYPE_UINT16:  out = gguf_get_val_u16(g, id); return true;
        case GGUF_TYPE_INT16:   out = gguf_get_val_i16(g, id); return true;
        case GGUF_TYPE_UINT32:  out = gguf_get_val_u32(g, id); return true;
        case GGUF_TYPE_INT32:   out = gguf_get_val_i32(g, id); return true;
        case GGUF_TYPE_UINT64:  out = (int64_t) gguf_get_val_u64(g, id); return true;
        case GGUF_TYPE_INT64:   out = gguf_get_val_i64(g, id); return true;
        default:
            err = "qwen4exp: metadata key '" + key + "' is not an integer";
            return false;
    }
}

bool kv_f32(gguf_context * g, const std::string & key, float & out, std::string & err) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) { err = "qwen4exp: missing metadata key '" + key + "'"; return false; }
    if (gguf_get_kv_type(g, id) == GGUF_TYPE_FLOAT32) { out = gguf_get_val_f32(g, id); return true; }
    if (gguf_get_kv_type(g, id) == GGUF_TYPE_FLOAT64) { out = (float) gguf_get_val_f64(g, id); return true; }
    err = "qwen4exp: metadata key '" + key + "' is not a float";
    return false;
}

// Optional integer: absent is not an error, it leaves `out` untouched.
bool kv_i64_opt(gguf_context * g, const std::string & key, int64_t & out) {
    std::string ignored;
    return kv_i64(g, key, out, ignored);
}

bool kv_i32_array(gguf_context * g, const std::string & key,
                  std::vector<int64_t> & out, std::string & err) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) { err = "qwen4exp: missing metadata key '" + key + "'"; return false; }
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY) {
        err = "qwen4exp: metadata key '" + key + "' is not an array";
        return false;
    }
    const size_t n = gguf_get_arr_n(g, id);
    const gguf_type et = gguf_get_arr_type(g, id);
    out.clear();
    out.reserve(n);
    const void * data = gguf_get_arr_data(g, id);
    for (size_t i = 0; i < n; ++i) {
        switch (et) {
            case GGUF_TYPE_INT32:  out.push_back(((const int32_t *)  data)[i]); break;
            case GGUF_TYPE_UINT32: out.push_back(((const uint32_t *) data)[i]); break;
            case GGUF_TYPE_INT64:  out.push_back(((const int64_t *)  data)[i]); break;
            case GGUF_TYPE_UINT64: out.push_back((int64_t) ((const uint64_t *) data)[i]); break;
            default:
                err = "qwen4exp: array '" + key + "' has a non-integer element type";
                return false;
        }
    }
    return true;
}

// ── the shape equations ───────────────────────────────────────────────────
// Each ties a value derived from metadata to an `ne` the file carries. They
// exist because this architecture has no reference implementation here: a
// mis-read dimension otherwise produces a model that loads, runs, and is
// wrong. Failing here names both numbers.

struct ShapeCheck {
    const GgufShardSet & shards;
    std::string & err;
    bool ok = true;

    // ne[axis] of a tensor that must exist.
    int64_t ne(const std::string & name, int axis) {
        if (!ok) return -1;
        GgufShardTensor t;
        std::string e;
        if (!shards.find(name.c_str(), t, e) || !t.meta) {
            err = "qwen4exp: " + e;
            ok = false;
            return -1;
        }
        return t.meta->ne[axis];
    }

    void equal(const char * what, int64_t derived, int64_t from_file) {
        if (!ok || derived == from_file) return;
        err = std::string("qwen4exp: ") + what + " -- the metadata implies " +
              std::to_string(derived) + " but the file's tensor says " +
              std::to_string(from_file) +
              ". The hyperparameters and the weights disagree; loading would "
              "produce a model that runs and is wrong.";
        ok = false;
    }

    void at_most(const char * what, int64_t a, int64_t b) {
        if (!ok || a <= b) return;
        err = std::string("qwen4exp: ") + what + " -- " + std::to_string(a) +
              " exceeds " + std::to_string(b);
        ok = false;
    }
};

}  // namespace

bool read_qwen4exp_hparams(const GgufShardSet & shards,
                           TargetWeights & out,
                           std::string & err) {
    gguf_context * g = shards.meta();
    if (!g) { err = "qwen4exp: shard set is not open"; return false; }

    int64_t v = 0;
    auto need = [&](const char * key, int & dst) {
        if (!err.empty()) return;
        if (!kv_i64(g, std::string("qwen4exp.") + key, v, err)) return;
        dst = (int) v;
    };

    err.clear();
    need("block_count",                       out.n_layer);
    need("embedding_length",                  out.n_embd);
    need("attention.head_count",              out.n_head);
    need("attention.head_count_kv",           out.n_head_kv);
    need("attention.key_length",              out.n_embd_head_k);
    need("attention.value_length",            out.n_embd_head_v);
    need("expert_count",                      out.n_expert);
    need("expert_used_count",                 out.n_expert_used);
    need("expert_feed_forward_length",        out.n_ff_exp);
    need("full_attention_interval",           out.full_attention_interval);
    need("rope.dimension_count",              out.rope_dimension_count);
    need("ssm.conv_kernel",                   out.ssm_d_conv);
    need("ssm.state_size",                    out.ssm_d_state);
    need("ssm.group_count",                   out.ssm_n_group);
    need("ssm.time_step_rank",                out.ssm_dt_rank);
    need("ssm.inner_size",                    out.ssm_d_inner);
    // qwen4exp's gated delta net differs from Qwen3.5's in exactly one place.
    out.gdn_sigmoid_output_gate = true;

    need("hyper_connection.count",            out.n_hc);
    need("hyper_connection.low_rank",         out.hc_low_rank);
    need("attention.indexer.head_count",      out.n_indexer_head);
    need("attention.indexer.key_length",      out.indexer_key_length);
    need("attention.indexer.top_k",           out.indexer_top_k);
    if (!err.empty()) return false;

    // Optional: the shared expert may size differently from the routed ones.
    int64_t shexp = out.n_ff_exp;
    kv_i64_opt(g, "qwen4exp.expert_shared_feed_forward_length", shexp);
    out.n_ff_shexp = (int) shexp;

    if (!kv_f32(g, "qwen4exp.attention.layer_norm_rms_epsilon", out.rms_eps, err)) return false;
    kv_f32(g, "qwen4exp.rope.freq_base", out.rope_theta, err);
    err.clear();

    std::vector<int64_t> sections;
    if (!kv_i32_array(g, "qwen4exp.rope.dimension_sections", sections, err)) return false;
    if (sections.size() != 4) {
        err = "qwen4exp: rope.dimension_sections has " + std::to_string(sections.size()) +
              " entries, expected 4";
        return false;
    }
    for (int i = 0; i < 4; ++i) out.rope_sections[i] = (int) sections[(size_t) i];

    // PLE is optional in principle; this checkpoint names exactly one layer.
    std::vector<int64_t> ple_layers;
    std::string ple_err;
    if (kv_i32_array(g, "qwen4exp.ple.layers", ple_layers, ple_err) && !ple_layers.empty()) {
        out.ple_layer = (int) ple_layers[0];
        int64_t t = 0;
        if (kv_i64_opt(g, "qwen4exp.ple.ngram_size", t))   out.ple_ngram_size = (int) t;
        if (kv_i64_opt(g, "qwen4exp.ple.conv_kernel", t))  out.ple_conv_kernel = (int) t;
        if (kv_i64_opt(g, "qwen4exp.embedding_length_per_layer_input", t))
            out.n_embd_per_layer_input = (int) t;
        if (kv_i64_opt(g, "qwen4exp.ple.heads_per_ngram", t))
            out.ple_heads_per_ngram = (int) t;
        if (kv_i64_opt(g, "qwen4exp.ple.eos_token_id", t))
            out.ple_eos_token_id = (int32_t) t;
        if (kv_i64_opt(g, "qwen4exp.ple.image_token_id", t))
            out.ple_image_token_id = (int32_t) t;

        // ple_n_heads is derived, not stored, and the two head arrays must
        // carry exactly that many entries -- a mismatch means the hash would
        // index past its own constants.
        out.ple_n_heads = (out.ple_ngram_size - 1) * out.ple_heads_per_ngram;

        std::vector<int64_t> mult, offs, vocabs;
        std::string e;
        if (!kv_i32_array(g, "qwen4exp.ple.layer_multipliers", mult, e) ||
            !kv_i32_array(g, "qwen4exp.ple.head_offsets", offs, e) ||
            !kv_i32_array(g, "qwen4exp.ple.head_vocab_sizes", vocabs, e)) {
            err = "qwen4exp: PLE is declared on layer " + std::to_string(out.ple_layer) +
                  " but its hash constants are missing: " + e;
            return false;
        }
        if ((int) mult.size() != out.ple_ngram_size ||
            (int) offs.size() != out.ple_n_heads ||
            (int) vocabs.size() != out.ple_n_heads) {
            err = "qwen4exp: PLE hash constants have the wrong lengths -- "
                  "layer_multipliers " + std::to_string(mult.size()) + " (want " +
                  std::to_string(out.ple_ngram_size) + "), head_offsets " +
                  std::to_string(offs.size()) + " and head_vocab_sizes " +
                  std::to_string(vocabs.size()) + " (want " +
                  std::to_string(out.ple_n_heads) + " each)";
            return false;
        }
        out.ple_layer_multipliers.assign(mult.begin(), mult.end());
        out.ple_head_offsets.assign(offs.begin(), offs.end());
        out.ple_head_vocab_sizes.assign(vocabs.begin(), vocabs.end());
    }

    // ── verify against the weights ────────────────────────────────────────
    ShapeCheck c{shards, err};

    // The delta-net projection and its convolution both span
    // inner + 2 * groups * state. Getting any of the three ssm dimensions
    // wrong changes this sum, so one equation covers all of them.
    const int64_t conv_channels =
        (int64_t) out.ssm_d_inner + 2LL * out.ssm_n_group * out.ssm_d_state;
    c.equal("ssm inner + 2*groups*state vs blk.0.ssm_conv1d",
            conv_channels, c.ne("blk.0.ssm_conv1d.weight", 1));
    c.equal("ssm inner + 2*groups*state vs blk.0.attn_qkv",
            conv_channels, c.ne("blk.0.attn_qkv.weight", 1));
    c.equal("ssm inner / time_step_rank vs blk.0.ssm_norm",
            out.ssm_d_inner / (out.ssm_dt_rank ? out.ssm_dt_rank : 1),
            c.ne("blk.0.ssm_norm.weight", 0));

    // Hyper-connections: the state is n_hc streams of n_embd.
    const int64_t hc_width = (int64_t) out.n_hc * out.n_embd;
    c.equal("hc count * embedding_length vs blk.0.hc_attn_norm",
            hc_width, c.ne("blk.0.hc_attn_norm.weight", 0));
    c.equal("hc low_rank vs blk.0.hc_attn_down",
            out.hc_low_rank, c.ne("blk.0.hc_attn_down.weight", 1));
    c.equal("hc low_rank vs blk.0.hc_attn_up",
            out.hc_low_rank, c.ne("blk.0.hc_attn_up.weight", 0));
    c.equal("hc count vs blk.0.hc_attn_inject",
            out.n_hc, c.ne("blk.0.hc_attn_inject.weight", 1));

    // m-RoPE: the four sections are half-dimensions.
    int64_t section_sum = 0;
    for (int i = 0; i < 4; ++i) section_sum += out.rope_sections[i];
    c.equal("2 * sum(rope.dimension_sections) vs rope.dimension_count",
            2 * section_sum, out.rope_dimension_count);
    c.at_most("rope.dimension_count vs attention.key_length",
              out.rope_dimension_count, out.n_embd_head_k);

    // The router names the expert count, and the expert stack must carry it.
    c.equal("expert_count vs blk.0.ffn_gate_inp",
            out.n_expert, c.ne("blk.0.ffn_gate_inp.weight", 1));
    c.equal("expert_count vs blk.0.ffn_gate_exps",
            out.n_expert, c.ne("blk.0.ffn_gate_exps.weight", 2));
    c.equal("expert_feed_forward_length vs blk.0.ffn_gate_exps",
            out.n_ff_exp, c.ne("blk.0.ffn_gate_exps.weight", 1));

    if (out.n_embd_per_layer_input > 0 && shards.has("per_layer_token_embd.weight")) {
        c.equal("embedding_length_per_layer_input vs per_layer_token_embd",
                out.n_embd_per_layer_input, c.ne("per_layer_token_embd.weight", 0));
    }

    // The layer schedule: a full-attention layer has attn_q, a delta-net layer
    // has attn_qkv, and the interval says which is which. Disagreement here
    // means the graph would build the wrong block for the wrong layer.
    if (c.ok && out.full_attention_interval > 0) {
        for (int il = 0; il < out.n_layer; ++il) {
            const bool want_full = ((il + 1) % out.full_attention_interval) == 0;
            const std::string q   = "blk." + std::to_string(il) + ".attn_q.weight";
            const std::string qkv = "blk." + std::to_string(il) + ".attn_qkv.weight";
            const bool has_full = shards.has(q.c_str());
            const bool has_gdn  = shards.has(qkv.c_str());
            if (has_full != want_full || has_gdn == want_full) {
                err = "qwen4exp: layer " + std::to_string(il) + " should be " +
                      (want_full ? "full attention" : "gated delta net") +
                      " by full_attention_interval=" +
                      std::to_string(out.full_attention_interval) +
                      ", but it carries " +
                      (has_full ? "attn_q" : has_gdn ? "attn_qkv" : "neither");
                c.ok = false;
                break;
            }
        }
    }

    return c.ok;
}

namespace {

// A tensor we want, and where its bytes are.
struct Binding {
    ggml_tensor *   dst = nullptr;   // in the context TargetWeights owns
    GgufShardTensor src;             // into whichever shard holds it
    bool            to_f32 = false;  // dequantise on the host during upload
    // Cluster sharding. `axis` says which of the source's dimensions this rank
    // holds a slice of; begin/count are in elements along it.
    ShardAxis       axis  = ShardAxis::None;
    int64_t         begin = 0;
    int64_t         count = 0;
};

// Bind one tensor by name. Optional names may be absent; required ones fail.
struct Binder {
    const GgufShardSet & shards;
    ggml_context *       ctx;
    std::vector<Binding> & out;
    std::string &        err;
    const Qwen4ExpClusterRuntime * cluster = nullptr;
    std::set<std::string> warned_unsplit;
    bool ok = true;

    // Bind a tensor and dequantise it to F32 on the way to the device.
    //
    // For a weight with very few rows this costs almost nothing in memory and
    // avoids the quantised matmul entirely. hc_*_inject is [n_hc*n_embd, n_hc]
    // -- four rows -- which is narrower than the row tiles the ROCmFP4 kernels
    // are built around, and narrow enough that quantising it saves under 8 MiB
    // across the whole model.
    ggml_tensor * take_f32(const std::string & name, bool required) {
        ggml_tensor * t = take(name, required);
        if (t && ggml_is_quantized(t->type)) {
            out.back().to_f32 = true;
            t->type = GGML_TYPE_F32;
            t->nb[0] = ggml_type_size(GGML_TYPE_F32);
            for (int d = 1; d < GGML_MAX_DIMS; ++d) {
                t->nb[d] = t->nb[d - 1] * t->ne[d - 1];
            }
        }
        return t;
    }

    // Bind a tensor holding only this rank's slice of it.
    //
    // Rows are a contiguous run and cost nothing to cut. Columns are a range
    // inside every row, so the cut must land on a quantisation block boundary
    // -- qwen4exp's expert width of 640 divides by two and by four and stays
    // on the 32-element block of every type these files use, which is why the
    // MoE splits by width rather than by expert index. That also leaves the
    // router untouched: every rank routes to the same ten experts and computes
    // its slice of their hidden dimension, and one all-reduce completes the
    // sum. Splitting by expert index would have needed the ids remapped on
    // device and would still have read a sentinel expert per unowned route.
    ggml_tensor * take_shard(const std::string & name, ShardAxis axis,
                             int64_t granularity, bool required) {
        ggml_tensor * t = take(name, required);
        if (!t || axis == ShardAxis::None || !cluster || !cluster->sharded()) {
            return t;
        }
        Binding & b = out.back();
        const int64_t extent = axis == ShardAxis::Rows ? b.src.meta->ne[1]
                                                       : b.src.meta->ne[0];
        const ShardRange r = qwen4exp_shard_range(extent, granularity,
                                                  cluster->rank(), cluster->size());
        if (!r.split) {
            if (!warned_unsplit.count(name)) {
                warned_unsplit.insert(name);
                std::fprintf(stderr,
                             "[qwen4exp-cluster] %s: %lld does not divide by %d on a "
                             "%lld boundary, keeping it replicated\n",
                             name.c_str(), (long long) extent, cluster->size(),
                             (long long) granularity);
            }
            return t;
        }
        b.axis  = axis;
        b.begin = r.begin;
        b.count = r.count();
        if (axis == ShardAxis::Rows) {
            t->ne[1] = r.count();
        } else {
            t->ne[0] = r.count();
        }
        // Recompute the strides for the narrower tensor.
        t->nb[0] = ggml_type_size(t->type);
        t->nb[1] = ggml_row_size(t->type, t->ne[0]);
        for (int d = 2; d < GGML_MAX_DIMS; ++d) {
            t->nb[d] = t->nb[d - 1] * t->ne[d - 1];
        }
        return t;
    }

    ggml_tensor * take(const std::string & name, bool required) {
        if (!ok) return nullptr;
        if (!required && !shards.has(name.c_str())) return nullptr;
        GgufShardTensor src;
        std::string e;
        if (!shards.find(name.c_str(), src, e) || !src.meta) {
            if (!required) return nullptr;
            err = "qwen4exp: " + e;
            ok = false;
            return nullptr;
        }
        ggml_tensor * dst = ggml_dup_tensor(ctx, src.meta);
        if (!dst) {
            err = "qwen4exp: metadata context exhausted while binding " + name;
            ok = false;
            return nullptr;
        }
        ggml_set_name(dst, name.c_str());
        out.push_back(Binding{dst, src});
        return dst;
    }
};

// The granularity every expert-width cut must land on. 32 is the block size of
// every quantisation these checkpoints use for the expert matrices, and the
// coarsest that still divides 640 by four. A finer cut would break a
// quantisation block; a coarser one would refuse the four-node split.
constexpr int64_t kFfnShardGran = 32;

size_t align_up_to(size_t v, size_t a) { return a ? ((v + a - 1) / a) * a : v; }

}  // namespace

bool load_qwen4exp_gguf(const std::string & path,
                        ggml_backend_t backend,
                        TargetWeights & out,
                        const Qwen4ExpClusterRuntime * cluster) {
    GgufShardSet shards;
    std::string err;
    if (!shards.open(path, err)) {
        std::fprintf(stderr, "[qwen4exp] %s\n", err.c_str());
        return false;
    }
    if (!read_qwen4exp_hparams(shards, out, err)) {
        std::fprintf(stderr, "[qwen4exp] %s\n", err.c_str());
        return false;
    }

    // One context of our own holding a copy of every wanted tensor's shape and
    // type. The shard set has one context per file and closes at the end of
    // this function; TargetWeights owns exactly one context, so the metadata is
    // duplicated rather than borrowed.
    const size_t max_tensors = (size_t) shards.total_tensors() + 16;
    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * max_tensors;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) {
        std::fprintf(stderr, "[qwen4exp] metadata context allocation failed\n");
        return false;
    }

    std::vector<Binding> bindings;
    bindings.reserve(max_tensors);
    Binder b{shards, out.ctx, bindings, err, cluster};

    auto blk = [](int il, const char * suffix) {
        return "blk." + std::to_string(il) + "." + suffix;
    };

    out.layers.resize((size_t) out.n_layer);
    for (int il = 0; il < out.n_layer && b.ok; ++il) {
        TargetLayer & L = out.layers[(size_t) il];
        const bool full_attn = ((il + 1) % out.full_attention_interval) == 0;

        // Hyper-connections replace the usual pre-norms and are on every layer.
        L.hc_attn_norm   = b.take(blk(il, "hc_attn_norm.weight"),   true);
        L.hc_attn_down   = b.take(blk(il, "hc_attn_down.weight"),   true);
        L.hc_attn_up     = b.take(blk(il, "hc_attn_up.weight"),     true);
        L.hc_attn_inject = b.take_f32(blk(il, "hc_attn_inject.weight"), true);
        L.hc_ffn_norm    = b.take(blk(il, "hc_ffn_norm.weight"),    true);
        L.hc_ffn_down    = b.take(blk(il, "hc_ffn_down.weight"),    true);
        L.hc_ffn_up      = b.take(blk(il, "hc_ffn_up.weight"),      true);
        L.hc_ffn_inject  = b.take_f32(blk(il, "hc_ffn_inject.weight"),  true);

        if (full_attn) {
            L.wq     = b.take(blk(il, "attn_q.weight"),      true);
            L.wk     = b.take(blk(il, "attn_k.weight"),      true);
            L.wv     = b.take(blk(il, "attn_v.weight"),      true);
            L.wo     = b.take(blk(il, "attn_output.weight"), true);
            L.q_norm = b.take(blk(il, "attn_q_norm.weight"), true);
            L.k_norm = b.take(blk(il, "attn_k_norm.weight"), true);
            L.indexer_q_proj = b.take(blk(il, "indexer.q_proj.weight"), true);
            L.indexer_k_proj = b.take(blk(il, "indexer.k_proj.weight"), true);
            L.indexer_q_norm = b.take(blk(il, "indexer.q_norm.weight"), true);
            L.indexer_k_norm = b.take(blk(il, "indexer.k_norm.weight"), true);
        } else {
            L.wqkv        = b.take(blk(il, "attn_qkv.weight"),   true);
            L.wqkv_gate   = b.take(blk(il, "attn_gate.weight"),  true);
            L.ssm_conv1d  = b.take(blk(il, "ssm_conv1d.weight"), true);
            L.ssm_alpha   = b.take(blk(il, "ssm_alpha.weight"),  true);
            L.ssm_beta    = b.take(blk(il, "ssm_beta.weight"),   true);
            L.ssm_a       = b.take(blk(il, "ssm_a"),             true);
            L.ssm_dt_bias = b.take(blk(il, "ssm_dt.bias"),       true);
            L.ssm_norm    = b.take(blk(il, "ssm_norm.weight"),   true);
            L.ssm_out     = b.take(blk(il, "ssm_out.weight"),    true);
        }

        // MoE: 512 routed experts top-10, plus a shared expert whose scalar
        // sigmoid gate qwen35moe already knows how to apply.
        L.ffn_gate_inp       = b.take(blk(il, "ffn_gate_inp.weight"),       true);
        L.ffn_gate_inp_shexp = b.take(blk(il, "ffn_gate_inp_shexp.weight"), false);
        // The expert hidden width is what splits, on all three matrices at
        // once: gate and up produce it, down consumes it. Every rank routes to
        // the same experts and holds a slice of each one's width, so the
        // routed and shared outputs are partial sums that one all-reduce per
        // layer completes.
        L.ffn_gate_exps      = b.take_shard(blk(il, "ffn_gate_exps.weight"),
                                            ShardAxis::Rows, kFfnShardGran, true);
        L.ffn_up_exps        = b.take_shard(blk(il, "ffn_up_exps.weight"),
                                            ShardAxis::Rows, kFfnShardGran, true);
        L.ffn_down_exps      = b.take_shard(blk(il, "ffn_down_exps.weight"),
                                            ShardAxis::Cols, kFfnShardGran, true);
        L.ffn_gate_shexp     = b.take_shard(blk(il, "ffn_gate_shexp.weight"),
                                            ShardAxis::Rows, kFfnShardGran, false);
        L.ffn_up_shexp       = b.take_shard(blk(il, "ffn_up_shexp.weight"),
                                            ShardAxis::Rows, kFfnShardGran, false);
        L.ffn_down_shexp     = b.take_shard(blk(il, "ffn_down_shexp.weight"),
                                            ShardAxis::Cols, kFfnShardGran, false);

        if (il == out.ple_layer) {
            L.ple_key        = b.take(blk(il, "ple_key.weight"),        true);
            L.ple_value      = b.take(blk(il, "ple_value.weight"),      true);
            L.ple_conv1d     = b.take(blk(il, "ple_conv1d.weight"),     true);
            L.ple_norm_key   = b.take(blk(il, "ple_norm_key.weight"),   true);
            L.ple_norm_query = b.take(blk(il, "ple_norm_query.weight"), true);
            L.ple_norm_conv  = b.take(blk(il, "ple_norm_conv.weight"),  true);
        }
    }

    out.output_hc_norm = b.take("output_hc_norm.weight", true);
    out.output_hc_down = b.take("output_hc_down.weight", true);
    out.output_hc_up   = b.take("output_hc_up.weight",   true);
    out.output         = b.take("output.weight",         true);

    if (!b.ok) {
        std::fprintf(stderr, "[qwen4exp] %s\n", err.c_str());
        ggml_free(out.ctx);
        out.ctx = nullptr;
        return false;
    }

    // token_embd and per_layer_token_embd stay off the device: the first is
    // read a row at a time by the embedder, and the second is tens of GiB of
    // table touched a few kilobytes per token. Both are metadata only here.
    GgufShardTensor tok, ple_tab;
    std::string ignored;
    if (!shards.find("token_embd.weight", tok, err) || !tok.meta) {
        std::fprintf(stderr, "[qwen4exp] token_embd.weight: %s\n", err.c_str());
        ggml_free(out.ctx);
        out.ctx = nullptr;
        return false;
    }
    out.tok_embd = ggml_dup_tensor(out.ctx, tok.meta);
    if (out.tok_embd) ggml_set_name(out.tok_embd, "token_embd.weight");
    out.n_vocab = (int) tok.meta->ne[1];

    // The embedder owns its copy: the shard set's mappings close when this
    // function returns, and a row lookup that read through a dangling mmap
    // would fault at the first token rather than at load.
    out.embedder.tok_embd_owned.resize(tok.size);
    std::memcpy(out.embedder.tok_embd_owned.data(), tok.data, tok.size);
    out.embedder.tok_embd_bytes = out.embedder.tok_embd_owned.data();
    out.embedder.tok_embd_type  = tok.meta->type;
    out.embedder.n_embd         = out.n_embd;
    out.embedder.n_vocab        = out.n_vocab;
    out.embedder.row_bytes      = tok.size / (size_t) out.n_vocab;
    if (shards.find("per_layer_token_embd.weight", ple_tab, ignored) && ple_tab.meta) {
        out.per_layer_token_embd = ggml_dup_tensor(out.ctx, ple_tab.meta);
        if (out.per_layer_token_embd)
            ggml_set_name(out.per_layer_token_embd, "per_layer_token_embd.weight");

        // Its own mapping of the same shard, kept alive by the embedder. A
        // second mapping of one file costs nothing beyond the descriptor --
        // the pages are shared -- and it avoids both a 36 GiB copy and 36 GiB
        // of device memory this box does not have to spare.
        GgufMmap table_map;
        std::string map_err;
        if (!table_map.open(ple_tab.path, map_err)) {
            std::fprintf(stderr, "[qwen4exp] PLE table: %s\n", map_err.c_str());
        } else {
            GgufMmap::OwnedRegion r = table_map.release();
            out.ple_table.mmap_addr = const_cast<void *>(r.data);
            out.ple_table.mmap_len  = r.size;
#if !defined(_WIN32)
            out.ple_table.mmap_fd   = r.fd;
#endif
            out.ple_table.tok_embd_bytes =
                static_cast<const uint8_t *>(r.data) + ple_tab.file_offset;
            out.ple_table.tok_embd_type = ple_tab.meta->type;
            out.ple_table.n_embd  = ple_tab.meta->ne[0];
            out.ple_table.n_vocab = ple_tab.meta->ne[1];
            out.ple_table.row_bytes =
                ple_tab.size / (size_t) out.ple_table.n_vocab;
            std::fprintf(stderr,
                "[qwen4exp] PLE table mapped: %lld rows of %lld, %.1f GiB, "
                "%zu bytes/row\n",
                (long long) out.ple_table.n_vocab, (long long) out.ple_table.n_embd,
                (double) ple_tab.size / (1024.0*1024.0*1024.0),
                out.ple_table.row_bytes);
        }
    }

    // ── one buffer, one pass ──────────────────────────────────────────────
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    size_t total = 0;
    std::vector<size_t> offsets;
    offsets.reserve(bindings.size());
    for (const Binding & bind : bindings) {
        total = align_up_to(total, alignment);
        offsets.push_back(total);
        total += ggml_backend_buft_get_alloc_size(buft, bind.dst);
    }

    out.buf = ggml_backend_alloc_buffer(backend, total);
    if (!out.buf) {
        std::fprintf(stderr,
            "[qwen4exp] weight buffer allocation failed, %.1f GiB requested\n",
            (double) total / (1024.0 * 1024.0 * 1024.0));
        ggml_free(out.ctx);
        out.ctx = nullptr;
        return false;
    }
    ggml_backend_buffer_set_usage(out.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    char * base = static_cast<char *>(ggml_backend_buffer_get_base(out.buf));

    for (size_t i = 0; i < bindings.size(); ++i) {
        if (ggml_backend_tensor_alloc(out.buf, bindings[i].dst, base + offsets[i]) !=
            GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "[qwen4exp] tensor allocation failed for %s\n",
                         ggml_get_name(bindings[i].dst));
            ggml_backend_buffer_free(out.buf);
            out.buf = nullptr;
            ggml_free(out.ctx);
            out.ctx = nullptr;
            return false;
        }
    }
    std::vector<float> scratch;
    for (const Binding & bind : bindings) {
        // find() already bounds-checked the source against its own shard.
        shards.advise_willneed(bind.src);
        if (bind.axis != ShardAxis::None) {
            // Copy this rank's slice out of the mapped source. Rows are one
            // contiguous run per higher-dimension slab; columns are a range
            // inside every row, which is why the cut has to be block-aligned.
            const ggml_type st = bind.src.meta->type;
            const int64_t   s0 = bind.src.meta->ne[0];
            const int64_t   s1 = bind.src.meta->ne[1];
            const int64_t   s2 = ggml_nelements(bind.src.meta) / (s0 * s1);
            const size_t    src_row = ggml_row_size(st, s0);
            const char *    sp = static_cast<const char *>(bind.src.data);
            size_t          off = 0;
            if (bind.axis == ShardAxis::Rows) {
                const size_t run = (size_t) bind.count * src_row;
                for (int64_t i2 = 0; i2 < s2; ++i2) {
                    ggml_backend_tensor_set(
                        bind.dst,
                        sp + ((size_t) i2 * s1 + (size_t) bind.begin) * src_row,
                        off, run);
                    off += run;
                }
            } else {
                const size_t skip = ggml_row_size(st, bind.begin);
                const size_t run  = ggml_row_size(st, bind.count);
                for (int64_t r = 0; r < s1 * s2; ++r) {
                    ggml_backend_tensor_set(bind.dst,
                                            sp + (size_t) r * src_row + skip,
                                            off, run);
                    off += run;
                }
            }
            continue;
        }
        if (!bind.to_f32) {
            ggml_backend_tensor_set(bind.dst, bind.src.data, 0, bind.src.size);
            continue;
        }
        const ggml_type      st = bind.src.meta->type;
        const ggml_type_traits * tr = ggml_get_type_traits(st);
        const int64_t nc   = bind.dst->ne[0];
        const int64_t rows = ggml_nelements(bind.dst) / nc;
        const size_t  srow = ggml_row_size(st, nc);
        scratch.resize((size_t) nc * (size_t) rows);
        const char * sp = static_cast<const char *>(bind.src.data);
        for (int64_t r = 0; r < rows; ++r) {
            tr->to_float(sp + (size_t) r * srow, scratch.data() + (size_t) r * nc, nc);
        }
        ggml_backend_tensor_set(bind.dst, scratch.data(), 0,
                                scratch.size() * sizeof(float));
        if (std::getenv("DFLASH_QWEN4EXP_RMS")) {
            const int64_t nc2   = bind.dst->ne[0];
            const int64_t rows2 = (int64_t) scratch.size() / nc2;
            for (int64_t r = 0; r < rows2 && r < 4; ++r) {
                double sum = 0.0, sq = 0.0;
                const float * rp = scratch.data() + (size_t) r * nc2;
                for (int64_t i = 0; i < nc2; ++i) { sum += rp[i]; sq += (double) rp[i] * rp[i]; }
                std::fprintf(stderr,
                    "[q4e-w] %-28s row%lld mean=%+.5f rms=%.5f  first=%+.4f %+.4f %+.4f\n",
                    ggml_get_name(bind.dst), (long long) r,
                    sum / (double) nc2, std::sqrt(sq / (double) nc2),
                    rp[0], rp[1], rp[2]);
            }
        }
    }

    // File integrity: what do the quantised weights actually contain?
    //
    // This checkpoint is a third-party quantisation into ggml types that only
    // this tree understands, and nothing has ever run it. A tensor family that
    // dequantises to zeros, or to values orders away from its neighbours,
    // would explain a model that runs and says nothing -- and no amount of
    // work on the graph would fix it. One layer of each kind is enough to see
    // it; the first 65536 values of each tensor are enough for the statistic.
    if (std::getenv("DFLASH_QWEN4EXP_RMS")) {
        std::vector<float> buf(65536);
        for (const Binding & bd : bindings) {
            const char * nm = ggml_get_name(bd.dst);
            const std::string n(nm ? nm : "");
            if (n.rfind("blk.0.", 0) != 0 && n.rfind("blk.3.", 0) != 0 &&
                n.rfind("output", 0) != 0) {
                continue;
            }
            const ggml_type st = bd.src.meta->type;
            const ggml_type_traits * tr = ggml_get_type_traits(st);
            if (!tr || !tr->to_float) continue;
            const int64_t nc = bd.src.meta->ne[0];
            const int64_t rows = std::min<int64_t>(
                ggml_nelements(bd.src.meta) / nc, (int64_t) buf.size() / nc);
            if (rows < 1) continue;
            const size_t srow = ggml_row_size(st, nc);
            double sq = 0.0;
            int64_t zeros = 0;
            for (int64_t r = 0; r < rows; ++r) {
                tr->to_float(static_cast<const char *>(bd.src.data) + (size_t) r * srow,
                             buf.data(), nc);
                for (int64_t k = 0; k < nc; ++k) {
                    sq += (double) buf[(size_t) k] * buf[(size_t) k];
                    zeros += buf[(size_t) k] == 0.0f;
                }
            }
            const int64_t seen = rows * nc;
            std::fprintf(stderr, "[q4e-file] %-34s %-18s rms=%.5f zeros=%.1f%%\n",
                         n.c_str(), ggml_type_name(st),
                         std::sqrt(sq / (double) seen),
                         100.0 * (double) zeros / (double) seen);
        }
    }

    // Bring-up check: does the unembedding know the embedding?
    //
    // Every transformer correlates output.weight[v] with token_embd[v] for the
    // same v, tied or not, because both encode the identity of token v. If the
    // self-similarity is not clearly above the cross-similarity then the two
    // tensors are not being read in the same basis, and no amount of work on
    // the blocks between them will produce sensible logits.
    if (std::getenv("DFLASH_QWEN4EXP_RMS")) {
        GgufShardTensor outw;
        std::string oe;
        if (shards.find("output.weight", outw, oe) && outw.meta) {
            const ggml_type_traits * otr = ggml_get_type_traits(outw.meta->type);
            const size_t orow = ggml_row_size(outw.meta->type, out.n_embd);
            // Placement check before the basis check. A vocabulary padded up
            // to a round size leaves its last rows zero in both tables; if the
            // low rows are non-zero and the top rows are zero, both tensors sit
            // where the shard says they do and the comparison below is about
            // the bases and not about the offsets.
            {
                const int32_t probe[7] = { 0, 100, 248000, 248200, 248300,
                                           248318, 248319 };
                std::vector<float> r((size_t) out.n_embd);
                for (int a = 0; a < 7; ++a) {
                    double se = 0.0, so = 0.0;
                    if (out.embedder.embed(&probe[a], 1, r.data())) {
                        for (int k = 0; k < out.n_embd; ++k) se += (double) r[(size_t) k] * r[(size_t) k];
                    }
                    otr->to_float(static_cast<const char *>(outw.data) +
                                      (size_t) probe[a] * orow, r.data(), out.n_embd);
                    for (int k = 0; k < out.n_embd; ++k) so += (double) r[(size_t) k] * r[(size_t) k];
                    std::fprintf(stderr, "[q4e-tie] row %6d  embd_rms=%.6f  out_rms=%.6f\n",
                                 probe[a], std::sqrt(se / out.n_embd),
                                 std::sqrt(so / out.n_embd));
                }
            }
            const int32_t ids[5] = { 100, 1000, 15043, 50000, 200000 };
            std::vector<float> e((size_t) out.n_embd), o((size_t) out.n_embd);
            for (int a = 0; a < 5; ++a) {
                if (!out.embedder.embed(&ids[a], 1, e.data())) continue;
                for (int b = 0; b < 5; ++b) {
                    otr->to_float(static_cast<const char *>(outw.data) +
                                      (size_t) ids[b] * orow,
                                  o.data(), out.n_embd);
                    double dot = 0.0, ne2 = 0.0, no2 = 0.0;
                    for (int k = 0; k < out.n_embd; ++k) {
                        dot += (double) e[(size_t) k] * o[(size_t) k];
                        ne2 += (double) e[(size_t) k] * e[(size_t) k];
                        no2 += (double) o[(size_t) k] * o[(size_t) k];
                    }
                    std::fprintf(stderr, "[q4e-tie] embd(%6d) . out(%6d) cos=%+.4f%s\n",
                                 ids[a], ids[b],
                                 dot / (std::sqrt(ne2) * std::sqrt(no2) + 1e-30),
                                 a == b ? "   <- same token" : "");
                }
            }
        }
    }

    std::fprintf(stderr,
        "[qwen4exp] hparams: n_embd=%d n_layer=%d n_head=%d n_head_kv=%d head_dim=%d\n"
        "[qwen4exp]   full_attn_every=%d n_expert=%d/%d n_ff_exp=%d rope=%.0f/%d,%d,%d,%d\n"
        "[qwen4exp]   ssm: d_inner=%d d_state=%d n_group=%d dt_rank=%d conv=%d\n"
        "[qwen4exp]   hc: n_hc=%d low_rank=%d  ple: layer=%d ngram=%d heads=%d conv=%d\n"
        "[qwen4exp]   n_rot=%d rms_eps=%.2e n_ff_shexp=%d indexer: heads=%d key_len=%d top_k=%d\n",
        out.n_embd, out.n_layer, out.n_head, out.n_head_kv,
        out.n_embd_head_k,
        out.full_attention_interval, out.n_expert_used, out.n_expert, out.n_ff_exp,
        (double) out.rope_theta, out.rope_sections[0], out.rope_sections[1],
        out.rope_sections[2], out.rope_sections[3],
        out.ssm_d_inner, out.ssm_d_state, out.ssm_n_group, out.ssm_dt_rank,
        out.ssm_d_conv,
        out.n_hc, out.hc_low_rank, out.ple_layer, out.ple_ngram_size,
        out.ple_n_heads, out.ple_conv_kernel,
        out.rope_dimension_count, (double) out.rms_eps, out.n_ff_shexp,
        out.n_indexer_head, out.indexer_key_length, out.indexer_top_k);
    std::fprintf(stderr,
        "[qwen4exp] loaded %zu tensors from %s, %.1f GiB on device\n",
        bindings.size(), shards.describe().c_str(),
        (double) total / (1024.0 * 1024.0 * 1024.0));
    return true;
}

}  // namespace dflash::common
