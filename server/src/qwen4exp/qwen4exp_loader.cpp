#include "qwen4exp/qwen4exp_internal.h"

#include <cstdio>
#include <string>
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

}  // namespace dflash::common
