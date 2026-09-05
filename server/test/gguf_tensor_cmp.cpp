// gguf_tensor_cmp - what is different between two conversions of one model?
//
// Bringing up an architecture against a third-party checkpoint leaves a
// question that no amount of reading the graph answers: is this file the same
// model the reference implementation was written against? Metadata only says
// so much -- two files can agree on every hyperparameter and disagree on a
// tensor's basis, its orientation, or (with an abliterated repack) its values.
//
// This dequantises the same tensor from both files and reports the cosine
// between them, so the answer is a number. The two need not share a
// quantisation: each is decoded through its own type's to_float.
//
//   gguf_tensor_cmp <a.gguf> <b.gguf> [name ...]
//
// With no names it walks every tensor the two have in common and prints the
// ones that disagree, which is the useful direction: a short list of what a
// repack actually touched. Split GGUFs are handled; pass any part.

#include "common/gguf_shards.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace dflash::common;

namespace {

// Cap the work per tensor: an expert stack is half a gigabyte, and the first
// rows answer the question just as well as all of them.
constexpr int64_t kMaxElems = 1 << 20;

bool dequant_prefix(const GgufShardTensor & t, std::vector<float> & out) {
    const ggml_type_traits * tr = ggml_get_type_traits(t.meta->type);
    if (!tr || !tr->to_float) return false;
    const int64_t nc = t.meta->ne[0];
    const int64_t rows = std::min<int64_t>(ggml_nelements(t.meta) / nc, kMaxElems / nc);
    if (rows < 1) return false;
    const size_t rb = ggml_row_size(t.meta->type, nc);
    out.resize((size_t) (rows * nc));
    for (int64_t r = 0; r < rows; ++r) {
        tr->to_float(static_cast<const char *>(t.data) + (size_t) r * rb,
                     out.data() + (size_t) (r * nc), nc);
    }
    return true;
}

// Returns false when the tensor is missing from either side or cannot be read.
bool compare(const GgufShardSet & A, const GgufShardSet & B, const std::string & name,
             double & cos_out, double & rms_a, double & rms_b) {
    GgufShardTensor ta, tb;
    std::string e;
    if (!A.find(name.c_str(), ta, e) || !B.find(name.c_str(), tb, e)) return false;
    if (ta.meta->ne[0] != tb.meta->ne[0] || ggml_nelements(ta.meta) != ggml_nelements(tb.meta)) {
        std::printf("  %-46s SHAPE DIFFERS\n", name.c_str());
        return false;
    }
    std::vector<float> a, b;
    if (!dequant_prefix(ta, a) || !dequant_prefix(tb, b)) return false;
    const size_t n = std::min(a.size(), b.size());
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; ++i) {
        dot += (double) a[i] * b[i];
        na  += (double) a[i] * a[i];
        nb  += (double) b[i] * b[i];
    }
    cos_out = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
    rms_a = std::sqrt(na / (double) n);
    rms_b = std::sqrt(nb / (double) n);
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <a.gguf> <b.gguf> [tensor ...]\n", argv[0]);
        return 2;
    }
    GgufShardSet A, B;
    std::string err;
    if (!A.open(argv[1], err)) { std::fprintf(stderr, "a: %s\n", err.c_str()); return 1; }
    if (!B.open(argv[2], err)) { std::fprintf(stderr, "b: %s\n", err.c_str()); return 1; }

    std::vector<std::string> names;
    for (int i = 3; i < argc; ++i) names.emplace_back(argv[i]);

    if (!names.empty()) {
        for (const std::string & n : names) {
            double c = 0.0, ra = 0.0, rb = 0.0;
            if (compare(A, B, n, c, ra, rb)) {
                std::printf("  %-46s cos=%+.6f  rms %.5f / %.5f\n", n.c_str(), c, ra, rb);
            } else {
                std::printf("  %-46s (not comparable)\n", n.c_str());
            }
        }
        return 0;
    }

    // No names: sweep the layers and the model-level tensors, and report only
    // what disagrees. Anything above 0.999 is quantisation noise between two
    // different encodings of the same numbers.
    std::vector<std::string> probe;
    const char * per_layer[] = {
        "attn_qkv.weight", "attn_gate.weight", "attn_q.weight", "attn_k.weight",
        "attn_v.weight", "attn_output.weight", "attn_q_norm.weight",
        "attn_k_norm.weight", "hc_attn_norm.weight", "hc_attn_down.weight",
        "hc_attn_up.weight", "hc_attn_inject.weight", "hc_ffn_norm.weight",
        "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight",
        "ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight",
        "ffn_down_exps.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
        "ffn_down_shexp.weight", "ffn_gate_inp_shexp.weight",
        "ssm_alpha.weight", "ssm_beta.weight", "ssm_out.weight",
        "ssm_conv1d.weight", "ssm_norm.weight", "ssm_a", "ssm_dt.bias",
        "indexer.q_proj.weight", "indexer.k_proj.weight",
        "indexer.q_norm.weight", "indexer.k_norm.weight",
        "ple_key.weight", "ple_value.weight", "ple_norm_key.weight",
        "ple_norm_query.weight", "ple_norm_conv.weight", "ple_conv1d.weight",
    };
    for (int il = 0; il < 48; ++il) {
        for (const char * s : per_layer) {
            probe.push_back("blk." + std::to_string(il) + "." + s);
        }
    }
    for (const char * s : { "token_embd.weight", "output.weight",
                            "output_hc_norm.weight", "output_hc_down.weight",
                            "output_hc_up.weight" }) {
        probe.emplace_back(s);
    }

    int compared = 0, differ = 0;
    for (const std::string & n : probe) {
        double c = 0.0, ra = 0.0, rb = 0.0;
        if (!compare(A, B, n, c, ra, rb)) continue;
        compared++;
        if (c < 0.999) {
            differ++;
            std::printf("  %-46s cos=%+.6f  rms %.5f / %.5f\n", n.c_str(), c, ra, rb);
        }
    }
    std::printf("compared %d tensors, %d disagree beyond quantisation noise\n",
                compared, differ);
    return differ ? 3 : 0;
}
