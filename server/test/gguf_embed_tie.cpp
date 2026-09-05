// gguf_embed_tie - does a checkpoint's unembedding know its embedding?
//
// Every transformer correlates output.weight[v] with token_embd[v] for the
// same token v, whether or not the two are tied, because both encode the
// identity of v. The cosine between them is well above the 1/sqrt(n_embd)
// floor that two unrelated rows give.
//
// That makes the comparison a load-time check on a model this tree has never
// run before: if the self-similarity sits in the noise, the two tables are not
// being read in the same basis, and nothing built on top of them can produce
// sensible logits. It is worth its own tool because the interesting run is the
// CONTROL -- the same numbers for a model that is known to generate correctly,
// which is what says whether an unfamiliar checkpoint is broken or whether the
// expectation was wrong.
//
//   gguf_embed_tie <model.gguf> [n_ids]
//
// Split GGUFs are handled: pass any part.

#include "common/gguf_shards.h"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace dflash::common;

namespace {

// Dequantise row `r` of `t` into `dst`, which must hold ne[0] floats.
bool row_to_float(const GgufShardTensor & t, int64_t r, float * dst) {
    const ggml_type_traits * tr = ggml_get_type_traits(t.meta->type);
    if (!tr || !tr->to_float) {
        std::fprintf(stderr, "no to_float for type %s\n",
                     ggml_type_name(t.meta->type));
        return false;
    }
    const int64_t nc = t.meta->ne[0];
    const size_t  rb = ggml_row_size(t.meta->type, nc);
    tr->to_float(static_cast<const char *>(t.data) + (size_t) r * rb, dst, nc);
    return true;
}

double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double) a[i] * b[i];
        na  += (double) a[i] * a[i];
        nb  += (double) b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [n_ids]\n", argv[0]);
        return 2;
    }
    const int n_ids = argc > 2 ? std::atoi(argv[2]) : 8;

    GgufShardSet shards;
    std::string err;
    if (!shards.open(argv[1], err)) {
        std::fprintf(stderr, "open: %s\n", err.c_str());
        return 1;
    }

    GgufShardTensor emb, unemb;
    if (!shards.find("token_embd.weight", emb, err)) {
        std::fprintf(stderr, "token_embd.weight: %s\n", err.c_str());
        return 1;
    }
    if (!shards.find("output.weight", unemb, err)) {
        // A tied checkpoint has no separate head; the answer is then trivial.
        std::printf("no output.weight: embeddings are tied, nothing to compare\n");
        return 0;
    }

    const int64_t n_embd  = emb.meta->ne[0];
    const int64_t n_vocab = emb.meta->ne[1];
    if (unemb.meta->ne[0] != n_embd || unemb.meta->ne[1] != n_vocab) {
        std::fprintf(stderr, "shape mismatch: token_embd [%lld,%lld] vs output [%lld,%lld]\n",
                     (long long) n_embd, (long long) n_vocab,
                     (long long) unemb.meta->ne[0], (long long) unemb.meta->ne[1]);
        return 1;
    }
    std::printf("%s\n  token_embd %s [%lld x %lld]   output %s\n",
                argv[1], ggml_type_name(emb.meta->type),
                (long long) n_embd, (long long) n_vocab,
                ggml_type_name(unemb.meta->type));

    // Spread the ids over the vocabulary rather than taking a prefix: the low
    // ids are special tokens and the high ones are often padding, and neither
    // is representative.
    std::vector<int64_t> ids;
    for (int i = 0; i < n_ids; ++i) {
        ids.push_back((int64_t) ((double) (i + 1) / (n_ids + 1) * (double) n_vocab));
    }

    std::vector<float> e((size_t) n_embd), o((size_t) n_embd);
    double self_sum = 0.0, cross_sum = 0.0, cross_abs = 0.0;
    int self_n = 0, cross_n = 0;

    for (int a = 0; a < n_ids; ++a) {
        if (!row_to_float(emb, ids[(size_t) a], e.data())) return 1;
        for (int b = 0; b < n_ids; ++b) {
            if (!row_to_float(unemb, ids[(size_t) b], o.data())) return 1;
            const double c = cosine(e, o);
            if (a == b) { self_sum += c; self_n++; }
            else        { cross_sum += c; cross_abs += std::fabs(c); cross_n++; }
        }
    }

    const double self_mean  = self_n  ? self_sum  / self_n  : 0.0;
    const double cross_mean = cross_n ? cross_sum / cross_n : 0.0;
    const double cross_mag  = cross_n ? cross_abs / cross_n : 0.0;
    const double floor_     = 1.0 / std::sqrt((double) n_embd);

    std::printf("  mean cos(v, v)      = %+.4f   over %d ids\n", self_mean, self_n);
    std::printf("  mean cos(v, w), v!=w= %+.4f   mean |cos| = %.4f\n", cross_mean, cross_mag);
    std::printf("  random floor 1/sqrt(n_embd) = %.4f\n", floor_);
    const bool tied_basis = self_mean > 3.0 * cross_mag;
    std::printf("  => %s\n", tied_basis
        ? "the two tables share a basis"
        : "SELF-SIMILARITY IS IN THE NOISE: the tables do not share a basis");
    return tied_basis ? 0 : 3;
}
