#include "qwen4exp/qwen4exp_internal.h"

#include <cstdint>
#include <vector>

namespace dflash::common {

// Row indices into per_layer_token_embd, one per (token, head).
//
// ple_n_heads = (ngram_size - 1) * heads_per_ngram: for each n-gram order n in
// 2..ngram_size, the token and its n-1 predecessors are folded into one 64-bit
// value, and that value feeds heads_per_ngram heads, each with its own modulus
// and offset. With ngram_size 3 and heads_per_ngram 8 that is 16 heads, which
// is exactly how many head_vocab_sizes the file carries.
//
// EOS cuts context: a predecessor that is EOS, or missing because the sequence
// has not got that far back, makes it and everything older read as EOS. The
// token's own EOS does not cut its own context.
//
// `prev` holds ngram_size-1 predecessors per token, oldest first, with a
// negative value for "not available".
void qwen4exp_ple_rows(const TargetWeights & w,
                       const int32_t * tokens,
                       const int32_t * prev,
                       int n_tokens,
                       std::vector<int32_t> & out) {
    const int n_gram   = w.ple_ngram_size;
    const int n_heads  = w.ple_n_heads;
    const int per_gram = w.ple_heads_per_ngram;
    const int64_t eos  = w.ple_eos_token_id;
    const int n_prev   = n_gram - 1;

    out.assign((size_t) n_heads * n_tokens, 0);
    if (n_gram < 2 || n_heads <= 0) return;

    std::vector<int64_t> ctx((size_t) n_gram);
    for (int i = 0; i < n_tokens; ++i) {
        ctx[0] = tokens[i];
        bool cut = false;
        for (int s = 1; s < n_gram; ++s) {
            const int32_t t = cut ? -1 : prev[(size_t) i * n_prev + (n_prev - s)];
            cut = cut || t < 0 || (int64_t) t == eos;
            ctx[(size_t) s] = cut ? eos : (int64_t) t;
        }

        for (int n = 2; n <= n_gram; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * w.ple_layer_multipliers[0];
            for (int j = 1; j < n; ++j) {
                mixed ^= (uint64_t) ctx[(size_t) j] * w.ple_layer_multipliers[(size_t) j];
            }
            const int base = (n - 2) * per_gram;
            for (int g = 0; g < per_gram; ++g) {
                const int h = base + g;
                out[(size_t) i * n_heads + h] = (int32_t) (
                    mixed % w.ple_head_vocab_sizes[(size_t) h] +
                    w.ple_head_offsets[(size_t) h]);
            }
        }
    }
}

}  // namespace dflash::common
