// qwen4exp — a hybrid gated-delta-net / attention MoE with hyper-connections.
//
// Layer structure, 48 blocks: full attention where (il + 1) % 4 == 0 (layers
// 3, 7, ... 47, twelve of them), gated delta net everywhere else. Both block
// kinds are the ones qwen35 already builds; what is new here is the residual
// carrier around them.
//
// HYPER-CONNECTIONS. This architecture carries n_hc = 4 residual streams of
// width n_embd instead of one, so the state between blocks is [n_hc * n_embd].
// Each sublayer reads a single n_embd vector out of that state through a
// low-rank pair (hc_*_down [n_hc*n_embd, low_rank], hc_*_up [low_rank,
// n_hc*n_embd]) and writes its output back weighted per stream by hc_*_inject
// [n_hc*n_embd, n_hc]. hc_attn_norm / hc_ffn_norm REPLACE the attn_norm and
// ffn_norm of an ordinary transformer -- this model has no tensor by either of
// those names. output_hc_{norm,down,up} collapse the four streams to one
// before the output projection.
//
// PLE. One layer (ple.layers[0], layer 1 in this checkpoint) carries a
// query/key/value retrieval over the hyper-connection state, with its own
// norms and a causal convolution that keeps recurrent state across tokens.
//
// QSA. Every full-attention layer carries a four-tensor indexer that scores
// keys and keeps the top 2048 before attention proper. Below a prompt length
// of top_k it selects everything, so dense attention is exactly equivalent
// there -- which is what makes it safe to leave out of a first version.
//
// Include convention: #include "qwen4exp/qwen4exp_internal.h"

#pragma once

#include "common/gguf_shards.h"
#include "internal.h"

#include <string>
#include <vector>

namespace dflash::common {

// Read qwen4exp.* hyperparameters from the shard set into `out`, and verify
// every derived scalar against the shape of a tensor that must agree with it.
//
// The verification is the point. There is no reference implementation of this
// architecture in this tree, so a loader that silently mis-reads a dimension
// produces a model that runs and talks nonsense. Each equation below ties a
// metadata key to an `ne` that the file itself carries, and a mismatch fails
// the load with the two numbers named.
//
// Returns false with a human-readable `err`.
bool read_qwen4exp_hparams(const GgufShardSet & shards,
                           TargetWeights & out,
                           std::string & err);

// Row indices into per_layer_token_embd for one batch: ple_n_heads entries per
// token, head-minor. `prev` carries ngram_size-1 predecessors per token, oldest
// first, negative where the sequence does not reach that far back.
//
// This is host-side on purpose. The table is ~36 GiB and the gather reads
// ple_n_heads rows of 160 values per token -- about 10 KB -- so mapping it and
// gathering here keeps 36 GiB off a device that has no room for it, at no
// measurable bandwidth cost.
void qwen4exp_ple_rows(const TargetWeights & w,
                       const int32_t * tokens,
                       const int32_t * prev,
                       int n_tokens,
                       std::vector<int32_t> & out);

// Load a qwen4exp model from a (possibly split) GGUF into `out`.
bool load_qwen4exp_gguf(const std::string & path,
                        ggml_backend_t backend,
                        TargetWeights & out);

}  // namespace dflash::common
