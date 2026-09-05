// qwen4exp — hyper-connection residual carrier.
//
// This architecture keeps `n_hc` parallel residual streams of width n_embd
// between blocks, shaped [n_embd, n_hc, T], and has no layer norms at all: the
// mixer that reads a stream out is the norm. Every sublayer is therefore
// wrapped in a pair --
//
//   cur, inject = hc_mix(state, norm, down, up, inject_w)   // [n_embd, T]
//   cur         = <the ordinary attention / delta-net / FFN block>
//   state       = hc_combine(state, cur, inject)            // [n_embd, n_hc, T]
//
// -- and the blocks between them are exactly the ones qwen35 already builds,
// because they see the same [n_embd, T] they always did.
//
// The final mixer is the output norm; there is no separate one.
//
// The semantics here follow the reference implementation of this architecture
// in upstream llama.cpp (merged as ggml-org/llama.cpp#27742). They are not
// reconstructed from tensor shapes: the shapes alone do not say how four
// streams collapse to one, and guessing would produce a model that runs and
// cannot be shown to be wrong.
//
// Include convention: #include "qwen4exp/qwen4exp_graph.h"

#pragma once

#include "internal.h"

#include "ggml.h"

namespace dflash::common {

// Read one [n_embd, n_tokens] working vector out of the [n_embd, n_hc,
// n_tokens] hyper-connection state.
//
//   grouped RMSNorm over a single stream, scaled by the [n_hc*n_embd] gamma
//   gate = sigmoid(up(silu(down(xn) / n_hc)))       elementwise over n_hc*n_embd
//
// `cluster` non-null means this rank holds a slice of the low-rank dimension:
// down produces part of it, up consumes that part, and the product is a
// partial sum that one reduction completes before the sigmoid.
//   mixed = mean over the n_hc streams of (xn * gate)
//
// When `inject_out` is non-null it receives the [n_hc, n_tokens] scatter
// weights for the matching hc_combine; pass nullptr for the final mixer, which
// has no block to write back.
ggml_tensor * qwen4exp_hc_mix(ggml_context * ctx,
                              ggml_cgraph *  gf,          // for the magnitude probe
                              struct Qwen4ExpClusterRuntime * cluster,  // null = whole
                              ggml_tensor *  state,       // [n_embd, n_hc, T]
                              ggml_tensor *  w_norm,      // [n_hc*n_embd]
                              ggml_tensor *  w_down,      // [n_hc*n_embd, low_rank]
                              ggml_tensor *  w_up,        // [low_rank, n_hc*n_embd]
                              ggml_tensor *  w_inject,    // [n_hc*n_embd, n_hc] or null
                              ggml_tensor ** inject_out,  // [n_hc, T] or null
                              int            n_embd,
                              int            n_hc,
                              float          rms_eps);

// Write a block's [n_embd, T] output back into every stream, weighted by
// 2*sigmoid(inject / n_hc). Centring on 1 means a zero injection is a plain
// residual add rather than a halving.
ggml_tensor * qwen4exp_hc_combine(ggml_context * ctx,
                                  ggml_tensor *  state,      // [n_embd, n_hc, T]
                                  ggml_tensor *  block_out,  // [n_embd, T]
                                  ggml_tensor *  inject,     // [n_hc, T]
                                  int            n_embd,
                                  int            n_hc);

// Seed the carrier: n_hc identical copies of the token embedding.
ggml_tensor * qwen4exp_hc_init(ggml_context * ctx,
                               ggml_tensor *  embd,   // [n_embd, T]
                               int            n_embd,
                               int            n_hc);

// PLE, on the single layer named by ple.layers.
//
// A retrieval over the hyper-connection state: the n-gram embedding is
// projected to a key and a value, the key is scored against the state per
// stream, and a signed-square-root sigmoid of that score gates the value into
// every stream. A causal convolution dilated by the n-gram size runs alongside
// and carries state across tokens.
//
// `ngram_embd` is [n_embd, T], gathered on the host -- see qwen4exp_ple_rows.
// `conv_state` is [(conv_kernel-1)*ngram_size, n_hc*n_embd, 1] and is updated:
// the new tail is written back so a chunked prefill matches a single-shot one.
//
// Returns the updated [n_embd, n_hc, T] state.
ggml_tensor * qwen4exp_ple(ggml_context * ctx,
                           ggml_cgraph *  gf,
                           ggml_tensor *  state,        // [n_embd, n_hc, T]
                           ggml_tensor *  ngram_embd,   // [n_embd, T]
                           ggml_tensor *  w_key,        // [n_embd, n_hc*n_embd]
                           ggml_tensor *  w_value,      // [n_embd, n_embd]
                           ggml_tensor *  w_norm_key,   // [n_hc*n_embd]
                           ggml_tensor *  w_norm_query, // [n_hc*n_embd]
                           ggml_tensor *  w_norm_conv,  // [n_hc*n_embd]
                           ggml_tensor *  w_conv1d,     // [conv_kernel, n_hc*n_embd]
                           ggml_tensor *  conv_state,
                           int            n_embd,
                           int            n_hc,
                           int            conv_kernel,
                           int            ngram_size,
                           float          rms_eps);

// One qwen4exp block on the carrier: the mixer pair around an attention or
// delta-net block and around the MoE. Defined beside qwen35's layer builders,
// because the blocks between the mixers are exactly qwen35's.
//
// `hc_state` is [n_embd, n_hc, n_tokens] and is updated in place. The MTP head
// builds its single block through this too, presenting itself as a one-layer
// model so the weights come from its own module rather than the target's.
ggml_tensor * build_qwen4exp_layer(ggml_context *        ctx,
                                   ggml_cgraph *         gf,
                                   const TargetWeights & w,
                                   TargetCache &         cache,
                                   int                   layer_idx,
                                   ggml_tensor **        hc_state,
                                   ggml_tensor *         positions,
                                   ggml_tensor *         attn_mask,
                                   int                   kv_start,
                                   int                   n_tokens,
                                   int                   fa_window,
                                   ggml_tensor *         kv_write_rows,
                                   ggml_tensor *         parent_ids);

}  // namespace dflash::common
