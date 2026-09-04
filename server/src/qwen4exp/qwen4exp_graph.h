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

#include "ggml.h"

namespace dflash::common {

// Read one [n_embd, n_tokens] working vector out of the [n_embd, n_hc,
// n_tokens] hyper-connection state.
//
//   grouped RMSNorm over a single stream, scaled by the [n_hc*n_embd] gamma
//   gate = sigmoid(up(silu(down(xn) / n_hc)))       elementwise over n_hc*n_embd
//   mixed = mean over the n_hc streams of (xn * gate)
//
// When `inject_out` is non-null it receives the [n_hc, n_tokens] scatter
// weights for the matching hc_combine; pass nullptr for the final mixer, which
// has no block to write back.
ggml_tensor * qwen4exp_hc_mix(ggml_context * ctx,
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

}  // namespace dflash::common
