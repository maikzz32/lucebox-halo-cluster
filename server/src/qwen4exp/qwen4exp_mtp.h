// qwen4exp_mtp.h - the multi-token-prediction head, and the one number that
// says whether it is wired correctly.
//
// Qwen3.8-Flash-Next ships an MTP module: one more qwen4exp layer plus the
// glue that feeds it. Given the carrier the target produced for token t and
// the token t+1 that was sampled from it, the head predicts token t+2, so a
// verify pass can check two tokens for the price of one forward.
//
// That is the lever this cluster needs. Its decode step is bound by
// synchronisation rather than bandwidth -- the collective costs the forward
// its CUDA graph, and 84 collectives cost the same as 97 -- so q tokens per
// forward divides the whole of that cost by q, on top of making every matmul
// q times larger.
//
// THE WIRING IS INFERRED, AND THE INFERENCE IS TESTABLE. Upstream llama.cpp
// does not implement this path and the converter only names the tensors, so
// the shapes had to settle it:
//
//   enorm   [n_embd]                  normalises the token embedding
//   hnorm   [n_hc*n_embd]             grouped norm over the incoming carrier
//   eh_proj [2*n_embd, n_embd]        per stream: fc(concat(e, stream_c))
//   the layer                         one ordinary qwen4exp layer
//   hc_head_{norm,down,up}            its output mixer, as output_hc_* is
//   output                            the head's own lm_head
//
// eh_proj taking 2*n_embd while hnorm is n_hc*n_embd wide is what fixes it:
// the projection cannot see the whole carrier at once, so it sees one stream
// at a time, and its output is a stream of the carrier the layer then runs on.
//
// Before anything is built on top of this, qwen4exp_mtp_acceptance() measures
// it: draft a token, let the target produce its own, and count the matches. A
// correct head agrees with the target most of the time; a mis-wired one agrees
// at chance, which for a 248320-token vocabulary is never. That number decides
// whether the reading above is right, and it costs one run to get.
//
// Include convention: #include "qwen4exp/qwen4exp_mtp.h"

#pragma once

#include "internal.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <string>
#include <vector>

namespace dflash::common {

// The head's weights. Its own embedding and lm_head come with the file rather
// than being borrowed from the target: the module is distributed standalone,
// and a target that has been abliterated or requantised separately would not
// match anyway.
struct Qwen4ExpMtpWeights {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    CpuEmbedder embedder;          // token_embd, on the host like the target's
    ggml_tensor * output    = nullptr;   // [n_embd, n_vocab]
    ggml_tensor * enorm     = nullptr;   // [n_embd]
    ggml_tensor * hnorm     = nullptr;   // [n_hc*n_embd]
    ggml_tensor * eh_proj   = nullptr;   // [2*n_embd, n_embd]
    ggml_tensor * head_norm = nullptr;   // [n_hc*n_embd]
    ggml_tensor * head_down = nullptr;   // [n_hc*n_embd, hc_low_rank]
    ggml_tensor * head_up   = nullptr;   // [hc_low_rank, n_hc*n_embd]
    TargetLayer   layer;                 // the one qwen4exp block

    int layer_index = -1;                // its index in the file (block_count-1)
    bool ok() const { return ctx != nullptr; }
    ~Qwen4ExpMtpWeights();
};

// Load the module. `target` supplies the hyperparameters every shape is
// checked against, so a module built for a different configuration fails here
// rather than as a draft that is never accepted.
bool load_qwen4exp_mtp(const std::string & path,
                       const TargetWeights & target,
                       ggml_backend_t backend,
                       Qwen4ExpMtpWeights & out);

// Build the draft from the carrier the target produced and the tokens sampled
// from it, giving one predicted token per position.
//
// NOT YET IMPLEMENTED. The loader lands first on purpose: it is what proves
// the module can be read and its shapes agree with the target, and the draft
// graph is only worth writing against weights that are known to be there.
ggml_tensor * build_qwen4exp_mtp_draft(ggml_context * ctx,
                                       ggml_cgraph *  gf,
                                       const Qwen4ExpMtpWeights & w,
                                       const TargetWeights & target,
                                       ggml_tensor * carrier,     // [n_embd, n_hc, T]
                                       ggml_tensor * embed_next,  // [n_embd, T]
                                       TargetCache & cache);

}  // namespace dflash::common
