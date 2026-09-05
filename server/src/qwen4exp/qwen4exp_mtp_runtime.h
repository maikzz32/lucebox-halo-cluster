// qwen4exp_mtp_runtime.h - drafting with the MTP head, and counting how often
// it is right.
//
// The head's wiring had to be inferred (see qwen4exp_mtp.h). Its acceptance
// rate is the number that settles the inference, and it is worth measuring on
// its own before any verify loop is built: drafting costs one small forward
// and needs neither rollback nor a second target pass, while the verify loop
// needs both and would confound a wiring mistake with an integration one.
//
// DFLASH_QWEN4EXP_MTP=<path> turns the measurement on. Each decode step drafts
// the token after next; the following step compares it against what the target
// actually produced and counts. The rate is printed when the request ends.
//
// A correct head agrees with the target most of the time. A mis-wired one
// agrees at chance, and against a 248320-token vocabulary chance rounds to
// never -- so a single run separates the two without ambiguity.
//
// WHERE IT STANDS. The measurement runs; the head is not yet good enough to
// speculate with. Acceptance on free prose:
//
//   4%   the first reading that ran
//   0%   with the two halves of the concatenation swapped -- so the
//        reference's order, embedding first, is confirmed
//   4%   with the carrier collapsed to one vector before the projection
//        instead of projected per stream, so that choice is not what matters
//   8%   reading enorm and hnorm as (1 + w) rather than as w
//
// Chance against a 248320-token vocabulary is 0.0004%, so even the first
// reading was four orders of magnitude better than noise: the head is doing
// something real and something is still wrong. Speculation needs roughly 60%
// to pay for itself.
//
// The (1 + w) finding is the useful one. The target's hyper-connection norms
// average about one, which is what a converter that folded the plus-one leaves
// behind; enorm and hnorm average 0.24 and 0.67, which is what it leaves when
// it did not, and reading them the other way doubled the rate.
//
// Giving the head its own KV history over the context changed nothing, and
// neither did rotating at the target's absolute position rather than its own
// -- expected in hindsight, since attention only sees the differences.
//
// The block runs against a one-token KV cache that is reset every step, which
// is not what a served drafter would do. It costs the head its own attention
// history and therefore some accuracy, but almost all of the signal reaches it
// through eh_proj from the target's carrier, so the measurement still
// separates a correct wiring from a wrong one -- which is all it is for.
//
// Include convention: #include "qwen4exp/qwen4exp_mtp_runtime.h"

#pragma once

#include "qwen4exp/qwen4exp_mtp.h"

#include "internal.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct Qwen4ExpMtpRuntime {
    Qwen4ExpMtpWeights   w;
    TargetCache          cache;          // one attention block's worth
    ggml_backend_t       backend = nullptr;
    bool                 owns_backend = false;

    // Graph, allocator and inputs, built once and reused. Everything else in
    // this tree keeps its StepGraph and its gallocr alive across steps; a
    // fresh allocator per step means a backend buffer created and freed inside
    // the decode loop, next to the target's live one.
    ggml_context * ctx   = nullptr;
    ggml_cgraph *  gf    = nullptr;
    ggml_gallocr_t alloc = nullptr;
    // The head's own position in its own KV cache. It attends to everything it
    // has drafted from, which is what a next-token predictor needs; with a
    // single visible token the head still beats chance by four orders of
    // magnitude but only reaches 3-5%.
    int            pos   = 0;
    int            max_ctx = 0;
    ggml_tensor *  in_carrier = nullptr;
    ggml_tensor *  in_embed   = nullptr;
    ggml_tensor *  in_pos     = nullptr;
    ggml_tensor *  out_draft  = nullptr;

    // The draft made at the previous step, waiting to be compared against the
    // token the target produces next. -1 when there is none.
    int32_t pending = -1;

    uint64_t drafted = 0;
    uint64_t matched = 0;

    bool ready() const { return w.ok(); }
    double acceptance() const {
        return drafted ? (double) matched / (double) drafted : 0.0;
    }
};

// Open the module named by DFLASH_QWEN4EXP_MTP, if any. Returns true when
// there is nothing to do as well as when the head loaded; false only when a
// path was given and could not be honoured, so a typo is not silently ignored.
bool qwen4exp_mtp_open(const TargetWeights & target,
                       ggml_backend_t backend,
                       int max_ctx,
                       Qwen4ExpMtpRuntime & rt);

// Draft from the carrier the target just produced and the token sampled from
// it. Records the draft for the next call to score.
void qwen4exp_mtp_draft_step(Qwen4ExpMtpRuntime & rt,
                             const TargetWeights & target,
                             const float * carrier,   // [n_embd * n_hc]
                             int32_t next_token,
                             int abs_pos);            // the target's position

// Score the pending draft against what the target actually produced.
void qwen4exp_mtp_score(Qwen4ExpMtpRuntime & rt, int32_t actual_token);

// One line, at the end of a request.
void qwen4exp_mtp_report(Qwen4ExpMtpRuntime & rt);

}  // namespace dflash::common
