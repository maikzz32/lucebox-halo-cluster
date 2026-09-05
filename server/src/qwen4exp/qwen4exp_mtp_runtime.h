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
// KNOWN FAILURE, and where it is not. Enabling this crashes inside
// ggml_backend_graph_compute on the draft graph. What that is NOT, each ruled
// out by a run rather than by reading:
//
//   - the block. DFLASH_QWEN4EXP_MTP_NO_BLOCK=1 leaves it out, reducing the
//     graph to the glue plus a mixer and a head, and it still crashes.
//   - the CUDA graph cache. GGML_CUDA_DISABLE_GRAPHS=1 does not help, so it is
//     not the target's cached graph objecting to a foreign one mid-loop.
//   - everything before the compute. Checkpoints under
//     DFLASH_QWEN4EXP_MTP_TRACE=1 reach "compute" every time.
//
// What it needs next is a stack trace, which needs a debug build; a segfault
// inside a release container leaves nothing to go on. The default path is
// unaffected -- without DFLASH_QWEN4EXP_MTP nothing here runs, and the model
// still scores 8/10 at 24-25 tok/s on one node.
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
                             int32_t next_token);

// Score the pending draft against what the target actually produced.
void qwen4exp_mtp_score(Qwen4ExpMtpRuntime & rt, int32_t actual_token);

// One line, at the end of a request.
void qwen4exp_mtp_report(Qwen4ExpMtpRuntime & rt);

}  // namespace dflash::common
