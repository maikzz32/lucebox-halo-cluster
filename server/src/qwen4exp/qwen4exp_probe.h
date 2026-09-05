// qwen4exp — per-sublayer magnitude probe.
//
// A hyper-connection carrier has no layer norms to hold it in place: every
// block reads through a mixer and writes back through a gate, so an error in
// either compounds silently down 48 layers. The output stays fluent long after
// it stops being right, which makes generated text useless as a signal.
//
// This records the RMS of the carrier and of each block's output. A correct
// forward pass settles into a stable band within a few layers; a wrong reading
// of the mixer shows up as a drift or a collapse that is visible by layer 5,
// long before the tokens look wrong.
//
// Off unless DFLASH_QWEN4EXP_RMS=1. When off, add() returns immediately and no
// nodes are added to the graph.
//
// Include convention: #include "qwen4exp/qwen4exp_probe.h"

#pragma once

#include "ggml.h"

namespace dflash::common {

bool qwen4exp_probe_enabled();

// Append a sum-of-squares reduction of `t` to the graph and remember it under
// `label` (layer index appended when il >= 0). `t` is not modified.
//
// `gf` may be null where a caller has a context but no graph -- inside a
// shared builder, say, whose signature should not grow a parameter for an
// instrument. The reduction nodes are then held until the next
// qwen4exp_probe_expand().
void qwen4exp_probe_add(ggml_context * ctx, ggml_cgraph * gf,
                        const char * label, int il, ggml_tensor * t);

// Attach everything added with a null `gf` to this graph.
void qwen4exp_probe_expand(ggml_cgraph * gf);

// Read every probe registered since the last report, print one line each, and
// clear the list. Call after the graph has been computed.
void qwen4exp_probe_report();

}  // namespace dflash::common
