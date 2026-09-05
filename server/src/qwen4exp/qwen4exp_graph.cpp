#include "qwen4exp/qwen4exp_graph.h"
#include "qwen4exp/qwen4exp_cluster.h"
#include "qwen4exp/qwen4exp_probe.h"

#include <cmath>
#include <cstdlib>

namespace dflash::common {

ggml_tensor * qwen4exp_hc_init(ggml_context * ctx,
                               ggml_tensor *  embd,
                               int            n_embd,
                               int            n_hc) {
    const int64_t nt = embd->ne[1];
    ggml_tensor * seed = ggml_reshape_3d(ctx, embd, n_embd, 1, nt);
    return ggml_repeat_4d(ctx, seed, n_embd, n_hc, nt, 1);
}

ggml_tensor * qwen4exp_hc_mix(ggml_context * ctx,
                              ggml_cgraph *  gf,
                              Qwen4ExpClusterRuntime * cluster,
                              ggml_tensor *  state,
                              ggml_tensor *  w_norm,
                              ggml_tensor *  w_down,
                              ggml_tensor *  w_up,
                              ggml_tensor *  w_inject,
                              ggml_tensor ** inject_out,
                              int            n_embd,
                              int            n_hc,
                              float          rms_eps) {
    const int64_t hc_dim = (int64_t) n_hc * n_embd;
    const int64_t nt     = state->ne[2];

    // Grouped RMSNorm: the reduction runs over ne[0], i.e. over ONE stream, and
    // the resulting scale is then applied across all of them by the flat
    // [n_hc*n_embd] gamma. Normalising the flattened vector instead would mix
    // the streams' magnitudes together and is a different function.
    ggml_tensor * xn3 = ggml_rms_norm(ctx, state, rms_eps);

    // Stream c lives at c*n_embd + i in the flat [n_hc*n_embd] weights, not at
    // i*n_hc + c. The two readings accept exactly the same tensor shapes, so
    // nothing in the file's metadata distinguishes them, but the weights
    // themselves do: grouped stream-major, the four blocks of every hc norm
    // have visibly different statistics (output_hc_norm means 2.45, 3.62,
    // 4.98, 3.96), which is what four separately trained per-stream gammas
    // look like. Grouped by stride n_hc they come out identical to three
    // decimals -- the signature of four distinct blocks averaged together.
    ggml_tensor * xn = ggml_reshape_2d(ctx, xn3, hc_dim, nt);
    xn = ggml_mul(ctx, xn, w_norm);

    // Low-rank gate over the whole flattened state. The 1/n_hc before the silu
    // keeps the pre-activation in the range the weights were trained for.
    ggml_tensor * lo = ggml_mul_mat(ctx, w_down, xn);
    lo = ggml_silu(ctx, ggml_scale(ctx, lo, 1.0f / (float) n_hc));
    ggml_tensor * up = ggml_mul_mat(ctx, w_up, lo);
    // The low-rank dimension is what splits here, so `up` summed only over
    // this rank's part of it. Complete the sum before the sigmoid: after it,
    // the values are no longer additive.
    if (cluster) {
        up = qwen4exp_cluster_allreduce_node(ctx, ggml_cont(ctx, up), *cluster);
    }
    ggml_tensor * gate = ggml_sigmoid(ctx, up);

    qwen4exp_probe_add(ctx, gf, "  xn",   -1, xn);
    qwen4exp_probe_add(ctx, gf, "  lo",   -1, lo);
    qwen4exp_probe_add(ctx, gf, "  gate", -1, gate);

    // Gate and collapse. Written out this is a multiply, one contiguous copy
    // per stream, n_hc-1 adds and a scale -- nine kernels, in a path that runs
    // ninety-six times a token, and those copies alone were 424 of the ~3500
    // dispatches a decode token costs. DFLASH_QWEN4EXP_FUSE_HC=1 does the same
    // arithmetic in one kernel. The unfused form stays because it is the
    // reference the fused one is checked against, and because it runs on
    // backends the fused op does not.
    static const bool fuse_hc = []() {
        const char * e = std::getenv("DFLASH_QWEN4EXP_FUSE_HC");
        return e && std::atoi(e) == 1;
    }();

    ggml_tensor * mixed = nullptr;
    if (fuse_hc) {
        mixed = ggml_hc_collapse(ctx, xn, gate, n_embd, n_hc);
    } else {
        ggml_tensor * gated = ggml_mul(ctx, xn, gate);
        gated = ggml_reshape_3d(ctx, gated, n_embd, n_hc, nt);

        // Collapse the streams by their mean. Each stream is a strided view of
        // the gated block; summing views and scaling once avoids materialising
        // a permutation.
        const size_t stream_stride = ggml_row_size(gated->type, n_embd);
        mixed = ggml_cont(
            ctx, ggml_view_2d(ctx, gated, n_embd, nt, stream_stride * n_hc, 0));
        for (int c = 1; c < n_hc; ++c) {
            // ggml_cont: the second operand of an add is a strided view here,
            // and whether every backend honours its row stride is not
            // something to leave to chance. The copy is one [n_embd, T] block
            // and the allocator reuses it.
            ggml_tensor * s = ggml_cont(ctx, ggml_view_2d(ctx, gated, n_embd, nt,
                                                          stream_stride * n_hc,
                                                          stream_stride * (size_t) c));
            mixed = ggml_add(ctx, mixed, s);
        }
        mixed = ggml_scale(ctx, mixed, 1.0f / (float) n_hc);
    }

    if (inject_out) {
        *inject_out = w_inject ? ggml_mul_mat(ctx, w_inject, xn) : nullptr;
    }
    return mixed;
}

namespace {

// The norm PLE uses everywhere: reduce over one stream, scale the flattened
// n_hc*n_embd layout by the gamma. Same grouping as hc_mix, different weight.
ggml_tensor * ple_grouped_norm(ggml_context * ctx, ggml_tensor * x,
                               ggml_tensor * w, int n_embd, int n_hc,
                               int64_t nt, float eps) {
    ggml_tensor * t = ggml_reshape_3d(ctx, x, n_embd, n_hc, nt);
    t = ggml_rms_norm(ctx, t, eps);
    t = ggml_reshape_2d(ctx, t, (int64_t) n_hc * n_embd, nt);
    t = ggml_mul(ctx, t, w);
    return ggml_reshape_3d(ctx, t, n_embd, n_hc, nt);
}

}  // namespace

ggml_tensor * qwen4exp_ple(ggml_context * ctx,
                           ggml_cgraph *  gf,
                           ggml_tensor *  state,
                           ggml_tensor *  ngram_embd,
                           ggml_tensor *  w_key,
                           ggml_tensor *  w_value,
                           ggml_tensor *  w_norm_key,
                           ggml_tensor *  w_norm_query,
                           ggml_tensor *  w_norm_conv,
                           ggml_tensor *  w_conv1d,
                           ggml_tensor *  conv_state,
                           int            n_embd,
                           int            n_hc,
                           int            conv_kernel,
                           int            ngram_size,
                           float          rms_eps,
                           ggml_tensor *  conv_input_capture) {
    const int64_t hc_dim = (int64_t) n_hc * n_embd;
    const int64_t nt     = state->ne[2];

    ggml_tensor * key   = ggml_mul_mat(ctx, w_key,   ngram_embd);   // [hc_dim, T]
    ggml_tensor * value = ggml_mul_mat(ctx, w_value, ngram_embd);   // [n_embd, T]

    key = ple_grouped_norm(ctx, key, w_norm_key, n_embd, n_hc, nt, rms_eps);
    ggml_tensor * query =
        ple_grouped_norm(ctx, state, w_norm_query, n_embd, n_hc, nt, rms_eps);

    // Per-stream dot product, then a signed square root before the sigmoid.
    // The clamp keeps sqrt away from a zero derivative; the sign is restored
    // afterwards so the gate stays centred rather than folded.
    ggml_tensor * s = ggml_sum_rows(ctx, ggml_mul(ctx, key, query));  // [1, n_hc, T]
    s = ggml_scale(ctx, s, 1.0f / sqrtf((float) n_embd));
    ggml_tensor * mag =
        ggml_sqrt(ctx, ggml_clamp(ctx, ggml_abs(ctx, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx, ggml_mul(ctx, ggml_sgn(ctx, s), mag));

    ggml_tensor * v3 = ggml_reshape_3d(ctx, value, n_embd, 1, nt);
    v3 = ggml_repeat_4d(ctx, v3, n_embd, n_hc, nt, 1);
    ggml_tensor * gated = ggml_mul(ctx, v3, gate);                   // [n_embd, n_hc, T]

    qwen4exp_probe_add(ctx, gf, "  ple_ngram", -1, ngram_embd);
    qwen4exp_probe_add(ctx, gf, "  ple_key",   -1, key);
    qwen4exp_probe_add(ctx, gf, "  ple_value", -1, value);
    qwen4exp_probe_add(ctx, gf, "  ple_s",     -1, s);
    qwen4exp_probe_add(ctx, gf, "  ple_gate",  -1, gate);
    qwen4exp_probe_add(ctx, gf, "  ple_gated", -1, gated);
    qwen4exp_probe_add(ctx, gf, "  ple_cstat", -1, conv_state);

    ggml_tensor * normed = ple_grouped_norm(
        ctx, ggml_reshape_2d(ctx, gated, hc_dim, nt), w_norm_conv,
        n_embd, n_hc, nt, rms_eps);
    normed = ggml_reshape_2d(ctx, normed, hc_dim, nt);

    // Depthwise causal convolution, dilated by the n-gram size, built as a sum
    // of shifted copies. ggml_conv_1d_dw is not used here: the reference notes
    // it as unreliable, and the shifted-sum form makes the tap arithmetic
    //   out[c, t] = sum_k w[k, c] * x[c, t - (K-1-k)*dilation]
    // visible at the call site.
    const int64_t hist = (int64_t) (conv_kernel - 1) * ngram_size;

    // History first, then this batch, on the token axis: [hist + T, hc_dim].
    ggml_tensor * state_2d = ggml_reshape_2d(ctx, conv_state, hist, hc_dim);
    ggml_tensor * batch_2d = ggml_cont(ctx, ggml_transpose(ctx, normed));  // [T, hc_dim]
    ggml_tensor * padded   = ggml_concat(ctx, state_2d, batch_2d, 0);

    ggml_tensor * conv_out = nullptr;
    for (int k = 0; k < conv_kernel; ++k) {
        const int64_t start = hist - (int64_t) (conv_kernel - 1 - k) * ngram_size;
        ggml_tensor * shifted = ggml_cont(ctx, ggml_transpose(ctx,
            ggml_view_2d(ctx, padded, nt, hc_dim, padded->nb[1],
                         ggml_row_size(padded->type, start))));   // [hc_dim, T]

        // Column k of the [kernel, hc_dim] kernel is one weight per channel.
        ggml_tensor * wk = ggml_cont(ctx,
            ggml_view_2d(ctx, w_conv1d, 1, hc_dim, w_conv1d->nb[1],
                         (size_t) k * w_conv1d->nb[0]));
        wk = ggml_reshape_1d(ctx, wk, hc_dim);
        if (wk->type != GGML_TYPE_F32) wk = ggml_cast(ctx, wk, GGML_TYPE_F32);

        ggml_tensor * term = ggml_mul(ctx, shifted, wk);
        conv_out = conv_out ? ggml_add(ctx, conv_out, term) : term;
    }

    // The same array the tail is cut from is what a rollback needs, so it is
    // handed out whole rather than re-derived: history and batch already lie
    // adjacent on the token axis, and the state after any prefix of the batch
    // is just a different window into it.
    if (conv_input_capture) {
        ggml_tensor * dst = ggml_view_2d(ctx, conv_input_capture, hist + nt,
                                         hc_dim, conv_input_capture->nb[1], 0);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, padded, dst));
    }

    // Carry the tail forward so a chunked prefill matches a single-shot one.
    ggml_tensor * tail = ggml_view_2d(ctx, padded, hist, hc_dim, padded->nb[1],
                                      ggml_row_size(padded->type, nt));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_cont(ctx, tail), state_2d));

    qwen4exp_probe_add(ctx, gf, "  ple_convraw", -1, conv_out);
    conv_out = ggml_silu(ctx, conv_out);
    conv_out = ggml_reshape_3d(ctx, ggml_cont(ctx, conv_out), n_embd, n_hc, nt);

    qwen4exp_probe_add(ctx, gf, "  ple_conv", -1, conv_out);
    return ggml_add(ctx, state, ggml_add(ctx, gated, conv_out));
}

ggml_tensor * qwen4exp_hc_combine(ggml_context * ctx,
                                  ggml_tensor *  state,
                                  ggml_tensor *  block_out,
                                  ggml_tensor *  inject,
                                  int            n_embd,
                                  int            n_hc) {
    const int64_t nt = state->ne[2];

    // 2*sigmoid centres the scatter weights on 1, so an injection of zero is a
    // plain residual add rather than a halving of the block's contribution.
    //
    // The injections come out near -15 per stream, which drives the gate
    // nearly shut, and that looked wrong for a long time: the weights are
    // zero-mean with rms 0.011, and a random alignment against a unit-rms xn
    // over 10240 dimensions predicts about -0.5. Six other readings of this
    // formula were tried against a calibrated score and every one produced
    // nonsense. Upstream's own trace settles it -- its hc_inject-0 is
    // [-14.93, -14.93, -8.63, -14.62] against this tree's [-15.90, -16.20,
    // -9.16, -16.15]. Trained weights are not randomly aligned with their
    // input, and the formula is right as written.
    ggml_tensor * w = ggml_scale(
        ctx, ggml_sigmoid(ctx, ggml_scale(ctx, inject, 1.0f / (float) n_hc)), 2.0f);

    // Materialised to the carrier's shape rather than left to broadcast: the
    // scatter weight would otherwise broadcast along ne[0], the fastest axis,
    // from one to n_embd, which is the rarest shape of broadcast there is.
    w = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, w, 1, n_hc, nt),
                       n_embd, n_hc, nt, 1);

    ggml_tensor * b = ggml_reshape_3d(ctx, block_out, n_embd, 1, nt);
    b = ggml_repeat_4d(ctx, b, n_embd, n_hc, nt, 1);

    return ggml_add(ctx, state, ggml_mul(ctx, b, w));
}

}  // namespace dflash::common
