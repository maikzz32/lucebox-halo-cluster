#include "qwen4exp/qwen4exp_graph.h"
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
    ggml_tensor * gate = ggml_sigmoid(ctx, ggml_mul_mat(ctx, w_up, lo));

    qwen4exp_probe_add(ctx, gf, "  xn",   -1, xn);
    qwen4exp_probe_add(ctx, gf, "  lo",   -1, lo);
    qwen4exp_probe_add(ctx, gf, "  gate", -1, gate);

    ggml_tensor * gated = ggml_mul(ctx, xn, gate);
    gated = ggml_reshape_3d(ctx, gated, n_embd, n_hc, nt);

    // Collapse the streams by their mean. Each stream is a strided view of the
    // gated block; summing views and scaling once avoids materialising a
    // permutation.
    const size_t stream_stride = ggml_row_size(gated->type, n_embd);
    ggml_tensor * mixed = ggml_cont(
        ctx, ggml_view_2d(ctx, gated, n_embd, nt, stream_stride * n_hc, 0));
    for (int c = 1; c < n_hc; ++c) {
        ggml_tensor * s = ggml_view_2d(ctx, gated, n_embd, nt,
                                       stream_stride * n_hc,
                                       stream_stride * (size_t) c);
        mixed = ggml_add(ctx, mixed, s);
    }
    mixed = ggml_scale(ctx, mixed, 1.0f / (float) n_hc);

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
                           float          rms_eps) {
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

    // Carry the tail forward so a chunked prefill matches a single-shot one.
    ggml_tensor * tail = ggml_view_2d(ctx, padded, hist, hc_dim, padded->nb[1],
                                      ggml_row_size(padded->type, nt));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_cont(ctx, tail), state_2d));

    conv_out = ggml_silu(ctx, conv_out);
    conv_out = ggml_reshape_3d(ctx, ggml_cont(ctx, conv_out), n_embd, n_hc, nt);

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
    // DFLASH_QWEN4EXP_HC_VARIANT selects a different reading of the same
    // tensors. It exists because the measured injections are far outside the
    // range this formula assumes -- around -19 where random alignment predicts
    // -0.5 -- which saturates every gate to zero and stops the blocks writing
    // back at all. Variant 1 is the control that matters: with the gate pinned
    // to 1 the four streams stay identical and the model degenerates to an
    // ordinary residual network, so coherent text there would say the fault is
    // in this formula and nowhere else.
    static const int variant = []() {
        const char * s = std::getenv("DFLASH_QWEN4EXP_HC_VARIANT");
        return s ? std::atoi(s) : 0;
    }();

    ggml_tensor * w = nullptr;
    switch (variant) {
        case 1:  // control: plain residual add into every stream
            w = ggml_scale(ctx, ggml_sigmoid(ctx, ggml_scale(ctx, inject, 0.0f)), 2.0f);
            break;
        case 2: {
            // Centre the injection across the streams before the sigmoid. The
            // measured offset is nearly the same for all four, so if it is an
            // artefact rather than a signal, removing it restores a gate that
            // spans zero to two. (1 + tanh(x) is not a separate variant: it
            // equals 2*sigmoid(2x), which only saturates harder.)
            ggml_tensor * mu = ggml_repeat(ctx, ggml_mean(ctx, inject), inject);
            ggml_tensor * centred = ggml_sub(ctx, inject, mu);
            w = ggml_scale(
                ctx, ggml_sigmoid(ctx, ggml_scale(ctx, centred, 1.0f / (float) n_hc)), 2.0f);
            break;
        }
        case 3:  // no 1/n_hc: the scale is the only free constant in the formula
            w = ggml_scale(ctx, ggml_sigmoid(ctx, inject), 2.0f);
            break;
        case 4:  // 1/sqrt(n_hc*n_embd): the projection's own width
        case 5: {
            // Divide by the square root of the contracted width instead of by
            // n_hc. The injection is a dot product over 10240 dimensions and
            // comes out near -19, which drives the gate to 0.02 and leaves the
            // blocks contributing nothing. Scaling by sqrt of that width puts
            // it at -0.19 and the gate near 0.9, which is the range a trained
            // depth connection is initialised in. Variant 1 reached a gate of
            // one too, but by discarding the per-stream variation with it --
            // the streams stayed identical and the carrier stopped being wide
            // at all. This keeps the variation and only changes the scale.
            const float d = variant == 4
                ? (float) ((int64_t) n_hc * n_embd)
                : (float) n_embd;
            w = ggml_scale(
                ctx, ggml_sigmoid(ctx, ggml_scale(ctx, inject, 1.0f / sqrtf(d))), 2.0f);
            break;
        }
        default:
            w = ggml_scale(
                ctx, ggml_sigmoid(ctx, ggml_scale(ctx, inject, 1.0f / (float) n_hc)), 2.0f);
            break;
    }
    w = ggml_reshape_3d(ctx, w, 1, n_hc, nt);

    ggml_tensor * b = ggml_reshape_3d(ctx, block_out, n_embd, 1, nt);
    b = ggml_repeat_4d(ctx, b, n_embd, n_hc, nt, 1);

    return ggml_add(ctx, state, ggml_mul(ctx, b, w));
}

}  // namespace dflash::common
