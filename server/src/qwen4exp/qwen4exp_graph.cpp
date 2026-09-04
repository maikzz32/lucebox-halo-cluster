#include "qwen4exp/qwen4exp_graph.h"

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
    ggml_tensor * xn = ggml_rms_norm(ctx, state, rms_eps);
    xn = ggml_reshape_2d(ctx, xn, hc_dim, nt);
    xn = ggml_mul(ctx, xn, w_norm);

    // Low-rank gate over the whole flattened state. The 1/n_hc before the silu
    // keeps the pre-activation in the range the weights were trained for.
    ggml_tensor * lo = ggml_mul_mat(ctx, w_down, xn);
    lo = ggml_silu(ctx, ggml_scale(ctx, lo, 1.0f / (float) n_hc));
    ggml_tensor * gate = ggml_sigmoid(ctx, ggml_mul_mat(ctx, w_up, lo));

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

ggml_tensor * qwen4exp_hc_combine(ggml_context * ctx,
                                  ggml_tensor *  state,
                                  ggml_tensor *  block_out,
                                  ggml_tensor *  inject,
                                  int            n_embd,
                                  int            n_hc) {
    const int64_t nt = state->ne[2];

    // 2*sigmoid centres the scatter weights on 1, so an injection of zero is a
    // plain residual add rather than a halving of the block's contribution.
    ggml_tensor * w = ggml_sigmoid(ctx, ggml_scale(ctx, inject, 1.0f / (float) n_hc));
    w = ggml_scale(ctx, w, 2.0f);
    w = ggml_reshape_3d(ctx, w, 1, n_hc, nt);

    ggml_tensor * b = ggml_reshape_3d(ctx, block_out, n_embd, 1, nt);
    b = ggml_repeat_4d(ctx, b, n_embd, n_hc, nt, 1);

    return ggml_add(ctx, state, ggml_mul(ctx, b, w));
}

}  // namespace dflash::common
