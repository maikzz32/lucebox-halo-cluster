// hc_collapse_check - is the fused hyper-connection collapse the same function
// as the graph it replaces?
//
// The fused op computes, in one kernel,
//
//   dst[i, t] = (1/n_hc) * sum_c a[c*n_embd + i, t] * b[c*n_embd + i, t]
//
// which the unfused mixer builds out of a multiply, one contiguous copy per
// stream, n_hc-1 adds and a scale. Swapping them in changed the model's output
// from the fourth token on, and a difference that early is not something to
// explain away as rounding without checking. So: both formulations, same
// inputs, same backend, compared elementwise.
//
//   hc_collapse_check [n_embd] [n_hc] [n_tokens]
//
// Exits non-zero if the two disagree by more than float noise.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    const int n_embd = argc > 1 ? std::atoi(argv[1]) : 2560;
    const int n_hc   = argc > 2 ? std::atoi(argv[2]) : 4;
    const int nt     = argc > 3 ? std::atoi(argv[3]) : 3;
    const int64_t hc_dim = (int64_t) n_embd * n_hc;

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "no GPU backend\n"); return 1; }

    ggml_init_params ip = { (size_t) 64 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc_dim, nt);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc_dim, nt);
    ggml_set_input(a);
    ggml_set_input(b);

    // The reference, transcribed from qwen4exp_hc_mix.
    ggml_tensor * gated = ggml_mul(ctx, a, b);
    gated = ggml_reshape_3d(ctx, gated, n_embd, n_hc, nt);
    const size_t stream_stride = ggml_row_size(gated->type, n_embd);
    ggml_tensor * ref = ggml_cont(
        ctx, ggml_view_2d(ctx, gated, n_embd, nt, stream_stride * n_hc, 0));
    for (int c = 1; c < n_hc; ++c) {
        ggml_tensor * s = ggml_cont(ctx, ggml_view_2d(ctx, gated, n_embd, nt,
                                                      stream_stride * n_hc,
                                                      stream_stride * (size_t) c));
        ref = ggml_add(ctx, ref, s);
    }
    ref = ggml_scale(ctx, ref, 1.0f / (float) n_hc);
    ggml_set_output(ref);
    ggml_build_forward_expand(gf, ref);

    ggml_tensor * fused = ggml_hc_collapse(ctx, a, b, n_embd, n_hc);
    ggml_set_output(fused);
    ggml_build_forward_expand(gf, fused);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        std::fprintf(stderr, "graph allocation failed\n");
        return 1;
    }

    // Deterministic, and spread over several orders of magnitude so a
    // summation-order difference has somewhere to show.
    std::vector<float> ha((size_t) hc_dim * nt), hb((size_t) hc_dim * nt);
    uint32_t s = 12345u;
    auto next = [&s]() { s = s * 1664525u + 1013904223u; return (float) ((s >> 8) & 0xffff) / 32768.0f - 1.0f; };
    for (size_t i = 0; i < ha.size(); ++i) { ha[i] = next() * 4.0f; hb[i] = next(); }
    ggml_backend_tensor_set(a, ha.data(), 0, sizeof(float) * ha.size());
    ggml_backend_tensor_set(b, hb.data(), 0, sizeof(float) * hb.size());

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "compute failed\n");
        return 1;
    }

    std::vector<float> r((size_t) n_embd * nt), f((size_t) n_embd * nt);
    ggml_backend_tensor_get(ref,   r.data(), 0, sizeof(float) * r.size());
    ggml_backend_tensor_get(fused, f.data(), 0, sizeof(float) * f.size());

    // And a host reference too, so a shared misreading of the layout by both
    // graph forms cannot pass unnoticed.
    double worst_gpu = 0.0, worst_host = 0.0;
    int worst_idx = -1;
    for (int t = 0; t < nt; ++t) {
        for (int i = 0; i < n_embd; ++i) {
            double acc = 0.0;
            for (int c = 0; c < n_hc; ++c) {
                const size_t o = (size_t) t * hc_dim + (size_t) c * n_embd + (size_t) i;
                acc += (double) ha[o] * (double) hb[o];
            }
            acc /= (double) n_hc;
            const size_t o = (size_t) t * n_embd + (size_t) i;
            const double d_gpu  = std::fabs((double) r[o] - (double) f[o]);
            const double d_host = std::fabs(acc - (double) f[o]);
            if (d_gpu > worst_gpu) { worst_gpu = d_gpu; worst_idx = (int) o; }
            if (d_host > worst_host) worst_host = d_host;
        }
    }

    std::printf("n_embd=%d n_hc=%d n_tokens=%d\n", n_embd, n_hc, nt);
    std::printf("  fused vs unfused graph: worst |diff| = %.3e (at %d)\n", worst_gpu, worst_idx);
    std::printf("  fused vs host double:   worst |diff| = %.3e\n", worst_host);
    std::printf("  sample ref[0]=%.6f fused[0]=%.6f\n", r[0], f[0]);

    const bool ok = worst_host < 1e-3;
    std::printf("=> %s\n", ok ? "the fused op computes the reference function"
                              : "MISMATCH: the fused op is not the reference function");

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return ok ? 0 : 3;
}
