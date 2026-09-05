#include "qwen4exp/qwen4exp_probe.h"

#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace dflash::common {

namespace {

struct Probe {
    std::string   label;
    ggml_tensor * sumsq;
    // The mean as well as the magnitude. A hyper-connection write-back passes
    // through 2*sigmoid(inject/n_hc), which saturates once |inject| is a few
    // times n_hc: at that point the gate is decided by the SIGN alone, and an
    // RMS cannot tell a stream that is written twice from one that is not
    // written at all.
    ggml_tensor * sum;
    int64_t       n;
};

std::vector<Probe> & probes() {
    static std::vector<Probe> v;
    return v;
}

}  // namespace

bool qwen4exp_probe_enabled() {
    static const bool on = []() {
        const char * s = std::getenv("DFLASH_QWEN4EXP_RMS");
        return s && std::atoi(s) == 1;
    }();
    return on;
}

void qwen4exp_probe_add(ggml_context * ctx, ggml_cgraph * gf,
                        const char * label, int il, ggml_tensor * t) {
    if (!qwen4exp_probe_enabled() || !t) {
        return;
    }
    // A caller may hand us a strided view -- one stream out of the injection,
    // say. ggml_sqr wants contiguous input, so make it so here rather than at
    // every probe site.
    if (!ggml_is_contiguous(t)) {
        t = ggml_cont(ctx, t);
    }
    ggml_tensor * ss = ggml_sum(ctx, ggml_sqr(ctx, t));
    ggml_set_output(ss);
    ggml_build_forward_expand(gf, ss);

    ggml_tensor * sm = ggml_sum(ctx, t);
    ggml_set_output(sm);
    ggml_build_forward_expand(gf, sm);

    char name[64];
    if (il >= 0) {
        std::snprintf(name, sizeof(name), "%s[%d]", label, il);
    } else {
        std::snprintf(name, sizeof(name), "%s", label);
    }
    probes().push_back({ name, ss, sm, ggml_nelements(t) });
}

void qwen4exp_probe_report() {
    if (!qwen4exp_probe_enabled() || probes().empty()) {
        return;
    }
    std::fprintf(stderr, "[q4e-rms] %-22s %12s %12s %10s\n",
                 "probe", "rms", "mean", "n");
    for (const Probe & p : probes()) {
        float acc = 0.0f;
        float tot = 0.0f;
        ggml_backend_tensor_get(p.sumsq, &acc, 0, sizeof(float));
        ggml_backend_tensor_get(p.sum,   &tot, 0, sizeof(float));
        const double rms  = p.n > 0 ? std::sqrt((double) acc / (double) p.n) : 0.0;
        const double mean = p.n > 0 ? (double) tot / (double) p.n : 0.0;
        std::fprintf(stderr, "[q4e-rms] %-22s %12.5f %12.5f %10lld\n",
                     p.label.c_str(), rms, mean, (long long) p.n);
    }
    probes().clear();
}

}  // namespace dflash::common
