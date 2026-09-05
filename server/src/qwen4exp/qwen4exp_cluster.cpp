#include "qwen4exp/qwen4exp_cluster.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace dflash::common {

// Mirrors the loader's granularity; part of the scheme the ranks must agree on.
static constexpr uint64_t kFfnShardGran = 32;


uint64_t qwen4exp_cluster_placement_hash(const Qwen4ExpClusterRuntime & rt) {
    if (!rt.cfg) {
        return 0;
    }
    // The scheme, not this rank's share of it. The handshake compares the
    // hashes across ranks and demands they agree, so what it is asking is
    // "are we all splitting the same way" -- and every axis here is a pure
    // function of the rank count: experts and the shared expert by hidden
    // width, in contiguous equal parts. Folding the rank in would make every
    // rank disagree with every other by construction.
    uint64_t h = 0x9e3779b97f4a7c15ull;
    h ^= (uint64_t) rt.cfg->size * 0x100000001b3ull;
    h = (h << 7) | (h >> 57);
    h ^= (uint64_t) kFfnShardGran * 0xff51afd7ed558ccdull;
    return h ? h : 1;
}

namespace {

// DFLASH_CLUSTER_ALLREDUCE_NOOP=1 skips the collective. The output is WRONG --
// every rank keeps its own partial -- and it exists only so a run can be timed
// with and without the reduction to price it.
bool allreduce_noop() {
    static const bool on = []() {
        const char * s = std::getenv("DFLASH_CLUSTER_ALLREDUCE_NOOP");
        return s && std::atoi(s) == 1;
    }();
    return on;
}

void allreduce_graph_callback(void * user, void * data, size_t n, void * stream) {
    auto * rt = static_cast<Qwen4ExpClusterRuntime *>(user);
    if (!rt || !rt->comm || rt->comm->size() <= 1 || n == 0 || !data) return;
    if (allreduce_noop()) return;
    std::string err;
    if (!rt->comm->allreduce_sum_f32(data, n, (cluster::DeviceStream) stream, &err)) {
        // Keep the first failure: the rest of the graph still runs, and the
        // caller turns this into a failed forward once the compute returns.
        if (rt->node_error.empty()) {
            rt->node_error = err.empty() ? "in-graph all-reduce failed" : err;
        }
        return;
    }
    rt->telemetry.allreduce_calls += 1;
    rt->telemetry.allreduce_bytes += (uint64_t) n * sizeof(float);
}

}  // namespace

ggml_tensor * qwen4exp_cluster_allreduce_node(ggml_context * ctx,
                                              ggml_tensor *  partial,
                                              Qwen4ExpClusterRuntime & rt) {
    if (!ctx || !partial) return partial;
    if (!rt.comm || rt.comm->size() <= 1) return partial;
    if (partial->type != GGML_TYPE_F32 || !ggml_is_contiguous(partial)) {
        std::fprintf(stderr,
                     "[qwen4exp-cluster] in-graph all-reduce needs a contiguous "
                     "F32 partial, got %s%s\n",
                     ggml_type_name(partial->type),
                     ggml_is_contiguous(partial) ? "" : " (non-contiguous)");
        return partial;
    }
    return ggml_cluster_allreduce(ctx, partial, &allreduce_graph_callback, &rt);
}

ShardRange qwen4exp_shard_range(int64_t extent, int64_t granularity,
                                int rank, int size) {
    ShardRange r;
    r.begin = 0;
    r.end   = extent;
    if (size <= 1 || granularity <= 0 || extent <= 0) {
        return r;
    }
    // Refuse anything that does not divide evenly on the granularity. An
    // uneven split is expressible, but every rank would then need to know the
    // other ranks' extents to place its slice, and a weight that cannot be cut
    // cleanly is better left replicated than cut wrongly.
    const int64_t units = extent / granularity;
    if (units * granularity != extent || units % size != 0) {
        return r;
    }
    const int64_t per = (units / size) * granularity;
    r.begin = (int64_t) rank * per;
    r.end   = r.begin + per;
    r.split = true;
    return r;
}

}  // namespace dflash::common
