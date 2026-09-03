// deepseek4_cluster.cpp - DeepSeek4 expert-parallel cluster runtime (path 3a).
//
// See deepseek4_cluster.h for the model. This file has no RCCL dependency:
// collectives go through IClusterComm, the stream comes from
// ggml_backend_cuda_get_stream (Agent A's ggml accessor).

#include "deepseek4_cluster.h"

#include "common/moe_hybrid_routing_stats.h"

#include "ggml-alloc.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace dflash::common {

namespace {

using ClusterClock = std::chrono::steady_clock;

uint64_t us_since(ClusterClock::time_point t0) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        ClusterClock::now() - t0).count();
}

// Slots cached from ClusterExpertPlacement::slot_validity. Covers decode
// (1) and every DSpark verify width (<= 5, q5 opt-in) plus headroom; larger
// prefill batches use placement.slot_owner() directly.
constexpr int kSlotValidityCacheSlots = 8;

void set_err(std::string * err, const std::string & msg) {
    if (err) *err = msg;
}

// Mixed ROCmFP qtypes need decode tables registered per resident tensor
// (register_deepseek4_moe_hybrid_mix_tables), which today requires a
// materialized GPU cold owner. Refuse instead of silently loading garbage.
// TODO(cluster-verify): extend register_deepseek4_moe_hybrid_mix_tables to
// hot-only storage so mixed-precision experts can run under cold owner None.
bool ds4_cluster_has_mix_experts(const DeepSeek4Weights & w, std::string * what) {
    for (const auto & L : w.layers) {
        const struct { ggml_tensor * t; int qtype; const char * name; } mix[] = {
            { L.ffn_gate_exps, GGML_TYPE_Q3_1_ROCMFP3_MIX, "qtype-105 (mixed ROCmFP3) gate" },
            { L.ffn_up_exps,   GGML_TYPE_Q3_1_ROCMFP3_MIX, "qtype-105 (mixed ROCmFP3) up"   },
            { L.ffn_down_exps, GGML_TYPE_Q3_1_ROCMFP3_MIX, "qtype-105 (mixed ROCmFP3) down" },
            { L.ffn_gate_exps, GGML_TYPE_Q2_1_ROCMFP2_MIX, "qtype-106 (mixed ROCmFP2) gate" },
            { L.ffn_up_exps,   GGML_TYPE_Q2_1_ROCMFP2_MIX, "qtype-106 (mixed ROCmFP2) up"   },
            { L.ffn_down_exps, GGML_TYPE_Q2_1_ROCMFP2_MIX, "qtype-106 (mixed ROCmFP2) down" },
        };
        for (const auto & m : mix) {
            if (m.t && m.t->type == m.qtype) {
                if (what) *what = m.name;
                return true;
            }
        }
    }
    return false;
}

const char * placement_source_label(cluster::PlacementSource s) {
    return cluster::placement_source_name(s);
}

// Enqueue the collective for n floats at dev_ptr on the backend stream.
// force_wait: block the host until the reduction completed (host-resident
// consumers need this); otherwise stay stream-ordered unless rt.trace.
bool allreduce_device(Ds4ClusterRuntime & rt,
                      ggml_backend_t backend,
                      void * dev_ptr,
                      size_t n,
                      int layer,
                      bool force_wait,
                      DeepSeek4StepTelemetry * telemetry,
                      std::string * err) {
    if (!rt.comm || !rt.cfg) {
        set_err(err, "cluster runtime has no communicator");
        return false;
    }
    if (rt.comm->size() <= 1 || n == 0) {
        return true;  // single rank: the partial already is the sum
    }
    if (!dev_ptr) {
        set_err(err, "cluster all-reduce: null device pointer");
        return false;
    }

    // TODO(cluster-verify): ggml_backend_cuda_get_stream must return the same
    // HIP stream ggml_backend_graph_compute enqueues on (cuda_ctx->stream()),
    // otherwise the collective is not ordered after the producing kernels.
    cluster::DeviceStream stream = ggml_backend_cuda_get_stream(backend);

    const uint64_t bytes_f32 = (uint64_t) n * sizeof(float);
    const cluster::AllreduceDType dtype = rt.cfg->allreduce_dtype;
    // M1 correctness first: `auto` resolves to F32 for every size. The bf16
    // compressed collective rounds each per-layer partial to 8 mantissa
    // bits; over 43 layers of a multi-token prefill (>256 KiB payload, which
    // `auto` would have sent as bf16) that visibly changes the first
    // generated token. bf16 stays available as an explicit
    // --cluster-allreduce-dtype bf16 for the WP5 prefill measurements.
    // TODO(cluster-verify): re-enable the auto threshold once a bf16 prefill
    // run is token-identical to f32 on the qualification prompts.
    const bool use_bf16 = dtype == cluster::AllreduceDType::BF16;
    if (dtype == cluster::AllreduceDType::Auto && bytes_f32 > cluster::kAutoBf16ThresholdBytes) {
        static bool logged_auto_f32 = false;
        if (!logged_auto_f32) {
            logged_auto_f32 = true;
            std::fprintf(stderr,
                         "[deepseek4-cluster] allreduce dtype auto: using f32 for %llu-byte "
                         "partials (bf16 auto-switch disabled until verified)\n",
                         (unsigned long long) bytes_f32);
        }
    }

    bool ok = false;
    uint64_t payload = 0;
    if (use_bf16) {
        if (!rt.ensure_scratch(backend, n, err)) return false;
        ok = rt.comm->allreduce_sum_bf16_compressed(dev_ptr, n, rt.scratch_bf16->data, stream, err);
        payload = (uint64_t) n * 2u;
    } else {
        ok = rt.comm->allreduce_sum_f32(dev_ptr, n, stream, err);
        payload = bytes_f32;
    }
    if (!ok) {
        if (err && err->empty()) *err = "cluster all-reduce enqueue failed";
        return false;
    }
    rt.telemetry.allreduce_calls += 1;
    rt.telemetry.allreduce_bytes += payload;
    if (telemetry) telemetry->cluster_allreduce_bytes += payload;

    if (force_wait || rt.trace) {
        const auto wait_t0 = ClusterClock::now();
        if (!rt.comm->wait_stream(stream, rt.cfg->timeout_ms, err)) {
            if (err && err->empty()) *err = "cluster all-reduce wait timed out";
            return false;
        }
        const uint64_t wait_us = us_since(wait_t0);
        rt.telemetry.allreduce_wait_us += wait_us;
        if (rt.trace) {
            std::fprintf(stderr,
                         "[deepseek4-cluster] rank %d layer %d allreduce n=%zu %s wait=%llu us\n",
                         rt.rank(), layer, n, use_bf16 ? "bf16" : "f32",
                         (unsigned long long) wait_us);
        }
    }
    return true;
}

}  // namespace

// ─── Ds4ClusterRuntime ─────────────────────────────────────────────────

Ds4ClusterRuntime::~Ds4ClusterRuntime() {
    free_scratch();
}

bool Ds4ClusterRuntime::evaluates(int layer, int expert, int slot) const {
    if (slot >= 0 && slot < slot_validity_n_slots && !slot_validity_cache.empty() &&
        layer >= 0 && layer < placement.n_layer && expert >= 0 && expert < placement.n_expert) {
        const size_t idx =
            ((size_t) layer * (size_t) slot_validity_n_slots + (size_t) slot) *
                (size_t) placement.n_expert + (size_t) expert;
        if (idx < slot_validity_cache.size()) return slot_validity_cache[idx] != 0;
    }
    return placement.slot_owner(layer, expert, slot) == rank();
}

bool Ds4ClusterRuntime::ensure_scratch(ggml_backend_t backend, size_t n_elems, std::string * err) {
    if (scratch_buf && scratch_backend == backend && scratch_elems >= n_elems) {
        return true;
    }
    free_scratch();
    ggml_init_params ip{};
    ip.mem_size = 4 * ggml_tensor_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    scratch_ctx = ggml_init(ip);
    if (!scratch_ctx) {
        set_err(err, "cluster scratch: ggml_init failed");
        return false;
    }
    scratch_f32 = ggml_new_tensor_1d(scratch_ctx, GGML_TYPE_F32, (int64_t) n_elems);
    ggml_set_name(scratch_f32, "ds4_cluster_scratch_f32");
    // bf16 payload for the compressed collective: 2 bytes per element.
    scratch_bf16 = ggml_new_tensor_1d(scratch_ctx, GGML_TYPE_I16, (int64_t) n_elems);
    ggml_set_name(scratch_bf16, "ds4_cluster_scratch_bf16");
    scratch_buf = ggml_backend_alloc_ctx_tensors(scratch_ctx, backend);
    if (!scratch_buf) {
        free_scratch();
        set_err(err, "cluster scratch: device allocation failed");
        return false;
    }
    scratch_backend = backend;
    scratch_elems = n_elems;
    return true;
}

void Ds4ClusterRuntime::free_scratch() {
    if (scratch_buf) { ggml_backend_buffer_free(scratch_buf); scratch_buf = nullptr; }
    if (scratch_ctx) { ggml_free(scratch_ctx); scratch_ctx = nullptr; }
    scratch_f32 = nullptr;
    scratch_bf16 = nullptr;
    scratch_elems = 0;
    scratch_backend = nullptr;
}

// ─── Config / placement ────────────────────────────────────────────────

MoeHybridConfig ds4_cluster_moe_config(const DeepSeek4Weights & w) {
    MoeHybridConfig cfg;
    cfg.n_embd = w.n_embd;
    cfg.n_expert = w.n_expert;
    cfg.n_expert_used = w.n_expert_used;
    cfg.n_ff_exp = w.n_ff_exp;
    cfg.n_ff_shexp = w.n_ff_exp;
    cfg.n_layer = w.n_layer;
    cfg.first_moe_layer = 0;
    cfg.swiglu_clamp = w.swiglu_clamp_exp;
    cfg.cold_expert_backend = MoeHybridColdBackend::None;
    cfg.materialize_hot_experts = true;
    cfg.materialize_cold_experts = false;
    // Same as the single-node DS4 hybrid config: reduced hot stacks stay on
    // the q<=4 MMVQ sub-batch path on gfx1151 (mmq_safe_full_batch=false).
    return cfg;
}

bool ds4_cluster_build_placement(const cluster::ClusterConfig & cfg,
                                 const DeepSeek4Weights & w,
                                 const std::string * hotness_csv_path,
                                 Ds4ClusterRuntime & rt,
                                 std::string * err) {
    if (!cfg.enabled() || cfg.rank < 0 || cfg.rank >= cfg.size) {
        set_err(err, "cluster config is not enabled or rank is out of range");
        return false;
    }
    if (cfg.shared_expert != cluster::SharedExpertMode::Replicate) {
        // TODO(cluster-verify): Shard (n_ff/N slice summed in the same
        // all-reduce) and Rank0 are M3 work; the per-layer path below adds
        // the locally computed shared expert after the reduction.
        set_err(err, std::string("shared expert mode '") +
                     cluster::shared_expert_mode_name(cfg.shared_expert) +
                     "' is not implemented yet; use replicate");
        return false;
    }

    const MoeHybridConfig mcfg = ds4_cluster_moe_config(w);
    MoeHybridRoutingStats stats;
    bool have_stats = false;
    const bool need_stats =
        cfg.placement_source == cluster::PlacementSource::Balanced ||
        (cfg.placement_source == cluster::PlacementSource::Uniform && cfg.replicate_hot > 0);
    if (need_stats) {
        if (!hotness_csv_path || hotness_csv_path->empty()) {
            set_err(err, std::string(placement_source_label(cfg.placement_source)) +
                         " placement with replicate_hot/balanced requires DFLASH_DS4_HOTNESS_CSV");
            return false;
        }
        if (!MoeHybridRoutingStats::load_csv(*hotness_csv_path, stats, err)) return false;
        if (!stats.matches(mcfg)) {
            set_err(err, "routing hotness CSV shape does not match the DeepSeek V4 target "
                         "(n_layer/n_expert/n_expert_used)");
            return false;
        }
        have_stats = true;
    }

    switch (cfg.placement_source) {
        case cluster::PlacementSource::Uniform:
            if (!cluster::ClusterExpertPlacement::build_uniform(
                    cfg.size, mcfg, cfg.replicate_hot, have_stats ? &stats : nullptr,
                    rt.placement, err)) {
                return false;
            }
            break;
        case cluster::PlacementSource::Balanced:
            if (!cluster::ClusterExpertPlacement::build_balanced(
                    cfg.size, stats, cfg.replicate_hot, rt.placement, err)) {
                return false;
            }
            break;
        case cluster::PlacementSource::File:
            if (!cluster::ClusterExpertPlacement::load_json(cfg.placement_file, rt.placement, err)) {
                return false;
            }
            if (rt.placement.n_ranks != cfg.size) {
                set_err(err, "placement file was built for " + std::to_string(rt.placement.n_ranks) +
                             " ranks, cluster size is " + std::to_string(cfg.size));
                return false;
            }
            if (!rt.placement.matches(mcfg)) {
                set_err(err, "placement file dimensions do not match the DeepSeek V4 target");
                return false;
            }
            break;
    }

    if (!rt.placement.to_rank_placement(cfg.rank, rt.rank_placement, err)) return false;
    rt.placement.slot_validity(cfg.rank, kSlotValidityCacheSlots, rt.slot_validity_cache);
    rt.slot_validity_n_slots = kSlotValidityCacheSlots;
    if (&cfg != &rt.cfg_storage) rt.cfg_storage = cfg;
    rt.cfg = &rt.cfg_storage;
    rt.trace = cluster::cluster_env_trace();

    std::fprintf(stderr, "[deepseek4-cluster] rank %d/%d %s source=%s resident_experts=%d/%d\n",
                 cfg.rank, cfg.size,
                 rt.placement.describe(have_stats ? &stats : nullptr).c_str(),
                 placement_source_label(cfg.placement_source),
                 rt.rank_placement.total_hot, w.n_layer * w.n_expert);
    return true;
}

bool ds4_cluster_init_experts(const std::string & model_path,
                              ggml_backend_t backend,
                              const DeepSeek4Weights & w,
                              Ds4ClusterRuntime & rt,
                              MoeHybridStorage & storage,
                              std::string * err) {
    std::string mix_what;
    if (ds4_cluster_has_mix_experts(w, &mix_what)) {
        set_err(err, mix_what + " experts need a materialized decode table per owner and "
                     "are not supported with cluster expert sharding yet");
        return false;
    }
    const MoeHybridConfig mcfg = ds4_cluster_moe_config(w);
    if (!rt.rank_placement.matches(mcfg)) {
        set_err(err, "rank placement does not match the loaded model dimensions");
        return false;
    }

    // The non-mmap builder materializes exactly the hot (resident) slices of
    // every expert tensor from the file and unmaps afterwards; with cold owner
    // None nothing else is read, so no streaming engine or retained mmap is
    // needed. (build_deepseek4_moe_hybrid_storage_from_file_with_mmap would
    // keep the whole file mapped for cold-expert streaming.)
    if (!build_deepseek4_moe_hybrid_storage_from_file(
            model_path, backend, w, rt.rank_placement, &mcfg, storage, err)) {
        return false;
    }
    if (storage.cold_backend_kind != MoeHybridColdBackend::None ||
        storage.materialized_cold_experts) {
        set_err(err, "cluster expert storage did not come back with cold owner None");
        return false;
    }

    uint64_t bytes = 0;
    int resident = 0;
    int layers_with_experts = 0;
    for (const MoeHybridLayerStorage & layer : storage.layers) {
        if (!layer.hot_buf) continue;
        bytes += (uint64_t) ggml_backend_buffer_get_size(layer.hot_buf);
        resident += (int) layer.hot_expert_ids.size();
        ++layers_with_experts;
        if (!layer.cold_expert_ids.empty() || layer.cold_buf) {
            set_err(err, "cluster expert storage unexpectedly holds cold experts");
            return false;
        }
    }
    rt.resident_expert_bytes = bytes;
    const int total_experts = w.n_layer * w.n_expert;
    std::fprintf(stderr,
                 "[deepseek4-cluster] rank %d/%d resident routed experts: %d/%d (%.1f%%) "
                 "%.2f GiB (%.1f MiB/layer over %d layers), owned=%d replicated=%d\n",
                 rt.rank(), rt.size(), resident, total_experts,
                 total_experts > 0 ? 100.0 * (double) resident / (double) total_experts : 0.0,
                 (double) bytes / (1024.0 * 1024.0 * 1024.0),
                 layers_with_experts > 0
                     ? (double) bytes / (1024.0 * 1024.0) / (double) layers_with_experts : 0.0,
                 layers_with_experts,
                 rt.placement.total_owned(rt.rank()),
                 resident - rt.placement.total_owned(rt.rank()));
    return true;
}

// ─── Diagnostics ────────────────────────────────────────────────────────

void ds4_cluster_checksum(const float * data, size_t n, double * sum, double * sum_abs) {
    double s = 0.0;
    double a = 0.0;
    if (data) {
        for (size_t i = 0; i < n; ++i) {
            s += (double) data[i];
            a += (double) (data[i] < 0.0f ? -data[i] : data[i]);
        }
    }
    if (sum) *sum = s;
    if (sum_abs) *sum_abs = a;
}

bool ds4_cluster_env_prefill_single_token() {
    static const bool enabled = [] {
        const char * v = std::getenv("DFLASH_CLUSTER_PREFILL_SINGLE_TOKEN");
        return v && v[0] && std::strcmp(v, "0") != 0;
    }();
    return enabled;
}

// ─── Route masking ──────────────────────────────────────────────────────

void ds4_cluster_mask_routes(const Ds4ClusterRuntime & rt,
                             int layer,
                             int32_t * selected,
                             float * weights,
                             int route_width,
                             int n_tokens) {
    if (!selected || !weights || route_width <= 0 || n_tokens <= 0) return;
    for (int t = 0; t < n_tokens; ++t) {
        int32_t * ids = selected + (size_t) t * (size_t) route_width;
        float * wts = weights + (size_t) t * (size_t) route_width;
        for (int j = 0; j < route_width; ++j) {
            const int32_t e = ids[j];
            if (e < 0) continue;
            if (!rt.evaluates(layer, (int) e, t)) {
                ids[j] = -1;
                wts[j] = 0.0f;
            }
        }
    }
}

// ─── All-reduce ─────────────────────────────────────────────────────────

bool ds4_cluster_allreduce_layer(Ds4ClusterRuntime & rt,
                                 ggml_backend_t backend,
                                 ggml_tensor * partial,
                                 int layer,
                                 DeepSeek4StepTelemetry * telemetry,
                                 std::string * err) {
    if (!partial) {
        set_err(err, "cluster all-reduce: null partial tensor");
        return false;
    }
    if (partial->type != GGML_TYPE_F32 || !ggml_is_contiguous(partial)) {
        set_err(err, "cluster all-reduce: partial must be contiguous F32");
        return false;
    }
    const auto t0 = ClusterClock::now();
    const size_t n = (size_t) ggml_nelements(partial);
    const bool ok = allreduce_device(rt, backend, partial->data, n, layer,
                                     /*force_wait=*/false, telemetry, err);
    if (telemetry) telemetry->cluster_allreduce_us += us_since(t0);
    return ok;
}

bool ds4_cluster_allreduce_layer_host(Ds4ClusterRuntime & rt,
                                      ggml_backend_t backend,
                                      float * partial_host,
                                      int n_embd,
                                      int n_tokens,
                                      int layer,
                                      DeepSeek4StepTelemetry * telemetry,
                                      std::string * err) {
    if (!rt.comm || rt.comm->size() <= 1) return true;
    if (!partial_host || n_embd <= 0 || n_tokens <= 0) {
        set_err(err, "cluster all-reduce: invalid host partial");
        return false;
    }
    const auto t0 = ClusterClock::now();
    const size_t n = (size_t) n_embd * (size_t) n_tokens;
    if (!rt.ensure_scratch(backend, n, err)) return false;
    // ggml_backend_tensor_set completes the H2D copy before returning
    // (ggml-cuda copies on cudaStreamPerThread and synchronizes it), so the
    // collective enqueued next on the backend stream reads final data.
    // TODO(cluster-verify): confirm on ROCm 7 / gfx1151 that the HIP build
    // keeps this synchronous set_tensor contract.
    ggml_backend_tensor_set(rt.scratch_f32, partial_host, 0, n * sizeof(float));
    if (!allreduce_device(rt, backend, rt.scratch_f32->data, n, layer,
                          /*force_wait=*/true, telemetry, err)) {
        return false;
    }
    ggml_backend_tensor_get(rt.scratch_f32, partial_host, 0, n * sizeof(float));
    if (telemetry) telemetry->cluster_allreduce_us += us_since(t0);
    return true;
}

}  // namespace dflash::common
