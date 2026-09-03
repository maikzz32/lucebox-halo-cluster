// Unit tests for the backend feature/architecture gate.
//
// check_feature_compatibility(), collect_feature_warnings() and the
// model_capabilities.h table are pure functions over resolved facts, so this
// binary needs no model file, no GPU, and none of the backend stack — it
// compiles against feature_gate.cpp and placement_config.cpp alone. Keeping
// it separate from test_server_unit keeps that true: a gate rule stays
// testable in seconds rather than behind a full CUDA build.
//
// Build: cmake --build . --target test_feature_gate
// Run:   ./test_feature_gate

#include "CppUnitTestFramework.hpp"
#include "cluster/cluster_config.h"
#include "common/draft_block_size.h"
#include "common/feature_gate.h"
#include "common/model_capabilities.h"
#include "common/paged_attention_config.h"
#include "placement/placement_config.h"

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace CppUnitTestFramework;
using namespace dflash::common;

// ── Backend compatibility gate ──────────────────────────────────────────
// One case per rule cluster in check_feature_compatibility(). All resolved
// facts are parameters, so none of this needs a model file or GPU.

namespace {
struct FeatureGateFixture : CommonFixture {
    using CommonFixture::CommonFixture;

static BackendArgs gate_args_hip_deepseek4() {
    BackendArgs args;
    args.model_path = "/nonexistent/model.gguf";
    args.device.backend = PlacementBackend::Hip;
    args.device.gpu = 0;
    return args;
}

static std::string gate_result(
    const BackendArgs & args,
    const std::string & arch,
    PlacementBackend backend,
    const BackendFeatureConfig & features = {}) {
    return check_feature_compatibility(
        args, features, arch, backend, backend);
}

static std::string gate_result_for_binary(
    const BackendArgs & args,
    const std::string & arch,
    PlacementBackend target_backend,
    PlacementBackend compiled_backend,
    const BackendFeatureConfig & features = {}) {
    return check_feature_compatibility(
        args, features, arch, target_backend, compiled_backend);
}

void test_feature_gate_accepts_plain_launch() {
    BackendArgs args;
    args.model_path = "/nonexistent/model.gguf";
    CHECK(gate_result(
        args, "qwen35", PlacementBackend::Cuda).empty());
}

void test_feature_gate_rejects_undetected_arch() {
    BackendArgs args;
    args.model_path = "/nonexistent/model.gguf";
    CHECK(!gate_result(
        args, "", PlacementBackend::Cuda).empty());
}

void test_feature_gate_requires_compiled_target_backend() {
    BackendArgs args;
    args.model_path = "/nonexistent/model.gguf";
    args.device.backend = PlacementBackend::Hip;
    CHECK(!gate_result_for_binary(
        args, "qwen35", PlacementBackend::Hip,
        PlacementBackend::Cuda).empty());
}

void test_feature_gate_ipc_options_require_ipc_binary() {
    BackendArgs draft;
    draft.model_path = "/nonexistent/model.gguf";
    draft.remote_draft.work_dir = "/tmp/draft";
    CHECK(!gate_result(
        draft, "qwen35", PlacementBackend::Cuda).empty());

    BackendArgs target;
    target.model_path = "/nonexistent/model.gguf";
    target.remote_target_shard.work_dir = "/tmp/target";
    CHECK(!gate_result(
        target, "qwen35", PlacementBackend::Cuda).empty());
}

void test_feature_gate_mixed_draft_placement_requires_ipc() {
    BackendArgs args;
    args.model_path = "/nonexistent/model.gguf";
    args.draft_path = "/nonexistent/draft.gguf";
    args.device.backend = PlacementBackend::Cuda;
    args.draft_device.backend = PlacementBackend::Hip;

    CHECK(!gate_result(
        args, "qwen35", PlacementBackend::Cuda).empty());

    args.remote_draft.ipc_bin = "/usr/bin/draft-ipc";
    CHECK(gate_result(
        args, "qwen35", PlacementBackend::Cuda).empty());

    args.draft_device.backend = PlacementBackend::Cuda;
    CHECK(!gate_result(
        args, "qwen35", PlacementBackend::Cuda).empty());
}

void test_feature_gate_draft_block_size_requires_local_draft() {
    BackendArgs args;
    args.model_path = "/nonexistent/model.gguf";
    args.draft_block_size = 8;
    CHECK(!gate_result(args, "qwen35", PlacementBackend::Cuda).empty());

    args.draft_path = "/nonexistent/draft.gguf";
    CHECK(gate_result(args, "qwen35", PlacementBackend::Cuda).empty());

    args.device.backend = PlacementBackend::Cuda;
    args.draft_device.backend = PlacementBackend::Hip;
    args.remote_draft.ipc_bin = "/usr/bin/draft-ipc";
    CHECK(!gate_result(args, "qwen35", PlacementBackend::Cuda).empty());
}

void test_draft_block_size_override_respects_checkpoint_horizon() {
    CHECK(draft_block_size_override_supported(0, 8));
    CHECK(draft_block_size_override_supported(2, 8));
    CHECK(draft_block_size_override_supported(7, 8));
    CHECK(draft_block_size_override_supported(8, 8));
    // Greedy verify keeps output exact at any width, so widening is allowed
    // up to 2x the checkpoint horizon (measured: acceptance extrapolates to
    // 16 on Qwen3.8 DFlash2, step time cliffs past it).
    CHECK(draft_block_size_override_supported(12, 8));
    CHECK(draft_block_size_override_supported(16, 8));
    CHECK(!draft_block_size_override_supported(1, 8));
    CHECK(!draft_block_size_override_supported(17, 8));
    CHECK(!draft_block_size_override_supported(32, 8));
}

void test_feature_gate_pflash_requires_drafter_and_supported_arch() {
    BackendArgs args;
    args.model_path = "/nonexistent/model.gguf";

    BackendFeatureConfig features;
    features.pflash_enabled = true;
    CHECK(!gate_result(
        args, "qwen35", PlacementBackend::Cuda, features).empty());

    features.pflash_drafter_configured = true;
    CHECK(gate_result(
        args, "gemma4", PlacementBackend::Cuda, features).empty());

    args.device.backend = PlacementBackend::Cuda;
    args.draft_device.backend = PlacementBackend::Hip;
    args.remote_draft.ipc_bin = "/usr/bin/draft-ipc";
    CHECK(!gate_result(
        args, "gemma4", PlacementBackend::Cuda, features).empty());
    CHECK(gate_result(
        args, "qwen35", PlacementBackend::Cuda, features).empty());
}

void test_feature_gate_validates_target_split_topology() {
    BackendArgs weights;
    weights.model_path = "/nonexistent/model.gguf";
    weights.device.layer_split_weights = {1.0, 1.0};
    CHECK(!gate_result(
        weights, "qwen35", PlacementBackend::Cuda).empty());

    BackendArgs mixed;
    mixed.model_path = "/nonexistent/model.gguf";
    CHECK(parse_placement_device_list(
        "cuda:0,hip:0", mixed.device));
    CHECK(!gate_result(
        mixed, "qwen35", PlacementBackend::Cuda).empty());

    mixed.remote_target_shard.ipc_bin = "/usr/bin/target-shard";
    CHECK(gate_result(
        mixed, "qwen35", PlacementBackend::Cuda).empty());

    BackendArgs two_boundaries;
    two_boundaries.model_path = "/nonexistent/model.gguf";
    CHECK(parse_placement_device_list(
        "cuda:0,hip:0,cuda:1", two_boundaries.device));
    two_boundaries.remote_target_shard.ipc_bin =
        "/usr/bin/target-shard";
    CHECK(!gate_result(
        two_boundaries, "qwen35", PlacementBackend::Cuda).empty());
}

void test_feature_gate_tensor_parallel_requirements() {
    BackendArgs valid;
    valid.model_path = "/nonexistent/model.gguf";
    CHECK(parse_placement_device_list(
        "cuda:0,cuda:1", valid.device));
    valid.device.split_mode = TargetSplitMode::Tensor;
    CHECK(gate_result(
        valid, "qwen35", PlacementBackend::Cuda).empty());

    BackendArgs missing_devices;
    missing_devices.model_path = "/nonexistent/model.gguf";
    missing_devices.device.split_mode = TargetSplitMode::Tensor;
    CHECK(!gate_result(
        missing_devices, "qwen35", PlacementBackend::Cuda).empty());

    CHECK(!gate_result(
        valid, "laguna", PlacementBackend::Cuda).empty());

    BackendArgs hip;
    hip.model_path = "/nonexistent/model.gguf";
    CHECK(parse_placement_device_list("hip:0,hip:1", hip.device));
    hip.device.split_mode = TargetSplitMode::Tensor;
    CHECK(!gate_result(
        hip, "qwen35", PlacementBackend::Hip).empty());

    BackendArgs mixed = valid;
    CHECK(parse_placement_device_list(
        "cuda:0,hip:0", mixed.device));
    mixed.device.split_mode = TargetSplitMode::Tensor;
    CHECK(!gate_result(
        mixed, "qwen35", PlacementBackend::Cuda).empty());

    BackendArgs weighted = valid;
    weighted.device.layer_split_weights = {1.0, 1.0};
    CHECK(!gate_result(
        weighted, "qwen35", PlacementBackend::Cuda).empty());

    BackendArgs remote = valid;
    remote.remote_target_shard.ipc_bin = "/usr/bin/target-shard";
    CHECK(!gate_result(
        remote, "qwen35", PlacementBackend::Cuda).empty());

    BackendFeatureConfig pflash;
    pflash.pflash_enabled = true;
    pflash.pflash_drafter_configured = true;
    CHECK(!gate_result(
        valid, "qwen35", PlacementBackend::Cuda, pflash).empty());

    BackendArgs draft = valid;
    draft.draft_path = "/nonexistent/draft.gguf";
    CHECK(gate_result(
        draft, "qwen35", PlacementBackend::Cuda).empty());
}

void test_feature_gate_ds4_prefill_requires_deepseek4() {
    BackendArgs args = gate_args_hip_deepseek4();
    args.ds4_prefill_mode_set = true;
    args.ds4_prefill_mode = PrefillAttentionMode::Dense;

    CHECK(!gate_result(
        args, "qwen35", PlacementBackend::Hip).empty());
    CHECK(gate_result(
        args, "deepseek4", PlacementBackend::Hip).empty());
}

void test_feature_gate_approximate_ds4_prefill_requires_local_hip() {
    BackendArgs args = gate_args_hip_deepseek4();
    args.ds4_prefill_mode_set = true;
    args.ds4_prefill_mode = PrefillAttentionMode::Sparse;

    // CUDA has no approximate prefill path.
    CHECK(!gate_result(
        args, "deepseek4", PlacementBackend::Cuda).empty());

    // Neither does the layer-split adapter, even on HIP.
    BackendArgs split = args;
    CHECK(parse_placement_device_list("hip:0,hip:1", split.device));
    CHECK(!gate_result(
        split, "deepseek4", PlacementBackend::Hip).empty());

    // Nor a remote target shard.
    BackendArgs remote = args;
    remote.remote_target_shard.ipc_bin = "/usr/bin/shard";
    CHECK(!gate_result(
        remote, "deepseek4", PlacementBackend::Hip).empty());

    // Single local HIP device is the supported placement.
    CHECK(gate_result(
        args, "deepseek4", PlacementBackend::Hip).empty());

    // Exact prefill is unrestricted.
    BackendArgs exact = gate_args_hip_deepseek4();
    exact.ds4_prefill_mode_set = true;
    exact.ds4_prefill_mode = PrefillAttentionMode::Exact;
    CHECK(gate_result(
        exact, "deepseek4", PlacementBackend::Cuda).empty());
}

void test_feature_gate_ds4_decode_options_require_monolithic_hip() {
    BackendArgs fused = gate_args_hip_deepseek4();
    fused.ds4_fused_decode = true;
    CHECK(!gate_result(
        fused, "deepseek4", PlacementBackend::Cuda).empty());
    CHECK(gate_result(
        fused, "deepseek4", PlacementBackend::Hip).empty());

    BackendArgs f16_kv = gate_args_hip_deepseek4();
    f16_kv.ds4_fused_verify_f16_kv = true;
    CHECK(!gate_result(
        f16_kv, "deepseek4", PlacementBackend::Cuda).empty());
    CHECK(gate_result(
        f16_kv, "deepseek4", PlacementBackend::Hip).empty());

    BackendArgs split_f16_kv = f16_kv;
    split_f16_kv.device.layer_split_gpus = {0, 1};
    CHECK(!gate_result(
        split_f16_kv, "deepseek4", PlacementBackend::Hip).empty());

    BackendArgs topk = gate_args_hip_deepseek4();
    topk.ds4_expert_top_k = 4;
    CHECK(!gate_result(
        topk, "qwen35", PlacementBackend::Hip).empty());
    CHECK(gate_result(
        topk, "deepseek4", PlacementBackend::Hip).empty());

    // Top-k is a model policy in the monolithic backend and is independent of
    // the GPU vendor. Unlike fused decode, mixed CUDA-primary expert
    // placement can therefore use it.
    BackendArgs cuda_topk = topk;
    cuda_topk.device.backend = PlacementBackend::Cuda;
    CHECK(gate_result(
        cuda_topk, "deepseek4", PlacementBackend::Cuda).empty());

    BackendArgs split_topk = topk;
    split_topk.device.layer_split_gpus = {0, 1};
    CHECK(!gate_result(
        split_topk, "deepseek4", PlacementBackend::Hip).empty());
}

void test_feature_gate_remote_draft_requires_supported_arch() {
    BackendArgs args;
    args.model_path = "/nonexistent/model.gguf";
    args.draft_path = "/nonexistent/draft.gguf";
    args.device.backend = PlacementBackend::Cuda;
    args.draft_device.backend = PlacementBackend::Hip;
    args.remote_draft.ipc_bin = "/usr/bin/draft-ipc";

    CHECK(!gate_result(
        args, "gemma4", PlacementBackend::Cuda).empty());
    CHECK(gate_result(
        args, "qwen35", PlacementBackend::Cuda).empty());

    // Without a draft model or PFlash, remote draft IPC is unnecessary.
    BackendArgs no_draft = args;
    no_draft.draft_path = nullptr;
    CHECK(!gate_result(
        no_draft, "gemma4", PlacementBackend::Cuda).empty());
}

void test_feature_gate_layer_split_requires_supported_arch() {
    BackendArgs args;
    args.model_path = "/nonexistent/model.gguf";
    CHECK(parse_placement_device_list("cuda:0,cuda:1", args.device));

    // These four have a layer-split adapter.
    for (const char * arch : {"qwen35", "laguna", "gemma4", "deepseek4"}) {
        CHECK(gate_result(args, arch, PlacementBackend::Cuda).empty());
    }
    // These two do not: the factory would hand the split placement to a
    // monolithic backend, which reads only the primary GPU.
    for (const char * arch : {"qwen35moe", "qwen3"}) {
        CHECK(!gate_result(args, arch, PlacementBackend::Cuda).empty());
    }

    // Single-device placement is unaffected for the same architectures.
    BackendArgs single;
    single.model_path = "/nonexistent/model.gguf";
    CHECK(gate_result(single, "qwen35moe", PlacementBackend::Cuda).empty());
    CHECK(gate_result(single, "qwen3", PlacementBackend::Cuda).empty());
}

void test_feature_gate_paged_attention_requires_qwen35_monolithic() {
    BackendArgs args;
    args.model_path = "/nonexistent/model.gguf";
    args.paged_attention = true;
    CHECK(gate_result(args, "qwen35", PlacementBackend::Cuda).empty());
    CHECK(gate_result(args, "qwen35", PlacementBackend::Hip).empty());

    // Only qwen35 has a paged decode path. qwen35moe shares Qwen35Config, so
    // its rejection is this gate's job — the factory's field-presence
    // cross-check cannot tell the two apart.
    for (const char * arch : {"qwen35moe", "laguna", "qwen3",
                              "gemma4", "deepseek4"}) {
        CHECK(!gate_result(args, arch, PlacementBackend::Cuda).empty());
    }

    // Only the monolithic qwen35 backend owns a paged K/V pool. Both
    // placements are supported qwen35 launches without the flag, so the
    // rejection has to come from the paged rule.
    BackendArgs split = args;
    CHECK(parse_placement_device_list("cuda:0,cuda:1", split.device));
    CHECK(!gate_result(split, "qwen35", PlacementBackend::Cuda).empty());

    BackendArgs remote_shard = args;
    remote_shard.remote_target_shard.ipc_bin = "/usr/bin/target-shard";
    CHECK(!gate_result(
        remote_shard, "qwen35", PlacementBackend::Cuda).empty());

    for (BackendArgs * relaxed : {&split, &remote_shard}) {
        relaxed->paged_attention = false;
        CHECK(gate_result(
            *relaxed, "qwen35", PlacementBackend::Cuda).empty());
    }
}

void test_feature_gate_paged_attention_allows_fixed_local_chains() {
    BackendArgs base;
    base.model_path = "/nonexistent/model.gguf";
    base.paged_attention = true;

    BackendArgs draft = base;
    draft.draft_path = "/nonexistent/draft.gguf";
    CHECK(!gate_result(draft, "qwen35", PlacementBackend::Cuda).empty());

    BackendArgs concurrent_chain = draft;
    concurrent_chain.max_concurrency = 16;
    CHECK(gate_result(
        concurrent_chain, "qwen35", PlacementBackend::Cuda).empty());
    CHECK(gate_result(
        concurrent_chain, "qwen35", PlacementBackend::Hip).empty());

    BackendFeatureConfig request_scoped;
    request_scoped.draft_residency = DraftResidencyPolicy::RequestScoped;
    CHECK(!gate_result(
        concurrent_chain, "qwen35", PlacementBackend::Cuda,
        request_scoped).empty());

    BackendFeatureConfig persistent;
    persistent.draft_residency = DraftResidencyPolicy::Persistent;
    CHECK(gate_result(
        concurrent_chain, "qwen35", PlacementBackend::Cuda,
        persistent).empty());

    BackendArgs remote_chain = concurrent_chain;
    remote_chain.remote_draft.ipc_bin = "/usr/bin/draft-ipc";
    CHECK(!gate_result(
        remote_chain, "qwen35", PlacementBackend::Cuda).empty());

    BackendArgs ddtree = base;
    ddtree.ddtree_mode = true;
    CHECK(!gate_result(ddtree, "qwen35", PlacementBackend::Cuda).empty());

    BackendArgs windowed = base;
    windowed.fa_window = 4096;
    CHECK(!gate_result(
        windowed, "qwen35", PlacementBackend::Cuda).empty());

    BackendFeatureConfig pflash;
    pflash.pflash_enabled = true;
    pflash.pflash_drafter_configured = true;
    CHECK(!gate_result(
        base, "qwen35", PlacementBackend::Cuda, pflash).empty());

    BackendFeatureConfig kvflash;
    kvflash.kvflash_enabled = true;
    CHECK(!gate_result(
        base, "qwen35", PlacementBackend::Cuda, kvflash).empty());

    // The pool rounds max_ctx up to whole blocks, so both ends of the range
    // are rejected: nothing to allocate, and rounding that overflows int.
    BackendArgs empty_ctx = base;
    empty_ctx.device.max_ctx = 0;
    CHECK(!gate_result(
        empty_ctx, "qwen35", PlacementBackend::Cuda).empty());

    BackendArgs huge_ctx = base;
    huge_ctx.device.max_ctx = INT_MAX;
    CHECK(!gate_result(
        huge_ctx, "qwen35", PlacementBackend::Cuda).empty());

    BackendArgs max_ctx = base;
    max_ctx.device.max_ctx = INT_MAX - PAGED_BLOCK_SIZE + 1;
    CHECK(gate_result(
        max_ctx, "qwen35", PlacementBackend::Cuda).empty());

    // None of these are rules about paged attention itself: without the flag
    // every one of them is a supported qwen35 launch.
    for (BackendArgs * args : {&draft, &ddtree, &windowed, &empty_ctx,
                               &huge_ctx}) {
        args->paged_attention = false;
        CHECK(gate_result(*args, "qwen35", PlacementBackend::Cuda).empty());
    }
}

void test_feature_gate_parallel_and_kv_pool_rules() {
    // A valid paged qwen35 monolithic launch is the baseline every rule
    // below perturbs.
    BackendArgs paged;
    paged.model_path = "/nonexistent/model.gguf";
    paged.paged_attention = true;

    // --max-concurrency is validated even without any other flag: zero decode
    // slots is meaningless on every backend.
    BackendArgs plain;
    plain.model_path = "/nonexistent/model.gguf";
    plain.max_concurrency = 0;
    CHECK(!gate_result(plain, "qwen35", PlacementBackend::Cuda).empty());
    plain.max_concurrency = 1;
    CHECK(gate_result(plain, "qwen35", PlacementBackend::Cuda).empty());

    // More than one slot exists only in the paged qwen35 backend.
    BackendArgs dense;
    dense.model_path = "/nonexistent/model.gguf";
    dense.max_concurrency = 2;
    CHECK(!gate_result(dense, "qwen35", PlacementBackend::Cuda).empty());

    BackendArgs parallel = paged;
    parallel.max_concurrency = 2;
    CHECK(gate_result(parallel, "qwen35", PlacementBackend::Cuda).empty());

    // Slot counts need not be powers of two. Decode graph buckets pad via
    // active_slot_ids rather than changing the physical slot allocation.
    parallel.max_concurrency = 3;
    CHECK(gate_result(parallel, "qwen35", PlacementBackend::Cuda).empty());

    // 64 slots is the top of the supported range.
    parallel.max_concurrency = 64;
    CHECK(gate_result(parallel, "qwen35", PlacementBackend::Cuda).empty());
    parallel.max_concurrency = 65;
    CHECK(!gate_result(parallel, "qwen35", PlacementBackend::Cuda).empty());

    // --kv-pool-tokens sizes the shared pool, so it needs slots to share.
    BackendArgs pool = paged;
    pool.kv_pool_tokens = 4096;
    CHECK(!gate_result(pool, "qwen35", PlacementBackend::Cuda).empty());
    pool.max_concurrency = 2;
    CHECK(gate_result(pool, "qwen35", PlacementBackend::Cuda).empty());

    // The pool must hold at least one block, and stay addressable with int
    // after rounding up to whole blocks.
    pool.kv_pool_tokens = PAGED_BLOCK_SIZE - 1;
    CHECK(!gate_result(pool, "qwen35", PlacementBackend::Cuda).empty());
    pool.kv_pool_tokens = PAGED_BLOCK_SIZE;
    CHECK(gate_result(pool, "qwen35", PlacementBackend::Cuda).empty());
    const long long max_pool_tokens =
        ((long long)INT_MAX - PAGED_BLOCK_SIZE) /
        PAGED_BLOCK_SIZE * PAGED_BLOCK_SIZE;
    pool.kv_pool_tokens = max_pool_tokens + 1;
    CHECK(!gate_result(pool, "qwen35", PlacementBackend::Cuda).empty());
    pool.kv_pool_tokens = max_pool_tokens;
    CHECK(gate_result(pool, "qwen35", PlacementBackend::Cuda).empty());

    BackendArgs chain_pool = paged;
    chain_pool.max_concurrency = 16;
    chain_pool.draft_path = "/nonexistent/draft.gguf";
    const long long chain_scratch =
        (long long)chain_pool.max_concurrency * paged_token_capacity(16);
    const long long max_chain_pool_tokens =
        ((long long)INT_MAX - PAGED_BLOCK_SIZE - chain_scratch) /
        PAGED_BLOCK_SIZE * PAGED_BLOCK_SIZE;
    chain_pool.kv_pool_tokens = max_chain_pool_tokens;
    CHECK(gate_result(
        chain_pool, "qwen35", PlacementBackend::Hip).empty());
    chain_pool.kv_pool_tokens = max_chain_pool_tokens + PAGED_BLOCK_SIZE;
    CHECK(!gate_result(
        chain_pool, "qwen35", PlacementBackend::Hip).empty());

    // The automatic pool is memory-derived, so a logical slot/context product
    // larger than the physical tensor address space is legal.
    BackendArgs overflow = paged;
    overflow.max_concurrency = 2;
    overflow.device.max_ctx = 1 << 30;
    CHECK(gate_result(overflow, "qwen35", PlacementBackend::Cuda).empty());
    // An explicit addressable pool remains accepted as well.
    overflow.kv_pool_tokens = 1 << 20;
    CHECK(gate_result(overflow, "qwen35", PlacementBackend::Cuda).empty());
}

// ── Inert-flag warnings ─────────────────────────────────────────────────
// Warnings must never gate admission, so each case also asserts the same
// configuration passes check_feature_compatibility().

std::vector<std::string> warn_result(
    const BackendArgs & args,
    const std::string & arch,
    const BackendFeatureConfig & features = {}) {
    CHECK(check_feature_compatibility(
        args, features, arch, compiled_placement_backend(),
        compiled_placement_backend()).empty());
    return collect_feature_warnings(args, features, arch);
}

static bool warns_about(const std::vector<std::string> & warnings,
                        const std::string & flag) {
    for (const std::string & w : warnings) {
        if (w.rfind(flag + " ignored:", 0) == 0) return true;
    }
    return false;
}

void test_feature_warnings_silent_when_supported() {
    BackendArgs args;
    args.model_path = "/nonexistent/model.gguf";
    args.draft_path = "/nonexistent/draft.gguf";
    args.ddtree_mode = true;
    args.fa_window = 512;
    args.draft_block_size = 8;
    args.draft_swa_window = 2048;
    // qwen35 forwards every one of these.
    CHECK(warn_result(args, "qwen35").empty());
}

void test_feature_warnings_report_inert_draft() {
    BackendArgs args;
    args.model_path = "/nonexistent/model.gguf";
    args.draft_path = "/nonexistent/draft.gguf";

    // qwen3 and deepseek4 never forward a draft model.
    CHECK(warns_about(warn_result(args, "qwen3"), "--draft"));
    CHECK(warns_about(warn_result(args, "deepseek4"), "--draft"));
    // laguna and gemma4 forward it only when monolithic.
    CHECK(!warns_about(warn_result(args, "laguna"), "--draft"));
    CHECK(!warns_about(warn_result(args, "gemma4"), "--draft"));

    BackendArgs split = args;
    CHECK(parse_placement_device_list("cuda:0,cuda:1", split.device));
    const std::vector<std::string> w = collect_feature_warnings(split, {}, "laguna");
    CHECK(warns_about(w, "--draft"));
    CHECK(w[0].find("single-device placement") != std::string::npos);
}

void test_feature_warnings_report_inert_decode_tunables() {
    BackendArgs ddtree;
    ddtree.model_path = "/nonexistent/model.gguf";
    ddtree.ddtree_mode = true;
    CHECK(warns_about(warn_result(ddtree, "gemma4"), "--ddtree"));
    CHECK(!warns_about(warn_result(ddtree, "laguna"), "--ddtree"));

    BackendArgs vw;
    vw.model_path = "/nonexistent/model.gguf";
    vw.verify_width = 8;
    CHECK(!warns_about(warn_result(vw, "laguna"), "--verify-width"));
    CHECK(warns_about(warn_result(vw, "qwen35"), "--verify-width"));
    BackendArgs db;
    db.model_path = "/nonexistent/model.gguf";
    db.draft_path = "/nonexistent/draft.gguf";
    db.draft_block_size = 8;
    CHECK(!warns_about(warn_result(db, "qwen35"), "--draft-block-size"));
    CHECK(warns_about(warn_result(db, "qwen35moe"), "--draft-block-size"));
    CHECK(parse_placement_device_list("cuda:0,cuda:1", db.device));
    CHECK(warns_about(warn_result(db, "qwen35"), "--draft-block-size"));

    BackendArgs fa;
    fa.model_path = "/nonexistent/model.gguf";
    fa.fa_window = 4096;
    // gemma4 honors --fa-window on both paths; laguna has no such option.
    CHECK(!warns_about(warn_result(fa, "gemma4"), "--fa-window"));
    CHECK(warns_about(warn_result(fa, "laguna"), "--fa-window"));

    BackendArgs swa;
    swa.model_path = "/nonexistent/model.gguf";
    swa.draft_swa_window = 2048;
    CHECK(!warns_about(warn_result(swa, "qwen35moe"), "--draft-swa"));
    CHECK(warns_about(warn_result(swa, "gemma4"), "--draft-swa"));
}

void test_feature_warnings_report_inert_moe_options() {
    BackendArgs args;
    args.model_path = "/nonexistent/model.gguf";

    BackendFeatureConfig moe_opts;
    moe_opts.routing_stats_requested = true;
    moe_opts.adaptive_experts_requested = true;

    CHECK(warn_result(args, "laguna", moe_opts).empty());
    CHECK(warn_result(args, "qwen35moe", moe_opts).empty());
    CHECK(warn_result(args, "qwen35", moe_opts).size() == 2);
    CHECK(warn_result(args, "deepseek4", moe_opts).size() == 2);
}

void test_model_capability_tables() {
    // Table integrity: one row per architecture, no blanks, no duplicates.
    for (const ArchCapabilities & row : kArchCapabilities) {
        CHECK(row.arch != nullptr && row.arch[0] != '\0');
        CHECK(find_arch_capabilities(row.arch) == &row);
    }

    // arch_is_supported() must match create_backend()'s dispatch chain.
    for (const char * arch : {"qwen35", "qwen35moe", "laguna",
                              "qwen3", "gemma4", "deepseek4"}) {
        CHECK(arch_is_supported(arch));
    }
    CHECK(!arch_is_supported(""));
    CHECK(!arch_is_supported("qwen36"));  // model_card has a branch; the factory does not
    CHECK(!arch_is_supported("llama"));

    CHECK(arch_has_expert_offload("laguna"));
    CHECK(arch_has_expert_offload("qwen35moe"));
    CHECK(!arch_has_expert_offload("qwen35"));
    // deepseek4 is mixture-of-experts but has no hot/cold offload path.
    CHECK(!arch_has_expert_offload("deepseek4"));

    // Every capability predicate must be false for an architecture the
    // factory cannot build, so no rule can admit an unbuildable model.
    CHECK(!arch_supports_layer_split("qwen36"));
    CHECK(!arch_supports_remote_draft("qwen36"));
    CHECK(!arch_supports_pflash_compression("qwen36"));
    CHECK(!arch_supports_decode_draft("qwen36", false));
    CHECK(!arch_supports_ddtree("qwen36", false));
    CHECK(!arch_supports_verify_width("qwen36", false));
    CHECK(!arch_supports_draft_block_size("qwen36", false));
    CHECK(!arch_supports_fa_window("qwen36", false));
    CHECK(!arch_supports_draft_swa("qwen36", false));
    CHECK(!arch_supports_paged_attention("qwen36", false));

    // Paged decode lives in the monolithic qwen35 backend alone.
    CHECK(arch_supports_paged_attention("qwen35", false));
    CHECK(!arch_supports_paged_attention("qwen35", true));
    CHECK(!arch_supports_paged_attention("qwen35moe", false));
    CHECK(arch_supports_draft_block_size("qwen35", false));
    CHECK(!arch_supports_draft_block_size("qwen35", true));
    CHECK(!arch_supports_draft_block_size("qwen35moe", false));

    // Multi-node expert parallelism is a DeepSeek4-only path.
    CHECK(arch_supports_cluster_ep("deepseek4"));
    for (const char * arch : {"qwen35", "qwen35moe", "bailingmoe3", "laguna",
                              "qwen3", "gemma4", "qwen36", ""}) {
        CHECK(!arch_supports_cluster_ep(arch));
    }
}

// ── Cluster gate (--cluster-*) ──────────────────────────────────────────
// The cluster block in check_feature_compatibility() and the GPU-free
// helpers in cluster_config.cpp. None of this touches a socket or RCCL.

static BackendArgs cluster_args(int rank = 0, int size = 2) {
    BackendArgs args = gate_args_hip_deepseek4();
    args.cluster.rank = rank;
    args.cluster.size = size;
    args.cluster.head_host = "10.0.0.1";
    args.cluster.head_port = 9400;
    return args;
}

void test_feature_gate_cluster_accepts_deepseek4_hip() {
    CHECK(gate_result(cluster_args(), "deepseek4", PlacementBackend::Hip).empty());
    CHECK(gate_result(cluster_args(1, 2), "deepseek4", PlacementBackend::Hip).empty());
    CHECK(gate_result(cluster_args(3, 4), "deepseek4", PlacementBackend::Hip).empty());
    CHECK(gate_result(cluster_args(7, dflash::cluster::kClusterMaxSize),
                      "deepseek4", PlacementBackend::Hip).empty());

    // The model-default top-k (0 or 6) stays admissible; sparse prefill
    // and the other monolithic-HIP DS4 options are unaffected.
    BackendArgs topk6 = cluster_args();
    topk6.ds4_expert_top_k = 6;
    topk6.ds4_prefill_mode_set = true;
    topk6.ds4_prefill_mode = PrefillAttentionMode::Sparse;
    CHECK(gate_result(topk6, "deepseek4", PlacementBackend::Hip).empty());

    // A cluster config that is not enabled is invisible to the gate.
    BackendArgs off = gate_args_hip_deepseek4();
    off.cluster.rank = 3;  // stale rank without a size must not matter here
    CHECK(gate_result(off, "deepseek4", PlacementBackend::Hip).empty());
    CHECK(gate_result(off, "qwen35", PlacementBackend::Hip).empty());
}

void test_feature_gate_cluster_requires_deepseek4_and_hip() {
    // Wrong architecture.
    for (const char * arch : {"qwen35", "qwen35moe", "laguna", "qwen3", "gemma4"}) {
        const std::string e = gate_result(cluster_args(), arch, PlacementBackend::Hip);
        CHECK(!e.empty());
        CHECK(e.find("cluster") != std::string::npos);
    }

    // CUDA binary / CUDA target.
    BackendArgs cuda = cluster_args();
    cuda.device.backend = PlacementBackend::Cuda;
    CHECK(!gate_result_for_binary(cuda, "deepseek4", PlacementBackend::Cuda,
                                  PlacementBackend::Cuda).empty());
}

void test_feature_gate_cluster_rejects_split_paged_and_concurrency() {
    BackendArgs split = cluster_args();
    split.device.layer_split_gpus = {0, 1};
    CHECK(!gate_result(split, "deepseek4", PlacementBackend::Hip).empty());

    BackendArgs tensor = cluster_args();
    tensor.device.layer_split_gpus = {0, 1};
    tensor.device.split_mode = TargetSplitMode::Tensor;
    CHECK(!gate_result(tensor, "deepseek4", PlacementBackend::Hip).empty());

    BackendArgs remote = cluster_args();
    remote.remote_target_shard.ipc_bin = "/usr/bin/shard-ipc";
    CHECK(!gate_result(remote, "deepseek4", PlacementBackend::Hip).empty());

    BackendArgs paged = cluster_args();
    paged.paged_attention = true;
    CHECK(!gate_result(paged, "deepseek4", PlacementBackend::Hip).empty());

    BackendArgs concurrent = cluster_args();
    concurrent.max_concurrency = 2;
    CHECK(!gate_result(concurrent, "deepseek4", PlacementBackend::Hip).empty());

    BackendFeatureConfig pflash;
    pflash.pflash_enabled = true;
    pflash.pflash_drafter_configured = true;
    CHECK(!gate_result(cluster_args(), "deepseek4", PlacementBackend::Hip, pflash).empty());

    BackendFeatureConfig kvflash;
    kvflash.kvflash_enabled = true;
    CHECK(!gate_result(cluster_args(), "deepseek4", PlacementBackend::Hip, kvflash).empty());
}

void test_feature_gate_cluster_rank_size_and_head_rules() {
    // Rank outside [0, size).
    CHECK(!gate_result(cluster_args(2, 2), "deepseek4", PlacementBackend::Hip).empty());
    CHECK(!gate_result(cluster_args(-1, 2), "deepseek4", PlacementBackend::Hip).empty());

    // Size 1 is not a cluster; size above the maximum is refused.
    CHECK(!gate_result(cluster_args(0, 1), "deepseek4", PlacementBackend::Hip).empty());
    CHECK(!gate_result(cluster_args(0, dflash::cluster::kClusterMaxSize + 1),
                       "deepseek4", PlacementBackend::Hip).empty());

    // Missing head endpoint.
    BackendArgs no_head = cluster_args();
    no_head.cluster.head_host.clear();
    CHECK(!gate_result(no_head, "deepseek4", PlacementBackend::Hip).empty());

    // Structural validate() failures surface through the gate too.
    BackendArgs bad_file = cluster_args();
    bad_file.cluster.placement_source = dflash::cluster::PlacementSource::File;
    CHECK(!gate_result(bad_file, "deepseek4", PlacementBackend::Hip).empty());
}

void test_feature_gate_cluster_rejects_unsupported_ds4_options() {
    BackendArgs fused = cluster_args();
    fused.ds4_fused_decode = true;
    const std::string fused_err = gate_result(fused, "deepseek4", PlacementBackend::Hip);
    CHECK(!fused_err.empty());
    CHECK(fused_err.find("not yet supported on the cluster path") != std::string::npos);

    BackendArgs f16_kv = cluster_args();
    f16_kv.ds4_fused_verify_f16_kv = true;
    CHECK(!gate_result(f16_kv, "deepseek4", PlacementBackend::Hip).empty());

    BackendArgs topk4 = cluster_args();
    topk4.ds4_expert_top_k = 4;
    CHECK(!gate_result(topk4, "deepseek4", PlacementBackend::Hip).empty());
    // ... while the same flag stays legal outside cluster mode.
    BackendArgs topk4_single = gate_args_hip_deepseek4();
    topk4_single.ds4_expert_top_k = 4;
    CHECK(gate_result(topk4_single, "deepseek4", PlacementBackend::Hip).empty());
}

void test_feature_warnings_cluster_replicate_hot_without_hotness() {
    // collect_feature_warnings() is called directly (not via warn_result):
    // the cluster gate only admits HIP binaries, and the warning logic must
    // be checkable on the CUDA CI leg as well.
    BackendArgs args = cluster_args();
    args.cluster.replicate_hot = 2;
    const std::vector<std::string> w = collect_feature_warnings(args, {}, "deepseek4");
    // Whether the warning fires depends on DFLASH_DS4_HOTNESS_CSV in the test
    // environment; with the variable unset it must fire, and it must never
    // fire for a placement file (which carries its own replication).
    if (std::getenv("DFLASH_DS4_HOTNESS_CSV") == nullptr) {
        CHECK(warns_about(w, "--cluster-replicate-hot"));
    }
    BackendArgs from_file = args;
    from_file.cluster.placement_source = dflash::cluster::PlacementSource::File;
    from_file.cluster.placement_file = "/nonexistent/placement.json";
    CHECK(!warns_about(collect_feature_warnings(from_file, {}, "deepseek4"),
                       "--cluster-replicate-hot"));
    // No warning without the flag.
    CHECK(!warns_about(collect_feature_warnings(cluster_args(), {}, "deepseek4"),
                       "--cluster-replicate-hot"));
}

void test_cluster_config_validate() {
    using namespace dflash::cluster;
    ClusterConfig off;
    CHECK(!off.enabled());
    CHECK(!off.is_head());
    CHECK(!off.is_worker());
    CHECK(off.validate().empty());  // disabled configs are always valid

    ClusterConfig ok;
    ok.rank = 0;
    ok.size = 2;
    ok.head_host = "10.0.0.1";
    CHECK(ok.enabled());
    CHECK(ok.is_head());
    CHECK(!ok.is_worker());
    CHECK(ok.validate().empty());

    ClusterConfig worker = ok;
    worker.rank = 1;
    CHECK(worker.is_worker());
    CHECK(worker.validate().empty());

    ClusterConfig bad = ok;
    bad.size = 1;
    CHECK(!bad.validate().empty());
    bad = ok;
    bad.size = kClusterMaxSize + 1;
    CHECK(!bad.validate().empty());
    bad = ok;
    bad.rank = 2;
    CHECK(!bad.validate().empty());
    bad = ok;
    bad.rank = -1;
    CHECK(!bad.validate().empty());
    bad = ok;
    bad.head_host.clear();
    CHECK(!bad.validate().empty());
    bad = ok;
    bad.head_port = 0;
    CHECK(!bad.validate().empty());
    bad = ok;
    bad.head_port = 70000;
    CHECK(!bad.validate().empty());
    bad = ok;
    bad.replicate_hot = -1;
    CHECK(!bad.validate().empty());
    bad = ok;
    bad.placement_source = PlacementSource::File;
    CHECK(!bad.validate().empty());
    bad.placement_file = "/tmp/placement.json";
    CHECK(bad.validate().empty());
    bad = ok;
    bad.timeout_ms = 0;
    CHECK(!bad.validate().empty());
    bad = ok;
    bad.verify_hash_every = -3;
    CHECK(!bad.validate().empty());
}

void test_cluster_config_parsers() {
    using namespace dflash::cluster;
    std::string host;
    int port = 0;
    std::string err;

    CHECK(parse_host_port("10.0.0.1:9400", host, port, &err));
    CHECK(host == "10.0.0.1");
    CHECK(port == 9400);
    CHECK(parse_host_port("halo1", host, port, &err));
    CHECK(host == "halo1");
    CHECK(port == kClusterDefaultPort);
    CHECK(parse_host_port("[fe80::1]:7000", host, port, &err));
    CHECK(host == "fe80::1");
    CHECK(port == 7000);
    CHECK(parse_host_port("[fe80::1]", host, port, &err));
    CHECK(host == "fe80::1");
    CHECK(port == kClusterDefaultPort);
    CHECK(!parse_host_port("", host, port, &err));
    CHECK(!err.empty());
    CHECK(!parse_host_port(":9400", host, port, &err));
    CHECK(!parse_host_port("halo1:", host, port, &err));
    CHECK(!parse_host_port("halo1:abc", host, port, &err));
    CHECK(!parse_host_port("halo1:0", host, port, &err));
    CHECK(!parse_host_port("halo1:65536", host, port, &err));
    CHECK(!parse_host_port("[fe80::1", host, port, &err));

    SharedExpertMode se = SharedExpertMode::Shard;
    CHECK(parse_shared_expert_mode("replicate", se, &err));
    CHECK(se == SharedExpertMode::Replicate);
    CHECK(parse_shared_expert_mode("shard", se, &err));
    CHECK(se == SharedExpertMode::Shard);
    CHECK(parse_shared_expert_mode("rank0", se, &err));
    CHECK(se == SharedExpertMode::Rank0);
    CHECK(!parse_shared_expert_mode("Replicate", se, &err));
    CHECK(!parse_shared_expert_mode("", se, &err));
    CHECK(std::string(shared_expert_mode_name(SharedExpertMode::Rank0)) == "rank0");

    AllreduceDType dt = AllreduceDType::F32;
    CHECK(parse_allreduce_dtype("f32", dt, &err));
    CHECK(dt == AllreduceDType::F32);
    CHECK(parse_allreduce_dtype("bf16", dt, &err));
    CHECK(dt == AllreduceDType::BF16);
    CHECK(parse_allreduce_dtype("auto", dt, &err));
    CHECK(dt == AllreduceDType::Auto);
    CHECK(!parse_allreduce_dtype("fp16", dt, &err));
    CHECK(std::string(allreduce_dtype_name(AllreduceDType::BF16)) == "bf16");

    PlacementSource ps = PlacementSource::Balanced;
    std::string file = "stale";
    CHECK(parse_placement_source("uniform", ps, file, &err));
    CHECK(ps == PlacementSource::Uniform);
    CHECK(file.empty());
    CHECK(parse_placement_source("balanced", ps, file, &err));
    CHECK(ps == PlacementSource::Balanced);
    CHECK(parse_placement_source("/etc/lucebox/placement.json", ps, file, &err));
    CHECK(ps == PlacementSource::File);
    CHECK(file == "/etc/lucebox/placement.json");
    CHECK(!parse_placement_source("random", ps, file, &err));
    CHECK(!parse_placement_source("placement.yaml", ps, file, &err));
    CHECK(!parse_placement_source(".json", ps, file, &err));
    CHECK(std::string(placement_source_name(PlacementSource::File)) == "file");
}

};
}  // namespace

TEST_CASE(FeatureGateFixture, feature_gate_suite) {
    test_feature_gate_accepts_plain_launch();
    test_feature_gate_rejects_undetected_arch();
    test_feature_gate_requires_compiled_target_backend();
    test_feature_gate_ipc_options_require_ipc_binary();
    test_feature_gate_mixed_draft_placement_requires_ipc();
    test_feature_gate_draft_block_size_requires_local_draft();
    test_draft_block_size_override_respects_checkpoint_horizon();
    test_feature_gate_pflash_requires_drafter_and_supported_arch();
    test_feature_gate_validates_target_split_topology();
    test_feature_gate_tensor_parallel_requirements();
    test_feature_gate_ds4_prefill_requires_deepseek4();
    test_feature_gate_approximate_ds4_prefill_requires_local_hip();
    test_feature_gate_ds4_decode_options_require_monolithic_hip();
    test_feature_gate_remote_draft_requires_supported_arch();
    test_feature_gate_layer_split_requires_supported_arch();
    test_feature_gate_paged_attention_requires_qwen35_monolithic();
    test_feature_gate_paged_attention_allows_fixed_local_chains();
    test_feature_gate_parallel_and_kv_pool_rules();
    test_feature_warnings_silent_when_supported();
    test_feature_warnings_report_inert_draft();
    test_feature_warnings_report_inert_decode_tunables();
    test_feature_warnings_report_inert_moe_options();
    test_model_capability_tables();
    test_feature_gate_cluster_accepts_deepseek4_hip();
    test_feature_gate_cluster_requires_deepseek4_and_hip();
    test_feature_gate_cluster_rejects_split_paged_and_concurrency();
    test_feature_gate_cluster_rank_size_and_head_rules();
    test_feature_gate_cluster_rejects_unsupported_ds4_options();
    test_feature_warnings_cluster_replicate_hot_without_hotness();
    test_cluster_config_validate();
    test_cluster_config_parsers();
}
