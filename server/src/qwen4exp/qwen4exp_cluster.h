// qwen4exp_cluster.h - the rank-local view of a sharded qwen4exp model.
//
// DeepSeek's cluster splits routed experts and nothing else, which works
// because its experts dominate the bytes a decode step reads. qwen4exp is the
// opposite shape: with ten of 512 experts active it reads only 1.17 GB of
// experts against 4.27 GB in total, so expert parallelism alone caps out
// around 27 tok/s on two nodes against a 32 tok/s target. The measured budget
// per decoded token, from the official Q3_K_M conversion:
//
//   routed experts   1167 MB   27.4%    expert-parallel
//   attn_qkv          649 MB   15.2%    delta-net, split by value head
//   output (lm_head)  521 MB   12.2%    vocabulary split
//   ssm_out           423 MB    9.9%    delta-net, row-parallel
//   attn_gate         243 MB    5.7%    delta-net, split by value head
//   attn_q            236 MB    5.5%    attention, split by query head
//   everything else   1027 MB  24.1%
//
// So this runtime shards along three axes at once, and the reduction that
// makes the ranks agree happens twice per layer: once after the attention or
// delta-net output projection, once after the FFN. At 2560 floats that
// collective is pure latency -- measured p50 41.8 us on two nodes and 80.4 us
// on four (48.1 us in bf16) -- which is 4.0 ms per token across 96 of them,
// against a step budget of 27 ms for the two-node target.
//
// Include convention: #include "qwen4exp/qwen4exp_cluster.h"

#pragma once

#include "cluster/cluster_comm.h"
#include "cluster/cluster_config.h"
#include "cluster/cluster_decision_hooks.h"
#include "cluster/cluster_telemetry.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>

namespace dflash::common {

struct Qwen4ExpClusterRuntime {
    // cfg points at cfg_storage so the caller's ClusterConfig need not outlive
    // the backend; comm is borrowed and attached after the model is loaded.
    cluster::ClusterConfig         cfg_storage;
    const cluster::ClusterConfig * cfg  = nullptr;
    cluster::IClusterComm *        comm = nullptr;
    cluster::Ds4ClusterHooks *     hooks = nullptr;
    cluster::ClusterStepTelemetry  telemetry;
    bool trace = false;

    // A graph node cannot return an error, so an in-graph collective records
    // its first failure here and the forward fails after the compute.
    std::string node_error;

    int rank() const { return cfg ? cfg->rank : 0; }
    int size() const { return cfg ? cfg->size : 1; }
    bool sharded() const { return cfg && cfg->enabled() && cfg->size > 1; }
};

// A digest of what this rank holds, exchanged during the handshake so a
// mismatched shard set fails at startup rather than as wrong output.
uint64_t qwen4exp_cluster_placement_hash(const Qwen4ExpClusterRuntime & rt);

}  // namespace dflash::common
