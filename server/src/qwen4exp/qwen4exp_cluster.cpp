#include "qwen4exp/qwen4exp_cluster.h"

namespace dflash::common {

uint64_t qwen4exp_cluster_placement_hash(const Qwen4ExpClusterRuntime & rt) {
    if (!rt.cfg) {
        return 0;
    }
    // The shard set is a pure function of (size, rank) for every axis this
    // runtime splits -- experts by index modulo size, heads by contiguous
    // range, the vocabulary by contiguous range -- so the identity of the
    // split is the pair. Mixing in a constant keeps the hash away from zero,
    // which the handshake reads as "no placement".
    uint64_t h = 0x9e3779b97f4a7c15ull;
    h ^= (uint64_t) rt.cfg->size * 0x100000001b3ull;
    h = (h << 7) | (h >> 57);
    h ^= (uint64_t) rt.cfg->rank * 0xff51afd7ed558ccdull;
    return h ? h : 1;
}

}  // namespace dflash::common
