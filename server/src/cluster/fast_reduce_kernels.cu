// fast_reduce_kernels.cu - the device half of the lean decode-path reduction.
//
// Three tiny kernels per reduction, all on the backend's own stream so they sit
// in the graph exactly where the collective used to:
//
//   set_flag    raise this rank's publish flag once the payload has been
//               copied out. The copy itself is a DMA, enqueued ahead of this
//               on the same stream, so stream order is what publishes it.
//   wait_flags  spin until every peer's flag has arrived.
//   add_peers   add the peers' partials, by then copied into device memory.
//
// WHY THE PAYLOAD IS NOT MOVED BY A KERNEL. It was, and it cost 33 ms per
// reduction against RCCL's 117 us -- 250 times worse than the thing it was
// meant to replace. Pinned host memory is mapped uncached for the GPU, so a
// kernel reading or writing bulk data there crawls; the copy engine does not
// care, because it is not reading through the GPU's caches. So the payload goes
// by hipMemcpyAsync and only the flags -- four bytes, three times per
// reduction -- are touched by a kernel at zero-copy speed.
//
// WHY THE FLAGS ARE PINNED AND NOT MANAGED. gfx1151 reports XNACK disabled, so
// the GPU cannot take a page fault, and a managed page the CPU has touched
// faults the moment a kernel reads it. Pinned pages never migrate, so they need
// no fault to be reached -- and the round trip through one measures 0.79 us
// (server/test/cluster_flag_latency.cu), which is what makes this worth doing.
//
// Every spin is bounded and counted. A rank that dies must surface as a visible
// failure, not as a GPU that never returns.

#include <hip/hip_runtime.h>

#include <cstdint>

namespace dflash::cluster {

namespace {
constexpr int kBlock = 256;

// `volatile` is not enough on gfx11. It keeps the compiler from caching the
// value in a register, but the load it emits can still be answered from the
// GPU's own L2, and nothing invalidates that line when the host writes through
// its own caches. Acquire at system scope is what forces the invalidate; the
// symptom of getting it wrong is not a fault but a kernel that spins out its
// timeout on a flag the host set microseconds ago.
__device__ __forceinline__ uint32_t sys_load(const uint32_t * p) {
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void sys_store(uint32_t * p, uint32_t v) {
    __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
}  // namespace

__global__ void fast_reduce_set_flag_kernel(uint32_t * pub_flag,
                                            uint32_t * slot_done,
                                            uint32_t * progress,
                                            uint32_t seq,
                                            uint32_t slot_span,
                                            uint64_t spin_limit) {
    if (threadIdx.x || blockIdx.x) return;
    // Do not overwrite a slot the progress thread has not finished with. The
    // GPU can be a step's worth of reductions ahead, so the ring is defended
    // rather than assumed wide enough.
    if (seq > slot_span) {
        uint64_t spins = 0;
        while (sys_load(slot_done) + slot_span < seq) {
            if (++spins > spin_limit) break;
        }
    }
    sys_store(progress, seq * 4u + 1u);
    sys_store(pub_flag, seq);
}

__global__ void fast_reduce_wait_flags_kernel(uint32_t * const * __restrict__ peer_flags,
                                              uint32_t * timed_out,
                                              uint32_t * progress,
                                              uint32_t seq,
                                              int n_peers,
                                              uint64_t spin_limit) {
    if (threadIdx.x || blockIdx.x) return;
    sys_store(progress, seq * 4u + 2u);
    for (int p = 0; p < n_peers; ++p) {
        uint64_t spins = 0;
        while (sys_load(peer_flags[p]) != seq) {
            if (++spins > spin_limit) {
                sys_store(timed_out, seq);
                return;
            }
        }
    }
}

// The peers' partials are in device memory by now -- the copy engine moved them
// there after the wait -- so this is an ordinary cached add.
__global__ void fast_reduce_add_kernel(float * __restrict__ dst,
                                       const float * __restrict__ scratch,
                                       int n_peers,
                                       int n,
                                       int stride,
                                       uint32_t * slot_done,
                                       uint32_t * progress,
                                       uint32_t seq) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int k = i; k < n; k += blockDim.x * gridDim.x) {
        float acc = dst[k];
        for (int p = 0; p < n_peers; ++p) {
            acc += scratch[(size_t) p * (size_t) stride + (size_t) k];
        }
        dst[k] = acc;
    }
    // The slot is free once its data has been consumed, not once it arrived.
    // Saying so from one thread is safe only because the whole grid's writes
    // are complete at the kernel boundary -- which is why the flag is raised
    // here and not by a fence inside the loop.
    if (i == 0) {
        sys_store(slot_done, seq);
        sys_store(progress, seq * 4u + 3u);
    }
}

void fast_reduce_launch_set_flag(uint32_t * pub_flag, uint32_t * slot_done,
                                 uint32_t * progress, uint32_t seq,
                                 uint32_t slot_span, uint64_t spin_limit,
                                 hipStream_t stream) {
    hipLaunchKernelGGL(fast_reduce_set_flag_kernel, dim3(1), dim3(1), 0, stream,
                       pub_flag, slot_done, progress, seq, slot_span, spin_limit);
}

void fast_reduce_launch_wait_flags(uint32_t * const * peer_flags,
                                   uint32_t * timed_out, uint32_t * progress,
                                   uint32_t seq, int n_peers,
                                   uint64_t spin_limit, hipStream_t stream) {
    hipLaunchKernelGGL(fast_reduce_wait_flags_kernel, dim3(1), dim3(1), 0, stream,
                       peer_flags, timed_out, progress, seq, n_peers, spin_limit);
}

void fast_reduce_launch_add(float * dst, const float * scratch, int n_peers,
                            int n, int stride, uint32_t * slot_done,
                            uint32_t * progress, uint32_t seq,
                            hipStream_t stream) {
    int grid = (n + kBlock - 1) / kBlock;
    if (grid < 1) grid = 1;
    if (grid > 256) grid = 256;
    hipLaunchKernelGGL(fast_reduce_add_kernel, dim3(grid), dim3(kBlock), 0, stream,
                       dst, scratch, n_peers, n, stride, slot_done, progress, seq);
}

}  // namespace dflash::cluster
