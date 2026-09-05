// fast_reduce_kernels.cu - the device half of the lean decode-path reduction.
//
// Two kernels per reduction, both launched on the backend's own stream so they
// sit in the graph exactly where the collective used to:
//
//   publish   copy this rank's partial into a pinned staging buffer the NIC
//             can read, then raise a flag the host progress thread is
//             spinning on.
//   wait_add  spin until every peer's flag has arrived, then add their
//             partials into the output.
//
// The flags live in pinned host memory mapped into the GPU's address space.
// That is not managed memory: gfx1151 reports XNACK disabled, so the GPU
// cannot take a page fault, and a managed page the CPU has touched faults the
// moment a kernel reads it. Pinned pages never migrate, so they need no fault
// to be reached -- and the round trip through one measures 0.79 us
// (server/test/cluster_flag_latency.cu), which is what makes this worth doing
// against RCCL's 117.
//
// Every spin is bounded. A rank that dies must not leave a kernel spinning on
// a flag that will never arrive: the GPU has no watchdog here, and a hung
// kernel takes the whole server with it.

#include <hip/hip_runtime.h>

#include <cstdint>

namespace dflash::cluster {

namespace {
constexpr int kBlock = 256;
}

// Copy out, fence, then raise the flag. The fence is what makes the copy
// visible to the host before the flag is: without it the progress thread can
// see the flag and send a buffer that is still being written.
__global__ void fast_reduce_publish_kernel(const float * __restrict__ src,
                                           float * __restrict__ staging,
                                           volatile uint32_t * pub_flag,
                                           volatile uint32_t * slot_done,
                                           uint32_t seq,
                                           uint32_t slot_span,
                                           int n,
                                           uint64_t spin_limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Wait for the slot to come free before writing into it. The GPU can be a
    // whole step's worth of reductions ahead of the progress thread, so the
    // ring has to be defended rather than assumed wide enough.
    if (i == 0 && seq > slot_span) {
        uint64_t spins = 0;
        while (*slot_done + slot_span < seq) {
            if (++spins > spin_limit) break;
        }
    }
    __syncthreads();

    for (int k = i; k < n; k += blockDim.x * gridDim.x) {
        staging[k] = src[k];
    }
    __threadfence_system();

    if (i == 0) {
        *pub_flag = seq;
        __threadfence_system();
    }
}

// Spin on the peers' flags, then add. One thread watches; the rest wait on it,
// so the flag is read by one lane instead of by every lane in the grid.
__global__ void fast_reduce_wait_add_kernel(float * __restrict__ dst,
                                            const float * const * __restrict__ peer_bufs,
                                            volatile uint32_t * const * __restrict__ peer_flags,
                                            volatile uint32_t * slot_done,
                                            uint32_t seq,
                                            int n_peers,
                                            int n,
                                            uint64_t spin_limit,
                                            volatile uint32_t * timed_out) {
    __shared__ int ready;
    if (threadIdx.x == 0) {
        ready = 1;
        for (int p = 0; p < n_peers; ++p) {
            uint64_t spins = 0;
            while (*peer_flags[p] != seq) {
                if (++spins > spin_limit) { ready = 0; break; }
            }
        }
        if (!ready && blockIdx.x == 0) *timed_out = seq;
    }
    __syncthreads();
    if (!ready) return;
    __threadfence_system();

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int k = i; k < n; k += blockDim.x * gridDim.x) {
        float acc = dst[k];
        for (int p = 0; p < n_peers; ++p) {
            acc += peer_bufs[p][k];
        }
        dst[k] = acc;
    }

    // The slot is only free once its data has been consumed, not once it has
    // arrived.
    __threadfence_system();
    if (i == 0) {
        *slot_done = seq;
        __threadfence_system();
    }
}

void fast_reduce_launch_publish(const float * src, float * staging,
                                uint32_t * pub_flag, uint32_t * slot_done,
                                uint32_t seq, uint32_t slot_span, int n,
                                uint64_t spin_limit, hipStream_t stream) {
    const int grid = (n + kBlock - 1) / kBlock;
    hipLaunchKernelGGL(fast_reduce_publish_kernel, dim3(grid > 0 ? grid : 1),
                       dim3(kBlock), 0, stream, src, staging,
                       (volatile uint32_t *) pub_flag,
                       (volatile uint32_t *) slot_done,
                       seq, slot_span, n, spin_limit);
}

void fast_reduce_launch_wait_add(float * dst, const float * const * peer_bufs,
                                 uint32_t * const * peer_flags,
                                 uint32_t * slot_done, uint32_t seq,
                                 int n_peers, int n, uint64_t spin_limit,
                                 uint32_t * timed_out, hipStream_t stream) {
    const int grid = (n + kBlock - 1) / kBlock;
    hipLaunchKernelGGL(fast_reduce_wait_add_kernel, dim3(grid > 0 ? grid : 1),
                       dim3(kBlock), 0, stream, dst, peer_bufs,
                       (volatile uint32_t * const *) peer_flags,
                       (volatile uint32_t *) slot_done, seq, n_peers, n,
                       spin_limit, (volatile uint32_t *) timed_out);
}

}  // namespace dflash::cluster
