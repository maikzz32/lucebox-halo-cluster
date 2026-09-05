// cluster_flag_latency - can a GPU wait on a flag another agent writes, and
// what does that cost?
//
// This is the one unknown left in a lean decode-path reduction. The parts that
// are already measured, two Strix Halo over 25 GbE RoCE:
//
//   13.65 us   ib_write_lat, 10 KiB, typical
//   117 us     what an in-situ RCCL all-reduce of the same 10 KiB costs inside
//              a decode step (11.4 ms for 97 of them)
//
// The gap is not the wire. A lean path would be: the GPU publishes its partial
// and raises a flag, a host progress thread sees the flag and posts an RDMA
// write to the peer, the peer's GPU sees the arriving flag and adds. The wire
// leg is known; what is not is the two GPU<->host legs, and on an APU with
// unified memory there is no reason for them to be slow -- both agents are
// looking at the same DRAM.
//
// So: a GPU kernel raises a flag and spins on a reply; a host thread spins on
// the flag and replies. The round trip of that is the cost the lean path would
// add to the 13.65 us. If it is a few microseconds, the design holds and a
// reduction lands near 30 us against RCCL's 117. If it is tens, it does not.
//
//   cluster_flag_latency [iterations]
//
// Nothing here is cluster-specific: it is one process, one GPU, one thread.

#include "ggml-cuda.h"

#include <hip/hip_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#define HIP_OK(x)                                                              \
    do {                                                                       \
        hipError_t e_ = (x);                                                   \
        if (e_ != hipSuccess) {                                                \
            std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x,   \
                         hipGetErrorString(e_));                               \
            return 1;                                                          \
        }                                                                      \
    } while (0)

// One iteration: publish `seq` for the host, then wait until the host has
// echoed it back. System scope on both sides -- device scope would let the
// store sit in a cache the host never looks into.
__global__ void gpu_publish(volatile uint32_t * flag, uint32_t value) {
    if (threadIdx.x || blockIdx.x) return;
    *flag = value;
    __threadfence_system();
}

__global__ void gpu_observe(volatile uint32_t * flag, uint32_t want,
                            uint32_t * out, uint64_t spin_limit) {
    if (threadIdx.x || blockIdx.x) return;
    uint64_t spins = 0;
    while (*flag != want) {
        if (++spins > spin_limit) { *out = 0; return; }
    }
    __threadfence_system();
    *out = 1;
}

// Plain volatile accesses and a system fence, not C11 atomics: an atomic on
// fine-grained system memory selects an instruction this APU faults on.
__global__ void ping_pong(volatile uint32_t * to_host,
                          volatile uint32_t * from_host,
                          int iters,
                          uint64_t spin_limit) {
    if (threadIdx.x || blockIdx.x) return;
    for (int i = 1; i <= iters; ++i) {
        *to_host = (uint32_t) i;
        __threadfence_system();
        uint64_t spins = 0;
        while (*from_host != (uint32_t) i) {
            if (++spins > spin_limit) return;
        }
        __threadfence_system();
    }
}

int main(int argc, char ** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 20000;

    // Fine-grained coherent host memory: the GPU and the CPU see each other's
    // stores without an explicit copy. On an APU this is the same physical
    // DRAM the weights live in, which is the whole reason this design is
    // possible here without GPUDirect.
    // A page each, and the device pointer asked for by name rather than
    // assumed: a four-byte host allocation handed straight to a kernel faults
    // on this runtime.
    uint32_t * to_host = nullptr;
    uint32_t * from_host = nullptr;
    // Pinned host memory, mapped into the GPU's address space once. Managed
    // memory is not an option here: gfx1151 reports XNACK disabled, so the GPU
    // cannot take a page fault, and a managed page the CPU has touched faults
    // the moment a kernel reads it. Pinned pages never migrate, so they need
    // no fault to be reached.
    HIP_OK(hipHostMalloc((void **) &to_host, 4096, hipHostMallocDefault));
    HIP_OK(hipHostMalloc((void **) &from_host, 4096, hipHostMallocDefault));
    *to_host = 0;
    *from_host = 0;
    uint32_t * to_host_dev = nullptr;
    uint32_t * from_host_dev = nullptr;
    HIP_OK(hipHostGetDevicePointer((void **) &to_host_dev, to_host, 0));
    HIP_OK(hipHostGetDevicePointer((void **) &from_host_dev, from_host, 0));

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> host_saw{0};

    // Each direction on its own first. Concurrent fine-grained access is the
    // thing in doubt, and a failure that only shows up when both agents spin
    // means something different from one that shows up immediately.
    uint32_t * verdict = nullptr;
    HIP_OK(hipHostMalloc((void **) &verdict, 4096, hipHostMallocDefault));
    *verdict = 0;

    hipLaunchKernelGGL(gpu_publish, dim3(1), dim3(1), 0, 0, to_host_dev, 7u);
    HIP_OK(hipDeviceSynchronize());
    std::printf("GPU -> host store visible: %s\n", (*to_host == 7u) ? "yes" : "NO");

    *from_host = 9u;
    hipLaunchKernelGGL(gpu_observe, dim3(1), dim3(1), 0, 0, from_host_dev, 9u,
                       verdict, (uint64_t) 50000000);
    HIP_OK(hipDeviceSynchronize());
    std::printf("host -> GPU store visible: %s\n", (*verdict == 1u) ? "yes" : "NO");

    *to_host = 0;
    *from_host = 0;

    std::thread progress([&]() {
        uint32_t last = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            const uint32_t v = *(volatile uint32_t *) to_host;
            if (v != last) {
                last = v;
                *(volatile uint32_t *) from_host = v;
                host_saw.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });

    hipLaunchKernelGGL(ping_pong, dim3(1), dim3(1), 0, 0, to_host_dev,
                       from_host_dev, 128, (uint64_t) 200000000);
    HIP_OK(hipDeviceSynchronize());
    *to_host = 0;
    *from_host = 0;

    const auto t0 = std::chrono::steady_clock::now();
    hipLaunchKernelGGL(ping_pong, dim3(1), dim3(1), 0, 0, to_host_dev,
                       from_host_dev, iters, (uint64_t) 200000000);
    HIP_OK(hipDeviceSynchronize());
    const auto t1 = std::chrono::steady_clock::now();

    stop.store(true);
    progress.join();

    const double total_us =
        std::chrono::duration<double, std::micro>(t1 - t0).count();
    const uint32_t done = *from_host;

    std::printf("round trips completed: %u of %d\n", done, iters);
    if (done < (uint32_t) iters) {
        std::printf("=> the GPU gave up waiting; this memory is not coherent "
                    "the way the design needs\n");
        HIP_OK(hipHostFree(to_host));
        HIP_OK(hipHostFree(from_host));
        return 2;
    }
    std::printf("GPU->host->GPU round trip: %.2f us\n", total_us / (double) iters);
    std::printf("with the measured 13.65 us wire leg, a lean reduction would "
                "cost about %.1f us, against RCCL's 117\n",
                total_us / (double) iters + 13.65);

    HIP_OK(hipHostFree(to_host));
    HIP_OK(hipHostFree(from_host));
    return 0;
}
