// device_read_bandwidth - what this GPU can actually read, so a decode step's
// rate can be compared against something real.
//
// A qwen4exp decode step moves 4266 MB of weights in 38.9 ms on one node, and
// 2765 MB per rank in 29.6 ms on two: 110 and 93 GB/s. LPDDR5X-8000 on a
// 256-bit bus is 256 GB/s on paper. Whether the step is leaving half the
// machine on the table or is already at the metal decides whether the next
// thing to work on is the kernels or the parallelism -- and paper numbers do
// not decide it.
//
//   device_read_bandwidth [MiB] [iterations]
//
// Streams a device buffer through the simplest possible read: float4 loads, a
// grid-stride loop, one add per element. Nothing here is a model kernel; it is
// the ceiling those kernels are being measured against.

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define HIP_OK(x)                                                              \
    do {                                                                       \
        hipError_t e_ = (x);                                                   \
        if (e_ != hipSuccess) {                                                \
            std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x,   \
                         hipGetErrorString(e_));                               \
            return 1;                                                          \
        }                                                                      \
    } while (0)

__global__ void stream_read(const float4 * __restrict__ src, size_t n4,
                            float * __restrict__ sink) {
    const size_t stride = (size_t) blockDim.x * gridDim.x;
    float acc = 0.0f;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        const float4 v = src[i];
        acc += v.x + v.y + v.z + v.w;
    }
    // Never true, and the compiler cannot know it: without a use the loads are
    // dead and the whole loop disappears.
    if (acc == 1.2345e-30f) sink[0] = acc;
}

int main(int argc, char ** argv) {
    const size_t mib = argc > 1 ? (size_t) std::atoll(argv[1]) : 4096;
    const int iters = argc > 2 ? std::atoi(argv[2]) : 20;

    const size_t bytes = mib * 1024ull * 1024ull;
    const size_t n4 = bytes / sizeof(float4);

    float4 * buf = nullptr;
    float * sink = nullptr;
    if (hipMalloc((void **) &buf, bytes) != hipSuccess) {
        std::fprintf(stderr, "hipMalloc %zu MiB failed\n", mib);
        return 1;
    }
    HIP_OK(hipMalloc((void **) &sink, sizeof(float)));
    HIP_OK(hipMemset(buf, 0, bytes));

    hipDeviceProp_t prop{};
    HIP_OK(hipGetDeviceProperties(&prop, 0));
    const int blocks = prop.multiProcessorCount * 8;

    hipLaunchKernelGGL(stream_read, dim3(blocks), dim3(256), 0, 0, buf, n4, sink);
    HIP_OK(hipDeviceSynchronize());

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        hipLaunchKernelGGL(stream_read, dim3(blocks), dim3(256), 0, 0, buf, n4, sink);
    }
    HIP_OK(hipDeviceSynchronize());
    const auto t1 = std::chrono::steady_clock::now();

    const double secs = std::chrono::duration<double>(t1 - t0).count();
    const double gb = (double) bytes * (double) iters / 1e9;
    std::printf("%s: %zu MiB x %d, %.3f s -> %.1f GB/s\n",
                prop.name, mib, iters, secs, gb / secs);
    std::printf("a decode step reads at 110 GB/s on one node and 93 on two;\n"
                "against this ceiling that is %.0f%% and %.0f%%\n",
                100.0 * 110.0 / (gb / secs), 100.0 * 93.0 / (gb / secs));

    HIP_OK(hipFree(buf));
    HIP_OK(hipFree(sink));
    return 0;
}
