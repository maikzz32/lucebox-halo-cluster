# lucebox-halo-cluster: DeepSeek V4 Flash expert-parallel across 2-4 Strix Halo nodes

Operator guide for the multi-node mode of this Lucebox fork. Everything here
is specific to the fork (`-DDFLASH27B_CLUSTER=ON`); upstream Lucebox is a
single-host engine. Numbers marked **TBD** have not been measured yet and must
not be quoted.

## 1. Architecture in one page

* **SPMD expert parallelism.** One `dflash_server` process per node. Every rank
  loads the *dense* part of DeepSeek V4 Flash replicated (attention/MLA,
  hidden-context, indexer, KV cache, embeddings, lm_head) and only **1/N of the
  routed experts of every MoE layer** (`--cluster-expert-placement uniform`:
  expert `e` -> rank `e % N`). Per rank that is roughly `dense + experts/N +
  KV (+ DSpark drafter on rank 0)`.
* **One all-reduce per MoE layer.** Each rank evaluates its local experts as a
  partial sum (the existing masked-route hybrid path with cold owner `None`),
  then a single RCCL `ncclAllReduce` over the `[4096 x n_tokens]` partial
  (always f32: bf16 partials measurably change greedy outputs, see "Measured
  results") makes the hidden state identical on all ranks. 43 collectives per forward; at the measured ~77 us per 4-rank
  all-reduce that is ~3 ms against a ~35 ms decode step.
* **Rank 0 decides, everybody follows.** Rank 0 (head) serves HTTP, broadcasts
  the request descriptor, runs the same generate loop as the workers and
  broadcasts every sampling / acceptance / EOS / cancel decision (a few int32
  per step) over a TCP control channel. Kernel non-determinism therefore
  cannot make ranks diverge.
* **Path 3b: the all-reduce is a graph node (default).** `ggml_cluster_allreduce()`
  adds a node whose value is its input summed across the ranks; the CUDA/HIP
  backend runs the collective on the same stream as the surrounding kernels, so
  no host round trip separates producer and consumer. That is what lets a
  cluster rank use the **fused whole-model graph**: one graph per forward
  instead of 43 host-driven per-layer graphs plus a separate router graph, a
  separate shared-expert graph and three synchronizing copies around every
  all-reduce. ggml itself stays free of any collective library: the node calls
  a callback the server registers. Because a rank holds only its shard, the
  graph masks foreign routes on device through the owner LUT that the hybrid
  machinery already builds (`valid_lut`), with id -1 so the MMVQ kernels retire
  those routes early instead of computing them and multiplying by zero. The
  shared expert is left out of the routed partial and added after the
  reduction, so it is counted exactly once. `DFLASH_CLUSTER_NO_INGRAPH_ALLREDUCE=1`
  returns to path 3a, which is still the path prefill takes.
* **DSpark on the cluster (WP4).** Every rank loads the drafter and drafts, but
  only rank 0's block, its adaptive width, its accept count and its bonus token
  are authoritative; they travel as `Draft` and `Accept` frames inside the
  decode loop. The redundant draft forwards on the workers run at the same time
  as the head's, so they cost no wall-clock, and they keep every rank's feature
  window warm. Verification runs on every rank through the expert-parallel
  graph, which is the same per-layer forward AR decode uses, so a verify batch
  of q tokens is q columns in the same all-reduce. The seed token (the first
  token of a request, an argmax over prefill logits that differ in their last
  bits between ranks) is decided by rank 0 as well. An EMPTY draft block is the
  head's end marker: it ends a request only the head can see ending (client
  cancel, closed connection). Every other stop reason (EOS, token budget) is a
  function of the accepted tokens and is reached on all ranks in the same step.
  `--cluster-verify-hash` probes the AR loop only; a speculative request does
  not hash per step.
* **Transport.** RCCL over RoCE v2 with `NCCL_NET_GDR_LEVEL=0` (gfx1151 has no
  GPUDirect; the host bounce is a memcpy in the same DRAM on Strix Halo).
  Bootstrap over TCP: workers connect to `--cluster-head`, send `Hello{rank,
  build_sha, model_sha, placement_hash}`, the head checks equality and answers
  `Welcome{ncclUniqueId, ...}`; every rank then calls
  `ncclCommInitRankConfig` (non-blocking, so a watchdog can `ncclCommAbort`).

Source layout: `server/src/cluster/` (`cluster_config.h` CLI, `cluster_protocol.h`
wire format, `cluster_control.h` TCP channel, `cluster_comm.h` RCCL wrapper,
`cluster_expert_placement.h` expert->rank map). ggml changes are tracked as a
patch series in `server/deps/patches/cluster/`.

## 2. Hardware inventory (verified 2026-09-03)

| Node | ssh alias | LAN IP | RoCE IP | NIC | RoCE iface | HCA |
|---|---|---|---|---|---|---|
| StrixHalo1 | `strix1` | 192.168.1.15 | 192.168.100.1 | Intel E810-C (4-port) | `enp197s0f3np3` | `rocep197s0f3` |
| StrixHalo2 | `strix2` | 192.168.1.16 | 192.168.100.2 | Intel E810-C (4-port) | `enp197s0f3np3` | `rocep197s0f3` |
| StrixHalo3 | `strix3` | 192.168.1.17 | 192.168.100.3 | Intel E810-C (4-port) | `enp197s0f3np3` | `rocep197s0f3` |
| StrixHalo4 | `strix4` | 192.168.1.18 | 192.168.100.4 | Intel E830-L (2-port) | `enp197s0f1np1` | `rocep197s0f1` |

All nodes: AMD Ryzen AI MAX+ 395, Radeon 8060S (gfx1151), 124 GB RAM, Fedora
Linux 45 Server (prerelease), kernel 7.2.0-61.fc45, user `maik`, podman 6.1
(docker additionally on strix1), toolbox 0.3, gcc 16.2 on the host, **no ROCm
on the hosts** (no `/opt/rocm`, no `hipcc`), cmake only on strix3, ninja on
strix1/2. Fabric: RoCE v2, 192.168.100.0/24, **MTU 9000, 25 GbE** (not 100).
No DNS in the RDMA net. Measured with the user's vLLM project: RCCL all-reduce
over 4 ranks constant ~77 us; 11.7 million all-reduces in 15 min without a
hang; ~5 us RDMA latency.

Kernel command line already present on all nodes:
`iommu=pt pci=realloc amdgpu.gttsize=126976 ttm.pages_limit=32505856
ttm.page_pool_size=32505856 amdgpu.gpu_recovery=1 amdgpu.lockup_timeout=10000
pcie_aspm=off`.

**The nodes normally run a vLLM Ray cluster** that holds ~100 GB of GPU memory
and leaves ~12 GB RAM free per node. Every GPU test below requires that
cluster to be stopped first. None of the scripts in `scripts/cluster/` will
stop, kill or remove a container they did not start.

## 3. Host prerequisites

```sh
# check only (idempotent), per node:
ssh strix1 'bash -s' < scripts/cluster/host_prep.sh
# or all at once:
make cluster-host-check CLUSTER_HOSTS="strix1 strix2 strix3 strix4"
# fix MTU / memlock / /etc/hosts interactively:
ssh -t strix4 'bash -s -- --apply' < scripts/cluster/host_prep.sh
```

Checked: kernel parameters, RoCE interface up with the 192.168.100.x address
and MTU 9000, `rdma link` ACTIVE for the HCA, rdma-core version (**>= v64 is
required for the E830 on StrixHalo4**; Fedora 44 shipped v61, the E810 nodes
do not care), the GID table (RoCE v2 entry for the node IP; expected at index
1 = `--cluster-gid-index 1`), `ulimit -l unlimited`, `/dev/infiniband`
`/dev/kfd` `/dev/dri`, podman, and the `/etc/hosts` lines
`192.168.100.N StrixHaloN`. Kernel parameters are never changed automatically;
the script prints the `grubby` command.

## 4. Container image

Base: `ghcr.io/maikzz32/strix-vllm-gfx1151:dev-rocm10` (Fedora 44, ROCm 10.0.0
from the TheRock pip SDK at `/opt/rocm`, RCCL 2.30.4 with gfx1151 kernels,
hipcc, gcc 16.2, ninja, libibverbs, python 3.12; **no cmake** - the Dockerfile
adds it). Stock TheRock RCCL supports gfx1151 (ROCm/TheRock#4935, #5376); no
custom RCCL build is needed.

```sh
# on strix1 (docker + buildx available there; podman also works):
git clone https://github.com/maikzz32/lucebox-halo-cluster.git && cd lucebox-halo-cluster
git checkout cluster/main
make build-rocm-cluster                       # -> lucebox-halo-cluster:dev
# or explicitly:
docker buildx bake rocm-cluster-local --load
# reproducible variant from fedora:44 + pip SDK instead of the prebuilt base:
ROCM_SDK_FROM_PIP=1 CLUSTER_ROCM_VERSION=10.0.0 docker buildx bake rocm-cluster-local --load
# tag + distribute to the other nodes (or push to GHCR):
docker tag lucebox-halo-cluster:dev ghcr.io/maikzz32/lucebox-halo-cluster:dev
docker push ghcr.io/maikzz32/lucebox-halo-cluster:dev
for h in strix2 strix3 strix4; do ssh $h podman pull ghcr.io/maikzz32/lucebox-halo-cluster:dev; done
```

Stages: `base` (toolchain + rdma-core/perftest/OpenMPI + RCCL env defaults),
`rccl-tests` (`/opt/rccl-tests/bin/all_reduce_perf` etc., MPI on),
`builder` (Lucebox with `-DDFLASH27B_CLUSTER=ON -DGGML_HIP_GRAPHS=ON
-DDFLASH27B_ROCMFP2_AFFINE=ON`, runs `test_cluster_unit` and
`test_feature_gate` at build time), `runtime` (binaries in `/opt/lucebox/bin`,
`ENTRYPOINT dflash_server`). Building needs no GPU. Expect the HIP compile of
`deepseek4_graph.cpp` to dominate; the first build takes well over an hour on
one node.

Toolchain recipe inside the image (verified in dev-rocm10; both the
Dockerfile and `build_in_container.sh` use exactly this): `dnf install gcc-c++
libstdc++-devel glibc-devel cmake ninja-build`, `PATH` with `/usr/bin` **before**
`/opt/rocm/lib/llvm/bin` (otherwise CMake takes ROCm's clang++ as host compiler
and the link fails with `unable to find library -lstdc++`), and
`cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
-DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++
-DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_HIP_FLAGS=-fPIC` (g++ emits
non-PIE objects while ROCm's clang links PIE; without the two PIC flags the
final `dflash_server` link dies with `ld.lld: relocation R_X86_64_32S cannot be
used against local symbol; recompile with -fPIC`). Build resources on a node:
32 cores, ~68 GiB RAM available even with vLLM running; `JOBS=6` with
`nice -n 10` is the default.

ROCm host/driver rule: the container userspace must be at least as new as the
host's amdgpu driver (kernel 7.2 here). The upstream `Dockerfile.rocm`
(ROCm 6.4.1) segfaults at model load on this combination; do not use it on the
Halos.

Dev loop without a full image rebuild:

```sh
scripts/cluster/build_in_container.sh strix1               # rsync + incremental build in ~/lucebox-halo-cluster/build-cluster
scripts/cluster/build_in_container.sh strix1 --tests-only  # GPU-free: test_cluster_unit + test_feature_gate
scripts/cluster/build_in_container.sh strix1 --shell       # poke around in the build container
```

## 5. Model download

Verified artifacts (2026-09-03; `deepseek4` architecture, ROCMFPX mixed
quantisation for Strix Halo):

| Role | HF repo | File | Bytes | sha256 |
|---|---|---|---|---|
| Target | `Lucebox/DeepSeek-V4-Flash-0731-ROCmFP3` (the repo upstream's README links; the identical file is also in `Lucebox/DeepSeek-V4-Flash-0731-ROCMFPX`) | `DeepSeek-V4-Flash-0731-ROCMFPX-MIX-STRIX.gguf` | 98,294,917,184 | `7c0789d190fdd2acad93255825822ca276f29d13f9410f2ac65f5f7a542b0a38` |
| DSpark drafter | `Lucebox/DeepSeek-V4-Flash-0731-DSpark-GGUF` | `DeepSeek-V4-Flash-0731-DSpark-draft-Q4RMFP4-denseF16.gguf` | 10,648,656,160 | `58e7337597a917fdd033aca30a38263740dc53999dac187a676a36b8daf9e63d` |

The uniform `DeepSeek-V4-Flash-ROCMFP2-STRIX.gguf` (102,320,631,200 bytes,
sha256 `8fa6c30d…9208`) also loads and was the only shardable artifact until
2026-09-04. It is slower and markedly less accurate than the adaptive one and
should not be used: two nodes measure 38.4 against 42.3 tok/s on the benchmark
prompt, 17.2 against 20.6 on a free prompt, and 12/60 against 60/60 on
exact-copy fidelity. Also unused: the drafter variant without `0731`,
`DeepSeek-V4-Flash-DSpark-draft-Q4RMFP4-denseF16.gguf` (11,304,737,056 bytes,
sha256 `48883d35…f746`).

Models dir on every node: `/home/maik/gguf/ds4/` (identical files, identical
paths; the head refuses workers whose `model_sha` differs).

```sh
# resumable curl -C - from the URLs above + sha256 check, on the first node:
make cluster-fetch-models CLUSTER_HOSTS="strix1 strix2 strix3 strix4"
# equivalent, and then copy over the RoCE net (~15 min for 109 GB at 25 GbE):
ssh strix1 'bash -s -- --dir /home/maik/gguf/ds4 --sync-to "strix2 strix3 strix4"' \
    < scripts/cluster/fetch_models.sh
```

`scripts/cluster/fetch_models.sh --help` lists the options (`--no-verify`,
`--only-sync`). The download on strix1 was started with the user's own
`~/gguf/ds4/download.sh`; `fetch_models.sh` resumes partial files.

## 6. Launch

```sh
# 2 nodes (head = first host):
scripts/cluster/launch_cluster.sh "strix1 strix2" \
    DeepSeek-V4-Flash-0731-ROCMFPX-MIX-STRIX.gguf \
    DeepSeek-V4-Flash-0731-DSpark-draft-Q4RMFP4-denseF16.gguf
# 4 nodes:
scripts/cluster/launch_cluster.sh "strix1 strix2 strix3 strix4" <target.gguf> <dspark.gguf>
# with extra dflash_server flags for every rank:
scripts/cluster/launch_cluster.sh "strix1 strix2" <target> <dspark> -- --cluster-verify-hash 8
scripts/cluster/launch_cluster.sh --status "strix1 strix2"
scripts/cluster/launch_cluster.sh --down   "strix1 strix2"
# Makefile shortcuts: make cluster-up / cluster-status / cluster-down CLUSTER_HOSTS="strix1 strix2"
```

The script starts rank 0 first, waits for its `cluster: listening` log line
(or 20 s), then the workers. Each rank is a podman container `lucebox-rank<N>`
started with the flags proven with RCCL on this cluster:

```
--device /dev/kfd --device /dev/dri --device /dev/infiniband --group-add keep-groups
--security-opt seccomp=unconfined --security-opt label=disable
--network host --ipc host --pids-limit -1 --ulimit memlock=-1:-1
--add-host StrixHalo1:192.168.100.1 ... --add-host StrixHalo4:192.168.100.4
-v /home/maik/gguf/ds4:/models:ro
```

Per-rank command (rank 0 additionally gets `--host 0.0.0.0 --port 8016`):

```
dflash_server /models/<target>.gguf \
  --cluster-rank R --cluster-size N --cluster-head 192.168.100.1:9400 \
  --cluster-ifname <iface> --cluster-ib-hca <hca> --cluster-gid-index 1 \
  --cluster-expert-placement uniform --target-device hip:0 \
  --ds4-expert-top-k 6 --ds4-prefill sparse --chunk 2048 --max-ctx 32768 \
  --prefix-cache-slots 32 --prefill-cache-slots 0
```

Environment in every container: `DFLASH_DS4_SPEC=1 DFLASH_DS4_DRAFT=/models/<dspark>
DFLASH_DS4_DRAFT_GPU=0 DFLASH_DS4_SPEC_Q=4 DFLASH_DS4_FUSED_VERIFY=1` plus the
RCCL defaults of section 8.

Manual equivalent for one rank: see the comment block at the end of
`Dockerfile.rocm-cluster`.

## 7. CLI flag reference (`server/src/cluster/cluster_config.h`)

| Flag | Meaning | Default |
|---|---|---|
| `--cluster-rank <r>` | rank in `[0, N)`; 0 = head (serves HTTP) | -1 (off) |
| `--cluster-size <N>` | ranks, `2 <= N <= 8`; 0 = not a cluster run | 0 |
| `--cluster-head <host:port>` | TCP control endpoint; rank 0 binds, workers connect | port 9400 |
| `--cluster-ifname <iface>` | -> `NCCL_SOCKET_IFNAME` | — |
| `--cluster-ib-hca <hca[:port]>` | -> `NCCL_IB_HCA` | — |
| `--cluster-gid-index <n>` | -> `NCCL_IB_GID_INDEX` (RoCE v2 GID of the node IP) | -1 |
| `--cluster-expert-placement uniform\|balanced\|<file.json>` | expert -> rank map; `balanced` needs `DFLASH_DS4_HOTNESS_CSV`; JSON from `scripts/cluster/build_expert_placement.py` | `uniform` |
| `--cluster-replicate-hot <k>` | k hottest experts per layer resident on every rank | 0 |
| `--cluster-shared-expert replicate\|shard\|rank0` | shared-expert strategy | `replicate` |
| `--cluster-allreduce-dtype f32\|bf16\|auto` | wire precision; `auto` = f32 below 256 KiB, bf16 above | `auto` |
| `--cluster-timeout-ms <ms>` | watchdog for collectives and control channel | 30000 |
| `--cluster-verify-hash <n>` | debug: cross-rank hidden-state hash every n steps | 0 |
| `--cluster-selftest` | bootstrap, run the collective self-test (1000 x 64 KiB, 43 x 16 MiB), exit | off |

Gate rules (feature gate): cluster requires `deepseek4`, HIP, no layer split, no
remote target shard, no paged attention, `max_concurrency == 1`, no PFlash /
KVFlash. `--prefix-cache-slots` is allowed since protocol 2; the disk and
prefill caches stay off.
Ranks must run identical binaries, identical model files and the same
placement (checked via `build_sha`, `model_sha`, `placement_hash` in `Hello`).

## 8. RCCL environment reference

Set as image defaults (`Dockerfile.rocm-cluster`) and by `launch_cluster.sh`;
`dflash_server` exports the three transport variables from the CLI flags.
Existing environment values always win.

| Variable | Default | Why |
|---|---|---|
| `NCCL_NET_GDR_LEVEL` | `0` | gfx1151 has no GPUDirect RDMA. Bounce via host memory, which on Strix Halo is the same DRAM the GPU uses. Always set. |
| `NCCL_IB_DISABLE` | `0` | Force the IB/RoCE transport. Without `/dev/infiniband` RCCL prints `NET/IB : No device found` and silently uses TCP (10x slower). |
| `NCCL_IB_HCA` | `rocep197s0f3` (node 4: `rocep197s0f1`) | Only the cabled port; the E810 has 4 ports. |
| `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` | `enp197s0f3np3` (node 4: `enp197s0f1np1`) | Bootstrap sockets on the RoCE net, not the LAN. |
| `NCCL_IB_GID_INDEX` | `1` | RoCE v2 GID of the node IP (verify with `host_prep.sh`). |
| `NCCL_IB_QPS_PER_CONNECTION` | `2` | Two QPs per peer improve 25 GbE utilisation for the 16 MiB prefill reductions. |
| `NCCL_IB_TIMEOUT` | `22` | 4.096 us x 2^22 ~ 17 s ACK timeout: tolerate RoCE without PFC/ECN on this switch. |
| `NCCL_IB_RETRY_CNT` | `7` | Max retries before a QP error; with the timeout above this survived 11.7 M all-reduces. |
| `NCCL_ASYNC_ERROR_HANDLING` | `1` | Surface asynchronous errors instead of hanging the collective. |
| `HIP_FORCE_DEV_KERNARG` | `1` | Kernel arguments in device memory; Strix Halo performance default. |
| `NCCL_ALGO` / `NCCL_PROTO` | unset | Start with `Ring` / `Simple` when debugging a hang; then remove. |
| `NCCL_DEBUG` | unset | `INFO` once to confirm `NET/IB` is selected; leave off in production. |

## 9. Fabric baseline

```sh
scripts/cluster/rccl_baseline.sh "strix1 strix2"                      # N=2 one-shot containers
scripts/cluster/rccl_baseline.sh "strix1 strix2 strix3 strix4" --record  # N=4, append below
scripts/cluster/rccl_baseline.sh --exec "strix1 strix2"               # inside running lucebox-rank* containers
scripts/cluster/rccl_baseline.sh --verbs "strix1 strix2 strix3 strix4"  # ib_write_bw / ib_write_lat per pair
```

`all_reduce_perf -b 8K -e 64M -f 2 -g 1`, one rank per node, launched by
`mpirun` inside the image (OpenMPI, `--mca btl_tcp_if_include 192.168.100.0/24`,
all `NCCL_*` exported; per-node interface/HCA via `rccl_rank_env.sh`).
Exit criteria from the plan (WP0): **64 KiB < 150 us, 16 MiB bf16 < 8 ms at N=4**.

### Measured baselines

| Date (UTC) | N | Hosts | 64 KiB time (us) | 16 MiB time (us) | 16 MiB busbw (GB/s) | Notes |
|---|---|---|---|---|---|---|
| 2026-09-03 | 2 | strix1 strix2 | 73 (p50), 818 (p90) | 6070 f32 / 3436 bf16 (p50) | ~2.8 (f32) | `dflash_server --cluster-selftest` (RCCL 2.30.4, 1000+43 collectives, sums exact) while the vLLM/Ray cluster was still running on both hosts, so the p90/p99 tails include co-tenant noise; rccl-tests `all_reduce_perf` still pending |
| 2026-09-03 | 4 | strix1 strix2 strix3 strix4 | 125 (p50), 136 (p90), 173-187 (p99) | 8511-8617 f32 / 4461-4585 bf16 (p50) | ~2.0 (f32) | `dflash_server --cluster-selftest`, RCCL 2.30.4, all 4 ranks OK, sums exact; first 4-node RCCL measurement on Strix Halo; vLLM/Ray cluster still running on all hosts (GPU busy, CPU idle) |
<!-- rccl-baseline-end -->

Reference from the vLLM project (different code, same fabric): 4-rank
all-reduce ~77 us constant for small messages; do not copy that into the table.

## 10. Verification protocol

Determinism is a merge gate, not a nice-to-have.

1. **Single-node reference.** Same binary, same model files, one node,
   `temperature: 0`, fixed `seed`, `max_tokens: 256`:
   ```sh
   curl -s http://192.168.100.1:8016/v1/chat/completions -H 'content-type: application/json' \
     -d '{"model":"dflash","messages":[{"role":"user","content":"Explain RoCE v2 in three sentences."}],"max_tokens":256,"temperature":0,"seed":1}' \
     | jq -r .choices[0].message.content | sha256sum
   ```
   Run 3x; hashes must be identical.
2. **Cluster run**, same request, 3x on N=2 (later N=4): all hashes must equal
   the single-node hash. Bit-identical *logits* are not expected (summation
   order changes); token-identical greedy output is.
3. During bring-up run with `--cluster-verify-hash 8`: ranks exchange FNV-1a
   hashes of the final hidden state and the accepted tokens every 8 steps; the
   first divergent step aborts the request with a rank-attributed error.
   `DFLASH_CLUSTER_TRACE=1` additionally reports the first divergent layer.
4. DSpark (M2): the 2-node DSpark hash must equal the 2-node AR hash;
   `usage.timings.spec_decode_ran == true`, `accept_rate` within +-0.02 of the
   single node.
5. Goldens for CI live in `server/tests/cluster-golden/` (see its README).

## 11. Telemetry

Per request (`usage.timings.cluster` in the HTTP response, WP6):
`size`, `per_rank[rank].{steps, compute_us, allreduce_calls, allreduce_bytes,
allreduce_wait_us, ctrl_wait_us, peak_device_bytes}`, `ctrl_us`. `/props.cluster`:
`{size, rank, ifname, hca, gid_index, rccl_version, placement, gdr: false}`.
`/status` per rank. `DFLASH_DS4_TIMING=1` adds `allreduce=` to the
`[ds4-spec-t]` banner. `DeepSeek4StepTelemetry` gains `cluster_allreduce_us`,
`cluster_allreduce_bytes`, `cluster_ctrl_wait_us`, `cluster_barrier_us`.
Field names are the contract from `cluster_protocol.h::RequestReportMsg`.

## 12. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `NET/IB : No device found`, all-reduce ~ms instead of ~us | Container lacks `--device /dev/infiniband` (RCCL fell back to TCP). Also check `NCCL_IB_DISABLE=0` and that `rdma link` is ACTIVE on the host. |
| `Unable to resolve hostname StrixHaloN` / bootstrap hangs | No DNS in the RDMA net: the container needs `--add-host StrixHaloN:192.168.100.N` (launch script does this) and the host `/etc/hosts` entries (`host_prep.sh --apply`). |
| `ibv_modify_qp failed` / `NET/IB : Got completion ... status=12` | Wrong GID index. Run `host_prep.sh` and look for the RoCE v2 entry of the node IP; pass its index to `--cluster-gid-index`. |
| Bandwidth far below 25 GbE, drops on 16 MiB messages | MTU mismatch (must be 9000 end to end incl. the switch), or `NCCL_IB_QPS_PER_CONNECTION` unset. Confirm with `--verbs`. |
| SIGSEGV at model load, absurd VRAM size reported | ROCm userspace older than the host driver (e.g. upstream `Dockerfile.rocm` with 6.4.1). Use this image (ROCm 10). |
| `hipErrorOutOfMemory` at load, `rocm-smi` shows ~100 GB used | The vLLM Ray cluster is still running. Stop it yourself (`podman ps` on each node); the scripts never do. |
| Worker exits with `Hello mismatch: build_sha` / `model_sha` / `placement_hash` | Different image or model files on a node. `sha256sum` the GGUFs, `podman images --digests`. |
| E830 on strix4: `ibv_devinfo` inside the container shows nothing although `/sys/class/infiniband/rocep197s0f1` exists; 4-rank `ncclCommInitRankConfig` fails with "internal error" / "remote process exited or there was a network error" | rdma-core in the **container** is too old. Measured 2026-09-03: Fedora 44's v61 hides the E830, and rdma-core built from the upstream `v64.0` tag still does not enumerate it; only the Fedora 45 package `rdma-core-64.0-3.fc45` does. Fix: `dnf upgrade --releasever=45 rdma-core libibverbs librdmacm libibverbs-utils librdmacm-utils` in the image (done in `Dockerfile.rocm-cluster` and `scripts/cluster/Containerfile.devbuild`). A freshly rebuilt `strix-vllm-gfx1151:dev-rocm10` base carries the regression too; only containers started from an older build still work. |
| Collective timeout (`wait_stream` deadline), request fails with `DecodeFailed` | One rank is slow or dead: check `--status`, heartbeats stop after 10 s. Raise `--cluster-timeout-ms` only for debugging. |
| `mpirun` in `rccl_baseline.sh` cannot reach peers | The launcher container uses your `~/.ssh` (mounted read-only) to ssh as `maik@192.168.100.N`; make sure the RoCE IPs are in `known_hosts` or accept the `accept-new` prompt once from the head node. |

## 13. Honest performance expectation

Single Strix Halo today: 25.3 tok/s AR, 32.1 tok/s with DSpark q=4. The
bandwidth model in the plan (experts ~75 % of bytes per q=4 verify, imbalance
1.25-1.4, communication 5-10 %) predicts **N=2 ~ 1.3x -> 40-45 tok/s** and
**N=4 ~ 1.7-1.9x -> 55-60 tok/s**. The 50 tok/s target therefore realistically
needs **four nodes** (or two nodes plus attention tensor parallelism, WP8).
Prefill: >= 0.9x single node at N=2, >= 1.5x at N=4 (bf16 16 MiB reduction per
layer at 2048-token chunks). These are predictions until the table in
section 9 and the benchmark in `scripts/cluster/bench_ds4_cluster.py` (WP7)
have real numbers; do not quote them as measurements.

## Measured results (2026-09-03/04, strix1+strix2, `DeepSeek-V4-Flash-ROCMFP2-STRIX.gguf`, path 3a, AR decode)

| Configuration | Decode tok/s (128 tok, temp 0, median of 3) | Output |
|---|---|---|
| single node, monolithic (strix1/strix3, same binary, no DSpark) | 16.6 | sha256 `87964cbd…` (3 runs byte-identical) |
| 2 nodes, uniform expert sharding (50 % experts per rank, 45.7 GiB) | 14.3 | sha256 `87964cbd…` — **byte-identical to the single node** (3 runs byte-identical) |
| 2 nodes, diagnostic placement all experts on rank 0 | — | per-layer FFN checksums match the monolithic path to 7 digits for layers 0-20, then 1e-6..3e-4 float drift |

Correctness proof from `DFLASH_CLUSTER_TRACE=1`: for every layer `partial_pre(rank0) + partial_pre(rank1) == partial_post == full routed sum` to 7 digits; the shared expert is added once. Free-form prompts can still diverge from the single node at greedy near-ties because the HC / attention path amplifies 1e-7 summation-order noise (measured: `hc_after_ffn(7)` 4e-7 -> `attn_out(8)` 2e-4); the same drift exists between the monolithic fused and per-layer paths. Do not use bf16 for the decode/prefill reduction (`--cluster-allreduce-dtype auto` reduces in f32 since 05466b2); the first runs with bf16 prefill partials produced a wrong early EOS.

Quality probe (5 short reasoning/translation questions, 48 tokens): cluster 3/5 = single node 3/5 (both misses are truncations).

## Measured results, adaptive artifact (2026-09-04)

`DeepSeek-V4-Flash-0731-ROCMFPX-MIX-STRIX.gguf`, DSpark q=4, uniform placement,
`LUCE_MMVF_MAX_NCOLS_F16=4`. Benchmark prompt: 128 tokens at temperature 0,
median of 5, every run byte-identical to the single-node reference `87964cbd…`.
Free prompt: a 60-token technical question, 128 tokens generated. Fidelity is
the exact-copy protocol from `DS4.md` (20 identifiers x 3 repeats).

| Configuration | benchmark tok/s | free prompt tok/s | fidelity | step ms | verify ms |
|---|---|---|---|---|---|
| single node | 34.5 | — | — | 115.4 | — |
| **2 nodes** | **42.3** | **20.6** | 60/60 | 95.0 | 76.2 |
| 4 nodes | 48.2 | 19.7 | 60/60 | 83.8 | 67.0 |

Two things follow. The benchmark prompt keeps a 100 % acceptance rate (32 steps
for 127 tokens at every rank count), so it measures verify cost and scales with
rank count; a free prompt accepts about 2.3 of 4 and is acceptance-bound, where
**two nodes beat four**. And `F + V/N` over the two cluster points gives
F ≈ 58 ms fixed, V ≈ 37 ms shardable, so the asymptote of expert parallelism on
this model is about **53 tok/s** no matter how many nodes are added.

Remaining unsharded work, measured at two nodes: the DSpark drafter (9.3 ms of
the 95.0 ms step). Every rank already loads it and runs the draft forward in
lockstep, discarding its own tokens in favour of rank 0's broadcast, so
splitting its routed experts needs no protocol change -- `ggml_mul_mat_id`
already treats negative ids as masked owner routes. Worth about +1.6 tok/s at
two nodes and +2.4 at four; not implemented, because it changes the drafter's
numerics and the acceptance rate is the more valuable quantity (a 4-rank
reduction on the uniform artifact cost 28 % that way).

## Measured results, path 3b (2026-09-04, same nodes and artifact)

All decode numbers below are the 128-token benchmark prompt at temperature 0,
median of 3 runs, `DeepSeek-V4-Flash-ROCMFP2-STRIX.gguf`, uniform placement.

| Configuration | AR decode tok/s | DSpark q=4 tok/s | Output |
|---|---|---|---|
| single node, monolithic | 16.6 | 27.2 | sha256 `87964cbd…` |
| 2 nodes, path 3a (host all-reduce per layer) | 14.3 | 22.4 | `87964cbd…` |
| **2 nodes, path 3b (in-graph all-reduce, fused graph)** | **21.5** | **29.8** | `87964cbd…` |
| 4 nodes, path 3b | 21.5 | 20.3 | differs (see below) |

Path 3b is worth **1.50x** on AR decode and **1.33x** with DSpark against the
cluster's own path 3a, and it is the first configuration that beats a single
node: 1.29x (AR) and 1.10x (DSpark). Every 2-node run is byte-identical to the
single-node completion. On a free-form 200-token prompt the 2-node cluster
reaches 16.75 tok/s against 12.1 on one node (1.38x).

Where the win comes from, measured over 63 AR decode steps at N=2: path 3a
spent 357 ms in `cluster_allreduce` (three synchronizing copies plus a stream
wait per layer, 132 us x 43 layers x 63 steps) and another 337 ms in the
host-driven router and hybrid evaluator that the fused graph does not need.
Path 3b replaces all of it with one graph compute per step: 33 ms per step
against 70 ms.

**Four nodes do not pay for this model.** AR decode is unchanged at 21.5 tok/s
and DSpark drops to 20.3. Halving the expert work again does not help because
at batch 1 the expert matmuls are launch-bound, while the collective gets more
expensive (4-rank all-reduce p50 125 us against 73 us at 2 ranks, 43 of them
per token) and every rank waits for the slowest. The graph compute per AR step
rises from 33 ms at N=2 to 44 ms at N=4. The 4-node output is coherent and
stable across runs but differs from the 2-node one: a different reduction
partition changes the last bits and flips a greedy near-tie, the same float
chaos documented for free-form prompts below. **Two nodes is the sweet spot
for DeepSeek V4 Flash on this fabric**, and the 50 tok/s target of the original
plan is not reachable by adding Strix Halo nodes.

## Measured results, DSpark on path 3a (2026-09-04, `DFLASH_DS4_SPEC=1 DFLASH_DS4_SPEC_Q=4`)

| Configuration | 128-token benchmark prompt, tok/s (median of 3) | Free-form prompt, 200 tokens, tok/s | Output |
|---|---|---|---|
| single node, monolithic, DSpark | 27.2 | 12.1 | sha256 `87964cbd…` |
| 2 nodes, uniform sharding, DSpark | 22.2 | 11.9 | sha256 `87964cbd…` — **byte-identical to every AR and single-node run above** |

`accept_rate` is 1.00 on the benchmark prompt for both configurations and
0.645 (cluster) against 0.671 (single node) on the free-form prompt, where the
two produce different token streams and are therefore not directly comparable.
`spec_decode_ran` is true in every cluster run.

DSpark is worth 1.55x over the cluster's own AR decode (22.2 against 14.3) and
still 0.82x of a single node with DSpark — the same ratio the AR decode has
(14.3/16.6 = 0.86), so speculation neither gains nor loses relative ground.
What holds the cluster back is unchanged: path 3a waits on the host after every
layer's all-reduce.

The cluster pays one extra price for speculation. Without a fused verify graph,
a q-token verify batch is cut at the learned compressor's boundaries
(`deepseek4_safe_compressor_batch_tokens`, ratios 4 and 128), so a batch that
starts off-boundary runs as two or more forwards — and each forward pays all 43
all-reduces. Measured cost: 133-140 ms per verify step against ~70 ms for one
AR step. Removing it needs the in-graph all-reduce (path 3b) so the fused
verifier can be used, or a batched non-fused verify that crosses a boundary in
one forward. Both belong to WP5/3b.

Client cancel was tested against a running speculative request (streaming curl
killed mid-generation): the head stopped at step 38 and rank 1 stopped at the
same step 38 with the same accept rate, and the next request was served
normally. That is the empty-draft end marker doing its job.

## Measured results, prefill (WP5, 2026-09-04, two nodes)

Prefill keeps path 3a: the fused graph covers q=1 decode and small verify
batches, not 2048-token chunks. Wall time of the prefill phase, `--ds4-prefill
sparse --chunk 2048`:

| Prompt | single node | cluster before WP5 | cluster after WP5 | + `DFLASH_DS4_HYBRID_PREFILL_GPU_HC=1` |
|---|---|---|---|---|
| 1517 tokens | 8.03 s | 34.1 s | 11.45 s | **7.58 s** (1.06x) |
| 12017 tokens | 64.9 s | 283.9 s | 96.9 s | **68.6 s** (0.95x) |

Two independent findings.

**A cluster rank was evaluating its experts one token at a time.** A rank holds
a reduced expert stack, and `mmq_safe_full_batch` is false for such a stack
because MMQ's `mul_mat_id` illegal-accesses on it. The batched evaluator
therefore fell into its sub-batch loop, whose size is
`min(mmq_safe_sub_batch(), prefill limit)` = **1** on gfx1151: 1517 tokens x 43
layers = 65k graph computes, 25.9 s of FFN against 7.6 s for a single node's
whole prefill graph. The fix is the packing the heterogeneous prefill already
uses: with cold owner `None` there is exactly one owner, every surviving route
is resident, and the whole batch is packed by expert into one graph per layer
(`eval_moe_owner_expert_major_batched`). FFN drops to 3.7 s and the output is
unchanged. This is cluster-only: no upstream configuration uses cold owner
`None`.

**Host-side HC is then the largest remaining item** (3.3 s of 11.5 s at 1517
tokens, 36 s of 97 s at 12017). `DFLASH_DS4_HYBRID_PREFILL_GPU_HC=1` moves it
into the graph and brings prefill to parity with a single node. It stays
**opt-in**, exactly as upstream ships it, because HC computed on the GPU
differs from the host implementation in its last bits: with the flag the
greedy completion of the 128-token benchmark changes (sha256 `8d1ec7d8…`
instead of `87964cbd…`), and the byte-identity that proves the sharding exact
is worth more than 30 % of prefill by default. Turn it on for long-prompt
serving, leave it off when comparing against a single node.

The all-reduce is not the prefill bottleneck: 535 ms of 7.6 s at 1517 tokens,
3.5 s of 68.6 s at 12017. At 12k tokens attention (24.7 s) and the expert FFN
(29.2 s) dominate, and attention is replicated on every rank, so long-context
prefill cannot gain much from more ranks.

## Upstream pull requests

Four pieces of this fork are not cluster-specific and were offered back to
`Luce-Org/lucebox` on 2026-09-04. Each is compile-verified against
`upstream/main` on its own branch (`upstream-pr/*`), not just as part of this
fork:

| PR | What | Branch |
|---|---|---|
| [#698](https://github.com/Luce-Org/lucebox/pull/698) | **A real upstream bug**: the cached q=1 output graph never fills the verifier's all-logits hook, so a verify batch split at a compressor boundary comes back empty and speculative decode fails | `upstream-pr/verify-all-logits` |
| [#699](https://github.com/Luce-Org/lucebox/pull/699) | `ggml_backend_cuda_get_stream` / `_get_device_id`, so work submitted outside ggml can be stream-ordered with a backend | `upstream-pr/ggml-backend-stream-accessors` |
| [#700](https://github.com/Luce-Org/lucebox/pull/700) | `ggml_cluster_allreduce`: an in-graph collective node that calls a caller-registered callback, so ggml gains no collective-library dependency | `upstream-pr/ggml-ingraph-collective` |
| [#701](https://github.com/Luce-Org/lucebox/pull/701) | `MoeHybridColdBackend::None` plus the expert-major prefill it unlocks (34.1 s -> 11.45 s on a 1517-token prompt) | `upstream-pr/moe-cold-owner-none` |
| [#702](https://github.com/Luce-Org/lucebox/pull/702) | `LUCE_MMVF_MAX_NCOLS_F16`: gfx1151 inherits the discrete-RDNA3 F16 `mul_mat_vec` ceiling of 3, and a verify width of 4 lands one column past it, where rocBLAS runs a tall-skinny F16 weight in a single workgroup (27.9 -> 34.5 tok/s) | `upstream-pr/mmvf-f16-ceiling` |
| [#703](https://github.com/Luce-Org/lucebox/pull/703) | Register mixed-qtype decode tables for hot-only storage, so an owner with cold owner `None` can run an adaptive artifact. **Stacked on #701** | `upstream-pr/mix-tables-hot-only` |

Deliberately **not** offered: the N-rank expert placement, the control channel,
the decision hooks and the feature-gate rows. They only mean something with a
cluster attached and would be dead weight in a single-node tree.

## Shared-expert sharding (WP8, opt-in)

`--cluster-shared-expert shard` gives each rank `n_ff / N` of the shared
expert's intermediate axis. The down projection contracts over that axis, so a
rank's slice is a **partial sum** — it is added to the routed partial *before*
the reduction and both ride in the same message. No extra collective.

`gate` and `up` are sliced by output row (a contiguous view). `down` is not:
it contracts over the intermediate axis, so slicing it would need a strided
view of every row. The local intermediate is zero-padded back to full width
instead and the full `down` is applied — 2 of the 3 weights saved per rank.

Prefill takes the host-driven path and computes the shared expert replicated
even in this mode; both forms compute the same quantity, and only the fused
decode/verify graph can carry the sliced one.

**Measured, two nodes, on top of head parallelism:**

| | replicated | sharded |
|---|---|---|
| DSpark q=4, 128 tokens | 30.00 tok/s | **30.20 tok/s** |
| AR decode | 21.6 tok/s | **22.1 tok/s** |

Byte-identical to the single-node reference in both. It stays **opt-in**
because it changes the summation order — the shared expert is summed inside
the reduction rather than added once after it — and the default keeps the
order a single node uses. `replicate` remains the default;
`--cluster-shared-expert rank0` is still unimplemented.

## Attention head parallelism (WP8)

Attention used to run in full on every rank. Each rank now builds only its
share of the 64 heads and the ranks all-reduce the result.

Why that works without a second collective type: the DS4 output projection is
grouped, and its second stage contracts over `n_lora_o * n_out_group = 8192`.
A rank that owns half the heads produces a **partial sum over that contraction**,
so the recombination is an all-reduce of `[n_embd, n_tokens]` F32 — the same
shape, type and size as the MoE one, using the same in-graph node. No
all-gather, no new ggml op.

Three weights are sliced, all as plain views: `attn_q_b` by output rows,
`attn_sinks` by head, `attn_output_a` by group. The dense-mix codebook registry
keys a slice by `(pointer - base) / stride`, so a view at a slice boundary
resolves to the right codebook. The second projection stage is **not** sliced:
its weight would need a strided view of every row, so instead this rank's
groups are put back at their own offset with the foreign groups zeroed
(16 KiB per layer at `n_tokens = 1`). The zeros contribute nothing, and summing
the ranks reproduces the full contraction.

The split has to be in whole output groups, because the projection's first
stage is indexed by group. A rank count that does not divide the 8 groups (or
the 64 heads) keeps attention replicated. `DFLASH_CLUSTER_NO_ATTENTION_PARALLEL=1`
does the same.

**Measured, two nodes, same binary, kill-switch off vs on:**

| | replicated attention | head-parallel |
|---|---|---|
| DSpark q=4, 128 tokens | 29.40 tok/s | **30.00 tok/s** (+2.0 %) |
| AR decode, 128 tokens | 21.25 tok/s | **21.6 tok/s** (+1.6 %) |

Byte-identical to the single-node reference in both configurations.

**Why only 2 %, when the arithmetic says 12 %.** The gross saving is real —
half of `attn_q_b` and `attn_output_a`, and half the per-head attention work —
but it is bought with a **second collective per layer**: 86 per forward instead
of 43. At 63 us each that is 2.7 ms added to a ~46 ms step, which eats roughly
two thirds of the gain. Two further things bound it: the latent KV cache cannot
be sharded (MLA has one KV head shared by all 64), so the KV read is unchanged
and its share grows with context; and the second projection stage still reads
its full weight on every rank.

Two environment sweeps were run against this and neither helped:
`NCCL_PROTO=LL` gave 29.6 tok/s (slightly worse) and
`NCCL_IB_QPS_PER_CONNECTION=1` gave 30.0 (unchanged). The 63 us is fixed
overhead — kernel launch, host bounce, RDMA doorbell — not something a protocol
knob moves. At 25 GbE the wire time for 16 KiB is about 1 us.

## Where a decode step actually goes (measured 2026-09-04)

Two diagnostics were added to answer this with numbers instead of reasoning:

* `DFLASH_CLUSTER_ALLREDUCE_NOOP=1` — the in-graph all-reduce returns without
  calling RCCL. **The output is wrong** (each rank keeps its own partial); the
  point is to time a run with and without the collective.
* `DFLASH_CLUSTER_GRAPH_CAPTURE=1` — allow HIP graph capture of a graph that
  contains a cluster collective. Off by default.

**The collectives are not the bottleneck.** AR decode, two nodes, same prompt:
45.5 ms per step with the collective disabled against 48.2 ms with it. That is
**2.7 ms for all 43 collectives, under 6 % of the step** — 63 us each, matching
the 73 us the RCCL selftest measures at 64 KiB. Every idea that makes the
message cheaper (bf16, reduce-scatter/all-gather, RCCL protocol tuning) is
therefore optimizing under 6 % of a decode step, and bf16 additionally risks
the numerics that already produced a wrong early EOS in prefill.

**HIP graph capture works and is bit-exact, and it is slower.** With capture
allowed the completion stays byte-identical to the reference — so RCCL
collectives *can* be captured and replayed on this stack, which the code
previously only called "unverified". But the DS4 fused graph has **5445 nodes**,
`GGML_CUDA_GRAPH_STATS=1` shows 23 replays per 25 forwards (so this is not
capture churn), and throughput drops: AR decode 21.5 -> **19.15 tok/s**, DSpark
29.45 -> **26.65 tok/s**. Replaying a 5.4k-node graph containing collectives
costs more on this runtime than launching it eagerly. The switch stays off and
stays in the tree, because "graphs are disabled, that must be the problem" is
the first thing anyone will think when reading this code.

**What is left is the replicated work.** Only the routed experts are sharded.
Attention, the indexer, HC, embeddings, lm_head and the shared expert run in
full on every rank. On a single node attention is 1326 ms of a 3607 ms decode
(37 %) and the FFN 1311 ms; sharding halves only the second half. That is the
structural ceiling of the current design, and it is why four nodes do not pay.

## Prefix cache across ranks (M4)

Until protocol 2 a cluster run had to pass `--prefix-cache-slots 0`, so every
request paid a full prefill. With `DeepSeek-V4-Flash` that is 8 s for a
1500-token prompt and over a minute for 12k — by far the largest cost in a
chat or agent workload, much larger than anything left in the decode loop.

Two halves had to travel for the snapshots to be replicated, and only one of
them did:

* `snapshot_save` / `snapshot_free` were already broadcast backend ops.
* The **inline snapshot position** was not on the wire (`snap_pos = -1` on the
  worker), so the workers never took the checkpoint the head took. A later
  restore would have resumed from a slot only the head held — silent
  divergence, not an error.
* The restore slot and the inline save slot were the *same* wire field, but
  the HTTP layer can resume from one slot and check-point into another in the
  same request.

Protocol 2 therefore carries `restore_slot`, `kv_offset`, `snapshot_slot` and
`snapshot_pos` as four independent fields, and `--prefix-cache-slots` is
allowed again (`PREFIX_SLOTS` in `launch_cluster.sh`, default 32). Two ranks
that disagree about a save now fail loudly: if the head's own save fails after
the broadcast it frees the slot everywhere, and a worker whose save fails
aborts the run instead of continuing with a slot the head has.

The disk and prefill caches stay off: they adopt deserialized snapshots
through `snapshot_adopt`, which only the head could do.

Measured, three-turn conversation over a 1000-token document, two nodes:

| Turn | prefilled | cache | wall |
|---|---|---|---|
| 1 | 1023 | miss | 10.6 s |
| 2 | 1047 | miss | 12.6 s |
| 3 | **31** | **hit, 1037 restored** | **1.65 s** |

The same conversation with `PREFIX_SLOTS=0` takes 9.42 s for turn 3, so the
cache is worth **5.7x** on a continued conversation — and turn 3's completion
is **identical with and without the cache**, which is the proof that the
restored snapshot is the same KV a cold prefill would have produced. A single
node run of the same conversation hits the cache at the same turn with the
same 1037 restored tokens and returns the same answer.

Note for benchmarking: the 128-token reference benchmark does not restore
(single short prompt, no shared history), so it stays byte-identical either
way. Use `PREFIX_SLOTS=0` when comparing outputs against a single node so no
run can silently resume another run's KV.

## Faults: what happens when a rank dies (M4)

A rank that dies used to be noticed only *between* forwards. Path 3b removed
the last host-side wait for a collective, so a killed worker left the head
blocked inside HIP forever: no error, no HTTP answer, no exit. Measured before
the fix: still hanging 60 s after the kill, threads parked in
`kfd_wait_on_events`.

Both sides now run a watchdog (250 ms poll) that, **while a request is in
flight**, checks the control channel's heartbeats and the communicator's async
error. On a fault it calls `ncclCommAbort` first — that is what makes a
collective already enqueued on the GPU return an error instead of waiting for
a rank that will never arrive — and only then tears down.

Measured chain, worker killed 15 s into a 400-token request:

```
[cluster] head: FATAL (1): rank 1 stopped responding during a request
[deepseek4-cluster] in-graph all-reduce failed: RCCL communicator aborted
[ds4-verify] step_layer_range returned false (n_tokens=2 kv_start=255)
[server] chat DONE ... ok=false ... finish=error error=decode_failed
[cluster] head: exiting so the supervisor restarts all ranks
```

The request returned **~1 s after the kill**. The head then waits 2 s so the
HTTP response is written and exits 3.

**How the failure reaches a client.** The response is HTTP 200 with an empty
completion and `finish_reason: "stop"` — upstream's non-streaming path sends
200 regardless of the backend result, for every backend, and this fork does
not change that global contract. The machine-readable signal is
`usage.timings.cluster.error`:

```json
"cluster": {"size": 2, "complete": false, "per_rank": [],
            "error": "rank 1 stopped responding during a request"}
```

A client that cares about cluster faults checks that field; the server log
line says `finish=error`.

**Recovery.** `RESTART=on-failure:3` on `launch_cluster.sh` puts a podman
restart policy on every rank. Since all ranks exit non-zero on a fault, they
reform the cluster by themselves: the head listens again and the workers retry
the handshake. Verified end to end: after the head restarted and rank 1 was
started again, the cluster was serving after **75 s** and the benchmark was
byte-identical at the unchanged 29.8 tok/s. One caveat worth knowing: podman
does not restart a container the *operator* killed, so a `podman kill` needs a
`podman start`; a real crash is restarted by the policy. The default stays
`RESTART=no`, because during development a crash should stay a crash instead
of turning into a restart loop that reloads 50 GB per rank.

`scripts/cluster/fault_drill.sh "strix1 strix2" <target> <dspark>` runs the
whole drill and fails loudly on each step.

## Observability (WP6)

Three surfaces, all filled on a two-node run:

* **`usage.timings.cluster`** — per-request, per-rank: `compute_ms`,
  `allreduce_calls` / `_bytes` / `_wait_ms`, `ctrl_wait_ms`, `device_bytes`,
  plus `verify_hash` counts when `--cluster-verify-hash` is on and `error`
  when the report gather failed. Present in every response shape, streaming
  included, because it rides on `GenTimings` — the one struct that reaches all
  six places that emit `usage.timings`. Documented in `server/docs/API.md`.
* **`/props.cluster`** — what this cluster is: size, rank, `ifname`, `ib_hca`,
  `gid_index`, placement and its hash, shared-expert mode, all-reduce dtype,
  `ingraph_allreduce` (path 3b), `gpudirect` (always false here) and this
  rank's `resident_expert_bytes`. Constant after bootstrap.
  `{"active": false}` on a single node. Spec: `docs/specs/props-endpoint.md`
  §4.11b.
* **`/status/json.cluster`** and `perf_history[].cluster_size` — size, rank,
  placement and whether path 3b is active. The status *page* renders the JSON
  it already knew about; a per-rank panel in `share/status.html` is not built.

The seam is two virtuals on `ModelBackend` returning plain structs
(`server/src/common/cluster_view.h`), the same shape as `get_routing_stats()`.
`server/src/server/` therefore still contains no cluster include and is
unchanged with `-DDFLASH27B_CLUSTER=OFF`.

Reading a two-node request:

```
per_rank[0] compute_ms 3607.0  allreduce_wait_ms  11.1  ctrl_wait_ms  1.1
per_rank[1] compute_ms 3606.4  allreduce_wait_ms 128.4  ctrl_wait_ms 17.8
```

Equal `compute_ms` means the placement is balanced. Rank 1's larger
`allreduce_wait_ms` and `ctrl_wait_ms` are it waiting for rank 0, which also
samples, drafts and serves HTTP — the expected asymmetry, and the number to
watch if a node ever falls behind.

Two limits worth knowing. `allreduce_wait_ms` is **0 for decode on path 3b**:
the collective runs inside the graph, so its cost is part of `compute_ms` and
not separable; the non-zero values come from prefill, which still takes path
3a. For the same reason the `[deepseek4-timing]` banner's `cluster_allreduce`
field stays 0 on path 3b while `cluster_bytes` is correct.

### M1 artifact and diagnostics
- Diagnostic placement JSON: `python server/scripts/cluster/build_expert_placement.py --dims 43,256,6 -n 2 --all-on-rank 0 -o all_rank0.json`
- Trace run: `EXTRA_ENV="DFLASH_CLUSTER_TRACE=1 DFLASH_CLUSTER_PREFILL_SINGLE_TOKEN=1" scripts/cluster/launch_cluster.sh ... -- --ds4-prefill exact --chunk 1`, then `podman logs lucebox-rank0 | grep cluster-stage`
- Reference: same binary monolithic with `DFLASH_CLUSTER_TRACE=1 --ds4-prefill exact --chunk 1` prints the same lines as rank -1.
