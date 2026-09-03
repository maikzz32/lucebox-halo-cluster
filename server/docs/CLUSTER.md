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

Not used (for the record): `DeepSeek-V4-Flash-ROCMFP2-STRIX.gguf`
(102,320,631,200 bytes, sha256 `8fa6c30d…9208`) and the drafter variant without
`0731`, `DeepSeek-V4-Flash-DSpark-draft-Q4RMFP4-denseF16.gguf`
(11,304,737,056 bytes, sha256 `48883d35…f746`).

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
  --prefix-cache-slots 0 --prefill-cache-slots 0
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
KVFlash, `--prefix-cache-slots 0` (until snapshot broadcast lands in M4).
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

## Measured results, DSpark (2026-09-04, same nodes, artifact and binary, `DFLASH_DS4_SPEC=1 DFLASH_DS4_SPEC_Q=4`)

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

Not yet done: prefill performance (WP5), `usage.timings.cluster` (WP6), 4-node model run, performance work (M2/M3).

### M1 artifact and diagnostics
- Diagnostic placement JSON: `python server/scripts/cluster/build_expert_placement.py --dims 43,256,6 -n 2 --all-on-rank 0 -o all_rank0.json`
- Trace run: `EXTRA_ENV="DFLASH_CLUSTER_TRACE=1 DFLASH_CLUSTER_PREFILL_SINGLE_TOKEN=1" scripts/cluster/launch_cluster.sh ... -- --ds4-prefill exact --chunk 1`, then `podman logs lucebox-rank0 | grep cluster-stage`
- Reference: same binary monolithic with `DFLASH_CLUSTER_TRACE=1 --ds4-prefill exact --chunk 1` prints the same lines as rank -1.
