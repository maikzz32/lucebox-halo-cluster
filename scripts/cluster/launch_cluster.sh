#!/usr/bin/env bash
# launch_cluster.sh - start / inspect / stop the lucebox-halo-cluster ranks.
#
#   launch_cluster.sh <hosts> <target.gguf> <dspark.gguf> [dflash_server args...]
#   launch_cluster.sh --status <hosts>
#   launch_cluster.sh --down   <hosts>
#
# <hosts> is either a file (one ssh alias per line, '#' comments allowed) or a
# quoted space-separated list ("strix1 strix2"). The FIRST host becomes rank 0
# (head, serves HTTP on :8016); the others become workers, in order.
#
# Every rank runs in a podman container named lucebox-rank<N> on its host,
# started over ssh with the device/network/ulimit flags that are proven with
# RCCL on this cluster (/home/maik/cluster_tpn.sh). Containers this script did
# not start are never touched: --down only removes lucebox-rank*.
#
# GPU memory: the nodes normally run a vLLM Ray cluster (~100 GB GPU memory in
# use). Stop it before launching; this script refuses to kill it for you.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  launch_cluster.sh <hosts-file|"h1 h2 ..."> <target.gguf> <dspark.gguf> [-- dflash_server args...]
  launch_cluster.sh --status <hosts>
  launch_cluster.sh --down   <hosts>
  launch_cluster.sh --help

Model files are names (or relative paths) inside MODELS_DIR on every node,
bind-mounted read-only at /models in the containers.

Environment:
  IMAGE          container image   (default ghcr.io/maikzz32/lucebox-halo-cluster:dev)
  MODELS_DIR     GGUF dir on nodes (default /home/maik/gguf/ds4)
  RDMA_NET       RoCE /24 prefix   (default 192.168.100)
  HEAD_PORT      control channel   (default 9400)
  HTTP_PORT      rank-0 HTTP port  (default 8016)
  GID_INDEX      RoCE v2 GID index (default 1; check with host_prep.sh)
  PLACEMENT      --cluster-expert-placement (default uniform)
  HEAD_WAIT      seconds to wait for the head's "cluster: listening" line (default 20)
  SPEC_Q         DFLASH_DS4_SPEC_Q (default 4); SPEC=0 disables DSpark entirely
  MMVF_F16       LUCE_MMVF_MAX_NCOLS_F16 (default 4; 0 = upstream per-arch value)
  PINNED_ROLLBACK DFLASH_DS4_PINNED_ROLLBACK (default 1; 0 = pageable staging)
  FUSED_CACHE_SLOTS DFLASH_DS4_TP_FUSED_CACHE_SLOTS (default 12 = the hard cap; upstream is 2 at q<=4)
  SHARD_DRAFTER  DFLASH_CLUSTER_SHARD_DRAFTER (default 1; 0 = drafter replicated)
  SPLIT_LM_HEAD  DFLASH_CLUSTER_SPLIT_LM_HEAD (default 1; 0 = head replicated)
  COMP_PAD_STRIDE DFLASH_DS4_COMP_PAD_STRIDE (default 128; upstream is 16)
  RESTART        podman --restart policy (default no). "on-failure:3" makes the
                 ranks reform the cluster after a peer failure by themselves.
  PREFIX_SLOTS   --prefix-cache-slots on every rank (default 32; 0 disables)
  PULL           1 = podman pull IMAGE on every host first (default 0)
  BIN_DIR        host dir with server/build/dflash_server (+ deps/**/lib*.so*);
                 mounted at /opt/lucebox-dist and used as entrypoint (dev loop
                 with IMAGE=localhost/lucebox-build:rocm10; default: unset)
  SKIP_MODEL_CHECK 1 = do not require the GGUFs on every node (selftest)
  DRY_RUN        1 = print the podman commands instead of running them
  IFACE_<N>/HCA_<N>  override interface/HCA for node N (default: node 4 ->
                 enp197s0f1np1/rocep197s0f1, others enp197s0f3np3/rocep197s0f3)

Per-rank flags passed to dflash_server (from server/src/cluster/cluster_config.h):
  --cluster-rank R --cluster-size N --cluster-head <head-ip>:HEAD_PORT
  --cluster-ifname <iface> --cluster-ib-hca <hca> --cluster-gid-index GID_INDEX
  --cluster-expert-placement PLACEMENT --target-device hip:0
  --ds4-expert-top-k 6 --ds4-prefill sparse --chunk 2048 --max-ctx 32768
  --prefix-cache-slots PREFIX_SLOTS --prefill-cache-slots 0
  plus --host 0.0.0.0 --port HTTP_PORT on rank 0 only.
Extra arguments after the three positionals are appended on EVERY rank.

Env inside every container: DFLASH_DS4_SPEC=1 DFLASH_DS4_DRAFT=/models/<dspark>
  DFLASH_DS4_DRAFT_GPU=0 DFLASH_DS4_SPEC_Q=SPEC_Q DFLASH_DS4_FUSED_VERIFY=1
  (the drafter is only *used* on rank 0; workers get the same env so the
  AR-vs-spec routing decision is identical everywhere).
USAGE
}

IMAGE="${IMAGE:-ghcr.io/maikzz32/lucebox-halo-cluster:dev}"
MODELS_DIR="${MODELS_DIR:-/home/maik/gguf/ds4}"
RDMA_NET="${RDMA_NET:-192.168.100}"
HEAD_PORT="${HEAD_PORT:-9400}"
HTTP_PORT="${HTTP_PORT:-8016}"
GID_INDEX="${GID_INDEX:-1}"
PLACEMENT="${PLACEMENT:-uniform}"
HEAD_WAIT="${HEAD_WAIT:-20}"
SPEC="${SPEC:-1}"
# F16 mul_mat_vec ceiling. Upstream gives gfx1151 the RDNA3 value of 3, measured
# on discrete RX 7000 cards; at a verify width of 4 that sends DeepSeek V4's
# per-layer F16 router gate to rocBLAS, which runs it in a single workgroup.
# Two nodes, q=4: 30.5 -> 38.6 tok/s, byte-identical. 0 restores the upstream
# per-architecture value.
MMVF_F16="${MMVF_F16:-4}"
# Stage the speculative rollback rows through pinned host memory. The copy is
# on the critical path between verify and the next draft, and pageable memory
# makes it a synchronous staging copy: 4.9 -> 0.5 ms per step, 42.9 -> 44.9
# tok/s on two nodes, output byte-identical. 0 restores pageable staging.
PINNED_ROLLBACK="${PINNED_ROLLBACK:-1}"
# Slots in the fused verify graph cache. The upstream default of 2 for q<=4
# misses on every step: 5.4 ms of pure graph construction per step, 15.0 ms on
# a free prompt where the adaptive width produces more shapes. Four slots
# already pin the benchmark prompt, and a free prompt keeps improving up to
# Ds4FusedVerifyCache::kSlotCount, which is 12 and is a hard cap -- a larger
# value here is silently clamped to it, so 12 is the honest maximum. The slots
# are nearly free: the cache measured 303 MiB at 2 slots and 351 MiB at the
# cap, about 0.8 MiB each. Proposed upstream as #704.
FUSED_CACHE_SLOTS="${FUSED_CACHE_SLOTS:-12}"
# Split the DSpark drafter's routed experts across the ranks. Every rank
# already loads the drafter and runs the draft forward in lockstep, so this
# needs no protocol change. Two nodes 42.1 -> 43.1 tok/s, four 48.3 -> 49.4,
# byte-identical and with the acceptance rate unchanged.
SHARD_DRAFTER="${SHARD_DRAFTER:-1}"
# Project only this rank's slice of the DSpark head's vocabulary and sum the
# padded logits back to full width before the argmax. Requires the vocabulary
# to divide evenly by the rank count; the runtime refuses and stays replicated
# when it does not. Two nodes 48.2 -> 49.4 tok/s, four 55.0 -> 56.3.
SPLIT_LM_HEAD="${SPLIT_LM_HEAD:-1}"
# Rounding granularity for the compressor row count. It enters the fused
# verify graph's shape key, so a fine stride makes the key wander as the
# sequence grows and the graph cache misses -- 11.2 ms of pure rebuild per
# step on a free-form prompt. Padded rows are masked to -1e30 and underflow to
# zero in the softmax, so a coarser stride is bit-identical by construction;
# it only reads a few more masked rows. Free prompt 22.1 -> 25.2 tok/s,
# rebuild 11.2 -> 3.6 ms. Accepts 16, 32, 64, 128.
COMP_PAD_STRIDE="${COMP_PAD_STRIDE:-128}"
# Container restart policy. Every rank exits non-zero on a cluster fault (the
# head with code 3), so "on-failure:N" lets the ranks reform the cluster by
# themselves: the workers retry the handshake and the head waits for them.
# Default "no" because during development a crash should stay a crash - a
# restart loop reloading 50 GB per rank hides the reason it happened.
RESTART="${RESTART:-no}"
# Prefix-cache slots. Replicated across ranks since protocol 2 (snapshot save
# and free are broadcast, the request carries the restore and the inline slot).
# 0 disables it, which is what a byte-identity comparison against a single node
# should use so no run silently resumes another run's KV.
PREFIX_SLOTS="${PREFIX_SLOTS:-32}"
SPEC_Q="${SPEC_Q:-4}"
PULL="${PULL:-0}"
DRY_RUN="${DRY_RUN:-0}"
# Dev loop without a release image: BIN_DIR is a host directory holding the
# build tree (e.g. ~/lucebox-cluster-dist with server/build/dflash_server and
# server/build/deps/**/lib*.so*, rpath is $ORIGIN-relative). It is mounted at
# /opt/lucebox-dist and the container entrypoint is overridden to that binary,
# so IMAGE only needs the ROCm runtime (e.g. localhost/lucebox-build:rocm10).
BIN_DIR="${BIN_DIR:-}"
# SKIP_MODEL_CHECK=1 skips the preflight test for the GGUF files (for
# --cluster-selftest runs before the models are synced to every node).
SKIP_MODEL_CHECK="${SKIP_MODEL_CHECK:-0}"
# Extra container environment, space-separated KEY=VALUE pairs, e.g.
# EXTRA_ENV="NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET" for transport debugging.
EXTRA_ENV="${EXTRA_ENV:-}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10)

MODE=up
case "${1:-}" in
    -h|--help|"") usage; [ -n "${1:-}" ] && exit 0; exit 2 ;;
    --down)   MODE=down;   shift ;;
    --status) MODE=status; shift ;;
esac

[ $# -ge 1 ] || { usage >&2; exit 2; }
HOSTS_ARG="$1"; shift

# ── host list ───────────────────────────────────────────────────────────────
HOSTS=()
if [ -f "$HOSTS_ARG" ]; then
    while IFS= read -r line; do
        line="${line%%#*}"; line="${line//[[:space:]]/}"
        [ -n "$line" ] && HOSTS+=("$line")
    done <"$HOSTS_ARG"
else
    read -r -a HOSTS <<<"$HOSTS_ARG"
fi
N="${#HOSTS[@]}"
[ "$N" -ge 1 ] || { echo "ERROR: empty host list" >&2; exit 2; }

run_ssh() { # run_ssh <host> <command string>
    local h="$1"; shift
    if [ "$DRY_RUN" = 1 ]; then echo "ssh $h $*"; return 0; fi
    ssh "${SSH_OPTS[@]}" "$h" "$@"
}

# RDMA IP of a host: strixN alias -> RDMA_NET.N, otherwise ask the host.
rdma_ip() {
    local h="$1" n
    if [[ "$h" =~ ^strix([0-9]+)$ ]]; then echo "${RDMA_NET}.${BASH_REMATCH[1]}"; return; fi
    n="$(ssh "${SSH_OPTS[@]}" "$h" "ip -4 -o addr show | awk -v n='$RDMA_NET' '\$4 ~ \"^\"n\"\\\\.\" {sub(/\\/.*/,\"\",\$4); print \$4; exit}'")"
    [ -n "$n" ] || { echo "ERROR: $h has no ${RDMA_NET}.x address" >&2; exit 1; }
    echo "$n"
}
node_iface() { local id="$1" v; v="IFACE_${id}"; if [ -n "${!v:-}" ]; then echo "${!v}"; elif [ "$id" = 4 ]; then echo enp197s0f1np1; else echo enp197s0f3np3; fi; }
node_hca()   { local id="$1" v; v="HCA_${id}";   if [ -n "${!v:-}" ]; then echo "${!v}"; elif [ "$id" = 4 ]; then echo rocep197s0f1; else echo rocep197s0f3; fi; }

# ── --down / --status ───────────────────────────────────────────────────────
if [ "$MODE" = down ]; then
    for h in "${HOSTS[@]}"; do
        echo "== $h: removing lucebox-rank* containers"
        run_ssh "$h" 'ids=$(podman ps -a --filter "name=^lucebox-rank" --format "{{.Names}}"); if [ -n "$ids" ]; then echo "$ids" | xargs -r podman rm -f -t 10; else echo "  (none)"; fi' || echo "  ssh to $h failed"
    done
    exit 0
fi
if [ "$MODE" = status ]; then
    for h in "${HOSTS[@]}"; do
        echo "== $h"
        run_ssh "$h" 'podman ps -a --filter "name=^lucebox-rank" --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"; for c in $(podman ps -a --filter "name=^lucebox-rank" --format "{{.Names}}"); do echo "-- $c (last 15 log lines)"; podman logs --tail 15 "$c" 2>&1 | sed "s/^/   /"; done' || echo "  ssh to $h failed"
    done
    exit 0
fi

# ── up ──────────────────────────────────────────────────────────────────────
[ $# -ge 2 ] || { echo "ERROR: need <target.gguf> <dspark.gguf>" >&2; usage >&2; exit 2; }
TARGET="$1"; DSPARK="$2"; shift 2
[ "${1:-}" = "--" ] && shift
EXTRA_ARGS=("$@")
if [ "$N" -lt 2 ]; then
    echo "ERROR: a cluster needs at least 2 hosts (kClusterMinSize); use plain dflash_server for one node" >&2; exit 2
fi
if [ "$N" -gt 8 ]; then echo "ERROR: at most 8 ranks (kClusterMaxSize)" >&2; exit 2; fi

HEAD_IP="$(rdma_ip "${HOSTS[0]}")"
echo "== cluster: ${N} ranks, head ${HOSTS[0]} (${HEAD_IP}:${HEAD_PORT}), image ${IMAGE}"
echo "   target=/models/${TARGET}  dspark=/models/${DSPARK}  models=${MODELS_DIR}"

# /etc/hosts lines for the whole RDMA net (no DNS there).
ADD_HOSTS=()
for n in 1 2 3 4; do ADD_HOSTS+=(--add-host "StrixHalo${n}:${RDMA_NET}.${n}"); done

# Pre-flight on every host: image, model files, no stale rank container,
# warn about foreign GPU users. Never stops anything.
for i in "${!HOSTS[@]}"; do
    h="${HOSTS[$i]}"
    echo "-- preflight $h (rank $i)"
    if [ "$PULL" = 1 ]; then run_ssh "$h" "podman pull '$IMAGE'"; fi
    run_ssh "$h" "
        set -e
        if [ '${SKIP_MODEL_CHECK}' != 1 ]; then
            test -f '${MODELS_DIR}/${TARGET}' || { echo 'ERROR: ${MODELS_DIR}/${TARGET} missing on $h'; exit 1; }
            test -f '${MODELS_DIR}/${DSPARK}' || { echo 'ERROR: ${MODELS_DIR}/${DSPARK} missing on $h'; exit 1; }
        fi
        if [ -n '${BIN_DIR}' ]; then
            test -x '${BIN_DIR}/server/build/dflash_server' || { echo 'ERROR: ${BIN_DIR}/server/build/dflash_server missing on $h'; exit 1; }
        fi
        podman image exists '${IMAGE}' || { echo 'ERROR: image ${IMAGE} not present on $h (PULL=1 or build it)'; exit 1; }
        if podman ps -a --format '{{.Names}}' | grep -q '^lucebox-rank${i}\$'; then
            echo 'ERROR: lucebox-rank${i} already exists on $h; run --down first'; exit 1; fi
        others=\$(podman ps --format '{{.Names}}' | grep -iE 'vllm|ray' || true)
        [ -z \"\$others\" ] || echo \"WARN: GPU-holding containers running on $h: \$others (vLLM cluster occupies ~100 GB GPU memory; stop it yourself)\"
        test -e /dev/infiniband || { echo 'ERROR: /dev/infiniband missing on $h'; exit 1; }
    " || { echo "preflight failed on $h" >&2; exit 1; }
done

# Compose the podman command for rank <i> on host <h>.
podman_cmd() {
    local i="$1" h="$2" ip id iface hca
    ip="$(rdma_ip "$h")"; id="${ip##*.}"
    iface="$(node_iface "$id")"; hca="$(node_hca "$id")"
    local cmd=(podman run -d --name "lucebox-rank${i}" --hostname "lucebox-rank${i}"
        --restart "$RESTART"
        --device /dev/kfd --device /dev/dri --device /dev/infiniband
        --group-add keep-groups
        --security-opt seccomp=unconfined --security-opt label=disable
        --network host --ipc host --pids-limit -1 --ulimit memlock=-1:-1
        "${ADD_HOSTS[@]}"
        -v "${MODELS_DIR}:/models:ro"
        -e "DFLASH_DS4_SPEC=${SPEC}" -e "DFLASH_DS4_DRAFT=/models/${DSPARK}"
        -e DFLASH_DS4_DRAFT_GPU=0 -e "DFLASH_DS4_SPEC_Q=${SPEC_Q}" -e DFLASH_DS4_FUSED_VERIFY=1
        -e NCCL_NET_GDR_LEVEL=0 -e NCCL_IB_DISABLE=0 -e "NCCL_IB_GID_INDEX=${GID_INDEX}"
        -e "NCCL_SOCKET_IFNAME=${iface}" -e "GLOO_SOCKET_IFNAME=${iface}" -e "NCCL_IB_HCA=${hca}"
        -e NCCL_ASYNC_ERROR_HANDLING=1 -e NCCL_IB_QPS_PER_CONNECTION=2
        -e NCCL_IB_TIMEOUT=22 -e NCCL_IB_RETRY_CNT=7 -e HIP_FORCE_DEV_KERNARG=1
        -e "LUCE_MMVF_MAX_NCOLS_F16=${MMVF_F16}"
        -e "DFLASH_DS4_PINNED_ROLLBACK=${PINNED_ROLLBACK}"
        -e "DFLASH_DS4_TP_FUSED_CACHE_SLOTS=${FUSED_CACHE_SLOTS}"
        -e "DFLASH_CLUSTER_SHARD_DRAFTER=${SHARD_DRAFTER}"
        -e "DFLASH_CLUSTER_SPLIT_LM_HEAD=${SPLIT_LM_HEAD}"
        -e "DFLASH_DS4_COMP_PAD_STRIDE=${COMP_PAD_STRIDE}")
    local kv
    for kv in $EXTRA_ENV; do cmd+=(-e "$kv"); done
    if [ -n "$BIN_DIR" ]; then
        cmd+=(-v "${BIN_DIR}:/opt/lucebox-dist:ro"
              --entrypoint /opt/lucebox-dist/server/build/dflash_server)
    fi
    cmd+=("$IMAGE"
        "/models/${TARGET}"
        --cluster-rank "$i" --cluster-size "$N" --cluster-head "${HEAD_IP}:${HEAD_PORT}"
        --cluster-ifname "$iface" --cluster-ib-hca "$hca" --cluster-gid-index "$GID_INDEX"
        --cluster-expert-placement "$PLACEMENT"
        --target-device hip:0 --ds4-expert-top-k 6 --ds4-prefill sparse
        --chunk 2048 --max-ctx 32768
        --prefix-cache-slots "$PREFIX_SLOTS" --prefill-cache-slots 0)
    if [ "$i" = 0 ]; then cmd+=(--host 0.0.0.0 --port "$HTTP_PORT"); fi
    cmd+=("${EXTRA_ARGS[@]}")
    printf '%q ' "${cmd[@]}"
}

# Head first.
echo "-- starting rank 0 on ${HOSTS[0]}"
run_ssh "${HOSTS[0]}" "$(podman_cmd 0 "${HOSTS[0]}")"
if [ "$DRY_RUN" != 1 ]; then
    echo "   waiting up to ${HEAD_WAIT}s for 'cluster: listening' ..."
    t=0; ready=0
    while [ "$t" -lt "$HEAD_WAIT" ]; do
        if ! run_ssh "${HOSTS[0]}" "podman ps --format '{{.Names}}' | grep -q '^lucebox-rank0\$'"; then
            echo "ERROR: lucebox-rank0 exited early; last log lines:" >&2
            run_ssh "${HOSTS[0]}" "podman logs --tail 40 lucebox-rank0" >&2 || true
            exit 1
        fi
        if run_ssh "${HOSTS[0]}" "podman logs lucebox-rank0 2>&1 | grep -q 'cluster: listening'"; then ready=1; break; fi
        sleep 1; t=$((t + 1))
    done
    if [ "$ready" = 1 ]; then echo "   head is listening (after ${t}s)"; else
        echo "   WARN: no 'cluster: listening' line after ${HEAD_WAIT}s; starting workers anyway (they retry the connect for 5 min)"; fi
fi

# Workers.
for i in $(seq 1 $((N - 1))); do
    h="${HOSTS[$i]}"
    echo "-- starting rank $i on $h"
    run_ssh "$h" "$(podman_cmd "$i" "$h")"
done

cat <<EOF

== started ${N} ranks. Useful next steps:
   $0 --status "${HOSTS[*]}"
   ssh ${HOSTS[0]} podman logs -f lucebox-rank0
   curl -s http://${HEAD_IP}:${HTTP_PORT}/props | jq .cluster
   curl -s http://${HEAD_IP}:${HTTP_PORT}/v1/chat/completions -d '{"model":"dflash","messages":[{"role":"user","content":"Say hi"}],"max_tokens":64,"temperature":0}' | jq -r .choices[0].message.content
   $0 --down "${HOSTS[*]}"
EOF
