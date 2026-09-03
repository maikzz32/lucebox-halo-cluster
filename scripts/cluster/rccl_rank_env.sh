#!/usr/bin/env bash
# rccl_rank_env.sh - per-rank launcher used by rccl_baseline.sh under mpirun.
#
# mpirun's -x passes ONE value to every rank, but StrixHalo4 has a different
# NIC (E830-L: enp197s0f1np1 / rocep197s0f1) than nodes 1-3 (E810-C:
# enp197s0f3np3 / rocep197s0f3). This wrapper derives the node id from the
# local 192.168.100.x address and exports the per-node RCCL variables before
# exec'ing the real command. Values already present in the environment win.
set -euo pipefail
RDMA_NET="${RDMA_NET:-192.168.100}"
ip="$(ip -4 -o addr show 2>/dev/null | awk -v n="$RDMA_NET" '$4 ~ "^"n"\\." {sub(/\/.*/,"",$4); print $4; exit}')"
id="${ip##*.}"
if [ "$id" = 4 ]; then iface=enp197s0f1np1; hca=rocep197s0f1; else iface=enp197s0f3np3; hca=rocep197s0f3; fi
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$iface}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$iface}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-$hca}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-1}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-2}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-7}"
export HIP_FORCE_DEV_KERNARG="${HIP_FORCE_DEV_KERNARG:-1}"
[ -n "${RCCL_RANK_ENV_QUIET:-}" ] || echo "[rccl_rank_env] $(hostname) ip=${ip:-?} iface=${NCCL_SOCKET_IFNAME} hca=${NCCL_IB_HCA} gid=${NCCL_IB_GID_INDEX}" >&2
exec "$@"
