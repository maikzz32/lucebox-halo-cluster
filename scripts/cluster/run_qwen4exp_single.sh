#!/usr/bin/env bash
# run_qwen4exp_single.sh - one qwen4exp server on one node, for bring-up.
#
# The cluster launcher is DeepSeek-shaped: it always passes a DSpark drafter,
# the expert-placement flags and the RCCL environment. qwen4exp has none of
# that yet, so bring-up needs its own entry point rather than a growing pile of
# opt-outs in the shared script.
#
#   run_qwen4exp_single.sh <host> [dflash_server args...]
#   run_qwen4exp_single.sh --down <host>
#
# Environment:
#   IMAGE      container image (default localhost/lucebox-build:rocm10)
#   BIN_DIR    host dir with server/build/dflash_server (default
#              /home/maik/lucebox-cluster-dist)
#   MODELS_DIR GGUF dir on the node (default /home/maik/gguf/qwen4exp)
#   MODEL      first shard's file name
#   HTTP_PORT  default 8017, so a DeepSeek cluster on 8016 can stay up
#   RMS        1 = DFLASH_QWEN4EXP_RMS=1 (per-sublayer magnitude probe)
#   EXTRA_ENV  space-separated KEY=VALUE passed through with -e
set -euo pipefail

IMAGE=${IMAGE:-localhost/lucebox-build:rocm10}
BIN_DIR=${BIN_DIR:-/home/maik/lucebox-cluster-dist}
MODELS_DIR=${MODELS_DIR:-/home/maik/gguf/qwen4exp}
MODEL=${MODEL:-Qwen3.8-Flash-Next-Uncensored-Q4_0-ROCmFP4-STRIX_LEAN-00001-of-00003.gguf}
HTTP_PORT=${HTTP_PORT:-8017}
RMS=${RMS:-0}
EXTRA_ENV=${EXTRA_ENV:-}
NAME=lucebox-q4e

if [ "${1:-}" = "--down" ]; then
    ssh "$2" "podman rm -f $NAME >/dev/null 2>&1 || true"
    echo "down"
    exit 0
fi

HOST=$1; shift

# Always remove first: podman run --name fails on an exited container of the
# same name, and the failure looks like a start that produced no logs.
ssh "$HOST" "podman rm -f $NAME >/dev/null 2>&1 || true"

cmd="podman run -d --name $NAME --hostname $NAME"
cmd="$cmd --device /dev/kfd --device /dev/dri --group-add keep-groups"
cmd="$cmd --security-opt seccomp=unconfined --security-opt label=disable"
cmd="$cmd --network host --ipc host --pids-limit -1 --ulimit memlock=-1:-1"
cmd="$cmd -v ${MODELS_DIR}:/models:ro -v ${BIN_DIR}:/opt/lucebox-dist:ro"
cmd="$cmd -e HIP_FORCE_DEV_KERNARG=1 -e LUCE_MMVF_MAX_NCOLS_F16=4"
cmd="$cmd -e DFLASH_QWEN4EXP_RMS=${RMS}"
for kv in $EXTRA_ENV; do cmd="$cmd -e $kv"; done
cmd="$cmd --entrypoint /opt/lucebox-dist/server/build/dflash_server $IMAGE"
cmd="$cmd /models/${MODEL} --host 0.0.0.0 --port ${HTTP_PORT} $*"

ssh "$HOST" "$cmd"
echo "started $NAME on $HOST, http://${HOST}:${HTTP_PORT}"
