#!/usr/bin/env bash
set -euo pipefail

# Reproducible 2-node qualification for the lucebox-halo-cluster AR path (M1).
# Mirrors qualify_ds4_q5_amd.sh: one server process per node, explicit
# environment, manifest, warm-up, then N measured runs through
# server/scripts/cluster/bench_ds4_cluster.py, which enforces byte-identical
# output across runs and against the checked-in golden sha256.
#
# The head runs on this machine; the worker is started over SSH with the
# identical binary path, model path and environment. Both ranks must see the
# same GGUF (same bytes; the Hello handshake compares path + size).
#
# Golden files live in server/tests/cluster-golden/<name>.sha256 and are
# produced by a SINGLE-NODE run of the same binary (see README there); the
# cluster is only correct when it reproduces them.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECKOUT="${CHECKOUT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$CHECKOUT/server/build-cluster}"
SERVER_BIN="${SERVER_BIN:-$BUILD_DIR/dflash_server}"
BENCH_CLIENT="${BENCH_CLIENT:-$CHECKOUT/server/scripts/cluster/bench_ds4_cluster.py}"
GOLDEN_DIR="${GOLDEN_DIR:-$CHECKOUT/server/tests/cluster-golden}"

# Model artifacts (nodes: /home/maik/gguf/ds4, containers: /models).
MODELS_DIR="${MODELS_DIR:-/home/maik/gguf/ds4}"
TARGET_MODEL="${TARGET_MODEL:-$MODELS_DIR/DeepSeek-V4-Flash-0731-ROCMFPX-MIX-STRIX.gguf}"
TARGET_MODEL_SHA256="${TARGET_MODEL_SHA256:-7c0789d190fdd2acad93255825822ca276f29d13f9410f2ac65f5f7a542b0a38}"
TARGET_MODEL_BYTES="${TARGET_MODEL_BYTES:-98294917184}"
# DSpark drafter: not used by the M1 AR qualification, recorded for the manifest
# and for the WP4 (M2) variant of this script.
DRAFT_MODEL="${DRAFT_MODEL:-$MODELS_DIR/DeepSeek-V4-Flash-0731-DSpark-draft-Q4RMFP4-denseF16.gguf}"

# Cluster topology.
HEAD_HOST="${HEAD_HOST:-192.168.100.1}"        # control endpoint + HTTP bind
HEAD_PORT="${HEAD_PORT:-9400}"                 # --cluster-head port
HTTP_PORT="${HTTP_PORT:-8016}"
WORKER_SSH="${WORKER_SSH:?set WORKER_SSH to user@host of the rank-1 node}"
WORKER_CHECKOUT="${WORKER_CHECKOUT:-$CHECKOUT}"
WORKER_SERVER_BIN="${WORKER_SERVER_BIN:-${SERVER_BIN/#$CHECKOUT/$WORKER_CHECKOUT}}"
CLUSTER_SIZE="${CLUSTER_SIZE:-2}"
CLUSTER_IFNAME="${CLUSTER_IFNAME:?set CLUSTER_IFNAME to the RoCE interface (e.g. enp1s0)}"
CLUSTER_IB_HCA="${CLUSTER_IB_HCA:?set CLUSTER_IB_HCA to the HCA (e.g. rocep1s0)}"
CLUSTER_GID_INDEX="${CLUSTER_GID_INDEX:?set CLUSTER_GID_INDEX (see show_gids)}"
PLACEMENT="${PLACEMENT:-uniform}"
VERIFY_HASH="${VERIFY_HASH:-8}"                # --cluster-verify-hash; 0 = off
CLUSTER_TIMEOUT_MS="${CLUSTER_TIMEOUT_MS:-30000}"

# Run shape.
MAX_CTX="${MAX_CTX:-32768}"
EXPERT_TOP_K="${EXPERT_TOP_K:-6}"
WARMUP="${WARMUP:-1}"
RUNS="${RUNS:-3}"
MAX_TOKENS="${MAX_TOKENS:-512}"
GOLDEN_NAME="${GOLDEN_NAME:-beta-512-k${EXPERT_TOP_K}}"
PROMPT_FILE="${PROMPT_FILE:-}"                 # empty = bench script's BETA prompt
EXPECTED_SHA256="${EXPECTED_SHA256:-}"         # overrides the golden file
HASH_MODELS="${HASH_MODELS:-0}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-1800}"
RUN_ID="${RUN_ID:-}"
OUT_ROOT="${OUT_ROOT:-$CHECKOUT/results/ds4_cluster_qualification}"

for executable in "$SERVER_BIN"; do
    if [[ ! -f "$executable" || ! -x "$executable" ]]; then
        echo "required executable is missing or not executable: $executable" >&2
        exit 2
    fi
done
for input in "$TARGET_MODEL" "$BENCH_CLIENT"; do
    if [[ ! -f "$input" || ! -r "$input" ]]; then
        echo "required input is missing or unreadable: $input" >&2
        exit 2
    fi
done
if [[ -n "$PROMPT_FILE" && ( ! -f "$PROMPT_FILE" || ! -r "$PROMPT_FILE" ) ]]; then
    echo "prompt file is missing or unreadable: $PROMPT_FILE" >&2
    exit 2
fi
for numeric_setting in HEAD_PORT HTTP_PORT CLUSTER_SIZE CLUSTER_GID_INDEX VERIFY_HASH \
    CLUSTER_TIMEOUT_MS MAX_CTX EXPERT_TOP_K WARMUP RUNS MAX_TOKENS READY_TIMEOUT_S; do
    numeric_value="${!numeric_setting}"
    if [[ ! "$numeric_value" =~ ^(0|[1-9][0-9]{0,8})$ ]]; then
        echo "$numeric_setting must be a non-negative decimal integer with at most 9 digits" >&2
        exit 2
    fi
done
if (( HEAD_PORT < 1 || HEAD_PORT > 65535 || HTTP_PORT < 1 || HTTP_PORT > 65535 ||
      CLUSTER_SIZE < 2 || CLUSTER_SIZE > 8 || RUNS < 1 || MAX_TOKENS < 1 || MAX_CTX < 1 ||
      EXPERT_TOP_K < 1 || EXPERT_TOP_K > 6 || CLUSTER_TIMEOUT_MS < 1000 )); then
    echo "port/size/run parameters out of range" >&2
    exit 2
fi
if (( CLUSTER_SIZE != 2 )); then
    echo "this script starts exactly one worker over SSH; CLUSTER_SIZE must be 2 (use launch_cluster.sh for N>2)" >&2
    exit 2
fi
if (( MAX_TOKENS + 256 > MAX_CTX )); then
    echo "MAX_TOKENS plus prompt headroom exceeds MAX_CTX=$MAX_CTX" >&2
    exit 2
fi
case "$PLACEMENT" in
    uniform|balanced|*.json) ;;
    *) echo "PLACEMENT must be uniform, balanced or a .json file" >&2; exit 2 ;;
esac
case "$HASH_MODELS" in
    0|1) ;;
    *) echo "HASH_MODELS must be 0 or 1" >&2; exit 2 ;;
esac

# Golden sha256: explicit env wins, then the checked-in file.
GOLDEN_FILE="$GOLDEN_DIR/$GOLDEN_NAME.sha256"
if [[ -z "$EXPECTED_SHA256" ]]; then
    if [[ ! -r "$GOLDEN_FILE" ]]; then
        echo "no golden sha256: set EXPECTED_SHA256 or create $GOLDEN_FILE from a single-node run" >&2
        exit 2
    fi
    EXPECTED_SHA256="$(awk 'NF { print $1; exit }' "$GOLDEN_FILE")"
fi
if [[ ! "$EXPECTED_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "EXPECTED_SHA256 must contain exactly 64 lowercase hexadecimal characters" >&2
    exit 2
fi

actual_bytes="$(stat -c '%s' -- "$TARGET_MODEL")"
if [[ "$actual_bytes" != "$TARGET_MODEL_BYTES" ]]; then
    echo "target model size $actual_bytes != expected $TARGET_MODEL_BYTES bytes: $TARGET_MODEL" >&2
    exit 2
fi

if [[ -z "$RUN_ID" ]]; then
    RUN_ID="ds4-cluster-n${CLUSTER_SIZE}-ar-k${EXPERT_TOP_K}-${PLACEMENT##*/}-vh${VERIFY_HASH}-$(date -u +%Y%m%dT%H%M%SZ)"
fi
case "$RUN_ID" in
    .|..|*[!A-Za-z0-9._-]*)
        echo "RUN_ID may contain only letters, numbers, dot, underscore, and hyphen" >&2
        exit 2
        ;;
esac
OUT_DIR="$OUT_ROOT/$RUN_ID"
HEAD_LOG="$OUT_DIR/head.log"
WORKER_LOG="$OUT_DIR/worker-rank1.log"

for required_command in python3 sha256sum ssh stat; do
    if ! command -v "$required_command" >/dev/null 2>&1; then
        echo "required command is unavailable: $required_command" >&2
        exit 2
    fi
done

if pgrep -f '(^|/)dflash_server([[:space:]]|$)' >/dev/null 2>&1; then
    echo "another dflash_server is running on this node; stop it first" >&2
    exit 2
fi

mkdir -p "$OUT_ROOT"
if ! mkdir "$OUT_DIR"; then
    echo "refusing to reuse existing output directory: $OUT_DIR" >&2
    exit 2
fi

# Identical environment on both ranks. Kept minimal on purpose: the cluster
# AR path is the qualification subject, not the single-node tuning knobs
# (DFLASH_DS4_SPEC stays unset -> AR; M2 adds the DSpark variant).
server_env=(
    "HOME=$HOME"
    "USER=${USER:-unknown}"
    "PATH=$PATH"
    "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
    "DFLASH_DS4_TIMING=1"
    "DFLASH_DS4_TOPK=$EXPERT_TOP_K"
    "DFLASH_DS4_FUSED_HYBRID_DECODE=1"
    "NCCL_ALGO=Ring"
    "NCCL_PROTO=Simple"
    "NCCL_NET_GDR_LEVEL=0"
    "NCCL_SOCKET_IFNAME=$CLUSTER_IFNAME"
    "NCCL_IB_HCA=$CLUSTER_IB_HCA"
    "NCCL_IB_GID_INDEX=$CLUSTER_GID_INDEX"
)
for pass_var in HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES DFLASH_CLUSTER_TRACE \
    DFLASH_CLUSTER_NO_INGRAPH_ALLREDUCE DFLASH_CLUSTER_NO_GRAPH_CAPTURE NCCL_DEBUG; do
    if declare -p "$pass_var" >/dev/null 2>&1; then
        server_env+=("$pass_var=${!pass_var}")
    fi
done

common_args=(
    "$TARGET_MODEL"
    --cluster-size "$CLUSTER_SIZE"
    --cluster-head "$HEAD_HOST:$HEAD_PORT"
    --cluster-ifname "$CLUSTER_IFNAME"
    --cluster-ib-hca "$CLUSTER_IB_HCA"
    --cluster-gid-index "$CLUSTER_GID_INDEX"
    --cluster-expert-placement "$PLACEMENT"
    --cluster-timeout-ms "$CLUSTER_TIMEOUT_MS"
    --cluster-verify-hash "$VERIFY_HASH"
    --target-device hip:0
    --ds4-expert-top-k "$EXPERT_TOP_K"
    --ds4-prefill sparse
    --chunk 2048
    --max-ctx "$MAX_CTX"
    --prefix-cache-slots 0
    --prefill-cache-slots 0
    --hard-limit-reply-budget 0
)
head_args=("$SERVER_BIN" "${common_args[@]}" --cluster-rank 0 --host 0.0.0.0 --port "$HTTP_PORT")
worker_args=("$WORKER_SERVER_BIN" "${common_args[@]}" --cluster-rank 1)

head_pid=""
worker_pid=""
cleanup() {
    if [[ -n "$head_pid" ]] && kill -0 "$head_pid" 2>/dev/null; then
        kill -TERM "$head_pid" 2>/dev/null || true
        wait "$head_pid" 2>/dev/null || true
    fi
    if [[ -n "$worker_pid" ]] && kill -0 "$worker_pid" 2>/dev/null; then
        # The worker exits on the head's Shutdown; give it a moment, then force.
        for _ in $(seq 1 20); do
            kill -0 "$worker_pid" 2>/dev/null || break
            sleep 0.5
        done
        kill -TERM "$worker_pid" 2>/dev/null || true
        wait "$worker_pid" 2>/dev/null || true
    fi
    # Belt and braces: no stray worker on the remote node.
    ssh -o BatchMode=yes "$WORKER_SSH" "pkill -TERM -f '(^|/)dflash_server([[:space:]]|$)' || true" \
        >/dev/null 2>&1 || true
}
trap cleanup EXIT

{
    echo "schema_version=1"
    echo "run_id=$RUN_ID"
    echo "source_commit=$(git -C "$CHECKOUT" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "cluster_size=$CLUSTER_SIZE"
    echo "head_host=$HEAD_HOST head_port=$HEAD_PORT http_port=$HTTP_PORT"
    echo "worker_ssh=$WORKER_SSH"
    echo "ifname=$CLUSTER_IFNAME hca=$CLUSTER_IB_HCA gid_index=$CLUSTER_GID_INDEX"
    echo "placement=$PLACEMENT verify_hash=$VERIFY_HASH timeout_ms=$CLUSTER_TIMEOUT_MS"
    echo "expert_top_k=$EXPERT_TOP_K max_ctx=$MAX_CTX max_tokens=$MAX_TOKENS warmup=$WARMUP runs=$RUNS"
    echo "golden_name=$GOLDEN_NAME expected_sha256=$EXPECTED_SHA256"
    echo "prompt_file=$PROMPT_FILE"
    sha256sum -- "$SERVER_BIN"
    stat -c 'target_model=%n bytes=%s mtime=%y' -- "$TARGET_MODEL"
    echo "target_model_sha256_expected=$TARGET_MODEL_SHA256"
    if [[ -r "$DRAFT_MODEL" ]]; then
        stat -c 'draft_model=%n bytes=%s mtime=%y (unused in AR qualification)' -- "$DRAFT_MODEL"
    fi
    if [[ "$HASH_MODELS" == 1 ]]; then
        sha256sum -- "$TARGET_MODEL"
    fi
    printf 'server_env='; printf '%q ' "${server_env[@]}"; echo
    printf 'head_args='; printf '%q ' "${head_args[@]}"; echo
    printf 'worker_args='; printf '%q ' "${worker_args[@]}"; echo
    date -u '+started_utc=%Y-%m-%dT%H:%M:%SZ'
} >"$OUT_DIR/manifest.txt"

if [[ "$HASH_MODELS" == 1 ]]; then
    actual_sha="$(sha256sum -- "$TARGET_MODEL" | awk '{ print $1 }')"
    if [[ "$actual_sha" != "$TARGET_MODEL_SHA256" ]]; then
        echo "target model sha256 $actual_sha != expected $TARGET_MODEL_SHA256" >&2
        exit 2
    fi
fi

# Head first (it binds the control port), then the worker, which retries its
# connect for up to five minutes while the head loads the model.
env -i "${server_env[@]}" "${head_args[@]}" >"$HEAD_LOG" 2>&1 &
head_pid=$!

remote_cmd="$(printf 'env -i %q ' "${server_env[@]}"; printf '%q ' "${worker_args[@]}")"
ssh -o BatchMode=yes -o ServerAliveInterval=15 "$WORKER_SSH" "$remote_cmd" >"$WORKER_LOG" 2>&1 &
worker_pid=$!

ready=0
for _ in $(seq 1 "$READY_TIMEOUT_S"); do
    if grep -q "listening on" "$HEAD_LOG" 2>/dev/null; then
        ready=1
        break
    fi
    if ! kill -0 "$head_pid" 2>/dev/null; then
        echo "head exited during startup" >&2
        tail -120 "$HEAD_LOG" >&2
        tail -60 "$WORKER_LOG" >&2 || true
        exit 1
    fi
    if ! kill -0 "$worker_pid" 2>/dev/null; then
        echo "worker exited during startup" >&2
        tail -120 "$WORKER_LOG" >&2 || true
        exit 1
    fi
    sleep 1
done
if [[ "$ready" != 1 ]]; then
    echo "cluster did not become ready within ${READY_TIMEOUT_S}s" >&2
    tail -60 "$HEAD_LOG" >&2
    exit 1
fi
grep -E '^\[cluster\]' "$HEAD_LOG" | head -20 || true

bench_args=(
    --url "http://$HEAD_HOST:$HTTP_PORT"
    --model dflash
    --max-tokens "$MAX_TOKENS"
    --warmups "$WARMUP"
    --runs "$RUNS"
    --temperature 0
    --reference-sha256 "$EXPECTED_SHA256"
    --json-out "$OUT_DIR/bench.json"
)
if [[ -n "$PROMPT_FILE" ]]; then
    bench_args+=(--prompt-file "$PROMPT_FILE")
fi
set +e
python3 "$BENCH_CLIENT" "${bench_args[@]}" 2>&1 | tee "$OUT_DIR/bench.log"
bench_status=${PIPESTATUS[0]}
set -e

date -u '+finished_utc=%Y-%m-%dT%H:%M:%SZ' >>"$OUT_DIR/manifest.txt"
echo "bench_status=$bench_status" >>"$OUT_DIR/manifest.txt"

# Hash-check verdict from the head log (--cluster-verify-hash).
mismatches="$(grep -c 'HASH MISMATCH' "$HEAD_LOG" || true)"
echo "hash_mismatches=$mismatches" >>"$OUT_DIR/manifest.txt"
if (( VERIFY_HASH > 0 && mismatches > 0 )); then
    echo "FAIL: $mismatches hash mismatch line(s) in $HEAD_LOG" >&2
    grep 'HASH MISMATCH' "$HEAD_LOG" | head -5 >&2
    bench_status=2
fi

echo "OUT_DIR=$OUT_DIR"
grep -E '\[cluster\]|decode:|chat DONE' "$HEAD_LOG" | tail -60 || true
exit "$bench_status"
