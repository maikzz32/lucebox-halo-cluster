#!/usr/bin/env bash
# rccl_baseline.sh - RCCL / RDMA fabric baseline across N Strix Halo nodes.
#
# Default: rccl-tests `all_reduce_perf -b 8K -e 64M -f 2 -g 1` with one rank
# per node, launched by mpirun (OpenMPI inside the cluster image) over the
# RoCE net. Prints the 64 KiB and 16 MiB rows; --record appends them to the
# baseline table in server/docs/CLUSTER.md.
#
# Two ways to place the ranks:
#   one-shot (default): an ephemeral container per node from IMAGE
#                       (podman run --rm, GPU devices attached, removed on exit)
#   --exec:             `podman exec` into the running lucebox-rank* containers
#                       (needs a cluster started by launch_cluster.sh; shares
#                       the GPU with dflash_server, so numbers are noisier)
# --verbs runs perftest ib_write_bw / ib_write_lat for every host pair instead.
#
# mpirun runs in a launcher container on the first host and reaches the other
# nodes with ssh (your ~/.ssh is mounted read-only into that container), where
# it starts the remote rank container via podman. Nothing outside
# lucebox-rccl-* is ever touched.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: rccl_baseline.sh [options] <hosts-file|"h1 h2 ...">

Options:
  --exec           run inside existing lucebox-rank<N> containers (podman exec)
  --verbs          pairwise ib_write_bw / ib_write_lat instead of all_reduce_perf
  --record         append the 64 KiB / 16 MiB result rows to server/docs/CLUSTER.md
  -b <size>        min message size (default 8K)
  -e <size>        max message size (default 64M)
  -n <iters>       iterations per size (default 200)
  --help

Environment:
  IMAGE       (default ghcr.io/maikzz32/lucebox-halo-cluster:dev)
  RDMA_NET    (default 192.168.100)
  GID_INDEX   (default 1)          NCCL_ALGO / NCCL_PROTO   (optional, e.g. Ring / Simple)
  NCCL_DEBUG  (default WARN; INFO shows the chosen transport: look for "NET/IB")
  SSH_USER    user for mpirun's ssh into the peers (default maik)
  DOC         CLUSTER.md path for --record (default server/docs/CLUSTER.md)

Requires: the vLLM Ray cluster stopped (it holds ~100 GB GPU memory per node).
USAGE
}

IMAGE="${IMAGE:-ghcr.io/maikzz32/lucebox-halo-cluster:dev}"
RDMA_NET="${RDMA_NET:-192.168.100}"
GID_INDEX="${GID_INDEX:-1}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
SSH_USER="${SSH_USER:-maik}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DOC="${DOC:-$SCRIPT_DIR/../../server/docs/CLUSTER.md}"
MODE=oneshot; VERBS=0; RECORD=0; MINB=8K; MAXB=64M; ITERS=200
HOSTS_ARG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --exec) MODE=exec ;;
        --verbs) VERBS=1 ;;
        --record) RECORD=1 ;;
        -b) MINB="$2"; shift ;;
        -e) MAXB="$2"; shift ;;
        -n) ITERS="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "unknown option $1" >&2; usage >&2; exit 2 ;;
        *) HOSTS_ARG="$1" ;;
    esac
    shift
done
[ -n "$HOSTS_ARG" ] || { usage >&2; exit 2; }

HOSTS=()
if [ -f "$HOSTS_ARG" ]; then
    while IFS= read -r line; do line="${line%%#*}"; line="${line//[[:space:]]/}"; [ -n "$line" ] && HOSTS+=("$line"); done <"$HOSTS_ARG"
else
    read -r -a HOSTS <<<"$HOSTS_ARG"
fi
N="${#HOSTS[@]}"
[ "$N" -ge 2 ] || { echo "ERROR: need >= 2 hosts" >&2; exit 2; }
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10)

rdma_ip() {
    local h="$1"
    if [[ "$h" =~ ^strix([0-9]+)$ ]]; then echo "${RDMA_NET}.${BASH_REMATCH[1]}"; return; fi
    ssh "${SSH_OPTS[@]}" "$h" "ip -4 -o addr show | awk -v n='$RDMA_NET' '\$4 ~ \"^\"n\"\\\\.\" {sub(/\\/.*/,\"\",\$4); print \$4; exit}'"
}
node_hca() { if [ "$1" = 4 ]; then echo rocep197s0f1; else echo rocep197s0f3; fi; }

IPS=(); for h in "${HOSTS[@]}"; do IPS+=("$(rdma_ip "$h")"); done
echo "== hosts: ${HOSTS[*]}  ips: ${IPS[*]}  image: ${IMAGE}  mode: $([ "$VERBS" = 1 ] && echo verbs || echo "$MODE")"

# Common podman flags for a GPU+RDMA container (same set as launch_cluster.sh).
PODMAN_GPU_FLAGS=(--device /dev/kfd --device /dev/dri --device /dev/infiniband
    --group-add keep-groups --security-opt seccomp=unconfined --security-opt label=disable
    --network host --ipc host --pids-limit -1 --ulimit memlock=-1:-1)
for n in 1 2 3 4; do PODMAN_GPU_FLAGS+=(--add-host "StrixHalo${n}:${RDMA_NET}.${n}"); done

STAMP="$(date -u +%Y-%m-%dT%H:%MZ)"

# ── --verbs: pairwise perftest ──────────────────────────────────────────────
if [ "$VERBS" = 1 ]; then
    PORT=18515
    for ((i = 0; i < N; i++)); do
        for ((j = i + 1; j < N; j++)); do
            hs="${HOSTS[$i]}"; hc="${HOSTS[$j]}"; ips="${IPS[$i]}"
            hca_s="$(node_hca "${ips##*.}")"; hca_c="$(node_hca "${IPS[$j]##*.}")"
            for tool in ib_write_bw ib_write_lat; do
                echo "-- ${tool}: ${hs} (${hca_s}) <- ${hc} (${hca_c})"
                ssh "${SSH_OPTS[@]}" "$hs" "podman run --rm -d --name lucebox-rccl-verbs-srv $(printf '%q ' "${PODMAN_GPU_FLAGS[@]}") --entrypoint '' '$IMAGE' $tool -d $hca_s -x $GID_INDEX -F -p $PORT --report_gbits" >/dev/null
                sleep 2
                ssh "${SSH_OPTS[@]}" "$hc" "podman run --rm --name lucebox-rccl-verbs-cli $(printf '%q ' "${PODMAN_GPU_FLAGS[@]}") --entrypoint '' '$IMAGE' $tool -d $hca_c -x $GID_INDEX -F -p $PORT --report_gbits $ips" \
                    | grep -vE '^\s*$' | tail -12 | sed 's/^/   /' || echo "   ${tool} client failed"
                ssh "${SSH_OPTS[@]}" "$hs" "podman rm -f lucebox-rccl-verbs-srv >/dev/null 2>&1 || true"
            done
        done
    done
    echo "== done (perftest reference on this fabric: ~5 us RDMA latency, 25 GbE line rate ~23 Gb/s payload)"
    exit 0
fi

# ── all_reduce_perf under mpirun ────────────────────────────────────────────
HEAD="${HOSTS[0]}"
HOSTLIST="$(IFS=,; echo "${IPS[*]}")"
NCCL_X=(-x NCCL_DEBUG -x NCCL_IB_GID_INDEX -x NCCL_NET_GDR_LEVEL -x NCCL_IB_DISABLE
        -x NCCL_ASYNC_ERROR_HANDLING -x NCCL_IB_QPS_PER_CONNECTION -x NCCL_IB_TIMEOUT
        -x NCCL_IB_RETRY_CNT -x HIP_FORCE_DEV_KERNARG -x RDMA_NET -x RCCL_RANK_ENV_QUIET)
[ -n "${NCCL_ALGO:-}" ] && NCCL_X+=(-x NCCL_ALGO)
[ -n "${NCCL_PROTO:-}" ] && NCCL_X+=(-x NCCL_PROTO)

# rsh agent used by mpirun for the REMOTE ranks: ssh to the node, then either
# exec into its lucebox-rank container or start a one-shot container that runs
# orted. Written into the launcher container via a bind-mounted temp dir.
TMPD="$(mktemp -d)"; trap 'rm -rf "$TMPD"' EXIT
if [ "$MODE" = exec ]; then
    cat >"$TMPD/rsh.sh" <<EOF
#!/usr/bin/env bash
# usage by OpenMPI: rsh.sh <host> <orted command...>
host="\$1"; shift
exec ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new ${SSH_USER}@"\$host" \
  "c=\\\$(podman ps --filter name=^lucebox-rank -q | head -1); [ -n \\\"\\\$c\\\" ] || { echo 'no lucebox-rank container on '\\\$(hostname) >&2; exit 1; }; podman exec -i \\\$c \$(printf '%q ' "\$@")"
EOF
else
    cat >"$TMPD/rsh.sh" <<EOF
#!/usr/bin/env bash
host="\$1"; shift
exec ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new ${SSH_USER}@"\$host" \
  "podman run --rm -i --name lucebox-rccl-rank-\\\$\\\$ $(printf '%q ' "${PODMAN_GPU_FLAGS[@]}") --entrypoint '' '$IMAGE' \$(printf '%q ' "\$@")"
EOF
fi
chmod +x "$TMPD/rsh.sh"
scp -q "${SSH_OPTS[@]}" "$TMPD/rsh.sh" "$HEAD:/tmp/lucebox-rccl-rsh.sh"

MPI_CMD=(mpirun --allow-run-as-root -np "$N" -H "$HOSTLIST" --map-by ppr:1:node --bind-to none
    --prefix /usr/lib64/openmpi
    --mca plm_rsh_agent /tmp/lucebox-rccl-rsh.sh
    --mca btl_tcp_if_include "${RDMA_NET}.0/24" --mca oob_tcp_if_include "${RDMA_NET}.0/24"
    --mca btl tcp,self
    "${NCCL_X[@]}"
    /opt/lucebox/scripts/cluster/rccl_rank_env.sh
    /opt/rccl-tests/bin/all_reduce_perf -b "$MINB" -e "$MAXB" -f 2 -g 1 -n "$ITERS" -w 20 -c 1)

LAUNCH_ENV=(-e "NCCL_DEBUG=${NCCL_DEBUG}" -e "NCCL_IB_GID_INDEX=${GID_INDEX}" -e "RDMA_NET=${RDMA_NET}"
    -e RCCL_RANK_ENV_QUIET=1 -e "NCCL_ALGO=${NCCL_ALGO:-}" -e "NCCL_PROTO=${NCCL_PROTO:-}")

echo "-- launching mpirun on ${HEAD} (${N} ranks, one per node)"
OUT="$TMPD/all_reduce.txt"
ssh "${SSH_OPTS[@]}" "$HEAD" "podman run --rm -i --name lucebox-rccl-launcher $(printf '%q ' "${PODMAN_GPU_FLAGS[@]}") \
    -v \$HOME/.ssh:/root/.ssh:ro -v /tmp/lucebox-rccl-rsh.sh:/tmp/lucebox-rccl-rsh.sh:ro \
    $(printf '%q ' "${LAUNCH_ENV[@]}") --entrypoint '' '$IMAGE' $(printf '%q ' "${MPI_CMD[@]}")" | tee "$OUT"

echo
echo "== key rows (out-of-place time is column 6; bandwidth in GB/s):"
printf '   %-10s %-8s %-10s %-10s %-10s\n' 'size(B)' count type 'time(us)' busbw
ROW64="$(awk '$1==65536 && $3=="float" {print; exit}' "$OUT" || true)"
ROW16M="$(awk '$1==16777216 && $3=="float" {print; exit}' "$OUT" || true)"
for r in "$ROW64" "$ROW16M"; do
    [ -n "$r" ] || continue
    # columns: size count type redop root oop_time oop_algbw oop_busbw oop_wrong ip_time ...
    printf '   %-10s %-8s %-10s %-10s %-10s\n' "$(awk '{print $1}' <<<"$r")" "$(awk '{print $2}' <<<"$r")" \
        "$(awk '{print $3}' <<<"$r")" "$(awk '{print $6}' <<<"$r")" "$(awk '{print $8}' <<<"$r")"
done
[ -n "$ROW64" ] || echo "   (no 65536-byte row found; check the raw output above)"
if grep -q "NET/IB : No device found" "$OUT"; then
    echo "   WARN: RCCL fell back to TCP (NET/IB : No device found) -> /dev/infiniband not in the container?"
fi

if [ "$RECORD" = 1 ]; then
    [ -f "$DOC" ] || { echo "ERROR: $DOC not found for --record" >&2; exit 1; }
    t64="$(awk '{print $6}' <<<"$ROW64")"; t16="$(awk '{print $6}' <<<"$ROW16M")"; bw16="$(awk '{print $8}' <<<"$ROW16M")"
    row="| ${STAMP} | ${N} | ${HOSTS[*]} | ${t64:-n/a} | ${t16:-n/a} | ${bw16:-n/a} | ${NCCL_ALGO:-default}/${NCCL_PROTO:-default}, ${MODE} |"
    python3 - "$DOC" "$row" <<'PY'
import sys
path, row = sys.argv[1], sys.argv[2]
s = open(path, encoding="utf-8").read()
marker = "<!-- rccl-baseline-end -->"
if marker not in s:
    sys.exit("marker %s missing in %s" % (marker, path))
s = s.replace(marker, row + "\n" + marker, 1)
open(path, "w", encoding="utf-8", newline="\n").write(s)
print("recorded:", row)
PY
fi
