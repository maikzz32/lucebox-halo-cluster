#!/usr/bin/env bash
# fault_drill.sh - kill a rank mid-request and check the cluster behaves.
#
# This is the M4 gate. It exists because the failure mode it checks is not
# obvious: with the in-graph all-reduce of path 3b nothing on the host waits
# for a collective any more, so before the watchdog landed a killed worker left
# the head blocked inside HIP forever - no error, no HTTP answer, no exit.
#
# What a pass looks like:
#   1. a long request is in flight;
#   2. one worker is killed;
#   3. the head notices within FAULT_DEADLINE_S, aborts the communicator, and
#      answers the request with usage.timings.cluster.error set;
#   4. the head exits 3 (the supervisor's cue) - with RESTART=on-failure:N it
#      comes back by itself;
#   5. after every rank is up again the cluster reforms and the benchmark
#      output is byte-identical to the reference.
#
# Note on step 4: podman does NOT restart a container the operator killed, so
# the drill starts the killed worker again itself. A real crash (the process
# dying inside the container) is restarted by the policy.
#
# Usage:
#   scripts/cluster/fault_drill.sh "strix1 strix2" <target.gguf> <dspark.gguf> [--no-launch]
#
# Environment: the same variables launch_cluster.sh takes (IMAGE, BIN_DIR,
# MODELS_DIR, SPEC, ...) plus
#   REFERENCE_SHA256   expected completion hash for the recovery check
#   FAULT_DEADLINE_S   seconds the head may take to fail the request (default 30)
#   RECOVER_DEADLINE_S seconds for the cluster to come back  (default 900)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

HOSTS_ARG="${1:-}"
TARGET="${2:-}"
DSPARK="${3:-}"
LAUNCH=1
[ "${4:-}" = "--no-launch" ] && LAUNCH=0
if [ -z "$HOSTS_ARG" ] || [ -z "$TARGET" ] || [ -z "$DSPARK" ]; then
    sed -n '2,30p' "$0" >&2
    exit 2
fi

read -r -a HOSTS <<< "$HOSTS_ARG"
N=${#HOSTS[@]}
[ "$N" -ge 2 ] || { echo "fault drill needs at least two ranks" >&2; exit 2; }
HEAD="${HOSTS[0]}"
VICTIM="${HOSTS[1]}"          # rank 1
HTTP_PORT="${HTTP_PORT:-8016}"
FAULT_DEADLINE_S="${FAULT_DEADLINE_S:-30}"
RECOVER_DEADLINE_S="${RECOVER_DEADLINE_S:-900}"
REFERENCE_SHA256="${REFERENCE_SHA256:-}"

head_ip() { ssh "$HEAD" "getent hosts $HEAD >/dev/null 2>&1 || true"; echo "${HEAD_IP:-192.168.100.1}"; }
URL="http://$(head_ip):${HTTP_PORT}"

fail() { echo "FAIL: $*" >&2; exit 1; }

if [ "$LAUNCH" = 1 ]; then
    echo "== launching ${N} ranks (RESTART=${RESTART:-on-failure:3})"
    RESTART="${RESTART:-on-failure:3}" \
        bash "$REPO/scripts/cluster/launch_cluster.sh" "$HOSTS_ARG" "$TARGET" "$DSPARK" >/dev/null
fi

echo "== waiting for the head to serve"
for _ in $(seq 1 60); do
    ssh "$HEAD" "curl -s -m 3 ${URL}/health >/dev/null 2>&1" && break
    sleep 15
done
ssh "$HEAD" "curl -s -m 3 ${URL}/health >/dev/null 2>&1" || fail "head never came up"

echo "== step 1/5: long request in flight"
ssh "$HEAD" "rm -f ~/fault_drill_resp.json ~/fault_drill_code.txt; \
    nohup bash -c 'curl -s -m 300 -o ~/fault_drill_resp.json -w \"%{http_code} %{time_total}\" \
    ${URL}/v1/chat/completions -H \"content-type: application/json\" \
    -d \"{\\\"model\\\":\\\"dflash\\\",\\\"messages\\\":[{\\\"role\\\":\\\"user\\\",\\\"content\\\":\\\"Write a detailed essay about the history of computing.\\\"}],\\\"max_tokens\\\":400,\\\"temperature\\\":0}\" \
    > ~/fault_drill_code.txt 2>&1' >/dev/null 2>&1 &"
sleep 15

echo "== step 2/5: killing rank 1 on ${VICTIM}"
ssh "$VICTIM" "podman kill lucebox-rank1 >/dev/null"
KILLED_AT=$(date +%s)

echo "== step 3/5: the request must fail within ${FAULT_DEADLINE_S}s"
answered=0
for _ in $(seq 1 "$FAULT_DEADLINE_S"); do
    if ssh "$HEAD" "test -s ~/fault_drill_code.txt" 2>/dev/null; then answered=1; break; fi
    sleep 1
done
[ "$answered" = 1 ] || fail "the request never returned; the head is stuck in a collective"
took=$(( $(date +%s) - KILLED_AT ))
reason=$(ssh "$HEAD" "python3 -c \"
import json
d = json.load(open('/home/\$USER/fault_drill_resp.json'))
print(d['usage']['timings'].get('cluster', {}).get('error', ''))
\"" 2>/dev/null || true)
[ -n "$reason" ] || fail "the response carries no usage.timings.cluster.error"
echo "   answered ${took}s after the kill: ${reason}"

echo "== step 4/5: the head must exit for the supervisor"
gone=0
for _ in $(seq 1 30); do
    st=$(ssh "$HEAD" "podman inspect lucebox-rank0 --format '{{.State.Status}}:{{.State.ExitCode}}' 2>/dev/null" || echo "")
    case "$st" in exited:*|"") gone=1; break ;; esac
    # A restart policy may already have it back up; a fresh uptime counts too.
    up=$(ssh "$HEAD" "podman inspect lucebox-rank0 --format '{{.RestartCount}}' 2>/dev/null" || echo 0)
    [ "${up:-0}" -gt 0 ] && { gone=1; echo "   head restarted by the policy (RestartCount=${up})"; break; }
    sleep 2
done
[ "$gone" = 1 ] || fail "the head neither exited nor restarted"

echo "== step 5/5: bring rank 1 back and check the cluster reforms"
ssh "$VICTIM" "podman start lucebox-rank1 >/dev/null" || true
ssh "$HEAD" "podman start lucebox-rank0 >/dev/null 2>&1" || true
back=0
deadline=$(( $(date +%s) + RECOVER_DEADLINE_S ))
while [ "$(date +%s)" -lt "$deadline" ]; do
    if ssh "$HEAD" "curl -s -m 3 ${URL}/health >/dev/null 2>&1"; then back=1; break; fi
    sleep 15
done
[ "$back" = 1 ] || fail "the cluster did not come back within ${RECOVER_DEADLINE_S}s"

bench_args=(--url "$URL" --max-tokens 128 --runs 2 --warmups 1 --timeout 1800)
[ -n "$REFERENCE_SHA256" ] && bench_args+=(--reference-sha256 "$REFERENCE_SHA256")
ssh "$HEAD" "python3 ~/bench_ds4_cluster.py ${bench_args[*]}" || fail "post-recovery benchmark failed"

echo "== fault drill PASSED"
