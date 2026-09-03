#!/usr/bin/env bash
# host_prep.sh - check (default) or apply (--apply) the host prerequisites for
# one lucebox-halo-cluster node. Run ON the node (or `ssh strixN 'bash -s' <
# scripts/cluster/host_prep.sh`). Idempotent; check mode never changes anything.
#
# Node identity is derived from the node's own 192.168.100.x address exactly
# like /home/maik/cluster_tpn.sh: StrixHalo4 (192.168.100.4) has the Intel
# E830-L (enp197s0f1np1 / rocep197s0f1), nodes 1-3 have the Intel E810-C
# (enp197s0f3np3 / rocep197s0f3). Override with IFACE=... HCA=... env.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: host_prep.sh [--apply] [--yes] [--help]

Checks (always):
  * kernel cmdline: iommu=pt pci=realloc amdgpu.gttsize=126976
        ttm.pages_limit=32505856 ttm.page_pool_size=32505856
        amdgpu.gpu_recovery=1 amdgpu.lockup_timeout=10000 pcie_aspm=off
  * RoCE interface up, carries the 192.168.100.x address, MTU 9000
  * `rdma link` state ACTIVE for the expected HCA
  * rdma-core version >= 64 (needed for the E830 on StrixHalo4; E810 works with older)
  * GID table of the HCA (show_gids-like; RoCE v2 entries for the 192.168.100.x IP)
  * memlock unlimited (ulimit -l, /etc/security/limits.d)
  * /dev/infiniband, /dev/kfd, /dev/dri present
  * podman present; note about the running vLLM cluster
  * /etc/hosts entries 192.168.100.N StrixHaloN (N=1..4)

--apply fixes what it can (MTU, /etc/hosts, memlock limits.d drop-in). Kernel
cmdline changes are only PRINTED as a grubby command; they need a reboot and
are never applied automatically. --apply asks before each change unless --yes.

Env overrides: IFACE, HCA, RDMA_NET (default 192.168.100), MTU (default 9000).
Exit status: 0 = all checks passed, 1 = at least one check failed, 2 = usage.
USAGE
}

APPLY=0
YES=0
for a in "$@"; do
    case "$a" in
        --apply) APPLY=1 ;;
        --yes|-y) YES=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $a" >&2; usage >&2; exit 2 ;;
    esac
done

RDMA_NET="${RDMA_NET:-192.168.100}"
MTU_WANT="${MTU:-9000}"
FAIL=0

ok()   { printf '  [ OK ] %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; FAIL=1; }
warn() { printf '  [WARN] %s\n' "$*"; }
info() { printf '  [info] %s\n' "$*"; }

confirm() {
    # confirm <prompt>: true if --yes or the operator answers y.
    if [ "$YES" = 1 ]; then return 0; fi
    read -r -p "$1 [y/N] " ans </dev/tty || return 1
    [ "$ans" = "y" ] || [ "$ans" = "Y" ]
}

need_root() {
    if [ "$(id -u)" -ne 0 ]; then
        if command -v sudo >/dev/null 2>&1; then SUDO="sudo"; else
            echo "ERROR: --apply needs root or sudo" >&2; exit 1; fi
    else
        SUDO=""
    fi
}

# ── Node identity ───────────────────────────────────────────────────────────
echo "== lucebox-halo-cluster host check on $(hostname) ($(date -Is))"
MY_IP="$(ip -4 -o addr show 2>/dev/null | awk -v n="$RDMA_NET" '$4 ~ "^"n"\\." {sub(/\/.*/,"",$4); print $4; exit}')"
if [ -z "$MY_IP" ]; then
    bad "no address in ${RDMA_NET}.0/24 on this host (RDMA net not configured?)"
    NODE_ID=0
else
    NODE_ID="${MY_IP##*.}"
    ok "RDMA address ${MY_IP} (node ${NODE_ID})"
fi
if [ -z "${IFACE:-}" ]; then
    if [ "$NODE_ID" = 4 ]; then IFACE=enp197s0f1np1; else IFACE=enp197s0f3np3; fi
fi
if [ -z "${HCA:-}" ]; then
    if [ "$NODE_ID" = 4 ]; then HCA=rocep197s0f1; else HCA=rocep197s0f3; fi
fi
info "expected iface=${IFACE} hca=${HCA} (override with IFACE=/HCA=)"

# ── Kernel cmdline ──────────────────────────────────────────────────────────
echo "-- kernel cmdline"
CMDLINE="$(cat /proc/cmdline)"
WANT_PARAMS=(iommu=pt pci=realloc amdgpu.gttsize=126976 ttm.pages_limit=32505856
             ttm.page_pool_size=32505856 amdgpu.gpu_recovery=1 amdgpu.lockup_timeout=10000 pcie_aspm=off)
MISSING=()
for p in "${WANT_PARAMS[@]}"; do
    case " $CMDLINE " in
        *" $p "*) ok "$p" ;;
        *) bad "missing kernel parameter $p"; MISSING+=("$p") ;;
    esac
done
if [ "${#MISSING[@]}" -gt 0 ]; then
    warn "add with:  sudo grubby --update-kernel=ALL --args='${MISSING[*]}'  && reboot   (NOT applied automatically)"
fi

# ── Interface / MTU ─────────────────────────────────────────────────────────
echo "-- RoCE interface ${IFACE}"
if [ -d "/sys/class/net/${IFACE}" ]; then
    STATE="$(cat "/sys/class/net/${IFACE}/operstate" 2>/dev/null || echo unknown)"
    MTU_HAVE="$(cat "/sys/class/net/${IFACE}/mtu" 2>/dev/null || echo 0)"
    [ "$STATE" = up ] && ok "operstate up" || bad "operstate ${STATE}"
    if ip -4 -o addr show dev "$IFACE" | grep -q " ${MY_IP}/"; then
        ok "${IFACE} carries ${MY_IP}"
    else
        bad "${IFACE} does not carry ${MY_IP}; check IFACE= (ip -4 -o addr show)"
    fi
    if [ "$MTU_HAVE" = "$MTU_WANT" ]; then
        ok "MTU ${MTU_HAVE}"
    else
        bad "MTU ${MTU_HAVE}, want ${MTU_WANT}"
        if [ "$APPLY" = 1 ]; then
            need_root
            if confirm "Set MTU ${MTU_WANT} on ${IFACE} now (non-persistent; make it persistent in NetworkManager)?"; then
                $SUDO ip link set dev "$IFACE" mtu "$MTU_WANT"
                info "applied. persistent: sudo nmcli con mod \"$(nmcli -g GENERAL.CONNECTION dev show "$IFACE" 2>/dev/null || echo '<con>')\" 802-3-ethernet.mtu ${MTU_WANT}"
            fi
        fi
    fi
    SPEED="$(cat "/sys/class/net/${IFACE}/speed" 2>/dev/null || echo '?')"
    info "link speed ${SPEED} Mb/s (expected 25000: the fabric is 25 GbE)"
else
    bad "interface ${IFACE} not found"
fi

# ── rdma link / HCA ─────────────────────────────────────────────────────────
echo "-- RDMA device ${HCA}"
if command -v rdma >/dev/null 2>&1; then
    RL="$(rdma link show 2>/dev/null | grep -E "link ${HCA}/" || true)"
    if [ -z "$RL" ]; then
        bad "rdma link shows no ${HCA} (rdma link show: $(rdma link show 2>/dev/null | tr '\n' ';'))"
    else
        info "$RL"
        if echo "$RL" | grep -q "state ACTIVE" && echo "$RL" | grep -q "physical_state LINK_UP"; then
            ok "${HCA} ACTIVE / LINK_UP"
        else
            bad "${HCA} not ACTIVE"
        fi
        if echo "$RL" | grep -q "netdev ${IFACE}"; then
            ok "${HCA} is bound to ${IFACE}"
        else
            warn "${HCA} netdev is not ${IFACE}; check IFACE/HCA mapping"
        fi
    fi
else
    bad "rdma tool (iproute) missing"
fi

# rdma-core version
echo "-- rdma-core"
RC_VER="$(rpm -q --qf '%{VERSION}' rdma-core 2>/dev/null || echo '')"
if [ -z "$RC_VER" ]; then
    RC_VER="$(ibv_devinfo -h 2>&1 | grep -oE 'v?[0-9]+' | head -1 || true)"
fi
RC_MAJ="${RC_VER%%.*}"
if [ -n "$RC_MAJ" ] && [ "$RC_MAJ" -ge 64 ] 2>/dev/null; then
    ok "rdma-core ${RC_VER} (>= 64)"
elif [ "$NODE_ID" = 4 ]; then
    bad "rdma-core ${RC_VER:-unknown} < 64: the E830-L on this node needs >= v64 (Fedora 44 ships v61; use Fedora 45 or build rdma-core from source)"
else
    warn "rdma-core ${RC_VER:-unknown} < 64 (fine for the E810 on this node; StrixHalo4/E830 needs >= 64)"
fi

# GID table
echo "-- GID table ${HCA} port 1 (show_gids-like)"
GIDDIR="/sys/class/infiniband/${HCA}/ports/1/gids"
if [ -d "$GIDDIR" ]; then
    printf '  %-4s %-40s %-8s %s\n' IDX GID TYPE NDEV
    V2_IDX=""
    for f in "$GIDDIR"/*; do
        idx="$(basename "$f")"
        gid="$(cat "$f" 2>/dev/null || echo '')"
        [ "$gid" = "0000:0000:0000:0000:0000:0000:0000:0000" ] && continue
        [ -z "$gid" ] && continue
        typ="$(cat "/sys/class/infiniband/${HCA}/ports/1/gid_attrs/types/${idx}" 2>/dev/null || echo '?')"
        ndev="$(cat "/sys/class/infiniband/${HCA}/ports/1/gid_attrs/ndevs/${idx}" 2>/dev/null || echo '?')"
        printf '  %-4s %-40s %-8s %s\n' "$idx" "$gid" "$typ" "$ndev"
        # IPv4-mapped GID ::ffff:a.b.c.d encodes the IP in the last two groups.
        if [ -n "$MY_IP" ] && echo "$typ" | grep -qi "v2"; then
            IFS=. read -r a b c d <<<"$MY_IP"
            hex="$(printf '%02x%02x:%02x%02x' "$a" "$b" "$c" "$d")"
            case "$gid" in *"ffff:${hex}") V2_IDX="${V2_IDX:-$idx}" ;; esac
        fi
    done
    if [ -n "$V2_IDX" ]; then
        if [ "$V2_IDX" = 1 ]; then
            ok "RoCE v2 GID for ${MY_IP} at index ${V2_IDX} (matches --cluster-gid-index 1 / NCCL_IB_GID_INDEX=1)"
        else
            warn "RoCE v2 GID for ${MY_IP} at index ${V2_IDX}, not 1: pass --cluster-gid-index ${V2_IDX}"
        fi
    else
        bad "no RoCE v2 GID for ${MY_IP} found on ${HCA}"
    fi
else
    bad "${GIDDIR} missing"
fi

# ── memlock ─────────────────────────────────────────────────────────────────
echo "-- memlock"
ML="$(ulimit -l)"
if [ "$ML" = unlimited ]; then ok "ulimit -l unlimited"; else
    bad "ulimit -l = ${ML} (containers are started with --ulimit memlock=-1:-1, but the host shell limit should be unlimited too)"
    if [ "$APPLY" = 1 ]; then
        need_root
        if confirm "Write /etc/security/limits.d/90-lucebox-memlock.conf (* soft/hard memlock unlimited)?"; then
            printf '* soft memlock unlimited\n* hard memlock unlimited\n' | $SUDO tee /etc/security/limits.d/90-lucebox-memlock.conf >/dev/null
            info "applied; takes effect on next login"
        fi
    fi
fi

# ── Devices ─────────────────────────────────────────────────────────────────
echo "-- devices"
for d in /dev/infiniband /dev/kfd /dev/dri; do
    if [ -e "$d" ]; then ok "$d"; else bad "$d missing"; fi
done
[ -e /dev/infiniband/rdma_cm ] && ok "/dev/infiniband/rdma_cm" || bad "/dev/infiniband/rdma_cm missing (rdma_ucm module)"

# ── Container runtime ───────────────────────────────────────────────────────
echo "-- container runtime"
if command -v podman >/dev/null 2>&1; then
    ok "podman $(podman --version 2>/dev/null | awk '{print $NF}')"
    RUNNING="$(podman ps --format '{{.Names}}' 2>/dev/null | tr '\n' ' ')"
    [ -n "$RUNNING" ] && info "running containers: ${RUNNING}"
    if echo "$RUNNING" | grep -qiE 'vllm|ray'; then
        warn "a vLLM/Ray container is running and holds ~100 GB of GPU memory; GPU tests need it stopped first (this script never stops foreign containers)"
    fi
else
    bad "podman missing"
fi

# ── /etc/hosts ──────────────────────────────────────────────────────────────
echo "-- /etc/hosts (no DNS in the RDMA net)"
HOSTS_MISSING=()
for n in 1 2 3 4; do
    if grep -qE "^[[:space:]]*${RDMA_NET}\.${n}[[:space:]]+.*\bStrixHalo${n}\b" /etc/hosts 2>/dev/null; then
        ok "${RDMA_NET}.${n} StrixHalo${n}"
    else
        bad "missing: ${RDMA_NET}.${n} StrixHalo${n}"
        HOSTS_MISSING+=("${RDMA_NET}.${n} StrixHalo${n}")
    fi
done
if [ "${#HOSTS_MISSING[@]}" -gt 0 ] && [ "$APPLY" = 1 ]; then
    need_root
    if confirm "Append ${#HOSTS_MISSING[@]} line(s) to /etc/hosts?"; then
        printf '%s\n' "${HOSTS_MISSING[@]}" | $SUDO tee -a /etc/hosts >/dev/null
        info "applied"
    fi
fi

echo
if [ "$FAIL" = 0 ]; then
    echo "== RESULT: all checks passed on $(hostname)"
else
    echo "== RESULT: some checks FAILED on $(hostname) (see [FAIL] lines; --apply fixes MTU, memlock, /etc/hosts)"
fi
exit "$FAIL"
