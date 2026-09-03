#!/usr/bin/env bash
# check_ggml_patches.sh - keep the fork's ggml modifications visible and
# regenerable as a patch series (server/deps/patches/cluster/*.patch).
#
# ggml is vendored (server/deps/llama.cpp, not a submodule), so weekly upstream
# merges can silently drop a hunk. This script (a) greps for every symbol the
# cluster code depends on and (b) regenerates the patch series from
# `git diff upstream/main -- server/deps/llama.cpp`.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: check_ggml_patches.sh [--regen] [--verify-fresh] [--upstream <ref>] [--help]

  (default)        verify the patched symbols exist in the vendored ggml (grep)
  --regen          regenerate server/deps/patches/cluster/*.patch from
                   `git diff <upstream> -- server/deps/llama.cpp` (needs the
                   `upstream` remote: git remote add upstream https://github.com/Luce-Org/lucebox.git)
  --verify-fresh   regenerate into a temp dir and fail if it differs from the
                   committed series (CI use)
  --upstream <ref> diff base (default upstream/main)

Exit: 0 ok, 1 missing required symbol / stale series, 2 usage.
USAGE
}

REGEN=0; VERIFY=0; UPSTREAM=upstream/main
while [ $# -gt 0 ]; do
    case "$1" in
        --regen) REGEN=1 ;;
        --verify-fresh) VERIFY=1 ;;
        --upstream) UPSTREAM="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
GGML="$REPO/server/deps/llama.cpp/ggml"
PATCHES="$REPO/server/deps/patches/cluster"
FAIL=0

# ── symbol checks ───────────────────────────────────────────────────────────
# REQUIRED: landed in this fork; missing means an upstream merge dropped it.
# PLANNED : owned by other work packages (WP3b in-graph all-reduce); reported
#           as [todo] until they land, then move them up.
check() { # check <level> <file> <regex> <description>
    local level="$1" file="$2" re="$3" desc="$4"
    if [ -f "$file" ] && grep -qE -- "$re" "$file"; then
        printf '  [ OK ] %s\n' "$desc"
    elif [ "$level" = required ]; then
        printf '  [FAIL] %s  (%s: /%s/)\n' "$desc" "${file#"$REPO"/}" "$re"; FAIL=1
    else
        printf '  [todo] %s  (%s)\n' "$desc" "${file#"$REPO"/}"
    fi
}
echo "== ggml patch symbols ($GGML)"
check required "$GGML/include/ggml-cuda.h" \
    'void \* ggml_backend_cuda_get_stream\(ggml_backend_t backend\);' \
    'ggml_backend_cuda_get_stream declared'
check required "$GGML/include/ggml-cuda.h" \
    'int ggml_backend_cuda_get_device_id\(ggml_backend_t backend\);' \
    'ggml_backend_cuda_get_device_id declared'
check required "$GGML/src/ggml-cuda/ggml-cuda.cu" \
    '^void \* ggml_backend_cuda_get_stream\(ggml_backend_t backend\)' \
    'ggml_backend_cuda_get_stream defined'
check required "$GGML/src/ggml-cuda/ggml-cuda.cu" \
    '^int ggml_backend_cuda_get_device_id\(ggml_backend_t backend\)' \
    'ggml_backend_cuda_get_device_id defined'
check planned "$GGML/src/ggml-cuda/moe-fused.cuh" \
    'GGML_MOE_FUSED_CLUSTER_ALLREDUCE' \
    'GGML_MOE_FUSED_CLUSTER_ALLREDUCE sub-op (WP3b)'
check planned "$GGML/src/ggml-cuda/cluster-allreduce.cu" \
    'ggml_cuda_cluster_allreduce|cluster_allreduce' \
    'ggml-cuda/cluster-allreduce.cu (WP3b)'
check planned "$GGML/include/ggml-cuda.h" \
    'ggml_backend_cuda_set_cluster_comm|ggml_backend_cuda_cluster_set_comm' \
    'comm setter for the in-graph all-reduce (WP3b)'

# ── regenerate / verify the series ──────────────────────────────────────────
gen_series() { # gen_series <outdir>
    local out="$1"
    git -C "$REPO" rev-parse --verify --quiet "$UPSTREAM" >/dev/null \
        || { echo "ERROR: $UPSTREAM not found; git fetch upstream" >&2; return 1; }
    mkdir -p "$out"
    rm -f "$out"/*.patch
    local base; base="$(git -C "$REPO" rev-parse "$UPSTREAM")"
    # Topic split by path; everything else lands in 0090-misc.
    local -a p1=(server/deps/llama.cpp/ggml/include/ggml-cuda.h server/deps/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu)
    local -a p2=(server/deps/llama.cpp/ggml/src/ggml-cuda/moe-fused.cu server/deps/llama.cpp/ggml/src/ggml-cuda/moe-fused.cuh
                 server/deps/llama.cpp/ggml/src/ggml-cuda/cluster-allreduce.cu server/deps/llama.cpp/ggml/src/ggml-cuda/cluster-allreduce.cuh
                 server/deps/llama.cpp/ggml/src/ggml-cuda/CMakeLists.txt server/deps/llama.cpp/ggml/src/ggml-hip/CMakeLists.txt)
    local -a excl=()
    for f in "${p1[@]}" "${p2[@]}"; do excl+=(":(exclude)$f"); done
    git -C "$REPO" diff --no-color "$base" -- "${p1[@]}" >"$out/0010-ggml-cuda-stream-and-device-accessors.patch"
    git -C "$REPO" diff --no-color "$base" -- "${p2[@]}" >"$out/0020-ggml-cuda-cluster-allreduce-op.patch"
    git -C "$REPO" diff --no-color "$base" -- server/deps/llama.cpp "${excl[@]}" >"$out/0090-ggml-misc.patch"
    for f in "$out"/*.patch; do [ -s "$f" ] || rm -f "$f"; done
    printf '%s\n' "$base" >"$out/BASE"
    echo "-- series regenerated against $UPSTREAM ($base):"; ls -1 "$out"
}

if [ "$REGEN" = 1 ]; then
    gen_series "$PATCHES"
fi
if [ "$VERIFY" = 1 ]; then
    TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
    gen_series "$TMP" >/dev/null
    for f in "$TMP"/*.patch; do
        n="$(basename "$f")"
        if [ ! -f "$PATCHES/$n" ]; then echo "  [FAIL] $n missing in $PATCHES (run --regen)"; FAIL=1; continue; fi
        if ! diff -q "$f" "$PATCHES/$n" >/dev/null; then echo "  [FAIL] $n is stale (run --regen)"; FAIL=1; else echo "  [ OK ] $n up to date"; fi
    done
    for f in "$PATCHES"/*.patch; do
        [ -f "$f" ] || continue
        [ -f "$TMP/$(basename "$f")" ] || { echo "  [FAIL] $(basename "$f") no longer produced by the diff (delete it)"; FAIL=1; }
    done
fi

if [ "$FAIL" = 0 ]; then echo "== OK"; else echo "== FAILED"; fi
exit "$FAIL"
