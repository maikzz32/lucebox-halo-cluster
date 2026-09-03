#!/usr/bin/env bash
# build_in_container.sh - dev loop: sync this checkout to one Strix Halo node
# and build the cluster targets there inside an ephemeral ROCm 10 container.
#
# There is no ROCm on the hosts (no hipcc, cmake only on strix3), so the build
# runs in the dev-rocm10 image with the repo bind-mounted. The build directory
# stays on the host (~/lucebox-halo-cluster/build-cluster) so rebuilds are
# incremental. No GPU is needed to build or to run --tests-only; the vLLM
# cluster can keep running.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: build_in_container.sh [node] [options]

  node               ssh alias of the build node (default strix1)
  --tests-only       build + run test_cluster_unit and test_feature_gate only (no GPU)
  --no-sync          skip the source sync (rebuild what is already on the node)
  --configure        force a fresh cmake configure (deletes CMakeCache.txt)
  --clean            delete the remote build dir first
  --shell            open an interactive shell in the build container instead
  --help

Environment:
  IMAGE     build image (default ghcr.io/maikzz32/strix-vllm-gfx1151:dev-rocm10;
            the lucebox-halo-cluster image also works since it contains the toolchain)
  JOBS      parallel compile jobs (default 6 with nice -n 10; the nodes have 32
            cores, ~68 GiB RAM available even with vLLM running - HIP compiles
            of deepseek4_graph.cpp are fat, raise JOBS when vLLM is stopped)
  REMOTE    remote checkout dir (default ~/lucebox-halo-cluster)
  BUILD_DIR remote build dir     (default $REMOTE/build-cluster)
  TARGETS   cmake targets        (default dflash_server test_cluster_unit test_feature_gate)
  HIP_ARCHES (default gfx1151)

Sync: rsync (preferred, incremental) or, when rsync is missing locally (Git
Bash on Windows), a tar pipe over ssh (full copy). .git/ and build*/ are never
copied; the vendored ggml under server/deps/llama.cpp is.
USAGE
}

NODE=strix1; TESTS_ONLY=0; SYNC=1; RECONF=0; CLEAN=0; SHELL_MODE=0
for a in "$@"; do
    case "$a" in
        --tests-only) TESTS_ONLY=1 ;;
        --no-sync) SYNC=0 ;;
        --configure) RECONF=1 ;;
        --clean) CLEAN=1 ;;
        --shell) SHELL_MODE=1 ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "unknown option $a" >&2; usage >&2; exit 2 ;;
        *) NODE="$a" ;;
    esac
done

IMAGE="${IMAGE:-ghcr.io/maikzz32/strix-vllm-gfx1151:dev-rocm10}"
JOBS="${JOBS:-6}"
REMOTE="${REMOTE:-~/lucebox-halo-cluster}"
BUILD_DIR="${BUILD_DIR:-$REMOTE/build-cluster}"
HIP_ARCHES="${HIP_ARCHES:-gfx1151}"
if [ "$TESTS_ONLY" = 1 ]; then
    TARGETS="${TARGETS:-test_cluster_unit test_feature_gate}"
else
    TARGETS="${TARGETS:-dflash_server test_cluster_unit test_feature_gate}"
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10)

echo "== node=${NODE} image=${IMAGE} jobs=${JOBS} targets=${TARGETS}"

# ── sync ────────────────────────────────────────────────────────────────────
EXCLUDES=(--exclude .git --exclude '.git/' --exclude 'build' --exclude 'build-*' --exclude '.venv'
          --exclude '__pycache__' --exclude '*.gguf' --exclude '*.safetensors' --exclude '.claude'
          --exclude 'server/models' --exclude 'server/deps/Block-Sparse-Attention/.git')
if [ "$SYNC" = 1 ]; then
    "${SSH[@]}" "$NODE" "mkdir -p $REMOTE/src $BUILD_DIR"
    if command -v rsync >/dev/null 2>&1 && "${SSH[@]}" "$NODE" 'command -v rsync >/dev/null'; then
        echo "-- rsync $REPO/ -> $NODE:$REMOTE/src/"
        rsync -az --delete "${EXCLUDES[@]}" -e "ssh -o BatchMode=yes" "$REPO/" "$NODE:$REMOTE/src/"
    else
        echo "-- tar pipe (rsync unavailable): full copy of $REPO -> $NODE:$REMOTE/src/"
        (cd "$REPO" && tar "${EXCLUDES[@]}" -cf - .) | "${SSH[@]}" "$NODE" "tar -xf - -C $REMOTE/src"
    fi
fi

# ── remote build ────────────────────────────────────────────────────────────
# PATH: /usr/bin BEFORE /opt/rocm/lib/llvm/bin, otherwise CMake takes ROCm's
# clang++ as host compiler and the link fails with "unable to find library
# -lstdc++" (verified in the dev-rocm10 image). With gcc/g++ as host compiler
# the HIP link additionally needs CMAKE_POSITION_INDEPENDENT_CODE=ON and
# CMAKE_HIP_FLAGS=-fPIC, otherwise ld.lld reports "relocation R_X86_64_32S
# cannot be used against local symbol; recompile with -fPIC" (verified).
# Flags only; the image and bash's arguments are appended per use. The
# entrypoint is reset to bash because the cluster image's ENTRYPOINT is
# dflash_server. -i keeps stdin open so the build script can be piped in.
PODMAN_FLAGS="podman run --rm -i --security-opt label=disable --network host --pids-limit -1 \
  -v $REMOTE/src:/src -v $BUILD_DIR:/src/server/build-cluster \
  -e ROCM_PATH=/opt/rocm -e PATH=/usr/bin:/usr/local/bin:/bin:/opt/rocm/bin:/opt/rocm/lib/llvm/bin \
  -w /src --entrypoint bash"

if [ "$SHELL_MODE" = 1 ]; then
    exec ssh -t "$NODE" "$PODMAN_FLAGS -t $IMAGE"
fi

INNER=$(cat <<EOF
set -euo pipefail
echo "-- toolchain: \$(hipcc --version 2>/dev/null | head -1 || echo 'hipcc missing')"
# dev-rocm10 has no cmake and may lack the host C++ dev packages; the
# lucebox-halo-cluster image has everything (dnf is a no-op there).
if ! command -v cmake >/dev/null 2>&1 || ! command -v g++ >/dev/null 2>&1 || [ ! -f /usr/include/c++/*/iostream ]; then
    echo "-- installing host toolchain bits (gcc-c++ libstdc++-devel glibc-devel cmake ninja-build)"
    if ! dnf install -y --setopt=install_weak_deps=False gcc-c++ libstdc++-devel glibc-devel cmake ninja-build >/dev/null 2>&1; then
        dnf install -y --setopt=install_weak_deps=False gcc-c++ libstdc++-devel glibc-devel ninja-build >/dev/null 2>&1 || true
        command -v cmake >/dev/null 2>&1 || python3 -m pip install --no-cache-dir --quiet cmake
    fi
    command -v cmake >/dev/null 2>&1 || { echo "cannot install cmake"; exit 1; }
fi
cmake --version | head -1
B=/src/server/build-cluster
if [ "$CLEAN" = 1 ]; then rm -rf "\$B"/*; fi
if [ "$RECONF" = 1 ]; then rm -f "\$B/CMakeCache.txt"; fi
if [ ! -f "\$B/build.ninja" ] || [ ! -f "\$B/CMakeCache.txt" ]; then
    nice -n 10 cmake -S /src/server -B "\$B" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH=/opt/rocm \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_HIP_FLAGS=-fPIC \
        -DDFLASH27B_GPU_BACKEND=hip \
        -DDFLASH27B_HIP_ARCHITECTURES="$HIP_ARCHES" \
        -DDFLASH27B_CLUSTER=ON \
        -DGGML_HIP_GRAPHS=ON \
        -DDFLASH27B_ROCMFP2_AFFINE=ON \
        -DDFLASH27B_ENABLE_BSA=OFF \
        -DDFLASH27B_FA_ALL_QUANTS=OFF
fi
nice -n 10 cmake --build "\$B" -j "$JOBS" --target $TARGETS
if [ "$TESTS_ONLY" = 1 ]; then
    echo "-- running GPU-free unit tests"
    "\$B/test_cluster_unit"
    "\$B/test_feature_gate"
    echo "-- tests passed"
fi
ls -la "\$B"/dflash_server "\$B"/test_cluster_unit "\$B"/test_feature_gate 2>/dev/null || true
EOF
)

echo "-- building on $NODE in $IMAGE (build dir $BUILD_DIR)"
"${SSH[@]}" "$NODE" "$PODMAN_FLAGS $IMAGE -s" <<<"$INNER"
echo "== done. binaries: $NODE:$BUILD_DIR/{dflash_server,test_cluster_unit,test_feature_gate}"
echo "   (they link against /opt/rocm inside the image; run them in a container from $IMAGE, e.g. --shell)"
