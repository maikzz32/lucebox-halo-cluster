#!/usr/bin/env bash
# fetch_models.sh - download the DeepSeek V4 Flash target GGUF and its DSpark
# drafter (resumable), verify sha256, and optionally copy the directory to the
# other nodes over the RoCE net.
#
# Run ON a node (or `ssh strix1 'bash -s -- --dir /home/maik/gguf/ds4' <
# scripts/cluster/fetch_models.sh`). Every rank needs identical files at the
# identical path; the head refuses workers whose model_sha differs.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: fetch_models.sh [--dir <models-dir>] [--no-verify] [--sync-to "strix2 strix3 ..."] [--help]

  --dir <d>        target directory (default /home/maik/gguf/ds4)
  --no-verify      skip the sha256 check (verification of 109 GB takes ~5-10 min)
  --sync-to <l>    after download/verify, rsync the directory to these ssh
                   aliases over 192.168.100.x (alias strixN -> 192.168.100.N;
                   other aliases are used as-is)
  --only-sync      do not download, just --sync-to
  --help

Files (verified 2026-09-03, use verbatim):
  Lucebox/DeepSeek-V4-Flash-0731-ROCmFP3 (same file also in Lucebox/DeepSeek-V4-Flash-0731-ROCMFPX)
    DeepSeek-V4-Flash-0731-ROCMFPX-MIX-STRIX.gguf   98,294,917,184 bytes
    sha256 7c0789d190fdd2acad93255825822ca276f29d13f9410f2ac65f5f7a542b0a38
  Lucebox/DeepSeek-V4-Flash-0731-DSpark-GGUF
    DeepSeek-V4-Flash-0731-DSpark-draft-Q4RMFP4-denseF16.gguf   10,648,656,160 bytes
    sha256 58e7337597a917fdd033aca30a38263740dc53999dac187a676a36b8daf9e63d

Env: HF_TOKEN (optional, sent as Bearer header), RDMA_NET (default 192.168.100),
     HF_BASE (default https://huggingface.co).
USAGE
}

DIR=/home/maik/gguf/ds4; VERIFY=1; SYNC_TO=""; DOWNLOAD=1
while [ $# -gt 0 ]; do
    case "$1" in
        --dir) DIR="$2"; shift ;;
        --no-verify) VERIFY=0 ;;
        --sync-to) SYNC_TO="$2"; shift ;;
        --only-sync) DOWNLOAD=0 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done
RDMA_NET="${RDMA_NET:-192.168.100}"
HF_BASE="${HF_BASE:-https://huggingface.co}"

# name|repo|bytes|sha256
FILES=(
  "DeepSeek-V4-Flash-0731-ROCMFPX-MIX-STRIX.gguf|Lucebox/DeepSeek-V4-Flash-0731-ROCmFP3|98294917184|7c0789d190fdd2acad93255825822ca276f29d13f9410f2ac65f5f7a542b0a38"
  "DeepSeek-V4-Flash-0731-DSpark-draft-Q4RMFP4-denseF16.gguf|Lucebox/DeepSeek-V4-Flash-0731-DSpark-GGUF|10648656160|58e7337597a917fdd033aca30a38263740dc53999dac187a676a36b8daf9e63d"
)

fsize() { stat -c %s "$1" 2>/dev/null || echo 0; }

if [ "$DOWNLOAD" = 1 ]; then
    mkdir -p "$DIR"
    AUTH=()
    [ -n "${HF_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer ${HF_TOKEN}")
    for entry in "${FILES[@]}"; do
        IFS='|' read -r name repo bytes sha <<<"$entry"
        dst="$DIR/$name"
        url="${HF_BASE}/${repo}/resolve/main/${name}"
        have="$(fsize "$dst")"
        if [ "$have" = "$bytes" ]; then
            echo "== $name: complete (${bytes} bytes)"
        else
            echo "== $name: have ${have}/${bytes} bytes, resuming from ${url}"
            # -C - resumes; --retry survives HF CDN hiccups; -L follows the CDN redirect.
            curl -L -C - --fail --retry 20 --retry-delay 10 --retry-all-errors \
                 "${AUTH[@]}" -o "$dst" "$url"
            have="$(fsize "$dst")"
            [ "$have" = "$bytes" ] || { echo "ERROR: $name has ${have} bytes, expected ${bytes}" >&2; exit 1; }
        fi
        if [ "$VERIFY" = 1 ]; then
            echo "-- sha256 $name ..."
            got="$(sha256sum "$dst" | awk '{print $1}')"
            if [ "$got" = "$sha" ]; then echo "   OK ${got}"; else
                echo "ERROR: sha256 mismatch for $name: got ${got}, expected ${sha}; delete the file and rerun" >&2; exit 1; fi
        fi
    done
    echo "== all files present in $DIR"
    ls -la "$DIR"
fi

if [ -n "$SYNC_TO" ]; then
    for h in $SYNC_TO; do
        if [[ "$h" =~ ^strix([0-9]+)$ ]]; then ip="${RDMA_NET}.${BASH_REMATCH[1]}"; else ip="$h"; fi
        echo "== rsync $DIR/ -> ${h} (${ip}):$DIR/  (over the RoCE net, ~15 min for 109 GB at 25 GbE)"
        ssh -o BatchMode=yes "$ip" "mkdir -p '$DIR'"
        rsync -a --partial --inplace --info=progress2 "$DIR/" "${ip}:$DIR/"
        if [ "$VERIFY" = 1 ]; then
            echo "-- remote sha256 on $h"
            for entry in "${FILES[@]}"; do
                IFS='|' read -r name repo bytes sha <<<"$entry"
                got="$(ssh -o BatchMode=yes "$ip" "sha256sum '$DIR/$name'" | awk '{print $1}')"
                [ "$got" = "$sha" ] && echo "   OK $name" || { echo "ERROR: $h:$DIR/$name sha256 ${got} != ${sha}" >&2; exit 1; }
            done
        fi
    done
fi
