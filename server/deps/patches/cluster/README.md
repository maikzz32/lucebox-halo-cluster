# ggml patch series (lucebox-halo-cluster)

`server/deps/llama.cpp` is **vendored**, not a submodule. Every change this
fork makes to ggml is therefore an ordinary edit in the tree and can vanish in
a weekly `merge upstream/main` without anybody noticing. This directory keeps
those changes visible as a regenerable patch series, and
`scripts/cluster/check_ggml_patches.sh` guards them.

## Contents

| File | Topic | Owner |
|---|---|---|
| `0010-ggml-cuda-stream-and-device-accessors.patch` | `ggml_backend_cuda_get_stream()` / `ggml_backend_cuda_get_device_id()` in `ggml-cuda.h` + `ggml-cuda.cu` so `ClusterComm` can enqueue RCCL collectives on the backend's own stream | Agent A (landed) |
| `0020-ggml-cuda-cluster-allreduce-op.patch` | `ggml_cluster_allreduce()` (`ggml.h`/`ggml.c`) and the `GGML_MOE_FUSED_CLUSTER_ALLREDUCE` sub-op in `moe-fused.cu` (template: `GGML_MOE_FUSED_DEFERRED_PEER_COPY`), plus the graph-capture guard in `ggml-cuda.cu`. The node calls a **callback** the server registers, so no collective library becomes a ggml dependency and there is no comm setter to maintain | WP3b (landed) |
| `0090-ggml-misc.patch` | anything else under `server/deps/llama.cpp` that differs from upstream | — |
| `BASE` | the `upstream/main` commit the series was generated against | — |

Patches that would be empty are not written.

## Workflow

```sh
# after editing ggml (or after an upstream merge):
scripts/cluster/check_ggml_patches.sh            # symbols still there?
scripts/cluster/check_ggml_patches.sh --regen    # rewrite *.patch + BASE
scripts/cluster/check_ggml_patches.sh --verify-fresh   # CI: committed series == regenerated
```

`--regen` needs the read-only upstream remote:

```sh
git remote add upstream https://github.com/Luce-Org/lucebox.git
git fetch upstream
```

The patches are generated with `git diff upstream/main -- server/deps/llama.cpp`
(split by path). They are **documentation and a safety net**, not the source
of truth: the vendored tree is what gets built. If an upstream merge conflicts
inside ggml, resolve it in the tree, run the symbol check, then `--regen`.

## Rules for new ggml changes

1. Keep them minimal and upstream-friendly (public accessor + comment, no
   behavioural change for non-cluster builds).
2. Guard cluster-only code with `GGML_USE_HIP`/`GGML_USE_NCCL` or the
   `DFLASH27B_CLUSTER` define as appropriate; a build with the option OFF
   must stay byte-identical to upstream.
3. Add the new symbol to the `check` list in `check_ggml_patches.sh` (level
   `required` once landed) and to the table above.
4. Regenerate the series in the same commit.
