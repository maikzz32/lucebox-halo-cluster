# Makefile: single entry point for the common dev/CI ops on lucebox-hub.
#
# Most targets shell out to `uv` and `docker buildx bake`. Pre-release
# software: targets favor simplicity over portability (assumes bash + GNU
# coreutils + a working docker buildx + uv on PATH).
#
# Quick start:
#   make help        # what's available
#   make lint        # ruff check + format check
#   make build       # docker buildx bake cuda12-local --load
#   make serve       # docker run the local image, gemma-4-26b
#   make clean       # drop containers + dangling images

.DEFAULT_GOAL := help
SHELL := /bin/bash

# ── Build args ──────────────────────────────────────────────────────────
# Narrow the CUDA arch list to your local GPU to cut build time 5-6×:
#   make build DFLASH_CUDA_ARCHES=120
DFLASH_CUDA_ARCHES ?= 75;80;86;89;90;120

# Where to mount models into the container.
MODELS_DIR ?= $(HOME)/models

# Image name (local tag the buildx bake produces).
IMAGE ?= lucebox-hub:cuda12

# ── Targets ─────────────────────────────────────────────────────────────

.PHONY: help
help:  ## Show this help message.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: sync
sync:  ## uv sync the workspace (incl. dev extras).
	uv sync --extra dev

.PHONY: lint
lint: sync  ## Ruff check + format-check (no auto-fix).
	uv run ruff check
	uv run ruff format --check

.PHONY: fix
fix: sync  ## Ruff auto-fix + format.
	uv run ruff check --fix
	uv run ruff format

.PHONY: build
build:  ## Build lucebox-hub:cuda12 locally via docker buildx bake.
	DFLASH_CUDA_ARCHES="$(DFLASH_CUDA_ARCHES)" docker buildx bake cuda12-local --load

.PHONY: serve
serve:  ## Run the local image, foreground. Models bind-mounted from $(MODELS_DIR).
	docker run --rm --gpus all -p 8000:8080 \
		-v $(MODELS_DIR):/opt/lucebox-hub/server/models:ro \
		--name lucebox-gemma \
		$(IMAGE) serve

.PHONY: stop
stop:  ## Stop a running lucebox-gemma container.
	-docker stop lucebox-gemma
	-docker rm lucebox-gemma

.PHONY: shell
shell:  ## Drop into a bash shell inside the image (debug).
	docker run --rm -it --gpus all $(IMAGE) shell

.PHONY: ci-trigger-build
ci-trigger-build:  ## Trigger GH Actions docker build+push for the current branch.
	gh workflow run docker.yml --ref "$$(git branch --show-current)" -f push=true
	@echo "view: gh run watch"

.PHONY: clean
clean:  ## Drop stopped containers, dangling images, build cache (~10 GB+).
	-docker container prune -f
	-docker image prune -f
	-docker buildx prune -f --filter "until=24h"

.PHONY: clean-models
clean-models:  ## Remove downloaded models from $(MODELS_DIR). Destructive.
	@# Guard against catastrophic overrides: MODELS_DIR=/ or empty would
	@# rm -rf the host. Also reject $$HOME and other top-level user dirs to
	@# avoid surprising blast radius when someone runs this with
	@# MODELS_DIR=~ in muscle memory.
	@set -eu; \
	  dir='$(MODELS_DIR)'; \
	  if [ -z "$$dir" ]; then \
	    echo "ERROR: MODELS_DIR is empty; refusing to clean." >&2; exit 1; \
	  fi; \
	  resolved=$$(cd "$$dir" 2>/dev/null && pwd -P || echo "$$dir"); \
	  case "$$resolved" in \
	    /|/home|/root|/Users|"$$HOME"|/usr|/etc|/var|/opt|/bin|/sbin|/lib|/lib64|/boot|/dev|/proc|/sys|/tmp) \
	      echo "ERROR: refusing to rm -rf $$resolved/*" >&2; exit 1 ;; \
	  esac; \
	  if [ ! -d "$$resolved" ]; then \
	    echo "MODELS_DIR=$$resolved does not exist; nothing to clean."; exit 0; \
	  fi; \
	  echo "WARN: about to rm -rf $$resolved/*"; \
	  read -r -p "Continue? [y/N] " ans; \
	  [ "$$ans" = "y" ] || { echo "Aborted."; exit 0; }; \
	  rm -rf -- "$$resolved"/*

# ── Cluster (lucebox-halo-cluster) ───────────────────────────────────────
# Multi-node expert-parallel DeepSeek V4 across 2-4 Strix Halo nodes.
# Full guide: server/docs/CLUSTER.md. The scripts talk to the nodes over ssh
# aliases (strix1..strix4) and never touch containers they did not start.
DFLASH_HIP_ARCHES ?= gfx1151
CLUSTER_HOSTS   ?= strix1 strix2
CLUSTER_IMAGE   ?= ghcr.io/maikzz32/lucebox-halo-cluster:dev
CLUSTER_MODELS  ?= /home/maik/gguf/ds4
CLUSTER_TARGET  ?= DeepSeek-V4-Flash-0731-ROCMFPX-MIX-STRIX.gguf
CLUSTER_DRAFT   ?= DeepSeek-V4-Flash-0731-DSpark-draft-Q4RMFP4-denseF16.gguf

.PHONY: build-rocm-cluster
build-rocm-cluster:  ## Bake the ROCm 10 / RCCL cluster image (Dockerfile.rocm-cluster) locally.
	DFLASH_HIP_ARCHES="$(DFLASH_HIP_ARCHES)" docker buildx bake rocm-cluster-local --load

.PHONY: cluster-host-check
cluster-host-check:  ## Check-only host prerequisites on every CLUSTER_HOSTS node (RDMA, MTU, kernel cmdline, ...).
	@for h in $(CLUSTER_HOSTS); do \
	  echo "== $$h"; \
	  ssh "$$h" 'bash -s' < scripts/cluster/host_prep.sh || true; \
	done

.PHONY: cluster-fetch-models
cluster-fetch-models:  ## Resumable, sha256-verified download of target + DSpark GGUF into CLUSTER_MODELS on the first host.
	ssh "$(firstword $(CLUSTER_HOSTS))" 'bash -s -- --dir $(CLUSTER_MODELS)' < scripts/cluster/fetch_models.sh

.PHONY: cluster-baseline
cluster-baseline:  ## all_reduce_perf 8K..64M over CLUSTER_HOSTS inside the running lucebox-rank* containers.
	scripts/cluster/rccl_baseline.sh "$(CLUSTER_HOSTS)"

.PHONY: cluster-up
cluster-up:  ## Start rank 0 + workers (podman on each host) with the DS4 target/DSpark GGUFs.
	IMAGE="$(CLUSTER_IMAGE)" MODELS_DIR="$(CLUSTER_MODELS)" \
	  scripts/cluster/launch_cluster.sh "$(CLUSTER_HOSTS)" "$(CLUSTER_TARGET)" "$(CLUSTER_DRAFT)"

.PHONY: cluster-status
cluster-status:  ## podman ps + log tail of lucebox-rank* on every host.
	scripts/cluster/launch_cluster.sh --status "$(CLUSTER_HOSTS)"

.PHONY: cluster-down
cluster-down:  ## Stop and remove only the lucebox-rank* containers on every host.
	scripts/cluster/launch_cluster.sh --down "$(CLUSTER_HOSTS)"
