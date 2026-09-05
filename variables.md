# Environment Variables Reference

Summary of `DFLASH_*` / `DFLASH27B_*` environment variables recognized across the
codebase, grouped by subsystem. Most are runtime toggles read via `getenv` /
`os.environ`; a few are build/compile-time or harness knobs (noted where relevant).

> Policy (see `server/docs/ENVIRONMENT.md`): new features should ship as CLI flags
> or defaults. Env vars are reserved for burn-in kill switches and debug
> instrumentation. Treat undocumented variables as internal.

### Status legend

Tags in the tables below flag variables that are **not** part of the intended
long-term serving surface:

- 🐛 **debug** — profiling/telemetry/ablation instrumentation. Zero-cost when
  unset and never required for correct serving; safe to ignore in production.
- 🔀 **kill-switch** — burn-in toggle for a landed default, documented with the
  intent to be deleted once the feature has soaked.
- 🧪 **test/bench** — only read by tests, benchmarks, or harness scripts.
- ⚠️ **removal candidate** — legacy/one-off/likely-obsolete; prefer the CLI flag
  or default where one exists.

Untagged variables are operational tuning knobs.

## Server / runtime configuration

| Variable | Purpose |
|---|---|
| `DFLASH_HOST` | Server bind host. |
| `DFLASH_PORT` | Server bind port. |
| `DFLASH_BIN` / `DFLASH_SERVER_BIN` | Path to the server binary (harness/scripts). |
| `DFLASH_BIN_AR` | Alternate/AR binary path for benchmarks. |
| `DFLASH_DIR` | Base working directory. |
| `DFLASH_SHARE_DIR` | Static/share asset directory served by the HTTP server. |
| `DFLASH_MODEL_CARDS_DIR` | Directory of model-card definitions. |
| `DFLASH_MODEL_NAME` | Model name/identifier. |
| `DFLASH_TOKENIZER` | Tokenizer path/identifier. |
| `DFLASH_TARGET` | Target model path/spec. |
| `DFLASH_DRAFT` | Draft model path/spec. |
| `DFLASH_IMAGE_INFO_PATH` | Path to image/build info metadata. |
| `DFLASH_MAX_CONTEXT` / `DFLASH_MAX_CTX` | Maximum context length. |
| `DFLASH_DEFAULT_MAX_TOKENS` | Default generation token cap. |
| `DFLASH_IGNORE_EOS` | Ignore EOS token during generation. |
| `DFLASH_LAZY` | Lazy model/weight loading. |
| `DFLASH_VERBOSE` | 🐛 **debug** Verbose logging. |

## GPU / backend placement

| Variable | Purpose |
|---|---|
| `DFLASH_TARGET_GPU` / `DFLASH_TARGET_GPUS` | GPU(s) assigned to the target model. |
| `DFLASH_TARGET_LAYER_SPLIT` | Layer-split placement across GPUs for the target. |
| `DFLASH_DRAFT_GPU` | GPU assigned to the drafter. |
| `DFLASH_CUDA_ARCHES` / `DFLASH_HIP_ARCHES` | CUDA/HIP architecture targets (build). |
| `DFLASH27B_GPU_BACKEND` / `DFLASH27B_BACKEND_CUDA` / `DFLASH27B_BACKEND_HIP` | Backend selection (build/compile-time). |
| `DFLASH_HIP_NO_AUTO_UMA` | Disable automatic HIP UMA (unified memory) selection. |
| `DFLASH_HIP_UMA_MIN_FRAC` | Minimum VRAM fraction before HIP UMA kicks in. |
| `DFLASH_WAVE_SIZE` | HIP wave size (compile flag, e.g. gfx1151 needs 32). |

## Speculative decoding — drafter

| Variable | Purpose |
|---|---|
| `DFLASH_DRAFT_KV` | 🔀 **kill-switch** `=0` restores per-step drafter window recompute instead of the ring cache. |
| `DFLASH_DRAFT_PERSIST` | Persist drafter state across steps. |
| `DFLASH_DISABLE_DRAFT_ATTN` | 🐛 **debug** Disable drafter attention block (ablation). |
| `DFLASH_DISABLE_DRAFT_ATTN_GATE` | 🐛 **debug** Disable drafter attention gate (ablation). |
| `DFLASH_DISABLE_DRAFT_AUX_NORMS` | 🐛 **debug** Disable auxiliary norms in the drafter (ablation). |
| `DFLASH_DISABLE_DRAFT_FFN` | 🐛 **debug** Disable drafter FFN block (ablation). |
| `DFLASH_DISABLE_DRAFT_SWA` | 🐛 **debug** Disable drafter sliding-window attention (ablation). |
| `DFLASH_DOMINO_ZERO_START` | Domino head zero-start behavior. |
| `DFLASH27B_DRAFT_FP16` | Load drafter in FP16. |
| `DFLASH27B_DRAFT_SWA` | Enable drafter SWA. |
| `DFLASH27B_DRAFT_CTX_MAX` | Drafter max context. |
| `DFLASH27B_DRAFT_BLOCK_SIZE` / `DFLASH27B_DRAFT_LAYERS` / `DFLASH27B_DRAFT_N_TARGET_LAYERS` / `DFLASH27B_DRAFT_MASK_TOKEN_ID` | Drafter geometry (build/config). |

## Draft IPC transport

| Variable | Purpose |
|---|---|
| `DFLASH_DRAFT_IPC_TRANSPORT` | IPC transport for the draft process. |
| `DFLASH_DRAFT_IPC_SHARED_BYTES` | Shared-memory size for draft IPC. |
| `DFLASH_DRAFT_IPC_RING_CAP` | Ring buffer capacity for draft IPC. |
| `DFLASH_DRAFT_IPC_BIN` | Draft IPC daemon binary. |
| `DFLASH_DRAFT_IPC_GPU` | GPU for the draft IPC daemon. |
| `DFLASH_DRAFT_IPC_WORK_DIR` | Working directory for draft IPC. |

## Verification / sampling

| Variable | Purpose |
|---|---|
| `DFLASH_SAMPLED_VERIFY` | Use sampled verification. |
| `DFLASH_VERIFY_WIDTH` | Verify batch width. |
| `DFLASH_GPU_SAMPLE` | GPU sampling path. |
| `DFLASH_GPU_ARGMAX` / `DFLASH_GPU_VERIFY_ARGMAX` | GPU argmax for sampling/verification. |
| `DFLASH_GPU_DRAFT_TOPK` | GPU draft top-k. |
| `DFLASH_TQ3_VERIFY` | TQ3-quantized verify path. |
| `DFLASH_N_SAMPLE` / `DFLASH_SAMP` | 🧪 **test/bench** Sample count/mode. |
| `DFLASH_SAMPLER_BENCH` | 🧪 **test/bench** Sampler benchmark mode. |
| `DFLASH_SV_DEBUG` | 🐛 **debug** Sampled-verify debug output. |

## Adaptive experts / adaptive verify width

| Variable | Purpose |
|---|---|
| `DFLASH_ADAPTIVE_K_TAU` | Cumulative combine-weight threshold for per-token expert gating (prefer `--adaptive-experts`). |
| `DFLASH_ADAPTIVE_K_DENSE` | CSV of MoE layers kept dense under adaptive-K. |
| `DFLASH_ADAPTIVE_WIDTH_MIN` | Minimum adaptive verify width. |
| `DFLASH_ADAPTIVE_WIDTH_THETA` | Threshold controlling adaptive verify width. |
| `DFLASH_HYBRID_HOT_PCT` | Hot-expert percentage for hybrid MoE. |

## KVFlash (KV cache pager)

| Variable | Purpose |
|---|---|
| `DFLASH_KVFLASH` | Enable KVFlash (prefer CLI `--kvflash`; token count or `auto`). |
| `DFLASH_KVFLASH_DRAFTER` | KVFlash for the drafter cache. |
| `DFLASH_KVFLASH_MAX_POOL` | Max KVFlash pool size. |
| `DFLASH_KVFLASH_POLICY` | KVFlash eviction/placement policy. |
| `DFLASH_KVFLASH_TAU` | KVFlash tau threshold. |
| `DFLASH_LAGUNA_SWA_RING` | 🔀 **kill-switch** `=0` keeps SWA layers on pool-sized caches under KVFlash. |

## KV cache quantization / dtype

| Variable | Purpose |
|---|---|
| `DFLASH_CACHE_TYPE_K` / `DFLASH_CACHE_TYPE_V` | KV cache K/V dtype. |
| `DFLASH_KV_TYPE` | KV cache type selector. |
| `DFLASH_FEATURE_DTYPE` | Draft feature-ring dtype. |
| `DFLASH27B_KV_F16` | F16 KV cache. |
| `DFLASH27B_KV_K` / `DFLASH27B_KV_V` | Per-side (K/V) KV quantization. |
| `DFLASH27B_KV_Q4` / `DFLASH27B_KV_TQ3` / `DFLASH27B_KV_TBQ` | Quantized KV formats (Q4 / TQ3 / TBQ). |
| `DFLASH_PFLASH_K_TYPE` | PFlash K dtype. |

## FlashPrefill / prefill

| Variable | Purpose |
|---|---|
| `DFLASH_FP_USE_BSA` | Use block-sparse attention in flash prefill. |
| `DFLASH_FP_ALPHA` | FlashPrefill alpha parameter. |
| `DFLASH_FP_CHUNK_S` | FlashPrefill chunk size. |
| `DFLASH_FP_NOPE_TAIL` | NoPE tail handling in flash prefill. |
| `DFLASH_FP_HIP_ROW` | HIP row-kernel path for flash prefill. |
| `DFLASH_FP_SKIP_PREWARM` | Skip flash-prefill prewarm. |
| `DFLASH_FP_PROFILE` / `DFLASH_FP_DUMP_COUNTS` / `DFLASH_FP_DEBUG_LAYER0` | 🐛 **debug** FlashPrefill profiling/debug. |
| `DFLASH_PREFILL_MODE` | Prefill mode selector. |
| `DFLASH_PREFILL_THRESHOLD` | Prefill length threshold. |
| `DFLASH_PREFILL_DRAFTER` | Drafter participation during prefill. |
| `DFLASH_PREFILL_KEEP` | Keep prefill cache across requests. |
| `DFLASH_PREFILL_CACHE_SLOTS` / `DFLASH_PREFIX_CACHE_SLOTS` | Optional prefill/prefix cache slot override. When unset, the container preserves the native server defaults (prefix: 32; exact prefill: 0). Set either value to `0` for an explicit opt-out. |
| `DFLASH_PREFILL_CACHE_TEST_LOG` / `DFLASH_PREFILL_CACHE_TEST_PORT` | 🧪 **test/bench** Prefill-cache test harness. |
| `DFLASH27B_LAYER_PREFILL` / `DFLASH27B_PREFILL_UBATCH` | Layer-split prefill / prefill micro-batch. |
| `DFLASH27B_CHUNKED` / `DFLASH27B_CHUNKED_CHUNK` / `DFLASH27B_CHUNKED_Q_BATCH` / `DFLASH27B_CHUNKED_THRESHOLD` | Chunked prefill controls. |
| `DFLASH27B_LAGUNA_CHUNK` | Laguna prefill chunk size. |
| `DFLASH27B_FA_WINDOW` / `DFLASH_FA_WINDOW` | Flash-attention window size. |

## Laguna backend

| Variable | Purpose |
|---|---|
| `DFLASH_LAGUNA_PROFILE` / `DFLASH_LAGUNA_TELEMETRY` | 🐛 **debug** Profiling / telemetry. |
| `DFLASH_LAGUNA_AUTO_HEAD_MAJOR` / `DFLASH_LAGUNA_KV_HEAD_MAJOR` | Head-major KV layout. |
| `DFLASH_LAGUNA_CACHE_SLOTS` | Cache slot count. |
| `DFLASH_LAGUNA_DRAFT_PAD` | Drafter padding. |
| `DFLASH_LAGUNA_DSPARK` / `DFLASH_LAGUNA_DSPARK_TREE` / `DFLASH_LAGUNA_DSPARK_CONFIDENCE_THRESHOLD` | DSpark speculative controls. |
| `DFLASH_LAGUNA_EXPERT_CACHE` | Expert cache toggle. |
| `DFLASH_LAGUNA_FUSED_DOMINO` / `DFLASH_LAGUNA_FUSED_DSPARK` / `DFLASH_LAGUNA_FUSED_QK` / `DFLASH_LAGUNA_FUSE_FFN` / `DFLASH_LAGUNA_MOE_FUSED_COMBINE` | Kernel fusion toggles. |
| `DFLASH_LAGUNA_GPU_ARGMAX` / `DFLASH_LAGUNA_GPU_REMAP` | GPU argmax / expert remap. |
| `DFLASH_LAGUNA_HOTNESS` | Expert hotness tracking. |
| `DFLASH_LAGUNA_LAYER_SPLIT_UBATCH` | Layer-split micro-batch. |
| `DFLASH_LAGUNA_MOE_STUB` | 🐛 **debug** Stub MoE (ablation). |
| `DFLASH_LAGUNA_NEXT_PLACEMENT_OUT` | 🐛 **debug** Dump next-placement plan. |
| `DFLASH_LAGUNA_NO_KVPAD` / `DFLASH_LAGUNA_PAD_CPY` | KV padding controls. |
| `DFLASH_LAGUNA_NO_SINGLE_GRAPH` | Disable single-graph capture. |
| `DFLASH_LAGUNA_PERSIST_VERIFY` | Persist verify graph. |
| `DFLASH_LAGUNA_PREGATE_MAX` / `DFLASH_LAGUNA_PREGATE_TRACE` | Pre-gating max; 🐛 **debug** trace. |
| `DFLASH_LAGUNA_SWAP_MAX` / `DFLASH_LAGUNA_SWAP_MIN_GAIN` | Expert-swap thresholds. |
| `DFLASH_LAGUNA_VERIFY_WIDTH` / `DFLASH_LAGUNA_VERIFY_WIDTH_MAX` | Verify width limits. |
| `DFLASH_LAGUNA_BENCH_NO_LOGITS` | 🧪 **test/bench** Skip logits in benchmarks. |

## Qwen3.5 MoE backend

| Variable | Purpose |
|---|---|
| `DFLASH_QWEN35MOE_CACHE_SLOTS` | Cache slot count. |
| `DFLASH_QWEN35MOE_HOTNESS` | Expert hotness tracking. |
| `DFLASH_QWEN35MOE_SWAP_MAX` / `DFLASH_QWEN35MOE_SWAP_MIN_GAIN` | Expert-swap thresholds. |
| `DFLASH_QWEN35MOE_TELEMETRY` | 🐛 **debug** Telemetry. |
| `DFLASH_QWEN35MOE_NEXT_PLACEMENT_OUT` / `DFLASH_QWEN35MOE_RUNTIME_STATS_OUT` | 🐛 **debug** Dump placement / runtime stats. |
| `DFLASH_QWEN35MOE_NO_KVPAD` / `DFLASH_QWEN35_NO_KVPAD` | KV padding controls. |
| `DFLASH_QWEN35MOE_NO_ROUTED` | 🐛 **debug** Disable routed experts (ablation). |
| `DFLASH_QWEN35MOE_PREFILL_CHUNK` | Prefill chunk size. |
| `DFLASH_QWEN35MOE_HYBRID_SPEC_MIN_ACCEPT_RATE` / `DFLASH_QWEN35MOE_HYBRID_SPEC_MIN_STEPS_BEFORE_AR` | Hybrid speculative acceptance thresholds. |

## Gemma4 backend

| Variable | Purpose |
|---|---|
| `DFLASH_GEMMA4_LAYER_SPLIT_UBATCH` | Layer-split micro-batch. |
| `DFLASH_GEMMA4_NO_KVPAD` | Disable KV padding. |
| `DFLASH_G4_BSA_CHUNK` | Block-sparse attention chunk size. |

## DeepSeek4 (DS4) backend

| Variable | Purpose |
|---|---|
| `DFLASH_DS4_TIMING` | 🐛 **debug** DS4 timing instrumentation. |
| `DFLASH_DS4_CUDA_LAYERS` | Number of DS4 layers on CUDA. |
| `DFLASH_DS4_SPEC` / `DFLASH_DS4_DRAFT` / `DFLASH_DS4_DRAFT_GPU` | Enable the local DSpark drafter, select its GGUF, and choose its HIP device. |
| `DFLASH_DS4_MOE_TP` / `DFLASH_DS4_MOE_TP_INPROC` / `DFLASH_DS4_MOE_TP_GPU` | Burn-in controls for in-process route-owner expert parallelism and the cold-owner HIP device. |
| `DFLASH_DS4_HOTNESS_CSV` | Optional per-layer expert routing profile for hot placement. |
| `DFLASH_MOE_COLD_BACKEND` | Cold-expert compute backend. |
| `DFLASH_NO_PREAD` | Disable pread-based weight loading. |

## MoE expert compute / IPC

| Variable | Purpose |
|---|---|
| `DFLASH_MOE_HYBRID_PREFILL_EAGER` / `DFLASH_MOE_PREFILL_TRACE` | Model-neutral heterogeneous prefill policy and tracing. The legacy `DFLASH_DS4_*` spellings remain aliases. |
| `DFLASH_MOE_TP_GROUPED_MMVQ` / `DFLASH_MOE_TP_FUSED_GATE_UP` | Model-neutral grouped and fused routed-FFN kernel qualification switches. The legacy `DFLASH_DS4_*` spellings remain aliases. |
| `DFLASH_MOE_TP_COARSE_OWNER` / `DFLASH_MOE_TP_COARSE_OWNER_SPLIT` | Model-neutral owner-op lowering switches. The legacy `DFLASH_DS4_*` spellings remain aliases. |
| `DFLASH_MOE_TP_DEVICE_JOIN` / `DFLASH_MOE_TP_ROUTE_PREFORK` | Model-neutral cross-owner scheduling switches. The legacy `DFLASH_DS4_*` spellings remain aliases. |
| `DFLASH_MOE_EXPERT_COMPUTE_THREADS` / `DFLASH_COLD_THREADS` | CPU threads for expert compute. |
| `DFLASH_MOE_EXPERT_COMPUTE_BATCH` / `DFLASH_MOE_EXPERT_COMPUTE_BATCH_MAX` | Expert compute batch sizing. |
| `DFLASH_MOE_EXPERT_COMPUTE_IPC_MODE` | Expert-compute IPC mode. |
| `DFLASH_MOE_EXPERT_COMPUTE_IPC_TRANSPORT` | IPC transport. |
| `DFLASH_MOE_EXPERT_COMPUTE_IPC_SHARED_BYTES` | Shared-memory size. |
| `DFLASH_MOE_EXPERT_COMPUTE_IPC_BATCH_CAPACITY` | IPC batch capacity. |
| `DFLASH_MOE_EXPERT_COMPUTE_IPC_DTYPE` | IPC payload dtype. |
| `DFLASH_MOE_EXPERT_COMPUTE_IPC_PROFILE` | 🐛 **debug** IPC profiling. |
| `DFLASH_MOE_EXPERT_COMPUTE_IPC_BIN` / `DFLASH_MOE_EXPERT_COMPUTE_IPC_GPU` / `DFLASH_MOE_EXPERT_COMPUTE_IPC_WORK_DIR` / `DFLASH_MOE_EXPERT_COMPUTE_IPC_REQUIRED` | IPC daemon binary / GPU / work dir / required flag. |
| `DFLASH_MOE_FIXED_SLOT_GRAPHS` / `DFLASH_MOE_FIXED_SLOT_MAX` | Fixed-slot MoE graph controls. |
| `DFLASH_MOE_PREFILL_HOT_SUB_BATCH` | Hot-expert prefill sub-batch. |
| `DFLASH_MOE_PREFILL_PERSISTENT_OWNER_ALLOC` | Kill switch for persistent long-prefill route and owner arenas. |
| `DFLASH_NO_MOE_ROUTER_FUSE` / `DFLASH_NO_MOE_SWIGLU_FUSE` | Disable router / SwiGLU fusion. |
| `DFLASH_EXPERT_BUDGET_MB` / `DFLASH_EXPERT_BUDGET_PCT` | Expert VRAM budget (absolute / percent). |
| `DFLASH_DROP_COLD` | Drop cold experts. |
| `DFLASH_COLLECT_ROUTING` | 🐛 **debug** Collect routing statistics. |

## Target-shard IPC

| Variable | Purpose |
|---|---|
| `DFLASH_TARGET_SHARD_IPC_TRANSPORT` | Transport for target-shard IPC. |
| `DFLASH_TARGET_SHARD_IPC_SHARED_BYTES` | Shared-memory size for target-shard IPC. |

## Matmul / MMID / MMVQ kernels

| Variable | Purpose |
|---|---|
| `DFLASH_MMID_GROUPED` | Grouped `MUL_MAT_ID` kernel for small verify batches. |
| `DFLASH_MMID_GROUPED_TYPES` | Types eligible for the grouped MMID kernel. |
| `DFLASH_MMID_GROUPED_DEVICE` | Optional device restriction for the grouped MMID kernel. |
| `GGML_CUDA_BATCH_PEER_COPIES` | Batch ordered CUDA/HIP peer copies behind one dependency per source/destination pair. |
| `DFLASH_MMQ_FULL_BATCH_MIN` / `DFLASH_MMQ_SUB_BATCH` | MMQ batch thresholds. |
| `DFLASH_CUDA_MMVQ_TOKENWISE` / `DFLASH_CUDA_MMVQ_MOE_TOKENWISE` / `DFLASH_CUDA_MMVQ_MOE_KERNEL` | MMVQ token-wise / MoE kernel selection. |
| `DFLASH_GDN_FORCE_GROUPED_COLS` / `DFLASH_GDN_NO_GROUPED_COLS` | Gated-delta-net grouped-column control. |
| `DFLASH_NO_MASK` | 🐛 **debug** Disable attention masking (ablation). |

## Top-k kernels

| Variable | Purpose |
|---|---|
| `DFLASH_TOPK_PROFILE` | 🐛 **debug** Top-k kernel profiling. |
| `DFLASH_TOPK_SPLIT` | Top-k split strategy. |
| `DFLASH_TOPK_CASE` / `DFLASH_TOPK_CONSUME` / `DFLASH_TOPK_LAUNCH` | Top-k kernel case/consume/launch tuning. |

## Spark

| Variable | Purpose |
|---|---|
| `DFLASH_SPARK` | Enable Spark. |
| `DFLASH_SPARK_VRAM_MB` | Spark VRAM budget. |
| `DFLASH_SPARK_CLAUDE_DIR` / `DFLASH_SPARK_CODEX_DIR` | Spark corpus directories. |

## KV / context compression

| Variable | Purpose |
|---|---|
| `DFLASH_COMPRESS_NO_PARK` | Disable parking of compressed blocks. |
| `DFLASH_COMPRESS_ANCHOR_RADIUS` / `DFLASH_COMPRESS_MAX_ANCHOR_HITS` | Anchor radius / max hits. |
| `DFLASH_COMPRESS_HEAD_CHUNKS` / `DFLASH_COMPRESS_TAIL_CHUNKS` | Head/tail chunks kept uncompressed. |
| `DFLASH_COMPRESS_QUERY_TOKENS` | Query tokens considered for compression. |
| `DFLASH_COMPRESS_REPEAT_CHUNKS` / `DFLASH_COMPRESS_REPEAT_MIN` / `DFLASH_COMPRESS_REPEAT_MAX` | Repeat-chunk detection bounds. |
| `DFLASH_COMPRESS_POOL_KERNEL` | Pooling kernel for compression. |

## Generation / thinking control

| Variable | Purpose |
|---|---|
| `DFLASH_THINK_MAX` | Max thinking tokens. |
| `DFLASH_THINK_SOFT_CLOSE_MIN_RATIO` | Soft-close ratio for thinking blocks. |
| `DFLASH_DEBUG_THINKING_LOGITS` | 🐛 **debug** Debug thinking logits. |
| `DFLASH_DEGENERATE_RUN_TOKENS` | Degenerate-run token threshold. |
| `DFLASH_STALL_TOOL_PREFIX` | Tool-call stall prefix handling. |
| `DFLASH_MIN_TOKENS` | Minimum generated tokens. |
| `DFLASH_BUDGET` | Token/compute budget. |
| `DFLASH_ANTHROPIC_RAW_SYSTEM` / `DFLASH_ANTHROPIC_RAW_USER` | Pass raw system/user content on the Anthropic-compatible path. |

## Profiling / debug instrumentation

| Variable | Purpose |
|---|---|
| `DFLASH_PROF` | 🐛 **debug** Comma list of profilers (`step,verify,prefill`). |
| `DFLASH_TQ3_VERIFY` | See Verification (also a debug quant path). |
| `DFLASH27B_LM_HEAD_FIX` | ⚠️ **removal candidate** LM-head correctness fix toggle. |
| `DFLASH27B_TESTS` | 🧪 **test/bench** Enable test-only code paths (build). |

## Benchmark / harness

| Variable | Purpose |
|---|---|
| `DFLASH_BENCH_MIX` | 🧪 **test/bench** Benchmark workload mix. |
| `DFLASH_BENCH_SEED` | 🧪 **test/bench** Benchmark RNG seed. |
| `DFLASH_CHUNK` | 🧪 **test/bench** Generic chunk-size knob (bench/scripts). |
| `DFLASH_HAS_CURL` | 🧪 **test/bench** Whether curl is available (scripts). |
| `DFLASH_REQUIRED_ENV` | 🧪 **test/bench** Required-env assertion list (scripts). |
| `DFLASH_SERVER_VERSION` | 🧪 **test/bench** Reported server version (scripts). |

---

### Regenerating

Runtime C/C++ variables can be re-listed with:

```sh
grep -rE 'getenv\("DFLASH[A-Z0-9_]*"\)' server/src
```

See `server/docs/ENVIRONMENT.md` for the canonical generated inventory and the
policy on promoting env vars to CLI flags.

## Cluster (lucebox-halo-cluster)

Multi-node expert-parallel DeepSeek V4 (fork-only, `-DDFLASH27B_CLUSTER=ON`, see
`server/docs/CLUSTER.md`). Following the policy above, **every cluster setting
is a CLI flag** (`--cluster-rank`, `--cluster-size`, `--cluster-head`,
`--cluster-ifname`, `--cluster-ib-hca`, `--cluster-gid-index`,
`--cluster-expert-placement`, `--cluster-replicate-hot`, `--cluster-shared-expert`,
`--cluster-allreduce-dtype`, `--cluster-timeout-ms`, `--cluster-verify-hash`,
`--cluster-selftest`; reference in `server/src/cluster/cluster_config.h`). The
RCCL transport variables (`NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`,
`NCCL_NET_GDR_LEVEL=0`) are *exported by the server* from those flags; values
already present in the environment win. Only the burn-in kill switches and the
trace toggle below are environment variables.

| Variable | Purpose |
|---|---|
| `DFLASH_CLUSTER_NO_INGRAPH_ALLREDUCE` | 🔀 **kill-switch** Disable the in-graph `GGML_MOE_FUSED_CLUSTER_ALLREDUCE` node (path 3b, the default since 2026-09-04) and fall back to the host-enqueued `ClusterComm::allreduce_*` after the FFN graph (path 3a). Measured cost of doing so on two nodes: 29.8 -> 22.4 tok/s with DSpark, 21.5 -> 14.3 AR. Use it to isolate graph-integration bugs; prefill takes path 3a either way. |
| `DFLASH_CLUSTER_NO_GRAPH_CAPTURE` | 🔀 **kill-switch** Never capture HIP graphs for graphs that contain a cluster collective (RCCL under HIP graph capture on gfx1151 is unverified). Default in M1; intended to become opt-in once measured. |
| `DFLASH_DS4_HYBRID_PREFILL_GPU_HC` | ⚡ **performance, opt-in** Compute the hybrid prefill's HC pre/post on the GPU instead of the host. Upstream ships it off; it matters most on a cluster rank, where the hybrid prefill path is mandatory. Two nodes, 12017-token prompt: 96.9 s → 68.6 s (0.95x of a single node). It changes HC in its last bits, so greedy completions differ from a run without it — leave it off when comparing outputs against a single node. |
| `DFLASH_CLUSTER_NO_ATTENTION_PARALLEL` | 🔀 **kill-switch** Keep attention replicated on every rank instead of splitting the 64 heads across them. Head parallelism is on by default where the rank count divides the 8 output groups; measured worth +2.0 % (DSpark) / +1.6 % (AR) on two nodes, with byte-identical output. |
| `DFLASH_CLUSTER_ALLREDUCE_NOOP` | 🐛 **debug** The in-graph all-reduce returns without calling RCCL. **Produces wrong output** (every rank keeps its own partial sum); it exists so a run can be timed with and without the collective. Measured this way: 43 collectives cost 2.7 ms of a 48 ms AR decode step at two ranks. |
| `DFLASH_CLUSTER_GRAPH_CAPTURE` | ⚡ **performance, opt-in** Allow HIP graph capture of a graph containing a cluster collective (blocked by default). Measured: capture is bit-exact on this RCCL build but **slower** — the 5445-node DS4 fused graph replays at 19.15 tok/s AR against 21.5 eager. Kept for the next person who assumes the disabled capture is the problem. |
| `DFLASH_QWEN4EXP_SPEC` | qwen4exp only. Verify the MTP head draft inside the target step: the batch becomes [committed token, draft] and an accepted draft yields two tokens for one pass. Greedy only. 25.1 -> 30.7 tok/s on one node. Needs `DFLASH_QWEN4EXP_MTP`. |
| `DFLASH_QWEN4EXP_SPEC_FORCE_REJECT` | Diagnostic. Reject every draft, so every step pays the wide pass and undoes half of it. Byte-identical output to the accepting run is what says the rollback is exact. |
| `DFLASH_QWEN4EXP_SPEC_NO_CAPTURE` | Cost probe. Wide pass without the per-token states and without rollback; the output drifts. Prices the capture (~4 ms) separately from the second token’s ten routed experts (~8 ms). |
| `DFLASH_QWEN4EXP_STEP_TIMING` | Per-decode-step build/compute split, every 32 steps. |
| `DFLASH_QWEN4EXP_MTP_GROWING_KV` | Give the MTP head its own growing KV history instead of one row. Worth ~6 points of acceptance and, because kv_start is baked into the graph, a rebuild per token: 4 ms of draft becomes 55. |
| `DFLASH_QWEN4EXP_NO_SHARD_SSM` | Keep the delta net whole on a cluster instead of splitting dt_rank across ranks. Measured **worse** (26.0 against 27.2 on two nodes); kept because the opposite is the natural assumption. |
| `DFLASH_CLUSTER_TRACE` | 🐛 **debug** Per-step trace of control-channel messages, collective sizes/latencies and, together with `--cluster-verify-hash n`, the first divergent layer between ranks. Very verbose; never required for serving. |
| `LUCE_MMVF_MAX_NCOLS_F16` | ⚡ **performance** Ceiling on `ne11` below which an F16 `mul_mat` takes ggml's `mul_mat_vec` kernel instead of rocBLAS. Upstream gives gfx1151 the RDNA3 value of 3, measured on discrete RX 7000 cards. At a verify width of 4 that one-column gap is a cliff: rocBLAS serves a tall-skinny F16 weight with a macro tile covering the whole output in a **single workgroup**, about 1/40 of the GPU. DeepSeek V4's per-layer router gate `ffn_gate_inp` is `[4096, 256]` F16 and runs in all 43 layers, so a verify step paid it 43 times. Two nodes, q=4: 30.5 → 38.6 tok/s, verify 113.2 → 84.6 ms/step, output byte-identical. `launch_cluster.sh` sets it to 4; `0` restores the per-architecture value. |
| `DFLASH_CLUSTER_SHARD_DRAFTER` | ⚡ **performance, opt-in** Split the DSpark drafter's routed experts across the ranks. The drafter stays fully resident on every rank -- it is 10.6 GB -- so this splits compute, not storage: each rank evaluates its own routes (`ggml_mul_mat_id` skips negative ids, and the matching route weights are zeroed) and the partial sums are reduced before the replicated shared expert is added. No protocol change is needed: every rank already loads the drafter and runs the draft forward in lockstep, discarding its own tokens for rank 0's broadcast. Measured on the adaptive artifact, byte-identical and 60/60 exact-copy fidelity in every case, acceptance unchanged at 32 steps for 127 tokens: two nodes 42.1 → 43.1 tok/s (draft 9.2 → 7.8 ms/step), four nodes 48.3 → 49.4 (draft 8.7 → 7.3), free prompt at four nodes 19.7 → 20.1. Opt-in because it changes the drafter's arithmetic, and the acceptance rate it feeds is worth more than the milliseconds: verify both before enabling it on a new artifact. |
| `DFLASH_CLUSTER_SPLIT_LM_HEAD` | ⚡ **performance, opt-in** Project only this rank's slice of the DSpark head's vocabulary. Each rank sweeps its share of `lm_head` `[4096, 129280]` and `markov_w2` `[256, 129280]`, the sliced logits are zero-padded to full width and summed across ranks (exact in F32 — each column comes from exactly one rank), and `ggml_argmax` then runs over the full, **unchanged** width. That last part is the design, not a detail: `ggml_argmax` compares with a strict `>`, so ties are settled by the shuffle-butterfly topology at the column count, and reducing per-rank argmaxes would resolve them differently from a single node. `markov_w1` stays whole — `get_rows` indexes it by a global token id without a bounds check. Refused unless the vocabulary divides evenly by the rank count. While active, the private fallback to the non-fused chain is removed: a rank taking it would leave its peers inside a collective until the watchdog fires. Measured, byte-identical and 60/60 fidelity throughout: two nodes 48.2 → 49.4 tok/s (head 4.5 → 2.8 ms), four nodes 55.0 → 56.3 (head 3.9 → 2.1). |
| `DFLASH_DSPARK_NO_CHAIN_GRAPH_CACHE` | 🔀 **kill-switch** Rebuild the DSpark Markov chain graph on every call instead of keeping the last one. The graph depends only on the candidate count, the confidence head and the weights it closes over, all fixed for a loaded model. Keeping it is worth head 4.5 → 4.3 ms/step; the switch exists to isolate a suspected stale-graph problem. |
| `DFLASH_DS4_SKIP_DEAD_INDEXER` | ⚡ **performance, opt-in — NOT safe with the prefix cache** Skip `build_indexer_compressor_step` on the ratio-4 layers of a graph whose only consumer, `build_indexer_topk`, is gated off. The fused verify graph passes neither `SparseFlash` nor f32 array inputs, so during decode this maintains a buffer nothing in that graph reads. Worth 1.7 ms of a 93.3 ms step, 25.40 → 25.75 tok/s on two nodes, byte-identical. **Left opt-in deliberately:** `build_indexer_topk` asserts that the attention and index compressed caches advance together (`deepseek4_graph.cpp:1911`), and skipping the producer during decode makes them diverge. A prefill that resumes a prefix slot after such a decode trips that assert. Use it only with `--prefix-cache-slots 0` and single-turn requests; 0.35 tok/s does not pay for a broken multi-turn path. |
