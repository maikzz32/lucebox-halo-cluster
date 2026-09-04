# qwen4exp — a native Lucebox backend

Target model: `kingjones777/Qwen3.8-Flash-Next-Uncensored-ROCmFP4-STRIX_LEAN-GGUF`,
1224 tensors in 3 shards, 105.7 GB, `general.architecture = qwen4exp`.

## Stage 0 findings (verified against all three shard headers)

Every derived scalar reproduces a real tensor's `ne`. Nine equations, nine passes:

| equation | value |
|---|---|
| `ssm.inner_size + 2*ssm.group_count*ssm.state_size` == `blk.0.ssm_conv1d.ne[1]` | 10240 |
| the same == `blk.0.attn_qkv.ne[1]` | 10240 |
| `ssm.inner_size / ssm.time_step_rank` == `blk.0.ssm_norm.ne[0]` | 128 |
| `hyper_connection.count * embedding_length` == `blk.0.hc_attn_norm.ne[0]` | 10240 |
| `hyper_connection.low_rank` == `hc_attn_down.ne[1]` == `hc_attn_up.ne[0]` | 320 |
| `hyper_connection.count` == `hc_attn_inject.ne[1]` | 4 |
| `2 * sum(rope.dimension_sections)` == `rope.dimension_count` | 64 |
| `rope.dimension_count` <= `attention.key_length` | 64 <= 256 |
| `embedding_length_per_layer_input` == `per_layer_token_embd.ne[0]` | 160 |

`sum(ple.head_vocab_sizes)` is 320,001,446 against `per_layer_token_embd.ne[1]`
of 320,001,536 -- a 90-row difference to resolve in the loader, most likely
padding or an off-by-one in the head-offset table. Not blocking; noted so it is
checked rather than assumed.

## Layer structure

48 layers. Full attention at 3, 7, 11 ... 47 (12 layers, `(il+1) % 4 == 0`,
matching `full_attention_interval = 4`); gated delta net at the other 36.

A full-attention layer carries `attn_q [2560, 12288]`, `attn_k [2560, 512]`,
`attn_v [2560, 512]`, `attn_output [6144, 2560]`, `attn_q_norm/k_norm [256]`,
and four indexer tensors. `attn_q.ne[1] = 12288 = 2 x 6144`, so Q is packed
with a gate exactly as qwen35 packs it (`qwen35_target_graph.cpp:1165-1185`).
`attn_k/v.ne[1] = 512 = 2 kv heads x 256`.

A delta-net layer carries `attn_qkv [2560, 10240]` and the seven ssm tensors.

## What this shares with qwen35 (the reason this is tractable)

`qwen35_target_graph.cpp:1530` computes `conv_channels = ssm_d_inner +
2 * ssm_n_group * ssm_d_state`. With qwen4exp's own hyperparameters that is
6144 + 2*16*128 = 10240 -- exactly `blk.0.ssm_conv1d.ne[1]`. `internal.h`
already carries `ssm_d_conv=4, ssm_d_inner=6144, ssm_d_state=128,
ssm_dt_rank=48, ssm_n_group=16`, `full_attention_interval=4` and
`rope_sections={11,11,10,0}` as defaults. The delta-net block, the full
attention block, m-RoPE and the 512-expert top-10 MoE with a sigmoid-gated
shared expert are all present and generalise.

## What is new

| component | evidence | size |
|---|---|---|
| multi-shard GGUF reader | `grep -ci split` over vendored `gguf.cpp` / `gguf.h` is 0/0 | ~220 lines |
| hyper-connection layer plumbing | `hc_*_norm` REPLACE `attn_norm`/`ffn_norm` -- neither exists in this model | ~450 |
| PLE (layer 1 only) | `ple_key [2560,10240]`, `ple_value [2560,2560]`, `ple_conv1d [4,10240]`, three norms, plus a global `per_layer_token_embd [160, 320001536]` | ~700 |
| QSA indexer | `indexer.{q_proj [2560,512], k_proj [2560,128], q_norm, k_norm [128]}` on the 12 full-attention layers | ~500 |

## Quantisation

772 tensors are type 101 (`Q4_0_ROCMFP4_FAST`), 60 are type 100
(`Q4_0_ROCMFP4`), 388 are F32, plus one each of types 1, 7, 13, 14.

This matters because `mmq.cu:105-118` has no case for type 100. The type-100
tensors are `attn_qkv`, `attn_k` and `attn_v` -- **not** the experts, which are
all 101. So the MoE prefill path keeps MMQ; only the attention projections fall
back. That is the opposite of the worst case.

## Tokenizer and modality

`tokenizer.ggml.model = gpt2`, `pre = qwen35`, bos 248044, eos 248046, pad
248044. `eot_token_id` is absent, which is fine -- eos exists, so generation
terminates. No `v.*` or vision tensors in the main shards, so the model is
loadable text-only; the vision tower ships separately.

## Stage 1: the multi-shard reader

`server/src/common/gguf_shards.h/.cpp`. `GgufShardSet::open()` expands a
`-00001-of-00003.gguf` name into its siblings, opens each with its own
`gguf_context` and `GgufMmap`, and indexes every tensor name across all parts.
`find()` bounds-checks against the file size of the shard that holds the
tensor, never against a sum -- the invariant in `gguf_bounds.h` is
parameterised, not relaxed. A single unsplit file is the N == 1 case, so a
loader can use this unconditionally.

**A finding that corrected the design.** gguf-split does NOT duplicate the
metadata into every part. Measured on this model: part 1 carries 66 key/value
pairs, parts 2 and 3 carry three each (`split.no`, `split.count`,
`split.tensors.count`) and nothing else -- no `general.architecture`. The first
version of the validator compared the architecture across all parts and
rejected the model. It now requires part 1 to declare one and compares only
against parts that do.

Verified against the real model:

```
opened: 3 part(s), 1224 tensors, 98.5 GiB
resolved 1224 tensors, 0 failures, 98.5 GiB of tensor data
per part: part1=115 part2=810 part3=299
absent-tensor probe: rejected as expected
```

1224 matches `split.tensors.count`, and the per-part counts match the header
dump. The probe list in the harness is the full per-layer tensor inventory, so
resolving 1224 of 1224 also confirms the inventory itself.

## Stage 2a: hyperparameters, validated against the weights

`server/src/qwen4exp/qwen4exp_internal.h` and `qwen4exp_loader.cpp`.
`read_qwen4exp_hparams()` reads all 21 `qwen4exp.*` keys and then checks each
derived scalar against an `ne` the file itself carries.

This validation is the point of the stage, not decoration. There is no
reference implementation of this architecture in the tree, so a loader that
mis-reads one dimension produces a model that loads, runs, and is quietly
wrong -- the failure mode with no cheap detector. Every equation names both
numbers when it fails.

Verified against the real model:

```
layers=48  n_embd=2560  heads=24/2  head_dim=256/256
experts=10/512  ff_exp=640  ff_shexp=640
ssm: conv=4 state=128 groups=16 dt_rank=48 inner=6144  (conv_ch=10240)
hc: count=4 low_rank=320   full_attn_interval=4
rope: dims=64 sections=[11,11,10,0] theta=10000000  rms_eps=1e-06
indexer: heads=4 key_len=128 top_k=2048
ple: layer=1 ngram=3 conv=4 per_layer_input=160
ALL SHAPE EQUATIONS PASSED
```

Negative control: forcing `n_hc = 3` yields "the metadata implies 7680 but the
file's tensor says 10240", i.e. rejection rather than silent acceptance.

The per-layer schedule is checked too: for all 48 layers, a layer carries
`attn_q` exactly when `(il + 1) % full_attention_interval == 0` and `attn_qkv`
otherwise. A graph that built the wrong block kind for a layer would be caught
here rather than in the numerics.

`TargetLayer` and `TargetWeights` (`server/src/internal.h`) gained the fields
this architecture needs and no other model has: the eight hyper-connection
tensors per layer, the four indexer tensors, the six PLE tensors, and the
`n_hc` / `hc_low_rank` / indexer / PLE scalars. Every addition is a
nullptr-initialised pointer or a scalar whose default (`n_hc = 1`) means "this
model does not use hyper-connections", so the other backends are unaffected.

## Stage 2b: bind and upload

`load_qwen4exp_gguf()` walks the 48 layers, binds every tensor by name through
the shard set, allocates one device buffer and uploads.

Two design points worth stating. The metadata is **duplicated** into a context
this loader owns rather than borrowed from the shard set: the set has one
`ggml_context` per file and closes when the function returns, while
`TargetWeights` owns exactly one. And `token_embd` and `per_layer_token_embd`
are bound as metadata only, never uploaded -- the first is read a row at a time
by the embedder and the second is tens of GiB of table touched a few kilobytes
per token. That is what keeps the resident set at 62.3 GiB out of a 98.5 GiB
model.

Verified on a Radeon 8060S with the real model:

```
[qwen4exp] loaded 1222 tensors from 3 part(s), 98.5 GiB, 62.3 GiB on device
[qwen4exp] layers=48  full-attn=12  delta-net=36  hc=48  indexer=12  ple=1
[qwen4exp] experts=10/512  ff_exp=640  shared_gate=yes
[qwen4exp] output head=bound  output_hc=bound  tok_embd=metadata  per_layer_embd=metadata
[qwen4exp] readback of blk.0.ffn_gate_exps: 63/64 non-zero bytes -> data present
```

1222 of 1224 tensors bound (the two embedding tables are the exceptions), the
layer counts match the schedule the metadata declares, and a readback from the
device confirms real weights rather than zeros.

## Status

Stages 0, 1, 2a and 2b complete: the model loads. Next is the arch
registration that makes `--model` accept the file through the normal factory
path, and then stage 3, the hyper-connection layer builder -- the first state
that produces a token.

Two things to carry into the next stage:

- `http_server.cpp` derives `speculative_supported` from
  `arch.rfind("qwen", 0) == 0`, which `"qwen4exp"` matches. That line must be
  changed, or `/props` will advertise speculative decode that the capability
  row sets to `kNever`.
- The 90-row difference between `sum(ple.head_vocab_sizes)` (320,001,446) and
  `per_layer_token_embd.ne[1]` (320,001,536) is still unexplained. It does not
  block loading, but PLE cannot be called correct until it is understood.

Note for stage 2b: `http_server.cpp` derives `speculative_supported` from
`arch.rfind("qwen", 0) == 0`, which `"qwen4exp"` matches. That line must be
changed, or `/props` will advertise speculative decode that the capability row
sets to `kNever`.
