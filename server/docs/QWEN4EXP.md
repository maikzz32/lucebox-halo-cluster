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

## Stage 2c: registration

`Qwen4ExpBackend` extends `Qwen35Backend` and overrides only the loader and
the banner, the same shape bailingmoe3 uses. The capability row is all
`false`/`kNever`: one sequence, no speculative decode, no paged serving, no
expert offload. Those come after the autoregressive path has a baseline to be
measured against.

`http_server.cpp` derived `speculative_supported` from
`arch.rfind("qwen", 0) == 0`. `"qwen4exp"` matches that prefix, so `/props`
would have advertised speculative decode that the capability row refuses. It
now asks the table instead of the name.

The server starts:

```
[backend_factory] detected arch=qwen4exp
[tokenizer] loaded vocab=248320 merges=247587 bos=248044 eos=248046 pre=qwen35
[qwen4exp] loaded 1222 tensors from 3 part(s), 98.5 GiB, 62.3 GiB on device
[server] listening on http://127.0.0.1:18120
```

## Stage 3a: the hyper-connection pair

`server/src/qwen4exp/qwen4exp_graph.{h,cpp}`. Two functions in pure ggml, no
new kernel:

    cur, inject = hc_mix(state, norm, down, up, inject_w)   // [n_embd, T]
    cur         = <ordinary attention / delta-net / FFN block>
    state       = hc_combine(state, cur, inject)            // [n_embd, n_hc, T]

`hc_mix` runs a grouped RMSNorm -- the reduction is over ONE stream, then the
flat `[n_hc*n_embd]` gamma scales all of them -- gates the flattened state with
`sigmoid(up(silu(down(xn)/n_hc)))`, and collapses the streams by their mean.
`hc_combine` writes a block's output back into every stream weighted by
`2*sigmoid(inject/n_hc)`; the factor 2 centres that on 1, so a zero injection
is a plain residual add rather than a halving.

**Where the semantics came from.** Not from the tensor shapes: they do not say
how four streams collapse to one, and a guess would have produced a model that
runs and cannot be shown wrong. qwen4exp is in upstream llama.cpp
(ggml-org/llama.cpp#27742, merged 2026-08-27), and that implementation is the
specification this follows.

Verified numerically on the CPU backend:

| check | error |
|---|---|
| `hc_combine` with zero injection equals `state + block_out` in every stream | 0 |
| `hc_mix` stream collapse equals `0.5 * rms_norm(stream)` when the gate is forced to 0.5 | 5.7e-08 |

The second is the sharper one: it forces `w_down = 0` so the gate is
`sigmoid(0) = 0.5` everywhere and makes all four streams identical, which turns
the expected value into something computable by hand. A wrong stride in the
collapse loop would show up as a factor or as the wrong stream.

## Status

Stages 0, 1, 2 and 3a complete. `--model` accepts the file, the weights land on
the device, and the server listens.

**It cannot generate correct text yet, and this is the point to be careful
about.** The hyper-connection pair is written and verified, but nothing calls
it: the graph is still qwen35's, which expects one residual stream of width
2560 where this model carries four. A request would produce tokens, and they
would be wrong.

Stage 3b is the wiring -- expose qwen35's `build_delta_net_block` and
`build_full_attn_block`, then drive them from a loop that carries the
`[n_embd, n_hc, T]` state. The blocks themselves need no change: they see the
same `[n_embd, T]` they always did. The danger in that stage is that its output
will *sound* fine long before it is right, which is why the per-layer RMS check
and a recorded quality baseline belong with it rather than after it.

Two open items:

- The 90-row difference between `sum(ple.head_vocab_sizes)` (320,001,446) and
  `per_layer_token_embd.ne[1]` (320,001,536) is unexplained. It does not block
  loading, but PLE cannot be called correct until it is understood.
- 60 tensors are quant type 100, which `mmq.cu` has no case for: `attn_qkv`,
  `attn_k` and `attn_v`. The experts are all type 101 and keep MMQ, so this
  costs prefill speed on the attention projections only.

Note for stage 2b: `http_server.cpp` derives `speculative_supported` from
`arch.rfind("qwen", 0) == 0`, which `"qwen4exp"` matches. That line must be
changed, or `/props` will advertise speculative decode that the capability row
sets to `kNever`.


## Stage 4: PLE — the specification, extracted

PLE is the one piece whose semantics cannot be recovered from tensor shapes,
and it is what stands between the current output and correct text. The
reference is `src/models/qwen4exp.cpp` in upstream llama.cpp. Recorded here so
the implementation has something to be checked against.

### The row indices are an n-gram hash, computed on the host

`ple_n_heads = (ngram_size - 1) * heads_per_ngram` = (3-1)*8 = **16**, which is
exactly the count of `head_vocab_sizes` and `head_offsets`. For each token:

```
ctx[0] = token[i]
cut = false
for s in 1 .. n_gram-1:
    t   = predecessor s positions back      # from the KV cells; missing reads as EOS
    cut = cut or t < 0 or t == eos          # an EOS resets everything at or before it
    ctx[s] = cut ? eos : t

for n in 2 .. n_gram:
    mixed = ctx[0] * layer_multipliers[0]
    for j in 1 .. n-1:
        mixed ^= ctx[j] * layer_multipliers[j]
    base = (n-2) * heads_per_ngram
    for g in 0 .. heads_per_ngram-1:
        h = base + g
        idx[i*16 + h] = mixed % head_vocab_sizes[h] + head_offsets[h]
```

The token's own EOS does not cut its own context. Tokens shared by several
sequences are rejected outright, because their predecessors would be
ambiguous.

### The graph

```
emb    = get_rows(per_layer_token_embd, idx)      # [160, 16*T] -> [2560, T]
key    = ple_key   * emb                          # [n_hc*n_embd, T]
value  = ple_value * emb                          # [n_embd, T]
key    = grouped_norm(key,    ple_norm_key)       # rms over ONE stream, flat gamma
query  = grouped_norm(hidden, ple_norm_query)
s      = sum_rows(key * query) / sqrt(n_embd)     # per-stream dot product
gate   = sigmoid(sgn(s) * sqrt(clamp(|s|, 1e-6, inf)))     # signed square root
gated  = repeat(value, n_hc) * gate
conv   = silu(depthwise_causal_conv(grouped_norm(gated, ple_norm_conv)))
return hidden + gated + conv
```

The convolution is dilated by the n-gram size, kernel 4, and carries
`(kernel-1) * ngram_size * n_hc * n_embd` = 3*3*4*2560 = **92,160 floats
(360 KiB) of recurrent state per sequence**. The reference builds it as a sum
of shifted copies rather than `ggml_conv_1d_dw`, which it documents as
unreliable.

### One adaptation this hardware wants

Upstream keeps `per_layer_token_embd` resident and calls `ggml_get_rows` on it.
Here that is 35.8 GiB on top of the 62.3 GiB of weights, on a 124 GiB box.
But the gather reads 16 rows of 160 values per token -- about 10 KB -- so
doing it on the host and uploading the result as a graph input keeps 35.8 GiB
off the device and costs nothing in bandwidth. That is the same trade the
`CpuEmbedder` already makes for `token_embd`.
