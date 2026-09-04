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

## Status

Stage 0 complete. Next: the multi-shard reader (stage 1), which is a
prerequisite for loading anything at all.
