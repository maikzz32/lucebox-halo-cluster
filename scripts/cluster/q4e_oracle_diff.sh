#!/usr/bin/env bash
# Compare this fork's qwen4exp forward pass against upstream llama.cpp's.
#
# Bringing up qwen4exp eliminated every checkable component -- hyperparameters,
# tokenizer, embedding, head, MoE, the stream layout, the file itself -- and
# still produced nonsense. What was missing was a second implementation to
# disagree with. Upstream llama.cpp implements this architecture and can read
# the official conversion, which is in standard ggml types, so both can run the
# same file and their intermediates can be put side by side.
#
# llama-eval-callback prints every named tensor of a forward pass. The names
# come from the reference's cb() calls, so hc_norm, hc_gate, hc_mixed and
# hc_inject are all there -- the injection in particular, which this fork
# measures at about -19 per stream where the weights predict -0.5.
#
#   q4e_oracle_diff.sh <host> <model.gguf> [prompt]
#
# Prints the per-layer statistic for the tensors that matter, from upstream.
# Run the same prompt through the fork with DFLASH_QWEN4EXP_RMS=1 and compare.
set -euo pipefail

HOST=${1:?usage: q4e_oracle_diff.sh <host> <model.gguf> [prompt]}
MODEL=${2:?}
PROMPT=${3:-"The capital of France is"}
LLAMA=${LLAMA:-/home/maik/llama.cpp/build-cpu/bin}

ssh "$HOST" "cd /home/maik && '$LLAMA/llama-eval-callback' \
    -m '$MODEL' -p '$PROMPT' -n 1 -c 512 --no-warmup -t 28 2>&1 \
  | awk '
      /^ggml_debug:/ {
          name = \$2
          keep = (name ~ /hc_norm|hc_gate|hc_mixed|hc_inject|hc_combine|hc_init|ple_embd|ffn_out|attn_gated|result_norm/)
          if (keep) { print; getline; print; getline; print }
      }' " | head -200
