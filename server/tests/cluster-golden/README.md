# Cluster golden outputs

One `<name>.sha256` file per qualification prompt. Each holds the sha256 of the
`choices[0].message.content` string a **single-node** `dflash_server` of the
same binary produced at `temperature 0` for that prompt and `max_tokens`. The
2-node run in `harness/qualification/deepseek4/qualify_ds4_cluster.sh` passes
only when every measured run reproduces it byte for byte (M1 exit gate).

Format: first whitespace-separated token is the hash; anything after it is a
comment (`sha256sum`-style lines work).

## Producing a golden

Run the single-node server with the *same* decode-relevant flags the cluster
script uses (`--ds4-expert-top-k`, `--ds4-prefill sparse`, `--chunk 2048`,
`--max-ctx`, `--prefix-cache-slots 0`, `--hard-limit-reply-budget 0`, no
`DFLASH_DS4_SPEC`), then:

```
python3 server/scripts/cluster/bench_ds4_cluster.py \
    --url http://127.0.0.1:8016 --max-tokens 512 --runs 3 --json-out /tmp/single.json
```

The script prints the sha256 of the output; if the three runs agree, store it:

```
echo "<sha256>  beta 512 tokens top-k 6 <binary git describe>" \
    > server/tests/cluster-golden/beta-512-k6.sha256
```

Names follow `<prompt>-<max_tokens>-k<top_k>`; the default the qualification
script looks for is `beta-512-k6.sha256`. Golden values are tied to the model
file (`DeepSeek-V4-Flash-0731-ROCMFPX-MIX-STRIX.gguf`, sha256
`7c0789d1…42b0a38`) and to kernels that change numerics; regenerate them when
either changes and note the binary in the comment.

No golden file is checked in until the first single-node run on the target
hardware has produced one; a fabricated hash would only make the gate lie.
