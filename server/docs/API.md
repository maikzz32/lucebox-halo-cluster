# DFlash Server API Reference

HTTP server exposing OpenAI-compatible, Anthropic, and Responses API endpoints
for inference with speculative decoding. Model-independent — works with any
backend (qwen35, qwen3, gemma4, laguna).

---

## Endpoints

| Method | Path | Description | Status |
|--------|------|-------------|--------|
| GET | `/health`, `/` | Health check | ✅ |
| GET | `/v1/models` | List available models | ✅ |
| POST | `/v1/chat/completions` | OpenAI Chat Completions | ✅ |
| POST | `/v1/messages` | Anthropic Messages | ✅ |
| POST | `/v1/responses` | OpenAI Responses API | ✅ |

---

## Agent Turn Cache

Start the server with `--agent-turn-cache` to extend the existing in-memory
prefix cache through generated tool calls. After the model emits a valid tool
call, the server reuses its deepest compatible prefix checkpoint, replays the
uncached tail once, and saves the canonical completed turn. On the next OpenAI
Chat Completions or Responses request, only the appended tool result and new
suffix need prefill.

This is a server-wide optimization; request bodies do not change. It requires
`--prefix-cache-slots` to be nonzero. Compressed or token-rewritten prompts and
requests without a compatible checkpoint safely fall back to ordinary prefix
caching.

The replay moves prefill work out of the follow-up request when tool execution
is long enough to overlap it; it does not eliminate that work. Paged attention
and `--max-concurrency` do not yet support shared prefix blocks, so they cannot
be combined with Agent Turn Cache.

The exact full-prompt cache may remain enabled for identical-request hits.
Disable it only when benchmarking the incremental Agent Turn Cache benefit.

Successful Chat Completions and Responses requests expose the measured result
under `usage.timings`:

- `agent_turn_cache_hit`: the restored prefix includes a generated agent turn.
- `cached_prefix_tokens`: backend-confirmed tokens restored from KV state.
- `prefilled_tokens`: prompt tokens computed for this request.

### `usage.timings.cluster` (multi-node runs only)

Present only when the server is rank 0 of a `--cluster-*` run (see
`server/docs/CLUSTER.md`). It reports what each rank actually did for **this**
request, which is how an imbalanced expert placement or a slow node is found.
The section appears in every response shape, streaming included.

```json
"cluster": {
  "size": 2, "request_id": 1, "complete": true, "ctrl_wait_ms": 1.1,
  "per_rank": [
    {"rank": 0, "steps": 40, "compute_ms": 3607.0,
     "allreduce_calls": 1806, "allreduce_bytes": 51429376,
     "allreduce_wait_ms": 11.1, "ctrl_wait_ms": 1.1,
     "device_bytes": 67561107456},
    {"rank": 1, "...": "..."}
  ]
}
```

| Field | Meaning |
|---|---|
| `size` | ranks that served the request; `per_rank[0]` is the head |
| `complete` | every worker's report arrived; `false` means the numbers are partial |
| `compute_ms` | that rank's own prefill+decode wall time. Ranks should be within a few percent of each other; a gap is placement imbalance or a slow node |
| `allreduce_calls` / `allreduce_bytes` | collectives issued and payload summed across the ranks |
| `allreduce_wait_ms` | host time blocked on collectives. **0 for decode on path 3b**, where the collective runs inside the graph and its cost is part of `compute_ms`; non-zero for prefill, which still takes path 3a |
| `ctrl_wait_ms` | host time blocked on the control channel. A worker's value is how long it waited for the head's decisions |
| `device_bytes` | device memory in use on that rank, high-water mark sampled at request ends |
| `verify_hash` | only with `--cluster-verify-hash n`: probe and mismatch counts, plus the first divergent rank and step |
| `error` | only when the report gather failed |

---

## POST `/v1/chat/completions` (OpenAI-compatible)

### Supported Request Parameters

| Parameter | Type | Default | Description | Status |
|-----------|------|---------|-------------|--------|
| `model` | string | server default | Model identifier | ✅ |
| `messages` | array | required | Conversation messages | ✅ |
| `stream` | bool | `false` | SSE streaming | ✅ |
| `max_tokens` | int | 4096 | Max output tokens | ✅ |
| `max_output_tokens` | int | 4096 | Alias for max_tokens | ✅ |
| `max_completion_tokens` | int | 4096 | Alias for max_tokens | ✅ |
| `temperature` | float | 0.0 | Sampling temperature (0 = greedy) | ✅ |
| `top_p` | float | 1.0 | Nucleus sampling threshold | ✅ |
| `top_k` | int | 0 | Top-k filtering (0 = disabled) | ✅ |
| `seed` | int | 0 | Random seed for reproducibility | ✅ |
| `frequency_penalty` | float | 0.0 | Penalize frequent tokens (range: -2.0 to 2.0) | ✅ |
| `presence_penalty` | float | 0.0 | Penalize present tokens (range: -2.0 to 2.0) | ✅ |
| `repetition_penalty` | float | 1.0 | HF-style multiplicative penalty (>1 penalizes) | ✅ |
| `rep_pen` | float | 1.0 | Alias for repetition_penalty | ✅ |
| `rep_window` | int | 256 | Token lookback window for penalties | ✅ |
| `tools` | array | none | Tool/function definitions | ✅ |
| `reasoning` | object | — | Reasoning effort control (`{"effort":"medium"}`) | ✅ |
| `chat_template_kwargs` | object | — | Direct template control (`{"enable_thinking":true}`) | ✅ |
| `stop` | string/array | — | Stop sequences | ✅ |
| `n` | int | — | Number of completions | ❌ TODO |
| `logprobs` | bool | — | Return log probabilities | ❌ TODO |
| `top_logprobs` | int | — | Number of top logprobs per token | ❌ TODO |
| `response_format` | object | — | JSON mode / structured output | ❌ TODO |
| `tool_choice` | string/object | — | Tool choice / force tool usage | ✅ |
| `logit_bias` | object | — | Per-token logit adjustments | ❌ TODO |
| `user` | string | — | End-user identifier (tracking) | ❌ TODO |
| `stream_options` | object | — | Streaming options (e.g. include_usage) | ❌ TODO 🔴 |

### Response Fields

| Field | Status |
|-------|--------|
| `id` | ✅ `chatcmpl_<hex>` |
| `object` | ✅ `"chat.completion"` / `"chat.completion.chunk"` |
| `model` | ✅ |
| `choices[].message.role` | ✅ |
| `choices[].message.content` | ✅ |
| `choices[].message.tool_calls` | ✅ |
| `choices[].finish_reason` | ✅ (`stop`, `length`, `tool_calls`) |
| `choices[].delta` (streaming) | ✅ |
| `usage.prompt_tokens` | ✅ |
| `usage.completion_tokens` | ✅ |
| `usage.total_tokens` | ✅ |
| `choices[].logprobs` | ❌ TODO |

---

## POST `/v1/messages` (Anthropic-compatible)

### Supported Request Parameters

| Parameter | Type | Default | Description | Status |
|-----------|------|---------|-------------|--------|
| `model` | string | server default | Model identifier | ✅ |
| `messages` | array | required | Conversation messages | ✅ |
| `system` | string/array | — | System prompt (top-level) | ✅ |
| `stream` | bool | `false` | SSE streaming | ✅ |
| `max_tokens` | int | 4096 | Max output tokens | ✅ |
| `temperature` | float | 0.0 | Sampling temperature | ✅ |
| `top_p` | float | 1.0 | Nucleus sampling | ✅ |
| `top_k` | int | 0 | Top-k filtering | ✅ |
| `seed` | int | — | Random seed | ✅ |
| `frequency_penalty` | float | 0.0 | Penalize frequent tokens | ✅ |
| `presence_penalty` | float | 0.0 | Penalize present tokens | ✅ |
| `thinking` | object | — | Thinking mode (`{"type":"enabled"}`) | ✅ |
| `tools` | array | — | Tool definitions | ✅ |
| `tool_choice` | string/object | — | Tool choice / force tool usage | ✅ |
| `stop_sequences` | array | — | Stop sequences | ✅ |
| `metadata` | object | — | Request metadata (tracing) | ❌ TODO |

### Response Structure

Follows Anthropic Messages API structure with `content` blocks:
- `type: "text"` — text content
- `type: "thinking"` — reasoning/thinking content
- `type: "tool_use"` — tool call

---

## POST `/v1/responses` (OpenAI Responses API)

### Supported Request Parameters

| Parameter | Type | Default | Description | Status |
|-----------|------|---------|-------------|--------|
| `model` | string | server default | Model identifier | ✅ |
| `input` | string/array | required | Input messages or text | ✅ |
| `instructions` | string | — | System instructions | ✅ |
| `stream` | bool | `false` | SSE streaming | ✅ |
| `max_output_tokens` | int | 4096 | Max output tokens | ✅ |
| `temperature` | float | 0.0 | Sampling temperature | ✅ |
| `top_p` | float | 1.0 | Nucleus sampling | ✅ |
| `seed` | int | — | Random seed | ✅ |
| `frequency_penalty` | float | 0.0 | Penalize frequent tokens | ✅ |
| `presence_penalty` | float | 0.0 | Penalize present tokens | ✅ |
| `reasoning` | object | — | Reasoning effort | ✅ |
| `tools` | array | — | Tool definitions | ✅ |
| `tool_choice` | string/object | — | Tool choice / force tool usage | ✅ |
| `parallel_tool_calls` | bool | — | Allow parallel tool calls | ❌ TODO 🔴 |
| `store` | bool | — | Persist response | ❌ TODO 🔴 |
| `include` | array | — | Include extra response fields | ❌ TODO 🔴 |
| `text` | object | — | Structured output / JSON schema | ❌ TODO 🔴 |
| `service_tier` | string | — | Routing hint | ❌ TODO 🔴 |
| `previous_response_id` | string | — | Multi-turn chaining | ❌ TODO |

---

## Sampling Chain

The sampler applies penalties and sampling in this order:

```
logits (from GPU)
  → repetition_penalty (multiplicative, HF-style)
  → frequency_penalty + presence_penalty (additive, OpenAI-style)
  → top_k filtering
  → temperature softmax
  → top_p (nucleus) filtering
  → random draw (or argmax if temp=0)
```

All penalties respect `rep_window` (default 256 tokens lookback).

When `temperature = 0`:
- Penalties are still applied to logits
- Argmax is used (deterministic, no random sampling)
- Speculative decode is enabled only when no logit processing is needed

---

## Streaming

- **OpenAI format**: `text/event-stream` with `data: {...}\n\n` chunks, terminated by `data: [DONE]\n\n`
- **Anthropic format**: SSE with `event: message_start`, `content_block_start`, `content_block_delta`, `message_stop`
- **Responses format**: SSE with `response.created`, `response.output_item.*`, `response.completed`
- During long prefill or cleanup phases with no token data, the server emits a
  valid SSE comment (`: keep-alive`) every 15 seconds. Clients ignore the
  comment as payload while HTTP body-idle timers remain active. If the peer
  closes, the server propagates cancellation into the backend at its next safe
  prefill or decode boundary.

---

## Features

| Feature | Status | Notes |
|---------|--------|-------|
| Multi-turn conversation | ✅ | Full message history |
| Tool/function calling | ✅ | XML and JSON tool parsing |
| Stop sequences | ✅ | OpenAI `stop` and Anthropic `stop_sequences` |
| Thinking/reasoning | ✅ | OpenAI `reasoning.effort`, Anthropic `thinking.type` |
| Prefix cache (memory) | ✅ | Automatic KV cache reuse |
| Prefix cache (disk) | ✅ | Persistent across restarts |
| PFlash (speculative prefill) | ✅ | Compresses long prompts |
| Client disconnect detection | ✅ | Aborts generation on disconnect |
| CORS | ✅ | Enabled by default |
| Tool memory | ✅ | Caches tool call results |

---

## TODO

### 🔴 High Priority (used by Codex and/or Claude Code)

These parameters are sent by real Codex CLI or Claude Code clients. Missing
support causes errors or silent feature degradation.

| Feature | Used By | Notes |
|---------|---------|-------|
| **`parallel_tool_calls`** | Codex (Responses API) | Codex always sends `true`. Can accept and ignore (we serialize calls). |
| **`store`** | Codex (Responses API) | Controls response persistence. Accept field; can be no-op locally. |
| **`include`** | Codex (Responses API) | Controls what's included in response events. Accept field. |
| **`text` (structured output)** | Codex (Responses API) | JSON schema output formatting (`{"format":{"type":"json_schema","schema":{...}}}`). Needed for structured tool outputs. |
| **`service_tier`** | Codex (Responses API) | Routing hint (e.g., `"default"`). Accept and ignore. |

### 🟡 Medium Priority

| Feature | Used By | Notes |
|---------|---------|-------|
| **`response_format`** | Chat Completions API | JSON mode / structured output (OpenAI Chat format). |
| **`metadata`** | Claude Code (Anthropic) | Request metadata for tracing. Accept and ignore. |
| **`stream_options`** | Some OpenAI clients | `{"include_usage": true}` — usage in final streaming chunk. |
| **Input validation** | — | Clamp penalty ranges [-2,2], reject invalid params with 400. |
| **`previous_response_id`** | Codex (Responses API) | Multi-turn response chaining (we handle via `input` already). |

### 🟢 Low Priority

| Feature | Notes |
|---------|-------|
| **`logprobs` / `top_logprobs`** | Token probabilities in response. Debugging/analysis only. |
| **`n` (multiple completions)** | Generate N choices per request. No known agent uses this. |
| **`logit_bias`** | Per-token logit adjustments. |
| **`user`** | End-user identifier (tracking only). |
| **`prompt_cache_key`** | Codex sends this for server-side caching hints. We have our own prefix cache. |
