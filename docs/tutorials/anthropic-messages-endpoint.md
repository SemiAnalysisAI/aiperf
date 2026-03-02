<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Benchmarking the Anthropic Messages API

Profile Anthropic Messages API servers using AIPerf with the `anthropic_messages` endpoint type.

## Overview

The `anthropic_messages` endpoint type targets the Anthropic Messages API (`/v1/messages`). Use it when benchmarking:

- Anthropic API directly (api.anthropic.com)
- Self-hosted servers that implement the Anthropic Messages protocol
- Proxy servers that translate to the `/v1/messages` format

It supports streaming and non-streaming responses, text content, extended thinking, tool use, and `raw_messages` for verbatim trace replay.

### Key Differences from OpenAI Chat

| Feature | `chat` (OpenAI) | `anthropic_messages` |
|---|---|---|
| Endpoint path | `/v1/chat/completions` | `/v1/messages` |
| Auth header | `Authorization: Bearer <key>` | `x-api-key: <key>` |
| System message | In messages array as `{"role": "system"}` | Top-level `system` field |
| Streaming format | OpenAI SSE with `choices[].delta` | Anthropic SSE with typed events |
| Max tokens default | Uses `max_completion_tokens` | `max_tokens` (defaults to 1024) |
| Reasoning content | `reasoning_content` field | `thinking` content blocks |
| Version header | None | `anthropic-version: 2023-06-01` |

---

## Basic Usage

### Non-Streaming

```bash
aiperf profile \
    --model claude-sonnet-4-20250514 \
    --endpoint-type anthropic_messages \
    --url api.anthropic.com \
    --api-key $ANTHROPIC_API_KEY \
    --request-count 20 \
    --concurrency 4
```

### Streaming

Enable streaming to measure time-to-first-token (TTFT) and inter-token latency (ITL):

```bash
aiperf profile \
    --model claude-sonnet-4-20250514 \
    --endpoint-type anthropic_messages \
    --streaming \
    --url api.anthropic.com \
    --api-key $ANTHROPIC_API_KEY \
    --request-count 20 \
    --concurrency 4
```

### With Synthetic Input Tokens

Control input and output token distributions:

```bash
aiperf profile \
    --model claude-sonnet-4-20250514 \
    --endpoint-type anthropic_messages \
    --streaming \
    --synthetic-input-tokens-mean 500 \
    --synthetic-input-tokens-stddev 50 \
    --output-tokens-mean 200 \
    --output-tokens-stddev 20 \
    --url api.anthropic.com \
    --api-key $ANTHROPIC_API_KEY \
    --request-count 50 \
    --concurrency 8
```

### With a System Prompt

Add a shared system prompt across all requests using `--shared-system-prompt-length`. The system message is placed in the top-level `system` field (not in the messages array), matching the Anthropic API specification:

```bash
aiperf profile \
    --model claude-sonnet-4-20250514 \
    --endpoint-type anthropic_messages \
    --shared-system-prompt-length 100 \
    --url api.anthropic.com \
    --api-key $ANTHROPIC_API_KEY \
    --request-count 20
```

---

## Request Format

The endpoint produces payloads conforming to the Anthropic Messages API. A formatted request looks like:

```json
{
    "model": "claude-sonnet-4-20250514",
    "messages": [
        {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 1024,
    "stream": true
}
```

Key formatting rules:

- **Model**: Taken from the last turn's `model` field, falling back to `--model`
- **Max tokens**: Taken from the last turn's `max_tokens` field, defaulting to **1024** if not set
- **Stream**: Set from `--streaming` flag (default: `false`)
- **System**: Placed as a top-level string field when a system message is provided (never in the messages array)
- **Tools**: Included at the top level when tool definitions are present in the conversation
- **Extra inputs**: Merged into the payload via `--extra-inputs` (e.g., `--extra-inputs temperature:0.7`)

### Simple Text Content

When a turn has a single text with a single content string and no images, audios, or videos, the content is sent as a plain string:

```json
{"role": "user", "content": "Hello!"}
```

### Complex Content Blocks

When a turn has multiple text items or images, content is sent as a list of typed blocks:

```json
{"role": "user", "content": [
    {"type": "text", "text": "Describe this image"},
    {"type": "image", "source": {"type": "url", "url": "https://example.com/photo.jpg"}}
]}
```

### Raw Messages (Verbatim Replay)

When a turn has `raw_messages` set, those message dicts are expanded directly into the messages list. This is used by trace replay dataset types for faithful reproduction of tool use conversations:

```json
{
    "messages": [
        {"role": "user", "content": "Read the file a.py"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Let me check."},
            {"type": "tool_use", "id": "tu-1", "name": "read_file", "input": {"path": "a.py"}}
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu-1", "content": "file data"}
        ]}
    ]
}
```

---

## Response Parsing

The endpoint handles both streaming and non-streaming responses.

### Non-Streaming Responses

Non-streaming responses have `"type": "message"`. The parser extracts content from the `content` array and usage from the `usage` object:

```json
{
    "type": "message",
    "content": [
        {"type": "text", "text": "Hello, how can I help?"}
    ],
    "usage": {"input_tokens": 10, "output_tokens": 8}
}
```

When both `thinking` and `text` blocks are present, the response is parsed as a `ReasoningResponseData` with separate `content` and `reasoning` fields:

```json
{
    "type": "message",
    "content": [
        {"type": "thinking", "thinking": "Let me analyze this..."},
        {"type": "text", "text": "The answer is 42"}
    ]
}
```

### Streaming Responses

Streaming uses Anthropic's SSE event format. The endpoint handles the following event types:

| Event Type | Action |
|---|---|
| `message_start` | Extracts `usage` (input tokens) from `message.usage` |
| `content_block_delta` with `text_delta` | Extracts incremental text |
| `content_block_delta` with `thinking_delta` | Extracts incremental reasoning |
| `content_block_delta` with `input_json_delta` | Ignored (tool input) |
| `content_block_delta` with `signature_delta` | Ignored |
| `message_delta` | Extracts `usage` (output tokens) |
| `ping`, `content_block_start`, `content_block_stop`, `message_stop` | Ignored |
| `error` | Logged as warning |

A typical streaming sequence:

1. `message_start` -- carries input token usage
2. `ping`
3. `content_block_start` -- signals a new content block
4. `content_block_delta` (text_delta) -- incremental text tokens
5. `content_block_stop`
6. `message_delta` -- carries output token usage and stop reason
7. `message_stop`

---

## Authentication

The Anthropic Messages endpoint uses `x-api-key` header authentication (not `Authorization: Bearer`):

```bash
aiperf profile \
    --model claude-sonnet-4-20250514 \
    --endpoint-type anthropic_messages \
    --api-key sk-ant-api03-your-key-here \
    --url api.anthropic.com
```

The endpoint also sets the `anthropic-version: 2023-06-01` header by default. To override it or add beta headers (e.g., for extended thinking), use `--header`:

```bash
aiperf profile \
    --model claude-sonnet-4-20250514 \
    --endpoint-type anthropic_messages \
    --api-key $ANTHROPIC_API_KEY \
    --header anthropic-beta:extended-thinking-2025-04-11 \
    --url api.anthropic.com
```

Custom headers are merged with the defaults. If you provide `anthropic-version` via `--header`, it takes precedence.

---

## Extra Inputs

Pass additional API parameters with `--extra-inputs`:

```bash
aiperf profile \
    --model claude-sonnet-4-20250514 \
    --endpoint-type anthropic_messages \
    --extra-inputs temperature:0.7 \
    --extra-inputs top_p:0.9 \
    --url api.anthropic.com \
    --api-key $ANTHROPIC_API_KEY
```

These are merged directly into the request payload, producing:

```json
{
    "model": "claude-sonnet-4-20250514",
    "messages": [...],
    "max_tokens": 1024,
    "stream": false,
    "temperature": 0.7,
    "top_p": 0.9
}
```

---

## Custom Dataset Integration

### Single-Turn with Custom Prompts

```bash
cat > prompts.jsonl << 'EOF'
{"text": "Explain the theory of relativity."}
{"text": "Write a Python function to sort a list."}
{"text": "What causes tides?"}
EOF

aiperf profile \
    --model claude-sonnet-4-20250514 \
    --endpoint-type anthropic_messages \
    --input-file prompts.jsonl \
    --custom-dataset-type single_turn \
    --streaming \
    --url api.anthropic.com \
    --api-key $ANTHROPIC_API_KEY \
    --concurrency 4
```

### Multi-Turn Conversations

```bash
cat > conversations.jsonl << 'EOF'
{"session_id": "s1", "turns": [{"text": "What is Rust?"}, {"text": "Compare it to C++."}]}
{"session_id": "s2", "turns": [{"text": "Explain async/await."}, {"text": "Show a Python example."}]}
EOF

aiperf profile \
    --model claude-sonnet-4-20250514 \
    --endpoint-type anthropic_messages \
    --input-file conversations.jsonl \
    --custom-dataset-type multi_turn \
    --streaming \
    --url api.anthropic.com \
    --api-key $ANTHROPIC_API_KEY \
    --concurrency 2
```

---

## Server Token Counts

To use token counts reported by the server (from `usage` fields) instead of client-side tokenization:

```bash
aiperf profile \
    --model claude-sonnet-4-20250514 \
    --endpoint-type anthropic_messages \
    --use-server-token-count \
    --tokenizer gpt2 \
    --streaming \
    --url api.anthropic.com \
    --api-key $ANTHROPIC_API_KEY
```

The Anthropic Messages API returns usage data at two points during streaming:

- `message_start`: reports `input_tokens` (and cache-related fields)
- `message_delta`: reports `output_tokens`

The usage object may include `cache_creation_input_tokens` and `cache_read_input_tokens` when prompt caching is active.

---

## Tips

- **Tokenizer**: Anthropic models use a proprietary tokenizer. When using `--use-server-token-count`, you still need to specify a tokenizer for dataset generation (e.g., `--tokenizer gpt2`). For accurate client-side token counts, use a tokenizer that matches the model.
- **Max tokens default**: The endpoint defaults to `max_tokens: 1024` when no value is specified in the turn data. Use `--osl` to control output sequence length.
- **Audio and video**: The Anthropic Messages API does not support audio or video content blocks. If your dataset includes these, the endpoint logs a warning and drops them.
- **Custom endpoint path**: The default path is `/v1/messages`. Override with `--custom-endpoint /my/path` if your server uses a different path.
- **Connection strategy**: For multi-turn benchmarks where server-side session affinity matters, use `--connection-reuse-strategy sticky-user-sessions`.
