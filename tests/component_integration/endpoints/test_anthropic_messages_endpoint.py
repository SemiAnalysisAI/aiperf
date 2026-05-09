# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for /v1/messages endpoint."""

from __future__ import annotations

import time
from time import perf_counter

import orjson
import pytest
from aiperf_mock_server.app import (
    _build_anthropic_response_data,
    _should_emit_tool_use,
)
from aiperf_mock_server.config import MockServerConfig
from aiperf_mock_server.models import AnthropicMessagesRequest
from aiperf_mock_server.utils import make_ctx, stream_anthropic_messages

from aiperf.common.models import RequestRecord, SSEMessage, TextResponse
from aiperf.endpoints.anthropic_messages import AnthropicMessagesEndpoint
from aiperf.plugin.enums import EndpointType
from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
)


@pytest.mark.component_integration
class TestAnthropicMessagesEndpoint:
    """Tests for /v1/messages endpoint."""

    def test_basic_messages(self, cli: AIPerfCLI):
        """Basic non-streaming Anthropic Messages request."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model claude-sonnet-4-20250514 \
                --tokenizer gpt2 \
                --endpoint-type anthropic_messages \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count

    def test_streaming_messages(self, cli: AIPerfCLI):
        """Streaming Anthropic Messages with metrics validation."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model claude-sonnet-4-20250514 \
                --tokenizer gpt2 \
                --endpoint-type anthropic_messages \
                --streaming \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count
        assert result.has_streaming_metrics

    def test_messages_with_output_tokens(self, cli: AIPerfCLI):
        """Anthropic Messages with explicit output sequence length."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model claude-sonnet-4-20250514 \
                --tokenizer gpt2 \
                --endpoint-type anthropic_messages \
                --osl 10 \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count

    def test_messages_with_system_prompt_length(self, cli: AIPerfCLI):
        """Anthropic Messages with system prompt (via shared system prompt tokens)."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model claude-sonnet-4-20250514 \
                --tokenizer gpt2 \
                --endpoint-type anthropic_messages \
                --shared-system-prompt-length 20 \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count


# ============================================================================
# Tool-use round-trip tests
#
# These exercise the integration between the AIPerf mock server's Anthropic
# response generator and ``AnthropicMessagesEndpoint.build_assistant_turn``:
# the endpoint's FORK-mode replay only works if the mock's wire shape (both
# non-streaming JSON and streaming SSE) feeds back through the endpoint's
# accumulator producing a Turn whose ``raw_messages`` round-trips the
# tool_use block (id, name, parsed input dict).
#
# The CLI-driven tests above use FakeTransport, which is outside this scope
# to repair; these tests skip the CLI and call the mock server's response
# helpers directly, which is enough to cover the request -> mock-response
# -> ``build_assistant_turn`` path the production code relies on.
# ============================================================================


_TOOL_DEF = {
    "name": "calculator",
    "description": "Adds two numbers.",
    "input_schema": {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
        },
        "required": ["a", "b"],
    },
}


def _make_anthropic_request(
    *, stream: bool, with_tools: bool, with_thinking: bool = False
) -> AnthropicMessagesRequest:
    """Build an AnthropicMessagesRequest exercising a chosen feature combo."""
    payload: dict = {
        "model": "claude-sonnet-4-20250514",
        "messages": [{"role": "user", "content": "Add 2 and 3, please."}],
        "max_tokens": 64,
        "stream": stream,
    }
    if with_tools:
        payload["tools"] = [_TOOL_DEF]
        payload["tool_choice"] = {"type": "any"}
    if with_thinking:
        # Trigger the mock's reasoning-content branch via ignore_eos so the
        # tokenizer emits reasoning tokens (mock-server convention).
        payload["thinking"] = {"type": "enabled", "budget_tokens": 16}
    return AnthropicMessagesRequest.model_validate(payload)


def _make_ctx(req: AnthropicMessagesRequest):
    """Construct a RequestCtx with zero-latency config (fast tests)."""
    return make_ctx(
        req,
        endpoint="/v1/messages",
        start_time=perf_counter(),
        config=MockServerConfig(fast=True),
    )


def _record_from_json(json_data: dict) -> RequestRecord:
    """Build a RequestRecord from a single non-streaming JSON response."""
    perf_ns = time.perf_counter_ns()
    return RequestRecord(
        start_perf_ns=perf_ns,
        end_perf_ns=perf_ns + 1,
        timestamp_ns=perf_ns,
        status=200,
        responses=[
            TextResponse(
                perf_ns=perf_ns,
                content_type="application/json",
                text=orjson.dumps(json_data).decode("utf-8"),
            )
        ],
    )


async def _record_from_stream(req: AnthropicMessagesRequest) -> RequestRecord:
    """Drive ``stream_anthropic_messages`` to completion, build a RequestRecord."""
    ctx = _make_ctx(req)
    tool_use_block = _should_emit_tool_use(req)
    responses: list[SSEMessage] = []
    async for chunk in stream_anthropic_messages(
        ctx, "/v1/messages", tool_use_block=tool_use_block
    ):
        # Each yielded chunk is a single SSE event; trim the trailing
        # delimiter so SSEMessage.parse sees one event per call.
        perf_ns = time.perf_counter_ns()
        responses.append(SSEMessage.parse(chunk.rstrip(b"\n"), perf_ns))
    return RequestRecord(
        start_perf_ns=responses[0].perf_ns,
        end_perf_ns=responses[-1].perf_ns,
        timestamp_ns=responses[0].perf_ns,
        status=200,
        responses=responses,
    )


@pytest.fixture
def anthropic_endpoint() -> AnthropicMessagesEndpoint:
    """Real AnthropicMessagesEndpoint instance for replay-side assertions."""
    return create_endpoint_with_mock_transport(
        AnthropicMessagesEndpoint,
        create_model_endpoint(EndpointType.ANTHROPIC_MESSAGES),
    )


@pytest.mark.component_integration
class TestAnthropicMessagesToolUseRoundTrip:
    """Mock-server -> AnthropicMessagesEndpoint round-trip for tool_use."""

    def test_should_emit_tool_use_only_when_tools_present(self):
        """Sanity-check the trigger predicate: tools required, ``none`` opts out."""
        no_tools = AnthropicMessagesRequest.model_validate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
            }
        )
        assert _should_emit_tool_use(no_tools) is None

        with_tools = _make_anthropic_request(stream=False, with_tools=True)
        block = _should_emit_tool_use(with_tools)
        assert block is not None
        assert block["type"] == "tool_use"
        assert block["name"] == _TOOL_DEF["name"]

        opt_out = AnthropicMessagesRequest.model_validate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
                "tools": [_TOOL_DEF],
                "tool_choice": {"type": "none"},
            }
        )
        assert _should_emit_tool_use(opt_out) is None

    def test_non_streaming_tool_use_round_trip(
        self,
        anthropic_endpoint: AnthropicMessagesEndpoint,
    ):
        """Non-streaming: mock returns tool_use -> endpoint round-trips it."""
        req = _make_anthropic_request(stream=False, with_tools=True)
        ctx = _make_ctx(req)

        response_data = _build_anthropic_response_data(ctx, req)

        # Mock-side invariants we depend on for replay correctness.
        assert response_data["stop_reason"] == "tool_use"
        tool_blocks = [
            b for b in response_data["content"] if b.get("type") == "tool_use"
        ]
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["name"] == _TOOL_DEF["name"]
        assert tool_blocks[0]["id"] == "toolu_mock_1"
        assert isinstance(tool_blocks[0]["input"], dict)

        record = _record_from_json(response_data)
        turn = anthropic_endpoint.build_assistant_turn(record)

        assert turn is not None
        assert turn.raw_messages is not None, (
            "Expected raw_messages on FORK-mode replay turn"
        )
        msg = turn.raw_messages[0]
        assert msg["role"] == "assistant"
        round_tripped = [b for b in msg["content"] if b.get("type") == "tool_use"]
        assert len(round_tripped) == 1
        rt = round_tripped[0]
        assert rt["id"] == "toolu_mock_1"
        assert rt["name"] == _TOOL_DEF["name"]
        assert rt["input"] == {"arg": "value"}

    @pytest.mark.asyncio
    async def test_streaming_tool_use_round_trip(
        self,
        anthropic_endpoint: AnthropicMessagesEndpoint,
    ):
        """Streaming: SSE input_json_delta fragments reassemble into a tool_use."""
        req = _make_anthropic_request(stream=True, with_tools=True)
        record = await _record_from_stream(req)

        # Mock-side invariant: at least one input_json_delta event exists,
        # otherwise we'd only be testing the non-streaming code path.
        deltas = [
            r.get_json()
            for r in record.responses
            if r.get_json() and r.get_json().get("type") == "content_block_delta"
        ]
        assert any(
            (d.get("delta") or {}).get("type") == "input_json_delta" for d in deltas
        ), "mock streaming did not emit input_json_delta"

        turn = anthropic_endpoint.build_assistant_turn(record)
        assert turn is not None
        assert turn.raw_messages is not None
        msg = turn.raw_messages[0]
        round_tripped = [b for b in msg["content"] if b.get("type") == "tool_use"]
        assert len(round_tripped) == 1
        rt = round_tripped[0]
        assert rt["name"] == _TOOL_DEF["name"]
        assert rt["id"] == "toolu_mock_1"
        # Streaming reassembly must parse the input_json_delta fragments
        # back into the original input dict, not leave them as a raw string.
        assert isinstance(rt["input"], dict)
        assert rt["input"] == {"arg": "value"}

    @pytest.mark.asyncio
    async def test_streaming_thinking_then_text_then_tool_use(
        self,
        anthropic_endpoint: AnthropicMessagesEndpoint,
    ):
        """Streaming thinking + tool_use: ordering is [thinking, text?, tool_use].

        The mock server's Anthropic streamer only emits ``thinking`` blocks when
        the underlying tokenizer produces ``reasoning_content_tokens`` (which
        currently happens only for ``ChatCompletionRequest`` against
        gpt-oss/qwen models). To still exercise the
        thinking + text + tool_use ordering invariant against the mock's
        actual SSE formatter, we run the mock streamer with a tool_use_block
        and prepend a thinking sequence built with the mock's own
        ``_anthropic_sse`` byte formatter, then feed the combined event
        stream back through the real endpoint.
        """
        from aiperf_mock_server.utils import _anthropic_sse

        req = _make_anthropic_request(stream=True, with_tools=True)

        # Build the thinking prefix using the mock's SSE formatter so the
        # bytes match what the mock server emits on a real reasoning request.
        thinking_chunks: list[bytes] = [
            _anthropic_sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
            ),
            _anthropic_sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "weighing..."},
                },
            ),
            _anthropic_sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "signature_delta",
                        "signature": "mock-signature",
                    },
                },
            ),
            _anthropic_sse(
                "content_block_stop",
                {"type": "content_block_stop", "index": 0},
            ),
        ]

        ctx = _make_ctx(req)
        tool_use_block = _should_emit_tool_use(req)
        responses: list[SSEMessage] = []
        for chunk in thinking_chunks:
            perf_ns = time.perf_counter_ns()
            responses.append(SSEMessage.parse(chunk.rstrip(b"\n"), perf_ns))
        async for chunk in stream_anthropic_messages(
            ctx, "/v1/messages", tool_use_block=tool_use_block
        ):
            perf_ns = time.perf_counter_ns()
            responses.append(SSEMessage.parse(chunk.rstrip(b"\n"), perf_ns))

        record = RequestRecord(
            start_perf_ns=responses[0].perf_ns,
            end_perf_ns=responses[-1].perf_ns,
            timestamp_ns=responses[0].perf_ns,
            status=200,
            responses=responses,
        )

        turn = anthropic_endpoint.build_assistant_turn(record)
        assert turn is not None
        assert turn.raw_messages is not None
        content = turn.raw_messages[0]["content"]
        types = [b.get("type") for b in content]

        # Endpoint orders blocks: thinking first, then text (if any),
        # then tool_use last. Verify this exact relative ordering.
        assert "tool_use" in types, f"missing tool_use in {types}"
        assert "thinking" in types, f"missing thinking in {types}"
        assert types.index("thinking") < types.index("tool_use")
        if "text" in types:
            assert types.index("text") < types.index("tool_use")
            assert types.index("thinking") < types.index("text")

        thinking_block = next(b for b in content if b.get("type") == "thinking")
        assert thinking_block["thinking"] == "weighing..."
        assert thinking_block["signature"] == "mock-signature"

        tool_use = next(b for b in content if b.get("type") == "tool_use")
        assert tool_use["name"] == _TOOL_DEF["name"]
        assert tool_use["input"] == {"arg": "value"}
