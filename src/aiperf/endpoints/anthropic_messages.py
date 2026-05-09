# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, ClassVar

from aiperf.common.enums import CaseInsensitiveStrEnum, MediaType
from aiperf.common.models import (
    BaseResponseData,
    ExtractedPayload,
    InferenceServerResponse,
    ParsedResponse,
    ReasoningResponseData,
    RequestInfo,
    RequestRecord,
    TextResponseData,
    Turn,
)
from aiperf.common.types import JsonObject
from aiperf.endpoints import _anthropic_internals as _internals
from aiperf.endpoints.base_endpoint import BaseEndpoint

_ANTHROPIC_VERSION: str = "2023-06-01"


class ContentBlockType(CaseInsensitiveStrEnum):
    """Content block types in Anthropic Messages API responses."""

    TEXT = "text"
    THINKING = "thinking"
    TOOL_USE = "tool_use"


class DeltaType(CaseInsensitiveStrEnum):
    """Delta types within content_block_delta SSE events."""

    TEXT_DELTA = "text_delta"
    THINKING_DELTA = "thinking_delta"
    INPUT_JSON_DELTA = "input_json_delta"
    SIGNATURE_DELTA = "signature_delta"


class EventType(CaseInsensitiveStrEnum):
    """Payload type values in Anthropic Messages API responses."""

    MESSAGE = "message"
    MESSAGE_START = "message_start"
    CONTENT_BLOCK_START = "content_block_start"
    CONTENT_BLOCK_DELTA = "content_block_delta"
    CONTENT_BLOCK_STOP = "content_block_stop"
    MESSAGE_DELTA = "message_delta"
    MESSAGE_STOP = "message_stop"
    PING = "ping"
    ERROR = "error"


class AnthropicMessagesEndpoint(BaseEndpoint):
    """Anthropic Messages endpoint.

    Supports text content, tool use, extended thinking, and both
    streaming and non-streaming responses via /v1/messages.

    Message-array construction reuses the generic
    ``BaseEndpoint.build_messages`` flow. Anthropic's wire shape differs
    from OpenAI chat in two places:

    - the system prompt lives at the top level of the payload (not in
      ``messages`` as a ``role: system`` entry);
    - the image content part is ``{"type": "image", "source": {...}}``
      rather than ``{"type": "image_url", ...}``.

    Audio and video content blocks are not part of the Anthropic Messages
    API; ``_render_audio_part`` / ``_render_video_part`` raise immediately
    so misuse fails at format-time rather than producing an opaque server
    4xx.
    """

    # Anthropic content-part type names: ``text`` (same as default) and
    # bare ``image`` (Anthropic uses ``{"type": "image", "source": {...}}``,
    # not OpenAI's ``image_url``). Audio/video are unsupported - empty
    # sets prevent ``extract_payload_inputs`` from miscounting parts that
    # happen to share OpenAI's type names.
    PART_TYPES: ClassVar[dict[MediaType, set[str]]] = {
        MediaType.TEXT: {"text"},
        MediaType.IMAGE: {"image"},
        MediaType.AUDIO: set(),
        MediaType.VIDEO: set(),
    }

    def get_endpoint_headers(self, request_info: RequestInfo) -> dict[str, str]:
        """Get Anthropic-specific headers using x-api-key auth."""
        cfg = self.model_endpoint.endpoint
        headers: dict[str, str] = {"content-type": "application/json"}
        if cfg.headers:
            headers.update(cfg.headers)
        if cfg.api_key:
            headers["x-api-key"] = cfg.api_key
        headers.setdefault("anthropic-version", _ANTHROPIC_VERSION)
        return headers

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format Anthropic Messages API request payload.

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            Anthropic Messages API payload
        """
        if not request_info.turns:
            raise ValueError("Anthropic Messages endpoint requires at least one turn.")

        turns = request_info.turns
        model_endpoint = request_info.model_endpoint

        messages: list[dict[str, Any]] = []
        if request_info.user_context_message:
            messages.append(
                {"role": "user", "content": request_info.user_context_message}
            )
        messages.extend(self.build_messages(turns))

        raw_tools = self._latest_turn_attr(turns, "raw_tools")
        raw_system = self._latest_turn_attr(turns, "raw_system")
        max_tokens = self._latest_turn_attr(turns, "max_tokens")
        extra_body = self._latest_turn_attr(turns, "extra_body")
        model_name = self._latest_turn_attr(turns, "model")

        payload: dict[str, Any] = {
            "model": model_name or model_endpoint.primary_model_name,
            "messages": messages,
            # Anthropic requires max_tokens; default mirrors the API's
            # historical minimum-friendly value when no per-turn cap is set.
            "max_tokens": max_tokens if max_tokens is not None else 1024,
            "stream": model_endpoint.endpoint.streaming,
        }

        # raw_system (Turn-level list-of-blocks) wins over the
        # conversation-level system_message string. Lets callers attach
        # cache_control / Anthropic-specific extensions per-block.
        if raw_system is not None:
            payload["system"] = raw_system
        elif request_info.system_message:
            payload["system"] = request_info.system_message

        if raw_tools is not None:
            payload["tools"] = raw_tools

        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)

        if extra_body:
            payload.update(extra_body)

        self.trace(lambda: f"Formatted payload: {payload}")
        return payload

    # --- Content-part hooks (override only the Anthropic-specific shapes) ----

    def _render_image_part(self, url_or_data_uri: str) -> dict[str, Any]:
        """Anthropic image part: ``{"type": "image", "source": {"type": "url", ...}}``."""
        return {
            "type": "image",
            "source": {"type": "url", "url": url_or_data_uri},
        }

    def _render_audio_part(self, format_and_b64: str) -> dict[str, Any]:
        """Anthropic Messages API does not accept audio content blocks.

        Raise immediately so misuse fails at ``format_payload`` rather than
        producing an opaque server 4xx after the request is dispatched.
        """
        raise NotImplementedError(
            "Anthropic Messages API does not support audio input. "
            "Use a different endpoint, or remove audio content from the turn."
        )

    def _render_video_part(self, url_or_data_uri: str) -> dict[str, Any]:
        """Anthropic Messages API does not accept video content blocks.

        Raise immediately so misuse fails at ``format_payload`` rather than
        producing an opaque server 4xx after the request is dispatched.
        """
        raise NotImplementedError(
            "Anthropic Messages API does not support video input. "
            "Use a different endpoint, or remove video content from the turn."
        )

    # --- Payload -> inputs extraction ----------------------------------------

    def extract_payload_inputs(self, payload: dict[str, Any]) -> ExtractedPayload:
        """Anthropic single-pass extraction.

        Inherits the base-class walk for ``messages`` (which dispatches
        content parts via ``PART_TYPES``) and additionally:

        - prepends top-level ``system`` (string OR list of
          ``{"type":"text","text":...}`` blocks);
        - collects ``input_schema`` for top-level ``tools``
          (Anthropic's equivalent of OpenAI's ``parameters``); ``name``
          and ``description`` are already harvested by the base walk;
        - collects ``name``/``input`` from ``tool_use`` content blocks
          and the ``content`` text of ``tool_result`` blocks - parts the
          server tokenises on agentic-history replay that the base walk
          would otherwise drop because they are not in ``PART_TYPES``.
        """
        result = super().extract_payload_inputs(payload)
        _internals.walk_system(payload, result)
        _internals.walk_tool_schemas(payload, result)
        _internals.walk_tool_blocks(payload, result)
        return result

    def build_assistant_turn(self, record: RequestRecord) -> Turn | None:
        """Capture text + thinking + ``tool_use`` blocks from an Anthropic
        response for replay.

        Walks the raw responses on ``record``, accumulating text deltas,
        reassembling streaming ``thinking`` blocks (``thinking_delta`` +
        ``signature_delta`` per index) and ``tool_use`` blocks
        (``input_json_delta`` fragments per index), then returns a Turn
        whose ``raw_messages`` re-renders as an assistant message carrying
        the full content array - thinking, then text, then tool_use - so a
        FORK-mode DAG child inheriting the parent's history sees the
        parent's complete reply, not just the text.

        Falls back to the base text-only behaviour when neither thinking
        nor tool_use blocks are present, so callers that don't use either
        feature see no behavioural change.
        """
        text_parts: list[str] = []
        thinking_blocks_by_index: dict[int, dict[str, Any]] = {}
        tool_uses_by_index: dict[int, dict[str, Any]] = {}

        for response in record.responses:
            json_obj = response.get_json()
            if not json_obj:
                continue
            _internals.absorb_event(
                json_obj,
                text_parts,
                thinking_blocks_by_index,
                tool_uses_by_index,
            )

        if not thinking_blocks_by_index and not tool_uses_by_index:
            return super().build_assistant_turn(record)

        content_blocks: list[dict[str, Any]] = []
        # Anthropic emits thinking before text/tool_use; preserve that order
        # so the wire shape matches a fresh assistant reply.
        for idx in sorted(thinking_blocks_by_index):
            content_blocks.append(
                _internals.finalise_thinking(thinking_blocks_by_index[idx])
            )
        text = "".join(text_parts)
        if text:
            content_blocks.append({"type": "text", "text": text})
        for idx in sorted(tool_uses_by_index):
            content_blocks.append(_internals.finalise_tool_use(tool_uses_by_index[idx]))

        assistant_msg = {"role": "assistant", "content": content_blocks}
        return Turn(role="assistant", raw_messages=[assistant_msg])

    def _render_text_part(self, text: str) -> dict[str, Any]:
        """Anthropic text part shape: ``{"type": "text", "text": ...}``.

        Identical to the chat default; named explicitly here so the file
        documents the full Anthropic content-part shape contract in one place.
        """
        return {"type": "text", "text": text}

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse Anthropic Messages response.

        Handles both streaming SSE events and non-streaming JSON responses.
        Uses the ``type`` field present in all Anthropic payloads to dispatch:
        ``"message"`` for non-streaming, streaming event types otherwise.

        Args:
            response: Raw response from inference server

        Returns:
            Parsed response with extracted text/reasoning content and usage data
        """
        json_obj = response.get_json()
        if not json_obj:
            return None

        event_type = json_obj.get("type")
        if event_type == EventType.MESSAGE:
            return self._parse_non_streaming(response, json_obj)
        if event_type is not None:
            return self._parse_streaming_event(response, json_obj, event_type)
        return None

    def _parse_non_streaming(
        self, response: InferenceServerResponse, json_obj: JsonObject
    ) -> ParsedResponse | None:
        """Parse non-streaming Anthropic Messages response."""
        data = self._extract_content_data(json_obj)
        usage = json_obj.get("usage")

        if data or usage:
            return ParsedResponse(perf_ns=response.perf_ns, data=data, usage=usage)
        return None

    def _extract_content_data(self, json_obj: JsonObject) -> BaseResponseData | None:
        """Extract content from Anthropic non-streaming response content array."""
        content_blocks = json_obj.get("content")
        if not content_blocks or not isinstance(content_blocks, list):
            return None

        text_parts: list[str] = []
        thinking_parts: list[str] = []

        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == ContentBlockType.TEXT:
                text_val = block.get("text")
                if text_val:
                    text_parts.append(text_val)
            elif block_type == ContentBlockType.THINKING:
                thinking_val = block.get("thinking")
                if thinking_val:
                    thinking_parts.append(thinking_val)

        text = "".join(text_parts) or None
        thinking = "".join(thinking_parts) or None

        if thinking:
            return ReasoningResponseData(content=text, reasoning=thinking)
        return self.make_text_response_data(text)

    def _parse_streaming_event(
        self,
        response: InferenceServerResponse,
        json_obj: JsonObject,
        event_type: str,
    ) -> ParsedResponse | None:
        """Parse a streaming SSE event from the Anthropic Messages API."""
        match event_type:
            case EventType.MESSAGE_START:
                message = json_obj.get("message", {})
                usage = message.get("usage")
                if usage:
                    return ParsedResponse(perf_ns=response.perf_ns, usage=usage)
                return None

            case EventType.CONTENT_BLOCK_DELTA:
                return self._parse_content_block_delta(response, json_obj)

            case EventType.MESSAGE_DELTA:
                usage = json_obj.get("usage")
                if usage:
                    return ParsedResponse(perf_ns=response.perf_ns, usage=usage)
                return None

            case (
                EventType.PING
                | EventType.CONTENT_BLOCK_START
                | EventType.CONTENT_BLOCK_STOP
                | EventType.MESSAGE_STOP
            ):
                return None

            case EventType.ERROR:
                error_detail = json_obj.get("error", {})
                self.warning(
                    lambda: f"Anthropic streaming error: "
                    f"type={error_detail.get('type')}, "
                    f"message={error_detail.get('message')}"
                )
                return None

            case _:
                self.debug(lambda: f"Unknown Anthropic SSE event type: {event_type!r}")
                return None

    def _parse_content_block_delta(
        self, response: InferenceServerResponse, json_obj: JsonObject
    ) -> ParsedResponse | None:
        """Parse a ``content_block_delta`` SSE event.

        Split out of ``_parse_streaming_event`` so that method stays under
        the cyclomatic-complexity guardrail; the delta has its own
        sub-dispatch on ``delta.type``.
        """
        delta = json_obj.get("delta", {})
        delta_type = delta.get("type")

        if delta_type == DeltaType.TEXT_DELTA:
            text = delta.get("text")
            if text:
                return ParsedResponse(
                    perf_ns=response.perf_ns,
                    data=TextResponseData(text=text),
                )
            return None

        if delta_type == DeltaType.THINKING_DELTA:
            thinking = delta.get("thinking")
            if thinking:
                return ParsedResponse(
                    perf_ns=response.perf_ns,
                    data=ReasoningResponseData(reasoning=thinking),
                )
            return None

        # input_json_delta / signature_delta / unknown -> drop silently;
        # they carry tool-call argument fragments and signature material
        # that the streaming text/thinking accumulators don't consume.
        return None
