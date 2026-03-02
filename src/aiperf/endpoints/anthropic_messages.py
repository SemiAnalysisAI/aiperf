# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from aiperf.common.enums import CaseInsensitiveStrEnum
from aiperf.common.models import (
    BaseResponseData,
    InferenceServerResponse,
    ParsedResponse,
    ReasoningResponseData,
    RequestInfo,
    TextResponseData,
    Turn,
)
from aiperf.common.types import JsonObject
from aiperf.endpoints.base_endpoint import BaseEndpoint

_DEFAULT_ROLE: str = "user"
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
    """

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
        messages, system = self._create_messages(
            turns, request_info.system_message, request_info.user_context_message
        )

        payload: dict[str, Any] = {
            "model": turns[-1].model or model_endpoint.primary_model_name,
            "messages": messages,
            "max_tokens": turns[-1].max_tokens
            if turns[-1].max_tokens is not None
            else 1024,
            "stream": model_endpoint.endpoint.streaming,
        }

        if system:
            payload["system"] = system

        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)

        self.trace(lambda: f"Formatted payload: {payload}")
        return payload

    def _create_messages(
        self,
        turns: list[Turn],
        system_message: str | None,
        user_context_message: str | None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Create messages and extract system prompt for Anthropic Messages API.

        Args:
            turns: List of turns in the request
            system_message: Optional shared system message (becomes top-level system param)
            user_context_message: Optional per-conversation user context to prepend

        Returns:
            Tuple of (messages list, system string or None)
        """
        messages: list[dict[str, Any]] = []

        if user_context_message:
            messages.append({"role": "user", "content": user_context_message})

        for turn in turns:
            role = turn.role or _DEFAULT_ROLE
            content = self._build_turn_content(turn)
            messages.append({"role": role, "content": content})

        return messages, system_message

    def _build_turn_content(self, turn: Turn) -> str | list[dict[str, Any]]:
        """Build content for a single turn.

        Returns a plain string for simple single-text turns,
        or a list of content blocks for complex turns.
        """
        has_images = len(turn.images) > 0
        has_audios = len(turn.audios) > 0
        has_videos = len(turn.videos) > 0

        if has_audios:
            self.warning(
                lambda: "Anthropic Messages API does not support audio content blocks; "
                "audio inputs will be dropped"
            )
        if has_videos:
            self.warning(
                lambda: "Anthropic Messages API does not support video content blocks; "
                "video inputs will be dropped"
            )

        is_simple = (
            len(turn.texts) == 1
            and len(turn.texts[0].contents) == 1
            and not has_images
            and not has_audios
            and not has_videos
        )
        if is_simple:
            return turn.texts[0].contents[0] if turn.texts[0].contents else ""

        content_blocks: list[dict[str, Any]] = []
        for text in turn.texts:
            for item in text.contents:
                if not item:
                    continue
                content_blocks.append({"type": ContentBlockType.TEXT, "text": item})

        for image in turn.images:
            for item in image.contents:
                if not item:
                    continue
                content_blocks.append(
                    {
                        "type": "image",
                        "source": {"type": "url", "url": item},
                    }
                )

        return content_blocks

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
                delta = json_obj.get("delta", {})
                delta_type = delta.get("type")

                if delta_type == DeltaType.TEXT_DELTA:
                    text = delta.get("text")
                    if text:
                        return ParsedResponse(
                            perf_ns=response.perf_ns,
                            data=TextResponseData(text=text),
                        )

                elif delta_type == DeltaType.THINKING_DELTA:
                    thinking = delta.get("thinking")
                    if thinking:
                        return ParsedResponse(
                            perf_ns=response.perf_ns,
                            data=ReasoningResponseData(reasoning=thinking),
                        )

                elif delta_type in (
                    DeltaType.INPUT_JSON_DELTA,
                    DeltaType.SIGNATURE_DELTA,
                ):
                    return None

                return None

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
