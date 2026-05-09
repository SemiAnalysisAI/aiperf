# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Module-private helpers for ``AnthropicMessagesEndpoint``.

Split out of ``anthropic_messages.py`` to keep the main endpoint module
under the file-size guardrail. All callers route through
``AnthropicMessagesEndpoint``: ``extract_payload_inputs`` consumes the
walk helpers; ``build_assistant_turn`` consumes the streaming-event
absorber and finaliser.
"""

from __future__ import annotations

import contextlib
from typing import Any

import orjson

from aiperf.common.models import ExtractedPayload
from aiperf.common.types import JsonObject

# String literals match ``ContentBlockType`` / ``DeltaType`` / ``EventType``
# values defined in anthropic_messages.py. Duplicated here as plain strings
# rather than imported to keep this helpers module a leaf in the import
# graph (avoids a circular import with the endpoint module).
_TYPE_TEXT = "text"
_TYPE_TOOL_USE = "tool_use"
_TYPE_TOOL_RESULT = "tool_result"
_TYPE_MESSAGE = "message"
_TYPE_CONTENT_BLOCK_START = "content_block_start"
_TYPE_CONTENT_BLOCK_DELTA = "content_block_delta"
_DELTA_TEXT = "text_delta"
_DELTA_INPUT_JSON = "input_json_delta"


# ---------------------------------------------------------------------------
# Read side: payload -> ExtractedPayload (extends base walk for Anthropic shapes)
# ---------------------------------------------------------------------------


def walk_system(payload: dict[str, Any], result: ExtractedPayload) -> None:
    """Prepend the top-level ``system`` field to ``result.texts``.

    Accepts both string and list-of-content-parts shapes (the Anthropic
    spec permits either). List form items must be ``{"type":"text","text":...}``;
    other types are skipped (the spec reserves them for future use).
    """
    system = payload.get("system")
    if isinstance(system, str):
        if system:
            result.texts.insert(0, system)
        return
    if not isinstance(system, list):
        return
    collected: list[str] = []
    for part in system:
        if isinstance(part, dict) and part.get("type") == _TYPE_TEXT:
            text = part.get(_TYPE_TEXT)
            if isinstance(text, str) and text:
                collected.append(text)
        elif isinstance(part, str) and part:
            collected.append(part)
    for text in reversed(collected):
        result.texts.insert(0, text)


def walk_tool_schemas(payload: dict[str, Any], result: ExtractedPayload) -> None:
    """Collect ``input_schema`` text from top-level Anthropic tools.

    The base ``_walk_tools_schema`` (called by the inherited
    ``extract_payload_inputs``) already harvests ``name`` and
    ``description`` fields from each tool dict. Anthropic's tool schema
    field is named ``input_schema`` (not OpenAI's ``parameters``), so we
    serialise it ourselves here. Without this the tokeniser undercounts
    every agentic request that declares tools.
    """
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        input_schema = tool.get("input_schema")
        if isinstance(input_schema, dict):
            with contextlib.suppress(TypeError):
                result.texts.append(orjson.dumps(input_schema).decode())


def walk_tool_blocks(payload: dict[str, Any], result: ExtractedPayload) -> None:
    """Collect tokenisable text from ``tool_use`` and ``tool_result`` content blocks.

    The base content-part walk dispatches via ``PART_TYPES`` and only
    knows about media (text/image/audio/video). Anthropic's agentic
    history replay also includes:

    - ``{"type":"tool_use","id":...,"name":...,"input":{...}}`` -
      assistant blocks. Server tokenises ``name`` and the serialised
      ``input`` JSON.
    - ``{"type":"tool_result","tool_use_id":...,"content":...}`` -
      user-role blocks containing the tool output the model previously
      saw. ``content`` is either a string or a list of
      ``{"type":"text","text":...}`` blocks.

    Without this walk, agent-history replays silently undercount ISL by
    everything inside these blocks.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == _TYPE_TOOL_USE:
                _collect_tool_use(part, result)
            elif part_type == _TYPE_TOOL_RESULT:
                _collect_tool_result(part, result)


def _collect_tool_use(part: dict[str, Any], result: ExtractedPayload) -> None:
    """Collect ``name`` and serialised ``input`` from one tool_use block."""
    name = part.get("name")
    if isinstance(name, str) and name:
        result.texts.append(name)
    input_value = part.get("input")
    if isinstance(input_value, dict):
        with contextlib.suppress(TypeError):
            result.texts.append(orjson.dumps(input_value).decode())


def _collect_tool_result(part: dict[str, Any], result: ExtractedPayload) -> None:
    """Collect text from one tool_result block.

    ``content`` is either a string (legacy shorthand) or a list of
    ``{"type":"text","text":...}`` blocks. Other block types (image,
    etc.) are skipped here - image content already counts via the base
    walk's image branch when it encounters the part's ``type``.
    """
    content = part.get("content")
    if isinstance(content, str):
        if content:
            result.texts.append(content)
        return
    if not isinstance(content, list):
        return
    for sub in content:
        if not isinstance(sub, dict):
            continue
        if sub.get("type") == _TYPE_TEXT:
            text = sub.get(_TYPE_TEXT)
            if isinstance(text, str) and text:
                result.texts.append(text)


# ---------------------------------------------------------------------------
# Replay side: assistant response -> tool_use accumulator
# ---------------------------------------------------------------------------


def absorb_event(
    json_obj: JsonObject,
    text_parts: list[str],
    tool_uses_by_index: dict[int, dict[str, Any]],
) -> None:
    """Fold one Anthropic response payload (streaming SSE or non-streaming
    ``message``) into the running assistant accumulators.

    Non-streaming ``type=message`` responses already carry the full
    ``content`` array; streaming responses arrive as a sequence of
    ``content_block_start`` (with the empty block envelope, including
    ``index``) and ``content_block_delta`` (with ``text_delta`` or
    ``input_json_delta`` fragments) events that must be reassembled.
    """
    event_type = json_obj.get("type")
    if event_type == _TYPE_MESSAGE:
        _absorb_message(json_obj, text_parts, tool_uses_by_index)
    elif event_type == _TYPE_CONTENT_BLOCK_START:
        _absorb_content_block_start(json_obj, tool_uses_by_index)
    elif event_type == _TYPE_CONTENT_BLOCK_DELTA:
        _absorb_content_block_delta(json_obj, text_parts, tool_uses_by_index)


def _absorb_message(
    json_obj: JsonObject,
    text_parts: list[str],
    tool_uses_by_index: dict[int, dict[str, Any]],
) -> None:
    """Non-streaming ``type=message``: walk the full ``content`` array."""
    for block in json_obj.get("content") or []:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == _TYPE_TEXT:
            text = block.get(_TYPE_TEXT)
            if isinstance(text, str):
                text_parts.append(text)
        elif block_type == _TYPE_TOOL_USE:
            idx = len(tool_uses_by_index)
            tool_uses_by_index[idx] = {
                "id": block.get("id"),
                "name": block.get("name"),
                "input": block.get("input"),
            }


def _absorb_content_block_start(
    json_obj: JsonObject,
    tool_uses_by_index: dict[int, dict[str, Any]],
) -> None:
    """Streaming ``content_block_start``: open a tool_use accumulator slot."""
    block = json_obj.get("content_block") or {}
    if block.get("type") != _TYPE_TOOL_USE:
        return
    idx = json_obj.get("index", len(tool_uses_by_index))
    tool_uses_by_index[idx] = {
        "id": block.get("id"),
        "name": block.get("name"),
        # Input streams in as JSON fragments via input_json_delta;
        # accumulate the raw string here, parse once at finalise.
        "_input_json": "",
    }


def _absorb_content_block_delta(
    json_obj: JsonObject,
    text_parts: list[str],
    tool_uses_by_index: dict[int, dict[str, Any]],
) -> None:
    """Streaming ``content_block_delta``: dispatch text vs input_json fragments."""
    delta = json_obj.get("delta") or {}
    delta_type = delta.get("type")
    if delta_type == _DELTA_TEXT:
        text = delta.get(_TYPE_TEXT)
        if isinstance(text, str):
            text_parts.append(text)
        return
    if delta_type != _DELTA_INPUT_JSON:
        return
    idx = json_obj.get("index")
    if idx is None or idx not in tool_uses_by_index:
        return
    fragment = delta.get("partial_json") or ""
    if isinstance(fragment, str):
        tool_uses_by_index[idx]["_input_json"] = (
            tool_uses_by_index[idx].get("_input_json", "") + fragment
        )


def finalise_tool_use(accumulator: dict[str, Any]) -> dict[str, Any]:
    """Convert a streaming/non-streaming tool_use accumulator into a wire block.

    Streaming accumulators carry ``_input_json`` (raw concatenated
    fragments from ``input_json_delta`` chunks); we parse once at the
    end and drop the raw string so the resulting block round-trips
    through ``build_messages`` unchanged. Malformed JSON is preserved as
    a string under ``input`` so the request still serialises (the server
    will reject it loudly rather than us silently dropping data).
    """
    if "_input_json" in accumulator:
        raw = accumulator.pop("_input_json")
        if raw:
            try:
                accumulator["input"] = orjson.loads(raw)
            except orjson.JSONDecodeError:
                accumulator["input"] = raw
        else:
            accumulator.setdefault("input", {})
    block = {"type": _TYPE_TOOL_USE}
    block.update({k: v for k, v in accumulator.items() if v is not None})
    return block
