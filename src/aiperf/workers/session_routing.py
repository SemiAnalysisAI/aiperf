# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session-routing transforms: per-session identity on the wire.

One plugin category unifies every mechanism that tells an external router
which session a request belongs to, whether header-based (SGLang Model
Gateway routing keys, Dynamo session headers, generic tiered identity
headers) or body-based (Dynamo ``nvext.session_control``). The selected
plugin is instantiated once per worker by ``InferenceClient`` and invoked at
the request-serialization chokepoint.

Contracts:
- Options instances are canonicalized at config resolution; ``self.options``
  is the plugin's own typed model.
- ``transform_body`` must NEVER mutate its input (the structured path
  includes cached ``Turn.raw_payload`` dicts shared with the dataset).
- ``on_session_end`` fires strictly AFTER the session's last worker-side
  activity, on every terminal path (final turn, cancellation, terminal
  context overflow, cancel-before-start). It MUST be idempotent.
- Stateful plugins key instance state on ``ctx.x_correlation_id`` ONLY:
  a session tree deliberately spans workers, so tree-keyed worker state
  fragments. Tree-scoped behavior uses the stateless per-request facts
  (``root_correlation_id`` + ``is_tree_final``) instead.
"""

from __future__ import annotations

import re
from abc import ABC
from typing import Annotated, Any, ClassVar, Generic, Literal, TypeVar

from msgspec import Struct
from pydantic import BeforeValidator, ConfigDict, Field, model_validator

from aiperf.common.models import AIPerfBaseModel

# RFC 9110 token: the only characters legal in an HTTP field name. Rejecting
# anything else at config time turns a guaranteed runtime wire failure (or
# header-injection vector via CR/LF) into an immediate, actionable error.
_HEADER_TOKEN_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")


class EmptyRoutingOptions(AIPerfBaseModel):
    """Options model for parameterless plugins; rejects every opt key."""

    model_config = ConfigDict(extra="forbid")


OptionsT = TypeVar("OptionsT", bound=AIPerfBaseModel)


class RoutingContext(Struct, frozen=True, gc=False):
    """Per-request identity facts handed to a routing plugin.

    Field naming mirrors ``RequestInfo`` verbatim. ``is_parent_final`` /
    ``is_tree_final`` are stamped issuer-side from ``SessionTreeRegistry``
    state and are conservative: ``is_tree_final`` is False whenever
    indeterminate, ``is_parent_final`` is None for roots or when unknown.

    msgspec.Struct (not a frozen dataclass): constructed once per request on
    the worker hot path, and frozen-dataclass __init__ goes through
    object.__setattr__ per field (~4x slower to create, measured). gc=False
    is safe: every field is a str/bool/None, so no reference cycles.
    """

    x_correlation_id: str
    """This session's stable key (same on every turn)."""
    parent_correlation_id: str | None
    """Immediate parent session's key; None for root sessions."""
    root_correlation_id: str | None
    """Session-tree root key, verbatim from RequestInfo (never None on the
    dispatch path: the worker passes ``credit.effective_root_correlation_id``)."""
    is_final_turn: bool
    """True when this is the current session's last request."""
    is_parent_final: bool | None
    """True when the parent session had already returned its final turn
    at credit-issue time; None for roots or when not determinable."""
    is_tree_final: bool
    """Best-effort: True only when this is provably the last request the
    whole session tree will send."""


class SessionRoutingBase(ABC, Generic[OptionsT]):  # noqa: B024  # ABC marks the plugin protocol; every method has a working default so passthrough subclasses instantiate directly.
    """Base for session-routing plugins (``session_routing`` category)."""

    mutates_body: ClassVar[bool] = False
    """True when ``transform_body`` changes the payload. Gates the plugin off
    the verbatim PAYLOAD_BYTES mmap fast path at dataset build, cache hit,
    and runtime. CONTRACT: a plugin overrides ``transform_body`` if and only
    if it declares ``mutates_body = True`` -- the two must agree, and
    ``InferenceClient`` enforces this at worker init (fail-fast) because a
    transform without the declaration would be silently skipped on
    PAYLOAD_BYTES datasets, and a declaration without a transform would
    force runs off the fast path for nothing."""

    # ClassVar cannot reference the OptionsT type parameter, so the base type
    # is kept here; subclasses narrow it via their Generic parameterization.
    Options: ClassVar[type[AIPerfBaseModel]] = EmptyRoutingOptions
    """Per-plugin options model, populated from --session-routing-opt
    key=value pairs. Every Options model must set extra='forbid'."""

    def __init__(self, options: OptionsT) -> None:
        self.options: OptionsT = options

    def headers(self, ctx: RoutingContext) -> dict[str, str]:
        """Extra HTTP headers for this request (merged into endpoint headers)."""
        return {}

    def transform_body(
        self, payload: dict[str, Any], ctx: RoutingContext
    ) -> dict[str, Any]:
        """Return a (possibly new) payload dict; never mutate the input."""
        return payload

    def on_session_end(self, x_correlation_id: str) -> None:
        """Post-session cleanup: no further requests will be sent for this
        session by this worker. Idempotent; default no-op."""
        return None


class DynamoHeadersRouting(SessionRoutingBase[EmptyRoutingOptions]):
    """Dynamo session affinity via X-Dynamo-Session-ID / X-Dynamo-Parent-Session-ID.

    Pair with a Dynamo frontend running ``--router-session-affinity-ttl-secs``.
    """

    def headers(self, ctx: RoutingContext) -> dict[str, str]:
        headers = {"X-Dynamo-Session-ID": ctx.x_correlation_id}
        if ctx.parent_correlation_id:
            headers["X-Dynamo-Parent-Session-ID"] = ctx.parent_correlation_id
        return headers


class DynamoNvextOptions(AIPerfBaseModel):
    """Options for the deprecated-upstream nvext.session_control transport."""

    model_config = ConfigDict(extra="forbid")

    timeout_seconds: int = Field(
        default=300,
        ge=1,
        description="Dynamo session_control inactivity timeout carried on every bind.",
    )
    scope: Literal["conversation", "lineage"] = Field(
        default="conversation",
        description="Affinity-key scope. 'conversation' (default) binds each "
        "session with its own correlation ID and closes on its final turn. "
        "'lineage' binds every session in an agent tree with the tree ROOT's "
        "correlation ID so the whole lineage co-locates on the worker holding "
        "the shared parent prefix (for Dynamo deployments without KV-event "
        "prefix indexing); the shared key is closed only on a request stamped "
        "provably-last for the whole tree (is_tree_final, agentic replay) -- "
        "otherwise the session_control TTL reclaims it.",
    )


class DynamoNvextRouting(SessionRoutingBase[DynamoNvextOptions]):
    """Dynamo session affinity via nvext.session_control request-body metadata.

    Modern contract only: 'bind' on every non-final turn (idempotent on the
    router, refreshes the TTL), 'close' on the final turn. Targets Dynamo
    builds that implement session_control; current upstream Dynamo main does
    not (use dynamo_headers there).

    Under ``scope=lineage`` the affinity key is the tree root's correlation ID
    and the close discipline changes: a shared key must never be torn down
    while sibling sessions may still run (a later bind would re-place the
    straggler arbitrarily and lose co-location), so close fires only on a
    request the issuer stamped ``is_tree_final`` -- conservative-exact under
    agentic replay, never under indeterminate modes, where the TTL reaper
    reclaims the key instead.
    """

    mutates_body: ClassVar[bool] = True
    Options: ClassVar[type[AIPerfBaseModel]] = DynamoNvextOptions

    def __init__(self, options: DynamoNvextOptions) -> None:
        super().__init__(options)
        # Hot path: one attribute hop per request instead of two through the
        # pydantic options model.
        self._timeout = options.timeout_seconds
        self._lineage = options.scope == "lineage"

    def transform_body(
        self, payload: dict[str, Any], ctx: RoutingContext
    ) -> dict[str, Any]:
        if self._lineage:
            session_id = ctx.root_correlation_id or ctx.x_correlation_id
            is_close = ctx.is_tree_final
        else:
            session_id = ctx.x_correlation_id
            is_close = ctx.is_final_turn
        if is_close:
            session_control: dict[str, Any] = {
                "session_id": session_id,
                "action": "close",
            }
        else:
            session_control = {
                "session_id": session_id,
                "action": "bind",
                "timeout": self._timeout,
            }
        merged = dict(payload)
        raw_nvext = merged.get("nvext")
        if raw_nvext is None:
            # Common case: dataset payloads carry no nvext. Skip the two
            # defensive dict copies below (~40% faster, measured).
            merged["nvext"] = {"session_control": session_control}
            return merged
        nvext = dict(raw_nvext) if isinstance(raw_nvext, dict) else {}
        raw_sc = nvext.get("session_control")
        if isinstance(raw_sc, dict):
            session_control = {**raw_sc, **session_control}
        nvext["session_control"] = session_control
        merged["nvext"] = nvext
        return merged


class SmgRoutingKeyRouting(SessionRoutingBase[EmptyRoutingOptions]):
    """SGLang Model Gateway manual-policy stickiness via X-SMG-Routing-Key."""

    def headers(self, ctx: RoutingContext) -> dict[str, str]:
        return {"X-SMG-Routing-Key": ctx.x_correlation_id}


class SessionIdHeaderRouting(SessionRoutingBase[EmptyRoutingOptions]):
    """Preset: additive X-Session-ID header carrying the session's correlation ID.

    Equivalent to ``identity_headers`` with ``session=X-Session-ID``; use
    ``identity_headers`` for custom names or parent/root tiers.
    """

    def headers(self, ctx: RoutingContext) -> dict[str, str]:
        return {"X-Session-ID": ctx.x_correlation_id}


def _split_header_names(value: Any) -> Any:
    """Comma-split a raw string opt value into a list of stripped names.

    HTTP header names are RFC 9110 tokens and can never contain a comma, so
    the split is unambiguous. List inputs (canonicalized opts, YAML lists)
    keep their item boundaries but are stripped the same way, so the
    validator's non-empty/duplicate/token checks see identical canonical
    names regardless of input form.
    """
    if isinstance(value, str):
        return [item.strip() for item in value.split(",")]
    if isinstance(value, list):
        return [item.strip() if isinstance(item, str) else item for item in value]
    return value


_HeaderNames = Annotated[list[str], BeforeValidator(_split_header_names)]


class IdentityHeadersOptions(AIPerfBaseModel):
    """Options for the fully generic tiered identity headers.

    Every tier defaults EMPTY: the plugin emits exactly the headers configured
    and nothing else (unlike the presets, which exist to have opinions).
    """

    model_config = ConfigDict(extra="forbid")

    session: _HeaderNames = Field(
        default_factory=list,
        description="Header name(s) carrying this session's correlation ID "
        "(same value on every turn). Comma-separate for multiple names.",
    )
    parent: _HeaderNames = Field(
        default_factory=list,
        description="Header name(s) carrying the immediate parent session's "
        "correlation ID; omitted on requests whose session has no parent. "
        "Comma-separate for multiple names.",
    )
    root: _HeaderNames = Field(
        default_factory=list,
        description="Header name(s) carrying the session-tree root's "
        "correlation ID (the session's own ID for root sessions), keying "
        "whole-tree affinity. Comma-separate for multiple names.",
    )

    @model_validator(mode="after")
    def validate_header_names(self) -> IdentityHeadersOptions:
        """Require >=1 valid RFC 9110 token overall; reject duplicates.

        Uniqueness is case-insensitive over ALL configured names combined
        (within a tier and across tiers): a repeated name means one value
        silently overwriting another -- the exact affinity corruption this
        feature exists to prevent.
        """
        all_names = [*self.session, *self.parent, *self.root]
        if not all_names:
            raise ValueError(
                "identity_headers requires at least one header name across "
                "the session/parent/root tiers."
            )
        for name in all_names:
            if not name:
                raise ValueError("header names must be non-empty")
            if not _HEADER_TOKEN_RE.match(name):
                raise ValueError(
                    f"invalid header name {name!r}: HTTP field names are "
                    "RFC 9110 tokens (letters, digits, and !#$%&'*+-.^_`|~; "
                    "no spaces, colons, or control characters)"
                )
        lowered = [name.lower() for name in all_names]
        if len(set(lowered)) != len(lowered):
            raise ValueError(
                f"duplicate header name (names must be unique across all "
                f"tiers combined) in session={self.session!r} "
                f"parent={self.parent!r} root={self.root!r}"
            )
        return self


class IdentityHeadersRouting(SessionRoutingBase[IdentityHeadersOptions]):
    """Fully generic tiered identity headers: any name(s) per identity tier.

    Stamps each configured header with its tier's correlation ID: ``session``
    (this session), ``parent`` (immediate parent; omitted for roots), and
    ``root`` (session-tree root -- whole-tree affinity, e.g. pinning an entire
    agent tree to one replica for prefix-cache locality). The named presets
    are expressible in terms of this mode: ``session_id_header`` is
    ``session=X-Session-ID``, ``smg_routing_key`` is
    ``session=X-SMG-Routing-Key``, and ``dynamo_headers`` is
    ``session=X-Dynamo-Session-ID`` + ``parent=X-Dynamo-Parent-Session-ID``.
    """

    Options: ClassVar[type[AIPerfBaseModel]] = IdentityHeadersOptions

    def __init__(self, options: IdentityHeadersOptions) -> None:
        super().__init__(options)
        # Hot path: immutable snapshots read once per request, skipping the
        # two-hop pydantic options attribute walk (~17% faster on the common
        # single-tier shape, measured).
        self._session_names = tuple(options.session)
        self._parent_names = tuple(options.parent)
        self._root_names = tuple(options.root)

    def headers(self, ctx: RoutingContext) -> dict[str, str]:
        xcorr = ctx.x_correlation_id
        headers = {name: xcorr for name in self._session_names}
        parent = ctx.parent_correlation_id
        if parent and self._parent_names:
            for name in self._parent_names:
                headers[name] = parent
        root = ctx.root_correlation_id
        if root and self._root_names:
            for name in self._root_names:
                headers[name] = root
        return headers
