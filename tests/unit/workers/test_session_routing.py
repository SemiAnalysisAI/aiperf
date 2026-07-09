# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.workers.session_routing import (
    DynamoHeadersRouting,
    DynamoNvextOptions,
    DynamoNvextRouting,
    IdentityHeadersOptions,
    IdentityHeadersRouting,
    RoutingContext,
    SessionIdHeaderRouting,
    SessionRoutingBase,
    SmgRoutingKeyRouting,
)


def _ctx(**overrides) -> RoutingContext:
    defaults = dict(
        x_correlation_id="corr-1",
        parent_correlation_id=None,
        root_correlation_id="corr-1",
        is_final_turn=False,
        is_parent_final=None,
        is_tree_final=False,
    )
    defaults.update(overrides)
    return RoutingContext(**defaults)


class TestDynamoHeadersRouting:
    def test_root_emits_session_header_only(self):
        plugin = DynamoHeadersRouting(DynamoHeadersRouting.Options())
        assert plugin.headers(_ctx()) == {"X-Dynamo-Session-ID": "corr-1"}
        assert plugin.mutates_body is False

    def test_child_emits_parent_header(self):
        plugin = DynamoHeadersRouting(DynamoHeadersRouting.Options())
        headers = plugin.headers(_ctx(parent_correlation_id="parent-1"))
        assert headers == {
            "X-Dynamo-Session-ID": "corr-1",
            "X-Dynamo-Parent-Session-ID": "parent-1",
        }

    def test_body_untouched(self):
        plugin = DynamoHeadersRouting(DynamoHeadersRouting.Options())
        payload = {"messages": []}
        assert plugin.transform_body(payload, _ctx()) is payload


class TestDynamoNvextRouting:
    def test_non_final_turn_binds_with_timeout(self):
        plugin = DynamoNvextRouting(DynamoNvextOptions(timeout_seconds=123))
        merged = plugin.transform_body({"messages": []}, _ctx())
        assert merged["nvext"]["session_control"] == {
            "session_id": "corr-1",
            "action": "bind",
            "timeout": 123,
        }
        assert plugin.mutates_body is True

    def test_final_turn_closes_without_timeout(self):
        plugin = DynamoNvextRouting(DynamoNvextOptions())
        merged = plugin.transform_body({}, _ctx(is_final_turn=True))
        assert merged["nvext"]["session_control"] == {
            "session_id": "corr-1",
            "action": "close",
        }

    def test_never_mutates_input_payload(self):
        nested_sc = {"existing": "keep"}
        nvext = {"trace": "keep", "session_control": nested_sc}
        payload = {"nvext": nvext}
        plugin = DynamoNvextRouting(DynamoNvextOptions())
        merged = plugin.transform_body(payload, _ctx())
        assert payload == {
            "nvext": {"trace": "keep", "session_control": {"existing": "keep"}}
        }
        assert nvext == {"trace": "keep", "session_control": {"existing": "keep"}}
        assert merged is not payload
        assert merged["nvext"]["session_control"]["existing"] == "keep"

    def test_options_default_and_bounds(self):
        assert DynamoNvextOptions().timeout_seconds == 300
        with pytest.raises(ValidationError):
            DynamoNvextOptions(timeout_seconds=0)

    def test_options_reject_unknown_keys(self):
        with pytest.raises(ValidationError):
            DynamoNvextOptions(timeout_secs=5)

    def test_typed_options_access(self):
        plugin = DynamoNvextRouting(DynamoNvextOptions(timeout_seconds=42))
        assert plugin.options.timeout_seconds == 42

    def test_plugin_session_control_wins_over_dataset_shipped_keys(self):
        """Merge precedence: the plugin's live session identity must override
        any session_control keys the dataset shipped, or a recorded
        session_id would leak into live routing."""
        payload = {
            "nvext": {
                "session_control": {
                    "session_id": "recorded-stale",
                    "action": "open",
                    "keep": "me",
                }
            }
        }
        plugin = DynamoNvextRouting(DynamoNvextOptions(timeout_seconds=42))
        merged = plugin.transform_body(payload, _ctx())
        sc = merged["nvext"]["session_control"]
        assert sc["session_id"] == "corr-1"
        assert sc["action"] == "bind"
        assert sc["timeout"] == 42
        assert sc["keep"] == "me"

    def test_nvext_present_without_session_control_preserved(self):
        plugin = DynamoNvextRouting(DynamoNvextOptions())
        merged = plugin.transform_body({"nvext": {"trace": "keep"}}, _ctx())
        assert merged["nvext"]["trace"] == "keep"
        assert merged["nvext"]["session_control"]["action"] == "bind"

    def test_non_dict_nvext_replaced(self):
        """A malformed (non-dict) nvext value is replaced rather than crashed on."""
        plugin = DynamoNvextRouting(DynamoNvextOptions())
        merged = plugin.transform_body({"nvext": "bogus"}, _ctx())
        assert merged["nvext"]["session_control"]["session_id"] == "corr-1"


class TestSmgRoutingKeyRouting:
    def test_emits_routing_key(self):
        plugin = SmgRoutingKeyRouting(SmgRoutingKeyRouting.Options())
        assert plugin.headers(_ctx()) == {"X-SMG-Routing-Key": "corr-1"}

    def test_rejects_any_opt(self):
        with pytest.raises(ValidationError):
            SmgRoutingKeyRouting.Options(anything="x")


class TestSessionIdHeaderRouting:
    def test_emits_x_session_id(self):
        plugin = SessionIdHeaderRouting(SessionIdHeaderRouting.Options())
        assert plugin.headers(_ctx()) == {"X-Session-ID": "corr-1"}
        assert plugin.mutates_body is False

    def test_rejects_any_opt(self):
        with pytest.raises(ValidationError):
            SessionIdHeaderRouting.Options(header_name="X-Affinity")


class TestIdentityHeadersRouting:
    def test_session_tier_single_name(self):
        plugin = IdentityHeadersRouting(IdentityHeadersOptions(session="X-Affinity"))
        assert plugin.headers(_ctx()) == {"X-Affinity": "corr-1"}
        assert plugin.mutates_body is False

    def test_session_tier_multiple_names_comma_separated(self):
        """One mode, N additive headers, same correlation-ID value on each --
        the layered-router topology (e.g. ingress LB + SMG)."""
        plugin = IdentityHeadersRouting(
            IdentityHeadersOptions(session="X-Session-ID, X-SMG-Routing-Key")
        )
        assert plugin.headers(_ctx()) == {
            "X-Session-ID": "corr-1",
            "X-SMG-Routing-Key": "corr-1",
        }

    def test_list_form_not_resplit(self):
        """Canonicalized opts re-enter as lists; must not be re-split."""
        plugin = IdentityHeadersRouting(IdentityHeadersOptions(session=["X-A", "X-B"]))
        assert plugin.headers(_ctx()) == {"X-A": "corr-1", "X-B": "corr-1"}

    def test_parent_tier_omitted_for_roots(self):
        plugin = IdentityHeadersRouting(
            IdentityHeadersOptions(session="X-S", parent="X-P")
        )
        assert plugin.headers(_ctx()) == {"X-S": "corr-1"}

    def test_parent_tier_emitted_for_children(self):
        plugin = IdentityHeadersRouting(
            IdentityHeadersOptions(session="X-S", parent="X-P")
        )
        headers = plugin.headers(_ctx(parent_correlation_id="parent-1"))
        assert headers == {"X-S": "corr-1", "X-P": "parent-1"}

    def test_root_tier_whole_tree_affinity(self):
        plugin = IdentityHeadersRouting(IdentityHeadersOptions(root="X-Tree"))
        headers = plugin.headers(
            _ctx(parent_correlation_id="parent-1", root_correlation_id="root-1")
        )
        assert headers == {"X-Tree": "root-1"}

    def test_root_tier_equals_session_for_roots(self):
        """effective_root == x_corr for root sessions, so root-tier affinity
        degrades gracefully on flat (non-tree) workloads."""
        plugin = IdentityHeadersRouting(
            IdentityHeadersOptions(session="X-S", root="X-Tree")
        )
        assert plugin.headers(_ctx()) == {"X-S": "corr-1", "X-Tree": "corr-1"}

    def test_dynamo_headers_preset_expressible(self):
        """The dynamo_headers preset semantics, spelled via the generic."""
        plugin = IdentityHeadersRouting(
            IdentityHeadersOptions(
                session="X-Dynamo-Session-ID", parent="X-Dynamo-Parent-Session-ID"
            )
        )
        preset = DynamoHeadersRouting(DynamoHeadersRouting.Options())
        for ctx in (_ctx(), _ctx(parent_correlation_id="parent-1")):
            assert plugin.headers(ctx) == preset.headers(ctx)

    def test_no_names_anywhere_rejected(self):
        with pytest.raises(ValidationError, match="at least one header name"):
            IdentityHeadersOptions()

    def test_duplicate_name_across_tiers_rejected_case_insensitive(self):
        with pytest.raises(ValidationError, match="duplicate header name"):
            IdentityHeadersOptions(session="X-Affinity", root="x-affinity")

    def test_duplicate_name_within_tier_rejected(self):
        with pytest.raises(ValidationError, match="duplicate header name"):
            IdentityHeadersOptions(session="X-Affinity,x-affinity")

    def test_empty_name_entry_rejected(self):
        with pytest.raises(ValidationError, match="non-empty"):
            IdentityHeadersOptions(session="X-Affinity,")

    def test_unknown_opt_rejected(self):
        with pytest.raises(ValidationError):
            IdentityHeadersOptions(header_name="X-Affinity")

    @pytest.mark.parametrize(
        "bad_name",
        [
            param("X Foo", id="interior-space"),
            param("X:Foo", id="colon"),
            param("X-Foo\r\nX-Evil: 1", id="crlf-injection"),
            param("X-Fo\u00e9", id="non-ascii"),
        ],
    )  # fmt: skip
    def test_non_token_header_names_rejected(self, bad_name):
        """Names must be RFC 9110 tokens; anything else fails at config time
        instead of corrupting (or injecting into) the wire request."""
        with pytest.raises(ValidationError, match="RFC 9110"):
            IdentityHeadersOptions(session=[bad_name])

    def test_list_form_whitespace_only_name_rejected(self):
        """YAML list inputs are stripped like CLI comma-splits; a
        whitespace-only entry must hit the non-empty check, not emit a
        malformed header."""
        with pytest.raises(ValidationError, match="non-empty"):
            IdentityHeadersOptions(session=["   "])

    def test_list_form_stripped_before_duplicate_check(self):
        """' X-A ' (YAML list) and 'x-a' (another tier) are the same header
        on the wire; stripping must happen before the cross-tier dup check."""
        with pytest.raises(ValidationError, match="duplicate header name"):
            IdentityHeadersOptions(session=[" X-A "], root=["x-a"])

    def test_list_form_names_stripped_in_output(self):
        plugin = IdentityHeadersRouting(IdentityHeadersOptions(session=[" X-A "]))
        assert plugin.headers(_ctx()) == {"X-A": "corr-1"}


class TestBaseDefaults:
    def test_on_session_end_default_noop_and_idempotent(self):
        class Passthrough(SessionRoutingBase):
            pass

        plugin = Passthrough(Passthrough.Options())
        plugin.on_session_end("corr-1")
        plugin.on_session_end("corr-1")

    def test_stateful_open_once_lifecycle_expressible(self):
        """The legacy-nvext shape: open-once instance state, bounded by on_session_end."""

        class OpenOnce(SessionRoutingBase):
            mutates_body = True

            def __init__(self, options):
                super().__init__(options)
                self.opened: set[str] = set()

            def transform_body(self, payload, ctx):
                merged = dict(payload)
                if ctx.x_correlation_id not in self.opened:
                    self.opened.add(ctx.x_correlation_id)
                    merged["action"] = "open"
                return merged

            def on_session_end(self, x_correlation_id):
                self.opened.discard(x_correlation_id)

        plugin = OpenOnce(OpenOnce.Options())
        assert plugin.transform_body({}, _ctx())["action"] == "open"
        assert "action" not in plugin.transform_body({}, _ctx())
        plugin.on_session_end("corr-1")
        plugin.on_session_end("corr-1")  # idempotent
        assert plugin.opened == set()
