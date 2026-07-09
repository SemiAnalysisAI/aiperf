# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytest import param

from aiperf.common.config import EndpointConfig
from aiperf.common.config.endpoint_config import _parse_session_routing_opts


def _config(**kwargs) -> EndpointConfig:
    return EndpointConfig(model_names=["test-model"], **kwargs)


def test_defaults_off():
    config = _config()
    assert config.session_routing is None
    assert config.session_routing_opts == {}


def test_mode_with_valid_opts():
    config = _config(
        session_routing="dynamo_nvext",
        session_routing_opt=["timeout_seconds=600"],
    )
    assert str(config.session_routing) == "dynamo_nvext"


def test_opts_canonicalized_to_typed_values():
    config = _config(
        session_routing="dynamo_nvext",
        session_routing_opt=["timeout_seconds=600"],
    )
    assert config.session_routing_opts == {"timeout_seconds": 600}
    assert isinstance(config.session_routing_opts["timeout_seconds"], int)


def test_canonicalization_from_config_file_dict():
    """Opts set directly (config-file form) are canonicalized the same way."""
    config = _config(
        session_routing="dynamo_nvext",
        session_routing_opts={"timeout_seconds": 600},
    )
    assert config.session_routing_opts == {"timeout_seconds": 600}


def test_identity_headers_tiers_canonicalized_to_lists():
    """Comma-separated tier values canonicalize to typed lists, so the
    pickled UserConfig that reaches workers re-validates without re-splitting."""
    config = _config(
        session_routing="identity_headers",
        session_routing_opt=[
            "session=X-Session-ID,X-SMG-Routing-Key",
            "root=X-Tree-ID",
        ],
    )
    assert config.session_routing_opts == {
        "session": ["X-Session-ID", "X-SMG-Routing-Key"],
        "root": ["X-Tree-ID"],
    }


def test_identity_headers_no_names_rejected():
    with pytest.raises(ValueError, match="at least one header name"):
        _config(session_routing="identity_headers")


def test_session_id_header_preset_rejects_opts():
    """The preset lost its header_name option when identity_headers took over
    customization; it must reject what it used to accept."""
    with pytest.raises(ValueError):
        _config(
            session_routing="session_id_header",
            session_routing_opt=["header_name=X-Affinity"],
        )


def test_opts_without_mode_rejected():
    with pytest.raises(ValueError, match="session-routing-opt"):
        _config(session_routing_opt=["session=X-A"])


def test_unknown_opt_key_rejected():
    with pytest.raises(ValueError, match="timeout_secs"):
        _config(
            session_routing="dynamo_nvext",
            session_routing_opt=["timeout_secs=600"],
        )


def test_invalid_opt_value_rejected():
    with pytest.raises(ValueError):
        _config(
            session_routing="dynamo_nvext",
            session_routing_opt=["timeout_seconds=0"],
        )


def test_parameterless_mode_rejects_any_opt():
    with pytest.raises(ValueError):
        _config(
            session_routing="smg_routing_key",
            session_routing_opt=["anything=x"],
        )


def test_parse_routing_opts_duplicate_key_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        _parse_session_routing_opts(["session=X-A", "session=X-B"])


@pytest.mark.parametrize(
    "item",
    [
        param("noequals", id="no_separator"),
        param("key=", id="empty_value"),
    ],
)  # fmt: skip
def test_parse_routing_opts_malformed_pair_rejected(item):
    with pytest.raises(ValueError, match="expected non-empty key=value"):
        _parse_session_routing_opts([item])
