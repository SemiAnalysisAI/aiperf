# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for `_build_socket_address` — the Linux/macOS ipc:// vs Windows tcp:// helper.

Covers Bug 2: ZMQ ipc:// is not supported on Windows (pyzmq wheels disable it
due to crashes), so AIPerf falls back to tcp://127.0.0.1:<deterministic-port>
on Windows. Same path/filename inputs must hash to the same port so bind and
connect sides agree without coordination.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from pytest import param

from aiperf.common.config.zmq_config import (
    _WINDOWS_TCP_BASE_PORT,
    _WINDOWS_TCP_PORT_RANGE,
    _build_socket_address,
)


class TestBuildSocketAddressLinux:
    """`_build_socket_address` returns ipc:// when not on Windows."""

    @pytest.fixture(autouse=True)
    def _force_not_windows(self):
        with patch("aiperf.common.config.zmq_config.IS_WINDOWS", False):
            yield

    def test_returns_ipc_url_with_path_and_filename(self, tmp_path: Path) -> None:
        address = _build_socket_address(tmp_path, "event_bus.ipc")
        assert address == f"ipc://{tmp_path / 'event_bus.ipc'}"

    def test_path_none_raises_value_error(self) -> None:
        with pytest.raises(
            ValueError, match="Path is required for socket address derivation"
        ):
            _build_socket_address(None, "event_bus.ipc")


class TestBuildSocketAddressWindows:
    """`_build_socket_address` returns tcp:// with deterministic port on Windows."""

    @pytest.fixture(autouse=True)
    def _force_windows(self):
        with patch("aiperf.common.config.zmq_config.IS_WINDOWS", True):
            yield

    def test_returns_tcp_loopback_url(self, tmp_path: Path) -> None:
        address = _build_socket_address(tmp_path, "event_bus.ipc")
        assert address.startswith("tcp://127.0.0.1:")

    def test_port_within_configured_range(self, tmp_path: Path) -> None:
        address = _build_socket_address(tmp_path, "event_bus.ipc")
        port = int(address.rsplit(":", 1)[1])
        assert (
            _WINDOWS_TCP_BASE_PORT
            <= port
            < _WINDOWS_TCP_BASE_PORT + _WINDOWS_TCP_PORT_RANGE
        )

    def test_same_inputs_produce_same_port(self, tmp_path: Path) -> None:
        addr1 = _build_socket_address(tmp_path, "event_bus.ipc")
        addr2 = _build_socket_address(tmp_path, "event_bus.ipc")
        assert addr1 == addr2

    def test_different_filenames_produce_different_ports(self, tmp_path: Path) -> None:
        addr1 = _build_socket_address(tmp_path, "event_bus.ipc")
        addr2 = _build_socket_address(tmp_path, "credit_router.ipc")
        assert addr1 != addr2

    def test_different_paths_produce_different_ports(self, tmp_path: Path) -> None:
        other_path = tmp_path / "other"
        other_path.mkdir()
        addr1 = _build_socket_address(tmp_path, "event_bus.ipc")
        addr2 = _build_socket_address(other_path, "event_bus.ipc")
        assert addr1 != addr2

    def test_path_none_raises_value_error(self) -> None:
        with pytest.raises(
            ValueError, match="Path is required for socket address derivation"
        ):
            _build_socket_address(None, "event_bus.ipc")

    @pytest.mark.parametrize(
        "filename",
        [
            param("event_bus_proxy_frontend.ipc", id="event_bus_frontend"),
            param("event_bus_proxy_backend.ipc", id="event_bus_backend"),
            param("records_push_pull.ipc", id="records"),
            param("credit_router.ipc", id="credit"),
            param("dataset_manager_proxy_frontend.ipc", id="dataset_frontend"),
            param("dataset_manager_proxy_backend.ipc", id="dataset_backend"),
            param("raw_inference_proxy_frontend.ipc", id="raw_inference_frontend"),
            param("raw_inference_proxy_backend.ipc", id="raw_inference_backend"),
        ],
    )
    def test_realistic_filenames_all_within_range(
        self, tmp_path: Path, filename: str
    ) -> None:
        address = _build_socket_address(tmp_path, filename)
        port = int(address.rsplit(":", 1)[1])
        assert (
            _WINDOWS_TCP_BASE_PORT
            <= port
            < _WINDOWS_TCP_BASE_PORT + _WINDOWS_TCP_PORT_RANGE
        )


class TestBuildSocketAddressHashDistribution:
    """Sanity check that the hash distributes across the port range."""

    @pytest.fixture(autouse=True)
    def _force_windows(self):
        with patch("aiperf.common.config.zmq_config.IS_WINDOWS", True):
            yield

    def test_realistic_socket_set_has_no_collisions(self, tmp_path: Path) -> None:
        """Within a single AIPerf run, the 8 production socket filenames must hash to distinct ports.

        If this ever fails, increase _WINDOWS_TCP_PORT_RANGE or change a filename.
        Birthday paradox at RANGE=20000 with n=8 sockets: ~0.14% collision chance.
        """
        filenames = [
            "event_bus_proxy_frontend.ipc",
            "event_bus_proxy_backend.ipc",
            "records_push_pull.ipc",
            "credit_router.ipc",
            "dataset_manager_proxy_frontend.ipc",
            "dataset_manager_proxy_backend.ipc",
            "raw_inference_proxy_frontend.ipc",
            "raw_inference_proxy_backend.ipc",
        ]
        ports = {
            int(_build_socket_address(tmp_path, fn).rsplit(":", 1)[1])
            for fn in filenames
        }
        assert len(ports) == len(filenames), (
            f"Hash collision detected for the production socket set: "
            f"{len(filenames)} sockets but only {len(ports)} unique ports. "
            f"Consider widening _WINDOWS_TCP_PORT_RANGE."
        )
