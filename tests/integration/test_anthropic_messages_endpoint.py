# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for /v1/messages endpoint."""

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
class TestAnthropicMessagesEndpoint:
    """Tests for /v1/messages endpoint."""

    async def test_basic_messages(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Basic non-streaming Anthropic Messages request."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model claude-sonnet-4-20250514 \
                --tokenizer gpt2 \
                --url {aiperf_mock_server.url} \
                --endpoint-type anthropic_messages \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count

    async def test_streaming_messages(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Streaming Anthropic Messages with metrics validation."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model claude-sonnet-4-20250514 \
                --tokenizer gpt2 \
                --url {aiperf_mock_server.url} \
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
