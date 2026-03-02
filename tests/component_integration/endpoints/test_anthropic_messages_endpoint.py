# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for /v1/messages endpoint."""

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI


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
