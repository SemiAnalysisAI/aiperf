# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.models import (
    ConversationMetadata,
    DatasetMetadata,
    ReplayTurnReference,
    TurnMetadata,
)
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.replay_dependencies import ReplayBarrierCoordinator


def _metadata() -> DatasetMetadata:
    abc = [ReplayTurnReference(conversation_id=name, turn_index=0) for name in "abc"]
    return DatasetMetadata(
        sampling_strategy=DatasetSamplingStrategy.RANDOM,
        conversations=[
            ConversationMetadata(
                conversation_id=name, turns=[TurnMetadata(timestamp_ms=0)]
            )
            for name in "abc"
        ]
        + [
            ConversationMetadata(
                conversation_id="d",
                turns=[TurnMetadata(timestamp_ms=10, replay_predecessors=abc)],
            )
        ],
    )


def _turn(name: str, root: str = "root") -> TurnToSend:
    return TurnToSend(
        conversation_id=name,
        x_correlation_id=f"{root}:{name}",
        turn_index=0,
        num_turns=1,
        root_correlation_id=root,
    )


def _credit(name: str, root: str = "root") -> Credit:
    return Credit(
        id=0,
        phase=CreditPhase.PROFILING,
        conversation_id=name,
        x_correlation_id=f"{root}:{name}",
        turn_index=0,
        num_turns=1,
        issued_at_ns=0,
        root_correlation_id=root,
    )


@pytest.mark.asyncio
async def test_abc_issue_together_and_d_waits_for_every_member() -> None:
    coordinator = ReplayBarrierCoordinator(_metadata())
    coordinator.activate()
    issued: list[str] = []

    async def submit(name: str) -> None:
        await coordinator.submit(
            _turn(name),
            lambda name=name: _record_issue(issued, name),
        )

    await submit("a")
    await submit("b")
    await submit("c")
    await submit("d")
    assert issued == ["a", "b", "c"]

    coordinator.complete(_credit("a"))
    await asyncio.sleep(0)
    assert issued == ["a", "b", "c"]
    coordinator.complete(_credit("b"))
    await asyncio.sleep(0)
    assert issued == ["a", "b", "c"]
    coordinator.complete(_credit("c"))
    await asyncio.sleep(0)
    assert issued == ["a", "b", "c", "d"]


@pytest.mark.asyncio
async def test_any_terminal_outcome_releases_barrier() -> None:
    coordinator = ReplayBarrierCoordinator(_metadata())
    coordinator.activate()
    issued: list[str] = []
    for name in "abcd":
        await coordinator.submit(
            _turn(name), lambda name=name: _record_issue(issued, name)
        )

    for name in "abc":
        coordinator.complete(_credit(name))
    await asyncio.sleep(0)

    assert issued[-1] == "d"


@pytest.mark.asyncio
async def test_runtime_roots_are_independent() -> None:
    coordinator = ReplayBarrierCoordinator(_metadata())
    coordinator.activate()
    issued: list[str] = []
    await coordinator.submit(_turn("d", "one"), lambda: _record_issue(issued, "one:d"))
    await coordinator.submit(_turn("d", "two"), lambda: _record_issue(issued, "two:d"))

    for name in "abc":
        coordinator.complete(_credit(name, "one"))
    await asyncio.sleep(0)

    assert issued == ["one:d"]


@pytest.mark.asyncio
async def test_scalar_peak_would_slip_d_after_only_one_completion() -> None:
    coordinator = ReplayBarrierCoordinator(_metadata())
    coordinator.activate()
    issued: list[str] = []
    for name in "abcd":
        await coordinator.submit(
            _turn(name), lambda name=name: _record_issue(issued, name)
        )

    coordinator.complete(_credit("a"))
    await asyncio.sleep(0)

    assert "d" not in issued


async def _record_issue(issued: list[str], name: str) -> bool:
    issued.append(name)
    return True
