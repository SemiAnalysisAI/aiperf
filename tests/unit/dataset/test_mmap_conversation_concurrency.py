# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concurrency regression tests for MemoryMapDatasetClient.get_conversation().

The store dispatches it via ``run_in_executor``, so several threads read one
shared ``mmap``. Any reader that depends on the mmap's file position can then
serve another conversation's bytes.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest

from aiperf.common.enums import MemoryMapFormat
from aiperf.common.models import Conversation, Text, Turn
from aiperf.dataset.memory_map_utils import (
    MemoryMapDatasetBackingStore,
    MemoryMapDatasetClient,
)

# Distinct, wildly varying sizes so a cross-read lands mid-record and is
# detected either as a decode error or as a mismatched session_id.
_CONVERSATION_COUNT = 24
_THREADS = 8
_LOOKUPS_PER_THREAD = 150


def _make_conversation(index: int) -> Conversation:
    """Conversation whose size grows with ``index`` and whose text encodes it."""
    text = f"conv-{index}:" + ("x" * (64 * (index + 1)))
    return Conversation(
        session_id=f"conv-{index}",
        turns=[Turn(role="user", texts=[Text(contents=[text])])],
    )


@pytest.mark.asyncio
async def test_get_conversation_is_thread_safe(tmp_path, monkeypatch):
    """Concurrent get_conversation() calls must not read each other's bytes."""
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))

    store = MemoryMapDatasetBackingStore(
        benchmark_id="test_conv_concurrency", format=MemoryMapFormat.CONVERSATION
    )
    await store.initialize()
    for i in range(_CONVERSATION_COUNT):
        await store.add_conversation(f"conv-{i}", _make_conversation(i))
    await store.finalize()

    metadata = store.get_client_metadata()
    client = MemoryMapDatasetClient(
        metadata.data_file_path,
        metadata.index_file_path,
    )

    def _read_many(thread_index: int) -> None:
        for n in range(_LOOKUPS_PER_THREAD):
            # Stagger the starting id per thread so the interleaving covers
            # many (offset, size) pairs rather than one hot record.
            i = (thread_index + n) % _CONVERSATION_COUNT
            conv = client.get_conversation(f"conv-{i}")
            assert conv.session_id == f"conv-{i}"

    with ThreadPoolExecutor(max_workers=_THREADS) as pool:
        # list() forces every future to be resolved so exceptions propagate.
        list(pool.map(_read_many, range(_THREADS)))


@pytest.mark.asyncio
async def test_get_conversation_ignores_shared_mmap_position(tmp_path, monkeypatch):
    """A stale mmap file position must not affect what get_conversation reads."""
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))

    store = MemoryMapDatasetBackingStore(
        benchmark_id="test_conv_position", format=MemoryMapFormat.CONVERSATION
    )
    await store.initialize()
    for i in range(3):
        await store.add_conversation(f"conv-{i}", _make_conversation(i))
    await store.finalize()

    metadata = store.get_client_metadata()
    client = MemoryMapDatasetClient(
        metadata.data_file_path,
        metadata.index_file_path,
    )

    # Simulate another thread having just moved the shared position.
    client.data_mmap.seek(client.index.offsets["conv-2"].offset)
    assert client.get_conversation("conv-0").session_id == "conv-0"
