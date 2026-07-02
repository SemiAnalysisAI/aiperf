from pathlib import Path

import orjson

from aiperf.common.environment import Environment
from aiperf.dataset.loader.weka_trace import WekaTraceLoader
from tests.unit.dataset.loader.test_weka_async_subagent import (
    _build_trace,
    _make_loader,
    _normal,
)
from tests.unit.dataset.loader.test_weka_async_subagent import (
    _mk_user_config as _mk_async_user_config,
)
from tests.unit.dataset.loader.test_weka_trace import (
    _mk_user_config,
    _stub_prompt_generator_for_reconstructor,
)


def test_generic_top_level_fa_chains_merge_but_aux_remains(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(Environment.DATASET, "WEKA_AUX_MAX_REQUESTS", 1)
    monkeypatch.setattr(Environment.DATASET, "WEKA_AUX_CROSS_MODEL", True)

    def request(
        t: float,
        model: str,
        hash_ids: list[int],
        input_length: int,
        output_length: int,
    ) -> dict:
        return {
            "t": t,
            "type": "n",
            "model": model,
            "in": input_length,
            "out": output_length,
            "api_time": 1.0,
            "hash_ids": hash_ids,
        }

    trace = {
        "id": "mix",
        "models": ["opus", "haiku"],
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": [
            request(0.0, "opus", [1, 2, 3], 100_000, 100),
            request(2.0, "opus", [1, 2, 50], 20_000, 5000),
            request(3.0, "haiku", [99], 1000, 10),
            request(5.0, "opus", [1, 2, 50, 51], 21_000, 5000),
            request(10.0, "opus", [1, 2, 3, 4], 101_000, 100),
        ],
    }
    path = tmp_path / "mix.json"
    path.write_bytes(orjson.dumps(trace))
    user_config = _mk_user_config()
    user_config.endpoint.model_names = ["opus", "haiku"]
    loader = WekaTraceLoader(
        filename=str(path),
        user_config=user_config,
    )
    _stub_prompt_generator_for_reconstructor(loader)

    convs = {
        c.session_id: c for c in loader.convert_to_conversations(loader.load_dataset())
    }

    assert not any("::fa:" in sid for sid in convs), sorted(convs)
    assert set(convs) == {"mix", "mix::aux:001"}
    assert [turn.timestamp for turn in convs["mix"].turns] == [
        0.0,
        2000.0,
        5000.0,
        10000.0,
    ]
    assert [turn.source_trace_id for turn in convs["mix"].turns] == ["mix"] * 4
    assert [turn.source_outer_idx for turn in convs["mix"].turns] == [0, 1, 3, 4]
    assert [turn.source_inner_idx for turn in convs["mix"].turns] == [None] * 4
    assert [turn.source_kind for turn in convs["mix"].turns] == ["weka_main"] * 4
    assert convs["mix"].branches[0].child_conversation_ids == ["mix::aux:001"]


def test_generic_nested_fa_chains_merge_into_subagent_main(
    tmp_path: Path, monkeypatch
) -> None:
    def inner(t: float, api_time: float, hash_ids: list[int]) -> dict:
        return {
            "t": t,
            "type": "n",
            "model": "m",
            "in": 10,
            "out": 1,
            "api_time": api_time,
            "hash_ids": hash_ids,
        }

    subagent = {
        "t": 1.0,
        "type": "subagent",
        "agent_id": "a1",
        "subagent_type": "X",
        "duration_ms": 20_000,
        "total_tokens": 0,
        "tool_use_count": 0,
        "status": "completed",
        "requests": [
            inner(1.0, 10.0, [1]),
            inner(6.0, 3.0, [50]),
            inner(12.0, 0.5, [1, 2]),
            inner(13.0, 1.0, [50, 51]),
        ],
        "models": ["m"],
    }
    data = _build_trace("t_affinity", [_normal(t=0.0), subagent, _normal(t=30.0)])
    path = tmp_path / "trace.json"
    path.write_bytes(orjson.dumps(data))

    loader = _make_loader(path, _mk_async_user_config(), monkeypatch)
    convs = {
        c.session_id: c for c in loader.convert_to_conversations(loader.load_dataset())
    }

    assert not any(":fa:" in sid for sid in convs), sorted(convs)
    assert set(convs) == {"t_affinity", "t_affinity::sa:a1"}
    assert [turn.timestamp for turn in convs["t_affinity::sa:a1"].turns] == [
        1000.0,
        6000.0,
        12000.0,
        13000.0,
    ]
    assert [
        turn.source_trace_id for turn in convs["t_affinity::sa:a1"].turns
    ] == ["t_affinity"] * 4
    assert [
        turn.source_outer_idx for turn in convs["t_affinity::sa:a1"].turns
    ] == [1] * 4
    assert [
        turn.source_inner_idx for turn in convs["t_affinity::sa:a1"].turns
    ] == [0, 1, 2, 3]
    assert [
        turn.source_kind for turn in convs["t_affinity::sa:a1"].turns
    ] == ["weka_subagent"] * 4
