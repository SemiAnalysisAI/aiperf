# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from aiperf.accuracy.benchmarks.aime import GENERATION_SIZE, TASK_NAME, AIMEBenchmark
from aiperf.accuracy.models import BenchmarkProblem
from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.config.accuracy_config import AccuracyConfig
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType


def _make_user_config() -> UserConfig:
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.COMPLETIONS,
            streaming=False,
        ),
        accuracy=AccuracyConfig(benchmark=AccuracyBenchmarkType.AIME),
    )


def _make_row(problem: str = "What is 1+1?", answer: int = 2) -> dict:
    return {"Problem": problem, "Answer": answer}


def _make_fake_dataset(rows: list[dict]) -> MagicMock:
    ds = MagicMock()
    ds.__iter__ = MagicMock(side_effect=lambda: iter(rows))
    ds.__len__ = MagicMock(return_value=len(rows))
    ds.select = MagicMock(side_effect=lambda indices: [rows[i] for i in indices])
    return ds


class TestAIMEBenchmarkFormatPrompt:
    def setup_method(self) -> None:
        self.bench = AIMEBenchmark(user_config=_make_user_config())

    def test_format_prompt_zero_shot_no_cot(self) -> None:
        row = _make_row("Find x if x+5=10.", 5)
        prompt = self.bench._format_prompt(row, few_shots=[], enable_cot=False)

        assert "The answer is a non-negative integer" in prompt
        assert "Find x if x+5=10." in prompt
        assert prompt.endswith("Answer:")
        assert "step by step" not in prompt

    def test_format_prompt_zero_shot_with_cot(self) -> None:
        row = _make_row("Find x if x+5=10.", 5)
        prompt = self.bench._format_prompt(row, few_shots=[], enable_cot=True)

        assert "Let's think step by step" in prompt
        assert prompt.endswith("Answer:")

    def test_format_prompt_instruction_prefix_present(self) -> None:
        row = _make_row("Compute 2^10.", 1024)
        prompt = self.bench._format_prompt(row, few_shots=[], enable_cot=False)

        assert prompt.startswith("Solve the following competition math problem.")

    def test_format_prompt_with_few_shots_includes_examples(self) -> None:
        shot = self.bench._format_example(_make_row("1+1?", 2))
        row = _make_row("2+2?", 4)
        prompt = self.bench._format_prompt(row, few_shots=[shot], enable_cot=False)

        assert "1+1?" in prompt
        assert "Answer: 2" in prompt
        assert "2+2?" in prompt

    def test_format_prompt_few_shot_precedes_test_problem(self) -> None:
        shot = self.bench._format_example(_make_row("1+1?", 2))
        row = _make_row("2+2?", 4)
        prompt = self.bench._format_prompt(row, few_shots=[shot], enable_cot=False)

        assert prompt.index("1+1?") < prompt.index("2+2?")


class TestAIMEBenchmarkBuildChatMessages:
    def setup_method(self) -> None:
        self.bench = AIMEBenchmark(user_config=_make_user_config())

    def test_zero_shot_produces_single_user_message(self) -> None:
        row = _make_row("Solve x=5.", 5)
        msgs = self.bench._build_chat_messages(row, few_shots=[], enable_cot=False)

        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert "Solve x=5." in msgs[0]["content"]

    def test_zero_shot_instruction_in_first_user_message(self) -> None:
        row = _make_row("Solve x=5.", 5)
        msgs = self.bench._build_chat_messages(row, few_shots=[], enable_cot=False)

        assert "non-negative integer" in msgs[0]["content"]

    def test_cot_adds_step_by_step_to_user_message(self) -> None:
        row = _make_row("Solve x=5.", 5)
        msgs = self.bench._build_chat_messages(row, few_shots=[], enable_cot=True)

        assert "step by step" in msgs[-1]["content"]

    def test_few_shot_produces_user_assistant_pairs_plus_query(self) -> None:
        shot = self.bench._format_example(_make_row("1+1?", 2))
        row = _make_row("2+2?", 4)
        msgs = self.bench._build_chat_messages(row, few_shots=[shot], enable_cot=False)

        assert len(msgs) == 3
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert msgs[2]["role"] == "user"

    def test_first_few_shot_user_message_has_instruction(self) -> None:
        shot = self.bench._format_example(_make_row("1+1?", 2))
        row = _make_row("2+2?", 4)
        msgs = self.bench._build_chat_messages(row, few_shots=[shot], enable_cot=False)

        assert "non-negative integer" in msgs[0]["content"]

    def test_subsequent_few_shot_user_messages_omit_instruction(self) -> None:
        shots = [
            self.bench._format_example(_make_row("1+1?", 2)),
            self.bench._format_example(_make_row("3+3?", 6)),
        ]
        row = _make_row("4+4?", 8)
        msgs = self.bench._build_chat_messages(row, few_shots=shots, enable_cot=False)

        assert "non-negative integer" not in msgs[2]["content"]

    def test_assistant_message_contains_answer(self) -> None:
        shot = self.bench._format_example(_make_row("1+1?", 2))
        row = _make_row("2+2?", 4)
        msgs = self.bench._build_chat_messages(row, few_shots=[shot], enable_cot=False)

        assert msgs[1]["content"] == "2"


class TestAIMEBenchmarkFormatExample:
    def setup_method(self) -> None:
        self.bench = AIMEBenchmark(user_config=_make_user_config())

    def test_format_example_fields(self) -> None:
        row = _make_row("What is 3+3?", 6)
        example = self.bench._format_example(row)

        assert example["problem"] == "What is 3+3?"
        assert example["answer"] == "6"
        assert "Problem:" in example["formatted"]
        assert "Answer: 6" in example["formatted"]

    def test_format_example_answer_is_string(self) -> None:
        row = _make_row("Compute.", 42)
        example = self.bench._format_example(row)

        assert isinstance(example["answer"], str)


class TestAIMEBenchmarkLoadProblems:
    @pytest.mark.asyncio
    async def test_load_problems_returns_benchmark_problems(self) -> None:
        rows = [_make_row("Problem A", 100), _make_row("Problem B", 200)]
        fake_ds = _make_fake_dataset(rows)

        with patch(
            "aiperf.accuracy.benchmarks.aime.asyncio.to_thread",
            return_value=fake_ds,
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )

        assert len(problems) == 2
        assert all(isinstance(p, BenchmarkProblem) for p in problems)

    @pytest.mark.asyncio
    async def test_load_problems_ground_truth_is_string(self) -> None:
        rows = [_make_row("Some problem", 42)]
        fake_ds = _make_fake_dataset(rows)

        with patch(
            "aiperf.accuracy.benchmarks.aime.asyncio.to_thread",
            return_value=fake_ds,
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )

        assert problems[0].ground_truth == "42"

    @pytest.mark.asyncio
    async def test_load_problems_task_name(self) -> None:
        rows = [_make_row("Some problem", 7)]
        fake_ds = _make_fake_dataset(rows)

        with patch(
            "aiperf.accuracy.benchmarks.aime.asyncio.to_thread",
            return_value=fake_ds,
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )

        assert problems[0].task == TASK_NAME

    @pytest.mark.asyncio
    async def test_load_problems_metadata_includes_generation_size(self) -> None:
        rows = [_make_row("Some problem", 7)]
        fake_ds = _make_fake_dataset(rows)

        with patch(
            "aiperf.accuracy.benchmarks.aime.asyncio.to_thread",
            return_value=fake_ds,
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )

        assert problems[0].metadata["generation_size"] == GENERATION_SIZE

    @pytest.mark.asyncio
    async def test_load_problems_chat_messages_set(self) -> None:
        rows = [_make_row("Some problem", 7)]
        fake_ds = _make_fake_dataset(rows)

        with patch(
            "aiperf.accuracy.benchmarks.aime.asyncio.to_thread",
            return_value=fake_ds,
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )

        assert problems[0].chat_messages is not None
        assert isinstance(problems[0].chat_messages, list)

    @pytest.mark.asyncio
    async def test_load_problems_n_shots_passes_few_shots(self) -> None:
        rows = [_make_row(f"Problem {i}", i) for i in range(5)]
        fake_ds = _make_fake_dataset(rows)

        with patch(
            "aiperf.accuracy.benchmarks.aime.asyncio.to_thread",
            return_value=fake_ds,
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            problems = await bench.load_problems(
                tasks=None, n_shots=2, enable_cot=False
            )

        assert len(problems[0].few_shot_examples) == 2

    @pytest.mark.asyncio
    async def test_load_problems_zero_shots_has_empty_few_shots(self) -> None:
        rows = [_make_row("Problem X", 3)]
        fake_ds = _make_fake_dataset(rows)

        with patch(
            "aiperf.accuracy.benchmarks.aime.asyncio.to_thread",
            return_value=fake_ds,
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )

        assert problems[0].few_shot_examples == []

    @pytest.mark.asyncio
    async def test_load_problems_tasks_ignored(self) -> None:
        """AIME has no sub-tasks; the tasks parameter is a no-op."""
        rows = [_make_row("Problem A", 1), _make_row("Problem B", 2)]
        fake_ds = _make_fake_dataset(rows)

        with patch(
            "aiperf.accuracy.benchmarks.aime.asyncio.to_thread",
            return_value=fake_ds,
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            all_problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
            filtered = await bench.load_problems(
                tasks=["aime"], n_shots=0, enable_cot=False
            )

        assert len(all_problems) == len(filtered)
