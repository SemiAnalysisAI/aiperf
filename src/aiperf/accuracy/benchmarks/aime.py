# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIME benchmark loader.

Loads AIME competition math problems from Maxwell-Jia/AIME_2024.
Designed to be paired with MathGrader for numerical answer equivalence.
"""

from __future__ import annotations

import asyncio

from datasets import Dataset, load_dataset

from aiperf.accuracy.models import BenchmarkProblem
from aiperf.common.config import UserConfig
from aiperf.common.mixins import AIPerfLoggerMixin

DATASET_NAME = "Maxwell-Jia/AIME_2024"
TASK_NAME = "aime"
GENERATION_SIZE = 1024


class AIMEBenchmark(AIPerfLoggerMixin):
    """AIME (American Invitational Mathematics Examination) benchmark loader.

    Loads all problems from Maxwell-Jia/AIME_2024 (train split).
    Answers are non-negative integers; paired with MathGrader.
    """

    def __init__(self, user_config: UserConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.user_config = user_config

    async def load_problems(
        self, tasks: list[str] | None, n_shots: int, enable_cot: bool
    ) -> list[BenchmarkProblem]:
        ds: Dataset = await asyncio.to_thread(load_dataset, DATASET_NAME, split="train")
        few_shots = self._build_few_shots(ds, n_shots)

        problems: list[BenchmarkProblem] = []
        for row in ds:
            prompt = self._format_prompt(row, few_shots, enable_cot)
            chat_msgs = self._build_chat_messages(row, few_shots, enable_cot)
            problems.append(
                BenchmarkProblem(
                    prompt=prompt,
                    ground_truth=str(row["Answer"]),
                    task=TASK_NAME,
                    metadata={"generation_size": GENERATION_SIZE},
                    few_shot_examples=few_shots,
                    chat_messages=chat_msgs,
                )
            )
        return problems

    def _build_few_shots(self, ds: Dataset, n_shots: int) -> list[dict[str, str]]:
        if n_shots <= 0:
            return []
        return [
            self._format_example(row) for row in ds.select(range(min(n_shots, len(ds))))
        ]

    def _format_example(self, row: dict) -> dict[str, str]:
        answer = str(row["Answer"])
        return {
            "problem": row["Problem"],
            "answer": answer,
            "formatted": f"Problem: {row['Problem']}\nAnswer: {answer}",
        }

    def _format_prompt(
        self,
        row: dict,
        few_shots: list[dict[str, str]],
        enable_cot: bool,
    ) -> str:
        instruction = (
            "Solve the following competition math problem. "
            "The answer is a non-negative integer.\n\n"
        )

        few_shot_text = "\n\n".join(ex["formatted"] for ex in few_shots)
        if few_shot_text:
            few_shot_text += "\n\n"

        if enable_cot:
            query = f"Problem: {row['Problem']}\nLet's think step by step.\nAnswer:"
        else:
            query = f"Problem: {row['Problem']}\nAnswer:"

        return instruction + few_shot_text + query

    def _build_chat_messages(
        self,
        row: dict,
        few_shots: list[dict[str, str]],
        enable_cot: bool,
    ) -> list[dict[str, str]]:
        """Build multi-turn chat messages following lighteval's PromptManager style.

        - First few-shot user message includes the instruction prefix.
        - Each few-shot answer is a separate assistant message.
        - Main query follows the same format without the instruction prefix.
        """
        instruction = (
            "Solve the following competition math problem. "
            "The answer is a non-negative integer.\n\n"
        )

        messages: list[dict[str, str]] = []

        for ix, ex in enumerate(few_shots):
            q = f"Problem: {ex['problem']}\nAnswer:"
            if ix == 0:
                q = instruction + q
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": ex["answer"]})

        if enable_cot:
            main_q = f"Problem: {row['Problem']}\nLet's think step by step.\nAnswer:"
        else:
            main_q = f"Problem: {row['Problem']}\nAnswer:"

        if not few_shots:
            main_q = instruction + main_q

        messages.append({"role": "user", "content": main_q})
        return messages
