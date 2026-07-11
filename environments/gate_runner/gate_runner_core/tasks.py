from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gate_runner_core.market import MarketData


@dataclass(frozen=True)
class TaskRecord:
    """Platform-neutral representation of one hidden-cutoff episode."""

    question: str
    as_of_index: int
    as_of_date: str
    data_source: str
    answer: str = ""

    @property
    def info(self) -> dict[str, int | str]:
        return {
            "as_of_index": self.as_of_index,
            "as_of_date": self.as_of_date,
            "data_source": self.data_source,
        }

    def to_dict(self, task_id: str | None = None) -> dict[str, object]:
        row: dict[str, object] = {
            "question": self.question,
            "answer": self.answer,
            "info": self.info,
        }
        if task_id is not None:
            row["task_id"] = task_id
        return row


class TaskFactory:
    """Build embargoed train/eval tasks without choosing a dataset framework."""

    def __init__(
        self,
        market: MarketData,
        windows: int,
        window_days: int,
        seed: int,
    ) -> None:
        self.market = market
        self.windows = windows
        self.window_days = window_days
        self.seed = seed

    def build(
        self,
        train_examples: int,
        eval_examples: int,
    ) -> tuple[tuple[TaskRecord, ...], tuple[TaskRecord, ...]]:
        if train_examples < 1 or eval_examples < 1:
            raise ValueError("train_examples and eval_examples must both be positive")
        horizon = self.windows * self.window_days
        first_start = 300
        last_start = len(self.market.dates) - horizon - 1
        split_start = first_start + int(0.70 * (last_start - first_start))
        train_candidates = np.arange(
            first_start,
            split_start - horizon + 1,
            5,
            dtype=int,
        )
        eval_candidates = np.arange(split_start, last_start, 5, dtype=int)
        if train_examples > len(train_candidates):
            raise ValueError(
                f"requested {train_examples} train examples but the embargoed "
                f"panel only supports {len(train_candidates)}"
            )
        if eval_examples > len(eval_candidates):
            raise ValueError(
                f"requested {eval_examples} eval examples but the held-out "
                f"panel only supports {len(eval_candidates)}"
            )
        rng = np.random.default_rng(self.seed)
        train_indices = rng.permutation(train_candidates)[:train_examples]
        eval_indices = rng.permutation(eval_candidates)[:eval_examples]
        return self._records(train_indices), self._records(eval_indices)

    def _records(self, indices: np.ndarray) -> tuple[TaskRecord, ...]:
        return tuple(
            TaskRecord(
                question=self.market.render_prompt(int(as_of_index)),
                as_of_index=int(as_of_index),
                as_of_date=self.market.dates[int(as_of_index)],
                data_source=self.market.source_label,
            )
            for as_of_index in indices
        )
