from __future__ import annotations

from pathlib import Path
from typing import Sequence

from gate_runner_core.evaluator import EvaluationRecord, GroupEvaluator
from gate_runner_core.market import MarketData
from gate_runner_core.scoring import HonestScorer, StrategyBacktester
from gate_runner_core.tasks import TaskFactory, TaskRecord


DATASETS = ("synthetic", "ecb_fx", "ecb_fx_carry")


class GateRunnerBenchmark:
    """Platform-neutral task generation and grouped scoring interface."""

    def __init__(
        self,
        seed: int = 17,
        windows: int = 8,
        window_days: int = 42,
        dataset: str = "synthetic",
        data_path: str | Path | None = None,
    ) -> None:
        if windows < 4 or windows % 2:
            raise ValueError("windows must be an even integer of at least 4 for CSCV")
        if window_days < 20:
            raise ValueError("window_days must be at least 20")
        self.seed = seed
        self.windows = windows
        self.window_days = window_days
        self.dataset = dataset
        self.data_path = Path(data_path) if data_path is not None else None
        self.market = self._load_market()
        self.task_factory = TaskFactory(
            market=self.market,
            windows=windows,
            window_days=window_days,
            seed=seed,
        )
        self.scorer = HonestScorer(
            backtester=StrategyBacktester(
                market=self.market,
                windows=windows,
                window_days=window_days,
            )
        )
        self.evaluator = GroupEvaluator(self.scorer)

    def _load_market(self) -> MarketData:
        if self.data_path is not None:
            if self.dataset != "synthetic":
                raise ValueError(
                    "data_path and a non-default dataset cannot be combined"
                )
            return MarketData.from_csv(self.data_path)
        if self.dataset == "synthetic":
            return MarketData.synthetic(seed=self.seed)
        if self.dataset == "ecb_fx":
            return MarketData.ecb_fx()
        if self.dataset == "ecb_fx_carry":
            return MarketData.ecb_fx(include_carry=True)
        allowed = ", ".join(repr(value) for value in DATASETS)
        raise ValueError(f"dataset must be one of {allowed}")

    def build_tasks(
        self,
        train_examples: int = 64,
        eval_examples: int = 24,
    ) -> tuple[tuple[TaskRecord, ...], tuple[TaskRecord, ...]]:
        return self.task_factory.build(
            train_examples=train_examples,
            eval_examples=eval_examples,
        )

    def evaluate_group(
        self,
        completions: Sequence[str],
        as_of_index: int,
    ) -> tuple[EvaluationRecord, ...]:
        return self.evaluator.evaluate(
            completions=completions,
            as_of_index=as_of_index,
        )
