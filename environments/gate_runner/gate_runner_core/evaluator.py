from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from gate_runner_core.config import StrategyConfig, StrategyParser
from gate_runner_core.scoring import HonestScore, HonestScorer


@dataclass(frozen=True)
class EvaluationRecord:
    """One completion's platform-neutral reward, metrics, and parse error."""

    score: HonestScore
    error: str = ""

    @property
    def reward(self) -> float:
        return self.score.reward

    def to_dict(self) -> dict[str, object]:
        return {
            "reward": float(self.score.reward),
            "metrics": self.score.metrics(),
            "error": self.error,
        }


class GroupEvaluator:
    """Parse and score all trials sampled for one episode cutoff together."""

    def __init__(self, scorer: HonestScorer) -> None:
        self.scorer = scorer

    def evaluate(
        self,
        completions: Sequence[str],
        as_of_index: int,
    ) -> tuple[EvaluationRecord, ...]:
        if not completions:
            raise ValueError("completion group must not be empty")
        backtester = self.scorer.backtester
        horizon_end = as_of_index + backtester.windows * backtester.window_days
        if as_of_index < 253 or horizon_end > len(backtester.market.dates):
            raise ValueError("as_of_index does not support the required history and horizon")
        strategies: list[StrategyConfig | None] = []
        errors: list[str] = []
        for completion in completions:
            try:
                strategies.append(StrategyParser.parse(completion))
                errors.append("")
            except ValueError as exc:
                strategies.append(None)
                errors.append(str(exc))
        scores = self.scorer.score_group(
            strategies=strategies,
            as_of_index=as_of_index,
        )
        return tuple(
            EvaluationRecord(score=score, error=error)
            for score, error in zip(scores, errors)
        )

    @staticmethod
    def invalid_group(count: int, error: str) -> tuple[EvaluationRecord, ...]:
        return tuple(
            EvaluationRecord(
                score=HonestScore(trial_count=float(count)),
                error=error,
            )
            for _ in range(count)
        )
