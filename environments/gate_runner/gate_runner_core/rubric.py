from typing import Any

import verifiers as vf

from gate_runner_core.config import StrategyConfig, StrategyParser
from gate_runner_core.scoring import HonestScore, HonestScorer


class HonestRubric(vf.Rubric):
    METRIC_NAMES = (
        "validity",
        "raw_sharpe",
        "dsr",
        "pbo",
        "pbo_contribution",
        "window_tail_score",
        "reference_window_risk",
        "daily_expected_shortfall",
        "expected_shortfall_ratio",
        "complexity",
        "parameter_count",
        "trial_count",
        "passed",
        "turnover",
        "carry_contribution",
        "active_fraction",
        "active_windows",
    )

    def __init__(self, scorer: HonestScorer) -> None:
        self.scorer = scorer
        super().__init__(funcs=[self.honest_reward], weights=[1.0])
        for metric_name in self.METRIC_NAMES:
            self.add_metric(self._metric_function(metric_name))

    async def honest_reward(
        self,
        completions: list[object],
        infos: list[dict[str, Any]],
        states: list[vf.State],
    ) -> list[float]:
        strategies: list[StrategyConfig | None] = []
        errors: list[str] = []
        for completion in completions:
            try:
                strategies.append(
                    StrategyParser.parse(StrategyParser.completion_text(completion))
                )
                errors.append("")
            except ValueError as exc:
                strategies.append(None)
                errors.append(str(exc))

        as_of_indices = {int(info["as_of_index"]) for info in infos}
        if len(as_of_indices) != 1:
            scores = [HonestScore(trial_count=float(len(states))) for _ in states]
            errors = ["rollout group mixed multiple episode cutoffs"] * len(states)
        else:
            scores = self.scorer.score_group(
                strategies=strategies,
                as_of_index=as_of_indices.pop(),
            )
        for state, score, error in zip(states, scores, errors):
            state["gate_runner_score"] = score.metrics()
            state["gate_runner_error"] = error
        return [score.reward for score in scores]

    @staticmethod
    def _metric_function(metric_name: str):
        def metric(state: vf.State) -> float:
            score = state.get("gate_runner_score", {})
            return float(score.get(metric_name, 0.0))

        metric.__name__ = metric_name
        return metric
