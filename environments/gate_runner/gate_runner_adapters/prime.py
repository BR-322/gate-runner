from __future__ import annotations

import json
from dataclasses import fields
from typing import Any

import verifiers as vf
from datasets import Dataset

from gate_runner_core.benchmark import GateRunnerBenchmark
from gate_runner_core.evaluator import GroupEvaluator
from gate_runner_core.scoring import HonestScore
from gate_runner_core.tasks import TaskRecord


def _completion_text(completion: object) -> str:
    if isinstance(completion, str):
        return completion
    if not isinstance(completion, list):
        return ""
    for message in reversed(completion):
        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif isinstance(getattr(part, "text", None), str):
                    parts.append(part.text)
            if parts:
                return "".join(parts)
    return ""


class HonestRubric(vf.Rubric):
    """Verifiers state/metric adapter over the neutral grouped evaluator."""

    METRIC_NAMES = tuple(
        field.name for field in fields(HonestScore) if field.name != "reward"
    )

    def __init__(self, evaluator: GroupEvaluator) -> None:
        self.evaluator = evaluator
        super().__init__(funcs=[self.honest_reward], weights=[1.0])
        for metric_name in self.METRIC_NAMES:
            self.add_metric(self._metric_function(metric_name))

    async def honest_reward(
        self,
        completions: list[object],
        infos: list[dict[str, Any]],
        states: list[vf.State],
    ) -> list[float]:
        completion_texts = [
            _completion_text(completion) for completion in completions
        ]
        as_of_indices = {int(info["as_of_index"]) for info in infos}
        if len(as_of_indices) != 1:
            records = self.evaluator.invalid_group(
                count=len(states),
                error="rollout group mixed multiple episode cutoffs",
            )
        else:
            records = self.evaluator.evaluate(
                completions=completion_texts,
                as_of_index=as_of_indices.pop(),
            )
        for state, record in zip(states, records):
            state["gate_runner_score"] = record.score.metrics()
            state["gate_runner_error"] = record.error
        return [record.reward for record in records]

    @staticmethod
    def _metric_function(metric_name: str):
        def metric(state: vf.State) -> float:
            score = state.get("gate_runner_score", {})
            return float(score.get(metric_name, 0.0))

        metric.__name__ = metric_name
        return metric


def _to_huggingface_dataset(tasks: tuple[TaskRecord, ...]) -> Dataset:
    return Dataset.from_list(
        [
            {
                "question": task.question,
                "answer": task.answer,
                "info": json.dumps(task.info),
            }
            for task in tasks
        ]
    )


def load_environment(
    seed: int = 17,
    train_examples: int = 64,
    eval_examples: int = 24,
    windows: int = 8,
    window_days: int = 42,
    dataset: str = "synthetic",
    data_path: str | None = None,
) -> vf.Environment:
    """Load the Prime/Verifiers adapter without changing benchmark semantics."""
    benchmark = GateRunnerBenchmark(
        seed=seed,
        windows=windows,
        window_days=window_days,
        dataset=dataset,
        data_path=data_path,
    )
    train_tasks, eval_tasks = benchmark.build_tasks(
        train_examples=train_examples,
        eval_examples=eval_examples,
    )
    return vf.SingleTurnEnv(
        dataset=_to_huggingface_dataset(train_tasks),
        eval_dataset=_to_huggingface_dataset(eval_tasks),
        rubric=HonestRubric(evaluator=benchmark.evaluator),
        pass_threshold=0.5,
    )
