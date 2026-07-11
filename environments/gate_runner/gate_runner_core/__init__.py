"""Platform-neutral task generation and deterministic scoring for Gate Runner."""

from gate_runner_core.benchmark import GateRunnerBenchmark
from gate_runner_core.config import StrategyConfig, StrategyParser
from gate_runner_core.evaluator import EvaluationRecord, GroupEvaluator
from gate_runner_core.market import MarketData
from gate_runner_core.scoring import HonestScore, HonestScorer, StrategyBacktester
from gate_runner_core.tasks import TaskFactory, TaskRecord

__all__ = [
    "EvaluationRecord",
    "GateRunnerBenchmark",
    "GroupEvaluator",
    "HonestScore",
    "HonestScorer",
    "MarketData",
    "StrategyBacktester",
    "StrategyConfig",
    "StrategyParser",
    "TaskFactory",
    "TaskRecord",
]
