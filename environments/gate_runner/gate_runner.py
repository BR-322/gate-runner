from pathlib import Path

import verifiers as vf

from gate_runner_core.market import MarketData, TaskDatasetFactory
from gate_runner_core.rubric import HonestRubric
from gate_runner_core.scoring import HonestScorer, StrategyBacktester


def load_environment(
    seed: int = 17,
    train_examples: int = 64,
    eval_examples: int = 24,
    windows: int = 8,
    window_days: int = 42,
    dataset: str = "synthetic",
    data_path: str | None = None,
) -> vf.Environment:
    """Load Gate Runner's single-turn strategy-design environment."""
    if train_examples < 1 or eval_examples < 1:
        raise ValueError("train_examples and eval_examples must both be positive")
    if windows < 4 or windows % 2:
        raise ValueError("windows must be an even integer of at least 4 for CSCV")
    if window_days < 20:
        raise ValueError("window_days must be at least 20")

    if data_path is not None:
        if dataset != "synthetic":
            raise ValueError("data_path and a non-default dataset cannot be combined")
        market = MarketData.from_csv(Path(data_path))
    elif dataset == "synthetic":
        market = MarketData.synthetic(seed=seed)
    elif dataset == "ecb_fx":
        market = MarketData.ecb_fx()
    else:
        raise ValueError("dataset must be 'synthetic' or 'ecb_fx'")
    task_factory = TaskDatasetFactory(
        market=market,
        windows=windows,
        window_days=window_days,
        seed=seed,
    )
    train_dataset, eval_dataset = task_factory.build(
        train_examples=train_examples,
        eval_examples=eval_examples,
    )
    scorer = HonestScorer(
        backtester=StrategyBacktester(
            market=market,
            windows=windows,
            window_days=window_days,
        )
    )

    return vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        rubric=HonestRubric(scorer=scorer),
        pass_threshold=0.5,
    )
