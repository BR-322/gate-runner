import asyncio
import json

import numpy as np
import pytest

from gate_runner import load_environment
from gate_runner_core.config import StrategyParser
from gate_runner_core.market import ECB_FX_CURRENCIES, MarketData, TaskDatasetFactory
from gate_runner_core.scoring import (
    DeflatedSharpeRatio,
    HonestScorer,
    ProbabilityBacktestOverfitting,
    StrategyBacktester,
)


HONEST_CONFIG = {
    "entry": {
        "type": "momentum_threshold",
        "lookback_days": 120,
        "threshold": 0.02,
    },
    "exit": {"type": "time_exit", "max_holding_days": 63},
    "universe_filter": {"side": "top", "k": 5},
    "sizing": {"method": "equal_weight", "max_positions": 5},
}

OVERFIT_CONFIG = {
    "entry": {
        "type": "channel_breakout",
        "lookback_days": 10,
        "buffer_pct": 0.0,
        "confirmation_days": 1,
    },
    "exit": {"type": "trailing_stop", "trail_pct": 0.02},
    "universe_filter": {"side": "top", "k": 1},
    "sizing": {"method": "equal_weight", "max_positions": 1},
}


def test_parser_accepts_only_one_schema_clean_json_object() -> None:
    parsed = StrategyParser.parse(json.dumps(HONEST_CONFIG))
    assert parsed.entry.type == "momentum_threshold"
    assert parsed.parameter_count == 5

    fenced = f"```json\n{json.dumps(HONEST_CONFIG)}\n```"
    with pytest.raises(ValueError, match="markdown fences"):
        StrategyParser.parse(fenced)

    out_of_bounds = json.loads(json.dumps(HONEST_CONFIG))
    out_of_bounds["sizing"]["max_positions"] = 6
    with pytest.raises(ValueError, match="schema violation"):
        StrategyParser.parse(json.dumps(out_of_bounds))

    extra_key = json.loads(json.dumps(HONEST_CONFIG))
    extra_key["leak"] = True
    with pytest.raises(ValueError, match="schema violation"):
        StrategyParser.parse(json.dumps(extra_key))


def test_market_brief_is_strictly_point_in_time() -> None:
    market = MarketData.synthetic(seed=17)
    as_of_index = 1_500
    original = market.feature_snapshot(as_of_index)
    altered_close = market.close.copy()
    altered_close[as_of_index:] *= np.linspace(
        0.5, 2.0, len(market.symbols), dtype=float
    )
    altered = MarketData(
        dates=market.dates,
        symbols=market.symbols,
        close=altered_close,
        spread_bps=market.spread_bps.copy(),
        source_label="future-mutated test panel",
    )
    assert altered.feature_snapshot(as_of_index) == original


def test_bundled_ecb_fx_snapshot_is_complete_and_supports_p3_scale() -> None:
    market = MarketData.ecb_fx()
    assert market.dates[0] == "2009-01-02"
    assert market.dates[-1] == "2024-12-31"
    assert market.close.shape == (4_098, len(ECB_FX_CURRENCIES))
    assert market.symbols == tuple(
        f"EUR{currency}" for currency in ECB_FX_CURRENCIES
    )
    assert np.all(np.isfinite(market.close))
    assert np.all(market.close > 0)
    assert np.all(market.spread_bps == 5.0)

    train_dataset, eval_dataset = TaskDatasetFactory(
        market=market,
        windows=8,
        window_days=42,
        seed=17,
    ).build(train_examples=400, eval_examples=200)
    assert len(train_dataset) == 400
    assert len(eval_dataset) == 200


def test_dsr_deflates_the_same_returns_as_trial_count_grows() -> None:
    returns = np.random.default_rng(4).normal(0.001, 0.01, 336)
    one_trial = DeflatedSharpeRatio.probability(returns, trials=1)
    one_hundred_trials = DeflatedSharpeRatio.probability(returns, trials=100)
    assert one_trial > one_hundred_trials
    assert 0.0 <= one_hundred_trials <= 1.0


def test_cscv_detects_a_split_selected_overfit_strategy() -> None:
    window_scores = np.asarray(
        [
            [4.0, 4.0, 4.0, 4.0, -4.0, -4.0, -4.0, -4.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        ]
    )
    assert ProbabilityBacktestOverfitting.estimate(window_scores) > 0.40


def test_signature_overfit_config_scores_below_parsimonious_config() -> None:
    market = MarketData.synthetic(seed=17)
    scorer = HonestScorer(StrategyBacktester(market=market))
    honest = StrategyParser.parse(json.dumps(HONEST_CONFIG))
    overfit = StrategyParser.parse(json.dumps(OVERFIT_CONFIG))

    honest_score, overfit_score = scorer.score_group(
        [honest, overfit], as_of_index=1_500
    )
    assert honest_score.raw_sharpe > overfit_score.raw_sharpe
    assert honest_score.reward > overfit_score.reward + 0.10


def test_invalid_completion_fails_closed_to_exactly_zero() -> None:
    environment = load_environment(train_examples=1, eval_examples=1)
    honest_rubric = environment.rubric.rubrics[0]
    info = json.loads(environment.eval_dataset[0]["info"])
    completions = [
        [{"role": "assistant", "content": json.dumps(HONEST_CONFIG)}],
        [{"role": "assistant", "content": "not json"}],
    ]
    states = [{}, {}]

    rewards = asyncio.run(
        honest_rubric.honest_reward(
            completions=completions,
            infos=[info, info],
            states=states,
        )
    )
    assert rewards[0] != 0.0
    assert rewards[1] == 0.0
    assert states[1]["gate_runner_score"]["validity"] == 0.0
    assert states[1]["gate_runner_error"].startswith("invalid JSON")


def test_environment_exposes_disjoint_train_and_eval_cutoffs() -> None:
    environment = load_environment(
        seed=23,
        train_examples=4,
        eval_examples=3,
    )
    train_cutoffs = {
        json.loads(row["info"])["as_of_index"] for row in environment.dataset
    }
    eval_cutoffs = {
        json.loads(row["info"])["as_of_index"] for row in environment.eval_dataset
    }
    assert train_cutoffs.isdisjoint(eval_cutoffs)
    assert max(train_cutoffs) + 8 * 42 <= min(eval_cutoffs)
    assert environment.requires_group_rollouts
