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


def test_prompt_contract_is_valid_json_without_range_placeholders() -> None:
    example = StrategyParser.parse(StrategyParser.ACTION_CONTRACT)
    assert example.entry.type == "channel_breakout"
    assert ".." not in StrategyParser.ACTION_CONTRACT

    prompt = MarketData.synthetic(seed=17).render_prompt(as_of_index=1_500)
    assert "deliberately inactive syntax example" in prompt
    assert 'entry.type exactly "momentum_threshold"' in prompt
    assert "aliases and alternative layouts are invalid" in prompt


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
    pbo, contributions = ProbabilityBacktestOverfitting.decompose(window_scores)
    assert pbo == pytest.approx(17.0 / 70.0)
    assert np.sum(contributions) == pytest.approx(pbo)


def test_cscv_splits_loss_credit_across_tied_winners() -> None:
    tied_scores = np.ones((2, 8), dtype=float)
    pbo, contributions = ProbabilityBacktestOverfitting.decompose(tied_scores)
    assert pbo == 0.0
    assert contributions[0] == pytest.approx(contributions[1])
    assert np.sum(contributions) == pytest.approx(pbo)


def test_lower_tail_window_score_averages_the_two_weakest_windows() -> None:
    returns = np.asarray(
        [
            [0.01, 0.01],
            [-0.01, -0.01],
            [0.02, 0.02],
            [-0.02, -0.02],
            [0.00, 0.00],
            [0.03, 0.03],
            [0.01, 0.00],
            [0.02, 0.00],
        ]
    )
    expected = np.mean(
        [
            np.sum(np.log1p(returns[1])),
            np.sum(np.log1p(returns[3])),
        ]
    )
    score = StrategyBacktester.lower_tail_window_score(
        returns,
        reference_window_risk=1.0,
    )
    assert score == pytest.approx(expected)


def test_tail_score_uses_exogenous_risk_and_does_not_reward_flat_returns() -> None:
    flat = np.zeros((8, 42), dtype=float)
    weak = flat.copy()
    weak[:2] = -0.001
    flat_score = StrategyBacktester.lower_tail_window_score(flat, 0.10)
    weak_score = StrategyBacktester.lower_tail_window_score(weak, 0.10)
    assert flat_score == 0.0
    assert weak_score < flat_score


def test_daily_expected_shortfall_reports_the_mean_tail_loss() -> None:
    returns = np.asarray([-0.05, -0.03, -0.01, 0.01, 0.02])
    assert StrategyBacktester.expected_shortfall(
        returns, tail_fraction=0.40
    ) == pytest.approx(0.04)


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


def test_reward_tiers_make_every_pass_outrank_the_best_nonpass() -> None:
    scorer = HonestScorer(StrategyBacktester(market=MarketData.synthetic(seed=17)))
    least_complex = 5.0 / 8.0
    most_complex = 6.0 / 8.0

    worst_pass = scorer._shaped_reward(
        dsr=np.nextafter(scorer.DSR_PASS_THRESHOLD, 1.0),
        window_tail_score=np.nextafter(scorer.WINDOW_TAIL_PASS_THRESHOLD, 1.0),
        expected_shortfall_ratio=np.nextafter(
            scorer.EXPECTED_SHORTFALL_PASS_THRESHOLD, 0.0
        ),
        complexity=most_complex,
        active_fraction=scorer.MIN_ACTIVE_FRACTION,
        active_windows=scorer.MIN_ACTIVE_WINDOWS,
    )
    best_dsr_failure = scorer._shaped_reward(
        dsr=scorer.DSR_PASS_THRESHOLD,
        window_tail_score=scorer.WINDOW_TAIL_ROBUST_TARGET,
        expected_shortfall_ratio=scorer.EXPECTED_SHORTFALL_ROBUST_TARGET,
        complexity=least_complex,
        active_fraction=1.0,
        active_windows=8,
    )
    best_tail_failure = scorer._shaped_reward(
        dsr=1.0,
        window_tail_score=scorer.WINDOW_TAIL_PASS_THRESHOLD,
        expected_shortfall_ratio=scorer.EXPECTED_SHORTFALL_ROBUST_TARGET,
        complexity=least_complex,
        active_fraction=1.0,
        active_windows=8,
    )
    best_expected_shortfall_failure = scorer._shaped_reward(
        dsr=1.0,
        window_tail_score=scorer.WINDOW_TAIL_ROBUST_TARGET,
        expected_shortfall_ratio=scorer.EXPECTED_SHORTFALL_PASS_THRESHOLD,
        complexity=least_complex,
        active_fraction=1.0,
        active_windows=8,
    )

    assert worst_pass > max(
        best_dsr_failure,
        best_tail_failure,
        best_expected_shortfall_failure,
    )


def test_fail_tier_rewards_only_the_binding_gate_gap() -> None:
    scorer = HonestScorer(StrategyBacktester(market=MarketData.synthetic(seed=17)))
    common = {
        "complexity": scorer.MIN_COMPLEXITY,
        "expected_shortfall_ratio": scorer.EXPECTED_SHORTFALL_ROBUST_TARGET,
        "active_fraction": 1.0,
        "active_windows": 8,
    }

    tail_binding = scorer._shaped_reward(
        dsr=0.91, window_tail_score=-2.0, **common
    )
    tail_binding_with_perfect_dsr = scorer._shaped_reward(
        dsr=1.0, window_tail_score=-2.0, **common
    )
    closer_tail = scorer._shaped_reward(
        dsr=1.0, window_tail_score=-1.0, **common
    )
    dsr_binding = scorer._shaped_reward(
        dsr=0.50, window_tail_score=0.0, **common
    )
    dsr_binding_with_perfect_tail = scorer._shaped_reward(
        dsr=0.50, window_tail_score=1.0, **common
    )

    assert tail_binding == pytest.approx(tail_binding_with_perfect_dsr)
    assert closer_tail > tail_binding
    assert dsr_binding == pytest.approx(dsr_binding_with_perfect_tail)

    expected_shortfall_binding = scorer._shaped_reward(
        dsr=1.0,
        window_tail_score=scorer.WINDOW_TAIL_ROBUST_TARGET,
        expected_shortfall_ratio=10.0,
        complexity=scorer.MIN_COMPLEXITY,
        active_fraction=1.0,
        active_windows=8,
    )
    expected_shortfall_binding_with_extra_tail = scorer._shaped_reward(
        dsr=1.0,
        window_tail_score=1.0,
        expected_shortfall_ratio=10.0,
        complexity=scorer.MIN_COMPLEXITY,
        active_fraction=1.0,
        active_windows=8,
    )
    closer_expected_shortfall = scorer._shaped_reward(
        dsr=1.0,
        window_tail_score=1.0,
        expected_shortfall_ratio=7.0,
        complexity=scorer.MIN_COMPLEXITY,
        active_fraction=1.0,
        active_windows=8,
    )
    assert expected_shortfall_binding == pytest.approx(
        expected_shortfall_binding_with_extra_tail
    )
    assert closer_expected_shortfall > expected_shortfall_binding


def test_pass_tier_rewards_the_weaker_robustness_margin() -> None:
    scorer = HonestScorer(StrategyBacktester(market=MarketData.synthetic(seed=17)))
    common = {
        "complexity": scorer.MIN_COMPLEXITY,
        "expected_shortfall_ratio": scorer.EXPECTED_SHORTFALL_ROBUST_TARGET,
        "active_fraction": 1.0,
        "active_windows": 8,
    }

    gate_tap = scorer._shaped_reward(
        dsr=0.91,
        window_tail_score=np.nextafter(scorer.WINDOW_TAIL_PASS_THRESHOLD, 1.0),
        **common,
    )
    perfect_dsr_same_tail = scorer._shaped_reward(
        dsr=1.0,
        window_tail_score=np.nextafter(scorer.WINDOW_TAIL_PASS_THRESHOLD, 1.0),
        **common,
    )
    robust_pass = scorer._shaped_reward(
        dsr=0.98,
        window_tail_score=scorer.WINDOW_TAIL_ROBUST_TARGET,
        **common,
    )

    assert gate_tap == pytest.approx(perfect_dsr_same_tail)
    assert robust_pass > gate_tap

    expected_shortfall_gate_tap = scorer._shaped_reward(
        dsr=0.98,
        window_tail_score=scorer.WINDOW_TAIL_ROBUST_TARGET,
        expected_shortfall_ratio=np.nextafter(
            scorer.EXPECTED_SHORTFALL_PASS_THRESHOLD, 0.0
        ),
        complexity=scorer.MIN_COMPLEXITY,
        active_fraction=1.0,
        active_windows=8,
    )
    assert robust_pass > expected_shortfall_gate_tap


def test_activity_gate_blocks_near_no_trade_strategies() -> None:
    scorer = HonestScorer(StrategyBacktester(market=MarketData.synthetic(seed=17)))
    robust_es = scorer.EXPECTED_SHORTFALL_ROBUST_TARGET
    assert not scorer._passes_gate(1.0, 0.0, robust_es, 0.099, 8)
    assert not scorer._passes_gate(1.0, 0.0, robust_es, 1.0, 3)
    assert scorer._passes_gate(1.0, 0.0, robust_es, 0.10, 4)


def test_expected_shortfall_gate_blocks_concentrated_downside() -> None:
    scorer = HonestScorer(StrategyBacktester(market=MarketData.synthetic(seed=17)))
    assert not scorer._passes_gate(
        1.0,
        scorer.WINDOW_TAIL_ROBUST_TARGET,
        scorer.EXPECTED_SHORTFALL_PASS_THRESHOLD,
        1.0,
        8,
    )
    assert scorer._passes_gate(
        1.0,
        scorer.WINDOW_TAIL_ROBUST_TARGET,
        np.nextafter(scorer.EXPECTED_SHORTFALL_PASS_THRESHOLD, 0.0),
        1.0,
        8,
    )


def test_pbo_is_diagnostic_and_does_not_affect_per_strategy_reward() -> None:
    scorer = HonestScorer(StrategyBacktester(market=MarketData.synthetic(seed=17)))
    honest = StrategyParser.parse(json.dumps(HONEST_CONFIG))
    first, second = scorer.score_group([honest, honest], as_of_index=1_500)
    assert first.pbo == pytest.approx(second.pbo)
    assert first.pbo_contribution == pytest.approx(second.pbo_contribution)
    assert first.reward == pytest.approx(second.reward)


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
