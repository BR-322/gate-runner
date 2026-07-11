import asyncio
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gate_runner import load_environment
from gate_runner_cli import main as cli_main
from gate_runner_core.benchmark import GateRunnerBenchmark
from gate_runner_core.config import StrategyParser
from gate_runner_core.market import ECB_FX_CURRENCIES, MarketData
from gate_runner_core.scoring import (
    DeflatedSharpeRatio,
    HonestScorer,
    ProbabilityBacktestOverfitting,
    StrategyBacktester,
)
from gate_runner_core.tasks import TaskFactory


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
    assert parsed.universe_filter.rank_by == "relative_strength_252d"

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
    assert 'rank_by exactly "relative_strength_252d" or "long_eur_carry"' in prompt
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

    train_dataset, eval_dataset = TaskFactory(
        market=market,
        windows=8,
        window_days=42,
        seed=17,
    ).build(train_examples=400, eval_examples=200)
    assert len(train_dataset) == 400
    assert len(eval_dataset) == 200


def test_public_short_rates_cover_redistributable_ecb_pairs_with_pit_features() -> None:
    market = MarketData.ecb_fx(include_carry=True)
    assert market.has_carry
    assert len(market.symbols) == 28
    assert "EURSGD" not in market.symbols
    assert market.carry_returns.shape == market.close.shape
    assert np.all(np.isfinite(market.carry_returns))
    assert np.any(market.carry_returns != 0.0)

    as_of_index = market.dates.index("2014-11-07")
    snapshot = market.feature_snapshot(as_of_index)
    aud = next(row for row in snapshot.rows if row["symbol"] == "EURAUD")
    tenor = 0.25
    expected_forward_points = (
        (
            1.0 + float(aud["foreign_reference_rate_pct"]) / 100.0 * tenor
        )
        / (1.0 + float(aud["eur_reference_rate_pct"]) / 100.0 * tenor)
        - 1.0
    ) * 100.0
    assert aud["long_eur_carry_pct_pa"] == pytest.approx(
        float(aud["eur_reference_rate_pct"])
        - float(aud["foreign_reference_rate_pct"])
    )
    assert aud["cip_forward_points_3m_pct"] == pytest.approx(
        expected_forward_points
    )
    prompt = market.render_prompt(as_of_index)
    assert "long EUR funded in XXX" in prompt
    assert "not an executable forward quote" in prompt


def test_carry_uses_prior_available_rates_and_actual_calendar_days() -> None:
    market = MarketData.ecb_fx(include_carry=True)
    monday_index = market.dates.index("2014-11-10")
    assert market.dates[monday_index - 1] == "2014-11-07"
    aud_index = market.symbols.index("EURAUD")
    expected = (
        market.base_reference_rates_percent[monday_index - 1, aud_index]
        - market.foreign_reference_rates_percent[monday_index - 1, aud_index]
    ) / 100.0 * 3.0 / 365.0
    assert market.carry_returns[monday_index, aud_index] == pytest.approx(expected)
    assert expected < 0.0


def test_future_rate_changes_do_not_change_a_point_in_time_brief() -> None:
    market = MarketData.ecb_fx(include_carry=True)
    as_of_index = market.dates.index("2014-11-07")
    original = market.feature_snapshot(as_of_index)
    foreign_rates = market.foreign_reference_rates_percent.copy()
    base_rates = market.base_reference_rates_percent.copy()
    carry_returns = market.carry_returns.copy()
    foreign_rates[as_of_index:] += 100.0
    base_rates[as_of_index:] -= 100.0
    carry_returns[as_of_index:] *= -10.0
    altered = MarketData(
        dates=market.dates,
        symbols=market.symbols,
        close=market.close.copy(),
        spread_bps=market.spread_bps.copy(),
        source_label="future-rate-mutated test panel",
        carry_returns=carry_returns,
        foreign_reference_rates_percent=foreign_rates,
        base_reference_rates_percent=base_rates,
        rate_source_label="future-rate-mutated test source",
    )
    changed = altered.feature_snapshot(as_of_index)
    assert changed.as_of_date == original.as_of_date
    assert changed.regime == original.regime
    rate_fields = {
        "eur_reference_rate_pct",
        "foreign_reference_rate_pct",
        "long_eur_carry_pct_pa",
        "cip_forward_points_3m_pct",
    }
    for original_row, changed_row in zip(original.rows, changed.rows):
        assert original_row["symbol"] == changed_row["symbol"]
        for field in rate_fields:
            assert original_row[field] == changed_row[field]


def test_spot_only_profile_is_a_carry_ablation() -> None:
    spot_market = MarketData.ecb_fx()
    carry_market = MarketData.ecb_fx(include_carry=True)
    strategy = StrategyParser.parse(json.dumps(HONEST_CONFIG))
    spot_result = StrategyBacktester(spot_market).evaluate(strategy, 1_500)
    carry_result = StrategyBacktester(carry_market).evaluate(strategy, 1_500)
    assert spot_result.carry_contribution == 0.0
    assert carry_result.carry_contribution != 0.0
    assert not np.array_equal(spot_result.daily_returns, carry_result.daily_returns)


def test_carry_ranking_is_actionable_and_fails_closed_without_rates() -> None:
    carry_rank_config = json.loads(json.dumps(HONEST_CONFIG))
    carry_rank_config["entry"]["threshold"] = -0.10
    carry_rank_config["universe_filter"]["rank_by"] = "long_eur_carry"
    carry_rank = StrategyParser.parse(json.dumps(carry_rank_config))

    synthetic_result = StrategyBacktester(MarketData.synthetic(seed=17)).evaluate(
        carry_rank, 1_500
    )
    assert synthetic_result.active_fraction == 0.0
    assert synthetic_result.carry_contribution == 0.0

    carry_market = MarketData.ecb_fx(include_carry=True)
    carry_result = StrategyBacktester(carry_market).evaluate(carry_rank, 1_500)
    relative_config = json.loads(json.dumps(carry_rank_config))
    relative_config["universe_filter"]["rank_by"] = "relative_strength_252d"
    relative_rank = StrategyParser.parse(json.dumps(relative_config))
    relative_result = StrategyBacktester(carry_market).evaluate(relative_rank, 1_500)
    assert carry_result.active_fraction > 0.0
    assert not np.array_equal(carry_result.daily_returns, relative_result.daily_returns)


def test_boe_forward_panel_is_pinned_as_diagnostic_data() -> None:
    path = (
        Path(__file__).parents[1]
        / "gate_runner_core"
        / "data"
        / "boe_gbpusd_spot_forward_2009_2024.csv.gz"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "5913c8b8dabf0967281c33ae1bb15d2390378aa7bae5840f40e6747756e41c9c"
    )


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


def test_environment_loads_the_carry_aware_ecb_profile() -> None:
    environment = load_environment(
        dataset="ecb_fx_carry",
        train_examples=1,
        eval_examples=1,
    )
    prompt = environment.eval_dataset[0]["question"]
    assert "public short-rate carry proxy" in prompt
    assert "CIP_3m_forward_points_pct" in prompt


def test_neutral_tasks_match_the_prime_adapter_datasets() -> None:
    benchmark = GateRunnerBenchmark(seed=23)
    train_tasks, eval_tasks = benchmark.build_tasks(
        train_examples=4,
        eval_examples=3,
    )
    environment = load_environment(
        seed=23,
        train_examples=4,
        eval_examples=3,
    )
    for task, row in zip(train_tasks, environment.dataset):
        assert row["question"] == task.question
        assert json.loads(row["info"]) == task.info
    for task, row in zip(eval_tasks, environment.eval_dataset):
        assert row["question"] == task.question
        assert json.loads(row["info"]) == task.info


def test_neutral_evaluator_matches_prime_rewards_and_metrics() -> None:
    benchmark = GateRunnerBenchmark(seed=17)
    _, eval_tasks = benchmark.build_tasks(train_examples=1, eval_examples=1)
    task = eval_tasks[0]
    completion_texts = [json.dumps(HONEST_CONFIG), "not json"]
    neutral = benchmark.evaluate_group(
        completions=completion_texts,
        as_of_index=task.as_of_index,
    )

    environment = load_environment(seed=17, train_examples=1, eval_examples=1)
    rubric = environment.rubric.rubrics[0]
    states = [{}, {}]
    prime_rewards = asyncio.run(
        rubric.honest_reward(
            completions=[
                [{"role": "assistant", "content": value}]
                for value in completion_texts
            ],
            infos=[task.info, task.info],
            states=states,
        )
    )
    assert prime_rewards == [record.reward for record in neutral]
    assert [state["gate_runner_score"] for state in states] == [
        record.score.metrics() for record in neutral
    ]
    assert [state["gate_runner_error"] for state in states] == [
        record.error for record in neutral
    ]


def test_core_modules_do_not_import_platform_adapters() -> None:
    core_path = Path(__file__).parents[1] / "gate_runner_core"
    forbidden = {"datasets", "verifiers", "gate_runner_adapters"}
    imported: set[str] = set()
    for path in core_path.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
    assert imported.isdisjoint(forbidden)


def test_jsonl_cli_round_trip_preserves_grouped_trial_accounting(
    tmp_path: Path,
) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    input_path = tmp_path / "completions.jsonl"
    output_path = tmp_path / "results.jsonl"
    assert cli_main(
        [
            "tasks",
            "--split",
            "eval",
            "--examples",
            "1",
            "--output",
            str(tasks_path),
        ]
    ) == 0
    task = json.loads(tasks_path.read_text(encoding="utf-8"))
    task["completions"] = [json.dumps(HONEST_CONFIG), "not json"]
    input_path.write_text(json.dumps(task) + "\n", encoding="utf-8")
    assert cli_main(
        [
            "score",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
    ) == 0
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["task_id"] == "eval-000000"
    assert len(result["results"]) == 2
    assert result["results"][0]["metrics"]["trial_count"] == 2.0
    assert result["results"][0]["metrics"]["validity"] == 1.0
    assert result["results"][1]["metrics"]["validity"] == 0.0
    assert result["results"][1]["reward"] == 0.0
    assert result["results"][1]["error"].startswith("invalid JSON")
