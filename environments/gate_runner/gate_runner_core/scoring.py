from dataclasses import asdict, dataclass
from itertools import combinations
from statistics import NormalDist

import numpy as np

from gate_runner_core.config import (
    ChannelBreakout,
    MeanReversionZScore,
    MomentumThreshold,
    StopLossPct,
    StrategyConfig,
    TimeExit,
    TrailingStop,
)
from gate_runner_core.market import MarketData


@dataclass(frozen=True)
class BacktestResult:
    daily_returns: np.ndarray
    window_sharpes: np.ndarray
    window_tail_score: float
    reference_window_risk: float
    daily_expected_shortfall: float
    expected_shortfall_ratio: float
    raw_sharpe: float
    turnover: float
    active_fraction: float
    active_windows: int


@dataclass(frozen=True)
class HonestScore:
    reward: float = 0.0
    validity: float = 0.0
    raw_sharpe: float = 0.0
    dsr: float = 0.0
    pbo: float = 0.0
    pbo_contribution: float = 0.0
    window_tail_score: float = 0.0
    reference_window_risk: float = 0.0
    daily_expected_shortfall: float = 0.0
    expected_shortfall_ratio: float = 0.0
    complexity: float = 0.0
    parameter_count: float = 0.0
    trial_count: float = 0.0
    passed: float = 0.0
    turnover: float = 0.0
    active_fraction: float = 0.0
    active_windows: float = 0.0

    def metrics(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


class StrategyBacktester:
    WINDOW_TAIL_FRACTION = 0.25
    DAILY_EXPECTED_SHORTFALL_FRACTION = 0.05

    def __init__(
        self,
        market: MarketData,
        windows: int = 8,
        window_days: int = 42,
        cost_bps_per_side: float = 10.0,
    ) -> None:
        self.market = market
        self.windows = windows
        self.window_days = window_days
        self.cost_bps_per_side = cost_bps_per_side

    @staticmethod
    def annualized_sharpe(returns: np.ndarray) -> float:
        if len(returns) < 2:
            return 0.0
        standard_deviation = float(np.std(returns, ddof=1))
        if standard_deviation <= 1e-12:
            return 0.0
        return float(np.mean(returns) / standard_deviation * np.sqrt(252.0))

    @staticmethod
    def lower_tail_window_score(
        window_returns: np.ndarray,
        reference_window_risk: float,
        tail_fraction: float = WINDOW_TAIL_FRACTION,
    ) -> float:
        """Average the weakest cost-adjusted window returns in reference-risk units."""
        values = np.asarray(window_returns, dtype=float)
        if values.ndim != 2 or values.shape[0] < 1:
            raise ValueError("window_returns must be windows x sessions")
        if not np.all(np.isfinite(values)) or np.any(values <= -1.0):
            return -np.inf
        if not np.isfinite(reference_window_risk) or reference_window_risk <= 0.0:
            raise ValueError("reference_window_risk must be finite and positive")
        if not 0.0 < tail_fraction <= 1.0:
            raise ValueError("tail_fraction must be in (0, 1]")

        window_growth = np.sum(np.log1p(values), axis=1)
        scaled_growth = window_growth / reference_window_risk
        tail_count = min(
            len(scaled_growth),
            max(2, int(np.ceil(tail_fraction * len(scaled_growth)))),
        )
        return float(np.mean(np.partition(scaled_growth, tail_count - 1)[:tail_count]))

    @staticmethod
    def expected_shortfall(
        returns: np.ndarray,
        tail_fraction: float = DAILY_EXPECTED_SHORTFALL_FRACTION,
    ) -> float:
        """Return the positive mean loss over the weakest daily-return tail."""
        values = np.asarray(returns, dtype=float)
        if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
            return 0.0
        if not 0.0 < tail_fraction <= 1.0:
            raise ValueError("tail_fraction must be in (0, 1]")
        tail_count = max(1, int(np.ceil(tail_fraction * len(values))))
        tail_mean = float(np.mean(np.partition(values, tail_count - 1)[:tail_count]))
        return max(0.0, -tail_mean)

    def _reference_window_risk(self, as_of_index: int) -> float:
        """Use exogenous asset volatility so strategy inactivity cannot shrink risk."""
        end = as_of_index + self.windows * self.window_days
        asset_returns = self.market.returns[as_of_index:end]
        asset_volatility = np.std(asset_returns, axis=0, ddof=1)
        finite_positive = asset_volatility[
            np.isfinite(asset_volatility) & (asset_volatility > 1e-12)
        ]
        if not len(finite_positive):
            return 1e-6
        return max(
            1e-6,
            float(np.median(finite_positive) * np.sqrt(self.window_days)),
        )

    def evaluate(self, strategy: StrategyConfig, as_of_index: int) -> BacktestResult:
        all_returns: list[np.ndarray] = []
        window_sharpes: list[float] = []
        total_turnover = 0.0
        total_active_days = 0
        active_windows = 0
        for window_index in range(self.windows):
            start = as_of_index + window_index * self.window_days
            end = start + self.window_days
            window_returns, turnover, active_days = self._run_window(
                strategy, start, end
            )
            all_returns.append(window_returns)
            window_sharpes.append(self.annualized_sharpe(window_returns))
            total_turnover += turnover
            total_active_days += active_days
            active_windows += int(active_days > 0)
        daily_returns = np.concatenate(all_returns)
        window_returns = np.vstack(all_returns)
        reference_window_risk = self._reference_window_risk(as_of_index)
        daily_expected_shortfall = self.expected_shortfall(daily_returns)
        reference_daily_risk = reference_window_risk / np.sqrt(self.window_days)
        return BacktestResult(
            daily_returns=daily_returns,
            window_sharpes=np.asarray(window_sharpes, dtype=float),
            window_tail_score=self.lower_tail_window_score(
                window_returns,
                reference_window_risk,
            ),
            reference_window_risk=reference_window_risk,
            daily_expected_shortfall=daily_expected_shortfall,
            expected_shortfall_ratio=(
                daily_expected_shortfall / reference_daily_risk
            ),
            raw_sharpe=self.annualized_sharpe(daily_returns),
            turnover=total_turnover,
            active_fraction=total_active_days / (self.windows * self.window_days),
            active_windows=active_windows,
        )

    def _run_window(
        self, strategy: StrategyConfig, start: int, end: int
    ) -> tuple[np.ndarray, float, int]:
        symbol_count = len(self.market.symbols)
        active = np.zeros(symbol_count, dtype=bool)
        entry_price = np.zeros(symbol_count, dtype=float)
        peak_price = np.zeros(symbol_count, dtype=float)
        holding_days = np.zeros(symbol_count, dtype=int)
        previous_weights = np.zeros(symbol_count, dtype=float)
        daily_returns = np.zeros(end - start, dtype=float)
        total_turnover = 0.0
        active_days = 0

        for offset, day_index in enumerate(range(start, end)):
            prior_index = day_index - 1
            prior_close = self.market.close[prior_index]
            active = self._apply_exits(
                strategy=strategy,
                active=active,
                prior_close=prior_close,
                entry_price=entry_price,
                peak_price=peak_price,
                holding_days=holding_days,
            )

            relative_strength = prior_close / self.market.close[prior_index - 252] - 1.0
            order = np.argsort(relative_strength, kind="stable")
            k = strategy.universe_filter.k
            eligible = order[-k:][::-1] if strategy.universe_filter.side == "top" else order[:k]
            entry_signal = self._entry_signal(strategy, prior_index)
            slots = strategy.sizing.max_positions - int(np.count_nonzero(active))
            if slots > 0:
                candidates = [
                    int(index)
                    for index in eligible
                    if entry_signal[index] and not active[index]
                ][:slots]
                if candidates:
                    active[candidates] = True
                    entry_price[candidates] = prior_close[candidates]
                    peak_price[candidates] = prior_close[candidates]
                    holding_days[candidates] = 0

            weights = np.zeros(symbol_count, dtype=float)
            active_count = int(np.count_nonzero(active))
            if active_count:
                weights[active] = 1.0 / active_count
                active_days += 1

            traded_weight = np.abs(weights - previous_weights)
            per_side_cost = (
                self.cost_bps_per_side + self.market.spread_bps[prior_index]
            ) / 10_000.0
            transaction_cost = float(np.dot(traded_weight, per_side_cost))
            total_turnover += float(np.sum(traded_weight))
            asset_returns = self.market.close[day_index] / prior_close - 1.0
            daily_returns[offset] = max(
                -0.99, float(np.dot(weights, asset_returns)) - transaction_cost
            )

            if active_count:
                peak_price[active] = np.maximum(
                    peak_price[active], self.market.close[day_index, active]
                )
                holding_days[active] += 1
            previous_weights = weights

        if np.any(previous_weights):
            liquidation_cost = float(
                np.dot(
                    previous_weights,
                    (
                        self.cost_bps_per_side + self.market.spread_bps[end - 1]
                    )
                    / 10_000.0,
                )
            )
            daily_returns[-1] = max(-0.99, daily_returns[-1] - liquidation_cost)
            total_turnover += float(np.sum(previous_weights))
        return daily_returns, total_turnover, active_days

    @staticmethod
    def _apply_exits(
        strategy: StrategyConfig,
        active: np.ndarray,
        prior_close: np.ndarray,
        entry_price: np.ndarray,
        peak_price: np.ndarray,
        holding_days: np.ndarray,
    ) -> np.ndarray:
        active = active.copy()
        if isinstance(strategy.exit, StopLossPct):
            exit_mask = active & (
                prior_close / np.maximum(entry_price, 1e-12) - 1.0
                <= -strategy.exit.stop_pct
            )
        elif isinstance(strategy.exit, TrailingStop):
            exit_mask = active & (
                prior_close / np.maximum(peak_price, 1e-12) - 1.0
                <= -strategy.exit.trail_pct
            )
        elif isinstance(strategy.exit, TimeExit):
            exit_mask = active & (holding_days >= strategy.exit.max_holding_days)
        else:
            raise TypeError(f"unsupported exit config: {type(strategy.exit)}")
        active[exit_mask] = False
        entry_price[exit_mask] = 0.0
        peak_price[exit_mask] = 0.0
        holding_days[exit_mask] = 0
        return active

    def _entry_signal(self, strategy: StrategyConfig, prior_index: int) -> np.ndarray:
        entry = strategy.entry
        prior_close = self.market.close[prior_index]
        if isinstance(entry, MomentumThreshold):
            trailing_return = (
                prior_close / self.market.close[prior_index - entry.lookback_days] - 1.0
            )
            return trailing_return >= entry.threshold
        if isinstance(entry, MeanReversionZScore):
            history = self.market.close[
                prior_index - entry.lookback_days + 1 : prior_index + 1
            ]
            mean = np.mean(history, axis=0)
            standard_deviation = np.std(history, axis=0, ddof=1)
            z_score = (prior_close - mean) / np.maximum(standard_deviation, 1e-12)
            return z_score <= -entry.entry_z
        if isinstance(entry, ChannelBreakout):
            confirmed = np.ones(len(self.market.symbols), dtype=bool)
            for lag in range(entry.confirmation_days):
                signal_index = prior_index - lag
                channel = np.max(
                    self.market.close[
                        signal_index - entry.lookback_days : signal_index
                    ],
                    axis=0,
                )
                confirmed &= (
                    self.market.close[signal_index]
                    >= channel * (1.0 + entry.buffer_pct)
                )
            return confirmed
        raise TypeError(f"unsupported entry config: {type(entry)}")


class DeflatedSharpeRatio:
    EULER_GAMMA = 0.5772156649015329

    @classmethod
    def probability(
        cls,
        returns: np.ndarray,
        trials: int,
        trial_sharpe_std: float | None = None,
    ) -> float:
        values = np.asarray(returns, dtype=float)
        observations = len(values)
        if observations < 3 or not np.all(np.isfinite(values)):
            return 0.0
        standard_deviation = float(np.std(values, ddof=1))
        if standard_deviation <= 1e-12:
            return 0.0
        centered = values - float(np.mean(values))
        daily_sharpe = float(np.mean(values) / standard_deviation)
        skew = float(np.mean(centered**3) / standard_deviation**3)
        kurtosis = float(np.mean(centered**4) / standard_deviation**4)

        expected_max = 0.0
        if trials > 1:
            null_standard_error = 1.0 / np.sqrt(observations - 1.0)
            sharpe_standard_deviation = (
                null_standard_error
                if trial_sharpe_std is None or not np.isfinite(trial_sharpe_std)
                else max(0.0, float(trial_sharpe_std))
            )
            first_quantile = NormalDist().inv_cdf(1.0 - 1.0 / trials)
            second_quantile = NormalDist().inv_cdf(
                1.0 - 1.0 / (trials * np.e)
            )
            expected_max = sharpe_standard_deviation * (
                (1.0 - cls.EULER_GAMMA) * first_quantile
                + cls.EULER_GAMMA * second_quantile
            )

        denominator = np.sqrt(
            max(
                1e-12,
                1.0
                - skew * daily_sharpe
                + ((kurtosis - 1.0) / 4.0) * daily_sharpe**2,
            )
        )
        test_statistic = (
            (daily_sharpe - expected_max)
            * np.sqrt(observations - 1.0)
            / denominator
        )
        return float(np.clip(NormalDist().cdf(test_statistic), 0.0, 1.0))


class ProbabilityBacktestOverfitting:
    @classmethod
    def estimate(cls, window_scores: np.ndarray) -> float:
        probability, _ = cls.decompose(window_scores)
        return probability

    @staticmethod
    def decompose(window_scores: np.ndarray) -> tuple[float, np.ndarray]:
        scores = np.asarray(window_scores, dtype=float)
        if scores.ndim != 2:
            raise ValueError("window_scores must be strategies x windows")
        strategy_count, window_count = scores.shape
        if strategy_count < 2 or window_count < 4 or window_count % 2:
            return 0.0, np.zeros(strategy_count, dtype=float)

        losses: list[float] = []
        contributions = np.zeros(strategy_count, dtype=float)
        all_windows = set(range(window_count))
        for in_sample in combinations(range(window_count), window_count // 2):
            out_of_sample = tuple(sorted(all_windows.difference(in_sample)))
            in_performance = np.mean(scores[:, in_sample], axis=1)
            out_performance = np.mean(scores[:, out_of_sample], axis=1)
            winners = np.flatnonzero(
                np.isclose(in_performance, np.max(in_performance))
            )
            split_losses: list[float] = []
            for winner in winners:
                winner_value = out_performance[winner]
                lower = float(np.sum(out_performance < winner_value))
                equal = (
                    float(np.sum(np.isclose(out_performance, winner_value))) - 1.0
                )
                average_rank = 1.0 + lower + max(0.0, equal) / 2.0
                relative_rank = average_rank / (strategy_count + 1.0)
                split_losses.append(float(relative_rank < 0.5))
            split_loss = float(np.mean(split_losses))
            losses.append(split_loss)
            contributions[winners] += np.asarray(split_losses) / len(winners)
        if not losses:
            return 0.0, contributions
        return float(np.mean(losses)), contributions / len(losses)


class HonestScorer:
    DSR_PASS_THRESHOLD = 0.90
    WINDOW_TAIL_PASS_THRESHOLD = -0.50
    WINDOW_TAIL_ROBUST_TARGET = 0.25
    WINDOW_TAIL_FAILURE_FLOOR = -2.50
    EXPECTED_SHORTFALL_PASS_THRESHOLD = 5.0
    EXPECTED_SHORTFALL_ROBUST_TARGET = 3.0
    EXPECTED_SHORTFALL_FAILURE_CEILING = 12.0
    MIN_ACTIVE_FRACTION = 0.10
    MIN_ACTIVE_WINDOWS = 4

    VALID_FLOOR = 0.10
    FAIL_PROXIMITY_SPAN = 0.35
    PASS_FLOOR = 0.60
    PASS_ROBUSTNESS_SPAN = 0.30
    COMPLEXITY_WEIGHT = 0.01
    MIN_COMPLEXITY = 5.0 / 8.0
    MAX_COMPLEXITY = 6.0 / 8.0

    def __init__(self, backtester: StrategyBacktester) -> None:
        self.backtester = backtester
        self._cache: dict[tuple[int, str], BacktestResult] = {}

    @classmethod
    def _passes_gate(
        cls,
        dsr: float,
        window_tail_score: float,
        expected_shortfall_ratio: float,
        active_fraction: float,
        active_windows: float,
    ) -> bool:
        return (
            dsr > cls.DSR_PASS_THRESHOLD
            and window_tail_score > cls.WINDOW_TAIL_PASS_THRESHOLD
            and expected_shortfall_ratio < cls.EXPECTED_SHORTFALL_PASS_THRESHOLD
            and active_fraction >= cls.MIN_ACTIVE_FRACTION
            and active_windows >= cls.MIN_ACTIVE_WINDOWS
        )

    def _shaped_reward(
        self,
        dsr: float,
        window_tail_score: float,
        expected_shortfall_ratio: float,
        complexity: float,
        active_fraction: float,
        active_windows: float,
    ) -> float:
        passed = self._passes_gate(
            dsr,
            window_tail_score,
            expected_shortfall_ratio,
            active_fraction,
            active_windows,
        )

        if passed:
            dsr_margin = np.clip(
                (dsr - self.DSR_PASS_THRESHOLD)
                / (1.0 - self.DSR_PASS_THRESHOLD),
                0.0,
                1.0,
            )
            window_tail_margin = np.clip(
                (window_tail_score - self.WINDOW_TAIL_PASS_THRESHOLD)
                / (
                    self.WINDOW_TAIL_ROBUST_TARGET
                    - self.WINDOW_TAIL_PASS_THRESHOLD
                ),
                0.0,
                1.0,
            )
            expected_shortfall_margin = np.clip(
                (
                    self.EXPECTED_SHORTFALL_PASS_THRESHOLD
                    - expected_shortfall_ratio
                )
                / (
                    self.EXPECTED_SHORTFALL_PASS_THRESHOLD
                    - self.EXPECTED_SHORTFALL_ROBUST_TARGET
                ),
                0.0,
                1.0,
            )
            robustness = min(
                float(dsr_margin),
                float(window_tail_margin),
                float(expected_shortfall_margin),
            )
            reward = self.PASS_FLOOR + self.PASS_ROBUSTNESS_SPAN * robustness
        else:
            violations = (
                np.clip(
                    (self.DSR_PASS_THRESHOLD - dsr) / self.DSR_PASS_THRESHOLD,
                    0.0,
                    1.0,
                ),
                np.clip(
                    (self.WINDOW_TAIL_PASS_THRESHOLD - window_tail_score)
                    / (
                        self.WINDOW_TAIL_PASS_THRESHOLD
                        - self.WINDOW_TAIL_FAILURE_FLOOR
                    ),
                    0.0,
                    1.0,
                ),
                np.clip(
                    (
                        expected_shortfall_ratio
                        - self.EXPECTED_SHORTFALL_PASS_THRESHOLD
                    )
                    / (
                        self.EXPECTED_SHORTFALL_FAILURE_CEILING
                        - self.EXPECTED_SHORTFALL_PASS_THRESHOLD
                    ),
                    0.0,
                    1.0,
                ),
                np.clip(
                    (self.MIN_ACTIVE_FRACTION - active_fraction)
                    / self.MIN_ACTIVE_FRACTION,
                    0.0,
                    1.0,
                ),
                np.clip(
                    (self.MIN_ACTIVE_WINDOWS - active_windows)
                    / self.MIN_ACTIVE_WINDOWS,
                    0.0,
                    1.0,
                ),
            )
            proximity = 1.0 - max(float(value) for value in violations)
            reward = self.VALID_FLOOR + self.FAIL_PROXIMITY_SPAN * proximity

        complexity_excess = np.clip(
            (complexity - self.MIN_COMPLEXITY)
            / (self.MAX_COMPLEXITY - self.MIN_COMPLEXITY),
            0.0,
            1.0,
        )
        reward -= self.COMPLEXITY_WEIGHT * float(complexity_excess)
        return float(np.clip(reward, -1.0, 1.0))

    def score_group(
        self,
        strategies: list[StrategyConfig | None],
        as_of_index: int,
    ) -> list[HonestScore]:
        trial_count = len(strategies)
        backtests: list[BacktestResult | None] = []
        valid_results: list[BacktestResult] = []
        for strategy in strategies:
            if strategy is None:
                backtests.append(None)
                continue
            cache_key = (as_of_index, strategy.canonical_json())
            result = self._cache.get(cache_key)
            if result is None:
                result = self.backtester.evaluate(strategy, as_of_index)
                self._cache[cache_key] = result
            backtests.append(result)
            valid_results.append(result)

        if valid_results:
            pbo, pbo_contributions = ProbabilityBacktestOverfitting.decompose(
                np.vstack([result.window_sharpes for result in valid_results])
            )
        else:
            pbo = 0.0
            pbo_contributions = np.asarray([], dtype=float)
        trial_daily_sharpes = np.asarray(
            [result.raw_sharpe / np.sqrt(252.0) for result in valid_results],
            dtype=float,
        )
        trial_sharpe_std = (
            float(np.std(trial_daily_sharpes, ddof=1))
            if len(trial_daily_sharpes) > 1
            else None
        )
        scores: list[HonestScore] = []
        valid_index = 0
        for strategy, result in zip(strategies, backtests):
            if strategy is None or result is None:
                scores.append(HonestScore(trial_count=float(trial_count)))
                continue
            pbo_contribution = float(pbo_contributions[valid_index])
            valid_index += 1
            dsr = DeflatedSharpeRatio.probability(
                result.daily_returns,
                trials=trial_count,
                trial_sharpe_std=trial_sharpe_std,
            )
            complexity = strategy.normalized_complexity
            passed = self._passes_gate(
                dsr,
                result.window_tail_score,
                result.expected_shortfall_ratio,
                result.active_fraction,
                result.active_windows,
            )
            reward = self._shaped_reward(
                dsr=dsr,
                window_tail_score=result.window_tail_score,
                expected_shortfall_ratio=result.expected_shortfall_ratio,
                complexity=complexity,
                active_fraction=result.active_fraction,
                active_windows=result.active_windows,
            )
            scores.append(
                HonestScore(
                    reward=reward,
                    validity=1.0,
                    raw_sharpe=result.raw_sharpe,
                    dsr=dsr,
                    pbo=pbo,
                    pbo_contribution=pbo_contribution,
                    window_tail_score=result.window_tail_score,
                    reference_window_risk=result.reference_window_risk,
                    daily_expected_shortfall=result.daily_expected_shortfall,
                    expected_shortfall_ratio=result.expected_shortfall_ratio,
                    complexity=complexity,
                    parameter_count=float(strategy.parameter_count),
                    trial_count=float(trial_count),
                    passed=float(passed),
                    turnover=result.turnover,
                    active_fraction=result.active_fraction,
                    active_windows=float(result.active_windows),
                )
            )
        return scores
