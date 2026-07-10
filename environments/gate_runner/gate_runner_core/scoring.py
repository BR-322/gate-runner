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
    raw_sharpe: float
    turnover: float


@dataclass(frozen=True)
class HonestScore:
    reward: float = 0.0
    validity: float = 0.0
    raw_sharpe: float = 0.0
    dsr: float = 0.0
    pbo: float = 0.0
    complexity: float = 0.0
    parameter_count: float = 0.0
    trial_count: float = 0.0
    passed: float = 0.0
    turnover: float = 0.0

    def metrics(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


class StrategyBacktester:
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

    def evaluate(self, strategy: StrategyConfig, as_of_index: int) -> BacktestResult:
        all_returns: list[np.ndarray] = []
        window_sharpes: list[float] = []
        total_turnover = 0.0
        for window_index in range(self.windows):
            start = as_of_index + window_index * self.window_days
            end = start + self.window_days
            window_returns, turnover = self._run_window(strategy, start, end)
            all_returns.append(window_returns)
            window_sharpes.append(self.annualized_sharpe(window_returns))
            total_turnover += turnover
        daily_returns = np.concatenate(all_returns)
        return BacktestResult(
            daily_returns=daily_returns,
            window_sharpes=np.asarray(window_sharpes, dtype=float),
            raw_sharpe=self.annualized_sharpe(daily_returns),
            turnover=total_turnover,
        )

    def _run_window(
        self, strategy: StrategyConfig, start: int, end: int
    ) -> tuple[np.ndarray, float]:
        symbol_count = len(self.market.symbols)
        active = np.zeros(symbol_count, dtype=bool)
        entry_price = np.zeros(symbol_count, dtype=float)
        peak_price = np.zeros(symbol_count, dtype=float)
        holding_days = np.zeros(symbol_count, dtype=int)
        previous_weights = np.zeros(symbol_count, dtype=float)
        daily_returns = np.zeros(end - start, dtype=float)
        total_turnover = 0.0

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
        return daily_returns, total_turnover

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
    @staticmethod
    def estimate(window_scores: np.ndarray) -> float:
        scores = np.asarray(window_scores, dtype=float)
        if scores.ndim != 2:
            raise ValueError("window_scores must be strategies x windows")
        strategy_count, window_count = scores.shape
        if strategy_count < 2 or window_count < 4 or window_count % 2:
            return 0.0

        losses: list[float] = []
        all_windows = set(range(window_count))
        for in_sample in combinations(range(window_count), window_count // 2):
            out_of_sample = tuple(sorted(all_windows.difference(in_sample)))
            in_performance = np.mean(scores[:, in_sample], axis=1)
            winner = int(np.argmax(in_performance))
            out_performance = np.mean(scores[:, out_of_sample], axis=1)
            winner_value = out_performance[winner]
            lower = float(np.sum(out_performance < winner_value))
            equal = float(np.sum(np.isclose(out_performance, winner_value))) - 1.0
            average_rank = 1.0 + lower + max(0.0, equal) / 2.0
            relative_rank = average_rank / (strategy_count + 1.0)
            losses.append(1.0 if relative_rank <= 0.5 else 0.0)
        return float(np.mean(losses)) if losses else 0.0


class HonestScorer:
    def __init__(
        self,
        backtester: StrategyBacktester,
        dsr_weight: float = 0.70,
        pbo_weight: float = 0.20,
        complexity_weight: float = 0.05,
        validity_weight: float = 0.10,
    ) -> None:
        self.backtester = backtester
        self.dsr_weight = dsr_weight
        self.pbo_weight = pbo_weight
        self.complexity_weight = complexity_weight
        self.validity_weight = validity_weight
        self._cache: dict[tuple[int, str], BacktestResult] = {}

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

        pbo = (
            ProbabilityBacktestOverfitting.estimate(
                np.vstack([result.window_sharpes for result in valid_results])
            )
            if valid_results
            else 0.0
        )
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
        for strategy, result in zip(strategies, backtests):
            if strategy is None or result is None:
                scores.append(HonestScore(trial_count=float(trial_count)))
                continue
            dsr = DeflatedSharpeRatio.probability(
                result.daily_returns,
                trials=trial_count,
                trial_sharpe_std=trial_sharpe_std,
            )
            complexity = strategy.normalized_complexity
            reward = (
                self.dsr_weight * np.tanh(dsr)
                - self.pbo_weight * pbo
                - self.complexity_weight * complexity
                + self.validity_weight
            )
            scores.append(
                HonestScore(
                    reward=float(np.clip(reward, -1.0, 1.0)),
                    validity=1.0,
                    raw_sharpe=result.raw_sharpe,
                    dsr=dsr,
                    pbo=pbo,
                    complexity=complexity,
                    parameter_count=float(strategy.parameter_count),
                    trial_count=float(trial_count),
                    passed=1.0 if dsr > 0.90 and pbo < 0.25 else 0.0,
                    turnover=result.turnover,
                )
            )
        return scores
