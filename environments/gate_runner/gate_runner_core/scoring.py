from dataclasses import asdict, dataclass
from itertools import combinations
from statistics import NormalDist

import numpy as np

from gate_runner_core.config import (
    ChannelBreakout,
    EqualWeightSizing,
    FractionalKellySizing,
    InverseVolatilitySizing,
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
    carry_contribution: float
    active_fraction: float
    exposure_weighted_active_fraction: float
    active_session_fraction: float
    average_gross_exposure: float
    median_gross_exposure: float
    mean_active_gross_exposure: float
    cash_fraction: float
    max_weight: float
    effective_position_count: float
    realized_volatility: float
    active_windows: int


@dataclass(frozen=True)
class _WindowResult:
    daily_returns: np.ndarray
    turnover: float
    carry_contribution: float
    gross_exposure: np.ndarray
    meaningful_active: np.ndarray
    effective_position_count: np.ndarray
    max_weight: np.ndarray


@dataclass(frozen=True)
class HonestScore:
    reward: float = 0.0
    validity: float = 0.0
    raw_sharpe: float = 0.0
    dsr: float = 0.0
    diagnostic_dsr: float = 0.0
    reward_minus_diagnostic_dsr: float = 0.0
    behavioral_effective_rank: float = 0.0
    behavioral_effective_rank_ratio: float = 0.0
    mean_pairwise_absolute_correlation: float = 0.0
    observed_trial_sharpe_std: float = 0.0
    reward_trial_sharpe_std: float = 0.0
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
    carry_contribution: float = 0.0
    active_fraction: float = 0.0
    exposure_weighted_active_fraction: float = 0.0
    active_session_fraction: float = 0.0
    average_gross_exposure: float = 0.0
    median_gross_exposure: float = 0.0
    mean_active_gross_exposure: float = 0.0
    cash_fraction: float = 0.0
    max_weight: float = 0.0
    effective_position_count: float = 0.0
    realized_volatility: float = 0.0
    active_windows: float = 0.0

    def metrics(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


class StrategyBacktester:
    WINDOW_TAIL_FRACTION = 0.25
    DAILY_EXPECTED_SHORTFALL_FRACTION = 0.05
    MEANINGFUL_WEIGHT = 0.01
    KELLY_MEAN_SHRINKAGE = 0.50
    VOLATILITY_FLOOR_ANNUALIZED = 0.05

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
        all_gross_exposure: list[np.ndarray] = []
        all_meaningful_active: list[np.ndarray] = []
        all_effective_position_counts: list[np.ndarray] = []
        all_max_weights: list[np.ndarray] = []
        window_sharpes: list[float] = []
        total_turnover = 0.0
        total_carry_contribution = 0.0
        active_windows = 0
        for window_index in range(self.windows):
            start = as_of_index + window_index * self.window_days
            end = start + self.window_days
            window = self._run_window(strategy, start, end)
            all_returns.append(window.daily_returns)
            all_gross_exposure.append(window.gross_exposure)
            all_meaningful_active.append(window.meaningful_active)
            all_effective_position_counts.append(window.effective_position_count)
            all_max_weights.append(window.max_weight)
            window_sharpes.append(self.annualized_sharpe(window.daily_returns))
            total_turnover += window.turnover
            total_carry_contribution += window.carry_contribution
            active_windows += int(np.any(window.meaningful_active))
        daily_returns = np.concatenate(all_returns)
        window_returns = np.vstack(all_returns)
        gross_exposure = np.concatenate(all_gross_exposure)
        meaningful_active = np.concatenate(all_meaningful_active)
        effective_position_counts = np.concatenate(all_effective_position_counts)
        max_weights = np.concatenate(all_max_weights)
        reference_window_risk = self._reference_window_risk(as_of_index)
        daily_expected_shortfall = self.expected_shortfall(daily_returns)
        reference_daily_risk = reference_window_risk / np.sqrt(self.window_days)
        average_gross_exposure = float(np.mean(gross_exposure))
        mean_active_gross_exposure = (
            float(np.mean(gross_exposure[meaningful_active]))
            if np.any(meaningful_active)
            else 0.0
        )
        effective_position_count = (
            float(np.mean(effective_position_counts[meaningful_active]))
            if np.any(meaningful_active)
            else 0.0
        )
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
            carry_contribution=total_carry_contribution,
            active_fraction=average_gross_exposure,
            exposure_weighted_active_fraction=average_gross_exposure,
            active_session_fraction=float(np.mean(meaningful_active)),
            average_gross_exposure=average_gross_exposure,
            median_gross_exposure=float(np.median(gross_exposure)),
            mean_active_gross_exposure=mean_active_gross_exposure,
            cash_fraction=float(np.mean(1.0 - np.clip(gross_exposure, 0.0, 1.0))),
            max_weight=float(np.max(max_weights, initial=0.0)),
            effective_position_count=effective_position_count,
            realized_volatility=(
                float(np.std(daily_returns, ddof=1) * np.sqrt(252.0))
                if len(daily_returns) > 1
                else 0.0
            ),
            active_windows=active_windows,
        )

    def _run_window(
        self, strategy: StrategyConfig, start: int, end: int
    ) -> _WindowResult:
        symbol_count = len(self.market.symbols)
        active = np.zeros(symbol_count, dtype=bool)
        entry_price = np.zeros(symbol_count, dtype=float)
        peak_price = np.zeros(symbol_count, dtype=float)
        holding_days = np.zeros(symbol_count, dtype=int)
        previous_weights = np.zeros(symbol_count, dtype=float)
        daily_returns = np.zeros(end - start, dtype=float)
        gross_exposure = np.zeros(end - start, dtype=float)
        meaningful_active = np.zeros(end - start, dtype=bool)
        effective_position_count = np.zeros(end - start, dtype=float)
        max_weight = np.zeros(end - start, dtype=float)
        total_turnover = 0.0
        total_carry_contribution = 0.0

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

            k = strategy.universe_filter.k
            if strategy.universe_filter.rank_by == "relative_strength_252d":
                ranking_signal = (
                    prior_close / self.market.close[prior_index - 252] - 1.0
                )
                order = np.argsort(ranking_signal, kind="stable")
                eligible = (
                    order[-k:][::-1]
                    if strategy.universe_filter.side == "top"
                    else order[:k]
                )
            elif strategy.universe_filter.rank_by == "long_eur_carry":
                if self.market.has_carry:
                    ranking_signal = (
                        self.market.base_reference_rates_percent[prior_index]
                        - self.market.foreign_reference_rates_percent[prior_index]
                    )
                    order = np.argsort(ranking_signal, kind="stable")
                    eligible = (
                        order[-k:][::-1]
                        if strategy.universe_filter.side == "top"
                        else order[:k]
                    )
                else:
                    eligible = np.asarray([], dtype=int)
            else:
                raise TypeError(
                    f"unsupported universe rank: {strategy.universe_filter.rank_by}"
                )
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

            active_count = int(np.count_nonzero(active))
            weights = self._position_weights(strategy, active, prior_index)
            gross_exposure[offset] = float(np.sum(np.abs(weights)))
            meaningful_active[offset] = bool(
                np.any(np.abs(weights) >= self.MEANINGFUL_WEIGHT)
            )
            if gross_exposure[offset] > 0.0:
                effective_position_count[offset] = (
                    gross_exposure[offset] ** 2 / float(np.sum(weights**2))
                )
                max_weight[offset] = float(np.max(np.abs(weights)))

            traded_weight = np.abs(weights - previous_weights)
            per_side_cost = (
                self.cost_bps_per_side + self.market.spread_bps[prior_index]
            ) / 10_000.0
            transaction_cost = float(np.dot(traded_weight, per_side_cost))
            total_turnover += float(np.sum(traded_weight))
            spot_returns = self.market.close[day_index] / prior_close - 1.0
            carry_returns = self.market.carry_returns[day_index]
            asset_returns = (1.0 + spot_returns) * (1.0 + carry_returns) - 1.0
            carry_component = (1.0 + spot_returns) * carry_returns
            total_carry_contribution += float(np.dot(weights, carry_component))
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
        return _WindowResult(
            daily_returns=daily_returns,
            turnover=total_turnover,
            carry_contribution=total_carry_contribution,
            gross_exposure=gross_exposure,
            meaningful_active=meaningful_active,
            effective_position_count=effective_position_count,
            max_weight=max_weight,
        )

    def _position_weights(
        self,
        strategy: StrategyConfig,
        active: np.ndarray,
        prior_index: int,
    ) -> np.ndarray:
        weights = np.zeros(len(self.market.symbols), dtype=float)
        active_indices = np.flatnonzero(active)
        if not len(active_indices):
            return weights

        sizing = strategy.sizing
        if isinstance(sizing, EqualWeightSizing):
            weights[active_indices] = 1.0 / len(active_indices)
            return weights

        history = self._trailing_total_returns(
            prior_index=prior_index,
            lookback_days=sizing.lookback_days,
        )[:, active_indices]
        volatility_floor = self.VOLATILITY_FLOOR_ANNUALIZED / np.sqrt(252.0)

        if isinstance(sizing, InverseVolatilitySizing):
            volatility = np.maximum(
                np.std(history, axis=0, ddof=1),
                volatility_floor,
            )
            weights[active_indices] = self._normalize_with_cap(
                1.0 / volatility,
                max_weight=sizing.max_weight,
            )
            return weights

        if isinstance(sizing, FractionalKellySizing):
            expected_return = (
                self.KELLY_MEAN_SHRINKAGE * np.mean(history, axis=0)
            )
            variance = np.maximum(
                np.var(history, axis=0, ddof=1),
                volatility_floor**2,
            )
            allocations = sizing.fraction * np.maximum(expected_return, 0.0) / variance
            allocations = np.clip(allocations, 0.0, sizing.max_weight)
            gross = float(np.sum(allocations))
            if gross > 1.0:
                allocations /= gross
            weights[active_indices] = allocations
            return weights

        raise TypeError(f"unsupported sizing config: {type(sizing)}")

    def _trailing_total_returns(
        self,
        prior_index: int,
        lookback_days: int,
    ) -> np.ndarray:
        start = prior_index - lookback_days + 1
        spot_returns = self.market.returns[start : prior_index + 1]
        carry_returns = self.market.carry_returns[start : prior_index + 1]
        return (1.0 + spot_returns) * (1.0 + carry_returns) - 1.0

    @staticmethod
    def _normalize_with_cap(
        raw_weights: np.ndarray,
        max_weight: float,
    ) -> np.ndarray:
        raw = np.asarray(raw_weights, dtype=float)
        result = np.zeros_like(raw)
        remaining = np.ones(len(raw), dtype=bool)
        remaining_gross = 1.0
        while np.any(remaining) and remaining_gross > 1e-12:
            remaining_raw = raw[remaining]
            raw_total = float(np.sum(remaining_raw))
            if raw_total <= 0.0:
                break
            proposed = remaining_gross * remaining_raw / raw_total
            capped = proposed > max_weight
            remaining_indices = np.flatnonzero(remaining)
            if not np.any(capped):
                result[remaining_indices] = proposed
                break
            capped_indices = remaining_indices[capped]
            result[capped_indices] = max_weight
            remaining[capped_indices] = False
            remaining_gross = max(0.0, 1.0 - float(np.sum(result)))
        return result

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


class BehavioralDiversity:
    @staticmethod
    def summarize(return_streams: np.ndarray) -> tuple[float, float]:
        """Return effective rank and mean absolute pairwise correlation."""
        streams = np.asarray(return_streams, dtype=float)
        if streams.ndim != 2:
            raise ValueError("return_streams must be strategies x sessions")
        if not streams.shape[0]:
            return 0.0, 0.0
        centered = streams - np.mean(streams, axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1)
        informative = norms > 1e-12
        if not np.any(informative):
            return 0.0, 0.0
        normalized = centered[informative] / norms[informative, None]
        singular_values = np.linalg.svd(normalized, compute_uv=False)
        energy = singular_values**2
        probabilities = energy / float(np.sum(energy))
        positive = probabilities > 0.0
        effective_rank = float(
            np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
        )
        if len(normalized) < 2:
            mean_absolute_correlation = 0.0
        else:
            correlations = normalized @ normalized.T
            upper = correlations[np.triu_indices(len(normalized), k=1)]
            mean_absolute_correlation = float(np.mean(np.abs(upper)))
        return effective_rank, mean_absolute_correlation


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
    MIN_MEAN_ACTIVE_GROSS_EXPOSURE = 0.25

    VALID_FLOOR = 0.10
    FAIL_PROXIMITY_SPAN = 0.35
    PASS_FLOOR = 0.60
    PASS_ROBUSTNESS_SPAN = 0.30
    COMPLEXITY_WEIGHT = 0.01
    MIN_COMPLEXITY = (
        StrategyConfig.MIN_PARAMETER_COUNT / StrategyConfig.MAX_PARAMETER_COUNT
    )
    MAX_COMPLEXITY = 1.0

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
        mean_active_gross_exposure: float = 1.0,
    ) -> bool:
        return (
            dsr > cls.DSR_PASS_THRESHOLD
            and window_tail_score > cls.WINDOW_TAIL_PASS_THRESHOLD
            and expected_shortfall_ratio < cls.EXPECTED_SHORTFALL_PASS_THRESHOLD
            and active_fraction >= cls.MIN_ACTIVE_FRACTION
            and active_windows >= cls.MIN_ACTIVE_WINDOWS
            and mean_active_gross_exposure
            >= cls.MIN_MEAN_ACTIVE_GROSS_EXPOSURE
        )

    def _shaped_reward(
        self,
        dsr: float,
        window_tail_score: float,
        expected_shortfall_ratio: float,
        complexity: float,
        active_fraction: float,
        active_windows: float,
        mean_active_gross_exposure: float = 1.0,
    ) -> float:
        passed = self._passes_gate(
            dsr,
            window_tail_score,
            expected_shortfall_ratio,
            active_fraction,
            active_windows,
            mean_active_gross_exposure,
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
                np.clip(
                    (
                        self.MIN_MEAN_ACTIVE_GROSS_EXPOSURE
                        - mean_active_gross_exposure
                    )
                    / self.MIN_MEAN_ACTIVE_GROSS_EXPOSURE,
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
        observed_trial_sharpe_std = (
            float(np.std(trial_daily_sharpes, ddof=1))
            if len(trial_daily_sharpes) > 1
            else 0.0
        )
        observation_count = (
            len(valid_results[0].daily_returns) if valid_results else 0
        )
        reward_dispersion_floor = (
            1.0 / np.sqrt(observation_count - 1.0)
            if observation_count > 1
            else 0.0
        )
        reward_trial_sharpe_std = max(
            observed_trial_sharpe_std,
            reward_dispersion_floor,
        )
        if valid_results:
            behavioral_effective_rank, mean_absolute_correlation = (
                BehavioralDiversity.summarize(
                    np.vstack([result.daily_returns for result in valid_results])
                )
            )
            effective_trial_count = max(
                1,
                int(np.ceil(behavioral_effective_rank)),
            )
            effective_rank_ratio = behavioral_effective_rank / len(valid_results)
        else:
            behavioral_effective_rank = 0.0
            mean_absolute_correlation = 0.0
            effective_trial_count = 1
            effective_rank_ratio = 0.0
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
                trial_sharpe_std=reward_trial_sharpe_std,
            )
            diagnostic_dsr = DeflatedSharpeRatio.probability(
                result.daily_returns,
                trials=effective_trial_count,
                trial_sharpe_std=observed_trial_sharpe_std,
            )
            complexity = strategy.normalized_complexity
            passed = self._passes_gate(
                dsr,
                result.window_tail_score,
                result.expected_shortfall_ratio,
                result.active_fraction,
                result.active_windows,
                result.mean_active_gross_exposure,
            )
            reward = self._shaped_reward(
                dsr=dsr,
                window_tail_score=result.window_tail_score,
                expected_shortfall_ratio=result.expected_shortfall_ratio,
                complexity=complexity,
                active_fraction=result.active_fraction,
                active_windows=result.active_windows,
                mean_active_gross_exposure=result.mean_active_gross_exposure,
            )
            scores.append(
                HonestScore(
                    reward=reward,
                    validity=1.0,
                    raw_sharpe=result.raw_sharpe,
                    dsr=dsr,
                    diagnostic_dsr=diagnostic_dsr,
                    reward_minus_diagnostic_dsr=dsr - diagnostic_dsr,
                    behavioral_effective_rank=behavioral_effective_rank,
                    behavioral_effective_rank_ratio=effective_rank_ratio,
                    mean_pairwise_absolute_correlation=mean_absolute_correlation,
                    observed_trial_sharpe_std=observed_trial_sharpe_std,
                    reward_trial_sharpe_std=reward_trial_sharpe_std,
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
                    carry_contribution=result.carry_contribution,
                    active_fraction=result.active_fraction,
                    exposure_weighted_active_fraction=(
                        result.exposure_weighted_active_fraction
                    ),
                    active_session_fraction=result.active_session_fraction,
                    average_gross_exposure=result.average_gross_exposure,
                    median_gross_exposure=result.median_gross_exposure,
                    mean_active_gross_exposure=(
                        result.mean_active_gross_exposure
                    ),
                    cash_fraction=result.cash_fraction,
                    max_weight=result.max_weight,
                    effective_position_count=result.effective_position_count,
                    realized_volatility=result.realized_volatility,
                    active_windows=float(result.active_windows),
                )
            )
        return scores
