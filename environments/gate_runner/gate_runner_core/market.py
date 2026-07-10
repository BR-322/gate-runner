import csv
import gzip
import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from datasets import Dataset

from gate_runner_core.config import StrategyParser


ECB_FX_CURRENCIES = (
    "AUD",
    "BGN",
    "BRL",
    "CAD",
    "CHF",
    "CNY",
    "CZK",
    "DKK",
    "GBP",
    "HKD",
    "HUF",
    "IDR",
    "ILS",
    "INR",
    "JPY",
    "KRW",
    "MXN",
    "MYR",
    "NOK",
    "NZD",
    "PHP",
    "PLN",
    "RON",
    "SEK",
    "SGD",
    "THB",
    "TRY",
    "USD",
    "ZAR",
)
ECB_FX_SNAPSHOT_SHA256 = (
    "2e39da0d83bfbdb6d0575b1e481f2094ea974018b33473785d9349e2d04d7f3f"
)


@dataclass(frozen=True)
class FeatureSnapshot:
    as_of_date: str
    regime: str
    rows: tuple[dict[str, float | str], ...]
    summary: dict[str, float]


class MarketData:
    def __init__(
        self,
        dates: tuple[str, ...],
        symbols: tuple[str, ...],
        close: np.ndarray,
        spread_bps: np.ndarray,
        source_label: str,
    ) -> None:
        close_array = np.asarray(close, dtype=float)
        spread_array = np.asarray(spread_bps, dtype=float)
        expected_shape = (len(dates), len(symbols))
        if close_array.shape != expected_shape or spread_array.shape != expected_shape:
            raise ValueError("close and spread_bps must match dates x symbols")
        if len(dates) < 1_600 or len(symbols) < 2:
            raise ValueError("market panel needs at least 1,600 dates and two symbols")
        if not np.all(np.isfinite(close_array)) or np.any(close_array <= 0):
            raise ValueError("close prices must be finite and positive")
        if not np.all(np.isfinite(spread_array)) or np.any(spread_array < 0):
            raise ValueError("spread proxies must be finite and non-negative")
        if tuple(sorted(dates)) != dates or len(set(dates)) != len(dates):
            raise ValueError("dates must be unique and sorted")
        if len(set(symbols)) != len(symbols):
            raise ValueError("symbols must be unique")

        self.dates = dates
        self.symbols = symbols
        self.close = close_array
        self.spread_bps = spread_array
        self.source_label = source_label
        self.returns = np.zeros_like(close_array)
        self.returns[1:] = close_array[1:] / close_array[:-1] - 1.0

    @classmethod
    def synthetic(cls, seed: int = 17) -> "MarketData":
        rng = np.random.default_rng(seed)
        symbols = tuple([f"ASSET_{i:02d}" for i in range(1, 21)] + ["INDEX_A", "INDEX_B"])
        n_days = 3395
        dates = cls._business_dates(date(2012, 1, 3), n_days)
        n_symbols = len(symbols)

        market = np.zeros(n_days)
        latent_alpha = np.zeros((n_days, n_symbols))
        returns = np.zeros((n_days, n_symbols))
        betas = rng.uniform(0.75, 1.25, size=n_symbols)
        idio_scale = rng.uniform(0.006, 0.012, size=n_symbols)

        for day_index in range(1, n_days):
            slow_cycle = np.sin(day_index / 180.0)
            regime_vol = 0.0065 + 0.0045 * (0.5 + 0.5 * np.sin(day_index / 95.0))
            market[day_index] = (
                0.00022
                + 0.00018 * slow_cycle
                + regime_vol * rng.standard_normal()
            )
            latent_alpha[day_index] = (
                0.992 * latent_alpha[day_index - 1]
                + rng.normal(0.0, 0.00016, size=n_symbols)
            )
            returns[day_index] = (
                betas * market[day_index]
                + latent_alpha[day_index - 1]
                + idio_scale * rng.standard_normal(n_symbols)
            )

        for shock_day in range(420, n_days - 20, 233):
            symbol_index = int(rng.integers(0, n_symbols - 2))
            returns[shock_day, symbol_index] += 0.10
            returns[shock_day + 1 : shock_day + 8, symbol_index] -= 0.018

        returns = np.clip(returns, -0.22, 0.22)
        close = 100.0 * np.exp(np.cumsum(np.log1p(returns), axis=0))
        volume_scale = rng.lognormal(mean=0.0, sigma=0.35, size=(n_days, n_symbols))
        spread_bps = np.clip(
            1.5 + 170.0 * np.abs(returns) / np.sqrt(volume_scale),
            1.0,
            25.0,
        )
        return cls(
            dates=dates,
            symbols=symbols,
            close=close,
            spread_bps=spread_bps,
            source_label=f"deterministic synthetic panel (seed={seed})",
        )

    @classmethod
    def ecb_fx(cls) -> "MarketData":
        """Load the bundled, source-standard ECB reference-rate snapshot."""
        snapshot_path = (
            Path(__file__).parent
            / "data"
            / "ecb_exr_reference_29_2009_2024.csv.gz"
        )
        if not snapshot_path.is_file():
            raise ValueError(f"ECB FX snapshot is not a file: {snapshot_path}")
        snapshot_sha256 = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
        if snapshot_sha256 != ECB_FX_SNAPSHOT_SHA256:
            raise ValueError("ECB FX snapshot checksum does not match provenance")

        series: dict[str, dict[str, float]] = {
            currency: {} for currency in ECB_FX_CURRENCIES
        }
        with gzip.open(
            snapshot_path,
            mode="rt",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            reader = csv.DictReader(handle)
            required = {
                "FREQ",
                "CURRENCY",
                "CURRENCY_DENOM",
                "EXR_TYPE",
                "EXR_SUFFIX",
                "TIME_PERIOD",
                "OBS_VALUE",
                "OBS_STATUS",
                "TITLE_COMPL",
            }
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError("ECB FX snapshot is missing required source columns")
            for row in reader:
                currency = row["CURRENCY"]
                if currency not in series:
                    raise ValueError(f"unexpected currency in ECB FX snapshot: {currency}")
                if (
                    row["FREQ"] != "D"
                    or row["CURRENCY_DENOM"] != "EUR"
                    or row["EXR_TYPE"] != "SP00"
                    or row["EXR_SUFFIX"] != "A"
                    or not row["TITLE_COMPL"].startswith(
                        "ECB reference exchange rate"
                    )
                ):
                    raise ValueError("ECB FX snapshot contains an unexpected series")
                if not row["OBS_VALUE"]:
                    if row["OBS_STATUS"] != "H":
                        raise ValueError("ECB FX snapshot contains an empty non-holiday row")
                    continue
                if row["OBS_STATUS"] != "A":
                    raise ValueError("ECB FX snapshot contains a non-final observation")
                observation_date = row["TIME_PERIOD"]
                if observation_date in series[currency]:
                    raise ValueError(
                        f"duplicate ECB FX row for {observation_date} / {currency}"
                    )
                series[currency][observation_date] = float(row["OBS_VALUE"])

        date_sets = [set(series[currency]) for currency in ECB_FX_CURRENCIES]
        if any(not values for values in date_sets):
            raise ValueError("ECB FX snapshot is missing a configured currency")
        common_dates = set.intersection(*date_sets)
        if any(values != common_dates for values in date_sets):
            raise ValueError("ECB FX snapshot does not form a rectangular panel")
        dates = tuple(sorted(common_dates))
        if (
            len(dates) != 4_098
            or dates[0] != "2009-01-02"
            or dates[-1] != "2024-12-31"
        ):
            raise ValueError("ECB FX snapshot does not match the pinned 2009-2024 panel")

        symbols = tuple(f"EUR{currency}" for currency in ECB_FX_CURRENCIES)
        close = np.asarray(
            [
                [series[currency][observation_date] for currency in ECB_FX_CURRENCIES]
                for observation_date in dates
            ],
            dtype=float,
        )
        return cls(
            dates=dates,
            symbols=symbols,
            close=close,
            spread_bps=np.full_like(close, 5.0),
            source_label=(
                "ECB daily euro foreign exchange reference rates "
                "(2009-2024; source values unmodified)"
            ),
        )

    @classmethod
    def from_csv(cls, path: Path) -> "MarketData":
        if not path.is_file():
            raise ValueError(f"data_path is not a file: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"date", "symbol", "close"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError("CSV must contain date,symbol,close columns")
            records = list(reader)
        if not records:
            raise ValueError("CSV contains no rows")

        dates = tuple(sorted({row["date"] for row in records}))
        symbols = tuple(sorted({row["symbol"] for row in records}))
        date_index = {value: index for index, value in enumerate(dates)}
        symbol_index = {value: index for index, value in enumerate(symbols)}
        close = np.full((len(dates), len(symbols)), np.nan)
        high = np.full_like(close, np.nan)
        low = np.full_like(close, np.nan)
        volume = np.full_like(close, np.nan)

        for row in records:
            i = date_index[row["date"]]
            j = symbol_index[row["symbol"]]
            if np.isfinite(close[i, j]):
                raise ValueError(f"duplicate CSV row for {row['date']} / {row['symbol']}")
            close[i, j] = float(row["close"])
            if row.get("high"):
                high[i, j] = float(row["high"])
            if row.get("low"):
                low[i, j] = float(row["low"])
            if row.get("volume"):
                volume[i, j] = float(row["volume"])
        if np.any(~np.isfinite(close)):
            raise ValueError("CSV panel must be rectangular with every date/symbol close")

        if np.all(np.isfinite(high)) and np.all(np.isfinite(low)):
            spread_bps = np.clip(0.02 * (high - low) / close * 10_000.0, 1.0, 25.0)
        elif np.all(np.isfinite(volume)) and np.all(volume > 0):
            median_volume = np.median(volume, axis=0)
            spread_bps = np.clip(4.0 / np.sqrt(volume / median_volume), 1.0, 25.0)
        else:
            spread_bps = np.full_like(close, 5.0)
        return cls(
            dates=dates,
            symbols=symbols,
            close=close,
            spread_bps=spread_bps,
            source_label=f"caller-provided CSV ({path.name})",
        )

    @staticmethod
    def _business_dates(start: date, count: int) -> tuple[str, ...]:
        values: list[str] = []
        cursor = start
        while len(values) < count:
            if cursor.weekday() < 5:
                values.append(cursor.isoformat())
            cursor += timedelta(days=1)
        return tuple(values)

    @staticmethod
    def _percentile_ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(1, len(values) + 1, dtype=float)
        return ranks / len(values)

    def feature_snapshot(self, as_of_index: int) -> FeatureSnapshot:
        if as_of_index < 253 or as_of_index >= len(self.dates):
            raise ValueError("as_of_index lacks the required trailing history")
        last = as_of_index - 1
        ret_252 = self.close[last] / self.close[last - 252] - 1.0
        ret_20 = self.close[last] / self.close[last - 20] - 1.0
        vol_20 = np.std(self.returns[as_of_index - 20 : as_of_index], axis=0, ddof=1) * np.sqrt(252.0)
        rank_20 = self._percentile_ranks(ret_20)
        rank_252 = self._percentile_ranks(ret_252)

        historical_vol: list[float] = []
        for cursor in range(max(20, as_of_index - 252), as_of_index):
            window = self.returns[cursor - 20 : cursor]
            historical_vol.append(float(np.median(np.std(window, axis=0, ddof=1) * np.sqrt(252.0))))
        current_vol = float(np.median(vol_20))
        lower, upper = np.quantile(np.asarray(historical_vol), [1.0 / 3.0, 2.0 / 3.0])
        regime = "low-vol" if current_vol <= lower else "high-vol" if current_vol >= upper else "mid-vol"

        rows = tuple(
            {
                "symbol": symbol,
                "return_252d": float(ret_252[index]),
                "realized_vol_20d": float(vol_20[index]),
                "rs_rank_20d": float(rank_20[index]),
                "rs_rank_252d": float(rank_252[index]),
            }
            for index, symbol in enumerate(self.symbols)
        )
        return FeatureSnapshot(
            as_of_date=self.dates[as_of_index],
            regime=regime,
            rows=rows,
            summary={
                "median_return_252d": float(np.median(ret_252)),
                "median_realized_vol_20d": current_vol,
                "dispersion_return_252d": float(np.std(ret_252, ddof=1)),
            },
        )

    def render_prompt(self, as_of_index: int) -> str:
        snapshot = self.feature_snapshot(as_of_index)
        lines = [
            "Design one strategy for the Gate Runner honest-grading benchmark.",
            f"Data source: {self.source_label}.",
            f"Point-in-time cutoff: {snapshot.as_of_date}. Every feature below uses rows strictly before that date.",
            "Future grading windows and their outcomes are hidden. Costs are 10 bps per traded side plus a spread proxy.",
            f"Regime: {snapshot.regime}.",
            (
                "Cross-sectional summary: "
                f"median_252d_return={snapshot.summary['median_return_252d']:.4f}, "
                f"median_20d_vol={snapshot.summary['median_realized_vol_20d']:.4f}, "
                f"return_dispersion={snapshot.summary['dispersion_return_252d']:.4f}."
            ),
            "Symbols (252d_return, 20d_realized_vol, 20d_RS_percentile, 252d_RS_percentile):",
        ]
        lines.extend(
            (
                f"- {row['symbol']}: {row['return_252d']:.4f}, "
                f"{row['realized_vol_20d']:.4f}, {row['rs_rank_20d']:.3f}, "
                f"{row['rs_rank_252d']:.3f}"
            )
            for row in snapshot.rows
        )
        lines.extend(
            [
                (
                    "Return exactly one JSON object. The object below is a "
                    "deliberately inactive syntax example, not a recommended "
                    "strategy. Choose types and parameters for the market brief."
                ),
                StrategyParser.ACTION_CONTRACT,
                StrategyParser.ACTION_RULES,
                "Do not include markdown fences, commentary, NaN, Infinity, extra keys, or multiple objects.",
            ]
        )
        return "\n".join(lines)


class TaskDatasetFactory:
    def __init__(self, market: MarketData, windows: int, window_days: int, seed: int) -> None:
        self.market = market
        self.windows = windows
        self.window_days = window_days
        self.seed = seed

    def build(self, train_examples: int, eval_examples: int) -> tuple[Dataset, Dataset]:
        horizon = self.windows * self.window_days
        first_start = 300
        last_start = len(self.market.dates) - horizon - 1
        split_start = first_start + int(0.70 * (last_start - first_start))
        train_candidates = np.arange(
            first_start,
            split_start - horizon + 1,
            5,
            dtype=int,
        )
        eval_candidates = np.arange(split_start, last_start, 5, dtype=int)
        if train_examples > len(train_candidates):
            raise ValueError(
                f"requested {train_examples} train examples but the embargoed panel only supports {len(train_candidates)}"
            )
        if eval_examples > len(eval_candidates):
            raise ValueError(
                f"requested {eval_examples} eval examples but the held-out panel only supports {len(eval_candidates)}"
            )
        rng = np.random.default_rng(self.seed)
        train_indices = rng.permutation(train_candidates)[:train_examples]
        eval_indices = rng.permutation(eval_candidates)[:eval_examples]

        def rows(indices: np.ndarray) -> list[dict[str, str]]:
            return [
                {
                    "question": self.market.render_prompt(int(as_of_index)),
                    "answer": "",
                    "info": json.dumps(
                        {
                            "as_of_index": int(as_of_index),
                            "as_of_date": self.market.dates[int(as_of_index)],
                            "data_source": self.market.source_label,
                        }
                    ),
                }
                for as_of_index in indices
            ]

        return Dataset.from_list(rows(train_indices)), Dataset.from_list(rows(eval_indices))
