import csv
import gzip
import hashlib
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

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
PUBLIC_SHORT_RATES_SNAPSHOT_SHA256 = (
    "b19ec2ce26925be2b5cec048c51e3da20cd55ee4626393c259591405a45a5cfc"
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
        carry_returns: np.ndarray | None = None,
        foreign_reference_rates_percent: np.ndarray | None = None,
        base_reference_rates_percent: np.ndarray | None = None,
        rate_source_label: str | None = None,
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

        rate_inputs = (
            carry_returns,
            foreign_reference_rates_percent,
            base_reference_rates_percent,
        )
        if any(value is not None for value in rate_inputs) and not all(
            value is not None for value in rate_inputs
        ):
            raise ValueError("carry and both reference-rate arrays must be supplied together")
        if all(value is not None for value in rate_inputs):
            carry_array = np.asarray(carry_returns, dtype=float)
            foreign_rate_array = np.asarray(
                foreign_reference_rates_percent, dtype=float
            )
            base_rate_array = np.asarray(base_reference_rates_percent, dtype=float)
            for name, value in (
                ("carry_returns", carry_array),
                ("foreign_reference_rates_percent", foreign_rate_array),
                ("base_reference_rates_percent", base_rate_array),
            ):
                if value.shape != expected_shape or not np.all(np.isfinite(value)):
                    raise ValueError(f"{name} must be finite and match dates x symbols")
            if np.any(carry_array <= -1.0):
                raise ValueError("carry returns must be greater than -100%")
            if not rate_source_label:
                raise ValueError("rate_source_label is required with reference rates")
        else:
            carry_array = np.zeros(expected_shape, dtype=float)
            foreign_rate_array = np.zeros(expected_shape, dtype=float)
            base_rate_array = np.zeros(expected_shape, dtype=float)
            if rate_source_label is not None:
                raise ValueError("rate_source_label requires reference-rate arrays")

        self.dates = dates
        self.symbols = symbols
        self.close = close_array
        self.spread_bps = spread_array
        self.source_label = source_label
        self.carry_returns = carry_array
        self.foreign_reference_rates_percent = foreign_rate_array
        self.base_reference_rates_percent = base_rate_array
        self.rate_source_label = rate_source_label
        self.has_carry = rate_source_label is not None
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
    def ecb_fx(cls, include_carry: bool = False) -> "MarketData":
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

        profile_currencies = ECB_FX_CURRENCIES
        symbols = tuple(f"EUR{currency}" for currency in profile_currencies)
        close = np.asarray(
            [
                [series[currency][observation_date] for currency in ECB_FX_CURRENCIES]
                for observation_date in dates
            ],
            dtype=float,
        )
        kwargs: dict[str, object] = {}
        source_label = (
            "ECB daily euro foreign exchange reference rates "
            "(2009-2024; source values unmodified)"
        )
        if include_carry:
            profile_currencies = tuple(
                currency for currency in ECB_FX_CURRENCIES if currency != "SGD"
            )
            keep_indices = [
                ECB_FX_CURRENCIES.index(currency) for currency in profile_currencies
            ]
            close = close[:, keep_indices]
            symbols = tuple(f"EUR{currency}" for currency in profile_currencies)
            foreign_rates, eur_rates = cls._load_public_short_rates(
                dates=dates,
                currencies=profile_currencies,
            )
            calendar_days = np.diff(np.asarray(dates, dtype="datetime64[D]")).astype(
                float
            )
            carry_returns = np.zeros_like(close)
            # EURXXX is foreign-currency units per EUR. A positive position is
            # long EUR funded in XXX, so its annual carry is i_EUR - i_XXX.
            carry_returns[1:] = (
                (eur_rates[:-1] - foreign_rates[:-1])
                / 100.0
                * calendar_days[:, None]
                / 365.0
            )
            kwargs = {
                "carry_returns": carry_returns,
                "foreign_reference_rates_percent": foreign_rates,
                "base_reference_rates_percent": eur_rates,
                "rate_source_label": (
                    "BIS policy rates and the BNB base rate; economic-PIT "
                    "availability rules; SGD excluded for redistribution"
                ),
            }
            source_label += " with public short-rate carry proxy (28 pairs)"
        return cls(
            dates=dates,
            symbols=symbols,
            close=close,
            spread_bps=np.full_like(close, 5.0),
            source_label=source_label,
            **kwargs,
        )

    @classmethod
    def _load_public_short_rates(
        cls,
        dates: tuple[str, ...],
        currencies: tuple[str, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        snapshot_path = (
            Path(__file__).parent
            / "data"
            / "public_short_rates_2008_2024.csv.gz"
        )
        if not snapshot_path.is_file():
            raise ValueError(f"public short-rate snapshot is not a file: {snapshot_path}")
        snapshot_sha256 = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
        if snapshot_sha256 != PUBLIC_SHORT_RATES_SNAPSHOT_SHA256:
            raise ValueError("public short-rate snapshot checksum does not match provenance")

        required_currencies = set(currencies) | {"EUR"}
        observations: dict[str, list[tuple[str, str, float]]] = {
            currency: [] for currency in required_currencies
        }
        rate_types: dict[str, set[str]] = {
            currency: set() for currency in required_currencies
        }
        with gzip.open(snapshot_path, mode="rt", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required_columns = {
                "currency",
                "observation_date",
                "available_date",
                "rate_percent",
                "rate_type",
                "source_name",
                "source_series",
                "source_area_code",
            }
            if reader.fieldnames is None or set(reader.fieldnames) != required_columns:
                raise ValueError("public short-rate snapshot has unexpected columns")
            seen: set[tuple[str, str]] = set()
            for row in reader:
                currency = row["currency"]
                if currency not in required_currencies:
                    continue
                key = (currency, row["observation_date"])
                if key in seen:
                    raise ValueError(f"duplicate public short-rate observation: {key}")
                seen.add(key)
                if row["available_date"] < row["observation_date"]:
                    raise ValueError("short-rate availability precedes observation")
                value = float(row["rate_percent"])
                if not np.isfinite(value):
                    raise ValueError("public short-rate snapshot contains a non-finite rate")
                observations[currency].append(
                    (row["available_date"], row["observation_date"], value)
                )
                rate_types[currency].add(row["rate_type"])

        expected_types = {
            currency: (
                {"base_rate"}
                if currency == "BGN"
                else {"overnight_benchmark"}
                if currency == "SGD"
                else {"policy_rate"}
            )
            for currency in required_currencies
        }
        if rate_types != expected_types:
            raise ValueError("public short-rate snapshot has unexpected rate types")

        aligned: dict[str, np.ndarray] = {}
        for currency, values in observations.items():
            if not values:
                raise ValueError(f"no public short-rate rows for {currency}")
            values.sort(key=lambda item: (item[0], item[1]))
            result = np.full(len(dates), np.nan, dtype=float)
            source_index = 0
            latest = np.nan
            for market_index, market_date in enumerate(dates):
                while (
                    source_index < len(values)
                    and values[source_index][0] <= market_date
                ):
                    latest = values[source_index][2]
                    source_index += 1
                result[market_index] = latest
            if not np.all(np.isfinite(result)):
                raise ValueError(f"public short-rate history does not cover {currency}")
            aligned[currency] = result

        foreign_rates = np.column_stack([aligned[currency] for currency in currencies])
        eur_rates = np.repeat(aligned["EUR"][:, None], len(currencies), axis=1)
        return foreign_rates, eur_rates

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

        rows_list: list[dict[str, float | str]] = []
        for index, symbol in enumerate(self.symbols):
            row: dict[str, float | str] = {
                "symbol": symbol,
                "return_252d": float(ret_252[index]),
                "realized_vol_20d": float(vol_20[index]),
                "rs_rank_20d": float(rank_20[index]),
                "rs_rank_252d": float(rank_252[index]),
            }
            if self.has_carry:
                foreign_rate = float(
                    self.foreign_reference_rates_percent[last, index]
                )
                eur_rate = float(self.base_reference_rates_percent[last, index])
                tenor_years = 0.25
                forward_ratio = (1.0 + foreign_rate / 100.0 * tenor_years) / (
                    1.0 + eur_rate / 100.0 * tenor_years
                )
                row.update(
                    {
                        "eur_reference_rate_pct": eur_rate,
                        "foreign_reference_rate_pct": foreign_rate,
                        "long_eur_carry_pct_pa": eur_rate - foreign_rate,
                        "cip_forward_points_3m_pct": (forward_ratio - 1.0) * 100.0,
                    }
                )
            rows_list.append(row)
        rows = tuple(rows_list)
        summary = {
            "median_return_252d": float(np.median(ret_252)),
            "median_realized_vol_20d": current_vol,
            "dispersion_return_252d": float(np.std(ret_252, ddof=1)),
        }
        if self.has_carry:
            summary["median_long_eur_carry_pct_pa"] = float(
                np.median(
                    self.base_reference_rates_percent[last]
                    - self.foreign_reference_rates_percent[last]
                )
            )
        return FeatureSnapshot(
            as_of_date=self.dates[as_of_index],
            regime=regime,
            rows=rows,
            summary=summary,
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
        ]
        if self.has_carry:
            lines.extend(
                [
                    (
                        f"Reference short rates: {self.rate_source_label}. Values "
                        "respect their recorded availability dates."
                    ),
                    (
                        "EURXXX means foreign-currency units per EUR. A positive "
                        "position is long EUR funded in XXX; grading accrues the "
                        "displayed EUR-minus-foreign carry over actual calendar days."
                    ),
                    (
                        "The 3m forward-points field is a covered-interest-parity "
                        "proxy, not an executable forward quote."
                    ),
                    (
                        "Symbols (252d_return, 20d_realized_vol, 20d_RS_percentile, "
                        "252d_RS_percentile, EUR_rate_pct, foreign_rate_pct, "
                        "long_EUR_carry_pct_pa, CIP_3m_forward_points_pct):"
                    ),
                ]
            )
            lines.extend(
                (
                    f"- {row['symbol']}: {row['return_252d']:.4f}, "
                    f"{row['realized_vol_20d']:.4f}, {row['rs_rank_20d']:.3f}, "
                    f"{row['rs_rank_252d']:.3f}, "
                    f"{row['eur_reference_rate_pct']:.4f}, "
                    f"{row['foreign_reference_rate_pct']:.4f}, "
                    f"{row['long_eur_carry_pct_pa']:.4f}, "
                    f"{row['cip_forward_points_3m_pct']:.4f}"
                )
                for row in snapshot.rows
            )
        else:
            lines.append(
                "Symbols (252d_return, 20d_realized_vol, 20d_RS_percentile, 252d_RS_percentile):"
            )
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
