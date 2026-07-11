#!/usr/bin/env python3
"""Compare the public-rate CIP proxy with untouched BoE GBP/USD forwards."""

from __future__ import annotations

import csv
import gzip
from datetime import datetime
from pathlib import Path

import numpy as np

from gate_runner_core.market import MarketData


BOE_PATH = (
    Path(__file__).parents[1]
    / "environments"
    / "gate_runner"
    / "gate_runner_core"
    / "data"
    / "boe_gbpusd_spot_forward_2009_2024.csv.gz"
)
TENORS = {
    "XUDLDS1": 1.0 / 12.0,
    "XUDLDS3": 0.25,
    "XUDLDS6": 0.50,
    "XUDLDSY": 1.00,
}


def main() -> None:
    market = MarketData.ecb_fx(include_carry=True)
    market_date_index = {value: index for index, value in enumerate(market.dates)}
    gbp_index = market.symbols.index("EURGBP")
    usd_index = market.symbols.index("EURUSD")
    errors: dict[str, list[float]] = {series: [] for series in TENORS}
    actual_premiums: dict[str, list[float]] = {series: [] for series in TENORS}
    implied_premiums: dict[str, list[float]] = {series: [] for series in TENORS}

    with gzip.open(BOE_PATH, mode="rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            iso_date = datetime.strptime(row["DATE"], "%d %b %Y").date().isoformat()
            market_index = market_date_index.get(iso_date)
            if market_index is None or not row["XUDLUSS"]:
                continue
            spot = float(row["XUDLUSS"])
            usd_rate = market.foreign_reference_rates_percent[
                market_index, usd_index
            ]
            gbp_rate = market.foreign_reference_rates_percent[
                market_index, gbp_index
            ]
            for series, tenor in TENORS.items():
                if not row[series]:
                    continue
                actual = (float(row[series]) / spot - 1.0) / tenor
                implied_forward = spot * (1.0 + usd_rate / 100.0 * tenor) / (
                    1.0 + gbp_rate / 100.0 * tenor
                )
                implied = (implied_forward / spot - 1.0) / tenor
                # Annualized premium error in basis points.
                errors[series].append((implied - actual) * 10_000.0)
                actual_premiums[series].append(actual)
                implied_premiums[series].append(implied)

    print("BoE GBP/USD actual forwards vs public-rate CIP proxy")
    print("series,observations,mean_error_bp,mae_bp,correlation")
    for series in TENORS:
        error = np.asarray(errors[series], dtype=float)
        actual = np.asarray(actual_premiums[series], dtype=float)
        implied = np.asarray(implied_premiums[series], dtype=float)
        correlation = float(np.corrcoef(actual, implied)[0, 1])
        print(
            f"{series},{len(error)},{np.mean(error):.2f},"
            f"{np.mean(np.abs(error)):.2f},{correlation:.4f}"
        )


if __name__ == "__main__":
    main()
