#!/usr/bin/env python3
"""Build Gate Runner's pinned public short-rate snapshot.

The output is a normalized subset of official publisher data. It is not a raw
source export, so the accompanying manifest records each transformation and
the checksum of every downloaded input.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html.parser
import io
import json
import math
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path


START_DATE = date(2008, 1, 1)
END_DATE = date(2024, 12, 31)
BIS_URL = "https://data.bis.org/static/bulk/WS_CBPOL_csv_flat.zip"
BNB_URL = (
    "https://www.bnb.bg/Statistics/StHistoricalData/StBIRAndIndices/"
    "StBIBaseInterestRate/index.htm"
)

# Currency -> BIS reference-area code. BGN is sourced from its national
# publisher because WS_CBPOL has no daily series for it. SGD is excluded until
# redistribution rights for the technically suitable MAS SORA file are clear.
BIS_AREA_BY_CURRENCY = {
    "AUD": "AU",
    "BRL": "BR",
    "CAD": "CA",
    "CHF": "CH",
    "CNY": "CN",
    "CZK": "CZ",
    "DKK": "DK",
    "EUR": "XM",
    "GBP": "GB",
    "HKD": "HK",
    "HUF": "HU",
    "IDR": "ID",
    "ILS": "IL",
    "INR": "IN",
    "JPY": "JP",
    "KRW": "KR",
    "MXN": "MX",
    "MYR": "MY",
    "NOK": "NO",
    "NZD": "NZ",
    "PHP": "PH",
    "PLN": "PL",
    "RON": "RO",
    "SEK": "SE",
    "THB": "TH",
    "TRY": "TR",
    "USD": "US",
    "ZAR": "ZA",
}


@dataclass(frozen=True)
class RateRow:
    currency: str
    observation_date: str
    available_date: str
    rate_percent: float
    rate_type: str
    source_name: str
    source_series: str
    source_area_code: str


class _BnbTableParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.current_class = ""
        self.current_text: list[str] = []
        self.pending_date: str | None = None
        self.values: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "td":
            self.current_class = dict(attrs).get("class") or ""
            self.current_text = []

    def handle_data(self, data: str) -> None:
        if self.current_class:
            self.current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "td" or not self.current_class:
            return
        value = "".join(self.current_text).strip()
        if self.current_class == "first" and value:
            self.pending_date = value
        elif "last" in self.current_class and self.pending_date and value:
            self.values.append((self.pending_date, value))
            self.pending_date = None
        self.current_class = ""
        self.current_text = []


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download(url: str, data: bytes | None = None) -> bytes:
    request = urllib.request.Request(
        url,
        data=data,
        headers={"User-Agent": "Gate-Runner-public-data-builder/0.3"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _next_business_day(value: date) -> date:
    cursor = value + timedelta(days=1)
    while cursor.weekday() >= 5:
        cursor += timedelta(days=1)
    return cursor


def _bis_rows(zip_bytes: bytes) -> tuple[list[RateRow], dict[str, object]]:
    currency_by_area = {
        area: currency for currency, area in BIS_AREA_BY_CURRENCY.items()
    }
    rows: list[RateRow] = []
    sources: dict[str, str] = {}
    seen: set[tuple[str, str]] = set()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        names = archive.namelist()
        if len(names) != 1 or not names[0].endswith(".csv"):
            raise ValueError("BIS archive must contain exactly one CSV")
        with archive.open(names[0]) as raw:
            text = (line.decode("utf-8-sig") for line in raw)
            reader = csv.DictReader(text)
            for source_row in reader:
                if source_row["FREQ:Frequency"] != "D: Daily":
                    continue
                area = source_row["REF_AREA:Reference area"].split(":", 1)[0]
                currency = currency_by_area.get(area)
                if currency is None:
                    continue
                observation_text = source_row["TIME_PERIOD:Time period or range"]
                if len(observation_text) != 10:
                    continue
                observation_date = date.fromisoformat(observation_text)
                if not START_DATE <= observation_date <= END_DATE:
                    continue
                value = float(source_row["OBS_VALUE:Observation Value"])
                if not math.isfinite(value):
                    continue
                if not source_row["OBS_STATUS:Observation Status"].startswith("A:"):
                    raise ValueError("BIS subset contains a non-normal observation")
                if not source_row["OBS_CONF:Observation confidentiality"].startswith(
                    "F:"
                ):
                    raise ValueError("BIS subset contains a non-public observation")
                if (
                    source_row["UNIT_MEASURE:Unit of measure"]
                    != "368: Per cent per year"
                    or source_row["UNIT_MULT:Unit Multiplier"] != "0: Units"
                ):
                    raise ValueError("BIS subset contains an unexpected unit")
                key = (currency, observation_text)
                if key in seen:
                    raise ValueError(f"duplicate BIS observation: {key}")
                seen.add(key)
                source_name = source_row["SOURCE_REF:Publication Source"].strip()
                sources[currency] = source_name
                rows.append(
                    RateRow(
                        currency=currency,
                        observation_date=observation_text,
                        available_date=_next_business_day(observation_date).isoformat(),
                        rate_percent=value,
                        rate_type="policy_rate",
                        source_name=source_name,
                        source_series="BIS WS_CBPOL daily central bank policy rate",
                        source_area_code=area,
                    )
                )
    missing = sorted(set(BIS_AREA_BY_CURRENCY).difference(sources))
    if missing:
        raise ValueError(f"BIS snapshot is missing currencies: {missing}")
    return rows, {
        "name": "BIS Central Bank Policy Rates",
        "url": BIS_URL,
        "sha256": _sha256(zip_bytes),
        "publisher_sources_by_currency": dict(sorted(sources.items())),
        "transformation": (
            "Selected public, normal-status daily observations for configured "
            "areas; assigned next weekday as a conservative availability date."
        ),
    }


def _bnb_rows() -> tuple[list[RateRow], dict[str, object]]:
    rows: list[RateRow] = []
    inputs: list[dict[str, str]] = []
    # The publisher limits a query to less than 60 months.
    for first_year, last_year in (
        (2008, 2011),
        (2012, 2015),
        (2016, 2019),
        (2020, 2023),
        (2024, 2024),
    ):
        query = urllib.parse.urlencode(
            {
                "firstDays": "01",
                "firstMonths": "01",
                "firstYear": str(first_year),
                "lastDays": "31",
                "lastMonths": "12",
                "lastYear": str(last_year),
                "searchAction": "true",
            }
        )
        url = f"{BNB_URL}?{query}"
        raw = _download(url)
        parser = _BnbTableParser()
        parser.feed(raw.decode("utf-8-sig"))
        if not parser.values:
            raise ValueError(f"BNB query returned no rate rows: {url}")
        inputs.append({"url": url, "sha256": _sha256(raw)})
        for date_text, rate_text in parser.values:
            observation_date = datetime.strptime(date_text, "%d.%m.%Y").date()
            rows.append(
                RateRow(
                    currency="BGN",
                    observation_date=observation_date.isoformat(),
                    available_date=_next_business_day(observation_date).isoformat(),
                    rate_percent=float(rate_text),
                    rate_type="base_rate",
                    source_name="Bulgarian National Bank",
                    source_series="Base Interest Rate",
                    source_area_code="BG",
                )
            )
    if len(rows) != 17 * 12:
        raise ValueError(f"expected 204 monthly BNB rows, found {len(rows)}")
    return rows, {
        "name": "Bulgarian National Bank Base Interest Rate",
        "url": BNB_URL,
        "inputs": inputs,
        "transformation": (
            "Parsed the publisher's monthly HTML table without changing values; "
            "assigned next weekday as a conservative availability date."
        ),
    }


def _render_csv(rows: list[RateRow]) -> bytes:
    output = io.StringIO(newline="")
    fieldnames = list(asdict(rows[0]))
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        values = asdict(row)
        values["rate_percent"] = format(row.rate_percent, ".10g")
        writer.writerow(values)
    return output.getvalue().encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bis-zip",
        type=Path,
        help="Optional local WS_CBPOL_csv_flat.zip; otherwise download it.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    bis_bytes = args.bis_zip.read_bytes() if args.bis_zip else _download(BIS_URL)
    bis_rows, bis_source = _bis_rows(bis_bytes)
    bnb_rows, bnb_source = _bnb_rows()
    rows = sorted(
        bis_rows + bnb_rows,
        key=lambda row: (row.currency, row.observation_date),
    )
    currencies = sorted({row.currency for row in rows})
    expected_currencies = sorted(set(BIS_AREA_BY_CURRENCY) | {"BGN"})
    if currencies != expected_currencies:
        raise ValueError("normalized snapshot has unexpected currency coverage")

    csv_bytes = _render_csv(rows)
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", filename="", mtime=0) as handle:
        handle.write(csv_bytes)
    gzip_bytes = compressed.getvalue()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip_bytes)

    rows_by_currency = {
        currency: sum(row.currency == currency for row in rows)
        for currency in currencies
    }
    manifest = {
        "format_version": 1,
        "dataset": "Gate Runner public reference short rates",
        "retrieved": date.today().isoformat(),
        "coverage": {
            "start": START_DATE.isoformat(),
            "end": END_DATE.isoformat(),
            "currencies": currencies,
            "rows": len(rows),
            "rows_by_currency": rows_by_currency,
        },
        "pit_policy": {
            "BIS": "next weekday after observation date",
            "BNB": "next weekday after effective date",
            "meaning": (
                "economic point-in-time reconstruction, not a historical "
                "archive of BIS or BNB release vintages"
            ),
        },
        "excluded": {
            "SGD": (
                "MAS SORA is technically suitable, but redistribution rights "
                "were not established when this snapshot was built."
            )
        },
        "sources": [bis_source, bnb_source],
        "output": {
            "path": args.output.name,
            "normalized_csv_sha256": _sha256(csv_bytes),
            "bundled_gzip_sha256": _sha256(gzip_bytes),
        },
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {len(rows):,} rows across {len(currencies)} currencies to "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
