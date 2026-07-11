# Public data provenance

Gate Runner bundles three pinned public-data artifacts. The `ecb_fx` profile is
the 29-pair spot-only ablation. The recommended `ecb_fx_carry` profile combines
28 of those pairs with economically point-in-time reference short rates. The
Bank of England panel is diagnostic only and never enters prompts or rewards.

| Artifact | Runtime role | Coverage |
| --- | --- | --- |
| `ecb_exr_reference_29_2009_2024.csv.gz` | ECB spot/reference rates | 29 EUR pairs, 2009–2024 |
| `public_short_rates_2008_2024.csv.gz` | Carry and CIP proxy inputs | EUR plus 28 foreign currencies, 2008–2024 |
| `boe_gbpusd_spot_forward_2009_2024.csv.gz` | External proxy validation | GBP/USD spot and 1m/3m/6m/12m forwards |

Gate Runner's Apache-2.0 license covers the project code. It does not relicense
publisher data. Users must retain the attributions and comply with the source
terms below.

## ECB EUR reference rates

- **Publisher:** European Central Bank
- **Dataset:** EXR, daily euro foreign exchange reference rates
- **Coverage:** 29 currencies, 2009-01-01 through 2024-12-31 in the query;
  4,098 common observed dates from 2009-01-02 through 2024-12-31 after holiday
  rows without values are excluded at runtime
- **Retrieved:** 2026-07-10
- **API query:**
  `https://data-api.ecb.europa.eu/service/data/EXR/D.AUD+BGN+BRL+CAD+CHF+CNY+CZK+DKK+GBP+HKD+HUF+IDR+ILS+INR+JPY+KRW+MXN+MYR+NOK+NZD+PHP+PLN+RON+SEK+SGD+THB+TRY+USD+ZAR.EUR.SP00.A?startPeriod=2009-01-01&endPeriod=2024-12-31&format=csvdata`
- **Source CSV SHA-256:**
  `c95c69519eb159bff0287f33cde600d065876bc401103c542b1da439d88b4756`
- **Bundled gzip SHA-256:**
  `2e39da0d83bfbdb6d0575b1e481f2094ea974018b33473785d9349e2d04d7f3f`

Runtime removes source rows whose `OBS_STATUS` is `H` and whose observation is
empty, validates the series dimensions and final status, labels each series
`EUR` plus the source currency, and attaches a generated constant 5 bps spread
proxy. It does not invert, interpolate, or adjust source observations.

The ECB permits free reuse of publicly available ESCB statistics when the
source is quoted and the statistics are reproduced accurately:

- https://www.ecb.europa.eu/stats/ecb_statistics/governance_and_quality_framework/html/usage_policy.en.html
- https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html

## Public reference short rates

The normalized snapshot contains 133,850 observations from 2008-01-01 through
2024-12-31. Its full source checksums, publisher attribution by currency, row
counts, and transformation descriptions are machine-readable in
`gate_runner_core/data/public_short_rates_2008_2024.manifest.json`.

- **Normalized CSV SHA-256:**
  `e2e13c0e21a42d87d0dfe77dc475dcd11447795688dd0cb285350c256c5e653c`
- **Bundled gzip SHA-256:**
  `b19ec2ce26925be2b5cec048c51e3da20cd55ee4626393c259591405a45a5cfc`
- **Retrieved:** 2026-07-11

### Sources and redistribution

**Bank for International Settlements.** Daily central-bank policy-rate series
provide EUR and 27 foreign-currency rates. Only observations marked normal and
free/public, expressed in percent per year, are selected. The snapshot retains
the national publisher named by BIS for every currency.

- Dataset: https://data.bis.org/topics/CBPOL
- Bulk source: https://data.bis.org/static/bulk/WS_CBPOL_csv_flat.zip
- Documentation: https://www.bis.org/statistics/cbpol/cbpol_doc.pdf
- Terms: https://www.bis.org/terms_statistics.htm
- Downloaded ZIP SHA-256:
  `2af3b1991275891ff1bcb5778a4dd95b451ed37d349ca2cb5b2f95ceac9345fa`

BIS permits these series to be downloaded, reproduced, and disseminated when
the appropriate national source is quoted and the BIS terms are observed.

**Bulgarian National Bank.** BGN uses the publisher's monthly Base Interest
Rate because BIS WS_CBPOL has no daily BGN series. The 204 published values are
parsed from five official queries of less than 60 months each; the exact URLs
and response checksums are retained in the manifest.

- Series: https://www.bnb.bg/Statistics/StHistoricalData/StBIRAndIndices/StBIBaseInterestRate/index.htm
- Reuse terms: https://www.bnb.bg/AboutUs/PressOffice/PORightsUsing/index.htm

BNB permits files and data to be saved, distributed, and reproduced when the
source is identified and the material is not changed or distorted. Gate Runner
does not alter the rate observations; its added source and availability fields
are disclosed transformations.

**SGD exclusion.** MAS SORA is technically suitable and its official download
contains both value date and publication date. However, Gate Runner did not
establish redistribution permission for that file while preparing this public
release. `ecb_fx_carry` therefore excludes `EURSGD`; `ecb_fx` retains its ECB
spot observations. No zero rate, interpolation, or substitute is fabricated.

### Normalized fields

```text
currency
observation_date
available_date
rate_percent
rate_type
source_name
source_series
source_area_code
```

The rate types are `policy_rate` for BIS observations and `base_rate` for BGN.
They are reference short-rate proxies, not a homogeneous set of executable
deposit rates.

### Point-in-time policy

The snapshot implements an **economic PIT reconstruction**:

- BIS observations become available on the next weekday after their stated
  observation/effective date.
- BNB monthly values become available on the next weekday after their stated
  effective date.
- Runtime performs an as-of join using `available_date`, never a backward fill.
- Prompt features end at the last market date strictly before the cutoff.
- Carry for an interval uses the rates available on the interval's prior market
  date.

The current BIS and BNB downloads are not archives of every historical release
vintage. The one-weekday lag is intentionally conservative about same-day use,
but this remains economic PIT rather than source-vintage PIT. Future source
downloads should be archived with their checksums to create a true vintage
history.

### Carry and implied-forward calculations

ECB quotes `EURXXX` as foreign-currency units per EUR. A positive Gate Runner
position is long EUR and funded in XXX. For a market interval of `d` calendar
days, the carry proxy is:

```text
carry = (i_EUR - i_XXX) / 100 * d / 365
```

The backtester compounds spot and carry before subtracting transaction costs:

```text
total_return = (1 + spot_return) * (1 + carry) - 1 - costs
```

For a tenor `t` in years, the prompt's covered-interest-parity proxy is:

```text
F_t = S_t * (1 + i_XXX*t) / (1 + i_EUR*t)
```

The prompt reports three-month forward points as `(F_t / S_t - 1) * 100`.
These are explicitly labeled implied proxies, not actual forward quotes.
Strategies can select `universe_filter.rank_by="long_eur_carry"` to rank pairs
by `i_EUR - i_XXX`; the existing `relative_strength_252d` rank remains available
for matched ablations.

## Bank of England forward validation

The bundled source-standard CSV contains daily GBP/USD spot plus 1-, 3-, 6-,
and 12-month forward series. There are 4,041 spot rows through 2024-12-31 and
3,157 forward rows through 2021-06-30.

- **Publisher:** Bank of England
- **Series:** `XUDLUSS`, `XUDLDS1`, `XUDLDS3`, `XUDLDS6`, `XUDLDSY`
- **Query documentation:**
  https://www.bankofengland.co.uk/boeapps/database/Help.asp
- **Series catalogue:**
  https://www.bankofengland.co.uk/boeapps/database/FromShowColumns.asp?CategId=6&FromCategoryList=Yes&HighlightCatValueDisplay=Exchange+rate+%28forward%29+-+US+dollar+into+sterling&NewMeaningId=RUSF&Travel=NIxAZxI1x
- **Source CSV SHA-256:**
  `0aa39a8a48c1163a871ed9fd3a75fd34c3235739b717d3881099fd2aa15a26bc`
- **Bundled gzip SHA-256:**
  `5913c8b8dabf0967281c33ae1bb15d2390378aa7bae5840f40e6747756e41c9c`
- **Terms:** https://www.bankofengland.co.uk/legal

The Bank of England states that Database reproduction is subject to the UK Open
Government Licence. This panel is never used for training reward, pass status,
or task generation.

`scripts/validate_cip_proxy.py` compares untouched BoE forwards with a proxy
formed from Gate Runner's GBP and USD reference rates. On the pinned data's
3,149 common observations, the current result is:

| Tenor | Mean proxy error | Mean absolute error | Premium correlation |
| --- | ---: | ---: | ---: |
| 1 month | -28.05 bp annualized | 28.33 bp | 0.9519 |
| 3 months | -24.71 bp annualized | 25.33 bp | 0.9613 |
| 6 months | -23.80 bp annualized | 25.31 bp | 0.9501 |
| 12 months | -23.50 bp annualized | 27.86 bp | 0.9259 |

The high correlation supports using the differential as a directional carry
feature. The persistent 24–28 bp annualized error is equally important: policy
rates omit term structure, cross-currency basis, liquidity, and dealing costs,
so the proxy must not be presented as an executable forward price.

## Rebuilding and validating

From the repository root:

```bash
python3 scripts/build_public_short_rates.py \
  --output environments/gate_runner/gate_runner_core/data/public_short_rates_2008_2024.csv.gz \
  --manifest environments/gate_runner/gate_runner_core/data/public_short_rates_2008_2024.manifest.json

python3 scripts/build_boe_forward_validation.py \
  --output environments/gate_runner/gate_runner_core/data/boe_gbpusd_spot_forward_2009_2024.csv.gz

uv run --project environments/gate_runner python scripts/validate_cip_proxy.py
```

Upstream publishers can revise history. A rebuild is expected to fail pinned
checksum tests or produce new checksums when source data changes; review those
changes rather than updating hashes mechanically.

## General limitations

ECB reference rates are informational rather than executable quotes. The panel
has no OHLCV or bid/ask history, so Gate Runner uses a disclosed 5 bps spread
proxy. Reference policy/base rates are not directly investable funding rates.
The setup omits forward curves, cross-currency basis, collateral conventions,
capital controls, credit, taxes, and market impact. Results are benchmark
scores, not evidence of realizable investment performance or investment advice.
