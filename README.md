# Gate Runner

Gate Runner is a public [Prime Intellect](https://www.primeintellect.ai/)
[Verifiers](https://github.com/PrimeIntellect-ai/verifiers) environment for
training and evaluating models on honest quantitative-strategy design.

A model receives a point-in-time market brief and returns one strict JSON
strategy. Gate Runner evaluates it across hidden sequential windows, charges
trading and funding costs, deflates Sharpe for repeated trials, and measures
weak-regime and downside behavior. The objective is not to find the prettiest
backtest; it is to reward strategies that remain credible after the search
process and economic frictions are acknowledged.

## Why it exists

Backtests are easy to optimize after the fact. A policy can sample many
plausible strategies, retain the lucky one, and appear skilled when it has only
searched noise. Gate Runner makes that search part of the grade:

- every rollout in an episode group counts as another trial;
- Deflated Sharpe Ratio adjusts for selection and non-normal returns;
- lower-tail window performance rewards stability across regimes;
- normalized expected shortfall catches concentrated daily losses;
- trading costs and FX funding carry enter realized returns;
- CSCV/PBO describes group selection risk without becoming a training target;
- strict point-in-time boundaries keep future observations out of prompts; and
- bounded complexity and activity rules discourage fragile or trivial policies.

The shaped reward is dense on both sides of the hard gate. A failing strategy
improves only by closing its worst normalized violation; crossing every gate
creates a hard reward jump; a passing strategy continues improving toward a
robust margin.

## Data profiles

| Profile | Purpose | Coverage | Carry |
| --- | --- | --- | --- |
| `synthetic` | Self-contained tests and signature oracle | 22 deterministic assets | None |
| `ecb_fx_carry` | Recommended public training/evaluation profile | 28 EUR FX pairs, 2009–2024 | BIS/BNB reference-rate proxy |
| `ecb_fx` | Legacy spot-only ablation | 29 EUR FX pairs, 2009–2024 | None |

`ecb_fx_carry` uses source ECB observations plus economically point-in-time
reference short rates. For `EURXXX`, a positive position is long EUR funded in
XXX, so its annual carry is approximately `i_EUR - i_XXX`. The environment
accrues that differential over actual calendar days before subtracting costs.

Prompts report the underlying rates, annualized long-EUR carry, and a
three-month covered-interest-parity forward-points proxy. The proxy is not an
executable forward quote. Strategies can rank the universe by either
`relative_strength_252d` or `long_eur_carry`, allowing the model to compose a
price signal with an economic funding signal rather than treating momentum as
the only available cross-sectional ordering.

SGD remains in `ecb_fx` but is excluded from `ecb_fx_carry`: MAS SORA is
technically suitable, but its redistribution permission was not established
for this public release. Gate Runner does not fabricate a replacement.

## Quickstart

Prerequisites are Python 3.12+, `uv`, and an authenticated Prime CLI.

Install the public Hub environment:

```bash
prime env install br-322/gate-runner --plain
```

Run the deterministic synthetic smoke test:

```bash
prime eval run br-322/gate-runner
```

Run a 20-example carry-aware preflight:

```bash
prime eval run br-322/gate-runner \
  -a '{"dataset":"ecb_fx_carry","eval_examples":200}' \
  -n 20 -r 3
```

Inspect the reward distribution, hard-pass metric, errors, and samples before
raising `-n` to `200`. Grouped rollouts are required for representative DSR
trial accounting and PBO diagnostics.

For local development:

```bash
prime env install gate-runner --plain
prime eval run gate-runner
uv run --project environments/gate_runner --group dev pytest -q
```

## What the model returns

The action is one strict JSON object containing:

- an entry rule: momentum threshold, mean-reversion z-score, or channel
  breakout;
- an exit rule: fixed stop, trailing stop, or time exit;
- a universe rank, side, and breadth; and
- equal-weight sizing with at most five concurrent positions.

Unknown keys, markdown fences, aliases, invalid primitive names, and
out-of-range values fail closed to zero reward. The exact schema and numeric
bounds are emitted in every prompt and defined in
[`config.py`](environments/gate_runner/gate_runner_core/config.py).

The current FX grammar expresses positive EUR exposure only. It does not yet
support short-EUR positions, leverage, forward-tenor choice, or market-neutral
pair portfolios.

## Scoring

A hard pass requires all of the following:

- `DSR > 0.90`;
- `window_tail_score > -0.50`;
- `expected_shortfall_ratio < 5.0`;
- active positions on at least 10% of grading sessions; and
- activity in at least four of eight grading windows.

PBO is logged as a group diagnostic and does not affect reward or pass status.
The evaluation headline is the logged `passed` metric, not Verifiers' generic
`pass@k` threshold over the shaped reward.

See the [environment contract](environments/gate_runner/README.md) for the full
reward definition, logged metrics, action bounds, and data-path interface.

## Reproducibility and validation

The public package contains:

- the source-standard ECB spot snapshot;
- a normalized BIS/BNB short-rate snapshot with observation date, availability
  date, source, rate type, and checksums;
- a machine-readable source manifest; and
- untouched Bank of England GBP/USD spot/forward observations used only to
  validate the CIP proxy.

The data rebuild scripts are under [`scripts/`](scripts/). On the pinned common
sample, the policy-rate CIP proxy has a 0.93–0.96 correlation with actual BoE
forward premia across 1–12 month tenors, but a roughly 25–28 bp annualized mean
absolute error. That is useful directional validation, not evidence that the
proxy is a tradable price.

Publisher data remains subject to its original terms and is not relicensed
under Apache-2.0. The complete source queries, checksums, transformations,
attributions, licenses, and limitations are in the
[data provenance record](environments/gate_runner/DATA_PROVENANCE.md).

## Spot-only compatibility

The `ecb_fx` profile is intentionally retained. It provides:

- a matched way to measure how much funding carry changes results;
- continuity for existing callers that explicitly request `dataset="ecb_fx"`;
- interpretation of the original spot-only baseline; and
- a guard against accidentally coupling all evaluation behavior to the new
  rate pipeline.

Existing v0.2 strategy JSON remains schema-compatible because a missing
`universe_filter.rank_by` defaults to `relative_strength_252d`. This is not a
promise of bit-for-bit v0.2 environment behavior: the current prompt, metric
set, documentation, and package version are v0.3. For an exact historical
checkout, use commit [`99ac503`](https://github.com/BR-322/gate-runner/tree/99ac503).

New training and evaluation configs use `ecb_fx_carry`; `ecb_fx` should be used
as an ablation or historical comparison, not as the default baseline.

## Repository layout

```text
configs/                         # Prime evaluation and RL model-family configs
scripts/                         # Reproducible public-data and validation tools
environments/gate_runner/
├── gate_runner.py               # Verifiers entrypoint
├── gate_runner_core/
│   ├── config.py                # Strict strategy schema and parser
│   ├── market.py                # Panels, PIT joins, features, and task builder
│   ├── scoring.py               # Backtester, DSR, CSCV/PBO, shaped reward
│   ├── rubric.py                # Group reward and logged metrics
│   └── data/                    # Pinned public snapshots and source manifest
├── tests/test_gate_runner.py    # Deterministic, PIT, carry, and signature tests
├── DATA_PROVENANCE.md           # Source, rights, checksums, and limitations
└── README.md                    # Full environment contract
```

Credentials belong in environment variables or a machine-local
`configs/endpoints.toml`, which is ignored by Git. Never place credentials in
the example configs.

## Current status

- v0.3 code, bundled data, PIT joins, carry accounting, carry ranking, and
  forward validation are implemented.
- The deterministic suite passes and the public artifacts rebuild
  byte-for-byte.
- The earlier v0.2 Qwen 4B preflight used the spot-only profile and should not
  be combined with v0.3 carry-aware results.
- The next milestone is a fresh multi-model preflight and baseline report on
  `ecb_fx_carry`.

The deterministic synthetic panel is a test fixture, not evidence of
real-world investment performance. ECB reference rates are informational rather
than dealing quotes; policy/base rates are not directly investable funding
rates. Gate Runner omits forward curves, cross-currency basis, collateral
conventions, capital controls, market impact, and intraday execution.

## Method references

- David H. Bailey and Marcos López de Prado,
  [The Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
- David H. Bailey, Jonathan M. Borwein, Marcos López de Prado, and Qiji Jim Zhu,
  [The Probability of Backtest Overfitting](https://escholarship.org/uc/item/4w1110bb)
- R. Tyrrell Rockafellar and Stanislav Uryasev,
  [Optimization of Conditional Value-at-Risk](https://doi.org/10.21314/JOR.2000.038)

Gate Runner is an evaluation and research environment. It does not provide
investment advice.

## License

Code is licensed under the [Apache License 2.0](LICENSE).
