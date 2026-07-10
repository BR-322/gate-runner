# Gate Runner

Gate Runner is a [Prime Intellect](https://www.primeintellect.ai/)
[Verifiers](https://github.com/PrimeIntellect-ai/verifiers) environment for
training and evaluating models on honest quantitative-strategy design.

The model receives a point-in-time market brief and returns one strategy as
strict JSON. Gate Runner evaluates that proposal on hidden sequential windows,
charges trading costs, deflates Sharpe for repeated trials, and penalizes the
probability of backtest overfitting. The reward is designed to make blind
parameter search less attractive than simple strategies that generalize.

## Why this environment exists

Backtests are easy to optimize after the fact. A model can generate many
plausible strategies, keep the lucky winner, and appear skilled even when it has
only searched noise. Gate Runner makes the extent of that search part of the
grade:

- each rollout in an episode group counts as another trial;
- Deflated Sharpe Ratio adjusts for selection and non-normal returns;
- combinatorially symmetric cross-validation estimates PBO;
- a strict point-in-time boundary keeps future data out of the prompt; and
- costs and a small complexity penalty discourage fragile, high-turnover rules.

The training reward remains continuous, while a separate `passed` metric
requires both `DSR > 0.90` and `PBO < 0.25`.

## Current status

- The environment, strategy schema, synthetic benchmark panel, bundled ECB FX
  profile, walk-forward backtester, DSR/PBO scorer, and deterministic regression
  suite are implemented.
- The signature test ranks a noisy short-horizon breakout below a parsimonious
  momentum strategy.
- Local Prime installation and a 5-example × 3-rollout live smoke evaluation
  succeed.
- The next release milestone is the multi-model baseline report on the public
  ECB profile.

The synthetic panel is a deterministic test fixture, not evidence of real-world
investment performance.

## Quickstart

Prerequisites: Python 3.12+, `uv`, and the Prime CLI authenticated with
`prime login`.

Install and evaluate the public Hub environment:

```bash
prime env install br-322/gate-runner --plain
prime eval run br-322/gate-runner
```

Those commands use the deterministic synthetic panel. To run a real-data
preflight on the bundled ECB profile:

```bash
prime eval run br-322/gate-runner \
  -a '{"dataset":"ecb_fx","eval_examples":200}' \
  -n 20 -r 3
```

For development from this checkout, install the local environment instead:

```bash
prime env install gate-runner --plain
prime eval run gate-runner
```

Run the deterministic tests:

```bash
uv run --project environments/gate_runner --group dev pytest -q
```

The default evaluation uses grouped rollouts because DSR trial accounting and
PBO operate across candidate strategies from the same prompt.

## Repository layout

```text
environments/gate_runner/
├── gate_runner.py             # Verifiers entrypoint
├── gate_runner_core/
│   ├── config.py              # Strict strategy schema and parser
│   ├── market.py              # Market panels and point-in-time task builder
│   ├── data/                  # Pinned source-standard ECB API snapshot
│   ├── rubric.py              # Group reward and logged metrics
│   └── scoring.py             # Backtester, DSR, CSCV/PBO, shaped reward
├── tests/test_gate_runner.py  # Deterministic and signature tests
└── README.md                  # Full environment contract
```

Prime Lab model-family examples live under `configs/`. They contain model and
environment identifiers only; credentials belong in environment variables or a
machine-local `configs/endpoints.toml`, which is ignored by Git.

To define a reusable endpoint alias, copy `configs/endpoints.example.toml` to
`configs/endpoints.toml`, then edit the environment-variable name or endpoint
metadata as needed. Never put a credential value in either file.

## Data policy

The default environment generates a deterministic 22-asset panel. The optional
`ecb_fx` profile uses a pinned, source-standard ECB API export containing 29
daily euro reference-rate series from 2009 through 2024. The package filters
holiday rows at runtime without changing source observation values and clearly
labels its generated 5 bps spread proxy.

The ECB source snapshot remains subject to the ECB's reuse terms; Gate Runner's
Apache-2.0 license covers the project code and does not relicense the data. See
the [data provenance record](environments/gate_runner/DATA_PROVENANCE.md) for the
query, checksums, exact processing, attribution, and limitations.

`data_path` can also load a caller-owned rectangular CSV with
`date,symbol,close` plus optional `high,low,volume` columns. Put private or
experimental downloads in `data-local/`; that directory is ignored.

See the [environment documentation](environments/gate_runner/README.md) for the
action schema, reward definition, metrics, data contract, and limitations.

## Method references

- David H. Bailey and Marcos López de Prado,
  [The Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
- David H. Bailey, Jonathan M. Borwein, Marcos López de Prado, and Qiji Jim Zhu,
  [The Probability of Backtest Overfitting](https://escholarship.org/uc/item/4w1110bb)

Gate Runner is an evaluation and research environment. It does not provide
investment advice.

## License

Licensed under the [Apache License 2.0](LICENSE).
