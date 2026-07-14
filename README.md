# Gate Runner

Gate Runner is a platform-neutral benchmark for training and evaluating models
on honest quantitative-strategy design. Its deterministic core can be used as
a Python API or JSONL command-line evaluator. A maintained
[Prime Intellect](https://www.primeintellect.ai/)
[Verifiers](https://github.com/PrimeIntellect-ai/verifiers) adapter supports
hosted evaluation and RL without defining the benchmark itself.

A model receives a point-in-time market brief and returns one strict JSON
strategy. Gate Runner evaluates it across hidden sequential windows, charges
trading and funding costs, deflates Sharpe for repeated trials, and measures
weak-regime and downside behavior. The objective is not to find the prettiest
backtest; it is to reward strategies that remain credible after the search
process and economic frictions are acknowledged.

## See it work

From this checkout, with Python 3.12+ and `uv` installed:

```bash
uv run --project environments/gate_runner gate-runner demo
```

The command builds a real synthetic task and scores two strategies together:

```text
Gate Runner demo
Dataset: deterministic synthetic panel (seed=17)
Grouped trials: 2
1. 120-day momentum
   reward=0.4463 passed=0 dsr=0.8905 ...
2. 10-day concentrated breakout
   reward=0.3333 passed=0 dsr=0.6063 ...
```

There is no model call, account, or canned score in this path. The demo invokes
the same task builder and grouped evaluator used by every supported interface.

## What a strategy looks like

A completion is exactly one JSON object:

```json
{
  "entry": {
    "type": "momentum_threshold",
    "lookback_days": 120,
    "threshold": 0.02
  },
  "exit": {
    "type": "time_exit",
    "max_holding_days": 63
  },
  "universe_filter": {
    "rank_by": "relative_strength_252d",
    "side": "top",
    "k": 5
  },
  "sizing": {
    "method": "equal_weight",
    "max_positions": 5
  }
}
```

The schema supports:

- momentum-threshold, mean-reversion-z-score, and channel-breakout entries;
- fixed-stop, trailing-stop, and time exits;
- relative-strength or long-EUR-carry universe ranking; and
- equal-weight, inverse-volatility, or constrained fractional-Kelly sizing with
  at most five concurrent positions.

Unknown keys, markdown fences, aliases, invalid primitive names, and
out-of-range values fail closed to zero reward. Every task includes the exact
schema and bounds; the implementation is in
[`config.py`](environments/gate_runner/gate_runner_core/config.py).

The current FX grammar expresses positive EUR exposure only. It does not yet
support short-EUR positions, leverage, forward-tenor choice, or market-neutral
pair portfolios. Fractional Kelly is deliberately long-only, capped, unlevered,
and allowed to retain cash.

## Why it exists

Backtests are easy to optimize after the fact. A policy can sample many
plausible strategies, retain the lucky one, and appear skilled when it has only
searched noise. Gate Runner makes the extent of that search part of the grade.

The benchmark is designed to make simple strategies that survive hidden
windows and economic frictions more attractive than blind parameter search.

## How grading works

- Every rollout in an episode group counts as another trial.
- Reward DSR uses a fixed-N dispersion floor so a policy cannot reduce its own
  deflation by emitting behaviorally duplicate trials.
- Descriptive DSR, effective rank, and pairwise return correlation expose when
  that floor is doing load-bearing work.
- Lower-tail window performance rewards stability across regimes.
- Normalized expected shortfall catches concentrated daily losses.
- Trading costs and FX funding carry enter realized returns.
- CSCV/PBO describes group selection risk without becoming a training target.
- Strict point-in-time boundaries keep future observations out of prompts.
- Bounded complexity and exposure-aware activity rules discourage fragile or
  trivial policies.

A hard pass requires all of the following:

- `DSR > 0.90`;
- `window_tail_score > -0.50`;
- `expected_shortfall_ratio < 5.0`;
- at least 10% exposure-weighted activity across the grading horizon;
- meaningful activity in at least four of eight grading windows; and
- at least 25% mean gross exposure on meaningfully active sessions.

The shaped reward is dense on both sides of that boundary. A failing strategy
improves only by closing its worst normalized violation; crossing every gate
creates a hard reward jump; a passing strategy continues improving toward a
robust margin.

PBO is logged as a group diagnostic and does not affect reward or pass status.
The evaluator also logs the signed reward-minus-diagnostic DSR gap, behavioral
effective rank, mean absolute return correlation, gross exposure, cash,
concentration, and realized volatility.
In the Prime adapter, the evaluation headline is the logged `passed` metric,
not Verifiers' generic `pass@k` threshold over the shaped reward. See the
[environment contract](environments/gate_runner/README.md) for the exact reward
definition and complete metric list.

## Choose an interface

| Interface | Use it when |
| --- | --- |
| Python API | Embedding Gate Runner in an application, evaluator, or training loop |
| JSONL CLI | Connecting any local model, hosted API, or batch inference system |
| Prime adapter | Running hosted evaluation or reinforcement learning on Prime |

All three call the same grouped evaluator.

### Python API

Save the strategy above as `strategy.json`, then:

```python
from pathlib import Path

from gate_runner_core import GateRunnerBenchmark

benchmark = GateRunnerBenchmark(dataset="ecb_fx_carry")
_, eval_tasks = benchmark.build_tasks(train_examples=1, eval_examples=24)
strategy_json = Path("strategy.json").read_text()

results = benchmark.evaluate_group(
    completions=[strategy_json],
    as_of_index=eval_tasks[0].as_of_index,
)
```

Each result contains its reward, complete metric record, and any parse error.
Pass every rollout sampled for one task in a single `completions` list.

### JSONL CLI

Generate portable tasks:

```bash
uv run --project environments/gate_runner gate-runner tasks \
  --dataset ecb_fx_carry --split eval --examples 24 \
  --output tasks.jsonl
```

Have the inference system add one non-empty `completions` string array to each
task record, then score the groups:

```bash
uv run --project environments/gate_runner gate-runner score \
  --dataset ecb_fx_carry --input completions.jsonl \
  --output results.jsonl
```

The model provider and inference stack never enter the scoring process. The
portable record format is documented in [the adapter contract](docs/ADAPTERS.md).

### Prime Intellect adapter

With the Prime CLI authenticated, install the public Hub environment:

```bash
prime env install br-322/gate-runner --plain
prime eval run br-322/gate-runner
```

Run a 20-example carry-aware preflight with:

```bash
prime eval run br-322/gate-runner \
  -a '{"dataset":"ecb_fx_carry","eval_examples":200}' \
  -n 20 -r 3
```

Inspect reward distribution, hard-pass metrics, errors, and samples before
raising `-n` to `200`. For local adapter development:

```bash
prime env install gate-runner --plain
prime eval run gate-runner
uv run --project environments/gate_runner --group dev pytest -q
```

## Architecture

```text
market data ──> task records ──> grouped evaluator ──> reward + metrics
                    │                    │
                    ├── JSONL CLI        └── Prime/Verifiers adapter
                    └── Python API           (same evaluator instance)
```

`gate_runner_core` owns data loading, PIT joins, task generation, strategy
parsing, backtesting, DSR/PBO, reward shaping, and result records. It imports
neither Verifiers nor Hugging Face Datasets. Platform code may translate tasks
and results, but it must not reimplement the grader.

Grouped scoring is part of the public contract: every completion sampled for
one cutoff must be submitted together so DSR sees the correct trial count and
PBO sees the same candidate group. Scoring completions independently is not
Gate Runner-compatible. The full invariants are in
[`docs/ADAPTERS.md`](docs/ADAPTERS.md).

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

## Reproducibility and validation

The public package contains:

- the source-standard ECB spot snapshot;
- a normalized BIS/BNB short-rate snapshot with observation date, availability
  date, source, rate type, and checksums;
- a machine-readable source manifest; and
- untouched Bank of England GBP/USD spot/forward observations used only to
  validate the CIP proxy.

The data rebuild and fixed-baseline scripts are under [`scripts/`](scripts/).
The v0.5 sizing-control results and exact per-cutoff records are in
[`reports/`](reports/). On the pinned common
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
promise of bit-for-bit v0.2 behavior: the current prompt, metrics,
documentation, and package version are v0.5. For an exact historical checkout,
use commit [`99ac503`](https://github.com/BR-322/gate-runner/tree/99ac503).

New training and evaluation configs use `ecb_fx_carry`; `ecb_fx` should be used
as an ablation or historical comparison, not as the default baseline.

## Repository layout

```text
configs/                         # Prime evaluation and RL model-family configs
docs/                            # Platform adapter contract
reports/                         # Reproducible fixed-baseline results
scripts/                         # Reproducible public-data and validation tools
environments/gate_runner/
├── gate_runner.py               # Stable Prime Hub entrypoint
├── gate_runner_cli.py           # JSONL CLI and zero-auth demo
├── gate_runner_adapters/
│   └── prime.py                 # Verifiers/Hugging Face translation layer
├── gate_runner_core/
│   ├── benchmark.py             # Public benchmark API
│   ├── config.py                # Strict strategy schema and parser
│   ├── evaluator.py             # Grouped completion evaluation
│   ├── examples.py              # Valid strategies used by the CLI demo
│   ├── market.py                # Panels, PIT joins, and market features
│   ├── scoring.py               # Backtester, DSR, CSCV/PBO, shaped reward
│   ├── tasks.py                 # Task records and embargoed splits
│   └── data/                    # Pinned public snapshots and source manifest
├── tests/test_gate_runner.py    # Core, parity, CLI, PIT, carry, signature tests
├── DATA_PROVENANCE.md           # Source, rights, checksums, and limitations
└── README.md                    # Full environment contract
```

Credentials belong in environment variables or a machine-local
`configs/endpoints.toml`, which is ignored by Git. Never place credentials in
the example configs.

## Current status

- v0.5 adds exposure-aware activity, behavioral-diversity diagnostics,
  inverse-volatility sizing, and constrained fractional Kelly.
- The benchmark core remains separate from its Prime adapter and supports
  Python, JSONL, and Prime interfaces.
- Adapter parity tests require every interface to produce exactly the same
  rewards, metrics, and errors.
- Bundled data, PIT joins, carry accounting, carry ranking, forward validation,
  and byte-for-byte artifact rebuilds are implemented.
- The earlier v0.2 Qwen 4B preflight used the spot-only profile and should not
  be combined with v0.5 carry-aware results.
- The fixed sizing matrix is complete; the next milestone is a fresh
  multi-model preflight on `ecb_fx_carry` before any training run.

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
