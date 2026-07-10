# Gate Runner

Gate Runner is a single-turn strategy-design environment whose grader penalizes
the usual backtest search game. A model receives a point-in-time market brief
and must return one strict JSON strategy config. The environment then evaluates
that config on hidden sequential windows with trading costs, deflates Sharpe by
the number of sampled configs in the rollout group, and applies a CSCV/PBO
penalty.

## Environment contract

- **Environment ID:** `gate-runner`
- **Task:** single-turn JSON generation
- **Default panel:** deterministic synthetic 22-asset daily panel, used so the
  package is self-contained and redistributable
- **Point-in-time rule:** prompt features use data strictly before the episode
  cutoff; all grading windows begin at or after the cutoff
- **Execution:** close-to-close signals, equal-weight long-only positions,
  at most five concurrent positions
- **Costs:** 10 bps per traded side plus a per-symbol spread proxy
- **Walk-forward horizon:** eight sequential 42-session windows by default

Malformed JSON, markdown-wrapped JSON, unknown keys, wrong primitive names, and
out-of-range parameters receive exactly zero reward.

## Reward

For every schema-valid config:

```text
reward = 0.70*tanh(DSR) - 0.20*PBO - 0.05*complexity + 0.10*validity
```

- **DSR** is the Deflated Sharpe Ratio probability. Its expected-max-Sharpe
  benchmark uses every sampled rollout in the same episode group as a trial,
  including invalid attempts, and the valid trials' Sharpe dispersion. A null
  standard error is used when fewer than two valid Sharpe estimates exist.
- **PBO** uses combinatorially symmetric cross-validation across the sequential
  windows and the valid configs in that episode group. It is `0` when fewer
  than two valid configs exist because cross-strategy selection risk is then
  undefined.
- **complexity** is the active numeric parameter count normalized by eight.
- **passed** is a headline metric, not a separate reward: `DSR > 0.90` and
  `PBO < 0.25`.

The rubric also logs `validity`, `raw_sharpe`, `dsr`, `pbo`, `complexity`,
`parameter_count`, `trial_count`, `passed`, and `turnover`.

Verifiers' generic `pass@k` display thresholds the continuous shaped reward.
Because Gate Runner's binary gate is a conjunction of DSR and PBO conditions,
no single shaped-reward threshold is equivalent. Use the logged `passed` metric
as the evaluation headline, not generic `pass@k`.

## Quickstart

From the Prime Lab workspace root:

```bash
prime env install gate-runner --plain
prime eval run gate-runner
```

The environment defaults in `pyproject.toml` run five examples with three
rollouts each. Group scoring is required for trial-count deflation and PBO, so
do not use independent scoring for representative results.

Run the deterministic test suite with:

```bash
uv run --project environments/gate_runner --group dev pytest -q
```

## Environment arguments

| Argument | Default | Meaning |
| --- | ---: | --- |
| `seed` | `17` | Task sampling and synthetic panel seed |
| `train_examples` | `64` | Number of training cutoffs |
| `eval_examples` | `24` | Number of held-out eval cutoffs |
| `windows` | `8` | Even number of sequential CSCV windows, at least four |
| `window_days` | `42` | Sessions per hidden window, at least 20 |
| `data_path` | `null` | Optional caller-owned long-form market CSV |

Example with local data:

```bash
prime eval run gate-runner \
  -a '{"data_path":"/absolute/path/market.csv"}'
```

The CSV must be a complete rectangular panel with at least 1,600 dates and these
columns:

```text
date,symbol,close
```

Optional `high`, `low`, and `volume` columns improve the spread proxy. Missing
values and duplicate date/symbol rows fail early.

Training and evaluation cutoffs are sampled from chronological regions separated
by a full grading-horizon embargo, so no training episode's hidden return window
overlaps an evaluation episode's hidden return window.

## Signature test

`tests/test_gate_runner.py` fixes the benchmark seed and cutoff, then compares a
parsimonious 120-day momentum config with a noisy 10-day single-name breakout.
The honest config must have both higher out-of-sample Sharpe and a shaped reward
at least 0.10 higher. This is the environment's regression guard against losing
its defining behavior.

## Data status and limitations

No third-party market rows are bundled. The default synthetic panel makes the
core scoring behavior deterministic and the package self-contained, but it is
not a substitute for a public-market baseline. Any bundled market dataset must
have explicit redistribution terms, source attribution, and reproducible
provenance.

This compact backtester intentionally omits shorts, corporate actions, tax
effects, borrow, and intraday execution. Results are environment scores, not
investment advice.

## Method references

- Bailey and López de Prado, [The Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
- Bailey, Borwein, López de Prado, and Zhu, [The Probability of Backtest Overfitting](https://escholarship.org/uc/item/4w1110bb)
