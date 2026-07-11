# Gate Runner

Gate Runner is a single-turn strategy-design environment whose grader penalizes
the usual backtest search game. A model receives a point-in-time market brief
and must return one strict JSON strategy config. The environment then evaluates
that config on hidden sequential windows with trading costs, deflates Sharpe by
the number of sampled configs in the rollout group, and rewards lower-tail
cross-window performance. CSCV/PBO is reported as a selection-process
diagnostic and does not affect training reward or pass status.

## Environment contract

- **Environment ID:** `gate-runner`
- **Task:** single-turn JSON generation
- **Default panel:** deterministic synthetic 22-asset daily panel, used so the
  package is self-contained and redistributable
- **Public real-data panels:** `ecb_fx` is a 29-pair ECB spot-only ablation;
  `ecb_fx_carry` is the recommended 28-pair profile with public PIT reference
  short rates and carry-aware returns, 2009-2024
- **Point-in-time rule:** prompt features use data strictly before the episode
  cutoff; all grading windows begin at or after the cutoff
- **Execution:** close-to-close signals, equal-weight positive positions, at
  most five concurrent positions. In EURXXX, positive means long EUR funded in
  XXX; the current action grammar does not express short-EUR positions.
- **Costs:** 10 bps per traded side plus a per-symbol spread proxy
- **Walk-forward horizon:** eight sequential 42-session windows by default
- **Activity gate:** at least 10% active sessions across at least four grading
  windows

Malformed JSON, markdown-wrapped JSON, unknown keys, wrong primitive names, and
out-of-range parameters receive exactly zero reward.

`universe_filter.rank_by` can be `relative_strength_252d` on every panel or
`long_eur_carry` on the carry-aware panel. This makes the rate differential an
actionable ranking signal rather than prompt-only context. Selecting carry
ranking on a panel without reference rates produces no eligible assets and
therefore fails the activity gate.

## Reward

Malformed output receives `0`. Every schema-valid config receives a dense score
inside one of two non-overlapping tiers. Define normalized violations for the
DSR, window-tail, expected-shortfall, active-session, and active-window gates;
then:

```text
fail_proximity = 1 - max(normalized gate violations)
fail_reward = 0.10 + 0.35*fail_proximity

pass_robustness = min(
    normalized DSR margin,
    normalized window-tail margin,
    normalized expected-shortfall margin,
)
pass_reward = 0.60 + 0.30*pass_robustness

reward -= 0.01*complexity_excess
```

- **DSR** is the Deflated Sharpe Ratio probability. Its expected-max-Sharpe
  benchmark uses every sampled rollout in the same episode group as a trial,
  including invalid attempts, and the valid trials' Sharpe dispersion. A null
  standard error is used when fewer than two valid Sharpe estimates exist.
- **window_tail_score** computes each window's cost-adjusted log growth and
  divides it by an exogenous reference risk: the median underlying-asset daily
  volatility over the hidden horizon, scaled to the window length. It averages
  the weakest `max(2, ceil(25% * windows))` outcomes. Using market risk rather
  than the strategy's realized volatility prevents inactivity from shrinking
  the denominator. The provisional pass boundary is `>-0.50` reference-risk
  units and the full robustness target is `0.25`.
- **PBO** remains a diagnostic CSCV estimate across valid configs in the
  episode group. An IS winner counts as a loss only when it ranks strictly
  below the OOS median; median ranks and exact ties are neutral. Split losses
  remain attributed for analysis, but neither group PBO nor its attribution
  changes reward or pass status. PBO is `0` when fewer than two valid configs
  exist.
- **daily_expected_shortfall** is the positive mean loss over the weakest 5%
  of cost-adjusted daily returns. **expected_shortfall_ratio** divides that
  value by exogenous reference daily risk, defined as `reference_window_risk`
  divided by `sqrt(window_days)`. Ratios below `5.0` pass; `3.0` or lower
  receives the full robustness margin. This catches concentrated daily losses
  that profitable 42-session windows can otherwise conceal.
- **complexity_excess** is `0` for five active numeric parameters and `1` for
  six; its bounded penalty cannot overturn reward-tier ordering.
- **passed** requires `DSR > 0.90`, `window_tail_score > -0.50`,
  `expected_shortfall_ratio < 5.0`, active positions on at least 10% of grading
  sessions, and activity in at least four of eight grading windows. Activity is
  rewarded only until eligibility is reached, preventing an incentive for
  gratuitous exposure or turnover.

The rubric also logs `validity`, `raw_sharpe`, `dsr`, `pbo`,
`pbo_contribution`, `window_tail_score`, `reference_window_risk`,
`daily_expected_shortfall`, `expected_shortfall_ratio`, `complexity`,
`parameter_count`, `trial_count`, `passed`, `turnover`, `carry_contribution`,
`active_fraction`, and `active_windows`.

`carry_contribution` is the sum of daily portfolio return components attributable
to the carry proxy before transaction costs; it is zero for the synthetic,
caller-provided, and `ecb_fx` spot-only panels.

Verifiers' generic `pass@k` display applies its own threshold to the shaped
reward. Use the logged `passed` metric as the evaluation headline because it
records Gate Runner's explicit DSR/window-tail/expected-shortfall conjunction
directly.

## Quickstart

Install and evaluate the public Hub environment:

```bash
prime env install br-322/gate-runner --plain
prime eval run br-322/gate-runner
```

Run the bundled real-data profile with enough held-out rows for the 200-example
baseline:

```bash
prime eval run br-322/gate-runner \
  -a '{"dataset":"ecb_fx_carry","eval_examples":200}' \
  -n 20 -r 3
```

Keep the 20-example run as a preflight. Once its reward distribution and samples
look healthy, raise `-n` to `200` for the baseline report.

For development from a Prime Lab workspace containing this source:

```bash
prime env install gate-runner --plain
prime eval run gate-runner
```

The environment defaults in `pyproject.toml` run five examples with three
rollouts each. Group scoring is required for trial-count deflation and
diagnostic PBO, so do not use independent scoring for representative results.

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
| `dataset` | `synthetic` | Built-in panel: `synthetic`, `ecb_fx`, or `ecb_fx_carry` |
| `data_path` | `null` | Optional caller-owned long-form market CSV |

Example with local data:

```bash
prime eval run gate-runner \
  -a '{"data_path":"/absolute/path/market.csv"}'
```

`data_path` is an alternative to the built-in profiles and cannot be combined
with either ECB dataset profile.

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

The deterministic synthetic panel remains the default and the signature-test
oracle. `ecb_fx_carry` is the recommended public real-data baseline, while
`ecb_fx` provides the matched spot-only ablation. The package contains the exact
standard CSV returned by the documented ECB API query plus a normalized,
manifested reference-rate snapshot built from BIS and BNB observations. SGD is
excluded from the carry profile because MAS SORA redistribution rights were not
established; it remains available in the spot profile. Runtime filtering,
availability rules, transformations, checksums, and source terms are listed in
[DATA_PROVENANCE.md](DATA_PROVENANCE.md).

For EURXXX, the carry profile uses the rate differential available on the prior
market date and accrues it over actual calendar days:

```text
carry = (i_EUR - i_XXX) / 100 * calendar_days / 365
total_return = (1 + spot_return) * (1 + carry) - 1 - costs
```

Prompts expose the EUR and foreign reference rates, annualized long-EUR carry,
and a three-month covered-interest-parity forward-points proxy. The proxy is
not called or treated as an executable forward rate. A bundled Bank of England
GBP/USD spot/forward panel is used only by `scripts/validate_cip_proxy.py`; it
never affects training, reward, or pass status. On the pinned common sample,
the proxy's forward-premium correlation is 0.93-0.96 across tenors, with a
24-28 bp annualized mean absolute error.

ECB rates are reference rates published for information, not executable dealing
quotes. They provide close-like observations but no OHLCV or transaction-cost
data, so Gate Runner uses a disclosed constant spread proxy. The 29 pairs also
share the euro as their base currency; this is an FX cross-section, not the
equity universe represented by the synthetic fixture.

ECB rates are reference rates rather than dealing quotes, and policy/base rates
are not directly investable funding rates. The compact backtester omits
short-EUR positions, forward curves, cross-currency basis, collateral
conventions, capital controls, tax, market impact, and intraday execution.
Results are environment scores, not investment advice.

## Method references

- Bailey and López de Prado, [The Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
- Bailey, Borwein, López de Prado, and Zhu, [The Probability of Backtest Overfitting](https://escholarship.org/uc/item/4w1110bb)
- Rockafellar and Uryasev, [Optimization of Conditional Value-at-Risk](https://doi.org/10.21314/JOR.2000.038)

## License

Gate Runner's code is licensed under the
[Apache License 2.0](https://github.com/BR-322/gate-runner/blob/main/LICENSE).
The bundled ECB source snapshot remains subject to the ECB reuse terms recorded
in [DATA_PROVENANCE.md](DATA_PROVENANCE.md); it is not relicensed under Apache-2.0.
