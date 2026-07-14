# Gate Runner v0.5 fixed sizing baseline

This is a deterministic control matrix, not a model evaluation or evidence of
tradable performance. It checks whether the v0.5 sizing grammar and exposure
guards behave coherently before a live-model preflight.

## Protocol

- Data profiles: `synthetic`, `ecb_fx`, and `ecb_fx_carry`.
- Seeds: 17, 23, and 41.
- Held-out cutoffs: eight per seed and profile.
- Candidates per cutoff: three fixed signal families crossed with three fixed
  sizing methods, for nine candidates scored as one group.
- Signal families: 120-day momentum, 60-day mean reversion, and 120-day
  channel breakout.
- Sizing methods: equal weight, 126-day inverse volatility, and 126-day 0.25
  fractional Kelly.
- Total observations: 648.

All nine candidates share the same cutoff and are submitted together. Trial
count, DSR dispersion flooring, behavioral effective rank, and PBO therefore
see a fixed comparison surface at every cutoff. The spot and carry-aware ECB
profiles also use matched dates, making their difference a carry ablation.

The exact protocol, aggregate records, and per-candidate observations are in
[`baseline_matrix_v0_5.json`](baseline_matrix_v0_5.json). Reproduce it with:

```bash
uv run --project environments/gate_runner \
  python scripts/run_baseline_matrix.py
```

## Aggregate sizing results

Values are means over 72 observations per row.

| Profile | Sizing | Pass rate | Reward | DSR | Diagnostic DSR | Window tail | ES ratio | Exposure | Cash | Effective positions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ecb_fx` | equal weight | 1.4% | 0.165 | 0.173 | 0.330 | -1.110 | 3.072 | 0.790 | 0.210 | 3.166 |
| `ecb_fx` | fractional Kelly | 1.4% | 0.153 | 0.148 | 0.296 | -0.696 | 1.817 | 0.419 | 0.581 | 2.240 |
| `ecb_fx` | inverse volatility | 1.4% | 0.152 | 0.140 | 0.280 | -0.946 | 2.168 | 0.725 | 0.275 | 2.944 |
| `ecb_fx_carry` | equal weight | 0.0% | 0.128 | 0.086 | 0.210 | -1.382 | 3.098 | 0.791 | 0.209 | 3.141 |
| `ecb_fx_carry` | fractional Kelly | 0.0% | 0.119 | 0.071 | 0.185 | -0.774 | 1.726 | 0.393 | 0.607 | 2.088 |
| `ecb_fx_carry` | inverse volatility | 0.0% | 0.123 | 0.074 | 0.190 | -1.134 | 2.185 | 0.725 | 0.275 | 2.923 |
| `synthetic` | equal weight | 1.4% | 0.170 | 0.177 | 0.381 | -0.867 | 1.605 | 0.871 | 0.129 | 3.486 |
| `synthetic` | fractional Kelly | 0.0% | 0.164 | 0.191 | 0.410 | -0.296 | 0.913 | 0.485 | 0.515 | 2.436 |
| `synthetic` | inverse volatility | 0.0% | 0.163 | 0.177 | 0.374 | -0.783 | 1.480 | 0.817 | 0.183 | 3.438 |

Only four of 648 observations passed. All three sizing variants passed for the
same ECB spot breakout at the 2022-09-23 cutoff; one synthetic equal-weight
breakout also passed. No fixed candidate passed on the carry-aware profile.

## Gate behavior

DSR was the binding gate in 644 of 648 observations. The window-tail gate also
failed frequently, while normalized expected shortfall was rarely binding.
This is consistent with a nine-trial DSR correction applied to deliberately
simple, unselected controls; it is not by itself evidence that the DSR boundary
should be loosened.

The Kelly controls improved the average tail and expected-shortfall metrics,
but did so with materially more cash. Across profiles their average exposure
was 0.39-0.49, versus 0.79-0.87 for equal weight. The exposure contract caught
the extreme cases: fractional Kelly failed the 10% exposure-weighted activity
floor 52 times and the 25% mean active gross-exposure floor 30 times. It also
failed the active-window requirement in 15 cases. Tiny nonzero allocations
therefore do not receive full activity credit.

Inverse volatility remained much closer to equal weight in exposure while
generally reducing expected shortfall. Fractional Kelly did not consistently
outperform inverse volatility on reward or DSR. Its strongest apparent safety
advantage is partly a cash-allocation effect, so it remains experimental rather
than becoming the default.

Behavioral effective rank averaged 2.56 on synthetic data, 3.52 on ECB spot,
and 3.60 on ECB carry, out of nine submitted candidates. Mean absolute
pairwise return correlation was 0.64, 0.41, and 0.40 respectively. The fixed
matrix is therefore much less diverse behaviorally than its nine distinct JSON
configs imply. The reward DSR's fixed-N dispersion floor was load-bearing: its
mean value was 0.11-0.22 below the descriptive DSR, and the signed gap is now a
first-class metric.

## Decision

- Keep equal weight as the default and backward-compatible control.
- Keep inverse volatility as the robust risk-allocation alternative.
- Keep fractional Kelly long-only, capped, unlevered, and explicitly
  experimental.
- Retain the provisional exposure gates for the live-model preflight.
- Do not recalibrate the DSR or window-tail pass boundaries from this small
  hand-authored matrix. Revisit them only after the fixed multi-model preflight
  provides a broader, naturally generated candidate distribution.
