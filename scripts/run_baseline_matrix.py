#!/usr/bin/env python3
"""Run the fixed Gate Runner v0.5 sizing baseline matrix."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median
from typing import Iterable

from gate_runner_core.benchmark import DATASETS, GateRunnerBenchmark


DEFAULT_DATASETS = ("synthetic", "ecb_fx", "ecb_fx_carry")
DEFAULT_SEEDS = (17, 23, 41)
DEFAULT_EVAL_EXAMPLES = 8

SIZING_CONFIGS = (
    (
        "equal_weight",
        {"method": "equal_weight", "max_positions": 5},
    ),
    (
        "inverse_volatility",
        {
            "method": "inverse_volatility",
            "max_positions": 5,
            "lookback_days": 126,
            "max_weight": 0.50,
        },
    ),
    (
        "fractional_kelly",
        {
            "method": "fractional_kelly",
            "max_positions": 5,
            "lookback_days": 126,
            "fraction": 0.25,
            "max_weight": 0.35,
        },
    ),
)

STRATEGY_FAMILIES = (
    (
        "momentum",
        {
            "entry": {
                "type": "momentum_threshold",
                "lookback_days": 120,
                "threshold": 0.02,
            },
            "exit": {"type": "time_exit", "max_holding_days": 63},
            "universe_filter": {
                "rank_by": "relative_strength_252d",
                "side": "top",
                "k": 5,
            },
        },
    ),
    (
        "mean_reversion",
        {
            "entry": {
                "type": "mean_reversion_zscore",
                "lookback_days": 60,
                "entry_z": 1.25,
            },
            "exit": {"type": "time_exit", "max_holding_days": 20},
            "universe_filter": {
                "rank_by": "relative_strength_252d",
                "side": "bottom",
                "k": 5,
            },
        },
    ),
    (
        "breakout",
        {
            "entry": {
                "type": "channel_breakout",
                "lookback_days": 120,
                "buffer_pct": 0.005,
                "confirmation_days": 2,
            },
            "exit": {"type": "trailing_stop", "trail_pct": 0.10},
            "universe_filter": {
                "rank_by": "relative_strength_252d",
                "side": "top",
                "k": 5,
            },
        },
    ),
)

SUMMARY_METRICS = (
    "reward",
    "passed",
    "raw_sharpe",
    "dsr",
    "diagnostic_dsr",
    "reward_minus_diagnostic_dsr",
    "window_tail_score",
    "expected_shortfall_ratio",
    "active_fraction",
    "active_session_fraction",
    "mean_active_gross_exposure",
    "cash_fraction",
    "turnover",
    "effective_position_count",
    "realized_volatility",
    "behavioral_effective_rank",
    "behavioral_effective_rank_ratio",
    "mean_pairwise_absolute_correlation",
    "pbo",
)


def _candidates() -> tuple[dict[str, object], ...]:
    candidates: list[dict[str, object]] = []
    for family, strategy in STRATEGY_FAMILIES:
        for sizing_method, sizing in SIZING_CONFIGS:
            payload = json.loads(json.dumps(strategy))
            payload["sizing"] = sizing
            candidates.append(
                {
                    "family": family,
                    "sizing_method": sizing_method,
                    "payload": payload,
                    "completion": json.dumps(
                        payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )
    return tuple(candidates)


def _summarize(
    rows: Iterable[dict[str, object]],
    keys: tuple[str, ...],
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)

    summaries: list[dict[str, object]] = []
    for group_key, group_rows in sorted(groups.items()):
        summary = dict(zip(keys, group_key))
        summary["observations"] = len(group_rows)
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in group_rows]
            summary[f"mean_{metric}"] = fmean(values)
            summary[f"median_{metric}"] = median(values)
        summaries.append(summary)
    return summaries


def _print_sizing_summary(summaries: list[dict[str, object]]) -> None:
    print(
        "dataset          sizing               n  pass%  reward    dsr  diag_dsr  "
        "tail   es_ratio  exposure  cash  eff_pos"
    )
    for row in summaries:
        print(
            f"{str(row['dataset']):16} "
            f"{str(row['sizing_method']):20} "
            f"{int(row['observations']):3d} "
            f"{100.0 * float(row['mean_passed']):6.1f} "
            f"{float(row['mean_reward']):7.3f} "
            f"{float(row['mean_dsr']):6.3f} "
            f"{float(row['mean_diagnostic_dsr']):9.3f} "
            f"{float(row['mean_window_tail_score']):6.3f} "
            f"{float(row['mean_expected_shortfall_ratio']):9.3f} "
            f"{float(row['mean_active_fraction']):8.3f} "
            f"{float(row['mean_cash_fraction']):5.3f} "
            f"{float(row['mean_effective_position_count']):7.3f}"
        )


def run_matrix(
    datasets: tuple[str, ...],
    seeds: tuple[int, ...],
    eval_examples: int,
) -> dict[str, object]:
    candidates = _candidates()
    completions = [str(candidate["completion"]) for candidate in candidates]
    rows: list[dict[str, object]] = []

    for dataset in datasets:
        for seed in seeds:
            benchmark = GateRunnerBenchmark(dataset=dataset, seed=seed)
            _, eval_tasks = benchmark.build_tasks(
                train_examples=1,
                eval_examples=eval_examples,
            )
            for task in eval_tasks:
                records = benchmark.evaluate_group(
                    completions=completions,
                    as_of_index=task.as_of_index,
                )
                for candidate, record in zip(candidates, records):
                    row: dict[str, object] = {
                        "dataset": dataset,
                        "seed": seed,
                        "as_of_index": task.as_of_index,
                        "as_of_date": task.as_of_date,
                        "family": candidate["family"],
                        "sizing_method": candidate["sizing_method"],
                        "error": record.error,
                    }
                    row.update(record.score.metrics())
                    rows.append(row)

    by_sizing = _summarize(rows, ("dataset", "sizing_method"))
    by_family_and_sizing = _summarize(
        rows,
        ("dataset", "family", "sizing_method"),
    )
    return {
        "protocol": {
            "name": "gate-runner-v0.5-fixed-sizing-baseline",
            "datasets": list(datasets),
            "seeds": list(seeds),
            "eval_examples_per_seed": eval_examples,
            "candidate_count_per_group": len(candidates),
            "strategy_families": {
                name: config for name, config in STRATEGY_FAMILIES
            },
            "sizing_configs": {
                name: config for name, config in SIZING_CONFIGS
            },
            "grouping": (
                "All nine family-by-sizing candidates are scored together at each "
                "cutoff, so trial count and group diagnostics are fixed and comparable."
            ),
        },
        "by_sizing": by_sizing,
        "by_family_and_sizing": by_family_and_sizing,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DEFAULT_DATASETS),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--eval-examples",
        type=int,
        default=DEFAULT_EVAL_EXAMPLES,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/baseline_matrix_v0_5.json"),
    )
    args = parser.parse_args()
    if args.eval_examples < 1:
        parser.error("--eval-examples must be positive")

    result = run_matrix(
        datasets=tuple(args.datasets),
        seeds=tuple(args.seeds),
        eval_examples=args.eval_examples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _print_sizing_summary(result["by_sizing"])
    print(f"\nWrote {len(result['rows'])} observations to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
