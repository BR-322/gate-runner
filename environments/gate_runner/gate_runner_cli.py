from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, TextIO

from gate_runner_core.benchmark import DATASETS, GateRunnerBenchmark


def _add_benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=DATASETS, default="synthetic")
    parser.add_argument("--data-path")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--window-days", type=int, default=42)


def _benchmark(args: argparse.Namespace) -> GateRunnerBenchmark:
    return GateRunnerBenchmark(
        seed=args.seed,
        windows=args.windows,
        window_days=args.window_days,
        dataset=args.dataset,
        data_path=args.data_path,
    )


def _write_jsonl(rows: Sequence[dict[str, object]], output: str) -> None:
    handle: TextIO
    should_close = output != "-"
    if should_close:
        handle = Path(output).open("w", encoding="utf-8", newline="")
    else:
        handle = sys.stdout
    try:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    finally:
        if should_close:
            handle.close()


def _read_jsonl(path: str) -> list[dict[str, object]]:
    if path == "-":
        lines = sys.stdin.readlines()
    else:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number}: record must be a JSON object")
        rows.append(value)
    return rows


def _generate_tasks(args: argparse.Namespace) -> int:
    benchmark = _benchmark(args)
    train_examples = args.examples if args.split == "train" else 1
    eval_examples = args.examples if args.split == "eval" else 1
    train_tasks, eval_tasks = benchmark.build_tasks(
        train_examples=train_examples,
        eval_examples=eval_examples,
    )
    tasks = train_tasks if args.split == "train" else eval_tasks
    rows = [
        task.to_dict(task_id=f"{args.split}-{index:06d}")
        for index, task in enumerate(tasks)
    ]
    _write_jsonl(rows, args.output)
    return 0


def _score_groups(args: argparse.Namespace) -> int:
    benchmark = _benchmark(args)
    inputs = _read_jsonl(args.input)
    outputs: list[dict[str, object]] = []
    for line_number, row in enumerate(inputs, start=1):
        info = row.get("info")
        completions = row.get("completions")
        if not isinstance(info, dict) or not isinstance(info.get("as_of_index"), int):
            raise ValueError(
                f"line {line_number}: info.as_of_index must be an integer"
            )
        if (
            not isinstance(completions, list)
            or not completions
            or not all(isinstance(value, str) for value in completions)
        ):
            raise ValueError(
                f"line {line_number}: completions must be a non-empty string list"
            )
        records = benchmark.evaluate_group(
            completions=completions,
            as_of_index=info["as_of_index"],
        )
        output: dict[str, object] = {
            "info": info,
            "results": [record.to_dict() for record in records],
        }
        if "task_id" in row:
            output["task_id"] = row["task_id"]
        outputs.append(output)
    _write_jsonl(outputs, args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gate-runner",
        description="Platform-neutral Gate Runner task and grouped-scoring CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    tasks = subparsers.add_parser("tasks", help="Generate point-in-time tasks")
    _add_benchmark_arguments(tasks)
    tasks.add_argument("--split", choices=("train", "eval"), default="eval")
    tasks.add_argument("--examples", type=int, default=24)
    tasks.add_argument("--output", default="-")
    tasks.set_defaults(handler=_generate_tasks)

    score = subparsers.add_parser(
        "score",
        help="Score JSONL records containing grouped completions",
    )
    _add_benchmark_arguments(score)
    score.add_argument("--input", required=True)
    score.add_argument("--output", default="-")
    score.set_defaults(handler=_score_groups)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "examples", 1) < 1:
        parser.error("--examples must be positive")
    try:
        return int(args.handler(args))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
