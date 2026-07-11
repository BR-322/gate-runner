# Gate Runner adapter contract

Gate Runner's benchmark semantics live in `gate_runner_core`. Adapters translate
between a platform's task/completion types and the neutral records; they do not
reimplement parsing, task sampling, backtesting, metrics, or reward shaping.

## Neutral API

```python
from gate_runner_core import GateRunnerBenchmark

benchmark = GateRunnerBenchmark(
    dataset="ecb_fx_carry",
    seed=17,
    windows=8,
    window_days=42,
)
train_tasks, eval_tasks = benchmark.build_tasks(
    train_examples=400,
    eval_examples=200,
)
records = benchmark.evaluate_group(
    completions=[candidate_1, candidate_2, candidate_3],
    as_of_index=eval_tasks[0].as_of_index,
)
```

`TaskRecord` contains the prompt, cutoff index/date, source label, and empty
reference answer. `EvaluationRecord` contains one `HonestScore` plus its parsing
error. `EvaluationRecord.to_dict()` returns JSON-compatible reward, metrics,
and error fields.

## Required invariants

An adapter is conformant only if it preserves all of these rules:

1. **Group identity.** All completions submitted to one `evaluate_group` call
   must belong to the same task cutoff.
2. **Complete trial count.** Invalid, duplicate, or empty completions still
   count as trials. Do not filter before scoring.
3. **Stable ordering.** Result `i` belongs to completion `i`.
4. **Raw completion text.** Pass the model's complete text to the core parser.
   Do not extract JSON, remove markdown, repair values, or normalize aliases.
5. **No independent scoring.** Calling `evaluate_group([completion], ...)`
   repeatedly is not equivalent to scoring the original group: it changes DSR
   and removes group PBO information.
6. **No metric substitution.** Expose `passed` as Gate Runner's hard-pass flag.
   A host platform's generic pass threshold is a separate display convention.
7. **PIT task integrity.** Preserve `as_of_index` exactly; never reconstruct it
   from prompt text or a model-provided field.
8. **Core ownership.** Platform code may serialize records and publish metrics,
   but reward math must remain in `gate_runner_core`.

## Prime/Verifiers adapter

`gate_runner_adapters.prime` is the reference adapter:

- neutral `TaskRecord` values become Hugging Face Dataset rows;
- Verifiers completion objects become raw completion strings;
- one Verifiers rollout group becomes one neutral `evaluate_group` call; and
- neutral score/error fields are written into Verifiers state for metrics.

The public `gate_runner.load_environment(...)` entrypoint delegates to this
adapter, preserving the Prime Hub interface.

The deterministic suite compares neutral and Prime execution field-for-field.
Any difference in reward, metric, error, task prompt, or cutoff fails the parity
tests.

## JSONL interface

Generate portable task records with:

```bash
gate-runner tasks --dataset ecb_fx_carry --split eval --examples 24 \
  --output tasks.jsonl
```

Each line has this shape:

```json
{
  "task_id": "eval-000000",
  "question": "...",
  "answer": "",
  "info": {
    "as_of_index": 3123,
    "as_of_date": "2021-04-15",
    "data_source": "..."
  }
}
```

The inference system adds every rollout for that task as one string array:

```json
{
  "task_id": "eval-000000",
  "info": {"as_of_index": 3123},
  "completions": ["{...}", "{...}", "not json"]
}
```

Then score the file:

```bash
gate-runner score --dataset ecb_fx_carry --input completions.jsonl \
  --output results.jsonl
```

The output retains `task_id` and `info` and adds an ordered `results` array.
This format can be produced by any local model, hosted API, batch system, or RL
orchestrator without giving that system control over benchmark math.

## Adding another platform

A new adapter normally needs only three operations:

1. construct one `GateRunnerBenchmark` with the platform's environment args;
2. translate `TaskRecord` values into the platform's task representation; and
3. collect same-task completion groups and call `evaluate_group` once per
   group.

Add a parity test using at least one valid and one invalid completion. The new
adapter must match direct neutral evaluation exactly before it is considered
supported.
