# Temporal Harbor integration

Run [harbor](https://github.com/harbor-framework/harbor) evaluation jobs durably, with each trial as
its own Temporal Activity. Published as [`temporalio-harbor`](https://pypi.org/project/temporalio-harbor/)
and imported as `temporalio.harbor`.

This package is experimental and may change in future versions.

Wrapping a whole `harbor run` in one Activity means a lost worker throws away every trial that
already finished, and the Temporal UI shows one opaque Activity for hours. With this plugin:

- A lost worker costs only the trials that were in flight. Every finished trial's result is
  already in workflow history.
- Each trial is visible in the UI, with its task, agent, and the harbor phase it is in (`agent-start`,
  `verification-start`, …) in its heartbeat.
- The job's statistics come from harbor's own code: reward and error statistics, token and cost
  totals, each dataset's metrics, and pass@k. They match what `harbor run` reports for the same job.

## Install

```bash
uv add temporalio-harbor
```

Requires Python 3.12 or later, like harbor itself. Harbor brings its own sandbox-provider
dependencies with it.

## Usage

Register `HarborPlugin` on the Client. It registers the plugin's Activities on every Worker built
from that Client and installs Temporal's pydantic data converter, which harbor's models need.

```python
from temporalio.client import Client
from temporalio.harbor import HarborPlugin
from temporalio.worker import Worker


async def main() -> None:
    client = await Client.connect("localhost:7233", plugins=[HarborPlugin()])
    worker = Worker(
        client,
        task_queue="evals",
        workflows=[EvalJob],
        # The worker's activity slots are the concurrency budget for trials.
        max_concurrent_activities=16,
    )
    await worker.run()
```

A workflow plans a harbor `JobConfig` into trials, runs them, and aggregates:

```python
import asyncio
from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from harbor.models.job.config import JobConfig
    from harbor.models.job.result import JobStats

    from temporalio.harbor import aggregate_job, execute_trial, plan_job


@workflow.defn
class EvalJob:
    @workflow.run
    async def run(self, config: JobConfig) -> JobStats:
        plan = await plan_job(config)
        results = await asyncio.gather(
            *(
                execute_trial(
                    trial,
                    retry=config.retry,
                    start_to_close_timeout=timedelta(hours=2),
                )
                for trial in plan.trials
            )
        )
        return await aggregate_job(plan, results)
```

Start it with the same `JobConfig` you would give `harbor run`:

```python
from pathlib import Path

from harbor.models.job.config import JobConfig
from temporalio.client import Client
from temporalio.harbor import HarborPlugin


async def start() -> None:
    client = await Client.connect("localhost:7233", plugins=[HarborPlugin()])
    config = JobConfig.model_validate_json(Path("job.json").read_text())
    stats = await client.execute_workflow(
        EvalJob.run, config, id=f"eval-{config.job_name}", task_queue="evals"
    )
    print(stats.evals)
```

Build the `JobConfig` outside the workflow, as above, and pass it in. `JobConfig` defaults
`job_name` from the clock, which workflow code must not read.

## What the plugin does

`plan_job(config)` resolves the job's datasets in an Activity, since that can reach a dataset
registry. It then expands every attempt of every task for every agent into a `TrialConfig`, exactly
as harbor's `Job` does. Each trial gets a name that is stable across replay, so a retried trial
re-runs in the same slot.

`execute_trial(trial, ...)` runs one trial as one Activity.

- It heartbeats every `heartbeat_interval` (30 seconds by default, set on `HarborPlugin`) for the
  whole trial, even while an agent runs silently for hours. Each heartbeat carries the trial's
  current harbor phase and elapsed time.
- It returns harbor's `TrialResult`. Rollout details (token IDs and log-probabilities) and agent
  metadata are omitted, because harbor's aggregation never reads them and they are most of a
  result's size. The complete result stays in the trial directory, where harbor writes it.
- `start_to_close_timeout` is required: trials range from seconds to hours.

`aggregate_job(plan, results)` computes harbor's `JobStats` from the results. Statistics and pass@k
are computed in the workflow. Each dataset's metrics are computed in an Activity, from rewards
alone, because a dataset's metric can be a script that harbor runs with `uv`.

## Errors and retries

A trial that runs and fails is a result, just as `harbor run` records it. It comes back with
`exception_info` set and counts toward `JobStats.n_errored_trials`; it is not a failed workflow.

Whether a failed trial is tried again follows harbor's `RetryConfig` (pass `JobConfig.retry`):

- An exception in `exclude_exceptions`, or not in `include_exceptions`, is recorded immediately.
- Otherwise the trial is re-run up to `max_retries` times, backing off `min_wait_sec * wait_multiplier ** n`
  (capped at `max_wait_sec`), exactly as harbor does.
- Harbor's defaults allow no retries.

Some failures happen outside anything harbor records: a worker lost mid-trial, a heartbeat timeout,
or a task that cannot be loaded. Temporal retries those up to `infrastructure_retries` extra times
(3 by default). If they are exhausted, `execute_trial` raises `ActivityError`, whose cause is an
`ApplicationError` with `type` set to the exception's class name.

`retry_policy_from_harbor(retry_config)` returns the `RetryPolicy` this uses, if you schedule the
trial Activity yourself.

## Scaling a job

Every trial result a workflow receives is kept in its history. That is what makes a finished trial
survive a lost worker, and it also bounds how many trials one workflow should run. A slimmed result
is typically a few kilobytes. For jobs of more than a few hundred trials, run slices of `plan.trials`
in child workflows, and use continue-as-new as your workflow's history grows.

Aggregate once, over all of the job's results. Pass@k averages over tasks, so it cannot be combined
from shards aggregated separately. Have children return their results, and call `aggregate_job` in
the parent. A child's return value is a single payload, so keep slices small enough that their
results fit within Temporal's payload size limit.

`JobConfig.n_concurrent_trials` is not used. The worker's activity slots
(`max_concurrent_activities`) bound how many trials run at once, across every job on the task queue.

## Not yet supported

- Regrade jobs (`JobConfig.source_jobs`). `plan_job` rejects them.
- Uploading trial directories off the worker. They stay on the disk of the worker that ran the
  trial.

## Composing with other plugins

The plugin installs the pydantic data converter only when the Client has none configured. If you
use a custom data converter, it must handle pydantic models. The plugin adds no interceptors, so
tracing plugins such as OpenTelemetry compose with it in any order.

## Develop

```bash
make sync   # install (non-editable) into .venv
make lint
make test
```

The tests run real harbor trials with harbor's `oracle` agent, in an environment that executes
task scripts on the host. They need `bash`, but no container runtime and no credentials.
