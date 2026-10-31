# Temporal Harbor integration

> Release stage: **Pre-release**.

Run [harbor](https://github.com/harbor-framework/harbor) evaluation jobs durably, with each trial as
its own Temporal Activity. Published as [`temporalio-harbor`](https://pypi.org/project/temporalio-harbor/)
and imported as `temporalio.harbor`.

Wrapping a whole `harbor run` in one Activity means a lost worker throws away every trial that
already finished, and the Temporal UI shows one opaque Activity for hours. With this plugin:

- A lost worker costs only the trials that were in flight. Every finished trial's result is
  already in workflow history.
- Each trial is visible in the UI, labeled with its task and agent. Its heartbeat carries the
  harbor phase it is in (`agent-start`, `verification-start`, and so on) and the time elapsed.
- The job's statistics come from harbor's own code: reward and error statistics, token and cost
  totals, each dataset's metrics, and pass@k. The job directory gets harbor's `config.json`,
  `lock.json` and `result.json`, so harbor's own tools can read it.

## Install

```bash
uv add temporalio-harbor
```

Requires Python 3.12 or later, like harbor itself. Docker and other local container runtimes need
nothing more. Cloud sandbox providers such as Daytona, Modal and E2B are harbor extras; install the
ones your jobs use on every Worker, for example `uv add 'harbor[daytona]'`.

## Usage

Register `HarborPlugin` on the Client. It registers the plugin's Activities on every Worker built
from that Client, and makes the Client's data converter use Temporal's pydantic payload converter,
which harbor's models need. A payload codec or failure converter you configure is kept.

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
        # Each running trial takes one activity slot.
        max_concurrent_activities=16,
    )
    await worker.run()
```

A workflow plans a harbor `JobConfig` into trials, runs them, and aggregates what came back:

```python
import asyncio

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from harbor.models.job.config import JobConfig
    from harbor.models.job.result import JobStats

    from temporalio.harbor import TrialOutcome, aggregate_job, execute_trial, plan_job


@workflow.defn
class EvalJob:
    @workflow.run
    async def run(self, config: JobConfig) -> JobStats:
        plan = await plan_job(config)
        outcomes = await asyncio.gather(
            *(execute_trial(trial) for trial in plan.trials), return_exceptions=True
        )
        result = await aggregate_job(
            plan, [o for o in outcomes if isinstance(o, TrialOutcome)]
        )
        return result.stats
```

`return_exceptions=True` lets the job finish when some trial could not run at all, and the job is
aggregated from the outcomes that came back; a trial with no outcome counts as pending in the
job's statistics, not as errored. `aggregate_job` returns harbor's `JobResult`, which carries every
trial's result, so the workflow returns only its `stats`: a whole `JobResult` for a large job can
exceed Temporal's 2 MB payload limit.

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

Give each run its own `job_name`: a second run under the same name and config writes its trials and
job files into the first run's job directory.

## What the plugin does

`plan_job(config)` resolves the job's datasets into tasks in an Activity, as harbor's `Job` does.
That Activity checks each agent's options against the options the agent declares and looks up each
dataset's metrics, so a misconfigured job fails here, once, without retrying. Options a job leaves
out, which an agent can read from the worker's environment variables, and sandbox credentials are
checked on the worker as each trial starts. The workflow then expands each task, for every agent
and each of the job's `n_attempts`, into a trial, in harbor's order, and gives each trial a name
that is stable across replay.

`execute_trial(trial)` runs one trial as one Activity and returns a `TrialOutcome`.
`outcome.result` is harbor's `TrialResult`, with rollout details (token IDs and log-probabilities)
and agent metadata left out and long exception text shortened, because harbor's aggregation never
reads them and they are most of a result's size. Harbor writes the complete result to the trial
directory on the worker that ran the attempt, and a later attempt of the trial that uses the same
directory starts by deleting it. `outcome.output` is what the worker's `TrialHooks.output`
returned, or `None` when the attempt could not carry it, for example because the hook raised. The
example below raises to have a trial run again; if it raises for every attempt, `execute_trial`
returns an outcome whose `output` is `None`.

Harbor's own phase timeouts still apply inside a trial: harbor stops an agent that runs out of time
and scores what it did. The plan records, for each trial, the sum of those timeouts (environment
build, agent setup, agent, verifier and the rest), and the Activity's `start_to_close_timeout`
defaults to 1.5 times that sum plus `execute_trial`'s `extra_time`. `extra_time` (15 minutes by
default) covers what harbor does not time, such as the worker's hooks, fetching the task and
tearing the environment down. An agent given no timeout by its task or its config counts as
`plan_job`'s `untimed_agent_timeout`, 2 hours by default. Harbor still runs that agent without a
limit, so one that runs far longer hits the Activity's timeout, and Temporal runs the trial again.

`aggregate_job(plan, outcomes)` computes harbor's `JobResult`. A dataset's metric can be a script
that harbor runs with `uv`, so workers need `uv` installed and access to the dataset registries.

## Errors and retries

A trial harbor ran is a result, even when the result records an exception, such as an agent that
ran out of time or exited with an error, or a verifier that wrote no reward. Harbor scores those
trials, so `execute_trial` returns them, and they count toward `JobStats.n_errored_trials`.

Temporal runs a trial again when:

- it could not run: its task could not be fetched, or harbor's environment preflight failed on the
  worker, for example because a sandbox provider's credentials are missing;
- harbor's result shows it failed before its agent started, for example because its environment
  did not start or the agent's setup failed;
- an agent run ended in one of harbor's transient provider errors. After a rate limit, the next
  attempt waits 2 minutes.

A trial harbor rejects as misconfigured when the worker creates it, for example because a `${VAR}`
in its config is unset there, fails without a retry. Pass `retry_policy=` to `execute_trial` to
replace the default policy. A worker can also send a recorded trial back to Temporal, for example
one whose verifier failed, by raising an `ApplicationError` from `TrialHooks.output` (see below).

When Temporal stops retrying a trial, `execute_trial` returns the most recent outcome any attempt
produced, or raises if no attempt produced one.

## Customizing trials

A worker takes part in running each trial through `TrialHooks`. Subclass it, override what you
need, and pass it to the plugin as `HarborPlugin(trial_hooks=...)`. For each attempt, the trial
Activity:

1. enters `scope(context)`, an async context manager around the whole attempt;
2. creates harbor's `Trial` and passes it to `trial_created(trial, context)`, before it runs;
3. runs the trial;
4. if harbor produced a result, passes its text through `redact(text, context)` and returns
   `output(result, context)` to the workflow as `outcome.output`.

`context` is a `TrialContext`: the trial's config, the attempt number, its directory, and the
`data` the workflow passed to `execute_trial`.

```python
import asyncio
import contextlib
import os
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path

from harbor.models.trial.result import TrialResult
from pydantic import JsonValue

from temporalio import activity
from temporalio.exceptions import ApplicationError
from temporalio.harbor import TrialContext, TrialHooks


class EvalHooks(TrialHooks):
    def __init__(self) -> None:
        # Read once, so a worker without the key fails at startup, not in every trial.
        self._api_key = os.environ["PROVIDER_API_KEY"]

    @contextlib.asynccontextmanager
    async def scope(self, context: TrialContext) -> AsyncIterator[None]:
        try:
            yield
        finally:
            # The next attempt of this trial starts by deleting its directory.
            prefix = f"{context.config.trial_name}/{context.attempt}"
            try:
                await asyncio.to_thread(upload, context.trial_dir, prefix)
            except Exception:
                # Raising here would have Temporal run the whole trial again.
                activity.logger.exception("Could not upload %s", prefix)

    async def output(self, result: TrialResult, context: TrialContext) -> JsonValue:
        exc = result.exception_info
        if exc is not None and exc.exception_type == "VerifierTimeoutError":
            raise ApplicationError("verifier timed out", next_retry_delay=timedelta(minutes=1))
        return {"run": context.data}

    def redact(self, text: str, context: TrialContext) -> str:
        # An empty key would match between every character.
        return text.replace(self._api_key, "[redacted]") if self._api_key else text


def upload(trial_dir: Path, prefix: str) -> None:
    for root, _, files in os.walk(trial_dir):
        for name in files:
            path = Path(root) / name
            # An agent can leave symlinks to anywhere on the worker; skip them.
            if not path.is_symlink():
                bucket.upload_file(str(path), f"{prefix}/{path.relative_to(trial_dir)}")
```

Hooks run on the worker's event loop, beside every other trial that worker is running, so a
blocking call stalls all of them, heartbeats included. Use async libraries or `asyncio.to_thread`
for I/O, as `scope` does above. Together, the async hooks of one attempt may spend the `extra_time`
the workflow gives `execute_trial`; a hook that runs past it fails the attempt, and Temporal runs
the trial again.

`output` sees the complete result, unredacted. What it returns is stored in workflow history
without passing through `redact`, so keep it small and free of secrets. `redact` rewrites the text
of the result and of every error an attempt raises before it reaches workflow history, but some of
that text reaches history as harbor recorded it; `TrialHooks.redact` says which.

## Differences from `harbor run`

- `JobConfig.retry` and each agent's `n_concurrent` and `concurrency_group` are ignored, and
  `plan_job` logs a warning when they are set. `n_concurrent_trials` is unused.
- Concurrency is bounded per worker process, by `max_concurrent_activities`. Harbor runs 4 trials
  at a time by default, so set it to what one machine and your providers can take.
- Any worker can run any trial, so local task paths and credentials must exist on every worker.
- Literal secret values in `env` are masked when a config crosses Temporal, so give them as
  `${VAR}`, with the variable set on every worker.
- Each trial writes its directory on the worker that ran it. `jobs_dir` must be storage that every
  worker shares for harbor's tools (`harbor view`, `harbor upload`, `harbor job resume`) to see the
  whole job.
- The metrics of registry and repo datasets are looked up again at aggregation. A package dataset
  is pinned to the content hash planning resolved.
- Agent skills given as a git ref resolve again for each trial, on the worker that runs it. Use an
  immutable tag, or a local path present on every worker.
- Regrade jobs (`JobConfig.source_jobs`) are not supported. `plan_job` rejects them.

One workflow keeps every trial's outcome in its history and every unfinished trial pending.
Temporal's limits on pending Activities and on message size put the ceiling for one workflow at
about 2,000 trials; past it, the workflow cannot schedule its trials and makes no progress.

## Composing with other plugins

A custom payload converter is left in place, and must handle pydantic models. The plugin adds no
interceptors, so tracing plugins such as OpenTelemetry compose with it in any order.

## Develop

```bash
make sync   # install (non-editable) into .venv
make lint
make test
```

The tests run real harbor trials with harbor's `oracle` agent, in an environment that executes
task scripts on the host. They need `bash`, but no container runtime and no credentials.
