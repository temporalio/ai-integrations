# Temporal TypeSafe integration

> Release stage: [Pre-release](https://docs.temporal.io/develop/python/integrations/typesafe).

Temporal integration for [TypeSafe](https://docs.typesafe.ai)
decision calls, published as [`temporalio-typesafe`](https://pypi.org/project/temporalio-typesafe/)
and imported as `temporalio.typesafe`.

One Activity is one `POST /v1/systemone`: a state plus any number of questions
(choice, score, noul), answered in a single request. Per-call thresholds and
confidence routing stay in workflow code.

## Install

```bash
uv add temporalio-typesafe
```

## Usage

Register `TypeSafePlugin` on the worker. The HTTP client and its credentials stay
there, out of workflow history:

```python
import os

import httpx2
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from temporalio.client import Client
from temporalio.typesafe import TypeSafePlugin

typesafe_client = AsyncTypeSafeClient(
    api_key=os.environ["TYPESAFE_API_KEY"],
    model="jev-1.13.0",
    timeout=httpx2.Timeout(30, connect=5),
    retry=RetryPolicy(max_retries=0),  # Temporal owns retries
)

client = await Client.connect(
    "localhost:7233",
    plugins=[TypeSafePlugin(typesafe_client)],
)
```

Workflow code asks questions through `TemporalTypeSafe`:

```python
import asyncio
from typing import Any

from temporalio import workflow
from temporalio.typesafe.workflow import TemporalTypeSafe
from typesafe_sdk import Noul, NoulAnswer


@workflow.defn
class Triage:
    """Rank items by how likely each needs attention today."""

    @workflow.run
    async def run(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        typesafe = TemporalTypeSafe()
        questions = {
            "needs_attention": Noul(
                instructions="Does this item need attention today?",
            )
        }
        results = await asyncio.gather(
            *(
                typesafe.system_one(state=item, questions=questions)
                for item in items
            )
        )
        ranked = []
        for item, result in zip(items, results):
            answer = result.response.answers.get("needs_attention")
            if not isinstance(answer, NoulAnswer):
                continue
            ranked.append({"id": item["id"], "urgency": answer.noul})
        return sorted(ranked, key=lambda d: -d["urgency"])
```

`system_one(state, questions)` matches the SDK's method name and sends several
questions about one state in a single request. Add more questions as keys in
the mapping.

For multiple states, the example uses `asyncio.gather()` to schedule one
Activity per item and collect results in input order. Each call is independent;
repeated states produce separate requests. Deduplicate inputs in workflow code
if that is the behavior you need.

The number of System One Activities running at once on a worker is capped by
Temporal's `Worker(max_concurrent_activities=...)` setting. It caps simultaneous
Activity execution across that worker's workflows; the queued batch itself
stays unbounded, so a slow run queues Activities instead of dropping them.

Each result is a `SystemOneResult`: `.response` is the SDK's own response, with
`.answers` keyed by question name, the served `.model`, and the request's token
`.usage`. A `NoulAnswer` carries a single `.noul`: the probability (0-1) that
the answer is yes, with 0.5 meaning undecided. Noul answers have no separate
confidence field; `.noul` is both the answer and the certainty. When the yes/no
boundary is subtle, add `criteria` with `true` and `false` descriptions of what
each outcome means.

Retries ride Temporal's RetryPolicy. Configure the SDK client with
`typesafe-sdk max_retries=0`, so no retry loop runs inside the SDK; the
server's `retry-after` hint becomes `next_retry_delay`, and the caller's policy
owns the timing. The plugin rejects clients that enable the SDK's own retries,
and points callers at
`TemporalTypeSafe(activity_config={"retry_policy": ...})` instead.

Payloads at Workflow/Activity and Workflow/client boundaries go through the Pydantic payload
converter: the plugin upgrades a default payload converter and leaves an
explicitly configured custom one alone. Caller-supplied payload codecs and
failure converter settings are preserved, so a registered response instance
and score maps with integer keys round-trip in both directions.

The plugin uses the supplied SDK client's HTTP settings as configured. Set the
client's HTTP timeout and the Activity timeout together: HTTP timeouts bound
individual network operations, while Temporal's Activity timeout bounds an
attempt. The plugin does not rewrite the client's HTTP options to fit an
Activity deadline. Temporal timing out an attempt does not itself stop an
in-flight provider request.

Tune these layers:

| Layer | What it bounds | Default | Knob |
| --- | --- | --- | --- |
| HTTP operation | One connect, read, write, or pooled-connection acquire | SDK client setting | `AsyncTypeSafeClient(timeout=...)` |
| Activity attempt | One whole Activity Task Execution, including all HTTP waits | 30s in `TemporalTypeSafe` | `start_to_close_timeout` in `activity_config`, on `TemporalTypeSafe(...)` or on a call |
| Total duration | Queueing, every attempt, retries, and backoff | unbounded | `schedule_to_close_timeout` in `activity_config`, on `TemporalTypeSafe(...)` or on a call |

```python
# Worker side: configure HTTP bounds on the SDK client
typesafe_client = AsyncTypeSafeClient(
    api_key=os.environ["TYPESAFE_API_KEY"],
    timeout=httpx2.Timeout(30, connect=5),
    retry=RetryPolicy(max_retries=0),
)
plugin = TypeSafePlugin(typesafe_client)

# Workflow side: widen the budget and bound the overall deadline even
# when it retries.
s1 = TemporalTypeSafe(
    activity_config={
        "start_to_close_timeout": timedelta(seconds=45),
        "schedule_to_close_timeout": timedelta(minutes=5),
    }
)
await s1.system_one(state=state, questions=questions)
```

`start_to_close_timeout` restarts per attempt, so it does not bound total
duration across retries. `schedule_to_close_timeout` covers queueing, attempts,
and backoff, including delays from server `retry-after` hints.

## Bring your own client

Configure the TypeSafe SDK client yourself, then pass it to the plugin:

```python
import os

from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

client = AsyncTypeSafeClient(
    api_key=os.environ["TYPESAFE_API_KEY"],
    base_url="http://localhost:8000",
    headers={"X-Request-Source": "triage"},
    retry=RetryPolicy(max_retries=0),
)
plugin = TypeSafePlugin(client)
```

The caller owns the client and must close it with `await client.aclose()` when
the Workers using it have stopped. SDK-level retries are rejected; configure
`RetryPolicy(max_retries=0)` and use Temporal's Activity retry policy instead.

## Typed responses

Questions are the SDK's native `Noul`, `Score`, and `Choice` models (or
plain dicts with the same wire shape). To get SDK-validated
answers, register the response class on the plugin and name it on the call.
The workflow receives answers on the `SystemOneResult` envelope and decodes them
as usual.

```python
from typesafe_sdk import NoulAnswer, SystemOneResponse


class BillingResponse(SystemOneResponse):
    billing: NoulAnswer


# Worker side:
TypeSafePlugin(
    typesafe_client,
    response_models={"billing": BillingResponse},
)

# Workflow side:
result = await TemporalTypeSafe().system_one(
    state,
    {"billing": Noul(instructions="Is this about billing?")},
    response_model="billing",
)

answer = result.response.answers["billing"]
assert isinstance(answer, NoulAnswer)
```

Naming is a `str` because the model class itself cannot cross the
Workflow/Activity boundary durably; the registry lives worker-side. Names must
map to `SystemOneResponse` subclasses; the plugin rejects other classes at
construction, and an unregistered name fails the Activity without retrying.

## Configuration

The TypeSafe SDK reads these environment variables when you construct the
client. `TypeSafePlugin` accepts the configured client and does not duplicate
its configuration options.

- `TYPESAFE_API_KEY`: key for requests to the API.
- `TYPESAFE_BASE_URL`: endpoint override.
- `TYPESAFE_DEFAULT_MODEL`: model used when the call doesn't name one. The
  call can name one per question set (`TemporalTypeSafe(model=...)` or a
  per-call `model=`), which the Activity input also records.
  Pin an exact version (`jev-1.13.0`, not `jev-latest`) once you have tuned
  thresholds: calibration can change between releases.
- `TYPESAFE_LOG_LEVEL`: SDK logging level.

## Develop

```bash
bazel run //python/_tools:dev -- venv --plugin-dir python/typesafe
bazel test //python/typesafe:lint
bazel test //python/typesafe:tests
```
